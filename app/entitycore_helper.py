"""Download MEModel and run MEModel validation"""

import itertools
import pathlib
import subprocess
from concurrent.futures import ThreadPoolExecutor

from entitysdk.models.emodel import EModel
from entitysdk.models.ion_channel_model import IonChannelModel
from entitysdk.models.memodel import MEModel
from entitysdk.models.memodelcalibrationresult import MEModelCalibrationResult
from entitysdk.models.morphology import ReconstructionMorphology
from entitysdk.models.validation_result import ValidationResult

from app.logger import L


def download_hoc(emodel, client, access_token, hoc_dir="./hoc"):
    """Download hoc file

    Args:
        emodel (EModel): EModel entitysdk object
        client (Client): EntitySDK client
        access_token (str): access token for authentication
        hoc_dir (str or Pathlib.Path): directory to save the hoc file
    """
    asset_id = None
    asset_path = None
    # Download the emodel hoc file
    if emodel.assets is None:
        raise ValueError(f"No assets found in the emodel {emodel.name}.")
    for asset in emodel.assets:
        if "hoc" in asset.content_type:
            asset_id = asset.id
            asset_path = asset.path
    if asset_id is None:
        raise ValueError(f"No hoc file found in the emodel {emodel.name}.")
    hoc_dir = pathlib.Path(hoc_dir)
    hoc_dir.mkdir(parents=True, exist_ok=True)
    hoc_output_path = hoc_dir / asset_path
    client.download_file(
        asset_id=asset_id,
        entity_id=emodel.id,
        entity_type=EModel,
        token=access_token,
        output_path=hoc_output_path,
    )
    return hoc_output_path


def download_one_mechanism(ic, client, access_token, mechanisms_dir="./mechanisms"):
    """Download one mechanism file

    Args:
        ic (IonChannelModel): IonChannelModel entitysdk object
        client (Client): EntitySDK client
        access_token (str): access token for authentication
        mechanisms_dir (str or Pathlib.Path): directory to save the mechanism file
    """
    if not ic.assets:
        raise ValueError(f"No assets found in the ion channel model {ic.name}.")
    asset = ic.assets[0]
    asset_id = asset.id
    asset_path = asset.path
    client.download_file(
        asset_id=asset_id,
        entity_id=ic.id,
        entity_type=IonChannelModel,
        token=access_token,
        output_path=mechanisms_dir / asset_path,
    )


def download_morphology(
    morphology, client, access_token, morph_dir="./morphology", file_type="asc"
):
    """Download morphology file
    Args:
        morphology (ReconstructionMorphology): Morphology entitysdk object
        client (Client): EntitySDK client
        access_token (str): access token for authentication
        morph_dir (str or Pathlib.Path): directory to save the morphology file
        file_type (str or None): type of the morphology file (asc, swc or h5).
            Will take the first one if None.
    """
    if morphology.assets is None:
        raise ValueError(f"No assets found in the morphology {morphology.name}.")
    morph_dir = pathlib.Path(morph_dir)
    morph_dir.mkdir(parents=True, exist_ok=True)

    asset_id = None
    asset_path = None
    if not morphology.assets:
        raise ValueError(f"No file found in the morphology {morphology.name}.")
    # try to fetch morphology with the specified file type
    for asset in morphology.assets:
        if file_type is None or file_type in asset.content_type:
            asset_id = asset.id
            asset_path = asset.path
            break
    # fallback #1: we expect at least a asc or swc file
    if asset_id is None:
        for asset in morphology.assets:
            if 'asc' in asset.content_type or 'swc' in asset.content_type:
                L.warning(
                    "No %s file found in the morphology %s, will select the one with %s.",
                    file_type,
                    morphology.name,
                    asset.content_type,
                )
                asset_id = asset.id
                asset_path = asset.path
                break
    # fallback #2: we take the first asset
    if asset_id is None:
        L.warning(
            "No %s file found in the morphology %s, will select the first one.",
            file_type,
            morphology.name,
        )
        asset_id = morphology.assets[0].id
        asset_path = morphology.assets[0].path

    morph_out_path = morph_dir / asset_path
    client.download_file(
        asset_id=asset_id,
        entity_id=morphology.id,
        entity_type=ReconstructionMorphology,
        token=access_token,
        output_path=morph_out_path,
    )
    return morph_out_path


def get_holding_and_threshold(calibration_result):
    """Get holding and threshold currents from the MEModel.

    Args:
        calibration_result (MEModelCalibrationResult or None): Calibration result entitysdk object
    """
    holding_current = None
    threshold_current = None
    if calibration_result is not None:
        holding_current = calibration_result.holding_current
        threshold_current = calibration_result.threshold_current

    return holding_current, threshold_current


def download_memodel(client, access_token, memodel_id):
    """Download MEModel

    Args:
        client (Client): EntitySDK client
        access_token (str): access token for authentication
        memodel_id (str): id of the MEModel to download
    """

    memodel = client.get_entity(
        entity_type=MEModel,
        entity_id=memodel_id,
        token=access_token,
    )

    morphology = memodel.morphology
    # we have to get the emodel to get the ion channel models.
    emodel = client.get_entity(entity_id=memodel.emodel.id, entity_type=EModel, token=access_token)

    holding_current, threshold_current = get_holding_and_threshold(memodel.calibration_result)

    # + 2 for hoc and morphology
    # len of ion_channel_models should be around 10 for most cases,
    # and always < 100, even for genetic models
    max_workers = len(emodel.ion_channel_models) + 2
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        hoc_future = executor.submit(download_hoc, emodel, client, access_token, "./hoc")
        morph_future = executor.submit(
            download_morphology, morphology, client, access_token, "./morphology"
        )
        mechanisms_dir = pathlib.Path("./mechanisms")
        mechanisms_dir.mkdir(parents=True, exist_ok=True)
        executor.map(
            download_one_mechanism,
            emodel.ion_channel_models,
            itertools.repeat(client),
            itertools.repeat(access_token),
            itertools.repeat(mechanisms_dir),
        )

        hoc_path = hoc_future.result()
        morphology_path = morph_future.result()

    return memodel, hoc_path, mechanisms_dir, morphology_path, holding_current, threshold_current


def create_bluecellulab_cell(hoc_path, morphology_path, hold_curr, thres_curr):
    """Create a bluecellulab cell from the hoc and morphology files.

    Args:
        hoc_path (str or Pathlib.Path): path to the hoc file
        morphology_path (str or Pathlib.Path): path to the morphology file
        hold_curr (float or None): holding current
        thres_curr (float or None): threshold current
    """
    hold_curr = hold_curr or 0.0
    # 0. is default value, will be overriden in validation
    thres_curr = thres_curr or 0.0

    # importing bluecellulab AFTER compiling the mechanisms to avoid segmentation fault
    from bluecellulab import Cell
    from bluecellulab.circuit.circuit_access import EmodelProperties

    emodel_properties = EmodelProperties(
        threshold_current=thres_curr, holding_current=hold_curr, AIS_scaler=1.0
    )
    return Cell(
        hoc_path, morphology_path, template_format="v6", emodel_properties=emodel_properties
    )


def register_calibration(client, access_token, memodel, calibration_dict):
    """Register the calibration result

    Args:
        client (Client): EntitySDK client
        access_token (str): access token for authentication
        memodel (MEModel): MEModel entitysdk object
        calibration_dict (dict): should contain the fields 'holding_current', 'rheobase' and 'rin'
    """
    # do not register MEModelCalibrationResult if it already exists
    # Once we are able to delete the CalibrationResult, we should move to the following logic:
    # if no MEModelCalibrationResult exists, register a new one
    # if one exists with exactly the same values, do nothing
    # if one exists with different values, delete the old one and register a new one
    iterator = client.search_entity(
        entity_type=MEModelCalibrationResult,
        query={"calibrated_entity_id": memodel.id},
        token=access_token,
    )
    cal = iterator.first()
    if cal is not None:
        return

    # register validation result
    calibration_result = MEModelCalibrationResult(
        holding_current=calibration_dict["holding_current"],
        threshold_current=calibration_dict["rheobase"],
        rin=calibration_dict["rin"],
        calibrated_entity_id=memodel.id,
    )
    client.register_entity(
        entity=calibration_result,
        token=access_token,
    )


def register_validations(client, access_token, memodel, validation_dict, val_details_out_dir=None):
    """Register the validation results, with figures and validation details as assets

    Args:
        client (Client): EntitySDK client
        access_token (str): access token for authentication
        memodel (MEModel): MEModel entitysdk object
        validation_dict (dict): dict containing the validation results
        val_details_out_dir (str or Pathlib.Path or None): directory to save
            the validation details files.
    """
    if val_details_out_dir is None:
        val_details_out_dir = pathlib.Path("./validation_details") / memodel.name
    else:
        val_details_out_dir = pathlib.Path(val_details_out_dir)
    val_details_out_dir.mkdir(parents=True, exist_ok=True)
    for key, val_dict in validation_dict.items():
        # not a validation
        if key == "memodel_properties":
            continue
        # do not register ValidationResult if it already exists
        # Once we are able to delete the ValidationResult, we should move to the following logic:
        # delete the ValidationResult if it already exists
        # register the new one
        iterator = client.search_entity(
            entity_type=ValidationResult,
            query={"name": val_dict["name"], "validated_entity_id": memodel.id},
            token=access_token,
        )
        cal = iterator.first()
        if cal is not None:
            continue

        # register validation result
        validation_result = ValidationResult(
            name=val_dict["name"],
            passed=val_dict["passed"],
            validated_entity_id=memodel.id,
        )
        registered = client.register_entity(
            entity=validation_result,
            token=access_token,
        )

        # register figure(s) as asset(s)
        for fig_path in val_dict["figures"]:
            client.upload_file(
                entity_id=registered.id,
                entity_type=ValidationResult,
                file_path=str(fig_path),
                file_content_type=f"application/{str(fig_path).split('.')[-1]}",
                token=access_token,
            )

        if val_dict["validation_details"]:
            # write down validation details to a file
            val_details_fname = f"{val_dict['name'].replace(' ', '')}_validation_details.txt"
            val_details_path = val_details_out_dir / val_details_fname
            with open(val_details_path, "w") as f:
                f.write(val_dict["validation_details"])
            # register validation details as asset
            client.upload_file(
                entity_id=registered.id,
                entity_type=ValidationResult,
                file_path=str(val_details_path),
                file_content_type="application/txt",
                token=access_token,
            )


def run_and_save_calibration_validation(client, access_token, memodel_id):
    """Download MEModel, run MEModel validation, and save validation and calibration results.

    Args:
        client (Client): EntitySDK client
        access_token (str): access token for authentication
        memodel_id (str): id of the MEModel to download
    """
    memodel, hoc_path, mechanisms_dir, morphology_path, hold_curr, thres_curr = download_memodel(
        client, access_token, memodel_id
    )
    # compile the mechanisms
    subprocess.run(["nrnivmodl", str(mechanisms_dir)], check=True)

    cell = create_bluecellulab_cell(hoc_path, morphology_path, hold_curr, thres_curr)
    # importing bluecellulab AFTER compiling the mechanisms to avoid segmentation fault
    from bluecellulab.validation.validation import run_validations

    validation_dict = run_validations(cell, memodel.name, output_dir="./figures")

    register_calibration(client, access_token, memodel, validation_dict["memodel_properties"])
    register_validations(client, access_token, memodel, validation_dict)
