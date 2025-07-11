"""Download MEModel and run MEModel validation"""

import itertools
import pathlib
import subprocess
from concurrent.futures import ThreadPoolExecutor

from entitysdk import Client
from entitysdk.downloaders.emodel import download_hoc
from entitysdk.downloaders.ion_channel_model import download_ion_channel_mechanism
from entitysdk.downloaders.memodel import download_memodel
from entitysdk.downloaders.morphology import download_morphology
from entitysdk.models.emodel import EModel
from entitysdk.models.memodel import MEModel
from entitysdk.models.memodelcalibrationresult import MEModelCalibrationResult
from entitysdk.models.validation_result import ValidationResult

from app.logger import L


def compile_mechanisms(mechanisms_dir):
    """Compile mechanisms in the given directory.

    Args:
        mechanisms_dir (str or Pathlib.Path): path to the directory with mechanisms
    """
    subprocess.run(
        [
            "nrnivmodl",
            "-incflags",
            "-DDISABLE_REPORTINGLIB",
            str(mechanisms_dir),
        ],
        check=True,
    )


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


def get_memodel_and_create_cell(client: Client, memodel_id: str):
    """Get MEModel, compile the mechanisms and create a bluecellulab cell.

    Args:
        client (Client): EntitySDK client
        memodel_id (str): id of the MEModel to download
    Returns:
        memodel (MEModel): MEModel entitysdk object
        cell (Cell): bluecellulab Cell object
    """
    L.info("Downloading MEModel")
    memodel = client.get_entity(
        entity_type=MEModel,
        entity_id=memodel_id,
    )
    downloaded_memodel = download_memodel(client, memodel, output_dir=".")
    holding_current, threshold_current = get_holding_and_threshold(memodel.calibration_result)

    L.info(f"Model holding current: {holding_current}")
    L.info(f"Model threshold current: {threshold_current}")

    L.info("Compiling mechanisms")
    compile_mechanisms(downloaded_memodel.mechanisms_dir)

    cell = create_bluecellulab_cell(
        downloaded_memodel.hoc_path,
        downloaded_memodel.morphology_path,
        holding_current,
        threshold_current,
    )

    return memodel, cell


def register_calibration(client, memodel, calibration_dict):
    """Register the calibration result

    Args:
        client (Client): EntitySDK client
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
    )


def register_validations(client: Client, memodel, validation_dict, val_details_out_dir=None):
    """Register the validation results, with figures and validation details as assets

    Args:
        client (Client): EntitySDK client
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
        )
        val = iterator.first()
        if val is not None:
            continue

        # register validation result
        validation_result = ValidationResult(
            name=val_dict["name"],
            passed=val_dict["passed"],
            validated_entity_id=memodel.id,
        )
        registered = client.register_entity(
            entity=validation_result,
        )

        # register figure(s) as asset(s)
        for fig_path in val_dict["figures"]:
            if fig_path.suffix != ".pdf":
                msg = "Only pdf files are supported for validation result figures."
                raise ValueError(msg)

            client.upload_file(
                entity_id=registered.id,
                entity_type=ValidationResult,
                file_path=str(fig_path),
                file_content_type="application/pdf",
                asset_label="validation_result_figure",
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
                file_content_type="text/plain",
                asset_label="validation_result_details",
            )


def run_and_save_calibration(client: Client, memodel_id: str):
    """Download MEModel, run MEModel calibration and save results.

    Args:
        client (Client): EntitySDK client
        memodel_id (str): id of the MEModel to download
    """
    memodel, cell = get_memodel_and_create_cell(client, memodel_id)

    if cell.threshold == 0.0:
        L.info("No threshold current found, will compute it.")
        # importing bluecellulab AFTER compiling the mechanisms to avoid segmentation fault
        from bluecellulab.tools import compute_memodel_properties

        memodel_properties = compute_memodel_properties(cell)

        L.info("Saving calibration and validation results")
        register_calibration(client, memodel, memodel_properties)


def run_and_save_validation(client: Client, memodel_id: str):
    """Download MEModel, run MEModel validation, and save validation results.

    Args:
        client (Client): EntitySDK client
        memodel_id (str): id of the MEModel to download
    """
    memodel, cell = get_memodel_and_create_cell(client, memodel_id)

    # importing bluecellulab AFTER compiling the mechanisms to avoid segmentation fault
    from bluecellulab.validation.validation import run_validations

    L.info("Running validations")
    validation_dict = run_validations(cell, memodel.name, output_dir="./figures")

    L.info("Saving validation results")
    register_validations(client, memodel, validation_dict)

    L.success("Done")
