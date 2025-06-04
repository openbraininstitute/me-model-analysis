"""Download MEModel and run MEModel validation"""

import itertools
import os
import pathlib
import subprocess

from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

from entitysdk.client import Client
from entitysdk.common import ProjectContext
from entitysdk.models.emodel import EModel
from entitysdk.models.ion_channel_model import IonChannelModel
from entitysdk.models.memodel import MEModel
from entitysdk.models.morphology import ReconstructionMorphology
from obi_auth import get_token


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


def download_morphology(morphology, client, access_token, morph_dir="./morphology", file_type="asc"):
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
    for asset in morphology.assets:
        if file_type is None or file_type in asset.content_type:
            asset_id = asset.id
            asset_path = asset.path
            break
    if asset_id is None:
        ftype = file_type if file_type else ""
        raise ValueError(f"No {ftype} file found in the morphology {morphology.name}.")
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


def download_memodel(client, access_token, memodel_id=None, memodel_name=None):
    """Download MEModel
    
    Args:
        client (Client): EntitySDK client
        access_token (str): access token for authentication
        memodel_id (str or None): id of the MEModel to download
        memodel_name (str or None): name of the MEModel to download
    """
    if memodel_id is None and memodel_name is None:
        raise ValueError("Either memodel_id or memodel_name must be provided.")
    if memodel_id is not None:
        memodel = client.get_entity(
            entity_type=MEModel,
            entity_id=memodel_id,
            token=access_token,
        )
    else:
        iterator = client.search_entity(
            entity_type=MEModel,
            query={"name": memodel_name},
            token=access_token,
            limit=1,
        )
        memodel = next(iterator)

    morphology = memodel.morphology
    # we have to get the emodel to get the ion channel models.
    emodel = client.get_entity(
        entity_id=memodel.emodel.id, entity_type=EModel, token=access_token
    )

    holding_current, threshold_current = get_holding_and_threshold(memodel.calibration_result)

    # + 2 for hoc and morphology
    # len of ion_channel_models should be around 10 for most cases,
    # and always < 100, even for genetic models
    max_workers = len(emodel.ion_channel_models) + 2
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        hoc_future = executor.submit(download_hoc, emodel, client, access_token, "./hoc")
        morph_future = executor.submit(download_morphology, morphology, client, access_token, "./morphology")
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
    hold_curr = hold_curr if hold_curr else 0.
    # 0. is default value, will be overriden in validation
    thres_curr = thres_curr if thres_curr else 0.

    # importing bluecellulab AFTER compiling the mechanisms to avoid segmentation fault
    from bluecellulab import Cell
    from bluecellulab.circuit.circuit_access import EmodelProperties

    emodel_properties = EmodelProperties(
        threshold_current=thres_curr,
        holding_current=hold_curr,
        AIS_scaler=1.
    )
    return Cell(hoc_path, morphology_path, template_format="v6", emodel_properties=emodel_properties)


def download_and_run_validation(client, access_token, memodel_id=None, memodel_name=None):
    """Download MEModel and run MEModel validation

    Args:
        client (Client): EntitySDK client
        access_token (str): access token for authentication
        memodel_id (str or None): id of the MEModel to download
        memodel_name (str or None): name of the MEModel to download
    """
    memodel, hoc_path, mechanisms_dir, morphology_path, hold_curr, thres_curr = download_memodel(
        client, access_token, memodel_id=memodel_id, memodel_name=memodel_name
    )
    # compile the mechanisms
    subprocess.run(["nrnivmodl", str(mechanisms_dir)], check=True)

    cell = create_bluecellulab_cell(hoc_path, morphology_path, hold_curr, thres_curr)
    # importing bluecellulab AFTER compiling the mechanisms to avoid segmentation fault
    from bluecellulab.validation.validation import run_validations

    validation_dict = run_validations(cell, memodel.name, output_dir="./figures")

    return validation_dict
