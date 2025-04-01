"""Get EModel, modify morphology, plot analysis and upload MEModel."""

# Initial code taken from
# https://github.com/BlueBrain/BluePyEModel/blob/main/examples/memodel/memodel.py
# with some modifications for calling run_me_model_analysis

# pylint: disable=too-many-locals,import-error,import-outside-toplevel,too-many-statements,too-many-arguments

# Attention! This will overwrite figures made with the same access point.
# It is highly recommended to clean the figures folder between two runs
# to avoid any leftover figure to be wrongly associated with a MEModel.

import copy
import pathlib
from urllib.parse import unquote

from bluepyemodel.access_point.forge_access_point import get_brain_region_notation
from bluepyemodel.access_point.nexus import NexusAccessPoint
from bluepyemodel.emodel_pipeline.memodel import MEModel
from bluepyemodel.emodel_pipeline.plotting import plot_models, scores
from bluepyemodel.evaluation.evaluation import (
    compute_responses,
    get_evaluator_from_access_point,
)
from bluepyemodel.tools.search_pdfs import copy_emodel_pdf_dependencies_to_new_path
from bluepyemodel.validation.validation import compute_scores
from kgforge.core import KnowledgeGraphForge, Resource
from kgforge.specializations.resources import Dataset

from app.logger import L
from app.config import settings

mime_type_dict = {"png": "image/png", "pdf": "application/pdf"}


def connect_forge(
    bucket: str, endpoint: str, access_token: str, forge_path: str | None = None
) -> KnowledgeGraphForge:
    """Creation of a forge session."""
    if not forge_path:
        msg = "Missing path to forge config file"
        raise ValueError(msg)
    return KnowledgeGraphForge(
        forge_path, bucket=bucket, endpoint=endpoint, token=access_token, debug=True
    )


def get_ids_from_memodel(memodel: Resource) -> tuple[str, str]:
    """Get EModel and Morphology ids from MEModel resource metadata."""
    emodel_id = None
    morph_id = None
    if not hasattr(memodel, "hasPart"):
        msg = "ME-Model resource has no 'hasPart' metadata"
        raise AttributeError(msg)

    for haspart in memodel.hasPart:
        if haspart.type == "EModel":
            emodel_id = haspart.id
        elif haspart.type == "NeuronMorphology":
            morph_id = haspart.id

    if emodel_id is None:
        msg = "Could not find any EModel resource id link in MEModel resource."
        raise TypeError(msg)

    if morph_id is None:
        msg = "Could not find any NeuronMorphology resource id link in MEModel resource."
        raise TypeError(msg)

    return emodel_id, morph_id


def get_morph_mtype(annotation: dict) -> str:
    """Get morph mtype from annotation."""
    morph_mtype = None
    if hasattr(annotation, "hasBody"):
        if hasattr(annotation.hasBody, "label"):
            morph_mtype = annotation.hasBody.label
        else:
            msg = "Morphology resource has no label in annotation.hasBody."
            raise ValueError(msg)
    else:
        msg = "Morphology resource has no hasBodz in annotation."
        raise ValueError(msg)

    return morph_mtype


def get_morph_metadata(access_point, morph_id):
    """Get morph metadata."""
    resource = access_point.access_point.retrieve(morph_id)
    if resource is None:
        msg = f"Could not find the morphology resource with id {morph_id}"
        raise TypeError(msg)

    morph_brain_region = None
    if hasattr(resource, "brainLocation"):
        if hasattr(resource.brainLocation, "brainRegion"):
            if hasattr(resource.brainLocation.brainRegion, "label"):
                morph_brain_region = resource.brainLocation.brainRegion.label
            else:
                msg = "Morphology resource has no label in brainLocation.brainRegion"
                raise AttributeError(msg)
        else:
            msg = "Morphology resource has no brainRegion in brainLocation."
            raise AttributeError(msg)
    else:
        msg = "Morphology resource has no brainLocation."
        raise AttributeError(msg)

    morph_mtype = None
    if not hasattr(resource, "annotation"):
        msg = "Morphology resource has no annotation."
        raise AttributeError(msg)

    if isinstance(resource.annotation, dict):
        if hasattr(resource.annotation, "type") and (
            "MTypeAnnotation" in resource.annotation.type
            or "nsg:MTypeAnnotation" in resource.annotation.type
        ):
            morph_mtype = get_morph_mtype(resource.annotation)
    elif isinstance(resource.annotation, list):
        for annotation in resource.annotation:
            if hasattr(annotation, "type") and (
                "MTypeAnnotation" in annotation.type or "nsg:MTypeAnnotation" in annotation.type
            ):
                morph_mtype = get_morph_mtype(annotation)

    if morph_mtype is None:
        msg = "Could not find mtype in morphology resource"
        raise TypeError(msg)

    return morph_mtype, morph_brain_region


# pylint: disable-next=too-many-arguments
def get_new_emodel_metadata(
    access_point,
    morph_id,
    morph_name,
    update_emodel_name,
    use_brain_region_from_morphology,
    use_mtype_in_githash,
):
    """Get new emodel metadata."""
    new_emodel_metadata = copy.deepcopy(access_point.emodel_metadata)
    new_mtype, new_br = get_morph_metadata(access_point, morph_id)
    new_emodel_metadata.mtype = new_mtype

    if update_emodel_name:
        new_emodel_metadata.emodel = f"{new_emodel_metadata.etype}_{new_mtype}"

    if use_brain_region_from_morphology:
        new_emodel_metadata.brain_region = new_br
        new_emodel_metadata.allen_notation = get_brain_region_notation(
            new_br,
            access_point.access_point.access_token,
            access_point.forge_ontology_path,
            access_point.access_point.endpoint,
        )

    if use_mtype_in_githash:
        new_emodel_metadata.iteration = f"{new_emodel_metadata.iteration}-{morph_name}"

    return new_emodel_metadata


def get_cell_evaluator(access_point, morph_name, morph_format, morph_id):
    """Get cell evaluator."""
    # create cell evaluator from access point
    cell_evaluator = get_evaluator_from_access_point(
        access_point,
        include_validation_protocols=True,
        record_ions_and_currents=access_point.pipeline_settings.plot_currentscape,
    )

    # get morphology path
    morph_path = access_point.download_morphology(
        name=morph_name,  # optional in BPEMnexus 0.0.9.dev3 onwards if id_ is given
        format_=morph_format,
        id_=morph_id,
    )

    # modify the evaluator to use the 'new' morphology
    cell_evaluator.cell_model.morphology.morphology_path = morph_path

    return cell_evaluator


def plot_scores(access_point, cell_evaluator, mapper, figures_dir, seed):
    """Plot scores figures and return total fitness (sum of scores)."""
    emodel_score = None
    emodels = compute_responses(
        access_point,
        cell_evaluator=cell_evaluator,
        seeds=[seed],
        map_function=mapper,
        preselect_for_validation=False,  # model is probably already validated. ignore preselection.
    )
    if not emodels:
        raise ValueError(f"In plot_scores, no emodels for {access_point.emodel_metadata.emodel}")

    # we iterate but we expect only one emodel to be in the list
    for model in emodels:
        compute_scores(model, access_point.pipeline_settings.validation_protocols)

        figures_dir_scores = figures_dir / "scores" / "all"
        scores(model, figures_dir_scores)  # plotting fct
        # the scores have been added to the emodel at the compute_scores step
        emodel_score = sum(list(model.scores.values()))

    return emodel_score


def get_default_threshold_search_protocol():
    """Create a default protocol to use to search for the threshold with no holding current."""
    from bluepyemodel.evaluation.evaluator import (
        define_efeatures,
        define_preprotocols,
        soma_loc,
    )
    from bluepyemodel.evaluation.fitness_calculator_configuration import (
        FitnessCalculatorConfiguration,
    )
    from bluepyemodel.evaluation.protocols import ProtocolRunner

    rmp_prot_name = "RMPProtocol_noholding"
    rin_prot_name = "RinProtocol_noholding"
    efeatures_dict = [
        {
            "efel_feature_name": "steady_state_voltage_stimend",
            "protocol_name": rmp_prot_name,
            "recording_name": "soma.v",
            "mean": 0,
        },
        {
            "efel_feature_name": "ohmic_input_resistance_vb_ssse",
            "protocol_name": rin_prot_name,
            "recording_name": "soma.v",
            "mean": 0,
        },
    ]
    fcc = FitnessCalculatorConfiguration(
        efeatures=efeatures_dict,
        name_rmp_protocol=rmp_prot_name,
        name_rin_protocol=rin_prot_name,
    )
    efeatures = define_efeatures(
        fcc,
        include_validation_protocols=False,
        protocols={},
        efel_settings={},
    )

    protocols = define_preprotocols(
        efeatures=efeatures,
        location=soma_loc,
        fitness_calculator_configuration=fcc,
        rmp_key="bpo_rmp_noholding",
        hold_key="bpo_holding_current_noholding",
        rin_key="bpo_rin_noholding",
        thres_key="bpo_threshold_current_noholding",
        rmp_prot_name=rmp_prot_name,
        hold_prot_name="SearchHoldingCurrent_noholding",
        rin_prot_name=rin_prot_name,
        thres_prot_name="SearchThresholdCurrent_noholding",
        recording_name="soma.v",
        no_holding=True,
    )

    return ProtocolRunner(protocols)


def get_threshold(cell_evaluator, access_point, mapper):
    """Compute and return threshold current."""
    evaluator = copy.deepcopy(cell_evaluator)

    protocol = get_default_threshold_search_protocol()
    evaluator.fitness_protocols = {"main_protocol": protocol}
    emodels = compute_responses(
        access_point,
        evaluator,
        mapper,
    )

    return emodels[0].responses.get("bpo_threshold_current_noholding", None)


def plot(access_point, seed, cell_evaluator, figures_dir, mapper):
    """Plot figures and return total fitness (sum of scores), holding and threshold currents."""
    # compute scores
    # we need to do this outside of main plotting function with custom function
    # so that we do not take old emodel scores in scores figure
    emodel_score = plot_scores(access_point, cell_evaluator, mapper, figures_dir, seed)

    emodels = plot_models(
        access_point=access_point,
        mapper=mapper,
        seeds=[seed],
        figures_dir=figures_dir,
        plot_distributions=True,
        plot_scores=False,  # scores figure done outside of this
        plot_traces=True,
        plot_thumbnail=True,
        plot_currentscape=access_point.pipeline_settings.plot_currentscape,
        plot_bAP_EPSP=access_point.pipeline_settings.plot_bAP_EPSP,
        plot_dendritic_ISI_CV=True,  # for detailed cADpyr cells. will be skipped otherwise
        plot_dendritic_rheobase=True,  # for detailed cADpyr cells. will be skipped otherwise
        only_validated=False,
        save_recordings=False,
        load_from_local=False,
        cell_evaluator=cell_evaluator,  # <-- feed the modified evaluator here
    )
    emodel_holding = emodels[0].responses.get("bpo_holding_current", None)
    emodel_threshold = emodels[0].responses.get("bpo_threshold_current", None)

    return emodel_score, emodel_holding, emodel_threshold


def get_nexus_images(access_point, seed, new_emodel_metadata, morph_id, emodel_id):
    """Get the nexus images from memodel method using new emodel metadata."""
    # create MEModel (easier to get images with it)
    memodel = MEModel(
        seed=seed,
        emodel_metadata=access_point.emodel_metadata,
        emodel_id=emodel_id,
        morphology_id=morph_id,
        validated=False,
    )

    # update MEModel metadata
    memodel.emodel_metadata = new_emodel_metadata

    return memodel.build_pdf_dependencies(memodel.seed)


def update_memodel(
    forge,
    memodel_r,
    seed,
    new_emodel_metadata,
    subject_ontology,
    brain_location_ontology,
    nexus_images,
    emodel_score=None,
    emodel_holding=None,
    emodel_threshold=None,
    new_status="done",
):
    """Update ME-Model."""
    # update metadata in resource
    metadata_for_resource = new_emodel_metadata.for_resource()
    metadata_for_resource["subject"] = subject_ontology
    metadata_for_resource["brainLocation"] = brain_location_ontology
    for key, item in metadata_for_resource.items():
        setattr(memodel_r, key, item)

    memodel_r.seed = seed
    memodel_r.objectOfStudy = {
        "@id": "http://bbp.epfl.ch/neurosciencegraph/taxonomies/objectsofstudy/singlecells",
        "label": "Single Cell",
    }
    memodel_r.status = new_status
    if emodel_score is not None:
        memodel_r.score = emodel_score
    if emodel_threshold is not None:
        memodel_r.threshold_current = emodel_threshold
        if emodel_holding is not None:
            memodel_r.holding_current = emodel_holding
        else:
            memodel_r.holding_current = 0

    # do not add any description: we expect it to be already present

    # make memodel resource into a Dataset to be able to add images
    # have store_metadata=True to be able to update resource
    memodel_r = Dataset.from_resource(forge, memodel_r, store_metadata=True)

    # add images in memodel resource
    # Do NOT do this BEFORE turning resource into a Dataset.
    # That would break the storing LazyAction into a string

    for path in nexus_images:
        resource_type = path.split("__")[-1].split(".")[0]
        file_ext = path.split(".")[-1]

        memodel_r.add_image(
            path=path,
            content_type=mime_type_dict.get(file_ext, "application/octet-stream"),
            about=resource_type,
        )

    # update memodel resource
    L.debug("## Model after update")
    L.debug(memodel_r)
    forge.update(memodel_r)


def retrieve_resources(forge, memodel_id):
    """Retrieve ME-Model, EModel and Morphology resources."""
    memodel_r = forge.retrieve(memodel_id, cross_bucket=True)
    emodel_id, morph_id = get_ids_from_memodel(memodel_r)
    emodel_r = forge.retrieve(emodel_id, cross_bucket=True)
    morph_r = forge.retrieve(morph_id, cross_bucket=True)
    return memodel_r, emodel_r, morph_r, emodel_id, morph_id


def extract_emodel_metadata(emodel_r):
    """Extract metadata from EModel resource."""
    emodel = emodel_r.eModel if hasattr(emodel_r, "eModel") else None
    etype = emodel_r.eType if hasattr(emodel_r, "eType") else None
    ttype = emodel_r.tType if hasattr(emodel_r, "tType") else None
    mtype = emodel_r.mType if hasattr(emodel_r, "mType") else None
    species = None

    if hasattr(emodel_r, "subject"):
        if hasattr(emodel_r.subject, "species"):
            species = (
                emodel_r.subject.species.label
                if hasattr(emodel_r.subject.species, "label")
                else None
            )

    brain_region = None
    if hasattr(emodel_r, "brainLocation"):
        if hasattr(emodel_r.brainLocation, "brainRegion"):
            brain_region = (
                emodel_r.brainLocation.brainRegion.label
                if hasattr(emodel_r.brainLocation.brainRegion, "label")
                else None
            )

    iteration_tag = emodel_r.iteration if hasattr(emodel_r, "iteration") else None
    synapse_class = emodel_r.synapse_class if hasattr(emodel_r, "synapseClass") else None
    seed = int(emodel_r.seed if hasattr(emodel_r, "seed") else 0)

    subject_ontology = emodel_r.subject if hasattr(emodel_r, "subject") else None

    return {
        "emodel": emodel,
        "etype": etype,
        "ttype": ttype,
        "mtype": mtype,
        "species": species,
        "brain_region": brain_region,
        "iteration_tag": iteration_tag,
        "synapse_class": synapse_class,
        "seed": seed,
        "subject_ontology": subject_ontology,
    }


def extract_morph_metadata(morph_r):
    """Extract metadata from Morphology resource."""
    morph_name = morph_r.name if hasattr(morph_r, "name") else None
    morph_format = "swc"  # assumes swc is always present
    brain_location_ontology = morph_r.brainLocation if hasattr(morph_r, "brainLocation") else None

    return {
        "morph_name": morph_name,
        "morph_format": morph_format,
        "brain_location_ontology": brain_location_ontology,
    }


def create_access_point(emodel_metadata, organisation, project, endpoint, forge_path, access_token):
    """Create and configure BluePyEModel Nexus access point."""
    access_point = NexusAccessPoint(
        emodel=emodel_metadata["emodel"],
        etype=emodel_metadata["etype"],
        ttype=emodel_metadata["ttype"],
        mtype=emodel_metadata["mtype"],
        species=emodel_metadata["species"],
        brain_region=emodel_metadata["brain_region"],
        iteration_tag=emodel_metadata["iteration_tag"],
        synapse_class=emodel_metadata["synapse_class"],
        project=project,
        organisation=organisation,
        endpoint=endpoint,
        forge_path=forge_path,
        access_token=access_token,
    )

    # update settings for better threshold precision
    access_point.pipeline_settings.current_precision = 2e-3

    return access_point


def perform_analysis(access_point, cell_evaluator, seed, mapper, add_score=True):
    """Perform the model analysis and generate plots."""
    figures_dir = pathlib.Path("./figures") / access_point.emodel_metadata.emodel

    # Get scores from plot_scores, so that we don't have to run the model twice
    emodel_score, emodel_holding, emodel_threshold = plot(
        access_point, seed, cell_evaluator, figures_dir, mapper
    )

    if not add_score:
        emodel_score = None

    # If we have absolute amplitude protocols, and threshold current was not computed,
    # then compute it
    if emodel_threshold is None:
        # assume holding = 0
        emodel_holding = 0
        emodel_threshold = get_threshold(cell_evaluator, access_point, mapper)

    return emodel_score, emodel_holding, emodel_threshold, figures_dir


def update_memodel_status(forge: KnowledgeGraphForge, memodel: Resource, status: str) -> None:
    """Update the ME-Model status and save it."""
    memodel.status = status
    forge.update(memodel)


def run_me_model_analysis(memodel_self_url: str, access_token: str) -> None:
    """Run the analysis for a ME-Model."""
    forge_path = f"./nexus/forge-{settings.DEPLOYMENT_ENV}.yml"

    base_and_id = memodel_self_url.split("/")
    memodel_id = unquote(base_and_id[-1])

    # Extract values from the self url
    endpoint = "/".join(base_and_id[:-5])
    organisation = base_and_id[-4]
    project = base_and_id[-3]

    mapper = map
    # Configuration options
    update_emodel_name = True
    use_brain_region_from_morphology = True
    use_mtype_in_githash = True  # to distinguish from other MEModel
    add_score = True

    L.info(f"ME-Model ID: {memodel_id}")
    L.debug(f"Endpoint: {endpoint}")
    L.debug(f"Forge_path: {forge_path}")
    L.debug(f'Bucket: "{organisation}/{project}"')

    # Create forge client
    L.debug("Creating forge client")
    forge = connect_forge(
        bucket=f"{organisation}/{project}",
        endpoint=endpoint,
        access_token=access_token,
        forge_path=forge_path,
    )

    # Retrieve ME-Model resource
    try:
        L.debug("Retrieving ME-Model related resources")
        memodel_r, emodel_r, morph_r, emodel_id, morph_id = retrieve_resources(forge, memodel_id)

        # Set status to running
        L.debug("Setting ME-Model status to 'running'")
        update_memodel_status(forge, memodel_r, "running")

        # Extract metadata
        emodel_metadata = extract_emodel_metadata(emodel_r)
        morph_metadata = extract_morph_metadata(morph_r)

        # Create access point
        L.debug("Creating BluePyEModel Nexus access point")
        access_point = create_access_point(
            emodel_metadata, organisation, project, endpoint, forge_path, access_token
        )

        # Get cell evaluator with 'new' morphology
        L.debug("Creating cell evaluator")
        cell_evaluator = get_cell_evaluator(
            access_point, morph_metadata["morph_name"], morph_metadata["morph_format"], morph_id
        )

        # Get new emodel metadata
        L.info("Getting new EModel metadata")
        new_emodel_metadata = get_new_emodel_metadata(
            access_point,
            morph_id,
            morph_metadata["morph_name"],
            update_emodel_name,
            use_brain_region_from_morphology,
            use_mtype_in_githash,
        )

        # Perform analysis and generate plots
        L.debug("Creating plots")
        emodel_score, emodel_holding, emodel_threshold = perform_analysis(
            access_point, cell_evaluator, emodel_metadata["seed"], mapper, add_score
        )

        # Move figures to correspond to combined metadata
        copy_emodel_pdf_dependencies_to_new_path(
            access_point.emodel_metadata,
            new_emodel_metadata,
            True,
            True,
            emodel_metadata["seed"],
            overwrite=True,
        )

        # Get images for Nexus
        L.debug("Retrieving images from Nexus")
        nexus_images = get_nexus_images(
            access_point, emodel_metadata["seed"], new_emodel_metadata, morph_id, emodel_id
        )

        # Update ME-Model with results
        L.debug("Updating ME-Model")
        update_memodel(
            forge,
            memodel_r,
            emodel_metadata["seed"],
            new_emodel_metadata,
            emodel_metadata["subject_ontology"],
            morph_metadata["brain_location_ontology"],
            nexus_images,
            emodel_score,
            emodel_holding,
            emodel_threshold,
        )

        L.info("ME-Model analysis completed successfully")

    except Exception as e:
        L.error(f"Error during ME-Model analysis: {e!s}")
        try:
            # Set status to failed
            update_memodel_status(forge, memodel_r, "failed")
            L.info("ME-Model status set to 'failed'")
        except Exception as update_error:  # noqa: BLE001
            L.error(f"Failed to update ME-Model status: {update_error!s}")

        raise
