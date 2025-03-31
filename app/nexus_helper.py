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
from kgforge.core import KnowledgeGraphForge
from kgforge.specializations.resources import Dataset

from app.logger import L

mime_type_dict = {"png": "image/png", "pdf": "application/pdf"}


def connect_forge(bucket, endpoint, access_token, forge_path=None):
    """Creation of a forge session."""
    if not forge_path:
        raise ValueError("Missing path to forge config file")
    forge = KnowledgeGraphForge(
        forge_path, bucket=bucket, endpoint=endpoint, token=access_token, debug=True
    )
    return forge


def get_ids_from_memodel(memodel_r):
    """Get EModel and Morphology ids from MEModel resource metadata."""
    emodel_id = None
    morph_id = None
    if not hasattr(memodel_r, "hasPart"):
        raise AttributeError("ME-Model resource has no 'hasPart' metadata")
    for haspart in memodel_r.hasPart:
        if haspart.type == "EModel":
            emodel_id = haspart.id
        elif haspart.type == "NeuronMorphology":
            morph_id = haspart.id
    if emodel_id is None:
        raise TypeError("Could not find any EModel resource id link in MEModel resource.")
    if morph_id is None:
        raise TypeError("Could not find any NeuronMorphology resource id link in MEModel resource.")

    return emodel_id, morph_id


def get_morph_mtype(annotation):
    """Get morph mtype from annotation."""
    morph_mtype = None
    if hasattr(annotation, "hasBody"):
        if hasattr(annotation.hasBody, "label"):
            morph_mtype = annotation.hasBody.label
        else:
            raise ValueError("Morphology resource has no label in annotation.hasBody.")
    else:
        raise ValueError("Morphology resource has no hasBodz in annotation.")

    return morph_mtype


def get_morph_metadata(access_point, morph_id):
    """Get morph metadata."""
    resource = access_point.access_point.retrieve(morph_id)
    if resource is None:
        raise TypeError(f"Could not find the morphology resource with id {morph_id}")

    morph_brain_region = None
    if hasattr(resource, "brainLocation"):
        if hasattr(resource.brainLocation, "brainRegion"):
            if hasattr(resource.brainLocation.brainRegion, "label"):
                morph_brain_region = resource.brainLocation.brainRegion.label
            else:
                raise AttributeError(
                    "Morphology resource has no label in brainLocation.brainRegion"
                )
        else:
            raise AttributeError("Morphology resource has no brainRegion in brainLocation.")
    else:
        raise AttributeError("Morphology resource has no brainLocation.")

    morph_mtype = None
    if not hasattr(resource, "annotation"):
        raise AttributeError("Morphology resource has no annotation.")

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
        raise TypeError("Could not find mtype in morphology resource")

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


def run_me_model_analysis(memodel_self_url, access_token):
    """Run the analysis."""
    forge_path = "./nexus/forge.yml"

    base_and_id = memodel_self_url.split("/")
    memodel_id = unquote(base_and_id[-1])

    # extract values from the self url
    endpoint = "/".join(base_and_id[:-5])
    organisation = base_and_id[-4]
    project = base_and_id[-3]

    mapper = map
    # also available:
    # from bluepyemodel.tools.multiprocessing import get_mapper

    # mapper = get_mapper(backend="ipyparallel")
    # mapper = get_mapper(backend="multiprocessing")

    # MEModel metadata-related config
    update_emodel_name = True
    use_brain_region_from_morphology = True
    use_mtype_in_githash = True  # to distinguish from other MEModel
    add_score = True

    L.debug(f"Endpoint: {endpoint}")
    L.debug(f"Forge_path: {forge_path}")
    L.debug(f'Bucket: "{organisation}/{project}"')

    # create forge and retrieve ME-Model
    L.debug("Creating forge client")
    forge = connect_forge(
        bucket=f"{organisation}/{project}",
        endpoint=endpoint,
        access_token=access_token,
        forge_path=forge_path,
    )

    # memodel resource
    L.debug("Retrieving ME-Model related resources")
    memodel_r = forge.retrieve(memodel_id, cross_bucket=True)
    emodel_id, morph_id = get_ids_from_memodel(memodel_r)
    emodel_r = forge.retrieve(emodel_id, cross_bucket=True)
    morph_r = forge.retrieve(morph_id, cross_bucket=True)

    L.debug("Setting ME-Model status to 'running'")
    memodel_r.status = "running"
    forge.update(memodel_r)

    # get metadata from EModel resource
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

    # get morph metadata
    morph_name = morph_r.name if hasattr(morph_r, "name") else None
    morph_format = "swc"  # assumes swc is always present and we do not care about small differences between format

    # additional metadata we will need when saving me-model resource
    subject_ontology = emodel_r.subject if hasattr(emodel_r, "subject") else None
    brain_location_ontology = morph_r.brainLocation if hasattr(morph_r, "brainLocation") else None

    # feed nexus access point with appropriate data
    L.debug("Creating BluePyEModel Nexus access point")
    access_point = NexusAccessPoint(
        emodel=emodel,
        etype=etype,
        ttype=ttype,
        mtype=mtype,
        species=species,
        brain_region=brain_region,
        iteration_tag=iteration_tag,
        synapse_class=synapse_class,
        project=project,
        organisation=organisation,
        endpoint=endpoint,
        forge_path=forge_path,
        access_token=access_token,
    )

    # update settings for better threshold precision
    access_point.pipeline_settings.current_precision = 2e-3

    # get cell evaluator with 'new' morphology
    L.debug("Creating cell evaluator")
    cell_evaluator = get_cell_evaluator(access_point, morph_name, morph_format, morph_id)

    # get new emodel metadata (mtype, emodel, brain region, iteration/githash)
    # to correspond to combined metadata of emodel and morphology
    L.info("Getting new EModel metadata")
    new_emodel_metadata = get_new_emodel_metadata(
        access_point,
        morph_id,
        morph_name,
        update_emodel_name,
        use_brain_region_from_morphology,
        use_mtype_in_githash,
    )

    L.debug("Creating plots")
    figures_dir = pathlib.Path("./figures") / access_point.emodel_metadata.emodel
    # trick: get scores from plot_scores, so that we don't have to run the model twice
    emodel_score, emodel_holding, emodel_threshold = plot(
        access_point, seed, cell_evaluator, figures_dir, mapper
    )
    if not add_score:
        emodel_score = None

    # if we have absolute amplitude protocols, and threshold current was not computed,
    # then compute it
    if emodel_threshold is None:
        # assume holding = 0
        emodel_holding = 0
        emodel_threshold = get_threshold(cell_evaluator, access_point, mapper)

    # Attention! after this step, do NOT push EModel again.
    # It has been modified and would overwrite the correct one on nexus

    # move figures: to correspond to combined metadata of emodel and morphology
    copy_emodel_pdf_dependencies_to_new_path(
        access_point.emodel_metadata,
        new_emodel_metadata,
        True,
        True,
        seed,
        overwrite=True,
    )

    L.debug("Retrieving images from Nexus")
    nexus_images = get_nexus_images(access_point, seed, new_emodel_metadata, morph_id, emodel_id)

    L.debug("Updating ME-Model")
    update_memodel(
        forge,
        memodel_r,
        seed,
        new_emodel_metadata,
        subject_ontology,
        brain_location_ontology,
        nexus_images,
        emodel_score,
        emodel_holding,
        emodel_threshold,
    )
