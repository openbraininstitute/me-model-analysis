"""Actions."""

from concurrent.futures import ProcessPoolExecutor
from typing import Any

from entitysdk import Client

from app.config import settings
from app.entitycore_helper import (
    run_and_save_calibration_validation as run_me_model_analysis_entitycore,
)
from app.logger import L
from app.nexus_helper import run_me_model_analysis as run_me_model_analysis_nexus
from app.types import ModelAnalysisRequest, ModelOrigin


def run_analysis(values: dict) -> Any:
    """Run analysis."""
    request = ModelAnalysisRequest(**values)
    access_token = request.access_token
    config = request.config

    if config.model_origin == ModelOrigin.ENTITYCORE:
        L.info("Creating EntityCore client")
        client = Client(
            environment=settings.DEPLOYMENT_ENV,
            project_context=config.project_context,
            token_manager=access_token,
        )
        L.info("About to run analysis")
        run_me_model_analysis_entitycore(client, config.model_id)

        with ProcessPoolExecutor() as executor:
            future = executor.submit(run_me_model_analysis_entitycore, client, config.model_id)
            result = future.result()
            L.info(result)

    elif config.model_origin == ModelOrigin.NEXUS:
        run_me_model_analysis_nexus(config.self_url, access_token)

    else:
        msg = "Unsupported model origin"
        raise ValueError(msg)

    L.info("Analysis done")
