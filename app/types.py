"""Types"""

from enum import StrEnum
from typing import Annotated, Literal

from entitysdk import ProjectContext
from pydantic import BaseModel, Field


class ModelOrigin(StrEnum):
    """Model origin."""

    NEXUS = "nexus"
    ENTITYCORE = "entitycore"


class NexusModelAnalysisConfig(BaseModel):
    """Model analysis configuration for Nexus."""

    model_origin: Literal[ModelOrigin.NEXUS]
    self_url: str


class EntitycoreModelAnalysisConfig(BaseModel):
    """Model analysis configuration for EntityCore."""

    model_origin: Literal[ModelOrigin.ENTITYCORE]
    model_id: str
    project_context: ProjectContext | None = None


class ModelAnalysisRequest(BaseModel):
    """Wrapper for model analysis configuration."""

    access_token: str

    config: Annotated[
        NexusModelAnalysisConfig | EntitycoreModelAnalysisConfig,
        Field(discriminator="model_origin"),
    ]
