"""Actions."""

from app.logger import L
from app.nexus_helper import run_me_model_analysis

TOKEN = None
MODEL_SELF_URL = None


def set_token(token):
    """Set token to fetch model in the future."""
    global TOKEN  # pylint: disable=global-statement
    L.debug("Setting token...")
    TOKEN = token.replace("Bearer ", "")


def set_model(values):
    """Set model."""
    L.debug("Setting model: %s", values)
    model_self_url = values.get("model_self_url")

    if model_self_url is None:
        raise Exception("Missing model url")

    global MODEL_SELF_URL  # pylint: disable=global-statement
    MODEL_SELF_URL = model_self_url
    return True


def run_analysis(values):
    """Run analysis."""
    L.info("Running analysis %s", values)
    try:
        run_me_model_analysis(MODEL_SELF_URL, TOKEN)
    except Exception as e:
        L.exception(e)
        raise e
    L.info("Analysis done")
    return True
