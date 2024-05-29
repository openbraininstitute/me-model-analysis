"""Actions."""
from .nexus_helper import run_me_model_analysis
from .settings import L

TOKEN = None
MODEL_ID = None


def set_token(token):
    """Set token to fetch model in the future."""
    global TOKEN  # pylint: disable=global-statement
    L.debug('Setting token...')
    TOKEN = token.replace('Bearer ', '')


def set_model(values):
    """Set model."""
    L.debug('Setting model: %s', values)
    model_id = values.get('model_id')

    if model_id is None:
        raise Exception('Missing model id')

    global MODEL_ID  # pylint: disable=global-statement
    MODEL_ID = model_id
    L.info('Setting model: %s', model_id)
    return True


def run_analysis(values):
    """Run analysis."""
    L.debug('Running analysis %s', values)
    try:
        run_me_model_analysis(MODEL_ID, TOKEN)
    except Exception as e:
        L.exception(e)
        raise e
    L.info('Analysis done')
    return True
