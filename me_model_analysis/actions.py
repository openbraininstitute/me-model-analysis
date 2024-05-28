"""Actions."""
from .settings import L

TOKEN = None


def set_token(token):
    """Set token to fetch model in the future."""
    global TOKEN  # pylint: disable=global-statement
    L.info('Setting token...')
    TOKEN = token


def set_model(values):
    """Set model."""
    model_id = values.get('model_id')

    if model_id is None:
        raise Exception('Missing model id')

    L.debug('Setting model: %s', model_id)
    return True


def run_analysis(values):
    """Run analysis."""
    L.debug('Running analysis %s', values)
    return True
