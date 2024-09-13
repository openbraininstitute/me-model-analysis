"""Handler."""
from .actions import run_analysis, set_model, set_token

function_mapping = {
    'set_model': set_model,
    'run_analysis': run_analysis,
}


def message_handler(msg):
    """Handle message."""
    if 'token' in msg:
        set_token(msg['token'])
        return {'message': 'Token set'}

    if 'cmd' not in msg:
        return {'message': 'Service up'}

    command_name = msg['cmd']
    data = msg['data']

    if command_name not in function_mapping:
        raise Exception('Unknown command: ' + command_name)

    try:
        result = function_mapping[command_name](data)
    except Exception:
        return {
            'cmd': f"{command_name}_error"
        }

    if result is None:
        raise Exception('Empty result')

    return {
        'cmd': f"{command_name}_done",
        'data': result
    }
