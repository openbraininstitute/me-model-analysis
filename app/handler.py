"""Handler."""

from app.actions import run_analysis
from app.logger import L

function_mapping = {
    "run_analysis": run_analysis,
    "ping": lambda _: {"message": "Service up"},
}


def message_handler(msg):
    """Handle message."""
    if "cmd" not in msg:
        return {"message": "Service up"}

    command_name = msg["cmd"]
    data = msg["data"]

    if command_name not in function_mapping:
        raise Exception("Unknown command: " + command_name)

    try:
        result = function_mapping[command_name](data)
    except Exception as e:
        L.error(e)
        return {"cmd": f"{command_name}_error"}

    return {"cmd": f"{command_name}_done", "data": result}
