"""Main."""

import asyncio
import json
import os
import psutil
import signal
import threading
from contextlib import asynccontextmanager
from sched import scheduler
from threading import Thread
from typing import Annotated

import boto3
from botocore.exceptions import ClientError
from fastapi import BackgroundTasks, Depends, FastAPI, Header, Response
from fastapi.responses import JSONResponse

from app.config import settings
from app.handler import message_handler
from app.logger import L

SHUTDOWN_TIMER = 1800  # sec (30 min)

scheduler = scheduler()


def _cpu_monitor_worker(stop_event: threading.Event) -> None:
    """Worker function to monitor CPU in a separate thread."""
    while not stop_event.is_set():
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()

            # Get process metrics
            process = psutil.Process()
            process_cpu = process.cpu_percent()
            process_memory = process.memory_info()

            L.info(
                "CPU Monitor - System CPU: {:.1f}%, System Memory: {:.1f}%, "
                "Process CPU: {:.1f}%, Process Memory: {:.1f}MB",
                cpu_percent,
                memory_info.percent,
                process_cpu,
                process_memory.rss / 1024 / 1024,
            )
        except Exception as e:
            L.error("Error in CPU monitoring: {}", e)

        # Wait for 9 more seconds (since cpu_percent already waited 1 second)
        if not stop_event.wait(9):
            continue


async def cpumon_task() -> None:
    """Monitor and log CPU consumption every 10 seconds using a background thread."""
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=_cpu_monitor_worker, args=(stop_event,), daemon=True, name="CPUMonitor"
    )

    try:
        monitor_thread.start()
        L.info("CPU monitoring thread started")

        # Keep the task alive until cancelled
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        L.info("CPU monitoring task cancelled, stopping thread...")
        stop_event.set()
        monitor_thread.join(timeout=2)  # Wait up to 2 seconds for thread to finish
        raise


def _shutdown_timer() -> None:
    if len(scheduler.queue) == 0:
        L.info("Server idle timeout, shutting down...")
        os.kill(os.getpid(), signal.SIGTERM)


def no_cache(response: Response) -> Response:
    """Add Cache-Control: no-cache to the response headers."""
    response.headers["Cache-Control"] = "no-cache"
    return response


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Start idle shutdown timer, CPU monitor and close websocket connection when done."""
    scheduler.enter(SHUTDOWN_TIMER, 1, _shutdown_timer)
    Thread(target=scheduler.run, daemon=True).start()

    # Start CPU monitoring task
    cpumon_task_handle = asyncio.create_task(cpumon_task())
    L.info("CPU monitoring task started")

    try:
        yield
    finally:
        # Cancel CPU monitoring task
        cpumon_task_handle.cancel()
        try:
            await cpumon_task_handle
        except asyncio.CancelledError:
            L.info("CPU monitoring task cancelled")

        if settings.APIGW_ENDPOINT is None:
            return

        apigw = boto3.client(
            "apigatewaymanagementapi",
            endpoint_url=settings.APIGW_ENDPOINT,
            region_name=settings.APIGW_REGION,
        )
        try:
            apigw.delete_connection(ConnectionId=settings.APIGW_CONN_ID)
        except ClientError:
            L.exception("Couldn't post to connection")
        except apigw.exceptions.GoneException:
            L.exception("Connection is gone.")


app = FastAPI(lifespan=lifespan)


def send_message(apigw, conn, data):
    """Send message to frontend."""
    try:
        apigw.post_to_connection(Data=json.dumps(data), ConnectionId=conn)
    except ClientError:
        L.exception("Couldn't post to connection")
    except apigw.exceptions.GoneException:
        L.exception("Connection is gone.")


def process_message(msg: dict) -> None:
    """Call different functions based on message."""
    apigw = settings.APIGW_ENDPOINT
    region = settings.APIGW_REGION
    conn = settings.APIGW_CONN_ID

    try:
        result = message_handler(msg)
    except Exception as e:
        L.exception("Error during processing msg=%s: %s", msg, e)
        raise Exception(e) from e

    if apigw is None:
        L.info("No apigw endpoint, skipping message send.")
        return

    apigw = boto3.client("apigatewaymanagementapi", endpoint_url=apigw, region_name=region)
    send_message(apigw, conn, result)


def process_message_threaded(*args) -> None:
    """Process message in background."""
    threading.Thread(target=process_message, args=args).start()


@app.post("/init")
def init(msg: dict):
    """Initialize service."""
    L.info("INIT msg=%s", msg.keys())
    try:
        message_handler(msg)
    except Exception as e:
        L.exception("Error during processing msg=%s: %s", msg, e)
        raise Exception(e) from e


@app.post("/default")
def default(msg: dict, background_tasks: BackgroundTasks) -> JSONResponse:
    """Process message."""
    scheduler.enter(SHUTDOWN_TIMER, 1, _shutdown_timer)
    background_tasks.add_task(process_message_threaded, msg)

    response = (
        {"cmd": f"{msg['cmd']}_processing"} if "cmd" in msg else {"message": "Processing message"}
    )

    return JSONResponse(status_code=202, content=response)


@app.post("/shutdown")
def shutdown() -> Response:
    """Shutdown the service."""
    L.info("shutdown")
    os.kill(os.getpid(), signal.SIGTERM)
    return Response(status_code=202)


@app.get("/health", dependencies=[Depends(no_cache)])
async def health() -> Response:
    """Health endpoint."""
    L.info("health")
    return Response(status_code=204)


@app.post("/test-run")
def run(
    msg: dict,
    background_tasks: BackgroundTasks,
    authorization: Annotated[str | None, Header()] = None,
) -> Response:
    """Run analysis."""
    if authorization is None:
        err_msg = "Missing authorization header"
        raise ValueError(err_msg)

    message_handler(
        {
            "cmd": "run_analysis",
            "data": {"config": msg, "access_token": authorization.replace("Bearer ", "")},
        }
    )

    return Response(status_code=204)
