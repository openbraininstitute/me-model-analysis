"""Main."""

import json
import os
import signal
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
    """Start idle shutdown timer and close websockt connection when done."""
    scheduler.enter(SHUTDOWN_TIMER, 1, _shutdown_timer)
    Thread(target=scheduler.run, daemon=True).start()
    yield

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

    L.info(f"apigw={apigw} region={region} conn={conn}")
    L.info(f"processing msg={msg} ...")

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
    L.info("SVC DEFAULT msg=%s", msg)
    # reset idle shutdown timer
    scheduler.enter(SHUTDOWN_TIMER, 1, _shutdown_timer)
    background_tasks.add_task(process_message, msg)

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

    background_tasks.add_task(
        message_handler,
        {
            "cmd": "run_analysis",
            "data": {"config": msg, "access_token": authorization.replace("Bearer ", "")},
        },
    )

    return Response(status_code=204)
