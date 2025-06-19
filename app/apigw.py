import json

import boto3
from botocore.exceptions import ClientError

from app.config import settings
from app.logger import L

apigw_client = boto3.client(
    "apigatewaymanagementapi",
    endpoint_url=settings.APIGW_ENDPOINT,
    region_name=settings.APIGW_REGION,
)


def send_message(data):
    """Send message to frontend."""
    try:
        apigw_client.post_to_connection(Data=json.dumps(data), ConnectionId=settings.APIGW_CONN_ID)
    except ClientError:
        L.exception("Couldn't post to connection")
    except apigw_client.exceptions.GoneException:
        L.exception("Connection is gone.")
