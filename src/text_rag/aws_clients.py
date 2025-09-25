import boto3
import os
from text_rag.config import (
                            AWS_REGION,
                            AWS_ACCESS_KEY_ID,
                            AWS_SECRET_ACCESS_KEY,
                            LOCALSTACK_URL,
                            APP_ENV,
                            OPENSEARCH_HOST)
from typing import Any
from text_rag.logger import get_logger
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

logger = get_logger("text_rag.clients")

_session = boto3.Session(region_name=AWS_REGION)

def get_boto3_client(service):
    if APP_ENV == "localstack":
        # LocalStack setup
        logger.info(f"Initializing client {service} locally")
        return boto3.client(
            service,
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            endpoint_url=LOCALSTACK_URL
        )
    else:
        os.environ.pop("AWS_ACCESS_KEY_ID", None)
        os.environ.pop("AWS_SECRET_ACCESS_KEY", None)

        logger.info(f"Initializing client {service} in production")
        aws_profile = os.getenv("AWS_PROFILE")
        if aws_profile:
            logger.info(f"Initializing client {service} in production using AWS_PROFILE {aws_profile}")
            session = boto3.Session(region_name=AWS_REGION, profile_name=aws_profile)
            return session.client(service)
        else:
            # No profile → IAM Role will be used (via metadata service)
            logger.info(f"Initializing client {service} in production using IAM Role")
            return boto3.client(service, region_name=AWS_REGION)

def s3_client() -> Any:
    return get_boto3_client("s3")

def bedrock_client() -> Any:
    return get_boto3_client("bedrock-runtime") # changed "bedrock" to "bedrock-runtime"


def opensearch_client():


    credentials = _session.get_credentials()
    awsauth = AWS4Auth(region=AWS_REGION,
                        service="es",
                        refreshable_credentials=credentials)

    client = OpenSearch(
        hosts={"host": OPENSEARCH_HOST.replace("https://", "").replace("http://", ""), "port": 443},
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )
    return client
