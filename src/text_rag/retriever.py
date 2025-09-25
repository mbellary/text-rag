import json
import aiohttp
import boto3
from typing import List, Dict
from text_rag.aws_clients import opensearch_client
from text_rag.config import OPENSEARCH_INDEX, OPENSEARCH_HOST, AWS_REGION
from text_rag.logger import get_logger
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

logger = get_logger("text_rag.retriever")

def _sign_request(method: str, url: str, body: bytes = b"", service="es", region=None):
    """
    Create headers with AWS SigV4 signature for raw HTTP request to OpenSearch.
    Returns dict(headers).
    """
    if OPENSEARCH_HOST.startswith("http://localhost"):
        return {}

    region = AWS_REGION
    # Get credentials synchronously from boto3
    boto_session = boto3.session.Session()
    creds = boto_session.get_credentials()
    frozen = creds.get_frozen_credentials()
    #aws_credentials = Credentials(creds.access_key, creds.secret_key, creds.token)
    request = AWSRequest(method=method, url=url, data=body)
    SigV4Auth(frozen, service, region).add_auth(request)
    return dict(request.headers.items())

def vector_search_v1(query_embedding: List[float], k: int) -> List[Dict]:
    client = opensearch_client()
    body = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": k
                }
            }
        },
        "min_score": 0.6
    }
    resp = client.search(index=OPENSEARCH_INDEX, body=body)
    results = []
    for hit in resp.get("hits", {}).get("hits", []):
        src = hit.get("_source", {})
        results.append({
            "id": hit.get("_id"),
            "score": hit.get("_score"),
            "metadata": src.get("metadata"),
            "chunk": src.get("text"),
            "embedding": src.get("embedding")
        })
    logger.info("Successfully retrieved the results.")
    return results

def _parse_opensearch_results(results, id_field="_id", text_field="text"):
    """
    Parse OpenSearch vector search results into a structured list of dictionaries.

    Args:
        results (dict): Raw response from OpenSearch (JSON).
        id_field (str): Field to use as document ID (default: "_id").
        text_field (str): Field inside `_source` that contains the text (default: "text").

    Returns:
        list[dict]: A list of dicts with keys: doc_id, text, score.
    """
    if not results or "hits" not in results or "hits" not in results["hits"]:
        return []

    parsed = []
    for hit in results["hits"]["hits"]:
        doc_id = hit.get(id_field, None)
        text = hit.get("_source", {}).get(text_field, None)
        score = hit.get("_score", None)

        parsed.append({
            "doc_id": doc_id,
            "text": text,
            "score": score
        })

    return parsed

async def vector_search(vector: list[float], k: int = 5):
    """
    Run a k-NN vector similarity search in OpenSearch.
    """
    url = f"{OPENSEARCH_HOST}/{OPENSEARCH_INDEX}/_search"
    body = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": vector,
                    "k": k
                }
            }
        },
        "min_score": 0.50
    }

    body_bytes = json.dumps(body).encode("utf-8")
    headers = _sign_request("POST", url, body_bytes, service="es")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, data=body_bytes, headers={**headers, "Content-Type": "application/json"}
        ) as resp:
            text =  await resp.text()
            logger.info(f"Vector search succeeded.")
            if resp.status != 200:
                logger.error("Vector search failed %s %s", resp.status, text)
                raise RuntimeError(f"Vector search failed: {resp.status} {text}")
            results = _parse_opensearch_results(json.loads(text))
            return results