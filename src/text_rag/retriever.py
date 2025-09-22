from typing import List, Dict
from text_rag.aws_clients import opensearch_client
from text_rag.config import OPENSEARCH_INDEX

def vector_search(query_embedding: List[float], k: int) -> List[Dict]:
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
        }
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
    return results
