import json
from text_rag.aws_clients import bedrock_client
from text_rag.config import BEDROCK_RERANK_MODEL
from typing import List

def rerank(query: str, candidates: List[str], top_n: int) -> List[int]:
    client = bedrock_client()
    prompt = {"input": {"query": query, "candidates": candidates}}
    try:
        resp = client.invoke_model(
            modelId= BEDROCK_RERANK_MODEL,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(prompt)
        )
        payload = json.loads(resp['body'].read())
        scores = payload['scores']
    except Exception:
        return list(range(min(len(candidates), top_n)))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked[:top_n]
