import json
import os
from openai import OpenAI, AsyncOpenAI
from typing import List, Dict, Any
from text_rag.aws_clients import bedrock_client
from text_rag.config import RERANK_MODEL, MODEL_PROVIDER, OPENAI_API_KEY
from text_rag.logger import get_logger
from typing import List, Dict

logger = get_logger("text_rag.reranker")

def bedrock_reranker(query: str, candidates: List[Dict], top_n: int) -> List[int]:
    client = bedrock_client()
    prompt = {"input": {"query": query, "candidates": candidates}}
    try:
        resp = client.invoke_model(
            modelId= RERANK_MODEL,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(prompt)
        )
        payload = json.loads(resp['body'].read())
        scores = payload['scores']
        logger.info(f"successfully reranked answer.")
    except Exception as e:
        logger.error(f"failed to rerank answer - {e}.")
        return list(range(min(len(candidates), top_n)))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked[:top_n]



# Initialize OpenAI client (ensure OPENAI_API_KEY is set in env)

def normalize_rerank_scores(rerank_data):
    """
    Normalize rerank_data into a list of {"id": str, "score": int}.
    Handles different possible shapes returned by the model.
    """
    score_map = {}
    if isinstance(rerank_data, list):
        score_map = {str(item["id"]): item["score"] for item in rerank_data}
        return score_map

    elif isinstance(rerank_data, dict):
        if "id" in rerank_data and "score" in rerank_data:
            score_map = {str(rerank_data["id"]): rerank_data["score"]}
            return score_map

        # Case 2: dict with a key wrapping the list
        for v in rerank_data.values():
            if isinstance(v, list):
                score_map = {str(item["id"]): item["score"] for item in v}
                return score_map
    # If nothing matches, raise error
    raise ValueError(f"Unexpected rerank format: {rerank_data}")

async def openai_reranker(query: str, candidates: List[Dict], top_n: int) -> List[int]:
    """
    Rerank retrieved documents using OpenAI models.

    Args:
        query (str): The user query
        candidates (List[Dict]): List of candidate docs, each with {"id": ..., "text": ..., "score": ...}
        model (str): OpenAI model to use (default: gpt-4o-mini)

    Returns:
        List[Dict]: Reranked candidates with an added "rerank_score"
    """

    # Format the candidate docs for the model
    #docs_text = "\n".join([f"[{i+1}] {doc['text']}" for i, doc in enumerate(candidates)])
    docs_text = "\n".join([f"{doc['doc_id']} {doc['text']}" for i, doc in enumerate(candidates)])
    system_prompt = (
        "You are a reranking assistant. "
        "Given a query and candidate documents, you assign a relevance score (0-1). "
        "Higher score = more relevant to the query. Only score based on semantic relevance."
    )

    user_prompt = f"Query: {query}\n\nDocuments:\n{docs_text}\n\nReturn a JSON list of objects in the form: " \
                  f'[{{"id": <candidate_id>, "score": <float 0-1>}}] sorted by score desc.'
    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model=RERANK_MODEL,
            temperature=1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as api_error:
        raise ValueError(f"Open API reranker call failed : {api_error}")

    rerank_result = response.choices[0].message.content

    try:
        rerank_scores = json.loads(rerank_result)
        logger.info(f"reranked the scores {rerank_scores}")
    except Exception as e:
        raise ValueError(f"Failed to parse rerank response: {rerank_result}") from e

    try:
        score_map = normalize_rerank_scores(rerank_scores)
        for doc in candidates:
            doc_id = str(doc["doc_id"])
            doc["rerank_score"] = score_map.get(doc_id, 0)
    except Exception as scoring_error:
        raise ValueError(f"Failed to Merge ranks! - {scoring_error}")

    # Sort by rerank_score
    reranked_scores = [candidate['rerank_score'] for candidate in candidates]
    ranked = sorted(range(len(reranked_scores)), key=lambda i: reranked_scores[i], reverse=True)
    return ranked[:top_n]


def invoke_reranking_model(query: str, candidates: List[Dict], top_n: int) -> List[int]:
    if MODEL_PROVIDER == 'openai':
        logger.info("Initializing Open API reranking model")
        results = openai_reranker(query, candidates, top_n)
        return results
    elif MODEL_PROVIDER == 'bedrock':
        logger.info("Initializing Bedrock reranking model")
        results = bedrock_reranker(query, candidates, top_n)
        return results
    else:
        raise ValueError(f"Unknown MODEL_PROVIDER: {MODEL_PROVIDER}")

