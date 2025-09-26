import asyncio
from text_rag.retriever import vector_search
from text_rag.reranker import invoke_reranking_model
from text_rag.generator import invoke_generator_model
from text_rag.utils import invoke_embedding_model
from text_rag.config import RETRIEVAL_K, RERANK_TOP_N
from text_rag.cache import get_cached_response, set_cached_response
from text_rag.logger import get_logger

logger = get_logger("text_rag.worker")

async def handle_query(query: str, k: int = None, n: int = None, do_reflection: bool = False):
    #check cache
    cached = await get_cached_response(query)
    if cached:
        return cached

    k = k or RETRIEVAL_K
    n = n or RERANK_TOP_N

    #embedding
    query_embedding = await invoke_embedding_model(query)

    #retrieve top-k
    raw_candidates = await vector_search(query_embedding, k)
    if not raw_candidates:
        logger.info("No documents found")
        return {"answer": "No documents found.", "sources": []}
    #candidate_texts = [c['chunk'] for c in raw_candidates]

    #rerank
    ranked_indices = await invoke_reranking_model(query, raw_candidates, n)
    top_chunks = [raw_candidates[i] for i in ranked_indices]

    #generate
    answer = await invoke_generator_model(query, top_chunks)

    return {"answer": answer, "sources": [{"doc_id": c['doc_id'], "score": c['score'], "text": c['text']} for c in top_chunks]}
