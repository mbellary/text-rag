import asyncio
from text_rag.retriever import vector_search
from text_rag.reranker import rerank
from text_rag.generator import generate_answer
from text_rag.utils import embed_text
from text_rag.config import RETRIEVAL_K, RERANK_TOP_N
from text_rag.cache import get_cached_response, set_cached_response


async def handle_query(query: str, k: int = None, n: int = None, do_reflection: bool = False):
    #check cache
    cached = await get_cached_response(query)
    if cached:
        return cached

    k = k or RETRIEVAL_K
    n = n or RERANK_TOP_N

    #embedding
    query_embedding = await embed_text(query)

    #retrieve top-k
    raw_candidates = await asyncio.to_thread(vector_search, query_embedding, k)
    if not raw_candidates:
        return {"answer": "No documents found.", "sources": []}
    candidate_texts = [c['chunk'] for c in raw_candidates]

    #rerank
    ranked_indices = await asyncio.to_thread(rerank, query, candidate_texts, n)
    top_chunks = [raw_candidates[i] for i in ranked_indices]

    #generate
    answer = await asyncio.to_thread(generate_answer, query, top_chunks)

    return {"answer": answer, "sources": [{"id": c['id'], "score": c['score'], "text": c['chunk']} for c in top_chunks]}
