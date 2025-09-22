import aioredis
import hashlib
import json
from text_rag.config import REDIS_HOST, REDIS_PORT
from text_rag.logger import get_logger

logger = get_logger("text_rag.cache")

_redis = None

async def get_redis():
    global _redis
    if _redis is None:
        _redis = await aioredis.from_url(
            f"redis://{REDIS_HOST}:{REDIS_PORT}",
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis

def _make_key(query: str) -> str:
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
    return f"rag:query:{digest}"

async def get_cached_response(query: str) -> dict | None:
    redis = await get_redis()
    key = _make_key(query)
    cached = await redis.get(key)
    if cached:
        logger.info("cache_hit", query=query)
        return json.loads(cached)
    return None

async def set_cached_response(query: str, response: dict, ttl: int = 300):
    redis = await get_redis()
    key = _make_key(query)
    await redis.set(key, json.dumps(response), ex=ttl)
    logger.info("cache_set", query=query)
