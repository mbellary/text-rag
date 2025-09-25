import json
import os
from text_rag.aws_clients import bedrock_client
from text_rag.config import EMBEDDING_MODEL, OPENAI_API_KEY
from text_rag.logger import get_logger
from openai import OpenAI

logger = get_logger("text_rag.utils")

# async def embed_text(text: str) -> list:
#     client = bedrock_client()
#     payload = {"input": text}
#     resp = client.invoke_model(
#         modelId=EMBEDDING_MODEL,
#         contentType="application/json",
#         accept="application/json",
#         body=json.dumps(payload)
#     )
#     data = json.loads(resp['body'].read())
#     embedding = data.get('embedding') or data.get('output', [{}])[0].get('embedding')
#     logger.info("Successfully embedded the query.")
#     return embedding

async def invoke_bedrock_embedding(text: str):
    """
    Calls Bedrock to get embeddings for provided text.
    Returns list[float]
    """
    # Convert to the invocation shape your model expects.
    # Many embedding models accept: {"input": "<text>"} or a JSON wrapper.
    #client = await get_aboto3_client("bedrock-runtime")

    async with bedrock_client() as br:
        try:
            model_id = EMBEDDING_MODEL
            # Prepare input; model-specific. This is generic JSON body.
            payload = {"input": text}
            resp = await br.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload).encode("utf-8"),
            )
            # resp is a streaming/binary body. Read and parse
            body_bytes = await resp['body'].read()
            data = json.loads(body_bytes.decode("utf-8"))
            # Assume model returns {"embeddings": [ ... ]} or {"embedding":[...]}
            if "embedding" in data:
                return data["embedding"]
            if "embeddings" in data:
                return data["embeddings"]
            # If model returns text, attempt to parse numeric list
            if isinstance(data, dict):
                # try common keys
                for key in data:
                    if isinstance(data[key], list):
                        return data[key]
            #raise RuntimeError(f"Unexpected bedrock response: {data}")
        except Exception as dre:
            logger.error(f'Failed to embed the text - {dre}')


async def invoke_embedding_model(text: str):
    """
    Adapter to invoke LLM across providers:
      - openai → OpenAI API (gpt-4, gpt-4o-mini, etc.)
      - bedrock → AWS Bedrock
    """
    mode = os.getenv("MODEL_PROVIDER", "bedrock").lower()

    if mode == "openai":
        # OpenAI API (requires OPENAI_API_KEY in env)
        logger.info("Initializing Open API model")
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding

    elif mode == "bedrock":
        # AWS Bedrock
        logger.info("Initializing Bedrock API model")
        output = invoke_bedrock_embedding(text)
        return output
    else:
        raise ValueError(f"Unknown MODEL_PROVIDER: {mode}")
