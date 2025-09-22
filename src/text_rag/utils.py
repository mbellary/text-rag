import json
from text_rag.aws_clients import bedrock_client
from text_rag.config import BEDROCK_EMBEDDING_MODEL

async def embed_text(text: str) -> list:
    client = bedrock_client()
    payload = {"input": text}
    resp = client.invoke_model(
        modelId=BEDROCK_EMBEDDING_MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload)
    )
    data = json.loads(resp['body'].read())
    embedding = data.get('embedding') or data.get('output', [{}])[0].get('embedding')
    return embedding
