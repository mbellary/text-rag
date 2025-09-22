import json
from text_rag.aws_clients import bedrock_client
from text_rag.config import BEDROCK_COMPLETION_MODEL

PROMPT_TEMPLATE = """
You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer concisely and cite context chunks by id where appropriate.
"""

def generate_answer(question: str, context_chunks: list) -> str:
    client = bedrock_client()
    context_text = "\n\n".join([f"[id={c['id']}] {c['chunk']}" for c in context_chunks])
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=question)
    try:
        resp = client.invoke_model(
            modelId=BEDROCK_COMPLETION_MODEL,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": prompt, "maxTokens": 512})
        )
        payload = json.loads(resp['body'].read())
        answer = payload.get('outputText') or payload.get('choices', [{}])[0].get('text')
    except Exception:
        answer = "[error] failed to generate answer"
    return answer
