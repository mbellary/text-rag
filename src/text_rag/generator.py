import json
from typing import Dict, Any, List

from openai import AsyncOpenAI
from text_rag.aws_clients import bedrock_client
from text_rag.config import COMPLETION_MODEL, MODEL_PROVIDER
from text_rag.logger import get_logger
from dataclasses import dataclass

from text_rag.config import OPENAI_API_KEY

logger = get_logger("text_rag.generator")

BEDROCK_SYSTEM_PROMPT = """
You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer concisely and cite context chunks by id where appropriate.
"""

OPENAI_SYSTEM_PROMPT = (
    "You are a helpful assistant. You MUST only use the information in the 'CONTEXT' "
    "sections below to answer the user's question. If the answer is not contained in "
    "the provided context, say you don't know."
)

@dataclass
class AnswerResult:
    answer: str
    raw_model_response: Dict[str, Any]
    metadata: Dict[str, Any]

def bedrock_generator(question: str, context_chunks: list) -> str:
    client = bedrock_client()
    context_text = "\n\n".join([f"[id={c['id']}] {c['chunk']}" for c in context_chunks])
    prompt = BEDROCK_SYSTEM_PROMPT.format(context=context_text, question=question)
    try:
        resp = client.invoke_model(
            modelId=COMPLETION_MODEL,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": prompt, "maxTokens": 512})
        )
        payload = json.loads(resp['body'].read())
        answer = payload.get('outputText') or payload.get('choices', [{}])[0].get('text')
        logger.info(f"successfully generated answer.")
    except Exception as e:
        answer = "[error] failed to generate answer"
        logger.error(f"failed to generate answer: {e}")
    return answer

def build_messages(question: str, context_chunks: List[str]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": OPENAI_SYSTEM_PROMPT}]
    for i, c in enumerate(context_chunks, start=1):
        messages.append({"role": "user", "content": f"--- CONTEXT [{i}] ---\n{c}"})
    messages.append({"role": "user", "content": f"QUESTION: {question}"})
    return messages

async def openai_generator(
    question: str,
    context: str | List[str],
) -> AnswerResult:
    """
    Given a question and context, produce an answer using gpt-5-nano.
    Args:
        question: user’s question
        context: either a string or list of strings
    """
    chunks = [context] if isinstance(context, str) else list(context)
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    messages = build_messages(question, chunks)

    response = await client.chat.completions.create(
        model=COMPLETION_MODEL,
        messages=messages,
        max_completion_tokens=512,
        temperature=1,
    )

    # Extract assistant text
    assistant_text = response.choices[0].message.content.strip()

    return AnswerResult(
        answer=assistant_text,
        raw_model_response=response.to_dict(),
        metadata={"model": COMPLETION_MODEL, "temperature": 1},
    )


async def invoke_generator_model(question: str, context_chunks: list) -> str :
    if MODEL_PROVIDER == 'openai':
        logger.info("Initializing Open API generator model")
        results = await openai_generator(question, context_chunks)
        return results.answer
    elif MODEL_PROVIDER == 'bedrock':
        logger.info("Initializing Bedrock generator model")
        results = bedrock_generator(question, context_chunks)
        return results
    else:
        raise ValueError(f"Unknown MODEL_PROVIDER: {MODEL_PROVIDER}")