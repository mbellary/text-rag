from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from text_rag.worker import handle_query
from text_rag.logger import get_logger
import uvicorn

logger = get_logger("text_rag.api")
app = FastAPI(title="text-rag - RAG Playground", version="0.1.0")


class GenerateRequest(BaseModel):
    query: str
    k: int | None = None
    n: int | None = None
    reflection: bool = False


@app.get("/healthz")
async def healthz():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/generate")
async def generate(req: GenerateRequest):
    try:
        logger.info("generate_request", query=req.query)
        resp = await handle_query(
            req.query,
            k=req.k,
            n=req.n,
            do_reflection=req.reflection,
        )
        return resp
    except Exception as e:
        logger.error("generate_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    uvicorn.run("text_rag.api:app", host="0.0.0.0", port=8080, reload=True)
