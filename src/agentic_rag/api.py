"""FastAPI serving layer. Run: uvicorn agentic_rag.api:app --reload"""

from fastapi import FastAPI
from pydantic import BaseModel

from agentic_rag.graph import ask

app = FastAPI(title="Agentic RAG", version="0.1.0")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    retries: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    return AskResponse(**ask(req.question))
