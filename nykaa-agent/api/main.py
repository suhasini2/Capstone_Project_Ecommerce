"""FastAPI deployment (Tasks 11-12).  Run: uvicorn api.main:app --port 8080"""
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from agent.factory import build_agent
from logging_utils import RequestLogger


class AskRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    query: str = Field(min_length=1, max_length=1000)


class Escalation(BaseModel):
    score: float
    threshold: float
    recommended: bool


class AskResponse(BaseModel):
    trace_id: str
    session_id: str
    query: str
    intent: str
    answer: str
    sources: List[str]
    grounded: bool
    refused: bool
    guardrails: List[str]
    escalation: Optional[Escalation]
    top_score: Optional[float]


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(min_length=1, max_length=32)
    topic: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=20, max_length=2000)


class AddDocumentResponse(BaseModel):
    trace_id: str
    doc_id: str
    chunks_added: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent = build_agent()
    app.state.logger = RequestLogger()
    yield


app = FastAPI(title="Nykaa Support Agent", lifespan=lifespan)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, response: Response):
    trace_id, t0, resp, status = uuid.uuid4().hex, time.perf_counter(), None, 200
    try:
        resp = app.state.agent.ask(req.session_id, req.query, trace_id=trace_id)
        return resp
    except Exception:
        status = 500
        raise HTTPException(status_code=500, detail=f"internal error (trace_id={trace_id})")
    finally:
        response.headers["X-Trace-Id"] = trace_id
        app.state.logger.log(trace_id, "/ask", req.query, session_id=req.session_id, status=status,
                             intent=(resp or {}).get("intent"), refused=(resp or {}).get("refused"),
                             guardrails=(resp or {}).get("guardrails", []),
                             latency_ms=round((time.perf_counter() - t0) * 1000, 1))


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document(req: AddDocumentRequest):
    trace_id, t0, status = uuid.uuid4().hex, time.perf_counter(), 200
    try:
        kb = app.state.agent.sv.kb
        if req.doc_id in kb.docs:
            status = 409
            raise HTTPException(status_code=409, detail=f"doc_id {req.doc_id} already exists")
        added = kb.add_documents([req.model_dump()])
        return {"trace_id": trace_id, "doc_id": req.doc_id, "chunks_added": added}
    finally:
        app.state.logger.log(trace_id, "/add-document", "", doc_id=req.doc_id, status=status,
                             latency_ms=round((time.perf_counter() - t0) * 1000, 1))
