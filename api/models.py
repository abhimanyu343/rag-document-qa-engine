"""
Pydantic models for RAG API request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000,
                          description="Natural language question to answer from documents")
    collection: str = Field("documents", description="ChromaDB collection to query")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to use for generation")
    rewrite_query: bool = Field(True, description="Enable LLM-based query rewriting")

    @validator("question")
    def strip_question(cls, v):
        return v.strip()


class SourceCitation(BaseModel):
    file: str
    page: Any
    section: Optional[str] = None
    excerpt: str


class PerformanceMetrics(BaseModel):
    retrieval_ms: float
    generation_ms: float
    chunks_retrieved: int
    chunks_after_rerank: int


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
    rewritten_query: Optional[str] = None
    performance: PerformanceMetrics
    confidence: str = Field(..., pattern="^(low|medium|high)$")


class IngestResponse(BaseModel):
    status: str
    doc_id: str
    filename: str
    chunks_created: int = 0
    message: str


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    doc_type: str
    collection: str
    size_bytes: int


class HealthResponse(BaseModel):
    status: str
    rag_chain_ready: bool
    indexed_documents: int
    total_chunks: int
    local_mode: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EvalRequest(BaseModel):
    qa_pairs: List[Dict[str, str]] = Field(
        ..., description="List of {question, ground_truth} pairs"
    )
    collection: str = "documents"
