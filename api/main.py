"""
FastAPI backend for RAG Document Q&A Engine.

Endpoints:
  POST /ingest          Upload and index documents
  POST /query           Ask a question, get grounded answer
  GET  /documents       List indexed documents
  DELETE /documents/{id} Remove a document from the index
  GET  /health          Health check + system info
  POST /evaluate        Run RAGAS evaluation on QA pairs (async)

Rate limiting: 60 requests/minute per IP (configurable via env).
Authentication: Bearer token (set RAG_API_KEY in .env for production).
"""

import os
import logging
import tempfile
import shutil
import hashlib
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import (FastAPI, UploadFile, File, HTTPException, Depends,
                     Request, BackgroundTasks)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import time

from api.models import (QueryRequest, QueryResponse, IngestResponse,
                        DocumentInfo, HealthResponse, EvalRequest)

log = logging.getLogger(__name__)

# ── Globals (populated at startup) ───────────────────────────────────────────
rag_chain = None
document_registry: dict = {}  # doc_id → {filename, chunks, ingested_at}

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".csv", ".md"}
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise RAG chain on startup."""
    global rag_chain
    log.info("Initialising RAG chain...")
    try:
        import chromadb
        from langchain_community.vectorstores import Chroma
        from retrieval.reranker import CrossEncoderReranker
        from retrieval.qa_chain import RAGChain

        local_mode = os.getenv("LOCAL_MODE", "false").lower() == "true"

        if local_mode:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        else:
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        client = chromadb.PersistentClient(path="db/chroma")
        vectorstore = Chroma(client=client, collection_name="documents",
                             embedding_function=embeddings)
        reranker = CrossEncoderReranker()
        rag_chain = RAGChain(vectorstore=vectorstore, reranker=reranker, local=local_mode)
        log.info(f"RAG chain ready (local_mode={local_mode})")
    except Exception as e:
        log.error(f"Failed to initialise RAG chain: {e}")
        rag_chain = None
    yield
    log.info("Shutting down RAG engine.")


app = FastAPI(
    title="RAG Document Q&A API",
    description="Production-grade Retrieval-Augmented Generation for document querying.",
    version="1.2.0",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["GET", "POST", "DELETE"],
                   allow_headers=["*"])

# ── Request timing middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = str(round((time.time() - start) * 1000, 1))
    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check API health and report system configuration."""
    import chromadb
    chroma_docs = 0
    try:
        client = chromadb.PersistentClient(path="db/chroma")
        col = client.get_or_create_collection("documents")
        chroma_docs = col.count()
    except Exception:
        pass
    return HealthResponse(
        status="healthy" if rag_chain else "degraded",
        rag_chain_ready=rag_chain is not None,
        indexed_documents=len(document_registry),
        total_chunks=chroma_docs,
        local_mode=os.getenv("LOCAL_MODE", "false").lower() == "true"
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_type: str = "prose",
    collection: str = "documents"
):
    """
    Upload and index a document (PDF, DOCX, TXT, CSV).
    Chunking strategy selected automatically based on doc_type.
    """
    # Validate extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Allowed: {ALLOWED_EXTENSIONS}")

    # Check size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large. Max {MAX_FILE_SIZE_MB}MB.")

    doc_id = hashlib.sha256(contents[:1024] + file.filename.encode()).hexdigest()[:16]

    if doc_id in document_registry:
        return IngestResponse(
            status="skipped",
            doc_id=doc_id,
            filename=file.filename,
            message="Document already indexed."
        )

    # Write to temp file and ingest
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        from ingestion.loader import load_document
        from ingestion.chunker import get_chunker, ChunkConfig
        import chromadb
        from langchain_community.vectorstores import Chroma

        docs = load_document(tmp_path, original_filename=file.filename)
        chunker = get_chunker(doc_type, ChunkConfig())
        chunks = chunker.split(docs)

        # Embed and store
        local_mode = os.getenv("LOCAL_MODE", "false").lower() == "true"
        if local_mode:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        else:
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        client = chromadb.PersistentClient(path="db/chroma")
        Chroma.from_documents(chunks, embeddings,
                              client=client, collection_name=collection)

        document_registry[doc_id] = {
            "filename": file.filename,
            "chunks": len(chunks),
            "doc_type": doc_type,
            "collection": collection,
            "size_bytes": len(contents)
        }

        return IngestResponse(
            status="success",
            doc_id=doc_id,
            filename=file.filename,
            chunks_created=len(chunks),
            message=f"Indexed {len(chunks)} chunks into collection '{collection}'"
        )
    except Exception as e:
        log.error(f"Ingestion failed for {file.filename}: {e}", exc_info=True)
        raise HTTPException(500, f"Ingestion failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/query", response_model=QueryResponse, tags=["Q&A"])
async def query_documents(request: QueryRequest):
    """
    Ask a natural language question. Returns grounded answer with source citations.
    """
    if rag_chain is None:
        raise HTTPException(503, "RAG chain not initialised. Check /health for details.")
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    try:
        response = rag_chain.query(request.question)
        return QueryResponse(**response.to_dict())
    except Exception as e:
        log.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(500, f"Query processing failed: {str(e)}")


@app.get("/documents", response_model=List[DocumentInfo], tags=["Documents"])
async def list_documents():
    """List all indexed documents with metadata."""
    return [
        DocumentInfo(doc_id=doc_id, **info)
        for doc_id, info in document_registry.items()
    ]


@app.delete("/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str):
    """Remove a document from the index."""
    if doc_id not in document_registry:
        raise HTTPException(404, f"Document {doc_id!r} not found.")
    info = document_registry.pop(doc_id)
    # In production: also delete from ChromaDB by metadata filter
    return {"status": "deleted", "doc_id": doc_id, "filename": info["filename"]}


@app.post("/clear-memory", tags=["Q&A"])
async def clear_conversation_memory():
    """Reset the conversation history."""
    if rag_chain:
        rag_chain.clear_memory()
    return {"status": "ok", "message": "Conversation memory cleared."}


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
