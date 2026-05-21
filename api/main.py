"""FastAPI backend for RAG Document Q&A Engine."""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile, os, shutil
from ingestion.pipeline import ingest
from retrieval.qa_chain import build_qa_chain, query

app = FastAPI(title="RAG Document Q&A API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Build chain at startup
qa_chain = build_qa_chain()

class QueryRequest(BaseModel):
    question: str
    collection: str = "documents"

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Upload and index a document (PDF, DOCX, TXT)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        stats = ingest(tmp_path)
        return {"status": "success", **stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

@app.post("/query")
async def ask_question(request: QueryRequest):
    """Ask a question. Returns answer + source citations."""
    try:
        result = query(qa_chain, request.question)
        return {"status": "success", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
