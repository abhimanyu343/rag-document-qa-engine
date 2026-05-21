"""
Document ingestion pipeline for RAG engine.
Supports PDF, DOCX, TXT. Chunks, embeds and stores in ChromaDB.
"""
import os
from pathlib import Path
from typing import List, Optional
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_PATH = "db/chroma"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def get_embeddings(local: bool = False):
    if local:
        return HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    return OpenAIEmbeddings(model="text-embedding-3-small")


def load_document(file_path: str) -> List:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path)
    return loader.load()


def chunk_documents(docs: List, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(docs)


def ingest(file_path: str, local_embeddings: bool = False, collection_name: str = "documents") -> dict:
    """
    Full ingestion pipeline: load → chunk → embed → store.
    Returns stats dict with chunk count and document name.
    """
    print(f"Loading: {file_path}")
    docs = load_document(file_path)
    chunks = chunk_documents(docs)
    
    embeddings = get_embeddings(local=local_embeddings)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=collection_name)
    
    texts = [c.page_content for c in chunks]
    metadatas = [{"source": file_path, "page": c.metadata.get("page", 0)} for c in chunks]
    ids = [f"{Path(file_path).stem}_{i}" for i in range(len(chunks))]
    
    emb_vectors = embeddings.embed_documents(texts)
    collection.add(documents=texts, embeddings=emb_vectors, metadatas=metadatas, ids=ids)
    
    print(f"Indexed {len(chunks)} chunks from {Path(file_path).name}")
    return {"file": Path(file_path).name, "chunks": len(chunks), "collection": collection_name}
