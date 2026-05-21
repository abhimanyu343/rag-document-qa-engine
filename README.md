# RAG Document Q&A Engine

> Production-ready Retrieval-Augmented Generation pipeline — upload PDFs, ask questions, get grounded answers with source citations.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![LangChain](https://img.shields.io/badge/LangChain-0.2-green) ![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-purple) ![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal)

## Overview

A full-stack RAG system built to demonstrate practical LLM engineering. Ingests PDF/DOCX documents, chunks and embeds them into ChromaDB, then answers natural language queries with citations pointing to exact source pages.

**Real-world use case:** Built a version of this for an SME client to query their 200+ internal SOPs and contracts — reduced research time from hours to seconds.

## Architecture

```
Documents (PDF/DOCX)
      │
      ▼
 Text Extraction (PyMuPDF)
      │
      ▼
 Chunking (RecursiveCharacterTextSplitter, 512 tokens, 50 overlap)
      │
      ▼
 Embedding (OpenAI text-embedding-3-small / local BAAI/bge-base)
      │
      ▼
 Vector Store (ChromaDB persistent)
      │
      ▼
 Query → Retriever (MMR, top-k=5) → LLM (GPT-4o / Ollama) → Answer + Sources
```

## Features

- Multi-document ingestion (PDF, DOCX, TXT, CSV)
- Hybrid retrieval: semantic + BM25 keyword search
- Source citation with page numbers
- Conversation memory (multi-turn Q&A)
- REST API via FastAPI
- Streamlit UI for non-technical users
- Supports local LLMs via Ollama (no API key needed)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Document parsing | PyMuPDF, python-docx |
| Embeddings | OpenAI / HuggingFace BAAI/bge |
| Vector DB | ChromaDB (persistent) |
| LLM orchestration | LangChain |
| LLM backend | GPT-4o / Ollama (Llama3) |
| API | FastAPI |
| UI | Streamlit |

## Quick Start

```bash
git clone https://github.com/abhimanyu343/rag-document-qa-engine
cd rag-document-qa-engine
pip install -r requirements.txt

# Option A: OpenAI
export OPENAI_API_KEY=your_key_here
python api/main.py

# Option B: Local (no API key)
ollama pull llama3
python api/main.py --local

# UI
streamlit run ui/app.py
```

## API Endpoints

```
POST /ingest          Upload and index documents
POST /query           Ask a question, get answer + sources
GET  /documents       List indexed documents
DELETE /documents/{id} Remove a document
```

---
*[LinkedIn](https://linkedin.com/in/abhimanyusarda343) · Part of my AI Engineering portfolio*
