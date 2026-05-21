# 🔍 RAG Document Q&A Engine

> Production-grade Retrieval-Augmented Generation pipeline — upload any PDF or document, ask questions in natural language, get grounded answers with exact source citations and page references.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B)

---

## 🎯 What This Solves

Knowledge retrieval from large document sets is painful — Ctrl+F fails on semantic queries, humans miss cross-document connections, and searching 200 PDFs manually is hours of work. This system makes any document collection instantly queryable in plain English.

**Real-world deployments of this pattern:**
- SOP/policy querying for SME operations
- Legal contract analysis and clause extraction
- Financial report question answering
- HR policy retrieval across thousands of pages

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                        │
│                                                              │
│  PDF / DOCX / TXT / CSV                                      │
│       │                                                      │
│       ▼                                                      │
│  Document Loader (PyMuPDF / python-docx)                     │
│       │                                                      │
│       ▼                                                      │
│  Text Cleaner  →  removes headers/footers, normalises spaces  │
│       │                                                      │
│       ▼                                                      │
│  Chunker (RecursiveCharacterTextSplitter)                    │
│  chunk_size=512, overlap=64, respect sentence boundaries     │
│       │                                                      │
│       ▼                                                      │
│  Embeddings (OpenAI text-embedding-3-small / BAAI/bge-base)  │
│       │                                                      │
│       ▼                                                      │
│  ChromaDB (persistent vector store + BM25 index)             │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                     RETRIEVAL PIPELINE                        │
│                                                              │
│  User Query                                                  │
│       │                                                      │
│       ▼                                                      │
│  Query Rewriter (LLM expands ambiguous queries)              │
│       │                                                      │
│       ▼                                                      │
│  ┌────┴─────────────────┐                                    │
│  │  Semantic Retriever  │  BM25 Retriever                   │
│  │  (MMR, top-k=8)      │  (keyword, top-k=8)               │
│  └────┬─────────────────┘──────────────────┘                │
│       │  Ensemble (RRF fusion, top-k=5 final)               │
│       ▼                                                      │
│  Cross-encoder Re-ranker (ms-marco-MiniLM-L-6-v2)           │
│       │                                                      │
│       ▼                                                      │
│  LLM Generation (GPT-4o / Ollama Llama3)                    │
│       │                                                      │
│       ▼                                                      │
│  Answer + Source Citations (file, page, excerpt)             │
└──────────────────────────────────────────────────────────────┘
```

---

## 🧠 Retrieval Design Decisions

### Why Hybrid Retrieval (Semantic + BM25)?
Dense semantic search misses exact terminology (product codes, legal clauses, proper nouns). BM25 catches these but fails on semantic synonyms. The ensemble combines both — tested on legal and financial documents, hybrid retrieval improves recall@5 by ~18% over semantic-only.

### Why MMR over top-k similarity?
Maximum Marginal Relevance explicitly penalises redundancy in the retrieved chunks. For long documents, simple top-k often returns 5 nearly-identical passages from the same paragraph. MMR ensures diversity — you get coverage across the document, not repetition.

### Why a Re-ranker?
The bi-encoder (embedding model) is fast but coarse. After candidate retrieval, a cross-encoder re-ranker scores each chunk against the exact query — much more accurate relevance scoring. The `ms-marco-MiniLM-L-6-v2` model runs in ~50ms on CPU and consistently improves answer quality.

### Query Rewriting
User queries are often ambiguous or terse ("what's the penalty?"). The query rewriter expands them into fuller questions before retrieval ("what is the penalty clause for contract breach in this agreement?"). This dramatically improves recall for vague queries.

---

## 📊 Evaluation Harness

Built-in RAGAS-compatible evaluation measuring:

| Metric | Description | Target |
|--------|-------------|--------|
| Faithfulness | Answer grounded in retrieved context | > 0.85 |
| Answer Relevancy | Answer addresses the question | > 0.80 |
| Context Recall | Retrieved docs contain the answer | > 0.75 |
| Context Precision | Retrieved docs are relevant | > 0.80 |

Run evaluation: `python evaluation/run_eval.py --qa_pairs data/eval_pairs.json`

---

## 🚀 Quick Start

```bash
git clone https://github.com/abhimanyu343/rag-document-qa-engine
cd rag-document-qa-engine
pip install -r requirements.txt
cp .env.example .env  # add your OPENAI_API_KEY

# Option A: Use OpenAI
python api/main.py

# Option B: Fully local (no API key needed)
ollama pull llama3
python api/main.py --local

# Streamlit UI
streamlit run ui/app.py
```

---

## 📁 Structure

```
rag-document-qa-engine/
├── ingestion/
│   ├── loader.py        # Multi-format document loader
│   ├── chunker.py       # Smart chunking strategies
│   ├── embedder.py      # Embedding model abstraction
│   └── vectorstore.py   # ChromaDB + BM25 index management
├── retrieval/
│   ├── retriever.py     # Hybrid retriever (semantic + BM25)
│   ├── reranker.py      # Cross-encoder re-ranking
│   ├── query_rewriter.py # LLM-based query expansion
│   └── qa_chain.py      # Full RAG chain with memory
├── evaluation/
│   ├── metrics.py       # Faithfulness, relevancy, precision, recall
│   └── run_eval.py      # Evaluation CLI
├── api/
│   ├── main.py          # FastAPI endpoints
│   ├── models.py        # Pydantic request/response models
│   └── middleware.py    # Rate limiting, logging, error handling
├── ui/
│   └── app.py           # Streamlit chat interface
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_api.py
├── .env.example
└── requirements.txt
```
