"""
Full RAG QA chain with conversation memory, query rewriting, hybrid retrieval, and re-ranking.

The chain follows this sequence per query:
1. Rewrite query for clarity (optional, LLM-based)
2. Parallel retrieval: semantic (MMR) + BM25 keyword
3. RRF fusion of results
4. Cross-encoder re-ranking
5. LLM generation with grounded context
6. Return answer + source citations + confidence indicators
"""

import logging
import time
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass, field
from langchain.schema import Document
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

log = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Structured response from the RAG chain."""
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    rewritten_query: Optional[str] = None
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    num_chunks_retrieved: int = 0
    num_chunks_after_rerank: int = 0
    confidence: str = "medium"  # low / medium / high

    def to_dict(self) -> Dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "rewritten_query": self.rewritten_query,
            "performance": {
                "retrieval_ms": round(self.retrieval_time_ms, 1),
                "generation_ms": round(self.generation_time_ms, 1),
                "chunks_retrieved": self.num_chunks_retrieved,
                "chunks_after_rerank": self.num_chunks_after_rerank,
            },
            "confidence": self.confidence
        }


SYSTEM_PROMPT = """You are a precise document analysis assistant. Your job is to answer questions
based strictly on the provided context documents.

Rules:
1. Answer ONLY from the provided context. Do not use external knowledge.
2. If the context doesn't contain enough information, say so explicitly.
3. Cite your sources using [Source: filename, Page: N] notation inline.
4. Be concise but complete — cover all relevant information from the context.
5. If the question has multiple parts, address each one separately.
6. For numerical data, quote exact figures from the documents.

Context Documents:
{context}
"""

USER_PROMPT = """Question: {question}

Please provide a thorough answer based on the context documents above."""


class RAGChain:
    """
    Full RAG QA chain combining hybrid retrieval, re-ranking, and LLM generation.
    Maintains conversation history for multi-turn dialogue.
    """

    def __init__(
        self,
        vectorstore,
        llm=None,
        reranker=None,
        use_query_rewriter: bool = True,
        top_k_retrieve: int = 10,
        top_k_rerank: int = 5,
        memory_window: int = 6,
        local: bool = False
    ):
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.use_query_rewriter = use_query_rewriter
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
        self.local = local
        self.memory = ConversationBufferWindowMemory(
            k=memory_window, return_messages=True, memory_key="history"
        )
        self._llm = llm
        self._query_rewriter_llm = None

    @property
    def llm(self):
        if self._llm is None:
            if self.local:
                from langchain_community.llms import Ollama
                self._llm = Ollama(model="llama3", temperature=0.1)
            else:
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(model="gpt-4o", temperature=0.1, streaming=False)
        return self._llm

    def _rewrite_query(self, query: str) -> str:
        """Expand a terse or ambiguous query into a more complete question."""
        if not self.use_query_rewriter:
            return query
        try:
            if self._query_rewriter_llm is None:
                if self.local:
                    from langchain_community.llms import Ollama
                    self._query_rewriter_llm = Ollama(model="llama3", temperature=0)
                else:
                    from langchain_openai import ChatOpenAI
                    self._query_rewriter_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

            prompt = (
                "Rewrite the following question to be more explicit and self-contained, "
                "without changing its meaning. Output ONLY the rewritten question, nothing else.\n\n"
                f"Original: {query}\nRewritten:"
            )
            rewritten = self._query_rewriter_llm.invoke(prompt)
            result = rewritten.content if hasattr(rewritten, "content") else str(rewritten)
            result = result.strip().strip('"')
            log.debug(f"Query rewritten: {query!r} → {result!r}")
            return result
        except Exception as e:
            log.warning(f"Query rewriting failed: {e}. Using original.")
            return query

    def _retrieve(self, query: str) -> List[Document]:
        """Hybrid retrieval: semantic MMR + BM25, then RRF fusion."""
        from retrieval.reranker import reciprocal_rank_fusion

        # Semantic retrieval (MMR for diversity)
        semantic_docs = self.vectorstore.similarity_search(
            query, k=self.top_k_retrieve, fetch_k=self.top_k_retrieve * 3
        )

        # Try BM25 if available
        bm25_docs = []
        if hasattr(self.vectorstore, "bm25_retriever"):
            try:
                bm25_docs = self.vectorstore.bm25_retriever.get_relevant_documents(query)
            except Exception:
                pass

        # Fuse results
        if bm25_docs:
            merged = reciprocal_rank_fusion([semantic_docs, bm25_docs])
        else:
            merged = semantic_docs

        return merged[:self.top_k_retrieve]

    def _format_context(self, docs: List[Document]) -> str:
        """Format retrieved documents into context string with citations."""
        parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            source = meta.get("source", "Unknown").split("/")[-1]
            page = meta.get("page", "?")
            section = meta.get("section", "")
            header = f"[Document {i} | Source: {source} | Page: {page}"
            if section:
                header += f" | Section: {section}"
            header += "]"
            parts.append(f"{header}\n{doc.page_content.strip()}")
        return "\n\n---\n\n".join(parts)

    def _extract_sources(self, docs: List[Document]) -> List[Dict]:
        """Extract structured source citations from retrieved documents."""
        seen = set()
        sources = []
        for doc in docs:
            meta = doc.metadata
            source_file = meta.get("source", "Unknown").split("/")[-1]
            page = meta.get("page", "?")
            key = (source_file, page)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "file": source_file,
                    "page": page,
                    "section": meta.get("section", ""),
                    "excerpt": doc.page_content[:200].strip() + "..."
                })
        return sources

    def _assess_confidence(self, docs: List[Document], answer: str) -> str:
        """Simple heuristic confidence scoring based on retrieval quality."""
        if len(docs) == 0:
            return "low"
        avg_length = sum(len(d.page_content) for d in docs) / len(docs)
        # If answer is very short vs context, likely low confidence
        if len(answer) < 50:
            return "low"
        if len(docs) >= 4 and avg_length > 200:
            return "high"
        return "medium"

    def query(self, question: str) -> RAGResponse:
        """
        Full RAG pipeline: rewrite → retrieve → rerank → generate → cite.

        Args:
            question: Natural language question

        Returns:
            RAGResponse with answer, sources, and performance metrics
        """
        # Step 1: Query rewriting
        rewritten = self._rewrite_query(question)

        # Step 2: Hybrid retrieval
        t_ret = time.time()
        docs = self._retrieve(rewritten)
        retrieval_ms = (time.time() - t_ret) * 1000
        n_retrieved = len(docs)

        # Step 3: Re-ranking
        if self.reranker and docs:
            reranked = self.reranker.rerank(rewritten, docs, top_k=self.top_k_rerank)
            docs = [doc for doc, _ in reranked]
        else:
            docs = docs[:self.top_k_rerank]
        n_reranked = len(docs)

        # Step 4: Build context and generate
        context = self._format_context(docs)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("human", USER_PROMPT),
        ])
        chain = prompt | self.llm

        t_gen = time.time()
        history = self.memory.load_memory_variables({}).get("history", [])
        response = chain.invoke({
            "context": context,
            "question": rewritten,
            "history": history
        })
        generation_ms = (time.time() - t_gen) * 1000

        answer = response.content if hasattr(response, "content") else str(response)

        # Update memory
        self.memory.save_context({"input": question}, {"output": answer})

        return RAGResponse(
            answer=answer,
            sources=self._extract_sources(docs),
            rewritten_query=rewritten if rewritten != question else None,
            retrieval_time_ms=retrieval_ms,
            generation_time_ms=generation_ms,
            num_chunks_retrieved=n_retrieved,
            num_chunks_after_rerank=n_reranked,
            confidence=self._assess_confidence(docs, answer)
        )

    def clear_memory(self) -> None:
        """Reset conversation history."""
        self.memory.clear()
        log.info("Conversation memory cleared")
