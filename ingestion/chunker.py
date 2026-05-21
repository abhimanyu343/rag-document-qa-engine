"""
Smart document chunking strategies.

Strategy selection logic:
- PDFs with many pages → StructuredChunker (respects headings/sections)
- Plain text / reports → RecursiveChunker (sentence-boundary aware)
- CSVs / tables → TabularChunker (row-level chunks with schema context)

All strategies:
- Preserve metadata (source, page, section)
- Avoid orphan sentences at boundaries
- Output LangChain Document objects
"""

import re
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    SentenceTransformersTokenTextSplitter
)

log = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for chunking behaviour."""
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 80         # Discard chunks smaller than this
    max_chunks_per_doc: int = 5000   # Safety cap
    add_start_index: bool = True
    strip_whitespace: bool = True
    separators: List[str] = field(default_factory=lambda: [
        "\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""
    ])


class RecursiveChunker:
    """
    General-purpose chunker using recursive character splitting.
    Respects sentence boundaries by using punctuation as preferred split points.
    Best for: prose documents, reports, emails, SOPs.
    """

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            add_start_index=self.config.add_start_index,
            strip_whitespace=self.config.strip_whitespace,
            length_function=len,
        )

    def split(self, docs: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(docs)
        # Filter undersized chunks (likely headers or page numbers)
        chunks = [c for c in chunks if len(c.page_content.strip()) >= self.config.min_chunk_size]
        log.info(f"RecursiveChunker: {len(docs)} docs → {len(chunks)} chunks")
        return chunks[:self.config.max_chunks_per_doc]


class StructuredChunker:
    """
    Section-aware chunker for structured documents (reports with headers, legal docs).
    Splits on markdown-style headers first, then recursively on content.
    Preserves section context in metadata.
    Best for: annual reports, structured PDFs, policy documents.
    """

    HEADER_PATTERNS = [
        ("##", "section"),
        ("###", "subsection"),
    ]

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.HEADER_PATTERNS,
            strip_headers=False
        )
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
        )

    def _detect_and_inject_headers(self, text: str) -> str:
        """
        Convert common heading styles to markdown-style headers.
        Handles ALL-CAPS headings, numbered sections (1.2.3), and bold patterns.
        """
        # ALL CAPS headings
        text = re.sub(r"^([A-Z][A-Z\s]{3,})$", r"## \1", text, flags=re.MULTILINE)
        # Numbered sections like "1. " or "1.1 "
        text = re.sub(r"^(\d+\.\d*\s+[A-Z].{5,60})$", r"### \1", text, flags=re.MULTILINE)
        return text

    def split(self, docs: List[Document]) -> List[Document]:
        all_chunks = []
        for doc in docs:
            text = self._detect_and_inject_headers(doc.page_content)
            try:
                md_splits = self.md_splitter.split_text(text)
            except Exception:
                md_splits = [Document(page_content=text, metadata=doc.metadata)]

            for split in md_splits:
                if isinstance(split, dict):
                    split_doc = Document(
                        page_content=split.get("content", ""),
                        metadata={**doc.metadata, **{k: v for k, v in split.items() if k != "content"}}
                    )
                else:
                    split_doc = Document(
                        page_content=split.page_content,
                        metadata={**doc.metadata, **split.metadata}
                    )
                sub_chunks = self.recursive_splitter.split_documents([split_doc])
                all_chunks.extend(sub_chunks)

        all_chunks = [c for c in all_chunks if len(c.page_content.strip()) >= self.config.min_chunk_size]
        log.info(f"StructuredChunker: {len(docs)} docs → {len(all_chunks)} chunks")
        return all_chunks[:self.config.max_chunks_per_doc]


class TabularChunker:
    """
    Row-level chunker for CSV and tabular data.
    Each row becomes a chunk prefixed with column schema context.
    Best for: product catalogues, financial tables, CRM exports.
    """

    def __init__(self, rows_per_chunk: int = 5):
        self.rows_per_chunk = rows_per_chunk

    def split(self, docs: List[Document]) -> List[Document]:
        """
        Expects docs where page_content is raw CSV text.
        Groups rows into chunks, prepending header to each.
        """
        import csv, io
        all_chunks = []
        for doc in docs:
            reader = csv.reader(io.StringIO(doc.page_content))
            rows = list(reader)
            if not rows:
                continue
            header = rows[0]
            data_rows = rows[1:]
            schema_prefix = f"Table columns: {', '.join(header)}\n"

            for i in range(0, len(data_rows), self.rows_per_chunk):
                batch = data_rows[i:i + self.rows_per_chunk]
                content = schema_prefix + "\n".join(
                    ", ".join(f"{col}: {val}" for col, val in zip(header, row))
                    for row in batch
                )
                all_chunks.append(Document(
                    page_content=content,
                    metadata={
                        **doc.metadata,
                        "chunk_type": "tabular",
                        "row_start": i + 1,
                        "row_end": i + len(batch)
                    }
                ))
        log.info(f"TabularChunker: {len(docs)} docs → {len(all_chunks)} chunks")
        return all_chunks


def get_chunker(doc_type: str = "prose", config: Optional[ChunkConfig] = None):
    """Factory function — returns the appropriate chunker for document type."""
    mapping = {
        "prose":      RecursiveChunker,
        "structured": StructuredChunker,
        "tabular":    TabularChunker,
        "report":     StructuredChunker,
        "legal":      StructuredChunker,
        "csv":        TabularChunker,
    }
    cls = mapping.get(doc_type.lower(), RecursiveChunker)
    if cls == TabularChunker:
        return cls()
    return cls(config)
