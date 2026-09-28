from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import List

from rag.loaders.pdf_loader import DocumentChunk, PDFLoader
from rag.retrievers.hybrid_retriever import HybridRetriever
from rag.vectorstore.qdrant_store import QdrantVectorStore


class PDFIndexer:
    """Watch the local PDFs directory and index new or changed PDFs."""

    def __init__(self, pdfs_dir: Path, vector_store: QdrantVectorStore, retriever: HybridRetriever) -> None:
        self.pdfs_dir = pdfs_dir
        self.vector_store = vector_store
        self.retriever = retriever
        self.loader = PDFLoader()
        self._index_lock = threading.Lock()
        self._indexed_files: dict[str, int] = {}

    def index_all(self) -> List[DocumentChunk]:
        if not self.pdfs_dir.exists():
            self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        pdf_paths = sorted(self.pdfs_dir.glob("*.pdf"))
        docs: List[DocumentChunk] = []
        changed_files: List[Path] = []
        for pdf_path in pdf_paths:
            mtime = pdf_path.stat().st_mtime
            last_seen = self._indexed_files.get(str(pdf_path))
            if last_seen != mtime:
                changed_files.append(pdf_path)
                self._indexed_files[str(pdf_path)] = mtime
        for pdf_path in pdf_paths:
            if pdf_path in changed_files or str(pdf_path) not in self._indexed_files:
                docs.extend(self.loader.load_pdf(pdf_path))
        if docs:
            all_docs = []
            for pdf_path in pdf_paths:
                all_docs.extend(self.loader.load_pdf(pdf_path))
            self.retriever.index_documents(all_docs)
        return docs

    def watch_and_index(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._monitor())

    async def _monitor(self) -> None:
        while True:
            await asyncio.sleep(5)
            self.index_all()
