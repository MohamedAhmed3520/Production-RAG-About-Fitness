from __future__ import annotations

import threading
from pathlib import Path
from typing import List

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from app.loaders.pdf_loader import PDFKnowledgeLoader
from app.vectorstore.qdrant_store import QdrantVectorStoreService
from langchain_core.documents import Document


class PDFIndexingService:
    """Index PDFs from the local directory and watch for changes with Watchdog."""

    def __init__(self, pdfs_dir: Path, vector_store: QdrantVectorStoreService) -> None:
        self.pdfs_dir = pdfs_dir
        self.vector_store = vector_store
        self.loader = PDFKnowledgeLoader(pdfs_dir)
        self._lock = threading.Lock()
        self._observer: Observer | None = None

    def index_all(self) -> List[Document]:
        with self._lock:
            documents = self.loader.load_all()
            if documents:
                self.vector_store.add_documents(documents)
            return documents

    def start_watching(self) -> None:
        if self._observer is not None:
            return
        self._observer = Observer()
        self._observer.schedule(_PDFHandler(self), str(self.pdfs_dir), recursive=True)
        self._observer.start()

    def stop_watching(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=2)
            self._observer = None


class _PDFHandler(FileSystemEventHandler):
    def __init__(self, service: PDFIndexingService) -> None:
        self.service = service

    def on_created(self, event) -> None:
        if event.is_directory:
            return
        self.service.index_all()

    def on_modified(self, event) -> None:
        if event.is_directory:
            return
        self.service.index_all()

    def on_deleted(self, event) -> None:
        if event.is_directory:
            return
        self.service.index_all()
