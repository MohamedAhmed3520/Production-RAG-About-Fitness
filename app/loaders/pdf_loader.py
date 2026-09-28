from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


class PDFKnowledgeLoader:
    """Load PDFs from a directory or from a single PDF file path."""

    def __init__(self, pdfs_dir: Path | str) -> None:
        self.pdfs_dir = Path(pdfs_dir)

    def load_file(self, pdf_path: Path | str) -> List[Document]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists() or not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
            return []

        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata.update(
                {
                    "source": str(pdf_path),
                    "filename": pdf_path.name,
                    "page": doc.metadata.get("page", 1),
                }
            )
        return docs

    def load_all(self) -> List[Document]:
        documents: List[Document] = []
        if not self.pdfs_dir.exists():
            self.pdfs_dir.mkdir(parents=True, exist_ok=True)
            return documents

        if self.pdfs_dir.is_file() and self.pdfs_dir.suffix.lower() == ".pdf":
            return self.load_file(self.pdfs_dir)

        if self.pdfs_dir.is_dir():
            pdf_paths = sorted(self.pdfs_dir.rglob("*.pdf"))
            for pdf_path in pdf_paths:
                documents.extend(self.load_file(pdf_path))
        return documents
