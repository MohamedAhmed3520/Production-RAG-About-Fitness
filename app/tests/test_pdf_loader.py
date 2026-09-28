from __future__ import annotations

from pathlib import Path

import fitz

from app.loaders.pdf_loader import PDFKnowledgeLoader


def test_loader_load_file_loads_one_exact_pdf_path(tmp_path: Path) -> None:
    pdf_path = tmp_path / "uploaded.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "uploaded file content")
    doc.save(pdf_path)
    doc.close()

    documents = PDFKnowledgeLoader(tmp_path).load_file(pdf_path)

    assert documents, "expected one document from the uploaded PDF file"
    assert any(doc.metadata.get("source") == str(pdf_path) for doc in documents)
    assert any(doc.metadata.get("filename") == "uploaded.pdf" for doc in documents)


def test_loader_discovers_pdfs_in_nested_directories(tmp_path: Path) -> None:
    pdfs_dir = tmp_path / "pdfs"
    nested_dir = pdfs_dir / "nested"
    nested_dir.mkdir(parents=True)

    pdf_path = nested_dir / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "hello from pdf")
    doc.save(pdf_path)
    doc.close()

    documents = PDFKnowledgeLoader(pdfs_dir).load_all()

    assert documents, "expected at least one document from the PDF file"
    assert any(doc.metadata.get("filename") == "sample.pdf" for doc in documents)
    assert any(str(pdf_path) in doc.metadata.get("source", "") for doc in documents)
