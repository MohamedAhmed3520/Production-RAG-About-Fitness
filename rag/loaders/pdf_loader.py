from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz


@dataclass
class DocumentChunk:
    text: str
    metadata: dict


class PDFLoader:
    """Load PDF documents and split them into semantically meaningful chunks."""

    def __init__(self, chunk_size: int = 600, overlap: int = 120) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_pdf(self, pdf_path: Path) -> List[DocumentChunk]:
        doc = fitz.open(pdf_path)
        chunks: List[DocumentChunk] = []
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text("text")
            if not text:
                continue
            blocks = [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]
            for block in blocks:
                if len(block) < 80:
                    continue
                metadata = {
                    "document_name": pdf_path.name,
                    "page_number": page_num + 1,
                    "chapter": self._extract_chapter(block),
                    "section_title": self._extract_section_title(block),
                    "source": str(pdf_path),
                }
                chunks.append(DocumentChunk(text=block, metadata=metadata))
        doc.close()
        return chunks

    def _extract_chapter(self, text: str) -> str:
        heading = re.search(r"^(chapter|chapter\s*\d+|SECTION|SECTION\s*\d+)", text, re.I | re.M)
        if heading:
            return heading.group(0).strip()
        return ""

    def _extract_section_title(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            return lines[0][:120]
        return ""
