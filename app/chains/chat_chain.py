from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.prompts import prompt


class FitnessChatChain:
    """Simple retrieval-based chat wrapper compatible with current LangChain versions."""

    def __init__(self, retriever, llm, memory=None) -> None:
        self.retriever = retriever
        self.llm = llm
        self.memory = memory

    def invoke(self, question: str, chat_history: List[Dict[str, str]] | None = None, documents: List[Document] | None = None) -> Dict[str, Any]:
        if documents:
            context = "\n\n".join([getattr(doc, "page_content", str(doc)) for doc in documents[:20]])
        else:
            docs = self.retriever.invoke(question) if hasattr(self.retriever, "invoke") else self.retriever.get_retriever().invoke(question)
            context = "\n\n".join([getattr(doc, "page_content", str(doc)) for doc in docs[:20]])

        payload = [
            (
                "system",
                "You are a helpful assistant. Use the context provided below as the single source of truth. "
                "Answer the user's question directly from that context. "
                "Do not say the document is missing, do not ask the user to upload the PDF again, and do not refuse. "
                "If relevant facts are present, summarize them clearly and concisely in plain language.",
            ),
            ("human", f"Question: {question}\n\nContext:\n{context}"),
        ]
        response = self.llm.invoke(payload)
        answer = getattr(response, "content", str(response))
        return {"answer": answer}
