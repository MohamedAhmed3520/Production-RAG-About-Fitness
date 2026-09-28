from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = """You are an expert fitness coach and sports nutrition specialist. Use the retrieved context from the knowledge base whenever possible. If the context contains relevant facts, synthesize a practical answer from them directly and do not refuse just because the document set is narrow. If the context does not contain a matching fact, say so briefly and avoid hallucination. Keep answers concise, practical, and evidence-based."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
