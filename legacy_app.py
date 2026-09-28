from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from app.config import get_settings
from app.embeddings.embedding_service import EmbeddingService
from app.loaders.pdf_loader import PDFKnowledgeLoader
from app.services.indexing import PDFIndexingService
from app.vectorstore.qdrant_store import QdrantVectorStoreService
from app.retrievers.hybrid import HybridRetriever
from app.rerankers.cross_encoder import RerankerService

load_dotenv()
settings = get_settings()
logging.basicConfig(level=logging.INFO)


@st.cache_resource(show_spinner=False)
def build_services() -> Dict[str, Any]:
    embedding_service = EmbeddingService(settings.embedding_model)
    vector_store = QdrantVectorStoreService("http://localhost:6333", "fitness_nutrition", embedding_service)
    indexer = PDFIndexingService(settings.pdfs_dir, vector_store)
    documents = indexer.index_all()
    if documents:
        vector_store.add_documents(documents)
    llm = init_chat_model(settings.model, model_provider="openai", api_key=settings.openrouter_api_key)
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": 30})
    retriever = HybridRetriever(dense_retriever, None)
    reranker = RerankerService(settings.reranker_model)
    return {"indexer": indexer, "llm": llm, "retriever": retriever, "reranker": reranker}


services = build_services()


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


st.set_page_config(page_title="Fitness & Nutrition Assistant", page_icon="🏋️", layout="wide")
init_session_state()


def render_sidebar() -> None:
    st.sidebar.title("Fitness Assistant")
    st.sidebar.caption("LangChain + Streamlit + RAG")
    if st.sidebar.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.chat_history = []
    st.sidebar.progress(1.0, text="PDF indexing complete")


def build_response(user_input: str) -> str:
    retriever = services["retriever"].get_retriever()
    docs = retriever.invoke(user_input)
    context = "\n\n".join([doc.page_content for doc in docs[:5]])
    return f"Retrieved context:\n{context[:1500]}"


def main() -> None:
    render_sidebar()
    st.title("AI Fitness & Nutrition Assistant")
    st.caption("Evidence-based nutrition and training guidance backed by your local PDF knowledge base")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    user_input = st.chat_input("Ask about workout plans, nutrition, or calculators")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            answer = build_response(user_input)
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
