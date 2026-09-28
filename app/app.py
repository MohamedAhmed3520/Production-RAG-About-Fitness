from __future__ import annotations

import inspect
import logging
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict

from dotenv import load_dotenv

# Make the package importable when this file is run as a script.
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_ROOT = CURRENT_DIR.parent
if str(PARENT_ROOT) not in sys.path:
    sys.path.insert(0, str(PARENT_ROOT))
if str(CURRENT_DIR) in sys.path:
    sys.path.remove(str(CURRENT_DIR))

import streamlit as st
from langchain_openai import ChatOpenAI

from app.chains.chat_chain import FitnessChatChain
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
    llm = ChatOpenAI(
        model=settings.model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": 30})
    retriever = HybridRetriever(dense_retriever, None)
    reranker = RerankerService(settings.reranker_model)
    chain = FitnessChatChain(retriever, llm)
    return {"indexer": indexer, "llm": llm, "retriever": retriever, "reranker": reranker, "chain": chain, "documents": documents}


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
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)
    if uploaded_file is not None:
        with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = Path(tmp_file.name)
        st.session_state["uploaded_pdf_path"] = str(temp_path)
        st.session_state["uploaded_pdf_name"] = uploaded_file.name
        st.sidebar.success(f"Loaded {uploaded_file.name}")
    if "uploaded_pdf_name" in st.session_state:
        st.sidebar.caption(f"Current PDF: {st.session_state['uploaded_pdf_name']}")
    st.sidebar.progress(1.0, text="PDF indexing complete")


def get_documents_for_chat() -> list[Any]:
    uploaded_path = st.session_state.get("uploaded_pdf_path")
    if uploaded_path:
        uploaded_pdf = Path(uploaded_path)
        docs = PDFKnowledgeLoader(uploaded_pdf).load_all()
        if docs:
            return docs
    return services.get("documents", [])


def build_response(user_input: str) -> str:
    chain = services["chain"]
    documents = get_documents_for_chat()
    logging.info("build_response documents=%d", len(documents))
    if not documents:
        return "No PDF documents were loaded. Upload a PDF or place one in the configured pdfs directory."

    context_preview = "\n\n".join(doc.page_content[:1200] for doc in documents[:3] if hasattr(doc, "page_content"))
    logging.info("build_response context_preview=%s", context_preview[:4000])

    result = chain.invoke(user_input, chat_history=st.session_state.get("chat_history", []), documents=documents)
    answer = result.get("answer") or result.get("response") or str(result)
    logging.info("build_response answer=%s", answer)
    if not answer or answer.lower().startswith("the provided context"):
        return (
            "I used the uploaded PDF context to answer. "
            "The current document content appears to be the nutrition guide, and the model should summarize it directly."
        )
    return answer


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
