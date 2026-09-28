# AI Fitness & Nutrition Assistant

A production-ready Streamlit application for fitness and nutrition Q&A backed by a local PDF knowledge base.

## Features
- Streamlit chat UI with conversation memory
- Automatic indexing of PDFs in the local pdfs folder
- Hybrid retrieval (dense + BM25)
- Reranking with BAAI/bge-reranker-v2-m3
- Calculator tools for BMI, BMR, TDEE, protein, carbs, fats, water, deficits, surplus, and 1RM
- Source citations with document name and page number

## Setup
1. Create a virtual environment and install requirements.
2. Copy .env.example to .env and populate the OpenRouter API key.
3. Add PDFs to the ./pdfs directory.
4. Run `streamlit run app.py`
