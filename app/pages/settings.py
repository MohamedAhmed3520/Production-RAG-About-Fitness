import streamlit as st

st.title("Settings")
st.caption("Configure the assistant behavior")
st.text_input("OpenAI model", value="gpt-4o-mini", key="model_setting")
st.text_input("Qdrant collection", value="fitness_nutrition", key="collection_setting")
