import asyncio

import streamlit as st

from pages.chat.ui import chat_container
from pages.langchain.conversational import get_conversation_memory_chat

from pages.langchain.document.loader.web_loader import load_documents

st.subheader("LLM")
st.write("Load LLM Mistral ...")



def main():
    st.title("Simple chatbot")

    chat_container(get_conversation_memory_chat())


main()
