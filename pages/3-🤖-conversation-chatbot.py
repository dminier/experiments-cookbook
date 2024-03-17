import streamlit as st

from pages.ui.chat_ui import ConversationalChatContainer
from pages.backend.conversational import get_conversation_memory_chat


def main():
    st.title("🤖 Simple chatbot")
    ConversationalChatContainer(get_conversation_memory_chat()).build_ui()


main()
