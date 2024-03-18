import streamlit as st

from pages.backend import CONVERSATION_CHATBOT
from pages.ui.chat_ui import ConversationalChatContainer


def main():
    st.title("🤖 Simple chatbot")
    ConversationalChatContainer(CONVERSATION_CHATBOT).build_ui()


main()
