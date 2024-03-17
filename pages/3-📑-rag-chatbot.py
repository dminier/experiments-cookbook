import streamlit as st

from pages.chat.ui import chat_container
from pages.langchain.document.loader.web_loader import load_documents
from pages.langchain.rag import RAG_CHAIN

st.subheader("LLM")
st.write("Load LLM Mistral ...")


def configuration_container():
    st.subheader('Web document loader')
    url = st.text_input("Url",
                        value="https://fr.wikipedia.org/wiki/Gravit%C3%A9_quantique#:~:text=La%20gravit%C3%A9%20quantique%20est%20une,quantique%20et%20la%20relativit%C3%A9%20g%C3%A9n%C3%A9rale.")
    if st.button("Extract web document"):
        load_documents(url)
        st.write("done")


def main():
    st.title("Assistant")

    configuration_container()

    chat_container(RAG_CHAIN)


main()
