import streamlit as st

from pages.backend.document.database import clean_collection
from pages.backend.document.web_loader import load_web_documents
from pages.backend.rag.default_prompt import FR, \
    EN
from pages.backend.rag.rag import build_rag
from pages.ui.chat_ui import RagChatContainer


def dataloader_config():
    url = st.text_input("Url",
                        value="https://fr.wikipedia.org/wiki/Gravit%C3%A9_quantique#:~:text=La%20gravit%C3%A9%20quantique%20est%20une,quantique%20et%20la%20relativit%C3%A9%20g%C3%A9n%C3%A9rale.")
    if st.button("Extract web document"):
        load_web_documents(url)
        st.write("done")
    if st.button("Clean collection"):
        clean_collection()


def prompts_config(lang):
    if lang == 'FR':
        lang_prompt = FR
    else:
        lang_prompt = EN

    query_prompt_template = st.text_area("Query Prompt Template", lang_prompt.DEFAULT_QUERY_PROMPT)
    answer_prompt_template = st.text_area("Answer Prompt Template", lang_prompt.DEFAULT_ANSWER_PROMPT)
    document_prompt_template = st.text_area("Document Prompt Template", lang_prompt.DEFAULT_DOCUMENT_PROMPT)

    return {
        "query_prompt_template": query_prompt_template,
        "answer_prompt_template": answer_prompt_template,
        "document_prompt_template": document_prompt_template
    }


def main():
    if "rag_understanding" not in st.session_state:
        st.session_state.rag_understanding = []
    chat_column, config_column = st.columns([0.5, 0.5])

    with st.container():
        with config_column:
            st.write('## ⚙️ Configuration')
            with st.expander("📑 Web Loader'"):
                dataloader_config()
            with st.expander('🗣️ Prompts Config'):
                lang = st.selectbox(
                    'Default Prompt Langage ?',
                    ('EN', 'FR'))

                prompts = prompts_config(lang)
    with chat_column:
        st.write('## 🤖 Assistant')
        llm_chain = build_rag(
            query_prompt_template=prompts["query_prompt_template"],
            answer_prompt_template=prompts["answer_prompt_template"],
            document_prompt_template=prompts["document_prompt_template"],
        )
        RagChatContainer(llm_chain).build_ui()
    with st.container():
        st.write('## 🔬 Understanding')
        st.write('### Extracted documents : ')
        st.write(st.session_state.rag_understanding)
        st.write('### Buffer Memory ')
        if "chat_memory" in st.session_state:
            for entity, summary in st.session_state.chat_memory.store.items():
                # st.sidebar.write(f"{entity}: {summary}")
                st.sidebar.write(f"Entity: {entity}")
                st.sidebar.write(f"{summary}")


main()
