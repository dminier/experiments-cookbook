import time

import streamlit as st
from langchain_core.runnables.base import RunnableSequence


def chat_container(llm_chain):
    def response_generator(prompt):
        if isinstance(llm_chain, RunnableSequence):
            resp = llm_chain.invoke({"question": prompt})
            for word in resp["answer"].content.split():
                yield word + " "
                time.sleep(0.05)
            for doc in resp["docs"]:
                st.write(doc)
        else:
            resp = llm_chain.run(prompt)

            for word in resp.split():
                yield word + " "
                time.sleep(0.05)

    with st.container(height=600, border=2):
        st.subheader('Chat')
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Accept user input
        if prompt := st.chat_input("What is up?"):
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                response = st.write_stream(response_generator(prompt))
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
