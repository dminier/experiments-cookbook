import time
from abc import abstractmethod, ABC

import streamlit as st

from pages.backend.rag import RetrievalAugmentedGenerationChatBot


class ChatContainer(ABC):

    @abstractmethod
    def _response_generator(self, prompt):
        pass

    def __init__(self, llm_chain):
        self.llm_chain = llm_chain

    def build_ui(self):
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
                    response = st.write_stream(self._response_generator(prompt))
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})


class ConversationalChatContainer(ChatContainer):

    def __init__(self, llm_chain):
        super().__init__(llm_chain)

    def _response_generator(self, prompt):
        resp = self.llm_chain.run(prompt)

        for word in resp.split():
            yield word + " "
            time.sleep(0.05)


class RagChatContainer(ChatContainer):
    def __init__(self, bot: RetrievalAugmentedGenerationChatBot):
        self.bot: RetrievalAugmentedGenerationChatBot = bot
        super().__init__(bot.chain)

    def _response_generator(self, prompt):
        resp = self.llm_chain.invoke({"question": prompt})
        answer = resp["answer"]
        self.bot.memory.save_context(
            {"question": prompt},
            {"answer": answer.content}
        )

        for word in answer.content.split():
            yield word + " "
            time.sleep(0.05)
        st.session_state.rag_understanding = []
        for doc in resp["docs"]:
            st.session_state.rag_understanding.append(doc)
