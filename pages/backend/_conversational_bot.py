from langchain import LLMChain, PromptTemplate
from langchain.memory import VectorStoreRetrieverMemory
from loguru import logger

from pages.backend.infrastructure import CONV_MEMORY_LOADER, LLM_CHAT


def get_conversation_memory_chat():
    template = """You are a useful assistant of a children. You must Be Very kind.

chat_history: 
{history}

User: 
{input}
    """

    prompt = PromptTemplate(
        input_variables=["history", "human_input"], template=template
    )
    logger.debug(prompt)
    memory = VectorStoreRetrieverMemory(retriever=CONV_MEMORY_LOADER.as_retriever())

    return LLMChain(
        llm=LLM_CHAT,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )
