from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
from loguru import logger

from pages.langchain.llm import llm


def get_conversation_memory_chat():
    template = """Tu es l'assistant d'un enfant. 
    C'est une petite fille qui s'appelle Violette.
 

    {chat_history}
    {human_input}
    """

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"], template=template
    )
    logger.debug(prompt)
    memory = ConversationBufferMemory(memory_key="chat_history")

    return LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )



