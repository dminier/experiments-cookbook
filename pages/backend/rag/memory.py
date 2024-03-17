# little hack to avoir clearing memory :
from langchain.memory import VectorStoreRetrieverMemory

from pages.backend.document.database import get_memory_retriever


# https://python.langchain.com/docs/modules/memory/types/vectorstore_retriever_memory

def build_memory():
    memory = VectorStoreRetrieverMemory(retriever=get_memory_retriever())

    return memory
