from abc import ABC

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from pages.backend.infrastructure._vector_store_client import VectorStoreClient


class DocumentLoaderClient(ABC):
    def __init__(self, vector_store_client: VectorStoreClient, collection_name: str, llm_embeddings: Embeddings):
        self._vector_store_client = vector_store_client
        self._collection_name = collection_name
        self._llm_embeddings = llm_embeddings
        self._vector_store: VectorStore = self._init_vector_store()

    def _init_vector_store(self):
        return self._vector_store_client.build(self._collection_name, self._llm_embeddings)

    def recreate(self):
        self._vector_store_client.clean_collection(self._collection_name)
        self._vector_store = self._init_vector_store()

    def as_retriever(self):
        return self._vector_store.as_retriever()
