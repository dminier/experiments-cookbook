from langchain_core.embeddings import Embeddings

from pages.backend.infrastructure._loader import DocumentLoaderClient
from pages.backend.infrastructure._vector_store_client import VectorStoreClient


class MemoryLoaderClient(DocumentLoaderClient):
    def __init__(self, vector_store_client: VectorStoreClient, collection_name: str, llm_embeddings: Embeddings):
        super().__init__(vector_store_client, collection_name, llm_embeddings)
