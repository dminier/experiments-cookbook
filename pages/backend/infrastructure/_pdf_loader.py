from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.embeddings import Embeddings
from loguru import logger

from pages.backend.infrastructure._loader import DocumentLoaderClient
from pages.backend.infrastructure._vector_store_client import VectorStoreClient


class PdfDocumentLoaderClient(DocumentLoaderClient):

    def __init__(self, vector_store_client: VectorStoreClient, collection_name: str, llm_embeddings: Embeddings):
        super().__init__(vector_store_client, collection_name, llm_embeddings)

    def load_documents(self, directory: str):
        logger.debug(f"Import PDF document from directory {directory} to {self._collection_name} VectorStore")
        loader = PyPDFDirectoryLoader(directory)
        docs = loader.load()
        self._vector_store.from_documents(
            docs,
            self._llm_embeddings
        )
