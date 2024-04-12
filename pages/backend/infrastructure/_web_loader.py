from langchain_community.document_loaders import WebBaseLoader
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from pages.backend.infrastructure._loader import DocumentLoaderClient
from pages.backend.infrastructure._vector_store_client import VectorStoreClient


class WebDocumentLoaderClient(DocumentLoaderClient):

    def __init__(self, vector_store_client: VectorStoreClient, collection_name: str, llm_embeddings: Embeddings):
        super().__init__(vector_store_client, collection_name, llm_embeddings)

    def load_documents(self, url: str):
        logger.debug(f"load {url}")
        loader = WebBaseLoader(
            web_paths=(url,)
        )
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(data)
        self._vector_store.from_documents(
            documents,
            self._llm_embeddings
        )
