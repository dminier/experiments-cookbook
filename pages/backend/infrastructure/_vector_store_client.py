from langchain_community.vectorstores import Qdrant
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from loguru import logger
from qdrant_client import qdrant_client
from qdrant_client.http.models import Distance, VectorParams


class VectorStoreClient:
    def __init__(self, host: str, port: int):
        logger.debug(f"Build Qdrant client on GRPC {host}:{port}")
        self._native_qdrant_client = qdrant_client.QdrantClient(
            host=host,
            grpc_port=port,
            prefer_grpc=True,
        )

    def build(self, collection_name: str, llm_embeddings: Embeddings, init_if_needed: bool = True) -> VectorStore:
        logger.debug(f"Build Langchain vector store {collection_name}")
        if init_if_needed:
            self._init_if_needed(collection_name, llm_embeddings)

        return Qdrant(
            client=self._native_qdrant_client, collection_name=collection_name,
            embeddings=llm_embeddings,
        )

    def _init_if_needed(self, collection_name: str, llm_embeddings: Embeddings):
        if not self._native_qdrant_client.collection_exists(collection_name):
            logger.debug(f"Init Qdrant VectorStore {collection_name}")
            # Dynamics sizing
            partial_embeddings = llm_embeddings.embed_documents(["Hello world"])
            vector_size = len(partial_embeddings[0])
            self._native_qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

    def clean_collection(self, collection_name: str):
        logger.debug(f"Clean Qdrant VectorStore {collection_name}")
        self._native_qdrant_client.delete_collection(collection_name=collection_name)
