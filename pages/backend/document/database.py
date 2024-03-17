import os

from langchain_community.vectorstores import Qdrant
from loguru import logger
from qdrant_client import qdrant_client
from qdrant_client.http.models import Distance, VectorParams

from pages.backend.llm import LLM_EMBEDDINGS

MEMORY_COLLECTION_NAME = "MEMORY"

VECTOR_DB_COLLECTION_NAME = os.getenv("QDRANT_DATA_COLLECTION", "MYDATA_COLLECTION")
QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_HTTP_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6333"))

NATIVE_QDRANT_CLIENT = qdrant_client.QdrantClient(
    host=QDRANT_HOST,
    grpc_port=QDRANT_GRPC_PORT,
    prefer_grpc=True,
)

DATA_CLIENT = Qdrant(
    client=NATIVE_QDRANT_CLIENT, collection_name=VECTOR_DB_COLLECTION_NAME,
    embeddings=LLM_EMBEDDINGS,
)

MEMORY_CLIENT = Qdrant(
    client=NATIVE_QDRANT_CLIENT, collection_name=MEMORY_COLLECTION_NAME,
    embeddings=LLM_EMBEDDINGS,
)


def init_if_needed(collection_name):
    if not NATIVE_QDRANT_CLIENT.collection_exists(collection_name):
        # Dynamics sizing
        partial_embeddings = LLM_EMBEDDINGS.embed_documents("Hello world")
        vector_size = len(partial_embeddings[0])

        NATIVE_QDRANT_CLIENT.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )


init_if_needed(MEMORY_COLLECTION_NAME)
init_if_needed(VECTOR_DB_COLLECTION_NAME)


def clean_collection():
    logger.debug(f"Clean {VECTOR_DB_COLLECTION_NAME}")
    NATIVE_QDRANT_CLIENT.delete_collection(collection_name=VECTOR_DB_COLLECTION_NAME)


def insert_documents_vector(documents):
    # Weird, need collection_name and embeddings in argument and when inserting
    # Without embedding, there is an error
    # Without collection_name, documents aren't store in the good place.
    DATA_CLIENT.from_documents(documents=documents,
                               embedding=LLM_EMBEDDINGS,
                               collection_name=VECTOR_DB_COLLECTION_NAME)
    logger.debug(f"{len(documents)}")


def get_data_retriever():
    return DATA_CLIENT.as_retriever()


def get_memory_retriever():
    return MEMORY_CLIENT.as_retriever()

# hack
