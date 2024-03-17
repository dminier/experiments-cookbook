from langchain_community.vectorstores import Qdrant
from loguru import logger

from pages.langchain.document.vector_conf import QDRANT_URL, VECTOR_DB_COLLECTION
from pages.langchain.llm import llm_embeddings


def process_document(documents):
    try:
        Qdrant.from_documents(
            documents=documents,
            embedding=llm_embeddings,
            url=QDRANT_URL,
            prefer_grpc=True,
            collection_name=VECTOR_DB_COLLECTION,
        )
    except Exception as e:
        logger.exception(f"Error when parsing {documents}")
