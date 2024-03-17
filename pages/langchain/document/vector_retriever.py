import qdrant_client
from langchain_community.vectorstores import Qdrant

from pages.langchain.document.vector_conf import QDRANT_URL, VECTOR_DB_COLLECTION
from pages.langchain.llm import llm_embeddings


def get_qdrant_retriever():
    embeddings = llm_embeddings
    gc = qdrant_client.QdrantClient(
        url=QDRANT_URL,
        prefer_grpc=True)

    qdrant = Qdrant(client=gc,
                    collection_name=VECTOR_DB_COLLECTION,
                    embeddings=embeddings)
    return qdrant.as_retriever()


retriever = get_qdrant_retriever()
