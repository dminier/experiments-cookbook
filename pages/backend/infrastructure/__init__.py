import os

from langchain.retrievers import MultiVectorRetriever
from langchain_core.embeddings import Embeddings

from pages.backend.infrastructure._llm import ollama_config
from pages.backend.infrastructure._loader import DocumentLoaderClient
from pages.backend.infrastructure._memory_loader import MemoryLoaderClient
from pages.backend.infrastructure._vector_store_client import VectorStoreClient
from pages.backend.infrastructure._web_loader import WebDocumentLoaderClient

_MYPDF_COLLECTION_NAME = os.getenv("MYPDF_COLLECTION_NAME", "MYPDF_COLLECTION")
_MYWEB_COLLECTION_NAME = os.getenv("MYWEB_COLLECTION_NAME", "MYWEB_COLLECTION")
_RAG_MEMORY_COLLECTION_NAME = os.getenv("RAG_MEMORY_COLLECTION_NAME", "RAG5_MEMORY_COLLECTION")
_CONV_MEMORY_COLLECTION_NAME = os.getenv("CONV_MEMORY_COLLECTION_NAME", "CONV_MEMORY_COLLECTION")

_QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
_QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
_OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "mistral")

LLM, LLM_EMBEDDINGS, LLM_CHAT, LLM_CHAT_STRICT = ollama_config(_OLLAMA_MODEL_NAME)

# ultra naive
_MYPDF_EMBEDDINGS: Embeddings = LLM_EMBEDDINGS
_MYWEB_EMBEDDINGS: Embeddings = LLM_EMBEDDINGS
_MEMORY_EMBEDDINGS: Embeddings = LLM_EMBEDDINGS

_VECTOR_STORE_CLIENT = VectorStoreClient(host=_QDRANT_HOST, port=_QDRANT_GRPC_PORT)

WEB_LOADER = WebDocumentLoaderClient(collection_name=_MYWEB_COLLECTION_NAME,
                                     vector_store_client=_VECTOR_STORE_CLIENT,
                                     llm_embeddings=_MYWEB_EMBEDDINGS)

PDF_LOADER = WebDocumentLoaderClient(collection_name=_MYPDF_COLLECTION_NAME,
                                     vector_store_client=_VECTOR_STORE_CLIENT,
                                     llm_embeddings=_MYWEB_EMBEDDINGS)

RAG_MEMORY_LOADER = MemoryLoaderClient(collection_name=_RAG_MEMORY_COLLECTION_NAME,
                                       vector_store_client=_VECTOR_STORE_CLIENT,
                                       llm_embeddings=_MEMORY_EMBEDDINGS)

CONV_MEMORY_LOADER = MemoryLoaderClient(collection_name=_CONV_MEMORY_COLLECTION_NAME,
                                        vector_store_client=_VECTOR_STORE_CLIENT,
                                        llm_embeddings=_MEMORY_EMBEDDINGS)

