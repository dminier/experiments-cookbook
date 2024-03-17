from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from pages.backend.document.database import insert_documents_vector


def load_web_documents(url: str):
    logger.debug(f"load {url}")
    # todo async
    loader = WebBaseLoader(
        web_paths=(url,)
    )
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(data)

    insert_documents_vector(documents)
