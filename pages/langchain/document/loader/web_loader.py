from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from pages.langchain.document.loader.vector_insert import process_document


def load_documents(url: str):
    logger.debug(f"load {url}")
    # todo async
    loader = WebBaseLoader(
        web_paths=(url,)
    )
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(data)

    process_document(documents)
