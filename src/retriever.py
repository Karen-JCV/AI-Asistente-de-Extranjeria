"""Retriever de recuperación semántica sobre la base vectorial."""
from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever

from . import config


def get_retriever(vectorstore: Chroma) -> BaseRetriever:
    """Construye un retriever por similitud sobre la base vectorial dada."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.RETRIEVER_K},
    )
