"""Modelo de embeddings de Gemini utilizado para indexar y consultar la base vectorial."""
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from . import config


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Instancia el modelo de embeddings de Gemini configurado para el proyecto."""
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GEMINI_API_KEY,
    )
