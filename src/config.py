"""Configuración global del proyecto: variables de entorno, modelos y rutas."""
import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "No se ha encontrado la variable de entorno GEMINI_API_KEY. "
        "Definela en un archivo .env (ver .env.example) o como variable de entorno "
        "del sistema antes de lanzar la aplicación."
    )

# Modelos de Gemini
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"

# Rutas del proyecto
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"

# Parámetros de recuperación (RAG)
RETRIEVER_K = 4

# Parámetros de chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Adjuntos puntuales del usuario (no se indexan, solo se usan como contexto de la pregunta actual)
MAX_ATTACHMENT_CHARS = 12000
