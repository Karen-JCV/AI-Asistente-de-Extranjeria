"""Construcción y carga de la base vectorial (ChromaDB) con la documentación oficial."""
from pathlib import Path

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .document_loader import load_base_documents
from .embeddings import get_embeddings


def build_or_load_vectorstore(force_rebuild: bool = False) -> Chroma:
    """Devuelve la base vectorial, construyéndola desde cero si todavía no existe.

    Si ya existe una base persistida en ``config.CHROMA_DIR`` se reutiliza
    directamente (evitando recalcular embeddings en cada arranque). Pasa
    ``force_rebuild=True`` para regenerarla, por ejemplo tras actualizar los
    PDF oficiales en ``data/``.

    Parameters
    ----------
    force_rebuild : bool
        Si es True, ignora cualquier base persistida y la reconstruye.

    Returns
    -------
    Chroma
        Base vectorial lista para crear un retriever sobre ella.
    """
    embeddings = get_embeddings()
    persist_dir = Path(config.CHROMA_DIR)
    already_built = persist_dir.exists() and any(persist_dir.iterdir())

    if already_built and not force_rebuild:
        return Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )

    documents = load_base_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )
