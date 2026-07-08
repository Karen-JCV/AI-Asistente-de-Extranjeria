"""Carga de los documentos oficiales que alimentan la base de conocimiento del asistente."""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from . import config


def load_base_documents(data_dir: str = config.DATA_DIR) -> list[Document]:
    """Carga todos los PDF oficiales presentes en `data_dir`.

    Cada página de cada PDF se convierte en un ``Document`` independiente,
    conservando en los metadatos la ruta de origen y el número de página,
    lo que permite trazar posteriormente de qué documento procede cada
    fragmento recuperado.

    Parameters
    ----------
    data_dir : str
        Carpeta que contiene los PDF oficiales (Hoja Informativa nº12, nº55, etc.).

    Returns
    -------
    list[Document]
        Lista de páginas cargadas de todos los PDF encontrados.
    """
    pdf_paths = sorted(Path(data_dir).glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No se encontró ningún PDF en '{data_dir}'. "
            "Añade las Hojas Informativas oficiales antes de construir la base vectorial."
        )

    documents: list[Document] = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())

    return documents
