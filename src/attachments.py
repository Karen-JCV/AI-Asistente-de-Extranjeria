"""Extracción de texto de documentos adjuntados puntualmente por el usuario en el chat.

Estos adjuntos (PDF, Word o TXT) se usan únicamente como contexto adicional para
la pregunta actual de la conversación. A diferencia de los documentos oficiales,
NO se indexan en ChromaDB ni persisten más allá de la sesión del navegador.
"""
from io import BytesIO

import docx
from pypdf import PdfReader

from . import config

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt")


def extract_text(uploaded_file) -> str:
    """Extrae el texto de un archivo subido desde Streamlit.

    Parameters
    ----------
    uploaded_file : streamlit.runtime.uploaded_file_manager.UploadedFile
        Archivo devuelto por ``st.file_uploader``.

    Returns
    -------
    str
        Texto extraído, truncado a ``config.MAX_ATTACHMENT_CHARS`` caracteres.

    Raises
    ------
    ValueError
        Si el formato del archivo no está soportado.
    """
    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith(".pdf"):
        text = _extract_pdf(data)
    elif name.endswith(".docx"):
        text = _extract_docx(data)
    elif name.endswith(".txt"):
        text = data.decode("utf-8", errors="ignore")
    else:
        raise ValueError(
            f"Formato no soportado: '{uploaded_file.name}'. "
            f"Formatos admitidos: {', '.join(SUPPORTED_EXTENSIONS)}."
        )

    text = text.strip()
    if len(text) > config.MAX_ATTACHMENT_CHARS:
        text = text[: config.MAX_ATTACHMENT_CHARS] + "\n[... documento truncado ...]"

    return text


def _extract_pdf(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_docx(data: bytes) -> str:
    document = docx.Document(BytesIO(data))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)
