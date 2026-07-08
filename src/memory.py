"""Utilidades de memoria conversacional para la sesión de Streamlit.

La memoria "real" del agente la gestiona el checkpointer de LangGraph
(``MemorySaver``), indexado por ``thread_id``. Este módulo solo se encarga de
generar y recuperar el ``thread_id`` asociado a la sesión del navegador.
"""
import uuid


def get_or_create_thread_id(session_state) -> str:
    """Devuelve el ``thread_id`` de la conversación actual de la sesión.

    Si es la primera interacción de la sesión, genera un identificador nuevo.

    Parameters
    ----------
    session_state : streamlit.runtime.state.SessionStateProxy
        El ``st.session_state`` de la app.

    Returns
    -------
    str
        Identificador de hilo de conversación.
    """
    if "thread_id" not in session_state:
        session_state.thread_id = str(uuid.uuid4())
    return session_state.thread_id


def reset_thread_id(session_state) -> str:
    """Genera un nuevo ``thread_id``, empezando una conversación en blanco."""
    session_state.thread_id = str(uuid.uuid4())
    return session_state.thread_id
