"""Interfaz Streamlit del Asistente de Extranjería.

Funcionalidades:
- Chat con respuesta en streaming (efecto máquina de escribir).
- Adjuntar un documento (PDF/Word/TXT) como contexto puntual de la conversación
  (no se indexa en ChromaDB).
- Preguntar por voz: grabación en vivo desde el navegador o subida de un audio.
- Memoria conversacional gestionada por LangGraph (checkpointer en memoria).
"""
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src import config
from src.attachments import extract_text
from src.audio import transcribe_audio
from src.graph import build_graph, format_context
from src.memory import get_or_create_thread_id, reset_thread_id
from src.retriever import get_retriever
from src.vectorstore import build_or_load_vectorstore

st.set_page_config(
    page_title="Asistente de Extranjería",
    page_icon="🛂",
    layout="centered",
)

AUDIO_MIME_BY_EXT = {
    "wav": "audio/wav",
    "mp3": "audio/mp3",
    "m4a": "audio/mp4",
    "ogg": "audio/ogg",
}


# --------------------------------------------------------------------------- #
# Recursos cacheados (se construyen una única vez por proceso)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Cargando base de conocimiento...")
def load_resources():
    vectorstore = build_or_load_vectorstore()
    retriever = get_retriever(vectorstore)
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=0,
        google_api_key=config.GEMINI_API_KEY,
    )
    graph = build_graph(retriever, llm)
    return retriever, llm, graph


retriever, llm, graph = load_resources()


# --------------------------------------------------------------------------- #
# Estado de sesión
# --------------------------------------------------------------------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"|"assistant", "content": str}]
if "attachment_text" not in st.session_state:
    st.session_state.attachment_text = ""
if "attachment_name" not in st.session_state:
    st.session_state.attachment_name = None
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

thread_id = get_or_create_thread_id(st.session_state)


# --------------------------------------------------------------------------- #
# Barra lateral: info del proyecto + adjuntar documento + nueva conversación
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🛂 Asistente de Extranjería")
    st.caption(
        "Autorizaciones de residencia y trabajo en España · RAG + LangGraph + Gemini"
    )

    st.divider()

    if st.button("🆕 Nueva conversación", use_container_width=True):
        st.session_state.messages = []
        st.session_state.attachment_text = ""
        st.session_state.attachment_name = None
        reset_thread_id(st.session_state)
        st.rerun()

    st.divider()

    st.subheader("📎 Adjuntar documento")
    st.caption(
        "Se usa solo como contexto de esta conversación. **No** se guarda en la "
        "base de conocimiento oficial."
    )
    uploaded_doc = st.file_uploader(
        "PDF, Word o TXT", type=["pdf", "docx", "txt"], key="doc_uploader"
    )
    if uploaded_doc is not None and uploaded_doc.name != st.session_state.attachment_name:
        try:
            with st.spinner("Leyendo documento..."):
                st.session_state.attachment_text = extract_text(uploaded_doc)
            st.session_state.attachment_name = uploaded_doc.name
        except ValueError as exc:
            st.error(str(exc))

    if st.session_state.attachment_name:
        st.success(f"Adjunto activo: {st.session_state.attachment_name}")
        if st.button("Quitar adjunto"):
            st.session_state.attachment_text = ""
            st.session_state.attachment_name = None
            st.rerun()

    st.divider()
    st.caption(
        "⚠️ Este asistente ofrece información orientativa a partir de las Hojas "
        "Informativas nº12 y nº55. No sustituye el asesoramiento legal oficial."
    )


# --------------------------------------------------------------------------- #
# Historial de chat
# --------------------------------------------------------------------------- #
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------------------------------- #
# Entrada por voz (grabación en vivo o archivo subido)
# --------------------------------------------------------------------------- #
with st.expander("🎙️ Preguntar por voz"):
    tab_mic, tab_file = st.tabs(["Grabar", "Subir audio"])

    with tab_mic:
        mic_audio = st.audio_input("Graba tu pregunta")
        if mic_audio is not None:
            if st.button("Transcribir grabación", key="transcribe_mic"):
                with st.spinner("Transcribiendo audio..."):
                    texto = transcribe_audio(mic_audio.getvalue(), mime_type="audio/wav")
                if texto:
                    st.session_state.pending_input = texto
                    st.rerun()
                else:
                    st.warning("No se detectó voz en la grabación. Inténtalo de nuevo.")

    with tab_file:
        audio_file = st.file_uploader(
            "Archivo de audio (wav, mp3, m4a, ogg)",
            type=list(AUDIO_MIME_BY_EXT.keys()),
            key="audio_uploader",
        )
        if audio_file is not None and st.button("Transcribir archivo", key="transcribe_file"):
            ext = audio_file.name.rsplit(".", 1)[-1].lower()
            mime_type = AUDIO_MIME_BY_EXT.get(ext, "audio/wav")
            with st.spinner("Transcribiendo audio..."):
                texto = transcribe_audio(audio_file.getvalue(), mime_type=mime_type)
            if texto:
                st.session_state.pending_input = texto
                st.rerun()
            else:
                st.warning("No se detectó voz en el archivo. Prueba con otro audio.")


# --------------------------------------------------------------------------- #
# Entrada de texto
# --------------------------------------------------------------------------- #
typed_input = st.chat_input("Escribe tu pregunta sobre extranjería...")
user_input = st.session_state.pending_input or typed_input
st.session_state.pending_input = None


def stream_answer(question: str):
    """Recupera contexto, llama al LLM en streaming y persiste el turno en LangGraph."""
    docs = retriever.invoke(question)
    context = format_context(docs, st.session_state.attachment_text)

    from src.prompts import QA_PROMPT  # import local para evitar ciclos

    messages = QA_PROMPT.format_messages(context=context, question=question)

    full_answer = ""
    for chunk in llm.stream(messages):
        token = chunk.content or ""
        full_answer += token
        yield token

    graph.update_state(
        config={"configurable": {"thread_id": thread_id}},
        values={
            "messages": [HumanMessage(content=question), AIMessage(content=full_answer)],
            "context": docs,
        },
    )


if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        full_response = st.write_stream(stream_answer(user_input))

    st.session_state.messages.append({"role": "assistant", "content": full_response})
