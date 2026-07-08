"""Definición del agente mediante LangGraph: retrieve -> generate -> memory.

Este módulo reproduce la arquitectura definida en el notebook y añade
soporte opcional para un "adjunto puntual" (texto de un documento subido por el
usuario en el chat), que se combina con los fragmentos recuperados de ChromaDB
únicamente para la pregunta en curso.
"""
from typing import Annotated

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from .prompts import QA_PROMPT


class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: list[Document]
    attachment_text: str


def format_context(docs: list[Document], attachment_text: str = "") -> str:
    """Combina los fragmentos recuperados de la base oficial con el adjunto puntual.

    El adjunto NUNCA se indexa en ChromaDB: solo viaja en el estado de la
    conversación actual y se descarta al terminar la sesión.
    """
    official_context = "\n\n".join(doc.page_content for doc in docs)
    parts = [official_context] if official_context else []
    if attachment_text:
        parts.append("DOCUMENTO ADJUNTADO POR EL USUARIO:\n" + attachment_text)
    return "\n\n".join(parts)


def build_graph(retriever, llm):
    """Construye y compila el grafo del asistente con memoria conversacional.

    Parameters
    ----------
    retriever : BaseRetriever
        Retriever sobre la base vectorial ChromaDB.
    llm : BaseChatModel
        Modelo de chat de Gemini.

    Returns
    -------
    CompiledStateGraph
        Grafo compilado, con checkpointer en memoria (``MemorySaver``) para
        mantener el historial de conversación por ``thread_id``.
    """
    rag_chain = QA_PROMPT | llm | StrOutputParser()

    def retrieve_documents(state: State) -> dict:
        question = state["messages"][-1].content
        docs = retriever.invoke(question)
        return {"context": docs}

    def generate_response(state: State) -> dict:
        context = format_context(state["context"], state.get("attachment_text", ""))
        question = state["messages"][-1].content
        answer = rag_chain.invoke({"context": context, "question": question})
        return {"messages": [AIMessage(content=answer)]}

    def update_memory(state: State) -> dict:
        # La persistencia del historial la gestiona el checkpointer (MemorySaver).
        # Nodo mantenido explícito para futuras ampliaciones (auditoría, métricas...).
        return {}

    builder = StateGraph(State)
    builder.add_node("retrieve", retrieve_documents)
    builder.add_node("generate", generate_response)
    builder.add_node("memory", update_memory)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "memory")
    builder.add_edge("memory", END)

    return builder.compile(checkpointer=MemorySaver())
