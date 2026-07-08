"""Transcripción de audio a texto mediante Gemini.

Soporta tanto el audio grabado en vivo desde el navegador (``st.audio_input``)
como archivos de audio subidos por el usuario (mp3, wav, m4a...).

Usa el SDK oficial ``google-genai`` (el paquete ``google-generativeai`` está
deprecado por Google).
"""
from google import genai
from google.genai import types

from . import config

_client = genai.Client(api_key=config.GEMINI_API_KEY)

_TRANSCRIBE_PROMPT = (
    "Transcribe literalmente, en español, el siguiente audio. "
    "Devuelve únicamente el texto transcrito, sin comentarios, encabezados ni "
    "explicaciones adicionales. Si el audio no contiene voz inteligible, "
    "devuelve una cadena vacía."
)


def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
    """Transcribe un fragmento de audio a texto usando el modelo Gemini.

    Parameters
    ----------
    audio_bytes : bytes
        Contenido binario del audio.
    mime_type : str
        Tipo MIME del audio (p. ej. "audio/wav", "audio/mp3", "audio/mp4").

    Returns
    -------
    str
        Texto transcrito (puede ser una cadena vacía si no se detectó voz).
    """
    response = _client.models.generate_content(
        model=config.LLM_MODEL,
        contents=[
            _TRANSCRIBE_PROMPT,
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
        ],
    )
    return (response.text or "").strip()
