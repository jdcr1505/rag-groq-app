"""Utilidades compartidas por las partes del taller (cliente de Groq, imágenes y guardado)."""

import base64
import json
import mimetypes
import os
import re
import sys

from dotenv import load_dotenv

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
IMAGENES = os.path.join(AQUI, "imagenes")
AUDIOS = os.path.join(AQUI, "audios")
RESULTADOS = os.path.join(AQUI, "resultados")

# El .env vive en la raíz del proyecto (nunca se sube al repositorio)
load_dotenv(os.path.join(RAIZ, ".env"))
sys.stdout.reconfigure(encoding="utf-8")

VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo")
TEXT_MODEL = os.getenv("TEXT_MODEL", "openai/gpt-oss-120b")


def cliente():
    from groq import Groq

    clave = os.getenv("GROQ_API_KEY", "").strip()
    if not clave or "PEGA_TU" in clave:
        raise SystemExit("Falta GROQ_API_KEY: créala en el archivo .env de la raíz del proyecto.")
    return Groq(api_key=clave)


def imagen_data_url(ruta: str, mime: str = None) -> str:
    """Imagen local -> data URL en base64 (como en multimodal.py)."""
    mime = mime or mimetypes.guess_type(ruta)[0] or "image/jpeg"
    with open(ruta, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def parte_imagen(ruta: str, mime: str = None) -> dict:
    return {"type": "image_url", "image_url": {"url": imagen_data_url(ruta, mime)}}


def sin_razonamiento(texto: str) -> str:
    """Los modelos de razonamiento pueden devolver <think>...</think> en el contenido."""
    return re.sub(r"<think>.*?</think>", "", texto or "", flags=re.DOTALL).strip()


def guardar(nombre: str, datos):
    os.makedirs(RESULTADOS, exist_ok=True)
    ruta = os.path.join(RESULTADOS, nombre)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)
    print(f"\n(guardado en {os.path.relpath(ruta, RAIZ)})")


def titulo(t: str):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)
