"""
Parte 2. Audio y transcripción.

    python parte2_audio.py              # transcribe todos los audios de audios/ y responde cada pregunta
    python parte2_audio.py --solo-transcribir

Los audios y sus datos se describen en audios/audios.json:
  [{"archivo": "a1_juan.m4a", "hablante": "Juan", "ruido": false,
    "lo_que_se_dijo": "texto exacto que se leyó"}]

2.1 Transcripción con Whisper en Groq (WHISPER_MODEL, por defecto whisper-large-v3-turbo).
2.2 La transcripción se pasa al agente del proyecto (rag_core.agent_pipeline), que
    responde con la base de conocimientos y cita la fuente. Con --sin-agente se usa
    el LLM directamente, sin base.
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata

from comun import AUDIOS, RAIZ, TEXT_MODEL, WHISPER_MODEL, cliente, guardar, titulo

LIMITE_BYTES = 25 * 1024 * 1024  # plan gratuito de Groq
FORMATOS = (".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac", ".mp4", ".mpeg", ".mpga")


def duracion(ruta: str):
    """Duración en segundos (wav con la librería estándar; otros formatos si está mutagen)."""
    try:
        if ruta.lower().endswith(".wav"):
            import wave
            with wave.open(ruta) as w:
                return round(w.getnframes() / w.getframerate(), 1)
        from mutagen import File
        return round(File(ruta).info.length, 1)
    except Exception:
        return None


def _palabras(t: str) -> list:
    t = unicodedata.normalize("NFKD", t or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", " ", t).split()


def wer(referencia: str, hipotesis: str):
    """Word Error Rate: (sustituciones + borrados + inserciones) / palabras de la referencia."""
    r, h = _palabras(referencia), _palabras(hipotesis)
    if not r:
        return None
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return round(d[len(h)] / len(r), 3)


def transcribir(ruta: str) -> str:
    tam = os.path.getsize(ruta)
    if tam > LIMITE_BYTES:
        raise ValueError(f"{os.path.basename(ruta)} pesa {tam / 1e6:.1f} MB (límite 25 MB)")
    with open(ruta, "rb") as f:
        r = cliente().audio.transcriptions.create(
            file=(os.path.basename(ruta), f.read()),
            model=WHISPER_MODEL,
            language="es",
            temperature=0,
        )
    return r.text.strip()


def responder(transcripcion: str, con_agente: bool) -> dict:
    if con_agente:
        sys.path.insert(0, RAIZ)
        cwd = os.getcwd()
        os.chdir(RAIZ)  # el índice (chroma/) y los PDFs están en la raíz
        try:
            import rag_core
            if not hasattr(responder, "_store"):
                responder._store = rag_core.ensure_vector_store(progress_cb=print)
                responder._llm = rag_core.get_llm()
            r = rag_core.agent_pipeline(transcripcion, responder._store, responder._llm, registrar=False)
        finally:
            os.chdir(cwd)
        return {"modo": "agente RAG", "respuesta": r["respuesta"],
                "tools": [f"{h['nombre']}({h['args']})" for h in r["herramientas"]],
                "fuentes": [f"{f['fuente']}, {f['ubicacion']}" for f in r["fuentes"]]}
    r = cliente().chat.completions.create(
        model=TEXT_MODEL, temperature=0,
        messages=[{"role": "system", "content": "Eres un asistente experto en normativa colombiana. Responde en español, breve y preciso."},
                  {"role": "user", "content": transcripcion}])
    return {"modo": f"LLM directo ({TEXT_MODEL})", "respuesta": r.choices[0].message.content.strip()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-transcribir", action="store_true")
    ap.add_argument("--sin-agente", action="store_true", help="Responde con el LLM sin la base de conocimientos")
    args = ap.parse_args()

    meta_path = os.path.join(AUDIOS, "audios.json")
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = {m["archivo"]: m for m in json.load(f)}
    archivos = sorted(a for a in os.listdir(AUDIOS) if a.lower().endswith(FORMATOS)) if os.path.isdir(AUDIOS) else []
    if not archivos:
        raise SystemExit("No hay audios en taller_multimodal/audios/ (ver README).")

    filas = []
    for a in archivos:
        ruta = os.path.join(AUDIOS, a)
        m = meta.get(a, {})
        titulo(f"2.1 Transcripción: {a}")
        t0 = time.perf_counter()
        try:
            texto = transcribir(ruta)
        except Exception as e:
            print("ERROR:", e)
            filas.append({"archivo": a, "error": str(e)})
            continue
        fila = {
            "archivo": a, "hablante": m.get("hablante"), "ruido": m.get("ruido"),
            "duracion_s": m.get("duracion_s") or duracion(ruta),
            "lo_que_se_dijo": m.get("lo_que_se_dijo"), "transcripcion": texto,
            "wer": wer(m.get("lo_que_se_dijo", ""), texto) if m.get("lo_que_se_dijo") else None,
            "tiempo_transcripcion_s": round(time.perf_counter() - t0, 2),
        }
        print(f"Hablante: {fila['hablante']} | Ruido: {fila['ruido']} | Duración: {fila['duracion_s']} s")
        print(f"Lo que se dijo: {fila['lo_que_se_dijo']}")
        print(f"Transcripción ({WHISPER_MODEL}): {texto}")
        if fila["wer"] is not None:
            print(f"WER: {fila['wer']:.1%}")
        if not args.solo_transcribir:
            titulo(f"2.2 Respuesta a partir de la transcripción: {a}")
            fila["respuesta"] = responder(texto, con_agente=not args.sin_agente)
            print(json.dumps(fila["respuesta"], ensure_ascii=False, indent=1))
        filas.append(fila)
    guardar("2_audio.json", {"modelo_transcripcion": WHISPER_MODEL, "audios": filas})
