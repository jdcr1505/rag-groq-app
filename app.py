"""
Backend Flask para el sistema RAG conversacional con Groq.

Ejecutar con:
    python app.py        ->  http://localhost:5000

El servidor es stateless respecto a la conversación: el navegador envía el
historial en cada petición, así varios usuarios no mezclan sus diálogos.
"""

import os
import threading
import traceback

from flask import Flask, render_template, request, jsonify

import rag_core

import json
from datetime import datetime

LOG_FILE = "interactions.jsonl"


def registrar_interaccion(pregunta, r, k):
    registro = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "pregunta": pregunta,
        "pregunta_independiente": r["pregunta_independiente"],
        "respuesta": r["respuesta"],
        "sin_respuesta": r["sin_respuesta"],
        "k": k,
        "contextos": [f["texto"] for f in r["fuentes"]],
        "fuentes": [{"fuente": f["fuente"], "pagina": f["pagina"]} for f in r["fuentes"]],
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    except OSError:
        pass

app = Flask(__name__)

state = {
    "vector_store": None,
    "llm": None,
    "log_lines": [],
    "init_status": "pendiente",  # pendiente | inicializando | lista | error
    "init_error": None,
}


def log(msg: str):
    state["log_lines"].append(msg)
    print(msg, flush=True)


def inicializar_base():
    state["init_status"] = "inicializando"
    try:
        state["vector_store"] = rag_core.ensure_vector_store(progress_cb=log)
        state["init_status"] = "lista"
    except Exception as e:
        traceback.print_exc()
        state["init_error"] = str(e)
        state["init_status"] = "error"
        log(f"[ERROR] {e}")


threading.Thread(target=inicializar_base, daemon=True).start()


def _limpiar_historial(raw) -> list:
    """Valida el historial recibido del cliente."""
    if not isinstance(raw, list):
        return []
    limpio = []
    for m in raw[-rag_core.MAX_HISTORY_MESSAGES:]:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
            limpio.append({"role": m["role"], "content": str(m.get("content", ""))[:2000]})
    return limpio


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    try:
        rag_core.get_groq_api_key()
        api_key_ok, api_key_error = True, None
    except ValueError as e:
        api_key_ok, api_key_error = False, str(e)

    return jsonify({
        "api_key_ok": api_key_ok,
        "api_key_error": api_key_error,
        "pdfs": rag_core.list_pdfs(),
        "init_status": state["init_status"],
        "init_error": state["init_error"],
        "log": state["log_lines"],
    })


@app.route("/api/indexed")
def api_indexed():
    if state["vector_store"] is None:
        return jsonify({"error": "La base vectorial aún se está preparando."}), 503
    try:
        return jsonify({"fuentes": rag_core.get_indexed_sources(state["vector_store"])})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(force=True) or {}
    pregunta = str(data.get("pregunta", "")).strip()
    try:
        k = max(1, min(int(data.get("k", rag_core.DEFAULT_K)), 15))
    except (TypeError, ValueError):
        k = rag_core.DEFAULT_K
    historial = _limpiar_historial(data.get("historial"))

    if not pregunta:
        return jsonify({"error": "La pregunta está vacía."}), 400
    if state["vector_store"] is None:
        return jsonify({"error": "La base vectorial aún se está preparando. Intenta de nuevo en unos segundos."}), 503

    try:
        if state["llm"] is None:
            state["llm"] = rag_core.get_llm()

        r = rag_core.rag_pipeline(
            pregunta, state["vector_store"], state["llm"], k=k, historial=historial
        )
        registrar_interaccion(pregunta, r, k)

        fuentes = [
            {**f, "texto": f["texto"][:500]} for f in r["fuentes"]
        ]

        return jsonify({
            "respuesta": r["respuesta"],
            "sin_respuesta": r["sin_respuesta"],
            "pregunta_independiente": r["pregunta_independiente"],
            "fuentes": [] if r["sin_respuesta"] else fuentes,
            "tokens_contexto_aprox": r["tokens_contexto_aprox"],
            "n_recuperados": len(r["fuentes"]),
        })
    
    except Exception as e:
        traceback.print_exc()  # el detalle técnico queda solo en la consola/logs
        if rag_core.es_rate_limit(e):
            return jsonify({"error": "El servicio está recibiendo muchas consultas. Espera unos segundos e inténtalo de nuevo."}), 429
        return jsonify({"error": "Ocurrió un error al procesar tu consulta. Inténtalo de nuevo en un momento."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)