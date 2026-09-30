"""
Backend Flask para el sistema RAG con Groq.

Ejecutar con:
    python app.py

    Abre http://localhost:5000 en tu navegador.
"""

import os
import threading
import traceback

from flask import Flask, render_template, request, jsonify

import rag_core

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
    """
    Carga (o construye, si los PDFs cambiaron) la base vectorial a partir de
    los PDFs precargados en la carpeta pdfs/. Se ejecuta una sola vez en
    segundo plano al arrancar, para no bloquear el servidor.
    """
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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    try:
        rag_core.get_groq_api_key()
        api_key_ok = True
        api_key_error = None
    except ValueError as e:
        api_key_ok = False
        api_key_error = str(e)

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
    data = request.get_json(force=True)
    pregunta = (data or {}).get("pregunta", "").strip()
    k = int((data or {}).get("k", rag_core.DEFAULT_K))

    if not pregunta:
        return jsonify({"error": "La pregunta está vacía."}), 400

    if state["vector_store"] is None:
        return jsonify({"error": "La base vectorial aún se está preparando. Intenta de nuevo en unos segundos."}), 503

    try:
        if state["llm"] is None:
            state["llm"] = rag_core.get_llm()

        resultado = rag_core.rag_pipeline(pregunta, state["vector_store"], state["llm"], k=k)

        fragmentos = [
            {
                "fuente": os.path.basename(d.metadata.get("source", "?")),
                "pagina": d.metadata.get("page", "?"),
                "texto": d.page_content[:400],
            }
            for d in resultado["fragmentos"]
        ]

        return jsonify({
            "respuesta": resultado["respuesta"],
            "fragmentos": fragmentos,
            "tokens_contexto_aprox": resultado["tokens_contexto_aprox"],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)