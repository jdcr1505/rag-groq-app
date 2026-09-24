"""
Backend Flask para el sistema RAG con Groq.

Ejecutar con:
    python app.py

    Abre http://localhost:5000 en tu navegador.
"""

import os
import traceback

from flask import Flask, render_template, request, jsonify

import rag_core

app = Flask(__name__)

state = {
    "vector_store": None,
    "llm": None,
    "log_lines": [],
}


def log(msg: str):
    state["log_lines"].append(msg)
    print(msg)


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
        "vector_store_on_disk": rag_core.vector_store_exists(),
        "vector_store_loaded": state["vector_store"] is not None,
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    archivos = request.files.getlist("pdfs")
    if not archivos:
        return jsonify({"error": "No se recibió ningún archivo."}), 400

    os.makedirs(rag_core.PDF_DIR, exist_ok=True)
    guardados = []
    for archivo in archivos:
        if archivo.filename.lower().endswith(".pdf"):
            destino = os.path.join(rag_core.PDF_DIR, archivo.filename)
            archivo.save(destino)
            guardados.append(archivo.filename)

    return jsonify({"guardados": guardados, "pdfs": rag_core.list_pdfs()})

@app.route("/api/delete_pdf", methods=["POST"])
def api_delete_pdf():
    data = request.get_json(force=True)
    nombre = (data or {}).get("nombre", "").strip()

    if not nombre or "/" in nombre or "\\" in nombre:
        return jsonify({"error": "Nombre de archivo inválido."}), 400

    ruta = os.path.join(rag_core.PDF_DIR, nombre)
    if not os.path.isfile(ruta):
        return jsonify({"error": f"No se encontró '{nombre}' en la carpeta pdfs/."}), 404

    try:
        os.remove(ruta)
        return jsonify({"ok": True, "pdfs": rag_core.list_pdfs()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/build", methods=["POST"])
def api_build():
    state["vector_store"] = None  # libera referencia previa (evita bloqueos de archivo en Windows)
    state["log_lines"] = []
    try:
        vector_store, _ = rag_core.build_vector_store(progress_cb=log)
        state["vector_store"] = vector_store
        state["llm"] = rag_core.get_llm()
        return jsonify({"ok": True, "log": state["log_lines"]})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e), "log": state["log_lines"]}), 400


@app.route("/api/load", methods=["POST"])
def api_load():
    try:
        vector_store, _ = rag_core.load_existing_vector_store()
        state["vector_store"] = vector_store
        state["llm"] = rag_core.get_llm()
        return jsonify({"ok": True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/api/indexed")
def api_indexed():
    if state["vector_store"] is None:
        return jsonify({"error": "No hay base vectorial cargada."}), 400
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
        return jsonify({"error": "Primero debes construir o cargar la base vectorial."}), 400

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
    app.run(debug=True, port=5000)
