"""
Backend Flask para el sistema RAG conversacional con Groq.

Ejecutar con:
    python app.py        ->  http://localhost:5000

Páginas:   /  (landing)   /acceso (iniciar sesión o crear cuenta)   /chat (asistente)
Cuentas:   /api/auth/*          Conversaciones: /api/conversaciones/*

Cada usuario tiene sus conversaciones guardadas en la base de datos (Neon/PostgreSQL en
producción, SQLite en local; ver db.py). El historial que recibe el modelo se lee de esa
base, no del navegador.
"""

import base64
import binascii
import os
import re
import secrets
import threading
import traceback
from datetime import timedelta
from urllib.parse import urlparse

from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.exceptions import HTTPException
from dotenv import load_dotenv

# El .env debe leerse ANTES de importar db: db.py toma DATABASE_URL al importarse.
# (Dentro de Docker las variables ya vienen del entorno y esto no cambia nada.)
load_dotenv()

import auth
import db
import rag_core

import json
from datetime import datetime

LOG_FILE = "interactions.jsonl"


MAX_IMAGEN_BYTES = 4 * 1024 * 1024   # límite de Groq para imágenes en base64
MAX_AUDIO_BYTES = 20 * 1024 * 1024
MIME_IMAGEN = ("image/jpeg", "image/png", "image/webp")


def registrar_interaccion(pregunta, r, k):
    registro = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "pregunta": pregunta,
        "pregunta_independiente": r["pregunta_independiente"],
        "respuesta": r["respuesta"],
        "sin_respuesta": r["sin_respuesta"],
        "modo": r.get("modo"),
        "herramientas": r.get("herramientas", []),
        "k": k,
        "contextos": [f["texto"] for f in r["fuentes"]],
        "fuentes": [{"fuente": f["fuente"], "pagina": f["pagina"], "ubicacion": f["ubicacion"]} for f in r["fuentes"]],
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    except OSError:
        pass

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

# Firma de la cookie de sesión. En Cloud Run debe venir de la variable SECRET_KEY: si cada
# instancia inventara la suya, la sesión se perdería al cambiar de instancia o reiniciar.
_secret = os.environ.get("SECRET_KEY", "").strip()
if not _secret:
    _secret = secrets.token_hex(32)
    print("[AVISO] SECRET_KEY no está definida: las sesiones se perderán al reiniciar el servidor.", flush=True)
app.config.update(
    SECRET_KEY=_secret,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=bool(os.environ.get("K_SERVICE")) or os.environ.get("COOKIE_SECURE") == "1",
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
)

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


def preparar_base_de_datos():
    """Crea las tablas y despierta la base (Neon se suspende cuando no se usa)."""
    try:
        db.iniciar()
        log("Base de datos de usuarios lista.")
    except Exception as e:
        print(f"[AVISO] No se pudo preparar la base de datos: {e}", flush=True)


threading.Thread(target=preparar_base_de_datos, daemon=True).start()


# ───────────────── Seguridad común ─────────────────

@app.before_request
def proteger_peticiones():
    """Las peticiones que cambian datos deben venir de nuestro propio sitio."""
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return None
    origen = request.headers.get("Origin")
    if origen and urlparse(origen).netloc != request.host:
        return jsonify({"error": "Origen no permitido."}), 403
    return None


@app.after_request
def cabeceras(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    if request.path.startswith("/api/") or request.path in ("/chat", "/acceso"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.errorhandler(Exception)
def error_inesperado(e):
    if isinstance(e, HTTPException):
        if request.path.startswith("/api/"):
            return jsonify({"error": e.description}), e.code
        return e
    traceback.print_exc()
    if request.path.startswith("/api/"):
        return jsonify({"error": "No se pudo completar la operación. Inténtalo de nuevo en un momento."}), 503
    return "Servicio no disponible por ahora. Intenta de nuevo en un momento.", 503


def _json():
    datos = request.get_json(silent=True)
    return datos if isinstance(datos, dict) else {}


# ───────────────── Páginas ─────────────────

@app.route("/")
def landing():
    return render_template("landing.html", usuario=auth.usuario_actual())


@app.route("/acceso")
def acceso():
    if auth.usuario_actual():
        return redirect(url_for("chat"))
    modo = "registro" if request.args.get("modo") == "registro" else "login"
    return render_template("auth.html", modo=modo)


@app.route("/chat")
@auth.login_required
def chat():
    return render_template("chat.html", usuario=auth.usuario_actual())


# ───────────────── Cuenta ─────────────────

def _usuario_publico(u):
    return {"id": u["id"], "nombre": u["nombre"], "correo": u["correo"]}


@app.route("/api/auth/registro", methods=["POST"])
def api_registro():
    d = _json()
    try:
        u = auth.registrar(d.get("nombre"), d.get("correo"), d.get("password"))
    except auth.ErrorAuth as e:
        return jsonify({"error": e.mensaje}), e.codigo
    return jsonify({"usuario": _usuario_publico(u)}), 201


@app.route("/api/auth/login", methods=["POST"])
def api_login():
    d = _json()
    try:
        u = auth.ingresar(d.get("correo"), d.get("password"))
    except auth.ErrorAuth as e:
        return jsonify({"error": e.mensaje}), e.codigo
    return jsonify({"usuario": _usuario_publico(u)})


@app.route("/api/auth/logout", methods=["POST"])
def api_logout():
    auth.cerrar_sesion()
    return jsonify({"ok": True})


@app.route("/api/auth/cuenta", methods=["DELETE"])
@auth.login_required
def api_borrar_cuenta():
    """Borra la cuenta y, por cascada, todas sus conversaciones y chats."""
    u = auth.usuario_actual()
    try:
        auth.verificar_password(u, _json().get("password"))
    except auth.ErrorAuth as e:
        return jsonify({"error": e.mensaje}), e.codigo
    db.borrar_usuario(u["id"])
    auth.cerrar_sesion()
    return jsonify({"ok": True})


# ───────────────── Conversaciones ─────────────────

@app.route("/api/conversaciones")
@auth.login_required
def api_conversaciones():
    return jsonify({"conversaciones": db.listar_conversaciones(auth.usuario_actual()["id"])})


@app.route("/api/conversaciones/<cid>")
@auth.login_required
def api_conversacion(cid):
    conv = db.obtener_conversacion(auth.usuario_actual()["id"], cid)
    if conv is None:
        return jsonify({"error": "No encontramos esa conversación."}), 404
    return jsonify(conv)


@app.route("/api/conversaciones/<cid>", methods=["PATCH"])
@auth.login_required
def api_renombrar(cid):
    titulo = " ".join(str(_json().get("titulo", "")).split())
    if not titulo:
        return jsonify({"error": "Escribe un nombre para la conversación."}), 400
    conv = db.renombrar_conversacion(auth.usuario_actual()["id"], cid, titulo)
    if conv is None:
        return jsonify({"error": "No encontramos esa conversación."}), 404
    return jsonify(conv)


@app.route("/api/conversaciones/<cid>", methods=["DELETE"])
@auth.login_required
def api_borrar_conversacion(cid):
    if not db.borrar_conversacion(auth.usuario_actual()["id"], cid):
        return jsonify({"error": "No encontramos esa conversación."}), 404
    return jsonify({"ok": True})


# ───────────────── Asistente ─────────────────

@app.route("/api/status")
@auth.login_required
def api_status():
    try:
        rag_core.get_groq_api_key()
        api_key_ok, api_key_error = True, None
    except ValueError as e:
        api_key_ok, api_key_error = False, str(e)

    return jsonify({
        "api_key_ok": api_key_ok,
        "api_key_error": api_key_error,
        "web_ok": rag_core.web_search_disponible(),
        "documentos": rag_core.list_documents(),
        "init_status": state["init_status"],
        "init_error": state["init_error"],
        "log": state["log_lines"],
    })


@app.route("/api/indexed")
@auth.login_required
def api_indexed():
    if state["vector_store"] is None:
        return jsonify({"error": "La base vectorial aún se está preparando."}), 503
    try:
        return jsonify({"fuentes": rag_core.get_indexed_sources(state["vector_store"])})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/ask", methods=["POST"])
@auth.login_required
def api_ask():
    uid = auth.usuario_actual()["id"]
    data = request.get_json(silent=True) or {}
    pregunta = str(data.get("pregunta", "")).strip()
    try:
        k = max(1, min(int(data.get("k", rag_core.DEFAULT_K)), 15))
    except (TypeError, ValueError):
        k = rag_core.DEFAULT_K
    usar_web = bool(data.get("web"))
    cid = data.get("conversacion_id") or None

    try:
        imagen = _leer_imagen(data.get("imagen"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not pregunta and imagen:
        pregunta = "¿Qué dice la normativa sobre lo que aparece en esta imagen?"
    if not pregunta:
        return jsonify({"error": "La pregunta está vacía."}), 400
    if state["vector_store"] is None:
        return jsonify({"error": "La base vectorial aún se está preparando. Intenta de nuevo en unos segundos."}), 503

    # El historial sale de la conversación guardada, no de lo que mande el navegador
    historial = []
    if cid:
        if not db.existe_conversacion(uid, cid):
            return jsonify({"error": "Esa conversación ya no existe. Empieza una nueva."}), 404
        historial = db.historial_para_modelo(uid, cid, rag_core.MAX_HISTORY_MESSAGES)

    try:
        if state["llm"] is None:
            state["llm"] = rag_core.get_llm()

        descripcion = None
        if imagen:
            descripcion = rag_core.describir_imagen(imagen["base64"], imagen["mime"], pregunta)

        r = rag_core.agent_pipeline(
            pregunta, state["vector_store"], state["llm"], k=k, historial=historial,
            usar_web=usar_web, descripcion_imagen=descripcion,
        )
        registrar_interaccion(pregunta, r, k)

        fuentes = [
            {**f, "texto": f["texto"][:500]} for f in r["fuentes"]
        ]
        fuentes_web = [
            {**w, "texto": w["texto"][:500]} for w in r.get("fuentes_web", [])
        ]

        respuesta = {
            "respuesta": r["respuesta"],
            "sin_respuesta": r["sin_respuesta"],
            "pregunta_independiente": r["pregunta_independiente"],
            "herramientas": r.get("herramientas", []),
            "descripcion_imagen": descripcion,
            "tiene_imagen": bool(imagen),
            "fuentes": [] if r["sin_respuesta"] else fuentes,
            "fuentes_web": [] if r["sin_respuesta"] else fuentes_web,
            "tokens_contexto_aprox": r["tokens_contexto_aprox"],
            "n_recuperados": len(r["fuentes"]),
        }
    except Exception as e:
        traceback.print_exc()  # el detalle técnico queda solo en la consola/logs
        if rag_core.es_rate_limit(e):
            return jsonify({"error": "El servicio está recibiendo muchas consultas. Espera unos segundos e inténtalo de nuevo."}), 429
        return jsonify({"error": "Ocurrió un error al procesar tu consulta. Inténtalo de nuevo en un momento."}), 500

    # Guardar el turno. Si la base falla, el usuario igual recibe su respuesta.
    try:
        extra = {k_: v for k_, v in respuesta.items() if k_ != "respuesta"}
        conv = db.guardar_turno(uid, cid, pregunta, r["respuesta"], extra)
        respuesta["conversacion"] = conv
    except Exception:
        traceback.print_exc()
        respuesta["conversacion"] = {"id": cid, "titulo": None}
        respuesta["aviso"] = "No se pudo guardar esta respuesta en tu historial."

    return jsonify(respuesta)


@app.route("/api/transcribe", methods=["POST"])
@auth.login_required
def api_transcribe():
    """Recibe el audio grabado en el navegador y lo transcribe con Whisper (Groq)."""
    archivo = request.files.get("audio")
    if archivo is None:
        return jsonify({"error": "No se recibió audio."}), 400
    audio = archivo.read()
    if not audio:
        return jsonify({"error": "La grabación está vacía."}), 400
    if len(audio) > MAX_AUDIO_BYTES:
        return jsonify({"error": "La grabación es demasiado larga."}), 400

    try:
        texto = rag_core.transcribir_audio(audio, archivo.filename or "grabacion.webm")
        return jsonify({"texto": texto})
    except Exception as e:
        traceback.print_exc()
        if rag_core.es_rate_limit(e):
            return jsonify({"error": "El servicio está recibiendo muchas consultas. Espera unos segundos e inténtalo de nuevo."}), 429
        return jsonify({"error": "No se pudo transcribir el audio. Inténtalo de nuevo."}), 500


def _leer_imagen(raw):
    """Valida la imagen enviada como data URL (data:image/jpeg;base64,...)."""
    if not raw:
        return None
    m = re.match(r"^data:(image/[a-z+]+);base64,([A-Za-z0-9+/=]+)$", str(raw))
    if not m or m.group(1) not in MIME_IMAGEN:
        raise ValueError("Formato de imagen no admitido. Usa JPG, PNG o WebP.")
    if len(m.group(2)) > MAX_IMAGEN_BYTES:
        raise ValueError("La imagen es demasiado grande (máximo 4 MB).")
    try:
        base64.b64decode(m.group(2), validate=True)
    except binascii.Error:
        raise ValueError("La imagen está dañada.")
    return {"mime": m.group(1), "base64": m.group(2)}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)