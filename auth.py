"""
Autenticación: registro, inicio de sesión y sesión por cookie firmada.

- La contraseña nunca se guarda: solo su hash (werkzeug, scrypt).
- La sesión es la cookie firmada de Flask (SECRET_KEY) y guarda únicamente el id del usuario.
  En cada petición se vuelve a consultar el usuario, así una cuenta borrada pierde el acceso.
- Las rutas que cambian datos aceptan solo JSON y comprueban el origen (ver app.py).
- Intentos fallidos de inicio de sesión limitados por correo e IP (en memoria de cada instancia).
"""

import re
import threading
import time
from functools import wraps

from flask import g, jsonify, redirect, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

import db

CORREO_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s.]{2,}$")
MIN_PASSWORD = 8
MAX_PASSWORD = 128
MAX_NOMBRE = 60

# Hash de relleno: al iniciar sesión con un correo inexistente se compara igual, para que
# el tiempo de respuesta no revele qué correos están registrados.
_HASH_FALSO = generate_password_hash("relleno-para-igualar-tiempos")

# Limitador de intentos fallidos
MAX_INTENTOS = 5
VENTANA_SEG = 10 * 60
_intentos = {}          # clave -> [marcas de tiempo de fallos]
_intentos_lock = threading.Lock()


class ErrorAuth(Exception):
    def __init__(self, mensaje, codigo=400):
        super().__init__(mensaje)
        self.mensaje = mensaje
        self.codigo = codigo


# ───────────────── Validación ─────────────────

def normalizar_correo(correo):
    return str(correo or "").strip().lower()


def validar_registro(nombre, correo, password):
    nombre = " ".join(str(nombre or "").split())
    correo = normalizar_correo(correo)
    password = str(password or "")
    if not nombre:
        raise ErrorAuth("Escribe tu nombre.")
    if len(nombre) > MAX_NOMBRE:
        raise ErrorAuth(f"El nombre admite hasta {MAX_NOMBRE} caracteres.")
    if not CORREO_RE.match(correo):
        raise ErrorAuth("Escribe un correo válido, por ejemplo nombre@dominio.com.")
    if len(password) < MIN_PASSWORD:
        raise ErrorAuth(f"La contraseña debe tener al menos {MIN_PASSWORD} caracteres.")
    if len(password) > MAX_PASSWORD:
        raise ErrorAuth(f"La contraseña admite hasta {MAX_PASSWORD} caracteres.")
    return nombre, correo, password


# ───────────────── Limitador de intentos ─────────────────

def _clave_intentos(correo):
    return f"{request.remote_addr}|{correo}"


def _bloqueado(clave):
    ahora = time.time()
    with _intentos_lock:
        recientes = [t for t in _intentos.get(clave, []) if ahora - t < VENTANA_SEG]
        if recientes:
            _intentos[clave] = recientes
        else:
            _intentos.pop(clave, None)
        return len(recientes) >= MAX_INTENTOS


def _registrar_fallo(clave):
    with _intentos_lock:
        _intentos.setdefault(clave, []).append(time.time())


def _limpiar_fallos(clave):
    with _intentos_lock:
        _intentos.pop(clave, None)


# ───────────────── Operaciones ─────────────────

def registrar(nombre, correo, password):
    nombre, correo, password = validar_registro(nombre, correo, password)
    try:
        usuario = db.crear_usuario(correo, nombre, generate_password_hash(password))
    except db.CorreoDuplicado:
        raise ErrorAuth("Ya existe una cuenta con ese correo. Inicia sesión.", 409)
    iniciar_sesion(usuario["id"])
    return usuario


def ingresar(correo, password):
    correo = normalizar_correo(correo)
    password = str(password or "")
    clave = _clave_intentos(correo)
    if _bloqueado(clave):
        raise ErrorAuth("Demasiados intentos fallidos. Espera unos minutos e inténtalo de nuevo.", 429)

    fila = db.usuario_por_correo(correo) if correo else None
    hash_guardado = fila["password_hash"] if fila else _HASH_FALSO
    correcto = check_password_hash(hash_guardado, password[:MAX_PASSWORD])
    if not fila or not correcto:
        _registrar_fallo(clave)
        raise ErrorAuth("Correo o contraseña incorrectos.", 401)

    _limpiar_fallos(clave)
    iniciar_sesion(fila["id"])
    return {"id": fila["id"], "correo": fila["correo"], "nombre": fila["nombre"]}


def verificar_password(usuario, password):
    """Comprueba la contraseña del usuario actual (para acciones delicadas)."""
    clave = _clave_intentos(f"cuenta:{usuario['id']}")
    if _bloqueado(clave):
        raise ErrorAuth("Demasiados intentos fallidos. Espera unos minutos e inténtalo de nuevo.", 429)
    fila = db.usuario_por_correo(usuario["correo"])
    if not fila or not check_password_hash(fila["password_hash"], str(password or "")[:MAX_PASSWORD]):
        _registrar_fallo(clave)
        raise ErrorAuth("La contraseña no es correcta.", 403)
    _limpiar_fallos(clave)


# ───────────────── Sesión ─────────────────

def iniciar_sesion(uid):
    session.clear()
    session["uid"] = uid
    session.permanent = True


def cerrar_sesion():
    session.clear()


def usuario_actual():
    """Usuario de la sesión (dict) o None. Se consulta una sola vez por petición."""
    if "usuario" in g:
        return g.usuario
    g.usuario = None
    uid = session.get("uid")
    if uid:
        g.usuario = db.usuario_por_id(uid)  # si la base falla, la excepción llega al manejador de app.py
        if g.usuario is None:
            session.clear()  # cuenta borrada
    return g.usuario


def login_required(vista):
    """Protege una vista: JSON 401 en /api/*, redirección a /acceso en las páginas."""
    @wraps(vista)
    def envoltura(*args, **kwargs):
        if usuario_actual() is None:
            if request.path.startswith("/api/"):
                return jsonify({"error": "Tu sesión terminó. Inicia sesión de nuevo."}), 401
            return redirect(url_for("acceso"))
        return vista(*args, **kwargs)
    return envoltura