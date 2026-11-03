"""
Capa de datos: usuarios, conversaciones y chats.

Producción (Cloud Run): PostgreSQL en Neon, indicado con la variable de entorno
DATABASE_URL (la cadena que entrega Neon). La conexión nunca se escribe en el código.

Desarrollo local: si DATABASE_URL no existe se usa un archivo SQLite (usuarios.db),
así la app corre sin instalar nada más. Las consultas son las mismas en ambos motores.

Modelo:
    usuarios        (id, correo, nombre, password_hash, creado)
    conversaciones  (id, user_id -> usuarios, titulo, creado, actualizado)
    chats           (id, conversacion_id -> conversaciones, user_id -> usuarios,
                     pregunta, respuesta, extra, creado)

Todas las llaves foráneas usan ON DELETE CASCADE: al borrar un usuario se borran sus
conversaciones y todos sus chats; al borrar una conversación se borran sus chats.
"""

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
SQLITE_PATH = os.environ.get("SQLITE_PATH", "usuarios.db")
USA_POSTGRES = DATABASE_URL.startswith(("postgres://", "postgresql://"))

if USA_POSTGRES:
    import psycopg
    from psycopg.rows import dict_row
    _ERRORES_UNICO = (psycopg.errors.UniqueViolation,)
else:
    _ERRORES_UNICO = (sqlite3.IntegrityError,)

MAX_TITULO = 80

_esquema_listo = False
_esquema_lock = threading.Lock()


# ───────────────── Esquema ─────────────────

def _ddl():
    ts = "TIMESTAMPTZ" if USA_POSTGRES else "TEXT"
    ahora = "now()" if USA_POSTGRES else "CURRENT_TIMESTAMP"
    return [
        f"""CREATE TABLE IF NOT EXISTS usuarios (
            id            TEXT PRIMARY KEY,
            correo        TEXT NOT NULL UNIQUE,
            nombre        TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            creado        {ts} NOT NULL DEFAULT {ahora}
        )""",
        f"""CREATE TABLE IF NOT EXISTS conversaciones (
            id          TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL REFERENCES usuarios(id) ON DELETE CASCADE,
            titulo      TEXT NOT NULL,
            creado      {ts} NOT NULL DEFAULT {ahora},
            actualizado {ts} NOT NULL DEFAULT {ahora}
        )""",
        f"""CREATE TABLE IF NOT EXISTS chats (
            id              TEXT PRIMARY KEY,
            conversacion_id TEXT NOT NULL REFERENCES conversaciones(id) ON DELETE CASCADE,
            user_id         TEXT NOT NULL REFERENCES usuarios(id) ON DELETE CASCADE,
            pregunta        TEXT NOT NULL,
            respuesta       TEXT NOT NULL,
            extra           TEXT,
            creado          {ts} NOT NULL DEFAULT {ahora}
        )""",
        "CREATE INDEX IF NOT EXISTS idx_conv_usuario ON conversaciones (user_id, actualizado)",
        "CREATE INDEX IF NOT EXISTS idx_chats_conv ON chats (conversacion_id, creado)",
        "CREATE INDEX IF NOT EXISTS idx_chats_usuario ON chats (user_id)",
    ]


# ───────────────── Conexión ─────────────────

class _Tx:
    """Una transacción: q() ejecuta SQL con marcadores ? y devuelve filas como dicts."""

    def __init__(self, conn):
        self._conn = conn

    def q(self, sql, params=()):
        if USA_POSTGRES:
            cur = self._conn.execute(sql.replace("?", "%s"), params)
            return cur.fetchall() if cur.description else []
        cur = self._conn.execute(sql, params)
        if cur.description is None:
            return []
        return [dict(fila) for fila in cur.fetchall()]

    def uno(self, sql, params=()):
        filas = self.q(sql, params)
        return filas[0] if filas else None


def _abrir():
    if USA_POSTGRES:
        # prepare_threshold=None: el endpoint "-pooler" de Neon usa PgBouncer, que no
        # admite sentencias preparadas del lado del servidor.
        return psycopg.connect(
            DATABASE_URL, row_factory=dict_row, prepare_threshold=None, connect_timeout=15
        )
    conn = sqlite3.connect(SQLITE_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")  # necesario para el ON DELETE CASCADE
    return conn


@contextmanager
def transaccion():
    """Abre una conexión corta; confirma al salir y revierte si hay una excepción.

    Neon suspende la base cuando no hay uso, y la primera conexión puede fallar mientras
    despierta: se reintenta un par de veces antes de rendirse.
    """
    conn = None
    for intento in range(3):
        try:
            conn = _abrir()
            break
        except Exception:
            if intento == 2:
                raise
            time.sleep(1.5 * (intento + 1))
    try:
        yield _Tx(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def iniciar():
    """Crea las tablas si no existen (idempotente). Se llama al arrancar y antes de usar la base."""
    global _esquema_listo
    if _esquema_listo:
        return
    with _esquema_lock:
        if _esquema_listo:
            return
        with transaccion() as tx:
            if USA_POSTGRES:
                # Evita carreras si varias instancias de Cloud Run arrancan a la vez
                tx.q("SELECT pg_advisory_xact_lock(727274)")
            for sentencia in _ddl():
                tx.q(sentencia)
        _esquema_listo = True


# ───────────────── Utilidades ─────────────────

def _nuevo_id():
    return uuid.uuid4().hex


def _ahora():
    """Marca de tiempo UTC: datetime para Postgres, texto ISO para SQLite."""
    ahora = datetime.now(timezone.utc)
    return ahora if USA_POSTGRES else ahora.isoformat()


def _iso(valor):
    return valor.isoformat() if isinstance(valor, datetime) else valor


def _conversacion_publica(fila):
    return {
        "id": fila["id"],
        "titulo": fila["titulo"],
        "creado": _iso(fila["creado"]),
        "actualizado": _iso(fila["actualizado"]),
    }


def titulo_desde_pregunta(pregunta):
    limpio = " ".join(str(pregunta).split())
    if len(limpio) <= 60:
        return limpio or "Conversación"
    return limpio[:59].rstrip() + "…"


# ───────────────── Usuarios ─────────────────

class CorreoDuplicado(Exception):
    pass


def crear_usuario(correo, nombre, password_hash):
    """Crea el usuario. Devuelve su fila (sin el hash). Lanza CorreoDuplicado si ya existe."""
    iniciar()
    uid = _nuevo_id()
    try:
        with transaccion() as tx:
            if tx.uno("SELECT id FROM usuarios WHERE correo = ?", (correo,)):
                raise CorreoDuplicado()
            tx.q(
                "INSERT INTO usuarios (id, correo, nombre, password_hash, creado) VALUES (?, ?, ?, ?, ?)",
                (uid, correo, nombre, password_hash, _ahora()),
            )
    except _ERRORES_UNICO:
        raise CorreoDuplicado()  # dos registros simultáneos con el mismo correo
    return {"id": uid, "correo": correo, "nombre": nombre}


def usuario_por_correo(correo):
    """Fila completa (incluye password_hash) o None."""
    iniciar()
    with transaccion() as tx:
        return tx.uno("SELECT id, correo, nombre, password_hash FROM usuarios WHERE correo = ?", (correo,))


def usuario_por_id(uid):
    iniciar()
    with transaccion() as tx:
        return tx.uno("SELECT id, correo, nombre FROM usuarios WHERE id = ?", (uid,))


def borrar_usuario(uid):
    """Borra el usuario; sus conversaciones y chats se eliminan por cascada."""
    iniciar()
    with transaccion() as tx:
        tx.q("DELETE FROM usuarios WHERE id = ?", (uid,))


# ───────────────── Conversaciones ─────────────────

def listar_conversaciones(uid):
    iniciar()
    with transaccion() as tx:
        filas = tx.q(
            "SELECT id, titulo, creado, actualizado FROM conversaciones "
            "WHERE user_id = ? ORDER BY actualizado DESC",
            (uid,),
        )
    return [_conversacion_publica(f) for f in filas]


def obtener_conversacion(uid, cid):
    """La conversación con todos sus turnos, solo si pertenece al usuario. Si no, None."""
    iniciar()
    with transaccion() as tx:
        conv = tx.uno(
            "SELECT id, titulo, creado, actualizado FROM conversaciones WHERE id = ? AND user_id = ?",
            (cid, uid),
        )
        if not conv:
            return None
        turnos = tx.q(
            "SELECT id, pregunta, respuesta, extra, creado FROM chats "
            "WHERE conversacion_id = ? AND user_id = ? ORDER BY creado ASC",
            (cid, uid),
        )
    resultado = _conversacion_publica(conv)
    resultado["turnos"] = [
        {
            "id": t["id"],
            "pregunta": t["pregunta"],
            "respuesta": t["respuesta"],
            "extra": json.loads(t["extra"]) if t["extra"] else {},
            "creado": _iso(t["creado"]),
        }
        for t in turnos
    ]
    return resultado


def existe_conversacion(uid, cid):
    iniciar()
    with transaccion() as tx:
        return tx.uno("SELECT id FROM conversaciones WHERE id = ? AND user_id = ?", (cid, uid)) is not None


def renombrar_conversacion(uid, cid, titulo):
    """Devuelve la conversación actualizada o None si no existe / no es del usuario."""
    iniciar()
    titulo = " ".join(str(titulo).split())[:MAX_TITULO]
    with transaccion() as tx:
        if not tx.uno("SELECT id FROM conversaciones WHERE id = ? AND user_id = ?", (cid, uid)):
            return None
        tx.q("UPDATE conversaciones SET titulo = ? WHERE id = ? AND user_id = ?", (titulo, cid, uid))
        fila = tx.uno("SELECT id, titulo, creado, actualizado FROM conversaciones WHERE id = ?", (cid,))
    return _conversacion_publica(fila)


def borrar_conversacion(uid, cid):
    """True si se borró. Sus chats desaparecen por cascada."""
    iniciar()
    with transaccion() as tx:
        if not tx.uno("SELECT id FROM conversaciones WHERE id = ? AND user_id = ?", (cid, uid)):
            return False
        tx.q("DELETE FROM conversaciones WHERE id = ? AND user_id = ?", (cid, uid))
    return True


# ───────────────── Chats ─────────────────

def historial_para_modelo(uid, cid, max_mensajes):
    """Últimos turnos de la conversación en el formato que espera el pipeline."""
    iniciar()
    turnos = max(1, max_mensajes // 2)
    with transaccion() as tx:
        filas = tx.q(
            "SELECT pregunta, respuesta, extra FROM chats "
            "WHERE conversacion_id = ? AND user_id = ? ORDER BY creado DESC LIMIT ?",
            (cid, uid, turnos),
        )
    historial = []
    for f in reversed(filas):
        extra = json.loads(f["extra"]) if f["extra"] else {}
        pregunta = f["pregunta"]
        if extra.get("descripcion_imagen"):
            # Igual que en la sesión original: la descripción de la imagen acompaña al turno
            pregunta = f"[Imagen adjunta: {extra['descripcion_imagen']}]\n{pregunta}"
        historial.append({"role": "user", "content": pregunta})
        historial.append({"role": "assistant", "content": f["respuesta"]})
    return historial


def guardar_turno(uid, cid, pregunta, respuesta, extra):
    """Guarda un turno. Si cid es None crea la conversación (título = inicio de la pregunta).

    Devuelve {"id", "titulo"} de la conversación y actualiza su fecha para ordenar la lista.
    """
    iniciar()
    ahora = _ahora()
    with transaccion() as tx:
        if cid is None:
            cid = _nuevo_id()
            titulo = titulo_desde_pregunta(pregunta)
            tx.q(
                "INSERT INTO conversaciones (id, user_id, titulo, creado, actualizado) VALUES (?, ?, ?, ?, ?)",
                (cid, uid, titulo, ahora, ahora),
            )
        else:
            conv = tx.uno("SELECT titulo FROM conversaciones WHERE id = ? AND user_id = ?", (cid, uid))
            if not conv:
                raise LookupError("La conversación no existe.")
            titulo = conv["titulo"]
            tx.q("UPDATE conversaciones SET actualizado = ? WHERE id = ?", (ahora, cid))
        tx.q(
            "INSERT INTO chats (id, conversacion_id, user_id, pregunta, respuesta, extra, creado) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (_nuevo_id(), cid, uid, pregunta, respuesta, json.dumps(extra, ensure_ascii=False), ahora),
        )
    return {"id": cid, "titulo": titulo}