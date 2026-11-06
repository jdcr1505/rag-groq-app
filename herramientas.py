"""
Tools del agente (Avance 3).

- buscar_normativa(consulta)                     -> búsqueda en la base de conocimientos (Chroma, local)
- calcular_plazo_habil(fecha_inicio, dias_habiles) -> tarea del dominio: plazo en días hábiles (Colombia)
- buscar_en_web(consulta)                         -> búsqueda web controlada (Tavily, solo sitios permitidos)

Los límites de cada tool están escritos aquí, en el código, y no en el prompt:
el LLM no los puede cambiar porque no son argumentos de la tool.

Prueba sin LLM:
    python herramientas.py
"""

import os
from datetime import date, datetime, timedelta
from urllib.parse import urlparse

from langchain_core.tools import tool

# ──────────────────────────────────────────────────────────────
# Límites (en el código, no en el prompt)
# ──────────────────────────────────────────────────────────────

MAX_FRAGMENTOS_POR_LLAMADA = 8      # top-k máximo por búsqueda en la base
MAX_CHARS_FRAGMENTO = 800           # longitud máxima de cada fragmento devuelto
MAX_CHARS_CONSULTA = 300            # una consulta más larga se recorta
MAX_LLAMADAS_POR_PREGUNTA = {       # cuántas veces puede usarse cada tool en una pregunta
    "buscar_normativa": 4,
    "calcular_plazo_habil": 3,
    "buscar_en_web": 2,
}

MAX_DIAS_HABILES = 365

WEB_MAX_RESULTADOS = 3
WEB_MAX_CHARS_EXTRACTO = 300
# Lista fija de sitios oficiales. Va dentro de la tool: el LLM solo escribe la consulta.
WEB_DOMINIOS_PERMITIDOS = (
    "secretariasenado.gov.co",   # Secretaría del Senado: leyes y códigos con notas de vigencia
    "suin-juriscol.gov.co",      # SUIN-Juriscol (MinJusticia): sistema oficial de normas
    "funcionpublica.gov.co",     # Gestor normativo de Función Pública
    "corteconstitucional.gov.co",  # sentencias de constitucionalidad
    "dian.gov.co",               # tributario: UVT, calendario y conceptos
    "sic.gov.co",                # consumidor y protección de datos personales
    "alcaldiabogota.gov.co",     # Régimen Legal de Bogotá: compilación de normas nacionales
)


# ──────────────────────────────────────────────────────────────
# Festivos de Colombia (Ley 51 de 1983 y Pascua)
# ──────────────────────────────────────────────────────────────

def _domingo_de_pascua(anio: int) -> date:
    """Algoritmo gregoriano anónimo (Meeus/Jones/Butcher)."""
    a = anio % 19
    b, c = divmod(anio, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes = (h + l - 7 * m + 114) // 31
    dia = (h + l - 7 * m + 114) % 31 + 1
    return date(anio, mes, dia)


def _siguiente_lunes(d: date) -> date:
    return d + timedelta(days=(7 - d.weekday()) % 7)


def festivos_colombia(anio: int) -> dict:
    """{fecha: nombre} con los 18 festivos nacionales del año."""
    fijos = {
        date(anio, 1, 1): "Año Nuevo",
        date(anio, 5, 1): "Día del Trabajo",
        date(anio, 7, 20): "Independencia",
        date(anio, 8, 7): "Batalla de Boyacá",
        date(anio, 12, 8): "Inmaculada Concepción",
        date(anio, 12, 25): "Navidad",
    }
    # Ley Emiliani (Ley 51 de 1983): se trasladan al lunes siguiente
    trasladables = {
        date(anio, 1, 6): "Reyes Magos",
        date(anio, 3, 19): "San José",
        date(anio, 6, 29): "San Pedro y San Pablo",
        date(anio, 8, 15): "Asunción de la Virgen",
        date(anio, 10, 12): "Día de la Raza",
        date(anio, 11, 1): "Todos los Santos",
        date(anio, 11, 11): "Independencia de Cartagena",
    }
    pascua = _domingo_de_pascua(anio)
    out = dict(fijos)
    out.update({_siguiente_lunes(d): n for d, n in trasladables.items()})
    out[pascua - timedelta(days=3)] = "Jueves Santo"
    out[pascua - timedelta(days=2)] = "Viernes Santo"
    out[pascua + timedelta(days=43)] = "Ascensión del Señor"
    out[pascua + timedelta(days=64)] = "Corpus Christi"
    out[pascua + timedelta(days=71)] = "Sagrado Corazón"
    return out


DIAS_SEMANA = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def calcular_vencimiento(fecha_inicio: date, dias_habiles: int) -> dict:
    """Cuenta días hábiles desde el día siguiente a fecha_inicio (sin sábados, domingos ni festivos)."""
    festivos = {}
    for a in range(fecha_inicio.year, fecha_inicio.year + 3):
        festivos.update(festivos_colombia(a))
    d, contados, excluidos = fecha_inicio, 0, []
    while contados < dias_habiles:
        d += timedelta(days=1)
        if d.weekday() >= 5:
            continue
        if d in festivos:
            excluidos.append(f"{d.isoformat()} ({festivos[d]})")
            continue
        contados += 1
    return {"vencimiento": d, "festivos_excluidos": excluidos,
            "dias_calendario": (d - fecha_inicio).days}


# ──────────────────────────────────────────────────────────────
# Formato de los fragmentos
# ──────────────────────────────────────────────────────────────

def _fuente(d) -> str:
    return os.path.basename(d.metadata.get("source", "?"))


def _ubicacion(d) -> str:
    p = d.metadata.get("page")
    if isinstance(p, int):
        return f"Pág. {p + 1}"
    return d.metadata.get("ubicacion") or "?"


def formatear_fragmentos(docs) -> str:
    return "\n\n---\n\n".join(
        f"[Fuente: {_fuente(d)} — {_ubicacion(d)}]\n{d.page_content}" for d in docs
    )


def _dominio_permitido(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in WEB_DOMINIOS_PERMITIDOS)


def web_search_disponible() -> bool:
    return bool(os.getenv("TAVILY_API_KEY", "").strip())


def _tavily(consulta: str) -> list:
    from tavily import TavilyClient  # import perezoso: la app funciona sin el paquete

    data = TavilyClient(api_key=os.environ["TAVILY_API_KEY"].strip()).search(
        query=consulta,
        max_results=WEB_MAX_RESULTADOS,
        include_domains=list(WEB_DOMINIOS_PERMITIDOS),
        search_depth="basic",
    )
    resultados = []
    for r in data.get("results", []):
        url = r.get("url", "")
        if not url or not _dominio_permitido(url):  # doble control por si el servicio no filtra
            continue
        resultados.append({
            "origen": "web",
            "titulo": (r.get("title") or "").strip(),
            "url": url,
            "texto": " ".join((r.get("content") or "").split())[:WEB_MAX_CHARS_EXTRACTO],
        })
    return resultados[:WEB_MAX_RESULTADOS]


# ──────────────────────────────────────────────────────────────
# Fábrica de tools
# ──────────────────────────────────────────────────────────────

def crear_tools(vector_store, k: int = MAX_FRAGMENTOS_POR_LLAMADA, usar_web: bool = False,
                registro: dict = None) -> list:
    """
    Crea las tools conectadas a la base vectorial. `registro` acumula lo que
    devolvió cada tool (fragmentos y resultados web) para mostrar las fuentes
    y para la evaluación con Ragas (retrieved_contexts).
    """
    registro = registro if registro is not None else {}
    registro.setdefault("docs", [])
    registro.setdefault("web", [])
    registro.setdefault("usos", {})
    k = max(1, min(int(k), MAX_FRAGMENTOS_POR_LLAMADA))

    def _cupo(nombre: str) -> bool:
        registro["usos"][nombre] = registro["usos"].get(nombre, 0) + 1
        return registro["usos"][nombre] <= MAX_LLAMADAS_POR_PREGUNTA[nombre]

    @tool
    def buscar_normativa(consulta: str) -> str:
        """Busca en la base de conocimientos (leyes y códigos colombianos del proyecto) los fragmentos más relevantes para la consulta.
        Úsala SIEMPRE PRIMERO para cualquier pregunta sobre normas, derechos, deberes, plazos legales, sanciones o trámites.
        Si la pregunta tiene dos o más temas, llámala una vez por tema.
        No la uses para saludos ni para conversación casual.
        Devuelve como máximo 8 fragmentos, cada uno con su documento y página o artículo para citar.

        Args:
            consulta: pregunta independiente y completa sobre un solo tema, en español.
        """
        if not _cupo("buscar_normativa"):
            return "Límite de búsquedas en la base alcanzado para esta pregunta. Responde con lo que ya tienes."
        consulta = str(consulta)[:MAX_CHARS_CONSULTA]
        docs = vector_store.similarity_search(consulta, k=k)
        for d in docs:
            d.page_content = d.page_content[:MAX_CHARS_FRAGMENTO]
        registro["docs"].extend(docs)
        return formatear_fragmentos(docs) or "No se encontraron fragmentos."

    @tool
    def calcular_plazo_habil(fecha_inicio: str, dias_habiles: int) -> str:
        """Calcula la fecha de vencimiento de un plazo legal en días hábiles en Colombia.
        Cuenta desde el día siguiente a fecha_inicio y excluye sábados, domingos y festivos nacionales (Ley 51 de 1983).
        Úsala cuando el usuario dé una fecha (de radicación, notificación, recibo, etc.) y pregunte hasta cuándo tiene plazo.
        Si no sabes cuántos días hábiles da la norma, primero búscalo con buscar_normativa.
        No considera vacancia judicial ni días de cierre propios de cada entidad.

        Args:
            fecha_inicio: fecha en formato AAAA-MM-DD (por ejemplo 2026-11-03).
            dias_habiles: número de días hábiles del plazo (entero de 1 a 365).
        """
        if not _cupo("calcular_plazo_habil"):
            return "Límite de cálculos alcanzado para esta pregunta."
        try:
            inicio = datetime.strptime(str(fecha_inicio).strip(), "%Y-%m-%d").date()
        except ValueError:
            return "Error: fecha_inicio debe tener el formato AAAA-MM-DD."
        if not 1900 <= inicio.year <= 2100:
            return "Error: el año debe estar entre 1900 y 2100."
        try:
            dias = int(dias_habiles)
        except (TypeError, ValueError):
            return "Error: dias_habiles debe ser un número entero."
        if not 1 <= dias <= MAX_DIAS_HABILES:
            return f"Error: dias_habiles debe estar entre 1 y {MAX_DIAS_HABILES}."
        r = calcular_vencimiento(inicio, dias)
        v = r["vencimiento"]
        festivos = ", ".join(r["festivos_excluidos"]) or "ninguno"
        return (
            f"[Cálculo: calcular_plazo_habil]\n"
            f"Fecha de inicio: {inicio.isoformat()} ({DIAS_SEMANA[inicio.weekday()]}). "
            f"Plazo: {dias} días hábiles, contados desde el día siguiente.\n"
            f"Vence el {v.isoformat()} ({DIAS_SEMANA[v.weekday()]}), {r['dias_calendario']} días calendario después.\n"
            f"Festivos excluidos en el periodo: {festivos}."
        )

    tools = [buscar_normativa, calcular_plazo_habil]

    if usar_web and web_search_disponible():
        @tool
        def buscar_en_web(consulta: str) -> str:
            """Busca información pública o actual en sitios oficiales colombianos (Secretaría del Senado, SUIN-Juriscol, Función Pública, Corte Constitucional, DIAN, SIC, Régimen Legal de Bogotá).
            Úsala SOLO si buscar_normativa no tiene la respuesta Y la pregunta trata de información pública o actual:
            una norma vigente o reformada, un valor que cambia cada año (UVT, salario mínimo), una definición general o una sentencia.
            Nunca envíes en la consulta texto copiado de los documentos de la base ni datos personales del usuario.
            Devuelve como máximo 3 resultados con título, URL y un extracto corto, marcados como origen web.

            Args:
                consulta: búsqueda corta en español (palabras clave, sin datos personales).
            """
            if not _cupo("buscar_en_web"):
                return "Límite de búsquedas web alcanzado para esta pregunta."
            try:
                resultados = _tavily(str(consulta)[:MAX_CHARS_CONSULTA])
            except Exception as e:
                return f"La búsqueda web falló: {type(e).__name__}."
            registro["web"].extend(resultados)
            if not resultados:
                return "La búsqueda web no devolvió resultados en los sitios permitidos."
            return "\n\n---\n\n".join(
                f"[origen: web | {r['titulo']} | {r['url']}]\n{r['texto']}" for r in resultados
            )

        tools.append(buscar_en_web)

    return tools


if __name__ == "__main__":
    # Prueba de las tools sin el LLM (.invoke() directo)
    import sys
    import rag_core

    sys.stdout.reconfigure(encoding="utf-8")
    store = rag_core.ensure_vector_store(progress_cb=print)
    t = {x.name: x for x in crear_tools(store, usar_web=web_search_disponible())}

    print("=== buscar_normativa.invoke ===")
    salida = t["buscar_normativa"].invoke({"consulta": "¿En cuántos días se debe resolver un derecho de petición?"})
    print(salida[:1800], "\n...")
    print("\n=== calcular_plazo_habil.invoke ===")
    print(t["calcular_plazo_habil"].invoke({"fecha_inicio": "2026-12-18", "dias_habiles": 15}))
    print("\n=== calcular_plazo_habil.invoke (fecha inválida) ===")
    print(t["calcular_plazo_habil"].invoke({"fecha_inicio": "18/12/2026", "dias_habiles": 15}))
    if "buscar_en_web" in t:
        print("\n=== buscar_en_web.invoke ===")
        print(t["buscar_en_web"].invoke({"consulta": "valor UVT 2026"}))
