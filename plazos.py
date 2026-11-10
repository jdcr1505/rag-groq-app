"""
Cálculo de plazos con festivos de Colombia (solo biblioteca estándar).

Lo usa la tool `calcular_plazo` de rag_core.py. Está separado para poder
probarlo sin LLM ni base vectorial.

Convención (se documenta también en el README y en la tool):
  - El conteo empieza el día SIGUIENTE a la fecha de inicio.
  - Días hábiles: lunes a viernes, excluyendo los festivos de Colombia.
  - Días calendario: todos los días corridos.
Festivos: Ley 51 de 1983 (Ley Emiliani): los festivos "trasladables" pasan al
lunes siguiente; los de Semana Santa dependen de la fecha de Pascua.
Es orientativo: el plazo legal real puede tener reglas propias (por ejemplo,
días de vacancia judicial o términos de meses y años).
"""

from datetime import date, timedelta

MAX_DIAS = 3650  # límite de seguridad: 10 años


def _pascua(anio: int) -> date:
    """Domingo de Pascua (algoritmo gregoriano de Meeus/Jones/Butcher)."""
    a = anio % 19
    b, c = divmod(anio, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes, dia = divmod(h + l - 7 * m + 114, 31)
    return date(anio, mes, dia + 1)


def _al_lunes(d: date) -> date:
    """Traslada al lunes siguiente (si ya es lunes, se queda)."""
    return d + timedelta(days=(7 - d.weekday()) % 7)


def festivos_colombia(anio: int) -> dict:
    """{fecha: nombre} de los festivos de Colombia en ese año."""
    f = {}
    # Fijos
    for (m, d), nombre in {
        (1, 1): "Año Nuevo", (5, 1): "Día del Trabajo", (7, 20): "Día de la Independencia",
        (8, 7): "Batalla de Boyacá", (12, 8): "Inmaculada Concepción", (12, 25): "Navidad",
    }.items():
        f[date(anio, m, d)] = nombre
    # Trasladables al lunes (Ley Emiliani)
    for (m, d), nombre in {
        (1, 6): "Reyes Magos", (3, 19): "San José", (6, 29): "San Pedro y San Pablo",
        (8, 15): "Asunción de la Virgen", (10, 12): "Día de la Raza",
        (11, 1): "Todos los Santos", (11, 11): "Independencia de Cartagena",
    }.items():
        f[_al_lunes(date(anio, m, d))] = nombre
    # Semana Santa y fiestas que dependen de la Pascua
    p = _pascua(anio)
    f[p - timedelta(days=3)] = "Jueves Santo"
    f[p - timedelta(days=2)] = "Viernes Santo"
    f[_al_lunes(p + timedelta(days=39))] = "Ascensión del Señor"
    f[_al_lunes(p + timedelta(days=60))] = "Corpus Christi"
    f[_al_lunes(p + timedelta(days=68))] = "Sagrado Corazón"
    return f


def calcular(fecha_inicio: str, cantidad_dias: int, tipo: str) -> dict:
    """
    Devuelve {fecha_fin, omitidos:[{fecha, motivo}], ...}.
    Lanza ValueError con un mensaje claro si los argumentos no son válidos.
    """
    try:
        inicio = date.fromisoformat(str(fecha_inicio).strip())
    except ValueError:
        raise ValueError("fecha_inicio debe tener formato AAAA-MM-DD, por ejemplo 2026-03-15.")
    tipo = str(tipo).strip().lower()
    if tipo in ("habil", "hábiles", "habiles", "hábil"):
        tipo = "habiles"
    if tipo in ("calendario", "calendarios", "corridos"):
        tipo = "calendario"
    if tipo not in ("habiles", "calendario"):
        raise ValueError('tipo debe ser "habiles" o "calendario".')
    try:
        n = int(cantidad_dias)
    except (TypeError, ValueError):
        raise ValueError("cantidad_dias debe ser un número entero.")
    if not 1 <= n <= MAX_DIAS:
        raise ValueError(f"cantidad_dias debe estar entre 1 y {MAX_DIAS}.")

    if tipo == "calendario":
        return {"fecha_inicio": inicio, "cantidad": n, "tipo": tipo,
                "fecha_fin": inicio + timedelta(days=n), "omitidos": []}

    cache, omitidos, actual, contados = {}, [], inicio, 0
    while contados < n:
        actual += timedelta(days=1)
        if actual.year not in cache:
            cache[actual.year] = festivos_colombia(actual.year)
        if actual.weekday() >= 5:
            continue  # sábado o domingo (no se listan: son obvios)
        if actual in cache[actual.year]:
            omitidos.append({"fecha": actual, "motivo": cache[actual.year][actual]})
            continue
        contados += 1
    return {"fecha_inicio": inicio, "cantidad": n, "tipo": tipo,
            "fecha_fin": actual, "omitidos": omitidos}


DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def texto_resultado(r: dict) -> str:
    """Resultado legible que recibe el LLM como ToolMessage."""
    fin = r["fecha_fin"]
    lineas = [
        f"Fecha de inicio: {r['fecha_inicio'].isoformat()} (el conteo empieza el día siguiente).",
        f"Plazo: {r['cantidad']} días {'hábiles (lunes a viernes, sin festivos de Colombia)' if r['tipo'] == 'habiles' else 'calendario'}.",
        f"Fecha de vencimiento: {fin.isoformat()} ({DIAS[fin.weekday()]}).",
    ]
    if r["tipo"] == "habiles":
        if r["omitidos"]:
            lineas.append("Festivos omitidos en el conteo: " + "; ".join(
                f"{o['fecha'].isoformat()} ({o['motivo']})" for o in r["omitidos"]) + ".")
        else:
            lineas.append("No hubo festivos dentro del conteo.")
    lineas.append("Cálculo orientativo: verifica si la norma aplicable fija reglas propias de conteo.")
    return "\n".join(lineas)