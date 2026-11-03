"""
Lectores de documentos .docx y .tex para la base de conocimiento.

Solo usan la biblioteca estándar (sin dependencias nuevas). Un .docx o un .tex
no tienen páginas fijas (la paginación se calcula al abrir o compilar el
archivo), así que aquí el texto se divide en SECCIONES y cada sección recibe
una "ubicación" que se usa para citar:

    "Art. 5"      -> cuando el texto tiene líneas "Artículo N. ..."
    "Inicio"      -> texto anterior al primer artículo (encabezado)
    <título>      -> si no hay artículos pero sí títulos (estilos de título
                     en Word, \\section en LaTeX)
    "Texto completo" -> si no hay ni artículos ni títulos

Los PDF no pasan por aquí: conservan su número de página real.
"""

import re
import unicodedata
import zipfile
import xml.etree.ElementTree as ET

# "Artículo 5.", "ARTÍCULO 555-2", "Artículo 1o." ... al inicio de línea
ART_RE = re.compile(r"^\s*(?i:art[íi]culo)\s+(\d+(?:-\d+)?[A-Z]?)[oº°]?(?![A-Za-z0-9])")

MAX_BYTES_XML = 50 * 1024 * 1024  # protección básica ante archivos .docx enormes

# Un elemento es (es_titulo, texto)
Item = tuple


# ──────────────────────────────────────────────────────────────
# Agrupación común: items -> [(ubicacion, texto)]
# ──────────────────────────────────────────────────────────────

def agrupar_secciones(items: list) -> list:
    items = [(t, x.strip()) for t, x in items if x and x.strip()]
    if not items:
        return []

    hay_articulos = any(ART_RE.match(x) for _, x in items)
    hay_titulos = any(t for t, _ in items)

    secciones, ubicacion, buffer = [], None, []

    def cerrar():
        if buffer:
            secciones.append((ubicacion or "Inicio", "\n\n".join(buffer)))

    for es_titulo, texto in items:
        m = ART_RE.match(texto)
        if m:
            cerrar()
            ubicacion, buffer = f"Art. {m.group(1).upper()}", [texto]
        elif not hay_articulos and es_titulo:
            cerrar()
            ubicacion, buffer = texto[:60], [texto]
        else:
            buffer.append(texto)
    cerrar()

    if not hay_articulos and not hay_titulos:
        return [("Texto completo", "\n\n".join(x for _, x in items))]
    return secciones


# ──────────────────────────────────────────────────────────────
# .docx
# ──────────────────────────────────────────────────────────────

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _texto_parrafo(p) -> str:
    partes = []
    for el in p.iter():
        if el.tag == _W + "t" and el.text:
            partes.append(el.text)
        elif el.tag in (_W + "tab", _W + "br", _W + "cr"):
            partes.append(" ")
    return re.sub(r"\s+", " ", "".join(partes)).strip()


def _es_titulo_docx(p) -> bool:
    ppr = p.find(_W + "pPr")
    if ppr is None:
        return False
    estilo = ppr.find(_W + "pStyle")
    if estilo is not None:
        nombre = (estilo.get(_W + "val") or "").lower()
        if nombre.startswith(("heading", "ttulo", "titulo", "title")):
            return True
    return ppr.find(_W + "outlineLvl") is not None


def leer_docx(ruta: str) -> list:
    """Devuelve [(ubicacion, texto)] a partir de un .docx."""
    with zipfile.ZipFile(ruta) as z:
        info = z.getinfo("word/document.xml")
        if info.file_size > MAX_BYTES_XML:
            raise ValueError(f"{ruta}: el documento es demasiado grande.")
        raiz = ET.fromstring(z.read("word/document.xml"))

    cuerpo = raiz.find(_W + "body")
    items = []
    for el in (cuerpo if cuerpo is not None else []):
        if el.tag == _W + "p":
            items.append((_es_titulo_docx(el), _texto_parrafo(el)))
        elif el.tag == _W + "tbl":
            for fila in el.iter(_W + "tr"):
                celdas = [
                    " ".join(filter(None, (_texto_parrafo(p) for p in c.iter(_W + "p"))))
                    for c in fila.findall(_W + "tc")
                ]
                items.append((False, " | ".join(c for c in celdas if c)))
        elif el.tag == _W + "sdt":  # controles de contenido
            for p in el.iter(_W + "p"):
                items.append((False, _texto_parrafo(p)))
    return agrupar_secciones(items)


# ──────────────────────────────────────────────────────────────
# .tex
# ──────────────────────────────────────────────────────────────

_MARCA = "\x00T\x00"
_ACENTOS = {"'": "\u0301", "`": "\u0300", "^": "\u0302", '"': "\u0308", "~": "\u0303"}
_ARG = r"((?:[^{}]|\{[^{}]*\})*)"  # argumento entre llaves (admite un nivel anidado)


def _acento(m) -> str:
    return unicodedata.normalize("NFC", m.group(2) + _ACENTOS[m.group(1)])


def _tex_a_items(tex: str) -> list:
    tex = re.sub(r"(?<!\\)%.*", "", tex)  # comentarios

    titulo = None
    m = re.search(r"\\title\s*\{" + _ARG + r"\}", tex)
    if m:
        titulo = m.group(1)

    ini, fin = tex.find("\\begin{document}"), tex.rfind("\\end{document}")
    if ini != -1:
        tex = tex[ini + len("\\begin{document}"): fin if fin != -1 else len(tex)]

    # Encabezados -> línea marcada como título
    tex = re.sub(
        r"\\(?:part|chapter|section|subsection|subsubsection|paragraph)\*?(?:\[[^\]]*\])?\s*\{" + _ARG + r"\}",
        lambda m: f"\n\n{_MARCA}{m.group(1)}\n\n",
        tex,
    )

    # Comandos que se descartan junto con su argumento
    tex = re.sub(r"\\(?:label|ref|eqref|cite|index|hspace|vspace|setlength|pagestyle|thispagestyle)\*?\s*\{[^{}]*\}", "", tex)
    tex = re.sub(r"\\href\s*\{[^{}]*\}\s*\{" + _ARG + r"\}", r"\1", tex)
    tex = re.sub(r"\\footnote\s*\{" + _ARG + r"\}", r" (\1)", tex)
    tex = re.sub(r"\\(?:maketitle|tableofcontents|newpage|clearpage|noindent|centering|hline|bigskip|medskip|smallskip)\b", "", tex)

    # Tablas: se descarta la especificación de columnas ({ll}, {|c|c|}...)
    tex = re.sub(r"\\begin\s*\{(?:tabular|tabularx|array|longtable)\*?\}(?:\[[^\]]*\])?\s*\{[^{}]*\}", "\n\n", tex)

    # Listas y entornos
    tex = re.sub(r"\\item\b(?:\s*\[([^\]]*)\])?", lambda m: f"\n\n- {m.group(1) + ' ' if m.group(1) else ''}", tex)
    tex = re.sub(r"\\(?:begin|end)\s*\{[^{}]*\}(?:\[[^\]]*\])?", "\n\n", tex)
    tex = tex.replace("\\par", "\n\n")

    # Tablas simples: & separa celdas, \\ separa filas
    tex = re.sub(r"(?<!\\)&", " | ", tex).replace("\\\\", "\n\n")

    # Acentos y caracteres especiales
    tex = re.sub(r"\\(['`^\"~])\s*\{?\\?([A-Za-z])\}?", _acento, tex)
    for esc, car in (("\\%", "%"), ("\\&", "&"), ("\\_", "_"), ("\\$", "$"), ("\\#", "#"),
                     ("\\{", "{"), ("\\}", "}"), ("\\ldots", "…"),
                     ("\\,", " "), ("\\;", " "), ("\\!", ""), ("\\ ", " ")):
        tex = tex.replace(esc, car)
    tex = tex.replace("---", "—").replace("--", "–").replace("``", "“").replace("''", "”").replace("~", " ")

    # Formato en línea (\textbf{x}, \emph{x}...) -> x ; se repite para anidados
    previo = None
    while previo != tex:
        previo = tex
        tex = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])*\s*\{([^{}]*)\}", r"\1", tex)
    tex = re.sub(r"\\[A-Za-z]+\*?", "", tex)  # comandos sin argumento que sobren
    tex = tex.replace("{", "").replace("}", "")

    items = []
    if titulo:
        t = titulo.replace("``", "“").replace("''", "”")
        t = re.sub(r"\\[A-Za-z]+\*?|[{}]|\\\\", " ", t)
        items.append((True, re.sub(r"\s+", " ", t).strip()))
    for bloque in re.split(r"\n\s*\n", tex):
        es_titulo = bloque.strip().startswith(_MARCA)
        limpio = re.sub(r"\s+", " ", bloque.replace(_MARCA, "")).strip()
        if limpio:
            items.append((es_titulo, limpio))
    return items


def leer_tex(ruta: str) -> list:
    """Devuelve [(ubicacion, texto)] a partir de un .tex."""
    with open(ruta, "rb") as f:
        crudo = f.read()
    try:
        tex = crudo.decode("utf-8")
    except UnicodeDecodeError:
        tex = crudo.decode("latin-1")
    return agrupar_secciones(_tex_a_items(tex))
