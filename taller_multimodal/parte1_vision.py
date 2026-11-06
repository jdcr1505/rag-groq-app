"""
Parte 1. Visión sobre imágenes del dominio (normativa colombiana).

    python parte1_vision.py            # ejecuta 1.1, 1.2, 1.3 y 1.4
    python parte1_vision.py --punto 1.2

Requiere GROQ_API_KEY en el .env de la raíz. Modelo: VISION_MODEL (por defecto qwen/qwen3.8-27b).
Las imágenes se generan con generar_imagenes.py (documentos ficticios).
"""

import argparse
import json
import os
import re
import time
import unicodedata
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from comun import (IMAGENES, VISION_MODEL, TEXT_MODEL, AQUI, cliente, guardar, parte_imagen,
                   sin_razonamiento, titulo)


def img(nombre: str) -> str:
    return os.path.join(IMAGENES, nombre)


def preguntar(contenido: list, max_tokens: int = 1500, json_mode: bool = False) -> str:
    """Una solicitud al modelo de visión. Devuelve el texto de la respuesta."""
    kwargs = dict(model=VISION_MODEL, temperature=0, max_tokens=max_tokens,
                  messages=[{"role": "user", "content": contenido}])
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    c = cliente()
    try:
        r = c.chat.completions.create(reasoning_format="hidden", **kwargs)
    except TypeError:  # versión del SDK sin reasoning_format
        r = c.chat.completions.create(**kwargs)
    except Exception as e:
        if "reasoning_format" not in str(e) and "response_format" not in str(e):
            raise
        kwargs.pop("response_format", None)
        r = c.chat.completions.create(**kwargs)
    return sin_razonamiento(r.choices[0].message.content)


# ───────────────────────── 1.1 Pregunta sobre una imagen ─────────────────────────

def punto_1_1():
    titulo("1.1 Pregunta sobre una imagen")
    imagen = "02_factura_compra.jpg"
    pregunta = ("Soy el cliente de esta factura. ¿Qué producto compré, cuánto pagué en total y hasta qué fecha "
                "tengo garantía según lo que dice el documento? ¿Puedo ejercer el derecho de retracto?")
    t0 = time.perf_counter()
    respuesta = preguntar([{"type": "text", "text": pregunta}, parte_imagen(img(imagen))])
    print(f"Imagen: {imagen}\nPregunta: {pregunta}\n\nRespuesta del modelo ({VISION_MODEL}):\n{respuesta}")
    guardar("1_1_pregunta_imagen.json", {"modelo": VISION_MODEL, "imagen": imagen, "pregunta": pregunta,
                                         "respuesta": respuesta, "tiempo_s": round(time.perf_counter() - t0, 2)})


# ───────────────────────── 1.2 Extracción estructurada ─────────────────────────

class DocumentoJuridico(BaseModel):
    """Esquema de extracción para documentos jurídicos del dominio."""
    tipo_documento: str = Field(min_length=3, description="Clase de documento (derecho de petición, factura, contrato, acto administrativo...)")
    emisor: str = Field(min_length=2, description="Persona o entidad que expide o firma el documento")
    destinatario: str = Field(min_length=2, description="Persona o entidad a la que va dirigido")
    fecha: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="Fecha de expedición o radicación, AAAA-MM-DD")
    numero_referencia: str = Field(min_length=1, description="Número de radicado, factura o acto")
    valor_cop: Optional[int] = Field(default=None, ge=0, description="Valor total en pesos colombianos, entero sin puntos; null si no hay")
    norma_citada: str = Field(min_length=3, description="Norma principal que cita el documento")


CAMPOS = list(DocumentoJuridico.model_fields)

PROMPT_EXTRACCION = """Extrae los datos del documento jurídico de la imagen y responde SOLO con un objeto JSON válido, sin texto adicional, con exactamente estas claves:
{esquema}
Reglas: fecha en formato AAAA-MM-DD; valor_cop como número entero sin puntos ni símbolos (el valor total del documento) o null si no aparece ningún valor; si un dato no aparece usa null."""


def extraer_json(texto: str):
    """Saca el primer objeto JSON del texto (por si el modelo agrega ``` o comentarios)."""
    texto = re.sub(r"^```(?:json)?|```$", "", texto.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", texto, flags=re.DOTALL)
    if not m:
        raise ValueError("la respuesta no contiene un objeto JSON")
    return json.loads(m.group(0))


def validar(texto: str):
    """Devuelve (datos_validos | None, error | None). Nunca lanza excepción."""
    try:
        bruto = extraer_json(texto)
    except (ValueError, json.JSONDecodeError) as e:
        return None, f"JSON mal formado: {e}"
    try:
        return DocumentoJuridico.model_validate(bruto).model_dump(), None
    except ValidationError as e:
        errores = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
        return None, f"No cumple el esquema: {errores}"


def normalizar(v) -> str:
    s = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\bs\.?\s?a\.?\s?s\.?", "sas", s)
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split())


def coincide(campo: str, real, extraido) -> bool:
    """Regla de comparación: fechas, números y referencias exactos; textos tras normalizar
    (minúsculas, sin tildes ni signos), aceptando que uno contenga al otro."""
    if real is None or extraido is None:
        return real is None and extraido is None
    if campo == "valor_cop":
        return int(real) == int(extraido)
    if campo in ("fecha", "numero_referencia"):
        return re.sub(r"[^0-9A-Za-z]", "", str(real)).lower() == re.sub(r"[^0-9A-Za-z]", "", str(extraido)).lower()
    a, b = normalizar(real), normalizar(extraido)
    return bool(a) and bool(b) and (a in b or b in a)


def punto_1_2():
    titulo("1.2 Extracción estructurada")
    with open(os.path.join(AQUI, "valores_reales.json"), encoding="utf-8") as f:
        reales = json.load(f)
    esquema = json.dumps({c: DocumentoJuridico.model_fields[c].description for c in CAMPOS},
                         ensure_ascii=False, indent=1)
    prompt = PROMPT_EXTRACCION.format(esquema=esquema)

    tabla, salidas = [], []
    for nombre, real in reales.items():
        texto = preguntar([{"type": "text", "text": prompt}, parte_imagen(img(nombre))], json_mode=True)
        datos, error = validar(texto)
        print(f"\n--- {nombre}\nSalida del modelo: {texto}")
        if error:
            print(f"VALIDACIÓN: {error}  -> se cuentan todos los campos como incorrectos")
        else:
            print("VALIDACIÓN: el JSON cumple el esquema")
        salidas.append({"imagen": nombre, "salida": texto, "valido": error is None, "error": error})
        for c in CAMPOS:
            extraido = datos.get(c) if datos else None
            ok = datos is not None and coincide(c, real.get(c), extraido)
            tabla.append({"imagen": nombre, "campo": c, "valor_real": real.get(c),
                          "valor_extraido": extraido, "correcto": ok})

    aciertos = sum(f["correcto"] for f in tabla)
    pct = 100 * aciertos / len(tabla)
    print("\n| Imagen | Campo | Valor real | Valor extraído | ¿Correcto? |\n|---|---|---|---|---|")
    for f in tabla:
        print(f"| {f['imagen']} | {f['campo']} | {f['valor_real']} | {f['valor_extraido']} | {'Sí' if f['correcto'] else 'No'} |")
    print(f"\nPorcentaje de aciertos: {aciertos} de {len(tabla)} campos = {pct:.1f} %")
    guardar("1_2_extraccion.json", {"modelo": VISION_MODEL, "esquema": esquema, "salidas": salidas,
                                    "tabla": tabla, "aciertos": aciertos, "total": len(tabla),
                                    "porcentaje": round(pct, 1)})


def demo_validacion():
    """Muestra que un JSON inválido se informa en lugar de romper el programa."""
    titulo("1.2 (demostración) Validación de un JSON que no cumple el esquema")
    malo = '{"tipo_documento": "factura", "emisor": "X", "fecha": "28/10/2026", "valor_cop": "dos millones"}'
    print("Entrada:", malo)
    print("Resultado:", validar(malo)[1])


# ───────────────────────── 1.3 Varias imágenes en una solicitud ─────────────────────────

def punto_1_3():
    titulo("1.3 Varias imágenes en una solicitud")
    imagenes = ["01_derecho_peticion.jpg", "04_articulo_14_cpaca.jpg"]
    pregunta = ("La primera imagen es un derecho de petición radicado ante una entidad y la segunda es el "
                "artículo 14 del CPACA. Según el artículo, ¿qué tipo de petición es la de la primera imagen, "
                "cuántos días tiene la entidad para responderla y desde qué fecha se cuenta ese plazo? "
                "¿Qué pasa si la entidad no responde a tiempo?")
    contenido = [{"type": "text", "text": pregunta}] + [parte_imagen(img(n)) for n in imagenes]
    respuesta = preguntar(contenido, max_tokens=2000)
    print(f"Imágenes: {imagenes}\nPregunta: {pregunta}\n\nRespuesta del modelo:\n{respuesta}")
    guardar("1_3_varias_imagenes.json", {"modelo": VISION_MODEL, "imagenes": imagenes,
                                         "pregunta": pregunta, "respuesta": respuesta})


# ───────────────────────── 1.4 Error provocado ─────────────────────────

def punto_1_4():
    titulo("1.4 Errores provocados")
    todas = sorted(os.listdir(IMAGENES))
    pruebas = [
        ("Más imágenes de las permitidas (4 en una solicitud; el límite es 3)", VISION_MODEL,
         [{"type": "text", "text": "Resume estos documentos."}] + [parte_imagen(img(n)) for n in todas[:4]]),
        ("Modelo sin visión (openai/gpt-oss-120b) recibiendo una imagen", TEXT_MODEL,
         [{"type": "text", "text": "¿Qué dice este documento?"}, parte_imagen(img(todas[0]))]),
        ("Tipo de imagen mal declarado (JPEG enviado como image/gif)", VISION_MODEL,
         [{"type": "text", "text": "¿Qué dice este documento?"}, parte_imagen(img(todas[0]), mime="image/gif")]),
    ]
    resultados = []
    for nombre, modelo, contenido in pruebas:
        print(f"\n--- {nombre}")
        try:
            r = cliente().chat.completions.create(model=modelo, max_tokens=300,
                                                  messages=[{"role": "user", "content": contenido}])
            msg = "SIN ERROR. Respuesta: " + sin_razonamiento(r.choices[0].message.content)[:300]
            status = 200
        except Exception as e:
            status = getattr(e, "status_code", None)
            cuerpo = getattr(e, "body", None)
            msg = f"{type(e).__name__} (HTTP {status}): {json.dumps(cuerpo, ensure_ascii=False) if cuerpo else e}"
        print(msg)
        resultados.append({"prueba": nombre, "modelo": modelo, "http_status": status, "mensaje": msg})
    guardar("1_4_errores.json", resultados)


PUNTOS = {"1.1": punto_1_1, "1.2": punto_1_2, "1.3": punto_1_3, "1.4": punto_1_4}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--punto", choices=list(PUNTOS) + ["todo"], default="todo")
    args = ap.parse_args()
    if not os.path.isdir(IMAGENES) or not os.listdir(IMAGENES):
        raise SystemExit("No hay imágenes: ejecuta primero python generar_imagenes.py")
    for p, f in PUNTOS.items():
        if args.punto in (p, "todo"):
            f()
            if p == "1.2":
                demo_validacion()
