"""
Genera las imágenes del dominio (documentos jurídicos FICTICIOS) que usa la Parte 1.

Todas llevan la marca "DOCUMENTO FICTICIO - EJERCICIO ACADÉMICO": las entidades,
personas, números y valores son inventados. El texto del artículo de ley es un
extracto de una norma pública. Se les aplica una leve rotación, ruido y tono
de papel para que parezcan escaneadas.

Los valores reales de cada documento quedan en valores_reales.json (lo que una
persona lee en la imagen), para comparar con lo que extrae el modelo.

    python generar_imagenes.py
"""

import json
import os
import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont

AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(AQUI, "imagenes")
FUENTES = "C:/Windows/Fonts"
W, H = 1240, 1600  # ~A4 a 150 ppp


def fuente(nombre: str, tam: int):
    for n in (nombre, "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(os.path.join(FUENTES, n), tam)
        except OSError:
            try:
                return ImageFont.truetype(n, tam)
            except OSError:
                continue
    return ImageFont.load_default()


SERIF = lambda t: fuente("times.ttf", t)
SERIF_B = lambda t: fuente("timesbd.ttf", t)
SANS = lambda t: fuente("arial.ttf", t)
SANS_B = lambda t: fuente("arialbd.ttf", t)
MONO = lambda t: fuente("cour.ttf", t)


def envolver(draw, texto, f, ancho):
    lineas, actual = [], ""
    for palabra in texto.split():
        prueba = (actual + " " + palabra).strip()
        if draw.textlength(prueba, font=f) <= ancho:
            actual = prueba
        else:
            lineas.append(actual)
            actual = palabra
    if actual:
        lineas.append(actual)
    return lineas


class Hoja:
    def __init__(self):
        self.img = Image.new("RGB", (W, H), (252, 250, 244))
        self.d = ImageDraw.Draw(self.img)
        self.y = 90

    def texto(self, t, f, x=110, color=(25, 25, 25), salto=10, ancho=W - 220, centrado=False):
        for linea in envolver(self.d, t, f, ancho):
            if centrado:
                x_ = (W - self.d.textlength(linea, font=f)) / 2
            else:
                x_ = x
            self.d.text((x_, self.y), linea, font=f, fill=color)
            self.y += f.size + salto
        return self

    def espacio(self, n=18):
        self.y += n
        return self

    def linea(self, grosor=2):
        self.d.line((110, self.y, W - 110, self.y), fill=(60, 60, 60), width=grosor)
        self.y += 16
        return self

    def tabla(self, filas, cols, f, alto=46):
        x0 = 110
        for i, fila in enumerate(filas):
            x = x0
            for j, celda in enumerate(fila):
                self.d.rectangle((x, self.y, x + cols[j], self.y + alto), outline=(70, 70, 70), width=2)
                self.d.text((x + 10, self.y + 11), str(celda), font=SANS_B(f.size) if i == 0 else f, fill=(25, 25, 25))
                x += cols[j]
            self.y += alto
        self.y += 14
        return self

    def sello(self, texto, x, y, color=(170, 30, 40)):
        capa = Image.new("RGBA", (420, 170), (0, 0, 0, 0))
        dd = ImageDraw.Draw(capa)
        dd.rounded_rectangle((6, 6, 414, 164), radius=18, outline=color + (210,), width=6)
        for i, t in enumerate(texto.split("\n")):
            f = SANS_B(30 if i == 0 else 24)
            dd.text(((420 - dd.textlength(t, font=f)) / 2, 22 + i * 40), t, font=f, fill=color + (215,))
        capa = capa.rotate(random.uniform(-9, 9), expand=True, resample=Image.BICUBIC)
        self.img.paste(capa, (x, y), capa)
        return self

    def marca_ficticio(self):
        f = SANS_B(22)
        t = "DOCUMENTO FICTICIO - EJERCICIO ACADÉMICO"
        self.d.text(((W - self.d.textlength(t, font=f)) / 2, H - 70), t, font=f, fill=(150, 150, 150))
        return self

    def escanear(self, ruta, angulo=None):
        img = self.img.rotate(angulo if angulo is not None else random.uniform(-1.2, 1.2),
                              expand=False, fillcolor=(235, 233, 226), resample=Image.BICUBIC)
        ruido = Image.effect_noise((W, H), 14).convert("RGB")
        img = Image.blend(img, ruido, 0.06).filter(ImageFilter.GaussianBlur(0.5))
        img.save(ruta, quality=88)


def doc_peticion():
    h = Hoja()
    h.texto("Barranquilla, 18 de noviembre de 2026", SERIF(30)).espacio(30)
    h.texto("Señores", SERIF(30), salto=4).texto("ALCALDÍA DISTRITAL DE VILLA SERENA", SERIF_B(30), salto=4)
    h.texto("Secretaría de Movilidad", SERIF(30), salto=4).texto("Ciudad", SERIF(30)).espacio(24)
    h.texto("Referencia: DERECHO DE PETICIÓN DE INFORMACIÓN", SERIF_B(30)).espacio(16)
    h.texto("Yo, MARÍA FERNANDA ROJAS PÉREZ, identificada con cédula de ciudadanía No. 1.045.678.912, "
            "en ejercicio del derecho fundamental de petición consagrado en el artículo 23 de la Constitución "
            "Política y regulado por la Ley 1755 de 2015, respetuosamente solicito:", SERIF(29), salto=8).espacio(10)
    h.texto("1. Copia del comparendo electrónico No. 08001000000045821 y de la prueba fotográfica que lo sustenta.", SERIF(29), x=150, ancho=W - 260, salto=8)
    h.texto("2. Información sobre el estado del proceso contravencional y la fecha de la audiencia.", SERIF(29), x=150, ancho=W - 260, salto=8).espacio(14)
    h.texto("Recibiré notificaciones en el correo mfrojas.ejemplo@correo.com y en la Calle 72 No. 45-18, apartamento 302.", SERIF(29), salto=8).espacio(40)
    h.texto("Atentamente,", SERIF(29)).espacio(60)
    h.texto("María Fernanda Rojas Pérez", SERIF_B(29), salto=4).texto("C.C. 1.045.678.912", SERIF(29))
    h.sello("RADICADO\nNo. 2026-ER-004521\n18/11/2026  10:42", 760, 60)
    h.marca_ficticio()
    return h


def doc_factura():
    h = Hoja()
    h.texto("ELECTROHOGAR DEL CARIBE S.A.S.", SANS_B(40), centrado=True, salto=4)
    h.texto("NIT 901.234.567-8 · Régimen común · Cra. 54 No. 68-120, Barranquilla", SANS(24), centrado=True).espacio(10)
    h.linea()
    h.texto("FACTURA ELECTRÓNICA DE VENTA No. FE-10234", SANS_B(32), salto=6)
    h.texto("Fecha de expedición: 28/10/2026      Forma de pago: Contado (tarjeta débito)", SANS(26), salto=6)
    h.texto("Cliente: Juan Camilo Ortega Díaz · C.C. 72.345.901", SANS(26)).espacio(10)
    h.tabla([
        ["Ref.", "Descripción", "Cant.", "Valor unitario", "Total"],
        ["TV-55U8", "Televisor LED 55\" 4K UHD", "1", "$ 1.974.706", "$ 1.974.706"],
        ["SOP-09", "Soporte de pared articulado", "1", "$ 0", "$ 0"],
    ], [150, 440, 100, 160, 170], SANS(23))
    h.texto("Subtotal: $ 1.974.706", SANS(27), x=700, salto=6)
    h.texto("IVA 19 %: $ 375.194", SANS(27), x=700, salto=6)
    h.texto("TOTAL A PAGAR: $ 2.349.900", SANS_B(30), x=700).espacio(20)
    h.linea(1)
    h.texto("GARANTÍA: 12 meses contados a partir de la entrega del producto, conforme a la Ley 1480 de 2011 "
            "(Estatuto del Consumidor). Para hacerla efectiva presente esta factura en cualquier punto de venta.",
            SANS(24), salto=6).espacio(10)
    h.texto("Derecho de retracto: aplica solo a ventas a distancia o no tradicionales (art. 47, Ley 1480 de 2011).",
            SANS(24), salto=6)
    h.texto("CUFE: 3f9a1c77e0b24d5e9a8c1b2d3e4f5a6b7c8d9e0f", MONO(20)).espacio(10)
    h.marca_ficticio()
    return h


def doc_contrato():
    h = Hoja()
    h.texto("CONTRATO DE ARRENDAMIENTO DE VIVIENDA URBANA", SERIF_B(36), centrado=True).espacio(14)
    datos = [
        ("ARRENDADOR:", "Inversiones Los Almendros Ltda., NIT 800.765.432-1"),
        ("ARRENDATARIO:", "Carlos Andrés Mejía Torres, C.C. 1.140.812.334"),
        ("INMUEBLE:", "Apartamento 501, Edificio Brisas, Calle 84 No. 51B-20, Barranquilla"),
        ("CANON MENSUAL:", "$ 1.450.000 (un millón cuatrocientos cincuenta mil pesos)"),
        ("FECHA DE INICIO:", "1 de febrero de 2026"),
        ("DURACIÓN:", "Doce (12) meses"),
    ]
    for k, v in datos:
        h.d.text((110, h.y), k, font=SERIF_B(28), fill=(25, 25, 25))
        h.texto(v, SERIF(28), x=400, ancho=W - 510, salto=8)
    h.espacio(16)
    for t in [
        "PRIMERA. El canon se pagará por mensualidades anticipadas dentro de los cinco (5) primeros días de cada mes.",
        "SEGUNDA. Vencido cada año de ejecución, el canon podrá reajustarse en una proporción que no supere el "
        "ciento por ciento (100 %) del incremento del IPC del año calendario inmediatamente anterior (art. 20, Ley 820 de 2003).",
        "TERCERA. El arrendatario no podrá subarrendar ni ceder el contrato sin autorización expresa y escrita del arrendador.",
        "CUARTA. Para la terminación unilateral, el arrendatario dará aviso con no menos de tres (3) meses de anticipación.",
    ]:
        h.texto(t, SERIF(27), salto=7).espacio(10)
    h.espacio(50)
    h.d.line((140, h.y, 520, h.y), fill=(40, 40, 40), width=2)
    h.d.line((720, h.y, 1100, h.y), fill=(40, 40, 40), width=2)
    h.y += 10
    h.d.text((180, h.y), "EL ARRENDADOR", font=SERIF(26), fill=(25, 25, 25))
    h.d.text((770, h.y), "EL ARRENDATARIO", font=SERIF(26), fill=(25, 25, 25))
    h.marca_ficticio()
    return h


def doc_articulo():
    h = Hoja()
    h.texto("LEY 1437 DE 2011 - CÓDIGO DE PROCEDIMIENTO ADMINISTRATIVO", SERIF_B(30), centrado=True, salto=4)
    h.texto("Y DE LO CONTENCIOSO ADMINISTRATIVO (sustituido por la Ley 1755 de 2015)", SERIF_B(26), centrado=True).espacio(30)
    h.texto("ARTÍCULO 14. TÉRMINOS PARA RESOLVER LAS DISTINTAS MODALIDADES DE PETICIONES.", SERIF_B(29)).espacio(4)
    for t in [
        "Salvo norma legal especial y so pena de sanción disciplinaria, toda petición deberá resolverse dentro de "
        "los quince (15) días siguientes a su recepción. Estará sometida a término especial la resolución de las "
        "siguientes peticiones:",
        "1. Las peticiones de documentos y de información deberán resolverse dentro de los diez (10) días siguientes "
        "a su recepción. Si en ese lapso no se ha dado respuesta al peticionario, se entenderá, para todos los "
        "efectos legales, que la respectiva solicitud ha sido aceptada y, por consiguiente, la administración ya no "
        "podrá negar la entrega de dichos documentos al peticionario, y como consecuencia las copias se entregarán "
        "dentro de los tres (3) días siguientes.",
        "2. Las peticiones mediante las cuales se eleva una consulta a las autoridades en relación con las materias "
        "a su cargo deberán resolverse dentro de los treinta (30) días siguientes a su recepción.",
        "PARÁGRAFO. Cuando excepcionalmente no fuere posible resolver la petición en los plazos aquí señalados, la "
        "autoridad deberá informar esta circunstancia al interesado, antes del vencimiento del término señalado en "
        "la ley, expresando los motivos de la demora y señalando a la vez el plazo razonable en que se resolverá o "
        "dará respuesta, el cual no podrá exceder del doble del inicialmente previsto.",
    ]:
        h.texto(t, SERIF(28), salto=8).espacio(14)
    h.texto("Diario Oficial No. 49.559 de 30 de junio de 2015 · pág. 3", SERIF(22), color=(90, 90, 90))
    h.marca_ficticio()
    return h


def doc_emplazamiento():
    h = Hoja()
    h.texto("DIRECCIÓN SECCIONAL DE IMPUESTOS DE VILLA SERENA", SANS_B(30), centrado=True, salto=4)
    h.texto("División de Fiscalización y Liquidación Tributaria", SANS(26), centrado=True).espacio(14)
    h.linea()
    h.texto("EMPLAZAMIENTO PARA DECLARAR No. 102382026000457", SANS_B(30), salto=6)
    h.texto("Fecha: 15 de septiembre de 2026", SANS(27), salto=6)
    h.texto("Contribuyente: Distribuidora La Esperanza S.A.S. · NIT 900.111.222-3", SANS(27), salto=6)
    h.texto("Impuesto: Renta y complementarios · Año gravable 2025", SANS(27)).espacio(16)
    h.texto("Se emplaza al contribuyente para que, dentro del mes siguiente a la notificación de este acto, presente "
            "la declaración omitida, liquidando la sanción por extemporaneidad prevista en el artículo 642 del "
            "Estatuto Tributario.", SANS(26), salto=7).espacio(12)
    h.tabla([
        ["Concepto", "Valor (COP)"],
        ["Impuesto a cargo estimado", "$ 18.540.000"],
        ["Sanción por extemporaneidad (art. 642 E.T.)", "$ 3.708.000"],
        ["Total estimado", "$ 22.248.000"],
    ], [640, 360], SANS(24))
    h.texto("Si no presenta la declaración, la administración podrá proferir liquidación de aforo y aplicar la "
            "sanción por no declarar del artículo 643 del Estatuto Tributario.", SANS(26), salto=7).espacio(40)
    h.texto("Funcionario: Laura P. Gómez · Inspectora de Impuestos", SANS(25))
    h.sello("NOTIFICADO\n22/09/2026", 780, 1180, color=(30, 60, 150))
    h.marca_ficticio()
    return h


def doc_autorizacion():
    h = Hoja()
    h.texto("AUTORIZACIÓN PARA EL TRATAMIENTO DE DATOS PERSONALES", SANS_B(32), centrado=True).espacio(16)
    h.texto("En cumplimiento de la Ley 1581 de 2012 y del Decreto 1377 de 2013, autorizo a CLÍNICA SAN RAFAEL DEL "
            "NORTE S.A.S. (NIT 890.555.444-0), como responsable del tratamiento, para recolectar, almacenar, usar y "
            "suprimir mis datos personales con las siguientes finalidades:", SANS(26), salto=7).espacio(10)
    for t in ["☐ Agendamiento de citas y envío de recordatorios.",
              "☐ Gestión de la historia clínica (dato sensible: estado de salud).",
              "☐ Envío de información comercial y encuestas de satisfacción."]:
        h.texto(t, SANS(26), x=150, salto=8)
    h.espacio(12)
    h.texto("Declaro que se me informó que el suministro de datos sensibles es facultativo y que, como titular, "
            "puedo conocer, actualizar, rectificar y suprimir mis datos, y revocar esta autorización, escribiendo a "
            "protecciondatos@clinicasanrafael.ejemplo.co.", SANS(26), salto=7).espacio(30)
    h.texto("Nombre: ______________________________   C.C.: ________________", SANS(26), salto=20)
    h.texto("Firma: _______________________________   Fecha: ____/____/2026", SANS(26))
    h.marca_ficticio()
    return h


VALORES_REALES = {
    "01_derecho_peticion.jpg": {
        "tipo_documento": "derecho de petición",
        "emisor": "María Fernanda Rojas Pérez",
        "destinatario": "Alcaldía Distrital de Villa Serena (Secretaría de Movilidad)",
        "fecha": "2026-11-18",
        "numero_referencia": "2026-ER-004521",
        "valor_cop": None,
        "norma_citada": "Ley 1755 de 2015",
    },
    "02_factura_compra.jpg": {
        "tipo_documento": "factura electrónica de venta",
        "emisor": "Electrohogar del Caribe S.A.S.",
        "destinatario": "Juan Camilo Ortega Díaz",
        "fecha": "2026-10-28",
        "numero_referencia": "FE-10234",
        "valor_cop": 2349900,
        "norma_citada": "Ley 1480 de 2011",
    },
    "05_emplazamiento_declarar.jpg": {
        "tipo_documento": "emplazamiento para declarar",
        "emisor": "Dirección Seccional de Impuestos de Villa Serena",
        "destinatario": "Distribuidora La Esperanza S.A.S.",
        "fecha": "2026-09-15",
        "numero_referencia": "102382026000457",
        "valor_cop": 22248000,
        "norma_citada": "Estatuto Tributario, artículo 642",
    },
}


if __name__ == "__main__":
    random.seed(7)
    os.makedirs(OUT, exist_ok=True)
    docs = {
        "01_derecho_peticion.jpg": doc_peticion,
        "02_factura_compra.jpg": doc_factura,
        "03_contrato_arrendamiento.jpg": doc_contrato,
        "04_articulo_14_cpaca.jpg": doc_articulo,
        "05_emplazamiento_declarar.jpg": doc_emplazamiento,
        "06_autorizacion_datos.jpg": doc_autorizacion,
    }
    for nombre, f in docs.items():
        f().escanear(os.path.join(OUT, nombre))
        print("Generada", nombre)
    with open(os.path.join(AQUI, "valores_reales.json"), "w", encoding="utf-8") as fh:
        json.dump(VALORES_REALES, fh, ensure_ascii=False, indent=2)
    print("valores_reales.json actualizado")
