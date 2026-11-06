# Taller: entrada multimodal y audio

Visión y transcripción de audio aplicadas al dominio del proyecto (normativa colombiana).

| Parte | Script | Modelo (Groq) |
|---|---|---|
| Imágenes del dominio | `generar_imagenes.py` | — (genera 6 documentos jurídicos ficticios con Pillow) |
| 1. Visión | `parte1_vision.py` | `qwen/qwen3.8-27b` (`VISION_MODEL`) |
| 2. Audio | `parte2_audio.py` | `whisper-large-v3-turbo` (`WHISPER_MODEL`) + agente RAG del proyecto |

## Variables de entorno

Se leen del archivo `.env` de la **raíz** del repositorio (está en `.gitignore`; ninguna clave va en el código):

```
GROQ_API_KEY=gsk_...                 # obligatoria
VISION_MODEL=qwen/qwen3.8-27b        # opcional
WHISPER_MODEL=whisper-large-v3-turbo # opcional (o whisper-large-v3)
TEXT_MODEL=openai/gpt-oss-120b       # opcional: modelo sin visión usado en 1.4 y con --sin-agente
```

## Cómo ejecutar

Con el entorno del proyecto activo (`pip install -r requirements.txt`) y desde esta carpeta:

```bash
cd taller_multimodal
python generar_imagenes.py        # crea imagenes/ y valores_reales.json
python parte1_vision.py           # 1.1, 1.2, 1.3 y 1.4 (o --punto 1.2)
python parte2_audio.py            # 2.1 transcripción y 2.2 respuesta de cada audio
```

Cada punto imprime la entrada, la salida del modelo y su análisis, y guarda el resultado en `resultados/`.

## Parte 1. Visión

- **Imágenes** (`imagenes/`): derecho de petición radicado, factura de compra, contrato de arrendamiento,
  artículo 14 del CPACA escaneado, emplazamiento para declarar y autorización de datos personales.
  Son documentos **ficticios** (marcados así en la imagen) con aspecto de escaneo. Los valores que una persona
  lee en ellas están en `valores_reales.json`.
- **1.1** Pregunta sobre la factura (producto, total, garantía, retracto).
- **1.2** Esquema `DocumentoJuridico` (Pydantic, 7 campos: `tipo_documento`, `emisor`, `destinatario`, `fecha`,
  `numero_referencia`, `valor_cop`, `norma_citada`) extraído de 3 imágenes en JSON. El JSON se valida con
  `DocumentoJuridico.model_validate`: si está mal formado o no cumple el esquema, el programa lo informa y
  cuenta esos campos como incorrectos, sin fallar. Luego compara campo por campo con `valores_reales.json` y
  reporta el porcentaje de aciertos. Regla de comparación: fecha, número y valor exactos; textos tras
  normalizar (minúsculas, sin tildes ni signos), aceptando que uno contenga al otro.
- **1.3** Dos imágenes en una solicitud: el derecho de petición y el artículo 14 del CPACA.
- **1.4** Tres errores provocados: 4 imágenes (límite 3), modelo sin visión y tipo MIME mal declarado.

Límites de `qwen/qwen3.8-27b` en Groq: máximo 3 imágenes y 20 MB por solicitud; cada imagen cuenta como 2.048 tokens.

## Parte 2. Audio

1. Graben al menos 4 audios de 10 a 30 s (dos integrantes distintos, uno con ruido de fondo) con preguntas que
   usen términos del dominio, y guárdenlos en `audios/` (wav, mp3, m4a, ogg o webm; máximo 25 MB).
2. Registren en `audios/audios.json` el nombre del archivo, quién habla, si tiene ruido y lo que se dijo
   (el texto exacto que leyeron). Ya trae 4 preguntas sugeridas para leer.
3. `python parte2_audio.py` transcribe cada audio (**2.1**), calcula el WER contra lo que se dijo y pasa la
   transcripción al agente del proyecto (**2.2**), que responde con la base de conocimientos y cita la fuente.
   `--sin-agente` responde con el LLM directamente; `--solo-transcribir` omite la respuesta.

Cada solicitud de audio se cobra como mínimo por 10 segundos.
