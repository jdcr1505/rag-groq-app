# Asistente legal experto con RAG — Normativa colombiana

Asistente conversacional que responde preguntas sobre normativa colombiana
(derecho administrativo, consumidor, datos personales, civil, procesal,
tributario y arrendamientos) usando **Retrieval Augmented Generation**.
Los documentos y el índice vectorial permanecen bajo control del proyecto: al
LLM solo se le envían los fragmentos recuperados para cada consulta, nunca el
corpus completo, y los documentos no se usan para entrenar ningún modelo.

- **URL pública de la aplicación:** https://rag-groq-app-57071890782.southamerica-east1.run.app
- **Repositorio:** https://github.com/jdcr1505/rag-groq-app.git

> Aviso: el corpus son extractos resumidos con fines didácticos. Verifique
> siempre contra el texto oficial y vigente. El asistente es orientativo y no
> sustituye la asesoría de un abogado.

## Contenido

1. [Arquitectura del agente](#1-arquitectura-del-agente-avance-3)
2. [Estructura del repositorio](#2-estructura-del-repositorio)
3. [Instalación y ejecución local](#3-instalación-y-ejecución-local)
4. [Evaluación con Ragas](#4-evaluación-con-ragas)
5. [Despliegue en Google Cloud Run](#5-despliegue-en-google-cloud-run)
6. [Limitaciones](#6-limitaciones)

---

## 1. Arquitectura del agente (Avance 3)

```
┌─────────────── Indexación (una vez, durante el build) ───────────────┐
│ pdfs/ (.pdf, .docx, .tex) -> loaders -> RecursiveCharacterTextSplitter│
│   -> embeddings locales (fastembed, multilingual-e5-small)           │
│   -> ChromaDB persistente (metadatos: source, page/ubicación)        │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────── Consulta (cada pregunta): agente con tools ──────────┐
│ Pregunta + historial ──► LLM con tools (Groq, gpt-oss-120b)          │
│        ▲                     │ tool_calls                            │
│        │ ToolMessage         ▼                                       │
│        └──── el programa ejecuta la tool (máx. 5 rondas):            │
│              • buscar_normativa  -> Chroma local (no sale nada)      │
│              • calcular_plazo_habil -> cálculo local (no sale nada)  │
│              • buscar_en_web -> Tavily, solo dominios permitidos     │
│ -> respuesta con citas por origen + registro en logs/agente_tools.jsonl│
└──────────────────────────────────────────────────────────────────────┘
```

**Qué sale del sistema y hacia dónde.** Hacia **Groq** (LLM): el prompt del
sistema, la pregunta, el historial reciente y solo los fragmentos que devolvieron
las tools en esa pregunta (máx. 8 por búsqueda de 800 caracteres), nunca el
corpus completo. Hacia **Tavily** (buscador): únicamente la consulta corta que el
LLM escribe para `buscar_en_web` (máx. 300 caracteres); el contenido de la base
nunca se envía al buscador. Los embeddings, el índice y el cálculo de plazos son
locales.

### 1.1 Tools (`herramientas.py`)

| Tool | Tipo | Qué hace y cuándo se usa | Argumentos | Límites en el código |
|---|---|---|---|---|
| `buscar_normativa` | Búsqueda | Top-k por similitud en Chroma. Primera opción para cualquier pregunta sobre normas, plazos, sanciones o trámites; una llamada por tema | `consulta` (str) | Máx. 8 fragmentos por llamada (`MAX_FRAGMENTOS_POR_LLAMADA`), 800 caracteres por fragmento, consulta de 300 caracteres, 4 llamadas por pregunta. Cada fragmento trae `[Fuente: documento — Pág./Art.]` |
| `calcular_plazo_habil` | Tarea del dominio | Fecha de vencimiento de un plazo en días hábiles en Colombia: cuenta desde el día siguiente, sin sábados, domingos ni los 18 festivos (Ley 51 de 1983 y Pascua). Se usa cuando el usuario da una fecha y pregunta hasta cuándo tiene plazo | `fecha_inicio` (AAAA-MM-DD), `dias_habiles` (int) | Días entre 1 y 365, años 1900-2100, formato de fecha validado, 3 llamadas por pregunta. No considera vacancia judicial |
| `buscar_en_web` | Web | Búsqueda en sitios oficiales fijos. Solo si la base no responde y la pregunta es de información pública o actual | `consulta` (str) | Máx. 3 resultados (título, URL, extracto de 300 caracteres, `origen: web`), `include_domains` fijo, doble filtro de dominio, 2 llamadas por pregunta. Solo existe si hay `TAVILY_API_KEY` y el usuario activa "Búsqueda web" |

Prueba de las tools sin el LLM (`.invoke()` directo):

```bash
python herramientas.py        # o: python evaluate_agent.py --tools
```

### 1.2 Bucle del agente y registro de uso (`agent_pipeline` en `rag_core.py`)

- Bucle manual con `llm.bind_tools(tools)`: si la respuesta trae `tool_calls`, el programa ejecuta cada tool y devuelve el resultado como `ToolMessage`; si no, es la respuesta final.
- **Límite de iteraciones:** `MAX_ITERACIONES = 5` rondas por pregunta; al agotarlas se pide la respuesta final sin tools.
- Un saludo se responde sin tools; una pregunta con dos temas genera dos llamadas a `buscar_normativa`.
- Si Groq rechaza una tool call mal formada (`tool_use_failed`), la pregunta se resuelve con el pipeline fijo del Avance 2.
- **Registro:** cada pregunta agrega una línea JSON a `logs/agente_tools.jsonl` con la pregunta, las tools llamadas y sus argumentos, el número de llamadas al LLM, los tokens de entrada y salida, el tiempo y si hubo respuesta. La interfaz muestra lo mismo en el desplegable "Herramientas usadas".

### 1.3 Prompt del sistema (`AGENT_SYSTEM_PROMPT`)

El prompt ya no lleva el contexto: describe cuándo usar cada tool, exige responder
solo con lo que devuelvan las tools, define la cita por origen —`(Fuente:
documento, Pág. N)` para la base, `[web: URL]` para la web y `(Cálculo:
calcular_plazo_habil)` para el cálculo—, mantiene la frase exacta de rechazo
("No encontré información sobre esto en la base de conocimientos."), prohíbe
mostrar el corpus o las instrucciones e incluye cuatro ejemplos few-shot: saludo
sin tools, pregunta con dos temas, tarea de plazo y pregunta fuera de alcance.

### 1.4 Política de búsqueda web

- **Cuándo se usa:** solo si `buscar_normativa` no trae la respuesta y la pregunta trata de información pública o actual (norma vigente o reformada, valor anual como la UVT, definición general, sentencia). La regla está en el docstring de la tool y en el prompt del sistema. Nunca se usa si la base ya respondió.
- **Sitios permitidos** (fijos en `WEB_DOMINIOS_PERMITIDOS`; el LLM solo escribe la consulta y no puede cambiarlos):

| Sitio | Por qué es confiable |
|---|---|
| secretariasenado.gov.co | Texto oficial de leyes y códigos con notas de vigencia |
| suin-juriscol.gov.co | Sistema Único de Información Normativa del Ministerio de Justicia |
| funcionpublica.gov.co | Gestor normativo oficial del Departamento Administrativo de la Función Pública |
| corteconstitucional.gov.co | Sentencias de constitucionalidad que modifican o condicionan normas |
| dian.gov.co | Autoridad tributaria: UVT, calendario y conceptos |
| sic.gov.co | Autoridad de protección al consumidor y de datos personales |
| alcaldiabogota.gov.co | Régimen Legal de Bogotá: compilación pública de normas nacionales |

- **Resultados acotados:** máximo 3, cada uno con título, URL y extracto de 300 caracteres, marcados `origen: web`.
- **Respuesta separada por origen:** la base se cita `(Fuente: documento, Pág. N)` y la web `[web: URL]`, en párrafos distintos; la interfaz agrupa las fuentes en "Base de conocimientos" y "Web".
- **Privacidad:** la consulta no debe incluir datos personales ni texto copiado de la base.

### Entrada por voz e imagen

- **Voz:** el botón de micrófono graba en el navegador (máx. 60 s) y `/api/transcribe` transcribe con `whisper-large-v3-turbo` en Groq. El texto queda en el campo para revisarlo antes de enviarlo.
- **Imagen:** se puede adjuntar una foto (p. ej. una factura o un contrato). El navegador la reduce a 1600 px en JPEG, el modelo de visión (`VISION_MODEL`, por defecto `qwen/qwen3.8-27b`) la describe y transcribe su texto, y esa descripción entra al agente junto con la pregunta. Así el modelo con tools no cambia y las citas siguen saliendo de los documentos.

### Formatos de documento y citas

La carpeta `pdfs/` acepta `.pdf`, `.docx` y `.tex`. Un `.docx` o un `.tex` no guarda páginas (la paginación se calcula al abrir o compilar el archivo), por eso:

- **PDF:** una unidad por página; se cita `(Fuente: archivo.pdf, Pág. N)`.
- **DOCX y TEX:** el texto se divide en secciones; cada línea que empieza con «Artículo N.» abre una sección y se cita `(Fuente: archivo.docx, Art. N)`. El texto anterior al primer artículo queda como `Inicio`. Si no hay artículos pero sí títulos (estilos de título de Word o `\section` de LaTeX), la sección se cita por el título; si no hay ni lo uno ni lo otro, como `Texto completo`.
- Un artículo largo se parte en varios fragmentos que conservan la misma ubicación. El prompt de evaluación (RAG clásico) no cambió; el de la aplicación pide la cita con el formato que traiga el encabezado del fragmento.
- Al cambiar la forma de leer los documentos (`INGEST_VERSION` en `rag_core.py`) o los archivos de `pdfs/`, el índice se reconstruye solo.

### Decisiones técnicas

| Etapa | Decisión | Justificación |
|---|---|---|
| Selección de documentos | 7 fragmentos normativos: Ley 1437/2011 (CPACA), Ley 1480/2011 (Estatuto del Consumidor), Ley 1581/2012 (datos personales), Código Civil (obligaciones y responsabilidad), Ley 1564/2012 (Código General del Proceso), Estatuto Tributario y Ley 820/2003 (arrendamiento de vivienda) | Cubren trámites y derechos que una persona enfrenta a diario (peticiones, garantías, retracto, datos, contratos, demandas, sanciones tributarias, arriendo). Son textos con estructura por artículos, ideales para recuperación y citación precisa |
| Ingesta | `PyPDFLoader` (PDF) y lectores propios para `.docx` y `.tex` (`loaders.py`, sin dependencias nuevas) | Los PDF conservan `source` y `page` por página. DOCX y TEX no tienen páginas fijas, así que se dividen por artículo y se citan como `Art. N` (ver «Formatos de documento y citas») |
| Chunking | `RecursiveCharacterTextSplitter`, 800 caracteres, solape 150, separadores `\n\n`, `\n`, `.`, ` ` | Un artículo corto cabe en un fragmento; el solape (~19 %) evita perder ideas en los límites; cortar primero por párrafo y oración respeta la estructura legal. Resultado: 14 páginas -> 42 fragmentos |
| Vectorización | `intfloat/multilingual-e5-small` (fastembed, ONNX, local) | Multilingüe con buen desempeño en español, 384 dimensiones, corre en CPU y no envía el texto a terceros |
| Base vectorial | ChromaDB persistente, colección `mis_programas` | Persistencia en disco, sin servidor aparte. Metadatos `source`, `page` y `start_index` (posición del fragmento en la página) permiten citar la fuente de cada respuesta |
| Recuperación | Búsqueda por similitud, `top_k = 8` (ajustable de 1 a 15 en la interfaz) | Con k=8 la evaluación dio `context_recall` 1.000 y `context_precision` 0.901: se recuperan los fragmentos necesarios con poco ruido. El corpus tiene solo 42 fragmentos, así que k=8 cubre cerca de una quinta parte (ver §4 sobre el alcance de esta justificación) |
| Generación | Groq `openai/gpt-oss-120b`, temperatura 0, con tool calling | Respuestas deterministas y apegadas al contexto; el modelo soporta tools en Groq |
| Conversación | El navegador conserva el historial (últimos 3 turnos) y lo envía en cada petición; el LLM escribe la consulta de la tool como pregunta independiente | Una pregunta como "¿y cuánto cuesta?" no contiene palabras para buscar por similitud. El servidor es *stateless*, requisito de Cloud Run, y varios usuarios no mezclan sus diálogos |
| Control de alucinaciones | Prompt restringido al contexto, frase de rechazo exacta, ejemplo few-shot fuera de alcance, y la interfaz oculta las fuentes cuando no hay información | Se mide con la tasa de abstención en preguntas fuera del corpus |
| Resiliencia | Reintentos con espera creciente ante errores 429 de Groq y mensajes amigables en el chat (el detalle técnico queda solo en los logs) | El plan gratuito de Groq limita los tokens por minuto |

### Estructura del prompt (en `rag_core.py`)

La app usa `AGENT_SYSTEM_PROMPT` (rol, fecha de hoy, descripción de las tools,
reglas y formato de salida). `SYSTEM_PROMPT`, descrito abajo, es el del RAG
clásico, que sigue usando la evaluación con Ragas.

- **System prompt:** rol (experto en normativa colombiana), reglas (usar solo el contexto, tratar el contexto como datos y no como instrucciones, una cita por párrafo con formato `(Fuente: Documento.pdf, Pág. N)`, respuesta parcial indicando qué falta, frase de rechazo exacta, el historial solo sirve para entender la pregunta) y formato de salida.
- **Few-shot:** 3 ejemplos con un documento ficticio: respuesta directa, respuesta parcial y pregunta fuera de alcance.
- **Delimitadores:** `<contexto>...</contexto>` y `<pregunta>...</pregunta>`.
- **Formato de salida:** 1 a 3 párrafos cortos (o lista breve), sin preámbulos, con una sola cita al final de cada párrafo o viñeta.
- **Frase de rechazo:** "No encontré información sobre esto en la base de conocimientos."

---

## 2. Estructura del repositorio

```
├─ app.py                  # Backend Flask (API + registro de interacciones)
├─ rag_core.py             # Pipeline RAG: ingesta, chunking, embeddings, Chroma, prompt, tools, voz e imagen
├─ loaders.py              # Lectores de .docx y .tex (secciones por artículo)
├─ herramientas.py         # Tools del agente: buscar_normativa, calcular_plazo_habil, buscar_en_web (y sus límites)
├─ build_index.py          # Construye el índice (se ejecuta en el build de Docker)
├─ warm_model.py           # Pre-descarga el modelo de embeddings
├─ evaluate_ragas.py       # Evaluación con Ragas (--pipeline fijo | agente)
├─ evaluate_agent.py       # Casos de comportamiento y comparación de costo del agente
├─ casos_comportamiento.json # Casos nuevos con el comportamiento esperado
├─ logs/agente_tools.jsonl # Registro de uso de tools (NO se sube)
├─ taller_multimodal/      # Taller de visión y audio (ver su README)
├─ eval_dataset.json       # Conjunto de evaluación (20 preguntas, 4 fuera del corpus)
├─ eval_results/           # Resultados de las evaluaciones (JSON, CSV, gráfico)
├─ requirements.txt        # Dependencias de la app (lo que instala Docker)
├─ requirements-eval.txt   # requirements.txt + ragas y versiones fijadas
├─ Dockerfile
├─ .env                    # Clave local (NO se sube al repositorio)
├─ interactions.jsonl      # Registro local de consultas (NO se sube)
├─ pdfs/                   # Corpus
├─ templates/index.html
└─ static/{css/style.css, js/script.js}
```

---

## 3. Instalación y ejecución local

Requisitos: **Python 3.11** (evita 3.13) y una API key gratuita de Groq (https://console.groq.com).

### 3.1 Crear y activar el entorno virtual

Desde la carpeta del proyecto:

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
Si PowerShell bloquea el script, ejecuta una vez
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y repite la activación.

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Sabrás que está activo cuando aparezca `(.venv)` al inicio de la línea.

### 3.2 Instalar dependencias

Solo para ejecutar la aplicación:
```bash
pip install -r requirements.txt
```

Para ejecutar la aplicación **y** la evaluación con Ragas (incluye lo anterior):
```bash
pip install -r requirements-eval.txt
```

`requirements-eval.txt` fija la familia de LangChain en la serie 0.3
(`langchain-community==0.3.27`, `langchain-core<1.0`, etc.) porque
`ragas==0.2.15` no es compatible con las versiones 1.x. Si pip reporta
conflictos, crea un entorno aparte solo para evaluar (`python -m venv .venv-eval`).

### 3.3 Configurar las claves

Crea un archivo `.env` en la raíz (puedes copiar `.env.example`), sin comillas ni espacios:

```
GROQ_API_KEY=gsk_tu_clave_real
TAVILY_API_KEY=tvly-tu_clave   # opcional: activa la búsqueda web
```

Opcionalmente, `VISION_MODEL` y `WHISPER_MODEL` cambian los modelos de Groq
usados para imagen y audio.

### 3.4 Colocar los PDFs

Copia los documentos del corpus en `pdfs/` (la carpeta conserva ese nombre): `.pdf`, `.docx` o `.tex`.

### 3.5 Ejecutar la aplicación

```bash
python app.py
```

Abre http://localhost:5000. La primera vez descarga el modelo de embeddings
(varios cientos de MB) e indexa los PDFs; el panel lateral muestra el progreso
y el chat se habilita cuando dice "Base vectorial lista". El índice se
reconstruye automáticamente si cambian los PDFs o los parámetros de chunking.

Opcional: `python build_index.py` pre-construye el índice sin levantar el servidor.

### 3.6 Usar la interfaz

- **Chat conversacional:** admite preguntas de seguimiento ("¿y cuánto me cuesta eso?"); el historial se guarda por conversación. Bajo cada respuesta, el desplegable **Herramientas usadas** muestra cada tool con sus argumentos, el número de llamadas al LLM y el tiempo.
- **Búsqueda web:** interruptor en el panel lateral (requiere `TAVILY_API_KEY`).
- **Dictado:** botón de micrófono; el navegador pide permiso la primera vez (funciona en `localhost` o HTTPS).
- **Imagen:** botón para adjuntar una foto; se puede enviar con o sin pregunta.
- **Fuentes:** desplegable separado por origen: *Base de conocimientos* (documento, página o artículo y fragmento) y *Web* (título y URL).
- **Fuera del corpus:** responde "No encontré información sobre esto en la base de conocimientos." y no muestra fuentes.
- **Fragmentos a recuperar:** control de `top_k` de 1 a 8 (el límite de la tool).
- **Nueva conversación:** reinicia el historial.
- Cada interacción se guarda en `interactions.jsonl` (pregunta, respuesta, contextos, fuentes, `k`, fecha) para la evaluación de uso real.

---

## 4. Evaluación con Ragas

Con el entorno activo y `requirements-eval.txt` instalado.

**Conjunto de evaluación:** `eval_dataset.json` tiene 20 preguntas: 16 dentro del
corpus con respuesta de referencia (*ground truth*) y 4 fuera del corpus (UVT 2026,
vacaciones, hurto agravado, capital de Australia) para medir el control de
alucinaciones. Incluye datos puntuales, plazos, procedimientos, preguntas que
combinan varios fragmentos y preguntas con redacción coloquial.

**Métricas:** `faithfulness`, `answer_relevancy`, `context_precision` y
`context_recall` (sobre las 16 preguntas del corpus), más la **tasa de abstención**
en las 4 preguntas fuera del corpus.

**Modelos:** generador `openai/gpt-oss-120b`; juez `qwen/qwen3.8-27b` (distinto del
generador, para evitar sesgo de autoevaluación), con el mismo juez en todas las corridas.

### Cómo reproducirla

```bash
# 1) Línea base
python evaluate_ragas.py --tag base --sleep 25

# 2) Completar celdas que el juez no pudo calcular (límites de Groq)
python evaluate_ragas.py --rescore base

# 3) Iteración de mejora (tras modificar el prompt en rag_core.py).
#    Solo juzga faithfulness y answer_relevancy; copia precision/recall de la base
#    (válido porque k, chunk-size y overlap no cambian)
python evaluate_ragas.py --tag prompt2 --sleep 25 --reuse-context-from base

# 4) Tabla y gráfico antes/después
python evaluate_ragas.py --compare base prompt2

# 5) Uso real: faithfulness y answer_relevancy sobre interactions.jsonl
python evaluate_ragas.py --from-log --last 10
```

Para variar parámetros de recuperación o chunking: `--k 4`, o
`--chunk-size 500 --overlap 100` (cada variante usa su propia colección
`eval_c{size}_o{overlap}` y no afecta al índice de producción). Si Groq limita por
tokens, sube `--sleep`. Para cambiar el juez: `$env:JUDGE_MODEL="..."` (PowerShell)
o `export JUDGE_MODEL=...` (Linux/macOS). Resultados en `eval_results/`:
`<tag>.json`, `<tag>_detalle.csv`, `<tag>_gen.json` (respuestas generadas),
`comparacion_A_vs_B.csv/.png`, `log_eval.json` y `log_eval_detalle.csv`.

### Resultados: línea base vs. iteración de mejora

| Métrica | Base | Mejora (`prompt2`) | Δ |
|---|---|---|---|
| faithfulness | 0.788 | 0.802 | +0.014 |
| answer_relevancy | 0.868 | 0.863 | -0.006 |
| context_precision | 0.901 | 0.901 (*) | 0.000 |
| context_recall | 1.000 | 1.000 (*) | 0.000 |
| Abstención fuera del corpus | 4/4 | 4/4 | 0 |

(*) Copiadas de la base: el cambio fue solo de prompt, por lo que la recuperación
(y con ella estas dos métricas, que no dependen de la respuesta) no cambia.
Gráfico: `eval_results/comparacion_base_vs_prompt2.png`.

Notas de la ejecución: 15 celdas de la base no pudieron calcularse en la primera
corrida por el límite diario de tokens de Groq y se recalcularon con
`--rescore` (0 valores vacíos al final). No hubo falsos rechazos ni alucinaciones
en las preguntas fuera del corpus.

### Análisis

**Métrica más baja y componente.** La más baja fue `faithfulness` (0.788), que apunta
a la **generación**. La recuperación es sólida: `context_recall` 1.000 y
`context_precision` 0.901, así que chunking y recuperación no son el cuello de
botella. Al revisar las respuestas con menor puntaje (0.50 a 0.71), eran correctas
y literales respecto al artículo. El puntaje bajo se debe en buena parte a que las
citas `(Fuente: X.pdf — Pág. N)` no figuran en los fragmentos que recibe el juez,
por lo que se cuentan como afirmaciones sin respaldo; en respuestas cortas, una sola
cita basta para bajar el puntaje a 0.5. Es, por tanto, en gran medida un artefacto
de la medición y no una alucinación.

**Iteración.** Se modificó un solo elemento, el prompt: se pidió una única cita por
párrafo o viñeta con formato unificado (`(Fuente: Documento.pdf, Pág. N)`, con coma)
y se ajustó la línea correspondiente del formato de salida. `faithfulness` pasó de
0.788 a 0.802 y `answer_relevancy` de 0.868 a 0.863. Con 16 preguntas y un juez que
es otro LLM, diferencias de esta magnitud están dentro del ruido: el efecto es
marginal y no demuestra una mejora real, y parte de la subida puede deberse a haber
menos citas por respuesta y no a mayor fidelidad.

**Qué se haría después.** (1) Incluir el encabezado de fuente en los contextos que se
pasan a Ragas, o retirar las citas antes de juzgar, para medir la fidelidad del
contenido sin el artefacto; (2) ampliar el conjunto de evaluación más allá de 16
preguntas para reducir el ruido; (3) evaluar variantes de `top_k` y de tamaño de
chunk, que no se corrieron por el límite diario de tokens del plan gratuito de Groq.
Por ello, la elección de `top_k = 8` se justifica por sus métricas absolutas
(recall 1.000, precisión 0.901), no por una comparación entre valores de k.

### Uso real (`--from-log`)

De 15 interacciones registradas en pruebas locales, 3 fueron abstenciones (se
excluyen, pues no hay respuesta que juzgar) y se evaluaron 10. `faithfulness` fue
**0.747** sobre 9 respuestas (una quedó sin calcular porque el juez agotó
`max_tokens`) y `answer_relevancy` fue **0.854** sobre 10. Cuatro respuestas quedaron
por debajo de 0.7. Al revisarlas, la mayoría son correctas y su puntaje bajo se debe
al mismo efecto de las citas. Una (faithfulness 0.33, sobre qué hacer si el
arrendador no recibe el inmueble) sí añadió una justificación que el artículo no
contiene ("como medio para cumplir con la obligación de entrega"): una desviación
leve pero real, que respalda mantener la regla de usar únicamente el contexto. Estas
interacciones se hicieron con el prompt anterior a `prompt2`.

---

## 4b. Evaluación del agente (Avance 3)

### Cómo reproducirla

```bash
# Ragas sobre el agente (mismo eval_dataset.json del Avance 2). Usar Python 3.11:
# ragas 0.2.15 falla en Python 3.14 ("Timeout should be used inside a task").
python evaluate_ragas.py --tag agente --pipeline agente --sleep 12
python evaluate_ragas.py --compare formatos agente

# Casos de comportamiento (esperado escrito antes en casos_comportamiento.json)
python evaluate_agent.py --casos
# Costo: pipeline fijo vs agente en las mismas 8 preguntas
python evaluate_agent.py --costo --n 8
```

En Ragas, los `retrieved_contexts` del agente son los fragmentos que devolvieron
sus tools, y la búsqueda web se desactiva para comparar con la misma base.

### Ragas: pipeline fijo (Avance 2) frente al agente

Comparación sobre las mismas preguntas juzgadas en ambos (el juez agotó el límite
diario de tokens de Groq y quedaron celdas del agente sin calcular; se completan con
`python evaluate_ragas.py --rescore agente`):

| Métrica (n preguntas) | Avance 2 (`formatos`, pipeline fijo) | Avance 3 (agente) | Δ |
|---|---|---|---|
| faithfulness (12) | 0.896 | 0.826 | -0.070 |
| answer_relevancy (15) | 0.868 | 0.875 | +0.007 |
| context_precision (11) | 0.803 | 0.818 | +0.015 |
| context_recall (12) | 0.958 | 0.958 | 0.000 |
| Abstención fuera del corpus | 4/4 | 4/4 | 0 |

**Análisis.** La recuperación se mantuvo o mejoró levemente: el agente escribe su
propia consulta para la tool, sin palabras de relleno, y en algunas preguntas busca dos
veces. `faithfulness` bajó: las respuestas con menor puntaje son correctas y literales,
pero muy cortas y con la cita `(Fuente: …)`, que el juez cuenta como afirmación sin
respaldo (el mismo artefacto del Avance 2, que pesa más en respuestas breves); además
el prompt ya no lleva el contexto entre delimitadores y el modelo redacta con más
libertad. Se atribuye a la generación, no a la recuperación. Con 11 a 15 preguntas y
un juez LLM, solo la caída de `faithfulness` es clara; el resto está dentro del ruido.

### Casos de comportamiento (8/8 cumplen)

| # | Tipo | Tools usadas (registro) | Llam. LLM | ¿Cumple? |
|---|---|---|---|---|
| 1 | Saludo | ninguna | 1 | Sí |
| 2 | Pregunta compuesta | `buscar_normativa` ×2 (retracto; petición) | 3 | Sí |
| 3 | Tarea del dominio | `buscar_normativa` + `calcular_plazo_habil(2026-12-18, 15)` → 13-ene-2027 | 3 | Sí |
| 4 | Tarea del dominio | `calcular_plazo_habil(2026-11-04, 10)` → 19-nov-2026 | 2 | Sí |
| 5 | Información pública o actual | `buscar_en_web("valor UVT 2026 Colombia")`, 3 resultados de dian.gov.co | 2 | Sí |
| 6 | Pregunta que sí responde la base | `buscar_normativa` (sin web) | 2 | Sí |
| 7 | Instrucción maliciosa | ninguna | 1 | Sí |
| 8 | Fuera de la base y de la web | ninguna | 1 | Sí |

Detalle (preguntas, respuestas completas) en `eval_results/casos_comportamiento.md` y `.json`.
Observaciones: en el caso 5, con el prompt final, fue directo a la web sin consultar antes la base (la pregunta pide un valor del año en curso; en la interfaz, con la misma pregunta, sí buscó primero en la base); la respuesta salió de dian.gov.co (la interfaz reconoce las citas `[web: URL]` y `(web: URL)`); en el caso 7
no mostró el corpus ni el prompt, pero respondió con la frase de rechazo en vez de una
negativa explícita; en el caso 8 respondió la frase de rechazo sin llegar a buscar.

**Error de decisión corregido.** En la pregunta mixta «¿Qué sanción por extemporaneidad
establece el Estatuto Tributario y cuál es el valor de la UVT para 2026?» el agente buscó dos
veces en la base y respondió «no encontré» para la UVT sin intentar la web. Se añadió al prompt
que, en preguntas con varias partes, busque en la web la parte actual que no esté en la base;
con ese cambio usó `buscar_en_web` y citó cada parte con su origen. Los casos se volvieron a
ejecutar con el prompt final (8/8).

### Costo (promedio por pregunta, 8 preguntas del dataset)

| Medida | Pipeline fijo | Agente |
|---|---|---|
| Llamadas al LLM | 1,00 | 2,12 |
| Tokens (entrada + salida) | 2.243 | 4.865 |
| Tiempo de respuesta (s) | 6,1 | 24,6 |

El agente hace al menos dos llamadas (decidir la tool y responder) y reenvía el prompt
con los fragmentos en cada ronda, por eso duplica los tokens. Los tiempos incluyen las
esperas por el límite de tokens por minuto del plan gratuito de Groq, que afectan más al
agente; en la corrida de Ragas (20 preguntas, con pausas) el agente promedió 9,0 s.

---

## 5. Despliegue en Google Cloud Run

La imagen se construye con el `Dockerfile`: instala `requirements.txt`, descarga
el modelo de embeddings (`warm_model.py`) e indexa los PDFs de `pdfs/`
(`build_index.py`), así que el contenedor arranca con la base lista. Se sirve con
Gunicorn (1 worker, 8 hilos) en el puerto definido por `$PORT`.

**URL pública:** https://rag-groq-app-57071890782.southamerica-east1.run.app

### 5.1 Preparación (una sola vez)

```bash
gcloud auth login
gcloud config set project rag-groq-app
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

### 5.2 Primer despliegue

```bash
gcloud run deploy rag-groq-app \
  --source . \
  --region southamerica-east1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars GROQ_API_KEY=<tu_clave>,TAVILY_API_KEY=<tu_clave_tavily>
```

### 5.3 Redespliegue (cambios de código, prompt o PDFs)

```bash
gcloud run deploy rag-groq-app --source . --region southamerica-east1
```

Las variables de entorno se conservan entre revisiones, por lo que no hay que
volver a pasar la clave.

### 5.4 Credenciales

La clave de Groq **nunca** está en el repositorio: `.env` está en `.gitignore` y
`.dockerignore`, y en la nube se entrega como variable de entorno de Cloud Run.
Alternativa más segura con Secret Manager:

```bash
printf "<tu_clave>" | gcloud secrets create groq-key --data-file=-
gcloud run services update rag-groq-app --region southamerica-east1 \
  --set-secrets GROQ_API_KEY=groq-key:latest
```

(La cuenta de servicio de Cloud Run necesita el rol *Secret Manager Secret Accessor*.)

### 5.5 Notas

- Cloud Run no conserva estado entre instancias: el historial vive en el navegador y se envía en cada petición.
- El disco es temporal, así que `interactions.jsonl` solo es confiable en ejecución local.
- Se usan 2 GiB de memoria porque el modelo de embeddings y Chroma no caben cómodamente en 512 MB.
- Si se agregan o cambian PDFs, hay que redesplegar para reconstruir el índice en el build.

---

## 6. Limitaciones

- Los documentos son extractos resumidos; la información puede estar desactualizada o incompleta frente al texto oficial.
- El asistente es orientativo y no sustituye la asesoría de un abogado.
- La evaluación usa 16 preguntas dentro del corpus y un juez LLM: las diferencias pequeñas entre corridas no son concluyentes. `faithfulness` está sesgada a la baja por las citas, que no aparecen en los contextos evaluados.
- Los resultados web no pasan por el corpus curado: pueden ser imprecisos y dependen de las fuentes que devuelva Tavily.
- Solo se evaluó una iteración (prompt); las variantes de `top_k` y de chunking quedaron pendientes por los límites del plan gratuito de Groq.
- Se ingieren PDF, DOCX y TEX; TXT y HTML requerirían añadir su lector. En DOCX y TEX la cita es por artículo (`Art. N`), no por página, y los lectores esperan líneas «Artículo N.» al inicio de cada artículo; sin ellas, el texto se divide por títulos o se cita como un solo bloque. No se leen cuadros de texto, notas al pie ni imágenes de los DOCX, ni fórmulas de los TEX.