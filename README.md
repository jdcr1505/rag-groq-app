# Asistente legal experto con RAG — Normativa colombiana

Asistente conversacional que responde preguntas sobre normativa colombiana
(derecho administrativo, consumidor, datos personales, civil, procesal,
tributario y arrendamientos) usando **Retrieval Augmented Generation** con un
**agente que usa herramientas**, con cuentas de usuario y conversaciones guardadas.
Los documentos y el índice vectorial permanecen bajo control del proyecto: al
LLM solo se le envían los fragmentos que devuelve la herramienta de búsqueda para
cada consulta, nunca el corpus completo, y los documentos no se usan para entrenar ningún modelo.

- **URL pública de la aplicación:** https://rag-groq-app-57071890782.southamerica-east1.run.app
- **Repositorio:** https://github.com/jdcr1505/rag-groq-app.git

> Aviso: el corpus son extractos resumidos con fines didácticos. Verifique
> siempre contra el texto oficial y vigente. El asistente es orientativo y no
> sustituye la asesoría de un abogado.

## Contenido

1. [Flujo RAG implementado](#1-flujo-rag-implementado)
2. [Cuentas, conversaciones y seguridad](#2-cuentas-conversaciones-y-seguridad)
3. [Estructura del repositorio](#3-estructura-del-repositorio)
4. [Instalación y ejecución local](#4-instalación-y-ejecución-local)
5. [Evaluación](#5-evaluación)
6. [Despliegue en Google Cloud Run](#6-despliegue-en-google-cloud-run)
7. [Limitaciones](#7-limitaciones)

---

## 1. Flujo RAG implementado

```
┌─────────────── Indexación (una vez, durante el build) ───────────────┐
│ pdfs/ (.pdf, .docx, .tex)                                            │
│   PDF  -> PyPDFLoader (una unidad por página)                        │
│   DOCX/TEX -> loaders.py (una unidad por artículo)                   │
│   -> RecursiveCharacterTextSplitter                                  │
│   -> embeddings locales (fastembed, multilingual-e5-small)           │
│   -> ChromaDB persistente (metadatos: source, ubicación, start_index)│
└──────────────────────────────────────────────────────────────────────┘

┌──────────────── Consulta (cada pregunta): agente con tools ──────────┐
│ [audio -> Whisper (Groq) -> texto]   [imagen -> modelo de visión     │
│                                       (Groq) -> descripción en texto]│
│ Pregunta + historial -> LLM con tools (Groq, bind_tools)             │
│   ↻ el LLM pide buscar_normativa / calcular_plazo / web_search,      │
│     el backend ejecuta la tool y devuelve el resultado (ToolMessage),│
│     hasta que el LLM responde sin pedir más tools (máx. 5 rondas)    │
│   -> respuesta con citas + fuentes + registro de uso                 │
└──────────────────────────────────────────────────────────────────────┘
```

### Agente con herramientas

La búsqueda deja de ser un paso fijo y se convierte en una **tool** que el LLM
decide cuándo usar (`agent_pipeline` en `rag_core.py`). El bucle es manual:
el LLM (ChatGroq con `bind_tools`) responde con `tool_calls`, el backend ejecuta
cada tool con `.invoke(args)`, agrega un `ToolMessage` con el resultado y vuelve a
llamar al LLM, hasta que responde sin pedir más tools.

| | RAG clásico (`rag_pipeline`) | Agente (`agent_pipeline`, el que usa la app) |
|---|---|---|
| Búsqueda | Siempre, una vez | Cuando el LLM la pide; puede repetirla con otras consultas |
| Saludos ("hola") | Busca igual y suele responder "No encontré información" | Responde sin herramientas |
| Preguntas de seguimiento | Llamada extra al LLM para reformular | El LLM escribe la consulta de la tool ya reformulada |
| Fechas de vencimiento | No puede calcularlas | Llama a `calcular_plazo` |
| Información actual | No | `web_search` controlada, solo si la base no basta |
| Costo | 1-2 llamadas al LLM | 2 o más llamadas (una por ronda de tools) |

El pipeline clásico se conserva: lo usa Ragas como línea base y es el respaldo si
Groq rechaza una tool call mal formada (`tool_use_failed`).

#### Herramientas y límites (escritos en el código)

| Herramienta | Tipo | Qué hace y cuándo | Argumentos | Límites en el código |
|---|---|---|---|---|
| `buscar_normativa` | Búsqueda | Fragmentos de la base de conocimientos. Se usa **primero** para cualquier pregunta normativa | `consulta: str` | Máx. **8 fragmentos** por llamada (`MAX_FRAGMENTOS_POR_LLAMADA`), aunque la interfaz pida más; cada fragmento se corta a **800 caracteres** (`MAX_CHARS_FRAGMENTO`). Cada fragmento trae documento y página (PDF) o artículo (DOCX/TEX) |
| `calcular_plazo` | Tarea del dominio | Fecha de vencimiento de un plazo con festivos de Colombia (Ley Emiliani y Semana Santa), en `plazos.py`. El LLM extrae los argumentos de la pregunta | `fecha_inicio: AAAA-MM-DD`, `cantidad_dias: int`, `tipo: "habiles"\|"calendario"` | 1 a 3650 días; fecha y tipo validados (un argumento inválido devuelve un mensaje claro, no una excepción) |
| `web_search` | Web (opcional) | Información pública y actual cuando la base no la tiene | `consulta: str` | Solo 6 dominios; máx. **3 resultados**; extractos de **400 caracteres**; **2 llamadas** por pregunta; solo después de `buscar_normativa` |

Convención de `calcular_plazo`: el conteo empieza el día siguiente a la fecha de
inicio; los días hábiles son de lunes a viernes sin festivos. Es orientativo: la
norma aplicable puede tener reglas propias.

Otros límites: `MAX_ITERACIONES = 5` rondas de tools por pregunta (si se agotan, se
pide la respuesta final con una llamada sin tools) y reintentos con espera ante
errores 429 de Groq.

#### Búsqueda web controlada ("primero la base")

La web se consulta **solo si** (a) `buscar_normativa` no contiene la respuesta y
(b) la información es pública y actual sobre normativa o administración colombiana
(valor de la UVT, reformas recientes, normas que no están en la base). La política
se aplica en **dos capas**: el prompt y el código (la tool rechaza una llamada web
si no hubo antes una búsqueda en la base y limita a 2 por pregunta; las llamadas
bloqueadas quedan en el registro).

Servicio: [Tavily](https://tavily.com) (`TAVILY_API_KEY`, opcional). Solo se ofrece
al LLM si existe la clave y el usuario activa "Búsqueda web". Está restringida con
`include_domains` y cada URL se vuelve a verificar en el código.

| Sitio permitido | Por qué es confiable |
|---|---|
| secretariasenado.gov.co | Senado de la República: leyes y códigos con notas de vigencia |
| funcionpublica.gov.co | Función Pública: Gestor Normativo (normas y conceptos) |
| dian.gov.co | Autoridad tributaria: UVT, resoluciones, calendario |
| sic.gov.co | Superintendencia de Industria y Comercio: autoridad de consumo y de protección de datos |
| suin-juriscol.gov.co | Ministerio de Justicia: normativa y jurisprudencia |
| corteconstitucional.gov.co | Corte Constitucional: sentencias |

Cada resultado trae título, URL y extracto corto, y se marca como web. La respuesta
cita la base como `(Fuente: documento, Pág./Art. N)`, el plazo como
`(Cálculo: calcular_plazo)` y la web como `[web: URL]`, y separa lo que viene de
internet bajo "Información de fuentes web:". El prompt incluye la fecha de hoy.

#### Registro de uso

Cada pregunta deja una línea JSON en `agent_usage.jsonl` (variable `USO_LOG_FILE`):
fecha, origen (`app`, `casos`, `ragas_agente`), pregunta, modo, herramientas con
**argumentos completos** (y si la política en código las bloqueó), fragmentos y
resultados web, **llamadas al LLM, tokens y segundos**. No se sube al repositorio
(contiene preguntas de usuarios). `interactions.jsonl` sigue guardando preguntas,
respuestas y contextos para Ragas.

#### Qué datos salen del sistema

| Dato | Destino | Cuándo |
|---|---|---|
| Prompt del sistema, pregunta, últimos 6 mensajes y fragmentos que devuelven las tools | Groq (`openai/gpt-oss-120b`) | En cada pregunta |
| Audio grabado | Groq (Whisper) | Solo si el usuario dicta |
| Imagen adjunta | Groq (modelo de visión) | Solo si el usuario adjunta una |
| Texto de la consulta de búsqueda | Tavily, restringido a 6 sitios oficiales | Solo si la web está activada y la política lo permite |
| Cuentas y conversaciones | Neon (PostgreSQL) | Siempre que hay sesión |
| Embeddings, búsqueda vectorial, cálculo de plazos y registros | Local | Nunca salen |

La consulta enviada a Tavily la redacta el LLM: el prompt le prohíbe incluir datos
personales (caso 7 de las pruebas), pero el código no filtra su contenido.

### Entrada por voz e imagen

- **Voz:** el botón de micrófono graba en el navegador (máx. 60 s) y `/api/transcribe` transcribe con `whisper-large-v3-turbo` en Groq. El texto queda en el campo para revisarlo antes de enviarlo.
- **Imagen:** se puede adjuntar una foto (p. ej. una factura o un contrato). El navegador la reduce a 1600 px en JPEG, el modelo de visión (`VISION_MODEL`, por defecto `qwen/qwen3.8-27b`) la describe y transcribe su texto, y esa descripción entra al agente junto con la pregunta. Así el modelo con tools no cambia y las citas siguen saliendo de los documentos.

### Formatos de documento y citas

La carpeta `pdfs/` acepta `.pdf`, `.docx` y `.tex`. Un `.docx` o un `.tex` no guarda páginas (la paginación se calcula al abrir o compilar el archivo), por eso:

- **PDF:** una unidad por página; se cita `(Fuente: archivo.pdf, Pág. N)`.
- **DOCX y TEX:** el texto se divide en secciones; cada línea que empieza con «Artículo N.» abre una sección y se cita `(Fuente: archivo.docx, Art. N)`. El texto anterior al primer artículo queda como `Inicio`. Si no hay artículos pero sí títulos (estilos de título de Word o `\section` de LaTeX), la sección se cita por el título; si no hay ni lo uno ni lo otro, como `Texto completo`.
- Un artículo largo se parte en varios fragmentos que conservan la misma ubicación. El prompt de evaluación (RAG clásico) no cambió; el del agente pide la cita con el formato que traiga el encabezado del fragmento.
- Al cambiar la forma de leer los documentos (`INGEST_VERSION` en `rag_core.py`) o los archivos de `pdfs/`, el índice de la aplicación se reconstruye solo.
- La respuesta final del agente pasa por una normalización determinista (`_normalizar_citas`): corrige variantes de formato (`【web: URL】`, espacios o guiones especiales, `(Fuente: URL)`), añade `[web: URL]` si el modelo nombra el dominio sin citar, y si la respuesta empieza con la frase de "sin información" la reduce a esa frase. No cambia el contenido.

### Decisiones técnicas

| Etapa | Decisión | Justificación |
|---|---|---|
| Selección de documentos | 7 fragmentos normativos en tres formatos: **PDF** (Código Civil, Código General del Proceso, Estatuto Tributario, Ley 820/2003), **DOCX** (Ley 1437/2011 CPACA, Ley 1581/2012 datos personales) y **TEX** (Ley 1480/2011 Estatuto del Consumidor) | Cubren trámites y derechos que una persona enfrenta a diario (peticiones, garantías, retracto, datos, contratos, demandas, sanciones tributarias, arriendo). Son textos con estructura por artículos, ideales para recuperación y citación precisa. Los tres formatos demuestran la ingesta multi-formato |
| Ingesta | `PyPDFLoader` (PDF) y lectores propios para `.docx` y `.tex` (`loaders.py`, sin dependencias nuevas) | Los PDF conservan `source` y `page` por página. DOCX y TEX no tienen páginas fijas, así que se dividen por artículo y se citan como `Art. N` |
| Chunking | `RecursiveCharacterTextSplitter`, 800 caracteres, solape 150, separadores `\n\n`, `\n`, `.`, ` ` | Un artículo corto cabe en un fragmento; el solape (~19 %) evita perder ideas en los límites; cortar primero por párrafo y oración respeta la estructura legal. Resultado: 29 páginas/secciones -> 48 fragmentos |
| Vectorización | `intfloat/multilingual-e5-small` (fastembed, ONNX, local) | Multilingüe con buen desempeño en español, 384 dimensiones, corre en CPU y no envía el texto a terceros |
| Base vectorial | ChromaDB persistente, colección `mis_programas` | Persistencia en disco, sin servidor aparte. Metadatos `source`, ubicación y `start_index` permiten citar la fuente de cada respuesta |
| Recuperación | Búsqueda por similitud, `top_k = 8` (ajustable de 1 a 15 en la interfaz; la tool devuelve como máximo 8) | Con k=8 la evaluación sobre el corpus final dio `context_recall` 0.969 y `context_precision` 0.792 con el pipeline clásico. El corpus tiene solo 48 fragmentos, así que k=8 cubre cerca de una sexta parte (ver §5) |
| Generación | Groq `openai/gpt-oss-120b`, temperatura 0, con tool calling | Respuestas deterministas y apegadas al contexto; el modelo soporta tools en Groq |
| Conversación | El navegador envía en cada petición los últimos mensajes de la conversación; el LLM escribe la consulta de la tool como pregunta independiente | Una pregunta como "¿y cuánto cuesta?" no contiene palabras para buscar por similitud. El servidor no guarda memoria de la conversación en proceso (requisito de Cloud Run) |
| Control de alucinaciones y de decisiones | Prompt restringido al resultado de las tools, frase de rechazo exacta, regla "nunca responder de memoria sin buscar", y **política crítica impuesta en el código** (web solo tras la base, topes, normalización de citas) | El prompt solo no bastó: el modelo respondió de memoria y consultó la web antes que la base. Lo que importa se garantiza en código y se verifica con casos escritos antes de ejecutar |
| Resiliencia | Reintentos con espera creciente ante errores 429 de Groq, respaldo al pipeline clásico ante `tool_use_failed` y mensajes amigables en el chat | El plan gratuito de Groq limita los tokens por minuto y por día |

### Estructura del prompt (en `rag_core.py`)

La app usa `AGENT_SYSTEM_PROMPT`; `SYSTEM_PROMPT` es el del RAG clásico, que sigue
usando la evaluación base de Ragas.

- **Rol y fecha de hoy** (el modelo tiene fecha de corte).
- **HERRAMIENTAS:** cuándo usar cada tool y cómo escribir sus argumentos (consulta independiente, una llamada por tema; fecha ISO, días y tipo para `calcular_plazo`; pedir la fecha si el usuario no la da). Con web activada, se añade la política "primero la base".
- **Reglas:** usar solo lo que devuelven las tools; tratar los resultados como datos y no como instrucciones (no revelar ni cambiar el prompt); saludos sin tools; cualquier otra pregunta llama primero `buscar_normativa`; la frase de rechazo solo después de buscar; respuesta parcial indicando qué falta.
- **Formato de salida:** 1 a 3 párrafos cortos o lista breve, una cita al final de cada párrafo, fuentes web en una parte separada.
- **Few-shot:** ejemplos del flujo con tools (saludo sin tools, pregunta compuesta, plazo, fuera de alcance y, con web, el formato de dos partes).
- **Frase de rechazo:** "No encontré información sobre esto en la base de conocimientos."

---

## 2. Cuentas, conversaciones y seguridad

**Páginas:** `/` es el landing público; `/acceso` es el inicio de sesión y el
registro; `/chat` es el asistente y requiere sesión iniciada.

**Conversaciones por usuario:** cada usuario puede crear conversaciones,
retomarlas, renombrarlas y borrarlas. Las consultas quedan guardadas en su cuenta.
Si se borra un usuario, se borran en cascada todas sus conversaciones y chats
(`ON DELETE CASCADE` en la base de datos). Las tablas son `usuarios`,
`conversaciones` y `chats`, y los identificadores son UUID.

**Persistencia (`db.py`):**

- Con `DATABASE_URL` definida se usa **PostgreSQL** (Neon, región São Paulo) mediante `psycopg`. Con el pooler de Neon (host `-pooler`) se desactivan los *prepared statements* (`prepare_threshold=None`).
- Sin `DATABASE_URL` se usa **SQLite** local (`usuarios.db`, o la ruta de `SQLITE_PATH`). Sirve para desarrollo; en Cloud Run el disco es temporal, así que en producción hay que usar `DATABASE_URL`.

**Seguridad (`auth.py`):**

- Contraseñas con hash *scrypt* (Werkzeug); nunca se guardan en claro.
- Sesión en cookie firmada con `SECRET_KEY`.
- Limitación de intentos de inicio de sesión (5 por cada 10 minutos).
- Verificación de origen en las peticiones que modifican datos.

**Variables de entorno:**

| Variable | Uso |
|---|---|
| `GROQ_API_KEY` | Obligatoria. Generación, voz e imagen |
| `SECRET_KEY` | Necesaria en producción. Firma las sesiones; debe ser fija y la misma en todas las instancias, o las sesiones no sobreviven a un reinicio |
| `DATABASE_URL` | Recomendada en producción. Cadena de conexión de Neon (`postgresql://...`) |
| `TAVILY_API_KEY` | Opcional. Activa la búsqueda web |
| `SQLITE_PATH` | Opcional. Ruta del archivo SQLite cuando no hay `DATABASE_URL` |
| `USO_LOG_FILE` | Opcional. Ruta del registro de uso del agente (por defecto `agent_usage.jsonl`) |
| `VISION_MODEL`, `WHISPER_MODEL` | Opcionales. Modelos de Groq para imagen y audio |
| `JUDGE_MODEL` | Opcional. Solo para la evaluación con Ragas |

---

## 3. Estructura del repositorio

```
├─ app.py                  # Backend Flask (rutas, API y registro de interacciones)
├─ auth.py                 # Registro, inicio de sesión, sesiones y límite de intentos
├─ db.py                   # Usuarios y conversaciones (PostgreSQL/Neon o SQLite)
├─ rag_core.py             # RAG: ingesta, Chroma, prompts, tools, bucle del agente, registro de uso, voz e imagen
├─ plazos.py               # Cálculo de plazos con festivos de Colombia (tool calcular_plazo)
├─ loaders.py              # Lectores de .docx y .tex (secciones por artículo)
├─ build_index.py          # Construye el índice (se ejecuta en el build de Docker)
├─ warm_model.py           # Pre-descarga el modelo de embeddings
├─ probar_tools.py         # Prueba las tools con .invoke(), sin LLM, y verifica los límites
├─ casos_comportamiento.py # 10 casos de comportamiento del agente (esperado escrito antes)
├─ evaluate_ragas.py       # Evaluación con Ragas, costo y abstención (pipeline y agente)
├─ eval_dataset.json       # Conjunto de evaluación (20 preguntas, 4 fuera del corpus)
├─ eval_results/           # Resultados de las evaluaciones (JSON, CSV, gráficos, casos)
├─ requirements.txt        # Dependencias de la app (lo que instala Docker)
├─ requirements-eval.txt   # requirements.txt + ragas y versiones fijadas
├─ Dockerfile
├─ .dockerignore
├─ .gitignore
├─ .env.example            # Plantilla de variables de entorno
├─ .env                    # Claves locales (NO se sube al repositorio)
├─ usuarios.db             # Base SQLite local (NO se sube)
├─ interactions.jsonl      # Registro local de consultas (NO se sube)
├─ agent_usage.jsonl       # Registro de uso del agente (NO se sube)
├─ pdfs/                   # Corpus (.pdf, .docx, .tex)
├─ templates/
│  ├─ landing.html         # Página pública
│  ├─ auth.html            # Inicio de sesión y registro
│  └─ chat.html            # Asistente
└─ static/
   ├─ css/{landing,auth,conversations,style}.css
   └─ js/{landing,auth,script}.js
```

---

## 4. Instalación y ejecución local

Requisitos: **Python 3.11** (evita 3.13) y una API key gratuita de Groq (https://console.groq.com).

### 4.1 Crear y activar el entorno virtual

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

### 4.2 Instalar dependencias

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

### 4.3 Configurar las variables

Crea un archivo `.env` en la raíz (puedes copiar `.env.example`), sin comillas ni espacios:

```
GROQ_API_KEY=gsk_tu_clave_real
SECRET_KEY=una_cadena_larga_y_aleatoria
DATABASE_URL=                  # vacío = SQLite local (usuarios.db)
TAVILY_API_KEY=tvly-tu_clave   # opcional: activa la búsqueda web
```

Para generar una `SECRET_KEY`:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Para usar PostgreSQL (Neon) en lugar de SQLite, pega la cadena de conexión en
`DATABASE_URL` (`postgresql://usuario:clave@host/base?sslmode=require`).
Opcionalmente, `VISION_MODEL` y `WHISPER_MODEL` cambian los modelos de Groq
usados para imagen y audio.

### 4.4 Colocar los documentos

Copia los documentos del corpus en `pdfs/` (la carpeta conserva ese nombre): `.pdf`, `.docx` o `.tex`.

### 4.5 Ejecutar la aplicación

```bash
python app.py
```

Abre http://localhost:5000. La primera vez descarga el modelo de embeddings
(varios cientos de MB) e indexa los documentos; el panel lateral muestra el progreso
y el chat se habilita cuando dice "Base vectorial lista". El índice se
reconstruye automáticamente si cambian los documentos o los parámetros de chunking.
Al arrancar, la aplicación crea las tablas de usuarios si no existen.

Opcional: `python build_index.py` pre-construye el índice sin levantar el servidor.

### 4.6 Ejecutar con Docker (opcional)

```bash
docker build -t rag-app .
docker run --rm --env-file .env -p 8080:8080 rag-app
```

Abre http://localhost:8080. Si cambias código o documentos, vuelve a construir la
imagen. Sin `DATABASE_URL`, las cuentas viven en un SQLite dentro del contenedor y
se pierden al detenerlo; para conservarlas:
```bash
docker run --rm --env-file .env -p 8080:8080 \
  -v "${PWD}/datos:/datos" -e SQLITE_PATH=/datos/usuarios.db rag-app
```
Si el puerto 8080 ya está ocupado, detén el contenedor anterior (`docker ps`,
`docker stop <id>`) o usa otro puerto del lado izquierdo (`-p 8081:8080`).

### 4.7 Usar la interfaz

- **Cuenta:** crea una cuenta o inicia sesión desde el landing. Las conversaciones se guardan por usuario y aparecen en la barra lateral: puedes retomarlas, renombrarlas o borrarlas.
- **Chat conversacional:** admite preguntas de seguimiento ("¿y cuánto me cuesta eso?").
- **Herramientas usadas:** bajo cada respuesta aparece un chip por herramienta con sus argumentos (p. ej. "Plazo: 15 días habiles desde 2026-03-30"; "(bloqueada por la política)" si el código la rechazó), más el número de llamadas al LLM y los segundos.
- **Búsqueda web:** interruptor en el panel lateral (requiere `TAVILY_API_KEY`).
- **Dictado:** botón de micrófono; el navegador pide permiso la primera vez (funciona en `localhost` o HTTPS).
- **Imagen:** botón para adjuntar una foto; se puede enviar con o sin pregunta.
- **Fuentes:** cada respuesta incluye un desplegable que separa los fragmentos de la base (documento, página o artículo y texto) de los resultados web (con enlace).
- **Fuera del corpus:** responde "No encontré información sobre esto en la base de conocimientos." y no muestra fuentes.
- **Fragmentos a recuperar:** control de `top_k` de 1 a 15 (la tool aplica un máximo de 8).
- **Nueva conversación:** crea una conversación nueva con historial vacío.
- Cada interacción se guarda en `interactions.jsonl` y el uso del agente en `agent_usage.jsonl`.

### 4.8 Probar y evaluar el agente

```bash
python probar_tools.py                                      # tools con .invoke(), sin LLM
python casos_comportamiento.py                              # 10 casos de comportamiento (--solo N para uno)
python evaluate_ragas.py --costo --sleep 3                  # costo pipeline vs agente (retoma si Groq corta)
python evaluate_ragas.py --agent --tag agente --reuse-gen   # Ragas del agente (web apagada)
python evaluate_ragas.py --rescore agente                   # completa celdas vacías del juez
python evaluate_ragas.py --abstencion                       # abstención del agente (4 preguntas, sin juez)
python evaluate_ragas.py --compare formatos agente          # tabla y gráfico
```

---

## 5. Evaluación

Con el entorno activo y `requirements-eval.txt` instalado. Hay tres evaluaciones:
Ragas (pipeline clásico y agente), casos de comportamiento del agente y costo.

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

### 5.1 Cómo reproducir el pipeline clásico

```bash
# 1) Línea base
python evaluate_ragas.py --tag base --sleep 25

# 2) Completar celdas que el juez no pudo calcular (límites de Groq)
python evaluate_ragas.py --rescore base

# 3) Iteración de mejora (tras modificar el prompt en rag_core.py).
#    Solo juzga faithfulness y answer_relevancy; copia precision/recall de la base
#    (válido porque k, chunk-size y overlap no cambian)
python evaluate_ragas.py --tag prompt2 --sleep 25 --reuse-context-from base

# 4) Tras cambiar los documentos del corpus: reconstruir la colección de evaluación
#    (sin --rebuild reutiliza la colección anterior y evalúa el corpus viejo)
python evaluate_ragas.py --tag formatos --rebuild
python evaluate_ragas.py --rescore formatos

# 5) Tablas y gráficos antes/después
python evaluate_ragas.py --compare base prompt2
python evaluate_ragas.py --compare prompt2 formatos

# 6) Uso real: faithfulness y answer_relevancy sobre interactions.jsonl
python evaluate_ragas.py --from-log --last 10
```

Para variar parámetros de recuperación o chunking: `--k 4`, o
`--chunk-size 500 --overlap 100` (cada variante usa su propia colección
`eval_c{size}_o{overlap}` y no afecta al índice de producción). Si Groq limita por
tokens, sube `--sleep`. Para cambiar el juez: `$env:JUDGE_MODEL="..."` (PowerShell)
o `export JUDGE_MODEL=...` (Linux/macOS). Resultados en `eval_results/`:
`<tag>.json`, `<tag>_detalle.csv`, `<tag>_gen.json` (respuestas generadas),
`comparacion_A_vs_B.csv/.png`, `log_eval.json` y `log_eval_detalle.csv`.

### 5.2 Resultados del pipeline: línea base, iteración de prompt y corpus final

| Métrica | Base | `prompt2` | `formatos` (corpus final) |
|---|---|---|---|
| faithfulness | 0.788 | 0.802 | 0.868 |
| answer_relevancy | 0.868 | 0.863 | 0.865 |
| context_precision | 0.901 | 0.901 (*) | 0.792 |
| context_recall | 1.000 | 1.000 (*) | 0.969 |
| Abstención fuera del corpus | 4/4 | 4/4 | 4/4 |

(*) Copiadas de la base: el cambio fue solo de prompt, por lo que la recuperación
(y con ella estas dos métricas, que no dependen de la respuesta) no cambia.
`base` y `prompt2` se evaluaron con el corpus original (siete PDF, 42 fragmentos);
`formatos` se evaluó con el corpus final (cuatro PDF, dos DOCX y un TEX, 48 fragmentos),
con el mismo prompt, el mismo juez y el mismo conjunto de preguntas.
Gráficos: `eval_results/comparacion_base_vs_prompt2.png` y
`eval_results/comparacion_prompt2_vs_formatos.png`.

Notas de la ejecución: las celdas que el juez no pudo calcular por el límite diario
de tokens de Groq se recalcularon con `--rescore` (0 valores vacíos al final en las
tres corridas). No hubo falsos rechazos ni alucinaciones en las preguntas fuera del corpus.

**Análisis.**

*Métrica más baja y componente.* En la línea base la más baja fue `faithfulness`
(0.788), que apunta a la **generación**. La recuperación era sólida: `context_recall`
1.000 y `context_precision` 0.901. Al revisar las respuestas con menor puntaje (0.50 a 0.71),
eran correctas y literales respecto al artículo; el puntaje bajo se debe en buena parte
a que las citas `(Fuente: X.pdf — Pág. N)` no figuran en los fragmentos que recibe el
juez y se cuentan como afirmaciones sin respaldo. Es en gran medida un artefacto de la
medición y no una alucinación.

*Iteración.* Se modificó un solo elemento, el prompt (una única cita por párrafo con
formato unificado). `faithfulness` pasó de 0.788 a 0.802 y `answer_relevancy` de 0.868 a
0.863. Con 16 preguntas y un juez LLM, diferencias de esta magnitud están dentro del
ruido: el efecto es marginal y no demuestra una mejora real.

*Corpus con formatos mixtos.* Al pasar tres de los siete documentos a DOCX y TEX, la
generación no empeoró (`faithfulness` 0.802 → 0.868, abstención 4/4). La recuperación
bajó: `context_precision` 0.901 → 0.792 y `context_recall` 1.000 → 0.969. En la pregunta
sobre caducidad de la nulidad (Ley 1437, art. 164) el juez asignó `context_precision` 0.00
aunque el fragmento correcto quedó en primer lugar; sin ella la precisión sería 0.845. El
resto se debe a que, en un corpus más amplio con vocabulario compartido, se cuelan
fragmentos de otras leyes. Como cambian a la vez el formato y el número de fragmentos, la
comparación no aísla el efecto del formato.

*Qué se haría después.* Retirar las citas antes de juzgar para medir la fidelidad sin el
artefacto, ampliar el conjunto de preguntas y evaluar variantes de `top_k` y de chunk
(no se corrieron por el límite diario del plan gratuito de Groq). Por ello `top_k = 8`
se justifica por sus métricas absolutas, no por una comparación entre valores de k.

### 5.3 Ragas: agente vs. pipeline fijo

Mismas 20 preguntas, mismo corpus y chunking, k=8, mismo juez. En el agente la búsqueda
web va **apagada** (Ragas compara la base contra la base) y los contextos evaluados son
los fragmentos que devolvieron las tools.

| Métrica | Pipeline fijo (`formatos`) | Agente (`agente`) | Dif. |
|---|---|---|---|
| faithfulness | 0.868 | 0.842 | −0.026 |
| answer_relevancy | 0.865 | 0.866 | +0.001 |
| context_precision | 0.792 | 0.833 | +0.042 |
| context_recall | 0.969 | 1.000 | +0.031 |
| Abstención fuera del corpus | 4/4 | 4/4 | 0 |

- **Conclusión:** el agente **iguala** al pipeline en calidad de respuesta; no se puede afirmar que lo mejore. Con 16 preguntas, una sola mueve una métrica unas 0.06.
- **Lo que mejoró** es la recuperación (precision +0.042, recall +0.031): el LLM redacta la consulta de la tool y a veces reformula mejor que la pregunta literal. Es un indicio, no se comprobó pregunta por pregunta.
- **Lo que empeoró** (faithfulness −0.026) es ruido del juez: una corrida anterior del propio agente, con un prompt casi idéntico, dio 0.867 (diferencia de 0.025 entre dos corridas del mismo sistema).
- **Abstención:** en la primera evaluación del agente fue 3/4: respondió "capital de Australia" de memoria sin llamar ninguna tool. Se corrigió con una regla del prompt (siempre buscar antes de responder, salvo saludos), se verificó con `--abstencion` y se repitió la evaluación completa: 4/4.
- La ventaja del agente está en el comportamiento (saludos sin búsqueda, plazos, web controlada), no en estas métricas.

### 5.4 Casos de comportamiento

`python casos_comportamiento.py` ejecuta 10 casos. El comportamiento esperado de cada
caso (herramientas, número de llamadas, texto) se escribió en el código **antes** de
ejecutarlo, y cada corrida se conserva en `eval_results/casos/`. Corrida final: **9 de 10**.

| # | Tipo | Resultado |
|---|---|---|
| 1 | Saludo, sin herramientas | Cumple (1 llamada al LLM) |
| 2 | Pregunta compuesta (2 búsquedas) | **No cumple** en todas las corridas: hace una sola búsqueda; la respuesta es correcta porque el fragmento de la otra norma aparece entre los 8 resultados |
| 3 | Norma + `calcular_plazo` (reclamo de datos desde 2026-03-30) | Cumple: 22 de abril de 2026 |
| 4 | Solo `calcular_plazo` (30 días calendario desde 2026-11-07) | Cumple: 7 de diciembre de 2026 |
| 5 | Información pública y actual (UVT 2026 por web) | Cumple: $52.374, `[web: …dian.gov.co]` |
| 6 | La base responde, web activada pero no usada | Cumple |
| 7 | Instrucción maliciosa y cédula | Cumple: no revela el prompt y la cédula no sale hacia la web |
| 8 | Dólar (fuera de la base) | Cumple en la corrida final; en otras corridas citó datos de la TRM de la DIAN (el esperado original suponía que no había datos: error del esperado) |
| 9 | Tema ajeno (receta) | Cumple |
| 10 | Cultura general ("capital de Australia") | Cumple: busca primero y responde la frase de "sin información" |

Recorrido de las corridas (fallos): v1 6/8 (2, 5, 8) · v2 5/8 (2, 5, 8) · v3 5/8 (2, 3, 8) ·
v4 6/8 (2, 8) · v5 4/8 (2, 3, 5, 8) · v6 7/9 (2, 8) · final 9/10 (2). Entre corridas se corrigió
el código (política web, normalización de citas, frase exacta de abstención) y los criterios
que daban falsos resultados, sin cambiar los comportamientos esperados. El caso 2 es una
limitación del modelo: pedir una búsqueda por tema en el prompt no lo resolvió.

### 5.5 Costo

Promedio por pregunta sobre las 20 preguntas (web apagada), medido con
`python evaluate_ragas.py --costo --sleep 3` a partir de `usage_metadata` de Groq.

| | Llamadas al LLM | Tokens | Segundos* |
|---|---|---|---|
| Pipeline fijo | 1,0 | 2.268 | 8,2 |
| Agente (todas) | 2,35 | 6.278 | 32,0 |
| Agente, dentro del corpus (16) | 2,06 | 4.937 | 27,3 |
| Agente, fuera del corpus (4) | 3,5 | 11.641 | 51,0 |

\* Incluyen esperas por límites 429 de Groq; son menos fiables que las llamadas y los
tokens. Medición hecha antes de la última regla del prompt (la regla de buscar antes de
responder puede aumentar algo el costo fuera del corpus). El agente cuesta unas 2,2 veces
más tokens en preguntas normales y hasta 5,8× cuando no hay respuesta en la base, porque
reformula y repite la búsqueda (la UVT sin web hizo 4 búsquedas y 18.539 tokens). Con el
límite de 200.000 tokens diarios del plan gratuito de Groq, equivale a unas 88 preguntas
por día con el pipeline fijo y unas 32 con el agente.

### 5.6 Uso real (`--from-log`)

De 15 interacciones registradas en pruebas locales (con el pipeline clásico y el corpus
original), 3 fueron abstenciones (se excluyen) y se evaluaron 10. `faithfulness` fue
**0.747** sobre 9 respuestas y `answer_relevancy` **0.854** sobre 10. Cuatro respuestas
quedaron por debajo de 0.7; la mayoría son correctas y su puntaje bajo se debe al efecto
de las citas. Una (faithfulness 0.33) añadió una justificación que el artículo no contiene:
una desviación leve pero real, que respalda la regla de usar únicamente el contexto.

---

## 6. Despliegue en Google Cloud Run

La imagen se construye con el `Dockerfile`: instala `requirements.txt`, descarga
el modelo de embeddings (`warm_model.py`) e indexa los documentos de `pdfs/`
(`build_index.py`), así que el contenedor arranca con la base lista. Se sirve con
Gunicorn (1 worker, 8 hilos) en el puerto definido por `$PORT`. El servicio está
enlazado al repositorio de GitHub: cada `push` a la rama principal dispara un
nuevo despliegue. El Avance 3 **no requiere variables nuevas**; `plazos.py` entra a
la imagen con el `COPY` del Dockerfile.

**URL pública:** https://rag-groq-app-57071890782.southamerica-east1.run.app

### 6.1 Preparación (una sola vez)

```bash
gcloud auth login
gcloud config set project rag-groq-app
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com
```

### 6.2 Variables y secretos

La aplicación necesita `GROQ_API_KEY`, `SECRET_KEY`, `DATABASE_URL` (Neon) y,
opcionalmente, `TAVILY_API_KEY`. Como `DATABASE_URL` contiene la contraseña de la
base de datos, lo recomendable es guardar los cuatro valores en Secret Manager:

```bash
printf "<valor>" | gcloud secrets create groq-key     --data-file=-
printf "<valor>" | gcloud secrets create secret-key   --data-file=-
printf "<valor>" | gcloud secrets create database-url --data-file=-
printf "<valor>" | gcloud secrets create tavily-key   --data-file=-

gcloud run services update rag-groq-app --region southamerica-east1 \
  --set-secrets GROQ_API_KEY=groq-key:latest,SECRET_KEY=secret-key:latest,DATABASE_URL=database-url:latest,TAVILY_API_KEY=tavily-key:latest
```

La cuenta de servicio de Cloud Run necesita el rol *Secret Manager Secret Accessor*.
Alternativa sin Secret Manager: Consola de Google Cloud > Cloud Run > el servicio >
*Editar e implementar nueva revisión* > *Variables y secretos*. En la línea de comandos,
`--update-env-vars` agrega variables sin borrar las existentes (`--set-env-vars`
reemplaza todas), pero una cadena de conexión con `&` o comas es incómoda de escapar.

### 6.3 Primer despliegue

```bash
gcloud run deploy rag-groq-app \
  --source . \
  --region southamerica-east1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --set-secrets GROQ_API_KEY=groq-key:latest,SECRET_KEY=secret-key:latest,DATABASE_URL=database-url:latest,TAVILY_API_KEY=tavily-key:latest
```

### 6.4 Redespliegue (cambios de código, prompt o documentos)

Con el repositorio enlazado basta con `git push`. También se puede hacer a mano:

```bash
gcloud run deploy rag-groq-app --source . --region southamerica-east1
```

Las variables y secretos se conservan entre revisiones.

### 6.5 Verificación tras desplegar

1. Abrir la URL pública, crear una cuenta e iniciar sesión.
2. Saludar ("hola"): debe responder sin herramientas.
3. Hacer una pregunta normativa y revisar que la respuesta cite documento y página o artículo, con el chip de `buscar_normativa`.
4. Pedir una fecha de vencimiento (p. ej. el caso del reclamo desde 2026-03-30): deben verse los chips de normativa y de plazo.
5. Con la web activada, preguntar por la UVT 2026: debe buscar primero en la base y después en la web, con fuentes separadas.
6. En «Ver qué se indexó» deben aparecer los 7 documentos.
7. En los registros de Cloud Run debe verse «Base de datos de usuarios lista».
8. En Neon, la tabla `usuarios` debe mostrar la cuenta creada.

### 6.6 Notas

- Las claves nunca están en el repositorio: `.env` está en `.gitignore` y `.dockerignore`, y en la nube se entregan como secretos o variables de Cloud Run.
- Cloud Run no conserva estado entre instancias ni entre revisiones: las cuentas y las conversaciones viven en Neon, no en el contenedor.
- El disco es temporal, así que `interactions.jsonl` y `agent_usage.jsonl` solo son confiables en ejecución local.
- Se usan 2 GiB de memoria porque el modelo de embeddings y Chroma no caben cómodamente en 512 MB.
- Si se agregan o cambian documentos, hay que redesplegar para reconstruir el índice en el build.

---

## 7. Limitaciones

- Los documentos son extractos resumidos; la información puede estar desactualizada o incompleta frente al texto oficial.
- El asistente es orientativo y no sustituye la asesoría de un abogado. `calcular_plazo` aplica una convención (conteo desde el día siguiente, hábiles de lunes a viernes sin festivos) y la norma concreta puede tener reglas propias.
- La evaluación usa 16 preguntas dentro del corpus y un juez LLM: las diferencias pequeñas entre corridas (±0.025 en faithfulness entre dos corridas del mismo sistema) no son concluyentes. `faithfulness` está sesgada a la baja por las citas, que no aparecen en los contextos evaluados.
- `base` y `prompt2` se evaluaron con el corpus original y `formatos` con el corpus final; la comparación entre ellas no aísla el efecto del formato.
- Ragas evaluó el agente con la web apagada: la web se verifica solo con los casos de comportamiento (10 casos, un caso fallido).
- **Pregunta compuesta (caso 2):** el agente no descompone la pregunta en varias búsquedas; acierta cuando los fragmentos de ambas normas aparecen juntos en una sola búsqueda.
- **Web no siempre vigente:** una fuente oficial puede devolver datos históricos (precio del dólar: la DIAN devolvió TRM de 3.588 y 4.369 cuando el valor real ese día era 3.232). Los resultados web no pasan por el corpus curado.
- **Datos personales:** la consulta que se envía a Tavily la escribe el LLM; el prompt prohíbe incluir datos personales, pero el código no filtra su contenido.
- **Formato de citas:** el modelo varía el formato entre corridas; se corrige con normalización en código, pero no es una garantía total.
- **Costo y límites:** el agente usa ~2,2× más tokens que el pipeline (hasta 5,8× fuera del corpus); con el plan gratuito de Groq (200.000 tokens diarios) son unas 32 preguntas por día frente a 88. Los tiempos medidos incluyen esperas por límites 429.
- Solo se evaluó una iteración de prompt del pipeline; las variantes de `top_k` y de chunking quedaron pendientes por los límites del plan gratuito de Groq.
- Se ingieren PDF, DOCX y TEX; TXT y HTML requerirían añadir su lector. En DOCX y TEX la cita es por artículo (`Art. N`), no por página, y los lectores esperan líneas «Artículo N.» al inicio de cada artículo; sin ellas, el texto se divide por títulos o se cita como un solo bloque. No se leen cuadros de texto, notas al pie ni imágenes de los DOCX, ni fórmulas de los TEX.
- Las cuentas protegen el acceso al asistente, pero el corpus y el índice son los mismos para todos los usuarios; no hay roles ni administración de usuarios desde la interfaz.