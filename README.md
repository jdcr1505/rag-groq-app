# RAG con Groq — Flask + HTML/CSS/JS

Aplicación de **RAG (Retrieval Augmented Generation)** sobre archivos PDF, con:
- **Embeddings locales** (Sentence Transformers, corre en CPU, sin costo)
- **ChromaDB** como base vectorial local
- **Groq** como LLM
- **Backend Flask** + **frontend en HTML/CSS/JS puro** (carpetas `templates/` y `static/`), sin frameworks de frontend

Convertido desde el notebook `flujo_rag_groq.ipynb`.

## 1. Requisitos

- Python 3.10 o superior
- Visual Studio Code (con la extensión de Python instalada)

## 2. Instalación

Abre esta carpeta en VS Code y en la terminal integrada ejecuta:

```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## 3. Configurar la API key de Groq

1. Crea una cuenta gratuita en **https://console.groq.com**
2. Ve a **API Keys → Create API Key** y copia la key
3. Copia `.env.example` y renómbralo a `.env`
4. Pega tu key:

```
GROQ_API_KEY=gsk_tu_key_real_aqui
```

## 4. Agregar tus PDFs

Coloca tus PDFs en la carpeta `pdfs/` (o súbelos desde la propia interfaz web una vez esté corriendo).

## 5. Ejecutar la aplicación

```bash
python app.py
```

Abre en tu navegador: **http://localhost:5000**


Desde la interfaz:
1. Verifica que la API key se detectó correctamente (panel lateral).
2. Confirma o sube tus PDFs.
3. Presiona **"Construir"** para procesar los PDFs (chunking + embeddings + ChromaDB). Puedes ver el progreso en el recuadro de log.
4. Escribe tus preguntas en el cuadro de chat inferior.

Si ya construiste la base antes y solo reabres el proyecto, usa **"Cargar existente"** en vez de reconstruir desde cero.

## 6. Estructura del proyecto

```
rag_groq_flask/
├── app.py                 # Backend Flask (rutas /api/...)
├── rag_core.py             # Lógica del pipeline RAG (idéntica al notebook)
├── requirements.txt
├── .env.example
├── .env                     # (lo creas tú)
├── templates/
│   └── index.html           # Estructura de la página
├── static/
│   ├── css/style.css        # Estilos
│   └── js/script.js         # Lógica del frontend (llamadas a la API)
├── pdfs/                     # Coloca aquí tus PDFs
└── chroma/                    # Base vectorial (se genera sola)
```

## Notas

- El modelo de Groq usado por defecto es `openai/gpt-oss-120b`. Cámbialo en `rag_core.py` (variable `GROQ_MODEL`) si prefieres otro (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, etc.).
- El backend mantiene la base vectorial cargada en memoria mientras el proceso de Flask siga corriendo (variable `state` en `app.py`). Si reinicias `python app.py`, usa "Cargar existente" para no reprocesar los PDFs.
- Si ves `Acceso denegado` al reconstruir la base en Windows, cierra la app, espera unos segundos (por si OneDrive está sincronizando la carpeta `chroma/`) y vuelve a intentar.