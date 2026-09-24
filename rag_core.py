"""
Núcleo del sistema RAG (Retrieval Augmented Generation).

Pipeline:
  PDF -> Carga -> Chunking -> Embeddings locales (Sentence Transformers)
      -> ChromaDB -> Retrieval -> Prompt aumentado -> Groq LLM -> Respuesta

Este módulo replica exactamente la lógica del notebook original
(flujo_rag_groq.ipynb), pero encapsulada en funciones reutilizables
para poder llamarse desde una interfaz gráfica (Streamlit).
"""

import chromadb
import os
import shutil
from typing import Callable, Optional

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# ──────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────

load_dotenv()  # Carga variables desde el archivo .env

PDF_DIR = "pdfs"
PERSIST_DIR = "./chroma"
COLLECTION_NAME = "mis_programas"

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

DEFAULT_K = 8

_chroma_client = None


def _get_chroma_client(persist_dir: str = PERSIST_DIR):
    """
    Cliente de Chroma único y reutilizable para todo el proceso. Crear
    varios PersistentClient distintos apuntando a la misma carpeta dentro
    del mismo proceso de Python es lo que causa los bloqueos de archivo en
    Windows (WinError 32): Chroma cachea el cliente internamente, así que
    reutilizar siempre la misma instancia evita el problema.
    """
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=persist_dir)
    return _chroma_client

PROMPT_TEMPLATE = '''Eres un asistente legal experto en reglamentos y normas.
Responde la pregunta usando ÚNICAMENTE la información del contexto proporcionado. Sintetiza y relaciona la información de los distintos fragmentos cuando haga falta para dar una respuesta completa, en vez de citar un solo fragmento aislado. Al final de cada parte de la respuesta, incluye entre paréntesis la fuente y la página del fragmento de donde proviene la información, por ejemplo: (Fuente: NombreDocumento.pdf, Pág. X).
Si una parte específica de la pregunta no está cubierta por el contexto, indícalo brevemente en esa parte en vez de negar toda la respuesta. Si NADA de lo preguntado está en el contexto, responde exactamente: "No encontré información sobre esto en la base de conocimientos."

Contexto recuperado del documento:
{context}

Pregunta del usuario: {question}

Respuesta:'''


def get_groq_api_key() -> str:
    """Obtiene la API key de Groq desde las variables de entorno (.env)."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "gsk_PEGA_TU_KEY_AQUI":
        raise ValueError(
            "No se encontró GROQ_API_KEY. Crea un archivo .env en la raíz del "
            "proyecto (puedes copiar .env.example) y pega ahí tu API key de Groq."
        )
    return api_key


def list_pdfs(pdf_dir: str = PDF_DIR) -> list:
    """Lista los PDFs disponibles en el directorio indicado."""
    if not os.path.isdir(pdf_dir):
        os.makedirs(pdf_dir, exist_ok=True)
        return []
    return sorted([f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")])

def build_vector_store(
    pdf_dir: str = PDF_DIR,
    persist_dir: str = PERSIST_DIR,
    progress_cb: Optional[Callable[[str], None]] = None,
):
    """
    Ejecuta los pasos 1-4 del pipeline:
    Carga de PDFs -> Chunking -> Embeddings -> Almacenamiento en ChromaDB.

    progress_cb: función opcional para reportar avance (para la interfaz gráfica).
    """
    def log(msg: str):
        if progress_cb:
            progress_cb(msg)

    pdf_files = list_pdfs(pdf_dir)
    if not pdf_files:
        raise FileNotFoundError(
            f"No se encontraron archivos PDF en la carpeta '{pdf_dir}/'. "
            "Coloca al menos un PDF ahí antes de construir la base de conocimiento."
        )

    # PASO 1 — Carga de documentos
    log(f"Cargando {len(pdf_files)} PDF(s)...")
    documents = []
    for pdf_file in pdf_files:
        path = os.path.join(pdf_dir, pdf_file)
        loader = PyPDFLoader(path)
        pages = loader.load()
        documents.extend(pages)
        log(f"  {pdf_file}: {len(pages)} páginas cargadas")

    # PASO 2 — Chunking
    log("Dividiendo documentos en fragmentos...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = text_splitter.split_documents(documents)
    log(f"  {len(documents)} páginas -> {len(chunks)} fragmentos")

    # PASO 3 — Embeddings locales
    log(f"Cargando modelo de embeddings local ({EMBEDDING_MODEL})...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

        # PASO 4 — Almacenamiento en ChromaDB
    client = _get_chroma_client(persist_dir)

    vector_store = Chroma(
        client=client,
        embedding_function=embeddings_model,
        collection_name=COLLECTION_NAME,
    )

    existentes = vector_store.get()
    ids_existentes = existentes.get("ids", [])
    if ids_existentes:
        vector_store.delete(ids=ids_existentes)
        log(f"  {len(ids_existentes)} fragmentos anteriores eliminados de la colección.")

    log(f"Indexando {len(chunks)} fragmentos en ChromaDB (puede tardar unos minutos)...")
    vector_store.add_documents(chunks)
    total = vector_store._collection.count()
    log(f"[OK] Base vectorial creada: {total} fragmentos indexados.")

    return vector_store, embeddings_model

def load_existing_vector_store(persist_dir: str = PERSIST_DIR):
    """Carga una base vectorial ya existente en disco, sin reprocesar los PDFs."""
    embeddings_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    client = _get_chroma_client(persist_dir)
    vector_store = Chroma(
        client=client,
        embedding_function=embeddings_model,
        collection_name=COLLECTION_NAME,
    )
    return vector_store, embeddings_model

def vector_store_exists(persist_dir: str = PERSIST_DIR) -> bool:
    return os.path.isdir(persist_dir) and len(os.listdir(persist_dir)) > 0


def get_llm(api_key: Optional[str] = None) -> ChatGroq:
    """Construye el cliente del LLM (Groq)."""
    api_key = api_key or get_groq_api_key()
    return ChatGroq(model=GROQ_MODEL, temperature=0.0, api_key=api_key)

def get_indexed_sources(vector_store) -> list:
    """
    Devuelve un resumen de qué archivos (y cuántos fragmentos de cada uno)
    quedaron efectivamente indexados en la base vectorial. Útil para
    diagnosticar si un PDF se indexó correctamente o no.
    """
    data = vector_store._collection.get(include=["metadatas", "documents"])
    metadatas = data.get("metadatas", []) or []
    documentos = data.get("documents", []) or []

    resumen = {}
    for meta, texto in zip(metadatas, documentos):
        fuente = os.path.basename(meta.get("source", "?"))
        if fuente not in resumen:
            resumen[fuente] = {"fragmentos": 0, "caracteres_totales": 0, "muestra": texto[:200]}
        resumen[fuente]["fragmentos"] += 1
        resumen[fuente]["caracteres_totales"] += len(texto)

    return [
        {"archivo": fuente, **datos}
        for fuente, datos in sorted(resumen.items())
    ]

def rag_pipeline(
    pregunta: str,
    vector_store,
    llm: ChatGroq,
    k: int = DEFAULT_K,
) -> dict:
    """
    Pipeline RAG de extremo a extremo (Pasos 5, 6 y 7):
    consulta -> recuperación (local) -> prompt aumentado -> respuesta (Groq).
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    docs = retriever.invoke(pregunta)

    contexto = "\n\n---\n\n".join(
        f"[Fuente: {os.path.basename(d.metadata.get('source', '?'))} — Pág. {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )

    prompt = prompt_template.invoke({"context": contexto, "question": pregunta})
    texto = llm.invoke(prompt).content

    return {
        "pregunta": pregunta,
        "fragmentos": docs,
        "contexto": contexto,
        "respuesta": texto,
        "tokens_contexto_aprox": len(contexto) // 4,
    }
