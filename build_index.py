"""
Construye la base vectorial a partir de los PDFs precargados en pdfs/.

Se ejecuta durante el build del contenedor (ver Dockerfile) para que la
app arranque con el índice ya listo y no tenga que procesar los PDFs en
producción. También puede correrse a mano: python build_index.py
"""
import rag_core

rag_core.ensure_vector_store(progress_cb=print)
