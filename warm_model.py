"""
Pre-descarga el modelo de embeddings durante el build de Render, para que
el primer 'Construir' en producción no tenga que descargarlo (evitando así
el pico de memoria de descargar + procesar al mismo tiempo).
"""
import rag_core

rag_core._ensure_embedding_model_registered()
rag_core.FastEmbedEmbeddings(model_name=rag_core.EMBEDDING_MODEL, batch_size=8).embed_query("prueba")
print("Modelo de embeddings pre-descargado correctamente.")