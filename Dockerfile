FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Descarga el modelo de embeddings en el build
RUN python warm_model.py

# Indexa los PDFs de pdfs/ en el build: la app arranca con la base lista
RUN python build_index.py

EXPOSE 8080

# 1 worker: cada worker cargaría su propia copia del modelo y de Chroma.
# Los hilos atienden varias peticiones en paralelo.
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 8 --timeout 300"]