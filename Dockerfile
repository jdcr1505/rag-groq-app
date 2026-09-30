FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python warm_model.py

# Indexa los PDFs de pdfs/ en el build: la app arranca con la base lista.
RUN python build_index.py

EXPOSE 8080

CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300"]