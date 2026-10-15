# doc-summarizer/Dockerfile
# Packages the FastAPI service (core/ + api/) into a container.

FROM python:3.11-slim

WORKDIR /app

# Only requirements.txt is copied first -- thanks to Docker layer caching,
# pip install is not rerun just because the code changed; it only reruns
# when requirements.txt itself changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core/ ./core/
COPY api/ ./api/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
