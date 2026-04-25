# Backend Dockerfile
FROM python:3.13-slim

WORKDIR /app

# Установка uv
RUN pip install --no-cache-dir uv

# Копирование корневого pyproject.toml и workspace зависимостей
COPY pyproject.toml ./
COPY rag ./rag
COPY outlier-detection ./outlier-detection

# Копирование backend зависимостей
COPY backend/pyproject.toml ./backend/

# Установка зависимостей из backend
WORKDIR /app/backend
RUN uv pip install --system -e .
RUN uv pip install --system --group dev

# Копирование backend кода
COPY backend ./

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
