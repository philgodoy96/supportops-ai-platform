FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync \
    --frozen \
    --no-dev \
    --no-editable

RUN useradd \
    --create-home \
    --shell /usr/sbin/nologin \
    supportops

USER supportops

EXPOSE 8000

CMD [".venv/bin/uvicorn", "supportops.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
