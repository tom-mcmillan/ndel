FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies and ndel package
COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install --no-cache-dir .

ENTRYPOINT ["ndel-mcp"]
