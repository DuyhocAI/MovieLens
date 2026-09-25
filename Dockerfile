FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MOVIE_ASSISTANT_HOST=0.0.0.0 \
    PORT=8765

WORKDIR /app
# The service uses only the standard library, so there is nothing to pip install.
COPY movie_assistant ./movie_assistant
COPY data ./data

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8765\")}/health', timeout=4)"

CMD ["python", "-m", "movie_assistant.web"]
