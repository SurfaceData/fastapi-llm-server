FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

# Copy over source code.
COPY ./requirements.txt /app/requirements.txt

# Install all requirements.
RUN --mount=type=cache,target=/root/.cache/pip pip \
  install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy over the service files.
COPY ./main.py /app/main.py
COPY ./llm_server /app/llm_server

# Used by the container when starting up the service.
COPY ./prestart.sh /app/prestart.sh

# Set server ports.
ENV PORT 8080
EXPOSE 8080

# Uvicorn will do the rest automatically.
