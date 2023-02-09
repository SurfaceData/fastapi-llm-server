FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Ensure the libcuda* libraries are available.
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib

WORKDIR /app

# Copy over source code.
COPY ./requirements.txt /app/requirements.txt

# Install all requirements.
RUN --mount=type=cache,target=/root/.cache \
  pip install -r /app/requirements.txt

# Copy over the service files.
COPY ./main.py /app/main.py
COPY ./llm_server /app/llm_server

# Used by the container when starting up the service.
COPY ./prestart.sh /app/prestart.sh

# Set server ports.
ENV PORT 8080
EXPOSE 8080

# Run!
CMD ["uvicorn", "main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8080"]
