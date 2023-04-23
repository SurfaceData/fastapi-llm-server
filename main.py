from fastapi import FastAPI
from starlette_exporter import PrometheusMiddleware, handle_metrics

from llm_server.routes import chat, generate


app = FastAPI()
app.add_middleware(PrometheusMiddleware, app_name="llm_server")
app.add_route("/metrics", handle_metrics)
app.include_router(chat.router)
app.include_router(generate.router)


@app.get("/health")
def health():
    """Response to basic health checks. Has no response content."""
    return {}
