import torch

from loguru import logger
from transformers import pipeline

from llm_server.config import settings

logger.info(f"Loading {settings.MODEL} on {settings.DEVICE}, GPU: {settings.USE_GPU}")
generator = pipeline(
    settings.MODEL_TYPE,
    model=settings.MODEL,
    tokenizer=settings.MODEL,
    device=settings.DEVICE,
    use_auth_token=settings.AUTH_TOKEN,
)
