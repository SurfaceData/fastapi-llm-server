from loguru import logger
from transformers import pipeline

from llm_server.config import settings
from llm_server.raven import create_raven_model
from llm_server.stable_lm import create_stablelm_model

logger.info(f"Loading")
if settings.MODEL_TYPE == "raven":
    generator = create_raven_model()
if settings.MODEL_TYPE == "stable_lm":
    generator = create_stablelm_model()
else:
    generator = pipeline(
        settings.MODEL_TYPE,
        model=settings.MODEL,
        tokenizer=settings.MODEL,
        device=settings.DEVICE,
        cache_dir=settings.MODEL_CACHE_DIR,
    )
