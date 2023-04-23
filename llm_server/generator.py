from loguru import logger
from transformers import pipeline

from llm_server.config import settings
from llm_server.raven import create_raven_model

logger.info(f"Loading")
if settings.MODEL_TYPE == "raven":
    generator = create_raven_model()
else:
    generator = pipeline(
        settings.MODEL_TYPE,
        model=settings.MODEL,
        tokenizer=settings.MODEL,
        device=settings.DEVICE,
        use_auth_token=settings.AUTH_TOKEN,
    )
