from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM

from llm_server.config import settings

logger.info("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(
    settings.MODEL, cache_dir=settings.MODEL_CACHE_DIR
)
logger.info(f"Loading model on {settings.DEVICE}, GPU: {settings.USE_GPU}")
model = AutoModelForCausalLM.from_pretrained(
    settings.MODEL,
    device_map="auto" if settings.USE_GPU else None,
    load_in_8bit=settings.USE_GPU,
    cache_dir=settings.MODEL_CACHE_DIR,
)
