from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM

logger.info("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
logger.info("Loading model")
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-neo-2.7B", device_map="auto", load_in_8bit=True
)
