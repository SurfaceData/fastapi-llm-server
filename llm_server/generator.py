import torch

from pathlib import Path
from loguru import logger
from transformers import pipeline

from llm_server.config import settings

from speedster import optimize_model, save_model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

optimized_model_path = Path(f"{settings.MODEL_CACHE_DIR}/model_save_path")

if not optimized_model_path.exists():
    tokenizer = AutoTokenizer.from_pretrained(
        settings.MODEL,
        use_auth_token=settings.AUTH_TOKEN,
        cache_dir=settings.MODEL_CACHE_DIR,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        settings.MODEL,
        use_auth_token=settings.AUTH_TOKEN,
        cache_dir=settings.MODEL_CACHE_DIR,
    )
    question = "What's the meaning of life?"
    input_dict = tokenizer(question, return_tensors="pt")
    answer = "The answer is:"
    input_dict["decoder_input_ids"] = tokenizer(answer, return_tensors="pt").input_ids
    input_data = [input_dict for _ in range(100)]

    optimized_model = optimize_model(
        model,
        optimization_time="constrained",
        metric_drop_ths=0.05,
        input_data=input_data,
    )
    save_model(optimized_model, str(optimized_model_path))

logger.info(f"Loading {settings.MODEL} on {settings.DEVICE}, GPU: {settings.USE_GPU}")

from speedster import load_model

model = load_model(str(optimized_model_path))
tokenizer = AutoTokenizer.from_pretrained(
    settings.MODEL,
    use_auth_token=settings.AUTH_TOKEN,
    cache_dir=settings.MODEL_CACHE_DIR,
)

# generator = pipeline(
#    settings.MODEL_TYPE,
#    model=optimized_model,
#    tokenizer=settings.MODEL,
#    device=settings.DEVICE,
#    use_auth_token=settings.AUTH_TOKEN,
#    cache_dir=settings.MODEL_CACHE_DIR,
# )
