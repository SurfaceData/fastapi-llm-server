from fastapi import APIRouter
from loguru import logger
from prometheus_client import Histogram
from pydantic import BaseModel
from typing import List
from llm_server.config import settings
from llm_server.generator import model, tokenizer  # generator

router = APIRouter()

GENERATE_TIME = Histogram("generate_time", "Time spent generating prompt completions")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    n: int = 1


class BatchGenerateRequest(BaseModel):
    prompt: List[str]
    max_new_tokens: int = 128


class GenerateResult(BaseModel):
    completion: str


@router.post("/generate")
def generate(request: GenerateRequest) -> List[GenerateResult]:
    logger.info(f"generate: {request.prompt}")
    tokenized_input = tokenizer(request.prompt, return_tensors="pt").to(settings.DEVICE)
    answer = "The answer is:"
    tokenized_input["decoder_input_ids"] = (
        tokenizer(answer, return_tensors="pt").to(settings.DEVICE).input_ids
    )
    print(tokenized_input)
    with GENERATE_TIME.time():
        outputs = model(**tokenized_input)
        results = tokenizer.batch_decode(outputs)
    return [GenerateResult(completion=result) for result in results]


@router.post("/generate-batch")
def generate_batch(request: BatchGenerateRequest) -> List[GenerateResult]:
    with GENERATE_TIME.time():
        results = generator(request.prompt, max_length=request.max_new_tokens)
    return [GenerateResult(completion=result["generated_text"]) for result in results]
