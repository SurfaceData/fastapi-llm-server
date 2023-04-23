from fastapi import APIRouter
from loguru import logger
from prometheus_client import Histogram
from pydantic import BaseModel
from typing import List
from llm_server.config import settings
from llm_server.generator import generator

router = APIRouter()

GENERATE_TIME = Histogram("generate_time", "Time spent generating prompt completions")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    n: int = 1


class GenerateResult(BaseModel):
    completion: str


@router.post("/generate")
def generate(request: GenerateRequest) -> List[GenerateResult]:
    logger.info(f"generate: {request.prompt}")
    with GENERATE_TIME.time():
        results = generator(request.prompt, max_length=request.max_new_tokens)
    return [GenerateResult(completion=result["generated_text"]) for result in results]
