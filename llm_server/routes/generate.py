from fastapi import APIRouter
from prometheus_client import Histogram
from pydantic import BaseModel
from typing import List

from llm_server.config import settings
from llm_server.generator import model, tokenizer

router = APIRouter()

GENERATE_TIME = Histogram("generate_time", "Time spent generating prompt completions")


class GenerateRequest(BaseModel):
    prompt: str
    n: int = 1


class GenerateResult(BaseModel):
    completion: str


@router.post("/generate")
def generate(request: GenerateRequest) -> List[GenerateResult]:
    tokenized_input = tokenizer(request.prompt, return_tensors="pt").to(settings.DEVICE)
    with GENERATE_TIME.time():
        outputs = model.generate(
            **tokenized_input, num_beams=request.n, num_return_sequences=request.n
        )
    results = tokenizer.batch_decode(outputs)
    return [GenerateResult(completion=result) for result in results]
