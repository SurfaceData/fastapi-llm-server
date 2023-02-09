from fastapi import APIRouter
from prometheus_client import Histogram
from pydantic import BaseModel
from typing import List

from llm_server.config import settings
from llm_server.generator import model, tokenizer

router = APIRouter()

GENERATE_TIME = Histogram("generate_time", "Time spent generating prompt completions")


class GenerateResult(BaseModel):
    completion: str


@router.get("/generate")
def generate(prompt: str) -> List[GenerateResult]:
    tokenized_input = tokenizer(prompt, return_tensors="pt").to(settings.DEVICE)
    with GENERATE_TIME.time():
        outputs = model.generate(**tokenized_input)
    results = tokenizer.batch_decode(outputs)
    return [GenerateResult(completion=result) for result in results]
