from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel
from typing import List

from llm_server.config import settings
from llm_server.generator import generator
from llm_server.structs import PromptedConversation

router = APIRouter()


class ChatRequest(BaseModel):
    conversation: PromptedConversation
    max_new_tokens: int = 128


class ChatCompletion(BaseModel):
    content: str


class ChatResponse(BaseModel):
    results: List[ChatCompletion]


@router.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    results = generator(request.conversation, max_length=request.max_new_tokens)
    return ChatResponse(
        results=[ChatCompletion(content=r["generated_text"]) for r in results]
    )
