from enum import Enum
from pydantic import BaseModel
from typing import List, Union


class MessageSource(str, Enum):
    assistant = "assistant"
    user = "user"


class Message(BaseModel):
    source: MessageSource
    content: str


class PromptedConversation(BaseModel):
    system_prompt: Union[str, None] = None
    messages: List[Message]
