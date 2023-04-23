from enum import Enum
from pydantic import BaseModel
from typing import List


class MessageSource(str, Enum):
    system = "system"
    user = "user"


class Message(BaseModel):
    message_source: MessageSource
    content: str


class PromptedConversation(BaseModel):
    system_prompt: str = ""
    messages: List[Message]
