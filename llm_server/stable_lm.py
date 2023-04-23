import torch

from jinja2 import Template
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from typing import Union

from llm_server.config import settings
from llm_server.structs import Message, MessageSource, PromptedConversation


DEFAULT_SYSTEM_PROMPT = """<|SYSTEM|># StableLM Tuned (Alpha version)
	- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
	- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
	- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
	- StableLM will refuse to participate in anything that could harm a human.
	"""


def create_stablelm_model():
    tokenizer = AutoTokenizer.from_pretrained(
        settings.MODEL, cache_dir=settings.MODEL_CACHE_DIR
    )
    model = AutoModelForCausalLM.from_pretrained(
        settings.MODEL, cache_dir=settings.MODEL_CACHE_DIR
    )
    model.half().to(settings.DEVICE)

    raw_template = """{% if conversation.system_prompt is not none %}{{ conversation.system_prompt }}{% else %}{{ default_system_prompt }}{% endif %}
{% for message in conversation.messages %}
{% if message.source == "assistant" %}<|ASSISTANT|>{{ message.content }}{% elif message.source == "user" %}<|USER|>{{ message.content }}{% endif %}{% endfor %}
<|ASSISTANT|>
"""
    template = Template(raw_template)

    class StopOnTokens(StoppingCriteria):
        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            stop_ids = [50278, 50279, 50277, 1, 0]
            for stop_id in stop_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    def get_prompt(ctx: Union[str, PromptedConversation]) -> str:
        if isinstance(ctx, str):
            return template.render(
                PromptedConversation(
                    system_prompt=DEFAULT_SYSTEM_PROMPT,
                    messages=[Message(source=MessageSource.user, content=ctx)],
                )
            )
        return template.render(
            {
                "conversation": ctx,
                "default_system_prompt": DEFAULT_SYSTEM_PROMPT,
            }
        )

    def generator(ctx: Union[str, PromptedConversation], max_length):
        prompt = get_prompt(ctx)
        inputs = tokenizer(prompt, return_tensors="pt").to(settings.DEVICE)
        tokens = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()])
        )
        input_ids = inputs["input_ids"]
        generated = tokenizer.batch_decode(
            tokens[:, input_ids.shape[1] :], skip_special_tokens=True
        )
        return [{"generated_text": result} for result in generated]

    return generator
