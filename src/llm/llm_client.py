import logging
import os
from typing import Generator

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        model: str = "mistral-small-2603",
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ):
        from mistralai import Mistral

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        api_key = os.environ.get("MISTRAL_API_KEY", "")
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "too short"
        logger.info(f"LLM Client init: MISTRAL_API_KEY length={len(api_key)}, masked={masked_key}")
        self.client = Mistral(api_key=api_key)
        logger.info(f"LLM client ready — model: {model}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.complete(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content
        usage = response.usage
        logger.info(
            f"Generated {len(text)} chars | "
            f"input_tokens={usage.prompt_tokens} "
            f"output_tokens={usage.completion_tokens}"
        )
        return text

    def stream(self, system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
        stream_response = self.client.chat.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        for chunk in stream_response:
            content = chunk.data.choices[0].delta.content
            if content is not None:
                yield content
