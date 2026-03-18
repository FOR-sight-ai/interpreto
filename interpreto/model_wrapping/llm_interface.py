# MIT License
#
# Copyright (c) 2025 IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL and FOR are research programs operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod


class LLMInterface(ABC):
    @abstractmethod  # TODO: update example in tutorials
    def batch_generate(self, system_prompt: str, user_prompts: list[str]) -> list[str | None]:
        pass


class OpenAILLM(LLMInterface):
    def __init__(self, api_key: str, model: str = "gpt-4.1-nano", num_try: int = 5):
        try:
            import openai  # noqa: PLC0415  # ruff: disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError("Install openai to use OpenAI API.") from e

        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.num_try = num_try

    async def in_batch_generate(self, system_prompt, user_prompt, semaphore):
        async with semaphore:
            response = await self.client.responses.create(
                model=self.model,
                prompt_cache_key="shared-system-prompt-v1",
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            return response.output_text

    async def async_batch_generate(self, system_prompt: str, user_prompts: list[str]) -> list[str | None]:
        semaphore = asyncio.Semaphore(10)

        tasks = [self.in_batch_generate(system_prompt, p, semaphore) for p in user_prompts]

        return await asyncio.gather(*tasks, return_exceptions=True)  # type: ignore

    def batch_generate(self, system_prompt: str, user_prompts: list[str]) -> list[str | None]:
        return asyncio.run(self.async_batch_generate(system_prompt, user_prompts))
