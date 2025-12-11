from __future__ import annotations

from typing import Callable, Protocol


class LLMGenerate(Protocol):
    def __call__(self, prompt: str) -> str: ...


Detector = Callable[..., None]


__all__ = ["LLMGenerate", "Detector"]
