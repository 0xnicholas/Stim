"""The torch-free contract between the protocol layer and a decision model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Option:
    """One option the engine scores, with its wire key and prompt text."""

    key: str
    text: str


@dataclass(frozen=True)
class Question:
    """A question normalized by the protocol layer for the engine."""

    name: str
    type: str
    instructions: str
    options: tuple[Option, ...]


@dataclass(frozen=True)
class Decision:
    """Raw per-option scores and the input token count for one request."""

    probabilities: tuple[tuple[float, ...], ...]
    input_tokens: int


class DecisionEngine(Protocol):
    """The only interface the protocol layer uses to reach a model."""

    @property
    def member_name(self) -> str:
        """The loaded member's name, reported as the response's model."""
        ...

    def count_tokens(self, text: str) -> int:
        """Count tokens in text, for state limits and usage reporting."""
        ...

    def decide(self, state_text: str, questions: Sequence[Question]) -> Decision:
        """Score every option of every question against the rendered state."""
        ...
