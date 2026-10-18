"""Deterministic engine used to exercise the protocol layer without model weights."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from stim_systemone.engine import Decision, Question

_WORD = re.compile(r"\S+")


class StubDecisionEngine:
    """Return fixed per-question scores so protocol behavior stays reproducible.

    ``raw_scores`` maps a question name to unnormalized option scores in wire order.
    Questions absent from the mapping score noul questions as ``(1 - yes_probability,
    yes_probability)`` and every other question uniformly.
    """

    def __init__(
        self,
        *,
        member_name: str = "stim-stub",
        yes_probability: float = 0.75,
        raw_scores: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        self._member_name = member_name
        self._yes_probability = yes_probability
        self._raw_scores = {name: tuple(scores) for name, scores in (raw_scores or {}).items()}

    @property
    def member_name(self) -> str:
        """The name reported as the serving member."""
        return self._member_name

    def count_tokens(self, text: str) -> int:
        """Count whitespace-separated tokens, standing in for a real tokenizer."""
        return len(_WORD.findall(text))

    def decide(self, state_text: str, questions: Sequence[Question]) -> Decision:
        """Score each question with its configured raw scores."""
        rows = tuple(self._scores(question) for question in questions)
        input_tokens = self.count_tokens(state_text) + sum(
            self.count_tokens(question.instructions)
            + sum(self.count_tokens(option.text) for option in question.options)
            for question in questions
        )
        return Decision(probabilities=rows, input_tokens=input_tokens)

    def _scores(self, question: Question) -> tuple[float, ...]:
        configured = self._raw_scores.get(question.name)
        if configured is not None:
            return configured
        if question.type == "noul":
            return (1.0 - self._yes_probability, self._yes_probability)
        return tuple(1.0 for _ in question.options)
