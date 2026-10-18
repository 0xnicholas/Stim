"""Pydantic wire models for the supported System One surface."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

# Unknown fields are ignored recursively, as the compatibility surface requires. Nested
# content is carried as JSON so caller-defined keys and order survive validation.
STRICT_WIRE_MODEL = ConfigDict(extra="ignore", strict=True)
DEFAULT_MODEL = "stim-latest"


class NoulCriteria(BaseModel):
    """Optional descriptions of the yes and no outcomes."""

    model_config = STRICT_WIRE_MODEL

    true: JsonValue = None
    false: JsonValue = None


class NoulQuestion(BaseModel):
    """A yes/no question."""

    model_config = STRICT_WIRE_MODEL

    type: Literal["noul"]
    instructions: JsonValue = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    """A single selection from a non-empty, ordered map of 1-255 options."""

    model_config = STRICT_WIRE_MODEL

    type: Literal["choice"]
    instructions: JsonValue = None
    criteria: dict[str, JsonValue] = Field(min_length=1, max_length=255)


QuestionModel = NoulQuestion | ChoiceQuestion


class SystemOneRequest(BaseModel):
    """The supported subset of a System One request body."""

    model_config = STRICT_WIRE_MODEL

    state: JsonValue
    questions: dict[str, JsonValue] = Field(min_length=1)
    model: str = DEFAULT_MODEL
