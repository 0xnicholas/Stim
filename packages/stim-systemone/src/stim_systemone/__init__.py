"""System One wire protocol, rendering, and derived quantities for Stim."""

from importlib.metadata import PackageNotFoundError, version

from stim_systemone.engine import Decision, DecisionEngine, MemberMetadata, Option, Question
from stim_systemone.protocol import Limiter, OverloadCheck, SystemOneApp
from stim_systemone.rendering import render_content, render_option
from stim_systemone.stub import StubDecisionEngine
from stim_systemone.wire import (
    ChoiceQuestion,
    NoulCriteria,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
)

try:
    __version__ = version("stim-systemone")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "ChoiceQuestion",
    "Decision",
    "DecisionEngine",
    "Limiter",
    "MemberMetadata",
    "NoulCriteria",
    "NoulQuestion",
    "Option",
    "OverloadCheck",
    "Question",
    "ScoreQuestion",
    "StubDecisionEngine",
    "SystemOneApp",
    "SystemOneRequest",
    "__version__",
    "render_content",
    "render_option",
]
