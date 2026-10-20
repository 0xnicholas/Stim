"""End-to-end checks for the choice slice over a real HTTP socket.

Every request goes through the same HTTP seam a TypeSafe SDK client uses. The stub
engine supplies controlled distributions behind ``DecisionEngine``; the recording
engine observes the protocol layer's normalized inputs without bypassing it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from typing import Any

import pytest
from typesafe_sdk import AsyncTypeSafeClient, Choice, TypeSafeClient

from stim_serve.app import create_app
from stim_systemone import Decision, Question, StubDecisionEngine
from tests.http_support import STUB_METADATA, ServerAddress, post, running_server

REQUEST_ID_HEADER = "x-typesafe-request-id"


@pytest.fixture(scope="module")
def server() -> Iterator[ServerAddress]:
    with running_server(create_app(StubDecisionEngine())) as address:
        yield address


class RecordingEngine:
    """A ``DecisionEngine`` that records the protocol layer's normalized inputs."""

    member_name = "stim-stub"
    catalog_metadata = STUB_METADATA

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Question, ...]]] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def decide(self, state_text: str, questions: Sequence[Question]) -> Decision:
        self.calls.append((state_text, tuple(questions)))
        return Decision(
            probabilities=tuple(tuple(1.0 for _ in question.options) for question in questions),
            input_tokens=1,
        )


def choice_payload(
    criteria: dict[str, Any],
    *,
    name: str = "tone",
    instructions: Any = "What is the tone?",
) -> dict[str, Any]:
    return {
        "state": "The customer seems furious.",
        "questions": {
            name: {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }
        },
    }


def expected_confidence(probabilities: dict[str, float]) -> float:
    count = len(probabilities)
    if count == 1:
        return 1.0
    p_max = max(probabilities.values())
    return round(max(0.0, (p_max - 1 / count) / (1 - 1 / count)), 4)


def test_sync_sdk_round_trips_a_choice_question() -> None:
    engine = StubDecisionEngine(raw_scores={"tone": [0.25, 0.75]})

    with (
        running_server(create_app(engine)) as server,
        TypeSafeClient(api_key="test-key", base_url=server.base_url) as client,
    ):
        response = client.system_one(
            state="The customer seems furious.",
            questions={
                "tone": Choice(
                    instructions="What is the tone?",
                    criteria={"calm": "A neutral message", "angry": "A hostile message"},
                )
            },
        )

    answer = response.choices["tone"]
    assert response.model == "stim-stub"
    assert answer.type == "choice"
    assert answer.choice == "angry"
    assert answer.probabilities == {"calm": 0.25, "angry": 0.75}
    assert answer.confidence == 0.5
    assert response.request_id


@pytest.mark.parametrize("option_count", [1, 255])
def test_sync_sdk_round_trips_choice_option_extremes(option_count: int) -> None:
    criteria = {f"option-{index:03d}": None for index in range(option_count)}

    with (
        running_server(create_app(StubDecisionEngine())) as server,
        TypeSafeClient(api_key="test-key", base_url=server.base_url) as client,
    ):
        response = client.system_one(
            state="Pick one.",
            questions={"pick": Choice(criteria=criteria)},
        )

    answer = response.choices["pick"]
    assert list(answer.probabilities) == list(criteria)
    assert answer.choice == next(iter(criteria))
    assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=0.02)
    assert answer.confidence == expected_confidence(answer.probabilities)


@pytest.mark.parametrize("option_count", [1, 255])
def test_async_sdk_round_trips_choice_option_extremes(option_count: int) -> None:
    criteria = {f"option-{index:03d}": None for index in range(option_count)}

    async def call(base_url: str) -> Any:
        async with AsyncTypeSafeClient(api_key="test-key", base_url=base_url) as client:
            return await client.system_one(
                state="Pick one.",
                questions={"pick": Choice(criteria=criteria)},
            )

    with running_server(create_app(StubDecisionEngine())) as server:
        response = asyncio.run(call(server.base_url))

    answer = response.choices["pick"]
    assert list(answer.probabilities) == list(criteria)
    assert answer.choice == next(iter(criteria))
    assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=0.02)
    assert answer.confidence == expected_confidence(answer.probabilities)


def test_choice_answer_contains_exactly_the_contract_fields() -> None:
    engine = StubDecisionEngine(raw_scores={"tone": [1.0, 3.0, 6.0]})
    criteria = {"calm": None, "tense": None, "angry": None}

    with running_server(create_app(engine)) as server:
        result = post(server, choice_payload(criteria))

    assert result.status == 200
    answer = result.body["answers"]["tone"]
    assert set(answer) == {"type", "choice", "probabilities", "confidence"}
    assert answer["type"] == "choice"
    assert answer["choice"] == "angry"
    assert answer["probabilities"] == {"calm": 0.1, "tense": 0.3, "angry": 0.6}
    assert answer["confidence"] == 0.4
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=0.02)
    assert result.headers[REQUEST_ID_HEADER]


@pytest.mark.parametrize(
    ("names", "scores"),
    [
        (["only"], [5.0]),
        (["a", "b"], [2.0, 8.0]),
        (["a", "b", "c"], [1.0, 1.0, 1.0]),
        (["a", "b", "c"], [9.0, 1.0, 0.0]),
    ],
)
def test_choice_and_confidence_are_derived_from_the_reported_probabilities(
    names: list[str], scores: list[float]
) -> None:
    engine = StubDecisionEngine(raw_scores={"q": scores})
    payload = {
        "state": "x",
        "questions": {"q": {"type": "choice", "criteria": dict.fromkeys(names)}},
    }

    with running_server(create_app(engine)) as server:
        result = post(server, payload)

    assert result.status == 200
    answer = result.body["answers"]["q"]
    probabilities = answer["probabilities"]
    assert list(probabilities) == names
    assert all(probability == round(probability, 4) for probability in probabilities.values())
    best = max(probabilities.values())
    assert answer["choice"] == next(
        name for name, probability in probabilities.items() if probability == best
    )
    assert answer["confidence"] == expected_confidence(probabilities)


def test_choice_confidence_measures_the_unrounded_normalized_distribution() -> None:
    engine = StubDecisionEngine(raw_scores={"q": [65000.0, 920.0]})
    payload = {
        "state": "x",
        "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": None}}},
    }

    with running_server(create_app(engine)) as server:
        result = post(server, payload)

    assert result.status == 200
    answer = result.body["answers"]["q"]
    assert answer["probabilities"] == {"a": 0.986, "b": 0.014}
    assert answer["confidence"] == 0.9721


def test_zero_sum_scores_fall_back_to_a_uniform_distribution() -> None:
    engine = StubDecisionEngine(raw_scores={"q": [0.0, 0.0, 0.0]})
    payload = {
        "state": "x",
        "questions": {"q": {"type": "choice", "criteria": dict.fromkeys(["a", "b", "c"])}},
    }

    with running_server(create_app(engine)) as server:
        result = post(server, payload)

    assert result.status == 200
    assert result.body["answers"]["q"] == {
        "type": "choice",
        "choice": "a",
        "probabilities": {"a": 0.3333, "b": 0.3333, "c": 0.3333},
        "confidence": 0.0,
    }


def test_choice_options_render_through_the_shared_renderer_in_request_order() -> None:
    engine = RecordingEngine()
    payload = choice_payload(
        {
            "calm": None,
            "angry": "hostile and upset",
            "policy": {"signals": ["late", "damaged"]},
            "blank": "",
            "empty_object": {},
            "empty_list": [],
        },
        instructions={"task": "Classify the tone"},
    )

    with running_server(create_app(engine)) as server:
        result = post(server, payload)

    assert result.status == 200
    _, questions = engine.calls[0]
    assert questions[0].type == "choice"
    assert questions[0].instructions == "task: Classify the tone"
    assert [(option.key, option.text) for option in questions[0].options] == [
        ("calm", "calm"),
        ("angry", "angry: hostile and upset"),
        ("policy", "policy:\n  signals:\n    - late\n    - damaged"),
        ("blank", "blank"),
        ("empty_object", "empty_object"),
        ("empty_list", "empty_list"),
    ]


def test_choice_names_stay_raw_on_the_wire_and_are_delimiter_safe_in_prompts() -> None:
    engine = RecordingEngine()
    payload = choice_payload({"<|box_start|>": "<|box_end|>"})

    with running_server(create_app(engine)) as server:
        result = post(server, payload)

    assert result.status == 200
    _, questions = engine.calls[0]
    assert [(option.key, option.text) for option in questions[0].options] == [
        ("<|box_start|>", "<¦box_start¦>: <¦box_end¦>")
    ]
    assert list(result.body["answers"]["tone"]["probabilities"]) == ["<|box_start|>"]


def test_choice_questions_and_noul_questions_share_a_request() -> None:
    engine = StubDecisionEngine(raw_scores={"tone": [0.25, 0.75], "billing": [0.9, 0.1]})
    payload = {
        "state": "I was charged twice and the agent shouted.",
        "questions": {
            "tone": {"type": "choice", "criteria": {"calm": None, "angry": None}},
            "billing": {"type": "noul", "instructions": "Is this about billing?"},
        },
    }

    with running_server(create_app(engine)) as server:
        result = post(server, payload)

    assert result.status == 200
    assert result.body["answers"]["tone"]["choice"] == "angry"
    assert result.body["answers"]["billing"] == {"type": "noul", "noul": 0.1}


def test_unknown_fields_in_a_choice_question_are_ignored(server: ServerAddress) -> None:
    payload = choice_payload({"calm": None, "angry": None})
    payload["questions"]["tone"]["unknown"] = {"nested": True}
    payload["extra"] = True

    result = post(server, payload)

    assert result.status == 200
    assert set(result.body["answers"]["tone"]) == {
        "type",
        "choice",
        "probabilities",
        "confidence",
    }


@pytest.mark.parametrize(
    "criteria",
    [
        pytest.param(None, id="null"),
        pytest.param([], id="array"),
        pytest.param("calm", id="string"),
        pytest.param(7, id="number"),
        pytest.param(True, id="boolean"),
        pytest.param({}, id="empty"),
        pytest.param({f"option-{index}": None for index in range(256)}, id="too-many"),
        pytest.param({"": None}, id="empty-name"),
        pytest.param({"calm": None, "": None}, id="one-empty-name"),
    ],
)
def test_invalid_choice_criteria_return_422_naming_the_field(
    server: ServerAddress, criteria: object
) -> None:
    payload = {
        "state": "x",
        "questions": {"q": {"type": "choice", "criteria": criteria}},
    }

    result = post(server, payload)

    assert result.status == 422
    assert result.body["code"] == "invalid_request"
    assert "questions.q.criteria" in result.body["detail"]
    assert set(result.body) == {"detail", "code"}
    assert "latency_ms" not in result.body


def test_missing_choice_criteria_returns_422_naming_the_field(server: ServerAddress) -> None:
    result = post(server, {"state": "x", "questions": {"q": {"type": "choice"}}})

    assert result.status == 422
    assert result.body["code"] == "invalid_request"
    assert "questions.q.criteria" in result.body["detail"]


def test_choice_accepts_exactly_255_options(server: ServerAddress) -> None:
    criteria = {f"option-{index:03d}": None for index in range(255)}

    result = post(server, choice_payload(criteria))

    assert result.status == 200
    answer = result.body["answers"]["tone"]
    assert len(answer["probabilities"]) == 255
    assert list(answer["probabilities"]) == list(criteria)
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=0.02)
