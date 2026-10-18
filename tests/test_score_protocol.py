"""End-to-end checks for the score slice over a real HTTP socket.

Every request goes through the same HTTP seam a TypeSafe SDK client uses. The stub
engine supplies controlled distributions behind ``DecisionEngine``; the recording
engine observes the protocol layer's normalized inputs without bypassing it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pytest
from typesafe_sdk import AsyncTypeSafeClient, Score, TypeSafeClient

from stim_serve.app import create_app
from stim_systemone import Decision, Question, StubDecisionEngine
from tests.http_support import ServerAddress, post, running_server


@pytest.fixture(scope="module")
def server() -> Iterator[ServerAddress]:
    with running_server(create_app(StubDecisionEngine())) as address:
        yield address


class RecordingEngine:
    """A ``DecisionEngine`` that records the protocol layer's normalized inputs."""

    member_name = "stim-stub"

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


def score_payload(
    criteria: Any,
    *,
    name: str = "priority",
    instructions: Any = "How urgent is this?",
) -> dict[str, Any]:
    return {
        "state": "The customer needs a reply today.",
        "questions": {
            name: {
                "type": "score",
                "instructions": instructions,
                "criteria": criteria,
            }
        },
    }


def expected_score(probabilities: Mapping[str, float]) -> float:
    weighted = sum(int(level) * probability for level, probability in probabilities.items())
    return round(weighted, 4)


def expected_confidence(probabilities: Mapping[str, float]) -> float:
    levels = list(probabilities.values())
    count = len(levels)
    mode = max(range(count), key=levels.__getitem__)
    spread = sum(probability * abs(level - mode) for level, probability in enumerate(levels))
    uniform_spread = sum(abs(level - (count - 1) / 2) for level in range(count)) / count
    return round(max(0.0, 1 - spread / uniform_spread), 4)


def test_sync_sdk_round_trips_a_score_question() -> None:
    engine = StubDecisionEngine(raw_scores={"priority": [1.0, 3.0]})

    with (
        running_server(create_app(engine)) as server,
        TypeSafeClient(api_key="test-key", base_url=server.base_url) as client,
    ):
        response = client.system_one(
            state="The customer needs a reply today.",
            questions={
                "priority": Score(
                    instructions="How urgent is this?",
                    criteria=["low", "high"],
                )
            },
        )

    answer = response.scores["priority"]
    assert answer.type == "score"
    assert answer.score == 0.75
    assert answer.legend == {0: "low", 1: "high"}
    assert answer.probabilities == {0: 0.25, 1: 0.75}
    assert answer.confidence == 0.5


def test_sync_sdk_round_trips_ten_score_levels() -> None:
    engine = StubDecisionEngine(raw_scores={"priority": [1.0] * 10})

    with (
        running_server(create_app(engine)) as server,
        TypeSafeClient(api_key="test-key", base_url=server.base_url) as client,
    ):
        response = client.system_one(
            state="The customer needs a reply today.",
            questions={
                "priority": Score(
                    instructions="How urgent is this?",
                    criteria=[f"level-{index}" for index in range(10)],
                )
            },
        )

    answer = response.scores["priority"]
    assert answer.type == "score"
    assert answer.score == 4.5
    assert answer.legend == {index: f"level-{index}" for index in range(10)}
    assert answer.probabilities == {index: 0.1 for index in range(10)}
    assert answer.confidence == 0.0


@pytest.mark.parametrize("level_count", [2, 10])
def test_async_sdk_round_trips_score_level_extremes(level_count: int) -> None:
    engine = StubDecisionEngine(raw_scores={"priority": [1.0] * level_count})
    criteria = [f"level-{index}" for index in range(level_count)]

    async def call(base_url: str) -> Any:
        async with AsyncTypeSafeClient(api_key="test-key", base_url=base_url) as client:
            return await client.system_one(
                state="The customer needs a reply today.",
                questions={"priority": Score(criteria=criteria)},
            )

    with running_server(create_app(engine)) as server:
        response = asyncio.run(call(server.base_url))

    answer = response.scores["priority"]
    assert answer.type == "score"
    assert answer.legend == dict(enumerate(criteria))
    assert answer.probabilities == {
        index: round(1 / level_count, 4) for index in range(level_count)
    }
    assert answer.score == pytest.approx((level_count - 1) / 2, abs=1e-4)
    assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=0.02)


def test_score_answer_contains_exactly_the_contract_fields() -> None:
    engine = StubDecisionEngine(raw_scores={"priority": [2.0, 9.0, 5.0, 4.0]})
    criteria = ["low", "medium", "high", "critical"]

    with running_server(create_app(engine)) as address:
        result = post(address, score_payload(criteria))

    assert result.status == 200
    answer = result.body["answers"]["priority"]
    assert set(answer) == {"type", "score", "legend", "probabilities", "confidence"}
    assert answer["type"] == "score"
    assert answer["score"] == 1.55
    assert answer["legend"] == {"0": "low", "1": "medium", "2": "high", "3": "critical"}
    assert answer["probabilities"] == {"0": 0.1, "1": 0.45, "2": 0.25, "3": 0.2}
    assert answer["confidence"] == 0.25
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=0.02)
    assert result.headers["x-typesafe-request-id"]


@pytest.mark.parametrize(
    ("criteria", "scores"),
    [
        (["low", "high"], [1.0, 3.0]),
        (["low", "medium", "high"], [2.0, 9.0, 5.0]),
        (["a", "b", "c", "d"], [2.0, 9.0, 5.0, 4.0]),
        (
            [f"level-{index}" for index in range(10)],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ),
        (["first", "second", "third"], [5.0, 5.0, 1.0]),
        (["a", "b", "c", "d"], [0.051758, 0.553999, 0.554085, 0.469260]),
    ],
)
def test_score_legend_probabilities_score_and_confidence_are_self_consistent(
    criteria: list[str], scores: list[float]
) -> None:
    engine = StubDecisionEngine(raw_scores={"q": scores})
    payload = score_payload(criteria, name="q")

    with running_server(create_app(engine)) as address:
        result = post(address, payload)

    assert result.status == 200
    answer = result.body["answers"]["q"]
    probabilities = answer["probabilities"]
    assert list(probabilities) == [str(index) for index in range(len(criteria))]
    assert all(probability == round(probability, 4) for probability in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=0.02)
    assert answer["legend"] == {str(index): level for index, level in enumerate(criteria)}
    assert answer["score"] == expected_score(probabilities)
    assert answer["confidence"] == expected_confidence(probabilities)
    assert answer["score"] == pytest.approx(
        sum(
            index * probability
            for index, probability in enumerate(score / sum(scores) for score in scores)
        ),
        abs=5e-3,
    )
    assert 0.0 <= answer["confidence"] <= 1.0


def test_score_confidence_uses_the_first_maximum_reported_level() -> None:
    engine = StubDecisionEngine(raw_scores={"q": [0.051758, 0.553999, 0.554085, 0.469260]})

    with running_server(create_app(engine)) as address:
        result = post(address, score_payload(["a", "b", "c", "d"], name="q"))

    assert result.status == 200
    answer = result.body["answers"]["q"]
    assert answer["probabilities"] == {"0": 0.0318, "1": 0.3401, "2": 0.3401, "3": 0.288}
    assert answer["score"] == 1.8843
    assert answer["confidence"] == 0.0521


@pytest.mark.parametrize(
    "criteria",
    [
        pytest.param(None, id="null"),
        pytest.param([], id="zero-levels"),
        pytest.param(["only"], id="one-level"),
        pytest.param([f"level-{index}" for index in range(11)], id="eleven-levels"),
        pytest.param("low", id="string"),
        pytest.param(7, id="number"),
        pytest.param(True, id="boolean"),
        pytest.param({"0": "low", "1": "high"}, id="object"),
    ],
)
def test_invalid_score_criteria_return_422_naming_the_field(
    server: ServerAddress, criteria: object
) -> None:
    payload = {
        "state": "x",
        "questions": {"q": {"type": "score", "criteria": criteria}},
    }

    result = post(server, payload)

    assert result.status == 422
    assert result.body["code"] == "invalid_request"
    assert "questions.q.criteria" in result.body["detail"]
    assert set(result.body) == {"detail", "code"}
    assert "latency_ms" not in result.body
    assert result.headers["x-typesafe-request-id"]


def test_missing_score_criteria_returns_422_naming_the_field(server: ServerAddress) -> None:
    result = post(server, {"state": "x", "questions": {"q": {"type": "score"}}})

    assert result.status == 422
    assert result.body["code"] == "invalid_request"
    assert "questions.q.criteria" in result.body["detail"]


@pytest.mark.parametrize("level_count", [2, 10])
def test_score_accepts_the_level_count_extremes(server: ServerAddress, level_count: int) -> None:
    criteria = [f"level-{index}" for index in range(level_count)]

    result = post(server, score_payload(criteria))

    assert result.status == 200
    answer = result.body["answers"]["priority"]
    assert list(answer["legend"]) == [str(index) for index in range(level_count)]
    assert list(answer["probabilities"]) == [str(index) for index in range(level_count)]
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=0.02)


def test_score_lands_between_levels() -> None:
    engine = StubDecisionEngine(raw_scores={"priority": [1.0, 2.0]})

    with running_server(create_app(engine)) as server:
        result = post(server, score_payload(["low", "high"]))

    assert result.status == 200
    score = result.body["answers"]["priority"]["score"]
    assert score == pytest.approx(2 / 3, abs=1e-4)
    assert 0 < score < 1
    assert score == round(score, 4)


def test_score_levels_render_through_the_shared_renderer_in_criteria_order() -> None:
    engine = RecordingEngine()
    criteria: list[Any] = [
        "low",
        {"signals": ["wait", "no reply"]},
        "<|box_start|>",
        None,
        "",
        [],
    ]

    with running_server(create_app(engine)) as server:
        result = post(
            server,
            score_payload(criteria, instructions={"task": "Rate the urgency"}),
        )

    assert result.status == 200
    _, questions = engine.calls[0]
    assert questions[0].type == "score"
    assert questions[0].instructions == "task: Rate the urgency"
    assert [(option.key, option.text) for option in questions[0].options] == [
        ("0", "0: low"),
        ("1", "1:\n  signals:\n    - wait\n    - no reply"),
        ("2", "2: <¦box_start¦>"),
        ("3", "3"),
        ("4", "4"),
        ("5", "5"),
    ]
    assert result.body["answers"]["priority"]["legend"] == {
        "0": "low",
        "1": "signals:\n  - wait\n  - no reply",
        "2": "<¦box_start¦>",
        "3": "",
        "4": "",
        "5": "",
    }


def test_score_questions_share_a_request_with_noul_and_choice() -> None:
    engine = StubDecisionEngine(
        raw_scores={"tone": [0.25, 0.75], "billing": [0.9, 0.1], "priority": [1.0] * 3}
    )
    payload = {
        "state": "I was charged twice and the agent shouted.",
        "questions": {
            "tone": {"type": "choice", "criteria": {"calm": None, "angry": None}},
            "billing": {"type": "noul", "instructions": "Is this about billing?"},
            "priority": {"type": "score", "criteria": ["low", "medium", "high"]},
        },
    }

    with running_server(create_app(engine)) as server:
        result = post(server, payload)

    assert result.status == 200
    assert result.body["answers"]["tone"]["choice"] == "angry"
    assert result.body["answers"]["billing"] == {"type": "noul", "noul": 0.1}
    score_answer = result.body["answers"]["priority"]
    assert score_answer["score"] == 0.9999
    assert score_answer["probabilities"] == {"0": 0.3333, "1": 0.3333, "2": 0.3333}
    assert score_answer["confidence"] == 0.0
