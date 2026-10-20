"""Forward-compatibility and question-isolation checks over the real HTTP seam.

Every case goes through ``POST /v1/systemone`` as an SDK client would. The
deterministic stub engine fixes each question's distribution, so any drift introduced
by parsing, logging, batching, or repetition is exact rather than statistical. The
same invariants are meant to be rerun against a real artifact by the later
real-engine ticket, with the published 0.011 tolerance.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from typesafe_sdk import Choice, Noul, TypeSafeClient

from stim_serve.app import create_app
from stim_systemone import StubDecisionEngine
from tests.http_support import ServerAddress, post, running_server

DANGER_ZONE_LONG_CONTEXT = "TypeSafe-Danger-Zone-Long-Context"
LOGGER_NAME = "stim_systemone"
UNKNOWN_FIELD_MESSAGE = "ignored unknown request field: "
REPO_ROOT = Path(__file__).resolve().parents[1]

STATE = "I was charged twice and the agent shouted."
NOUL = {"type": "noul", "instructions": "Is this about billing?"}
CHOICE = {
    "type": "choice",
    "instructions": "What is the tone?",
    "criteria": {"calm": None, "tense": None, "angry": None},
}
SCORE = {
    "type": "score",
    "instructions": "How urgent is this?",
    "criteria": ["low", "medium", "high"],
}
QUESTIONS = {"billing": NOUL, "tone": CHOICE, "priority": SCORE}
RAW_SCORES = {"billing": [0.9, 0.1], "tone": [1.0, 4.0, 5.0], "priority": [2.0, 3.0, 5.0]}
PAYLOAD = {"state": STATE, "questions": QUESTIONS}


@pytest.fixture(scope="module")
def server() -> Iterator[ServerAddress]:
    engine = StubDecisionEngine(raw_scores=RAW_SCORES)
    with running_server(create_app(engine)) as address:
        yield address


def logged_unknown_paths(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage().removeprefix(UNKNOWN_FIELD_MESSAGE)
        for record in caplog.records
        if record.name == LOGGER_NAME and record.getMessage().startswith(UNKNOWN_FIELD_MESSAGE)
    ]


def test_unknown_fields_at_every_level_are_ignored_and_logged(
    server: ServerAddress, caplog: pytest.LogCaptureFixture
) -> None:
    clean: dict[str, Any] = {
        "state": STATE,
        "model": "stim-stub",
        "questions": {
            "billing": {
                "type": "noul",
                "instructions": "Is this about billing?",
                "criteria": {
                    "true": "the customer was charged",
                    "false": "the customer was not charged",
                },
            },
            "tone": CHOICE,
            "priority": SCORE,
        },
    }
    dirty = {
        **clean,
        "temperature": 0.0,
        "future_body_field": {"nested": "SECRET_BODY_VALUE"},
        "questions": {
            **clean["questions"],
            "billing": {
                **clean["questions"]["billing"],
                "future_question_field": "SECRET_QUESTION_VALUE",
                "criteria": {
                    **clean["questions"]["billing"]["criteria"],
                    "future_criterion": "SECRET_CRITERIA_VALUE",
                },
            },
        },
    }

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        baseline = post(server, clean)
        result = post(server, dirty)

    assert baseline.status == 200
    assert result.status == 200
    assert result.body["answers"] == baseline.body["answers"]
    assert result.body["usage"] == baseline.body["usage"]
    assert sorted(logged_unknown_paths(caplog)) == sorted(
        [
            "temperature",
            "future_body_field",
            "questions.billing.future_question_field",
            "questions.billing.criteria.future_criterion",
        ]
    )
    assert all(
        record.levelno == logging.WARNING for record in caplog.records if record.name == LOGGER_NAME
    )
    assert "SECRET" not in caplog.text


def test_unknown_fields_inside_caller_content_are_not_request_fields(
    server: ServerAddress, caplog: pytest.LogCaptureFixture
) -> None:
    payload = {
        "state": {"future": "SECRET_STATE_VALUE"},
        "questions": {
            "q": {
                "type": "noul",
                "instructions": {"future": "SECRET_INSTRUCTION_VALUE"},
                "criteria": {"true": {"future": "SECRET_CRITERIA_VALUE"}, "false": None},
            }
        },
    }

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = post(server, payload)

    assert result.status == 200
    assert logged_unknown_paths(caplog) == []
    assert "SECRET" not in caplog.text


def test_unknown_field_paths_cannot_forge_log_lines(
    server: ServerAddress, caplog: pytest.LogCaptureFixture
) -> None:
    payload = {
        "state": "x",
        "future\nFORGED BODY LINE": 1,
        "questions": {
            "q\nFORGED QUESTION LINE": {
                "type": "noul",
                "future\nFORGED QUESTION FIELD": 2,
            }
        },
    }

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = post(server, payload)

    assert result.status == 200
    ignored = [record for record in caplog.records if record.name == LOGGER_NAME]
    assert ignored
    assert all("\n" not in record.getMessage() for record in ignored)
    assert all("\\u000a" in record.getMessage() for record in ignored)


def test_sdk_extra_body_temperature_is_accepted_and_ignored(server: ServerAddress) -> None:
    questions: dict[str, Noul | Choice] = {
        "billing": Noul(instructions="Is this about billing?"),
        "tone": Choice(criteria={"calm": None, "tense": None, "angry": None}),
    }

    with TypeSafeClient(api_key="test-key", base_url=server.base_url) as client:
        baseline = client.system_one(state=STATE, questions=questions)
        frozen = client.system_one(
            state=STATE, questions=questions, extra_body={"temperature": 0.0}
        )
        hot = client.system_one(state=STATE, questions=questions, extra_body={"temperature": 2.0})

    assert frozen.answers == baseline.answers
    assert hot.answers == baseline.answers
    assert frozen.usage == baseline.usage
    assert hot.usage == baseline.usage


def test_danger_zone_long_context_header_is_accepted_and_ignored(server: ServerAddress) -> None:
    baseline = post(server, PAYLOAD)
    zoned = post(server, PAYLOAD, headers={DANGER_ZONE_LONG_CONTEXT: "true"})

    assert baseline.status == 200
    assert zoned.status == 200
    assert zoned.body["answers"] == baseline.body["answers"]
    assert zoned.body["usage"] == baseline.body["usage"]


def test_danger_zone_header_does_not_raise_the_state_limit() -> None:
    with running_server(create_app(StubDecisionEngine(), max_state=10)) as address:
        accepted = post(
            address,
            {"state": " ".join(["word"] * 10), "questions": {"q": NOUL}},
            headers={DANGER_ZONE_LONG_CONTEXT: "true"},
        )
        rejected = post(
            address,
            {"state": " ".join(["word"] * 11), "questions": {"q": NOUL}},
            headers={DANGER_ZONE_LONG_CONTEXT: "true"},
        )

    assert accepted.status == 200
    assert rejected.status == 400
    assert rejected.body["code"] == "max_tokens_exceeded"
    assert "11" in rejected.body["detail"]
    assert "10" in rejected.body["detail"]


def test_each_question_is_answered_identically_alone_or_with_siblings(
    server: ServerAddress,
) -> None:
    together = post(server, PAYLOAD)

    assert together.status == 200
    for name, question in QUESTIONS.items():
        alone = post(server, {"state": STATE, "questions": {name: question}})

        assert alone.status == 200
        assert alone.body["answers"][name] == together.body["answers"][name]


def test_question_batch_composition_does_not_change_answers(server: ServerAddress) -> None:
    forward = post(server, PAYLOAD)
    reversed_questions = dict(reversed(list(QUESTIONS.items())))
    reverse = post(server, {"state": STATE, "questions": reversed_questions})

    assert forward.status == 200
    assert reverse.status == 200
    assert reverse.body["answers"] == forward.body["answers"]


def test_repeated_requests_return_identical_answers(server: ServerAddress) -> None:
    first = post(server, PAYLOAD)
    second = post(server, PAYLOAD)

    assert first.status == 200
    assert second.status == 200
    assert second.body["answers"] == first.body["answers"]


def test_cross_request_batch_composition_does_not_change_answers(
    server: ServerAddress,
) -> None:
    solo_payloads = {
        name: {"state": STATE, "questions": {name: question}}
        for name, question in QUESTIONS.items()
    }
    solo_answers = {
        name: post(server, payload).body["answers"][name] for name, payload in solo_payloads.items()
    }
    batches: list[dict[str, Any]] = [
        *solo_payloads.values(),
        {"state": STATE, "questions": {"billing": NOUL, "priority": SCORE}},
        {"state": STATE, "questions": {"tone": CHOICE}},
        PAYLOAD,
        PAYLOAD,
    ]

    with ThreadPoolExecutor(max_workers=len(batches)) as pool:
        results = list(pool.map(lambda payload: post(server, payload), batches))

    assert all(result.status == 200 for result in results)
    for result in results:
        for name, answer in result.body["answers"].items():
            assert answer == solo_answers[name]


def test_success_response_contains_only_documented_fields(server: ServerAddress) -> None:
    result = post(server, {**PAYLOAD, "temperature": 0.0, "future": {"nested": True}})

    assert result.status == 200
    assert set(result.body) == {"model", "answers", "usage", "latency_ms"}
    assert set(result.body["usage"]) == {"input_tokens", "output_tokens"}
    assert set(result.body["answers"]) == set(QUESTIONS)
    assert set(result.body["answers"]["billing"]) == {"type", "noul"}
    assert set(result.body["answers"]["tone"]) == {
        "type",
        "choice",
        "probabilities",
        "confidence",
    }
    assert set(result.body["answers"]["priority"]) == {
        "type",
        "score",
        "legend",
        "probabilities",
        "confidence",
    }
    assert set(result.body["answers"]["tone"]["probabilities"]) == {"calm", "tense", "angry"}
    assert set(result.body["answers"]["priority"]["legend"]) == {"0", "1", "2"}
    assert set(result.body["answers"]["priority"]["probabilities"]) == {"0", "1", "2"}


def test_forward_compat_surfaces_are_registered_in_the_compatibility_matrix() -> None:
    matrix: dict[str, Any] = json.loads(
        (REPO_ROOT / "docs" / "compatibility.json").read_text(encoding="utf-8")
    )
    entries = {entry["id"]: entry for entry in matrix["entries"]}

    expected_test_ids = {
        "request.extra_body.temperature": (
            "ignored",
            "tests/test_forward_compat.py::test_sdk_extra_body_temperature_is_accepted_and_ignored",
        ),
        "request.unknown_fields": (
            "ignored",
            "tests/test_forward_compat.py::"
            "test_unknown_fields_at_every_level_are_ignored_and_logged",
        ),
        "header.typesafe-danger-zone-long-context": (
            "ignored",
            "tests/test_forward_compat.py::"
            "test_danger_zone_long_context_header_is_accepted_and_ignored",
        ),
        "limit.danger-zone-long-context": (
            "supported",
            "tests/test_forward_compat.py::test_danger_zone_header_does_not_raise_the_state_limit",
        ),
        "request.question_isolation": (
            "supported",
            "tests/test_forward_compat.py::"
            "test_each_question_is_answered_identically_alone_or_with_siblings",
        ),
        "request.repeated_requests": (
            "supported",
            "tests/test_forward_compat.py::test_repeated_requests_return_identical_answers",
        ),
        "request.cross_request_batching": (
            "supported",
            "tests/test_forward_compat.py::"
            "test_cross_request_batch_composition_does_not_change_answers",
        ),
    }
    for entry_id, (status, test_id) in expected_test_ids.items():
        entry = entries[entry_id]
        assert entry["status"] == status
        assert entry["expected_behavior"]
        assert entry["test_id"] == test_id
