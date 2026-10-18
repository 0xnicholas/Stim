"""End-to-end checks for the noul slice over a real HTTP socket.

Every request below goes through the same HTTP seam a TypeSafe SDK client uses. The
deterministic stub engine sits behind ``DecisionEngine``; the recording engine used by
the rendering test is injected at the same contract and does not bypass the protocol
layer.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import re
import socket
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import pytest
import uvicorn
from typesafe_sdk import AsyncTypeSafeClient, Noul, SystemOneResponse, TypeSafeClient

from stim_serve.app import create_app
from stim_systemone import Decision, Question, StubDecisionEngine

REQUEST_ID_HEADER = "x-typesafe-request-id"
NOUL_QUESTION = {"type": "noul", "instructions": "Is this about billing?"}
PAYLOAD = {"state": "I was charged twice. Please help.", "questions": {"billing": NOUL_QUESTION}}
SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._~-]{1,128}")
GENERATED_REQUEST_ID = re.compile(r"[0-9a-f]{32}")
ONE_DECIMAL_LATENCY = re.compile(r'"latency_ms":-?\d+\.\d(?:[,}])')


@dataclass(frozen=True)
class ServerAddress:
    base_url: str
    port: int


@dataclass(frozen=True)
class HttpResult:
    status: int
    body: dict[str, Any]
    headers: dict[str, str]
    text: str


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return cast(int, sock.getsockname()[1])


@contextmanager
def running_server(app: Any) -> Iterator[ServerAddress]:
    port = free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="off")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("server exited during startup")
        if time.monotonic() > deadline:
            raise RuntimeError("server did not start")
        time.sleep(0.01)
    try:
        yield ServerAddress(f"http://127.0.0.1:{port}", port)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError("server did not stop")


def request(
    server: ServerAddress,
    method: str,
    path: str,
    payload: object | bytes | None = None,
    *,
    headers: dict[str, str] | None = None,
) -> HttpResult:
    if payload is None:
        body = None
    elif isinstance(payload, bytes):
        body = payload
    else:
        body = json.dumps(payload).encode()
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    try:
        request_headers = {"Content-Type": "application/json", **(headers or {})}
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        text = response.read().decode()
        parsed = cast(dict[str, Any], json.loads(text)) if text else {}
        response_headers = {name.lower(): value for name, value in response.getheaders()}
        return HttpResult(response.status, parsed, response_headers, text)
    finally:
        connection.close()


def post(
    server: ServerAddress,
    payload: object | bytes,
    *,
    headers: dict[str, str] | None = None,
) -> HttpResult:
    return request(server, "POST", "/v1/systemone", payload, headers=headers)


@pytest.fixture(scope="module")
def server() -> Iterator[ServerAddress]:
    with running_server(create_app(StubDecisionEngine())) as address:
        yield address


class TestEngine:
    member_name = "stim-stub"

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class RecordingEngine(TestEngine):
    """A ``DecisionEngine`` that records the protocol layer's normalized inputs."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Question, ...]]] = []

    def decide(self, state_text: str, questions: Sequence[Question]) -> Decision:
        self.calls.append((state_text, tuple(questions)))
        return Decision(probabilities=tuple((1.0, 1.0) for _ in questions), input_tokens=1)


class FailingEngine(TestEngine):
    """A ``DecisionEngine`` whose failure must not leak through the wire."""

    def decide(self, state_text: str, questions: Sequence[Question]) -> Decision:
        raise RuntimeError("engine failure details")


def test_sync_sdk_round_trips_a_noul_question(server: ServerAddress) -> None:
    with TypeSafeClient(api_key="test-key", base_url=server.base_url) as client:
        response = client.system_one(
            state="I was charged twice. Please help.",
            questions={"billing": Noul(instructions="Is this about billing?")},
        )

    assert response.model == "stim-stub"
    assert 0 <= response.nouls["billing"].noul <= 1
    assert response.nouls["billing"].noul == 0.75
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 1
    assert response.request_id


def test_async_sdk_round_trips_a_noul_question(server: ServerAddress) -> None:
    async def call() -> SystemOneResponse:
        async with AsyncTypeSafeClient(api_key="test-key", base_url=server.base_url) as client:
            return await client.system_one(
                state="I was charged twice. Please help.",
                questions={"billing": Noul(instructions="Is this about billing?")},
            )

    response = asyncio.run(call())

    assert response.model == "stim-stub"
    assert response.nouls["billing"].noul == 0.75
    assert response.request_id


def test_success_response_contains_only_the_contract_fields(server: ServerAddress) -> None:
    result = post(server, PAYLOAD)

    assert result.status == 200
    assert set(result.body) == {"model", "answers", "usage", "latency_ms"}
    assert result.body["model"] == "stim-stub"
    assert set(result.body["answers"]) == {"billing"}
    answer = result.body["answers"]["billing"]
    assert answer == {"type": "noul", "noul": 0.75}
    assert "confidence" not in answer
    usage = result.body["usage"]
    assert set(usage) == {"input_tokens", "output_tokens"}
    assert isinstance(usage["input_tokens"], int)
    assert isinstance(usage["output_tokens"], int)
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] > 0
    assert ONE_DECIMAL_LATENCY.search(result.text)
    assert result.headers[REQUEST_ID_HEADER]


def test_criteria_and_state_are_rendered_for_the_engine() -> None:
    engine = RecordingEngine()
    payload = {
        "state": {"ticket": "Refund requested for a delayed order.", "urgent": True},
        "questions": {
            "urgent": {
                "type": "noul",
                "instructions": {"task": "Is the customer urgent?"},
                "criteria": {"true": "the customer is urgent", "false": "not urgent"},
            }
        },
    }

    with running_server(create_app(engine)) as server:
        result = post(server, payload)

    assert result.status == 200
    state_text, questions = engine.calls[0]
    assert state_text == "ticket: Refund requested for a delayed order.\nurgent: true"
    assert len(questions) == 1
    assert questions[0].type == "noul"
    assert questions[0].instructions == "task: Is the customer urgent?"
    assert [(option.key, option.text) for option in questions[0].options] == [
        ("no", "no: not urgent"),
        ("yes", "yes: the customer is urgent"),
    ]


def test_noul_without_criteria_uses_bare_option_names() -> None:
    engine = RecordingEngine()
    payload = {"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}}

    with running_server(create_app(engine)) as server:
        result = post(server, payload)

    assert result.status == 200
    _, questions = engine.calls[0]
    assert [(option.key, option.text) for option in questions[0].options] == [
        ("no", "no"),
        ("yes", "yes"),
    ]


def test_probabilities_are_normalized_before_rounding() -> None:
    engine = StubDecisionEngine(raw_scores={"rounded": [1.0, 2.0], "exact": [1.0, 3.0]})
    payload = {
        "state": "x",
        "questions": {
            "rounded": {"type": "noul", "instructions": "?"},
            "exact": {"type": "noul", "instructions": "?"},
        },
    }

    with running_server(create_app(engine)) as server:
        result = post(server, payload)

    assert result.status == 200
    assert result.body["answers"]["rounded"] == {"type": "noul", "noul": 0.6667}
    assert result.body["answers"]["exact"] == {"type": "noul", "noul": 0.75}


def test_zero_sum_scores_fall_back_to_a_uniform_distribution() -> None:
    engine = StubDecisionEngine(raw_scores={"q": [0.0, 0.0]})

    with running_server(create_app(engine)) as server:
        result = post(
            server, {"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}}
        )

    assert result.status == 200
    assert result.body["answers"]["q"] == {"type": "noul", "noul": 0.5}


def test_questions_are_answered_independently() -> None:
    engine = StubDecisionEngine(raw_scores={"yes": [0.0, 1.0], "no": [1.0, 0.0]})
    payload = {
        "state": "x",
        "questions": {
            "yes": {"type": "noul", "instructions": "?"},
            "no": {"type": "noul", "instructions": "?"},
        },
    }

    with running_server(create_app(engine)) as server:
        result = post(server, payload)

    assert result.status == 200
    assert result.body["answers"] == {
        "yes": {"type": "noul", "noul": 1.0},
        "no": {"type": "noul", "noul": 0.0},
    }


def test_unknown_fields_are_ignored(server: ServerAddress) -> None:
    payload = {
        "state": "x",
        "model": "stim-latest",
        "temperature": 0.3,
        "questions": {"q": {"type": "noul", "instructions": "?", "extra": {"nested": True}}},
    }

    result = post(server, payload)

    assert result.status == 200
    assert result.body["answers"]["q"] == {"type": "noul", "noul": 0.75}


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"questions": {"q": NOUL_QUESTION}}, "state"),
        ({"state": "x"}, "questions"),
        ({"state": "x", "questions": {}}, "questions"),
        ({"state": "x", "questions": []}, "questions"),
        ({"state": "x", "questions": {"": NOUL_QUESTION}}, "questions"),
        ({"state": "x", "questions": {"q": {"type": "choice"}}}, "questions.q.type"),
        ({"state": "x", "questions": {"q": {}}}, "questions.q.type"),
        ({"state": "x", "questions": {"q": "not-a-question"}}, "questions.q"),
    ],
)
def test_invalid_requests_return_422_naming_the_field(
    server: ServerAddress, payload: object, field: str
) -> None:
    result = post(server, payload)

    assert result.status == 422
    assert result.body["code"] == "invalid_request"
    assert field in result.body["detail"]
    assert set(result.body) == {"detail", "code"}
    assert "latency_ms" not in result.body
    assert result.headers[REQUEST_ID_HEADER]


def test_malformed_json_returns_422(server: ServerAddress) -> None:
    result = post(server, b"{not json")

    assert result.status == 422
    assert result.body["code"] == "invalid_request"
    assert "body" in result.body["detail"]
    assert "latency_ms" not in result.body


def test_safe_request_id_is_echoed(server: ServerAddress) -> None:
    result = post(server, PAYLOAD, headers={REQUEST_ID_HEADER: "req-123.abc_DEF~9"})

    assert result.headers[REQUEST_ID_HEADER] == "req-123.abc_DEF~9"


def test_unsafe_request_id_is_replaced(server: ServerAddress) -> None:
    result = post(server, PAYLOAD, headers={REQUEST_ID_HEADER: "bad id!"})

    assert GENERATED_REQUEST_ID.fullmatch(result.headers[REQUEST_ID_HEADER])


def test_missing_request_id_is_generated(server: ServerAddress) -> None:
    result = post(server, PAYLOAD)

    assert GENERATED_REQUEST_ID.fullmatch(result.headers[REQUEST_ID_HEADER])


def test_error_responses_carry_the_request_id(server: ServerAddress) -> None:
    result = post(server, {"questions": {}}, headers={REQUEST_ID_HEADER: "error-1"})

    assert result.status == 422
    assert result.headers[REQUEST_ID_HEADER] == "error-1"


def test_unexpected_engine_failure_returns_500_without_details() -> None:
    with running_server(create_app(FailingEngine())) as server:
        result = post(server, PAYLOAD, headers={REQUEST_ID_HEADER: "failure-1"})

    assert result.status == 500
    assert result.body == {"detail": "Internal server error.", "code": "internal_error"}
    assert "engine failure details" not in result.text
    assert "latency_ms" not in result.body
    assert result.headers[REQUEST_ID_HEADER] == "failure-1"


def test_state_over_the_configured_limit_returns_400() -> None:
    with running_server(create_app(StubDecisionEngine(), max_state=10)) as server:
        accepted = post(
            server, {"state": " ".join(["word"] * 10), "questions": {"q": NOUL_QUESTION}}
        )
        rejected = post(
            server, {"state": " ".join(["word"] * 11), "questions": {"q": NOUL_QUESTION}}
        )

    assert accepted.status == 200
    assert rejected.status == 400
    assert rejected.body["code"] == "max_tokens_exceeded"
    assert "11" in rejected.body["detail"]
    assert "10" in rejected.body["detail"]
    assert "latency_ms" not in rejected.body
    assert rejected.headers[REQUEST_ID_HEADER]


def test_default_max_state_is_8192(server: ServerAddress) -> None:
    result = post(server, {"state": " ".join(["word"] * 8193), "questions": {"q": NOUL_QUESTION}})

    assert result.status == 400
    assert result.body["code"] == "max_tokens_exceeded"
    assert "8193" in result.body["detail"]
    assert "8192" in result.body["detail"]


@pytest.mark.parametrize("model", ["stim-latest", "jev-latest", "stim-stub"])
def test_supported_model_names_resolve_to_the_loaded_member(
    server: ServerAddress, model: str
) -> None:
    result = post(server, {**PAYLOAD, "model": model})

    assert result.status == 200
    assert result.body["model"] == "stim-stub"


def test_unknown_model_name_returns_404(server: ServerAddress) -> None:
    result = post(server, {**PAYLOAD, "model": "stim-9b"})

    assert result.status == 404
    assert result.body["code"] == "model_not_found"
    assert "stim-9b" in result.body["detail"]
    assert "latency_ms" not in result.body


def test_non_post_methods_are_rejected(server: ServerAddress) -> None:
    result = request(server, "GET", "/v1/systemone")

    assert result.status == 405
    assert result.body["code"] == "method_not_allowed"
    assert result.headers[REQUEST_ID_HEADER]


def test_unknown_path_returns_404(server: ServerAddress) -> None:
    result = request(server, "POST", "/v1/unknown", b"{}")

    assert result.status == 404
    assert result.body["code"] == "not_found"
    assert result.headers[REQUEST_ID_HEADER]


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_documentation_routes_are_not_exposed(server: ServerAddress, path: str) -> None:
    result = request(server, "GET", path)

    assert result.status == 404
    assert result.body["code"] == "not_found"
    assert result.headers[REQUEST_ID_HEADER]


def test_assets_are_rejected_explicitly(server: ServerAddress) -> None:
    payload = {
        "state": "Product photo",
        "questions": {"q": NOUL_QUESTION},
        "assets": {"photo": {"type": "image", "part": "asset-0"}},
    }

    result = post(server, payload)

    assert result.status == 415
    assert result.body["code"] == "unsupported_media_type"
    assert "assets" in result.body["detail"]
    assert "latency_ms" not in result.body
    assert result.headers[REQUEST_ID_HEADER]


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_standard_json_constants_are_rejected(server: ServerAddress, constant: str) -> None:
    body = (
        b'{"state": '
        + constant.encode()
        + b', "questions": {"q": {"type": "noul", "instructions": "?"}}}'
    )

    result = post(server, body)

    assert result.status == 422
    assert result.body["code"] == "invalid_request"
    assert "body" in result.body["detail"]
