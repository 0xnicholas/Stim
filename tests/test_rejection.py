"""Raw-HTTP checks for the documented rejection surface.

Every case goes through a real socket: no request reaches the engine unless it is a
supported, well-formed, in-limit System One request. Error bodies are exactly
``{"detail", "code"}``, carry a non-empty request id, and never carry ``latency_ms``.
"""

from __future__ import annotations

import logging
import socket
import threading
from collections.abc import Iterator, Sequence

import pytest

from stim_serve.app import create_app
from stim_systemone import Decision, Question, StubDecisionEngine
from tests.http_support import (
    HttpResult,
    ServerAddress,
    post,
    read_raw_response,
    request,
    running_server,
)

REQUEST_ID_HEADER = "x-typesafe-request-id"
BODY_LIMIT = 32 * 1024 * 1024
NOUL = {"type": "noul", "instructions": "Is this about billing?"}
PAYLOAD = {"state": "I was charged twice.", "questions": {"billing": NOUL}}


@pytest.fixture(scope="module")
def server() -> Iterator[ServerAddress]:
    with running_server(create_app(StubDecisionEngine())) as address:
        yield address


def assert_rejected(result: HttpResult, status: int, code: str) -> None:
    assert result.status == status
    assert set(result.body) == {"detail", "code"}
    assert result.body["code"] == code
    assert isinstance(result.body["detail"], str) and result.body["detail"]
    assert "latency_ms" not in result.body
    assert result.headers[REQUEST_ID_HEADER]


def post_head(*header_lines: str) -> bytes:
    return (
        "\r\n".join(
            (
                "POST /v1/systemone HTTP/1.1",
                "Host: 127.0.0.1",
                *header_lines,
                "Connection: close",
                "",
                "",
            )
        )
    ).encode("ascii")


def chunked_post(
    server: ServerAddress, chunks: Sequence[bytes], *, terminate: bool = True
) -> HttpResult:
    """POST a chunked body, reading the response while another thread writes."""
    sock = socket.create_connection(("127.0.0.1", server.port), timeout=10)
    head = post_head("Content-Type: application/json", "Transfer-Encoding: chunked")

    def send() -> None:
        try:
            sock.sendall(head)
            for chunk in chunks:
                sock.sendall(f"{len(chunk):x}\r\n".encode("ascii"))
                sock.sendall(chunk)
                sock.sendall(b"\r\n")
            if terminate:
                sock.sendall(b"0\r\n\r\n")
        except OSError:
            pass

    sender = threading.Thread(target=send, daemon=True)
    sender.start()
    try:
        return read_raw_response(sock)
    finally:
        sock.close()
        sender.join(timeout=5)


def test_state_plus_one_question_over_the_context_limit_returns_400() -> None:
    accepted = {
        "state": "a b c",
        "questions": {"q": {"type": "noul", "instructions": "d e f g h"}},
    }
    rejected = {
        "state": "a b c",
        "questions": {"q": {"type": "noul", "instructions": "d e f g h i"}},
    }

    with running_server(create_app(StubDecisionEngine(), max_context=10)) as address:
        accepted_result = post(address, accepted)
        rejected_result = post(address, rejected, headers={REQUEST_ID_HEADER: "row-1"})

    assert accepted_result.status == 200
    assert_rejected(rejected_result, 400, "max_tokens_exceeded")
    assert "11" in rejected_result.body["detail"]
    assert "10" in rejected_result.body["detail"]
    assert rejected_result.headers[REQUEST_ID_HEADER] == "row-1"


def test_context_limit_counts_choice_option_text() -> None:
    accepted = {
        "state": "one two",
        "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": None}}},
    }
    rejected = {
        "state": "one two",
        "questions": {
            "q": {"type": "choice", "criteria": {f"option-{index}": None for index in range(9)}}
        },
    }

    with running_server(create_app(StubDecisionEngine(), max_context=10)) as address:
        accepted_result = post(address, accepted)
        rejected_result = post(address, rejected)

    assert accepted_result.status == 200
    assert_rejected(rejected_result, 400, "max_tokens_exceeded")
    assert "11" in rejected_result.body["detail"]
    assert "10" in rejected_result.body["detail"]


@pytest.mark.parametrize("path", ["/v1/systemone/permute", "/v1/systemone/separate"])
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_extension_routes_are_not_registered(server: ServerAddress, path: str, method: str) -> None:
    result = request(server, method, path, PAYLOAD if method == "POST" else None)

    assert_rejected(result, 404, "not_found")


def test_wrong_method_is_rejected_with_allow(server: ServerAddress) -> None:
    result = request(server, "GET", "/v1/systemone")

    assert_rejected(result, 405, "method_not_allowed")
    assert result.headers["allow"] == "POST"


def test_unknown_model_returns_404(server: ServerAddress) -> None:
    result = post(server, {**PAYLOAD, "model": "stim-9b"})

    assert_rejected(result, 404, "model_not_found")
    assert "stim-9b" in result.body["detail"]


def test_oversized_content_length_is_rejected_before_reading_the_body(
    server: ServerAddress,
) -> None:
    head = post_head(
        "Content-Type: application/json",
        f"Content-Length: {BODY_LIMIT + 1}",
    )

    with socket.create_connection(("127.0.0.1", server.port), timeout=10) as sock:
        sock.sendall(head)
        result = read_raw_response(sock)

    assert_rejected(result, 413, "payload_too_large")
    assert str(BODY_LIMIT) in result.body["detail"]


def test_oversized_chunked_body_is_rejected_while_reading(server: ServerAddress) -> None:
    megabyte = 1024 * 1024
    chunks = [b" " * megabyte] * (BODY_LIMIT // megabyte + 1)

    # No terminating chunk: a server that waits for the whole body cannot answer.
    result = chunked_post(server, chunks, terminate=False)

    assert_rejected(result, 413, "payload_too_large")
    assert str(BODY_LIMIT) in result.body["detail"]


@pytest.mark.parametrize(
    "content_type",
    [
        pytest.param("text/plain", id="text"),
        pytest.param("application/octet-stream", id="binary"),
        pytest.param("multipart/form-data; boundary=stim", id="multipart"),
        pytest.param(None, id="missing"),
    ],
)
def test_non_json_content_type_returns_415(server: ServerAddress, content_type: str | None) -> None:
    result = post(server, PAYLOAD, content_type=content_type)

    assert_rejected(result, 415, "unsupported_media_type")
    assert "content-type" in result.body["detail"]


@pytest.mark.parametrize(
    "content_type",
    [
        "application/json",
        "application/json; charset=utf-8",
        "Application/JSON",
        "application/vnd.stim+json",
    ],
)
def test_json_content_types_are_accepted(server: ServerAddress, content_type: str) -> None:
    assert post(server, PAYLOAD, content_type=content_type).status == 200


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "br", "zstd", "gzip, identity"])
def test_non_identity_content_encoding_returns_415(server: ServerAddress, encoding: str) -> None:
    result = post(server, PAYLOAD, headers={"Content-Encoding": encoding})

    assert_rejected(result, 415, "unsupported_content_encoding")
    assert "content-encoding" in result.body["detail"]


@pytest.mark.parametrize("encoding", ["identity", " Identity "])
def test_identity_content_encoding_is_accepted(server: ServerAddress, encoding: str) -> None:
    assert post(server, PAYLOAD, headers={"Content-Encoding": encoding}).status == 200


@pytest.mark.parametrize(
    "assets",
    [None, {}, {"photo": {"type": "image", "part": "asset-0"}}],
    ids=["null", "empty", "image"],
)
def test_assets_are_rejected_explicitly(server: ServerAddress, assets: object) -> None:
    result = post(server, {**PAYLOAD, "assets": assets})

    assert_rejected(result, 415, "unsupported_media_type")
    assert "assets" in result.body["detail"]


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        pytest.param({"questions": {"q": NOUL}}, "state", id="missing-state"),
        pytest.param({"state": "x"}, "questions", id="missing-questions"),
        pytest.param({"state": "x", "questions": {}}, "questions", id="empty-questions"),
        pytest.param({"state": "x", "questions": {"": NOUL}}, "questions", id="empty-id"),
        pytest.param(
            {"state": "x", "questions": {"q": "not-an-object"}},
            "questions.q",
            id="non-object-question",
        ),
        pytest.param(
            {"state": "x", "questions": {"q": {"type": "box"}}},
            "questions.q.type",
            id="box",
        ),
        pytest.param(
            {"state": "x", "questions": {"q": {"type": "ranking"}}},
            "questions.q.type",
            id="unknown-type",
        ),
        pytest.param(
            {"state": "x", "questions": {"q": {"type": "choice"}}},
            "questions.q.criteria",
            id="missing-choice-criteria",
        ),
        pytest.param(
            {"state": "x", "questions": {"q": {"type": "choice", "criteria": {}}}},
            "questions.q.criteria",
            id="zero-choice-options",
        ),
        pytest.param(
            {
                "state": "x",
                "questions": {
                    "q": {
                        "type": "choice",
                        "criteria": {f"option-{index}": None for index in range(256)},
                    }
                },
            },
            "questions.q.criteria",
            id="too-many-choice-options",
        ),
        pytest.param(
            {"state": "x", "questions": {"q": {"type": "score"}}},
            "questions.q.criteria",
            id="missing-score-criteria",
        ),
        pytest.param(
            {"state": "x", "questions": {"q": {"type": "score", "criteria": ["only"]}}},
            "questions.q.criteria",
            id="one-score-level",
        ),
        pytest.param(
            {
                "state": "x",
                "questions": {
                    "q": {"type": "score", "criteria": [f"level-{index}" for index in range(11)]}
                },
            },
            "questions.q.criteria",
            id="eleven-score-levels",
        ),
    ],
)
def test_invalid_requests_return_422_naming_the_field(
    server: ServerAddress, payload: object, field: str
) -> None:
    result = post(server, payload)

    assert_rejected(result, 422, "invalid_request")
    assert field in result.body["detail"]


def test_invalid_requests_do_not_echo_caller_content(server: ServerAddress) -> None:
    payload = {
        "state": "SECRET_STATE_VALUE",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "SECRET_INSTRUCTION_VALUE",
                "criteria": "SECRET_CRITERIA_VALUE",
            }
        },
    }

    result = post(server, payload)

    assert_rejected(result, 422, "invalid_request")
    assert "SECRET" not in result.text


def test_malformed_json_does_not_echo_the_body(server: ServerAddress) -> None:
    result = post(server, b'{"state": "SECRET_STATE_VALUE"')

    assert_rejected(result, 422, "invalid_request")
    assert "SECRET_STATE_VALUE" not in result.text


def test_row_limit_errors_do_not_echo_caller_content() -> None:
    payload = {
        "state": "a b c",
        "questions": {
            "q": {
                "type": "noul",
                "instructions": "SECRET_INSTRUCTION_VALUE",
                "criteria": {"true": "SECRET_YES_VALUE", "false": "SECRET_NO_VALUE"},
            }
        },
    }

    with running_server(create_app(StubDecisionEngine(), max_context=3)) as address:
        result = post(address, payload)

    assert_rejected(result, 400, "max_tokens_exceeded")
    assert "SECRET" not in result.text


class FailingEngine:
    member_name = "stim-stub"

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def decide(self, state_text: str, questions: Sequence[Question]) -> Decision:
        raise RuntimeError("SECRET_INTERNAL_DETAILS")


def test_internal_errors_return_500_without_details() -> None:
    with running_server(create_app(FailingEngine())) as address:
        result = post(address, PAYLOAD, headers={REQUEST_ID_HEADER: "failure-1"})

    assert_rejected(result, 500, "internal_error")
    assert result.headers[REQUEST_ID_HEADER] == "failure-1"
    assert "SECRET_INTERNAL_DETAILS" not in result.text


def test_internal_errors_do_not_log_details(caplog: pytest.LogCaptureFixture) -> None:
    with (
        caplog.at_level(logging.ERROR, logger="stim_systemone"),
        running_server(create_app(FailingEngine())) as address,
    ):
        post(address, PAYLOAD)

    assert "Unhandled error" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "SECRET_INTERNAL_DETAILS" not in caplog.text
