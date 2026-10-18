"""Pure-ASGI implementation of the System One HTTP protocol."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections.abc import (
    Awaitable,
    Callable,
    Collection,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)
from typing import Any, NoReturn
from uuid import uuid4

from pydantic import JsonValue, ValidationError

from stim_systemone.engine import DecisionEngine, Option, Question
from stim_systemone.rendering import render_content
from stim_systemone.wire import DEFAULT_MODEL, NoulQuestion, SystemOneRequest

LOGGER = logging.getLogger("stim_systemone")

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

REQUEST_ID_HEADER = b"x-typesafe-request-id"
REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._~-]{1,128}")
SYSTEM_ONE_PATH = "/v1/systemone"
DEFAULT_MAX_STATE = 8192
DEFAULT_MODEL_ALIASES = (DEFAULT_MODEL, "jev-latest")


class ProtocolError(Exception):
    """An error that maps directly onto the documented HTTP error body."""

    def __init__(
        self,
        status: int,
        code: str,
        detail: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail
        self.headers = dict(headers or {})


class _ClientDisconnected(Exception):
    """The requester went away before sending a complete body."""


class SystemOneApp:
    """Serve ``POST /v1/systemone`` as an ASGI application.

    The app owns every wire concern: parsing, rendering, normalization, error mapping,
    request IDs, usage, and latency. A ``DecisionEngine`` only scores normalized
    questions and never sees HTTP or pydantic types.
    """

    def __init__(
        self,
        engine: DecisionEngine,
        *,
        max_state: int = DEFAULT_MAX_STATE,
        model_aliases: Collection[str] = DEFAULT_MODEL_ALIASES,
    ) -> None:
        if max_state < 1:
            raise ValueError("max_state must be positive")
        self._engine = engine
        self._max_state = max_state
        self._model_aliases = frozenset(model_aliases)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope["type"] != "http":
            return

        request_id = _request_id(scope.get("headers", []))
        started = time.perf_counter()
        try:
            payload = await self._handle(scope, receive)
        except _ClientDisconnected:
            return
        except ProtocolError as error:
            await self._send_json(
                send,
                error.status,
                {"detail": error.detail, "code": error.code},
                request_id,
                headers=error.headers,
            )
            return
        except Exception:
            LOGGER.exception("Unhandled error while answering a System One request")
            await self._send_json(
                send,
                500,
                {"detail": "Internal server error.", "code": "internal_error"},
                request_id,
            )
            return

        payload["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        await self._send_json(send, 200, payload, request_id)

    async def _handle(self, scope: Scope, receive: Receive) -> dict[str, Any]:
        path = scope.get("path", "")
        if path != SYSTEM_ONE_PATH:
            raise ProtocolError(404, "not_found", f"path: no route for {path!r}")
        if scope.get("method") != "POST":
            raise ProtocolError(
                405,
                "method_not_allowed",
                "method: only POST is allowed",
                headers={"Allow": "POST"},
            )

        data = _decode_json(await _read_body(receive))
        request, questions = _parse_request(data)

        state_text = render_content(request.state)
        state_tokens = self._engine.count_tokens(state_text)
        if state_tokens > self._max_state:
            raise ProtocolError(
                400,
                "max_tokens_exceeded",
                f"state: {state_tokens} tokens exceeds the maximum of {self._max_state} tokens",
            )

        member_name = self._resolve_model(request.model)
        engine_questions = tuple(
            _engine_question(name, question) for name, question in questions.items()
        )
        decision = self._engine.decide(state_text, engine_questions)
        answers = _answers(engine_questions, decision.probabilities)
        serialized_answers = json.dumps(answers, ensure_ascii=False, separators=(",", ":"))
        return {
            "model": member_name,
            "answers": answers,
            "usage": {
                "input_tokens": decision.input_tokens,
                "output_tokens": self._engine.count_tokens(serialized_answers),
            },
        }

    def _resolve_model(self, requested: str) -> str:
        member_name = self._engine.member_name
        if requested == member_name or requested in self._model_aliases:
            return member_name
        raise ProtocolError(404, "model_not_found", f"model: unknown model {requested!r}")

    async def _send_json(
        self,
        send: Send,
        status: int,
        payload: Mapping[str, Any],
        request_id: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        response_headers: list[tuple[bytes, bytes]] = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
            (REQUEST_ID_HEADER, request_id.encode("ascii")),
        ]
        for name, value in (headers or {}).items():
            response_headers.append((name.lower().encode("ascii"), value.encode("latin-1")))
        await send({"type": "http.response.start", "status": status, "headers": response_headers})
        await send({"type": "http.response.body", "body": body})

    async def _lifespan(self, receive: Receive, send: Send) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return


def _parse_request(data: object) -> tuple[SystemOneRequest, dict[str, NoulQuestion]]:
    if not isinstance(data, dict):
        _invalid("body: request body must be a JSON object")
    if data.get("assets") is not None:
        raise ProtocolError(
            415,
            "unsupported_media_type",
            "assets: image and media inputs are not supported",
        )
    try:
        request = SystemOneRequest.model_validate(data)
    except ValidationError as error:
        _invalid(_format_validation_error(error))

    questions: dict[str, NoulQuestion] = {}
    for name, raw in request.questions.items():
        if not name:
            _invalid("questions: question ids must be non-empty strings")
        if not isinstance(raw, dict):
            _invalid(f"questions.{name}: question must be a JSON object")
        if "type" not in raw:
            _invalid(f"questions.{name}.type: Field required")
        question_type = raw["type"]
        if not isinstance(question_type, str) or not question_type:
            _invalid(f"questions.{name}.type: question type must be a non-empty string")
        if question_type != "noul":
            _invalid(f"questions.{name}.type: unsupported question type {question_type!r}")
        try:
            questions[name] = NoulQuestion.model_validate(raw)
        except ValidationError as error:
            _invalid(_format_validation_error(error, f"questions.{name}"))
    return request, questions


def _invalid(detail: str) -> NoReturn:
    raise ProtocolError(422, "invalid_request", detail)


def _format_validation_error(error: ValidationError, prefix: str = "") -> str:
    first = error.errors(include_url=False)[0]
    location = ".".join(str(part) for part in first["loc"])
    path = ".".join(part for part in (prefix, location) if part)
    return f"{path}: {first['msg']}" if path else str(first["msg"])


def _engine_question(name: str, question: NoulQuestion) -> Question:
    criteria = question.criteria
    return Question(
        name=name,
        type="noul",
        instructions=render_content(question.instructions),
        options=(
            Option(key="no", text=_render_option("no", criteria.false if criteria else None)),
            Option(key="yes", text=_render_option("yes", criteria.true if criteria else None)),
        ),
    )


def _render_option(key: str, description: JsonValue) -> str:
    rendered = render_content(description)
    if not rendered:
        return key
    if "\n" not in rendered:
        return f"{key}: {rendered}"
    indented = "\n".join(f"  {line}" for line in rendered.splitlines())
    return f"{key}:\n{indented}"


def _answers(
    questions: Sequence[Question], rows: Sequence[Sequence[float]]
) -> dict[str, dict[str, Any]]:
    if len(rows) != len(questions):
        raise ValueError("engine returned a different number of answers than questions")
    answers: dict[str, dict[str, Any]] = {}
    for question, scores in zip(questions, rows, strict=True):
        if question.type != "noul" or len(question.options) != 2:
            raise ValueError(f"unsupported engine question {question.name!r}")
        if len(scores) != 2:
            raise ValueError("a noul answer needs one score per option")
        probabilities = _normalize(scores)
        answers[question.name] = {"type": "noul", "noul": probabilities[1]}
    return answers


def _normalize(scores: Sequence[float]) -> list[float]:
    if not scores:
        raise ValueError("engine returned no option scores")
    if any(not math.isfinite(score) or score < 0 for score in scores):
        raise ValueError("engine option scores must be finite and non-negative")
    total = math.fsum(scores)
    if total == 0:
        return [round(1 / len(scores), 4) for _ in scores]
    return [round(score / total, 4) for score in scores]


def _decode_json(body: bytes) -> object:
    try:
        return json.loads(body, parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError):
        _invalid("body: request body must be valid JSON")


def _reject_constant(constant: str) -> object:
    raise ValueError(f"invalid JSON constant: {constant}")


async def _read_body(receive: Receive) -> bytes:
    chunks: list[bytes] = []
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            raise _ClientDisconnected
        if message["type"] != "http.request":
            continue
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            return b"".join(chunks)


def _request_id(headers: Iterable[tuple[bytes, bytes]]) -> str:
    for name, value in headers:
        if name.lower() == REQUEST_ID_HEADER:
            candidate = value.decode("latin-1")
            if REQUEST_ID_PATTERN.fullmatch(candidate):
                return candidate
            break
    return uuid4().hex
