"""Shared HTTP harness for the protocol seam tests."""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from typing import Any, cast

import uvicorn

from stim_systemone import MemberMetadata

STUB_METADATA = MemberMetadata(
    description="Deterministic stub of a Stim decision member.",
    release_date=date(2026, 1, 1),
    validated_context=8192,
)
REQUEST_ID_HEADER = "x-typesafe-request-id"


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


def assert_error(result: HttpResult, status: int, code: str) -> None:
    """Assert the documented error shape for a rejected request."""
    assert result.status == status
    assert set(result.body) == {"detail", "code"}
    assert result.body["code"] == code
    assert isinstance(result.body["detail"], str) and result.body["detail"]
    assert "latency_ms" not in result.body
    assert result.headers[REQUEST_ID_HEADER]


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
    content_type: str | None = "application/json",
) -> HttpResult:
    if payload is None:
        body = None
    elif isinstance(payload, bytes):
        body = payload
    else:
        body = json.dumps(payload).encode()
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    try:
        request_headers = dict(headers or {})
        if content_type is not None:
            request_headers.setdefault("Content-Type", content_type)
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
    content_type: str | None = "application/json",
) -> HttpResult:
    return request(
        server,
        "POST",
        "/v1/systemone",
        payload,
        headers=headers,
        content_type=content_type,
    )


def read_raw_response(sock: socket.socket) -> HttpResult:
    """Read one HTTP response from a socket, for requests built by hand."""
    received = b""
    while b"\r\n\r\n" not in received:
        chunk = sock.recv(65536)
        if not chunk:
            raise RuntimeError("connection closed before the response headers")
        received += chunk
    head, _, body = received.partition(b"\r\n\r\n")
    status_line, *header_lines = head.split(b"\r\n")
    status = int(status_line.split(b" ", 2)[1])
    response_headers: dict[str, str] = {}
    for line in header_lines:
        name, _, value = line.partition(b":")
        response_headers[name.decode("latin-1").lower()] = value.strip().decode("latin-1")
    expected = int(response_headers.get("content-length", "0"))
    while len(body) < expected:
        chunk = sock.recv(65536)
        if not chunk:
            break
        body += chunk
    text = body[:expected].decode()
    parsed = cast(dict[str, Any], json.loads(text)) if text else {}
    return HttpResult(status, parsed, response_headers, text)
