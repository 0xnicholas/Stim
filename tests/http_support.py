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
from typing import Any, cast

import uvicorn


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
