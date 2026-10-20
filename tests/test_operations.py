"""Raw-HTTP checks for the operator-facing access surface.

With no API key every endpoint is open. Setting an API key turns every ``/v1/*`` route
into a Bearer-only route while ``/healthz`` and ``/readyz`` stay open. The limiter and
overload hooks are injected through the application factory; their rejections are stable
``{detail, code}`` errors with a request id and no ``latency_ms``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from stim_serve.app import create_app
from stim_systemone import StubDecisionEngine
from tests.http_support import (
    REQUEST_ID_HEADER,
    ServerAddress,
    assert_error,
    post,
    request,
    running_server,
)

API_KEY = "sk-stim-test-9f3c"
AUTHORIZATION = "Authorization"
BEARER = f"Bearer {API_KEY}"
NOUL = {"type": "noul", "instructions": "Is this about billing?"}
PAYLOAD = {"state": "I was charged twice.", "questions": {"billing": NOUL}}
PROBE_PATHS = ("/healthz", "/readyz")
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def open_server() -> Iterator[ServerAddress]:
    with running_server(create_app(StubDecisionEngine())) as address:
        yield address


@pytest.fixture(scope="module")
def secured_server() -> Iterator[ServerAddress]:
    with running_server(create_app(StubDecisionEngine(), api_key=API_KEY)) as address:
        yield address


def test_without_an_api_key_every_endpoint_is_open(open_server: ServerAddress) -> None:
    assert post(open_server, PAYLOAD).status == 200
    assert request(open_server, "GET", "/v1/models", content_type=None).status == 200
    for path in PROBE_PATHS:
        assert request(open_server, "GET", path, content_type=None).status == 200


@pytest.mark.parametrize(
    "authorization",
    [
        pytest.param(None, id="missing"),
        pytest.param("Bearer wrong-key", id="wrong-key"),
        pytest.param(f"Basic {API_KEY}", id="wrong-scheme"),
        pytest.param("Bearer", id="no-credentials"),
        pytest.param("Bearer ", id="empty-credentials"),
        pytest.param(f"Bearer {API_KEY} extra", id="extra-credential"),
    ],
)
def test_missing_or_invalid_bearer_is_unauthorized(
    secured_server: ServerAddress, authorization: str | None
) -> None:
    headers = {} if authorization is None else {AUTHORIZATION: authorization}
    result = post(secured_server, PAYLOAD, headers=headers)

    assert_error(result, 401, "unauthorized")
    assert result.headers["www-authenticate"] == "Bearer"
    assert API_KEY not in result.text


def test_correct_bearer_is_accepted(secured_server: ServerAddress) -> None:
    result = post(secured_server, PAYLOAD, headers={AUTHORIZATION: BEARER})

    assert result.status == 200
    assert result.body["model"] == "stim-stub"


def test_bearer_scheme_is_case_insensitive(secured_server: ServerAddress) -> None:
    result = post(secured_server, PAYLOAD, headers={AUTHORIZATION: f"bearer {API_KEY}"})

    assert result.status == 200


def test_models_endpoint_requires_the_bearer(secured_server: ServerAddress) -> None:
    unauthenticated = request(secured_server, "GET", "/v1/models", content_type=None)
    authenticated = request(
        secured_server,
        "GET",
        "/v1/models",
        headers={AUTHORIZATION: BEARER},
        content_type=None,
    )

    assert_error(unauthenticated, 401, "unauthorized")
    assert unauthenticated.headers["www-authenticate"] == "Bearer"
    assert authenticated.status == 200


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_unknown_v1_routes_require_authentication_first(
    secured_server: ServerAddress, method: str
) -> None:
    payload = PAYLOAD if method == "POST" else None
    unauthenticated = request(secured_server, method, "/v1/unknown", payload)
    authenticated = request(
        secured_server,
        method,
        "/v1/unknown",
        payload,
        headers={AUTHORIZATION: BEARER},
    )

    assert_error(unauthenticated, 401, "unauthorized")
    assert_error(authenticated, 404, "not_found")


@pytest.mark.parametrize("path", PROBE_PATHS)
def test_probes_do_not_require_authentication(secured_server: ServerAddress, path: str) -> None:
    result = request(secured_server, "GET", path, content_type=None)

    assert result.status == 200
    assert result.body == {"status": "ok"}
    assert "latency_ms" not in result.body
    assert result.headers[REQUEST_ID_HEADER]


@pytest.mark.parametrize("path", PROBE_PATHS)
def test_probes_reject_other_methods(path: str) -> None:
    with running_server(create_app(StubDecisionEngine())) as address:
        result = request(address, "POST", path, PAYLOAD)

    assert_error(result, 405, "method_not_allowed")
    assert result.headers["allow"] == "GET"


def test_probes_bypass_the_limiter_and_overload_hooks() -> None:
    with running_server(
        create_app(
            StubDecisionEngine(),
            limiter=lambda scope: 1,
            is_overloaded=lambda: True,
        )
    ) as address:
        assert_error(post(address, PAYLOAD), 529, "overloaded")
        for path in PROBE_PATHS:
            assert request(address, "GET", path, content_type=None).status == 200


def test_api_key_is_never_logged(
    secured_server: ServerAddress, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG):
        post(secured_server, PAYLOAD, headers={AUTHORIZATION: BEARER})
        post(secured_server, PAYLOAD, headers={AUTHORIZATION: "Bearer wrong-key"})
        post(secured_server, PAYLOAD)
        request(secured_server, "GET", "/v1/models", content_type=None)

    assert API_KEY not in caplog.text
    assert "wrong-key" not in caplog.text


def test_rejecting_limiter_returns_429_with_a_valid_retry_after() -> None:
    with running_server(create_app(StubDecisionEngine(), limiter=lambda scope: 1.4)) as address:
        result = post(address, PAYLOAD, headers={REQUEST_ID_HEADER: "limited-1"})

    assert_error(result, 429, "rate_limited")
    assert result.headers["retry-after"] == "2"
    assert result.headers[REQUEST_ID_HEADER] == "limited-1"


def test_allowing_limiter_keeps_the_service_open() -> None:
    with running_server(create_app(StubDecisionEngine(), limiter=lambda scope: None)) as address:
        assert post(address, PAYLOAD).status == 200
        assert request(address, "GET", "/v1/models", content_type=None).status == 200


def test_limiter_receives_v1_requests_only() -> None:
    seen: list[str] = []

    def limiter(scope: Any) -> float | None:
        seen.append(scope["path"])
        return None

    with running_server(create_app(StubDecisionEngine(), limiter=limiter)) as address:
        assert post(address, PAYLOAD).status == 200
        assert request(address, "GET", "/v1/models", content_type=None).status == 200
        for path in PROBE_PATHS:
            assert request(address, "GET", path, content_type=None).status == 200

    assert seen == ["/v1/systemone", "/v1/models"]


def test_limiter_does_not_guard_the_probes() -> None:
    with running_server(create_app(StubDecisionEngine(), limiter=lambda scope: 30)) as address:
        assert_error(post(address, PAYLOAD), 429, "rate_limited")
        for path in PROBE_PATHS:
            assert request(address, "GET", path, content_type=None).status == 200


def test_authentication_precedes_rate_limiting() -> None:
    with running_server(
        create_app(StubDecisionEngine(), api_key=API_KEY, limiter=lambda scope: 1)
    ) as address:
        unauthenticated = post(address, PAYLOAD)
        authenticated = post(address, PAYLOAD, headers={AUTHORIZATION: BEARER})

    assert_error(unauthenticated, 401, "unauthorized")
    assert_error(authenticated, 429, "rate_limited")


class OverloadSwitch:
    """A mutable overload hook so one server can demonstrate both states."""

    def __init__(self) -> None:
        self.overloaded = False

    def __call__(self) -> bool:
        return self.overloaded


def test_overload_hook_is_switchable_and_returns_529() -> None:
    switch = OverloadSwitch()

    with running_server(create_app(StubDecisionEngine(), is_overloaded=switch)) as address:
        assert post(address, PAYLOAD).status == 200

        switch.overloaded = True
        system_one = post(address, PAYLOAD, headers={REQUEST_ID_HEADER: "overloaded-1"})
        models = request(address, "GET", "/v1/models", content_type=None)

        assert_error(system_one, 529, "overloaded")
        assert system_one.headers[REQUEST_ID_HEADER] == "overloaded-1"
        assert_error(models, 529, "overloaded")

        for path in PROBE_PATHS:
            assert request(address, "GET", path, content_type=None).status == 200

        switch.overloaded = False
        assert post(address, PAYLOAD).status == 200


def test_authentication_precedes_overload() -> None:
    with running_server(
        create_app(StubDecisionEngine(), api_key=API_KEY, is_overloaded=lambda: True)
    ) as address:
        unauthenticated = post(address, PAYLOAD)
        authenticated = post(address, PAYLOAD, headers={AUTHORIZATION: BEARER})

    assert_error(unauthenticated, 401, "unauthorized")
    assert_error(authenticated, 529, "overloaded")


def test_overload_precedes_rate_limiting() -> None:
    with running_server(
        create_app(
            StubDecisionEngine(),
            limiter=lambda scope: 1,
            is_overloaded=lambda: True,
        )
    ) as address:
        result = post(address, PAYLOAD)

    assert_error(result, 529, "overloaded")


def test_operational_surfaces_are_registered_in_the_compatibility_matrix() -> None:
    matrix: dict[str, Any] = json.loads(
        (REPO_ROOT / "docs" / "compatibility.json").read_text(encoding="utf-8")
    )
    entries = {entry["id"]: entry for entry in matrix["entries"]}

    expected = {
        "auth.default_open": (
            "supported",
            "tests/test_operations.py::test_without_an_api_key_every_endpoint_is_open",
        ),
        "auth.bearer": (
            "supported",
            "tests/test_operations.py::test_correct_bearer_is_accepted",
        ),
        "auth.key_not_logged": (
            "supported",
            "tests/test_operations.py::test_api_key_is_never_logged",
        ),
        "auth.probes_unprotected": (
            "supported",
            "tests/test_operations.py::test_probes_do_not_require_authentication",
        ),
        "error.unauthorized": (
            "supported",
            "tests/test_operations.py::test_missing_or_invalid_bearer_is_unauthorized",
        ),
        "error.rate_limited": (
            "supported",
            "tests/test_operations.py::test_rejecting_limiter_returns_429_with_a_valid_retry_after",
        ),
        "error.overloaded": (
            "supported",
            "tests/test_operations.py::test_overload_hook_is_switchable_and_returns_529",
        ),
        "endpoint.healthz": (
            "extension",
            "tests/test_operations.py::test_probes_bypass_the_limiter_and_overload_hooks",
        ),
        "endpoint.readyz": (
            "extension",
            "tests/test_operations.py::test_probes_bypass_the_limiter_and_overload_hooks",
        ),
    }
    for entry_id, (status, test_id) in expected.items():
        entry = entries[entry_id]
        assert entry["status"] == status
        assert entry["expected_behavior"]
        assert entry["test_id"] == test_id
