"""Model identity, compatibility aliases, and the model catalog over the HTTP seam.

The service answers with the loaded member's name no matter which accepted alias the
client sends, and ``GET /v1/models`` advertises only ``stim-latest`` plus that member.
The official TypeSafe SDK is the primary client for these checks.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from typesafe_sdk import AsyncTypeSafeClient, Noul, TypeSafeClient

from stim_serve.app import create_app
from stim_systemone import MemberMetadata, StubDecisionEngine
from tests.http_support import ServerAddress, post, request, running_server

REQUEST_ID_HEADER = "x-typesafe-request-id"
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
REPO_ROOT = Path(__file__).resolve().parents[1]
STATE = "I was charged twice."
QUESTION = {"type": "noul", "instructions": "Is this about billing?"}
PAYLOAD = {"state": STATE, "questions": {"billing": QUESTION}}
LATEST = "stim-latest"
MEMBER = "stim-stub"
CARD_FIELDS = {
    "name",
    "description",
    "release_date",
    "max_state_tokens",
    "validated_context",
}


@pytest.fixture(scope="module")
def server() -> Iterator[ServerAddress]:
    with running_server(create_app(StubDecisionEngine())) as address:
        yield address


def get_models(server: ServerAddress, **kwargs: Any) -> Any:
    return request(server, "GET", "/v1/models", **kwargs)


def model_names(server: ServerAddress) -> list[str]:
    result = get_models(server)
    assert result.status == 200
    return [card["name"] for card in result.body["models"]]


def test_models_endpoint_lists_the_latest_alias_and_loaded_member(
    server: ServerAddress,
) -> None:
    result = get_models(server, content_type=None)

    assert result.status == 200
    assert set(result.body) == {"models"}
    assert [card["name"] for card in result.body["models"]] == [LATEST, MEMBER]
    assert "latency_ms" not in result.body
    assert result.headers[REQUEST_ID_HEADER]


def test_catalog_lists_exactly_one_loaded_member(server: ServerAddress) -> None:
    assert model_names(server) == [LATEST, MEMBER]
    assert "jev-latest" not in model_names(server)
    assert "stim-9b" not in model_names(server)


def test_model_cards_carry_the_documented_fields(server: ServerAddress) -> None:
    result = get_models(server)

    assert result.status == 200
    for card in result.body["models"]:
        assert set(card) == CARD_FIELDS
        assert card["description"].isascii() and card["description"]
        assert ISO_DATE.fullmatch(card["release_date"])
        date.fromisoformat(card["release_date"])
        assert isinstance(card["max_state_tokens"], int)
        assert isinstance(card["validated_context"], int)


def test_model_cards_report_the_serving_state_limit() -> None:
    with running_server(create_app(StubDecisionEngine(), max_state=3456)) as address:
        result = get_models(address)

    assert result.status == 200
    assert [card["max_state_tokens"] for card in result.body["models"]] == [3456, 3456]


def test_model_cards_report_the_loaded_member_metadata() -> None:
    engine = StubDecisionEngine(
        member_name="stim-0.8b",
        catalog_metadata=MemberMetadata(
            description="Small Stim member for CPU smoke tests.",
            release_date=date(2026, 9, 15),
            validated_context=4096,
        ),
    )

    with running_server(create_app(engine, max_state=8192)) as address:
        result = get_models(address)

    assert result.status == 200
    latest, member = result.body["models"]
    assert member == {
        "name": "stim-0.8b",
        "description": "Small Stim member for CPU smoke tests.",
        "release_date": "2026-09-15",
        "max_state_tokens": 8192,
        "validated_context": 4096,
    }
    assert latest["release_date"] == "2026-09-15"
    assert latest["max_state_tokens"] == 8192
    assert latest["validated_context"] == 4096


def test_official_sync_sdk_parses_the_model_list(server: ServerAddress) -> None:
    with TypeSafeClient(api_key="test-key", base_url=server.base_url) as client:
        models = client.models.list()

    assert [model.name for model in models.models] == [LATEST, MEMBER]
    for model in models.models:
        assert model.description
        assert ISO_DATE.fullmatch(model.release_date)
        date.fromisoformat(model.release_date)


def test_official_async_sdk_parses_the_model_list(server: ServerAddress) -> None:
    async def call() -> Any:
        async with AsyncTypeSafeClient(api_key="test-key", base_url=server.base_url) as client:
            return await client.models.list()

    models = asyncio.run(call())

    assert [model.name for model in models.models] == [LATEST, MEMBER]


def test_omitted_model_uses_the_latest_alias_and_reports_the_member(
    server: ServerAddress,
) -> None:
    result = post(server, PAYLOAD)

    assert result.status == 200
    assert result.body["model"] == MEMBER


@pytest.mark.parametrize("model", [LATEST, "jev-latest", MEMBER])
def test_alias_requests_always_report_the_loaded_member(server: ServerAddress, model: str) -> None:
    result = post(server, {**PAYLOAD, "model": model})

    assert result.status == 200
    assert result.body["model"] == MEMBER


def test_jev_latest_is_accepted_but_never_reported_or_advertised(
    server: ServerAddress,
) -> None:
    result = post(server, {**PAYLOAD, "model": "jev-latest"})

    assert result.status == 200
    assert result.body["model"] == MEMBER
    assert "jev-latest" not in model_names(server)


def test_unconfigured_official_sdk_uses_jev_latest_and_gets_the_member(
    server: ServerAddress,
) -> None:
    with TypeSafeClient(api_key="test-key", base_url=server.base_url) as client:
        response = client.system_one(
            state=STATE,
            questions={"billing": Noul(instructions="Is this about billing?")},
        )

    assert response.model == MEMBER


def test_unknown_model_returns_model_not_found(server: ServerAddress) -> None:
    result = post(server, {**PAYLOAD, "model": "stim-9b"})

    assert result.status == 404
    assert result.body["code"] == "model_not_found"
    assert "stim-9b" in result.body["detail"]
    assert "latency_ms" not in result.body


def test_models_endpoint_rejects_other_methods(server: ServerAddress) -> None:
    result = request(server, "POST", "/v1/models", b"{}")

    assert result.status == 405
    assert result.body["code"] == "method_not_allowed"
    assert result.headers["allow"] == "GET"
    assert result.headers[REQUEST_ID_HEADER]


def test_models_endpoint_echoes_a_safe_request_id(server: ServerAddress) -> None:
    result = get_models(server, headers={REQUEST_ID_HEADER: "models-1"})

    assert result.headers[REQUEST_ID_HEADER] == "models-1"


def test_model_identity_surfaces_are_registered_in_the_compatibility_matrix() -> None:
    matrix: dict[str, Any] = json.loads(
        (REPO_ROOT / "docs" / "compatibility.json").read_text(encoding="utf-8")
    )
    entries = {entry["id"]: entry for entry in matrix["entries"]}

    expected = {
        "endpoint.models": (
            "supported",
            "tests/test_models.py::test_models_endpoint_lists_the_latest_alias_and_loaded_member",
        ),
        "endpoint.models.sdk_sync": (
            "supported",
            "tests/test_models.py::test_official_sync_sdk_parses_the_model_list",
        ),
        "endpoint.models.sdk_async": (
            "supported",
            "tests/test_models.py::test_official_async_sdk_parses_the_model_list",
        ),
        "model.default": (
            "supported",
            "tests/test_models.py::test_omitted_model_uses_the_latest_alias_and_reports_the_member",
        ),
        "model.alias.stim-latest": (
            "supported",
            "tests/test_models.py::test_alias_requests_always_report_the_loaded_member",
        ),
        "model.alias.jev-latest": (
            "supported",
            "tests/test_models.py::test_unconfigured_official_sdk_uses_jev_latest_and_gets_the_member",
        ),
        "model.response_member": (
            "supported",
            "tests/test_models.py::test_alias_requests_always_report_the_loaded_member",
        ),
        "model.unknown": (
            "supported",
            "tests/test_models.py::test_unknown_model_returns_model_not_found",
        ),
        "models.catalog.single_member": (
            "supported",
            "tests/test_models.py::test_catalog_lists_exactly_one_loaded_member",
        ),
        "models.card.fields": (
            "supported",
            "tests/test_models.py::test_model_cards_carry_the_documented_fields",
        ),
        "models.card.max_state_tokens": (
            "extension",
            "tests/test_models.py::test_model_cards_report_the_serving_state_limit",
        ),
        "models.card.validated_context": (
            "extension",
            "tests/test_models.py::test_model_cards_report_the_loaded_member_metadata",
        ),
    }
    for entry_id, (status, test_id) in expected.items():
        entry = entries[entry_id]
        assert entry["status"] == status
        assert entry["expected_behavior"]
        assert entry["test_id"] == test_id
