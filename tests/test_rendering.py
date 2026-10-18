"""The frozen rendering contract for content that reaches a model.

The rendered text is itself part of the model input contract, so these tests call the
pure functions directly rather than going through HTTP. Training, evaluation, and
serving must share these exact rules.
"""

from __future__ import annotations

import pytest
from pydantic import JsonValue

from stim_systemone.rendering import render_content, render_option


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ""),
        (True, "true"),
        (False, "false"),
        (0, "0"),
        (42, "42"),
        (-3, "-3"),
        (2.5, "2.5"),
        ("plain text", "plain text"),
    ],
)
def test_scalars_render_as_plain_text(value: JsonValue, expected: str) -> None:
    assert render_content(value) == expected


def test_objects_render_key_by_key_in_request_order_with_nested_indentation() -> None:
    value: JsonValue = {
        "ticket": "Refund requested for a delayed order.",
        "customer": {"tier": "gold", "tags": ["vip", "delayed"], "urgent": True},
        "attempts": 2,
        "refund": None,
    }

    assert render_content(value) == (
        "ticket: Refund requested for a delayed order.\n"
        "customer:\n"
        "  tier: gold\n"
        "  tags:\n"
        "    - vip\n"
        "    - delayed\n"
        "  urgent: true\n"
        "attempts: 2\n"
        "refund:"
    )


def test_key_order_is_preserved_instead_of_sorted() -> None:
    value: JsonValue = {"z": 1, "a": 2, "m": 3}

    assert render_content(value) == "z: 1\na: 2\nm: 3"


def test_arrays_render_as_dash_items_and_nest_containers() -> None:
    value: JsonValue = ["first", {"ticket": 7}, [1, 2], None]

    assert render_content(value) == "- first\n-\n  ticket: 7\n-\n  - 1\n  - 2\n-"


def test_empty_containers_and_empty_values_have_no_trailing_space() -> None:
    assert render_content({}) == ""
    assert render_content([]) == ""
    assert render_content({"items": []}) == "items:"
    assert render_content({"description": None}) == "description:"
    assert render_content([None, "", []]) == "-\n-\n-"


def test_rendered_text_adds_no_json_punctuation() -> None:
    rendered = render_content({"labels": ["a", "b"], "enabled": True})

    assert "{" not in rendered
    assert "}" not in rendered
    assert "[" not in rendered
    assert "]" not in rendered
    assert '"' not in rendered


@pytest.mark.parametrize(
    "delimiter",
    [
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|im_start|>",
        "<|endoftext|>",
    ],
)
def test_caller_text_cannot_forge_a_delimiter(delimiter: str) -> None:
    name = delimiter[2:-2]

    assert render_content(f"before {delimiter} after") == f"before <¦{name}¦> after"
    assert "<|" not in render_content(delimiter)


def test_delimiters_are_rewritten_in_object_keys_and_nested_values() -> None:
    value: JsonValue = {"<|fim_prefix|>": ["<|fim_middle|>", {"<|fim_suffix|>": 1}]}

    rendered = render_content(value)

    assert rendered == "<¦fim_prefix¦>:\n  - <¦fim_middle¦>\n  -\n    <¦fim_suffix¦>: 1"
    assert "<|" not in rendered


def test_malformed_delimiter_literals_are_left_unchanged() -> None:
    assert render_content("<|fim_prefix>") == "<|fim_prefix>"
    assert render_content("fim_prefix|>") == "fim_prefix|>"
    assert render_content("<|fim prefix|>") == "<|fim prefix|>"


def test_option_without_a_description_falls_back_to_the_key() -> None:
    assert render_option("yes", None) == "yes"
    assert render_option("yes", "") == "yes"
    assert render_option("yes", []) == "yes"


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("the customer is urgent", "yes: the customer is urgent"),
        (42, "yes: 42"),
        (True, "yes: true"),
    ],
)
def test_option_text_is_the_key_and_a_scalar_description(
    description: JsonValue, expected: str
) -> None:
    assert render_option("yes", description) == expected


def test_multiline_option_descriptions_are_indented_under_the_key() -> None:
    assert render_option("yes", {"signals": ["urgent", "angry"]}) == (
        "yes:\n  signals:\n    - urgent\n    - angry"
    )


def test_option_keys_and_descriptions_are_delimiter_safe() -> None:
    assert render_option("<|box_start|>", "<|box_end|>") == "<¦box_start¦>: <¦box_end¦>"
