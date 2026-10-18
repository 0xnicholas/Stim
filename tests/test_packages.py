from __future__ import annotations

import importlib
import importlib.metadata
import subprocess
import sys
from pathlib import Path
from typing import Protocol, cast

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

DISTRIBUTIONS = (
    ("stim-systemone", "stim_systemone"),
    ("stim-ai", "stim_ai"),
    ("stim-serve", "stim_serve"),
)


class VersionedModule(Protocol):
    __version__: str


def core_requirements(distribution: str) -> dict[str, Requirement]:
    declared = importlib.metadata.requires(distribution) or []
    return {
        canonicalize_name(requirement.name): requirement
        for requirement in map(Requirement, declared)
        if requirement.marker is None or "extra" not in str(requirement.marker)
    }


@pytest.mark.parametrize(("distribution", "module"), DISTRIBUTIONS)
def test_distribution_and_import_name_agree(distribution: str, module: str) -> None:
    imported = cast(VersionedModule, importlib.import_module(module))
    assert imported.__version__ == importlib.metadata.version(distribution)


def test_protocol_layer_stays_free_of_model_and_server_dependencies() -> None:
    assert set(core_requirements("stim-systemone")) == {"pydantic"}


def test_shared_core_depends_on_protocol_layer() -> None:
    core = core_requirements("stim-ai")
    assert "stim-systemone" in core
    assert core["stim-systemone"].extras == set()
    assert "torch" in core


def test_shared_core_keeps_training_and_evaluation_optional() -> None:
    core = set(core_requirements("stim-ai"))
    assert {"accelerate", "datasets", "scipy"}.isdisjoint(core)

    extras = set(importlib.metadata.metadata("stim-ai").get_all("Provides-Extra") or ())
    assert {"train", "eval"} <= extras


def test_service_layer_depends_on_both_lower_layers() -> None:
    core = core_requirements("stim-serve")
    assert "stim-systemone" in core
    assert "stim-ai" in core
    assert core["stim-ai"].extras == set()
    assert {"fastapi", "uvicorn"} <= set(core)


def test_serve_install_does_not_pull_training_or_evaluation_extras() -> None:
    core = set(core_requirements("stim-serve"))
    assert {"accelerate", "datasets", "scipy"}.isdisjoint(core)


def test_typesafe_sdk_is_pinned_to_the_compatibility_baseline() -> None:
    assert importlib.metadata.version("typesafe-sdk") == "0.7.5"


@pytest.mark.parametrize(
    "code",
    [
        (
            "import sys, stim_systemone; "
            "assert 'torch' not in sys.modules; "
            "assert 'fastapi' not in sys.modules"
        ),
        "import sys, stim_ai; assert 'torch' not in sys.modules",
    ],
)
def test_light_modules_do_not_import_heavy_dependencies(code: str) -> None:
    subprocess.run([sys.executable, "-c", code], check=True)


def test_license_and_notice_credit_the_required_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    license_text = (root / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in license_text
    assert "Version 2.0" in license_text

    notice = (root / "NOTICE").read_text(encoding="utf-8")
    assert "Qwen" in notice
    assert "Kev" in notice


def test_no_kev_runtime_dependency() -> None:
    for distribution, _ in DISTRIBUTIONS:
        assert "kev" not in core_requirements(distribution)
