"""ASGI application factory for serving Stim models."""

from __future__ import annotations

from collections.abc import Collection

from fastapi import FastAPI

from stim_serve import __version__
from stim_systemone import DecisionEngine, StubDecisionEngine, SystemOneApp
from stim_systemone.protocol import DEFAULT_MAX_STATE, DEFAULT_MODEL_ALIASES


def create_app(
    engine: DecisionEngine | None = None,
    *,
    max_state: int = DEFAULT_MAX_STATE,
    model_aliases: Collection[str] = DEFAULT_MODEL_ALIASES,
) -> FastAPI:
    """Build the service app around a decision engine.

    The protocol layer's ASGI app is mounted at the root and owns every route, so the
    service treats unknown paths, methods, and errors the same way. FastAPI's generated
    documentation routes are disabled: they are outside the System One surface and would
    answer without the protocol's request-id header.
    """
    app = FastAPI(
        title="Stim",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.mount(
        "/",
        SystemOneApp(
            engine if engine is not None else StubDecisionEngine(),
            max_state=max_state,
            model_aliases=model_aliases,
        ),
    )
    return app
