"""ASGI application factory for serving Stim models."""

from __future__ import annotations

from fastapi import FastAPI

from stim_serve import __version__
from stim_systemone import (
    DecisionEngine,
    Limiter,
    OverloadCheck,
    StubDecisionEngine,
    SystemOneApp,
)
from stim_systemone.protocol import DEFAULT_MAX_STATE


def create_app(
    engine: DecisionEngine | None = None,
    *,
    max_state: int = DEFAULT_MAX_STATE,
    max_context: int | None = None,
    api_key: str | None = None,
    limiter: Limiter | None = None,
    is_overloaded: OverloadCheck | None = None,
) -> FastAPI:
    """Build the service app around a decision engine.

    The protocol layer's ASGI app is mounted at the root and owns every route, so the
    service treats unknown paths, methods, and errors the same way. FastAPI's generated
    documentation routes are disabled: they are outside the System One surface and would
    answer without the protocol's request-id header.

    ``api_key`` enables Bearer authentication on every ``/v1/*`` route. ``limiter`` is an
    optional hook that returns the seconds to retry after when a request should be
    refused; no limiter allows every request. ``is_overloaded`` reports whether the
    service is draining or over capacity. ``/healthz`` and ``/readyz`` bypass all three.
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
            max_context=max_context,
            api_key=api_key,
            limiter=limiter,
            is_overloaded=is_overloaded,
        ),
    )
    return app
