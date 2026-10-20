# stim-serve

The service layer for Stim: the `DecisionEngine` implementation, the FastAPI
application, and the `stim-serve` command-line entry point.

This distribution depends on `stim-systemone`, `stim-ai`, FastAPI, and uvicorn. Installing
it is the serve-only installation; the training and evaluation extras of `stim-ai` are
not pulled in.

`create_app(engine=None, *, max_state=8192, max_context=None, api_key=None,
limiter=None, is_overloaded=None)` builds the FastAPI application around a
`DecisionEngine`. The protocol layer's ASGI application is mounted at the root, so the
service behaves the same whether it is run through FastAPI or bare ASGI. FastAPI's
generated documentation routes are disabled so that every response comes from the protocol
layer. With no engine supplied, the deterministic `StubDecisionEngine` answers noul,
choice, and score questions and exposes catalog metadata for `GET /v1/models`.
`max_context` is the optional backbone window: when set, state plus any single question
branch must fit inside it, or the request is refused with `400 max_tokens_exceeded`.

`api_key` enables Bearer authentication on every `/v1/*` route; `GET /healthz` and
`GET /readyz` stay open. `limiter` is an optional callable receiving the ASGI scope; it
returns `None` to allow a request or the number of seconds to retry after to refuse it
with `429 rate_limited` and `Retry-After`. `is_overloaded` is an optional callable
returning whether the service is draining or over capacity, which refuses `/v1/*` with
`529 overloaded`. With no hooks the service allows every request.
