# stim-serve

The service layer for Stim: the `DecisionEngine` implementation, the FastAPI
application, and the `stim-serve` command-line entry point.

This distribution depends on `stim-systemone`, `stim-ai`, FastAPI, and uvicorn. Installing
it is the serve-only installation; the training and evaluation extras of `stim-ai` are
not pulled in.

`create_app(engine=None, *, max_state=8192, max_context=None)` builds the FastAPI
application around a `DecisionEngine`. The protocol layer's ASGI application is mounted
at the root, so the service behaves the same whether it is run through FastAPI or bare
ASGI. FastAPI's generated documentation routes are disabled so that every response comes
from the protocol layer. With no engine supplied, the deterministic `StubDecisionEngine`
answers noul, choice, and score questions and exposes catalog metadata for
`GET /v1/models`. `max_context` is the optional backbone window: when set, state plus any
single question branch must fit inside it, or the request is refused with
`400 max_tokens_exceeded`.
