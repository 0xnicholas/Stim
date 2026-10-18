# stim-systemone

The System One wire protocol layer for Stim: request and response models, rendering,
derived quantities, error mapping, middleware, and the ASGI router.

This distribution depends only on `pydantic` and the Python standard library. It must
not depend on torch, transformers, FastAPI, or uvicorn; the service layer integrates it
with those libraries.
