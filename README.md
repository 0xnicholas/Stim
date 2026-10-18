# Stim

Stim is an open family of small decision models that speaks TypeSafe's System One
protocol, with its weights, corpus, recipe, and evaluation published together as one
record.

This repository currently contains the service-foundation scaffold: three Python
distributions with a strict one-way dependency direction.

## Packages

| Distribution | Import | Role | Depends on |
| --- | --- | --- | --- |
| `stim-systemone` | `stim_systemone` | System One wire protocol, rendering, derived quantities, ASGI router | pydantic |
| `stim-ai` | `stim_ai` | Shared model, artifact loading, root CLI | `stim-systemone`, torch, transformers, peft |
| `stim-serve` | `stim_serve` | Decision engine, FastAPI app, serve CLI | `stim-systemone`, `stim-ai`, FastAPI, uvicorn |

The protocol layer never depends on torch or FastAPI. The shared core never depends on
the service layer. Training and evaluation dependencies are optional extras of
`stim-ai`; installing `stim-serve` is the serve-only installation.

## Development

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

The compatibility suite is pinned to the official `typesafe-sdk==0.7.5`; the SDK is a
development dependency, not a runtime dependency of any distribution.

## Build

```sh
uv build --all-packages
```

This writes one wheel and one source distribution per package to `dist/`.

## Licence

Apache-2.0. See `LICENSE` and `NOTICE`.
