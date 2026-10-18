# stim-systemone

The System One wire protocol layer for Stim: request and response models, rendering,
derived quantities, error mapping, middleware, and the ASGI router.

This distribution depends only on `pydantic` and the Python standard library. It must
not depend on torch, transformers, FastAPI, or uvicorn; the service layer integrates it
with those libraries.

The `stim-systemone` console entry point reports the installed protocol-layer version.

## Implemented surface

The current vertical slice serves `POST /v1/systemone` for `noul`, `choice`, and `score`
questions through `SystemOneApp`, a pure ASGI application. A `DecisionEngine` receives
the rendered state and normalized questions and returns raw option scores plus an input
token count; every wire concern stays in this package.

- A noul answer is exactly `{"type": "noul", "noul": p}`, where `p` is the probability
  of yes. It never carries `confidence`.
- Noul maps to the options `no` and `yes`. Optional `criteria.true` and `criteria.false`
  descriptions render into the option text as `yes: <description>` and
  `no: <description>`.
- Choice `criteria` is a required, non-empty object of 1–255 option names to optional
  descriptions. Request order is preserved, and names must be non-empty. A null or empty
  description falls back to the bare name; any other description renders through
  `render_option`, as `name: <description>` on one line or with a multi-line
  description indented under `name:`.
- A choice answer is exactly `{"type": "choice", "choice": ..., "probabilities": {...},
  "confidence": ...}`. Probabilities are keyed by option name in request order and
  rounded to four decimal places after normalization. `choice` is the argmax of the
  reported probabilities. `confidence` is `(p_max − 1/K) / (1 − 1/K)`, measured on the
  normalized distribution before probability rounding and then rounded to four decimal
  places; the single-option answer is pinned to probability 1.0 and confidence 1.0.
- Score `criteria` is a required, ordered array of 2–10 level descriptions. Criteria
  order is preserved and each level renders through `render_option` under its index key,
  as `0: <description>`, `1: <description>`, and so on.
- A score answer is exactly `{"type": "score", "score": ..., "legend": {...},
  "probabilities": {...}, "confidence": ...}`. Probabilities are keyed by the level index
  as a string, in criteria order, and rounded to four decimal places after
  normalization. `legend` maps each index key to its rendered level text. `score` is the
  probability-weighted mean `Σ i·pᵢ` over the reported probabilities, so it may fall
  between two levels. `confidence` is `max(0, 1 − Σ pᵢ|i − mode| / D)` with
  `D = (1/L)Σ|i − (L−1)/2|` and `mode` the first highest reported probability, both
  measured on the reported probabilities and rounded to four decimal places.
- Option scores are normalized before being rounded to four decimal places. A zero sum
  falls back to a uniform distribution.
- Every response, including errors, carries a non-empty `x-typesafe-request-id`. A
  caller-supplied id is echoed when it is 1–128 characters from `A-Z`, `a-z`, `0-9`,
  `.`, `_`, `~`, and `-`; otherwise a UUID hex is generated.
- `usage.input_tokens` counts the rendered state plus each question branch, and
  `usage.output_tokens` counts the serialized answers. Both are integer workload counts,
  **not billing units**: this slice has no generation step and no per-token charge.
- `latency_ms` appears only on successful responses and is rounded to one decimal place.
  It is the only extra field a success response carries.
- `state` is limited to `max_state` tokens (default 8192; the future CLI flag will be
  `--max-state`). An overlong state returns `400 max_tokens_exceeded` with the actual and
  maximum token counts.
- Error bodies are always `{"detail": "...", "code": "..."}` and never include
  `latency_ms`. Invalid requests return `422 invalid_request`, with `detail` naming the
  offending field. Image and media `assets` are rejected with `415
  unsupported_media_type`; this slice is text-only.

`StubDecisionEngine` is a deterministic engine used by the protocol tests; it implements
the same contract as a model-backed engine and never handles HTTP or wire types.

## Content rendering

`stim_systemone.rendering` is the single renderer for every piece of caller content:
state, instructions, noul criteria, choice option names and descriptions, and score
level texts. It renders JSON values as plain text
without JSON punctuation, preserves object key order, and rewrites `<|name|>`
special-token literals to `<¦name¦>` so caller text cannot forge a structural delimiter.
The rendered form is frozen with the artifact and is shared by training, evaluation, and
serving.

The exact rules and examples are in
[the artifact contract](../../docs/artifact-contract.md).
