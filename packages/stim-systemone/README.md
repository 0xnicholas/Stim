# stim-systemone

The System One wire protocol layer for Stim: request and response models, rendering,
derived quantities, error mapping, middleware, and the ASGI router.

This distribution depends only on `pydantic` and the Python standard library. It must
not depend on torch, transformers, FastAPI, or uvicorn; the service layer integrates it
with those libraries.

The `stim-systemone` console entry point reports the installed protocol-layer version.

## Implemented surface

The current vertical slice serves `POST /v1/systemone` for `noul` questions through
`SystemOneApp`, a pure ASGI application. A `DecisionEngine` receives the rendered state
and normalized questions and returns raw option scores plus an input token count; every
wire concern stays in this package.

- A noul answer is exactly `{"type": "noul", "noul": p}`, where `p` is the probability
  of yes. It never carries `confidence`.
- Noul maps to the options `no` and `yes`. Optional `criteria.true` and `criteria.false`
  descriptions render into the option text as `yes: <description>` and
  `no: <description>`.
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
state, instructions, noul criteria, and (once those question types land) choice option
descriptions and score level texts. It renders JSON values as plain text without JSON
punctuation, preserves object key order, and rewrites `<|name|>` special-token literals
to `<¦name¦>` so caller text cannot forge a structural delimiter. The rendered form is
frozen with the artifact and is shared by training, evaluation, and serving.

The exact rules and examples are in
[the artifact contract](../../docs/artifact-contract.md).
