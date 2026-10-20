# stim-systemone

The System One wire protocol layer for Stim: request and response models, rendering,
derived quantities, error mapping, middleware, and the ASGI router.

This distribution depends only on `pydantic` and the Python standard library. It must
not depend on torch, transformers, FastAPI, or uvicorn; the service layer integrates it
with those libraries.

The `stim-systemone` console entry point reports the installed protocol-layer version.

## Implemented surface

The current vertical slice serves `POST /v1/systemone` for `noul`, `choice`, and `score`
questions, plus `GET /v1/models`, through `SystemOneApp`, a pure ASGI application. A
`DecisionEngine` receives the rendered state and normalized questions and returns raw
option scores plus an input token count; it also exposes the loaded member's name and
catalog metadata. Every wire concern stays in this package.

- An omitted `model` defaults to `stim-latest`. `stim-latest` and `jev-latest` are
  aliases for the loaded member; every successful `POST /v1/systemone` response reports
  that member's name, never an alias. Any other name returns `404 model_not_found`.
- `GET /v1/models` returns exactly `stim-latest` and the one loaded member. `jev-latest`
  is never advertised, and a process never lists a member it has not loaded.
- Every card carries `name`, an English `description`, an ISO 8601 `release_date`, and
  the registered Stim extension fields `max_state_tokens` (the configured serving limit)
  and `validated_context` (from the member's metadata). The official TypeSafe SDK ignores
  the extension fields when it parses the list.

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
- `latency_ms` appears only on successful `POST /v1/systemone` responses and is rounded
  to one decimal place. It is the only extra field those responses carry; the models
  catalog is exactly `{models: [...]}`.
- `state` is limited to `max_state` tokens (default 8192; the future CLI flag will be
  `--max-state`). An overlong state returns `400 max_tokens_exceeded` with the actual and
  maximum token counts.
- When `max_context` is configured (the backbone window), the rendered state plus any
  single question branch is limited to that many tokens. Exceeding either limit returns
  `400 max_tokens_exceeded`, again with the actual and maximum counts.
- Request bodies are limited to 32 MiB. A `Content-Length` over the limit is refused
  before the body is read; a body without `Content-Length` is counted as it streams in.
- Only `application/json` (or a `+json` media type) with no content encoding or
  `Content-Encoding: identity` is accepted. Multipart bodies, any `assets` field, and
  other content types return `415 unsupported_media_type`; other encodings return `415
  unsupported_content_encoding`.
- `permute` and `separate` are not registered under `/v1/systemone`; both return 404.
- Error bodies are always `{"detail": "...", "code": "..."}`, carry a request id, and
  never include `latency_ms` or caller content (state, instructions, or criteria).
  Invalid requests return `422 invalid_request`, with `detail` naming the offending
  field.

| Status | `code` | Trigger |
| --- | --- | --- |
| 400 | `max_tokens_exceeded` | state or one state-plus-question row over the limit |
| 404 | `not_found` | unknown path, including unregistered extension routes |
| 404 | `model_not_found` | unknown model name |
| 405 | `method_not_allowed` | any method other than `POST` on `/v1/systemone` or `GET` on `/v1/models` |
| 413 | `payload_too_large` | body over 32 MiB |
| 415 | `unsupported_media_type` | non-JSON content type, multipart, or `assets` |
| 415 | `unsupported_content_encoding` | content encoding other than identity |
| 422 | `invalid_request` | schema, field, question type, or count errors |
| 500 | `internal_error` | unexpected failure; no internal detail is returned |

`StubDecisionEngine` is a deterministic engine used by the protocol tests; it implements
the same contract as a model-backed engine and never handles HTTP or wire types.

## Forward compatibility and question isolation

Unknown request fields are ignored at every level of the request schema: the body
(including SDK `extra_body` keys such as `temperature`), each question object, and
`noul.criteria`. Ignoring never changes an answer or `usage`. Each ignored field is
logged as a warning on the `stim_systemone` logger by dotted path — for example
`questions.billing.unexpected` — and field values are never logged. Non-printable
characters in caller-chosen path segments are escaped, so keys cannot forge log lines.
Keys inside caller content (`state`, `instructions`, choice descriptions, and score
levels) are content, not request fields, and are never reported this way.

`extra_body.temperature` is therefore accepted and ignored: the SDK merges it into the
request body as a top-level `temperature` field, and probabilities always come from the
loaded member's fitted temperature.

`TypeSafe-Danger-Zone-Long-Context` is accepted and ignored. It does not enable a 262k
limit: `state` is still validated against the configured `max_state` (and, when
configured, `max_context`).

Questions within one request are answered independently. With a deterministic engine, a
question answered alone and answered beside sibling questions produces identical
answers; repeated requests and different cross-request batch compositions likewise
agree exactly. Against a model-backed engine the committed tolerance is 0.011 per
probability. These rows are registered in the
[compatibility matrix](../../docs/compatibility.json).

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
