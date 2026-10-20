# Stim

[![CI](https://github.com/0xnicholas/Stim/actions/workflows/ci.yml/badge.svg)](https://github.com/0xnicholas/Stim/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Status: scaffold](https://img.shields.io/badge/status-scaffold-orange.svg)](#status)

English | [简体中文](README.zh-CN.md)

Stim is an open family of small decision models built on Qwen3.5 base models and
TypeSafe's [System One](https://docs.typesafe.ai/api) protocol. A Stim model does not
write an answer: it returns a probability distribution over the caller's options. One
state, a batch of typed questions, one response.

The differentiator is not the architecture alone. Stim publishes the **weights, corpus,
recipe, and evaluation together as one reproducible record**, so an independent third
party can retrain the family and re-derive the published scores.

> **Status: service-foundation scaffold.** What exists today is the package skeleton,
> three CLI entry points, the dependency boundaries, and CI. There are no weights, no
> corpus, no training recipe, no evaluation, no protocol implementation, and no server
> yet. Sections marked **Planned** describe committed design direction, not shipped
> behaviour.

## Highlights

- **Decisions, not chat.** The entire output is a distribution over caller-defined
  options, so a probability can be thresholded, routed, or escalated to a human.
- **System One surface.** The target is `POST /v1/systemone` for `noul`, `choice`, and
  `score` questions plus `GET /v1/models`, verified through TypeSafe's official Python
  SDK.
- **Qwen3.5 inside.** The family is designed around Qwen3.5 Base checkpoints: open
  weights, a permissive licence, multiple small sizes, and standard `transformers` /
  `peft` tooling. [Why Qwen](#why-qwen) explains the choice.
- **One record, four artifacts.** Weights, corpus, training recipe, and evaluation are
  published as one versioned unit rather than four unrelated releases.
- **Permissive-only corpus.** Training data is restricted to sources whose licences
  permit unconditional redistribution; share-alike and unknown-licence sources are
  excluded and synthetic data fills the gap.
- **Dual-track evaluation.** A public comparability track measures Stim against the
  ecosystem; a separate verification track is published in full so a third party can
  rerun it.
- **Small enough to retrain.** Family size is bounded by the *reproducer cost* — the
  GPU-hours a stranger spends to reach the same magnitude from the published recipe.
- **A Chinese line is planned.** The first language variant is `zh-v1.0.0`, independently
  versioned from the English family, sharing the recipe skeleton but not its corpus or
  evaluation.

## Status

| Area | State |
| --- | --- |
| uv workspace with three installable distributions | Implemented |
| One-way dependency boundaries, packaging checks, CI | Implemented |
| Console entry points (`stim`, `stim-serve`, `stim-systemone`) | Placeholder: help and version only |
| System One request/response models, rendering, router | Not implemented |
| Model core: encoder, pointer head, artifact loading | Not implemented |
| `DecisionEngine`, FastAPI application, routes | Not implemented |
| Weights, corpus, recipe, evaluation | Not published |

The work is ordered bottom-up: protocol layer, then model core, then service, then the
first family release. The scaffold enforces that order so the protocol layer can never
quietly acquire torch or FastAPI, and the service layer can never leak into the model
core.

## Why Qwen

Qwen is not incidental packaging. Stim's base model determines what can be redistributed,
which sizes are affordable to retrain, and whether the family can support more than one
language. The design targets Qwen3.5 Base checkpoints because they satisfy five
requirements at once:

| Requirement | How Qwen3.5 meets it |
| --- | --- |
| Licence clarity | Qwen3.5 Base weights are published under Apache-2.0. Adapters, fine-tuned weights, and the corpus can be redistributed without a field-of-use restriction. |
| A neutral starting point | The `-Base` checkpoints are pretrained, not post-trained on a vendor instruction corpus. The recipe does not silently inherit an undisclosed SFT stage. |
| A family, not one size | Base checkpoints exist at 0.8B, 4B, and 9B, which is the range Stim's family design targets. Member size can follow reproducer cost instead of licence renegotiation. |
| Standard tooling | Qwen is supported by `transformers`, `peft`, and the usual LoRA/quantization ecosystem, so the training and serving paths stay portable. |
| Multilingual coverage | Qwen's multilingual pretraining makes a Chinese line feasible without changing the architecture. The line still needs its own corpus, calibration, and evaluation. |

Two boundaries matter:

- **Qwen is the base model, not automatically the teacher.** Teachers that generate
  training labels are open-weight models pinned by revision; closed models may judge or
  filter data but never supply the label. The base model and the teacher are separate
  roles recorded per artifact.
- **The base licence is not the corpus licence.** Every corpus source is tracked
  separately. Qwen's Apache-2.0 terms do not make a third-party dataset redistributable.

Qwen3.5's hybrid attention stack also shapes the serving design: some layers are
recurrent and ignore attention masks, so questions cannot always be packed into one
block-causal sequence. Stim's planned design runs each question as its own row and reuses
the state cache, which makes question isolation exact by construction.

## Planned model family

Stim has not fixed its member list. The first family release will publish the exact
members, parameter counts, adapters, temperatures, validated context, and scores. The
design is:

- each **member** is one size in the family, with its own checkpoint or adapter and its
  own fitted temperature;
- all members share one corpus and recipe skeleton;
- a **family release** versions every member together; a member can take a patch release
  only when the shared inputs are unchanged;
- a **language variant** is versioned independently, starting with `zh-v1.0.0`.

The architecture is a small Jev-style decision model:

```text
state ──► rendered once ──► Qwen backbone (prefill only)
                                │
question 1 ──► branch ──► pointer head ──► distribution over options
question 2 ──► branch ──► pointer head ──► distribution over options
question 3 ──► branch ──► pointer head ──► distribution over options
```

There is no token generation. The backbone runs prefill-only; a pointer head scores each
option's closing hidden state against the question's decide state; a softmax turns those
scores into the answer distribution. Questions in one request share the state but are
answered independently. A single scalar temperature per member, fitted on a held-out
pool, calibrates the distribution without changing which option wins.

## System One compatibility

Stim's compatibility claim is scoped, not absolute. The committed target is:

| Surface | Target |
| --- | --- |
| `POST /v1/systemone` | `noul`, `choice`, `score` |
| `GET /v1/models` | Model cards for accepted names |
| Official SDK | `typesafe-sdk==0.7.5` |
| `box` questions | Out of scope |
| Images and multipart input | Out of scope |
| 262k-token inputs | Out of scope |

`box` is excluded because it is not a routing gap: it needs an image-capable state, an
image backbone, and a coordinate head. Claiming "fully compatible" while returning a
422 for `box` would make the claim meaningless. The README, the SDK conformance tests,
and the per-field compatibility matrix are meant to say the same thing.

Forward-compatible clients are safe by default. Unknown request fields — including an
SDK `extra_body.temperature` — and the `TypeSafe-Danger-Zone-Long-Context` header are
accepted and ignored; each ignored field is logged by path, never by value, and the
configured state limit still applies. Questions within one request are answered
independently, and repeated or re-batched requests agree within the published 0.011
probability tolerance. These rows live in
[`docs/compatibility.json`](docs/compatibility.json).

The derived quantities follow the published reference formulas rather than model output:

- **`noul`** returns the probability of "yes" and no `confidence` field.
- **`choice`** confidence is `(p_max − 1/K) / (1 − 1/K)` for `K` options.
- **`score`** confidence is `max(0, 1 − Σ p_i |i − m| / MAD_uniform)`, where `m` is the
  modal level and `MAD_uniform` is the uniform distribution's mean absolute deviation
  about the centre.

An unconfigured TypeSafe client may connect with the SDK's default model name. Stim
accepts that name as a **compatibility alias** only: it is never listed as a family
member and never reported as the serving model.

## The reproducible record

The publication unit is one record, not one model file:

| Artifact | What is published | Acceptance bar |
| --- | --- | --- |
| Weights | Checkpoints or adapters, configs, fitted temperatures | Downloadable and licence-clear |
| Corpus | Records, per-source manifests, provenance, licences | Downloadable as a first-class artifact |
| Recipe | Scripts, hyperparameters, compute envelope, seeds | A stranger can retrain to the same magnitude |
| Evaluation | Item-level data, manifests, code, reference scores | A third party can rerun and reproduce the scores |

Regenerating the corpus from its generation prompts is transparency, but it is not part
of the acceptance bar. The bar is a downloadable corpus, a retrainable recipe, and a
rerunnable evaluation.

Evaluation runs on two tracks:

| Track | Purpose | What is published |
| --- | --- | --- |
| Public comparability | Compare Stim with Kev, Decision Index, and JevBench | Existing public frozen suites; sealed or private portions are not rerunnable |
| Verification | Carry the reproducibility promise | Item-level data, provenance, manifests, evaluation code, reference scores |

Reported numbers use chance-corrected skill, so 0 is random guessing and 100 is perfect,
with each suite's recorded baseline. Calibration is reported separately. A label-shuffled
control — trained on the same corpus with labels permuted within question type and option
count — shows whether a member's skill is more than memorised label marginals.

## Relationship to Jev, Kev, and AutoJev

- **Jev** is TypeSafe's hosted decision model. Stim is an independent open family that
  speaks the same protocol; it is not Jev and does not use Jev's weights.
- **Kev** (`jaredpalmer/kev`) is the open reference implementation Stim measures itself
  against and borrows conventions from: the small-model recipe skeleton, evaluation
  discipline, and the clean-room approach to compatibility. Stim contains no Kev code,
  corpus, or runtime dependency; `NOTICE` records the design reference.
- **AutoJev** is another open reproduction of a Jev-style model. It is evidence that
  full-weight SFT on pooled public data is table stakes, not a differentiator.

## Quick start

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/0xnicholas/Stim.git
cd Stim

uv sync
uv run pytest
```

`uv sync` installs the three workspace packages and the development tools. Add
`--all-extras` to include the training and evaluation dependencies.

PyTorch does not publish Intel macOS wheels. On that platform `stim-ai` skips torch and
peft so the packages remain importable, but model execution is unavailable.

### Command-line entry points

| Command | Distribution | Current behaviour |
| --- | --- | --- |
| `stim` | `stim-ai` | Prints help; `--version` reports the model-core version |
| `stim-serve` | `stim-serve` | Prints help; `--version` reports the service version |
| `stim-systemone` | `stim-systemone` | Prints help; `--version` reports the protocol-layer version |

```sh
uv run stim --help
uv run stim-serve --version
```

The commands are placeholders until the protocol, model, and service layers land.

## Development

Run the full gate locally:

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

The tests enforce the structure, not just behaviour:

- `stim-systemone` depends only on `pydantic` and the standard library;
- importing `stim_systemone` loads neither torch nor FastAPI;
- `stim-ai` keeps training and evaluation dependencies optional;
- `stim-serve` does not pull in the training or evaluation extras;
- the SDK baseline stays pinned at `typesafe-sdk==0.7.5`;
- package import names agree with distribution names;
- the licence and NOTICE credit the required sources.

CI runs the same commands on every pull request, builds all packages, and verifies that
the three wheels are importable.

## Build

The workspace root is virtual, so build every member at once:

```sh
uv build --all-packages
```

This writes one wheel and one source distribution per package to `dist/`.

## Repository layout

```text
packages/
  stim-systemone/   System One wire protocol, rendering, derived quantities, ASGI router
  stim-ai/          shared model core, artifact loading, root CLI
  stim-serve/       decision engine, FastAPI app, serve CLI
tests/              packaging, dependency, and CLI contract checks
docs/               research records and agent guidance
CONTEXT.md          domain glossary and project vocabulary
```

## Documentation

- `CONTEXT.md` defines the project vocabulary: System One, members, family releases,
  language variants, the compatibility surface, the reproducibility bar, and the
  evaluation tracks.
- `docs/research/` records the primary-source research behind those decisions,
  including the Kev comparison.

## Licence

Apache-2.0. See `LICENSE` and `NOTICE`. The NOTICE file credits the Qwen base models and
the Kev project, which informs Stim's design.
