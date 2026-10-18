# Kev (`jaredpalmer/kev`) — research record

**Revision described:** `1f04cfa28e6a708485b28a9585010b9e436b776e` on `main` ("kev-finetune: reject a fractional score index instead of truncating it (#241)", committed 2026-10-15T22:41:06Z), observed **2026-10-16**. Line numbers and file contents below are that commit unless a different revision is named. The repo is active and committed to daily; expect drift.

**How this was read:** GitHub REST API via `gh` (repo metadata, file contents, code search, git tree) at the revision above; the official TypeSafe Python SDK and reference adapter cloned at their own HEADs; TypeSafe's official docs over HTTP. `huggingface.co` was **not reachable** from the research environment (see [Open questions](#open-questions--could-not-verify)). No secondary write-ups were used.

---

## 1. What Kev is

Kev is a family of small "Jev-style" decision models — in Stim's sense, a model whose entire output is a probability distribution over a caller's options. The repo's own description is "Jev-like family of decision models built on top of Qwen3.5/3.8 you can train and run on your own"; the README opens "Small Jev-like decision models you can train and run yourself." Both name it as *Jev-like*, never as the origin — consistent with Stim's glossary's "sibling, not the origin".

| Fact | Value | Source |
|---|---|---|
| Owner / maintainer | Jared Palmer (`@jaredpalmer`), 319 commits — 11 contributors total | [contributors API](https://api.github.com/repos/jaredpalmer/kev/contributors) |
| Other committers | `devin-ai-integration[bot]` 16, then single-digit one-offs. README: "Built with Devin." | contributors API; [README](https://github.com/jaredpalmer/kev/blob/1f04cfa28e6a708485b28a9585010b9e436b776e/README.md) |
| Created / last push | 2026-09-17T20:49:39Z / 2026-10-15T22:41:13Z | [repo API](https://api.github.com/repos/jaredpalmer/kev) |
| Archived? | `false`; `disabled: false` | repo API |
| Stars / forks / open issues | 9,223 / 622 / 34 | repo API |
| Repo license field | Apache-2.0 | repo API |
| Releases | `v0.1.0` (kev-0.5b, 2026-09-17), `kev-family` (2026-09-20, "superseded by Kev 1.0"), `kev-1.0` (2026-10-01) | [releases API](https://api.github.com/repos/jaredpalmer/kev/releases) |
| Git tags | `v0.1.0`, `research-archive-2026-09-24`, `kev-family`, `kev-1.0` | tags API |
| Languages | Python 2,207,685 B, TypeScript 88,479 B, CSS, Shell, JS, HTML | languages API |
| Package | **Not on PyPI.** README warns `pip install kev` installs an unrelated project (K.E.V. ORM) | README |

Two things about the repo's shape matter for Stim. First, it is a **research log as much as a library**: `PLAN.md` is 3,185 lines, `AGENTS.md` 463 lines, and `runs/**` carries committed `report.json`/`rows.json` for every experiment. Second, the numbers are machine-checked: `docs/claims.json` holds **1,019** claims with a verifier at `scripts/verify_claims.py`. The README says the numbers "are checked in CI"; `.github/workflows/ci.yml` at this commit runs only the listed pytest modules plus playground lint/typecheck — see [Open questions](#open-questions--could-not-verify).

CI at this commit: `uv sync --extra serve` then pytest over `tests/test_unit.py test_compare.py test_device.py test_research.py test_generators.py test_conventions.py test_documents_tools.py test_hard_v1.py test_devtools_v1.py test_breadth_v1.py test_rounds.py test_skill_scripts.py` — no weights, no server. `.github/workflows/ci.yml@1f04cfa2`.

## 2. Model family and architecture

Kev 1.0 is four sizes released together on 2026-10-01, each at Hub revision tag `v1.0` ([`docs/releases/kev-1.0.md`](https://github.com/jaredpalmer/kev/blob/1f04cfa28e6a708485b28a9585010b9e436b776e/docs/releases/kev-1.0.md)):

| Model | Base (license) | Weights form | Temperature | Validated context | breadth-v1 test index | transfer-v4 test acc / Brier |
|---|---|---|---|---|---|---|
| Kev-0.8B | Qwen3.5-0.8B-Base (Apache-2.0) | LoRA adapter + head | 2.35 | 8,192 | 23.3 | 0.697 / 0.397 |
| Kev-4B | Qwen3.5-4B-Base (Apache-2.0) | LoRA adapter + head | 2.41 | 8,192 | 38.0 | 0.838 / 0.224 |
| Kev-9B | Qwen3.5-9B-Base (Apache-2.0) | LoRA adapter + head | 2.19 | 8,192 | 41.0 | 0.852 / 0.199 |
| Kev-27B | Qwen3.8-27B, post-trained (Apache-2.0) | **full bf16 weights**, 51.3 GB | 1.32 | 65,536 | 52.3 | 0.889 / 0.154 |
| Jev (reference) | — | hosted | — | — | 54.0 | 0.857 / 0.211 (development) |

Architecture, `kev/model.py:205-223`:

```python
class PointerHead(nn.Module):
    def __init__(self, d, dp=256): self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp); self.scale = 1/math.sqrt(dp)
    def forward(self, h_decide, h_opts):  # [d], [K,d] -> logits [K]
        z = (self.k(h_opts) @ self.q(h_decide)) * self.scale
        return z if self.training or self.temperature == 1.0 else z / self.temperature
```

The backbone is a causal LM run **prefill-only** (loaded via `AutoModelForCausalLM(...).model` — no vocab head, no generation) with a two-projection pointer head over the options. The head scores each option's closing `</opt>` hidden state against the question's `<decide>` hidden state; a softmax makes the distribution. Because `<decide>` is last in the branch it can attend to the whole option list.

**Delimiters reuse existing Qwen special tokens** rather than adding embedding rows (`kev/model.py:9-11`): `<|fim_prefix|>` = state, `<|fim_middle|>` = question, `<|box_start|>`/`<|box_end|>` = option open/close, `<|fim_suffix|>` = `<decide>`. The comment gives the rationale: "so no embedding rows need to be added/trained; LoRA adapts their meaning". **`box` here is a Qwen special token, not the System One `box` question type** — the repo has zero occurrences of "bounding" ([code search](https://api.github.com/search/code?q=bounding+repo:jaredpalmer/kev)).

Sequence construction is `encode()` at `kev/model.py:87-127`, packed as `[<state> …]` then per question `[<q> instr <opt> o </opt>… <decide>]`, with `seg` (0 = state, k = question k), `pos` (branch positions restart just after the state), and per-token `opt` index. Two rendering forms produce the same probabilities to fp32 noise (asserted in `tests/test_model.py`):

- **Packed block-causal mask** — attention-only backbones. The mask is `attend(i,j) iff j<=i and (seg[j]==0 or seg[j]==seg[i])` (`kev/model.py:154-185`), so each question reads the state and itself but not its siblings.
- **Per-question rows** — every current Kev, because **Qwen3.5/3.8 mix attention layers with Gated DeltaNet layers, which are recurrent and ignore attention masks** (`kev/model.py:69-72, 259-263`). Each question runs as its own row continuing from the state; the state is computed once and its cache reused. Isolation is exact by construction.

Training context is `MAX_STATE, MAX_BRANCH, MAX_PACKED = 384, 1024, 2048` (`kev/model.py:14`). Serving accepts `SERVE_MAX_STATE = 65536` and `SERVE_MAX_BRANCH = 65536 + 8192` (`kev/model.py:18-19`), with the comment "twice Jev's 32k; the Qwen3.5 / Qwen3.8 bases' window is 262k".

The 27B differs: base is Qwen's post-trained `Qwen/Qwen3.8-27B`, **every** backbone weight is fine-tuned (not an adapter), 64 layers (48 DeltaNet, 16 full attention), hidden 5,120, "vision tower, LM head and multi-token-prediction layers are not loaded" → 25.6B backbone params + 2.6M head ([`docs/model-cards/kev-27b.md`](https://github.com/jaredpalmer/kev/blob/1f04cfa28e6a708485b28a9585010b9e436b776e/docs/model-cards/kev-27b.md)).

## 3. The recipe

There are **two recipes**, and the repo is explicit that they are different.

### 3.1 Small family (0.8B / 4B / 9B) — LoRA on a frozen base

Verbatim from [`docs/model-cards/kev-4b.md`](https://github.com/jaredpalmer/kev/blob/1f04cfa28e6a708485b28a9585010b9e436b776e/docs/model-cards/kev-4b.md):

1. **Base recipe** on `decision-v7` from the base: two epochs; LoRA rank 16, α 32; lr 5e-5 (4B) / 1e-4 (0.8B); one-cycle schedule; effective batch 8 (4 × 2 accum); bf16 autocast; gradient checkpointing; seed 2. Loss is **cross-entropy over each question's options**.
2. **Dates and missing evidence**: one epoch at lr 2e-5 (0.8B: 4e-5) with 2,000 replayed records.
3. **Real documents** (`documents-v1` train): one epoch at lr 2e-5, 2,000 replay, batch 2 × 4 accum, states ≤ 7,552 tokens.
4. **Skills** (`hard-v1` + `devtools-v1` train): one epoch at lr 2e-5, 4,000 replay, batch 2 × 4 accum, 1,915 optimizer steps.
5. **Calibration**: one scalar temperature fitted by NLL (see §7).

Augmentation is part of the recipe and is applied at training time, not baked into the suite: option order shuffled, "none of the above" options and distractors inserted at random, and "a quarter of choice records also yield a minimal pair (the question with a 'none of the above' option, once with the correct option present and once with it removed)". CLI flags: `--p_none 0.1 --p_none_distract 0.12 --p_distract 0.15 --p_none_pair 0.25` (`kev/train.py:400-404`).

LoRA targets are chosen from the model config: `kev/train.py:398` `--lora_targets {all,dense,attn,qv}` where `all`/`attn` additionally include the Gated DeltaNet projections `in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj` on hybrid bases (`kev/model.py:268-274`).

Exact reproduction commands are in the README (§Training), e.g.

```bash
uv run python -m kev.train --suite evals/v7/decision-v7 --base Qwen/Qwen3.5-4B-Base \
    --base_revision 1001bb4d826a52d1f399e183466143f4da7b741b \
    --epochs 2 --lr 5e-5 --batch 4 --accum 2 --dtype bf16 --checkpointing 1 \
    --p_none_pair 0.25 --device cuda --out runs/kev-4b
```

Compute (recorded per stage, `kev-4b.md` §Compute): base recipe ≈ 56 min on one H100 (peak 24.6 GB); dates 9 min on one H100; documents 43 min on an H200; skills 1.4 h on an H200 (peak 47.7 GB). **The whole small-family recipe is a handful of GPU-hours** and is stated in enough detail to rerun.

### 3.2 Kev-27B — full-weight SFT

One epoch over a **145,840-record / 337,130-question** corpus with states up to 32,768 tokens, on 8 × H200, `--full_ft 1`, lr 2e-6, then averaged 0.85 / 0.15 with the earlier adapter-trained Kev-27B. Corpus composition is in the [Kev-27B card](https://github.com/jaredpalmer/kev/blob/1f04cfa28e6a708485b28a9585010b9e436b776e/docs/model-cards/kev-27b.md) §Training data (Kev corpus 78,786; licensed task families 24,000; long states 3,200; long documents 7,617; OOD decisions 7,312; tone 7,544; prompt injection 2,699; agent sessions 4,875; PII 4,152; grounding 5,655).

### 3.3 What is *not* in the recipe

- **No DPO or RL stage anywhere.** `kev/train.py` has no preference-optimisation option. `PLAN.md` finding 13 is explicit: "reinforcement learning has not been tried by us; the Laya review argued that REINFORCE against a proper score has the same optimum as the log loss we minimise" ([`PLAN.md@1f04cfa2`](https://github.com/jaredpalmer/kev/blob/1f04cfa28e6a708485b28a9585010b9e436b776e/PLAN.md), "What we have learned").
- The alternative losses exist but ship in nothing: `--perm_kl`, `--ord_w`, `--label_smoothing`, `--brier_w`, `--focal_gamma` (`kev/train.py:381-386`). README: "The released models don't use the optional `--perm_kl` or `--ord_w` losses." `AGENTS.md` §Calibration Research: the smoothing/brier/focal losses "exist but ship in nothing".
- **"No Jev output in training, ever."** Standing rule, `PLAN.md` §Standing rules.

## 4. Data: exactly what is public and what is not

This is more layered than a single "private corpus". **The mechanism** is `kev/suite.py:32-40`:

> "Frozen suites are mirrored on the Hub. Manifests (with the sha256 of every partition) and the development/test partitions live in git; large training partitions are fetched from this dataset on first use and verified against the manifest… A suite whose partitions must never be public (a held-out test set) names its own mirror in the manifest, `{"mirror": {"dataset": …, "revision": …}}`, usually the private `PRIVATE_DATASET`; only its manifest is in git, which publishes the hashes but not the text."

`SUITES_DATASET = "jaredpalmer/kev-suites"`, `PRIVATE_DATASET = "jaredpalmer/kev-private-evals"`, `SUITES_REVISION = "cc4bac803e73112689ec327ffa481c519cbc7a05"`, `GIT_LIMIT = 10 MiB` (`kev/suite.py:37-40`). `load_split` re-verifies each partition's sha256 against the manifest (`kev/suite.py:204-221`); a private mirror answers "not found" to an unauthenticated fetch and raises `PermissionError` naming the repo (`kev/suite.py:236-241`).

### 4.1 Training corpora

| Suite | Records | In git? | Where the data actually lives |
|---|---|---|---|
| `decision-v7` (small family base set) | 12,576 rec / 15,576 q | **train.jsonl NOT in git** (only calibration/dev/test) | no `mirror` key → default `jaredpalmer/kev-suites@cc4bac80` |
| `hard-v1` train (skills) | 6,000 rec, 23 MB | no (`in_git: false`) | `jaredpalmer/kev-suites@cc4bac80`, published 2026-09-30 |
| `documents-v1` train (CFPB) | 5,219 rec, 23 MB | no | `jaredpalmer/kev-suites@cc4bac80`, published 2026-09-30 |
| `devtools-v1` train | 5,320 rec, 8.9 MB | **yes, in git** | — |
| `sft-v1`, `sft-v2`, `sft-v2-r21/r22/r25/r26` (27B corpus) | 145,840 rec (sft-v2-r22) | **manifest only** | `jaredpalmer/kev-private-train` |
| `decision-v1`, `decision-v2`, `v3/decision-v3` train | — | **yes, in git** (older generations) | — |

Sources: `evals/*/manifest.json` at `1f04cfa2`; git tree via [trees API](https://api.github.com/repos/jaredpalmer/kev/git/trees/main?recursive=1); `PLAN.md` §Next item 3 ("both train partitions (23 MB each, not in git) are in `jaredpalmer/kev-suites` at `cc4bac80…`") and release notes ("Training data published. The documents-v1 and hard-v1 training partitions are in the `jaredpalmer/kev-suites` dataset, so the small models' training data can be fetched and hash-checked").

### 4.2 Evaluation panels

Seven suites carry a `mirror` pointing at the private dataset and have **no data in git** — only metadata files (`manifest.json`, plus `overlap.json` for `breadth-v1` and `longdoc-v1`):

`evals/breadth-v1`, `evals/tasksource-heldout-v1`, `evals/longdoc-v1`, `evals/ood-v2`, `evals/agents-ood-v1`, `evals/guardrails-ood-v1`, `evals/documents-v2` — all → `jaredpalmer/kev-private-evals` (revisions `aeaaed0f75`, `d6498d4c42`, `a54a79c4a4` as recorded per manifest).

The reason is stated in `breadth-v1`'s `mirror_note`:

> "Added after the freeze (2026-09-24): the same frozen breadth-v1 content, not a new version. The partitions are byte-identical (sha256 in `files`) and live only in this private mirror because several sources do not allow public redistribution (SATA-Bench CC-BY-NC-4.0, SGD CC-BY-SA-4.0, Humicroedit with no licence, cfcolor with a code-only notice). Only this manifest is in git; `scripts/build_breadth_v1.py` rebuilds the partitions byte for byte and the manifest without this entry."

**Note a stale flag:** `breadth-v1`'s manifest still records `"in_git": true` for `development.jsonl` and `test.jsonl`, but the git tree at `1f04cfa2` contains only `manifest.json` and `overlap.json` under `evals/breadth-v1/`. The `in_git` flags and `mirror_note` contradict each other; `mirror_note` and the tree agree, and the note explains the flag was not updated when the mirror was added.

Public evaluation panels that *are* in git: `transfer-v4`, `transfer-v9`, `decision-v7` (calibration/dev/test), `documents-v1` (dev/test), `hard-v1` (dev/test), `devtools-v1` (all), `public-pool-v4/v5/v6`, plus older generations and `evals/external/*`.

### 4.3 The written data policy

`PLAN.md` §"Data policy for the SFT work (decided by Jared, 2026-09-24)":

- "The Kev-27B SFT corpus stays private. Its data lives in a private Hub dataset (planned `jaredpalmer/kev-private-train`); this public repo holds only each suite's `manifest.json`, with a `"mirror"` entry, as `evals/documents-v2` does."
- "Its builders and generation prompts live in a private companion repo, not in this public repo. Manifests record the private repo's commit and the code's sha256."
- "Training labels and generations come only from open-weight teachers (DeepSeek, Qwen and similar) or programmatic solvers. Closed frontier models may judge or filter evaluation labels only. Jev never."
- "Model cards disclose each source's kind, size, licence, generation method and the contamination screens, not the text."

A third private repo, `jaredpalmer/kev-knowledge`, is referenced in commit `5e42a7a0` ("the graph is the private repo `jaredpalmer/kev-knowledge`").

### 4.4 Licence handling in the corpus — the opposite choice to Stim's

Kev does **not** operate a permissive-only corpus. It filters by licence in some places and *tracks and quarantines* in others:

- **devtools-v1 does filter permissively.** CodeReviewer: "kept only projects whose current GitHub licence … is permissive; GPL/AGPL/LGPL/MPL/OSL/MS-PL, NOASSERTION and no-licence projects dropped." CommitPackFT: "rows kept only when the per-row `license` … is one of mit, apache-2.0, bsd-2-clause, bsd-3-clause, isc, cc0-1.0, unlicense". FlakeFlagger dropped wildfly (LGPL-2.1) and logback (EPL-1.0 / LGPL-2.1). Method and results committed at `scripts/devtools_v1_licences.json`.
- **breadth-v1 does not filter** — it keeps CC-BY-NC-4.0, CC-BY-SA-4.0, no-licence and code-only sources and makes the panel private instead.
- **The 27B's "licensed task families" (24,000 records) are filtered to "licences that permit commercial use"; "the family list is not published."**
- Share-alike records in the SFT corpus are tracked, not excluded: `evals/sft-v2/manifest.json` `licences: "per source in the component manifests (private mirror); share-alike records flagged per record"`.
- **Gap:** `decision-v7`, `public-pool-v6` and `hard-v1` manifests record dataset ids and pinned revisions but **no licence field**. Only `breadth-v1`, `devtools-v1`, `longdoc-v1`, `sft-v1` and `sft-v2` carry licence metadata. `documents-v1` records a `rights` field: "US CFPB consumer complaint database, US government work (public domain)".

## 5. System One compatibility surface

### 5.1 Endpoints (`kev/serve.py`)

| Method | Path | Notes | Source |
|---|---|---|---|
| `POST` | `/v1/systemone` | the TypeSafe-shaped endpoint | `kev/serve.py:249-252` |
| `GET` | `/v1/models` | one model card per accepted name + Kev serving details | `kev/serve.py:298-312` |
| `POST` | `/v1/systemone/permute` | demo extra: one `choice` question under `n_perm` 1–64 option orders | `kev/serve.py:255-275` |
| `POST` | `/v1/systemone/separate` | demo extra: each question in its own forward pass | `kev/serve.py:278-286` |

Every response carries `x-typesafe-request-id` and `server-timing`; when `KEV_API_KEY` is set, `/v1/*` requires `Authorization: Bearer <key>` (`kev/serve.py:232-242`).

**Model names:** `MODEL_NAMES = ("kev-latest", "jev-latest")` — "both names serve this checkpoint; `jev-latest` is the TypeSafe SDK default model, so an unconfigured client works" (`kev/serve.py:35`). Kev therefore answers to TypeSafe's own default model name.

### 5.2 Question types — support matrix

| Question type | Kev | TypeSafe docs / official SDK | Notes |
|---|---|---|---|
| `noul` | ✅ | ✅ | `criteria: {true?, false?}`. Answer is `{"type":"noul","noul":p}` — **no `confidence` field** |
| `choice` | ✅ 1–255 options | ✅ max 255 | Answer: `choice`, `confidence`, `probabilities` (keys = option names) |
| `score` | ✅ **1–255 levels** | docs: "should have at least two levels; the API accepts up to 10" | Kev is *wider* than the documented service limit |
| `box` | ❌ **422** | ✅ in SDK and docs | See below |
| image `state` | ❌ | ✅ (`Image(photo)`) | Kev's `state` is text/JSON only |

Sources: `kev/api.py:14-40`; [TypeSafe API reference](https://docs.typesafe.ai/api); [Box](https://docs.typesafe.ai/go/box); `typesafe_sdk/_core/question_types.py` at SDK `f36e6854`.

### 5.3 Request / response shape (as implemented)

```jsonc
// request
{"state": "…",                    // string | object | array — rendered to labelled text
 "model": "kev-latest",
 "questions": {"<id>": {"type": "noul"|"choice"|"score",
                        "instructions": …,   // string | object | array, optional
                        "criteria": …}}}     // noul: {true?,false?}; choice: {option: desc|null}; score: [level,…]

// response
{"model": "kev-latest",
 "answers": {"<id>": {…}},
 "usage": {"input_tokens": N, "output_tokens": M},
 "latency_ms": N}
```

`kev/api.py:43-46, 102-117, 149-160, 211-220`. `output_tokens` is explicitly *not* a generation count: "tokens of the serialised answers. Not a measure of generation (there is none)" (`kev/api.py:163-165`). Objects and arrays are flattened to labelled text by `render()` (`kev/api.py:49-55`). Caller text is tokenized through `user_tokens()`, which rewrites `<|name|>` to `<¦name¦>` "so callers cannot forge option/branch delimiter tokens" (`kev/model.py:75-81`).

### 5.4 Confidence formulas — implemented exactly as the reference

`kev/api.py:120-140`:

```python
def choice_confidence(p):   # (p_max - 1/K) / (1 - 1/K)
    K = len(p); return 1.0 if K == 1 else (max(_normalize(p)) - 1/K) / (1 - 1/K)
def score_confidence(p):    # max(0, 1 - E|level - mode| / D), D = uniform MAD about (L-1)/2
    L = len(p)
    if L == 1: return 1.0
    p = _normalize(p); mode = max(range(L), key=p.__getitem__)
    D = sum(abs(i - (L-1)/2) for i in range(L)) / L
    return max(0.0, 1.0 - sum(pi*abs(i-mode) for i, pi in enumerate(p)) / D)
```

The comment says both "mirror TypeSafe's reference adapter, `system-one-adapter` 0.2.1 (`src/system_one_adapter/_utils/confidence_metrics.py`)". **Verified by direct comparison**: I cloned [`typesafe-ai/system-one-adapter-python`](https://github.com/typesafe-ai/system-one-adapter-python) at `e1d4cc938204b22fc5a3c3aca7044072fe3f712d` and read `src/system_one_adapter/_utils/confidence_metrics.py` — `choice_confidence`, `score_confidence` and `_normalize` are the same formulas, including the single-option → 1.0 case and the zero-sum → uniform fallback. Kev also matches the reference's `_normalize` behaviour of normalising before measuring.

Score answer construction is the probability-weighted mean: `score = sum(i * p_i for i, p_i in enumerate(p))` (`kev/api.py:158`), so it can land between levels — as TypeSafe documents. Probabilities are rounded to **4 decimals**, with a stated reason: "255 × 0.00005 < 0.02" keeps the rounded distribution inside TypeSafe's `|sum − 1| < 0.02` tolerance (`kev/api.py:143-146`).

**noul has no confidence**, matching TypeSafe: the [Noul docs](https://docs.typesafe.ai/go/noul) say "There is no separate confidence value for a Noul" because a two-outcome distribution is fully described by one number. Kev returns only `noul` (`kev/api.py:152-153`).

### 5.5 Batching many questions per state

- A request may carry **any number of questions**; there is no cap in the schema (`SystemOneRequest.questions: dict[str, Question] = Field(min_length=1)`).
- Questions are answered independently. Exact isolation is by construction on hybrid bases (one row per question). `tests/test_api.py::test_packed_equals_separate` asserts answers do not change when a sibling question is removed from the request, using a tolerance of 0.011; `tests/test_model.py` asserts packed vs rows agreement.
- The server batches across *requests*: `MAX_BATCH = 64` requests at once (`kev/serve.py:34`), and within a request it runs "one maximal row of 16,384 tokens per forward pass, counting the cached document once per question in that pass, so memory does not grow with the question count and the answers do not depend on the split" (`kev/model.py:23` `ROW_PASS_TOKENS`; `rows_per_pass`).
- A state-prefix LRU cache (`KEV_PREFIX_CACHE`, default 4 states / 65,536 state tokens) means a repeated `state` only pays for its question rows (`kev/serve.py:27-29, 38-91`).

### 5.6 Deviations from TypeSafe's documented behaviour

1. **`box` is absent, and more deeply than "unimplemented".** Per TypeSafe's own [Box page](https://docs.typesafe.ai/go/box), a Box question "find[s] where something is in an image", "needs an image in the request", takes **no `criteria`**, and "Box questions need Jev 1.14". Its answer is `box: {top, bottom, left, right}` as fractions of the image plus optional `confidence`; "A Box answer is always a box. There is no 'not found' answer." Kev's `state` type is `Union[str, dict, list, int, float, bool, None]` (`kev/api.py:13`) — there is no image/asset content type in the schema at all — and `Question = Union[Noul, Choice, Score]` (`kev/api.py:40`), so a `{"type": "box"}` question fails validation. `tests/test_api.py::test_validation_422` pins the behaviour with an unknown type. So closing the `box` gap is not a routing change: it needs image input, an image-capable backbone, and a box regressor head. Stim's glossary calling `box` "the most complete open implementation"'s "clearest known hole" is right about the hole, but understates its depth.
2. **`score` accepts 1–255 levels** where TypeSafe's docs say up to 10. Kev is permissive where TypeSafe is not, so a request that works against Kev may 422 against Jev — the compatibility is one-directional in this respect.
3. **Extra response fields** that TypeSafe does not model: `latency_ms`, and (on a `KEV_TRUNCATE_STATES=1` server) `truncated` + `usage.state_tokens`/`state_tokens_used`. `tests/test_api.py::test_truncating_server_marks_every_response` asserts the official SDK still parses those responses.
4. **Over-length states are refused, not truncated**, by default: a 422 naming the token count and the limit (`kev/model.py:130-142`; `tests/test_api.py::test_over_length_state_is_refused_not_truncated`). The release notes list this as a 1.0 change: "The server used to cut a state longer than its limit without saying so."
5. **`GET /v1/models` returns a `models` list where each entry carries `name`, `description`, `release_date`** plus Kev-specific keys. `tests/test_api.py::test_sdk_models` asserts `{"kev-latest","jev-latest"} <= {c.name for c in cards}` and that the official SDK parses it.
6. **Deviation confirmed against the TypeSafe Python SDK** at `f36e68549ceca3191e5791e5f99ea1baa1a258b5`: the SDK's `Question` union is `Noul | Choice | Score | Box | QuestionModel` (`src/typesafe_sdk/_core/question_types.py:169`) and `BoxAnswer` carries `box: BoxCoordinates` + `confidence: float | None` (`src/typesafe_sdk/_core/response_types.py:67-76`). The docs changelog lists "support `Box` questions and answers" and "add docs for `Box` type" as separate entries.

## 6. Evaluation methodology

### 6.1 Suites

| Suite | What it measures |
|---|---|
| `decision-v7` | held-out items of the ten training datasets, the generated policies and the rule structures ("trained sources") |
| `transfer-v4` | 764 records from sources Kev never trained on: QNLI, SciQ, PAWS, MMLU, Emotion, TweetEval-offensive, plus held-out policy and rule structures ("new sources") |
| `transfer-v9` | `transfer-v4` + 10-way MMLU-Pro + records buried in unrelated text + "unknowable" records whose deciding evidence was removed |
| `breadth-v1` | 14 held-out public datasets in five areas, scored with the community Decision Index 0.2 chance-corrected formula (private mirror) |
| `tasksource-heldout-v1` | 24 whole task families of a public multi-task collection, "family names private" (private mirror) |
| `hard-v1` | 7 programmatically generated skill families; test holds out generator templates |
| `devtools-v1` | developer-tooling decisions from licence-checked sources (4 trained, 2 evaluation-only) |
| `documents-v1` / `documents-v2` | CFPB consumer-complaint narratives; **v2 is a private held-out test** |
| `longdoc-v1` | CUAD contracts + generated agreement bundles, states 4k–64k tokens |
| `evals/external/` | other projects' test sets converted to this format, with published Jev results (e.g. SemIf) |

### 6.2 How a panel is frozen and held out

- Every partition is pinned by sha256 in the suite's `manifest.json`; `load_split` re-checks the digest and the record count on read (`kev/suite.py:204-221`).
- The **test partition is locked behind `--allow-test`** and the code comment is blunt: `if split == "test" and not allow_test: raise ValueError("locked test requires explicit --allow-test; never use it for search")` (`kev/suite.py:208-209`).
- Standing rule: "**Test and locked partitions are read once per candidate**, only for the candidate the committed rule selected, only after that rule passed. Never for a second candidate; never re-read. A read panel becomes a selection or regression set for everything after it." (`PLAN.md` §Standing rules.)
- `breadth-v1` records its `label_protocol` as "no LLM labels: every label is the source dataset's own (human, engine, recorded execution or by construction); candidate subsets always contain the reference answer". Its `selection` rule: "native dev/test splits kept where they exist, otherwise groups dealt to halves in a seeded hash order … normalised-text state dedupe across all datasets and both partitions (test first); no group spans the partitions".
- Clean/perturbed variants share a `group_id`, and every interval resamples **whole records**, "so questions sharing a state move together".

### 6.3 Metrics — the exact definitions

- **ECE** (`kev/metrics.py:15-21`): **10 equal-width bins over [0, 1]**, last bin closed at both ends, weighted by bin population share; scored on `p_max` vs correctness.
  ```python
  def ece(conf, correct, bins=10):
      edges = np.linspace(0, 1, bins + 1); e = 0.0
      for lo, hi in zip(edges[:-1], edges[1:]):
          m = (conf >= lo) & (conf < hi) if hi < 1 else (conf >= lo) & (conf <= hi)
          if m.any(): e += m.mean() * abs(correct[m].mean() - conf[m].mean())
      return float(e)
  ```
- **Accuracy**: `p.argmax() == label`. **Brier**: `Σ(p − onehot)²` over the question's options. **NLL**: exact from recorded logits when present, otherwise from floored probabilities (`EPSILON = 1e-9`).
- **Selective prediction**: `coverage_at_error(confidence, correct, budget)` walks the risk–coverage curve in descending confidence "over whole equal-confidence groups" and returns the largest accepted share whose empirical error stays ≤ budget; plus `aurc` (right-step integral) and `select_threshold`. `metric_policy.version = 2` is written into every report. Documented caveat: "The empirical coverage-at-error envelope is not an unseen-data guarantee; freeze thresholds on a separate calibration set" (`AGENTS.md` §Calibration Research).
- **Confident-error rate**: share of *all* questions answered at `p_max ≥ 0.9` and wrong (`kev/metrics.py:60-61, 82`).
- **Score-specific**: `score_mae` and `ranked_probability_score` (`kev/metrics.py:75-77, 98-99`).
- **Permutation**: each record is re-asked with a permuted option order (`variant: "permuted"`), reporting `mean_max_delta` and `flip_rate`.
- **Unknowable**: records whose deciding evidence was removed are scored on *confidence only* ("Accuracy on the unknowable records is meaningless by construction"), paired against intact controls (`kev/metrics.py:167-180`). Reported as "Unanswerable items answered with p ≥ 0.9" — Kev-4B 0.00, Jev 0.09.
- **Bootstraps**: `paired_bootstrap` is a "paired cluster percentile bootstrap; full statistic recomputed in each resample", 2,000 resamples, seed 0, resampling units = `(source, group)` clusters, "so sibling questions and variants of one record move together". Non-additive metrics (`ece`, `coverage_at_*`, `aurc`) are recomputed per draw rather than averaged (`kev/metrics.py:372-426`).

### 6.4 Suites removed as unsound

The repo removes gates it finds broken rather than patching them, and keeps past verdicts. `kev/suite.py:49-…` `REMOVED_SUITES`:

| Suite | Removed | Last round | Reason (verbatim summary) |
|---|---|---|---|
| `evals/external/scienthoon-v1` | 2026-09-27 | 22 | "291 templated synthetic support tickets × 3 questions; `queue` is saturated …; `priority` is unlearnable by construction …; `angry` has 15 of 291 gold labels that contradict the text" |
| `evals/external/wanli-v2` | 2026-09-30 | 26 | "271 of its 1,002 WANLI test pairs (27 %) are ones WANLI's two crowd annotators labelled differently, and the published gold is one of the two labels" |
| `evals/external/wanli-v1` | 2026-09-30 | 5 | same family |
| TypeSafe's public evals | 2026-09-30 | — | "gold is the averaged answer of two closed frontier models, and on 89 questions it cannot tell checkpoints apart" (README) |

`load_split` refuses a removed suite with its reason; `kev.rounds validate` refuses any later round naming it. From round 27 "SemIf is the only external read, report only."

## 7. Temperature calibration — the 0.059 → 0.0085 record, verified

Stim's glossary records "Kev's measured failure mode (ECE 0.059 → 0.0085 when fitted properly)". **This is confirmed by three independent committed primary sources, and by the underlying JSON.**

**The rule**, [`docs/autoresearch.md`](https://github.com/jaredpalmer/kev/blob/1f04cfa28e6a708485b28a9585010b9e436b776e/docs/autoresearch.md) §3 item 4:

> "**MUST: every served or shipped temperature comes from a pool of held-out datasets, never from a partition of the training corpus** … The held-out *items* of the training sources … round 19's SFT arms were served at T 0.955 fitted on `sft-v1` development rows and failed every calibration criterion (breadth-v1 ECE 0.059); round 20's held-out-datasets pool gave 0.0085 on the same checkpoint."

**The mechanism**, [`scripts/calibrate_checkpoint.py`](https://github.com/jaredpalmer/kev/blob/1f04cfa28e6a708485b28a9585010b9e436b776e/scripts/calibrate_checkpoint.py) docstring (lines 6–14):

> "The fit rows must be held-out DATASETS, not held-out items of what the checkpoint trained on. The script refuses rows that share data with the checkpoint's training (`kev.rounds.pool_conflicts`) … That is round 19's failure mode: its SFT arms were served at T 0.955 fitted on `sft-v1` development rows (held-out items of the training sources, in distribution) and missed breadth-v1 calibration (ECE 0.059); a pool of held-out datasets gave 0.0085 on the same checkpoint (round 20)."

**The narrative**, `PLAN.md` §"What we have learned" item 5 — including "Every temperature shipped before round 20 was fitted on the checkpoint's own decision-v7 (or suite) development rows, which this script now refuses without `--allow-in-distribution`."

**The exact numbers**, [`runs/r20-readout/round20.json`](https://github.com/jaredpalmer/kev/blob/1f04cfa28e6a708485b28a9585010b9e436b776e/runs/r20-readout/round20.json):

| Arm | Fit pool | T | breadth-v1 n | ECE candidate | ECE parent (Kev-27B) |
|---|---|---|---|---|---|
| `27b-a` (= round 19 arm `lr2e6/00-trial-0`) | 648 questions, 8 held-out sources + `mmlu_pro` | **1.4142135623730947** | 3,075 | **0.008459701388236573** | 0.011822307606708295 |
| `27b-b` (`lr5e6`) | same pool, same 648 questions | 1.381912879967776 | 3,075 | 0.017294873885174568 | 0.011822307606708295 |

The registered criteria recorded in that file: `3_breadth_ece_at_most_parent_plus_0.01: True`, `3_kev_ece_at_most_parent_plus_0.01: True`. The arm still **failed overall** (`passed: false`) on accuracy, WANLI-v2, scienthoon and pooled-external guards — round 20 selected no candidate.

**Scope matters.** This is the breadth-v1 ECE of **one round-19 full-weight SFT candidate for Kev-27B**, not a property of the released checkpoints. The number is fit-pool-dependent: `PLAN.md` finding 5 records an exploratory fit on held-out datasets (chosen after seeing breadth-v1) giving arm (a) 0.017, before the registered 648-question pool gave 0.0085. Two of the four shipped Kev temperatures (0.8B 2.35, 4B 2.41) were themselves fitted in-distribution and **kept**, because registered refits on held-out datasets failed their criteria (`docs/model-cards/kev-4b.md` §Calibration; `kev-0.8b.md`; release notes §Calibration).

### 7.1 A correction to Stim's glossary on reordering

Stim's glossary says "One temperature can rescale confidences but cannot reorder them." Two Kev primary sources narrow that:

- **Kev's own `AGENTS.md` §Calibration Research: "Temperature can reorder confidence across multiclass questions, even at fixed option count."**
- README §Calibration: "A temperature never changes which answer wins." — true, and consistent, because within one question a softmax over a fixed option set is invariant in argmax under a positive scalar.

The two statements do not conflict, but they mean different things, and the difference is load-bearing for Stim:
- Within a question: argmax and accuracy are temperature-invariant ("Argmax never changes; accuracy is identical before and after" — `calibrate_checkpoint.py`).
- **Across questions: the ordering of `p_max` is not invariant**, so anything that selects on confidence — `coverage_at_5pct_error`, `aurc`, `confident_error_rate`, `select_threshold` — **does change with temperature**. The README states the measured effect on Kev-9B: "On new sources it takes Kev-9B's calibration error from 0.103 to 0.041 and its confident errors (wrong answers with probability ≥ 0.9) from 8.2% to **2.4%**, below Jev's 3.7%. The accuracy numbers above are the same either way."
- `kev/metrics.py:373` classifies `coverage_at_5pct_error`, `coverage_at_1pct_error`, `aurc` and `ece` as **nonlinear** — recomputed on every bootstrap resample rather than averaged — which is a code-level admission that they are not invariants of the raw distribution.

So the safe Stims-side phrasing is: *one temperature cannot change which option wins, but it does change selective-automation coverage and the confident-error rate.*

### 7.2 How the scalar is fitted (implementation)

`kev/metrics.py:269-294`:

```python
TEMPERATURE_FIT = {"aggregation": "micro", "points": 121}
# "min micro mean NLL over a 121-point log grid 0.25..4"
def fit_temperature(rows, aggregation="macro", points=81):
    candidates = np.exp(np.linspace(np.log(0.25), np.log(4), points))
    ...
```

Shipped temperatures and workload fits use the 121-point grid on `[0.25, 4]` with micro aggregation and every question weighted equally; research trials and the finetune skill use the default 81-point grid and are "screening reports, not shipped values". `out_of_fold_rows` fits per fold on group-disjoint folds and applies to held-out rows; `scripts/calibrate_checkpoint.py` reports that alongside the in-sample fit. Code enforces "fit temperature on raw logits, not previously calibrated outputs" (`kev/metrics.py:284-285`).

## 8. Licensing

| Artifact | License | Source of truth |
|---|---|---|
| Repository code | **Apache-2.0**, "Copyright 2026 Jared Palmer"; no NOTICE file, no additional terms | [`LICENSE`](https://github.com/jaredpalmer/kev/blob/1f04cfa28e6a708485b28a9585010b9e436b776e/LICENSE) (202 lines); repo API `license.spdx_id` |
| Weights + head (0.8B/4B/9B) | **Apache-2.0** — "License: Apache-2.0 (adapter and head); the base model is Apache-2.0" | `docs/model-cards/kev-4b.md:81`, `kev-0.8b.md:86`, front-matter `license: apache-2.0` |
| Weights (27B) | **Apache-2.0** — "License: Apache-2.0 (weights and head); the base model is Apache-2.0" | `docs/model-cards/kev-27b.md` |
| Base models | Apache-2.0 (Qwen3.5 `-Base` and Qwen3.8-27B post-trained) | model cards; README §License |
| Training data | **No single license.** Per source, per suite. README: "Training datasets have their own licenses; see the model cards." | README §License |
| `documents-v1` | public domain (US CFPB complaint narratives, "US government work") | `evals/documents-v1/manifest.json` `rights` |
| `devtools-v1` | per source: CodeReviewer CC-BY-4.0 (permissively filtered to the repo level), CommitPackFT MIT + per-row permissive only, Aegis CC-BY-4.0, FlakeFlagger CC-BY-4.0 + BSD-3-Clause, When2Call CC-BY-4.0 (eval-only), prompt-injection Apache-2.0 + MIT (eval-only) | `evals/devtools-v1/manifest.json` `sources`; `scripts/devtools_v1_licences.json` |
| `breadth-v1` | mixed and **partly non-redistributable**: MuSR CC-BY-4.0, SATA-Bench CC-BY-NC-4.0, ChessBench Apache-2.0 + CC0, CUAD CC-BY-4.0, and (per `mirror_note`) SGD CC-BY-SA-4.0, Humicroedit no licence, cfcolor code-only notice | `evals/breadth-v1/manifest.json` |
| `sft-v1` / `sft-v2` (27B corpus) | "per source in the component manifests (private mirror); share-alike records flagged per record" | `evals/sft-v1/manifest.json`, `evals/sft-v2/manifest.json` |
| `hard-v1`, `decision-v7`, `public-pool-v6` | **no licence field recorded** — dataset ids and pinned revisions only | those manifests |

**What this means for someone republishing a derived artifact.** Code and weights are Apache-2.0 with no field-of-use restriction, so redistributing the adapter/head or a fine-tune is unencumbered — with the usual Apache-2.0 obligations on the *code*. The data is the hazard: the repo deliberately separates "publishable" from "keep private" by licence, using a private Hub mirror for anything it judges non-redistributable (`breadth-v1`, `tasksource-heldout-v1`, `longdoc-v1`, `ood-v2`, `agents-ood-v1`, `guardrails-ood-v1`, `documents-v2`). A republisher who mirrors `jaredpalmer/kev-suites` wholesale inherits whatever the constituent source licences say — and for `decision-v7`'s ten public datasets the repo itself records no licence metadata, so each Hub card would have to be checked independently. Note also that the CodeReviewer filtering was a *repository-level* check at lookup time, not at collection time (`licence_check`: "The licence is the repository's at lookup time, not at collection time"), which the manifest records as a known limitation.

## 9. Reproducibility, concretely

**What a stranger can do today, from published artifacts alone:**

1. Clone, `uv sync --extra serve`, and serve any released checkpoint from the Hub, including under an API key, on CUDA or MLX (`README` §Quick Start). 2. Point the **official TypeSafe Python SDK** at it unchanged — with an extra `--extra serve`; the SDK runs against the local server in `tests/test_api.py`.
3. Run `kev.benchmark` against `transfer-v4`, `transfer-v9`, `decision-v7`, `hard-v1` (dev/test), `devtools-v1` (all splits), `documents-v1` (dev/test) — all in git — and re-derive the README/model-card numbers for those panels.
4. Retrain the **small-family recipe** from `decision-v7` (`uv run python -m kev.train --suite evals/v7/decision-v7 …`), **if** `evals/v7/decision-v7/train.jsonl` downloads from `jaredpalmer/kev-suites` — see the caveat below.
5. Rebuild `hard-v1` and `documents-v1` train byte-for-byte from their builders (`scripts/build_hard_v1.py`, and the documented sha256s), and rebuild `breadth-v1` byte-for-byte from `scripts/build_breadth_v1.py` given network access to the four original sources.
6. Parse every published number back to its committed report (`docs/claims.json`, 1,019 claims; `scripts/verify_claims.py`).
7. Run the CI test set (no weights, no server) and the server conformance tests.

**What a stranger cannot reproduce:**

| Missing | Consequence |
|---|---|
| `jaredpalmer/kev-private-train`: `sft-v1`, `sft-v2`, `sft-v2-r21/r22/r25/r26` | **Kev-27B cannot be retrained at all.** The 145,840-record corpus, and the builders and generation prompts that produced it, are in private repos. |
| `jaredpalmer/kev-private-evals` (7 suites) | breadth-v1, tasksource-heldout-v1, longdoc-v1, ood-v2, agents-ood-v1, guardrails-ood-v1 and documents-v2 **cannot be rescored**. Every headline "new sources" number in the README (breadth-v1 index) and the "held-out task families" number rest on the private mirror. |
| `tasksource-heldout-v1` family names | Even with the data, the 24 task families are not named — "family names private" — so the panel is not independently reconstructible from its source collection. |
| The 27B's "licensed task families" list (24,000 records) | "the family list is not published" (Kev-27B card). |
| Non-redistributable sources inside `breadth-v1` | Rebuilding breadth-v1 requires re-fetching CC-BY-NC / CC-BY-SA / unlicensed sources from their own origins at pinned revisions. Transparency, but not a downloadable artifact. |
| Private GPU/Modal volumes (`kev-runs`) and per-round `rows.json` for private panels | Committed `runs/**` covers public panels; private-panel rows are restorable only "for an account with access" via `scripts/private_rows.json` + `scripts/private_rows.py restore`. |

**The honest shape of the gap.** For the **small family (0.8B/4B/9B)** Kev is much closer to Stim's reproducible-record bar than the glossary's "private corpus" implies: the base recipe's data for `hard-v1`, `documents-v1` and `devtools-v1` is published and hash-checkable, the recipe is a handful of GPU-hours, the exact commands are in the README, and the *training* data for all four stages is reachable if `kev-suites` is public. For **Kev-27B** it is the opposite: the corpus, the builders and the prompts are private by written policy, and three key metrics (breadth-v1, tasksource-heldout-v1, longdoc-v1) are on private panels. Stim's glossary statement is accurate for the 27B and the held-out panels, and understates what the small family publishes.

**One reproducibility caveat that bites any comparison:** Kev defines "fp32-exact" as *bit-reproducible for a fixed kernel set*, not universally. `AGENTS.md` §Notes records that putting `causal-conv1d`'s CUDA kernel in the Modal image "alone moved Kev-27B v1 by max |Δp| 0.03-0.06 (0-0.4 % flips, ECE +0.0008 on breadth-v1)", and that every evaluation report now records the kernel set its logits depend on. Two reads from different kernel sets are not comparable. Stim should record the same provenance or its reference scores will drift without any code change.

## 10. Lessons and conventions Stim should adopt or avoid

*This section is my reading of the evidence, not a claim from the sources.*

**Adopt**

1. **Lock the test partition behind an explicit flag and write the reason in the error string.** `kev/suite.py:208-209` makes "never use it for search" part of the code, not a convention. Cheap, and it makes accidental test peeking a deliberate act.
2. **Hash-pin every partition and re-verify on read** (`load_split` re-checks sha256 and record count). This is the cheapest part of Stim's reproducible-record bar and Kev shows it costs almost nothing.
3. **Cluster the bootstrap by record, not by question.** Kev resamples `(source, group)` clusters so sibling questions and perturbation variants move together (`kev/metrics.py:297-314`). Answering N questions about one state is one observation, not N.
4. **Report `confident_error_rate`, `coverage_at_error` and `AURC`, not only ECE and accuracy.** These are the numbers a deployer thresholds on, and they are the metrics temperature moves (§7.1). ECE alone would have hidden the Kev-9B 8.2 % → 2.4 % change.
5. **Fit the temperature on a held-out pool, and record *which* rows were used inside the checkpoint** (`head.pt["temperature_fit"]["fit_rows"]`), with a hard refusal otherwise. Kev's `calibrate_checkpoint.py` refuses a fit set that shares data with the checkpoint's training, records every conflict, and needs `--reason` for a hand-set value. This is exactly Stim's temperature-calibration term and the implementation is worth copying.
6. **Report an out-of-fold calibration estimate next to the in-sample one** so a reader can see the gap.
7. **Score "unknowable" items on confidence, not accuracy.** Kev's `unknowable_report` removes the deciding evidence and asks whether the model still answers at `p ≥ 0.9` — the single most transferable idea in the eval suite for a decision model, and directly usable by Stim.
8. **Remove unsound suites rather than patch them, and keep past verdicts** (`REMOVED_SUITES` with reasons, dates, last round read, and committed rows). This is a stronger norm than "freeze the suite": freezing a broken gate just makes the breakage permanent.
9. **Publish the negative results.** `PLAN.md` finding 13 lists a dozen things tried and abandoned (calibration losses, question-side LoRA, checkpoint averaging, self-distillation, WiSE-FT interpolation, `KEV_DATE_FACTS` as default). This is the highest-value-per-byte content in the repo for anyone about to spend GPU-hours.
10. **Register the rule before the read, and commit the registration.** "The commit time is the registration time" (`PLAN.md` §Standing rules). This is what makes "read once, ever" credible rather than asserted.

**Consider carefully / avoid the pattern**

1. **Do not claim "we can retrain" without checking which splits are actually fetchable.** Kev's own README says "you can train and run yourself" and the exact commands are published, while the *27B's* corpus is private by written policy. The gap between the marketing verb and the artifact is the exact failure Stim's reproducible-record bar is designed to prevent. Stim should state its publication scope per artifact, as Kev's model cards do, rather than per family.
2. **Kev's licence handling is inconsistent, and Stim's permissive-only corpus is the stricter and cleaner position.** Kev filters permissively for `devtools-v1` and the 27B's task families, but keeps CC-BY-NC / CC-BY-SA / unlicensed sources in `breadth-v1` and quarantines them in a private mirror instead; and `decision-v7`, `hard-v1` and `public-pool-v6` carry **no licence metadata at all**. For a corpus that is meant to be a downloadable, licence-clear first-class artifact, recording per-source licences in the manifest is non-negotiable — and it is the thing Kev most conspicuously under-does.
3. **The `box` gap is architectural, not a routing omission** (§5.6). If Stim wants System One compatibility claims to survive contact with `box`, it needs an image-capable state and a coordinate head; if it does not, the honest move is to say plainly that `box` is out of scope and that the claim covers `noul`/`choice`/`score`. Kev currently does neither explicitly.
4. **Publishing a training corpus that mixes share-alike sources creates a downstream obligation.** Kev flags share-alike per record in `sft-v2` but publishes only the manifest. Stim's permissive-only corpus avoids the problem; if Stim ever relaxes it, per-record licence flags are the minimum.
5. **A "checked in CI" claim needs the workflow to actually contain it.** README asserts the numbers are checked in CI; the committed `ci.yml` runs only unit tests and playground lint. Trust the check, not the sentence — and make Stim's own verification path runnable by a stranger with one command.
6. **Kernel provenance is a real reproducibility axis.** Kev found a kernel change moving probabilities by 0.03–0.06 with no code change. Stim should record package versions, GPU, dtype and attention implementation alongside every reference score.
7. **Kev answers to the model name `jev-latest`** (`kev/serve.py:35`). That makes an unconfigured TypeSafe client work, and it is a deliberate compatibility choice by a sibling project. Stim should decide explicitly whether it does the same, because it is a claim of substitutability, not just a default.

## 11. On AutoJev (the ecosystem sibling)

`denis-pplx/autojev` is the AutoJev that `PLAN.md` refers to — it matches the description "AutoJev-27B (full-weight SFT on 73k broad synthetic decisions)" against that repo's README headline "One H200 · full-weight SFT · **73,000** unique training examples · 286 updates".

| Fact | Value |
|---|---|
| Repo | [github.com/denis-pplx/autojev](https://github.com/denis-pplx/autojev), MIT code license, created 2026-09-19, last push 2026-09-22, 132 stars / 15 forks |
| Base | Qwen3.8-27B, full-weight SFT, one H200, ~49 GiB bf16 |
| API | `POST /v1/systemone` supporting `choice`, `noul`, `score`, **plus optional base64 `images`** — so it advertises image input where Kev does not, but no `box` |
| Weights | Apache-2.0 (Hub `denis-pplx/autojev-27b`) |
| Reported | overall accuracy 84.60 %, ECE 0.04282, Brier 0.22027 vs Jev 82.79 % / 0.05274 / 0.25400 |
| Corpus | "**The exact curated training corpus is not bundled.**"; at README time the weights were behind `hf auth login` |

**Glossary note:** Stim's CONTEXT.md describes AutoJev as "full-weight SFT on pooled public data". The "full-weight SFT" part is exactly right. "Pooled public data" is not something AutoJev's own README states — it says the curated corpus is not bundled — and the previous sentence's price does not rest on it. Worth softening on Stim's side, or restating as Kev states it (`PLAN.md`: "73k broad synthetic decisions").

## 12. Sources

All repo paths read at **`1f04cfa28e6a708485b28a9585010b9e436b776e`** (2026-10-15T22:41:06Z), observed 2026-10-16, via the GitHub REST API unless noted.

**Kev repository** — <https://github.com/jaredpalmer/kev>
- Repo/commits/releases/contributors/languages: `api.github.com/repos/jaredpalmer/kev[/commits|/releases|/tags|/contributors|/languages|/issues]`
- Git tree: `api.github.com/repos/jaredpalmer/kev/git/trees/main?recursive=1` (3,057 blobs)
- `README.md`, `LICENSE`, `PLAN.md` (3,185 lines), `AGENTS.md` (463 lines), `docs/autoresearch.md` (237 lines), `docs/releases/kev-1.0.md`, `docs/claims.json` (1,019 claims), `.github/workflows/ci.yml`, `.gitignore`
- Model cards: `docs/model-cards/kev-{0.5b,0.6b-qwen3,0.8b,4b,4b-qwen3,8b-qwen3,9b,27b}.md`
- Code: `kev/{api,model,metrics,calibrate,benchmark,serve,train,suite,data,rounds}.py`; `scripts/calibrate_checkpoint.py`; `scripts/devtools_v1_licences.json`; `tests/test_api.py`, `tests/test_model.py`
- Suite manifests: `evals/{v7/decision-v7,v8/decision-v8,v9/transfer-v9,v4/transfer-v4,breadth-v1,documents-v1,documents-v2,devtools-v1,hard-v1,longdoc-v1,ood-v2,tasksource-heldout-v1,agents-ood-v1,guardrails-ood-v1,sft-v1,sft-v2,sft-v2-r21,sft-v2-r22,sft-v2-r25,sft-v2-r26,public-pool-v6}/manifest.json`
- Reports: `runs/r20-readout/round20.json`, `runs/r19-readout/round19.json`
- Code search: `api.github.com/search/code?q={bounding,0.0085,"noul"+and+"box"}+repo:jaredpalmer/kev`

**First-party author material**
- [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked) — credited in Kev's README as the architecture write-up Kev is based on (not fetched here)

**TypeSafe (official)**
- [API reference](https://docs.typesafe.ai/api); [Box](https://docs.typesafe.ai/go/box) (primitives: bounding box on an image; needs Jev 1.14; no `criteria`; `box` + optional `confidence`); [Noul](https://docs.typesafe.ai/go/noul) (no separate confidence for a Noul)
- [`typesafe-ai/system-one-adapter-python`](https://github.com/typesafe-ai/system-one-adapter-python) at `e1d4cc938204b22fc5a3c3aca7044072fe3f712d`, MIT — `src/system_one_adapter/_utils/confidence_metrics.py`
- [`typesafe-ai/typesafe-sdk-python`](https://github.com/typesafe-ai/typesafe-sdk-python) at `f36e68549ceca3191e5791e5f99ea1baa1a258b5`, MIT — `src/typesafe_sdk/_core/{question_types,response_types}.py`, `docs/changelog.md`

**Ecosystem**
- [`denis-pplx/autojev`](https://github.com/denis-pplx/autojev) — README, repo API (MIT code, weights Apache-2.0)
- [Hugging Face collection `jaredpalmer/kev`](https://huggingface.co/collections/jaredpalmer/kev-6aad9d0ea49f2589665e07cd) and dataset `jaredpalmer/kev-suites` — linked from the README; **not fetched** (see below)

## 13. Open questions / could not verify

1. **Is `jaredpalmer/kev-suites` public?** `huggingface.co` was unreachable from the research environment (`curl` returned HTTP code 000 for both `kev-suites` and `kev-private-evals`; `FetchURL` and the built-in browser both failed). The evidence points to public — the README badges it as a public dataset, `PLAN.md` calls it "the public mirror", `kev/suite.py:32-40` contrasts it with `PRIVATE_DATASET`, and `evals/breadth-v1` overrides it precisely because that panel must not be public — but **I could not confirm it directly**, and I could not enumerate which files it actually contains. This is the single biggest open question, because it decides whether `decision-v7`'s training split (and therefore small-family retraining) is reproducible by a stranger. **Recommend checking `https://huggingface.co/datasets/jaredpalmer/kev-suites` before relying on it.**
2. **Is `evals/v7/decision-v7/train.jsonl` actually present in `kev-suites@cc4bac80`?** Its manifest has no `mirror` key, so `fetch_partition` will look there; the release notes only *name* `documents-v1` and `hard-v1` train as published. Presence is likely but unverified.
3. **Are the released checkpoint weights on the Hub Apache-2.0 without an access gate?** The model cards and front-matter say `license: apache-2.0`, and Kev's `release_verify` step resolved each `@v1.0` "with no token and a fresh cache" (`PLAN.md` §Released: Kev 1.0). I did not fetch the Hub pages to confirm the card's rendered license field or repo visibility.
4. **Does CI actually run `scripts/verify_claims.py`?** The README says the numbers "are checked in CI"; `.github/workflows/ci.yml@1f04cfa2` contains no such step. Either there is another workflow, a scheduled job, or the README overstates it — I only read `ci.yml`, and did enumerate `.github/workflows/` only via that one file.
5. **Is Kev-27B's released checkpoint reproducible in practice?** Even with data access, the README's 27B measurements are on 8 × H200 with per-pass memory ceilings and resume points; `PLAN.md` records round 21 failing at startup with OOM. I did not attempt to cost or run it.
6. **`breadth-v1`'s `in_git: true` flags contradict the tree and its own `mirror_note`.** I report `mirror_note` + the tree as authoritative; I did not find a commit that explains the mismatch, and when the flags were left stale is unverified.
7. **The exact provenance of the 0.059 figure at its rawest level.** Three committed documents (script docstring, `autoresearch.md` §3.4, `PLAN.md` finding 5) state breadth-v1 ECE 0.059 for the T 0.955 in-distribution fit, and the 0.0085 side is exact in `runs/r20-readout/round20.json` (`0.008459701388236573`). I did **not** find the raw `runs/…/report.json` holding 0.059 itself — the round-19 breadth report is a summary table, and the per-arm breadth ECE I could read in `runs/r19-readout/round19.json` was for a different arm (`27b-olddata`, ECE 0.05985 at T 1.0). Worth a second look if the number becomes load-bearing.
8. **Does the "no reordering" nuance change any published Kev claim?** Kev reports accuracy and selective metrics separately and states accuracy is temperature-invariant, so no published number looks wrong. But I did not verify that every `coverage_at_5pct_error` in the model cards is at the shipped temperature rather than raw — the README says "Published accuracy numbers use fp32 evaluation" and that Brier numbers are raw, without saying which temperature the coverage column uses.
9. **Stim-glossary wording that this research does not support as stated:** (a) "private corpus" — accurate for the 27B and the private panels, but the small family's `hard-v1` / `documents-v1` / `devtools-v1` train partitions are published and hash-checked, so the right phrasing is per-artifact; (b) "One temperature … cannot reorder them" — see §7.1; (c) AutoJev "on pooled public data" — see §11.
10. **Concepts Stim's glossary lacks, which this document needed:** a word for *which splits of a corpus are actually fetchable* (published vs mirrored-private vs manifest-only), and a word for *the fitted-temperature provenance class* (in-distribution vs held-out-dataset fit). Both are load-bearing for Stim's reproducible-record bar, and both currently have to be spelled out in prose.
