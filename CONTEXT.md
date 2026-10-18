# Stim

Stim is an open family of small decision models that speaks TypeSafe's System One protocol, with its weights, corpus, recipe, and evaluation published together as one record.

## Language

### The System One surface

**System One**:
TypeSafe's decision API: one state plus a batch of typed questions in a single request. Speaking this protocol is what makes a model drop-in for existing clients.
_Avoid_: using it for TypeSafe's model (that is Jev), or for Jev-style models generally.

**state**:
The content being evaluated, sent once per request and shared by every question in it.
_Avoid_: prompt, context, document

**question**:
One typed probe over the state. Four types exist — noul, choice, score, box — and questions in the same request are answered independently of one another.
_Avoid_: task, query, item

**noul**:
A yes/no question; its answer is the probability of "yes". The API returns no `confidence` field for it — the official docs suggest deriving `|2p − 1|`.

**choice**:
A question that picks one option from a caller-defined set of at most 255. Its answer carries `choice`, `probabilities`, and `confidence`.

**score**:
A question that rates the state against an ordered list of levels. Its answer's `score` is the probability-weighted mean across levels, so it can land between two levels.
_Avoid_: rating, grade

**box**:
The bounding-box question type — `{top, bottom, left, right}` plus optional confidence. Present in TypeSafe's docs and official SDK, but unimplemented in Kev, the most complete open implementation. The clearest known hole in any "fully compatible" claim.

**confidence**:
A value derived from an answer's probability distribution by published formulas, never emitted by the model. Choice: `(p_max − 1/K) / (1 − 1/K)`; score: `max(0, 1 − Σᵢpᵢ|i−m| / MAD_unif)`.
_Avoid_: certainty; probability (the raw distribution is the probability, the derived summary is the confidence)

**compatibility surface**:
The subset of System One that Stim commits to matching: `POST /v1/systemone` for noul/choice/score and `GET /v1/models`, verified through the official Python SDK. `box`, images/multipart, and the 262k danger-zone are outside it.
_Avoid_: "fully compatible" without naming the surface.

**compatibility claim**:
The scoped, published statement that Stim matches a named subset of System One, backed by a per-field matrix; "compatible" means what the matrix says.
_Avoid_: using "compatible" as an unqualified adjective.

**compatibility alias**:
A model name accepted only so an unconfigured client connects (`jev-latest`); it is never listed as a Stim member and never reported as the serving model.
_Avoid_: calling it a Stim model name.

### The ecosystem

**Jev**:
TypeSafe's hosted decision model — closed weights, reachable only through System One. The behaviour every Jev-style model imitates.
_Avoid_: using "Jev" for open models, or for the protocol itself.

**Jev-style model**:
A small model whose entire output is a probability distribution over a caller's options, rather than free text.
_Avoid_: classifier, LLM, agent, decision engine

**Kev**:
`jaredpalmer/kev` — the open reference implementation of a Jev-style family: Apache-2.0 code and weights, a public recipe, but a private corpus and private held-out panels. The bar Stim is measured against, and the source of the recipe and evaluation conventions Stim adopts.
_Avoid_: calling it "the open Jev" (it is a sibling, not the origin)

**AutoJev**:
The other open reproduction of a Jev-style model — full-weight SFT on pooled public data. Evidence that the recipe alone is table stakes.

**RLCD**:
Jev's published training route — described as a third path alongside RLHF and RLVR — whose algorithm is unpublished and has no public implementation. Stim v1 trains with SFT only.
_Avoid_: using "RLCD" for any RL or preference-optimisation method, or as a synonym for SFT.

**frozen suite**:
An evaluation panel whose questions, weights, and scoring code are fixed and published, so scores stay comparable across models and over time.
_Avoid_: benchmark (any test set is a benchmark; a frozen suite is one we commit not to change)

**Decision Index**, **JevBench**:
The public leaderboards that rank Jev-style models. Both mix public and sealed material, and their versions drift — scores are not comparable across versions.

### Stim

**Stim**:
An open family of Jev-style decision models speaking System One, publishing weights, corpus, recipe, and evaluation as one record. Its bet: the first such family a stranger can retrain and re-verify.
_Avoid_: "the open Jev", "a Jev clone"

**member**:
One size in the Stim family (e.g. 0.8B, 4B, 9B), with its own checkpoint, adapter, and temperature, sharing the family's corpus and recipe skeleton.
_Avoid_: "model" when you mean one member (the family is also a model).

**family release**:
One versioned release containing every member of the family, sharing one recipe, corpus, and evaluation protocol; a member may take a patch release only when those shared inputs are unchanged.
_Avoid_: "model release" when you mean the whole family.

**language variant**:
An independently versioned Stim line for one language, sharing the recipe skeleton but not the English family's corpus, evaluation, or SemVer; the first is `zh-v1.0.0`.
_Avoid_: family minor, translated release.

**reproducible record**:
Stim's central claim, stated as an acceptance bar — the corpus published as a downloadable, licence-clear first-class artifact; recipe and scripts sufficient for a stranger to retrain to the same magnitude on comparable compute; the evaluation rerunnable by a third party to the published reference scores. Regenerating the corpus from its generation prompts is transparency, not part of the bar.

**dual-track evaluation**:
Running two evaluation tracks at once — the public comparability track, for comparability with the ecosystem, and the verification track, for verifiability.

**public comparability track**:
The dual-track half that runs existing public frozen suites (Kev's public suites, later Decision Index and JevBench) to compare Stim against the ecosystem. It is not the track that carries the reproducibility promise.
_Avoid_: calling it the verification track, or treating its sealed/private portions as rerunnable.

**verification track**:
The dual-track half that Stim builds and publishes in full — item-level data, provenance, manifests, and evaluation code — so a third party can rerun it and reproduce the reference scores. It is the only track that carries the "evaluation can be independently rerun" promise.
_Avoid_: calling it the public track, or using it for leaderboard comparability.

**validated context**:
The longest state length a member's published evaluation actually supports, as distinct from the serving context limit. A descriptive claim about measured accuracy at length, not an equivalence proof.
_Avoid_: using it for the serving limit, or claiming the maximum context the model accepts.

**chance-corrected skill**:
Accuracy rescaled so 0 is random guessing and 100 is perfect, using each suite's recorded baseline; used to compare and aggregate suites, never as a single headline score.
_Avoid_: using "skill" for raw accuracy or for a leaderboard rank.

**label-shuffled control**:
A control model trained on the same corpus with answer labels permuted within question type and option/level count, used to show a model's skill is not memorised label marginals.
_Avoid_: random baseline, chance baseline.

**clean-room compatibility**:
Deriving System One compatibility only from public documentation, the official MIT adapter, and published write-ups — never from probing the service or from its outputs.
_Avoid_: "reverse-engineered compatibility"

**permissive-only corpus**:
A corpus restricted to sources whose licences permit unconditional redistribution; share-alike and unknown-licence sources are excluded, with synthetic data filling the gap.

**record**:
One state plus the batch of questions asked about it, with their answers. The unit of corpus splitting, dedupe, and bootstrap clustering; sibling questions on one state move together.
_Avoid_: row, sample; "question" when you mean the whole record.

**teacher**:
A model that generates training labels for synthetic decision records. Stim uses open-weight teachers only, pinned by version; closed models may judge or filter but never label.
_Avoid_: using "teacher" for a model that only evaluates or filters (that is a judge), or for Jev.

**judge**:
A model that filters or scores generated data without supplying the training label. Closed frontier models may serve as judges; never as teachers.
_Avoid_: using "judge" for the teacher.

**temperature calibration**:
Fitting one scalar temperature so a model's confidences are honest — fitted on a held-out pool, never in-distribution, which was Kev's measured failure mode (ECE 0.059 → 0.0085 when fitted properly). One temperature can rescale confidences but cannot reorder them.
_Avoid_: calibration (say temperature calibration when you mean the scalar)

**reproducer cost**:
What a stranger spends — GPU-hours and wall-clock — to retrain Stim from the published recipe alone and reach the same magnitude. The only externally checkable face of training-level reproducibility, so Stim fixes a ceiling for it up front and lets that ceiling bind family size.
_Avoid_: "reproduction cost" (that is Stim's own spend, not the reproducer's)
