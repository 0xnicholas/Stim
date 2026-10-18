# Artifact contract

An artifact is trained, evaluated, and served against the same input contract. This
document records the parts of that contract that are frozen. It currently covers content
rendering; artifact layout and manifest fields are added by the artifact-loader work.

## Rendering contract (v1, frozen)

A model never sees JSON. The wire `state`, each question's `instructions`, each `noul`
criteria description, each `choice` option description, and each `score` level text are
rendered to plain text before they enter a model input. The rendering is part of the
frozen artifact contract: training, evaluation, and serving must produce byte-identical
text for the same JSON value. Changing any rule below requires a new contract version
and a new artifact; it is not a serving-only change.

The one implementation is `stim_systemone.rendering` in
`packages/stim-systemone/src/stim_systemone/rendering.py`. It depends only on
`pydantic.JsonValue` and the Python standard library, never on torch, transformers,
FastAPI, or a tokenizer. Training and evaluation import the same functions; they must
not reimplement these rules.

### Accepted values

`state` and `instructions` accept every JSON value: string, object, array, number,
boolean, and null, nested to the limits of the JSON parser and Python's recursion limit.
Object key order is the order in the request's JSON text; it is never sorted.

### Rules

| Input | Output |
| --- | --- |
| `null` | the empty string |
| `true` / `false` | `true` / `false` |
| number | its Python `str()` form: `42`, `2.5`, `-3`, or scientific notation for very large or small floats |
| string | the string itself, with delimiter literals rewritten (see below) |
| empty object / empty array | the empty string |
| object | one line per key in request order: `key: <rendered value>` |
| array | one line per item in order: `- <rendered scalar>` |

Nesting adds two spaces per level under the parent key or dash. A value that renders to
an empty string leaves no trailing space: a null object value is `key:` and a null array
item is `-`. A container item in an array puts the dash on its own line and indents the
container beneath it. No JSON braces, brackets, or quotes are added; the only
punctuation in the output comes from caller strings or from the rules above.

An option is rendered by `render_option(key, description)`:

- a null or empty description renders as the option key alone;
- a one-line description renders as `key: <description>`;
- a multi-line description renders as `key:` followed by the rendered description
  indented two spaces.

`noul` uses this for the `false` and `true` descriptions under the keys `no` and `yes`.
`choice` uses it for every option name and its description, in request order. `score`
reuses it for level texts when that question type is added.

### Delimiter safety

The encoder will insert structural delimiters around state, question, option, and decide
segments. Caller text must never be able to forge one. Before any caller string or
object key is written into rendered text, every literal of the form
`<|name|>` (name: 1–64 characters from `A-Z`, `a-z`, `0-9`, `_`, `.`, `-`) is rewritten
to `<¦name¦>` by replacing the two pipe characters with U+00A6 BROKEN BAR. The rewrite
is deterministic and idempotent, preserves the visible name, and covers every Qwen
special-token delimiter the encoder will insert. When the encoder lands it must also
assert that delimiters appear only at the positions it inserted them.

Malformed fragments such as `<|name>` or `name|>` are left unchanged; they cannot form a
special token. Delimiters inside object keys are rewritten too, so a key cannot break
the `key: value` line into a forged structural boundary.

### Examples

```json
{
  "ticket": "Refund requested for a delayed order.",
  "customer": {"tier": "gold", "tags": ["vip", "delayed"], "urgent": true},
  "attempts": 2,
  "refund": null
}
```

renders to:

```text
ticket: Refund requested for a delayed order.
customer:
  tier: gold
  tags:
    - vip
    - delayed
  urgent: true
attempts: 2
refund:
```

`render_option("yes", {"signals": ["urgent", "angry"]})` renders to:

```text
yes:
  signals:
    - urgent
    - angry
```

`render_content("<|fim_prefix|>")` renders to `<¦fim_prefix¦>`.
