<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Pipeline · Stage 8 — Verifier `[DRAFT]`

> **Input:** `files/processed_proof_graph/*.txt` (chapters + `global_theorem_list.txt` + `external_theorems.txt`) + `files/GL_binaries/GL_binary_<Tag>.json` + `files/config/ConfigVisu.json`.
> **Output:** stdout — a ~41-line tally of per-tag success/failure counts + a final "airtight" or "FAILED" line.
> **Owner:** `verifier.py` (~7100 lines).
> **Entry:** `main` at [`verifier.py`](../../verifier.py), which parses CLI arguments (positional `base_dir`, repeatable `--include-globals PATH`), calls `run_verifier(base_dir, extra_global_lists)` at [`verifier.py`](../../verifier.py), then `print_report(state)` at [`verifier.py`](../../verifier.py).
>
> **CLI surface (D-35).**
> - `python verifier.py` — verify the main pipeline output (`files/processed_proof_graph/`, the historical default).
> - `python verifier.py files/incubator/processed_proof_graph --include-globals files/processed_proof_graph/global_theorem_list.txt` — verify the incubator output, unioning the main pipeline's global theorem registry into the incubator's so cross-batch theorem citations resolve.
> - `--include-globals` is repeatable; entries are unioned in order with local-batch entries taking precedence on key collisions.

---

## What this stage does

The verifier is the external proof checker. It consumes the same data the customer-facing HTML export consumes (the *processed* proof graph), and for each chapter, re-checks every proof step against the claim of its tag.

**Core principle: independence.** The verifier has its own copies of every algorithm it needs — expression parsing, disintegration, normalisation, reformulation checks, digit-arg/immutable-arg computation. It does **not** import from `expression_utils.py` or from any prover code. This separation is the entire point: if the prover and the verifier share a bug, they can't catch each other.

Invariant [I-16](../30_invariants.md#i-16) is absolute: never weaken a verifier check to "pass a test". A failure is a real bug, not a false positive.

---

## Output — the ~41-line tally

`print_report(state)` at [`verifier.py`](../../verifier.py) emits a fixed-format report:

```
theorem goal reached                         success N, failure M
implication                                  success N, failure M
expansion                                    success N, failure M
...
(31 TAG_CHECKERS tags + the meta counters that recorded at least one event)
...
(final summary line — one of:)
   N checks, 0 failures — airtight.
   Verifier: N checks, M FAILED.
```

Column width: `W = 44`. The tag name is left-justified to 44 chars, followed by `success <count>, failure <count>`.

Row composition on a typical run: 1 (`theorem goal reached`) + 31 (TAG_CHECKERS in dispatch order) + ~9 meta counters that were hit (`self-reference`, `anchor handling uniqueness`, `anchor handling trace`, `contradiction trace`, `vacuous truth trace`, `origin`, `definition set consistency`, `origin chain termination`, `operator registry consistency`) + 1 headline ≈ 42 rows. Meta counters not hit on a given run are omitted, so the row count drifts with chapter shape.

**On a clean release run, every row reports `failure 0`.** In-flight FTA-ladder branches (e.g. ) regularly have non-zero failures — these indicate the rung currently being worked on, not regressions against the release baseline. for the reference clean counts.

---

## TAG_CHECKERS — the registry

At [`verifier.py`](../../verifier.py). A dictionary mapping each tag string to its checker function. 31 registry entries, 31 unique tags (one entry per tag; `or branch proven` and `or branch assumption` added by [D-35](../40_decisions.md#d-35); `compilation` added per [D-76](../40_decisions.md#d-76); the historical `equalize variable` alias was removed since only `multiplied from` was ever emitted).

Full table:

| Tag | Checker | File |
|---|---|---|
| `implication` | `check_implication` | [`verifier.py`](../../verifier.py) |
| `expansion` | `check_expansion` | [`verifier.py`](../../verifier.py) |
| `compilation` | `check_compilation` | [`verifier.py`](../../verifier.py) |
| `disintegration` | `check_disintegration` | [`verifier.py`](../../verifier.py) |
| `task formulation` | `check_task_formulation` | [`verifier.py`](../../verifier.py) |
| `equality1` | `check_equality1` | [`verifier.py`](../../verifier.py) |
| `equality2` | `check_equality2` | [`verifier.py`](../../verifier.py) |
| `symmetry of equality` | `check_symmetry_of_equality` | [`verifier.py`](../../verifier.py) |
| `symmetry of inequality` | `check_symmetry_of_inequality` | [`verifier.py`](../../verifier.py) |
| `recursion` | `check_recursion` | [`verifier.py`](../../verifier.py) |
| `theorem` | `check_theorem_tag` | [`verifier.py`](../../verifier.py) |
| `reformulation for integration and` | `check_reformulation_for_integration_and` | [`verifier.py`](../../verifier.py) |
| `reformulation for integration >[bound]` | `check_reformulation_for_integration_bound` | [`verifier.py`](../../verifier.py) |
| `reformulation for integration >[]` | `check_reformulation_for_integration_empty` | [`verifier.py`](../../verifier.py) |
| `expansion for integration` | `check_expansion_for_integration` | [`verifier.py`](../../verifier.py) |
| `premise element` | `check_premise_element` | [`verifier.py`](../../verifier.py) |
| `validity name` | `check_validity_name` | [`verifier.py`](../../verifier.py) |
| `anchor handling` | `check_anchor_handling` | [`verifier.py`](../../verifier.py) |
| `or theorem` | `check_or_theorem` | [`verifier.py`](../../verifier.py) |
| `reformulated from` | `check_reformulated_from` | [`verifier.py`](../../verifier.py) |
| `incubator back reformulation` | `check_incubator_back_reformulation` | [`verifier.py`](../../verifier.py) |
| `externally provided theorem` | `check_externally_provided_theorem` | [`verifier.py`](../../verifier.py) |
| `variable copy` | `check_variable_copy` | [`verifier.py`](../../verifier.py) |
| `multiplied from` | `check_equalize_variable` | [`verifier.py`](../../verifier.py) |
| `contradiction` | `check_contradiction` | [`verifier.py`](../../verifier.py) |
| `or disintegration` | `check_or_disintegration` | [`verifier.py`](../../verifier.py) |
| `or convergence` | `check_or_convergence` | [`verifier.py`](../../verifier.py) |
| `or branch proven` | `check_or_branch_proven` | [`verifier.py`](../../verifier.py) |
| `or branch assumption` | `check_or_branch_assumption` | [`verifier.py`](../../verifier.py) |
| `vacuous truth` | `check_vacuous_truth` | [`verifier.py`](../../verifier.py) |

**Cross-doc check.** `TAG_CHECKERS` has **30 entries, 30 unique tags** — verified by `python -c "from verifier import TAG_CHECKERS; print(len(TAG_CHECKERS))"`. The historical `equalize variable` alias was retired (only `multiplied from` was ever emitted), so there is no shared-checker entry today; `multiplied from` → `check_equalize_variable` is the only naming asymmetry, and it occupies one entry. The two `or branch proven` / `or branch assumption` entries were promoted from "claimed retired" to first-class by [D-35](../40_decisions.md#d-35).

### Validity matching on dep cells

Each checker that compares a row's dep validity to a stored validity uses **exact `==`** equality where possible. Per `D-56`, `buildStack` emits every chapter cell (`row[1]` and every dep cell) at the **closest-to-`main` ancestor of the requested validity for which an origin entry exists in the emitting LB's `exprOriginMap`**, so a row's `row[1]` and each cited dep's recorded validity are already at the same lifted scope by construction. The `_ns_matches_or_strict_prefix` helper at `verifier.py` is consumed by **three checkers**: `check_implication` (the validity-stack deposit pair-check), `check_equality1`, and `check_equality2` — the latter two because the source/target validity relationship is on the *expression* side of an equivalence-class rewrite, not the rule-application side. The helper requires `tgt_ns == src_ns` or `tgt_ns == src_ns + "_boundary_" + …` (separator-aware ancestor); a bare byte-prefix that does not respect the `_boundary_` separator is rejected.

**Per-tag structural contracts.** The exact `rest`-field shape each checker expects is documented in [`20_core_concepts/08_proof_tags.md`](../20_core_concepts/08_proof_tags.md). Key examples to watch when authoring new origins:

- `equality1` — `rest[0]=source_expr`, `rest[1]=source_ns`, interleaved `(eq, ns)` pairs thereafter. Minimum `len(rest)=4`. Source must be in the row's namespace. See [`08_proof_tags.md#equality1`](../20_core_concepts/08_proof_tags.md#equality1).
- Every tag's `rest` alternates `(expr, ns)` — odd-length `rest` is silent-reject by the length check in every checker. See the [row-shape recap](../20_core_concepts/08_proof_tags.md#row-shape-recap).
- An unknown origin tag (not in `TAG_CHECKERS`) is treated as 1 failure per occurrence — `<unknown:tag>` row in the verifier summary. Reusing an existing tag whose structural contract fits is safer than adding a new tag (which requires `verifier.py` modification — see [I-16](../30_invariants.md#i-16)).

**Non-checker (meta) categories** tracked and validated outside the `TAG_CHECKERS` dispatch:

- `self-reference` — error counter; increments when a chapter cites its own theorem as a justification (a `theorem`-tagged row whose expression equals the chapter's target).
- `anchor handling uniqueness` — at most one `anchor handling` row per chapter. Multiple rows record failures.
- `anchor handling trace` — per-step chain of `_copy` variable rewrites produced by `anchor handling`. Every user row that mentions a `_copy` var in its rest sources must trace back to the chapter's `anchor handling` row.
- `contradiction trace` — chapter-local origin-chain walk; at least one of the two contradicting ingredients in a `contradiction` row must trace back to a `task formulation` row whose expression is the cited `cleanOp`.
- `vacuous truth trace` — chapter-local origin-chain walk added by commit. At least one contradicting ingredient in a `vacuous truth` row must trace back to the recursion-hypothesis cited in `rest[4..5]`. Counter reports success/failure separately from `vacuous truth` itself; a failing trace means the LB's vacuous-truth claim is unsound even if the `check_vacuous_truth` structural check passes. Pairs with the prover-side `mb.level`-in-ingredients gate to prevent induction-step LBs from collapsing on anchor-only contradictions. See [`08_proof_tags.md#vacuous-truth`](../20_core_concepts/08_proof_tags.md#vacuous-truth).
- `origin` — every dependency cited in a row's `rest` (at even indices) must resolve: either to a chapter-local left-side expression, an `_integration_goal`-postfixed placeholder, an exempt tag, or (for `implication` / `multiplied from` / `reformulated from` at `rest[0]`) the global theorem registry / external theorems. Full algorithm: [`origin` meta-check](#origin-meta-check).
- `definition set consistency` — D-41 per-row variable-port type-consistency check. Every expression in every row is parsed; shared variable names across child sub-trees must agree on type label. See [`definition set consistency` meta-check](#definition-set-consistency-meta-check).
- `origin chain termination` — chapter-local cycle detection on the origin graph. Every (expression, namespace) node on a back-edge is recorded as a failure. See [`origin chain termination`](#origin-chain-termination).
- `operator registry consistency` — I-23: every spontaneous compact-operator name (`implication<N>`, `existence<N>`, `or<N>`, `and<N>`) must encode the same operator across every binary that carries it. See [Operator-registry consistency (I-23)](#operator-registry-consistency-i-23).

---

## Key helper algorithms

Independent copies of algorithms the verifier needs. These must stay in sync with the C++ side *semantically*, but are not allowed to *call* any C++ code or share implementation.

| Function | File | Role |
|---|---|---|
| `disintegrate_implication_head` | [`verifier.py`](../../verifier.py) | Peel all `(>[vars](premise)(body))` layers; return innermost conclusion as-is (no renaming). |
| `disintegrate_implication_full` | [`verifier.py`](../../verifier.py) | Peel all layers; return `(premises_list, head)`. |
| `_normalize_expr_list` | [`verifier.py`](../../verifier.py) | α-rename v-variables across a list of expressions by order of first appearance in arguments. |
| `_alpha_canonicalize_bound_vars` | [`verifier.py`](../../verifier.py) | Canonicalize every name introduced inside a `>[...]` binder to `b1, b2, …` in declaration order. Used by the origin check (D-35) to compare a chapter's `rest[0]` rule against the global theorem registry across batches when bound variables were renamed differently by `process_proof_graphs.py` (e.g. incubator chapter has `>[i2]` where Peano global has `>[v1]`). Stronger than `_normalize_expr_list`, which renames `[vV]\d+` (lowercase v digits + uppercase V sets, both case-folded into the same canonical sequence — sandbox `var_typing` 2026-05-04). |
| `_check_reformulation` | [`verifier.py`](../../verifier.py) | Verify `target` is a valid reformulation — expand existence head via GL binary into left+right elements, check `definedSet`, permute non-anchor premises, normalise, compare. |
| `_find_digit_args` | [`verifier.py`](../../verifier.py) | Replicates the C++ `findDigitArgs`: collect `inputIndices`-arg positions, subtract anchor args, subtract output args. |
| `_find_immutable_args` | [`verifier.py`](../../verifier.py) | Replicates the C++ `findImmutableArgs`: seed from digit-args minus induction variable, propagate through outputs where all inputs are immutable. |
| `_replace_arg_safe` | [`verifier.py`](../../verifier.py) | Argument-level replacement with lookaround regex — prevents `v1` from matching inside `v10`. |
| `_extract_args` | [`verifier.py`](../../verifier.py) | Extract args from the **outermost** `[…]` group of an expression (e.g. `(in3[a,b,c,+])` → `[a,b,c,+]`). Caller-side default — used by 30+ checkers that operate on atomic expression arguments. Misses nested `[…]` content by design; for compound source expressions (negated implications etc.) where args may be buried inside an outer binder list, use `_extract_all_args` instead. |
| `_extract_all_args` | [`verifier.py`](../../verifier.py) | Recursive variant — flattens **every** `[…]` group via `re.findall` and splits each by comma. Used only by `check_variable_copy`'s trace-back filter so the walker can follow citation edges through compound sources like `!(>[v6](in[v6,N])!(in2[v6,i1_copy,s]))` (where `_extract_args` would return only `[v6]` and the nested `i1_copy` would be invisible). Added by commit to fix a chapter-85 trace-back failure. |

---

## Top-level flow

Function: `run_verifier(base_dir)` at [`verifier.py`](../../verifier.py). Steps (in order):

1. **Load global theorem list.** `load_global_theorem_list(base_dir)` populates `state.global_theorems` + `state.global_theorem_list`. Each row is `(expression, method, reference)`.
2. **Load GL binaries.** `load_gl_binaries(binaries_dir)` — looks first in `os.path.dirname(base_dir)/GL_binaries`, falls back to `files/GL_binaries`. Silently uses fallback — no warning. Since [D-54](../40_decisions.md#d-54) the incubator-side fallback is the only path taken on disk: the duplicate `files/incubator/GL_binaries/` was removed and all per-batch dictionaries — incubator and main alike — live in `files/GL_binaries/`.
3. **Load external theorems.** Read `external_theorems.txt`. Each line becomes an entry in `state.external_theorems` (a set).
4. **Enumerate chapter files.** Glob all `.txt` files except `global_theorem_list.txt` and `external_theorems.txt`. Sort via `chapter_sort_key` ([`verifier.py`](../../verifier.py)) which extracts the leading numeric prefix.
5. **Build chapter → theorem mapping.** `build_chapter_theorem_map` at [`verifier.py`](../../verifier.py). Maps each filename to its theorem tuple. **Enforces induction triads** — `<N>_induction_typing` + `<N+1>_check_zero` + `<N+2>_check_induction_condition` must appear in strict filename order. Violation raises `AssertionError`.
6. **Iterate chapters.** For each chapter file `cf`:
 - Parse lines into `ProofLine` records via `parse_chapter_file(filepath)` at [`verifier.py`](../../verifier.py).
 - Extract `chapter_type` from filename.
 - Fetch theorem via `chapter_thm_map.get(cf)`.
 - Call `verify_chapter(cf, lines, chapter_type, state, chapter_thm)` at [`verifier.py`](../../verifier.py).
7. **Emit report** via `print_report(state)` at [`verifier.py`](../../verifier.py).

### `verify_chapter` — per-chapter pipeline

Stages (in order within a single chapter, all inside `verify_chapter` at [`verifier.py`](../../verifier.py)):

1. `check_theorem_goal_reached` → `state.goal_reached`.
2. Set transient chapter context (`current_chapter_thm`, `current_chapter_type`, `current_gl_binary`, `current_resolved_defsets`) via the **premise-anchor binary resolution** described below.
3. Self-reference check → `state.counter_for("self-reference")`.
4. Anchor-handling uniqueness → `state.counter_for("anchor handling uniqueness")`.
5. Anchor-handling trace → `state.counter_for("anchor handling trace")`.
6. Contradiction trace → `state.counter_for("contradiction trace")`.
7. Vacuous-truth trace → `state.counter_for("vacuous truth trace")`.
8. **Dispatch each row to its tag checker** via `TAG_CHECKERS[row.tag](line, lines, state)`. Unknown tags record under `state.counter_for("<unknown:{tag}>")`.
9. General `origin` meta-check → `state.counter_for("origin")`.
10. `definition set consistency` meta-check (D-41) → `state.counter_for("definition set consistency")`.
11. `origin chain termination` cycle detection → `state.counter_for("origin chain termination")`.

---

## Self-reference detection

Inside `verify_chapter` at [`verifier.py`](../../verifier.py). After loading `chapter_thm`:

```python
if chapter_thm is not None:
    thm_expr = chapter_thm[0]
    for line in lines:
        if line.tag == "theorem" and line.expression == thm_expr:
            state.counter_for("self-reference").record(False)
```

Rationale: a theorem cannot be its own justification. An emitted `theorem`-tagged row whose expression matches the chapter's target is axiomatic self-reference — and therefore a failure.

---

## External theorems — handling

Loaded inside `run_verifier` at [`verifier.py`](../../verifier.py) into `state.external_theorems` (set).

The `check_externally_provided_theorem` checker at [`verifier.py`](../../verifier.py) requires:

- `line.namespace == "main"`;
- `line.expression` is a direct member of `state.external_theorems` (raw or pre-renamed form), optionally after a `w → v` revert on theorem-anchor citations.

Failure → False recorded.

External theorems are also consulted as valid sources during origin validation (in the `i == 0` branch of the `origin` meta-check inside `verify_chapter`) — for tags like `implication`, `multiplied from`, `reformulated from` that cite a source theorem.

**External theorems are not blanket-axiomatic.** They must pass the same structural checks (reformulation shape) as internal theorems. They are "external" only in the sense that their own proof graphs are not included in the verifier's scope.

---

## Chapter-file naming — the load-bearing assertion

The filename pattern is:

```
<N>_<type>.txt
```

where `<type>` is one of the known chapter types: `direct_proof`, `check_zero`, `check_induction_condition`, `induction_typing`, `reformulated_statement`, `back_reformulated_statement`, `or_theorem`.

`build_chapter_theorem_map` ([`verifier.py`](../../verifier.py)) enforces:

- Induction chapter triples (`induction_typing` + `check_zero` + `check_induction_condition`) must be consecutive and in exact order.
- Filename's leading-number prefix must parse as an integer.
- Filename's suffix must be a known type.

Any violation raises `AssertionError`, terminating the verifier with a stack trace.

---

## Loading GL binaries

```python
binaries_dir = find_gl_binaries(base_dir)   # tries os.path.dirname(base_dir)/GL_binaries, falls back to files/GL_binaries
state.gl_binaries = load_gl_binaries(binaries_dir)
```

Since [D-54](../40_decisions.md#d-54) the auto-detect for the incubator-side `base_dir` (`files/incubator/processed_proof_graph`) always falls back to `files/GL_binaries/` — the previously sibling `files/incubator/GL_binaries/` directory was removed when the C++ writer was redirected to a single canonical path.

`load_gl_binaries` walks `sorted(os.listdir(binaries_dir))` rather than raw `os.listdir`, so the resulting `state.gl_binaries` insertion order (and therefore every downstream first-match-wins iteration over it) is the same on NTFS, ext4-with-dir_index, ext4-without, tmpfs, overlayfs and APFS. Without the sort, the chapter-context filter described below was load-bearing on filesystems that happened to return entries alphabetically, but degenerate on filesystems that did not — same input bytes, different verdict, contradicting the "deterministic computer architecture" claim the project rests on. The sort makes the load order a function of the filename set alone.

The `state.gl_binaries` map is consulted by:

- `check_expansion` — to expand a named expression into its compiled-structure form.
- `_check_reformulation` — to expand an existence head via the binary into left+right elements.
- `check_anchor_handling` — to match anchor-slot positions.

The verifier selects the correct binary per chapter via the **premise anchor** — the `Anchor<Tag>` substring inside the LEFT side of the chapter's outer-implication theorem, which is the world the proof's *assumptions* live in. Implemented inside `verify_chapter`:

```python
premises, _head = disintegrate_implication_full(thm_expr)
premise_expr = premises[0] if premises else thm_expr  # naked-anchor fallback
m = re.search(r'Anchor([A-Za-z0-9_]+)', premise_expr)
if m is not None:
    tag = m.group(1)
    state.current_gl_binary = state.gl_binaries.get(tag)
```

Premise-anchor extraction makes the binding a pure function of the theorem expression. No iteration over `gl_binaries`. No first-match-wins. No filesystem-order dependence. For a cross-anchor connection theorem like chapter 103's

```text
(>[N,i0,s,+,*,i1](AnchorGauss[N,i0,s,+,*,i1,i2,id])(AnchorPeano[N,i0,s,+,*,i1]))
```

the premise (`(AnchorGauss[...])`) deterministically resolves to `tag == "Gauss"`, regardless of which order `gl_binaries.items` happens to yield. Silent fallback to `None` survives for the rare naked-anchor case (no implication structure, no premise to extract from) — checkers that require a GL binary then fail with messages that do not always make the root cause obvious.

### Cross-anchor binding: premise wins

Before the premise-anchor rewrite (, 2026-05-12) `verify_chapter` iterated `state.gl_binaries.items` and broke on the first tag whose `Anchor<Tag>` substring appeared anywhere in the theorem expression. For single-anchor chapters this was harmless. For **cross-anchor connection theorems** like chapter 103's

```text
(>[N,i0,s,+,*,i1](AnchorGauss[N,i0,s,+,*,i1,i2,id])(AnchorPeano[N,i0,s,+,*,i1]))
```

both `AnchorGauss` and `AnchorPeano` appear, and the iteration race decided which one won. The race was masked by NTFS alphabetical listdir order on the developer's box (`Gauss` < `Peano`, so Gauss won), but a Linux container with non-alphabetical directory-entry order (claude.ai sandbox, 2026-05-12) put `Peano` ahead of `Gauss` and produced 3 verifier failures (`expansion`: 1, `disintegration`: 2) on the *same* input bytes — Peano was bound as `current_gl_binary` on a chapter whose AND-compound expansion cited `AnchorGauss`, the Peano binary had no `AnchorGauss` entry, lookups failed.

Premise-anchor extraction collapses this entire failure mode: the premise of chapter 103's theorem is `(AnchorGauss[...])`, so `tag == "Gauss"` independent of dict iteration order. The previous `AnchorIncubator`-substring filter (D-54 era, restricted iteration to `Incubator`-prefixed tags when `"AnchorIncubator" in thm_expr`) is now redundant in `verify_chapter` — premise-anchor extraction picks the right binary directly — and is retained only in `binaries_for_chapter`'s fallback path as defense-in-depth.

### `binaries_for_chapter` fallback

Two ancillary sites still iterate `state.gl_binaries.items` for operator lookups in cases where `current_gl_binary` is `None` (rare — only when premise extraction fails on a naked-anchor or otherwise unbound chapter). They both retain the historical `AnchorIncubator` substring filter:

1. `binaries_for_chapter` — returns `[current_gl_binary]` when set (the common path now), otherwise iterates with the Incubator filter applied.
2. `_check_reformulation`'s D-35 fallback — same Incubator-prefix scan for the `tag == "Incubator"` existence-head case.

```python
def binaries_for_chapter(self) -> list:
    if self.current_gl_binary is not None:
        return [self.current_gl_binary]
    thm = self.current_chapter_thm
    if thm is not None and "AnchorIncubator" in thm[0]:
        return [b for tag, b in self.gl_binaries.items()
                if tag.startswith("Incubator")]
    return list(self.gl_binaries.values())
```

Combined with `load_gl_binaries`'s `sorted(os.listdir(...))` (also 2026-05-12), these residual fallback paths are deterministic on every supported filesystem.

### Operator-registry consistency (I-23)

After loading every `GL_binary_<Tag>.json`, `run_verifier` calls `check_operator_registry_consistency(state)` (, 2026-05-12). It enforces I-23 — *spontaneous compact operator names are stable across batches* — by walking the closure of cross-binary name presence:

1. Bucket every spontaneous-compact entry (categories `implication` / `existence` / `or` / `and`; atomic and anchor entries skipped) across all loaded binaries: `by_name[name] = {tag: entry}`.
2. For each name appearing in ≥ 2 binaries, recursively verify the definition closure on the common tag set:
 - **Surface match:** `(category, signature, arity, elements)` byte-identical across every entry.
 - **Closure:** every spontaneous-compact name CITED inside `elements` (extracted via `re.findall(r'\(([A-Za-z_]\w*)\[', elem)`) is checked recursively on the same tag set. Memoised on `(name, frozenset(tags))` so shared subgraphs cost once and citation cycles terminate.
3. Records one increment per top-level name into `state.tag_counters["operator registry consistency"]` (`success` on transitive consistency, `failure` with a stderr report otherwise).

**Failure mode this catches.** On a clean-room run (`CLEAN_RUN=True` wipes `files/GL_binaries/`), IncubatorPeano runs first with an empty shared registry → `existenceCounter = 0` → allocates `existence0..2` for its own splitNKs. `_merge_into_shared` skips Incubator tags by design (D-54), so the shared registry stays empty when Peano main starts. Peano also begins cold → allocates its own `existence0..2` against *different* splitNKs. Disk-side, `GL_binary_IncubatorPeano.json` carries `existence2[u_1..u_8]` and `GL_binary_Peano.json` carries `existence2[u_1..u_3]` — same name, two semantically different operators. The trap inside `prover.hpp::excludeRepetitions` (captured 2026-05-12) recorded both allocation events with `existence2 NOT in compiledExpressions before this event` on both sides, confirming the cold-start mechanism.

Per I-16 / [I-19](../30_invariants.md#i-19) the check is sacred: failures here are real bugs that point at the C++ allocator (or its Python orchestration), and the right remediation is to make name allocation deterministic across batches — for example, seed every batch's spontaneous counters from a shared registry that incubator batches also contribute to — not to soften the verifier check.

---

## `origin` meta-check

In addition to the per-tag checkers in `TAG_CHECKERS`, the verifier runs a single chapter-level meta-check that ensures every dependency cited by a chapter row is actually derivable. This is the `origin` tag in the per-tag tally — it is a *check name*, not a producer-side proof tag. No row in any chapter is ever emitted with `tag == "origin"`; the check counts go up purely from this meta-pass.

### What it checks

[`verifier.py`](../../verifier.py). After all per-tag checkers have run on a chapter:

```python
_ORIGIN_EXEMPT_TAGS = {
    "incubator back reformulation",
    "contradiction",
    "or disintegration",
    "or convergence",
    "or branch proven",
    "or branch assumption",
    "or theorem",
}
originated = {line.expression for line in lines}     # left-hand side of every row in this chapter
for line in lines:
    if line.tag in _ORIGIN_EXEMPT_TAGS:
        continue
    for i in range(0, len(line.rest), 2):            # rest is alternating expr/validity pairs
        dep = line.rest[i]
        if dep in originated:
            continue
        if dep.endswith("_integration_goal"):
            continue
        # Special i==0 path for implication-style rules
        if i == 0 and line.tag in ("implication", "multiplied from", "reformulated from"):
            norm_dep = _normalize_expr_list([dep])
            alpha_dep = _alpha_canonicalize_bound_vars(dep)
            # w/W -> v/V revert for cited theorem-anchor implications.
            # Pass 3 in process_proof_graphs.py renames non-anchor inner
            # bvars from v/V to w/W in cells at column >= 3 (citation
            # form). The registry stores v/V; reverting before the
            # membership check keeps the fast exact-string path intact.
            dep_v = (_revert_w_to_v_in_theorem_citation(dep)
                     if _is_theorem_anchor_impl_local(dep) else dep)
            registry = state.global_theorems.keys() | state.external_theorems
            found = (
                dep in state.global_theorems
                or dep in state.external_theorems
                or dep_v in state.global_theorems            # citation-form fast path
                or dep_v in state.external_theorems          # citation-form fast path
                or any(_normalize_expr_list([gt]) == norm_dep for gt in registry)
                or any(_alpha_canonicalize_bound_vars(gt) == alpha_dep for gt in registry)
            )
            state.counter_for("origin").record(found)
            continue
        # Generic origin failure: dep neither in chapter nor in registry, not _integration_goal, not i==0 implication-style
        state.counter_for("origin").record(False)
```

For every row's `rest[]` field iterated at even indices (the expression component of each `(expr, validity)` pair), the check asks: **is this dependency derivable?**

- **In-chapter derivation.** If `dep` appears as the left-hand side of any row in this chapter (`dep in originated`), pass — the chapter justifies it locally.
- **Integration-goal placeholder.** If `dep` ends with `_integration_goal`, pass — these are placeholders inserted by `reformulation for integration` flow, deliberately underived.
- **Cross-chapter rule citation (the `i==0` path).** When the row's tag is `implication` / `multiplied from` / `reformulated from` and we're looking at `rest[0]`, the dep is the *rule* applied. Rules can be globally proved, externally provided, or alpha-canonical equivalents of either. Pass if any of those four match. Fail otherwise.
- **All other unmatched deps.** Fail — record `False` for the `origin` counter.

### What alpha-canonicalisation does (and doesn't)

`_alpha_canonicalize_bound_vars(expr)` ([`verifier.py`](../../verifier.py)) renames every `>[a,b,…]` bound variable in declaration order to a canonical sequence (`v_1`, `v_2`, …). This makes the rule `(>[N,i0,s](AnchorPeano[N,i0,s,…])(>[v1](in[v1,N])(=[v1,i0])))` and its alpha-equivalent `(>[a,b,c](AnchorPeano[a,b,c,…])(>[d](in[d,a])(=[d,b])))` produce the same canonical string.

What it does **not** do:

- It does **not** rewrite compiled structural operators (`existence2[1,7,3]` stays `existence2[1,7,3]`; never expanded to `!(>[8](in[8,1])!(in2[8,7,3]))` or vice versa).
- It does **not** consult `state.gl_binaries` to perform compact↔expanded conversion.
- It does **not** normalise free variables (only bound).

This was the substrate of the [D-40](../40_decisions.md#d-40) bug. Chapter rows cite rules in compact form (with `existence2`); the externals seed used to come from `theorems.txt` (expanded form, no `existence2`); alpha-canon couldn't bridge the form gap. The fix was to switch the seed source to `compiled_theorems.txt` (compact form), so registry entries are now in the same form chapter rows cite. The `origin` check itself is unchanged.

### `w → v` revert for cited theorem-anchor implications

`process_proof_graphs.py` Pass 3 (see [`06_process_proof_graph.md`](06_process_proof_graph.md#pass-3-vw-rename-of-cited-theorem-anchor-implications)) renames non-anchor inner bvars in cited theorem-anchor implications from `v / V` to `w / W` (case-preserving letter swap, digit preserved). The registry (`state.global_theorems`, `state.external_theorems`) stores the `v / V` form. To keep the fast-path exact-string membership test intact:

- `_is_theorem_anchor_impl_local(expr)` in [`verifier.py`](../../verifier.py) detects whether a citation has the theorem-anchor shape `(>[…]( Anchor<Tag>[…])body)`.
- `_revert_w_to_v_in_theorem_citation(expr)` applies the symmetric inverse swap (`w<N> → v<N>`, `W<N> → V<N>`) on every `[wW]\d+` argument-position token.

The revert is invoked at three sites:

1. The origin meta-check (above), at `rest[0]` for `implication` / `multiplied from` / `reformulated from` rows.
2. `check_theorem_tag` — defensive only; HEAD column is never `v→w`-swapped by the processor, but the revert path catches a future regression.
3. `check_externally_provided_theorem` — same defensive shape against `state.external_theorems`.

`_normalize_expr_list` (already widened to `[vVwW]\d+`) provides a slower fallback that also matches; the explicit revert preserves the fast path and documents the round-trip invariant `processor: v→w` ⇄ `verifier: w→v`. `check_implication` itself needs no revert: it runs one substitution-based path for every implication (theorem-anchor rules included); the former anchor-only branch and its `_normalize_all_vars_in_list` helper were removed by [D-75](../40_decisions.md#d-75) — see the single-path note immediately below.

### `check_implication` single substitution path

Pre-[D-75](../40_decisions.md#d-75) `check_implication` had two branches: a theorem-level branch (first premise an `(Anchor…` ⇒ treat every variable as changeable, compare via the now-removed `_normalize_all_vars_in_list`) and a general substitution branch. They were unified into one path. For every implication it computes `unchangeables = _collect_all_expr_vars(impl) − _collect_bound_vars(impl)` (every name bound in any `>[...]` is changeable; the rest must match literally), then searches a premise permutation under which core names align position-wise and a consistent `changeable_map` exists. Under the unified binder rule a definition rule's only unchangeables are its `u_` formal parameters; a raw theorem has none (every non-`u_` variable is now bound), so `unchangeables` is empty and the path accepts exactly what the removed normalize-all branch accepted — theorem handling is thereby equalized to non-theorem handling. Benign asymmetry: `changeable_map` permits a non-injective map, so the unified path is at most marginally more permissive than the old anchor branch's first-appearance bijection — it can only turn a former reject into an accept, never newly reject a previously-accepted theorem-anchor row.

### Why some tags are exempted

| Exempt tag | Reason |
|---|---|
| `incubator back reformulation` | The row's rewrite step (witness elimination) is verified structurally by `check_incubator_back_reformulation`; the origin meta-check still skips the citation because the cited source uses a back-reformulated form that may not exactly match a registry entry (the source's own proof is an incubator-side verification target). |
| `contradiction` | Reductio dependencies include the assumption being negated, which by construction does not appear as a derived row in the same chapter (the chapter proves its negation). The contradiction-trace check (separate tag) handles the actual chain integrity. |
| `or disintegration`, `or convergence`, `or branch proven`, `or branch assumption`, `or theorem` | OR-flow rows have validity-aware dependency chains that the bare `dep in originated` check cannot represent; their dedicated `or *` checkers cover them. |

### Failure modes

A non-zero `origin` count typically means one of:

1. **Compact-vs-expanded form mismatch in the registry.** Registry has expanded form; chapter cites compact. Fix at producer: ensure the seed source carries compact-form rules. (Resolved by D-40 for the cross-batch externals seed.)
2. **A genuinely missing rule.** A chapter cites a rule that was never proved nor externally provided. Indicates a producer-side bug (the rule firing wasn't recorded as a global theorem, or the rule itself was elided by the compressor but the chapter row still uses it).
3. **An exempt-tag-row whose dependency went through a non-exempt path.** E.g. an OR-flow row whose `rest[0]` somehow falls through the i==0 implication-style branch. Indicates a tag-misclassification at the producer.

Diagnostic recipe: run a small inspector that mirrors the loop above and prints the failing `(chapter, line, tag, dep)` tuple. The  script (used during D-40 localisation) is one example.

---

## `definition set consistency` meta-check

A second chapter-level meta-check (added by [D-41](../40_decisions.md#d-41)). Like `origin`, it is a check-name in the tally rather than a producer-side proof tag. It enforces [I-29](../30_invariants.md#i-29): variable-port type consistency across every expression in every chapter row.

### What it checks

`check_defset_consistency` at [`verifier.py`](../../verifier.py). For every row in a chapter, every expression in `[line.expression] + line.rest[::2]` is parsed independently with `_parse_subtree`. The parser builds a per-node `remainingArgsDefs` map (`var → type_label`) and removes bound variables at every `>[…]` quantifier so their names can be reused at outer scopes without collision. `_merge_maps` is the actual mismatch detector — when two child sub-trees share a variable name with disagreeing types, it raises `_DefsetMismatch`, which `check_defset_consistency` catches and records as a row failure.

### Faithful translation of the C++ compiler's algorithm

The Python implementation translates `compiler.hpp` (`ArgumentAnalyzer`) symbol-for-symbol:

| Python | C++ |
|---|---|
| `_merge_maps` | `ArgumentAnalyzer::mergeMaps` |
| `_process_leaf` | `RecursiveParser::processLeaf` |
| `_parse_subtree` | `RecursiveParser::parseSubtree` (implication / conjunction / leaf, with negated counterparts) |
| `check_defset_consistency` | `ArgumentAnalyzer::checkDefinitionConsistency`, called from `compileCoreExpressionMap` in `prover.hpp` |

The intent: any verifier-side failure points squarely at a producer-side bug, because the verifier and producer enforce the same algorithm.

### Per-batch resolution

The compiler's analyzer is constructed `ArgumentAnalyzer(this->coreExpressionMap)` inside `prover.hpp` — per-batch by construction. Compact operator names (e.g. `implication26`) are allocated per-batch with potentially different shapes; the analyzer naturally sees only its own batch's allocations.

The verifier mirrors this with **per-tag resolved-defset indices**: `state.resolved_defsets_per_tag` maps tag → resolved defsets. Each tag's map is the union of:

1. Atomic operators from `state.definition_sets` (ConfigVisu.json baseline).
2. Composites from `gl_binaries['shared']` (cross-batch fallback for spontaneous-category compact names — `_SPONTANEOUS_CATEGORIES` in `run_modes.py`).
3. Composites from `gl_binaries[tag]` — overriding shared on collision (per-batch is authoritative for its own chapters).

Per chapter, `verify_chapter` selects the right tag's map alongside `current_gl_binary` (anchor-substring match on the chapter's theorem expression). For chapters whose theorem doesn't disclose an anchor, `state.resolved_defsets_atomic_only` is the fallback (atomic seeds without composites — composite-using rows then have unknown-op skip behaviour).

`build_resolved_defsets_per_tag` runs once at `run_verifier` startup — fixed-point iteration over each tag's merged binary against atomic seeds. Per-row check is then a flat dict lookup tree-walk, no per-row recursion into `gl_binaries.elements`.

### Failure modes

A non-zero `definition set consistency` count means one of:

1. **ConfigVisu drift from per-batch configs.**  catches this: each operator's defsets in `ConfigVisu.json` is checked against batch configs; mismatch → drift. Fix: sync ConfigVisu to per-batch authoritative.
2. **Genuine producer-side type contention in a chapter row.** A row's expression has a variable wired through ports of incompatible types. The compiler's `ArgumentAnalyzer` would have asserted at compile time if the *rule itself* were ill-typed — so a chapter-row failure points at a *runtime* misuse: a rule application substituting an arg with the wrong type. Investigate the row's tag and the producer site that emitted it.
3. **Inter-batch compact-name collision unreachable via current resolution.** A composite operator allocated in two batches with different shapes, neither inheriting from `shared`. The per-tag selection at `verify_chapter` picks based on anchor-substring match; if the chapter doesn't disclose the right tag, the wrong batch's allocation may apply. Surfaces only when a batch is added without anchor disclosure in chapters.

Diagnostic recipe:  — a localizer that calls `check_defset_consistency` per row and prints the offending expression.

### Synthetic-corruption sanity

 constructs ProofLine instances with hand-crafted type-mismatched expressions (e.g. `(in[v,v])` — `v` as element AND set; `(&(in[a,N])(=[N,b]))` — `N` as set then element) and confirms the check returns `False`; well-typed cases (including bound-var rebinding `(>[v](in[v,N])(=[v,a]))`) return `True`. Eight cases pass. Used as a regression guard if the parser is ever modified.

---

## Weaknesses

### Known & tracked

- **No CI per-row failure gate.** The verifier emits its tally, but no CI step fails the build on non-zero failure counts. A silent regression in a single checker persists across commits until a maintainer notices.

### Suspected fragility

- **Hardcoded `"main"` namespace.** Many locations check `line.namespace == "main"` as a gate. If the proof graph ever introduces alternative namespaces outside the already-enumerated cases, those checkers silently reject valid lines as failures.
- **Chapter filename regex assumption.** `chapter_type = base.split("_", 1)[1]` assumes a numeric prefix + underscore. A filename without underscore raises `IndexError`; a filename with a non-numeric prefix corrupts the sort order (falls back to the `999999` sentinel in `chapter_sort_key`).
- **Induction triad `AssertionError`.** A malformed triad (missing one chapter, wrong order) does not report a verification failure — it crashes. The user sees a Python stack trace instead of a tag-row failure.
- **Anchor-prefix hardcoding.** Anchor names must match the `Anchor<Tag>` pattern. A typo or non-standard name causes silent `current_gl_binary = None` and cascading checker failures.
- **Variable suffix conventions.** `_copy` suffix and `_integration_goal` suffix are hardcoded at several call sites across `check_anchor_handling`, `check_variable_copy`, `check_or_branch_assumption`, `check_premise_element`, `check_expansion_for_integration`, `_strip_integration_goal`. A rename upstream (prover or processor) would silently break verification.
- **Regex-based argument boundaries.** `_replace_arg_safe` uses lookaround on `[\[,]` and `[\],]`. Malformed expressions with unbalanced brackets could produce incorrect argument boundaries with no validation.
- **GL binary directory fallback is silent.** No log output indicates the verifier is using the fallback path. If the fallback is empty or out of date, checkers that need the binary fail with unclear messages.
- **Multiple anchors in one theorem (premise side).** The premise-anchor extraction returns the *first* `Anchor<Tag>` substring inside the theorem's outer-implication premise. For cross-anchor theorems this is correct by D-pending (premise wins), but a malformed theorem with two `Anchor<Tag>` substrings *in the premise* would bind only the first with no warning.
- **`_check_reformulation` early-rejects `< 2` premises.** A single-premise source theorem cannot be checked as a reformulation source. This is correct semantically but masks producer-side regressions where a single-premise rule is accidentally cited via the wrong tag.

### Documented stub-accepts

These checkers structurally pass any row meeting trivial gates. Soundness of the matching tag is therefore *not* enforced by the verifier; the structural witness lives upstream.

- **`check_or_theorem`** — accepts any row with `namespace == "main"` and `len(rest) >= 2`. The OR theorem's structural witness lives in the existence theorem proved upstream; this row marks registration only. Hardened tightening is an [open question](#open-questions).

(`check_incubator_back_reformulation` was formerly listed here; it now performs a real structural check — see [`incubator back reformulation`](../20_core_concepts/08_proof_tags.md#incubator-back-reformulation).)

### Tightening that deliberately fails on current production

- **`check_or_convergence`** — emits the post-clean-fail row-layout contract. The producer side has not yet been updated to emit the new layout (the buildstack + history-tracking follow-on), so every `or convergence` row on current chapters fails the `len(rest) >= 6` step. This is **intentional** per the "Failures are first-class" guidance; once the producer-side fix lands, the layout will match. Until then, expect non-zero `or convergence` failures on baseline runs.

### Parsing tolerance for malformed rows

- `parse_chapter_file` synthesises `tag = "<malformed>"` when the row has fewer than three tab-separated columns. The dispatcher then routes such rows to `state.counter_for("<unknown:<malformed>>")` so the report flags them without crashing. The fallback is mute about *which* row is malformed — a future enhancement could include the row number in the synthesised tag.

### Not exercised by tests

- **Some checkers lack positive (`assert_pass`) coverage.** `test_verifier_positive.py` exists specifically as a rig-sanity guard against fixtures that mangle inputs into trivially-failing rows. A handful of TAG_CHECKERS entries currently have failure-only test coverage; see the open question below.
- **Large-chapter performance.** The verifier loads chapter files line-by-line into memory. No streaming mode. Very large FTA-era chapters could hit memory issues; not measured.
- **Multiple external theorems with same shape.** If `external_theorems.txt` contains two rows with identical MPL (should not happen by construction, but not enforced), `state.external_theorems` silently deduplicates — a checker looking for a specific provenance would have no way to distinguish.

---

## Open questions

- **OPEN-2 — RESOLVED (correction).** Previous framing as "non-checker" was overstated. The `origin` path inside `verify_chapter` *is* a real check: it actively validates whether every dependency citation resolves to a real source (global theorem, external theorem, or earlier chapter row), recording into `state.counter_for("origin")`. What *is* true: `origin` is not in the `TAG_CHECKERS` dispatch table — it is a direct call from `verify_chapter`, not a per-tag-row checker. So: counted + validated, but outside the `TAG_CHECKERS` mechanism.
- **OPEN-18.** Induction-typing checker (when it lands per [I-18](../30_invariants.md#i-18)) — name and behaviour. Planned in [`induction_typing_plan.md`](../induction_typing_plan.md) stage 3: a new `"induction typing"` key added to `TAG_CHECKERS`; handler walks every `method == "induction"` row in `global_theorem_list.txt`, locates the typing chapter by naming convention (`<N>_induction_typing.txt` paired with `<N+1>_check_zero.txt` / `<N+2>_check_induction_condition.txt`), asserts its existence, and verifies the chapter's task-formulation claims the typing head `(in[ind_var, N_arg])`. See plan file stage 3 for the full specification.
- **OPEN-or-theorem-stub.** `check_or_theorem` is currently a stub-accept. A tightening would require either (a) decoding the OR's compiled form and checking it expands to one of the prover-side existence theorems registered as `rest[0..3]`, or (b) demanding a matching existence-theorem row exist in the chapter at `namespace == "main"` with the right `rest` shape. Either tightening could surface real producer-side issues that are currently masked.

---

## See also

- [`10_pipeline/06_process_proof_graph.md`](06_process_proof_graph.md) — producer of the data the verifier consumes.
- [`10_pipeline/07_html_export.md`](07_html_export.md) — parallel consumer.
- [`20_core_concepts/08_proof_tags.md`](../20_core_concepts/08_proof_tags.md) — tag vocabulary.
- [I-16](../30_invariants.md#i-16) — verifier is sacred.
- — the "don't modify verifier.py without consent" rule.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
