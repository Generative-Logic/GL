<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Pipeline · Stage 10 — Verifier `[DRAFT]`

> **Input:** `files/processed_proof_graph/*.txt` (chapters + `global_theorem_list.txt` + `external_theorems.txt`) + `files/GL_binaries/GL_binary_<Tag>.json` + `files/config/ConfigVisu.json`.
> **Output:** stdout — a ≈32-line tally of per-tag success/failure counts + a final "airtight" or "FAILED" line.
> **Owner:** `verifier.py` (≈3030 lines).
> **Entry:** `main` at [`verifier.py`](../../verifier.py), which parses CLI arguments (positional `base_dir`, repeatable `--include-globals PATH`), calls `run_verifier(base_dir, extra_global_lists)` at [`verifier.py`](../../verifier.py), then `print_report(state)` at [`verifier.py`](../../verifier.py).
>
> **CLI surface (D-35).**
> - `python verifier.py` — verify the main pipeline output (`files/processed_proof_graph/`, the historical default).
> - `python verifier.py files/incubator/processed_proof_graph --include-globals files/processed_proof_graph/global_theorem_list.txt` — verify the incubator output, unioning the main pipeline's global theorem registry into the incubator's so cross-batch theorem citations resolve.
> - `--include-globals` is repeatable; entries are unioned in order with local-batch entries taking precedence on key collisions.

---

## What this stage does

The verifier is the external proof checker. It consumes the same data the customer-facing HTML export consumes (the *processed* proof graph), and for each chapter, re-checks every proof step against the claim of its tag.

**Core principle: independence.** The verifier has its own copies of every algorithm it needs — expression parsing, disintegration, normalisation, mirror and reformulation checks, digit-arg/immutable-arg computation. It does **not** import from `expression_utils.py` or from any prover code. This separation is the entire point: if the prover and the verifier share a bug, they can't catch each other.

Invariant [I-16](../30_invariants.md#i-16) is absolute: never weaken a verifier check to "pass a test". A failure is a real bug, not a false positive.

---

## Output — the ≈32-line tally

`print_report(state)` at [`verifier.py`](../../verifier.py) emits a fixed-format report:

```
theorem goal reached                         success N, failure M
implication                                  success N, failure M
expansion                                    success N, failure M
...
(28 more rows, one per TAG_CHECKERS tag + non-checker categories)
...
(final summary line — one of:)
   N checks, 0 failures — airtight.
   Verifier: N checks, M FAILED.
```

Column width: `W = 44`. The tag name is left-justified to 44 chars, followed by `success <count>, failure <count>`.

**On a clean release run, every row reports `failure 0`.** In-flight FTA-ladder branches (e.g. ) regularly have non-zero failures — these indicate the rung currently being worked on, not regressions against the release baseline. for the reference clean counts.

---

## TAG_CHECKERS — the registry

At [`verifier.py`](../../verifier.py). A dictionary mapping each tag string to its checker function. 30 registry entries, 30 unique tags (one entry per tag; `or branch proven` and `or branch assumption` added by [D-35](../40_decisions.md#d-35); the historical `equalize variable` alias was removed since only `multiplied from` was ever emitted).

Full table:

| Tag | Checker | File |
|---|---|---|
| `implication` | `check_implication` | [`verifier.py`](../../verifier.py) |
| `expansion` | `check_expansion` | [`verifier.py`](../../verifier.py) |
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
| `mirrored from` | `check_mirrored_from` | [`verifier.py`](../../verifier.py) |
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

**Note on the project conventions drift.** the project conventions's narrative tag list names 27 distinct tags. The actual count is 30 distinct tags + 31 registry entries (one shared checker for `equalize variable` and `multiplied from`; the two new entries `or branch proven` and `or branch assumption` were promoted from "claimed retired" to first-class by [D-35](../40_decisions.md#d-35)). The missing tags in the project conventions's narrative are `symmetry of inequality`, `or branch proven`, and `or branch assumption`. See [40_decisions.md](../40_decisions.md) for the reconciliation note.

### Validity matching on dep cells

Each checker that compares a row's dep validity to a stored validity uses **exact `==`** equality. Per `D-56`, `buildStack` emits every chapter cell (`row[1]` and every dep cell) at the **closest-to-`main` ancestor of the requested validity for which an origin entry exists in the emitting LB's `exprOriginMap`**, so a row's `row[1]` and each cited dep's recorded validity are already at the same lifted scope by construction. The legacy `_ns_matches_or_strict_prefix` helper at `verifier.py` is retained only for `equality1` / `equality2` (where the source/target validity relationship is on the *expression* side of an equivalence-class rewrite, not the rule-application side). Other checkers do not need ancestor-or-equal matching after lifting.

**Per-tag structural contracts.** The exact `rest`-field shape each checker expects is documented in [`20_core_concepts/08_proof_tags.md`](../20_core_concepts/08_proof_tags.md). Key examples to watch when authoring new origins:

- `equality1` — `rest[0]=source_expr`, `rest[1]=source_ns`, interleaved `(eq, ns)` pairs thereafter. Minimum `len(rest)=4`. Source must be in the row's namespace. See [`08_proof_tags.md#equality1`](../20_core_concepts/08_proof_tags.md#equality1).
- Every tag's `rest` alternates `(expr, ns)` — odd-length `rest` is silent-reject by the length check in every checker. See the [row-shape recap](../20_core_concepts/08_proof_tags.md#row-shape-recap).
- An unknown origin tag (not in `TAG_CHECKERS`) is treated as 1 failure per occurrence — `<unknown:tag>` row in the verifier summary. Reusing an existing tag whose structural contract fits is safer than adding a new tag (which requires `verifier.py` modification — see [I-16](../30_invariants.md#i-16)).

**Non-checker categories** tracked but not validated:

- `anchor handling trace` — per-step chain of `_copy` variable rewrites produced by `anchor handling`.
- `origin` — provenance chain for a `contradiction`'s derivation tree.
- `self-reference` — error counter; increments when a chapter cites its own theorem as a justification.
- `vacuous truth trace` — chapter-local origin-chain walk added by commit. At least one contradicting ingredient in a `vacuous truth` row must trace back to the recursion-hypothesis cited in `rest[4..5]`. Counter reports success/failure separately from `vacuous truth` itself; a failing trace means the LB's vacuous-truth claim is unsound even if the `check_vacuous_truth` structural check passes. Pairs with the prover-side `mb.level`-in-ingredients gate to prevent induction-step LBs from collapsing on anchor-only contradictions. See [`08_proof_tags.md#vacuous-truth`](../20_core_concepts/08_proof_tags.md#vacuous-truth).

---

## Key helper algorithms

Independent copies of algorithms the verifier needs. These must stay in sync with the C++ side *semantically*, but are not allowed to *call* any C++ code or share implementation.

| Function | File | Role |
|---|---|---|
| `disintegrate_implication_head` | [`verifier.py`](../../verifier.py) | Peel all `(>[vars](premise)(body))` layers; return innermost conclusion as-is (no renaming). |
| `disintegrate_implication_full` | [`verifier.py`](../../verifier.py) | Peel all layers; return `(premises_list, head)`. |
| `_normalize_expr_list` | [`verifier.py`](../../verifier.py) | α-rename v-variables across a list of expressions by order of first appearance in arguments. |
| `_alpha_canonicalize_bound_vars` | [`verifier.py`](../../verifier.py) | Canonicalize every name introduced inside a `>[...]` binder to `b1, b2, …` in declaration order. Used by the origin check (D-35) to compare a chapter's `rest[0]` rule against the global theorem registry across batches when bound variables were renamed differently by `process_proof_graphs.py` (e.g. incubator chapter has `>[i2]` where Peano global has `>[v1]`). Stronger than `_normalize_expr_list`, which renames `[vV]\d+` (lowercase v digits + uppercase V sets, both case-folded into the same canonical sequence — sandbox `var_typing` 2026-05-04). |
| `_check_mirror` | [`verifier.py`](../../verifier.py) | Verify `target` is a valid mirror of `source` — disintegrate both, swap output-matched premises, permute, normalise, compare. |
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

Stages (in order within a single chapter):

1. Check theorem goal reached ([`verifier.py`](../../verifier.py)) → updates `state.goal_reached`.
2. Self-reference check ([`:2644–2649`](../../verifier.py)) → `state.counter_for("self-reference")`.
3. Anchor-handling uniqueness ([`:2651–2654`](../../verifier.py)) → `state.counter_for("anchor handling uniqueness")`.
4. Anchor-handling trace ([`:2661–2689`](../../verifier.py)) → `state.counter_for("anchor handling trace")`.
5. Contradiction trace ([`:2691–2706`](../../verifier.py)) → `state.counter_for("contradiction trace")`.
6. **Dispatch each line to its tag checker** ([`:2708–2715`](../../verifier.py)) — the main loop.
7. General origin check ([`:2717–2749`](../../verifier.py)) → `state.counter_for("origin")`.

---

## Self-reference detection

[`verifier.py–2649`](../../verifier.py). After loading `chapter_thm`:

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

Loaded at [`verifier.py–2793`](../../verifier.py) into `state.external_theorems` (set).

The `check_externally_provided_theorem` checker at [`verifier.py`](../../verifier.py) requires:

- `line.namespace == "main"`;
- `line.expression` is either a direct member of `state.external_theorems` (raw or pre-renamed form), or a valid mirror of some member (fallback via `_check_mirror`).

Failure → False recorded.

External theorems are also consulted as valid sources during origin validation ([`:2741–2748`](../../verifier.py)) — for tags like `implication`, `multiplied from`, `mirrored from`, `reformulated from` that cite a source theorem.

**External theorems are not blanket-axiomatic.** They must pass the same structural checks (mirror shape, reformulation shape) as internal theorems. They are "external" only in the sense that their own proof graphs are not included in the verifier's scope.

---

## Chapter-file naming — the load-bearing assertion

The filename pattern is:

```
<N>_<type>.txt
```

where `<type>` is one of the known chapter types: `direct_proof`, `check_zero`, `check_induction_condition`, `induction_typing`, `mirrored_statement`, `reformulated_statement`, `back_reformulated_statement`, `or_theorem`.

`build_chapter_theorem_map` ([`verifier.py`](../../verifier.py)) enforces:

- Induction chapter triples (`induction_typing` + `check_zero` + `check_induction_condition`) must be consecutive and in exact order ([`:179–194`](../../verifier.py)).
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

The `state.gl_binaries` map is consulted by:

- `check_expansion` — to expand a named expression into its compiled-structure form.
- `_check_reformulation` — to expand an existence head via the binary into left+right elements.
- `check_anchor_handling` — to match anchor-slot positions.

The verifier selects the correct binary per chapter via substring match for `Anchor<Tag>` in the theorem expression ([`:2639–2641`](../../verifier.py)):

```python
anchor_substring_match = re.search(r'Anchor([A-Za-z0-9_]+)', theorem_expr)
if anchor_substring_match:
    tag = anchor_substring_match.group(1)
    current_gl_binary = state.gl_binaries.get(tag)
else:
    current_gl_binary = None
```

Silent fallback to `None` — checkers that require a GL binary then fail with messages that do not always make the root cause obvious.

### Chapter-context filter on the binary-selection path

After the duplicate-folder cleanup ([D-54](../40_decisions.md#d-54)), main and incubator binaries share `files/GL_binaries/`, and their spontaneous compact-operator names collide on shape — for example, `existence4` is arity 5 in `Gauss` / `shared` but arity 4 in `IncubatorGauss1`, and `existence2` is arity 3 in main batches but arity 8 in `IncubatorPeano`. Three sites in the verifier route operator lookups; each one now restricts to `Incubator`-prefixed tags when the chapter's theorem expression contains `AnchorIncubator`:

1. `verify_chapter`'s `current_gl_binary` selection (`verifier.py` around line 3340). The cross-anchor connection chapter `(>[..](AnchorIncubator[..])(AnchorPeano[..]))` contains *both* `AnchorIncubator` and `AnchorPeano` substrings; without the filter, alphabetical iteration over loaded tags would match `AnchorPeano` first and bind `current_gl_binary` to the Peano binary — breaking 3 incubator chapter checks (`expansion`: 1 failure, `disintegration`: 2 failures).
2. `binaries_for_chapter` (`verifier.py` ≈ line 109). Returns the fallback list when `current_gl_binary` is `None`. With the same `AnchorIncubator` substring guard, the fallback returns only `Incubator`-prefixed tags. Affects `check_expansion`, `check_disintegration`, the OR-family checkers (when they fire) and others that iterate this list looking for an operator's compiled definition.
3. `_check_reformulation`'s D-35 fallback (`verifier.py` ≈ line 600+). When the target's anchor is the literal `AnchorIncubator` (so `tag == "Incubator"`), the existence-head scan iterates only `Incubator`-prefixed tags.

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

All three edits are pure tightenings. They remove from consideration only those binaries that the previous duplicate-folder layout never exposed on the incubator side anyway.

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
        if i == 0 and line.tag in ("implication", "multiplied from", "mirrored from", "reformulated from"):
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
- **Cross-chapter rule citation (the `i==0` path).** When the row's tag is `implication` / `multiplied from` / `mirrored from` / `reformulated from` and we're looking at `rest[0]`, the dep is the *rule* applied. Rules can be globally proved, externally provided, or alpha-canonical equivalents of either. Pass if any of those four match. Fail otherwise.
- **All other unmatched deps.** Fail — record `False` for the `origin` counter.

### What alpha-canonicalisation does (and doesn't)

`_alpha_canonicalize_bound_vars(expr)` ([`verifier.py`](../../verifier.py)) renames every `>[a,b,…]` bound variable in declaration order to a canonical sequence (`v_1`, `v_2`, …). This makes the rule `(>[N,i0,s](AnchorPeano[N,i0,s,…])(>[v1](in[v1,N])(=[v1,i0])))` and its alpha-equivalent `(>[a,b,c](AnchorPeano[a,b,c,…])(>[d](in[d,a])(=[d,b])))` produce the same canonical string.

What it does **not** do:

- It does **not** rewrite compiled structural operators (`existence2[1,7,3]` stays `existence2[1,7,3]`; never expanded to `!(>[8](in[8,1])!(in2[8,7,3]))` or vice versa).
- It does **not** consult `state.gl_binaries` to perform compact↔expanded conversion.
- It does **not** normalise free variables (only bound).

This was the substrate of the [D-40](../40_decisions.md#d-40) bug. Chapter rows cite rules in compact form (with `existence2`); the externals seed used to come from `proved_theorems.txt` (expanded form, no `existence2`); alpha-canon couldn't bridge the form gap. The fix was to switch the seed source to `compiled_proved_theorems.txt` (compact form), so registry entries are now in the same form chapter rows cite. The `origin` check itself is unchanged.

### `w → v` revert for cited theorem-anchor implications

`process_proof_graphs.py` Pass 3 (see [`06_process_proof_graph.md`](06_process_proof_graph.md#pass-3-vw-rename-of-cited-theorem-anchor-implications)) renames non-anchor inner bvars in cited theorem-anchor implications from `v / V` to `w / W` (case-preserving letter swap, digit preserved). The registry (`state.global_theorems`, `state.external_theorems`) stores the `v / V` form. To keep the fast-path exact-string membership test intact:

- `_is_theorem_anchor_impl_local(expr)` in [`verifier.py`](../../verifier.py) detects whether a citation has the theorem-anchor shape `(>[…]( Anchor<Tag>[…])body)`.
- `_revert_w_to_v_in_theorem_citation(expr)` applies the symmetric inverse swap (`w<N> → v<N>`, `W<N> → V<N>`) on every `[wW]\d+` argument-position token.

The revert is invoked at three sites:

1. The origin meta-check (above), at `rest[0]` for `implication` / `multiplied from` / `mirrored from` / `reformulated from` rows.
2. `check_theorem_tag` — defensive only; HEAD column is never `v→w`-swapped by the processor, but the revert path catches a future regression.
3. `check_externally_provided_theorem` — same defensive shape against `state.external_theorems`.

`_normalize_expr_list` (already widened to `[vVwW]\d+`) provides a slower fallback that also matches; the explicit revert preserves the fast path and documents the round-trip invariant `processor: v→w` ⇄ `verifier: w→v`. `check_implication`'s anchor branch needs no revert: it uses `_normalize_all_vars_in_list`, which folds *every* argument token to `v1, v2, …` order-wise — so `w` and `v` collapse identically.

### Why some tags are exempted

| Exempt tag | Reason |
|---|---|
| `incubator back reformulation` | The dep is an external theorem reference using a back-reformulated form that may not exactly match any registry entry; back-reformulation is a one-way rewrite. |
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

[`verifier.py:check_defset_consistency`](../../verifier.py). For every row in a chapter, every expression in `[line.expression] + line.rest[::2]` is parsed independently with `_parse_subtree`. The parser builds a per-node `remainingArgsDefs` map (`var → type_label`) and removes bound variables at every `>[…]` quantifier so their names can be reused at outer scopes without collision. `_merge_maps` is the actual mismatch detector — when two child sub-trees share a variable name with disagreeing types, it raises `_DefsetMismatch`, which `check_defset_consistency` catches and records as a row failure.

### Faithful translation of the C++ compiler's algorithm

The Python implementation translates `compiler.hpp` (`ArgumentAnalyzer`) line-for-line:

| Python | C++ |
|---|---|
| `_merge_maps` | `ArgumentAnalyzer::mergeMaps` (line 1250-1264) |
| `_process_leaf` | `RecursiveParser::processLeaf` (line 1409-1426) |
| `_parse_subtree` | `RecursiveParser::parseSubtree` (line 1281-1407) — implication / conjunction / leaf, with negated counterparts |
| `check_defset_consistency` | `ArgumentAnalyzer::checkDefinitionConsistency` (line 1499-1538), called as in `compileCoreExpressionMap` (`prover.hpp`) |

The intent: any verifier-side failure points squarely at a producer-side bug, because the verifier and producer enforce the same algorithm.

### Per-batch resolution

The compiler's analyzer is constructed `ArgumentAnalyzer(this->coreExpressionMap)` at `prover.hpp` — per-batch by construction. Compact operator names (e.g. `implication26`) are allocated per-batch with potentially different shapes; the analyzer naturally sees only its own batch's allocations.

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

- **No CI per-row failure gate.** The verifier emits a 28-row tally, but no CI step fails the build on non-zero failure counts. A silent regression in a single checker persists across commits until a maintainer notices.
- **the project conventions tag-count drift.** the project conventions claims 28 tags in TAG_CHECKERS; the actual count is 28 distinct tags + 29 registry entries (shared checker). Logged as a minor documentation debt; fixed by this chapter.

### Suspected fragility

- **Hardcoded `"main"` namespace.** ≈27 locations check `line.namespace == "main"` as a gate. If the proof graph ever introduces alternative namespaces outside the already-enumerated cases, those checkers silently reject valid lines as failures.
- **Chapter filename regex assumption.** `chapter_type = base.split("_", 1)[1]` assumes a numeric prefix + underscore. A filename without underscore raises `IndexError`; a filename with a non-numeric prefix corrupts the sort order.
- **Induction triad `AssertionError`.** A malformed triad (missing one chapter, wrong order) does not report a verification failure — it crashes. The user sees a Python stack trace instead of a tag-row failure.
- **Anchor-prefix hardcoding.** Anchor names must match the `Anchor<Tag>` pattern. A typo or non-standard name causes silent `current_gl_binary = None` and cascading checker failures.
- **Variable suffix conventions.** `_copy` suffix and `_integration_goal` suffix are hardcoded at ≈2338, ≈2671, ≈1786, ≈1791. A rename upstream (prover or processor) would silently break verification.
- **Regex-based argument boundaries.** `_replace_arg_safe` uses lookaround on `[\[,]` and `[\],]`. Malformed expressions with unbalanced brackets could produce incorrect argument boundaries with no validation.
- **GL binary directory fallback is silent.** No log output indicates the verifier is using the fallback path. If the fallback is empty or out of date, checkers that need the binary fail with unclear messages.
- **Multiple anchors in one theorem.** The anchor-substring match returns the *first* match. A theorem containing two distinct `Anchor<Tag>` substrings would get only the first's binary applied, with no warning.

### Not exercised by tests

- **Per-tag regression tests.** There is no unit test per TAG_CHECKERS entry asserting "known-good row passes, known-bad row fails". Changes to a single checker can regress silently.
- **Large-chapter performance.** The verifier loads chapter files line-by-line into memory. No streaming mode. Very large FTA-era chapters could hit memory issues; not measured.
- **Multiple external theorems with same shape.** If `external_theorems.txt` contains two rows with identical MPL (should not happen by construction, but not enforced), `state.external_theorems` silently deduplicates — a checker looking for a specific provenance would have no way to distinguish.

---

## Open questions

- **OPEN-2 — RESOLVED (correction).** Previous framing as "non-checker" was overstated. The `origin` path at [`verifier.py–2749`](../../verifier.py) *is* a real check: its comment reads *"General origin check: every expression referenced as a dependency..."*, and it calls `state.counter_for("origin").record(found)` / `record(False)` — actively validating whether every dependency citation resolves to a real source (global theorem, external theorem, or earlier chapter row). What *is* true: `origin` is not in the `TAG_CHECKERS` dispatch table. It is a direct call from `verify_chapter`, not a per-tag-row checker. So: counted + validated, but outside the `TAG_CHECKERS` mechanism.
- **OPEN-18.** Induction-typing checker (when it lands per [I-18](../30_invariants.md#i-18)) — name and behaviour. Planned in [`induction_typing_plan.md`](../induction_typing_plan.md) stage 3: a new `"induction typing"` key added to `TAG_CHECKERS`; handler walks every `method == "induction"` row in `global_theorem_list.txt`, locates the typing chapter by naming convention (`<N>_induction_typing.txt` paired with `<N+1>_check_zero.txt` / `<N+2>_check_induction_condition.txt`), asserts its existence, and verifies the chapter's task-formulation claims the typing head `(in[ind_var, N_arg])`. See plan file stage 3 for the full specification.

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
