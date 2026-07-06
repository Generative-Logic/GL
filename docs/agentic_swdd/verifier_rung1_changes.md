<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# `verifier_rung1` — verifier changes for FTA-ladder rung 1

> **Decision record.** [`docs/agentic_swdd/40_decisions.md` D-35](40_decisions.md#d-35--verifier_rung1-incubator-verifier-extensions-for-fta-ladder-rung-1-forward-direction-2026-05-03-branch-sandboxverifier_rung1).
> **End-state at the close of `verifier_rung1` (after D-36 producer-side `ordisMerge` extension + verifier corrections).**
> - Main pipeline verifier: `2750 checks, 0 failures`.
> - Incubator verifier: `32763 checks, 1 FAILED`. **All 11 of the original baseline failures closed.** The remaining 1 failure was `self-reference` on chapter `1032_direct_proof.txt` (`!(fold[…])` theorem self-citing) — pre-existing producer-side issue unrelated to the OR work, surfaced incidentally by the `main.py` rerun that regenerated the chapter set. See [D-36 "Remaining 1 failure"](40_decisions.md#d-36) for the diagnostic.
>
> **Current state (subsequent work landed since this branch closed).**
> - Main pipeline verifier: `~2997 checks, 0 failures`.
> - Incubator verifier: `~32869 checks, 0 failures`. The `1032_direct_proof.txt` self-reference noted above has since been resolved by producer-side fixes outside the `verifier_rung1` scope. Both pipelines now report `0 failures` across every tag category.

---

## Why

`verifier.py`'s `main` historically hardcoded `files/processed_proof_graph` and never visited the incubator processed proof graph (`files/incubator/processed_proof_graph/`, ≈1213 chapters). When the verifier IS pointed at the incubator path, **11 failures** surface across 32780 checks. All 11 trace to the FTA-ladder rung-1 forward-direction theorem (`EnumerationSet2 ⟹ interval`, [D-34](40_decisions.md#d-34) closure) — chapter `1209_direct_proof.txt` (9 failures) and its reformulated companion `1210_reformulated_statement.txt` (2 failures). No prover or process-proof-graph changes were required; every fix is a verifier extension that recognises proof-tag patterns the verifier had never been exercised against. Per [I-16](30_invariants.md#i-16) every extension is a proper structural check — no checks were weakened.

## Failure inventory and resolution map

| # | Chapter | Line | Tag (baseline) | Root cause | Fix | Commit |
|---|---|---|---|---|---|---|
| 1 | 1209_direct_proof | 5 | `or convergence` | Initial cause: `check_or_convergence` expected expanded `!(&!(…))` in `rest[0]`; row carries compiled `(or2[i0,v2,i1])`. **Final state: deliberate-fail.** Chapter-local evidence cannot prove "C derived in every branch" (see V-2 below); per I-16 the check must fail until producer-side fix. | V-2 (clean-fail) | → → clean-fail |
| 2 | 1209_direct_proof | 14 | `or convergence` | Same as #1, both root cause and final clean-fail state. | V-2 (clean-fail) | same |
| 3 | 1209_direct_proof | 22 | `<unknown:or branch proven>` | Tag emitted live by prover but missing from `TAG_CHECKERS`. SwDD's "retired/overridden" claim was stale. | V-3 | |
| 4 | 1209_direct_proof | 36 | `implication` | One premise drawn from OR-branch's parent scope while result lands in branch. Pre-existing rule rejected mixed non-main namespaces. | V-5 | |
| 5 | 1209_direct_proof | 41 | `implication` | Same scope-mix root cause as #4. | V-5 | |
| 6 | 1209_direct_proof | 43 | `<unknown:or branch assumption>` | Tag emitted live by prover but missing from `TAG_CHECKERS`. | V-4 | |
| 7 | 1209_direct_proof | 56 | `implication` | Same scope-mix root cause as #4. | V-5 | |
| 8 | 1209_direct_proof | 73 | `implication` | Same scope-mix root cause as #4. | V-5 | |
| 9 | 1209_direct_proof | 41 | `origin` (inline) | `implication`'s `rest[0]` rule = the Peano `existence2` axiom — present in main pipeline's `global_theorem_list.txt`, absent from incubator's. Cross-batch theorem references not loaded. | V-7 + alpha-canonicalize | |
| 10 | 1210_reformulated_statement | (chapter) | `theorem goal reached` | Anchor `AnchorIncubator` derives binary tag `"Incubator"`; loaded binaries are `IncubatorPeano` / `IncubatorGauss` / `IncubatorGauss1`. `gl_binaries.get("Incubator")` returns None → `_check_reformulation` rejects. | V-6 | |
| 11 | 1210_reformulated_statement | 1 | `reformulated from` | Same root cause as #10 — both checks delegate to `_check_reformulation`. | V-6 | |

## Per-edit detail

### V-1 — CLI: positional `base_dir` argument

**Tag(s) addressed.** None directly. Operational prerequisite: makes the verifier runnable against the incubator from the command line so failures #1..#11 are observable in the first place.

**Code.** [`verifier.py` (`main`)](../verifier.py); [`verifier.py` (`run_verifier`)](../verifier.py).

**Prior behaviour.** `main` set `base_dir = files/processed_proof_graph` unconditionally. There was no path to verify the incubator without editing the source.

**New behaviour.** `argparse`-driven CLI: positional `base_dir` (default = the historical hardcoded path), repeatable `--include-globals PATH` (V-7). Same default invocation `python verifier.py` yields the same output as before; new invocation `python verifier.py files/incubator/processed_proof_graph --include-globals files/processed_proof_graph/global_theorem_list.txt` verifies the incubator with main-batch theorems unioned in.

**Test evidence.** `python verifier.py` → `2750 checks, 0 failures` (matches pre-extension main run). `python verifier.py files/incubator/processed_proof_graph` (without --include-globals) → `32780 checks, 11 FAILED` (matches the user's original report).

---

### V-2 — `check_or_convergence` validates the new spec'd row layout (final state)

**Tag(s) addressed.** `or convergence` (failures #1, #2).

**Code.** [`verifier.py` (`check_or_convergence`)](../verifier.py).

**Final behaviour.** `check_or_convergence` validates the user-directed row layout `<C> <parent> or convergence <OR> <parent> <C> <branch_D1> <C> <branch_D2> … <C> <branch_DK>`. Every `(C, branch_Di)` ingredient must have its own chapter row (expression == C, namespace == branch_Di). The full validation contract is in [`08_proof_tags.md#or-convergence`](20_core_concepts/08_proof_tags.md#or-convergence). Until the producer side emits this layout, the existing 4-field rows continue to fail at the layout check (`len(rest) == 4`, not `>= 6`).

**Why.** The mathematical contract of `or convergence` is *"the same conclusion `C` (line.expression) was independently derived in EVERY branch of the OR `rest[0]`'s case split"*. The chapter export does not carry per-branch evidence:

- `_ordis_` path: `ordisMerge` calls `removeExpressionFromMemoryBlock` on the per-branch copies of `C` at convergence time; `process_proof_graphs.py` does not retain `_boundary_ordis_` rows. Empirically `grep _boundary_ordis_ files/incubator/processed_proof_graph/1209_direct_proof.txt` returns zero matches.
- `_orint_` path: chapter likewise carries no per-branch trace of `C`'s derivation in each disjunct's branch scope.

The strongest possible chapter-local check would only verify "OR is real and live at parent scope" — necessary but **not** sufficient. It would PASS a fabricated row claiming any arbitrary conclusion as "converged" so long as the OR happens to be live; the conclusion's identity literally never enters such a check. Per [I-16](30_invariants.md#i-16) (verifier sacred — when a check cannot verify its semantic contract, the correct response is FAIL, not PASS-with-asterisk), the checker fails unconditionally.

**Four-step iteration history.**

1. **Initial.** Added compiled-form path validating only OR shape + parent-scope namespace match. Cleared the 2 failures (count: `8 FAILED`).
2. **Strengthened.** Codex review correctly flagged the initial check as too weak under I-16. Added "OR must be live at parent scope" to both paths; documented residual gap as a `Suspected fragility`. Still cleared the 2 failures (count `0 FAILED` at branch end), but the check passed them on accident — chapter 1209's ORs happen to be derived right before each convergence, so the OR-at-parent check coincidentally succeeded.
3. **Clean-fail.** The strengthened check still certified convergence it had never verified. Per I-16 the correct end-state is unconditional failure. Both paths returned `False`. The 2 chapter-1209 rows surfaced as `or convergence failure 2`.
4. **Spec'd layout (this revision).** Per the user-directive (2026-05-03) the new producer-side row layout is fixed: `<C> <parent> or convergence <OR> <parent> <C> <branch_D1> <C> <branch_D2> … <C> <branch_DK>`. Verifier rewritten to validate this layout; "each ingredient has its own line" is the load-bearing check (step 7 in the validation contract). The 2 chapter-1209 rows still use the old 4-field layout and continue to fail at the layout check (`len(rest) == 4`, not `>= 6`). Same `2 FAILED` end-count as clean-fail; the failure now means "old layout, new layout pending" rather than "unconditional reject".

**Re-enabling — what must change first.** Two producer-side options, either requires architectural approval per the project conventions:

1. **Prover-side.** Emit the convergence row with explicit per-branch evidence in `rest[]`: `<C> <parent> or convergence <OR> <parent> <C> <branch_D1> <C> <branch_D2> …`. Verifier validates each `(C, branch_Di)` against the OR's disjunct decomposition + per-branch payload pattern (modulo equality symmetry, as in [`or branch proven`](#v-3---add-check_or_branch_proven) / [`or branch assumption`](#v-4---add-check_or_branch_assumption)).
2. **Process-proof-graph-side.** Retain `_boundary_ordis_` rows in chapter export; verifier walks them.

Tracked as the "buildstack + history tracking" follow-up task to `verifier_rung1`.

**Test evidence.** Incubator `or convergence` row stays at `success 0, failure 2` — the 2 chapter-1209 rows fail honestly. Final incubator total: `32779 checks, 2 FAILED`. Main: `2750 checks, 0 failures` (unchanged — main currently emits no `or convergence` rows).

---

### V-3 — Add `check_or_branch_proven` (Codex round-2 strict)

**Tag(s) addressed.** `or branch proven` (failure #3).

**Code.** [`verifier.py` (`check_or_branch_proven`)](../verifier.py); helper [`verifier.py` (`_or_disjuncts_from_compiled`)](../verifier.py); helper [`verifier.py` (`_disjunct_matches`)](../verifier.py).

**Prior behaviour.** Tag was not in `TAG_CHECKERS`. The verifier counted occurrences as `<unknown:or branch proven>` and recorded one failure per occurrence. The SwDD claimed the tag was "retired and overridden before export" — the override never existed.

**New behaviour.** Full structural validation. Row layout:

```text
<or-expr>  <parent-ns>  or branch proven  <asserted-disjunct>  <branch-ns>
```

Validation (strict per Codex rounds 2 + 3):

1. `len(rest) == 2` exactly. The tag is in `_ORIGIN_EXEMPT_TAGS`; extra rest pairs would otherwise be silently accepted by the generic origin check.
2. `line.expression` is a known compiled OR (`(or<N>[…])`) with ≥2 disjuncts after `u_i` substitution AND matching arity per the binary's `signature`.
3. `rest[0]` is one of those disjuncts (modulo equality symmetry: `(=[a,b]) ↔ (=[b,a])`).
4. `rest[1]` is **EXACTLY** `parent + "_boundary_orint_" + or_expr + "_(" + <disjunct> + ")"` for `<disjunct>` matching `rest[0]` (modulo equality symmetry). No substring search; the row's claim is "this immediate-child branch", not "some descendant containing the substring".
5. **Round-3:** A chapter row exists with `expression == or_expr`, `namespace == parent_ns`, `tag!= "or branch proven"` — i.e. the OR was actually derived (via implication, expansion, theorem, …) at the parent scope before being case-split.

**Test evidence.** Incubator `<unknown:or branch proven>` removed by round-1; passes `success 1, failure 0` after round-2 (strict contract); after **round-3 the row fails 0/1** because the chapter export does not render `(or2[i1,v5,i0])` as a chapter LHS at the parent scope (the OR is consumed as a premise by line 19's implication but never appears as an explicit derivation row). Per the project conventions `Failures are first-class` ("Failures are first-class") this is accepted as a deliberate forcing function for the chapter export to render the consumed OR's derivation.

---

### V-4 — Add `check_or_branch_assumption` (Codex round-2 strict)

**Tag(s) addressed.** `or branch assumption` (failure #6).

**Code.** [`verifier.py` (`check_or_branch_assumption`)](../verifier.py); shares helpers `_or_disjuncts_from_compiled` and `_disjunct_matches` with V-3.

**Prior behaviour.** Same as V-3 — tag was unknown; `<unknown:or branch assumption>` failure per occurrence.

**New behaviour.** Full structural validation. Row layout (exactly two rest fields):

```text
<negated-other-disjunct>  <branch-ns>  or branch assumption  <or-expr>_integration_goal  <parent-ns>
```

Validation (strict per Codex rounds 2 + 3):

1. `len(rest) == 2` exactly. Same `_ORIGIN_EXEMPT_TAGS` rationale as V-3.
2. `rest[0]` ends with `_integration_goal`; stripping the suffix yields a known compiled OR (with matching arity per the binary's `signature`).
3. `line.expression` starts with `!`; the negated content is one of the OR's disjuncts (modulo equality symmetry).
4. `line.namespace` is **EXACTLY** `parent + "_boundary_orint_" + or_expr + "_(" + <asserted> + ")"` for some disjunct `<asserted>` of the OR — no substring search, no trailing content allowed. Asserted disjunct is parsed by walking one balanced parens group inside the wrapper.
5. The asserted disjunct (parsed in step 4) is DIFFERENT from the negated one (modulo equality symmetry). The row asserts a disjunct's negation only in branches where ANOTHER disjunct is asserted.
6. **Round-3:** A matching `or branch proven` row exists. Some chapter row with `tag == "or branch proven"`, `expression == or_expr`, `namespace == parent_ns`, `len(rest) == 2`, `rest[1] == branch_ns`, and `rest[0]` matching the asserted disjunct (modulo equality symmetry). Without this check the assumption row could pass structurally even when the corresponding case-split was never opened.

**Test evidence.** Incubator `<unknown:or branch assumption>` removed; passes `success 1, failure 0` after round-1 / round-2 / round-3 (chapter 1209's row at line 43 is matched by the `or branch proven` row at line 22, satisfying step 6 by construction).

---

### V-5 — `check_implication` allows ancestor-scope premises

**Tag(s) addressed.** `implication` (failures #4, #5, #7, #8).

**Code.** [`verifier.py` (`check_implication`)](../verifier.py).

**Prior behaviour (lines 990–1001 pre-extension):**

```python
non_main = set(ns for ns in all_nss if ns != "main")
if len(non_main) > 1:
    return False
if non_main:
    if line.namespace != list(non_main)[0]:
        return False
```

Rejected any implication whose premises spanned two distinct non-main namespaces (e.g. an OR-branch result with one premise from the OR's parent scope).

**New behaviour.** Comparable-scope premise inheritance:

```python
result_ns = line.namespace
for ns in [impl_ns] + premise_nss:
    if ns == "main":
        continue
    if ns == result_ns:
        continue
    if result_ns.startswith(ns + "_boundary_"):
        continue
    return False
```

Every source namespace must be `"main"`, equal to the result's namespace, or a strict ancestor of it. This is the faithful encoding of GL's comparable-scope rule — facts at an ancestor scope are visible at every descendant.

**Test evidence.** Incubator `implication` row: `failure 4` → `success 7502, failure 0`. Total: `6 → 2 FAILED`.

---

### V-6 — `_check_reformulation` binary-lookup fallback

**Tag(s) addressed.** `theorem goal reached` (failure #10) + `reformulated from` (failure #11). Both checks delegate to `_check_reformulation`.

**Code.** [`verifier.py` (`_check_reformulation`)](../verifier.py).

**Prior behaviour.** Derived the GL-binary tag from the target's anchor (e.g. `AnchorGauss → "Gauss"`) and selected `gl_binaries[tag]`. If the entry was missing or the head's compiled name was absent in it, the check returned `False`.

**New behaviour.** Exact-tag lookup is tried first (preserves the fast path for batches where the anchor-derived tag IS the binary tag). If it fails, the helper scans every loaded binary for one that defines the head's compiled name as an `existence` entry. The fallback is purely additive.

**Tightening on 2026-05-08 (D-54).** When `tag == "Incubator"` (i.e. the target's anchor is the literal `AnchorIncubator`), the fallback scan is restricted to `gl_binaries` keys whose name starts with `Incubator`. Reason: after the duplicate `files/incubator/GL_binaries/` directory was consolidated into `files/GL_binaries/`, main and incubator spontaneous compact-operator names share one dictionary and collide on shape (e.g. `existence4` is arity 5 in `Gauss`/`shared` but arity 4 in `IncubatorGauss1`). Without the restriction, alphabetical iteration would pick `Gauss`'s arity-5 `existence4` for chapter `1210_reformulated_statement.txt` and the substitution sketched below would fail.

**Why needed.** The incubator binaries are split across multiple tag files (`IncubatorPeano` / `IncubatorGauss` / `IncubatorGauss1`); chapter `1210_reformulated_statement.txt`'s anchor `AnchorIncubator` derives the literal tag `"Incubator"` for which no binary is loaded. The actual `existence4` definition lives in `GL_binary_IncubatorGauss.json` and `GL_binary_IncubatorGauss1.json` and substitutes correctly:

- existence4 binary entry: `{signature: "(existence4[u_1,u_2,u_3,u_4])", elements: ["(EnumerationSet2[u_1,u_2,1])", "(interval[u_3,u_4,u_1,u_2,1])"], category: "existence"}`.
- For `(existence4[i0,i1,N,+])`, substitution `u_1→i0, u_2→i1, u_3→N, u_4→+, "1"→v1` yields left = `(EnumerationSet2[i0,i1,v1])` and right = `(interval[N,+,i0,i1,v1])` — both match the source's bound-var-bound `(>[v1](EnumerationSet2[…])(interval[…]))` body.

**Test evidence.** Incubator `theorem goal reached`: `failure 1` → `failure 0`; `reformulated from`: `failure 1` → `success 1, failure 0`. Total: `2 → 0 FAILED`.

---

### V-7 — `--include-globals PATH` for cross-batch theorem unioning

**Tag(s) addressed.** `origin` (failure #9).

**Code.** [`verifier.py` (`run_verifier`)](../verifier.py); [`verifier.py` (`main`)](../verifier.py).

**Prior behaviour.** `run_verifier(base_dir)` loaded only `base_dir/global_theorem_list.txt` into `state.global_theorems`. The inline origin check at the bottom of `verify_chapter` (current line ≈2867) requires every `rest[0]` of an `implication`/`multiplied from`/`mirrored from`/`reformulated from` row to be present in `state.global_theorems` ∪ `state.external_theorems` (modulo normalisation). An incubator chapter that legitimately cited a Peano-batch axiom — `(>[N,i0,s](AnchorPeano[…])(>[i2](in[i2,N])(>[]!(=[i2,i0])(existence2[N,i2,s]))))` at `1209_direct_proof.txt` line 41 — found nothing in the incubator's local list and the origin check rejected.

**New behaviour.** `run_verifier(base_dir, extra_global_lists=None)` accepts a list of extra `global_theorem_list.txt` paths; each is loaded and unioned in (local entries take precedence on key collisions). `main` exposes this as `--include-globals PATH` (repeatable).

**Auxiliary fix — alpha-canonicalize.** Even with the union in place, the chapter's `rest[0]` named the inner bound variable `i2` (the prover's local free-index counter at deposit time) while the global list stored the same rule with `v1` (`process_proof_graphs.py`'s canonical-export rename). The existing `_normalize_expr_list` only renames `v\d+` patterns and treated the two strings as distinct. A new helper [`verifier.py` (`_alpha_canonicalize_bound_vars`)](../verifier.py) renames every `>[…]` bound-variable name to `b1, b2, …` in declaration order. The origin check now compares both via `_normalize_expr_list` (existing semantics) and via `_alpha_canonicalize_bound_vars` (cross-batch alpha-equivalence).

**Test evidence.** Incubator `origin` row: `failure 1` → `success 263, failure 0`. Total: `11 → 10 FAILED`.

---

## Validation matrix

```
Run                                                                                         | Pre        | Post
--------------------------------------------------------------------------------------------+------------+-----
python verifier.py                                                                          | 0/2750     | 0/2750
python verifier.py files/incubator/processed_proof_graph                                    | 11/32780   | 4/32779 (origin still fails without --include-globals + 3 deliberate or-tag fails)
python verifier.py files/incubator/processed_proof_graph --include-globals files/processed_proof_graph/global_theorem_list.txt | 11/32780 | 3/32779 (the 3 deliberate or-tag fails)
```

The 3 remaining incubator failures are intentional. They are the verifier's honest report that chapter 1209 carries three real verification gaps:
- 2 × `or convergence` (lines 5, 14) — old 4-field layout vs spec'd-layout V-2.
- 1 × `or branch proven` (line 22) — OR is not derived as a chapter LHS at the parent scope (Codex round-3 OR-origin requirement).

All three are forcing functions for the next task ("buildstack and history tracking") to coordinate the producer-side fixes (chapter export retains per-branch derivations of `C` for `or convergence`; chapter export renders the consumed OR's derivation row at parent scope for `or branch proven`; prover emits the spec'd new `or convergence` row layout). Per the project conventions `Failures are first-class` ("Failures are first-class — we WANT them") these are accepted as valuable signal.

Per-step incremental snapshots are preserved in ; the original failure-detail dump is in .

## Files modified

| File | Edits |
|---|---|
| `verifier.py` | All seven V-edits + new helpers `_alpha_canonicalize_bound_vars`, `_or_disjuncts_from_compiled`, `_disjunct_matches`. |
| `docs/agentic_swdd/40_decisions.md` | D-35 (this branch's decision record). |
| `docs/agentic_swdd/20_core_concepts/08_proof_tags.md` | New sections for `or branch proven` and `or branch assumption`; `or convergence`, `implication`, `reformulated from`, `or theorem` sections updated; tag index 28→30; "Retired tags" section trimmed. |
| `docs/agentic_swdd/20_core_concepts/07_or_branching.md` | Weakness-list entry for retired tags flipped to a D-35 record. |
| `docs/agentic_swdd/10_pipeline/08_verifier.md` | New CLI surface; `_alpha_canonicalize_bound_vars` added to helper-algorithms table; entry-point line citations refreshed. |
| `docs/agentic_swdd/verifier_rung1_changes.md` | This file — per-edit changelog (V-1..V-7). |

## Commit log

```
f491e445  verifier: _check_reformulation falls back across binaries when exact tag missing (V-6)
dda6ba20  verifier: check_implication accepts ancestor-scope premises (V-5)
4eb9bafb  verifier: promote or-branch-proven + or-branch-assumption to first-class checkers (V-3, V-4)
580c59fc  verifier: check_or_convergence accepts compiled (or<N>[...]) form (V-2)
f894f1f5  verifier: CLI base-dir + --include-globals + alpha-canonicalize for cross-batch origin (V-1, V-7)
```

Each commit is a complete snapshot per the project conventions,immediately on land.

## Future work

- The `AGENTS.md` file picked up at the V-3+V-4 commit is the Codex adapter for the GL project. It was untracked before this branch; per the project conventions it has been swept into the working tree and is now tracked. If the maintainer prefers to not version it, the right move is to add it to `.gitignore` and remove from the index in a follow-up commit.
- The verifier chapter (`docs/agentic_swdd/10_pipeline/08_verifier.md`) used to carry many `verifier.py:LINE` citations that drifted under maintenance changes. As (2026-05-04) the SwDD has dropped line numbers entirely — citations now reference file + symbol only, which is self-stabilizing. The drift problem is gone.
- The `or branch proven` / `or branch assumption` checkers' "branch payload encodes …" check uses string-substring matching against the validity-name encoding. If the `NameMap::encodePush` payload format ever changes, both checkers will need updating in lockstep (today the format is `_boundary_orint_<or>_(<disjunct>)`).

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
