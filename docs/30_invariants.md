<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Invariants `[DRAFT]`

> Numbered, named, cross-referenced. Every invariant in this document has:
>
> - A stable anchor (`#i-N`) for cross-linking from chapter weaknesses sections and from [`AGENT_SwDD.md`](AGENT_SwDD.md#invariant-quick-reference).
> - **Scope** — what code paths it applies to.
> - **Rule** — the exact statement.
> - **Why** — the past incident, semantic constraint, or design commitment that motivates it.
> - **Spot** — the symptom(s) that indicate a violation.
> - **Fix** — the correct pattern to restore the invariant.
> - **Code** — the relevant file + symbol references.
>
> Invariants are numbered in the order they entered the document; the numbering is a stable identifier, *not* a priority ordering.

When you (future agent) add an invariant, append a new `## I-N` section, update the quick-reference table in [`AGENT_SwDD.md`](AGENT_SwDD.md#invariant-quick-reference), and do **not** renumber existing invariants — the numbers are referenced from external places (memory files, commit messages, code comments).

---

<a id="i-1"></a>
## I-1 Precompile structural operators on every theorem-load path

**Scope.** Compiler + prover. Every path that feeds a theorem string into `addTheoremToMemory` / `disintegrateImplication` / `addToHashMemory` — the initial load in `analyzeExpressions`, broadcast paths that forward theorems between LBs, and the compressor's proof-pool ingestion.

**Rule.** Before such a call, invoke `ExpressionAnalyzer::precompileStructuralOperators(thm)` so that raw `!(&...)` and `!(>...)` subexpressions are rewritten into their compiled `or<N>` / `existence<N>` names.

**Why.** Uncompiled structural operators become LB `exprKey`s verbatim. The hash engine keys on string form, so two LBs with the same semantic content but one carrying `!(&...)` and another carrying `or3` will never match. Downstream, disintegration asserts that every compound category (`and` / `existence` / `implication`) resolves to an entry in `compiledExpressions`; a raw `!(&...)` resolves to nothing, and the disintegration helper asserts and crashes the core.

**Spot.**

- An LB assertion fires inside `disintegrateExprCore2` or a similar disintegration helper, with an `exprKey` visibly containing `!(&` or `!(>` at string level.
- Silent CE-filter divergence: the same theorem admitted to one LB matches its hash target, admitted to another LB does not — because one LB got the compiled form and the other did not.

**Fix.** Every enumerate-theorems-to-feed-LBs site must filter through `precompileStructuralOperators`. The theorem-load paths currently known to honour this:

- Initial load of `compressed_external_theorems.txt` — at the entry of the prover.
- Broadcast paths — `mailOut.statements` preparation sites.
- Compressor pool — `Compressor::run` ingestion.

If a new path is added (e.g. a second external-theorem file, or a merging utility), it must call `precompileStructuralOperators` as the *first* operation on the raw string.

**Code.** `ExpressionAnalyzer::precompileStructuralOperators` at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp). Referenced from the project conventions rule set.

---

<a id="i-2"></a>
## I-2 Non-`main` validityName minted only via `NameMap::encodePush`

**Scope.** Validity-stack machinery in every LB.

**Rule.** Every per-LB `validityName` other than the literal `"main"` root must be minted via `NameMap::encodePush(parentId, payload)`, where `parentId` resolves through `MAIN_ID` (either directly or via a chain of previous `encodePush` calls). Raw string concatenation of scope names — for example `"foo_" + validityName` — bypasses `pairMap` / `stackOfValidity` registration, produces orphan roots, and breaks `comparable` / `deeperOf`.

**Why.** Scope-depth comparisons and ancestor queries depend on `pairMap` being the authoritative source of parent-child relationships. An orphan scope (never registered in `pairMap`) yields false answers to "is scope A deeper than scope B?", which corrupts admission logic and OR-branch bookkeeping.

**Spot.**

- A validity-scope comparison returns an unexpected answer during OR-branch handling.
- A statement is emitted into a scope that was never registered — e.g. it disappears under escalation because its ancestors are not recognised.

**Fix.** When introducing a new scope kind (hypothesis, integration goal, OR branch, sentinel), encode the scope *role* in the `encodePush` payload prefix. Do not synthesize the full scope-name string by hand elsewhere. Extract payloads via `idToSub[stack.back]` and **copy** the result (see [I-3](#i-3)).

**Code.** `NameMap::encodePush` at [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp).

---

<a id="i-3"></a>
## I-3 NameMap decode / idToSub return references — copy before any nested mint

**Scope.** Anywhere a payload or decoded scope-name is read from `NameMap::decode` or `idToSub[...]`.

**Rule.** Both `NameMap::decode` and `idToSub[]` return references into `std::vector<std::string>`. Always **copy** the return value immediately into a local `std::string` before any nested call that could cause a `push_back` on the underlying vector.

**Why.** Any nested mint path (`encodePush`, `encodeExpression`, `nameMap.encode(...)`, …) may trigger a `push_back` that reallocates the vector. A reference taken before the reallocation then dangles, and subsequent reads produce garbage — or, worse, read memory that the allocator has handed out to another vector entry.

**Spot.**

- Mysterious string corruption in a scope-name or payload during a path that involves nested encoding.
- Intermittent crashes in scope-comparison paths that are not reproducible on every run.

**Fix.** Pattern:

```cpp
// WRONG — dangling reference after nested mint
const std::string& payload = nameMap.decode(stack.back());
doSomethingThatMayMintAnotherScope();
// payload now may be garbage

// RIGHT — copy before nested calls
std::string payload = nameMap.decode(stack.back());   // explicit copy
doSomethingThatMayMintAnotherScope();
// payload still valid
```


---

<a id="i-4"></a>
## I-4 `reconstructImplicationFullBind` only at disintegration/integration sites

**Scope.** Prover, compiler — expression reconstruction.

**Rule.** Use `reconstructImplicationFullBind` **only** for expressions that have been expanded from GL-binary definitions — that is, expressions where `u_` variables mark the formal parameters that must stay free, and all other (numbered) variables must be universally quantified. Current FullBind call sites: `disintegrateExprCore2` (main disintegration path) and the integration helpers `expandSignatureForIntegration`, `buildIntegrationInstruction`.

**Why.** Without FullBind in these contexts, the verifier's implication checker treats unbound numbered variables as unchangeable constants, breaking structural matching during disintegration/integration.

**Spot.**

- Disintegration produces an implication that the verifier subsequently rejects with a structural-mismatch message on `implication` or `expansion for integration` tags.
- Integration produces an implication that refuses to match even though the underlying semantics should succeed.

**Fix.** If the expression is a product of GL-binary expansion (has `u_` variables), use FullBind. Otherwise (theorem-level reconstruction), use the non-FullBind [`reconstructImplication`](#i-5).

**Code.** `reconstructImplicationFullBind` at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp). the project conventions.

---

<a id="i-5"></a>
## I-5 `reconstructImplication` (non-FullBind) for theorem-level reconstruction

**Scope.** Prover — theorem-level reconstruction sites.

**Rule.** Use `reconstructImplication` (non-FullBind) for theorem-level reconstruction — specifically at `addTheoremToMemory`, `updateGlobalDirect`, and the back-reformulation path. This variant binds only variables that appear more than once across chain+head in `>[...]`. Variables appearing once are left free.

**Why.** At theorem-level, single-occurrence "variables" are fixed entities — typically anchor function symbols (`s`, `+`, `*`) that must not be bound. Binding them would turn a concrete theorem (applying to `+`) into a universal statement over all possible function symbols, which is semantically wrong and changes the hash signature.

**Spot.**

- A newly-proven theorem fails to match a theorem cite in a later chapter even though the expressions are structurally identical — because FullBind was used where non-FullBind was required, and a single-occurrence `s` was wrongly bound.

**Fix.** Use `reconstructImplication` (non-FullBind) at the three theorem-level sites listed above. Anywhere else involving GL-binary-expanded material, use FullBind — [I-4](#i-4).

**Code.** `reconstructImplication` at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp).

---

<a id="i-6"></a>
## I-6 Pass B single-input-operator gate — do not widen

**Scope.** Prover — Pass B admission.

**Rule.** The standalone fallback admission rule `isAllowedAsOperatorInput` in `prover.hpp` — the non-map-based path for `it_…` (Pass B) variables — must fire only for operators with `cfg.inputIndices.size == 1`. In practice this means `in` and `in2`. Widening to multi-input operators (`in3`, `fold`, `residual`, `interval`, `preorder`) broke Gauss summation.

**Why.** `isAllowedAsOperatorInput` admits a variable under a minimal guard set (single-input only, the lone input-arg position holds the variable, standard max-iteration guards). This guard set is insufficient for multi-input operators — without the full admission-map reasoning, multi-input operators fire far too eagerly and explode the RT surface.

**Spot.**

- A Gauss batch suddenly produces a wall-clock regression with the prover consuming many seconds per theorem where previously it took a fraction of a second. This is the classic symptom.
- A Pass B proof graph contains iteration variables that were never registered on any admission-map producer path.

**Fix.** Keep the `inputIndices.size == 1` check. When adding a new operator, either ensure it has the single-input shape (and rely on this gate), or ensure its admission map is populated from consumer-side registrations via `updateAdmissionMap`. Do not add a new operator to the fallback rule without careful RT measurement.

**Code.** `isAllowedAsOperatorInput` at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp). the project conventions.

---

<a id="i-7"></a>
## I-7 Pass B guarded by `!parameters.ban_disintegration` (and Pass B + back-reformulation + hypo-disintegration share that flag)

**Scope.** Prover — Pass B entry, plus back-reformulation, hypothetical disintegration, and necessity-for-equality-hypo paths.

**Rule.** The guard that decides whether Pass B runs is `!parameters.ban_disintegration` (combined with `!parameters.compressor_mode`). The same flag also gates back-reformulation ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)), hypothetical disintegration ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)), and necessity-for-equality-hypo ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)) — every disintegration-shaped path in the prover. Pass B used to gate on `!parameters.incubator_mode` until 2026-04-29; the short-lived `parameters.allow_disintegration` flag introduced earlier on 2026-04-29 was collapsed into `ban_disintegration` later the same day after the per-config matrix turned out to be perfectly symmetric (`ban_disintegration == !allow_disintegration` in every existing config). See [D-27](40_decisions.md#d-27) and [D-28](40_decisions.md#d-28).

**Why.** Pre-2026-04-29, every config that needed Pass B had to also be a non-incubator config, and every incubator config lost Pass B as a consequence. That foreclosed the FTA-ladder rung-1 §4.1 proof, which needs Pass B (for `EnumerationSet2` / `interval` body disintegration) AND `multiplyImplication` off AND `incubator_mode` true (for contradiction LBs / skip CE filter). The decoupling lets a config independently set them. The `incubator_mode`/`ban_disintegration` collapse keeps the flag set minimal — `incubator_mode` still governs the other things it always did (`head-in-wholeExpressions`, OR-generation suppression, integration reformulation, tempArgs assert, conjecturer behaviours), but no longer governs disintegration.

**Spot.**

- Pass B unexpectedly fires (or doesn't fire) for a config that has `incubator_mode=true`. Check `ban_disintegration` first, not `incubator_mode`.

**Fix.** The guard must read `if (!parameters.compressor_mode && !parameters.ban_disintegration)` at the Pass B entry. Do not substitute `!incubator_mode`.

**Code.** Pass B entry at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp). Sibling decoupling for `multiplyImplication` lives at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp), gated by `parameters.allow_multiplication` (with the `!ceFilteringActive` carve-out preserved).

**Local-premise refinement (D-29, 2026-04-29).** Inside `checkLocalEncodedMemoryStatic` at [`memory.cpp`](../GL_Quick_VS/GL_Quick/src/memory.cpp) (the body moved from `prover.cpp`, 2026-05-04), when both `parameters.incubator_mode == true` and `!parameters.ban_disintegration` (the SE2-migration combination), disintegration of a matched rule's head is additionally gated by a two-part rule:

1. **Anchor LB always blocks.** If `memoryBlock.exprKey` starts with `"(" + anchorInfo.name` (i.e. this is the anchor LB itself), `doNotDisintegrate = true` unconditionally. The anchor LB never disintegrates in this mode — its job is to handle external rules and broadcast their conclusions to descendants without local fan-out.
2. **Non-anchor LB needs a local premise.** Otherwise, gate on at least one of the matched premises being in the LB's `localEncodedStatementsSet`. The set is populated by both the LB's own work AND `prehandleAnchor` (the anchor x-prefixed `(in[X, N])` deposits at `prover.cpp/8236`); without rule (1), the anchor LB's own anchor-deposit-saturated set would always satisfy this check and the gate would be inert.

Without this gate the prover crashed with NameMap exhaustion at `prover.hpp` after burst 2 (~4570 expressions) when fold/sequence/etc. external rules from a prior incubator batch fanned out across local `(in[X, N])` rows. With the gate, the same prove pass plateaus at ~7730 expressions and runs to completion. The gate is dormant for every other config. See [D-29](40_decisions.md#d-29).

---

<a id="i-8"></a>
## I-8 Trivial equality `(=[x,x])` forbidden in head only, allowed in premises

**Scope.** Conjecturer + prover.

**Rule.** Conjectures whose *head* is the trivial equality `(=[x,x])` (same variable on both sides) must be rejected. Trivial equalities in premises are fine — they reduce to a tautological hypothesis that the prover simply discharges.

**Why.** `(=[x,x])` as a head is universally true independently of any premise — the conjecturer would generate an explosion of useless tautology theorems otherwise.

**Spot.**

- Conjecturer output `theorems.txt` contains `(=[v1,v1])` heads.

**Fix.** Guard in the conjecturer's head-construction path: reject any proposed head where both arg-positions of `(=[...])` resolve to the same variable name.

**Code.** Conjecturer filter; see [`10_pipeline/02_conjecturer.md`](10_pipeline/02_conjecturer.md) `[STUB]`. the project conventions for the equivalent incubator-mode handling.

---

<a id="i-9"></a>
## I-9 Equality mirror guarded by `args[0]!= args[1]`

**Scope.** Compiler — mirror generation.

**Rule.** When generating a mirrored form of a theorem, the equality mirror emission is guarded by `args[0]!= args[1]`. Trivial equalities are not mirrored.

**Why.** Mirroring `(=[a,b])` produces `(=[b,a])`, which is semantically distinct. Mirroring `(=[a,a])` produces `(=[a,a])` again — no new information. Emitting the identical mirror wastes space and can silently break deduplication checks.

**Spot.**

- Duplicate mirror entries in `reshuffled_mirrored_theorems.txt`.

**Fix.** Preserve the `args[0]!= args[1]` guard in `createReshuffledMirrored`.

**Code.** `createReshuffledMirrored` at [`compiler.hpp`](../GL_Quick_VS/GL_Quick/src/compiler.hpp).

---

<a id="i-10"></a>
## I-10 Chapter v-numbering seeded from theorem expression

**Scope.** `process_proof_graphs.py` — variable renaming.

**Rule.** The renamer must assign `v1`, `v2`, … to non-anchor variables by scanning the **theorem expression** left-to-right *first*, then continuing with any remaining variables found in chapter lines. This seeding ensures chapter and global theorem list share the same v-numbering.

**Why.** Without seeding from the theorem, the chapter might assign `v1` to a variable that appears first in chapter lines — which is a different variable from the one `v1` refers to in `global_theorem_list.txt`. Verifier checks that cite the theorem by v-number then spuriously fail.

**Spot.**

- Verifier fails `task formulation` or `theorem` checks with "expected v1, got v3" messages.

**Fix.** Preserve the four-priority renaming in `process_proof_graphs.py` (anchor → theorem scan → chapter scan → `_copy` derivation) — see the project conventions "Variable renaming in process_proof_graphs.py" for the full specification.

**Code.** `process_proof_graphs.py`. See [`10_pipeline/06_process_proof_graph.md`](10_pipeline/06_process_proof_graph.md) `[STUB]`.

---

<a id="i-11"></a>
## I-11 Anchor variables must not appear in `>[...]` bound-variable lists

**Scope.** Theorem construction — prover + conjecturer.

**Rule.** The anchor-slot names (`N`, `i0`, `s`, `+`, `*`, `i1`, …) must never appear in the bound-variable list `>[...]` of any implication in a theorem. They are free; the anchor application at the outermost level supplies them.

**Why.** If an anchor-slot name is bound inside a theorem, the theorem no longer refers to *the* anchor (the unique `AnchorPeano` instance fixed by the anchor atom) — it quantifies over all possible instantiations of that slot. That changes the theorem's meaning and breaks the anchor-handling tag chain.

**Spot.**

- `anchor handling` checker fails, or an `implication` check fails with anchor-slot variables listed in a bound set.

**Fix.** Conjecturer and prover must keep anchor-slot names out of bound-var lists by construction. Verifier enforces this structurally via `_find_digit_args` / `_find_immutable_args`.

**Code.** `_find_digit_args` at [`verifier.py`](../verifier.py); `_find_immutable_args` at [`verifier.py`](../verifier.py).

---

<a id="i-12"></a>
## I-12 `addStatement` applies equivalence classes to `!(=[a,b])` one-sidedly

**Scope.** Prover — `addStatement`.

**Rule.** When `addStatement` receives a negated equality `!(=[a,b])`, it applies equivalence classes to emit sibling inequalities, but only one-sidedly:

- For each `c ∈ class(a) \ {a}` — emit `!(=[c, b])`.
- For each `d ∈ class(b) \ {b}` — emit `!(=[a, d])`.

The symmetric cross-product (both args substituted simultaneously) is **deliberately not emitted**.

**Why.** Two-sided expansion would blow up combinatorially (|class(a)| × |class(b)| emissions per input) without clear semantic gain — anything the cross-product would conclude is already derivable by two one-sided steps.

**Spot.**

- A proof unexpectedly stalls because a two-sided-expansion fact that should be derivable is not being emitted.

**Fix.** Before changing this, profile the alternatives. The two-sided form has never been implemented, and the one-sided form is the load-bearing simplification.

**Code.** `applyEquivalenceClassToNegatedEquality` at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp). Caller: `addStatement` at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp).

---

<a id="i-13"></a>
## I-13 `ChunkPool` uses static `char[]` — never `malloc`/`new`

**Scope.** Low-level memory management.

**Rule.** Any hot-path allocator used by the prover's core loops must use a static `char[]` array — not `malloc`, not `new`, not `std::vector` internals that might reallocate.

**Why.** Hot-path allocations in the inner loop contend with mimalloc's caches and cause latency spikes. The `ChunkPool` pattern pre-allocates a static buffer and hands out chunks from it — fixed, predictable, no heap interaction after startup.

**Spot.**

- Inner-loop timing has a long tail with occasional multi-millisecond allocation spikes.
- A refactor introduces a `std::vector<char>` in a path that used to use a static buffer.

**Fix.** Restore the static buffer. The user has requested this pattern explicitly at least four times across prior debugging sessions.


---

<a id="i-14"></a>
## I-14 Never destroy git history

**Scope.** Process.

**Rule.** No `git push --force`, no `git rebase` that removes commits, no deletion of branches with unmerged work. The commit log is sacred and eternal. This is the ONE absolute restriction in the project conventions.

**Why.** Multi-agent collaboration depends on the history being a reliable shared reference. A force-push silently invalidates other agents' local branches and can erase in-flight work.

**Spot.**

- A branch that was at commit `X` on origin no longer contains `X` in its history.

**Fix.** Preventative: never run history-rewriting commands against shared branches. Detection: `git reflog` of the affected local clone is the only recovery.

**Code.** the project conventions.

---

<a id="i-15"></a>
## I-15 `git reset --hard` only; never soft or mixed

**Scope.** Process.

**Rule.** When moving `HEAD` between commits, always use `git reset --hard`. Never use `--soft` or `--mixed`.

**Why.** Soft/mixed resets leave the working tree polluted with stale changes that look like "your" changes but belong to the commit you just moved away from. The next commit then unintentionally rolls them in. Hard reset is the only form that guarantees a clean slate.

**Spot.**

- Unexpected "modified" entries in `git status` after a `git reset`.

**Fix.** `git reset --hard <target>`. If you meant to preserve local changes, stash them *before* the reset.

**Code.** the project conventions.

---

<a id="i-16"></a>
## I-16 `verifier.py` is sacred — failures are real bugs

**Scope.** Verifier.

**Rule.** Never modify `verifier.py` without explicit user consent. When a verifier check fails, it is a real bug, not a "false positive". The correct response is to find the upstream cause in the prover / processor / compiler, not to "make the verifier pass".

**Why.** The verifier is the sole independent oracle for proof-graph correctness. Relaxing a checker to make a failure disappear hides real semantic errors. The verifier's independence (no imports from `expression_utils` or prover code, own copies of every algorithm) is explicitly designed to resist this pattern.

**Spot.**

- A PR diff touches `verifier.py` with the effect of loosening a check.
- A commit message claims "verifier false positive fixed".

**Fix.** If you believe the verifier is genuinely wrong, raise it with the user with a concrete example. Otherwise, treat the failure as a bug in whatever produced the data.

**Code.** See also [`10_pipeline/08_verifier.md`](10_pipeline/08_verifier.md).

---

<a id="i-17"></a>
## I-17 `savedStartInt` freshness check assumes one monotonic counter

**Scope.** Prover — Pass B.

**Rule.** `disintegrateExpr2`'s `savedStartInt` freshness check assumes there is exactly one monotonically-increasing counter feeding the iteration-level numbers. A parallel counter breaks the freshness contract.

**Why.** The freshness check tests whether a level seen now is strictly greater than the saved start — i.e. whether it was produced after this Pass B invocation began. Two counters advancing independently produce values that are not totally ordered, so the freshness test returns wrong answers.

**Spot.**

- Pass B produces iteration variables that should have been rejected as stale, or vice versa.

**Fix.** Maintain a single global counter. If a second counter is needed for a legitimate reason (e.g., a separate numbering scheme for a new scope kind), update `savedStartInt` logic to reason about the Cartesian product, not just the scalar.

**Code.** `disintegrateExpr2` at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp).

---

<a id="i-18"></a>
## I-18 Induction scheduled on a bound variable must first prove its typing

**Scope.** Prover — induction setup.

**Rule.** Before scheduling induction on a bound variable `n`, the prover must prove `(in[n, N])` from the current chain, where `N = anchor_args[0]` — the first slot of the anchor (in Peano, `N` itself). The typing sub-theorem inherits the original chain's non-anchor premises and anchor; only the head changes.

Induction succeeds **iff all three discharges complete**:

1. typing: `(>[bounds](Anchor)(chain_premises)(in[n, N]))` — direct-proof only.
2. base: `(>[bounds](Anchor)(chain_premises[n:= i0])(head[n:= i0]))`.
3. step: `(>[bounds](Anchor)(chain_premises[n:= s(m)], head[m:= n])(head[n:= s(m)]))`.

**Why.** Without the typing discharge, induction on `n` is soundness-unsound for any `n` not in `N` — the proof's range silently widens from "all natural numbers" to "all entities whatsoever". The current branch was created precisely because a negative Peano theorem was going through via this unsound path.

**Spot.**

- A theorem in `global_theorem_list.txt` with method `induction` whose induction variable has no typing derivation in its premises.
- A `check_zero` or `check_induction_condition` chapter with no corresponding `induction_typing` chapter at `<N-1>`.

**Fix.** **Shipped.** The prover now proves `(in[digitArg, N])` as an auxiliary induction triad before promoting an induction theorem. The chapter set carries `<N>_induction_typing.txt` files holding the typing derivation. The verifier's `induction typing` checker walks every `method = induction` row in `global_theorem_list.txt` and verifies its accompanying typing chapter. See [`docs/induction_typing_plan.md`](induction_typing_plan.md) for the historical implementation plan.

**Code.** Implemented in `parameters.hpp` (config flag), `memory.hpp` (induction-triad bookkeeping), `prover.cpp` (induction setup + promotion gate), `prover.hpp`, `visualizer.cpp` (chapter rendering), `generate_full_proof_graph.py` (induction-typing chapter ingest), `process_proof_graphs.py` (chapter emission), `run_modes.py`, `run_modes.cpp`, `verifier.py` (`induction typing` checker registry).

---

<a id="i-19"></a>
## I-19 Asserts are first-class — never weaken or remove to pass a test

**Scope.** Process — C++ and Python alike.

**Rule.** Add `assert(...)` whenever a function has a precondition that the caller already guards. GL must have all states defined; an unguarded-but-expected state is a silent contract. If an assert fires, it has surfaced a previously unknown code path — do not suppress it. The correct response is to understand why that path is reached and fix the caller.

**Why.** GL's correctness is deterministic — every state is definable or the system is in an inconsistent state. Asserts are the contract-enforcement surface. Weakening an assert to pass a test hides the exact bug the assert was designed to catch.

**Spot.**

- A PR comment reading "removed assert because it was firing" — without an accompanying explanation of *why* the caller reached the precondition-violating path.
- A test that previously passed is "fixed" by weakening an assert rather than by fixing the test harness or the code under test.

**Fix.** Find the upstream caller. Document why it reached the asserted-violating state. If the assert is genuinely wrong, discuss with the user before changing it.

**Code.** the project conventions.

---

<a id="i-20"></a>
## I-20 Auto-commit + push on source/config changes, detailed message

**Scope.** Process.

**Rule.** After each prompt that modifies source code or configs (`*.cpp`, `*.hpp`, `*.h`, `*.py`, definition files in `files/definitions/`, config JSON in `files/config/`):

- `git add <changed files> && git commit -m "<prompt summary>"`.
- Verify with `git status` that the commit succeeded.
- Push the current branch: `git push -u origin <current-branch>`.

Commit message must include: (1) one-line summary of what changed, (2) root cause or motivation — why this change was needed, (3) what was investigated/ruled out during debugging if applicable, (4) before/after results if measurable.

HTML or other non-source files do **not** require auto-commit.

**Why.** Multi-agent collaboration on the private origin depends on every source change being visible promptly. A detailed commit message is the only artifact a future agent has to reconstruct *why* a change was made — code shows *what*, message shows *why*.

**Spot.**

- A local branch diverges from origin by many unpushed commits.
- A commit message reads only "fix bug" with no motivation.

**Fix.** Follow the rule as stated. If a commit goes in with a terse message, the next commit's message should include the missing motivation for the prior one.

**Code.** the project conventions.

---

<a id="i-21"></a>
## I-21 `internalMailIn` cleared at top of hashburst after absorb (not end)

**Scope.** Prover — `performElementaryLogicalStep` and the hashburst entry.

**Rule.** `Memory::internalMailIn` (integration-revival channel) is cleared **immediately after the top-of-hashburst absorb loop** — **not** at end of hashburst. Inserts during the hashburst body (from `applyEquivalenceClassToRejectedMapIntegration` and `revisitRejectedIntegration2`) must survive to the NEXT hashburst. This asymmetry vs. legacy `mailIn` (cleared at end of hashburst, `prover.cpp`) is intentional.

**Why.** The cleanup timing differs because the *producer* timing differs. Legacy `mailOut → smashMail → mailIn` populates `mailIn` between cycles at the mail-drain step; clearing at end of cycle N empties the inbox after cycle N has absorbed it, so cycle N+1 starts empty and receives fresh routing. `internalMailIn` is populated *during* a cycle's body (via eq-class rewrites and admission-key revisits triggered inside `addStatement`). If cleared at end of cycle, entries produced mid-cycle would be lost. Clearing at top, immediately after absorb, guarantees that every insert during a cycle's body lands intact in the next cycle's inbox.

**Spot.**

- A `body.internalMailIn.statements.clear` call added in the same block as `body.mailIn.statements.clear` at `prover.cpp`. That is the *wrong* placement for `internalMailIn`.
- Revival statements missing after an equi-class rewrite — next hashburst's absorb sees an empty `internalMailIn` when traces show mid-cycle inserts did happen.

**Fix.** Clear `internalMailIn.statements` and `internalMailIn.exprOriginMap` in the top-of-hashburst absorb block, right after the absorb loop, not at the bottom. See `prover.cpp` in the internal-mail absorb section.

**Code.** `memory.hpp` — `struct Mail` used as `Memory::internalMailIn` (typed `Mail` since 2026-05-07; [D-53](40_decisions.md#d-53), renumbered from main's D-46); `prover.cpp` hashburst entry absorb-then-clear block.

---

<a id="i-22"></a>
## I-22 `rejectedMapIntegration` entry NOT cleaned on successful revival

**Scope.** Prover — `revisitRejectedIntegration2` + `applyEquivalenceClassToRejectedMapIntegration`.

**Rule.** When an integration-side rejection is revived (via equi-class rewrite or new-admission-key trigger), the `admissionMapIntegration` / `admissionSetIntegration` entry that enabled the revival is **not** erased afterwards. Asymmetric with algebra: `revisitRejected2` (the algebra counterpart) calls `cleanAdmissionMap` after successful revival; the integration analog deliberately does not.

**Why.** A single admission template can admit multiple distinct `int_` witnesses over the proof lifetime. The admission rule is a *template*; consuming it once does not exhaust it. Cleaning it would disable future revivals of different compound shapes whose rewritten keys happen to match the same template. Algebra can afford to clean because its revisit goes through `addExprToMemoryBlock` which can re-populate admission maps on future need; integration revival is mailIn-only and has no such re-population pathway — once gone, the template is unrecoverable.

**Spot.**

- Any `cleanAdmissionMap(...)` call added inside `revisitRejectedIntegration2` or inside the match-branch of `applyEquivalenceClassToRejectedMapIntegration`.
- A second integration revival of the same marker-template failing because the admission entry was silently removed by a previous revival.

**Fix.** Do not add cleanup. The `rejectedMapIntegration` entry itself IS erased after revival (the rejection is resolved), but the admission-map entry stays live.

**Code.** `prover.cpp::revisitRejectedIntegration2`; `prover.hpp::applyEquivalenceClassToRejectedMapIntegration` match branch.

---

<a id="i-23"></a>
## I-23 Spontaneous compact operator names are stable across batches

**Scope.** C++ compiler (`ExpressionAnalyzer` constructor + `excludeRepetitions` + `compileCoreExpressionMapCore`) and Python pipeline (`run_modes.py::_run_batch`).

**Rule.** Every spontaneous operator name (`implication<N>`, `existence<N>`, `or<N>`, `and<N>`) allocated by any batch in a run keeps the same `<N>` for every later batch in the same run. The shared registry file `files/GL_binaries/GL_binary_shared.json` is the canonical source of truth; per-batch files `GL_binary_<Tag>.json` are seeded from shared before each `gl_quick.exe <Tag>` invocation, and any new entries the batch creates are merged back into shared after the invocation returns. The four spontaneous-operator counter members (`implCounter`, `existenceCounter`, `andCounter`, `orCounter`) are seeded from the maxima of names already in the per-batch file rather than zeroed; only `variableCounter` resets to zero per batch.

**Why.** The proof-graph emission pipeline keys on string-form compact names. `compiled_proved_theorems.txt` (the Python pruner's essential set) and `raw_proof_graph/global_theorem_list.txt` (the Python pruner's all-theorems set) must use the same name for the same logical fact, otherwise `_prune_proof_graph` computes an empty intersection for that theorem and drops both forms; chapter contents that cite the Gauss-batch internal name then fail the verifier's origin check because the renamed template is absent from `state.global_theorems`. The chapter-85 successor-existence failure under sandbox/cancellation_v2 was the concrete incident. See [D-22](40_decisions.md#d-22) for the architecture record.

**Spot.**

- A spontaneous operator appears in `compiled_proved_theorems.txt` under one name (e.g. `existence3`) and in `raw_proof_graph/global_theorem_list.txt` under a different name (e.g. `existence2`) for the same structural form.
- `process_proof_graphs.py:_prune_proof_graph` drops a theorem because `essential ∩ all_thm_exprs` is empty.
- The verifier reports an `origin` failure citing an implication template `(>[…](AnchorXxx[…])(<spontaneous>[…]))` that is absent from `files/processed_proof_graph/global_theorem_list.txt`.
- `loadGlBinary` log line is missing or shows `loaded 0 entries` after the first batch of a run that used spontaneous operators in an earlier batch.

**Fix.** Verify `_seed_per_batch_binary` runs immediately before every `run_gl_quick(tag)` call and `_merge_into_shared` runs immediately after; verify `loadGlBinary` is called from the `ExpressionAnalyzer` constructor before any compilation work; verify `precompileStructuralOperators` was called only after the loader has populated `repetitionExclusionMap` so `excludeRepetitions` finds existing entries. If a config-level change has invalidated existing shared entries (e.g. an `existence_variable_position` change), delete `files/GL_binaries/GL_binary_shared.json` manually and rerun — there is no automatic staleness detection. Never edit `GL_binary_shared.json` by hand to "fix" a name collision; if the registry contains an unwanted entry, wipe and rerun.

**Code.** `GL_Quick_VS/GL_Quick/src/visualizer.cpp::loadGlBinary`; `GL_Quick_VS/GL_Quick/src/prover.cpp` constructor (counter init + loader call); `run_modes.py::_seed_per_batch_binary`, `_merge_into_shared`, and `_run_batch`.

---

<a id="i-24"></a>
## I-24 `multiplyImplication` may not equate two distinct free `u_*` anchor parameters

**Scope.** Prover — `ExpressionAnalyzer::multiplyImplication` and the Python verifier's `check_equalize_variable`.

**Rule.** When `multiplyImplication` enumerates Bell partitions over the `(1)`-typed argument set of a rule, any partition whose equivalence classes contain two or more distinct free `u_*` anchor parameters in the same class must be skipped. Bound→bound merging (the standard Bell partition use case) and bound→free merging (specialising a bound variable to a known anchor) are both permitted; only free→free with distinct names is forbidden. The verifier enforces the same gate on the Python side as a defence in depth: `check_equalize_variable` parses every `>[…]` binder to classify each name as bound or free, then rejects any `(orig_arg → copy_arg)` mapping where both ends are free and the names differ.

**Why.** Free `u_*` parameters are pre-bound to specific elements of the rule's underlying definition sets. Equating two distinct ones rewrites a free slot of the rule body and emits a logically stronger rule than the source — provable conclusions then include theorems the source rule never authorised. The chapter-1115 incubator regression (HTML title *"sum(i=0..0) ≠ 0"* — a mathematically false claim derived through this exact path) is the load-bearing incident; commit cf358271 had deliberately removed an earlier `if (hasDoubleU) continue;` skip while pursuing a u\_-equalisation extension, and the unsoundness sat dormant on Peano/Gauss until the FTA-ladder branch exercised the bad shape. See [D-25](40_decisions.md#d-25) and the prover-chapter [Soundness gate](10_pipeline/04_prover.md#multiplyimplication-bell-partition-equalisation) write-up.

**Spot.**

- A `multiplied from` chapter row whose source and copy differ in a free anchor slot — e.g. source `existence2[v2,N,0,id,0,1,s,+]`, copy `existence2[v2,N,0,id,0,0,s,+]` (slot 6 collapsed `1 → 0`).
- The Python verifier emits a `multiplied from` failure with the row payload above.
- A theorem appears in `proved_theorems.txt` whose statement is mathematically false and whose chapter graph traces back through a `multiplied from` step that touches a free anchor.

**Fix.** Re-enable the `hasDoubleU` skip inside `multiplyImplication` (commit 992fe2b1 restored it). Independently, keep the `check_equalize_variable` free-anchor-merge guard live (commit 38701644) so that any future regression that bypasses the prover-side gate is caught before the chapter ships.

**Code.** `multiplyImplication` at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp); `hasDoubleU` skip at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp). Verifier guard: `check_equalize_variable` at [`verifier.py`](../verifier.py); `_extract_bound_vars` helper colocated.

---

<a id="i-26"></a>
## I-26 Mail-out implications/statements are MAIN-ONLY; mail-out exprOriginMap is ALL-SCOPES

**Scope.** Mail subsystem — [`addStatement`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`addExprToMemoryBlock`](../GL_Quick_VS/GL_Quick/src/prover.cpp), and every site that inserts into `Memory::mailOut.implications` / `Memory::mailOut.statements`.

**Rule.** A non-main expression must NEVER enter `mailOut.statements` or `mailOut.implications`. Both channels are gated at the sender:

- **Statements**: `addStatement`'s `local`-branch push into `mailOut.statements` is conditional on `validityName == "main"`.
- **Implications**: `addExprToMemoryBlock`'s post-disintegration loop at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) gates the `mailOut.implications.insert` on `impValidity == "main" && allowedForMail(impStr, memoryBlock)`. The `impValidity` is each imp's actual scope as returned by `disintegrateExpr2`'s `imps` set, NOT the function-parameter `validityName`.

`mailOut.exprOriginMap` is the exception: it carries entries for ALL scopes per `trackExpansionHistory`'s convention. Receivers' merged `body.exprOriginMap` therefore covers the sender's full provenance graph — main, hypothetical, ordis-branch, and orint-branch alike — even though the rules themselves stay local to the sender at non-main scopes.

**Why.** Hashmem rules are conditional on the hypothesis stack at their install scope. A rule at `v=main_boundary_(impl24[…])` is conditional on the impl24 hypothesis being active; shipping it to another LB and reinstalling at hardcoded `v=main` (the receiver's `mailIn.implications` absorb at [`prover.cpp` `performElementaryLogicalStep`](../GL_Quick_VS/GL_Quick/src/prover.cpp)) discards that conditionality and produces unsound rule firings — and, secondarily, breaks visualizer's `buildStack` walk because the firing recorder writes the dep at `(rule, "main")` while the rule's `exprOriginMap` entry came in keyed at the sender's deeper scope. The `internalMailIn` channel is the per-LB cross-cycle inbox that DOES carry validity for non-main revival messages — see [I-21](#i-21).

**Spot.**

- A new `mailOut.implications.insert` site that does not gate on `validityName == "main"` (or its equivalent).
- A new `mailOut.statements.insert` site whose `ExpressionWithValidity` is constructed with anything other than the literal `"main"` for `validityName` — every routing-channel sender must wrap with `ExpressionWithValidity(expr, "main")` (the sender's own `validityName == "main"` gate guarantees the surrounding scope, but the EWV constructor argument is what gets shipped).
- A change that adds a validity field to the `mailOut.implications` 5-tuple beyond the existing 5th element. The tuple's 5th element carries OR-derivation context (`orImpl`/`theorem`), not install scope — adding a parallel validity field is a code-smell that the install gate is being bypassed.
- A receiver-side `addToHashMemory` in the `mailIn.implications` absorb that uses anything other than `"main"` for the install validity. The hardcoded `"main"` is the contract counterpart of the sender gate.
- A new `mailIn.statements` consumption site (or a refactor of the existing CE-mode / normal-mode absorb loops) that does not assert `it->first.validityName == "main"` before consuming the EWV. The runtime enforcement of I-26 for the statements channel is the per-element `assert(vName == "main")` carried inside both absorb loops in `performElementaryLogicalStep`. The `mailIn.statements` shape is `pair<ExpressionWithValidity, levels>` per [D-53](40_decisions.md#d-53) (renumbered from main's D-46); the EWV's `validityName` is read at consumption rather than hardcoded because the non-`"main"` branch of the same code shape is exercised by the per-LB `internalMailIn` channel on a separate code path (the top-of-burst drain block has no main-only assert).

**Fix.** Add the `impValidity == "main"` gate at the sender. If a non-main rule legitimately needs to cross LB boundaries, route it via `internalMailIn` (the integration-revival channel, which DOES carry validity in the tuple) — but only if the receiver actually needs the rule at the same non-main scope, which is rare.

**Code.** Sender gates: [`prover.hpp::addStatement`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (`mailOut.statements` site, plus `addEquality` / `addNegatedEquality` / `addAnchorHandling` / `applyEquivalenceClass` deposit branch — every routing sender wraps with `ExpressionWithValidity(expr, "main")`), [`prover.cpp::addExprToMemoryBlock`](../GL_Quick_VS/GL_Quick/src/prover.cpp) (`mailOut.implications` site). Receiver absorbs: [`prover.cpp::performElementaryLogicalStep`](../GL_Quick_VS/GL_Quick/src/prover.cpp) — the `mailIn.statements` CE-mode and normal-mode absorb loops carry `assert(vName == "main")` per-element as the runtime enforcement of the statement-channel contract; the `mailIn.implications` block hardcodes `"main"` directly. History: implication-side gate landed with [D-34](40_decisions.md#d-34) in 2026-05-02; pre-D-34 the implications side was missing the gate, ran latent until the FTA-rung-1 §4.1 chain reached contradiction-LB closure, then crashed `visualizer.cpp buildStack`. Statement-side EWV migration + receiver assert landed 2026-05-07 ([D-53](40_decisions.md#d-53), renumbered from main's D-46); pre-migration the receiver hardcoded `"main"` for both channels.

---

<a id="i-27"></a>
## I-27 Site F / Site H — ancestor-scan dedupe at `addExprToMemoryBlock` entry

**Scope.** Prover — [`addExprToMemoryBlock`](../GL_Quick_VS/GL_Quick/src/prover.cpp) at the function entry, before any disintegration / kernel logic.

**Rule.** Every deposit into `addExprToMemoryBlock` first walks `memoryBlock.nameMap.ancestorsOf[valId]` (which includes `valId` itself plus every strict prefix scope registered via `encodePush`). If the expression is already in `memoryBlock.intKnownStatements` at ANY of those validities, the function returns immediately without depositing.

Two parallel scans, both at function entry:

- **Site F** ([`prover.cpp–4804`](../GL_Quick_VS/GL_Quick/src/prover.cpp)) — duplicate suppression. Returns early if any ancestor scope already has the expression.
- **Site H** ([`prover.cpp+`](../GL_Quick_VS/GL_Quick/src/prover.cpp)) — blacklist scan over `memoryBlock.nameMap.ancestorsOf[valId]` against `intValidityNamesToFilter`. Filters out depositary scopes that have been blacklisted.

**Why.** Comparable-scope semantics: a fact at scope `S` is observably true at every descendant of `S`. Re-installing the same fact at a strict descendant adds no information the prover couldn't already see via inheritance. Site F's early return prevents the redundant deposit and short-circuits any downstream work (disintegration, mail emission, ordisMerge convergence registration) for the duplicate.

**Spot.**

- New code that pushes into `intKnownStatements` outside `addStatement`'s `local`-branch.
- A new entry path into the kernel that bypasses Site F (e.g. directly calling `addExprToMemoryBlockKernel` without the outer dedupe). This is generally wrong unless the caller has independently established that the deposit is non-redundant.
- A flat-namespace `validityName` (no `_boundary_`) where `ancestorsOf[valId]` degenerates to `{valId}`. Safe — the scan still runs, just with a single entry — but worth noting for sanity.

**Fix.** Trust Site F. If the deposit "should" land but doesn't, the diagnostic is to check `intKnownStatements` for any ancestor entry — almost always the redundancy is real.

**Compressor mode bypass.** Site F is gated on `!parameters.compressor_mode`. The compressor stage allows redundant deposits at strict-descendant scopes for reasons specific to the compression algorithm (currently undocumented in detail; an open question for the compressor chapter).

**Interactions.**

- **D-34's `ordisMerge`** pushes converged expressions onto `internalMailIn` for the next-hashburst absorb. The absorb's `addExprToMemoryBlock` call hits Site F. If the converged expression is already at a strict ancestor (e.g. via an unrelated derivation chain), the deposit short-circuits silently and no `or convergence` entry is emitted at the convergence target scope. This is correct behaviour — see the impl24-down case in [D-34](40_decisions.md#d-34)'s verification trace, where the same `(or2[2,repl_lev_1_0,6])` orSig at two stack positions races and the impl24-internal promotion gets dedupe'd by Site F because the top-level OR-convergence promotion already landed the fact at `v=main`.
- **The `_orint_` block at [`prover.cpp+`](../GL_Quick_VS/GL_Quick/src/prover.cpp)** emits the wrapping OR at the parent scope via a fresh `addExprToMemoryBlock` call. Site F dedupe applies — if the OR is already known at any ancestor of the parent scope, the emit is silent.

**Code.** [`prover.cpp–4812`](../GL_Quick_VS/GL_Quick/src/prover.cpp). The `nameMap.ancestorsOf` table is built at every `encodePush` site — see [`memory.hpp::NameMap::encodePush`](../GL_Quick_VS/GL_Quick/src/memory.hpp).

---

<a id="i-29"></a>
## I-29 Variable-port type consistency: every variable in a chapter row connects ports with identical type labels

**Scope.** Producer side: [`compiler.hpp`](../GL_Quick_VS/GL_Quick/src/compiler.hpp) (`ArgumentAnalyzer`) called from [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (`compileCoreExpressionMap`). Consumer side: [`verifier.py:check_defset_consistency`](../verifier.py) (D-41).

**Rule.** Within any single MPL expression in a proof-graph chapter row, every variable that appears at multiple operator-call positions must connect to ports whose `definition_sets` type label is identical. Bound-variable rebinding at `>[v1,v2]` quantifiers introduces alpha-distinct names — the same string `v1` inside an inner quantifier scope is not the same variable as one outside. Variables whose name is reused across alpha-distinct scopes are independent.

**Why.** Consistency of types across an expression's variable connections is the structural prerequisite for correct semantics. A variable used at one position with type `(1)` (an element of N) and at another with type `P(1)` (a subset of N) cannot meaningfully refer to the same object. The compiler enforces this at producer time via the per-batch `ArgumentAnalyzer`'s `mergeMaps` mismatch raise; the verifier confirms the contract holds on the artefacts that get shipped (chapter rows in `processed_proof_graph/*.txt`). Together these form the formal-completeness pillar.

**Spot.**

- A new producer-side code path that emits a chapter row with mismatched argument types would surface as a `definition set consistency` failure in the verifier tally.
- A new compact operator added to a per-batch GL binary whose elements lead to inter-batch arity collisions will manifest if `gl_binaries[<batch>]` and `gl_binaries['shared']` both define the operator with different signatures.
- A change to ConfigVisu.json's atomic operator definition_sets that drifts away from per-batch configs will surface as widespread chapter-row failures (the  script catches this case).

**Fix.**

- Producer-side type mismatch in a chapter row: trace upstream from the verifier's per-row failure report. The compiler's analyzer would have asserted at compile time if the rule itself were ill-typed; surface from a verifier failure on a chapter row indicates either a typing edge case the compiler missed or a rule application at runtime that ignores the operator's defsets. Investigate the row's tag (which producer site emitted it).
- ConfigVisu drift: sync to the per-batch authoritative version; per-batch configs win on disagreement (they're the live source for the prover).
- Inter-batch compact-name collision: per-tag resolution in `build_resolved_defsets_per_tag` selects the right batch's allocation per chapter; if a new batch is added, ensure its binary is loaded.

**Code.** Producer-side: [`compileCoreExpressionMap`](../GL_Quick_VS/GL_Quick/src/prover.hpp). Consumer-side: [`verifier.py:check_defset_consistency`](../verifier.py) + helpers (`build_resolved_defsets_per_tag`, `_parse_subtree`, `_merge_maps`, `_process_leaf`).

---

<a id="i-28"></a>
## I-28 Cross-LB writes during `proveKernel`'s parallel phase are forbidden — defer to post-`pool.join` collectors

**Scope.** [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) (`proveKernel`'s worker pool + post-join drains), every code path reachable from `performElementaryLogicalStep` on a worker thread.

**Rule.** Code that runs on a worker thread inside `proveKernel`'s parallel phase (between `pool.emplace_back(worker, t)` at line 6263 and `pool.join` at line 6264) **must not write to any other LB's mutable state** — `exprOriginMap`, `encodedStatements`, `mailIn`, `internalMailIn`, `simpleMap`, `toBeProved`, etc. Worker threads may freely write their **own** LB's `mailOut.*` (which is single-LB-scoped during parallel phase), and may stage entries into class-level collectors on `ExpressionAnalyzer` under a mutex (`inductionMemoryBlocksMutex`, `pendingAncestorOriginsMutex`, etc.). The class-level collectors are drained single-threaded after `pool.join`, in sorted order; that drain is the only place where ancestor-side mutable state is touched.

**Why.** `logicalCores = std::max(1u, std::thread::hardware_concurrency)` ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp); the `=1` workaround at line 245 is commented out). The pool spawns `workers` real threads, each pulling LBs from a shared `bodies` vector via `std::atomic<size_t> next` and calling `performElementaryLogicalStep` independently. Two threads can be on different LBs at the same time; one reaching across into another's `std::map` is undefined behavior — the bimodal pattern fixed by [D-39](40_decisions.md#d-39) was exactly this.

**Spot.**
- Any descendant code that does `pred = memoryBlock.parentMemory; while (pred) { someWrite(pred->...); pred = pred->parentMemory; }` outside the post-join block.
- Any code that takes a raw `Memory*` pointer to a LB other than the one currently on this thread's stack and writes to its containers.
- A new collector that buffers cross-LB effects but doesn't go under a mutex, or doesn't sort before drain — both break the invariant.

**Fix.** Stage the effect into a class-level vector under the appropriate mutex (or add one); drain it in `proveKernel` after the existing `inductionMemoryBlocks` drain (`prover.cpp`), single-threaded, sorted by deterministic key. The drain may freely touch any LB's state. See `pendingAncestorOrigins` ([`prover.hpp`-area](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`prover.cpp`-area](../GL_Quick_VS/GL_Quick/src/prover.cpp)) as the canonical pattern.

**Code.** Existing collectors: `inductionMemoryBlocks` + `inductionMemoryBlocksMutex`, `pendingAncestorOrigins` + `pendingAncestorOriginsMutex`, `updateGlobalTuples` + `updateGlobalMutex`, `updateGlobalDirectTuples` + `updateGlobalDirectMutex` ([`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp)). Drain block: [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp).

---

<a id="i-25"></a>
## I-25 `addStatement` returns `ExpressionWithValidity` pairs; cross-scope deposits ride through `newStatements`

**Scope.** Prover — [`addStatement`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`applyEquivalenceClass`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`applyEquivalenceClassToNegatedEquality`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`updateEquivalenceClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`cleanUpExpressions`](../GL_Quick_VS/GL_Quick/src/prover.hpp), and the kernel post-loop in [`addExprToMemoryBlockKernel`](../GL_Quick_VS/GL_Quick/src/prover.cpp).

**Rule.** `addStatement` and every helper that pushes new facts into the caller's emit channel uses the type `std::vector<ExpressionWithValidity>` for that channel. Each entry carries the deposit's actual scope. The kernel's post-`addStatement` loop uses each entry's own `validityName` for `statementLevelsMap` lookup, admission-map updates, `toBeProved` discharge, and validity-name promotion — never the kernel-call's own `validityName` parameter. There must be no separate "side sink" vector for cross-scope deposits.

**Why.** Cross-scope eq-class deposits ([D-33](40_decisions.md#d-33)) land at `deeperOf(class.scope, expr.scope)`, which can differ from the kernel-call's `validityName`. A WIP design that routed them to side sinks (`crossScopeSink`, `descendantSink`, `ancestorSink`) avoided tripping the kernel's same-scope assert at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) but bypassed the kernel's discharge logic for those facts. Matching `toBeProved` entries never closed; FTA-rung-1 §9b's `(implication23[2,8,int_lev_4_2349])` validity-name promotion never fired even with the boundary fact present in `encodedStatements`; Gauss summation regressed from proved to unproved.

The pair-based channel carries the deposit's scope into the kernel so the lookup uses the deposit's own validity, the assert holds, and the discharge logic runs uniformly for same-scope and cross-scope deposits.

**Spot.**

- A new code path that pushes into the eq-class emit vector with a bare `std::string` (compile error after this invariant — the type is `ExpressionWithValidity`).
- A new local `crossScopeSink`, `descendantSink`, or `ancestorSink` variable inside `addStatement` or `updateEquivalenceClasses`. Routes that bypass `newStatements` re-open the discharge gap.
- The kernel post-loop building `EncodedExpression(addExpression, validityName)` with the function-parameter `validityName`. After this invariant the loop uses `effectiveValidity` (the per-entry scope from the pair).

**Fix.** Push the new emission into `newStatements` as `ExpressionWithValidity(applied, depositValidity)`. If the receiver only needs the string, project it inside the receiver — do not re-introduce a parallel string-only channel.

**Code.** Helper signatures: [`prover.hpp::applyEquivalenceClass`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`prover.hpp::applyEquivalenceClassToNegatedEquality`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`prover.hpp::cleanUpExpressions`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`prover.hpp::updateEquivalenceClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`prover.hpp::addStatement`](../GL_Quick_VS/GL_Quick/src/prover.hpp). Kernel loop: [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp). The `ExpressionWithValidity` type and its `operator<` are in [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp).

---

<a id="i-30"></a>
## I-30 `applyEquivalenceClassToRejectedMapIntegration` is additive — original rmi entries are never erased

**Scope.** [`prover.hpp::applyEquivalenceClassToRejectedMapIntegration`](../GL_Quick_VS/GL_Quick/src/prover.hpp). The integration-side equivalence-class hook on `HashMemory::rejectedMapIntegration`.

**Rule.** When a class C rewrites an rmi entry K1 to a new key K2:
- The original K1 entry stays in `rmi` (both its key and its value set untouched).
- The match path emits a revival via `internalMailIn` for K2's substituted value.
- The no-match path inserts the rewritten value at K2 alongside K1.
The function never calls `rmi.erase` on its own. There is no `toErase` queue.

**Why.** K1 is registered in `rmi` because it failed admission at registration time. It is a real candidate for revival via [`revisitRejectedIntegration2`](../GL_Quick_VS/GL_Quick/src/prover.hpp), which iterates `rmi` keys against the current admission landscape on every `makeAdmissionKeys` write. K1 and K2 probe distinct admission slots (u_-form against `admissionMapIntegration`, bare form against `admissionSetIntegration`). The class's a→b substitution does not propagate to admission keys — admission is keyed structurally — so K2 failing admission today does not imply K1 will fail admission tomorrow under a different admission landscape. Erasing K1 silently lost those revivals (pre-[D-43](40_decisions.md#d-43)). This invariant mirrors [D-33](40_decisions.md#d-33)'s additive-rewrite principle, which the expression-side path already honors: original facts are never overwritten.

**Spot.**

- Any `rmi.erase` call inside `applyEquivalenceClassToRejectedMapIntegration` or any new helper it calls.
- A re-introduction of `toErase.push_back(keyEv)` inside the per-entry rewrite loop.
- Verifier failures or theorem regressions that trace to a missing revival from a key that was rewritten by a class earlier in the run.

**Fix.** Restore the additive contract: the rewrite loop must only `toInsert.emplace_back(...)` for the no-match path; the match path emits to `internalMailIn` and returns without modifying `rmi`. The post-loop apply-mutations block contains only the `toInsert` consumption.

**Code.** [`applyEquivalenceClassToRejectedMapIntegration`](../GL_Quick_VS/GL_Quick/src/prover.hpp). See also [D-43](40_decisions.md#d-43), [D-33](40_decisions.md#d-33), [I-22](#i-22).

---

<a id="i-31"></a>
## I-31 `updateEquivalenceClasses` ancestor-pass never modifies ancestor-scope class state

**Scope.** [`prover.hpp::updateEquivalenceClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp). The cross-scope merge pass added by [D-44](40_decisions.md#d-44).

**Rule.** When merging absorbs an ancestor class `C_a @ V_a` into `mergedClass @ validityName`:
- `mb.equivalenceClassesMap[V_a]` is **never** written.
- `mb.eqClassSttmntIndexMapMap[V_a]` is **never** written (no `erase`, no overwrite).
- The ancestor class is read via a `const&` argument to `mergeTwoEquivalenceClasses`; the merge logic mutates only `mergedClass` (the destination at `validityName`).

**Why.** A class at `V_a` is only valid under the equalities that have been admitted at `V_a` and its ancestors. The new equality bridging into `mergedClass @ V` was admitted at the descendant scope `V` only — `V_a` cannot see it under the same-or-deeper visibility rule. Writing `V_a`'s class state from this descendant-scope merge would conjure equivalences that `V_a` should not yet know, breaking the invariant that each scope's class state is consistent with the equalities visible AT that scope. This mirrors the additive principle established by [D-33](40_decisions.md#d-33) (apply-side cross-scope deposits land at the deeper scope; ancestor original is never overwritten) and [I-30](#i-30) / [D-43](40_decisions.md#d-43) (rmi rewrites are additive at the new key; original ancestor-scope rmi entries kept).

**Spot.**

- Any write to `mb.equivalenceClassesMap[ancestorV]` inside `updateEquivalenceClasses`'s ancestor-pass block.
- Any `mb.eqClassSttmntIndexMapMap[ancestorV].erase(...)` or `[ancestorV][...] =...` inside the ancestor-pass block.
- A `mergeTwoEquivalenceClasses` overload or call that takes `classB` by mutable reference instead of `const&`.
- Theorem regression where a fact provable at an ancestor scope vanishes after a descendant scope ingests an equality — the ancestor class was destructively mutated.

**Fix.** Restore the read-only contract: the ancestor pass must only mutate `mergedClass` (the destination class at `validityName`). `mergeTwoEquivalenceClasses` already enforces this at the type level (`const EquivalenceClass& classB`); the call-site discipline is to pass the ancestor class as the `classB` argument and never the `classA` argument.

**Code.** [`updateEquivalenceClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp), the ancestor-pass block guarded by `mb.nameMap.stringAncestorsOf[validityName]`. See also [D-44](40_decisions.md#d-44), [D-33](40_decisions.md#d-33), [I-30](#i-30).

---

<a id="i-33"></a>
## I-33 `mergeTwoEquivalenceClasses` cross-vN preconditions: ancestor-only direction; eqArgs-subset assert is same-vN only

**Scope.** [`prover.hpp::mergeTwoEquivalenceClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp). The function takes `validityName` (mergedClass / `classA` scope) and `classBValidityName` as its two scope parameters.

**Rule.** Three sub-rules:

1. **Ancestor-only direction.** When `classBValidityName!= validityName`, `classBValidityName` must be a strict ancestor of `validityName` per `memoryBlock.nameMap.stringAncestorsOf[validityName]`. Descendant-direction merge is forbidden by symmetry with [D-33](40_decisions.md#d-33)'s class-deeper exclusion (a descendant class is invisible at the ancestor's scope and carries no information for an equality admitted there).

2. **Cross-vN parent-subset early-exit.** When cross-vN AND `classB.variables ⊆ classA.variables`, return immediately. The descendant `mergedClass` already covers every variable the ancestor class would contribute; no merge action is required, no cross-pair emissions, no origin folding. The ancestor class stays at its scope per [I-31](#i-31).

3. **eqArgs-subset assert is same-vN only.** The `assert(!isSubsetOf(eqArgs, classB.variables))` precondition holds for same-vN merges only. Cross-vN allows `eqArgs ⊆ classB.variables` because the equality `(=[eqArgs[0], eqArgs[1]])` may already be established at the ancestor's scope (via mail, prior derivation, or independent admission at `V_a`). Symmetrically, the `tmp.size == 1` assert (single-bridge invariant) holds for same-vN only — cross-vN allows `tmp.size == 2`, in which case `commonArg = *tmp.begin` picks the lexicographically smallest as a deterministic bridge. Cross-pair records via that single bridge cover every (varA, varB) pair; alternatives via the second bridge would be parallel origin records that [I-32](#i-32) suppresses as already-known.

**Why.** [D-44](40_decisions.md#d-44) extended `updateEquivalenceClasses` with an ancestor-scope merge pass. The original `mergeTwoEquivalenceClasses` was written under same-vN assumptions: if any same-vN class contained both `eqArgs`, that class would be iterated first by `updateEquivalenceClasses`'s sequential overlap loop and absorbed via the subset path before `mergedClass` could grow past `eqArgs`. The eqArgs-subset and single-bridge asserts encoded that invariant. Cross-vN ancestor classes break the assumption — they may legitimately contain both `eqArgs` independently of the descendant's iteration order. The Gauss main batch on the branch hit the asserted condition during Hash burst 2, aborting via `0xC0000409`. The cross-vN early-exit (sub-rule 2) further short-circuits the case where the ancestor adds nothing new, avoiding spurious cross-pair emissions for variables already in the descendant class.

**Spot.**

- Assertion failure `!isSubsetOf(eqArgs, classB.variables)` at the function in production runs of the prover that mix descendant scopes with ancestor classes containing both args.
- Assertion failure `tmp.size == 1` immediately following — the single-bridge invariant is broken when both eqArgs are in classB.
- A cross-vN call site that passes the same `validityName` argument twice (collapsing the cross-vN distinction into apparent same-vN, hiding the bug).
- A descendant class accidentally passed as `classB` (descendant-direction merge) — caught by the ancestorship assertion.

**Fix.** Both call sites of `mergeTwoEquivalenceClasses` in `updateEquivalenceClasses` must pass `classBValidityName` correctly:
- Same-vN call (line ~5950): `validityName, validityName`.
- Ancestor-vN call (line ~5987): `validityName, ancestorV`.

The function then validates the relationship and gates the assertions and the early-exit accordingly. Do not weaken the assertions — they are correct for same-vN. Do not skip the ancestorship validation — it is the only gate against descendant-direction misuse.

**Code.** [`mergeTwoEquivalenceClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`updateEquivalenceClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp). See also [D-44](40_decisions.md#d-44), [I-31](#i-31), [I-32](#i-32).

---

<a id="i-32"></a>
## I-32 Cross-pair `equality2` emission gated on existing class/LB origin

**Scope.** [`prover.hpp::mergeTwoEquivalenceClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp), the merged-pair history block (the `if (parameters.trackHistory)` branch with `varA!= commonArg && varB!= commonArg`).

**Rule.** Before pushing an `equality2` cross-pair origin record for target `(=[varA, varB]) @ validityName` (or its mirror `(=[varB, varA])`), check whether any of the following already carry a non-empty origin entry for that target:

- `mergedOriginMap` — origins accumulated in the in-flight merge so far (covers prior cross-pair pushes within the same `updateEquivalenceClasses` call where multiple existing classes overlap `eqArgs` and get folded sequentially).
- `classB.equalityOriginMap` — origins on the class being absorbed (these are about to fold into `mergedClass` via the post-loop overwrite at `prover.hpp::mergeTwoEquivalenceClasses`'s `tmpMap` blocks; treat them as already-known).
- `memoryBlock.exprOriginMap` — origins at LB level (covers mail-arrived origins synced into the class via the bulk-merge + mail-sync at [`prover.cpp::performElementaryLogicalStep`](../GL_Quick_VS/GL_Quick/src/prover.cpp); also covers any other prior derivation that already wrote to the LB's exprOriginMap).

If any of the three holds, **skip** all three `addOrigin` calls (`mergedOriginMap`, `memoryBlock.exprOriginMap`, `memoryBlock.mailOut.exprOriginMap`) for that target.

**Why.** When a clique of equalities arrives via mail (e.g. the contradiction-cascade clique `{(=[v1,v2]), (=[v1,i1]), (=[v2,i1])}` mailed into a child LB), the bulk-merge places mail origins in `body.exprOriginMap` and the mail-sync places them in the class's `equalityOriginMap`. Subsequent `mergeTwoEquivalenceClasses` calls iterate possible bridge variables (`commonArg`) and unconditionally generate `equality2` cross-pair records for every (varA, varB) combination. Different bridge choices for the same target produce DIFFERENT cross-pair source pairs that are individually valid but mutually dependent: e.g. `(=[v1,v2]) ← equality2 | (=[v1,i1]) (=[i1,v2])` (bridge `i1`) and `(=[v1,i1]) ← equality2 | (=[v1,v2]) (=[v2,i1])` (bridge `v2`). The verifier's origin-chain DFS at [`verifier.py`](../verifier.py) detects the cycle. The historical case is the chapter-22 / theorem-12 (induction zero-case) failure documented in [G-31](50_gotchas.md#g-31) and [D-46](40_decisions.md#d-46).

The gate is sound because `equality2` cross-pair records are *transitive convenience records* — they document that the equality is derivable through the merge bridge. When the equality is already established by a separate path (mail, prior merge, anchor handling, etc.), the convenience record contributes no new deductive content but introduces a parallel origin that can form cycles with other parallel origins.

**Spot.**

- New chapter rows of the form `(=[a,b]) equality2 (=[a,c]) (=[c,b])` where `(=[a,b])` already has another origin row in the same chapter.
- Verifier failures under the `origin chain termination` tag — the DFS reports cyclic nodes within an `equality2` clique.
- Direct inspection of `memoryBlock.exprOriginMap` for an equality showing both a non-`equality2` origin (e.g. `recursion`, `merging origin`, mail-derived) AND an `equality2` origin whose sources transitively depend on the equality itself.

**Fix.** Restore the gate. Do **not** weaken the check — all three sources (`mergedOriginMap`, `classB.equalityOriginMap`, `memoryBlock.exprOriginMap`) must be consulted. Removing any one re-opens the cycle path it covers (e.g. dropping `mergedOriginMap` re-opens cycles between two same-call merges; dropping `memoryBlock.exprOriginMap` re-opens cycles against mail-arrived origins not yet visible to the class's local map).

**Code.** [`mergeTwoEquivalenceClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp). See also [D-46](40_decisions.md#d-46), [G-31](50_gotchas.md#g-31), [I-12](#i-12), [I-31](#i-31).

---

<a id="i-34"></a>
## I-34 Cross-substitution `equality1` emission gated on existing target origin

**Scope.** [`prover.hpp::applyEquivalenceClass`](../GL_Quick_VS/GL_Quick/src/prover.hpp), the `if (parameters.trackHistory)` block that emits the `equality1` origin record for a class-rewritten expression `applied @ depositValidity`.

**Rule.** Before pushing an `equality1` origin record for target `applied @ depositValidity`, check whether `memoryBlock.exprOriginMap[appliedWithValidity]` already has at least one entry. If it does, **skip** both `addOrigin` calls (`memoryBlock.exprOriginMap` and `memoryBlock.mailOut.exprOriginMap`) for that target.

The `exprOriginMapLocal` populate at the rewrite-enumeration site stays unconditional — this is required so the FIRST emission for a target carries its `setEqualities` justifiers. Skipping the populate would produce a malformed `len(rest) < 4` origin record that the verifier's `check_equality1` rejects (see the comment block at the populate site in `applyEquivalenceClass`). The gate fires at the *emission* step only.

**Why.** `applyEquivalenceClass` runs once per equivalence class per relevant statement. When two members of the same class both appear in admissible expressions, the rewrite loop fires both directions: `(in2[i0,i0,id]) [i0→v10] → (in2[i0,v10,id])` and the reverse `(in2[i0,v10,id]) [v10→i0] → (in2[i0,i0,id])`. Pre-gate, both directions emit `equality1` origin records that point at each other — a 2-cycle. The verifier's `origin chain termination` DFS at [`verifier.py`](../verifier.py) walks the chain and reports the cycle.

The gate is a **producer-side redundancy guard**, not the system-level cycle-resolution mechanism. It suppresses redundant equality1 emissions inside `applyEquivalenceClass` when the target already has any origin in `body.exprOriginMap`. Two important caveats:

1. The gate alone **does not close the swap-cycle case** (`(in2[i0,v10,id]) ↔ (in2[v10,i0,id])` mutually-substituted via the equivalence class with both targets being first-emissions). Both syntactically distinct targets pass the gate independently because each target's origin map is empty when its emission fires. The system-level resolution is **[I-35](#i-35)** / [D-49](40_decisions.md#d-49) — `addOrigin`'s cap-full preference replacement at intake.
2. The gate's scope is the **producer LB only** (the LB that runs `applyEquivalenceClass`). Cycles formed by mail import (multiple LBs each emitting one half of a cycle, then aggregated via `smashMail`) are unaffected by this gate. They are resolved at the receiver via the bulk-merge routing through `addOrigin` (see [I-35](#i-35)).

Both observations were established empirically by the chapter-96/97 (theorem 96 Gauss `fold` induction zero-case + step) failures: the original cycle was `(in2[i0,v10,id]) ↔ (in2[i0,i0,id])`; the swap-cycle was `(in2[i0,v10,id]) ↔ (in2[v10,i0,id])`. Both shapes were ultimately resolved by I-35/D-49, not by this gate. See [G-39](50_gotchas.md#g-39) and [D-48](40_decisions.md#d-48) for the gate; [G-40](50_gotchas.md#g-40) and [D-49](40_decisions.md#d-49) for the resolution.

**Spot.**

- New chapter rows of the form `(P[…X…]) equality1 (P[…Y…]) (=[X,Y])` where the same chapter also has a row `(P[…Y…]) equality1 (P[…X…]) (=[Y,X])` — both rows reference each other and neither has a non-`equality1` origin.
- Verifier failures under the `origin chain termination` tag whose cyclic node set consists of two expressions related by an equivalence-class substitution.
- Direct inspection of `memoryBlock.exprOriginMap` showing two expressions whose only origins are `equality1` records pointing at each other.

**Fix.** Keep the gate — `auto alreadyHasOrigin = [&](const ExpressionWithValidity& ev) -> bool { auto it = memoryBlock.exprOriginMap.find(ev); return it!= memoryBlock.exprOriginMap.end && !it->second.empty; };` — applied before the `addOrigin` calls inside the `trackHistory` block. The gate is producer-side noise reduction; the actual cycle-resolution mechanism is [I-35](#i-35). Sites 2 (`applyEquivalenceClassToNegatedEquality`) and 3 (`emitIntegrationRevivalToInternalMailIn`) can also emit `equality1`; they are not gated because their internal early-exits (`statementLevelsMap` dedup at site 2; mail absorbance routing through I-35 at site 3) cover the cycle shape via the receiver's preference replacement.

**Code.** [`applyEquivalenceClass`](../GL_Quick_VS/GL_Quick/src/prover.hpp). See also [D-48](40_decisions.md#d-48), [I-35](#i-35), [D-49](40_decisions.md#d-49), [G-40](50_gotchas.md#g-40), [I-32](#i-32).

---

<a id="i-35"></a>
## I-35 `addOrigin` cap-full preference: anything beats `equality1`/`equality2` — superseded by [D-51](40_decisions.md#d-51), 2026-05-08

> **Status amendment (2026-05-08).** Superseded by [D-51](40_decisions.md#d-51) for the chapter-emission cycle case. The cap-full preference logic remains in `addOrigin` at HEAD but is rarely active under the post-D-51 configuration: `max_origin_per_expr = 30` (non-compressor configs match compressor mode) means the per-key origin vector is rarely at cap, so the replacement branch runs only on a handful of high-traffic keys. The structural cycle-prevention work has moved into `buildStack` itself (chapter goal in `g_buildStackPath` so the existing path-cycle filter rejects self-applying origins; chapter-boundary `__contradiction__` simpleMap fallback as the only LB switch). The text below describes the original rule for traceability of the chapter-100/101 fix; the rule is no longer load-bearing for any current chapter shape. Cleanup of the now-mostly-inert preference branch is deferred.

**Scope.** [`prover.hpp::addOrigin`](../GL_Quick_VS/GL_Quick/src/prover.hpp), the universal origin-emission helper used by every producer site (`applyEquivalenceClass`, `mergeTwoEquivalenceClasses`, `addStatement` mirror block, `addExprToMemoryBlock` history, `performElementaryLogicalStep` mailIn bulk-merge, etc.).

**Rule.** When the per-key origin vector is at cap (`maxOrigins`) and a new origin arrives:

- If the new origin's tag is **not** `equality1` or `equality2`, scan the vector for the first slot whose tag **is** `equality1` or `equality2` and replace it in place with the new origin. Return.
- Otherwise (new origin is equality-convenience-tagged, or no slot is replaceable), keep existing — drop the new origin per the legacy "insertion order" tiebreak.

Below cap, the legacy append behavior is unchanged (push at end, dedup on exact-match, D-44 trap intact).

**Why.** `equality1` and `equality2` are *transitive convenience records*: they document derivability via equivalence-class substitution (`equality1` for argument substitution, `equality2` for cross-pair transitivity). Both are inherently susceptible to swap/bridge cycles when the class has multiple members — see [I-32](#i-32) for the equality2 cross-pair gate, [I-34](#i-34) for the equality1 substitution gate. Any other origin tag (`implication`, `recursion`, `theorem`, `expansion`, `disintegration`, `task formulation`, `premise element`, `reformulation for integration...`, `mirrored from`, `vacuous truth`, `symmetry of equality/inequality`,...) refers to a direct deductive step that does not have this cyclic structure.

When two origins exist for the same key (legitimate situation — e.g. mail-bulk-merge brings a foundational `implication` origin AND a cyclic `equality1` origin both shipped in the same broadcast for the same target) and `max_origin_per_expr = 1` (non-compressor) forces the choice down to one, **the choice must not be left to insertion order** — that silently picks whichever origin happened to be pushed first, which has been the cyclic `equality1` in the chapter-100/101 swap-cycle case (theorem-96 Gauss `fold` induction). Foundation must displace convenience. The verifier's `origin chain termination` walks the surviving origin; a foundational origin terminates, a cyclic one loops.

**Spot.**

- `body.exprOriginMap[key]` showing only an `equality1`/`equality2` origin for an expression that *also* has a foundational origin earlier in the same hashburst's `mailIn.exprOriginMap` snapshot. Compare burst N's mailIn dump (multi-origin) against burst N+1's `body.exprOriginMap` dump (single-origin); a downgrade to equality-only is the symptom.
- Verifier failures under the `origin chain termination` tag where the cyclic node set is exactly two `equality1`-tagged or `equality2`-tagged rows and the prover-side trace shows a competing non-equality origin in `mailIn.exprOriginMap`.

**Fix.** Restore the cap-full preference replacement in `addOrigin`. Do **not** weaken to "first non-equality wins" (existing equality slot must be replaced by an arriving non-equality, regardless of arrival order). Do **not** extend preference to all tags — the rule is binary (equality-convenience vs. anything else) because that is the categorical distinction; a finer-grained per-tag priority list would couple the helper to producer-side semantics and is unnecessary for cycle suppression.

**Code.** [`addOrigin`](../GL_Quick_VS/GL_Quick/src/prover.hpp). See also [D-49](40_decisions.md#d-49), [I-32](#i-32), [I-34](#i-34).

---

<a id="i-39"></a>
## I-39 Chapter rows are emitted at the closest-to-`main` ancestor with an origin


**Scope.** [`visualizer.cpp::buildStack`](../GL_Quick_VS/GL_Quick/src/visualizer.cpp) — the chapter-row emission walker. All consumers of `files/raw_proof_graph/*.txt`: [`verifier.py`](../verifier.py), [`process_proof_graphs.py`](../process_proof_graphs.py), [`generate_full_proof_graph.py`](../generate_full_proof_graph.py).

**Rule.** Every chapter row's `row[1]` (new-expression validity) and every dep cell's validity at `row[3 + 2k + 1]` is the **closest-to-`"main"` ancestor of the requested validity for which `(expression, ancestor)` has an origin in the emitting LB's `exprOriginMap`**, subject to the **OR-branch barrier**: the walk may not cross `_boundary_orint_` or `_boundary_ordis_` delimiters. The deepest `orint_`/`ordis_` ancestor sets the shallowest allowed lift target. Ancestor chains are recovered by splitting the validity string on `"_boundary_"` per [I-2](#i-2); the helper `liftToShallowestOriginAncestor` in [`visualizer.cpp`](../GL_Quick_VS/GL_Quick/src/visualizer.cpp) is the single authority. Producer-side: applied at `buildStack` entry, at the dep-emission lambda, at every recursive `buildStack` call, at the candidate-loop cycle filter, and at the last-resort `front`-emit fallback. Consumer-side: a chapter row's `row[1]` and each dep cell's validity already agree at the same lifted scope, so verifier checkers can compare dep validities with exact `==` against the corresponding stored validity.

**Why the OR barrier.** OR-branch scopes (`_boundary_orint_<sig>_(<disjunct>)`, `_boundary_ordis_<sig>_(<disjunct>)`) are conditional on a disjunct hypothesis. The OR-family verifier checkers (`check_or_convergence`, `check_or_branch_proven`, `check_or_branch_assumption`, `check_or_disintegration`) require branch-distinct namespaces on the dep cells to recognise the convergence / branch-proof pattern. Without the barrier, both branches' deps would lift to the parent boundary (where the post-convergence origin lives), collapsing the convergence row to identical dep pairs and tripping `origin chain termination` cycles in the verifier.

**Why.** Pre-lifting, `buildStack` did an exact `(expr, validity)` lookup with a "fall through to `(expr, "main")`" shadow when the exact-key missed, and `emitRow` wrote the *requested* `proved.validityName` into `row[1]` regardless of which scope the origin was actually fetched from. Result: chapter rows could claim a derivation happened at a deep boundary scope while the origin's deps were at `main` — a falsified row. Concrete instance: `files/raw_proof_graph/193_check_induction_condition.txt` line 102 emitted `(=[it_0_lev_0_32,2])` at `main_boundary_(implication23[2,8,int_lev_4_2365])` with deps cited at `main`, despite the only origin record being at `main`. Lifting eliminates the falsification by emitting at the scope where the origin actually lives. The chapter shape becomes truthful, deduplicates by construction (one row per `(expr, lifted_v)`), and makes HTML namespace-tag jumps deterministic (every cited validity has a card because that is exactly where buildStack emitted).

**Spot.**

- A chapter row whose `row[1]` is a non-`main` scope but every dep cell is on `main`. Pre-lifting, this was the falsification signal; post-lifting it should not appear unless the origin really is recorded at that scope.
- Verifier failures under `expansion`, `disintegration`, `symmetry of equality`, `symmetry of inequality`, or other dep-validity-matching checkers where the stored validity exists at an ancestor of the row's cited dep validity. Pre-lifting, the fix was to widen verifier matching; post-lifting, the fix is to confirm `buildStack` is invoking `liftToShallowestOriginAncestor` at every dep emission and recursion site.
- Inspect  if it appears — the "no origin found" assertion's dump now reflects a *lifted* `proved`, so a missing origin really means no ancestor (including `main`) has an entry.

**Fix.** Restore the lifting at any site in `buildStack` where it was removed. The helper `liftToShallowestOriginAncestor(memoryBlock, ExpressionWithValidity)` is the single point of authority. Do not bypass it for "performance"; the cost is one hash-map lookup per ancestor level, capped by validity-stack depth.

**Code.** [`visualizer.cpp::liftToShallowestOriginAncestor`](../GL_Quick_VS/GL_Quick/src/visualizer.cpp), [`visualizer.cpp::buildStack`](../GL_Quick_VS/GL_Quick/src/visualizer.cpp). See also [I-2](#i-2), [I-38](#i-38), [D-56](40_decisions.md#d-56), [G-41](50_gotchas.md#g-41).

**Relation to [I-38](#i-38).** Complementary fixes for the same falsified-row class. This invariant is producer-side (lifting prevents emission of the falsified row in the first place). The deeperOf-equality invariant is verifier-side (rejects any falsified row that slips through). With both in place, the chapter-193 line-102 example is fixed at the producer (lift moves the row to `main`); the verifier check stays as forcing-function for any future producer-side regression.

---

<a id="i-38"></a>
## I-38 `implication`-row deposit lives at `deeperOf` of constituents

**Scope.** Verifier — `check_implication` at [`verifier.py`](../verifier.py). Mirrors C++ prover behaviour: [`generateEncodedRequestsStatic` + `growBaseCandidates`](../GL_Quick_VS/GL_Quick/src/memory.cpp) accumulate validity ids via `nm.deeperOf(...)`, so the result of combining facts lives at the deepest scope of the inputs.

**Rule.** For every chapter row tagged `implication`:

1. **Pair-wise comparability.** Every pair of namespaces among the constituents (the rule itself + every cited premise) must be **comparable** in the validity-stack sense (`nm.comparable(a, b)` per [`04_validity_stack.md` § `comparable / deeperOf`](20_core_concepts/04_validity_stack.md)). One must be an ancestor of the other (or they are equal). Sibling scopes are rejected.

2. **Deeperof-equality on result.** The row's namespace (`line.namespace`) must EQUAL the deepest constituent namespace. Equivalently, `line.namespace ∈ {impl_ns} ∪ {premise_nss}`. If every constituent sits at `"main"` while `line.namespace` is `"main_boundary_<X>"`, the row is rejected: an implication firing deposits its conclusion at `deeperOf(constituents)`, never at a strictly deeper scope no constituent reaches.

**Relation to [D-35](40_decisions.md#d-35).** D-35 (comparable-scope premise inheritance) is the weaker baseline: every constituent ns must be at-or-above `result_ns`. The deeperOf-equality rule is the strengthening: at least one constituent must REACH `result_ns`. D-35 alone admitted rows where every constituent was an ancestor of `result_ns` but none reached it; those rows imply a derivation step the prover cannot have performed.

**Relation to [I-39](#i-39).** Producer-side complement. After `buildStack` lifting, no chapter row should have `line.namespace` strictly deeper than every dep — lifting moves the result up to the closest-to-`main` ancestor with origin, so at least one dep is at the result's scope. This invariant therefore acts as a backstop: if a future producer-side regression bypasses lifting, the verifier still catches the falsified row.

**Why.** Sound under GL's validity-stack semantics ([`20_core_concepts/04_validity_stack.md`](20_core_concepts/04_validity_stack.md) — `deeperOf(a, b)` returns the deeper of two comparable ids and is the canonical "fact's effective scope when joined"). A verified proof graph that admits result_ns strictly deeper than every input means a producer-side bug forged a scope-promotion the prover's hash kernel cannot emit. Catching such rows at the verifier (rather than chasing them through downstream regressions) follows the project's "failures are first-class" stance.

**Spot.**

- Verifier failure under `implication` on a chapter row whose `line.namespace` is strictly deeper than every namespace in `rest[1::2]`.
- Concrete example (rung-1 incubator branch, pre-fix): `(=[it_0_lev_0_32,2]) main_boundary_(implication23[2,8,int_lev_4_2365]) implication (>[1](in[1,u_1])(>[2](in2[2,1,u_3])(>[3](in2[3,1,u_3])(=[2,3])))) main (in2[2,6,3]) main (in2[it_0_lev_0_32,6,3]) main (in[6,1]) main`. Every constituent at `main`; result at `main_boundary_(implication23[2,8,int_lev_4_2365])`. D-35 alone passes; the deeperOf-equality rule rejects. Post-lifting, the row no longer exists (lifted to `main`).

**Fix.** Producer side — fix the prover-side path that emitted the misnamespaced row (typically a `validityName` set/copy that bypassed `nm.deeperOf` accumulation). The verifier check is forcing-function for the producer-side audit; do NOT weaken the check. For the chapter-193 line-102 instance, the producer-side fix is [I-39](#i-39): `buildStack` now lifts the row's namespace to the scope where its origin actually lives.

**Code.** `check_implication` namespace block at [`verifier.py`](../verifier.py). Tests covering the failure mode: `tests/test_verifier_implication.py` (`test_implication_result_deeper_than_every_constituent`, `test_implication_result_deeper_than_every_constituent_simple`, `test_implication_premise_ns_sibling_via_pair_comparable`). C++ source of truth: [`generateEncodedRequestsStatic` + `growBaseCandidates`](../GL_Quick_VS/GL_Quick/src/memory.cpp).

**Decision ref.** [D-58](40_decisions.md#d-58).

---

<a id="i-37"></a>
## I-37 Algebra `rejectedMap` is never written by equi-class application

**Scope.** Prover — algebra equivalence-class hook surface. Specifically the new `prover.hpp::applyEquivalenceClassToAdmissionMap` and (in negation) any future temptation to mirror the integration side's `applyEquivalenceClassToRejectedMapIntegration` pattern on algebra.

**Rule.** Equi-class application on the algebra side reads but **never writes** `HashMemory::rejectedMap`. The map is mutated only by:

1. The original disintegration path (`disintegrateExpr2` → Pass B rejection-buffer commit) inserting fresh rejection records.
2. `prover.cpp::revisitRejected2` snapshotting + erasing entries during revival.

Any code that takes an existing `rejectedMap` entry, substitutes its variables via an equivalence class, and re-inserts the substituted form is a violation.

**Why.** `rejectedMap` holds real disintegration products whose `disintegration` origins were written at production site by `prover.cpp::disintegrateExprCore2::trackExpansionHistory` (the lambda emits `originDisintegration` for each element of an existence / and / or compound). Those origins are bound to the original (un-substituted) expressions and live in `Memory::exprOriginMap` from the moment the disintegration walk completes. Inserting a substituted constituent into `rejectedMap` would create a deferred-match record for an expression that has no corresponding `disintegration` origin in `exprOriginMap` — a provenance gap. Subsequent revival of that record would either (a) emit a chapter row with a soft `equality1` self-source origin (no foundation origin present to displace it via `addOrigin`'s cap-full preference, so the verifier sees `len(rest) < 4` and fails), or (b) tag the constituent under a borrowed `disintegration` origin pointing to the pre-substitution expanded form — a row that does not correspond to a real prover step.

The integration side (`prover.hpp::applyEquivalenceClassToRejectedMapIntegration`) currently does perform this kind of substituted insertion (D-43 additive-on-no-match behavior). That is acknowledged as a provenance-leak bug; per user direction it is not fixed on this branch. The algebra side follows the correct playbook from inception: rewrite admission keys instead (admission map is metadata; no proof-graph history attached — see [I-36](#i-36) for the companion arg-equalization rule).

**Spot.**

- A new function name on the algebra side containing `RejectedMap` and `Equiv`. Reject at PR review.
- Any write to `HashMemory::rejectedMap` (`insert`, `[].insert`, `emplace`, or container modification) inside a function whose name or doc references equivalence-class application. The two legitimate writers are `prover.hpp::updateRejectedMap` (called once from `disintegrateExpr2`'s rejection commit) and `prover.cpp::revisitRejected2` (snapshot + erase).
- A chapter row tagged `equality1` or `disintegration` whose source expression cannot be located in any earlier chapter row at any ancestor scope — a likely symptom of a substituted entry whose pre-substitution form was never independently registered.

**Fix.** When equi-class application would expand admission to cover a previously-rejected cohort, route the broadening through the admission key (call `applyEquivalenceClassToAdmissionMap` instead). `revisitRejected2(K', mb, depositValidity)` then walks the unchanged `rejectedMap[K']` and mail-emits the cohort verbatim.

**Code.** [`prover.hpp::applyEquivalenceClassToAdmissionMap`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`prover.cpp::revisitRejected2`](../GL_Quick_VS/GL_Quick/src/prover.cpp), [`prover.cpp::disintegrateExprCore2`](../GL_Quick_VS/GL_Quick/src/prover.cpp) (the production-site origin emitter). See also [D-57](40_decisions.md#d-57), [I-36](#i-36).

---

<a id="i-36"></a>
## I-36 Algebra equi-class rewrites preserve positional collision pattern

**Scope.** Prover — `prover.hpp::applyEquivalenceClassToAdmissionMap`. Algebra equi-class hook on the admission map.

**Rule.** For every candidate rewrite of an `admissionMap` key K → K' produced by `enumerateEqClassRewrites`, the rewrite is admissible only if it preserves the positional collision pattern. Concretely:

> For every pair `(i, j)` with `i < j` in K's arg list: if `origArgs[i]!= origArgs[j]` then `newArgs[i]!= newArgs[j]` must hold.

If a class substitution would collapse two previously-distinct arg slots into the same value, the candidate rewrite is dropped — no admission insert, no `revisitRejected2` call. The filter lives inside the sink lambda passed to `enumerateEqClassRewrites` in `applyEquivalenceClassToAdmissionMap`.

**Why.** An `admissionMap` key K is a marker-form expression encoding the positional structure of the admission rule. Two slots holding the same arg encode "these two positions take the same value"; two slots holding different args encode "these positions can take independent values". Substituting via an equi-class so that distinct slots collapse to the same value produces a K' whose positional structure differs from K's. `revisitRejected2(K', mb, validityName)` would then probe `rejectedMap[K']` — which only contains rejections rejected under K''s collapsed shape, NOT rejections rejected under K's original distinct-arg shape. Matching mixes semantically distinct rejections; reviving them under K's rule semantics is unsound.

The integration side does NOT have this filter (it allows arg-equalizing rewrites; see `prover.hpp::applyEquivalenceClassToRejectedMapIntegration`). User clarification: "regarding application of equi classes. i see for integration it admits equal variables where original had none. it is nonsense. we do not correct it for integration but for algebra it must be like: normalized signature after renaming == before renaming. no arg equalization here." For algebra the filter is mandatory.

**Spot.**

- A chapter row generated post-revival under a rule whose marker key has identical args at slots that were distinct in the producer admission key.
- A spurious admission-fire at a fact whose argument positions don't match the original admission rule's distinct-slot signature.
- Removal of the arg-equalization filter loop in `applyEquivalenceClassToAdmissionMap` (the nested `for (i) for (j > i)` collision check before each `uniqueRewrites[...] =` insert in the sink lambda).

**Fix.** Restore the filter. The check is O(|args|²) per candidate rewrite — negligible at GL's admission-key arity (typically ≤ 5 args). Do not bypass for "performance"; correctness depends on the filter.

**Code.** [`prover.hpp::applyEquivalenceClassToAdmissionMap`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (the sink-lambda filter). See also [D-57](40_decisions.md#d-57), [I-37](#i-37).

---

## Meta-remarks

- **Ordering rationale.** Invariants are sorted by when they entered the document, not by priority. For a priority-ordered view (what an agent checks first when debugging), the quick-reference table in [`AGENT_SwDD.md`](AGENT_SwDD.md#invariant-quick-reference) groups by scope.
- **Stability of numbering.** Once an invariant is numbered, the number is immutable. If an invariant is retired, its section becomes a short "retired — superseded by I-K" stub, preserving the anchor.
- **External cross-references.** Memory files referenced by name are the long-form rationale; this document is the operational rule.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
