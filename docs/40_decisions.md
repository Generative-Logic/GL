<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Decisions `[DRAFT]`

> A dated log of architectural and tactical decisions that shape GL's current form. Each entry answers: **what was the choice, when, and why**. Entries are append-only. When a previous decision is revised, do not delete it — add a new entry that supersedes it, and annotate the old one.

Entries sorted reverse-chronological (newest first). Unknown precise dates are flagged.

---

<a id="d-59"></a>
## D-59 — Local MSVC introsort port at the 4 `generateEncodedRequests*` sites (2026-05-12) — **TEMPORARY workaround to unblock the v0.8.1 bug-fix release; migrate to `stable_sort` once RT is again under control**


**Status — temporary workaround.** This entry documents a deliberately-temporary measure that ships in **v0.8.1 only** to unblock the bug-fix release. The intended end state is `std::stable_sort` (emergence-order tie resolution) at the same four sites, which is ~1.6× faster on the full pipeline but currently regresses the Gauss/fold theorem. The migration trigger is documented in the "Migration" subsection below.

**What.** A new header [`msvc_sort.hpp`](../GL_Quick_VS/GL_Quick/src/msvc_sort.hpp) provides `gl::msvc_sort` — a faithful re-implementation of MSVC STL's `_Sort_unchecked` (introsort dispatcher with insertion-sort cutoff at 32, heap-sort fallback when the `1.5·log₂(N)` recursion budget is exhausted, 3-way Hoare partition with Tukey's-ninther pivot selection for ranges > 40). It is wired into four sort sites in the rule-firing hot path:

- `memory.cpp::generateEncodedRequestsStatic` — `filteredIdx[]` sort
- `memory.cpp::generateEncodedRequestsStaticPairs` — `filteredIdx[]` sort
- `memory.cpp::generateEncodedRequestsStaticPairs` — `merged[]` pair-merge sort
- `filter.cpp::generateEncodedRequestsStaticCE` — CE-batch `filteredIdx[]` sort

Each replaces a `std::stable_sort` with `gl::msvc_sort`. The comparator is unchanged (name-only: `nm.decode(.nameId)`-based).

**Why (the immediate problem).** Two prior approaches both broke the Gauss / fold theorem proof relative to historical Win11 main HEAD behavior:

1. **4-site total-order tiebreaker** ( /, commit ). Widened the comparator from `name` to `(name, originalId, validityId)` — no ties at all. Produces deterministic output across hosts BUT yields a different tie ordering than MSVC's introsort. On with this comparator: Win11 ran the full pipeline in 51 min, 35,448 verifier checks airtight, 40 proved theorems, **no `fold[…]` theorem**.
2. **`std::stable_sort` + name-only comparator** (emergence-order ties). Identical on Win11 and Linux/WSL by the `stable_sort` contract. WSL ran the full pipeline in 31 min (1.6× faster), 35,403 verifier checks airtight, 40 proved theorems, **no `fold[…]` theorem**.

Both miss what historical Win11 main HEAD (`std::sort` with name-only comparator, MSVC introsort) was apparently doing: producing a tie order in which the Gauss / fold proof goes through. Reading the deltas, MSVC's introsort tie ordering was load-bearing for the fold-theorem search path; neither widening the comparator nor switching to stable_sort preserves it.

**Why (cross-host concern).** `std::sort` is unstable, and MSVC STL (`_Sort_unchecked`) and libstdc++ (`__introsort_loop`) resolve ties using different pivot strategies, cutoff thresholds, and recursion-budget formulas. For inputs with many equivalent elements (typical of `IntEncodedExpr` arrays where multiple entries share an operator name) the two implementations produce visibly different byte output. `gl::msvc_sort` is byte-deterministic regardless of which standard library is in use: cross-host divergence at these four sites is structurally impossible.

**Empirical result (validation).** WSL run with `gl::msvc_sort` at all 4 sites:

| Metric | Win11 (total-order) | WSL (stable_sort) | **WSL (gl::msvc_sort)** |
|---|---|---|---|
| Proved theorems | 40 | 40 | **42** |
| `fold[…]` theorems | 0 | 0 | **2** |
| Verifier checks | 35,448 | 35,403 | 35,746 |
| Verifier failures | 0 | 0 | 0 |
| Runtime | 51 min | 31 min | 34 min |

42 vs 40 proved theorems — both forward and mirror Gauss / fold variants land in `proved_theorems.txt`. 0 verifier failures on 35,746 checks. Runtime sits between total-order (slow) and stable_sort (fast).

**Cost of the workaround.**

- **Runtime.** `gl::msvc_sort` is unstable; it pays the partitioning-overhead penalty that stable_sort avoided. Measured cost on WSL: 34 min vs 31 min for stable_sort (~10% slower). On Win11 historically: 51 min for the comparator-widened path; `gl::msvc_sort` on Win11 not yet measured but expected near the WSL number.
- **Maintenance.** A re-implementation of MSVC's std::sort that we now own. The implementation is line-by-line translated from `microsoft/STL/stl/inc/algorithm`; the upstream is stable (the algorithm is settled), so drift is low-risk. But it is non-zero code to keep in the tree.
- **Conceptual.** GL's prover-pipeline behavior is sensitive to sort tie order. This is a **latent correctness fragility** — the proof of the fold theorem should not depend on which way `std::sort` resolves ties on a particular STL implementation. The right long-term fix is to make the prover insensitive to tie order (e.g. by deduping symmetric search paths, or by using an explicit priority queue keyed on emergence index or some other principled criterion). `gl::msvc_sort` simply freezes the historical tie order so the bug-fix release can ship.

**Migration plan.**

Once the fragility above is addressed (specifically: once the prover's RT does NOT depend on `std::sort`-vs-`stable_sort` tie choice for the Gauss / fold theorem and other regressing theorems), all four sites migrate to `std::stable_sort` with the same name-only comparator. Trigger:

1. The RT campaign lands. The 100–1000× expected memory reduction and the static-prover refactor (project_next_rt_campaign / project_asic_0_1_release) are expected to substantially restructure rule-firing order discipline.
2. After RT lands, re-run the four-site sort experiment: try `stable_sort` at all 4 sites and confirm fold + the other 41 theorems still prove. If yes → migrate, delete `msvc_sort.hpp`, recover the ~10% runtime.
3. If no → the tie-order dependency persists. At that point either deepen the search-order discipline in the rule-firing loop (e.g. introduce an emergence-index secondary key on the comparator) or accept `gl::msvc_sort` as the permanent answer.

The expectation is that the RT campaign rewrites enough of the rule-firing layer that the fragility evaporates and `stable_sort` becomes viable. This decision should be revisited explicitly when RT lands; until then, `gl::msvc_sort` is the configured sort.

**Alternatives considered.**

- **`std::stable_sort` at all 4 sites (emergence-order ties).** Rejected: 1.6× faster but does not prove the Gauss / fold theorem on either host. Excluded for v0.8.1; first candidate for re-evaluation post-RT.
- **4-site total-order tiebreaker `(name, original, validity)`.** Rejected: deterministic and platform-independent but BREAKS the fold theorem by producing a different tie order than MSVC introsort. The investigation made this clear empirically; documented in commit f5654dc0's body.
- **Ship just the sandbox-branch `stable_sort` for the speedup, lose fold.** Rejected: the bug-fix release explicitly cannot regress a theorem that the previous release proved.
- **Patch upstream MSVC + libstdc++ to agree.** Not feasible; we don't control either STL.
- **Use boost::sort or another third-party sort.** Rejected: adds a dependency for a problem we can solve in ~250 LOC in-tree.
- **Sort-free firing order (priority queue).** Out of scope for v0.8.1; correct long-term direction; depends on RT-campaign restructuring.

**Investigated / ruled out during execution.**

- Whether MSVC `std::sort` and `std::stable_sort` produce the same output for our typical inputs. Answer: confirmed they do NOT, both empirically (the fold-theorem regression) and from MSVC STL source (`_Sort_unchecked` is introsort = unstable; `stable_sort` is a separate function with its own implementation and explicit-overflow analysis at `_ISORT_MAX = 32`).
- Whether the speedup observed when switching to `stable_sort` was from the algorithm or from Linux/g++ vs Win11/MSVC. Answer: not fully isolated — needs a Win11 + stable_sort run to fully separate. Best read so far: ~10% comes from stable_sort's avoidance of partitioning overhead on big tie groups; the rest may be host. Not load-bearing for this decision (we're already accepting the 10% to keep the fold theorem).
- Whether other `std::sort` call sites in the prover hot path could regress similarly. Answer: there are 5 other `std::sort` calls in `memory.cpp` / `filter.cpp` (lines 1374, 1419, 1432, 129, 430) that we left untouched. Empirically the WSL run with `gl::msvc_sort` only at the 4 named sites already proves fold; the others are not load-bearing for the immediate problem.

**Code.** [`msvc_sort.hpp`](../GL_Quick_VS/GL_Quick/src/msvc_sort.hpp), [`memory.cpp::generateEncodedRequestsStatic`](../GL_Quick_VS/GL_Quick/src/memory.cpp), [`memory.cpp::generateEncodedRequestsStaticPairs`](../GL_Quick_VS/GL_Quick/src/memory.cpp), [`filter.cpp::generateEncodedRequestsStaticCE`](../GL_Quick_VS/GL_Quick/src/filter.cpp).

---

<a id="d-57"></a>
## D-57 — algebra equi-class hook rewrites admissionMap, never rejectedMap (2026-05-12)


**What.** A new method `ExpressionAnalyzer::applyEquivalenceClassToAdmissionMap` in `prover.hpp` is the algebra-side equi-class hook on the admission map. It walks `admissionMap` entries; for each entry at marker key K with `AdmissionMapValue` set V, it enumerates rewrites K' via the equivalence class (using the shared `enumerateEqClassRewrites` helper) with an arg-equalization filter, applies the substitution to both K (the marker key) and the contents of each `AdmissionMapValue` (its `key` vector and `remainingArgs` set), and ADDITIVELY inserts the (K', V') entry into `admissionMap`. Each new K' is followed by `revisitRejected2(K', mb, depositValidity)` which walks the unchanged `rejectedMap[K']` and mails any matching rejection cohort to `internalMailIn`. `rejectedMap` is never written by this function. The hook is wired into `addStatement` at the four equi-class application sites (same-NS, ancestor, descendant per D-33, fixpoint) immediately after each `applyEquivalenceClassToRejectedMapIntegration` call.

**Why.** The integration-side `applyEquivalenceClassToRejectedMapIntegration` (prover.hpp::`applyEquivalenceClassToRejectedMapIntegration`) additively inserts substituted rejection records into `rejectedMapIntegration` on its no-match path. The substituted `concreteConstituent` and `siblings` in the new entry have `disintegration` provenance recorded only for their pre-substitution forms at the original production site (via `trackExpansionHistory` inside `disintegrateExprCore2`), not for the post-substitution forms. That is a provenance gap. Per user direction the integration code stays as-is; for algebra a different playbook applies:

1. **`rejectedMap` is sacred.** It holds real disintegration products with `disintegration` origins recorded at production site. Equi-class application never writes to `rejectedMap` — see [I-37](30_invariants.md#i-37).
2. **`admissionMap` keys and values are metadata.** No proof-graph history attached. Equi-class rewrites them freely.
3. **Keys are ADDED, not replaced.** The original K stays in `admissionMap`; K' coexists alongside it. Multiple rewrites at multiple class instances accumulate without erasing earlier entries.
4. **Arg-equalization is forbidden.** Rewrites that collapse two previously-distinct arg slots to the same value are dropped — see [I-36](30_invariants.md#i-36). Preserves the positional collision pattern of the admission key.

The revival path (mail-emit on `internalMailIn` via `revisitRejected2`) handles two cases uniformly:
- **Direct admission insert** (existing `revisitRejected2` callers at prover.cpp::`updateAdmissionMapRecursion` and memory.cpp::`makeMandatoryEncodedStatementLists1Static`): no rewrite, K is the bare-marker admission key, pre==post in the mail, no equalities. The mail's `equality1` origin is a soft placeholder — see *Composition with D-49* below.
- **Equi-class revival** (the new hook): K' is the rewritten admission key; the rejection record at `rejectedMap[K']` carries the unchanged cohort with intact `disintegration` provenance from production site. revisitRejected2 mails the cohort verbatim.

**Composition with D-49 / I-35.** The mail-deposited `equality1` origin (single-entry `origin.second` = pre-form, no equalities) is structurally insufficient for `verifier.py::check_equality1` (rejects `len(rest) < 4`). It is harmless because the proper `disintegration` origin for each child was written at the original production site by `disintegrateExprCore2::trackExpansionHistory` (the lambda inside `disintegrateExprCore2` emits originDisintegration per child); when the mail-deposited child lands in `body.exprOriginMap` via `addExprToMemoryBlock`'s `addOrigin` call, `addOrigin`'s cap-full preference clause (`prover.hpp::addOrigin`) refuses to displace a foundation origin with a convenience equality1 — existing slot wins. Chapter export reads the `disintegration` origin; verifier accepts.

**Scope handling.** Three directions admitted, mirroring [D-33](#d-33): same-NS, class-shallower (ancestor of entry), class-deeper (descendant of entry). Deposit scope is `deeperOf(classScope, entryScope)`. Visibility soundness: a class at descendant V can rewrite an admission entry at ancestor S because the admission rule's contents are visible at V; the rewritten admission entry lives at V (descendant), valid at V and below.

**`AdmissionMapValue` substitution.** Both fields that carry variables (`key` vector and `remainingArgs` set) get `ce::replaceKeysInString` with the rewrite's substMap. The substMap keys are bare variable names (u_-prefixed names are never class members), so `u_`-prefixed args in the value are naturally excluded from substitution. `standardMaxAdmissionDepth`, `standardMaxSecondaryNumber`, and `flag` are copied unchanged. `admissionStatusMap[K']` is inherited from `admissionStatusMap[K]` only when no prior entry exists at K' (preserve existing entries — the additive principle).

**Performance.** A new monotonically-growing cache `HashMemory::varsInAdmissionMapKeys` is populated at every `admissionMap` insert (4 sites: `prover.hpp::prepareIntegration`, `memory.cpp::makeMandatoryEncodedStatementLists1Static`, `prover.cpp::updateAdmissionMapRecursion`, and the new hook itself). The hook short-circuits in O(|class|) when no class variable appears in any admission key — same pattern as `varsInRejectedMapIntegrationKeys`. Without this cache the hook would walk all admission entries per class call (Gauss-scale: ~10⁴–10⁵ entries × ~10⁶ class calls = catastrophic). Final cost depends on overlap density; bounded above by per-entry class-member scan + mapping enumeration.

**Iteration safety.** Inserts, status updates, cache populates, and `revisitRejected2` calls are queued in `toInsert` during the admissionMap walk and applied after the walk completes. `revisitRejected2` internally calls `cleanAdmissionMap` which may erase the just-inserted K' (if marker is in the operator's output slot — see `prover.hpp::cleanAdmissionMap`). That erasure is permissible: principle (3) only requires the original K to be preserved, and K' has fulfilled its purpose (it carried the revival).

**Alternatives considered.**

- **Mirror integration's rejectedMap rewrite (additive, with arg-equalization filter).** Rejected: the substituted constituents in the new K2 entry would still lack post-substitution `disintegration` provenance, regenerating the same gap that motivated this design. The arg-equalization filter would close one class of bad rewrites but not the provenance issue.
- **Rewrite admissionMap WITHOUT calling revisitRejected2.** Rejected: equi-class application would broaden future admission but skip retroactive revival of already-rejected cohorts. The whole point is to retroactively unblock rejections that an equi-class makes admissible.
- **No arg-equalization filter.** Rejected: a rewrite where the substitution collapses two previously-distinct slots in K to the same value produces an admission key whose positional structure differs from K's. Probing `rejectedMap[K']` for such a collapsed K' returns false matches (the rejected cohort at K' was rejected for the collapsed shape, not for the original; matching mixes semantically distinct rejections).
- **Replace integration's buggy rmi-rewrite at the same time.** Rejected per user direction "we do not correct it for integration" — integration's gap is acknowledged and scoped out of this branch.

**Investigated / ruled out during execution.**

- Whether `revisitRejected2`'s degenerate `equality1` mail origin would leak to the chapter and fail verifier `check_equality1`. Traced through addOrigin's cap-full preference (`prover.hpp::addOrigin`) and confirmed empirically: 0 of 153 sampled `equality1` rows in shipped chapters are degenerate (all have `len(rest) ≥ 4`).
- Whether `disintegrateExprCore2` autonomously emits an `expansion` origin for absorbed compounds. Answer: yes, via `trackExpansionHistory` (`prover.cpp::disintegrateExprCore2`); the `expansion`+`disintegration` origin chain is written at original production time, so the revival path does not need to re-emit it.
- Whether the bare-form-of-K' would mismatch the rejectedMap key form. Answer: admission map keys are bare-marker form (no `u_`) — the rewritten K' is directly usable as a rejectedMap key, no `u_`-strip needed.

**Code.** [`prover.hpp::applyEquivalenceClassToAdmissionMap`](../GL_Quick_VS/GL_Quick/src/prover.hpp), four wire sites in `addStatement` immediately after each `applyEquivalenceClassToRejectedMapIntegration` call. Cache populate at every admissionMap insert in `prover.hpp::prepareIntegration`, `memory.cpp::makeMandatoryEncodedStatementLists1Static`, and `prover.cpp::updateAdmissionMapRecursion`. `revisitRejected2` body refactored separately to mail-first emission (see C2 commit on this branch).

---

<a id="d-56"></a>
## D-56 — buildStack lifts every chapter-row scope to closest-to-`main` ancestor with an origin (2026-05-11)


**What.** `visualizer.cpp::buildStack` lifts every `(expr, validity)` it visits to the closest-to-`"main"` ancestor of `validity` whose `(expr, ancestor)` key has an origin entry in the emitting LB's `exprOriginMap`, **subject to an OR-branch barrier** (lift never crosses `_boundary_orint_` or `_boundary_ordis_` delimiters). The lift applies to the entry-side `proved`, to every dep written into a chapter row cell, and to every dep passed to a recursive `buildStack` call. The `covered` dedup set keys on lifted forms, so each `(expr, lifted_v)` pair has at most one row in the chapter.

The OR-branch barrier emerged from the first verification run: the unbounded lift produced 4 incubator verifier failures (2 `or convergence`, 2 `origin chain termination`) on chapter `1209_direct_proof.txt` (FTA-rung-1 forward direction). Both branches of two OR-convergence rows had their `(preorder, branch_scope)` deps lifted to the parent boundary where the post-convergence origin was recorded. The convergence row's deps collapsed to identical pairs, breaking `check_or_convergence` (which requires branch-distinct dep namespaces) and producing the corresponding `origin chain termination` cycles. The barrier restores branch-distinct namespaces by stopping the lift at the deepest `orint_`/`ordis_` ancestor.

**Why.** Pre-lifting, `buildStack`'s lookup was "exact-key, else `(expr, "main")` shadow"; on a hit via the shadow, `emitRow` wrote `proved.validityName` (the un-shifted deep boundary) into `row[1]` while citing the shallow-scope origin's main-scope deps. The chapter row claimed the derivation happened at a deep scope when in fact the origin lived at `main`. Concrete bug instance: chapter `193_check_induction_condition.txt` line 102 emitted `(=[it_0_lev_0_32,2])` on `main_boundary_(implication23[2,8,int_lev_4_2365])` with implication-rule deps all on `main`. A targeted trap on `buildStack::emitRow` confirmed the falsification path ( at investigation time).

Three alternatives were considered:

- **(A) Widen consumers.** Extend `verifier.py`'s dep-validity equality checks to "ancestor-or-equal" matching (using the existing `_ns_matches_or_strict_prefix` helper, enhanced with `"_boundary_"` delimiter check), and extend `generate_full_proof_graph.py`'s JS click handler to do a DOM-walk ancestor fallback when an exact-namespace card doesn't exist. Preserves chapter shape; spreads new semantics across two consumers.
- **(B) Lift at the producer.** Change `buildStack` to walk ancestors and emit at the closest-to-`main` scope with an origin. Verifier and HTML stay exact-match.
- **(C) Leave as is.** Chapter rows remain falsified; future readers (other agents, auditors) work around the mismatch.

(B) wins because: (i) chapter rows become truthful — `row[1]` is where the derivation actually lives, (ii) deduplication happens automatically via the existing `covered` set, (iii) the verifier and HTML consumer semantics simplify (exact equality / `getElementById` continue to work), (iv) the change is concentrated at one site (`buildStack`) instead of being distributed across consumers. (A) would have required a JS-side DOM-walk handler with both string-prefix-proximity AND DOM-distance-nearest as the two criteria for the jump target — adding complexity that lifting makes unnecessary. (C) is a hard pass per [I-16](30_invariants.md#i-16) and the project's general "failures are first-class" stance.

**Soundness argument.** An origin entry at `(expr, V)` in `exprOriginMap` was recorded by the prover when a rule fired producing `expr` with every premise available at scope `V`. Per [I-2](30_invariants.md#i-2), every non-root `V` is `parent + "_boundary_" + payload` and inherits all parent-scope facts. Therefore lifting from `(expr, V_deep)` to `(expr, V_root)` where `V_root` is the closest-to-`main` ancestor with origin → the rule did fire at `V_root` → premises were at `V_root` or shallower → lifted row is true. The asymmetric direction holds: ancestor facts are universally available in descendants, descendant-only facts are not — and lifting only moves toward `main`, never away.

**Consequences.**

- Chapter row counts shift. A chapter that previously had K rows at deep scopes (via the now-retired main-fallback retag) may now have K rows at `main`, deduplicated against existing main-scope rows. Verifier check counts move; the new counts are stable across re-runs but do not match the pre-lifting baseline. See [G-41](50_gotchas.md#g-41).
- The verifier's dep-validity matching stays at exact `==` (no `_ns_matches_or_strict_prefix` extension required outside of `equality1` / `equality2`, which use the helper for a different purpose).
- HTML namespace-tag jumps are deterministic by construction: every `(<ns>)` cited in a row points to a scope that has rows, so the corresponding subproof card exists.
- and debugging methods continue to apply; the chapter side now reflects the lifted form, simplifying the comparison.

**Code.** Helper `liftToShallowestOriginAncestor` at [`visualizer.cpp`](../GL_Quick_VS/GL_Quick/src/visualizer.cpp). Five invocation sites inside [`visualizer.cpp::buildStack`](../GL_Quick_VS/GL_Quick/src/visualizer.cpp): entry, candidate-loop dep lift (per candidate), `emitRow`'s dep emission, recursive call's `ingredient`, last-resort `front`-fallback. Related: [I-39](30_invariants.md#i-39), [G-41](50_gotchas.md#g-41).

**Relation to [D-58](#d-58).** Same falsified-row class as the chapter-193 line-102 example named in both entries. This decision is the producer-side fix (lift moves the row to `main`, eliminating the falsified emission). D-58 is the verifier-side complement (rejects the falsified row at the verifier layer). With both landed, the chapter-193 instance is fixed at the producer; the verifier check stays as forcing-function for any future producer-side regression.

**Supersedes.** The chapter-emission portion of [D-51](#d-51): the `__contradiction__` LB fallback and the chapter-goal in path mechanisms still apply; only the "exact-key + main-shadow" lookup is replaced by the ancestor walk.

---

<a id="d-58"></a>
## D-58 — Verifier `check_implication` enforces `deeperOf`-equality on result namespace (2026-05-11)


**What.** `check_implication` (the verifier's most frequently-fired per-tag checker) gained a structural rule mirroring C++ `generateEncodedRequestsStatic` + `growBaseCandidates`'s `nm.deeperOf` accumulation. On top of the existing [D-35](#d-35) comparable-scope premise inheritance:

1. Every PAIR of constituent namespaces (impl + each premise) must be **comparable** (`_ns_matches_or_strict_prefix` in either direction).
2. The row's namespace (`line.namespace`) must EQUAL the deepest constituent — i.e. `line.namespace ∈ {impl_ns} ∪ {premise_nss}`.

Rule formalised as [I-38](30_invariants.md#i-38).

**Why.** Sound under GL's validity-stack semantics. The prover's hash kernel accumulates the joined-scope of combined facts via `nm.deeperOf` ([`20_core_concepts/04_validity_stack.md`](20_core_concepts/04_validity_stack.md): *"when a fact derived in scope `a` is broadcast to an LB active in scope `b`, the fact's effective scope becomes `deeperOf(a, b)`"*). An `implication` row whose result lives at a strictly deeper scope than every constituent encodes a derivation step the kernel cannot have emitted — must be a producer-side bug.

D-35 alone catches sibling and deeper-than-result-premise scopes, but admits the case where every constituent is a strict ancestor of `result_ns`. The new rule closes that gap.

**Relation to [D-56](#d-56).** Producer-side complement on the same row-falsification class. The buildStack lifting eliminates the chapter-193 line-102 instance at emission time (the row now lives at `main`, where its origin actually lives). The verifier deeperOf-equality check is forcing-function for any future producer-side regression that bypasses lifting — a falsified row would fail the check rather than slip through.

**Spot.** Verifier failure under `implication` on a row where `line.namespace` is strictly deeper than every namespace in `rest[1::2]`. Concrete pre-fix example documented in [I-38](30_invariants.md#i-38)'s *Spot* section (rung-1 incubator branch, `(=[it_0_lev_0_32,2])` deposited at `main_boundary_(implication23[2,8,int_lev_4_2365])` from constituents all at `main`). Post-merge: the row no longer exists because `buildStack` lifting moved it to `main`; the verifier check stays as a backstop.

**Trade-off considered.** Could have been encoded prover-side as an assert at the emission site. Verifier-side is preferred because (a) the verifier is the airtight regression gate per [I-16](30_invariants.md#i-16); (b) the C++ assert mechanism already covers the in-prover invariants — the verifier covers post-emission audit; (c) the check is cheap (O(K²) pairs for K ≤ ~10 premises, well within the verifier's per-row budget).

**Files touched.** `verifier.py` (15 LOC structural rule added to `check_implication`), `tests/test_verifier_implication.py` (+3 failure tests exercising the rule), `docs/20_core_concepts/08_proof_tags.md::implication` (Rule 10 doc-sync), `docs/30_invariants.md` (new I-pending entry), `docs/40_decisions.md` (this entry).

**Code.** [`check_implication`](../verifier.py); C++ source of truth [`generateEncodedRequestsStatic` + `growBaseCandidates`](../GL_Quick_VS/GL_Quick/src/memory.cpp).

---

<a id="d-55"></a>
## D-55 — OR-disintegration's K mutual-exclusion sub-implications get auditable provenance (2026-05-09)


**What.** Six coordinated edits close three related concerns around OR-disintegration: (a) auditability for the K mutual-exclusion sub-implications, (b) deduplication / parent-removal at OR-construction time, and (c) the firing-site cross-scope rule that was masking (b)'s viability.

**Part A — K-implication provenance.**

1. **Prover origin recording.** [`disintegrateExprCore2`](../GL_Quick_VS/GL_Quick/src/prover.cpp)'s OR case now stamps each of the K implications `(>[](!d_others) … d_k)` with `("disintegration", [(expandedOrSignature, validityName)])` in `exprOriginMap` and `mailOut.exprOriginMap`. The expanded-OR signature is computed once per OR via the existing `expandSignature(ent)` helper, mirroring the &/existence pattern already in `trackExpansionHistory`. KEY u_-stripped to match the &/existence pattern.
2. **`trackExpansionHistory` hoisted above the orAdmitted gate.** Pre-fix, the OR's expansion-origin record was recorded only when `orAdmitted == true` (i.e. when per-branch case-split fires). The K implications, which fire **regardless** of `orAdmitted`, then cited an expanded form that had no matching expansion-origin record when admission failed — leaving the verifier's `check_disintegration` dispatch (which walks chapter for the matching `expansion` row) without a target. Post-fix, `trackExpansionHistory(ent, false)` runs unconditionally inside the `currentOrDepth < max_or_depth` block, so the expansion row exists whenever any K-implication is emitted.
3. **Verifier `or` branch in `check_disintegration`.** New helper `_check_or_disintegration_implication(line, compact, entry)` ([`verifier.py`](../verifier.py)) substitutes the compound's args into the binary entry's disjunct templates, peels `line.expression` via `disintegrate_implication_full`, and verifies that the head matches one disjunct with premise multiset = `{!d_j: j!= head_index}`. Premise order is irrelevant to the check.

**Part B — OR-construction dedup + parent removal in `run_modes.cpp::fullRun`.**

4. **Disjunct-set dedup.** `orPairsFromHeadSwitch` is symmetrically populated by `headSwitchOne` walking `globalTheoremList` — for each theorem with a negated premise, both the original and its contrapositive are emitted as a pair. This produced `(mirror_a, mirror_b)` AND `(mirror_b, mirror_a)` for the same disjunct set, and `constructOrTheorem` was called for both, registering `or<N>` and `or<N+1>` for the same logical OR with disjuncts in opposite order (the source of `or0` / `or1` duplication on Peano). Post-fix, the OR-construction loop canonicalizes each pair as `(min, max)` lexicographically and skips pairs whose canonical form has already produced an OR. One OR per disjunct set.
5. **Parent removal at OR creation.** When an OR is constructed from `(exist, comp)`, both parent theorems are added to a `consumedParents` set. After the construction loop:
 - `consumedParents` are filtered out of `survivingTheorems` before `saveProvedTheoremsFiltered` writes `proved_theorems.txt` / `compiled_proved_theorems.txt`.
 - `consumedParents` are appended to `compressed_out_theorems.txt` — same procedure the compressor applies to redundant theorems, done manually here at OR-creation time so the cleanup lands in the same emit pass rather than racing the compressor.
 The parents are subsumed by the OR via OR-disintegration's K mutual-exclusion implications (Part A) — they are recoverable from `or<N>` + `disintegrateExprCore2`'s OR case, no longer cited as standalone theorems. Viability of this cleanup depends on Part C: with the parents broadcast as v=main universals, the original strict-eq firing-site check happened to mask the cross-scope bug; remove the parents and the K-impls fail to fire across scope without Part C's realignment.

**Part C — firing-site cross-scope rule realigned with `nm.comparable` (the line-1078 fix).**

6. **`checkLocalEncodedMemoryStatic` head-LMV scope check** ([`memory.cpp`](../GL_Quick_VS/GL_Quick/src/memory.cpp)). Pre-fix:
 ```cpp
   if (lmv.validityName != "main" && expressionListValidityName != "main"
       && lmv.validityName != expressionListValidityName) {
       continue;
   }
   ```
 Strict equality on the rule's scope vs the consensus scope, with both-main masking. Pre-Part-B the chapter-6 / chapter-11 mirror reformulations were broadcast Peano theorems at v=main, so the left arm of the AND short-circuited and the bug never surfaced. Post-Part-B, the parent mirrors are gone and the K-implication carries the OR's parent scope (impl26 in incubator's IncubatorGauss1 batch, not main); the rule's scope is non-main AND distinct from the Branch-A consensus scope, and the strict-eq check rejected legitimate firings of the rule on descendant-scope facts.
 This violates the prover's own comparable-scope inheritance rule — request generation already uses `nm.comparable` (per [`docs/20_core_concepts/02_hash_engine.md`](20_core_concepts/02_hash_engine.md#locality-semantics-a-local-rule-can-fire-against-entirely-non-local-premises) §Locality semantics). Firing-site was the only place still on strict-eq.
 Post-fix:
 ```cpp
   const int16_t lmvVid = nm.encode(lmv.validityName);
   if (!nm.comparable(lmvVid, consensusValidityId)) {
       continue;
   }
   ```
 `nm.comparable(a, b)` returns true iff one is an ancestor of the other on the same root-to-leaf path — the same "parent / strict-prefix in either direction" relation as the request-generation `pairMap` lookup. Both ends of the firing pipeline now agree on what scopes are mutually visible.
 Diagnostic trail: a firing-path trap (gated on the chap-1209 SE2 LB chain) confirmed rejection at exactly this site for the rung-1 K-impl `(>[]!(=[u_repl_lev_1_2,u_2])(existence2[u_1,u_repl_lev_1_2,u_3]))` — `[lmv 0 SKIP] step7_cross_scope_validity ruleV=…impl26 consensusV=Branch-A`. Trap stripped post-fix; same trap output was the empirical evidence that no legitimate `step7_scope_not_comparable` rejections occur in 35866-check airtight runs (counter: 0 across the fullest run).

**Why.**

- *Part A:* Pre-fix the K rules were anonymous hash-memory entries: when they fired downstream as `implication`-tagged chapter rows, the cited rule expression had no chapter to click through to and no provenance trail back to the originating OR. The "every inference step links to its antecedent" mandate (the project conventions "When working on this project") was silently violated for any chapter using an OR-derived rule. The K rules are deterministic structural consequences of the OR's De-Morgan form — they should be recoverable by anyone reading the proof graph, which now they are.
- *Part B:* Pre-fix the parent mirror reformulations stayed in `proved_theorems.txt` after their OR was constructed. Downstream chapters (notably FTA-rung-1's `1209_direct_proof.txt`) cited them directly, shadowing the OR theorem they had become equivalent to. With Part A's K-implication provenance in place, the parents are formally redundant — Part B is the corresponding cleanup so `proved_theorems.txt` reflects what's actually essential.
- *Part C:* The strict-eq firing-site check was a long-standing latent bug masked by the historical existence of v=main mirror reformulations as broadcast universals. Part B's parent removal removed the mask. Without Part C's realignment, Part B alone breaks rung-1: the K-impls land at the OR's parent scope (impl26), Branch-A facts at the descendant orint scope, and the strict-eq check rejects every cross-scope firing despite the scopes being on the same root-to-leaf path and request generation having already accepted the (rule, fact) pair. Part C restores the comparable-scope inheritance rule the rest of the prover already honours and unblocks Part B.

**Where it does NOT change behaviour.** Part A's prover origin records do not introduce new statements or new firings — they are pure metadata. Part B's run_modes-level cleanup happens after the prover and compressor have already settled; no proof closes that did not close before, and no proof fails that did not fail before. The observable differences:
- More `disintegration`-tagged rows in chapters that disintegrate ORs (today: incubator chapter 1209 and a handful of FTA-ladder chapters; main pipeline OR theorem chapters). Verifier check count rises by exactly the number of new rows.
- Fewer entries in `proved_theorems.txt` / `compiled_proved_theorems.txt` — the parent mirrors and the duplicate OR variant are gone, replaced by a single OR theorem and the compressed-out parent record.

**Tag description follow-on.** `tag_descriptions.json::disintegration` was extended to mention OR as a third disintegration target. Per the project conventions: if the new verifier branch surfaces failures on existing artefacts, those represent real provenance gaps the prior soft-checker hid — the response is to fix the producer side, not weaken the check.

**Cross-link.** OR-construction logic itself, including the dedup and parent-removal cleanup, is documented in [`20_core_concepts/07_or_branching.md`](20_core_concepts/07_or_branching.md#or-theorem-construction-run_modescppfullrun).

---

<a id="d-54"></a>
## D-54 — Single canonical `files/GL_binaries/` directory; verifier filters by chapter context (2026-05-08)


**What.** Three coordinated edits eliminate the duplicate `files/incubator/GL_binaries/` directory:

1. **C++ writer-path consolidation.** `visualizer.cpp::generateRawProofGraph`'s `glBinDir` switches from `outDir.parent_path / "GL_binaries"` to a project-rooted `<__FILE__>/../../../../files/GL_binaries`. The writer path now matches the reader path in `prover.cpp::compileCoreExpressionMap` exactly. Every batch — incubator and main alike — writes its post-batch dictionary to the single canonical `files/GL_binaries/` directory.
2. **Python merge skip for incubator tags.** `run_modes.py::_merge_into_shared` early-returns for tags whose name begins with `Incubator`, while still printing the `+0 new entries (shared total: <N>)` line for log parity. Incubator-allocated spontaneous compact-operator names are batch-local and must not propagate into the shared cross-batch registry — propagating them would shift the next main batch's counters and rename main's spontaneous operators, breaking [I-23](30_invariants.md#i-23) for the main pipeline.
3. **Verifier chapter-context filter.** Three surgical edits in `verifier.py`:
 - `verify_chapter`'s `current_gl_binary` selection (around line 3340) — when the chapter's theorem expression contains `AnchorIncubator`, restrict the candidate-tag iteration to `Incubator`-prefixed tags. Without this, the cross-anchor connection chapter `(>[..](AnchorIncubator[..])(AnchorPeano[..]))` would match `AnchorPeano` first (alphabetical) and bind `current_gl_binary` to the Peano binary, breaking 3 incubator chapter checks (`expansion: failure 1`, `disintegration: failure 2`).
 - `binaries_for_chapter` — when `current_gl_binary` is `None` and the chapter's theorem expression contains `AnchorIncubator`, restrict the fallback list to tags starting with `Incubator`.
 - `_check_reformulation`'s D-35 fallback (the V-6 helper added in `docs/verifier_rung1_changes.md`) — when `tag == "Incubator"`, restrict the existence-head scan to `Incubator`-prefixed tags.

The duplicate directory is then `git rm -r`'d. Three tracked files removed.

**Why.** Pre-2026-05-08 the writer path was asymmetric. Configs for incubator batches set `raw_proof_graph_folder = "files/incubator/raw_proof_graph"`, so `outDir.parent_path / "GL_binaries"` resolved to `files/incubator/GL_binaries/` — a directory the prover loader (`prover.cpp:194-198`, project-rooted) never read from and the Python `_merge_into_shared` step (also project-rooted) ignored. Incubator-allocated spontaneous compact-operator names were stranded on disk in a stale parallel folder. The user's directive: there must be a single `files/GL_binaries/` and nothing under `files/incubator/`.

The naive consolidation (just redirect the writer path and remove the directory) breaks the verifier. The duplicate directory was providing accidental name-isolation between main-batch and incubator-batch spontaneous allocations, which collide on shape:

| op | shape in main (Peano/Gauss/shared) | shape in incubator |
|------------|------------------------------------|--------------------|
| existence0 | arity 3 | arity 3 (same) |
| existence1 | arity 4 | arity 4 (same) |
| existence2 | arity 3 | **arity 8 in IncubatorPeano** |
| existence3 | arity 8 | arity 8 (same) |
| existence4 | arity 5 (Gauss/shared) | **arity 4 in IncubatorGauss1** |

Once both folders' contents share `files/GL_binaries/`, alphabetical iteration in `binaries_for_chapter` and `_check_reformulation`'s fallback would route incubator chapters' existence-head lookups against main-batch binaries — chapter `1210_reformulated_statement.txt` (arity-4 `existence4` substitution) would fail. The verifier filter restores the isolation explicitly.

**Trade-off considered.**

- *Namespace incubator allocations under a separate prefix* (e.g. `inc_existence4` instead of reusing `existence4`). Rejected as out-of-scope: would require C++ changes to `ExpressionAnalyzer`'s spontaneous-name allocator plus coordinated changes in process_proof_graphs.py and the verifier. The chapter-context filter is the minimum-blast-radius fix.
- *Skip the C++ export entirely for incubator batches.* Rejected: would leave `files/GL_binaries/GL_binary_IncubatorPeano.json` as the 4-byte `{}` Python seed and deprive any future cross-batch consumer of the post-batch incubator dictionary. The unified-write approach keeps full state on disk; the Python merge skip plus verifier filter handle the isolation.
- *Modify the chapter format to embed the originating tag explicitly.* Rejected: process_proof_graphs.py emits chapters today using just `AnchorIncubator`; embedding tag would require coordinated changes in three more places and the chapter format is referenced from many test artefacts.
- *Leave the duplicate directory alone.* Rejected: the user explicitly directed cleanup. The duplicate was a consequence of an undocumented path asymmetry, not a deliberate design choice.

**Verification.** A full `python main.py` run reproduces  exactly — same theorem yields, same `[merge_into_shared] +N new entries (shared total: K)` lines, same 35865 verifier-checks-airtight tally — modulo the trailing `Overall runtime` time and per-burst `dt=` jitter.

**Cross-references.**

- [I-16](30_invariants.md#i-16) (`verifier.py` is sacred — failures are real bugs). The verifier edits are pure tightenings: no check is weakened, no new binary becomes consultable; binaries that were never visible to the incubator-side verifier under the duplicate-folder layout stay invisible.
- [I-23](30_invariants.md#i-23) (spontaneous compact operator names stable across batches). Continues to hold for *main* batches; the merge skip explicitly excludes incubator allocations from the shared registry that I-23 governs.
- D-35 / V-6 reformulation fallback (`docs/verifier_rung1_changes.md` § V-6) — the chapter-context filter narrows V-6's fallback scope when the target's anchor is `AnchorIncubator`.

---

<a id="d-53"></a>
## D-53 — `InternalMail` absorbed into `Mail`; mail subsystem unified to a single struct (2026-05-07) — renumbered from main's D-46


**What.** `struct InternalMail` (formerly defined alongside `Mail` in `memory.hpp`) is deleted. `struct Mail`'s `statements` element type migrates from `pair<string, set<int>>` to `pair<ExpressionWithValidity, set<int>>`, reusing the existing `ExpressionWithValidity` type (`memory.hpp::ExpressionWithValidity`, fields `original` and `validityName`) — already the key type of `Mail::exprOriginMap`. `Memory::internalMailIn` becomes type `Mail`. `performElementaryLogicalStep`'s `mailIn` statements absorb (CE-mode and normal-mode loops) gains a per-element `assert(vName == "main")` as the runtime enforcement of [I-26](30_invariants.md#i-26)'s sender contract for the statements channel.

After this, the routing channels (`mailIn`/`mailOut`) and the integration-revival channel (`internalMailIn`) share the same struct. They are distinguished only by lifecycle ([I-21](30_invariants.md#i-21): top-of-burst absorb for revival vs end-of-burst for routing) and by absorb status (`status=1` vs `status=3`) — not by struct type.

**Why.** [D-19](#d-19) created `InternalMail` as a separate struct *to defer* the migration cost of the existing pair-shape callsites of `Mail::statements`. The deferred-migration era ends here — that cost is paid on this branch. Result: one mail struct to reason about, one absorb shape, one mail invariant set. The apparent asymmetry between routing mail (no validity) and revival mail (validity carried) was a definition-site artefact, not a real semantic distinction; the same `Mail` struct can serve both channels because the `validityName` field on the EWV is "main" for routing traffic by sender contract and reflects the actual revival scope for `internalMailIn` traffic.

**Trade-off considered.**

- *Inventing a fresh 3-tuple* `tuple<string, set<int>, string>` for `Mail::statements`. The obvious alternative. Rejected: reusing `ExpressionWithValidity` is cleaner — one type ("expression + scope") used consistently across mail statements and origin maps. Suggested by user during planning ("u need to migrate mail to ExpressionWithValidity").
- *Single atomic commit vs multi-commit refactor.* Multi-commit chosen by user direction for bisectability:
 - C1: `Mail.statements` → `pair<EWV, levels>`; 7 routing senders + 6 receiver iterator blocks migrated atomically (`std::set` element type is concrete, no compilable intermediate state).
 - C2: `Memory.internalMailIn` typed `Mail`; 2 internal senders + drain block migrated.
 - C3: delete `struct InternalMail` (orphan after C2).
 - C4: per-element `assert(vName == "main")` at the routing-channel `mailIn` absorb in `performElementaryLogicalStep`.
 - C6: D-46 entry; D-19 supersession marker; AGENT_SwDD.md decisions-count refresh; final DoD verification.
- *Per-channel asserts at smashMail / sendMail / sender callsites.* User redirected to a single consumption-time assert in `performElementaryLogicalStep` ("main assert is enough in performElementary for external mail"). Captures the same contract violation at one location instead of three.
- *D-numbering.* Original numbering picked D-46 (skipping D-44, D-45 which were reserved for sibling). On2026-05-08 the 46 slot was already taken by incub_fix's cross-pair `equality2` decision; this entry renumbered to D-53.

**Operational consequence.** Receiver-side absorb in `mailIn` no longer hardcodes `"main"`; reads `pair.first.validityName`. For routing traffic this evaluates to `"main"` (sender contract preserved by every routing sender wrapping with `ExpressionWithValidity(expr, "main")`). For `internalMailIn` traffic the EWV carries the actual revival scope. The receiver-side assert traps any future sender that slips past the existing main-only gate at `mailOut.statements.insert`. Behaviour is observationally unchanged for both channels; the assert is purely defensive.

**Verified (on ).** Full `python main.py` clean run on the worktree `.worktree/mail_unification/` from the C6 commit.

- Build clean (Release x64; only known C4267 / C4101 warnings).
- Pipeline runtime 1997 s (≈33 min) — within Gauss-batch baseline tolerance (C1 baseline 1619 s; the variance is from incubator Gauss volatility, not refactor).
- `proved_theorems.txt` — 41 rows; `AnchorGauss` matches 13 (Gauss summation derived per hard-done criterion).
- `verifier.py` — 2750 main checks + 35545 total proof-graph checks, **0 failures** across all 35 tag categories.
- The receiver-side `assert(vName == "main")` at the routing-channel `mailIn` absorb did not fire (Release-build asserts compile out under `NDEBUG`; the gate is informational at runtime, structural at code-review time).

Re-verification on the post-merge tree is recorded in the merge commit body alongside the incub_fix-side D-52 fullest verification (35865 checks airtight, identical to D-52 baseline).

**Files touched (across C1-C4 + the C6 commit).**

- `GL_Quick_VS/GL_Quick/src/memory.hpp` — `Mail.statements` shape; `struct InternalMail` deletion; `Memory::internalMailIn` type.
- `GL_Quick_VS/GL_Quick/src/prover.cpp` — 5 routing senders, 6 receiver iterator blocks (incl. diagnostic dump), `internalMailIn` drain block, `mailIn` absorb assert. `smashMail` set-merge type-driven (no source change).
- `GL_Quick_VS/GL_Quick/src/prover.hpp` — 2 routing senders, 2 internal senders. `sendMail` set-merge type-driven (no source change).
- `docs/02_glossary.md` — `InternalMail` glossary entry rewritten as retired redirect.
- `docs/01_overview.md` — `memory.hpp` file-content table row.
- `docs/20_core_concepts/01_logic_blocks.md` — `Memory`-fields table, `internalMailIn` row.
- `docs/20_core_concepts/03_mail_system.md` — `Mail` struct table; Main-only gate section split per-channel; Scope subsection; InternalMail subsection retired.
- `docs/30_invariants.md` — I-26 receiver bullet, Code section, and Why-section line-number citation refreshed; I-21 code citation refreshed.
- `docs/20_core_concepts/07_or_branching.md` — link refresh in `ordisMerge` description.
- `docs/40_decisions.md` — D-19 marked superseded; link refresh in D-34 body; D-46 entry (renumbered to this D-53).
- `docs/AGENT_SwDD.md` — Decisions row count refreshed.

**Supersedes.** [D-19](#d-19) (deferred-migration era; the cost is now paid). [I-21](30_invariants.md#i-21) and [I-26](30_invariants.md#i-26) wording adjusted in the same C1-C4 series.

---

<a id="d-52"></a>
## D-52 — `_orint_` goal-flow row carries sub-implication, not bare disjunct (2026-05-08)

**What.** Two coordinated changes — one producer-side, one verifier-side — that fix the malformed history line emitted by Case OR in `prepareIntegrationCore2`:

1. **Producer side** — [`prover.hpp::prepareIntegrationCore2`](../GL_Quick_VS/GL_Quick/src/prover.hpp) Case OR (the `if (le.category == "or" && allSigArgsAreU)` block):
 - For each branch `k` of an OR with `K` disjuncts, build the per-branch sub-implication `(>[](AND-of-negated-others)(D_k))`. Premise = left-nested AND of `!D_j` for every `j!= k`; for `K = 2` the AND wrapper collapses to a single negation. Head = chosen disjunct `D_k`. Empty bound-var list — `_orint_` branches don't introduce new quantifiers.
 - Emit one row per branch: `KEY = subImpl_k + "_integration_goal"` at parent `validityName`; `ORIGIN = "expansion for integration" ← cleanSignature + "_integration_goal"` at parent `validityName`. Both sides at parent scope — passes the verifier's same-scope strictness on `expansion for integration`.
 - Drop the previous emission `ev(head, branchValidity)` whose KEY was the bare disjunct at branch namespace. That row was structurally absurd (a single disjunct is not the structural expansion of `or<N>`) and namespace-mismatched (line.namespace deeper than right_ns).
 - Branch-ns scaffolding is unchanged: `encodePush` of the branch payload, `or branch assumption` rows for negated other-disjuncts at branch ns, head-as-toBeProved at branch ns. Only the parent-ns goal-flow row's shape changes.

2. **Verifier side** — [`verifier.py::_try_expand`](../verifier.py) for category `or`:
 - Add a sibling helper `_build_or_subimpls_from_elements(elements)` that constructs all K per-branch sub-implication forms.
 - In `_try_expand`'s `or`-branch acceptance: keep the De Morgan form `_build_or_from_elements` as the primary (existing) acceptance path, and additionally accept LEFT side if it equals (modulo `_normalize_with_unchangeables`) any of the K sub-implication forms. Adds, never weakens — every chapter that passed pre-D-52 still passes.

**Why.** The producer's pre-D-52 goal-flow row was emitted by code copy-pasted from Case A (compact-name → body expansion). The "mirrors Case A" comment at the original site captures the bug: Case A's tag is for compact-implication-name → its body's structural unfolding; Case OR's analog requires unfolding `or<N>` into a structurally-honest expansion. The bare disjunct is one piece of the OR but not the OR's full structural unfolding — the verifier's `_try_expand` rightly rejects it.

The mathematical content of `_orint_` is the constructive proof rule "to prove `(D_1 ∨ D_2 ∨ … ∨ D_K)`, suffices to prove `(!D_1 ∧ !D_2 ∧ … ∧ !D_{k-1} ∧ !D_{k+1} ∧ … ∧ !D_K → D_k)` for some `k`" (any one of K sub-implications closes the OR). The producer was already structurally setting up each branch correctly (negated other-disjuncts seeded as `or branch assumption`, chosen disjunct as branch goal), but the parent-ns history row that should declare "this branch's job is to prove sub-implication k" was malformed. After D-52, the chapter carries one well-shaped `expansion for integration` row per branch, each citing `(or<N>)_integration_goal` as the source — the verifier's K-way acceptance lets all K rows pass, leaving the actual proof obligation to the in-branch derivation of the sub-implication's conclusion.

**Failures dissolved.**
- Failure 2 in chapter `1209_direct_proof.txt` (HTML `chapter1210.html` — off-by-one): the orint-branch's mis-tagged `expansion for integration` row at `(=[i1,v4])` no longer exists as KEY — replaced by the parent-ns sub-implication rows.
- Failure 1 in the same chapter: once `(or2)` is integrated through the new sub-implication path, impl21's modus ponens on `(or2[i1,v4,i0])` at impl26 boundary closes `(in[v4,V1])` at boundary, the chain reaches impl26's body, and `buildStack` walks the body's `_integration_goal` row that was already in `exprOriginMap` (trace line 92, the chained-preorder body of impl26).

**Verification.** Repro with the skip-Gauss-main hack: chapter `1209_direct_proof.txt` parses clean; incubator verifier reports 0 failures across all categories. Yields preserved (IncubatorPeano 513, IncubatorGauss 91, IncubatorGauss1 1, Peano main 49). Pre-D-52 baseline: 32825 checks, 2 FAILED.

**Connection to earlier decisions.** D-52 does not touch the [D-51](#d-51) `buildStack` policy — the path-cycle filter and contradiction-LB fallback remain. The mis-tag was upstream of the walker's selection rule; the walker was faithfully rendering whatever the producer emitted. D-49's cap-full preference is unaffected: the new producer rows compete on the standard `addOrigin` cap-and-tie-break path.

---

<a id="d-51"></a>
## D-51 — Contradiction record stays in `__contradiction__` LB; chapter goal in `buildStack`'s path stack (2026-05-08) — supersedes [D-49](#d-49) / I-35 cap-full preference

**What.** Two coordinated changes that retire the upward propagation of contradiction recipes and let the chapter walker enter the contradiction LB explicitly:

1. **Prover side** — [`prover.cpp::addExprToMemoryBlockKernel`](../GL_Quick_VS/GL_Quick/src/prover.cpp) `primedForContradiction` handler:
 - Stop pushing `(emitter, ev, ("contradiction", deps))` onto `pendingAncestorOrigins`. The drain loop in `proveKernel` previously walked the queued emitter's `parentMemory` chain and wrote the same contradiction record into every ancestor LB's `exprOriginMap`, leaving the inner contradiction's full recipe (its three contradicting deps) duplicated at every ancestor up to root.
 - Drop the parallel write to `memoryBlock.mailOut.exprOriginMap`. Mail transports go parent → children only (`sendMail` iterates `index.find(sender)->second` = children), so an upward-direction `mailOut` write was structurally dead — preserved only because the original drain wrote to ancestors directly.
 - Keep the local write to `memoryBlock.exprOriginMap`. The contradiction record now lives only inside the `__contradiction__` LB that proved it.
2. **Visualizer side** — [`buildStack`](../GL_Quick_VS/GL_Quick/src/visualizer.cpp) (and `directStack` at chapter entry):
 - At `directStack` entry, insert the chapter goal expression (the wrapped theorem from `theoremList`) into the `thread_local g_buildStackPath`. The existing path-cycle filter then rejects any origin candidate whose deps include the chapter goal — exactly the self-application shape the prover's forward-inference produces when it instantiates a proved theorem under its own anchor (origin tag `implication` with deps `[wrapped theorem, anchor]`). Without this, the chapter walker would emit a `theorem`-tag leaf row whose expression matches the chapter goal, tripping `verifier.py::check_chapter` self-reference.
 - When all sorted candidates of a negated head fail (path-cycle filter rejected them all, or no direct origin existed), fall back to walking the LB chain (current → ancestors via `parentMemory`) for a child memory block keyed `"__contradiction__" + positive`. On hit, switch into that contradiction LB — the contradiction record is local there.
 - Remove the per-candidate-loop and `front`-fallback `contradiction`-tag LB switches. Each contradiction is an independent LB with its own chapter; nested contradiction LBs are NOT children of an outer contradiction LB. The only LB switch `buildStack` performs is the chapter-boundary `__contradiction__` fallback above. All other recursion stays in the current LB.

**Why.** The cap-full preference rule introduced in [D-49](#d-49) (foundation displaces convenience: a non-equality origin replaces an equality1/equality2 slot at cap) was the immediate response to [chapter-100/101 cycles](#d-49) on the branch but rested on three more-fragile assumptions:

1. The `pendingAncestorOrigins` upward write was conjoined with — but not justified by — D-49. The two fixes shipped together; D-49's commit body explained the cap-full preference but did not explain *why ancestors needed a copy of the contradiction recipe at all*.
2. With `max_origin_per_expr = 1`, a single origin survived per expression and D-49's tie-break determined which one. The chapter walker's `front` then read whatever D-49 left.
3. The walker's `covered` set spanned every LB it walked into, so the same expression could not be re-emitted as a different proof step in a different LB context.

Raising `max_origin_per_expr` to 30 (matching compressor mode) to support multi-origin selection broke (2) — `front` no longer returned D-49's surviving choice, since D-49 only kicks in at cap-full and cap=30 is rarely full. The walker started picking insertion-order origins which differed from D-49's preferred origins. Symptom: chapter-37-style nested-contradiction chapters lost their inner `task formulation` row because the outer LB's `(in2[i3,i5,s])` was now derived via `implication` (forward-inference of the inner wrapped theorem) instead of being task-formulated, and the inner LB's task-formulation row was suppressed by `covered`.

The architectural fix (this entry) replaces the cap-full preference with explicit selection at the chapter walker:

- The chapter goal expression goes into `g_buildStackPath` so the walker rejects self-applying origins by the same path-cycle mechanism that handles run-of-the-mill cycles.
- Negated heads with no acyclic direct origin in the LB before the head fall through to a `__contradiction__` LB lookup, which is the principled entry point — the contradiction record now lives only there.
- The per-candidate special-case `contradiction`-tag switch in the recursion loop is gone. With the recipe local, walking the contradiction's deps in the same LB is correct: derivations of the contradicting pair sit there, and any nested contradiction reached via a negated dep falls into the same chapter-boundary fallback the next level down (into a sibling `__contradiction__` LB attached to the anchor).

**What was investigated and ruled out.**

- **Knuth (1977) shortest-derivation greedy origin selection** (foundation-distance Bellman-Ford). Tried first as a more powerful selection criterion. Mass self-reference: every chapter goal had a `(theorem, [])` origin at level 0, which Knuth picked over real derivations. Treating `theorem` and `task formulation` as foundation-tag-equivalents amplified the problem. Rejected — the problem isn't selection criterion, it's the *records present at AnchorIncubator level being wrong*.
- **Greedy DFS with path-stack cycle filter, no backtracking** (option-1 greedy). Rejected — fails on ordering-sensitive subtrees where the first valid candidate locally traps an ancestor; needs backtracking.
- **Greedy DFS + backtracking, but keep ancestor-side contradiction record.** Tested: 28804 / 1078 FAILED (539 contradiction + 539 contradiction trace, all nested-contradiction missing-task-formulation). The `covered` set spanning the ancestor LB and the `__contradiction__` LB blocks the inner LB's task-formulation re-emission of an expression already visited via implication-derivation in the outer LB. Rejected — the LB-switch's `covered` set semantics are at fault, *and* leaving the contradiction record at ancestor level is the structural violation that produced the conflicting paths in the first place.
- **Force-emit the inner cleanOp's task-formulation row at the contradiction-LB switch, plus parent-chain walk for nested cases.** Tested: 37622 / 779 FAILED, mostly `task formulation` (inner cleanOps don't satisfy `verifier.py::check_task_formulation`'s "premise of chapter theorem or chapter cleanOp" requirement). Rejected — produces structurally invalid task-formulation rows.

**Trade-off.** The `pendingAncestorOrigins` queue, mutex, and drain loop in `proveKernel` are now dead code — the queue is never pushed and the drain iterates an empty container. Cleanup deferred (next session). No functional cost since the empty drain is O(0).

**Verified.** Incube-only Peano (RUN_INCUBATOR=True, RUN_MAIN_PATH=False, tags=["Peano"]) under HEAD: yield 513, verifier 28804 checks, **0 failures — airtight**. Compared to:

| Configuration | Yield | Verifier (incube-only Peano) |
|---|---:|---|
| pre-D-51 baseline (cap=1, original buildStack) | 513 | 28793 / 2 FAILED (chapter-193 cycle) |
| D-51 mid (cap=30 + option-1 + backtracking, ancestor-side contradiction record kept) | 513 | 28804 / 1078 FAILED (nested-contradiction task-formulation gap) |
| **D-51 final (this entry)** | 513 | **28804 / 0 FAILED — airtight** |

Fullest pipeline (RUN_INCUBATOR=True, RUN_MAIN_PATH=True, tags=["Peano", "Gauss"]) verification pending in this commit window.

**Files touched.**

- `GL_Quick_VS/GL_Quick/src/prover.cpp`: `pendingAncestorOrigins` push retired; `mailOut` upward-write removed.
- `GL_Quick_VS/GL_Quick/src/visualizer.cpp`: chapter goal inserted into `g_buildStackPath` at `directStack`; per-candidate-loop and front-fallback `contradiction`-tag LB switches removed; new `__contradiction__` fallback in the all-candidates-failed branch.
- `docs/30_invariants.md`: I-35 (cap-full preference) marked superseded.
- `docs/40_decisions.md`: this entry; D-49 / I-35 status amendment.
- `docs/AGENT_SwDD.md`: invariant quick-reference table updated.
- `docs/10_pipeline/04_prover.md`: addStatement / contradiction LB section updated.
- `docs/10_pipeline/06_process_proof_graph.md`: buildStack section updated.

**What does NOT change.**

- D-50 (`addStatement` `&& local` gate) — preserved.
- `max_origin_per_expr = 30` across all configs — preserved (multi-origin storage is what makes the chapter walker have alternatives to choose from when the cycle filter rejects the first candidate).
- Verifier — untouched per [I-16](30_invariants.md#i-16). Reaching 0 failures without verifier modification confirms the fix is on the producer side.
- D-49 / I-35 the *invariant*: superseded but kept in the doc with a status amendment for traceability. The cap-full preference logic still exists in `addOrigin`; it is just rarely triggered (cap=30 is rarely full) and no longer load-bearing for any chapter shape. Cleanup deferred.

---

<a id="d-50"></a>
## D-50 — `addStatement` equality-mirror push gated on `local` (2026-05-07) — restores 366-theorem incubator-Peano yield lost to [](../) `&& false` band-aid

**What.** The `isEquality(expr)` branch in [`addStatement`](../GL_Quick_VS/GL_Quick/src/prover.hpp) — which pushes the mirrored equality `(=[args[1],args[0]])` onto the function's `newStatements` return vector — has its gate changed from `&& false` (dead-coded since [](../)) to `&& local`. The `local` parameter is the existing `bool local` parameter of `addStatement`; the kernel's call site (`addExprToMemoryBlockKernel` in [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)) passes `isLocal = (status == 0 || status == 1)` into it. So mirror-push fires when an equality enters via local addition (status 0/1), and is skipped when it enters via mail-in absorb (status=3).

**Why.** Bisection traced incubator-Peano yield drop 513 → 147 to commit [](../) "addStatement: disable unconditional equality-mirror push (`&& false`)". Producer-LB hashburst trace (chapter-237 producer, LB chain `[0] (in3[2,2,15,4]) → [1] (AnchorIncubator[…]) → [2] <root>`) localized the regression to a single side-effect:

- At burst #2 of this LB, both runs (with-`&& false` vs without) have byte-identical state including `toBeProved` of size 9.
- Between burst #2 and burst #3:
 - Without `&& false`: `addEquality(allowSymmetry=true)` registers `(=[2,15])` and mirror `(=[15,2])` in `encodedStatements` + `statementLevelsMap`. `addStatement(expr=(=[2,15]), local=true)` then pushes the mirror `(=[15,2])` onto `newStatements`. The post-`addStatement` loop in `addExprToMemoryBlockKernel` iterates `newStatements`, calls `toBeProved.find((=[15,2]))`, finds the open goal, and erases it. `toBeProved` shrinks to 8.
 - With `&& false`: `addEquality` adds the same entries to `encodedStatements`, but `addStatement`'s mirror push is gated off. `newStatements` does not contain `(=[15,2])`. The discharge loop iterates only over `(=[2,15])` (which is not in `toBeProved`), never matches `(=[15,2])`, and the goal stays open. `toBeProved` remains at 9.

The discharge loop iterates `newStatements`, **not** `encodedStatements`. The compensation chain ([](../) `allowSymmetry`, [](../) `updateEquivalenceClasses` `#if 0`, [](../) kernel `isLocal` gating) covered the `encodedStatements`-side mirror registration but did not cover this discharge side-effect. Result: the producer LB never closed; its wrapped implication never broadcast to AnchorIncubator's `mailIn`; AnchorIncubator stalled (`stmts=508` plateau from burst #8 onwards). Cascade across all 9 numeric atoms × 2 forms (right-identity-of-`+` and its uniqueness mirror) plus their downstream consumers cost 366 incubator-Peano theorems (513 → 147, ≈ 71% yield loss).

The original [](../) commit body is candid: it silenced an assert at `prover.cpp::addExprToMemoryBlockKernel` (the `statementLevelsMap.find` lookup that follows the mirror push in the discharge loop) which fired on **mail-in absorb** (status=3) paths, where `addEquality(allowSymmetry=false)` had skipped registering the mirror. The author closed with: *"Awaiting direction on whether to remove the dead block entirely or keep `&& false` as a documented switch."* — i.e. flagged as unfinished. `&& local` is the correct narrow gate: status 0/1 takes the mirror push (and its discharge side-effect) because `addEquality(allowSymmetry=true)` has registered the mirror; status 3 skips the push because no mirror entry exists, avoiding the assert.

**What was investigated and ruled out.**

- **Removing the dead block entirely.** Rejected — drops the `toBeProved` discharge side-effect on local paths just as fully as `&& false` did. Yield loss persists.
- **Routing the discharge through `encodedStatements` instead of `newStatements`.** Considered. Would decouple the discharge from `addStatement`'s return value, but the discharge loop in `addExprToMemoryBlockKernel` is heavily coupled to `newStatements` (admission map updates, levels lookup, scope handling all use the per-entry data). Refactor scope is large; the local-gate is a one-line change with the same outcome.
- **Triggering discharge inside `addEquality` directly (when it pushes the mirror to `encodedStatements`).** Considered. Would re-implement the discharge logic at the registration site. Risks duplication / divergence with the `addStatement` post-loop. Defer pending need.

**Trade-off.** Mail-in absorb (status=3) paths still skip the `toBeProved` discharge for incoming mirrored equalities — same as before D-50. If a mail-in absorb arrives carrying an equality whose mirror is already an open `toBeProved` goal in the receiving LB, the goal will not be discharged on absorb. Whether this scenario occurs in practice and matters is open; the chapter-237 case studied here is fully a local-derivation scenario, so the gate suffices. If a follow-up trace surfaces a mail-in absorb scenario, the fix will be a parallel discharge inside the absorb path, not a removal of the `local` gate.

**Verified.** Incubator-only Peano via `main.py` (RUN_INCUBATOR=True, RUN_MAIN_PATH=False, tags=["Peano"]) at HEAD with `&& local`: `Number proven theorems: 513` (matches pre- baseline), Verifier `28793 checks, 2 FAILED` (same 2 `origin chain termination` failures as the unconditional no-false test — pre-existing and not introduced by this gate; tracked separately). Snapshot .

**Files touched.**

- `GL_Quick_VS/GL_Quick/src/prover.hpp`: `addStatement` `isEquality` branch gate `&& false` → `&& local`.
- `docs/40_decisions.md`: this entry.

**What does NOT change.**

- `addStatement` signature / parameter list — `bool local` already existed long before [](../); no new parameter added.
- `addEquality`'s `allowSymmetry` parameter — unchanged. Continues to gate `encodedStatements` + `statementLevelsMap` mirror registration on the kernel-supplied `isLocal`.
- The discharge loop in `addExprToMemoryBlockKernel` — unchanged. Still iterates `newStatements`, still calls `toBeProved.find` per entry, still erases on hit.
- The 2 pre-existing `origin chain termination` failures in the no-incubator-only-Peano run — not regressed, not addressed; their root cause is a separate investigation.
- Verifier — untouched per [I-16](30_invariants.md#i-16).

---

<a id="d-49"></a>
## D-49 — `addOrigin` cap-full preference: foundation displaces convenience (2026-05-07) — superseded by [D-51](#d-51), 2026-05-08

> **Status amendment (2026-05-08).** Superseded by [D-51](#d-51). The cap-full preference rule was the immediate fix for the chapter-100/101 swap-cycle and remains in `addOrigin` as inert code at HEAD (cap=30 is rarely full, so the replacement branch is rarely taken; behavior depends on `front` of the stored origins, not on the cap-full preference). The structural fix is D-51's combination: contradiction record stays in the `__contradiction__` LB only (no `pendingAncestorOrigins` upward write), and `buildStack` enters the contradiction LB explicitly via the chapter-boundary `__contradiction__` simpleMap fallback, with the chapter goal inserted into the path stack so the existing cycle filter rejects self-applying origins. Cleanup of the now-inert cap-full preference code in `addOrigin` is deferred.


**What.** `addOrigin` ([`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp), `addOrigin`) gains a cap-full preference replacement step. When the per-key origin vector is at `maxOrigins` and a new origin arrives whose tag is **not** `equality1` and not `equality2`, the helper scans the vector for the first `equality1`/`equality2` slot and replaces it with the new origin. Below-cap behavior unchanged (append + dedup + D-44 symmetry-source trap intact). New invariant: [I-35](30_invariants.md#i-35).

Coupled change at [`prover.cpp::performElementaryLogicalStep`](../GL_Quick_VS/GL_Quick/src/prover.cpp): the bulk-merge step `expr_origin_map = mail_in | current` is rewritten to route through `addOrigin`. Pre-D-49 it was a raw `std::map` swap with body-wins-on-conflict. The raw merge bypassed both the cap and the new preference; mail-arrived origin vectors landed verbatim, and existing body entries overrode mail entries on conflict by insertion-order semantics. Routing through `addOrigin` puts every arriving origin through the same cap+preference gate.

**Why.** [`verifier.py::check_origin_chain_termination`](../verifier.py) flagged 4 cyclic rows on the iter-35 verification run (Peano + Gauss, no incubator):

- `files/processed_proof_graph/100_check_zero.txt` rows 56–57 — `(in2[i0,v10,id]) ↔ (in2[v10,i0,id])`.
- `files/processed_proof_graph/101_check_induction_condition.txt` rows 72–73 — analogous shape.

Both belong to theorem 96 (the Gauss `fold` induction in `proved_theorems.txt:2`). The new `mailIn.exprOriginMap` dump section revealed the actual mechanism: at burst 1 of the chapter-100 zero-case LB, `mailIn.exprOriginMap` carried **two origins per cycle member**:

```
(in2[2,it_0_lev_0_32,8])
    <- equality1 | (in2[it_0_lev_0_32,2,8])  (=[2,it_0_lev_0_32])  (=[it_0_lev_0_32,2])
    <- implication | (>[1](in[1,u_1])(>[2](=[1,2])(in2[1,2,u_8])))
                   | (=[2,it_0_lev_0_32])  (in[2,1])

(in2[it_0_lev_0_32,2,8])
    <- implication | (...) ...
    <- equality1 | (in2[2,it_0_lev_0_32,8])  ...
```

Both expressions had a foundational `implication` origin (the rule "if `1 ∈ N` and `1 = 2` then `in2[1,2,id]`") **and** a cyclic `equality1` origin pointing at the other. With `max_origin_per_expr = 1` the bulk-merge from `mailIn.exprOriginMap` to `body.exprOriginMap` had to keep just one origin per key. Because the merge was raw `std::map` swap with body-wins-on-conflict and `addOrigin` was uninvolved, the choice was governed by mailIn's vector ordering, which silently picked the cyclic `equality1` origin and dropped the foundational `implication` one. Once installed in `body.exprOriginMap`, every downstream consumer (chapter projection, verifier chain walk) saw only the cyclic origin.

The fix has two coupled halves:

1. **Mechanism (`addOrigin`).** When at cap and a non-equality origin arrives, replace any existing `equality1`/`equality2` slot. The rule is categorical: foundation displaces convenience.
2. **Routing (bulk-merge).** Replace the raw `std::map` swap with a per-origin loop that calls `addOrigin`. Every arriving mail origin goes through the same gate as direct producer-side emissions, so the preference applies uniformly across producer-side and bulk-merge code paths.

Together these resolve the chapter-100/101 swap-cycle: at the bulk-merge, the foundational `implication` origin (when present) replaces any pre-existing `equality1`/`equality2` slot in `body.exprOriginMap`, and the chain walk reaches a base.

**What was investigated and ruled out.**

- **Bumping `max_origin_per_expr` from 1 to 2+ for Gauss.** Rejected: 1 is the operating contract; the bug was that the choice mechanism wasn't smart enough. Bumping the cap would defer the choice to the verifier's compressor / projection step and shift the failure mode rather than fixing it.
- **Capping `smashMail`'s mailIn aggregation per `max_origin_per_expr` (commit, reverted).** Rejected after user feedback. Capping at smashMail does the same thing as the existing cap inside `addOrigin` during bulk-merge, just one step earlier. It does not fix the choice — it only changes which origin happens to be first in mailIn. The structural issue is "how to choose", not "where the cap is enforced".
- **Producer-side gate inside `applyEquivalenceClass` only.** Insufficient on its own. The gate guards local emissions but does not affect mail-imported origins. Kept as a producer-side redundancy guard ([I-34](30_invariants.md#i-34)); D-49 is the system-level fix.
- **Per-tag priority table.** Rejected: the binary categorical distinction (equality-convenience vs. anything else) is the structurally correct one. Adding `implication > recursion > theorem > expansion >...` couples the helper to producer-side tag semantics and yields no extra cycle suppression.
- **Restricting preference to non-compressor mode only.** Rejected: in compressor mode the cap is 30, so cap-full is rare; but when it occurs the same preference applies. Uniform behavior is simpler and equally sound.

**Trade-off.** `addOrigin` becomes slightly less ordering-symmetric: a non-equality origin arriving after an equality1/equality2 slot displaces the equality slot, while previously it would have been silently dropped. Producer-side determinism is unchanged (same ordering, same keys, but different surviving origin per key when both kinds of tag are present). One previously-dropped foundational origin per cycle now wins; the cyclic origin disappears. Theorem count and verifier coverage expected unchanged on Peano / Gauss baselines that were already cycle-free; the chapter-100/101 cycle disappears.

**Verified.** Pending Peano + Gauss `python main.py` (no incubator, iter cap 35) re-run after this commit. Acceptance: `origin chain termination` failures drop to 0; Gauss `Number proven theorems` ≥ 11.

**Files touched.**

- `GL_Quick_VS/GL_Quick/src/prover.hpp`: `addOrigin` cap-full preference replacement step.
- `GL_Quick_VS/GL_Quick/src/prover.cpp`: `performElementaryLogicalStep` bulk-merge from `body.mailIn.exprOriginMap` to `body.exprOriginMap` rewritten to route through `addOrigin`.
- `docs/30_invariants.md`: I-35 added.
- `docs/AGENT_SwDD.md`: invariant quick-reference table updated (I-35).
- `docs/40_decisions.md`: this entry.

**What does NOT change.**

- Below-cap append behavior in `addOrigin` — unchanged. Dedup logic, D-44 symmetry-source trap, vector ordering all preserved.
- `smashMail` aggregation into `body.mailIn.exprOriginMap` — unchanged. Stays uncapped (compressor-mode multi-origin support); preference applies downstream at bulk-merge.
- Producer-side cap enforcement — unchanged. Each producer's local `addOrigin` call already respects `max_origin_per_expr`; D-49's preference adds replacement on top of that.
- D-46 cross-pair `equality2` gate — unchanged. Composes with D-49.
- D-48 cross-substitution `equality1` gate — unchanged. Defensible producer-side redundancy guard; D-49 is the system-level cycle resolution.
- Verifier — untouched per [I-16](30_invariants.md#i-16).

---

<a id="d-48"></a>
## D-48 — Cross-substitution `equality1` emission gated on existing LB origin (2026-05-07) — superseded as cycle-fix by [D-49](#d-49); kept as producer-side redundancy guard

> **Status amendment (2026-05-07, post-iter-35 verification).** D-48's gate is real and lands in the codebase, but the chapter-96/97 cycle it was *intended to close* re-emerged in a different shape (`(in2[i0,v10,id]) ↔ (in2[v10,i0,id])`, the swap-cycle, in chapters 100/101 of the same theorem) once the iter cap was raised from 29 to 35 and Gauss reached its full theorem count. The system-level cycle-resolution mechanism is **[D-49](#d-49) / [I-35](30_invariants.md#i-35)** — `addOrigin`'s cap-full preference replacement plus the bulk-merge routing change. D-48 stays in the codebase as a producer-side redundancy guard ([I-34](30_invariants.md#i-34)) but is not the load-bearing fix. The original "What / Why / Verified" text below describes D-48's intent at write-time; it is preserved verbatim for traceability.

**What.** `applyEquivalenceClass` ([`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp), `applyEquivalenceClass`) — the `if (parameters.trackHistory)` block that emits the `equality1` origin record for a class-rewritten expression — now skips the two `addOrigin` calls (`memoryBlock.exprOriginMap` and `memoryBlock.mailOut.exprOriginMap`) when the target `applied @ depositValidity` already carries any origin entry in `memoryBlock.exprOriginMap`. New invariant: [I-34](30_invariants.md#i-34). The `exprOriginMapLocal` populate at the rewrite-enumeration site stays unconditional (required for the FIRST emission's well-formedness — `check_equality1` rejects `len(rest) < 4`). A one-shot discovery trap inside `performElementaryLogicalStep` ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)) dumps the parent chain of every recursion-equality zero-case LB to , supporting follow-up debug rounds if the gate proves insufficient.

**Why.** [`verifier.py::check_origin_chain_termination`](../verifier.py) flagged 4 cyclic rows on the branch (Peano + Gauss, no incubator):

- `files/processed_proof_graph/96_check_zero.txt` rows 56–57 — self-cycle on `(in2[i0,v10,id]) ↔ (in2[i0,i0,id])`.
- `files/processed_proof_graph/97_check_induction_condition.txt` rows 72–73 — self-cycle on `(in2[i0,v6,id]) ↔ (in2[v6,i0,id])`.

Both chapters are sub-blocks of theorem 96 (`proved_theorems.txt:2`, the Gauss `fold` induction theorem):

```
(>[1,2,3,4,5,7,8](AnchorGauss[1,2,3,4,5,6,7,8])
  (>[12,9](fold[1,3,4,8,2,9,12])
    (>[10](in2[9,10,3])
      (>[11](in3[7,12,11,5])
        (in3[9,10,11,5])))))
```

Cycle pattern (chapter 96):

```
56: (in2[i0,v10,id]) ← equality1 | (in2[i0,i0,id])  (=[i0,v10])
57: (in2[i0,i0,id])  ← equality1 | (in2[i0,v10,id]) (=[v10,i0])
```

Each row's only origin points at the other; the DFS loops indefinitely. Same shape as the chapter-22 / theorem-12 cycle that [I-32](30_invariants.md#i-32) / [D-46](#d-46) closed on the `equality2` cross-pair path — but here on the `equality1` substitution path. `applyEquivalenceClass` had a previously-commented-out guard `if (memoryBlock.exprOriginMap.find(appliedWithValidity) == memoryBlock.exprOriginMap.end)` at the addOrigin site; the comment said it was disabled to allow multi-origin accumulation (fixing two unrelated rt_conjecturer-reshuffle equality1 failures). The fix keeps the rationale intact (populate `exprOriginMapLocal` unconditionally so any first emission is well-formed) but reinstates the emission gate against existing origins to break the back-direction-cycle vector.

The fix is one-sided: gate only the emission site in `applyEquivalenceClass`. Site 2 (`applyEquivalenceClassToNegatedEquality`) has its own `statementLevelsMap` early-exit that prevents the same cycle shape on negated equalities. Site 3 (`emitIntegrationRevivalToInternalMailIn`) emits to `internalMailIn`; mail absorb at the next hashburst would invoke site 1's gate downstream. Both sites 2 and 3 are documented in [I-34](30_invariants.md#i-34) as future-extension candidates if a different cycle shape ever surfaces.

**What was investigated and ruled out.**

- **Producer-side mail-origin sync** (mirroring D-46's coupled producer-side change in `performElementaryLogicalStep`). Rejected: D-46's producer-side sync exists because the equality2 cross-pair logic consults `class.equalityOriginMap`, which the bulk-merge alone doesn't update. The equality1 gate only consults `body.exprOriginMap`, which the bulk-merge already populates from `body.mailIn.exprOriginMap` at the top of every hashburst. No producer-side change needed.
- **Loose form of the gate** (skip only when target has a *non-equality1* origin entry, allowing multiple equality1 origins from different classes). Rejected for parity with [I-32](30_invariants.md#i-32) and to be conservative — if a legitimate proof needs multiple equality1 records on the same target, the user can relax the predicate after measurement. The strong form closes the chapter-96/97 cycle without false-suppressing legitimate first emissions.
- **Gating sites 2 and 3** (`applyEquivalenceClassToNegatedEquality`, `emitIntegrationRevivalToInternalMailIn`). Deferred. Site 2's `statementLevelsMap` early-exit at `emitNew` is already a stronger guard than the proposed gate (it skips ALL deposit, not just origin). Site 3 emits to `internalMailIn`; redundant origins there get filtered by site 1 once mail flows back through `applyEquivalenceClass` on subsequent iterations. Adding gates at sites 2/3 would be defense-in-depth but is out of DoD scope for this fix.
- **Removing the discovery trap immediately**. Rejected per — the trap stays through verification; removed only after the fix is confirmed (Stage 2 strict chain match would replace it if needed).

**Trade-off.** Equality1 origin records become single-origin-per-target instead of multi-origin-accumulated. Verifier-visible chapter rows: equality1 substitutions for already-known targets disappear. The previously-cited "two equality1 failures from rt_conjecturer reshuffle" are NOT regressed because the populate step (`exprOriginMapLocal[r.rewrittenExpr] = eqs`) stays unconditional — only the emission step is gated. The chapter-96/97 cycle disappears; foundation-only chapters where each target has a single class-substitution origin are unaffected.

**Verified.** Two run sequences:

1. **Iter cap 29 (Peano + Gauss, no incubator).** `origin chain termination`: 0 failures (passes the DoD on this metric). But Gauss `Number proven theorems`: 9 vs. baseline 11 — the iter cap stopped 2 theorems before the cycle-prone proof paths fired, masking the issue.
2. **Iter cap 35 (Peano + Gauss, no incubator).** Gauss recovers to 11 proved. `origin chain termination`: 4 failures resurface in `100_check_zero.txt` rows 56–57 and `101_check_induction_condition.txt` rows 72–73 — the same theorem 96 cycle in a swap-form `(in2[i0,v10,id]) ↔ (in2[v10,i0,id])`. D-48's gate does not cover this shape because both targets are syntactically distinct keys both first-emitted; each target's `body.exprOriginMap` entry is empty when the gate fires, so emission proceeds.

The shape-B finding triggered [D-49](#d-49). Final acceptance verified there.

**Files touched.**

- `GL_Quick_VS/GL_Quick/src/prover.hpp`: `applyEquivalenceClass` — `alreadyHasOrigin` lambda + gated `addOrigin` calls inside the `trackHistory` block.
- `GL_Quick_VS/GL_Quick/src/prover.cpp`: `performElementaryLogicalStep` — Phase A discovery trap dumping recursion-equality zero-case LB chains to .
- `docs/20_core_concepts/05_equivalence_classes.md`: new *Cross-substitution `equality1` emission gating* subsection.
- `docs/30_invariants.md`: I-34 added.
- `docs/AGENT_SwDD.md`: invariant quick-reference table updated (I-34).
- `docs/50_gotchas.md`: G-39 added.
- `docs/40_decisions.md`: this entry.

**What does NOT change.**

- Statement deposit (`memoryBlock.statementLevelsMap`, `encodedStatements`, `localEncodedStatements*`, `newStatements`, mailOut statement) — unchanged. The gate only suppresses the origin emission; the rewritten statement still flows through `newStatements` and admission. [I-25](30_invariants.md#i-25) preserved.
- The `exprOriginMapLocal` populate inside the `enumerateEqClassRewrites` callback — unchanged. Required for first-emission well-formedness.
- `applyEquivalenceClassToNegatedEquality` and `emitIntegrationRevivalToInternalMailIn` — unchanged. Documented in [I-34](30_invariants.md#i-34) as deferred-extension sites.
- D-46 cross-pair `equality2` gate — unchanged. The two gates compose: equality2 cross-pair gate inside `mergeTwoEquivalenceClasses` + equality1 substitution gate inside `applyEquivalenceClass`.
- D-43 `applyEquivalenceClassToRejectedMapIntegration` additive contract — unchanged. The rmi side does not emit `equality1` origin records (it routes through `emitIntegrationRevivalToInternalMailIn` → `internalMailIn`).
- Verifier — untouched per [I-16](30_invariants.md#i-16).

---

<a id="d-45"></a>
## D-45 — `addOrigin` symmetry-source assert disabled; absorption fallbacks hardened; dead `proved` parameter removed (2026-05-07)

**What.** Three coordinated changes, all under the D-45 tag:

1. **`addOrigin` symmetry-source assert disabled.** [`prover.hpp::addOrigin`](../GL_Quick_VS/GL_Quick/src/prover.hpp) — the D-44 trap that fires when adding a `symmetry of equality` / `symmetry of inequality` origin whose source has no entry in the same map keeps its diagnostic dump to  but no longer asserts. The original assert was set for a different issue (cycle-class bugs caught by the chapter cycle-detection verifier check) and conflated the real bug (origin lines generated with no history at all) with the legitimate cross-map asymmetry between `body.exprOriginMap` (fed by the mailIn bulk-merge in `performElementaryLogicalStep`) and `body.mailOut.exprOriginMap` (local-delta only). Diagnostic value preserved; abort path removed.

2. **Absorption fallbacks hardened.** [`prover.cpp::performElementaryLogicalStep`](../GL_Quick_VS/GL_Quick/src/prover.cpp), the `internalMailIn`-absorb and `mailIn.statements`-absorb blocks. Pre-D-45 these had a defensive empty-origin fallback (silently used an empty origin pair when the matching `exprOriginMap` entry was missing). Post-D-45 they assert `it!= exprOriginMap.end && !it->second.empty`. Rationale: the silent fallback masked a sender-side bug where statements were mailed without their paired origin records. Hard-asserting surfaces the producer-side gap immediately.

3. **Dead `proved` parameter removed from `addTheoremToMemory`.** [`prover.cpp::addTheoremToMemory`](../GL_Quick_VS/GL_Quick/src/prover.cpp). The single caller (inside the conjecture-batch loop) always passed `false`; the body's `proved == true` branch was already dead. Parameter dropped; signature simplified.

**Why.** Cleanup pass following the chapter-22 / theorem-12 cycle investigation. The D-44 trap had served its purpose (revealed the `mergeTwoEquivalenceClasses` cross-pair cycle generator that [D-46](#d-46) closed), and its abort path was now spuriously firing on the legitimate map-asymmetry case. The absorption-fallback hardening turns a class of "silent empty-origin" bugs into immediate aborts, narrowing the search space when future origin-chain anomalies surface. The `proved` parameter cleanup is pure dead-code removal.

**What was investigated and ruled out.**

- **Removing the trap dump entirely** (alongside the assert). Rejected: the dump-to-file path is cheap and continues to be useful for diagnosing future cross-map asymmetry cases.
- **Tightening the trap to fire only when both maps are missing the source.** Considered but skipped — the dump now logs both maps' state for the trapped key + sources, which lets the future agent reason about asymmetry without a tighter predicate.
- **Replacing the absorption assert with a soft warning + skip.** Rejected: a missing origin in `mailIn.exprOriginMap` for a statement that *did* mail through means a sender-side bug. Soft-skip would mask the bug; hard-assert surfaces it with full context.

**Trade-off.** Aborts on the absorption hard-assert path replace silent skips. Net: one class of bugs surfaces immediately instead of propagating into chapter shape; in exchange, any pre-existing latent gap on the mail/origin pairing aborts the prover at first mail-arriving statement.

**Verified.** Inline with the D-44 / D-46 / D-47 work-stream verification. No standalone re-run for this commit.

**Files touched.**

- `GL_Quick_VS/GL_Quick/src/prover.hpp` — `addOrigin` D-44 trap: assert disabled, dump retained.
- `GL_Quick_VS/GL_Quick/src/prover.cpp` — `performElementaryLogicalStep` absorption blocks: hard asserts; `addTheoremToMemory` dead `proved` parameter dropped.
- (No SwDD docs touched at original commit time. This entry added 2026-05-07 by audit-fix to close the gap.)

**What does NOT change.**

- The trap's diagnostic dump format and target file () — unchanged.
- Mailing protocol semantics (which origins paired with which statements) — unchanged. D-45 only hardens the receiver-side check.
- D-44's ancestor-pass merge contract — unchanged.

---

<a id="d-47"></a>
## D-47 — `mergeTwoEquivalenceClasses` cross-vN preconditions: ancestor-only direction + eqArgs-subset assert is same-vN only (2026-05-07)

**What.** `mergeTwoEquivalenceClasses` ([`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp), `mergeTwoEquivalenceClasses`) gains a `classBValidityName` parameter. Three derived rules — see [I-33](30_invariants.md#i-33):

1. Cross-vN: `classBValidityName` must be a strict ancestor of `validityName` (`memoryBlock.nameMap.stringAncestorsOf[validityName]`). Asserted at function entry.
2. Cross-vN early-exit: if `classB.variables ⊆ classA.variables`, return immediately. Descendant `mergedClass` already covers every variable the ancestor class would contribute.
3. The existing `assert(!isSubsetOf(eqArgs, classB.variables))` and `assert(tmp.size == 1)` are gated on `sameVN`. Cross-vN allows both — the multi-bridge case picks `commonArg = *tmp.begin` deterministically.

Both call sites in `updateEquivalenceClasses` updated:
- Same-vN pass: pass `validityName, validityName`.
- Ancestor pass (D-44): pass `validityName, ancestorV`.

**Why.** [D-44](#d-44) added the ancestor-scope merge pass to `updateEquivalenceClasses`. The original `mergeTwoEquivalenceClasses` was written under same-vN assumptions: if any same-vN class held both `eqArgs`, it would be iterated first by the sequential overlap loop and absorbed via the subset path before `mergedClass` could grow past `eqArgs`. The `eqArgs ⊄ classB` and `tmp.size == 1` asserts encoded that invariant. Ancestor classes break it — they may independently contain both `eqArgs` (e.g. the equality was already admitted at the ancestor scope via mail or prior derivation), and the descendant's iteration order can't influence the ancestor's class structure.

The Gauss main batch (commit, after the D-46 squash) hit the assertion at Hash burst 2 and aborted via `0xC0000409`. The fix preserves the same-vN contract (assertions stay enforced for legacy merges) while permitting the cross-vN call legitimately added by D-44.

The early-exit on `classB ⊆ classA` (rule 2) is a separate optimisation: it avoids the merged-pair logic when the ancestor's contribution is already represented in the descendant. Without it, the merged-pair would emit cross-pair records for already-known equalities; [I-32](30_invariants.md#i-32) would suppress them downstream, but stopping early is cheaper and clearer.

**What was investigated and ruled out.**

- **Removing the assertions entirely.** Rejected — the same-vN contract is real and a violation indicates a `updateEquivalenceClasses` iteration-order regression. Keep the asserts; gate them.
- **Adding a single `bool isCrossScope` parameter.** Considered. Rejected because the ancestor-direction validity check needs the actual ancestor scope name, not just a boolean. Carrying the full `classBValidityName` is cleaner and supports the validation assert.
- **Multi-bridge cross-pair emission (record cross-pairs through both bridges when `tmp.size == 2`).** Rejected — would produce parallel origin records that [I-32](30_invariants.md#i-32) suppresses anyway. Single deterministic bridge is sufficient.
- **Folding classB.equalityOriginMap into classA on cross-vN early-exit.** Rejected — the ancestor class stays at its scope per [I-31](30_invariants.md#i-31); the descendant's class doesn't need a copy of the ancestor's origin records to function. The buildStack walker resolves cross-scope dependencies via the ancestor's exprOriginMap directly.

**Trade-off.** One more parameter on the merge helper; one more invariant to remember. In exchange: cross-vN merges are now safely admitted without false-positive aborts on Gauss / FTA-ladder runs that exercise the ancestor pass at scale.

**Verified.** Pending Peano + Gauss `python main.py` re-run after this commit. Acceptance: prior 0xC0000409 abort gone; verifier reports clean; no theorem regression vs. baseline.

**Files touched.**

- `GL_Quick_VS/GL_Quick/src/prover.hpp`: `mergeTwoEquivalenceClasses` signature + body; both call sites in `updateEquivalenceClasses`.
- `docs/30_invariants.md`: I-33 added.
- `docs/AGENT_SwDD.md`: invariant quick-reference table updated (I-33).
- `docs/40_decisions.md`: this entry.

**What does NOT change.**

- D-44's ancestor-scope merge contract — preserved. The ancestor class is still read by `const&`, never written; `equivalenceClassesMap[ancestorV]` and `eqClassSttmntIndexMapMap[ancestorV]` untouched per [I-31](30_invariants.md#i-31).
- Same-vN merge semantics — unchanged.
- Verifier — untouched per [I-16](30_invariants.md#i-16).
- The D-46 cross-pair gate — unchanged. The two fixes are independent (D-46 prevents redundant records; D-47 admits the cross-vN call legitimately).

---

<a id="d-46"></a>
## D-46 — Cross-pair `equality2` emission gated on existing class/LB origin (2026-05-07)

**What.** `mergeTwoEquivalenceClasses` ([`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp), `mergeTwoEquivalenceClasses`) — the merged-pair history block — now skips its `equality2` cross-pair `addOrigin` calls when the target equality `(=[varA, varB]) @ validityName` (or its mirror) already has an origin entry in (a) `mergedOriginMap`, (b) `classB.equalityOriginMap`, or (c) `memoryBlock.exprOriginMap`. New invariant: [I-32](30_invariants.md#i-32). Coupled with a producer-side change at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) (`performElementaryLogicalStep`) that, after the bulk-merge of `body.mailIn.exprOriginMap` into `body.exprOriginMap`, additionally syncs mail-arrived equality origins into the class's `equalityOriginMap` for any class whose variables already contain both args of `(=[a,b])`.

**Why.** [`verifier.py::verify_origin_chain_termination`](../verifier.py) — the origin-chain-termination check — flagged 3 cyclic rows in `files/processed_proof_graph/22_check_zero.txt` (theorem 12 induction zero-case). Root cause:

The zero-case LB chain — `AnchorPeano → in2[v1,i0,s] → in2[i0,v2,s] → (=[v1,i0])` — is internally contradictory. `in2[v1,i0,s]` says "i0 = S(v1)" (v1 is a predecessor of zero), which contradicts the Peano "no successor of zero" axiom mailed in as `!(in2[v1,i0,s])`. The parent-scope contradiction-cascade emits the full equality clique `{(=[v1,v2]), (=[v1,i1]), (=[v2,i1])}` + mirrors and ships them to the LB via mail. The bulk-merge at `performElementaryLogicalStep` placed mail origins in `body.exprOriginMap`, but the equivalence-class state was unchanged. Then `mergeTwoEquivalenceClasses`, iterating possible bridge variables, produced cross-pair `equality2` records via different bridges that point at each other:

- `(=[v1,v2]) ← equality2 | (=[v1,i1]) (=[i1,v2])` (bridge `i1`, emitted when `(=[i1,v2])` arrived and merged a class containing `v1` with one containing `v2`).
- `(=[v1,i1]) ← equality2 | (=[v1,v2]) (=[v2,i1])` (bridge `v2`, emitted when `(=[v1,v2])` arrived against a class containing `i1`).

Both records survived into chapter 22; the verifier's DFS reported the cycle.

The fix has two coupled halves:

1. **Producer-side sync** (`prover.cpp`). After bulk-merge into `body.exprOriginMap`, walk `body.mailIn.exprOriginMap` and register mail equality origins in the matching class's `equalityOriginMap`. Closes the information gap that prevented the merge logic from seeing mail derivations as already-known.
2. **Consumer-side gate** (`prover.hpp`). Cross-pair `equality2` push only fires when the target is *not* already established by some other path. The `equality2` record is a transitive convenience — it documents derivability through the merge bridge. When the equality is already established (mail, prior merge, anchor handling, recursion premise), the convenience record contributes no new deductive content but can form cycles with parallel `equality2` records via different bridges.

**What was investigated and ruled out.**

- **Recent symmetry-handling commits** were initially suspected — they touch `addStatement`'s mirror push and `updateEquivalenceClasses`'s "symmetry of equality" emission blocks. Ruled out: none of them touch `mergeTwoEquivalenceClasses`'s `equality2` cross-pair emission. The cycle generator predates these commits. The new origin-chain-termination check in `verifier.py:3438-3492` is what made the latent cycle visible.
- **Symmetric-emission-only suppression** (skip cross-pair when the *symmetric* target `(=[varB, varA])` is already known but emit on the asymmetric one). Rejected as inconsistent: equality is symmetric by I-9, so suppressing one without the other produces lopsided origin maps.
- **Source-side check on cross-pair sources** (skip if `(=[varA, commonArg])` or `(=[commonArg, varB])` lacks an origin). Rejected: cycle requires the *target* to be already-established; checking the sources is orthogonal and would over-suppress.
- **Pure consumer-side gate without the mail-sync producer change.** Half-correct: `memoryBlock.exprOriginMap` already learns mail origins via the bulk-merge, so the gate against `memoryBlock.exprOriginMap` alone would close the verifier-visible cycle. Adopted both halves anyway because the class state should be the source of truth that the merge logic consults; making it complete is a correctness invariant in its own right (matches the user's "equi classes have their own originMap" framing).

**Trade-off.** Cross-pair pushes are now skipped for already-derived targets. Consequence on `equality2`-tagged chapter rows: fewer rows where a separate path already produced the target's origin. Verifier still receives complete `equality2` foundation chains for cases where the equality genuinely originates only via the merge bridge (no prior mail, no recursion premise, no anchor handling). The chapter-22 / theorem-12 cycle disappears; foundation-only Peano chapters are unaffected.

**Verified.** Peano-only `python main.py` clean run (planned in this commit's verification step). Acceptance criteria documented in commit message.

**Files touched.**

- `GL_Quick_VS/GL_Quick/src/prover.cpp`: producer-side mail-origin sync into class `equalityOriginMap` after bulk-merge in `performElementaryLogicalStep`. (Landed in the prior commit on this sub-branch.)
- `GL_Quick_VS/GL_Quick/src/prover.hpp`: consumer-side cross-pair gate inside `mergeTwoEquivalenceClasses`.
- `docs/20_core_concepts/05_equivalence_classes.md`: *Origin tracking and the mail-sync rule* subsection (prior commit) + cross-pair-gate subsection (this commit).
- `docs/20_core_concepts/03_mail_system.md`: cycle-boundary protocol step 4 mentions equality-origin sync (prior commit).
- `docs/30_invariants.md`: I-32 added.
- `docs/AGENT_SwDD.md`: invariant quick-reference table updated (I-32).
- `docs/50_gotchas.md`: G-31 added (cyclic `equality2` chains from cross-pair re-emission of mailed equalities).
- `docs/40_decisions.md`: this entry.

**What does NOT change.**

- `addStatement` — unchanged. Equality ingestion still routes through `updateEquivalenceClasses` line 5886 (incoming-equality origin) and the existing class-merge / cross-scope-deposit machinery.
- `mergeTwoEquivalenceClasses`'s subset path (lines 5566-5589) — unchanged. No cross-pair emission there.
- `applyEquivalenceClass` / `applyEquivalenceClassToRejectedMapIntegration` — unchanged. They consume class state but do not write `equality2` cross-pair origins.
- Verifier — untouched per [I-16](30_invariants.md#i-16). The new origin-chain-termination check that surfaced the bug was already present.
- D-numbering: D-45 is reserved (referenced from `prover.cpp:1393, 1494` for an assert-tightening decision not yet entered into this log; future agent should add D-45 entry separately).

---

<a id="d-44"></a>
## D-44 — `updateEquivalenceClasses` ancestor-scope merge: cross-NS extension preserves ancestor class (2026-05-05)

**What.** `updateEquivalenceClasses` ([`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp), `updateEquivalenceClasses`) gains an ancestor pass that runs after the existing same-NS merge loop. For each strict ancestor `V_a` of `validityName`, the pass iterates `mb.equivalenceClassesMap[V_a]` and absorbs every class whose `variables` overlap `eqArgs` into the new `mergedClass` at `validityName` via `mergeTwoEquivalenceClasses`. Ancestor classes themselves stay UNCHANGED at `V_a` — `mb.equivalenceClassesMap[V_a]` and `mb.eqClassSttmntIndexMapMap[V_a]` are never written ([I-31](30_invariants.md#i-31)).

**Why.** Pre-D-44, an equality `(=[a,b])` admitted at scope `V` (descendant) with `a` already in an ancestor class `C_a @ V_a` would build a `mergedClass = {a, b, …same-NS-overlaps…}` at `V` but never absorb `C_a`'s members. The transitive `b ≡ all of C_a at V` was recoverable only at runtime via the apply-side machinery — the descendant scope's stored class never explicitly carried the cross-scope picture.

This mirrors the apply-side bidirectionality landed in [D-33](#d-33). D-33 gave the apply machinery cross-scope reach (a class at one scope can rewrite a statement at a comparable scope, with the rewrite landing at the deeper of the two). D-44 gives the merge machinery the symmetric cross-scope reach: when the merge driver runs at descendant `V`, it pulls in ancestor-scope classes that the new equality bridges to.

Soundness rests on descendant-inheritance: a class at `V_a` is observably visible at every descendant of `V_a` (every `var ≡ var'` pair holds at every descendant). An equality at descendant `V` that bridges to an ancestor class can therefore legitimately propagate the ancestor's equivalences into the descendant's merged class. The original ancestor class must not be touched — the new equality is invisible at `V_a` (same-or-deeper visibility rule), so writing `V_a`'s state from a descendant-scope operation is unsound.

**What was investigated and ruled out.**

- **Modify the ancestor class in place.** Rejected. The new equality is admitted at the descendant scope only; the ancestor scope cannot see it, so adding the equality's transitive consequences to `V_a`'s class would conjure equivalences `V_a` should not yet know.
- **Descendant-scope class merge (class strictly deeper than the equality's scope).** Excluded by symmetry with [D-33](#d-33)'s rmi class-deeper exclusion. A class at a deeper scope is invisible at the equality's scope; pulling its members into the equality's merged class would conjure equivalences the merging scope should not know.
- **Chained ancestor merges (overlap test against the growing `mergedClass` instead of `eqArgs`).** Defer. Matches the existing same-NS contract — `mergeTwoEquivalenceClasses`'s bridge invariant requires `commonArg ∈ eqArgs ∩ classA ∩ classB`, so the overlap test against `eqArgs` is what feeds the bridge. Transitivity through ancestor classes that overlap only with same-NS-absorbed vars is still recoverable at runtime via the apply machinery, just not explicitly stored in the descendant's merged class. Documented as a *Known & tracked* weakness in `docs/20_core_concepts/05_equivalence_classes.md`.

**Trade-off.** Merged classes at descendant scopes now carry members from every relevant ancestor class. Applies-loop cost scales as `|class|^2` per substitution; at Gauss scale classes are typically small (≤5 members), ancestor merges may push them to 10–20. Origin-graph fan-out grows: `mergeTwoEquivalenceClasses` builds `mergedOriginMap` entries for every cross-pair. Verifier `equality1` / `equality2` checks pattern-match on origin chains — the merge logic is reused unchanged, so origin shape stays consistent.

**Verified.**

- Pipeline: full `python main.py` clean run.
- Verifier: main 2750 checks, 0 failures; incubator 32795 checks, 0 failures; total airtight.
- `proved_theorems.txt`: ≥41 lines (no theorem loss vs. main baseline).
- `global_theorem_list.txt`: Gauss summation induction theorem present; FTA-rung-1 lemmas (`interval`, `limitSet`, `limitSequence`, `sequence`, `fXY` rows under AnchorGauss) present.

**Files touched.**

- `GL_Quick_VS/GL_Quick/src/prover.hpp`: `updateEquivalenceClasses` — ancestor-pass block inserted after the same-NS merge loop, before the `mergedClass` push into `newClasses`.
- `docs/20_core_concepts/05_equivalence_classes.md`: new *Cross-scope class merge* section + chained-merge weakness entry.
- `docs/30_invariants.md`: I-31 added (ancestor classes read-only inputs to `updateEquivalenceClasses`).
- `docs/AGENT_SwDD.md`: invariant quick-reference + count update (30 → 31).
- `docs/40_decisions.md`: this entry.

**What does NOT change.**

- `mergeTwoEquivalenceClasses` — used as-is, takes `classB` by `const&`, so ancestor classes are read-only by construction.
- `applyEquivalenceClass` ([D-33](#d-33)) — unchanged. It already sees ancestor classes via `applyClassesFrom`'s ancestor-NS pass; now it sees a richer descendant `mergedClass` too, which composes naturally.
- `applyEquivalenceClassToRejectedMapIntegration` ([D-43](#d-43)) — unchanged. The rmi side remains additive at the new key; the ancestor-merge extension only affects the descendant-scope class-storage shape.
- `enumerateEqClassRewrites` shared helper — unchanged.
- Verifier — untouched per [I-16](30_invariants.md#i-16).
- Same-NS merge logic — unchanged. The ancestor pass runs after, and is purely additive on `mergedClass`.

---

<a id="d-43"></a>
## D-43 — `applyEquivalenceClassToRejectedMapIntegration` keep-old: rmi rewrites are additive (2026-05-05)

**What.** `applyEquivalenceClassToRejectedMapIntegration` ([`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp), function `applyEquivalenceClassToRejectedMapIntegration`) no longer erases the original rmi entry K1 when an equivalence class rewrites it to K2. Both the match path (revival via `internalMailIn`) and the no-match path (insert at K2) now leave K1 in place. Only K2 is added to `rmi`; the `toErase` queue and the post-loop `rmi.erase` apply-mutations block are removed. New invariant: [I-30](30_invariants.md#i-30).

**Why.** K1 is registered in `rmi` because at registration time it failed admission. It remains a candidate for revival via [`revisitRejectedIntegration2`](../GL_Quick_VS/GL_Quick/src/prover.hpp), which fires every time `makeAdmissionKeys` writes a new admission entry. `revisitRejectedIntegration2` iterates `rmi` keys and probes each against the current admission landscape. **A K1 erased by class application can never be revived by a future admission write**, even if that future admission would have matched K1 directly without any class involvement.

K1 and K2 also probe distinct admission slots: u_-form against `admissionMapIntegration`, bare form against `admissionSetIntegration`. The class's a→b substitution does not propagate to admission keys (admission is keyed structurally). So K2 failing admission today does not imply K1 will fail admission tomorrow under a different admission landscape.

This mirrors the additive principle [D-33](#d-33) already established for the expression-side cross-scope rewrite: original facts at the ancestor scope are never overwritten; the rewrite is purely additive at the new scope. Pre-D-43, the rmi side violated that principle by erasing K1 unconditionally — the same destructive pattern D-33 reverted on the descendant-class direction was still alive in the same-NS / class-shallower paths and was never reverted there.

**What was investigated and ruled out.**

- **Erase only on no-match path** (keep K1 on match, erase K1 on no-match). Considered as a partial fix. Rejected: the no-match path's erasure has the same revival-loss problem as the match path. K1 might match a future admission entry that the class's K2 cannot. The cleanest fix is uniform additive semantics across both paths.
- **Cap rmi growth** (e.g., LRU eviction, time-based decay). Not implemented in this commit; deferred until measurement shows growth is a problem at FTA scale. The existing `varsInRejectedMapIntegrationKeys` overlap short-circuit already suppresses additions for non-overlapping classes, which is the dominant fan-out.

**Trade-off.** `rmi` grows monotonically under class application instead of moving in place. Different keys (K1 + K2 + K3 from cross-class fan-out) accumulate in the map. Same-key inserts merge into the per-key value set (`std::map<ExpressionWithValidity, std::set<RejectedMapIntegrationValue>>` collapses duplicates), so identical (K, V) pairs from re-applied classes are idempotent. The fixpoint in `applyClassesFrom` terminates on `encodedStatements.size` plateau, **independent of `rmi` state** — kept K1 entries do not prolong the fixpoint unless they themselves drive new statements through revival. Memory overhead at Gauss scale is bounded; FTA-scale measurement still pending.

**Verified.**

- Pipeline: full `python main.py` clean run.
- Verifier: main 2750 checks, 0 failures; incubator 32795 checks, 0 failures; total airtight.
- `proved_theorems.txt`: ≥41 lines (no theorem loss vs. main baseline).
- `global_theorem_list.txt`: Gauss summation induction theorem present; FTA-rung-1 lemmas (`interval`, `limitSet`, `limitSequence`, `sequence`, `fXY` rows under AnchorGauss) present.

**Files touched.**

- `GL_Quick_VS/GL_Quick/src/prover.hpp`: `applyEquivalenceClassToRejectedMapIntegration` — drop `toErase` declaration, drop `toErase.push_back(keyEv)`, drop the post-loop `for k in toErase: rmi.erase(k)`. Update three comment blocks to document additive semantics.
- `docs/20_core_concepts/05_equivalence_classes.md`: Extension section items 4–5 rewritten; "Why additive (D-43)" + "rmi growth under additive semantics" paragraphs added; rmi-growth weakness entry added.
- `docs/30_invariants.md`: I-30 added (keep-old).
- `docs/40_decisions.md`: this entry.

**What does NOT change.**

- I-22 (admission-map entry not cleaned on revival) — orthogonal; unchanged.
- I-21 (`internalMailIn` cleared at top of hashburst after absorb) — unchanged.
- D-33 scope-direction admission for rmi (rejects descendant direction) — unchanged.
- `applyEquivalenceClass` (expressions path) — unchanged; was already additive per D-33.
- `revisitRejectedIntegration2` — unchanged. It already operates on whatever's in `rmi`; with more entries kept it has more candidates to revive.
- `enumerateEqClassRewrites` shared helper ([D-NN](20_core_concepts/05_equivalence_classes.md#shared-inner-loop-helper), commit ) — unchanged.
- Verifier — untouched per [I-16](30_invariants.md#i-16).

---

<a id="d-42"></a>
## D-42 — `v / V` registry form, `w / W` citation form for theorem-anchor implications (2026-05-05)

**What.** Two coordinated changes — one pass added on each side of the producer/verifier boundary:

1. `process_proof_graphs.py` ITERATION 5 (`_v_to_w_in_theorem_citation`): for every chapter cell at column index ≥ 3 (the rest fields, i.e. cited dependencies — never the chapter's own HEAD claim at column 0) that passes `is_theorem_anchor_implication`, apply a case-preserving `v→w / V→W` letter swap to every `[vV]\d+` token at argument positions. Anchor variables (`N, i0, s, +, *, i1, i2, id, …`) never match `[vV]\d+` and are untouched. `<digit>_copy` forms are excluded by the lookahead. `global_theorem_list.txt` is built independently from `renamed_theorems` and remains in `v / V`.

2. `verifier.py` (`_revert_w_to_v_in_theorem_citation` + `_is_theorem_anchor_impl_local`): symmetric inverse. At three sites (`origin` meta-check `rest[0]` for `implication / multiplied from / mirrored from / reformulated from`, `check_theorem_tag`, `check_externally_provided_theorem`), if the cited expression has the theorem-anchor shape, apply `w→v / W→V` and try the reverted form in `state.global_theorems` / `state.external_theorems` before falling through to `_normalize_expr_list`. `check_implication`'s anchor branch needs no change (`_normalize_all_vars_in_list` already folds *every* arg).

**Why.** Pre-D-42, cited foreign-theorem inner bvars rendered with the same `v / V` letter as chapter-global free variables. A reader scanning `(in3[v1, i0, v2, +])` inside a chapter could not tell at sight whether the `v1, v2` were chapter-global free variables (Priority-2 numbering, shared with the head row) or bvars scoped inside an applied foreign theorem. Reserving `w / W` for the citation case removes the ambiguity at sight while keeping `v / V` as the registry/title form (so `global_theorem_list.txt` and the chapter HEAD of mirror / reformulation / OR-theorem chapters continue to present each theorem in its canonical registry form).

The user's stated three-step shape: *find applied/foreign theorem in a chapter → replace `v` by `w` → verifier recognises the citation, reverses `w → v`, and looks up the `v` form in `global_theorem_list.txt`*. The Pass 3 + revert pair implements this verbatim.

**What was investigated and rejected.**

- **HTML-side rename** (the historical `_v_to_w_display` in `generate_full_proof_graph.py`, removed). Flawed: verifier reads the processed proof graph, not the HTML, so verifier and customer saw different forms — verifier-vs-display split. Rejected permanently. The fix lives at the producer side (`process_proof_graphs.py`), so verifier and HTML consume the same form.
- **Renumbering per cell** (give each cited theorem fresh `w1, w2, …`). Rejected: breaks the digit-preserving round-trip with the registry, and makes the verifier-side revert non-trivial. The case-preserving letter swap is the simplest construction that satisfies "both sides agree on the digit, both sides differ only by letter".
- **Relying on `_normalize_expr_list` fold alone** (no explicit revert in verifier). Functionally correct — the fold already widens to `[vVwW]\d+` (per [D-NN c?590885 — implicit in the commit's verifier widening]) — but loses the fast `dep in state.global_theorems` exact-string path on every citation lookup. The explicit revert preserves the fast path and makes the round-trip invariant auditable.

**Why HEAD column 0 stays `v / V`.** Mirror / reformulation / OR-theorem chapters carry the chapter's *own* theorem in column 0 — that is the chapter's "title" / claim being proved. Rendering it in the same form as `global_theorem_list.txt` keeps title/registry parity; a reader can copy the HEAD verbatim and grep for it in the registry. Direct-proof and induction chapter HEADs are smaller fragments (not theorem-anchor implications), so the `is_theorem_anchor_implication` guard skips them implicitly anyway — the column-0 guard is the load-bearing rule that protects the mirror/reformulation/OR cases.

**Verified.**

- Main verifier (post-process re-run, raw graph unchanged): 2750 checks, 0 failures.
- Incubator verifier: 32795 checks, 0 failures.
- Spot-checked artefacts: `14_direct_proof.txt` rest cells in `w / W` form; `38_mirrored_statement.txt` HEAD in `v / V`, rest[0] in `w / W`; `109_reformulated_statement.txt` HEAD has `v1, v2, V1`, rest[0] has `w1, w2, W1, W2` (V → W proven on the set-typed bvars); `12_or_theorem.txt` HEAD `>[v1]`, rest[0]+rest[2] both `>[w1]`; `global_theorem_list.txt` carries zero `[wW]\d+` tokens.

**Files touched.**

- `process_proof_graphs.py`: `_VW_SWAP_PATTERN` + `_v_to_w_in_theorem_citation` helpers; ITERATION 5 in `create_processed_proof_graph`. ~30 LOC additive.
- `verifier.py`: `_WV_REVERT_PATTERN` + `_is_theorem_anchor_impl_local` + `_revert_w_to_v_in_theorem_citation` helpers; revert-then-membership added at the three lookup sites named above. ~25 LOC additive; no existing checker semantics modified (the revert is wired *before* the existing exact-string membership tests, with the fold-based fallback unchanged behind it). I-16 honoured.
- `docs/10_pipeline/06_process_proof_graph.md`: new `## Pass 3 — v→w rename of cited theorem-anchor implications` section.
- `docs/10_pipeline/07_html_export.md`: w-rename row in the Selected-functions table extended to mention Pass 3 alongside Pass 2.
- `docs/10_pipeline/08_verifier.md`: origin-check code excerpt updated to show the `dep_v` fast path; new `### w → v revert for cited theorem-anchor implications` subsection under the `origin` meta-check.
- `docs/40_decisions.md`: this entry.

**What does NOT change.**

- `generate_full_proof_graph.py` — already neutralized; HTML renders processed cells as-is.
- C++ prover, conjecturer, compressor, MPL configs — orthogonal.
- `_normalize_expr_list`, `_normalize_with_unchangeables`, `_alpha_canonicalize_bound_vars` — already widened to `[vVwW]`; remain widened, used as the safety net behind the explicit revert.
- The per-cell `w / W` rename for non-anchor implications (`w_rename_impl_local`, Pass 2). Pass 3 is *additive* — Pass 2 still handles products of disintegration with raw `\d+` bvars; Pass 3 handles cited theorem-anchor implications. The two passes act on disjoint cell categories.
- Renumbering. Pass 3 is a pure letter swap; the digit is preserved.

---

<a id="d-41"></a>
## D-41 — Verifier `definition set consistency` meta-check + ConfigVisu defset drift fix (2026-05-03)

**What.** Two coupled changes:

1. New verifier meta-check `definition set consistency` (a per-row check counted in the standard ~32-line tally) implementing the C++ compiler's defset-consistency algorithm at `compiler.hpp` (`ArgumentAnalyzer` + `RecursiveParser::parseSubtree` + `mergeMaps` + `processLeaf` + `checkDefinitionConsistency`) faithfully in Python. For every chapter row, every expression in the row (the left-hand expression and each `rest[i]` at even indices) is parsed independently with bound-variable scoping at `>[…]` quantifier nodes; `_merge_maps` flags type-label mismatch when two child sub-trees share a variable name with disagreeing types. Per-batch resolved defsets are pre-computed once in `run_verifier` startup and selected per-chapter via the same anchor-substring match the verifier already uses for `state.current_gl_binary` (mirrors the compiler's per-batch `ArgumentAnalyzer(this->coreExpressionMap)` construction at `prover.hpp`).

2. `ConfigVisu.json` defset drift fix. Audit between `ConfigVisu.json` and the per-batch configs (`ConfigPeano`, `ConfigGauss`, `ConfigIncubatorPeano`, `ConfigIncubatorGauss[1]`) found one drifted defset (`interval` position 2: ConfigVisu had `P(x(1)(1))`, all batch configs had `P(x(1)(x(1)(1)))`) and six operators present in batch configs but absent from ConfigVisu (`infiniteSequence`, `limitSequence`, `limitSet`, `constSeq`, `nonInterval`, `nonSequence`). All 7 issues fixed by syncing `ConfigVisu.json` to the per-batch authoritative versions.

**Why.** Coverage gap surfaced by the user: the verifier did not check whether a chapter row's variable connections respected definition-set typing. A theorem could in principle have variables wired through ports with conflicting type labels and every existing verifier check would still pass — formal completeness gap. The compiler enforces this on the producer side (its `ArgumentAnalyzer` runs at `compileCoreExpressionMap` time), but the verifier's purpose is to be the independent oracle on shipped artefacts. Mirroring the compiler's algorithm in the verifier closes the gap.

The drift fix in ConfigVisu was discovered as a Stage-1-style finding when the new check first ran: 46 main + 357 incubator = 403 failures fired, all rooted in `interval` pos-2 type contention. Sync'd ConfigVisu, re-ran: 1 + 266 = 267 failures (down 137). Remaining failures decomposed into a verifier-algorithm scoping bug (Class A — initial regex-based pooling did not respect bound-variable rebinding at `>[v1,v2]` quantifiers, treating same-name variables across alpha-distinct scopes as the same variable) and a per-batch compact-name collision case (Class B — `implication26` is allocated arity 2 in Gauss main but arity 5 in IncubatorGauss; my initial single-flat-resolved-defsets dict picked one binary's allocation arbitrarily). Class A fixed by translating the C++ recursive parser faithfully (Python `_parse_subtree` + `_merge_maps` + `_process_leaf`). Class B fixed by per-tag resolution (the analog of the compiler's per-batch analyzer construction).

**Verified.**

- Main side: 2632 successful defset checks, 0 failures (35545 / 0 — airtight).
- Incubator side: 31584 successful defset checks, 0 failures (32795 / 0 — airtight).
- Synthetic-corruption sanity test: 4 hand-crafted type-mismatched expressions correctly flagged; 4 well-typed expressions correctly accepted (including the bound-var-rebinding case `(>[v](in[v,N])(=[v,a]))` that confirms scoping works).

**Files touched.**

- `verifier.py`: `VerifierState` now carries `resolved_defsets_per_tag`, `resolved_defsets_atomic_only`, and per-chapter `current_resolved_defsets`. New helpers: `build_resolved_defsets_per_tag`, `_try_derive_from_elements`, `_merge_maps`, `_process_leaf`, `_parse_subtree`, `check_defset_consistency`. Wiring at `run_verifier` startup (build the per-tag indices) and `verify_chapter` (select per-chapter `current_resolved_defsets` alongside `current_gl_binary`). New per-row check loop after the `origin` meta-check. ~280 LOC additive; no existing checker modified. I-16 honoured via additive-only change with explicit user consent for this specific extension.
- `files/config/ConfigVisu.json`: `interval` pos-2 sync to ternary; six operators added (`infiniteSequence`, `limitSequence`, `limitSet`, `constSeq`, `nonInterval`, `nonSequence`).
- `docs/40_decisions.md`: this entry.
- `docs/30_invariants.md`: I-29 (variable-port type consistency invariant).
- `docs/10_pipeline/08_verifier.md`: new "definition set consistency meta-check" section.
- `docs/20_core_concepts/06_anchors_and_scopes.md`: weakness at line 172 ("Anchor-slot typing table is not validated at load") updated — the verifier-side check now closes the in-flight artefact half of the gap; the load-time half (config↔MPL-definition cross-validation) remains.
- `docs/04_configs.md`: note added that ConfigVisu.json should mirror per-batch configs for shared operators.

**Why "compiler's variant" in particular.** Translating the C++ algorithm verbatim instead of inventing a new check has three benefits: (a) it inherits the compiler's known-correct scoping rules (bound-var removal at `>[…]` is the only place where variables leave scope, and `mergeMaps` is the only mismatch detector); (b) it matches the producer-side contract exactly, so any verifier-side failure points squarely at a producer-side bug rather than a verifier-side definition gap; (c) Python translation of ~150 LOC of C++ is auditable, with `verifier.py`'s comments citing the C++ line numbers for every translation point.

**Per-batch resolution and shared-binary fallback.** `build_resolved_defsets_per_tag` builds one resolved-defset map per `tag` in `gl_binaries`. Each tag's map starts from atomic seeds (ConfigVisu.json), unions in `GL_binary_shared.json`'s composites (cross-batch fallback for spontaneous-category compact names — `_SPONTANEOUS_CATEGORIES = {"implication", "existence", "or", "and"}`), then resolves the tag's own composites with override semantics on collision (per-batch is authoritative for its own chapters). `verify_chapter` picks the right tag's map at chapter start. Chapters whose theorem doesn't disclose an anchor (rare) fall back to atomic-only.

**What does NOT change.**

- No existing verifier checker modified. I-16 honoured.
- ConfigVisu drift fix is data-correction (sync), not schema change.
- Per-batch configs unchanged.
- Verifier returns the same 28-line tally + 1 new line; airtight runs stay airtight.

**Open follow-up.**

- Cross-row in-chapter aggregation (Phase 2 from the original plan): variables in `chapter row N` and `chapter row M` at the same `validityName` could be type-checked together. Deferred — not exercised by current chapter shapes; would be additive when needed.
- Cross-chapter rule citations (Phase 3): rules cited from one chapter's `rest[0]` lookup match against `state.global_theorems` registry — the registry entries themselves could be defset-checked on load. Deferred — the producer-side compiler already enforces this for any rule entered into its `coreExpressionMap`.

---

<a id="d-40"></a>
## D-40 — Cross-batch externals seed switched from `proved_theorems.txt` (expanded) to `compiled_proved_theorems.txt` (compact) (2026-05-03)

**What.** `run_modes.py` (the per-tag incubator-stage seed step) was reading `files/theorems/proved_theorems.txt` (expanded form — compiled structural operators like `existence2` / `or0` rewritten to base form by `prover.cpp`'s `expandToBaseForm`) and writing it as the next-tag incubator's `externally_provided_theorems.txt`. Switched to `files/theorems/compiled_proved_theorems.txt` (compact form — keeps `existence2` / `or0` as compact heads; `prover.cpp`).

**Why.** Stage-2 cleanup of the residual verifier failure exposed by D-39's determinism fix. Once execution was deterministic, the incubator verifier reliably reported one `origin`-tag failure at `1209_direct_proof.txt` row 57, an `implication` row whose `rest[0]` cited the Peano rule `(>[N,i0,s](AnchorPeano[N,i0,s,+,*,i1])(>[i2](in[i2,N])(>[]!(=[i2,i0])(existence2[N,i2,s]))))` (with the compact head `existence2`). The verifier's `origin` check looks `rest[0]` up in `state.global_theorems ∪ state.external_theorems` via alpha-canonical match, but `_alpha_canonicalize_bound_vars` only renames bound variables — it does not expand or compact structural operators. The seed file was carrying the rule's expanded form (`!(>[8](in[8,1])!(in2[8,7,3]))` instead of `existence2[1,7,3]`), so chapter rows that cite the compact form had no registry hit.

The expansion at `prover.cpp` was originally added to keep `proved_theorems.txt` parseable by a downstream batch that might not have the same compact-name dictionary loaded at parse time. By the time `GL_binary_shared.json` infrastructure (`run_modes.py:_seed_per_batch_binary` + `_merge_into_shared`) was added — which makes the spontaneous-category compact dictionary (existence/or/and/implication) cross-batch — the expansion at the seed-propagation step became unnecessary for actual cross-batch parsing of those categories. Non-spontaneous categories (anchor entries, atomic entries) are still excluded from the shared binary, so the inter-batch `proved_theorems.txt` retains its expanded form and remains the safe source for category-agnostic cross-batch parsers; only the seed-propagation source for the next-tag incubator switches to the compact-form file.

**Caveat documented inline (`run_modes.py`).** Cross-batch parsing of compact-form externals is supported only because `GL_binary_shared.json` carries the spontaneous compact-name dictionary. If a future batch references compact heads outside the spontaneous categories — i.e. names that are batch-local — the seed switch will reintroduce parser failures. The fix in that case is to expand the shared-binary coverage or to revert the seed source.

**Verified.** 3 full-pipeline runs (Peano + Gauss, both incubator and main) post-fix: per-tag verifier counts identical; processed-proof-graph md5 identical; proved-theorems md5 identical; total verifier checks `35545`, **0 failures across all tags** (incubator: 32795 checks, 0; main: 2750 checks, 0). All Stage-1 + Stage-2 DOD criteria met.

**A/B against pre-Stage-2.**

| | Pre-Stage-2 (post-D-39) | Post-Stage-2 (post-D-40) |
|---|---|---|
| Incubator checks | 32796 | 32795 (-1: redundant `origin` lookup gone) |
| Incubator failures | 1 (`origin` tag) | 0 |
| Main checks | 2750 | 2750 |
| Main failures | 0 | 0 |
| Total | 35546 / 1 failed | 35545 / 0 — airtight |
| 3-run identity | ✅ (D-39) | ✅ (preserved) |

**What does NOT change.**
- `proved_theorems.txt` continues to be written in expanded form (`prover.cpp`) — its role as inter-batch parser-input (for downstream consumers that aren't gated on the compact dictionary) is unchanged.
- `compiled_proved_theorems.txt` continues to be written in compact form — its role as proof-graph pruning source is unchanged.
- `--mirror-externals` C++ step's input file (`externally_provided_theorems.txt`) shape is unchanged; only the source content (compact instead of expanded). The mirror logic operates on whatever form it receives.
- Verifier (`verifier.py`) is untouched (I-16).
- D-39's determinism fix is unaffected; the deferred-action collector remains intact.

**Files touched.**
- `run_modes.py`: source file changed from `proved_theorems.txt` to `compiled_proved_theorems.txt`; new comment block documenting the rationale + cross-batch caveat.
- `docs/40_decisions.md`: this entry.
- `docs/10_pipeline/09_incubator.md`: externals-seed paragraph updated to reflect the new source.

**Open follow-up.**
- If a future Stage proves theorems with compact heads outside `_SPONTANEOUS_CATEGORIES` (`run_modes.py`) that need to cross batches via the externals seed, the shared-binary coverage will need to expand. Track via a future invariant or D-entry as needed.
- The `expandToBaseForm` call at `prover.cpp` is no longer needed for the externals propagation path. It is retained for any other consumer of `proved_theorems.txt` that may need the expanded form. Could be revisited if no such consumer exists.

---

<a id="d-39"></a>
## D-39 — Race-free contradiction-origin propagation via deferred-action collector (`pendingAncestorOrigins`) (2026-05-03)

**What.** Replace the `addStatement` `primedForContradiction` handler's direct walk over `memoryBlock.parentMemory` (`prover.cpp` was 4763-4768) with a class-level deferred-action collector that drains in `proveKernel` after `pool.join`, single-threaded, in sorted order. The walk-and-`addOrigin` step itself is unchanged — it is moved from the parallel-phase descendant thread to the post-join single-threaded drain. Producer site stages a `PendingAncestorOrigin{ &memoryBlock, ev, origin, maxOrig }` under `pendingAncestorOriginsMutex`; consumer site sorts by `(emitter.exprKey, ev.original, ev.validityName, origin.first)` and walks each emitter's `parentMemory` chain calling `addOrigin` on each ancestor.

**Why.** GL was non-deterministic on `main` HEAD: 3 consecutive `main.py` runs produced different verifier check counts and proof graphs. Investigation traced the root cause to the contradiction handler: when a "primedForContradiction" LB detected a reductio (`(A) ∧ ¬(A) ⊢ ¬(assumption)`), the handler walked its own `parentMemory` chain and called `addOrigin(pred->exprOriginMap, …)` on every ancestor. `addOrigin` (`prover.hpp`) is a plain `std::map[ev]; vec.push_back(...)` — not thread-safe. Meanwhile the parallel `proveKernel` worker pool (`prover.cpp`, `workers = std::thread::hardware_concurrency`) had OTHER threads reading and writing those same ancestor maps via their own `performElementaryLogicalStep`. Concurrent `std::map` insert + iterate is undefined behavior. The race produced a bimodal pattern: an entry was sometimes recorded, sometimes lost.

**Diagnostic evidence.** Hashburst dump retargeted to AnchorIncubator base LB (the single mailing recipient of the entire incubator deductive process). 3 incubator-Peano-only `main.py` runs:

| Run | hashburst hash | bytes | verifier |
|---|---|---|---|
| 1 | `4eb46e2c…` | 48,000,529 | 28723 checks, 2 FAILED (self-reference) |
| 2 | `055bb9cd…` | 47,995,681 | 28778 checks, 0 failures (airtight) |
| 3 | `055bb9cd…` | 47,995,681 | 28778 checks, 0 failures (airtight) |

First divergence at HASHBURST #10: Run 1 has `origins=3690`, Run 2/3 have `origins=3691`. The missing entry in Run 1 is exactly:
```
!(in3[10,11,2,4]) | v=main
  <- contradiction | (in2[14,2,3]) | !(in2[14,2,3]) | (in3[10,11,2,4])
```
A reductio chain `(in3[10,11,2,4]) ⊢ (in2[14,2,3]) ∧ ¬(in2[14,2,3]) ⇒ ¬(in3[10,11,2,4])`. When the entry survives, the verifier traces 55 additional checks and reports 0 failures. When it vanishes, the verifier reports 2 self-reference failures (the chain has no contradiction origin to follow).

**Verified.** After the fix, on:

- 3 incubator-Peano-only runs: byte-identical hashburst (`055bb9cd…`), all 28778 checks, 0 failures.
- 3 full-pipeline runs (Peano + Gauss, both incubator and main): byte-identical hashburst (`40d18b02…`), per-tag verifier counts identical, byte-identical processed-proof-graph md5, byte-identical proved-theorems md5. One verifier failure remains in tag `origin` (success 266, failure 1) on the incubator side — single-digit and reproducible across all 3 runs, consistent with a pre-existing IncubatorGauss origin chain bug now exposed by deterministic execution. Stage 2 of the determinism work (separate phase) addresses it.

**A/B against the abandoned predecessor ( D-38, commit ).** The earlier branch attempted a "canonical-min origin selection" fix that made verifier counts identical but introduced 1078 verifier failures and left intermediate state non-identical. That branch is reference material only and is not cherry-picked. D-39 is the canonical determinism fix on `main`-derived branches; D-38 (on the abandoned branch) is superseded.

**Why "induction-precedent collector" over a `mailOutAncestor` mail channel.** Two viable patterns, both deferring the cross-LB write to a post-`pool.join` single-threaded drain:

- *Mail-channel* (`mailOutAncestor` on `Memory`): per-LB inbox/outbox mirror of `mailOut`. Race-safe via same-LB writes during parallel phase; new field on every Memory; new smashMail variant.
- *Class-level collector* (`pendingAncestorOrigins` on `ExpressionAnalyzer`): mutex-guarded vector + post-join sort + walk. Direct precedent in `inductionMemoryBlocks` (`prover.hpp, 112`; `prover.cpp, 6269-6278`).

Picked the collector pattern. Reasons: direct existing precedent (the induction collector is structurally identical — same skeleton, different payload, drained in the same post-join block), tighter memory footprint (only firing LBs contribute entries), and keeps the mail abstraction (`mailOut`/`mailIn`) reserved for grid-broadcast cycle communication rather than special-cased ancestor writes.

**Files touched.**
- `GL_Quick_VS/GL_Quick/src/prover.hpp`: `PendingAncestorOrigin` struct, `pendingAncestorOrigins` vector, `pendingAncestorOriginsMutex`.
- `GL_Quick_VS/GL_Quick/src/prover.cpp`: constructor init list; producer swap at the contradiction handler; drain block in `proveKernel` after the existing induction-block drain. Also: hashburst-dump retarget to AnchorIncubator base LB, pointer-address scrub, `encodedMap` sort-on-emit, dump toggled OFF for production. Rule-12 stale-comment fix at the `exprOriginMap` block.
- `GL_Quick_VS/GL_Quick/src/filter.cpp`: reset in `releaseCEBatchMemory` (whole CE-batch teardown extracted from `prover.cpp`, 2026-05-04 — same body, new TU).
- `run_modes.py`: temporarily set to incubator-Peano-only during the iteration loop; restored to full pipeline for Phase 4 validation.
- `docs/40_decisions.md`: this entry.
- `docs/30_invariants.md`: I-22 (cross-LB writes during parallel phase forbidden — defer to post-`pool.join`).
- `docs/20_core_concepts/03_mail_system.md`: stale `logicalCores = 1` claim corrected (Rule 12); deferred-action collector pattern noted alongside `internalMailIn`.

**What does NOT change.**
- `verifier.py` is untouched (I-16).
- `parameters.max_origin_per_expr = 1` is untouched.
- The contradiction handler's own-LB writes (`memoryBlock.exprOriginMap`, `memoryBlock.mailOut.exprOriginMap`) are kept — they were always race-safe.
- Mail routing (`mailOut → smashMail → mailIn`) is untouched — descendant-direction propagation continues unchanged.
- Theorem set proved is unchanged (md5-identical to baseline).

**Open follow-up (Stage 2).** The 1 residual `origin`-tag failure in the IncubatorGauss path is now reproducible byte-identically and can be debugged by standard trap-debug methodology with the determinism guarantee from this fix in hand.

---

<a id="d-37"></a>
## D-37 — HTML export: matryoshka sub-proof nesting (arbitrary depth) (2026-05-03, branch `main`)

**What.** `generate_full_proof_graph.py`'s sub-proof renderer is now recursive. Sub-sub-…-proofs render as collapsed cards inside their parent's collapsed card — Russian-doll style, no fixed depth. Implementation: `_partition_stack_subproofs` recurses on each child scope's row group; `_render_subproof_card` recurses on `nested_subproofs`. Per-depth CSS classes (`.subproof-depth-1` … `.subproof-depth-5`, plus a generic `:not` rule for depth 6+) differentiate nested cards visually with color + indent.

**Why.** D-36's producer-side `ordisMerge` extension caused chapter exports to surface deeper nested scopes (`_ordis_` branches inside an outer `_orint_` subproof, etc.). The previous one-level-only sub-proof renderer flattened these into the main stack of the parent subproof, losing the structural information. The matryoshka rendering preserves it.

**Detection.** Scope hierarchy is derived purely from primary-namespace ancestry (`row[1]` of each row): a row at namespace `A_boundary_<X>` is a child of the scope at namespace `A`. No tag-level marker is required for nesting detection. Title inference for a child scope reuses `validity name` row metadata when present (preserves existing rich titles for implication subproofs); falls back to the namespace's payload pattern (`_orint_` / `_ordis_`) for OR-branch sub-subproofs that have no `validity name` introduction.

**Verification.** Chapter `1209_direct_proof.txt` (FTA-rung-1 forward direction, `EnumerationSet2 ⟹ interval`) renders as 3 depth-1 cards (impl24/impl25/impl26 subproofs) with 5 depth-2 cards nested inside (2 `_ordis_` branches under impl24, 2 under impl25, 1 `_orint_` subproof under impl26). DOM-depth walk confirms the depth-2 cards are inside their parent's `<div class='subproof-body'>`. Other chapters with no nested scopes render identically to before.

**Files touched.** `generate_full_proof_graph.py` (rewrite of `_partition_stack_subproofs`, new `_scope_title_info`, new `_render_subproof_card`, rewrite of `render_stack_with_subproofs`, new CSS depth classes); `docs/10_pipeline/07_html_export.md` (new "Matryoshka subproof structure" section).

---

<a id="d-36"></a>
## D-36 — Producer-side `or convergence` spec'd row + verifier correction on `or branch proven` (2026-05-03)

**Background.** `verifier_rung1` ended with 3 deliberate verifier failures on the incubator path. This entry covers their resolution. Two go away via a producer-side `ordisMerge` extension; one goes away via a verifier-side check correction (a check based on a wrong premise).

**Corrected `_orint_` mental model (user clarification, 2026-05-03).** `_orint_` does not case-split. To prove `(A ∨ B)` it rewrites the goal into two implication-sub-proofs `(!A → B)` and `(!B → A)`; each is a subproof scope. Inside each scope the negated antecedent is assumed (emitted as `or branch assumption`), and the subproof tries to derive the consequent. If one subproof fires, the OR is derived at parent scope and emitted as `or branch proven` — that row IS the OR's derivation by design. There is no separate non-`or branch proven` derivation row, by construction.

This is distinct from `_ordis_` (case-split, all-branches-converge — see [D-34](#d-34) and [`07_or_branching.md` §3b](20_core_concepts/07_or_branching.md#3b-_ordis_-or-disintegration-all-branches-converge)). `_ordis_` consumes an existing OR; `_orint_` produces an OR.

The "branch" terminology in `or branch proven` and `or branch assumption` is a historical misnomer — these are subproof rows. Renaming has been deferred to avoid touching every consumer at once; the doc now flags this clearly.

**Stage P0 — Verifier correction (`check_or_branch_proven` round-3 check #1 dropped).** The pre-fix check required a chapter row with `expression == or_expr`, `namespace == parent_ns`, and `tag!= "or branch proven"`. Codex round-3's rationale was "chapter 1209 line 19 derives the OR before line 22 splits it"; closer reading shows line 19 USES `(or2[i1,v5,i0])` as a premise of an implication firing (rule premise → conclusion), it does not derive the OR. Under correct `_orint_` semantics the OR has no separate derivation. The check is unsatisfiable for legitimate proofs.

Per the user's directive ("no relaxations of verifier") this is a CORRECTION (the check encoded a wrong premise) rather than a relaxation (softening a real check to pass tests). The four remaining `check_or_branch_proven` validations (layout, OR-shape, disjunct-membership, exact-namespace) and the round-3 check #2 on `check_or_branch_assumption` (matching `or branch proven` row) are unaffected — both are well-founded under the corrected semantics.

**Stage P1 — Producer-side `or convergence` spec'd row layout (`ordisMerge` extension).** The `_ordis_` convergence row is the legitimate gap. Pre-fix the row had 4 fields `(C, parent, or convergence, OR, parent)` and the verifier's `check_or_convergence` deliberately failed it (clean-fail per [D-35](#d-35)). Per the user-directive of 2026-05-03 the new producer-side row layout is fixed:

```text
<C>  <parent>  or convergence  <OR>  <parent>  <C>  <branch_D1>  <C>  <branch_D2>  …  <C>  <branch_DK>
```

`ordisMerge` (`prover.hpp` ~line 6020) now extends `mergeOrigin.second` with one `(C, fullBranchValidity)` per branch in `memoryBlock.orBookkeeping[mergeKey]`. The `fullBranchValidity` is reconstructed via the existing `branchPrefix + branchPayload` formula. The chapter export pipeline naturally renders the per-branch derivation rows of `C` because `removeExpressionFromMemoryBlock(state=0)` (called by `ordisMerge` to collapse the per-branch copies) does NOT touch `exprOriginMap` — the branch derivations remain available, and `buildStack` (visualizer.cpp) recurses into each new ingredient via `exprOriginMap.find`.

**Stage P1b — Tighten `check_or_disintegration` for compiled OR (newly-exposed by P1).** Stage P1's producer-side `ordisMerge` extension caused `buildStack` to recurse into per-branch ingredients of the new convergence row layout. Side-effect: 4 previously-hidden `or disintegration` rows (chapter 1209 lines 17, 21, 29, 33) surfaced. They use the compiled-OR shape `(or<N>[…])` in `rest[0]`, but the legacy `check_or_disintegration` accepted only the expanded `!(&!(…))` form and never validated namespace structure or OR-origin. Tightened in P1b to:

1. `len(rest) == 2` exactly.
2. `rest[0]` is a known compiled OR with ≥2 disjuncts via GL-binary (matching arity).
3. `line.expression` is one of the disjuncts (modulo equality symmetry).
4. `line.namespace` is EXACTLY `rest[1] + "_boundary_ordis_" + rest[0] + "_(" + <disjunct> + ")"` for `<disjunct>` matching `line.expression` (modulo equality symmetry).
5. The OR has an independent derivation row at parent scope (`expression == rest[0]`, `namespace == rest[1]`, `tag!= "or disintegration"`). **This check IS well-founded for `_ordis_`** — case-split CONSUMES an existing OR. The analogous check on `check_or_branch_proven` was dropped above because `_orint_` PRODUCES an OR.

The asymmetry between `check_or_branch_proven` (no OR-origin check) and `check_or_disintegration` (OR-origin check required) is exactly the `_orint_` (produces) vs `_ordis_` (consumes) semantic distinction.

**End state (D-36).** Main pipeline `2750/0` (unchanged). Incubator `32763/1` — the 3 originally-targeted failures all closed:
- Failures #1, #2 (`or convergence` lines 5, 14) — closed by Stage P1.
- Failure #3 (`or branch proven` line 22) — closed by Stage P0 (verifier correction).
- The 4 newly-exposed `or disintegration` rows (chapter 1209 lines 17, 21, 29, 33) — closed by Stage P1b.

**Remaining 1 failure (separate, pre-existing).** `self-reference failure 1` on chapter `1032_direct_proof.txt`: the chapter's target theorem `(>[…](AnchorIncubator)!(fold[N,s,+,id,i0,i1,i7]))` appears in the incubator's `global_theorem_list.txt` as `direct -1`, and the chapter's only proof step cites this same implication as a `theorem` rule — circular self-citation. The verifier's self-reference counter is correct to flag it. Cause: producer-side issue (likely conjecturer or processor — the theorem ends up in `global_theorem_list` and is re-cited in its own chapter's proof). NOT caused by the `ordisMerge` change (the chapter-1032 theorem has no OR involvement); surfaced incidentally by the `main.py` rerun that regenerated the chapter set. Tracked as a separate next-task investigation per the project conventions `Failures are first-class` ("Failures are first-class").

**Files touched.** `verifier.py` (Stage P0), `GL_Quick_VS/GL_Quick/src/prover.hpp` (Stage P1), `docs/20_core_concepts/{07_or_branching,08_proof_tags}.md`, `docs/10_pipeline/{04_prover,06_process_proof_graph}.md`, `docs/40_decisions.md`, `docs/verifier_rung1_changes.md`.

---

<a id="d-35"></a>
## D-35 — `verifier_rung1`: incubator-verifier extensions for FTA-ladder rung-1 forward direction (2026-05-03)

**What.** A coordinated extension of `verifier.py` to clear the 11 verifier failures that surface when the verifier is pointed at `files/incubator/processed_proof_graph/` (the historical `verifier.py` `main` hardcoded only the main pipeline output, so the incubator regression was invisible). All 11 failures stem from the FTA-ladder rung-1 forward direction (`EnumerationSet2 ⟹ interval`) — chapter `1209_direct_proof.txt` and its reformulated companion `1210_reformulated_statement.txt`.

The extension consists of seven logical edits, lettered V-1..V-7. **This entry is seeded with V-1 + V-7 + the alpha-canonicalize helper; subsequent commits on this branch extend it as V-2..V-6 land.** Per Rule 10 each commit updates this entry as it lands its slice.

**V-1 — CLI base-dir argument.** `main` now accepts a positional `base_dir` (default = the historical `files/processed_proof_graph`). Same default behaviour for old invocations; new invocation `python verifier.py files/incubator/processed_proof_graph …` verifies the incubator output.

**V-7 — `--include-globals PATH` (repeatable).** Unions sibling-batch `global_theorem_list.txt` entries into `state.global_theorems`/`state.global_theorem_list` so cross-batch theorem citations in the origin check (e.g. an incubator chapter citing the Peano `existence2` axiom) resolve. Entries are loaded in supplied order; local-batch entries take precedence on key collisions.

**V-2 — `check_or_convergence` deliberately fails until producer-side evidence lands (clean-fail final state, post-Codex-review).** Three iterations:

1. **Initial.** Added a compiled-form `(or<N>[…])` path validating only OR shape + parent-scope namespace match. Cleared the 2 `or convergence` failures in chapter `1209_direct_proof.txt`.
2. **Strengthened.** Codex review correctly flagged the initial check as too weak under [I-16](30_invariants.md#i-16). Added "OR must be live at parent scope" to both expanded and compiled paths; documented residual gap as a `Suspected fragility`. Still cleared the 2 failures.
3. **Clean-fail.** The strengthened check still PASSED the 2 rows — but on accident, because the OR happened to be derived right before each convergence row. Per I-16, when a check cannot verify its semantic contract the correct response is FAIL, not PASS-with-asterisk. `check_or_convergence` returned `False` unconditionally for both `rest[0]` shapes. The 2 chapter-1209 rows surfaced as `or convergence failure 2`.
4. **Spec'd layout (this entry).** Per the user-directive of 2026-05-03 the new producer-side row layout is fixed:
 ```text
   <C>  <parent>  or convergence  <OR>  <parent>  <C>  <branch_D1>  <C>  <branch_D2>  …  <C>  <branch_DK>
   ```
 `check_or_convergence` is rewritten to validate this layout: layout shape (`len(rest) == 2 + 2*K`), parent-scope match, OR is a known compiled `(or<N>[…])` with `K` disjuncts, conclusion repetition, branch-scope ancestry, branch distinctness, and the load-bearing **per-branch derivation evidence** — for every `(C, branch_Di)` pair the chapter must contain a row with `expression == C` and `namespace == branch_Di` (the user's "each ingredient has its own line" requirement). The check is now real verification, not a structural pass-through.

 The 2 chapter-1209 rows still use the old 4-field layout and continue to fail at the layout check (step 1: `len(rest) == 4`, not `>= 6`). Same end-state count (`32779/2`), but the failure now means "old layout, new layout pending" rather than "unconditional reject". Producer-side fix (next task) makes the rows pass.

**Why the chapter-local check WAS fundamentally insufficient before this revision.** The contract is "the same conclusion `C` was independently derived in EVERY branch of this OR's case split". The chapter export historically doesn't carry that evidence: `ordisMerge` removes the per-branch copies of `C` at convergence (`07_or_branching.md` §3b), and `process_proof_graphs.py` does not retain `_boundary_ordis_` rows. The new spec'd layout closes the gap by carrying the per-branch evidence into the row's rest fields AND requiring the chapter to retain the per-branch derivations of `C`.

**Producer-side fix — the next task ("buildstack and history tracking").** Two coordinated changes must land together:

a. **Prover.** Emit the new convergence row layout. `ordisMerge` (or its equivalent at the prover's emit site) records the per-branch validity names alongside the converged conclusion and writes them into the row's rest fields.
b. **Process-proof-graph.** Retain per-branch derivation rows of `C` in chapter export — i.e. don't suppress chapter rows whose namespace is a branch scope and whose expression is the converged `C` cited by an `or convergence` row. Simplest implementation: walk the convergence rows first, collect the cited `branch_Di` namespaces, then suppress only branch rows that are NOT cited.

Either change without the other leaves the verifier failing.

**V-3 + V-4 — `or branch proven` and `or branch assumption` promoted to first-class `TAG_CHECKERS` entries.** Both tags were previously claimed retired/overridden in the SwDD; the override path never existed and the prover always emitted them live. FTA-rung-1 chapter `1209_direct_proof.txt` lines 22 (`or branch proven`) and 43 (`or branch assumption`) carry the tags, and pre-extension they showed up as `<unknown:…>` failures. The new checkers validate:

- **`check_or_branch_proven`.** `line.expression` = the OR (compiled `(or<N>[…])`); `line.namespace` = the OR's parent scope; `rest[0]` = the asserted disjunct; `rest[1]` = the branch's namespace. Validation requires parent-as-strict-ancestor of branch, GL-binary disjunct membership of `rest[0]` (modulo equality symmetry), and the branch payload to encode the OR + asserted disjunct via the `_boundary_orint_<or>_(<disjunct>)` substring.
- **`check_or_branch_assumption`.** `line.expression` = `!<other-disjunct>`; `line.namespace` = the branch's namespace; `rest[0]` = `<or>_integration_goal`; `rest[1]` = the OR's parent scope. Validation requires parent-as-strict-ancestor of branch, the OR-with-suffix to strip cleanly to a known compiled OR, the negated content to be a disjunct of the OR (modulo equality symmetry), and the branch's `_boundary_orint_<or>_(<asserted>)` payload to name a DIFFERENT disjunct (the one the branch asserts).
- A shared helper `_or_disjuncts_from_compiled` substitutes `u_i` placeholders in the OR binary's elements list with the OR's args; `_disjunct_matches` adds equality-symmetric matching for `(=[a,b])` ↔ `(=[b,a])`.
- Both tags are added to `_ORIGIN_EXEMPT_TAGS` so the inline origin check does not require chapter-LHS membership for `_integration_goal`-suffixed deps or for asserted disjuncts whose own derivation lives in a different chapter.

**V-5 — `check_implication` accepts ancestor-scope premises (comparable-scope inheritance).** The pre-extension rule was "at most one distinct non-main namespace among premises + implication, and the result must equal that one". This rejected legitimate FTA-rung-1 firings (chapter `1209_direct_proof.txt` lines 36, 41, 56, 73) where the result lands in an OR-branch scope but some premises live at the OR's parent scope. The new rule iterates each source namespace and accepts if it is `"main"`, equal to the result's namespace, or a strict ancestor of it (`result_ns.startswith(ns + "_boundary_")`). This is the faithful encoding of GL's comparable-scope inheritance — facts at an ancestor scope are visible at every descendant.

**V-6 — `_check_reformulation` binary-lookup fallback for split-tag anchors.** The helper derives the GL-binary tag from the target's anchor (e.g. `AnchorGauss → "Gauss"`) and selects `gl_binaries[tag]`. The incubator's GL-binaries are split across multiple tag files (`IncubatorPeano` / `IncubatorGauss` / `IncubatorGauss1`); chapter `1210_reformulated_statement.txt`'s anchor `AnchorIncubator` derives the literal tag `"Incubator"` for which no binary is loaded, so the lookup returned `None` and the check rejected immediately. The new behaviour: if the exact-tag lookup is empty or the head core is missing/non-existence in it, the helper scans every loaded binary for one that defines the head's compiled name as an `existence` entry. The fallback is purely additive — exact-tag lookups still take precedence for batches where they succeed (e.g. all main-pipeline runs).

**Codex round-3 tightening (post-round-2, 2026-05-03).** A third Codex review surfaced two cross-row soundness gaps in the OR-branch checkers — both addressed in one commit:

1. **`check_or_branch_proven` requires an independent OR-derivation row at parent scope.** Pre-tightening the checker validated structure but not provenance: the bookkeeping split row could appear in isolation and pass. Tightened: a chapter row must exist with `expression == or_expr`, `namespace == parent_ns`, and `tag!= "or branch proven"`. **Surfaces a new deliberate failure on chapter 1209**: the OR `(or2[i1,v5,i0])` is consumed as a premise by line 19's implication but never appears as a chapter LHS at the parent scope (the chapter export does not render the consumed OR's derivation). Per the project conventions `Failures are first-class` this is a feature — the failure is a forcing function for the next task to fix the chapter export (or for the prover to emit an explicit OR-derivation row).
2. **`check_or_branch_assumption` requires a matching `or branch proven` row.** Pre-tightening the checker validated structure but not the case-split's existence: an assumption row could pass even if the branch was never opened by a `or branch proven` row. Tightened: a chapter row must exist with `tag == "or branch proven"`, `expression == or_expr`, `namespace == parent_ns`, `len(rest) == 2`, `rest[1] == branch_ns`, and `rest[0]` matching the asserted disjunct (modulo equality symmetry). Chapter 1209's assumption row at line 43 continues to pass — line 22 satisfies all sub-conditions.

**End-state count after round-3.** Incubator `32779 checks, 3 FAILED` (was 2 after round-2): the 2 pre-existing deliberate `or convergence` fails plus 1 new deliberate `or branch proven` fail surfaced by round-3 step 1. Main pipeline `2750 checks, 0 failures` (unchanged).

**Codex round-2 tightening (post-spec'd-layout, 2026-05-03).** A second Codex review surfaced three soundness gaps in the `or branch proven` / `or branch assumption` checkers and the `_or_disjuncts_from_compiled` helper. All three fixed in one commit:

1. **`len(rest) == 2` exactly** for both `check_or_branch_proven` and `check_or_branch_assumption`. Both tags sit in `_ORIGIN_EXEMPT_TAGS`, so any extra `(expression, namespace)` rest pairs were silently accepted by the generic origin check — no audit trail for hidden ingredients. The pre-tightening checks used `len(rest) >= 2`; tightened to `==`. Empirically chapter 1209's two rows have exactly 2 rest fields, so no regression.
2. **Exact branch-namespace match** in both checkers. The pre-tightening code used substring search (`needle in branch_ns` for `check_or_branch_proven`, `find` for `check_or_branch_assumption`), which would PASS a nested or unrelated descendant scope that merely contained the expected `_boundary_orint_<or>_(<disjunct>)` substring. The contract says branch is at exactly `parent + "_boundary_orint_<or>_(<disjunct>)"` — tightened to literal equality on the full namespace string (with a balanced-parens parser to extract the asserted disjunct in `check_or_branch_assumption`).
3. **Arity check** in `_or_disjuncts_from_compiled`. The pre-tightening code did not verify `len(args)` against the binary's `arity` field; a malformed `(or<N>[…])` with too many or too few args was treated as a valid known OR if the disjunct placeholders happened to substitute. Tightened: read `arity` from the binary entry (or parse `signature` if `arity` absent), reject when `len(args)!= arity`. Used by both `check_or_branch_proven`, `check_or_branch_assumption`, and the spec'd-layout `check_or_convergence`.

**End state (corrected through Codex round-3).** With V-1..V-7 + the `_alpha_canonicalize_bound_vars` helper + the V-2 clean-fail revision + the V-2 spec'd-layout revision + Codex round-2 tightening + Codex round-3 tightening:

- Main pipeline: **2750 checks, 0 failures** (no regressions; main does not currently emit any `or convergence` / `or branch proven` / `or branch assumption` rows).
- Incubator: **32779 checks, 3 FAILED**:
 - 2 × `or convergence` (chapter 1209 lines 5, 14) — old 4-field layout, awaiting spec'd-layout producer-side fix.
 - 1 × `or branch proven` (chapter 1209 line 22) — OR not derived at parent scope as a chapter LHS, awaiting chapter-export fix to render the OR's derivation row.

8 of the 11 baseline failures genuinely cleared via real verifier-side checks (4 `implication`, 1 `<unknown:or branch assumption>`, 1 `origin`, 2 reformulated_statement). The 3 remaining are deliberate forcing functions for the next task ("buildstack and history tracking"). Per the project conventions `Failures are first-class` ("Failures are first-class") these failures are accepted as valuable signal, not softened to recover "0 failures". The honest answer to "is the FTA-rung-1 forward chain verifier-clean?" is **no, not yet** — three real verification gaps are surfaced and need producer-side work to close. Pretending otherwise (the original "32779/0 airtight" claim) would have been a violation of [I-16](30_invariants.md#i-16) in spirit. The `verifier_rung1` task is closed in the corrected state.

**Alpha-canonicalize helper.** A new `_alpha_canonicalize_bound_vars` helper canonicalizes every `>[…]` bound-variable name to `b1, b2, …` in declaration order. Used by the inline origin check at `verifier.py` so a chapter row whose `rest[0]` rule names a bound variable `i2` (the prover's local free-index counter at deposit time) matches a global-list entry that names the same bound variable `v1` (process_proof_graphs.py's canonical-export rename). Pre-extension, `_normalize_expr_list` (which only renames `v\d+`) treated the two as distinct strings and the origin check rejected the legitimate citation.

**Why this branch exists.** Rung 1's forward closure (D-34, sandbox/ordis_merge) introduced OR-disintegration machinery whose chapter rows use proof-tag patterns that the verifier had never been exercised against — most notably the live (not, as the SwDD claimed, retired) tags `or branch proven` / `or branch assumption`, OR-branch ancestor-scope premise inheritance in `implication`, the compiled-form `(or<N>[…])` argument shape in `or convergence`, and the 4-argument-head `existence4` reformulation in `_check_reformulation`. None were caught earlier because the verifier was never pointed at the incubator tree. Per [I-16](30_invariants.md#i-16) the response is to extend the verifier with proper checks, not to suppress.

**Cross-checks.**
- `python verifier.py` against the main pipeline: 0 failures (no regression).
- `python verifier.py files/incubator/processed_proof_graph --include-globals files/processed_proof_graph/global_theorem_list.txt` against the incubator: 11 → 0 failures (target end-state, reached as V-2..V-6 land).

**Files touched (this entry).** `verifier.py`, `docs/10_pipeline/08_verifier.md`, `docs/40_decisions.md`. (Subsequent commits on this branch will also touch `docs/20_core_concepts/07_or_branching.md`, `docs/20_core_concepts/08_proof_tags.md`, and add `docs/verifier_rung1_changes.md` as the per-edit changelog.)

---

<a id="d-34"></a>
## D-34 — `_ordis_` merge: kernel-level `ordisMerge` over `internalMailIn` (2026-05-02)

**What.** OR-disintegration convergence bookkeeping moves from `trackOrBookkeeping` (a stale inline member previously called from inside [`addStatement`](../GL_Quick_VS/GL_Quick/src/prover.hpp)) to a new inline member [`ordisMerge`](../GL_Quick_VS/GL_Quick/src/prover.hpp) called from [`addExprToMemoryBlockKernel`](../GL_Quick_VS/GL_Quick/src/prover.cpp)'s post-`addStatement` loop, sibling to the `toBeProved`-discharge logic and to the `_orint_/NotOrScope` block. The old `trackOrBookkeeping` function is deleted.

`ordisMerge` runs once per `(addExpression, effectiveValidity)` pair returned by `addStatement`. With [D-33](#d-33)'s single-channel routing, `effectiveValidity` is the deposit's actual scope — so the function observes per-branch deposits including descendant-direction cross-scope rewrites that the legacy `addStatement`-call-site bookkeeping never saw. It records each deposit in `Memory::orBookkeeping[(expr, orSignature)] → set<branchDisjunct>` and tests against `Memory::orDisjunctCount[orSignature]` (registered at OR-disintegration mint time, [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)). On convergence:

1. **Promotion via `internalMailIn`.** `(addExpression, addExpressionLevels, parentValidity)` is pushed onto `Memory::internalMailIn.statements` — the same revival channel `revisitRejectedIntegration2` uses ([`memory.hpp` `InternalMail`](../GL_Quick_VS/GL_Quick/src/memory.hpp)). Drained at the top of the next hashburst body via `addExprToMemoryBlock(..., status=1,..., parentValidity,...)`, so the parent-scope deposit goes through the full kernel pipeline (parent's `toBeProved` discharge included). Deferring to `internalMailIn` instead of calling `addExprToMemoryBlock` directly avoids reentrant kernel invocation in the post-`addStatement` loop.
2. **Per-branch cleanup of the converged expression.** For each branch validity in the recorded disjunct set, `ordisMerge` calls `removeExpressionFromMemoryBlock(EncodedExpression(addExpression, branchValidity), mb, /*state=*/0)`. The N per-branch copies of the converged expression collapse into the single parent-scope copy. The parent-scope copy remains visible to each branch via comparable-scope inheritance; branches reuse it through standard scope-walk reads. Branches keep all their *other* facts and stay live (unlike `_orint_`'s `cleanUpOrIntegrationBranches` which wipes the entire branch).

`exprOriginMap` (history) is not touched — `removeExpressionFromMemoryBlock(state=0)` only erases the encoded-statement vectors. `statementLevelsMap` and `intKnownStatements` are also retained, so a re-deposit attempt at a wiped branch scope short-circuits at the standard `intKnownStatements` gate at `addStatement`. `orBookkeeping` is also retained — re-fires for the same `(expr, orSig)` on a subsequent deposit (e.g. when the user re-runs the same case via a different path) are absorbed by the `std::set` dedup of `internalMailIn.statements`.

**Why — visibility gap.** FTA-rung-1 §9a / §9b symptom (`fta_ladder/rung1/current_proof_state.md`): all four ordis cells under impl24 / impl25 contain both per-branch preorder facts (D-33 working — descendant-direction eq-class rewrites land at the branch scopes), but the OR-convergence promotion to the immediate impl24 / impl25 boundary scope was never firing. Diagnosis: `trackOrBookkeeping` was called from inside `addStatement` at line 6126 with the *original* `validityName` parameter the caller passed in. After D-33 made `applyEquivalenceClass` push descendant-direction cross-scope rewrites to `newStatements` at the deeper deposit scope, those branch-scope deposits never reached the bookkeeping — they landed in `encodedStatements` but the convergence map never saw them. Moving the bookkeeping to the kernel's per-pair loop closes the gap structurally: the kernel iterates pairs at their deposit scope, so every branch-scope deposit is observed.

**Why — `internalMailIn` route.** The user's design directive: "we have internal mail. revisitRejectedIntegration uses. let's put stuff after merge there. it prevents endless recursion and is our future general path." `internalMailIn` is the existing per-LB validity-preserving inbox for integration-side revival messages, populated by `applyEquivalenceClassToRejectedMapIntegration` and `revisitRejectedIntegration2` during a hashburst body and drained at the top of the next hashburst's body. Using it for `_ordis_` convergence promotion gives three properties for free:

- **No reentrance.** `addExprToMemoryBlockKernel` does not call itself recursively during the post-`addStatement` loop iteration. The deferred drain runs the parent-scope deposit through a clean kernel entry at the next hashburst.
- **Full kernel pipeline at the parent scope.** The `status=1` absorb path runs the parent-scope deposit through `toBeProved` discharge, the `_orint_/NotOrScope` block, *and* `ordisMerge` again at the parent scope — so a parent-scope deposit that itself sits inside another `_ordis_` chain participates in further convergences cleanly.
- **Set-dedup absorbs idempotent re-fires.** `internalMailIn.statements` is `std::set<std::tuple<std::string, std::set<int>, std::string>>`; pushing the same tuple twice is a no-op.

**Why — narrow cleanup.** The legacy `trackOrBookkeeping` cleanup wiped *every* statement at the branch scopes via prefix-match and added the branch validities to `validityNamesToFilter` to block all future inserts. That was strictly wrong for `_ordis_`: a case-split over a known disjunction has every branch contributing usable derivations under its case condition; wiping siblings discards those. Worse, the filter-block prevented further convergences — once one expression converged, the OR scope was dead for any other expression. The narrow cleanup keeps branches alive so the second preorder of the same impl scope's body (`(preorder[1,4,p,1])` after `(preorder[1,4,0,p])` for impl24, etc.) can converge in turn.

**Verification.** Pipeline-run verification target: full `main.py`, both Gauss summation copies present in `proved_theorems.txt`, and the four target FTA-rung-1 §9a / §9b preorder rows present at the parent scopes (`(preorder[1,4,2,repl_lev_1_0])` and `(preorder[1,4,repl_lev_1_0,6])` at `main_boundary_(implication24[15,1,4,2])`; `(preorder[1,4,2,repl_lev_1_1])` and `(preorder[1,4,repl_lev_1_1,6])` at `main_boundary_(implication25[15,1,4,6])`).

**Verified — full main.py run HEAD (post-mailOut gate fix).** Exit 0, runtime 1266 s, verifier 2750 checks 0 failures airtight. Per-batch theorem counts: IncubatorPeano 513, Peano 49, IncubatorGauss 91, IncubatorGauss1 **2 (vs baseline 0)**, Gauss 11. Both Gauss summation copies in `files/theorems/proved_theorems.txt` (`fold[…,5]`-anchored implications, 41 theorems total = D-33 baseline parity). IncubatorGauss1 saves the §4.1 forward direction `(>[1,2,4,6](AnchorIncubator[…])(>[15](EnumerationSet2[2,6,15])(interval[1,4,2,6,15])))` — first time the FTA-rung-1 `{0,1}=[0,1]` forward conjecture has closed end-to-end.

Three of the four §9a / §9b target preorders land at the clean immediate-parent scope (impl24-up: 50 hits, impl25-up: 46 hits, impl25-down: 50 hits in the trace). The fourth (`(preorder[1,4,repl_lev_1_0,6])` at impl24-down) lands at v=`main` instead of impl24: the orSignature `(or2[2,repl_lev_1_0,6])` exists at two stack positions (impl24-internal AND top-level main), the top-level OR-convergence promotes to `main` first, then the impl24-internal convergence's `internalMailIn` deposit gets dedupe'd by Site F's ancestor scan at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) because the fact is already in `intKnownStatements` at the strict ancestor v=`main`. Comparable-scope inheritance makes the `main` entry visible at impl24, so the §4.1 forward chain closes regardless. `or convergence` origins appear 106 084 times in the IncubatorGauss1 hashburst trace — the mechanism fires extensively.

**Mail-out gate (added with the same change set).** [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)'s `mailOut.implications` insert is now gated on `impValidity == "main"` in addition to the pre-existing `allowedForMail` check. The mailOut.implications/statements channels are MAIN-ONLY by contract (mailOut.exprOriginMap continues to carry history for all scopes per `trackExpansionHistory`); the old code shipped non-main rules unconditionally, the receiver re-installed them at hardcoded v=main with the rule's origin still keyed at the sender's deeper scope, and visualizer's `buildStack` walked the firing-time dep at `(rule, "main")` with no matching exprOriginMap entry. With the gate, non-main rules stay local; receivers re-derive them from the mailed v=main statements via their own disintegration, yielding properly-scoped origins via `trackExpansionHistory`.

**Code.**
- `ordisMerge` definition — [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp), inline member of `ExpressionAnalyzer`, around the location where `trackOrBookkeeping` lived.
- Kernel call site — [`prover.cpp` `addExprToMemoryBlockKernel`](../GL_Quick_VS/GL_Quick/src/prover.cpp), single line at the bottom of the post-`addStatement` `for (idx...)` loop.
- `trackOrBookkeeping` deleted (function body + `addStatement` call site at the old line ~6126).
- `Memory::orBookkeeping` and `Memory::orDisjunctCount` (declared in [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp)) — fields retained, semantics unchanged. `Memory::internalMailIn` — used by `ordisMerge` exactly the same way `revisitRejectedIntegration2` uses it.

**Supersedes.** The `addStatement`-call-site `trackOrBookkeeping` design. Both the placement (now kernel-level) and the cleanup shape (now narrow) change.

---

<a id="d-33"></a>
## D-33 — Bidirectional equivalence-class application + single-channel deposit routing (2026-05-01)

**What — bidirectional eq-class application.** Equivalence-class application now runs in both directions whenever the class's validity and the expression's validity are comparable in the scope tree. Three cases admit:

| Class scope | Expression scope | Deposit scope |
|---|---|---|
| `S` | `S` | `S` |
| `S_a` (strict ancestor of `S_d`) | `S_d` | `S_d` |
| `S_d` (strict descendant of `S_a`) | `S_a` | `S_d` |

The deposit scope is `deeperOf(class.scope, expr.scope)` — see [`memory.hpp::NameMap::deeperOf`](../GL_Quick_VS/GL_Quick/src/memory.hpp). For the legacy directions (same / class-shallower) the deposit collapses to the expression's own scope; for the new direction (class strictly deeper) the deposit lands at the class's deeper scope. The original at the ancestor scope is never overwritten — the rewrite is purely additive. Both copies coexist.

Trigger sites: [`updateEquivalenceClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp)'s merge-postscan (walks every encoded statement, applies merged class to comparable-scope statements), [`addStatement`](../GL_Quick_VS/GL_Quick/src/prover.hpp)'s per-statement block (iterates classes at every comparable scope of the new fact's scope), and the same fixpoint repetition for late-appearing classes.

**What — single-channel deposit routing.** Every deposit — same-scope, shallower-class, deeper-class — flows into the single `newStatements` vector that `addStatement` returns. There are no separate sinks. Each entry is an [`ExpressionWithValidity`](../GL_Quick_VS/GL_Quick/src/memory.hpp) pair carrying the deposit's actual scope.

The kernel's post-`addStatement` loop ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)) iterates the pairs and uses each pair's `validityName` for `statementLevelsMap` lookup, admission-map updates, `toBeProved` discharge, and validity-name promotion. Cross-scope deposits go through the same discharge logic as same-scope deposits — the only difference is which `validityName` drives the lookup. See [I-25](30_invariants.md#i-25).

**What — rejected-integration map exclusion.** [`applyEquivalenceClassToRejectedMapIntegration`](../GL_Quick_VS/GL_Quick/src/prover.hpp) does NOT admit the descendant direction. The rmi map is a deferred-match registry — its entries represent auxy integrations whose preconditions live in the world visible at the entry's scope. A class at a strictly deeper scope is invisible there, so it carries no information for that entry's deferred match; rewriting the entry would manufacture a synthetic deferred match that did not actually defer at that scope, and erasing the original would destroy a real revival path at the ancestor.

**Why — bidirectional application.** FTA-rung-1 §9b (upper-bound forward conjunct) and §9b-style cases need a class registered at an OR-disintegration branch scope (descendant of `main`) to rewrite a ground `preorder` fact at `main`. Pre-D-33, eq-class application only walked descendants of the class scope, so a deeper class never reached up into ancestor-scope statements. The new direction closes the FTA-§9b gap by letting the descendant-scope class produce the rewrite at its own scope.

Soundness rests on descendant-inheritance: a fact at `S_a` is observably true at every descendant of `S_a`, including `S_d` where the class lives. Substituting under the descendant-scope equivalence and depositing at the descendant scope is locally sound at the descendant.

**Why — single-channel routing.** An earlier WIP attempt at D-33 (the original bad commit ) routed cross-scope deposits to separate sinks (`crossScopeSink` in `addStatement`, `descendantSink` / `ancestorSink` in the merge-postscan) to avoid tripping the kernel's same-scope `assert(sit!= memoryBlock.statementLevelsMap.end)` ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)) — the lookup was built as `EncodedExpression(addExpression, validityName)` with `validityName` = the kernel's caller-scope, and a cross-scope deposit at a different scope key would miss.

The side-sink design avoided the assert but bypassed the kernel's entire post-loop, including the `toBeProved` discharge at all five sites (recursion-LB head match, direct-theorem proof, validity-name promotion for OR-integration, validity-name promotion for NotOrScope, vacuous-truth in induction). Cross-scope deposits sat in `statementLevelsMap` and `intKnownStatements` but never closed their matching `toBeProved` entries.

Concretely on the Gauss summation induction-condition LB: the boundary fact `(in2[2,repl_lev_4_0,int_lev_4_2349]) @ main_boundary_(implication23[2,8,int_lev_4_2349])` was synthesized via the merge-postscan ancestorSink (equality1 from main-scope source using boundary class `{2,repl_lev_4_0}`). The fact lived in `encodedStatements`. The matching `toBeProved` entry stayed open. The validity-name promotion that would have lifted `(implication23[2,8,int_lev_4_2349])` to `v=main` never fired. The IH-application chain cascaded as MISSING. Gauss summation stayed unproved.

The pair-based return type for `addStatement` carries each deposit's scope into the kernel, so the lookup uses the deposit's own validity, the assert holds, and the discharge logic runs. That single change closes the cross-scope discharge gap without sacrificing the FTA-§9b unblock.

**Why — rmi exclusion.** The rmi map is keyed by `(expression, validity)` and lookup happens against entries the prover registered as deferred. A descendant-scope equality applied to an ancestor-scope rmi entry would synthesize a key the descendant scope never deferred (and may never need); writing it manufactures rmi state out of nothing. The legacy `toErase.push_back(keyEv)` (replace-not-add) compounded the problem by destroying the ancestor's revival path. Restricting rmi application to same-scope and class-shallower keeps the rmi pool a faithful record of deferred matches per scope.

**Verification.**

- Pre-D-33 (commit baseline, no-incubator full main.py): 41 proved theorems, both Gauss summation copies present, verifier 2758 checks 0 failures.
- WIP D-33 (commit, no-incubator full main.py): 39 proved theorems, both Gauss summation copies MISSING, verifier 2484 checks 0 failures (passes only because the missing-Gauss checks don't run).
- Final D-33 (commit, no-incubator full main.py): 41 proved theorems, both Gauss summation copies present, verifier 2750 checks 0 failures — airtight. Slight check-count delta vs the baseline (2750 vs 2758) reflects path differences in how some derivations route through the new pair-based discharge but does not indicate theorem loss.
- FTA-rung-1 §9b unblock: see in-flight `fta_ladder/rung1/current_proof_state.md` and trap dumps under the `EnumerationSet2[2,6,15] → AnchorIncubator` LB in the fullest pipeline run.

**Code.**

- `applyEquivalenceClass`: depositValidity computation at function entry — [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp). Parameter type `std::vector<ExpressionWithValidity>& newStatements`.
- `applyEquivalenceClassToNegatedEquality`: same parameter-type change — [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp).
- `cleanUpExpressions`: takes/returns `std::vector<ExpressionWithValidity>` — [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp).
- `updateEquivalenceClasses`: takes/returns `std::vector<ExpressionWithValidity>`; merge-postscan walks every encoded statement, single-channel emit — [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp).
- `addStatement`: returns `std::vector<ExpressionWithValidity>`; descendant-classes block and fixpoint descendant pass route to `newStatements` — [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp).
- `applyEquivalenceClassToRejectedMapIntegration`: scope-match gate restricted to same-NS and class-shallower — [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp).
- Kernel post-`addStatement` loop iterates pairs and uses `effectiveValidity` from each pair throughout — [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp).
- `NameMap::deeperOf` helper for strings — [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp).
- Verifier `equality1` source-namespace relaxation (admits source at strict ancestor of result) — [`verifier.py::check_equality1`](../verifier.py).

**Supersedes.** Bad commit 's WIP design with separate cross-scope sinks. The user-facing consequences of that design (Gauss regression, cross-scope discharge gap) are gone; the FTA-9b unblock is preserved.

---

<a id="d-32"></a>
## D-32 — Sharper OR-disintegration gate: product-of-disintegration head + checkLocal-routed (2026-05-01)

**What.** Replaces D-31's broad implication-scope OR-disintegration bypass with a sharper gate. OR-disintegration in [`disintegrateExprCore2`](../GL_Quick_VS/GL_Quick/src/prover.cpp)'s OR case is now allowed iff a new boolean parameter `allowOrDisintegration == true` is threaded into the call. The flag is set by [`checkLocalEncodedMemoryStatic`](../GL_Quick_VS/GL_Quick/src/prover.cpp) at the head-firing call site, fed from a new `LocalMemoryValue::productOfDisintegration` field stamped at install time inside [`addToHashMemory`](../GL_Quick_VS/GL_Quick/src/prover.cpp). The stamp criterion is: at least one premise (chain element) of the implication has an argument whose name starts with `"u_"`.

The `"u_"` prefix is reserved for bound-variable placeholders introduced by `prefixArgumentsWithU` during `disintegrateExpr2`. An anchor-bound theorem's chain (e.g. starts with `(AnchorPeano[1,2,3,...])`) carries only concrete integer args; its premises never have `u_*` args, so the stamp is `false` and the bypass never fires. A body-element implication produced by disintegrating a compound (e.g. ES2-forward `implication20`: `(in[u_p, u_M]) ⇒ (or2[u_2, u_p, u_6])`) carries `u_*` args, gets stamped `true`, and triggers the bypass when its head deposit is an OR.

Coupling with general disintegration: if `addExprToMemoryBlock`'s `doNotDisintegrate == true`, `allowOrDisintegration` is forced to `false` at function entry. OR-disintegration cannot fire when general disintegration is forbidden.

The legacy `orAdmissionSet` fallback gate (line ~7191) is preserved as a no-op extension hook — `orAdmissionSet` still has no `.insert` site anywhere in the codebase.

**Why.** D-31's broad bypass — "any OR landing in any implication scope disintegrates unconditionally" — caused a runtime explosion. The IncubatorGauss1 hashburst trace under D-31 contained 13 191 `_boundary_ordis_(or…)` rows at `(implication24[15,1,4,2])` scope, plus more elsewhere; the cascade fanned out across every implication scope in the batch. Anchor-bound theorems (P1 left-identity, P2 successor, etc.) firing into implication scopes would each spawn case-split branches even when the OR was a known concrete instantiation with no actual case-split needed. Full main.py runtime ballooned beyond the D-30 baseline of 1183 s.

The narrowing: only ORs delivered by an implication that is *itself* a product of disintegration (= came from disintegrating a compound's body, not a top-level anchor-bound theorem) trigger the bypass. The dominant FTA-rung-1 case — `implication20` (ES2-forward) firing at `(implication24[15,1,4,2])` scope and depositing `(or2[2, repl_lev_1_0, 6])` — survives because `implication20`'s chain `(in[u_p, u_M])` has `u_p`/`u_M` args. Anchor-bound rules' broad fan-out is filtered out.

**Verification (full main.py + IncubatorGauss1 standalone hashburst).**

- Steps 1–8 + Step 9 lower (impl24) of the FTA-rung-1 `{0,1}=[0,1]` proof all reach the same trace evidence as under D-31. Burst-by-burst toBeProved trajectory `6 → 5 → 3 → 2` (bursts 1–8 → 9 → 10 → 11+) is identical to D-31's. The Step-9-lower closure (`(preorder[1,4,2,repl_lev_1_0])` deposited at `(implication24[15,1,4,2])` scope, 42 occurrences) is unchanged. The `_boundary_ordis_` rows at impl24 dropped from 13 191 (D-31) to 1 501 (D-32) — an 88% reduction in OR-disint fan-out at that scope while preserving the closure path.
- Per-batch theorem-count parity vs D-31: see commit message for the 5-batch totals.
- Verifier: see commit message for check counts and failure totals.
- Full main.py runtime: see commit message; target is the D-30 baseline (~1183 s).

**Code.** Gate site: [`prover.cpp` `disintegrateExprCore2` OR case](../GL_Quick_VS/GL_Quick/src/prover.cpp) (around line 7180, replaces D-31's `IMPL_PREFIX` peel block). Stamp site: [`prover.cpp` `addToHashMemory`](../GL_Quick_VS/GL_Quick/src/prover.cpp) (around line 1131, immediately before the `encodedMap[intIgnoredKey].push_back(lmv)`). Forwarding: [`prover.cpp` `addExprToMemoryBlock`](../GL_Quick_VS/GL_Quick/src/prover.cpp) → `disintegrateExpr2` → `disintegrateExprCore2`, all with `allowOrDisintegration` default-`false` so the 28+ existing call sites of `addExprToMemoryBlock` and the 2 of `disintegrateExpr2` need no edit. Field declaration: [`memory.hpp` `LocalMemoryValue`](../GL_Quick_VS/GL_Quick/src/memory.hpp) (around line 84).

**Supersedes.** D-31. The implication-scope bypass at `disintegrateExprCore2:7146` no longer exists; the legacy `orAdmissionSet` fallback gate stays.

---

<a id="d-31"></a>
## D-31 — Implication-scope OR-disintegration bypass — closes Step 9 lower (impl24) (2026-04-29 evening,) — SUPERSEDED by D-32 on 2026-05-01

**What.** Adds a bypass at the OR-disintegration admission gate inside [`disintegrateExprCore2`](../GL_Quick_VS/GL_Quick/src/prover.cpp) (around line 7146). Pre-D-31 the gate consulted `Memory::orAdmissionSet`, which has no `.insert` site anywhere in the codebase — so `orAdmitted` was permanently `false` and OR-disintegration never fired (zero `or disintegration` verifier successes, zero `_boundary_ordis_` scopes in any pre-D-31 trace). The D-31 bypass: if the OR arrives at a validity scope whose last `_boundary_<payload>` segment matches `(implicationNN[…])`, set `orAdmitted = true` unconditionally. Detection peels the last boundary segment via `validityName.rfind(NameMap::BOUNDARY_STR,...)` and string-compares the prefix `"(implication"`. The legacy `orAdmissionSet` gate remains as a no-op fallback.

**Why.** FTA-rung-1 Step 9 lower-bound conjunct (impl24's preorder body `(in[p, M]) ⇒ preorder(0, p)`) was stuck. Path: ES2-forward (`implication20`) deposits `or2[2, p, 6]` at impl24 scope; OR-disintegration should mint two `_ordis_` branches with seed equalities `(=[2, p])` and `(=[6, p])`; each branch substitutes into the preorder body via P1 left-identity to derive `preorder(0, 0)` / `preorder(0, 1)`; ordis convergence promotes `preorder(0, p)` to impl24 scope. Step 1 of the chain (ordis branch minting) never fired because of the structurally-dead `orAdmissionSet` gate.

**Verification (full main.py + 35 458-check verifier — done at commit time).**

- Step 9 lower toBeProved `(preorder[1,4,2,repl_lev_1_0])` at impl24 scope closes — burst-trace toBeProved trajectory drops 4 → 3 in the IncubatorGauss1 burst trace. Closure path: preorder integration with P1 witness `k = p` + variable-copy primitive at impl24 scope (not via the case-split, but enabled by the broader exploration that D-31 unblocks).
- 27 530 `_boundary_ordis_` scope occurrences in the IncubatorGauss1 hashburst trace (was 0 pre-D-31).
- Verifier: 35 458 checks, 0 failures across both proof graphs (main 32 700 / 0, incubator 2 758 / 0). No regression on Peano OR-branching, Gauss main, or any other batch.
- Per-batch theorem counts unchanged (513 / 49 / 91 / 0 / 11) — IncubatorGauss1 still saves 0 because Step 9 upper (impl25's preorder upper conjunct) and Step 10 (top-level interval) remain open.

**Why superseded.** D-31's bypass fires for *every* OR that lands in *any* implication scope, including ORs delivered by anchor-bound theorems. Runtime explosion observed in subsequent IncubatorGauss1 / full main.py runs. D-32 narrows the gate to ORs delivered by implications that are themselves products of disintegration (premise has a `u_*` arg).

---

<a id="d-30"></a>
## D-30 — OR-integration emits the wrapping OR at the OR-branches' parent scope, not unconditionally at `"main"` (2026-04-29 evening,)

**What.** At [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp), the `addExprToMemoryBlock` call that emits the wrapping OR expression after a branch's head proves no longer hard-codes `"main"` as the destination scope. Instead, the destination is computed by peeling the last `_boundary_<payload>` segment off the proving branch's `validityName`:

```cpp
std::size_t lastBoundary =
    validityName.rfind(NameMap::BOUNDARY_STR, std::string::npos,
                       NameMap::BOUNDARY_LEN);
const std::string orEmitScope =
    (lastBoundary == std::string::npos)
        ? std::string("main")
        : validityName.substr(0, lastBoundary);
```

For top-level OR-branches whose parent IS main (Peano OR-branching, the original `or0[7,2,1,3]` use case), this still yields `"main"` — semantics unchanged. For ORs nested under a hypothetical / integration scope (the FTA-rung-1 `{0,1}=[0,1]` proof's `or2[6,p,2]` under `_boundary_(implication26[1,4,2,6,15])`), the OR is now emitted at the implication's hypothetical scope rather than at main.

**Why.** Step 8 of the FTA-ladder rung 1 (OR-integration into `(in[p, M])`) was failing to close even though every input condition for the trigger was met. Trap-debug at `prover.cpp` showed the full chain:

1. Branch A head `(=[6, repl_lev_1_2])` deposited at Branch A scope `…_orint_(or2[6,repl_lev_1_2,2])_((=[6,repl_lev_1_2]))` with `status=1` and `inToBeProved=1`. Gate at line 4460 fires.
2. The OR `(or2[6,repl_lev_1_2,2])` emitted at `v=main` per the hard-coded destination.
3. `implication21` (`or2 ⇒ (in[p, M])`, ES2 backward direction) then fired at `v=main`, deriving `(in[repl_lev_1_2, 15])` at `v=main`.
4. The toBeProved goal `(in[repl_lev_1_2, 15])` lived at `_boundary_(implication26[1,4,2,6,15])` — the implication's hypothetical scope, NOT at main. The deposit at `v=main` did not match the goal at the implication scope (toBeProved lookup is exact-scope, keyed by `EncodedExpression(expr, validityName)`).
5. Step 8 stalled at `toBeProved=4` indefinitely.

The hard-coded `"main"` was a latent bug from the period when ORs lived only at the top-level scope (Peano OR-branching, `or0[7,2,1,3]` directly at v=main). Once OR-branching moved into the body of hypothetical / integration scopes — the FTA-ladder pattern starting with rung 1 — the destination became wrong.

**Verification (full main.py + verifier).** With the fix on:

- IncubatorGauss1 standalone burst #50 toBeProved drops from 4 (pre-fix) to **3** (post-fix). The `(in[repl_lev_1_2, 15])` entry at `implication26` scope is matched and erased; Step 8 closes. Step 9 (forward preorder conjuncts) and Step 10 (top-level interval) remain open as documented.
- Full main.py: 5 batches complete (IncubatorPeano 513 / Peano 49 / IncubatorGauss 91 / IncubatorGauss1 0 / Gauss 11), runtime 1183 s (was 1332 s — Step-8-closure cleanup pays back ~10% wall-clock).
- Verifier: 35 458 checks, 0 failures across both proof graphs (main 32 700 / 0, incubator 2 758 / 0). No regression on Peano OR-branching, Gauss main, or the four pre-existing configs.

**Risk and scope.** Behaviour change for ORs nested under hypothetical scopes (FTA-ladder rung 1 onward); no behaviour change for ORs at top-level (the only OR shape exercised pre-FTA-ladder). Runs cleanly on the full 35K-check verifier sweep — the existing OR machinery's invariants are preserved.

**Code.** [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp). Sibling `cleanUpOrIntegrationBranches` call at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) is unchanged (it walks branch payloads via the OR signature, scope-independent).

**Open follow-up.** Step 9's forward preorder conjuncts (`(preorder[1,4,2,p])` and `(preorder[1,4,p,6])` for universal `p ∈ M`) still don't fire automatically. The mechanism needed there is **OR-disintegration** (`ordisint`), not OR-integration: ES2-forward (`(in[p, M]) ⇒ or2[2, p, 6]`) plus a per-disjunct case re-deriving the same preorder consequence, then emit at parent scope. If `ordisint`'s emission site has the same hard-coded destination-scope bug, the same `rfind+substr` fix applies. To investigate next.

---

<a id="d-29"></a>
## D-29 — Two-part disintegration gate (anchor-LB always blocks; non-anchor needs local premise) under incubator + !ban_disintegration (2026-04-29 evening,)

**What.** New conditional gate inside `checkLocalEncodedMemoryStatic` ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)). After the existing `doNotDisintegrate = (lmv.justification == "integration")`, an additional check fires when both `parameters.incubator_mode == true` and `!parameters.ban_disintegration`:

1. **Anchor LB always blocks** — if `memoryBlock.exprKey` starts with `"(" + anchorInfo.name`, set `doNotDisintegrate = true` unconditionally.
2. **Non-anchor LB needs a local premise** — otherwise, iterate `orderedPremises`; if **none** are in `memoryBlock.localEncodedStatementsSet`, set `doNotDisintegrate = true`.

The first iteration of D-29 (committed earlier the same day) had only the second clause and was inert — the anchor LB itself processed most rule firings, and `localEncodedStatements` is populated by `prehandleAnchor` ([`prover.cpp/8236`](../GL_Quick_VS/GL_Quick/src/prover.cpp)) so anchor's own statement set looked "local" to itself; almost every rule firing satisfied the local-premise check. Adding clause 1 fixed it.

Supporting infrastructure: new `Memory::localEncodedStatementsSet` (`std::set<EncodedExpression>`, declared at [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp)) maintained in lockstep with the existing `localEncodedStatements` vector. 11 push_back sites in `prover.cpp/.hpp` got mirror `.insert` calls; the one assignment site at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp) rebuilds the set after the wholesale vector replacement.

**Why.** With `ConfigIncubatorGauss1.json` (the SE2 migration target) running with `incubator_mode=true && ban_disintegration=false`, the prover loaded the 1209 proved theorems from the prior IncubatorGauss batch (shared `theorems_folder`) into hashmem. The anchor LB then processed every loaded external rule whose premise (typically `(in[X, N])`) matched any of its anchor-deposited `(in[X, N])` rows; Pass B fan-out on the heads minted fresh `it_*` / `int_*` / scope-name variables; growth was 76 → 88 → 538 → 2702 → 4586 expressions in 2 main-phase bursts before the int16_t NameMap (capped at `ExecutionParameters::MAX_NAME_IDS` = 16384) exhausted at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp).

Clause 1 of the gate cuts the explosion at its source: the anchor LB stops disintegrating heads of broadcast-driven rule firings (it still receives them and emits their heads as whole statements via the `else` branch at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp), which goes through normal mailing without name-fan-out). Clause 2 is a finer filter for descendant LBs where the anchor-broadcast rules also fire: only LB-originated rule firings get to disintegrate.

**Why this preserves §4.1.** The §4.1 proof's required disintegrations all happen at the SE2 LB and below (the existence0 body, the `(in3[…])` preorder body, the OR branches via case-OR mint). Those LBs are non-anchor, and the §4.1-relevant rule firings have premises the SE2 LB itself derives during proof — so clause 2 keeps the gate open for them.

**Performance.** O(log N) lookup per premise via `std::set<EncodedExpression>` (the parallel container) vs O(N) on the vector. With LBs reaching thousands of statements, the set is justified. Memory cost: ~doubling the local-statement footprint per LB. Acceptable for an active SE2 LB; for the 1209-LB IncubatorGauss batch the cost is per-LB and bounded.

**Mode gating.** New gate is dormant outside `incubator_mode && !ban_disintegration`. ConfigPeano / ConfigGauss (no incubator_mode) and the legacy ConfigIncubator{Peano,Gauss} (ban_disintegration=true) follow the legacy path verbatim.

**Verification.** `gl_quick.exe IncubatorGauss1` now runs to completion (exit 0, prover finished, plateau at 7730 expressions across all 48 main-phase bursts). Pre-D-29 the same run crashed at burst 2 / ~4570 expressions. Gate-firing stats during the run: anchor blocks ≈ 3236, non-anchor non-local blocks ≈ 64, allowed (LB-derived rule firings, disintegration proceeds) ≈ 56700. §4.1 / SE2 not proved yet on this run (saved 0 theorems) — acceptable per user "se2 might be unproved" framing; further conjecturer / prover tuning can come later. Crash mitigation milestone: the architecture supports the SE2-migration combination without exhausting the int16_t NameMap.

**Code.** Gate at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp). Set-container infrastructure: [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp), 11 push_back sites and 1 rebuild-after-assign site across `prover.cpp` / `prover.hpp` — grep `localEncodedStatementsSet.insert` for the full list.

---

<a id="d-28"></a>
## D-28 — Collapse `allow_disintegration` into `ban_disintegration`; single flag for every disintegration path (2026-04-29 evening,)

**What.** The `allow_disintegration` flag introduced earlier the same day in [D-27](#d-27) is removed. Pass B at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) is now gated by `!parameters.ban_disintegration` (combined with `!parameters.compressor_mode`), making `ban_disintegration` the **single** gate for every disintegration-shaped path: Pass B, back-reformulation ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)), hypothetical disintegration ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)), necessity-for-equality-hypo ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)).

`allow_multiplication` (the multiplyImplication gate at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)) stays — different concern, different flag.

Per-config matrix:

| Config | `ban_disintegration` | `allow_multiplication` |
|---|---|---|
| `ConfigPeano.json` | `false` (default; field omitted) | `false` |
| `ConfigGauss.json` | `false` (default; field omitted) | `false` |
| `ConfigIncubatorPeano.json` | `true` | `true` |
| `ConfigIncubatorGauss.json` | `true` | `true` |
| `ConfigIncubatorGauss1.json` | `false` | `false` |

**Why.** D-27's two-flag split was over-cautious. Once the per-config matrix was filled in, every config's `ban_disintegration` happened to equal `!allow_disintegration` exactly — there was no config that wanted to allow Pass B while banning back-reformulation, or vice versa. Two flags doing the same job under opposite signs ("ban" + true blocks vs "allow" + false blocks) is just confusion to read. Collapsing into one flag drops that confusion, removes the awkward mixed-sign naming, deletes one row from the config schema, and reduces the "five-flag interplay" suspected fragility in [`docs/10_pipeline/09_incubator.md`](10_pipeline/09_incubator.md) to a four-flag one.

**Behavior change.** None for any of the 5 existing configs (the matrix above is exactly what each config did pre-D-28). The compressor's Phase 1 toggle (`ban_disintegration = true` at [`compressor.cpp`](../GL_Quick_VS/GL_Quick/src/compressor.cpp), restored at [`compressor.cpp`](../GL_Quick_VS/GL_Quick/src/compressor.cpp)) is unchanged and now also disables Pass B during Phase 1 directly through this flag — previously Pass B during Phase 1 was disabled by `compressor_mode=true` and the `ban_disintegration=true` was orthogonal noise; now both flags are aligned.

**Alternatives considered.** (a) Keep both flags with different semantics — rejected, no use case for divergent semantics has emerged in 5 configs and the mixed-sign naming would persist. (b) Rename `ban_disintegration` to `allow_disintegration` (positive sense, easier to read) and remove the `ban_*` flag — rejected as more churn (existing reads at multiple sites, no behaviour gain), and `ban_*` carries useful "this thing is normally on, here we explicitly stop it" intent.

**Verification.** Read all 5 configs after the collapse — `ban_disintegration` field present and correctly set in 3 (the two legacy incubator + ConfigIncubatorGauss1), default-false (field omitted) in 2 main configs. JSON parses; conjecturer-only run (`gl_quick.exe --conjecture IncubatorGauss1`) still emits the single SE2 conjecture byte-identical.

**Memory implication.** The "did we really need two flags" question is answered. Future agents inheriting this branch should not be confused by a brief D-27 era; the I-7 invariant text now reflects the collapsed state.

---

<a id="d-27"></a>
## D-27 — Decouple Pass B and `multiplyImplication` from `incubator_mode`; migrate `EnumerationSet2` to incubator side (2026-04-29)

**What.** Two new `prover_parameters` boolean flags, `allow_disintegration` and `allow_multiplication`, replace the previous role of `parameters.incubator_mode` at the Pass B and `multiplyImplication` gates respectively:

- [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) — Pass B entry. Old: `!compressor_mode && !incubator_mode`. New: `!compressor_mode && allow_disintegration`. Updates [I-7](30_invariants.md#i-7).
- [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) — `multiplyImplication` early return. Old: `!incubator_mode && !ceFilteringActive`. New: `!allow_multiplication && !ceFilteringActive` (CE-filter carve-out preserved).

`incubator_mode` keeps its other semantics (`head-in-wholeExpressions` short-circuit, integration reformulation, tempArgs assert, conjecturer behaviors). `ban_disintegration` keeps its narrower scope (back-reformulation, hypothetical disintegration, necessity-for-equality-hypo).

Per-config flag values match current behaviour for the four pre-existing configs:

| Config | `allow_disintegration` | `allow_multiplication` |
|---|---|---|
| `ConfigPeano.json` | `true` | `false` |
| `ConfigGauss.json` | `true` | `false` |
| `ConfigIncubatorPeano.json` | `false` | `true` |
| `ConfigIncubatorGauss.json` | `false` | `true` |
| `ConfigIncubatorGauss1.json` (NEW) | `true` | `false` |

Ride-along change to the orchestrator: `run_modes.py` now globs `^Config<base_tag>\d*\.json$` per base tag and runs the matches alphanumerically (the dot-vs-digit ASCII ordering puts the un-suffixed config first). The new `ConfigIncubatorGauss1.json` slots between `ConfigIncubatorGauss.json` and `ConfigGauss.json` in the Gauss group's run order.

`EnumerationSet2` migrated from `ConfigGauss.json` (deleted entry + 17 `prohibited_combinations` entries) into the new `ConfigIncubatorGauss1.json`. `files/theorems/proved_theorems.txt` is byte-identical post-migration because §4.1 / §4.2 were never proved on the Gauss main path — only enumerated as conjectures and stalled in the prove pass. `files/incubator/theorems/proved_theorems.txt` may grow if `ConfigIncubatorGauss1.json` succeeds in proving §4.1 (not required for this PR).

**Why.** Pre-decoupling, no batch could combine `Pass B on + multiplyImplication off + incubator_mode on (for contradiction LBs / skip CE filter)`. The FTA-ladder rung-1 proof of `{0,1}=[0,1]` (§4.1 forward = `ES2 ⟹ interval`) needs exactly that combination: it disintegrates the `EnumerationSet2` and `interval` body universal quantifiers (Pass B), must not fan out via Bell-partition multiplication (which interferes with the proof's specific shape), and benefits from incubator-side `0≤0`, `0≤1`, `1≤1` preorder facts that aren't visible at Gauss-main `v=main`. The decoupling makes that combination expressible.

**Alternatives considered.** (a) Keep `EnumerationSet2` in Gauss main but flip a hidden flag to disable multiplication for §4.1's specific shape — rejected as ad-hoc and fragile. (b) Move only `incubator_mode`'s Pass-B role onto the new flag, leave `multiplyImplication` tied to `incubator_mode` — rejected because it forces ConfigIncubatorGauss1 to choose between Pass B (needs `incubator_mode=false`) and contradiction LBs / skip CE filter (needs `incubator_mode=true`). (c) Keep three-flag interplay (`incubator_mode`, `ban_disintegration`, `compressor_mode`) — rejected, that interplay is already flagged in `docs/10_pipeline/09_incubator.md` weaknesses as a known fragility.

**Risk.** New flag-multiplication risks: a config that omits one of the new flags falls back to the default (`allow_disintegration=true`, `allow_multiplication=false` — main-path semantics), which is sensible for a non-incubator config but wrong for an incubator config that forgets to set them. Mitigation: every existing config got both flags written explicitly; the per-tag glob makes new incubator-side configs (`ConfigIncubator*1.json`, `ConfigIncubator*2.json`, …) increasingly common, and they all need to set both explicitly.

**Verification.** `files/theorems/proved_theorems.txt` byte-identical (no §4.1/§4.2 lines were there to lose). `files/incubator/theorems/proved_theorems.txt` is a strict superset (no removals). FTA-ladder Steps 1–7 from `fta_ladder/rung1/current_proof_state.md` continue to fire (now on the incubator side via `ConfigIncubatorGauss1.json` instead of stalling in the Gauss-main `(EnumerationSet2[…])` LB). Steps 8–10 may now be reachable because the missing preorder facts are local — empirical, not gated.

---

<a id="d-26"></a>
## D-26 — `multiplyImplication` double-`u_` skip restored + verifier free-anchor-merge guard added (2026-04-28)

**What.** Two-sided defence against `multiplyImplication` equating distinct free `u_*` anchor parameters. (1) Prover side: re-enabled `if (hasDoubleU) continue;` at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) inside the partition-iteration loop, restoring the soundness skip that commit cf358271 had deleted while pursuing a `u_`-equalisation extension. (2) Verifier side: extended `check_equalize_variable` ([`verifier.py`](../verifier.py)) with a post-mapping free-anchor-merge guard. The new helper `_extract_bound_vars` parses every `>[…]` binder of source and copy implications; any `(orig_arg → copy_arg)` pair where both names are free (not in any binder) and the names differ now fails the `multiplied from` row.

**Why.** Chapter-1115 of the IncubatorGauss proof graph carried a `multiplied from` step that collapsed slot 6 of `existence2` from anchor element `1` to `0` while preserving the rest of the rule body. Downstream, the chapter used the forged rule to derive `!fold[N,s,+,id,0,0,0]` (HTML title: "sum(i=0..0) ≠ 0") — a mathematically false statement. Trapping the input to `multiplyImplication` confirmed the input was byte-identical to the raw `1114_direct_proof.txt:3` source (the prover received a sound rule); the bug was wholly inside `multiplyImplication`'s partition loop. cf358271's verifier-clean (1818 checks) had not exercised the FTA-ladder shape that triggers the bad partition.

**Why both gates rather than one.** The prover-side skip is the primary fix — it prevents the unsound copy from being emitted, so no chapter cites it. The verifier-side guard is independent defence: if a future change re-introduces u\_-equalisation in a different code path (a new compress mode, an alternate hash-burst rule, an externally-provided theorem with a `multiplied from` row), the verifier catches it before the proof ships. Per [I-16](30_invariants.md#i-16) the verifier is sacred — soundness gates that can live there should.

**Why parse binders for free/bound classification, not preserve `u_*` in processed graph.** The original instruction was to preserve `u_*` prefixes in implication source columns of `multiplied from` rows so the verifier could distinguish free anchors from bound variables by string prefix. I implemented it and reverted: keeping `u_*` survives `process_proof_graphs.py` iteration 1 but iteration 2/3 walks the modified cell, sees the `u_*` args as unmapped, and assigns chapter `v`-numbers — bypassing the anchor map and breaking the origin check (~1740 multiplied-from sources fail to resolve in `state.global_theorems`). Skipping iteration 2/3 selectively for the multiplied-source cell would propagate inconsistencies to cross-references; storing the raw form in an extra column would change the row layout for every consumer. Parsing `>[…]` binders inside the verifier itself sidesteps both — same soundness guarantee, zero change to the row format.

**Verification.**

| | Pre-fix (incubator-only run) | Post-fix (full chain, all four batches) |
|---|---:|---:|
| Verifier checks | 32585 / 0 | 35458 / 0 (incubator + main) |
| Incubator HTML chapters | 1342 (incl. false `!fold[…,0,0,0]`) | 1202 (140 chapters dropped — every theorem whose only proof path used a double-`u_` partition; the false statement is gone) |
| FTA-ladder Steps 1–7 (HB#36 SE2 LB) | reached | reached, identical Branch A validity, identical witness names |
| Synthetic adversarial test (`existence2` slot-6 `i1 → i0`) | — | verifier returns `False` ✓ |
| Synthetic sound tests (identity, Bell partition, bound→free specialise) | — | verifier returns `True` ✓ |

**SwDD touches.**
- This entry (D-26).
- [I-24](30_invariants.md#i-24) — new invariant.
- `docs/AGENT_SwDD.md` quick-reference — I-24 row added.
- [`docs/10_pipeline/04_prover.md`](10_pipeline/04_prover.md) — *Soundness gate* subsection inside the `multiplyImplication` chapter.
- [`docs/20_core_concepts/08_proof_tags.md`](20_core_concepts/08_proof_tags.md) — `multiplied from` entry now describes both the prover-side skip and the verifier guard.

---

<a id="d-25"></a>
## D-25 — Compressor invocation moved out of `analyzeExpressions` (2026-04-28)

**What.** The compressor is no longer invoked from inside `ExpressionAnalyzer::analyzeExpressions`. The prover's main entry now does prove → post-prove `headSwitchOne` walk → return; `run_modes.cpp::fullRun` then conditionally invokes `Compressor::run` after `analyzeExpressions` returns, gated by `parameters.skipCompression`. The gate's truthiness is unchanged (`incubator_mode || ban_disintegration`).

**Why.** Two reasons. (1) Restore master flow — the legacy in-`analyzeExpressions` compressor invocation was a leftover from the deleted multi-iteration loop ([D-24](#d-24)), where compressor output fed the next big-iteration broadcast set. With the loop gone there is nothing inside `analyzeExpressions` that consumes the compressor's result, so the call's only effect is to extend the prover's wall-time profile with a phase that semantically belongs to the orchestrator. (2) Symmetry with `run_modes` — every other post-prove pipeline step (`exportCompiledExpressionsJSON`, the OR-pair construction over `orPairsFromHeadSwitch`, the proof-graph emission) already lives in `run_modes.cpp`. Keeping the compressor next to its peers makes the post-prove sequence readable in one file; nothing in `prover.cpp` after the head-switch walk now competes for the reader's attention.

**Why not behind a flag.** The relocation preserves byte-equivalence: the same `Compressor::run` is called with the same inputs in the same order, just from a different translation unit. There is no behaviour to gate. A flag would only document the move, which is what this entry exists for.

**Code.** Compressor invocation site moved from `prover.cpp` (deleted) to `run_modes.cpp::fullRun`. The class member `parameters.skipCompression` keeps its semantics. The dropped `prover.cpp` block is recoverable via `git log` of commit.

---

<a id="d-24"></a>
## D-24 — Multi-iteration prover loop deleted; single-pass head-switch pre-emit becomes the only path (2026-04-28)

**What.** The big-iteration loop in `ExpressionAnalyzer::analyzeExpressions` is gone. The function now does one prove → one compress → one orPairsFromHeadSwitch population, in that order, and returns. With it go:

- `parameters.pre_emit_head_switch` (the trial flag from D-23-era exploration, now unconditional behaviour).
- `parameters.maxBigIterations` (the iteration cap, no longer meaningful).
- The grid teardown + rebuild block (`destroyGrid`, the cache-proof-stacks scan, `compileAndRegister(remaining)`, `buildGrid`) — every consumer was the second prove pass that no longer happens.
- The `headSwitchedTheorems` and `alreadySwitched` locals — only existed to seed the deleted next iteration.
- The "subsequent iteration" branch of the prove call (the one that broadcast all-of-`globalTheoremList` instead of `provedTheorems`) — same reason.
- About 400 lines of `prover.cpp` and a dozen lines elsewhere.

**Why.** D-23 closed the only gap between single-pass and two-pass on Peano: the single-pass trial ( then ) reached **27 proved theorems** in **377 s** versus the two-pass baseline **28 proved in 626 s** — and the sole "missing" theorem (`(>[5,6](AnchorPeano)(in3[8,7,6,5])(in3[7,8,6,5]))`) is the slot-6 instance of the more general `(>[5](AnchorPeano)(in3[8,7,9,5])(in3[7,8,9,5]))` which IS proved on the first pass. So the second pass was deriving content already entailed by the first. Keeping it was burning ~250 s of wall time per Peano run for no semantic gain.

**Why not keep the flag for safety.** Two reasons. First, the flag was *only* a kill-switch for code we now know is functionally dead — keeping it preserves zero behaviour and adds maintenance noise (every config has to know about it, the SwDD has to document a non-feature, future readers have to understand a "legacy two-pass" path that nobody ever uses). Second, the multi-iteration machinery had non-trivial state (the cache-proof-stacks scan, the per-iteration broadcast difference between iter-0 and iter-N>0, the dedup against `allProvedEver`); leaving it in as dead code makes the prover harder to read, not safer. If a future need re-emerges, `git log` recovers the path.

**What survives.** `headSwitchOne` ([`prover.cpp+`](../GL_Quick_VS/GL_Quick/src/prover.cpp)) — the stateless contrapositive helper extracted — stays. Both the pre-emit pass (now unconditional, just before the first `compileAndRegister`) and the post-prove walk (now the only call site that populates `orPairsFromHeadSwitch`) call it. Output between the two paths is byte-identical to the legacy in-loop construction.

**Verification on Peano-only `main.py` (post-deletion):**

| | Two-pass baseline (pre-D-23) | Single-pass trial (D-23) | Post-D-24 (this commit) |
|---|---:|---:|---:|
| Wall | 626 s | 377 s | matches D-23 (refactor only) |
| Verifier | 2105 / 0 airtight | 2098 / 0 airtight | 2098 / 0 airtight |
| Proved theorems | 28 | 27 (general slot-N variant subsumes the missing slot-6 specialisation) | 27 |
| `prove(...)` calls per batch | up to `1 + N×2` | `2` (warm-up + main, gated loop) | `2` (warm-up + main, no loop) |

`fullTheoremList` dedup is now a local `alreadyInFull` set inside the compress block (used to live as `allProvedEver` across the loop). All other downstream consumers (`run_modes.cpp::fullRun`, `visualizer.cpp::generateRawProofGraph`) read the same class members as before — `globalTheoremList`, `fullTheoremList`, `lastCompressionSurvivors`, `orPairsFromHeadSwitch`, `cachedProofStacks` — and their producer-paths are unchanged for the parts that still exist.

**Out of scope.** The warm-up vs main-iteration split *inside* the surviving single prove call (Phase 1 + Phase 2) is unchanged. That's the user's instruction; it's a different mechanism (`prove(preIterations, …)` → broadcast → `prove(remainingIterations, …)`) and not part of "multi-pass".

**SwDD touches in this commit.**
- This entry (D-24).
- `docs/10_pipeline/04_prover.md` — replaced the "Big-iteration loop and head-switch" + "Single-pass mode (`pre_emit_head_switch`)" subsections with a flat "Single-pass head-switch model" + "Contrapositive construction" pair.
- `docs/04_configs.md` — removed the `maxBigIterations` and `pre_emit_head_switch` rows from the `prover_parameters` field table.

---

<a id="d-23"></a>
## D-23 — D-21 relaxations 2 and 3 walked back; anchor-membership-axiom premise filter added (2026-04-28)

**What.** Two of D-21's three conjecturer relaxations are reverted, and one new structural-rejection filter is added. Net change visible in `gl_quick.exe --conjecture Peano`: 1785 → 536 surviving conjectures (−70 %). The cancellation theorem still emits (its ascending head `=[2,8]` reaches `theorems.txt:20` independently of any relaxation).

| D-21 relaxation | Status after D-23 |
|---|---|
| Relaxation 1 — `passesInPremiseFilter` cnt==2 neutralisation rule | **Kept.** Genuine typing premises like `(in[a, N])` next to `(in3[a, b, i0, +])` still need this rule; without it the cancellation theorem's typing premise can't pair with its operator-equation companion. |
| Relaxation 2 — `controlEquality` accepts descending `=[a, b]` at nse ≤ 3 | **Removed.** The pair-combination enumeration produces both ascending `=[a, b]` and descending `=[b, a]` forms independently — the cancellation theorem head exists in ascending form `=[i0, b]` (e.g. `=[2, 8]`), so dropping the descending sibling loses no genuine theorem. Function reverts to the original "reject any descending `=[a, b]`" rule. |
| Relaxation 3 — `passesComplexityAfterExistence` returns true unconditionally at complexityLevel ≤ 3 | **Removed.** The hardcoded `complexityLevel <= 2 → return true` and the 73-line `complexityLevel == 3` carve-out are deleted. The function now consults only `max_complexity_if_anchor_parameter_connected_after_existence` (per-type 2-tuple `[complexity_cap, arity_sum_cap]` from D-22's predecessor ) and rejects when **all three** of `complexity > complexity_cap`, `arity_sum > arity_sum_cap`, and the slot is present in non-anchor leaves. The cancellation theorem (arity_sum 4+2+2 = 8 ≤ 8) still escapes via the arity-sum dimension; explosion shapes (3 × in3 = 12 > 8) are rejected. |

**Plus, new filter — anchor-membership-axiom rejection.** A non-anchor `(in[v, X])` premise (or its negation) where BOTH `v` AND `X` are anchor-slot values — e.g. `(in[2, 1])` reads "i0 ∈ N", which is one of `AnchorPeano`'s own axioms — is vacuous as a premise. The previous complexityLevel == 3 carve-out had this check inline; with the carve-out removed, the check moved to `passesInPremiseFilter` as a top-of-function gate so it now runs for every conjecture, not just nse=3..

**Why.** D-21 was authored before the arity-sum dimension existed. Once both branches were merged, the relaxation 3 sat at the top of `passesComplexityAfterExistence` and short-circuited before the arity-sum check ever ran. Result on the merged branch: 1108 explosion-class conjectures of the form `(>[…anchor_slot_idx](AnchorPeano…)(in3[…6…])(in3[…6…])(in3[…6…]))` — slot-6 (i1) used 3× per conjecture — slipped through every gate. The arity-sum dimension is the precise tool for this discrimination; deleting the relaxation lets it work.

Relaxation 2's removal is a similarly cheap simplification: `controlEquality`'s pair-combination enumeration produces both orderings of every `=` head, so dropping descending forms loses nothing. The relaxation was a defensive measure when the conjecture-generation order was uncertain; the empirical evidence (descending and ascending heads both present at nse=3 across `theorems.txt`) shows the defensive measure isn't load-bearing.

**Why not the alternatives.**

- *Tighten Peano `(1)` arity_sum_cap from 8 to 7* to also catch the 8 in2-based `=[…]` 3-occurrence shapes that survive D-23's tightening. Would also block the cancellation theorem (arity_sum exactly 8). Rejected.
- *Add a separate "max occurrences per anchor slot value" cap field*. Duplicates the arity-sum dimension's role. Rejected.
- *Keep relaxation 3 with cross-leaf aggregation*. Less elegant; hard-coded literals stay; doesn't address D-22's filter mismatch. Rejected per user direction ("`passesComplexityAfterExistence` … shld use only config and not hard coded at all").
- *Keep relaxation 2 with anchor-involvement gate* (descending kept only when one arg is anchor-slot value, dropped when both bound). Rejected after empirical check showed the ascending mirror of every cancellation-shape conjecture is already in `theorems.txt` independently — relaxation 2 is unnecessary.

**Effect on `theorems.txt` (Peano alone, full conjecturer pass).**

| State | Count | Notes |
|---|---:|---|
| Pre-merge cancellation_theorem (D-21 active, no D-22) | 5656 | (per D-21 commit body) |
| Post-merge before D-23 | 1785 | D-22 brings the registry; D-21 still active in this state |
| After D-23 commit (drop relaxation 3) | 677 | `(1)`-pinned 3-`in3` shapes rejected via arity-sum cap |
| After D-23 commit (anchor-membership filter) | 635 | `(in[anchor, anchor])` premises rejected |
| After D-23 commit (drop relaxation 2) | 536 | descending `=[a, b]` symmetric duplicates rejected |

The cancellation theorem is at `theorems.txt:20` in the final state and was confirmed proved by the prover in the post-merge full Peano run (per user observation in run.log).

**SwDD touches.** Conjecturer chapter (`docs/10_pipeline/02_conjecturer.md`) updated to describe the post-D-23 form of the three filters: `passesComplexityAfterExistence` no longer has hard-coded complexity bands; `passesInPremiseFilter` has the anchor-membership gate at top; `controlEquality` is back to original strict-canonicalisation. Configs chapter (`docs/04_configs.md`) unchanged — the field semantics (`[complexity_cap, arity_sum_cap]` vector) were already documented at D-22-merge time.

---

<a id="d-22"></a>
## D-22 — Cross-batch shared registry for spontaneous compact operators (2026-04-25,, merged 2026-04-26)

**What.** Spontaneous compact operator names (`implication<N>`, `existence<N>`, `or<N>`, `and<N>`) are persisted across batches via a single growing JSON file `files/GL_binaries/GL_binary_shared.json`. Per batch:

- **Pre-batch.** Python copies `GL_binary_shared.json` to `GL_binary_<Tag>.json` (Python is the sole writer of shared; it is also the sole creator of each per-batch file). On a clean run when shared does not yet exist, an empty `{}` is written instead.
- **C++ startup.** The `ExpressionAnalyzer` constructor calls a new `loadGlBinary(path)` method that parses the per-batch JSON, populates `compiledExpressions` for every entry, registers a `repetitionExclusionMap` row for every spontaneous entry (using the JSON's `elements` field directly as `splitNK`, which is exactly what `excludeRepetitions` stored when the entry was first allocated), and seeds the four shared counter members from the trailing-integer maxima of the loaded names: `implCounter = max(N for "implication<N>") + 1` and analogously for `existenceCounter`, `andCounter`, `orCounter`. `variableCounter` continues to start at zero each batch — bound-variable identifiers have no cross-batch identity to preserve.
- **C++ shutdown.** `exportCompiledExpressionsJSON` (unchanged) writes the entire `compiledExpressions` map to `GL_binary_<Tag>.json`. Because shared entries were preloaded into that map, the per-batch file at end of run contains the inherited shared entries plus any newly-allocated this-batch entries.
- **Post-batch.** Python reads `GL_binary_<Tag>.json`, filters to entries whose `category` is in `{implication, existence, or, and}`, and adds any name not already present to `GL_binary_shared.json`. Anchor and atomic entries are excluded — they are batch-local. The shared file grows monotonically over the run.

**Why.** Each batch was previously assigning compact identifiers from counters that reset to zero in the `ExpressionAnalyzer` constructor. The same logical operator therefore acquired different names in different batches: a Peano successor-existence theorem named `existence2[1,6,3]` was renamed to `existence3[1,6,3]` when `gl_quick.exe Gauss` re-ran `precompileStructuralOperators` on Peano's expanded-form `proved_theorems.txt` entries. Because Gauss never adds those re-imported theorems back into its `globalTheoremList`, the Gauss-named form (`existence3`) ended up cited inside chapter 85 of the proof graph but missing from `raw_proof_graph/global_theorem_list.txt` (which holds Peano's `existence2` only). The Python pruning step in `process_proof_graphs.py:_prune_proof_graph` then dropped both forms because the essential set (`compiled_proved_theorems.txt`) and the raw-proof-graph set used disjoint identifiers, and the verifier failed the chapter-85 origin check on the cited template. The shared registry eliminates the rename: Peano allocates `existence2` and writes it to shared; Gauss loads shared at startup, finds the `existence2` entry already in `repetitionExclusionMap`, and `excludeRepetitions` returns the existing name instead of allocating `existence3`. Same name everywhere — the chapter-85 failure resolves itself.

**Why not the alternatives.** *Re-emitting prev-batch theorems into Gauss's `globalTheoremList`* would push the Gauss-renamed identifier into the raw graph, but `compiled_proved_theorems.txt` would still need a synonym mapping to reconcile with the Peano-named form already in earlier raw-graph entries — net effect: two identifiers for one theorem with bookkeeping overhead. *Verifier-side synonym resolution* would touch [I-16](30_invariants.md#i-16) (verifier sacred). The shared-registry approach removes the divergence at its origin.

**Operational consequence.** A clean run still wipes `files/GL_binaries/` (existing behaviour in `run_modes.py:full_run` lines 227-230). To force re-allocation under fresh numbering after a config change to a structural operator's `definedSet` or `existence_variable_position`, delete `GL_binary_shared.json` manually and rerun; there is no automatic staleness detection. See [I-23](30_invariants.md#i-23).

**Renaming consequence.** The `int statementCounter` member on `ExpressionAnalyzer` was renamed to `andCounter` in the same change. Every operational use was passing it into `compileCoreExpressionMapCore`'s `andCounter` parameter slot; no `statement<N>` operator exists anywhere in the codebase. The four spontaneous-operator counter members are now spelled by their actual roles: `implCounter`, `existenceCounter`, `andCounter`, `orCounter`.


---

<a id="d-21"></a>
## D-21 — Three nse≤3 conjecturer relaxations for the cancellation family (2026-04-24)

**What.** Three related relaxations in the conjecturer's filter cascade, all scoped to small (`complexityLevel / nse <= 3`) conjectures, landing together so the additive-cancellation theorem `(in[a,N]), (in3[a,b,i0,+]) -> (=[b,i0])` actually ends up in `theorems.txt`:

1. The `cnt == 2` branch of `Conjecturer::passesInPremiseFilter` at [`conjecturer.cpp`](../GL_Quick_VS/GL_Quick/src/conjecturer.cpp) now accepts an additional shape: a positive `(in[v,X])` premise together with one non-anchor companion premise, provided `v` (first arg of the `in`) also appears as an argument in the companion premise or in the head. Originally the `cnt == 2` branch required at least one of the two premises to be negated; this rule is kept as rule (2) and the new check is rule (3). `cnt >= 3` remains rejected unconditionally.

2. `Conjecturer::controlEquality` at [`conjecturer.cpp`](../GL_Quick_VS/GL_Quick/src/conjecturer.cpp) no longer rejects descending-ordered `(=[a,b])` (`a > b` as integers) when the conjecture has `nse <= 3`. The original canonicalisation still applies at `nse > 3`. Motivation: after anchor-pinning, a bound-var ID lands in arg-1 and an anchor-slot ID in arg-2, producing the descending form `=[8, 2]` — the `b = i0` head of the cancellation theorem. Rejecting that form silently erases the family.

3. `Conjecturer::passesComplexityAfterExistence` at [`conjecturer.cpp`](../GL_Quick_VS/GL_Quick/src/conjecturer.cpp) now returns `true` unconditionally when `complexityLevel <= 3`. The full per-def-set cap logic still applies at `complexityLevel >= 4`. Motivation: Peano's `(1)` cap is 2, meaning "no chain arg may be pinned to anchor's i0 or i1 slots when the conjecture has 3+ `(>[` layers". The cancellation theorem has `complexityLevel = 3` AND pins `in3`'s c-arg to anchor slot 2 (i0), so the un-relaxed filter drops it. Discovered via the hard-coded-target tracer protocol (see ): every earlier filter passed, every other anchor-attach variant that reached this point was dropped here. The carve-out admits the cancellation family but also a wide set of other `nse=3` (1)-pinned shapes that were previously filtered on blow-up grounds — see trade-off note below.

**Why.** The FTA ladder's next rung needs the additive-cancellation family — `(in[a,N]), (in3[a,b,i0,+]) → (=[b,i0])` and its successor/product siblings — to be conjectured. Under the original rule (negation required at `cnt == 2`) the cancellation shape was enumerated by the combinatorial core but silently dropped by the filter, because `in[a,N]` carries semantic weight (typing) without needing a negation partner. The relaxation scopes strictly to `cnt == 2` (i.e. `nse = 3`) so the conjecture population cannot explode on larger shapes.

**Why this shape, not a config flag.** "Neutralises a free variable" is a structural property of the conjecture (does `v` participate elsewhere?), not a per-batch policy. Putting it in code keeps the rule auditable. A batch that genuinely needs to reject it can still do so via `prohibited_heads` or the prohibited-combinations list.

**Side correction.** While tracing this change, found that `apply_in_premise_filter` — documented as a per-batch gate — is dead code. The flag is declared and loaded but never consulted at the filter entry or at its callsites. SwDD `OPEN-9` and `docs/04_configs.md` updated. `ConfigGauss.json`'s `"apply_in_premise_filter": false` has no effect; Gauss avoids the filter only because its `in[]`-bearing conjectures place `in` at the head (so `hasIn` is false and the function short-circuits).

**Before / after (conjecturer output, strict-subset confirmed each step).**
- Peano: 501 → 524 (relaxation 1, +23 generalised-cancellation lines) → 559 (relaxation 2, +35 descending-`=` mirrors) → 5656 (relaxation 3, +5097 `(1)`-pinned nse=3 shapes). The target cancellation theorem `(>[1,2,4](AnchorPeano[1,2,3,4,5,6])(>[7,8](in3[7,8,2,4])(>[](in[7,1])(=[8,2]))))` is in the final set.
- Gauss: 395 → 397 (relaxation 1, +2 interval neutralisation) → 397 (relaxation 2, no change) → 397 (relaxation 3, no change). Relaxation 3 has zero effect on Gauss because its anchor has no `(1)`-typed cap in this config slot.

**Trade-off — relaxation 3 is wide.** Moving `complexityLevel > 2 (1)-pinned` from "reject" to "accept at complexityLevel=3" adds ≈5000 new Peano conjectures. Most are unproved (prover + CE filter will discard), but the batch wall-time grows proportionally. Alternatives considered:
- Raise the config cap from `(1):2` to `(1):3` — equivalent effect, same 5000-conjecture blow-up, parameterised in config rather than in code. Rejected because the cap is *semantically* "max non-anchor chain length when (1) pinned", not a cancellation-specific knob, and touching it via config reads as a general policy change rather than a cancellation-family unlock.
- Pattern-match specifically for `in3[*,*,anchor-slot,+/*]` + `in[*,N]` + `=[*,anchor-slot]` and skip only for that shape — narrower, but couples the filter to a specific conjecture family and would need extending for each new family.
- Leave the filter as-is and seed the cancellation theorem via `externally_provided_theorems.txt` — abandons "conjectured first".
The current choice (unconditional pass at `complexityLevel <= 3`) is the blunt version; tightening is open if the 5000-conjecture blow-up turns out to cost meaningful prover wall-time.

**Alternatives considered.**
- *Config flag to gate the relaxation.* Rejected — the neutralisation property is intrinsic to the conjecture, not a batch-level policy. Also parallels the OPEN-9 dead-flag we just found: adding another unused flag does not help.
- *Widen to `cnt == 3`.* Rejected per user direction — risks a conjecture-population blow-up without a proven need; can be revisited if the FTA ladder needs it.

---

<a id="d-20"></a>
## D-20 — Conjecturer int-story performance campaign (2026-04-24, branch `rt_conjecturer_session_24042026`)

**What.** Two perf-only changes on the conjecturer hot path that preserve the byte-for-byte output contract (`theorems.txt`, `reshuffled_theorems.txt`, `reshuffled_mirrored_theorems.txt`) on both Peano and Gauss:

1. **Path B reshuffle** — `Conjecturer::reshuffle`'s permutation inner loop replaced: pre-parse each chainEntry + head into an `EntryTemplate` that pinpoints every `[...]`-token byte position with an `isAtom` flag; per permutation, walk atom slots in permuted byte order to build the first-occurrence rename map into a thread_local `int[]` indexed by dense token id; render the rebuilt string directly into a thread_local `std::vector<char>` with inline rename substitution; winner selected by `memcmp` on the rendered buffer. Eliminates the per-permutation `std::string` rebuild + `replaceKeysInString` pass. Semantics byte-for-byte equivalent to OLD (memcmp on renamed rebuilt strings IS `std::string::operator<`).

2. **Disintegrate cache** — `compiler.hpp::disintegrateImplication` carries a thread_local single-slot cache keyed by the input expression. Matches the hot call pattern: one candidate passes through ~10 filters that each re-disintegrate the same string; each new candidate misses once, every subsequent filter call on the same string hits. thread_local gives zero-contention per-worker reuse.

**Why.** Profiling (session baseline) showed the conjecturer at ~30 s wall for Peano. Top consumers of thread-time: `reshuffle` 218 s (86 % inside the permutation inner loop, all `std::string` / `replaceKeysInString` work) and `disintegrateImplication` 198 s (17.3 M calls, 11 µs / call dominated by heap-allocated TreeNode1 trees). The SwDD's `docs/10_pipeline/02_conjecturer.md` already identifies the string-path as "the older lane... unfinished portion of the 100x acceleration campaign" (line 120). These two changes address two distinct string-path pain points without touching semantics.

**Measured effect (Peano, 3-run avg, 32 HW threads).**

| commit | wall | reshuffle | disintegrate |
|---|---|---|---|
| (baseline) | 30.5 s | 218 s | 198 s |
| (+ Path B) | 27.2 s | 111 s | 198 s |
| (+ cache) | 23.8 s | 114 s | 75 s |

Total: −6.7 s wall (−22 %) across two commits.

**Alternatives considered.**

- *Path A for reshuffle* — keep the `std::string` rebuild but make it cheap (reserve + direct char writes, skip `replaceKeysInString`). Would have given ~half the win with lower design risk. Rejected because the byte-identity proof is just as easy for Path B (memcmp equivalent to string compare) and the payoff is ~2× bigger.
- *Full iterative `disintegrateImplication`* (replace `parseExpr` + TreeNode1 tree build with an allocation-free string walk). Would give a bigger win than the cache on cache-miss calls, at the cost of a higher byte-identity risk (treeToExpr's canonicalization edges). Reserved for a later commit if we want to squeeze more.
- *Port the remaining string filters to int* (`checkInputVariablesOrder`, `evaluateOperatorExprs2`, `checkInputVariablesHead`) — touches filter signatures + many callers. Parked pending user direction.

**Follow-ups.**

- The profiling scaffold (`prof::` namespace in `conjecturer.cpp`, `g_disint*` atomics in `compiler.hpp`) is temporary, left in-tree while the campaign is active so before/after measurements stay reproducible. Remove in a clean-up commit when the campaign closes.
- `exprGood2Int` at 278 s / 28 % of CPU is now the top remaining consumer. It's already int-based; the cost is pure call volume (30.3 M) × per-scan work. It internally decodes and calls `evaluateOperatorExprs2` on strings — porting those to int is the next wins candidate.

---

<a id="d-19"></a>
## D-19 — `rejectedMapIntegration` revival via `internalMailIn` only (2026-04-24)

**Status.** Superseded by [D-53](#d-53) (2026-05-07, renumbered from main's D-46). The "dedicated `InternalMail` channel" rationale below is preserved for historical context; `struct InternalMail` was deleted and `Memory::internalMailIn` is now type `Mail`. The lifecycle and absorb-status distinctions described below remain intact.

**What.** Integration-side rejection recovery uses a dedicated `InternalMail internalMailIn` channel on `Memory`. Its statements re-enter the LB at next hashburst with `status=1` (full disintegration pipeline). It does not go through `addExprToMemoryBlock` directly from the equi-class rewrite site; it does not share `Mail::statements` with the routing `mailIn`.

**Why.** Three reasons.

1. *Avoid cyclic re-entry.* The algebra-side counterpart (`revisitRejected2`) calls `addExprToMemoryBlock`, which internally can call `updateRejectedMap` again — a structural cycle guarded by `revisitInProgress`. The mailIn-style deposit is linear: the producer writes to `internalMailIn`, the next hashburst absorbs, the absorb triggers normal `addStatement` flow. No re-entry risk.
2. *Scope fidelity.* Legacy `Mail` is main-scope-only (every `mailOut.statements.insert` is gated on `validityName == "main"` and absorb hard-codes `"main"`). Integration rejections can live at non-main validities (e.g. Branch A inside an OR integration). Mixing scope semantics into legacy mail would force 8 existing call sites to migrate from `pair<expr,levels>` to `tuple<expr,levels,validity>`. A new `InternalMail` struct with a tuple-shaped `statements` set isolates the change.
3. *Status asymmetry.* Legacy `mailIn` absorb at `prover.cpp` uses `status=3`, which skips `disintegrateExpr2` — correct for broadcast recipients of main-scope facts. Integration revival is the opposite: we *want* the revived constituent to re-enter Pass B so a fresh `int_` mint can succeed under the now-rewritten args. `status=1` is the right status, distinct from legacy.

**Operational consequence.** `Memory` carries two inboxes: `Mail mailIn` (routing) + `InternalMail internalMailIn` (revival). Same drain site (top of hashburst) but different absorb status and different lifecycle ([I-21](30_invariants.md#i-21)). Revival origin uses the existing `equality1` tag (eq-class substitution), with `origin.second[0]` = pre-rewrite constituent form so the verifier's `equality1` checker at [`verifier.py`](../verifier.py) accepts the shape.

---

<a id="d-18"></a>
## D-18 — the project conventions tag-count reconciliation (2026-04-23)

**What.** Corrected the project conventions's implicit claim of 28 tags in `TAG_CHECKERS` to 28 distinct tags + 29 registry entries. The missing tag in the project conventions's narrative list was `symmetry of inequality`; the extra registry entry is the shared `equalize variable` (dead alias) / `multiplied from` checker.

**Why.** Documented here during SwDD authoring so future agents don't inherit the drift.

**Implication.** When editing `TAG_CHECKERS`, remember one checker may serve two tag keys. Removing the `equalize variable` alias would be cosmetic — no emission site produces it.

---

<a id="d-17"></a>
## D-17 — Dual-license stamp on documentation (2026-04-23)

**What.** Every `.md` under `docs/` carries the AGPLv3 + commercial dual-license HTML comment at top, matching the Python/cpp stamp pattern.

**Why.** Documentation is part of the commodity. Proof graphs, theorems, and now the SwDD all inherit AGPL terms. Commercial-license buyers get the same text without the AGPL obligation. The HTML-comment form keeps the stamp non-rendering in markdown viewers while still embedding it in the file.

---

<a id="d-16"></a>
## D-16 — Hard-coded Peano-theorem drop on Gauss batch (2026-04-22, commit )

**What.** `run_modes.cpp–150` contains a hard-coded filter that removes a specific Peano theorem (`"(>[1,3,6](AnchorPeano[1,2,3,4,5,6])!(>[7](in[7,1])!(in2[7,6,3])))"`) from the inherited `proved_set` when `anchor_id == "Gauss"`.

**Why.** The branch's investigation found that this particular Peano theorem (an unsound-by-induction-typing artefact — see [I-18](30_invariants.md#i-18)) poisons downstream Gauss proofs of the `fold` / `limitSequence` cascade. Dropping it restores the cascade.

**Status.** Temporary mitigation. Real fix is the induction-typing sub-theorem rollout per [`docs/induction_typing_plan.md`](induction_typing_plan.md). The hard-coded drop should be removed once the induction-typing fix lands.

**Root-cause note on the drop:** reason the drop restores the cascade is not fully understood as of the commit message. An active debugging line.

---

<a id="d-15"></a>
## D-15 — Induction-typing sub-theorem — design approved (2026-04-??, pre-)

**What.** To close the induction-soundness gap, induction on a bound variable `n` must be preceded by proving `(in[n, N])` from the current chain. The typing sub-theorem uses the same prover with a `typingProofOnly = true` flag to forbid induction re-entry.

**Why.** Without typing, induction can be scheduled on variables that are not in `N`, ranging the theorem over all entities. This is unsound and produced `proved_theorems.txt` entries that should not be there. See [I-18](30_invariants.md#i-18) and [`docs/induction_typing_plan.md`](induction_typing_plan.md).

**Trade-off considered.** Could weaken the induction scheduler to reject all cases where typing is not trivially derivable — less intrusive but also loses some valid proofs. The chosen approach (full typing sub-theorem, artefact-visible via a dedicated chapter) preserves auditability.

---

<a id="d-14"></a>
## D-14 — `passesInPremiseFilter` behind a config gate (2026-04-??, commit )

**What.** The recent `passesInPremiseFilter` conjecturer filter is now config-gated rather than always-on.

**Why.** The filter is beneficial for some batches and detrimental for others. A config flag lets per-batch decisions flow through `Config<Tag>.json`. See [`10_pipeline/02_conjecturer.md`](10_pipeline/02_conjecturer.md) — `Conjecturer::passesInPremiseFilter` at [`conjecturer.cpp`](../GL_Quick_VS/GL_Quick/src/conjecturer.cpp).

---

<a id="d-13"></a>
## D-13 — Validity-stack NameMap migration (2026-04-??, branch, landed subsequently)

**What.** The per-LB validity-name system was migrated from free-form strings + side tables to the `NameMap` + `pairMap` + `stackOfValidity` canonical form. Non-`main` scopes must now be minted via `encodePush(parent, payload)`. See [I-2](30_invariants.md#i-2).

**Why.** Free-form scope strings made ancestor queries ambiguous; side tables required coordinated updates per scope kind. Centralising in `pairMap` gives a single source of truth for parent-child relationships and makes `comparable` / `deeperOf` reliable under branching.

**Trade-off considered.** Keep the old form, add assertions. Rejected — the underlying ambiguity is structural, not a discipline problem.

**Subsequent extensions** (`or_5`, `or_6`):

- Equivalence classes propagate to descendants, not just the defining scope.
- `vacuous truth` confined to scope `"main"` (the vacuous-truth tag fires only at root).
- Sentinel scope kind renamed to a less ambiguous payload prefix.

---

<a id="d-12"></a>
## D-12 — Variable-copy tag subsumes retired tags (pre)

**What.** The `variable copy` tag now covers what used to be two separate tags: `reaction to hypo` and `necessity for equality (hypo)`. The old tags are retired — no chapter row carries them.

**Why.** The two cases are semantically the same — a fresh `_copy` duplicate of an existing variable introduced during hypothesis handling. Separate tags were a historical accident; consolidation simplifies the verifier.

---

<a id="d-11"></a>
## D-11 — OR-branch scope classification via `classifyOrScope` (pre)

**What.** OR-scope bookkeeping no longer uses side-table flags; instead, `classifyOrScope` at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp) reads the scope's payload prefix and returns `NotOrScope | Integration | Disintegration`.

**Why.** Same motivation as D-13 — centralising scope-role info in the payload avoids parallel-state bugs. Complements the NameMap migration.

---

<a id="d-10"></a>
## D-10 — `incubator_mode` vs `ban_disintegration` — separate flags (per the project conventions)

**What.** Two distinct `ProverParameters` flags control Pass B disintegration gating: `incubator_mode` (for the incubator pipeline) and `ban_disintegration` (for other contexts — currently: compressor Phase 1). Pass B is gated on `!incubator_mode` [I-7](30_invariants.md#i-7) — not on the umbrella `ban_disintegration`.

**Why.** A previous attempt conflated them, which turned out to gate more than intended. Separating the flags clarifies intent per caller.

---

<a id="d-9"></a>
## D-9 — Pass B single-input operator gate (empirical, commit reverted)

**What.** The standalone fallback admission rule `isAllowedAsOperatorInput` at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp) is restricted to operators with `inputIndices.size == 1`. Widening to multi-input operators broke Gauss summation.

**Why.** Empirical. The gate is not theoretically justified; it is a measurement-grounded guardrail. See [I-6](30_invariants.md#i-6) + memory.

**Open question.** Whether a theoretical argument exists for single-input-only, or whether future expansion is possible under stricter admission-map rigour. See [AGENT_SwDD.md OPEN-3](AGENT_SwDD.md#open-questions).

---

<a id="d-8"></a>
## D-8 — `logicalCores = 1` hardcode

**What.** `logicalCores` is set to `1` unconditionally in the binary. Parallel smashMail was explicitly rejected in a previous session.

**Why.** Parallelism would reorder mail-drain operations in ways that break determinism guarantees. The LB grid is designed to be parallelisable, but the *specific* parallel implementation proposed was not safe.

**Future direction.** The RT campaign explores alternative parallelism schemes — per-LB lock-free mailbox, coarser cycle semantics with commit barriers — that preserve determinism.

---

<a id="d-7"></a>
## D-7 — C++ conjecturer replaces Python `create_expressions.py` (2025-late to 2026-early)

**What.** The conjecturer is now a C++ component invoked via `gl_quick.exe --conjecture <tag>`. The Python `create_expressions.py` is retired (file removed).

**Why.** Measured ~6.5× exe speedup and enabled hot-path int-path enumeration. The retired Python code produced the same conjecture set but was the pipeline bottleneck pre-migration.

**Status.** Landed. The `expression_utils.py` survivor module (string-parsing helpers) is what remained of the Python conjecture machinery.

---

<a id="d-6"></a>
## D-6 — Division-free Gauss summation (by design)

**What.** GL's Gauss batch proves `n · (n+1) = 2 · Σ i` — not `Σ i = n(n+1)/2`.

**Why.** GL avoids division by design. Expressing the Gauss formula in division-free form keeps the theorem-space within the `(+, *, s)` closed fragment. See the project conventions.

---

<a id="d-5"></a>
## D-5 — One-sided negated-equality expansion (per [I-12](30_invariants.md#i-12))

**What.** `applyEquivalenceClassToNegatedEquality` emits one-sided sibling inequalities from `!(=[a,b])`, not the symmetric cross-product.

**Why.** Combinatorial containment. Two-sided expansion blows up `|class(a)| × |class(b)|` per input. Anything the cross-product would conclude is reachable by two one-sided steps.

---

<a id="d-4"></a>
## D-4 — Incubator as separate pipeline (per the project conventions)

**What.** The incubator has its own config (`ConfigIncubator<Tag>.json`), its own theorem storage (`files/theorems_incubator/`), and its own CE-filter behaviour (`skip_ce_filter = true`). Output never enters the main proof graph.

**Why.** The incubator is a producer of ground-level fact tables, not a main theorem source. Isolating its pipeline prevents its heuristic output (via `try_contradiction`) from contaminating the main pipeline's provenance-complete proof graph.

---

<a id="d-3"></a>
## D-3 — Verifier as independent proof checker (per the project conventions)

**What.** `verifier.py` has its own copies of every algorithm it needs. It does not import from `expression_utils` or from any prover code.

**Why.** The verifier is the sole independent oracle for proof-graph correctness. If it shared code with the prover, a shared bug could go undetected. Independence is the verification commodity.

**Operational consequence.** Never modify `verifier.py` to "pass a test". See [I-16](30_invariants.md#i-16).

---

<a id="d-2"></a>
## D-2 — Hash-based inference, not search (architectural — project inception 2024-10)

**What.** GL's central bet: reformulate inference as hash-table lookup. Every implication rule is indexed by the integer-normalised signature of its premise; queries are O(1).

**Why.** Search-based provers pay the cost of discovering what to prove. GL's LBs discover that they already know what was asked. This pays forward into the ASIC roadmap — each LB becomes a silicon core with on-die SRAM for its hash memory.

**Operational consequence.** Every proof step must normalise to a stable hash key. Expression canonicalisation (`reshuffleTheorems`, `precompileStructuralOperators`) and the `IntNormalizedKey` form are what make this feasible.

---

<a id="d-1"></a>
## D-1 — Dual AGPLv3 + commercial licensing (project inception)

**What.** Every source file carries the dual-license header; commercial terms available at [https://generative-logic.com/license](https://generative-logic.com/license).

**Why.** AGPLv3 keeps derivatives open (including SaaS users); commercial licensing funds development. The AGPL stamp is also a deliberate scraper-deterrent for LLM-training pipelines — pipelines have to ingest a very explicit copyright notice on every file.

---

## Meta

- **Numbering is append-only.** D-1 is the oldest, D-42 is the newest. Once a D-number is assigned, it is immutable. (Note: D-38 is reserved — it was used by the abandoned branch's superseded canonical-min fix; see D-39.)
- **Revision pattern.** If a decision is revised, add a new D-N+1 entry and annotate the old D-K with "superseded by D-N+1 on YYYY-MM-DD".
- **Scope.** This log captures *architectural* and *tactical-but-persistent* decisions. Routine bugfixes and refactors do not warrant entries — they live in commit messages. A decision that shapes how multiple chapters of this SwDD are written belongs here.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
