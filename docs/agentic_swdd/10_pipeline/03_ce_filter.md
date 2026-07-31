<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Pipeline · Stage 2 — Counterexample filter `[DRAFT]`

> Normalisation/reshuffle is no longer a separately orchestrated stage — it is a workaround inside the conjecturer's emit logic. The `reshuffled_*.txt` files are still written, but the conjecturer writes them itself.

> **Input:** `files/theorems/conjectures.txt` (conjecturer output) + `files/theorems/mirror_pairs.txt` (conjecturer-written mirror pairing for the mirror-refutation pass) + `files/simple_facts/simple_facts_<anchor>_<n>.txt` (fact tables) + `parameters.skip_ce_filter` / `parameters.mirror_refutation` flags.
> **Output:** `files/theorems/filtered_conjectures.txt` (surviving conjectures).
> **Owner:** `filter.cpp` (extracted from `prover.cpp`, 2026-05-04). Public surface (top-level entry points):
> - `readSimpleFacts` ([`filter.cpp`](../../GL_Quick_VS/GL_Quick/src/filter.cpp)) — load `files/simple_facts/*` for the active anchor.
> - `filterConjecturesWithCE` ([`filter.cpp`](../../GL_Quick_VS/GL_Quick/src/filter.cpp)) — top-level orchestrator (the per-fact-file loop body).
> - `saveFilteredConjectures` ([`filter.cpp`](../../GL_Quick_VS/GL_Quick/src/filter.cpp)) — write survivors.
>
> Internal surface (only called from inside `filterConjecturesWithCE` / the request generator):
> - `loadFactsForCEFiltering` ([`filter.cpp`](../../GL_Quick_VS/GL_Quick/src/filter.cpp)) — install a fact list into every per-core CE body.
> - `addConjectureForCEFiltering` ([`filter.cpp`](../../GL_Quick_VS/GL_Quick/src/filter.cpp)) — register one conjecture into a per-core CE LB.
> - `releaseCEBatchMemory` ([`filter.cpp`](../../GL_Quick_VS/GL_Quick/src/filter.cpp)) — DFS-delete every CE node + reset CE containers.
> - `generateEncodedRequestsStatic` ([`memory.cpp`](../../GL_Quick_VS/GL_Quick/src/memory.cpp)) — the ONE hash-request generator, invoked here with an obligatory stump of length **0**: no element is mandatory, so a grown base candidate is already the finished request and is emitted inside the search (which is what preserves the contradiction early-exit). It calls `filterIntEncodedStatements` with `alsoAcceptFullKeys = true`, and that one flag is the whole difference between CE mode's statement universe and regular mode's: CE accepts statements appearing in `normalizedEncodedSubkeys` **or** `normalizedEncodedKeys`, regular mode accepts subkeys only, because with a non-empty stump every survivor must still be growable.
>
> The CE-only `ContradictionItem` row type lives in [`filter.hpp`](../../GL_Quick_VS/GL_Quick/src/filter.hpp). All bodies remain member functions of `ExpressionAnalyzer` (declared in `prover.hpp`); only the definitions moved.
> **Entry site:** invoked near the top of `ExpressionAnalyzer::analyzeExpressions` after reading `conjectures.txt`, gated on `!parameters.skip_ce_filter`. Seen at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp).

---

## What happens here

The counterexample (CE) filter is a pre-prover pruning stage. It takes the conjecturer's candidate list and tests each candidate against a pre-computed fact table over a small finite model. If a candidate can be shown to *contradict* the facts, it is dropped before the expensive main-prover phase ever sees it.

Two observations explain why this is load-bearing:

1. **Small-model testing is cheap.** A model of size 5 (the natural numbers restricted to `{0, 1, 2, 3, 4}`) gives a finite fact table of a few thousand rows. Running a single logic-block on this table to test one conjecture takes milliseconds.
2. **Prover runtime on garbage is expensive.** Without the CE filter, the main prover would attempt to prove tautologically-false conjectures and waste its full warm-up + main-iteration budget before giving up. The prover has no affordance for "try briefly then move on"; it grinds.

Per the project conventions's "Current results" section: the CE filter consumes approximately 80% of the Gauss batch's total runtime. Optimising the fact-base size is therefore the highest-leverage performance lever for batch-level wall-clock.

---

## The fact tables — `files/simple_facts/*.txt`

One file per model size per anchor. Filename pattern (case-insensitive on the `<actual_name>` part):

```
simple_facts_<actual_name>.txt
simple_facts_<actual_name>_<n>.txt
```

The C++ reader derives `<actual_name>` from the anchor — for `AnchorPeano`, that is `peano`; for `AnchorGauss`, `gauss`. The optional `_<n>` suffix distinguishes multiple tables. When multiple files match, they are read in ascending `n` order and applied as a sequence — each pass shrinks the surviving set.

Current state ( checkout):

```
files/simple_facts/simple_facts_peano_5.txt   2680 lines
files/simple_facts/simple_facts_peano_6.txt   3752 lines
```

No Gauss or incubator fact files on this branch — they are generated by the incubator pipeline when it runs.

### What the fact table contains

Sample prefix of `simple_facts_peano_5.txt` (line 1 through 30):

```text
(AnchorPeano[N,j0,s,+,*,j1])
!(=[i4,i5])
!(=[i4,j5])
!(=[j4,i5])
!(=[j4,j5])
!(=[i0,i4])
!(=[i0,j4])
!(=[j0,i4])
!(=[j0,j4])
!(=[i0,i5])
!(=[i0,j5])
!(=[j0,i5])
!(=[j0,j5])
!(in2[i4,i0,s])
!(in2[i4,j0,s])
!(in2[j4,i0,s])
!(in2[i0,i4,s])
!(in2[i0,j4,s])
!(in2[j0,i4,s])
!(in2[i5,i0,s])
!(in2[i5,j0,s])
!(in2[j5,i0,s])
!(in2[i0,i5,s])
!(in2[i0,j5,s])
!(in2[j0,i5,s])
!(in2[i1,i0,s])
!(in2[i1,j0,s])
!(in2[j1,i0,s])
(>[v1](in2[i0,v1,s])(=[v1,i1]))
(in2[i0,i1,s])
```

Decoded:

- **Anchor** at line 1. Second slot `j0` (parallel j-copy of `i0 = 0`), sixth slot `j1` (parallel j-copy of `i1 = 1`). This is the j-copy pattern — see [j-copy strategy](#the-j-copy-strategy) below.
- **Inequalities** (lines 2–13). Every pair of distinct constants is asserted unequal.
- **Anti-successor facts** (lines 14–28). `i4 = 4` is not the successor of `0`, etc. These prune any conjecture that would require `4` to be a direct successor of `0`.
- **Quantified rule** at line 29 — `(>[v1](in2[i0,v1,s])(=[v1,i1]))`: the successor of `i0 = 0` is exactly `i1 = 1`. This is a universally quantified fact; the CE filter consumes it as a rule (hash-memory entry) rather than as a ground statement.
- **Ground arithmetic** from line 30 onward. `(in2[i0,i1,s])` — `s(0) = 1`. Lines 47 onwards show `!(in3[...])` facts for `+` — negative rows of the addition table.

The table continues through every in-model equality, every in-model negation, and every applicable operator-output combination.

### The j-copy strategy

Every `(1)`-typed constant appears in two forms: `i<k>` and `j<k>` — two parallel copies of the same model element. `i0` and `j0` both denote `0`; `i1` and `j1` both denote `1`; and so on.

The anchor line `(AnchorPeano[N,j0,s,+,*,j1])` pins the j-variables; the main-slot `i`-variables are pinned elsewhere. Every operator-output fact is emitted once per `i*j*` combination where distinctness is semantically trivial — `(in3[i4,i5,i0,+])` and `(in3[i4,j5,i0,+])` are both valid rows.

**Why.** The hash-request generator running at stump length 0 (`generateEncodedRequestsStatic`; [`memory.cpp`](../../GL_Quick_VS/GL_Quick/src/memory.cpp)) needs **distinct fact entries** to drive its rule-firing pattern — without j-copies, rules that require two different-looking arguments to match never fire on the table. This is the "CE filter j-copy requirement" documented in memory file.

### Growth pattern

Table size grows combinatorially with model size. For Peano:

| Model size | Rows | Ratio |
|---|---:|---:|
| 5 | 2680 | — |
| 6 | 3752 | ×1.40 |
| 7 | (expected ×1.5–2) | — |

The AnchorIncubator pipeline is how these tables get *built* — see [`10_pipeline/09_incubator.md`](09_incubator.md). The main pipeline only consumes them.

---

## The filtering loop

The entry point is in the prover's initialisation routine, gated on `!parameters.skip_ce_filter` (the flag is false by default; the incubator pipeline sets it to `true` because the incubator is the facts' *producer*, not consumer).

High-level flow:

```
readSimpleFacts()                         # returns vector<vector<string>>, one inner vector per file
for each fact_file in simple_facts_files:
    filteredConjectures = filterConjecturesWithCE(filteredConjectures, fact_file_contents)
saveFilteredConjectures(filteredConjectures)
```

The per-file filter is where the real work lives. Flow of `filterConjecturesWithCE` ([`filter.cpp`](../../GL_Quick_VS/GL_Quick/src/filter.cpp)):

1. **Global setup.**
 - Set `ceFilteringActive = true`.
 - Clear + reserve `contradictionTable` — one `ContradictionItem(conjecture, false)` per input conjecture. A worker sets the per-conjecture `successful` flag to `true` when the conjecture's negation is refuted by the facts.
2. **Load the fact base once** into a single *template* LB via `loadFactsForCEFiltering(facts, 1)`. Facts install through the status-4 statement path of `addExprToMemoryBlock`, so the template holds them as statements and its `overallHashMemory` stays empty — the precondition `Memory::cloneFactsTemplate` relies on.
3. **Pre-intern the CE `exprKey`s, single-threaded.** Before the pool spawns, intern `std::to_string(i)` for every conjecture index `i ∈ [0, conjectures.size)` into the process-global `skeletonInterner`. Each worker below sets its clone's identity with `lb->setExprKey(std::to_string(i))`; pre-interning makes those calls take the lookup-only (read) path, because minting the shared, lock-free `skeletonInterner` from the parallel pool is a data race — the `minted key not findable at its id` desync (see [I-82](../30_invariants.md#i-82)).
4. **Dedicated thread pool over a global conjecture queue.** A pool of `max(1, logicalCores)` workers shares one atomic conjecture index. Each worker first **publishes its scratch slot** (`g_currentCoreId = coreId`) at the top of its lambda — the whole per-conjecture task, including `addConjectureForCEFiltering`, reaches `g_currentCoreId`-resolved per-slot scratch arenas (via `addToHashMemory` → `insertRemainingArgsNormKey`, and `disintegrateExpr2`), so leaving it at the `-1` default would collapse every worker onto the single shared reserved slot and race its cursor (see [G-56](../50_gotchas.md#g-56)). Then, until the queue drains, each worker:
 - grabs the next conjecture index;
 - clones the facts template into a fresh single-use LB (`Memory::cloneFactsTemplate` — a deep value-copy of the fact containers + `nameMap`, everything else reset);
 - sets the clone's identity (`lb->setExprKey(std::to_string(i))` — a lookup-only hit after the step-3 pre-intern) and installs the conjecture's rule into the clone (`addConjectureForCEFiltering`, which also sets the clone's `contradictionIndex`);
 - runs **exactly one hashburst** on the clone — `performElemPhase1` → `performElem2` (`generateEncodedRequestsStatic` at stump length 0) → `performElemPhase2` → `performElemPhase3` — on a per-conjecture `SealedPageSet` (record chain + sealed strings) plus the worker's per-slot scratch arenas, with empty mail boxes (`splitCount = 1`, an isolated leaf LB);
 - `burstDeactivates` stops the burst the instant a refuting head fires, and phase 3's `dischargeContradiction` CE branch records the refutation into the clone's disjoint `contradictionTable` slot;
 - deletes the clone and grabs the next conjecture.
 There is **no batch barrier** — a freed worker takes new work immediately, so no core idles on the slowest conjecture in a batch.
5. **Tear-down.** `releaseCEBatchMemory` frees the template.
6. **Mirror refutation** (gated on `parameters.mirror_refutation`, default on — [D-229](../40_decisions.md#d-229)). The single-threaded flip pass `applyMirrorRefutations` ([`filter.cpp`](../../GL_Quick_VS/GL_Quick/src/filter.cpp)) snapshots which slots this pass's CE bursts refuted and, for each, flips the `contradictionTable` slots of its mirror partners to refuted as well. Pairing comes from `mirror_pairs.txt` (conjecturer-written, both columns byte-identical to `conjectures.txt` lines), loaded once per batch in `runCeFilterOnly` by `loadMirrorPairs` into the symmetric `mirrorPartnerMap` member. Only CE-confirmed refutations seed — no cascading — so the flip set is a pure function of the CE verdicts plus the pairs file. One summary line per pass: `CE filter: mirror refutation flipped N conjectures.` Refutation-side only; no mirror ever re-enters as a proof step ([I-81](../30_invariants.md#i-81) / D-112). Across multiple fact files the passes compose: a flipped conjecture drops out of the next pass's input.
7. **Collect survivors.** For each `i`, if `contradictionTable[i].successful == false` (no contradiction discovered, no mirror flip), the conjecture is kept.

CE filtering runs **one hashburst per conjecture**: `numberIterationsConjectureFiltering` is `1` in every config (asserted before the pool), so the single burst checks whether the conjecture's negation immediately contradicts the facts — there is no iterative deepening.

CE mode differs from main-prover mode in one key detail: **no mandatory elements**. Regular hash-request generation runs `generateEncodedRequestsStatic` with an obligatory stump of one or two elements, requiring an element already known to the LB to be present in the generated request. CE mode passes a stump of length 0, dropping this requirement — any combination of fact arguments is a valid hash target, so the filter can explore freely within the model.

### Data flow diagram

```
facts template LB (loaded once)  +  conjectures.txt / simple_facts_*
                 |                  (+ mirror_pairs.txt -> mirrorPartnerMap)
                 v
   filterConjecturesWithCE  --  pool of max(1, logicalCores) workers over a
                 |              global atomic conjecture queue (no batch
                 |              barrier). Per conjecture:
                 |                clone template -> install conjecture rule
                 |                one hashburst: phase1 / burst / phase3
                 |                burstDeactivates stops the burst early
                 |                dischargeContradiction -> contradictionTable[i]
                 |                delete clone
                 |              post-join: applyMirrorRefutations flips the
                 |              mirror partners of every refuted conjecture
                 v
   filtered_conjectures.txt  (survivors)
```

---

## Parameters & knobs

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `parameters.skip_ce_filter` | `bool` | `false` | Skip CE filtering entirely. Set to `true` for incubator batches via config. |
| `parameters.numberIterationsConjectureFiltering` | `int` | `1` | Hashbursts per conjecture. `1` in every config (asserted in `filterConjecturesWithCE`); the CE filter does exactly one burst per conjecture, no iterative deepening. |
| `parameters.simple_facts_parameters` | `vector<int>` | config-driven | Dead knob — parsed into the conjecturer's parameter struct only, never read by `loadFactsForCEFiltering` or any other CE code. Cleanup candidate. |
| `parameters.mirror_refutation` | `bool` | `true` | Mirror-refutation heuristic ([D-229](../40_decisions.md#d-229)): after each pass's pool joins, `applyMirrorRefutations` flips the `contradictionTable` slots of every CE-refuted conjecture's mirror partners (pairing from `mirror_pairs.txt`, loaded once in `runCeFilterOnly` via `loadMirrorPairs`). `false` disables the pass entirely — the pairs file is not even loaded. |
| `parameters.logicalCores` | `unsigned` | host logical cores | Worker count of the CE pool — `max(1, logicalCores)` threads drain the global conjecture queue in parallel. |

---

## Relationship to the incubator pipeline

The incubator pipeline (separate six-stage pipeline; see [`09_incubator.md`](09_incubator.md)) is the *producer* of `files/simple_facts/*.txt` for a given anchor-plus-model-size tuple. It enumerates every in-model fact by autonomous proof, then writes the results as a table for future main-pipeline consumption.

In other words: the CE filter is *fast* because someone else (the incubator) did the *slow* work in advance and froze the result to disk. Memory file records the current state of the incubator campaign (71% of conjectures proved/disproved at time of writing).

---

## Weaknesses

### Known & tracked

- **Historically ~80% of Gauss runtime.** the project conventions records the original figure. The per-conjecture thread pool (see Decisions) roughly halved CE wall-clock (Gauss CE 51.8s → 30.1s, Peano 37.3s → 16.2s on a 32-core host); remaining leverage is shrinking the Gauss simple-facts table or accelerating the request generator at stump length 0.
- **Incubator ⟷ main cross-contamination.** A Gauss incubator config change affected Peano CE behaviour — pipeline isolation is suspect. Memory file.
- **Clone empty-template assumption.** `Memory::cloneFactsTemplate` asserts the template's `overallHashMemory` is empty (facts load as status-4 statements, never hash rules). If a future fact base installs implication-shaped fact *rules*, the assert fires and the clone must be extended to rebuild those rules into the clone's own `keyArena` rather than value-copy the arena-backed keys.

### Suspected fragility

- **Anchor-name-based filename inference.** `readSimpleFacts` derives `<actual_name>` from the anchor string by removing the `Anchor` prefix and lowercasing. A typo in the anchor name, or a non-`Anchor*` anchor identifier, silently falls back to a no-match (empty `hits`), yielding zero fact files read and zero filtering. The only symptom is "no filtering happened" — no warning printed.
- **`simple_facts_parameters` semantics.** Corrected 2026-07-26: the field is a dead knob — parsed only into the conjecturer's parameter struct, never read by `loadFactsForCEFiltering` or any CE-side code. `ConfigPeano.json` sets it to `[]`. Cleanup candidate rather than a semantics risk.
- **j-copy assumption symmetry.** The fact tables always include `i`-copy and `j`-copy for every constant. If a future fact generator produces asymmetric j-copies (e.g., `i0` but no `j0`), the hash-request generator's expected pattern breaks with no fail-loud path.

### Not exercised by tests

- **`skip_ce_filter = true` path.** Used by incubator batches; not unit-tested per se. A regression in the non-CE-filtering codepath would surface only when running an incubator batch.
- **Multiple fact files applied sequentially.** The code reads all matching files and applies them in ascending `n` order, treating each as a shrinking pass. No regression test asserts the final survivor set is the same as if all facts had been loaded in one pass — it *should* be, but the ordering invariant is not verified.

---

## Open questions

- **OPEN-10 — RESOLVED.** `loadFactsForCEFiltering` at [`filter.cpp`](../../GL_Quick_VS/GL_Quick/src/filter.cpp) iterates `batchSize` (= `logicalCores`) and for each per-core CE body calls, for every line in the facts list, `addExprToMemoryBlock(s, *lb0, 0, 4, lvl0, origin, -1, -1, "main", false)`. So: facts are added via the **generic** `addExprToMemoryBlock` path — which handles atomic statements and implication-shaped facts differently based on their MPL shape. Atomic facts like `(in2[i0,i1,s])` become registered statements (the status-4 fact load); implication-shaped facts like `(>[v1](in2[i0,v1,s])(=[v1,i1]))` are disintegrated and installed as hash-rule implications. One loading primitive, polymorphic over fact shape.
- **OPEN-11 — RESOLVED.** No per-conjecture timeout exists. The only timing control is `parameters.numberIterationsConjectureFiltering` applied to the outer prover loop uniformly across the whole batch. A single pathologically-slow conjecture can thus starve the rest of its batch of iterations. In practice this has not bitten the project (CE filter is dominated by the number of fact-file applications, not per-conjecture work), but the asymmetry is real — a deeply-nested conjecture could consume the budget others needed. A per-conjecture step cap would be a sensible hardening. **Note (2026-06-08, ):** the per-part submatch cap `maxNumberHashRequests` added for the main-path LB split is explicitly **bypassed** in CE mode — `BurstSink::canAccept` short-circuits on `ceFilteringActive`, so a CE burst runs its full enumeration and is halted only by the contradiction early-exit, never truncated by the cap. Capping CE would make it miss refutations and wrongly keep conjectures. See [I-73](../30_invariants.md#i-73), [D-109](../40_decisions.md#d-109).

---

## See also

- [`10_pipeline/02_conjecturer.md`](02_conjecturer.md) — producer of `conjectures.txt`.
- [`10_pipeline/04_prover.md`](04_prover.md) — consumer of `filtered_conjectures.txt`.
- [`10_pipeline/09_incubator.md`](09_incubator.md) — producer of `files/simple_facts/*`.
- — j-copy requirement history.
- — isolation weakness.
- — standard debugging approach for CE filter issues.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
