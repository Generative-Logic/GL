<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Pipeline · Stage 4 — Compressor `[DRAFT]`

> **Input:** the batch's proved-theorem set (in-memory after stage 3).
> **Output:** a minimal subset of proved theorems that preserves derivability of every individual theorem; persisted as the pruned `files/theorems/theorems.txt` and (when externals are being rebuilt) the pruned `files/theorems/compressed_external_theorems.txt`.
> **Owner:** `compressor.cpp` / `compressor.hpp`.
> **Entry:** `Compressor::run` at [`compressor.cpp`](../../GL_Quick_VS/GL_Quick/src/compressor.cpp).

---

## What this stage does

After the prover finishes its phase, it typically has many *redundant* theorems — theorems whose statement is derivable from other theorems in the set via the already-proved implications. The compressor identifies such redundancies and removes them, producing a minimal-as-possible essential subset.

The compressor is **not** a proof-graph optimiser (it does not simplify the per-theorem proof). It is a *theorem-set* optimiser — a pruner over the full list.

Two phases:

1. **Phase 1** — Build per-theorem proof graphs. For each theorem, spin up an independent LB with *all* theorems as hash-memory rules, *target theorem's premises* as fuel, *target theorem's head* as proof goal. Run hash bursts. Extract a lightweight dependency graph (`CompressorNode`).
2. **Phase 2** — Greedy elimination. Count per-theorem usage. Sort ascending by usage, then theorem text. For each candidate, tentatively add to the dead set; check if every extracted theorem head remains derivable without it, including heads of removed theorems. If yes, keep dead. If not, mark essential.

End result: the non-dead theorems are the compressed set.

---

## Phase 1 — per-theorem proof graphs

### Setup

Entry at [`compressor.cpp`](../../GL_Quick_VS/GL_Quick/src/compressor.cpp). Flags flipped:

```
analyzer.parameters.compressor_mode    = true
analyzer.parameters.ban_disintegration = true
```

Pass B disintegration is disabled — the compressor needs only the hash-rule path to measure derivability. This is one of the two correct uses of `ban_disintegration`; contrast with [I-7](../30_invariants.md#i-7) for the incubator case.

### Per-theorem LB creation

For each of `N` proved theorems, create an independent `Memory` (named `CompressorNode_<i>`):

```cpp
Memory* lb = analyzer.lbStore.create<Memory>();
lb->level    = 0;
lb->isActive = true;
lb->setExprKey("CompressorNode_" + std::to_string(i));
```

`Compressor::runPhase1` first parses each theorem once into its premise chain and head. It then loads *every* proved theorem, including the target, into each LB as an implication rule in the original input order. Parsed strings are reused; each LB's rule installation and interners remain independent:

The complete rule-loading loop shares one existing rule-index staging window.
It closes before seeding, so every installed rule is visible to the next stage.
Premise and goal deposits share a second window, which closes before proving.
Nested installs retain the outer window; removers retain their existing
flush-before-read behavior. This changes flush frequency, preserving the
sequential records, key identities, and insertion order. Both closes assert
that they own the single outer window. The direct setup test compares all seven
index maps in all four hash memories and decoded origin rows against the
per-install-flush path, before and after seeding.

```cpp
for (const std::string& rule : all_theorems) {
    // chain and head are the cached parser result for rule.
    analyzer.addToHashMemory(chain, head, ..., *lb, lb->overallHashMemory, ...);
    const int64_t key = packStatementKey(
        lb->nameMap.encode(rule), lb->nameMap.encode("main"));
    lb->intStatementLevelsMap.assignSetRange(key, lvlNonDerived, lvlNonDerived + 1);
    upsertStatementKey(lb->intKnownStatements, key, /*local=*/true);
}
```

Split the target theorem into premises + head:

```cpp
const std::string& targetHead = theoremHeads[i];
CompressorNode pNode;
pNode.originalTheorem = theorem;
// pNode.head = targetHead; pNode.premises = targetChain
```

Run hash bursts. When the hash engine derives new expressions, their provenance is recorded in `exprOriginMap`. After the burst stabilises, walk `exprOriginMap` to build `pNode.graph` — an adjacency map from each derived expression to its dependency lists.

### The `CompressorNode` data structure

```text
CompressorNode {
    originalTheorem: string          // the full implication
    head:            ExpressionWithValidity
    premises:        set<ExpressionWithValidity>
    graph:           map<ExpressionWithValidity, vector<vector<ExpressionWithValidity>>>
                                     // expression and scope → alternative dependency lists
}
```

The `graph`'s value type is a *list of alternative dep-sets* because the hash engine may have found multiple derivation paths for the same expression. Each inner list is one way to derive the expression; at `isDerivable` time, *any* of them suffices.

---

## Phase 2 — Greedy elimination

Entry at [`compressor.cpp`](../../GL_Quick_VS/GL_Quick/src/compressor.cpp).

Steps:

1. **Count per-theorem usage** across all `CompressorNode.graph` dependency entries whose expression text matches a theorem. Repeated entries each count, independently of their scope.
2. **Sort candidates** ascending by usage, then lexicographically by theorem text. Least-used first — these are the cheapest to attempt removing (least likely to break anything).
3. **Greedy loop.** For each candidate `t`:
 - Tentatively add `t` to the `dead` set.
 - For every extracted theorem graph, test whether its head is derivable from its premises and the surviving theorem seeds. If every head remains derivable, keep `t` dead. Otherwise, mark `t` essential and remove it from `dead`.
4. **Result.** The non-dead set is the output — the minimal-essential set under this greedy heuristic.

### Removed theorems still need a derivation

`Compressor::runPhase2` checks every extracted graph. Removing a theorem removes its unconditional theorem seed; it does not remove the obligation to derive its head from the remaining seeds and its own premises. Thus two mutually dependent theorems cannot both disappear merely because both were tentatively marked dead. The check is a least fixed point, so an unseeded cycle proves neither endpoint.

---

## `CompressorReachability` — the derivability oracle

Signature:

```cpp
CompressorReachability(const CompressorNode& node,
                      const std::vector<std::string>& theorems);
bool isDerivable(const std::set<std::string>& deadTheorems) const;
```

`Compressor::runPhase2` constructs one immutable index per extracted graph before testing candidates. Each complete expression and scope pair gets a local integer index; each alternative derivation gets a head index, a dependency count, and one reverse link per dependency occurrence. Repeated dependencies retain repeated links and counter units. Premises and empty alternatives become unconditional seeds. Only main-scope theorem seeds that occur in the graph are retained.

Each query creates independent Boolean reachability state, remaining dependency counts, and a queue. It seeds premises, unconditional derivations, and surviving theorem expressions. A newly reachable expression enters the queue once and decrements only its consumers' counters. A zero counter reaches that alternative's head. The query succeeds as soon as the target is reached, or returns false when the queue is exhausted.

This computes the same least fixed point as repeatedly scanning the string graph. An unseeded cycle cannot make its first member reachable. Removed theorems may still be derived through graph edges; removal affects their initial theorem seed only. Scope identities are unchanged. The query emits no proof rows and cannot change candidate ordering or provenance.

Defined at [`compressor.cpp`](../../GL_Quick_VS/GL_Quick/src/compressor.cpp).

Measured on matched full native CUDA runs on 2026-09-06, the indexed oracle
and shared parser results reduced Peano compression from 20.0558 to 4.3120
seconds and Gauss from 4.4198 to 1.3710 seconds. All 3,490 captured output
files remained byte-identical; the full verifier and Lean kernel checks passed.
Exact commands, phase timings, and comparison evidence are in
[`rt_parts_plan.md`](../rt_parts_plan.md). `Compressor::run` prints separate
Phase 1 and Phase 2 durations for future attribution.

The Phase 1 report also separates parsing, logic-block creation, rule installation,
rule-statement registration, premise/goal seeding, the complete `prove` call,
origin decoding, graph assembly, decoded-snapshot cleanup, and logic-block
destruction. These timers wrap serial compressor orchestration; no timer enters
the statified inference functions. The complete `prove` duration includes work
outside its individual logged hash-burst iterations, so the two measurements
must not be treated as interchangeable.

The full attribution run on 2026-09-06 measured 5.39595 seconds in Phase 1:
3.003492 seconds installing rules, 1.159563 seconds seeding premises and goals,
1.053759 seconds in the complete prover call, and 0.179136 seconds across all
other stages. Graph decoding, assembly, and snapshot release together took
only 0.1143974 seconds. Setup therefore accounts for the largest remaining
cost on this workload; the detailed per-batch evidence is in
[`rt_parts_plan.md`](../rt_parts_plan.md).

---

## When the compressor runs — and when it doesn't

In the main pipeline, the compressor runs after the prover. The flag `skipCompression` is set by `run_modes::fullRun` based on the config:

```cpp
if (pp.contains("ban_disintegration") && pp["ban_disintegration"].get<bool>()) {
    skipCompression = true;
}
if (pp.contains("incubator_mode") && pp["incubator_mode"].get<bool>()) {
    skipCompression = true;
}
```

So the compressor is skipped whenever `ban_disintegration` or `incubator_mode` is active — the incubator has its own post-processing (see [`10_pipeline/09_incubator.md`](09_incubator.md)).

---

## Per-run debug dumps

Historical compressor investigations used the following development artifacts:

```
files/debug/compressor_graph_TARGET.txt           — the full CompressorNode of a target theorem
files/debug/compressor_hash_memory_TARGET.txt     — hash memory state after phase 1 target burst
files/debug/compressor_input.txt                   — the input theorem list fed to the compressor
files/debug/compressor_isderivable_TARGET_node36.txt ... — per-node isDerivable trace dumps
files/debug/compressor_origins_TARGET.txt         — origin map snapshot
files/debug/compressor_phase2_target.txt          — Phase 2 decision trace
```

The current `compressor.cpp` has no writers for these files. They are historical examples, not current observability. Any new temporary instrumentation follows the repository's dedicated trap-file contract.

---

## Weaknesses

### Known & tracked

- **Phase 1 rule storage scales O(N²).** For `N` proved theorems, each of `N` LBs loads all `N` theorems. Pool sharing or streaming would require a separate design. Current batch counts and memory costs must be measured, not inferred from earlier releases; the memory question remains OPEN-5 in `AGENT_SwDD.md`.
- **Greedy is not optimal.** The greedy order (least-used first) is a heuristic, not an optimum. A different order could produce a strictly smaller set. Current expectation: the reduction is "good enough" because the true minimum is not materially smaller on Peano/Gauss corpora. Not measured.

### Suspected fragility

- **Compiler optimization is enabled.** The current translation unit contains no active `#pragma optimize` directive.
- **`ban_disintegration` double meaning.** The compressor uses `ban_disintegration = true` as its Pass-B-disable flag. The incubator uses `incubator_mode = true` with its own gate. Confusing — the same end (Pass B off) is reached through two different mechanisms. See [I-7](../30_invariants.md#i-7).
- **Per-LB hash-memory installation is slow.** Every LB rebuilds the entire implication table via `addToHashMemory` for every rule. There is no sharing.

### Not exercised by tests

- **Full greedy-order regression coverage.** The comparator explicitly breaks usage-count ties by theorem text. Existing generic stable-sort tests alone do not pin the complete elimination result.
- **Workload-dependent performance.** The indexed oracle has direct cyclic, repeated-dependency, scope, empty-alternative, and removed-seed tests. A generated test compares 4,096 candidate sets against an independent string-set fixed-point implementation. Runtime still depends on the measured graph sizes and candidate sets; full-run timings are recorded in [`rt_parts_plan.md`](../rt_parts_plan.md).

---

## Open questions

- **OPEN-5.** What is the current Phase 1 memory footprint on the Gauss batch? Measure before FTA (where theorem counts spike). Answer belongs here.
- **OPEN-13 — RESOLVED.** `Compressor::runPhase2` sorts candidates by ascending usage count, then lexicographic theorem text; the stable sort preserves input order only for identical comparator keys. The final `essential_list` is sorted lexicographically by expanded theorem text, not by content hash. Both orders are deterministic on identical input.

---

## See also

- [`10_pipeline/04_prover.md`](04_prover.md) — producer of the uncompressed theorem set.
- [`10_pipeline/06_process_proof_graph.md`](06_process_proof_graph.md) — consumer of the compressed set + per-theorem proof graphs.
- the project conventions — "Compressor" section.
- [I-7](../30_invariants.md#i-7) — Pass B gating.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
