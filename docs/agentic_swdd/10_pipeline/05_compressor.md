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
2. **Phase 2** — Greedy elimination. Count per-theorem usage. Sort ascending (least-used first). For each candidate, tentatively add to the dead set; check if every surviving theorem remains derivable without it. If yes, keep dead. If not, mark essential.

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
Memory* lb = new Memory();
lb->level    = 0;
lb->isActive = true;
lb->exprKey  = "CompressorNode_" + std::to_string(i);
```

Into this LB, load *every* proved theorem as an implication rule:

```cpp
for (const std::string& rule : all_theorems) {
    std::string head = ce::disintegrateImplication(rule, tempChain, analyzer.coreExpressionMap);
    analyzer.addToHashMemory(chain, head, ..., *lb, lb->overallHashMemory, ...);
    upsertStatementKey(lb->intKnownStatements,
        packStatementKey(lb->nameMap.encode(rule), lb->nameMap.encode("main")),
        true, /*registered=*/true, /*known=*/false);
}
```

Split the target theorem into premises + head:

```cpp
std::string targetHead = ce::disintegrateImplication(theorem, targetChain, analyzer.coreExpressionMap);
CompressorNode pNode;
pNode.originalTheorem = theorem;
// pNode.head = targetHead; pNode.premises = targetChain
```

Run hash bursts. When the hash engine derives new expressions, their provenance is recorded in `exprOriginMap`. After the burst stabilises, walk `exprOriginMap` to build `pNode.graph` — an adjacency map from each derived expression to its dependency lists.

### The `CompressorNode` data structure

```text
CompressorNode {
    originalTheorem: string          // the full implication
    head:            string          // target expression to derive
    premises:        list<string>    // the "fuel" — the chain's premises
    graph:           map<string, list<list<string>>>
                                     // expr → list of alternative dep-sets (from exprOriginMap)
}
```

The `graph`'s value type is a *list of alternative dep-sets* because the hash engine may have found multiple derivation paths for the same expression. Each inner list is one way to derive the expression; at `isDerivable` time, *any* of them suffices.

---

## Phase 2 — Greedy elimination

Entry at [`compressor.cpp`](../../GL_Quick_VS/GL_Quick/src/compressor.cpp).

Steps:

1. **Count per-theorem usage** across all `CompressorNode.graph` edges. Each rule cite against a non-premise expression is a usage.
2. **Sort candidates** ascending by usage. Least-used first — these are the cheapest to attempt removing (least likely to break anything).
3. **Greedy loop.** For each candidate `t`:
 - Tentatively add `t` to the `dead` set.
 - For every surviving (non-dead) theorem `s`, run `isDerivable(nodes[s], dead)`. If all surviving theorems are derivable, keep `t` dead. Otherwise, mark `t` essential and remove from `dead`.
4. **Result.** The non-dead set is the output — the minimal-essential set under this greedy heuristic.

### The surviving-only detail

**Critical.** When checking derivability of theorems, only the non-dead theorems are checked. Checking dead theorems would create false dependencies where two dead "twin" theorems protect each other from elimination (each derives the other, so removing both breaks neither's derivability — but they are both dead by assumption, which is nonsensical).

This detail is recorded explicitly in the project conventions's "Compressor" section and is load-bearing for the heuristic's correctness.

---

## `isDerivable` — the oracle

Signature:

```cpp
bool Compressor::isDerivable(const CompressorNode& node,
                              const std::set<std::string>& dead_theorems)
```

Algorithm (forward reachability):

1. **Seed the `alive` set:**
 - Every premise of `node` is alive (definitional — they are the fuel).
 - Every non-`dead` theorem is alive.
 - Every expression with an empty dep-set in `node.graph` is alive (unconditionally derivable).
2. **Iterate.** For each expression `e` in `node.graph` not yet alive: if any of its dep-sets (alternative derivations) has all its deps alive, mark `e` alive.
3. **Converge.** Repeat until no new additions.
4. **Return.** Whether `node.head` is alive.

Defined at [`compressor.cpp`](../../GL_Quick_VS/GL_Quick/src/compressor.cpp).

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

When running with compressor debug hooks enabled (development only), the compressor writes diagnostic files to `files/debug/`:

```
files/debug/compressor_graph_TARGET.txt           — the full CompressorNode of a target theorem
files/debug/compressor_hash_memory_TARGET.txt     — hash memory state after phase 1 target burst
files/debug/compressor_input.txt                   — the input theorem list fed to the compressor
files/debug/compressor_isderivable_TARGET_node36.txt ... — per-node isDerivable trace dumps
files/debug/compressor_origins_TARGET.txt         — origin map snapshot
files/debug/compressor_phase2_target.txt          — Phase 2 decision trace
```

These are present on the current branch from an earlier debugging session. Useful when a compression pass produces a theorem set that breaks later stages.

---

## Weaknesses

### Known & tracked

- **Phase 1 memory cost scales O(N²).** For `N` proved theorems, each of `N` LBs loads all `N` theorems. At current batch sizes (Peano 58, Gauss ≈ 60 pre-compression), this is fine. At FTA scale (expected ~100K per run per memory file ), Phase 1 becomes untenable without a rework — pool sharing or streaming. Flagged as OPEN-5 in `AGENT_SwDD.md`.
- **Greedy is not optimal.** The greedy order (least-used first) is a heuristic, not an optimum. A different order could produce a strictly smaller set. Current expectation: the reduction is "good enough" because the true minimum is not materially smaller on Peano/Gauss corpora. Not measured.

### Suspected fragility

- **`#pragma optimize("", off)` at [`compressor.cpp`](../../GL_Quick_VS/GL_Quick/src/compressor.cpp).** Disables optimisation for Phase 1 — historically a debugging artefact. the project conventions notes this pragma is commented out for full optimisation in the incubator-related code, but the compressor's still carries it. Impact: possibly slower Phase 1 than it could be.
- **`ban_disintegration` double meaning.** The compressor uses `ban_disintegration = true` as its Pass-B-disable flag. The incubator uses `incubator_mode = true` with its own gate. Confusing — the same end (Pass B off) is reached through two different mechanisms. See [I-7](../30_invariants.md#i-7).
- **Per-LB hash-memory installation is slow.** Every LB rebuilds the entire implication table via `addToHashMemory` for every rule. There is no sharing.

### Not exercised by tests

- **Phase 2 tie-breaking determinism.** Two theorems with identical usage counts have an undefined order. A non-deterministic sort (e.g., unstable `std::sort` on an unhashed key) could produce different "minimal" sets on different runs — and since the compressor output is the prover's published set, this would surface as flaky regressions across runs. Not tested.
- **`isDerivable` correctness on circular dep-set graphs.** The alive-set iteration handles cycles correctly by construction (fixed-point). But a targeted test for "A → B, B → A, neither is a premise" is absent.

---

## Open questions

- **OPEN-5.** What is the current Phase 1 memory footprint on the Gauss batch? Measure before FTA (where theorem counts spike). Answer belongs here.
- **OPEN-13 — RESOLVED.** The Phase 2 candidate sort uses `std::stable_sort` at [`compressor.cpp`](../../GL_Quick_VS/GL_Quick/src/compressor.cpp) over `sorted_theorems` — the insertion order of the input `all_theorems` vector breaks ties in `stable_sort`, which is deterministic because `all_theorems` is loaded in a fixed order from `theorems.txt`. A separate `std::sort` at [`compressor.cpp`](../../GL_Quick_VS/GL_Quick/src/compressor.cpp) sorts the final `essential_list` for output ordering — not load-bearing because that sort operates on content hashes, not on the elimination heuristic. Conclusion: compression output is deterministic across runs on identical input.

---

## See also

- [`10_pipeline/04_prover.md`](04_prover.md) — producer of the uncompressed theorem set.
- [`10_pipeline/06_process_proof_graph.md`](06_process_proof_graph.md) — consumer of the compressed set + per-theorem proof graphs.
- the project conventions — "Compressor" section.
- [I-7](../30_invariants.md#i-7) — Pass B gating.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
