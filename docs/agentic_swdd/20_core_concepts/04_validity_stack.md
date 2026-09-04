<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Core concept — Validity stack `[DRAFT]`

> Every deposited statement carries a `validityName` — the scope it is true in. Scope names are not arbitrary strings: they are minted through a `NameMap` that registers parent-child relationships, ensuring scope-depth comparisons remain correct even as hypotheses, integrations, and OR branches compose.

---

## Why scope has to be first-class

A GL proof does not live in a single namespace. The prover frequently needs to say:

- *"Assume premise P; then derive Q."* — Q is valid only under the assumption of P.
- *"Case split on (A ∨ B ∨ C)."* — each branch derives things valid only in that branch.
- *"During integration of head H, stage conclusion C."* — C is valid only in the integration context.
- *"Prove the typing sub-goal under induction on n."* — typing is valid only under the induction framework.

A flat "is this proved?" flag is not enough. The prover must distinguish between facts that hold *always* (at root scope `"main"`) and facts that hold *only under specific assumptions*. Escalating a fact from a hypothetical scope to its parent requires explicit discharge; doing so incorrectly produces unsound theorems.

GL tracks this as a literal stack of scope IDs, encoded via a per-LB `NameMap`.

---

## `NameMap` — the encoder

Defined at [`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp). Per-LB instance. Purpose: map scope-name strings to `int16_t` IDs for fast comparison, while registering parent-child relationships so that ancestor queries are cheap. The string dictionaries are façades over cold string tables in `LbMemory` (`nameStrings` / `subStrings`); the per-id scope hierarchy is a flat paged parent-pointer forest (`validityNodes`, one `ValidityNode` each) on the LB arena — the string tables and the forest all deload with the LB (see [`09_static_memory.md`](09_static_memory.md)).

Key methods:

| Method | Purpose |
|---|---|
| `encodePush(parentId, payload)` | Mint a new scope as a child of `parentId`. Returns a fresh `int16_t`. |
| `encode(name)` | Given a canonical scope-name string, return its ID (creating if needed). |
| `lookup(name)` | Non-minting probe — returns the existing ID or `0`. Burst-safe (read-only). |
| `decode(id)` | Reverse lookup — returns an **owned `std::string` copy** (the bytes live on cold pages). |
| `decodeView(id)` | Zero-copy `StrSpan` over the cold bytes — valid only while the LB is resident; do not hold across a mint/deload (see [I-3](../30_invariants.md#i-3)). |
| `stackLen`/`stackAt`/`stackBack`/`stackEmpty`, `ancLen`/`ancAt` | Element accessors over the paged payload stack / ancestor list (no reference escapes). |
| `verdict` / `comparable` / `deeperOf` | Scope-comparison, **derived from `ancestorsOf`** — there is no separate `pairMap` cache. |

The `"main"` scope is the eternal root id 1 (`MAIN_ID`). It is lazily interned as cold-table id 1 on the LB's first `encode`, so a NameMap id equals its cold-table id directly (no offset) and an unused `Memory` consumes zero pool blocks.

---

## Minting a scope — `encodePush`

The authoritative way to create a new scope. Example pattern (schematic, not real code):

```cpp
// Hypothesis scope under current scope:
int16_t parent = stackOfValidity.back();                  // current top
std::string payload = "hypo_" + exprKey;                  // role + identifier
int16_t newScope = nameMap.encodePush(parent, payload);   // mint
stackOfValidity.push_back(newScope);                       // push onto stack

// Work in new scope...

// Discharge:
stackOfValidity.pop_back();
```

The `payload` string encodes both the *role* of the scope (hypothesis, integration, OR branch, …) and an identifier distinguishing it from sibling scopes of the same role. The role prefix is conventional — a well-known prefix lets `classifyOrScope` and friends recognise OR-branch scopes without consulting a side table.

**Invariant [I-2](../30_invariants.md#i-2):** every non-`"main"` validityName must be minted via `encodePush`. Raw string concatenation bypasses `ancestorsOf` registration, producing orphan scopes that corrupt depth comparisons.

### Ancestor walk on export

The string form of a validity name carries enough structure to recover the ancestor chain without consulting `pairMap`: by [I-2](../30_invariants.md#i-2), every non-root validity is `parent_canonical + "_boundary_" + payload`. Split on `"_boundary_"` and the *strict prefixes ending at a `_boundary_` boundary* are exactly the ancestor chain — from `"main"` (always present, always root) outward to the full name. The delimiter check is mandatory: a bare string-prefix without the `"_boundary_"` boundary could match a payload that happens to share leading characters with another scope.

Two consumers walk ancestors this way:

- **`visualizer.cpp::buildStack`** (per `D-56`) — for every `(expr, validity)` it visits, it lifts to the closest-to-`"main"` ancestor whose `(expr, ancestor)` has an origin in the emitting LB's `exprOriginMap`. The chapter row emitted at the lifted scope is truthful: "the derivation lives at this scope." The walk has an **OR-branch barrier**: it may not cross `_boundary_orint_` or `_boundary_ordis_` delimiters because those scopes are conditional on a disjunct hypothesis and the OR-family verifier checkers require branch-distinct chapter-cell namespaces. See [`10_pipeline/04_prover.md`](../10_pipeline/04_prover.md#buildstack-chapter-walker-d-51-algorithm-lifting-per-d-56).
- **`verifier.py::_ns_matches_or_strict_prefix`** — used by `check_equality1` / `check_equality2` to match a source-validity to a target-validity when the equivalence-class rewrite spans an ancestor relationship. The helper enforces the `"_boundary_"` delimiter on the prefix match.

The runtime ancestor metadata (`ancestorsOf`, and `verdict` derived from it) lives inside `Memory` and is never exported. All export-side ancestor reasoning therefore goes through string-prefix-with-delimiter.

---

## `stackOfValidity` / `ancestorsOf` — the per-id scope metadata

Per-LB, indexed by validity id. Both are DERIVED views over one flat paged container — `validityNodes`, a `PagedVector<ValidityNode{parentId, ownSubId}>` (owned by `LbMemory`, bound into `NameMap` as a façade pointer), not heap vectors. Walking a scope's `parentId` chain reconstructs either list:

- `stackOfValidity[id]` — the payload sub-id stack of scope `id` (the `_boundary_`-separated payloads from root outward; empty for a root like `"main"`).
- `ancestorsOf[id]` — the ancestor validity ids of scope `id`, root-first, with `id` itself at the back.

They are **lazily seeded**: a freshly bound `NameMap` holds zero rows (zero pool blocks for a transient `Memory`); the first `encode` / `encodePush` appends the slot-0 sentinel + `"main"` row, then each minted scope appends its own run. `MAIN_ID` queries resolve to their eternal-root values (`{main}` ancestors / empty stack) even before the seed, so the answer is byte-identical seeded or not.

The active innermost scope is tracked by the prover's traversal; when `addStatement` tags a new statement, the `validityName` is the canonical name of that scope (decoded from its id). Scope comparison reads `ancestorsOf` directly — there is no `pairMap` cache (dropped; see below).

---

## Payloads — encoding the scope role

A scope's `payload` is the arbitrary string handed to `encodePush`. Convention: the payload encodes the role (self-describing — no Memory side table; memory file ). Current shapes, verified against the live mint sites:

| Role | Payload shape | Mint site |
|---|---|---|
| `"main"` | N/A (reserved root, id 1) | The top-level scope. Every theorem's head sits here. |
| Sub-implication subproof, goal-driven on MAIN | `<goal>_subproof_(implicationN[…])` | `prepareIntegrationCore2` Case A: `<goal>` is the root `toBeProved` goal (verbatim, optionally `!`-negated) whose integration spawned the subproof, so a disproof of the goal can enumerate and wipe its scopes. Parsed by `splitSubproofPayload` / `stripSubproofPrefixView`. |
| Sub-implication subproof, statement-driven or non-main | `(implicationN[…])` | `prepareIntegrationCore2` Case A with an empty `rootGoal` (admission / marker replays, load-time prefill) or a non-main parent (nested subproofs — they sit inside a goal-owned subtree already). |
| OR branch (integration) | `orint_<cleanSig>_(<head>)`; goal-driven on MAIN: `<goal>_subproof_orint_<cleanSig>_(<head>)` | `prepareIntegrationCore2` Case OR. `classifyOrScope` / `classifyOrScopeView` strip the goal prefix before classifying. |
| OR branch (disintegration) | `ordis_<orSig>_(<cleanExpr>)` | `disintegrateExpr2`. |
| Hypothetical-disintegration working scope | `_var0_<x>[_var1_<y>…]_hypo_<expr>` | `disintegrateExprHypothetically` (embeds the integrated expression — self-identifying). |
| Hypothetical-disintegration sentinel | `product_of_hypo_disintegration_of_integration_goal_<expr>` | `disintegrateExprHypothetically` — throw-away structural-probe scope; its products must never reach chapter emission (`visualizer.cpp::buildStack` tripwire). |
| Main-goal closure boundary | `<head>` pushed on MAIN (`main_boundary_<head>`) | `dischargeToBeProved` — derived purely as the wipe-subtree root for the proved goal. |

The goal-carrying `<goal>_subproof_` prefix and the goal-qualified prep gates exist for disproof cleanup ([D-239](../40_decisions.md#d-239), [I-170](../30_invariants.md#i-170)): two goals needing the same compact each prep their own subproof (accepted duplication), so wiping one goal's machinery cannot starve the other.

---

## Scope-comparison operations

### `verdict(a, b)`

The primitive: `0` if `a == b`, `-1` if `a` is a strict ancestor of `b`, `+1` if `b` is a strict ancestor of `a`, "diverge" otherwise. **Derived from `ancestorsOf`** — `a` is a strict ancestor of `b` iff `a` appears in `ancestorsOf[b]` (which carries every ancestor including self). No cache is stored; the former `pairMap` hash was dropped because it held no information `ancestorsOf` does not, and it carried an O(N) per-scope fill loop in `encodePush`.

### `comparable(a, b)`

Returns whether scope `a` and scope `b` are on the same root-to-leaf path (a `verdict` exists). Used whenever the prover needs to decide "does a fact valid in scope `a` also apply in scope `b`" — it applies iff `b` is a descendant of `a`.

### `deeperOf(a, b)`

Returns whichever of `a`, `b` is deeper in the scope tree. Used during scope-joining: a fact derived in scope `a` broadcast to an LB active in scope `b` takes effective scope `deeperOf(a, b)` — it holds wherever both hold.

All three are O(scope-depth) int16 scans of the ancestor run (single digits in practice) and run only on single-threaded paths (never the parallel burst, which uses non-minting `lookup`). Orphan scopes (bypassing `encodePush`) break them — their ancestor list would be wrong.

---

## Migration milestone

Per memory: the NameMap-backed validity stack was migrated. Extended on `or_5`/`or_6` with:

- Equivalence-class propagation to descendants (classes defined in scope `S` become visible in every scope deeper than `S`).
- `vacuous truth` confined to scope `"main"` (the vacuous-truth tag fires only at root; see [verifier.py](../../verifier.py)).
- Sentinel scope renamed (specific name change not captured here; grep `sentinel_` in current code to locate).

Before this migration, validityNames were free-form strings with scope relationships tracked in side tables. The migration centralised the relationship registry and made depth comparisons reliable.

## Migration milestone

The NameMap's id-form bookkeeping moved onto the cold paged tier (statification):

- `stackOfValidity` / `ancestorsOf` became derived walks over a flat paged parent-pointer forest (`validityNodes` in `LbMemory`, one `ValidityNode{parentId, ownSubId}` per id), deload-persisted with the LB, lazily seeded on first encode (`D-138`; [`09_static_memory.md`](09_static_memory.md)).
- `pairMap` was dropped — `verdict` / `comparable` / `deeperOf` derive from `ancestorsOf` (an int16 membership scan).
- The tid+1 / off-table-`"main"` indirection was removed: a NameMap id equals its cold-table id directly, with `"main"` lazily interned as cold-table id 1 (the eternal-root contract covers the pre-intern window). This restores the direct int↔string mapping `main` always had.

---

## Hot-path safety — [I-3](../30_invariants.md#i-3)

`NameMap::decode(id)` now returns an **owned `std::string`** (the bytes live on cold pages), so it is always safe. The reference hazard survives only for **views into paged storage**: `decodeView(id)` returns a `StrSpan` over the cold name bytes, and an element read of the paged metadata aliases page memory. A nested mint (`encode` / `encodePush`) or a deload can grow/relocate a page and dangle such a view.

The former `stackOf(id)` — which returned a `const std::vector<int16_t>&` into the heap metadata — has been **retired**; callers use the element accessors (`stackLen`/`stackAt`/`stackBack`/`ancLen`/`ancAt`), and `encodePush` reads a parent run into a transient scratch *before* the append that grows the page. So no long-lived reference into the paged containers escapes.

Pattern to avoid (for `decodeView`):

```cpp
StrSpan payload = nameMap.decodeView(scopeId);   // span into cold bytes
doSomethingThatMintsAnotherScope();              // a page may relocate
useStringBasedOn(payload);                       // DANGLING — may read garbage
```

Fix: use `decode` (owned copy), or finish using the span before any mint. This bit the codebase at least once (memory file ).

---

## Mode-specific scope rules

### OR branches

An OR disintegration opens one scope per disjunct via `encodePush(current, "ordis_" + index)`. Each branch runs independently. When all branches converge on the same conclusion, `cleanUpOrIntegrationBranches` ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)) detects convergence via scope classification (`classifyOrScope`) and promotes the common conclusion to the parent scope. See [`20_core_concepts/07_or_branching.md`](07_or_branching.md).

### Contradiction

A contradiction LB carries its proof assumption structurally, while its inference scopes still begin at that LB's `main`. The assumption discharges only when both `X` and `!X` are present at that `main`; a pair whose shallowest common validity is an `_ordis_` branch cannot discharge the LB. If an expression also has a deeper duplicate, its separate main row still qualifies. The contradiction LB stays alive via `primedForContradiction` long enough for the main-scope discharge to fire. Contradictory-branch rejection and `_ordis_` cohort rewriting remain unimplemented.

### Integration

Reformulation-for-integration stages intermediate expressions in a scoped child. The `validity name` tag in the proof graph records which scope a conclusion is bound to. Once integration completes, the staged expressions commit to the parent as a single `expansion for integration` step.

### Cleanup on closure — radical subtree wipe

When an implication subproof at rooted scope `S = main_boundary_…_boundary_(implicationN[…])` closes via successful proof, every piece of per-LB state whose attached validity equals `S` or starts with `S + "_boundary_"` is physically removed. The mechanism — `Memory::wipeSubtree(closedVid)` defined in [`memory.cpp`](../../GL_Quick_VS/GL_Quick/src/memory.cpp), taking the closed scope's `NameMap` validity id (minted + non-main, assert-enforced) and decoding it to `S` at the top — replaces the pre-existing selective cleanup (`cleanUpIntegrationPreparation` + `cleanUpIntegrationPreparationCore`) that erased only the rooted scope and its `_var0_*` fresh-binding children, leaving deeper sub-scopes (`_orint_…`, nested hypotheticals) as inert orphans behind the `intValidityNamesToFilter` ancestor-scan filter.

**Deferral to burst boundary.** The wipe does not run immediately at the impl-discharge call site. The closed scope's validity id is queued into `Memory::pendingWipeScopes`; at the end of the burst (`performElemPhase3`, after `sanitizeHashMemory` + `sanitizeToBeProved`, before the EXIT trap) the queue is drained and `wipeSubtree` runs once per queued scope id. Immediate mid-kernel wipe would erase `intStatementLevelsMap` / `equivalenceClassesMap` entries that subsequent iterations of `addExprToMemoryBlockKernel`'s `sortedNew` loop assert-look-up. See [I-50](../30_invariants.md#i-50).

`cleanUpOrIntegrationBranches` (the OR-convergence sibling-wipe path) routes through the same queue — each victim scope is inserted into `pendingWipeScopes` rather than processed inline.

### Cleanup on disproof — the disproved-goal drain

When a primed `__contradiction__` child settles a MAIN goal, the post-join theorem drain deposits the contradiction seed onto the parent's `Memory::pendingDisprovedGoals` (persistent-pool byte set, writable while the parent is cold); the parent's next end-of-burst `drainDisprovedGoals` — running immediately before the `pendingWipeScopes` drain — probes both directions. A DISPROOF hit (seed == goal) erases the goal row, queues every goal-embedding MAIN scope for the radical wipe (the `<goal>_subproof_` payloads plus the two hypo shapes), erases the goal-template gates, the dead goal's `exprOriginMap` rows, internal-mail origin runs at wiped scopes, and dead OR cohorts; full mechanism, the two disproof-only deviations (origin-map erase vs I-44; cohort erase vs I-167), and the sanctioned-remnant list: [D-240](../40_decisions.md#d-240). A PROOF hit (`negate(seed)` == goal — the emitted theorem's head IS the goal) closes the goal with success semantics: row erased, the same goal-embedding scopes queued for the wipe, everything else — history, gates, cohorts — kept ([D-279](../40_decisions.md#d-279)); this is what lets a level-poor contradiction-scope proof release its LB chain even though the level-gated local discharge never fires.

The full structure list (containers wiped, containers preserved on purpose) lives at [D-72](../40_decisions.md#d-72). Known interaction with monotonic iter-var counters: [G-43](../50_gotchas.md#g-43).

---

## Weaknesses

### Known & tracked

- **Scope-role encoding in payloads is stringly-typed.** A typo in a payload prefix (e.g. `"or_dis "` vs `"ordis_"`) produces a scope that doesn't match `classifyOrScope`'s expectations — silent mis-classification.
- **[I-3](../30_invariants.md#i-3) is discipline, not code-enforced.** `decode` still returns a reference in hot paths where a copy would be safer. Adding a `decodeCopy` helper that always copies is a candidate refactor — currently every call site must remember.

### Suspected fragility

- **Payload-prefix convention is load-bearing but not asserted.** `classifyOrScope` scans the payload prefix. If a new scope kind is added with a prefix that accidentally collides with an existing one, classification silently misroutes.
- **Validity-metadata growth.** Every `encodePush` appends one `ValidityNode` to `validityNodes` (4 bytes); it is never pruned (the forest grows monotonically), even for discharged scopes. It is paged on the LB arena and deloads with the LB, so the footprint is cold rather than heap, and the flat parent-pointer form is ~7× leaner than the earlier jagged lists — but still a memory-budget consideration for FTA-era runs. (The former `pairMap` hash that compounded this is gone — `verdict` walks the forest.)
- **Cross-LB scope consistency.** Each LB has its own `NameMap`. When a statement broadcasts from one LB to another, the receiver must translate the sender's scope ID to its own. This happens through the canonical payload string — but a mismatch in payload convention between LBs would silently misroute.
- **Settled-goal verbatim match.** `drainDisprovedGoals` matches the deposited contradiction seed against MAIN goals and scope payloads verbatim, in both settlement directions; a goal rewritten by equivalence classes between spawn and settlement would miss the probe indistinguishably from the legitimate already-closed miss. Not exercised today — the contradiction pipeline runs only in incubator batches, which skip equivalence classes ([D-240](../40_decisions.md#d-240)). A canonicalization-aware fallback (canonicalizing both sides under the classes at that validity, the way `filterIterationsCore` selects a canonical) is the known fix if a main batch ever combines classes with `try_contradiction`.

### Not exercised by tests

- **Scope discharge correctness.** No test asserts "after discharge + appropriate cleanup, the scope is no longer consulted by `comparable`". A regression that left stale scope IDs in `ancestorsOf` could silently persist observations across a discharge.

---

## See also

- [`20_core_concepts/01_logic_blocks.md`](01_logic_blocks.md) — the LB owning `NameMap` and `validityNodes`.
- [`20_core_concepts/07_or_branching.md`](07_or_branching.md) — OR-scope lifecycle.
- [`20_core_concepts/09_static_memory.md`](09_static_memory.md) — the paged `validityNodes` parent-pointer forest + its deload.
- [I-2](../30_invariants.md#i-2), [I-3](../30_invariants.md#i-3) — scope-mint and ref-copy rules; plus `I-104`, `I-105`, `D-138`.
-,,.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
