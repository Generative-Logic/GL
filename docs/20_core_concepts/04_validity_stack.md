<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


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

Defined at [`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp). Per-LB instance. Purpose: map scope-name strings to `int16_t` IDs for fast comparison, while registering parent-child relationships so that ancestor queries are cheap.

Key methods:

| Method | Purpose |
|---|---|
| `encodePush(parentId, payload)` | Mint a new scope as a child of `parentId`. Returns a fresh `int16_t`. |
| `encode(name)` | Given a canonical scope-name string, return its ID (creating if needed). |
| `decode(id)` | Reverse lookup — returns the string form. **Returns a reference** — must be copied before any nested `encode` call (see [I-3](../30_invariants.md#i-3)). |
| `idToSub[id]` | Array-indexed reverse lookup. Same reference-return caveat. |
| `pairMap` | Internal table of parent-child relationships. Consulted by `comparable` / `deeperOf`. |

The `"main"` scope is a reserved root ID (sometimes referred to as `MAIN_ID` in comments) minted at LB construction.

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

**Invariant [I-2](../30_invariants.md#i-2):** every non-`"main"` validityName must be minted via `encodePush`. Raw string concatenation bypasses `pairMap` registration, producing orphan scopes that corrupt depth comparisons.

### Ancestor walk on export

The string form of a validity name carries enough structure to recover the ancestor chain without consulting `pairMap`: by [I-2](../30_invariants.md#i-2), every non-root validity is `parent_canonical + "_boundary_" + payload`. Split on `"_boundary_"` and the *strict prefixes ending at a `_boundary_` boundary* are exactly the ancestor chain — from `"main"` (always present, always root) outward to the full name. The delimiter check is mandatory: a bare string-prefix without the `"_boundary_"` boundary could match a payload that happens to share leading characters with another scope.

Two consumers walk ancestors this way:

- **`visualizer.cpp::buildStack`** (per `D-56`) — for every `(expr, validity)` it visits, it lifts to the closest-to-`"main"` ancestor whose `(expr, ancestor)` has an origin in the emitting LB's `exprOriginMap`. The chapter row emitted at the lifted scope is truthful: "the derivation lives at this scope." The walk has an **OR-branch barrier**: it may not cross `_boundary_orint_` or `_boundary_ordis_` delimiters because those scopes are conditional on a disjunct hypothesis and the OR-family verifier checkers require branch-distinct chapter-cell namespaces. See [`10_pipeline/04_prover.md`](../10_pipeline/04_prover.md#buildstack-chapter-walker-d-51-algorithm--lifting-per-d-pending-buildstack-lifting).
- **`verifier.py::_ns_matches_or_strict_prefix`** — used by `check_equality1` / `check_equality2` to match a source-validity to a target-validity when the equivalence-class rewrite spans an ancestor relationship. The helper enforces the `"_boundary_"` delimiter on the prefix match.

The runtime `pairMap` / `comparable` tables live inside `Memory` and are never exported. All export-side ancestor reasoning therefore goes through string-prefix-with-delimiter.

---

## `stackOfValidity` — the active scope stack

Per-LB. A `std::vector<int16_t>` with scope IDs pushed as scopes open and popped as they discharge. The top is the current active scope.

When `addStatement` is called, it tags the new statement with `validityName = nameMap.decode(stackOfValidity.back)` — i.e. the current innermost scope.

Scope depth is just the stack length. Two scopes are comparable via `pairMap` + `idToSub`:

- `comparable(a, b)` — returns `true` if one is an ancestor of the other (pairMap says yes).
- `deeperOf(a, b)` — returns the deeper of the two (longer ancestor chain).

Both are O(log N) in scope-tree depth with the current hash-map `pairMap`.

---

## Payloads — encoding the scope role

A scope's `payload` is the arbitrary string handed to `encodePush`. Convention: the payload *prefix* encodes the role. Current roles (inferred from `classifyOrScope`'s classification into `NotOrScope | Integration | Disintegration`):

| Role | Payload prefix | Meaning |
|---|---|---|
| `"main"` | N/A (reserved root) | The top-level scope. Every theorem's head sits here. |
| Hypothesis | `"hypo_"` (and similar) | Opened when a premise is assumed. |
| Integration | `"int_"` + specifier | Opened during reformulation-for-integration. |
| OR branch (disintegration) | `"or_dis_"` + index | Opened per disjunct on an OR disintegration. |
| OR branch (integration) | `"or_int_"` + index | Opened during OR re-integration. |
| Hypothetical-disintegration sentinel | `"hypothetical_disintegration"` (single canonical scope `main_boundary_hypothetical_disintegration`) | Throw-away structural-probe scope used by `disintegrateExprHypothetically`. Its products MUST never reach chapter emission — `visualizer.cpp::buildStack` carries a hard-assert TRIPWIRE at entry (substring match `"_boundary_hypothetical_disintegration"`) that fires if any expression on this scope reaches the chapter walker. If the TRIPWIRE fires, a new leak path has appeared; the assertion message names the offending expression. |
| Sentinel | `"sentinel_"` | Placeholder scope used for specific bookkeeping. |

*Specific payload prefixes subject to drift — grep for `"encodePush("` to enumerate current use sites. Memory file documents the convention: scope role must live in the payload, not in a Memory side table.*

---

## Scope-comparison operations

### `comparable(a, b)`

Returns whether scope `a` and scope `b` are on the same root-to-leaf path. Consults `pairMap` for the ancestor chain of each.

Used whenever the prover needs to decide "does a fact valid in scope `a` also apply in scope `b`" — it applies iff `b` is a descendant of `a`.

### `deeperOf(a, b)`

Returns whichever of `a`, `b` is deeper in the scope tree (longer ancestor chain). Used during scope-joining: when a fact derived in scope `a` is broadcast to an LB active in scope `b`, the fact's effective scope becomes `deeperOf(a, b)` — it holds wherever both scopes hold.

Both operations require `pairMap` to be correctly populated. Orphan scopes (bypassing `encodePush`) break both.

---

## Migration milestone

Per memory: the NameMap-backed validity stack was migrated. Extended on `or_5`/`or_6` with:

- Equivalence-class propagation to descendants (classes defined in scope `S` become visible in every scope deeper than `S`).
- `vacuous truth` confined to scope `"main"` (the vacuous-truth tag fires only at root; see [verifier.py](../../verifier.py)).
- Sentinel scope renamed (specific name change not captured here; grep `sentinel_` in current code to locate).

Before this migration, validityNames were free-form strings with scope relationships tracked in side tables. The migration centralised the relationship registry in `pairMap` and made depth comparisons reliable.

---

## Hot-path safety — [I-3](../30_invariants.md#i-3)

Both `NameMap::decode(id)` and `idToSub[id]` return **references** into `std::vector<std::string>`. Any nested call that may append to the vector (via `encodePush` / `encode` / any mint path) potentially invalidates the reference.

Pattern to avoid:

```cpp
const std::string& payload = nameMap.decode(scopeId);   // reference
doSomethingThatMintsAnotherScope();                      // vector may reallocate
useStringBasedOn(payload);                               // DANGLING — may read garbage
```

Fix: copy into a local immediately.

```cpp
std::string payload = nameMap.decode(scopeId);   // copy
doSomethingThatMintsAnotherScope();               // fine
useStringBasedOn(payload);                        // safe
```

This bit the codebase at least once (memory file ). Current policy: every `decode` / `idToSub[...]` read site that is followed by a mint must copy.

---

## Mode-specific scope rules

### OR branches

An OR disintegration opens one scope per disjunct via `encodePush(current, "or_dis_" + index)`. Each branch runs independently. When all branches converge on the same conclusion, `cleanUpOrIntegrationBranches` ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)) detects convergence via scope classification (`classifyOrScope`) and promotes the common conclusion to the parent scope. See [`20_core_concepts/07_or_branching.md`](07_or_branching.md).

### Contradiction

Opens a hypothetical scope with the conclusion's negation. When the branch derives both `X` and `!X`, the assumption discharges and the original conclusion is emitted into the parent scope. The contradiction-LB stays alive via `primedForContradiction` long enough for the discharge to fire.

### Integration

Reformulation-for-integration stages intermediate expressions in a scoped child. The `validity name` tag in the proof graph records which scope a conclusion is bound to. Once integration completes, the staged expressions commit to the parent as a single `expansion for integration` step.

---

## Weaknesses

### Known & tracked

- **Scope-role encoding in payloads is stringly-typed.** A typo in a payload prefix (e.g. `"or_dis "` vs `"or_dis_"`) produces a scope that doesn't match `classifyOrScope`'s expectations — silent mis-classification.
- **[I-3](../30_invariants.md#i-3) is discipline, not code-enforced.** `decode` still returns a reference in hot paths where a copy would be safer. Adding a `decodeCopy` helper that always copies is a candidate refactor — currently every call site must remember.

### Suspected fragility

- **Payload-prefix convention is load-bearing but not asserted.** `classifyOrScope` scans the payload prefix. If a new scope kind is added with a prefix that accidentally collides with an existing one, classification silently misroutes.
- **`pairMap` growth.** Every `encodePush` registers the parent-child relationship. The `pairMap` is never pruned — even discharged scopes stay in it. For long-running batches with many scope open/close cycles, `pairMap` grows unboundedly. Not currently a problem, but a memory-budget consideration for FTA-era runs.
- **Cross-LB scope consistency.** Each LB has its own `NameMap`. When a statement broadcasts from one LB to another, the receiver must translate the sender's scope ID to its own. This happens through the canonical payload string — but a mismatch in payload convention between LBs would silently misroute.

### Not exercised by tests

- **Scope discharge correctness.** No test asserts "after `pop_back` + appropriate cleanup, the scope is no longer consulted by `comparable`". A regression that left stale scope IDs in `pairMap` or `idToSub` could silently persist observations across a discharge.

---

## See also

- [`20_core_concepts/01_logic_blocks.md`](01_logic_blocks.md) — the LB owning `NameMap`, `stackOfValidity`, `pairMap`.
- [`20_core_concepts/07_or_branching.md`](07_or_branching.md) — OR-scope lifecycle.
- [I-2](../30_invariants.md#i-2), [I-3](../30_invariants.md#i-3) — scope-mint and ref-copy rules.
-,,.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
