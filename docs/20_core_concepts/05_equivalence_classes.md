<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Core concept — Equivalence classes `[DRAFT]`

> When the prover derives `(=[a,b])`, it records that `a` and `b` are equal in the current scope. That fact is not just stored — it propagates. Subsequent expressions mentioning `a` are admitted with `b` substituted, and vice versa. Inequality propagation is the dual — `!(=[a,b])` with an existing equality class expands into sibling inequalities.

---

## The data structure — `EquivalenceClass`

Defined at [`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp). Fields:

| Field | Type | Role |
|---|---|---|
| `variables` | `set<string>` | The members of this class — all known-equal. |
| `equalityLevelsMap` | `map<var-set, levels>` | Per-pair admission levels (for admission-map gating). |
| `equalityOriginMap` | origin entries | Provenance — which equalities produced this class. |

An `EquivalenceClass` is owned per-`validityName` on each `Memory`: `Memory::equivalenceClassesMap` is a `map<validityName, EquivalenceClass>`. Different scopes can have different classes — an equality asserted under a hypothesis does not bleed into the parent unless the hypothesis discharges.

---

## Registration — when a class forms

`addStatement` (see [prover chapter](../10_pipeline/04_prover.md#addstatement-central-ingestion)) handles equality ingestion. When `(=[a,b])` arrives:

1. If `a == b`, the statement is trivially true and is dropped (no new information).
2. If neither `a` nor `b` is in an existing class, create a new class `{a, b}`.
3. If one of `a` or `b` is in an existing class `C`, add the other to `C`.
4. If both are in existing classes `C_a` and `C_b`, merge them — the union becomes the new class.

Per-class levels are maintained to respect the admission-map gating — a fact derivable through equality at level `L` is itself at level `L`.

### Origin tracking and the mail-sync rule

Each class carries `equalityOriginMap` — provenance for the equalities the class represents. The map is populated from three sources:

- The incoming equality's `origin` parameter, recorded by `updateEquivalenceClasses` at [`prover.hpp::updateEquivalenceClasses`](../../GL_Quick_VS/GL_Quick/src/prover.hpp).
- Cross-pair `equality2` records emitted by `mergeTwoEquivalenceClasses` ([`prover.hpp::mergeTwoEquivalenceClasses`](../../GL_Quick_VS/GL_Quick/src/prover.hpp)) — see *Cross-pair `equality2` emission* below.
- **Mail-bulk-merge sync** at [`prover.cpp::performElementaryLogicalStep`](../../GL_Quick_VS/GL_Quick/src/prover.cpp). Immediately after the LB's bulk-merge of `body.mailIn.exprOriginMap` into `body.exprOriginMap`, every positive 2-arg equality entry `(=[a,b]) @ V` whose vars are already class-bound at `V` has its mail origins additionally registered in that class's `equalityOriginMap`. Equalities whose vars are not yet class-bound are picked up later via `updateEquivalenceClasses`'s line-5886 seed when their own absorption (`prover.cpp:1483-1502`) fires.

The mail-sync exists because the class's `equalityOriginMap` is the source of truth that the merge cross-pair logic consults: without sync, mail-arrived derivations are invisible to the class, and the merge re-derives them via `equality2` transitive routes — producing cycles when multiple bridges exist. See [03_mail_system.md](03_mail_system.md#the-cycle-boundary-protocol) for the producer side and [50_gotchas.md](../50_gotchas.md) for the historical chapter-22 / theorem-12 cycle that motivated this rule.

---

## Propagation — the mirror emission

When `(=[a,b])` is admitted, the prover automatically emits `(=[b,a])` (guarded by `a!= b`, see [I-9](../30_invariants.md#i-9)). Equality is thereby made reflexive *by construction* at the statement level, without requiring a `reflexivity` rule in every implication table.

Argument substitution (the "equality1" tag) is handled during hash-request generation: when producing requests from an expression, the prover also considers variants where each argument is replaced by any known equivalent. This means a rule keyed on `(P[a, …])` will also fire on `(P[b, …])` whenever `a ≡ b`, without the index needing a separate entry for every equivalence class.

---

## Transitivity

`equality2` (transitivity of equality) is captured through the class structure itself. If `(=[a,b])` and `(=[b,c])` are both admitted, both `a` and `c` end up in the same class — the transitivity conclusion `(=[a,c])` is a direct class-membership query, not a derived rule.

In the processed proof graph, the verifier still expects an explicit `equality2` tag row when transitivity is the justification for a derivation step. The checker walks the origin chain to verify the two equality premises that produced the transitive conclusion.

---

## Negated equality — the asymmetric expansion

`addStatement` has special handling for `!(=[a,b])`. When this arrives at scope `V` with existing equivalence class(es) touching either arg, `applyEquivalenceClassToNegatedEquality` ([`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp)) emits sibling inequalities:

- For each `c ∈ class(a) \ {a}` — emit `!(=[c, b])`.
- For each `d ∈ class(b) \ {b}` — emit `!(=[a, d])`.

**The symmetric cross-product is deliberately not emitted.** That is, the prover does *not* additionally emit `!(=[c, d])` for `c ∈ class(a)`, `d ∈ class(b)` — even though that combination is semantically derivable. See [I-12](../30_invariants.md#i-12).

Why skip the cross-product: it would blow up combinatorially (|class(a)| × |class(b)| emissions per input). The one-sided expansions are sufficient — anything the cross-product would derive is derivable by two one-sided steps, one for each arg.

---

## Per-deposit emission block

Every emission from `applyEquivalenceClassToNegatedEquality` goes through the standard local-deposit block inside `addStatement`:

```
wholeExpressions.insert(encoded)
statementLevelsMap[encoded] = levels
intKnownStatements.insert(intKey)
encodedStatements.insert(encoded)
intEncodedStatements.insert(intKey)
if (local) {
    localEncodedStatements{,Delta}.insert(encoded)
    intLocalEncodedStatements{,Delta}.insert(intKey)
}
newStatements.push_back(encoded)
if (validityName == "main")
    mailOut.statements.push_back(expr, levels)
```

So each sibling inequality enters the same admission pipeline as a freshly-derived expression — it is not short-circuited.

**Precondition assert:** entry requires `isNegatedEquality(expr)`. Callers must gate on the same check. Violating this assert has been a historical bug-source; the user has explicitly required the assert stay intact (memory: ).

---

## Equivalence-class visibility

Per memory file — on the migration and forward — equivalence classes defined in scope `S` become visible in every scope deeper than `S`. A class registered in `"main"` is visible everywhere; a class registered in an OR-branch scope is visible only in that branch and its sub-scopes.

The classification algorithm uses `comparable(scope, classScope)` — see [`20_core_concepts/04_validity_stack.md`](04_validity_stack.md) for the depth-comparison mechanics.

---

## Cross-scope application — bidirectional ([D-33](../40_decisions.md#d-33))

Equivalence-class application now runs in both directions whenever the class's validity scope and the expression's validity scope are comparable:

| Class scope | Expression scope | Deposit scope | Direction |
|---|---|---|---|
| `S` | `S` | `S` | same-NS (legacy) |
| `S_a` (strict ancestor of `S_d`) | `S_d` | `S_d` | class shallower (legacy) |
| `S_d` (strict descendant of `S_a`) | `S_a` | `S_d` | class deeper (NEW — D-33) |

The deposit scope is `deeperOf(class.scope, expr.scope)` — see [`memory.hpp::NameMap::deeperOf`](../../GL_Quick_VS/GL_Quick/src/memory.hpp). The new direction is sound under descendant-inheritance: a fact at `S_a` is observably true at every descendant of `S_a`, including `S_d`. A class registered at `S_d` can therefore legitimately rewrite the inherited fact and produce a fact at `S_d`.

The original at `S_a` is **never overwritten** — the rewrite is purely additive. Both copies coexist:

| | Lives at | Touched by class application? |
|---|---|---|
| Original `(P[a]) @ S_a` | `S_a` only | No |
| Rewrite `(P[b]) @ S_d` | `S_d` only | Yes — newly inserted |

This is what unblocked FTA-rung-1 §9b: a class registered at an OR-disintegration branch scope (descendant of `main`) rewrites a ground `preorder` fact at `main` and deposits the result at the branch.

### Trigger sites

The cross-scope rewrite fires from three sites, each iterating over a different pairing of (class scope) × (expression scope):

1. **`updateEquivalenceClasses` merge-postscan** ([prover.hpp](../../GL_Quick_VS/GL_Quick/src/prover.hpp)) — a new equality merges into a class at `validityName`; the postscan walks every encoded statement and applies the merged class to any whose scope is comparable to the class's. Three direction cases (same / shallower / deeper) all route into the caller's `newStatements`.
2. **`addStatement` per-statement block** ([prover.hpp](../../GL_Quick_VS/GL_Quick/src/prover.hpp)) — a new fact arrives at `validityName`; the block iterates classes at `validityName`, at each strict ancestor, AND (NEW) at each strict descendant, applying each. Descendant-class deposits land at the class's deeper scope.
3. **`addStatement` fixpoint** ([prover.hpp](../../GL_Quick_VS/GL_Quick/src/prover.hpp)) — `applyClassesFrom` lambda re-runs each iteration, picking up classes that get registered mid-`addStatement` (equality propagation can derive new equalities). Same three direction cases.

### Single-channel deposit ([I-25](../30_invariants.md#i-25))

Every deposit — same-scope, shallower-class, deeper-class — flows into the **single** `newStatements` vector that `addStatement` returns. There is no separate sink for cross-scope deposits. Each entry is an `ExpressionWithValidity` pair carrying the deposit's actual scope (see [I-25](../30_invariants.md#i-25)).

The kernel's post-`addStatement` loop ([prover.cpp](../../GL_Quick_VS/GL_Quick/src/prover.cpp)) iterates these pairs and uses each pair's `validityName` for `statementLevelsMap` lookup, admission-map updates, `toBeProved` discharge, and validity-name promotion. Cross-scope deposits therefore go through the same discharge logic as same-scope deposits — the only difference is which `validityName` drives the lookup.

This routing replaced an earlier WIP design (the original D-33 commit, prior to the routing fix) where cross-scope deposits went into separate sinks (`crossScopeSink`, `descendantSink`, `ancestorSink`) to avoid tripping the kernel's same-scope assert. That design caused cross-scope deposits to bypass `toBeProved` discharge entirely — facts were inserted at the right key but matching `toBeProved` entries never fired. The pair-based return type closes that gap.

### What the rejected-integration map does NOT do

`applyEquivalenceClassToRejectedMapIntegration` does NOT admit the descendant direction (class strictly deeper than rmi entry). The rmi map is a deferred-match registry — its entries represent auxy integrations whose preconditions live in the world visible at the entry's scope. A class at a strictly deeper scope is invisible there, so it carries no information for that entry's deferred match; rewriting and erasing the entry would destroy the original revival path without compensating at the descendant. See [D-33](../40_decisions.md#d-33).

---

## Cross-scope class merge ([D-44](../40_decisions.md#d-44))

`updateEquivalenceClasses` ([`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp), `updateEquivalenceClasses`) absorbs same-NS classes that overlap `eqArgs` (the new equality's args) into a fresh `mergedClass` at `validityName`, then **also iterates strict ancestors** of `validityName` and absorbs any ancestor classes that overlap `eqArgs`. The ancestor classes themselves stay UNCHANGED at their scope ([I-31](../30_invariants.md#i-31)).

| Class scope | Equality scope | Resulting merged class scope | Ancestor class touched? |
|---|---|---|---|
| `S` | `S` | `S` (replaces same-NS originals) | n/a — same-NS |
| `S_a` (strict ancestor of `S_d`) | `S_d` | `S_d` (additive — ancestor's vars/levels/origins copied in) | **No** — `equivalenceClassesMap[S_a]` and `eqClassSttmntIndexMapMap[S_a]` untouched |
| `S_d` (strict descendant of `S_a`) | `S_a` | not merged (out of scope) | n/a — descendant class invisible at `S_a` |

Soundness. A class at `V_a` is observably visible at every descendant of `V_a` (every pair `var ≡ var'` in the ancestor class holds at every descendant). An equality `(=[a,b])` admitted at `V` (descendant) with `a ∈ C_a` therefore propagates `b ≡ all of C_a` at `V`. Adding `C_a`'s members to `mergedClass @ V` is locally sound at `V`. The original `C_a @ V_a` retains its semantics — the new equality is invisible at `V_a` (same-or-deeper visibility rule), so `V_a`'s class must not be modified.

Mirrors the additive principle from [D-33](../40_decisions.md#d-33) (cross-scope deposits never overwrite the original at the ancestor) and [D-43](../40_decisions.md#d-43) (rmi rewrites are additive at the new key). All three — apply-side ([D-33](../40_decisions.md#d-33)), rmi ([D-43](../40_decisions.md#d-43)), and merge-side ([D-44](../40_decisions.md#d-44)) — share one rule: ancestor scope state is never written by descendant-scope work.

The descendant direction (class strictly deeper than `validityName`) is **not** admitted, by symmetry with [D-33](../40_decisions.md#d-33)'s rmi exclusion: a class at a deeper scope is invisible at the equality's scope, so it carries no information for an equality admitted at the ancestor.

The ancestor pass uses the same primitive — `mergeTwoEquivalenceClasses` ([`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp), `mergeTwoEquivalenceClasses`) — that the same-NS pass uses. The function takes `classB` by `const&`, so the ancestor class is read-only by construction. The bridge invariant (`commonArg ∈ eqArgs ∩ classA ∩ classB`) is preserved because the overlap test is against `eqArgs` directly.

---

## Cross-substitution `equality1` emission gating ([D-48](../40_decisions.md#d-48), [I-34](../30_invariants.md#i-34)) — producer-side redundancy guard

> **Status note.** D-48's gate is a producer-side redundancy guard, not the system-level cycle-resolution mechanism. The actual cycle-resolution is **[D-49](../40_decisions.md#d-49) / [I-35](../30_invariants.md#i-35)** (`addOrigin`'s cap-full preference replacement plus the bulk-merge routing change). This subsection describes D-48's gate verbatim; see *Origin selection at cap* below for D-49.

`applyEquivalenceClass` rewrites a non-equality expression by substituting equivalence-class peers into its arguments. For each rewrite, it (a) deposits the rewritten expression into `memoryBlock.encodedStatements` and (b) emits an `equality1` origin record naming the source expression + the justifying equalities.

**Gate.** The emission only fires when the target rewrite is *not already in* `memoryBlock.exprOriginMap`. The lambda lives inside the `trackHistory` block:

```cpp
auto alreadyHasOrigin = [&](const ExpressionWithValidity& ev) -> bool {
    auto it = memoryBlock.exprOriginMap.find(ev);
    return it != memoryBlock.exprOriginMap.end() && !it->second.empty();
};
```

The deposit (statementLevelsMap, encodedStatements, newStatements, mailOut.statements) is *not* gated — only the origin emission is. The previously-required populate of `exprOriginMapLocal[rewrittenExpr] = eqs` inside `enumerateEqClassRewrites`'s callback also stays unconditional, so any *first* emission for a target carries its full `setEqualities` justifier list (verifier `check_equality1` rejects `len(rest) < 4`).

**What this gate covers.** It suppresses redundant equality1 emissions inside `applyEquivalenceClass` when the target is already established **inside this LB**. The canonical case where the gate fires usefully: a rewrite produces a target whose origin was already recorded by an earlier non-equality emission in the same hashburst — emitting another equality1 origin would be a parallel record adding nothing.

**What this gate does not cover.** Two known cases:

1. **Swap-cycle across syntactically distinct keys.** When `applyEquivalenceClass` emits in both directions (e.g. source `(in2[2,X,8])` → target `(in2[X,2,8])` and source `(in2[X,2,8])` → target `(in2[2,X,8])` via the simultaneous slot-0/slot-1 swap mapping), each target's exprOriginMap entry is empty when its emission fires, so the gate does not fire. Cycle vector remains.
2. **Mail-imported cycles.** Multiple producer LBs may each emit one half of a cycle; `smashMail` then aggregates them into a receiver's `mailIn.exprOriginMap` uncapped. The gate is producer-local; it cannot suppress what arrives at a different receiver.

Both cases are resolved at the receiver via [D-49](../40_decisions.md#d-49)'s preference replacement at `addOrigin`.

**Site coverage.** Of the three `equality1` emission sites in `prover.hpp` — `applyEquivalenceClass`, `applyEquivalenceClassToNegatedEquality`, `emitIntegrationRevivalToInternalMailIn` — only `applyEquivalenceClass` is gated by [I-34](../30_invariants.md#i-34). The other two have internal early-exits that prevent the same cycle shape (`statementLevelsMap` dedup at site 2; mail absorbance at site 3 routes through site 1's gate downstream). With D-49 in place, the receiver-side preference replacement handles any residual cycle vectors from any of these sites.

**Producer-side sync — not needed at the producer.** The gate consults only `body.exprOriginMap`, which is already kept current by the bulk-merge of `body.mailIn.exprOriginMap`. (D-49 changes how the bulk-merge writes to `body.exprOriginMap`, but D-48's gate predicate is unaffected.)

---

## Origin selection at cap ([D-49](../40_decisions.md#d-49), [I-35](../30_invariants.md#i-35))

When `max_origin_per_expr = 1` (non-compressor mode) and multiple legitimate origins exist for the same key (e.g. mail-bulk-merge brings a foundational `implication` origin AND a cyclic `equality1` origin in the same broadcast for the same target), only one origin can survive per key in `body.exprOriginMap`. Pre-D-49 the survivor was determined by insertion order — silently picking whichever origin happened to be pushed first, which had been the cyclic `equality1` in the chapter-100/101 swap-cycle case.

**Policy.** Foundation displaces convenience. `equality1` and `equality2` are *transitive convenience records* susceptible to swap/bridge cycles; any other origin tag refers to a direct deductive step. When `addOrigin` is called with a non-equality origin and the per-key vector is at cap, scan for the first `equality1`/`equality2` slot and replace it. Below cap, append (legacy behavior).

**Mechanism.** Two coupled halves:

1. **`addOrigin`** ([`prover.hpp::addOrigin`](../../GL_Quick_VS/GL_Quick/src/prover.hpp)) gains a cap-full preference replacement step. Below-cap behavior unchanged (append + dedup + D-44 symmetry-source trap).
2. **Bulk-merge** at [`prover.cpp::performElementaryLogicalStep`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) is rewritten to route through `addOrigin` per arriving (key, origin) pair, instead of the legacy raw `std::map` swap with body-wins-on-conflict. Every arriving mail origin now passes through the same cap+preference gate as direct producer-side emissions.

**Determinism.** Preserved — see [I-35](../30_invariants.md#i-35) and [D-49](../40_decisions.md#d-49) "trade-off" / "verified" sections.

**Composition with other invariants.** [I-32](../30_invariants.md#i-32) (cross-pair `equality2` gate) and [I-34](#cross-substitution-equality1-emission-gating-d-48-i-34--producer-side-redundancy-guard) (cross-substitution `equality1` gate) both compose with I-35: the gates suppress redundant emissions at the producer side; the preference resolves which origin survives at the receiver side when multiple legitimate origins arrive.

---

## Cross-pair `equality2` emission gating ([D-46](../40_decisions.md#d-46), [I-32](../30_invariants.md#i-32))

`mergeTwoEquivalenceClasses` builds a `mergedClass` from two overlapping classes (the in-flight `classA` and the existing `classB`), bridged through a `commonArg`. The body's history block iterates every `(varA ∈ classA \ {commonArg}, varB ∈ classB \ {commonArg})` pair and records an `equality2` cross-pair origin for the new equality `(=[varA, varB])` and its mirror — documenting that the equality is derivable through the bridge.

**Gate.** The push only fires when the target equality is *not* already established. The gate consults three sources:

| Source | Covers |
|---|---|
| `mergedOriginMap` | Origins from prior cross-pair pushes within the same `updateEquivalenceClasses` call (multiple existing classes overlap `eqArgs` and get folded sequentially). |
| `classB.equalityOriginMap` | Origins on the absorbed class — about to fold into `mergedClass` via the post-loop overwrite. Treat as already-known. |
| `memoryBlock.exprOriginMap` | LB-level origins, including mail-arrived ones synced into the class via the bulk-merge + mail-sync at [`prover.cpp::performElementaryLogicalStep`](../../GL_Quick_VS/GL_Quick/src/prover.cpp). |

If any source has a non-empty entry for the target, all three `addOrigin` calls (`mergedOriginMap`, `memoryBlock.exprOriginMap`, `memoryBlock.mailOut.exprOriginMap`) are skipped.

**Why.** When a clique of equalities arrives via mail (e.g. parent-scope contradiction-cascade), each equality's mail origin lands in `body.exprOriginMap` and the class's `equalityOriginMap`. Without the gate, subsequent merges iterate possible bridges and produce `equality2` cross-pair records via DIFFERENT bridges that point at each other:

```text
(=[v1,v2]) ← equality2 | (=[v1,i1]) (=[i1,v2])    -- bridge i1
(=[v1,i1]) ← equality2 | (=[v1,v2]) (=[v2,i1])    -- bridge v2
```

Both records survive to the chapter; the verifier's origin-chain DFS detects the cycle. The gate suppresses the redundant transitive-closure records — they add zero deductive content when the target is already established by a separate path. See [G-38](../50_gotchas.md#g-38) for the historical chapter-22 / theorem-12 case.

---

## Worked example — class expansion

Setup (all in scope `"main"`):

```text
(=[a,b])    → class {a, b}
(=[b,c])    → class grows to {a, b, c}
```

Now `!(=[a,d])` arrives. `applyEquivalenceClassToNegatedEquality` emits:

```text
!(=[b,d])       — one-sided substitution on a's position
!(=[c,d])       — one-sided substitution on a's position
```

Does *not* emit (skipped intentionally):

```text
!(=[b,e])  if d had been in class {d,e}
```

That would require the symmetric cross-product. Two one-sided steps reach it: `!(=[a,e])` first (by substituting `d → e`), then `!(=[b,e])` (by substituting `a → b`).

---

## Weaknesses

### Known & tracked

- **[I-12](../30_invariants.md#i-12) — one-sided-only is deliberate.** Anyone questioning the design should first profile alternatives.
- **Class-visibility confinement to deeper-scopes.** A fact derived via equality in scope `S` is admitted at scope `S`, not promoted to `"main"`. If `S` eventually discharges, the class itself goes with it (scope-scoped).

### Suspected fragility

- **`applyEquivalenceClassToNegatedEquality` precondition assert.** A caller that mis-gates (passes a non-negated-equality) crashes the prover. The guard is deliberate, but adding a new call site without reviewing the guard is a regression risk.
- **Class-merge level computation.** When two classes merge, the new class's per-pair level is the maximum of the two — or is it? Not inspected for this chapter; worth confirming.
- **Per-scope class-map lookup cost.** On each `addStatement`, the per-validity class map is queried. For deeply-nested scope stacks, this is O(depth) per emission. Not currently a hot path but warrants measurement before FTA scale.
- **rmi memory growth under [D-43](../40_decisions.md#d-43) keep-old.** `rmi` now grows monotonically under class application instead of moving in place. Bounded per-call by class fan-out × overlap with `varsInRejectedMapIntegrationKeys`, but cumulative growth at FTA scale (richer fact bases, larger classes) is unmeasured. The cache short-circuit suppresses additions for non-overlapping classes; overlapping classes can still produce K1+K2 pairs at every iteration. Watch peak `rmi.size` during FTA-rung runs; if it exceeds 5× the pre-D-43 baseline, profile.
- **Chained ancestor merges under [D-44](../40_decisions.md#d-44).** The ancestor-pass overlap test in `updateEquivalenceClasses` is against `eqArgs` (the new equality's two args), not against the growing `mergedClass`. If a same-NS class adds var `x` to `mergedClass` and an ancestor class contains `x` but not `a` or `b`, the ancestor class is not absorbed by the merge. The transitivity is still recoverable at runtime via the apply machinery (`mergedClass` substitutes `a ↔ x`; the ancestor class substitutes `x ↔ y` via [D-33](../40_decisions.md#d-33)'s class-shallower direction), so no soundness loss — but the descendant scope's stored class does not explicitly carry `y`. If a downstream rule needs `y` in the same class, the apply chain delivers it; if direct class-membership lookup of `y` is needed, it would miss. Not currently a known failure mode; flagged for future work.

### Not exercised by tests

- **Class-merge origin tracking.** Is the origin of a merged class correctly a combination of the origins of the two input classes? Believed yes, but no targeted test verifies.
- **Class cleanup on scope discharge.** When a scope closes, its class should be discarded. If it persists, stale classes would influence later unrelated scopes. Believed correctly handled; not targeted-tested.

---

## Extension — `rejectedMapIntegration`

As of commit landing [D-19](../40_decisions.md#d-19), equivalence-class application has a second hook: `applyEquivalenceClassToRejectedMapIntegration`. Lives in `prover.hpp` alongside `applyEquivalenceClass`; called once per class at all three sites where the main per-class hook runs — same-NS loop, ancestor-NS loop, AND the fixpoint re-iter lambda (`applyClassesFrom`). The fixpoint site is critical for catching classes created mid-`addStatement` by equality propagation; measured to be ~9% FASTER overall than the no-fixpoint variant (classes applied early clear rmi entries sooner, shrinking later-iteration scan targets).

**What it does.** For each entry in `HashMemory::rejectedMapIntegration` (the integration-side counterpart to `rejectedMap`, see [`10_pipeline/04_prover.md`](../10_pipeline/04_prover.md) Pass B section):

1. Scope-match the entry's validity against the class's validity (same-NS or ancestor-NS).
2. If any of the key's args is in the class, iterate the `allMappingsAna[(|indices|, |eqList|)]` table (same mapping table the main `applyEquivalenceClass` uses) and produce one rewrite per mapping. Per-entry dedup collapses distinct mappings that happen to yield the same rewritten key, so each unique rewrite is probed against admission exactly once per entry per class call.
3. Probe the rewritten marker key against `admissionMapIntegration` (with u_-prefix transform on the key — see [G-32](../50_gotchas.md#g-32)) and `admissionSetIntegration` (bare-form).
4. On match: emit the rewritten concrete constituent + siblings to `internalMailIn` via `emitIntegrationRevivalToInternalMailIn`. Do NOT erase the admission-map entry ([I-22](../30_invariants.md#i-22)). The original `rejectedMapIntegration` entry K1 is **kept** ([D-43](../40_decisions.md#d-43), [I-30](../30_invariants.md#i-30)) — both K1 and K2 remain valid deferred-match registrations: K1 may yet match a future admission entry directly via `revisitRejectedIntegration2`, independently of any class.
5. On no match: insert at the rewritten key (K2) **alongside** the original K1 — both forms coexist in `rmi` ([D-43](../40_decisions.md#d-43), [I-30](../30_invariants.md#i-30)). Insertions are deferred to end-of-function via `toInsert` to avoid mutating `rmi` while iterating.

**Why additive (D-43).** K1 was registered in `rmi` because it failed admission at registration time. A class application produces an alternative rewrite K2 — but K1's deferred-match status is independent of K2. `revisitRejectedIntegration2` iterates `rmi` keys on every admission write; if a future admission entry matches K1 directly, K1 must still be in `rmi` to be revived. Erasing K1 (the pre-D-43 behavior) silently lost those revivals, mirroring the same destructive pattern [D-33](../40_decisions.md#d-33) reverted on the descendant-class direction.

**rmi growth under additive semantics.** `rmi` grows monotonically under class application (modulo set-merge dedup on identical keys: `std::map<ExpressionWithValidity, std::set<RejectedMapIntegrationValue>>` collapses duplicate inserts at the same key). Different keys (K1 + K2 + K3 from cross-class fan-out) accumulate. The fixpoint in `applyClassesFrom` ([`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp), lambda inside `addStatement`) terminates on `encodedStatements.size` plateau, **independent of `rmi` state** — kept K1 entries do not prolong the fixpoint unless they themselves drive new statements through revival.

**Fan-out control.** Two layers of defense keep per-class runtime from dominating:

- `varsInRejectedMapIntegrationKeys` cache on `HashMemory` — monotonically-growing set of non-marker args present in any rmi key. Lets the helper early-exit for classes whose `clss.variables` have zero overlap with this set. Monotonic (never shrinks) — false positives cost a walk, but no correctness risk.
- Per-key `indices` scan — before running the full mapping loop for an rmi entry, we compute which of the key's arg positions are in the class. Empty indices → skip the whole mapping iteration for this entry (no `allMappingsAna` lookup, no string allocations).

Measured cost on the Gauss batch (`{0,1}=[0,1]` rung): full-permutation helper with these defenses runs at ~525s overall pipeline vs ~405s for the canonical-only shortcut — +30% for strict plan compliance, no theorem count difference on this rung. For later FTA-ladder rungs (richer fact bases, larger rmi populations) the full-permutation version is expected to be load-bearing in exactly the cases where canonical-only silently misses a revival.

The `applyEquivalenceClassToRejectedMapIntegration` hook is gated on `!parameters.skip_eq_classes`, same as the main eq-class machinery — incubator batches (which set `skip_eq_classes = true`) bypass both paths.

### Shared inner-loop helper

Both `applyEquivalenceClass` and `applyEquivalenceClassToRejectedMapIntegration` route their per-mapping substitution through the shared template helper `enumerateEqClassRewrites` in `prover.hpp`. The helper owns ONLY the inner loop — given a precomputed `eqList` (typically `reduceEqClass(clss.variables, mb, validityName)` materialised as a vector), it iterates `allMappingsAna[(|indices|, |eqList|)]` and emits each rewrite via a callback as an `EqClassRewrite` struct (`rewrittenExpr`, `setEqualities`, `extraLevels`, `substMap`, `isIdentity`). All policy stays at the call site:

- **Wrappers** — caller chooses `wrapLeft` / `wrapRight` and the `baseExpr` (e.g. expressions handles both `(...)` and `!(...)`; rmi only `(...)` markers).
- **Scope-direction admission** — call site decides which directions are sound. `applyEquivalenceClass` admits all three (same / class-shallower / class-deeper, [D-33](../40_decisions.md#d-33)). `applyEquivalenceClassToRejectedMapIntegration` admits only same-NS and class-shallower; the class-deeper direction is intentionally NOT admitted (rmi entries are deferred-match registrations whose preconditions live at the entry's scope, so a deeper class carries no information for them — see "What the rejected-integration map does NOT do" above).
- **Identity-skip and per-entry dedup** — rmi-only optimisations (avoid no-op admission probes; collapse mappings that yield the same rewritten key). Live in the rmi sink lambda; expressions caller does not skip identity (downstream `statementLevelsMap` dedup handles it).
- **Outer guards** — `varsInRejectedMapIntegrationKeys` overlap short-circuit and `rmi.empty` early exit live at the rmi caller, before the per-entry loop. The helper is called per rmi entry, so the rmi caller passes the SAME `eqList` (computed once per `applyEquivalenceClassToRejectedMapIntegration` call) to every helper invocation — recomputing per entry would be a 10⁵× regression at Gauss scale.
- **Downstream emission** — call site decides whether to emit to `encodedStatements` / origin map (`equality1`) or to admission-probe → `internalMailIn` revival vs. move-in-place. The helper emits nothing.

The helper exists purely to remove duplication of the mapping/substitution arithmetic. Any change to the substitution algorithm itself (mapping enumeration, equality-string format `"(=[from,to])"`, levels-set merging) lives in one place; any change to policy (scope admission, dedup, outer guards) lives at the call site.

---

## Extension — `admissionMap` (algebra-side equi-class hook)

As of [D-57](../40_decisions.md#d-57), equivalence-class application has a third hook on the algebra side: `applyEquivalenceClassToAdmissionMap` in `prover.hpp`. Called once per class at all four sites where the rmi hook runs — same-NS loop, ancestor-NS loop, descendant-NS loop, and the fixpoint re-iter lambda (`applyClassesFrom`) — immediately after each `applyEquivalenceClassToRejectedMapIntegration` call.

**What it does.** For each entry in `HashMemory::admissionMap`:

1. **Short-circuit on `HashMemory::varsInAdmissionMapKeys`** — symmetric to `varsInRejectedMapIntegrationKeys`. Monotonically-growing set of non-marker args appearing in any admission key. Populated at every admission insert (`prover.hpp::prepareIntegration`, `memory.cpp::makeMandatoryEncodedStatementLists1Static`, `prover.cpp::updateAdmissionMapRecursion`, and the hook itself's post-loop insert). Lets the hook early-exit for classes whose `clss.variables` have zero overlap with this set.

2. **Scope-match** the entry's validity against the class's validity. Three directions admitted, mirroring [D-33](../40_decisions.md#d-33): same-NS, class-shallower (ancestor of entry), class-deeper (descendant of entry). Deposit scope is `deeperOf(classScope, entryScope)`.

3. **Enumerate rewrites** of the marker key via the shared `enumerateEqClassRewrites` helper. The sink lambda filters:
 - Identity rewrites (no substitution happened).
 - Rewrites whose rewritten string equals the original marker key.
 - **Arg-equalization filter** ([I-36](../30_invariants.md#i-36)): rewrites that collapse previously-distinct arg slots into the same value are dropped. For every pair `(i, j)` with `origArgs[i]!= origArgs[j]`, the rewrite must satisfy `newArgs[i]!= newArgs[j]`. This filter is **algebra-only**; the integration mirror does not have it.

4. **Substitute the AdmissionMapValue contents**: each `AdmissionMapValue.key` element and each member of `remainingArgs` gets `ce::replaceKeysInString` with the rewrite's substMap. `standardMaxAdmissionDepth`, `standardMaxSecondaryNumber`, and `flag` are copied unchanged. `u_`-prefixed args inside the value are naturally excluded from substitution (substMap keys are bare variable names; u_-prefixed names are never class members).

5. **Additive insert** into `admissionMap[K']` (the rewritten key at deposit validity). Old key K is preserved — keys are ADDED, not replaced. Insertions are queued and applied post-loop to avoid mid-iteration mutation.

6. **`admissionStatusMap[K']` inherits from K** only when K' doesn't already have a status entry — preserve existing entries (additive principle).

7. **Fire `revisitRejected2(K', mb, depositValidity)`** for each new K'. revisitRejected2 walks the unchanged `rejectedMap[K']` and mail-emits any matching rejection cohort via `emitIntegrationRevivalToInternalMailIn`. K' is bare-marker form already (admissionMap keys carry no u_ prefix) — directly usable as a rejectedMap key, no u_-strip needed.

**Why never touch `rejectedMap`.** Per [I-37](../30_invariants.md#i-37): `rejectedMap` holds real disintegration products whose `disintegration` origins were recorded at production site by `prover.cpp::disintegrateExprCore2::trackExpansionHistory`. Substituted rejection records would have no corresponding origin in `exprOriginMap` — a provenance gap. The integration mirror's D-43 additive-on-no-match behavior has exactly this gap (acknowledged-bug, scoped out of this branch). Algebra rewrites the admission map and lets `revisitRejected2` find matches against the unchanged `rejectedMap`.

**Iteration safety.** Inserts, status updates, cache populates, and `revisitRejected2` calls are queued in `toInsert` and applied after the outer admissionMap walk. `revisitRejected2` internally calls `cleanAdmissionMap` (see [`10_pipeline/04_prover.md`](../10_pipeline/04_prover.md) — admission cleanup section) which may erase the just-inserted K' if the marker sits in the operator's output slot. That erasure is permissible — K's preservation (the additive principle) is what matters; K' has carried the revival and consumed itself.

The `applyEquivalenceClassToAdmissionMap` hook is gated on `!parameters.skip_eq_classes` at the same outer site as the other equi-class machinery — incubator batches (which set `skip_eq_classes = true`) bypass it.

### Algebra `revisitRejected2` mail-first emission

As of the same branch, `prover.cpp::revisitRejected2`'s body emits the stored cohort (`concreteConstituent` + `siblings` + `levels` per the widened `RejectedMapValue` schema) directly via `emitIntegrationRevivalToInternalMailIn`, without re-expanding the compact form through `prepareIntegrationCore` and without writing per-child `disintegration` origins inline. The proper `disintegration` origin for each child was already written at the original production site by `disintegrateExprCore2::trackExpansionHistory`; the mail's degenerate `equality1` self-source origin cannot displace it because `addOrigin`'s cap-full preference clause (`prover.hpp::addOrigin` — [D-49](../40_decisions.md#d-49) / [I-35](../30_invariants.md#i-35)) only allows non-equality to replace equality, not the reverse — existing slot wins. Chapter export reads `disintegration`; verifier accepts.

The pre-refactor body re-ran `prepareIntegrationCore` on the compact form, did `find_if` for the matching `LogicalEntity`, extracted the bound variable, swapped it for the rejected variable, expanded the modified entity, and emitted per-child `addExprToMemoryBlock(status=0)` with explicit `expansion` + `disintegration` origins. All of that machinery existed solely because `RejectedMapValue` stashed the compact form without the per-element children. Widening the struct ([C1 commit on the branch](#)) put the cohort in place at rejection time; the body refactor ([C2 commit](#)) collapsed to a single helper call. Asserts preserved: `markerIndex!= -1`, `argsIdentical`. The `ent.category == "existence"` assert went with the `prepareIntegrationCore` path. The `statementLevelsMap.find!= end` assert was REMOVED on this branch — the compact form's level entry may legitimately be missing when a Pass-B rejection commits before the kernel-loop's compound-stmt addStatement runs (the commit happens inside `disintegrateExpr2`; the level write happens later in the kernel's per-stmt `addStatement` loop, and the equi-class hook from C3 can fire from an earlier-stmt's `addStatement` before the compound's own write). `val.levels` (captured at buffer time, may be empty) is the authoritative deposit-time levels set.

## See also

- [`10_pipeline/04_prover.md`](../10_pipeline/04_prover.md) — `addStatement`, Pass B, and equivalence handling.
- [`20_core_concepts/04_validity_stack.md`](04_validity_stack.md) — scope-based visibility rules.
- [`20_core_concepts/03_mail_system.md`](03_mail_system.md) — `internalMailIn` is the emission target of revival matches.
- [I-9](../30_invariants.md#i-9), [I-12](../30_invariants.md#i-12), [I-21](../30_invariants.md#i-21), [I-22](../30_invariants.md#i-22), [I-37](../30_invariants.md#i-37), [I-36](../30_invariants.md#i-36).
- [D-19](../40_decisions.md#d-19), [D-57](../40_decisions.md#d-57), [G-31](../50_gotchas.md#g-31), [G-32](../50_gotchas.md#g-32).
-.
- Verifier tags: `equality1`, `equality2`, `symmetry of equality`, `symmetry of inequality` — see [`20_core_concepts/08_proof_tags.md`](08_proof_tags.md).

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
