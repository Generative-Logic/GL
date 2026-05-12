<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Induction-typing gap — architecture plan

## The gap

The prover schedules induction on a bound variable `n` whose type is inferred
implicitly from its occurrence at a typed position (def-set `(1)` in
`AnchorPeano`, `AnchorGauss`, etc.). It never checks that `n` is actually in
`N`. For a variable used only inside negations, existence heads, or bare
equalities, the implicit-typing inference is unsound — the theorem ranges
over everything, not just over `N`.

Canonical Peano failure case (currently in `proved_theorems.txt`):

```
(>[1,2,3](AnchorPeano[1,2,3,4,5,6])
  (>[7]
    !(=[2,7])
    !(>[8](in[8,1])!(in2[8,7,3]))))
```

This reads as `∀n: ¬(i0=n) → ∃m∈N: s(m)=n`. For `n ∉ N`, head is false
while premise is true — theorem is false in general. The prover proves it
by induction on `7` (= `n`) and the verifier passes because both prover and
verifier treat the typed-position occurrence as implicit typing.

## Fix — induction typing sub-theorem

Before scheduling induction on `n`, the prover must prove `(in[n,N])`
from the current chain. `N` = `anchor_args[0]` (slot 0 of the anchor —
in Peano that's `N`, in Gauss that's whatever the first Gauss anchor slot
names, in Incubator the same). The typing sub-theorem inherits the
original chain's non-anchor premises and anchor; only the head changes.

Induction succeeds **iff all three discharges complete**:
1. typing: `(>[bounds](Anchor)(chain_premises)(in[n,N]))` — direct-proof only
2. base: `(>[bounds](Anchor)(chain_premises[n:= i0])(head[n:= i0]))`
3. step: `(>[bounds](Anchor)(chain_premises[n:= s(m)], head[m:= n])(head[n:= s(m)]))`

The existing axiom framework already provides the typing lemma:

- `fXY[f,X,Y]` contains `(>[x,y](in2[x,y,f])(& (in[x,X])(in[y,Y])))`
- `fXYZ[f,X,Y,Z]` contains the analogous triple-variable version.

So for the currently-sound induction theorems such as
`(>[i0,s,+,i1](Anchor)(>[v1](in2[v1,i0,s])(in3[i1,v1,i0,+])))`, the typing
sub-theorem `(in[v1,N])` is derivable via `fXY` applied to
`(in2[v1,i0,s])`. For the unsound theorem above, no positive premise
mentions `v1`, so `(in[v1,N])` is unprovable → induction rejected.

## Expected impact — Peano (current branch, 501-conjecture run)

Current global theorem list has 55 entries:
- `direct`: 9 (unaffected)
- `induction`: 23 (audit below)
- `mirrored statement`: 23 (mirrors of induction / direct — cascade follows)

Hand-audit of the 23 induction theorems:

| # | theorem | induction var | typing derivable? |
|---|---|---|---|
| 1 | `(>[N,i0,s](Anchor)(>[v1]!(=[i0,v1])(existence2[N,v1,s])))` | v1 | ✗ no positive `v1`-premise |
| 2–22 | all carry at least one positive `(in2[v1,…])` or `(in3[…,v1,…])` | v1 or v2 | ✓ via fXY/fXYZ |
| 23 | (same pattern as 2–22) | v1 | ✓ |

Expected loss: **theorem 1 + its mirrored statement ≈ 2 theorems**. Verified
OR-theorem form (`(in[n,N])` as explicit premise) remains sound and
untouched — it uses `method = direct` after OR-convergence, not
`method = induction`.

## Implementation plan

### Stage 1 — prover-side gate + 3rd recursion block

**Files touched**: `parameters.hpp`, `memory.hpp` (DependencyTable or similar), `prover.cpp`, `prover.hpp`.

1. `ProverParameters` — add `bool typingProofOnly = false;`.
2. At induction setup (`prover.cpp`, inside the block that creates
 `recursion` sub-blocks):
 - Build typing sub-goal head: `(in[digitArg,anchor_args[0]])`.
 - Create a third `Memory* tempMb3` with that head as its `exprKey`
 and register toBeProved / recursion marker as the existing two
 sub-blocks do.
 - Register `tempMb3` in `permanentBodies` + wire `auxyOriginalMap` /
 `originalAuxyMap` so it counts toward `origItem.auxies` at the
 dependency-table level.
 - While processing `tempMb3`, each proof step must set
 `parameters.typingProofOnly = true` in its local execution context
 to forbid induction re-entry on the same (or any other) variable.
3. At promotion (`prover.cpp`):
 - Before `emplace_back(expr, "induction", indVar, recCounter)`,
 verify the typing sub-block completed successfully. If not,
 the existing `promote=true` path is skipped (do not insert into
 globalTheoremList) — the induction is silently rejected.
4. `deactivateUnnecessary` / `deactivateRecursively` — extend to handle
 the third block the same way the other two are handled.

### Stage 2 — chapter emission

**Files touched**: `visualizer.cpp`, `generate_full_proof_graph.py`,
`process_proof_graphs.py`, `run_modes.py`, `run_modes.cpp`.

- Chapter filename convention:
 `N_induction_typing.txt` paired with the existing
 `N+1_check_zero.txt` and `N+2_check_induction_condition.txt`.
 (Or pick another three-chapter numbering scheme — the constraint is
 deterministic ordering so the verifier can match typing-chapter to
 parent theorem.)
- `visualizer.cpp` — when building the raw proof graph for the induction
 target, walk the third recursion block too and emit it as a chapter
 with all the usual tags.
- `process_proof_graphs.py` — recognize the new filename, apply the
 4-priority variable renaming scheme (anchor args first, then theorem
 vars left-to-right, etc.), write to `processed_proof_graph/`.
- Global theorem list entry for the induction theorem continues to use
 `method = induction` + induction variable. Verifier will find the
 typing chapter by the numbering convention.

### Stage 3 — verifier

**Files touched**: `verifier.py`.

- Add `"induction typing"` to `TAG_CHECKERS` with a new handler.
- For every row in `global_theorem_list.txt` with `method == "induction"`:
 - Locate the typing chapter `N_induction_typing.txt` (numbering tied
 to the parent theorem's chapter index).
 - Check the chapter file exists.
 - Parse the chapter's implicit theorem (first tag `task formulation`
 row = the theorem being proved). Verify head = `(in[ind_var,N_arg])`
 and chain matches parent's non-anchor chain (after
 `_normalize_expr_list`).
 - Verify every tag inside the typing chapter via the existing tag
 checkers (it's an ordinary direct-proof chapter — all checks
 already-implemented apply unchanged).
- Failure of any step above → fail the `induction typing` row for that
 theorem. Other tag-row counts are unaffected.

## Anti-patterns to guard

- **Infinite typing recursion.** The typing sub-goal is proved with
 `typingProofOnly=true` so induction cannot re-enter. If the typing
 sub-goal itself required induction on the same or another variable,
 we fail fast rather than recurse.
- **Typing sub-goal succeeds via derived theorem.** If `(in[v1,N])` is
 derivable via a previously-proved theorem (e.g. a prior batch provided
 `(>[n](in2[n,_,_])(in[n,N]))`), the typing sub-goal is still legitimate
 direct-proof territory. No special case.
- **Multi-batch proved_theorems.txt.** Theorems from an earlier batch
 already in `proved_theorems.txt` that relied on the old unsound
 induction will NOT be re-proved — they're loaded as external theorems
 on subsequent batches. A backward-compat pass (not in this branch)
 would be needed to re-audit those in a future sweep.

## Rule 8 checkpoint

- Approved by user (explicit in conversation):
 - Typing sub-theorem proved by same prover with `typingProofOnly=true`.
 - Target is semantic soundness over preserving theorem count.
 - Full artifact: chapter + verifier check.
- Architectural breaks potentially affected:
 - Provenance recording — the new chapter enters the proof graph.
 - Scope/namespace handling — the typing sub-block is a new child of
 the induction-target memory block under the same `validityName`
 convention (`NameMap::encodePush` path); no new scope kind.
 - Proof-graph contract — **extended**, not broken: existing chapters
 keep their shape; one new per-induction chapter added.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
