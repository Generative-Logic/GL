<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# B10 runtime investigation (2026-08-12)

Diagnosis-only session (no fixes). Trigger: adding B10 (shortlist row 44, strict
reflection `a·c < b·c ⟹ a < b`) raised the shortcut prover wall by roughly
sixty seconds over the B9 baseline band. The maintainer flagged a sudden
runtime jump around hash burst 25 and asked for: the offending logic block(s),
a phase attribution (1/2/3), a hashburst-dump content analysis, the split
behaviour, and — added mid-session — the reason the grid still holds active
logic blocks after the last conjecture proves.

## 1. The jump, from the run log

`main.py --shortcut` on the 44-row pool (, natural
mode, prover 199.7 s; reproduced in , 187.6 s):

| burst | wall | swept | note |
|---|---|---|---|
| 22 | 1.3 s | 6 | normal endgame band |
| 23 | 3.3 s | 10 | |
| 24 | 5.1 s | 10 | |
| **25** | **49.3 s** | **5** | the jump |
| **26** | **16.6 s** | **3** | echo |
| 27 | 0.2 s | 1 | B10's theorem emits |

Bursts 25 + 26 ≈ 66 s ≈ the whole B10 surcharge; every earlier burst is in the
established band. The cost is the endgame of B10's own proof search.

## 2. The bad logic blocks (full chains, Rule 12)

RT-tracker run (`RT_MEASUREMENT=1`, trigger 5 s, `disable_lb_split`,
, `.rt/` tables): eleven logic blocks crossed the
5-second trigger over the whole run; hashburst 25 is owned by **all three
induction step-blocks of row 44**, children of the chain
`(root) → (AnchorFTA[1..8]) → (in3[9,10,11,5]) → (in3[12,10,13,5]) → (strictOrder[1,4,11,13])`:

| leaf | induction variable | serial call wall |
|---|---|---|
| `(in2[rec2,9,3])` | 9 = a | 66.2 s |
| `(in2[rec0,10,3])` | 10 = c | 62.3 s |
| `(in2[rec1,12,3])` | 12 = b | 57.9 s |

The three run concurrently, so the natural-mode burst wall is the slowest of
the three after intra-block splitting.

## 3. Phase attribution: phase 2, the request-generation grow search

- RT tables: **99.7–99.8 % of each call inside
 `GENERATE_ENCODED_REQUESTS_STATIC`** — request generation, not firing
 (`FIXPOINT_LOOP` does not register).
- Burst-level wall split (temporary `[BURST-PH]` print in `proveKernel`),
 natural mode, burst 25: `ph1=0.19 s ph2=43.4 s ph3=6.1 s post=0.04 s`
 (burst 26: `ph2=14.5 s ph3=2.1 s`).

Phase 2 owns ~87 % of the monster burst; phase 3 (post-burst
standardProcessing / sanitize over the enlarged deposit set) is a visible but
secondary ~12 %. Phases 1 and post are negligible. Per the session directive
the analysis therefore went to hashburst content (section 5).

## 4. Split behaviour

The trigger works, the split does not pay:

- Natural-mode `[SPLIT]` timeline for `(in2[rec0,10,3])`: promoted straggler
 at burst 21's end (work 59,897, ran that burst unsplit), back to unsplit by
 23 (work fell below the bar), re-promoted at 23's end, then split 32-way at
 24 (work 211,944, 647 stumps), **25 (work 1,146,732, 1230 stumps)** and 26
 (877,351, 994 stumps).
- **The LB is split during both heavy bursts** — this is NOT the historical
 stumps=0 serial fallback (stumps are produced and bucketed).
- But the busiest bucket carries 105,808 submatches ≈ **3× the ideal
 intra-block share** (1,146,732 / 32 ≈ 35,835); at the established
 ~0.5 ms/submatch rate that one bucket ≈ the 49.3 s wall. Round-robin stump
 dealing (I-158) balances stump COUNT, not stump WORK.
- Net split gain is small: one serial core does the same burst in 62 s; 32
 cores deliver 49.3 s — a 1.26× speedup from a 32-way split.

**Early-exit involvement (maintainer question).** There is no submatch cap —
`BurstSink::canAccept` runs every burst to completion; the only early exit is
the doom-stop (fired head = contradiction / vacuous truth / goal reached),
honoured single-part only (I-76).

- **Burst 25: early exit is NOT the story.** The serial RT control ran the
 same burst single-part with the doom-stop armed and still took 62–66 s to
 completion — no deactivating head fired; the work is genuine.
- **Burst 26: yes — mostly wasted-once-doomed work.** The block's one goal
 `(strictOrder[1,4,9,12])@main` fires during burst 26 (dump EXIT #29 shows
 `toBeProved 1 → 0`). Multi-part semantics suppress the stop, so all 32
 buckets ground through 877 k submatches; the serial control's matching burst
 cost 0.77 s. Roughly the whole 16.6 s natural-mode burst 26 is post-goal
 grinding that a single-part run would have skipped.

## 5. Hashburst content analysis (dump retargeted at `(in2[rec0,10,3])`)

Trace:  (435 MB, 29 ENTRY/EXIT pairs; dump
#28 = grid burst 25, #29 = grid burst 26). ENTRY #28 state: 4,871 statements,
3,086 rules, 2,687 hash-map keys, nameMap 14,604 ids.

Anomaly inventory at ENTRY #28:

- **or7 totality flood — the headline.** 3,124 statements (64 %) are
 or-compacts; **1,656 are `or7` (totality `a≤b ∨ b≤a`) instances**, and 1,699
 or-compacts cite TWO minted witnesses — the quadratic witness-pair space.
 The monster burst mints 557 more (`or7` 1,656 → 2,213 at EXIT #28). This is
 the same quadratic pair pathology recorded for B5/B7, now denser (64 %
 or-compacts vs 27 % then).
- **Witness saturation:** 3,822 statements (78 %) cite minted `it_`
 witnesses; 1,538 are existence compacts.
- **strictOrder machinery amplification:** 1,320 of 3,086 rules (43 %) cite
 negated equalities — the `strictOrder = preorder ∧ ¬=` expansion doubles
 the rule surface relative to the preorder-only B9, which is why B10 pays
 where B9 did not. 1,157 rules are or-machinery (mutual-exclusion /
 subset-exclusion families).
- **Registry step-jumps:** rules 552 → 1,712 at dump #20 (×3.1 in one burst)
 and 3,086 → 5,764 at #28 (×1.87); statements 4,871 → 6,428 at #28. The
 grow search is super-linear in this population (the known knee).
- **Branch-scope duplication:** ~40 % of statements live in `_ordis_`
 predecessor-split branch scopes (registry-level ancestor-dedup holds — these
 are branch-local variants, not main∧branch copies).
- **Clean:** zero x-citing statements or rules (x-leak fix holding), no
 admission-map anomalies (47 admission / ~2,200 parked rejected rows in the
 usual proportions), mailIn empty at ENTRY.

Mechanistic summary: the strictOrder head forces the negated-equality +
trichotomy machinery; every induction step-block accumulates a witness
population whose PAIRS instantiate totality/trichotomy or-compacts; or-compact
statements feed or-machinery rules that mint more witnesses; the grow search
walks subkey combinations over this quadratically-fed population — 1.15 M
submatches at burst 25 with per-burst work quintupling (35 k → 212 k → 1.15 M).

## 6. Brainstorm — candidate levers (maintainer's pick; NONE implemented)

1. **Witness-pair subsumption / cap** (open proposal #1 from the B-RT
 campaign; the or7 flood is its concrete target): refuse or demand-gate an
 or-compact instantiation over two minted witnesses when neither cites a
 goal-relevant term — e.g. admit `or7[it_i, it_j]` only when the pair
 co-occurs in a non-or statement.
2. **Grow-search argument-level discrimination** (open proposal #3): the
 1.15 M submatches explore combinations the argument structure already
 rules out.
3. **Work-weighted stump dealing:** I-158 deals stumps round-robin by count;
 dealing by estimated stump work (e.g. subkey fan-out) would attack the 3×
 busiest-bucket imbalance directly — the cheapest split-side win.
4. **Deterministic staged doom-stop for split blocks:** burst 26's ~16 s is
 post-goal grinding. A determinism-safe variant — e.g. checking the stop
 flag only at fixed bucket boundaries in a fixed order, so every run skips
 the identical suffix — would recover most of it without the sibling-race
 that I-76 forbids.
 **STATUS: SHIPPED** (2026-08-13,,
 D-282) in a sharper, user-designed form than the
 bucket-boundary sketch above: a doom trigger publishes its part's own
 submatch counter + invocation ordinal onto a per-block CAS-min **doom
 line**; every part stops strictly past the line's position in its OWN
 stream; the finalize merges the WINNING part's chain alone. Measured on
 this branch's 46-row pool: shortcut prover wall 228.89 → 211.56 s
 (−17.3 s); the baseline's 18.7 s `swept=3` doomed burst vanishes from the
 burst timeline while every other heavy burst reproduces within noise;
 theorems 45/46 by design on both sides and every tracked artifact
 byte-identical. Levers 1–3 and 5 remain open; lever 3's imbalance is the
 declared target of the next campaign (online split with state
 preservation — see the project memory).
5. **Unfair-advantage route** (standing preference): a pool lemma that lets
 the strict-reflection step close without the trichotomy grind — though B10
 already proved; the lever matters for Part C rows that will re-enter this
 machinery over `strictOrder[1,5,…]` (proper divisorship).

## 7. Post-batch active blocks (maintainer contract: grid fully inactive)

Both 44-row natural runs end with **4 logic blocks active** through the final
~16 idle bursts (quiescent-skipped, ~2 ms/burst — cheap, but a contract
violation). The census trap names them — one chain, NOT contradiction twins
(`primed=0` everywhere):

```
(root)  ←  held by child
(AnchorFTA[1,2,3,4,5,6,7,8])  ←  held by child
(in[9,1])  ←  held by child
(=[9,10])   goals_main=1   ← the actual holdout
```

`(=[9,10])` under `(in[9,1])` is the shared premise block of shortlist rows 4
(`a=b ⟹ a≤b`, head `preorder[1,4,9,10]`) and 29 (`a=b ⟹ ¬(a<b)`, head
`!(strictOrder[1,4,9,10])`). At warm-up end it holds TWO main-scope goals; at
batch end ONE remains — yet BOTH theorems are proved (`global_theorem_list`
method `direct` for both). The goal-text decode
() names the survivor:

```
[ACTIVE-CENSUS] goals_main=1 chain: (=[9,10]) <- (in[9,1]) <- (AnchorFTA[...]) <- (root)
[ACTIVE-CENSUS]   goal@main: !(strictOrder[1,4,9,10])
```

Row 4's positive head discharged its `toBeProved` row; row 29's never
closed. **Root cause — proven end-to-end with the dump retargeted at
`(=[9,10])` () plus a census
`headKnownHere` probe:**

1. Row 29's head `!(strictOrder[1,4,9,10])` becomes a KNOWN statement at
 `(=[9,10])@main` in the LB's **first burst** — deposited by the
 negated-compound contradiction-scope machinery (scope
 `contradiction_!(strictOrder[1,4,9,10])`: assume `strictOrder`, expand to
 `≤ ∧ ≠`, collide `!(=[9,10])` with the premise `=[9,10]`). The deposit's
 level row is the union of the two collision antecedents — both level 2 —
 so the head carries **`levels={2}`**. The proof genuinely never consumes
 the level-1 premise `(in[9,1])` (assume `a<b` → `a≤b ∧ a≠b` → collision
 with `a=b`; no membership needed).
2. `dischargeToBeProved`'s non-recursion gate
 (`isProved(...) && allLevelsInvolved`, `prover.hpp`) requires the head's
 level run to be `{0..level}` or `{1..level}`; at `level=2` the run `{2}`
 can NEVER pass. Contrast: row 4's head arrived with `{0,1,2}` and
 discharged at the LB's third sweep (`toBeProved 2 → 1`). Row 29's goal is
 therefore **permanently undischargeable at its own block** although the
 head is known at the exact goal key.
3. The theorem still emits through the primed contradiction twin
 (`dischargeContradiction`, chapter `39_direct_proof.txt` tag
 `contradiction`) — the twin route does no level accounting.
4. The emission drain (`updateGlobalDirect` → `deactivateUnnecessary`)
 retires the twins and recursion auxiliaries but deliberately refuses
 bubble-up deactivation while a main-scope goal remains — and the
 parent-side goal-erase that DOES exist on this drain misses by design:
 [D-240](../agentic_swdd/40_decisions.md#d-240)'s `pendingDisprovedGoals`
 → `drainDisprovedGoals` erases a parent MAIN goal matching the twin's
 SEED — built for the DISPROOF direction (seed == conjecture head). For a
 negated-head PROOF twin the seed is `negate(goal)`, the probe misses —
 D-240's own "complement-twin seed … a defined no-op" — and deliberately
 so, since the disproof cleanup wipes the goal's machinery as dead, which
 would be wrong for a goal that succeeded. The design defers a proved
 goal's closure to the success path (broadcast re-fire →
 `dischargeToBeProved`) — the very path the level gate blocks here. Every
 subsequent I-48 `deactivateRecursively` sweep refuses for the same
 reason. The chain stays active to batch end: exactly the 4 undead
 blocks. The uncovered quadrant: twin PROVES × head LEVEL-POOR — the
 other three quadrants (disproof × anything, proof × level-complete) all
 have closure paths.

**Classification: architecture gap**, not a miscoded contract — two
individually-correct disciplines compose badly: the level-gated local
discharge (soundness: an emitted implication must have consumed every
premise) meets a contradiction-scope proof whose level row is legitimately
narrow, while the level-free twin route emits the theorem globally and no
path retires the local goal on settlement. The recursion path already has a
maintainer-approved precedent for exactly this shape — the digit-arg
membership exemption ("only missing level is the induction digit-arg's own
membership premise") lives directly below the gate.

Candidate repairs (maintainer's pick, NONE implemented):
- **(a) Emission-time goal retirement** — close the uncovered quadrant at
 the existing D-240 drain: alongside the seed probe, probe `negate(seed)`
 as a PROVED main goal and retire just the goal row with success semantics
 (no machinery wipe, history kept). Surgical: same single-threaded seam,
 same deposit, one extra probe; the disproof branch stays untouched.
- **(b) Settled-theorem exemption at the gate** — a goal whose head is known
 at the exact goal key AND whose theorem has already emitted counts as
 closed regardless of level coverage.
- **(c) Stamp the contradiction-scope deposit with the full premise-level
 run** — rejected-by-default flavour: it widens level semantics, since the
 proof genuinely did not consume level 1.

The mid-flight censuses incidentally confirm that primed contradiction twins
DO retire correctly during the run (many active mid-run, none at batch end),
so the earlier twin-immortality hypothesis is falsified.

## 7b. Maintainer resolution — three problems, fixed in this order (2026-08-12)

The undead-blocks finding decomposes into THREE independent problems
(maintainer-identified); they must land in this order, each gated by a full
run, because fixing them out of order destroys the reproduction case for the
ones behind it.

**Problem 1 — a contradiction-twin PROOF never erases the parent's goal.**
The fix (maintainer-approved direction): at the existing D-240 deposit/drain
seam, probe `negate(seed)` against the parent's MAIN goals alongside the
existing seed probe; on a hit, erase the goal row and stage the goal's scope
wipes with the ORDINARY success-path semantics (`wipeSubtree`: statements /
gates / cohorts go, `exprOriginMap` stays — Rule 16 documentation, read by
nothing at runtime). No origin-erase, no exception lists — those exist only
in the disproof branch because a dead goal's history is scrubbing-worthy
garbage; a proved goal's history stays like any proved statement's. No
walker changes.

**Problem 2 — the levels gate guards REGISTRATION, not the goal lifecycle.**
`allLevelsInvolved` exists for artifact fidelity of the EMITTED theorem (an
emitted implication must match its consumed premise levels). It must not
keep a goal row alive: a naturally-produced conjecture with one premise too
many would otherwise hold its LB chain active forever. Fix: decouple —
when the head becomes known at the exact goal key at main, the goal is
CLOSED and erased (LB may die) regardless of level coverage; `updateGlobal`
registration stays level-gated, so a level-poor closure emits NOTHING.
Accepted consequences (maintainer-stated): (i) a later level-complete
re-derivation no longer triggers registration through the erased goal row —
the narrower theorem is the conjecturer's to restate; (ii) an over-premised
row without a twin proves-but-does-not-register (in the current pool row 29
still registers only because the level-blind twin emits it; with twins off
the pool would go 43/44 BY DESIGN until the row is corrected).

**Problem 3 — the conjecture is over-premised (fix LAST).** Row 29's
`(in[9,1])` is mathematically unnecessary: `strictOrder` is
membership-implying (false outside N), so the negated head holds vacuously
for non-naturals and the typing premise adds nothing — unlike row 4, whose
POSITIVE head `preorder` is false outside N and genuinely needs the typing.
Authoring-convention refinement: a negated head over a membership-implying
relation does not need the `=`-exception typing premise. Corrected row:
`(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](=[9,10])!(strictOrder[1,4,9,10])))`
— one premise level, so the reductio's level run `{1}` satisfies the gate
and the row closes + registers through the normal path.

**Fix order (forced, not preferred) — problem numbering == fix order:**
1. **Contra-proof erase** → full run. Acceptance: census
 `total_active_at_batch_end=0`, 44/44, verifier airtight, theorems
 byte-identical. (Fix 2 first would close the goal locally before the
 twin drain runs — fix 1 untestable.)
 **STATUS: SHIPPED**
 (D-279): `drainDisprovedGoals` now probes
 `negate(seed)` alongside the seed and closes a proved goal with success
 semantics.
2. **Levels-gate decoupling** → full run. Acceptance: no regression
 (44/44, airtight, theorem set byte-identical — fix 1's warm-up-burst-0
 drain still wins the race on this pool); plus a direct unit test for
 close-without-register.
 **STATUS: SHIPPED** together with fix 3
 (maintainer directive 2026-08-13: fixes 2+3 land as one milestone), with
 a maintainer-directed scope extension beyond the original phrasing: the
 registration gate applies to EVERY route — `dischargeContradiction` now
 seals its own verdict (colliding-pair level-row union excluding the
 twin's own level), so an over-premised row no longer registers through
 the level-blind twin either. Closure, wipes, twin retirement, and the
 D-240 deposit stay level-free
 (D-278,
 I-192; units
 `prover.level_poor_goal_closes_without_registering` /
 `prover.level_complete_goal_seals_registration_verdict` /
 `prover.contradiction_twin_seal_carries_level_verdict`).
 **Aftermath (2026-08-13, ):** the first
 acceptance run lost B5/B7/B10 — the strict verdict exposed a
 PRE-EXISTING equality1 level-propagation defect (class rewrites
 depositing with the source statement's run only, dropping the justifying
 pair's levels; B5's refusal then starved B7/B10 of the or0-opening
 demand its 4-premise rule provides). Fixed by
 D-280 (four parked-map hooks + the I-12
 negated-equality expansion); final ladder: 46-pool 45/46 with row 29 the
 designed refusal, census 0, full pipeline 138704/0 with `1·a=a`
 RECOVERED (a silent G-46-class casualty of the old accounting),
 determinism byte-identical across two full runs.
3. **Conjecture correction** → full run. Original phrasing (drop `in[9,1]`
 from row 29) SUPERSEDED by maintainer directive 2026-08-13: row 29 is
 KEPT as the standing close-without-register reproduction case for fix 2,
 the corrected row is APPENDED as row 46, and B11's row 45 is restored —
 the 46-row pool registers 45 BY DESIGN (row 29 closes without
 registering on both routes). Emission dedup for the corrected row (twin
 route and normal route can both settle the same head) rides
 `appendGlobalTheorem`'s string dedup.
 **STATUS: SHIPPED** (same milestone as fix 2).

## 8. Evidence files

-  — original B10 campaign run (burst timeline).
-  + `.rt/*.log` — serial RT attribution run.
-  — natural-mode evidence run ([BURST-PH],
 [SPLIT], census).
-  — census run with goal-text decode.
-  — 29-burst trace of
 `(in2[rec0,10,3])` (dump #28/#29 = the heavy bursts).

## 9. Temporary instrumentation still in tree (Rule 30)

Diagnosis-only session — removal timing is the maintainer's call:

1. `prover.cpp::proveKernel` — `trapT*` timestamps + `[BURST-PH]` print.
 **REMOVED** (maintainer directive
 after fix 1's acceptance).
2. `prover.cpp::prove` — `[ACTIVE-CENSUS]` end-of-batch census (+ goal-text
 decode). **REMOVED** (same
 directive); the production per-burst `active_bodies=` header remains the
 grid-inactivity observable.
3. `files/shortcut/theorems/conjectures.txt` on this branch was trimmed to
 the 44-row B10 pool during diagnosis. RESOLVED with fix 3: B11's row 45
 is restored byte-identical to `main` and the corrected strictOrder row
 is appended as row 46 (46 rows; 45 register by design — row 29 is the
 standing close-without-register case).
4. The hashburst dump target now points at B10's `(in2[rec0,10,3])` chain
 (Rule 14 retarget under explicit user direction — stays until retargeted).

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
