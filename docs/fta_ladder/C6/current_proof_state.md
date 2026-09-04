<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# C6 — sum closure of divisibility — current proof state

**Branch:** (diagnosis, Steps 1–8) → (fix campaign, Step 9; export-crash diagnosis + fix, Steps 10–11). **Status: C6 COMPLETE AND VERIFIED — 52/52 saved, export clean, verifier 7697 checks / 0 failures (Step 11). The Step-9 export crash was root-caused (Step 10) to a pre-existing mail-closure gap and closed by the maintainer-approved P2 fix ([D-286](../../agentic_swdd/40_decisions.md#d-286)): canonicalization-dropped delta rows ship their origin history before the drop. All Rule-30 traps removed.** The two pre-Step-1 C6 trials remain set aside by maintainer directive.

## Reading convention

Append-only, newest step last. Every claim names its evidence file (, , trap output files). Human proof: [`proof_c6_sum_closure.md`](proof_c6_sum_closure.md).

## Compiled-name glossary

- `in[x,1]` — x ∈ N. `in2[a,b,3]` — b = s(a). `in3[x,y,z,4]` — x + y = z. `in3[x,y,z,5]` — x·y = z.
- `preorder[1,4,a,b]` — a ≤ b (∃k: a+k=b). `preorder[1,5,d,n]` — d | n (∃k: d·k=n). `strictOrder[1,4,a,b]` — a < b.
- `it_<n>_lev_<m>` — minted witness variables (generation-capped, D-232 / `maxIterationNumberVariable`); `repl_lev_*` — universal replacement variables; `u_*` — free anchor parameters.
- AnchorFTA slots: 1=N, 2=0, 3=s, 4=+, 5=·, 6=1, 7=2, 8=fold carrier.

## The row

Row 52 of `files/shortcut/theorems/conjectures.txt`:

```text
(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](preorder[1,5,9,10])(>[11](preorder[1,5,9,11])(>[12](in3[10,11,12,4])(preorder[1,5,9,12])))))
```

9=d, 10=a, 11=b, 12=a+b.

## Target LB (Rule-12 chain, derived from `addTheoremToMemory`'s chain walk)

The innermost premise block carries the head goal `(preorder[1,5,9,12])` in `toBeProved`; the sacred hashburst dump is retargeted there:

```text
root (empty exprKey, parentMemory == nullptr)
 → (AnchorFTA[1,2,3,4,5,6,7,8])
 → (preorder[1,5,9,10])
 → (preorder[1,5,9,11])
 → (in3[10,11,12,4])        ← dump target
```

## Steps

### Step 1 — campaign setup (2026-08-17)

- created from `main` (C5 squash; ladder 51/51 fully proved).
- Row 52 appended to the pool (52 rows total).
- Human direct proof written ([`proof_c6_sum_closure.md`](proof_c6_sum_closure.md)): §1–§6, flat — two premise witnesses, one minted sum witness for the pair (x, y), one four-premise corpus rule firing (distributivity composition direction, `externally_provided_theorems.txt` line 25), one existence integration. The only non-premise, non-corpus ingredient is §4: the sum-witness existence `(in3[it_<x>,it_<y>,it_<w>,4])`.
- Sacred dump retargeted at the innermost premise LB (chain above), full parent chain per Rule 12.
- Contrast to keep in view during the frontier walk: C5 (row 51, transitivity) closed FIRST RUN on this same engine, and its proof consumes a minted PRODUCT witness over two minted witnesses (`x·y = z`); C6's §4 needs a minted SUM witness over two minted witnesses. Whatever mechanism produced C5's compound witness must be traced on both rows — the discriminator between the two is expected to localize the stall.

### Step 2 — run 1 frontier walk: the stall reproduces; frontier pinned to one unpaired totality request (2026-08-17)

Run: `main.py --shortcut`,  — **51/52 saved** (row 52 unproved), verifier 7489/0 on the survivors. Trace:  (92.5 MB, 28 bursts at the innermost premise LB; statement fixpoint 750 rows from burst 24, `toBeProved` stuck at 1).

Frontier walk against the human proof (all evidence from the final EXIT #28 dump):

- **§1–§3 present.** Premise `(in3[10,11,12,4])` is registry row [0]. The it_-flavored witness facts `(in3[9,it_0_lev_1_0,10,5])` / `(in3[9,it_0_lev_2_0,11,5])` exist in the origin map (mailed process documentation) but are NOT local registry rows — the proof-relevant witness carriers at THIS LB are the int_-flavored twins, which ARE local rows: `[44] (in3[9,int_lev_1_1,10,5])`, `[45] (in3[9,int_lev_2_1,11,5])`, `[49] (in[int_lev_1_1,1])`, `[50] (in[int_lev_2_1,1])` — all @main, all registered in burst 2.
- **§5 request EXISTS, parked one premise short.** The LB's `admissionMap` holds exactly the distributivity request (both orders): `markerExpr=(in3[int_lev_1_1,int_lev_2_1,marker,4])`, key = {AnchorPeano, (in3[10,11,12,4]), (in3[9,int_lev_1_1,10,5]), (in3[9,int_lev_2_1,11,5])}. GL assembled the whole §5 firing and awaits exactly §4 — the sum fact for the witness pair.
- **§4 is the frontier.** The totality rule `(>[1](in[1,u_1])(>[2](in[2,u_1])(existence1[u_1,1,2,u_4])))` is live at this LB and fired 262 `existence1` rows — for every locally-registered membership pair EXCEPT the one needed: `existence1[1,int_lev_1_1,int_lev_2_1,·]` never formed (either order, either operator), and no `existence1` row anywhere in the 92.5 MB trace mentions `it_0_lev_1_0` / `it_0_lev_2_0` either.
- **Freshness/semi-naive timing exonerated:** `(in[10,1])` ([53]) registered in the same burst 2 as [49]/[50]; in burst 3 the stump `(in[int_lev_1_1,1])` paired with `(in[10,1])` (72 ground existence1 rows fired) but not with `(in[int_lev_2_1,1])` from the same universe.
- **Pairing matrix:** `int_lev_1_1` paired with ground names {2,6,7,9,10,11,12} + the level-0 mint `it_0_lev_0_30`; `int_lev_2_1` additionally with the level-2 mints `it_0_lev_2_50/74`; never with each other, never self-pairs. Statement levels: `(in[int_lev_1_1,1])`={1}, `(in[int_lev_2_1,1])`={2}, ground ={0,...}, or-machinery mints ={-1} or {0}.
- **C5 positive control (this run's own chapter `44_direct_proof.txt`):** C5 fired the SAME totality rule over its two cross-level premise witnesses (v7,v8) and closed by integration. C5's head sits at its level-2 LB (two premises); C6's head sits at level 3 (three premises). C5 and C6 share the level-1 LB `(preorder[1,5,9,10])`.

Conclusion of Step 2: the failing step is exactly §4, localized to the request generator inside the innermost premise LB: with both membership premises locally registered, the rule installed, and standing admission demand for the head, the candidate pair `(in[int_lev_1_1,1]) × (in[int_lev_2_1,1])` never becomes a fired request. The dump cannot show which gate refuses it → Step 3 instruments the generator (playbook step 6).

### Step 3 — [C6TRAP] instrumentation installed (2026-08-17)

Temporary Rule-30 traps (all gated on the Rule-12 full-chain `isTargetLB`, logging to ): per-burst pool census of the two membership rows across `intEncodedStatements` / `intLocalEncodedStatements` / `intLocalEncodedStatementsDelta` / `intExternalStatements`; `filterIntEncodedStatements` verdicts (subkey-owner / full-key / iteration-cap) for those rows; grow-DFS candidate verdicts (`requestGatesPass`, subkey probe, record probe) for any candidate mentioning a pair var; merge-phase scope/dup drop and emit verdicts; seed-probe verdicts; `BurstSink::consume` dependency-skip vs firing-check for any such request. Dump retargeted back at the innermost premise LB.

### Step 4 — trap runs 1–3: the refusal caught red-handed and attributed to its map (2026-08-17)

Evidence files:  (19,076 lines),  (round 2: `preEvaluateFromEncoded` / `ownerKeyAccepts` sub-verdicts),  (round 3: map-instance addresses). Stall reproduced 51/52 in every run.

- **The pair IS assembled and refused.** `[MERGE b2] emit=0 (in[int_lev_1_1,1]) @main | (in[int_lev_2_1,1]) @main` — both stump/base orders, burst 2, the ONLY two whole-key merge attempts of the pure pair in 28 bursts. Adjacent `[OWNER]` lines prove the cause: `MISS` on the probed `normalizedEncodedKeys` map — not the shape gates (`requestGatesPass` passed), not comparability, not partition, not u_-satisfaction.
- **Map attribution (round 3, `[MAPS b2]` addresses).** The two b2 misses probed `localHashMemory.normalizedEncodedKeys` — batch 4 (external stumps × local rule map). Across all 28 bursts the pure pair produced **zero** probes against `overallHashMemory.normalizedEncodedKeys`, `localHashMemoryDelta.normalizedEncodedKeys`, or `workingMemory.normalizedEncodedKeys` — every map that ever holds the mail-recovered totality rule. All `cmp=1 part=1 uSat=1` accepts of the pair are grow-phase subkey/minus-one probes (base-candidate growth), never whole-key.
- **Controls (same bursts, same LB, same rule).** `[MERGE b3] emit=1 (in[int_lev_1_1,1]) | (in[10,1])` + `FIRING-CHECK` (batch 2: localDelta ground stump × `overallHashMemory`); the 4-premise distributivity request emits and plants the admission marker; `[SEED b2] keyOk=1 (in[int_lev_3_1,1])@hypo | (in[int_lev_1_1,1])@main` (batch-3 local×external pair against `overallHashMemory` — two distinct `int_lev` variables are NOT refused per se).
- **Pool census.** The two memberships ride `intExternalStatements` ONLY in their mail-arrival burst 2 and are never in `intLocalEncodedStatements` / `intLocalEncodedStatementsDelta` (mail-arrived = non-local). They pass `filterIntEncodedStatements` (subOk=1) throughout — the universe never drops them; only the stump/rule-map batch structure does.

**Root cause (architecture gap, trap-proven at the goal LB).** The rule-install fan-out (`addExprToMemoryBlock` implication branch, the status distribute) deliberately excludes a status-3 (external-mail-recovered) rule from `localHashMemory` / `localHashMemoryDelta`; it lives only in `overallHashMemory` + the per-burst `workingMemory`. The 5-batch obligatory-stump pipeline in `performElem2` then has no combination that can fire it on an all-mailed premise set:

| batch | rule map | stump source | pure pair (both memberships mail-arrived) |
|---|---|---|---|
| 1 | workingMemory (mail rules ✓) | local statements | no local member — impossible |
| 2 | overall (mail rules ✓) | local delta | impossible |
| 3 | overall (mail rules ✓) | local × external pairs | pair is external × external — impossible |
| 4 | localHashMemory (mail rules ✗) | external | forms — MISS by design (the two b2 refusals) |
| 5 | localHashMemoryDelta (mail rules ✗) | all statements | rule never in the map — dead |

Every request in the batches whose rule maps contain mail rules must contain at least one LOCALLY-derived statement. A mail-recovered rule therefore can never fire on premises that all arrived by mail. C6 §4 is exactly that firing: the corpus totality rule (mailed) over the two premise-witness memberships (minted at the level-1/level-2 ancestor LBs, mailed to level 3, never local there). C5 and every earlier row escaped because their goal LB is the LAST premise LB, which disintegrates its own compact premise LOCALLY — one witness is always a local statement there, and batch 2/3 carry the firing against `overallHashMemory`.

### Step 5 — run 4 (level-2 LB): the sum fact IS derived one level up and the mail-out gate strands it there (2026-08-17)

Run 4 retargeted the dump + traps at the level-2 LB `(preorder[1,5,9,11])` ( run-4 log, ; pairVarMask widened to `it_0_lev_1_0`/`it_0_lev_2_0`). Findings, all dump/trap-evidenced:

- **At level 2 the pair configuration is legal and it FIRED.** `(in[int_lev_2_1,1])` is LOCAL there (own premise disintegration, pool census b1: local+localDelta); `(in[int_lev_1_1,1])` arrives external in b2. Batch 3 (local × external, `overallHashMemory`) formed the pair — `[CONSUME b2] FIRING-CHECK` both orders — and the totality head registered: registry rows `[55] (existence1[1,int_lev_1_1,int_lev_2_1,4])`, `[63]` mirror.
- **The §4 sum instance exists at level 2 — but NOT as a statement** *(corrected 2026-08-17 after maintainer challenge; the original phrasing overstated)*. The registered head is the ONLY statement of the chain: `(in3[int_lev_1_1,int_lev_2_1,int_lev_2_17,4])` has ZERO indexed `encodedStatements` rows anywhere in the level-2 trace. The concrete sum instance lives in exactly two non-statement places: (1) `overallHashMemory.rejectedMapIntegration` as `concrete=(in3[int_lev_1_1,int_lev_2_1,int_lev_2_17,4]) | compound=(existence1[1,int_lev_1_1,int_lev_2_1,4]) | siblings=1` — a PARKED integration part (concrete instance prepared, admission refused); (2) the `exprOriginMap` (process documentation, Rule 16). The `it_`-flavored twin `(in3[int_lev_1_1,int_lev_2_1,it_0_lev_2_16,4])` is likewise origin-map-only, and `(in[int_lev_2_17,1])` is not a registry row either. WHY the `it_`-flavored disintegration products failed to register at level 2 (same-level fresh-witness exclusion vs admission parking) is code-inferred, not trap-proven. This parked state is the "disintegration gap — relay of parked witness-bearing existence parts to children" deferred at the end of the session.
- **The transfer down is blocked by `allowedForMail`** (`prover.cpp`; scanner `scanSingleDistinctIntLev` in `str_ops.hpp`): an expression carrying TWO OR MORE distinct `int_lev_*` tokens is categorically refused for mail (verdict 2 → false; a single token passes only via the `canBeSentIds` / marker-form memo). Level-2's EXIT #2 `mailOut.statements` (16 rows) carries every single-`int_lev` product (`existence1[1,int_lev_2_1,2/6/7,·]` — verdict 1, marker-memoized) and NONE of the pair chain (head, expansions, sum facts — all ≥2 distinct `int_lev` tokens). Level-3's origin map (run 1, I-26 all-scopes mail mirror) confirms nothing of the chain ever arrived.
- **Cross-check — the matrix asymmetries all decode.** `existence1[1,int_lev_2_1,it_0_lev_2_50,·]` rows at level 3 exist because they FIRED AT LEVEL 2 (both names local there) and their heads carry ONE `int_lev` token → mailable; `(int_lev_1_1 × it_0_lev_2_50)` at level 3 shows the Gap-A signature again (int_1 never local, 2_50 external one burst). The witness facts `(in3[9,int_lev_1_1,10,5])` etc. reached level 3 because each carries one `int_lev` token.

### Step 6 — final classification (2026-08-17)

**Architecture gap — two independent, individually sufficient blocking gates, both operating exactly as designed; no coding bug found.** The complete causal chain of the 51/52 stall:

1. The head `(preorder[1,5,9,12])` closes only through the BG1 distributivity firing at level 3, parked in `admissionMap` on the marker `(in3[int_lev_1_1,int_lev_2_1,marker,4])` (§5 — assembled, waiting; run 1).
2. **Gap A (request fan-out, level 3):** the §4 totality firing over the two witness memberships is impossible at level 3 — both memberships are mail-arrived (never local), and the 5-batch obligatory-stump pipeline gives mail-rule-holding maps only to local-stump batches (Step 4 table; trap-proven, map-attributed).
3. The §4 firing IS possible at level 2 (one membership local) and happens there; its head registers, but its disintegration products never become statements — the concrete sum instance parks in `rejectedMapIntegration` (Step 5, corrected).
4. **Gap B (mail-out gate, level 2 → 3):** the head — the chain's only registered statement — carries 2 distinct `int_lev` witnesses and is categorically unmailable (`allowedForMail` multi-`int_lev` refusal); the parked integration record and the origin entries are LB-local by design. Nothing of the chain can reach the LB whose demand waits for it.
5. Marker never completes → BG1 never fires → integration never receives its witness → row 52 unproved. Reproduced 51/52 across four runs.

**Why C6 is first:** every A/B/C row so far has its witness-bearing compact as the LAST premise, so the goal LB disintegrates it locally and no multi-witness product must travel. C6's last premise is the atomic `(in3[10,11,12,4])`; both witness compacts sit higher. The same shape recurs later in the ladder (C8, D2, G-rows) — the gap is load-bearing for the ladder's continuation, not a C6 quirk.

**Relation to the two earlier trials (set aside by maintainer directive):** trial 1's claim ("requests from mail-arrived statements never probe the general rule map") was an imprecise half of Gap A — batch 3 DOES probe the general map with mail statements, but only paired with a local one; all-mailed premise sets are what cannot. Trial 2's claim ("depth-1 demand blocks the level-2 sum witness") named the right level split with the wrong mechanism — the blocker is the mail-out gate, not a demand depth.

### Step 7 — trap runs 5–6: WHY the sum instance never registered, proven at probe time (2026-08-17, maintainer-directed)

Question: why is `(in3[int_lev_1_1,int_lev_2_1,int_lev_2_17,4])` never a registry row at the level-2 LB. Traps: `[PASSA]` / `[PASSB-IT]` / `[PASSB-INT]` verdict hooks inside `disintegrateExpr2`'s registration decision (Pass A unconditional set, Pass B witness admission), chain-gated on `isTargetLB(memoryBlock)` directly — run 5 proved the thread_local activation misses this path because the firing-record apply runs OUTSIDE `performElem2`. Evidence:  (run 6), stall 51/52 reproduced, units 1459/1459.

The registration decision, verbatim from the trap log (all four flavor/orientation variants behave identically — `int_lev_2_17/19/33/35`, `it_0_lev_2_16/18/32/34`, ops 4 and 5, both argument orders):

1. **Pass A excludes it:** `hasNewVars=1 trigger=int_lev_2_17 savedStartInt=16 mbLevel=2` — the statement carries the disintegration's own freshly minted witness (name level 2 == `mb.level`, id 17 ≥ `savedStartInt` 16), so it is barred from the unconditional registration set. The HEAD `(existence1[1,int_lev_1_1,int_lev_2_1,4])` in the same disintegration logs `hasNewVars=0` and registers — exactly the `[55]` registry row.
2. **Pass B rejects the witness on an EMPTY demand map:** int_ side — `REJECTED (isAdmittedIntegration=0, setProbe=0) admMapIntegration=0 admSetIntegration=0 marked=(in3[int_lev_1_1,int_lev_2_1,marker,4]) v=main`; it_ side — `REJECTED (isAdmitted=0, opInput=0) admMap=0`. Both admission probes ran against admission maps holding ZERO entries at probe time; the rejections then park the constituents (`rejectedMapIntegration` — the observed `concrete=|compound=` record — and `rejectedMap`).
3. **The demand that would have admitted it exists one LB below, byte-identical:** the probed marked template `(in3[int_lev_1_1,int_lev_2_1,marker,4]) @main` is exactly the LEVEL-3 LB's parked `admissionMap` marker (Step 2). Admission demand is strictly LB-local; the level-2 LB is goal-less (it never sees level-3's premise `(in3[10,11,12,4])`, so no BG1 request — hence no marker — can ever be planted there).

**Conclusion:** the sum instance is derived at level 2 and refused registration there because witness admission is demand-driven and the demand lives only at level 3 — the third face of the same structural split (Gap A: the firing cannot happen at level 3; Gap B: the head cannot mail down; Gap C/refinement: the witness cannot be admitted at level 2 because the demand cannot exist there). One more observed richness: the head fired in BOTH argument orders and BOTH operators (sum AND product existence for the pair) — all eight concrete instances parked identically.

**Directional completeness (maintainer question: does a PARENT LB's admissionMap hold an admitting key?).** No LB above the goal LB holds one, and none can: (a) the deriving level-2 LB — trap-proven zero admission entries at probe time (this step); (b) every ancestor above level 2 — structurally impossible, since the admitting key must encode both `int_lev_1_1` (a level-1 mint) and `int_lev_2_1` (a level-2 mint), and mail flows only ancestor → descendant (I-57), so no expression naming `int_lev_2_1` ever exists above level 2. The ONLY admissionMap in the chain holding the marker key is the level-3 goal LB — the CHILD of the deriving LB. The Pass-B admission probes are strictly LB-local (`mb.overallHashMemory.*`); no parent or child walk exists on this path (the one known cross-level arming, `updateAdmissionMap3`'s recursion digit-arg climb, is not in play — the chain has no recursion blocks). The demand and the derivation face each other across one parent→child seam, each on the wrong side.

### Step 8 — maintainer question: can a mail-arrived existence disintegrate at `(in3[10,11,12,4])`? No — the fourth gate (2026-08-17)

Code (`addExprToMemoryBlock`, the status-3 disintegration guard, `prover.cpp`): status-3 (external-mail) deposits reach `disintegrateExpr2` ONLY as the two rule-carrier shapes — `(implication<N>…` and `!(existence<N>…` (a universal in De-Morgan clothing; installs its rule, mints NO witnesses). A plain mailed statement — including a POSITIVE existence compact — registers as a bare row and never disintegrates; the comment names the reasons (fresh-mint `MAX_NAME_IDS` overflow, non-convergence runaway). Deliberate design, not a bug.

Empirical (run 1, level-3 EXIT 28): the mail-arrived head `(existence1[1,int_lev_2_1,it_0_lev_2_50,4])` is a registry row with ZERO local disintegration products. Stronger: the registry holds ZERO main-scope `lev_3`-minted witness rows at all (all 23 `lev_3` rows are hypothesis-scope, goal-side) — even the 262 locally fired existence heads produced no registered witnesses, since Pass-B admission is demand-gated universally and the only demanded markers at level 3 are the two distributivity ones.

**Classification:** a fourth co-sufficient gate — **Gap D** — on the mail route. In the actual run it was never exercised for the pair (Gap B blocked the head's transport first), but it is causally decisive in the counterfactual: with Gap B open, the head would arrive at level 3 as a bare status-3 statement and mint nothing, and the witness FACTS (three `int_lev` tokens each) are unmailable in their own right — the mail route is therefore blocked independently at two points.

Full gate stack for §4 at the demand LB: **A** — cannot fire there (all-mailed premise set × 5-batch structure); **B** — the level-2 head cannot mail down (multi-`int_lev` refusal); **C** — the level-2 witnesses cannot be admitted (demand is LB-local, unplantable at the deriving LB); **D** — even a mailed head would not disintegrate on arrival (status-3 rule-carrier-only gate).

### Step 9 — fix campaign: ROW 52 PROVED; one residual export gap (2026-08-17)

Maintainer design implemented in five commits ([D-284](../../agentic_swdd/40_decisions.md#d-284), [I-195](../../agentic_swdd/30_invariants.md#i-195)): (1) the parent-level pass in `allowedForMail` (an expression whose `int_lev` tokens all carry a level strictly below the LB's own level is sendable — the prior refusal of mail-arrived premises was a bug; it broke the `allGood` chain, Gap B); (2) a per-statement flag side column on cross-LB mail (the `disintegrationSignals` pattern — set key and sacred dump untouched); (3) sender-side relay selection (highest uncovered existence group per disintegration; staged per LB, local statuses only) + `fillMailOut` staging; (4) the status-5 receiver door (flag-5 arrivals run the FULL disintegration with witness minting at the carried generation stamp; carrier non-local, admitted products local — Gaps C, D; the local products also let batch 2 fire the parked distributivity, bypassing Gap A); (5) all `[C6TRAP]` instrumentation removed (Rule 30), dump retargeted at the goal LB.

**Result: `Saved 52 theorems` — row 52 IS in `files/shortcut/theorems/theorems.txt` (verified by exact `grep -F`). The four-gate stall is closed end to end** (). One scale error was caught by its own Rule-19 tripwire on the first run (): relay staging is NOT C6's ~8 compacts — EVERY undemanded fired existence head stages a relay, so one LB legitimately ships hundreds per pass; the `fillMailOut` bookkeeping went arena-backed (unbounded) in a follow-up commit.

**OPEN CRASH — the chapter export aborts after the save** ( first observation;  with the census trap; a follow-up session will debug). The exporting theorem, the failing LB, the missing dependency, and the referring row, verbatim from the trap run:

```text
[buildStack] no origin for: (=[it_0_lev_1_192,2]) | validity=main | exprKey=(=[9,2])_induction_rec0_
[buildStack] theorem: (>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](in[9,1])(>[]!(=[2,9])(>[]!(=[6,9])(preorder[1,4,7,9])))))
[buildStack] LB chain: <- (=[9,2])_induction_rec0_ <- !(=[6,9]) <- !(=[2,9]) <- (in[9,1]) <- (AnchorFTA[1,2,3,4,5,6,7,8]) <-
[C6ORIGIN] lb=(=[9,2])_induction_rec0_ originRows=0 registered=0
[C6ORIGIN] lb=!(=[6,9]) originRows=2 registered=1
[C6ORIGIN] lb=!(=[2,9]) originRows=1 registered=1
[C6ORIGIN] lb=(in[9,1]) originRows=2 registered=0
[C6ORIGIN] lb=(AnchorFTA[1,2,3,4,5,6,7,8]) originRows=0 registered=0
[buildStack] referring row: | (in3[9_copy,it_0_lev_0_30,2,5]) | main | equality1 | (in3[9_copy,2,it_0_lev_1_192,5]) | main | (=[2,it_0_lev_0_30]) | main | (=[it_0_lev_1_192,2]) | main
Assertion failed: false && "buildStack: no origin found", visualizer.cpp
```

Trap-proven facts, nothing more: the missing dependency `(=[it_0_lev_1_192,2]) @main` has origin rows at three ancestors — at `(in[9,1])` it holds 2 origin rows while NOT being a registered statement there (`registered=0` — per the maintainer there is no legitimate such state; WHY that LB holds origin rows for an expression it never registered is itself part of what the follow-up must explain); at the two hypothesis LBs it is registered with 1–2 rows — and the exporting recursion LB `(=[9,2])_induction_rec0_` holds neither rows nor registration, while its own origin map DOES hold the `equality1` referring row that cites the dependency. `it_0_lev_1_192` (a level-1 `it_` mint, id 192) does not exist in the baseline runs; the name volume is new to the relay traffic. NOT yet established: where the referring `equality1` row was emitted, how it reached the recursion LB (mailed origin mirror vs local emission), where and why `(in[9,1])` acquired origin rows for the unregistered equality, and why the recursion LB lacks what its two direct ancestors hold. The crash reproduces identically across both post-fix runs (52/52 saved, then this abort — the pipeline never reaches `process_proof_graphs.py` / `verifier.py`, so the airtightness of the 52 saved theorems is UNVERIFIED).

**Remaining instrumentation (Rule 30 report):** all `[C6TRAP]` blocks are REMOVED (fix-campaign commit 5). One temporary trap remains for the open export gap: the `[C6ORIGIN]` ancestor census in `visualizer.cpp` at the `buildStack` no-origin assert — remove with the export-gap fix. The sacred dump is retargeted at the C6 GOAL LB (`(in3[10,11,12,4])` chain). Diagnosis-era evidence files:  +  (run 6), , ; fix-campaign runs: .

### Step 10 — export crash ROOT-CAUSED: the canonicalization sweep eats a cited equality's mail inside its registration burst (2026-08-17)

Two instrumented runs, crash reproduced byte-identically in both ( +  + , dump at the exporting recursion LB;  +  + , dump + all traps single-LB-gated at `(in[9,1])` per maintainer directive — one LB at a time, full Rule-12 chain). The failing theorem is `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](in[9,1])(>[]!(=[2,9])(>[]!(=[6,9])(preorder[1,4,7,9])))))` (9∈N, 9≠0, 9≠1 ⟹ 2≤9); the crash LB is its born-parked induction-zero block `(=[9,2])_induction_rec0_`.

**The complete trap-proven causal chain** (diag5 seq numbers are one burst window — burst #8 of `(in[9,1])` per the dump; diag4 seq numbers where noted):

1. The flag-5 relay delivers `(existence1[1,9_copy,2,5])` (∃w∈N: 9_copy·0=w) to `(in[9,1])`; the status-5 door disintegrates it, minting witness `it_0_lev_1_192` (`[MINT-it]`; the name is subfan-unique — five disjoint level-1 subfans each mint their own, maintainer-confirmed by design).
2. The rule `(>[1](in[1,u_1])(>[2](in3[1,u_2,2,u_5])(=[2,u_2])))` (x·0=w ⟹ w=0) fires on the witness facts; the equality `(=[it_0_lev_1_192,2])` @main arrives on internal mail with `implication` origin rows (`[ABSORB-int]` seq 55/56) and registers local+fresh (`[ADD-EQ]` seq 108, delta row created; mirror co-registered).
3. `applyEquivalenceClass` emits the all-distinct rewrite variant `(in3[9_copy,it_0_lev_0_30,2,5])` with an `equality1` history line citing `(=[it_0_lev_1_192,2])` @main (`[EMIT-applyEqCls]` seq 179). The emitter's citation-closure assert passes LEGITIMATELY — the equality's origin rows exist at this LB.
4. `cleanUpExpressions`' `filterIterations` sweep (the class {it_0_lev_1_192, 2, …} formed this same burst from this very equality) drops the equality, its mirror, and 8 sibling non-canonical rows from `intLocalEncodedStatements` AND `intLocalEncodedStatementsDelta` (`[CLEANUP-local-drop]` seq 200–209, `[CLEANUP-delta-drop]` seq 210–214) — **before this burst's `fillMailOut`**. The levels/known rows are retained by design (I-58/D-93); the origin rows are retained per Rule 16.
5. The same burst's `fillMailOut` then ships the REWRITE with its origin mirror — including the citing history line (`[SHIP-stmt]`/`[SHIP-origins]` seq 219/220) — but the cited equality is no longer in the delta, so neither its statement nor its origin rows ever reach `mailOut`. No later burst re-ships it (the known-bit dedup keeps it out of every future delta). Diag4 proved the universality: across the whole run, ZERO ship events for `(=[it_0_lev_1_192,2])` @main at ANY LB, and zero `[ERASE-*]` events — no origin row was ever erased anywhere; the gap is purely never-mailed.
6. The born-parked induction-zero LB `(=[9,2])_induction_rec0_` catch-up-absorbs the full mail history on activation (diag4 seq 67718–71203) — receiving the citing `equality1` row (seq 68936) and the rewrite source's `disintegration` row (seq 68934), but never the cited equality's rows (they are in no mail stream). Its two hypothesis-LB ancestors hold rows for the equality only because they independently RE-DERIVED it locally (≤-antisymmetry and ≤0⟹=0 routes, diag4 seq 51962+ — those registrations were also cleanup-dropped before mailing, same mechanism). The zero block derives nothing itself.
7. `buildStack` at the recursion LB walks the citing row, probes `(=[it_0_lev_1_192,2])` @main in ITS `exprOriginMap`, finds zero rows, and asserts — correctly reporting a genuine origin-closure violation.

**Classification.** A PRE-EXISTING mail-closure gap, not a relay defect: an expression registered and canonicalization-dropped within one burst is invisible to `fillMailOut` (the sweep runs before the mail pass in phase 3), while `equality1` rewrite rows citing it DO mail — so the mail-out origin mirror can ship a history line whose cited dependency's history can never follow. Every earlier configuration escaped because receivers either re-derived the cited equality locally or never rendered a chapter through the citing row. The relay's witness minting at level-1 LBs (closing Gap D) created the first instance where a mailed citing row reaches a receiver — a born-parked induction-zero block — that can never re-derive the dependency. The maintainer's "no legitimate such state" observation (origin rows without registration at `(in[9,1])`) is resolved: DURING the run the equality was registered there (its `statementLevelsMap` row is in every dump section from EXIT #8 through the final EXIT #21); the runtime statement lists were legitimately pruned by the sweep while Rule 16 preserved the rows.

**Residual open observation (non-causal, unattributed):** the equality's levels row is present at `(in[9,1])` in the final burst's dump, yet the export-time census reads `registered=0` there — some post-run transition (discharge image write, reload, or the census probe itself) loses or misses the row. No trapped un-know path fired (`[UNKNOW-*]`/`[REMOVE-EXPR]` all zero at the target LB). Does not affect the crash: `buildStack` reads only origin rows.

**Fix proposals (Rule 8 — maintainer decides; nothing implemented):**

- **(P1) Ship-before-drop.** In `cleanUpExpressions`, ship each delta row (statement + origin mirror) before `filterIterations` drops it — or reorder phase 3 so `fillMailOut` walks the delta before the sweep prunes it. Restores "everything registered reaches descendants once"; also mails the non-canonical statement itself (policy change: canonicalization currently suppresses exactly that).
- **(P2) Origin-mirror-only ship for dropped rows — recommended.** For each delta row the sweep drops, call the existing `copyOriginRowsToMailOut` before dropping (all scopes, like the fillMailOut origin copy). Statements stay unmailed — the canonicalization policy is untouched — but every receiver gets the dependency's history rows, restoring the closure invariant the mail origin mirror is built on (the same closure the emitter-side `citeFound` asserts enforce locally). Pure Rule-16 process documentation; smallest semantic footprint.
- **(P3) Export-side parent-chain fallback.** `buildStack`, on a missing origin, could walk `parentMemory` (precedent: the D-51 contradiction-twin switch). It would resolve THIS crash only because the hypothesis ancestors happen to hold re-derived rows — it does not close the producer-side closure gap and fails wherever no ancestor re-derives; insufficient alone.

**Remaining instrumentation after Step 10 (Rule 30 report — diagnosis-only session, traps stay until the fix):** the `[C6T2]` helper block + `c6tWatched` matchers (memory.hpp), the single-LB-gated `c6tIsTargetLB`/`c6tOriginRecord`/`c6tEvent` helpers and all `[EMIT-*]`/`[ABSORB-*]`/`[SHIP-*]`/`[DOOR-*]`/`[P2CLEAR]`/`[CLEANUP-*]`/`[UNKNOW-*]` site traps (prover.hpp), the `[ADD-EQ]`/`[SITEF-refuse]`/`[MINT-it]`/`[ERASE-*]`/`[REMOVE-EXPR]`/`[UNKNOW-parkedOr]`/`[EMIT-ordisRevive/ordis2Revive]` traps (prover.cpp), and the `[C6ORIGIN]`+`[C6ORIGIN2]` census (visualizer.cpp). The sacred dump is currently retargeted at `(in[9,1])` (root → AnchorFTA → (in[9,1])). Evidence files: , ,  (diag5),  (recursion-LB dump),  (diag5, `(in[9,1])` dump).

### Step 11 — P2 fix shipped and verified end to end: 52/52, export clean, verifier 7697/0 (2026-08-17)

Maintainer approved proposal P2. Implemented as [D-286](../../agentic_swdd/40_decisions.md#d-286): in `cleanUpExpressions`' delta filter, every row `filterIterations` drops calls the existing `copyOriginRowsToMailOut` (all scopes, the `fillMailOut` cap expression) BEFORE the drop — the statement stays unmailed (canonicalization policy unchanged), only the origin history travels, restoring the mail-mirror closure invariant ("every dep cited by a shipped origin row has its own rows shipped"). `decodeView` spans are I-3-safe at the call (the mailOut origin door mints only the mailbox's private interner). SwDD updated in the same commit: I-26 amended (+ AGENT_SwDD.md quick-ref row), `20_core_concepts/03_mail_system.md` new paragraph, `docs/user_swdd/provenance.html` §7.2 closure note.

**Rule 30 completed:** ALL temporary instrumentation from Steps 9–10 is removed in the fix commit — the `[C6T2]` blocks in memory.hpp / prover.hpp / prover.cpp, the `[C6ORIGIN]`+`[C6ORIGIN2]` census in visualizer.cpp — and the sacred dump is retargeted back at the C6 GOAL LB (`(in3[10,11,12,4])` chain, full Rule-12 parent chain). No asserts touched. The final verification therefore ran the production path.

**Verification ():** full MSBuild rebuild clean; `gl_quick --unit-tests` 1465/1465; full `main.py --shortcut` pipeline: `Saved 52 theorems` (row 52 confirmed by exact `grep -F` in `files/shortcut/theorems/theorems.txt`), chapter export completes with ZERO `buildStack` aborts, and the verifier — reached for the FIRST time on this branch — reports **7697 checks, 0 failures — airtight** across every tag category and meta-check (C5 baseline was 7489/0 on 51 rows). Overall runtime 157.3 s.

**Residual open observation (carried from Step 10, non-causal, unattributed):** at the diag5 crash the missing equality's `statementLevelsMap` row was present at `(in[9,1])` in every dump through the final burst EXIT, yet the export-time census read `registered=0` there — some post-run transition (discharge image write, reload, or the census probe) lost or missed the row. All census instrumentation is now removed; if this matters later, re-instrument at the discharge/reload seam.

### Step 12 — induction implication progression: the order-dependent progression bug (diagnosis + solution design; implementation next session) (2026-08-17)

**Symptom.** The FULL pipeline (not `--shortcut`) on this branch loses both Gauss fold theorems: `files/theorems/theorems.txt` has 24 `AnchorGauss` rows and zero `fold` rows (main: 26 and 2). The shortcut ladder (52/52) is unaffected.

**Diagnosis (the common understanding: an order-dependent progression bug).** The marker-form admissionMap keys are the premise CURSORS of the induction (auxy) implication: the implication installs with its full premise chain, and GL eats its way through it as premise matches arrive — `updateAdmissionMapRecursion` substitutes each arrived output value into the chain and derives the next cursor key. This progression must be order-independent: the final eaten state a pure function of the SET of arrived matches. It is not. A cursor key has two consumers — the implication's own progression and the revival of parked witness products — and consumption is permanent (`cleanAdmissionMap` erases the key's admissionMap value AND status row and mints `consumedAdmissionKeys`; every writer's consumed-check then skips forever). A match arriving BEFORE its cursor exists parks; the cursor's first write revives it; the consumption kills the key; the implication can never advance through that slot. On this branch the flag-5 relay changed the arrival order (round-3 census: status-5 existence compacts disintegrate in the fold successor induction LB at bursts 1–2 and their products park under the cursor family before any demand exists) — which exposed the bug; any other reordering (mail schedule, LB split) can reach it.

**Gauss instance, trap-proven.** Cursor `(in2[6,marker,3])` is consumed at burst 4 immediately after its first write; the cursor chain through `(in2[2,marker,3])`, `(in2[9,marker,3])`, `(in2[rec,marker,int_lev_2_1])` up to `(in3[Σ,9,marker,4])` never runs; the parked `Σ(rec)+9` totality witness sleeps in `rejectedMap` (`levels={}`) from burst 22 to the final EXIT #32; the fold-step rule never fires; the linear-combination head `(in3[7,12,11,5])` never closes. Main runs the whole chain and closes at burst 28, matching reference chapter `294_check_induction_condition.txt`.

**Method + evidence.** Four A/B trap rounds vs `main`, everything single-LB-gated (full Rule-12 chain root → `(AnchorGauss[1,2,3,4,5,6,7,8])` → `(in3[9,10,11,5])` → `(fold[1,3,4,8,2,9,12])` → `(in2[9,10,3])` → `(in2[rec0,9,3])`): sacred-dump retarget both sides; `[GAUSST]` demand/registration/sweep/revival timeline; `[GAUSST-UAMR]` derivation-walk verdicts; the round-3 admission-key write/erase/rendezvous census. Main-side traps on the user-directed auxiliary (tip ). Evidence: , , , .

**Solution (maintainer-designed 2026-08-17).** At every admissionMap writer's consumed-key check — today a bare skip — enrich the consumed branch: a consumed key MEANS that premise slot is already satisfied by a registered statement (the consumption event was exactly the admission of a witness for that slot), so FIND the matching registered expression and FEED it to the eating machinery so the implication progresses past the dead slot immediately.

Implementation contract for the next session:
1. **One shared helper, called from all four writer consumed-skips** — `updateAdmissionMap`'s install branch, `updateAdmissionMapRecursion`'s derive loop, `drainAdmissionKeysAlgebra`, and `applyEquivalenceClassToAdmissionMap`'s rekey drain. Patching only one site leaves the hole reachable through the others (the census shows consumed-skips at the algebra drain AND the equi rekey on BOTH branches).
2. **Advance from the value in hand, never through the front door.** `cleanAdmissionMap` erased both the admissionMap value and the status row, so re-offering the found statement to `updateAdmissionMapRecursion` is a no-op (status/key gates). Every blocked writer holds the partially-eaten chain (derive loop's `newKey`/`newRemaining`, the staged algebra record, the rekey record, the install's chain) — run the substitution step on that.
3. **Satisfier search canonicalizes first.** Direct consumption registers an exact-form satisfier; an I-41 closure victim's satisfier exists in class-canonical form only. Search the statement registry for the marked template's satisfiers under `canonicalizeUnderClasses`; if none found even then, the consumption invariant is broken — Rule-19 assert, never a silent skip.
4. **Loop, not one step.** The next cursor derived from the satisfier may itself be consumed — advance again until an unconsumed cursor inserts (normal insert path; revival fires as today) or the chain is fully eaten, landing in the existing fully-bound `prepareIntegration` end case. Terminates: each step eats one slot, chain length capped by `MAX_ADMISSION_KEY_ELEMENTS`. Multiple satisfiers per slot advance once each, in decoded-lex order (I-84 discipline).
5. **Statified discipline:** heap-free helper — arena worklist instead of recursion, scratch-arena buffers, decoded-lex snapshot for the registry walk (Rules 26/28, cookbooks 09b/09c).
6. **Non-issue, no compensating machinery:** a witness product parking under an already-consumed key sleeps forever and that is fine — the parked record quarantines the head AND its witness siblings, the fresh witness name is referenced nowhere else, its only content ("a value exists for this slot") is already delivered by the satisfier, and a revival would only hand it to the equivalence logic for elimination as a non-canonical member.

**Verification plan:** implement, remove ALL `[GAUSST]` traps in the fix commit (Rule 30; both this branch and become obsolete — the aux branch is preserved, never deleted), restore the sacred dump target if the maintainer directs, then: full rebuild + unit tests; FULL pipeline on this branch must recover both Gauss fold theorems (`files/theorems/theorems.txt`: 26 `AnchorGauss` rows including 2 `fold` rows, `grep -F` against ) with an airtight verifier; `--shortcut` must stay 52/52 with an airtight verifier. SwDD: new decision + invariant placeholders per Rule 15 (the order-independence contract of induction-implication progression and the consumed-key advance), Rule 10/20 chapter updates in the same commit.

**Rule 30 inventory (traps stay until the fix):** `[GAUSST]` helpers (declarations memory.hpp, definitions memory.cpp), round-1 traps (`updateAdmissionMap`, `prepareIntegration`, `cleanUpExpressions` drops, `addExprToMemoryBlock` watched arrivals, `revisitRejected2` entry/hit), round-2 `[GAUSST-UAMR]` derivation-walk traps, round-3 census (`[GAUSST-CAM/INS/EQK/ALG/RDV]`). The sacred dump is retargeted at the fold successor LB on this branch.

### Step 13 — from-scratch step-trace campaign: first failing step S49; purity-gate root cause TRAP-PROVEN both sides (2026-08-18)

Maintainer directive after both prior diagnoses failed: start from scratch, trace main's successful proof step by step in human notation, match every step against its branch analogue, find the first failing step, then trap-debug it. Deliverables and results, full detail in the two campaign documents:

- [`gauss_fold_main_proof.md`](gauss_fold_main_proof.md) — main's successor-case proof (chapter `294_check_induction_condition.txt`, 199 rows) in human notation, bottom to top, no gaps, stages A–N.
- [`gauss_fold_step_match.md`](gauss_fold_step_match.md) — all 199 steps matched under an evidence-pinned witness-name bijection: **194/199 exist on the branch** (including the fold integration Σ(rec) at branch burst 26 and the induction-hypothesis firing at burst 27 — both prior diagnoses had implicitly placed the death far earlier); exactly five missing, one chain: **S49 (Σ(rec)+9 sum-witness registration, the first failing step) → S50 → S40 → S51 → S199 (head)**.
- **Trap round `[SUMT]`** (marker-capture census in `checkLocalEncodedMemoryStatic` + `productsOfRecursionIds` mint census, identical traps both sides — this branch commit, commit ): the sum demand `(in3[Σ,9,marker,4])` is **captured pure on main** (premises cite the witness-form id-image fact `(in2[9,it_0_lev_4_0,8])` = chapter S39; `it_0_lev_4_0` ∈ productsOfRecursion, minted locally by the induction progression) and **refused impure on the branch** (all five instances, culprit named inline: the relay-minted `it_0_lev_3_78`, mailed from the ancestor, never admitted locally, not a product of recursion). The purity gate operates correctly on both sides; the defect is the displacement — the relay's foreign witness-form becomes the sole surviving id-image instance while the branch's own induction-built pure twin (`it_0_lev_4_94` ∈ POR) is removed by the class/door handling (the Step-12 "missing admission key" was the symptom two links downstream; the Step-14 swallow is the removal of the pure twin; the ground form `(in2[9,9,8])` plays no causal role — cited by no captured request on main).

Fix directions under maintainer decision (Rule 8, nothing implemented): classmate-POR enrichment (recommended), door-side canonicalization, or relay-side form. `[SUMT]` traps active on both branches until the fix (Rule 30).

### Step 14 — FIX SHIPPED AND VERIFIED: classmate-POR enrichment; Gauss fold RECOVERED; campaign CLOSED (2026-08-18)

Maintainer approved the classmate-POR design ("equi args also enter product of recursion set" / "try the fix"). Shipped as [D-285](../../agentic_swdd/40_decisions.md#d-285) / [I-194](../../agentic_swdd/30_invariants.md#i-194): `ExpressionAnalyzer::enrichProductsOfRecursionFromChangedClasses` — at the single-threaded `standardProcessing` seam, immediately after the `applyEquiClasses` fixpoint, every MAIN-scope class in `changedClassesThisStep` holding at least one `productsOfRecursionIds` member has all its members minted into the set (write-side closure; the phase-2 purity probe stays a plain `contains`; non-main scopes excluded). Unit test `prover, classmate_por_enrichment` (positive / no-member / non-main / idempotency).

**Verification, all green:**

1. **Instrumented trial** (, traps kept per the keep-traps-during-fix-verification discipline): fold RECOVERED — 26 AnchorGauss / 2 fold; the trap log shows the mechanism red-handed — `[SUMT-POR-mint site=classmate name=it_0_lev_3_78]` at startInt 1304 (the burst the relay witness joined the 9-class), then `SUMT-CAPTURE` of the sum demand; verifier 138412/0 airtight.
2. **Rule-30 cleanup**: ALL temporary traps removed from this branch — [GAUSST] rounds 1–3 (helpers, registration door, revival, PREPINT family, CUE drops, UAMR walk, UAM/INS/CAM/EQK/ALG/RDV census) and all [SUMT] sites. No asserts touched; the sacred dump untouched (still targeted at the fold successor LB). The aux keeps its traps (preserved dead-end trap branch, maintainer directive). Clean rebuild + units 1466/1466.
3. **Clean `--shortcut`** (): **52/52 saved, verifier 8329/0 airtight** (check count up from the 7697 baseline — the enrichment lets more demand stage; check-count deltas are not loss signals).
4. **Clean FULL pipeline** (, production path): **26 AnchorGauss / 2 fold — both fold rows byte-exact matches of main's (`grep -F -x` against ) — verifier 16095/0 (Peano leg) and 138412/0 (full) airtight.**

**Rule 30 report:** no temporary instrumentation remains on this branch. Evidence files for the whole campaign: , , , , , . The two campaign documents ([`gauss_fold_main_proof.md`](gauss_fold_main_proof.md), [`gauss_fold_step_match.md`](gauss_fold_step_match.md)) are the permanent record of the method and the A/B proof.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
