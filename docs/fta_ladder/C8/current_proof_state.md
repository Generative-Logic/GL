<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.



# C8 — difference closure of divisibility — current proof state

**Branch:** (the campaign line; `main` was reset back to the C7 squash on 2026-08-19 — Steps 1–8 were committed to `main` directly and moved here). Machinery work: (Steps 9–10), superseding the abandoned gate-widening experiment. **Status: PROVED — 56/56 saved on the first run of the input-slot demand machinery, verifier 8647/0 airtight (Step 10).** Human proof: [`proof_c8_difference_closure.md`](proof_c8_difference_closure.md).

## Reading convention

Append-only, newest step last. Every claim names its evidence file (, , trap output files).

## Compiled-name glossary

- `in[x,1]` — x ∈ N. `in2[a,b,3]` — b = s(a). `in3[x,y,z,4]` — x + y = z. `in3[x,y,z,5]` — x·y = z.
- `preorder[1,4,a,b]` — a ≤ b (∃k: a+k=b). `preorder[1,5,d,n]` — d | n (∃k: d·k=n). `strictOrder[1,4,a,b]` — a < b.
- `it_<n>_lev_<m>` — minted witness variables (generation-capped); `repl_lev_*` — universal replacement variables; `u_*` — free anchor parameters.
- AnchorFTA slots: 1=N, 2=0, 3=s, 4=+, 5=·, 6=1, 7=2, 8=fold carrier.

## The row

Row 54 of `files/shortcut/theorems/conjectures.txt`:

```text
(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](preorder[1,5,9,10])(>[11,12](in3[10,11,12,4])(>[](preorder[1,5,9,12])(preorder[1,5,9,11])))))
```

9=d, 10=a, 11=b, 12=c (the sum a+b).

## Target LB (Rule-12 chain, derived from `addTheoremToMemory`'s chain walk)

Each premise of the disintegrated implication becomes one nested LB keyed by that premise expression; the innermost premise block carries the head goal `(preorder[1,5,9,11])` in `toBeProved`. The sacred hashburst dump is retargeted there:

```text
root (empty exprKey, parentMemory == nullptr)
 → (AnchorFTA[1,2,3,4,5,6,7,8])
 → (preorder[1,5,9,10])
 → (in3[10,11,12,4])
 → (preorder[1,5,9,12])     ← dump target
```

## Steps

### Step 1 — campaign setup (2026-08-18)

- Row 54 appended on `main` directly after the C7 squash; pool 53 → 54 rows..
- Corpus- and pool-duplicate pre-checks 0 hits.
- Human proof written before the first frontier walk ([`proof_c8_difference_closure.md`](proof_c8_difference_closure.md)): §1–§11, with a case split at §5 and four minted witnesses (`x`, `y` premise existentials, `z` from the order fact, `q` from multiplication totality).
- Contrast to keep in view: C5/C6/C7 each closed by CONSTRUCTING one witness out of the premise witnesses plus one corpus rule firing. C8 must EXTRACT its head witness from an order fact, which is why the chain is longer and needs the `d = 0` split.

### Step 2 — run 1: the stall reproduces (2026-08-18)

Run: `main.py --shortcut`,  — **53/54 saved** (row 54 unproved), verifier 8421 checks / 0 failures on the survivors (unchanged from the 53-row C7 baseline, as expected — the same 53 rows proved). Overall runtime 181.3 s.

- The row survived CE filtering (`CE filter: 54 conjectures`), so this is a derivation gap, not a false conjecture.
- No machinery change had been made; the dump was still targeted at the C6-era Gauss fold LB and therefore never fired in `--shortcut` mode.

### Step 3 — dump retargeted at the C8 innermost premise LB (2026-08-18)

`hashburst_dump::isTargetLB` retargeted at the chain above, full parent chain to the root sentinel per Rule 12, nothing else in the dump touched (Rule 14). Diagnostic run pending; evidence will land as  + .

### Step 4 — run 2 frontier walk: the stall is a missing `≤`-producer, not a machinery gap (2026-08-18)

Run: `main.py --shortcut` with the retargeted dump,  — 53/54 saved again, verifier 8421/0 on the survivors, 184.5 s. Trace:  (94.8 MB, 25 bursts at the target LB). The dump fired, so the Rule-12 chain above is correct. `toBeProved` stays at 1 in every ENTRY and EXIT; the statement count reaches its fixpoint at 730 in burst 24 and does not move in burst 25 — a frozen stall, not a slow search.

Walk of the human proof against the final EXIT #25 statement set:

| § | expected | verdict |
|---|---|---|
| §1 `d·x = a` | `(in3[9,int_lev_1_1,10,5])` | **present** (`v=main`) |
| §2 `d·y = c` | `(in3[9,int_lev_3_1,12,5])` | **present** (`v=main`) |
| §3 `a + b = c` | `(in3[10,11,12,4])` | **present** (`v=main`) |
| §4 `a ≤ c` | `(preorder[1,4,10,12])` | **ABSENT — the frontier** |
| §5–§11 | — | absent (downstream of §4) |

The LB carries exactly 18 `preorder[1,4,…]` facts and every one of them is of the trivial `0 ≤ X` family (`(preorder[1,4,2,X])` and `(preorder[1,4,it_0_lev_0_30,X])`, that witness being a name in zero's class) — the products of pool row 3. Not one non-trivial order fact was ever derived.

**Mechanism.** No rule in the corpus or the pool introduces `≤` from an addition equation. Every `preorder[1,4,…]`-headed row (pool rows 3, 4, 10, 11, 13, 16, 18, 23, 24, 26, 31, 32, …) derives an order fact from *another* order fact, an equality, or a successor fact; the introduction direction `a + k = c ⟹ a ≤ c` exists nowhere. Turning the present equation into the existence compact would require a forward integration, and integration is demand-driven — the LB's `overallHashMemory.admissionMap` is empty (0 entries), so nothing ever demanded `a ≤ c`. B9 (pool row 42) can therefore never fire: its `p ≤ q` premise is unreachable, independently of its `1 ≤ d` premise.

Secondary observations, none of them blocking:

- The goal's hypothesis scope is open and populated: `main_boundary__var0_9_var1_11_hypo_(preorder[1,5,9,11])` carries the assumed witness `int_lev_3_3` with `(in3[9,int_lev_3_3,11,5])`, `(in[int_lev_3_3,1])`, plus `existence0/existence1` compacts and an `or2` predecessor split over it — 24 statements in all.
- Or rules ARE registered at this LB (`or0[9,10,1,4]`, a family of `or10[u_1,u_4,…]` pairs), confirming that in-run or minting and broadcast (`constructOrTheoremsInRun`) reaches a live premise LB. The ladder README's "same-run or-headed theorems cannot drive ordis branches" section was stale and is corrected in this commit.

**Classification: not a machinery gap — a missing true fact-producer**, i.e. the ladder's standing unfair-advantage case. Repair added as two pool rows (55, 56):

- **Row 55 — `≤`-introduction:** `(>[9,10,11](in3[9,10,11,4])(preorder[1,4,9,11]))` — `a + k = c ⟹ a ≤ c`. The introduction direction of the `preorder` definition; one integration over a bound witness in its own grid, the same shape pool row 4 already proves.
- **Row 56 — difference transport (chain collapse):** `(>[9,10,11](in3[9,10,11,5])(>[12,13](in3[11,12,13,4])(>[14](in3[9,14,13,5])(>[15](in3[10,15,14,4])(in3[9,15,12,5])))))` — `d·x = a ∧ a + b = c ∧ d·y = c ∧ x + z = y ⟹ d·z = b`. This folds §8–§10 (multiplication totality on a minted pair, distributivity, additive cancellation) into ONE rule over bound variables, so the C8 grid fires it with names it already holds instead of manipulating four minted witnesses at once.

### Step 5 — run 3 with rows 55/56: frontier moves two steps, stops at the missing case split (2026-08-18)

Run: `main.py --shortcut`,  — **55/56 saved**, verifier 8517/0 airtight, 181.3 s. Both helper rows proved; row 54 still unproved. Trace  (94.8 MB, 25 bursts, statements at their fixpoint of 735).

Walk against the human proof at EXIT #25:

| § | expected | verdict |
|---|---|---|
| §4 `a ≤ c` | `(preorder[1,4,10,12])` | **present now** — row 55 fired (also `b ≤ c`) |
| §6 premises | `x·d = a` `(in3[int_lev_1_1,9,10,5])`, `y·d = c` `(in3[int_lev_3_1,9,12,5])` | **present** — the commutativity rewrites happened |
| §6 `1 ≤ d` | `(preorder[1,4,6,9])` | **ABSENT — the new frontier** |
| §6 `x ≤ y` | `(preorder[1,4,int_lev_1_1,int_lev_3_1])` | absent (B9 cannot fire) |

Both helper rules are registered at the LB (`(in3[9,10,11,4]) (preorder[1,4,9,11])` and `(in3[9,10,11,5]) (in3[11,12,13,4]) (in3[9,14,13,5]) (in3[10,15,14,4]) (in3[9,15,12,5])`), and B9 sits there too (`(preorder[1,4,6,9]) (in3[10,9,11,5]) (in3[12,9,13,5]) (preorder[1,4,11,13]) (preorder[1,4,10,12])`) with **three of its four premises satisfied**. Only positivity is missing.

**The split is present as a fact and never opens.** `(or3[2,9,1,4,6])` — `d = 0 ∨ 1 ≤ d`, minted in-run from pool row 24 — is a statement at `v=main` on this LB. But `orBookkeeping`, `orDisjunctCount` and `orPendingBranches` are all 0: no cohort ever opened. The or parks instead — `rejectedMapOrdis` holds 52 entries and `rejectedMapOrdis2` 83, while `admissionMap`, `admissionMapOrdis2` and `admissionSetIntegration` are all empty. Cohort opening is demand-driven (the compound-demand map), and nothing on this LB demands either disjunct: GL does not plant a demand for the missing premise of a partially-matched rule, so B9's `1 ≤ d` gap never becomes or-opening evidence.

Same vacuum blocks row 56 from firing: its fourth premise `x + z = y` needs the witness `z`, which only exists once `x ≤ y` does.

**Consequence for the repair.** With positivity in hand the whole chain is cohort-free and needs exactly ONE minted witness: B9 gives `x ≤ y`, its disintegration gives `z`, row 56 gives `d·z = b`, and goal integration closes `d | b`. Row 57 therefore states the positivity-guarded form of C8 — the form every downstream consumer (D2, G1, G5) actually uses, since their divisor is a prime:

```text
(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](preorder[1,4,6,9])(>[10](preorder[1,5,9,10])(>[11,12](in3[10,11,12,4])(>[](preorder[1,5,9,12])(preorder[1,5,9,11]))))))
```

Row 54 (the unrestricted form) stays in the pool as the open frontier: closing it needs either or-cohort opening on demand for a partially-matched rule's missing premise (a machinery question), or an or-carrying premise shape that the De-Morgan door splits inside a helper's own grid. Maintainer decision.

### Step 6 — row 57 dropped; the guarded form stalls too (2026-08-18)

Run: `main.py --shortcut`,  — **55 saved from 57 rows**: row 54 AND row 57 both unproved, verifier 8517/0 on the survivors, 214.7 s (+33 s for the extra row). The positivity-guarded form does not close either, and the reason is untraced: the sacred dump is aimed at row 54's chain, not row 57's.

Row 57 removed from the pool on maintainer instruction — it was never a step in C8's proof (row 54 would still need `1 ≤ d` before it could fire row 57, which is the very fact it cannot get), and it did not earn its slot on its own merits either. Pool back to 56 rows: C8 plus the two helpers that did earn theirs (55 and 56 both prove and both fire).

**Where C8 stalls, exactly.** At the innermost premise LB `(preorder[1,5,9,12])`, one atom is missing: `(preorder[1,4,6,9])` = `1 ≤ d`. Everything on either side of it is in place —

- upstream: `d·x = a`, `d·y = c`, their commuted forms `x·d = a`, `y·d = c`, `a + b = c`, `a ≤ c`, `b ≤ c` all live at `v=main`;
- the rule that needs it: B9 registered at the LB with three of its four premises satisfied;
- the fact that would supply it: `(or3[2,9,1,4,6])` = `d = 0 ∨ 1 ≤ d`, a statement at `v=main` on this very LB.

The or never opens: `orBookkeeping`, `orDisjunctCount`, `orPendingBranches` all 0, ors parked in `rejectedMapOrdis` (52) and `rejectedMapOrdis2` (83), and `admissionMap` / `admissionMapOrdis2` / `admissionSetIntegration` all empty. Cohort opening is demand-driven; no demand for either disjunct is ever planted, because a partially-matched rule does not turn its missing premise into or-opening evidence. Downstream of that atom the chain is dead in one line: no `1 ≤ d` → no `x ≤ y` → no witness `z` → row 56's fourth premise never matches → no `d·z = b` → the goal `(preorder[1,5,9,11])` stays in `toBeProved` for all 25 bursts.

### Step 7 — D-287: the demand installs, the cohort opens, B9 fires (2026-08-18)

Maintainer hypothesis, confirmed in code: the ordis2 demand slot filter refused B9's slot because of a CONSTANT. `ordis2DemandSlotQualifies`'s subset test (D-267) requires every argument of the candidate slot to be bound by one of the rule's other NON-anchor premises. For `(preorder[1,4,6,9])`, `9` is bound by the two product premises and `1` / `4` by `(preorder[1,4,11,13])` — but `6`, the numeral 1, is an anchor slot and the binder loop skips anchor premises. One constant disqualified the whole order family of consumers. Every other gate passed (`ordis2KeyEligible`, arity 4, four non-anchor premises).

Fix ([D-287](../../agentic_swdd/40_decisions.md#d-287)): an anchor-slot argument counts as bound — it is ground by construction, so requiring a substantive binder for it is unsatisfiable rather than selective. A slot the anchor alone binds still mints nothing (the A14 shape) via an explicit guard.

Run: `main.py --shortcut`, , trace  — 55/56 saved (row 54 still open), verifier 8517/0, 190.7 s (was 181.3 s), unit tests 1466/1466.

The machinery moved exactly as predicted:

| observable | before | after |
|---|---|---|
| `admissionMapOrdis2` | 0 | **4** |
| `orBookkeeping` / `orDisjunctCount` / `orPendingBranches` | 0 / 0 / 0 | **128 / 1 / 1** |
| bursts at the LB / statements | 25 / 735 | 27 / 866 |
| `1 ≤ d` | absent | **present** (131 statements in the branch scope) |
| `x ≤ y` | absent | **present** — B9 fired |

The cohort opened on `orSig=(or3[2,9,1,4,6])`, `count=2`, released branch `((preorder[1,4,6,9]))`, `pending={((=[2,9]))}` (the `d = 0` branch waits per the one-at-a-time release). Inside the released branch: `(preorder[1,4,6,9])`, `!(=[2,9])`, `(preorder[1,4,int_lev_1_1,int_lev_3_1])` = `x ≤ y` — §6 of the human proof, reached.

**New frontier — §7, the witness of `x ≤ y`.** *(The reading below is CORRECTED in Step 8: the witness IS minted; what fails is registration.)* The order existence is EXPANDED and — as Step 8 shows — also witnessed on the `it_` path:

```text
!(>[int_lev_3_687](in[int_lev_3_687,1])!(in3[int_lev_1_1,int_lev_3_687,int_lev_3_1,4]))
  <- expansion | (preorder[1,4,int_lev_1_1,int_lev_3_1])
```

Compare the predecessor existence in the SAME branch scope, which does get a concrete witness:

```text
!(>[it_0_lev_3_690](in[it_0_lev_3_690,1])!(in2[it_0_lev_3_690,9,3]))
  → (in2[it_0_lev_3_690,9,3])   <- disintegration
```

The difference is the bound name kind: the disintegrated one binds an `it_` witness, the order expansions bind `int_` names (`int_lev_3_685` for `x ≤ a`, `int_lev_3_687` for `x ≤ y`, `int_lev_3_689` for `y ≤ c`) and stop there. No `≤` fact anywhere on this LB has a concrete additive witness — `a ≤ c` at main only ever carries the premise `(in3[10,11,12,4])`, whose witness `b` was already a name. Multiplication totality by contrast did mint (`(in3[10,int_lev_3_1,it_0_lev_3_114,5])`).

So row 56's fourth premise `x + z = y` still has no `z`, and the goal stays open. Statement count is at its fixpoint (866 at bursts 26 and 27), so this is a refusal, not a budget truncation.

### Step 8 — correction and the real §7 blocker: every order witness is minted and PARKED (2026-08-18)

Step 7's reading was wrong on one point. The `it_` path of `disintegrateExpr2` DOES run for the order existences — the witnesses exist:

```text
(in3[int_lev_1_1,it_0_lev_3_686,int_lev_3_1,4])
  <- disintegration | !(>[it_0_lev_3_686](in[it_0_lev_3_686,1])!(in3[int_lev_1_1,it_0_lev_3_686,int_lev_3_1,4]))
```

What never happens is REGISTRATION. The product carries an origin row but no statement row anywhere in the 27-burst trace; it sits in `rejectedMap` under its marker key, inside the released ordis branch:

```text
key=(in3[int_lev_1_1,marker,int_lev_3_1,4]) | v=main_boundary_ordis_(or3[2,9,1,4,6])_((preorder[1,4,6,9]))
  [0] renamed=(in3[int_lev_1_1,it_0_lev_3_686,int_lev_3_1,4]) | expr=(preorder[1,4,int_lev_1_1,int_lev_3_1])
      | iter=0 | levels={} | siblings=1
      sib[0] (in[it_0_lev_3_686,1])
```

**This is systematic, not specific to B9's output.** Every `preorder` on this LB parks its witness the same way, the cohort's own seed included:

| source existence | minted witness | parked key | scope |
|---|---|---|---|
| `1 ≤ d` (the or-branch seed) | `it_0_lev_3_682` | `(in3[6,marker,9,4])` | ordis branch |
| `x ≤ y` (B9's output) | `it_0_lev_3_686` | `(in3[int_lev_1_1,marker,int_lev_3_1,4])` | ordis branch |
| `x ≤ a` | `it_0_lev_3_684` | `(in3[int_lev_1_1,marker,10,4])` | ordis branch |
| `y ≤ c` | `it_0_lev_3_692` | `(in3[int_lev_1_1,marker,12,4])` | ordis branch |
| the `preorder[1,5,…]` divisibility family | various | `(in3[…,marker,…,5])` | main |

Not one `≤` or `|` on this LB registers its witness. Every parked entry carries `levels={}`. The witnesses that DO register in the branch come from other existence families — the predecessor `existence11[1,9,3]` (`it_0_lev_3_690`, giving `(in3[6,it_0_lev_3_690,9,4])` and `(in3[it_0_lev_3_690,6,9,4])`) and multiplication totality (`it_0_lev_3_114`).

**Demand state at the LB (final EXIT #27):** `admissionMap` 0 · `admissionMapIntegration` 2 — both the goal's marker `(in3[u_9,marker,u_11,u_5])` for `markedGoal=(preorder[1,5,9,11])` · `admissionMapOrdis2` 4 · `rejectedMap` 283. Nothing demands an addition of the parked shape, so no park is ever released.

**What releasing it would buy.** With `x + z = y` registered, row 56's other three premises are already live at main (`d·x = a`, `a + b = c`, `d·y = c`), so it fires in the branch and yields `d·z = b` — exactly the shape of the standing goal demand — and the integration closes `d | b` inside the `1 ≤ d` branch. C8 itself then needs the cohort to converge: the second branch is still pending (`pending={((=[2,9]))}`), and its chain is fully available from proved material — `d = 0` with `d·x = a` gives `a = 0` (corpus `n ∈ N ∧ 0·n = m ⟹ 0 = m`), `d·y = c` gives `c = 0`, `a + b = c` gives `b = 0` (pool rows 6/7), and `d | b` is `0 | 0` from pool row 49.

**The open lever.** The parked product needs an admission key `(in3[int_lev_1_1,marker,int_lev_3_1,4])` at the branch scope. The goal's demand (`d·k = b`) does not propagate back one level through row 56's unmatched premise, so that key never appears. It is the same hunger idea [D-267](../../agentic_swdd/40_decisions.md#d-267) built for or cohorts — and that [D-287](../../agentic_swdd/40_decisions.md#d-287) just unblocked — but it would have to run on the algebra admission map with an ATOMIC `in3` slot, which `ordis2KeyEligible` excludes by design (compound entities only). Maintainer decision; nothing written.

**Campaign state as of this step.** Pool 56 rows: C8 (row 54, open) plus helpers 55 (`a + k = c ⟹ a ≤ c`) and 56 (difference transport), both proved and both firing. Last run  / : 55/56 saved, verifier 8517/0, 190.7 s, unit tests 1466/1466. Sacred dump targets C8's innermost premise LB. Frontier walk: §1–§6 reached, §7 blocked at registration of the order witness.

### Step 9 — input-slot demand keys with an ancestor-inclusive probe (2026-08-19)

**The abandoned detour first.** On the Step-8 blocker was attacked by unconditionally widening the Pass-B standalone gate `isAllowedAsOperatorInput` to two-input operators. RT exploded —  killed at hash burst 6 with 138,594 total expressions and a 129 s sweep against a ~190 s whole-run baseline: every two-input existence product registering at every LB is the historically proven blow-up (I-6's evidence), bounded arity notwithstanding. The branch is preserved with its findings; the gate stays single-input.

**The structural diagnosis behind the new repair.** The algebra demand language was OUTPUT-slot only: the regular admission-marker route always marks the missing premise's output argument, and the only input-slot marker producer — the ordis route — installs `ordisOnly`-tagged keys that Pass-B is blind to by design (I-177). Hence the exact Step-8 trichotomy: output-slot witnesses (multiplication totality) register through output-slot demand, single-input-operator witnesses (the `in2` predecessor) register through the I-6 standalone gate, and input-slot witnesses of two-input operators (every `preorder` expansion's `in3` atom) had NO path and parked forever.

**The repair ([D-288](../../agentic_swdd/40_decisions.md#d-288), maintainer-directed option 1).** Demand-side, two deltas: (1) a fourth, independent qualification pass in `makeNormalizedKeysForAdmission` installs UNTAGGED input-slot marker variants for premise slots passing `inputSlotDemandSlotQualifies` — candidate at a head input slot, confined, other premise args bound elsewhere, ≥ 4 non-anchor premises (`kInputSlotDemandMinPremises`); (2) the Pass-B `it_` admission probes through `isAdmittedIncludingAncestors` — exact deposit validity, then its strict ancestors only — because a fired key lands at `deeperOf(subkey constituents)` (= `main` for the C8 shape) while the witness-minting disintegration deposits inside the released `_ordis_` branch.

**Predicted chain.** Row 56 qualifies at its `x + z = y` slot (candidate `z`); its subkey (anchor + `d·x = a`, `a + b = c`, `d·y = c`) matches at `main` from the early bursts, so the instantiated key `(in3[int_lev_1_1,marker,int_lev_3_1,4])` — byte-identical to Step 8's parked marker form — stands in `admissionMap` long before the cohort opens. When B9's `x ≤ y` expands in the branch, the ancestor-inclusive probe admits the witness instead of parking it → row 56 fires → `d·z = b` → the goal's standing integration demand closes `d | b` in the `1 ≤ d` branch → cohort convergence via the `d = 0` chain from proved material.

Run pending; evidence will land as  + .

### Step 10 — C8 PROVED on the first run of the input-slot demand machinery (2026-08-19)

Run: `main.py --shortcut`, , trace snapshot  — **56/56 saved** (row 54's raw MPL confirmed by `grep -F` in `files/shortcut/theorems/theorems.txt`), verifier **8647 checks / 0 failures — airtight** on both report blocks. Unit tests 1469/1469 (three new: the predicate matrix, the untagged-marker install, the ancestor walk). The Step-9 chain executed as predicted with no further intervention: the or-consumption categories confirm the cohort ran to closure (`or disintegration` 7, `or convergence` 3, `contradiction` 16, all clean).

**Runtime.** Prover 242.1 s (overall 246.1 s) against the 190.7 s D-287 baseline — +51 s from the new marker rules and demand traffic. Accepted for the prove; an RT pass (e.g. tightening which rules mint input-slot demand) is a candidate follow-up if the cost compounds on later rungs, and stands in sharp contrast to the abandoned unconditional widening, which did not finish burst 7 in ~11 minutes.

**Campaign state as of this step.** Pool 56 rows, ALL proved — the ladder is fully proved again through C8. Machinery landed ([D-288](../../agentic_swdd/40_decisions.md#d-288), [I-196](../../agentic_swdd/30_invariants.md#i-196)): the fourth qualification pass + `isAdmittedIncludingAncestors`; the I-6 gate untouched.

**Full-pipeline gate (2026-08-19, maintainer-ordered ahead of the squash).** `main.py` (no flags),  — verifier **airtight on both graphs: incubator 122317 / 0, main 16095 / 0, total 138412 / 0**; incubator pools 1036/156/204/49/8; overall runtime 1770.6 s. Compared against the last prior full run on disk (, the C6-era products-of-recursion gate): **all three totals AND all 77 per-category verifier lines byte-identical** (`diff` of the extracted category lines, maintainer-ordered comparison). The whole machinery delta since that gate — D-287 plus the input-slot demand route — is therefore output-invisible on the full pipeline at category granularity. (Theorem and proof-graph outputs are untracked, so the comparison is log-count-based, the campaign's standard gate.) `origin/main` force-pushed back to the C7 squash the same day (maintainer-executed instruction; campaign history preserved). Sacred dump still targets C8's innermost premise LB.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
