<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Gauss fold — main proof steps matched against the branch, step by step

**Purpose.** For every step of main's successful fold proof ([`gauss_fold_main_proof.md`](gauss_fold_main_proof.md), steps S1..S199 of chapter `294_check_induction_condition.txt`), this document records the branch's analogue — the same statement under the branch run's minted names — or its absence. The walk is over proof steps, never over engine state: the two runs' algorithms differ (the flag-5 relay changes arrival orders and witness levels), so burst-by-burst state comparison is meaningless; only "does the branch ever derive this proof step" counts.

**Evidence.** Main side:  (sacred dump at the fold successor induction LB, 28 bursts) — every chapter statement was located in its `encodedStatements` sections with its first burst of appearance. Branch side:  (same LB, 32 bursts), searched for the exact analogue under the witness-name bijection below. Burst numbers in the table are first appearance in the respective run's statement registry at this LB.

## Witness-name bijection (main ↔ branch)

Minted names differ between the runs. The bijection below was established by matching each witness's *defining* statement (the disintegration that minted it) on both sides; applying it to a main statement gives the branch statement byte-exactly, which is how the table's branch column was verified.

| main | branch | meaning | note |
|---|---|---|---|
| `it_0_lev_4_0` | `it_0_lev_3_78` | the s(rec) witness (= 9) | branch mints it at **level 3** — relay effect |
| `it_0_lev_4_46` | `it_0_lev_4_140` | the rec+0 witness (= rec) | branch uses ONE witness where main has two |
| `it_0_lev_4_14` | `it_0_lev_4_140` | the 0+rec witness (= rec) | collapsed with the row above on the branch |
| `it_0_lev_4_60` | `it_0_lev_4_154` | the rec·9 product witness | |
| `it_0_lev_4_378` | `it_0_lev_4_88` | the 9·2 product witness | |
| `it_0_lev_2_54` | `it_0_lev_2_56` | level-2 s-totality witness | |
| `it_0_lev_0_30` | `it_0_lev_0_30` | predecessor-of-1 witness (= 0) | same name |
| `int_lev_2_0` | `int_lev_2_0` | the interval [0..9] | same name |
| `int_lev_2_1` | `int_lev_2_1` | the fold sequence on [0..9] | same name |
| `int_lev_4_1741` | `int_lev_4_2313` | the interval [0..rec], first instance | both first at burst 11 |
| `int_lev_4_1926` | `int_lev_4_2346` | the interval [0..rec], second instance | both first at burst 19 |
| `int_lev_4_1925` | `int_lev_4_2345` | the limit sequence on [0..rec] | |
| `int_lev_4_1960` | `int_lev_4_2380` | Σ(rec) — the value i4_1925(rec) | both first at burst 21 |
| `it_0_lev_4_2367` | `it_0_lev_4_2767` | boundary-subproof witness | |
| `it_0_lev_4_3091` | — | **the sum witness Σ(rec)+9** | **never minted as a statement on the branch** |
| `repl_lev_4_*` | `repl_lev_4_*` | boundary universals | same names (boundary scope names carry the int mapping) |

Side observation (not load-bearing for the verdict): the branch additionally derives ground-rewritten forms main lacks, e.g. `(in3[rec,2,rec,4])` / `(in3[2,rec,rec,4])` (rec+0 = rec) at burst 5 — the class {rec, it_0_lev_4_140} rewrote earlier there. The witness-level shift (level 3 vs level 4 for the s(rec) witness) and the two-into-one witness collapse are the visible fingerprints of the relay's changed arrival order.

## Verdict

**194 of 199 steps have a verified branch analogue. Exactly five steps are missing on the branch, and they form a single dependency chain — the sum bridge and everything above it:**

| step | main burst | statement (human) | role |
|---|---|---|---|
| **S49** | 23 | Σ(rec) + 9 = w4_3091 | **the first failing step** — sum-witness registration |
| S50 | 23 | Σ(rec) + w4_0 = w4_3091 | equality1 rewrite of S49 |
| S40 | 24 | i2_1(9) = w4_3091 | the fold-step law firing at position 9 (needs S50) |
| S51 | 25 | Σ(rec) + w4_0 = 12 | S50 grounded by w4_3091 = 12 (S45 — an equality row, statement-registry-invisible on main because classes consume equalities, and nonexistent on the branch since w4_3091 is never minted) |
| S199 | 28 | 2·12 = 11 | the head (needs S51, S40-chain) |

Every earlier stage — the anchor expansion, the whole Peano corpus, all witness arithmetic (stages A–F), the predecessor interval (G), the limit sequence and its value Σ(rec) = `int_lev_4_2380` (H), both integration subproofs (I, J), the full fold integration `Σ(rec) = i4_2380` (K, branch burst 26), and **even the induction-hypothesis firing 2·Σ(rec) = rec·9 (L, branch burst 27)** — exists on the branch. The branch proof dies at exactly one point:

**First failing step: S49.** Its input S47 — the +-totality existence `∃v: Σ(rec)+9 = v`, i.e. `(existence1[1,int_lev_4_2380,9,4])` — DOES fire on the branch (burst 22, same burst as main). The disintegration of that existence also RUNS on the branch: the branch origin map at EXIT #22 holds the witness binders (int-flavor `int_lev_4_3466`, it-flavor generation-0 `it_0_lev_4_3465` — trace lines 1877887/1881567) and the concrete instance `(in3[int_lev_4_2380,9,int_lev_4_3466,4])` with a `disintegration` origin row (line 1899568). What never happens on the branch is the **registration**: no flavor of the concrete sum fact ever becomes a statement. On main the it-flavor product `(in3[int_lev_4_1960,9,it_0_lev_4_3091,4])` registers in burst 23.

## Registration-side evidence at the failing step

Witness registration from an existence disintegration is admission-gated (Pass B, demand-driven — the same door C6's Step 7 documented). The dumps show the demand-side difference directly:

- **Main:** `overallHashMemory.consumedAdmissionKeys` at EXIT #23 contains `(in3[int_lev_4_1960,9,marker,4]) @ main` — the demand key for exactly the S49 sum was planted and consumed inside the burst-22→23 window (it never appears as a *standing* `admissionMap` row in any dump — plant, hit, and `cleanAdmissionMap` consumption all fall between dump points). Its consumption is the admission that let `it_0_lev_4_3091` register.
- **Branch:** no `(in3[int_lev_4_2380,9,marker,4])` (nor any second-argument variant) ever appears — not standing in any burst's `admissionMap`, not in `consumedAdmissionKeys` through the final EXIT #32. The demand for the sum is never planted, so Pass B has nothing to admit against, and the product parks.

The trap question is therefore precise: **on main, which writer plants `(in3[int_lev_4_1960,9,marker,4])` during bursts 22–23, from what trigger — and why does the branch's corresponding trigger not reach that writer?** That is the subject of the trap round below. Everything upstream of this single demand-plant event is proof-equal between the runs (194/199 steps, including every premise the sum-demand's planting could depend on: Σ(rec)'s value fact at burst 21/21, its membership `(in[int_lev_4_2380,1])` at 21/21, `9 ∈ N` at 2/2, the fold-step law installed at 2/2).

## The full walk table

One row per proof step: step, main first-burst, branch first-burst, verdict, statement. `nonstatement` = the row is not a statement-registry entry on EITHER side (rules live in hash memory, integration-goal artifacts and witness binders live in the origin map; their presence is verified indirectly through their consequence steps, which are all `ok` below S49). The branch column is the exact bijected expression's first burst.

```text
 step  mB  bB  verdict       expression @ scope
   S1   3   3  ok            (in3[rec,6,9,4]) @ main
   S2   1   1  ok            (in2[9,10,3]) @ main
   S3   2   2  ok            (implication14[1,3,4]) @ main
   S4   -   -  nonstatement  (>[1](in[1,u_1])(>[2](in2[1,2,u_3])(>[3,4](in3[3,1,4,u_4])(>[5](in2[4,5,u_3])(in3[3,2,5,u_4]))))) @ main
   S5   4   4  ok            (in3[rec,7,10,4]) @ main
   S6   5   5  ok            (in3[it_0_lev_4_46,7,10,4]) @ main
   S7   4   4  ok            (in3[9,rec,it_0_lev_4_60,5]) @ main
   S8   5   5  ok            (in3[9,it_0_lev_4_46,it_0_lev_4_60,5]) @ main
   S9   1   1  ok            (in3[9,10,11,5]) @ main
  S10   6   6  ok            (in3[it_0_lev_4_60,it_0_lev_4_378,11,4]) @ main
  S11  12  12  ok            (in[it_0_lev_4_14,int_lev_4_1741]) @ main
  S12  13  13  ok            (in[it_0_lev_4_14,int_lev_2_0]) @ main
  S13   2   2  ok            (implication0[int_lev_2_1,int_lev_2_0]) @ main
  S14   -   -  nonstatement  (>[1,2](in2[1,2,u_int_lev_2_1])(in[1,u_int_lev_2_0])) @ main
  S15   2   2  ok            (in[9,int_lev_2_0]) @ main
  S16  21  21  ok            (in2[it_0_lev_4_14,int_lev_4_1960,int_lev_4_1925]) @ main
  S17  22  22  ok            (in2[it_0_lev_4_14,int_lev_4_1960,int_lev_2_1]) @ main
  S18   4   4  ok            (in[it_0_lev_4_46,1]) @ main
  S19   2   2  ok            (implication11[1,2,4]) @ main
  S20   -   -  nonstatement  (>[1](in[1,u_1])(>[2](in3[1,u_2,2,u_4])(=[1,2]))) @ main
  S21   5   5  ok            (=[rec,it_0_lev_4_46]) @ main
  S22   7   5  ok            (in3[2,it_0_lev_4_46,it_0_lev_4_14,4]) @ main
  S23   3   5  ok            (in[it_0_lev_4_0,1]) @ main
  S24   4   6  ok            (=[it_0_lev_4_0,9]) @ main
  S25   2   2  ok            (implication12[1,2,4]) @ main
  S26   -   -  nonstatement  (>[1,2](=[1,2])(>[](in[1,u_1])(>[](in[2,u_1])(in3[1,u_2,2,u_4])))) @ main
  S27   5   5  ok            (in3[it_0_lev_4_0,2,9,4]) @ main
  S28   2   2  ok            (AnchorGauss[1,x2,3,4,5,6,7,8]) @ main
  S29   -   -  nonstatement  (>[1,2,3,4,5,6,7,8](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10,11](in3[9,10,11,4])(in3[10,9,11,4]))) @ main
  S30   6   5  ok            (in3[2,it_0_lev_4_0,9,4]) @ main
  S31   4   5  ok            (in3[rec,6,it_0_lev_4_0,4]) @ main
  S32   3   3  ok            (existence1[1,rec,2,4]) @ main
  S33   -   -  nonstatement  !(>[it_0_lev_4_46](in[it_0_lev_4_46,1])!(in3[rec,2,it_0_lev_4_46,4])) @ main
  S34   4   4  ok            (in3[rec,2,it_0_lev_4_46,4]) @ main
  S35   5   5  ok            (in2[it_0_lev_4_46,it_0_lev_4_0,3]) @ main
  S36   2   2  ok            (implication13[1,3,4]) @ main
  S37   -   -  nonstatement  (>[1](in[1,u_1])(>[2](in2[1,2,u_3])(>[3,4](in3[3,1,4,u_4])(>[5](in3[3,2,5,u_4])(in2[4,5,u_3]))))) @ main
  S38   8   5  ok            (in2[it_0_lev_4_14,9,3]) @ main
  S39   5   7  ok            (in2[9,it_0_lev_4_0,8]) @ main
  S40  24   -  **MISSING**   (in2[9,it_0_lev_4_3091,int_lev_2_1]) @ main
  S41   2   2  ok            (in2[9,12,int_lev_2_1]) @ main
  S42   -   -  nonstatement  (&(&(&(implication0[int_lev_2_1,int_lev_2_0])(implication1[int_lev_2_1,1]))(implication2[int_lev_2_0,1,int_lev_2_1]))(implication3[int_lev_2_0,int_...
  S43   2   2  ok            (implication3[int_lev_2_0,int_lev_2_1]) @ main
  S44   -   -  nonstatement  (>[1](in[1,u_int_lev_2_0])(>[2](in2[1,2,u_int_lev_2_1])(>[3](in2[1,3,u_int_lev_2_1])(=[2,3])))) @ main
  S45   -   -  nonstatement  (=[it_0_lev_4_3091,12]) @ main
  S46  21  21  ok            (in[int_lev_4_1960,1]) @ main
  S47  22  22  ok            (existence1[1,int_lev_4_1960,9,4]) @ main
  S48   -   -  nonstatement  !(>[it_0_lev_4_3091](in[it_0_lev_4_3091,1])!(in3[int_lev_4_1960,9,it_0_lev_4_3091,4])) @ main
  S49  23   -  **MISSING**   (in3[int_lev_4_1960,9,it_0_lev_4_3091,4]) @ main
  S50  23   -  **MISSING**   (in3[int_lev_4_1960,it_0_lev_4_0,it_0_lev_4_3091,4]) @ main
  S51  25   -  **MISSING**   (in3[int_lev_4_1960,it_0_lev_4_0,12,4]) @ main
  S52   2   2  ok            (implication2[1,1,3]) @ main
  S53   -   -  nonstatement  (>[1](in[1,u_1])(existence0[u_1,1,u_3])) @ main
  S54   3   3  ok            (existence0[1,rec,3]) @ main
  S55   -   -  nonstatement  !(>[it_0_lev_4_0](in[it_0_lev_4_0,1])!(in2[rec,it_0_lev_4_0,3])) @ main
  S56   3   5  ok            (in2[rec,it_0_lev_4_0,3]) @ main
  S57   2   2  ok            (implication3[1,3]) @ main
  S58   -   -  nonstatement  (>[1](in[1,u_1])(>[2](in2[1,2,u_3])(>[3](in2[1,3,u_3])(=[2,3])))) @ main
  S59   4   6  ok            (=[9,it_0_lev_4_0]) @ main
  S60   2   2  ok            (implication1[3,1]) @ main
  S61   -   -  nonstatement  (>[1,2](in2[1,2,u_3])(in[2,u_1])) @ main
  S62   2   2  ok            (in[7,1]) @ main
  S63   3   3  ok            (existence1[1,9,7,5]) @ main
  S64   -   -  nonstatement  !(>[it_0_lev_4_378](in[it_0_lev_4_378,1])!(in3[9,7,it_0_lev_4_378,5])) @ main
  S65   5   5  ok            (in3[9,7,it_0_lev_4_378,5]) @ main
  S66   5   5  ok            (in3[it_0_lev_4_0,7,it_0_lev_4_378,5]) @ main
  S67   6   6  ok            (in3[7,it_0_lev_4_0,it_0_lev_4_378,5]) @ main
  S68   -   -  nonstatement  (&(&(&(&(implication20[int_lev_2_0,1,4,2])(implication21[int_lev_2_0,1,4,9]))(implication22[1,4,2,9,int_lev_2_0]))(in[2,1]))(in[9,1])) @ main
  S69   2   2  ok            (in[9,1]) @ main
  S70   2   2  ok            (fXYZ[5,1,1,1]) @ main
  S71   -   -  nonstatement  (&(&(&(&(implication6[5,1])(implication7[5,1]))(implication8[5,1]))(implication9[1,1,1,5]))(implication10[1,1,5])) @ main
  S72   2   2  ok            (implication9[1,1,1,5]) @ main
  S73   -   -  nonstatement  (>[1](in[1,u_1])(>[2](in[2,u_1])(existence1[u_1,1,2,u_5]))) @ main
  S74   3   3  ok            (existence1[1,rec,9,5]) @ main
  S75   -   -  nonstatement  !(>[it_0_lev_4_60](in[it_0_lev_4_60,1])!(in3[rec,9,it_0_lev_4_60,5])) @ main
  S76   3   3  ok            (in3[rec,9,it_0_lev_4_60,5]) @ main
  S77  19  19  ok            (sequence[1,4,2,rec,int_lev_4_1925]) @ main
  S78  19  19  ok            (sequence[1,4,2,it_0_lev_4_14,int_lev_4_1925]) @ main
  S79   2   2  ok            (in[rec,1]) @ main
  S80   2   2  ok            (fXYZ[4,1,1,1]) @ main
  S81   -   -  nonstatement  (&(&(&(&(implication6[4,1])(implication7[4,1]))(implication8[4,1]))(implication9[1,1,1,4]))(implication10[1,1,4])) @ main
  S82   2   2  ok            (implication9[1,1,1,4]) @ main
  S83   -   -  nonstatement  (>[1](in[1,u_1])(>[2](in[2,u_1])(existence1[u_1,1,2,u_4]))) @ main
  S84   3   3  ok            (existence1[1,2,rec,4]) @ main
  S85   -   -  nonstatement  !(>[it_0_lev_4_14](in[it_0_lev_4_14,1])!(in3[2,rec,it_0_lev_4_14,4])) @ main
  S86   7   5  ok            (in3[2,rec,it_0_lev_4_14,4]) @ main
  S87   8   5  ok            (=[rec,it_0_lev_4_14]) @ main
  S88  11  11  ok            (interval[1,4,2,it_0_lev_4_14,int_lev_4_1741]) @ main
  S89   -   -  nonstatement  (>[1,2,3,4,5,6,7,8](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10](interval[1,4,2,9,10])(>[11](sequence[1,4,2,9,11])(fXY[11,10,1])))) @ main
  S90  20  20  ok            (fXY[int_lev_4_1925,int_lev_4_1741,1]) @ main
  S91   -   -  nonstatement  (&(&(&(implication0[int_lev_4_1925,int_lev_4_1741])(implication1[int_lev_4_1925,1]))(implication2[int_lev_4_1741,1,int_lev_4_1925]))(implication3[i...
  S92  20  20  ok            (implication2[int_lev_4_1741,1,int_lev_4_1925]) @ main
  S93   -   -  nonstatement  (>[1](in[1,u_int_lev_4_1741])(existence0[u_1,1,u_int_lev_4_1925])) @ main
  S94  21  21  ok            (existence0[1,rec,int_lev_4_1925]) @ main
  S95   -   -  nonstatement  !(>[int_lev_4_1960](in[int_lev_4_1960,1])!(in2[rec,int_lev_4_1960,int_lev_4_1925])) @ main
  S96  21  21  ok            (in2[rec,int_lev_4_1960,int_lev_4_1925]) @ main
  S97  11  11  ok            (implication21[int_lev_4_1741,1,4,rec]) @ main
  S98   -   -  nonstatement  (>[1](in[1,u_int_lev_4_1741])(preorder[u_1,u_4,1,u_rec])) @ main
  S99  21  21  ok            (preorder[1,4,repl_lev_4_7,rec]) @ main_boundary_(implication25[int_lev_4_1741,3,int_lev_4_1925,8,4])
 S100  20  20  ok            (in[repl_lev_4_7,int_lev_4_1741]) @ main_boundary_(implication25[int_lev_4_1741,3,int_lev_4_1925,8,4])
 S101  21  21  ok            (in[repl_lev_4_7,int_lev_2_0]) @ main_boundary_(implication25[int_lev_4_1741,3,int_lev_4_1925,8,4])
 S102  20  20  ok            (in[repl_lev_4_6,int_lev_4_1741]) @ main_boundary_(implication25[int_lev_4_1741,3,int_lev_4_1925,8,4])
 S103  11  11  ok            (limitSet[1,4,int_lev_2_0,rec,int_lev_4_1741]) @ main
 S104   -   -  nonstatement  (&(&(implication29[int_lev_4_1741,int_lev_2_0])(implication21[int_lev_4_1741,1,4,rec]))(implication30[int_lev_2_0,1,4,rec,int_lev_4_1741])) @ main
 S105  11  11  ok            (implication29[int_lev_4_1741,int_lev_2_0]) @ main
 S106   -   -  nonstatement  (>[1](in[1,u_int_lev_4_1741])(in[1,u_int_lev_2_0])) @ main
 S107  21  21  ok            (in[repl_lev_4_6,int_lev_2_0]) @ main_boundary_(implication25[int_lev_4_1741,3,int_lev_4_1925,8,4])
 S108  20  20  ok            (in3[repl_lev_4_8,repl_lev_4_9,repl_lev_4_10,4]) @ main_boundary_(implication25[int_lev_4_1741,3,int_lev_4_1925,8,4])
 S109  20  20  ok            (in2[repl_lev_4_7,repl_lev_4_9,8]) @ main_boundary_(implication25[int_lev_4_1741,3,int_lev_4_1925,8,4])
 S110  20  20  ok            (in2[repl_lev_4_6,repl_lev_4_8,int_lev_4_1925]) @ main_boundary_(implication25[int_lev_4_1741,3,int_lev_4_1925,8,4])
 S111  19  19  ok            (implication27[int_lev_4_1925,int_lev_2_1]) @ main
 S112   -   -  nonstatement  (>[1,2](in2[1,2,u_int_lev_4_1925])(in2[1,2,u_int_lev_2_1])) @ main
 S113  21  21  ok            (in2[repl_lev_4_6,repl_lev_4_8,int_lev_2_1]) @ main_boundary_(implication25[int_lev_4_1741,3,int_lev_4_1925,8,4])
 S114   -   -  nonstatement  (>[repl_lev_4_6](in[repl_lev_4_6,int_lev_4_1741])(>[repl_lev_4_7](in2[repl_lev_4_6,repl_lev_4_7,3])(>[](in[repl_lev_4_7,int_lev_4_1741])(>[repl_lev...
 S115  20  20  ok            (in2[repl_lev_4_6,repl_lev_4_7,3]) @ main_boundary_(implication25[int_lev_4_1741,3,int_lev_4_1925,8,4])
 S116   2   2  ok            (implication25[int_lev_2_0,3,int_lev_2_1,8,4]) @ main
 S117   -   -  nonstatement  (>[1](in[1,u_int_lev_2_0])(>[2](in2[1,2,u_3])(>[](in[2,u_int_lev_2_0])(>[3](in2[1,3,u_int_lev_2_1])(>[4](in2[2,4,u_8])(>[5](in3[3,4,5,u_4])(in2[2,5...
 S118  22  22  ok            (in2[repl_lev_4_7,repl_lev_4_10,int_lev_2_1]) @ main_boundary_(implication25[int_lev_4_1741,3,int_lev_4_1925,8,4])
 S119   -   -  nonstatement  (in2[repl_lev_4_7,repl_lev_4_10,int_lev_4_1925]) @ main_boundary_(implication25[int_lev_4_1741,3,int_lev_4_1925,8,4])
 S120  24  24  ok            (implication25[int_lev_4_1741,3,int_lev_4_1925,8,4]) @ main
 S121   -   -  nonstatement  (>[repl_lev_4_0](in2[2,repl_lev_4_0,8])(in2[2,repl_lev_4_0,int_lev_4_1925]))_integration_goal @ main
 S122  19  19  ok            (in2[2,repl_lev_4_0,8]) @ main_boundary_(implication24[2,8,int_lev_4_1925])
 S123   2   2  ok            (implication18[8]) @ main
 S124   -   -  nonstatement  (>[1,2](in2[1,2,u_8])(=[1,2])) @ main
 S125   -   -  nonstatement  (=[2,repl_lev_4_0]) @ main_boundary_(implication24[2,8,int_lev_4_1925])
 S126   5   5  ok            (=[it_0_lev_0_30,2]) @ main
 S127   -   -  nonstatement  (=[it_0_lev_0_30,repl_lev_4_0]) @ main_boundary_(implication24[2,8,int_lev_4_1925])
 S128   -   -  nonstatement  (>[1,2,3,4,5,6,7,8](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10](interval[1,4,2,9,10])(in[9,10]))) @ main
 S129  12  12  ok            (in[rec,int_lev_4_1741]) @ main
 S130   -   -  nonstatement  (>[1,2,3,4,5,6,7,8](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](interval[1,4,2,10,11])(existence8[1,4,11,9,2])))) @ main
 S131  11  11  ok            (existence8[1,4,int_lev_2_0,rec,2]) @ main
 S132   -   -  nonstatement  !(>[int_lev_4_1741](limitSet[1,4,int_lev_2_0,rec,int_lev_4_1741])!(interval[1,4,2,rec,int_lev_4_1741])) @ main
 S133  11  11  ok            (interval[1,4,2,rec,int_lev_4_1741]) @ main
 S134   -   -  nonstatement  (&(&(&(&(implication20[int_lev_4_1741,1,4,2])(implication21[int_lev_4_1741,1,4,rec]))(implication22[1,4,2,rec,int_lev_4_1741]))(in[2,1]))(in[rec,1]...
 S135  11  11  ok            (implication20[int_lev_4_1741,1,4,2]) @ main
 S136   -   -  nonstatement  (>[1](in[1,u_int_lev_4_1741])(preorder[u_1,u_4,u_2,1])) @ main
 S137  13  13  ok            (preorder[1,4,2,rec]) @ main
 S138   2   2  ok            (in[2,1]) @ main
 S139   2   2  ok            (in2[6,7,3]) @ main
 S140   2   2  ok            (fXY[3,1,1]) @ main
 S141   -   -  nonstatement  (&(&(&(implication0[3,1])(implication1[3,1]))(implication2[1,1,3]))(implication3[1,3])) @ main
 S142   2   2  ok            (implication0[3,1]) @ main
 S143   -   -  nonstatement  (>[1,2](in2[1,2,u_3])(in[1,u_1])) @ main
 S144   2   2  ok            (in[6,1]) @ main
 S145   4   4  ok            (existence3[1,6,3]) @ main
 S146   -   -  nonstatement  !(>[it_0_lev_0_30](in[it_0_lev_0_30,1])!(in2[it_0_lev_0_30,6,3])) @ main
 S147   4   4  ok            (in2[it_0_lev_0_30,6,3]) @ main
 S148   -   -  nonstatement  (&(&(&(&(&(&(&(&(&(&(&(&(in[2,1])(fXY[3,1,1]))(implication4[1,2,3]))(implication5[1,3]))(fXYZ[4,1,1,1]))(implication11[1,2,4]))(implication12[1,2,4...
 S149   2   2  ok            (implication5[1,3]) @ main
 S150   -   -  nonstatement  (>[1](in[1,u_1])(>[2](in2[2,1,u_3])(>[3](in2[3,1,u_3])(=[2,3])))) @ main
 S151   5   5  ok            (=[2,it_0_lev_0_30]) @ main
 S152   2   2  ok            (identity[1,8]) @ main
 S153   -   -  nonstatement  (&(&(implication0[8,1])(implication18[8]))(implication19[1,8])) @ main
 S154   2   2  ok            (implication19[1,8]) @ main
 S155   -   -  nonstatement  (>[1](in[1,u_1])(>[2](=[1,2])(in2[1,2,u_8]))) @ main
 S156   6   6  ok            (in2[2,it_0_lev_0_30,8]) @ main
 S157   2   2  ok            (and0[2,8,int_lev_2_1,9,12,int_lev_2_0,3,4]) @ main
 S158   -   -  nonstatement  (&(&(implication24[2,8,int_lev_2_1])(in2[9,12,int_lev_2_1]))(implication25[int_lev_2_0,3,int_lev_2_1,8,4])) @ main
 S159   2   2  ok            (implication24[2,8,int_lev_2_1]) @ main
 S160   -   -  nonstatement  (>[1](in2[u_2,1,u_8])(in2[u_2,1,u_int_lev_2_1])) @ main
 S161   7   7  ok            (in2[2,it_0_lev_0_30,int_lev_2_1]) @ main
 S162   2   2  ok            (interval[1,4,2,9,int_lev_2_0]) @ main
 S163   1   1  ok            (fold[1,3,4,8,2,9,12]) @ main
 S164   -   -  nonstatement  !(>[int_lev_2_0](interval[1,4,2,9,int_lev_2_0])!(existence2[int_lev_2_0,1,2,8,9,12,3,4])) @ main
 S165   2   2  ok            (existence2[int_lev_2_0,1,2,8,9,12,3,4]) @ main
 S166   -   -  nonstatement  !(>[int_lev_2_1](fXY[int_lev_2_1,int_lev_2_0,1])!(and0[2,8,int_lev_2_1,9,12,int_lev_2_0,3,4])) @ main
 S167   2   2  ok            (fXY[int_lev_2_1,int_lev_2_0,1]) @ main
 S168   -   -  nonstatement  (>[1,2,3,4,5,6,7,8](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10](fXY[9,10,1])(>[11](interval[1,4,2,11,10])(sequence[1,4,2,11,9])))) @ main
 S169   4   4  ok            (sequence[1,4,2,9,int_lev_2_1]) @ main
 S170   1   1  ok            (in2[rec,9,3]) @ main
 S171   -   -  nonstatement  (>[1,2,3,4,5,6,7,8](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](sequence[1,4,2,10,11])(existence15[1,4,9,11,2])))) @ main
 S172  19  19  ok            (existence15[1,4,rec,int_lev_2_1,2]) @ main
 S173   -   -  nonstatement  !(>[int_lev_4_1925](limitSequence[1,4,rec,int_lev_2_1,int_lev_4_1925])!(sequence[1,4,2,rec,int_lev_4_1925])) @ main
 S174  19  19  ok            (limitSequence[1,4,rec,int_lev_2_1,int_lev_4_1925]) @ main
 S175   -   -  nonstatement  (&(&(implication26[int_lev_4_1925,1,4,rec])(implication27[int_lev_4_1925,int_lev_2_1]))(implication28[1,4,rec,int_lev_2_1,int_lev_4_1925])) @ main
 S176  19  19  ok            (implication28[1,4,rec,int_lev_2_1,int_lev_4_1925]) @ main
 S177   -   -  nonstatement  (>[1](preorder[u_1,u_4,1,u_rec])(>[2](in2[1,2,u_int_lev_2_1])(in2[1,2,u_int_lev_4_1925]))) @ main
 S178  20  20  ok            (in2[2,it_0_lev_0_30,int_lev_4_1925]) @ main
 S179   -   -  nonstatement  (in2[2,repl_lev_4_0,int_lev_4_1925]) @ main_boundary_(implication24[2,8,int_lev_4_1925])
 S180  21  21  ok            (implication24[2,8,int_lev_4_1925]) @ main
 S181   -   -  nonstatement  (&(&(implication24[2,8,int_lev_4_1925])(in2[rec,int_lev_4_1960,int_lev_4_1925]))(implication25[int_lev_4_1741,3,int_lev_4_1925,8,4]))_integration_g...
 S182   -   -  nonstatement  (>[](implication24[u_2,u_8,u_int_lev_4_1925])(>[](in2[u_rec,u_int_lev_4_1960,u_int_lev_4_1925])(>[](implication25[u_int_lev_4_1741,u_3,u_int_lev_4_...
 S183  24  24  ok            (and0[2,8,int_lev_4_1925,rec,int_lev_4_1960,int_lev_4_1741,3,4]) @ main
 S184   -   -  nonstatement  (>[](fXY[int_lev_4_1925,int_lev_4_1741,1])(and0[2,8,int_lev_4_1925,rec,int_lev_4_1960,int_lev_4_1741,3,4]))_integration_goal @ main
 S185   -   -  nonstatement  (>[](fXY[u_int_lev_4_1925,u_int_lev_4_1741,u_1])(>[](and0[u_2,u_8,u_int_lev_4_1925,u_rec,u_int_lev_4_1960,u_int_lev_4_1741,u_3,u_4])(existence2[u_i...
 S186  25  25  ok            (existence2[int_lev_4_1741,1,2,8,rec,int_lev_4_1960,3,4]) @ main
 S187   -   -  nonstatement  (>[](interval[1,4,2,rec,int_lev_4_1741])(existence2[int_lev_4_1741,1,2,8,rec,int_lev_4_1960,3,4]))_integration_goal @ main
 S188   -   -  nonstatement  (>[](interval[u_1,u_4,u_2,u_rec,u_int_lev_4_1741])(>[](existence2[u_int_lev_4_1741,u_1,u_2,u_8,u_rec,u_int_lev_4_1960,u_3,u_4])(fold[u_1,u_3,u_4,u_...
 S189  26  26  ok            (fold[1,3,4,8,2,rec,int_lev_4_1960]) @ main
 S190   -   -  nonstatement  (>[10,11](in3[u_rec,10,11,u_5])(>[12](fold[u_1,u_3,u_4,u_8,u_2,u_rec,12])(>[](in2[u_rec,10,u_3])(in3[u_7,12,11,u_5])))) @ main
 S191  27  27  ok            (in3[7,int_lev_4_1960,it_0_lev_4_60,5]) @ main
 S192   2   2  ok            (in2[2,6,3]) @ main
 S193   1   1  ok            (AnchorGauss[1,2,3,4,5,6,7,8]) @ main
 S194   -   -  nonstatement  (&(&(&(NaturalNumbers[1,2,3,4,5])(in2[2,6,3]))(in2[6,7,3]))(identity[1,8])) @ main
 S195   2   2  ok            (NaturalNumbers[1,2,3,4,5]) @ main
 S196   -   -  nonstatement  (&(NaturalNumbers[1,2,3,4,5])(in2[2,6,3]))_integration_goal @ main
 S197   -   -  nonstatement  (>[](NaturalNumbers[u_1,u_2,u_3,u_4,u_5])(>[](in2[u_2,u_6,u_3])(AnchorPeano[u_1,u_2,u_3,u_4,u_5,u_6]))) @ main
 S198   2   2  ok            (AnchorPeano[1,2,3,4,5,6]) @ main
 S199  28   -  **MISSING**   (in3[7,12,11,5]) @ main
```

## Trap plan for the failing step (next action)

Rule-30 temporary traps, all single-LB-gated on the fold successor induction LB by full Rule-12 chain, both sides (branch: this branch; main side: ):

1. **[SUMDEM] demand-plant census** — every write of an `admissionMap` key whose marker matches `(in3[<Σ-name>,*,marker,4])` or whose key-set mentions the Σ name (`int_lev_4_1960` / `int_lev_4_2380`): writer site (`updateAdmissionMap` install / `updateAdmissionMapRecursion` derive / `drainAdmissionKeysAlgebra` / equi-class rekey), the full key, status value, and burst.
2. **[SUMDEM] the trigger** — at the same sites, log the request/statement that triggered the write attempt and every early-return taken before the write (consumed-key skip, status gate, key-eligibility gate).
3. **[SUMADM] Pass-B verdict at the product** — at `disintegrateExpr2`'s registration decision for the concrete sum instances (both flavors), the exact probe results (`isAdmitted`, `isAdmittedIntegration`, set probes) — the branch side is expected to show the empty-demand rejection; the main side shows which probe admits.

Expected outcome: the main-side trap names the exact writer and trigger statement for the sum-demand plant; the branch-side trap shows the same trigger arriving and which gate stops the writer — that gate is the bug surface. No fix without maintainer design (Rule 8).

## Trap round 1 — branch-side verdicts (2026-08-18, , run , fold loss reproduced 24/0)

The [SUMT] census instrumented the marker-capture branch of `checkLocalEncodedMemoryStatic` (the producer of every algebra admission key) and both `productsOfRecursionIds` mint sites, all gated on the target LB. Every marker evaluation ends in exactly one verdict. Branch totals: **608 evaluations → 45 captured, 549 refused by the purity gate, 14 refused by the consumed-key gate.**

**The sum demand is refused red-handed, culprit named.** All five sum-demand instances (`(in3[int_lev_4_2380,{9|it_0_lev_3_78},marker,4])`, startInt 4481 and 5507 — the same proof position where main plants the key) log `pure=0` → `SUMT-IMPURE-skip`. The decisive premises line (seq 8593): four of five premises pure — `(in[9,int_lev_2_0])`, `(in[rec,int_lev_2_0])`, `(in2[rec,9,3])`, `(in2[rec,int_lev_4_2380,int_lev_2_1])` — and exactly ONE impure argument: the id-image premise `(in2[9,it_0_lev_3_78,8])`, impurity `{IMPURE:it_0_lev_3_78}`. The `i2_1(rec)=?` demand (startInt 2325) dies identically, same culprit. The purity rule: an argument with a positive iteration stamp must be in `productsOfRecursionIds`.

**Why `it_0_lev_3_78` is impure here:** the [SUMT-POR-mint] census shows the branch mints exactly {9 (burst 1, recursion site), `it_0_lev_4_94`, `it_0_lev_4_154` (both at the isAdmitted site)} — `it_0_lev_3_78` is NEVER minted at this LB, because the flag-5 relay minted it at the ANCESTOR level-3 LB and it arrived as status-3 mail (`[GAUSST-ADD]` seq 4659/5514). No local admission event ⇒ no productsOfRecursion membership ⇒ every request instance citing it is impure.

**The three-mechanism composition (why no pure instance survives):** (a) the relay ships the id-image fact only in the `it_0_lev_3_78` form; (b) the branch's own pure twin `(in2[9,it_0_lev_4_94,8])` (local, `w4_94` ∈ productsOfRecursion, registered at seq 2122) is no longer offered to requests by the sum-demand window — the class {9, w3_78, w4_94} sweep drops the non-canonical local twin; (c) the canonical ground form `(in2[9,9,8])` never registers on the branch (the prior session's Step-14 "eqfilter swallow" observable — this is where it actually bites). Main, by contrast, holds both pure routes: the ground `(in2[9,9,8])` (burst 5) and `(in2[9,it_0_lev_4_0,8])` with `it_0_lev_4_0` ∈ its productsOfRecursion {9, it_0_lev_4_0, it_0_lev_4_60}.

**Internal control (same burst, same site):** the all-pure `(in2[9,marker,8])` instance (premises `(in[9,i2_0]) (in[rec,i2_0]) (in2[rec,9,3]) (in2[rec,int_lev_4_2380,int_lev_2_1])`) logs `pure=1` → `SUMT-CAPTURE` → `[GAUSST-ALG-insert]` (seq 8591/8609) — the gate discriminates, not the site.

## Trap round 1 — main-side verdicts: the A/B is complete (2026-08-18, , run , fold intact 26/2)

Identical [SUMT] traps. Totals: 1127 marker evaluations → 61 captured, 1052 impure-refused, 14 consumed-refused.

**The sum demand is captured pure, and the premises line names the mechanism.** `(in3[int_lev_4_1960,9,marker,4])` at startInt 4169 (the exact staging window the round-3 census saw drain at burst 23): `pure=1` → `SUMT-CAPTURE` (seq 9780–9782), premises all pure — `(in[it_0_lev_4_0,int_lev_2_0]) (in[rec,int_lev_2_0]) (in2[rec,it_0_lev_4_0,3]) (in2[it_0_lev_4_0,9,8]) (in2[rec,int_lev_4_1960,int_lev_2_1])`. The id-image premise is the **witness form** with `it_0_lev_4_0` — main's own induction-built s(rec) witness, in productsOfRecursion (POR census: {9 at the UAMR site burst 1; `it_0_lev_4_0`, `it_0_lev_4_60` at the isAdmitted site}). The second captured orientation (seq 9792–9794) cites `(in2[9,it_0_lev_4_0,8])` — literally chapter step S39. The ground fact `(in2[9,9,8])` is cited by NO captured request — it plays no role in the causal chain (maintainer challenge confirmed).

**Perfect in-run control:** main REFUSES the same head `pure=0` (seq 9786–9788) when the instance cites `it_0_lev_4_14` — the 0+rec witness, not in main's POR either. The gate behaves identically on both sides; the sole discriminator is whether one pure instance exists.

## The trap-proven root cause (iron-clad, both sides)

Same site, same head, same gate — `checkLocalEncodedMemoryStatic`'s marker-capture purity gate (`pure` = every iteration-bearing premise argument ∈ `productsOfRecursionIds`, where productsOfRecursion = names built by eating through the induction implication's premise cursors):

- **Main:** the id(9) witness `it_0_lev_4_0` is minted locally by the induction progression (admitted against its cursor → POR) → the fold-step marker requests citing it are pure → the sum demand `(in3[Σ,9,marker,4])` is captured, staged, drained (round-3 seq 7220), the parked sum witness revives, **S49 registers** → fold proved.
- **Branch:** the flag-5 relay mints the id(9) witness `it_0_lev_3_78` at the ANCESTOR LB and mails the image fact down (status 3); no local admission event ⇒ not in POR. The branch's own induction-built pure twin `it_0_lev_4_94` (∈ POR) is offered to the registration door but never survives to the request pools (the class {9, w3_78, w4_94} door/sweep handling — the prior session's Step-14 swallow observable is the removal of this pure twin), while the mailed impure form persists through re-delivery. Every sum-demand instance therefore cites `it_0_lev_3_78` → `pure=0` → refused (all five instances, culprit named inline) → no demand → **S49 never registers** → fold lost.

The purity gate operates correctly on both sides; the defect is the **displacement**: a relay-delivered foreign witness-form becomes the sole surviving instance of content whose local twin was a legitimate product of recursion.

**Fix directions discussed with the maintainer (Rule 8 — maintainer to decide; nothing implemented):** (a) classmate-POR enrichment — a name proven equal to a productsOfRecursion member enters the set (write-side at the single-threaded equality/class-merge seam recommended: hot-path probe unchanged, deterministic; watch the demand-key flavor dedup via the existing admission-map rekey hooks); (b) door-side canonicalization (the Step-14 surface) so the class-canonical ground form survives and is pure; (c) relay-side: deliver facts in a form that does not mask the receiver's own induction products. Firing itself is not purity-gated and the parked witness revives at drain time, so (a) alone is expected to carry S49 and the downstream chain.

**Rule 30 inventory:** [SUMT] traps active on BOTH branches (this branch commit, aux branch commit ) — remove in the fix commit together with the [GAUSST] rounds 1–3 per the standing plan. Evidence files: , , , , .

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
