<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Gauss fold theorem — main's successful proof in human notation (bottom to top, no gaps)

**Purpose.** The reference document for the Gauss-fold-loss debugging campaign. It transcribes, step by step and with no gaps, the proof that `main` finds and the branch loses. The companion document [`gauss_fold_step_match.md`](gauss_fold_step_match.md) matches every step here against its branch analogue and pins the first failing step.

**The lost theorem** (`files/theorems/theorems.txt` on main, absent on the branch):

```text
(>[1,2,3,4,5,6,7,8](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10,11](in3[9,10,11,5])(>[12](fold[1,3,4,8,2,9,12])(>[](in2[9,10,3])(in3[7,12,11,5])))))
```

In words: for n with n·s(n) = P and Σ(n) = S and s(n) = n′: **2·S = P** — Gauss summation, n(n+1) = 2·Σ_{i=0..n} i. Proved by induction on n; the induction-condition chapter is the proof that dies on the branch, so this document transcribes exactly that chapter.

**Source of every step:**  (main's raw proof graph, 199 rows). Step numbers S1..S199 are the chapter rows in **reverse file order**, so low numbers sit near the leaves and S199 is the head — the bottom-to-top order of this document is the stage order below, and inside each stage steps are listed in dependency order. Every one of the 199 rows appears exactly once; nothing is skipped.

**Where these statements live at run time:** the fold successor-case induction LB, Rule-12 chain
`root → (AnchorGauss[1,2,3,4,5,6,7,8]) → (in3[9,10,11,5]) → (fold[1,3,4,8,2,9,12]) → (in2[9,10,3]) → (in2[rec0,9,3])` — the target of the sacred hashburst dump on both branches (, 28 bursts; , 32 bursts).

## Reading legend

Ground names at this LB (the induction sub-block of the theorem above):

| name | meaning |
|---|---|
| `1` | N (also `u_1` inside rules) |
| `2` | the numeral 0 (`u_2`) |
| `3` | successor s (`u_3`) |
| `4` | + (`u_4`) |
| `5` | · (`u_5`) |
| `6` | the numeral 1 (`u_6`) |
| `7` | the numeral 2 (`u_7`) |
| `8` | id, the summand function of the fold (`u_8`) — AnchorGauss slot 8 expands to `identity[1,8]` |
| `9` | the successor-case value n (recursion assumption: s(rec) = 9) |
| `10` | s(9) |
| `11` | 9·10 (the premise product P) |
| `12` | Σ(9) (the premise fold value S) |
| `rec` | the induction variable (predecessor of 9) |

Minted names (run-specific): `w<L>_<N>` = `it_0_lev_<L>_<N>` (existential witness), `i<L>_<N>` = `int_lev_<L>_<N>` (integration/skolem name), `r<L>_<N>` = `repl_lev_<L>_<N>` (universal replacement variable in an integration subproof). Atom decodings: `(in[x,1])` = x ∈ N; `(in2[a,b,3])` = s(a) = b; `(in2[a,b,8])` = id(a) = b; `(in2[a,b,F])` = F(a) = b for a minted function name F; `(in3[a,b,c,4])` = a+b = c; `(in3[a,b,c,5])` = a·b = c; `(fold[1,3,4,8,2,n,v])` = Σ(n) = v; `(interval[1,4,2,u,S])` = S = [0..u]; `(sequence[1,4,2,u,f])` = f is a sequence on [0..u]; `(fXY[f,A,B])` = f: A → B.

Key minted names of this proof (main run):

| name | meaning | first burst |
|---|---|---|
| `i2_0` | the interval [0..9] (from unfolding Σ(9)=12) | 2 |
| `i2_1` | the fold sequence on [0..9]: i2_1(0)=id(0), i2_1(s(x))=i2_1(x)+id(s(x)), i2_1(9)=12 | 2 |
| `w0_30` | the predecessor-of-1 witness (equals 0) | 3 |
| `w4_0` | the s(rec) witness (equals 9) | 3 |
| `w4_46` | the rec+0 witness (equals rec) | 4 |
| `w4_14` | the 0+rec witness (equals rec) | 7 |
| `w4_60` | the rec·9 product witness | 3 |
| `w4_378` | the 9·2 product witness | 5 |
| `i4_1741` | the interval [0..rec] | 11 |
| `i4_1925` | the limit sequence on [0..rec] (restriction of i2_1) | 19 |
| `i4_1960` | the value i4_1925(rec) = i2_1(rec) = Σ(rec) | 21 |
| `w4_3091` | the sum witness Σ(rec) + 9 — **the witness the branch never mints** | 23 |

## Proof idea in one paragraph

Unfold the premise Σ(9) = 12 into its defining data: an interval [0..9] (`i2_0`) and a sequence `i2_1` with i2_1(0) = id(0), the step law i2_1(s(x)) = i2_1(x) + id(s(x)), and the final value i2_1(9) = 12. Build the predecessor's fold: the interval [0..rec] (`i4_1741`) and the limit sequence `i4_1925` (the restriction of i2_1), prove the two fold-defining laws for it (two integration subproofs), read off its final value i4_1925(rec) = `i4_1960`, and integrate everything back into the statement Σ(rec) = i4_1960. Now the induction hypothesis fires: 2·i4_1960 = rec·9. Separately, derive the sum fact i4_1960 + 9 = `w4_3091` (+-totality), fire the step law of `i2_1` once at position 9 to get i2_1(9) = w4_3091, and conclude w4_3091 = 12 by function-uniqueness against i2_1(9) = 12 — that is Σ(9) = Σ(rec) + s(rec). The head 2·12 = 11 then follows by the additivity-of-multiplication composition from 2·Σ(rec) = rec·9, 2·s(rec) = 2·9, Σ(rec)+s(rec) = 12, and rec·9 + 9·2 = 9·10 = 11.

---

## Stage A — the givens of the induction-condition block

- **S193** [task formulation] `(AnchorGauss[1,2,3,4,5,6,7,8])` — the anchor.
- **S2** [task formulation] s(9) = 10 — premise.
- **S9** [task formulation] 9·10 = 11 — premise.
- **S163** [task formulation] Σ(9) = 12 — premise.
- **S170** [recursion] s(rec) = 9 — the successor-case assumption.
- **S190** [recursion] the induction hypothesis as a rule: ∀10′,11′,12′: rec·10′ = 11′ ∧ Σ(rec) = 12′ ∧ s(rec) = 10′ ⟹ 2·12′ = 11′.

## Stage B — anchor expansion, corpus theorems

- **S194** [expansion ← S193] AnchorGauss = NaturalNumbers(N,0,s,+,·) ∧ s(0)=1 ∧ s(1)=2 ∧ 8 = id on N.
- **S195** [disintegration ← S194] `NaturalNumbers[1,2,3,4,5]`.
- **S192** [disintegration ← S194] s(0) = 1.
- **S139** [disintegration ← S194] s(1) = 2.
- **S152** [disintegration ← S194] 8 = id on N.
- **S196** [expansion for integration] the AnchorPeano integration goal (NaturalNumbers ∧ s(0)=1).
- **S197** [reformulation for integration ← S196] the AnchorPeano constructor rule.
- **S198** [implication ← S197, S195, S192] `AnchorPeano[1,0,3,4,5,1]` — the Peano anchor reconstructed inside the Gauss run (slot 6 carries 1), unlocking the Peano theorem corpus (cited as EXTERN below).
- **S28** [anchor handling ← S193] the x-marked anchor variant `AnchorGauss[1,x2,...]` used by anchor-parameter theorems.
- **S29** [theorem] +-commutativity: a+b = c ⟹ b+a = c.
- **S89** [theorem] f: [0..u] → N is a sequence on [0..u] (function + interval ⟹ sequence).
- **S128** [theorem] u ∈ [0..u] (upper endpoint membership).
- **S130** [theorem] s(x) = y ∧ I = [0..y] ⟹ the interval [0..x] exists (`existence8`).
- **S168** [theorem] f a sequence on [0..u] from f: I → N with I = [0..u].
- **S171** [theorem] s(x) = y ∧ f a sequence on [0..y] ⟹ the limit sequence of f at x exists (`existence15`).

## Stage C — the Peano corpus compacts unfold into rules

- **S148** [expansion ← S195] the 13-conjunct Peano compact: 0∈N ∧ s:N→N ∧ implication4/5 ∧ fXYZ[+] ∧ implication11/12/13/14 ∧ fXYZ[·] ∧ implication15/16/17.
- **S138** [disintegration ← S148] 0 ∈ N.
- **S140** [disintegration ← S148] s: N → N. **S141** [expansion ← S140] its function laws implication0..3. **S142**+**S143** [dis+exp] implication0: s(x)=y ⟹ x∈N. **S52**+**S53** [dis+exp] implication2: x∈N ⟹ ∃v s(x)=v (s-totality). **S57**+**S58** [dis+exp] implication3: s(x)=y ∧ s(x)=z ⟹ y=z (s-uniqueness). **S60**+**S61** [dis+exp] implication1: s(x)=y ⟹ y∈N (s-closure).
- **S149**+**S150** [dis+exp ← S148] implication5: s(y)=x ∧ s(z)=x ⟹ y=z (predecessor uniqueness).
- **S19**+**S20** [dis+exp ← S148] implication11: x+0=y ⟹ x=y.
- **S25**+**S26** [dis+exp ← S148] implication12: x=y ⟹ x+0=y.
- **S36**+**S37** [dis+exp ← S148] implication13: s(x)=y ∧ a+x=b ∧ a+y=c ⟹ s(b)=c (addition-successor, forward).
- **S3**+**S4** [dis+exp ← S148] implication14: s(x)=y ∧ a+x=b ∧ s(b)=c ⟹ a+y=c (addition-successor, other direction).
- **S70**+**S71**+**S72**+**S73** [dis+exp chain ← S148] fXYZ[·]: · is a function; implication9 gives ·-totality: x∈N ∧ y∈N ⟹ ∃v x·y=v (`existence1`).
- **S80**+**S81**+**S82**+**S83** [dis+exp chain ← S148] fXYZ[+]: likewise +-totality: x∈N ∧ y∈N ⟹ ∃v x+y=v.
- **S153** [expansion ← S152] the id laws compact. **S123**+**S124** [dis+exp] implication18: id(x)=y ⟹ x=y. **S154**+**S155** [dis+exp] implication19: x∈N ∧ x=y ⟹ id(x)=y.

## Stage D — ground numerals

- **S144** [← S143, S139] 1 ∈ N (from s(1)=2).
- **S62** [← S61, S139] 2 ∈ N.
- **S145** [← EXTERN Peano theorem, S198] ∃x: s(x)=1 (`existence3`). **S146** [expansion] the witness binder. **S147** [disintegration] s(w0_30) = 1.
- **S151** [← S150, S192, S147, S144] 0 = w0_30 (predecessor uniqueness on s(0)=1 and s(w0_30)=1). **S126** [same premises] w0_30 = 0 (mirror).
- **S156** [← S155, S151, S138] id(0) = w0_30.
- **S69** [disintegration ← S68] 9 ∈ N (from the [0..9] interval conjunction, Stage E).
- **S79** [← S143, S170] rec ∈ N (from s(rec)=9).

## Stage E — the fold premise Σ(9) = 12 unfolds

- **S164** [expansion ← S163] the interval existence binder: ¬∀I: I=[0..9] ⟹ ¬(fold data over I).
- **S162** [disintegration ← S164] **i2_0 = [0..9]**.
- **S68** [expansion ← S162] the interval's defining conjunction: implication20 (0 ≤ every member... in the [0..9] reading: membership bounds), implication21 (x∈i2_0 ⟹ x ≤ 9), implication22, plus 0∈N and 9∈N.
- **S165** [disintegration ← S164] the `existence2` fold-data compact over i2_0. **S166** [expansion] its sequence binder.
- **S167** [disintegration ← S166] **i2_1: i2_0 → N** — the fold sequence exists.
- **S157** [disintegration ← S166] the fold-data `and0` compact. **S158** [expansion ← S157] its three conjuncts: implication24[0,8,i2_1] ∧ i2_1(9)=12 ∧ implication25[i2_0,3,i2_1,8,4].
- **S41** [disintegration ← S158] **i2_1(9) = 12** — the final-value fact.
- **S159**+**S160** [dis+exp ← S158] implication24: id(0)=v ⟹ i2_1(0)=v — the base law.
- **S116**+**S117** [dis+exp ← S158] implication25 — **the fold-step law**: x∈i2_0 ∧ s(x)=y ∧ y∈i2_0 ∧ i2_1(x)=a ∧ id(y)=b ∧ a+b=c ⟹ i2_1(y)=c.
- **S42** [expansion ← S167] i2_1's function laws implication0..3. **S13**+**S14** [dis+exp] implication0: i2_1(x)=y ⟹ x∈i2_0. **S43**+**S44** [dis+exp] implication3: i2_1(x)=y ∧ i2_1(x)=z ⟹ y=z (**i2_1-uniqueness**).
- **S15** [← S14, S41] **9 ∈ i2_0** (from i2_1(9)=12).
- **S161** [← S160, S156] **i2_1(0) = w0_30** — the base value.

## Stage F — successor-case witness arithmetic

- **S54** [← S53 s-totality, S79] ∃v: s(rec)=v. **S55** [expansion] binder. **S56** [disintegration] **s(rec) = w4_0**. **S23** [← S55] w4_0 ∈ N.
- **S59** [← S58 s-uniqueness, S170, S56, S79] **9 = w4_0**. **S24** [same premises] w4_0 = 9.
- **S32** [← S83 +-totality, S138, S79] ∃v: rec+0=v. **S33** [expansion] binder. **S34** [disintegration] **rec+0 = w4_46**. **S18** [← S33] w4_46 ∈ N.
- **S21** [← S20, S34, S79] **rec = w4_46**.
- **S84** [← S83, S138, S79] ∃v: 0+rec=v. **S85** [expansion] binder. **S86** [disintegration] **0+rec = w4_14**. **S87** [← EXTERN 0+x=x theorem, S198, S86, S79] **rec = w4_14**.
- **S22** [equality1: S86 rewritten under S21] 0+w4_46 = w4_14.
- **S31** [← EXTERN x+1=s(x) theorem, S198, S56] rec+1 = w4_0.
- **S27** [← S26, S24, S69, S23] w4_0+0 = 9. **S30** [← S29 +-commutativity, S28, S27] 0+w4_0 = 9.
- **S35** [← S37 implication13, S192, S34, S31, S138] **s(w4_46) = w4_0**.
- **S38** [← S37, S35, S30, S22, S18] **s(w4_14) = 9**.
- **S1** [← EXTERN x+1=s(x), S198, S170] rec+1 = 9. **S5** [← S4 implication14, S139, S2, S1, S144] rec+2 = 10. **S6** [equality1: S5 under S21] w4_46+2 = 10.
- **S39** [← S155 id-rule, S59, S69] **id(9) = w4_0**.
- **S74** [← S73 ·-totality, S69, S79] ∃v: rec·9=v. **S75** [expansion] binder. **S76** [disintegration] **rec·9 = w4_60**.
- **S7** [← EXTERN ·-commutativity, S198, S76] 9·rec = w4_60. **S8** [equality1: S7 under S21] 9·w4_46 = w4_60.
- **S63** [← S73, S62, S69] ∃v: 9·2=v. **S64** [expansion] binder. **S65** [disintegration] **9·2 = w4_378**.
- **S66** [equality1: S65 under S59] w4_0·2 = w4_378. **S67** [← EXTERN ·-commutativity, S198, S66] **2·w4_0 = w4_378**.
- **S10** [← EXTERN left-distributivity composition (a·x=p ∧ a·y=q ∧ x+y=z ∧ a·z=r ⟹ p+q=r), S198, S9, S65, S8, S6] **w4_60 + w4_378 = 11** — the decomposition 9·10 = 9·rec + 9·2.

## Stage G — the predecessor interval [0..rec]

- **S131** [← theorem S130, S193, S170, S162] `existence8`: the interval below rec exists. **S132** [expansion] binder. **S133** [disintegration] **i4_1741 = [0..rec]**.
- **S103** [disintegration ← S132] the `limitSet` compact. **S104** [expansion] implication29 ∧ implication21 ∧ implication30.
- **S105**+**S106** [dis+exp ← S104] implication29: x∈i4_1741 ⟹ x∈i2_0 (**sub-interval inclusion**).
- **S97**+**S98** [dis+exp ← S104] implication21: x∈i4_1741 ⟹ x ≤ rec.
- **S134** [expansion ← S133] the [0..rec] interval conjunction. **S135**+**S136** [dis+exp] implication20: x∈i4_1741 ⟹ 0 ≤ x.
- **S129** [← theorem S128, S193, S133] **rec ∈ i4_1741**. **S137** [← S136, S129] 0 ≤ rec.
- **S88** [equality1: S133 under S87] i4_1741 = [0..w4_14]. **S11** [← S128, S193, S88] w4_14 ∈ i4_1741. **S12** [← S106, S11] **w4_14 ∈ i2_0**.

## Stage H — the limit sequence i4_1925 and its value at rec

- **S169** [← theorem S168, S193, S167, S162] i2_1 is a sequence on [0..9].
- **S172** [← theorem S171, S193, S170, S169] `existence15`: the limit sequence of i2_1 at rec exists. **S173** [expansion] binder. **S174** [disintegration] the `limitSequence` compact for **i4_1925**.
- **S175** [expansion ← S174] implication26 ∧ implication27 ∧ implication28.
- **S111**+**S112** [dis+exp ← S175] implication27: i4_1925(x)=y ⟹ i2_1(x)=y (**agreement, limit → base**).
- **S176**+**S177** [dis+exp ← S175] implication28: x ≤ rec ∧ i2_1(x)=y ⟹ i4_1925(x)=y (**agreement, base → limit**).
- **S77** [disintegration ← S173] i4_1925 is a sequence on [0..rec]. **S78** [equality1: S77 under S87] on [0..w4_14].
- **S90** [← theorem S89, S193, S88, S78] **i4_1925: i4_1741 → N**.
- **S91** [expansion ← S90] i4_1925's function laws. **S92**+**S93** [dis+exp] implication2: x∈i4_1741 ⟹ ∃v i4_1925(x)=v (totality).
- **S94** [← S93, S129] ∃v: i4_1925(rec)=v. **S95** [expansion] binder. **S96** [disintegration] **i4_1925(rec) = i4_1960**. **S46** [← S95] i4_1960 ∈ N.
- **S16** [equality1: S96 under S87] i4_1925(w4_14) = i4_1960. **S17** [← S112 agreement, S16] **i2_1(w4_14) = i4_1960** — Σ(rec) as a value of the base sequence.

## Stage I — integration subproof 1: i4_1925 starts at id(0) (implication24[0,8,i4_1925])

Boundary scope `main_boundary_(implication24[0,8,i4_1925])`; goal from **S121** [expansion for integration]: ∀r4_0: id(0)=r4_0 ⟹ i4_1925(0)=r4_0.

- **S122** [premise element ← S121] id(0) = r4_0 (assumed).
- **S125** [← S124 id-uniqueness rule, S122] 0 = r4_0.
- **S127** [equality2 ← S126, S125] w0_30 = r4_0.
- **S178** [← S177 agreement, S161, S137] i4_1925(0) = w0_30 (at main scope).
- **S179** [equality1: S178 under S127] i4_1925(0) = r4_0 — goal reached.
- **S180** [validity name] **implication24[0,8,i4_1925] proven**.

## Stage J — integration subproof 2: i4_1925 satisfies the fold step (implication25[i4_1741,3,i4_1925,8,4])

Boundary scope `main_boundary_(implication25[i4_1741,3,i4_1925,8,4])`; goal from **S114** [expansion for integration]: ∀r4_6,r4_7,r4_8,r4_9,r4_10: r4_6∈i4_1741 ∧ s(r4_6)=r4_7 ∧ r4_7∈i4_1741 ∧ i4_1925(r4_6)=r4_8 ∧ id(r4_7)=r4_9 ∧ r4_8+r4_9=r4_10 ⟹ i4_1925(r4_7)=r4_10.

- **S102**, **S115**, **S100**, **S110**, **S109**, **S108** [premise elements ← S114] the six assumptions above, in order.
- **S107** [← S106 inclusion, S102] r4_6 ∈ i2_0. **S101** [← S106, S100] r4_7 ∈ i2_0.
- **S99** [← S98, S100] r4_7 ≤ rec.
- **S113** [← S112 agreement, S110] i2_1(r4_6) = r4_8.
- **S118** [← S117 **the i2_1 fold-step law**, S115, S113, S109, S108, S107, S101] i2_1(r4_7) = r4_10.
- **S119** [← S177 agreement, S118, S99] i4_1925(r4_7) = r4_10 — goal reached.
- **S120** [validity name] **implication25[i4_1741,3,i4_1925,8,4] proven**.

## Stage K — fold integration: Σ(rec) = i4_1960

- **S181** [expansion for integration] the `and0` goal: implication24[0,8,i4_1925] ∧ i4_1925(rec)=i4_1960 ∧ implication25[i4_1741,3,i4_1925,8,4]. **S182** [reformulation] its constructor.
- **S183** [← S182, S180, S120, S96] **and0[0,8,i4_1925,rec,i4_1960,i4_1741,3,4]** — the fold data for rec assembled.
- **S184** [expansion for integration] the `existence2` goal (∃ sequence with that data). **S185** [reformulation] constructor. **S186** [← S185, S183, S90] **existence2[i4_1741,1,0,8,rec,i4_1960,3,4]**.
- **S187** [expansion for integration] the fold goal (∃ interval with that existence2). **S188** [reformulation] constructor. **S189** [← S188, S186, S133] **Σ(rec) = i4_1960**.

## Stage L — the induction hypothesis fires

- **S191** [← S190 IH, S189, S170, S76] **2·i4_1960 = w4_60** — twice the predecessor sum equals rec·9.

## Stage M — the sum bridge: Σ(9) = Σ(rec) + s(rec)

- **S47** [← S83 +-totality, S69, S46] ∃v: i4_1960+9 = v.
- **S48** [expansion ← S47] the witness binder ¬∀w4_3091: … — **mints w4_3091**.
- **S49** [disintegration ← S48] **i4_1960 + 9 = w4_3091**.
- **S50** [equality1: S49 under S59] **i4_1960 + w4_0 = w4_3091**.
- **S40** [← S117 **the fold-step law fires at position 9**: S12 (w4_14∈i2_0), S38 (s(w4_14)=9), S15 (9∈i2_0), S17 (i2_1(w4_14)=i4_1960), S39 (id(9)=w4_0), S50 (i4_1960+w4_0=w4_3091)] **i2_1(9) = w4_3091**.
- **S45** [← S44 i2_1-uniqueness, S41 (i2_1(9)=12), S40, S15] **w4_3091 = 12**.
- **S51** [equality1: S50 under S45] **i4_1960 + w4_0 = 12** — Σ(rec) + s(rec) = Σ(9).

## Stage N — the head

- **S199** [← EXTERN additivity-of-multiplication composition (a·x=p ∧ a·y=q ∧ x+y=z ∧ p+q=r ⟹ a·z=r), S198, S191 (2·i4_1960=w4_60), S67 (2·w4_0=w4_378), S51 (i4_1960+w4_0=12), S10 (w4_60+w4_378=11)] **2·12 = 11** — the head. ∎

---

## Appendix — the machine-generated full listing

A mechanically decoded listing of all 199 rows with per-step dependency references (same S-numbers) was generated from the chapter file and used to build the stages above; regenerate at will with the checked-in converter (`/c/Users/nikol/anaconda3/python.exe .scripts/chapter_to_human.py <chapter file>` — its siblings `.scripts/hashburst_timeline.py` and `.scripts/hashburst_section.py` extract statement first-appearance timelines and single dump sections from a hashburst trace; the three together are the proof-trace comparison toolkit this campaign used). The stages above are the human-curated form and take precedence where wording differs.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
