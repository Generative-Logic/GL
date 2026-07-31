<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# FTA ladder — rung index

> Rung-by-rung path toward the Fundamental Theorem of Arithmetic. Each rung is a self-contained subfolder with a human-notation proof file (`proof_NN_<slug>.md`) and an append-only `current_proof_state.md` that records the GL prover's progress against that rung. When a rung's directed conjectures all close, the rung is marked **closed** here and the in-flight subfolder freezes; subsequent prover work moves to the next rung.

The ladder exists because FTA is too large a step from Peano + Gauss in a single jump. Each rung exercises one new piece of OR / case-split / induction machinery, validates it on a smaller theorem, and leaves the infrastructure available for the next rung. By the time the ladder reaches FTA, every required mechanism has been exercised at least once.

---

## Status table

| Rung | Theorem (human) | Forward direction | Reverse direction | Folder |
|---|---|---|---|---|
| 1 | `{0, 1} = [0, 1]` (set-equality of enumerated and interval forms) | ✓ closed (D-34) | ⏳ open — next rung-1 frontier | [`rung1/`](rung1/) |
| 2 | `{0, 1, 2} = [0, 2]` (three-way set equality) | ✓ closed — flat atomic `_ordis_` | ✓ closed 2026-07-25 — `_orint_`-scoped secondary budget | [`rung2/`](rung2/) |
| 2.1 | `{0, 1, 2} ≠ [0, 1]` (disproof of the false cross-pair) | ⏳ open — reductio, not started | — (single negation theorem) | [`rung2_1/`](rung2_1/) |

Counts: **2 rungs in progress** (1, 2.1), **1 rung fully closed** (2), FTA not yet reached.

---

## Rung 1 — `{0, 1} = [0, 1]`

### Human formulation

Working inside a Peano structure `(N, 0, 1, s, +)` where
- `N` is the set of natural numbers,
- `0 ∈ N`, `1:= s(0) ∈ N`,
- `s: N → N` is the successor,
- `+: N × N → N` is the addition induced by `s`,

with the order defined additively as `a ≤ b:⇔ ∃ k ∈ N. a + k = b`, the two-element enumerated set and the closed interval coincide:

```
{0, 1}  =  [0, 1]
```

where:

- **Left (enumerated set)** `{0, 1}:= { x ∈ N: x = 0 ∨ x = 1 }`
- **Right (closed interval)** `[0, 1]:= { p ∈ N: 0 ≤ p ≤ 1 }`

The two sides denote the same subset of `N`. Full human-notation proof in [`rung1/proof_01_set_eq_interval.md`](rung1/proof_01_set_eq_interval.md).

### MPL formulation

GL stores the set equality as two **directed** conjectures — set equality decomposes into two implications, each proved separately. With the `AnchorIncubator` 14-arg anchor binding (positions: `1=N, 2=0, 3=s, 4=+, 5=*, 6=1, 7=2, 8=id, 9=Z, 10=…, 11=…, 12=…, 13=…, 14=…`), the two conjectures are:

**Forward — `EnumerationSet2 ⟹ interval` ✓ (proved under D-34):**

```text
(>[1,2,4,6](AnchorIncubator[1,2,3,4,5,6,7,8,9,10,11,12,13,14])
   (>[15](EnumerationSet2[2,6,15])(interval[1,4,2,6,15])))
```

Reading: *"For any `N`, `0`, `+`, `1` (slots 1, 2, 4, 6 of AnchorIncubator), and any set `15` (= `M`), if `M` is the enumerated `{0, 1}` then `M` is the interval `[0, 1]`."*

**Reverse — `interval ⟹ EnumerationSet2` ⏳ (open):**

```text
(>[1,2,4,6](AnchorIncubator[1,2,3,4,5,6,7,8,9,10,11,12,13,14])
   (>[15](interval[1,4,2,6,15])(EnumerationSet2[2,6,15])))
```

Reading: *"…if `M` is the interval `[0, 1]` then `M` is the enumerated `{0, 1}`."*

Both directed conjectures live in `files/theorems/theorems.txt` (lines 340 and 342 in current MPL ordering); the forward direction lands in `files/incubator/theorems/proved_theorems.txt` after a successful pipeline run.

### State

- **Forward direction closed** — saved in incubator's `proved_theorems.txt`. End-to-end closure required D-29 (disintegration gate), D-30 (OR-integration emit scope), D-32 (sharper OR-disint admission), D-33 (bidirectional eq-class application + single-channel discharge), and D-34 (kernel-level `ordisMerge` + mailOut main-only gate). See [`rung1/current_proof_state.md`](rung1/current_proof_state.md) Steps 1–10 for the layered investigation trail.

---

## Rung 2 — `{0, 1, 2} = [0, 2]`

### Human formulation

The second rung extends the enumerated set by one successor and requires one
additional predecessor split in the reverse proof:

```text
{0, 1, 2}  =  [0, 2]
```

The full proof is in
[`rung2/proof_02_set_eq_interval.md`](rung2/proof_02_set_eq_interval.md).

### MPL formulation

With `AnchorIncubator3` binding
`1=N, 2=0, 3=s, 4=+, 5=*, 6=1, 7=2, 8=id, 9=3`, the active positive
conjecture is:

```text
(>[1,2,3,4,5,6,7,8,9]
  (AnchorIncubator3[1,2,3,4,5,6,7,8,9])
  (>[10](EnumerationSet3[2,6,7,10])(interval[1,4,2,7,10])))
```

### State

- **CLOSED 2026-07-25** — the theorem proved end-to-end (both containments
 of the set equality ride the one `interval` conjecture). The forward
 clauses close through flat atomic `_ordis_` cohorts; the reverse clause
 closes through flat `_orint_` integration, the restored predecessor OR
 theorem, the injectivity contrapositive, K mutual-exclusion implications
 emitted at every OR depth, and the `_orint_`-scoped distinct-secondary
 budget (`maxNumberSecondaryVariablesOrint`). Verifier airtight
 (133,523 checks, zero failures), runtime under the pre-campaign baseline.

The complete chronological evidence is in
[`rung2/current_proof_state.md`](rung2/current_proof_state.md).

---

## Rung 2.1 — `{0, 1, 2} ≠ [0, 1]`

### Human formulation

The disproof rung split out of rung 2 at its closure: the IncubatorGauss3
batch's last unsettled conjecture is the false cross-pair asserting that
the three-element enumerated set equals the two-element interval. GL must
prove the negation by reductio: assuming the interval description forces
`2 ≤ 1`, and two rounds of successor-addition descent turn the witnessed
sum `2 + k = 1` into a successor with value `0`. Full disproof in
[`rung2_1/proof_02_1_es3_not_interval_0_1.md`](rung2_1/proof_02_1_es3_not_interval_0_1.md).

### MPL formulation

With the `AnchorIncubator3` binding `1=N, 2=0, 3=s, 4=+, 5=*, 6=1, 7=2,
8=id, 9=3`, the target negation theorem is:

```text
(>[1,2,3,4,5,6,7,8,9]
  (AnchorIncubator3[1,2,3,4,5,6,7,8,9])
  (>[10](EnumerationSet3[2,6,7,10])!(interval[1,4,2,6,10])))
```

### State

- **Not started** — created at the rung-2 closure; prover work begins in a
 new session. Anticipated first boundary: the reductio's descent runs at
 the contradiction LB's `main` scope, outside the `_orint_` widening's
 scope condition. See
 [`rung2_1/current_proof_state.md`](rung2_1/current_proof_state.md).

---

## How rungs are added

When the maintainer plans the next rung:

1. Create `rung<N>/` subfolder.
2. Drop `proof_<NN>_<slug>.md` with the human-notation proof (Theorem statement, Background facts, §-numbered proof, Conclusion, Structure of the proof GL needs to reconstruct).
3. Drop `current_proof_state.md` with the analysis-guide skeleton (Reading convention, Compiled-name glossary, then Steps appended as the prover progresses).
4. Add a row to the Status table above with the human + MPL formulation.

When all directed conjectures of a rung close:

1. Update its row to **closed**.
2. Optionally rename `current_proof_state.md` → `<theorem>_theorem_proof.md` per the lifecycle note in that file's header — preserves the diagnostic trail without contaminating the next rung's analysis.

---

## See also

- [`docs/agentic_swdd/40_decisions.md`](../agentic_swdd/40_decisions.md) — D-22 onward documents the architectural decisions that the FTA ladder forced.
- [`docs/agentic_swdd/20_core_concepts/07_or_branching.md`](../agentic_swdd/20_core_concepts/07_or_branching.md) — OR-branching machinery exercises every rung.
- [`docs/agentic_swdd/30_invariants.md`](../agentic_swdd/30_invariants.md) — load-bearing invariants the rungs validate.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
