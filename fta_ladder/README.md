<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

# FTA ladder — rung index

> Rung-by-rung path toward the Fundamental Theorem of Arithmetic. Each rung is a self-contained subfolder with a human-notation proof file (`proof_NN_<slug>.md`) and an append-only `current_proof_state.md` that records the GL prover's progress against that rung. When a rung's directed conjectures all close, the rung is marked **closed** here and the in-flight subfolder freezes; subsequent prover work moves to the next rung.

The ladder exists because FTA is too large a step from Peano + Gauss in a single jump. Each rung exercises one new piece of OR / case-split / induction machinery, validates it on a smaller theorem, and leaves the infrastructure available for the next rung. By the time the ladder reaches FTA, every required mechanism has been exercised at least once.

---

## Status table

| Rung | Theorem (human) | Forward direction | Reverse direction | Folder |
|---|---|---|---|---|
| 1 | `{0, 1} = [0, 1]` (set-equality of enumerated and interval forms) | ✓ closed (D-34, sandbox/ordis_merge) | ⏳ open — next rung-1 frontier | [`rung1/`](rung1/) |

Counts: **1 rung in progress** (one direction closed, one open), **0 rungs fully closed**, FTA not yet reached.

---

## Rung 1 — `{0, 1} = [0, 1]`

### Human formulation

Working inside a Peano structure `(N, 0, 1, s, +)` where
- `N` is the set of natural numbers,
- `0 ∈ N`, `1 := s(0) ∈ N`,
- `s : N → N` is the successor,
- `+ : N × N → N` is the addition induced by `s`,

with the order defined additively as `a ≤ b :⇔ ∃ k ∈ N. a + k = b`, the two-element enumerated set and the closed interval coincide:

```
{0, 1}  =  [0, 1]
```

where:

- **Left (enumerated set)**  `{0, 1} := { x ∈ N : x = 0 ∨ x = 1 }`
- **Right (closed interval)**  `[0, 1] := { p ∈ N : 0 ≤ p ≤ 1 }`

The two sides denote the same subset of `N`. Full human-notation proof in [`rung1/proof_01_set_eq_interval.md`](rung1/proof_01_set_eq_interval.md).

### MPL formulation

GL stores the set equality as two **directed** conjectures — set equality decomposes into two implications, each proved separately. With the `AnchorIncubator` 14-arg anchor binding (positions: `1=N, 2=0, 3=s, 4=+, 5=*, 6=1, 7=2, 8=id, 9=Z, 10=…, 11=…, 12=…, 13=…, 14=…`), the two conjectures are:

**Forward — `EnumerationSet2 ⟹ interval` ✓ (proved on sandbox/ordis_merge under D-34):**

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

- [`docs/40_decisions.md`](../docs/40_decisions.md) — D-22 onward documents the architectural decisions that the FTA ladder forced.
- [`docs/20_core_concepts/07_or_branching.md`](../docs/20_core_concepts/07_or_branching.md) — OR-branching machinery exercises every rung.
- [`docs/30_invariants.md`](../docs/30_invariants.md) — load-bearing invariants the rungs validate.
- `MEMORY.md project_fta_ladder.md` — auto-memory pointer to this directory.
- `MEMORY.md reference_fta_current_proof_state.md` — standing entry-point reference.
