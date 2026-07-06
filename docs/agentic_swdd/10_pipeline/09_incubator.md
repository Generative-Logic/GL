<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Pipeline · Incubator `[DRAFT]`

> **Input:** `files/config/ConfigIncubator<Tag>.json` + MPL definitions.
> **Output:** `files/theorems_incubator/theorems.txt` + `files/theorems_incubator/*` (separate theorem tree). Downstream consumers use the output to (re)populate `files/simple_facts/*.txt` tables.
> **Owner:** `run_modes.py` → `incubator_run`; shares prover and conjecturer code with the main pipeline.

---

## What the incubator is

A parallel pipeline that runs the same machinery as the main pipeline, but over a small *finite* anchor — `AnchorIncubator` with 14 arguments, 9 of which pin the constants `0..8`:

```text
(AnchorIncubator[N,i0,s,+,*,i1,i2,id,i3,i4,i5,i6,i7,i8])
```

Because the model is finite (the constants are fully enumerated), the incubator's purpose is not theorem discovery in the main-pipeline sense — it's *autonomous ground-level arithmetic* inside that model. Every proof it produces is a statement that is either true (provable) or false (disprovable via contradiction) about the `{0, 1, 2, 3, 4, 5, 6, 7, 8}` finite model.

The incubator's output then becomes the CE filter's input (`files/simple_facts/simple_facts_peano_<n>.txt`) — the incubator is the producer of the fact tables the main pipeline consumes.

Current status (from `MEMORY.md project_incubator_status.md`):

- 345 / 483 conjectures proved/disproved (71%).
- 47 positives, 298 negatives.
- 0 missing positives.
- 138 missing negatives — all require intermediate values outside the model range; fix is to expand the anchor with more constants.

---

## How the incubator differs from the main pipeline

Same `gl_quick.exe` binary, same prover, but with these flag changes via `ConfigIncubator<Tag>.json`:

| Parameter | Incubator (default) | IncubatorGauss3 (rung-1, AI3 — was IncubatorGauss1 before the anchor split) | Main |
|---|---|---|---|
| `incubator_mode` | `true` | `true` | `false` |
| `try_contradiction` | `true` | `true` | `false` |
| `skip_ce_filter` | `true` (incubator *produces* the facts — it does not consume them) | `true` | `false` |
| `ban_disintegration` | `true` (no Pass B / no back-reformulation — ground-level facts don't need them) | `false` (needs Pass B for ES2 / interval body disintegration) | `false` |
| `allow_multiplication` | `true` (multiplyImpl runs in incubator) | `false` (§4.1 must not multiply) | `false` (multiplyImpl off in plain main) |
| `theorems_folder` | `files/incubator/theorems` | `files/incubator/theorems` (shared) | `files/theorems` |
| `background_theorems_folder` | may point to a main theorems folder to provide context | same | `files/theorems` |
| `compressor` | skipped (output is intentionally huge — every harvested fact is needed downstream by the main batch's CE filter; compression would lose information) | skipped | runs after prover (see [`05_compressor.md`](05_compressor.md)) |

Defined at [`parameters.hpp–63`](../../GL_Quick_VS/GL_Quick/src/parameters.hpp):

```cpp
bool try_contradiction = false;
bool skip_ce_filter = false;
bool incubator_mode = false;
```

### `try_contradiction`

Enables contradiction-attempt LBs for negative conjectures. For each conjecture, the prover additionally spawns a contradiction-seeking LB that tries to derive `!conclusion` — if it succeeds, the conjecture is marked false.

This is how the incubator proves **negative** statements. On the main pipeline, negative statements come out of the proof-by-contradiction path only if the conjecturer emits them in that shape; the incubator routinely enumerates both signs and proves whichever holds.

### `skip_ce_filter`

Set to `true` for incubator batches because the incubator is the facts' producer, not consumer. The CE-filtering pass is bypassed entirely.

### `incubator_mode`

Used to be the master flag. As of 2026-04-29 it no longer gates Pass B (that's `!ban_disintegration` — see [I-7](../30_invariants.md#i-7)) and no longer gates `multiplyImplication` (that's `allow_multiplication`). What it still governs: the `head-in-wholeExpressions` check path inside `addTheoremToMemory`, the compressor skip in `run_modes::fullRun`, the operator-head reformulation in integration ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)), the relaxed-tempArgs assert at [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp), and several conjecturer behaviors (repeat-arg uniqueness, op-head reformulation, OR-generation suppression).

`ban_disintegration` now governs every disintegration-shaped path in the prover: Pass B at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp), back-reformulation at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp), hypothetical disintegration at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp), and necessity-for-equality-hypo at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp). The collapse from a brief two-flag split (`ban_disintegration` + `allow_disintegration`) is documented in [D-28](../40_decisions.md#d-28).

---

## AnchorIncubator — the 14-slot anchor

Defined at [`files/definitions/AnchorIncubator.mpl`](../../files/definitions/AnchorIncubator.mpl). Shape:

```text
(&
    (NaturalNumbers[N,i0,s,+,*])
    (&
        (in2[i0,i1,s])
        (&
            (in2[i1,i2,s])
            ...
            (in2[i7,i8,s])
        )
    )
)
```

The 14 arguments: `[N, i0, s, +, *, i1, i2, id, i3, i4, i5, i6, i7, i8]`. The first 5 are the standard Peano slots; `i1..i8` pin the constants `1..8`; `id` is the identity function (used by Gauss-style incubator variants).

Of the 14 arguments, 9 are `(1)`-typed (the constants `i0..i8`), 1 is `P(1)` (`N`), 2 are `P(x(1)(1))` (`s`, `id`), 2 are `P(x(1)(x(1)(1)))` (`+`, `*`).

The `(1)`-typed args are the ones `multiplyImplication` considers for Bell partitions, and the ones `createMapAnchor` computes permutation tables for. With 9 such args, the `leftMax = 9` in `createMapAnchor`, and the memory footprint is governed by `right = max_values_for_uncomb_def_sets + max_values_for_def_sets`. **`right > 3` causes RAM explosion** — see [Conjecturer](#conjecturer-config-differences) below.

---

## ConfigIncubatorPeano.json — config differences

Anchor name is `AnchorIncubator`. Per-expression `max_count_per_conjecture` is typically `0` for predicates like `in` (meaning: no cap, or a different interpretation for incubator's exhaustive enumeration). `max_size_expression` replaces the main pipeline's `max_size_expression_before_existence` / `max_size_expression_after_existence` split.

Key fields for the conjecturer:

- `min_number_simple_expressions = 1` — enables the preliminary nse=1 pass (the main pipeline uses `2`). This is how the incubator produces `(>[...](Anchor[...])(single_head))` conjectures that the main pipeline would skip.
- `max_values_for_uncomb_def_sets` + `max_values_for_def_sets` — controls the `createMapAnchor` memory footprint. Both typically small (≤ 3).

---

## Conjecturer config differences

The incubator triggers the conjecturer's nse=1 path (`singleExprAnchorConnection`) — which:

- Skips the pre-combination filters (`check_def_sets`, `max_number_args_expr`).
- Skips the operator-head checks (`check_input_variables_theorem_operator_head`, `evaluate_operator_exprs2`) that assume nse ≥ 2.
- Keeps `check_input_variables_order`, `pattern_in_conjecture`, `control_equality`.

This produces direct `Anchor → head` conjectures. Combined with the incubator's finite model, each conjecture can be exhaustively tested.

---

## Positive vs negative proofs

### Positive

A conjecture is proved by the standard hash-burst derivation. The `head-in-wholeExpressions` check inside `addTheoremToMemory` catches facts that become derivable from anchor disintegration alone — the incubator's anchor, which enumerates every fact about `0..8` directly, typically makes most ground-level positives derivable in a single layer.

### Negative

A conjecture is disproved by `try_contradiction`:

- A separate LB is spawned with the conjecture's negation as its goal.
- This LB runs the prover with `primedForContradiction = true` — which keeps the LB alive in `deactivateRecursively` even when it has no `toBeProved` entries.
- If the LB derives a contradiction (both `X` and `!X` for some `X`), the original conjecture is marked false; the contradiction LB broadcasts via direct `updateGlobalDirect`.
- Tagged `contradiction` in the proof graph.

Negative detection requires the intermediate values stay within the model range. For a conjecture involving `4 + 3 = 7` in a `{0..5}` model, the computation `4 + 3` is *not in range* — the contradiction chain breaks because intermediate facts do not exist. Fix: expand anchor with more elements (memory file ).

---

## Storage isolation

The incubator has its own theorem storage:

```
files/theorems_incubator/
    conjectures.txt
    reshuffled_conjectures.txt
    reshuffled_mirrored_conjectures.txt
    theorems.txt
    externally_provided_theorems.txt
    compressed_external_theorems.txt
    compressed_out_theorems.txt
```

The main pipeline's `files/theorems/` is never written by the incubator. This is enforced via the config's `theorems_folder` path override.

### Cross-batch externals seed

When the orchestrator iterates `tags = ["Peano", "Gauss"]` and reaches the second tag's incubator stage, `run_modes.py` seeds the next-tag incubator's `externally_provided_theorems.txt` from the prior-tag main batch's compiled proved-set. Source: `files/theorems/compiled_theorems.txt` (compact form, keeps `existence2` / `or0` heads — see [D-40](../40_decisions.md#d-40)). The C++ `--mirror-externals` step (`run_modes.py:_rebuild_compressed_externals`) then runs over that seed to produce `compressed_external_theorems.txt`, which `process_proof_graphs.py` writes out as the verifier registry's `external_theorems.txt`. The compact-form rules cited by chapter rows match this registry by alpha-canonical lookup.

The expanded-form file (`theorems.txt`, written by `prover.cpp`'s `expandToBaseForm`) is kept for any cross-batch consumer that needs operator-dictionary-free input. Compact-form propagation is supported because `GL_binary_shared.json` (`run_modes.py:_seed_per_batch_binary` + `_merge_into_shared`) carries the spontaneous-category compact dictionary across batches. Non-spontaneous compact heads (anchor / atomic) are not in the shared binary; if a future stage proves theorems whose compact heads fall outside `_SPONTANEOUS_CATEGORIES` and need cross-batch propagation, the shared binary's coverage must expand or the seed source must revert.

### Cross-batch state propagation — what crosses, what doesn't

The orchestrator (`run_modes.py:full_run`) iterates batches sequentially and propagates a small, explicit set of artifacts from one batch to the next. The full inventory:

| Artifact | Direction | Mechanism | Form / filter |
|---|---|---|---|
| Compact-name dictionary (spontaneous categories) | every batch ↔ shared binary | `_seed_per_batch_binary` (start) + `_merge_into_shared` (end) at `run_modes.py` | Categories in `_SPONTANEOUS_CATEGORIES = {"implication", "existence", "or", "and"}` only. Anchor and atomic entries DO NOT cross. |
| Compact-name dictionary (anchor / atomic) | within a batch only | `GL_binary_<tag>.json` per-batch file | Stays in the per-batch file; never enters shared. |
| Proved theorems (expanded form) | depends on consumer | `theorems.txt` (per-batch trunc, `prover.cpp`) | Inter-batch parser-input fallback. Expanded form is operator-dictionary-free. |
| Proved theorems (compact form) | next-tag incubator only | `compiled_theorems.txt` → seeded into next-tag's incubator `externally_provided_theorems.txt` ([D-40](../40_decisions.md#d-40)) | Compact form. Safe because spontaneous-category dictionary is shared. |
| External theorem registry | cross-batch | `compressed_external_theorems.txt` (per-batch, derived from seed via `--mirror-externals`) | Whatever form the seed is in; mirror variants added. |
| Proof graph chapters | per-batch only | `files/processed_proof_graph/` (main) or `files/incubator/processed_proof_graph/` (incubator) | Compact form (chapters cite rules with `existence2` / `or0` etc.). |
| Verifier registry | per-batch only | `external_theorems.txt` (in the same processed_proof_graph dir) | Compact form (after D-40). |

The "what doesn't cross" half is as load-bearing as the "what does":

- **Anchor compact names are batch-local.** `AnchorPeano`, `AnchorGauss`, `AnchorIncubator` are NOT in `GL_binary_shared.json`. If a downstream batch's prover or verifier tries to interpret a sender batch's anchor name without re-loading the per-batch binary, it will fail. The cross-batch externals seed avoids this by only carrying theorems whose compact heads are spontaneous-category — anchor names are referenced as MPL atoms, not as compact-dictionary lookups.
- **Atomic operator names are batch-local.** Per-batch atoms (operators introduced by the conjecturer) stay in `GL_binary_<tag>.json` and do not propagate.
- **`raw_proof_graph` is wiped per batch.** `empty_raw_proof_graph` runs at the start of each clean run; per-batch `raw_proof_graph` content is consumed by `create_processed_proof_graph` and not retained.

The minimal interface between batches is therefore:
1. The shared compact-name dictionary (spontaneous categories).
2. The previous main batch's compiled proved-set (compact form), seeded into the next-tag incubator's externals.

Everything else stays local. This keeps batches independently auditable while enabling the small set of cross-batch citations the proof graph requires.

### Back-reformulation

Incubator-only transformation. An operator-equality theorem of the form `(=[+[a,b], c])` is rewritten into the direct operator form `(in3[a,b,c,+])`. Tagged `incubator back reformulation` (see [verifier.py](../../verifier.py)).

---

## Pipeline position

Invoked as a separate Python entry point: `run_modes.incubator_run`. The main pipeline's `main.py` calls `full_run` which orchestrates both — incubator first (if configured), then main. Per memory file, the two paths are unified under one `full_run` entry, both enabled for the canonical multi-batch run.

---

## Weaknesses

### Known & tracked

- **Missing 138 negatives** — all require out-of-range intermediate values. Fixed by anchor expansion. Actively tracked in memory.
- **Anchor-expansion sizing tension.** Larger anchors mean more conjectures fit in range, but `createMapAnchor` memory cost grows multiplicatively with typed-arg count. 14 slots (9 typed) is the current balance; 15–16 would likely still work, 20+ won't without config-level knobs tightening.
- **Pipeline isolation suspect.** A Gauss incubator config change affected Peano CE — see. Root cause not yet isolated.

### Suspected fragility

- **`allow_multiplication` / `incubator_mode` / `ban_disintegration` / `compressor_mode` interplay.** Four flags now, after the brief introduction-and-collapse of `allow_disintegration` on 2026-04-29 (see [D-27](../40_decisions.md#d-27), [D-28](../40_decisions.md#d-28)). `ban_disintegration` is the single gate for every disintegration-shaped prover path; `allow_multiplication` is a separate gate for `multiplyImplication`. Risk reduced but not gone: a misconfiguration that sets `ban_disintegration=false` while `incubator_mode=true` (e.g. `ConfigIncubatorGauss1.json`) is now intentional rather than accidental, but the orthogonal flags multiply the test-matrix surface — every new incubator-side config should explicitly state `ban_disintegration` and `allow_multiplication`.
- **`head-in-wholeExpressions` short-circuit.** The path where `addTheoremToMemory` admits a head already known from anchor disintegration is incubator-specific. A refactor that changes the anchor-disintegration timing could silently break this.
- **Incubator results' propagation to CE.** The pipeline from incubator proved_theorems → `files/simple_facts/*.txt` goes through `incubator_to_simple_facts.py` (316 lines). That script's j-copy logic is documented only in the script; any change in anchor layout requires coordinated updates to both.

### Not exercised by tests

- **Anchor 4** (older anchor variant). 45 regressions exist for Anchor 4 relative to earlier runs (per ). Not prioritised for fix, but worth noting as "once-worked, now doesn't".
- **Batch sequencing.** The Incubator → Peano → Gauss sequence is the canonical run. Re-ordering (e.g. Gauss → Incubator) is not tested; the inter-batch external-theorem loading may depend on order.

---

## Open questions

- **OPEN-19 — RESOLVED.** The j-copy strategy in [`incubator_to_simple_facts.py`](../../incubator_to_simple_facts.py) has two rules (per the script's header comment at `:35–38` and implementation at `:142–190`):
 1. **Anchor-matching coverage**: always emit j0/j1 variants for every fact, up to `max_j=2`. This ensures the CE filter's hash-request generator can match anchor-related rules regardless of which j-copy slot the anchor pins.
 2. **Repetition-break**: when a fact has repeated i-value arguments (e.g. `(in3[i4,i4,i0,+])`), emit j-copy variants that break the repetition (e.g. `(in3[i4,j4,i0,+])`). This enables rules that require two different-looking arguments to match — without distinct copies, the hash engine would never fire them on the fact table. Matches the note about `generateEncodedRequestsStaticCE` needing distinct fact entries.
- **OPEN-20 — RESOLVED.** `compressed_out_theorems.txt` holds **eliminated external theorems** — theorems from `compressed_external_theorems.txt` that the compressor determined were redundant and eliminated from the external-theorem pool. Written at [`compressor.cpp`](../../GL_Quick_VS/GL_Quick/src/compressor.cpp); appended to at [`run_modes.cpp`](../../GL_Quick_VS/GL_Quick/src/run_modes.cpp) via *"Append eliminated externals to compressed_out_theorems.txt"*. Purpose: audit trail — "what did the compressor remove?" Referenced by [`run_modes.py`](../../run_modes.py) (Python orchestrator). Main pipeline has no equivalent because its `compressed_external_theorems.txt` is rebuilt each run rather than accumulated.

---

## See also

- [`10_pipeline/03_ce_filter.md`](03_ce_filter.md) — downstream consumer of `files/simple_facts/*.txt`.
- [`10_pipeline/04_prover.md`](04_prover.md) — shared prover engine.
- [I-7](../30_invariants.md#i-7) — Pass B gate in incubator mode.
-,.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
