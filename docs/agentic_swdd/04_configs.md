<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Configs `[DRAFT]`

> `files/config/*.json` controls almost every knob in the pipeline — which anchor, which expressions are in scope, per-expression size limits, conjecturer complexity budgets, prover parameters, and path overrides that let the incubator run in isolation from the main pipeline. This chapter is the unified reference.

---

## Chapter map

- [The config files](#the-config-files)
- [File-size landscape](#file-size-landscape)
- [Top-level schema](#top-level-schema)
- [Per-expression schema](#per-expression-schema)
- [`parameters` — conjecturer parameters](#parameters--conjecturer-parameters)
- [`prover_parameters` — prover parameters](#prover_parameters--prover-parameters)
- [Path overrides](#path-overrides)
- [Pattern / exclusion arrays](#pattern--exclusion-arrays)
- [Variant comparison](#variant-comparison)
- [ConfigVisu.json — the verifier's config](#configvisujson--the-verifiers-config)
- [tag_descriptions.json](#tag_descriptionsjson)
- [Loader cartography](#loader-cartography)
- [Adding a new config](#adding-a-new-config)
- [Weaknesses](#weaknesses)

---

## The config files

Location: `files/config/`. Every `Config<Tag>.json` corresponds to one batch; `ConfigVisu.json` is the verifier's parallel reference; `tag_descriptions.json` is the HTML-export glossary.

Current set:

| File | Purpose |
|---|---|
| `ConfigPeano.json` | Main-pipeline Peano batch. |
| `ConfigGauss.json` | Main-pipeline Gauss batch. **No longer carries `EnumerationSet2`** — that migrated to the rung-1 incubator config on 2026-04-29 ([D-25](40_decisions.md#d-25)). The file holding it is now `ConfigIncubatorGauss3.json` (was `ConfigIncubatorGauss1.json` before the AI3 anchor split). |
| `ConfigIncubatorPeano1.json` | Big-batch incubator over the 14-slot `AnchorIncubator8`. Renamed from `ConfigIncubatorPeano.json` to make room for an AI3 sibling. |
| `ConfigIncubatorPeano2.json` | AI3 mirror of `ConfigIncubatorPeano1.json`. Same expressions and parameters, anchored on the 9-slot `AnchorIncubator3` (i0..i3 only). Cheap pre-rung-1 batch that produces a fact base of AI3-tagged simple facts. |
| `ConfigIncubatorGauss1.json` | Big-batch incubator over the 14-slot `AnchorIncubator8`. Renamed from `ConfigIncubatorGauss.json` to make room for an AI3 sibling. |
| `ConfigIncubatorGauss2.json` | AI3 mirror of `ConfigIncubatorGauss1.json`. Same expressions and parameters, anchored on the 9-slot `AnchorIncubator3`. Runs after Peano2 and before Gauss3 to enrich the AI3 fact base. |
| `ConfigIncubatorGauss3.json` | Rung-1 **and** rung-2 incubator batch on the 9-slot `AnchorIncubator3` (i0..i3 only). Enables both `EnumerationSet2` and `EnumerationSet3` (`max_count_per_conjecture: 1`) so the conjecturer formulates both set-equalities — rung-1 `{0,1}=[0,1]` (`EnumerationSet2[2,6,10] ⟹ interval[1,4,2,6,10]`) and rung-2 `{0,1,2}=[0,2]` (`EnumerationSet3[2,6,7,10] ⟹ interval[1,4,2,7,10]`, where `7`=`i2`=2). Sets `allow_disintegration=true`, `allow_multiplication=false`, `incubator_mode=true`, `lb_split=true` (the one incubator config that runs the statistics-driven LB split, [D-231](40_decisions.md#d-231)). Renamed from `ConfigIncubatorGauss1.json` when the incubator anchor was split into a 9-slot AI3 and a 14-slot AI8. |
| `ConfigIncubatorGauss.json.bak` | Backup of a previous incubator config — kept for recovery, not loaded by the pipeline. |
| `ConfigVisu.json` | Verifier + HTML-export reference (covers all expressions appearing in any batch). |
| `tag_descriptions.json` | HTML-export tag glossary. |

Batch anchor tag matches the filename suffix: `Peano` → `AnchorPeano`, `Gauss` → `AnchorGauss`, `IncubatorPeano1` / `IncubatorGauss1` → `AnchorIncubator`. An optional `anchor_name` override at the top level can point at a different anchor (used when the incubator configs override the implicit mapping — `ConfigIncubatorGauss3.json` overrides to `AnchorIncubator3`).

**Per-tag config glob (since 2026-04-29).** The orchestrator at `run_modes.py` discovers every file matching `^Config<base_tag>\d*\.json$` for each base tag (Peano, Gauss; and `Incubator<base_tag>` for the incubator side). Files are sorted alphanumerically — Python's default string sort puts `ConfigPeano.json` before `ConfigPeano1.json` because `.` (ASCII 46) sorts before `1` (ASCII 49). So the un-suffixed config always runs first, and digit-suffixed configs follow in numeric order. Within one tag's incubator group, every batch shares the same `theorems_folder` (per the file's `theorems_folder` override) so batch-N+1 inherits batch-N's `theorems.txt` automatically.

---

## File-size landscape

Line counts on the current branch:

| File | Lines | Notes |
|---|---:|---|
| `ConfigPeano.json` | 184 | Smallest — 8 expression entries. |
| `ConfigGauss.json` | 1019 | Largest non-incubator — 18 expressions, the conjecturer's `max_values_*` matrices add bulk. |
| `ConfigIncubatorPeano.json` | 484 | 20 expressions. |
| `ConfigIncubatorGauss.json` | 428 | 21 expressions. |
| `ConfigVisu.json` | 290 | 20 expressions — verifier needs every expression that appears anywhere. |
| `tag_descriptions.json` | 29 | 27 tag entries (flat). |

Lineage: the Gauss config is much bigger than Peano because every expression in it carries per-def-set matrices that don't exist in Peano's simpler setup.

---

## Top-level schema

Every `Config<Tag>.json` is a JSON object keyed by:

- **Expression names** — one entry per MPL expression in scope. Value: per-expression config object (see below).
- **`parameters`** — conjecturer parameters object (see below). Optional; reasonable defaults apply.
- **`prover_parameters`** — prover parameters object (see below). Optional.
- **`patterns_to_exclude`** — array of string patterns the conjecturer rejects. Optional.
- **`only_in_head`** — array of predicate names that may appear only in the head of a conjecture, never in premises. Optional.
- **`prohibited_combinations`** — array-of-arrays listing forbidden co-occurrences of predicates. Optional.
- **`prohibited_heads`** — array of predicate names that may never be a head. Optional.
- **`anchor_name`** — override for the implicit anchor derivation from filename. Optional. Read at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp).
- **`theorems_folder`** — path override (incubator only). Optional. Read at [`run_modes.cpp`](../GL_Quick_VS/GL_Quick/src/run_modes.cpp).
- **`background_theorems_folder`** — path override (incubator only). Optional. Read at [`run_modes.cpp`](../GL_Quick_VS/GL_Quick/src/run_modes.cpp).
- **`raw_proof_graph_folder`** — path override (incubator only). Optional. Read at [`run_modes.cpp`](../GL_Quick_VS/GL_Quick/src/run_modes.cpp).

No JSON schema validation is enforced — unknown fields are silently ignored (see [Weaknesses](#weaknesses)).

---

## Per-expression schema

Every expression's config entry has (some fields are optional):

| Field | Type | Purpose |
|---|---|---|
| `arity` | int | Number of arguments. Must match the MPL `[arg1, arg2,...]` count. |
| `definition_sets` | object `{index → [type, combinable]}` | Per-argument type label (see [`03_mpl.md`](03_mpl.md#type-label-vocabulary)) + combinable flag. |
| `full_mpl` | string | Inline MPL body, OR a filename (must end in `.mpl`) to load from `files/definitions/`. |
| `short_mpl` | string | Canonical signature in `u_`-form for diagnostic output. |
| `max_count_per_conjecture` | int | Cap on copies of this predicate in one conjecture. `0` removes the expression from the conjecturer's seed list entirely (`Conjecturer::run` builds `exprList` only from positive-cap entries) — the single participation switch. |
| `input_args` | array of string | Argument positions that are "inputs" (flow in). Used by `findDigitArgs` + the `operators` set. |
| `output_args` | array of string | Argument positions that are "outputs". An expression with non-empty `output_args` is an **operator**; with empty, it's a **relation**. |
| `max_size_expression_before_existence` | int | Conjecturer size limit when the expression appears *before* an existence-head. |
| `max_size_expression_after_existence` | int | Conjecturer size limit when it appears *after* an existence-head. |
| `max_size_expression` | int | **Dead field.** Present in incubator/Gauss configs but never read by the conjecturer loader (`Conjecturer::loadConfiguration` reads only the before/after pair). |
| `min_size_expression` | int | Minimum complexity (rare). |
| `allow_to_constitute_existence` | bool | Whether this predicate may head an existence expression. |
| `existence_variable_position` | int | Which argument position carries the bound variable when heading an existence. |
| `allowed_for_existence` | array of int | Which argument positions may carry the bound variable. |
| `allow_negation` | bool | Whether this expression may appear negated (`!(...)`). |
| `allowed_combinations` | array | **Unimplemented.** Documented-only whitelist — no code reads it (the only combination controls are the blacklists `prohibited_combinations`, `only_in_head`, `patterns_to_exclude`). |

Fields that are absent get default-zero / default-empty semantics. The C++ conjecturer reads per-expression entries via `modifyCoreExpressionMap` at [`compiler.hpp`](../GL_Quick_VS/GL_Quick/src/compiler.hpp).

### Example — `ConfigPeano.json`'s `in3`

```json
"in3": {
    "arity": 4,
    "definition_sets": {
        "1": ["(1)", true], "2": ["(1)", true],
        "3": ["(1)", true], "4": ["P(x(1)(x(1)(1)))", false]
    },
    "full_mpl": "(in3[1,2,3,4])",
    "short_mpl": "(in3[1,2,3,4])",
    "max_count_per_conjecture": 5,
    "input_args": ["1", "2"],
    "output_args": ["3"],
    "max_size_expression_before_existence": 5,
    "max_size_expression_after_existence": 5
}
```

Decoded: `in3` takes 4 arguments — positions 1, 2, 3 are `(1)`-typed (elements of `N`) and combinable; position 4 is `P(x(1)(x(1)(1)))` (a binary function) and fixed. Inline body `(in3[1,2,3,4])` (atomic). Inputs at positions 1 and 2; output at position 3 — so `in3` is an operator (`a + b = c` shape). Up to 5 copies per conjecture. Size limits at 5 in both conjecturer contexts.

### Example — `ConfigGauss.json`'s `fold`

```json
"fold": {
    "arity": 6,
    "definition_sets": { /* 6 slots with type labels */ },
    "full_mpl": "fold.mpl",
    "short_mpl": "(fold[n,m,f,p,N,+])",
    ...
}
```

Decoded: 6-ary. `full_mpl = "fold.mpl"` means load the MPL body from [`files/definitions/fold.mpl`](../files/definitions/fold.mpl) — the complex folds-and-accumulates definition. Short form is the readable signature.

---

## `parameters` — conjecturer parameters

Top-level `parameters` object. Read at [`conjecturer.cpp–651`](../GL_Quick_VS/GL_Quick/src/conjecturer.cpp).

| Field | Type | Default | Purpose |
|---|---|---|---|
| `min_number_simple_expressions` | int | `2` | Minimum `nse`. When `1`, the conjecturer's nse=1 preliminary pass runs. |
| `max_number_simple_expressions` | int | `0` | Maximum `nse`. When `< 2`, the main combination loop is skipped. |
| `max_size_mapping_def_set` | int | `0` | Cap on mapping-sized `def_set`. |
| `max_number_args_expr` | int | `0` | Cap on argument count per expression in a conjecture. |
| `operator_threshold` | int | `0` | Operator-count threshold for one of the filter cascades. |
| `max_size_binary_list` | int | `0` | Cap on binary-list size. |
| `incubator_mode` | bool | `false` | Enables incubator-specific nse=1 path. Set by incubator configs. |
| `apply_in_premise_filter` | bool | `true` | **Dead field**. Declared at `conjecturer.hpp`, loaded at `conjecturer.cpp`, but its value is never consulted in `passesInPremiseFilter` nor at its callsites. `ConfigGauss.json` carries `false` with no effect. Leave the key in configs (harmless) or remove it; a missing key defaults to `true`. See [`10_pipeline/02_conjecturer.md`](10_pipeline/02_conjecturer.md#passesinpremisefilter) for the full correction. |
| `max_values_for_def_sets` | object | `{}` | Per-def-set cap on combinable values. Feeds `createMapAnchor`'s `rightMax`. |
| `max_values_for_uncomb_def_sets` | object | `{}` | Same but for uncombinable values. |
| `max_values_for_def_sets_prior_connection` | object | `{}` | Per-def-set cap applied before connection. |
| `max_complexity_if_anchor_parameter_connected_before_existence` | object | `{}` | Per-expression complexity cap (before-existence variant). |
| `max_complexity_if_anchor_parameter_connected_after_existence` | object | `{}` | After-existence variant. **Vector form** since commit: per-type value can be either a legacy int `complexity_cap` (auto-promoted to `[complexity_cap, 100]` by the loader at [`conjecturer.cpp`](../GL_Quick_VS/GL_Quick/src/conjecturer.cpp)) **or** a 2-tuple `[complexity_cap, arity_sum_cap]`. The arity-sum dimension counts the sum of arities across all non-anchor leaves (descending into nested existence-head structure); a conjecture is rejected only when *both* `complexity > complexity_cap` AND `arity_sum > arity_sum_cap`. Set `arity_sum_cap = 100` to disable the arity dimension for that type (no realistic conjecture exceeds 100). Used to admit small-arity-sum nse=3 shapes (e.g. cancellation-family) that exceed the legacy complexity cap but stay structurally simple. |
| `max_distinct_anchor_values_per_type` | object | `{}` | Per-def-set-type cap on the number of *distinct* anchor-slot values that may appear as args in the non-anchor leaves of a conjecture. The filter walks into nested `!(>[…]…)` existence heads when collecting leaves. Empty map → filter off (legacy default). Applied at the three `connExpr2`-producing filter-cascade sites in `singleThreadCalculationInt`, `singleExprAnchorConnectionInt`, and `singleExprAnchorConnection` (helper: `passesMaxDistinctAnchorValuesPerType` at [`conjecturer.cpp`](../GL_Quick_VS/GL_Quick/src/conjecturer.cpp)). Orthogonal to the complexity cap: this filter gates on *count* of distinct values regardless of complexity or existence-head presence. Introduced by commit to bound combinatorial growth without losing any baseline-proven shapes (config values were chosen per-tag to match observed proved-theorem maxes). |
| `simple_facts_parameters` | array[int] | `[]` | Pass-through to fact loading for the CE filter. |
| `fact_variable_kinds` | array | `[]` | Pass-through — meaning tbd (see [OPEN-10](AGENT_SwDD.md#open-questions)). |

**Critical sizing invariant.** For `createMapAnchor`, `rightMax = max over def_sets of (uncomb + comb) values`. When `rightMax > 3`, RAM explodes — millions of permutation dicts are materialised. Keep `max_values_for_uncomb_def_sets + max_values_for_def_sets ≤ 3` per def-set.

---

## `prover_parameters` — prover parameters

Top-level `prover_parameters` object. Read at [`prover.cpp–162`](../GL_Quick_VS/GL_Quick/src/prover.cpp). Defaults declared in `struct ProverParameters` at [`parameters.hpp–73`](../GL_Quick_VS/GL_Quick/src/parameters.hpp).

### Iteration-budget fields

| Field | Type | Default | Purpose |
|---|---|---|---|
| `sizeAllBinariesAna` | int | `10` | Cap on binary-expression count per analysis step. |
| `maxIterationNumberProof` | int | `30` | Main-prover iteration cap per LB. |
| `numberIterationsConjectureFiltering` | int | `1` | CE-filter iteration budget per batch. |
| `maxSizeDefSetMapping` | int | `5` | Cap on def-set mapping table size. |
| `maxSizeTargetSetMapping` | int | `12` | Cap on target-set mapping size. |
| `maxNumberSecondaryVariables` | int | `2` | Cap on DISTINCT secondary (`it_*_lev_*`) variables per hash request (`requestGatesPass`; also the Pass B admission count guard). |
| `maxNumberSecondaryVariablesOrint` | int | `2` | Scoped widening of the distinct-secondary cap: applies only to a request whose premises all sit at one shared `_orint_` branch scope ([D-210](40_decisions.md#d-210)). Default equal to the standard cap = no widening. |
| `sizeAllPermutationsAna` | int | `7` | Cap on permutation table size for analysis. |
| `minNumOperatorsKey` | int | `2` | Minimum operator-count in a hash key (main mode). |
| `minNumOperatorsKeyCE` | int | `4` | Same, CE mode. |
| `maxIterationNumberVariable` | int | `1` | The per-batch WITNESS-GENERATION cap, enforced at all three stations of the witness economy ([D-232](40_decisions.md#d-232)): request building (`filterIntEncodedStatements` drops statements whose `it_<N>_lev_...` names exceed it), revival (`revisitRejected2`'s strict-`>` doors), and admission (`isAdmitted`'s per-value depth is stamped from it). With the generation wire ([D-233](40_decisions.md)) a fired head's witnesses mint at max premise iteration + 1, so this bounds the witness-of-witness cascade depth; each increment opens one more totality layer per step. Values (maintainer-directed): `1` in every batch except `ConfigIncubatorGauss3.json` at `2` — the other batches keep their historical request cap (1) as the unified value (measured no-op on their pools: proven counts byte-equal to the pre-unification reference at value 1), while IncubatorGauss3 needs generation-1 admission for rung 1's second-layer sum witnesses and generation 2 for rung 2's deeper `_orint_` descents (rung 2 measured unproved at cap 1 with working admission; the earlier 2/3/5 raises compensated through the wrong parameter while admission stayed at depth 0 — the rung-1 regression). |
| `standardMaxSecondaryNumber` | int | `1` | Standard admission-path secondary-variable cap. |
| `inductionMaxAdmissionDepth` | int | `1` | Induction-specific admission depth. |
| `inductionMaxSecondaryNumber` | int | `2` | Induction-specific secondary-variable cap. |
| `counterExampleBoundary` | int | `6` | Counter-example boundary. |
| `minLenLongKey` | int | `5` | Minimum long-key length. |
| `maxLenHypoKey` | int | `2` | Maximum hypothesis-key length. |
| `max_or_depth` | int | `1` | Max nesting depth of OR branch scopes (1 = no nested `_ordis_` scopes). Gates only the per-branch case-split; the K mutual-exclusion implications emit at every depth ([D-211](40_decisions.md#d-211)). |
| `max_partition_size` | int | `5` | Cap on `multiplyImplication` Bell-partition size. |
| `min_split_work` | int | `20000` | The straggler trigger's ONLY tunable knob (the split fan-out is `logicalCores`, not a config number): an LB splits into `logicalCores` expression buckets next iteration only if its total submatch work this iteration exceeds BOTH the idle-core fair-share `T / logicalCores` AND this floor. The floor is the per-bucket setup break-even — each bucket re-pays `filterIntEncodedStatements` + the obligatory-stump builders over the whole statement universe — so below it splitting cannot pay off (it suppresses splitting a trivially cheap iteration). Consumed by `isStraggler`. See [D-201](40_decisions.md#d-201). |
| `axed_anchor_exception` | bool | `false` | Gates the axed-variable deposit filter's positive-anchor exception ([D-234](40_decisions.md)): when true, a positive anchor-category statement carrying an axed x-copy name deposits, letting an anchor-bridge firing land the external anchor's x-form as a live statement. Default (and every shipped config) false — the live x-anchor opens the external anchor's rule universe at x-arguments and explodes the batch runtime. |
| `disable_lb_split` | bool | `false` | Diagnostic / RT-profiling flag. When true, `proveKernel` runs every LB unsplit (one phase-2 part) so the per-call `RTTracker` homed in `performElem2` measures the LB's whole hashburst (one `.rt/<chain>.log` per LB). Default false leaves the production parallel path untouched. See [D-110](40_decisions.md#d-110), [I-59](30_invariants.md#i-59). |
| `lb_split` | bool | `true` | The split master switch: when false, `proveKernel` excludes every LB of the batch from the statistics-driven split ([D-201](40_decisions.md#d-201)) — the gate is `mainPath = lb_split && !disable_lb_split`; `incubator_mode` is no longer consulted. Set false by `ConfigIncubatorPeano1/2` and `ConfigIncubatorGauss1/2` (thousands of small LBs whose per-part setup is redundant); set true by `ConfigIncubatorGauss3` (heavy rung LBs). Absent in `ConfigPeano`/`ConfigGauss` (default true). Independent of the diagnostic `disable_lb_split`. See [D-231](40_decisions.md#d-231). |
| `maxNumberHashRequests` | int | `40000` | **Unused** no-op field. Was the per-part submatch cap that triggered the split; retired when the trigger became statistics-driven and preemptive ([D-201](40_decisions.md#d-201)) — main-path bursts now run to completion. Retained only so existing config files keep parsing. |
| `fixed_number_splits` | int | `20` | **Unused** no-op field. Was the rule-split escalation target / bucket count; the fan-out is now `logicalCores`. Retained only so existing config files keep parsing. See [D-201](40_decisions.md#d-201). |
| `second_split_submatch_cap` | int | `10000` | **Unused** no-op field. Was the second cap of the retired two-cap rule→stump escalation. Retained only so existing config files keep parsing. See [D-201](40_decisions.md#d-201). |
| `split_growth_factor` | int | `2` | **Unused** no-op field (long retired graduated policy). Retained only so existing config files keep parsing. |
| `max_number_splits` | int | `32` | **Unused** no-op field (long retired graduated policy). Retained only so existing config files keep parsing. |
| `split_fallback_ratio` | double | `0.10` | **Unused** no-op field. Was the coarsen threshold of the bang-bang cap policy; the trigger is now the statistics pass (`isStraggler`), which de-splits when an LB's total work falls below the fair-share. Retained only so existing config files keep parsing. See [D-201](40_decisions.md#d-201). |
| `static_pool_bytes` | int64 | | Statification sizing: total bytes of the ONE static-memory reservation made at program start (never freed, never grown). The current 12 GiB cap provides 49152 blocks at `static_block_bytes = 256 KiB`, giving the heavy rung-2 incubator batch resident headroom; the working-set pager still engages whenever a batch's live set exceeds the pool. The historical sequence was 2 GiB → 4 GiB for cold-string floor consumption, then 4 GiB → 8 GiB for added headroom, a user-directed return to 4 GiB ([D-209](40_decisions.md#d-209)), then a raise to 6 GiB to clear the `IncubatorGauss1` static-pool exhaustion and reach the rung-2 `IncubatorGauss3` batch, then 6 GiB → 12 GiB alongside the `NameId` int16→int32 migration ([D-212](40_decisions.md#d-212)), which raised `MAX_NAME_IDS` from 32000 to 1,000,000 and cleared the id-width ceiling that had blocked the `IncubatorGauss3` proof. Must be a whole multiple of `static_block_bytes`; the pair is validated by `isValidStaticMemoryConfig` ([`parameters.hpp`](../GL_Quick_VS/GL_Quick/src/parameters.hpp)) and asserted in the `ExpressionAnalyzer` constructor. |
| `static_block_bytes` | int32 | `262144` | Statification sizing: bytes per block — the grant unit the global static-memory manager dispenses to each LB's bump arena (`LbArena`). Must be a power of two, so a virtual offset resolves to a physical address by one shift and one mask. |
| `static_page_bytes` | int32 | `8192` | Statification sizing: bytes per page — the fixed allocation unit each LB's manager carves a granted block into and hands to the paged containers. Must be a power of two and divide `static_block_bytes` evenly; validated by `isValidStaticPageConfig` ([`parameters.hpp`](../GL_Quick_VS/GL_Quick/src/parameters.hpp)) and asserted in the `ExpressionAnalyzer` constructor. |
| `static_persistent_pool_bytes` | int64 | | Statification sizing: total bytes of the SECOND (persistent) static-memory reservation — a pool separate from `static_pool_bytes` that is never deloaded and backs `Memory::intToBeProved` (the determinism fix; see [`20_core_concepts/09_static_memory.md`](20_core_concepts/09_static_memory.md) and [I-95](30_invariants.md#i-95)). Sized by peak simultaneously-active-LB count (each active LB pins ≥1 persistent block for its whole active life, reclaimed only at discharge), not by content volume. Reuses `isValidStaticMemoryConfig` with `static_persistent_block_bytes`; asserted in the `ExpressionAnalyzer` constructor. Exhaustion asserts naming this knob. |
| `static_persistent_block_bytes` | int32 | `32768` | Statification sizing: bytes per block of the persistent pool — smaller than `static_block_bytes` (the per-LB persistent content is tiny). 32 KiB = 4 × `static_page_bytes`. Must be a power of two and a whole multiple of `static_page_bytes` (the page tier is shared); validated by `isValidStaticMemoryConfig` (pool/block) and `isValidStaticPageConfig` (block/page). |
| `static_mail_pool_bytes` | int64 | | Statification sizing: total bytes of the THIRD (mail) static-memory reservation — a stand-alone pool separate from both `static_pool_bytes` and `static_persistent_pool_bytes` that is never deloaded and backs the cross-LB pull-model mail log (see [`20_core_concepts/09_static_memory.md`](20_core_concepts/09_static_memory.md) and [`20_core_concepts/03_mail_system.md`](20_core_concepts/03_mail_system.md)). Nothing reads its grant ledger, so it plays no role in any deload/throttle/steward decision. Sized by peak total mail content across a batch (tune by the pool telemetry). Reuses `isValidStaticMemoryConfig` with `static_mail_block_bytes`; asserted in the `ExpressionAnalyzer` constructor. Exhaustion asserts naming this knob. |
| `static_mail_block_bytes` | int32 | `262144` | Statification sizing: bytes per block of the mail pool — matched to `static_block_bytes` (mail content is bulky, so large blocks keep grant traffic low). Must be a power of two and a whole multiple of `static_page_bytes` (the page tier is shared); validated by `isValidStaticMemoryConfig` (pool/block) and `isValidStaticPageConfig` (block/page). |

> **Memory-sizing constants are not config keys.** The final seven rows above — `static_pool_bytes`, `static_block_bytes`, `static_page_bytes`, `static_persistent_pool_bytes`, `static_persistent_block_bytes`, `static_mail_pool_bytes`, `static_mail_block_bytes` — are fixed compile-time constants in `struct ProverParameters` ([`parameters.hpp`](../GL_Quick_VS/GL_Quick/src/parameters.hpp)), identical for every batch and never read from a config ([D-139](40_decisions.md#d-139)). They are tabulated here only for their values and carving contract. (`hot_arena_bytes` was removed with the HOT substrate — [D-185](40_decisions.md).)

### Compressor fields

| Field | Type | Default | Purpose |
|---|---|---|---|
| `compressor_mode` | bool | `false` | True inside Phase 1 compressor LBs. Set by the compressor internally, not the config. |
| `ban_disintegration` | bool | `false` | Gates **every disintegration-shaped path** in the prover: Pass B at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp), back-reformulation at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp), hypothetical disintegration at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp), necessity-for-equality-hypo at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp). The compressor sets this true during Phase 1. Also read by `run_modes::fullRun` to decide whether to skip the compressor entirely. Pre-2026-04-29 this gated only the latter three; on 2026-04-29 the `incubator_mode` Pass-B coupling was first redirected to a brief `allow_disintegration` flag and then collapsed into `ban_disintegration` ([I-7](30_invariants.md#i-7), [D-28](40_decisions.md#d-28)). |
| `max_origin_per_expr` | int | `1` | Origin-record cap in normal prover runs. |
| `compressor_max_origins_per_expr` | int | `30` | Origin-record cap during Phase 1 hash bursts. |
| `compressor_hash_bursts` | int | `15` | Number of hash bursts per Phase 1 LB. |

### Incubator fields

| Field | Type | Default | Purpose |
|---|---|---|---|
| `try_contradiction` | bool | `false` | Enables contradiction-attempt LBs for negative conjectures. Set by every incubator config and by both main configs (`ConfigPeano.json` / `ConfigGauss.json` — main-path activation, [D-213](40_decisions.md#d-213)). |
| `try_contradiction_negated_head` | bool | `false` | Complement of `try_contradiction`: every registered conjecture also gets a reductio LB that assumes the negation of its head and proves the conjecture on a main-scope contradiction ([D-214](40_decisions.md#d-214)). Set by `ConfigIncubatorPeano2.json` and both main configs. |
| `skip_ce_filter` | bool | `false` | Bypasses CE filtering. Incubator-only (the incubator produces the facts it would otherwise consume). |
| `mirror_refutation` | bool | `true` | CE-filter mirror-refutation heuristic ([D-229](40_decisions.md#d-229)): a CE-refuted operator-only conjecture flips its `mirror_pairs.txt` partner to refuted too. Gates only the CE-filter flip pass — never a proof step. Set explicitly `true` in `ConfigPeano.json` / `ConfigGauss.json`. |
| `skip_eq_classes` | bool | `false` | Bypasses equivalence-class registration. *(Not yet covered in detail — see [OPEN-CFG-1](#open-questions).)* |
| `incubator_mode` | bool | `false` | Used to be the master incubator flag. As of 2026-04-29 (see [D-27](40_decisions.md#d-27)) it **no longer** gates Pass B — that's `!ban_disintegration` per [I-7](30_invariants.md#i-7) — nor `multiplyImplication` — that's `allow_multiplication`. Still governs the head-already-registered short-circuit in `addTheoremToMemory`, integration reformulation at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp), the relaxed-tempArgs assert at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp), conjecturer repeat-arg / op-head reformulation / OR-generation suppression, and the compressor skip in `run_modes::fullRun`. |
| `allow_multiplication` | bool | `false` | Gates `multiplyImplication` at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp). Replaces the older `incubator_mode` use; the `ceFilteringActive` carve-out is preserved (CE filter multiplies regardless). True for legacy incubator configs (multiplication preserved); false for main configs and `ConfigIncubatorGauss3.json` (the rung-1 AI3 batch, renamed from the older `ConfigIncubatorGauss1.json`). |

### Debug / observability fields

| Field | Type | Default | Purpose |
|---|---|---|---|
| `trackHistory` | bool | `true` | Keep per-expression history for debug dumps. |
| `debug` | bool | `false` | Enable verbose debug output. |

---

## Compile-time profiling

A second family of parameter lives **inside** `parameters.hpp` and is *not* loaded from the JSON config. These are read by the preprocessor or by `static constexpr` substitution at compile time; flipping them requires a rebuild. The block sits at the top of the header alongside the `ExecutionParameters` sizing constants.

| Symbol | Where | Type | Default | Purpose |
|---|---|---|---|---|
| `RT_MEASUREMENT` | macro at file head | `#define` (0 or 1) | `0` | Master gate for the per-LB runtime-measurement infrastructure inside `performElementaryLogicalStep`. `0` removes every call site at preprocess time; `1` enables them. Modeled on `GL_DISINT_PROFILE` in `compiler.hpp`. |
| `RTMeasurementParameters::RT_TIME_TRIGGER_SECONDS` | inside `namespace gl` | `static constexpr int` | `120` | A `.rt/<chain>.log` is written only after a single `performElementaryLogicalStep` call has been running this many wall-clock seconds. Below the threshold, no artefact appears. |
| `RTMeasurementParameters::RT_MIN_PERCENTAGE` | inside `namespace gl` | `static constexpr int` | `2` | Sections below this share of the call's total self-time fold into a single trailing line. |
| `RTMeasurementParameters::RT_MAX_SECTIONS` | inside `namespace gl` | `static constexpr int` | `64` | Fixed capacity of the tracker's per-call section array. No heap allocation; overflow asserts per the project conventions. |

Full design: [`_meta/rt_measurement.md`](_meta/rt_measurement.md). The block is intentionally separate from the JSON `prover_parameters` body because changing it cannot be done at runtime and because the gate macro must be visible to the preprocessor.

---

## Path overrides

Incubator configs set these to direct output to `files/theorems_incubator/` and `files/raw_proof_graph/` (or a dedicated incubator raw-graph folder).

| Field | Example value | Purpose |
|---|---|---|
| `theorems_folder` | `"files/theorems_incubator"` | Override `files/theorems/` for this batch. Routes `conjectures.txt`, `theorems.txt`, `compressed_external_theorems.txt` to a separate tree. |
| `background_theorems_folder` | `"files/theorems"` | If set, `proved_set` is union-loaded from both `theorems_folder/theorems.txt` and `background_theorems_folder/theorems.txt`. Lets the incubator carry main-pipeline theorems as external axioms. |
| `raw_proof_graph_folder` | `"files/raw_proof_graph_incubator"` | Override the raw-proof-graph output directory. |

When omitted, defaults point into `files/theorems/` and `files/raw_proof_graph/` — the main-pipeline locations.

Read at [`run_modes.cpp–124`](../GL_Quick_VS/GL_Quick/src/run_modes.cpp).

---

## Pattern / exclusion arrays

These four arrays shape the conjecturer's filter cascade.

### `patterns_to_exclude`

Array of string patterns. Conjectures whose stringified form matches any pattern are rejected by `patternInConjecture` at [`conjecturer.cpp`](../GL_Quick_VS/GL_Quick/src/conjecturer.cpp). The match is a `std::regex_search` over the *whole* rendered conjecture, so a pattern anchored on one sub-expression (`\(EnumerationSet2\[…\]\)`) pins that operator's argument slots via negative lookahead.

**Performance contract.** Every pattern runs on every string-lane candidate through the backtracking `std::regex` engine, so pattern cost multiplies into conjecturer wall time. Write patterns `^`-anchored (one evaluation position instead of one per character) and hop with negated character classes instead of `.*` where a delimiter char is unique to an atom — e.g. `=` occurs only inside `(=[` atoms, so `[^=]*\(=\[[^=]*\(=\[` detects two equality atoms in one deterministic scan. An unanchored lookahead of the form `(?=(?:.*X){2})` is quadratic per start position and measured ~10x total Peano conjecturer wall time (200 s vs 47 s for the semantically identical linear set). Two further scope facts: negated-premise variants are generated *after* this filter (bar the positive base form — see the gotcha `patterns_to_exclude`-misses-negated-variants), and the D-215 template rows DO pass through it (an all-negated-equality pattern guard like `[^!]\(=\[` keeps them safe).

**Worked example — `ConfigIncubatorGauss3.json` pins two set-equalities.** The pins keep `EnumerationSet2` at `[2,6,*]` and `EnumerationSet3` at `[2,6,7,*]` (the enumerated values `{0,1}` and `{0,1,2}`), and `interval` at `[1,4,2,*,*]` with the upper bound `m` relaxed to `{6,7}` (`…,(?!6,)(?!7,)\d+,…`). That yields both `{0,1}=[0,1]` (m=6) and `{0,1,2}=[0,2]` (m=7). Because the m-pin is relaxed rather than correlated to the enumeration head, the two false cross-pairings `{0,1}=[0,2]` and `{0,1,2}=[0,1]` are also generated and later disproved by the incubator. `EnumerationSet3`'s three distinct `(1)` element values need a three-wide `(1)` budget; it is carried entirely on the combinable side (`max_values_for_def_sets["(1)"]=3`, `max_values_for_uncomb_def_sets["(1)"]=0`) so `rightMax = 3+0 = 3` stays under the `createMapAnchor` RAM boundary, which in turn requires every `(1)` element/bound slot (`EnumerationSet2.a`, `interval.n`) to be flagged combinable. Because those all-combinable candidates carry more combinable args, `max_number_args_expr` must be raised to `3` (the minimum that admits both conjectures — `2` emits only rung-1, and the pre-existing value `1` emitted neither).

### `only_in_head`

Array of predicate names. These predicates may only appear in the head of a conjecture, not in any premise. Checked by `onlyInHeadGood` at [`conjecturer.cpp`](../GL_Quick_VS/GL_Quick/src/conjecturer.cpp).

### `prohibited_combinations`

Array of arrays. Each inner array is a list of predicate names that may not all co-occur in one conjecture. Checked by `checkProhibitedCombinations` at [`conjecturer.cpp`](../GL_Quick_VS/GL_Quick/src/conjecturer.cpp).

### `prohibited_heads`

Array of predicate names that may never serve as the head of a conjecture. Checked by `prohibitedHeadsGood` at [`conjecturer.cpp`](../GL_Quick_VS/GL_Quick/src/conjecturer.cpp).

---

## Variant comparison

Which expressions appear in which config:

| Expression | Peano | Gauss | IncubatorPeano1 | IncubatorPeano2 | IncubatorGauss1 | IncubatorGauss2 | IncubatorGauss3 | Visu |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `AnchorPeano` | ✓ | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `AnchorGauss` | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `AnchorIncubator8` | — | — | ✓ | — | ✓ | — | — | ✓ |
| `AnchorIncubator3` | — | — | — | ✓ | — | ✓ | ✓ | ✓ |
| `NaturalNumbers` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `in` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `=` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `in2` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `in3` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `fXY` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `fXYZ` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `fold` | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `identity` | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `interval` | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `limitSet` | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| `limitSequence` | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| `preorder` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `residual` | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `sequence` | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `split` | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `infiniteSequence` | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| `EnumerationSet2` | — | — | — | — | — | — | ✓ | ✓ |
| `EnumerationSet3` | — | — | — | — | ✓ | ✓ | ✓ | ✓ |
| `constSeq` | — | — | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| `nonInterval` | — | — | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| `nonSequence` | — | — | ✓ | ✓ | ✓ | ✓ | ✓ | — |

Observations:

- **Peano ⊂ Gauss ⊂ Incubator*** — each broader config is a superset of the narrower ones on shared expressions.
- **`ConfigVisu` was a near-superset but not quite** — historically `limitSet`, `limitSequence`, `infiniteSequence`, `constSeq`, `nonInterval`, `nonSequence` were absent on the assumption that the verifier only needed expressions appearing in chapters (a subset of what the conjecturer and prover generate). [D-41](40_decisions.md#d-41) found this assumption broke once the new `definition set consistency` meta-check started looking up composite operators reached via chapter rows. Fixed by syncing `ConfigVisu.json` to the per-batch authoritative versions for all six absent operators + correcting one drifted defset (`interval` pos 2). **Invariant going forward:** ConfigVisu's atomic-operator `definition_sets` must mirror the per-batch configs for any operator both contain. Drift detection: .
- **Incubator configs reference all three anchors** — because they can load main-pipeline theorems as external axioms via `background_theorems_folder`.

---

## ConfigVisu.json — the verifier's config

Special-purpose. Loaded by `verifier.py` via `load_gl_binaries` (with fallback). Covers every expression the verifier may encounter in any chapter file.

Crucial fields per expression:

- `input_args` / `output_args` — the verifier's `_find_digit_args` and `_find_immutable_args` (`verifier.py`) consult these to replicate the prover's digit/immutable computation.
- `definition_sets` — for anchor-handling type validation.

Top-level fields beyond per-expression entries:

- `external_anchors` — list of anchor-expression names whose externally-provided theorems should receive the same v/V → w/W bound-variable rename treatment as internal-theorem citations. Currently `["AnchorPeano", "AnchorGauss"]`. Read by `configuration_reader.py` (`Configuration.external_anchors`); consumed by `process_proof_graphs.py::rename_external_theorem` to gate which external batches' citations get renamed (other anchors fall through unchanged). See [`10_pipeline/06_process_proof_graph.md`](10_pipeline/06_process_proof_graph.md) §External-theorem citation renaming.

Unlike per-batch configs, `ConfigVisu.json` is agnostic to which batch produced the chapters — it's loaded once, used against any chapter.

Per the project conventions's "ConfigVisu.json reference":

- `in2` → output index 1
- `in3` → output index 2
- `residual` → output index 2
- `fold` → output index 6
- `in` → input indices `[0]`
- `=` → input indices `[0, 1]`
- `in2` → input indices `[0]`
- `in3` → input indices `[0, 1]`
- `residual` → input indices `[0, 1]`
- `interval` → input indices `[0, 1]`
- `fold` → input indices `[0, 1]`
- `preorder` → input indices `[0, 1]`

---

## tag_descriptions.json

32-line file at `files/config/tag_descriptions.json`. Flat object keyed by tag name; values are human-readable descriptions of each tag.

**Note: documentation-only mirror.** No code reads this JSON file. The HTML legend at `tags.html` and the right-click popup on tag badges are driven by the in-code constant `TAG_DESCRIPTIONS` in [`generate_full_proof_graph.py`](../generate_full_proof_graph.py) (≈line 141). Per the convention, the JSON mirrors the in-code dict 1:1 — same keys, plain-text equivalents of the same descriptions — so the two stay synchronised by hand. When updating tag descriptions, edit `TAG_DESCRIPTIONS` first (the canonical source for HTML rendering) and propagate to the JSON.

Current entries (30):

- `anchor handling`
- `contradiction`
- `disintegration`
- `equality1`
- `equality2`
- `expansion`
- `expansion for integration`
- `externally provided theorem`
- `implication`
- `incubator back reformulation`
- `multiplied from`
- `or branch assumption`
- `or branch proven`
- `or convergence`
- `or disintegration`
- `or theorem`
- `origin`
- `premise element`
- `recursion`
- `reformulated from`
- `reformulation for integration and`
- `reformulation for integration >[bound]`
- `reformulation for integration >[]`
- `symmetry of equality`
- `symmetry of inequality`
- `task formulation`
- `theorem`
- `vacuous truth`
- `validity name`
- `variable copy`

The five entries the previous draft listed as "missing" (`symmetry of inequality`, `variable copy`, `or theorem`, `vacuous truth`, plus `or branch proven` / `or branch assumption`) all landed. The retired `necessity for equality (hypo)` entry has also been removed.

`tag_descriptions.json` now mirrors the verifier's tag registry minus the meta-tags that don't surface in chapter rows (`definition set consistency` is one such — emitted as a per-row meta-check, not a row tag).

---

## Loader cartography

Which code reads which field.

### C++ prover (`prover.cpp`, entry around `:120–200`)

- Opens `Config<Tag>.json` based on `anchorID` passed to `ExpressionAnalyzer` constructor.
- Reads `prover_parameters` sub-object (`:129–162`).
- Reads `anchor_name` override (`:213`).
- Builds `coreExpressionMap` via `ce::modifyCoreExpressionMap(anchorID)` — this reads every per-expression entry.
- Populates `operators` set from `coreExpressionMap` (`:194–200`).

### C++ conjecturer (`conjecturer.cpp`, entry `loadConfiguration` at `:593`)

- Opens the same `Config<Tag>.json`.
- Reads `parameters` sub-object (`:616–651`).
- Reads `patterns_to_exclude`, `only_in_head`, `prohibited_combinations`, `prohibited_heads` arrays (`:653–686`).
- Reads `theorems_folder` override (`:688`).
- Inherits per-expression metadata via the shared `coreExpressionMap`.

### C++ run-mode orchestrator (`run_modes.cpp–124`)

- Opens `Config<Tag>.json` independently.
- Reads `theorems_folder`, `background_theorems_folder`, `raw_proof_graph_folder` path overrides.
- Reads `prover_parameters.ban_disintegration` and `prover_parameters.incubator_mode` — to decide whether to skip the compressor.

### Python stages (`configuration_reader.py`)

- Loads `ConfigVisu.json` via `configuration_reader.py` (≈550 lines).
- Exposes `input_args`, `output_args`, `indices_input_args`, `indices_output_args`, `definition_sets`, `theorems_folder`.
- Consumed by `process_proof_graphs.py`, `generate_full_proof_graph.py`, `verifier.py`.

### Python verifier (`verifier.py`)

- Loads `ConfigVisu.json` via `configuration_reader`.
- Loads per-batch `GL_binary_<Tag>.json` from `files/GL_binaries/` — regenerated by the compiler each run.

---

## Adding a new config

To add a new batch — say, `ConfigEuclid.json`:

1. Copy `ConfigGauss.json` as a starting point (richer feature set).
2. Change `anchor_name` (optional) and ensure the default anchor derivation maps to the intended `AnchorEuclid` (if you're creating a new anchor — see [`20_core_concepts/06_anchors_and_scopes.md`](20_core_concepts/06_anchors_and_scopes.md)).
3. Adjust the per-expression entries to include every expression the batch will use. Cross-reference `ConfigVisu.json` — the verifier must have the same expressions registered.
4. Tune `parameters` — especially `max_values_for_def_sets` / `max_values_for_uncomb_def_sets` — keeping each arg's `(uncomb + comb) ≤ 3` to avoid the `createMapAnchor` RAM explosion.
5. Tune `prover_parameters` — `maxIterationNumberProof`, `numberIterationsConjectureFiltering`, and the various `max_*` caps control runtime.
6. Run `main.py --run-descriptor euclid` through the full pipeline and inspect the Euclid batch in ; never invoke a proof batch directly because it lacks the preceding pipeline state.

To add a new expression to an existing config:

1. Add a per-expression entry with the fields from [per-expression schema](#per-expression-schema).
2. Add a matching entry to `ConfigVisu.json` so the verifier can consume chapters that mention the new expression.
3. Add a definition file to `files/definitions/` if the body is complex; inline via `full_mpl` otherwise.
4. Rebuild (`gl_quick.exe` kill + MSBuild) — no code change needed, the new expression is picked up at runtime.

---

## Weaknesses

### Known & tracked

- **`tag_descriptions.json` out of date.** Missing `symmetry of inequality`, `variable copy`, `or theorem`, `vacuous truth`; includes retired `necessity for equality (hypo)`. Non-load-bearing but visible to HTML viewers.
- **`createMapAnchor` RAM explosion unguarded.** Config misconfiguration (`rightMax > 3`) crashes with an OOM, not a helpful assert.

### Suspected fragility

- **No JSON schema validation.** Unknown fields are silently ignored. A typo (`prove_parameters` instead of `prover_parameters`) leaves defaults untouched and the intended override invisible. Detection requires reading the code path that consumes the field and knowing it's expected.
- **Field-name drift across code paths.** `conjecturer.cpp` reads `parameters`; `prover.cpp` reads `prover_parameters`; `run_modes.cpp` reads paths. A hypothetical refactor that renames `prover_parameters` must update both `prover.cpp` and `run_modes.cpp` or silent drift ensues.
- **Default-zero semantics.** Several integer fields default to `0`. `0` sometimes means "no cap" and sometimes means "cap at zero = reject all". Context-dependent. Reading the consumer code is required to understand intent.
- **Three separate config reads for one file.** `prover.cpp`, `conjecturer.cpp`, and `run_modes.cpp` each open `Config<Tag>.json` independently. File-system consistency (the file doesn't change between reads) is assumed but not asserted.
- **Per-expression fields are strict-typed but not validated.** `definition_sets` expects an object with specific structure; malformed entries silently produce nonsense.

### Not exercised by tests

- **Config-variant invariants.** No test asserts "every expression in `ConfigPeano` is also in `ConfigVisu`", or "every expression in `ConfigVisu` has matching `input_args` across configs". These invariants hold by maintenance discipline, not by verification.
- **`prover_parameters` coverage.** The defaults in `parameters.hpp` are reasonable, but no test verifies that the defaults produce a working batch — test coverage is indirect via the full-pipeline runs.
- **`anchor_name` override on a non-incubator config.** The override is only exercised by incubator configs currently. A main-pipeline batch using the override would work in principle but is untested.

---

## Open questions

- **OPEN-CFG-1 — RESOLVED.** `skip_eq_classes` (default `false`, declared at [`parameters.hpp`](../GL_Quick_VS/GL_Quick/src/parameters.hpp), loaded at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)) gates the equality-registration paths in `addStatement` at [`prover.hpp, 5662, 5712`](../GL_Quick_VS/GL_Quick/src/prover.hpp). When `true`: receipt of `(=[a,b])` does not update `equivalenceClassesMap`; the one-sided `!(=[a,b])` expansion through class members is skipped. Effect: the prover still emits symmetric `(=[b,a])` mirrors but does not propagate equality to dependent rules. Used by incubator configs where the combinatorial blow-up from class-propagation is not worth the proof-completeness trade-off — the incubator only needs ground-level facts, not derived-through-equality conclusions.
- **OPEN-CFG-2 — RESOLVED (dead field).** `fact_variable_kinds` is declared at [`conjecturer.hpp`](../GL_Quick_VS/GL_Quick/src/conjecturer.hpp) and loaded at [`conjecturer.cpp–647`](../GL_Quick_VS/GL_Quick/src/conjecturer.cpp), but a grep finds **no consumer** — the vector is populated from config and never read. Either a planned feature whose consumer code was not landed, or a dead field left over from a refactor. Safe to ignore in config; candidates for removal in a future cleanup pass.
- **OPEN-CFG-3 — RESOLVED.** Both fields are consumed:
 - `operator_threshold` — used at [`conjecturer.cpp–2599`](../GL_Quick_VS/GL_Quick/src/conjecturer.cpp) and [`:3580–3581`](../GL_Quick_VS/GL_Quick/src/conjecturer.cpp) in two filter-cascade sites. Gating condition: `opExprs2.size >= operator_threshold || (opExprs2.size == operator_threshold - 1 && freeArgs)` — i.e. an operator-count threshold that allows passage when enough operators are present (or enough-minus-one + free args left to fill).
 - `max_size_mapping_def_set` — passed directly to `createMap(N)` at [`conjecturer.cpp`](../GL_Quick_VS/GL_Quick/src/conjecturer.cpp) as the `N` parameter, controlling the size of the mapping table the conjecturer materialises for def-set combinations.
- **OPEN-CFG-4 — RESOLVED.** `.bak` files are not read. The pipeline loads configs by **exact filename** (`Config<Tag>.json` — constructed via string concatenation from the anchor tag), never by glob. There is no directory-scan that would pick up `.json.bak` variants. So `ConfigIncubatorGauss.json.bak` is effectively invisible to the pipeline — a manual backup kept for recovery, not consumed by any code path.

---

## See also

- [`03_mpl.md`](03_mpl.md) — MPL expressions the configs describe.
- [`10_pipeline/01_mpl_definitions.md`](10_pipeline/01_mpl_definitions.md) — how configs are consumed during compilation.
- [`10_pipeline/02_conjecturer.md`](10_pipeline/02_conjecturer.md) — conjecturer parameter effects.
- [`10_pipeline/03_ce_filter.md`](10_pipeline/03_ce_filter.md) — CE-filter parameters.
- [`10_pipeline/09_incubator.md`](10_pipeline/09_incubator.md) — incubator config variants.
- [`20_core_concepts/06_anchors_and_scopes.md`](20_core_concepts/06_anchors_and_scopes.md) — anchor-name override mechanics.
- — anchor-expansion recipe, including config-sizing cautions.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
