<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Pipeline · Stage 6 — Process proof graph `[DRAFT]`

> **Input:** `files/raw_proof_graph/*.txt` (prover output from stage 5).
> **Output:** `files/processed_proof_graph/*.txt` (tab-separated chapters + `global_theorem_list.txt` + `external_theorems.txt`).
> **Owner:** `process_proof_graphs.py` (≈600 lines).
> **Entry:** `create_processed_proof_graph(config, …)` at [`process_proof_graphs.py`](../../process_proof_graphs.py).

---

## What this stage does

The prover's raw per-theorem chapters are full of internal names — `x<N>` x-prefixed anchor-slot variables, `repl_lev_*_*` substitution-chain placeholders, `_copy` suffixes mixed in, and raw integer bound-variable indices from the conjecturer. Raw chapters are not intended to be read by either the verifier or the HTML generator — they need to be:

1. Pruned — remove chapters for theorems that did not survive the compressor.
2. Renamed — translate internal names to a stable `v1`, `v2`, …, `N`, `i0`, `s`, `+`, `*`, `i1`, …, `<N>_copy` scheme.
3. Canonicalised — reorder rows, strip anchor-variable lists from outer implications where they are redundant.
4. Tag-extended — associate each chapter with its theorem entry in `global_theorem_list.txt` so the verifier can match chapters to rows.

Stage 8 is where all of that happens. Output artefacts:

- `files/processed_proof_graph/<N>_direct_proof.txt` — one per direct-proof theorem.
- `files/processed_proof_graph/<N>_check_zero.txt` + `<N+1>_check_induction_condition.txt` + `<N+2>_induction_typing.txt` — per induction theorem (induction-typing chapter new on the current branch; see [I-18](../30_invariants.md#i-18)).
- `files/processed_proof_graph/<N>_reformulated_statement.txt` — per reformulation.
- `files/processed_proof_graph/<N>_back_reformulated_statement.txt` — per back-reformulation (incubator).
- `files/processed_proof_graph/<N>_or_theorem.txt` — per OR theorem.
- `files/processed_proof_graph/global_theorem_list.txt` — the authoritative theorem registry (one row per theorem).
- `files/processed_proof_graph/external_theorems.txt` — three shapes per external (raw, v/V renamed, w/W citation form). The verifier's direct-string + reverted-string + fuzzy-fold lookup paths all hit cleanly. See *External-theorem citation renaming* below.

---

## The four-priority renaming scheme

This is the load-bearing algorithm of stage 6. Every raw variable in a chapter needs to be translated to its processed form deterministically. The four priorities run in order — each fills in `repl_map` entries without overwriting earlier ones.

### Priority 1 — anchor mapping from the theorem expression

For each chapter, look up the theorem it proves. Call `get_anchor_mapping_from_expr(theorem, config)` to pre-populate the repl-map with anchor-slot names: e.g. raw `1 → N`, `2 → i0`, `3 → s`, `4 → +`, `5 → *`, `6 → i1`.

Anchor variables have fixed, config-dictated names; they must never be renumbered.

*Seen:* [`process_proof_graphs.py`](../../process_proof_graphs.py).

### Priority 2 — left-to-right scan of the theorem expression

Walk the arguments of the theorem expression in order. Each non-anchor variable not already in `repl_map` is minted with a **type-aware case**: `v<digit_counter>` if the position's defset is `(1)` (digit / element of N) and `V<set_counter>` if the defset is `P(...)` (set). Two independent counters; both start at 1.

The defset is read from the operator's config entry (`config[op].definition_sets[1-based-position]`) by the helper `classify_args_by_defset` ([`process_proof_graphs.py`](../../process_proof_graphs.py)). Classification is computed ONCE per chapter from the union of the theorem expression and every chapter cell, so a variable that appears in multiple positions across the chapter is consistently digit OR consistently set; a true digit-vs-set conflict in the same chapter triggers an assert.

**This is the seeding step that matters.** Without it, chapters could assign `v1` to a variable that appears first in chapter lines but is a *different* raw variable from the `v1` in `global_theorem_list.txt`. The verifier's theorem-match and task-formulation checks would then spuriously fail. See [I-10](../30_invariants.md#i-10).

*Seen:* [`process_proof_graphs.py`](../../process_proof_graphs.py).

### Priority 2.5 — anchor-handling x-prefixed vars

Anchor-handling rows emit `x<N>`-prefixed variables — the raw-form names the prover uses for anchor-slot copies before they are resolved. Priority 2.5 scans chapter rows tagged `anchor handling`, matches the anchor line against the config's `short_mpl_raw`, and where a `t_arg` is `x`-prefixed and its corresponding `c_arg` is `i<digit>`, assign:

```python
repl_map[t_arg] = f"{digit}_copy"
```

That is: raw `x<k>` mapped to config slot `i<digit>` becomes processed `<digit>_copy`.

*Seen:* [`process_proof_graphs.py`](../../process_proof_graphs.py).

### Priority 3 — iterate over chapter lines

Walk every row in the stack, every cell in each row (skipping tag-name cells and `"main"` namespace cells). For any argument not already in `repl_map`, mint via `_mint_typed_var` ([`process_proof_graphs.py`](../../process_proof_graphs.py)) using the SAME `digit_counter` / `set_counter` pair started in Priority 2 — the pair is shared so the v/V numbering grows monotonically across both priorities.

**Skip theorem implications** that are cited as justifications — these get replaced wholesale later with the globally-renamed form from `global_theorem_list.txt`, so their bound vars don't participate in chapter v-numbering.

*Seen:* [`process_proof_graphs.py`](../../process_proof_graphs.py).

### Priority 4 — `_copy` derivation

Any raw variable ending in `_copy` must be renamed by appending `_copy` to the renamed base:

```python
for a in list(repl_map.keys()):
    if a.endswith("_copy"):
        base = a[:-5]
        if base in repl_map:
            repl_map[a] = repl_map[base] + "_copy"
```

So raw `repl_lev_3_2_copy` → if raw `repl_lev_3_2` was renamed to `v11`, the `_copy` form becomes `v11_copy`. Not `v12_copy`, not `v11_c1`, not anything else. This determinism is what lets the verifier's `_copy`-aware variable-substitution logic (`_replace_arg_safe` + the `_copy` suffix handling in `check_anchor_handling`) line up.

*Seen:* [`process_proof_graphs.py`](../../process_proof_graphs.py).

---

## Pass 2 — w/W rename of bound vars whose raw form is `\d+`

After the four-priority Pass-1 v/V rename, an additional pass runs over every chapter cell whose raw form contains `(>[`. For each non-anchor `(>[bvars]…)` the bvars are partitioned by the SHAPE of their RAW name (looked up by walking the RAW cell and the current cell in parallel through their `(>[` brackets, position-by-position):

- raw shape matches `^\d+$` (the prover's freshly-minted bound-var indices) → renamed to `w<N>` (digit) / `W<N>` (set), with a counter that **starts at 1 per cell** and grows monotonically by unique name within the cell.
- any other raw shape (`repl_lev_*`, `it_0_lev_*`, `_copy`, …) → keeps the Pass-1 v/V name with the chapter-global counter.

Free variables and anchor-implication bvars are not touched. Theorem-anchor-implications stay skipped here too — they get replaced wholesale in the global-theorem step below.

**Why the partition.** The prover's raw `\d+` bound-var indices have no chapter-global meaning — they're locally fresh per implication and exist only to disambiguate the bvar slot from anything outside the implication. Renaming them per cell with a fresh w-counter is faithful and makes them visually distinct from chapter-global v/V free-var slots in the rendered proof.

The other raw bvar shapes — especially the `repl_lev_*` family used in integration scopes — are not freely chosen. The prover deliberately gives them the same v-name (after Pass 1) as the boundary-scope free variable they bind to, so that `premise element` and `validity name` checks can match by exact-string equality. Renaming those would break the pinning.

**Why per-cell monotonic-by-unique-name.** When the prover reuses the same raw `\d+` across nested `>[…]` of the same cell (intentional shadow encoding variable identification — `multiplied from` rule cells), the same raw name maps to the same w-name. Shadow is preserved; the verifier's `check_equalize_variable` walk-pairs-and-build-mapping logic still satisfies.

**Why per-cell restart at 1.** Counter is local to one implication tree (one cell call). Different cells start fresh, so each rule reads as `w1, w2, …` from its own `>[…]` outward. Co-derived alpha-equivalent cells (e.g. two `disintegration` siblings of the same existence form) stay distinct because their raw `\d+` bvars sit inside structurally distinct contexts — the partition pulls them apart at the raw layer, not the renamed layer.

**Verifier counterpart.** `verifier.py`'s normalizing helpers (`_normalize_expr_list`, `_normalize_implication`, the fresh-index scanner in `_substitute_signature`, `_reconstruct_implication`'s candidate filter) widen their regex from `[vV]\d+` to `[vVwW]\d+` so a chapter cell rendered with w-bvars canonicalises to the same form as a v-form reconstruction built from the theorem. `_alpha_canonicalize_bound_vars` was already alpha-blind (it collects every name that appears inside any `>[…]` regardless of letter prefix) and needs no change.

*Seen:* `w_rename_impl_local` and `_collect_non_anchor_bvar_lists` in [`process_proof_graphs.py`](../../process_proof_graphs.py); ITERATION 4½ in `create_processed_proof_graph`.

---

## Pass 3 — `v→w` rename of cited theorem-anchor implications

After ITERATION 4 wholesale-replaces theorem-anchor implication cells with their `global_theorem_list.txt` form (v/V), a follow-on pass (ITERATION 5) applies a case-preserving `v→w / V→W` swap to every `[vV]\d+` token in those cells — but only when:

- the cell is at column index ≥ 3 (the rest fields, i.e. cited dependencies — never the chapter's own HEAD claim at column 0), and
- the cell is a theorem-anchor implication (passes `is_theorem_anchor_implication`).

Anchor variables (`N`, `i0`, `s`, `+`, `*`, `i1`, `i2`, `id`,...) are never `[vV]\d+`, so the swap leaves them untouched. The lookarounds `(?<=[\[,])` / `(?=[\],])` confine matches to argument positions, so operator names never mis-match. `<digit>_copy` forms (e.g. `v1_copy`) are excluded because the lookahead requires `,` or `]` immediately after the digits.

**Why.** Cited foreign-theorem inner bvars used to render with the same `v / V` letter as chapter-global free variables, making it ambiguous whether `v3` in a chapter line was a chapter-local free variable or a bound variable scoped inside an applied theorem. Reserving `w / W` for the citation case removes that ambiguity at sight.

**Why this asymmetry between the chapter HEAD and rest cells.** The HEAD column of reformulation / OR-theorem chapters *is* a theorem-anchor implication — it is the theorem this chapter "owns" and proves. Rendering the HEAD in the same `v / V` form as `global_theorem_list.txt` keeps title/registry parity. The rest cells are *citations of foreign theorems* and get the `w / W` decoration.

**Verifier counterpart.** `verifier.py` defines a symmetric inverse `_revert_w_to_v_in_theorem_citation`. At every site that performs an exact-string membership test against `state.global_theorems` / `state.external_theorems` (`check_theorem_tag`, `check_externally_provided_theorem`, the origin meta-check at `rest[0]` for `implication` / `multiplied from` / `reformulated from` rows), the verifier reverts `w → v` on theorem-anchor citations before the lookup. The fold-based fallback (`_normalize_expr_list`, `_alpha_canonicalize_bound_vars`) — already widened to `[vVwW]\d+` — provides a safety net.

`check_implication` needs no change: its anchor branch uses `_normalize_all_vars_in_list`, which folds *every* argument token to `v1, v2,...` order-wise — so `w` and `v` collapse identically. Its non-anchor branch only ever sees non-anchor implications, which never carry the citation `w / W` (those are products of disintegration, handled by Pass 2).

**No renumbering — pure letter swap.** `v3 → w3`, `V5 → W5`. The digit is preserved, so a chapter row's cited theorem and its registry counterpart share the same numbering. The verifier's revert is the symmetric inverse.

*Seen:* `_v_to_w_in_theorem_citation` and ITERATION 5 in `create_processed_proof_graph` ([`process_proof_graphs.py`](../../process_proof_graphs.py)); `_revert_w_to_v_in_theorem_citation` and `_is_theorem_anchor_impl_local` in [`verifier.py`](../../verifier.py).

---

## External-theorem citation renaming

Until commit, externally-provided theorems (theorems imported from a sibling pipeline's `compressed_external_theorems.txt`, e.g. main-pipeline citing incubator output or vice versa) fell through to the chapter-local `repl_map`'s anchor-slot-style minting. This produced "penniless orphan" rendering: an incubator chapter citing a Peano-anchor external theorem would show that theorem with raw `i2`/`i3` slot names instead of the v/V → w/W convention used for internal theorems.

The fix introduces `rename_external_theorem(raw_expr, config)` (`process_proof_graphs.py`):

- Identifies the external's anchor by walking its outermost `(>[...]` for `(Anchor<Tag>[...])`.
- Filters via `config.external_anchors` (a top-level list in `ConfigVisu.json` — see [`04_configs.md`](../04_configs.md#configvisujson-the-verifiers-config)). Currently `["AnchorPeano", "AnchorGauss"]`. Anchors not in the list fall through unchanged — the historical "rename via chapter-local repl_map" behaviour is preserved for unrecognised externals.
- For declared externals: maps raw anchor args to the anchor's canonical slot names from `config[anchor_name].short_mpl_raw` (e.g. for `AnchorPeano`: `1→N, 2→i0, 3→s, 4→+, 5→*, 6→i1`).
- Mints typed v/V names for non-anchor bound variables via `classify_args_by_defset` + `_mint_typed_var` — the same machinery used by Pass 1 / Pass 2, so external citations land in the same numbering scheme as internal ones.
- Returns the renamed expression. ITERATION 5 (Pass 3) then applies the same `v→w / V→W` swap to the citation form, yielding the w/W convention end-to-end.

Pipeline integration in `create_processed_proof_graph`:

- Loads `compressed_external_theorems.txt` once before ITERATION 4 into `raw_external_set`.
- Builds `external_renames = {raw: v/V_form}` for each external whose anchor is recognised.
- ITERATION 4 chapter-cell substitution gets a new branch: if `orig_cell ∈ external_renames`, use the v/V form. Otherwise the prior fallback (chapter-local repl_map) still applies.
- The post-iteration external-theorem collection emits `external_theorems.txt` carrying all three shapes (raw, v/V, w/W) — the verifier's three lookup paths each find a hit.

**Visual confirmation** on `1209_direct_proof.txt` (incubator chapter heavily citing Peano externals):

```text
before: (>[N,i0,s](AnchorPeano[..])(>[i2](in[i2,N])(>[]!(=[i2,i0])(existence2[N,i2,s]))))
after:  (>[N,i0,s](AnchorPeano[..])(>[w1](in[w1,N])(>[]!(=[w1,i0])(existence2[N,w1,s]))))
```

`i2` (a chapter-local mint slot) → `w1` (the w/W citation form for non-anchor bound variables in cited theorems).

*Seen:* `rename_external_theorem` and the `external_renames` plumbing in `create_processed_proof_graph` ([`process_proof_graphs.py`](../../process_proof_graphs.py)).

---

## Global theorem list renaming

Each row of `global_theorem_list.txt` — `(expression, method, reference)` — is renamed independently using Priorities 1–2 only (no chapter lines involved at this stage):

```python
thm_repl_map = get_anchor_mapping_from_expr(raw_thm_expr, config)
thm_arg_type = classify_args_by_defset(raw_thm_expr, config)
thm_counters = {"digit": 1, "set": 1}
for a in get_all_args(raw_thm_expr):
    if a not in thm_repl_map:
        thm_repl_map[a] = _mint_typed_var(a, thm_arg_type, thm_counters)
renamed_theorems[i][0] = replace_keys_in_string(raw_thm_expr, thm_repl_map)

# Induction variable (col 3) renamed with the same map
if method == "induction":
    renamed_theorems[i][2] = thm_repl_map.get(raw_ind_var, raw_ind_var)
```

Seeding from the theorem expression (same left-to-right scan, same type-aware case rule) guarantees the chapter's Priority-2 output matches. This is the *shared seed* that keeps chapter and theorem-list in sync — part of [I-10](../30_invariants.md#i-10).

---

## Anchored implications are replaced wholesale

Implications cited as `theorem` justifications — e.g. chapter row 2 of `0_direct_proof.txt`:

```text
(>[s,+,i1](AnchorPeano[N,i0,s,+,*,i1])(>[v1,v2](in2[v1,v2,s])(in3[v1,i1,v2,+])))	main	theorem
```

are NOT renamed via the chapter's repl_map. They are replaced verbatim with the globally-renamed form from the theorem list. This keeps chapter citations aligned with the authoritative theorem registry regardless of each chapter's local v-numbering.

`is_theorem_anchor_implication(expr)` at [`process_proof_graphs.py`](../../process_proof_graphs.py) detects these implications (outer `(>[...]` carrying an `(Anchor<Tag>[...])` premise).

---

## Pruning

`_prune_proof_graph(raw_theorems, raw_stacks, theorems_dir)` at [`process_proof_graphs.py`](../../process_proof_graphs.py) is the pruning step: drop chapters for theorems that did not survive the compressor. The compressor's output (the compressed `theorems.txt`) is the authoritative survivor set; chapters for non-survivors are removed before renaming.

---

## Anchor-implication normalisation

`normalize_anchor_implications(raw_theorems, raw_stacks)` at [`process_proof_graphs.py`](../../process_proof_graphs.py) handles a specific canonicalisation: when the outermost `(>[...]` of a theorem carries anchor variables in the bound list, those are stripped (the anchor application at the outermost position supplies them already). See [I-11](../30_invariants.md#i-11).

---

## Induction-triad detection

The verifier expects induction theorems to produce three consecutive chapter files in strict order:

```
<N>_induction_typing.txt
<N+1>_check_zero.txt
<N+2>_check_induction_condition.txt
```

The processor emits these chapters with consecutive numbers by design. The induction-typing chapter is new on the current branch — part of the fix in [`docs/agentic_swdd/induction_typing_plan.md`](../induction_typing_plan.md). On older branches, only the zero + condition pair exists (two consecutive chapters). The verifier's `build_chapter_theorem_map` asserts the triad by filename pattern — a mismatch raises `AssertionError`.

---

## Output format — row structure

Each chapter file is tab-separated. Each non-empty line:

```text
expression \t namespace \t tag \t [rest fields...]
```

The rest fields alternate `expression \t namespace \t expression \t namespace \t...`, citing dependencies and arguments specific to the tag.

Example row (from `0_direct_proof.txt`):

```text
(in3[i0,i1,v1,+])	main	implication	(>[s,+,i1](AnchorPeano[N,i0,s,+,*,i1])(>[v1,v2](in2[v1,v2,s])(in3[v1,i1,v2,+])))	main	(AnchorPeano[N,0_copy,s,+,*,i1])	main	(in2[i0,v1,s])	main
```

Decoded:

- Head: `(in3[i0,i1,v1,+])` — what this row asserts.
- Namespace: `main`.
- Tag: `implication` — fired an inference rule.
- Rest fields — alternating `(expr, ns)` pairs:
 - `(>[s,+,i1](AnchorPeano[…])(>[v1,v2](in2[…])(in3[…])))` in `main` — the rule that fired.
 - `(AnchorPeano[N,0_copy,s,+,*,i1])` in `main` — the anchor premise that matched.
 - `(in2[i0,v1,s])` in `main` — the other premise that matched.

Note the `0_copy` — that is the Priority 2.5 rewrite of a raw `x`-prefixed anchor variable.

---

## `global_theorem_list.txt` — the theorem registry

Three columns:

```
theorem_expression \t method \t reference
```

Methods: `direct | induction | reformulated statement`. Reference: `-1` for direct, induction variable name for induction, source-theorem expression for reformulated.

Representative rows (from the current branch's `global_theorem_list.txt`):

```text
(>[i0,s,+,i1](AnchorPeano[N,i0,s,+,*,i1])(>[v1](in2[i0,v1,s])(in3[i0,i1,v1,+])))	direct	-1
(>[i0,s,+,i1](AnchorPeano[N,i0,s,+,*,i1])(>[v1](in2[v1,i0,s])(in3[i1,v1,i0,+])))	induction	v1
(>[N,i0,s,+](AnchorGauss[N,i0,s,+,*,i1,i2,id])(>[v1,v2](in2[v1,v2,s])(>[v3](interval[N,+,i0,v2,v3])(existence4[N,+,v3,v1,i0]))))	reformulated statement	(>[1,2,3,4](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10,11](limitSet[1,4,9,10,11])(>[12](in2[10,12,3])(>[](interval[1,4,2,12,9])(interval[1,4,2,10,11])))))
```

The `reformulated statement` reference field uses the *raw integer-indexed* form of the source theorem — this is what the verifier's reformulation checker uses to match shapes.

---

## Weaknesses

### Known & tracked

- **Induction triad ordering is asserted, not defended.** The verifier raises `AssertionError` on malformed numbering. Stage 8 is the single producer of these filenames, so the invariant holds by construction — but a regression in the filename numbering would take down the verifier rather than degrade gracefully.

### Suspected fragility

- **Regex-based anchor parsing.** `get_anchor_mapping_from_expr` uses regex against the theorem expression to identify the anchor name and slot mapping. A theorem that contains the substring `Anchor` in a non-anchor position (unlikely in current codebase, but not ruled out) would confuse it.
- **`x`-prefixed anchor-var detection.** Priority 2.5 assumes anchor-handling rows use `x<N>` names. A rename of that convention upstream would silently break `_copy` derivation — the Priority-4 step then generates incorrect `_copy` names.
- **`replace_keys_in_string`'s safety.** Priority 2 replaces `raw_name → v<k>`. The replacement is done via regex with lookarounds (similar to verifier's `_replace_arg_safe`), but the implementation differs. A raw name that is a substring of another raw name (e.g. `v1` vs `v10`) would break without proper lookarounds. Current code uses argument-boundary lookarounds, but adding a new chapter format that could alter boundaries would need re-validation.

### Not exercised by tests

- **Cross-chapter repl-map consistency.** No test asserts that two chapters' `repl_map`s are *consistent* with the same theorem when they both reference it. Practically: the Priority-2 seeding means they should be, but a regression in seeding would only show up as a verifier failure on the affected chapters.

---

## Open questions

- **OPEN-14 — RESOLVED.** The `x`-prefix emission site is at [`prover.cpp–3449`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) in the anchor-handling step. For each slot in `anchorInfo.definitionSets` whose type is `(1)`:
 - Compute `index = std::stoi(slot) - 1` (1-based → 0-based).
 - If `args[index]` already starts with `"x"`, return (already prefixed — no double-prefix).
 - Otherwise, register the replacement `args[index] → "x" + args[index]` in `replacementMap`.

 The resulting `replacedAnchor` is deposited into the memory block's local and global statement registries. This is the *prover* side of the convention; the *processor* side (Priority 2.5 above) then renames each `x<N>` back to `<digit>_copy` based on the anchor's slot-digit. Changing the convention would require coordinated edits at both sites.
- **OPEN-15 — RESOLVED.** A chapter-less survivor would appear in `global_theorem_list.txt` but have no chapter file referring to it. The chapter iteration in `process_proof_graphs.py` (at `:349-375`) constructs `fname_to_raw_thm` by mapping expected filenames (per theorem method) to theorem expressions — so the map covers every survivor, but if the file doesn't exist on disk, no chapter-processing call visits it. Downstream: the HTML generator's per-file render loop similarly skips non-existent files; the verifier's chapter-glob at `verifier.py` also skips non-existent files (glob only returns existing ones). The theorem is effectively invisible to chapter-level audit — present in the registry but unverifiable. Detection: run the verifier and check whether the theorem's rendered chapter appears in its expected row count; if a `method = induction` theorem has no accompanying `induction_typing/check_zero/check_induction_condition` triad, the induction-triad assertion in `build_chapter_theorem_map` (at [`verifier.py`](../../verifier.py)) raises `AssertionError` loudly — but for `method = direct` / `reformulated`, no such assertion exists, so the miss is silent. Hardening: add a cross-check that every `global_theorem_list.txt` row has a corresponding chapter file.

---

## See also

- [`10_pipeline/07_html_export.md`](07_html_export.md) — downstream consumer.
- [`10_pipeline/08_verifier.md`](08_verifier.md) — downstream consumer + chapter-format enforcer.
- [I-10](../30_invariants.md#i-10), [I-11](../30_invariants.md#i-11) — renaming invariants.
- [`20_core_concepts/08_proof_tags.md`](../20_core_concepts/08_proof_tags.md) — tag vocabulary.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
