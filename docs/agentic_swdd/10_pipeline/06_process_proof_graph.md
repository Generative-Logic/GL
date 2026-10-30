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
> **Owner:** `process_proof_graphs.py` and `create_processed_proof_graph`.
> **Entry:** `create_processed_proof_graph(config, …)` at [`process_proof_graphs.py`](../../../process_proof_graphs.py): `_prepare_processed_graph` (load, prune, rename, externals), then the book arrangement and the writes. Between the incubator and main batches, `create_processed_theorem_list` writes only the incubator's theorem list ([D-537](../40_decisions.md#d-537)).

---

## What this stage does

The prover's raw per-theorem chapters are full of internal names — `x<N>` x-prefixed anchor-slot variables, `repl_lev_*_*` substitution-chain placeholders, `_copy` suffixes mixed in, and raw integer bound-variable indices from the conjecturer. Raw chapters are not intended to be read by either the verifier or the HTML generator — they need to be:

1. Pruned — remove chapters for theorems that did not survive the compressor.
2. Renamed — translate internal names to a stable `v1`, `v2`, …, `N`, `i0`, `s`, `+`, `*`, `i1`, …, `<N>_copy` scheme.
3. Citation-normalised — rewrite theorem references into their global naming forms.
4. Scope-ordered for the main book — place every local producer before its consumer in reader order, independently inside each existing scope; keep the first goal row and all namespace positions.
5. Compacted for the main book — shorten checked equality derivations and remove unused ordinary main-proof leaves without merging subproofs.
6. Grouped and book-ordered — minimize the frozen layout objective, place cited theorems before consumers, preserve canonical proof identity, and renumber complete induction bundles together. Repeat the accepted main-book transformations until the whole book is unchanged; Incubator uses one bounded arrangement pass.
7. Hierarchically placed for main and incubator books — after compaction stabilizes, permute complete rows into reverse expanded order, keeping each subproof contiguous and recursively inside its parent.
8. Tag-extended — associate each chapter with its theorem entry in `global_theorem_list.txt` so the verifier can match chapters to rows.

Stage 6 performs this processing. Output artefacts:

- `files/processed_proof_graph/<N>_single_chapter_proof.txt` — one per single-chapter theorem.
- `files/processed_proof_graph/<N>_induction_typing.txt` + `<N+1>_check_zero.txt` + `<N+2>_check_induction_condition.txt` — per induction theorem (complete induction bundle; see [I-18](../30_invariants.md#i-18)).
- `files/processed_proof_graph/<N>_reformulated_statement.txt` — per reformulation.
- `files/processed_proof_graph/<N>_back_reformulated_statement.txt` — per back-reformulation (incubator).
- `files/processed_proof_graph/<N>_or_theorem.txt` — per OR theorem.
- `files/processed_proof_graph/global_theorem_list.txt` — the processed proof registry (one row per reader chapter), derived from the retained theorem set.
- `files/processed_proof_graph/external_theorems.txt` — three shapes per external (raw, v/V renamed, w/W citation form). The verifier's direct-string + reverted-string + fuzzy-fold lookup paths all hit cleanly. See *External-theorem citation renaming* below.

---

## The four-priority renaming scheme

This is the load-bearing algorithm of stage 6. Every raw variable in a chapter needs to be translated to its processed form deterministically. The four priorities run in order — each fills in `repl_map` entries without overwriting earlier ones.

### Priority 1 — anchor mapping from the theorem expression

For each chapter, look up the theorem it proves. Call `get_anchor_mapping_from_expr(theorem, config)` to pre-populate the repl-map with anchor-slot names: e.g. raw `1 → N`, `2 → i0`, `3 → s`, `4 → +`, `5 → *`, `6 → i1`.

Anchor variables have fixed, config-dictated names; they must never be renumbered.

*Seen:* [`process_proof_graphs.py`](../../../process_proof_graphs.py).

### Priority 2 — left-to-right scan of the theorem expression

Walk the arguments of the theorem expression in order. Each non-anchor variable not already in `repl_map` is minted with a **type-aware case**: `v<digit_counter>` if the position's defset is `(1)` (digit / element of N) and `V<set_counter>` if the defset is `P(...)` (set). Two independent counters; both start at 1.

The defset is read from the operator's config entry (`config[op].definition_sets[1-based-position]`) by the helper `classify_args_by_defset` ([`process_proof_graphs.py`](../../../process_proof_graphs.py)). Classification is combined once per chapter from the theorem expression and every chapter cell. The first concrete classification wins; an unknown classification can be replaced by a concrete one. Conflicting uses across implication scopes do not assert in this display-naming pass.

**This is the seeding step that matters.** Without it, chapters could assign `v1` to a variable that appears first in chapter lines but is a *different* raw variable from the `v1` in `global_theorem_list.txt`. The verifier's theorem-match and task-formulation checks would then spuriously fail. See [I-10](../30_invariants.md#i-10).

*Seen:* [`process_proof_graphs.py`](../../../process_proof_graphs.py).

### Priority 2.5 — anchor-handling x-prefixed vars

Anchor-handling rows emit `x<N>`-prefixed variables — the raw-form names the prover uses for anchor-slot copies before they are resolved. Priority 2.5 scans chapter rows tagged `anchor handling`, matches the anchor line against the config's `short_mpl_raw`, and where a `t_arg` is `x`-prefixed and its corresponding `c_arg` is `i<digit>`, assign:

```python
repl_map[t_arg] = f"{digit}_copy"
```

That is: raw `x<k>` mapped to config slot `i<digit>` becomes processed `<digit>_copy`.

*Seen:* [`process_proof_graphs.py`](../../../process_proof_graphs.py).

### Priority 3 — iterate over chapter lines

Walk every row in the stack, every cell in each row (skipping tag-name cells and `"main"` namespace cells). For any argument not already in `repl_map`, mint via `_mint_typed_var` ([`process_proof_graphs.py`](../../../process_proof_graphs.py)) using the SAME `digit_counter` / `set_counter` pair started in Priority 2 — the pair is shared so the v/V numbering grows monotonically across both priorities.

**Skip theorem implications** that are cited as justifications — these get replaced wholesale later with the globally-renamed form from `global_theorem_list.txt`, so their bound vars don't participate in chapter v-numbering.

*Seen:* [`process_proof_graphs.py`](../../../process_proof_graphs.py).

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

*Seen:* [`process_proof_graphs.py`](../../../process_proof_graphs.py).

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

*Seen:* `w_rename_impl_local` and `_collect_non_anchor_bvar_lists` in [`process_proof_graphs.py`](../../../process_proof_graphs.py); ITERATION 4½ in `create_processed_proof_graph`.

---

## Pass 3 — `v→w` rename of cited theorem-anchor implications

After ITERATION 4 wholesale-replaces theorem-anchor implication cells with their `global_theorem_list.txt` form (v/V), a follow-on pass (ITERATION 5) applies a case-preserving `v→w / V→W` swap to every `[vV]\d+` token in those cells — but only when:

- the cell is at column index ≥ 3 (the rest fields, i.e. cited dependencies — never the chapter's own HEAD claim at column 0), and
- the cell is a theorem-anchor implication (passes `is_theorem_anchor_implication`).

Anchor variables (`N`, `i0`, `s`, `+`, `*`, `i1`, `i2`, `id`,...) are never `[vV]\d+`, so the swap leaves them untouched. The lookarounds `(?<=[\[,])` / `(?=[\],])` confine matches to argument positions, so operator names never mis-match. `<digit>_copy` forms (e.g. `v1_copy`) are excluded because the lookahead requires `,` or `]` immediately after the digits.

**Why.** Cited foreign-theorem inner bvars used to render with the same `v / V` letter as chapter-global free variables, making it ambiguous whether `v3` in a chapter line was a chapter-local free variable or a bound variable scoped inside an applied theorem. Reserving `w / W` for the citation case removes that ambiguity at sight.

**Why this asymmetry between the chapter HEAD and rest cells.** The HEAD column of reformulation / OR-theorem chapters *is* a theorem-anchor implication — it is the theorem this chapter "owns" and proves. Rendering the HEAD in the same `v / V` form as `global_theorem_list.txt` keeps title/registry parity. The rest cells are *citations of foreign theorems* and get the `w / W` decoration.

**Theorem implications outside the theorem list.** ITERATION 4 substitutes a theorem-anchor implication cell only on an exact text hit in the raw theorem list (or the external renames). Every other one — an or theorem's source implication, or a compact implication's canonical reconstruction of a proved theorem in a different variable numbering (`implication<N>[]` expansion and compilation rows; I-52) — binds all its variables and is renamed on its own ([D-489](../40_decisions.md#d-489)) by `theorem_rename_map`, the same map that names the global theorem list (anchor slots by the anchor's configured names, other tokens by first appearance via `_mint_typed_var`). Alpha-variants of one theorem therefore get one display form. It is never renamed with the chapter's map: that map only covers the chapter's own variables, so tokens beyond them stayed raw numbers (F14's chapter cited row 110 with binders `13` and `14`, which the Lean export cannot type), and the names it did assign belong to unrelated chapter variables.

**Verifier counterpart.** `verifier.py` defines a symmetric inverse `_revert_w_to_v_in_theorem_citation`. At every site that performs an exact-string membership test against `state.global_theorems` / `state.external_theorems` (`check_theorem_tag`, `check_externally_provided_theorem`, the origin meta-check at `rest[0]` for `implication` / `multiplied from` / `reformulated from` rows), the verifier reverts `w → v` on theorem-anchor citations before the lookup. The fold-based fallback (`_normalize_expr_list`, `_alpha_canonicalize_bound_vars`) — already widened to `[vVwW]\d+` — provides a safety net.

`check_implication` needs no change: its anchor branch uses `_normalize_all_vars_in_list`, which folds *every* argument token to `v1, v2,...` order-wise — so `w` and `v` collapse identically. Its non-anchor branch only ever sees non-anchor implications, which never carry the citation `w / W` (those are products of disintegration, handled by Pass 2).

**No renumbering — pure letter swap.** `v3 → w3`, `V5 → W5`. The digit is preserved, so a chapter row's cited theorem and its registry counterpart share the same numbering. The verifier's revert is the symmetric inverse.

*Seen:* `_v_to_w_in_theorem_citation` and ITERATION 5 in `create_processed_proof_graph` ([`process_proof_graphs.py`](../../../process_proof_graphs.py)); `_revert_w_to_v_in_theorem_citation` and `_is_theorem_anchor_impl_local` in [`verifier.py`](../../../verifier.py).

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
- ITERATION 4 chapter-cell substitution gets a new branch: if `orig_cell ∈ external_renames`, use the v/V form. Otherwise the cell is renamed on its own by `theorem_rename_map` (below).
- The post-iteration external-theorem collection emits `external_theorems.txt` carrying all three shapes (raw, v/V, w/W) — the verifier's three lookup paths each find a hit.

**Visual confirmation** on `1209_single_chapter_proof.txt` (incubator chapter heavily citing Peano externals):

```text
before: (>[N,i0,s](AnchorPeano[..])(>[i2](in[i2,N])(>[]!(=[i2,i0])(existence2[N,i2,s]))))
after:  (>[N,i0,s](AnchorPeano[..])(>[w1](in[w1,N])(>[]!(=[w1,i0])(existence2[N,w1,s]))))
```

`i2` (a chapter-local mint slot) → `w1` (the w/W citation form for non-anchor bound variables in cited theorems).

*Seen:* `rename_external_theorem` and the `external_renames` plumbing in `create_processed_proof_graph` ([`process_proof_graphs.py`](../../../process_proof_graphs.py)).

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

Implications cited as `theorem` justifications — e.g. chapter row 2 of `0_single_chapter_proof.txt`:

```text
(>[s,+,i1](AnchorPeano[N,i0,s,+,*,i1])(>[v1,v2](in2[v1,v2,s])(in3[v1,i1,v2,+])))	main	theorem
```

are NOT renamed via the chapter's repl_map. They are replaced verbatim with the globally-renamed form from the theorem list. This keeps chapter citations aligned with the authoritative theorem registry regardless of each chapter's local v-numbering.

`is_theorem_anchor_implication(expr)` at [`process_proof_graphs.py`](../../../process_proof_graphs.py) detects these implications (outer `(>[...]` carrying an `(Anchor<Tag>[...])` premise).

---

## Pruning

`_prune_proof_graph(raw_theorems, raw_stacks, theorems_dir)` at [`process_proof_graphs.py`](../../../process_proof_graphs.py) is the pruning step: drop chapters for theorems that did not survive the compressor. The compressor's output (the compressed `theorems.txt`) is the authoritative survivor set; chapters for non-survivors are removed before renaming.

---

## Anchor-implication normalisation

`normalize_anchor_implications(raw_theorems, raw_stacks)` at [`process_proof_graphs.py`](../../../process_proof_graphs.py) is a no-op: anchored theorem implications keep their original `>[...]` binder lists, anchor slots included ([I-11](../30_invariants.md#i-11), inverted by [D-75](../40_decisions.md#d-75)).

---

## Induction-scope deletion

`strip_induction_scopes(stack)` at [`process_proof_graphs.py`](../../../process_proof_graphs.py) runs on every raw chapter right after the anchor step, before any renaming reads a scope cell ([D-519](../40_decisions.md#d-519)). The prover's induction scopes (`_boundary_is_<chain>_<names>` segments) and int scopes (`_boundary_int_scope_<premise>_<witness>` segments, [D-578](../40_decisions.md#d-578)) carry no assumption, so readers see their rows at the enclosing scope:

1. Every validity name, the row's own and each dependency's, loses its `_boundary_is_…` and `_boundary_int_scope_…` segments (`strip_induction_scope_segments`); an OR branch or subproof opened inside such a scope moves up with it.
2. A row that now cites itself — the `induction scope` row depositing a goal at the root, the `int scope` row promoting a statement to the parent, or a `zoom in` of a source into either kind of scope — is replaced in place by the row it cited (following such rows to the first real derivation), which leaves its own place. The goal row therefore stays first.
3. Of rows sharing one key (expression, scope) after the merge the first stays; rows that were reached from the goal row before and are not reached any more are dropped. Sibling int scopes of one group merge this way.
4. Asserted: no stripped segment and no `induction scope` or `int scope` row remains, one row per key, and the removal creates no citation cycle. A chapter without either kind of scope is returned unchanged.

The verifier, the Lean export and the HTML read only the processed graph, so none of them needs to know induction scopes or int scopes.

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

Example row (from `0_single_chapter_proof.txt`):

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

Methods include `single chapter proof`, `induction`, `reformulated statement`, `incubator back reformulation`, `or theorem`, `or elimination`, and `proved not broadcast` (plus debug-only `debug`). Reference is `-1` for a single-chapter proof, the induction variable for induction, and the source-theorem expression for derived chapter kinds.

Historical rows illustrating the three-column layout. The first two predate complete anchor binding: their free anchor variables violate the current I-4 contract, so these rows must not be copied as fresh inputs.

```text
(>[i0,s,+,i1](AnchorPeano[N,i0,s,+,*,i1])(>[v1](in2[i0,v1,s])(in3[i0,i1,v1,+])))	single chapter proof	-1
(>[i0,s,+,i1](AnchorPeano[N,i0,s,+,*,i1])(>[v1](in2[v1,i0,s])(in3[i1,v1,i0,+])))	induction	v1
(>[N,i0,s,+](AnchorGauss[N,i0,s,+,*,i1,i2,id])(>[v1,v2](in2[v1,v2,s])(>[v3](interval[N,+,i0,v2,v3])(existence4[N,+,v3,v1,i0]))))	reformulated statement	(>[1,2,3,4](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10,11](limitSet[1,4,9,10,11])(>[12](in2[10,12,3])(>[](interval[1,4,2,12,9])(interval[1,4,2,10,11])))))
```

The `reformulated statement` reference field uses the *raw integer-indexed* form of the source theorem — this is what the verifier's reformulation checker uses to match shapes.

---

## Weaknesses

### Known & tracked

- **Induction triad ordering is asserted, not defended.** The verifier raises `AssertionError` on malformed numbering. Stage 6 is the single producer of these processed filenames, so the invariant holds by construction — but a regression in the filename numbering would take down the verifier rather than degrade gracefully.

### Suspected fragility

- **The book's last chapter is the deepest, not a declared goal.** Each anchor stretch ends with its deepest uncited theorem ([D-592](../40_decisions.md#d-592)). A stretch whose intended final result rests on fewer theorems than another uncited theorem ends with the deeper one. On the current run the FTA stretch ends with the uniqueness half (148 theorems below it) ahead of the existence half (143). Stretches follow anchors, not batches: FTA and FTA2 form one stretch.
- **Regex-based anchor parsing.** `get_anchor_mapping_from_expr` uses regex against the theorem expression to identify the anchor name and slot mapping. A theorem that contains the substring `Anchor` in a non-anchor position (unlikely in current codebase, but not ruled out) would confuse it.
- **`x`-prefixed anchor-var detection.** Priority 2.5 assumes anchor-handling rows use `x<N>` names. A rename of that convention upstream would silently break `_copy` derivation — the Priority-4 step then generates incorrect `_copy` names.
- **`replace_keys_in_string`'s safety.** Priority 2 replaces `raw_name → v<k>`. The replacement is done via regex with lookarounds (similar to verifier's `_replace_arg_safe`), but the implementation differs. A raw name that is a substring of another raw name (e.g. `v1` vs `v10`) would break without proper lookarounds. Current code uses argument-boundary lookarounds, but adding a new chapter format that could alter boundaries would need re-validation.

### Not exercised by tests

- **Cross-chapter repl-map consistency.** No test asserts that two chapters' `repl_map`s are *consistent* with the same theorem when they both reference it. Practically: the Priority-2 seeding means they should be, but a regression in seeding would only show up as a verifier failure on the affected chapters.

---

## Open questions

- **OPEN-14 — RESOLVED.** The x-prefix originates in `ProofGridBuilder::prehandleAnchor` in [`proof/proof_grid_builder.cpp`](../../../GL_Quick_VS/GL_Quick/src/proof/proof_grid_builder.cpp). The grid-wide pre-pass handles anchor-slot copies under its mode and recursion-subtree conditions; see [I-188](../30_invariants.md#i-188). Stage 6's Priority 2.5 translates these names to the configured copy spelling. Changes to the source convention require coordinated processing changes.
- **OPEN-15 — RESOLVED.** A chapter-less survivor would appear in `global_theorem_list.txt` but have no chapter file referring to it. The chapter iteration in `process_proof_graphs.py` constructs `fname_to_raw_thm` by mapping expected filenames (per theorem method) to theorem expressions — so the map covers every survivor, but if the file does not exist on disk, no chapter-processing call visits it. Downstream, the HTML generator and verifier likewise visit only existing files. The theorem is effectively invisible to chapter-level audit. Detection: run the verifier and check whether the theorem's rendered chapter appears in its expected row count; if a `method = induction` theorem has no accompanying `induction_typing/check_zero/check_induction_condition` triad, `build_chapter_theorem_map` raises loudly, while `method = single chapter proof` and reformulation chapters need the general correspondence cross-check.

---

## See also

- [`10_pipeline/07_html_export.md`](07_html_export.md) — downstream consumer.
- [`10_pipeline/08_verifier.md`](08_verifier.md) — downstream consumer + chapter-format enforcer.
- [I-10](../30_invariants.md#i-10), [I-11](../30_invariants.md#i-11) — renaming invariants.
- [`20_core_concepts/08_proof_tags.md`](../20_core_concepts/08_proof_tags.md) — tag vocabulary.

## Repeated expression processing

`process_proof_graphs.py` and `create_processed_proof_graph` memoize argument
extraction, type classification, theorem-shape recognition, and binder/citation
renaming within one call. The configuration is fixed for that call and the caches
are released on return. Chapter-specific replacement maps and variable counters
remain separate and retain their existing visitation order. Mutable chapter rows
are copied independently; immutable string cells are shared until replacement.

## Dependency-first order inside existing proof scopes

[`proof_processing.py`](../../../proof_processing.py), `order_main_scope_rows`, orders main and incubator rows after all renaming and external-tag resolution. Each exact namespace is an independent arrangement graph; recorded cross-scope references remain interfaces rather than local ordering edges.

`ordered_scope_indices` resolves exact local facts first, then bound-name-equivalent theorem-import rows. It recognizes registered theorem citations, integration-goal templates and the existing optional second OR license. An unresolved ordinary local citation or a cycle asserts. It does not infer a proof, change a tag, or remove a dependency.

The renderer reverses rows inside each scope, so the processor stores a dependency-first order backwards. `order_scope_rows` fills only that namespace's original positions, retaining the first chapter goal, every namespace position and the exact row multiset. This local pass preserves its input scope positions and the goal-row checker contract. The final hierarchical pass below determines the positions of nested cards. See [I-305](../30_invariants.md#i-305).

## Main book dependency order
`proof_processing.py`, `order_main_book`, orders the chapters in three steps ([D-592](../40_decisions.md#d-592)):

1. **Citation sort.** The independent certificate citation contract supplies the theorem dependencies. A stable topological order chooses the lowest incoming ready index; the registry arrives in the prover's order, batch by batch, so a chapter moves later only when it cites a theorem listed after it. Extra precedence keeps the original canonical proof before equivalent copies.
2. **Compaction inside anchor stretches.** `arrange_anchor_stretches` cuts that order into anchor stretches, maximal runs of consecutive chapters with one anchor, and runs the private-helper, frontier and insertion-search passes below inside each stretch, scoring only the stretch's own citations. No chapter leaves its stretch, so the stretches keep their order and no citation points forward. The main book has three stretches (Peano, Gauss, FTA; FTA and FTA2 share `AnchorFTA`); the incubator book alternates `AnchorIncubator8` and `AnchorIncubator3` in four.
3. **Deepest theorem last.** Each stretch ends with its deepest theorem that no other chapter of the stretch cites or must precede: the most distinct theorems it rests on through citations, across the whole book, ties to the larger theorem text. The choice is independent of the order, so the repeated processing rounds settle (a tie broken by position made them cycle).

The registry and all chapter filenames are permuted together; row bodies, subproof sections and induction bundles remain unchanged. Incubator uses the same ordering path with its insertion search bounded to distance 512. Live Lean selections use exact index lists with complete-anchor coverage, as described in the external proof export chapter.

## Direct equality-chain fusion
After scope ordering, `proof_processing.py`, `compact_main_equalities` and `fuse_equality_chains`, shorten same-namespace equality1 chains under the existing rule. The two original rows must explicitly cite every oriented source-to-final argument equality. Transitive paths or reverse equalities alone do not qualify. Delete the intermediate only when it has one explicit consumer and is not the first physical row of any namespace. This preserves the goal and subproof discovery order, never removes assumptions or witness/definition/boundary rows, and strictly decreases total rows. Iterate to a fixed point while preserving the remaining row order: each new citation was already a proof ancestor of the consumer. Then order the book. Incubator is unchanged. The existing GL checker and independent Lean terms validate the final artifact.

## Primitive disintegration retained
The main fixed-point pipeline preserves the original `expansion` and `disintegration` rows. It no longer calls `compact_main_definitions`; the positive-projection implementation and both checker rules remain available for historical compact artifacts and focused tests. Main graphs are regenerated from raw input, not expanded from a guessed compressed derivation. Reader-only disintegration blocks expose their source and externally consumed products while retaining all primitive rows.

## Private-helper layout candidates
`proof_layout.py`, `private_helper_order`, schedules each scope and each anchor stretch of the book backwards, choosing a ready private producer immediately before its sole consumer where precedence allows. `layout_score` uses the frozen equally weighted distance, private gaps, live frontier and crossing-arc objective with an exact integer common denominator. `improve_private_helpers` accepts only strict improvements, preserving the existing legal order on a tie or loss. Canonical equivalent-proof precedence and main goal-last constraints are required but excluded from cost. Scope grouping runs after compaction, so it changes only row positions within existing namespace slots. Book grouping works inside one anchor stretch and retains complete theorem bundles and exact index selections.

### Conservative main-scope slicing
`proof_goal_slice.py`, `slice_unused_main_rows`, removes only uncited ordinary main-scope leaves to a fixed point. Every subproof row and all assumption, witness, boundary, import and definition tags are retained. Alpha-equivalent scoped citations count as uses; cross-scope interfaces are unchanged. The resulting actual graphs require both verifier and connected Lean acceptance.
## Earlier equality support and unaffected properties
`proof_shortcuts.py`, `shorten_equality_paths`, runs after private grouping. It can replace two equality1 rewrites by one using directly oriented equality facts already earlier in that namespace. It can also bypass an equality rewrite of a named conjunction when the exact desired property is already a positive projection of the pre-rewrite source. Only the single-consumer equality1 intermediate is removed, outside each namespace's first row; all involved references stay in the same namespace. Both rules already exist and remain unchanged. `shortcut_main_equalities` requires the exact anchor registry and leaves incubator untouched.
## Frontier scheduling candidates
After private grouping, `proof_layout.py`, `frontier_order`, selects a ready vertex that leaves the fewest live producers, breaking ties by closed producers and existing rank. `improve_frontier` accepts only a strict improvement under the unchanged full layout cost, so a frontier-only gain cannot override a worse combined score. Each scope remains in its namespace slots; the main goal and canonical proof precedence remain fixed. Book scheduling works inside one anchor stretch and keeps induction bundles and exact source-index selections.

## Equality rewrite placement
`proof_rewrite_placement.py`, `place_conclusion_rewrite`, may replace multiple private equality1 premise rewrites with one explicit implication application on their original facts and one equality1 on its conclusion. Pure structural matching and exact oriented equality support must accept the candidate; the processor never calls the verifier. The completed graph still requires independent GL and Lean acceptance. Earlier exact intermediate conclusions may be reused; otherwise the new implication row is counted explicitly. Only same-scope single-consumer premise rows outside namespace-first positions can be removed, and each accepted move strictly reduces rows. Full verifier and connected Lean remain mandatory.

A newly needed implication result occupies one removed premise row's existing namespace slot. `place_conclusion_rewrite` then restores dependency order only inside the surviving namespace slots. The new namespace sequence is therefore a subsequence of the input sequence even when main and child rows interleave; registered theorem keys remain available to the strict ordering resolver.

Rewrite placement first checks the unchanged conclusion against the original premises. If that ordinary implication is valid, it removes the private premise substitutions without adding any new row. Only otherwise does it construct a mapped atomic conclusion and a final equality step. Every selected equality premise has its exact structural and oriented support checked before rewriting.
## Exact and bounded arrangement search
`proof_layout_search.py`, `search_layout`, exhaustively minimizes the frozen cost for up to eight vertices. Anchor-stretch ordering and scope-local search on larger graphs first reach an adjacent-swap local fixed point and explore two deterministic insertion sweeps with displacement at most six. A second strict-improvement search starts from that result, explores every legal insertion distance (at most the vertex count minus one) in two further sweeps, and ends at another adjacent fixed point. `LayoutState.adjacent` computes exact integer cost changes using fixed degree balances, neighbor-position bitsets and a histogram of frontier cuts. An adjacent exchange updates only the moved vertices' neighbors and one frontier cut; repeated evaluation of the same candidate reuses its result until the next applied exchange. The independent full-score oracle checks the incremental state. Hard proof/canonical precedence and terminal goals remain mandatory; scopes retain their slots and book induction bundles stay intact. The larger-graph result is a bounded local search, not a global minimum.

The separate reporting tool `.scripts/proof_processing_bounds.py`, `structural_lower_bound`, tightens the existing optimum interval using mandatory path lengths, distinct incident spans and forced crossing pairs. It leaves the frozen cost and all acceptance comparisons unchanged. Both baseline and final graphs receive the same report; large-graph intervals remain bounds rather than a claimed exact optimum.


## Processing to a fixed point
`proof_convergence.py`, `process_to_fixed_point`, repeats the accepted main-book stage-six pass order until its complete theorem registry and every physical row/position are unchanged. Recompute filename bundles after each book permutation using the existing naming callback. Every round preserves goals and namespace discovery and cannot grow a physical chapter before book renumbering. A repeated nonterminal state asserts instead of returning a capped partial result. Incubator dispatches to `process_incubator_book`: one combined dependency/private-helper/frontier/search scope pass and one book ordering whose search inside each anchor stretch uses insertion distance at most 512. It does not run semantic row reduction or repeat layout to a fixed point. Main retains the complete accepted transformation family and exact convergence. This is a fixed point of the stated rewrite and layout family, not a universal shortest proof or global layout optimum.


## Final recursive subproof placement

`proof_hierarchy.py`, `order_main_hierarchies`, runs after `process_to_fixed_point` in `process_proof_graphs.py`, `create_processed_proof_graph`. It is a pure, idempotent row permutation: the accepted mathematical rows are unchanged by this final pass, every row and its namespace remain verbatim, the book registry and induction bundles retain their order, and the main goal remains the first stored row. Earlier scope-local passes retain namespace positions; this final layout pass may change positions and discovery order while preserving the entire scope tree and membership.

`resolved_dependencies` replays the certificate's exact-scoped/alpha-theorem resolution precedence, including cross-scope edges and the existing template, compilation and optional OR-companion contracts. `proof_reader.py`, `reader_plan`, contracts intact scopes and semantic operation blocks into units, then `schedule_nodes` topologically orders those units. Disintegration groups one source's transparent expansion chain. Integration groups a recorded construction application and its private support, including whole subproofs; shared support remains outside. OR siblings are a consecutive scheduling cohort without an extra counted wrapper. `nested_order` remains the scope-only permutation helper. A cyclic contraction asserts; there is no scope splitting or flattening fallback. `hierarchical_scope_items` checks that reverse storage expands into contiguous nested blocks and supplies the renderer's exact sequence.

The complete expanded proof, including hidden theorem-import rows and cross-scope dependencies, is the ordering measurement unit. A reader chapter aggregates all its induction sections. A collapsed card changes visibility, never proof-step count or dependency obligations. Whole-proof layout costs must be compared under the same definition on baseline and result; the older independent-scope costs are not interchangeable. A perfect attainable layout can have unavoidable crossings. See I-306 and D-510.

## Exact runtime reuse

`proof_convergence.py`, `process_to_fixed_point`, owns separate call-local scope-order and layout-search caches. Scope-order keys include every row cell, the complete registered theorem-key set and every layout option. Search keys relabel vertices by their input-order ranks and include all proof edges, additional precedences, the terminal and displacement range. Only immutable results are retained, and callers receive independent copies. Context exit releases the caches, including exceptional and nested calls. The main-book fixed-point predicate, passes and complete search range remain unchanged; Incubator uses the bounded arrangement described below.

`proof_layout.py`, `layout_score`, counts crossing arcs by a producer-grouped endpoint prefix tree; shared-source and shared-target pairs are excluded exactly as in the pairwise definition. `proof_layout_search.py`, `LayoutState.improve_insertion`, scores each legal displacement using exact prefix membership and unchanged neighbor order. It applies only the first strictly best prefix; rejected candidates do not mutate the graph. Unit tests compare score calculations with independent pairwise counting, check each candidate insertion against full recomputation, and verify cache keys, lifetime and isolation.

`proof_layout.py`, `frontier_order`, keeps remaining consumer counts and ready-vertex prerequisite counts. Closing producers are exactly the incoming producers with one consumer left; the common current-frontier term cancels from all candidate comparisons. The candidate score and original-rank tie break are unchanged and are checked against an independent full-remaining-graph implementation.

Main-book selection uses the shared `proof_export/corpora.py` registry. `proof_export/rule_matching.py` owns pure implication matching for processing and captions; `visu_helpers.py` retains the established helper names as imported compatibility exports. The external verifier remains independent and receives the full completed chapter and real verifier state.

## Shared citation resolution policy

`proof_export/citation_policy.py`, `resolve_citation_source`, owns the precedence used by scope ordering, nested placement and certificate emission. Compilation prefers a registered theorem; otherwise exact local sources precede theorem-row aliases, and `zoom in` requires exact local identity. An unavailable exact producer blocks external fallback. Only an absent producer may resolve as a registered theorem, integration-goal template or permitted OR companion. Pending is a scheduling state; a graph with no ready row asserts.

Scope ordering supplies its complete same-namespace producer index to build prerequisites and keeps the main goal last. Nested placement and certificate emission supply only already emitted local sources and schedule by reverse source priority. The shared policy preserves each caller's namespace/proof-scope keys, scheduling obligations and existing OR-companion restriction. Focused tests cover precedence, source index zero, pending producers and scope isolation; full artifact audits compare the resulting edges with certified dependencies.


## Collapsed main-row KPI

`proof_reader.py`, `reader_metrics`, counts ordinary visible root rows plus one per collapsed root block. Hidden theorem imports do not contribute to reader rows. The total proof-step count still includes every formal row in every scope. Both counts sum all physical induction parts into one reader chapter. Semantic blocks are limited to scope boundaries and recorded integration/disintegration operations; arbitrary wrappers cannot manufacture a better count. Every operation preserves all formal rows, and its expanded sequence is written into the actual processed graph. HTML rebuilds the plan and asserts that its expansion matches that graph exactly.

## Incubator reshuffling

The maintainer extends chapter, scope and reader-block reshuffling to the incubator book. `proof_export/corpora.py`, `ORDERED_BOOK_ANCHORS`, shares selection between `proof_processing.py`, `order_main_scope_rows` / `order_main_book`, and `proof_hierarchy.py`, `is_reader_theorem` / `order_main_hierarchies`. Incubator uses one combined dependency/private-helper/frontier/search scope pass, followed by the book ordering of the main book (citation sort, search inside each anchor stretch with insertion distance at most 512, deepest theorem last). It retains exact search for tiny stretches, and uses the existing strict-improvement objective for larger graphs. Main retains its existing repeated full search. The bounded Incubator pass does not claim a layout fixed point or global optimum. Main and incubator remain separate books; mixing them still asserts. Complete theorem bundles and every incubator row, goal, namespace and citation are retained. The final recursive operation order is written into processed files and independently consumed by HTML and the GL verifier. Semantic proof shortening remains restricted by MAIN_ANCHORS; the incubator extension is a permutation, not a new inference or Lean export.

## Bounded Incubator arrangement

The Incubator path performs no semantic proof shortening. `proof_convergence.py`, `process_incubator_book`, therefore arranges each scope once with combined private-helper, frontier and local-search candidates, then calls `proof_processing.py`, `order_main_book`, with insertion distance 512 inside each anchor stretch. Every accepted search move still improves the unchanged four-component cost. The final reader hierarchy retains complete nested scopes, checks every dependency and writes the exact expanded order. Theorem and per-theorem row multisets, goals, namespace membership and citations remain intact. The measured acceptance budget is at most five percent regression in both chapter cost and whole-proof weighted cost against the same-input baseline, not a changed objective or a universal approximation guarantee.

`order_main_book` and `proof_reader.py`, `schedule_nodes`, maintain prerequisite counts and a priority queue instead of rescanning all pending units. They preserve their original ready-unit tie order; a cycle still asserts. `proof_hierarchy.py`, `order_main_hierarchies`, computes one immutable set of theorem alpha keys for the complete call and supplies it to each citation resolver. The set is not retained after the call and never changes resolution precedence. See D-515.


### Exact layout work reuse

`proof_layout_search.py`, `LayoutState.improve_insertion`, records the exact adjacent deltas, replacement frontier cuts and peaks along each legal insertion trajectory. Applying the chosen prefix supplies those already-scored candidates to `LayoutState.adjacent`; all graph-index updates and the final score assertion remain active. Sorted last-consumer positions provide the same closed-producer count by binary search. Search range, candidate order, strict improvement and tie handling are unchanged.

`proof_layout.py`, `private_helper_order` and `frontier_order`, use prerequisite counts and priority queues with their original preference keys. Private producers are indexed by their sole consumer. A producer reaching one remaining consumer changes only that consumer's frontier preference. Stale heap entries are superseded by the current ready set and preference. The same legal layouts are selected without repeated full scans; these implementation changes apply to both books.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
