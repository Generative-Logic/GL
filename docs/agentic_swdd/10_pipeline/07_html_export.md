<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Pipeline · Stage 7 — HTML export `[DRAFT]`

> **Input:** `files/processed_proof_graph/*.txt` (from stage 6).
> **Output:** `files/full_proof_graph/index.html`, `files/full_proof_graph/tags.html`, `files/full_proof_graph/chapter<N>.html` — a navigable, hyperlinked HTML proof-graph export.
> **Owner:** `generate_full_proof_graph.py` (≈3500 lines after the UX expansion) + `visu_helpers.py` (≈485 lines).

---

## What this stage does

Turns the tab-separated processed-proof-graph chapters into a human-facing HTML site. The output is a set of interlinked pages:

- **Index** (`index.html`) — lists every theorem with a link to its chapter page. Groups theorems by method (direct / induction / reformulated / OR) and by anchor.
- **Tags legend** (`tags.html`) — per-tag explanation pages, referenced from every proof step.
- **Chapter pages** (`chapter<N>.html`) — one page per theorem-kind chapter file (for induction theorems: three pages for the typing + zero + condition triad). Each line of the source chapter becomes a step on the page.

Every expression on a chapter page is clickable. Named expressions (`NaturalNumbers`, `fXY`, `fXYZ`, `implication<N>`, `existence<N>`, `or<N>`) expand in-place to their compiled-MPL form — the expansion table lives in an embedded JavaScript constant `GL_BINARY_MAP` populated at generation time. Theorem citations link to the cited theorem's chapter. Tag names link to their `tags.html` section.

This stage is treated as a first-class output: the customer-facing deliverable of a GL run is the HTML site, not the raw theorem list.

---

## The artefact layout

From `files/full_proof_graph/` after a run:

```
index.html                  top-level theorem index
tags.html                   tag glossary page
chapter1.html               theorem 1 proof
chapter2.html
...
chapter<N>.html
favicon.png                 (optional asset)
```

`chapter<N>.html` filename uses the **leading number from the source chapter filename**, no enumerate-counter offset — `chapter42.html` corresponds to whichever chapter file's number starts with `42_` (commit dropped the historical enumerate-counter offset). If theorem 42 is an induction, the three source files (`42_induction_typing`, `43_check_zero`, `44_check_induction_condition`) render into a **single `chapter42.html`** with three sub-sections (`#sub0`, `#sub1`, `#sub2`). HTML page count = theorem count.

---

## Style conventions

All generated pages share a dark theme:

- Background `#181B27`.
- Foreground text `#F0E8DC`.
- Accent `#5DCAA5` (links).
- Step badges `#6B6F82`.
- Dep-glow highlighting on hover.

The theme is embedded inline as `<style>` blocks — no external CSS file. This keeps chapter pages self-contained and portable.

---

## Page structure — `index.html`

Header — including the embedded `GL_BINARY_MAP` JS constant. Sample snippet (first entries visible in the current run's `index.html`):

```javascript
const GL_BINARY_MAP = {
    "AnchorPeano": {
        "signature": "(AnchorPeano[x1,x2,x3,x4,x5,x6])",
        "mpl": "(&(NaturalNumbers[x1,x2,x3,x4,x5])(in2[x2,x6,x3]))"
    },
    "NaturalNumbers": {
        "signature": "(NaturalNumbers[x1,x2,x3,x4,x5])",
        "mpl": "(&(in[x2,x1])(&(fXY[x3,x1,x1])(&(implication4[x1,x2,x3])(&(implication5[x1,x3])(...)))))"
    },
    "existence0": {"signature": "(existence0[x1,x2,x3])", "mpl": "!(>[y1](in[y1,x1])!(in2[x2,y1,x3]))"},
    ...
}
```

Body — list of theorems, usually grouped by method, each line linking to the corresponding chapter page.

A `processText(input)` JavaScript function (embedded in every generated page) pretty-prints MPL expressions with indentation on click, so an inline expression like `(&(NaturalNumbers[...])(in2[...]))` expands to a multi-line indented form.

---

## Page structure — `chapter<N>.html`

Head — includes title (the theorem's MPL expression + a human-readable caption). Sample:

```html
<title>(>[N,0,s](AnchorPeano[N,0,s,+,*,1])(>[v1](in[v1,N])(>[]!(=[v1,0])(existence2[N,v1,s]))))</title>
<title>from v1 ∈ N, v1 ≠ 0 follows (existence2[N,v1,s])</title>
```

Two `<title>` tags — the second is the human-readable form. The MPL in the first gets rendered in the tab, the readable caption shows in the page header.

Body — layout:

- **Header card** — theorem expression (clickable to re-expand), theorem metadata (method, anchor, induction variable if any). Theorem expression spans carry `data-parts` so right-click surfaces the GL binary expansion.
- **Proof steps** — one per chapter-file row. Each step is a `proof-line` `<div>` containing:
 - `step-badge` — global step index in the form `(K/M)` where `M` is the **chapter-wide common total** across the main stack and every subproof. Replaces the prior per-section badging — a row's badge is its position in the chapter as a whole, not within its enclosing scope.
 - Each badge wraps in an `<a class='step-badge-link' href='#cNsXm-entryK'>` permalink so any single step can be deep-linked.
 - `proof-line-content` — the derived expression + the rule citation + any referenced premises.
 - Dep-glow highlighting — dependency links glow on hover to visually trace the justification chain.
 - Ingredient links resolve dependencies by `(expression, namespace)` tuple, not by expression alone (commit — fixes wrong-target on or-convergence rows where the same expression exists in multiple branch scopes).
 - Cross-scope ingredient links land at the **top** of the target subproof (its title bar), not deep inside the body.
 - Subproof cards for nested reasoning. Each card has its own collapsed/expanded state, plus a chapter-level expand/collapse-all control. Card expansion is also auto-triggered when a same-page link points inside the card.
- **Goal-alias meta-line** — every proof section (main stack, OR-introduction subproof, OR-elimination branch, nested subproof) carries a meta-line of the form `Goal alias: <expr> (<ns>)` showing the inner-most theorem head the section is trying to derive (commit; resolves it to the inner-most head rather than the full implication; makes the namespace tag dead text). The induction-method `variable typing` subsection's goal alias uses `(in[v,N])`.
- **Subproof title explanations** — under each subproof title, a textual gloss explains what the subproof is doing:
 - **General-implication subproofs**: list each premise, show the head, name the implication being emitted, with the symbolic clause shown in **sequent shorthand `A, B ⊢ C`** rather than as a tautology.
 - **OR-introduction / OR-elimination subproofs**: textual explanation + symbolic `A ∨ B` example + the `or<N>`'s full MPL expansion in brackets.
- **Readable captions** — between steps, human-language glosses in grey-bold (`.readable-grey`).
- **Used-by panel** — opt-in side panel showing every step that depends on the current step.
- **Achieved-goal highlight** — the **last** non-theorem display row matching the goal-alias is rendered in gold (`.goal-highlight`); the prior generator highlighted the first match, which under multi-derivation chapters pointed at a row that wasn't the actual conclusion.
- **Subproof step ordering** matches the main proof body (earliest → latest); previously some subproofs rendered in reverse-discovery order.
- **Line-wrap toggle** — long expressions wrap rather than overflow.
- **Chapter statistics** — at the bottom: step counts, tag breakdown.

Step tags link to the corresponding `tags.html` section. Cited theorems link to their chapter page. Named expressions in-place-expand via the `GL_BINARY_MAP` + `processText`.

### Namespace tags — display only, unclickable

Every `(<ns>)` validity tag in a proof line is **non-interactive display text**, rendered as `<span class="validity-tag">…</span>` with no click handler and no jump target ([`generate_full_proof_graph.py::_format_validity_tag`](../../generate_full_proof_graph.py)). The `namespace_anchor_map` parameter is accepted for call-site compatibility but intentionally ignored — the prior subproof-cross-link path turned non-`main` namespaces into `<a class="ns-jump">` links, which were removed because they produced jump destinations that were sometimes confusing for readers (every cell carries a `(ns)` tag, so the screen filled with hyperlinks).

The user reading a chapter inspects the namespace label as a scope annotation, not as a navigation target. Subproof cards are still independently navigable via their headers; `(<ns>)` tags do not duplicate that navigation.

**Truthfulness by construction (post `D-56`).** Chapter rows are emitted at the lifted (closest-to-`main`-with-origin) validity by `visualizer.cpp::buildStack` — see [`10_pipeline/04_prover.md`](04_prover.md#buildstack-chapter-walker-d-51-algorithm-lifting-per-d-56). Every `(<ns>)` label therefore names the scope where the derivation actually lives. Pre-lifting, a tag could name a deep boundary scope while the origin actually lived at `main` (a falsified label); post-lifting, the label is honest.

### External-theorem links and orphan handling

Theorem citations split into three styles depending on resolution:

- **Same-batch internal** (mint green): the cited theorem has a chapter in this pipeline's `files/full_proof_graph/`. Standard same-tab link, no decoration.
- **Cross-batch external** (lavender, `↗` superscript, opens in new tab): the cited theorem is registered in this pipeline's `external_theorems.txt` AND a sibling pipeline (e.g. main-pipeline citing incubator output, or vice versa) has a chapter for it. Resolved via `sibling_graphs` parameter passed into `generate_proof_graph_pages` (`run_modes.py` plumbs main + incubator dirs in both directions)..
- **Orphan external** (lavender, dashed underline, no `↗`, `cursor: help`): registered as external in this pipeline but no chapter exists locally or in any sibling. Right-click reveals a popup with the explanation `"This theorem is external — registered for this pipeline but no chapter exists here or in any sibling pipeline."` Commit.

The legend (`tags.html`) covers all three styles. The legend itself was rearranged into a 3×3 grid with parenthesised `(Namespace)` labels.

The cross-batch / orphan distinction is computed at HTML-generation time by:

1. Reading `external_theorems.txt` from this pipeline (registers what this pipeline calls external).
2. For each sibling in `sibling_graphs`, reading the sibling's processed-graph dir to find its chapter list.
3. Matching cited expressions via `_resolve_theorem_target` against this pipeline's chapters first, then sibling chapters, then orphan-marking the rest.

---

## Rendering helpers — the key functions

(Selected — the full list is long; these are the highest-leverage.)

| Function | File | Role |
|---|---|---|
| `_alpha_normalize_theorem_expr` | [`generate_full_proof_graph.py`](../../generate_full_proof_graph.py) | Canonical α-rename for theorem-expression comparison. |
| `_resolve_theorem_target` | [`generate_full_proof_graph.py`](../../generate_full_proof_graph.py) | Given a theorem expression, find its chapter target (URL fragment). |
| (w-rename — none here) | — | The HTML generator no longer carries its own w-rename helper. The processed proof graph already renders bound vars whose raw form was `\d+` as `w<N>` / `W<N>` (per-cell counter, see [`06_process_proof_graph.md` §Pass 2](06_process_proof_graph.md#pass-2-ww-rename-of-bound-vars-whose-raw-form-is-d)) AND renders cited theorem-anchor implications' non-anchor inner bvars in `w / W` form via case-preserving letter swap (see [`06_process_proof_graph.md` §Pass 3](06_process_proof_graph.md#pass-3-vw-rename-of-cited-theorem-anchor-implications)). HEAD cells (column 0) and `global_theorem_list.txt` keep `v / V`. The generator renders cells as-is. |
| `_strip_i_prefix` | [`generate_full_proof_graph.py`](../../generate_full_proof_graph.py) | Strip `i`-prefix from constants (`i0 → 0`, `i1 → 1`) for readability. |
| `_htmlify_readable` | [`generate_full_proof_graph.py`](../../generate_full_proof_graph.py) | Produce the grey human-readable caption for a chapter row. |
| `_format_validity_tag` | [`generate_full_proof_graph.py`](../../generate_full_proof_graph.py) | Render a validity-name scope marker. |
| `format_stack_entries` | [`generate_full_proof_graph.py`](../../generate_full_proof_graph.py) | Top-level formatter for a chapter's stack of proof lines. |
| `wrap_clickable` | [`generate_full_proof_graph.py`](../../generate_full_proof_graph.py) | Make an expression clickable (JavaScript expansion handler). |
| `_diff_highlight_html` | [`generate_full_proof_graph.py`](../../generate_full_proof_graph.py) | Render a diff between two expressions with `.arg-changed` spans — used to highlight what changed in an `equality1` or `multiplied from` step. |
| `rename_expr_peano` / `rename_expr_gauss` | [`generate_full_proof_graph.py`](../../generate_full_proof_graph.py) / [`generate_full_proof_graph.py`](../../generate_full_proof_graph.py) | Per-anchor display renaming (e.g. `i0 → 0`, `i1 → 1`, `i2 → 2`). |
| `infer_anchor_kind_from_theorem` | [`generate_full_proof_graph.py`](../../generate_full_proof_graph.py) | Determine anchor type from a theorem expression (used to pick the right rename function). |
| `build_anchor_symbol_replacement_map` | [`generate_full_proof_graph.py`](../../generate_full_proof_graph.py) | Build the map of anchor-slot names to display strings for a given anchor application. |

---

## Readable captions

The readable captions (grey-bold text between steps) are optional but valuable. They turn an opaque chain like:

```text
(in3[i0,i1,v1,+])   implication   (>[s,+,i1](AnchorPeano...)...)   (AnchorPeano[...])   (in2[i0,v1,s])
```

into:

> *RULE: s((w1+w2)) = w5 = (s(w1)+w2) IMPLIES: s((v5+v1)) = v3 = (s(v5)+v1)*

The generator's `_htmlify_readable` + `visu_helpers.format_implication` / `format_reformulation` + the anchor-specific rename functions build these by:

1. Stripping `i`-prefixes (`i0 → 0`, `i1 → 1`).
2. Translating common predicates (`in`, `in2`, `in3`, `fold[`, `=`) to their natural-language or algebraic forms.
3. For implication-tag rows, dispatching to the **title-form** chain renderer:
 - Operator-headed chains (`(in2`, `(in3`, `(fold[`) → `make_readable_simple_implication_title` produces closed algebraic form `LHS = head_output = RHS` and walks `fully_resolve_markers` to substitute antecedent outputs back in. Body rows match chapter titles for the same theorem. Pre- body rows used the verbose `from <premises> follows <conclusion>` form via `make_readable_simple_implication`.
 - Equality-headed chains (`(=[a,b]`) → `make_readable_equality` (the body path) renders the chapter row's literal head form for the IMPLIES clause to avoid the marker-resolved tautology trap (commit — pre-fix, `from s(7)=2 follows V1(v1)=V1(v1)` collapsed both sides to the same expression). The post-fix form is `from s(7)=2 follows 2=8`, naming the actual derived equality. After commit, equality-headed implication-tag rows route through `make_readable_from_chain_title` → `make_readable_generic_chain` (`from X, Y follows EQ`) instead, which trades the "and" connector for comma but preserves the literal-head form for the IMPLIES clause.
4. Substituting bound-variable names with Unicode-friendly display forms — including blackboard-bold ℕ for the natural-numbers anchor slot (commit; weight + size tuned via,,, with reverting an over-aggressive em-scaled stroke).

Captions are advisory — they are generated from heuristics, not required for verifier correctness. The verifier consumes `processed_proof_graph/` directly and never touches the HTML.

---

## The embedded `GL_BINARY_MAP`

Every generated page embeds a copy of the compiled-expression signatures — the data required to render an expansion when the user clicks a named expression. Source: `files/GL_binaries/GL_binary_<Tag>.json` (from stage 1).

The embedded copy trades page size for offline usability: a chapter page works without a network round-trip for expansion. Page size is ≈20–50 KB per chapter on Peano; the binary map is a few KB of the total.

---

## Weaknesses

### Known & tracked

- **HTML export is first-class output** (per the project conventions). Any change that degrades the HTML — visually or in clickable-link correctness — is customer-facing. This is *stated policy*, not *enforced*. No regression test captures HTML output currently.

### Suspected fragility

- **No CI on generated HTML.** Changes to the generator are caught only by visual inspection. A regex drift in `_htmlify_readable` could silently produce wrong readable captions across every chapter; the verifier would still pass.
- **`GL_BINARY_MAP` is inlined per page.** If the binary grows (say, after FTA's new definitions), every chapter page grows. No deduplication.
- **Human-readable caption generation is heuristic.** Unusual expressions (rare predicates, deeply-nested negations) fall back to raw MPL. No mechanism distinguishes "caption succeeded" from "caption quietly punted".
- **Dark-theme hard-coded.** No alternative rendering. If anyone wants a light-mode proof export, it's a from-scratch fork of the style block.

### Not exercised by tests

- **Per-tag rendering correctness.** Each tag has a specific `format_stack_entries` branch — 30 tag descriptions in `tag_descriptions.json` (post-) plus the meta-tags from the verifier registry that don't surface as row tags. A regression in one (e.g. `or disintegration` — which has nested branch blocks) would only be caught by looking at the relevant chapter.
- **Cross-links.** When a chapter cites another theorem, the link target is computed from the theorem's expression. A rename drift (in either the processor or the generator) could produce dead links. No test verifies link integrity.
- **Matryoshka subproof nesting.** Sub-sub-…-proofs render via `_render_subproof_card` recursing on `nested_subproofs`. Visual spot-check is the only test today; a regression in the recursion (e.g. infinite loop on a malformed payload, or a child not being placed inside its parent's `<div class='subproof-body'>`) would not be auto-detected.

---

## Matryoshka subproof structure (D-37)

Sub-proofs nest arbitrarily deep. The renderer detects nesting purely from primary namespace ancestry: a row at namespace `A_boundary_<X>` is a child of the scope at namespace `A`. No tag-level marker is required — `validity name` rows provide rich titles when present, but OR-branch sub-subproofs (introduced by `or branch proven` / `or disintegration` rows) are detected via the namespace pattern alone.

Two functions:

- `_partition_stack_subproofs(stack, scope_ns)` at [`generate_full_proof_graph.py`](../../generate_full_proof_graph.py) — recursive. Returns `(own_rows, subproofs)` where each subproof has a `nested_subproofs` field (the recursive structure).
- `_render_subproof_card(sp, prefix, depth)` at [`generate_full_proof_graph.py`](../../generate_full_proof_graph.py) — also recursive. Renders one card; the card's body contains its own rows followed by a `_render_subproof_card` call per nested subproof.

Title inference for a child scope `child_ns` under parent `parent_ns`:

1. **Validity-name lookup.** Scan parent-scope rows for `validity name` rows whose `row[4] == child_ns`. If found, the row's `row[0]` is the implication expression (used as title) and `row[3]` is the goal alias.
2. **`_orint_` payload pattern.** If the child payload matches `orint_(or<N>[…])_((<disjunct>))`, render as "OR-introduction subproof — branch where `<disjunct>` is asserted (of OR `(or<N>[…])`)".
3. **`_ordis_` payload pattern.** If the child payload matches `ordis_(or<N>[…])_((<disjunct>))`, render as "OR-elimination branch — case `<disjunct>` (of OR `(or<N>[…])`)".
4. **Fallback.** Use the raw payload as the title.

Per-depth CSS classes `.subproof-depth-1` through `.subproof-depth-5` differentiate nested cards visually (color shift + indent per level). Depth >5 falls through a CSS `:not` rule with a generic look.

Example: chapter `1209_direct_proof.txt` (the FTA-rung-1 forward direction) renders as:
- 3 depth-1 cards (impl24, impl25, impl26 subproofs).
- 5 depth-2 cards nested inside their parents:
 - 2 `_ordis_` branches under impl24 (case-split of `(or2[i0,v2,i1])`).
 - 2 `_ordis_` branches under impl25 (case-split of `(or2[i0,v4,i1])`).
 - 1 `_orint_` subproof under impl26 (sub-implication for `(or2[i1,v5,i0])`).

All collapsed by default (the existing `.subproof-card.collapsed` toggle behaviour); each card opens independently on click.

---

## Open questions

- **OPEN-16 — RESOLVED.** Induction renders as **one HTML page per theorem**, not three. The generator aggregates the three source chapter files (`<N>_induction_typing.txt`, `<N+1>_check_zero.txt`, `<N+2>_check_induction_condition.txt`) into a single `chapter<M>.html` page with sub-sections. See [`generate_full_proof_graph.py–1489`](../../generate_full_proof_graph.py) (file aggregation) and [`:2199–2201`](../../generate_full_proof_graph.py) (render of both check_zero and check_induction_condition inside the same page with prefixes `c{idx}s0`, `c{idx}s1`). So: HTML page count = theorem count; induction-chapter-file count = 3 × induction theorems + 1 × direct theorems +...
- **OPEN-17 — RESOLVED.** No sitemap is generated by the HTML export. The pages do include `<meta name="robots" content="index, follow, noai, noimageai">` at [`generate_full_proof_graph.py`](../../generate_full_proof_graph.py) — so crawl behaviour is controlled by meta-robots (allow search indexing; forbid AI training crawlers via `noai` and `noimageai`). Crawlers must reach chapter pages through the index link graph, not a sitemap. For bot-traffic control: the `noai`/`noimageai` directives block compliant crawlers (GPTBot, ClaudeBot, CCBot) but not scrapers disguised as humans — that needs Cloudflare-level control.

---

## See also

- [`10_pipeline/06_process_proof_graph.md`](06_process_proof_graph.md) — upstream producer.
- [`10_pipeline/08_verifier.md`](08_verifier.md) — consumes the same processed proof graph, independently.
- [`20_core_concepts/08_proof_tags.md`](../20_core_concepts/08_proof_tags.md) — tag definitions referenced by the generator's per-tag rendering branches.
- — HTML proof visualization is strategic priority.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
