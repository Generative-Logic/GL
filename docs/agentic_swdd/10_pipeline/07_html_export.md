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
> **Owner:** `generate_full_proof_graph.py` and `visu_helpers.py`.

---

## What this stage does

Turns the tab-separated processed-proof-graph chapters into a human-facing HTML site. The output is a set of interlinked pages:

- **Index** (`index.html`) — lists every theorem with a link to its chapter page. Follows the processed theorem-list order; induction entries contain links to their three proof parts.
- **Tags legend** (`tags.html`) — per-tag explanation pages, referenced from every proof step. Below 640 px its reasoning-rules and readable-notation tables show one entry per block (`_generate_tags_page`), so the page never scrolls sideways on a phone.
- **Chapter pages** (`chapter<N>.html`) — one page per theorem; an induction theorem combines its typing, zero, and condition chapter files into three sections on that page. Main and incubator rows follow the recursively expanded reverse-stored processed order, with intact child cards interleaved between their prerequisites and consumers. Unsupported books retain rows reversed within separately rendered scopes. Rows tagged `theorem` supply citations but are omitted from the visible step list; they remain physical processed-proof rows. `generate_full_proof_graph.py` and `generate_proof_graph_pages` own this grouping.

Every expression on a chapter page is clickable. Named expressions (`NaturalNumbers`, `fXY`, `fXYZ`, `implication<N>`, `existence<N>`, `or<N>`) expand in-place to their compiled-MPL form — the expansion table is a JavaScript constant `GL_BINARY_MAP` embedded in each page, holding exactly the definitions that page's expressions name ([D-538](../40_decisions.md#d-538)). Theorem citations link to the cited theorem's chapter. Tag names link to their `tags.html` section.

This stage is treated as a first-class output: the customer-facing deliverable of a GL run is the HTML site, not the raw theorem list.

---

## The artefact layout

From `files/full_proof_graph/` after a run:

```
index.html                  top-level theorem index
tags.html                   tag glossary page
chapter0.html               first theorem's proof
chapter1.html
...
chapter<N>.html
favicon.png                 (optional asset)
```

`chapter<N>.html` is named by the **book chapter number** N, the theorem's position in `global_theorem_list.txt` from 0 (`book_chapter_numbers`, [D-591](../40_decisions.md#d-591)); headings, navigation, citations, "used by" lists, row identifiers and the Lean twin page use the same N. If theorem 42 is an induction, its three source files (for example `61_induction_typing`, `62_check_zero`, `63_check_induction_condition`) render into the **single page `chapter42.html`**, whose sections `#sub0`, `#sub1`, `#sub2` read 42.1 to 42.3, and the next theorem is `chapter43.html`. The processed files keep their own numbers, which the verifier and the Lean code (`chapter_<k>_line_<j>`) use. HTML page count = theorem count.

---

## Style conventions

All generated pages share one theme, defined in [`proof_theme.py`](../../../proof_theme.py) ([D-539](../40_decisions.md#d-539)):

- **Color roles, not colors.** The generators' CSS and inline styles name roles only (`var(--gl-bg)`, `var(--gl-text)`, `var(--gl-link)`, `var(--gl-goal)`, `var(--gl-target)`, `var(--gl-external)`, `var(--gl-namespace)`, the card depths `--gl-depth-1`..`--gl-depth-6`,...). `THEME_TOKENS_CSS` defines every role twice: the day side (warm paper, ink text, soft eucalyptus links, ember goals) is the default; the night side (dusk violet, cream text, gold goals) applies under `prefers-color-scheme: dark`. Glows use `rgb(var(--gl-goal-rgb) / alpha)`. Common-notation mode draws its lines in `--gl-text-common`, a softer ink between the grey captions of the mixed mode and body text (about 9:1 on paper instead of 15:1), at weight 500; titles and index captions keep 600 and achieved goals 700 (`proof_reader_html.STYLE`).
- **Type.** Interface text in a geometric sans (`--gl-font-ui`: Jost, Futura, Century Gothic,...), MPL in monospace (`--gl-font-mono`), common notation in a book serif (`--gl-font-math`: Newsreader, Palatino,...). No web font is linked; every family falls back to an installed relative.
- **Skin.** `THEME_SKIN_CSS` closes the last style block of every page (index, chapters, tag page, Lean pages): the sticky reader header with its horizon-gradient rule and brand mark (`gl-logo.png`), section headings, cards, pill controls, rule-name plaques. It overrides only look, never the reading mechanics of the earlier rules.
- Regression: `tests/test_proof_theme.py` (every referenced role defined on both sides, no literal palette in the generators); the browser test `header_semantics` compares the achieved-goal color with the page's own `--gl-goal`.

The theme is embedded inline as `<style>` blocks — no external CSS file. This keeps chapter pages self-contained and portable.

---

## Page structure — `index.html`

Header — including the page's `GL_BINARY_MAP` JS constant (the definitions the page's own expressions name). Sample snippet (entries of the current run's `index.html`):

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

Body — theorems in processed theorem-list order, each entry linking to its chapter page and induction parts where applicable.

A `processText(input)` JavaScript function (embedded in every generated page) pretty-prints MPL expressions with indentation on click, so an inline expression like `(&(NaturalNumbers[...])(in2[...]))` expands to a multi-line indented form.

---

## Lean twin pages — `chapter<N>_lean.html`

A run with a Lean toolchain leaves the mode's Lean export at
`<out_dir>/lean_export/` (written by `proof_export/live.py` before this stage;
the output-folder wipe at the top of `generate_proof_graph_pages` keeps that
one child). [`lean_pages.py`](../../lean_pages.py) reads it —
`load_lean_export(out_dir, proc_dir)` takes every corpus whose certificate
was built from exactly this graph (its `source.theorem_list_sha256` equals the
hash of `<proc_dir>/global_theorem_list.txt`; the shortcut folder also holds
the copied Peano and Gauss corpora, whose indices would otherwise collide),
requires a green `kernel_check.log` (while the pipeline's own kernel check still
runs beside the main HTML stage: the command line and no exit code yet;
[D-540](../40_decisions.md#d-540)), and slices each theorem's declarations
(`slice_theorem_source`: the public `theorem <id>` and its private
`<id>_…` induction helpers) out of the generated module. Every exported chapter
gets `chapter<N>_lean.html` (N the page's book chapter number) in the chapter style: the theorem statement and
readable caption, a card (Lean identifier, method, chapters with row-fact
counts, toolchain), links to the raw module, certificate, manifest and
kernel-check log, and the highlighted source (`highlight_lean`: comments and
keywords marked, everything escaped). The chapter page's navigation gains
`Lean 4 proof`, the index entry a `[Lean 4]` link and the index a one-line
export summary. Without an export (no toolchain) none of this appears —
`load_lean_export` returns None and every hook renders the empty string.
Regression: `tests/test_lean_pages.py`.

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
 - Ingredient links use the exact `(expression, namespace)` row anchor when available; entering a collapsed card opens its ancestors before scrolling to that row.
 - Subproof cards for nested reasoning. Each card has its own collapsed/expanded state, plus a chapter-level expand/collapse-all control. Card expansion is also auto-triggered when a same-page link points inside the card.
- **Goal-alias meta-line** — every proof section (main stack, OR-introduction subproof, OR-elimination branch, nested subproof) carries a meta-line of the form `Goal alias: <expr> (<ns>)` showing the inner-most theorem head the section is trying to derive (commit; resolves it to the inner-most head rather than the full implication; makes the namespace tag dead text). The induction-method `variable typing` subsection's goal alias uses `(in[v,N])`.
- **Subproof title explanations** — under each subproof title, a textual gloss explains what the subproof is doing:
 - **General-implication subproofs**: list each premise, show the head, name the implication being emitted, with the symbolic clause shown in **sequent shorthand `A, B ⊢ C`** rather than as a tautology.
 - **OR-introduction / OR-elimination subproofs**: textual explanation + symbolic `A ∨ B` example + the `or<N>`'s full MPL expansion in brackets.
- **Legacy readable captions (unsupported books)** — a grey line (`.readable-grey`) inside each row's content column, directly under the row it explains: `RULE: … APPLIED: …` under an implication row, `… REFORMULATED FROM: …` under a reformulation row, `READS: …` under every other row whose expression is not a plain relation. See [Readable captions](#readable-captions).
- **Used-by panel** — opt-in side panel showing every step that depends on the current step.
- **Achieved-goal highlight** — the **last** non-theorem display row matching the goal-alias is rendered in gold (`.goal-highlight`); the prior generator highlighted the first match, which under multi-derivation chapters pointed at a row that wasn't the actual conclusion.
- **Subproof step ordering** matches the main proof body (earliest → latest); previously some subproofs rendered in reverse-discovery order.
- **Line-wrap toggle** — long expressions wrap rather than overflow.
- **Chapter statistics** — at the bottom: step counts, tag breakdown.

Step tags link to the corresponding `tags.html` section. Cited theorems link to their chapter page. Named expressions in-place-expand via the `GL_BINARY_MAP` + `processText`.

### Namespace tags — display only, unclickable

Every `(<ns>)` validity tag in a proof line is **non-interactive display text**, rendered as `<span class="validity-tag">…</span>` with no click handler and no jump target ([`generate_full_proof_graph.py::_format_validity_tag`](../../../generate_full_proof_graph.py)). The `namespace_anchor_map` parameter is accepted for call-site compatibility but intentionally ignored — the prior subproof-cross-link path turned non-`main` namespaces into `<a class="ns-jump">` links, which were removed because they produced jump destinations that were sometimes confusing for readers (every cell carries a `(ns)` tag, so the screen filled with hyperlinks).

The user reading a chapter inspects the namespace label as a scope annotation, not as a navigation target. Subproof cards are still independently navigable via their headers; `(<ns>)` tags do not duplicate that navigation.

**Truthfulness by construction (post `D-56`).** Chapter rows are emitted at the lifted (closest-to-`main`-with-origin) validity by `proof/proof_reconstructor.cpp::buildStack` — see [`10_pipeline/04_prover.md`](04_prover.md#buildstack-chapter-walker-d-51-algorithm-lifting-per-d-56). Every `(<ns>)` label therefore names the scope where the derivation actually lives. Pre-lifting, a tag could name a deep boundary scope while the origin actually lived at `main` (a falsified label); post-lifting, the label is honest.

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
| `_alpha_normalize_theorem_expr` | [`generate_full_proof_graph.py`](../../../generate_full_proof_graph.py) | Canonical α-rename for theorem-expression comparison. |
| `_resolve_theorem_target` | [`generate_full_proof_graph.py`](../../../generate_full_proof_graph.py) | Given a theorem expression, find its chapter target (URL fragment). |
| (w-rename — none here) | — | The HTML generator no longer carries its own w-rename helper. The processed proof graph already renders bound vars whose raw form was `\d+` as `w<N>` / `W<N>` (per-cell counter, see [`06_process_proof_graph.md` §Pass 2](06_process_proof_graph.md#pass-2-ww-rename-of-bound-vars-whose-raw-form-is-d)) AND renders cited theorem-anchor implications' non-anchor inner bvars in `w / W` form via case-preserving letter swap (see [`06_process_proof_graph.md` §Pass 3](06_process_proof_graph.md#pass-3-vw-rename-of-cited-theorem-anchor-implications)). HEAD cells (column 0) and `global_theorem_list.txt` keep `v / V`. The generator renders cells as-is. |
| `_strip_i_prefix` | [`generate_full_proof_graph.py`](../../../generate_full_proof_graph.py) | Strip `i`-prefix from constants (`i0 → 0`, `i1 → 1`) for readability. |
| `_row_caption_html` / `_readable_theorem` | [`generate_full_proof_graph.py`](../../../generate_full_proof_graph.py) | The grey caption of a chapter row / of a theorem (heading, index, Lean twin page, tab title), through the graph's `visu_helpers.ReadableForm`. |
| `_format_validity_tag` | [`generate_full_proof_graph.py`](../../../generate_full_proof_graph.py) | Render a validity-name scope marker. |
| `format_stack_entries` | [`generate_full_proof_graph.py`](../../../generate_full_proof_graph.py) | Top-level formatter for a chapter's stack of proof lines. |
| `wrap_clickable` | [`generate_full_proof_graph.py`](../../../generate_full_proof_graph.py) | Make an expression clickable (JavaScript expansion handler). |
| `_diff_highlight_html` | [`generate_full_proof_graph.py`](../../../generate_full_proof_graph.py) | Render a diff between two expressions with `.arg-changed` spans — used to highlight what changed in an `equality1` or `multiplied from` step. |
| `rename_expr_peano` / `rename_expr_gauss` | [`generate_full_proof_graph.py`](../../../generate_full_proof_graph.py) / [`generate_full_proof_graph.py`](../../../generate_full_proof_graph.py) | Per-anchor display renaming (e.g. `i0 → 0`, `i1 → 1`, `i2 → 2`). |
| `infer_anchor_kind_from_theorem` | [`generate_full_proof_graph.py`](../../../generate_full_proof_graph.py) | Determine anchor type from a theorem expression (used to pick the right rename function). |
| `build_anchor_symbol_replacement_map` | [`generate_full_proof_graph.py`](../../../generate_full_proof_graph.py) | Build the map of anchor-slot names to display strings for a given anchor application. |

---

## Readable captions

The suppression rules in this section describe the retained compatibility renderer for unsupported books. Supported-book common notation is complete per displayed row, as described below.

The grey lines translate MPL into mathematical notation; the legend entry "Readable version" links to the notation key at the top of `tags.html` (`_READABLE_NOTATION_HTML`). [`visu_helpers.py`](../../../visu_helpers.py) builds them in `ReadableForm`, created once per graph by `_generate_proof_graph_pages` from the graph's GL binaries (`load_compiled_definitions`) and the anchor slot names of the visualizer configuration (`_anchor_signatures`). Every caption has an HTML spelling and a plain Unicode spelling (browser-tab title).

| Where | Caption | Method |
|---|---|---|
| Chapter heading, index entry, Lean twin page, tab title | the theorem | `ReadableForm.theorem` (via `_readable_theorem`) |
| Implication row | `RULE: <rule>`, a second line `APPLIED WITH: w1 = v4, …`, and `READS: <derived>` when the derived expression is not a plain relation | `ReadableForm.rule_application`, `ReadableForm.reading` |
| `reformulated from` / `incubator back reformulation` row | `<statement> REFORMULATED FROM: <source>` (`BACK-REFORMULATED FROM:`) | `ReadableForm.reformulation` |
| Any other row | `READS: <expression>`, omitted for a plain relation (`in`, `=`, `in2`, `in3`, orders, their negations) and anchors, and for an `expansion` row that would repeat the reading of its source or unfolds an anchor or `NaturalNumbers` | `ReadableForm.reading` |
| Subproof card | `PROVES: <implication>` (implication subproof), `ASSERTS: <disjunct> OF: <or>` (OR introduction), `CASE: <disjunct> OF: <or>` (OR elimination); the goal-alias line gets the goal's reading | `_subproof_title_reading`, `ReadableForm.formula` |

`_row_caption_html` places a row caption inside the row's `proof-line-content`, under the row it explains.

**Notation.** Expressions are parsed by the strict parser of [`proof_export/mpl.py`](../../proof_export/mpl.py). Relations read as equations: `in2[a,b,f]` → `f(a) = b`, `in3[a,b,c,op]` → `a + b = c` / `a · b = c` (another operation slot reads `op(a, b) = c`), a standard fold → `∑` / `∏` with stacked bounds (both included, see [Fold values](#fold-values)). Orders: `preorder` with `+` → `≤`, with `*` → `∣`; `strictOrder` with `+` → `<`, with `*` → `a ∣ b ∧ a ≠ b`; a negation takes the crossed glyph; any other operation slot asserts (every anchor names these two). User definitions have fixed templates: `interval` → `M = [n, m]`, `sequence` → `f: [n, m] → ℕ`, `limitSequence` → `g = f|≤n`, `limitSet` → `L = M ∩ [0, n]`, `appendToSequence` → `g = f ∪ {n ↦ a}`, `addToSet` → `L = M ∪ {a}`, `split` → `g = f|[n, m)`, `quotient` → `a div d = q`, `remainder` → `a mod d = r`, `prime`, `nontrivialFactor`, `EnumerationSet<k>` → `M = {…}`, `EnumerationRelationSet<k>` → `X = {a ↦ b, …}`, `identity` → `id is the identity on ℕ`, `fXY` / `fXYZ` → `f: X → Y` / `f: X × Y → Z`, `NaturalNumbers` and the anchors by name. The FTA and incubator definitions read through `ReadableForm._named` ([D-576](../40_decisions.md#d-576)): `hasPrimeFactorization` → `∃ primes p₀ ≥ p₁ ≥ … ≥ pₖ: n = ∏ᵢ₌₀ᵏ pᵢ`, `primeFactorSeq` → `n = ∏ᵢ₌₀ᵏ P(i) with primes P(0) ≥ P(1) ≥ … ≥ P(k)` (each product with the product sign and the stacked bounds of a fold's product, `_Spelling.big_operator`, binding its own index `i` as a fold does) (in the structured book the `with primes` part is an indented second line, `_continued_lines`), `bez1` → `∃ m, n ∈ ℕ: a · m = b · n ± 1`, `bezEq` → `a · x = b · y + 1`, `coprime` → `gcd(a, b) = 1`, `divisorSet` → `D = {d ∈ ℕ: 2 ≤ d ≤ k, d ∣ k}`, `minimum` → `m = min M`, `hasMinimum` → `M has a minimum`, `subset` → `X ⊆ Y`, `empty` → `X = ∅`, `ordered` → `P is non-increasing`, `primeseq` → `all values of P are prime`, `inSequence` → `p is a value of P`, `infiniteSequence` → `f: ℕ → ℕ`, `residual` → `p + m = n`, and `constSeq`, `nonInterval`, `nonSequence` as their few defining facts, one per conjunct. **A named definition is never unfolded.** A name its reading binds itself (`m, n` in `bez1`, `p₀ … pₖ` in `hasPrimeFactorization`, `d` in `divisorSet`) comes from `_TEMPLATE_LETTERS` (`d`, `k`, `m`, `n`, `p`), letters no proof page uses for a variable: the proof graph names `v`, `w`, `V`, `W` with digits, the numbered unfolding `y`, `Y` with digits, a fold its index `i`. So a template's name never meets a name of the surrounding formula, and a reading binds a letter only where that reads better than words (`all values of P are prime`, `p is a value of P`, `M ∩ [0, n]` need none). A compound argument inside a sentence is parenthesized. Only the numbered compiled names expand from their binary entry: `and<N>` → `∧`, `or<N>` → `∨`, `existence<N>` → `∃ y ∈ S: …`, `implication<N>` → `∀ …: … ⇒ …`; the expansion stops at every named definition it meets, which reads through its template. Their bound digits get fresh names `y<k>` (element positions, `_ELEMENT_SLOTS`) / `Y<k>` (sets, functions), numbered in order of display; a compiled theorem (`implication<N>[]` whose first element is an anchor) gives the anchor slots their canonical names. `∀ x, y: F(x) = y ⇒ G(x) = y` reads `F ⊆ G` (`_inclusion`; negated `F ⊈ G`). Tokens: `i<k>` → `k`, `N` → ℕ, `*` → `·`, indices as subscripts, `_copy` → a tilde over the symbol (`_copy_marked`; a prime would read as the successor). A negated existence reads `∄`; a negated conjunction of negations is a disjunction; nested disjunctions are flattened. A large operator gets enlarged parentheses only when something follows it (`(∏ V(i)) · x`, but `2 · ∑ i`). In HTML every relation is a no-break unit (`rf-nb`), so a caption wraps only between premises, connectives and keywords.

**Substitution (theorem and rule captions).** A chain reads `from C1, C2 follows H` after simplification: a bound variable that a positive premise defines is replaced by its value everywhere and that premise disappears, when the value says everything the premise said (`_substitutable`): values of the anchor operations `s`, `+`, `·` always; `F(x)` for any other relation (a sequence, `id`, a set variable) only when another premise restricts `x` (`_restricted`), since "∀ x: F(x) = …" would claim F defined everywhere; a fold only when its function is `id` or another premise mentions it (its sequence statement); quotient and remainder never (`a div d` presupposes `d ≠ 0`). A variable typed `y ∈ ℕ` by a premise stays a variable, and a definition stays a premise when another atom states a value of the same application (`_same_application`: `s(w1) = w2` beside `s(w1) = w1`). So `v1 + v2 = v3 ⇒ v2 + v1 = v3` reads `v1 + v2 = v2 + v1`, the F3 product step reads `from V1: [0, s(v1)] → ℕ follows (∏_{i=0}^{v1} V1(i)) · V1(s(v1)) = ∏_{i=0}^{s(v1)} V1(i)`, and `v1 mod v2 = v3 ∧ v2 ∣ v1 ⇒ v3 = 0` keeps its premise. A definition is further kept only when it is acyclic, its value differs from every other kept definition, its variable is used elsewhere (otherwise the premise still states a domain condition), the variable is not used only as a typed element, and no other atom becomes `E = E`; occurrences are read with compiled names expanded. A later premise defining the same variable stays as an equation. When the head's own result is substituted, the given form stands left: `value = head value`. A nested implication reads `∀ y ∈ S: … ⇒ …`, typing folded into the quantifier when every quantified variable has one; substitutions of the enclosing caption stay in force inside it. Binders form one flat set, as in the verifier; since [D-461](../40_decisions.md#d-461) no proof-graph cell binds a name twice. Reformulation captions render both forms without substitution, so the change of form stays visible.

**Applied part.** `rule_application` matches the row against its rule (`_rule_binding` over `_match_rule`: the verifier's implication-check semantics, every name bound anywhere in the rule mapping consistently, the anchor premise included) and lists, on its own line, the instantiation of the rule's chain variables that remain in the rule's rendering and change name (`w2 = v7, w4 = v1`), so the rule as shown turns into the row by renaming; variables substituted away are left out, and a rule with nothing to instantiate shows no application.

**Failure contract.** No fallbacks (Rule 19): an unparsable expression, a compiled name without its binary entry, an anchor without configured slot names, a named predicate without a template (even when the GL binaries hold its body), a template predicate with another argument count (`_check_arity`, also for plain relations that get no caption), a compiled-name arity mismatch, or an implication row whose premises do not instantiate its rule asserts. Captions are presentation only; the verifier reads `processed_proof_graph/` and never the HTML. Regression: `tests/test_readable_form.py` (renderer and generator hooks), `tests/test_html_readable_symbols.py` (order glyphs), `tests/test_fold_completion.py` (sums and products).

**Review.** An independent browser review of the v0.12.0 re-render (screenshots, about 150 grey lines checked against their rows) found no wrong operation, direction, bound or negation and no raw MPL; its findings — inclusions read as equations, tautological uniqueness rules, a vanished `mod` premise, typed compound terms, the prime copy mark, misaligned RULE / APPLIED halves, breaks inside formulas, repeated readings, missing subproof readings and notation key, caption weight — drove the rules above.

---

## The embedded `GL_BINARY_MAP`

Every generated page embeds the compiled-expression definitions its own expressions name — the data required to render an expansion when the user right-clicks a named expression ([D-538](../40_decisions.md#d-538)). Source: `files/GL_binaries/GL_binary_shared.json`, the run's shared registry, which holds every non-atomic entry of every batch (`visu_helpers.read_shared_definition_registry`, read by `build_gl_binary_map` and by `visu_helpers.load_compiled_definitions` for the grey captions).

The popup expands only the names listed in the clicked expression's `data-parts` attribute, so `generate_full_proof_graph.embed_page_definitions` collects the names of the page's `data-parts` attributes and replaces the popup script's single marker with those entries in name order. A page carries about 45 entries (≈6 KB) instead of all 3,349 (691 KB); every page stays self-contained and works offline without a sibling file.

---

## Weaknesses

### Known & tracked

- **HTML export is first-class output** (per the project conventions). Any change that degrades the HTML — visually or in clickable-link correctness — is customer-facing. This is *stated policy*, not *enforced*. No regression test captures HTML output currently.

### Suspected fragility

- **No standalone caption-semantic certificate.** Complete generated-HTML audits check row order, scope ancestry, citation direction and link targets, and caption rendering has unit tests. These checks do not establish that the grey text alone contains a complete proof or that every displayed mathematical abbreviation has the exact source meaning. Every displayed formal row receives common-notation rendering. A caption-only proof requires a separate semantic and completeness contract; visual readability also needs reader-facing evaluation.
- **Definition templates are hand-written.** A new named definition asserts in every caption that names it until it gets its reading in `ReadableForm._named` (or `_template`), its argument count in `_TEMPLATE_ARITY` and its element positions in `_ELEMENT_SLOTS`; numbered compiled names need none.
- **Substitution presupposes definedness heuristically.** A value `F(x)` of a non-total relation is substituted when another premise mentions `x` (a typing, an order, a sequence statement); that premise need not prove `x` in the domain of F, so a reader still takes `F(x)` as defined. The rule removes the unrestricted cases (inclusions, appended sequences, `id`), not every partial one.
- **Fresh names number across a caption.** `y<k>` counts on across independent conjuncts (`y13 … y16`), where restarting per conjunct would read shorter.
- **Graphs older than the binder fix rebind names.** The v0.12.0 incubator graph has 3,515 cells (1,524 `multiplied from` copies and the 1,991 implication rows citing them) that rebind a name an enclosing binder binds ([D-461](../40_decisions.md#d-461)). Captions read a chain's binders as one set, as the prover does, so such a cell reads as the instance GL applied; new graphs carry none ([I-290](../30_invariants.md#i-290)).
- **Theme follows the operating system.** The day or night side is chosen by `prefers-color-scheme`; a page has no switch of its own, and an embedding page (the website's proof windows) cannot force one side.

### Not exercised by tests

- **Per-tag rendering correctness.** Each tag has a specific `format_stack_entries` branch — 30 tag descriptions in `tag_descriptions.json` (post-) plus the meta-tags from the verifier registry that don't surface as row tags. A regression in one (e.g. `or disintegration` — which has nested branch blocks) would only be caught by looking at the relevant chapter.
- **Cross-links.** When a chapter cites another theorem, the link target is computed from the theorem's expression. A rename drift (in either the processor or the generator) could produce dead links. Complete generated-HTML artifact audits check link destinations and local citation direction; interactive browser acceptance separately checks navigation.
- **Matryoshka subproof nesting.** `_render_subproof_card` recurses through `nested_subproofs`. `tests/test_proof_hierarchy.py` directly checks nested card placement, preserved identifiers, continuous numbering and rejection of noncontiguous scopes. Full artifact audits compare actual HTML row order and card ancestry with the processed artifact and every certificate edge.

---

## Matryoshka subproof structure (D-37)

Sub-proofs nest arbitrarily deep. The renderer detects nesting purely from primary namespace ancestry: a row at namespace `A_boundary_<X>` is a child of the scope at namespace `A`. No tag-level marker is required — `validity name` rows provide rich titles when present, but OR-branch sub-subproofs (introduced by `or branch proven` / `or disintegration` rows) are detected via the namespace pattern alone.

Two functions:

- `_partition_stack_subproofs(stack, scope_ns)` at [`generate_full_proof_graph.py`](../../../generate_full_proof_graph.py) — recursive. Returns `(own_rows, subproofs)` where each subproof has a `nested_subproofs` field (the recursive structure).
- `_render_subproof_card(sp, prefix, depth)` at [`generate_full_proof_graph.py`](../../../generate_full_proof_graph.py) — also recursive. Renders one card. Main-book bodies interleave rows and nested cards according to processed order; other books retain rows followed by child cards.

Title inference for a child scope `child_ns` under parent `parent_ns`:

1. **Validity-name lookup.** Scan parent-scope rows for `validity name` rows whose `row[4] == child_ns`. If found, the row's `row[0]` is the implication expression (used as title) and `row[3]` is the goal alias.
2. **`_orint_` payload pattern.** If the child payload matches `orint_(or<N>[…])_((<disjunct>))`, render as "OR-introduction subproof — branch where `<disjunct>` is asserted (of OR `(or<N>[…])`)".
3. **`_ordis_` payload pattern.** If the child payload matches `ordis_(or<N>[…])_((<disjunct>))`, render as "OR-elimination branch — case `<disjunct>` (of OR `(or<N>[…])`)".
4. **Fallback.** Use the raw payload as the title.

Per-depth CSS classes `.subproof-depth-1` through `.subproof-depth-5` differentiate nested cards visually (color shift + indent per level). Depth >5 falls through a CSS `:not` rule with a generic look.

Example: chapter `1209_single_chapter_proof.txt` (the FTA-rung-1 forward direction) renders as:
- 3 depth-1 cards (impl24, impl25, impl26 subproofs).
- 5 depth-2 cards nested inside their parents:
 - 2 `_ordis_` branches under impl24 (case-split of `(or2[i0,v2,i1])`).
 - 2 `_ordis_` branches under impl25 (case-split of `(or2[i0,v4,i1])`).
 - 1 `_orint_` subproof under impl26 (sub-implication for `(or2[i1,v5,i0])`).

All collapsed by default (the existing `.subproof-card.collapsed` toggle behaviour); each card opens independently on click.

---

## Open questions

- **OPEN-16 — RESOLVED.** Induction renders as **one HTML page per theorem**, not three. The generator aggregates the three source chapter files (`<N>_induction_typing.txt`, `<N+1>_check_zero.txt`, `<N+2>_check_induction_condition.txt`) into a single `chapter<M>.html` page with sub-sections. HTML page count therefore equals theorem count; induction contributes three source files and a single-chapter theorem contributes one.
- **OPEN-17 — RESOLVED.** No sitemap is generated by the HTML export. The pages do include `<meta name="robots" content="index, follow, noai, noimageai">` at [`generate_full_proof_graph.py`](../../../generate_full_proof_graph.py) — so crawl behaviour is controlled by meta-robots (allow search indexing; forbid AI training crawlers via `noai` and `noimageai`). Crawlers must reach chapter pages through the index link graph, not a sitemap. For bot-traffic control: the `noai`/`noimageai` directives block compliant crawlers (GPTBot, ClaudeBot, CCBot) but not scrapers disguised as humans — that needs Cloudflare-level control.

---

## See also

- [`10_pipeline/06_process_proof_graph.md`](06_process_proof_graph.md) — upstream producer.
- [`10_pipeline/08_verifier.md`](08_verifier.md) — consumes the same processed proof graph, independently.
- [`20_core_concepts/08_proof_tags.md`](../20_core_concepts/08_proof_tags.md) — tag definitions referenced by the generator's per-tag rendering branches.
- — HTML proof visualization is strategic priority.

## Reverse citation index

`generate_full_proof_graph.py` and `_build_used_by` index renamed theorem
identities once in source order. Each citation selects the first matching record
other than its own, retaining duplicate-identity and self-citation behavior while
avoiding a full theorem-list scan for every reference. Chapter numbers and all
rendered links remain unchanged.

The public `generate_proof_graph_pages` entry confines clickable-string and
parsed-chapter caches to one render via a context-local token. Chapter reads
return independent mutable rows; cached payloads are immutable tuples. Success,
exceptions, and nested calls restore the previous cache. Pure alpha-normalization,
index-prefix formatting, and readable HTML formatting use bounded value caches.
No link cache is retained across graphs.

## Single-chapter proof heading

The machine contract calls every non-induction one-chapter derivation a `single chapter proof` and stores it as `<N>_single_chapter_proof.txt`. The final HTML keeps the human proof depiction: it reads the processed chapter's first row, renders **Contradiction Proof** when that goal row is tagged `contradiction`, and renders **Direct Proof** otherwise. Later contradiction rows are dependency subproofs and never relabel the enclosing chapter.

### Fold values

`visu_helpers.py` and `_fold_value` interpret the eight-argument Fold signature. Ordinary successor traversal over an additive interval reads as a sum for additive accumulation and a product for multiplicative accumulation, from the lower to the upper bound, both included; the identity function reads as the index `i`, any other function as `f(i)`. The value substitutes into a containing expression (a product used as an operand gets enlarged parentheses). Other traversals or accumulation operations keep explicit `fold(…)` notation. The HTML spelling stacks the upper bound, the symbol and `i = lower` (`rf-big` classes).

## Proof-row link identities

Local proof links name only rows actually rendered by `format_stack_entries` in `generate_full_proof_graph.py`. Hidden theorem-import rows retain their cross-chapter theorem links and do not acquire nonexistent local targets. `render_stack_with_subproofs` includes the induction-part prefix in every nested subproof identifier, so separate parts of one theorem page cannot mint the same anchor. Proof rows remain unchanged. Main-book section placement follows the recursive processed order described below.


## Inline main-book subproof cards

`is_main_theorem` in `proof_hierarchy.py` selects only Peano, Gauss and FTA theorem anchors. The proof-section wrappers pass that selection to `render_stack_with_subproofs`. `_render_scope_contents` emits ordinary rows and complete child cards in the processed expanded order, recursively. Each child stays collapsed initially; expand/collapse-all acts on every card inside the main proof section. Existing cross-scope row links still open every collapsed ancestor.

`format_stack_entries` accepts preserved per-scope entry indices, the scope's selected goal highlight and its complete local anchor map when rendering fragments between child cards. The shared expanded line counter is not restarted at a card boundary. Goal labels describe targets and are not new proof steps. Main and incubator use the same recursive scope placement. The processed graph, verifier and connected Lean remain the mathematical source; the renderer introduces no new inference.



## Main-book operation blocks and common notation

`proof_reader_html.py`, `render_reader_proof`, renders the main Peano/Gauss/FTA book from the same `proof_reader.reader_plan` used for processed ordering. Integration, disintegration and scope cards nest in one tree. Each header names its goal, or the disintegration source and products consumed outside the block. OR headers distinguish their case from their actual goal. Original tags and proof rows remain inside; headers are never supplied to the verifier or Lean as inferences.

`complete_caption` provides common notation under every displayed row, including plain relations and scope metadata. Its right citation column uses the exact dispositions returned by `proof_hierarchy.resolved_citations`, shared with the ordering graph. Common-notation citations target the exact retained proof row, opening every enclosing block needed to expose it. Formal MPL citations to externally used operation products retain their closed result-header destination; only enclosing ancestors open for those header links. Achieved MPL goals are orange and their common notation is bold.

Chapter-based part and block numbering adds one component per level. Common-notation mode hides MPL, namespace labels and the READS label, brightens mathematical text, and shows each row's (position/total) badge on the left in the same color. Independent numbered citations stay on the right. Mixed mode retains READS and the formal badge. `visu_helpers.ReadableForm`, `_chain`, lays out multiline HTML implications as indented premises, a large right arrow and the complete head; plain-text notation and unsupported legacy layouts retain their original spelling. Quantifiers and complete expressions remain present.

Block titles support mouse and keyboard activation with `aria-expanded`. Expand/collapse-all controls act within one physical proof section. Initial fragments, history navigation and later citation clicks reveal only the necessary ancestor cards. The section and chapter statistics display both collapsed main rows and total formal proof steps.

### Browser tests

Every HTML-affecting change must pass `.scripts/html_browser_tests.py` (AGENTS.md, *Proof HTML browser tests*), run through the shared queue as a test step after the HTML is regenerated. It serves the book on localhost, drives headless Chrome over the DevTools protocol with real mouse and keyboard input (`.scripts/html_browser_cdp.py`, every click hit-tested), selects its examples from the book's own markers (proof-step counts, block kinds, citations) and checks the cases of [html_browser_acceptance.md](../../testing/html_browser_acceptance.md) that a script can decide; `--list` prints which cases each test covers. A run takes about two minutes and writes `report.json` and screenshots under . Structural Python unit tests of the generator remain separate.

### Common-notation assurance boundary

Formal acceptance applies to the retained processed graph. The notation renderer is tested for expression preservation and reviewed visually; these checks do not constitute an independent formal proof of the grey prose's semantic completeness. Reader headers summarize actual certified operations and never replace their evidence.


The index and chapter pages share `proof_reader_html.notation_toggle_html`, a bold, bordered checkbox callout inviting the reader into common-notation mode. Index theorem captions remain direct chapter links when MPL is hidden. Citation links appear to the right of each common-notation line: a step of the same proof reads as its step number, a theorem as the chapter that proves it, `(chapter 12)` (`proof_reader_html.py`, `theorem_citation_label`; a theorem page is always `chapter<N>.html`, and a theorem without a page in the book has no link). Disintegration headers use the label **Products of disintegration used in proof**; their selection remains the products used outside the block.


Multiline implication parentheses use paired vector fences sized to the complete formula height in visu_helpers._Spelling.parentheses. Inline and plain-text parentheses retain their exact spelling; hidden literal fences preserve accessible reading.

Structured proof-book HTML recursively stacks every conjunction member on its own indented line. Nested implications retain a separate indented premise list, arrow and conclusion; explicit conjunction signs, quantifiers and full-height parentheses retain grouping. Plain-text notation and unsupported legacy rendering stay unchanged. See visu_helpers.py, ReadableForm._conjunction and ReadableForm._chain.

The notation preference is carried by the explicit common/mixed notation query parameter in book links. proof_reader_html.py, notation_preference_script, restores the checkbox and body mode, and preserves the choice through index, chapter, Lean and tag-page navigation, reloads and copied links without requiring shared storage for local files. Fragment identifiers and unrelated query fields remain unchanged; downloads and links outside the book are excluded. History restores the state encoded by each visited address. Every generated supported-book page includes the shared initialization script exactly once.

Implication layout uses content-sized flexible premise and conclusion groups; the arrow does not inherit an equally wide empty premise column. The premise group and the conclusion share the line two to one, each at most its natural width, and the conclusion moves below the premises only when their narrowest forms do not fit beside the arrow (`_READABLE_CAPTION_CSS` in `generate_full_proof_graph.py`). Formulas break only between words. At most 480 px wide, formula pieces may wrap between words, so each side claims at least its natural width or its share of the line (premises 40 %, conclusion 50 %, 6em at least) before it stays beside the arrow, and a common-notation formula takes the full row width below its badge. A negation sign, a restriction bar with its subscript and the opening parenthesis before a large operator never part from their operands (`_Spelling.negation`, `_Spelling.restriction`, `_Spelling.big_parentheses` in `visu_helpers.py`). Index and chapter pages end with one viewport of tail space inside the body, so the last entries scroll under the floating header and a notation switch at the end keeps its reading position. Default index caption links inherit bold grey from their container and brighten in common-only mode. Induction index subentries and chapter headings use induction_part_numbers: the parent chapter plus .1 for typing, .2 for zero and .3 for induction. Physical filenames, source graph identifiers and fragment anchors stay stable.

The implication arrow and conclusion stretch to the full group height and center their contents vertically. A dedicated conclusion-content wrapper keeps the complete formula in one inline formatting context, preserving subscript/superscript placement inside the flex container. Premises remain a centered vertical list.

Structured common notation uses conjunction idempotence to display exact repeated parsed conjuncts once, including under negation and in implication premise lists. ReadableForm._display_parts compares complete formal syntax before name rendering or substitution; it never merges different formulas because their displayed strings coincide. Original MPL, citations, all proof rows, plain-text notation, GL/Lean inputs remain unchanged.

Operation names and explanatory From/Goal/Case/product labels have dedicated semantic spans and stronger color/weight, separate from normally weighted MPL formulas and independently emphasized achieved goals. The complete navigation/legend/notation header is sticky at the viewport top on main chapter pages; the index groups its navigation, title and control in the same floating container. On an explicit checkbox change, notation_preference_script captures the first reading line that begins below the floating header — a proof row, block header or index entry, or a common-notation line inside a row or block header, which both modes show — and restores that element's vertical offset after changing presentation (only when no line begins in view, the innermost line reaching below the header's lower edge serves). A line whose top lies behind the header would vanish behind it when the new mode shortens it; a tall header filling the screen still offers its inner lines. A switch back without scrolling in between keeps the same line, so a round trip returns to the starting view. It leaves disclosure state and proof identity intact; above the proof, when no reading item is visible, no scroll correction is needed.

Multiline premise lists and conjunction groups receive full-height matching parentheses. The shared formatter recognizes a complete already-enclosed outer group without confusing it with several adjacent enclosed expressions, so scope parentheses do not duplicate an existing pair. Plain-text and legacy rendering remain unchanged. Right-side citation groups use their intrinsic content width and shrink only when space is constrained; individual references never split. The navigation script measures the full floating toolbar after mode changes and window resizing, uses that height for proof-target clearance, and anchors reading position beneath its lower edge.

Disintegration headers include Witnesses used in proof on one comma-separated row. proof_reader_html.py, disintegration_witnesses, follows the block's original projection rows in proof order: names bound by a negated implication source and free in its retained result are actual witness uses. Lexically rebound names, unused binders and positive universal variables are excluded; a repeated witness is listed once. Both internal and exported retained projections contribute. MPL uses original names and common notation uses the standard indexed token spelling. Headers introduce no new evidence, and the GL/Lean graph remains unchanged.

## Incubator reader and proof targets

Main and incubator use the same reader controls, recursive notation, operation blocks, witness summaries and popup definitions. `proof_export/corpora.py`, `ORDERED_BOOK_ANCHORS`, selects main anchors plus AnchorIncubator3 and AnchorIncubator8 without expanding the connected Lean corpus. The processor serializes the reshuffled graph; `proof_reader_html.py`, `render_reader_proof`, retains its exact expanded-order assertion. Main sibling links are regenerated whenever incubator chapter numbers change.

`proof_reader_html.py`, `complete_caption`, keeps a visible TO PROVE label on every integration-goal expression in mixed and common-only modes. Removing the machine suffix never turns a target into an established fact. The reader's MPL spans carry the constituent names from `generate_full_proof_graph.py`, `_extract_expr_parts`, in data-parts so the shared right-click popup can show compiled definitions in rows, headers and citations.

Reader prose is generated from the recorded step tag and retained in both notation modes. Integration targets remain explicitly TO PROVE. OR-introduction cases list the actual negated-premise assumption rows; their selected disjunct is the goal. OR-disintegration with no convergence target must have a recorded parent-scope refutation: its header says contradiction and case refuted, never Goal = Case by default. Mixed mode restores the existing subproof and OR explanation paragraphs. The corrected metadata comes from `generate_full_proof_graph.py`, `_partition_stack_subproofs`; `proof_reader_html.py`, `complete_caption` and `render_reader_proof`, render it without changing formal rows.

A refuted OR case highlights its recorded parent-scope contradiction receipt, not the assumed disjunct or an intermediate supporting formula. Its header links to that closing proof step; no new formal row is created.

The floating chapter header includes prominent Expand all and Collapse all buttons acting on every nested card across all proof parts. Section controls remain available with the same prominent styling. The index retains its notation switch without chapter-only block actions.

OR case headers combine the label and assumption instruction into one introductory line in each notation, without repeating Case above Assume this case.

Step captions describe recorded completed derivations. A validity-name result says Obtained as a result of a subproof; imperatives do not make established results sound like new proof obligations. Actual integration targets keep TO PROVE and case headers still identify the assumptions.

The reader follows an ordinary `expansion` row back to its recorded integration carrier when identifying a construction block (`proof_reader.py`, `reader_plan`). An ordinary expansion without an integration ancestor does not create an integration block. Certificate audits derive exact coverage from the current registry, so newly certified strong-induction chapters remain covered.

The graph audit maps certificate actions back to physical proof rows. A recorded integration carrier may have a derived-rule action plus its compact-fold action at the same source line; the audit validates that exact pair, retains every inter-row dependency, and omits only their internal same-row edge. Missing rows, duplicate action identifiers and other duplicate source-line shapes assert. These extra Lean actions do not inflate the processed proof-step KPI.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
