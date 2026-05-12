<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# GL Software Design Document

> **Status:** Living. Written primarily for AI agents who need to understand GL quickly.
> **Scope:** everything the project conventions hints at, plus byte-accurate MPL examples from the pipeline, weaknesses per chapter, and decision history.

---

## What this document is

This document (hereafter **SwDD**) is the on-demand long-form architectural reference for the GL project — the *why-and-where* layer behind the project's brief operational summary at the repo root (which carries rules, vocabulary, build commands). The summary tells you *what the rules are*; the SwDD tells you *why they exist, where the code enforces them, and what happens when they're violated*.

Whenever you (future agent) feel the shape of a question exceed code-level reading or summary density — *"how does the prover route an `or` disintegration into branch scopes?"*, *"what does `addTheoremToMemory` actually do with a freshly-compressed theorem?"*, *"why does the mirror check permute non-anchor premises but not anchor ones?"* — open this document and jump to the relevant chapter.

The SwDD is *not* a tutorial. It assumes you have read the operational summary and the README once. It is a reference — dense, cross-linked, with worked examples drawn from actual pipeline artefacts.

---

## How to read it — three suggested paths

1. **First exposure.** Read [`01_overview.md`](01_overview.md) end-to-end (≈10 minutes), skim [`02_glossary.md`](02_glossary.md), read the [Worked example](#worked-example) chapter start-to-finish. You now have a working mental model.

2. **Targeted lookup.** Jump to the stage you care about via the [Navigation](#navigation) section below. Every stage chapter is self-contained in the sense that it cites its own context — you don't need to read neighbours. Cross-references use relative links.

3. **Debug session.** Scan the [Invariant quick-reference](#invariant-quick-reference) for anything the failing behaviour might be violating, then drill into the relevant stage chapter's *Weaknesses* subsection. Known fragilities are tagged as such.

---

## Reading conventions

This document is written to tight conventions so that it stays greppable and scannable over time.

### MPL quoting

Every MPL expression quoted in this document is either:

- **Verbatim** — a byte-accurate copy-paste from a real pipeline artefact on disk (chapter file, `theorems.txt`, `global_theorem_list.txt`, `externally_provided_theorems.txt`, etc.). Verbatim quotes appear inside ` ```text ` or ``` ```mpl ``` fences. Where the quote is not self-explanatory, an annotation in prose follows immediately.
- **Schematic** — a template with obvious placeholder names (`α`, `β`, `a`, `b`, `bound_vars`, etc.) used only to explain structural patterns. Schematic forms appear in *inline* ` `code` ` form or in clearly-labelled "schema" blocks, never in ` ```text ` fences.

If you see an MPL expression in a ` ```text ` fence, trust that it exists on disk exactly like that.

### Code citations

Source citations use the `file.ext:line` form, e.g. `prover.cpp` for a function definition. Citations refer to the **definition** site unless suffixed with `(decl)`. Line numbers may drift under edits — when in doubt, grep for the symbol name. If you (future agent) find a citation that has drifted, **update it** — don't leave dead references.

For C++, the canonical files live under `GL_Quick_VS/GL_Quick/src/`. Paths are shortened to just the filename in citations since there is no ambiguity.

### Weaknesses tagging

Every pipeline and core-concept chapter has a `## Weaknesses` subsection at the end, with three buckets:

- **Known & tracked** — documented, understood, has an open plan or known mitigation. Example: *"induction typing soundness — see `docs/induction_typing_plan.md` and invariant [I-18](30_invariants.md#i-18)."*
- **Suspected fragility** — looks like it might break under specific conditions; not currently exercised or confirmed. Example: *"Pass B's single-input-operator gate relies on `inputIndices.size == 1`; if a new operator with `input_args.size == 1` is introduced with different semantics (e.g. a non-admissive projection), the gate admits it silently."*
- **Not exercised by tests** — has no regression coverage; surviving only on code review.

If you (future agent) notice a weakness not listed, add it. Prefer honest over silent.

### Tense and person

Present tense, impersonal. *"The prover performs hash bursts."* not *"We do hash bursts."* or *"The prover will perform..."*. The doc describes current system state.

### Full words

Per the project conventions, prose uses full words. Code identifiers appear as-is inside backticks — do not expand `mb` to `memoryBlock` when citing a specific variable from the code.

### Markdown

ATX-style headings (`#`, `##`, `###`). Line wrapping is up to the renderer — source text is unwrapped. Tables used where structure is genuinely tabular; bulleted lists where order matters but structure is light.

## Navigation

### Root-level sections

| File | Depth | Purpose |
|---|---|---|
| [`01_overview.md`](01_overview.md) | [DRAFT] | GL in one page — the dataflow, entry points, artefact layout. Read this first. |
| [`02_glossary.md`](02_glossary.md) | [DRAFT] | Every domain term used in the codebase. |
| [`03_mpl.md`](03_mpl.md) | [DRAFT] | The MPL language — grammar, variable conventions, type labels, shape patterns, reading exercises. |
| [`04_configs.md`](04_configs.md) | [DRAFT] | All `files/config/*.json` files — top-level schema, per-expression fields, `parameters` and `prover_parameters` full reference, variant comparison, loader cartography. |
| [`30_invariants.md`](30_invariants.md) | [DRAFT] | 35 numbered + named invariants with cross-linked anchors (I-1..I-35; numbering reflects insertion order, not strict sequence). |
| [`40_decisions.md`](40_decisions.md) | [DRAFT] | 54-entry dated trade-off log (D-1..D-55 with D-38 retired; D-44/D-45 inherited from sibling; D-53 = mail-unification renumbered from main's D-46 merge to disambiguate from incub_fix's cross-pair `equality2` D-46). |
| [`50_gotchas.md`](50_gotchas.md) | [DRAFT] | 43-entry chronic-failure-mode catalog (G-1..G-40 with gaps; numbering reflects insertion order, not strict sequence). |
| [`worked_example.md`](worked_example.md) | [DRAFT] | One Peano theorem walked through all ten pipeline stages end-to-end. |
| [`induction_typing_plan.md`](induction_typing_plan.md) | [FULL, pre-existing] | Architecture plan for the induction-typing soundness fix (in-flight). |
| [`_meta/testing.md`](_meta/testing.md) | [DRAFT] | In-tree unit-test harness — `gl_quick.exe --unit-tests` gate added. |

### Pipeline (the data arrow, stage by stage)

| File | Depth | Stage |
|---|---|---|
| [`10_pipeline/01_mpl_definitions.md`](10_pipeline/01_mpl_definitions.md) | [DRAFT] | MPL, `files/definitions/*.txt`, GL-binary schema |
| [`10_pipeline/02_conjecturer.md`](10_pipeline/02_conjecturer.md) | [DRAFT] | `conjecturer.cpp` — enumeration + filtering |
| [`10_pipeline/03_ce_filter.md`](10_pipeline/03_ce_filter.md) | [DRAFT] | `files/simple_facts/*` + peek-and-prune counterexample filter |
| [`10_pipeline/04_prover.md`](10_pipeline/04_prover.md) | [DRAFT] | The main proof engine — `prover.cpp`, `memory.hpp`, `compiler.cpp` |
| [`10_pipeline/05_compressor.md`](10_pipeline/05_compressor.md) | [DRAFT] | `compressor.cpp` — post-proof redundancy elimination |
| [`10_pipeline/06_process_proof_graph.md`](10_pipeline/06_process_proof_graph.md) | [DRAFT] | `process_proof_graphs.py` — variable renaming, pruning |
| [`10_pipeline/07_html_export.md`](10_pipeline/07_html_export.md) | [DRAFT] | `generate_full_proof_graph.py`, `visu_helpers.py` |
| [`10_pipeline/08_verifier.md`](10_pipeline/08_verifier.md) | [DRAFT] | `verifier.py` — external proof checker |
| [`10_pipeline/09_incubator.md`](10_pipeline/09_incubator.md) | [DRAFT] | Ground-level fact-table generation |

### Core concepts (cross-cutting)

| File | Depth | Concept |
|---|---|---|
| [`20_core_concepts/01_logic_blocks.md`](20_core_concepts/01_logic_blocks.md) | [DRAFT] | LB grid, parent/child hierarchy, the `Memory` class |
| [`20_core_concepts/02_hash_engine.md`](20_core_concepts/02_hash_engine.md) | [DRAFT] | Hash-based inference, `HashMemory`, encoded expressions |
| [`20_core_concepts/03_mail_system.md`](20_core_concepts/03_mail_system.md) | [DRAFT] | Inter-block communication, `Mail`, broadcast rules |
| [`20_core_concepts/04_validity_stack.md`](20_core_concepts/04_validity_stack.md) | [DRAFT] | Scope names, `NameMap::encodePush`, `stackOfValidity`, `pairMap` |
| [`20_core_concepts/05_equivalence_classes.md`](20_core_concepts/05_equivalence_classes.md) | [DRAFT] | `EquivalenceClass`, negated-equality expansion |
| [`20_core_concepts/06_anchors_and_scopes.md`](20_core_concepts/06_anchors_and_scopes.md) | [DRAFT] | `AnchorPeano`, `AnchorGauss`, `AnchorIncubator`, anchor handling |
| [`20_core_concepts/07_or_branching.md`](20_core_concepts/07_or_branching.md) | [DRAFT] | OR disintegration, branch scopes, convergence |
| [`20_core_concepts/08_proof_tags.md`](20_core_concepts/08_proof_tags.md) | [DRAFT] | The 31 `TAG_CHECKERS` + non-checker categories (30 distinct tag names; one shared checker. D-35 added `or branch proven` + `or branch assumption`; D-41 added `definition set consistency` as a per-row meta-check.) |

### Worked example

[`worked_example.md`](worked_example.md) — **must-read.** One Peano theorem (`(>[i0,s,+,i1](AnchorPeano[N,i0,s,+,*,i1])(>[v1](in2[i0,v1,s])(in3[i0,i1,v1,+])))` — "0 + 1 = 1") traced through every pipeline stage with byte-accurate artefacts.

### How to extend this document

1. Find the right file. Don't create a new top-level file unless a concept truly crosscuts every chapter.
2. Preserve the `[STUB] / [DRAFT] / [FULL]` tag at the top of each file — update it when you substantially extend.
3. If you (future agent) make a decision worth remembering, add it to [`40_decisions.md`](40_decisions.md) with today's date.
4. If you notice a fragility, add it to the chapter's `Weaknesses` subsection AND cross-reference from [`50_gotchas.md`](50_gotchas.md) if it's recurring.
5. Update only with things that will outlive this project state — the SwDD is where project-state knowledge lives.

---

## Invariant quick-reference

The full details live in [`30_invariants.md`](30_invariants.md). This table is for fast visual recall.

| ID | Name | Scope | Cross-link |
|---|---|---|---|
| I-1 | Precompile structural operators on every theorem-load path | Compiler / prover | [`30_invariants.md#i-1`](30_invariants.md#i-1) |
| I-2 | Non-main `validityName` minted only via `NameMap::encodePush` | Validity stack | [`30_invariants.md#i-2`](30_invariants.md#i-2) |
| I-3 | `NameMap::decode` / `idToSub[]` return refs — copy before any nested mint | Validity stack | [`30_invariants.md#i-3`](30_invariants.md#i-3) |
| I-4 | `reconstructImplicationFullBind` only at disintegration/integration sites | Compiler | [`30_invariants.md#i-4`](30_invariants.md#i-4) |
| I-5 | `reconstructImplication` (non-FullBind) for theorem-level reconstruction | Compiler | [`30_invariants.md#i-5`](30_invariants.md#i-5) |
| I-6 | Pass B single-input-operator gate (do not widen) | Prover disintegration | [`30_invariants.md#i-6`](30_invariants.md#i-6) |
| I-7 | Pass B + back-reformulation + hypo-disintegration all guarded by `!parameters.ban_disintegration` (decoupled from `incubator_mode`) | Prover disintegration | [`30_invariants.md#i-7`](30_invariants.md#i-7) |
| I-8 | Trivial equality `(=[x,x])` forbidden in head only | Conjecturer | [`30_invariants.md#i-8`](30_invariants.md#i-8) |
| I-9 | Equality mirror guarded by `args[0]!= args[1]` | Compiler | [`30_invariants.md#i-9`](30_invariants.md#i-9) |
| I-10 | Chapter v-numbering seeded from theorem expression | Process proof graph | [`30_invariants.md#i-10`](30_invariants.md#i-10) |
| I-11 | Anchor variables must not appear in `>[...]` bound-variable lists | Theorem construction | [`30_invariants.md#i-11`](30_invariants.md#i-11) |
| I-12 | `addStatement` applies equivalence classes to `!(=[a,b])` one-sidedly | Prover | [`30_invariants.md#i-12`](30_invariants.md#i-12) |
| I-13 | `ChunkPool` uses static `char[]` — never `malloc`/`new` | Memory | [`30_invariants.md#i-13`](30_invariants.md#i-13) |
| I-14 | Never destroy git history | Process | [`30_invariants.md#i-14`](30_invariants.md#i-14) |
| I-15 | `git reset --hard` only; never soft or mixed | Process | [`30_invariants.md#i-15`](30_invariants.md#i-15) |
| I-16 | `verifier.py` is sacred — failures are real bugs | Verifier | [`30_invariants.md#i-16`](30_invariants.md#i-16) |
| I-17 | `savedStartInt` freshness check assumes one monotonic counter | Prover disintegration | [`30_invariants.md#i-17`](30_invariants.md#i-17) |
| I-18 | Induction scheduled on a bound variable must first prove its typing | Prover | [`30_invariants.md#i-18`](30_invariants.md#i-18) |
| I-19 | Asserts are first-class — never weaken or remove to pass a test | Process | [`30_invariants.md#i-19`](30_invariants.md#i-19) |
| I-20 | Auto-commit + push on source/config changes, detailed message | Process | [`30_invariants.md#i-20`](30_invariants.md#i-20) |
| I-21 | `internalMailIn` cleared at top of hashburst after absorb, not end | Prover / mail | [`30_invariants.md#i-21`](30_invariants.md#i-21) |
| I-22 | `rejectedMapIntegration` revival does NOT clean admission-map entry | Prover / admission | [`30_invariants.md#i-22`](30_invariants.md#i-22) |
| I-23 | Spontaneous compact operator names stable across batches (shared registry) | Compiler + Python pipeline | [`30_invariants.md#i-23`](30_invariants.md#i-23) |
| I-24 | `multiplyImplication` may not equate two distinct free `u_*` anchor parameters | Prover + verifier | [`30_invariants.md#i-24`](30_invariants.md#i-24) |
| I-25 | `addStatement` returns `ExpressionWithValidity` pairs; cross-scope deposits ride through `newStatements` | Prover / equivalence classes | [`30_invariants.md#i-25`](30_invariants.md#i-25) |
| I-26 | Mail-out implications/statements MAIN-ONLY; mail-out exprOriginMap ALL-SCOPES | Mail subsystem | [`30_invariants.md#i-26`](30_invariants.md#i-26) |
| I-27 | Site F / Site H — ancestor-scan dedupe at `addExprToMemoryBlock` entry | Prover kernel entry | [`30_invariants.md#i-27`](30_invariants.md#i-27) |
| I-28 | Cross-LB writes during `proveKernel`'s parallel phase forbidden — defer to post-`pool.join` collectors | Prover / parallelism | [`30_invariants.md#i-28`](30_invariants.md#i-28) |
| I-29 | Variable-port type consistency: every chapter-row variable connects ports with identical type labels | Compiler + verifier | [`30_invariants.md#i-29`](30_invariants.md#i-29) |
| I-30 | `applyEquivalenceClassToRejectedMapIntegration` is additive — original rmi entries are never erased | Prover / equivalence classes | [`30_invariants.md#i-30`](30_invariants.md#i-30) |
| I-31 | `updateEquivalenceClasses` ancestor-pass never modifies ancestor-scope class state | Prover / equivalence classes | [`30_invariants.md#i-31`](30_invariants.md#i-31) |
| I-32 | Cross-pair `equality2` emission gated on existing class/LB origin | Prover / equivalence classes | [`30_invariants.md#i-32`](30_invariants.md#i-32) |
| I-33 | `mergeTwoEquivalenceClasses` cross-vN preconditions (ancestor-only direction; eqArgs-subset assert is same-vN only) | Prover / equivalence classes | [`30_invariants.md#i-33`](30_invariants.md#i-33) |
| I-34 | Cross-substitution `equality1` emission gated on existing target origin (`applyEquivalenceClass` only) | Prover / equivalence classes | [`30_invariants.md#i-34`](30_invariants.md#i-34) |
| I-35 | ~~`addOrigin` cap-full preference~~ — superseded by [D-51](40_decisions.md#d-51), 2026-05-08 | Prover / origin tracking | [`30_invariants.md#i-35`](30_invariants.md#i-35) |
| I-38 | `implication`-row deposit lives at `deeperOf(constituents)` — verifier-side mirror of `generateEncodedRequests` accumulation | Verifier / validity stack | [`30_invariants.md#i-38`](30_invariants.md#i-38) |
| I-37 | Algebra `rejectedMap` is never written by equi-class application — keeps production-site `disintegration` provenance intact | Prover / equivalence classes | [`30_invariants.md#i-37`](30_invariants.md#i-37) |
| I-36 | Algebra equi-class rewrites preserve positional collision pattern — drop any rewrite that collapses previously-distinct arg slots | Prover / equivalence classes | [`30_invariants.md#i-36`](30_invariants.md#i-36) |

---

## Open questions

Each entry owns a location — the chapter where the full write-up lives. As of 2026-04-23, every question enumerated during the initial SwDD build has been either resolved, partially resolved, or scoped out (remaining-open entries are flagged by reason).

### Resolved

- **OPEN-1 — PARTIAL.** All-disjuncts-refuted OR branch: each branch's contradiction LB proves `!disjunct_i`; parent scope ends up with `∧ !disjunct_i` across all branches. `cleanUpOrIntegrationBranches` is a *post-convergence* cleanup, not invoked here. No specific vacuous-truth emission on this path. Full write-up: [`20_core_concepts/07_or_branching.md`](20_core_concepts/07_or_branching.md#open-questions).
- **OPEN-2 — RESOLVED (correction).** `origin` IS a real validator at [`verifier.py–2749`](../verifier.py); just lives outside the `TAG_CHECKERS` dispatch table. Directly called from `verify_chapter`, counts success/failure. See [`10_pipeline/08_verifier.md`](10_pipeline/08_verifier.md#open-questions).
- **OPEN-3 — RESOLVED (final stance).** Empirical, reversion. No theoretical justification documented. Widening the gate remains forbidden ([I-6](30_invariants.md#i-6)). Close unless/until a theoretical argument is developed.
- **OPEN-4 — RESOLVED.** `ExpressionAnalyzer::operators` populated at [`prover.cpp–200`](../GL_Quick_VS/GL_Quick/src/prover.cpp) from `coreExpressionMap`. Canonical by construction.
- **OPEN-6 — RESOLVED.** `performDisintegration` absent from source; the project conventions stale. Real surface: `ce::disintegrateImplication` + `disintegrateExpr2`.
- **OPEN-8 — RESOLVED (no assert; recommended to add).** No `rightMax ≤ 3` assert in any form. Misconfiguration → silent OOM. Recommended fix: add an assert at `createMapAnchor` entry. See [`10_pipeline/02_conjecturer.md`](10_pipeline/02_conjecturer.md#open-questions).
- **OPEN-9 — CORRECTED.** `apply_in_premise_filter` is declared + loaded but dead code — never consulted. Setting `false` in a config has no effect. See [`10_pipeline/02_conjecturer.md#open-questions`](10_pipeline/02_conjecturer.md#open-questions) (OPEN-9 corrected entry).
- **OPEN-10 — RESOLVED.** `loadFactsForCEFiltering` uses generic `addExprToMemoryBlock` — polymorphic over fact shape. Atomic facts → statements; implication-shaped facts → hash rules.
- **OPEN-11 — RESOLVED.** No per-conjecture timeout — only the batch-wide `numberIterationsConjectureFiltering` budget. A slow conjecture can starve others; not currently a problem in practice.
- **OPEN-12 — RESOLVED.** Induction variable = every `digitArg` (from `findDigitArgs`). Prover spawns one recursion sub-block per digit-arg; the succeeding triad's digit-arg becomes the reference-column entry.
- **OPEN-13 — RESOLVED.** `std::stable_sort` at [`compressor.cpp`](../GL_Quick_VS/GL_Quick/src/compressor.cpp). Output deterministic.
- **OPEN-14 — RESOLVED.** x-prefix emitted at [`prover.cpp–3449`](../GL_Quick_VS/GL_Quick/src/prover.cpp) during anchor handling.
- **OPEN-15 — RESOLVED.** A chapter-less survivor is silently invisible to chapter-level audit. Induction triads raise `AssertionError` on missing; direct/mirrored/reformulated fail silently. Hardening: cross-check every `global_theorem_list.txt` row has a file.
- **OPEN-16 — RESOLVED.** Induction renders as **one** HTML page per theorem with typing/zero/condition sub-sections.
- **OPEN-17 — RESOLVED.** No sitemap generated. Pages carry `<meta name="robots" content="index, follow, noai, noimageai">` — allows search indexing, blocks compliant AI crawlers.
- **OPEN-18 — PLANNED.** Induction-typing checker: new `"induction typing"` key in `TAG_CHECKERS`; walks every `method == "induction"` row; verifies typing chapter exists + asserts its head. Full spec in [`induction_typing_plan.md`](induction_typing_plan.md) stage 3.
- **OPEN-19 — RESOLVED.** j-copy strategy: always emit j0/j1 anchor-matching variants (`max_j=2`); additionally break repeats in i-value args when facts have them.
- **OPEN-20 — RESOLVED.** `compressed_out_theorems.txt` holds eliminated externals (compressor-removed theorems from the external pool). Written at [`compressor.cpp`](../GL_Quick_VS/GL_Quick/src/compressor.cpp), appended at [`run_modes.cpp`](../GL_Quick_VS/GL_Quick/src/run_modes.cpp).
- **OPEN-21 — RESOLVED.** `buildPerCoreMailboxes` at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp). Per-LB mailbox of `logicalCores` slots; effectively per-LB with `logicalCores=1`.
- **OPEN-22 — RESOLVED.** `smashMail` at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) sorts recipients by `exprKey`; intra-recipient merge is set-based. Byte-reproducible.
- **OPEN-CFG-1 — RESOLVED.** `skip_eq_classes` bypasses equivalence-class registration in `addStatement`. Used by incubator to contain blow-up; mirrors still emitted, class-propagation dropped.
- **OPEN-CFG-2 — RESOLVED (dead field).** `fact_variable_kinds` is loaded but never consumed. Cleanup candidate.
- **OPEN-CFG-3 — RESOLVED.** `operator_threshold` feeds conjecturer filter-cascade gates at [`conjecturer.cpp–2599`](../GL_Quick_VS/GL_Quick/src/conjecturer.cpp) and [`:3580–3581`](../GL_Quick_VS/GL_Quick/src/conjecturer.cpp). `max_size_mapping_def_set` feeds `createMap(N)` at [`conjecturer.cpp`](../GL_Quick_VS/GL_Quick/src/conjecturer.cpp).
- **OPEN-CFG-4 — RESOLVED.** `.bak` files are never read — exact-filename load path, no glob. Manual backup only.
- **OPEN-MPL-1 — RESOLVED (partial normalisation).** Double negation cancelled at two sites ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp), [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)); no universal normaliser. Raw `!!X` in an external theorem would hash distinctly from `X`. Hardening: canonicalise at load.

### Remaining open

- **OPEN-5.** Compressor Phase 1 memory footprint on the Gauss batch — genuinely requires measurement, not code-reading. Measure before FTA.
- **OPEN-7 — PARTIAL.** `fXY` / `fXYZ` have `category = "existence"` in the generated binary; exact inference site inside the C++ compiler parse-tree walk has not been located (no `category = "existence"` string-match in `compiler.hpp`). Requires targeted read of the parse path.
- **OPEN-MPL-2.** `(&X)` and `(>[])` parser acceptance — untested. Neither shape is emitted by the conjecturer; behaviour moot unless an external theorem uses it. Low priority.

---

## Meta

- **License.** This document is dual-licensed under AGPLv3 and commercial terms alongside the code it describes. See [https://generative-logic.com/license](https://generative-logic.com/license).
- **the project conventions overlap.** Intentional while this document is maturing. Once [`01_overview.md`](01_overview.md) and each stage chapter reach `[FULL]`, the corresponding section in the project conventions will be trimmed to a one-liner pointer.
- **Primary customer.** Future agents. Every design choice in the document (stable anchors, file + symbol citations, invariant numbering, weakness tagging) exists to make consultation cheap for an LLM operating under token pressure.

---

## Commercial use

If GL is being evaluated for a closed-source product, a hosted service, or any deployment where the AGPLv3 network-distribution clause is incompatible with your use case, the commercial license removes that obligation and adds warranty + IP indemnity.

- Commercial licensing page: [https://generative-logic.com/license](https://generative-logic.com/license)
- Direct contact: [info@generative-logic.com](mailto:info@generative-logic.com)

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
