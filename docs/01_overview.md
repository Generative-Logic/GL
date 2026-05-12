<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Overview — GL in one page `[DRAFT]`

> If you read only one file in this documentation, read this one.

---

## What GL is

**Generative Logic (GL)** is a deterministic computer architecture for automated mathematical reasoning. It starts from a small set of user-supplied axiomatic definitions written in MPL (Mathematical Programming Language), and systematically explores their deductive neighbourhood — enumerating conjectures, filtering them with counterexamples, and attempting to prove or disprove each one by hash-based inference over a distributed grid of logic blocks. No human guidance, no goal-setting — the system discovers and proves theorems autonomously, with full provenance for every emitted fact.

GL is *not* a general theorem prover in the LEAN / Coq / Isabelle sense. It does not interpret tactics, does not accept human-written proofs as input, and does not aim for expressiveness parity with those systems. Its value comes from the opposite direction: a fixed, auditable execution model whose proof graphs are fully mechanical and designed to be verified, cached, and shipped as a commodity. The output of a GL run is a corpus of theorems plus the provenance-complete proof graph that derived each one — human-readable as HTML, machine-verifiable by an independent external checker.

Canonical results on the current public release:

- **Peano batch** — commutativity and associativity of `+` and `*`, distributivity, ≈58 theorems.
- **Gauss batch** — the autonomous-discovery summation formula in its division-free form `n · (n+1) = 2 · Σ i`.
- **Branching milestone** — `or0[7,2,1,3]` ("∀n ∈ N: n = 0 ∨ ∃ predecessor"), the foundation of the Fundamental Theorem of Arithmetic ladder.

The next milestone is the Fundamental Theorem of Arithmetic (FTA). See `fta_ladder/README.md` for the rung-by-rung index and `fta_ladder/rung<N>/current_proof_state.md` for each rung's in-flight state.

---

## The data arrow

A single GL run, invoked as `python main.py`, drives the full pipeline:

```
              definitions (MPL)         config (JSON)
                    │                        │
                    ▼                        ▼
              ┌──────────────────────────────────────┐
              │  1.  Definition compilation          │   compiler.cpp
              │      MPL → GL binary + anchors       │   files/GL_binaries/*.json
              └──────────────────────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────────────┐
              │  2.  Conjecture generation           │   conjecturer.cpp
              │      combinatorial enumeration        │   files/theorems/theorems.txt
              │      over regularised structures      │
              └──────────────────────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────────────┐
              │  3.  Normalisation + type filter     │   expression_utils.py
              │      signature canonicalisation       │
              └──────────────────────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────────────┐
              │  4.  Counterexample (CE) filter      │   conjecturer.cpp (C++ path)
              │      peek-and-prune against small     │   files/simple_facts/*.txt
              │      arithmetic tables                │   files/theorems/filtered_conjectures.txt
              └──────────────────────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────────────┐
              │  5.  Prover phase                    │   prover.cpp + memory.hpp
              │      batched LB execution            │   ExpressionAnalyzer state
              │      (warm-up + main iterations)     │   files/theorems/proved_theorems.txt
              └──────────────────────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────────────┐
              │  6.  Compressor                      │   compressor.cpp
              │      greedy redundancy elimination    │   (prunes proved_theorems.txt)
              └──────────────────────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────────────┐
              │  7.  Raw proof graph emission        │   visualizer.cpp
              │      per-theorem chapter txt         │   files/raw_proof_graph/*.txt
              └──────────────────────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────────────┐
              │  8.  Process proof graph             │   process_proof_graphs.py
              │      variable renaming, pruning       │   files/processed_proof_graph/*.txt
              └──────────────────────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────────────┐
              │  9.  HTML export                     │   generate_full_proof_graph.py
              │      navigable hyperlinked proofs     │   files/full_proof_graph/*.html
              └──────────────────────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────────────┐
              │  10. External verifier               │   verifier.py
              │      independent proof checker        │   stdout: 28-ish-line tally
              └──────────────────────────────────────┘
```

Stages 1–7 execute inside the C++ binary `gl_quick.exe`. Stages 8–10 are Python. `run_modes.py` is the orchestrator — it invokes `gl_quick.exe --conjecture <tag>` for stage 2, then `gl_quick.exe <tag>` for stages 1 + 4 + 5 + 6 + 7 (the native prover's `run_modes::fullRun` in `run_modes.cpp`), then runs stages 8–10 as Python.

Multi-batch runs (Peano → Gauss, or Incubator → Peano → Gauss) loop stages 1–10 with the previous batch's proved theorems re-loaded as external theorems for the next batch.

---

## Entry points

| Path | Purpose |
|---|---|
| `main.py` | Top-level Python entry. Calls `run_modes.full_run`. |
| `run_modes.py` → `full_run` | Orchestrates the full multi-batch pipeline. |
| `run_modes.py` → `incubator_run` | Separate ground-level-facts pipeline (own config, own theorem storage, own CE tables — does not touch `files/theorems/`). |
| `gl_quick.exe <tag>` | Native prover (single batch). Invoked as a subprocess by `run_modes.py`. |
| `gl_quick.exe --conjecture <tag>` | Conjecturer only — writes `files/theorems/theorems.txt`. |
| `gl_quick.exe --mirror-externals <tag>` | Rebuild `files/theorems/compressed_external_theorems.txt` without running the prover. |
| `verifier.py` | External proof checker. Reads `files/processed_proof_graph/` + `files/GL_binaries/`. Emits a 28-ish-line tally to stdout. |
| `analyze_incubator.py` | Standalone post-run analysis of incubator batch — compares proved vs. conjectured. |

---

## Artefact layout

```
files/
  definitions/         MPL definitions — user-authored input
  config/              batch configurations (Peano / Gauss / Incubator variants)
  GL_binaries/         compiled definition structures (generated by every prover run; tracked in git so cross-batch consumers and the verifier can resolve compact-operator names without first running the pipeline)
  theorems/            main-pipeline theorem files
    theorems.txt                           conjecturer output (unfiltered)
    filtered_conjectures.txt               conjectures that survived CE filter
    reshuffled_theorems.txt                post-normalisation reshuffled forms
    reshuffled_mirrored_theorems.txt       mirrors of the above
    proved_theorems.txt                    survivors of the prover phase (compressor-pruned)
    externally_provided_theorems.txt       user-supplied external theorems (never emptied by a run)
    compressed_external_theorems.txt       rebuilt each run (originals + mirrors, compressor-pruned)
    or_pairs.txt                           OR-shape theorem pair registry
  theorems_incubator/  parallel tree for incubator mode (own proved_theorems.txt etc.)
  simple_facts/        arithmetic tables for CE filtering (ground-level facts)
  raw_proof_graph/     prover output — pre-rename chapters
  processed_proof_graph/
    <N>_direct_proof.txt
    <N>_check_zero.txt + <N+1>_check_induction_condition.txt + <N+2>_induction_typing.txt   (induction triad)
    <N>_mirrored_statement.txt
    <N>_reformulated_statement.txt
    <N>_back_reformulated_statement.txt
    <N>_or_theorem.txt
    global_theorem_list.txt                every theorem + its method + its reference
    external_theorems.txt                  raw + renamed external theorems for verifier
  full_proof_graph/    generated HTML (index.html + chapter<N>.html + tags.html)
```

Artefacts under `files/raw_proof_graph/`, `files/processed_proof_graph/`, `files/full_proof_graph/`, `files/theorems/`, `files/GL_binaries/`, and `files/simple_facts/` are all run-output. Only `files/definitions/` and `files/config/` are user-authored input. `files/theorems/externally_provided_theorems.txt` is also user-authored (a durable input) — it is *not* emptied by a run.

---

## Key source files

Python (root level, each carries the dual-license header):

| File | Lines | Role |
|---|---:|---|
| `main.py` | ≈50 | Entry point. |
| `run_modes.py` | ≈340 | Orchestration. |
| `configuration_reader.py` | ≈550 | Reads JSON config, exposes `input_args` / `output_args` / `definition_sets` / `theorems_folder`. |
| `expression_utils.py` | ≈290 | String-parsing helpers surviving from the pre-C++-conjecturer era. |
| `incubator_to_simple_facts.py` | ≈316 | Converts proved incubator theorems to CE-filter tables (j-copy strategy). |
| `process_proof_graphs.py` | ≈600 | Variable renaming + pruning + global theorem list generation. |
| `generate_full_proof_graph.py` | ≈2250 | Processed proof graph → navigable HTML. |
| `visu_helpers.py` | ≈480 | HTML rendering helpers. |
| `verifier.py` | ≈2785 | **Independent** external proof checker. Must not import from prover or `expression_utils`. |
| `analyze_incubator.py` | ≈430 | Standalone post-run analysis. |
| `parameters.py` | ≈30 | A single debug flag, kept around for historical reasons. |

C++ (under `GL_Quick_VS/GL_Quick/src/`, each carries the dual-license header):

| File | Lines | Role |
|---|---:|---|
| `main.cpp` | ≈150 | Binary entry. Parses `--conjecture` / `--mirror-externals`. Dispatches. |
| `run_modes.cpp` | ≈310 | Thin C++ orchestrator — `fullRun(anchor_id)` runs prover → compressor → raw graph. |
| `parameters.hpp` | ≈80 | `ProverParameters`, `ExecutionParameters`. |
| `compiler.cpp/.hpp` | ≈280 / 1470 | MPL → compiled `CoreExpressionConfig`, anchors, `makeAnchorSignature`. |
| `memory.hpp` | ≈1500 | Per-LB proof state — `Memory`, `LogicalEntity`, `Mail` (routing channels `mailIn`/`mailOut` plus integration-revival via `Memory::internalMailIn`, [D-19](40_decisions.md#d-19); [D-53](40_decisions.md#d-53) unification 2026-05-07, renumbered from main's D-46), `HashMemory` (incl. `rejectedMapIntegration` + `varsInRejectedMapIntegrationKeys` cache), `EquivalenceClass`, `EncodedExpression`. |
| `prover.cpp/.hpp` | ≈8770 / 5925 | Main proof engine. All `ExpressionAnalyzer` state. |
| `visualizer.cpp` | ≈740 | `generateRawProofGraph`, `buildStack`, `findEnds`, `exportCompiledExpressionsJSON`. |
| `conjecturer.cpp/.hpp` | ≈4570 / 450 | C++ conjecture generator (replaces the retired Python `create_expressions.py`). |
| `compressor.cpp/.hpp` | ≈390 | Post-proof redundancy elimination. `Compressor::run`, `isDerivable`. |

Build: MSBuild (VS 2022 Community). Single output binary: `GL_Quick_VS/GL_Quick/gl_quick.exe`. See the project conventions for the exact build command.

---

## Core idea — why this architecture

GL's central bet is that **mathematical inference is a memory-access problem**, not a search problem. Traditional theorem provers explore a search tree of tactics or rewrites; GL reformulates inference as hash-table lookup:

1. Every proved expression is stored in a per-LB **hash memory** keyed by its structural signature.
2. Compiled implications (the axioms + previously-proved theorems) are stored as hash rules: `premise_signature → conclusion_template`.
3. On every step, the LB formulates hash requests from its known expressions. A hit fires the implication; the conclusion is emitted.

The upshot: each step is deterministic and cheap (O(1) hash lookup). Exploration happens by breadth — multiple LBs firing in parallel, communicating via the mail system between cycles.

This design has consequences that shape everything else in the codebase:

- **Expressions are structurally canonical.** Normalisation (`reshuffledTheorems`) and compilation (`CoreExpressionConfig`) ensure hash-signature uniqueness. Two theorems that differ only in bound-variable names must hash identically.
- **Provenance is first-class.** Every hit records who cited whom, into `exprOriginMap` — the raw material for the proof graph.
- **Scope is explicit.** A proof under a hypothesis lives in its own `validityName` scope, which is literally a stack of names encoded via `NameMap::encodePush`. Escalating a conclusion from a hypothesis scope to its parent requires explicit integration.

The **Logic Block grid** abstraction (see [`20_core_concepts/01_logic_blocks.md`](20_core_concepts/01_logic_blocks.md)) is what makes this scale: each LB is a self-contained hash engine with local memory, communicating only between cycles. The grid is conceptually distributed (maps naturally onto an ASIC fabric; for the silicon roadmap).

---

## Where to look for what

When you (future agent) hit a specific question, this table points you at the right chapter:

| Question | Chapter |
|---|---|
| *What does the MPL in `files/definitions/NaturalNumbers.txt` mean?* | [`10_pipeline/01_mpl_definitions.md`](10_pipeline/01_mpl_definitions.md) |
| *Why did conjecturer emit this specific theorem?* | [`10_pipeline/02_conjecturer.md`](10_pipeline/02_conjecturer.md) |
| *Why did the CE filter keep / drop this conjecture?* | [`10_pipeline/03_ce_filter.md`](10_pipeline/03_ce_filter.md) |
| *What is the prover doing when it runs stage 5?* | [`10_pipeline/04_prover.md`](10_pipeline/04_prover.md) |
| *What is `multiplyImplication` and when does it fire?* | [`10_pipeline/04_prover.md`](10_pipeline/04_prover.md) + [`20_core_concepts/04_validity_stack.md`](20_core_concepts/04_validity_stack.md) |
| *Why did the compressor eliminate this theorem?* | [`10_pipeline/05_compressor.md`](10_pipeline/05_compressor.md) |
| *How does a raw chapter become a processed chapter?* | [`10_pipeline/06_process_proof_graph.md`](10_pipeline/06_process_proof_graph.md) |
| *Why did the verifier report a failure on tag X?* | [`10_pipeline/08_verifier.md`](10_pipeline/08_verifier.md) + [`20_core_concepts/08_proof_tags.md`](20_core_concepts/08_proof_tags.md) |
| *What is a Logic Block, operationally?* | [`20_core_concepts/01_logic_blocks.md`](20_core_concepts/01_logic_blocks.md) |
| *How does the mail system ensure consistency?* | [`20_core_concepts/03_mail_system.md`](20_core_concepts/03_mail_system.md) |
| *What are scope names and how are they encoded?* | [`20_core_concepts/04_validity_stack.md`](20_core_concepts/04_validity_stack.md) |
| *What does `!(=[a,b])` propagate under an equivalence class?* | [`20_core_concepts/05_equivalence_classes.md`](20_core_concepts/05_equivalence_classes.md) |
| *What happens on a case split via OR?* | [`20_core_concepts/07_or_branching.md`](20_core_concepts/07_or_branching.md) |
| *What does the `variable copy` tag mean?* | [`20_core_concepts/08_proof_tags.md`](20_core_concepts/08_proof_tags.md) |
| *Can I make this change safely?* | [`30_invariants.md`](30_invariants.md) — check for a relevant I-row before touching code. |
| *Why is the code written this way and not another?* | [`40_decisions.md`](40_decisions.md) |
| *What keeps biting us?* | [`50_gotchas.md`](50_gotchas.md) |

---

## What this overview does not cover

- The full MPL grammar (see [`10_pipeline/01_mpl_definitions.md`](10_pipeline/01_mpl_definitions.md)).
- The per-tag verifier algorithms (see [`10_pipeline/08_verifier.md`](10_pipeline/08_verifier.md) + [`20_core_concepts/08_proof_tags.md`](20_core_concepts/08_proof_tags.md)).
- The incubator's role in generating future CE tables (see [`10_pipeline/09_incubator.md`](10_pipeline/09_incubator.md)).
- The RT campaign, the FTA ladder, or the ASIC roadmap — all tracked in project entries.
- Commercial licensing terms — see [https://generative-logic.com/license](https://generative-logic.com/license).

---

## Weaknesses

### Known & tracked

- **Induction-typing soundness — shipped.** The prover historically scheduled induction on bound variables without first verifying membership in `N`. For a bound variable that appears only in negations, existence heads, or bare equalities, this was unsound. **Fix shipped:** the prover now proves `(in[digitArg, N])` as an auxiliary triad before promoting an induction theorem; chapters of the form `<N>_induction_typing.txt` carry the typing derivation; the verifier's `induction typing` checker walks every `method = induction` row and verifies its accompanying typing chapter. See [I-18](30_invariants.md#i-18) and the historical record in [`docs/induction_typing_plan.md`](induction_typing_plan.md).
- **`files/GL_binaries/` is regenerated every run.** Tracked in git (so a fresh checkout can run the verifier without first running the prover), but each batch's per-batch JSON file is rewritten end-to-end during that batch. Consequences: (a) examples in this document that quote GL-binary JSON fragments must be regenerated when definitions change; (b) the on-disk content drifts from `HEAD` after every prover run, so a `git status` after running often shows these files as modified.

### Suspected fragility

- **No tested single-tag regression.** The verifier emits a 28-ish-row tally, but there is no CI that fails the build when any row goes non-zero. A silent regression in a single checker can persist across multiple commits before the maintainer notices.
- **Incubator ⟷ main cross-contamination.** A recent Gauss incubator config change affected Peano CE behaviour — suggesting the two pipelines share more state than intended (probably through the simple-facts fact-table loader).. Unresolved.

### Not exercised by tests

- **Multi-batch runs past two batches.** The `run_modes.full_run` loop is designed for arbitrarily many batches, but in practice only Incubator → Peano → Gauss is routinely exercised. A 4-batch run would be a first-time integration.
- **Non-`main` namespace handling in the verifier.** The verifier has 27 hardcoded `line.namespace == "main"` checks. Any future proof graph that legitimately uses non-`main` namespaces outside the already-enumerated cases would silently fail all those checkers. (Related: [`10_pipeline/08_verifier.md`](10_pipeline/08_verifier.md) weaknesses subsection.)

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
