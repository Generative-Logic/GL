# Generative Logic (GL)

Deterministic computer architecture for automated mathematical reasoning. GL starts from user-supplied axiomatic definitions written in MPL (Mathematical Programming Language) and systematically explores their deductive neighborhood. No human guidance, no goal-setting — the system discovers and proves theorems autonomously.

**Paper:** [arxiv.org/abs/2508.00017v3](https://arxiv.org/abs/2508.00017v3)

**Website:** [generative-logic.com](https://generative-logic.com)

**Author:** Nikolai Sergeev, Generative Logic UG (Germany)

---

## Current Results

### Main Pipeline (Peano + Gauss)

- **60 theorems** in processed proof graph across 76 chapter files
  - 22 direct proofs, 15 induction proofs, 21 mirrored statements, 2 reformulated statements
- **Batch 1 (Peano):** 32 theorems (17 essential after compression) — commutativity/associativity of addition and multiplication, distributivity
- **Batch 2 (Gauss):** 10 theorems including Gauss's summation formula (division-free variant `n(n+1) = 2·Σi`, autonomously discovered)
- Full run (main + incubator): ~1 hour on commodity hardware (Dell G16 7630, 32 cores, 32 GB RAM)

### Incubator (Peano)

- **1170 theorems** across 1171 chapter files (1069 direct proofs + 101 back-reformulated)
- 926 contradiction proofs (negation of false arithmetic statements) + 143 positive arithmetic facts
- First instance of GL performing **algorithmic multi-stage calculation** from axioms — including **fold (Σ) facts** derived through iterated for-loop-style evaluation
- AnchorIncubator with 14 arguments covering elements {0, 1, 2, 3, 4, 5, 6, 7, 8}

### Proof Graph Viewer

Interactive HTML proof graphs with:
- Clickable proof tags linking to reasoning rule reference
- Readable mathematical notation (subscript variables, sequence notation, scaffolding keywords)
- Hover-highlight dependency chains with magenta-to-amber gradient glow
- Collapsible subproofs (collapsed by default)
- Expression diff highlighting for equality steps
- Step counter, chapter statistics, search bar, "Used by" backlinks
- Right-click integration goal explanations

---

## Architecture

- **Logic Blocks (LBs):** Independent nodes in a distributed grid, each with local memory. Communicate between cycles, not during them.
- **Hash-based inference engine:** Inference is a memory-access problem — formulate hash requests from known expressions, receive new valid expressions on lookup hit.
- **Proof graphs:** Every emitted fact carries full provenance. Proofs export to navigable HTML with hyperlinked justification chains.
- **External verifier:** Independent proof checker operating on processed proof graphs — validates every inference step without importing prover code.

## Core Workflow

1. Definitions compiled into LB grid
2. Conjecture generation (combinatorial enumeration on regularized theorem structures)
3. Normalization and type filtering
4. Counterexample (CE) filtering against small arithmetic tables (peek-and-prune)
5. Prover phase (batched LB execution with warm-up + main iterations)
6. Post-proof compression (greedy elimination of redundant theorems)
7. Process proof graph (variable renaming, pruning)
8. HTML proof graph export
9. External verifier

---

## New Features (unpublished)

The following features have been developed after the v3 paper and are not yet covered in the publication. A paper update is planned after the upcoming runtime optimization campaign.

### Compressor

Post-proof redundancy elimination. After the prover generates theorems, many are logically redundant — derivable from others. The compressor finds a minimal essential subset.

**How it works:**
1. **Per-theorem proof graphs.** For each of N proved theorems, an independent Logic Block is created with all theorems loaded as inference rules. The target theorem's premises serve as fuel, its head as the proof goal. Hash bursts discover derivation paths, producing a lightweight dependency graph per theorem.
2. **Greedy elimination.** Theorems are sorted by usage (least-used first). Each candidate is tentatively removed: if all surviving theorems remain derivable without it, the candidate is dead. If any surviving theorem loses its derivation path, the candidate is essential.

The Peano batch compresses from 32 proved theorems down to 17 essential ones. Compression runs inside the C++ prover (`compressor.cpp`).

### Verifier

Independent external proof checker. Operates on the processed proof graph — the same data that generates the customer-facing HTML. Never imports from the prover codebase.

**Design principles:**
- **Chapter-local** — each chapter is verified in isolation, no cross-chapter jumps
- **Own copy of every algorithm** — disintegration, implication reconstruction, normalization all reimplemented independently
- **23 proof tag types checked** — from basic inference (implication, expansion, disintegration) to structural transforms (mirroring, reformulation, contradiction, anchor handling)
- **Trace-back verification** — for anchor handling and contradiction tags, recursively traces expressions back through the proof chain to verify they reach the expected origin (anchor line or task formulation)

Output: one line per tag type showing `success N, failure 0`. Both main pipeline (1753 checks) and incubator (32593 checks) must be airtight — zero failures.

Source: `verifier.py`

### Incubator

Heuristic mode for auto-generating ground-level arithmetic fact tables. The incubator proves concrete numeric statements directly from Peano axioms — statements like `2 + 3 = 5` or `NOT(3 * 2 = 7)`.

**Why it matters:** The main pipeline proves universal theorems (`∀n: ...`). But filtering conjectures requires concrete arithmetic facts (counterexample filtering). The incubator bootstraps these facts from scratch, closing the loop: axioms → ground facts → better conjecture filtering → more theorems.

**How it works:**
- Separate pipeline with its own config (`ConfigIncubatorPeano.json`) and theorem storage — never touches the main proof graph
- AnchorIncubator provides 14 arguments encoding elements {0, 1, 2, 3, 4, 5, 6, 7, 8} with successor, addition, and multiplication operations
- Positive proofs: derive true arithmetic facts (e.g., `s(0) = 1`, `2 + 3 = 5`)
- Contradiction proofs: derive both an expression and its negation, proving false statements impossible (e.g., `NOT(2 + 1 = 5)`)
- Back-reformulated theorems: operator-equality theorems converted to direct operator form for readability

**Current results:** 1170 theorems — 143 positive facts, 926 contradictions, 101 back-reformulated. Missing negatives require intermediate values exceeding the model range (e.g., proving `NOT(4 + 3 = 9)` needs computing `4 + 3 = 7`, but 7 is outside the {0..5} range of earlier anchor levels). Fix: expand the anchor with more elements.

Source: `run_modes.py` (full_run, RUN_INCUBATOR flag), `simple_facts_incubator.py`, `create_expressions.py` (single_expr_anchor_connection)

---

## Quick Start

### Prerequisites

- Python 3.9+ with the `regex` package (`pip install regex`)
- Windows: bundled native executable `GL_Quick_VS/GL_Quick/gl_quick.exe`
- Non-Windows: C++17 toolchain to rebuild the native component (see below)

### Run

From the repository root:

```bash
python main.py
```

**What happens:**
1. Python generates conjectures (parallelized)
2. Python calls the native prover executable (`GL_Quick_VS/GL_Quick/gl_quick.exe`)
3. Python processes the proof graph (variable renaming, pruning)
4. Python renders HTML proof graph pages
5. External verifier checks all proof steps

**Output:**
- Processed proof graph: `files/processed_proof_graph/`
- HTML proof viewer: `files/full_proof_graph/index.html`
- Incubator output: `files/incubator/full_proof_graph/index.html`

### Rebuilding the Native Executable (if needed)

**Windows (Visual Studio 2022)**

Open `GL_Quick_VS/GL_Quick.sln` in Visual Studio. Build Release x64. The binary will be at `GL_Quick_VS/GL_Quick/gl_quick.exe`.

**Optional: mimalloc (faster memory allocator)**

The bundled executable ships with mimalloc linked. If you rebuild from source, the build works without it (standard allocator). To enable mimalloc for better performance:

1. Install via vcpkg: `vcpkg install mimalloc:x64-windows`
2. In the vcxproj, add `USE_MIMALLOC` to Preprocessor Definitions
3. Add vcpkg include/lib paths to Additional Include/Library Directories
4. Add `mimalloc.dll.lib` to Additional Dependencies

**Linux / macOS (experimental)**

```bash
cd GL_Quick_VS/GL_Quick
c++ -std=c++17 -O3 src/*.cpp -o gl_quick
```

If you place the binary elsewhere, update the path in `run_modes.py`.

---

## File Structure

### Entry Point
- `main.py` → calls `run_modes.full_run()`

### Python Pipeline
- `create_expressions.py` — conjecture generation, CE filtering, variable renaming
- `configuration_reader.py` — reads JSON config files
- `run_modes.py` — orchestration: conjecture → prover → process → HTML → verifier
- `process_proof_graphs.py` — transforms raw proof graph to processed (variable renaming, pruning)
- `generate_full_proof_graph.py` — renders processed proof graph to navigable HTML
- `visu_helpers.py` — HTML rendering helpers
- `verifier.py` — external proof checker, independent of prover code
- `simple_facts_incubator.py` — arithmetic tables for incubator CE filtering
- `incubator_to_simple_facts.py` — converts incubator results to CE filter tables
- `analyze_incubator.py` — incubator result analysis

### C++ Prover
- `GL_Quick_VS/GL_Quick/src/` — native prover source
- `GL_Quick_VS/GL_Quick/gl_quick.exe` — prebuilt Windows x64 binary

### Data Files
- `files/config/` — batch configurations (ConfigPeano, ConfigGauss, ConfigIncubatorPeano, ConfigVisu)
- `files/definitions/` — MPL axiomatic definition files
- `files/GL_binaries/` — compiled definition structures
- `files/raw_proof_graph/` — prover output (before variable renaming)
- `files/processed_proof_graph/` — renamed/pruned proof graph (verifier input)
- `files/full_proof_graph/` — generated HTML proof viewer
- `files/simple_facts/` — arithmetic tables for CE filtering
- `files/theorems/` — theorem storage
- `files/incubator/` — incubator pipeline output (separate from main)

---

## Troubleshooting

**`FileNotFoundError: GL_Quick_VS/GL_Quick/gl_quick.exe`**
Rebuild the native executable or ensure the file exists at that path.

**No HTML output**
Check `files/raw_proof_graph/` was generated. Running `python main.py` regenerates everything.

**Slow run**
Ensure you're using a Release build of the native binary.

---

## Licensing

Generative Logic is dual-licensed under the AGPLv3 and a Commercial License.

**1. AGPLv3 License (Open-Source)**

For open-source projects, academic research, and personal use. This license requires that any derivative works must also be open-source under the same terms. See [LICENSE](LICENSE).

**2. Commercial License**

For proprietary, closed-source commercial applications. Includes enterprise-grade features such as limited warranty and IP indemnity.

For details: **https://generative-logic.com/license**

**Contributing**

All contributions require a signed Contributor License Agreement (CLA).
See `legal/CONTRIBUTOR_LICENSE_AGREEMENT.md` for details.

---

## 3rd Party Notices

**regex** — © Matthew Barnett — Apache-2.0 and CNRI-Python

**nlohmann/json** — © Niels Lohmann — MIT — https://github.com/nlohmann/json

**mimalloc** — © Microsoft — MIT — https://github.com/microsoft/mimalloc
