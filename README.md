# Generative Logic

Deterministic computer architecture for automated mathematical reasoning. From user-supplied axiomatic definitions in MPL (Mathematical Programming Language), GL enumerates conjectures, filters by counterexample, and proves or disproves each one by hash-based inference over a distributed grid of logic blocks. No human guidance, no goal-setting; theorems are discovered and proved autonomously, with a provenance-complete proof graph for every emitted fact.

- Paper: <https://arxiv.org/abs/2508.00017>
- Site:  <https://generative-logic.com>
- Author: Dr. Nikolai Sergeev — Generative Logic UG (haftungsbeschränkt)

## For AI agents working with GL

GL ships an agent-oriented Software Design Document at [`docs/agentic_swdd/SwDD.md`](docs/agentic_swdd/SwDD.md). It is a dense, cross-linked reference written for an AI assistant (or close-reader human) trying to make a code change to the project. It maps the ten pipeline stages, names every load-bearing invariant with stable cross-reference anchors, lists known weaknesses and gotchas, and walks one theorem end-to-end through the full pipeline.

The companion [`docs/fta_ladder/`](docs/fta_ladder/) folder is the rung-by-rung roadmap toward the Fundamental Theorem of Arithmetic — each rung has its own subdir (`docs/fta_ladder/rung<N>/`) carrying the math statement, intermediate proof obligations, and (for in-flight rungs) the live analysis trace.

GL's natural user is a researcher or engineer working alongside an AI coding assistant (Claude Code, Cursor, Codex, Copilot, etc.). The SwDD is sized for that workflow: paste an error message + the relevant chapter section into the assistant and it has enough context to proceed. For human-only readers, the [paper](https://arxiv.org/abs/2508.00017) and this README are the lighter entry points.

GL also ships a hardware-oriented design document: the **MPU 0.1 booklet** at [`docs/MPU/index.html`](docs/MPU/index.html) maps the live prover onto silicon — the logic block as a core, the static memory hierarchy as on-die SRAM with on-demand SSD paging, the cross-block mail system as an on-chip network, and logic-block split as the parallelism lever — together with an order-of-magnitude estimate of what building a Mathematical Processing Unit (MPU) would cost. It is written for a hardware / computer-architecture audience and is maintained as a tracked design document alongside the SwDD.

## Three pillars

GL's reasoning surface decomposes into three pillars, each contributing a distinct class of inference. Each pillar carries roughly 10 000 lines of C++ + Python and roughly eight months of build time, on top of shared infrastructure.

1. **Algebraic manipulations** — *Peano*, shipped 2025-09.
   Commutativity, associativity, distributivity of `+` and `*` on a recursively-defined natural-number structure.

2. **Logical transformations** — *Gauss*, shipped 2026-02.
   The autonomous-discovery summation formula `n · (n+1) = 2 · Σ i` in its division-free form, with accompanying anchor-bound consequences.

3. **Case differentiation — branching** — *in progress, version 0.1, 2026-05*.
   Sub-proofs split on disjunctive hypotheses, run independently in scoped sub-contexts, and recombine. **Initial version 0.1 closed the forward direction of FTA-ladder rung 1, `{0, 1} = [0, 1]`.**

The Fundamental Theorem of Arithmetic is the next major milestone, climbed one rung at a time along an FTA ladder of intermediate theorems — see [`docs/fta_ladder/`](docs/fta_ladder/) for the rung-by-rung index. Pillar internals are documented in [`docs/agentic_swdd/SwDD.md`](docs/agentic_swdd/SwDD.md): pipeline stages under `docs/agentic_swdd/10_pipeline/`, cross-cutting concepts (logic blocks, hash engine, mail system, validity stack, equivalence classes, anchors, OR-branching, proof tags) under `docs/agentic_swdd/20_core_concepts/`.

## Branching apparatus

Three coupled mechanisms underlie case differentiation. The names below match the runtime markers (`ordis`, `orint`) and the proof-graph tag (`or theorem`) emitted into validity stacks and chapter files.

### `or<N>` theorems — proof of a disjunctive goal (OR-introduction)

A theorem whose head is an `or<N>` expression — e.g. the Peano *branching milestone* `or0[7, 2, 1, 3]`, reading *∀n ∈ N. n = 0 ∨ ∃ predecessor* — is the structural proof object of a disjunction. GL constructs `or<N>` theorems by pairing two duals:

- an **existence form** `!(>[bound](¬a)!(b))`  — schematically, *"it is not the case that ¬a holds without b"*, i.e. *a ∨ b* via De Morgan,
- a **companion** `¬a → b`.

When both are independently proved, the OR theorem is emitted into the global theorem list with the `or theorem` method tag and the corresponding chapter is rendered in the proof graph.

### OR-disintegration (`ordis`) — case-split on a disjunctive hypothesis (OR-elimination)

Given an OR `d₁ ∨ … ∨ dₙ` in the active proof context, the prover opens *N* branch scopes. Each branch scope is named per the `_boundary_ordis_(orSignature)_((disjunct))` convention — a stack-rooted validity name that records the OR's compiled signature and which disjunct the branch is conditioned on. The branch is seeded with the disjunct asserted (the *head*) plus the negations of the other disjuncts (*assumptions*). Each branch then runs the standard prover machinery in its own scope; equalities introduced locally (e.g. `(=[a, p])` in a branch where `p = a`) propagate through the branch's sub-scopes via the bidirectional equivalence-class rewrite rule.

When every branch independently derives the same conclusion *C*, **convergence** fires and *C* is promoted to the OR-branches' shared parent scope. The current implementation tracks per-branch deposits in a kernel-level bookkeeping pass; on convergence, the promotion target is enqueued in a per-LB revival inbox (`internalMailIn`) and absorbed at the next hashburst through the full kernel pipeline. The *N* per-branch copies of the converged expression are then collapsed into the single parent-scope copy, and the branches themselves stay alive so further expressions can converge across them.

The disintegration mechanism is gated to prevent runaway fan-out. The gate admits only ORs delivered by an implication that is itself a *product of disintegration* — i.e. at least one premise of the firing implication carries a bound-variable (`u_*`-prefixed) argument, indicating it came from disintegrating a compound (e.g. the body of an enumerated-set definition) rather than from an anchor-bound theorem. Anchor-bound theorems' disjunctions therefore do not spawn case-split branches; only body-element implications do.

### OR-integration (`orint`) — recombining into a wrapping OR

OR-integration is the dual of disintegration, used when the proof has split on an OR-introduction goal. Given a `toBeProved` of OR shape, the prover mints `_boundary_orint_(orSignature)_((head))` branch scopes per disjunct, each carrying that disjunct's head as the local sub-goal. When one branch proves its head, the wrapping OR is emitted at the OR-branches' shared parent scope (recovered by peeling the trailing `_boundary_<payload>` segment off the proving branch's validity name) and the remaining sibling branches are wiped wholesale. The wiped branches' facts were conditional on the sibling disjuncts being true; once the OR has been proven via one branch, the other branches are unreachable case structures whose persistence would only contaminate later reasoning.

For implementation depth — exact validity-stack semantics, the `K` mutual-exclusion sub-implications produced by OR-disintegration with full provenance, the convergence kernel pipeline, and known weaknesses — see [`docs/agentic_swdd/20_core_concepts/07_or_branching.md`](docs/agentic_swdd/20_core_concepts/07_or_branching.md) and the firing-site walkthroughs under [`docs/agentic_swdd/10_pipeline/04_prover.md`](docs/agentic_swdd/10_pipeline/04_prover.md).

## Status snapshot (2026-07)

| Pillar | Status |
|---|---|
| Algebraic manipulations (Peano) | shipped 2025-09 |
| Logical transformations (Gauss) | shipped 2026-02 |
| Case differentiation — branching 0.1 | shipped 2026-05 (FTA-ladder rung 1, `EnumerationSet2 ⟹ interval`) |
| Runtime & memory — MPU 0.1 (static prover memory) | shipped 2026-07 (v0.9.0) |

The next frontier — see [`docs/fta_ladder/README.md`](docs/fta_ladder/README.md) for the rung-by-rung index, and `docs/fta_ladder/rung<N>/current_proof_state.md` for any in-flight rung's live analysis trace:

**Next release: v0.10.0 — incubator theorems that require branching.**

- **Incubator-side ground facts** to filter statements about intervals and sequences — extensions of the rung-1 pattern to larger fixed sets, e.g. `{0,1,2}=[0,2]`, `{0,1,2,3}=[0,3]`, and analogous claims. These are the concrete-instance theorems the CE filter needs in order to kill false conjectures about general intervals and sequences before they reach the prover.
- **Euclid's lemma**.
- **FTA proper**.

Runtime and memory — the **MPU 0.1** campaign — is done: the entire per-logic-block prover state (logic-block pool, hash memory, mail buffers, name map) now lives in a fixed static pool with on-demand SSD paging, shipped in v0.9.0. This removed the per-proof heap growth that made FTA-scale proofs infeasible at prior consumption, unblocking the FTA push above. See the [MPU 0.1 booklet](docs/MPU/index.html) and `RELEASE_NOTES.md`.

Verifier coverage: one entry per proof-graph tag in `verifier.py`'s `TAG_CHECKERS` dispatch table. On a clean release, every category reports `failure 0`. Tag semantics: [`docs/agentic_swdd/20_core_concepts/08_proof_tags.md`](docs/agentic_swdd/20_core_concepts/08_proof_tags.md). Verifier algorithm: [`docs/agentic_swdd/10_pipeline/08_verifier.md`](docs/agentic_swdd/10_pipeline/08_verifier.md).

---

# Run Mode & How to Run

## Run mode

There is only one run mode (previously called "Full Mode"). `python main.py` orchestrates the full ten-stage pipeline end-to-end (MPL definitions → conjecturer → CE filter → prover → compressor → process-proof-graph → HTML export → verifier; plus the incubator counterpart for ground-fact discovery). Each stage is documented in its own chapter under [`docs/agentic_swdd/10_pipeline/`](docs/agentic_swdd/10_pipeline/) — read those if you need to understand or modify a specific stage.

Latest full Windows reference (2026-07-12, Dell G16 7630, MSVC): 1,478.95102 seconds (24 minutes 38.951 seconds), with 1,241/1,241 native unit tests, 352/352 verifier unit tests, and 131,886 proof checks passing with zero failures. The largest main static-pool high-water was 16,105 of 16,384 256-KiB blocks in `IncubatorGauss1`: 4,026.250 MiB of the 4,096-MiB reservation (98.297%). Other pools and process overhead are separate; these pool peaks are not additive. The Linux build needs `libmimalloc-dev` installed (see *Why mimalloc is required by default* in the build section).

## Prerequisites

**Python 3.9+ with the `regex` package** — required for recursive pattern matching in HTML proof graph rendering.

| Platform | Install command |
|---|---|
| Windows | `pip install regex` (Anaconda includes pip) |
| Debian/Ubuntu | `sudo apt-get install -y python3-regex` |
| macOS (Homebrew Python) | `pip3 install regex` |

**Native executable.** When the release host has the toolchain available, the release bundles pre-built binaries for **Windows** (`gl_quick.exe`) and **Linux** (`gl_quick`, ELF 64-bit, mimalloc-linked) under `GL_Quick_VS/GL_Quick/`. If your platform's binary is absent (the publish host lacked that toolchain) or doesn't run on your system, build from source per the instructions below — Windows takes a couple of minutes, Linux and macOS roughly half a minute parallelised. macOS binaries are not bundled; macOS users always build from source.

If you prefer to rebuild any of the binaries yourself — or you're on a platform where the shipped binary doesn't run — see *Rebuilding the native executable* below.

## Quick start

Run from the repository root (where `main.py` lives) on Windows / macOS / Linux:

```bash
python main.py
```

What happens:

- Two unit-test gates run before any pipeline work. The native `gl_quick` binary executes its in-tree C++ unit-test suite; `tests/test_harness.py` runs the verifier-side Python suite. The first failing test aborts `main.py` before any proof run starts.
- Python creates conjectures (parallelized).
- Python calls the native prover executable — `gl_quick.exe` on Windows, `gl_quick` on Linux/macOS. Selection is automatic via `run_modes.py::_find_gl_quick_exe()`; you don't pick.
- Python post-processes the raw proof graph and renders the HTML pages.
- Python verifies every chapter row against its tag's structural checker, no human in the loop.

Outputs:

- Generated HTML lives under `files/full_proof_graph/index.html` (plus `chapter*.html`) for the main batches.
- Incubator-batch HTML lives under `files/incubator/full_proof_graph/`.

## Rebuilding the native executable (if needed)

The native prover is a C++ project compiled with Microsoft Visual Studio on Windows 11.

**Windows (Visual Studio 2022)**

Prerequisite — mimalloc: the `.vcxproj` links mimalloc via [vcpkg](https://github.com/microsoft/vcpkg). The prebuilt mimalloc package is **not bundled** in the release (it is Windows-only and large); restore it once with `vcpkg install` from the directory that holds `vcpkg.json`. If you do not need cross-platform proof-graph parity, build with `USE_MIMALLOC=0` instead.

1. Open `GL_Quick_VS/GL_Quick.sln` in Visual Studio.
2. Build Release x64 (recommended).
3. The binary will be at `GL_Quick_VS/GL_Quick/gl_quick.exe` (this is the path the Python code expects).

**Linux**

Prerequisites: a C++17 compiler (`g++` 9+ or `clang++` 10+), `make`, and `libmimalloc-dev`. On Debian/Ubuntu:

```bash
sudo apt-get install -y build-essential libmimalloc-dev
```

Build (parallel, all CPUs):

```bash
cd GL_Quick_VS/GL_Quick
make -j$(nproc)
```

**macOS**

Prerequisites: Xcode command-line tools (`clang++` is the system compiler) and mimalloc:

```bash
xcode-select --install
brew install mimalloc
```

Build (parallel, all CPUs):

```bash
cd GL_Quick_VS/GL_Quick
make -j$(sysctl -n hw.ncpu)
```

The Makefile uses `c++` which resolves to `clang++` on macOS by default. To force `g++` (e.g. from Homebrew): `make CXX=g++-14 -j$(sysctl -n hw.ncpu)`.

**Why mimalloc is required by default.** GL's prover is a deterministic computational architecture — the same inputs must produce bit-identical proof graphs across platforms. Different memory allocators produce different allocation patterns, which leak into hash-map iteration order, which produces slightly different proof-row orderings. The Windows MSVC build (which ships the prebuilt `gl_quick.exe`) hardcodes `USE_MIMALLOC` in its `.vcxproj`; for the Linux/macOS build to match it, the same allocator must be used. The Makefile defaults to `USE_MIMALLOC=1` to enforce this. If you don't care about cross-platform parity (single-host runs only) build with `make USE_MIMALLOC=0 -j$(nproc)` to skip the mimalloc dependency.

**The build itself**

The Makefile invokes one `c++` process per translation unit (9 units total) and links them into a single `gl_quick` binary in `GL_Quick_VS/GL_Quick/`. Memory peak is ~1 GB during compile. Measured wall-clock on a 32-core Intel i9-13900HX with `make -j32`: 41 seconds (vs. 92 seconds for single-process). Speedup tops out around 2× — `conjecturer.cpp` (257 KB) is the slowest unit and dominates the critical path regardless of `-j` value.

If you don't want to use `make`, the equivalent single-process invocation is:

```bash
c++ -std=c++17 -O3 -DUSE_MIMALLOC -Isrc src/*.cpp -o gl_quick -lmimalloc
```

The bundled `json.hpp` (nlohmann/json) lives in `src/` and is included as `<json.hpp>`, so the `-Isrc` flag is required either way.

If you place the binary somewhere else or name it differently, update the path in `run_modes.py` (function `_find_gl_quick_exe()`).

## Troubleshooting

- `FileNotFoundError: Native executable not found. Looked for: ...gl_quick.exe, ...gl_quick` — your platform's expected binary is missing. Rebuild per the platform-specific instructions above. The finder checks both names in priority order based on `sys.platform`.
- `error while loading shared libraries: libmimalloc.so.3: cannot open shared object` — Linux runtime is missing mimalloc. `sudo apt-get install -y libmimalloc-dev` (Debian/Ubuntu) or your distro's equivalent.
- `cannot open include file: 'mimalloc.h'` (compile-time, Linux) — same root cause; the `-dev` package wasn't installed. The Makefile defaults to `USE_MIMALLOC=1`; build with `make USE_MIMALLOC=0` to opt out.
- No HTML output — check `files/raw_proof_graph/*` was generated and that `files/full_proof_graph/` is created. Running `python main.py` regenerates these.
- Slow run — make sure you're using a Release build of the native binary; performance varies by CPU. Reference timing: full pipeline ~26 minutes on a 32-core i9-13900HX, similar on Linux+mimalloc.

**Stuck?** Most build / runtime issues fall to a single AI-assistant query if you're using one (Claude Code, Cursor, Codex, Copilot, etc.). Paste the error message, the command you ran, and your platform; the assistant has enough to debug from there. The agent-oriented [`docs/agentic_swdd/SwDD.md`](docs/agentic_swdd/SwDD.md) is also designed to give an LLM enough context to navigate the codebase end-to-end.

## Paths recap

- Entry point: `main.py`
- Native executable: `GL_Quick_VS/GL_Quick/gl_quick.exe` (Windows) / `GL_Quick_VS/GL_Quick/gl_quick` (Linux/macOS)
- Native source: `GL_Quick_VS/GL_Quick/src/`
- Native build (Linux/macOS): `GL_Quick_VS/GL_Quick/Makefile`
- Native build (Windows): `GL_Quick_VS/GL_Quick.sln`
- Output HTML (main): `files/full_proof_graph/`
- Output HTML (incubator): `files/incubator/full_proof_graph/`

---

# Licensing 📜

Generative Logic is dual-licensed under the AGPLv3 and a Commercial License.

**1. AGPLv3 License (Open-Source)**

For open-source projects, academic research, and personal use, Generative Logic is licensed under the AGPLv3. The AGPLv3 imposes two requirements you should be aware of before adopting GL in a commercial setting:

- **Copyleft on modifications.** Any derivative work that you distribute in any form must also be released under the AGPLv3, including your full source code.
- **Network distribution clause.** If you offer a hosted service or SaaS that includes any GL code — whether modified or not — you must make the complete corresponding source code (including your modifications) available to every user of that service. The commercial license below removes this obligation.

The full AGPLv3 text is available in the [LICENSE](LICENSE) file.

**2. Commercial License**

If you wish to use Generative Logic in a proprietary, closed-source commercial application, a commercial license is required. This license frees you from the obligations of the AGPLv3 and includes enterprise-grade features such as a limited warranty and IP indemnity.

**Who needs the commercial license?** Get one if any of the following applies:
- You embed GL in a closed-source product, distribution, or appliance.
- You offer GL-powered functionality as a hosted service, API, or SaaS.
- You do not want to publish your forks, modifications, or surrounding integration code.
- You need warranty, indemnity, or formal support for due-diligence, audit, or procurement purposes.

For more details and to request a quote, please visit our commercial licensing page:
**https://generative-logic.com/license**

Or email **info@generative-logic.com** directly.

**Contributing**

All contributions require a signed Contributor License Agreement (CLA).
See `legal/CONTRIBUTOR_LICENSE_AGREEMENT.md` for details.

---

# 3rd party notices

- **regex** — © Matthew Barnett — Apache-2.0 and CNRI-Python.
- **nlohmann/json** — © Niels Lohmann — MIT. <https://github.com/nlohmann/json>
- **mimalloc** — © Microsoft Corporation — MIT.
