# Generative Logic

Generative Logic (GL) is a deterministic architecture for automated mathematical reasoning. From axiomatic definitions written in MPL (Mathematical Programming Language) it explores their consequences — it generates conjectures, discards those a counterexample refutes and proves what it can by hash-based inference over a grid of logic blocks — and it also proves lists of goals a user supplies, without hints for their proofs. Every proof GL publishes is checked by an independent verifier, and those of the main proof graph (Peano, Gauss and FTA) also by Lean 4's kernel. Release v0.13.0 proves the Fundamental Theorem of Arithmetic this way, from the Peano axioms and GL's definitions, given the theorem's two halves and the lemmas leading to them as goals.

- Paper: <https://arxiv.org/abs/2508.00017>
- Site:  <https://generative-logic.com>
- Author: Dr. Nikolai Sergeev — Generative Logic UG (haftungsbeschränkt)

## For AI agents working with GL

GL ships an agent-oriented Software Design Document at [`docs/agentic_swdd/SwDD.md`](docs/agentic_swdd/SwDD.md). It is a dense, cross-linked reference written for an AI assistant (or close-reader human) trying to make a code change to the project. It maps the ten pipeline stages, names every load-bearing invariant with stable cross-reference anchors, lists known weaknesses and gotchas, and walks one theorem end-to-end through the full pipeline.

The companion [`docs/fta_ladder/`](docs/fta_ladder/) folder records the path to the Fundamental Theorem of Arithmetic: the early rungs, each in its own subfolder with the math statement and the analysis trace of its proof, the FTA shortlist ([`fta_shortlist.md`](docs/fta_ladder/fta_shortlist.md)) and the conjecture list the FTA proof rests on ([`fta_conjectures.md`](docs/fta_ladder/fta_conjectures.md)).

GL's natural user is a researcher or engineer working alongside an AI coding assistant (Claude Code, Cursor, Codex, Copilot, etc.). The SwDD is written for that workflow: paste an error message and the relevant chapter section into the assistant. For human-only readers, the [paper](https://arxiv.org/abs/2508.00017) and this README are the lighter entry points.

GL also ships a hardware-oriented design document: the **MPU 0.2 (GPU Edition) booklet** at [`docs/MPU/index.html`](docs/MPU/index.html) maps the live prover onto silicon — the logic block as a core, the static memory hierarchy as on-die SRAM with on-demand SSD paging, the cross-block mail system as an on-chip network, and logic-block split as the parallelism lever — together with an order-of-magnitude estimate of what building a Mathematical Processing Unit (MPU) would cost. It is written for a hardware / computer-architecture audience and is maintained as a tracked design document alongside the SwDD.

GL's proof artifacts also have an **independent Lean 4 checking path**. When a Lean 4 toolchain is installed, every full run (`python main.py`) exports the proofs of its main proof graph (Peano, Gauss and FTA) as one connected Lean project in `files/full_proof_graph/lean_export/` and kernel-checks it with `lake build`: typed, hashed certificates replayed over ordinary predicates, with no `sorry`, `admit`, added axiom, opaque proof or unsafe declaration, and every public theorem instantiated on Lean's natural numbers. Every theorem page of the main proof graph links its Lean twin; the incubator graph is checked by the verifier only. Without Lean the run prints one notice and produces no Lean export; the development shortcut (`--shortcut`) produces none either. Details and the manual commands: [`proof_export/README.md`](proof_export/README.md).

## Three pillars

GL's reasoning surface decomposes into three pillars, each contributing a distinct class of inference, on top of shared infrastructure.

1. **Algebraic manipulations** — *Peano*, shipped 2025-09.
   Commutativity, associativity, distributivity of `+` and `*` on a recursively-defined natural-number structure. The Peano batch conjectures its own theorems and is given one lemma to prove as well (successor cancellation).

2. **Logical transformations** — *Gauss*, shipped 2026-02.
   The summation formula `n · (n+1) = 2 · Σ i` in its division-free form, which the Gauss batch conjectures and proves itself, with accompanying anchor-bound consequences.

3. **Case differentiation — branching** — *0.1 shipped 2026-05, 0.2 shipped 2026-07*.
   Sub-proofs split on disjunctive hypotheses, run independently in scoped sub-contexts, and recombine. **Initial version 0.1 closed the forward direction of FTA-ladder rung 1, `{0, 1} = [0, 1]`.**

The Fundamental Theorem of Arithmetic is proved (v0.13.0): every natural number n ≥ 2 has a factorization into primes, written as a non-increasing sequence, and two such factorizations of the same number are equal. Unlike the other batches, the FTA batch generates no conjectures of its own: it is given its goals — the theorem's two halves and the lemmas leading to them, 103 statements — but no hints for their proofs, and GL finds the order and the proofs itself. The machinery this needed was built step by step along an FTA ladder of intermediate theorems and then the FTA shortlist — see [`docs/fta_ladder/`](docs/fta_ladder/) for the index. Pillar internals are documented in [`docs/agentic_swdd/SwDD.md`](docs/agentic_swdd/SwDD.md): pipeline stages under `docs/agentic_swdd/10_pipeline/`, cross-cutting concepts (logic blocks, hash engine, mail system, validity stack, equivalence classes, anchors, OR-branching, proof tags, static memory) under `docs/agentic_swdd/20_core_concepts/`.

## Branching apparatus

Three coupled mechanisms underlie case differentiation. The names below match the runtime markers (`ordis`, `orint`) and the proof-graph tag (`or theorem`) emitted into validity stacks and chapter files.

### `or<N>` theorems — proof of a disjunctive goal (OR-introduction)

A theorem whose head is an `or<N>` expression — e.g. the Peano *branching milestone* `or0[7, 2, 1, 3]`, reading *∀n ∈ N. n = 0 ∨ ∃ predecessor* — is the structural proof object of a disjunction. GL constructs `or<N>` theorems from a proved implication with a negated premise:

- an **existence form** `¬a → b` — classically already *a ∨ b*, so this one proved direction licenses the OR (a chain of several negated premises folds into one n-ary OR),
- its **companion**, the contrapositive `¬b → a`, recorded as the second parent reference; it need not be proved separately.

The OR theorem is built during the run, in the iteration that proves its existence form, and broadcast to the grid; it enters the global theorem list with the `or theorem` method tag, its chapter is rendered in the proof graph, and its base form is the De Morgan expression `!(&!a!b)`.

### OR-disintegration (`ordis`) — case-split on a disjunctive hypothesis (OR-elimination)

Given an OR `d₁ ∨ … ∨ dₙ` in the active proof context, the prover forms one case cohort of its *N* (flattened) disjuncts and releases their branch scopes **one at a time**: the most structured disjunct first, equalities last, and the next branch only when the live one has reached its goal or been refuted. Each branch scope is named per the `_boundary_ordis_(orSignature)_((disjunct))` convention — a stack-rooted validity name that records the OR's compiled signature and which disjunct the branch is conditioned on. The branch is seeded with its disjunct asserted (the negations of the other disjuncts are not seeded; the original OR stays available to exclude cases). Each branch then runs the standard prover machinery in its own scope; equalities introduced locally (e.g. `(=[a, p])` in a branch where `p = a`) propagate through the branch's sub-scopes via the bidirectional equivalence-class rewrite rule. A branch whose disjunct is refuted is retired, and the cohort counts it through its refutation.

When every remaining branch derives the same conclusion *C*, **convergence** fires and *C* is promoted to the OR-branches' shared parent scope. The prover books each branch's deposits per cohort; on convergence the parent-scope statement and its convergence history travel through the logic block's internal mail and are registered by the next drain. The *N* per-branch copies of the converged expression are collapsed into the single parent-scope copy. A branch that has reached every goal it serves is frozen: its runtime state is wiped while its contributions and proof history stay.

The disintegration mechanism is gated to prevent runaway fan-out. A case split opens only in a logic block with proof work (goals, or an active contradiction block), only for an OR none of whose disjuncts already holds there, and only through one of two routes: the OR is a *product of disintegration* (it came from disintegrating a compound, e.g. the body of an enumerated-set definition), a premise being integrated or a premise of a conjecture; or another rule demands one of its cases as its missing premise. Any other OR is parked and opens when such a demand arrives.

### OR-integration (`orint`) — recombining into a wrapping OR

OR-integration is the dual of disintegration, used when the proof has split on an OR-introduction goal. Given a `toBeProved` of OR shape, the prover mints `_boundary_orint_(orSignature)_((head))` branch scopes per disjunct, each carrying that disjunct's head as the local sub-goal. When one branch proves its head, the wrapping OR is emitted at the OR-branches' shared parent scope (recovered by peeling the trailing `_boundary_<payload>` segment off the proving branch's validity name) and the remaining sibling branches are wiped wholesale. The wiped branches' facts were conditional on the sibling disjuncts being true; once the OR has been proven via one branch, the other branches are unreachable case structures whose persistence would only contaminate later reasoning.

For implementation depth — exact validity-stack semantics, the mutual-exclusion rules an OR supplies on demand with full provenance, the convergence pipeline, and known weaknesses — see [`docs/agentic_swdd/20_core_concepts/07_or_branching.md`](docs/agentic_swdd/20_core_concepts/07_or_branching.md) and the firing-site walkthroughs under [`docs/agentic_swdd/10_pipeline/04_prover.md`](docs/agentic_swdd/10_pipeline/04_prover.md).

## Status snapshot (2026-10)

| Pillar | Status |
|---|---|
| Algebraic manipulations (Peano) | shipped 2025-09 |
| Logical transformations (Gauss) | shipped 2026-02 |
| Case differentiation — branching 0.1 | shipped 2026-05 (FTA-ladder rung 1, `EnumerationSet2 ⟹ interval`) |
| Runtime & memory — MPU 0.1 (static prover memory) | shipped 2026-07 (v0.9.0) |
| Case differentiation — branching 0.2 | shipped 2026-07 (v0.10.0, FTA-ladder rungs 2 + 2.1: nested case analysis proves `{0,1,2} = [0,2]` and `{0,1,2} ≠ [0,1]`) |
| Independent proof checking — Lean 4 export | shipped 2026-09 (v0.11.0): one connected project per full run (Peano, Gauss, FTA, natural-number instances), kernel-checked inside the run |
| Runtime — MPU 0.2 (GPU Edition): phase 2 of the hashburst on GPU threads | shipped 2026-09 (v0.11.0; `--GPU`, the same theorems as the processor route) |
| FTA shortlist — order, divisibility, division with remainder, primes, finite sequences and products, Euclid's lemma, ordered sequences, least divisors | every supplied lemma proved by 2026-10 (v0.13.0) |
| **Fundamental Theorem of Arithmetic** — existence and uniqueness of the prime factorization | **shipped 2026-10 (v0.13.0)**: one FTA batch from the Peano axioms and GL's definitions, given its 103 goals without proof hints; verifier and Lean kernel check inside the run |

The 160 conjectures the FTA proof rests on — the 103 the FTA batch is given and 57 Peano and Gauss rows, which GL conjectures itself except the one lemma supplied to the Peano batch — are listed in proof order in [`docs/fta_ladder/fta_conjectures.md`](docs/fta_ladder/fta_conjectures.md); the mathematical plan behind them is the FTA shortlist, [`docs/fta_ladder/fta_shortlist.md`](docs/fta_ladder/fta_shortlist.md), and the earlier rung-by-rung index is [`docs/fta_ladder/README.md`](docs/fta_ladder/README.md). The next planned phase is FTA discovery: GL finding more of the path to the theorem by itself, from a shorter supplied list.

Runtime and memory — the **MPU 0.1** campaign — is done (v0.9.0): the per-logic-block prover state (logic-block pool, hash memory, mail buffers, name map) lives in fixed static memory pools reserved at program start, and cold logic blocks are paged to the SSD on demand. This removed the per-proof heap growth that made FTA-scale proofs infeasible at prior consumption, unblocking the FTA push above. MPU 0.2 (GPU Edition) follows on the runtime axis: phase 2 of the hashburst — the engine itself and its main runtime consumer — runs on GPU threads with `python main.py --GPU`. The gain depends on the batch: where many logic blocks are active, as in the incubator batches, the 32 processor threads keep pace with the GPU; where few are, as in the FTA batch, the GPU's phase 2 takes 33.5 s against 2,307 s on the processor (under WSL2). A full run is about 9.5 times faster on the reference laptop (see *Reference timings*). Both routes prove the same theorems, each through valid proof steps of its own choosing. See the [MPU 0.2 booklet](docs/MPU/index.html) and `RELEASE_NOTES.md`.

Verifier coverage: one entry per proof-graph tag in `verifier.py`'s `TAG_CHECKERS` dispatch table. On a clean release, every category reports `failure 0`. Tag semantics: [`docs/agentic_swdd/20_core_concepts/08_proof_tags.md`](docs/agentic_swdd/20_core_concepts/08_proof_tags.md). Verifier algorithm: [`docs/agentic_swdd/10_pipeline/08_verifier.md`](docs/agentic_swdd/10_pipeline/08_verifier.md).

---

# Run Mode & How to Run

## Run mode

`python main.py` orchestrates the full ten-stage pipeline end-to-end (MPL definitions → conjecturer → CE filter → prover → compressor → process-proof-graph → HTML export → verifier → Lean export; plus the incubator counterpart for ground-fact discovery) over every batch: the incubators, Peano, Gauss and FTA. The FTA batch skips the conjecturer: its conjectures are the supplied list in `files/shortcut/theorems/conjectures.txt` (set in `files/config/ConfigFTA.json`). Each stage is documented in its own chapter under [`docs/agentic_swdd/10_pipeline/`](docs/agentic_swdd/10_pipeline/) — read those if you need to understand or modify a specific stage. The Lean export runs inside the full run whenever a Lean toolchain is installed (`lake` on the PATH or under `ELAN_HOME/bin`, by default `~/.elan/bin`); see *Lean 4 (optional)* below. `python main.py --shortcut` is a development mode: it replays only the FTA batch from the input a previous full run captured, and writes its own output tree under `files/shortcut/` without a Lean export.

Reference timings (v0.13.0 release gates, 2026-10, one laptop with a 32-thread Intel i9-13900HX and an RTX 4070 Laptop GPU): with `--GPU` the full pipeline takes about 5½ minutes under Windows and about 4½ minutes under WSL2 Ubuntu; on the processor about 51 minutes under Windows and about 43 minutes under WSL2. On the processor about nine tenths of the time is phase 2 of the FTA batch (2,307 of the run's 2,575 seconds under WSL2), whose steps run with few active logic blocks; the GPU route runs that phase in about 33 seconds, so a full `--GPU` run is about 9.5 times faster. In the incubator batches, where many logic blocks are active, the processor's phase 2 keeps pace with the GPU's. All four routes prove the same theorems, each route reproduces its own proofs exactly from run to run and across Windows and Linux, the verifier reports zero failures, and the Lean kernel check passes.

## What a release contains — source and proofs

A release carries the source, the pre-built binaries, the configuration, the documentation and the **proofs** of the release's full run — the browsable HTML proof graphs, the Lean 4 export, the theorem lists and the run's other outputs under `files/`. The proofs are committed with the source, at the paths a run writes them (main graph: `files/full_proof_graph/index.html`; incubator graph: `files/incubator/full_proof_graph/index.html`), and the same files are attached to the release as `GL_proofs_<version>.zip`, a download of the proofs alone.

- **Check the repository:** `sha256sum -c SHA256SUMS` checks every file of the repository, the proofs and `PROOFS.sha256` included (on Windows in Git Bash). The release turns git's line-ending conversion off (`.gitattributes`), so every clone and source download carries exactly the hashed bytes.
- **Check the download:** `GL_proofs_<version>.zip.sha256` beside it and the repository's `PROOFS.sha256` hold the archive's SHA-256 (`sha256sum -c GL_proofs_<version>.zip.sha256`; in Windows PowerShell `Get-FileHash GL_proofs_<version>.zip`), so the tagged repository pins exactly one proof archive. Unzipped (`unzip GL_proofs_<version>.zip`; on Windows `tar -xf GL_proofs_<version>.zip`), every file lands where a run writes it and `sha256sum -c PROOFS_SHA256SUMS` checks each one; they are the repository's proof files, byte for byte.
- **Check the proofs themselves:** `python verifier.py` re-checks every proof step of the main graph and `python verifier.py files/incubator/processed_proof_graph` those of the incubator graph; `lake build` in `files/full_proof_graph/lean_export/` re-checks the Lean export in Lean's kernel. Nothing has to be regenerated.
- **Or regenerate them:** `python main.py` proves the same theorems. The published proofs come from a full `--GPU` run on Windows; a processor run may choose different, equally valid proof steps, so compare `files/theorems/theorems.txt` rather than the HTML pages.

## Prerequisites

**Python 3.10 or newer.** The pipeline uses only Python's standard library; no package has to be installed.

**Hardware.** A 64-bit PC; no GPU is needed. A processor run's prover peaks at about 4–5 GB of memory: 8 GB of RAM is the minimum, 16 GB is recommended, and `--GPU` runs need the 16 GB. Free disk space: about 12 GB — a run writes well under 1 GB of proofs and other outputs (about 0.4 GB for v0.13.0), and on Windows the prover reserves 9 GB for its disk-backed memory (`.deload/extent.bin`, kept after the run and replaced by the next one; on Linux the file takes only the space it uses).

**Windows runtime.** The shipped `gl_quick.exe` needs the Microsoft Visual C++ 2015–2022 Redistributable (x64), which most Windows PCs already have; without it the executable cannot start (Windows names `MSVCP140.dll` or `VCRUNTIME140.dll` as missing) and `python main.py` stops at its unit-test gate. Download: <https://aka.ms/vs/17/release/vc_redist.x64.exe>.

**Native executable.** The release bundles pre-built binaries for **Windows** (`gl_quick.exe`, processor and CUDA routes) and **Linux** (`gl_quick`, ELF 64-bit, mimalloc-linked, processor route only) under `GL_Quick_VS/GL_Quick/`; before shipping, each passes the processor-route unit tests. The Linux binary is built on Ubuntu 26.04: it needs glibc 2.38 or newer and mimalloc 3 (`libmimalloc.so.3`, package `libmimalloc3`); on an older distribution, such as Ubuntu 24.04 with mimalloc 2, build it from source. If your platform's binary is absent or doesn't run on your system, or you want the CUDA route on Linux, build from source per the instructions below — a full build takes several minutes on the 32-thread reference laptop (a Windows rebuild about four minutes). macOS binaries are not bundled; macOS users always build from source. A continuous-integration job builds the macOS binary and passes the unit tests; the full-run measurements above come from Windows and Linux (WSL2) only.

If you prefer to rebuild any of the binaries yourself — or you're on a platform where the shipped binary doesn't run — see *Rebuilding the native executable* below.

## Quick start

Run from the repository root (where `main.py` lives) on Windows / macOS / Linux:

```bash
python main.py
```

This runs every batch on the processor; no GPU, CUDA installation or Lean toolchain is needed. On a PC with an NVIDIA GPU (driver with CUDA 13 API), `python main.py --GPU` runs every batch's Phase 2 on the GPU instead; it proves the same theorems (the proof steps may differ from a processor run's), the full run is about 9.5 times faster on the reference laptop, and a missing device or driver is reported as an assertion, never worked around. The shipped Windows executable includes the GPU route; on Linux, build with `make USE_CUDA=1` first (see *GPU (optional)*). The FTA shortcut takes the same flag: `python main.py --shortcut --GPU`.

Use a plain `python` command: GL's checks are `assert` statements, so `main.py` and `verifier.py` refuse to start under `python -O` or with `PYTHONOPTIMIZE` set. A run whose verifier reports a failure ends with exit status 1.

What happens:

- Two unit-test gates run before any pipeline work. The native `gl_quick` binary executes its in-tree C++ unit-test suite (a processor run leaves out the GPU device tests, which only a `--GPU` run needs); `tests/test_harness.py` runs the verifier-side Python suite. The first failing test aborts `main.py` before any proof run starts.
- For each batch, the native conjecturer (`gl_quick --conjecture`, multi-threaded) generates the conjectures; a batch's configuration can add supplied ones, and the FTA batch uses only its supplied list.
- Python calls the native prover executable — `gl_quick.exe` on Windows, `gl_quick` on Linux/macOS. Selection is automatic via `run_modes.py::_find_gl_quick_exe()`; you don't pick.
- Python post-processes the raw proof graph and renders the HTML pages.
- Python verifies every chapter row against its tag's structural checker, no human in the loop.
- With a Lean toolchain installed, Python exports the main proof graph to Lean and runs Lean's kernel check on it.

Outputs:

- Generated HTML lives under `files/full_proof_graph/index.html` (plus `chapter*.html`) for the main batches.
- Incubator-batch HTML lives under `files/incubator/full_proof_graph/`.

## GPU (optional) — what you need to run GL on an NVIDIA GPU

GL runs on the processor by default; nothing below is needed for that. With `--GPU` the prover's Phase 2 (the hashburst engine, its main runtime consumer) runs as CUDA threads; it proves the same theorems as the processor route, possibly through different, equally valid proof steps, and the full run is about 9.5 times faster on the reference laptop — almost all of the gain in the FTA batch. Everything the GPU route needs on a customer PC, and what happens when something is missing:

- **An NVIDIA GPU and its driver.** The shipped Windows executable links the CUDA runtime statically, so no CUDA installation is needed to *run* it: at `--GPU` time it loads `nvcuda.dll` and `nvcudart_hybrid64.dll`, both installed by the NVIDIA display driver. The driver must support the CUDA 13 API (`nvidia-smi` prints the supported "CUDA Version" in its header; 13.x or newer is fine). The kernels ship for `sm_89` (RTX 40 series) and `sm_120` (RTX 50 series).
- **Run:** `python main.py --GPU` (full pipeline) or `python main.py --shortcut --GPU` (development shortcut: the FTA batch only). `--phase2-backend cpu|cuda` is the explicit per-run override for comparisons.
- **When something is missing** — no NVIDIA GPU, a driver too old for the CUDA 13 API, or a Linux binary built without CUDA — a `--GPU` run stops with an assertion naming the missing piece. GL never silently falls back to the processor: a run either uses the route you asked for or tells you why it cannot.
- **Building the Windows executable yourself** needs the CUDA Toolkit 13.3 (the Visual Studio project imports its build customisation) in addition to Visual Studio 2022; without the toolkit the solution does not build. Download: <https://developer.nvidia.com/cuda-downloads> (Windows, `exe (local)`), then reopen Visual Studio.
- **Linux (Ubuntu / WSL2):** the shipped Linux binary and a plain `make` are processor-only; `make USE_CUDA=1 -j$(nproc)` adds the CUDA route and needs the toolkit (`CUDA_HOME` defaults to `/usr/local/cuda`; `NVCC` and `CUDA_ARCH` can be overridden). Install the toolkit from NVIDIA's repository — on WSL2 use the WSL-specific repository, which ships driver-free packages (never install the `cuda` or `cuda-drivers` meta-packages under WSL; the Windows driver serves the GPU):

  ```bash
  # Ubuntu 24.04+ on WSL2 (driver comes from Windows)
  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
  sudo apt-get install -y cuda-toolkit-13-3 libmimalloc-dev build-essential
  echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
  # native Ubuntu: same commands with the ubuntu2404/x86_64 repository path plus your distribution's NVIDIA driver
  ```

  Then `cd GL_Quick_VS/GL_Quick && make USE_CUDA=1 -j$(nproc)` and run `python3 main.py --GPU` from the repository root. `nvidia-smi` inside the shell must list the GPU first.
- **macOS** has no CUDA; the processor route is the only one.

## Lean 4 (optional) — independent kernel checking of the main proofs

When a Lean 4 toolchain is installed, every full run exports the proofs of its main proof graph to Lean and kernel-checks them: `python main.py` writes one connected project to `files/full_proof_graph/lean_export/` (Peano, Gauss and FTA, each later batch calling the earlier batches' proved theorems, plus every public theorem instantiated on Lean's natural numbers), and every theorem page of the main proof graph links its Lean twin (`chapter<N>_lean.html`); the incubator graph has no Lean export. Lake keeps its build folder `lean_export/.lake/` between runs and rebuilds only the modules whose sources or imports changed; `python main.py --fresh-lean` removes it first and rebuilds everything. The development shortcut produces no Lean export. Without a toolchain the run prints one notice and produces no Lean export — nothing else changes. The pinned toolchain version is in `lean_export/lean-toolchain`; `elan` installs it on first use.

| Platform | Install command |
|---|---|
| Windows | `winget install --id LeanProver.Elan` (then reopen the terminal) |
| Linux / macOS / WSL2 | `curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf \| sh` |

GL looks for `lake` on the `PATH` or under `ELAN_HOME/bin` (by default `~/.elan/bin`). A failing kernel check stops the run: the Lean export is a verification gate, never decoration.

## Rebuilding the native executable (if needed)

The native prover is a C++17 project with optional CUDA kernels: Visual Studio 2022 builds it on Windows, the Makefile on Linux and macOS.

**Windows (Visual Studio 2022)**

Prerequisite — mimalloc: the `.vcxproj` links mimalloc via [vcpkg](https://github.com/microsoft/vcpkg). The prebuilt mimalloc package is **not bundled** in the release (it is Windows-only and large); restore it once with `vcpkg install` from the directory that holds `vcpkg.json`. The Visual Studio project always defines `USE_MIMALLOC`; the `USE_MIMALLOC=0` opt-out exists only for the Makefile builds below.

Prerequisite — CUDA Toolkit 13.3: the project imports its build customisation, so the toolkit is needed to *build* on Windows even if you only use the processor route (no GPU is needed to build or to run; see *GPU (optional)* above). The shipped `gl_quick.exe` needs no toolkit.

1. Open `GL_Quick_VS/GL_Quick.sln` in Visual Studio.
2. Build Release x64 (recommended).
3. The binary will be at `GL_Quick_VS/GL_Quick/gl_quick.exe` (this is the path the Python code expects).

**Linux**

Prerequisites: a C++17 compiler (`g++` or `clang++`; the release's Linux binary is built with g++ 15), `make`, and `libmimalloc-dev`. On Debian/Ubuntu:

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

Running `make` again after editing a source file or a header, or with other options (`USE_CUDA`, `USE_MIMALLOC`, `CXX`), rebuilds what the change affects; `make clean` is not needed.

**Why mimalloc is required by default.** GL's prover is a deterministic computational architecture — the same inputs must produce bit-identical proof graphs across platforms, and that parity is tested with the same allocator, mimalloc, on Windows and Linux. The Windows MSVC build (which ships the prebuilt `gl_quick.exe`) hardcodes `USE_MIMALLOC` in its `.vcxproj`; the Makefile defaults to `USE_MIMALLOC=1` so the Linux/macOS build matches it. If you don't care about cross-platform parity (single-host runs only) build with `make USE_MIMALLOC=0 -j$(nproc)` to skip the mimalloc dependency.

**The build itself**

The Makefile invokes one `c++` process per translation unit and links them into a single `gl_quick` binary in `GL_Quick_VS/GL_Quick/`. A plain `make` builds the processor-only binary — the route every run uses by default — and needs no CUDA toolkit.

If you don't want to use `make`, the equivalent single-process invocation is:

```bash
c++ -std=c++17 -O3 -DUSE_MIMALLOC -Isrc src/*.cpp src/compiler/*.cpp src/proof/*.cpp src/inference/*.cpp src/logical_block/*.cpp src/infra/*.cpp src/memory_infra/*.cpp src/gpu/*.cpp src/tests/*.cpp -o gl_quick -rdynamic -lmimalloc
```

The bundled `json.hpp` (nlohmann/json) lives in `src/` and is included as `<json.hpp>`, so the `-Isrc` flag is required either way.

If you place the binary somewhere else or name it differently, update the path in `run_modes.py` (function `_find_gl_quick_exe()`).

## Troubleshooting

- `FileNotFoundError: Native executable not found. Looked for: ...gl_quick.exe, ...gl_quick` — your platform's expected binary is missing. Rebuild per the platform-specific instructions above. The finder checks both names in priority order based on `sys.platform`.
- `error while loading shared libraries: libmimalloc.so.3: cannot open shared object` — the shipped Linux binary needs mimalloc 3. Install it (`sudo apt-get install -y libmimalloc3` where the distribution has it), or build from source with your distribution's `libmimalloc-dev` (see *Rebuilding the native executable*).
- `cannot open include file: 'mimalloc.h'` (compile-time, Linux) — same root cause; the `-dev` package wasn't installed. The Makefile defaults to `USE_MIMALLOC=1`; build with `make USE_MIMALLOC=0` to opt out.
- No HTML output — check `files/raw_proof_graph/*` was generated and that `files/full_proof_graph/` is created. Running `python main.py` regenerates these.
- Slow run — make sure you're using a Release build of the native binary; performance varies by CPU. Reference: about 43–51 minutes for the full pipeline on the processor (about nine tenths of it the FTA batch) and 4½–5½ minutes with `--GPU` on a 32-thread i9-13900HX laptop.

**Stuck?** An AI coding assistant (Claude Code, Cursor, Codex, Copilot, etc.) can often help with build and runtime issues: paste the error message, the command you ran, and your platform. The agent-oriented [`docs/agentic_swdd/SwDD.md`](docs/agentic_swdd/SwDD.md) is written to give such an assistant the context to navigate the codebase.

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

- **nlohmann/json** — © Niels Lohmann — MIT. <https://github.com/nlohmann/json>
- **mimalloc** — © Microsoft Corporation — MIT.
- **Lean 4** (optional, kernel-checks the exported proofs) — © Microsoft Corporation, Lean FRO — Apache-2.0. <https://github.com/leanprover/lean4>
