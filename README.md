# Generative Logic

Deterministic computer architecture for automated mathematical reasoning. From user-supplied axiomatic definitions in MPL (Mathematical Programming Language), GL enumerates conjectures, filters by counterexample, and proves or disproves each one by hash-based inference over a distributed grid of logic blocks. No human guidance, no goal-setting; theorems are discovered and proved autonomously, with a provenance-complete proof graph for every emitted fact.

- Paper: <https://arxiv.org/abs/2508.00017>
- Site:  <https://generative-logic.com>
- Author: Dr. Nikolai Sergeev — Generative Logic UG (haftungsbeschränkt)

## For AI agents working with GL

GL ships an agent-oriented Software Design Document at [`docs/agentic_swdd/SwDD.md`](docs/agentic_swdd/SwDD.md). It is a dense, cross-linked reference written for an AI assistant (or close-reader human) trying to make a code change to the project. It maps the ten pipeline stages, names every load-bearing invariant with stable cross-reference anchors, lists known weaknesses and gotchas, and walks one theorem end-to-end through the full pipeline.

The companion [`docs/fta_ladder/`](docs/fta_ladder/) folder is the rung-by-rung roadmap toward the Fundamental Theorem of Arithmetic — each rung has its own subdir (`docs/fta_ladder/rung<N>/`) carrying the math statement, intermediate proof obligations, and (for in-flight rungs) the live analysis trace.

GL's natural user is a researcher or engineer working alongside an AI coding assistant (Claude Code, Cursor, Codex, Copilot, etc.). The SwDD is sized for that workflow: paste an error message + the relevant chapter section into the assistant and it has enough context to proceed. For human-only readers, the [paper](https://arxiv.org/abs/2508.00017) and this README are the lighter entry points.

GL also ships a hardware-oriented design document: the **MPU 0.2 (GPU Edition) booklet** at [`docs/MPU/index.html`](docs/MPU/index.html) maps the live prover onto silicon — the logic block as a core, the static memory hierarchy as on-die SRAM with on-demand SSD paging, the cross-block mail system as an on-chip network, and logic-block split as the parallelism lever — together with an order-of-magnitude estimate of what building a Mathematical Processing Unit (MPU) would cost. It is written for a hardware / computer-architecture audience and is maintained as a tracked design document alongside the SwDD.

GL's proof artifacts also have an **independent Lean 4 checking path**. When a Lean 4 toolchain is installed, every full run (`python main.py`) exports its Peano, Gauss and FTA-shortlist proofs as one connected Lean project in `files/full_proof_graph/lean_export/` and kernel-checks it with `lake build`: typed, hashed certificates replayed over ordinary predicates, with no `sorry`, `admit`, added axiom, opaque proof or unsafe declaration, and every public theorem instantiated on Lean's natural numbers. Every HTML proof page links its Lean twin. Without Lean the run prints one notice and produces no Lean export; the development shortcut (`--shortcut`) produces none either. Details and the manual commands: [`proof_export/README.md`](proof_export/README.md).

## Three pillars

GL's reasoning surface decomposes into three pillars, each contributing a distinct class of inference. Each pillar carries roughly 10 000 lines of C++ + Python and roughly eight months of build time, on top of shared infrastructure.

1. **Algebraic manipulations** — *Peano*, shipped 2025-09.
   Commutativity, associativity, distributivity of `+` and `*` on a recursively-defined natural-number structure.

2. **Logical transformations** — *Gauss*, shipped 2026-02.
   The autonomous-discovery summation formula `n · (n+1) = 2 · Σ i` in its division-free form, with accompanying anchor-bound consequences.

3. **Case differentiation — branching** — *0.1 shipped 2026-05, 0.2 shipped 2026-07*.
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

## Status snapshot (2026-10)

| Pillar | Status |
|---|---|
| Algebraic manipulations (Peano) | shipped 2025-09 |
| Logical transformations (Gauss) | shipped 2026-02 |
| Case differentiation — branching 0.1 | shipped 2026-05 (FTA-ladder rung 1, `EnumerationSet2 ⟹ interval`) |
| Runtime & memory — MPU 0.1 (static prover memory) | shipped 2026-07 (v0.9.0) |
| Case differentiation — branching 0.2 | shipped 2026-07 (v0.10.0, FTA-ladder rungs 2 + 2.1: nested case analysis proves `{0,1,2} = [0,2]` and `{0,1,2} ≠ [0,1]`) |
| Independent proof checking — Lean 4 export | shipped 2026-08; since v0.11.0 one connected project per full run (Peano, Gauss, FTA shortlist, natural-number instances), kernel-checked inside the run |
| Runtime — MPU 0.2 (GPU Edition): phase 2 of the hashburst on GPU threads | shipped 2026-09 (v0.11.0; `--GPU`, about ten times faster on phase 2, the same theorems as the processor route) |
| FTA shortlist — order, divisibility, division with remainder, primes, first finite products | in progress (v0.12.0: division with remainder complete, the first prime facts, singleton and appended sequences, the singleton product and the one-entry product step) |

The next frontier is the rest of the FTA shortlist — [`docs/fta_ladder/fta_shortlist.md`](docs/fta_ladder/fta_shortlist.md) lists every lemma and the proof order; the earlier rung-by-rung index is [`docs/fta_ladder/README.md`](docs/fta_ladder/README.md):

- **The remaining shortlist** — finite-sequence products and membership (part F), Euclid's lemma via symmetric Bézout (part G), order on indexed positions (part H), and the least-divisor construction of the canonical factorization (part I).
- **FTA proper.**

Runtime and memory — the **MPU 0.1** campaign — is done: the entire per-logic-block prover state (logic-block pool, hash memory, mail buffers, name map) now lives in a fixed static pool with on-demand SSD paging, shipped in v0.9.0. This removed the per-proof heap growth that made FTA-scale proofs infeasible at prior consumption, unblocking the FTA push above. MPU 0.2 (GPU Edition) follows on the runtime axis: phase 2 of the hashburst — the engine itself and its main runtime consumer — now runs on GPU threads (`python main.py --GPU`), about ten times faster than the 32 processor threads so far; both routes prove the same theorems, each through valid proof steps of its own choosing. See the [MPU 0.2 booklet](docs/MPU/index.html) and `RELEASE_NOTES.md`.

Verifier coverage: one entry per proof-graph tag in `verifier.py`'s `TAG_CHECKERS` dispatch table. On a clean release, every category reports `failure 0`. Tag semantics: [`docs/agentic_swdd/20_core_concepts/08_proof_tags.md`](docs/agentic_swdd/20_core_concepts/08_proof_tags.md). Verifier algorithm: [`docs/agentic_swdd/10_pipeline/08_verifier.md`](docs/agentic_swdd/10_pipeline/08_verifier.md).

---

# Run Mode & How to Run

## Run mode

`python main.py` orchestrates the full ten-stage pipeline end-to-end (MPL definitions → conjecturer → CE filter → prover → compressor → process-proof-graph → HTML export → verifier; plus the incubator counterpart for ground-fact discovery) over every batch: the incubators, Peano, Gauss and the FTA shortlist. Each stage is documented in its own chapter under [`docs/agentic_swdd/10_pipeline/`](docs/agentic_swdd/10_pipeline/) — read those if you need to understand or modify a specific stage. The Lean export runs inside the full run whenever a Lean toolchain is installed (`lake` on the PATH or under `~/.elan/bin`); see *Lean 4 (optional)* below. `python main.py --shortcut` is a development mode: it replays only the FTA batch from the input a previous full run captured, and writes its own output tree under `files/shortcut/` without a Lean export.

Reference timings (v0.12.0 release gates, 2026-10, one laptop with a 32-thread Intel i9-13900HX and an RTX 4070 Laptop GPU): the full pipeline takes about 25 minutes on the processor under Windows and about 18 minutes under WSL2 Ubuntu; with `--GPU` about 7 minutes under Windows and about 5 minutes under WSL2. On the processor most of the time is the FTA batch. All four routes prove the same theorems, the verifier reports zero failures, and the Lean kernel check passes.

## What a release contains — source in the repository, proofs as a download

A release has two parts. The **repository** holds the source, the pre-built binaries, the configuration and the documentation. The **proofs** of the release — the browsable HTML proof graphs, the Lean 4 export, the theorem lists and the other outputs of the release's full run — are attached to the GitHub release as `GL_proofs_<version>.zip`. They are about a gigabyte unpacked, grow with every step toward FTA, and will grow much faster in the next phase (FTA discovery), so they stay out of the repository.

- **Get them:** download `GL_proofs_<version>.zip` from the release page and unzip it in the repository root (`unzip GL_proofs_<version>.zip`; on Windows `tar -xf GL_proofs_<version>.zip` — Explorer's *Extract All* adds an extra `GL_proofs_<version>` folder whose contents belong in the root); every file lands where a run writes it (main graph: `files/full_proof_graph/index.html`; incubator graph: `files/incubator/full_proof_graph/index.html`).
- **Check the download:** `GL_proofs_<version>.zip.sha256` beside it and the repository's `PROOFS.sha256` hold the archive's SHA-256 (`sha256sum -c GL_proofs_<version>.zip.sha256`; in Windows PowerShell `Get-FileHash GL_proofs_<version>.zip`). After unzipping, `sha256sum -c PROOFS_SHA256SUMS` in the repository root checks every proof file (on Windows in Git Bash). `sha256sum -c SHA256SUMS` checks every file of the repository, `PROOFS.sha256` included, so the tagged repository pins exactly one proof archive. The release turns git's line-ending conversion off (`.gitattributes`), so every clone and source download carries exactly the hashed bytes.
- **Check the proofs themselves:** `python verifier.py` re-checks every proof step of the main graph and `python verifier.py files/incubator/processed_proof_graph` those of the incubator graph; `lake build` in `files/full_proof_graph/lean_export/` re-checks the Lean export in Lean's kernel. Nothing has to be regenerated.
- **Or regenerate them:** `python main.py` proves the same theorems. The published proofs come from a full `--GPU` run on Windows; a processor run may choose different, equally valid proof steps, so compare `files/theorems/theorems.txt` rather than the HTML pages.

## Prerequisites

**Python 3.10+ with the `regex` package** — `regex` is required for recursive pattern matching in HTML proof graph rendering.

| Platform | Install command |
|---|---|
| Windows | `pip install regex` (Anaconda includes pip) |
| Debian/Ubuntu | `sudo apt-get install -y python3-regex` |
| macOS (Homebrew Python) | `pip3 install regex` |

**Hardware.** A 64-bit PC; no GPU is needed. A processor run's prover peaks at about 4–5 GB of memory: 8 GB of RAM is the minimum, 16 GB is recommended, and `--GPU` runs need the 16 GB. Free disk space: about 12 GB — a run writes about 1 GB of proofs, and on Windows the prover reserves 9 GB for its disk-backed memory (`.deload/extent.bin`, kept after the run and replaced by the next one; on Linux the file takes only the space it uses).

**Windows runtime.** The shipped `gl_quick.exe` needs the Microsoft Visual C++ 2015–2022 Redistributable (x64), which most Windows PCs already have; without it the executable cannot start (Windows names `MSVCP140.dll` or `VCRUNTIME140.dll` as missing) and `python main.py` stops at its unit-test gate. Download: <https://aka.ms/vs/17/release/vc_redist.x64.exe>.

**Native executable.** When the release host has the toolchain available, the release bundles pre-built binaries for **Windows** (`gl_quick.exe`) and **Linux** (`gl_quick`, ELF 64-bit, mimalloc-linked) under `GL_Quick_VS/GL_Quick/`. The Linux binary is built on Ubuntu 26.04: it needs glibc 2.38 or newer and mimalloc 3 (`libmimalloc.so.3`, package `libmimalloc3`); on an older distribution, such as Ubuntu 24.04 with mimalloc 2, build it from source. If your platform's binary is absent (the publish host lacked that toolchain) or doesn't run on your system, build from source per the instructions below — on the 32-thread reference laptop Windows takes a couple of minutes and Linux about four to five minutes with `make -j$(nproc)`. macOS binaries are not bundled; macOS users always build from source.

If you prefer to rebuild any of the binaries yourself — or you're on a platform where the shipped binary doesn't run — see *Rebuilding the native executable* below.

## Quick start

Run from the repository root (where `main.py` lives) on Windows / macOS / Linux:

```bash
python main.py
```

This runs every batch on the processor; no GPU, CUDA installation or Lean toolchain is needed. On a PC with an NVIDIA GPU (driver with CUDA 13 API), `python main.py --GPU` runs every batch's Phase 2 on the GPU instead; it proves the same theorems (the proof steps may differ from a processor run's), the run is several times faster, and a missing device or driver is reported as an assertion, never worked around. The FTA shortcut takes the same flag: `python main.py --shortcut --GPU`.

Use a plain `python` command: GL's checks are `assert` statements, so `main.py` and `verifier.py` refuse to start under `python -O` or with `PYTHONOPTIMIZE` set. A run whose verifier reports a failure ends with exit status 1.

What happens:

- Two unit-test gates run before any pipeline work. The native `gl_quick` binary executes its in-tree C++ unit-test suite (a processor run leaves out the GPU device tests, which only a `--GPU` run needs); `tests/test_harness.py` runs the verifier-side Python suite. The first failing test aborts `main.py` before any proof run starts.
- Python creates conjectures (parallelized).
- Python calls the native prover executable — `gl_quick.exe` on Windows, `gl_quick` on Linux/macOS. Selection is automatic via `run_modes.py::_find_gl_quick_exe()`; you don't pick.
- Python post-processes the raw proof graph and renders the HTML pages.
- Python verifies every chapter row against its tag's structural checker, no human in the loop.

Outputs:

- Generated HTML lives under `files/full_proof_graph/index.html` (plus `chapter*.html`) for the main batches.
- Incubator-batch HTML lives under `files/incubator/full_proof_graph/`.

## GPU (optional) — what you need to run GL on an NVIDIA GPU

GL runs on the processor by default; nothing below is needed for that. With `--GPU` the prover's Phase 2 (the hashburst engine, its main runtime consumer) runs as CUDA threads; it proves the same theorems as the processor route, possibly through different, equally valid proof steps, and the run is several times faster. Everything the GPU route needs on a customer PC, and what happens when something is missing:

- **An NVIDIA GPU and its driver.** The shipped Windows executable links the CUDA runtime statically, so no CUDA installation is needed to *run* it: at `--GPU` time it loads `nvcuda.dll` and `nvcudart_hybrid64.dll`, both installed by the NVIDIA display driver. The driver must support the CUDA 13 API (`nvidia-smi` prints the supported "CUDA Version" in its header; 13.x or newer is fine). The kernels ship for `sm_89` (RTX 40 series) and `sm_120` (RTX 50 series).
- **Run:** `python main.py --GPU` (full pipeline) or `python main.py --shortcut --GPU` (FTA shortlist). `--phase2-backend cpu|cuda` is the explicit per-run override for comparisons.
- **When something is missing** — no NVIDIA GPU, a driver too old for the CUDA 13 API, or a Linux binary built without CUDA — a `--GPU` run stops with an assertion naming the missing piece. GL never silently falls back to the processor: a run either uses the route you asked for or tells you why it cannot.
- **Building the Windows executable yourself** needs the CUDA Toolkit 13.3 (the Visual Studio project imports its build customisation) in addition to Visual Studio 2022; without the toolkit the solution does not build. Download: <https://developer.nvidia.com/cuda-downloads> (Windows, `exe (local)`), then reopen Visual Studio.
- **Linux (Ubuntu / WSL2):** the Makefile builds the processor-only binary by default; `make USE_CUDA=1 -j$(nproc)` adds the CUDA route and needs the toolkit (`CUDA_HOME` defaults to `/usr/local/cuda`; `NVCC` and `CUDA_ARCH` can be overridden). Install the toolkit from NVIDIA's repository — on WSL2 use the WSL-specific repository, which ships driver-free packages (never install the `cuda` or `cuda-drivers` meta-packages under WSL; the Windows driver serves the GPU):

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

## Lean 4 (optional) — independent kernel checking of every proof

When a Lean 4 toolchain is installed, every full run exports its proofs to Lean and kernel-checks them: `python main.py` writes one connected project to `files/full_proof_graph/lean_export/` (Peano, Gauss and the FTA shortlist, each later batch calling the earlier batches' proved theorems, plus every public theorem instantiated on Lean's natural numbers), and every HTML proof page links its Lean twin (`chapter<N>_lean.html`). The development shortcut produces no Lean export. Without a toolchain the run prints one notice and produces no Lean export — nothing else changes. The pinned toolchain version is in `lean_export/lean-toolchain`; `elan` installs it on first use.

| Platform | Install command |
|---|---|
| Windows | `winget install --id LeanProver.Elan` (then reopen the terminal) |
| Linux / macOS / WSL2 | `curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf \| sh` |

GL looks for `lake` on the `PATH` or under `~/.elan/bin`. A failing kernel check stops the run: the Lean export is a verification gate, never decoration.

## Rebuilding the native executable (if needed)

The native prover is a C++ project compiled with Microsoft Visual Studio on Windows 11.

**Windows (Visual Studio 2022)**

Prerequisite — mimalloc: the `.vcxproj` links mimalloc via [vcpkg](https://github.com/microsoft/vcpkg). The prebuilt mimalloc package is **not bundled** in the release (it is Windows-only and large); restore it once with `vcpkg install` from the directory that holds `vcpkg.json`. If you do not need cross-platform proof-graph parity, build with `USE_MIMALLOC=0` instead.

Prerequisite — CUDA Toolkit 13.3: the project imports its build customisation, so the toolkit is needed to *build* on Windows even if you only use the processor route (no GPU is needed to build or to run; see *GPU (optional)* above). The shipped `gl_quick.exe` needs no toolkit.

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

Running `make` again after editing a source file or a header, or with other options (`USE_CUDA`, `USE_MIMALLOC`, `CXX`), rebuilds what the change affects; `make clean` is not needed.

**Why mimalloc is required by default.** GL's prover is a deterministic computational architecture — the same inputs must produce bit-identical proof graphs across platforms. Different memory allocators produce different allocation patterns, which leak into hash-map iteration order, which produces slightly different proof-row orderings. The Windows MSVC build (which ships the prebuilt `gl_quick.exe`) hardcodes `USE_MIMALLOC` in its `.vcxproj`; for the Linux/macOS build to match it, the same allocator must be used. The Makefile defaults to `USE_MIMALLOC=1` to enforce this. If you don't care about cross-platform parity (single-host runs only) build with `make USE_MIMALLOC=0 -j$(nproc)` to skip the mimalloc dependency.

**The build itself**

The Makefile invokes one `c++` process per translation unit and links them into a single `gl_quick` binary in `GL_Quick_VS/GL_Quick/`. A plain `make` builds the processor-only binary — the route every run uses by default — and needs no CUDA toolkit.

If you don't want to use `make`, the equivalent single-process invocation is:

```bash
c++ -std=c++17 -O3 -DUSE_MIMALLOC -Isrc src/*.cpp src/infra/*.cpp src/memory_infra/*.cpp src/gpu/*.cpp src/tests/*.cpp -o gl_quick -rdynamic -lmimalloc
```

The bundled `json.hpp` (nlohmann/json) lives in `src/` and is included as `<json.hpp>`, so the `-Isrc` flag is required either way.

If you place the binary somewhere else or name it differently, update the path in `run_modes.py` (function `_find_gl_quick_exe()`).

## Troubleshooting

- `FileNotFoundError: Native executable not found. Looked for: ...gl_quick.exe, ...gl_quick` — your platform's expected binary is missing. Rebuild per the platform-specific instructions above. The finder checks both names in priority order based on `sys.platform`.
- `error while loading shared libraries: libmimalloc.so.3: cannot open shared object` — the shipped Linux binary needs mimalloc 3. Install it (`sudo apt-get install -y libmimalloc3` where the distribution has it), or build from source with your distribution's `libmimalloc-dev` (see *Rebuilding the native executable*).
- `cannot open include file: 'mimalloc.h'` (compile-time, Linux) — same root cause; the `-dev` package wasn't installed. The Makefile defaults to `USE_MIMALLOC=1`; build with `make USE_MIMALLOC=0` to opt out.
- No HTML output — check `files/raw_proof_graph/*` was generated and that `files/full_proof_graph/` is created. Running `python main.py` regenerates these.
- Slow run — make sure you're using a Release build of the native binary; performance varies by CPU. Reference: about 18–25 minutes for the full pipeline on the processor (most of it the FTA batch) and 5–7 minutes with `--GPU` on a 32-thread i9-13900HX laptop.

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
- **Lean 4** (optional, kernel-checks the exported proofs) — © Microsoft Corporation, Lean FRO — Apache-2.0. <https://github.com/leanprover/lean4>
