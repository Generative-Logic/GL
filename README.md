# Licensing 📜

Generative Logic is dual-licensed under the AGPLv3 and a Commercial License.

**1. AGPLv3 License (Open-Source)**

For open-source projects, academic research, and personal use, Generative Logic is licensed under the AGPLv3.
This license requires that any derivative works or applications that use this software must also be made open-source under the same terms. The full license text is available in the [LICENSE](LICENSE) file.

**2. Commercial License**

If you wish to use Generative Logic in a proprietary, closed-source commercial application, a commercial license is required. This license frees you from the obligations of the AGPLv3 and includes enterprise-grade features such as a limited warranty and IP indemnity.

For more details and to request a quote, please visit our commercial licensing page:
**https://generative-logic.com/license**

**Contributing**

All contributions require a signed Contributor License Agreement (CLA).
See `legal/CONTRIBUTOR_LICENSE_AGREEMENT.md` for details.

---

# Generative Logic

Deterministic computer architecture for automated mathematical reasoning. From user-supplied axiomatic definitions in MPL (Mathematical Programming Language), GL enumerates conjectures, filters by counterexample, and proves or disproves each one by hash-based inference over a distributed grid of logic blocks. No human guidance, no goal-setting; theorems are discovered and proved autonomously, with a provenance-complete proof graph for every emitted fact.

- Paper: <https://arxiv.org/abs/2508.00017>
- Site:  <https://generative-logic.com>
- Author: Dr. Nikolai Sergeev — Generative Logic UG (haftungsbeschränkt)

## Three pillars

GL's reasoning surface decomposes into three pillars, each contributing a distinct class of inference. Each pillar carries roughly 10 000 lines of C++ + Python and roughly eight months of build time, on top of shared infrastructure.

1. **Algebraic manipulations** — *Peano*, shipped 2025-09.
   Commutativity, associativity, distributivity of `+` and `*` on a recursively-defined natural-number structure.

2. **Logical transformations** — *Gauss*, shipped 2026-02.
   The autonomous-discovery summation formula `n · (n+1) = 2 · Σ i` in its division-free form, with accompanying anchor-bound consequences.

3. **Case differentiation — branching** — *in progress, version 0.1, 2026-05*.
   Sub-proofs split on disjunctive hypotheses, run independently in scoped sub-contexts, and recombine. **Initial version 0.1 closed the forward direction of FTA-ladder rung 1, `{0, 1} = [0, 1]`.**

The Fundamental Theorem of Arithmetic is the next major milestone, climbed one rung at a time along an FTA ladder of intermediate theorems.

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

## Status snapshot (2026-05)

| Pillar | Status |
|---|---|
| Algebraic manipulations (Peano) | shipped 2025-09 |
| Logical transformations (Gauss) | shipped 2026-02 |
| Case differentiation — branching 0.1 | shipped 2026-05 (FTA-ladder rung 1, `EnumerationSet2 ⟹ interval`) |

The next frontier:

- **Incubator-side ground facts** to filter statements about intervals and sequences — extensions of the rung-1 pattern to larger fixed sets, e.g. `{0,1,2}=[0,2]`, `{0,1,2,3}=[0,3]`, and analogous claims. These are the concrete-instance theorems the CE filter needs in order to kill false conjectures about general intervals and sequences before they reach the prover.
- **Euclid's lemma**.
- **FTA proper**.

---

# Run Mode & How to Run

## Run mode

There is only one run mode (previously called "Full Mode").

End-to-end runtime, full pipeline (reference): ~21 minutes on a Dell G16 7630 (Intel-class laptop), driven by the incubator + branching stages.

## Prerequisites

Python 3.9+ with the `regex` package (`pip install regex`) — required for recursive pattern matching in HTML proof graph rendering.

For Windows users: the bundled native executable `GL_Quick_VS/GL_Quick/gl_quick.exe`.

For non-Windows or if the executable is missing: a C++17 toolchain to rebuild the native component (see below).

## Quick start

Run from the repository root (where `main.py` lives) on Windows / macOS / Linux:

```bash
python main.py
```

What happens:

- Python creates conjectures (parallelized).
- Python calls the native prover executable `GL_Quick_VS/GL_Quick/gl_quick.exe`.
- Python renders the proof graph pages.

Outputs:

- Generated HTML lives under `files/full_proof_graph/index.html` (plus `chapter*.html`) for the main batches.
- Incubator-batch HTML lives under `files/incubator/full_proof_graph/`.

## Rebuilding the native executable (if needed)

The native prover is a C++ project compiled with Microsoft Visual Studio on Windows 11.

**Windows (Visual Studio 2022)**

1. Open `GL_Quick_VS/GL_Quick.sln` in Visual Studio.
2. Build Release x64 (recommended).
3. The binary will be at `GL_Quick_VS/GL_Quick/gl_quick.exe` (this is the path the Python code expects).

**Linux / macOS (experimental)**

Source code is under `GL_Quick_VS/GL_Quick/src/`. If the code is portable, you can try:

```bash
cd GL_Quick_VS/GL_Quick
c++ -std=c++17 -O3 src/*.cpp -o gl_quick
```

If you place the binary somewhere else or name it differently, update the path in `run_modes.py` (function `run_gl_quick()`).

## Troubleshooting

- `FileNotFoundError: GL_Quick_VS/GL_Quick/gl_quick.exe` — rebuild the native executable or ensure the file exists at that path.
- No HTML output — check `files/raw_proof_graph/*` was generated and that `files/full_proof_graph/` is created. Running `python main.py` regenerates these.
- Slow run — make sure you're using a Release build of the native binary; performance varies by CPU.

## Paths recap

- Entry point: `main.py`
- Native executable (Windows): `GL_Quick_VS/GL_Quick/gl_quick.exe`
- Native source: `GL_Quick_VS/GL_Quick/src/`
- Output HTML (main): `files/full_proof_graph/`
- Output HTML (incubator): `files/incubator/full_proof_graph/`

---

# 3rd party notices

- **regex** — © Matthew Barnett — Apache-2.0 and CNRI-Python.
- **nlohmann/json** — © Niels Lohmann — MIT. <https://github.com/nlohmann/json>
- **mimalloc** — © Microsoft Corporation — MIT.
