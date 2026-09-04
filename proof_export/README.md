# GL Peano, Gauss, and FTA proof export to Lean 4

`proof_export` converts a processed GL proof graph into a typed,
backend-neutral certificate and renders that certificate as a Lean 4 project.
It does not invoke GL.

**Inside the run.** Since release 13 the export is a pipeline stage
(`proof_export/live.py`): `python main.py` exports and kernel-checks the Peano
and Gauss theorems into `files/full_proof_graph/lean_export/`, and
`python main.py --shortcut` exports the FTA shortlist into
`files/shortcut/full_proof_graph/lean_export/`, independent of the main path:
its only dependency is the shortlist's tracked externals snapshot, whose rows
become two theorem lists (Peano-anchored, Gauss-anchored) and enter Lean as
explicit premises of the citing theorem. The selection is derived from the
run — the anchor blocks of `global_theorem_list.txt`, the computed
theorem-list hash, cited theorems resolved by content (alpha-equivalence or
registry-independent base form) — so nothing is pinned by hand for a run.
Each folder holds the selections, the certificates under
`certificates/<kind>/`, a complete Lake project with the generated modules and
manifests, and `kernel_check.log`; every HTML chapter links its Lean twin
(`chapter<N>_lean.html`). The stage needs `lake` on the PATH or under
`~/.elan/bin`; without a toolchain the run prints one notice and produces no
Lean export. The two export folders are the release's Lean package; no corpus
is tracked in the repository.

The exported object theory uses ordinary types: one abstract carrier `α`, sets
as `α → Prop`, binary relations as `α → α → Prop`, and ternary relations as
`α → α → α → Prop`. Lean itself is founded on dependent type theory, but no GL
value becomes a Lean type and the export defines no term-indexed type family.

**What a run exports (2026-09-04 graphs).** The full run's Peano corpus covers
every Peano source index of its proof graph with an empty exclusion list (64
theorems), and its Gauss corpus every Gauss main source (30 theorems); the
Gauss certificate pins the Peano certificate hash and the Peano theorems Gauss
cites. The shortcut's FTA corpus covers the 79 shortlist theorems (109
chapters, 2,782 rows) with no internal support theorems. Counts follow the
run; nothing is pinned ahead of it.

**FTA and its externals.** The shortcut's certificate treats the externals
snapshot as two theorem lists with certificate-shaped documents under
`certificates/peano_externals/` and `certificates/gauss_externals/`. Cited
externals are resolved by content: alpha-equivalence first, then a checked
adaptation (alias plus premise permutation, or the registry-independent base
form for compact citations). No external proof body is imported: each FTA
theorem receives the exact validated, universally quantified target theorems
required by its dependency closure as explicit Lean proposition parameters,
each row specializes them at its FTA anchor, and internal FTA theorem calls
forward the complete parameters. The shortcut's Lake project is
self-contained: `Definitions.lean` carries the definition closure, `FTA.lean`
imports it, and the root imports FTA.

Gauss integration scopes are emitted as ordinary universally quantified
implications. GL existence compacts remain propositions: the renderer builds a
set-comprehension or relation-lambda witness for each `reformulated statement`
theorem and proves its defining predicate. No GL existence axiom or dependent
GL type family is introduced.

**Manual export.** `python -m proof_export.export_lean` still renders one
hand-written selection against a processed graph (`--selection`,
`--proof-graph`, `--config`, `--gl-binary`, `--certificate-output`,
`--lean-project`); the selections a run writes (`selection_peano.json`,
`selection_gauss.json`, `selection_fta.json`) are the reference shape.

Check the renderer contract and a run's Lean project:

```powershell
C:\Users\nikol\anaconda3\python.exe -m unittest tests.test_proof_export_lean -v
Set-Location files\full_proof_graph\lean_export
lake build
```

The Lean toolchain is pinned in `lean_export/lean-toolchain`, which every
export copies. The build must contain no `sorry`, `admit`, `axiom`, `opaque`,
or `unsafe` shortcut. Peano induction is an explicit theorem premise named
`relationalInduction`; it is not introduced as an axiom or hidden inside the
compiled anchor. FTA carries the same premise as an explicit
`AnchorFTA`-indexed schema, so a zero-argument compiled proposition can
introduce an arbitrary FTA context and replay its earlier theorem there. The
renderer and all generated proof modules contain no `grind`: induction typing,
base, successor, induction-hypothesis, and final composition steps apply their
named premises explicitly.

The shortcut export is checked the same way (`lake build` in
`files\shortcut\full_proof_graph\lean_export`) through the explicit
external-theorem parameters; FTA does not import Peano or Gauss proof content.
The renderer tests that read a run export are skipped on a host that has none.

The official release requires both run exports with a green
`kernel_check.log`, re-runs `lake build` in each, runs the contract suite, and
copies both folders into the public tree byte-for-byte. The private recovery
snapshot copies the same two folders. Both exclude the rebuildable `.lake/`
caches.
