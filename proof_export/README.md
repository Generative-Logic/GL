# GL Peano, Gauss, and FTA proof export to Lean 4

`proof_export` converts a processed GL proof graph into a typed,
backend-neutral certificate and renders that certificate as a Lean 4 project.
It does not invoke GL.

**Inside the run.** The export is a pipeline stage (`proof_export/live.py`):
the full run (`python main.py`) exports the Peano, Gauss and FTA theorems of its
single processed theorem list as one connected Lean project in
`files/full_proof_graph/lean_export/` and kernel-checks it. FTA is the third
main batch; its certificate depends on the same run's Peano and Gauss
certificates, and its Lean module calls their proved theorems. The selection is
derived from the run — the anchor blocks of `global_theorem_list.txt`, the
computed theorem-list hash, cited theorems resolved by content
(alpha-equivalence or registry-independent base form) — so nothing is pinned by
hand for a run. `proof_export/nat.py` proves the anchors on Lean's natural
numbers, supplies induction from `Nat.rec`, and instantiates every public
theorem; the single default Lake target includes all those instances. The
folder holds the selections, the certificates under `certificates/<kind>/`, a
complete Lake project with the generated modules and manifests, and
`kernel_check.log`; every HTML chapter links its Lean twin
(`chapter<N>_lean.html`). The development shortcut (`python main.py
--shortcut`) produces no Lean export. The stage needs `lake` on the PATH or
under `ELAN_HOME/bin` (default `~/.elan/bin`); without a toolchain the run
prints one notice and produces no Lean export. Kernel-check children receive
explicit `ELAN_HOME`, preserving a configured override or defaulting to
`~/.elan`, so elan does not rely on implicit home-directory discovery. This
changes neither the project's toolchain pin nor the rule that a kernel failure
fails the run. The export folder is the release's Lean package, published in
the release's proof archive; no corpus is tracked in the repository.

The exported object theory uses ordinary types: one abstract carrier `α`, sets
as `α → Prop`, binary relations as `α → α → Prop`, and ternary relations as
`α → α → α → Prop`. Lean itself is founded on dependent type theory, but no GL
value becomes a Lean type and the export defines no term-indexed type family.

**What a run exports.** Every Peano source index of the run's proof graph,
every Gauss main source and every FTA source, with an empty exclusion list;
each later certificate pins the earlier certificates' hashes and the theorems
it cites. Counts follow the run; nothing is pinned ahead of it.

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
python -m unittest tests.test_proof_export_lean -v
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

The renderer tests that read a run export are skipped on a host that has none.
Tests that kernel-check generated Lean files use the `lean` beside the Lake
the export would find (`PATH`, then `ELAN_HOME/bin`, default `~/.elan/bin`)
and are skipped on a host without a Lean toolchain.

The official release requires the full run's connected export with a green
`kernel_check.log`, re-runs `lake build` in it, runs the contract suite, and
publishes the folder in the proof archive (`GL_proofs_<version>.zip`, a GitHub
release attachment), checking that the archive carries it byte for byte. The
private recovery snapshot copies the same folder. Both exclude the rebuildable
`.lake/` cache.

`zoom in` rows replay as explicit scope inheritance from one exact local parent
step. The child expression is unchanged; the parent scope and proof stay explicit.
The Lean renderer reuses that cited proof under the child context, including its
parameters and witness guards, without proof search or a new axiom.
