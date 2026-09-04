<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# External proof export `[DRAFT]`

## Purpose

[`proof_export`](../../../proof_export) translates selected processed GL proof
chapters into a backend-neutral typed certificate and then renders that
certificate for Lean 4. It is downstream of processed-proof-graph generation
and independent of [`verifier.py`](../../../verifier.py): neither imports the
other, and the export command never runs GL.

**What a run exports (2026-09-04 graphs).** The full run's Peano corpus covers
every Peano source index of its proof graph with an empty exclusion list (64
theorems), and its Gauss corpus every Gauss main source (30 theorems); the
Gauss certificate pins the Peano certificate hash and the Peano theorems Gauss
cites. The shortcut's FTA corpus covers the 79 shortlist theorems (109
chapters, 2,782 rows: 55 direct, 15 induction, seven OR theorems, one OR
elimination, one `proved not broadcast` row) with no internal support
theorems. Empty Peano exclusions became possible when the inter-chapter
theorem-usage cycle was eliminated ([D-276](../40_decisions.md#d-276),
[I-190](../30_invariants.md#i-190)). Counts follow the run; nothing is pinned
ahead of it, and no corpus is tracked — two frozen fixture certificates under
`tests/fixtures/lean_certificates/` serve the renderer tests.

The FTA shortlist is a separate schema-3 export over the shortcut graph. It
imports its externals snapshot as two theorem lists (Peano-anchored,
Gauss-anchored) — 25 Peano and two Gauss theorems on the 2026-09-04 graph — as
certificate-shaped documents. In Lean these are complete universally quantified
theorem propositions, not calls into the full run's generated Peano or Gauss
modules; each FTA theorem receives exactly the propositions required by its
transitive FTA dependency closure, each row specializes them at its current
anchor, and internal FTA calls forward them. Cited externals are resolved by
content: alpha-equivalence first, then a checked adaptation — alias plus
universal-premise permutation, or the registry-independent base form for
compact citations (five Peano uses on the current graph). The certificate
validates the target interface structurally before Lean rendering.

## Data flow

```text
processed proof graph + config + tracked GL binary + selection
                              │
                              ▼
                 typed neutral certificate
                 expressions, steps, dependencies,
                 source hashes, row actions
                              │
                              ▼
                       Lean 4 renderer
              ordinary definitions + named row facts
                              │
                              ▼
       certificate/renderer tests + Peano/Gauss `lake build`
                    + separate FTA kernel compile
```

[`proof_export/certificate.py`](../../../proof_export/certificate.py) owns
certificate schemas 2 and 3. It resolves the theorem registry, assigns stable step
identities, topologically orders rows, records typed expressions and proof
scope, rejects missing self-contained dependencies, and emits portable source
labels plus SHA-256 hashes. Prior-theorem rows name the selected certificate
theorem identifier and source index. The complete selected theorem-dependency
graph is alpha-resolved and asserted acyclic before action emission. Schema 3
also hashes its dependency certificate, validates integration-scope boundaries,
and proves from the compiled definition closure that each reformulated output
has a non-recursive constructive witness.

FTA extends the same schema with an explicit external theorem-list scope. A raw
source index is resolved only inside its declared theorem list, so shortcut
indices cannot collide with Peano or Gauss indices. The certificate validates
every declared head alias by arity, category, and exact compiled elements, then
matches source and target theorems under a bijective variable renaming and a
permutation of universal Horn premises. It also validates OR branch scopes,
contradiction-assumption scopes, OR convergence, inequality symmetry, and the
single compilation action before emitting their neutral actions.

[`proof_export/lean.py`](../../../proof_export/lean.py) owns Lean syntax. It
generates the exact typed GL-binary definition closure, one named Lean fact for
every certificate row, theorem-specific induction helpers, public theorems, and
the manifests (`lean_manifest.json`, `lean_gauss_manifest.json`,
`lean_fta_manifest.json`) in the run's export folder. Manifest generation
asserts that every corpus row has one `status = emitted` disposition. The root
`GLExport.lean` of an export imports the theories generated into it.
Inequality-symmetry rows are rendered as the direct term
`fun equality => cited_inequality (Eq.symm equality)`; a scoped row first enters
its recorded scope and specializes that same cited fact. Scoped task-formulation
rows are the identity proof of their recorded scope premise and return that
premise directly. Every `equality2` row applies `Eq.trans` to its two cited
equalities in certificate order. A contradiction row either applies its cited
negation to the cited positive fact and eliminates `False`, or introduces the
recorded contradiction-scope premise and applies the scoped negation to that
premise and its main-scope complement. Every `premise element` row asserts that
its cited main-scope implication is the recorded scope template, introduces
that template, and returns the uniquely matching `scope_premise_N`. Every
`implication` row applies its cited rule and supplies the cited premises in the
rule's own order. A scoped implication first specializes each same-scope
citation to the recorded scope; a scope-local witness projection supplies the
exact matching component of its recorded witness bundle. Every empty-bound
integration reformulation applies the reverse direction of its cited
definitional equivalence and refutes the universal counterexample with the
selected output and the two introduced premises. Every vacuous-truth row finds
the unique distinct complementary pair in its citation and eliminates the
resulting `False`; a repeated citation does not create a second logical pair.
Every bounded integration reformulation has the same direct equivalence proof:
after entering any recorded scope, its concrete bound output and introduced
premises refute the cited universal counterexample. Every OR-disintegration row
enters its licensed branch scope and returns the selected branch assumption
directly; the exported row is the exact identity implication recorded by that
scope. Every constructed-OR theorem unfolds the compiled left-associated Lean
disjunction, performs an explicit case split on each disjunct except the last,
and applies its first cited parent to the accumulated negations to construct the
last disjunct. The second parent, when recorded, is an alternate licensed
direction rather than a premise needed by that proof. OR convergence unfolds
only the cited compact OR, splits its two branches, and applies the matching
cited branch result. When a branch is a compact existence proposition, the proof
keeps that compact assumption, decodes its witness through the shared
negated-universal proof-support theorem, and passes the exact recorded
witness-bundle component to the scoped result.
Theorem reformulation constructs the checked set or binary-relation output,
proves its defining totality condition, and then applies the single cited source
theorem in certificate order to that output and the recorded input premise.
The theorem-level FTA OR elimination obtains the domain membership needed by its
OR parent from the recorded multiplicative preorder witness and AnchorFTA's
multiplication output closure, splits the resulting compact OR, and applies the
two cited branch theorems directly.
Equality1 rows name the cited source and each cited equality, eliminate those
equalities in certificate order with Lean's equality recursor, and return the
rewritten source. The exporter first asserts that source and target normalize to
the same MPL expression under exactly those equality classes.
Disintegration follows one of three visible certificate shapes: a named
conjunction projection, a witness decoded once from GL's negated-universal
existence encoding and shared by its component rows, or a direct implication
constructed from the cited negated conjunction. Scoped forms first instantiate
the cited compound under the recorded scope prefix and use the same operations.

For FTA, the shortcut export is a self-contained Lake project.
`Definitions.lean` carries the complete definition closure the shortlist needs
(its own `AnchorFTA`, the shortcut-local compacts and `strictOrder`, and the
Peano and Gauss heads it shares); `FTA.lean` imports it, derives the Peano and
Gauss anchors from `AnchorFTA`, and emits one fact for every one of the corpus's
rows (2,782 on the 2026-09-04 shortcut graph). Externals-snapshot Peano and
Gauss theorem content enters through explicit target-fact parameters; generated
FTA source contains no calls to `peano_source_*` or `gauss_source_*`.

## Ordinary-type boundary

The exported GL object theory has one carrier `α`. `GLSet α` is `α → Prop`, a
binary relation is `α → α → Prop`, and a ternary relation is
`α → α → α → Prop`. Every compiled definition and theorem is polymorphic over
that carrier. No GL term indexes a Lean type, and no GL value is represented by
a Lean type family.

Lean's kernel is itself based on dependent type theory; using Lean cannot change
that foundation. The architectural constraint here concerns the encoded GL
theory, which stays in an ordinary many-sorted predicate fragment.

## Row replay and induction

Definition expansion and conjunction projection use explicit generated terms.
The 136 schema-3 Gauss and FTA integration-goal expansions introduce any
proposition-level binders explicitly and close their definitionally identical
unfolded sides with `Iff.rfl`; no generated Lean line ends in an unrestricted
`simp` tactic.
Implication application is structural in every corpus: the row applies the cited
rule and discharges each premise with the cited row, in the order
`_ordered_implication_dependencies` derives from the rule's own premise order.
No search stands between a GL rule firing and its Lean counterpart.

Gauss integration boundaries become ordinary universally quantified
implications. Free integration parameters are closed at the boundary. A scoped
row's proof enters its own scope through `_scope_entry_and_specializations`,
which introduces the prefix the scope template renders and then instantiates
every cited same-scope fact at that prefix. Each cited fact is instantiated at
**its own** prefix, not the consuming row's: a main-scope witness projection
binds one more integration parameter partway through a chapter, so two rows in
one namespace can carry different prefixes.

A scope-local existence witness is one variable GL's rows share. The projecting
row states it existentially; every other row about that witness is universally
quantified over it and guarded by the witness bundle. A consumer obtains the
witness once from the projecting row and applies each guarded row to exactly
that witness. Quantifying every row's witness separately would drop the sharing
and leave a step underivable from its complete citation — see
[D-293](../40_decisions.md#d-293).

Every row and every composition helper now uses an explicit Lean proof term;
the renderer and the tracked Peano, Gauss, and FTA modules contain no `grind`.
Equality substitution eliminates the named equalities in certificate order,
witness projection follows the named conjunction path, and scoped facts are
specialized at their recorded prefixes. Induction typing and zero helpers apply
their cited rules directly; successor assumptions are returned by name;
induction-hypothesis steps apply `induction_hypothesis` and discharge only the
introduced premises; the public result applies `inductionProperty` explicitly.
The result is stricter than context restriction around search: the generated
term itself records the route from each GL citation to its row.

Every result is assigned the stable source-row fact name recorded in the
manifest. Prior-theorem references point only backward in the combined acyclic
order.

FTA OR namespaces are ordinary implications from the selected disjunct to the
branch result. OR convergence unfolds the checked compiled OR definition and
eliminates those branch implications. Contradiction namespaces likewise become
ordinary assumption implications; their main-scope contradiction row derives
the recorded negation. Compilation compaction unfolds the checked compact
definition around the already selected expanded theorem. A zero-argument FTA
compact introduces an arbitrary carrier, operations, and `AnchorFTA`, then
replays the earlier FTA theorem at that context with the same explicit global
induction and external-theorem parameters. No direction is reversed and no
scoped premise escapes as a global fact.

The eight Gauss `reformulated statement` rows use GL's well-defined-output
contract constructively rather than adding an existence axiom. For `limitSet`,
the renderer supplies the set comprehension
`fun x => V x ∧ preorder N add x n`; for `limitSequence`, it supplies the
validated binary-relation lambda. It unfolds the non-recursive compiled
definition, proves the defining predicate, applies the cited source theorem to
that concrete output, and folds the two facts into the recorded GL existence
compact. These witnesses are ordinary predicate values, never dependent types.

Each induction theorem's typing, zero, and induction-condition chapters become
three private helper theorems. The public theorem states its induction predicate
explicitly and composes those helpers through the ordinary proposition premise
`relationalInduction`. For FTA this premise is a complete `AnchorFTA`-indexed
schema, so it can be specialized both at the current theorem context and at an
arbitrary context introduced by a globally quantified compiled proposition.
The induction principle is therefore visible at every public theorem boundary;
it is not an axiom and is not hidden inside an anchor.

## Trust and verification gates

- The renderer rejects `sorry`, `admit`, `axiom`, `opaque`, and `unsafe` in its
 generated source.
- The full-corpus writer asserts the corpus's recorded coverage (theorem
 count, exclusion list) against what it emitted. The renderer tests pin the
 historical excluded-pair era through the frozen fixture
 `tests/fixtures/lean_certificates/lean_full_65.json` (65 theorems,
 exclusions 24 and 25).
- The Gauss writer asserts the Peano dependency hash, a combined acyclic
 theorem graph, and its recorded theorem and row totals (fixture
 `lean_gauss_main_29.json`: 29 theorems, 1,511 rows).
- The FTA writer asserts its corpus theorem and row totals, its externals
 references (25 Peano and two Gauss on the 2026-09-04 shortcut graph), and
 the isolated definition heads. Its writer test runs on the live shortcut
 export and proves the self-contained layout: `Definitions.lean`, `FTA.lean`,
 the root, and the manifest, nothing else.
- `ExplicitRowReplayTests` scans the generated theory files of both run
 exports (skipped when no export is on disk) and rejects
 any remaining `grind`. Its per-action tests require the explicit proof shape;
 the inequality-symmetry test, for example, requires all 26 such rows to use
 `Eq.symm` directly. The corpus render tests also require explicit induction
 typing, zero, successor, hypothesis, and final-composition applications.
 `ScopeEntryTests` covers scope entry, per-dependency prefixes,
 and witness sharing directly, including the negative case where a cited
 guarded fact names a witness no cited projection establishes.
- `tests.test_proof_export_lean` checks the ordinary-type encoding,
 deterministic definition and theorem rendering, both corpus counts, schema-3
 scope metadata, constructive output totality, FTA external-theorem adaptations,
 OR and contradiction scopes, row-total manifests, explicit cycle exclusions,
 and trust-shortcut rejection.
- `lake build` in each run export checks its modules with the pinned Lean
 toolchain. The shortcut export is self-contained, so its `lake build` checks
 FTA through the complete universally quantified propositions of its externals
 references. The certificate/renderer suite validates those
 propositions and their adaptations and rejects any coupling to live
 `peano_source_*` or `gauss_source_*` proof terms.
- Adding a proof action requires a neutral certificate contract, renderer path,
 direct negative and positive test coverage, manifest coverage, and a successful
 kernel build when the module belongs to the default Lean target.

The tracked FTA artifacts pass both the Python certificate/renderer tests and
their separate Lean kernel compile through the pinned external-theorem
interface. Across the three tracked exports the renderer emits and
kernel-checks 172 public theorems, 280 chapters, and 7,453 named row facts; the
default import target contains the 93 Peano/Gauss theorems and their 4,618 named
row facts, while FTA remains a separately compiled interface module.

## Live export inside the run

[`proof_export/live.py`](../../../proof_export/live.py) runs the export as a
pipeline stage ([D-339](../40_decisions.md#d-339)):
`run_modes.full_run` calls `export_main_graph` after the main processed graph
and before the HTML pages, `run_modes.shortcut_run` calls
`export_shortcut_graph` at the same point of the shortcut. The incubator graph
has no export: its `incubator back reformulation` rows are unsound by design.

Nothing is pinned ahead of a run, so the selection is derived from the run:

- `theorem_list_blocks` partitions `global_theorem_list.txt` into its
 contiguous anchor blocks (`AnchorPeano`, then `AnchorGauss`; the shortcut is
 `AnchorFTA` only) — each block is a corpus' `source_range`, the theorem-list
 hash is computed, exclusions are empty, and the selection carries no
 `expected_coverage` (the builder then records coverage instead of asserting
 it; a tracked release selection still pins it).
- Corpus ids are `peano_live_N` / `gauss_live_N` / `fta_live_N`;
 `lean.corpus_kind` classifies every id by its prefix, so the renderer's
 anchor context, imports and manifest shape follow the kind, and the exact
 theorem and row counts are asserted only for the tracked release ids
 (`lean.TRACKED_CORPUS_COUNTS`).
- Gauss depends on the Peano certificate just written (hash computed); its
 `required_source_indices` are the Peano theorems Gauss actually cites,
 found by `same_list_imports` through the shared `certificate.theorem_citations`.
- FTA is independent of the main path (maintainer ruling, 2026-09-04): its
 only dependency is the tracked externals snapshot the shortcut proved
 against, `files/shortcut/theorems/externally_provided_theorems.txt`, which
 need not match the latest main run. `external_theorem_lists` splits the
 snapshot's base-form rows by anchor into a Peano-anchored and a
 Gauss-anchored list, each written as a certificate-shaped document
 (`certificates/<kind>_externals/certificate.json`: id `<kind>_externals_N`,
 the rows as theorem records, the snapshot's hash, no compiled definitions,
 no dependency edges), which the builder loads like any dependency
 certificate. `external_references` resolves every citation that is no FTA
 theorem: first alpha-equivalently, else by the structural search the
 validator accepts —
 `infer_head_aliases` finds the source names of renamed compiled heads
 (identical arity, category and elements under the aliases found so far,
 closed by iteration), and `certificate.theorem_adaptation_solutions` (the
 non-asserting twin of `_validate_theorem_adaptation`) matches one bijective
 renaming and one premise permitting permutation. Exactly one dependency
 theorem may match; the result is written in the selection's
 `theorem_adaptations` format, so the builder validates it on its declared path.
 When no alias-and-permutation match exists the search compares
 registry-independent **base forms** (`certificate.base_form_equivalent`):
 exactly the compiler's spontaneous compacts (`existence<N>`, `or<N>`,
 `implication<N>`) are unfolded, as the prover's own `expandToBaseForm`
 does — `and` to a conjunction and `or` to the negated conjunction of
 negated parts, both with parts in name-free shape order; implication
 compacts to their bound implication chains; existence compacts to
 `¬ ∀ x, e_1 → … → ¬ e_k`; a double negation cancels — while every
 config-defined operator (anchors, `NaturalNumbers`, `preorder`, `fold`, …)
 stays an atom. The two Horn spines are then compared alpha-equivalently
 with all binders merged into one group and under a premise permutation.
 This is how a compact citation (`∀ w1 ∈ N, or2[w1,i0,N,s]`) meets its
 base-form externals row (`…!(&!(=[7,2])(>[8](in[8,1])!(in2[8,7,3])))`).
 Such an adaptation is recorded with `base_form: true`, carries no head
 aliases and no renaming, and enters Lean like every external theorem of
 the FTA corpus: as the validated target proposition parameter of the
 citing theorem, never a proof-term import. A statement the dependency
 list proves more than once resolves to its lowest source index.
- The shortcut's Lean project is self-contained: `Definitions.lean` holds
 every definition the FTA corpus uses, `FTA.lean` imports only that, the
 root makes FTA the default target, and `lake build` is the whole check
 (`lean.write_lean` selects this layout when the certificate carries no
 `definition_isolation`; the tracked release corpus keeps the isolated
 layout over the reviewed Gauss module).
- Outputs land in the mode's own proof-graph folder,
 `files/full_proof_graph/lean_export/` and
 `files/shortcut/full_proof_graph/lean_export/`: the selections, the
 certificates under `certificates/<kind>/`, a complete Lake project (skeleton
 copied from the tracked `lean_export/`: lakefile, toolchain pin, manifest,
 `ProofSupport.lean`), the generated modules and manifests, and
 `kernel_check.log`.
- Toolchain policy (maintainer, 2026-09-04): `find_lake` looks for `lake` on
 the PATH and in `~/.elan/bin`. Absent → one notice line, no export, no Lean
 pages, no crash. Present → export, `lake build`, and for the shortcut the
 the self-contained FTA project's `lake build`; a kernel failure fails the
 run.

The live builder is also the re-pin tool: a tracked release selection is the
live selection with `expected_coverage` and the dependency hashes filled in.

**Writer case added by the live export (2026-09-04).** The current Peano
generation records an `expansion` row whose source is a negated existence
compact and whose conclusion is one of its implication compacts (I-209): the
negation of `¬ ∀ x, left → ¬ right` yields the positive universal, and the
compact states it as `left → ¬ right` or `right → ¬ left`.
`lean._existence_implication_direction` recognizes the shape from the two
compiled definitions, decides the orientation by alpha-comparing the
instantiated bound implications, and the row replays as: unfold the compact,
introduce the witnesses and the two hypotheses, unfold the negated existence at
the source fact, `apply` its double negation, introduce the positive universal,
and apply it to the hypotheses in the decided order (a scoped row enters its
scope first and instantiates the cited fact there). Regression:
`tests/test_lean_existence_compacts.py`.

**Further writer cases added by the live shortcut export (2026-09-04).** The
current shortlist generation exercises row shapes the tracked corpus did not:
a `compound_project` whose source row is guarded by a witness the projected
conjunct never mentions (the projection now inherits every guard of its
source, in dependency order); a contradiction row with both contradictory
facts inside the contradiction scope (the row enters the scope once and
instantiates every scoped fact there); a `compound_project` from a De Morgan
disjunction `¬(¬A ∧ ¬B)` whose conclusion is the implication compact
`implication<N>[…]` = `¬A → B` (the compact is unfolded first); an induction
step whose helper assumes the current value's membership (derived from the
successor edge through the anchor's `fXY` closure, `induction_m_member`); and
the pre-split-merge (or elimination) theorem, whose anchor projection paths,
closure compact name and disjunct order are now read off the corpus'
compiled definitions (`_left_associated_projection`) instead of being
assumed from one registry generation.

## Release and snapshot protocol

Nothing about the Lean export is tracked beyond the exporter, the renderer
tests, two frozen fixture certificates under `tests/fixtures/lean_certificates/`,
and the Lake skeleton in `lean_export/` (`lakefile.lean`, `lean-toolchain`,
`lake-manifest.json`, `GLExport/ProofSupport.lean`) that `proof_export/live.py`
copies into every export. The release package is the two exports the pipeline
writes: `files/full_proof_graph/lean_export/` and
`files/shortcut/full_proof_graph/lean_export/`.

[`release_public.py`](../../../.scripts/release_public.py) never invokes the
exporter. `check_live_lean_exports` requires both folders with a green
`kernel_check.log`, re-runs `lake build` in each, runs
`tests.test_proof_export_lean`, then copies both folders through the ordinary
source walk and compares every result byte against the source
(`lean_export_copy_issues`) before the release can pass.

[`snapshot_project.py`](../../../.scripts/snapshot_project.py) overlays the same
two folders into the private recovery copy and compares every result byte before
building the ZIP. Both protocols exclude `.lake/`: it is a host-local build
cache, not an export result.

## Weaknesses

### Known & tracked

- **Every export input is a gitignored run artifact.** Both
 `files/processed_proof_graph/` and `files/shortcut/processed_proof_graph/`
 match `.gitignore`'s `**/processed_proof_graph/`, and so do the exports
 themselves. A certificate is reproducible only on a host whose matching
 pipeline run is still on disk; the recorded `theorem_list_sha256` makes that
 dependence loud instead of silent. Nothing tracked pins a graph generation
 any more: each run derives its own selection and the release ships the run's
 folders.
- **The FTA export follows the externals snapshot, not the latest main run (by
 design, maintainer ruling 2026-09-04).** The shortcut's Peano and Gauss
 theorem lists are the rows of its tracked externals snapshot. A Peano theorem
 the current main batch no longer proves still enters FTA as an explicit
 premise, because the shortcut graph was produced against the snapshot.
- **The shortcut export is its own Lake project.** `Definitions.lean` carries
 the definition closure, `FTA.lean` imports it, and the root imports FTA, so
 `lake build` there checks FTA directly; there is no shared project with the
 full run's Peano and Gauss modules.
- **No corpus refresh step exists.** Source indices move with every prover
 generation; the live selection is derived per run, so nothing is remapped by
 hand and no re-pin follows a verified full run.
- **`lean_full_65` is a frozen fixture.**
 `tests/fixtures/lean_certificates/lean_full_65.json` records the excluded-pair
 era for the renderer tests and can never be rebuilt from a live graph;
 `lean_gauss_main_29.json` is the schema-3 fixture next to it.

See
[D-294](../40_decisions.md#d-294),
[D-275](../40_decisions.md#d-275),
and
[I-189](../30_invariants.md#i-189).

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
