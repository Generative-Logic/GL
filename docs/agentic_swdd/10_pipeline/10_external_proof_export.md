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

The one-chapter theorem method is `single chapter proof`; its processed source
file and certificate role are `single_chapter_proof`. Lean manifests preserve
that role and the `<N>_single_chapter_proof.txt` source filename. The final
HTML's **Direct Proof** versus **Contradiction Proof** depiction is deliberately
downstream-only and does not split the certificate role.

The full run exports all selected Peano, Gauss, and FTA source rows from its
single processed theorem list, with no silently excluded source. FTA is the
third main batch and consumes the nonempty external conjecture file as goals.
Its certificate depends on the actual same-run Peano and Gauss certificates,
and its Lean module calls their proved theorems. Counts follow the run.

Every public context requires induction for its fixed structure. A fully bound
anchor implication explicitly carries its own induction premise inside that
binder. This convention also applies to its compiled definition and expansion;
it does not assert that a bare anchor implies induction. `proof_export/nat.py`
/ `write_nat_model` proves all three anchors on Lean's Nat, supplies induction
from Nat.rec, and creates a concrete instance of every public theorem. The
single default Lake target includes all those instances.

The shortcut remains for development. It replays the captured full-run FTA
input, including preceding proof chapters, into its own output tree. It produces
no Lean export;
`proof_export/live.py` / `skip_export` removes stale generated certification
before its HTML pages are rendered. Historical external-snapshot certificates
remain renderer fixtures only. Their explicit theorem parameters now include
the induction requirement for the quantified structure and cannot be accepted
by the connected natural-number writer.

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
       certificate/renderer tests + one connected `lake build`
                    including all concrete Nat instances
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
recorded child-scope premise and applies the cited complementary facts there.
Non-main evidence may come from that child and its ancestors: the renderer
checks namespace ancestry and identical ordered binder/premise prefixes, then
applies each fact only to its own prefix. The child must add exactly the one
unbound assumption whose negation is concluded; sibling evidence is rejected. Every `premise element` row asserts that
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
only the cited compact OR, splits every branch using the definition's
left-associated grouping, and applies the matching cited branch result.
The convergence parent can be `main` or an established scope. The certificate
metadata in `certificate.py`, `_schema3_action_metadata`, requires the cited OR
at that exact parent and every child result under that same compact OR and
parent. Lean replay retains the parent's quantified variables, premises and
witness guards before supplying each branch assumption. A parent-bound variable
appearing in the child compact is not quantified a second time. A refuted case
may instead cite its negation from the parent or an established ancestor.
Exactly one dependency is required per definition case; missing, extra,
duplicated or mismatched case evidence is rejected. When a branch is a compact existence proposition, the proof
keeps that compact assumption, decodes its witness through the shared
negated-universal proof-support theorem, and passes the exact recorded
witness-bundle component to the scoped result.
The `or branch assumption` and `or branch proven` rows describe OR integration.
Their certificate actions, `or_integration_assume` and `or_introduce`, retain
the selected disjunct, exact parent and branch namespace, and the complements
of every other disjunct. The OR integration goal is a template, never an
assumed fact. The child context carries the parent layers followed by those
peer complements; an assumption row returns its matching peer premise. Lean
introduction supposes the compact OR false, extracts the negation of each
disjunct, applies the recorded child proof to the peer complements, and
contradicts the selected disjunct's negation. A negative peer is complemented
using classical double-negation elimination. Branch-local witnesses cannot
escape the discharge: the exporter obtains each such witness from its exact
recorded projector under the branch assumptions, then supplies its bundle to
the guarded branch result. This handles both main and nested parents without
adding any proof search.
Theorem reformulation constructs the checked set or binary-relation output,
proves its actual defining membership clauses, and then applies the single cited
source theorem in its premise order to that output and the recorded remaining
premises. The anchor is taken from the theorem, so Gauss and FTA use the same
constructive replay, including an existence with no remaining premise.
Two membership directions can name distinct compiled connective aliases.
`certificate.py`, `_membership_body_key`, compares their instantiated `and`/`or`
bodies while preserving connective grouping, argument positions, order and
polarity. `_defined_output_totality` and `lean.py`, `_defined_output_proof_term`,
both check that structure. Exact aliases use definition unfolding. When the
reverse body differs by positive binary conjunction swaps,
`membership_transport.py`, `membership_transport_plan`, records a transport
tree in `reverse_membership_transport`. Ordered OR cases retain their positions;
each conjunction projects both source children once and rebuilds them in target
order. Grouping, signs, fixed/repeated arguments and opaque binder expressions
remain exact. Negation and implication are not traversed for transport.
`lean.py`, `_membership_transport_term`, renders explicit projections,
constructors and OR elimination after the renderer recomputes and compares the
plan. The witness still uses the forward body. This covers exact aliases in
`addToSet` and the endpoint-conjunction swap in `appendToSequence`; it changes
neither the native prover nor Lean's kernel rules. See
[D-398](../40_decisions.md#d-398).
The theorem-level FTA OR elimination obtains the domain membership needed by its
OR parent from the recorded multiplicative preorder witness and AnchorFTA's
multiplication output closure, splits the resulting compact OR, and applies the
two cited branch theorems directly.
Equality1 rows name the cited source and each cited equality, eliminate those
equalities in certificate order with Lean's equality recursor, and return the
rewritten source. The exporter first asserts that source and target normalize to
the same MPL expression under exactly those equality classes.
When a cited row projects a scoped witness, equality replay selects that exact
conjunct from the shared witness guard. It never treats the existential statement
as the cited atomic fact or chooses a second witness. At `validity name`, a
guarded scoped result is discharged by obtaining its witnesses from the recorded
projectors in prerequisite order and applying the result to those same bundles.
An already bound witness cannot be replaced with a newly chosen one. The compact
implication therefore retains no unproved witness premises.
Disintegration follows one of three visible certificate shapes: a named
conjunction projection, a witness decoded once from GL's negated-universal
existence encoding and shared by its component rows, or a direct implication
constructed from the cited negated conjunction. Scoped forms first instantiate
the cited compound under the recorded scope prefix and use the same operations.
For a compact implication, the projection reconstructs every premise of its
instantiated chain without simplifying them: a reflexive equality must remain
that equality when constructing the cited conjunction, rather than become `True`.
`proof_export/lean.py` / `_negated_conjunction_projection` also replays the
flat exclusion chains of named conjunctions with more than two members. It
introduces every recorded premise and reconstructs the exact binary source
tree with one counterexample in the excluded position. Each premise must match
one source component literally, with its original polarity; missing or extra
premises and local binders assert. It does not search for a different proof.

One-witness negated implication expansions use the same shared-witness projection mechanism. `proof_export/lean.py` / `_counterexample_witness_data` collects the full implication chain, including a binder introduced after an earlier premise, and forms one conjunction of its premises and negated head. The renderer explicitly proves the witness exists by denying that conjunction, reconstructing the cited universal and contradicting its negation. All projections reuse that one witness; multiple quantified witnesses assert rather than being translated independently.

Negative OR expansions unfold the cited compact and explicitly rewrite `not_or` and `Classical.not_not` to obtain the conjunction of complemented disjuncts. The following disintegration rows use ordinary literal conjunction projections. The renderer uses only the cited fact, including when its proposition carries a scope prefix.

FTA shares the full run's definition registry with Peano and Gauss. Its module
imports Gauss, which imports Peano. Every same-list dependency is resolved by
its exact source index and certificate hash. Supplied conjectures do not become
Lean parameters; only their proved graph rows enter the exported theorem corpus.

## Ordinary-type boundary

The exported GL object theory has one carrier `α`. `GLSet α` is `α → Prop`, a
binary relation is `α → α → Prop`, and a ternary relation is
`α → α → α → Prop`. Every compiled definition and theorem is polymorphic over
that carrier. No GL term indexes a Lean type, and no GL value is represented by
a Lean type family.

Lean's kernel is itself based on dependent type theory; using Lean cannot change
that foundation. The architectural constraint here concerns the encoded GL
theory, which stays in an ordinary many-sorted predicate fragment.

`=` is typed per occurrence (`load_type_environment`, `infer_expression_variables`,
`_definition_variable_types`): it joins its two arguments, and a class only `=`
joins is an element (I-267). Lean's `Eq` works at every
type, so a set equality renders as `A = B` over `GLSet α`. A set- or
relation-typed `=` integration goal expands to its two subset implications by
extensionality, not by definition: the expansion row closes with
`GLExport.setEqIffSubsets` (or the binary / ternary relation lemma), the
`reformulation for integration and` row concludes with its `.mpr`, and the
certificate never turns an `=` expansion into a compiled override of `=`.
The three lemmas in `ProofSupport.lean` use `funext` and `propext`.

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
the recorded negation. Compilation compaction unfolds the checked compact definition around its cited
theorem, or around the expansion row it cites when that expansion was proved earlier in the same
chapter (a `local_step` dependency, no theorem reference; first met in F3's proof, where an
implication grouping is folded into `implication110`): `simpa only [<compact definition>] using
row_<n>`. A fully quantified anchor introduces both its structure and that
structure's `RelationalInduction` premise, then applies the cited theorem using
those inputs. The caller's induction proof is never generalized to unrelated
operations or domains. This applies to nullary selected-theorem carriers in
Peano, Gauss and FTA alike; they cannot use a reference already specialized to
the caller's structure.

The eight Gauss `reformulated statement` rows use GL's well-defined-output
contract constructively rather than adding an existence axiom. For `limitSet`,
the renderer supplies the set comprehension
`fun x => V x ∧ preorder N add x n`; for `limitSequence`, it supplies the
validated binary-relation lambda. It unfolds the non-recursive compiled
definition, proves the defining predicate, applies the cited source theorem to
that concrete output, and folds the two facts into the recorded GL existence
compact. These witnesses are ordinary predicate values, never dependent types.

FTA singleton reformulation follows the same contract. `EnumerationSet1`
supplies `fun x => x = zero`; the existing binary-relation route supports
characterizations such as `fun x y => x = zero ∧ y = a`. The renderer's
`_defined_output_proof_term` rechecks the certified forward and reverse
membership clauses, emits exact projections and conjunction constructors, and
preserves the compiled definition's first-use binder order and left-associated
outer conjunction. `_render_reformulated_theorem` matches each remaining
premise and the constructed definition to exactly one source premise, then
applies that cited theorem. Zero, one or several remaining premises use the
same checked route; source and result anchors must agree. No corpus-specific
existence axiom or theorem-specific proof search is introduced.

Each induction theorem's typing, zero, and induction-condition chapters become
three private helper theorems. The public theorem states its induction predicate
explicitly and composes those helpers through the ordinary proposition premise
`relationalInduction`. All three corpora use the same fixed-structure premise. The complete
project discharges it on Nat; abstract replay alone is insufficient acceptance.

A zero or successor helper may record the exact opposite of its terminal
property conclusion as a main-scope contradiction assumption. The induction
wrapper introduces that assumption only for the helper that requires it,
passes it explicitly, and discharges the contradiction. Positive conclusions
use `Classical.byContradiction`; negative conclusions introduce their positive
counterpart directly. An existing property premise takes precedence, and any
other unstated assumption still asserts. `proof_export/lean.py` /
`_render_induction_theorem` implements this composition; the regression cases
in `tests/test_lean_induction_reductio.py` cover both polarities, independent
base/successor requirements, existing premises, and rejected extra assumptions.

### Nested cases and shared witnesses

A case split may be inside a contradiction scope. Its template retains every
ancestor implication layer before its own branch premise. Facts from an
ancestor are applied only to that ancestor's prefix. Witness bundles are shared
across the child scope; a witness already bound by the context is not bound or
passed a second time. Contradiction discharge removes only the immediate child
assumption and obtains child-local witnesses inside that assumption. It never
exports a child witness or drops an ancestor premise.

Binary disjunction expansions use their actual compiled definitions and the
explicit equivalence with a negated conjunction. Both projection directions
and both conclusion polarities are supported. OR elimination validates the
disjunct structure instead of relying on a registry number such as `or3`.

Statement-only imports that match several equivalent proved copies pin the
lowest prior source index. Certificate validation then resolves the citation
against exactly the declared imports and selected local proofs; unselected
sources and multiple authorized matches still assert. Both original public
proof declarations remain in the preceding corpus.

Live selections also record `citation_source_indices`: one selected source per
alpha-equivalent proposition, chosen by lowest source index within the block.
`certificate.py` / `_citation_source_indices` validates uniqueness, membership
and complete proposition coverage. Every selected proof is still exported;
only citation resolution uses this explicit subset. Without the field, all
selected sources remain authorized and duplicate matches still assert.
`_ordered_selected_theorems` includes selected-theorem dependencies of
`compilation` rows before chapter emission, including schemas 1 and 2. Local
compilations without a selected theorem match retain local producer resolution.
Dependency cycles and ambiguous authorized imports remain errors.

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
- The FTA writer asserts its actual full-run theorem and row totals, complete
 same-list dependency closure, and isolated definition heads. Its live tests
 check the Peano/Gauss certificate hashes and actual cited proof declarations,
 and reject assumed external theorem parameters in the connected module.
- `ExplicitRowReplayTests` scans all three generated theories of the single full
 export (skipped when no export is on disk) and rejects
 any remaining `grind`. Its per-action tests require the explicit proof shape;
 the inequality-symmetry test, for example, requires every such row to use
 `Eq.symm` directly. The corpus render tests also require explicit induction
 typing, zero, successor, hypothesis, and final-composition applications.
 `ScopeEntryTests` covers scope entry, per-dependency prefixes,
 and witness sharing directly, including the negative case where a cited
 guarded fact names a witness no cited projection establishes.
- `tests.test_proof_export_lean` checks the ordinary-type encoding,
 deterministic definition and theorem rendering, all corpus counts, schema-3
 scope metadata, constructive output totality, FTA external-theorem adaptations,
 OR and contradiction scopes, row-total manifests, explicit cycle exclusions,
 and trust-shortcut rejection.
- One `lake build` checks the connected Peano, Gauss, and FTA modules and every
 generated Nat instance. The Nat writer rejects external theorem-list
 dependencies. A regression proves that double successor does not satisfy the
 induction principle, and the model uses only Lean's ordinary trusted axioms.
- Adding a proof action requires a neutral certificate contract, renderer path,
 direct negative and positive test coverage, manifest coverage, and a successful
 kernel build when the module belongs to the default Lean target.

## Live export inside the run

`run_modes.py` / `_finish_full_graphs` invokes `export_main_graph` after all
three batches' processed graph is ready and before main HTML rendering.
`theorem_list_blocks` identifies the Peano, Gauss, and FTA ranges.
`same_list_imports` finds the precise cross-range citations and pins a canonical
prior proof for repeated equivalent statements; each later certificate hashes
its earlier dependencies. `canonical_citation_sources` records the corresponding
explicit choices within each block without excluding any public proof.
`write_lean` renders the modules,
then `write_nat_model` emits `NatModel.lean` and `NatResults.lean` and includes
the latter in the Lake root. `kernel_check` runs the one full build with an
explicit child ELAN_HOME and records its exit status; a non-zero status raises
`RuntimeError` explicitly, so `python -O` cannot remove the gate.

The output is `files/full_proof_graph/lean_export/`: three selections,
certificates under `certificates/<kind>/`, generated modules and manifests,
concrete model and result modules, and `kernel_check.log`. The declaration
counts are derived from this run; all public results must receive concrete
instances. The development shortcut calls `skip_export` instead. The exporter tests read
this one full project; they do not require a shortcut export. Tests that
kernel-check generated Lean files take `lean` from `tests/lean_toolchain.py`
(beside the Lake that `find_lake` returns, run inside the tracked `lean_export/`
project for its toolchain pin) and skip without a toolchain; their scratch
files go to the system temporary directory. A test-only
`GL_TEST_LEAN_EXPORT` override supports captured complete graphs during repair.

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

**Additional explicit replay shapes.** The FTA corpus exercises:
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
copies into the connected export. The release package is
`files/full_proof_graph/lean_export/`, including its Nat model and instances.

`release_public.py` / `check_live_lean_exports` requires the connected modules,
a green `kernel_check.log`, and a root that imports NatResults. It reruns the
single build and the contract tests; the release then publishes the package in
its proof archive (`build_proofs_zip`) and checks that the archive carries every
result file byte for byte (`lean_export_zip_issues`,
[D-456](../40_decisions.md#d-456)). Old
shortcut Lean artifacts are excluded even if present on disk. The private
snapshot overlays the same connected result package. Neither package includes
the host-local `.lake/` cache.

### Flat integration construction

`reformulation for integration flattened` maps to `definition_construct`. The certificate validates the one same-scope integration-goal template, recursively enumerates permitted definition unfoldings, and records an exact match to the row's premises with distinct bound witnesses. `proof_export/flat_integration.py` and `render_flat_integration_term` supply explicit conjunction/existence constructors using only those recorded premises. `proof_export/lean.py` enters the recorded scope and introduces the row's binders and premises before applying that term. No proof search, additional theorem premise or admission shortcut is used.


### Binary OR expansion polarity

Binary OR definition expansion uses only the named definition, `GLExport.orIffNotAndNot`, and `Classical.not_not`. The last equivalence cancels the doubled negation introduced when a disjunct is already negative, matching the recorded MPL De Morgan expansion in all four polarity combinations. It is an explicit logical equivalence, not proof search or an extra theorem premise. The generated-row kernel regression covers positive/positive, positive/negative, negative/positive and negative/negative leaves.

## Weaknesses

- **Constructive reformulation coverage.** The checked definition must have one set or binary-relation output, an acyclic definition closure, matching forward membership clauses and exactly one reverse clause. Remaining source premises must match the recorded facts uniquely; malformed, missing or ambiguous matches assert. Ternary-relation outputs are outside the current totality contract.


### Known & tracked

- **Binary-only OR convergence — corrected.** `lean.py` /
 `_render_chapter_steps` replays every definition case with a left-associated
 case pattern and exactly one matching dependency per case. Direct renderer
 and kernel tests cover two through five cases, and malformed coverage is
 rejected. 

- **Every export input is a gitignored run artifact.** Both
 `files/processed_proof_graph/` and `files/shortcut/processed_proof_graph/`
 match `.gitignore`'s `**/processed_proof_graph/`, and so do the exports
 themselves. A certificate is reproducible only on a host whose matching
 pipeline run is still on disk; the recorded `theorem_list_sha256` makes that
 dependence loud instead of silent. Nothing tracked pins a graph generation
 any more: each run derives its own selection and the release publishes the
 run's export in its proof archive.
- **Abstract induction requires a model witness.** Passing row replay alone
 does not establish consistency of the hypotheses. Every connected public
 result must compile at Nat, with all anchor and induction inputs supplied.
- **Historical external snapshots are conditional fixtures.** They cannot
 replace same-run proved dependencies or enter the connected Nat export.
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

### Explicit scope inheritance

The `zoom in` tag becomes a `scope_inherit` action. `_chapter_certificate` requires one exact local parent step with identical expression and a strict ancestor scope. `_scope_inheritance_proof` introduces the child context and applies the cited parent fact to its own parameters, assumptions, and witness guards; existential bundles are retained. There is no source-origin copying, new axiom, or proof search. Both endpoint scopes remain explicit in the processed graph and typed certificate.

`lean.py`, `_render_chapter_steps`, retains free parameters from the complete
recorded ancestor chain even when a child goal no longer mentions a parameter
of the parent goal. Parent-scoped facts still consume those binders. Local
variable-copy reflexivity introduces the row's ordinary binders, scope premises
and witness guards before `rfl`; variable-copy certification is unchanged.

## Immutable expression caches

`proof_export/mpl.py` and `parse_mpl` share immutable parsed trees through a
bounded cache keyed by exact source text. `alpha_key` memoizes the canonical
bound-variable spelling of immutable expression values. Misses and evictions
recompute the same strict functions; parsing errors still fail. Source spellings,
free-variable identities, certificate bytes, and emitted Lean declarations are
unchanged.

## Implication-chain composition

Structural OR introductions retain the `expansion` tag and use the certificate action `structural_or_introduce`. `proof_export/or_introduction.py`, `match_or_introduction`, checks the cited compact implication's actual OR premise, complete ordered conjuncts, fixed arguments and scoped binders. `_render_chapter_steps` constructs the checked conjunction tree and follows the exact OR injection path. Ordinary expansion remains a separate action; an introduction must not override the compact implication's definition.

An `implication composition` row becomes the certificate action `implication_compose`. Its two dependencies retain exact source expressions and scopes. `build_implication_composition_plan` verifies a bijection of both complete source binder sets through the intermediate expression onto the result binder, rejects bound-to-fixed specialization and collapsed binders, and returns ordered source argument lists and the binder group at each premise. The producer and result have the same complete ordered premise list under the checked renaming, and the consumer has exactly one premise. `_render_chapter_steps` enters the recorded scope, specializes each cited proof at its own visible context, introduces every composed binder group and premise in order, then applies every producer premise followed by the unary consumer. No proof search or additional assumption is used; admission demand controls construction eligibility rather than supplying a logical proof premise.

## Grouping and contraposition certificates

`proof_export/contraposition.py`, `build_grouping_plan` and `build_contraposition_plan`, validate the exact cited unary sources, complete binder substitutions, fixed parameters, signs and scope joins. The `implication_group` certificate action carries a checked conjunction-construction tree; `implication_contrapose` carries the original source's argument substitution. The Lean renderer revalidates these plans and emits explicit source application, conjunction construction and contradiction. Both tags retain ordinary dependency closure and origin checking.

### Equality-supported OR convergence

The `or_eliminate` action optionally carries `or_equality_plan`, produced by `match_equality_convergence`. Each case identifies its actual source and equality dependencies, or its retired-case refutation. The renderer recomputes the plan, introduces the parent binders and case assumption in order, specializes each cited fact at its true scope, and eliminates the equalities inside that case. It never assumes a noncanonical target fact or promotes a branch equality to the parent. The default exact-expression action is unchanged.


## Contradiction evidence from nested scopes

`proof_export/lean.py`, `_render_chapter_steps`, selects the deepest cited non-main context and requires every other cited context to be either that context or an ancestor with an identical binder/premise prefix. Namespace ancestry and structural prefix equality are both required; identical templates in sibling scopes do not license combining their facts. The deepest context must extend the conclusion's context by exactly one binder-free assumption, equal to the expression being negated. Each dependency and shared-witness projector still receives its own original context arguments. Only the child assumption is discharged; parent assumptions and witness guards remain in scope.

`tests/test_lean_nested_contradiction.py` covers both source polarities, dependency orders, direct and deeper ancestors, same-scope evidence, and Lean kernel replay. Negative cases retain rejection of siblings, incompatible prefixes, the wrong discharge premise, skipped layers and bound child assumptions. Native inference, provenance and certificate scope records are unchanged.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
