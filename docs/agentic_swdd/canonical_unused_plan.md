<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Canonical unused encoded expressions

## Execution contract

- Objective: canonical-only storage in every encoded-expression list for rows whose signed core occurs in no installed premise and whose unsigned core has never been assigned as a goal in that logical block; prefilter request-ineligible rows before device transfer.
- Approval: user approved the proposed plan, per-logical-block eligibility, signed premise cores, unsigned goal cores, and encoded lists (overall, local, external, delta and peers).
- Base: main;.
- Landing approval: user requested the squash playbook to main, preserving the sandbox history. Reviewed source tip is; local and remote main both remain.
- Preserve non-list equivalence processing, historical known facts, proof provenance, scope boundaries, and assertions. Refresh eligibility after rule/goal changes; reprocess newly eligible rows independently of class waterlines.
- Simultaneous canonicalization uses completed class state and original row arguments, updates indexes and metadata, and repairs waterlines after compaction.
- Sorting: remove only pre-ordisMerge sorts proven unnecessary for observable results and repeatability; history caps and contradiction emission require focused coverage.
- Acceptance: direct unit tests; native full rebuild; shortcut and full CUDA pipeline verification, theorem-content comparisons, valid proof/Lean artifacts; measured rows, transfer bytes and timings. The originally planned processor comparisons have completed; the user's final policy reserves future processor runs for release preparation.
- Current state: complete and frozen. The exact source tree is squashed onto main as; its invariant is now I-225. Implementation is committed and pushed as; all 1,654 unit tests pass. Full runs pass Lean and 142,755 proof checks with zero failures, and all 1,661 captured theorem/processed-proof files match byte for byte between backends. Final shortcut runs pass 9,192 checks with zero failures and byte-identical backend artifacts. Full theorem sets and the shortcut theorem file match their respective fresh pre-list baselines.
- Next action: none; implementation, validation, and the requested squash landing are complete.
- Blockers: none. The user approved retaining assigned goal cores for the logical block lifetime.

## Initial source evidence

- Device projection currently copies the complete overall encoded list; split tasks share that slice.
- Equivalence products commit before a decoded-name sorted ordisMerge sweep. The consumer also emits capped contradiction history, so set idempotence alone does not justify deleting the sort.

## Validation evidence

- Baseline native rebuild: , success.
- Baseline CUDA shortcut: , 9,217 checks, zero failures, 65.58330 seconds. Manifest, theorem list and hashburst preserved in .
- Prefilter native rebuild: , success; , 1,642/1,642 unit tests passed.
- Direct projection tests cover polarity, retained order, removal, later singleton whole-key installation, and a retained count smaller than source capacity. The normalized-key device-probe fixture now supplies the singleton premises that production rule installation supplies.
- CUDA pipeline: , stops at the full-upload cardinality assertion in projectionWorker before completing the first warm-up hashburst. No assertion or temporary diagnostic trap was changed.
- Projection milestone is coherent after the remap gate passed; commit it before the canonical-list changes.

- User decision: retain existing equality, negated-equality and anchor shape exclusions; replace full-upload assertion with exact filtered-row validation.
- The next CUDA gate exposed unconverted processor stump ordinals at `phase2_cuda.cu` and `requestGatesPass`: host stumps address the unfiltered registry. Added a host-only source-ordinal column and exact binary-search translation before task upload; device bounds assertions remain unchanged. Direct tests cover a removed middle row and shard concatenation.

- Artifact correction: the initial baseline manifest selected full-run paths, not shortcut paths, and is not evidence of shortcut equality. The correct CUDA snapshot is ; the unchanged processor path is running as its semantic comparison in .

- Projection milestone committed and pushed. Processor shortcut: 227.37873 seconds, 9,217 checks, zero failures; exact comparison against  found zero theorem/processed-proof differences.
- Canonical-list implementation is in progress: eligibility snapshots, simultaneous replacement, preserved historical facts, collapsed-row restoration, and class-waterline overrides. Metadata uses standard cold-set key facets in deload tags 822..825. New focused tests include ordering-sensitive capped OR contradiction origins and metadata roundtrip/discharge.

- Canonical-list native full rebuild  passes;  reports 1,648/1,648 tests passed. The existing compact-rule regression proved that replacement must follow non-list hooks; the implementation now does so. The OR-uniqueness test fixture now fully encodes its manually installed goal, matching the production goal door.
- Sorting decision: retain the pre-ordisMerge sorts. `canonical_unused.ordis_deposit_order_selects_the_capped_contradiction_origin` proves that reversing the same p/q deposits selects a different retained contradiction witness at origin cap one. Sorting is observable, not redundant set bookkeeping.
- Prepared a git archive of the verified prefilter milestone under  for fresh full-run comparison; no working-branch files or history are changed by the archive.

- CUDA shortcut with canonical lists: , 59.82494 seconds overall, 39.8066 seconds prover, 9,192 checks with zero failures. Prefilter-only comparison: 75.68760 seconds overall, 51.8228 seconds prover. This pair shows a 23.19% prover reduction; no generalized speed claim from one pair. The theorem file is byte-identical (64 rows, none added or missing). Proof checks decrease because proof histories change.
- Transfer telemetry over this shortcut: 1,599,650 source statement rows, 374,795 uploaded rows (76.57% excluded), 431,148,960 encoded-statement bytes avoided across projections; projection time totals 0.8762554 seconds. Values are in .

- Processor shortcut with canonical lists: , 189.71151 seconds overall, 9,192 checks and zero failures. All captured theorem and processed-proof files are byte-identical to .

- Fresh full baseline from the archived pre-list milestone: , 267.76874 seconds overall, 141,136 checks with zero failures, Lean kernel check passed. Main and incubator artifacts and all theorem files are preserved under . The working branch was not switched; only the executable build artifact was selected for this run, with baseline data directories linked to the same workspace files.

- Extended lifecycle assertions pass in the 1,648-test gate: later goal eligibility reactivates collapsed rows after premise removal, and subtree wipe removes suppressed identities. The final canonical-list sweep now uses the same decoded statement order for OR deposits as the existing equivalence drain, consistent with the capped-origin counterexample. A fresh rebuild is running before final full validation.

- First changed full run stopped in Peano counterexample filtering at the new source-history assertion (). Source establishes the cause: the status-4 fact loader writes fact lists and known/levels metadata without origins; `cloneFactsTemplate` deliberately leaves origins empty. The canonicalizer now records bridges only outside `ceFilteringActive`, retaining the ordinary-proof assertion. Added a direct historyless-counterexample regression. No diagnostic trap or existing assertion was removed.
- Source review also found the new admission canonicalization had preceded raw ancestor-known refusal. Restored that refusal order and re-check canonical targets against strict ancestors; a direct regression covers an inherited raw fact rewritten under a child class.

- The canonical-target axed-variable regression failed before the target check (), establishing that raw admission checks alone were insufficient after moving canonicalization behind raw ancestor refusal. The target check is now restored, and restoration also applies iteration, secondary-variable and axed-variable filters. A complete rebuild succeeded; the combined unit/full-CUDA gate is running with completion notification.

- Corrected full CUDA run passed Peano but stopped in IncubatorGauss3 at `PagedVector<int>::operator[]`. The saved dump  resolves the stack through `closeRuleIndexStaging` -> `flushRemArgsSection` -> `BlobCsrValueStore::rebuildRunsInPlace` -> its carried-run mover.  proves run starts 155 and 162 equal 354 while the blob column size is 354: an empty trailing range was read as a blob.
- Fix preserves the pre-growth blob and byte endpoints and uses the byte endpoint for an empty trailing range. Existing bounds assertions remain; a direct regression covers both a no-growth rebuild and a growing rebuild with two empty trailing keys. No temporary traps were added.

- Full run after the endpoint fix: ; every C++ batch completes and all theorem sets equal the baseline. Lean export rejects empty `files/processed_proof_graph/132_check_zero.txt`; raw chapters 281 and 306 are empty base cases for the two orderings of the Peano nonzero-successor existence theorem. Diagnostic snapshot: .
- Goal lifetime conflict: `prover.cpp` explicitly seeds the theorem head as a goal in recursion block #2. `refreshEncodedCoreEligibility` currently forgets its core after the pending row disappears. `visualizer.cpp` and `checkZeroStack` select the base-case triad through runtime encoded membership (or its discharge snapshot), so collapsing a settled head can hide its original spelling from export. No export/verifier assertion has been changed. A goal-core lifetime decision has been requested; no dependent change is applied yet.

- User approval: retain every assigned unsigned goal core after its pending goal is proved, until block discharge. The goal-core set now accumulates rather than being cleared at refresh. A direct regression covers goal settlement, subsequent class application, and preservation of the original head in the discharge snapshot used by export. Export and verifier logic remain unchanged.

- Goal-lifetime rebuild and 1,653 unit tests passed. The next full run exposed a timing-dependent construction-shard overflow in the first IncubatorPeano1 burst, before goal lifetime changes affect that burst. The dump `gl_quick_abort_47464.dmp` and  show shard zero holding 345 of 512 blocks and 16,358 metadata bytes against a 16,384-byte reservation.
- The previous dynamic distribution assumed two average shares per worker without bounding actual assignments. Construction now uses consecutive waves whose summed unshared usage fits one existing shard in every column, so even a single worker claiming all work is safe. Waves append into the same original GPU chunk and canonical descriptor order. No capacity ceiling changes. The direct wave-planner test covers metadata overflow, another limiting column, complete coverage, and empty input.

- Full CUDA gate with goal lifetime and bounded construction waves succeeded: . Results and theorem comparisons are in ; source/transfer measurements are in `transfer_metrics.json`. No theorem additions or losses versus the fresh baseline.

- Validated implementation committed and pushed as. The remaining runs use this unchanged executable/source: full processor comparison and the final shortcut pair. Full CUDA result: 142,755 checks, zero failures, Lean passed, 272.59237 seconds; no theorem-set changes versus baseline.

- Full processor comparison succeeded: , 512.65402 seconds overall, 142,755 checks with zero failures, Lean passed.  records zero differences across 1,661 captured files per backend. Neither backend adds or loses a main or incubator theorem versus the fresh pre-list baseline.
- Full CUDA performance is 272.59237 seconds overall and 130.77052 seconds summed prover time. The fresh pre-list baseline is 267.76874 seconds overall: this full run is 1.80% slower overall. The earlier 23.19% prover improvement belongs to the intermediate shortcut comparison, not the final full run. Final full projection uploads 4,526,697 of 40,849,329 source rows, with 1,593,397,344 encoded-statement bytes uploaded, 600 construction waves, and 21.2768219 seconds summed projection time.

- Final shortcut comparison succeeded on the unchanged implementation:  reports 60.27769 seconds overall and 40.2036 seconds prover time;  reports 192.37841 seconds overall and 176.768 seconds prover time. Both pass 9,192 checks with zero failures.  records zero artifact differences and a byte-identical theorem file versus the pre-list shortcut baseline, with no theorem additions or losses. Final shortcut prover time is 22.42% below the pre-list GPU measurement of 51.8228 seconds.
- Final outcome: canonical-only unused encoded lists, simultaneous class canonicalization, later-eligibility restoration, and pre-upload signed-core filtering are implemented. Existing shape exclusions and non-list equivalence hooks remain. Sorting remains because the direct capped-origin regression proves that changing deposit order changes retained proof provenance. Full and shortcut verification is complete; no temporary diagnostic traps remain.
- Final user policy: development pipeline runs use CUDA only; processor runs are reserved for release preparation. The last already-started comparison completed normally, without cancellation. This ledger is frozen after recording the completed evidence.

## Squash landing

- Topology verified after fetching both branches: main is an ancestor of, and each branch equals its remote counterpart. The source has five commits beyond main; no main-side change needs integration.
- The required user-documentation sweep checked the equivalence mechanism, GPU projection, and MPU descriptions against the source diff. The canonical-storage diagram now includes the shape exclusions and uses the existing behavior label for eligible/excluded rows; its previous variants-only label was inaccurate for equality and anchor exclusions. Deposit wording now places canonicalization after raw admission checks. No production source changed during landing preparation.
- Acceptance uses the already completed native rebuild, 1,654 unit tests, full GPU pipeline and shortcut GPU pipeline for the identical production tree. Exact tree/parent verification and pending-invariant renumbering are the remaining landing gates; no new CPU run is part of this development operation.

- Final source snapshot:, preserved locally and on origin as. Squash has sole parent and the identical source tree. The primary checkout is on main; the squash is pushed.
- Main renumbering: the dry run assigned I-225 to the canonical-unused invariant; the real pass updates its declaration, quick-reference link, and invariant high-water. Renumbering and this landing record form a separate documentation-only commit. Production source remains byte-identical to the validated implementation; no new pipeline run was needed. The user-documentation sweep parsed the three reviewed HTML pages and all 27 embedded SVG diagrams.
- Landing also preserves the completed timing-opportunity notes and the corrected one-page induction-export description that arrived in the shared checkout during preparation. No source-branch history was rewritten or removed. This ledger is frozen at the completed landing.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
