<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Connected FTA export and induction correction

**Status: complete; landed on main.** The independent clean GPU rerun reproduces
all 5,552 retained files byte-for-byte and passes every verifier and kernel
check. The verified sandbox tree landed as squash, with the sandbox
history preserved. The separate main-only documentation step assigns D-346.
No implementation, verification, or landing work remains.

## Objective and acceptance

Keep the development shortcut. A single full Windows GPU run proves Peano,
Gauss, then FTA, and checks one connected Lean project. Existing batches have
empty external conjecture lists and retain generation. FTA has a nonempty
external conjecture list and does not run the conjecturer. External conjectures
are proof obligations, never premises. FTA uses the earlier batches' proved
theorems. Lean uses induction for the fixed structure, actual theorem imports,
and a checked shared natural-number model that discharges every public theorem's
mathematical assumptions. Explicit certificate row replay remains required.

## State and authorization

- Branch:, created from `main`.
- The user later authorized suitable GPU boundary adjustments after the full-run
 event-capacity failure ("Adjust boundaries suitably").
- The user approved the design and implementation, explicitly retained shortcut,
 and confirmed the new branch name and base. The later instruction authorizes
 landing on main after a successful determinism rerun.
- Initial tree has no modified tracked files.
- At task start, the FTA export had 79 public theorems and 124 universal induction
 parameters. Original change resolved four global compilation rows
 by strengthening assumptions; landed it. Those assumptions admit
 no intended natural-number instantiation. Successful compilation is therefore
 insufficient acceptance evidence.
- Do not strengthen hypotheses to close failed rows, introduce proof search,
 change verifier semantics, or discard prior work. Report any required change
 outside the approved mathematical and pipeline contract.

## Execution

1. Add explicit external conjecture input and the full FTA batch, preserve the
 shortcut, test routing and inputs, and commit the coherent pipeline change.
2. Connect the full-run FTA certificate to Peano/Gauss proofs and correct induction
 and compiled theorem scopes with direct positive and negative coverage.
3. Add the natural-number model and generated theorem instantiations; check the
 complete Lean project and its assumptions, then commit the coherent export.
4. Rebuild natively, run the relevant unit gates and one full GPU pipeline,
 resolve failures within scope, inspect theorem coverage and proof artifacts,
 update the design documents, and commit and push the verified result.

## Final outcome

The normal clean full pipeline completes Peano, Gauss, and FTA, then builds one
connected Lean project including every concrete Nat instance. No diagnostic
index reconstruction is needed. The development shortcut remains independent,
and the release/test workflow requires only the full export.

- Full run: , exit 0, 551.399 seconds.
- Verifier: 25,968 main checks plus 125,951 incubator checks; 151,919 total,
 zero failures.
- Lean: one successful 10-job Lake build; 64 Peano, 30 Gauss, and 79 FTA
 declarations, 7,646 replayed rows, and all 173 concrete Nat instances.
- All 65 supplied FTA conjectures match exported statements. The Peano, Gauss,
 and FTA statement multisets match the baseline in registry-independent form.
- All three fresh certificates are byte-identical to the validated captured
 graph's certificates. The fresh theorem pool has 126 correctly paired rows.
- Final axiom audit: every Nat result uses only Classical.choice, Quot.sound,
 and propext ().
- Final tests: 105 Python tests pass without skips on the normal full export;
 1,656 native unit tests passed after the final complete native rebuild.
- Artifacts: `files/full_proof_graph/lean_export/`, including
 `GLExport/NatModel.lean`, `GLExport/NatResults.lean`, all three certificates,
 and `kernel_check.log`. Structured audit: .

Implementation acceptance is complete. The subsequently authorized determinism
rerun and conditional main landing are tracked below.

## Determinism and landing

- Reference: the complete successful fourth full GPU run. Preserve generated
 main and incubator proof graphs, theorem pools, GL binaries, simple facts,
 HTML, and connected Lean sources/certificates before the clean rerun.
- Compare the complete retained file set, byte lengths, and SHA-256 hashes.
 Lake build cache and kernel execution logs are execution artifacts; preserve
 the logs separately and verify their successful results.
- Preserve locally and remotely. Create a one-parent
 squash whose tree exactly equals the verified sandbox, then renumber any
 pending SwDD entries separately on main. Finish on clean, pushed main.

- The required landing review found stale conditional-FTA prose, totals, and
 diagrams in the user overview/provenance pages and release notes, plus stale
 release status messages. Updated them to the actual three-batch full chain,
 one kernel build, and all-theorem Nat instantiation. No proof or pipeline
 semantics changed during this documentation correction. All 47 SVG diagrams
 in the six reviewed HTML pages parse, the pages contain no private memory
 or commit references, and both release preflight tests pass.

- Independent acceptance rerun: ,
 exit 0, 598.626 seconds. A fresh complete native rebuild and
 all 1,656 native unit tests passed before it.
- Exact comparison: all 5,552 paths and all 794,627,847 bytes match the successful
 fourth run by length and SHA-256; zero changed, missing, or added files.
 Coverage: 9 GL binaries, 469 raw main graph files, 287 processed main graph
 files, 370 main HTML/Lean files, 12 theorem-pool files, 3 simple-fact files,
 and 4,402 incubator artifacts. Build cache and kernel execution logs are
 excluded from byte equality; separate retained kernel logs both pass.
- All 81 verifier tag reports match. The rerun repeats 25,968 main checks plus
 125,951 incubator checks, totaling 151,919 with zero failures.
- The rerun's 10-job Lake build passes. All 173 Nat instances remain in the
 default target; the full 105-test Python regression suite passes without skips.
- Evidence: , its `before/`
 and `after/` snapshots and manifests, and `postcheck.json`. All theorem,
 certificate, Lean source, and manifest bytes match the accepted reference.

- Main landing: one-parent squash, parent, exactly the
 tree; verified tree hash
. Main was pushed, and the primary
 checkout switched to main without changing the reviewed files. Local and
 remote sandbox refs retain the complete source history.
- The separate main-only renumber assigns D-346 and updates the navigation
 high-water mark. The only changes after the exact-tree squash are that
 numbering and this final task-ledger outcome; production source and configs
 retain the verified sandbox bytes. Landing evidence is preserved in
 .

## Earlier integration evidence (superseded by final acceptance)

- Native full rebuild passed; 1,656/1,656 native unit tests passed.
- Python input, renderer, model, and export suite: 89 tests passed.
- The concrete Nat model for all three actual anchor definitions compiles with
 the pinned toolchain (). The proof
 uses Nat.rec for induction. The first probe exposed a simplifier rewrite loop;
 the forward correction fixes rewrite orientation and proves successor
 injectivity explicitly, without increasing limits or strengthening premises.
- Original sources, certificates, and processed graphs are preserved under
 .
- The full-run FTA connection and all-theorem Nat instantiations are implemented
 but have not yet passed the combined pipeline. Next: complete the direct
 regression gates, commit the coherent implementation, then run the full GPU
 pipeline and resolve its concrete integration failures.

- The double-successor rejection fixture compiles. The shared Nat induction
 theorem has no axioms; all three concrete anchor proofs use only propext and
 Quot.sound ().

- First full GPU integration run ended after 350.999 seconds. Peano and Gauss
 completed; FTA loaded 65 supplied conjectures and reached burst 16 before
 `requestIndex < capacity.rawRequests` asserted. The old GPU profile selector
 equated AnchorFTA with shortcut mode, so this full batch received the
 262,144-request shortcut tuple instead of the existing 524,288-request
 full-run tuple. Forward fix: pass the actual run mode into the analyzer and
 select its existing profile. No capacity value or assertion is changed.
 Evidence: ; `prover.cpp` / `proveKernel`;
 `phase2_projection.hpp` / `kFtaShortcutPhase2GrowthCapacity` and
 `kFullRunPhase2GrowthCapacity`.
- The saved 79-theorem FTA replay compiles with corrected induction scopes
 (). It remains conditional on its
 historical induction-conditioned external statements and is not claimed as
 the connected all-theorem Nat acceptance result.

- Added and passed direct routing/release regression tests: the full run schedules
 Peano, Gauss, FTA in order; release preflight needs no shortcut export and
 refuses a Lake root that omits concrete Nat results. Seven tests in that
 focused group pass ().

- The actual-mode profile correction passed a second full native rebuild and
 1,656/1,656 native unit tests. The expanded Python regression gate now passes
 92 tests. The existing capacity tuples are unchanged. The next full GPU run
 uses .

- Second full GPU run ended after 373.460 seconds at FTA burst 16, now on
 `eventIndex < capacity.acceptedEvents` with the correct full profile. Both
 existing profiles capped accepted events at 4,194,304. No full connected
 certificate was generated. Evidence: .
- The user explicitly authorized suitable boundary adjustments. The next
 measured run reserves 8,388,608 full-run accepted events and matching ordering
 entries. Shortcut capacities remain unchanged. The added event ledger and
 ordering columns require 400 MiB of device memory plus library sorting/scan
 workspace. Existing per-pass diagnostics now include raw requests alongside
 accepted events and unique requests so the final sizing can be checked.

- The enlarged full-run event/ordering reservation passed the third complete
 native rebuild and all 1,656 native unit tests. The test checks the agreed
 event ceiling and its exact ordering-capacity match. No GPU census trap or
 proof-semantic change was introduced; raw request count is an additional
 field of the existing permanent per-pass diagnostics.

- Third full GPU run completed FTA: 5,214,880 accepted events, 447,252 raw/unique
 requests, and a 2,888,479-node frontier. The fixed full GPU allocation is
 7,630,515,844 bytes. No further capacity adjustment is needed for this corpus.
- Finalization exposed the missing paired compiled roots on the uncompressed
 full FTA save path: 126 theorem rows but only the 62 preceding compiled roots.
 The unchanged pruner therefore retained only Peano/Gauss. The native writer
 now appends both columns for full FTA, with output assertions. A worker's
 legacy stdout encoding also masked the primary failure; its child environment
 now explicitly writes UTF-8, matching the parent reader.
- Preserved the complete native run under .
 In that diagnostic copy only, reconstructed the 64 missing compiled roots
 from the actual saved proof statements (including checked OR base-form matches).
 Reprocessing retains 173 declarations: 64 Peano, 30 Gauss, 79 FTA.
- Connected import resolution now pins a canonical source for duplicate equivalent
 Peano statements (28/29 and 61/62). Unselected imports and multiple explicitly
 authorized matches remain errors and have direct negative coverage.
- Full FTA source 147 exercises an OR case within a contradiction scope, with
 parent and child witnesses. Replay now retains ancestor premises, specializes
 ancestor facts at their own prefixes, preserves witness identity, and discharges
 only the child assumption. Binary negated-conjunction projection supports both
 sides and polarities; OR handling uses compiled structure rather than fixed ids.
- The single connected Lake build succeeds for all 173 abstract declarations and
 all 173 concrete Nat instances ().
 The exporter tests now use the full export alone. All 105 Python tests pass
 without skips against that captured project ().
- Fourth complete native rebuild and 1,656/1,656 native units passed. A final
 fresh full pipeline run is outstanding; the diagnostic reconstruction is not
 claimed as normal-pipeline completion.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
