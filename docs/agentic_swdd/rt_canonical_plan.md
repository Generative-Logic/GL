<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Dormant expression runtime

## Execution contract

- Objective: remove recurring dormant canonicalization and collapsed-row restoration by retaining unused encoded rows until their cores become used.
- Branch:, created from locally and remotely verified main.
- Acceptance: unchanged baseline theorem sets, successful verifier and connected Lean checks, two repeatable full and two repeatable shortcut GPU runs, and measured removal of both sweeps without disguising their work in another recurring scan.
- Approved design: preserve canonical admission and original-spelling history; no new known or levels entry for a rejected original. Premise cores remain signed. Assigned goals, admitted negative facts, and contradiction seeds retain unsigned core usage for both polarities until block discharge, including after scope wipe. Generalize the existing goal-core set, remove suppression metadata, keep unused rows through class cleanup, and reuse ordinary newly-eligible equivalence processing and contradiction checks.
- Constraints: existing shape exclusions, assertions, capacities, admission/rejection semantics, GPU premise upload filtering, static memory discipline and sacred hashburst dump remain. No separate opposing-pair canonicalizer. Main landing is now explicitly authorized without builds or reruns.
- Current measured state: the prior diagnosis attributes 14.83 worker seconds to dormant canonicalization and 2.53 to restoration (8,669,400 registry rows scanned for three restorations). These are historical attribution, not a matched speedup claim for this branch.
- Current state: landed on main as squash, with the complete tree of preserved sandbox snapshot. Documentation promotes the pending decision to D-361. Code remains byte-identical to accepted snapshot and the prior accepted runs.
- Next action: none. Landing is complete; this ledger is frozen with the code-identity evidence. No builds or reruns were performed for the landing.
- Blockers: none.

## Milestones

1. Capture the verified main baseline and preserve its full-input shortcut archive.
2. Implement retained core usage and removal of the two sweeps with direct tests and synchronized design documentation; commit and push the coherent verified change.
3. Complete matched performance and full/shortcut proof acceptance, account for differences, record the evidence and commit/push the final ledger.

## Evidence

- 2026-09-08: the primary working tree was clean before branch creation. The remote main reference exactly matched the local main tip. The earlier planning session read the complete SwDD navigation and invariant quick-reference through I-233; this implementation follows that approved design.
- Baseline: native rebuild 112.809 seconds; full GPU pipeline 327.075 seconds; shortcut GPU pipeline 65.532 seconds. Both pass 1,696 C++ and 423 verifier unit tests; the full export passes Lean. Snapshots preserve 5,565 full artifacts and 967 shortcut artifacts, byte-identical to the pre-run snapshots. Completion record: ; direct run logs:  and .
- Baseline full-input archive SHA-256: `d44de61836c0271559edcb566a16889a01d3f9a290c57ab0766f2b0e5966323e`; archive copies accompany each baseline snapshot. The main-tip sacred dump target produces an empty dump in both modes; the target and infrastructure were unchanged.
- Registration audit covers ordinary statement admission, direct equivalence products, negative equality and its mirror, fact loading, contradiction-block and contradiction-scope seeding, and fact-template cloning. Activation cleanup uses only scopes of older rows actually revisited below class progress markers; class-delta scopes retain the ordinary cleanup order.
- Unit milestone:  passes after 112.325 seconds;  passes 1,702/1,702 tests in 9.311 seconds. Coverage includes original-spelling nonregistration, dormant list/levels retention, all activation sources, negative admission refusal, deeper-scope contradiction, sibling isolation, scope-wipe usage retention, cloning and deload/discharge. No temporary diagnostics were introduced. Agent and user design documents and the MPU storage description were updated together.
- First changed runs (`rt_canonical_changed_01`): rebuild passes; shortcut 63.073 seconds process wall / 62.897 pipeline wall / 48.409 prover seconds; full 322.972 process wall / 322.745 pipeline wall. Shortcut retains exactly 127 theorems; full retains exactly 127 main and 1,460 incubator theorem rows. There are no added or missing artifacts, compiled-definition differences, or simple-fact differences. All proof checks pass (26,862 shortcut; 151,204 full); full Lean passes.
- First proof differences: shortcut 9 files (2 raw, 2 processed, 5 generated HTML); full 558 files (184 raw, 170 processed, 192 HTML, 12 Lean/export artifacts). Complete filename, theorem-set, per-tag and timing comparisons are in  and the corresponding full comparison. These are qualification runs, not the final repeatability pair: the first shortcut uses the baseline full-input archive, while later shortcuts must use the changed full-input archive consistently.
- First matched FTA attribution: full equivalence work 34.51 -> 20.18 worker seconds; dormant canonicalization 13.63 -> 0; ordinary rewrite closure 1.88 -> 3.20, scan 0.89 -> 1.16 and drain 0.48 -> 0.92 worker seconds. The broader usage policy increases ordinary batch rounds from 3,772/4,363 to 3,842/4,415 in Phase 1/3. Reactivation-only cleanup runs 23/3 times and rounds below 0.01 seconds per phase. Restoration has no remaining production function or call site. Phase wall time, rather than worker totals, explains the smaller pipeline improvement.
- Documentation QA parses both changed HTML documents and all 26 embedded SVG diagrams; a stale compact-carrier sentence was corrected in the existing owning chapter. The cross-scope regression checks whether encountering a child delta first can hide the source scope's older class waterline during activation cleanup.
- Cross-scope regression (`rt_canonical_activation_edge_01`): full rebuild passes, then `dormant_usage.child_delta_does_not_hide_reactivated_ancestor_cleanup` fails at the assertion that the downprioritized source left the local list. The class-token loop stopped on the first due class, so a child delta could prevent recording the older source-scope waterline. The fix keeps collecting applicable scopes only for newly used cores; ordinary used rows retain their early exit and unused rows still skip class processing. No existing assertion was removed or weakened.
- First proof audit: all 354 changed full proof files and all four changed shortcut proof files retain identical heads and scopes. The two shortcut chapters select different valid contradiction witnesses (successor of zero and an additive-zero identity); full changes predominantly remove equality and implication history. Generated HTML and Lean/export differences follow those proof changes; verifier and kernel checks validate the resulting derivations. Audit: .
- First registry/request workload: full source rows presented to projection 42,429,039 -> 43,780,728; uploaded rows 5,019,522 -> 5,021,435. Shortcut source rows 1,656,726 -> 2,104,648; uploaded rows 494,518 -> 494,707. These are cumulative visits, not unique resident rows or peak memory. Shortcut raw requests increase 1,307,070 -> 1,322,446. Complete counters: .
- Correction gate (`rt_canonical_activation_fix_01`): rebuild passes in 109.913 seconds; all 1,703 unit tests pass in 9.534 seconds. The new regression failed before the correction and passes afterward. No diagnostic instrumentation was introduced. The preserved baseline and first changed full-input archives have 392 members each; twenty prior raw proof chapters and the archive manifest differ. Final shortcut repeatability therefore compares runs produced from the final full histories, rather than the earlier baseline-history shortcut.
- First final pair (`rt_canonical_final_01`, corrected source, executable SHA-256 `20389c21588b08c146e762210cfc694a0c2c2cba7bf86e7f0ae32225c1d47d14`): full pipeline 300.100 seconds, 151,204 checks with zero failures and successful Lean; shortcut 59.291 seconds, 26,841 checks with zero failures. Every theorem set remains identical to baseline. Full artifacts match the first qualification run exactly (all 5,565), establishing that the cleanup correction does not alter this workload's outputs. Shortcut artifacts now include the changed prior proof history: 45 baseline differences (22 raw, 10 processed, 13 HTML), with mathematical inputs still identical.

## Final acceptance

- Both final pairs run the same corrected executable and identical source/configuration manifests. The current runtime sources match every recorded source hash. All four pipeline invocations pass 1,703 native and 423 verifier unit tests. Each full run passes 151,204 proof checks and connected Lean certification; each shortcut passes 26,841 checks. No failures.
- Repeatability: all 5,565 full artifacts match across final runs, as do all 967 shortcut artifacts. Manifest-file SHA-256 values are respectively `53d9daa57bdf2eba1d2bb1deb9c779a12be2eeeac2dbadc6d83b97046dfa4f7e` and `60ccda9d2e503032795ae35265b26badd9903f30496dcb56bcaa9107080c54c5`. Per-tag verifier results and all recorded GPU logical-work counters also match.
- Both full-input archives contain the same 392 members with identical member hashes; ZIP container bytes differ. Archive member hashes are preserved in . Full and shortcut agree on 786 common theorem/definition/proof output files. The two differing post-run scratch roles, `conjectures.txt` and `externally_provided_theorems.txt`, differ between modes in the baseline as well and remain individually unchanged from baseline.
- Baseline theorem sets remain exact: 127 main/shortcut rows and 1,460 incubator rows, with no additions or losses. No compiled-definition, simple-fact or theorem-channel artifact changed. Baseline proof differences are fully inventoried: 558 full artifacts and 45 shortcut artifacts, confined to proof histories and their generated HTML/Lean outputs. Every changed raw/processed proof retains its head and scope; every resulting processed proof is checked.  contains the per-file tag changes, timings, request work and acceptance assertions.

| Measurement | Main baseline | Final run 1 | Final run 2 |
|---|---:|---:|---:|
| Full pipeline wall seconds | 326.85963 | 300.10031 | 306.03958 |
| Shortcut pipeline wall seconds | 65.36199 | 59.29054 | 63.71460 |
| Shortcut prover wall seconds | 47.8111 | 45.1607 | 49.1745 |
| Full FTA equivalence worker seconds | 34.51 | 20.47 | 21.53 |
| Shortcut FTA equivalence worker seconds | 34.31 | 21.52 | 21.51 |
| Shortcut dormant canonicalization worker seconds | 13.57 | 0 | 0 |
| Shortcut ordinary rewrite closure worker seconds | 1.84 | 3.45 | 3.43 |

- The two targeted sweeps are eliminated: dormant canonicalization makes zero calls, and restoration has no function, call site or suppression metadata. Reactivation cleanup is rare (23 Phase-1 and three Phase-3 scopes in FTA) and rounds below 0.01 worker seconds per phase. Ordinary rewrite closure, scanning and draining grow under broader core usage; the net shortcut equivalence cost falls about 37.3 percent. These are rounded exclusive worker-time sums, not predicted wall-time savings.
- Retaining rows and broader usage increase ordinary work. In the matched shortcut, cumulative source-row visits rise from 1,656,726 to 2,104,648, uploaded-row visits from 494,518 to 494,707, and raw requests from 1,307,070 to 1,322,446. Both final runs reproduce those counts. These are visit counts, not unique rows or a peak-resident-memory measurement. Negative/seed activation is not isolated as a separate counterfactual; the figures report the complete approved policy.
- The shortcut prover samples straddle baseline, so they do not establish a consistent shortcut-prover speedup. Both measured full pipeline runs and both whole-shortcut runs are faster than the single baseline, but no broader statistical claim is made. Assertions and all capacity settings remain unchanged; no temporary production diagnostics were added. The sacred dump remains unchanged and empty for its pre-existing main-tip target.
- Evidence is directly named under `.debug`: `run_rt_canonical_final_01_full.log`, `run_rt_canonical_final_02_full.log`, `run_rt_canonical_final_01_shortcut.log`, `run_rt_canonical_final_02_shortcut.log`; the corresponding completion, comparison and runtime files accompany the preserved snapshots. Agent/user design documents and MPU storage descriptions are synchronized; all 26 SVG diagrams parse successfully.

## Main landing — 2026-09-08

- Authorization: squash onto `main`, control byte identity of the code, and perform no reruns. The existing acceptance above remains the run evidence.
- Accepted source:, tree. Local and remote source tips match. Main and origin/main both start; it is an ancestor of the source.
- Preflight identity: all 260 recorded runtime source/configuration SHA-256 values still match the accepted run. The 363 tracked Git entries outside `docs/` are captured for exact mode/type/blob comparison after landing and renumbering. Evidence: .
- Documentation sweep: the user mechanisms chapter and diagram describe retained unused rows, all four usage sources, block-wide polarity activation and ordinary cleanup after revival; the MPU storage description agrees. No user-facing architecture correction is needed for this landing. Only this ledger changes before the squash.
- Result: squash has sole parent and tree, exactly matching preserved source. The preparation differs from accepted snapshot only in this landing ledger. Main was pushed and the primary checkout switched to main without a reset; the sandbox branch remains local and remote.
- Renumbering: the dry run assigned the pending dormant-core decision to D-361 from main's D-360 high-water. The real pass changes the decision heading, its owning-chapter reference and navigation high-water only; this final landing ledger accompanies that separate documentation commit.
- Identity control: the squash has full-tree identity to its source. After renumbering, every changed path relative to the accepted source is a Markdown document under `docs/`; all 363 non-documentation Git entries and all 260 recorded runtime/configuration SHA-256 values remain identical. Final machine-readable evidence is , with the exact-tree squash record in . No code change, build, unit-test run, prover run or Lean rerun was performed for this landing.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
