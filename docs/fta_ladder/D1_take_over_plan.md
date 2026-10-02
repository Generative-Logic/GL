<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# D1 property transfer

## Current contract and state

- **Objective:** transfer only empty retained short subkeys and the integration typing-key prohibition from onto.
- **Base:** completed main tip; branch created from its clean primary Windows checkout on 2026-09-07. The compiled-name promotion and runtime restoration remain part of the base.
- **Approval:** the maintainer approved the complete plan, including the two architecture changes, branch creation, full GPU baseline and changed runs, independent determinism run, documentation, complete milestone commits and pushes.
- **Acceptance:** one- and two-premise normalized subkeys contain one nine-byte empty OwnerSet with no loose flag, signatures or owners; at most one staged item per key per window; rule removal and ownerless cleanup retain short keys until hash-memory reset. Longer keys keep existing ownership. Integration preparation never creates a key whose extracted predicate is exactly `in`, in either polarity; typing facts and instruction contents remain usable.
- **Hard boundaries:** no other D1 behavior, conjecture/configuration change, capacity increase, compiler change, processor pipeline, assertion weakening, or hashburst-dump modification. Any required architecture deviation returns to the maintainer.
- **Current state:** COMPLETE AND LANDED. The validated implementation is on main through exact-tree squash; sandbox history remains. Both requested properties, all baseline theorems and deterministic artifacts are preserved. The main registries name the contracts D-350 / D-351 and I-227 / I-228.
- **Next action:** none; this ledger is frozen at the completed main landing.
- **Blockers:** none.

## Milestones

1. Preserve a complete full baseline with source/executable hashes, dedicated logs and artifact manifests.
2. Implement both properties with direct tests and synchronized design documentation; full rebuild and unit gate; commit and push the complete verified snapshot.
3. Run the full GPU pipeline, compare theorem sets and complete artifacts against the baseline, and require no lost baseline theorem or verifier/export failure.
4. Independently rebuild and repeat the final full GPU run; require identical deterministic artifact paths, byte lengths and SHA-256 hashes; record results and commit/push the completed ledger.

## Validation contract

Direct tests cover both short lengths; repeated direct and 40,000 staged installs; repeated flushes; empty bytes and presence acceptance; all rule-index instances; retention through removal; deletion of ownerless long keys; removed rules remaining unable to fire; CUDA projection/growth; positive and negated typing exclusion; `in2`/`in3` controls; repeated integration preparation; equivalence re-keying; and typing usability.

Every build is native Windows MSBuild `-t:Rebuild` launched through Python with `env=dict(os.environ)` and CUDA 13.3. Pipelines use `main.py --GPU`. Each asynchronous command has one completion-event watcher and dedicated logs, without progress polling. Artifact manifests cover binaries, raw/processed/full proof graphs, theorem pools, simple facts and incubator output; generated Lean build caches and timing-bearing `kernel_check.log` are retained as evidence or excluded explicitly from deterministic comparison. Full-input archive contents are compared by member hashes, independent of ZIP metadata.

## Evidence

- Source donor: `prover.hpp` and `ensureEmptySubkeyRecord`, `stageEmptySubkeyRecord`, `flushSubkeySection`, `removeRuleSubkeyEntry`, `eraseOwnerlessEntries`, `prepareIntegrationCore2`; install sites in `memory.cpp`.
- Current processor and CUDA growth already accept short subkeys by presence; signature readers apply only from three premises onward.
- Baseline : rebuild 112.099 seconds; C++ 1,669/1,669 and verifier 407/407; full GPU pipeline 323.714 seconds internally (323.927 seconds process wall); 151,482 proof checks, zero failures; connected Lean kernel passed (Peano 64, Gauss 30, FTA 79 exported rows).
- Baseline manifest retains 5,553 files / 793,050,213 bytes under ; source commit, source manifest, executable hash, full-input archive members, timing and diagnostic evidence are preserved in the parent folder.
- The first filesystem completion watcher exceeded the tool transport's 300-second lifetime; the independent process completion record confirmed success. Subsequent build gates use one process-exit waiter with logs redirected and no progress polling.
- Changed gate : full rebuild 120.170 seconds, C++ 1,672/1,672 passed in 9.148 seconds. The existing production-install staging comparison checks all four hash-memory instances; the CUDA bulk-growth fixture uses empty short records; rule removal is followed by a zero-request generation check. Both new integration tests passed, including positive/negated typing, main/descendant scopes, ordinary fact registration, non-typing controls, cache exclusion, repeated preparation, instruction preservation and equivalence re-keying.
- Static scope review: no production GPU, capacity, compiler, Python pipeline, definition or configuration file changed. The two added user-SwDD SVGs parse as valid XML. `git diff --check` passed.
- Changed full run : all 1,672 C++ tests and 407 verifier tests passed; 151,482 proof checks, zero failures; connected Lean passed with the same exported theorem rows. Pipeline runtime 314.169 seconds internally, 314.358 seconds process wall, versus baseline 323.714 / 323.927 seconds. This single matched observation is not a general speedup claim.
- Comparison : all 5,553 artifacts / 793,050,213 bytes match by path, length and SHA-256; no missing, added or changed files; full-input archive members also identical. All six theorem/global-list sets match, with zero lost or gained rows. The independent baseline also matched the inherited main artifacts exactly.
- Independent gate : fresh full rebuild 111.559 seconds and all 1,672 C++ tests passed in 8.640 seconds. The following full-run capture references this separate build/unit record and uses the same capture logic and artifact roots as the preceding runs.
- Independent full run : 314.642 seconds internally, 314.829 seconds process wall; 1,672/1,672 C++ and 407/407 verifier unit tests; 151,482 proof checks (25,533 main and 125,949 incubator), zero failures; connected Lean kernel passed with unchanged Peano 64 / Gauss 30 / FTA 79 export rows.
- Strict comparison : identical source manifests (`a620dec902fd9268154367fa519eac73e07f285bcb842b947d9ce49bf43991cb`); all 5,553 artifact paths, lengths and SHA-256 hashes identical; full-input archive members identical; theorem sets and verifier counts identical. The comparison rehashed both preserved trees and asserted all acceptance conditions. Together with the baseline comparison this establishes exact preservation across all three complete runs.
- Final outcome: no production change beyond the two approved properties, no capacity or corpus change, no new diagnostic traps, and no outstanding verifier, export, determinism or implementation failure. Durable contracts are in the agent SwDD and the user mechanisms chapter; no MPU topology or sizing number changed.

## Matched batch runtime

Seconds from each log's `Runtime prover` and `Runtime of the executable (counter example filter + prover)` lines. The latter includes native filtering and export overhead; neither column includes Python orchestration or separate conjecture generation. Before is main; after is implementation. These are individual full GPU runs with identical complete proof artifacts, not averaged benchmarks. Exact numbers are retained in .

| Batch | Prover before | Prover after | Change | Executable before | Executable after |
|---|---:|---:|---:|---:|---:|
| IncubatorPeano1 | 41.72 | 43.38 | +4.0% | 43.08 | 44.90 |
| IncubatorPeano2 | 5.80 | 6.78 | +17.0% | 6.10 | 7.06 |
| Peano | 16.32 | 17.03 | +4.3% | 28.08 | 29.57 |
| IncubatorGauss1 | 20.48 | 19.37 | -5.5% | 30.64 | 27.81 |
| IncubatorGauss2 | 6.21 | 6.29 | +1.2% | 8.69 | 8.40 |
| IncubatorGauss3 | 21.13 | 20.77 | -1.7% | 21.22 | 20.87 |
| Gauss | 19.62 | 9.12 | -53.5% | 30.41 | 20.11 |
| FTA | 48.19 | 47.39 | -1.7% | 48.75 | 47.89 |

Total prover: 179.478 → 170.114 seconds (-5.2%). Full pipeline: 323.714 → 314.169 seconds (-2.9%). The largest observed saving is the Gauss prover; shorter incubator measurements include regressions, so the aggregate does not imply that every batch accelerated.

## Three-run runtime comparison

The maintainer requested all three completed full GPU measurements. Values below are the same `Runtime prover` lines as above; complete pipeline time is separate. Exact values are retained in .

| Batch | Main baseline | After changes | Independent rerun |
|---|---:|---:|---:|
| IncubatorPeano1 | 41.72 | 43.38 | 42.08 |
| IncubatorPeano2 | 5.80 | 6.78 | 5.84 |
| Peano | 16.32 | 17.03 | 17.21 |
| IncubatorGauss1 | 20.48 | 19.37 | 19.56 |
| IncubatorGauss2 | 6.21 | 6.29 | 6.19 |
| IncubatorGauss3 | 21.13 | 20.77 | 20.78 |
| Gauss | 19.62 | 9.12 | 9.06 |
| FTA | 48.19 | 47.39 | 47.55 |
| Total prover | 179.48 | 170.11 | 168.27 |
| Full pipeline | 323.71 | 314.17 | 314.64 |

## Main landing

- **Authorization:** the maintainer requested the squash playbook from the completed sandbox to main on 2026-09-07, then requested per-batch timing for all three runs.
- **Starting references:** clean source, destination; fetched source/destination remote references match, and main is an ancestor of the sandbox.
- **Accepted procedure:** preserve sandbox history locally and remotely; create a one-parent squash from its complete tree; verify exact tree identity before advancing main; push; switch the primary checkout to main; run the canonical SwDD renumber tool and commit that documentation-only change separately.
- **Pre-landing documentation sweep:** corrected the remaining ownership description in the user substrate table so empty short-key storage and retention agree with the already-updated mechanisms diagrams. No production code or configuration changed after the three accepted full runs.
- **Squash result:** has exactly one parent, the original main tip. Its tree exactly matches source. Main was advanced and pushed, and the primary checkout switched to main. Source and origin sandbox references retain the full original history.
- **Separate renumber:** the canonical tool maps the empty-short-subkey decision to D-350 and invariant to I-228; the typing-key decision to D-351 and invariant to I-227. No live pending identifiers remain. Its only production-file edits are three comment references in `prover.hpp`; an exact replacement comparison verified that all executable code and tests still match the validated source. No new build/run was necessary for those documentation-only changes.
- **Final landing state:** complete; commit and push this separate renumber/evidence snapshot, preserving the sandbox branch. No blockers or further implementation work.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
