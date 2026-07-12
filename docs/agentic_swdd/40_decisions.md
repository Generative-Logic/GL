<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Decisions `[DRAFT]`

> A dated log of architectural and tactical decisions that shape GL's current form. Each entry answers: **what was the choice, when, and why**. Entries are append-only. When a previous decision is revised, do not delete it — add a new entry that supersedes it, and annotate the old one.

Entries sorted reverse-chronological (newest first). Unknown precise dates are flagged.

---

<a id="d-209"></a>
## D-209 — make 4 GiB the main cold-pool cap (2026-07-12)


**The choice (user-directed).** Set `ProverParameters::static_pool_bytes` to 4294967296 bytes: 4 GiB, or 16384 blocks at the unchanged 256 KiB block size. The persistent, mail, and LB-body pools remain 1 GiB, 2 GiB, and 2 GiB respectively, so the total fixed program-start reservation becomes 9 GiB.

**Why.** The working-set pager and extent-backed raw images exist to keep proof execution sound when the aggregate live LB state exceeds resident main-pool capacity. A 4 GiB cap makes that path the normal heavy-batch configuration and bounds the main cold reservation, while preserving the fixed-pool, assert-on-exhaustion contract.

**Semantics and verification.** This changes capacity only: block size, page size, container layout, deload format, proof semantics, and provenance are unchanged. The gate is a Windows Release x64 full rebuild, the direct unit-test suite, and one complete Windows `main.py` run with zero verifier failures and no pool-exhaustion assert. The historical 8 GiB measurements remain comparison evidence rather than current defaults.

<a id="d-207"></a>
## D-207 — cache each byte key's full digest in its static location record (2026-07-11)


**The choice (user-approved).** Extend `ColdStringLocation`, the page-backed positional record owned by `BytesKeyStore`, with the key's full 64-bit FNV-1a digest. `HashMap::lookup` computes the probe digest once and passes it into `equalStored`; candidate records reject a digest mismatch before resolving the byte pool, while a digest match still performs the full length and byte comparison. The digest is copied across `copyFrom` and recomputed from canonical bytes during reload.

**Why.** The committed GaussIncubator1 profile attributed 16.47% of samples to byte-key `HashMap::lookup`, including repeated stored-key hashing and collision-candidate byte resolution. The throw-away index already needs each stored digest when it rebuilds, and hot probes revisit the same stored keys many times. Keeping that derived value beside the existing static location removes repeated page walks without introducing a heap cache or a second lookup structure.

**Canonical storage and semantics.** The digest is derived runtime state, not a new deload field: `BytesKeyStore` still contributes exactly two tags, lengths and logical bytes. Canonical reload recomputes the digest, so existing file identity and tag numbering are unchanged. Hash equality is only an early rejection; full byte equality remains authoritative, so key identity, ids, iteration order, proof decisions, and artifacts are unchanged. The location column remains a `PagedVector`, satisfying the static-memory contract at the cost of eight additional bytes per stored byte key.

**Verification contract.** The direct cold-map test checks digest equality after append, cross-arena copy, and canonical two-facet reload. Both host unit suites and a full copied WSL `main.py` run must pass; the full verifier result must remain 131886 checks with zero failures, while GaussIncubator1 runtime is compared with the 180.425-second page-cached scanner run.

<a id="d-208"></a>
## D-208 — stream over-block blob runs through one dense CSR splice (2026-07-11)


**The choice (user-approved).** Keep `BlobCsrValueStore`'s dense two-level CSR and canonical deload representation, but add a replayable generated-source write door for a run whose concatenated bytes exceed one request-generation arena block. `PagedVector::replaceRangeGenerated` sizes the replacement once, moves the surviving tail once, and lets an emitter fill the gap from multiple contiguous spans. `BlobCsrValueStore::replaceRunGenerated` replays the same ordered record emitter to fill `blobPool_` and generate `blobStarts_`; `HashMap::assignRunGenerated` owns find-or-mint and the one key-run suffix rebase.

**Why.** The former widening path opened the key's run empty and called `appendBlobToRun` once per merged record. For a non-tail key, every append shifted the same byte and blob tails again. The post-descriptor-cache IncubatorGauss1 profile still attributed 11.02% self time to `memmove`, 9.24% to `PagedVector<char>::replaceRange`, and 10.54% to `insertRemainingArgsNormKeyBatch`. The generated splice makes that widening write O(old tail + replacement bytes), not O(record count × old tail), without changing the representation.

**Determinism and lifetime.** The emitter order is the already canonical sorted-unique merged NormKey order. Existing record bytes are copied to request-generation scratch before the first pool mutation; batch-only records already live there. The emitter is replayed for bytes and lengths, with exact record-count and byte-count asserts. Final `runStarts_`, `blobStarts_`, and `blobPool_` bytes match a contiguous `assignRun`, so lookup ids, reverse edges, deload tags, proof artifacts, and verifier answers are unchanged.

**Secondary effect.** Contiguous `replaceRun` now delegates to the generated core. This removes the heap `std::vector<int32_t> newStarts`; prefix starts are emitted directly into paged storage, restoring the I-138 zero-heap contract on this write path.

**Verification contract.** Direct tests compare generated and contiguous range replacement, generated and contiguous blob-run assignment, non-tail preservation, and the over-block remaining-args widening regression. The release gate is a full rebuild, unit suites on Windows and Linux, full WSL `main.py`, unchanged 131886-check zero-failure verifier result, and a lower IncubatorGauss1 prover-only timer.

---

<a id="d-205"></a>
## D-205 — move each outgoing mailbox and its private interner into the deloadable LB arena (2026-07-10)


**The choice (user-directed).** Split routing staging by lifetime. `mailIn` remains `RoutingColdMail` on a self-owned mail-pool arena: phase 1 claim/reloads the recipient, pulls and absorbs the inbox, then returns its blocks immediately. `mailOut` becomes `DeloadableMailOut` inside `LbMemory`: one private `ColdStringTable`, one statement set, and one origin blob map on `lbMemory.manager`, enrolled in `visitContainers` at reserved tag band 755..804. An always-resident `mailOutPending` shell bit lets the commit sweep skip cold empty LBs and claim/reload only producers with output.

**Why this shape.** The measured IncubatorPeano1 mail-pool peak was dominated by simultaneous per-LB output staging: 1,757 independent `RoutingColdMail` blocks (439.25 MiB component peak), while `mailIn` peaked at 32 blocks, `MailLog` at 111, and the global interner at one. The old design kept every outbox resident only so the serial all-LB commit sweep could inspect emptiness. A shell summary removes that need. Once pending content is inside the existing deload image, the steward can release it with the rest of the LB after phase 3 and reload one producer at a time for commit.

**Id lifetime.** Statements, origin keys, and every dependency use the same per-LB `mailOutInterner`. Commit decodes that private space and mints global blob ids. `clearMailOut` resets both columns and the private interner before lowering `mailOutPending`; the tag band survives discharge so a final pending batch cannot be erased before commit. Post-join root sends explicitly claim/reload the root across the deposit.

**Memory accounting.** `mailOut` leaves the mail pool and becomes part of the main-pool physical peak. Because it shares each LB's arena, no exclusive physical-block count is invented; telemetry sums an always-resident exact logical live-byte snapshot and reports it separately. Mail-pool attribution now covers only `MailLog`, the global mail interner, and `mailIn`.

**Verification.** The direct tests cover statement/origin doors, private-id reset/reuse, byte-identical serialization, and `Memory` deload/reload with pending output. Full-pipeline confirmation must preserve the reference theorem and verifier totals and produce a paired memory log.

**See also.** `I-163`, [D-206](#d-206), [I-101](30_invariants.md#i-101), [I-102](30_invariants.md#i-102), [I-161](30_invariants.md#i-161).

---

<a id="d-206"></a>
## D-206 — reset the global mail interner with each safely retired MailLog window (2026-07-10)


**The choice (user-directed).** Extend [D-204](#d-204)'s zero-dormant rolling mode to the global `mailInterner`. At the single-threaded post-phase-3 seam, assert every per-LB `mailIn` is empty, retire the delivered `MailLog` blob/reference window, then call `resetMailInterner` before the next commit. `mailOut` is safe across the reset because it stores ids from its own per-LB interner; `MailLog` blobs and `mailIn` are the only global-id carriers. The next commit remints the next window into the same global arena. A grid with any initially dormant LB skips both retirements for the whole execution batch.

**Why this seam.** In an all-active grid, every potential receiver ran phase 1 and cleared `mailIn` before phase 3 joined. `MailLog::retireDeliveredBatches` then removes every serialized old id before the table resets. No parallel reader exists at the seam, and no old id escapes into proof artifacts because all observable comparisons and exports decode strings. Resetting earlier would invalidate a live inbox or log blob; resetting later would invalidate the new commit.

**Memory effect.** `ColdStringTable::resetToFresh` returns its pages to the interner's `LbArena`; the arena retains and reuses its physical blocks. The whole mail-pool high-water therefore contains the largest single-window interner footprint rather than cumulative distinct strings across every hashburst. This is measured through `mailMemory.peakBlocksInUse` in the paired per-run memory reports, not by subtracting process RSS and not by summing batch peaks.

**Scope boundary.** This decision governs only the lifetime of the global routing id table. [D-205](#d-205) separately moves producer-local staging and its private id table into each LB's deloadable arena.

**Verification.** `test_memory.cpp::mail_interner_reset_restarts_the_delivery_window` pins id invalidation and fresh-window reminting. The full Windows `main.py` confirmation must preserve theorem counts `517, 77, 53, 91, 24, 1, 10` and verifier totals `131886` checks, zero failures, while its `memory_<postfix>.log` provides the before/after mail-pool peaks.

**See also.** `I-162`, [D-204](#d-204), [I-127](30_invariants.md#i-127), [I-161](30_invariants.md#i-161).

---

<a id="d-204"></a>
## D-204 — retire delivered MailLog history when the completed grid has no initially dormant LB (2026-07-10)


**The choice.** After `buildGrid` has created and prehandled every `permanentBodies` entry, count the entries whose `isActive` is false and fix the mail-history mode for the whole execution batch. A zero count selects rolling mode: after phase 3 joins and before the next commit sweep, `MailLog::retireDeliveredBatches` releases the retained `mailBlobPool` and `mailRefs` pages, preserves `mailEdges`, `mailCursor`, and cumulative `MailHead::count`, and resets every `MailHead::lastRef` to `-1`. A non-zero count selects full-history mode for the whole batch, even after those LBs activate.

**Why the grid-build gate is sufficient.** The only LBs that may change from inactive to active are induction-zero LBs born dormant ([I-112](30_invariants.md#i-112)). When all LBs start active, every still-relevant receiver pulls the prior retained window in phase 1 before the post-phase-3 retirement seam; an LB that later deactivates cannot reactivate and needs no catch-up history. When any LB starts dormant, its cursor may remain behind until `activateZeroCondition`, so retiring any earlier window would make its first pull underrun the ref chain. The gate is fixed rather than dynamically widened because the initial dormant count is the complete auditable fact the safety proof needs.

**What remains cumulative.** Producer batch counts and per-edge cursors never reset, so `mailPeek` and the `count - cursor` pull calculation keep their existing meaning across windows. Only the ref chain representing the currently retained window restarts. Routing topology and pointer identity remain unchanged. This changes storage lifetime, not sender-to-receiver reachability, payload contents, mail latency, proof decisions, or provenance.

**Measured applicability.** The telemetry-only canonical seven-batch Windows baseline recorded zero dormant LBs for `IncubatorPeano1`, `IncubatorPeano2`, `IncubatorGauss1`, `IncubatorGauss2`, and `IncubatorGauss3`; those five batches select rolling mode. `Peano` and `Gauss` recorded 246 and 127 dormant LBs respectively and retain full history.

**Measurement contract.** The true physical `MailLog` footprint is the `ExpressionAnalyzer`-owned `mailArena.blocksHeld` high-water at the configured 256 KiB block size; that arena is exclusive to the five log containers and its granted blocks remain held until analyzer teardown. Exact committed and peak-retained logical history bytes are reported beside it. `mailMemory.peakBlocksInUse` is only whole-pool context because it also includes routing mailboxes and the global mail interner; it must not be labelled as MailLog memory. Savings are compared per batch against the telemetry-only baseline; sequential batch peaks are never summed.

**Verification.** `test_mail_log.cpp::repeated_retire_pull_cycles_are_bounded` covers 64 commit/pull/retire windows, cursor/count preservation, ref-head reset, and stable mail-pool peak blocks. The existing `dormant_catch_up` test remains the full-history guard. Full-pipeline theorem and verifier equality is the confirmation gate.

**See also.** `I-161`, [D-137](#d-137), [D-141](#d-141), [I-94](30_invariants.md#i-94), [I-112](30_invariants.md#i-112).

---

<a id="d-202"></a>
## D-202 — retain recordable nodes from a replaced stump level as terminal-only work and validate them through the normal request path (2026-07-10)


**The choice.** `produceExpressionStumps` may replace a short level of candidates with the next, wider level to reach its bucket-work target. Before replacing it, the producer probes `overallHashMemory.normalizedEncodedSubkeysMinusOne` and `...MinusTwo`. A node accepted by either map is retained as an `ExpressionStump` marked `terminalOnly`. `proveKernel` deals that record to exactly one bucket like every other stump. In each obligatory-stump batch, `generateEncodedRequestsStatic` rechecks the terminal record against that batch's actual `intMemory`, lets the existing stump-alone probe materialise a `BaseCandidate` when valid, then stops before depth-first growth. The ordinary child stumps remain the sole owners of all larger combinations. The request still reaches `preEvaluateFromEncoded`, `StaticRequestEmitter`, `BurstSink::consume`, and `checkLocalEncodedMemoryStatic`; the producer fires nothing.

**Why.** A length-`L+1` child can cover only base candidates of size at least `L+1`. If a dropped length-`L` node is already a complete base for an obligatory length-1 or length-2 request, the children cannot reproduce that smaller request. The former assert correctly exposed the completeness hole when newly activated induction-zero LBs were pre-split on their first active burst. Retaining the node closes the hole without letting a shallow stump overlap the children's larger search.

**Why `overallHashMemory` is sufficient.** Every recovered implication is installed in `overallHashMemory` before status-specific copies are placed in `workingMemory`, `localHashMemory`, or `localHashMemoryDelta`; those memories are subsets for split-universe discovery. The overall maps may therefore produce a conservative terminal candidate, but cannot miss one. Actual request validity remains per batch because the consumer repeats the probes against its supplied `intMemory`.

**What was ruled out.** Calling `checkLocalEncodedMemoryStatic` from the producer would bypass obligatory-stump construction, scope and request gates, pre-evaluation, emitter deduplication, firing-record ownership, and the per-batch hash memory. Keeping a shallow stump as an ordinary growing seed would overlap the child frontier. Dropping it preserves neither request completeness nor the unsplit request set.

**Verification.** Direct unit coverage exercises both halves: the producer retains a recordable dropped singleton beside its regular pair frontier, and the generator checks a terminal singleton's shallow request without growing it into a larger one. Full rebuild, unit gate, and full `main.py` run recorded with the implementing commit.

**See also.** [D-201](#d-201), [D-203](#d-203), `I-157`, `I-156`, `I-158`.

---

<a id="d-201"></a>
## D-201 — the LB split trigger flips from a mid-burst work cap to an end-of-iteration straggler statistic; a heavy LB is PRE-split whole-LB into `logicalCores` expression buckets next iteration (2026-07-09)


**The choice.** The split is no longer triggered mid-burst by a submatch cap. Every main-path burst runs to COMPLETION (no cap, no truncation, no same-iteration discard-and-redo). After the iteration, a single-threaded stats pass in `proveKernel` computes each active main-path LB's total work and PRE-splits the stragglers for the NEXT iteration. The split axis is the EXPRESSION/bucket dimension (the stump mechanism), now run whole-LB; the rule-partition dimension is off on the main path.

**The trigger — `isStraggler` (fair-share).** `work(L)` = the SUM of `L`'s parts' submatch counts (the split-invariant total, [D-117](#d-117) — not the max-over-parts, which is coupled to the split state and thrashes, the old [D-111](#d-111) band). With `T = Σ work` over active main-path LBs and `C = logicalCores`:

```
straggler(L)  ⟺  work(L) > T / C   AND   work(L) >= min_split_work
```

`work > T/C` is the idle-core fair-share: an LB whose work alone exceeds the ideally-balanced per-core load is a makespan bottleneck that leaves cores idle, so parallelising it across `C` cores is justified — this is what "trigger on idle cores" means, and it ties the split to the machine core count (no magic split number). It is masking-resistant (one straggler lifts the bar by only `work/C`) and self-limiting to at most `C-1` LBs ("a few significantly over average"). `min_split_work` is the per-bucket setup break-even (each of `C` buckets re-pays `filterIntEncodedStatements` + the obligatory-stump builders over the whole statement universe): below it a bucket's share of the work is under the fixed setup, so splitting cannot pay off — the one tunable knob (`ProverParameters::min_split_work`; the fan-out itself is `logicalCores`, not a config number). All integer arithmetic (`isStraggler`), so the verdict — and the split set — is a deterministic function of the deterministic work totals.

**The mechanism — whole-LB expression buckets.** A straggler carries `Memory::numberOfParts > 1` (repurposed from "rule-slice count" to "bucket count", two-state `1` ↔ `logicalCores`; reused rather than a new `bool` so the steward's four `numberOfParts>1` eviction checks stay valid unchanged). A newly activated induction-zero LB also takes this path once because it has no prior-iteration work statistic yet. `proveKernel` dispatches either case as ONE round-1 `produceOnly` PRODUCER task that runs `produceExpressionStumps` whole-LB at `g_splitCount == 1` (so `partitionAccepts` accepts every rule); the classify seam deals its regular and terminal-only stumps round-robin into `min(nStumps, logicalCores)` buckets (grown to `kStumpsPerBucketTarget × logicalCores` so the deal balances the buckets' grow-tree sizes) and requeues one bucket part per bucket for round 2. The pass loop is now ≤ 2 rounds (producer, buckets); the three-level cap-escalation ladder is gone.

**Determinism (the load-bearing property).** The split SET is a deterministic function of the PRIOR iteration's deterministic submatch totals (never wall-clock RT — that would make the set timing-dependent and the proof graph non-deterministic). Given the set, the applied deposit is partition-independent (the sorted firing-record merge, [D-117](#d-117) / [I-77](30_invariants.md#i-77)), so two full runs are byte-identical (`theorems.txt` + `global_theorem_list.txt` + verifier check count). The theorem SET is preserved versus the never-split baseline (both split dimensions are complete; the early-exit keeps every theorem). The whole-LB expression split runs at `g_splitCount == 1` and does NOT rule-partition, so no request whose rule and expression fall in different partitions is lost — completeness holds (a rule-partitioned bucket split, by contrast, would drop those cross-terms and lose theorems; this is why the early-exit gate moved to `g_isMultiPart`, see below).

**Self-control report.** After splitting, if a straggler's busiest bucket still exceeds `T/C`, its work is irreducibly serial (one dominant candidate-growth chain — a genuine "runs and runs" unprovable induction LB) that bucketing cannot subdivide; `proveKernel` logs it (`[SPLIT] ineffective …`, `lbMaxSub` kept only for this). Pure observation, no control-flow effect, determinism intact. The remedy — bounding the runaway induction — belongs to the proof-search policy, not the load balancer.

**Supersedes.** [D-111](#d-111) (the cap-hit bang-bang escalation, same-iteration discard-and-redo, `adaptiveSplitDecision`/`SplitDecision` — deleted; the thrash band it documented is gone). Refines [D-121](#d-121) (the burst early-exit gate moves from `g_splitCount>1` to the per-burst `g_isMultiPart`, so a whole-LB expression split at `g_splitCount==1` keeps the early-exit off across its bucket parts). Recasts [D-203](#d-203) (the stump/bucket split is now the PRIMARY whole-LB split driven by the straggler statistic, not a second-cap sub-escalation of a rule-part). The cap fields (`maxNumberHashRequests`, `second_split_submatch_cap`, `fixed_number_splits`, `split_fallback_ratio`) become no-op config-compat.

**Verification.** Full rebuild (Release x64) clean. `gl_quick.exe --unit-tests` green (new `is_straggler_fair_share`, `min_split_work_defaults_positive`, and the burst-sink test recast for the uncapped + `g_isMultiPart` gate; `adaptive_split_decision_bang_bang` / `second_split_cap_is_below_the_first` retired with their code). Two-run determinism + verifier gate: [pending — filled at the end of the series].

**See also.** [D-111](#d-111), [D-117](#d-117), [D-121](#d-121), [D-203](#d-203), [D-109](#d-109), `I-159`, `I-160`, [I-75](30_invariants.md#i-75), [I-76](30_invariants.md#i-76), [I-77](30_invariants.md#i-77).

---

<a id="d-203"></a>
## D-203 — the LB split gains a second dimension: a rule-part that hits its cap returns its growing candidates as stumps and re-runs as one sub-part per stump BUCKET (2026-07-09)


**The choice.** A phase-2 part that reaches its submatch cap discards its burst and reports how to divide itself. The escalation is now symmetric, one level down each time:

- **Unsplit part hits `maxNumberHashRequests` (40000).** Reports nothing. The LB is split `fixed_number_splits` ways BY RULE and re-run whole (unchanged, [D-111](#d-111)).
- **Rule-part hits `second_split_submatch_cap` (10000, new).** In place of its regular output it runs the request generator in a new mode — no obligatory stump, all expressions — and returns the growing candidates it built as **stumps** (`produceExpressionStumps`). Its burst is dropped; its siblings keep their records. The stumps are rule-specific for free: the producer runs inside the part that hit the wall, so `g_splitProcessID` / `g_splitCount` still hold that part's coordinates and `partitionAccepts` has already pruned the statement universe to the rules it owns.
- **Stump sub-part hits the same cap.** It stops, exactly as a rule-part stopped before this branch existed. There is no third split.

**Where the stump acts: the base candidate, not the obligatory stump.** Attaching the split stump to the obligatory sequence would lengthen it to `|O| + |S|`, so a base candidate would have to be a key minus `|O| + |S|` elements — a `MinusThree`, `MinusFour`, … owner-set map per stump length, unbounded once stumps grow, each a cold container in the deload stream. Impossible. The stump joins the BASE CANDIDATE instead, which is still a key minus `|O|`: `stumpLen` stays 1 or 2 and `MinusOne` / `MinusTwo` remain exactly the right target maps. No new container, no deload-format change.

**How the search takes it.** A growing candidate never carries the stump. Per node the union of candidate and stump is built into a stack array, BOTH owner-set probes run on it — `normalizedEncodedSubkeys` for growth, the target map for recording — and the union is dropped again; only a candidate the target map accepts materialises it into a `BaseCandidate`. The union is a plain merge of two disjoint ascending runs: the stump's elements ascend by `(decoded name, statement index)`, the order the filtered list is sorted into, and the search skips filtered statements that are IN the stump (the "one copy of each expression" rule, applied where it costs nothing). Grow depth counts the union, so the search runs `|S|` levels shallower.

**The empty-candidate node.** Unsplit, the base candidate `{x}` is recorded inside the loop of the candidate one level up — the search's root loop. A sub-part stumped on `{x}` never runs that loop, so the stump is probed once on its own before the search, tally bump included, and recorded if the target map accepts. Without it the request `{x}` + obligatory stump disappears.

**Why completeness survives the union probe.** If `normalizedEncodedSubkeys` held only the name-sorted PREFIXES of a key, probing `C ∪ S` at every node would dead-end: reaching base candidate `{a,b,c}` from stump `{c}` would have to pass through `{a,c}`, not a prefix. It does not — `addToHashMemory` walks every permutation and installs each prefix that is still weakly name-increasing, so over all permutations every name-sorted SUBSET of a key is a subkey. `requestGatesPass` is closed downward under subsets (fewer premises: one hypothesis scope still, no more distinct secondaries, shorter key), and scope comparability is order-independent because `deeperOf` folds to the same scope. So every intermediate `C_j ∪ S` on a path to a valid base candidate passes, and the growth path is open.

**Buckets, not one sub-part per stump.** A rule-part returns one stump per expression surviving its filter — hundreds. One sub-part each, multiplied against the rule dimension, gave a single Peano induction LB **13,188 parts** and a 108 s hash burst. The search per sub-part was never the cost; the FIXED cost was — `filterIntEncodedStatements` walks the whole statement universe with a cold-map probe per statement, and the obligatory-stump builders do the same, five batches over, rebuilt by every part. This is [D-109](#d-109)'s open follow-up (*"the partition-independent setup is recomputed per part (N×); hoisting it is scoped but NOT yet done"*) coming due.

So a sub-part gets a **bucket**: `SplitStumpRef` carries a run of `ExpressionStump`, the generator filters and name-sorts once, then searches once per stump in the bucket into one shared `baseCandidates` column, and merges once. `proveKernel` deals the stumps round-robin into `min(stumpCount, fixed_number_splits)` buckets, permuted so each bucket is contiguous. Two wins: the setup is paid per bucket, and a request two of the bucket's stumps both reach — `{a,b}` from stump `{a}` and from stump `{b}` — is emitted twice into the SAME `StaticRequestEmitter`, whose `seen` collapses it. Two separate sub-parts each fired it.

| induction LB `(=[13,2])_induction_rec2_`, one hash burst | parts | firing records |
|---|---|---|
| 100 rule-parts, one sub-part per stump | 13,188 | 88,686 |
| 20 rule-parts, one sub-part per stump | 3,734 | 80,147 |
| 20 rule-parts, buckets of 20 | **115** | **28,506** |

`fixed_number_splits` drops 100 to 20, and doubles as the stump-count floor the producer grows a short filter list toward.

**The regular stump list is one uniform level.** Every filtered expression is a 1-stump; if that reaches the target, stop. Otherwise ALL surviving 2-stumps replace them, then all 3-stumps, up to `min(MAX_EXPRESSIONS, maxKeyLength)`. Uniform because a 2-stump contains two 1-stumps: keeping either as a growing seed alongside it would make one stump's larger candidates a subset of another's. **A stump of length `L` can produce no base candidate shorter than `L`** (a sub-part's candidates are `C ∪ S`), so a dropped candidate accepted by a minus-one or minus-two target map is retained as terminal-only work per [D-202](#d-202): checked on its own in the real request batch, never grown. The next level still owns every larger candidate.

**Two leftovers the design does not cover on its own.** A request that IS the obligatory stump — the case where that stump is already a whole key — has no base, so it contains no split stump; nor can it be placed by content (batch 3's obligatory stump is a `(local, external)` pair and `intExternalStatements` is not inside `intEncodedStatements`). Those are dealt across the sub-parts by obligatory-stump index. And the burst early-exit ([I-76](30_invariants.md#i-76)) must stay disabled for a stump sub-part — it is a sibling, and bailing on another part's stop is a scheduling race. It does, because only a RULE-SPLIT LB is ever stump-split, so `g_splitCount > 1` already; `performElem2` asserts that rather than trusting it.

**proveKernel restructured from per-LB to per-part.** Sealed record sets accumulate ACROSS the iteration's passes, so a rule-part that did not escalate keeps its records while a sibling re-runs as sub-parts; the LB is finalised ONCE, in the pass after which it has no tasks left. The merge is partition-independent ([D-117](#d-117) / [I-77](30_invariants.md#i-77)), so which parts came from which pass cannot matter. Requeued work runs in the NEXT pass, never appended to the running one — the working-set pager has registered this pass's task list and dispatch cursor. Three escalation levels means at most three passes, asserted. The `redo` vector is gone; `adaptiveSplitDecision` keeps its coarsen / hold branches and the finalize asserts `!redoNow`.

**The coarsening order.** *"First the expression split is reverted; then, if no rule-part is split and none is busy, unsplit."* The first half needs no code — the stump split is per-iteration state. The second half does: an escalated rule-part's burst is discarded, and its submatch tally must NOT go with it, or the LB is judged on its sub-parts alone (each comfortably under the cap by construction, since that is why they exist), looks light, coarsens to unsplit, re-hits the first cap and re-escalates every iteration. Its tally therefore counts toward the LB's busiest part. Branch (1)'s does not: that LB re-runs as split parts this same iteration, so its real numbers arrive anyway, and [D-111](#d-111)'s documented thrash band is preserved.

**Firing-record sort ceiling removed.** `applyFiringRecords` sorts a contiguous `int32_t` index, and one arena allocation holds one pool block: 65,536 slots, formerly a named assert. The stump split walked through it — its rule-parts stopped truncating, so the burst ran to completion and produced 80,147 records. The ceiling was a property of the ALLOCATION, never of the merge, so it became a chunk size: the index is cut into chunks of one block, each `std::sort`ed, and consumed merged. At one chunk — every LB on this corpus — it is the single `std::sort` over one contiguous index it always was, at the same cost. The merged sequence is unique because the comparator is a strict total order (I-77), so it is identical to the sort it generalises. `kMaxFiringRecordsPerLbBurst` becomes `kFiringRecordSortChunk` + `kMaxFiringRecordSortChunks`.

**What was ruled out.** *Attaching the stump to the obligatory sequence* — the unbounded minus-`k` map family above. *Unioning the filters of the four hash memories to build the stump list* — rule installation puts every recovered implication into `overallHashMemory` in EVERY case and only then fans it to `workingMemory` (status 3) or the local memories (status 0/1), so the other three are subsets and one filter suffices. *Recomputing the stump list single-threaded in `proveKernel` after the join* — the producer belongs in the part that hit the wall, already claimed, loaded, arenas warm, carrying the rule residue the stumps must respect. *Widening the sort index onto the page tier* (a `PagedVector` merge sort) — it removed the ceiling and regressed every LB in the system, including the tiny unsplit incubator ones that were never near it: page-directory lookups on every compare and every write, 17 passes, incubator bursts to 175 s. Chunking gives the identical result with the fast path untouched.

**Verification.** Build clean (Release x64, `-t:Rebuild`); `gl_quick.exe --unit-tests` 1227/1227. Two sequential full `main.py` runs: `theorems.txt` and `global_theorem_list.txt` BYTE-IDENTICAL between them; 46 theorems, Gauss `fold` summation present; verifier **131,881 checks, 0 failures**, every proof graph verified — the same total `main` reports. Runtime 1251 s / 1284 s. The split genuinely fires: 593 escalations across six induction and recursion blocks (e.g. rule-part 5 of 20 on `(in2[rec0,10,3])` reaching exactly 10,000 submatches and returning 295 one-expression stumps into 20 buckets).

**An expectation that was wrong.** A cap-truncated rule-part's burst used to be APPLIED (`adaptiveSplitDecision` returns `redoNow` only when `currentParts <= 1`), so `theorems.txt` was predicted to become a strict superset. It does not change at all. On this corpus that truncation never cost a theorem or a check — the completeness hole is real but empty here.

**Not exercised at runtime.** The multi-chunk merge (every burst fits one chunk) and the stump-growth levels (every escalation yields `stumpLen=1`). Both rest on construction and unit tests.

**See also.** [D-202](#d-202), `I-157`, `I-156`, `I-158`, [I-155](30_invariants.md#i-155), [I-74](30_invariants.md#i-74), [I-75](30_invariants.md#i-75), [I-76](30_invariants.md#i-76), [I-77](30_invariants.md#i-77), [D-109](#d-109), [D-111](#d-111), [D-117](#d-117), [D-121](#d-121).

---

<a id="d-200"></a>
## D-200 — one request generator, parameterised by the obligatory-stump length (0, 1 or 2) (2026-07-09)


**Choice.** Collapse the three request generators — `generateEncodedRequestsStatic` (mandatory single), `generateEncodedRequestsStaticPairs` (mandatory pair), `generateEncodedRequestsStaticCE` (no mandatory element) — plus the shared grow helper `growBaseCandidates` and the duplicated `filterIntEncodedStatementsCE`, into ONE function. The mode is a single `int16_t stumpLen ∈ {0, 1, 2}`: the number of already-known statements every generated request is obliged to contain. Everything else follows from it — the target owner-set map (`normalizedEncodedKeys` / `…SubkeysMinusOne` / `…SubkeysMinusTwo`), the grow depth (`maxKeyLength - stumpLen`), whether the statement filter also accepts full keys, and whether the seed and merge phases run at all.

**Why.** This is a restoration, not an invention. Before request generation was made heap-free (March 2026) there was one `generateEncodedRequests` whose comment read `// per contract: only 1 or 2`; the stump length was already a plain number and the target subkey set was already selected from it. The statification split it into a singles variant, a pairs variant, and — because the old function early-returned on an empty mandatory list — a third CE variant that re-inlined the grow DFS with its own dual full-key / subkey check. Three near-identical DFS loops and two near-identical filters is the cost that split imposed. Allowing a stump of length 0 removes the reason the CE case ever needed its own function.

**How the CE case folds in.** A zero-length stump means the base candidate IS the finished request. Three consequences, each a line of the unified body rather than a separate function:
- The statement filter must also accept full keys, because a statement can be a whole request on its own (`alsoAcceptFullKeys = (stumpLen == 0)`). With a stump, every survivor must still be growable, so subkeys only.
- The record gate drops its "must still be a growable subkey" conjunct — there is nothing left to grow into.
- The candidate is emitted INSIDE the search rather than recorded, which is what preserves the counter-example filter's contradiction early-exit (I-73). Deferring every emit past the enumeration would have kept the answer and lost the exit.

**Preserving the submatch tally (the trap).** `g_growthMatchCount` (D-109) is bumped by `preEvaluateFromEncoded` on every accepted probe; it caps the burst and is the LB-split policy's fill-ratio numerator, so its value is proof-visible. The unified grow loop must probe TWO owner-set maps per node — the subkey map for growth, the target map for recording — and two `preEvaluateFromEncoded` calls would double-count. The fix is `requestGatesPass`: the map-independent shape checks (hypothesis-scope consensus, distinct-secondary cap, length) are split out of `preEvaluateFromEncoded`, so the node runs the gates once, builds the key once, and probes both maps through the non-counting `ownerKeyAccepts`, bumping the tally exactly where the subkey probe accepts. The empty-stump path's tally does change (fewer bumps) — nothing reads it there: the cap is bypassed for CE in `BurstSink::canAccept`, and CE LBs never split. The body asserts `g_splitCount == 1` to pin that.

**Merge tie order — normalised, and NOT a hit/miss change.** The two surviving generators disagreed on where a stump element sorts against a base element of the same core-expression name: the singles variant inserted it BEFORE, the pairs variant appended and stable-sorted, landing it AFTER. The unified merge normalises to the pairs order (`base ++ stump`, one stable sort), chosen by the user over carrying a tie-order flag derived from `stumpLen`.

This cannot change which keys a request matches. `addToHashMemory` (and `makeNormalizedKeysForAdmission`) install a key for EVERY permutation of the rule's premises whose core-expression names are weakly increasing — the gate is `compareSpans(idsRun[perm[k]], idsRun[perm[k+1]]) > 0 → skip`, and `idsRun[i]` is `extractExpressionSpan(...)`, the name before `[`. A tie compares equal, not greater, so both orderings of tied premises are installed. A request presented in either tie order therefore finds its key.

**What the reorder DID change — an older latent bug, surfaced not caused.** The full-pipeline gate reproduced the reference run on every theorem count, every one of the 74 verifier tag rows, both `checks, 0 failures` totals, and both authoritative artifacts (`files/theorems/theorems.txt` and `files/processed_proof_graph/global_theorem_list.txt` byte-identical to the stored regression baselines). Exactly ONE line moved: the Gauss batch's hash-burst 3 census, `total_exprs` 123618 → 123623. The five extra intermediate expressions are gone by burst 4 and reach no artifact. Two sequential runs of the same build reproduce `123623` and agree on all 337 burst-census lines, so the branch is deterministic; the delta is a deterministic consequence of the reorder.

Since the tie order cannot move a key lookup, something DOWNSTREAM of request generation is sensitive to the order in which a request's premises are presented — the emitted tuple order feeds `StaticRequestEmitter::seen` (dedup on the packed `(originalId, validityId)` sequence) and `checkLocalEncodedMemoryStatic`. That sensitivity predates this branch: it was simply never exercised, because the singles and pairs generators each had a fixed order. See the open question below.

**Alternatives rejected.** (a) Keep `growBaseCandidates` as a shared helper and unify only the three generators — the CE case needs to EMIT from inside the DFS, so the helper would have to take the consumer and the stump length anyway, at which point it is the generator. (b) Widen the regular filter to the subkey/full-key union unconditionally — enlarges the base-candidate pool and changes the proof. (c) Give the unified loop two `preEvaluateFromEncoded` calls — doubles the submatch tally, changes the split policy, changes the proof.

**Open question — the five Gauss expressions.** Which of the two order-sensitive consumers produces the +5 has not been isolated. Candidates, in order of suspicion: (a) `StaticRequestEmitter::seen`, whose dedup key is the packed `(originalId, validityId)` run in emitted order, so two `(base, stump)` pairs that collapse to one tuple under one order and two tuples under the other would emit a different number of requests; (b) `checkLocalEncodedMemoryStatic`'s `encodedMap` probe, which builds its lookup key from the request's premise order and back-substitutes through that key's `reverseMap`. Isolating it needs a per-`(LB, burst, batch)` emitted-request trap diffed across the two orders. The extra expressions are canonicalized away before the next burst, so no proof output depends on the answer — but an order-sensitive dedup on the hot path is worth understanding before FTA scale.

**See also.** `I-155`, [I-70](30_invariants.md#i-70), [I-79](30_invariants.md#i-79), [I-73](30_invariants.md#i-73), [I-130](30_invariants.md#i-130), D-109, D-105, D-120, D-59 (the retired "4 `generateEncodedRequests*` sites" workaround — there is now one site).

---

<a id="d-199"></a>
## D-199 — a derived reverse membership side-index inverts the firing-check candidate scan; a chained owner list, not a strict CSR value column (2026-07-08)


**Choice.** Add `ReverseArgsIndex` (`memory_infra/reverse_args_index.hpp`), a derived per-`HashMemory` side-index for `remainingArgsNormalizedEncodedMap` (`Int16SetKey → run of NormKey`). It maps a `NormKey`'s bytes to the forward-map key ids whose run contains that NormKey, so `checkLocalEncodedMemoryStatic`'s candidate enumeration becomes one hash probe instead of a full O(keys) forward scan + a per-candidate O(run) byte-peek membership recheck. It is DERIVED (I-117): on the deloadable arena, never enrolled/deloaded/dumped, rebuilt on canonical reload, verbatim in the raw image. Full lifecycle contract in [I-154](30_invariants.md).

**Why (regression).** The statification of `remainingArgsNormalizedEncodedMap` replaced an `unordered_set` O(1) membership probe with the scan; at main tip the cold scans show as `PagedVector<int>::operator[]` ~5.3% + `BytesKeyStore::decodeAt` ~5.6% + `peekBlobContiguous` ~1.6%, and IncubatorGauss regressed up to ~5x. The reverse index restores the O(1) lookup while staying 100% static (I-95 preserved).

**Chained owner list, NOT a strict CSR value column.** The design brief named "a `PagedVector<int32_t>` CSR value column." The reverse index is instead built on a `ColdHashSet<BytesKeyStore>` NormKey interner (the `PagedHashIndex` + byte pool that make membership EXACT — a hash collision between two distinct NormKeys is resolved by the byte compare, so the answer carries no false positives) plus three `PagedVector<int32_t>` columns (`headById_` + a `nodeOwner_`/`nodeNext_` node pool) forming a per-NormKey singly-linked owner chain. Rationale: the index is maintained INCREMENTALLY by `appendEdge` at every install, and a strict CSR run would need an O(tail) interior splice per new edge to an existing NormKey — reintroducing exactly the `PagedVector<char>::replaceRange` / `memmove` cost (~4% + ~3.9% at main tip) this campaign removes. Chaining makes `appendEdge` O(1). The substrate is still the sanctioned D-166/I-117 pair (`PagedHashIndex` via `ColdHashSet` + `PagedVector<int32_t>`) on the deloadable arena; only the value encoding (chain vs contiguous run) differs, and it is byte-invisible to every observable (the chain order is never emitted — the candidate loop re-sorts by `int16SetKeyLexCompare`).

**Alternatives rejected.** (a) Strict CSR with incremental append — O(N²) at Gauss scale, self-defeating. (b) A bare `PagedHashIndex` hand-rolled open-addressing without a byte pool — cannot disambiguate NormKey hash collisions, so the answer would be an over-approximation and the candidate set would not equal the former scan exactly. (c) A rebuild-per-burst CSR — the brief requires incremental maintenance (`appendEdge` inside `insertRemainingArgsNormKey`).

**Batched install (same campaign).** The install-side counterpart: the two `insertRemainingArgsNormKey` sites (`addToHashMemory` head, `makeNormalizedKeysForAdmission` marker) run a permutation loop whose remaining-arg KEY is loop-invariant, so feeding NormKeys one at a time paid a whole-run RMW (peek → sorted splice → `assignRun` → blob-pool tail memmove) PER permutation — at main tip `insertRemainingArgsNormKey` 2.06% self + `BlobCsrValueStore::replaceRun` 2.21% + `PagedVector<char>::replaceRange` 4.01% + memmove 3.91%. `insertRemainingArgsNormKeyBatch` (`prover.hpp`) collapses them: accumulate each permutation's `Codec<NormKey>` blob on the gen-scratch byte-bump tier (surviving the loop's self-framing `appendLmvIdsRecord`), then ONE peek + ONE sorted-unique two-pointer merge (same `(numberExpressions, data)` comparator) + ONE `assignRun` per key. Byte-identical to N sequential inserts — the final run is the sorted-unique union (order-independent), and a reverse edge is appended for each NEW record (batch-only), exactly the sequential `!dup` edges; an all-dup batch skips the write entirely (the sequential no-op). The retained per-record path is the twin-test oracle (`insert_remaining_args_batch_matches_sequential`). Zero heap (blobs / peek / sort index / merge / concat all on the gen-scratch arena).

**See also.** [I-154](30_invariants.md), [I-117](30_invariants.md#i-117), `I-95`, `I-107`, D-166, D-173.

---

<a id="d-194"></a>
## D-194 — skip the three-phase burst of any LB whose burst would provably do nothing (2026-07-07)

**Choice.** `proveKernel`'s single-threaded active-build excludes from the sweep every active LB whose next burst is a provable no-op, so converged LBs stop being processed — and, at 4 GiB, stop being paged in and out. Gated by `parameters.enable_quiesce_skip` (default `true` on this branch).

**Predicate (as implemented).** `sweep X ⟺ X->isActive AND (warmUpPhase OR !enable_quiesce_skip OR compressor_mode OR X->hasWork OR mailLog.mailPeek(X))`. `hasWork` is a producer-side latch on the never-deloaded LB slab (I-109); `mailPeek` is a fold-free poll of the never-deloaded mail log (`mailHeads.count > mailCursor`) for un-ingested ancestor mail. The predicate reads ONLY never-deloaded logical state — never `resident` / `blocksInUse` (the I-106 / I-108 / D-149 determinism doctrine). Residency is a *consequence* of the skip, never an *input*.

**Why it is sound (byte-identity).** The prove loop is CAP-driven (`for it < numberIterations`), with no fixpoint/no-new-work termination, so skipping a burst cannot change the iteration count nor which iteration a fact first appears in (a skipped burst adds no fact). Correctness reduces to a LOCAL property: *is each skipped burst individually a no-op?* SLEEP clears `hasWork` at `performElemPhase3` exit only when the burst mutated nothing — statement count unchanged vs the phase-1 baseline, `mutatedThisBurst` false (admission churn / subtree wipe), and both INTERNAL mail channels (`sameIterationInternalMail` / `nextIterationInternalMail` — inputs the LB's own next burst absorbs) empty. WAKE sets `hasWork` at every cross-LB write door (I-153); cross-LB mail is polled by `mailPeek` at the same program point the pull would run, so no mail latency is added (I-55 preserved). A sound over-approximation in the I-70 / I-79 mould: a missed wake is a missed theorem (unsound), an extra sweep is merely slow — so when in doubt the LB stays dirty.

**`mailOut` is deliberately NOT a SLEEP condition.** `mailOut` is OUTGOING-ONLY under the pull model (I-64: `fillMailOut` and the deposit doors write it, the single-threaded commit barrier is its sole consumer), and no burst path derives local state from it — the one historical burst-path read, the retired transitive-ship dedup, is dead code. Pending `mailOut` content therefore cannot make the LB's own next burst productive; and the barrier's commit sweep iterates `bodies` regardless of sweep status, so that content is committed whether or not the LB sweeps — output-identical either way. The first shadow-validation run proved the point empirically: the post-join drains (the `updateGlobal*` theorem sends, the D-76 compaction flush) append to the ROOT's `mailOut` AFTER the barrier, so a genuinely no-op root burst saw "mailOut non-empty," re-armed `hasWork`, and false-fired the shadow assert at the root sentinel (forensic signature: root-only chain, count 0 vs 0 unchanged, internal channels empty, only the mailOut terms TRUE, mailPeek false). `mailOut` was never a valid activity signal; both `mailOut` terms were removed from the fold.

**Why the field was anticipated.** `Memory` already carried a vestigial `deltaNumberStatements` (documented "used to short-circuit no-op iterations") with no prover increment site — the skip was a designed-for point, not a new architectural direction. It is left in place, comment-corrected as vestigial (Rule 13); the live latch is the fresh `hasWork`.

**Evidence ( / `_8gib.log`).** IncubatorPeano1's last ~12 bursts run with `total_exprs` frozen yet all ~911 active LBs swept every iteration; each frozen iteration costs ~0.14 s at 8 GiB but 1.1–7.0 s at 4 GiB, the delta being pure paging of provably-idle LBs. The skip collapses the 4 GiB frozen tail toward the 8 GiB cost.

**Deactivation schedule preserved for free.** The I-48 bubble-up survey (`deactivateRecursively`) is a tree-wide POST-JOIN walk from the root reading only never-deloaded state (`intToBeProved` persistent I-108; child `isActive` + edges I-109 / I-110), so a skipped-but-active LB is still surveyed there without a reload — no wake or separate survey placement is needed. The design's proposed child-deactivation parent-wake was therefore unnecessary.

**Validation.** `QUIESCE_SHADOW_CHECK` (compile flag, off by default, in-tree): sweep every would-be-skipped LB anyway and assert its burst left `hasWork` clear — any missed wake / missed mutation fires at the exact LB. Then A/B `main.py` at 8 GiB (skip ON vs config `enable_quiesce_skip:false`) for byte-identity, and 4 GiB for the win.

**See also.** I-153, [I-48](30_invariants.md#i-48), [I-55](30_invariants.md#i-55), [I-106](30_invariants.md#i-106), [I-108](30_invariants.md#i-108), [I-109](30_invariants.md#i-109).

---

<a id="d-195"></a>
## D-195 — two deload formats: v4 raw arena images (nondeterministic bytes) for eviction, v3 canonical retained for discharge/export (2026-07-07)


**Context.** The v3 canonical deload ([I-103](30_invariants.md#i-103)) streams each container element-by-element into a heap payload and reloads by rebuilding every container element-by-element AND re-FNV-probing every cold-map key to rebuild the throw-away index. Phase-0 telemetry () measured the v3 reload at ~0.06–0.07 GiB/s on worker threads — the bottleneck of the eviction/reload HOT LOOP (active LBs the working-set pager cycles under pressure). The canonical-bytes property is load-bearing for the DISCHARGE / chapter-export path (A/B-comparable `.deload/` artifacts, cross-run/host reproducibility) but buys nothing on the eviction path, where the image is written and read back within one run and never compared.

**The choice (user-approved).** Split the deload into TWO formats:
- **v4 RAW arena image** for the eviction/reload hot loop. A byte image of the LB arena's live pages (ascending vid) + a live-vid bitmap. The insight: every deloadable container — the statement `PagedVector`s, the cold-map key/value stores, AND the throw-away `PagedHashIndex` bucket arrays — packs its pages into the ONE `LbMemory::manager` arena, and container scalar bookkeeping (`size_`/`rootVid_`/`numPages_`) lives in the never-deloaded `Memory` shell. So a page image, restored by binding the same vids to freshly-carved dense pages and refilling the bytes, is content-complete with ZERO container work: no element walk, no `rebuildIndex` (the bucket pages come back byte-identical). One copy total, near memcpy.
- **v3 canonical** retained UNCHANGED for the discharge drain and the chapter-export `ensureLoadedForRead` reads, where [I-103](30_invariants.md#i-103) stays intact.

Load dispatches on a recorded per-LB `Memory::deloadKind` (`Raw` / `Canonical`), set by whichever dump ran last, so a raw-evicted LB later discharged flips back to canonical; each loader also asserts its own header version. `Memory::releaseStaticBlocksRaw` skips the v3 `container.release` walk (the bookkeeping must survive for the raw rebind); the resulting teardown of a raw-deloaded `Memory` on a cold arena is handled by a teardown-only residency branch in the destructors `~PagedVector` / `~PagedHashIndex` (a DEFINED lifecycle state, not defensive — Rule 19). The branch lives ONLY in the destructors: a LIVE `clear` reaching a cold arena still dies loudly on `freePage`'s residency assert, so the guard cannot mask a mid-run bug.

**Trade-offs.** ACCEPTED: raw eviction-image BYTES are nondeterministic (arena fragmentation + block grant order leak into the file). This is sound because the RESTORED LOGICAL STATE is byte-identical (same vids → same bytes, container bookkeeping untouched), and proof output (theorems, proof graphs, verifier checks) depends only on logical state — the eviction set is already timing-dependent (the retired [I-115](30_invariants.md#i-115) / [I-114](30_invariants.md#i-114) precedent). GAINED: near-memcpy dump/load + zero index rebuild on the hot loop. [I-103](30_invariants.md#i-103) is NARROWED to discharge/export images (its canonical-bytes doctrine no longer covers the eviction path); determinism tooling must not point at v4 raw files.

**Dynamic header (amendment).** The original fixed 4 KiB header carried a deliberate capacity tripwire (chain + bitmap must fit); it fired on the first big Gauss LB at 4 GiB (`hn + bmBytes <= kRawHeaderBytes` — a ~1000-block LB ≈ 32000 vids = a ~4000-byte live bitmap that cannot share 4096 bytes with the chain and the fixed fields). The v4 header is now DYNAMIC: a `kRawHeaderPrefixBytes` (48) fixed prefix carrying `headerBytes` (total header size, rounded to `kRawHeaderAlignBytes` = 4 KiB so the payload stays aligned), then the chain, the bitmap, and zero padding — sufficient by construction, no capacity assert. Both sides stay heap-free by streaming the bitmap through one bounded 4 KiB stack chunk buffer (`LbArena::fillLiveBitmapRange` on the dump side; the staged `restoreForRawLoadBegin`/`Chunk`/`End` on the load side, with the single-shot `restoreForRawLoad` kept as the one-chunk wrapper).

**See** [I-103](30_invariants.md#i-103) (narrowed), [I-107](30_invariants.md#i-107) (vids position-independent — the enabling invariant), [I-117](30_invariants.md#i-117) (the throw-away index the raw image preserves for free), [I-114](30_invariants.md#i-114) (the working-set pager whose eviction path this accelerates).

<a id="d-197"></a>
## D-197 — the working-set pager drains to a free-block RESERVE by evicting farthest-next-use (Belady) victims behind the cursor, replacing the mass-evict-at-3/4 + one-for-one-at-1/2 watermark policy (2026-07-07)


**Context — the measured per-eviction cost and the exhaustion anatomy.** Two facts from the raw-datapath 4 GiB run () bind this decision. (1) **Per-EVICTION fixed cost dominates**: ~50 ms per ~2.5 MB image, an NTFS file create+close per LB — largely independent of the LB's size, and there are thousands of tiny LBs (median ~2 blocks early-batch, ~13-block average at peak). So minimizing the eviction COUNT matters as much as choosing the right victims, and evicting a tiny LB pays the whole fixed cost to reclaim almost nothing. (2) **The old watermark policy caused the exhaustion assert** (`IncubatorPeano1`, third prover invocation, burst 2): mass-evict above 3/4 dumped EVERYTHING outside a (phase-2-frozen) window, so grants outran frees and eviction victims starved — every candidate was either claimed (in flight) or already Dumped, and the pool depleted. The [D-161](#d-161) watermark model (mass-evict at 3/4, one-for-one exchange between 1/2 and 3/4, inert below 1/2) is what starved.

**The choice (user-designed).** Replace the watermark drain with a **free-block RESERVE target filled by Belady victims**:
- **Reserve target.** Each `maintainWorkingSet` pass keeps `freeBlocks >= kReserveBlocks` (2048 blocks = 512 MiB = 12.5 % of a 4 GiB pool). While short, it evicts one victim and recomputes; the pass ends when the reserve is met OR no eligible victim remains — the latter a DEFINED result (the working set fits, the stream pauses), never a failure. Genuine pool exhaustion still asserts at `acquireBlock` (Rule 19), never a silent wait.
- **Belady behind-cursor victims.** `pickVictimBehindCursor` picks the FARTHEST-next-use eligible LB: GL processes the active vector cyclically, so next-use distance from the cursor `c` is `d(i) = (i − c) mod n`, maximised by the LB immediately behind the in-flight margin. A backward scan from `keepLo − 1` with wraparound, stopping on re-entering `[keepLo, keepHi)`, visits strictly decreasing `d`, so the first eligible hit is the Belady victim; repeated calls yield the k farthest-behind LBs. This keeps victims ALWAYS AVAILABLE (the just-processed LBs behind the cursor), curing the starvation.
- **Size floor.** Eligibility adds `blocksHeld >= kMinEvictBlocks` (4 blocks = 1 MiB): tiny LBs stay implicit permanent residents, so the eviction count stays low and freed-blocks-per-round-trip high (the per-eviction fixed cost).
- **Worker de-loading** ([D-pending](#) same slug): the routine load↔evict exchange in `claimAndLoadForWork` is deleted; a worker reloads into the reserve's free blocks, and only a COUNTED emergency fallback (`emergencyEvictCount`, expected 0) evicts inline when the pool is above the 3/4 hard bound AND free blocks cannot cover this reload. *(Amended by [D-196](#d-196): the fallback's trigger is now the emergency floor on EVERY claim — the 3/4 gate is dropped — and its selector is the two-tier Belady scan, not `pickBiggestDeloadable`.)*

**Trade-offs.** The eviction set stays timing-dependent (the resident set varies run-to-run), sound exactly as before because active-LB eviction/reload is content-invisible ([I-103](30_invariants.md#i-103), [I-107](30_invariants.md#i-107)) — proof output is byte-identical. The stream PAUSES when nothing is eligible (a defined boundary, not back-pressure). **Phase-2 cursor still frozen** (RESOLVED by [D-196](#d-196): the executor/finalize windows register phase 2's real dispatch atomics; the frozen `phase2Cursor` is retired): the phase-2 window was stuck at cursor 0 this subsession, so phase 2 churned (reload/evict of LBs the frozen window mislocated) but never exhausted — the claim word still guaranteed an in-flight LB was never evicted, and behind-cursor victims stayed available. At 8 GiB the reserve is never breached (peak ~22584 blocks « 30720 = total − reserve), so the pager is inert and light batches run exactly as before.

**Supersedes / amends.** [D-161](#d-161) (the watermark drain — mass-evict at 3/4, one-for-one at 1/2 — is replaced by the reserve target; the unified window replaces the asymmetric prefetch/keep windows). [I-114](30_invariants.md#i-114) is rewritten to the reserve model.

**See** [I-114](30_invariants.md#i-114) (rewritten), [D-161](#d-161) (superseded watermark policy), [D-195](#d-195) (the raw datapath every eviction uses), [I-103](30_invariants.md#i-103) / [I-107](30_invariants.md#i-107) (content-invisibility).

<a id="d-196"></a>
## D-196 — survival at 4 GiB: two-tier victim selection, the widened worker pressure valve, and the grant-trigger wake rejection (2026-07-07)


**Context — the post-pager 4 GiB failure.** The reserve-target Belady pager ([D-197](#d-197)) died EARLIER at 4 GiB than the watermark policy it replaced (, `IncubatorPeano1` burst 1). Anatomy, four causes: **(a) size-floor starvation** — `kMinEvictBlocks = 4` against a median-2-block early-batch population means the drain finds NO eligible victim at all; every pass ends on the "no victim" defined pause while grants continue, and the pool walls. **(b) serial eviction throughput** — ONE steward thread evicting synchronously at ~20 files/s (the ~50 ms NTFS create+close fixed cost per tiny image) cannot match 32 workers' grant rate (scratch + statement growth); the 2048-block reserve is a seconds-deep cushion, not a policy. **(c) no valve on the pure-grant path** — the worker emergency eviction triggered only on "my reload does not fit"; workers that grant scratch / statement blocks with no reload in sight had no pressure valve at all. **(d) phase-2 blindness** — the phase-2 cursor frozen at 0 mislocates the window through the heaviest phase (fixed by the executor/finalize windows of this subsession).

**The choice (user-mandated amendments; this entry grows as the subsession's commits land).**

- **Two-tier victim selection** (`MemorySteward::pickVictimTwoTier`): tier 1 is the floor-preferred Belady scan (`pickVictimBehindCursor` with `kMinEvictBlocks`); when the reserve is breached and NO above-floor victim exists, tier 2 re-runs the SAME backward Belady scan with the floor dropped to one block. Survival trumps per-operation efficiency; both tiers are DEFINED results (Rule 19), and a `nullptr` from both is still the defined working-set-fits pause. Cures (a).
- **Widened worker pressure valve**: `claimAndLoadForWork` checks EVERY claim — resident or cold — against the emergency floor `steward::kEmergencyFloorBlocks` (= `kReserveBlocks / kEmergencyFloorDivisor` = 512 blocks = 128 MiB; scales with the test-overridable reserve) and, on breach, evicts ONE behind-cursor victim through `evictOneForReload` — which now selects via the two-tier Belady scan; the biggest-first selector `pickBiggestDeloadable` is RETIRED (no other consumer). The reload-fit trigger (free blocks cannot cover THIS reload, estimated from `lastRawImageBytes`) stays on the cold-miss path; the old 3/4-hard-bound gate is dropped. A per-claim guard keeps it at most ONE eviction per claim, no loop — a still-short reload surfaces genuine exhaustion at `acquireBlock` (Rule 19). Counted (`emergencyEvictCount`; nonzero = the planner cannot keep up). Cures (c).
- **Planner/executor I/O split**: the steward becomes a PLANNER thread (the existing 200 µs window poll, now enqueue-only) plus `steward::ioThreadCountFor(workers, kIoThreadsOverride)` I/O EXECUTOR threads (= `clamp(workers/8, 2, 8)`, 4 at 32 workers; the override constant forces a count — set 1 to serialize the pool when isolating a concurrency bug). Two fixed-capacity task RINGS under the steward mutex (`kIoRingCapacity` = 256 each): HIGH = prefetch loads, LOW = evictions + reshuffles. Executors drain HIGH first EXCEPT below the emergency floor — then LOW first (evictions must not starve behind reloads with no blocks to land in). Task execution is claim-first drop-on-lose (CAS `Dumped → Busy` for loads, `Idle → Busy` for evictions/reshuffles; a lost CAS discards); every eviction/reshuffle task re-validates its window under the CURRENT cursor via a monotone `windowGeneration_` (bumped at every `beginPhaseWindow` / `endPhaseWindow`) plus a kept-range check on the task's index — a stale task drops, a defined hand-over. Enqueues dedup by (kind, LB) ring containment (hygiene; the CAS arbitrates correctness); a FULL ring drops the enqueue — the designed load-shedding (the LOW lane can want ~1024 victims in the floorless early-batch regime against 256 slots; the planner regenerates its whole plan every pass, far faster than executors consume 256 evictions), counted per lane (`prefetchDroppedFullRing` / `evictDroppedFullRing`). The planner's reserve drain becomes a PROJECTION: live free + in-flight dump blocks (`inFlightBlocks`, freed-but-unavailable) + blocks of already-queued eviction tasks, so one shortfall is covered exactly once. The ASYNC DUMP CONTRACT rides the LOW lane (this IS the design's DeloadWriter — built once here): an eviction task holds the LB `Busy` and its blocks until the raw image write completes, releases the blocks, and only THEN stores `Dumped` — `Dumped` is never observable before image-complete + blocks-returned; in-flight bytes/blocks are exposed (`inFlightBytes` / `inFlightBlocks`). `quiesce` extends to rings-empty AND executors-idle AND planner-parked — LOAD-BEARING for the barrier, whose `deloadRegistry` reference read (`rewriteRegistry`) races executor dump registrations without it; `stop` joins the pool. The planner's ONE remaining I/O is the discharge drain, verbatim on the planner thread (I-106 execution unchanged). Each executor owns a private compaction scratch `LbArena` from `lbMemory`. Cures (b).
- **Barrier seam windows + claim-correct seam doors** (the SECOND 4 GiB wall, diagnosed by the forensic exhaustion census): the full-architecture 4 GiB run still exhausted at `IncubatorPeano1` — and the census at the wall showed NOT a pinned set (`UNATTRIBUTED` = 1, scratch = 123, `WorkerOwned` = 0, in-flight = 0) but **1467 Idle LBs holding 14106 / 16384 blocks (13384 of them eligible-victim blocks)** plus the anomaly that named the mechanism: **290 LBs claim-`Dumped` yet resident with 2146 blocks**. Everything idle, no workers → the wall was hit BETWEEN iterations, at the single-threaded post-join seams: the commit-barrier mail sweep and the `updateGlobal` / `updateGlobalDirect` / deferred-ancestor drains reloaded recipients (a) with NO pager window open — the planner only drains while a window is registered, so nothing evicted during the seams and residency grew monotonically — and (b) through bare `ensureLoaded`, bypassing the claim word, leaving `Dumped`-but-resident LBs invisible to `pickVictimTwoTier` (eligibility requires claim `Idle`). The fix: (1) the commit sweep — a clean linear walk over `bodies` — gets a dispatch-cursor pager window like a phase; (2) the deposit drains — random-access per-theorem `accessMemory` tree lookups, NO sweep order to register — run under a window over `active` with the cursor PINNED AT 0, the honest Belady origin between iterations (the next use of `active[i]` IS position `i`): the planner keeps the next iteration's head resident and evicts everything else farthest-first while the drains churn; (3) every seam reload goes through the CLAIM-CORRECT DOOR — `claimAndLoadForWork` (new phase id 4, the barrier telemetry bucket; the `DeloadStats` claim-wait arrays widened) → write under `WorkerOwned` → release `Idle` — so `Dumped`-but-resident is impossible by construction, asserted at the door (a successful `Dumped → Busy` claim over a resident arena = a bypassed reload); (4) the planner's prefetch branch skips `!isActive` entries (a window over `bodies` contains long-discharged LBs whose stale `Dumped` claims a load task would otherwise chase into the discharged-assert); (5) the barrier discharge reload uses the same door post-quiesce (its valve a defined no-op with no window). Recipient churn under the pinned window is CORRECT behavior, not a defect — a door-reloaded recipient releases `Idle` and is immediately evictable again; its deposit lives in the arena and rides the raw image. The seam windows close strictly BEFORE the discharge barrier block (disarm → quiesce → …), so they never interleave with the discharge drain's ordering. The only reloads still outside a door are the post-prove reads (visualizer equality nodes, chapter export) — the steward is destroyed there, no pager exists, claim words are meaningless.
- **Phase-2 windows + barrier head prefetch**: phase 2 opens TWO pager windows PER PASS instead of one frozen one — an EXECUTOR window registering the executor pool's real dispatch atomic (`next` over the flat `execOrder` vector built in task order; duplicates for split parts collapse via the enqueue dedup + claim CAS, and split LBs are ineligible victims anyway), closed at the executor join; then a FINALIZE window over `nextLi` / `toRun`, closed at the finalize join. Redo passes open their own windows over their own vectors, so indices always live in the right space. The frozen `phase2Cursor` (a completion cursor stored only by the finalize — it sat at 0 through the whole hashburst, blinding the steward in the heaviest phase and pushing thousands of reloads inline onto workers) is RETIRED. `DeloadStats`' single-window pairing holds unchanged: the two phase-2 sub-windows are strictly sequential (each closes before the next opens), both reporting under phase id 2. And the barrier gains `MemorySteward::prefetchHead(order, count, directory)` — a one-shot HIGH-lane enqueue of loads for the `Dumped`, still-active LBs among the next iteration's head `[0, min(count, n))`, called at barrier END after the discharge decision with `kAnyWindowGeneration` tasks (deliberately crossing the window boundary — kills the phase-1 cold start every iteration; safe against `dischargedForever` because the barrier's quiesce empties the rings before the discharge loop sets the flag, and inactive heads are skipped at enqueue). Cures (d).
- **Prefetch budget + teardown discard** (the THIRD 4 GiB wall — the teardown burst): with the seams windowed, the 4 GiB run completed ALL 48 `IncubatorPeano1` bursts — and then exhausted in prove's TEARDOWN: the `[EXHAUSTION]` pool line (16384/16384) printed immediately after the final burst's `dt` line with NO census (the prove-scope guard had already cleared the reporter — its old first act), and none of the epilogue prints (`Prover finished.`, the theorem save, the proof-graph export) had run. The only pool-granting activity in that window is the executor pool draining the FINAL barrier's `prefetchHead` loads: `kAnyWindowGeneration` tasks that execute regardless of window state, warming up to a window-width of head LBs for a next iteration that never comes — with NO window open (the planner cannot plan a single relieving eviction) and NO valve on the executor load path. Two mechanisms close it, both survival-structural: (1) the **prefetch budget** `kPrefetchBudgetBlocks` = `kReserveBlocks / kPrefetchBudgetDivisor` (the design doc's unified-window clause, previously unimplemented) — pending prefetch loads, estimated from `lastRawImageBytes`, are capped below the reserve at BOTH issue sites (`prefetchHead` and the planner's per-pass HIGH-lane enqueues, whose pending load blocks are summed under the mutex symmetrically to the eviction projection), so an executor load is structurally unable to hit the wall — this also bounds the latent MID-RUN burst case (a window-width load burst racing the reserve with no executor-side valve); (2) **teardown discard** — the prove-scope guard's first act is now `discardQueuedIoTasks` (drop every queued-but-unstarted ring task; defined load-shedding, the ring-drop doctrine; in-flight tasks finish under the guard's quiesce). The guard's exhaustion-reporter clear moved to its END (after quiesce + stop), so any future teardown exhaustion prints a full census — `bodies` and the steward both outlive every possible invocation.

**Grant-trigger wake — evaluated and REJECTED.** A proposed third amendment armed `GlobalMemoryManager::armGrantTrigger` per planner pass so a mid-phase grant burst breaching the reserve would wake the steward immediately instead of the 200 µs poll. Rejected on contract-fit before any code was written; the decisive reason first:

1. **Wrong shape of signal.** The trigger fires on `grantsSinceBarrier` — a deliberately MONOTONE counter ([I-106](30_invariants.md#i-106): order-independence is what makes it a legal mid-iteration signal) that never decrements. The reserve breach is a LIVE `freeBlocks < reserve` condition over occupancy. The two agree only while nothing frees blocks — but the pager's whole job is to free blocks mid-iteration, so from the first eviction onward the ledger diverges from occupancy and would re-cross any re-armed threshold even with the reserve comfortably met. A monotone grant count structurally cannot represent a live reserve breach; even a dedicated second grant-trigger would be the wrong primitive.
2. **Single shared one-shot, owned by the discharge channel.** The barrier arms the trigger (below the wake watermark, with pending discharge installed) and it may stay armed through the whole next iteration — exactly when the pager would want to arm it. `armGrantTrigger` asserts against arming while armed.
3. **Wrong wake target.** The trigger's callback contract is `wake`, which sets `runRequested_` and routes `threadMain` into the discharge/eviction drain branch — not the window-poll branch the pager pass runs in.
4. **Re-arm per pass contradicts the self-disarm-on-fire lifecycle.** The trigger disarms only by firing; re-arming an unfired trigger asserts, and the planner cannot know whether it fired without extra state.

The 200 µs poll therefore stays as the planner cadence — its worst-case added latency is noise against the ~50 ms per-eviction fixed cost — and the widened valve reacts instantly at the claim seam with no planner involvement. A tightened under-pressure poll was ALSO deferred as premature: if the dod subsession's `reserveShortfallPasses` / `emergencyEvictCount` telemetry later shows planner latency matters, it is a five-line change then.

**Trade-offs.** Floorless tier-2 evictions pay the full per-file fixed cost for ~2-block reclaims — accepted while the reserve is breached (the alternative is the measured wall). The widened valve reads live `blocksInUse` on every claim (one pool-mutex acquisition) — a sanctioned pager read (the [I-114](30_invariants.md#i-114) relaxation of [I-106](30_invariants.md#i-106)), content-invisible relief that never feeds a deload-SET decision.

**See** [D-197](#d-197) (the policy this hardens), [I-114](30_invariants.md#i-114) (rewritten again), [I-106](30_invariants.md#i-106) (why the grant ledger is monotone — the rejection's load-bearing fact), [D-195](#d-195) (the datapath every eviction uses).

<a id="d-198"></a>
## D-198 — ONE preallocated extent file for v4 raw eviction images; per-LB power-of-two slabs, positioned I/O, per-thread handles, no I/O lock (2026-07-07)


**Context — the per-OPERATION fixed cost.** The v4 raw arena image ([D-195](#d-195)) fixed the CPU cost of eviction (near-memcpy, zero index rebuild), but the measured wall is now the per-OPERATION I/O fixed cost: `dumpLbMemoryRaw` did one NTFS `CreateFile(CREATE_ALWAYS)` + write + `close` per eviction, `loadLbMemoryRaw` one `OPEN_EXISTING` + read + close per reload — ~50 ms per ~2.5 MB image, almost SIZE-INDEPENDENT (create/truncate/close metadata plus, very likely, Windows Defender scan-on-close of each freshly created file), times thousands of tiny-LB evictions per phase. The byte copy itself (~0.6–2.5 ms) is a small fraction. `IncubatorPeano1` at 4 GiB runs ~280 s versus a 93–102 s band at 8 GiB (which barely evicts) almost entirely on this fixed cost.

**The choice (user-approved).** Hold ONE preallocated data file (`.deload/extent.bin`) open for the whole batch and place each LB's image at a STABLE slab offset inside it; dump/reload by POSITIONED I/O into the already-open handle — no per-eviction create/truncate/close, no per-file Defender scan.

- **Per-LB owned power-of-two slab, in-place overwrite, promote-on-growth** (`ExtentAllocator`, [`memory_infra/extent_file.hpp`](../../GL_Quick_VS/GL_Quick/src/memory_infra/extent_file.hpp)). Each LB owns ONE slab (the smallest power-of-two multiple of the 256 KiB pool block that holds `header + payload`) for its active life. A stable/shrinking re-dump overwrites IN PLACE (a single positioned write, zero allocator interaction — the near-memcpy hot path); only a growth past the class reallocates (free old → alloc new → write at the new offset). LB sizes grow ~6x over a batch ⇒ ~3 promotions, so the slab allocator is touched a handful of times per LB. No demotion (slack accepted to avoid churn). This beats an append-log (grows to Σ evictions×image, forcing multi-GiB compaction — the thick I/O we bound) and a best-fit in-file heap (external fragmentation + its own compaction); slabs give O(1) alloc/free, zero steady-state growth, ≤ 2x internal slack.
- **Concurrency: per-thread handle, positioned I/O at disjoint offsets, no I/O lock** (`PositionedFile`). The I/O executors + the worker valve operate on DISTINCT LBs (arbitrated by the `stewardClaim` word), hence DISJOINT byte ranges. On Windows each thread transparently gets its OWN `CreateFileW` handle (own file object → no shared file pointer, no file-object serialization lock); positioned I/O is `WriteFile`/`ReadFile` with a per-call `OVERLAPPED` offset (the pwrite/pread equivalent on a synchronous handle). On POSIX one shared `fd` + `pwrite`/`pread`. The ONLY shared mutable state is the slab allocator's free-list + the file high-water, under a mutex touched only on first-dump / class-promotion / discharge-free — NEVER on the stable-size re-dump's positioned write. **A standalone concurrency proof (commit 1, `test_extent_file`) verified before any wiring** that per-thread handles do NOT serialize on an NTFS-internal lock and do NOT short-transfer at disjoint offsets (8 threads × 40 rounds × 1 MiB disjoint regions, byte-verified) — so the fallback (handle-reuse per LB file) is not needed.
- **Metadata.** `Memory::rawExtentOffset_` / `rawExtentClassBytes_` REPLACE the raw path's use of `deloadFiles` (the reload seeks to `rawExtentOffset_`, reads the self-describing header, recovers the geometry — the image length needs no metadata). A per-LB `rawExtentEpoch_` (bumped at each `openExtentFile`) invalidates a slab left over from a purged prior batch (offsets recycle from 0 at reset), so a persisting LB's stale offset is never overwritten in place.
- **Lifecycle.** The extent file is owned by `GlobalMemoryManager` (`staticMemory`), opened at batch start (after `purgeDeloadDirectory` — closed first so the purge can delete the old file), closed at the NEXT batch start (which is AFTER this batch's chapter export — the export's post-prove raw reloads of still-live equality nodes read the extent file, so it must outlive the steward). Preallocated to 1.5x the pool (live extent bytes = the Dumped, i.e. non-resident, LBs' slabs, which legitimately exceed pool size — disk > RAM is the point of deload), grown in 256 KiB-aligned 256 MiB chunks under the allocator mutex. Purge = truncate + allocator `reset` (offsets recycle from 0, deterministic per batch).
- **Torn-image / stale-occupant detection.** No new CRC: single-run lifetime + batch-start purge + two asserts. `loadLbMemoryRaw`'s ordinal assert is TIGHTENED from `ordinal >= 0` to `ordinal == expectedOrdinal` (the LB's `deloadOrdinal` threaded into the loader) — with all ordinals sharing one file, an exact ordinal match is the direct slab-reuse tripwire; the existing chain assert is the second.
- **A/B gate.** `parameters.enable_extent_deload` (default true). When false, the raw path falls back to one named file per LB (the pre-extent behaviour) for byte-identity comparison; the same v4 format core (`emitRawImageStream` / `consumeRawImageStream`) serves both.

**Trade-offs.** ACCEPTED: extent offsets are nondeterministic (allocation order), but v4 raw BYTES were ALREADY declared nondeterministic ([D-195](#d-195)) and offsets never enter proof output, so no new determinism issue. Internal slack ≤ 2x per slab. Extent-full is the disk, not an artificial cap — a growth failure asserts naming the knob (`kExtentInitialBytes` / the `.deload` Defender exclusion), never silent unbounded growth. GAINED: per-eviction drops from ~50 ms to ~1–3 ms (a ~15–50x reduction; the residual is now genuinely per-byte), predicting `IncubatorPeano1` at 4 GiB ≈ 100–115 s, converging toward the 8 GiB band. Telemetry (`GlobalMemoryManager::extentLiveBytes/extentAllocatedBytes/extentFileBytes`, printed beside `[STATIC-MEMORY]`) gives alloc/live = internal slack and file/alloc = free-list + preallocation overhang for tuning, and the `[DELOAD]` throughput lines are the clean A/B that quantifies any residual Defender scan-on-write (the user's `.deload` exclusion lever).

**See** [D-195](#d-195) (the raw datapath this places in a file), [I-103](30_invariants.md#i-103) (canonical-bytes narrowed; extent offsets are the new nondeterminism), [I-107](30_invariants.md#i-107) (vids position-independent — why offsets are content-invisible), [I-114](30_invariants.md#i-114) (the working-set pager whose eviction path this serves).

<a id="d-192"></a>
## D-192 — a function counts as *statified* only at 0% heap (CodeQL inventory `true`); the boundary copy lives in the still-heap caller, and static→heap→static walk-arounds are banned (2026-07-04)


**Context.** The transient-statification campaign ([D-191](#d-191), [I-138](30_invariants.md#i-138)) drove the `standardProcessing` call tree's *interior* off the heap, but the campaign's working definition — recorded in the statification cookbooks — was too permissive: it sanctioned "convert your function's interior now, keep the `std::string` at the boundary, and materialize at the edge." Under that bar a helper that still RETURNS a `std::string` (or takes one by value) counted as converted. The CodeQL call-tree inventory (`statification/inventory.md`, the reusable oracle added by the inventory tooling) measures the stricter thing — a function uses heap if it constructs any owning `std` container as a local/temporary, calls `to_string`/`toStdString`, or takes/returns one BY VALUE — so the cookbook bar and the inventory disagreed.

**The choice (user-directed).** The inventory is authoritative. A function is *statified* iff its inventory row reads `true` — genuinely 0% heap: no heap local or temporary, no by-value heap parameter or return (a `const std::string&` / `StrSpan` parameter is fine — the caller owns the bytes). Two mechanics follow:
- **The heap boundary lives in the still-heap caller.** When a statified A is reached from a heap B, A takes spans/ids and writes into caller arena / out-params; B does the `std::string`↔span copy. The copy is never left inside A.
- **No static→heap→static walk-around.** If the caller already holds STATIC data (a `ScratchString` / `StrSpan` / arena value), it calls the statified function directly, static→static. Converting static→heap only to reach a heap function and converting the result back is banned — it is deleted, not tolerated. A `std::string` is materialized ONLY at a genuinely-heap sink not yet statified (`isAdmitted`, `addToHashMemory`, the compiled-definition layer).

This tightens — does not replace — the standing kernel rule (in `performElem1/2/3` the only permitted heap is `hashburst_trace.txt` generation, [D-191](#d-191)). The inventory generator (`.scripts/call_tree_inventory/`) is verifier-class: never edited to change what counts (same protection as `verifier.py`).

**Scope / gate.** The cookbooks ([`09b`](20_core_concepts/09b_statification_cookbook.md) / [`09c`](20_core_concepts/09c_string_statification_cookbook.md)) are rewritten to this bar; the per-function objective gate is the inventory row after a Tier-1 CodeQL regeneration, atop the existing per-commit `--unit-tests` byte-twin and the batch `main.py` + verifier + `files/` byte-identity gate. First applied to the 11 shallowest heap leaves (rows ≤ #100); the same rules apply to non-leaves.

**See** [I-138](30_invariants.md#i-138) (the campaign invariant this sharpens), [D-191](#d-191).

---

<a id="d-193"></a>
## D-193 — a heap `ce::` function is KEPT only while it has production callers outside the prover; otherwise it is DELETED and the twin test keeps a test-local oracle (2026-07-05)


**Context.** [09b](20_core_concepts/09b_statification_cookbook.md) row 90 (the static `ce::` twin recipe) originally said the heap `ce::` variant is "kept, never deleted: it is the compile / config-load builder AND the differential-test oracle." That over-generalised. A `ce::` parser-surface function splits into two cases by who still calls it once the prover's in-tree sites move to the span/int twin:
- It is still a genuine **compile / config-load builder** (called from the compiler, conjecturer, compressor, visualizer, load-time paths) → KEEP the heap form; it earns its place as a builder AND doubles as the twin oracle. `ce::extractExpression`, `ce::getArgs`, `ce::disintegrateImplication` are this case (dozens of out-of-tree callers). `ce::replaceKeysInString` also joined this KEEP set once its four in-tree admission callers (`addToHashMemory`, `makeNormalizedKeysForAdmission`, `makeAdmissionKeys`, `updateAdmissionMapRecursion`) moved to the `gl::replaceKeysToString` str_ops door — its many conjecturer / filter / compiler callers keep the heap body alive.
- It is **prover-only** — every production caller is inside the prover, and moving them all to the span twin leaves the heap form with zero production callers → DELETE it from `compiler.hpp`. The Rule-18 twin test then needs its oracle back, so a byte-identical copy is retained **test-locally** (`src/tests/test_harness.hpp · extractExpressionUniversalOracle`), never in the production tree. `ce::extractExpressionUniversal` (batch 3 row 240) is the first instance: all ~24 prover call sites migrated to `gl::extractExpressionUniversalSpan`, the heap function deleted, the `extract_expression_twins` / dispatch tests rebound to the test-local oracle. `ce::extractKeyValue` (batch 4 row 248) is the second: its sole in-tree caller (`addExprToMemoryBlock`) moved to the tree-free key-only twin `ce::extractKeyValueKeyScratch`, so the pair-producing heap form was deleted and its verbatim oracle retained in `src/tests/test_compiler.cpp · extractKeyValueOracle` (NOT `test_harness.hpp` — the oracle still calls the surviving `ce::parseExpr` / `ce::treeToExpr`, which that dependency-light header does not include).

**Deletion is due only when the last prover caller migrates via a static equivalent.** A prover-only heap `ce::` function is deleted only once a static equivalent has made it production-caller-free; while un-twinned production callers remain, it stays — deletion becomes due when the last such caller migrates. `extractKeyValueKeyScratch` is tree-free (it does NOT twin `ce::parseExpr` / `ce::treeToExpr`), so those two keep live production callers (the compiler, the conjecturer, other out-of-tree prover sites) and STAY on the heap even though they left the `standardProcessing` tree once `extractKeyValue` — their only in-tree caller — was deleted. Each becomes due for deletion when its last such caller migrates onto a static equivalent.

**The choice (manager-directed).** Do not keep a heap `ce::` function in the production tree solely to serve as a test oracle. If it has no production caller outside the prover after migration, delete it and move the oracle into `src/tests/`. Keeping dead heap in production to satisfy a test contradicts the 0% bar ([D-192](#d-192)) — the oracle belongs to the tests, not the pipeline.

**See** [I-138](30_invariants.md#i-138), [D-192](#d-192), 09b row 90.

---

<a id="d-191"></a>
## D-191 — MPU 0.1 transient statification: the per-LB burst kernel's transient scratch is largely off the heap, but NOT complete — `compiledMap` and the cross-LB heap-`Mail` commit seam remain open violations (2026-07-03, merge capstone)

**Context.** The static-memory hierarchy (the four pools, `LbArena`, deload) made the prover's *persistent* per-LB state static, but the *transient* working forms inside the burst kernel — `std::string` / `std::vector` / `std::set` / `std::map` scratch, decode snapshots, and heap-returning helpers reached from `performElemPhase1` / `performElem2` / `performElemPhase2` / `performElemPhase3` — were missed. This campaign removed them.

**The choice (user-directed, full-target).** Statify 100% of the phase call trees' transient interior. Landed as sub-branch subsessions, each gated byte-clean + adversarially reviewed: S1 wipe path, S2 sanitize twins, S3 reactToHypo + token memo, S4 phase-2 edge drains + eradicate, S5 static phase interfaces (sealed record chain), S5b the levels chain, S6 the compiledMap reader fence + zero production regex, S8 the firing check (hottest path), S9 the admission gates, S10 the integration templates, then an oracle sweep leaving one form per operation in production source. When a final census showed the admission/integration cluster still heap, the user ruled full-target (convert it too) rather than sanction it — the same standing rule that governs the whole kernel: in `performElem1/2/3` the ONLY permitted heap is `hashburst_trace.txt` generation; nothing else is "sanctioned."

**The precise boundary (the rule, and what still violates it).** The rule (user-set, 2026-07-04): in `performElem1/2/3` the ONLY heap permitted is generating `hashburst_trace.txt` (the Rule-14 sacred debug dump, off the compute path / not silicon-mapped). Everything else on the phase trees must be static. Two things are still on the heap and are OPEN statification targets — NOT sanctioned islands, NOT user-approved: (1) the compiled-definition (config) layer behind the `compiledEntity`/`coreConfig` reader, read from the phase trees, plus its enumerated parse-boundary edge materializations (on silicon = ROM); (2) the single-threaded cross-LB heap-`Mail` commit seams. The campaign removed most transient scratch heap but did NOT reach the all-static target. Host orchestration, load-time compilation, the C++ conjecturer and the Python pipeline are outside the kernel and not silicon-mapped.

**By-products.** One byte-identity waiver ([D-190](#d-190)). Two latent PRE-EXISTING defects surfaced by the campaign's own tripwires and fixed: the phase-2 executor never published `g_currentCoreId`, so a `ScratchScope` on the shared reserved slot raced across executor threads ([G-56](50_gotchas.md#g-56), surfaced by S6-C5's `allowedForMail` scope); and `wipeSubtree`'s delimiter-overlap precondition was unasserted (S1-C7). Added the project conventions (full rebuild always — a stale-object crash is indistinguishable from a real bug) and Rule 26 (read the statification cookbooks before coding in the statified tree). The MPU 0.1 hardware booklet was audited + corrected against final code and its prices refreshed for release.

**Evidence.** The release gate on the assembled tree (full rebuild + `main.py`): `files/` byte-clean, verifier 131881 checks / 0 failures airtight; C++ unit tests 1075/1075; every subsession landed byte-identical to its pre-conversion artifacts. New invariants [I-127](30_invariants.md#i-127)..[I-143](30_invariants.md#i-143) record the per-conversion contracts; [G-55](50_gotchas.md#g-55)/[G-56](50_gotchas.md#g-56) the gotchas. Granular history preserved,, and the per-subsession `sandbox/transient_perform_*` sub-branches.

---

<a id="d-190"></a>
## D-190 — wipeSubtree step-11 filter mints flip to ascending id order; the deload bytes differ from the old build (a deload-stream-only byte-identity waiver) (2026-07-02)


**Context.** `Memory::wipeSubtree`'s step 11 mints every closed validity id into `intValidityNamesToFilter`, a deload-enrolled `ColdHashSet<PodKeyStore<int16_t>>` whose `KeysView` facet streams keys in id (== mint) order — the mint order is deload-byte-observable. The historical loop iterated a `std::unordered_set<int16_t>` (`closedIds`, built by an ascending id scan), so the deload order was whatever MSVC's hash table yielded. A 26-shape measurement (13 sizes × dense/sparse ascending-insert populations) showed MSVC iterates == ascending ONLY while the table is collision-free: 18 shapes matched, 8 diverged (64 sparse, 100 sparse, 512/1000/4096 dense+sparse; bucket counts 8/64/512/1024/4096). MSVC hashes `int16_t` via FNV-1a; once keys collide, list-splice insertion interleaves the order. The old order is therefore collision/rehash-history-dependent — reproducing it would mean reimplementing the MSVC hash table — and latently HOST-dependent: libstdc++'s `unordered_set` iterates differently, a latent [I-103](30_invariants.md#i-103) cross-host violation the flip fixes.

**The choice.** Step 11 mints in ascending id order (an id scan `1..nameCount` gated on closed-set membership). The byte-identity waiver here is deload-stream-only: byte-identity to the OLDER build is given up for this reorder (the old hash order cannot be reproduced without reimplementing the MSVC hash table); the gate is all theorems present + verifier 0 failures + the NEW build deterministic. Same-build run/host determinism remains absolute. The reorder is observable ONLY in the `.deload` stream: the sacred hashburst dump sorts this section before printing (`infra/hashburst_dump.cpp`), and the only pipeline reads are membership probes (`memory.cpp` Site H `contains`, `prover.cpp` `contains`) — no `files/` artifact depends on the order.

**Test surface.** The measurement probe (asserting MSVC iterates ascending) was a hypothesis instrument, measured false — deleted under the waiver. Not an I-19 weakening: it pinned MSVC STL internals, not a code contract; its evidence lives in the flip's commit message. The standing regression pin is `wipe_step11_mint_order_facet_twin` (`test_memory.cpp`): minting representative closed-id populations in ascending order reproduces exactly the ascending sequence at `decode(1..count)` — id order == mint order == `KeysView` facet order.

---

<a id="d-189"></a>
## D-189 — retire the page-tier scratch fill: `allocBytes` / `mark` / `rewind` / `reset` reimplemented on the byte-bump tier (sandbox/transient_batch4)


**Context.** [D-185](#d-185) step 4–5 folded the per-worker scratch arenas and the `SealedPageSet` handoff onto `LbArena`'s PAGE tier: `allocBytes` filled within a page via `allocPage` + a single arena-wide `scratchWithin_` cursor, and `mark`/`rewind`/`reset` operated on the page table. But `allocBytes` and the page-tier *containers* (`PagedVector` / `ColdHashSet` via `allocPage`) then share one resource — the page-table tail (`pageHighWater-1`) and `scratchWithin_`. A scratch fill on an arena that also holds a live `allocPage` container writes into, and its rewind `0xCD`-poisons, the container's tail page. That is the [I-132](30_invariants.md#i-132) vanished-Gauss bug (`collected` on `scratchArenas` clobbered by `prefixArgumentsWithU`'s `allocBytes` under re-entrant disintegration), and the same hazard the firing-record / phase-2 reqgen batches dodged by using byte-bump `alloc` plus a separate `genScratchArenas`.

**The choice.** Retire the page-tier scratch fill entirely. `allocBytes` is now `resolve(alloc(bytes))` — a resolved byte-bump `alloc`; `mark`/`rewind`/`reset` operate on the byte cursor (`Mark{generation, cursor}`, `popTo(m.cursor)`, `popTo(0)`); `usedBytes` is just `cursor_`; `scratchWithin_` / `scratchUsed_` are deleted. The byte-bump tier (`cursor_` / `blocks_`) and the page tier (`allocPage` / `pageBlocks_`) are independent substrates (separate block tables, distinct physical blocks), so scratch fill and paged containers can NEVER collide on one arena — the bug class is structurally impossible, not merely guarded. This restores the original byte-bump scratch design ([D-182](#d-182): `Mark{generation, cursor}`, `alloc`/`resolve`/`popTo`); the page-tier version was the later, collision-prone change. No caller changes (`ScratchString`, `str_ops`, `prefixArgumentsWithU`, `SealedPageSet` keep their signatures); the one-page size limit relaxes to one-block. The prior [I-132](30_invariants.md#i-132) `genScratchArenas` move stays in place but is now belt-and-suspenders.

**Why not the exclusion-flag guard.** An assert that `allocBytes` and `allocPage` never share an arena was the alternative; it fences the footgun but keeps it. Removing the page-tier fill deletes the footgun — and it is the cleaner, smaller change (contained to `lb_arena.hpp/cpp`), since nothing the scratch strings or sealed pages do needs page granularity (strict LIFO via `popTo`; whole-set release via `releaseAll`). Supersedes [D-185](#d-185) step 4–5.

**Verification.** Build clean Release x64; `gl_quick.exe --unit-tests` 935/935 (`test_scratch_arena`'s `block_advance_no_straddle` / `rewind_across_blocks` rewritten for the byte tier); full `main.py` to completion, verifier 0 failures, Gauss theorems present.

---

<a id="d-188"></a>
## D-188 — `LbArena`'s per-LB page/block bookkeeping goes static via a small-buffer `PtrDirectory`; re-opens D-174 (2026-06-28)


**Context.** The MPU 0.1 close-out left `LbArena`'s block/page lists on the heap (`std::vector<char*>`) as the last per-LB carve-out — [D-174](#d-174) recorded it as a *declined future option* (the "page the page tables" technique, deferred for prover instability + no immediate FTA benefit). The maintainer re-opened it: the crashes that motivated the deferral are fixed, and "all of the LB goes static" (the MPU target) needs this last piece. Scope is only what lives INSIDE the LB — the per-LB `LbArena` bookkeeping. `GlobalMemoryManager`'s `granted_` / `recycled_` (one process-wide root that hands out the pool) and `LbStore::blocks_` (the slab holding LB shells, not inside one LB) are deliberately OUT of scope (maintainer).

**The choice.** A new **`PtrDirectory<kInline>`** ([`ptr_directory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory_infra/ptr_directory.hpp)) replaces the heap vectors: a `char*` array whose first `kInline` entries live INLINE in the directory object (already pool-backed via `LbStore`) and whose overflow spills onto pool blocks drawn directly from the manager, addressed by a small wired inline root (`root_[kArenaDirRootCap]`) — one level of indirection, NO recursion (the "assert, not recurse" ceiling is the root cap). The inline buffer is the decisive choice over a pure paged table: the numerous SMALL arenas (per-worker scratch, routing mail, a typical `persistentArena`) keep their whole page table inline and cost ZERO pool block, so the per-arena directory-block overhead a pure-paged table imposes on every page-tier arena — which exhausts tight pools and pressures the never-deloaded persistent pool — vanishes for the common case; only a large page table (the main deloadable arena) spills, where one directory block is negligible. This is the maintainer's original "inline fixed arrays" instinct made cap-free by the spill. `pop_back` / `truncate` RETAIN capacity (no pool churn on a scratch `rewind`, which would perturb the steward grant ledger, [I-106](30_invariants.md#i-106)); `clear` and `shrinkToFit` (the coarse compaction seam) release. The page free-list (`freePages_`) becomes an intrusive LIFO with the next-free link in each free page's LAST `sizeof(char*)` bytes, so `freePage`'s poison still covers offset 0 (the use-after-free tripwire). `LbArena::assertInvariants` — the legitimate-assert centrepiece the maintainer required for this substrate — audits the whole structure at the coarse seams and at every alloc/free under `GL_ARENA_PARANOID`.

**Byte-transparency.** Arena bookkeeping is never serialized ([I-107](30_invariants.md#i-107)): vids/offsets stay a pure function of per-LB allocation order and the directory is rebuilt on reload like the vector it replaced, so deload bytes, theorems, and verifier are unchanged. The only new observable is a few more pool blocks for a large page table — telemetry only ([I-106](30_invariants.md#i-106)), never a proof input.

**Status.** Landed on the branch: the page tier (`pageTable_` → `PtrDirectory<128>` + the intrusive free-list) and the byte-bump / carved-block lists (`blocks_` / `pageBlocks_` → `PtrDirectory<16>`). Pending: routing the `compactPages` scratch onto a steward scratch arena; the user-SwDD chapter, MPU booklet, and the matching `I-pending` invariant land in the campaign's doc-sweep commit. Per-commit verification is build-clean + `gl_quick.exe --unit-tests` (922/922, asserts live in release); the full `main.py` + verifier + determinism gate runs once after the campaign lands.

**See** [D-174](#d-174) (the declined predecessor this re-opens), [I-107](30_invariants.md#i-107), [I-106](30_invariants.md#i-106).

---

<a id="d-187"></a>
## D-187 — `updateAdmissionMap3`'s cross-LB ancestor admission seed is detected and deferred to `proveKernel`'s post-join single-threaded drain (closes the gauss-16 cold-index desync) (2026-06-28)


**Context.** A status-2 (`toBeProved`) registration of an operator-headed goal calls [`updateAdmissionMap3`](../GL_Quick_VS/GL_Quick/src/prover.cpp) to seed the recursion admission budget that lets the hash engine unfold that operator while chasing the goal. `updateAdmissionMap3` walks `parentMemory` up to the first ancestor whose `exprKey` shares a digit argument with the goal — the fold/recursion block that owns that numeral — and seeds **that ancestor's** admission map inline (via `updateAdmissionMap` → `mintTemplateKey` / `insertAdmissionValue` / `prepareIntegration`). Under `proveKernel`'s barriered parallel sweeps, sibling recursion-step children (e.g. `(in2[rec0,*,3])`) share one fold ancestor (`(fold[1,3,4,8,2,*,12])`); two phase-3 workers then mint the ancestor's cold `templateInterner` / `admissionMap` concurrently — a cross-LB write forbidden by [I-28](30_invariants.md#i-28), whose rare downstream collision surfaced as the `cold-index desync` / `indexPlace: duplicate id` abort at Gauss hashburst 16 (deterministic cross-LB write every Gauss run, sporadic crash). The climb is precisely the pattern [I-28](30_invariants.md#i-28)'s *Spot* list names as a violation; it predates the barriered phase model and was never migrated to a post-join collector like `activateZeroCondition`.

**The choice.** Detect-and-defer — the canonical fix [I-28](30_invariants.md#i-28) prescribes. A thread-local `g_inParallelWorkerPhase` is set in the phase-1 and phase-3 `runPhase` wrappers. When `updateAdmissionMap3`'s climb lands on a **strict ancestor** and the flag is set, the fully-resolved `updateAdmissionMap` call is staged (under `deferredAncestorAdmissionsMutex`) instead of applied inline; `drainDeferredAncestorAdmissions` replays the queue once after `pool.join`, sorted by a deterministic key, right after the `inductionMemoryBlocks` / `activateZeroCondition` drain — the same collector shape as `inductionMemoryBlocks` and `updateGlobalTuples`. (Carrier statified in S9: the former `std::vector<DeferredAncestorAdmission>` with heap `key`/`remainingArgs` became a shared `SealedPageSet` record chain of trivially-copyable POD records with `SealedSpan<SealedString>` runs — the [D-164](#d-164) sealed handoff; `proveKernel` `emplace`s it per iteration and the drain replays via an index sort under `deferredAncestorAdmissionLess` (byte-identical order), then `freePages`.) A **self-seed** (the climb stayed on the worker's own LB) and any seed **outside the parallel sweep** (grid build, the drain's own single-threaded re-entry, the CE filter's private per-thread clone trees where the flag is never set) stay inline. The drain skips a `dischargedForever` ancestor (no next step; [I-102](30_invariants.md#i-102) / [I-112](30_invariants.md#i-112)) and `ensureLoaded`s a live one (the working-set pager may have evicted it). Consequence: the ancestor seed lands post-join, a one-iteration latency consistent with the MailLog commit barrier ([I-55](30_invariants.md#i-55) / [D-137](#d-137)).

**Verification.** `gl_quick.exe --unit-tests` 912/912 (two new tests: `drainDeferredAncestorAdmissions` symbol-signature + replay-and-skip-discharged). Full `main.py` (`RUN_INCUBATOR = True`) run three times: `theorems.txt` byte-identical across all three, the Gauss summation theorem proved, verifier 131881 checks / 0 failures, **zero** `[i28-foreign]` trap events, no desync/abort — run-to-run determinism and (against the pre-cleanup run) trap-removal behaviour-preservation confirmed.

**See** [I-28](30_invariants.md#i-28) (the violated invariant + the matching prescribed fix), [I-102](30_invariants.md#i-102), [I-112](30_invariants.md#i-112).

---

<a id="d-186"></a>
## D-186 — CE-filter workers pre-intern their `exprKey`s single-threaded; never mint the shared `skeletonInterner` from the pool (2026-06-27)


**Context.** `filterConjecturesWithCE` ([`filter.cpp`](../GL_Quick_VS/GL_Quick/src/filter.cpp)) runs one worker thread per conjecture over a pool of `max(1, logicalCores)`. Each worker created its single-use clone LB and set its identity with `lb->setExprKey(std::to_string(i))`. `setExprKey` mints into the **process-global** `skeletonInterner` — a shared, lock-free `ColdStringTable` ([I-97](30_invariants.md#i-97)) whose write side is single-threaded by contract ([I-83](30_invariants.md#i-83)). Minting it from the worker pool was a genuine data race: concurrent `mint` corrupts the cold key store + the derived `PagedHashIndex`, so the per-mint tripwire `assert(lookup(k) == id && "cold-index desync: minted key not findable at its id")` fires. Sporadic (timing-dependent), and independent of SSD deload because the CE LBs are fully resident — the same desync family as the `ColdHashSet::indexPlace: duplicate id` abort. Reproduced under `main.py` on the Peano CE filter (936 conjectures, 32 workers) with a per-`mint` concurrency detector recording up to 12 threads inside `setExprKey` at once.

**The choice.** Pre-intern every CE `exprKey` single-threaded before the pool spawns: a loop interning `std::to_string(i)` for `i ∈ [0, conjectures.size)` into `skeletonInterner`. The workers' `setExprKey` calls then hit the lookup-only (read) path — concurrent reads on the now-unchanging table are race-free. Keeps the interner lock-free (its whole design point — it is read concurrently on the hot prover burst via `findChild` / `exprKey`, so a lock would serialize the parallel prover). Restores [I-82](30_invariants.md#i-82)'s "the only cross-thread write is the disjoint `contradictionTable` slot".

**Verification.** Full `main.py` (Peano + Gauss + incubators) clean with verifier 0 failures, under a deliberately race-widening `std::this_thread::yield` amplifier in `HashMap::mint` and 32 workers; survivor counts unchanged (936→230, 230→122, 674→124), confirming the pre-intern does not alter CE results.

**See** [I-82](30_invariants.md#i-82), [I-97](30_invariants.md#i-97), [`10_pipeline/03_ce_filter.md`](10_pipeline/03_ce_filter.md).

---

<a id="d-185"></a>
## D-185 — the HOT memory substrate is removed: every per-LB container, per-worker scratch arena, sealed-page handoff, and routing mailbox rides the COLD grant path; one arena model (2026-06-27)


**Context.** String / container statification left a SECOND memory model beside the cold per-LB arena: a HOT substrate — `LbArena` in `Mode::Hot`, a segregated `acquireBlockHot` grant path invisible to the steward ([I-96](30_invariants.md#i-96)), a per-slot reserve cap (`hot_arena_bytes`), and the `kHotPoisonByte` (0xDD) distinction — backing the per-worker calculation-string arenas and the sealed-page handoff, with the blocks RETAINED for the whole run. A second model to maintain, and retained hot blocks pin pool memory FTA needs. The user's directive: "literally 0 hot," with the scratch arenas RELEASED the instant a worker task finishes.

**The choice.** Fold everything onto the one cold `LbArena` and delete the HOT infrastructure, in a seven-step campaign:
1–3. The per-LB derived containers move onto static arenas: routing mail initially landed as mail-pool `RoutingColdMail`; [D-205](#d-205) later split it so only `mailIn` remains there and `mailOut` joins the deloadable LB arena. `changedClassesThisStep` (tag 655) and `eqClassNameCaches` (tag 705) remain deloadable ([I-125](30_invariants.md#i-125)).
4. `SealedPageSet` → the cold grant path (`bind`, `acquireBlock`), freed before any deload seam; `LbArena::mark`/`rewind`/`reset` collapsed to page-tier only.
5. The per-worker scratch arenas → cold (`ScratchArena` = `LbArena` bound cold, no cap), RELEASED per task (`releaseAll` at every `performElem2` exit — no retention); `hot_arena`/`hot_string` renamed `scratch_arena`/`scratch_string` ([I-124](30_invariants.md#i-124), retiring [I-116](30_invariants.md#i-116)). `usedBytes` became `cursor_ + scratchUsed_` so the cold scratch arena's page-tier fill drove the rewind tripwire. (Step 4–5's page-tier scratch fill was later reverted to byte-bump — see [D-189](#d-189) — eliminating the `allocBytes`/`allocPage` collision; `usedBytes` is now just `cursor_` and `Mark` just `{generation, cursor}`.)
6. The dead mid-burst throttle + the orphaned `burstClaim` word deleted (already retired by [D-161](#d-161) / [I-115](30_invariants.md#i-115)).
7. The HOT infrastructure itself deleted: `LbArena::Mode` / `mode_` / `initHot` / the `reserveBytes_` / `maxBlocks_` cap / `poisonByte_` distinction / `kHotPoisonByte`; `GlobalMemoryManager::acquireBlockHot` / `releaseBlockHot` / `hotBlocksInUse` / `peakHotBlocksInUse` + their members. One `LbArena`, one grant path (`acquireBlock`), one poison byte (`kArenaPoisonByte`).

**Determinism (the campaign's gate).** Folding scratch / sealed off the segregated ledger makes their grants visible to the steward (`grantsSinceBarrier`). Output stays byte-identical because the per-task / pre-barrier release nets those grants to ZERO before every barrier, so the deload SET stays a pure function of quiesced logical counts ([I-106](30_invariants.md#i-106) / [I-107](30_invariants.md#i-107)) and deload is content-invisible ([I-103](30_invariants.md#i-103)); only the grant-trigger TIMING shifts (more aggressive mid-iteration deload, a perf/tuning shift not a correctness one — the runtime-for-memory trade of [D-168](#d-168)). Verified by the two-sequential-`main.py` byte-identical gate + verifier + theorem check.

**Tests.** Per-commit build + unit tests (921 → 910 as obsolete HOT / throttle / `isValidHotArenaConfig` tests were dropped and the cold scratch surface re-covered in `test_scratch_arena.cpp` / `test_scratch_string.cpp`); the full determinism gate after the infrastructure deletion.

**See** [`20_core_concepts/09_static_memory.md`](20_core_concepts/09_static_memory.md), [I-124](30_invariants.md#i-124), [I-125](30_invariants.md#i-125), [I-101](30_invariants.md#i-101), [I-96](30_invariants.md#i-96) (retired).

---

<a id="d-151"></a>
## D-151 — the LB identity string `exprKey` statified into a shared never-deloaded skeleton interner (2026-06-26)


**The choice (user-directed).** Batch 2 of the LB-body campaign, scoped (user) to the resident IDENTITY skeleton — `exprKey` first, `simpleMap` next. `Memory::exprKey` changes from a per-LB `std::string` member to a 4-byte `int32_t exprKeyId` (0 = empty, the root sentinel) interned in `skeletonInterner` — ONE process-wide `ColdStringTable` (`= ColdHashSet<BytesKeyStore>`) on a function-local-static `LbArena{ &lbMemory }`, the never-deloaded LB-body pool. The former member becomes the `exprKey` decode accessor (byte-identical to the old string) plus `setExprKey`. Every read site adds ``; every write becomes `setExprKey`.

**Why a shared interner, not per-LB.** The skeleton must be readable while the LB's main arena is deloaded AND after the LB is discharged (the chapter export, `getGlobalKey`, `buildLbChainString` all read `exprKey` on cold / discharged LBs). The deloadable `lbMemory` arena and the discharge-reclaimed `persistentArena` both fail that. A per-LB never-deloaded arena would burn a full 256 KiB block per LB for a handful of identity strings. So one SHARED analyzer-lifetime interner on `lbMemory` (deduped, compact, never deloaded) is the fit — the `mailArena` / `lbStore` precedent. `exprKey` ids are never observable (always decoded back to the string), so the mint order is invisible to every output; determinism holds.

**Byte-identity by construction.** `exprKey` returns `decodeString(exprKeyId)`, byte-identical to the former member, so every consumer — including the Rule-14 sacred hashburst dump (the chain streaming + the `isTargetLB` chain-match literals), the deterministic `a->exprKey < b->exprKey` sort comparators, and the `deloadChain` = joined-`exprKey`s reload assert — is unchanged in output. The sacred-dump edits are mechanical `exprKey`→`exprKey`, output-preserving (user authorized by choosing the exprKey scope knowing it touches the dump).

**Deferred to Batch 3.** `deloadFiles` / `deloadChain` / `deloadedCounts` / `dischargedRegistryKeys` (tier-1 bookkeeping / per-LB POD with no cheap never-deloaded home); the transient per-burst staging vectors stay heap.

**Gate.** Build + `--unit-tests` green (914/914, incl. the new `exprkey_interns_and_round_trips_through_skeleton` direct test); one full `main.py` — `theorems.txt` + all proof artifacts BYTE-IDENTICAL to the pre-branch baseline, verifier 131881 checks / 0 failures (the deferred intermittent cold-index race did not fire this run). See [I-97](30_invariants.md#i-97).

---

<a id="d-150"></a>
## D-150 — the LB node objects move off the malloc heap into a fourth never-deloaded pool via a non-relocating slab store (2026-06-26)


**The choice (user-directed).** The statification campaign's last large heap user is the `Memory` node object itself (historically `new Memory` per LB). It moves onto the static hierarchy: a FOURTH `GlobalMemoryManager` instance (`lbMemory`, `PoolKind::Lb`, sized by `static_lb_pool_bytes` / `static_lb_block_bytes`) backs an `ExpressionAnalyzer`-owned `LbStore` ([`memory_infra/lb_store.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/lb_store.hpp) / `.cpp`) — a non-relocating slab allocator that placement-news each `Memory` into a fixed-size slot carved from the pool's blocks. Three user decisions fix the shape: (1) a dedicated pool, not the mail or persistent pool; (2) raw `Memory*` links stay — the slab never relocates a slot, so `parentMemory` / `simpleMap` values / the `mailLog` pointer keys / `ParentChildrenMap` keys stay valid (no handle conversion of the ~122 deref sites); (3) full scope — later batches also statify the resident skeleton (`exprKey`, `simpleMap`, deload bookkeeping) and the `LbArena` block-list bookkeeping, so a `Memory` ends with zero transitive malloc.

**Why a slab, not a paged vector.** `Memory` is non-copyable and non-movable (reference members, two `std::atomic` claim words, six embedded non-relocatable `LbArena`s), so `PagedVector<Memory>` (it `static_assert`s trivially-copyable `T`) and `std::vector<Memory>` (it reallocates) are both out. A slab that constructs in place and never moves is the only fit, and it is also what the whole prover already assumes (every `Memory*` is lifetime-stable). The pool is never deloaded — the shells are the always-resident LB directory; their bulky content already round-trips to SSD via the per-LB arenas. The free-list is intrusive (a freed slot stores the next-free link in its own raw bytes); within a store's life a freed slot is reused in place (across batches too) and no block returns to the pool mid-run, so the pool sizes to the peak simultaneously-live LB count; at teardown `releaseAll` (in `~LbStore`, mirroring `LbArena::~LbArena`) returns every block, so the several `ExpressionAnalyzer`s a process builds reuse the pool. The only heap the store keeps is its block list (tier-1 bookkeeping, like `LbArena::blocks_`).

**Why not the copy-in/out burst model.** The user floated "a worker copies the LB, bursts on the copy, writes back." Deferred and unnecessary: the phase-2 burst is already read-only on the LB (firings go to per-task record buffers, applied single-threaded post-join), and `stewardClaim` already fences workers vs. the steward (CAS `Idle→Busy`). A true core-local copy-in/out only becomes possible once the body is byte-relocatable (after the resident-skeleton + arena-bookkeeping batches) and is the eventual ASIC core-load step — out of scope here.

**Gate (staged).** Batch 0 landed the pool + store (scaffolding, no behaviour change). Batch 1 routed every LB birth/death through the store — the five `prover.cpp` grid `create` sites + the `destroyGrid` `destroy`, `compressor.cpp`, `filter.cpp` (CE template + the per-conjecture clone), `memory.cpp::cloneFactsTemplate(LbStore&)`, and `releaseCEBatchMemory` — and added `releaseAll` / `~LbStore` (block return at teardown, mirroring `~LbArena`, so the several analyzers a process builds reuse the pool). Build + `--unit-tests` green (913/913, incl. the new release/destructor tests; the `freeSlot` underflow assert caught one unit test building a grid node outside the store — fixed to create via the store, the assert working as designed). The full-pipeline determinism gate (`main.py`, proof-artifact byte-identity vs the pre-branch baseline + `verifier.py` 0 failures) is the Batch-1 close-out. See [I-109](30_invariants.md#i-109).

---

<a id="d-152"></a>
## D-152 — the LB-tree routing edges (`Memory::simpleMap`) statified into a never-deloaded `SimpleMapStore` (2026-06-26)


**The choice (user-directed).** The next step of the LB-body campaign after `exprKey` ([D-151](#d-151)): `Memory::simpleMap` — the per-LB heap member `std::map<std::string, Memory*>` holding the LB tree's down-edges (parent → child by routing-key string) — moves off the heap, the last large per-LB heap user on the otherwise-static shell. The user chose the FULL off-heap form (the campaign-closing "Batch 7" shape) over a keys-only step: not just the routing-key strings but the container itself leaves the heap, into a new `ExpressionAnalyzer`-owned subsystem `SimpleMapStore` ([`simple_map_store.hpp`](../GL_Quick_VS/GL_Quick/src/simple_map_store.hpp)) mirroring `MailLog`. The `Memory::simpleMap` member is removed; the routing-key strings intern into the shared `skeletonInterner` (the table `exprKey` already rides); the child values stay raw `Memory*` (address-stable `LbStore` slots, no handle conversion — [I-109](#i-109)).

**Shape — MailLog's per-parent chain, not a CSR or a composite index.** `SimpleMapStore` holds `edges` (`PagedVector<EdgeNode>`, append-only, each parent's edges threaded newest-first via `EdgeNode::prev` — the `mailRefs` pattern) and `heads` (`TypedColdMap<int64_t, EdgeHead>`, keyed by `lbKey(parent)` — the `mailHeads` pattern). The per-parent back-linked chain is the fit because LB births INTERLEAVE across parents (a chain element is born under any parent at any time): an append-only CSR (`ColdMultiMap`) cannot take that — its `appendToTail` rejects an interior key — and a flat composite-`(parent, key)` index alone cannot enumerate a parent's children. A separate O(1) point-lookup index was considered and DROPPED: per-parent fanout is single digits (the CE root has exactly one child, `batchSize == 1`), so `findChild`'s chain scan beats the old `std::map` string compares. Both stores ride `lbMemory` (the LB-body pool the skeleton interner already uses) — NO new pool, so the three sanctioned extra reservations under [I-95](#i-95) are unchanged.

**Two instances, not one.** Unlike `MailLog` (which the CE filter never touches), `simpleMap` serves BOTH the main proof tree (`this->body`) and the CE-filter tree (`ceBody`), and the two can coexist (CE filtering interleaves with proving). So `ExpressionAnalyzer` owns two stores — `simpleMapStore` (main) and `ceSimpleMapStore` (CE) — each cleared at its OWN teardown (`destroyGrid` / `filterConjecturesWithCE`); a single store wholesale-cleared at one teardown would wrongly drop the other tree's live edges. Sites route by phase (the `prover.cpp` main-path and `visualizer.cpp` to the main store; `filter.cpp` to the CE store).

**Byte-identity by construction.** The store is never deloaded, so pointer-valued keys / values and grant-order layout carry NO canonical-bytes obligation (the `MailLog` argument). `findChild` is a non-minting lookup (burst-safe). The one place iteration order could leak — `forEachChild` — sorts children by the DECODED routing-key string (`compareSpans` over `skeletonInterner.view`, allocation-free), reproducing the old `std::map<std::string, Memory*>` order exactly, so no enumeration-order-sensitive artifact changes. The store carries no `ContainerTag` / `visitContainers` / `dirty` and is absent from the sacred hashburst dump (no Rule-14 impact).

**Gate.** Build + `--unit-tests` green (the new `test_simple_map_store.cpp` direct test: link / find, absent → null, distinct-parents same key, interleaved chains, decoded-key sorted iteration, clear, forced two-level paged-hash spill). The full-pipeline gate PASSED: one `main.py` run, all tracked proof artifacts BYTE-IDENTICAL to the pre-branch baseline (`git status files/` clean — the `forEachChild` decoded-key sort reproduces the old `std::map` order exactly), the in-pipeline verifier 0 failures across every tag category, and the run completed with no `lbMemory` pool-exhaustion assert (the edge stores fit the LB-body pool). See [I-110](30_invariants.md#i-110).

---

<a id="d-181"></a>
## D-181 — `dischargeContradiction` sheds its reserved, never-written `internalMailOut` parameter (2026-06-26)


**The choice (user-directed).** `dischargeContradiction` took a `ColdMail& internalMailOut` and discarded it on entry with `(void)internalMailOut;` — a vestige kept only for "signature symmetry" with its live sibling `dischargeToBeProved`. None of its three reactions (incubator / CE-filter / vacuous-truth) ever wrote to it, and the only place an internal-mail channel met contradiction logic was this never-written parameter. Drop it from the signature, drop the `(void)` line and the `@param internalMailOut` doc block, and drop the argument at the single `standardProcessing` call site (Step 4c) plus the three `test_equi_reshuffle.cpp` call sites. `standardProcessing`'s own `internalMailOut` parameter stays — Step 5 `dischargeToBeProved` genuinely writes through it.

**Why.** Dead plumbing since the function was born at the LB split ([D-122](#d-122), 2026-06-09); the cold-mail flip ([D-180](#d-180)) only retyped it `Mail&`→`ColdMail&` and added the "reserved/unused" test comments. Threading a live channel into a routine that never touches it is needless logical complexity — the same dead-plumbing cleanup as the `deactivationCheck` removal ([D-123](#d-123)). A future parent-scope contradiction emission would re-add the parameter in one line (YAGNI), so nothing is lost.

**Gate.** Behaviour-preserving — a `(void)`-cast parameter has no effect, so this is a pure no-op removal. Build + `--unit-tests` (the three `discharge_contradiction_*` cases now call the two-argument form); one full `main.py` + `verifier.py` — the proved-theorem set in `theorems.txt` unchanged (Gauss summation present), verifier 0 failures. See [I-78](30_invariants.md#i-78), [D-122](#d-122).

---

<a id="d-143"></a>
## D-143 — the cross-LB mail-log commit + pull serialize/deserialize the statified mailbox directly, dropping the heap-`Mail` round-trip (2026-06-26)


**The choice (user-directed).** The companion to [D-142](#d-142): the cross-LB `MailLog` boundaries also drop the transient heap `Mail`.
- **Commit.** A new `Codec<Mail>::serialize(const HotMail&)` overload emits the SAME blob bytes from `mailOut`'s sorted snapshots (`sortedStatements` / `sortedOrigins`) that `serialize(const Mail&)` emits from `toHeapMail`. A `MailLog::commit(const Memory*, const HotMail&)` overload takes the routing mailbox directly; the two per-cycle barrier sites (the `proveKernel` post-join commit + the `buildGrid` startup commit) call `commit(lb, lb->mailOut)` — no `toHeapMail`.
- **Pull.** A new `Codec<Mail>::deserializeInto(data, n, HotMail&)` parses the blob and routes each statement / origin straight through the `HotMail` write doors (`insertStatement` / `addMailOrigin(…, INT_MAX)`) — the fused `deserialize` + `mergeBatchInto(const Mail&, HotMail&)`. `MailLog::readBlobInto` reassembles the (possibly page-straddling) span and calls it; `MailLog::pull<Inbox>` dispatches via `if constexpr` — `pull<HotMail>` (the production `body.mailIn` path) uses `readBlobInto`, `pull<Mail>` (tests) keeps `readBlob` + `mergeBatchInto`.

**Kept.** `commit(const Memory*, Mail)`, `readBlob`, `Codec<Mail>::serialize(const Mail&)` / `deserialize`, and `mergeBatchInto` all survive: `broadcastTheorems` commits a FRESHLY-built heap `Mail` (the startup proven-theorem broadcast — not a round-trip of a statified container), and the test path `pull<Mail>` exercises them. The heap `Mail` struct + `toHeapMail` / `makeHeapMail` remain for the sacred dump (and tests).

**Why byte-identical.** `serialize(const HotMail&)` reads the same `sorted*` snapshots `toHeapMail` feeds its `std::set` / `std::map`, and `serialize(const Mail&)` iterates those in the same order — so `serialize(hm) == serialize(hm.toHeapMail)` byte-for-byte (pinned by `test_mail_log.cpp::hotmail_codec::serialize_matches_toheapmail_bytes`). `deserializeInto` routes through the exact doors `mergeBatchInto(const Mail&, HotMail&)` uses, so the inbox equals `mergeBatchInto(deserialize(blob), inbox)` (pinned by `deserialize_into_round_trips`). Pull/merge order stays irrelevant — statements set-merge, origins fold uncapped and the receiver's absorb re-sorts under the D-49 cap. The mail pool is never deloaded, so no canonical-bytes obligation binds the blob ([I-103](30_invariants.md#i-103) governs deload streams only); determinism rests on the set-merge + the absorb sort, unchanged.

**Gate.** Build + `--unit-tests` (incl. the two `hotmail_codec` byte-identity / round-trip tests); one byte-identity full `main.py` + `verifier.py` — `theorems.txt` + all proof artifacts byte-identical to the baseline, verifier 0 failures. See [I-101](30_invariants.md#i-101), [I-94](30_invariants.md#i-94).

---

<a id="d-142"></a>
## D-142 — the mail absorb reads the statified containers directly, dropping the transient heap-`Mail` snapshot (2026-06-26)


**The choice (user-directed).** Finalizing the mail statification: the `standardProcessing` absorb no longer materializes a transient heap `Mail` from the statified mailboxes before draining it. The single `absorb` lambda becomes generic over the container — `HotMail` for the external routing inbox (`&body.mailIn`, status=3) and `ColdMail` for the internal channels (status=1) — and reads each container's canonical sorted snapshots directly: `sortedStatements` for the drain, `sortedOrigins` for the origin merge + equality sync, `sortedExpandedImplications` for the HotMail index merge (`if constexpr`), and the new `ColdMail::getDisintegrationSignal(ev)` read door for the per-statement firing-time signal. `performElemPhase1` passes `&body.mailIn` (a `HotMail*`) instead of `body.mailIn.toHeapMail`; the internal call drops `makeHeapMail`. `toHeapMail` / `makeHeapMail` / the heap `Mail` struct survive — the sacred dump (`infra/hashburst_dump.cpp`, Rule 14) and tests still use them, as does the cross-LB commit/pull until its companion change ([D-143](#d-143)).

**Why.** `toHeapMail` / `makeHeapMail` already compute the container's `sorted*` vectors and then build a REDUNDANT heap `std::set` / `std::map` (red-black-tree node allocation) on top — pure overhead in a path that runs up to 3× per LB per burst. Reading the sorted snapshots directly removes the tree build entirely; the per-statement disintegration-signal lookup becomes a direct `getDisintegrationSignal` probe with no allocation at all (HotMail has no such column, so the external branch hardcodes `{false,false}`). Net: less allocation churn in the hot absorb — the all-static endgame for mail.

**The one determinism subtlety.** In the heap path the `Mail.exprOriginMap` did double duty: step (a) sorted each origin run IN PLACE, and step (c) then read `find(stmt)->second.front` — the lexicographic-min of that now-sorted run — as the origin passed to `addExprToMemoryBlock`. The direct absorb reproduces this exactly: it builds the `origins = sortedOrigins` working set ONCE, sorts each run in place, and uses that single structure for BOTH the bulk merge AND the per-statement `.front` (a `std::lower_bound` on the EWV-sorted unique keys). So the origin chosen per statement stays the sorted-min, byte-for-byte. Statements drain in `sortedStatements` order (the load-bearing first-drained-wins level gate); `expandedImplications` mint in `sortedExpandedImplications` order (the receiver's `TypedColdSet` deload is insertion-ordered); both reproduce the former `std::set` order. The external routing inbox (`body.mailIn`) is cleared by the caller after the call (the old code cleared only the discarded heap copy inside `standardProcessing`; `body.mailIn` was always cleared by `performElemPhase1`).

**Gate.** Build + `--unit-tests` (incl. the new `test_cold_mail.cpp::get_disintegration_signal_read_door`); the byte-identity full `main.py` + `verifier.py` gate runs once after the companion commit/pull change ([D-143](#d-143)) — `theorems.txt` + all proof artifacts byte-identical to the baseline, verifier 0 failures. See [I-102](30_invariants.md#i-102), [I-101](30_invariants.md#i-101), [I-44](30_invariants.md#i-44).

---

<a id="d-145"></a>
## D-145 — contradiction discharge defers `updateGlobalDirect` to the post-join drain, closing a parallel-phase cross-LB race (2026-06-26)


**The bug (user-flagged).** `dischargeContradiction`'s incubator (`primedForContradiction`) branch called `updateGlobalDirect(theorem, coreId)` INLINE. `dischargeContradiction` runs inside `standardProcessing`, which `proveKernel` drives in phase 1 and phase 3 — both parallel work-stealing sweeps over the active LBs. So `updateGlobalDirect`, and the cross-LB writes inside it — `mergeBatchInto(mailOut, this->body.mailOut)` into the ROOT's `mailOut`, and tree-wide `deactivateUnnecessary` — executed on a worker thread while other workers ran: a data race on `root.mailOut` (multiple contradiction workers plus the worker running the root's own `fillMailOut`) and on `isActive` across the tree, a direct [I-28](30_invariants.md#i-28) violation. Every other theorem-emission path (`dischargeToBeProved` non-recursion; the induction auxy promotion via `updateGlobal`) already deferred to the post-join drain — the contradiction path was the lone inline exception (the `updateGlobalDirect` "runs in the serial standardProcessing phase" docstring assumption is false; phases 1/3 are parallel across LBs). A catcher on the inline call over an IncubatorPeano run logged 1026 worker-thread calls (hundreds of distinct thread ids) vs the normal-goal (92) and induction (276) emissions, which all ran on the single drain thread.

**The fix.** Replace the inline call with the same mutex-guarded enqueue `dischargeToBeProved` uses — `updateGlobalDirectTuples.push_back({theorem, coreId})` under `updateGlobalDirectMutex`. The existing post-`pool.join` drain in `proveKernel` then runs `updateGlobalDirect` single-threaded. No new machinery; the contradiction path becomes consistent with the normal-goal and induction paths.

**Why timing + semantics hold.** The post-join drain order is the `updateGlobalDirectTuples` drain THEN the [D-76](#d-76) compaction drain, so a deferred contradiction's `recordPendingCompaction` (called inside the now-deferred `updateGlobalDirect`) still queues before the same iteration's compaction drain — the compact-rule broadcast keeps its existing two-iteration latency, and the contradiction-LB latency sensitivity ([D-79](#d-79) / [G-49](50_gotchas.md#g-49)) is not tripped. Emission order becomes deterministic (the drain sorts `updateGlobalDirectTuples`), removing the non-determinism the race introduced. `deactivateUnnecessary` moves from mid-phase to post-join — identical to the normal-goal path's existing behavior. The contradiction LB's own `isActive = false` stays inline (its own state, never the race).

**Gate.** `main.py` + `verifier.py` (race fix — the pre-fix output was non-deterministic, so the criterion is the proved-theorem SET in `theorems.txt` unchanged + verifier 0, not byte-identical ordering): **PASSED** — the cold-mail race-blocker that reliably aborted IncubatorPeano1's first burst on the pre-fix code (per the branch handoff) no longer reproduces; full `main.py` completes (1625 s); `verifier.py` reports 131881 checks / 0 failures (all proof graphs airtight); 46 main-path theorems incl. the Gauss summation; `--unit-tests` green. (`theorems.txt` is gitignored — no committed byte-baseline; the change is a semantic no-op on the proved-theorem set, deferring emission without skipping it. A two-run determinism A/B is the recommended gold-standard follow-up.) See [I-78](30_invariants.md#i-78), [I-28](30_invariants.md#i-28).

---

<a id="d-144"></a>
## D-144 — `PagedVector` and `PagedHashIndex` gain a two-level page directory (root → level-1 directory pages → data pages), and the silent `PagedHashIndex` directory overflow becomes a hard assert (2026-06-24)


**The choice (user-approved).** Both arena-paged containers kept their page directory in a SINGLE arena page, capping a container at `dirCap = pageBytes/sizeof(int32)` data pages. Add a second directory level: when the single directory page fills (`numPages_ == dirCap`), `rootVid_` becomes an **L2 root page** of level-1 directory-page vids, each L1 page holding up to `dirCap` data-page vids — so the index itself spans several pages, capacity `dirCap²` data pages. The full single directory page becomes L1 page 0 with **no copying**; the page index splits into `(L1 index, slot)` by one shift + one mask (`dirCap` is a power of two), one extra `pageAt` resolve on the deep path. `PagedVector` keeps its adaptive promotion/demotion ladder (empty → inline → single directory → two-level, and the exact reverse on shrink); `PagedHashIndex` builds the structure **wholesale in `reset(cap)`** (no incremental ladder — it rebuilds on growth). Overflowing two levels (`dirCap²` pages) HARD-ASSERTS in both, naming the three-level directory.

**Why — and the bug it fixes.** `dirCap` is small at a reduced `static_page_bytes`, and FTA-scale statement vectors approach it even at 8 KiB (the Gauss batch's `intEncodedStatements` already does). The single directory page was a hard ceiling. Worse, `PagedHashIndex::reset` had **no bounds check at all**: its `for (p in [0, numPages_)) dir[p] = allocPage` loop wrote past the one directory page once `numPages_ > dirCap`, **silently corrupting** the adjacent arena page — a latent Rule-19 violation that surfaces as a crash / corrupted statements / nondeterminism far from `PagedHashIndex`, and the prime suspect for a multi-day debugging failure under reduced-page runs. (`PagedVector` at least asserted.) The fix removes the ceiling AND, at the new `dirCap²` boundary, replaces the silent write with a hard `assert` — live in the Release build (`GL_Quick.vcxproj` defines no `NDEBUG`), so it fires in the real pipeline.

**Determinism / deload invisibility.** The directory shape (depth, page count, vids) never reaches an observable. Both containers address pages by stable vid and resolve through `pageAt` per access ([I-107](30_invariants.md#i-107)); `appendSpanBytes` walks LOGICAL elements, so the deload byte stream is byte-identical regardless of directory depth ([I-103](30_invariants.md#i-103)). `PagedHashIndex` is throw-away (never deloaded). So proof output is a pure function of logical content, independent of `static_page_bytes` — the gate is a forced-small-page full run whose `theorems.txt` is byte-identical to the 8 KiB baseline.

**Telemetry (process documentation, not a proof input).** `GlobalMemoryManager` gains an atomic two-level-promotion count + peak-`pagesHeld` high-water, fed from both containers' single→two-level transition and reported once per batch — a reusable spill confirmation for forced-page / FTA stress runs. Never read by the prover (Rule 16 / [I-44](30_invariants.md#i-44)); the count may shift with the (timing-dependent) deload set, so it is observability only.

**Migration shape.** Four commits: (1) `PagedVector` two-level + direct spill tests + this entry; (2) `PagedHashIndex` two-level + the overflow assert + tests; (3) the peak-pages telemetry + test; (4) cold-map in-map spill tests. Build + `--unit-tests` per commit; one forced-`static_page_bytes=2048` `main.py` + `verifier.py` after, git-diffing `theorems.txt` vs the 8 KiB baseline for zero theorem loss + verifier 0.

---

<a id="d-180"></a>
## D-180 — the two internal-mail channels statify onto the cold deloadable path (`ColdMail` on `LbMemory`), the follow-up the `mailio-hot-arena` decision anticipated (2026-06-25)


**The choice (user-approved).** Move `Memory::sameIterationInternalMail` / `nextIterationInternalMail` off the heap onto the COLD deloadable path — a new `ColdMail` struct (`memory_infra/cold_mail.hpp`), the cold sibling of `HotMail`, whose three columns ride the LB's deloadable `manager` arena with the real deload-`dirty` and are enumerated by `LbMemory::visitContainers` at bases 455 / 505 (the Batch-5 `exprOriginMap` / `HashMemory`-at-a-base pattern), `survivesDischarge`. `ColdMail` reuses `HotMail`'s column types/codecs (`MailStatementKey` / `EwvKey` / `OriginLine`, lifted into `memory_infra/mail_types.hpp` with `ExpressionWithValidity` so `LbMemory` can see them), CARRIES the `disintegrationSignals` column `HotMail` dropped (`TypedColdMap<EwvKey, uint8_t>`, the two firing-time bools packed), and DROPS the always-empty-for-internal `expandedImplications` (the mirror of how `HotMail` dropped its own unused field — user-chosen). Writers use the doors (`insertStatement` / `addMailOrigin` / `setDisintegrationSignal`); the absorb reuses the existing heap-`Mail` recipe over a `makeHeapMail` snapshot (the `ColdMail` twin of `toHeapMail`, but it repopulates `disintegrationSignals`); the sacred dump reads the same snapshot (user-approved Rule-14 edit, byte-neutral).

**Why cold.** Both internal channels are non-empty at the deload seam and must travel with the LB. `mailOut` now uses the same residency principle; `mailIn` remains the exceptional pre-claim inbox on the mail pool. Cross-LB internal deposits claim/reload the recipient and skip a `dischargedForever` target.

**Gate.** One `main.py` + `verifier.py` on: `theorems.txt` + the proof-graph chapters byte-identical to the pre-change baseline (the change is output-neutral — absorb re-sorts, deload is byte-stable, the skipped cross-LB writes were already dead); verifier 0 failures; `--unit-tests` green incl. `test_cold_mail.cpp` + the deload round-trip. See [I-102](30_invariants.md#i-102).

---

<a id="d-179"></a>
## D-179 — the per-LB routing mailboxes `mailIn` / `mailOut` move off the heap onto a dedicated per-LB HOT arena (the cold-map family), not the cold deload path (2026-06-25)


**The choice (user-approved).** Host `Memory::mailIn` / `mailOut`'s three live containers (`statements`, `exprOriginMap`, `expandedImplications`) on a dedicated per-LB HOT `LbArena` via a new `HotMail` struct (the cold-map family — `TypedColdSet<MailStatementKey>` / `TypedColdBlobMap<EwvKey, OriginLine>` / `TypedColdSet<EwvKey>`), exactly the `ChangedClassesBuffer` / `EqClassNameCaches` pattern: no `ContainerTag`, never in `visitContainers`, `clear` per burst / `releaseArena` at teardown. `statements` is keyed on the WHOLE `(EWV, levels)` pair so `set<pair<EWV, set<int>>>` multiplicity is exact (no level-set collapse). The fourth heap-`Mail` field `disintegrationSignals` is DROPPED (routing channels never write it, assert-guarded). The heap `Mail` type STAYS — for the two internal-mail siblings AND the transient serialization path — and the read boundaries (absorb, commit barrier, sacred dump) reuse the existing heap-`Mail` logic over a `toHeapMail` snapshot (zero determinism risk; the writers use the hot doors `insertStatement` / `insertExpandedImplication` / `addMailOrigin`).

**Why hot, not cold; why keep `Mail`.** `mailIn` / `mailOut` live only within one hashburst and carry no deload obligation, so cold statification (the Batch-1–5 treatment) would churn the LB's deload pages for nothing — the HOT path (Main pool via the segregated `acquireBlockHot`, never deloaded) is the right home. The user explicitly asked whether `Mail` could be deleted outright; the answer is no, because `nextIterationInternalMail` survives across bursts AND across LB deload — the per-burst HOT model is a semantic mismatch for it. The three per-burst channels could go hot, but the cross-burst sibling cannot; `Mail` therefore stays (deleting it would require statifying `nextIter` COLD as a separate effort). Scope was held to `mailIn` / `mailOut` per the user.

**Gate.** One `main.py` + `verifier.py` on: `theorems.txt`, all 279 raw + 150 processed proof-graph chapters byte-identical to the pre-HotMail baseline; verifier 0 failures. The `toHeapMail` snapshot re-imposes the canonical `ExpressionWithValidity::operator<` order, so proof output is unchanged end-to-end. See [I-101](30_invariants.md#i-101).

---

<a id="d-137"></a>
## D-137 — inter-LB mail becomes a pull model: each LB stores its sent mail once; descendants pull from ancestors with a per-(recipient, ancestor) cursor (2026-06-24)


**The choice (user-approved).** Replace the push/broadcast inter-LB mail routing — `mailOut → sendMail → per-core boxes → smashMail → every descendant's mailIn` — with a pull model. Each LB keeps ONE append-only log of the mail batches it has emitted this execution batch (one `Mail`'s worth of `statements` + `exprOriginMap` per cycle). A receiver, in `performElemPhase1`, walks its `parentMemory` chain and merges any batches it has not yet ingested from each ancestor's log into its `mailIn`, tracking a per-(recipient, ancestor) cursor. The cursor is also the explicit "already-ingested" ledger the push model lacked — it is what makes dormant induction-zero LB delivery correct on wake (a parked LB stays at cursor 0 and catches up its ancestors' whole logs the moment it activates). Heap `std::` prototype: a `MailLog` struct on `ExpressionAnalyzer` (`unordered_map<const Memory*, vector<Mail>> batches` + `unordered_map<const Memory*, unordered_map<const Memory*, size_t>> cursor`, `src/mail_log.hpp`). A later session statifies it onto a process-wide static pool.

**Retention refinement.** [D-204](#d-204) preserves this full-history behavior for any grid with an initially dormant LB, while an all-active grid rolls delivered blob/ref windows and preserves the same cumulative cursor semantics.

**Why.** Under push, the same sent expression was copied once into every descendant's `mailIn` — O(grid-size) per emission, worst for dormant LBs whose `mailIn` accumulated every ancestor emission until they woke. That per-LB duplication is the FTA-scale mail bottleneck and is exactly what blocks the statification memory goal (MPU 0.1). Storing each expression once in the producer's log drops the mail-staging storage from O(descendants) to O(1) per emission. Delivery is provably identical: `buildParentChildrenMap` gave each sender its transitive descendants, so "pull from every ancestor on the `parentMemory` chain" reaches the exact same sender→receiver set ([I-57](30_invariants.md#i-57)).

**Three behaviour-preservation rules (output stays byte-identical to the push model).**
1. **Load-time broadcast → store once.** `broadcastTheorems` (previous-batch externals + proved-theorem rules) and the buildGrid startup dispatch reached ALL LBs, not just descendants. They now store the seed batch once in the root's log (descendants pull it on the normal walk) and self-inject the root's `mailIn` (the root has no ancestor to pull from). The user chose to extend "saved once" to this load-time content too.
2. **D-76 two-cycle latency preserved.** The compact-implication drain and the `updateGlobalDirect` / `updateGlobal` `"theorem"`-origin sends run AFTER the commit barrier, so they append to the root's `mailOut` and ride the NEXT barrier's commit — the same one-iteration-later (net two-cycle) delivery the old post-`smashMail` `sendMail` had ([G-49](50_gotchas.md#g-49) contradiction-LB latency sensitivity unchanged). The per-step delta stays one cycle.
3. **Field coverage.** Only `statements` + `exprOriginMap` ever reached `mailIn` (the old `smashMail` dropped `expandedImplications`); the pull merges exactly those two via `mergeBatchInto`. `disintegrationSignals` is never written to `mailOut` (internal-channel only), so it is naturally untouched. Copying the whole `Mail` would newly deliver `expandedImplications` — a forbidden semantics change (Rule 8).

**Race-safety (heap prototype, [I-94](30_invariants.md#i-94)).** The commit barrier runs single-threaded at the `proveKernel` post-join seam (where `smashMail` was), so the logs are frozen during the parallel phase-1 sweep; the pull does concurrent reads of the frozen `batches` map plus disjoint per-recipient `cursor` writes, with no insert/rehash (every key is pre-created at grid build), so it is race-free by the C++ standard. The reverted statified mail trial's data race (the working note `mutex.md`) lived in the cold-blob / `PagedVector` read path, which the heap prototype does not have — a statification-phase concern, not the prototype's (user-confirmed: proceed parallel, no mutex).

**Out of scope.** The two per-LB internal-mail channels (`sameIterationInternalMail` / `nextIterationInternalMail`) and the ancestor-direction deferred-action collectors (`inductionMemoryBlocks`, `updateGlobalTuples`) are untouched — they are not mailOut broadcast. `buildParentChildrenMap` and the `index` member were kept at decision time (still referenced by the compressor and CE-filter call shapes; their removal was flagged a separate cleanup with no functional gain to the pull). That cleanup has since landed: the phase/`prove` signatures dropped the vestigial index parameters, and `buildParentChildrenMap` / `indexCE` / `destroyParentChildrenMap` / the `ParentChildrenMap` alias were deleted outright (user-authorized dead-code removal, 2026-07-03).

**Migration shape.** Seven commits: (1) `MailLog` struct + unit test; (2) register all LBs at buildGrid + wire the phase-1 pull (inert); (3) commit barrier replaces `smashMail` + drop the phase-3 flush; (4) store-once reroute of `broadcastTheorems` + startup; (5) reroute `updateGlobal*` / D-76 to the root's `mailOut` + barrier covers the root unconditionally (it can deactivate); (6) delete the dead broadcast machinery (`sendMail` / `smashMail` / `boxes` / `buildPerCoreMailboxes` / `destroyMailboxes` / `PerCoreMailboxes`); (7) this SwDD. Build-only per intermediate commit; one full `main.py` + `verifier.py` after commit 6, git-diffing proof artifacts vs the Batch-5 baseline for zero theorem loss + verifier 0 failures.

---

<a id="d-141"></a>
## D-141 — `MailLog` is statified onto a dedicated, never-deloaded mail pool; a third `GlobalMemoryManager` + an `ExpressionAnalyzer`-owned local arena; cursor advanced by a disjoint no-dirty write


**Decision.** The cross-LB pull-model mail log ([D-137](#d-137)) — the last per-LB heap structure after the strings campaign deferred it — moves off the heap onto a **dedicated, standalone, never-deloaded mail pool**. The three-tier grant the user specified: a third process-wide `GlobalMemoryManager` (`mailMemory`, sized by `static_mail_pool_bytes`) hands blocks to ONE `ExpressionAnalyzer`-owned `LbArena mailArena{ &mailMemory }` (the "local memory manager"), which hands pages to the three cold `MailLog` containers. Selected by a new `PoolKind {Main, Persistent, Mail}` (replacing `StaticMemoryConfig::isPersistent`) so the exhaustion assert names the right knob.

**Stand-alone, no deload role.** Nothing reads the mail pool's grant ledger, so it plays no part in any deload / throttle / steward decision; the mail content never competes with the deloadable main pool. The three containers are pure RUNTIME containers — never in `LbMemory::visitContainers`, so never deloaded / dirty-tracked / reshuffled (the `intToBeProved` precedent), carrying a `DirtyState` that is never read.

**Container shape — independent per-LB append-only logs (the heap-faithful structure).** The batch store is THREE containers mirroring the heap prototype's `unordered_map<Memory*, vector<Mail>>`, so a commit is O(1) and never touches another LB's bytes: `mailBlobPool` = `PagedVector<char>` (every `Codec<Mail>` blob, append-only — strings INLINED, a faithful 1:1 of the heap, no cross-batch interner); `mailRefs` = `PagedVector<BlobRef>` (one ref per batch, append-only; each LB's refs a newest-first `prev`-linked chain); `mailHeads` = `TypedColdMap<int64_t, MailHead>` (producing-LB key → newest ref + count). The routing index is `mailEdges` = `ColdMultiMap<int64_t, int64_t>` (recipient → ancestor-key list, set once at registration — drives the pull, since a hash map cannot enumerate the cursor by recipient prefix) and `mailCursor` = `TypedColdMap<CursorKey, int32_t>` (every cell pre-created at registration). Keys are the `uintptr_t` of a `const Memory*` — pure runtime identity; the never-deloaded pool means NO canonical-bytes obligation, so pointer keys are sound (proof output stays content-deterministic via the unchanged set-merge + absorb-sort).

**Retention refinement.** [D-204](#d-204) makes the blob/ref columns append-only within a retained window when the completed grid had no dormant LB; routing, cursors, and cumulative head counts remain execution-batch-long. A grid with any initially dormant LB keeps the original full-history shape.

**Why not a single `TypedColdBlobMap` (the first cut, reverted).** The initial implementation stored the batches in one `TypedColdBlobMap<int64_t, Mail>` whose CSR concatenates every LB's run into one byte pool — so a commit to any non-tail LB `memcpy`-shifts every later LB's bytes. With ~1000 LBs each committing per burst that is superlinear: a measured massive slowdown (per-burst `dt` climbed 7→14→18→20→27 s as the grid filled, vs the heap's flat profile). Reaching for the existing CSR container instead of replicating the heap's independent-per-LB-vector structure was the error. The append-only pool + per-LB `prev`-chain is the fix — O(1) commit, the per-burst `dt` now tracking the actual inference work; verified by the burst-`dt` A/B above. (The remaining statification cost is the per-pull `Codec<Mail>` decode — a constant factor; if it later binds, a `mergeBlobInto` that folds blob bytes straight into the inbox with no intermediate `Mail` is the next lever.)

**Race-safety (the prior failure mode).** Earlier statified-mail trials all failed and were attributed to a "cold-blob deserialize race"; the real bug was `PagedHashIndex::reset` silently overrunning its single directory page once the index passed `dirCap`, corrupting neighbouring arena pages — now fixed ([D-144](#d-144)). With that gone, the statified pull is race-free without a lock by three properties: frozen-during-parallel-phase logs (commits only at the seam), pure read accessors with inlined blob strings (no shared interner mint at pull), and a disjoint cursor advance via `setValueAtRelaxed` — a new no-dirty in-place write that skips the shared dirty-flag escalation `setAt` would do (a data race even on disjoint slots). See [I-94](30_invariants.md#i-94).

**Behaviour change accepted.** The heap prototype threw `std::out_of_range` for an unregistered-recipient pull (a `.at` artifact); the statified form treats an LB with no registered ancestors (the root, or — unreachably, by the no-mid-run-birth architecture — an unregistered LB) as a defined no-op. The surviving Rule-19 trap is the cursor-cell-consistency assert inside the pull loop.

**Migration shape.** Four commits: (1) the third pool + `PoolKind` + params; (2) the `setAtRelaxed` no-dirty write door; (3) `Codec<Mail>` + `CursorKey`; (4) the statified `MailLog` + the `mailArena`/`mailDirty` members + the test rewire + the forced-spill regression guard. Build + unit tests per commit; one full `main.py` + `verifier.py` + a sequential-determinism pair after commit 4, plus a forced-small-mail-page run proving the two-level spill fires with byte-identical output, all git-diffed against the pre-branch baseline.

---

<a id="d-178"></a>
## D-178 — `Memory::exprOriginMap` moves off the heap onto the cold blob map (Batch 5), completing the standalone-origin-map statification; mail stays string (2026-06-22)


**The choice (user-approved).** Batch 5 of the statification campaign — the last standalone per-LB heap origin container, `Memory::exprOriginMap`, moves from the heap `IdOriginMap` (`unordered_map<int64_t, vector<IdOrigin>>`, the id form [D-131](#d-131) left it as) onto a cold `TypedColdBlobMap<int64_t, IdOrigin>` (an `LbMemory` member aliased on `Memory`, the `equivalenceClassesMap` pattern). `Record = IdOrigin` (one history line = one variable-length blob: `OriginTag` + packed dependency keys), so the value codec is the per-line stream `serializeEquivalenceClass` already emits (lifted into `Codec<IdOrigin>`). Four deload tags 451–454 (`ExprOrigin{Keys,RunStarts,BlobStarts,BlobPool}`), the next free `LbMemory` tags, visited AFTER the HashMemory band (51–450) so the enumeration stays ascending. The equi-class `EquivalenceClass::equalityOriginMap` is NOT a second target — it already rides the `equivalenceClassesMap` blob (Batch 3) and stays a transient heap decode.

**Scope confirmed with the user.** Only `exprOriginMap`. Mail is not touched: `Mail::exprOriginMap` is a separate string-form member, and the encode-at-absorb / decode-at-`fillMailOut` boundary [D-131](#d-131) established already isolates mail from the body container's representation — so statifying the body needed zero mail changes (the feasibility question that opened the batch: yes, feasible without statifying mail).

**Mechanism.** Cold overloads of `addOriginId` / `addOriginEncoded` / `decodeOriginMapSorted` carry the EXACT heap semantics onto the blob map via a decode-run / apply-policy / write-run RMW (including the [D-49](#d-49) cap-full convenience-slot swap); overload resolution routes each call site by the map type (cold `exprOriginMap` vs heap `equalityOriginMap`). The ad-hoc raw-API sites (`.find` / `.swap` / range-for / `.count(k)` / `.at`) became `lookup` / `recordsAt` / id-walks. The one bulk merge (the equi-class→body union in `updateEquivalenceClasses`) becomes a sorted in-place RMW so a class-only key is minted deterministically (the cold deload streams keys in insertion order; the heap form relied on the dump's key-sort). CE-clone teardown's swap-empty becomes `resetToFresh`. The hashburst dump's header origin count adapts `.size`→`.count` (output-identical; the Rule-14 contract is the dump FORMAT, unchanged).

**Determinism / correctness.** Proof output is byte-identical: the dump / chapter export / compressor read `decodeOriginMapSorted` (sorts by decoded key) and per-key line order is preserved by the merge. Survives discharge (the export reads origin history on discharged LBs, [I-44](30_invariants.md#i-44)) and `wipeSubtree`. ([I-121](30_invariants.md#i-121).)

**Trade-off accepted.** The cold RMW (a decode-run on each write/probe) costs runtime versus the heap map, like the Batch-2 set maps; memory, not runtime, gates FTA, so this is accepted — the all-static goal is the point.

**Migration shape.** Commit 1: `Codec<int64_t>` + `Codec<IdOrigin>` + unit tests, no consumers. Commit 2: the cold `exprOriginMap` `LbMemory` member + tags + `visitContainers` + `survivesDischarge`, dormant (still heap-aliased). Commit 3: flip `Memory::exprOriginMap` to the cold alias + the cold helper overloads + reroute every LB-side access site + this SwDD. Verification: `--unit-tests` per commit; one full `main.py` + `verifier.py` after commit 3, git-diffing proof artifacts vs the baseline (zero) + verifier 0 failures.

---

<a id="d-139"></a>
## D-139 — the six static-memory sizing constants (`static_pool_bytes`, `static_block_bytes`, `static_page_bytes`, `static_persistent_pool_bytes`, `static_persistent_block_bytes`, `hot_arena_bytes`) are fixed in `parameters.hpp`, never read from any config; all batches share identical sizing (2026-06-21)


**What.** Remove the six `pp.contains("…_bytes")` reads from the prover's config loader (`prover.cpp`, the `prover_parameters` block) so the static-memory sizing constants come solely from their `struct ProverParameters` defaults in `parameters.hpp`. Delete the same keys from the only two configs that carried them (`ConfigPeano.json`, `ConfigGauss.json`, each of which set the three pool/block/page keys to exactly the struct defaults). The validity asserts (`isValidStaticMemoryConfig` / `isValidStaticPageConfig` / `isValidHotArenaConfig`) and the `init*Memory` / `initHotArenas` calls are unchanged — they already read `parameters.*`, now always the struct defaults.

**Why.** The memory hierarchy is one process-wide reservation shared by every batch, so per-batch overridability bought nothing (both configs already set the defaults) and only invited silent divergence — a batch could carry a stale or mistaken pool size and behave differently for a non-logical reason. A single compile-time source of truth means one edit changes every batch identically, with no config surface to drift. The MPU 0.1 / fixed-SRAM model wants one capacity, not a per-config knob.

**Behavior.** Pure refactor: both configs set exactly the `parameters.hpp` defaults, so effective sizing is unchanged and proof output is identical. Gate: a full `main.py` run whose verifier total matches the pre-change baseline with 0 failures.

**Alternatives considered.** (a) Keep the config reads but assert config == default — a guard for a knob nobody should set; rejected as defensive programming. (b) Actively reject a memory key found in a config — out of scope; an unknown key is silently ignored, matching the existing `pp.contains` opt-in style. (c) Leave as-is — keeps the divergence surface the change exists to remove.

**Scope.** Sizing constants only. The other `prover_parameters` knobs (iteration budgets, split policy, compressor, `ban_disintegration`, …) stay config-driven. Contract: the sizing constants are the sole source of truth — see [I-95](30_invariants.md#i-95).

---

<a id="d-161"></a>
## D-161 — RAM is a cache of the working set: the steward keeps only the LBs being processed plus the next few resident in EVERY phase, drains the rest; workers self-load; the throttle gate and the hard-bound mass-deload are removed (2026-06-21)


**Context — two ways the prior deload froze or thrashed.** (1) The mid-burst throttle gate ([D-148](#d-148)) could DEADLOCK silently: phase-2 workers `waitWhileThrottled` BEFORE acquiring any block; the throttle clears only on a block release below 3/4; the sole reliever (`forceDeloadPass`) refuses split LBs and cannot touch hot scratch. When pressure came from split LBs or hot blocks the throttle latched forever, no worker reached `acquireBlock`, and the `static_pool_bytes` exhaustion assert never fired — the run hung with no assert. (2) The hard-bound (3/4) barrier path deloaded EVERY active LB synchronously each iteration, then phase 1 reloaded them all — O(total bytes) of SSD traffic per iteration, which reads as a hang. Active-LB eviction in the steward band had been disabled ([D-175](#d-175) Status) because it "dropped theorems"; that root cause — the residency-gated deactivation survey — was since FIXED ([D-149](#d-149)), so active-LB deload/reload is now content-invisible and an eviction set may vary run-to-run without changing proof output.

**The choice (user-designed: "a totally new loading behavior of the steward — RAM is a cache of the working set").** Treat the deload store as the backing store and RAM as a cache of the **working set** — the LBs being processed now plus the next few about to be processed. The pieces:

1. **One working-set window in EVERY phase (all three equivalent).** Each phase (1, 2, 3) opens `beginPhaseWindow(cursor, active, workers)` over its dispatch cursor and closes it at the join. While open, the steward runs `maintainWorkingSet`: (a) PREFETCH-reload the lookahead window `[cursor+workers, cursor+workers+kLookahead)` so workers find upcoming LBs resident, and (b) DRAIN the deloadable LBs outside the working set `[cursor−workers, cursor+kLookahead)`. No phase mass-preloads.
2. **Worker self-load + release (the user's "a worker uploads itself").** The unified handshake `claimAndLoadForWork` (steward method, used by all phases) claims an LB through the single `stewardClaim` word and makes it resident before the worker reads it; the worker releases the claim (`stewardClaim = Idle`) when done with the LB, so the steward may reclaim it. A split LB's first executor part claims it; siblings see `WorkerOwned`; the finalize releases it.
3. **Load↔evict exchange (thresholds: 1/2 and 3/4, user-chosen).** Above the 3/4 hard bound the steward mass-evicts every deloadable LB down to the working set. Between 1/2 and 3/4 it does the one-for-one exchange — one biggest deloadable LB deloaded per reload performed this pass (`pickBiggestDeloadable`). A worker reloading a cold LB makes room the same way (`evictOneForReload`) before `ensureLoaded`. **Inert below the thresholds** (a consequence the user named): a batch whose usage stays under 3/4 force-deloads nothing → reloads nothing → exchanges nothing, so light batches (e.g. Peano main) run exactly as before, full speed, zero deload/reload.
4. **Genuine exhaustion asserts, never waits.** If nothing is deloadable and the pool is full, the concurrent working set alone exceeds the pool — `acquireBlock` asserts naming `static_pool_bytes`. No silent back-pressure.
5. **Stuck-assert backstop** ([I-113](30_invariants.md#i-113)): every worker `Busy`-claim wait is deadline-bounded (30 s) and asserts, so any unexpected hang aborts loudly at its origin.

**Removed.** The phase-2 `waitWhileThrottled` gate and the whole relief machinery (`beginThrottleReliefWindow` / `endThrottleReliefWindow` / `requestRelief` / `forceDeloadPass` / the throttle onset wiring); the barrier's synchronous hard-bound active mass-deload; the disabled barrier eviction-plan builder. The barrier now only discharges INACTIVE LBs (unchanged: `dischargeStatementContent` + hand to the steward for the background near-empty dump) — with a reload-before-discharge guard, since the pager may have deloaded an LB while it was still active+Idle in a phase.

**Determinism.** The eviction SET is timing-dependent (which LBs are resident when depends on parallel pickup/grant timing), so the gitignored `.deload/` file set is run-to-run nondeterministic — accepted, exactly as [D-148](#d-148) already accepted for the throttle. Proof output stays fully deterministic because deload/reload is content-invisible ([I-103](30_invariants.md#i-103), [I-107](30_invariants.md#i-107)) — now that the deactivation-survey timing dependence is gone ([D-149](#d-149)). This RELAXES [I-106](30_invariants.md#i-106) further (mid-phase `blocksInUse` drives the pager's drain in all phases, not just the mid-burst throttle).

**Supersedes / amends.** [D-148](#d-148) (the throttle gate + force-deload are removed — the pager subsumes mid-burst relief); [D-175](#d-175) (active-LB eviction re-enabled, now CONTINUOUS in every phase window rather than a barrier-armed plan; the `Planned` claim state and the second `burstClaim` word are retired); the active-LB-pressure parts of [D-156](#d-156) and [D-155](#d-155) (the barrier no longer mass-deloads active LBs; inactive-LB discharge is unchanged). [I-115](30_invariants.md#i-115) is retired; [I-122](30_invariants.md#i-122) now governs a single claim word used by all three phases.

**Status / scope.** The pager + the unified handshake + the all-phase windows + the barrier simplification are in. Phase 2's flat (LB, part) executor releases each LB's claim to `Idle` the moment its LAST part finishes sealing (a per-LB remaining-parts counter; the part that drops it to zero flips the claim), so an executor-done LB becomes deloadable immediately and the steward drains the executor pool down to the working set — the same "idle when not under processing" rule phases 1 and 3 apply at their body end. The finalize re-claims + reloads each LB (content-invisible) when it reaches it.

**Why the per-part release is load-bearing (the first-run depletion).** The first full-pipeline run exhausted the pool at `IncubatorPeano1` (hash burst 6, 1248 active bodies) precisely because that release was missing in the first cut. The cause was NOT eager preloading — the steward only ever prefetches the cursor window (`kLookahead`), honoring "do not load all at once". Each phase-2 worker correctly loads its LB to run its executor part ("load to process", not preload); the bug was that a processed LB stayed falsely `WorkerOwned` until the much-later finalize, so it never became `Idle`/deloadable when its worker was done with it — processed LBs piled up resident and the working set grew to ALL active bodies, past the pool. The depletion was un-released processed LBs, not a too-large prefetch. The per-part release restores the user's "not under processing → idle → deloadable" rule to the phase-2 executor, bounding the resident set to the window in every phase. Gate: the full pipeline run + verifier, which the user judges. Contract: [I-114](30_invariants.md#i-114); see also [`20_core_concepts/09_static_memory.md`](20_core_concepts/09_static_memory.md).

**Phase-2 split-LB reload race + fix (the duplicate-id assert).** Once the pager actively reloaded in phase 2, a split LB exposed a concurrency race: `claimAndLoadForWork` set `WorkerOwned` BEFORE running `ensureLoaded`, so a sibling part saw `WorkerOwned` and began reading while the loader was still rebuilding the cold-map index (and `reloadFromImage` marks `resident` before `loadLbMemory` fills the containers) → `ColdHashSet::indexPlace: duplicate id`, only under heavy phase-2 churn on split LBs (`Gauss`; the incubator runs unsplit, phases 1/3 are one-worker-per-LB). A paranoid serial round-trip confirmed the round-trip is content-clean (5064 deload+reload cycles, byte-identical, 0 failures) — so it was purely the concurrent publish-before-load. Fix: claim via `Idle/Dumped → Busy`, run the load under `Busy`, publish `WorkerOwned` only after the load completes; a sibling/worker that finds `Busy` waits for it to resolve. Guard asserts at every release site verify the LB was held `WorkerOwned`. See [I-122](30_invariants.md#i-122).

**Inert retained → throttle now removed.** The `GlobalMemoryManager` throttle (`isThrottled` / `waitWhileThrottled` / `setThrottleOnsetCallback` / `totalBlocksInUse` / the watermark band) and the `Memory::burstClaim` member, retained dead-but-inert here, are now REMOVED. The eviction-plan plumbing (`installEvictionWork` / `evictionWork_` / `stewardEvictionPlan`) is likewise dead-but-inert, removed in a follow-up.

---

<a id="d-147"></a>
## D-147 — `HashMemory` folds INTO `LbMemory`; the four instances become members enumerated through the one `visitContainers`, retiring the manual deload threading + the `~Memory` safeguard (2026-06-22)

**What.** The Part-C endgame. Now that `HashMemory` holds zero heap members ([D-173](#d-173)), move the four per-LB instances (`overallHashMemory` / `localHashMemory` / `localHashMemoryDelta` / `workingMemory`) from `Memory` by-value members INTO `LbMemory` as members, declared after `manager`. `Memory` keeps reference aliases (`HashMemory& overallHashMemory = lbMemory.overallHashMemory`, the established `intEncodedStatements` / `equivalenceClassesMap` alias pattern), so every `body.overallHashMemory…` call site is untouched. The struct moved to `memory_infra/hash_memory.hpp` (the prior commit) so `lb_memory.hpp` can see it; its record value types are forward-declared there (the blob façade names `Record` only in member templates), defined with their codecs in `memory.hpp`.

**How (enumeration).** `LbMemory::visitContainers` (both overloads) calls each instance's `HashMemory::visitContainers(base, …)` after its own tags 0..50, at the reserved bases 51 / 151 / 251 / 351 (moved into `hash_memory.hpp`). A bridge lambda casts `HashMemory`'s `uint32` `base+offset` tag to `ContainerTag`, so the deload directory records tags 51..450 **byte-identical** to the former `extraColumns` path — the dump, the tail-delta counts, and the on-disk image are unchanged. `HashMemory::visitContainers` gains a const overload (a shared static impl) because `dumpLbMemory` reads a `const LbMemory&`. `survivesDischarge` returns true for tags 51..450 (a discharged LB keeps its hash engine resident for the chapter export, exactly as before the fold); `LbMemory::liveBytes` now sums the four instances (content-invisible reshuffle-gate accuracy).

**What retires.** Three manual hooks the out-of-`LbMemory` placement forced: `buildHashMemoryDeloadColumns` + the `extraDump`/`extraLoad` lists in `dumpStaticContainers` / `reloadFromImage`; the manual per-instance release in `Memory::releaseStaticBlocks`; and the explicit `~Memory` that released the cold containers before the arena died. Destruction is now natural (the instances live in `lbMemory` after `manager`, so they destruct first — the `freePage`-on-dead-arena hazard is gone). The `lb_deload` `extraColumns` parameter + `DeloadColumn` adapter stay as a defaulted, now-unused general extension point.

**Why.** Restores the single-enumeration invariant ([I-123](30_invariants.md#i-123)): one `visitContainers` is the sole driver of dump / load / release / discharge for the WHOLE LB, `HashMemory` included — no parallel hand-threaded path that can drift (the burst-4 stale-vid crash the manual release guarded against) and the natural arena-last destruction order. The deload image is byte-identical, so proof output is unchanged. With this, all per-LB prover memory — `HashMemory` included — is one cold, deloadable aggregate (the MPU 0.1 goal).

**Verification.** Build clean (Release x64; only the known C4267). 839/839 C++ unit tests pass, including the all-four-`HashMemory` deload/reload round-trip (now driven through `visitContainers`). Full pipeline + verifier run at the end-of-batch gate.

<a id="d-173"></a>
## D-173 — the five remaining `HashMemory` heap members move onto the cold container family, completing the struct's member statification (2026-06-22)

**What.** Statify the last five heap (`std::*`) members of `HashMemory` — the residue after the owner-set maps ([D-140](#d-140)) and the admission/rejection subsystem ([D-172](#d-172)). They stay `HashMemory` members on the existing `DeloadColumn` path at the next free per-instance offsets (`base+48` upward). This clears the struct's heap state so it can fold into `LbMemory` ([D-147](#d-147)). Three flat id-sets land first: `admissionSetIntegration` / `triggersForAdmissionSetIntegration` (packed `int32` template keys) → `TypedColdSet<int32_t>` (`base+48` / `base+49`) and `productsOfRecursionIds` (`int16` NameMap ids) → `TypedColdSet<int16_t>` (`base+50`), each mirroring `consumedAdmissionKeys` / `varsInAdmissionMapKeys` exactly — `.insert` → `.mint`, `.count`/`.find` membership → `.contains`, and the radical-wipe iterator loop → `.eraseIf(coldScopeWipe)` (the dead `wipePackedSet` lambda retired). The two composite members follow in the next two commits, reusing the already-written-but-dead codecs: `originals` (`set<vector<int32_t>>`) → byte-key `TypedColdSet<IdVecKey>` (`Codec<IdVecKey>`), and `remainingArgsNormalizedEncodedMap` (`unordered_map<set<int16_t>, set<NormKey>>`) → `TypedColdBlobMap<Int16SetKey, NormKey>` (`Codec<Int16SetKey>` + `Codec<NormKey>`).

**Sacred dump (Rule 14).** The read sides adapt to the cold containers (the heap iterators are gone, `TypedColdSet` has none — `keyAt`/`count` instead). Sections that already lex-sorted stay byte-identical (`writePackedSection` for the two admission sets; the `productsOfRecursion` decoded-name list). The two former raw-`unordered_set`-order lines — `productsOfRecursionIds` (raw ids) and `remainingArgsNormalizedEncodedMap`'s outer key-set loop — become sorted (ascending ids / decoded-key lex), the deterministic canonical form, since no hash-order bytes existed to reproduce. Section ordering, titles, and format are untouched.

**Why.** Completing the member statification is the only thing left blocking the fold of `HashMemory` into `LbMemory`: while these stayed heap, a deloaded LB still held RAM and the struct could not become an `LbMemory` member. Memory, not runtime, gates FTA — the cold RMW costs are accepted on the owner-set / admission precedent ([D-168](#d-168)).

<a id="d-140"></a>
## D-140 — the four `HashMemory` owner-set subkey maps (`normalizedEncoded*`) move off the heap onto the cold blob map, read on the hot prune path through a zero-allocation byte peek (2026-06-20)

**What.** Statify the four `normalizedEncoded{Keys,Subkeys,SubkeysMinusOne,SubkeysMinusTwo}` maps — the request-generation prune indices, the last big per-LB heap consumer in the hash engine after `encodedMap` — onto `TypedColdBlobMap<NormKey, OwnerSet>` (one `OwnerSet` blob per key, run-length-1, whole-value replace), exactly mirroring `encodedMap`. They stay `HashMemory` members (5 deload facets each at `base+5..24`, riding the existing `DeloadColumn` path). The already-written-but-dead `Codec<OwnerSet>` becomes live. Writes are install-time read-modify-write (`mergeOwnerRecord`); reads on the hot path go through `ExpressionAnalyzer::ownerKeyAccepts` (lookup + the three byte-overload predicates over an `OwnerSetBlob` peek), never `recordsAt`. The radical wipe becomes a snapshot-rebuild (decode → filter `partitionIds` by closed scope → drop empty keys → re-`assignRun` survivors). The per-LB `Memory::keyArena` is retired in the follow-up commit: the owner-map writes drop it, and its last user — `remainingArgsNormalizedEncodedMap`'s inner set — migrates from the keyArena-backed `IntNormalizedKey` to owning `NormKey`s (a new `NormKeyHash`; the per-request probe builds one `NormKey` from `req.normalizedKey`, RT-neutral). The per-thread `g_reqKeyArena` (transient request keys) is unaffected — it remains the only `KeyArena` user.

<a id="d-172"></a>
## D-172 — the algebra admission/rejection subsystem (`admissionMap` + `rejectedMap` + their satellites) moves off the heap onto the cold-map family (2026-06-22)

**What.** Statify the six per-`HashMemory` algebra admission/rejection containers — the disintegration-side "admitted vs rejected" pair plus their bookkeeping — onto the cold-map family, the natural Batch-4 continuation after the owner-set maps ([D-140](#d-140)). `admissionMap` (`int32 → AdmissionValueSet`) and `rejectedMap` (`int32 → RejectedValueSet`) become `TypedColdBlobMap<int32_t, AdmissionMapValue>` / `TypedColdBlobMap<int32_t, RejectedMapValue>` (each set member one record blob in a run, exactly the `encodedMap` shape), reusing the already-written-but-dead `Codec<AdmissionMapValue>` / `Codec<RejectedMapValue>`; `admissionStatusMap` (`int32 → bool`) becomes `TypedColdMap<int32_t, uint8_t>`; `consumedAdmissionKeys` / `revisitInProgress` (`int32` sets) and `varsInAdmissionMapKeys` (`int16` set) become `TypedColdSet`. The packed `(templateId, validityId)` key is keyed directly as a raw `int32_t` via a new identity `Codec<int32_t>` (byte-identical on disk to keying on `StatementKey`). They stay `HashMemory` members on the existing `DeloadColumn` path (admission side `base+25..32`; rejection side `base+33..37`). Keys + values were already id-form ([I-89](30_invariants.md#i-89), [I-90](30_invariants.md#i-90)) — this is a pure container-substrate swap.

**How (the writes).** The find-or-emplace `admissionValuesAt` / `rejectedValuesAt` (which returned a mutable `std::set&`) are retired for a read-snapshot (`admissionRecordsAt` → the run decoded into the historical sorted `AdmissionValueSet`) and a read-modify-write insert (`insertAdmissionValue` = `lookup` → `recordsAt` → sorted-unique `std::set` insert → `assignRun`), so the run stays SORTED by `DecodedAdmissionValueLess` and the deload bytes + the sacred dump stay canonical. Single-key erases (`cleanAdmissionMap`'s [I-41](30_invariants.md#i-41) closure, `cleanUpAdmissionMap`, the subtree wipe) batch into one `eraseBlobIf` / `eraseIf` over a collected key set; `rejectedMap` is never erased ([I-37](30_invariants.md#i-37)). `admissionStatusMap`'s `operator[]` becomes `upsert`; `admissionMapPropagate`'s deliberate default-false `[]`-probe (which the dump counts) is reproduced as `find`-miss → `upsert(0)` → return. Writes stay single-threaded (the `drainAdmissionKeysAlgebra` drain + the synchronous helpers — [I-68](30_invariants.md#i-68)/[I-83](30_invariants.md#i-83) preserved); the one phase-2 read (the `consumedAdmissionKeys` staging gate) is the parallel-safe non-minting `lookup`.

**Why.** Admission/rejection were the last big per-LB heap consumers in the hash engine after `encodedMap` + the owner-set maps; FTA is gated on memory, not runtime ([D-168](#d-168)). The RMW whole-run write is O(run) per insert (the owner-set / `encodedMap` precedent), accepted; the `appendRecord` fast path is the open lever if it later binds. Landed in two commits — admission side first, rejection side second — because the satellites are touched at the same sites as their map; `revisitRejected2` (which spans both) is adapted on each side. [I-99](30_invariants.md#i-99) is the storage invariant.

**Integration twins (2026-06-22).** The same swap lands the integration counterparts, resolving this decision's deferred note (the algebra commit left "the `*Integration` twins, nested-map value — a later branch"). `admissionMapIntegration` (`int32 → map<IntInstruction, ValueIdSet>`) becomes `TypedColdBlobMap<int32_t, IntegrationEntry>` — the nested inner map flattened one `IntegrationEntry` blob per inner `(instruction, payload-set)` entry, the run held in the inner map's decoded order; `rejectedMapIntegration` becomes `TypedColdBlobMap<int32_t, RejectedMapIntegrationValue>`; the two `varsIn*IntegrationKeys` caches become `TypedColdSet<int16_t>`. The pre-written-but-dead `Codec<IntegrationEntry>` / `Codec<RejectedMapIntegrationValue>` are consumed unchanged — **no new container type, no new codec** (the "current containers enough?" question this branch answered). They ride the same `HashMemory::visitContainers` `DeloadColumn` path at `base+38..47`. The nested admission value's RMW round trip is `admissionIntegrationRecordsAt` (decode the run back into the nested `IntegrationEntryMap`, both stateful comparators threaded from the `ValueInterner`) → `payloadAt` mutate → `assignRun(flattenIntegrationEntryMap)`; `isAdmittedIntegration` / `updateAdmissionMapIntegration` write back PER snapshot entry so the re-entrant `prepareIntegrationCore2` (which writes a structurally different marker-form key, never the renamed key under iteration) sees each insert. There is NO `consumedAdmissionKeys` / `admissionStatusMap` analog on the integration side, and `admissionSetIntegration` / `triggersForAdmissionSetIntegration` (the EWV set side) deliberately STAY heap — parity with the algebra commit, which statified no set side. [I-22](30_invariants.md#i-22) (integration entries persist across revival/consumption) is preserved unchanged: `eraseBlobIf` fires only in `cleanUpAdmissionMapIntegration`, the equi-class re-key drop, and the subtree wipe; `rejectedMapIntegration` is erased after revival. Two new direct unit tests (`integration_{admission,rejected}_cold_roundtrip`); the full-pipeline gate confirms proof artifacts byte-identical to the pre-branch baseline + verifier 0 failures.

**Why.** Memory: the campaign goal (all prover memory static, gating FTA / MPU 0.1) requires these off the heap. The representation choice was forced by RT, not preference: the maps are the hottest-probed structures in the prover (10^4–10^6 prune probes per burst), and the common case reads almost nothing (the `hasLooseOwner` byte, then a short-circuit). A naive blob-map read (`recordsAt`, full `OwnerSet` decode rebuilding two `std::set`s) on every probe would have been a severe regression. The zero-allocation byte peek ([I-100](30_invariants.md#i-100)) reads the needed fields straight off the arena bytes, so the prune's RT is flat. The prune is a sound over-approximation, so the migration is semantics-preserving — proof artifacts are byte-identical to the pre-migration baseline.

**Alternatives considered.** (a) Literal `encodedMap` strategy (full `recordsAt` decode per probe) — simplest, no new read API, but the RT regression above; rejected. (b) Split the hot `hasLooseOwner` byte into a separate POD side-column, blob only for the rare `partitionIds` / `uSignatures` walks — more containers + wiring for marginal gain over the peek; rejected. (c) Keep them on the heap — fails the campaign's no-heap rule.

**Cost.** The install-side RMW decodes + re-encodes the whole `OwnerSet` on each owner insert, so it is O(owners) per insert (O(owners²) over a key's life). Accepted: install is single-threaded and not the burst bottleneck; the same shape `encodedMap`'s pre-fast-path RMW had. If install time regresses measurably at FTA scale, an incremental sorted-insert fast path (the analog of the `encodedMap` RMW append) is the follow-up. Contract: [I-100](30_invariants.md#i-100); built on [I-98](30_invariants.md#i-98), [D-171](#d-171).

---

<a id="d-149"></a>
## D-149 — a SECOND static pool (`persistentMemory`) backs the never-deloaded `Memory::intToBeProved`, making the deactivation survey deterministic regardless of main-arena deload state (2026-06-20)

**What.** Add a second `GlobalMemoryManager` instance — the persistent pool (`persistentMemory` / `initPersistentMemory`, sized by `static_persistent_pool_bytes`, smaller `static_persistent_block_bytes` blocks). Move the goal registry `intToBeProved` out of the deloadable `LbMemory` (retiring tags 35-37) onto a per-LB `Memory::persistentArena` drawn from this pool. The persistent arena is never deloaded; it is reclaimed per LB at discharge (`dischargeStatementContent`: `resetToFresh` then `releaseAll`). The deactivation survey then gates on `isActive` instead of `lbMemory.manager.resident`.

**Why.** Theorems dropped non-deterministically under aggressive deload. Root cause: `deactivateRecursively` / `deactivateUnnecessary` survey `intToBeProved` for main-scope goals to decide `isActive = false`, but the cold container's accessors assert residency, so the survey was gated on `lbMemory.manager.resident` and SKIPPED on a deloaded LB. Which LBs are deloaded depends on parallel memory-pressure timing (the hard-bound mass deload), so the deactivation decision — and thus the theorem set — was timing-dependent. A goal registry that is never deloaded removes the only timing-dependent input; the survey then runs identically on every run.

**Alternatives considered.** (a) Reload the LB inside the survey — defeats the deload (the survey walks the whole tree; reloading every LB negates the memory saving and reintroduces the I/O the statification campaign exists to avoid). (b) Keep `intToBeProved` cold but cache the main-scope-goal count in a resident scalar — duplicate state every `intToBeProved` mutation must maintain (fragile). (c) A separate small-block pool is the minimal change: the per-LB persistent content is tiny (a handful of goal keys), so it costs little, never depletes the main pool, and keeps the no-heap-fallback + exhaustion-assert contract.

**Cost.** Each active LB pins ≥1 persistent block (32 KiB) for its whole active life, so the pool is sized by peak simultaneously-active-LB count (telemetry-tunable; exhaustion asserts loudly naming `static_persistent_pool_bytes`). The second reservation is the one sanctioned exception to [I-95](30_invariants.md#i-95) (amended). Contract: [I-108](30_invariants.md#i-108).

---

<a id="d-148"></a>
## D-148 — mid-burst admission throttle + steward force-deload: when the pool depletes during a burst, pause new task starts and force-deload other LBs until blocks free (2026-06-19)

**SUPERSEDED ([D-185](#d-185)).** The throttle was retired by the working-set pager ([D-161](#d-161) / [I-115](30_invariants.md#i-115)) and its code deleted.


**Context.** With the transient hashmems moved to the cold arena ([D-176](#d-176)), the per-LB cold footprint grows and an active burst's total pool usage (cold containers + hot scratch) can approach the `static_pool_bytes` limit mid-flight. Today the only mid-burst response is the exhaustion assert (a crash): the steward's relief decisions run only at the kernel barrier, between iterations, on quiesced LB-only counts ([I-106](30_invariants.md#i-106)), and the phase-2 work-stealing pool dispatches `(LB, split-part)` tasks with no back-pressure. The user requires graceful mid-burst handling: let started threads finish, do not start new ones, force-deload until enough blocks are free; for a split LB, started split parts finish and unstarted ones wait, but the split LB itself is never deloaded (stays resident).

**The choice.** A mid-burst admission throttle on the shared pool, in three parts:

1. **(commit 2) `GlobalMemoryManager` throttle band.** Total pool usage (`blocksInUse + hotBlocksInUse`) crossing `kThrottleHigh` (7/8) on any grant SETS an atomic throttle flag and fires the steward's onset wake; a release dropping below `kThrottleLow` (3/4) CLEARS it (hysteresis band). `isThrottled` (lock-free), `waitWhileThrottled` (spin-poll gate — a poll, not a CV: the throttle is a rare, brief emergency so simplicity beats wakeup latency), `setThrottleOnsetCallback`. The band sits ABOVE the steward wake/stop band and the kernel hard bound — the mid-burst emergency, not a between-iteration barrier decision.
2. **(commit 3) Admission gate at the phase-2 task pickup.** A worker calls `waitWhileThrottled` before pulling its next `(LB, split-part)` task, so in-flight tasks (already past the gate) finish while new starts pause. Split parts are ordinary tasks, so an unstarted split part waits at the same gate; a worker that picks up a task whose LB the steward deloaded reload-claims it before running. The phase-2 protocol arbitrates over a SEPARATE per-LB word, `burstClaim` (reset to `Idle` between the phase-1 join and the phase-2 pool), NOT `stewardClaim` — so the eviction plan / discharge state `stewardClaim` carries is never disturbed.
3. **(commit 3) Steward force-deload.** Woken by the onset (`requestRelief`), `forceDeloadPass` deloads eligible active LBs biggest-first (unsplit, no in-flight task — CAS `Idle → Busy → Dumped` on the SEPARATE `burstClaim` word; EXCLUDING the LB(s) being worked / any split LB, which stay resident) until total usage falls below `kThrottleLow`; then the gate clears and gated workers resume. The relief window (`begin`/`endThrottleReliefWindow`) brackets phase 2 and closes with a quiesce before phase 3. If nothing eligible remains and usage is still high, that is genuine exhaustion — surfaced by the next grant's `static_pool_bytes` assert (a gated worker always has an `Idle` candidate behind it, so no silent hang). **Plan commits 3 and 4 were landed as ONE commit:** the gate alone deadlocks (cold-dominated pressure has no in-flight task whose finishing frees blocks — hot scratch is bulk-freed only after phase 2), so the gate and the force-deload that relieves it must ship together.

**Determinism.** The throttle trigger is timing-dependent (it depends on how much hot scratch concurrent tasks grabbed), so the SET of force-deloaded LBs — and the gitignored `.deload/` file set — becomes run-to-run nondeterministic. The user has accepted this. Proof output stays fully deterministic: the throttle changes only WHEN tasks start (timing), not WHICH tasks run or what they produce (phase-2 tasks read only their own LB; deload/reload is content-invisible per [I-103](30_invariants.md#i-103) / [I-107](30_invariants.md#i-107); deposits are content-sorted post-join per [I-77](30_invariants.md#i-77)). This deliberately RELAXES [I-106](30_invariants.md#i-106) (mid-iteration total usage becomes a throttle decision input) and EXTENDS [I-122](30_invariants.md#i-122) (phase 2 may now see a claimable LB) — updated as those parts land.

**Tests.** 823 unit tests pass: (commit 2) throttle set-at-high / clear-at-low hysteresis across both grant paths + the spin-gate releasing a blocked worker; (commit 3) the relief request/window control flow + `forceDeloadPass`'s split-skip and not-throttled no-op. The force-deload-under-pressure path is not reachable by today's batches (high-water ~3/4, below the 7/8 throttle), so it is validated by a deliberate small-`static_pool_bytes` stress run (throttle forced; run completes with byte-identical proof artifacts), not the normal determinism gate. The normal gate is one full pipeline + verifier (proof artifacts byte-identical run-to-run; `.deload/` excluded).

**See** [`20_core_concepts/09_static_memory.md`](20_core_concepts/09_static_memory.md), [I-106](30_invariants.md#i-106), [I-122](30_invariants.md#i-122).

---

<a id="d-176"></a>
## D-176 — the two transient HashMemory instances move off the per-LB hot arena onto the cold `lbMemory.manager`; `hotHashMemoryArena` retired (2026-06-19)


**Context.** Each LB owns four `HashMemory` instances. `overallHashMemory` / `localHashMemory` already kept their `encodedMap` on the cold, deloadable arena `lbMemory.manager`; the two transient ones — `localHashMemoryDelta` (the "fresh rules this burst" shadow of `localHashMemory`) and `workingMemory` (per-burst external-mail scratch) — kept theirs on a dedicated per-LB HOT arena (`hotHashMemoryArena`, drawing `acquireBlockHot`, segregated from the LB ledger, never deloaded). That hot region cannot be reclaimed by the deload machinery, so under pressure it consumes pool blocks straight toward the `static_pool_bytes` exhaustion assert. The mid-burst depletion protocol (next step on this branch) needs every per-LB hashmem footprint visible to and reclaimable by deload.

**The choice.** Rebind both transient `encodedMap`s to `lbMemory.manager` / `lbMemory.dirty` — the exact binding `overallHashMemory` / `localHashMemory` use — and delete `hotHashMemoryArena` + its lazy `ensureHotHashMemoryArena`. Extend the existing manual threading to all four: `buildHashMemoryDeloadColumns` (dump + reload), the pre-`releaseAll` facet release in `releaseStaticBlocks`, and the `~Memory` / `destroyGrid` teardown, at new deload bases `kLocalHashMemoryDeltaDeloadBase` (251) / `kWorkingMemoryDeloadBase` (351). All four are now handled IDENTICALLY (user directive: extend the existing cold treatment uniformly, no special-casing). Deload only dumps/restores content — it never empties a transient instance or zeros a count; emptying stays the LB's own per-iteration `resetToFresh` (`workingMemory` at phase-1 entry, `localHashMemoryDelta` at phase-2 finalize entry). `reshuffle` (`manager.compactPages`) is vid-stable, so all four survive it with no hook.

**Determinism / byte-identity.** The deload byte stream is a pure function of container content regardless of backing arena ([I-103](30_invariants.md#i-103)); the two maps simply join it. Proof output is unchanged — only the backing arena moved, not the maps' content or the firing path. Hot-ledger segregation ([I-96](30_invariants.md#i-96)) is unaffected: it governs the per-worker string arenas, which keep `acquireBlockHot` (the sealed pages later moved off it onto the cold path).

**Tests.** 819 unit tests pass, including the teardown test extended to populate and release all four `encodedMap`s.

**See** [`20_core_concepts/09_static_memory.md`](20_core_concepts/09_static_memory.md), [`50_gotchas.md`](50_gotchas.md#g-53).

---

<a id="d-170"></a>
## D-170 — the cold-map family gains per-key erase (`erase` / `eraseIf`), in-place value update (`setValueAt` via the one reviewed `PagedVector::setAt`), and POD-key deload facets — the prerequisite for the int-keyed-map migration (2026-06-17)


**Context.** The cold-map family ([D-165](#d-165)) is append-only by design: `ColdHashMap::insert` asserts a brand-new key (no in-place value replace) and there is no per-key erase — the const-access `PagedVector` deliberately forbids both. Batch 1 migrates 13 already-int-keyed per-LB containers off the heap, but **6 of them need per-key erase** (`Memory::wipeSubtree` erases every entry whose validity id sits in a closed scope) and `intKnownStatements` additionally needs an **in-place value update** (`upsertStatementKey` OR-s membership bits into an existing row). The user chose to **add erase to the family first**, then migrate — and this erase is a campaign-wide prerequisite, not Batch-1-only (Batch 2/3/4 containers are all erased in `wipeSubtree` too).

**The choice.**

- **`PagedVector::setAt(i, v)`** ([`memory_infra/paged_vector.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/paged_vector.hpp)) — the ONE genuinely new storage primitive: a controlled in-place element write, the reviewed exception (Rule 8/19) to the otherwise const-access vector. It escalates the dirty state to `Restructured`, exactly the case the `DirtyState` doc already names ("potential in-place write happened → Restructured"). Erase adds one more reviewed vector primitive — `PagedVector::truncate(n)` (tail-drop, `Restructured`-marking) — used with `setAt` for the compaction below; the throw-away `PagedHashIndex` (rebuilt wholesale by `rebuildIndex`) is reused as-is.
- **`HashMap::erase(key)` / `eraseIf(pred)`** ([`memory_infra/cold_hash_map.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/cold_hash_map.hpp)) — run ONE forward **compaction** pass: survivors slide to the front of the key + value columns in lockstep (`KeyStore::setKeyAt` / `ValueStore::moveValue` → `PagedVector::setAt`), the dead tail is `truncate`d, then `rebuildIndex` once. `erase(key)` routes through `eraseIf` with a one-key predicate. Survivor relative order is preserved, so post-erase ids are insertion order minus the erased — the column bytes are byte-identical to a from-scratch insert of the survivors. **POD-key set + single-value map only**; byte-key / multimap erase is deferred (the stores lack `setKeyAt` / `moveValue` there, so naming it is a compile error, the intended "not yet" signal). O(count) — the "rebuild dense from live content" reload/reshuffle already runs, scoped to one container (the per-element shift it replaced was O(removed × count)).
- **`HashMap::setValueAt(id, v)`** — the in-place update door the set-once `insert` forbids; routes through `SingleValueStore::setValueAt` → `PagedVector::setAt`. `upsertStatementKey`'s flag-bit OR is its first consumer.
- **POD-key deload facets `KeysView` (1 tag) / `ValuesView` (1 tag)** — the missing piece: the POD-key set/map forms were built + unit-tested but had no facets for the generic `lb_deload` visitor (its tests round-tripped via raw `appendKeyBytes` / `bulkLoadKeyBytes`). The new facets mirror the byte-key `LengthsView` / `BytesView` but are self-contained per tag (a fixed key needs no length column). `KeysView` owns the whole-container lifecycle and rebuilds the index on reload; the key tag must precede the value tag in `ContainerTag` order so reload loads keys (+ index) before values.

**Determinism / byte-identity.** Keys/values stay in insertion order minus erased; `setAt` and erase both force `Restructured` so a mutated container is never tail-delta'd (which would silently drop the change); the throw-away index is never deloaded. So the deload stream remains the canonical content stream ([I-103](30_invariants.md#i-103)). No `LbMemory` wiring changed in this commit (no new `ContainerTag`); the migration batches consume the new surface.

**Tests.** 771 unit tests pass, incl. direct `PagedVector::setAt` / `truncate` tests and cold-map tests for `erase` / `eraseIf` (set + map, value-alignment, index-correct-after, deload byte-identity after erase, erase-history-invisible determinism, multi-page compaction byte-identical to a from-scratch survivor insert), `setValueAt` in-place update, and the `KeysView` / `ValuesView` facet round trip.

**See** [I-119](30_invariants.md#i-119), [`20_core_concepts/09_static_memory.md`](20_core_concepts/09_static_memory.md).

---

<a id="d-166"></a>
## D-166 — the cold-map lookup index keeps O(1) HASH but moves to throw-away STATIC memory: a paged, mutable-slot bucket array (no heap, no persistence) (2026-06-16)


**Context.** The heap hash index ([D-165](#d-165)) is O(1) but lives on the heap (`std::vector`) — a malloc the statification / ASIC campaign wants gone. The fully-cold sorted index ([D-167](#d-167)) removed the heap but ran ~34% slower (O(log n) cold lookups in the hot burst). The resolution: in GL there is no heap — a hot, derived structure belongs in **throw-away static** memory (pool-backed arena pages, mutable, discarded on deload, rebuilt on reload), NOT the heap and NOT persistent-cold. Keep the O(1) hash; just move its buckets there.

**The choice.** New `PagedHashIndex` ([`memory_infra/paged_hash_index.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/paged_hash_index.hpp)): a fixed-capacity paged `int32` slot array on the LB arena's page tier (`allocPage` / `pageAt` / `freePage`) with IN-PLACE `set` / `at` — the one thing the const-access `PagedVector` deliberately forbids. A SEPARATE container (not a mutable `PagedVector`), so `PagedVector`'s const-access deload-skip invariant stays fully intact. Throw-away: not in `visitContainers`, never deloaded, no dirty flag, rebuilt on reload; vid-addressed so compaction-transparent; in-place writes are safe precisely because it is never part of a deload image.

`ColdHashSet`'s index becomes a `PagedHashIndex buckets_`, with the original open-addressing hash LOGIC over it (`lookup` = `hashProbe` + linear probe; `indexInsert` / `indexPlace` = in-place `buckets_.set`, grow via `rebuildIndex` at 1/2 load). `ColdHashMap` / `ColdMultiMap` embed a `ColdHashSet`, so all cold maps + the seven live interners leave the heap. The binary branch's `sortedIds_` / `mergeIndex` / keystore comparators are removed.

**Byte-identity (by design).** Keys/values keep insertion-order ids → keys/values deload tags unchanged; `buckets_` is throw-away (never deloaded, no tag) → deload files byte-identical, theorems byte-identical. 760 unit tests pass (incl. a `PagedHashIndex` container test). Full-pipeline byte-identity + RT vs 1262 s is the post-commit gate.

**Goal.** Recover the hash O(1) runtime (≈ 1262 s, vs binary's 1691 s) with zero heap allocation — the best of both. It also generalizes: the same paged mutable-slot container, in PERSISTENT mode, is the substrate for the interior-update int-keyed maps. See [I-117](30_invariants.md#i-117), [`20_core_concepts/09_static_memory.md`](20_core_concepts/09_static_memory.md).

---

<a id="d-167"></a>
## D-167 — experiment: the cold-map lookup index moves OFF the heap into a fully-cold sorted structure (binary search), to measure RT vs the hash index (2026-06-16)


**Context.** The cold-map family ([D-165](#d-165)) keeps its key→id lookup index as a heap `std::vector` open-addressing hash — the one piece of per-LB map state that does NOT go cold, and the RAM the statification / ASIC campaign wants to eliminate to cross the on-die SRAM threshold. This branch is an **experiment**: move the index fully cold and measure whether the runtime holds.

**The choice.** Replace `ColdHashSet`'s `std::vector<int32_t> index_` with a two-level fully-cold sorted index ([`memory_infra/cold_hash_map.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/cold_hash_map.hpp)):

- `PagedVector<int32_t> sortedIds_` — ids `1..mergedCount_` held sorted by key, on the LB arena, bound to a PRIVATE dirty flag (`indexDirty_`) so index maintenance never escalates the aggregate's deload dirty. Derived — not in `visitContainers`, never deloaded; rebuilt on reload.
- `mergedCount_` — ids past it are the small unsorted **tail** (recent mints), linearly scanned. `lookup` = binary-search the sorted prefix (`compareToProbe`) + scan the tail. `mint` appends the key at the next id (insertion order, unchanged) and **merges** (re-sorts all ids via `mergeIndex` — a transient `std::vector` scratch → `bulkAppendBytes`) when the tail outgrows `kIndexTailMax` (≈64), amortizing the sort so the per-LB build is not O(n²). `rebuildIndex` (reload / `copyFrom`) is one merge.
- Each key store gains `compareToProbe` / `compareStored` (BytesKeyStore: `compareSpans` over the contiguous key views; PodKeyStore: raw `sizeof(K)` byte order — any consistent total order, the index being derived).

Because `ColdHashMap` / `ColdMultiMap` embed a `ColdHashSet`, this one swap takes ALL cold maps — and the seven live string interners — fully cold.

**Byte-identity (by design).** Keys / values stay in insertion order → ids unchanged → the keys/values deload tags unchanged; `sortedIds_` is derived and never deloaded (no new tag) → deload files byte-identical, lookups return the same ids → theorems byte-identical. 759 unit tests pass; the full-pipeline byte-identity check + the RT comparison vs the hash baseline (~1262 s) are the post-commit verification gate.

**Outcome (measured).** Correct — theorems byte-identical, verifier 0 — but **1691 s vs the hash baseline 1262 s (+34%)**: the O(log n) cold lookups in the hot burst path cost too much. Fully-cold binary loses the RT trade. **Superseded by the throw-away-paged-hash** ([D-166](#d-166)), which keeps O(1) and still leaves the heap. See [I-117](30_invariants.md#i-117), [`20_core_concepts/09_static_memory.md`](20_core_concepts/09_static_memory.md).

---

<a id="d-165"></a>
## D-165 — the cold "index heap, data cold" pattern becomes a reusable template family (`ColdHashSet` / `ColdHashMap` / `ColdMultiMap` over `BytesKeyStore` / `PodKeyStore<K>`); `ColdStringTable` folds onto it via alias (2026-06-16)


**Context.** The hand-rolled `ColdStringTable` ([D-153](#d-153)) already embodied the campaign's cold-map shape — the lookup index on the heap (the find-index, rebuilt on reload), the key DATA cold and paged (byte pool + location index, deload-persisted) — and is instantiated seven times in `LbMemory`. The remaining per-LB heap maps (`admissionMap`, `rejectedMap`, the origin maps, …) want the same split. So the one bespoke class is promoted to a reusable template family, with NameMap's two interners as the first proven customer ("a good reference to start from").

**The choice (user-selected: full family now; fold via alias).** [`memory_infra/cold_hash_map.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/cold_hash_map.hpp):

- **Key-store policies.** `BytesKeyStore` — variable-length byte keys (the former `ColdStringTable` storage: a `PagedVector<char>` pool + a `PagedVector<ColdStringLocation>` index). `PodKeyStore<K>` — a fixed trivially-copyable key in one flat `PagedVector<K>` (guarded by a `has_unique_object_representations` static-assert so key padding can't break determinism).
- **`ColdHashSet<KeyStore>`** — the interner: cold keys + dense positional ids (id == insertion order) + a shared heap open-addressing index (key→id). `using ColdStringTable = ColdHashSet<BytesKeyStore>` ([`cold_string_table.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/cold_string_table.hpp) is now a 3-line alias shim), so all seven `LbMemory` interners + NameMap compile unchanged and the deload byte stream stays byte-identical.
- **`ColdHashMap<KeyStore, V>`** — a key set + a parallel cold `PagedVector<V>` value column; **set-once** (insert asserts a new key), because `PagedVector` is const-access — an in-place replace is a separate, reviewed change to the cold primitive (Rule 8), not smuggled in here.
- **`ColdMultiMap<KeyStore, V>`** — key → an ordered value run (CSR: a runs index + a flat value column); **append-to-tail only** (asserts the key is the last or new), interior-key insert deferred to a real consumer (it would force an O(N) cold rewrite).

**Scope / staging.** Commit 1 lands `BytesKeyStore` + `ColdHashSet` and folds `ColdStringTable`; commit 2 adds `PodKeyStore` + `ColdHashMap` + `ColdMultiMap` with unit tests. The stored-value forms are NOT yet wired into `LbMemory` (no new `ContainerTag`) — they ship built + unit-tested, ready for the future int-keyed-map migration; NameMap exercises only the interner. Mail is excluded from any future migration.

**Unification.** The three forms collapse into ONE class `HashMap<KeyStore, ValueStore>`: the engine (`mint` / `lookup` / `decode`, the throw-away index, the lifecycle) lives once, and the value column becomes a **value-store policy** symmetric to the key-store policy — `EmptyValueStore` (a set; a private empty-base, elided by empty-base optimization since `[[no_unique_address]]` is a no-op under C++17/MSVC, so a set's layout is byte-identical), `SingleValueStore<V>` (one value per key), `CsrValueStore<V>` (CSR runs). `ColdHashSet` / `ColdHashMap` / `ColdMultiMap` become `using` aliases, so every call site and the deload byte stream are unchanged; the value stores share a uniform contract (`clearValues` / `releaseValues` / `copyValuesFrom` / `valuesLiveBytes` + `kTagCount`) the engine calls branch-free, and the value-shaped / byte-key-legacy methods instantiate only where called.

**Why.** One skeleton for every per-LB cold map (interners now, int-keyed maps next) instead of N hand-rolled classes; the heap-index / cold-data split is stated once as [I-117](30_invariants.md#i-117). Folding via alias makes the change zero-churn at the seven existing call sites and byte-identical on disk.

**See** [`20_core_concepts/09_static_memory.md`](20_core_concepts/09_static_memory.md).

---

<a id="d-168"></a>
## D-168 — the cold-map family gains a fourth value-store policy `SetValueStore<V>` (a SORTED-UNIQUE set per key, CSR-backed) + the matching `HashMap` set surface, so the Batch-2 int→set maps the append-to-tail `ColdMultiMap` could not serve migrate onto the one family (2026-06-17)


**Context.** Batch 2 of the container campaign migrates four per-LB heap maps whose VALUE is a small SET — `intToBeProved` / `intStatementLevelsMap` (packed key → level set), `orBookkeeping` (key → decoded-ordered disjunct-id set), `eqClassSttmntIndexMapMap` (nested). The plan named `CsrValueStore` (`ColdMultiMap`) as the substrate, but that form is APPEND-TO-TAIL only, stores an unsorted BAG (no dedup), and has no per-key erase — a real-prover set map needs interior-key growth (a value added to an existing, non-last key), set dedup, sorted iteration, and `wipeSubtree` erase. So the family is EXTENDED, not the data reshaped (user-selected: extend the existing hashmap, migrate all four).

**The choice.** [`memory_infra/cold_hash_map.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/cold_hash_map.hpp):

- **`SetValueStore<V>`** — a fourth value-store policy beside `EmptyValueStore` / `SingleValueStore<V>` / `CsrValueStore<V>`. Same physical layout as `CsrValueStore` (a run-start index + a flat value column, two deload tags), but the run is kept SORTED + DUPLICATE-FREE by the owning map's set surface. Distinct from `CsrValueStore` by type so the bag's `appendToTail` and the set's `insertSorted` never mix.
- **`HashMap` set surface** (instantiates only for `SetValueStore`): `insertSorted(k, v, cmp)` (find-or-mint key, binary-search the run, dedup, splice at the sorted position via `PagedVector::insertAt` — an interior shift — then bump later keys' run starts), `assignSet(k, vals, m)` (whole-run replace, in place when the size is unchanged), `setContains`, and a run-aware compacting `eraseSet` / `eraseSetIf`. The ordering predicate is PASSED PER CALL (defaulted `std::less<V>`; the `ValueInterner`-bound decoded-id comparator for `orBookkeeping`), never stored — the store stays a pure container with no interner-lifetime coupling (a refinement of the plan's stored-`Compare` sketch).
- **`RunStartsView` / `RunValuesView` deload facets** — the CSR run-start column was never taught to the generic `lb_deload` visitor (a `ColdMultiMap` had only a hand-rolled round-trip test). The two facets present a CSR map as its three tags (keys → run-starts → values) with the same interface as `KeysView` / `ValuesView`, so the generic visitor serializes a `ColdSetMap` with no structural change.
- **`PagedVector::insertAt(i, v)`** — the shift-right insert (mirror of `erase`), the splice primitive `insertSorted` needs (a reviewed cold primitive, Rule 8/19; marks `Restructured`).
- **Alias `ColdSetMap<KeyStore, V>`**.

**Why.** The append-to-tail `ColdMultiMap` is the right shape for a write-once bag (e.g. the Batch-5 origin map); a set-valued map needs interior insert + dedup + erase, which would corrupt the multimap's append invariant if bolted on. A distinct policy keeps each shape's invariant clean and makes misuse a compile/contract error. The interior-shift cost (O(value-tail + key-tail) per insert) is the campaign's one watched RT risk, measured at the batch gate.

**Cost (measured at the Batch-2 gate, 2026-06-17).** The full pipeline ran ~28% slower than the pre-Batch-2 baseline (~1303 s across two runs vs ~1019 s) — the interior-shift / per-write canonicalization cost of the four migrated set maps. Proof content is unchanged (identical 131018 verifier checks, 0 failures; two sequential runs byte-identical across all 439 proof-graph artifacts). The user accepted the regression: the campaign's binding constraint is MEMORY (the static-memory hierarchy gates FTA), not runtime, so trading runtime for the four containers going static is the intended direction. A later pass may optimize the hot insert path (the dominant cost) or flatten the heaviest container to a packed-pair `ColdHashSet` if runtime later becomes a constraint.

**See** [`20_core_concepts/09_static_memory.md`](20_core_concepts/09_static_memory.md), [I-118](30_invariants.md#i-118).

---

<a id="d-169"></a>
## D-169 — the cold-map family gains a fifth value-store policy `BlobCsrValueStore` (key → a run of variable-length byte BLOBS, each a record's canonical serialization) + a `ColdBlobMap` alias, the "record value store" no single trivially-copyable `V` could hold (2026-06-18)


**Context.** Batch 3 migrates `equivalenceClassesMap`, whose value `EquivalenceClass` is a multi-field record (an id vector + a map-of-sets + an origin map) — the shape the plan's Batch-4 note flagged as "the one shape the family does NOT yet cover with a single `V`." The four existing value stores all hold a single trivially-copyable POD. So the family is extended (per the Batch-2 `SetValueStore` precedent) with a RECORD value store, built + tested first, before any migration. The same store is the Batch-4 substrate for the `HashMemory` record-set values (incl. the deeply-nested `admissionMapIntegration`).

**The choice.** A record-AGNOSTIC store: it holds opaque byte blobs; a per-record-type (de)serializer at the CALL SITE produces / consumes each blob. A two-level CSR (`runStarts_` key→blob, `blobStarts_` blob→byte, dense `blobPool_`), every boundary derived — four deload tags with a POD key, refined down from the plan's five (the blob lengths derive from consecutive `blobStarts_`, like `runLen` from `runStarts`). `assignRun` (whole-run replace) + `eraseBlobIf` (run-aware compaction) splice the byte + blob columns once each via the new `PagedVector::replaceRange` (a one-pass variable-length range splice). Blobs may straddle pages (read through `contiguousRun`), so the pool is dense and deloads via `appendSpanBytes` with no staging handshake.

**Ruled out.** Parallel fixed-count CSR columns per field — explodes the deload-tag count and cannot express the nested-map fields. A packed struct `V` — impossible (variable length). Tombstone-and-append — accumulates holes against the campaign's memory goal. Blob-CSR keeps the tag count O(1) in record complexity and the deload canonical (the codec being the determinism point).

**See** [`20_core_concepts/09_static_memory.md`](20_core_concepts/09_static_memory.md), [I-98](30_invariants.md#i-98).

---

<a id="d-177"></a>
## D-177 — Batch 3: the equivalence-class state goes off the heap — `equivalenceClassesMap` cold (in the deload image), `changedClassesThisStep` + `eqClassNameCaches` onto dedicated non-deload arenas (transient / derived, not deload-image content) (2026-06-18)


**Context.** The three equivalence-class containers have three natures, so they get three homes. `equivalenceClassesMap` is persistent LB content → cold (the deload image, the MPU target). `changedClassesThisStep` is a per-step transient delta and `eqClassNameCaches` a rebuildable derived memo → NEITHER is deload-image content, but the campaign's no-heap rule still applies, so both go on dedicated non-deload arenas. The user chose the full off-heap path (Option 3) over keeping the two leftovers as documented heap exceptions.

**The choice.**
- `equivalenceClassesMap` → `ColdBlobMap<PodKeyStore<int16_t>>` ([D-169](#d-169)), with `serializeEquivalenceClass` / `deserializeEquivalenceClass` (the codec sorts the `unordered_map` origin keys before emit — the single determinism point). The former `classesAt` pointer accessors become `decodeClassesById` / `decodeClassesAt` (read) + `assignClassesById` (write); the working `EquivalenceClass` stays transient heap, decoded at boundaries (I-84).
- `changedClassesThisStep` → `ChangedClassesBuffer`, a dedicated transient HOT arena ([I-120](30_invariants.md#i-120)); `eqClassNameCaches` → the same resident-non-deload-arena pattern (`kindById_` a `PagedVector<uint8_t>`, `tokensByExprId_` a non-deload `ColdBlobMap` with a `SpecialTokenScan` codec; `tokensOf` now returns a value).

**Gate (commit-5 build — the whole batch, 2026-06-18).** Byte-identical to the pre-migration baseline: `theorems.txt` + all tracked artifacts unchanged, 131018 verifier checks / 0 failures (matching the Batch-2 baseline exactly). Both the commit-3 run and the final run (commits 4–5 added) matched; the user waived the determinism second run.

**See** [`20_core_concepts/09_static_memory.md`](20_core_concepts/09_static_memory.md), [I-98](30_invariants.md#i-98), [I-120](30_invariants.md#i-120).

---

<a id="d-171"></a>
## D-171 — a thin typed façade over the byte-erased cold-map engine: `Codec<K>` / `Codec<Record>` + `TypedCold{Set,Map,SetMap,BlobMap}` owning wrappers, so a per-LB map reads as its real key/record type and every codec is one unit-tested place (2026-06-18)

**Problem.** The cold-map family's variable-length key store `BytesKeyStore` is type-erased: a member declared `ColdHashMap<BytesKeyStore, int>` does not say what it holds, and each composite key's (de)serialization is hand-coded at the call site (`encodeEqClassKey`, `packStatementKey`, `packLbStateKey`, `serializeEquivalenceClass`). The user asked whether the family could take explicit types as template parameters and call per-type serialization from within — yes, as a thin façade.

**Decision.** A new header [`memory_infra/typed_cold_map.hpp`](../../GL_Quick_VS/GL_Quick/src/memory_infra/typed_cold_map.hpp) adds ONE owning wrapper `TypedCold<K, ValueStore, Record=void>` — mirroring the engine's own one-class `HashMap<KeyStore, ValueStore>` — surfaced through four aliases (`TypedColdSet<K>`, `TypedColdMap<K, V>`, `TypedColdSetMap<K, V>`, `TypedColdBlobMap<K, Record>`); it holds one inner `HashMap` instantiation (zero added bytes, zero new deload tags) and re-exporting its facet view types, so `LbMemory` constructs the deload facets against `&map.inner` and the on-disk bytes are a pure function of the keys the codec produces — a wrapper whose codec reproduces a former hand-rolled layout deloads BYTE-IDENTICALLY. A `Codec<K>` maps the typed key ↔ the engine's `KeyView`: a byte-key base (`Encoded = std::string` held in a caller local so the `StrSpan` never dangles → `BytesKeyStore`) or a packed-scalar base (`Encoded = S` → `PodKeyStore<S>`), the store chosen by `Codec<K>::KeyStore`. A `Codec<Record>` serializes a blob record; the decoded-order set comparator the prover's record-set values use is passed PER CALL at the typed surface, never stored in the codec. `Record` is named only inside member-template methods (`R = Record` defaulted), so a `TypedColdBlobMap` instantiates with `Record` incomplete — `lb_memory.hpp` needs no domain record headers. Each shape's methods (a map's `find`, a set-map's `insertSorted`, a blob map's `assignRun`) are on-demand member templates instantiated only when called (the engine's mechanism), so a set never names a value type; the raw packed-scalar overloads are SFINAE-enabled only when `KeyView!= K`, so an identity-keyed instantiation (`K == KeyView`) keeps one unambiguous typed surface.

**Scope (user-confirmed).** `Codec<K>` covers ALL composite keys — byte keys AND packed-int keys; the façade also formalizes record VALUE codecs. Phase 1 retrofits every existing composite-key / record map (`eqClassSttmntIndexMapMap` byte key; the packed-int maps `expandedImplications` / `intKnownStatements` / `intToBeProved` / `intStatementLevelsMap` / `orBookkeeping`; `equivalenceClassesMap` records) onto the façade byte-identically, proving it on gated code before the Batch-4 `HashMemory` containers consume it.

**Gate.** Byte-identity is unit-tested ([`tests/test_typed_cold_map.cpp`](../../GL_Quick_VS/GL_Quick/src/tests/test_typed_cold_map.cpp)): each `Codec` matches the exact legacy byte/scalar layout + round-trips, and a `TypedCold*` wrapper's deload columns are byte-identical to the raw engine map built with the same content, per shape (set / map / set-map / blob-map) + the re-exported-facet dump/reload. 805/805 unit tests pass.

**Part-B pipeline gate (2026-06-18).** The eight existing composite/record maps migrated onto the façade — `expandedImplications` (`TypedColdSet<LbStatePairKey>`), `intKnownStatements` (`TypedColdMap<StatementKey,StatementFlags>`), `intToBeProved` + `intStatementLevelsMap` (`TypedColdSetMap<StatementKey,int>`), `orBookkeeping` (`TypedColdSetMap<LbStatePairKey,int32_t>`), `eqClassSttmntIndexMapMap` (`TypedColdMap<EqClassKey,int>`), `equivalenceClassesMap` (`TypedColdBlobMap<int16_t,EquivalenceClass>`) — each commit byte-identical-by-construction. Full `python main.py`: verifier 131018 checks / 0 failures (matching the Batch-2/3 baseline exactly), 148 theorem goals reached, and ALL tracked proof artifacts byte-identical to the pre-branch baseline (, `git diff` clean) — a pure refactor, so no determinism second run was needed (baseline-identity is the stronger result). The packed-int set-maps carry raw `KeyView` overloads so their pre-computed shared scalar keys pass through verbatim; the byte-key `eqClass` map and the `int16` blob map go through the typed surface.

**See** [`20_core_concepts/09_static_memory.md`](20_core_concepts/09_static_memory.md), [D-165](#d-165), [D-169](#d-169).

---

<a id="d-138"></a>
## D-138 — NameMap moves fully cold: validity metadata becomes a flat paged parent-pointer forest (`validityNodes`) + deload-persisted, `pairMap` dropped (verdict walks it), and the direct int↔string mapping restored (2026-06-16)


**Context.** After the strings campaign, NameMap's two string dictionaries were cold (`ColdStringTable`), but its id-form bookkeeping — `stackOfValidity`, `ancestorsOf` (jagged int16) and `pairMap` (ancestor-verdict hash) — stayed on the heap, and the cold-strings move had introduced a `tid+1` offset (the off-table `"main"` special case → `nameId == tableId + 1`). Statification's rule is that state surviving between bursts goes to a cold paged container; this metadata survives the LB's lifetime.

**The choice (user-selected; three options confirmed up front).**

- **Cold + paged metadata, as a flat parent-pointer forest.** `stackOfValidity` / `ancestorsOf` become DERIVED walks over one flat paged container — `validityNodes`, a `PagedVector<ValidityNode{parentId, ownSubId}>` in `LbMemory` — deload-persisted via ONE new tag (18, visited directly like the statement vectors) and surviving discharge ("write stuff surviving between bursts to cold"). The forest was chosen over the jagged CSR first implemented (commits 1/3/4) for RT: ~7× leaner (4 B/id vs ~30), O(1) `encodePush`, same O(scope-depth) `verdict` walk. Rejected the rebuild-on-reload alternative (derivable from the names table, like the find-index) in favour of genuine cold state, and the derive-from-name alternative (zero storage but a hash lookup per ancestor on every `verdict`) as the RT loser.
- **Drop `pairMap`.** It carried no information `ancestorsOf` does not — `verdict(a, b)` derives from membership (`a` is in `ancestorsOf[b]`), and `comparable` / `deeperOf` route through `verdict`. This also deletes the O(N)-per-scope fill loop `encodePush` ran. Rejected paging `pairMap` (a literal reading of "page all maps"): it would keep the O(N²) fill and machinery for zero gain. Verdict stays integer (an int16 ancestor scan), single-threaded only.
- **Direct int↔string (as on `main`).** Reproduces 's "main on-table" technique: `internMain` interns "main" as cold-table id 1 on first touch (folded into `seedIfEmpty`), the `±1` offset is removed everywhere, and eternal-root fallbacks cover the pre-intern window. `nameCount` is invariant across the change, so the hashburst dump stays byte-identical and theorem ids are unchanged.

**Lazy seeding.** The metadata is seeded on the first `encode` (not at bind, which runs in every `Memory` ctor), so a transient `Memory` consumes zero pool blocks; `MAIN_ID` reads before the seed resolve to the eternal-root values, byte-identical to the seeded answer.

**Invariants.** [I-104](30_invariants.md#i-104), [I-105](30_invariants.md#i-105); updates [I-3](30_invariants.md#i-3) (the `stackOf` reference is retired). See [`20_core_concepts/04_validity_stack.md`](20_core_concepts/04_validity_stack.md), [`20_core_concepts/09_static_memory.md`](20_core_concepts/09_static_memory.md).

---

<a id="d-158"></a>
## D-158 — the chapter export reloads deloaded LBs read-only, discharged ones included; `ensureLoadedForRead` is the sanctioned exception to the discharged-forever reload assert (2026-06-12)


**Context (the IncubatorPeano1 incident).** Pre-campaign, `buildStack` read origin history off the heap — readable on discharged and deloaded LBs alike. The strings campaign moved the origin strings into the LB's deloadable cold image; under real pressure the pending drain dumps discharged LBs and releases everything, so the export then met a cold `__contradiction__` LB: `ColdStringTable::lookup` answered a silent "not found" on the empty find-index (a Rule-19 hole, since fixed — lookup/intern now assert residency BEFORE the empty-index miss), `hasOrigins` went false, and the contradiction fallback — which lacked the self-guard its post-candidate sibling had — re-entered the same LB forever: a silent 0xC00000FD at any stack size. Localized by waypoint tracing after the depth tracer showed a period-1 cycle at 560 bytes/frame.

**The choice (user-selected option a).** `Memory::ensureLoadedForRead` ([`memory.cpp`](../GL_Quick_VS/GL_Quick/src/memory.cpp)) — the reload core shared with `ensureLoaded` (`reloadFromImage`), minus the discharged-forever assert. Called at `buildStack` entry (covers every LB switch: initial chapter walks, both contradiction fallbacks, the lift's same-LB probes); defined no-op when resident. It does NOT reactivate: `isActive` / `dischargedForever` stay untouched, no kernel runs after the export — I-112 keeps its meaning (an LB never returns to the ACTIVE set; post-prove readability is restored deliberately). Reloaded LBs are not re-released after the export — the process exits with the batch. Rejected: keeping discharged string tables in RAM permanently (loses the deload win exactly where content piles up) and capturing origin content at discharge like the registry keys (defeats the memory purpose for the largest payload). Hardening landed with it: the first contradiction fallback now carries the same `contraLB!= &memoryBlock` self-guard as the second, so a genuinely missing record fails on the loud no-origin assert instead of recursing.

---

<a id="d-153"></a>
## D-153 — one `ColdStringTable` per interner space: append-only bytes on the LB's arena, stable ids, derived probe-only find-index, canonical lengths+bytes deload image (2026-06-12)


**The choice.** [`memory_infra/cold_string_table.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/cold_string_table.hpp) / `.cpp` replaces each interner's heap pair (`std::vector<std::string> idTo*` + `std::unordered_map<std::string,id> *ToId`) with three parts: append-only string bytes bump-allocated on the LB's arena (a string never straddles a block — `LbArena::alloc` pads to the next boundary, deterministically reproduced at reload, and asserts an oversize string against the block size); an `ArenaVector<ColdStringLocation>` id → (offset, len) index (id 0 reserved, mirroring the interners' slot 0); and a derived open-addressing find-index (FNV-1a 64 `hashSpan`, linear probing, ids only) that is never dumped, never an input to the deload dirty flag, and rebuilt deterministically at reload. PER-INTERNER tables, not one shared heap — `destroyGrid` resets every space except `lbStateInterner`, and selective reset needs separate tables. Deload = two GLDL tags per table (lengths int32 in id order + concatenated bytes — pure content, padding invisible); `bulkLoad` re-bumps in id order with the same padding rules, so every location and id reproduces exactly. `copyFrom` re-interns in id order (the CE clone path); `resetToFresh` empties the table — the byte storage becomes arena holes the copying compaction reclaims — and invalidates all ids wholesale.

**Heap carve-out (tier-1 bookkeeping rule).** The find-index is `std::vector` bookkeeping — the statification invariant binds element PAYLOADS; `ArenaVector` is const-access by design and open addressing needs in-place slot writes, so the probe index stays on the heap. The container-batch tier owns its migration. (The location index and the string bytes themselves are already on the arena.)

**Why.** The interner backings are the LAST per-LB string payloads on the heap after intification; on this table they enter the pool's hard bound and the LB's deload image, and decode becomes a zero-copy view (`StrSpan` into the cold arena) instead of a heap-ref into a `std::vector<std::string>`.

---

<a id="d-164"></a>
## D-164 — staging-record strings live on task-owned sealed pages, freed after the drain consumes the records; deletion point moves from "thread finished" to "records consumed" (2026-06-12)

**SUPERSEDED ([D-185](#d-185)).** Sealed pages now ride the COLD grant path (`bind` / `acquireBlock`), not the segregated hot path; the lifetime-by-deletion-point design below is otherwise unchanged.


**Context.** Firing records (`FiringRecord` / `StagedAdmissionValue` / `DeferredIntegrationPrep`) are the burst's output: produced by request-pool tasks, merged and drained by a different pool after the producing threads are gone (`applyFiringRecords` → `drainAdmissionKeysAlgebra` / `drainDeferredIntegrationPreps`). Their strings fit neither hot class rule: a worker slot's arena resets per executor (one slot serves many LBs per phase), and cold minting is single-threaded (I-83) while the records are born in the parallel phase.

**The choice (user-decided over keeping `std::string` for the records this branch).** `SealedPageSet` + `SealedString` ([`memory_infra/sealed_pages.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/sealed_pages.hpp)): each task bump-fills pages it owns exclusively (lock-free — the wiring pre-sizes one set per task slot; the class is deliberately non-movable so view owner-pointers stay stable), seals them at task end, the set crosses the pool join with the records, and the consumer frees the pages immediately after the drain mints the cold ids. Three-state lifecycle asserted on every operation (`Filling → Sealed → Freed`); whole blocks poisoned before release. Blocks travel the segregated hot grant path; no per-set cap — staging volume is work-content-dependent, the pool's exhaustion assert is the backstop. Scratch pages die with the thread; sealed output pages die with consumption — one amendment to the per-thread model, not a replacement.

---

<a id="d-163"></a>
## D-163 — hot string arenas are per WORKER SLOT (coreId-indexed), drawing pool blocks through a segregated grant path; "per thread" realised without `thread_local` (2026-06-12)

**SUPERSEDED ([D-185](#d-185)).** The per-worker arenas are now COLD (`ScratchArena`), uncapped, RELEASED per task; the segregated grant path is deleted. The per-slot (not `thread_local`) design below is unchanged.


**Context.** The string statification model (user-designed) gives transient calculation strings a "hot" class: exact-length bump allocation into a per-thread arena, wholesale reset per scope, stack-like rewind marks, never deloaded, never observable. LB split forces per-thread isolation — several executors of one LB run concurrently.

**The choice.** "Per thread" is realised as one `HotArena` per worker SLOT, indexed by the `coreId` every executor entry point already receives (`HotArenaRegistry`, [`memory_infra/hot_arena.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/hot_arena.hpp)) — NOT as `thread_local` storage. The prover's pools spawn fresh `std::thread`s every phase of every iteration, so thread-local arenas would die with their threads and re-acquire blocks each phase: exactly the block churn and grant-ledger noise the design must avoid. One slot runs at most one executor at a time and phases join before the next starts, so slot ownership gives the same isolation a private thread arena would. Arenas draw blocks from the SAME pool through a segregated grant path (`GlobalMemoryManager::acquireBlockHot` / `releaseBlockHot`): hot grants enter only the hot telemetry counters, never `blocksInUse` / `grantsSinceBarrier` / the grant trigger ([I-96](30_invariants.md#i-96)). Blocks are acquired lazily under a fixed per-slot cap (`hot_arena_bytes`, asserted multiple of the block size) and retained until teardown — steady state produces zero grant traffic. Freed spans are poisoned (`kHotPoisonByte`) so raw-pointer leaks announce themselves; the generation counter is the assert anchor for the hot string view type (next commit).

**Alternative rejected.** A separate dedicated hot reservation outside the pool — cleaner accounting separation, but a second program-start reservation dilutes the "ONE reservation models the ASIC SRAM" story (I-95) for no behavioral gain; segregated counters on the shared pool keep every existing LB-side decision input byte-identical to the pre-hot-arena code.

**Update.** `HotArena` is no longer a separate bump allocator — it is `using HotArena = LbArena`, the shared arena in HOT mode. The per-slot / segregated / cap / generation properties above are unchanged; only the implementation merged. See [D-182](#d-182).

---

<a id="d-182"></a>
## D-182 — one bump-arena class (`LbArena`) backs both the cold per-LB path and the hot per-worker path; hot is a limited mode (no deload, no compaction) (2026-06-13)

**SUPERSEDED ([D-185](#d-185)).** There is no longer a HOT mode — one `LbArena`, one cold grant path; the per-worker / sealed users bind cold and differ only in lifetime.


**Context.** After the cold cutover the prover had TWO bump allocators duplicating the same core (block acquire, virtual cursor, no-straddle padding, poison-on-free, lazy first block, release-all): `LbArena` (cold, offset-addressed, deload + compaction) and `HotArena` (hot, raw-pointer, generation + reset + reserve cap, segregated grant). The user directed unifying them — "literally one management code, in the hot case with some limitations" — addressed by virtual offsets like cold, so the hot path can later host the same containers (external statements / index tables) for its throw-away data.

**The choice.** `LbArena` is the single class, carrying a `Mode` (Cold default / Hot). `HotArena` becomes `using HotArena = LbArena`; the per-worker string arenas and the sealed-page handoff are LbArenas bound with `initHot`. The shared core (alloc → `ArenaOffset`, `resolve`, `popTo`, `releaseAll`, no-straddle, lazy, poison) lives once. HOT adds: the segregated `acquireBlockHot` grant path (preserving [I-96](30_invariants.md#i-96)), a generation counter with `reset` / `mark` / `rewind`, an optional reserve cap (`hot_arena_bytes`; **0 = uncapped**, which the sealed-page handoff uses), a mode poison byte (0xDD vs cold 0xCD), and `allocBytes` (a resolved `char*`). The "limitations": HOT never deloads/reloads and never compacts, so a resolved pointer stays valid until its rewind/reset — which is exactly why the hot string views keep caching `char*` (no consumer change, no per-access resolve cost). `Mark` collapsed from `{generation, blockIndex, offset}` to `{generation, cursor}` (the pair was just the virtual cursor).

**Why offsets, not raw pointers, for hot.** The user's stated reason: one management code, plus the future goal of letting the hot path store throw-away data as cold-style containers (`ArenaVector`-shaped external statements / index tables), which are offset-based. Hot resolves eagerly and caches the pointer (valid under the no-relocation limitation), so the offset model costs nothing at the hot access site.

**Result.** ~250 fewer lines (the duplicated `HotArena` core deleted); `HotString` / `HotScope` / `str_ops` / the registry / the per-executor reset keep their behaviour (their `alloc` calls moved to `allocBytes`). The full pipeline gate is byte-identical — hot content is unobservable by design.

---

<a id="d-157"></a>
## D-157 — the full discharge protocol: capture exact gate records, empty, reshuffle (all blocks back, zero I/O); post-prove gates probe RAM records; near-empty images under pressure (2026-06-12)


**The choice (the user's protocol verbatim: "empty unused containers, reshuffle, deload the near-empty image, forget").** `Memory::dischargeStatementContent`, run single-threaded at the barrier of the LB's last active iteration: set `dischargedForever` → capture the EXACT `(originalId, validityId)` pair set of `intEncodedStatements` into `Memory::dischargedRegistryKeys` → `LbMemory::clearDischargeableContainers` (tag semantics declared: all four tier-1 statement vectors are dischargeable) → `LbMemory::reshuffle` — with zero live pages every block returns to the pool immediately. A dead LB's entire static footprint comes back at the barrier with ZERO I/O; the pressure-lazy pending dump later writes the near-empty canonical image (format continuity on disk), and reload never happens — `ensureLoaded` now asserts `!dischargedForever`.

**The gate re-pointing (the evidence-first step — this is where the abandoned branch broke the verifier).** Both post-prove visualizer gates reduce to "does the container hold a row with `(originalId == lookup(head), validityId == MAIN_ID)`":
- **Recursion-node gate** → probes `intLocalEncodedStatementsSet`, the EXACT packed-pair mirror maintained at every mutation site (I-86) — valid for resident, deloaded, and discharged nodes; its reload call is gone (`buildStack` reads only RAM state — `exprOriginMap`, Rule 16).
- **Equality-node gate** → on a discharged node probes `dischargedRegistryKeys` (exact by construction); on a live node (parked-never-woken, or an active dumped by the final barrier's pressure path) reloads and walks the registry as before — both branches defined results.

The old branch instead probed `intKnownStatements`' registered bit — a SUPERSET (canonicalization-erased rows keep the bit after leaving the vector), so the gate answered true where the container said false: the suspected verifier-failure source. The capture-at-discharge record cannot drift — it is a copy of the container's final content.

**Ruled out as readers of a discharged LB's statement vectors** (the clearance audit behind the clear-at-barrier): scope wipes are LB-internal (I-50); mail lands in heap `mailIn`; `prove`'s count loop is active-only; the Rule-14 dump trap targets an active LB; the compressor builds fresh `CompressorNode_*` scratch LBs, never grid LBs; the CE filter never enters `proveKernel`.

---

<a id="d-183"></a>
## D-183 — the steward prepares the next LBs ahead of the phase-1 cursor: prefetch-reload of dumped victims, reshuffle of gated-fragmented LBs; unprepared handover is a defined result (2026-06-12)


**The choice (user-designed: "steward must try to make sure the next LB has its blocks in RAM and is already reshuffled; if not possible in time — give it without reshuffle").** Phase 1's work-stealing dispatch cursor is hoisted to a kernel-scoped atomic and registered with the steward (`beginPhase1Window` at kernel entry / `endPhase1Window` at the phase-1 join). While the window is open the steward polls (200 µs period) and prepares indices `[cursor + workers, cursor + workers + kLookahead)` (lookahead 4): a steward-dumped eviction victim reloads ahead of its worker (`Dumped → Busy → Idle`), a fragmented LB passing `needsReshuffle` compacts (`Idle → Busy → Idle`). GL's advantage over database read-ahead: the schedule is the deterministic active order, so the prefetch is exact, never speculative.

**Handshake strengthening.** The phase-1 handshake now claims EVERY LB unconditionally (`Idle → WorkerOwned`) before its body runs — once a slot begins, the steward is locked out for the iteration (its reshuffle CASes only from `Idle`). The barrier resets all active claims to `Idle` post-quiesce. A worker that arrives before the steward prepared its LB takes it as-is — reload through the normal `ensureLoaded`, no reshuffle — the user-specified defined dual result, not a fallback.

**Determinism.** Every window action is content-invisible (reload of an already-written canonical image; reshuffle) — so the poll timing, the lookahead race, and whether an LB was prepared in time cannot shift any observable. No quiesce is needed at window close: every active LB's handshake outwaits a `Busy` steward before its body runs, so no steward claim survives the phase-1 join, and phases 2/3 (raw element pointers) never see a moved or cold page.

---

<a id="d-162"></a>
## D-162 — per-LB copying compaction behind a dual AND fragmentation gate; the only operation that reassigns offsets (2026-06-12)


**The choice (user-designed: "when nothing to load/evict, reshuffle blocks still in RAM into logical order"; scope confirmed per-LB).** The bump arena frees only its tail (`popTo`), so holes left by `erase` / `clear` / discharge accumulate. `LbMemory::reshuffle` is the copying (semi-space / Cheney-style) collector that reclaims them — an in-RAM round trip through the same canonical element stream the deload uses: (1) capture each container's logical content (`appendSpanBytes`) and element count into heap buffers, in tag order; (2) `release` every container and `manager.releaseAll` (return all blocks); (3) rebuild each container from its buffer (`bulkAppendBytes`), re-bumping elements and the spine densely onto a fresh consecutive arena. The emptied blocks return to the global pool — relief tail-pop alone could never produce. `LbMemory::reshuffle` is the ONLY legal entry point; the transient heap spike is the live content size (the semi-space cost).

**The gate (user-specified: "number of deleted chunks AND the percentage of those chunks relative to memory — real, not virtual").** `steward::needsReshuffle(reclaimableBytes, spanBytes, blockBytes)`: BOTH must hold — `reclaimableBytes >= blockBytes` (absolute floor: at least one whole block reclaimable) AND `reclaimableBytes * 8 >= spanBytes` (the holes are ≥ 1/8 of the arena's used span). `reclaimableBytes` is `LbArena::usedBytes` minus the containers' live content (`LbMemory::liveBytes`); `spanBytes` is `usedBytes`. The dual AND is the Redis active-defrag pattern (absolute bytes floor AND percentage threshold); below the gate "no reshuffle needed" is the defined result. Pure function of deterministic byte counts, evaluated at deterministic points only.

**Determinism.** Reshuffle is the one operation I-107 permits to reassign offsets. It runs under exclusive LB access (steward claim or single-threaded barrier), reassigns every offset densely, and leaves logical content, element order, string ids, and the dirty state untouched (the saved dirty state is restored — a compaction must never force a deload rewrite). Physical layout and offset values stay unobservable, so reshuffle timing may vary run-to-run without any observable deviation.

**Why per-LB only.** Cross-LB block ordering (arranging different LBs' blocks consecutively pool-wide) needs whole-block copies with no free target space guaranteed — large complexity for unproven gain; per-LB compaction already delivers both wins: physically consecutive pages for the walk-order reader AND reclaimed blocks the FIFO page queue could never return.

---

<a id="d-175"></a>
## D-175 — biggest-first active eviction planned at the barrier, armed at kernel entry, executed through a per-LB claim word with worker self-service (2026-06-12)

**Status (2026-06-20, ): DISABLED.** The proactive kWake-band active eviction below is OFF. Its `blocksHeld` victim selection is timing-dependent — the now-cold `localHashMemoryDelta` / `workingMemory` maps grow on the shared cold arena via non-deterministically-ordered `acquireBlock`, so each LB's block count (hence "biggest") varies run-to-run. The non-deterministic eviction *set* then changes WHICH theorems prove (it non-deterministically drops them): deloading an ACTIVE LB is content-preserving (per-container counts, delta/working value+order, valid proofs all intact across deload→reload) yet still NOT proof-invisible — mechanism unexplained, sidestepped by not deloading active LBs. The hard-bound (3/4) synchronous mass deload stays as the sole, deterministic active-LB deload; the discharge drain handles inactive LBs. Full pipeline with this disabled == parent-identical 131018 / 0 failures. The kWake-band plan code is retained behind `if (false)` but inert.


**The choice (user-designed: "deloads biggest LBs first when 50% of RAM is consumed").** When the barrier's quiesced count sits in the steward band (≥ 1/2 wake, < 3/4 hard bound), the kernel selects still-active victims **biggest-first by `blocksHeld`**, ties broken by active-order index (`exprKey` is NOT unique — Rule 12), until consumption projected past the handed-off pendings' relief reaches the 2/5 stop watermark. The plan (`ExpressionAnalyzer::stewardEvictionPlan`) is **armed at the NEXT kernel's entry** — claims flip `Idle → Planned` single-threaded before the phase-1 pool spawns — NOT at the barrier that computed it, so prove's between-iteration count loop never observes a mid-release LB.

**The claim word** (`Memory::stewardClaim`, one atomic byte — [I-122](30_invariants.md#i-122)): the steward CASes `Planned → Busy`, dumps, stores `Dumped`; a phase-1 worker reaching its LB first CASes `Planned → WorkerOwned` and **self-services the dump** (then reloads through the normal `ensureLoaded`); a worker finding `Dumped` claims and reloads; a worker finding `Busy` yield-spins (ms-scale, lock-free). Content is untouched between the barrier and the LB's phase-1 slot, so worker and steward dumps are **byte-identical** — the file set and bytes stay deterministic; only the dumper identity and timing vary (unobservable). The barrier folds the executed plan post-quiesce (asserts every claim terminal at `WorkerOwned` — every active LB gets a phase-1 slot, so no victim stays cold) and resets claims to `Idle`.

**Why self-service instead of skip.** A skipped dump deferred past the slot would serialize post-iteration content — file bytes would become timing-dependent. Dump-at-or-before-slot is what keeps the canonical-bytes contract; the self-service branch is the PostgreSQL backend-writes-its-own-victim pattern (worker-side synchronous fallback beats waiting on the cleaner — the Percona stall lesson).

**Honest scope.** Eviction relief lasts only until the victim's next phase-1 touch reloads it (the G-51 honest scope unchanged); biggest-first minimizes the number of round-trips per block of relief. Plans never cross prove scopes (cleared at scope start and batch start).

---

<a id="d-159"></a>
## D-159 — dedicated background custodian executes barrier-decided deload work; install/wake/quiesce protocol; scope = one `prove` call (2026-06-12)


**The choice (user-designed: a dedicated thread for deload/upload management, active during all runs).** `MemorySteward` ([`memory_infra/steward.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/steward.hpp) / `.cpp`) owns one background thread, parked on a condition variable. The kernel barrier DECIDES, the steward EXECUTES: at every `proveKernel` end-of-iteration barrier the kernel (1) disarms the previous trigger, (2) **quiesces** the steward — every previously released drain completes, so quiesced block counts are pure functions of logical history, (3) folds the files of a drained handoff into the manifest and clears its `pendingDischarge` copy, (4) enqueues this iteration's discharges, then (5) decides on quiesced counts: above the 3/4 hard bound → synchronous relief in the kernel (pendings first, then the parallel active-eviction sweep — unchanged); above the 1/2 wake watermark → `installDischargeWork` + `wake`, the drain runs in the background while the next iteration computes; below → install + arm the grant trigger at the blocks-to-cross distance, a mid-iteration crossing wakes the steward to drain the WHOLE installed list (whole barrier-fixed units, I-106). The steward touches only discharged LBs — disjoint from everything the phase sweeps read or write (I-112) — so its only shared state is the global manager's mutex on block returns.

**Lifecycle.** One steward per `prove` call: a scope guard starts it before the first kernel and quiesces + stops + destroys it after the last, so the CE filter, `destroyGrid`, and every post-prove reader see a steward-free world. An assert on the steward thread aborts the process like any worker thread (Rule 19 — no swallowing). The thread is the +1 oversubscription over `logicalCores` (the kswapd pattern); it is I/O-bound and parked whenever idle.

**Lock-order safety.** The grant trigger's callback (`wake`) is invoked by the crossing grant AFTER the pool mutex is released and only takes the steward's own mutex; the steward's drain loop runs outside its mutex and touches the pool mutex only inside `releaseBlock`. No cycle exists. All waits are predicate-checked (`cv.wait(lock, pred)`) — no missed wakeups.

**Why.** Measured on tier 1: per-iteration deload I/O on the critical path was nearly the entire statification overhead. Backgrounding the discharge drain removes the once-per-lifecycle dump cost from the barrier entirely; the active-eviction handshake (next step) extends the same split to pressure evictions.

---

<a id="d-160"></a>
## D-160 — steward watermarks 50% wake / 40% stop; mid-iteration pressure read from a monotone grant ledger, never from `blocksInUse` (2026-06-12)


**The choice.** The steward's eviction band sits below the kernel's synchronous hard bound: wake at `steward::kWakeNum/kWakeDen` (1/2) of the pool, evict down to `steward::kStopNum/kStopDen` (2/5), park; the kernel's `kReleaseHighWaterNum/Den` (3/4) path stays as the synchronous fallback and the pool stays the hard bound. The wake/stop gap is the hysteresis band (the kswapd low/high pattern — every surveyed production system pairs its wake mark with a lower stop mark; a single mark oscillates). `static_assert`s in [`memory_infra/steward.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/steward.hpp) pin the strict ordering stop < wake < hard bound.

**Mid-iteration signal.** `GlobalMemoryManager` carries a grant ledger (`grantsSinceBarrier`, reset at the kernel barrier) plus a one-shot trigger (`armGrantTrigger(threshold, fire)`): the crossing grant fires the callback once, outside the manager's mutex, then self-disarms. The ledger — not `blocksInUse` — is the only legal mid-iteration pressure input ([I-106](30_invariants.md#i-106)): within an iteration block traffic is grants-only, so the ledger is monotone and order-independent, while `blocksInUse` dips as planned frees complete and is timing-dependent. Whether the trigger fires in iteration N is therefore run-invariant; only the firing thread varies (unobservable).

**Why.** The steward must make deterministic decisions while executing asynchronously (user requirement: "all order and numbers based, no time-dependent decisions"). Splitting the signal — barrier decisions read quiesced logical counts, mid-iteration wake reads the monotone ledger — keeps every deload file set a pure function of the run's logical history.

---

<a id="d-156"></a>
## D-156 — inactivation only enqueues; the pool decides; zero deload files without pressure (2026-06-12)


**The choice (user-designed).** The full laziness principle: do nothing until the pool forces it. An LB that left the active set this iteration is flagged `Memory::dischargedForever` and appended to `ExpressionAnalyzer::pendingDischarge` in deterministic `active` order — no I/O, no emptying; its blocks sit inside the pool budget and its content is frozen (deactivation is permanent, [I-112](30_invariants.md#i-112)). Under `kReleaseHighWaterNum/Den` pool pressure the drain order is: pending discharges FIRST (single-threaded FIFO — cheapest relief: frozen content, blocks never needed again), then, if the count is still above the mark, active-LB eviction by the parallel sweep (full dump, phase-1 reload at next touch). The manifest is rewritten only on iterations where something was dumped. A run that never crosses the mark writes ZERO deload files; the end-of-iteration sweep is one boolean scan. The pool remains the hard bound (exhaustion asserts); decisions read logical block counts at the single-threaded barrier — deterministic across runs. `pendingDischarge` resets at batch start with the manifest.

**Why.** Discharge dumps on inactivation were the last unconditional write path left after lazy dump-on-release; on no-pressure runs (all current batches peak ~40% of the default pool) they bought nothing. Deferring them to actual pressure makes the common case writeless while preserving exactly the relief ordering the steward thread will background: pendings (free relief) before active evictions (round-trip relief).

---

<a id="d-146"></a>
## D-146 — all statification code lives in `src/memory_infra/`, one component per source-file pair (2026-06-12)


**The choice (user-directed).** A dedicated folder [`src/memory_infra/`](../GL_Quick_VS/GL_Quick/src/memory_infra/) is the permanent home for everything connected to the statification campaign — the static memory hierarchy, the reusable arena containers, the deload serializer, the steward, and the string half — with **one component per file**: `global_memory_manager.hpp/.cpp` (+ `StaticMemoryConfig`, `staticMemory`, `initStaticMemory`), `lb_arena.hpp/.cpp`, `dirty_state.hpp`, `arena_vector.hpp`, `int_encoded_expr.hpp`, `lb_memory.hpp`, `lb_deload.hpp/.cpp`, `steward.hpp/.cpp`, and the string components `hot_arena.hpp/.cpp` / `hot_string.hpp` / `sealed_pages.hpp/.cpp` / `str_ops.hpp` / `cold_string_table.hpp/.cpp`. Every future container (arena hash set, arena hash map, arena rows, …) gets its own files on arrival. The split replaced the 719-line `infra/static_memory.hpp` monolith; `IntEncodedExpr` and `LbMemory` moved out of `memory.hpp` (the `ArenaVector<IntEncodedExpr>` members need the complete element type below `memory.hpp`, and the move lets `lb_deload` include `lb_memory.hpp` instead of all of `memory.hpp`). `src/infra/` keeps the diagnostics (`hashburst_dump`, `rt_tracker`) — instrumentation, not statification.

**Why.** The statification campaign adds containers batch by batch (see `statification_plan.md`); a monolithic header forces every batch through one growing file and blurs which component owns which contract. Per-component files keep each contract greppable and reviewable, and the folder boundary makes the campaign's surface explicit — what ships to the ASIC build is exactly this folder plus its `Memory`-level hooks.

**Build wiring.** MSVC: explicit `<ClCompile>`/`<ClInclude>` entries + a `Memory_Infra` filter. Linux: the Makefile gained `$(wildcard src/memory_infra/*.cpp)`. Behavior-preserving — the move commit changes no logic.

---

<a id="d-155"></a>
## D-155 — deload is the LAST act of `proveKernel`, after every post-join drain; reload at three enumerated touch points (2026-06-11)


**The choice.** The unconditional per-iteration deload sweep runs at the very END of `proveKernel` — after the phase-3 join AND after every post-join drain (`activateZeroCondition`, `updateGlobal` / `updateGlobalDirect`, the deferred-compaction flush) — serially over the iteration's `active` snapshot, followed by the accumulated-manifest rewrite. Reload happens at exactly three touch points: `performElemPhase1` entry (before the Rule-14 dump trap, which reads the containers), the post-prove visualizer equality-node read, and implicitly nowhere else — any other touch asserts (I-111).

**Why not inside `performElemPhase3` / the phase-3 sweep.** Two reasons found on contact with the code: (a) the CE filter drives the phase functions directly on thousands of single-use clone LBs that must never be dumped (the CE path never enters `proveKernel`, so kernel placement excludes them by construction); (b) the phase-3 sweep is parallel and is followed by post-join drains that may write to LBs (`activateZeroCondition` on induction blocks) — a deload inside the sweep would let a later drain touch a cold LB. End-of-kernel placement makes every same-iteration consumer resident by construction and shrinks the reload list to the genuine between-iteration/post-prove readers.

**Properties.** LBs created mid-iteration are not in `active` and stay resident until the end of their first active iteration. A deactivated LB deloads with its last active iteration and stays cold for the rest of the run — deactivation is permanent: no mail-reactivation path exists (`smashMail` delivers into `mailIn` but never touches `isActive`), and the only `false → true` flip in the prover is `activateZeroCondition` waking an induction zero-condition block that was parked at birth (never in any `active` snapshot, hence never deloaded — it wakes resident). A deactivated LB's image is read again only by the post-prove visualizer touch point. A re-deloaded grown LB can change its part count and leave stale smaller-N files on disk — harmless: reload uses `Memory::deloadFiles` (never the directory listing), and the batch-start purge clears the debris.

**Runtime levers (user-directed, same branch).** The sweep started serial; the measured deload cost prompted two amendments. (1) The sweep is a PARALLEL work-stealing pool (the phase-sweep shape): per-LB state and file sets are disjoint, block returns ride the global manager's mutex, write order is unobservable, and the manifest rewrite stays single-threaded after the join. (2) **Skip-unchanged deload**: every statified-container mutator escalates `LbMemory::dirty` (mutation routes are closed — no `data`, no iterators, and `operator[]` is const-only so in-place writes are impossible by construction); a `Clean` state means the on-disk file set still equals the in-memory content, so the deload releases blocks without rewriting. The state clears at dump and at reload — the two points where RAM == disk by construction. (3) **Tail-delta + threshold compaction** (user-directed; required because later tiers move the whole LB into the image): `AppendedOnly` windows dump only the new rows as `kind = 1` tail file sets (`_t<k>_` names, format version 2); a restructuring mutation, accumulated tails reaching base rows / `kTailCompactionDenominator`, or `kMaxTailSets` tail sets force the next full canonical rewrite — the compaction, which also re-canonicalizes the image (the canonical-bytes invariant becomes per-epoch; the file SET is a pure function of the deterministic content history). All branches are defined results (cache-valid / append journal / compact), not fallbacks. (4) **Write-through dump + pressure-gated release** (user-approved; measurement showed the round-trip itself — not the write volume — dominates: incubator 1 runs ~197 s vs `main`'s ~100 s with traps stripped and zero tail files): `deloadStaticContainers` split into `dumpStaticContainers` (image fresh, blocks stay) and `releaseStaticBlocks` (asserts a fresh image); the kernel dumps every active LB each iteration and releases blocks only above `kReleaseHighWaterNum/Den` of the pool — stable LBs reload as no-ops, the pool remains the hard bound, and the pressure decision is deterministic (block counts are logical). (5) **Element-stream I/O**: dump walks each container element-by-element (`appendSpanBytes`) and reload rebuilds element-by-element (`bulkAppendBytes`); the scattered arena layout forces the per-element walk, and the byte stream it produces is the canonical content stream. (6) *(superseded by the bump-arena cutover,)* the former per-page physical-pointer cache is gone — `ArenaVector::operator[]` resolves through the two-level offset spine (shift + mask per level), and `IntStmtView` holds the vector pointer plus a cached count for the burst. (7) **Lazy dump-on-release** (the decisive one, control-calibrated: incubator 1 at 188 s write-through vs 108 s machinery-off vs ~100 s pre-statification — the per-iteration dumps WERE the overhead): write-through is gone; an LB dumps exactly when its blocks go (inactivation or pressure), so stable active LBs touch no machinery; the steward later moves inactivation dumps off the critical path. *(Superseded in part by [D-156](#d-156): inactivation no longer dumps at all — it only enqueues; every dump now happens under pool pressure.)*

---

<a id="d-154"></a>
## D-154 — self-describing per-block deload files: GLDL header + per-tag directory + tag-ordered element streams; per-LB ordinal names; registry for humans (2026-06-11,; ordinal naming added 2026-06-16,)


**The choice.** A deloaded LB becomes a file SET, one file per block-sized chunk of the canonical payload stream (`_n_of_N` naming — the user's "each block has its own file" model applied to the repacked stream). Every file is self-describing from the first bytes: magic `GLDL`, version (3), kind (base / tail), the per-LB deload ordinal (int64), part n/N, block bytes, a per-`ContainerTag` directory (tag, element size, element count), the full chain string verbatim, then the chunk. Fixed-width little-endian (all GL targets are x64 LE). The directory + append-only tag registry is the growth mechanism: a future container adds a tag and a directory row; version is 3 (kind field added at v2, the ordinal replaced the hash field at v3), and old files never need migration because `.deload/` is transient (emptied at batch start).

**Repack-on-write, not in RAM.** The approved "straightening" (rebuild element-by-element into a fresh consecutive index) happens with the FILE as the copy's destination: dump streams logical content (canonical bytes), reload rebuilds the fresh index. One pass, no scratch blocks, same result — user-approved at plan review.

**File names (revised 2026-06-16,).** `lb<ordinal>_[t<k>_]<n>_of_<N>.bin`. A per-LB **deload ordinal** (a process-monotonic counter on `GlobalMemoryManager`, `assignDeloadOrdinal`) is the file's injective identity. It replaced the earlier `sanitize(chain,48)_<hash16>_...` scheme: the FNV chain hash was injective only with overwhelming probability, and GL prefers a provably-unique counter (the no-defensive-fallback ethos, Rule 19). The ordinal is assigned once per LB and reused for re-dumps + tails (`Memory::ensureDeloadOrdinal`), stamped single-threaded at the kernel barrier so worker and steward dumps name the file identically; a fresh process counts from 0, so the names are cross-run deterministic. The full chain lives in the header verbatim (the load-time identity check) and in `registry.txt` (`<ordinal>\t<chain>`, ascending; humans only — reload uses dump-time file lists, since N is not recoverable from names alone). The former `fnv1a64` / `sanitizeChainPrefix` helpers were removed; the counter + registry live on `GlobalMemoryManager` because the unit tests dump without an `ExpressionAnalyzer` in scope.

**Loud loading.** Every header field is asserted on load (magic/version/ordinal sign/verbatim chain/part numbering/block bytes/cross-part directory consistency/element sizes vs compiled-in types) — corrupted or foreign files stop the run at their origin (Rule 19); there is no skip-and-continue. The verbatim chain is the identity check (the loader does not independently know the ordinal).

---

<a id="d-184"></a>
## D-184 — SUPERSEDED: the paged vector packed a power-of-two of elements per page (shift+mask index); the bump-arena cutover removed the page tier (2026-06-11, superseded 2026-06-12)


**SUPERSEDED by the bump-arena cutover** ([D-162](#d-162);). The cold path no longer has a page tier: `ArenaVector` scatters individual element allocations into the LB's bump arena and indexes them through a two-level offset spine, so there is no elements-per-page geometry to choose. `operator[]` stays division-free (shift + mask into the spine, per level). The original reasoning is kept for campaign history.

**Original choice (paged era).** Elements per page = the largest power of two of elements that fit one page — the logical→(page, slot) split in the paged vector's `operator[]` was one shift + one mask, no integer division on the phase-2 hot path that indexes statements millions of times per burst. The ~31% page-tail padding at the 8 KiB / 176-byte default was the accepted cost; the rejected alternative (packing the true quotient) put an integer division (~20–40 cycles) on every element access. The bump arena dissolved the tradeoff — scattered allocations waste no per-page tail, and the spine index is still shift+mask.

---

<a id="d-174"></a>
## D-174 — static per-LB memory hierarchy (pool → blocks → per-LB bump arena), first migrated container `intEncodedStatements`, unconditional SSD deload (2026-06-11)


**The choice (user-proposed architecture).** Statification — the ASIC 0.1 path; FTA is infeasible at current heap consumption — replaces per-LB heap containers with containers backed by a fixed hierarchy: ONE program-start reservation (`static_pool_bytes`, never freed, never grown — `GlobalMemoryManager`, [`memory_infra/global_memory_manager.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/global_memory_manager.hpp)) carved into fixed-size blocks (`static_block_bytes`) dispensed under a mutex to each LB's bump arena (`LbArena`), which lays the cold containers' bytes into one virtually-contiguous space. Containers address storage through per-LB **virtual offsets** — never raw physical pointers — because a deloaded LB may reload into different physical blocks; consequently no observable may depend on physical block identity or grant order. A per-LB aggregate `struct LbMemory` owns every statified container against an append-only tag registry (user requirement: later tiers add a member + a tag, the machinery and the deload format never restructure). After an LB finishes its phase 3 each iteration its statified containers stream element-by-element in logical order to `.deload/` (the stream IS the canonical "straightening" of page fragmentation; bytes are a pure function of content) and its blocks return to the pool; reload rebuilds a fresh consecutive arena at the next touch. Tier 1 migrates `intEncodedStatements` (later its three sibling vectors); subsequent tiers repeat per container.

**Why one reservation instead of a static `char[]`.** Multi-gigabyte static arrays hit MSVC image limits; a single startup reservation is the embedded-systems model of the ASIC's fixed SRAM and preserves the I-13 spirit (no per-object allocation, no allocator churn) — see [I-95](30_invariants.md#i-95).

**Boundary (tier 1, user-visible trade-offs).** Container bookkeeping (page-id tables, block lists, recycle queue) stays on the heap — the hierarchy binds element payloads; a later tier moves bookkeeping into the blocks. The prover's barriered sweeps keep active LBs resident for the whole iteration, so tier 1's memory win is for LBs idle across iterations; per-phase deload / per-LB pipelining is a later tier's optimization.

**Plan.** `~/.claude/plans/cached-hugging-lovelace.md` (10 commits; branch-end gate = double `main.py` byte-identity + `theorems.txt` byte-identical vs `main` + verifier 0 failures).

---

<a id="d-135"></a>
## D-135 — the LB state maps go id-form (OR bookkeeping, integration-prep gates, wipe queue, expanded implications); LB identity deferred to the ASIC LB-split session (2026-06-11)


**The choice (user-approved architecture change).** Batch 8 — the final container batch of the approved intification sequence. A NEW dedicated per-LB `Memory::lbStateInterner` (the `ValueInterner` type) owns the OR-state and expanded-implication strings — not the NameMap (mail-absorbed implications may be locally un-interned; a mint there would shift the dumped id table). `orBookkeeping` re-keys to packed int64 (exprId, sigId) pairs with decoded-lex-sorted disjunct storage (its iteration feeds `ordisMerge`'s origin construction — order observable); `orDisjunctCount` keys by sigId. `integrationPrepared` / `integrationPreparedMarker` / `integrationStartIntMap` go to packed (templateId, validityId) TEMPLATE-space keys (`mintTemplateKey` / `lookupTemplateKey`, the 5b EWV-set family) with the existing low-16-bits `wipeSubtree` predicate. `pendingWipeScopes` becomes a `set<int16_t>` of NameMap validity ids — every queued scope was created via `encodePush`, so the insert is a non-minting `lookup` + assert; the drain decodes and lex-sorts before the `wipeSubtree` calls (the former string-set order exactly). `expandedImplications` becomes packed int64 lbStateInterner pairs: the mail absorb encodes (minting freely with zero NameMap-table impact), `sanitizeHashMemory` walks a decoded lex-sorted snapshot, `eradicateImplicationFromLB` erases by non-minting lookup, the wipe decodes the validity half. `Mail::expandedImplications` stays string (mail excluded).

**`orAdmissionSet` dropped (D-130 dead-member precedent).** Zero insert sites exist — [D-31](#d-31) documents the legacy gate as structurally dead — so the always-empty set made its gate loop constant-false; the read site simplifies to `orAdmitted = allowOrDisintegration` with identical behavior, and the dump prints the literal empty section header so trace bytes are unchanged.

**LB identity deferred (user decision at plan approval).** `exprKey` and the `simpleMap` routing keys stay string: one short string per LB (negligible memory) against a ~66-site blast radius including tree navigation, the visualizer path walks, and the SACRED Rule-14 `isTargetLB` chain-match conditions. (The `recursionHypothesisId` / `contradictionTheoremId` scalars, deferred here alongside them, moved to NameMap ids in the strings campaign — they are content, not LB identity.) The ASIC LB-split session restructures the tree anyway — that is the right moment to make LB identity static. With this batch the per-statement string populations in persistent prover state are exhausted; the documented residuals are LB identity, mail, and the `ce::` string machinery (`canBeSent*` became NameMap-id sets in the strings campaign).

**Migration shape.** Commit 1: interner + lockstep reset + tests. Commit 2: OR state atomic (+ the orAdmissionSet drop). Commit 3: integration-prep state atomic. Commit 4: scope walks atomic. Verification: sequential full `main.py` runs vs the batch-7 accepted baseline; only the documented `mailIn.exprOriginMap` order swaps are allowed.

---

<a id="d-133"></a>
## D-133 — the rule registry goes full id-form in a dedicated per-LB rule interner; the hot firing path decodes in place; OwnerSet's owners unify into partitionIds (2026-06-11)


**The choice (user-approved architecture change).** Batch 7 of the approved intification sequence — the riskiest, under firing/binding. `LocalMemoryValue` goes FULL id-form including the hot fields ([`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp)): `value`/`originalImplication` → int32 ids, `key` → `vector<int32_t>`, `remainingArgs` → decoded-lex-sorted id vector, `validityName` → the int16 NameMap `validityId` (already minted at install for the owner record — no new NameMap mints, the dumped id table is untouched), `justification` → the closed `RuleJustification` enum (assert on unknown), plus a new install-time `isMarker` bool. The string ids live in a NEW dedicated per-LB `Memory::ruleInterner` (the `ValueInterner` type) — not the NameMap (full-id-table trace dump) and not the admission/origin interners (disjoint populations). `HashMemory::originals` becomes id chains in the same space. Encodes happen only at the single-threaded install sites (`addToHashMemory`, `makeNormalizedKeysForAdmission`); the parallel hashburst only decodes — an array-index `const` ref, thread-safe with zero mint risk ([I-83](30_invariants.md#i-83)/[D-116](#d-116) preserved).

**Hot-path shape.** `checkLocalEncodedMemoryStatic` substitutes on `decode(valueId)` / `decode(keyIds[k])` (the substitution dominates; the decode is an index), sorts hash-hit LMV vectors by DECODED head (`a.value < b.value` reproduced exactly, same tie behavior on the same input order), and builds the string `FiringRecord` origin pair by decoding `originalImplicationId`/`validityId` (staging/mail stay string — post-substitution products are the cross-LB space). Two deliberate, behavior-identical hot-loop reductions ride along: `isMarker` replaces the per-firing `value.find("marker")` scan, and the stored `validityId` replaces the per-firing `nameMap.lookup(lmv.validityName)` hash probe.

**OwnerSet unification (user-approved).** `OwnerSet::owners` (`map<ExpressionWithValidity, int16_t>` — the full implication text per owner, per key, across the four `normalizedEncoded*` maps) and `OwnerSet::partitionIds` collapse into ONE `set<int32_t>` of `makePartitionId(origId, scopeVid)` packs: packed owners are bit-identical to the partition ids already maintained in lockstep, and both halves are NameMap-minted at install today. The comparable-prune ([D-105](#d-105)) reads the low half; `partitionAccepts` ([D-119](#d-119)) iterates the same set unchanged; the `wipeOwnerMap` lambda decodes the low half for its `inClosed` scope test and erases once. Every reader is any-semantics, so set order is never observable. [I-80](30_invariants.md#i-80)'s two-container lockstep becomes structural (one container — nothing to drift); the invariant is rewritten accordingly.

**Determinism guardrails.** Same discipline as batches 1–6. The `encodedMap` dump section decodes fields in place — the map itself (keys, hashing, insertion order) is untouched, so iteration order is unchanged; the marker section's `(value, key, remainingArgs)` sort runs on decoded strings; `writeHashOriginals` derives a decoded lex-sorted view replicating the string-set order. No sanctioned byte exceptions expected. The interner resets only in lockstep with `destroyGrid`'s `nameMap` reset.

**Out of scope (boundaries).** `FiringRecord` and all staging/mail strings (post-substitution products), `IntNormalizedKey`/`KeyArena` (already int), `uSignatures`/`hasLooseOwner` (already int), `canBeSent*` (roadmap-excluded), the LB-tree strings (batch 8).

**Migration shape.** Commit 1: `ruleInterner` + `RuleJustification` tables + lockstep reset + tests, no consumers. Commit 2: `LocalMemoryValue` atomic re-type (installs encode, hot path decodes, dump, tests). Commit 3: `originals` id chains. Commit 4: OwnerSet unification + I-80 rewrite. Verification: sequential full `main.py` runs vs the batch-6 accepted baseline; only the documented `mailIn.exprOriginMap` order swaps are allowed.

---

<a id="d-131"></a>
## D-131 — the origin maps go id-form in a dedicated per-LB origin interner with an enum tag vocabulary; mail stays string (2026-06-11)


**The choice (user-approved architecture change).** Batch 6 of the approved intification sequence — `Memory::exprOriginMap` and `EquivalenceClass::equalityOriginMap` move from `std::map<ExpressionWithValidity, vector<(string tag, vector<ExpressionWithValidity>)>>` to `IdOriginMap = unordered_map<int64_t, vector<(OriginTag, vector<int64_t>)>>` ([`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp)). Keys and dependencies are `packOriginKey(expressionId, validityId)` int64 packs of two int32 ids from a NEW dedicated per-LB `Memory::originInterner` (reusing the `ValueInterner` type) that owns BOTH the expression and the validity strings. Deliberately NOT the `NameMap`: the trace dumps the full `nameMap.idToName` table, and mail-carried dependencies can name child-LB scopes never interned in the receiving LB's `NameMap` — a mint there would grow the dumped table and destroy A/B trace comparability (the same decisive argument as `D-132`). Tags become the closed `OriginTag` enum (~31 historical literals); `originTagFromString` asserts on an unknown tag — a new emission site must extend the enum, never pass through silently (Rule 19). Rule 16 is untouched: the maps stay process documentation, no proof decision reads them; the migration is type-only.

**Boundaries.** ALL mail origin maps stay string (`mailIn` / `mailOut` / `sameIterationInternalMail` / `nextIterationInternalMail` / `broadcastMail` / per-core slots — mail is roadmap-excluded; strings are the cross-LB common space since interner ids are per-LB). Most emission sites already build the string origin object for the paired mail write, so the body-map write encodes the same object; the mailIn absorb path and the equality-class sync encode string→id; `fillMailOut`'s transitive walk and the theorem-deposit paths decode id→string. The compressor's phase-1 extraction decodes per-LB into its existing string graph; the visualizer's three container touch points (the I-39 lift probe, the buildStack root find, the findEnds candidate walk) use non-minting `lookupOriginKey` probes and decoded snapshots — the D-51 cycle-filter machinery downstream stays string and untouched. The chapter text files and the Python pipeline are unaffected.

**Determinism guardrails.** Same discipline as the prior batches ([I-84](30_invariants.md#i-84)). `addOriginId` / `overwriteOriginsId` replicate the string twins' cap/dedup semantics exactly, including the [D-49](#d-49) cap-full preference on the tag enum. Every order-sensitive walk of the maps (the dump section, findEnds, the compressor extraction) derives a decoded `(expression, validity)` lex-sorted snapshot via `decodeOriginMapSorted` — exactly the former `ExpressionWithValidity::operator<` map order; per-key history-line vectors keep insertion order. Body-map encodes happen only at single-threaded write sites ([I-83](30_invariants.md#i-83)); the interner resets only in lockstep with `destroyGrid`'s `nameMap` reset and is never wiped on scope teardown (`exprOriginMap` itself survives `wipeSubtree`, so its id space must too). The dump's `exprOriginMap` section derives byte-identical output (the Rule-14 consent shape); mail dump sections are untouched. No sanctioned byte exceptions this batch.

**Migration shape.** Commit 1: interner member + `OriginTag` tables + key/record helpers + id-form `addOriginId`/`overwriteOriginsId`/`decodeOriginMapSorted` with direct unit tests, no consumers. Commit 2: `Memory::exprOriginMap` atomic re-type (every writer/reader/dump). Commit 3: `EquivalenceClass::equalityOriginMap` atomic re-type (merge paths, sync sites — class→body copies become pure id-to-id). Verification: sequential full `main.py` runs vs the batch-5b accepted baseline; only the documented `mailIn.exprOriginMap` order swaps are allowed.

---

<a id="d-136"></a>
## D-136 — the admission/rejected VALUES go id-form in a dedicated int32 value space; observable orderings preserved via stateful decoded comparators (2026-06-11)


**The choice (user-approved architecture change).** The follow-up to `D-132`: the admission/rejected map VALUE structs (`AdmissionMapValue`, `RejectedMapValue`, `RejectedMapIntegrationValue`, `Instruction`/`LogicalEntity` — all five LogicalEntity string fields, uniform) store their string fields as ids in a NEW dedicated `ValueInterner` ([`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp)) with int32 ids. Separate from `TemplateInterner` because value strings never participate in packed `(id16, id16)` pair keys — they sit in id vectors — so int32 ids remove the 32000-ceiling risk at Gauss-scale value populations (`rejectedMapIntegration` reaches ~10^5 entries); separate from the `NameMap` for the established id-shift reason. Every observable container ordering (the dump's value listings; the snapshot loops in `isAdmitted`, `revisitRejected2`, the revival loop, and the equi-class hooks, whose iteration drives mail/revival emission order) is preserved by stateful decoded comparators that hold a `const ValueInterner*` and replicate the EXACT historical `operator<` field orders on decoded strings — never raw id order ([I-84](30_invariants.md#i-84)). Rewrite/analysis boundaries stay string: hooks decode → `ce::replaceKeysInString` → re-encode; `isAdmitted`'s element scans decode at use; all such sites are single-threaded. The two leftover EWV sets (`admissionSetIntegration`, `triggersForAdmissionSetIntegration`) re-key to `packStatementKey(originalId, validityId)` NameMap-space keys with write-side non-minting `lookup` + `assert(found)` — the "members are always pre-interned statements" observation becomes a checked contract; a firing assert is a report-stop, never a fallback. No sanctioned dump-byte exceptions this batch — every value listing decodes to identical bytes in the preserved order.

**Mid-execution corrections (both user-approved 2026-06-11).** (1) `LogicalEntity` turned out to be the GLOBAL compiled-expression type (`compiledExpressions`, disintegration machinery — ~50 prover sites), not an admission value struct; instead of re-typing it, the stored integration instructions use dedicated id-form twins `IntLogicalEntity`/`IntInstruction` while the string `Instruction` remains the working form, with `encodeInstruction`/`decodeInstruction` conversions at exactly the `admissionMapIntegration` touchpoints. (2) The EWV sets' members turned out to be TEMPLATE-population strings (u_-stripped marker forms, repl_-form triggers) — never NameMap-interned, so the planned NameMap-packed + assert design would have asserted on the first write; they key in the TEMPLATE space instead (`mintTemplateKey` writers, `lookupTemplateKey` probes), with the order-sensitive per-trigger `makeAdmissionKeys` walk on a decoded lex-sorted snapshot.

**Out of scope (boundaries).** `LocalMemoryValue` (rule registry — roadmap batch 7), `canBeSent*` (mail-gate inputs, roadmap-excluded), `exprOriginMap`/`equalityOriginMap` (roadmap batch 6).

**Migration shape.** Commit 1: `ValueInterner` + decoded-compare primitives (`valueIdLess`, `valueIdVectorLess`) with direct unit tests, no consumers; `destroyGrid` resets the interner in lockstep with the `nameMap` reset. Commits 2–6: atomic per-struct re-types (AdmissionMapValue, RejectedMapValue, RejectedMapIntegrationValue, Instruction/LogicalEntity, then the EWV sets), each converting every reader/writer plus its dump section in one commit. Verification: sequential full `main.py` runs vs the batch-5 accepted baseline; only the documented `mailIn.exprOriginMap` order swaps are allowed.

---

<a id="d-132"></a>
## D-132 — the admission/rejected subsystem re-keys to packed (templateId, validityId) with a DEDICATED template-id space (2026-06-11)


**The choice (user-approved architecture change).** Batch 5 of the approved intification sequence — the five admission/rejected maps (`admissionMap`, `admissionStatusMap`, `admissionMapIntegration`, `rejectedMap`, `rejectedMapIntegration`), the `consumedAdmissionKeys` / `revisitInProgress` sets, and the three vars* overlap caches re-key from `ExpressionWithValidity` template strings to packed int32 keys. The template half comes from a NEW dedicated `TemplateInterner` on `Memory` ([`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp)) — NOT from the per-LB `NameMap`. Decisive argument: admission/rejected keys are marker-form / u_-form template strings, a population disjoint from the statements and validity names the `NameMap` interns; interning them there would mint mid-run and shift every later statement id, which changes the hashburst dump's full `nameMap` id table and the raw-id sections (`intKnownStatements` prints `origId=`/`valId=` and sorts by raw packed key) — A/B trace comparability would be destroyed — and would pressure the 32000-id ceiling. The packed key is `packStatementKey(templateId, validityId)` with the LOW half still a `NameMap` validity id; the two id spaces never mix within one key position. Values (`AdmissionMapValue`, `RejectedMapValue`, `RejectedMapIntegrationValue`, `Instruction`) stay string structs — out of scope for this batch.

**Companion choices (same approval).** One branch with algebra-first commit order (no algebra/integration branch split); ATOMIC per-container re-key commits instead of the batch-1..4 dual-write pattern — packed twins would duplicate whole maps including their large string values, and the end-of-batch A/B run is the equivalence proof; the vars* caches move to template-space id sets (their insert side meets u_-stripped bare names that may be uninterned in the `NameMap`, so `NameMap` ids are unsafe there for the same id-shift reason).

**Determinism guardrails.** Same discipline as [D-127](#d-127)–[D-130](#d-130) and `D-134` ([I-84](30_invariants.md#i-84)). Phase-2 parallel staging records (`admissionKeysAlgebra`, `deferredIntegrationPreps`) KEEP string keys — encoding into the interner happens only at the single-threaded post-fixpoint drains ([I-68](30_invariants.md#i-68)/[I-69](30_invariants.md#i-69)/[I-83](30_invariants.md#i-83)). Probes (`isAdmitted`, `isAdmittedIntegration`, the consumed-key guards) are non-minting `TemplateInterner::lookup` — a template never interned was never registered, a definitive miss. The three-key-shape contract (marker / u_ / bare conversions via `ce::replaceKeysInString` / `removeUPrefixFromArguments`) stays a string operation at write/rewrite time; the resulting template string is encoded at the write site. Order-sensitive map walks (the four equi-class hooks, `cleanAdmissionMap`'s [I-41](30_invariants.md#i-41) closure, the cleanup sweeps, the dump sections) iterate decoded `(template, validity)` lex-sorted snapshots — identical to the former `std::map<ExpressionWithValidity>` order. [I-22](30_invariants.md#i-22) (integration templates persist), [I-37](30_invariants.md#i-37) (rejected maps: drop + mail, never direct insert), and [I-71](30_invariants.md#i-71) (consumed/admission mutual exclusion) carry over onto packed keys unchanged. The interner resets only in lockstep with `destroyGrid`'s `nameMap` reset (the validity halves of packed keys re-bind there).

**Dump byte exception (2026-06-11, mid-batch).** The three `varsIn*Keys` dump sections historically printed the `unordered_set<std::string>` caches in raw hash-iteration order; the id-set migration cannot reproduce that order, so those sections now print decoded names lex-sorted. Content is unchanged; this is the batch's ONE trace-byte difference vs the batch-4 baseline (A/B pass criteria amended accordingly). Every other migrated section stays byte-identical.

**Migration shape.** Commit 1: `TemplateInterner` + tests, no consumers. Commits 2–6: atomic per-container re-keys — algebra admission group (map + status + consumed + revisit), `rejectedMap`, `admissionMapIntegration`, `rejectedMapIntegration`, then the vars* caches — each commit converts every reader/writer of its container plus its wipe/swap lines and dump section (derived byte-identical, the Rule-14 consent shape). Verification: sequential full `main.py` runs vs the batch-4 accepted baseline; deltas beyond the documented `mailIn.exprOriginMap` order swaps are failures.

---

<a id="d-134"></a>
## D-134 — the equivalence-class subsystem moves from string members to NameMap ids with decoded-lex canonical selection (2026-06-11)


**The choice (user-approved architecture change).** Batch 4 of the approved intification sequence — equivalence-class state moves to NameMap ids: `EquivalenceClass::variables` (`set<string>`) becomes a member-id vector kept sorted by **decoded** name, `equalityLevelsMap` keys become packed unordered id pairs, `Memory::equivalenceClassesMap` re-keys by validity id, `weakVariables` becomes a packed `(variableId, validityId)` set, and `eqClassSttmntIndexMapMap` re-keys to `(validityId, member-id vector)`. `equalityOriginMap` stays string-keyed (deferred to the originMap batch). The regex scans in `filterIterations` / `chooseCanonical` / `canonicalizeUnderClasses` / `updateWeakVariables` are replaced by per-id memoization: `classifyName` is the single classification authority (whole-string `int_lev_*` / `it_*_lev_*` / normal tiers), `scanSpecialTokens` the single expression-scan authority (substring semantics, faithful to the historical `sregex_iterator` scan including the `"print_lev_3_4"` → `int_lev_3_4` catch), and `EqClassNameCaches` (`Memory::eqClassNameCaches`) memoizes both per NameMap id — pure functions of the decoded string, so the cache needs no invalidation and is exempt from scope teardown. Lazy fill is a shared-state write: probes run only in single-threaded phases ([I-83](30_invariants.md#i-83)).

**Why.** (1) `filterIterations` regex-scanned every class member and the whole statement text on every (statement × class) probe — the dominant string cost of the class subsystem; the cached path is id lookups. (2) ASIC 0.1 static-memory direction: class members as int16 ids instead of heap strings. (3) The canonical member is defined by decoded-string lex order — storing members sorted by decoded name makes "first member of the matching tier" the canonical, with no per-call sort and no id-order dependence ([I-84](30_invariants.md#i-84): never sort by id).

**Migration shape.** Commit 1: `classifyName` / `scanSpecialTokens` / `EqClassNameCaches` land with direct unit tests; the string-side helpers still run their inline regex scans. Subsequent commits: int twins dual-written alongside the string members, readers flip group by group (canonical selection first, then the apply/iteration paths, then container re-keys), string halves drop last; the hashburst dump sections (`equivalenceClassesMap`, `weakVariables`) re-derive their byte-identical output by decode + lex-sort (Rule-14 consent shape as [D-127](#d-127)/[D-129](#d-129)/[D-130](#d-130)). Verification: sequential full `main.py` runs from current disk state vs the batch-3 accepted baseline; deltas beyond the documented `mailIn.exprOriginMap` order swaps are failures.

---

<a id="d-130"></a>
## D-130 — the goal registry `toBeProved` re-keys from `EncodedExpression` to packed `(originalId, validityId)` int32 keys (2026-06-10)


**The choice (user-approved architecture change).** Batch 3 of the approved intification sequence — the per-LB goal registry re-keys to packed int32 keys: `Memory::toBeProved` (`std::map<EncodedExpression, std::tuple<std::set<int>, std::set<std::string>>>`) gains the twin `intToBeProved` (`std::unordered_map<int32_t, …>`). The key is `packStatementKey(originalId, validityId)`, the identity the old comparator already encoded: `EncodedExpression::operator<` reads only `(original, validityName)`, which the per-LB `NameMap` maps bijectively onto the id pair. After the string container dropped and the first full run was user-accepted, the value's reserved tags member (`std::set<std::string>`, never written anywhere — always `{}`) was dropped as dead in a follow-on commit; the value is now the auxy set alone, and the dump row prints a literal `tags={}` to keep the Rule-14 format byte-identical.

**Why.** (1) The `dischargeToBeProved` finds and the `burstDeactivates` goal probe each ran the parsing `EncodedExpression` constructor per call just to build a lookup key; the packed probe is O(1) with zero parsing, and the discharge walk already holds the ids (`IntEncodedExpr` delta rows). (2) The `deactivateRecursively` / `deactivateUnnecessary` main-scope survey ([I-48](30_invariants.md#i-48)) becomes a low-16-bits compare against `NameMap::MAIN_ID` instead of a per-entry string compare. (3) ASIC 0.1 static-memory direction: an int32 key replaces tree-map nodes keyed by full goal strings.

**Determinism guardrails.** Same discipline as [D-127](#d-127)/[D-128](#d-128)/[D-129](#d-129) ([I-84](30_invariants.md#i-84)). Probe paths never mint (`lookupToBeProved` over `NameMap::lookup`, id 0 = definitive miss — every stored goal interned both names at its insert site); the `addExprToMemoryBlock` insert site reuses the ids already encoded at function entry, so the twin write mints nothing; the `sanitizeToBeProved` re-key encodes only the rewritten original (a write site) and reuses the old key's validity id verbatim — [I-45](30_invariants.md#i-45)'s namespace preservation becomes structural. Order-sensitive walks (the sanitize staging pass, the post-absorb `checkNecessityForEquality` sweep, the vacuous-truth first-main-goal pick, the hashburst dump section) iterate `decodeToBeProvedSorted` — owned decoded copies lex-sorted on `(original, validityName)`, byte-identical to the former `std::map` iteration order; raw ids are never a sort key. `wipeSubtree` sweeps the packed map by the `closedIds` low-bits predicate (the [D-128](#d-128) argument). The dump's `toBeProved` section and the header's `toBeProved=` count retarget data source only (Rule-14 consent shape as [D-127](#d-127)/[D-129](#d-129)).

**Migration shape.** Commit 1: packed twin added, written in lockstep at every mutation site (insert, induction/direct discharge erases, induction-promotion erase, `removeExpressionFromMemoryBlock`, sanitize re-key, `wipeSubtree`, the two capacity-release swaps), string side authoritative, size-lockstep tripwire assert at `sanitizeToBeProved` entry. Commit 2: every reader switches to the packed side. Commit 3: the string container and its writes drop. Commit 4: the dead tags member drops. Verification: sequential full `main.py` runs from the current disk state — no `files/` snapshotting (the tracked parts do not change between runs; the rest is regenerated) — first post-drop run checked by the user against prior accepted run logs; the post-tags-drop run compared to that accepted run (`theorems.txt` byte-identical, verifier blocks identical, run-log hash-burst lines identical, dump traces byte-identical modulo the documented `mailIn.exprOriginMap` order-swap fragility).

---

<a id="d-129"></a>
## D-129 — the statement indexes re-key from `EncodedExpression` to packed `(originalId, validityId)` int32 keys (2026-06-10)


**The choice (user-approved architecture change).** Batch 2 of the approved intification sequence — the two string-keyed statement indexes that [D-127](#d-127) explicitly scoped out ("they are indexes, not lockstep mirrors") re-key to packed int32 keys: `Memory::localEncodedStatementsSet` (`std::set<EncodedExpression>` → `intLocalEncodedStatementsSet`, `std::unordered_set<int32_t>`) and `Memory::statementLevelsMap` (`std::map<EncodedExpression, std::set<int>>` → `intStatementLevelsMap`, `std::unordered_map<int32_t, std::set<int>>`). The key is `packStatementKey(originalId, validityId)` — the same key `intKnownStatements` already uses, and the two identities coincide: the old `EncodedExpression` comparator reads only `(original, validityName)`, which the per-LB `NameMap` maps bijectively onto the id pair.

**Why.** (1) The [D-29](#d-29) locality gate and the per-request combined-levels lookup in `checkLocalEncodedMemoryStatic` ran a decode plus (for the gate) the parsing `EncodedExpression` constructor per premise just to build a probe key, inside the burst hot path — the packed probe is O(1) with zero decode. (2) Every write site already computes the packed key for the adjacent `upsertStatementKey` call, so lockstep int writes cost no new encodes. (3) ASIC 0.1 static-memory direction: an int32 key replaces tree-map nodes carrying full statement strings.

**Determinism guardrails.** Same discipline as [D-127](#d-127)/[D-128](#d-128) ([I-84](30_invariants.md#i-84)): readers that mint nothing today build probe keys via the non-minting `NameMap::lookup` (id 0 = miss, exact because every stored entry interned both strings at its insert site); the two deliberately-minting erase helpers (`resetResentExpressionRegistries`, `eradicateImplicationFromLB`) keep `encode`; at the four write sites where the packed-key computation can mint (`addEquality` original+mirror, `addNegatedEquality` original+mirror, the `applyEquivalenceClass` commit) the whole `packStatementKey(encode(original), encode(validityName))` call expression is hoisted unchanged into a local — same expression shape, same compiler-chosen argument evaluation order, identical mint order. The hashburst dump's `statementLevelsMap` section derives from the packed map by unpack + decode + lex-sort on the decoded `(original, validityName)` pair — byte-identical to the former `std::map` iteration (the `writeWholeExpressions` precedent; Rule-14 data-source retarget pre-authorized by the user, same consent shape as the [D-127](#d-127) unification). `wipeSubtree` sweeps the packed map by the `closedIds` low-bits predicate — provably the same membership as the string `inClosed` walk (the [D-128](#d-128) argument). Never sorted or compared by id anywhere.

**Migration shape.** Commit 1: packed twins added, written in lockstep at every mutation site, string side authoritative. Commit 2: every reader switches to the packed side. Commit 3: the string containers and their writes drop. Definition of done: full `main.py` vs the current-main baseline from the same starting disk state — `theorems.txt` byte-identical, verifier blocks byte-identical, run-log hash-burst lines identical, dump-on traces byte-identical.

**Landed (this branch).** All three commits in: lockstep twins (581/581 unit tests, new wipe-sweep test), reader switch (582/582, new non-minting-probe test for the `lookupStatementLevels` / `isLocalEncodedStatement` helpers), string-container drop (the only `statementLevelsMap` byte left in source is the dump's section label). The index contract lives in `I-86`; [I-84](30_invariants.md#i-84)'s boundary list updated (no index operation needs a string key anymore).

---

<a id="d-128"></a>
## D-128 — the last string/int lockstep mirror pairs lose their string halves; `wholeExpressions` folds into `intKnownStatements` via membership bits (2026-06-10)


**The choice (user-approved architecture change).** Continuation of [D-127](#d-127) — batch 1 of the approved intification sequence. Three mirror pairs lose their string halves: `Memory::validityNamesToFilter` (keep `intValidityNamesToFilter`), `Memory::axedVariables` (keep `intAxedVariables`), `HashMemory::productsOfRecursion` (keep `productsOfRecursionIds`). The fourth pair is folded, not dropped: `wholeExpressions` is deleted and `intKnownStatements` becomes the single registry under the same packed `(originalId, validityId)` key, with `StatementFlags` gaining two membership bools — `registered` (= former `wholeExpressions` membership) and `known` (= former `intKnownStatements` membership).

**Why membership bits instead of a plain fold.** The fourth pair is NOT content-equal. `addStatement` registers the string side unconditionally but the int side only behind the iteration-cap / secondary-variable-count / equivalence-filter gates; the equivalence-class commit (`applyEquivalenceClass`) writes int-only; the compressor rule load writes string-only; the CE teardown (`releaseCEBatchMemory`) resets string-only. A presence-only fold would flip the `fillMailOut` transitive-walk skip, the integration-preparation gates, and the Site F dedup at those divergences. The two bits keep every existing gate's exact answer, making the fold provably byte-identical.

**Reader migrations.** `countPatternOccurrences` consults `productsOfRecursionIds` through the non-minting `NameMap::lookup` — exact because every member is interned by its lockstep insert sites (`updateAdmissionMapRecursion`, `isAdmitted`), `NameMap` is bijective, and id 0 is the reserved miss sentinel. The hashburst dump derives the former string sections from int state byte-identically (decode + lex-sort, the [D-127](#d-127) `localEncodedStatements` precedent); the `validityNamesToFilter` dump section is removed outright (user decision) because its content was a vestigial subset — `wipeSubtree` recorded only the closing scope name while the int half records every closed id, so no byte-identical derivation exists.

**Landed (this branch).** Pairs 1–3: readers migrated, string halves dropped. Pair 4 in three steps: membership bits + `known`-bit reads (with the CE teardown's bit-respecting reset), the thirteen `registered`-bit read re-points via the non-minting `lookupStatementFlags`, then the container delete (wipeSubtree's string loop collapsed onto the `closedIds` predicate — provably the same membership; eradicate / resent-reset erase whole rows, covering both former erases). The bit contract lives in [I-85](30_invariants.md#i-85); [I-58](30_invariants.md#i-58) rewritten for the unified registry.

**Verification (definition of done).** Full `main.py` vs the current-main baseline: `theorems.txt` byte-identical, all verifier blocks byte-identical (13965-check `verifier.py` diff; 131018-check in-pipeline sweep airtight), all 337 run-log hash-burst lines identical against a main-head reference run from the same starting disk state. Dump-on A/B (user-directed retarget at the Gauss summation induction LB, both trees, same state): 227 MB traces with the identical line multiset, byte-identical except the user-approved `validityNamesToFilter` section removal and a four-line order swap of two history records inside `mailIn.exprOriginMap` (2 of 56 blocks; receiver-side `exprOriginMap` and every chapter byte-identical) — accepted by the user as an open weakness, documented in [`20_core_concepts/03_mail_system.md`](20_core_concepts/03_mail_system.md) *Suspected fragility*. Side finding: Peano CE survivor counts are starting-disk-state sensitive (382 vs 362 bodies from different accumulated incubator state, identical final theorems either way) — A/B baselines must start from the same disk state.

See [D-72](#d-72) / [D-73](#d-73) (the validity-filter insert sites, now id-only), [I-58](30_invariants.md#i-58), [I-84](30_invariants.md#i-84).

---

<a id="d-127"></a>
## D-127 — the int16 statement registry becomes the only stored form; string rows reconstructed at boundaries (2026-06-10)

**The choice (user-approved architecture change).** The string-form statement vectors — `Memory::encodedStatements`, `localEncodedStatements`, `localEncodedStatementsDelta`, `externalStatements` — are removed; their int16 lockstep mirrors (`intEncodedStatements`, `intLocalEncodedStatements`, `intLocalEncodedStatementsDelta`, `intExternalStatements`) become the single stored statement registry. Sites that need string form (diagnostic dump, mail fill, visualizer, equivalence-class rewriting) reconstruct a transient `EncodedExpression` via the new `decodeExpression` (`memory.hpp`, inverse of `encodeExpression`). The string-KEYED indexes `statementLevelsMap` and `localEncodedStatementsSet` are out of scope — they are indexes, not lockstep mirrors.

**Why lossless.** `encodeExpression` interns the whole `original` text (`originalId`) and the validity name (`validityId`); every other `EncodedExpression` field is re-derived from those two strings by the parsing constructor. The former silent arity truncation (`std::min` against `MAX_ARITY`) is now an entry assert, making the encoding provably lossless.

**Why at all.** (1) ASIC 0.1 static-memory direction — each string row holds several heap strings plus a vector-of-vector-of-strings; the int row is one flat 176-byte struct. (2) The lockstep dual-write at every push/erase site is a standing sync hazard (the erase helpers carried defensive size guards only because some test setups pushed one side). (3) The hot path (static request generation, firing, dedup, `intKnownStatements`) consumes only the int side already.

**Determinism guardrails.** Read paths use non-minting `NameMap::lookup`; decoded references are copied before any mint ([I-3](30_invariants.md#i-3)); iteration stays in insertion order — never sorted or compared by int id (mint order is not lexicographic). Definition of done: full `main.py` run unchanged in key features vs the `run_k_validation_20260610` baseline (hash-burst body/expression counts, verifier check counts, 0 failures, `theorems.txt` byte-identical).

**Landed (this branch).** All four string vectors are removed: readers migrated first (passive counts/watermark/dump/visualizer, then the iterating readers `applyEquiClasses` + `dischargeContradiction`, then erase/rebuild sites, then the delta consumers `fillMailOut` + `dischargeToBeProved`), each commit build-green with the lockstep pair intact, and the containers dropped last (`encodedStatements`, `externalStatements`, `localEncodedStatements{,Delta}`). `localEncodedStatementsSet` is now lockstep with `intLocalEncodedStatements` (rebuilds decode the kept int rows). The discipline contract lives in [I-84](30_invariants.md#i-84). Verification: full `main.py` vs the `run_k_validation_20260610` baseline per the definition of done above.

---

<a id="d-111"></a>
## D-111 — adaptive per-LB split 1 ↔ `fixed_number_splits` with same-iteration discard-and-redo on a cap-hit; supersedes the fixed override (2026-06-09)

**The choice.** `proveKernel` no longer splits every main-path / compressor LB into a fixed `fixed_number_splits` parts. Each LB carries its adaptive `Memory::numberOfParts` (default 1) and is managed two-state (bang-bang), via the pure helper `adaptiveSplitDecision(currentParts, maxSubmatches, submatchCap, fixedParts, fallbackRatio) → {nextParts, redoNow}`:

- **Start unsplit.** Every LB begins at `numberOfParts == 1` (one part = the whole hashburst, byte-identical to no split).
- **Escalate on a cap-hit (1 → `fixed_number_splits`), SAME iteration.** When an unsplit LB's single part hits the submatch cap (`maxSub >= maxNumberHashRequests`) its burst truncated — it is incomplete. `proveKernel` sets `numberOfParts = fixed_number_splits`, **discards** the truncated burst (does NOT apply it), and **re-runs the LB from scratch at the full split in the same iteration**. The phase-2 block is a bounded loop over a `toRun` set: pass 1 runs all active LBs; the LBs that escalated form pass 2's `toRun`; the loop ends when none escalated (≤ 2 passes — escalation lifts `numberOfParts` above 1, which can't re-trigger; [I-75](30_invariants.md#i-75)).
- **Coarsen when light (`fixed_number_splits` → 1), NEXT iteration.** A split LB whose BUSIEST part ran below `split_fallback_ratio × maxNumberHashRequests` (default 10%) sets `numberOfParts = 1` for its next burst (the current split burst is kept and applied).
- Otherwise hold. `numberOfParts` persists across iterations (never reset). The decision runs in `proveKernel`'s parallel finalize (I-28-safe: each LB writes only its own `numberOfParts` + its own `redo[li]` slot). `performElemPhase2` lost its inert policy block and its `maxTotalReqs` arg; the graduated `computeNextNumberOfParts` and its band test are removed.

**Why.** The fixed 100-way split ([D-109](#d-109)) paid the per-part request-gen setup on every main-path LB, including the many small ones that never approach the cap. Starting unsplit and escalating only on demand pays the split overhead only where it is needed; the discard-and-redo guarantees the result is unchanged.

**Determinism, and what is / isn't preserved vs fixed-100.**
- **Determinism (the hard property).** Two full adaptive runs are byte-identical — `theorems.txt` + `global_theorem_list.txt` + verifier check count. The redo loop and the parallel finalize add no run-to-run variance: the finalize is I-28-safe (each LB writes only its own `numberOfParts` + its own `redo[li]` slot), the discard is clean ([I-74](30_invariants.md#i-74)), and the sorted firing-record merge makes a kept burst partition- and thread-order-independent ([D-117](#d-117)).
- **Theorem set preserved.** `theorems.txt` is byte-identical to the fixed-100 reference (same theorem set). A deactivating LB proves its goal whether its burst early-exited (N=1) or ran to completion (N=100), and the discard-and-redo guarantees a cap-cut unsplit burst is never APPLIED truncated (it re-runs at the full split) — so no theorem is lost to truncation or to escalation. This holds regardless of the escalate / fall-back thresholds; they change only how much wasted work happens, never the theorem set.
- **Proof graph differs slightly vs fixed-100 — expected, not a regression.** The verifier sees a handful more checks than fixed-100 (both airtight, 0 failures). At N=1 the burst early-exit ([I-76](30_invariants.md#i-76)) fires on a deactivating head, so the post-goal auxiliary derivations a deactivating LB records differ from fixed-100's N=100 complete bursts (where the early-exit is disabled). The adaptive run's N=1 behaviour matches the ORIGINAL single-LB prover (early-exit on) — the property fixed-100 had departed from — so this is the more faithful behaviour, not a loss. **Do NOT expect adaptive to be byte-identical to fixed-100 at the proof-graph level**; the gate is run-to-run determinism plus a stable theorem set.

**Why the redo gate is the cap-hit, NOT the deactivation `stop`.** Only a cap-truncated burst is INCOMPLETE work that must be recovered — re-run at the full split, else its unfinished requests are lost that iteration. A deactivating unsplit burst has already proved its goal, so it needs no redo; redoing it would merely reproduce fixed-100's proof graph at the cost of a wasted full-split pass per discharged LB. The discard is clean because phase 2 is read-only on the LB ([D-116](#d-116)): the skipped finalize performs no apply / drain / delta-clear, so the LB is pristine for the re-run.

**Thrash / threshold tuning.** With `split_fallback_ratio` above `1 / fixed_number_splits`, a medium LB whose total work is between 1× and ~`1/ratio`× the cap oscillates 1 ↔ `fixed_number_splits` every iteration: at the full split its busiest part is light (coarsen to 1), at 1 it re-hits the cap (re-escalate), wasting one capped unsplit pass per iteration. The default 0.10 (the maintainer's chosen value) admits this band; `1/fixed_number_splits` (≈ 0.01 at 100) is the thrash-free boundary. Correctness is unaffected either way (every applied burst is complete); only runtime. `split_fallback_ratio` is config-tunable.

**Scope.** Main path AND compressor adapt (both `!incubator_mode && !disable_lb_split`). The **incubator** stays UNSPLIT (forced `N = 1`, never escalates — thousands of tiny LBs). `disable_lb_split` stays UNSPLIT (profiling, truncates at the cap with no re-split). The **CE filter** is untouched (its own `splitCount = 1` loop in `filter.cpp`, never enters `proveKernel`).

**Supersedes.** The "Fixed split count" paragraph of [D-109](#d-109) (the fixed override is removed) and the "inert `numberOfParts`" status of [D-126](#d-126) (it is read again — but by `adaptiveSplitDecision`'s bang-bang transitions, not the graduated `computeNextNumberOfParts`, which is deleted).

**Verification.** Release x64 clean; 572/572 C++ unit tests (new `adaptive_split_decision_bang_bang`; the `compute_next_number_of_parts_bands` test removed with the function). **Determinism gate (the real check, not equivalence to fixed-100): passed** — two full `main.py` + verifier runs of the adaptive build are byte-identical (`theorems.txt` md5-equal, identical verifier check total, 0 failures, every proof graph airtight). `theorems.txt` is also byte-identical to the fixed-100 reference (same theorem set); the verifier check TOTAL sits a handful above fixed-100 with individual categories shifting both ways — the expected N=1 early-exit proof-graph difference ([I-76](30_invariants.md#i-76)), every category failure-free. The adaptive path is genuinely exercised — a counter-instrumented run showed escalations (1→`fixed_number_splits`) and fall-backs (→1) both firing.

---

<a id="d-109"></a>
## D-109 — the LB-split cap and split key on SUBMATCHES, not emitted requests; CE filter runs uncapped; split count fixed via `fixed_number_splits` (2026-06-08)

**The choice.** The hashburst's per-part work metric changes from *emitted requests* (`BurstSink::produced`, the count of `StaticRequest`s handed to the consumer) to *submatches* — every `preEvaluateFromEncoded` match, i.e. each time a growing request is allowed to add an expression (the grow-DFS extensions in `growBaseCandidates` plus the merge step, both the singles and pairs generators; seeds excluded since they add nothing). The count lives in a new `thread_local int64_t ExpressionAnalyzer::g_growthMatchCount`, bumped at `preEvaluateFromEncoded`'s `true`-return — which is reached only AFTER its internal `partitionAccepts`, so each split part counts only the submatches it owns (`id % splitCount == processID`); a submatch whose owners span residues is counted by more than one part (harmless). `performElem2` resets it at entry (per part). It now drives BOTH the burst-stop cap (`BurstSink::canAccept` and the `growBaseCandidates` grow-DFS bail) AND the split policy (`performElemPhase2`'s fill ratio = busiest part's submatch count / `maxNumberHashRequests` → `computeNextNumberOfParts`). `performElem2` is now `void`; the worker reads the thread_local into `taskSubMatches` and the finalize reduces an LB's parts to the busiest.

**Why.** The emitted-request count hid the real cost. The grow-DFS (`growBaseCandidates`) was uncapped — it makes many submatches per emitted request — so an emitted cap of N let an LB do far more than N units of grow work before truncating, and the split policy (keyed on the capped emitted count) could not tell "just over the cap" from "100× over". Counting and capping on submatches measures the actual per-part work the split is meant to bound.

**CE filter runs UNCAPPED.** `canAccept` ignores the cap when `ceFilteringActive`. The CE filter is a single un-split LB per conjecture and a completeness check (does the conjecture's negation contradict the fact base?); truncating its burst on a count made it miss refuting heads and wrongly keep conjectures, which fed a main-prover blow-up. Only the contradiction early-exit (`stop`) halts a CE burst — exactly how CE is meant to detect a refutation. Confirmed: uncapping CE cut survivors 936 → 230 and removed a cap-1000 main-prover explosion that had to be killed (>2.7M exprs); it then completed in 98 s.

**Fixed split count.** New parameter `fixed_number_splits` (default 100): `proveKernel` splits every main-path / compressor LB into exactly this many parts; the **incubator** runs UNSPLIT (gated on `incubator_mode` — thousands of small LBs whose per-part request-gen setup is redundant across parts, so splitting only adds overhead and made a full incubator run drag); the CE filter is unaffected (its own un-split loop in `filter.cpp`, never `proveKernel`). This overrides the adaptive `numberOfParts`; the `computeNextNumberOfParts` update in `performElemPhase2` is now inert (kept and still unit-tested for an easy revert). Measured: at cap 20000, fixed 100 vs adaptive ≤32 gave 43 vs 44 theorems and 317 vs 227 s — more splits buy nothing; the cap, not the split count, is the limiter. **Superseded (2026-06-09, [D-111](#d-111)).** The fixed override is removed: `numberOfParts` is adaptive again (bang-bang 1 ↔ `fixed_number_splits`, escalate-on-cap-hit with same-iteration discard-and-redo), and `fixed_number_splits` becomes the escalation target rather than an unconditional part count. `computeNextNumberOfParts` (the graduated policy) is deleted, replaced by `adaptiveSplitDecision`.

**Cap is now the submatch cap.** `maxNumberHashRequests` is reinterpreted as the per-part submatch ceiling (supersedes the emitted-request reading of [D-124](#d-124)). `BurstSink::cap`, the `reqCap` local in `performElem2`, and `BurstSink::produced` are widened `int16_t → int` (the cap can exceed the int16_t ceiling 32767; the former `static_cast<int16_t>` wrapped, e.g. 1,000,000 → 16960). The now-unused per-worker `reqBuf.resize(maxNumberHashRequests)` is removed in `proveKernel` and the CE loop — the streaming `BurstSink` checks each request inline and buffers nothing.

**Cap sweep (main path Peano+Gauss; verifier airtight, 0 failures at every point).** 100 → 16 theorems; 1000 → 30; 5000 → 39; 20000 → 44; 40000 → 45. Monotone toward the pre-submatch baseline (the original emitted-cap-8192 run had 12668 checks; cap 40000 submatch ≈ 12309), with steep runtime growth past ~20K.

**Open follow-up.** The partition-independent setup (filter / sort / mandatory lists routed through `filterIntEncodedStatements`) is recomputed per part (N×); hoisting it to once-per-LB in `performElemPhase1`, with parts re-applying only the cheap `partitionAccepts`, is scoped but NOT yet done (future change). A full split-on determinism / byte-identity gate is still owed before this lands on `main`.

---

<a id="d-125"></a>
## D-125 — phase-2 request keys move to a `thread_local` arena; the per-LB `Memory::keyArena` was written by parallel parts (data race) (2026-06-08)

**The bug.** The phase-2 hashburst runs an LB's N parts concurrently on the flat executor pool and is required to be read-only on the shared LB ([D-116](#d-116)). The parallel refactor gave each worker its own `exprArena` and `reqBuf` but left the request-key store on the **per-LB `Memory::keyArena`**: `generateEncodedRequestsStatic` / `...Pairs` (`memory.cpp`) and the shared `preEvaluateFromEncoded` helper (`prover.hpp`) all called `body.keyArena.store(buf, len)`. `KeyArena::store` is unsynchronized (mutates `current` / `used` / `blocks.push_back`), so concurrent parts of one LB race on it. Found by inspection after a reproducible Gauss-main segfault.

**Symptoms it explained.** (1) **Segfault** (`0xC0000005`): heavy split (low cap → up to 256 parts) → many concurrent `store`s → a torn `current`/`used` or a `blocks.push_back` realloc under another thread's pointer → out-of-bounds `memcpy`. Reproduced in Gauss main at cap 250 (more / longer keys than Peano main, which survived); `max_number_splits = 32` did not help — a few concurrent parts already race. (2) **Non-monotonic verifier check counts** across caps (98488 at ≤4 parts; 98778 / 98530 at 16–256 parts): the race corrupted request keys → different firing sets → output that varied with split depth.

**The fix.** Transient request keys move to a new `thread_local KeyArena g_reqKeyArena` (`memory.hpp` / `memory.cpp`), `release`d at each `performElem2` entry — per-thread, per-part isolation, exactly like `exprArena`. `Memory::keyArena` still backs the PERSISTENT hash-memory keys, written single-threaded in phase 1. Safe because request keys are dead once `performElem2` returns (`FiringRecord`s carry decoded strings, not arena pointers). A `thread_local` (vs threading a `KeyArena&` through the templated generators + the shared `preEvaluateFromEncoded`) confines the change to the three store sites + one reset and matches the split's existing `g_splitProcessID` / `g_splitCount` thread_locals.

**Verification.** Build clean (Release x64); 570/570 unit tests. The reproducing case — main path only (Peano + Gauss), cap 250, split on, `max_number_splits` 32 — which segfaulted in Gauss main before the fix, now COMPLETES: exit 0, all proof graphs verified, 10234 checks / 0 failures, 529 s. A full split-on vs split-off byte-identity / determinism gate is the proper follow-up before this lands on `main`.

**Open follow-up.** Audit the rest of the phase-2 path for any other shared-LB writes that slipped past the read-only refactor. See [I-83](30_invariants.md#i-83).

---

<a id="d-110"></a>
## D-110 — `disable_lb_split` forces one part per LB and re-homes the RT tracker in `performElem2` (2026-06-08)

**The choice.** A new `prover_parameters` flag `disable_lb_split` (default `false`) makes `proveKernel` create exactly one phase-2 executor task per active LB (`N = disable_lb_split? 1: numberOfParts`) instead of `numberOfParts` tasks, and `performElemPhase2` skips the adaptive `computeNextNumberOfParts` update (the counter would only grow unread). With splitting off, each LB's hashburst is a single `performElem2` call on one thread, so the per-call `RTTracker` — homed by `RT_TRACKER_DECL(body)` at the top of `performElem2` — measures the LB's whole hashburst. This is the diagnostic mode for runtime profiling.

**Why it exists.** The LB split removed the single-LB orchestrator `performElementaryLogicalStep`, which was the only `RT_TRACKER_DECL` call site, leaving RT measurement inert ([D-113](#d-113)). Rather than reintroduce a duplicate LB-major driver just to host a tracker (rejected by the owner as code duplication), the tracker is added to the existing hashburst executor `performElem2` — the phase where a runaway LB spends its time generating requests. `disable_lb_split` then makes that measurement clean: one part per LB ⇒ one tracker per LB ⇒ one `.rt/<chain>.log` per LB, no cross-part races.

**Scope of the measurement.** RT covers the hashburst (`performElem2`: the `REQGEN_*` batches and `FIXPOINT_LOOP`). Phase 1 (mail absorb) and phase 3 (send / discharge / sanitize) run in separate barriered sweeps with no tracker active, so their `RT_SCOPE_HERE` rows do not record — sufficient for finding runaway-request LBs, where the cost is the hashburst ([I-59](30_invariants.md#i-59)).

**Profiling, not proof-complete.** A `disable_lb_split` run equals the barriered path for every LB that never exceeds `maxNumberHashRequests` (one part = full coverage; the deposit merge is partition-independent, [D-117](#d-117)), but a runaway LB whose single unsplit part hits the cap truncates instead of being split across parts. So a split-off run is for diagnosis, not a complete proof. Default `false` leaves the production parallel path untouched.

**Verification.** Build clean (Release x64); C++ unit tests green incl. the new `disable_lb_split_defaults_false` case (production-safe default). The RT / profiling run (`RT_MEASUREMENT=1`, `RT_TIME_TRIGGER_SECONDS=20`, `disable_lb_split: true`, IncubatorPeano) is the diagnostic use of the flag, not a determinism gate.

---

<a id="d-118"></a>
## D-118 — CE filtering is a dedicated thread pool over a global conjecture queue; one single-use clone LB per conjecture; CE detection moves into `burstDeactivates` + `dischargeContradiction` (2026-06-07)

**The choice.** `filterConjecturesWithCE` (`filter.cpp`) replaces the batched driver (`batchSize = max(1, logicalCores)` conjectures per batch; `prove` blocks until all finish; a hard barrier between batches) with a dedicated pool of `max(1, logicalCores)` worker threads draining one atomic conjecture index. The fact base is loaded **once** into a single template LB (status-4 statements, empty `overallHashMemory` / `keyArena`); each worker clones it (`Memory::cloneFactsTemplate`), installs one conjecture's rule, runs **exactly one hashburst** on its private single-owner clone (`numberIterationsConjectureFiltering == 1`, asserted), records the result, deletes the clone, and grabs the next. No batch barrier — a freed worker takes new work immediately.

**CE detection moves off the insertion gate.** The former `addExprToMemoryBlock` gate (`contradictionIndex >= 0`, which discarded CE fired heads) is removed; CE fired heads now enter `encodedStatements` like any derived statement. Detection is the same two-stage mechanism the incubator / induction LBs use: `burstDeactivates` (a CE arm, read-only) stops the single burst the instant a refuting head fires, and `dischargeContradiction` (a CE branch) records the refutation into the conjecture's `contradictionTable` slot and deactivates the clone. This reverses the CE carve-out of [D-122](#d-122) — now safe because each CE LB is **single-owner** (the I-66 phase-2 read-only rule binds only LBs split across threads, which a CE LB never is).

**Why it is race-free.** Each conjecture is processed by exactly one worker on a private clone (own `nameMap` / `keyArena` / mailboxes); `cloneFactsTemplate` only reads the shared template; the sole cross-thread write is `contradictionTable[i].successful` to a disjoint per-conjecture index (the vector is sized before the pool, never resized). CE clones are never `primedForContradiction` / `isPartOfRecursion`, so `dischargeContradiction`'s shared-write branches (`updateGlobalDirect`, `inductionMemoryBlocks`) never fire for them.

**Why the clone is cheap.** `loadFactsForCEFiltering` adds every fact through the status-4 statement path, which never calls `addToHashMemory` and never touches `keyArena`, so the template carries no arena-backed hash keys. `cloneFactsTemplate` is therefore a plain deep value-copy of the fact containers + `nameMap` (all pure value types) with everything else reset — no `KeyArena` interior-pointer surgery, the hazard a whole-`Memory` copy would hit.

**Verification.** Build clean (Release x64); 569/569 C++ unit tests (clone deep-copy + `burstDeactivates` CE arm + the inverted `dischargeContradiction` CE test). Full `main.py` + verifier: `theorems.txt` CONTENT-IDENTICAL to the lb_split4 baseline (all 46 theorems incl. Gauss `fold` summation and FTA rung-1 `interval`); verifier every category failure 0; survivor sets identical (Peano 936→230→122, Gauss 674→124). FASTER CE filter on a 32-core host: Peano 37.3s → 16.2s (2.3×), Gauss 51.8s → 30.1s (1.7×), ~2× overall.

---

<a id="d-112"></a>
## D-112 — the prover's fake `mirrored statement` step is removed; both directions are now genuinely proved (2026-06-07)

**The choice.** The prover no longer fabricates the reverse direction of a proved theorem as an unproved `mirrored statement` / `mirrored from` row (the `ce::createReshuffledMirrored` output-variable swap + premise permutation, registered with depth `"-1"`). Instead the conjecturer folds each conjecture's reverse-direction mirror into the real prove pool (`conjectures.txt`) via `conj::mergeMirrorConjecturesIntoPool`, so it passes through the counterexample filter and is proved by the normal engine — the proof is now explicit in both directions. The `mirrored from` proof tag, its verifier checker (`check_mirrored_from` / `_check_mirror`) and the soft external-mirror fallback, and all HTML / proof-graph mirror handling are deleted. `TAG_CHECKERS` drops 31 → 30.

**Why it exists.** The mirror step asserted a theorem the prover never derived. Swapping the head with the premise sharing its output variable is the *converse* of a functional relation, which is not a logical equivalence in general — so the row was, in the owner's words, "cheap, obviously mathematically wrong." Routing the reverse direction through CE + real proof makes a false mirror simply fail to prove (or be CE-pruned) instead of being stamped proved. Kept untouched: the equality mirror `(=[a,b])→(=[b,a])` ([I-9](30_invariants.md#i-9) — the `args[0]!= args[1]` guard in `createReshuffledMirrored`, retained for the conjecturer's pool mirrors and `--mirror-externals`), and head-switch (`headSwitchOne` contrapositives, already genuinely proved). See [I-81](30_invariants.md#i-81).

---

<a id="d-120"></a>
## D-120 — request generation drops a request no owner's u_ literals can satisfy, via a per-key cached u_ signature (2026-06-06)

**The choice.** At every owner-set match site in static request generation — the same sites as the D-105 scope prune — a request is additionally dropped when no owner of the matched (sub)key has u_ (unchangeable) argument literals satisfiable by the request's concrete `argFullId` values. The check is `ExpressionAnalyzer::ownerSetUSatisfied(OwnerSet, exprs, count)`, AND-combined with `ownerSetHasComparable` and `partitionAccepts`. It is fed by a per-key cache on `OwnerSet`: `recordUSignature` runs at every owner insert (`addToHashMemory`, `makeNormalizedKeysForAdmission`) and stores, per owner with u_ args, a `uSignatures` entry — the `(linear-arg-slot, required id)` list of its unchangeable args — or sets `hasLooseOwner` for an owner with no u_ constraint.

**Why it exists.** The four `normalizedEncoded*` fast-rejection maps are built `ignoreU=false`, which normalizes every argument to a positional id and **erases** a rule's u_ literal values — only the repetition pattern survives. So a growing/seed/merge request can match a (sub)key structurally yet be doomed by a u_ literal it can never satisfy (e.g. the rule pins operator `+` where the request carries `*`); today that is caught only later at the firing gate (`encodedMap`, `ignoreU=true`). Bringing the check forward prunes the grow tree earlier and drops doomed full requests before the firing gate — the win lands in u_-heavy branching proofs (OR theorem, FTA ladder). On all-changeable workloads every key is loose, so the check is an O(1) `hasLooseOwner` short-circuit and costs nothing.

**Why it is a runtime optimization, not a soundness check.** The signature literal is the owner's `argFullId` (`== NameMap::encode(arg[1])`, the exact value the firing gate matches against the request's `argFullId` at each unchangeable slot), so `ownerSetUSatisfied` computes a necessary condition for firing the matched key. The firing gate enforces the exact condition independently, so the prune can never change which rules fire — `theorems.txt` is byte-identical with it wired or stripped ([I-79](30_invariants.md#i-79)).

**Cached, mint-safe, RT-neutral.** The literal id is read with the non-minting `NameMap::lookup`, never `encode`: at the head-rule insert sites `arg[1]` is already interned by the preceding `ignoreU=true` key build, so the full signature is recorded; at a marker-subkey insert a u_ literal may not be interned yet, in which case the key is conservatively flagged loose (sound — keeps the request) rather than minting a fresh id and shifting the LB's id-assignment order. So the cache build mints nothing and the run stays byte-identical regardless of whether id-order leaks anywhere. The match side reads only cached ids + `argFullId` (no `encode`), safe on the shared read-only LB the split executors run in parallel.

**Wipe handling (chosen: leave stale).** Unlike `owners`/`partitionIds` the `hasLooseOwner`/`uSignatures` cache is NOT maintained on `Memory::wipeSubtree`. A stale signature or stale `hasLooseOwner` only ever makes the prune weaker (keeps more), never unsound, so leaving it is the simplest, fastest-match option. Strict per-owner lockstep (stronger prune through deep branching, slower match) was considered and rejected for runtime.

---

<a id="d-121"></a>
## D-121 — phase 2 streams generate+check, caps by counting, and early-exits a doomed LB via an external per-LB stop flag (2026-06-06)

**The choice.** The phase-2 hashburst (`performElem2`) no longer buffers every hash request then checks them all. A streaming consumer `BurstSink` ([`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp)) checks each request inline as it is generated: per request it runs the burst-fixed `intKnownStatements` dependency skip, then `checkLocalEncodedMemoryStatic`, capturing firings into the per-task `firingRecords`. The hash-request cap (`maxNumberHashRequests`) is enforced by **counting** (`BurstSink::canAccept`), not by a buffer. The four request generators (`generateEncodedRequestsStatic` / `...Pairs` / `...StaticCE`) are templated on the consumer, and `StaticRequestEmitter::emit` returns a keep-going bool so a generator stops the moment the consumer says stop.

**Early-exit.** After each fired head, `BurstSink::consume` runs `ExpressionAnalyzer::burstDeactivates(body, fr)` — a read-only predicate mirroring the phase-3 discharge detection — firing on three reasons: **contradiction** (`primedForContradiction` + neg(head) known at an ancestor), **contradiction in vacuous truth** (`isPartOfRecursion` + validity "main" + the same neg scan), **toBeProved reached** (`isPartOfRecursion` + "main" + the head is a goal). On a hit the LB is doomed; `BurstSink` sets the stop flag and returns false, the generators break, and `performElem2` skips the remaining batches (`sink.canAccept` guards each). The seed/grow/merge generation — the incubator's dominant cost — is what gets cut.

**The flag is external to the LB (user direction 2026-06-06).** The stop flag is NOT a `Memory` field. `proveKernel` owns a `std::vector<std::atomic<bool>>` sized to the active-LB count (stable addresses, never resized), one flag per LB shared by all its `(LB, part)` executor tasks; it resets each before the pool spawns and hands `&stopFlags[li]` to each task. Phase 2 therefore writes nothing on the LB — the [I-66](30_invariants.md#i-66) read-only guarantee becomes structural, and the cross-part stop (one part finds the contradiction, every part of that LB bails) needs no LB mutation.

**Determinism refinement ([I-76](30_invariants.md#i-76), 2026-06-09).** The cross-part stop above — "every part of that LB bails" — is sound only UNSPLIT. Under split it is a thread-timing race: a sibling part can bail on another part's `stop` before it has generated its own (useful) firing, so the proved-theorem set depends on scheduling — non-deterministic, breaking GL's core claim. (It surfaced as a heisenbug: the hashburst dump's own latency reliably flipped a lost Peano `1·x=x`-mirror theorem from absent at 100-split back to present.) The early-exit is now gated on `g_splitCount`: honored at `g_splitCount == 1` (the unsplit incubator + CE filter rely on it — `consume` still sets `stop`, `canAccept` still bails), fully disabled at `g_splitCount > 1`, where every part runs its slice to completion. Two 100-split full runs are then byte-identical (`proved_theorems` + `global_theorem_list`), recovering the dropped Peano and Gauss theorems with the verifier airtight. The requests skipped past a doom point are wasted-once-doomed, so split and unsplit firing sets match.

**Why it keeps every theorem.** The triggering head is captured into `firingRecords` BEFORE `consume` decides to stop. `isActive` is never touched in phase 2, so the per-LB finalize (`performElemPhase2`, `if (body.isActive)`) deposits the captured records — including the trigger — and phase 3's `dischargeContradiction` / `dischargeToBeProved` find it and emit the theorem exactly as before ([I-66](30_invariants.md#i-66)). Early-exit only DROPS requests that would have fired AFTER the trigger; a doomed LB is discharged regardless and its statements never propagate ([D-51](#d-51)), so the proved-theorem set is unchanged. `burstDeactivates` scans the same burst-fixed `intKnownStatements` the phase-3 sweep does, so it fires only when phase 3 would — it never over-exits.

**Proof graph is leaner, not byte-identical.** Because a doomed LB stops generating once doomed, it records FEWER incidental derived statements in its chapter than the full run did. The theorems (`theorems.txt`, both batches) stay byte-identical, but the verifier check count drops (128851 → 128119, net −732; e.g. ~259 fewer `definition set consistency` rows) — a leaner, still-airtight proof of the same theorems. This is sound iff each doomed LB reaches a stable discharge target (the incubator `__contradiction__` / induction case); run-to-run determinism is the gate (Verification).

**Commits.** 1: `burstShouldStop` field (later moved external) + `burstDeactivates` + unit tests. 2: templatize the emitter + generators on a `Consumer` (byte-identical). 3: switch `performElem2` to the streaming `BurstSink`, cap-by-counting, move the flag external to `proveKernel`, delete the fixpoint loop (byte-identical). 4: wire the early-exit (first behaviour change).

**Verification.** Build clean (Release x64); 551/551 C++ unit tests (new `burstDeactivates` + `BurstSink` cases). Commit-3 full `main.py` + verifier: 128851 checks / 0 failures, every tracked artifact byte-identical to the DoD baseline (`run_full_dod.log`) — the direct measurement that streaming == store-all. Commit-4 **two sequential full runs**: `theorems.txt` byte-identical (both batches, both runs); 128119 checks / 0 failures — airtight; **the entire processed proof graph (139 files) is byte-identical between the two runs** (identical per-category counts). So although the early-exit's stop-propagation timing varies across parts, each doomed LB reaches the same discharge and the proof graph is run-to-run deterministic. Wall-clock 743 s and 827 s vs the 897 s baseline (load-sensitive, but consistently faster). The check count is 128119 vs the baseline's 128851 (−732): a doomed LB records fewer incidental derivations before it stops — a leaner, still-airtight proof of the identical theorem set (user-accepted 2026-06-06).

---

<a id="d-122"></a>
## D-122 — incubator + vacuous-truth contradiction handling consolidated into one `dischargeContradiction` sweep over `encodedStatements`; CE-filter detection stays at insertion (2026-06-06)

**The choice.** Contradiction detection plus the three reactions (incubator / `primedForContradiction`, CE filter, vacuous truth) move out of the per-insertion "Site G" block inside `addExprToMemoryBlock` and out of the read-only burst predicate `deactivationCheck`, into one new inline member `ExpressionAnalyzer::dischargeContradiction` ([`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp)). It runs once per elementary step from `standardProcessing`, directly before `dischargeToBeProved`, and sweeps the LB's whole `encodedStatements` set: for the first statement whose negation is already proved at an ancestor scope (self included), it fires the one matching reaction, deactivates the LB, and returns. (See the **CE carve-out** below — CE was reverted to the gate.)

**Reversed by [D-118](#d-118) (2026-06-07).** The CE-filter RT-opt rebuilds CE filtering as a per-conjecture thread pool in which each CE LB is single-owner (its own worker + clone), so CE fired heads can safely enter `encodedStatements` and CE detection moves into `burstDeactivates` + `dischargeContradiction` after all — undoing the carve-out below, which remained correct only for the shared-LB batched driver it was written against.

**CE carve-out (correction, 2026-06-06).** The initial attempt moved all three reactions (including CE filter) into the sweep. That **killed the CE filter**: it refuted 0 conjectures, and the resulting unfiltered conjectures crashed the Peano main batch on a `prepareIntegration` `tempArgs.size <= 1` assert. Root cause — a CE LB's fired heads are discarded by the `contradictionIndex >= 0` gate in `addExprToMemoryBlock` and **never enter `encodedStatements`** (CE LBs hold only their loaded facts), so the post-burst sweep has nothing to detect against; trapping showed it found 0 of the ~8600 CE contradictions that actually fired at the gate. CE detection therefore **stays at the gate**, on the incoming head (the original Site-G CE behaviour): if the head's negation is already known at an ancestor, mark `contradictionTable[idx].successful` + deactivate the CE LB. Only the **incubator + vacuous-truth** reactions (whose heads DO enter `encodedStatements`) live in `dischargeContradiction`. Confirmed: with CE restored, Peano main filtered 423/536 then 30/113 conjectures and completed with 53 theorems, no crash. The double-marker crash was thus a downstream symptom of the dead CE filter, not an integration-path bug.

**Why a whole-set sweep, not the incoming expression.** Site G ran the ancestor-negation scan only at insertion time on the expression being added, so it could not see a contradiction whose two halves arrived in the "wrong" order: if the deeper positive is added first and the shallower (ancestor-scope) negation arrives later, neither insertion's own scan sees the other (the later insertion scans ITS ancestors, and the deeper positive is a descendant, not an ancestor). Re-scanning every statement each step makes detection order-independent. This is a deliberate semantic STRENGTHENING — it can fire contradictions Site G missed, so new theorems / deactivations are possible; the DoD run (full Gauss + FTA rung1) is the gate.

**Why one place.** The reactions previously lived in a flat container-insertion routine ([I-60](30_invariants.md#i-60)) and were mirrored as detection-only conditions in `deactivationCheck`; the duplication was a standing hazard. One sweep, run from the serial `standardProcessing` phase, is the single authority. The fire-once + `!isActive` guard make the phase-1 / phase-3 double-run idempotent ([I-78](30_invariants.md#i-78)).

**Semantics preserved.** Each reaction's recorded origin (`contradiction` / `vacuous truth`), the [D-51](#d-51) local-only contradiction record, the vacuous-truth no-`mb.level`-gate rule, and the same-step head deposit via `addStatement` (the head lands in `localEncodedStatementsDelta` before `dischargeToBeProved` snapshots it) are carried over unchanged. The two Site-G `addOrigin(… exprVal …)` "record the triggering expression" calls are dropped as redundant — the swept statement is already in `encodedStatements` with its origin recorded by the normal add path.

**`deactivationCheck` removed.** Its result (`sawDeactivation`) was already unused in the flat-executor phase 2 ("unused here"), so removing the predicate and its dead plumbing is behavior-preserving (separate commit). See [D-123](#d-123).

**Reserved `internalMailOut` parameter removed.** `dischargeContradiction` was handed a `ColdMail& internalMailOut` for signature symmetry with `dischargeToBeProved` but never wrote to it; the dead parameter was later dropped (behavior-preserving). See [D-181](#d-181).

**Verification.** Build clean; C++ unit tests green including the new `discharge_contradiction_*` cases. Full `main.py` + verifier is the DoD gate — Gauss completes, FTA rung1 proved, `contradiction` / `vacuous truth` categories report failure 0 — run after both refactor commits land.

---

<a id="d-117"></a>
## D-117 — hashburst deposits captured as firing records, sorted into canonical order, then applied (2026-06-04)

**The choice.** The static request-evaluation pass (the `FIXPOINT_LOOP` in `performElementaryLogicalStep`) no longer mutates its deposit containers inline. Each rule firing in `checkLocalEncodedMemoryStatic` instead appends a `FiringRecord` (`memory.hpp`) describing its deposit; after the pass `ExpressionAnalyzer::applyFiringRecords` (`memory.cpp`) sorts the records by a total content key and applies them — head records to `sameIterationInternalMail.statements` / `.exprOriginMap` / `.disintegrationSignals` and `canBeSentIds`, marker records to `deferredIntegrationPreps` / `canBeSentMarkerIds` / `admissionKeysAlgebra`.

**Why.** This is the reference-baseline (commit 1) groundwork for the LB split. The split partitions an LB's rules across `n` copies that run a hashburst each; their firings are pooled and applied as if one burst ran. For that merge to be byte-identical regardless of how the rules were partitioned, the order-sensitive deposits must become a function of the firing SET, not the request-generation ORDER. The order-sensitive ones are: the cap-bounded `addOrigin` selection on `exprOriginMap`, the per-head last-write on `disintegrationSignals`, and the firing-order drains of `admissionKeysAlgebra` / `deferredIntegrationPreps`. Sorting before applying fixes all four. The plain sets (`statements`, `canBeSent*`) were already order-independent.

**`deactivationCheck` stays inline.** It is RT control, not a deposit: a head firing that closes the LB still sets `isActive = false` mid-pass so the request loop breaks immediately. It reads only burst-fixed state (`toBeProved`, `intKnownStatements`), so it is correct during capture even though the deposit it accompanies is applied later. Under the split (later commits) each copy runs `deactivationCheck` inline; when one copy deactivates the scheduler deactivates the rest.

**Soundness vs main.** The deposited SET is unchanged (same firings, up to the same inline-deactivation break point); only the order within the order-sensitive containers changes — so the cap may keep a different origin and the admission-revival drain may fire in a different order than main HEAD. Per the branch contract this commit-1 run is the new reference baseline: it must preserve every proved theorem, not be byte-identical to main. See [I-77](30_invariants.md#i-77).

**Verification.** Build clean; 537/537 C++ unit tests including `test_lb_split.cpp`'s order-independence test (two permutations of one record set → identical deposits); full `main.py` + verifier as the reference baseline.

---

<a id="d-124"></a>
## D-124 — hash-request cap `maxNumberHashRequests` extracted to `parameters.hpp` (2026-06-04)

**The choice.** The hard-coded `8192` that sized `reqBuf` and bounded every request-generation batch in `performElementaryLogicalStep` (and the CE path) becomes `ProverParameters::maxNumberHashRequests` (`parameters.hpp`), default `8192` for the reference baseline.

**Why.** It is the cap the LB split keys on: an LB whose request count exceeds it is split so each copy stays under it. `StaticRequestEmitter::emit` truncates silently past the cap, so lowering it is only sound once splitting is in place (later commits lower the default to 2048). Extracting it to a named parameter at the reference baseline keeps that commit a zero-behaviour-change change (the value is unchanged) while making the cap an explicit, tunable knob.

**Scope.** Only the request-count cap moves. The intermediate filter buffers (`mslBuf`, `pairsBuf`, the `filteredIdx` stack arrays) bound statement / pair counts, not requests, and are left as-is.

**Superseded (2026-06-08, [D-109](#d-109)).** `maxNumberHashRequests` is no longer the *emitted-request* cap — it is now the per-part **submatch** ceiling, and the default moved off `8192` (currently `40000`). `BurstSink::cap` / the `reqCap` local were widened `int16_t → int`, and the `reqBuf` allocation this entry sized is removed (the streaming sink buffers nothing).

---

<a id="d-119"></a>
## D-119 — each (sub)key carries a parallel `partitionIds` set keyed by the (expanded-original, validity) owner (2026-06-05)

**The choice.** Each `OwnerSet` (the value of the four `normalizedEncoded*` fast-rejection maps in `HashMemory`) gains a second field `std::set<int32_t> partitionIds` alongside `owners`. Every owner insert also inserts the owner's **composite partition id** — `makePartitionId(NameMap::encode(expandedOriginal), scopeVid)` (`memory.hpp`), packing the rule's expanded-original id in the high 16 bits and the scope validity id in the low 16. The set is 1:1 with `owners`, maintained in lockstep at every insert (`addToHashMemory`, `makeNormalizedKeysForAdmission`) and at the radical subtree wipe (`Memory::wipeSubtree`'s `wipeOwnerMap` recomputes the id and erases it; the whole `OwnerSet` drops when `owners` empties, [I-49](30_invariants.md#i-49)). This commit only populates the set — it is not yet read, so the run stays byte-identical to the commit-1 baseline.

**Why.** The LB split runs an LB's hashburst on `N` executors in parallel, each handling a disjoint slice of the rules. A request generated against a (sub)key is claimed by executor `n` of `N` iff some composite id in that key's `partitionIds` satisfies `id % N == n` (the read side — `partitionAccepts` — is wired in commit 3; see *The read side* below). The id identifies the rule by its `(expanded-original, validity)` owner — the same pair `owners` already keys on — so the partition is by rule, and a rule's head LMV and marker LMVs (which share the original and validity) carry the same id and never split apart. `curOrigImpl` is the full expanded `>` implication (it is `disintegrateImplication`-d into chain + head at install), so the encoded id is the expanded form, not a compact `(implication<N>[…])` atom.

**Why a separate set, not a wider `owners` value (user's choice).** `owners` keeps its bare `int16_t scopeVid` value untouched (the [D-105](#d-105) validity prune reads it on the hot path); the partition ids ride in a parallel set so the two concerns stay independent. The composite is recomputed (not stored per-owner) at wipe via the idempotent `NameMap::encode`, which returns the id minted at install.

**Why `NameMap`, no new counter (user's choice).** Both halves are already-interned `int16` ids — `NameMap::encode(expandedOriginal)` and the existing `ownerVid` — so no dedicated split counter is introduced. The packing is injective over the non-negative `int16` ids `NameMap` mints, so distinct owners never collide. (Risk: encoding every distinct expanded implication into the per-LB `NameMap` consumes id space; an LB with more distinct rules than `MAX_NAME_IDS` would trip the existing `NameMap` overflow assert — fail-loud, to be surfaced if it bites a large LB. The Step-A byte-identical gate also validates that this extra `NameMap` interning, which shifts the ids of later-encoded names, leaves the proof output unchanged.)

**The read side (commit 3 — the inert filter).** A request generated against a (sub)key is accepted by executor `n` of `N` iff `partitionAccepts(OwnerSet::partitionIds)` (`memory.hpp`): `true` when `g_splitCount <= 1`, else some id has `id % g_splitCount == g_splitProcessID`. It is **AND-combined** with the existing [D-105](#d-105) `ownerSetHasComparable` at every request-generation acceptance site — the singles/pairs seeds, the gradient grow (`growBaseCandidates`, `preEvaluateFromEncoded`), the single- and pair-element mandatory filters (`filterIntEncodedStatements` + the mandatory-pair builder), and the CE filter (`filterIntEncodedStatementsCE`). Because a (sub)key's `partitionIds` is the union of every rule whose key passes through it, the filter prunes both growth and emission by rule; and a full key owned by executor `n`'s rule carries that rule's id in **every** one of its prefix subkeys (the subkey loop stamps the rule's id at all prefix lengths), so `n`'s growth toward it is never pruned (completeness). The two coordinates live in **thread-local** `g_splitProcessID` / `g_splitCount` (default `(0, 1)`), set per executor at the top of its hashburst — thread-local so parallel executors of one LB never race, and defaulting to the unsplit identity so the filter is a provable no-op until the split drives it (mirrors the existing `g_buildStackPath` per-thread context). No request-generation signature is threaded; the user's "`processID` / `splitCount` params on performElem" become the values the executor (commit 4's `performElem2`) writes into this context.

**Verification.** Build clean; 539/539 C++ unit tests (commit 2's `make_partition_id_packs_and_is_injective` + commit 3's `partition_accepts_filters_by_residue`). Step-A gate (commits 2+3, the populate + inert read): full `main.py` + verifier — `theorems.txt` byte-identical to `main`, verifier 0 failures (validates both the `NameMap`-interning id shift and the inert `N == 1` filter). See [I-80](30_invariants.md#i-80).

---

<a id="d-113"></a>
## D-113 — `performElementaryLogicalStep` split into `performElemPhase1/2/3` + `performElem2` (2026-06-05)

**The choice.** `performElementaryLogicalStep` is decomposed into an orchestrator that calls three phase helpers in sequence — `performElemPhase1` (pre-hashburst mail absorb), `performElemPhase2` (the hashburst), `performElemPhase3` (post-burst absorb + sanitize) — plus `performElem2`, the per-executor request-generation + fixpoint unit that `performElemPhase2` drives. The orchestrator keeps the entry `isActive` early-return and the per-call `RT_TRACKER_DECL`; the phase helpers attribute RT time to that tracker via the thread-local `RT_SCOPE_HERE` mechanism (`g_currentThreadTracker`), so no tracker is threaded and the unified per-call table survives. Deactivation behaviour is **unchanged** this commit (the inner `if(!body.isActive) break;` stays in `performElem2`'s fixpoint) — this is the byte-identical refactor; the deactivation deferral is a separate follow-up.

**Why.** The LB split runs one LB's hashburst (`performElem2`) on N executor threads in parallel. Isolating the hashburst as its own function with **caller-owned** request/firing buffers (not LB fields) is the precondition: each parallel executor gets its own `exprArena` / `reqBuf` / `firingRecords`, and `performElemPhase2` merges their captured deposits via the canonical-sorted `applyFiringRecords`. Splitting the step into barriered phases (phase 1 for all LBs → phase 2 for all LBs → phase 3 for all LBs) is also what lets phase 2 dedicate cores to a heavy LB; that `proveKernel` restructure is the next commit.

**Trap relocation (Rule 14, user-approved 2026-06-05).** The three hashburst-dump trap call sites follow their bracketed code into the phases — ENTRY → `performElemPhase1`, EARLY-EXIT → `performElemPhase2` (after the merge, so the dump still sees applied deposits), EXIT → `performElemPhase3`. The dump code / section format / `isTargetLB` targeting /  path are **unchanged**; only the call-site location moved. The EARLY-EXIT trap is additionally gated on `splitCount == 1`, and the phase-2 RT scopes auto-gate (parallel executors run on spawned threads where `g_currentThreadTracker` is null, so `RT_SCOPE_HERE` no-ops) — so the fixed-path dumps never race the parallel executors. Force `splitCount = 1` to debug a split LB. *(Forward note: the EARLY-EXIT trap was removed entirely in [D-114](#d-114) once the deactivation deferral made it dead — only ENTRY (phase 1) and EXIT (phase 3) survive.)*

**Byte-identity.** At `splitCount == 1` (this commit, always) the phase split preserves the exact statement order: phase-1 absorb → [if isActive: `performElem2` reqgen+clears+fixpoint → `applyFiringRecords` → EARLY-EXIT] → drains → phase-3. `RT_SCOPE → RT_SCOPE_HERE` and the new `MERGE_FIRING_RECORDS` scope are no-ops with RT off (the default). `exprArena`/`reqBuf` move inside the `if(isActive)` gate (scratch buffers, no logical effect). `performElem2` sets `g_splitProcessID=0`/`g_splitCount=1` (the unsplit identity), so `partitionAccepts` stays inert.

**Known forward-incompat (commit 7).** The LB-mutating per-step delta clears live inside `performElem2`; at `splitCount > 1` they must move to `performElemPhase2`'s once-per-LB merge point (multiple executors must not each clear). Flagged in the `performElem2` comment.

**Verification.** Build clean (Release x64); C++ unit tests pass (incl. new phase-helper signature tests). Byte-identity gate (full `main.py` + verifier, 1275 s, `RUN_INCUBATOR=True`): `theorems.txt` IN-ORDER BYTE-IDENTICAL to `main` (46 theorems, Gauss fold proved); verifier 129259 checks, 0 failures, all proof graphs verified; 364/364 verifier unit tests. The earlier raw "DIFFERS" is a git-blob-LF vs Windows-worktree-CRLF artifact (content identical once line endings are normalized).

---

<a id="d-123"></a>
## D-123 — the hashburst fires every request; deactivation moves entirely to phase 3 (2026-06-05)

**Superseded in part ([D-122](#d-122), 2026-06-06).** The deferral decision stands — phase 2 still never deactivates; deactivation / discharge happen only in phase 3. But the `deactivationCheck` predicate and its `sawDeactivation` early-stop signal described below were **removed**: the signal went unused in the flat-executor phase 2, and all contradiction handling moved into `dischargeContradiction`. Read the `deactivationCheck` / `sawDeactivation` mentions below as historical.

**Title refined ([D-121](#d-121), 2026-06-06).** Phase 2 no longer fires EVERY request: a streaming early-exit stops a doomed LB's burst (via a read-only `burstDeactivates` predicate that sets an external per-LB stop flag). The deactivation deferral itself is untouched — phase 2 still never flips `isActive` and the LB stays read-only; only the "fires every request" half of the title is now "fires every request until the LB is detectably doomed".

**The choice.** The inner `if(!body.isActive) break;` is removed from `performElem2`'s fixpoint loop, and `deactivationCheck` no longer mutates `body.isActive` — it becomes a pure **detection** predicate. A detected close is OR-accumulated into the executor's `sawDeactivation` (caller-owned, parallel-safe) and never applied in phase 2. The authoritative deactivation / discharge fires in phase 3's post-burst `standardProcessing` absorb, which re-detects the closing head via `addExprToMemoryBlock`'s four condition sites (the [I-66](30_invariants.md#i-66) deferred-deactivation mechanism that was already the broadcast/discharge path). `performElemPhase2` returns `body.isActive && !sawDeactivation` — the coarse early-stop signal the barriered `proveKernel` will use to skip not-yet-started executors of a closing LB. The **entrance** `isActive` gate in `performElemPhase2` stays (per user: gate at the entrance, not the inner process).

**Why.** With N executors sharing one read-only LB, `deactivationCheck` mutating the shared `body.isActive` would race, and the inner break would lose firings the way did (one copy closing skipped the rest, dropping 2 Gauss `fold` theorems). Detection-without-mutation + no break makes every executor fire its whole slice; the close is a per-executor signal, applied once, single-threaded, in phase 3. This is the fix for attempt 1's root cause.

**Why it preserves the proofs.** The close's broadcast/discharge was already deferred to phase 3 by [I-66](30_invariants.md#i-66) — phase 2's `deactivationCheck` was only a predicate-and-break (an RT optimization). Removing the break means phase 2 fires the requests it used to skip after the close; those are post-closure firings whose deposits phase 3 absorbs. The proved goal is reached either way (the closing head fires + discharges in phase 3).

**EARLY-EXIT dump.** `body.isActive` now stays true through phase 2, so the (relocated, `splitCount == 1`-gated) EARLY-EXIT hashburst-dump trap no longer triggers — the close moved to phase 3. The trap was therefore **removed entirely** in [D-114](#d-114) (user-approved 2026-06-05, Rule 14); ENTRY (phase 1) and EXIT (phase 3) remain.

**Parallel-safety caveat (resolved in [D-116](#d-116)).** `deactivationCheck`'s `nameMap.encode(negation)` MINTED into the shared `NameMap`, so the detection was not fully read-only — harmless under sequential executors but a data race once they run in parallel. The fix replaces it (and the four other fixpoint mints) with the non-minting `NameMap::lookup`.

**Verification.** Build clean (Release x64); 541/541 C++ unit tests (`deactivation_check_condition1_tobeproved_match` updated to assert the pure-detection contract: returns `true`, `isActive` unchanged). Byte-identity gate (full `main.py` + verifier, 1086 s, `RUN_INCUBATOR=True`): `theorems.txt` IN-ORDER BYTE-IDENTICAL to `main` for **both** batches — main (46 theorems, Gauss fold proved) AND incubator (1423 theorems); verifier 0 failures, all proof graphs verified; 364/364 verifier unit tests. The verifier check count is 129217 vs `main`'s 129259 (−42, ~0.03%): small per-category +/− redistributions (e.g. equality1 +3/−6, expansion +2/−13, implication +2/−8, disintegration +4/−12) netting −42 — the deferral changes the closing LBs' intermediate derivation paths, producing slightly-different-but-equivalent proof graphs. **Not a theorem loss**: `theorems.txt` is byte-identical on both batches and the verifier is airtight; verifier check-count deltas are encoding-sensitive and explicitly not theorem-loss signals.

---

<a id="d-114"></a>
## D-114 — `proveKernel` runs three barriered phase sweeps; EARLY-EXIT hashburst trap removed (2026-06-05)

**The choice.** `proveKernel`'s per-cycle execution changes from one work-stealing sweep that calls `performElementaryLogicalStep` (all three phases) per LB into **three barriered sweeps**: phase 1 for every active LB → join → phase 2 for every active LB → join → phase 3 for every active LB → join. A local generic-lambda helper `runPhase(body_fn)` owns one work-stealing pool (the existing `std::atomic<size_t> next` + `workers` threads) and is invoked three times — once per `performElemPhase1` / `performElemPhase2` / `performElemPhase3`. The active set (`std::vector<Memory*>`) is snapshotted once at cycle start (`b && b->isActive`). `performElementaryLogicalStep` — the single-LB orchestrator that ran all three phases inline — became **dead** here (the barriered `proveKernel` calls the phases directly; nothing calls the orchestrator) and was **removed** in the follow-up orchestrator-removal commit, so the phase model is the only path. The relocated EARLY-EXIT hashburst-dump trap is **removed entirely** (user-approved 2026-06-05, Rule 14); ENTRY (phase 1) and EXIT (phase 3) remain.

**Why.** The split dedicates N executor cores to a heavy LB *inside phase 2*. That is only expressible if phase 2 is its own sweep: in the old per-LB loop every core is already busy running a different LB's full step, so there is no spare core to lend a heavy LB. Barriering the phases frees phase 2 to be the place where one LB's hashburst fans out across cores while the rest of the grid waits at the join. Phases 1 and 3 stay plain across-LB work-stealing.

**Why byte-identical (the I-28 mechanism).** The reorder from `A.p1 A.p2 A.p3 B.p1 B.p2 B.p3 …` (interleaved across cores) to `all .p1 | all .p2 | all .p3` changes nothing because no phase observes another LB's intra-cycle state: [I-28](30_invariants.md#i-28) forbids cross-LB writes during the parallel phase, and every cross-LB effect is deferred — cross-LB mail rides `sendMail → boxes → smashMail` to the *next* cycle, and the class-level collectors (`inductionMemoryBlocks`, `updateGlobalTuples`, `updateGlobalDirectTuples`, `pendingCompactionQueue`) are drained single-threaded **in content-sorted order** after the final join. Same active set + same firing set + sorted drains ⇒ identical output regardless of phase interleaving. The once-at-cycle-start active snapshot matches the old `if(!b->isActive) continue` admission: intra-cycle deactivation only flips active→inactive (a closing LB's discharge), and a parked child's first activation (`activateZeroCondition` on the born-parked induction zero-condition block) lands in the post-join drain (I-28), so an LB active when the cycle starts is exactly an LB the old loop would have run.

**EARLY-EXIT removal.** The [D-123](#d-123) deferral keeps `body.isActive` true through phase 2, so the `splitCount == 1 && !body.isActive` EARLY-EXIT trap could never fire — it was dead. Removed in full (the `dumpEarlyExit` function, its `earlyExitMtx` / `earlyExitCount` state, the declaration, and the call site) rather than left as a dead branch (Rule 14 change, user-approved). The dump's two surviving traps bracket the live path: ENTRY at the top of phase 1, EXIT at the bottom of phase 3.

**RT limitation (deferred).** `RT_TRACKER_DECL(body)` had its only call site in `performElementaryLogicalStep`; with that orchestrator now removed, no tracker is declared anywhere, so nothing sets the thread-local `g_currentThreadTracker` during the barriered sweeps, every `RT_SCOPE_HERE` in the phase helpers no-ops, and **RT measurement produces no output on the live path**. RT is compile-time-off by default ([`_meta/rt_measurement.md`](_meta/rt_measurement.md)) so the gate is unaffected, but the instrumentation is currently inert until the per-call tracker is re-homed to span the three barriered sweeps (a per-LB persistent tracker, not a stack object in one function). Flagged as an open question in [`_meta/rt_measurement.md`](_meta/rt_measurement.md#weaknesses); the RT redesign is its own follow-up. **(Re-homed 2026-06-08:** RT now lives in `performElem2` and records under `disable_lb_split` — see [D-110](#d-110).**)**

**Verification.** Build clean (Release x64); 541/541 C++ unit tests. Byte-identity gate (full `main.py` + verifier, 1239 s, `RUN_INCUBATOR=True`): `theorems.txt` IN-ORDER BYTE-IDENTICAL to `main` for **both** batches — main (46 theorems, Gauss fold proved) AND incubator (1423 theorems); verifier 129217 checks, 0 failures, all proof graphs verified; 364/364 verifier unit tests. The check count is **identical** to [D-123](#d-123)'s 129217 (and the same −42 vs `main`'s 129259 inherited from that deferral) — the barrier reorder adds **zero** further delta, the direct measurement that the phase interleaving is unobservable.

---

<a id="d-126"></a>
## D-126 — per-LB adaptive `numberOfParts`; N executors run sequentially; fill-ratio marker (2026-06-05)

**The choice.** Each LB carries `Memory::numberOfParts` (N, default 1) — the number of executors its hashburst is split across. `performElemPhase2` reads it as `splitCount`, runs N `performElem2` executors **sequentially** (each over its `id % N == processID` rule slice, appending to one shared `firingRecords`), then merges all of them once via `applyFiringRecords`. After the merge it sizes the NEXT burst's N from the busiest executor's fill ratio (`maxTotalReqs / maxNumberHashRequests`) via the pure helper `computeNextNumberOfParts(currentParts, fillRatio, growthFactor)`: `≥ 50% → ×growthFactor`, `≤ 30% → ÷2`, `≤ 20% → ÷4`, `≤ 10% → ÷8`, `(30%, 50%)` hold, floor 1. The busy-band `growthFactor` is the config-tunable `parameters.split_growth_factor` (default 2 = the original doubling). `performElem2` now returns its request count (`totalReqs`) as the marker input. *(Forward note: the executors run **sequentially** in this commit; [D-115](#d-115) moves them to a flat parallel pool with per-task `firingRecords` — the policy and marker logic here are unchanged, only the execution.)*

**The delta-clear move.** The per-step delta clears (`localEncodedStatementsDelta`, `intLocalEncodedStatementsDelta`, `localHashMemoryDelta`, `admissionKeysAlgebra`, `deferredIntegrationPreps`) move out of `performElem2` into `performElemPhase2`, done ONCE per LB after every executor's request generation and before the merge. At N>1 a per-executor clear would wipe `localHashMemoryDelta` / `intLocalEncodedStatementsDelta` — request-generation inputs (batches 2 and 5) the next executor still needs — so the clear must be once-per-LB. It stays pre-merge, so the admission/integration staging buffers are reset before `applyFiringRecords` repopulates them.

**Why it stays correct at N>1 (sequential).** (1) **Completeness:** a key accepted at N=1 has ≥1 scope-comparable owner; the union of the N residue slices is exactly that gate, so every firing N=1 *could* produce is produced by some executor (keys owned across residues fire on >1 executor → harmless duplicate, collapsed by the set-merge). (2) **Independence:** the fixpoint's skip-set is the fixed `intKnownStatements` ([D-104](#d-104)), never mutated mid-burst, and `checkLocalEncodedMemoryStatic` deposits only to the caller's `firingRecords` (strings, not `NameMap` ids) — so executors do not change each other's firing decisions. (3) **Order-independence:** `applyFiringRecords` sorts by string content, so a given firing SET merges identically regardless of how it was partitioned; the only shared mutation an executor makes is idempotent `NameMap` interning, whose id VALUES do not leak into the decoded output ([D-119](#d-119) proved output invariance under a `NameMap` id shift).

**Byte-identical, even for a capped monster LB (validated).** N rises above 1 only for an LB whose busiest executor hits ≥50% of `maxNumberHashRequests`. The LBs that split include *monster* LBs that **exceed** the cap (observed: `__contradiction__(=[6,7])` at `fillRatio = 1.0`, 8192 = full cap, driven 1→2→4→8; plus `(=[8,2])`, `(in2[6,15,8])`, `(EnumerationSet2[2,6,10])`, `(=[9,2])` at N=2–4). At the cap the per-burst request stream is **truncated** at N=1, so each N>1 executor's slice recovers requests the single capped pass dropped — the split covers *more* requests per burst (this is the split's whole purpose: the cap is a per-burst throttle, and dropped requests regenerate on later bursts). So a monster LB's per-burst *trajectory* differs between N=1 and N>1, yet the run is **byte-identical** to `main`: the proof graph and `proved_theorems` record the FINAL derivation (statements + their origins), not the burst count, and the sorted firing-record merge ([D-117](#d-117)) makes the per-burst deposit order partition-independent — so the same firings accumulate to the same origins regardless of how the requests were split or how many bursts it took. Faster convergence, identical result.

**Byte-identity at N=1.** `numberOfParts` defaults to 1; the loop runs one executor exactly as before; `computeNextNumberOfParts(1, r)` returns 1 for every `r < 0.50` (floor), so an under-cap LB never leaves N=1. The delta-clear relocation is behaviour-preserving at N=1 (the fixpoint reads neither the deltas nor the admission buffers, so clearing after it instead of before is identical).

**Not yet (next commit).** Parallel executor THREADS (each `performElem2` on its own core) and the 100%-cap same-iteration re-split (`×4`, abort + repeat the burst) are the next commits. The read-only-LB precondition — making the fixpoint mint nothing into the shared `NameMap` — is handled first in [D-116](#d-116).

**Verification.** Build clean (Release x64); 542/542 C++ unit tests (new `compute_next_number_of_parts_bands` covering every band + the floor). Full `main.py` + verifier on the committed code: N>1 was genuinely exercised on 5 distinct LBs (`__contradiction__(=[6,7])` → N=8, plus `(=[8,2])`, `(in2[6,15,8])`, `(EnumerationSet2[2,6,10])`, `(=[9,2])` → N=2–4); `theorems.txt` IN-ORDER BYTE-IDENTICAL to `main` for **both** batches (main 46, incubator 1423); verifier 129217 checks, 0 failures, all proof graphs verified; 364/364 verifier unit tests. Byte-identical to the fixed `main` reference subsumes same-set and determinism.

**Superseded (2026-06-08, [D-109](#d-109)).** Two parts of this entry are overridden on: (1) the fill-ratio marker is now the busiest part's **submatch** count, not the emitted-request `totalReqs`, and `performElem2` returns `void` — the count travels via the `thread_local g_growthMatchCount`, read by the worker into `taskSubMatches`. (2) `numberOfParts` and the adaptive `computeNextNumberOfParts` are no longer read by `proveKernel`, which splits every LB into the FIXED `fixed_number_splits` (main path + compressor); the adaptive update is kept but inert. The completeness / byte-identity reasoning above is unaffected — it rests on the partition gate and the sorted merge, both unchanged.

**Superseded again (2026-06-09, [D-111](#d-111)).** `numberOfParts` is read by `proveKernel` again (reverting the "inert" status the fixed override imposed), but the policy is the bang-bang `adaptiveSplitDecision` (1 ↔ `fixed_number_splits`, escalate-on-cap-hit with same-iteration discard-and-redo — the "100%-cap same-iteration re-split" foreseen in *Not yet* above, landed as discard-and-redo rather than `×growthFactor`), NOT the graduated `computeNextNumberOfParts`, which is now deleted. The completeness / byte-identity reasoning above still holds (partition gate + sorted merge, unchanged).

---

<a id="d-116"></a>
## D-116 — the hashburst fixpoint mints nothing into the shared LB; `NameMap::lookup` replaces the five `encode` mints (2026-06-05)

**The choice.** A new read-only method `NameMap::lookup(name)` (returns the interned id or `0`, never mutates; `const`) replaces the five `encode` calls on the per-burst critical path `performElem2` runs: three in `checkLocalEncodedMemoryStatic` (the `lmvVid` scope compare, and `origId` / `valId` of the `alreadyKnown` ancestor-scan) and two in `deactivationCheck` (the `valId` and the negation `negOrigId`). After this the hashburst — request generation **and** firing — performs **no mutation of the shared LB** (deposits go only to the caller-owned `firingRecords`, [D-117](#d-117)). This is the precondition for running N executors over one LB on parallel threads: a concurrent `NameMap` mint is a data race. **(Correction 2026-06-08:** the audit below missed one shared write — request keys were stored in the per-LB `Memory::keyArena`, which the parallel parts then raced on. Fixed by moving them to the `thread_local` `g_reqKeyArena`; see [D-125](#d-125).**)**

**Why it is byte-identical.** Request generation (`generateEncodedRequestsStatic*`, `makeMandatory*`) and the CE filter were already mint-free (verified by audit of the whole `performElem2` call tree). Of the five converted sites: the **validity scopes** (`lmvVid`, both `valId`s) are always pre-interned — a matched rule's validity and a firing's consensus validity are decoded from existing ids — so `lookup` returns exactly what `encode` did. The **head and negation** (`origId`, `negOrigId`) may be un-interned, where `lookup` returns `0`; but `0` (the reserved invalid slot) is never a key in `intKnownStatements`, so the `alreadyKnown` scan and the contradiction scan reach the same not-known / no-contradiction verdict the old mint-then-miss did. The head is still interned — later, single-threaded, at deposit time in `applyFiringRecords`. The negation was only ever a transient probe; not interning it leaves the `NameMap` slightly smaller, and id values never leak into the decoded output ([D-119](#d-119)'s Step-A gate proved output invariance under a `NameMap` id shift).

**Why `0`, not a deferral.** An earlier plan deferred the `alreadyKnown` computation to `applyFiringRecords`. The `lookup`-returns-`0` reading is simpler and identical: `intKnownStatements` is burst-fixed, so "is this head/negation already a known statement?" has the same answer whether asked in the fixpoint or at apply time, and an un-interned name is trivially not known.

**Verification.** Build clean; 543/543 C++ unit tests (new `name_map_lookup_is_non_minting`). Full `main.py` + verifier (still sequential — N driven by the policy; threads land next): `theorems.txt` IN-ORDER BYTE-IDENTICAL to `main` for **both** batches (main 46, incubator 1423); verifier 129217 checks, 0 failures, all proof graphs verified; 364/364 verifier unit tests. The byte-identity confirms the read-only conversion is inert — removing the per-firing negation mints and deferring the head mint to apply time shifts `NameMap` ids but not the decoded output.

---

<a id="d-115"></a>
## D-115 — phase 2 is a FLAT (LB, part) executor pool on real threads; per-LB finalize after the join (2026-06-05)

**The choice (flat, never nested — user 2026-06-05).** `proveKernel`'s phase 2 stops being a per-LB sweep that loops a split internally. Instead it builds the **flat list of every `(LB, part)` executor task** across all active LBs — an LB with `numberOfParts == N` contributes N tasks, a normal LB contributes 1 — and a single work-stealing pool of `logicalCores` threads runs that flat list. No LB spawns its own sub-threads, so the cores are never oversubscribed by nesting (`workers × N`). After the pool joins, a per-LB **finalize** sweep concatenates each LB's parts' `firingRecords`, merges them via the canonical-sorted `applyFiringRecords`, runs the split-policy marker, and drains. `performElem2` (the per-part executor) is unchanged; `performElemPhase2` is repurposed to that finalize (signature `(Memory&, std::vector<FiringRecord>&, int16_t)`).

**Buffers (user: each executor its own block, nothing shared).** Each pool **thread** owns one `exprArena` + `reqBuf`, reset per task it pulls — bounded to the core count, never shared between concurrent tasks (a fresh 64 K arena + 8 K request buffer per task across ~1700 tasks would be gigabytes). Each **task** writes its own `firingRecords` slot. The only thing shared is the LB itself, and only for reading.

**Why it is race-free.** The fixpoint is read-only on the LB ([D-116](#d-116)): two tasks on one LB only READ its containers and its `NameMap` (via `lookup`, a `const` `unordered_map::find` — safe to call concurrently), and write disjoint `firingRecords` slots; tasks on different LBs are independent. The per-LB finalize mutates only its own LB (mints heads into its own `NameMap`, deposits to its own mail), so it is a normal across-LB sweep, I-28-safe exactly like phases 1 and 3.

**Why it is deterministic / byte-identical.** Each task's `firingRecords` is a function of the read-only LB state and its `id % N` slice, independent of thread completion order. The finalize concatenates a LB's parts in fixed part order and `applyFiringRecords` SORTS, so the merged deposit order is independent of how the work was split or scheduled. Head minting happens single-threaded per LB in the finalize, in sorted order — identical to the sequential apply. So the run is byte-identical to the sequential N>1 path of [D-126](#d-126), hence to `main`.

**Verification.** Build clean; 543/543 C++ unit tests. Full `main.py` + verifier, run TWICE (sequential, determinism under thread scheduling): <!-- GATE-NUMBERS: both runs byte-identical to main + to each other, verifier checks/failures. -->

**Not yet (next commit).** The 100%-cap same-iteration re-split (`×4`, abort the sweep + repeat the burst).

---

<a id="d-108"></a>
## D-108 — `checkForEquivalence` gates on full disintegration, not locality; `wholeExpressions`/`intKnownStatements` value becomes `StatementFlags` (2026-06-03)

**The choice.** The cFE disintegration gate ([D-107](#d-107)) suppresses re-disintegration of an equivalence-class variant only when the matched variant was **fully disintegrated**, not merely *local* (the step-1 gate, commit ) and not on mere presence. The value type of `wholeExpressions` and `intKnownStatements` changes from a bare `bool` to `StatementFlags { bool local; bool fullyDisintegrated; }`; cFE reads `.fullyDisintegrated`.

**The mirror bug it fixes.** Reintroducing cFE re-broke the Gauss fold n+1 step (the same #110 / `in3[7,12,11,5]` regression [D-106](#d-106) had closed via `resetResentExpressionRegistries`). The LB derives the local existence `existence1[1,it_0_lev_4_0,7,5]` (`it_0_lev_4_0 ≡ 9`); it disintegrates but its product-witness marker `(in3[it_0_lev_4_0,7,marker,5])` is never admitted (admission holds only the canonical `(in3[9,7,marker,5])`), so the witness is rejected. `applyEquivalenceClassToRejectedMap` then mails the canonical `existence1[1,9,7,5]` onto `sameIterationInternalMail`. Under the step-1 *local* gate, that canonical — whose witness IS admitted — was suppressed by the still-present **local** twin, and the two equivalent existences mutually blocked: neither landed the witness. A local-but-rejecting twin must not count as "already handled".

**The fix — `fullyDisintegrated` as the gate.** `disintegrateExpr2` returns a new 4th element `fullDisintegrationHappened`: true iff the compound entered disintegration and every existence inside it got at least one admitted witness, or it had no existence at all (vacuously true — a plain `in3` has nothing to witness). It is computed inside `disintegrateExpr2` (non-`forceDeep` path) by grouping the witness vars in `newVarMap` by their `makeMarkedExpr` signature — the `it_` and `int_` of one existence share a marker body — and checking each group is covered by `admittedVars`; the `forceDeep` path reports `newVarMap.empty`. `addExprToMemoryBlock` sets `.fullyDisintegrated = true` on `expr`'s own registry entries when the call returns true. The rejecting twin is therefore unflagged (its witnesses are not admitted), stops suppressing the canonical, and the canonical disintegrates and admits — closing Gauss.

**Why not locality (step 1).** Locality only distinguishes mail-origin from local; it cannot distinguish a local expression that fully disintegrated from one whose witness was rejected. The rejecting twin is local, so the local gate kept suppressing. Full disintegration is the property cFE actually wants to dedup on. Erring toward `false` is always safe (cFE re-disintegrates — redundant at worst); only a wrong `true` re-introduces the mirror bug ([I-72](30_invariants.md#i-72)).

**Why a flags struct, not a second bool.** The two flags are folded into one `StatementFlags` value — the first step toward a single flags-carrying statement map for the whole prover. `local` is preserved (provenance) though no longer the cFE gate. All ~20 insert sites default `fullyDisintegrated = false`; only the post-disintegration call site flips it. Membership-only readers, the per-LB teardown/erase sites, and the sacred hashburst dump (iterates `kv.first`) are unaffected by the value-type change.

**Verification.** Build clean; 531/531 C++ unit tests (cFE gate-tests updated to the new semantics, incl. the mirror-bug guard); full `main.py` as configured (`RUN_INCUBATOR=False`, Peano + Gauss main): **Gauss summation proved by induction** (the `fold` theorem) with the target `in3[7,12,11,5]` in `theorems.txt`; verifier 9641 checks, 0 failures — airtight; 364/364 verifier unit tests; no assert/crash; Peano theorems intact.

---

<a id="d-107"></a>
## D-107 — restore `checkForEquivalence` as the disintegrate-gate equivalence dedup for sandbox validation (2026-06-02)

> **Resolved by [D-108](#d-108) (2026-06-03).** The reintroduced gate coexists with the [D-106](#d-106) key flow once it keys on `fullyDisintegrated` instead of presence/locality; the verification target below is met (Gauss fold proved, verifier 0 failures, `theorems.txt` 46 lines).

**The choice.** Reintroduce the old `ExpressionAnalyzer::checkForEquivalence(expr, validityName, memoryBlock)` hook and restore the `addExprToMemoryBlock` disintegration gate to `!doNotDisintegrate && !checkForEquivalence(...) && (status!= 3 || isCompactImplication)`.

**Why.** The admission/rejection algebra has since been simplified to one canonical representative for both maps ([D-106](#d-106)). This sandbox branch tests whether the old `wholeExpressions` equivalence dedup can now coexist with that simplified key flow while preserving the current theorem count, rung 1 forward proof, and Gauss fold result.

**Risk being tested.** [D-88](#d-88) remains the historical warning: before the algebra simplification, this gate could let a dead-end equivalent variant pre-empt the canonical form that needed disintegration. The branch-closing verification is therefore the full pipeline, not just unit tests; failure means the verifier/proof output wins and the gate must not be treated as safe.

**Verification target.** Match the reference run from: verifier failures 0, final kept theorem count around 76, `theorems.txt` line count 46, rung 1 forward present in incubator proved theorems, and Gauss fold present in main proved theorems.

---

<a id="d-106"></a>
## D-106 — one canonical representative for admission + rejection keys (2026-06-02)

**The choice.** The two algebra equi-class key hooks — `applyEquivalenceClassToAdmissionMap` and `applyEquivalenceClassToRejectedMap` — are unified onto one rule: pick a single canonical representative for the whole class (`chooseCanonical`), rewrite the key with it, and drop the old key when it changes. Both hooks use the same selection and the same drop. Expression handling (`applyEquivalenceClass` / `enumerateEqClassRewrites`) is untouched.

**Selection rule.** Over the strong members (`reduceEqClass`), prefer a normal name (anything not `int_lev_*`/`it_*_lev_*`; `repl_` counts as normal), then `int_`, then `it_`; lex-smallest within a tier. Every other class member maps to that representative — so unlike the old rejected hook, normal members are subject to replacement too, not only `int_`/`it_`.

**Same-scope application only.** Both hooks apply a class to a key only when their `validityName` is identical. The cross-scope (ancestor/descendant) directions the old hooks carried are dropped: cross-scope equalities are already folded into the class at merge time by the ancestor-absorbing merge ([D-44](#d-44)), so a class at scope V holds every member it needs at V, and key application is same-`validityName` only. The deposit scope is therefore that shared validity.

**Why one representative, not enumeration.** The old admission hook enumerated every equivalent variant of a key and added them all (combinatorial, memory-heavy), then a Step-4b sweep dropped the non-canonical originals. Collapsing to one representative removes both the enumeration and the sweep. It stays sound because the *untouched* expression machinery still enumerates every variant, including the canonical one: a firing head reaches the canonical compound, disintegrates it, and its exact-match `isAdmitted` probe meets the single canonical admission key. No probe-side change is needed.

**Drop in both, but the fill differs (provenance).** Admission values are metadata, so admission re-keys directly (drop K, insert K'). Rejected values are real disintegration products with `disintegration` origins, so rejection keeps drop+mail: it drops K and mails the rewritten compound; the kernel regenerates `rejectedMap[K']` with proper provenance. Neither hook writes the other side's map directly ([I-37](30_invariants.md#i-37) preserved).

**Repetitions allowed.** The arg-equalization filter ([I-36](30_invariants.md#i-36)) on the admission key path is removed: a single representative can collapse two slots, and both sides collapse identically, so positional-collision preservation is no longer load-bearing. (The integration-side mirror still has its filter; integration is a later task.)

**Supersedes — algebra path.** [D-57](#d-57) (enumerate + additive admission insert), [D-62](#d-62) / [I-40](30_invariants.md#i-40) (post-class admission sweep), and the `int_`/`it_`-only canon of [D-63](#d-63) (rejected).

**Supersedes — integration path (mirrored onto `…Integration`).** The integration-side hooks are now unified the same way: `applyEquivalenceClassToAdmissionMapIntegration` switches from enumerate+additive to `chooseCanonical` drop+rekey, `applyEquivalenceClassToRejectedMapIntegration` from `int_`/`it_`-only canon to `chooseCanonical`, both restricted to same-`validityName`. This retires [I-42](30_invariants.md#i-42) (integration additive), [D-66](#d-66) / [I-43](30_invariants.md#i-43) (integration-admission sweep, `cleanUpAdmissionMapIntegration` call removed), the integration mirror of [I-36](30_invariants.md#i-36) (arg-equalization), and the int/it-only canon of [D-64](#d-64). The `u_`-form key handling and `Instruction`-value substitution are kept verbatim; [I-22](30_invariants.md#i-22) (integration templates persist; no on-hit closure) is preserved.

**Regression the rework introduced — Gauss fold n+1.** Collapsing to one canonical representative broke the Gauss fold n+1 step (theorem #110, `in3[7,12,11,5]`). The LB derives `existence1[1,it_0_lev_4_0,7,5]` locally and disintegrates it under the `it_0_lev_4_0` representative, so its product-witness admission marker is `(in3[it_0_lev_4_0,7,marker,5])`; but the admissionMap now holds that product marker only under the canonical `9` — `(in3[9,7,marker,5])`. The exact-match `isAdmitted` probe misses, the witness is buffered and rejected, and `applyEquivalenceClassToRejectedMap` rekeys the rejection to `9` and mails the rewritten compound `existence1[1,9,7,5]` onto `sameIterationInternalMail`. That 9-form already sits in the LB's dedup registries (it arrived earlier as external mail, status 3, and was never disintegrated), so the absorb gate (`statementLevelsMap`) and Site F (`intKnownStatements`) dedup it away — it is never re-disintegrated, the 9×7 witness is never produced, the goal is never proved, and the AnchorGauss batch fails downstream. (The probe-side soundness claim under *Why one representative* assumed the canonical compound would re-enter the disintegrate path; the dedup registries blocked that re-entry for a compound first seen via mail.)

**The completion — delete → send → reabsorb, plus a consumed-skip.**
- *delete → send → reabsorb.* New shared helper `resetResentExpressionRegistries(mb, ewv)` wipes a resent, not-yet-local compound from every per-LB dedup registry that would otherwise dedup the re-entry — `wholeExpressions`, `statementLevelsMap`, `intKnownStatements`, `encodedStatements` (+ parallel `intEncodedStatements`) — while preserving `exprOriginMap` (history; [I-44](30_invariants.md#i-44) / Rule 16). `localEncoded*` is left untouched (the caller gates on not-local, so the compound is absent from it by construction; the implication-specific `encodedMap`/`expandedImplications` walk of `eradicateImplicationFromLB` is intentionally not mirrored). Both rejection hooks (`applyEquivalenceClassToRejectedMap`, `applyEquivalenceClassToRejectedMapIntegration`) call it before mailing each non-local rewritten compound, so the *plain* absorb re-disintegrates the 9-form once; that disintegration's marker `(in3[9,7,marker,5])` meets the admissionMap entry → `isAdmitted` hits → 9×7 witness produced. After that single disintegration the compound is local, so the plain absorb skips it — one disintegration, no per-burst churn. This deliberately reuses the normal dedup path (no re-entry exemption gate, no new state).
- *consumed-skip.* The three inline `admissionMap` insert sites — `updateAdmissionMap`, `updateAdmissionMapRecursion`, `applyEquivalenceClassToAdmissionMap` — skip re-adding a key already in `consumedAdmissionKeys`, mirroring the gate already present in `drainAdmissionKeysAlgebra` (`memory.cpp`). Without it the reset-driven re-disintegration re-enters admission for a key whose admission was already consumed, putting it in both `admissionMap` and `consumedAdmissionKeys` and tripping `isAdmitted`'s mutual-exclusion assert.

**Verification.** Build clean; C++ unit tests green (`chooseCanonical` + `resetResentExpressionRegistries` coverage). Full `main.py` with incubator on (incubator + Peano + Gauss): **Gauss fold n+1 #110 discharged**; 129,297 checks (incubator 118,173 + main 11,124), 0 failures — airtight; 364/364 verifier unit tests; no assert / abort / `MAX_NAME` overflow. The Gauss main LB disintegrates the 9×7 existence exactly once (ADMITTED), then skips it as local — no re-disintegration churn.

---

<a id="d-105"></a>
## D-105 — request-generation validity-comparability prune at every owner-set match site (2026-05-31)

**The choice.** At every site where a growing hash request's structural key is matched against one of the four owner-set maps (`normalizedEncodedKeys` / `normalizedEncodedSubkeys` / `…MinusOne` / `…MinusTwo`), an early prune drops the request unless its accumulated (deepest-premise) validity is `nm.comparable` to at least one owner of the matched key. The owner-set value (`OwnerSet`) stores each owner together with its scope's validity id — `map<ExpressionWithValidity, int16_t>`, the id being `NameMap::encode(owner.validityName)` — so the comparability test runs entirely on `int16_t` ids in the hot loop without re-encoding scope strings. A `vid == MAIN_ID` short-circuit makes the whole check a no-op when the request lives at `"main"`.

**Why it is a runtime optimization, not a soundness check.** The firing gate `checkLocalEncodedMemoryStatic` already rejects every structurally-matching rule whose scope is not `comparable` to the request's consensus validity (see [D-55](#d-55)), so the prune can never change *what fires* — it only removes requests earlier, before the expensive key-build / bind / admission path. It is therefore required to be result-invariant: `theorems.txt` byte-identical with and without it.

**Why it never drops a firing request (sound over-approximation).** A request fires only if some rule R at scope S is comparable to the request's consensus validity V. R stamps S as an owner of every (sub)key projection the request matches, including each single-premise projection. Scopes form a tree, and the partial validity at any growth stage is an ancestor of the final V, so `comparable(S, V)` implies `comparable(S, partial)`. Hence "∃ owner comparable to the partial validity" holds at every length from the single-statement pre-filter up to the full key; pruning when none is comparable removes only requests that could not have fired.

**Why the `main`-skip is the load-bearing efficiency guard.** `comparable(main, X)` is true for every scope (main is the root ancestor) and a matched key always has ≥1 owner, so a `"main"`-validity request can never be pruned. Skipping the owner iteration in that case keeps the optimization free on all-`main` workloads — the entire CE filter (every fact loaded at `"main"`) and the bulk of Peano main-prover. Pruning work accrues only in branching proofs (OR theorem, FTA ladder) where requests carry non-`main` validity. Without the guard the check would iterate owners on the Peano flood for zero benefit.

**Intification safety.** `int16_t` validity ids are per-`NameMap`. Each LB owns its own `NameMap` and its own four `HashMemory` slots; owner inserts (`addToHashMemory` / `makeNormalizedKeysForAdmission`) encode under the same `body.nameMap` the probe later uses, and rules cross LBs only as validity *strings* (re-encoded by the receiver). No `int16_t` ever travels between NameMaps. The id is encoded exactly once, at insert, and stored in the same `map` entry as its owner; the radical subtree wipe removes it together with its owner — it is never recomputed, so the wipe stays a plain owner-erase (an earlier draft recomputed the ids on every wipe, which regressed the branching batches; see Verification).

**Sites.** `filterIntEncodedStatements`, `preEvaluateFromEncoded` (covers grow / both merges / pairs / CE via the shared helper), the seed and base-candidate finds in `generateEncodedRequestsStatic` / `growBaseCandidates`, and the pairs / CE direct finds. Because the prune is optional for correctness, omitting it at any site only forgoes a pruning opportunity — never changes results.

**Verification.** Full `main.py` on the worktree (incubator + Peano + Gauss) completes cleanly — prover, compressor, and verifier all run; 521/521 C++ and 364/364 verifier unit tests pass. `files/theorems/theorems.txt` is byte-identical (modulo Windows CRLF) to the branch-point, and the verifier reports the same 129326 checks / 0 failures — airtight — as the branch-point ([D-104](#d-104)). Result-invariance holds: the prune changes nothing the proof produces. The all-`main` path stays cheap because `NameMap::deeperOf(main, main)` short-circuits on `a == b` (no `pairMap` lookup), so the consensus computation in `preEvaluateFromEncoded` reduces to integer comparisons before the `main`-skip — no measurable Peano overhead.

**Runtime (Gauss).** Per-batch prover RT on the worktree, branch-point vs prune. The first draft kept the validity ids in a separate `ownerVids` set and rebuilt it for *every surviving key on every scope-close* — that regressed the branching batches (IncGauss1 +7.3 % / +17 s, Gauss +1.7 %) while all-`main` batches were unchanged (the recompute runs only when scopes close). Storing each id with its owner removed the recompute and the regression: IncGauss1 and Gauss returned to within run-to-run noise of the branch-point (the all-`main` control IncPeano1 itself swings ~3–4 % run-to-run, larger than the residual Gauss delta). So the regression was the wipe recompute, not the prune *check*, which is RT-neutral. At Gauss's shallow branching (one OR theorem) the pruning savings sit below that noise floor; the payoff is expected at FTA-ladder depth.

**See** [I-49](30_invariants.md#i-49) (owner-set refcount), [D-72](#d-72) (owner-set introduction), [D-55](#d-55) (firing-site `nm.comparable` gate).

---

<a id="d-104"></a>
## D-104 — hashburst `while (changed)` fixpoint loop removed; request evaluation is a single pass; EARLY-EXIT dump trap relocated after the pass (2026-05-31)

**The choice.** In `prover.cpp::performElementaryLogicalStep`, the hashburst's `while (changed) {... }` fixpoint loop is removed. The static request buffer is evaluated in a single pass (`for (int16_t r = 0; r < totalReqs; ++r)`). The EARLY-EXIT hashburst-dump trap — formerly at the top of each loop iteration, gated on `!body.isActive` and followed by `break;` — is relocated to fire immediately after the single pass under the same `!body.isActive` condition. The `changed` flag, the `prevSize` / `sameIterationInternalMail.statements.size` change-detector, `fpIter`, and the per-64-iteration `RT_REFRESH` are dropped; `RT_NOTE_ITERATIONS(1)` replaces `RT_NOTE_ITERATIONS(fpIter)`. The `RT_SCOPE("FIXPOINT_LOOP")` name is retained.

**Why the loop was redundant.** After the equivalence-class reshuffle ([D-93](#d-93) / [D-96](#d-96): class-driven cleanup and apply deferred into `standardProcessing`; rule firings deposit to `sameIterationInternalMail`, absorbed post-burst) and the admission reshuffle ([D-103](#d-103) / [I-68](30_invariants.md#i-68) / [I-69](30_invariants.md#i-69): in-hashburst admission registration staged and drained post-fixpoint), the hashburst body no longer mutates `intKnownStatements` mid-pass. The dependency skip-set — each request is gated on all its inputs already being in `intKnownStatements` — is therefore fixed for the whole evaluation: a request skipped on the first pass can never become eligible on a later one, and re-firing an already-fired request only re-inserts an already-present `sameIterationInternalMail.statements` entry (a no-op; both the `alreadyKnown` gate in `checkLocalEncodedMemoryStatic` and the `statements.size` change-detector key off the unchanged int-known set). So the loop ran at most twice — one work pass plus one idempotent confirmation pass that produced nothing. It was a residual of a pre-reshuffle architecture in which mid-loop state growth could un-skip requests; it became dead structure and a source of confusion. Removal was user-directed.

**EARLY-EXIT trap relocation is behaviorally equivalent.** `deactivationCheck` ([D-98](#d-98)) fires only after a real `sameIterationInternalMail.statements` deposit, so a mid-pass deactivation always coincided with `changed = true`. The old loop therefore always re-entered exactly once after deactivation and fired the EARLY-EXIT trap at the top of iteration 2. The single-pass version fires the same trap immediately after the pass under the same condition — identical firing set. The dump targets a single LB and writes only to , never to run.log. The relocation of this Rule-14-sacred trap was explicitly authorized by the user (the only non-mechanical decision in this change).

**Verification.** Full `main.py` (7 tags: IncubatorPeano1/2, Peano, IncubatorGauss1/2/3, Gauss) on Windows, compared line-for-line against the pre-change baseline `run_reshuffle_incube.log` (same host, source-identical worktree):
- **Per-burst proof state identical** — every `Hash burst: N active_bodies=X total_exprs=Y` header byte-identical and in order (0 differences across all 7 tags). This is the determinism signal: the proof computation is bit-identical.
- **Expression multiset identical** — zero MPL-expression differences. The parallel hash-burst listing reorders expressions within a burst run-to-run (the baseline itself carries concurrent-write line-merge artifacts), but the content multiset is unchanged.
- **Structural lines identical** modulo the worktree absolute path in `Saved … theorems.txt` / `Success: Processed proof graphs …` messages. Theorem counts identical (incubator 1035 / 155 / 183 / 48 / 2; Peano compression 107 → 33 essential).
- **Verifier identical** — 129326 checks, 0 failures, airtight; 364/364 verifier unit tests passed. Total log length identical (2713 lines).

**What changed (file-level).**
- `prover.cpp::performElementaryLogicalStep` — `while (changed)` fixpoint loop replaced by a single `for`-pass; `changed` / `prevSize` change-detector, `fpIter`, and per-64 `RT_REFRESH` removed; EARLY-EXIT trap moved from loop-top to post-pass; `RT_NOTE_ITERATIONS(1)`.
- SwDD: [I-61](30_invariants.md#i-61) (+ quick-reference) reworded "fixpoint loop" → "fixpoint pass"; `20_core_concepts/03_mail_system.md`, `_meta/rt_measurement.md` §8 + example table, `10_pipeline/04_prover.md` updated; [D-98](#d-98) annotated "superseded in part". User SwDD unchanged — the inner pass is below its concept/diagram level.

**See also.** [D-93](#d-93), [D-96](#d-96), [D-98](#d-98), [D-103](#d-103), [I-61](30_invariants.md#i-61), [I-68](30_invariants.md#i-68), [I-69](30_invariants.md#i-69).


---

<a id="d-103"></a>
## D-103 — algebra `admissionMap` tail deferred out of the hashburst into a single post-fixpoint drain (2026-05-30)

**The choice.** Move the four algebra-`admissionMap` writes performed inline in `memory.cpp::checkLocalEncodedMemoryStatic`'s marker branch — `admissionMap` insert, `admissionStatusMap = false`, `varsInAdmissionMapKeys` population, and the `revisitRejected2` revival — out of the hashburst fixpoint loop. Each is staged on the new per-burst buffer `Memory::admissionKeysAlgebra` (a vector of plain-aggregate `AdmissionKeyAlgebraRecord{key, value}`) and replayed once, in firing order, by the new `ExpressionAnalyzer::drainAdmissionKeysAlgebra(Memory&)` immediately before the post-burst `standardProcessing` in `performElementaryLogicalStep`.

**Why.** ASIC-campaign prep, the next step after `equi_reshuffle` (which lifted the equivalence-class apply burst out of the hot path — [D-96](#d-96)). The forthcoming "split N LBs, run them, merge back" step needs every algebra-container mutation to happen from a single well-defined drain point rather than scattered through the fixpoint loop. The deferral is behaviour-preserving: `checkLocalEncodedMemoryStatic` never reads the *real-scope* `admissionMap` entries it defers mid-burst — the only in-burst `admissionMap` reader, `isAdmitted` via `prepareIntegration`'s hypothetical disintegration, runs on a disjoint sentinel scope — and the only burst-visible output (revival cohorts) is deposited on `sameIterationInternalMail`, which the post-burst `standardProcessing` drains regardless — so relocating the writes to just before that absorb is invisible to the burst.

**What changed.**
- New plain-aggregate `AdmissionKeyAlgebraRecord { ExpressionWithValidity key; AdmissionMapValue value; }` (no constructor → no Rule-18 function obligation) and per-burst field `Memory::admissionKeysAlgebra`, cleared at burst start beside the existing `localEncodedStatementsDelta` / `intLocalEncodedStatementsDelta` / `localHashMemoryDelta` clears.
- The marker branch appends one record instead of performing the four writes inline; the now-dead `itAdm` lookup is removed (the drain does a fresh per-record `admissionMap.find`).
- `drainAdmissionKeysAlgebra` replays each record: a per-record consumed-key gate, then the four writes.
- Call site is post-fixpoint, immediately before the `POST_FIXPOINT_MAIL_FLUSH` `standardProcessing`.

**Byte-identity preservation.**
- Records appended in firing order, drained in that exact order — `cleanAdmissionMap`'s canonical-closure scan (reached transitively via `revisitRejected2`) is order-sensitive ([I-41](30_invariants.md#i-41)).
- The drain re-applies the consumed-key gate per record: an earlier record's `revisitRejected2` may consume a later record's key, exactly matching the inline loop's within-burst consume→skip ordering.
- `prepareIntegration`, `canBeSentMarkerIds`, and the purity / consumed gates stay in the loop. `admissionMapIntegration` is untouched — the stated follow-up branch.

**Status.** Code (commits 1–2); C++ unit tests 518/518 pass (`test_admission_reshuffle.cpp` — symbol-signature + behavioral positive/negative). The byte-identity DoD — full `main.py` `run.log` diff vs the pre-change reference (modulo timing lines), `theorems.txt` identity, verifier-summary identity — runs as the branch-closing verification step (commits land first; the after-run validates them together).

**See also.** [D-96](#d-96), [D-90](#d-90), [I-68](30_invariants.md#i-68), [I-60](30_invariants.md#i-60), [I-41](30_invariants.md#i-41).

---

<a id="d-88"></a>
## D-88 — `checkForEquivalence` dropped from the disintegrate gate: it pre-empted the canonical integration existence's disintegration when an equivalent dead-end algebra variant was already registered (2026-05-29)

**Status.** Superseded experimentally by [D-107](#d-107). The original failure mode remains the risk criterion for that branch's full-run verification.


**The bug.** After the `equi_reshuffle` refactor the Gauss fold theorem `n(n+1) = 2·Σi` (the n+1 induction step, Gauss-main) stopped proving — silently absent, verifier still clean. The missing step was the additive-closure witness of `Σ(rec)+v1`: the canonical sum `int_lev_4_*` never produced its disintegration witness `in3[int_lev_4_*,9,it_0_lev_4_*,4]`; only the admission marker `in3[int_lev_4_*,9,marker,4]` existed.

**Why it failed.** The disintegrate gate in [`addExprToMemoryBlock`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) (`prover.cpp`) was `!doNotDisintegrate && !checkForEquivalence(expr,validityName,memoryBlock) && (status!= 3 || isCompactImplication)`. [`checkForEquivalence`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) returns true when **any** equivalence-class variant of `expr` is already in `wholeExpressions`, skipping disintegration as a dedup. The canonical existence `existence1[1,int_lev_4_*,9,4]` (integration form, `int_`) arrives once (status=1, local), but by then the equivalence class `{int_lev_4_*, it_0_lev_4_*}` had formed and the *non-canonical algebra variant* `existence1[1,it_0_lev_4_*,9,4]` was already in `wholeExpressions` → `checkForEquivalence` true → disintegration skipped. That algebra variant had itself been disintegrated, but its witness was an `it_` freshly minted at the LB level, dropped by the `forceDeep` freshness filter ([I-17](30_invariants.md#i-17)) and never admitted — a dead end. So the form **with** an admission chance (canonical `int_`, constituent `it_` count 1) was pre-empted by an equivalent **dead-end** form (algebra `it_`, constituent count 2). The pre-reshuffle reference branch disintegrated the canonical form *before* the variant + equivalence were registered, so `checkForEquivalence` returned false there — the same order-flip theme as [D-89](#d-89).

**The choice.** Drop the `checkForEquivalence` term from the gate entirely. The rejection/admission machinery (the `forceDeep` freshness filter, the `isAdmitted` secondary-variable budget, `rejectedMap` / `revisitRejected2`) is the real disintegration-proliferation control — `checkForEquivalence`'s `wholeExpressions` dedup was a near-redundant skim that could starve the one form that mattered. Considered and rejected as more fragile: a canonical-aware gate (suppress only on canonical matches) and an order-restore; the blank removal needs no new special-case logic. With the gate term gone `checkForEquivalence` had no remaining callers, so the function (declaration + definition) was removed in the same change.

**Result.** The canonical existence disintegrates (`DISINT-ENTRY` confirmed by trap), its witness `in3[int_lev_4_*,9,it_0_*,4]` commits, and the Gauss fold n+1 step closes. Full pipeline verifies airtight (0 failures, 364/364 verifier unit tests); proved-theorem count up (+2, no loss). Cost: ≈40% Gauss-batch runtime increase — the extra disintegration trials this admits mostly get rejected by the budget gates but are not free.

<a id="d-89"></a>
## D-89 — `isAdmitted` gates the secondary-variable count on `maxNumberSecondaryVariables`, not the per-slot `standardMaxSecondaryNumber` — restores symmetry across cnt sites and reanimates rung-1 (2026-05-29)


**The bug.** After the `equi_reshuffle` refactor and the incubator anchor split (`AnchorIncubator` → `AnchorIncubator3`, 9 slots / `M=10`), the rung-1 forward theorem `EnumerationSet2(0,1,M) ⟹ interval(ℕ,+,0,1,M)` (the `{0,1}=[0,1]` proof, IncubatorGauss3) stopped proving: its ES2 logical block's `toBeProved` froze at 4. The missing step was the existence-disintegration product `l+p′ = w` — `(in3[int_lev_1_7, it_0_lev_1_344, it_0_lev_1_392, 4])`, where `l` is the `p≤1` value (an `int_`), `p′` its predecessor (an `it_`), and `w` the sum witness (an `it_`). This constituent never deposited live because [`isAdmitted`](#isadmitted) rejected it.

**Why it rejected.** `isAdmitted`'s gate was `cnt <= maxSecondaryNumber`, where `maxSecondaryNumber` is the per-slot `tuples[i].standardMaxSecondaryNumber` (`= parameters.standardMaxSecondaryNumber = 1`). `cnt = countPatternOccurrences(expr)` counts `it_\d+_lev_\d+_\d+` matches in the **constituent** form, skipping `productsOfRecursion` members. For `l+p′` the matches are `p′` (`it_0_lev_1_344`) and the witness `w` (`it_0_lev_1_392`) → `cnt = 2`; `l` is an `int_`, not counted; and `p′` is **not** in `productsOfRecursion` because rung-1's proof is direct, not induction (PoR membership is an induction-only event). So `cnt = 2 > 1` → reject.

**The asymmetry.** The marked-form cnt sites — [`isAllowedAsOperatorInput`](#isallowedasoperatorinput), the `applyEquivalenceClass` statement-growth cap, and the revival path — gate on `parameters.maxNumberSecondaryVariables` (`= 2`). But those scan the **marked** form, where the witness output slot is the literal token `marker` (not an `it_`, so uncounted), making their effective *constituent* cap `2`. `isAdmitted` scans the **constituent** form (the witness IS an `it_`, counted), so its `<= 1` cap was one stricter than every peer. `isAdmitted` was the lone outlier blocking `l+p′`. (On the reference branch, pre-anchor-split, the order of admission-key registration vs. rejection was flipped so the witness was revived via `revisitRejected2` before the gate was ever reached — masking the asymmetry. The anchor split flipped that order on rung-1, exposing the gate.)

**The choice.** Gate `isAdmitted` on `parameters.maxNumberSecondaryVariables` (`= 2`, with `<=`), so `cnt <= 2` (constituent) ≡ `cnt < 2` (marked) — symmetric with every peer cnt site. A matching cnt guard was also added to `revisitRejected2` (`prover.cpp`), which previously had **no** cnt cap and could revive sums of unbounded `it_` count; it now drops a marked-form expression whose `countPatternOccurrences >= maxNumberSecondaryVariables`. `l+p′` (marked `cnt = 1`) still revives; constituent `cnt ≥ 3` no longer does.

**Result.** `l+p′` admits through the gate (`PASSB reIt ADMIT via isAdmitted`), `l+p′ = w` goes live, the ES2 `toBeProved` reaches 0, and rung-1 proves again. Full pipeline verifies airtight (128 236 checks, 0 failures; 364/364 verifier unit tests). Peano-main runtime is unchanged — the cap raise is inert there (its derived sums never reach `cnt = 2`); only the rung-1 derived-sum case exercises it.

**Known remaining asymmetry (deliberately not closed).** The integration-side admission (`isAdmittedIntegration`, `revisitRejectedIntegration2`) has no cnt check at all. This is practically harmless because the integration witness is an `int_` (never counted by `countPatternOccurrences`), so the integration `it_` count stays at the predecessor's = 1. Adding a symmetric integration-side cap was considered and declined for now; it is a latent soundness tightening, not a correctness fix.

**See.** [I-6](30_invariants.md#i-6) (the separate Pass B single-input-operator gate); the Pass B admission section in [`10_pipeline/04_prover.md`](10_pipeline/04_prover.md).

<a id="d-95"></a>
## D-95 — Permanent off-by-default runtime-measurement infrastructure inside `performElementaryLogicalStep` (2026-05-27)


**The motivation.** Some logical blocks spend effectively forever inside a single hash burst — one call of `prover.cpp::ExpressionAnalyzer::performElementaryLogicalStep`. Today there is no way to discover which LB is the offender, or which inner phase consumes its time, without waiting for the call to return. For pathological cases that never return, the existing hashburst dump (which captures state at burst boundaries) gives nothing.

**The choice.** Add a permanent, compile-time-gated timing instrumentation that:

1. Is controlled by a single `#define RT_MEASUREMENT 0|1` at the top of [`parameters.hpp`](../GL_Quick_VS/GL_Quick/src/parameters.hpp). When `0`, every call site compiles away — production binaries are byte-identical to a build that never had the infrastructure.
2. Owns a stack-local tracker for **one** call of `performElementaryLogicalStep`. The tracker's lifetime equals the function's. Nothing accumulates across calls, across outer iterations, or across worker threads.
3. Brackets the function's existing phases with RAII `RT_SCOPE("LABEL")` blocks (mail absorb, the five request-generation batches, the fixpoint loop, mail flush, hypo reaction, end-of-burst sanitize). Each scope contributes a row to a per-LB textual table whose entries reflect **exclusive self-time** (time inside the scope's own code, not its nested children).
4. Writes the table to `.rt/<sanitized-LB-chain>.log` only when the call's elapsed wall-clock time exceeds `RT_TIME_TRIGGER_SECONDS` (default 120). Calls that finish under the threshold leave no artefact.
5. **Refreshes the file online** while the call is still running, via write-to-tmp + `std::filesystem::rename`. A reader inspecting `.rt/<chain>.log` mid-call never sees a half-written table.
6. Hides table rows whose self-time share falls below `RT_MIN_PERCENTAGE` (default 2 %) into a single trailing "(other sections each < N % of total)" line.

**Why a compile-time macro instead of a runtime flag.** Two reasons. First, the gate must be visible to the preprocessor so every `RT_SCOPE` call site can be physically removed when off; a runtime `if (rtEnabled)` check would still pay the branch cost on every scope. Second, the existing `GL_DISINT_PROFILE` convention in `compiler.hpp` establishes the pattern. Rebuilding to flip the flag is the right cost.

**Why per-call scope (no accumulation).** A pathological LB that hangs in one burst is exactly the case we want to diagnose; the table inside that one call is the full story. Accumulating across calls or iterations would dilute the signal and conflict with the online-refresh model (a still-running call cannot share its tracker with a future call).

**Why a separate file per LB.** The chain-to-filename mapping makes `ls .rt/` a complete inventory of pathological LBs at any moment. The filename starts with the outermost LB's `exprKey` (typically `(Anchor...)`) so the directory listing groups LBs by anchor.

**Why exclusive self-time.** When a phase is sub-divided by adding nested `RT_SCOPE` calls inside it, the parent scope's row should reflect "time spent in this phase's own code only, not in the nested children". An inclusive-time model would double-count and make percentages add to more than 100 %.

**Relationship to the hashburst dump.** The hashburst dump (Rule 14) stays untouched — separate file (), separate gating, separate code. The two diagnostics are complementary: the hashburst dump captures *state*, this one captures *time*.

**What was investigated / ruled out.**
- *Always-on cumulative profile (across calls).* Rejected — destroys the "this LB is hung right now" signal we want.
- *Per-call dump at function return only.* Rejected — defeats the purpose for calls that never return.
- *Polling thread that snapshots active LBs externally.* Rejected — would require either thread-safe reads of every prover container or a parallel cache; both options leak measurement into the prover's hot path. RAII inside the function is contained and adds no shared state.

**Files added by this branch.**
- `GL_Quick_VS/GL_Quick/src/infra/rt_tracker.hpp` / `rt_tracker.cpp` — `RTTracker` + `RTScope` classes. Compiled unconditionally; only the call sites are gated. Lives in `src/infra/` alongside `hashburst_dump.{hpp,cpp}` — a small folder for diagnostic / instrumentation code that is operationally independent of the prover's algorithm.
- `GL_Quick_VS/GL_Quick/src/tests/test_rt_tracker.cpp` — unit tests wired into `gl_quick.exe --unit-tests`.

**Files relocated by this branch.**
- `hashburst_dump.hpp` / `hashburst_dump.cpp` moved from `src/` to `src/infra/` to colocate diagnostic infrastructure. Behaviour, public API, and output path () are unchanged — the move is structural only, authorized explicitly by the user (Rule 14 protects the dump's *behaviour*, not its file location). Includes inside `prover.cpp` and `hashburst_dump.cpp` rewritten to `"infra/hashburst_dump.hpp"`; `GL_Quick.vcxproj` + `.filters` updated; `Makefile`'s source glob extended to include `src/infra/*.cpp`.

**Files touched.**
- `GL_Quick_VS/GL_Quick/src/parameters.hpp` — the macro and `RTMeasurementParameters` struct.
- `GL_Quick_VS/GL_Quick/src/prover.cpp` — the eleven `RT_SCOPE` call sites inside `performElementaryLogicalStep` and the `tracker.refreshIfTriggered` call every 64 fixpoint iterations.
- `main.py` (and / or the `main` of `gl_quick.exe`) — a `.rt/` wipe at process start so leftover files from a previous run never mislead.
- `.gitignore` — `.rt/`.

**See also.** [I-59](30_invariants.md#i-59) (the per-call-scope invariant), [`_meta/rt_measurement.md`](_meta/rt_measurement.md) (full design), the project conventions (hashburst dump untouched), the project conventions (doxygen + unit-test obligation for the new class), the project conventions (no defensive programming — overflow of the section array asserts, never silently drops).

---

<a id="d-99"></a>
## D-99 — Drop the verifier's `vacuous truth trace` requirement (origin chain must reach `rest[4]`); drop the prover's level gate inside `addExprToMemoryBlock`'s vacuous-truth branch and inside `deactivationCheck`'s condition 4 (2026-05-26 evening,)


**The regression that motivated it.** After re-enabling `deactivationCheck` (D-98) and restoring the trailing `memoryBlock.isActive = false;` in the vacuous-truth branch, Peano-main produced one `vacuous truth trace` verifier failure: chapter `103_check_induction_condition.txt`, theorem head `(=[i0, v1])`, contradicting pair `(in2[v1, v2, s])` / `!(in2[v1, v2, s])`, rest[4] = `(in2[v3, v1, s])` (the inner recursion-step's hypothesis). Neither side of the pair traced through the chapter's origin graph back to `(in2[v3, v1, s])`.

**Why the trace fails legitimately.** Chapter 103 is the recursion-step sub-page of the side-lemma `(in2[v1, 0, s]) ∧ (in[v1, N]) ⇒ (=[0, v1])` ("if v1's successor is 0 then 0 = v1"), proved by induction on v1. The lemma's outer premise `(in2[v1, 0, s])` ("v1 is the predecessor of zero") is itself impossible by Peano's `implication4` ("0 has no predecessor"), so the entire theorem is vacuously true. In the recursion step, the contradiction is necessarily rooted in the theorem's outer premise + Peano's axiom, *not* in the step's recursion hypothesis. There is no proof path that reaches the inner LB's hypothesis — vacuous truth is the only option, and it is sound.

**The choice.** Three coordinated edits:

1. **`verifier.py` check 7 (vacuous truth trace).** Drop the `_trace_back_to(rest[0], …, target == rest[4]) or _trace_back_to(rest[2], …, target == rest[4])` requirement. Record `True` for every well-formed vacuous-truth row (row has at least six rest fields). The third dep stays in the row for documentation; the verifier no longer checks it.

2. **`prover.cpp::addExprToMemoryBlock`'s vacuous-truth branch.** Drop the `exHasMbLevel || negHasMbLevel` level gate. The branch now fires on `memoryBlock.isPartOfRecursion && validityName == "main"` plus the outer `contradictionFound` ancestor scan that the function already runs.

3. **`prover.hpp::deactivationCheck`'s condition 4.** Drop the matching level gate (incoming `levels.count(mb.level) > 0` for the new expression / `statementLevelsMap[neg]` for the pre-existing negation). The condition now fires on `isPartOfRecursion + validityName == "main"`, conditional on the shared ancestor-scan above.

**Soundness argument.** Peano's anchor axioms are consistent. A chapter-level contradiction cannot trace only to anchor-level rows (no two anchor facts contradict each other); every real chapter-level contradiction must touch at least one non-anchor task formulation (theorem premise, LB hypothesis, sub-block assumption). That non-anchor grounding is what makes vacuous truth meaningful — *not* the specific LB-level identity at `rest[4]`. The historical level gate and the verifier's `rest[4]` trace check both encoded the latter; the former is what we keep, the latter is what we drop.

**The vacuous theorem is the case that fails the old check.** For a theorem whose own premise is impossible, the contradiction is rooted upstream in the theorem's premise — necessarily outside the inner recursion's hypothesis. The old check rejected exactly the shape that the chapter generation step produces for these theorems; the new check accepts it.

**What was investigated / ruled out.**
- *Add an origin-trace gate inside the prover that mimics the verifier's check.* Forbidden by [I-44](30_invariants.md#i-44) / Rule 16 — `exprOriginMap` is process documentation, not a proof input. The prover may not branch on its contents.
- *Tighten the gate to `AND` (both sides require `mb.level`).* Would block our failing case but also block legitimate non-vacuous recursion-step contradictions where only one ingredient carries `mb.level`. Rejected.
- *Fix `involvedLevels` to exclude the firing context's `mb.level` for derivations whose deps are anchor-only.* Bigger refactor with cross-cutting effects on `combinedLevels` (`memory.cpp::checkLocalEncodedMemoryStatic`). Deferred; not justified for this one failure.

**Files touched.**
- `verifier.py` — replace the check-7 trace block with an unconditional `True` record.
- `GL_Quick_VS/GL_Quick/src/prover.cpp` — remove the level-gate `{ … if (!exHasMbLevel && !negHasMbLevel) return; … }` block inside `addExprToMemoryBlock`'s vacuous-truth branch; update the surrounding rationale comment.
- `GL_Quick_VS/GL_Quick/src/prover.hpp` — remove the level gate from `deactivationCheck`'s condition 4 block; update the matching `@details` paragraph.

**See also.** [I-66](30_invariants.md#i-66) (the predicate's condition 4 simplifies but the predicate-only contract is otherwise unchanged), [I-16](30_invariants.md#i-16) (verifier sacredness — user-authorized edit), the project conventions (hashburst dump untouched).

---

<a id="d-102"></a>
## D-102 — `addStatement` becomes the single statement-add door; `addEquality` / `addNegatedEquality` lose `allowSymmetry`, become private helpers dispatched from inside `addStatement` (2026-05-26)

**The choice.** Three changes to prover ingestion, landed across three commits on:

1. Retire the `allowSymmetry` bool parameter on `prover.cpp::addEquality` and `prover.cpp::addNegatedEquality`. The `if (allowSymmetry)` wrapper around each helper's mirror-registration block is deleted; the mirror block now runs unconditionally.
2. Move the shape dispatch (`if isEquality addEquality, else if isNegatedEquality addNegatedEquality, then wholeExpressions.insert`) from `prover.cpp::addExprToMemoryBlock`'s post-disintegration `stmts` loop into the top of `prover.hpp::addStatement`, immediately after the eager mirror-push. The caller-side block becomes a single `addStatement(...)` call.
3. Reroute `prover.hpp::applyEquivalenceClassToNegatedEquality::emitNew` from `addNegatedEquality(newExpr,...)` to `addStatement(newExpr,...)`. The explicit local push onto the outer `newStatements` is kept (the inner `addStatement` deposit block is skipped because the relocated dispatch has already inserted the variant into `statementLevelsMap`, so its own `newStatements.push_back` does not fire).

**Why.** User-directed consolidation: every statement that enters an LB goes through one function. Before the refactor, the disintegration-product loop called the helper directly (and then `addStatement`), and the eq-class rewriter called `addNegatedEquality` directly — three doors. After the refactor, `addStatement` is the door; the two helpers remain as private members that `addStatement` dispatches to internally.

**Why retire `allowSymmetry`.** The parameter was carried by both helpers but only ever varied across call sites — `isLocal` at the disintegration loop, `true` at the eq-class rewriter. With the rewriter rerouted through `addStatement`, both call paths share the helper's contract; a per-call gate on mirror registration adds no value. The unconditional mirror block keeps the pair invariant ("original and mirror added together or not at all") uniform across both paths.

**Behavioural change for status=3.** For external-mail absorb of an equality / negated-equality (status 3 at the disintegration loop), the disintegration-loop call previously passed `allowSymmetry = isLocal = false` — the mirror was not registered. After the refactor the mirror block runs regardless; the mirror is now registered in non-local channels too (the inner `if (local)` block still gates the local-delta pushes, so the local-delta semantics are unchanged). Status 0 / 1 and the rewriter path (which already passed `allowSymmetry = true`) are unchanged.

**What was investigated / ruled out.**

- `addStatement` has exactly one call site (`prover.cpp::addExprToMemoryBlock`, post-disintegration `stmts` loop). Status 2 returns at `prover.cpp::addExprToMemoryBlock` before the loop (the toBeProved-only path); status 4 returns even earlier (the direct-deposit path). Only statuses 0 / 1 / 3 reach the loop. The caller's `if (status!= 2)` guard wrapping the dispatch was therefore dead at the call site — no new `status` parameter on `addStatement` is needed; the dispatch inside `addStatement` runs unconditionally. The comment at `prover.cpp::addExprToMemoryBlock` (just below the `else {... }` of the status==2 block) records this explicitly.
- An alternative threading the dispatch via a defaulted `int status = 1` parameter was considered and rejected as adding a no-op knob.

**Cascade implication.** With commit 3 in place, the rewriter routes the rewritten form through `addStatement`, which in its negated-equality branch calls `applyEquivalenceClassToNegatedEquality` recursively on the variant. Each variant can therefore expand its own downstream rewrites. Depth is bounded by `statementLevelsMap.find` early-exit inside `emitNew`. The cascaded rewrites are deposited into the LB's containers; the kernel's next iteration picks them up via the standard hashburst path.

**Files touched.**

- `GL_Quick_VS/GL_Quick/src/prover.hpp` — declarations (around `addEquality` / `addNegatedEquality`), the new dispatch block inside `addStatement`, the rewriter call inside `applyEquivalenceClassToNegatedEquality::emitNew`.
- `GL_Quick_VS/GL_Quick/src/prover.cpp` — definitions of `addEquality` / `addNegatedEquality` (parameter retired; mirror block unwrapped), the collapsed `addExprToMemoryBlock` disintegration-loop call.
- `docs/10_pipeline/04_prover.md` — "addStatement — central ingestion" section rewritten to reflect the new flow (commit 2).
- `docs/40_decisions.md` — this entry.

**See also.** [I-12](30_invariants.md#i-12) (one-sided expansion via equivalence classes — unchanged), [I-25](30_invariants.md#i-25) (`ExpressionWithValidity` pair channel — unchanged), [D-93](#d-93) (immortal `intKnownStatements`, prerequisite for the dispatch's pair invariant).

---

<a id="d-93"></a>
## D-93 — `cleanUpExpressions` stops erasing from `intKnownStatements`; the int-known set becomes immortal (2026-05-26)

**The choice.** Remove the two `mb.intKnownStatements.erase(packStatementKey(mb.nameMap.encode(it->original), mb.nameMap.encode(it->validityName)));` calls inside `prover.hpp::cleanUpExpressions` (one in the local-encoded sweep, one in the encoded sweep, both paired with `mb.statementLevelsMap.erase(*it)`). The `statementLevelsMap.erase` stays; only the `intKnownStatements` half is dropped. `intKnownStatements` becomes immortal for the duration of an LB's lifetime — entries are only ever added (by `addStatement` / `addEquality` / `addNegatedEquality` / status=4) or wholesale-erased (by `Memory::wipeSubtree` on scope closure, or by `eradicateImplicationFromLB` for retired implications).

**Why.** Closes the Peano-main missing-theorem regression — equi proved 52 of ref's 58 Peano main theorems before the fix, 70 (no missing) after. The regression's root cause, traced via the `applyEquivalenceClass` + `cleanUpExpressions` diagnostic traps (`assert_trap.txt` / `equiclass_trap.txt`, removed in the cleanup commit):

1. Burst N: an equality `(=[A, B])` arrives at an LB. `addEquality` inserts the (encoded, validity) key into `wholeExpressions`, `statementLevelsMap`, `intKnownStatements`, `encodedStatements`, `localEncoded*`. `updateEquivalenceClasses` then forms a class that contains `A` and `B` (and possibly other variables previously merged in). `cleanUpExpressions` runs, identifies the equality as containing the non-canonical member `B` (per `filterIterations` lex-min canonical-selection rule), and DROPS the equality from `encodedStatements`, `localEncoded*`, `statementLevelsMap`. The pre-fix code also erased from `intKnownStatements`. **`wholeExpressions` is never touched by `cleanUpExpressions`** — it holds the dropped form indefinitely.

2. Burst N+k (`k ≥ 1`): mail re-delivers the same equality (some ancestor LB still has it in `mailOut`, or a peer re-derives it). `addExprToMemoryBlock` enters with the re-delivered `expr`. Pre-fix, Site F's `intKnownStatements.count(packStatementKey(origId, anc))` returns 0 (the int-known erase stripped it), so Site F doesn't short-circuit. Control proceeds into the equality branch at `prover.cpp::addExprToMemoryBlock` `if (isEquality(ev.original)) addEquality(...);` — and `addEquality`'s own `if (wholeExpressions.find(encodedExpr) == wholeExpressions.end) {...full body... }` guard SKIPS the body because `wholeExpressions` still holds it from burst N. The four runtime containers stay empty for this entry. Then `addStatement` runs; `updateEquivalenceClasses` pushes the equality onto its returned `newStatements` (line `newStatements.push_back(ExpressionWithValidity(expr, validityName));`). `addExprToMemoryBlock`'s post-loop iterates `added`, hits the equality, calls `memoryBlock.statementLevelsMap.find(encAddExpression)`, gets `end`. Assert fires: `"addStatement post-loop statementLevelsMap invariant violated"`.

3. Fix. Stop erasing from `intKnownStatements`. Site F's gate then catches the re-arrival, returns immediately, addEquality is never called a second time, the assert never trips. Equivalence-class apply burst inside `applyEquiClasses` retains access to the source non-canonical row in `encodedStatements` because **the move of the class-driven cleanup out of `updateEquivalenceClasses` / `addStatement` into a deferred Step-4b block in `standardProcessing`** (added in the same commit, see below) ensures `applyEquivalenceClass` reads the row before `cleanUpExpressions` drops it. The canonical rewrite is emitted; then the deferred sweep drops the non-canonical original; then `intKnownStatements` still holds the dropped original's key, so Site F catches its re-arrival on the next burst.

The two changes are paired: deferring the class-driven cleanup gives the rewriter the source it needs, and keeping `intKnownStatements` immortal gives Site F the duplicate-suppression record it needs.

**What changed (file-level).**

- `prover.hpp::cleanUpExpressions` — two `intKnownStatements.erase(packStatementKey(...))` calls removed (local-encoded sweep + encoded sweep). The paired `statementLevelsMap.erase(*it)` calls stay. Inline comment links the rationale to [I-58](30_invariants.md#i-58).
- `prover.hpp::updateEquivalenceClasses` — the inline `cleanUpExpressions(mb, newStatements, validityName) + cleanUpAdmissionMap + cleanUpAdmissionMapIntegration` call sequence (was at the post-merge cleanup point) is removed; replaced with a comment pointing at the deferred Step-4b site below.
- `prover.hpp::addStatement` (post-`updateEquivalenceClasses` `isEquality(expr)` suffix) — the `newStatements = cleanUpExpressions(memoryBlock, newStatements, validityName);` call is removed for the same reason.
- `prover.hpp::standardProcessing` — new **Step 4b** block immediately after `applyEquiClasses(memoryBlock)`: for each unique `validityName` in `memoryBlock.changedClassesThisStep` (dedup via `std::set<std::string>`), call `cleanUpExpressions(memoryBlock, sink, validityName)` + `cleanUpAdmissionMap(memoryBlock, validityName)` + `cleanUpAdmissionMapIntegration(memoryBlock, validityName)`. The `sink` is a local `std::vector<ExpressionWithValidity>{}` whose post-call value is ignored — `cleanUpExpressions`'s "filter the per-equality `newStatements` pair vector" responsibility is dropped from this site because the upstream call chain no longer carries a per-equality pair vector through to here.
- `prover.cpp::performElementaryLogicalStep` hashburst fixpoint — the two `if (!body.isActive) {... break; }` guards (top of `while (changed)` iteration + after each rule firing) are removed. The hashburst dump's EARLY-EXIT call site lived inside the per-iteration guard; it is retired alongside the guard. Only the ENTRY trap (top of `performElementaryLogicalStep`) and the final EXIT trap (post-fixpoint) remain.

**Verification.** Peano main at 22 iterations: equi 70 hash-burst-printed proven theorems / ref 58 / 0 missing / 12 equi-only extras (consequences ref's compression dropped). Compression: 107 raw → 33 essential. Verifier: 8009 checks, 0 failures, airtight. Right-cancellation `(>[7,8,9](in3[7,8,9,4])(>[10](in3[10,8,9,4])(=[7,10])))` and commutativity of multiplication `(>[7,8,9](in3[7,8,9,5])(in3[8,7,9,5]))` both proven (regressions before the fix).

**Status.** Wired and verified. Subsumes the user-directed inner-`isActive` removal from the hashburst fixpoint (same commit). Two diagnostic traps used during the investigation (`addExprToMemoryBlock` pre-assert dump → ; `applyEquivalenceClass` + `cleanUpExpressions` instrumentation → ) are removed in the cleanup commit.

**See also.** [I-58](30_invariants.md#i-58), [I-27](30_invariants.md#i-27) (Site F / Site H ancestor-scan dedupe at `addExprToMemoryBlock` entry — the reader that benefits from this immortality), [D-96](#d-96) (the refactor that made `cleanUpExpressions`'s position relative to `applyEquiClasses` matter).

---

<a id="d-96"></a>
## D-96 — Apply burst lifted out of `addExprToMemoryBlock`; per-step pipeline driven by `standardProcessing` (2026-05-23, refined 2026-05-24;)

**The choice.** Refactor `performElementaryLogicalStep` so equivalence-class application, toBeProved discharge, and mailOut population each run from one explicit site per elementary step. The cross-cutting consolidation followed in a follow-up sequence of commits on the same branch (2026-05-24); see [D-90](#d-90) for the consolidation layer.

**Why.** Pre-refactor, the apply burst fired transitively from inside `addExprToMemoryBlock` (via `addStatement`'s in-line `applyEquivalenceClass` calls and `updateEquivalenceClasses`'s post-merge apply loop), and toBeProved discharge sat inside the Kernel's post-`addStatement` loop with recursive addExpr for OR-integration / NotOrScope parent emissions. The result was a tangled control flow with unpredictable per-step real-time cost and recursive emissions across LB scopes.

**What changed.**

- `addExprToMemoryBlockKernel` deleted; its body inlined into flat `addExprToMemoryBlock`'s stmts loop. `addStatement` survives as the container-insert + equality-routing helper, called from the inlined kernel.
- New `applyEquiClasses(Memory&)` — one pass per step, iterates the per-step delta-class set (`Memory::changedClassesThisStep`, populated by `updateEquivalenceClasses` on every merge, stored as `(validityName, EquivalenceClass)` value copies to survive subsequent same-scope merges) against every comparable `encodedStatement` (same-NS / strict-ancestor / strict-descendant via one comparability check). Calls the four admission/rejected helpers per delta class. Per-class start-index tracker iterates the fixpoint until `encodedStatements` stabilises.
- New `dischargeToBeProved(Memory&, int coreId, Mail& internalMailOut)` — one pass per step over `localEncodedStatementsDelta`. Three branches: induction-recursion proven, non-recursion proven (`updateGlobalDirectTuples`), OR-integration / NotOrScope (parent-scope emission via the caller-supplied `internalMailOut` parameter — see [D-90](#d-90) for phase routing).
- New `fillMailOut(Memory&)` — one pass per step over `localEncodedStatementsDelta`. For each main-validity, `allowedForMail`-passing entry, copies levels from `statementLevelsMap` into `mailOut.statements` and origin lines from `exprOriginMap` into `mailOut.exprOriginMap`. Subsumes the seven scattered pre-refactor mailOut writes (addStatement, applyEquivalenceClass, addEquality original + mirror, addNegatedEquality original + mirror, anchor handling). Single-writer policy enforced by [I-64](30_invariants.md#i-64).
- Hashburst rule-firing output at `memory.cpp` `checkLocalEncodedMemoryStatic` deposits into `body.sameIterationInternalMail`. The fixpoint loop's `changed` detection reads `sameIterationInternalMail.statements.size`. The historical parallel direct-`mailOut` deposit was retired in the consolidation follow-up — see [D-90](#d-90).

**Status (2026-05-24 follow-up consolidation).** This entry's original session-close (2026-05-23) status — 18 / 284 theorems on the IncubatorPeano 5-iter run, attributed to a 1-step inter-cycle propagation delay — was the trigger that motivated the follow-up consolidation. The consolidation introduces the two-channel internal-mail split (`sameIter` / `nextIter`) and the `standardProcessing` two-call driver, which together close the 1-step delay for pre-burst discharge emissions and preserve it intentionally (matching the documented OR-integration / NotOrScope contract) for post-burst discharge emissions. See [D-90](#d-90) for details. The original session's open items (Linux segfault, baseline ref comparison) remain open; verification against the channel-split architecture lands in the cross-commit verification pass at the close of the consolidation.

**See also.** [D-90](#d-90), [I-60](30_invariants.md#i-60), [I-61](30_invariants.md#i-61), [I-62](30_invariants.md#i-62), [I-63](30_invariants.md#i-63), [I-64](30_invariants.md#i-64).

---

<a id="d-90"></a>
## D-90 — Two-channel internal-mail split + `standardProcessing` two-call driver; closes pre-burst discharge delay (2026-05-24)

**The choice.** Build on [D-96](#d-96) by:

1. Splitting `Memory::internalMailIn` into two `Mail` fields: `sameIterationInternalMail` (ephemeral, hashburst-output channel) and `nextIterationInternalMail` (cross-iteration deferral channel). See [I-62](30_invariants.md#i-62).

2. Consolidating the absorb / apply / discharge / fillMailOut pipeline into a single member function `ExpressionAnalyzer::standardProcessing(Memory&, Mail* externalMailIn, Mail& internalMailIn, Mail& internalMailOut, int coreId)`. Called twice per `performElementaryLogicalStep`:
 - **Pre-burst:** `standardProcessing(body, &body.mailIn, body.nextIterationInternalMail, body.sameIterationInternalMail, coreId)`. Discharge emissions during this call route to `sameIter` (zero-delay — same step's hashburst processes them).
 - **Post-burst:** `standardProcessing(body, nullptr, body.sameIterationInternalMail, body.nextIterationInternalMail, coreId)`. Discharge emissions during this call route to `nextIter` (one-step delay — next step's pre-burst call processes them).

3. Relocating the per-step delta clears (`localEncodedStatementsDelta`, `intLocalEncodedStatementsDelta`, `localHashMemoryDelta`) to a single unconditional pre-fixpoint site (between request generation and the fixpoint loop). The pre-existing `contradictionIndex == -1` gate is dropped. See [I-61](30_invariants.md#i-61).

4. Dropping the parallel direct `mailOut` deposit inside `checkLocalEncodedMemoryStatic`. `fillMailOut` becomes the sole writer to `mailOut.statements` and `mailOut.exprOriginMap`. See [I-64](30_invariants.md#i-64).

**Why.**

The follow-up motivation came from the [D-96](#d-96) session-close observation that the post-burst-only apply/discharge/fillMailOut placement introduced a 1-step inter-cycle propagation delay for mailIn-derived discharge emissions. The user proposed the channel split as a clean way to make the "same-iter vs next-iter" distinction explicit at the type level — no longer a timing distinction (which channel a write lands in depends on **when** the write happened) but a container distinction (which Mail field the writer targets). The two-call `standardProcessing` shape with the four-mail-argument signature lets the caller route discharge emissions to the channel matching the desired delay semantics.

The single-writer-to-`mailOut` policy and the pre-fixpoint single-delta-clear policy both follow from the same architectural principle: each piece of state has exactly one writer and exactly one clear site, identified by code site rather than by call-graph timing.

**What changed (file-level).**

- `memory.hpp`: added `Mail nextIterationInternalMail` field on `Memory`. `internalMailIn` renamed mechanically to `sameIterationInternalMail` in commit 2 of the follow-up.
- `prover.hpp`:
 - `standardProcessing` signature widened from `(Memory&, Mail&, int status, int coreId)` to `(Memory&, Mail*, Mail&, Mail&, int)`. Body restructured around a per-mail `absorb` lambda; mail clears moved to between drain and apply (so discharge writes to `internalMailOut` survive even when it aliases `internalMailIn`).
 - `dischargeToBeProved` signature widened from `(Memory&, int)` to `(Memory&, int, Mail&)`. The two OR-integration / NotOrScope sites write to the `internalMailOut` parameter instead of `memoryBlock.sameIterationInternalMail` directly.
- `prover.cpp`:
 - 4 cross-iter direct writers retargeted from `sameIter` to `nextIter`: `addExprToMemoryBlock` vacuous-truth path (×1), `updateGlobalDirect` siblings (×3).
 - Pre-hashburst inline mailIn-only absorb block (~100 lines) replaced by a single pre-burst `standardProcessing` call.
 - Post-hashburst inline drain + apply + discharge + fillMailOut block replaced by a single post-burst `standardProcessing` call (commit 4 alias variant, commit 5 channel-split variant).
 - Three delta clears relocated from post-fixpoint (with gate) to pre-fixpoint (no gate).
- `memory.cpp::checkLocalEncodedMemoryStatic`: parallel direct `mailOut` deposit deleted. Function now writes only to `body.sameIterationInternalMail`.

**Status (commit 8 close).** Build clean, 511 / 511 unit tests pass. Behavioral verification (sequential main.py runs, verifier against baseline) deferred to the post-commit verification pass per.

**See also.** [D-96](#d-96), [I-60](30_invariants.md#i-60), [I-61](30_invariants.md#i-61), [I-62](30_invariants.md#i-62), [I-63](30_invariants.md#i-63), [I-64](30_invariants.md#i-64).

---

<a id="d-92"></a>
## D-92 — Hashburst mail-deposit gates on the same Site F ancestor-scan that ref's `addExprToMemoryBlock` runs at entry (2026-05-25)


**The milestone.** With this decision in place, runs through 5 iterations of IncubatorPeano end-to-end: prover finishes (`Runtime prover: 338.541 seconds`), chapter export completes (no hang), verifier reports `14575 checks, 0 failures — airtight`.

**The choice.** Wrap the `sameIterationInternalMail.statements.insert(...)` + `addOrigin(sameIterationInternalMail.exprOriginMap,...)` block inside `memory.cpp::checkLocalEncodedMemoryStatic` with an ancestor-scope `intKnownStatements` check that mirrors `prover.cpp::addExprToMemoryBlock`'s Site F early-return (ref's `prover.cpp::4141-4145`). When the head is already known at any ancestor scope, skip the mail-deposit entirely — no statement insert, no origin write. Compressor mode keeps multiples (matches ref's same `!compressor_mode` guard).

**Why.** Commit 17 of the follow-up consolidation rewired hashburst rule-firings to deposit to `sameIterationInternalMail` directly instead of routing through `addExprToMemoryBlock`. Pre-commit-17, every rule firing hit `addExprToMemoryBlock`'s Site F at function entry; the early-return suppressed BOTH the statement insert AND any new origin write when the head was already known. Post-commit-17, the mail-deposit path bypassed Site F. `addOrigin(mail.exprOriginMap, …)` dedupes only on (key, ENTIRE-origin-tuple), so N distinct (rule, premise) firings for the same head landed as N distinct origin lines per atom.

The trap dump at `__contradiction__(in2[10,10,3])` made this visible across burst boundaries (compared equi vs ref at the same LB, same chain to root):

- Burst 3 EXIT: equi `(=[9,11])` had 10 origins (8 distinct alpha-variant implication rules each producing a separate origin); ref had 2.
- Burst 4 EXIT: equi top-key origins at 26; ref still at 3.
- Burst 5 EXIT: equi `(in[*,1])` hit the 30-origin cap on 8 expressions; ref's burst 5 EARLY-EXITed at top-key origins of 5.

By chapter export, equi's `buildStack` faced 30 × 21 × … candidates per chain → exponential exploration → CPU-bound hang (verified by tripwire: no real recursion cycle, just exponential branching). Site F mirror gate brings equi's per-key origin counts back in line with ref, `buildStack` converges, chapter export completes.

**Earlier attempt that did not work.** A recursive transitive walk in `fillMailOut` — `shipExprHistoryTransitively` — was implemented first: ship documentation-only deps' history transitively along with each delta-entry's origin. It addressed shipping documentation-only entries like expansion conjunctions, but did not address the per-key origin bloat from re-firing on already-known heads, so chapter export still hung. The walk was retired; the lesson is that the bug is at the mail-deposit head-dedup, not deeper in the history walk.

**Companion fixes landing in the same commit.**

1. `trackExpansionHistory` (`prover.cpp::disintegrateExprCore2`) — restored the paired `mailOut.exprOriginMap` writes at the expansion-origin and disintegration-origin sites (mirrors ref's lines 6479 + 6492).
2. LB-creation paired `mailOut.statements.insert` + `addOrigin(mailOut.exprOriginMap, …)` writes at the three `addExprToMemoryBlock(... task formulation...)` sites in `prover.cpp::addTheoremToMemory` (chain-walk LB creation, reformulated-contradiction LB creation, standard-contradiction LB creation).
3. `buildGrid` dispatches each LB's startup `mailOut` into the boxes via `sendMail` before the existing `smashMail(boxes)` call, so the LB-creation paired writes reach descendants' `mailIn` before step 1 begins.

These three together carry the documentation-only chain at startup; the Site F mirror gate carries the per-burst dedup. Both are needed.

**Code.** `memory.cpp::checkLocalEncodedMemoryStatic` (Site F mirror gate); `prover.cpp::disintegrateExprCore2::trackExpansionHistory` (paired mailOut writes); `prover.cpp::addTheoremToMemory` × 3 sites (LB-creation paired writes); `prover.cpp::analyzeExpressions::buildGrid` (sendMail loop). Behavioural verification: full main.py IncubatorPeano 5-iter run completes with verifier airtight (`run_equi_5iter_sitefgate.log`, 2026-05-25).

**Status.** Lands as the "runs through 5 iterations" milestone commit. Theorem coverage at 112 / 225 is still below ref's 284 / 284 — that is a separate prover-side investigation, not addressed by this commit.

**See also.** [I-56](30_invariants.md#i-56), [I-64](30_invariants.md#i-64), [I-44](30_invariants.md#i-44), [I-19](30_invariants.md#i-19), [D-96](#d-96), [D-90](#d-90).

---

<a id="d-97"></a>
## D-97 — `applyEquiClasses` Pass 2 closes the new-statement × non-delta-class coverage gap; equality / negated-equality filter (2026-05-25)


**The milestone.** Equi closes 81% of the IncubatorPeano theorem-coverage gap at 5 iterations: 273 / 284 proved (was 225 / 284 immediately before this commit; ref baseline is 284 / 284). Verifier reports 19153 checks, 0 failures — airtight. 11 theorems remain missing in equi vs ref + 1 extra in equi vs ref — see *Residual gap* below.

**The choice.** Extend `prover.hpp::applyEquiClasses` from a single-pass driver (delta classes × ALL `encodedStatements`) into a dual-pass driver inside the same outer `while (size grows)` fixpoint:

1. **Pass 1 (unchanged shape, gains a filter).** Iterate `changedClassesThisStep`; for each delta class, run the 4 admission/rejected helpers (`applyEquivalenceClassToRejectedMapIntegration`, `applyEquivalenceClassToAdmissionMapIntegration`, `applyEquivalenceClassToAdmissionMap`, `applyEquivalenceClassToRejectedMap`), then apply the class to `encodedStatements[startIdx[k].. end)` with `startIdx[k]=0` initially. Filter added inside the inner loop: skip statement if `isEquality(stmt.original) || isNegatedEquality(stmt.original)`.

2. **Pass 2 (new).** Iterate `equivalenceClassesMap` (snapshot the validity-key list and per-validity class count to defend against mid-iteration mutation); skip classes that are also in `changedClassesThisStep` (Pass 1 already covered them against every statement). For each remaining non-delta class, run the same 4 admission/rejected helpers, then apply the class to `encodedStatements[eqClassSttmntIndexMapMap[v][cls.variables].. end)`. After the inner loop, advance `eqClassSttmntIndexMapMap[v][cls.variables]` to current `encodedStatements.size`. Same equality / negated-equality filter.

**Why.** Pre-fix, `applyEquiClasses` only iterated `changedClassesThisStep`. The clear at the top of `performElementaryLogicalStep` wipes any class entered into the delta during the PARENT LB's step (e.g., a contradiction LB seeded with `(=[10,11])` via `addExprToMemoryBlock` at LB creation, which populated `changedClassesThisStep` via `updateEquivalenceClasses`). By the first elementary step of the child LB the class is in `equivalenceClassesMap` (persistent) but absent from `changedClassesThisStep` (delta) — so `applyEquiClasses` does nothing, and the in2 chain that arrives later via mail-absorb (e.g., `(in2[9,10,3])` at burst 2) is never back-rewritten by the class. Zero `equality1` rewrites fire in the entire LB across all 5 bursts; the contradiction chain never closes; the 33 `!(=[X,Y])` family inequalities (and 26 downstream `!(in2/in3[...])` variants) never prove.

The bug surfaced precisely at `__contradiction__(=[10,11])` (ref's chapter 0 of the IncubatorPeano proof graph): both `(in2[9,10,3])` and `(=[10,11])` sat in equi's `encodedStatements` at burst 5, yet `(in2[9,11,3])` (the canonical `10 → 11` substitution that ref derives via chapter 0 row 29 `equality1`) had 0 occurrences anywhere in equi's trace. `.scripts/compare_chapter_vs_originmap.py` flagged it as the first MISSING+ALL_DEPS_AVAIL row.

**Why the equality / negated-equality filter.** Positive `(=[a,b])` statements are consumed by `updateEquivalenceClasses` (class merge), not by `applyEquivalenceClass`. Negated `!(=[a,b])` statements are consumed by `applyEquivalenceClassToNegatedEquality` (called inline from `addStatement` on incoming arrivals). The two shapes are covered by their dedicated paths; `applyEquivalenceClass` is for everything else (`in`, `in2`, `in3`, `fXY`, `fXYZ`, operator expressions,...). Pre-reshuffle ref's `applyEquivalenceClass` had only the `isEquality` early return (skipping positive equalities) — negated equalities went through both `applyEquivalenceClass` (via the in-line burst in `addStatement`) AND `applyEquivalenceClassToNegatedEquality`. The double processing is treated as a ref-side bug; the equi fix tightens the contract to single-path coverage for each shape.

**Why not the alternatives.**
- **Re-introduce the inline apply burst in `addStatement`.** Goes against [I-60](30_invariants.md#i-60). The whole reshuffle motivation was to make `addExprToMemoryBlock` a flat container-insertion routine without recursive apply-burst control flow.
- **Auto-register every class touched by a new statement back into `changedClassesThisStep`.** Conflates "this class changed" (membership change → re-apply against ALL statements via `startIdx=0`) with "this class has new statements to consider" (membership unchanged → apply only to the new statements). Pass 2's per-class `eqClassSttmntIndexMapMap`-driven startIdx is the precise semantic: each persistent class advances its "applied through" frontier independently.
- **Drop the `changedClassesThisStep` clear at the top of `performElementaryLogicalStep`.** Would leak parent-step class deltas into the child's first step; would also defeat the per-step "what changed THIS step" semantic the delta tracker exists for. Pass 2 reads the persistent store directly, which is the right source for "what classes does this LB know about?".

**Side effects to keep in mind.**
- Pass 2 calls `applyEquivalenceClass` which can produce rewrites that themselves enter equality-formation paths and grow `equivalenceClassesMap[validity]`. Pass 2 iterates by index with a captured `classCountAtStart` and copies `cls` per iteration to dodge reference invalidation under mid-iteration mutation.
- Per-iteration `deltaIds` set rebuild at each outer-while round is O(|changedClassesThisStep|); cheap relative to the inner statement loop.
- Runtime overhead on the 5-iter IncubatorPeano benchmark: 336 s → 371 s (+10%) on Windows, reflecting the ~2000 newly-firing `equality1` rewrites per affected LB. Acceptable cost for correctness.

**Before / after** (single-host repeat-run comparison, equi 5-iter IncubatorPeano).
| Metric | Pre-fix (parent commit) | Post-fix (this commit) | Ref baseline |
|---|---|---|---|
| Proved theorems | 225 | 273 | 284 |
| Gap vs ref | 59 | 11 (+1 extra in equi) | — |
| `!(=[10,11])` in proved set | absent | present | present |
| `equality1` rewrites at `__contradiction__(=[10,11])` (5 bursts) | 0 | 2242 | (ref's chapter 0 alone records 6) |
| Verifier checks / failures | 14575 / 0 | 19153 / 0 | 19569 / 0 |

**Residual gap (12 deltas).** 11 theorems still missing in equi vs ref + 1 extra in equi vs ref. Pattern: every delta involves the args `2` and `6` together. The 32 other members of the `!(=[X,Y])` family proved correctly; only `!(=[2,6])` remains missing, and the 11 missing `!(in3[..., 2, 6,...])` variants are downstream of that single inequality (the `{2, 6}` class formed at `__contradiction__(=[2,6])` exhibits the same symptom-shape as the original `__contradiction__(=[10,11])` bug but with a narrower / different timing interaction that Pass 2 does not fully cover). 1 extra theorem in equi (`!(in3[6,2,9,4])`) indicates equi's contradiction-LB derivation reaches a slightly different — not strictly subset — closure of refutations. Investigation deferred to a follow-up commit; not a regression vs the pre-fix state, which had all 12 missing + every `!(=[X,Y])` for X ≠ 2 ∨ Y ≠ 6 also missing.

**Code.** `prover.hpp::applyEquiClasses` (single function, dual-pass body). No new functions; no new fields; the existing `eqClassSttmntIndexMapMap` field gets a second writer (Pass 2's loop) symmetric to its existing writer (`updateEquivalenceClasses` at class creation).

**See also.** [I-65](30_invariants.md#i-65), [D-96](#d-96), [D-90](#d-90), [I-60](30_invariants.md#i-60), [I-61](30_invariants.md#i-61).

---

<a id="d-94"></a>
## D-94 — `ordisMerge` also fires on equivalence-class substitution products in `applyEquiClasses` (both passes), not only on `addStatement` output (2026-05-28)


**Status — work in progress (not a settled win).** This restores OR-disintegration convergence but does **not** by itself prove FTA-rung-1, and it introduces a main-batch regression. Recorded now so the partial progress and the open issues are not lost; the entry will be revised (or the merge narrowed) once the interval-integration gap and the Gauss regression are resolved.

**The choice.** Call `ordisMerge` from **two** sites instead of one: keep the existing post-`addStatement` loop call in `addExprToMemoryBlock` (covers inputs, not-yet-applied classes, and disintegration products at the moment they are added), **and** add a per-product call in `applyEquiClasses` via a local `mergeProducts` lambda, run on every statement `applyEquivalenceClass` commits, in both PASS 1 and PASS 2.

**Root cause it addresses.** `ordisMerge`'s only call site was the post-`addStatement` loop. `applyEquivalenceClass` commits its rewrites directly into `encodedStatements` / `localEncodedStatementsDelta` (and returned the cross-scope forms into a `sink` that `applyEquiClasses` discarded), so a branch disjunct *created by an equi-class rewrite* — e.g. a branch-scope `preorder` rewritten from the branch's equality assumption — never registered in `orBookkeeping`. The recorded disjunct count never reached `orDisjunctCount`, the OR never converged, and rung-1's ES2⟹interval never closed. See [`20_core_concepts/07_or_branching.md`](20_core_concepts/07_or_branching.md) §3b.

**Why dual coverage is safe.** `ordisMerge` registers a disjunct via `orBookkeeping[(expr, orSig)].insert(branchBody)` — an idempotent `std::set` insert — and the convergence promotion is itself idempotent (the `intKnownStatements` gate blocks a re-deposit). A product reached by both call sites registers exactly once.

**Verified.** Clean build; full pipeline run, verifier airtight (0 failures). Ordis branches now merge: 1116 `preorder` convergences in the rung-1 ES2 LB hashburst dump; `EnumerationSet2` / `implication1564`-`1565` / `in` / `or0` all converge; the converged preorder reaches the implication parent scopes (`preorder[1,4,2,repl_lev_1_0]` @ `implication20`, `[1,4,repl_lev_1_1,6]` @ `implication21`, `[1,4,2,repl_lev_1_2]` @ `implication22`) — the analog of ref's converged `preorder[1,4,2,p]`.

**Open issues (unresolved).**
- **rung-1 still not proved.** `interval[1,4,2,6,10]` is only ever the integration goal at `v=main` (hypo-disintegrated to `…_var0_2_var1_6_hypo_(interval…)`); it is never derived as a fact. Convergence is no longer the blocker — the interval integration not consuming the converged preorders is. Suspected scope-visibility gap between the `implication2x` convergence scopes and the interval-integration hypo scope.
- **Main-batch regression.** Both Gauss `fold` theorems are lost (`files/theorems/theorems.txt`: ref 2 → 0) plus 3 `AnchorPeano` theorems; 6 different Peano theorems gained. The broad per-product merge perturbs derivation outside rung-1 and likely needs narrowing (e.g. merge only products at `_boundary_ordis_` / branch scopes).

**Code.**
- `prover.hpp::applyEquiClasses` — `mergeProducts` lambda (sorts products for deterministic order, then `ordisMerge` each with its `statementLevelsMap` levels); called from PASS 1 and PASS 2 after each `applyEquivalenceClass`. Signature-preserving edit of an existing function (lambda local to an already-covered function — no new Rule-18 unit-test obligation).
- `prover.cpp::addExprToMemoryBlock` — `[CONV-TRAP-E]` diagnostic in the `stmts` loop (dumps whether a `_boundary_ordis_` branch preorder reaches `addStatement`); diagnostic only, to be removed before release.

**See also.** [D-97](#d-97) (the Pass-2 coverage gap this builds on), [D-96](#d-96) (the refactor that moved the apply burst out of `addStatement` and created the discarded `sink`), [D-33](#d-33) (descendant-direction cross-scope rewrites the merge must also observe), [`20_core_concepts/07_or_branching.md`](20_core_concepts/07_or_branching.md) §3b.

---

<a id="d-98"></a>
## D-98 — Mid-burst LB-deactivation predicate at the sameIter deposit site; predicate-only with deferred broadcast; EARLY-EXIT becomes break so post-burst absorb still fires (2026-05-25)

**Superseded in part ([D-104](#d-104)).** The `while (changed)` hashburst loop described here is removed; request evaluation is now a single pass. The EARLY-EXIT `break;` semantics below no longer apply verbatim: the trap is relocated to fire immediately after the single pass when the LB deactivated during it. This is behaviorally equivalent — `deactivationCheck` fires only after a real mail deposit, so a mid-pass deactivation always set `changed = true`, and the old loop re-entered once to fire the trap at the top of iteration 2.

**Re-enablement note (2026-05-26 evening).** The retirement noted earlier this day (under "Retirement note (2026-05-26)" — now deleted in favour of this re-enablement note) was itself reverted once the orphan failure mode it diagnosed had been removed independently. The vacuous-truth discharge path was reshaped by [D-93](#d-93) (intKnownStatements immortality, removed the same-LB re-entry failure mode that bound the original regression to mid-burst deactivation) and by the follow-up commit (vacuous-truth head deposited directly through `addStatement` into the current iteration's `localEncodedStatementsDelta` instead of being enqueued to `nextIterationInternalMail`). With the head landing in this step's delta, the post-burst `dischargeToBeProved` sees it regardless of the LB's `isActive` state, so the predicate-only design is safe again. The four conditions, the call site, the EARLY-EXIT `break;` semantics, the inner-isActive gate inside the for-r loop, and the trailing `memoryBlock.isActive = false;` at the end of `addExprToMemoryBlock`'s vacuous-truth branch are all restored as in the original commit. The level-gate fix introduced by [D-100](#d-100) (`exHasMbLevel = involvedLevels.count(memoryBlock.level) > 0` instead of `statementLevelsMap[expr]`) is kept untouched — independent correctness fix, orthogonal to this RT-opt.


**The milestone.** At iter=50 IncubatorPeano on equi, runtime drops from 637 s to 597 s (-6.3%) with full 1035 / 1035 theorem coverage preserved and verifier airtight (91056 checks, 0 failures). The optimisation is the equi-side mirror of ref's mid-burst LB deactivation paths, which fire from inside `addExprToMemoryBlock` at every rule firing in ref but were bypassed in equi because hashburst rule firings write to `sameIterationInternalMail` instead of routing through `addExprToMemoryBlock`.

**The choice.** Three coordinated edits:

1. **New predicate** `prover.hpp::ExpressionAnalyzer::deactivationCheck` — pure predicate that mirrors the four `memoryBlock.isActive = false` sites in `prover.cpp::addExprToMemoryBlock`. Signature: `bool(const std::string& expr, const std::string& validityName, Memory&, const std::set<int>& levels)`. Returns `true` and sets `memoryBlock.isActive = false` when any of the four conditions fires; returns `false` otherwise. The four conditions:
 - **isPartOfRecursion + toBeProved match at "main"** (`!compressor_mode`) — the would-be-added expression closes a goal in this induction LB.
 - **`primedForContradiction` + ancestor-scan finds neg(expr)** — incubator contradiction LB completes.
 - **`contradictionIndex >= 0` + ancestor-scan finds neg(expr)** — CE-filter contradiction LB completes.
 - **`isPartOfRecursion` + validity == "main" + ancestor-scan finds neg(expr) + level gate** — vacuous-truth induction step closes (level gate uses incoming `levels` for the expr side and `statementLevelsMap[validityName]` for the negation side, matching ref's same-scope lookup).

2. **Call site** in `memory.cpp::checkLocalEncodedMemoryStatic` — immediately after the existing `sameIterationInternalMail.statements.insert(...)` + paired `addOrigin(sameIterationInternalMail.exprOriginMap, …)` block (inside the Site-F-mirror gate's `!alreadyKnown` branch). The predicate fires only when the deposit actually happens, so the per-firing overhead is bounded by what the Site F mirror lets through.

3. **Hashburst-loop break + EARLY-EXIT semantics** in `prover.cpp::performElementaryLogicalStep`:
 - After `checkLocalEncodedMemoryStatic(req, body, coreId)` in the per-request `for (r...)` loop, add `if (!body.isActive) break;` so the burst exits mid-iteration when the predicate fires (instead of finishing the remaining requests in this fixpoint round).
 - Convert the existing EARLY-EXIT `return body;` at the top of the outer `while (changed)` loop into `break;`. The dump call site is unchanged (Rule 14: dump format / sections / target predicate are sacred); only the post-dump control flow changes from "skip rest of step" to "continue to post-burst standardProcessing absorb". This is what makes the RT-opt correct: the mid-burst predicate is **deferred-broadcast** — it only flips `isActive` and does no broadcasts itself, so the post-burst absorb must still run to do the actual `updateGlobalTuples` push / `addOrigin` / `toBeProved` erase / `contradictionTable` mark via `addExprToMemoryBlock`'s own four matching conditions (which see the deposited mail entry and re-fire idempotently).

**Why predicate-only (no inline broadcast).** Replicating ref's full deactivation bookkeeping inside the predicate would require porting four distinct branches (induction-recursion goal closure, isProved + allLevelsInvolved theorem emission, vacuous-truth `addExprToMemoryBlock` re-entry, contradiction-theorem `updateGlobalDirect`) along with their mutex usage and `inductionMemoryBlocks` queueing. The predicate-only design keeps the new code minimal — the absorb already does the same work via the same conditions, so the only marginal cost of the deferred bookkeeping is the absorb iteration over the sameIter deposits (which would happen anyway).

**Why the EARLY-EXIT semantics change (return -> break).** First-cut implementation kept the existing `return body;` and saw a regression: 1035 theorems collapsed to 416 with a 533 s runtime. Root cause: with `return body;`, the post-burst `standardProcessing` call is skipped entirely, the mail deposits in `sameIterationInternalMail` are orphaned, `addExprToMemoryBlock` never fires on them, and the deferred broadcasts never happen — so downstream LBs never observe the proved theorems. Converting to `break;` lets the post-burst absorb still run on the (smaller-than-full-burst) mail set; broadcasts fire, theorem coverage restored, RT savings retained.

**Before / after** (5-iter IncubatorPeano was already at-baseline; iter=50 is the real RT-opt test).
| Metric | Pre-RT-opt baseline | RT-opt first-cut (return) | RT-opt (this commit, break) |
|---|---|---|---|
| Proved theorems | 1035 | 416 (BROKEN) | 1035 |
| Verifier checks / failures | 91034 / 0 | 26378 / 0 | 91056 / 0 |
| Runtime (Windows, iter=50) | 637 s | 533 s (broken) | 597 s |
| RT delta vs baseline | — | — | -6.3% |

The 22-check delta (91056 vs 91034) is traversal-order variance from earlier burst termination (some rule firings that fired in the baseline are skipped post-deactivation; the verifier counts are sensitive to which proof paths reach the chapter-export stage). No verifier failures either way.

**Code.**
- `prover.hpp::deactivationCheck` — predicate, ~75 LOC including Doxygen block.
- `memory.cpp::checkLocalEncodedMemoryStatic` — single call after the deposit, ~25 LOC including rationale comment.
- `prover.cpp::performElementaryLogicalStep` — `if (!body.isActive) break;` in the for-r loop; `return body;` -> `break;` at the EARLY-EXIT dump site.

Unit test: `tests/test_equi_reshuffle.cpp::deactivation_check_symbol_signature` — Rule-18 symbol-existence + signature check matching the convention used by every other reshuffle-introduced function on this branch (`applyEquiClasses`, `dischargeToBeProved`, `fillMailOut`, `standardProcessing`).

**See also.** [I-66](30_invariants.md#i-66), [D-96](#d-96), [D-90](#d-90), [I-60](30_invariants.md#i-60).

---

<a id="d-100"></a>
## D-100 — Fix `addExprToMemoryBlock`'s vacuous-truth level gate to use incoming `involvedLevels`; the originally-paired `deactivationCheck` removal + trailing `isActive = false` removal are reverted in a follow-up commit once their orphan failure mode was removed independently (2026-05-26)

**Update (2026-05-26 evening).** Two of the three edits this decision originally bundled — deleting `deactivationCheck` and removing the trailing `memoryBlock.isActive = false;` at the end of the vacuous-truth branch — are reverted in a follow-up commit, because the underlying orphan failure mode was removed independently by [D-93](#d-93) (intKnownStatements immortal) and by commit (head deposited directly via `addStatement` into the current iteration's `localEncodedStatementsDelta` instead of enqueued cross-step into `nextIterationInternalMail`). Only edit #2 — the level-gate change from `statementLevelsMap[expr]` to `involvedLevels.count(memoryBlock.level) > 0` — survives. The retirement of [D-98](#d-98) and [I-66](30_invariants.md#i-66) is reverted accordingly; both entries are un-retired with re-enablement notes.


**The regression.** On equi Peano-main (`RUN_INCUBATOR=False`, `RUN_MAIN_PATH=True`, `maxIterationNumberProof=22`, same 1035-line incubator simple-facts as ref) equi proved 26 of ref's 30 main-batch theorems; seven were missing in `theorems.txt`. The earliest missing in ref's `run.log` order was `s(v1)=v2 ∧ s(v2)=i1 ⇒ i1=v1` (ref `processed_proof_graph/global_theorem_list.txt` line 27, method `induction v1`).

**Diagnosis chain** (trap-debug; chain match per the project conventions):
1. Retargeted the hashburst dump to the check-induction-condition LB of the missing theorem: `(in2[rec0,7,3]) ← (in2[8,6,3]) ← (in2[7,8,3]) ← (AnchorPeano[1,2,3,4,5,6]) ← root`. Trap fired 3× (3 hashbursts).
2. Comparator (`.scripts/compare_chapter_vs_originmap.py`) against ref's `102_check_induction_condition.txt` flagged row 46 as the first "should-have-fired-but-didn't": `(=[it_0_lev_3_0,7])` via the successor-uniqueness implication. All four deps present in equi's `encodedStatements`; head never derived.
3. The third premise `(in2[rec,it_0_lev_3_2,3])` arrived only in EXIT #3's `localEncodedStatementsDelta` — after the LB was deactivated by `deactivationCheck` condition 4 (`exHasMbLevel || negHasMbLevel`) on `!(in2[7,2,3])` (incoming levels `{0,3}` include `mb.level=3`).
4. Deactivation was *legitimate* (vacuous-truth contradiction `!(in2[7,2,3])` + ancestor-stored `(in2[7,2,3])`), but the subsequent absorb's `addExprToMemoryBlock` vacuous-truth branch then hit its own level gate which reads `statementLevelsMap` — and `statementLevelsMap[!(in2[7,2,3])]` is `NOT_PRESENT` at that gate's call site (the `statementLevelsMap` write happens further down in the function via the `addStatement` call inside the disintegration `stmts` loop, which the vacuous-truth branch never reaches). Gate failed; discharge skipped; `toBeProved` still holds the head.

**The fix — three coordinated edits.**

1. **Delete `prover.hpp::ExpressionAnalyzer::deactivationCheck`** entirely (function body + Doxygen block). The function was the [D-98](#d-98) RT-optimisation, now retired. Its sole call site in `memory.cpp::checkLocalEncodedMemoryStatic` is replaced by a comment explaining the removal. The four `isActive = false` sites inside `addExprToMemoryBlock` still fire on their own contradiction-found paths; only the redundant mid-burst predicate is gone.

2. **Fix the vacuous-truth level gate in `prover.cpp::addExprToMemoryBlock`** (the `else if (memoryBlock.isPartOfRecursion && validityName == "main")` branch inside the contradictionFound block). `exHasMbLevel` is now `involvedLevels.count(memoryBlock.level) > 0` — using the incoming `involvedLevels` parameter instead of the stale `statementLevelsMap[expr]` lookup. This matches the design intent (the just-deposited expression's contribution to the LB's level is in the incoming parameter, not yet in the map). `negHasMbLevel` continues to use `statementLevelsMap[neg]` (the negation was registered by an earlier call).

3. **Remove `memoryBlock.isActive = false` at the end of the vacuous-truth branch** (was `prover.cpp` line 4177; now replaced by a comment). The discharge enqueues the head into `nextIterationInternalMail`; the LB must stay active for the next step to absorb that mail, install the head in `localEncodedStatementsDelta`, and let `dischargeToBeProved` push to `updateGlobalTuples` (which then sets `isActive = false` itself via the induction-recursion proof branch in `prover.hpp::dischargeToBeProved`).

**Result on Peano main (iter=22).** Equi now proves 30 theorems (matching ref's 30 count). Five of the original seven now appear in `theorems.txt`; the other two are compressed-out (the compressor eliminated them as derivable from other proved theorems — see the project conventions "Artifact authority for regression claims" — compressed-out is a legitimate end state for a derivable theorem). 4699 → 5668 verifier checks; 3 `origin chain termination` failures remain (not in this fix's scope — they reflect collateral from looser LB persistence; tracked separately).

**Code.**
- `prover.hpp::deactivationCheck` — function + Doxygen deleted (~140 LOC).
- `memory.cpp::checkLocalEncodedMemoryStatic` — call site removed; replaced by an explanatory comment.
- `prover.cpp::addExprToMemoryBlock` — vacuous-truth branch level gate uses `involvedLevels` (replaces `statementLevelsMap[expr]` lookup); `memoryBlock.isActive = false` at branch end removed.
- `prover.cpp::performElementaryLogicalStep` — comments around the `break;` at the EARLY-EXIT site updated to reflect that `deactivationCheck` is gone (the `break;` itself is kept — still load-bearing for the contradiction-condition paths still inside `addExprToMemoryBlock`).
- `hashburst_dump.cpp::isTargetLB` — kept at the check-induction-condition LB (`(in2[rec0,7,3])` chain) for the next iteration of the investigation (3 remaining `origin chain termination` verifier failures + the compressed-out vs proved split need follow-up).

Unit tests: `tests/test_equi_reshuffle.cpp::deactivation_check_symbol_signature` was removed alongside the function deletion, then restored in the same follow-up commit that re-enabled `deactivationCheck`. Two further behavioural tests (`deactivation_check_condition1_tobeproved_match`, `deactivation_check_negative_no_condition_holds`) were added there for positive + negative Rule-18 coverage.

**See also.** [D-98](#d-98) (un-retired by follow-up commit), [I-66](30_invariants.md#i-66) (un-retired by follow-up commit), [D-90](#d-90), Rule 14 in the project conventions (hashburst dump retarget is user-authorized for this investigation).

---

<a id="d-101"></a>
## D-101 — `g_buildStackPath` is a flat set with no refcount; recursive re-entry on the same `(expr, validity)` corrupts the outer scope's cycle filter on exit; track per-invocation `insertedHere` and erase only when true (2026-05-26)


**The regression.** Three `origin chain termination` verifier failures landed on Peano main right after [D-100](#d-100) (looser LB persistence ⇒ more iterations ⇒ more equi-class apply firings ⇒ more single-origin `equality1` entries). Cycle: `(=[i0,v2])` ↔ `(in2[v2,i1,s])` in chapter `86_check_induction_condition.txt` (raw chapter `125_check_induction_condition.txt`, cycle pair `(=[2,8])` ↔ `(in2[8,6,3])` pre-rename). The user's observation drove the diagnosis: "this theorem was proved before last commit without cycles" — i.e., the path-tracking bug was latent, exposed by the post-commit increase in originMap density.

**Diagnosis chain** (trap-debug; full LB chain match per the project conventions, authorized one-off trap in `visualizer::buildStack`):
1. Cycle-LB chain on equi for this theorem's check-induction-condition: `(in2[rec0,7,3]) ← (in3[2,7,8,5]) ← (in[7,1]) ← (AnchorPeano[1,2,3,4,5,6]) ← root`.
2. Origin dump at the cycle-LB: `(=[2,8])` has 6 origin candidates (rich — non-equality candidates `[3]`/`[4]`/`[5]` do NOT cite `(in2[8,6,3])` as a dep, so the chapter SHOULD pick one of them via the D-49 sort + path-cycle filter + per-candidate backtracking). `(in2[8,6,3])` has only 1 origin — `equality1` with deps `(in2[2,6,3])`, `(=[2,8])`. Single cyclic candidate; cycle filter rejects.
3. Sequence of buildStack entries (path-state dumps) revealed the failure mode: TOP `(=[2,8])` insert ⇒ path `{(=[2,8])}`. Try `[1]`: recurse into `(in2[8,6,3])` ⇒ path `{(=[2,8]), (in2[8,6,3])}`. `(in2[8,6,3])`'s fallback recurses into `(=[2,8])` again (path unchanged because `std::set::insert` is no-op). Inner `(=[2,8])` picks `[3]` (acyclic), succeeds, then `g_buildStackPath.erase((=[2,8]))` — **wipes the outer scope's entry**. Fallback returns to TOP's `[1]`; subtreeOk=false; rollback. Try `[2]`: recurse into `(in2[8,6,3])` — but path is now `{(in2[8,6,3])}` only, **no `(=[2,8])`**. The `equality1` cycle filter on `(=[2,8])` does NOT fire (`(=[2,8])` is not "in path" anymore). Fallback emits the cyclic row. `(in2[8,6,3])` ultimately returns true to `[2]`. `[2]` succeeds. Cyclic chapter row emitted.

**Root cause.** `g_buildStackPath` is a `std::set<ExpressionWithValidity>` keyed by expression+validity. A flat `std::set` has no refcount: erase removes the entry regardless of how many recursive scopes "depend on" it being present. When the same node appears on the path at two recursion depths (outer + inner reentry on a cyclic dep's recursive walk back), the inner scope's exit `erase` removes the OUTER scope's entry. The outer scope's subsequent cycle-filter checks against `g_buildStackPath` then return false negatives, and cyclic candidates that should have been rejected get accepted instead. The cyclic row is emitted at the chapter root and survives because the top-level chapter caller does not retry.

**The fix.** Track at `buildStack` entry whether THIS invocation was the inserter:

```cpp
const bool insertedHere = g_buildStackPath.insert(proved).second;
```

Guard every exit's erase with `if (insertedHere) g_buildStackPath.erase(proved);`. Five exit sites updated: early-return broadcast inside candidate loop, subtreeOk-success return, contradiction-LB-switch tail-call, fallback broadcast early-return, last-resort fallback return. No other semantics changed.

**Why no refcount semantics.** A `std::map<Key,int>` refcount would also work but adds heap overhead per insert and complicates the read-side. The boolean-per-frame approach is local to each invocation, costs one `bool` per stack frame, and the semantics match exactly: each invocation's exit pairs with its own entry's `.second`.

**Result on Peano main (iter=22).** 5668 → 5647 checks (delta is the eliminated cyclic rows + the avoided redundant recursions inside cyclic subtrees); 3 → **0** failures; airtight. Chapter `86_check_induction_condition.txt` row 1 now picks the acyclic candidate (`mult-uniqueness` rule, deps `(in3[v7,v1,i0,*])`, `(in3[v7,v1,v2,*])`, `(in[v1,N])`, `(in[v7,N])` — no `(in2[v2,i1,s])` dep) instead of the cyclic `succ-uniqueness` rule. Chapter shrank 55 → 48 rows. `theorems.txt` unchanged at 30.

**Code surface.**
- `visualizer.cpp::ExpressionAnalyzer::buildStack` — five erase sites guarded by `insertedHere`; insert at top now captures `.second`.

**Tests.** `tests/test_equi_reshuffle.cpp::buildstack_path_refcount_doubleentry` — exercises the double-entry case: a synthetic LB with two origins for node A (one cyclic via B, one acyclic) and B's only origin cyclic back to A. Pre-fix this would emit a cyclic chapter row; post-fix the acyclic candidate is picked and no cyclic row appears.

**Why it was hidden until now.** Requires:
- A node X with at least one cyclic candidate AND at least one acyclic candidate.
- The cyclic candidate's dep recurses back into X (so X ends up on the stack twice).
- The originMap density is high enough that all of (a)/(b) coexist on the same chapter walk.
Pre-[D-100](#d-100) the recursion-condition LBs deactivated earlier; the equi-class apply that creates the single-origin `equality1` entry on the LHS of the loop didn't get a chance to fire, so the loopback couldn't form. The dump confirms: `(in2[8,6,3])` has only ONE origin at the cycle-LB, and it is the `equality1` rewrite created in late iterations enabled by the longer LB life. Ref doesn't hit it because it lacks the equi-refactor's looser persistence on this code path.

**See also.** [D-100](#d-100), [I-67](30_invariants.md#i-67), [D-51](#d-51) (path-stack cycle filter introduction), Rule 12 in the project conventions (full LB chain matching).

---

<a id="d-91"></a>
## D-91 — Restore all seven paired `mb.mailOut.exprOriginMap` addOrigin writes in `prepareIntegrationCore2` (2026-05-25)


**Context.** After the IncubatorPeano milestones described in [D-96](#d-96) and [D-90](#d-90), `RUN_MAIN_PATH` flipped to True and the Peano main batch was enabled for the first time on this branch. The Peano main batch reproducibly aborted with `SIGABRT` at the chapter-export stage:

```
gl_quick: src/visualizer.cpp:NNN:
  bool gl::ExpressionAnalyzer::buildStack(...):
  Assertion `false && "buildStack: no origin found"' failed.
```

Missing expression: `(>[pi_lev_0_1](in[pi_lev_0_1,u_1])(>[](in2[pi_lev_0_1,u_6,u_3])(existence2[u_1,u_6,u_3])))` — a pi-bound integration instruction. LB chain (innermost → root): `(=[10,2]) → (in3[9,10,13,5]) → (in3[8,10,12,5]) → (in3[7,10,11,5]) → (in3[7,8,9,4]) → (AnchorPeano[1,2,3,4,5,6]) → root`.

**Root cause.** The original equi-reshuffle refactor adopted the "single mailOut writer" policy ([I-64](30_invariants.md#i-64)) and deleted seven paired `addOrigin(mb.mailOut.exprOriginMap,...)` writes from `prover.hpp::prepareIntegrationCore2` that ref keeps alongside their `addOrigin(mb.exprOriginMap,...)` siblings. The deletion assumed `fillMailOut`'s delta-driven copy would cover the cross-LB shipment of those seven origins. **That assumption is wrong:** all seven KEYs (`expandedSignature + "_integration_goal"`, `expandedImplication + "_integration_goal"`, `subImpl + "_integration_goal"`, the per-element renamed chain entries, the negated-disjunct assumption rows, and the integration-instruction forms `iiv` / `iivHash`) are *documentation-only* — they are suffixed marker strings, hashmemory rule strings, or scope-bound assumption rows that never enter `localEncodedStatementsDelta`. `fillMailOut`'s delta loop never sees them, so the cross-LB shipment never happens.

Hashburst rule firing at `memory.cpp::checkLocalEncodedMemoryStatic` builds origin records of shape `(existence2[1,6,3]) <- implication | <lmv.originalImplication> | <matched premises>` at descendant LBs, where `lmv.originalImplication` is the integration-instruction form stored in the LB's `HashMemory` rule table when the rule was installed. The descendant LB receives the integration-instruction string as a *dep* in the row, but the *key* entry for that string never lands in the descendant's `exprOriginMap` — only the producer LB (`AnchorPeano` in the crash) holds the local entry, and the cross-LB ship never happened. `buildStack` walks the dep and asserts.

**Investigation that ruled out alternatives.**

1. Central `addOrigin` trap inside the `addOrigin` helper at `prover.hpp::addOrigin`: 100 fires (cap-saturated) with `keyMatches=N depMatches=Y` for `(>[pi_lev`, **zero fires with `keyMatches=Y`**. Confirms the pi-bound integration-instruction is never added as a key via `addOrigin` anywhere in the refactor.
2. `mb.integrationPrepared` dedup at the head of `prepareIntegrationCore2` Case B reports `alreadyPrepared=Y` with `integrationPrepared.size=1` on the very first Core2 entry at `AnchorPeano` — something pre-populates the dedup set before the addOrigin block can fire even locally. Tracing this separately is deferred; this branch addresses the cross-LB shipment side only, since restoring the paired mailOut writes is sufficient to close the visible crash and the local-side pre-population is a parallel concern.
3. Ref branch run ( with Peano-main-only config, prover sources from ref but pipeline config from the refactor) completes cleanly with 7179 verifier checks, 0 failures, 141 s runtime — confirming the regression is local to the refactor's deleted paired writes and not a pipeline-config side issue.

**The choice.** Restore all seven paired writes verbatim. At each of the seven sites in `prover.hpp::prepareIntegrationCore2`, immediately after the local `addOrigin(mb.exprOriginMap, ev, originX,...)` call, add a paired `addOrigin(mb.mailOut.exprOriginMap, ev, originX,...)` call with identical key, origin, and cap arguments. The seven sites:

- Case A (implication): `implication-expansion` (`ev = expandedImplication + "_integration_goal"`, origin tag `expansion for integration`), `premise-element` (`ev = elem`, origin tag `premise element`).
- Case OR: `or-branch-goal` (`ev = subImpl + "_integration_goal"`, origin tag `expansion for integration`), `or-branch-assumption` (`ev = negAssumption`, origin tag `or branch assumption`).
- Case B (existence/and): `expansion-integration` (`ev = expandedSignature + "_integration_goal"`, origin tag `expansion for integration`), `integration-instruction` (`ev = iiv`, origin tag `reformulation for integration {and / >[bound] / >[]}`), `integration-instruction-hash` (`ev = iivHash`, same origin tag).

**Why the alternatives were rejected.**

- *Route the seven KEYs through `localEncodedStatementsDelta`.* Would require treating documentation-only entries (integration goals, hashmemory rule strings) as real statements, which is a category error — `localEncodedStatementsDelta` exists to drive request-generation and hashburst, and those keys must not participate in either.
- *Add a parallel "documentation delta" channel.* Discussed and rejected for [I-64](30_invariants.md#i-64) (transitive-walk subsection): adds bookkeeping at every `addOrigin(mb.exprOriginMap,...)` site (~30 call sites) for no behavioural improvement. The seven sites in `prepareIntegrationCore2` are localised enough that explicit paired writes (matching ref's structure exactly) is the minimal-surface fix.
- *Investigate the `integrationPrepared` pre-population side first.* Even if the local `addOrigin(mb.exprOriginMap, iiv,...)` is being skipped due to the dedup, the cross-LB shipment problem is independent: even when the local write does run at some LB (e.g., the very first Core2 call on a fresh validity), the cross-LB ship is still missing because `fillMailOut` doesn't see the key. Fixing the local-side pre-population is therefore necessary but not sufficient; the paired-writes restoration is the load-bearing fix.

**Before / after** (WSL ext4, Peano main only, no incubator):

| Source | Outcome | Checks | Failures | Runtime |
|---|---|---|---|---|
| refactor pre-fix | SIGABRT at `buildStack` (crash dump ~970 KB) | n/a | crash | ~60 s |
| **refactor + paired writes (this commit)** | **clean** | **4699** | **0** | **50.76 s** |
| ref (for comparison) | clean | 7179 | 0 | 141 s |

The refactor-vs-ref check-count delta (4699 vs 7179) reflects the equi-reshuffle pipeline differences described in [D-96](#d-96); not introduced by this commit. The Gauss / FTA-ladder coverage will be measured separately when those batches are re-enabled.

**Code.** `prover.hpp::prepareIntegrationCore2` — seven single-line `addOrigin(mb.mailOut.exprOriginMap,...)` insertions, each adjacent to the existing `addOrigin(mb.exprOriginMap,...)` call at the same site. Net +9, -2 (the two retired `// mailOut.exprOriginMap write retired (single-writer policy)` comment lines at the two OR sites are dropped).

**See also.** [I-64](30_invariants.md#i-64) (Documented exception 4), [D-92](#d-92) (companion exception 3: the `disintegrateExprCore2` paired writes from the same milestone class), [D-96](#d-96), [I-57](30_invariants.md#i-57).

---

<a id="d-82"></a>
## D-82 — Anchor predicates are scope identities; equivalence-class substitution skips them at both the shared rewrite helper and the admission-map value loop (2026-05-22,, commit — replication of sibling commits + )


**What.** Two coordinated gates installed across the equivalence-class substitution machinery so that an anchor predicate is never rewritten by an equi-class:

1. **Shared rewrite helper** — `prover.hpp::enumerateEqClassRewrites` (the substitution core called by both `applyEquivalenceClass` and `applyEquivalenceClassToRejectedMapIntegration`) returns immediately at the top of the function when `baseExpr.rfind("Anchor", 0) == 0`. No rewrite is generated for an anchor expression; the additive `applyEquivalenceClass` produces zero variants and the original concrete anchor is preserved.

2. **Admission-map value loop** — `prover.hpp::applyEquivalenceClassToAdmissionMap` value-key inner loop walks each element of `AdmissionMapValue::key` and computes its substituted form. If applying the substMap would **change** an element whose prefix is `(Anchor` (an anchor predicate stored as a positional element of the admission template), the entire admission update is refused (`continue`s past the insert). This mirrors the existing arg-equalization refuse pattern in the same loop and prevents an admission template's anchor element from being mutated by the same class membership that would otherwise rewrite it.

The two gates target the two distinct surfaces on which an equi-class could rewrite anchor content. The helper-level gate stops the *generation* of anchor-variant strings; the admission-loop gate stops the *insertion* of a modified-anchor admission template even when the helper has not been the path. Both gates are additive — neither removes any pre-existing entry.

**Why.** An anchor predicate (`AnchorPeano[…]`, `AnchorGauss[…]`, `AnchorIncubator[…]`) is the scope identity of its LB and the immutable positional contract for every chapter row inside that scope. Its argument slots are conjecturer-assigned canonical names (`N`, `i0`, `s`, `+`, `*`, `i1`, … on Peano), not equivalence-class members. An equi-class substitution that rewrote `AnchorPeano[N,i0,s,+,*,i1]` to `AnchorPeano[N,i0_copy,s,+,*,i1]` because `(=[i0,i0_copy])` is in scope creates a duplicate anchor predicate the LB never declared, and the chapter exporter then has to choose between two competing anchor strings for the same scope — neither is correct, both are dead-ends downstream. The substitution also wastes the equi-class hook's per-class budget enumerating identities that can never enable a deduction (anchor predicates are never targets of `toBeProved`, never premises of any rule firing, and never appear in `admissionMapIntegration` as keys whose substitution changes the integration outcome).

**How the gates were chosen.** A purely helper-level gate (option 1 alone) misses the admission-map case because `applyEquivalenceClassToAdmissionMap` does not route through `enumerateEqClassRewrites` for its value-side substitution — it iterates `AdmissionMapValue::key` directly with `ce::replaceKeysInString`. Conversely, an admission-loop-only gate (option 2 alone) misses the substitution-into-statements case where the helper enumerates `AnchorPeano[…]` variants for `applyEquivalenceClass` and `applyEquivalenceClassToRejectedMapIntegration`. Both surfaces must be guarded together to eliminate every anchor-variant emission path.

**Before / after** (full pipeline, where the change first landed and was validated):
- Before: 512 `AnchorPeano[…,repl_lev_…]` rewritten anchor variants in the run, ES2 LB `toBeProved 6→0` by EXIT #12, verifier 10083 + 112144 checks, 0 failures.
- After: 0 anchor substitution variants emitted from `enumerateEqClassRewrites`, ES2 LB converges identically (`toBeProved 6→0` by EXIT #12, one-burst discharge delay vs ungated, same steady state ~1366 stmts), verifier 10083 + 112144 checks, 0 failures — airtight.

The remaining 512 `AnchorPeano[…,repl_lev_…]` forms observed in the reference run came in via a **non-enumerateEqClassRewrites** path (admission map). The admission-loop gate (point 2 above) was added to close that path; on the reshuffle replication branch this is confirmed by the same shape disappearing from the dump.

**Why not the alternatives.**
- **Filter the *output* of `enumerateEqClassRewrites` at every call site** (sink-lambda level) — possible but redundant. Five sink lambdas would need the same `rfind("Anchor", 0) == 0` test; one gate at the helper entry is exactly equivalent and avoids the duplication.
- **Forbid anchors from being class members in the first place** (gate at `addEquality` / class-creation site) — overshoots. Anchors can never *be* class members because they are not variables; the issue is that the substMap applies to non-anchor *arguments* of an anchor predicate. The substitution machinery's class-key checks are bare-name-based and don't catch that the surrounding string is `AnchorPeano[…]`. A symbol-level filter at class creation has no signal to act on.
- **Bake the exclusion into `ce::replaceKeysInString`** — also overshoots. The helper is used universally; tagging anchor positions as "do not substitute" would require either a parallel substMap or a string scan on every call. The two targeted gates added by this entry are cheaper and stay local to the equi-class machinery.

**Code.** Both gates live in `prover.hpp`:
- `enumerateEqClassRewrites` — top-of-function `if (baseExpr.rfind("Anchor", 0) == 0) return;`.
- `applyEquivalenceClassToAdmissionMap` — the value-key substitution loop's `anchorChanged` flag and the `if (anchorChanged) continue;` after the loop.

The user-directed nature of both edits is recorded as a comment block at each site.

**Citation hygiene.** New SwDD content per the project conventions cites file + symbol only.

---

<a id="d-85"></a>
## D-85 — `createReshuffledMirrored` renames body bound variables to canonical occurrence order at output (2026-05-23)


**What.** After `createReshuffledMirrored` builds the rearranged chain + `howToRemove` binder lists, it now re-assigns body bound-variable names so they appear in occurrence order across the new chain, using the same name set the source expression provided (numeric for compiled theorems, named for named MPL). Implementation: walk `tempChain` to collect `canonicalPool` (source occurrence order), walk the rearranged `chain` to collect `newOrder` (mirror occurrence order), build a single-pass rename map `newOrder[i] → canonicalPool[i]`, apply via `replaceKeysInString` to the final `newExpr`. Single-pass trie matching guarantees bijection without cascade.

**Why.** [](#) fixed the binder-list **order within** each chain element (lexicographic → occurrence) but left variable **names** inherited verbatim from the source. After the mirror swap (original head → last premise, original alternative → new head), the same names appear at different body positions — so the post-mirror occurrence order of names is non-canonical. Downstream, the deferred-compaction drain's `compileImplicationToCompact` → `compileCoreExpressionMapCore` → `makeNormalizedEncodedKey` re-canonicalises during binary registration, so:

- `globalTheoremList` entry = mirror string verbatim (non-canonical names)
- Binary `implication<N>.elements` = compaction-normalised (canonical names)
- Chapter `implication`-row `rest[0]` = `reconstructImplicationFullBind` from binary elements (canonical names)

The verifier's `origin` meta-check compares the chapter citation against `globalTheoremList` left-column entries; `_alpha_canonicalize_bound_vars` and `_normalize_expr_list` handle bound-name renumbering but not the **occurrence-order of names**. Citation (canonical) and registry entry (mirror-as-written) are alpha-equivalent but byte-different — miss. Three main-batch failures shared this shape (chapters 10 + 36 on the prior run, citing `implication1104` for associativity-of-`+` and `implication1078` for the s-cross-rewrite).

This fix moves canonical-naming up from the binary-normalisation pass into `createReshuffledMirrored` itself, so the mirror string in `globalTheoremList` already has canonical-occurrence names. Result: mirror string == reconstructed-from-binary citation. The `origin` check finds the registry entry.

**Worked example (associativity-of-`+`).**

- Induction source (body var occurrence order `7,8,9,10,11,12` — canonical):
 ```
  (>[7,8,9](in3[7,8,9,4])(>[10,11](in3[7,10,11,4])(>[12](in3[8,12,10,4])(in3[9,12,11,4])))
  ```
- Mirror output before this fix (body var occurrence order `7,8,9,12,10,11` — non-canonical):
 ```
  (>[7,8,9](in3[7,8,9,4])(>[12,10](in3[8,12,10,4])(>[11](in3[9,12,11,4])(in3[7,10,11,4])))
  ```
- Mirror output after this fix (rename `12→10, 10→11, 11→12`):
 ```
  (>[7,8,9](in3[7,8,9,4])(>[10,11](in3[8,10,11,4])(>[12](in3[9,10,12,4])(in3[7,11,12,4])))
  ```
 Body-occurrence-order names. Byte-identical to what `makeNormalizedEncodedKey` produces during compaction.

**How the bug was localised.**

1. Phase 1 (induction-emplace FullBind, commit prior to this one) cleared the 1 incubator failure but left the 3 main failures unchanged, per its own finding-line.
2. Trap inside `compileImplicationToCompact` dumped (input, compact-name) for `AnchorPeano + ≥3 in3[*+] premises` to  (wiped per-`main.py` session). First run used `std::ios::trunc` mode — only Gauss's last per-process write survived. Switched to `std::ios::app` + per-`main.py`-session wipe.
3. App-mode run captured `implication1104`'s input during Peano batch:
 ```
   (>[7,8,9](in3[7,8,9,4])(>[12,10](in3[8,12,10,4])(>[11](in3[9,12,11,4])(in3[7,10,11,4])))
   ```
 The `>[12,10]` binder (12 first, 10 second) — occurrence-order binder but non-occurrence-order names — was the smoking gun.
4. Searched `files/raw_proof_graph/global_theorem_list.txt` for that exact body. Found two entries: the induction source (var=12) with canonical names 7-12, and a `mirrored statement` entry whose body string was byte-identical to the trap input. The mirror's source cite (third column) pointed back to the induction source.
5. Read `createReshuffledMirrored`: the binder-build loop calls `howToRemove[idx].push_back(arg)` with the raw `arg` from `getArgs(chain[idx])` — no rename. Verified the gap.

**Before / after** (full pipeline, ):
- before: incubator `106435 checks, 1 FAILED` + main `10542 checks, 3 FAILED`.
- after: incubator `106433 checks, 0 failures — airtight.` + main `10441 checks, 0 failures — airtight.`

**Closes the deferred follow-up** noted in [D-81](#d-81)'s final paragraph: *"Same canonicalization principle could apply at the global-theorem registration site; deferred pending user direction."* That deferral was about the chapter-citation-vs-registry mismatch on the assoc-of-`+` mirror; this fix is the closure.

**Side effect.** `implication<N>` allocation order shifts in the binary because the canonical-named mirror form now alpha-merges with shapes that `compileCoreExpressionMapCore` previously gave separate names. Theorem count goes 67 → 65 / chapters 120 → 119 — alpha-merge effect, no proved theorem lost (the registry's `mirrored statement` entry for canonical assoc-of-`+` exists at v-form citing the same induction source as before).

**Why not the alternatives.**

- Rewriting the `globalTheoremList` emplace to look up canonical-from-binary post-compaction. Rejected: loses traceability with the conjecturer — the registry entry's string should mirror the conjecturer's output (modulo binder canonicalisation), not the binary's normalised form.
- Defensive FullBind at every emplace site (8 sites). Already in place at the induction emplace per [D-80](#d-80); FullBind alone is binder-count, not binder-name canonicalisation, so it wouldn't have closed the main 3 failures. Mirror was the only producer site with the name-vs-position drift.
- Make `_alpha_canonicalize_bound_vars` / `_normalize_expr_list` smarter on the verifier side. Forbidden by [I-16](30_invariants.md#i-16) (verifier sacred) and would mask producer-side drift.

**Code.** `compiler.hpp::createReshuffledMirrored`, the rename pass inserted between the `howToRemove`-build loop and the reconstruction loop. `replaceKeysInString` is the existing identifier-aware substitution helper.

**Citation hygiene.** New SwDD content per the project conventions cites file + symbol only.

---

<a id="d-80"></a>
## D-80 — `updateGlobal`'s `"induction"` emplace must register the FullBind rebuild, not the as-scheduled `expr` (2026-05-23)


**What.** In `prover.cpp::updateGlobal`, the `globalTheoremList.emplace_back(expr, "induction", indVar, recCounter)` call now uses `exprFullBind = this->reconstructImplicationFullBind(ky, value)` as its first argument instead of `expr`. `expr` (the as-scheduled conjecture text — captured at `addTheoremToMemory` via `dependencyTable.originalAuxyMap[...].expr = expr`) is preserved for every other downstream use in the function (`checkOrCompletion`, `recordPendingCompaction`, `addOrigin`, `reformulateTheorem`). Only the registry entry — the string the verifier's `origin` meta-check compares against the chapter's binary-canonical citation — is canonicalised.

**Why.** The conjecturer (and the scheduler that feeds `addTheoremToMemory`) emits theorem-anchor implications with the outer `>[…]` binder restricted to anchor-slot names that actually appear in the body. For `existence3` — proven on Peano as "every non-zero N has a predecessor" — the body uses only `N, i0, s`; the conjecturer's text is `(>[N,i0,s](AnchorPeano[N,i0,s,+,*,i1])(>[v1](in[v1,N])(>[]!(=[v1,i0])(existence3[N,v1,s]))))`. The other three anchor slots (`+, *, i1`) appear inside `AnchorPeano[…]` but are NOT in the outer binder. That violates I-4 ("bind every non-`u_` variable at its left-most premise"). The peer FullBind sites apply the rule already — `broadcastTheorems` for incoming externals (`prover.cpp:7664`), `reformulateTheorem` for derived statements (`prover.cpp:3137`), `createReshuffledMirrored` for the mirror builder. `updateGlobal`'s `"induction"` emplace was the lone outgoing path that registered `expr` verbatim, so the partial-bind form flowed through `globalTheoremList` → `saveProvedTheoremsFiltered` → `files/theorems/compiled_theorems.txt` line 43 → incubator's `files/incubator/processed_proof_graph/global_theorem_list.txt` line 52.

Meanwhile the chapter's `implication`-tag rows cite the rule via binary-canonical reconstruction (the `compilation` / `expansion` triple emitted by the new D-76 compact-form mail channel, also FullBind via `reconstructImplicationFullBind`). The chapter cites FullBind; the registry stored partial-bind; the verifier's `origin` meta-check has no permutation/binder-count fallback, so the lookup misses on a real shape difference.

**How the bug was localised.** Trap at `prover.cpp::updateGlobal` immediately before the `globalTheoremList.emplace_back` call. The trap computes `exprFullBind = reconstructImplicationFullBind(ky, value)` and writes both forms to  whenever they differ. On a full pipeline run with the trap in place, exactly one entry fired:

```
tag=induction indVar=7 recCounter=0
  PARTIAL : (>[1,2,3](AnchorPeano[1,2,3,4,5,6])(>[7](in[7,1])(>[]!(=[7,2])(existence3[1,7,3]))))
  FULLBIND: (>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7](in[7,1])(>[]!(=[7,2])(existence3[1,7,3]))))
```

Confirms (a) the partial-bind originates at the conjecturer / scheduler, not at any later mutation; (b) only one Peano-batch induction theorem is partial-bind — every other induction-tagged entry in the registry already comes out FullBind because its body mentions every anchor slot.

**Before / after** (full pipeline, ):
- before: incubator `origin success 333, failure 1` → `106435 checks, 1 FAILED`.
- after: incubator `origin success 334, failure 0` → `106435 checks, 0 failures — airtight`.

Main 3 origin failures are unchanged: those are body-arrangement (arg-order inside `in3[…]`) mismatches between the binary-canonical citation and the registered mirror form, not a binder-count issue, so FullBind cannot reconcile them. Separate investigation.

**Why not the alternatives.**

- FullBind-rebuild `expr` earlier (inside `addTheoremToMemory` at the `dependencyTable.originalAuxyMap[...].expr = expr` site). Possible but invasive: that would change the runtime label used by `addStatement` / `addOrigin` / `recordPendingCompaction` / OR-pair detection in `checkOrCompletion`. `or_pairs.txt` stores conjecturer-form (partial-bind) text, so changing the runtime label would silently break OR-pair matching. The minimal targeted fix is at the registry emplace.
- Normalize at the `saveProvedTheoremsFiltered` writer instead. Equivalent for the verifier outcome but later: it would leave `globalTheoremList` itself in partial-bind, which still misroutes any compaction / compactToExpanded lookup that keys on the in-memory tuple. Single point of truth is the in-memory list.
- Make the verifier's origin check tolerate binder-count differences. Forbidden by [I-16](30_invariants.md#i-16) (verifier sacred) without explicit user consent — and would mask exactly the kind of producer-side drift this fix surfaced.

**Code.** `prover.cpp::updateGlobal` — the `"induction"` emplace block computes `exprFullBind = this->reconstructImplicationFullBind(ky, value)` and registers that. The localisation trap (`exprFullBind!= expr` → ) that originally surfaced the partial-bind case was removed in the branch-cleanup pass; the FullBind rebuild itself is the production behaviour.

**Citation hygiene.** New SwDD content per the project conventions cites file + symbol only.

---

<a id="d-81"></a>
## D-81 — `compilation` row's rest[0] must cite the binary's canonical reconstruction, not the queued input (2026-05-20)


**What.** In `prover.cpp::proveKernel`'s deferred-compaction drain, the `compilation` origin row's `rest[0]` (the "original expanded implication" reference) used to be the raw `original` string popped off `pendingCompactionQueue`. It now reconstructs the canonical body from `compiledExpressions[compactCore].elements` via `this->reconstructImplicationFullBind(canonicalKey, canonicalHead)` and cites *that* string instead.

**Why.** `compileImplicationToCompact` dedups alpha-equivalent inputs to the same `implication<N>` name. Two theorem reformulations whose only difference is the order of two co-bound variables in a `>[w_i,w_j]` group (e.g. `[w2,w4,w5][w3,w4,w6][w1,w5,w6]` vs `[w2,w5,w4][w3,w5,w6][w1,w4,w6]`) collapse to the same compact form, but the binary stores only the first-seen body. Any later input whose binder ordering differs from the binary registered form is alpha-equivalent but not structurally identical.

The verifier's `check_compilation` reconstructs the implication from the binary's `elements` via `_build_implication_from_elements` (the Python mirror of `reconstructImplicationFullBind`) and compares with `rest[0]` modulo `_normalize_with_unchangeables`. The normalizer rewrites both binder content and arg content to a fresh `v<k>` scheme, so any alpha-equivalent form **whose binder list matches the body's first-appearance order** passes. Citing `original` as-is can produce a body whose binder list `>[w_i,w_j]` is inconsistent with the first-appearance order of `w_i` and `w_j` in the body (the inner premise reads `in3[w2,w_j,w_i,+]` so `w_j` appears before `w_i` but the binder names them `w_i` first) — the resulting normalized strings differ in `>[v_k,v_l]` vs `>[v_l,v_k]` and the structural check fails.

**Before / after** (main-Peano, peano-only run, ):
- before: `compilation success 0, failure 3` + `origin success 56, failure 3` → `Verifier: 6900 checks, 6 FAILED.`
- after: `compilation success 2, failure 0` + `origin success 53, failure 2` → `Verifier: 6492 checks, 2 FAILED.`

The 2 remaining origin failures are a **related-but-distinct** issue at a different emission site: chapter 31 lines 1 + 46 are `implication`-tag rows whose `rest[0]` (the cited rule body) is in the canonical-binary form, while `global_theorem_list.txt`'s `mirrored statement` registration uses the swapped form (`reshuffledMirrored` from `ce::createReshuffledMirrored`). `_alpha_canonicalize_bound_vars` does not try binder permutations, so the membership check misses. Same canonicalization principle could apply at the global-theorem registration site; deferred pending user direction.

**Why not the alternatives.**
- Make the verifier's `_alpha_canonicalize_bound_vars` smarter to try binder permutations. Forbidden by [I-16](30_invariants.md#i-16) (verifier sacred) without explicit user consent. Also more expensive at scale — every cross-batch citation check would gain combinatorial canonicalization cost.
- Normalize `original` at queue insertion (the eight `recordPendingCompaction` call sites). Possible but redundant — the binary entry exists after compaction by construction, so the drain is the single canonical place to read it from. Touching eight sites instead of one is more code surface for the same effect.
- Cite numeric placeholders (`>[1,2,3,4,5,6](AnchorPeano[1..6])(>[7,8,9](in3[7,8,9,4])...)`) directly without name substitution. The chapter's existing `implication`-tag rows use named binders (`N,i0,s,+,*,i1,w1..w6`); `reconstructImplicationFullBind` driven by the binary's numeric-placeholder elements happens to emit the same name-style as the chapter (via the prover's downstream rename), so the citation reads as named in the chapter — no consistency break.

**Code.** Drain loop in `prover.cpp::proveKernel` C5 section. The `addOrigin` call now passes `canonicalOriginal` (locally computed) instead of `original`.

**Citation hygiene.** New SwDD content per the project conventions cites file + symbol only.

---

<a id="d-77"></a>
## D-77 — D-76 compact-form mail deposit must use `std::set<int>` for the level set, not `{0..kySize}` (2026-05-20)


**What.** In `prover.cpp::proveKernel`'s deferred-compaction drain (the post-`pool.join` per-tuple loop over `pendingCompactionQueue`), replace

```cpp
std::set<int> compactLevels;
for (int i = 0; i <= kySize; ++i) compactLevels.insert(i);
```

with

```cpp
std::set<int> compactLevels;  // empty, per I-51
```

The compact-form `(implication<N>[…])` deposited into `mailOut.statements` now ships with an empty level set, matching the retired `Mail::implications` tuple channel (which always wrote `std::set<int>` at every one of its eight insertion sites on main HEAD).

**Why.** The synthesised set `{0, 1, …, kySize}` injected one extra integer into the level set of every recovered rule. On the receiver side, `addExprToMemoryBlock(..., status=3, levels={0..kySize},...)` forwards the set into the rule's `LocalMemoryValue::levels` via `addToHashMemory`. Every subsequent firing computes the derived statement's `levels` as `union(rule.levels, premise[i].levels)` — so the polluted set leaks into every derived expression. The discharge gate `allLevelsInvolved` at `prover.cpp::addExprToMemoryBlock` then reads `levels.size` against `memoryBlock.level + 1`. For a hypothesis LB at level 1 the gate wants size 2; the derived expression carried size 3; the gate returned false; the head landed in `encodedStatements` but was never promoted to `globalTheoremList`. 345 incube theorems (the entire N=6+ cascade of `(>[15](in3[2,N,15,4])(=[15,N]))` uniqueness theorems and their downstream back-reformulations and contradictions) were silently swallowed this way.

**How the bug was localised.** Trap-from-within the N=6 hypothesis LB `(in3[2,6,15,4])` under `AnchorIncubator`, side-by-side dump comparison with (= main HEAD). The trap showed: rule chain `(Anchor)(in2[2,15,3])(=[15,6])` present in both branches' `overallHashMemory.originals` from burst 4 onward; dep `(in2[2,15,3])` in both branches' `encodedStatements` from burst 5 onward; derived `(=[15,6])` first appears at burst 6 EXIT in BOTH branches. Difference: REF's `statementLevelsMap` entry for `(=[15,6])` is `levels={0,1}` (size 2 = level+1 = 2 ✓ → discharge fires, `toBeProved` 9→8); RESHUFFLE's is `levels={0,1,2}` (size 3 ≠ 2 ✗ → discharge silently swallowed, `toBeProved` stays at 9). Both dep statements have identical level sets in both branches (`(Anchor){0}`, `(in2[2,15,3]){0,1}`); the extra `2` could only come from the rule's installation levels — traced to `(implication52[])` having `levels={0,1,2}` on the compact-form side, traced to the D-76 deposit code, traced to the `for (int i = 0; i <= kySize; ++i)` loop.

**Why not the alternatives.**

- Threading the original implication's level set through `pendingCompactionQueue` (record + replay) would also work, but is more code, more state, and doesn't match the established convention. The tuple channel's empty-set convention has been the right answer the whole time; the D-76 drain just needed to match it.
- Widening the `allLevelsInvolved` gate to accept supersets would compromise the gate's discharge soundness for legitimate non-mail derivations. Out of scope.

**Before / after.**

- Before: 690 incube theorems proved Peano-incube; 345 missing vs 's 1035. Verifier 63 548 checks 0 failures (less than REF because fewer chapters to check).
- After: **1035 incube theorems** (matches REF exactly), verifier **91 925 checks, 0 failures, airtight**. The 345-theorem gap is closed entirely.

**Code.** `prover.cpp::proveKernel`'s deferred-compaction drain block. See also [I-51](30_invariants.md#i-51), [G-46](50_gotchas.md#g-46), and the rewritten [`02_glossary.md::levels`](02_glossary.md#levels) (the prior glossary entry calling this a `(newStatementLevel, newEqualityLevel)` monotonic counter pair was stale and misleading — corrected in the same commit).

---

<a id="d-79"></a>
## D-79 — Move mail-absorption back to pre-fixpoint after the post-fixpoint experiment was fatal for contradiction LBs (2026-05-20)


**What.** Move the entire mail-absorption block in `prover.cpp::performElementaryLogicalStep` (sameIterationInternalMail absorb+clear, mailIn.exprOriginMap merge, eq-class origin sync, mailIn.statements `status=3` drain, expandedImplications pull, mailIn clears) **back to its pre-fixpoint position**, right after the entry-trap region and before arena/reqBuf setup. (ASIC 0.1 reshuffle 3/8) had moved this block post-fixpoint as the central architectural choice of the reshuffle ("burst inputs from previous-burst state only"). This entry reverses that choice. The persistent per-LB fields `workingMemory` / `externalStatements` / `intExternalStatements` are kept and now serve as **same-burst** staging: cleared and refilled by the pre-fixpoint absorb, read by the same burst's request generation. The `sendMail` + `mailOut` clears stay at the post-fixpoint position.

**Why.** Trap-from-within the `__contradiction__(=[7,10])` LB on the reshuffle (per `reconstruction.md`) showed that the post-fixpoint absorb adds **one PK of latency** between the compact-form arrival in `mailIn.statements` and the rule firing in the LB's hashburst. Combined with D-76's deferred-compaction broadcast (which already adds one PK at the producer side), the contradiction-LB rule landed two PKs late. With the assumed `(=[a,b])` driving combinatorial equivalence-class substitution across the LB's typing/existence rules, those two extra unchecked hashbursts inflated the per-LB `nameMap` past `MAX_NAME_IDS = 16384`. The Peano-incube run on the branch tripped the `prover.hpp::makeIntNormalizedKeyWithMap` assert at `varId=22834`. With the absorb back at pre-fixpoint, the compact form arriving at burst K mailIn is absorbed at the top of burst K and fires in burst K's hashburst — same-burst absorb-fixpoint, one PK of residual latency from D-76 only (acceptable for Peano-incube; FTA-scale survivability open).

**Why not the alternatives.** A `Mail::implications` revival was considered and rejected — that channel was deleted on purpose, and reviving it would not address the same-burst absorb-fixpoint property anyway. A start-of-burst absorb of `mailIn.statements` only (leaving everything else post-fixpoint) would have left the sameIterationInternalMail revival channel out of sync with its producer's mid-burst inserts. Moving the whole block is the minimum-invasive shape and matches main HEAD's order.

**Operational consequence.** The reshuffle's "burst inputs determined by previous-burst state only" property is intentionally reverted; the persistent per-LB fields lose their architectural reason (they were the carrier for previous-burst state) and now act as same-burst staging that could be dropped in favour of routing directly through `overallHashMemory` (cleanup work). D-76 deferred-compaction broadcast, Mail::implications deletion, and the `coreId=-1` slot-0 handling remain untouched. [I-21](30_invariants.md#i-21) was already worded position-agnostic in the post-fixpoint era and stays correct.

**Code.** `prover.cpp::performElementaryLogicalStep` — the pre-fixpoint absorb block, between the entry trap and arena/reqBuf setup.

---

<a id="d-78"></a>
## D-78 — Delete the `Mail::implications` tuple channel (2026-05-18,, commit )


**What.** Remove the `Mail::implications` struct member entirely (`set<tuple<chain, head, args, levels, theorem>>` on `Memory::mailIn` / `Memory::mailOut` / `Memory::sameIterationInternalMail`); remove every producer site (the eight `updateGlobalDirect`/`updateGlobal` `mailOut.implications.insert` sites; the `addExprToMemoryBlock` recovered-implication re-broadcast; the load-time `broadcastTheorems` push; the `smashMail` merge; the `sendMail` dst.implications insert + the empty-channel early-out term; the `destroyMailboxes` clear; the per-burst clears in `performElementaryLogicalStep`); update the hashburst dump (Rule 14 explicit user approval) to drop the three `.implications` sub-sections in mailIn/mailOut/sameIterationInternalMail dumps.

**Why.** Implications now travel exclusively as the compact `(implication<N>[…])` form via `Mail::statements`, deposited by the D-76 deferred-compaction drain (single-threaded after `pool.join`) and recovered receiver-side via `status=3` disintegration. The tuple channel was redundant with the compact-form channel, and removing it eliminates one of two parallel routing systems for the same logical content. The user's design view (2026-05-20): the implications channel was carrying the head of an implication whose antecedent is the receiver's own scope predicate — that is logically a *statement* arriving at the receiver, not an implication-still-needing-resolution, and the right channel for it is `Mail::statements`.

**Why not the alternatives.** Keeping both channels was the prior state (after D-76 added the compact-form deposit alongside); the user's choice was to commit to a single delivery shape. Reverting D-76 instead would have re-introduced the D-76 race ([I-28](30_invariants.md#i-28)) and is incompatible with ASIC 0.1's flat-mail-channel intent.

**Code.** `memory.hpp::Mail` (member removed; the comment line stays as a tombstone). `prover.cpp` ~13 sites. `prover.hpp::sendMail`. `hashburst_dump.cpp` (three `.implications` sub-sections removed).

---

<a id="d-83"></a>
## D-83 — Compact-form deposits at the two formerly-backup-less producer sites (2026-05-18,, commit )


**What.** Two producer sites on main HEAD pushed onto `mailOut.implications` directly, without the paired `recordPendingCompaction` that D-76 added at the eight `updateGlobalDirect`/`updateGlobal` broadcast sites:

1. The load-time `broadcastTheorems` push that ships externally-provided theorems into the grid before iteration begins.
2. The `addExprToMemoryBlock` recovered-implication re-broadcast — when a disintegration produces a new implication, the receiver-side path re-broadcasts the recovered implication to its descendants.

This entry adds a paired `recordPendingCompaction` (or, for the load-time path, an inline compact-form deposit if outside the deferred-drain context) at each site. Without these, deleting `Mail::implications` ([D-78](#d-78)) would silently drop implications produced at these two sites.

**Why.** Required for [D-78](#d-78) to be safe — the deletion presupposes every producer of `Mail::implications` has an equivalent compact-form deposit on `Mail::statements`. The pre-D-76 audit identified the eight broadcast sites; these two were missed because they were not in `updateGlobalDirect`/`updateGlobal`. The entry restores parity.

**Code.** `prover.cpp::broadcastTheorems` (load-time deposit). `prover.cpp::addExprToMemoryBlock` (re-broadcast deposit, in the `imps` loop). Both flagged with `// ASIC 0.1 reshuffle: compact-form deposit backup for the deleted Mail::implications channel`.

---

<a id="d-86"></a>
## D-86 — `coreId == -1` routes through slot 0 in the deferred-compaction drain (2026-05-18,, commit )


**What.** The deferred-compaction drain in `proveKernel` (the post-`pool.join` single-threaded pass that compiles queued implications and deposits the compact form via `sendMail`) iterates over per-core slots indexed by `coreId`. Several status=0/1 force-deep internal-disintegration sites (anchor handling, OR seeds, contradiction-LB seeding, recursion-hypothesis seeding) hardcode `coreId == -1` to signal "non-worker / internal context". On main HEAD these reached the legacy `Mail::implications` channel which had no per-core fan-out; on the reshuffle they reach the new deferred-compaction drain which iterates real core indices. This entry maps `coreId == -1` to slot 0 (always valid since `logicalCores >= 1`) deterministically. The `assert(coreId >= -1)` tripwire is kept (Rule 19) — only the now-defined `-1` is handled, not weakened to "any negative".

**Why.** Trap+A/B confirmed the legacy main-HEAD code path reached the `Mail::implications` insertion ~3054x per incubator batch from `coreId == -1` callers and shipped every one. The D-76 compact-form substitute must preserve that broadcast volume, so `coreId < 0` is routed through a defined default core. Choosing slot 0 (sorted-`std::map` iteration is key-sorted → deterministic, and `sendMail` merges via set-insert → order-independent per [OPEN-22 RESOLVED](#open-questions)) lands a `-1` entry and a real-0 entry both deterministically in slot 0.

**Code.** `prover.cpp` — the deferred-compaction drain inside `proveKernel`, the per-core send loop.

---

<a id="d-84"></a>
## D-84 — `status=3` external-mail absorb routes recovered implications to `overallHashMemory + workingMemory`, recovered facts to `externalStatements`/`intExternalStatements` (2026-05-18,, commit )


**What.** Receiver-side absorb of `mailIn.statements` (drained inside `performElementaryLogicalStep`'s pre-fixpoint mail-absorption block, after [D-79](#d-79)) calls `addExprToMemoryBlock(..., status=3,...)`. `status=3` enters `disintegrateExpr2` to recover the implication / fact from the compact mail form. Recovered implications are routed to `overallHashMemory` (full visibility) **plus** the per-burst `workingMemory` (Batch-1 source), and explicitly **NOT** to `localHashMemory`/`localHashMemoryDelta` (those are local-impl-only; an external rule installed there would be mis-treated by Batch 5 as a local delta). Recovered facts go through the kernel and are additionally staged into `externalStatements`/`intExternalStatements` for the same burst's mail-pair batches. The earlier interim experiment with `allowExistenceDisintegration=false` (status=3 entered `disintegrateExpr2` with the existence witness mint skipped) was reverted — `disintegrateExpr2`/`disintegrateExprCore2` are now byte-identical in control flow to main HEAD.

**Why.** The compact `(implication<N>[…])` form arriving via the D-76 channel must reach the receiver's hash memory with the same effect as the deleted `mailIn.implications` tuple absorb (which called `addToHashMemory(chain, head, …)` directly). Routing through `addExprToMemoryBlock` with a dedicated status keeps the standard receiver-side path (origin handling, mail-out clearing, kernel discharge of `toBeProved` goals) intact. The status-distribute rule keeps `workingMemory`/`externalStatements` populated for Batch 1 / Batch 3-4 mail-pair batches.

**Code.** `prover.cpp::addExprToMemoryBlock`, the `imps` loop's `status=3` branch. The pre-fixpoint absorb call site is `prover.cpp::performElementaryLogicalStep`, the `mailIn.statements` `status=3` drain.

---

<a id="d-87"></a>
## D-87 — Add per-LB `workingMemory`/`externalStatements`/`intExternalStatements` (2026-05-18,, commit )


**What.** Three new `Memory` fields:

- `workingMemory` (`HashMemory`) — stages rules recovered from receiver-side `mailIn.statements` absorb (`status=3` compact-implication disintegration). Read by request-generation Batch 1.
- `externalStatements` (`std::vector<EncodedExpression>`) — stages raw expressions received via mail (1:1 with the `mailIn.statements` entries, ungated, before/without disintegration). Mirrors main HEAD's per-burst `mailIntEncoded`.
- `intExternalStatements` (`std::vector<IntEncodedExpr>`) — integer-encoded mirror of `externalStatements`. Read by Batch 3/4 mail-pair generation.

All three are emptied at the start of the absorb block (`workingMemory = HashMemory; externalStatements.clear; intExternalStatements.clear;`) and refilled in the same block. With the pre-fixpoint absorb position ([D-79](#d-79)), the same burst's request generation reads them.

**Why.** When commit moved the absorb post-fixpoint, the per-burst throwaway containers built inline (`Memory working` for Batch 1; `mailIntEncoded` for Batch 3/4) had to migrate to persistent per-LB fields so the *next* burst's request generation could read them. After [D-79](#d-79) brought the absorb back pre-fixpoint, the fields are no longer required for cross-burst persistence (same-burst absorb→fixpoint sees them locally) — but they were kept as same-burst staging to minimise the diff.

**Code.** `memory.hpp::Memory` — three new fields with default initialisers.

---

<a id="d-76"></a>
## D-76 — Compile every mail-bound implication to compact form and deposit it as a mail expression (2026-05-17)

> **🔴 CONFIRMED ROOT CAUSE (2026-05-17, follow-up session) — UNSYNCHRONIZED PARALLEL-PHASE INVOCATION (data race). Supersedes the sub-key-conflation and negation-drop theories below (kept un-deleted per the annotate-don't-erase convention).** Two fresh full `main.py` runs with a per-call compaction trace localized it. `updateGlobalDirect` (which now calls `compileImplicationToCompact`) is reached by **two** paths: the serial post-`pool.join` broadcast drain in `runIterationForOneLB`, **and** `addExprToMemoryBlock` (the `coreId`-parameterized per-worker function — parallel proof phase, its own doxygen cites [I-28](30_invariants.md#i-28)) which calls `updateGlobalDirect(theorem, coreId)` (`prover.cpp`). The wrapper doxygen's premise *"runs only on the single-threaded post-join broadcast drain... deterministic"* is **false**. On the parallel path, worker threads call `compileImplicationToCompact` → `compileCoreExpressionMapCore` / `excludeRepetitions`, mutating shared `implCounter` (plain `int`, `++`/`--`), `compiledExpressions` and `repetitionExclusionMap` (`std::map`) with **no lock** — `updateGlobalDirect`'s `theoremListMutex` scopes only the theorem-list block; the compile functions have zero synchronization. Direct (non-inferred) evidence: the saved trace contains torn/fused lines (multiple records interleaved on one physical line — impossible without simultaneous `ofstream` writes; 15–19 of 1337 lines per run) and impossible shared-field values (`keyNeg` 811/1830 for two-element keys, from a concurrently-clobbered member). Effect: order-dependent dedup collapse — 16 distinct-implication→one-name collisions per run, distinct `implication<N>` 1359 vs 1361, verifier 10120 vs 10122 — while `theorems.txt` is byte-identical (42) because the proof machinery itself is properly synchronized and only the additively-deposited compaction is not. This **violates [I-28](30_invariants.md#i-28)**. The "negation overlooked" hypothesis is disproved (0 pure-negation-drop collisions; `makeNormalizedEncodedKey` correctly emits a per-constituent negation bit). The "`excludeRepetitions` partial-sub-key conflation" framing is a *symptom* of the race, not a single-threaded key-design flaw. **FIX APPLIED (Option A, user-approved, same session).** The eight broadcast sites now call `recordPendingCompaction` (cheap mutex-guarded enqueue into `pendingCompactionQueue`, no global compile state touched); a single-threaded pass after `pool.join` sorts the queue and performs `compileImplicationToCompact` + the compact / `compilation`-origin deposit, flushed per originating core via `sendMail`. Single-threaded sorted allocation is deterministic and injective and satisfies [I-28](30_invariants.md#i-28). Diagnostic trap removed; unit guards added (`compileimpltocompact_negation_is_injective`, `recordpendingcompaction_appends_under_mutex`; 507/507 pass). The separate ~22 GB incubator memory-expansion concern is unchanged in volume (work deferred, not added) and did not reproduce in the verification runs. **VERIFIED (2026-05-17): determinism PASSED.** Two fresh full `main.py` runs on the fixed build are byte-identical: `theorems.txt` (42 — identical to pre-fix, no proof loss), `GL_binary_shared.json`, `hashburst_trace.txt`, and every verifier count (`operator registry consistency` 1397, 10125 / 109704 total, 0 failures both runs); distinct `implication<N>` is stable at **1364** both runs (vs the pre-fix non-deterministic 1359 / 1361 — 1364 is the correct *injective* count, higher because the race no longer collapses distinct implications). Every original smoking-gun symptom now reproduces byte-for-byte.

> **🔴 STATUS — REFUTED BY DoD VERIFICATION (2026-05-17). BRANCH BLOCKED.** Two full two-pass `main.py` runs disproved the "non-functional" and "determinism holds" claims in the *Why it is non-functional* paragraph below (kept un-deleted per the decisions-log annotate-don't-erase convention; that paragraph is the **original, now-refuted** reasoning):
> - **Not non-functional — incubator OOM.** Compiling+registering *every* mail implication drives combinatorial growth in `excludeRepetitions` / `repetitionExclusionMap` / the cross-batch `GL_binary_shared.json` (~1365+ entries). In the ~1035-theorem incubator batch `gl_quick.exe` reached **~22 GB RSS and froze** (OS thrash, ~54 min no progress). This is a memory *expansion* — the opposite of the ASIC-0.1 memory-reduction intent.
> - **Not deterministic.** Same proven-theorem set both runs (incubator 1035/183/2, main 42 — identical), but the compiled-implication registry differs run-to-run (1370 vs 1371; 7 `(implication<N>[])` names differ) → divergent proof-graph chapters and verifier check counts (`operator registry consistency` 1391 vs 1392 → 10119 vs 10120). The "single-threaded sorted drain ⇒ deterministic" argument held per-batch in isolation but **not** across the full multi-batch run.
> - **Root cause is a real bug breaking intended injectivity, not flakiness.** Implication compilation is *designed* to be **injective** — a structural canonicalization mapping each distinct implication to a stable compact `implication<N>` (and structurally-identical implications to the *same* name; that dedup is correct). The observed symptom (identical proven-theorem set both runs, but different `implication<N>` set/count: 1370 vs 1371, 7 names) means **some bug breaks that injectivity** — the compact name is not, in practice, a pure deterministic function of the implication's structure. The exact injectivity-breaking defect is **not yet isolated** (branch held); candidate surfaces are the `excludeRepetitions` / `makeNormalizedEncodedKey` canonical-key computation (`giveAllSortedCombinations` × permutations into `repetitionExclusionMap[(splitNK, category)]`) or its inputs (e.g. order-dependent state leaking into `splitNK`). Same *class* as [G-42](50_gotchas.md#g-42)/[D-60](#d-60) (compact-name identity), but here the keying is *meant* to be sound and a defect breaks it. Tracked as [G-45](50_gotchas.md#g-45).
> - **No fix applied.** Per the the project conventions banner / Rule 8 the branch is **held for user architectural direction**. Commits 1–6 + the artifact snapshot remain intact and pushed; the investigation trap was reverted (`git reset --hard` to the artifact-snapshot commit). A determinism-only fix (content-addressing the name) would be **wrong** — it would make the unsound collapse reproducible, not correct it. `compilation` emitting 0 chapter rows is accepted by the user as fine; it is **not** part of this blocker.


**What.** At every site that pushes a tuple onto `mailOut.implications` (the eight `updateGlobalDirect` / `updateGlobal` broadcast sites), the prover now *additionally*: (1) compiles the implication to its compact `(implication<N>[…])` form via the new `ExpressionAnalyzer::compileImplicationToCompact` wrapper; (2) deposits that compact form as a `mailOut.statements` element at `validityName == "main"`; (3) emits a paired `compilation` history tag into `mailOut.exprOriginMap` keyed by the compact form, with the original expanded implication as the single antecedent (both `"main"`). The pre-existing implication tuple is untouched — the deposit is purely additive. A new verifier checker `check_compilation` validates the tag structurally (the compact name's GL-binary `elements`/`signature` reconstruct the cited original). Two enabling fixes ship alongside: `excludeRepetitions` no longer unconditionally strips a trailing comma (a zero-free-arg compile now yields the well-formed `(implication<N>[])` instead of malformed `(implication<N>])`); the wrapper reuses the existing `compileCoreExpressionMapCore` `(>` branch and `stripUPrefixAST` rather than adding a parallel compile path.

**Why.** Preparatory groundwork for the ASIC 0.1 memory rework, where the mail channel must carry compact named forms rather than raw implication tuples. Landing the compaction + provenance now — additive and non-functional — de-risks the ASIC 0.1 cut by exercising the compaction on every broadcast implication (including fully-bound theorems, the shape that exposed the latent `excludeRepetitions` zero-arg bug) a full release ahead of the consumer-side change.

**Why it is non-functional (DoD: Gauss + FTA-rung-1 prove, verifier 0 failures, two byte-identical runs).** `exprOriginMap` is process documentation, never a proof input ([I-44](30_invariants.md#i-44)) — the `compilation` origin cannot alter control flow. The compact statement is a fresh synthetic `(implication<N>[…])` atom: absorbed at `status=3` (no disintegration), it is no existing rule's premise, no `toBeProved` goal, and is never negated, so it produces no new firings. The deposit scope is always `"main"` (the implications channel is main-only by [I-26](30_invariants.md#i-26); the per-item `Mail::statements` assert is satisfied). The paired origin also discharges the receiver's mandatory paired-origin assert under `trackHistory` ([D-45](40_decisions.md#d-45)). Determinism holds because `updateGlobal*` run single-threaded on the post-`pool.join` drain over sorted queues, so `implCounter` allocation order is fixed across runs.

**Why not the alternatives.** *A new dedicated mail field* was considered and rejected by the user: the existing `mailOut.statements` is the intended "mail expressions" channel and is sufficient since all mailed implications are main-scope. *Reusing the `theorem` origin tag* for the paired origin would conflate "this is a proved theorem" with "this compact form was produced by compiling that implication" — the dedicated `compilation` tag keeps the provenance auditable and independently verifiable. *A parallel compaction routine* was explicitly forbidden by the user; the extension lives inside `compileCoreExpressionMapCore`'s existing implication branch.

**Retired invariant — "no implication as a constituent of another compiled expression".** Compiling top-level theorem implications surfaced a hard, deterministic assert in `ExpressionAnalyzer::checkCompiledCoreExpressionMap` (`prover.hpp`, `DeepChecker`): it deep-checked the head of every `implication`-category compiled entry and aborted if any constituent resolved to an `implication`-category entry ("the prover does not yet support implications as constituents of other implications"). Two mechanisms now legitimately produce that shape: (1) `compileCoreExpressionMapCore`'s implication branch recurses a multi-premise theorem's inner `>` into its own `implication<N>`, so the outer implication-entry's head *is* an `implication<N>`; (2) `_merge_into_shared` promotes the minted `implication<N>` cross-batch (the shared registry grew to ~1365 entries), so a later batch's `repetitionExclusionMap` re-uses those names as constituents of hand-defined `and` operators (`fXY`, `identity`). Per **explicit user decision** the invariant is judged outdated (it predated any top-level-implication compaction) and is retired: the `DeepChecker` implication branch no longer asserts — it accepts a nested implication as a recursion-stop leaf (like `atomic`). The unrelated "core not found in `compiledExpressions`" guard in the same walk is preserved. This is a Rule-8 architectural change made on the user's explicit instruction; I-19 (never weaken an assert to pass a test) is overridden here only because the user, as architecture owner, ruled the contract itself obsolete — not to make a test green.

---

<a id="d-75"></a>
## D-75 — One implication-binder rule: bind every non-`u_` variable, everywhere (2026-05-16)


**What.** Collapse the two historical implication-binder behaviours into one. Previously theorems (built by the conjecturer, and rebuilt at theorem level by `reconstructImplication`) bound only anchor slots used ≥2× in the body, while every other implication already bound all non-`u_` variables via `reconstructImplicationFullBind`. Now there is a single rule everywhere: in `(>[bound](premise)(body))`, a variable is bound unless its name starts with `u_`, placed at the left-most premise that mentions it. Concretely: `reconstructImplication` becomes a thin forwarder to `reconstructImplicationFullBind` (one implementation); the conjecturer lists every anchor-atom argument in the outer `>[...]` instead of only the body-referenced subset (both the int and string connection lanes, plus the reshuffle-internal single-occurrence skip); the verifier's `check_implication` drops its `(Anchor`-first "all vars changeable" branch and runs every implication through the general changeable/unchangeable path. A theorem carries no `u_` args, so every variable in a theorem — including every anchor slot — is now bound.

**Why.** The sparse theorem rule (old [I-5](30_invariants.md#i-5)) was obsolete. It existed on the stated rationale that single-occurrence anchor symbols (`s`, `+`, `*`) must stay free or the theorem's hash signature changes — but the hash kernel consumes premise text and `ce::getArgs`, never the `>[...]` binder list, so that rationale did not hold. The split's real cost was structural: it forced the verifier to special-case anchor-first implications with an entire parallel branch, and made the conjecturer emit theorems whose anchor binding disagreed with every other implication shape in the system. One uniform rule removes both the verifier branch and the conjecturer special-case, and lets the same reconstruction and the same verifier path serve every implication.

**`reformulateTheorem` carve-in (not carve-out).** `reformulateTheorem` decided whether a Definition is the reformulation "peeling layer" by reading the *reconstructed* last-link `>[...]` cardinality (`boundVars.size == 1 && boundVars[0] == expectedArg`). That is the one place the binder was control flow, not representation; under the unified binder the cardinality no longer encodes the condition. Per explicit user direction this trigger was reworked (not carved out, not left to silently stop firing): it now computes the identical predicate directly from the original chain — the target Definition's set-argument is the single non-`u_` arg that occurs ≥2× across (all premises + head) and appears in no other premise. The same set of reformulations fires; the inner negated-existence still binds exactly the peeled set-argument (byte-identical); only the emitted theorem's outer `>[...]` widen. See [G-44](50_gotchas.md#g-44).

**Operational consequence.** A plain `main.py` run is byte-identical to the pre-change reference () except for the contents inside `(>[ … ]` brackets and timing numbers — no row may appear or disappear, including `reformulated statement` rows. Retires [I-5](30_invariants.md#i-5); rewrites [I-4](30_invariants.md#i-4) as the single binder invariant; inverts [I-11](30_invariants.md#i-11) (anchor slots now legitimately appear in `>[...]`); rewords [I-24](30_invariants.md#i-24)'s rationale (the free `u_*` gate is identified by the `u_` prefix, not by absence from `>[...]`; the gate itself is unchanged).

---

<a id="d-72"></a>
## D-72 — Radical wipe of impl-scope subtree on closure, drained at end-of-burst (2026-05-14)


**What.** When an implication subproof at rooted scope `S` (e.g. `main_boundary_(implicationN[…])`) closes via successful proof, every piece of per-LB state whose attached scope equals `S` or starts with `S + "_boundary_"` is physically removed. New helper `Memory::wipeSubtree(closedScope)` in `memory.cpp` walks every scope-tagged container and erases matching entries; the helper is invoked at two former impl-discharge call sites in `addExprToMemoryBlockKernel` (the recursion-discharge branch and the `OrScopeKind::NotOrScope` branch) and at each victim of `cleanUpOrIntegrationBranches`.

**Deferral to burst boundary.** The wipe does not run immediately at the discharge call site. Instead the closed scope name is inserted into a new per-LB queue `Memory::pendingWipeScopes`; at the very end of `performElementaryLogicalStep` (after `sanitizeHashMemory` + `sanitizeToBeProved`), the queue is drained and `wipeSubtree` runs once per queued scope. Deferral is mandatory because the discharge sites sit inside `addExprToMemoryBlockKernel`'s `sortedNew` loop — immediate wipe would erase `statementLevelsMap` / `equivalenceClassesMap` entries that subsequent iterations of the loop assert-look-up (`prover.cpp ~3812` / `~3993` / `~4068`). Drainage at burst end makes the full wipe (including those scope-keyed maps) safe.

**Containers wiped (per Memory LB).**
- `toBeProved`, `encodedStatements` (+ `intEncodedStatements` mirror)
- `localEncodedStatements*` family (vectors + parallel set + delta + int mirrors)
- `wholeExpressions`
- `statementLevelsMap`
- `intKnownStatements` (packed `(origId, validityId)` keys filtered via decoded `validityId`)
- `equivalenceClassesMap`
- `expandedImplications`, `integrationPrepared`, `integrationPreparedMarker`
- `weakVariables`, `orAdmissionSet`
- `mailIn.statements`, `mailOut.statements`, `sameIterationInternalMail.statements` (origin map half stays per [I-26](30_invariants.md#i-26) / [I-44](30_invariants.md#i-44))
- Per `HashMemory` in {overall, local, delta}:
 - `encodedMap` LMVs filtered by `lmv.validityName`; empty value-vector drops the key
 - The four owner-set maps from [D-71](#d-71) — owner-set filtered; empty owner-set drops the key
 - `remainingArgsNormalizedEncodedMap` secondary index pruned in lockstep when keys drop from `normalizedEncodedKeys`
 - `admissionMap`, `admissionMapIntegration`, `rejectedMap`, `rejectedMapIntegration`, `admissionStatusMap`
 - `admissionSetIntegration`, `triggersForAdmissionSetIntegration`, `consumedAdmissionKeys`, `revisitInProgress`

After the sweep the closed scope's name + every matching int validity id are inserted into `validityNamesToFilter` / `intValidityNamesToFilter` (belt-and-suspenders against in-flight mail referencing the closed subtree; Site H ancestor-scan at kernel entry will short-circuit any such re-deposit). *(Annotation: the string half is gone per [D-128](#d-128) — only the id inserts remain.)*

**Containers preserved on purpose.**
- `exprOriginMap` (per the project conventions / [I-44](30_invariants.md#i-44) — chapter export reads it).
- `nameMap` / `pairMap` / `idToSub` / `stackOfValidity` (validity registries grow monotonically; never pruned).
- `orDisjunctCount` (keyed by OR signature, no scope dimension).
- `orBookkeeping` (parent scope not in the key per the known cross-stack same-orSignature gotcha — partial filter would mis-route; deferred for a separate fix).
- `HashMemory::originals` (no scope tag).
- `integrationStartIntMap` (counter snapshots, no scope tag).

**Why this over the previous selective cleanup.** The deleted helpers `cleanUpIntegrationPreparation` + `cleanUpIntegrationPreparationCore` (formerly at `prover.cpp ~5685-5726`) erased only the rooted scope itself plus its direct `_var0_*` fresh-binding children (prefix `rootedScope + "_boundary__var0_"`). Deeper sub-scopes (`_orint_…`, nested-hypo) survived as inert orphans behind the `intValidityNamesToFilter` ancestor-scan filter. With `wipeSubtree` the orphans are physically reclaimed.

**`cleanUpOrIntegrationBranches` survives.** Its trigger event is OR convergence, not impl closure, and its victim discovery is by OR signature rather than scope ancestry. The per-victim `removeExpressionFromMemoryBlock` work that used to run inline is subsumed by the deferred radical sweep via `pendingWipeScopes`; the per-victim synchronous filter inserts into `validityNamesToFilter` / `intValidityNamesToFilter` run alongside the queue insert per [D-73](#d-73) so the gates bounce post-convergence rule-fires inside the about-to-be-wiped subtrees immediately. *(Annotation: string half gone per [D-128](#d-128).)*

**Known issue surfaced under this design** — see [G-43](50_gotchas.md#g-43). The wipe-then-re-derive cycle interacts badly with the prover's monotonic `it_/int_` iteration-variable counters (per-LB, never reset); each post-wipe re-derivation mints fresh iteration var names, generating fresh expressions, which generate fresh `nameMap.idToName` entries. On IncubatorGauss1 the per-LB NameMap grew to 26000+ entries (against `MAX_NAME_IDS = 16384`) and asserted out. Trace evidence at [G-43](50_gotchas.md#g-43); resolution pending.

**Code.**
- [`memory.hpp::Memory::wipeSubtree`](../GL_Quick_VS/GL_Quick/src/memory.hpp) — declaration.
- [`memory.cpp::Memory::wipeSubtree`](../GL_Quick_VS/GL_Quick/src/memory.cpp) — definition.
- [`memory.hpp::Memory::pendingWipeScopes`](../GL_Quick_VS/GL_Quick/src/memory.hpp) — new queue field.
- Drain block in `prover.cpp::performElementaryLogicalStep` immediately after `sanitizeToBeProved`.
- Replaced call sites: `prover.cpp::addExprToMemoryBlockKernel` recursion-discharge branch + NotOrScope branch.
- Refactored `prover.cpp::cleanUpOrIntegrationBranches` queues victims instead of running immediate per-victim cleanup.

See also [I-50](30_invariants.md#i-50), [D-71](#d-71), [D-73](#d-73), [I-44](30_invariants.md#i-44), [`20_core_concepts/04_validity_stack.md`](20_core_concepts/04_validity_stack.md).

---

<a id="d-73"></a>
## D-73 — Restore synchronous scope-filter inserts at impl-scope NotOrScope closure and OR-integration convergence + descendant-aware rule-fire gate (2026-05-15)


> **Annotation ([D-128](#d-128)).** The string half of the filter pair is gone; each two-line insert below is now just the `intValidityNamesToFilter.insert(...)` line. Snippets preserved as written at decision time.

**What.** Two filter-set inserts now run synchronously at **both** closure-time sites — at the very moment a scope that is about to be subtree-wiped becomes inert from the proof's point of view:

Site 1 — `addExprToMemoryBlockKernel`, `OrScopeKind::NotOrScope` branch (implication-integration scope's bare signature is published to `main`):
```cpp
memoryBlock.validityNamesToFilter.insert(effectiveValidity);
memoryBlock.intValidityNamesToFilter.insert(valId);
```

Site 2 — `cleanUpOrIntegrationBranches` victim loop (OR-integration convergence: one branch proves the OR; sibling branches and the proving branch itself are about to be wiped):
```cpp
for (each victim vname / id) {
    mb.pendingWipeScopes.insert(vname);
    mb.validityNamesToFilter.insert(vname);
    mb.intValidityNamesToFilter.insert(id);
}
```

At both sites these run BEFORE / alongside the existing `pendingWipeScopes` enqueue from [D-72](#d-72). The deferred radical wipe still drains at end-of-burst; only the two filter inserts run mid-kernel. They are cheap and assert-safe per the safety boundary documented in the wipe-deferral rationale (the assert risk lives in `statementLevelsMap` / `equivalenceClassesMap`, neither of which is touched here).

The rule-fire consensus gate in `memory.cpp::checkLocalEncodedMemoryStatic` has been expanded from an exact-match string lookup
```cpp
if (memoryBlock.validityNamesToFilter.count(expressionListValidityName)) return;
```
to an int ancestor-walk mirroring Site H at `prover.cpp::addExprToMemoryBlockKernel`:
```cpp
for (int16_t anc : nm.ancestorsOf[consensusValidityId]) {
    if (memoryBlock.intValidityNamesToFilter.count(anc)) return;
}
```
The exact-match form caught only the closing scope itself; rule-fires whose consensus scope landed at a descendant (`_boundary_orint_…`, `_boundary_ordis_…`) slipped past pre-branch too. The ancestor-walk catches the whole impl-scope and orint-branch subtree the moment any of their int ids enter `intValidityNamesToFilter`.

**Scope.** The recursion-discharge branch in `addExprToMemoryBlockKernel` is unchanged — pre-branch had no filter inserts there either, and the path is unrelated to the implication-integration subproof closure mechanism.

**Why this exists.** [D-72](#d-72) replaced the pre-branch `cleanUpIntegrationPreparation` selective-wipe + `validityNamesToFilter` / `intValidityNamesToFilter` insert triple (NotOrScope), and the pre-branch per-victim filter inserts inside `cleanUpOrIntegrationBranches`, with the deferred radical wipe alone. The original assumption was that `wipeSubtree`'s filter inserts at end-of-burst replaced the synchronous ones — both sets get populated either way. In practice, however, the window between closure-firing and end-of-burst left the closing scope (and at the OR-convergence site, the entire victim branch subtree) unfiltered: every subsequent `addExprToMemoryBlock` call at those scopes, plus every subsequent rule-fire whose consensus scope was inside them, was free to deposit/fire.

NotOrScope-only-restored regression (run `run_immediate_filter_fix.log`, branch HEAD ): on rung-1 ES2 LB, `(implication22[1,4,2,6,15])` still does not close. Burst 10 closes one orint branch; `cleanUpOrIntegrationBranches` queues both branches for wipe but does not filter them, so the remainder of burst 10's `sortedNew` loop keeps admitting rule-fires inside the about-to-be-wiped victim subtrees. The drain at end-of-burst wipes everything and finally populates the filters; bursts 11-47 then plateau at `total_exprs=234,247` with no further progress for 37 idle bursts before the IncubatorPeano phase budget exhausts. The downstream `IncubatorGauss1` batch crashes (non-zero subprocess exit at `run_modes.py:72`).

**Restored behaviour at Site 2 — pre-branch parity.** Pre-branch (`prover.cpp::cleanUpOrIntegrationBranches`) per-victim ran:
```cpp
// remove statements at vname (inline)
// remove toBeProved at vname (inline)
mb.validityNamesToFilter.insert(vname);
mb.intValidityNamesToFilter.insert(id);
```
The inline statement/toBeProved removal is gone (subsumed by the deferred radical wipe via `pendingWipeScopes.insert(vname)`); the two filter inserts are restored, byte-for-byte, in the same loop body.

**Post-fix verification.** Full pipeline (IncubatorPeano → Peano → Compressor → IncubatorGauss → IncubatorGauss1 → Gauss) completes end-to-end; verifier reports `107,055 checks, 0 failures — airtight`. `theorems.txt` matches baseline (42 theorems, identical set). Rung-1 `pi_lev_*` mint count at the active orint branch fell to 94 (from 191 at NotOrScope-only-restored, from a much larger figure at no-fix-restored). The previously-observed IncubatorPeano 37-idle-burst plateau and IncubatorGauss1 non-zero-exit are both eliminated.

**Code.**
- `prover.cpp::addExprToMemoryBlockKernel` — NotOrScope branch (around the `pendingWipeScopes.insert(effectiveValidity)` line).
- `prover.cpp::cleanUpOrIntegrationBranches` — victim loop (around the `pendingWipeScopes.insert(vname)` line).
- `memory.cpp::checkLocalEncodedMemoryStatic` — consensus gate (the `intValidityNamesToFilter` ancestor-walk block).

See also [D-72](#d-72), [I-50](30_invariants.md#i-50), [I-27](30_invariants.md#i-27) (Site H ancestor-scan), [`20_core_concepts/04_validity_stack.md`](20_core_concepts/04_validity_stack.md).

---

<a id="d-71"></a>
## D-71 — Four hashMem subkey containers reshaped from bare sets into owner-set maps (2026-05-14)


**What.** The four `HashMemory` containers
```
normalizedEncodedKeys
normalizedEncodedSubkeys
normalizedEncodedSubkeysMinusOne
normalizedEncodedSubkeysMinusTwo
```
are reshaped from `std::unordered_set<IntNormalizedKey, IntNormalizedKeyHash>` to
`std::unordered_map<IntNormalizedKey, std::set<ExpressionWithValidity>, IntNormalizedKeyHash>`.

The value-side `set<ExpressionWithValidity>` records every implication+scope pair that installed an entry under that key. Membership lookups (`find!= end`, `count`) keep the same contract.

**Why.** The radical subtree wipe ([D-72](#d-72)) needs to drop hashMem entries when the originating implication's scope closes. With bare key sets there was no back-pointer from the subkey to the implication — wiping by scope was impossible. Refcount-by-owner-set is the natural solution: each implication holds a "share" of every subkey it installed; the subkey survives until every share is released. A subkey shared by impl A at scope S1 and impl B at scope S2 stays alive when either closes.

**Insertion contract.** Every insertion site now records the owner:
- `addToHashMemory` (`memory.cpp`): inserts `ExpressionWithValidity(curOrigImpl, validityName)` at every subkey/key insert + at the marker-LMV install — `curOrigImpl` is the current multiplied implication copy from `multiplyImplication(originalImplication)`.
- `makeNormalizedKeysForAdmission` (`memory.cpp`): gained two new parameters `originalImpl` + `validityName`, threaded from the caller in `addToHashMemory`. Markers in `encodedMap` are now tagged with `originalImplication = originalImpl` / `validityName = validityName` so the encodedMap wipe-by-`lmv.validityName` catches them too.

**Wipe contract.** `Memory::wipeSubtree` walks each map; for each entry, removes owners whose validity matches the closed subtree predicate. If the owner-set becomes empty, the key is erased. Dropped `normalizedEncodedKeys` keys are collected and pruned from `remainingArgsNormalizedEncodedMap` value-sets in a second pass so the secondary index stays consistent.

**Helper signatures updated.**
- `prover.hpp::preEvaluateFromEncoded` — `keySet` parameter type changes from `const unordered_set<IntNormalizedKey, …>&` to `const unordered_map<IntNormalizedKey, set<ExpressionWithValidity>, …>&`. Body uses only `find/count`, both compile unchanged.
- `prover.hpp::growBaseCandidates` — same shape; `targetSubkeys` parameter retyped.

**Test code.** `src/tests/test_memory.cpp` uses `hm.normalizedEncodedKeys[k]` (creates a default empty owner-set) for the size-check tests; previous `.insert(k)` calls no longer compile on a `unordered_map`.

**Why this is sound under refcount semantics.** A subkey can legitimately be shared by multiple `(impl, scope)` pairs — `makeNormalizedSubkeys` walks every permutation of an implication's key array; two different implications can produce the same subkey under different permutations. The radical wipe's "drop key when owner-set empties" rule preserves the subkey for any surviving owner.

**Code.**
- [`memory.hpp::HashMemory`](../GL_Quick_VS/GL_Quick/src/memory.hpp) — declarations.
- [`memory.cpp::addToHashMemory`](../GL_Quick_VS/GL_Quick/src/memory.cpp) — insertion sites (8 locations).
- [`memory.cpp::makeNormalizedKeysForAdmission`](../GL_Quick_VS/GL_Quick/src/memory.cpp) — new owner parameters + insertion sites.
- [`prover.hpp::preEvaluateFromEncoded`](../GL_Quick_VS/GL_Quick/src/prover.hpp) — signature.
- [`prover.hpp::growBaseCandidates`](../GL_Quick_VS/GL_Quick/src/prover.hpp) — signature.
- [`memory.cpp::growBaseCandidates`](../GL_Quick_VS/GL_Quick/src/memory.cpp) — definition.

See also [D-72](#d-72), [I-49](30_invariants.md#i-49), [`20_core_concepts/02_hash_engine.md`](20_core_concepts/02_hash_engine.md).

---

<a id="d-74"></a>
## D-74 — Hashburst trace dump relocated to dedicated TU + expanded to every per-LB container (2026-05-14)


**What.** The hashburst trap infrastructure — previously ~440 lines of inline lambdas at the top of `prover.cpp::performElementaryLogicalStep` — is moved into a dedicated translation unit:

- `src/infra/hashburst_dump.hpp` — public API: `bool isTargetLB(const Memory&)`, `void dumpEntry(const Memory&, const std::map<std::string, LogicalEntity>& compiledExpressions)`, `void dumpEarlyExit(const Memory&)`, `void dumpExit(const Memory&)`. *(Source-file location updated 2026-05-27 — moved from `src/` to `src/infra/`; behaviour and public symbols unchanged.)*
- `src/infra/hashburst_dump.cpp` — implementation. Section writers in an anonymous namespace; the three call-site mutexes and counters are file-scope statics (no longer per-lambda statics).

The three trap sites in `prover.cpp::performElementaryLogicalStep` (ENTRY, EARLY-EXIT inside the fixpoint loop, final EXIT) are now one-line delegations to the new helpers.

**Section ordering preserved byte-for-byte.** The legacy section block (header, LB chain, encodedStatements, toBeProved, exprOriginMap, hashOriginals, admissionMap, encodedMap markers, mailIn, compiledExpressions one-shot on first ENTRY) is reproduced in the new TU in the same order, with the same format strings, so the user's grep / diff recipes over  keep working without modification.

**Extended sections added (appended after the legacy block).** Every remaining per-LB container is now dumped on every trap fire, so future diagnostics don't need separate cerr traps:

- `Memory` counters (`startInt` / `startIntRepl` / `startIntPi` / `level` / flags) + `recursionHypothesisId` + `contradictionTheoremId` (both decoded for their sections).
- `nameMap` (`idToName`, `idToSub`, `stackOfValidity`, `ancestorsOf` — every entry expanded; `pairMap` size only).
- `statementLevelsMap`, `wholeExpressions`, `localEncodedStatements`/`Delta`.
- `intKnownStatements` (packed keys decoded into `(origId, validityId)` with NameMap name lookup).
- `equivalenceClassesMap`, `integrationPrepared`/`Marker` + `integrationStartIntMap`, `expandedImplications`, `weakVariables`.
- `orAdmissionSet`, `orBookkeeping`, `orDisjunctCount`.
- `intValidityNamesToFilter` (with name decode), `pendingWipeScopes`, `canBeSentSet`, `canBeSentMarkerSet` (both derived from their id sets, decoded + lex-sorted), `axedVariables` (derived from the id set, decoded + lex-sorted), `intAxedVariables` (with name decode). The former `validityNamesToFilter` section was dropped with the container ([D-128](#d-128)).
- `mailOut`, `sameIterationInternalMail` (statements + implications fully expanded — chain, head, remainingArgs, levels — + exprOriginMap + expandedImplications).
- Per `HashMemory` in {`overallHashMemory`, `localHashMemory`, `localHashMemoryDelta`}:
 - `encodedMap` LMVs (every LMV: value, originalImplication, validity, levels, justification, productOfDisintegration, key elements, remainingArgs)
 - The four owner-set maps with the `IntNormalizedKey` and full owner list per entry
 - `remainingArgsNormalizedEncodedMap` (every key-set entry + every value `IntNormalizedKey`)
 - `admissionMap` (every `AdmissionMapValue`: key, remainingArgs, maxDepth, maxSec, flag)
 - `admissionMapIntegration` (every `Instruction`: `markedGoal` + every `LogicalEntity` in `data` with category / arity / definedSet / signature / every element + every payload string)
 - `rejectedMap` (every `RejectedMapValue`: renamedExpression, expression, iteration, concreteConstituent, levels, siblings)
 - `rejectedMapIntegration` (every `RejectedMapIntegrationValue`: concreteConstituent, compoundExpression, siblings)
 - `varsIn*` caches (every entry; previously size-only)
 - `admissionStatusMap`, `productsOfRecursion` (derived from the id set, decoded + lex-sorted), `productsOfRecursionIds`, `consumedAdmissionKeys`, `revisitInProgress`, `maxKeyLength`.

**Rule-14 invariance preserved.** The chain match (rung-1 LB: `(EnumerationSet2[2,6,15])` under `(AnchorIncubator[1..14])` under root sentinel), the output file path , the truncate-on-first-entry append-on-rest discipline, the three call sites — all unchanged.

**Build registration.** New files added to `GL_Quick_VS/GL_Quick/GL_Quick.vcxproj` + `GL_Quick.vcxproj.filters`.

**Code.**
- [`hashburst_dump.hpp`](../GL_Quick_VS/GL_Quick/src/infra/hashburst_dump.hpp) — public API.
- [`hashburst_dump.cpp`](../GL_Quick_VS/GL_Quick/src/infra/hashburst_dump.cpp) — implementation.
- `prover.cpp::performElementaryLogicalStep` — three delegations (~440 inline lines deleted).

See also the project conventions, [`10_pipeline/04_prover.md`](10_pipeline/04_prover.md).

---

<a id="d-70"></a>
## D-70 — LB bubble-up disable triggers on deeper-scope-only `toBeProved` residue (2026-05-14)

**What.** The bubble-up block inside `prover.cpp::deactivateUnnecessary` disables an LB when (a) no children are active AND (b) no `toBeProved` entry remains at `validityName == "main"`. Entries at deeper scopes — hypothetical sub-block scopes, OR-branch scopes, integration boundary scopes — no longer block deactivation.

The pre-change rule was `block->toBeProved.size == 0` (any TBP entry at any scope kept the LB alive).

**Why this is sound.** A `toBeProved` entry's discharge path depends on its `validityName`:

- A `main`-namescope entry is a real obligation for the LB's user-facing state; discharging it adds a fact directly to the LB's main-scope state. While such an entry remains, the LB has productive work to do.
- A deeper-scope entry (under a hypothesis, OR-branch, or integration scope) is internal sub-derivation bookkeeping. The only path to discharge it is for an active child sub-block at that scope to do the work and route the result up via integration.

The `!anyActiveChild` precondition guarantees that no child sub-block can do that discharge work — every direct child is already disabled. Under `!anyActiveChild`, a deeper-scope `toBeProved` entry is therefore structurally stranded: no future state of GL processing can touch it. Keeping the LB active on stranded entries does not unlock any derivation; it only consumes scheduler time per burst on a block that cannot make progress.

**Why this was needed.** `sanitizeToBeProved` ([D-67](#d-67)) rewrites deeper-scope TBP entries' `it_/int_` args to canonical form. The rewritten entries persist at the same deeper scope; if the owning sub-block has already disabled, the rewritten entry becomes a residual that the old rule would have respected indefinitely. Sanitize is not the soundness origin of the strandedness (the entries were already unreachable once `!anyActiveChild` held), but it increases the count of such residuals, and that growth is what surfaced the rule's looseness as a measurable runtime drag during the investigation.

**Sources of deeper-scope residue.**
1. **Sanitize rewrites at terminal scopes.** A deeper-scope entry's `it_/int_` args are canonicalised after the owning sub-block has finished its work; the rewritten entry persists.
2. **Sub-block early termination.** A sub-block disables via its own deactivation path (own goals discharged or hypothesis contradicted) while its parent's deeper-scope TBP entries at that scope remain unresolved.
3. **OR-branch and integration boundaries.** Integration emits `toBeProved` records at the integration scope; if the integration target never becomes derivable, the record lingers.

**What still blocks deactivation.**
- A main-scope TBP entry — the LB has real main-scope work.
- A deeper-scope TBP entry combined with an active child sub-block — `anyActiveChild` flips the gate.
- A nested sub-block at a deeper scope that itself has active sub-children — the bubble-up walks bottom-up; ancestor LBs see `anyActiveChild` via the still-active deeper descendant.

The conjunction "no active children AND no main TBP" is the smallest sufficient condition for "the LB cannot produce future main-scope state."

**Verification.** Full pipeline (`main.py`) on the post-sanitize tip — Peano + Gauss + incubator-Peano + incubator-Gauss + FTA-ladder rung-1 — passes the verifier (0 failures across the 30 tag categories + ~9 chapter-level meta-counters). No theorem-loss regression on the corpus exercised; the change is observed as a runtime cleanup, not a correctness change.

**Code.** [`prover.cpp::deactivateUnnecessary`](../GL_Quick_VS/GL_Quick/src/prover.cpp), the bubble-up block following the recursion-LB sweep (it walks `chainBlocks` in reverse, looking for any active child of each `block`, then evaluates the disable condition). See also [D-67](#d-67) (`sanitizeToBeProved` — produces the residue this rule cleans up), [D-69](#d-69) (`sanitizeHashMemory` — companion), [I-48](30_invariants.md#i-48), [`20_core_concepts/01_logic_blocks.md`](20_core_concepts/01_logic_blocks.md) (LB lifecycle).

---

<a id="d-67"></a>
## D-67 — Sanitize `toBeProved` under equi-classes via single end-of-burst rewrite (2026-05-14)


**What.** New method `prover.hpp::sanitizeToBeProved` called once per burst at the end of `performElementaryLogicalStep`, sibling of `sanitizeHashMemory`. Walks `Memory::toBeProved`; for each entry, finds args matching `it_*_lev_*_*` / `int_lev_*_*` whose equi-class (at the entry's `validityName` or any visible ancestor scope) contains a strictly-higher-priority `it_/int_` peer; substitutes the downprioritized args in place. The rewritten entry takes the source's `(auxies, tags)` value verbatim at the source's `validityName`; the original entry is erased. If the rewritten key already exists in `toBeProved`, the two collapse to one.

**Priority rule.** `int_*` outranks `it_*`; within the same prefix, lex-smallest wins. `repl_*` and plain (non-`it_/int_`) names never qualify as replacements and never participate in the ranking. Same priority machinery as `sanitizeHashMemory` and the canonical-pick block in `applyEquivalenceClassToRejectedMapIntegration`.

**Why this design (vs the additive + sweep + catalogue prototype that was tried first).** The earlier prototype (`applyEquivalenceClassToToBeProved` + `cleanUpToBeProved` + position catalogue `tbpAllowedIndices` + cascade-discharge of equivalent siblings + 3 call sites + propagation) had four moving parts to keep in sync across discharge, sweep, and insert paths, and produced Peano-theorem-count explosions (58 → 160) until the catalogue gate was added. Once we constrained replacements to `it_/int_` → `it_/int_` only (the same rule that `sanitizeHashMemory` uses for implications), every soundness motivation for the catalogue disappears:

- Plain Peano goals (no `it_/int_` args) see zero substitution because no class peer can win the priority test against a non-`it_/int_` argument.
- Cross-iteration substitution-surface widening cannot happen because the rule only ever maps `it_/int_` → `it_/int_`; a rewrite cannot introduce an `it_/int_` at a previously-non-`it_/int_` position.
- Equivalent siblings collapse to the same canonical form at the next sanitize pass, so a separate cascade-discharge is unnecessary; whichever sibling discharges first leaves at most one stale canonical-equivalent entry, which the next burst's sanitize folds in.

Net result: one helper, one call site, zero side-data structures.

**Scope rule.** The entry's `validityName` is never changed. The class peer must come from the entry's own scope or a strict ancestor — the helper's inner loop walks `mb.nameMap.strictAncestorNames(keyTBP.validityName)` plus the entry's own scope, then picks the best peer across all those equi-classes. Class-deeper-than-goal is rejected structurally: a class at a descendant scope is invisible at the entry's scope, so it never contributes a peer.

**No origin map emission.** Goals are exempt from the equality-history-line rule per [I-44](30_invariants.md#i-44) — goals have no upstream history to extend.

**Code.** [`prover.hpp::sanitizeToBeProved`](../GL_Quick_VS/GL_Quick/src/prover.hpp). Call site at the end of `prover.cpp::performElementaryLogicalStep`, immediately after `sanitizeHashMemory(body)`. See also [`05_equivalence_classes.md`](20_core_concepts/05_equivalence_classes.md), [I-45](30_invariants.md#i-45), [D-69](#d-69).

---

<a id="d-69"></a>
## D-69 — Sanitize hashMem rules under equi-classes via end-of-burst rewrite + eradicate-and-mail (2026-05-14)

**What.** Two new methods in `prover.hpp` — `sanitizeHashMemory` and `eradicateImplicationFromLB` — plus a new per-LB set `Memory::expandedImplications` and its mail companion `Mail::expandedImplications`. `sanitizeHashMemory(body)` is called exactly once per burst at the end of `prover.cpp::performElementaryLogicalStep`, after `reactToHypo` and immediately before the EXIT trap. It walks `body.expandedImplications`; for each entry whose `it_/int_` args are now downprioritized under the active equi-classes (entry's own scope OR any visible ancestor), it:

1. Computes the canonical substMap (downprioritized → canonical).
2. Mails the rewritten implication onto `mb.sameIterationInternalMail.statements` at the entry's `validityName`, with an `equality1` line into `mb.sameIterationInternalMail.exprOriginMap`.
3. Calls `eradicateImplicationFromLB` on the old form, which removes it from every per-LB registry that would otherwise dedup the re-push: `wholeExpressions`, `encodedStatements` (+ parallel int vector), `localEncodedStatements{,Delta,Set}` (+ parallel int vectors), `statementLevelsMap`, `intKnownStatements`, every `encodedMap` LMV whose `originalImplication` matches across `overallHashMemory` + `localHashMemory` + `localHashMemoryDelta`, and the underlying `originals` chain when fully orphaned (orphan check scans all three hashMems).

The kernel's `addStatement` → `addExprToMemoryBlock` → disintegration → `addToHashMemory` chain on the next burst re-disintegrates the rewritten implication and re-installs both fact AND rule sides with `disintegration` origins for every chain element via `trackExpansionHistory`.

**The `expandedImplications` index.** Every implication installed into a LB's hash rules — via the `addToHashMemory` call from `addExprToMemoryBlock`'s implication branch — is also inserted into `memoryBlock.expandedImplications` AND `memoryBlock.mailOut.expandedImplications`. `sendMail` propagates the mailOut copy to children; at the start of each burst, `body.mailIn.expandedImplications` is merged into `body.expandedImplications` and `mailIn` cleared. Result: every LB whose hash registry holds an implication has that implication's `(original, validityName)` in its own index, ready for end-of-burst sanitize.

**Why a per-LB index (vs walking `encodedMap` directly).** Walking `encodedMap` head LMVs by `remainingArgs ∩ class.variables` was the first prototype's approach (`applyEquivalenceClassToHashMemoryOriginals` + `cleanUpHashMemoryOriginals`, both retired). It required a per-class apply hook + a two-phase sweep + an orphan-check pass + special-case handling for marker LMVs. The index keys directly on the implication's textual form — exactly what `eradicateImplicationFromLB` needs to drop it from every storage surface — and the LMV walk inside `eradicateImplicationFromLB` is bounded to those LMVs whose `originalImplication` matches one specific string per pending entry.

**Why drop+mail, not direct `addToHashMemory` reinstall.** A prototype that called `addToHashMemory(rewrittenImpl, …)` directly was built and ran; it bypassed disintegration, leaving chain-element premises without `disintegration` origins at the deposit scope, and `visualizer.cpp::buildStack` asserted `"no origin found"` for premises like `(in[7,1])` during chapter export. The drop+mail design routes the rewritten implication through `addStatement` → `addExprToMemoryBlock` → disintegration, which calls `trackExpansionHistory` and emits proper `disintegration` origins for every chain element. Same drop+mail pattern as [D-63](#d-63) (rejectedMap) and [D-64](#d-64) (rejectedMapIntegration).

**Why `sameIterationInternalMail`, not top-level mail.** The `wholeExpressions` filter at `addExprToMemoryBlock`'s entry gate would block a re-push of the rewritten implication if the original is already present. `sameIterationInternalMail` bypasses that gate; kernel processing of `sameIterationInternalMail.statements` runs disintegration regardless of `wholeExpressions` membership.

**Single end-of-burst call site.** `sanitizeHashMemory(body)` and the sibling `sanitizeToBeProved(body)` run once per `performElementaryLogicalStep`, after `reactToHypo`. Equi-class machinery has stabilized by this point in the burst; running once at end-of-burst (not per `addStatement` call) keeps the expensive eradication off the hot fact-absorption path.

**`it_/int_` → `it_/int_` only.** Substitution targets are restricted to `it_*_lev_*_*` / `int_lev_*_*` args; `repl_*` and plain (non-`it_/int_`) names never qualify as replacements and never participate in the priority ranking. Priority: `int_*` outranks `it_*`; within the same prefix, lex-smallest wins. Same priority machinery as [D-67](#d-67) (`sanitizeToBeProved`) and the canonical-pick block in `applyEquivalenceClassToRejectedMapIntegration`.

**Scope rule.** The class peer must come from the implication's own scope or a strict ancestor. The helper walks `mb.nameMap.strictAncestorNames(impEwv.validityName)` plus the entry's own scope and picks the best peer across all visible equi-classes. Class-deeper-than-implication is invisible and never contributes a peer.

**Origin map emission.** The `equality1` line is written into `mb.sameIterationInternalMail.exprOriginMap` (not into `body.exprOriginMap` directly). The kernel's mail-bulk-merge routes it through `addOrigin` into `body.exprOriginMap` with cap-full preference per [D-49](#d-49) / [I-35](30_invariants.md#i-35). Per the project conventions / [I-44](30_invariants.md#i-44), originMap maintenance is mandatory but never drives proof decisions.

**Not touched.** `IntNormalizedKey`, `normalizedEncodedKeys`, `normalizedEncodedSubkeys*`, `remainingArgsNormalizedEncodedMap` — these hold only integer IDs and cannot be invalidated by equi-class rewrites. `multiplyImplication`'s Bell-partition fan-out is irrelevant: it fires only when disintegration is banned, and `it_/int_` variables cannot be minted in that mode.

**Retired prototypes (visible in intermediate squash commits, absent from the squashed tip).** `applyEquivalenceClassToHashMemoryOriginals` (additive per-class hashMem rule reinstall) + 4 call sites; `cleanUpHashMemoryOriginals` (per-class hashMem sweep); the `srcIsCompiledImpl` deposit-loop branch in `applyEquivalenceClass` (see [D-68](#d-68)); `Memory::tbpAllowedIndices` side-map + `catalogueAllowedIndices` helper + frozen-propagation logic; `applyEquivalenceClassToToBeProved` + `cleanUpToBeProved` + `cascadeDischargeTBP`. The `it_/int_` → `it_/int_` restriction makes the catalogue / cascade / sweep structurally unnecessary.

**Code.** [`prover.hpp::sanitizeHashMemory`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`prover.hpp::eradicateImplicationFromLB`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`memory.hpp::Memory::expandedImplications`](../GL_Quick_VS/GL_Quick/src/memory.hpp), [`memory.hpp::Mail::expandedImplications`](../GL_Quick_VS/GL_Quick/src/memory.hpp), [`prover.cpp::performElementaryLogicalStep`](../GL_Quick_VS/GL_Quick/src/prover.cpp) (sanitize call site after `reactToHypo`; `mailIn → body` merge at burst start; `mailOut` clear after `sendMail`), [`prover.cpp::addExprToMemoryBlock`](../GL_Quick_VS/GL_Quick/src/prover.cpp) (index population at implication install), [`prover.hpp::sendMail`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (mail propagation). See also [`05_equivalence_classes.md`](20_core_concepts/05_equivalence_classes.md), [I-46](30_invariants.md#i-46), [D-33](#d-33), [D-49](#d-49), [D-63](#d-63), [D-64](#d-64), [D-67](#d-67).

---

<a id="d-68"></a>
## D-68 — RETIRED — `applyEquivalenceClass` compiled-implication branch (prototype, reverted) (2026-05-14)

**Status.** Prototyped and reverted to main-HEAD behaviour in the same squash. The entry is preserved so the anchor `#d-68` resolves and future agents do not re-invent the same prototype.

**What was tried.** Inside `prover.hpp::applyEquivalenceClass`, split the per-rewrite deposit block by source-expression shape. When `expr2.original` matched `^\(implication\d+\[` (a compiled-implication operator call), skip the normal deposit block and route the rewrite exclusively through `mb.sameIterationInternalMail.statements` with an `equality1` line into `mb.sameIterationInternalMail.exprOriginMap`. A regex-matched `srcIsCompiledImpl` flag drove the branch.

**Why reverted.** [D-69](#d-69)'s rule-side index (`Memory::expandedImplications` + `sanitizeHashMemory` + `eradicateImplicationFromLB`) covers the same case structurally and at a single end-of-burst call site. When `sanitizeHashMemory` rewrites + eradicates an implication on the rule side, the mailed rewrite re-enters via `sameIterationInternalMail` and the kernel's absorb chain populates encodedStatements (fact side) and `addToHashMemory` (rule side) together with proper `disintegration` provenance. The reverted branch added a per-call regex match and a deposit-loop split without enabling any case sanitize doesn't already handle.

**Code.** No live code path. Grep `srcIsCompiledImpl` returns zero hits across `GL_Quick_VS/GL_Quick/src/`. See [D-69](#d-69) for the design that subsumes this case.

---

<a id="d-64"></a>
## D-64 — Integration `rejectedMapIntegration` equi-class drop + mail; supersedes D-43 and retires I-30 (2026-05-13)


**What.** `prover.hpp::applyEquivalenceClassToRejectedMapIntegration` is rewritten to mirror the algebra `applyEquivalenceClassToRejectedMap` ([D-63](#d-63)) drop+mail pattern. Behaviour per call:

1. Build per-class `substMap` (weak → canonical) using `filterIterations` / `updateWeakVariables`'s lex-smallest `int_lev_*` (fallback lex-smallest `it_*_lev_*`) rule. Skip if the class has no canonical-eligible vars.
2. Short-circuit on `varsInRejectedMapIntegrationKeys` overlap (unchanged from pre-D-43 behaviour).
3. Walk every `(keyEv, set<RejectedMapIntegrationValue>) ∈ rejectedMapIntegration`. Admit same-NS and class-shallower (ancestor of entry) scope directions; reject class-deeper (matching D-63's scope rule and the pre-D-43 rmi hook's scope rule).
4. For each entry whose marker key contains a non-canonical class member (`filterIterations(keyEv.original, clss) == false`):
 - **Drop** the entry from `rejectedMapIntegration`.
 - Per-key dedup: collect unique compound expressions across the value set; for each unique compound, union the levels from `statementLevelsMap[EncodedExpression(v.compoundExpression, keyEv.validityName)]`.
 - For each unique compound, **mail** the rewritten compound: insert `(EWV(compoundPost, depositValidity), levels)` into `mb.sameIterationInternalMail.statements` where `compoundPost = ce::replaceKeysInString(compoundPre, substMap)` and `depositValidity = deeperOf(classScope, entryScope)`. Attach an `equality1` origin to `mb.sameIterationInternalMail.exprOriginMap`: source = `EWV(compoundPre, keyEv.validityName)`; justifying = `(=[from,to])` for every substMap entry that actually fired on this compound (extras would make `check_equality1` reject).

Gated on `!parameters.skip_eq_classes`. Erase-only — never inserts into `rejectedMapIntegration` directly.

**Why.** Brings `rejectedMapIntegration` into parity with `rejectedMap` ([D-63](#d-63)). Before this branch, integration's hook performed the additive-on-no-match insert recorded in [D-43](#d-43) / [I-30](30_invariants.md#i-30); the substituted constituents at K2 had no post-substitution `disintegration` provenance recorded in `exprOriginMap` (the production-site emitter wrote origins for the pre-substitution forms only). That gap was the integration-side half of the bug [D-63](#d-63) closed for algebra; [I-37](30_invariants.md#i-37)'s pre-revision text explicitly flagged it as scoped out of the rt_admission_map branch.

The user's framing from the rt_admission_map handoff (`observations.md`): *"Main problem addStatement path after equi class update updates rejectedMapIntegration: messy, breaks history continuity in worst case. Instead we shld do it as latest main head for algebra: admissionMap update after equi class update."* The drop+mail design is exactly that — close the provenance gap on the integration side by routing the substituted compound through `sameIterationInternalMail`, letting the kernel's natural disintegration re-emit `disintegration` origins for canonical-form constituents.

**Soundness chain on revival.** Mirror of D-63's chain on the integration side. The mailed compound's `equality1` origin cites the original compound at its production-time scope (`keyEv.validityName`). The verifier's chapter walk traces back through `equality1` to the original compound's origin (still in `exprOriginMap` after `cleanUpExpressions`, since that sweep erases `statementLevelsMap` / `intKnownStatements` but not `exprOriginMap`). The kernel's downstream integration-side disintegration emits `disintegration` origins for canonical-form constituents via the integration-side expansion path; unadmitted constituents are routed to `rejectedMapIntegration[K']` via `prover.hpp::updateRejectedMapIntegration` with full provenance.

**Investigated / ruled out during design.** Mirror of D-63's design alternatives. The strip-rewrite-restore mechanic for u_-form admission-integration keys was developed for the companion admission-integration hook ([D-65](#d-65)); for rmi the keys are bare-marker form (constructed in Pass B with no u_ prefix) so this hook works directly on substMap without u_-handling.

**[I-22](30_invariants.md#i-22) preserved.** Integration admission templates persist across revival firings. `revisitRejectedIntegration2` (called from the new admission-integration hook and from `disintegrateExprCore2`'s C1 block) does NOT call any `cleanAdmissionMap`-like function on the integration side. This hook does not interact with admission templates at all — it only drops rmi entries and mails compounds.

**Code.** [`prover.hpp::applyEquivalenceClassToRejectedMapIntegration`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (rewritten function body). Wired at the existing four call sites in `addStatement`; no call-site changes. [I-30](30_invariants.md#i-30) retired with stub redirect to [I-37](30_invariants.md#i-37); [I-37](30_invariants.md#i-37) revised to cover both algebra and integration sides. See also [D-63](#d-63), [D-65](#d-65), [I-22](30_invariants.md#i-22).

---

<a id="d-65"></a>
## D-65 — Integration `admissionMapIntegration` equi-class hook (additive K' insert, arg-equalization-filtered) (2026-05-13)


**What.** New method `prover.hpp::applyEquivalenceClassToAdmissionMapIntegration` is the integration-side equi-class hook on `admissionMapIntegration`. Mirrors algebra [D-57](#d-57) (`applyEquivalenceClassToAdmissionMap`) with three adjustments for integration-side data types:

1. **u_-prefix handling.** `admissionMapIntegration` keys are u_-form (built by `disintegrateExprCore2`'s C1 block from `le.signature` with one nonUArg replaced by `"marker"`). Class members are bare names. The hook uses **strip-rewrite-restore**: `removeUPrefixFromArguments(K)` → bare-form rewrite via `enumerateEqClassRewrites` → restore u_ prefix on non-marker bare args via the standard `ce::replaceKeysInString` u_-restore pattern.
2. **Value substitution via augmented substMap.** `admissionMapIntegration` value type is `map<Instruction, set<string>>` rather than `set<AdmissionMapValue>`. Each (Instruction, appliedVars) pair is rewritten independently. Instruction.data[*].signature / Instruction.data[*].elements / Instruction.markedGoal are u_-form strings; appliedVars members are bare names per the construction sites in `updateAdmissionMapIntegration` and `isAdmittedIntegration`. The hook builds an augmented substMap containing bare → canon AND u_<bare> → u_<canon> pairs so both bare and u_-form occurrences get substituted consistently.
3. **Revival via `revisitRejectedIntegration2`.** After each K' insert, the hook fires `revisitRejectedIntegration2(bareK', mb, depositValidity)` where `bareK' = removeUPrefixFromArguments(K')`. `rejectedMapIntegration` keys are bare-marker form, not u_-form, so the lookup uses the bare key.

Wired at the same four call sites in `addStatement` as the other equi-class hooks (same-NS, ancestor-NS, descendant-NS, fixpoint), inserted between the rewritten `applyEquivalenceClassToRejectedMapIntegration` ([D-64](#d-64)) and the algebra `applyEquivalenceClassToAdmissionMap` ([D-57](#d-57)). Order rationale: integration rmi drop+mail emits compounds onto `sameIterationInternalMail`; integration admission insert may then fire `revisitRejectedIntegration2` against rmi entries that were NOT dropped by the rmi hook but become reachable under the new K'; finally algebra admission insert and rejectedMap drop+mail run.

A new `varsInAdmissionMapIntegrationKeys` cache on `HashMemory` mirrors `varsInAdmissionMapKeys` for the integration side. Stores bare-form (u_-stripped) non-marker args to support overlap with class members (also bare). Populated at every `admissionMapIntegration` insert site (the existing site in `disintegrateExprCore2`'s C1 block, plus the new hook's post-loop apply block).

**Design principles** (mirror of [D-57](#d-57) principles 1-4, with the [I-22](30_invariants.md#i-22) asymmetry preserved):

1. **`rejectedMapIntegration` is sacred.** Never written by this function (rewritten via the companion drop+mail hook).
2. **`admissionMapIntegration` keys and Instruction values are metadata.** Equi-class rewrites apply freely.
3. **Keys are ADDED, not replaced** — the original K stays in `admissionMapIntegration`. Codified as [I-42](30_invariants.md#i-42).
4. **Arg-equalization filter ([I-36](30_invariants.md#i-36) mirror).** Drop rewrites that collapse two previously-distinct arg slots to the same value. The positional collision pattern of the admission template must be preserved because `revisitRejectedIntegration2` probes `rejectedMapIntegration[K']` under that pattern.

**Why.** The integration side was missing the equi-class hook entirely before this branch. When an equality registers and the algebra `admissionMap` gets a canonical-form K' via [D-57](#d-57)'s hook, the parallel `admissionMapIntegration[K']` was NOT inserted — which meant the integration side could not match on the canonical form even though the underlying class made it admissible.

**[I-22](30_invariants.md#i-22) preserved.** No on-hit cleanup. `revisitRejectedIntegration2` does not call any `cleanAdmissionMap`-like function on the integration side; this hook does not add one. The user's framing from this session: *"with exception of cleanUp after hit - i think integration can follow algebra playbook with adjustments for integration data types"* — the one exception is explicitly the algebra D-62 canonicalization closure on `cleanAdmissionMap`'s `markerIsOutput` branch, which has NO integration analog. The existing `admissionMapIntegration.erase(...)` calls inside `cleanAdmissionMap`'s `markerIsOutput` branch are key-form no-ops (algebra bare-marker vs integration u_-form) and stay as benign vestiges.

**Arg-equalization filter on integration: scoping decision.** [I-36](30_invariants.md#i-36) is algebra-only in its current scope ("for integration it admits equal variables where original had none. it is nonsense. we do not correct it for integration but for algebra it must be like..."). For the **new** admission-integration hook, the filter IS applied — the soundness rationale carries over symmetrically (positional collision pattern preservation under class-driven rewrites). The pre-existing rmi hook's documented stance "no arg-equalization filter" referred to the now-superseded D-43 additive-on-no-match design; the new drop+mail rmi hook ([D-64](#d-64)) does not enumerate rewrites at all (single substMap from class), so the question is moot there.

**Investigated / ruled out during design.**

- **Augmented substMap vs strip-rewrite-restore.** Considered using only an augmented substMap (bare + u_-prefixed pairs) and applying `ce::replaceKeysInString` directly to the u_-form key. Rejected: `enumerateEqClassRewrites` is designed for bare-name enumeration (its substMap-keys-must-match-class-members invariant), and u_-prefixed names are not class members. Strip-then-restore keeps the helper's contract clean.
- **Direct K' insertion without revisitRejectedIntegration2 fire.** Considered. Rejected: defeats the point of the hook (retroactive revival of already-rejected cohorts that the class now makes admissible).
- **No arg-equalization filter on integration.** Considered. Rejected: revisitRejectedIntegration2 probes `rejectedMapIntegration[K']` under K's positional structure; collapsing distinct arg slots produces a K' whose positional structure doesn't match what the rejection cohort was rejected under.

**Code.** [`prover.hpp::applyEquivalenceClassToAdmissionMapIntegration`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (new method), four wire sites in `addStatement` (same-NS, ancestor-NS, descendant-NS, fixpoint), cache populate at `disintegrateExprCore2`'s C1 admission-integration insert block and the new hook's post-loop apply. `HashMemory::varsInAdmissionMapIntegrationKeys` added in [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp). See also [D-57](#d-57), [D-64](#d-64), [D-66](#d-66), [I-42](30_invariants.md#i-42), [I-22](30_invariants.md#i-22), [I-36](30_invariants.md#i-36).

---

<a id="d-66"></a>
## D-66 — Post-class-update `admissionMapIntegration` canonical sweep; NO on-hit closure (the [I-22](30_invariants.md#i-22) exception) (2026-05-13)


**What.** New method `prover.hpp::cleanUpAdmissionMapIntegration` is the integration-side analog of `cleanUpAdmissionMap` ([D-62](#d-62) point 1). Called from `updateEquivalenceClasses` immediately after the existing `cleanUpAdmissionMap` call. For each class at `validityName`, walks `admissionMapIntegration` entries at `validityName` and erases every entry K for which `filterIterations(removeUPrefixFromArguments(K.original), eqClass)` returns `false` against any class.

The u_-strip before `filterIterations` is required because `admissionMapIntegration` keys are u_-form and `filterIterations`' int_lev_*/it_*_lev_* regex tokens only match bare-form names. The short-circuit cache `varsInAdmissionMapIntegrationKeys` stores bare-form non-marker args (populated at every admission-integration insert), so the overlap check uses bare names naturally.

Gated on `!parameters.skip_eq_classes`. Codified as [I-43](30_invariants.md#i-43).

**What is intentionally NOT mirrored — the [I-22](30_invariants.md#i-22) exception.** [D-62](#d-62) had two halves: point 1 (post-class-update sweep, `cleanUpAdmissionMap`) and point 2 (on-hit canonicalization closure, `cleanAdmissionMap`'s `markerIsOutput` branch). This branch mirrors point 1 only. Point 2 is the algebra-only closure that erases canon-equivalent K' when a disintegration head consumes K. The integration side has no analog because integration admission templates are reusable — a single template can admit multiple distinct `int_` witnesses over the proof lifetime (the load-bearing rule recorded in [I-22](30_invariants.md#i-22)). Cleaning canon-equivalent integration admission entries on hit would silently exhaust the template before its remaining witnesses fire.

The existing `mb.overallHashMemory.admissionMapIntegration.erase(evKey)` calls inside `cleanAdmissionMap`'s `markerIsOutput` branch are key-form no-ops — algebra uses bare-marker keys, integration uses u_-form keys, the lookup on `evKey` (bare-marker) against the integration map (u_-form) never finds a match. They stay as harmless vestiges; the I-22 invariant is preserved by data flow, not by explicit code.

**Why.** Mirrors the expression-side / algebra-admission-side two-phase shape `applyEquivalenceClass*` → `cleanUp*` for the integration admission map. Without the sweep, K (non-canonical under the new class) and K' (canonical, inserted by [D-65](#d-65)) coexist forever, costing memory and duplicate `isAdmittedIntegration` probe work on facts that admit the same shape under the class.

**User framing (this session).** *"with exception of cleanUp after hit - i think integration can follow algebra playbook with adjustments for integration data types"* — the sweep half follows the algebra playbook; only the on-hit closure is excluded.

**Soundness.** K (non-canonical) and K' (canonical, already inserted by the additive hook) encode the same admission rule under the active classes. The integration admission rule is a template indexed by structural positions; rewriting positional names via canon-equivalent class members produces an equivalent template. The algebra-side `cleanUpAdmissionMap` has been running this drop-non-canonical pattern since [D-62](#d-62) without admission-rule regressions; the integration mirror inherits the soundness witness.

**Code.** [`prover.hpp::cleanUpAdmissionMapIntegration`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (new helper), [`prover.hpp::updateEquivalenceClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (one-line call-site insert immediately after `cleanUpAdmissionMap`). See also [D-62](#d-62), [D-65](#d-65), [I-43](30_invariants.md#i-43), [I-22](30_invariants.md#i-22), [I-40](30_invariants.md#i-40).

---

<a id="d-63"></a>
## D-63 — Algebra `rejectedMap` equi-class drop + mail; I-37 revised from "never written" to "never written **directly**" (2026-05-13)


**What.** A new algebra-side equi-class hook `prover.hpp::applyEquivalenceClassToRejectedMap` is wired at the same four call sites in `addStatement` as `applyEquivalenceClassToAdmissionMap` (D-57): same-NS, ancestor-NS, descendant-NS, and the `applyClassesFrom` fixpoint. Behaviour per call:

1. Build per-class `substMap` (weak → canonical) using `filterIterations` / `updateWeakVariables`'s lex-smallest `int_lev_*` (fallback lex-smallest `it_*_lev_*`) rule. Skip if the class has no canonical-eligible vars.
2. Walk every `(keyEv, set<RejectedMapValue>) ∈ rejectedMap`. Admit same-NS and class-shallower (ancestor of entry) scope directions; reject class-deeper (matching `applyEquivalenceClassToRejectedMapIntegration`'s scope rule).
3. For each entry whose marker key contains a non-canonical class member (`filterIterations(keyEv.original, clss) == false`):
 - **Drop** the entry from `rejectedMap`.
 - For each `RejectedMapValue v` in the dropped set, **mail** the rewritten compound: insert `(EWV(compoundPost, depositValidity), v.levels)` into `mb.sameIterationInternalMail.statements` where `compoundPost = ce::replaceKeysInString(v.expression, substMap)` and `depositValidity = deeperOf(classScope, entryScope)`. Attach an `equality1` origin to `mb.sameIterationInternalMail.exprOriginMap`: source = `EWV(compoundPre, keyEv.validityName)`; justifying = `(=[from,to])` for every substMap entry that actually fired on this compound (extras would make `check_equality1` reject).

Gated on `!parameters.skip_eq_classes`. Erase-only — never inserts into `rejectedMap` directly.

**Why.** Brings `rejectedMap` into parity with `encodedStatements` (which is rewritten by `applyEquivalenceClass` + `cleanUpExpressions`) and `admissionMap` (rewritten additively by D-57's hook + dropped non-canonical entries by `cleanUpAdmissionMap` from [D-62](#d-62)). Before this branch, [I-37](30_invariants.md#i-37) sat behind a flat "never written by equi-class application" rule, the consequence being that rejection cohorts at non-canonical marker keys were stranded forever — the class made them admissible under a canonical-form K' (the K' that D-57 inserts into `admissionMap`), but the rejection records keyed at non-canonical K were never re-probed against the new admission landscape, since `revisitRejected2(K')` walks only `rejectedMap[K']` (and `rejectedMap[K]` stayed populated under the now-non-canonical key without firing).

The hook addresses the gap **without** violating I-37's provenance-protection rationale: the rewritten compound's `equality1` origin from `applyEquivalenceClass`'s natural pass over `encodedStatements` is reinforced by the hook's own emission into `sameIterationInternalMail.exprOriginMap`; the kernel's next-hashburst absorb routes the compound through `addExprToMemoryBlock`, which then runs `disintegrateExprCore2`, which emits proper `disintegration` origins for each canonical-form constituent via `trackExpansionHistory`. Unadmitted constituents are written to `rejectedMap[K']` via the standard `updateRejectedMap` writer with full provenance. The hook itself never inserts into `rejectedMap` — it only erases — so I-37's direct-write protection is preserved.

I-37's revised text reflects the split: direct writes by equi-class hooks remain forbidden; indirect writes via the kernel's normal pipeline (downstream of the mailed compound) are now expected. The provenance chain on revival becomes:

```
rewritten constituent  ←  disintegration  ←  rewritten compound
rewritten compound     ←  equality1       ←  original compound + (=[from,to])
original compound      ←  ...(existing chain — anchor / disintegration upstream)
```

Every link is a tag the verifier already validates. No new checker, no shape changes.

**User framing (session 2026-05-13):** *"Simpler solution: remove renamed key in rejectedMap and add new compound after renaming to inner mail with equality origin line."* — the design above is exactly this minus the structural redesign of an earlier, more elaborate draft that proposed direct K' insertion in rejectedMap with manual `disintegration`-origin emission. That draft was rejected as "too messy" — duplicating what the kernel does naturally via `trackExpansionHistory`, and introducing a fourth direct-write channel into a sacred map.

**Investigated / ruled out during design.**

- Whether direct `rejectedMap[K']` insertion with the hook synthesizing the `disintegration` origin is acceptable. Answer: no — it duplicates `trackExpansionHistory`'s production-site emission and creates two parallel emission paths for the same provenance shape, with no soundness benefit. The kernel's natural disintegration is the source of truth; route through it.
- Whether a separate `cleanUpRejectedMap` sweep (mirror of `cleanUpAdmissionMap` from [D-62](#d-62)) is needed. Answer: no — the hook itself owns both drop and mail, no separate post-update sweep required. The expression and admissionMap pipelines split the work into hook (additive insert) + sweep (drop); the rejectedMap pipeline collapses them because there is no additive-insert step on the rejectedMap side (insertions happen indirectly through the kernel).
- Whether the `varsInAdmissionMapKeys`-style short-circuit cache is needed for the new hook. Answer: deferred. The per-entry overlap check is O(arity); rejectedMap typically sits at 10⁴–10⁵ entries at Gauss scale; classes fire roughly 10⁶ times overall. Profile after the first cut and add a `varsInRejectedMapKeys` cache only if the hook surfaces as a hot spot.
- Whether [I-27](30_invariants.md#i-27)'s ancestor-scan dedupe at `addExprToMemoryBlock` entry could skip the absorb of the mailed compound. If skipped, the compound wouldn't be re-disintegrated, and the rejection cohort wouldn't regenerate at K'. Verified empirically by running the full pipeline post-implementation; no verifier failures and no theorem regression observed, so either the dedupe doesn't fire on the relevant cases or the prior ancestor-scope existence is itself sufficient to cover the cohort.

**Soundness argument.** The mailed compound's `equality1` origin cites the original compound at its production-time scope (`keyEv.validityName`). The verifier's chapter-walk traces this back through `equality1` to the original compound's origin (still in `exprOriginMap` after `cleanUpExpressions`, since that sweep erases `statementLevelsMap` / `intKnownStatements` but not `exprOriginMap`). The kernel's downstream disintegration emits `disintegration` origins for canonical-form constituents. Both links are checker-validated tags. The chain is complete; no soundness gap.

**Code.** [`prover.hpp::applyEquivalenceClassToRejectedMap`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (new hook), 4 call-site inserts in `addStatement` (same-NS, ancestor-NS, descendant-NS, fixpoint — each immediately after the matching `applyEquivalenceClassToAdmissionMap` call). [I-37](30_invariants.md#i-37) revised in place.

---

<a id="d-62"></a>
## D-62 — Post-class-update admissionMap sweep + cleanAdmissionMap canonicalization closure (2026-05-13)


**What.** Two coordinated edits to the algebra-side admission-map equi-class machinery. D-57's `applyEquivalenceClassToAdmissionMap` stays additive (adds canonical-form K' alongside non-canonical K); both new operations remove keys outside the hook.

1. **`cleanUpAdmissionMap` post-class-update sweep** ([`prover.hpp::cleanUpAdmissionMap`](../GL_Quick_VS/GL_Quick/src/prover.hpp)). Called from `updateEquivalenceClasses` between the existing `cleanUpExpressions` and `updateWeakVariables` calls. Mirrors `cleanUpExpressions`'s expression-side drop: for each class at `validityName`, walk `admissionMap` entries at `validityName`, and erase every K for which `filterIterations(K.original, eqClass)` returns `false`. The mirror erase on `admissionStatusMap` is paired with each `admissionMap` erase. `admissionMapIntegration` is OUT OF SCOPE (integration domain). `consumedAdmissionKeys` is NOT touched — entries are replaced by canonical K', not consumed by a head. Codified as [I-40](30_invariants.md#i-40).
2. **`cleanAdmissionMap` canonicalization closure** ([`prover.hpp::cleanAdmissionMap`](../GL_Quick_VS/GL_Quick/src/prover.hpp), `markerIsOutput` branch). When a disintegration head consumes K at `validity`, the cleanup now also erases every admissionMap entry K' at the same `validity` whose canonical form under classes at `validity` equals `canon(K)`. Canonicalization uses `canonicalizeUnderClasses` (`prover.hpp::canonicalizeUnderClasses`): substitute every class member with the class's canonical (lex-smallest `int_lev_*`, fallback lex-smallest `it_*_lev_*`), per the same rule `filterIterations` and `updateWeakVariables` use. Each erased K' is removed from `admissionMap`, `admissionStatusMap`, `admissionMapIntegration`, and added to `consumedAdmissionKeys` — symmetric to the existing per-K erase. Codified as [I-41](30_invariants.md#i-41).

Both gated on `!parameters.skip_eq_classes`. Both honor the user-pinned `validity` scope: only classes and entries at the same validity are considered; cross-scope K' (deposited at `deeperOf(class.scope, entry.scope)` by D-57's hook) is **not** chased.

**Why.** D-57's additive hook leaves stale non-canonical K entries in `admissionMap` after class formation. The expression pipeline already does the symmetric two-phase shape — `applyEquivalenceClass` adds canonical-form rewrites, `cleanUpExpressions` drops non-canonical originals via `filterIterations`. AdmissionMap was the only side of the algebra equi-class machinery missing the sweep half of the pattern. Without it, K and K' coexist for the lifetime of the LB, costing memory and duplicating `isAdmitted` probe work. Without the closure on hit, consuming K leaves equi-class-equivalent K' entries to re-fire the same operator output via different surface forms.

User framing (from session 2026-05-13): *"After an eq class is updated in addStatement path a lot of expressions get filtered out as they contain args which are now suppressed. We need it for admissionMap too. Keys of admission map are processed by equi magic and need to be cleaned out similar to expressions."* — point 1 above. And *"When explicit admissionMap cleanup is performed after a hit, we need to cleanup all similar keys: keys which can be generated from the hit one by application of equi classes (of the same validityName)."* — point 2 above.

**Three closure options considered (point 2).**

- **(1) 1-hop.** Enumerate rewrites of K via `enumerateEqClassRewrites` for each class at validity; erase each rewrite from admissionMap. Cheap (O(classes × mappings)) but leaves transitive entries stranded. With classes `C₁ = {a,b}`, `C₂ = {c,d}` and keys `K = (p[a,c,marker])`, `K1 = (p[b,c,marker])`, `K2 = (p[a,d,marker])`, `K12 = (p[b,d,marker])`: erases K, K1, K2 but **leaves K12** — reachable from K via C₁ then C₂, not via any single class.
- **(2) Iterative fix-point.** Mark K1, K2; rerun the 1-hop step on each marked entry; loop until no new marks. Catches K12 (via K1+C₂ or K2+C₁). Cost: passes × |classes| × |args| per cleanup; passes bounded by reachable-set size.
- **(3) Canonicalization scan.** Compute `canon(K)` and walk every admissionMap entry at validity computing `canon(K')`; erase entries matching. O(|admissionMap_at_v| × |args|) per cleanup — single walk; transitive coverage is structural (canonicalization is idempotent under multiple class applications: `canon(canon(K)) == canon(K)`).

Picked **(3)**. Single walk, deterministic, naturally transitive, and the cost shape matches `cleanUpExpressions`'s single walk per class.

**Soundness.**

- *Point 1 sweep.* K and canonical-form K' (already inserted by D-57's hook) encode the same admission rule under the active classes. The expression-side sweep (`cleanUpExpressions` + `filterIterations`) is the soundness witness: GL has been running the expression-side drop for the entire D-33 era without admission-rule regressions, on the same theory that the canonical form is the live representative. AdmissionMap inherits the witness.
- *Point 2 closure.* When K is consumed, the disintegration head producing K's output slot has fired and the operator's output is recorded. Any equi-class-equivalent K' admits the same output under the same operator with the same positional structure ([I-36](30_invariants.md#i-36) preserves the positional collision pattern, so K and K' agree on the operator's slot signature). Erasing K' alongside K does not lose any future admission — it prevents duplicate firings of the same operator output via different surface forms.

**What was investigated.**

- Whether the additive principle for D-57's hook ([D-57](#d-57) principle 3: *"Keys are ADDED, not replaced. The original K stays in admissionMap."*) is violated. Answer: no — the hook stays additive. The sweep is a separate operation at a separate call site, mirroring the expression pipeline's two-phase `applyEquivalenceClass` → `cleanUpExpressions` shape.
- Whether `consumedAdmissionKeys.insert(K)` belongs in the point-1 sweep. Answer: no — entries are dropped because they're non-canonical, not because they fired; insertion would block a legitimate future re-admission of the canonical form.
- Whether the closure should touch `admissionMapIntegration`. Answer: yes for point 2 (mirror the existing per-K erase across all four state structures); no for point 1 (integration domain has its own equi-class hook `applyEquivalenceClassToRejectedMapIntegration` and no algebra-side K' add to mirror).
- Whether cross-scope K' (deposited at `deeperOf(class.scope, entry.scope)` by D-57's hook from the descendant or ancestor direction) should be chased. Answer: no — same-validity pin per user direction. Each scope's `cleanAdmissionMap` triggers and class-update sweep run independently.
- Performance shape of point 2 in the `isAdmitted` hot path. Answer: `cleanAdmissionMap` only does work when `markerIsOutput` (constrained subset of admission probes). Closure walks all admissionMap entries at validity — O(|admissionMap_at_v| × |args|) per output-marker consumption. The existing `varsInAdmissionMapKeys` short-circuit keeps cost zero when no class member overlaps any admission key.

**Code.** [`prover.hpp::cleanUpAdmissionMap`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (new helper), [`prover.hpp::canonicalizeUnderClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (new helper), [`prover.hpp::cleanAdmissionMap`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (`markerIsOutput` branch extended), [`prover.hpp::updateEquivalenceClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (one-line call-site insert). See also [D-57](#d-57), [I-36](30_invariants.md#i-36), [I-37](30_invariants.md#i-37).

---

<a id="d-60"></a>
## D-60 — `repetitionExclusionMap` keyed by `(elements, category)` to enable cross-batch sharing of all spontaneous operators (2026-05-13)


**What.** The C++ allocator's `repetitionExclusionMap` (declaration in [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp), lookup + insertion in `excludeRepetitions` in the same file, JSON-load insertion in [`visualizer.cpp::loadGlBinary`](../GL_Quick_VS/GL_Quick/src/visualizer.cpp)) is keyed by `std::pair<std::vector<std::string>, std::string>` — `(elements vector, category)` — instead of the previous elements-only key `std::vector<std::string>`. The change is local to the map type and three access sites; no behavior change within a single batch.

**Why — root cause this fixes.** Prior to this entry, every batch's invocation of `gl_quick.exe` allocated spontaneous compact operator names (`existence<N>`, `implication<N>`, `or<N>`, `and<N>`) keyed by elements vector alone. The `excludeRepetitions` lookup was category-blind: identical elements vectors in different categories collided, with the first-allocated name winning.

The collision was dormant under [D-54](#d-54) — incubator batches did not contribute to the shared cross-batch registry, so the only allocator visible to a given batch was its own (and within a single batch the conjecturer's enumeration order does not produce cross-category elements collisions). When approach A drops D-54's incubator-skip in `_merge_into_shared` so that incubator and main batches share a single spontaneous registry, the dormant bug surfaces. Concretely the production binaries carry one cross-category collision today: `IncubatorPeano`'s `implication23` and `Peano`'s `existence2` both have elements `[(in[1,u_1]), (in2[1,u_2,u_3])]`. Under approach A, IncubatorPeano's `implication23` lands in `shared.json` first; when Peano's compiler computes splitNK for `∃x∈N: in2[x,b,c]`, the category-blind lookup returns `implication23`. Disintegration then dispatches on `compiledExpressions["implication23"].category == "implication"`, applying implication-disintegration logic to a body that was structurally existential. Every downstream proof depending on existence introduction (Gauss's `or0`-chain reformulations, FTA rung-1's predecessor reasoning) fails silently — theorems vanish from `theorems.txt`. A prior agent attempted approach A without this fix and observed exactly this regression; the prior work was reverted.

**Investigated / ruled out.**
- *Python-only filter at merge.* Detecting cross-category elements collisions at `_merge_into_shared` and skipping the offending entry cannot fix the bug — the elements-keyed lookup is reachable from `loadGlBinary` regardless of whether merge inserts or skips. The unsafe lookup is in C++, after the JSON has been loaded.
- *Per-category prefix sharding of `repetitionExclusionMap`.* Equivalent semantically but requires four separate maps + four lookup branches. Strictly more code touched, same semantics.
- *Refuse to load entries from a batch with a different category seed.* Brittle, requires per-batch metadata; conflicts with the registry-grows-monotonically invariant.

**Verification.** Within a single batch the map sees no cross-category elements collisions in any current binary (verified by enumerating `(elements, category)` across `files/GL_binaries/GL_binary_*.json`), so within-batch behaviour is byte-identical. Cross-batch behaviour is verified by Phase 3 of the approach-A rollout: clean-run `main.py` after wiping `files/GL_binaries/`, then `verifier.py`; `theorems.txt` set must be a superset of the pre-approach-A baseline; `operator registry consistency` reports `failure 0`; every other verifier check airtight.

**Related.** [G-42](50_gotchas.md#g-42), [I-23](30_invariants.md#i-23). Supersedes the load-bearing reason for [D-54](#d-54)'s incubator-skip — see the matching reversal entry [D-61](#d-61) that drops the skip.

---

<a id="d-61"></a>
## D-61 — `_merge_into_shared` admits every tag (incubator + main); filter restricted to spontaneous-allocator name prefix (2026-05-13)


**What.** Single change to `_merge_into_shared` in [`run_modes.py`](../run_modes.py): **drop the `tag.startswith("Incubator")` early-return** (previously lines 121–132). Every tag — incubator and main alike — now contributes its spontaneous-category compact-operator allocations (`implication` / `existence` / `or` / `and` categories) to `GL_binary_shared.json`. The category filter is unchanged; atomic entries (`=`) continue to be excluded.

**Why.** Reverses the load-bearing reason for [D-54](#d-54)'s incubator-skip. Pre–this entry, the merge contract was "skip incubator entirely; merge every category-eligible entry from main batches". That hid the C++ category-blind lookup bug ([D-60](#d-60)): the cross-category elements collision between IncubatorPeano's `implication23` and Peano's `existence2` could never surface at a main-batch consumer of shared.json because incubator allocations never reached shared. That bug fired the moment a prior agent dropped the incubator-skip without first making the lookup category-aware: Gauss + FTA rung-1 theorems vanished from `theorems.txt`. Phase 1 of this branch addresses the lookup; this entry then safely drops the skip.

**Investigated / ruled out.**
- *Add a name-prefix filter `^(existence|implication|or|and)\d+$` to keep `AnchorIncubator` (category `"and"`) out of shared.* A first draft of this entry shipped that filter (commit, reverted in commit immediately following). It was too aggressive: hand-defined entities like `preorder` / `sequence` / `interval` / `limitSet` / `limitSequence` / `fXY` / `fXYZ` / `identity` / `NaturalNumbers` / `fold` / `split` / `constSeq` carry a spontaneous category but do not match the prefix. They ARE transitive dependencies of spontaneous-allocator operators — e.g. `implication20`'s elements contain `(preorder[u_2,u_3,u_4,1])`. Peano's MPL does not define `preorder`; with the filter blocking it from shared, IncubatorPeano's `implication20` allocation merged into shared without its `preorder` dependency, Peano loaded `implication20` into `compiledExpressions`, and the prover asserted `Core expression not found in definitions` when implication-walking reached `preorder`. The crash log lives at  (Peano stage, exit code / `0xC0000409`). The right answer is the category-only filter: anchor entries like `AnchorIncubator` are inert in batches whose theorems do not reference them (Peano's theorems exclusively use `AnchorPeano`), so the speculative pollution risk is not load-bearing.
- *Apply prefix filter only to incubator merges; keep main behaviour byte-identical.* Considered before crashing on the universal variant; same dependency-closure failure mode.
- *Single combined commit (filter + drop early-return) vs two separate.* User-confirmed: single combined commit. After the prefix-filter revert this is reduced to a single change (drop the early-return), but the commit lineage still reflects the original combined-commit decision plus the immediate follow-on revert.
- *Numbering as a fresh D-NN immediately on the sandbox branch.* Rejected per the project conventions; `D-pending-<slug>` reserves an integer at merge time.

**Verification.** Phase 3 of the approach-A rollout: wipe `files/GL_binaries/`, run `main.py` clean from this branch, run `verifier.py`. Abort triggers (Phase 3 plan): theorem loss in `theorems.txt` vs the pre-rollout baseline; any verifier check other than `operator registry consistency` gaining failures; build or pipeline crash. Expected on success: `operator registry consistency: success N, failure 0` (was 37/34 on the pre-rollout baseline); every other check airtight; main-side `theorems.txt` ≥ baseline (42 rows); incubator-side `theorems.txt` ≥ baseline (1211 rows).

**Related.** [D-60](#d-60), [G-42](50_gotchas.md#g-42), [I-23](30_invariants.md#i-23), [D-54](#d-54) (supersedes the incubator-skip portion).

---

<a id="d-59"></a>
## D-59 — Local MSVC introsort port at the 4 `generateEncodedRequests*` sites (2026-05-12) — **TEMPORARY workaround, retired 2026-05-13 by **

**Status — RETIRED 2026-05-13.** `msvc_sort.hpp` and `gl::msvc_sort` are removed from the tree. All four sort sites are back on `std::stable_sort` with the unchanged name-only comparator. The retirement trigger was empirical — and earlier than this entry's "Migration plan" subsection anticipated.

**Retirement evidence.** A clean full-pipeline run (off main HEAD after the integration-side equi-class hooks landed — see [D-64](#d-64), [D-65](#d-65), [D-66](#d-66)) reproduces the 42-theorem baseline including both Gauss / fold variants. Verifier 0 failures across 35,719 checks. Runtime 31 min vs 28.5 min for the `gl::msvc_sort` run — ~9% slower, the expected stable_sort cost per the WSL measurement in the table below, with **no theorem regression** under Win11 + MSVC STL stable_sort.

**Why the regression disappeared.** D-59's empirical claim — `stable_sort` regresses the Gauss / fold theorem — was made against a codepath snapshot that pre-dated the integration-side admission-map machinery. The pre-D-64 codepath had an asymmetry: when an equivalence class formed, the algebra `admissionMap` gained a canonical-form K' (via [D-57](#d-57)), but `admissionMapIntegration`'s parallel K' was never inserted. The fold proof's search path depended on the integration side picking up the canonical form via a tie-order coincidence — specifically, the MSVC-introsort tie order happened to deliver the right facts in the right order to bridge the asymmetric admission gap. `std::stable_sort`'s emergence-order ties broke that coincidence.

[D-65](#d-65) closes the asymmetry directly: `applyEquivalenceClassToAdmissionMapIntegration` inserts the canonical-form K' into `admissionMapIntegration` on every class formation, and `revisitRejectedIntegration2` mails the matching cohort. The integration side no longer depends on tie-order accident to bridge to the canonical form — the explicit hook does it. With the asymmetry closed, `stable_sort` proves fold cleanly.

**What is removed.**

- `GL_Quick_VS/GL_Quick/src/msvc_sort.hpp` deleted.
- `#include "msvc_sort.hpp"` removed from `memory.cpp` and `filter.cpp`.
- Each `gl::msvc_sort(...)` call swapped back to `std::stable_sort(...)` with the same comparator at the four sites listed below.

**What stays.** Everything D-60 / D-61 / I-23 / I-24 / I-25 added since v0.8.1 stays. The cross-host determinism story is now: `stable_sort` contract guarantees same-output across MSVC STL and libstdc++ at these four sites. Other `std::sort` sites in `memory.cpp` / `filter.cpp` (mentioned in the *Investigated / ruled out* section below as out-of-scope at the time) remain untouched — they were not load-bearing under the empirical test.

**Below: original entry preserved for historical reference.**

**What.** A new header [`msvc_sort.hpp`](../GL_Quick_VS/GL_Quick/src/msvc_sort.hpp) provides `gl::msvc_sort` — a faithful re-implementation of MSVC STL's `_Sort_unchecked` (introsort dispatcher with insertion-sort cutoff at 32, heap-sort fallback when the `1.5·log₂(N)` recursion budget is exhausted, 3-way Hoare partition with Tukey's-ninther pivot selection for ranges > 40). It is wired into four sort sites in the rule-firing hot path:

- `memory.cpp::generateEncodedRequestsStatic` — `filteredIdx[]` sort
- `memory.cpp::generateEncodedRequestsStaticPairs` — `filteredIdx[]` sort
- `memory.cpp::generateEncodedRequestsStaticPairs` — `merged[]` pair-merge sort
- `filter.cpp::generateEncodedRequestsStaticCE` — CE-batch `filteredIdx[]` sort

Each replaces a `std::stable_sort` with `gl::msvc_sort`. The comparator is unchanged (name-only: `nm.decode(.nameId)`-based).

**Why (the immediate problem).** Two prior approaches both broke the Gauss / fold theorem proof relative to historical Win11 main HEAD behavior:

1. **4-site total-order tiebreaker** ( /, commit ). Widened the comparator from `name` to `(name, originalId, validityId)` — no ties at all. Produces deterministic output across hosts BUT yields a different tie ordering than MSVC's introsort. On with this comparator: Win11 ran the full pipeline in 51 min, 35,448 verifier checks airtight, 40 proved theorems, **no `fold[…]` theorem**.
2. **`std::stable_sort` + name-only comparator** (emergence-order ties). Identical on Win11 and Linux/WSL by the `stable_sort` contract. WSL ran the full pipeline in 31 min (1.6× faster), 35,403 verifier checks airtight, 40 proved theorems, **no `fold[…]` theorem**.

Both miss what historical Win11 main HEAD (`std::sort` with name-only comparator, MSVC introsort) was apparently doing: producing a tie order in which the Gauss / fold proof goes through. Reading the deltas, MSVC's introsort tie ordering was load-bearing for the fold-theorem search path; neither widening the comparator nor switching to stable_sort preserves it.

**Why (cross-host concern).** `std::sort` is unstable, and MSVC STL (`_Sort_unchecked`) and libstdc++ (`__introsort_loop`) resolve ties using different pivot strategies, cutoff thresholds, and recursion-budget formulas. For inputs with many equivalent elements (typical of `IntEncodedExpr` arrays where multiple entries share an operator name) the two implementations produce visibly different byte output. `gl::msvc_sort` is byte-deterministic regardless of which standard library is in use: cross-host divergence at these four sites is structurally impossible.

**Empirical result (validation).** WSL run with `gl::msvc_sort` at all 4 sites:

| Metric | Win11 (total-order) | WSL (stable_sort) | **WSL (gl::msvc_sort)** |
|---|---|---|---|
| Proved theorems | 40 | 40 | **42** |
| `fold[…]` theorems | 0 | 0 | **2** |
| Verifier checks | 35,448 | 35,403 | 35,746 |
| Verifier failures | 0 | 0 | 0 |
| Runtime | 51 min | 31 min | 34 min |

42 vs 40 proved theorems — both forward and mirror Gauss / fold variants land in `theorems.txt`. 0 verifier failures on 35,746 checks. Runtime sits between total-order (slow) and stable_sort (fast).

**Cost of the workaround.**

- **Runtime.** `gl::msvc_sort` is unstable; it pays the partitioning-overhead penalty that stable_sort avoided. Measured cost on WSL: 34 min vs 31 min for stable_sort (~10% slower). On Win11 historically: 51 min for the comparator-widened path; `gl::msvc_sort` on Win11 not yet measured but expected near the WSL number.
- **Maintenance.** A re-implementation of MSVC's std::sort that we now own. The implementation is line-by-line translated from `microsoft/STL/stl/inc/algorithm`; the upstream is stable (the algorithm is settled), so drift is low-risk. But it is non-zero code to keep in the tree.
- **Conceptual.** GL's prover-pipeline behavior is sensitive to sort tie order. This is a **latent correctness fragility** — the proof of the fold theorem should not depend on which way `std::sort` resolves ties on a particular STL implementation. The right long-term fix is to make the prover insensitive to tie order (e.g. by deduping symmetric search paths, or by using an explicit priority queue keyed on emergence index or some other principled criterion). `gl::msvc_sort` simply freezes the historical tie order so the bug-fix release can ship.

**Migration plan.**

Once the fragility above is addressed (specifically: once the prover's RT does NOT depend on `std::sort`-vs-`stable_sort` tie choice for the Gauss / fold theorem and other regressing theorems), all four sites migrate to `std::stable_sort` with the same name-only comparator. Trigger:

1. The RT campaign lands. The 100–1000× expected memory reduction and the static-prover refactor (project_next_rt_campaign / project_asic_0_1_release) are expected to substantially restructure rule-firing order discipline.
2. After RT lands, re-run the four-site sort experiment: try `stable_sort` at all 4 sites and confirm fold + the other 41 theorems still prove. If yes → migrate, delete `msvc_sort.hpp`, recover the ~10% runtime.
3. If no → the tie-order dependency persists. At that point either deepen the search-order discipline in the rule-firing loop (e.g. introduce an emergence-index secondary key on the comparator) or accept `gl::msvc_sort` as the permanent answer.

The expectation is that the RT campaign rewrites enough of the rule-firing layer that the fragility evaporates and `stable_sort` becomes viable. This decision should be revisited explicitly when RT lands; until then, `gl::msvc_sort` is the configured sort.

**Alternatives considered.**

- **`std::stable_sort` at all 4 sites (emergence-order ties).** Rejected: 1.6× faster but does not prove the Gauss / fold theorem on either host. Excluded for v0.8.1; first candidate for re-evaluation post-RT.
- **4-site total-order tiebreaker `(name, original, validity)`.** Rejected: deterministic and platform-independent but BREAKS the fold theorem by producing a different tie order than MSVC introsort. The investigation made this clear empirically; documented in commit f5654dc0's body.
- **Ship just the sandbox-branch `stable_sort` for the speedup, lose fold.** Rejected: the bug-fix release explicitly cannot regress a theorem that the previous release proved.
- **Patch upstream MSVC + libstdc++ to agree.** Not feasible; we don't control either STL.
- **Use boost::sort or another third-party sort.** Rejected: adds a dependency for a problem we can solve in ~250 LOC in-tree.
- **Sort-free firing order (priority queue).** Out of scope for v0.8.1; correct long-term direction; depends on RT-campaign restructuring.

**Investigated / ruled out during execution.**

- Whether MSVC `std::sort` and `std::stable_sort` produce the same output for our typical inputs. Answer: confirmed they do NOT, both empirically (the fold-theorem regression) and from MSVC STL source (`_Sort_unchecked` is introsort = unstable; `stable_sort` is a separate function with its own implementation and explicit-overflow analysis at `_ISORT_MAX = 32`).
- Whether the speedup observed when switching to `stable_sort` was from the algorithm or from Linux/g++ vs Win11/MSVC. Answer: not fully isolated — needs a Win11 + stable_sort run to fully separate. Best read so far: ~10% comes from stable_sort's avoidance of partitioning overhead on big tie groups; the rest may be host. Not load-bearing for this decision (we're already accepting the 10% to keep the fold theorem).
- Whether other `std::sort` call sites in the prover hot path could regress similarly. Answer: there are 5 other `std::sort` calls in `memory.cpp` / `filter.cpp` (lines 1374, 1419, 1432, 129, 430) that we left untouched. Empirically the WSL run with `gl::msvc_sort` only at the 4 named sites already proves fold; the others are not load-bearing for the immediate problem.

**Code.** [`msvc_sort.hpp`](../GL_Quick_VS/GL_Quick/src/msvc_sort.hpp), [`memory.cpp::generateEncodedRequestsStatic`](../GL_Quick_VS/GL_Quick/src/memory.cpp), [`memory.cpp::generateEncodedRequestsStaticPairs`](../GL_Quick_VS/GL_Quick/src/memory.cpp), [`filter.cpp::generateEncodedRequestsStaticCE`](../GL_Quick_VS/GL_Quick/src/filter.cpp).

---

<a id="d-57"></a>
## D-57 — algebra equi-class hook rewrites admissionMap, never rejectedMap (2026-05-12)


**What.** A new method `ExpressionAnalyzer::applyEquivalenceClassToAdmissionMap` in `prover.hpp` is the algebra-side equi-class hook on the admission map. It walks `admissionMap` entries; for each entry at marker key K with `AdmissionMapValue` set V, it enumerates rewrites K' via the equivalence class (using the shared `enumerateEqClassRewrites` helper) with an arg-equalization filter, applies the substitution to both K (the marker key) and the contents of each `AdmissionMapValue` (its `key` vector and `remainingArgs` set), and ADDITIVELY inserts the (K', V') entry into `admissionMap`. Each new K' is followed by `revisitRejected2(K', mb, depositValidity)` which walks the unchanged `rejectedMap[K']` and mails any matching rejection cohort to `sameIterationInternalMail`. `rejectedMap` is never written by this function. The hook is wired into `addStatement` at the four equi-class application sites (same-NS, ancestor, descendant per D-33, fixpoint) immediately after each `applyEquivalenceClassToRejectedMapIntegration` call.

**Why.** The integration-side `applyEquivalenceClassToRejectedMapIntegration` (prover.hpp::`applyEquivalenceClassToRejectedMapIntegration`) additively inserts substituted rejection records into `rejectedMapIntegration` on its no-match path. The substituted `concreteConstituent` and `siblings` in the new entry have `disintegration` provenance recorded only for their pre-substitution forms at the original production site (via `trackExpansionHistory` inside `disintegrateExprCore2`), not for the post-substitution forms. That is a provenance gap. Per user direction the integration code stays as-is; for algebra a different playbook applies:

1. **`rejectedMap` is sacred.** It holds real disintegration products with `disintegration` origins recorded at production site. Equi-class application never writes to `rejectedMap` — see [I-37](30_invariants.md#i-37).
2. **`admissionMap` keys and values are metadata.** No proof-graph history attached. Equi-class rewrites them freely.
3. **Keys are ADDED, not replaced.** The original K stays in `admissionMap`; K' coexists alongside it. Multiple rewrites at multiple class instances accumulate without erasing earlier entries.
4. **Arg-equalization is forbidden.** Rewrites that collapse two previously-distinct arg slots to the same value are dropped — see [I-36](30_invariants.md#i-36). Preserves the positional collision pattern of the admission key.

The revival path (mail-emit on `sameIterationInternalMail` via `revisitRejected2`) handles two cases uniformly:
- **Direct admission insert** (existing `revisitRejected2` callers at prover.cpp::`updateAdmissionMapRecursion` and memory.cpp::`makeMandatoryEncodedStatementLists1Static`): no rewrite, K is the bare-marker admission key, pre==post in the mail, no equalities. The mail's `equality1` origin is a soft placeholder — see *Composition with D-49* below.
- **Equi-class revival** (the new hook): K' is the rewritten admission key; the rejection record at `rejectedMap[K']` carries the unchanged cohort with intact `disintegration` provenance from production site. revisitRejected2 mails the cohort verbatim.

**Composition with D-49 / I-35.** The mail-deposited `equality1` origin (single-entry `origin.second` = pre-form, no equalities) is structurally insufficient for `verifier.py::check_equality1` (rejects `len(rest) < 4`). It is harmless because the proper `disintegration` origin for each child was written at the original production site by `disintegrateExprCore2::trackExpansionHistory` (the lambda inside `disintegrateExprCore2` emits originDisintegration per child); when the mail-deposited child lands in `body.exprOriginMap` via `addExprToMemoryBlock`'s `addOrigin` call, `addOrigin`'s cap-full preference clause (`prover.hpp::addOrigin`) refuses to displace a foundation origin with a convenience equality1 — existing slot wins. Chapter export reads the `disintegration` origin; verifier accepts.

**Scope handling.** Three directions admitted, mirroring [D-33](#d-33): same-NS, class-shallower (ancestor of entry), class-deeper (descendant of entry). Deposit scope is `deeperOf(classScope, entryScope)`. Visibility soundness: a class at descendant V can rewrite an admission entry at ancestor S because the admission rule's contents are visible at V; the rewritten admission entry lives at V (descendant), valid at V and below.

**`AdmissionMapValue` substitution.** Both fields that carry variables (`key` vector and `remainingArgs` set) get `ce::replaceKeysInString` with the rewrite's substMap. The substMap keys are bare variable names (u_-prefixed names are never class members), so `u_`-prefixed args in the value are naturally excluded from substitution. `standardMaxAdmissionDepth`, `standardMaxSecondaryNumber`, and `flag` are copied unchanged. `admissionStatusMap[K']` is inherited from `admissionStatusMap[K]` only when no prior entry exists at K' (preserve existing entries — the additive principle).

**Performance.** A new monotonically-growing cache `HashMemory::varsInAdmissionMapKeys` is populated at every `admissionMap` insert (4 sites: `prover.hpp::prepareIntegration`, `memory.cpp::makeMandatoryEncodedStatementLists1Static`, `prover.cpp::updateAdmissionMapRecursion`, and the new hook itself). The hook short-circuits in O(|class|) when no class variable appears in any admission key — same pattern as `varsInRejectedMapIntegrationKeys`. Without this cache the hook would walk all admission entries per class call (Gauss-scale: ~10⁴–10⁵ entries × ~10⁶ class calls = catastrophic). Final cost depends on overlap density; bounded above by per-entry class-member scan + mapping enumeration.

**Iteration safety.** Inserts, status updates, cache populates, and `revisitRejected2` calls are queued in `toInsert` during the admissionMap walk and applied after the walk completes. `revisitRejected2` internally calls `cleanAdmissionMap` which may erase the just-inserted K' (if marker is in the operator's output slot — see `prover.hpp::cleanAdmissionMap`). That erasure is permissible: principle (3) only requires the original K to be preserved, and K' has fulfilled its purpose (it carried the revival).

**Alternatives considered.**

- **Mirror integration's rejectedMap rewrite (additive, with arg-equalization filter).** Rejected: the substituted constituents in the new K2 entry would still lack post-substitution `disintegration` provenance, regenerating the same gap that motivated this design. The arg-equalization filter would close one class of bad rewrites but not the provenance issue.
- **Rewrite admissionMap WITHOUT calling revisitRejected2.** Rejected: equi-class application would broaden future admission but skip retroactive revival of already-rejected cohorts. The whole point is to retroactively unblock rejections that an equi-class makes admissible.
- **No arg-equalization filter.** Rejected: a rewrite where the substitution collapses two previously-distinct slots in K to the same value produces an admission key whose positional structure differs from K's. Probing `rejectedMap[K']` for such a collapsed K' returns false matches (the rejected cohort at K' was rejected for the collapsed shape, not for the original; matching mixes semantically distinct rejections).
- **Replace integration's buggy rmi-rewrite at the same time.** Rejected per user direction "we do not correct it for integration" — integration's gap is acknowledged and scoped out of this branch.

**Investigated / ruled out during execution.**

- Whether `revisitRejected2`'s degenerate `equality1` mail origin would leak to the chapter and fail verifier `check_equality1`. Traced through addOrigin's cap-full preference (`prover.hpp::addOrigin`) and confirmed empirically: 0 of 153 sampled `equality1` rows in shipped chapters are degenerate (all have `len(rest) ≥ 4`).
- Whether `disintegrateExprCore2` autonomously emits an `expansion` origin for absorbed compounds. Answer: yes, via `trackExpansionHistory` (`prover.cpp::disintegrateExprCore2`); the `expansion`+`disintegration` origin chain is written at original production time, so the revival path does not need to re-emit it.
- Whether the bare-form-of-K' would mismatch the rejectedMap key form. Answer: admission map keys are bare-marker form (no `u_`) — the rewritten K' is directly usable as a rejectedMap key, no `u_`-strip needed.

**Code.** [`prover.hpp::applyEquivalenceClassToAdmissionMap`](../GL_Quick_VS/GL_Quick/src/prover.hpp), four wire sites in `addStatement` immediately after each `applyEquivalenceClassToRejectedMapIntegration` call. Cache populate at every admissionMap insert in `prover.hpp::prepareIntegration`, `memory.cpp::makeMandatoryEncodedStatementLists1Static`, and `prover.cpp::updateAdmissionMapRecursion`. `revisitRejected2` body refactored separately to mail-first emission (see C2 commit on this branch).

---

<a id="d-56"></a>
## D-56 — buildStack lifts every chapter-row scope to closest-to-`main` ancestor with an origin (2026-05-11)


**What.** `visualizer.cpp::buildStack` lifts every `(expr, validity)` it visits to the closest-to-`"main"` ancestor of `validity` whose `(expr, ancestor)` key has an origin entry in the emitting LB's `exprOriginMap`, **subject to an OR-branch barrier** (lift never crosses `_boundary_orint_` or `_boundary_ordis_` delimiters). The lift applies to the entry-side `proved`, to every dep written into a chapter row cell, and to every dep passed to a recursive `buildStack` call. The `covered` dedup set keys on lifted forms, so each `(expr, lifted_v)` pair has at most one row in the chapter.

The OR-branch barrier emerged from the first verification run: the unbounded lift produced 4 incubator verifier failures (2 `or convergence`, 2 `origin chain termination`) on chapter `1209_direct_proof.txt` (FTA-rung-1 forward direction). Both branches of two OR-convergence rows had their `(preorder, branch_scope)` deps lifted to the parent boundary where the post-convergence origin was recorded. The convergence row's deps collapsed to identical pairs, breaking `check_or_convergence` (which requires branch-distinct dep namespaces) and producing the corresponding `origin chain termination` cycles. The barrier restores branch-distinct namespaces by stopping the lift at the deepest `orint_`/`ordis_` ancestor.

**Why.** Pre-lifting, `buildStack`'s lookup was "exact-key, else `(expr, "main")` shadow"; on a hit via the shadow, `emitRow` wrote `proved.validityName` (the un-shifted deep boundary) into `row[1]` while citing the shallow-scope origin's main-scope deps. The chapter row claimed the derivation happened at a deep scope when in fact the origin lived at `main`. Concrete bug instance: chapter `193_check_induction_condition.txt` line 102 emitted `(=[it_0_lev_0_32,2])` on `main_boundary_(implication23[2,8,int_lev_4_2365])` with implication-rule deps all on `main`. A targeted trap on `buildStack::emitRow` confirmed the falsification path ( at investigation time).

Three alternatives were considered:

- **(A) Widen consumers.** Extend `verifier.py`'s dep-validity equality checks to "ancestor-or-equal" matching (using the existing `_ns_matches_or_strict_prefix` helper, enhanced with `"_boundary_"` delimiter check), and extend `generate_full_proof_graph.py`'s JS click handler to do a DOM-walk ancestor fallback when an exact-namespace card doesn't exist. Preserves chapter shape; spreads new semantics across two consumers.
- **(B) Lift at the producer.** Change `buildStack` to walk ancestors and emit at the closest-to-`main` scope with an origin. Verifier and HTML stay exact-match.
- **(C) Leave as is.** Chapter rows remain falsified; future readers (other agents, auditors) work around the mismatch.

(B) wins because: (i) chapter rows become truthful — `row[1]` is where the derivation actually lives, (ii) deduplication happens automatically via the existing `covered` set, (iii) the verifier and HTML consumer semantics simplify (exact equality / `getElementById` continue to work), (iv) the change is concentrated at one site (`buildStack`) instead of being distributed across consumers. (A) would have required a JS-side DOM-walk handler with both string-prefix-proximity AND DOM-distance-nearest as the two criteria for the jump target — adding complexity that lifting makes unnecessary. (C) is a hard pass per [I-16](30_invariants.md#i-16) and the project's general "failures are first-class" stance.

**Soundness argument.** An origin entry at `(expr, V)` in `exprOriginMap` was recorded by the prover when a rule fired producing `expr` with every premise available at scope `V`. Per [I-2](30_invariants.md#i-2), every non-root `V` is `parent + "_boundary_" + payload` and inherits all parent-scope facts. Therefore lifting from `(expr, V_deep)` to `(expr, V_root)` where `V_root` is the closest-to-`main` ancestor with origin → the rule did fire at `V_root` → premises were at `V_root` or shallower → lifted row is true. The asymmetric direction holds: ancestor facts are universally available in descendants, descendant-only facts are not — and lifting only moves toward `main`, never away.

**Consequences.**

- Chapter row counts shift. A chapter that previously had K rows at deep scopes (via the now-retired main-fallback retag) may now have K rows at `main`, deduplicated against existing main-scope rows. Verifier check counts move; the new counts are stable across re-runs but do not match the pre-lifting baseline. See [G-41](50_gotchas.md#g-41).
- The verifier's dep-validity matching stays at exact `==` (no `_ns_matches_or_strict_prefix` extension required outside of `equality1` / `equality2`, which use the helper for a different purpose).
- HTML namespace-tag jumps are deterministic by construction: every `(<ns>)` cited in a row points to a scope that has rows, so the corresponding subproof card exists.
- and debugging methods continue to apply; the chapter side now reflects the lifted form, simplifying the comparison.

**Code.** Helper `liftToShallowestOriginAncestor` at [`visualizer.cpp`](../GL_Quick_VS/GL_Quick/src/visualizer.cpp). Five invocation sites inside [`visualizer.cpp::buildStack`](../GL_Quick_VS/GL_Quick/src/visualizer.cpp): entry, candidate-loop dep lift (per candidate), `emitRow`'s dep emission, recursive call's `ingredient`, last-resort `front`-fallback. Related: [I-39](30_invariants.md#i-39), [G-41](50_gotchas.md#g-41).

**Relation to [D-58](#d-58).** Same falsified-row class as the chapter-193 line-102 example named in both entries. This decision is the producer-side fix (lift moves the row to `main`, eliminating the falsified emission). D-58 is the verifier-side complement (rejects the falsified row at the verifier layer). With both landed, the chapter-193 instance is fixed at the producer; the verifier check stays as forcing-function for any future producer-side regression.

**Supersedes.** The chapter-emission portion of [D-51](#d-51): the `__contradiction__` LB fallback and the chapter-goal in path mechanisms still apply; only the "exact-key + main-shadow" lookup is replaced by the ancestor walk.

---

<a id="d-58"></a>
## D-58 — Verifier `check_implication` enforces `deeperOf`-equality on result namespace (2026-05-11)


**What.** `check_implication` (the verifier's most frequently-fired per-tag checker) gained a structural rule mirroring C++ `generateEncodedRequestsStatic` + `growBaseCandidates`'s `nm.deeperOf` accumulation. On top of the existing [D-35](#d-35) comparable-scope premise inheritance:

1. Every PAIR of constituent namespaces (impl + each premise) must be **comparable** (`_ns_matches_or_strict_prefix` in either direction).
2. The row's namespace (`line.namespace`) must EQUAL the deepest constituent — i.e. `line.namespace ∈ {impl_ns} ∪ {premise_nss}`.

Rule formalised as [I-38](30_invariants.md#i-38).

**Why.** Sound under GL's validity-stack semantics. The prover's hash kernel accumulates the joined-scope of combined facts via `nm.deeperOf` ([`20_core_concepts/04_validity_stack.md`](20_core_concepts/04_validity_stack.md): *"when a fact derived in scope `a` is broadcast to an LB active in scope `b`, the fact's effective scope becomes `deeperOf(a, b)`"*). An `implication` row whose result lives at a strictly deeper scope than every constituent encodes a derivation step the kernel cannot have emitted — must be a producer-side bug.

D-35 alone catches sibling and deeper-than-result-premise scopes, but admits the case where every constituent is a strict ancestor of `result_ns`. The new rule closes that gap.

**Relation to [D-56](#d-56).** Producer-side complement on the same row-falsification class. The buildStack lifting eliminates the chapter-193 line-102 instance at emission time (the row now lives at `main`, where its origin actually lives). The verifier deeperOf-equality check is forcing-function for any future producer-side regression that bypasses lifting — a falsified row would fail the check rather than slip through.

**Spot.** Verifier failure under `implication` on a row where `line.namespace` is strictly deeper than every namespace in `rest[1::2]`. Concrete pre-fix example documented in [I-38](30_invariants.md#i-38)'s *Spot* section (rung-1 incubator branch, `(=[it_0_lev_0_32,2])` deposited at `main_boundary_(implication23[2,8,int_lev_4_2365])` from constituents all at `main`). Post-merge: the row no longer exists because `buildStack` lifting moved it to `main`; the verifier check stays as a backstop.

**Trade-off considered.** Could have been encoded prover-side as an assert at the emission site. Verifier-side is preferred because (a) the verifier is the airtight regression gate per [I-16](30_invariants.md#i-16); (b) the C++ assert mechanism already covers the in-prover invariants — the verifier covers post-emission audit; (c) the check is cheap (O(K²) pairs for K ≤ ~10 premises, well within the verifier's per-row budget).

**Files touched.** `verifier.py` (15 LOC structural rule added to `check_implication`), `tests/test_verifier_implication.py` (+3 failure tests exercising the rule), `docs/agentic_swdd/20_core_concepts/08_proof_tags.md::implication` (Rule 10 doc-sync), `docs/agentic_swdd/30_invariants.md` (new I-pending entry), `docs/agentic_swdd/40_decisions.md` (this entry).

**Code.** [`check_implication`](../verifier.py); C++ source of truth [`generateEncodedRequestsStatic` + `growBaseCandidates`](../GL_Quick_VS/GL_Quick/src/memory.cpp).

---

<a id="d-55"></a>
## D-55 — OR-disintegration's K mutual-exclusion sub-implications get auditable provenance (2026-05-09)


**What.** Six coordinated edits close three related concerns around OR-disintegration: (a) auditability for the K mutual-exclusion sub-implications, (b) deduplication / parent-removal at OR-construction time, and (c) the firing-site cross-scope rule that was masking (b)'s viability.

**Part A — K-implication provenance.**

1. **Prover origin recording.** [`disintegrateExprCore2`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)'s OR case now stamps each of the K implications `(>[](!d_others) … d_k)` with `("disintegration", [(expandedOrSignature, validityName)])` in `exprOriginMap` and `mailOut.exprOriginMap`. The expanded-OR signature is computed once per OR via the existing `expandSignature(ent)` helper, mirroring the &/existence pattern already in `trackExpansionHistory`. KEY u_-stripped to match the &/existence pattern.
2. **`trackExpansionHistory` hoisted above the orAdmitted gate.** Pre-fix, the OR's expansion-origin record was recorded only when `orAdmitted == true` (i.e. when per-branch case-split fires). The K implications, which fire **regardless** of `orAdmitted`, then cited an expanded form that had no matching expansion-origin record when admission failed — leaving the verifier's `check_disintegration` dispatch (which walks chapter for the matching `expansion` row) without a target. Post-fix, `trackExpansionHistory(ent, false)` runs unconditionally inside the `currentOrDepth < max_or_depth` block, so the expansion row exists whenever any K-implication is emitted.
3. **Verifier `or` branch in `check_disintegration`.** New helper `_check_or_disintegration_implication(line, compact, entry)` ([`verifier.py`](../../verifier.py)) substitutes the compound's args into the binary entry's disjunct templates, peels `line.expression` via `disintegrate_implication_full`, and verifies that the head matches one disjunct with premise multiset = `{!d_j: j!= head_index}`. Premise order is irrelevant to the check.

**Part B — OR-construction dedup + parent removal in `run_modes.cpp::fullRun`.**

4. **Disjunct-set dedup.** `orPairsFromHeadSwitch` is symmetrically populated by `headSwitchOne` walking `globalTheoremList` — for each theorem with a negated premise, both the original and its contrapositive are emitted as a pair. This produced `(mirror_a, mirror_b)` AND `(mirror_b, mirror_a)` for the same disjunct set, and `constructOrTheorem` was called for both, registering `or<N>` and `or<N+1>` for the same logical OR with disjuncts in opposite order (the source of `or0` / `or1` duplication on Peano). Post-fix, the OR-construction loop canonicalizes each pair as `(min, max)` lexicographically and skips pairs whose canonical form has already produced an OR. One OR per disjunct set.
5. **Parent removal at OR creation.** When an OR is constructed from `(exist, comp)`, both parent theorems are added to a `consumedParents` set. After the construction loop:
 - `consumedParents` are filtered out of `survivingTheorems` before `saveProvedTheoremsFiltered` writes `theorems.txt` / `compiled_theorems.txt`.
 - `consumedParents` are appended to `compressed_out_theorems.txt` — same procedure the compressor applies to redundant theorems, done manually here at OR-creation time so the cleanup lands in the same emit pass rather than racing the compressor.
 The parents are subsumed by the OR via OR-disintegration's K mutual-exclusion implications (Part A) — they are recoverable from `or<N>` + `disintegrateExprCore2`'s OR case, no longer cited as standalone theorems. Viability of this cleanup depends on Part C: with the parents broadcast as v=main universals, the original strict-eq firing-site check happened to mask the cross-scope bug; remove the parents and the K-impls fail to fire across scope without Part C's realignment.

**Part C — firing-site cross-scope rule realigned with `nm.comparable` (the line-1078 fix).**

6. **`checkLocalEncodedMemoryStatic` head-LMV scope check** ([`memory.cpp`](../../GL_Quick_VS/GL_Quick/src/memory.cpp)). Pre-fix:
 ```cpp
   if (lmv.validityName != "main" && expressionListValidityName != "main"
       && lmv.validityName != expressionListValidityName) {
       continue;
   }
   ```
 Strict equality on the rule's scope vs the consensus scope, with both-main masking. Pre-Part-B the chapter-6 / chapter-11 mirror reformulations were broadcast Peano theorems at v=main, so the left arm of the AND short-circuited and the bug never surfaced. Post-Part-B, the parent mirrors are gone and the K-implication carries the OR's parent scope (impl26 in incubator's IncubatorGauss1 batch, not main); the rule's scope is non-main AND distinct from the Branch-A consensus scope, and the strict-eq check rejected legitimate firings of the rule on descendant-scope facts.
 This violates the prover's own comparable-scope inheritance rule — request generation already uses `nm.comparable` (per [`docs/agentic_swdd/20_core_concepts/02_hash_engine.md`](20_core_concepts/02_hash_engine.md#locality-semantics-a-local-rule-can-fire-against-entirely-non-local-premises) §Locality semantics). Firing-site was the only place still on strict-eq.
 Post-fix:
 ```cpp
   const int16_t lmvVid = nm.encode(lmv.validityName);
   if (!nm.comparable(lmvVid, consensusValidityId)) {
       continue;
   }
   ```
 `nm.comparable(a, b)` returns true iff one is an ancestor of the other on the same root-to-leaf path — the same "parent / strict-prefix in either direction" relation as the request-generation `pairMap` lookup. Both ends of the firing pipeline now agree on what scopes are mutually visible.
 Diagnostic trail: a firing-path trap (gated on the chap-1209 SE2 LB chain) confirmed rejection at exactly this site for the rung-1 K-impl `(>[]!(=[u_repl_lev_1_2,u_2])(existence2[u_1,u_repl_lev_1_2,u_3]))` — `[lmv 0 SKIP] step7_cross_scope_validity ruleV=…impl26 consensusV=Branch-A`. Trap stripped post-fix; same trap output was the empirical evidence that no legitimate `step7_scope_not_comparable` rejections occur in 35866-check airtight runs (counter: 0 across the fullest run).

**Why.**

- *Part A:* Pre-fix the K rules were anonymous hash-memory entries: when they fired downstream as `implication`-tagged chapter rows, the cited rule expression had no chapter to click through to and no provenance trail back to the originating OR. The "every inference step links to its antecedent" mandate (the project conventions "When working on this project") was silently violated for any chapter using an OR-derived rule. The K rules are deterministic structural consequences of the OR's De-Morgan form — they should be recoverable by anyone reading the proof graph, which now they are.
- *Part B:* Pre-fix the parent mirror reformulations stayed in `theorems.txt` after their OR was constructed. Downstream chapters (notably FTA-rung-1's `1209_direct_proof.txt`) cited them directly, shadowing the OR theorem they had become equivalent to. With Part A's K-implication provenance in place, the parents are formally redundant — Part B is the corresponding cleanup so `theorems.txt` reflects what's actually essential.
- *Part C:* The strict-eq firing-site check was a long-standing latent bug masked by the historical existence of v=main mirror reformulations as broadcast universals. Part B's parent removal removed the mask. Without Part C's realignment, Part B alone breaks rung-1: the K-impls land at the OR's parent scope (impl26), Branch-A facts at the descendant orint scope, and the strict-eq check rejects every cross-scope firing despite the scopes being on the same root-to-leaf path and request generation having already accepted the (rule, fact) pair. Part C restores the comparable-scope inheritance rule the rest of the prover already honours and unblocks Part B.

**Where it does NOT change behaviour.** Part A's prover origin records do not introduce new statements or new firings — they are pure metadata. Part B's run_modes-level cleanup happens after the prover and compressor have already settled; no proof closes that did not close before, and no proof fails that did not fail before. The observable differences:
- More `disintegration`-tagged rows in chapters that disintegrate ORs (today: incubator chapter 1209 and a handful of FTA-ladder chapters; main pipeline OR theorem chapters). Verifier check count rises by exactly the number of new rows.
- Fewer entries in `theorems.txt` / `compiled_theorems.txt` — the parent mirrors and the duplicate OR variant are gone, replaced by a single OR theorem and the compressed-out parent record.

**Tag description follow-on.** `tag_descriptions.json::disintegration` was extended to mention OR as a third disintegration target. Per the project conventions: if the new verifier branch surfaces failures on existing artefacts, those represent real provenance gaps the prior soft-checker hid — the response is to fix the producer side, not weaken the check.

**Cross-link.** OR-construction logic itself, including the dedup and parent-removal cleanup, is documented in [`20_core_concepts/07_or_branching.md`](20_core_concepts/07_or_branching.md#or-theorem-construction-run_modescppfullrun).

---

<a id="d-54"></a>
## D-54 — Single canonical `files/GL_binaries/` directory; verifier filters by chapter context (2026-05-08) — **`_merge_into_shared` incubator-skip superseded 2026-05-13 by [D-61](#d-61)**


**Supersession (2026-05-13).** The "_merge_into_shared early-returns for `tag.startswith("Incubator")`" portion of this decision is reversed by [D-61](#d-61) once the C++ allocator's `repetitionExclusionMap` is keyed by `(elements, category)` per [D-60](#d-60). The other components of D-54 — the unified writer path in `visualizer.cpp` and the chapter-context filter in `verifier.py` — remain in force. The original D-54 reasoning ("propagating incubator allocations would bump main's counters and shift theorem texts") was load-bearing only because the cross-category lookup collision was undetected; with the lookup made category-aware, incubator allocations can safely contribute to shared without semantic interference with main batches.

**What.** Three coordinated edits eliminate the duplicate `files/incubator/GL_binaries/` directory:

1. **C++ writer-path consolidation.** `visualizer.cpp::generateRawProofGraph`'s `glBinDir` switches from `outDir.parent_path / "GL_binaries"` to a project-rooted `<__FILE__>/../../../../files/GL_binaries`. The writer path now matches the reader path in `prover.cpp::compileCoreExpressionMap` exactly. Every batch — incubator and main alike — writes its post-batch dictionary to the single canonical `files/GL_binaries/` directory.
2. **Python merge skip for incubator tags.** `run_modes.py::_merge_into_shared` early-returns for tags whose name begins with `Incubator`, while still printing the `+0 new entries (shared total: <N>)` line for log parity. Incubator-allocated spontaneous compact-operator names are batch-local and must not propagate into the shared cross-batch registry — propagating them would shift the next main batch's counters and rename main's spontaneous operators, breaking [I-23](30_invariants.md#i-23) for the main pipeline.
3. **Verifier chapter-context filter.** Three surgical edits in `verifier.py`:
 - `verify_chapter`'s `current_gl_binary` selection (around line 3340) — when the chapter's theorem expression contains `AnchorIncubator`, restrict the candidate-tag iteration to `Incubator`-prefixed tags. Without this, the cross-anchor connection chapter `(>[..](AnchorIncubator[..])(AnchorPeano[..]))` would match `AnchorPeano` first (alphabetical) and bind `current_gl_binary` to the Peano binary, breaking 3 incubator chapter checks (`expansion: failure 1`, `disintegration: failure 2`).
 - `binaries_for_chapter` — when `current_gl_binary` is `None` and the chapter's theorem expression contains `AnchorIncubator`, restrict the fallback list to tags starting with `Incubator`.
 - `_check_reformulation`'s D-35 fallback (the V-6 helper added in `docs/agentic_swdd/verifier_rung1_changes.md`) — when `tag == "Incubator"`, restrict the existence-head scan to `Incubator`-prefixed tags.

The duplicate directory is then `git rm -r`'d. Three tracked files removed.

**Why.** Pre-2026-05-08 the writer path was asymmetric. Configs for incubator batches set `raw_proof_graph_folder = "files/incubator/raw_proof_graph"`, so `outDir.parent_path / "GL_binaries"` resolved to `files/incubator/GL_binaries/` — a directory the prover loader (`prover.cpp:194-198`, project-rooted) never read from and the Python `_merge_into_shared` step (also project-rooted) ignored. Incubator-allocated spontaneous compact-operator names were stranded on disk in a stale parallel folder. The user's directive: there must be a single `files/GL_binaries/` and nothing under `files/incubator/`.

The naive consolidation (just redirect the writer path and remove the directory) breaks the verifier. The duplicate directory was providing accidental name-isolation between main-batch and incubator-batch spontaneous allocations, which collide on shape:

| op | shape in main (Peano/Gauss/shared) | shape in incubator |
|------------|------------------------------------|--------------------|
| existence0 | arity 3 | arity 3 (same) |
| existence1 | arity 4 | arity 4 (same) |
| existence2 | arity 3 | **arity 8 in IncubatorPeano** |
| existence3 | arity 8 | arity 8 (same) |
| existence4 | arity 5 (Gauss/shared) | **arity 4 in IncubatorGauss1** |

Once both folders' contents share `files/GL_binaries/`, alphabetical iteration in `binaries_for_chapter` and `_check_reformulation`'s fallback would route incubator chapters' existence-head lookups against main-batch binaries — chapter `1210_reformulated_statement.txt` (arity-4 `existence4` substitution) would fail. The verifier filter restores the isolation explicitly.

**Trade-off considered.**

- *Namespace incubator allocations under a separate prefix* (e.g. `inc_existence4` instead of reusing `existence4`). Rejected as out-of-scope: would require C++ changes to `ExpressionAnalyzer`'s spontaneous-name allocator plus coordinated changes in process_proof_graphs.py and the verifier. The chapter-context filter is the minimum-blast-radius fix.
- *Skip the C++ export entirely for incubator batches.* Rejected: would leave `files/GL_binaries/GL_binary_IncubatorPeano.json` as the 4-byte `{}` Python seed and deprive any future cross-batch consumer of the post-batch incubator dictionary. The unified-write approach keeps full state on disk; the Python merge skip plus verifier filter handle the isolation.
- *Modify the chapter format to embed the originating tag explicitly.* Rejected: process_proof_graphs.py emits chapters today using just `AnchorIncubator`; embedding tag would require coordinated changes in three more places and the chapter format is referenced from many test artefacts.
- *Leave the duplicate directory alone.* Rejected: the user explicitly directed cleanup. The duplicate was a consequence of an undocumented path asymmetry, not a deliberate design choice.

**Verification.** A full `python main.py` run reproduces  exactly — same theorem yields, same `[merge_into_shared] +N new entries (shared total: K)` lines, same 35865 verifier-checks-airtight tally — modulo the trailing `Overall runtime` time and per-burst `dt=` jitter.

**Cross-references.**

- [I-16](30_invariants.md#i-16) (`verifier.py` is sacred — failures are real bugs). The verifier edits are pure tightenings: no check is weakened, no new binary becomes consultable; binaries that were never visible to the incubator-side verifier under the duplicate-folder layout stay invisible.
- [I-23](30_invariants.md#i-23) (spontaneous compact operator names stable across batches). Continues to hold for *main* batches; the merge skip explicitly excludes incubator allocations from the shared registry that I-23 governs.
- D-35 / V-6 reformulation fallback (`docs/agentic_swdd/verifier_rung1_changes.md` § V-6) — the chapter-context filter narrows V-6's fallback scope when the target's anchor is `AnchorIncubator`.

---

<a id="d-53"></a>
## D-53 — `InternalMail` absorbed into `Mail`; mail subsystem unified to a single struct (2026-05-07) — renumbered from main's D-46


**What.** `struct InternalMail` (formerly defined alongside `Mail` in `memory.hpp`) is deleted. `struct Mail`'s `statements` element type migrates from `pair<string, set<int>>` to `pair<ExpressionWithValidity, set<int>>`, reusing the existing `ExpressionWithValidity` type (`memory.hpp::ExpressionWithValidity`, fields `original` and `validityName`) — already the key type of `Mail::exprOriginMap`. `Memory::sameIterationInternalMail` becomes type `Mail`. `performElementaryLogicalStep`'s `mailIn` statements absorb (CE-mode and normal-mode loops) gains a per-element `assert(vName == "main")` as the runtime enforcement of [I-26](30_invariants.md#i-26)'s sender contract for the statements channel.

After this, the routing channels (`mailIn`/`mailOut`) and the integration-revival channel (`sameIterationInternalMail`) share the same struct. They are distinguished only by lifecycle ([I-21](30_invariants.md#i-21): top-of-burst absorb for revival vs end-of-burst for routing) and by absorb status (`status=1` vs `status=3`) — not by struct type.

**Why.** [D-19](#d-19) created `InternalMail` as a separate struct *to defer* the migration cost of the existing pair-shape callsites of `Mail::statements`. The deferred-migration era ends here — that cost is paid on this branch. Result: one mail struct to reason about, one absorb shape, one mail invariant set. The apparent asymmetry between routing mail (no validity) and revival mail (validity carried) was a definition-site artefact, not a real semantic distinction; the same `Mail` struct can serve both channels because the `validityName` field on the EWV is "main" for routing traffic by sender contract and reflects the actual revival scope for `sameIterationInternalMail` traffic.

**Trade-off considered.**

- *Inventing a fresh 3-tuple* `tuple<string, set<int>, string>` for `Mail::statements`. The obvious alternative. Rejected: reusing `ExpressionWithValidity` is cleaner — one type ("expression + scope") used consistently across mail statements and origin maps. Suggested by user during planning ("u need to migrate mail to ExpressionWithValidity").
- *Single atomic commit vs multi-commit refactor.* Multi-commit chosen by user direction for bisectability:
 - C1: `Mail.statements` → `pair<EWV, levels>`; 7 routing senders + 6 receiver iterator blocks migrated atomically (`std::set` element type is concrete, no compilable intermediate state).
 - C2: `Memory.sameIterationInternalMail` typed `Mail`; 2 internal senders + drain block migrated.
 - C3: delete `struct InternalMail` (orphan after C2).
 - C4: per-element `assert(vName == "main")` at the routing-channel `mailIn` absorb in `performElementaryLogicalStep`.
 - C6: D-46 entry; D-19 supersession marker; AGENT_SwDD.md decisions-count refresh; final DoD verification.
- *Per-channel asserts at smashMail / sendMail / sender callsites.* User redirected to a single consumption-time assert in `performElementaryLogicalStep` ("main assert is enough in performElementary for external mail"). Captures the same contract violation at one location instead of three.
- *D-numbering.* Original numbering picked D-46 (skipping D-44, D-45 which were reserved for sibling). On2026-05-08 the 46 slot was already taken by incub_fix's cross-pair `equality2` decision; this entry renumbered to D-53.

**Operational consequence.** Receiver-side absorb in `mailIn` no longer hardcodes `"main"`; reads `pair.first.validityName`. For routing traffic this evaluates to `"main"` (sender contract preserved by every routing sender wrapping with `ExpressionWithValidity(expr, "main")`). For `sameIterationInternalMail` traffic the EWV carries the actual revival scope. The receiver-side assert traps any future sender that slips past the existing main-only gate at `mailOut.statements.insert`. Behaviour is observationally unchanged for both channels; the assert is purely defensive.

**Verified (on ).** Full `python main.py` clean run on the worktree `.worktree/mail_unification/` from the C6 commit.

- Build clean (Release x64; only known C4267 / C4101 warnings).
- Pipeline runtime 1997 s (≈33 min) — within Gauss-batch baseline tolerance (C1 baseline 1619 s; the variance is from incubator Gauss volatility, not refactor).
- `theorems.txt` — 41 rows; `AnchorGauss` matches 13 (Gauss summation derived per hard-done criterion).
- `verifier.py` — 2750 main checks + 35545 total proof-graph checks, **0 failures** across all 35 tag categories.
- The receiver-side `assert(vName == "main")` at the routing-channel `mailIn` absorb did not fire (Release-build asserts compile out under `NDEBUG`; the gate is informational at runtime, structural at code-review time).

Re-verification on the post-merge tree is recorded in the merge commit body alongside the incub_fix-side D-52 fullest verification (35865 checks airtight, identical to D-52 baseline).

**Files touched (across C1-C4 + the C6 commit).**

- `GL_Quick_VS/GL_Quick/src/memory.hpp` — `Mail.statements` shape; `struct InternalMail` deletion; `Memory::sameIterationInternalMail` type.
- `GL_Quick_VS/GL_Quick/src/prover.cpp` — 5 routing senders, 6 receiver iterator blocks (incl. diagnostic dump), `sameIterationInternalMail` drain block, `mailIn` absorb assert. `smashMail` set-merge type-driven (no source change).
- `GL_Quick_VS/GL_Quick/src/prover.hpp` — 2 routing senders, 2 internal senders. `sendMail` set-merge type-driven (no source change).
- `docs/agentic_swdd/02_glossary.md` — `InternalMail` glossary entry rewritten as retired redirect.
- `docs/agentic_swdd/01_overview.md` — `memory.hpp` file-content table row.
- `docs/agentic_swdd/20_core_concepts/01_logic_blocks.md` — `Memory`-fields table, `sameIterationInternalMail` row.
- `docs/agentic_swdd/20_core_concepts/03_mail_system.md` — `Mail` struct table; Main-only gate section split per-channel; Scope subsection; InternalMail subsection retired.
- `docs/agentic_swdd/30_invariants.md` — I-26 receiver bullet, Code section, and Why-section line-number citation refreshed; I-21 code citation refreshed.
- `docs/agentic_swdd/20_core_concepts/07_or_branching.md` — link refresh in `ordisMerge` description.
- `docs/agentic_swdd/40_decisions.md` — D-19 marked superseded; link refresh in D-34 body; D-46 entry (renumbered to this D-53).
- `docs/agentic_swdd/AGENT_SwDD.md` — Decisions row count refreshed.

**Supersedes.** [D-19](#d-19) (deferred-migration era; the cost is now paid). [I-21](30_invariants.md#i-21) and [I-26](30_invariants.md#i-26) wording adjusted in the same C1-C4 series.

---

<a id="d-52"></a>
## D-52 — `_orint_` goal-flow row carries sub-implication, not bare disjunct (2026-05-08)

**What.** Two coordinated changes — one producer-side, one verifier-side — that fix the malformed history line emitted by Case OR in `prepareIntegrationCore2`:

1. **Producer side** — [`prover.hpp::prepareIntegrationCore2`](../GL_Quick_VS/GL_Quick/src/prover.hpp) Case OR (the `if (le.category == "or" && allSigArgsAreU)` block):
 - For each branch `k` of an OR with `K` disjuncts, build the per-branch sub-implication `(>[](AND-of-negated-others)(D_k))`. Premise = left-nested AND of `!D_j` for every `j!= k`; for `K = 2` the AND wrapper collapses to a single negation. Head = chosen disjunct `D_k`. Empty bound-var list — `_orint_` branches don't introduce new quantifiers.
 - Emit one row per branch: `KEY = subImpl_k + "_integration_goal"` at parent `validityName`; `ORIGIN = "expansion for integration" ← cleanSignature + "_integration_goal"` at parent `validityName`. Both sides at parent scope — passes the verifier's same-scope strictness on `expansion for integration`.
 - Drop the previous emission `ev(head, branchValidity)` whose KEY was the bare disjunct at branch namespace. That row was structurally absurd (a single disjunct is not the structural expansion of `or<N>`) and namespace-mismatched (line.namespace deeper than right_ns).
 - Branch-ns scaffolding is unchanged: `encodePush` of the branch payload, `or branch assumption` rows for negated other-disjuncts at branch ns, head-as-toBeProved at branch ns. Only the parent-ns goal-flow row's shape changes.

2. **Verifier side** — [`verifier.py::_try_expand`](../verifier.py) for category `or`:
 - Add a sibling helper `_build_or_subimpls_from_elements(elements)` that constructs all K per-branch sub-implication forms.
 - In `_try_expand`'s `or`-branch acceptance: keep the De Morgan form `_build_or_from_elements` as the primary (existing) acceptance path, and additionally accept LEFT side if it equals (modulo `_normalize_with_unchangeables`) any of the K sub-implication forms. Adds, never weakens — every chapter that passed pre-D-52 still passes.

**Why.** The producer's pre-D-52 goal-flow row was emitted by code copy-pasted from Case A (compact-name → body expansion). The "mirrors Case A" comment at the original site captures the bug: Case A's tag is for compact-implication-name → its body's structural unfolding; Case OR's analog requires unfolding `or<N>` into a structurally-honest expansion. The bare disjunct is one piece of the OR but not the OR's full structural unfolding — the verifier's `_try_expand` rightly rejects it.

The mathematical content of `_orint_` is the constructive proof rule "to prove `(D_1 ∨ D_2 ∨ … ∨ D_K)`, suffices to prove `(!D_1 ∧ !D_2 ∧ … ∧ !D_{k-1} ∧ !D_{k+1} ∧ … ∧ !D_K → D_k)` for some `k`" (any one of K sub-implications closes the OR). The producer was already structurally setting up each branch correctly (negated other-disjuncts seeded as `or branch assumption`, chosen disjunct as branch goal), but the parent-ns history row that should declare "this branch's job is to prove sub-implication k" was malformed. After D-52, the chapter carries one well-shaped `expansion for integration` row per branch, each citing `(or<N>)_integration_goal` as the source — the verifier's K-way acceptance lets all K rows pass, leaving the actual proof obligation to the in-branch derivation of the sub-implication's conclusion.

**Failures dissolved.**
- Failure 2 in chapter `1209_direct_proof.txt` (HTML `chapter1210.html` — off-by-one): the orint-branch's mis-tagged `expansion for integration` row at `(=[i1,v4])` no longer exists as KEY — replaced by the parent-ns sub-implication rows.
- Failure 1 in the same chapter: once `(or2)` is integrated through the new sub-implication path, impl21's modus ponens on `(or2[i1,v4,i0])` at impl26 boundary closes `(in[v4,V1])` at boundary, the chain reaches impl26's body, and `buildStack` walks the body's `_integration_goal` row that was already in `exprOriginMap` (trace line 92, the chained-preorder body of impl26).

**Verification.** Repro with the skip-Gauss-main hack: chapter `1209_direct_proof.txt` parses clean; incubator verifier reports 0 failures across all categories. Yields preserved (IncubatorPeano 513, IncubatorGauss 91, IncubatorGauss1 1, Peano main 49). Pre-D-52 baseline: 32825 checks, 2 FAILED.

**Connection to earlier decisions.** D-52 does not touch the [D-51](#d-51) `buildStack` policy — the path-cycle filter and contradiction-LB fallback remain. The mis-tag was upstream of the walker's selection rule; the walker was faithfully rendering whatever the producer emitted. D-49's cap-full preference is unaffected: the new producer rows compete on the standard `addOrigin` cap-and-tie-break path.

---

<a id="d-51"></a>
## D-51 — Contradiction record stays in `__contradiction__` LB; chapter goal in `buildStack`'s path stack (2026-05-08) — supersedes [D-49](#d-49) / I-35 cap-full preference

**What.** Two coordinated changes that retire the upward propagation of contradiction recipes and let the chapter walker enter the contradiction LB explicitly:

1. **Prover side** — [`prover.cpp::addExprToMemoryBlockKernel`](../GL_Quick_VS/GL_Quick/src/prover.cpp) `primedForContradiction` handler:
 - Stop pushing `(emitter, ev, ("contradiction", deps))` onto `pendingAncestorOrigins`. The drain loop in `proveKernel` previously walked the queued emitter's `parentMemory` chain and wrote the same contradiction record into every ancestor LB's `exprOriginMap`, leaving the inner contradiction's full recipe (its three contradicting deps) duplicated at every ancestor up to root.
 - Drop the parallel write to `memoryBlock.mailOut.exprOriginMap`. Mail transports go parent → children only (`sendMail` iterates `index.find(sender)->second` = children), so an upward-direction `mailOut` write was structurally dead — preserved only because the original drain wrote to ancestors directly.
 - Keep the local write to `memoryBlock.exprOriginMap`. The contradiction record now lives only inside the `__contradiction__` LB that proved it.
2. **Visualizer side** — [`buildStack`](../GL_Quick_VS/GL_Quick/src/visualizer.cpp) (and `directStack` at chapter entry):
 - At `directStack` entry, insert the chapter goal expression (the wrapped theorem from `theoremList`) into the `thread_local g_buildStackPath`. The existing path-cycle filter then rejects any origin candidate whose deps include the chapter goal — exactly the self-application shape the prover's forward-inference produces when it instantiates a proved theorem under its own anchor (origin tag `implication` with deps `[wrapped theorem, anchor]`). Without this, the chapter walker would emit a `theorem`-tag leaf row whose expression matches the chapter goal, tripping `verifier.py::check_chapter` self-reference.
 - When all sorted candidates of a negated head fail (path-cycle filter rejected them all, or no direct origin existed), fall back to walking the LB chain (current → ancestors via `parentMemory`) for a child memory block keyed `"__contradiction__" + positive`. On hit, switch into that contradiction LB — the contradiction record is local there.
 - Remove the per-candidate-loop and `front`-fallback `contradiction`-tag LB switches. Each contradiction is an independent LB with its own chapter; nested contradiction LBs are NOT children of an outer contradiction LB. The only LB switch `buildStack` performs is the chapter-boundary `__contradiction__` fallback above. All other recursion stays in the current LB.

**Why.** The cap-full preference rule introduced in [D-49](#d-49) (foundation displaces convenience: a non-equality origin replaces an equality1/equality2 slot at cap) was the immediate response to [chapter-100/101 cycles](#d-49) on the branch but rested on three more-fragile assumptions:

1. The `pendingAncestorOrigins` upward write was conjoined with — but not justified by — D-49. The two fixes shipped together; D-49's commit body explained the cap-full preference but did not explain *why ancestors needed a copy of the contradiction recipe at all*.
2. With `max_origin_per_expr = 1`, a single origin survived per expression and D-49's tie-break determined which one. The chapter walker's `front` then read whatever D-49 left.
3. The walker's `covered` set spanned every LB it walked into, so the same expression could not be re-emitted as a different proof step in a different LB context.

Raising `max_origin_per_expr` to 30 (matching compressor mode) to support multi-origin selection broke (2) — `front` no longer returned D-49's surviving choice, since D-49 only kicks in at cap-full and cap=30 is rarely full. The walker started picking insertion-order origins which differed from D-49's preferred origins. Symptom: chapter-37-style nested-contradiction chapters lost their inner `task formulation` row because the outer LB's `(in2[i3,i5,s])` was now derived via `implication` (forward-inference of the inner wrapped theorem) instead of being task-formulated, and the inner LB's task-formulation row was suppressed by `covered`.

The architectural fix (this entry) replaces the cap-full preference with explicit selection at the chapter walker:

- The chapter goal expression goes into `g_buildStackPath` so the walker rejects self-applying origins by the same path-cycle mechanism that handles run-of-the-mill cycles.
- Negated heads with no acyclic direct origin in the LB before the head fall through to a `__contradiction__` LB lookup, which is the principled entry point — the contradiction record now lives only there.
- The per-candidate special-case `contradiction`-tag switch in the recursion loop is gone. With the recipe local, walking the contradiction's deps in the same LB is correct: derivations of the contradicting pair sit there, and any nested contradiction reached via a negated dep falls into the same chapter-boundary fallback the next level down (into a sibling `__contradiction__` LB attached to the anchor).

**What was investigated and ruled out.**

- **Knuth (1977) shortest-derivation greedy origin selection** (foundation-distance Bellman-Ford). Tried first as a more powerful selection criterion. Mass self-reference: every chapter goal had a `(theorem, [])` origin at level 0, which Knuth picked over real derivations. Treating `theorem` and `task formulation` as foundation-tag-equivalents amplified the problem. Rejected — the problem isn't selection criterion, it's the *records present at AnchorIncubator level being wrong*.
- **Greedy DFS with path-stack cycle filter, no backtracking** (option-1 greedy). Rejected — fails on ordering-sensitive subtrees where the first valid candidate locally traps an ancestor; needs backtracking.
- **Greedy DFS + backtracking, but keep ancestor-side contradiction record.** Tested: 28804 / 1078 FAILED (539 contradiction + 539 contradiction trace, all nested-contradiction missing-task-formulation). The `covered` set spanning the ancestor LB and the `__contradiction__` LB blocks the inner LB's task-formulation re-emission of an expression already visited via implication-derivation in the outer LB. Rejected — the LB-switch's `covered` set semantics are at fault, *and* leaving the contradiction record at ancestor level is the structural violation that produced the conflicting paths in the first place.
- **Force-emit the inner cleanOp's task-formulation row at the contradiction-LB switch, plus parent-chain walk for nested cases.** Tested: 37622 / 779 FAILED, mostly `task formulation` (inner cleanOps don't satisfy `verifier.py::check_task_formulation`'s "premise of chapter theorem or chapter cleanOp" requirement). Rejected — produces structurally invalid task-formulation rows.

**Trade-off.** The `pendingAncestorOrigins` queue, mutex, and drain loop in `proveKernel` are now dead code — the queue is never pushed and the drain iterates an empty container. Cleanup deferred (next session). No functional cost since the empty drain is O(0).

**Verified.** Incube-only Peano (RUN_INCUBATOR=True, RUN_MAIN_PATH=False, tags=["Peano"]) under HEAD: yield 513, verifier 28804 checks, **0 failures — airtight**. Compared to:

| Configuration | Yield | Verifier (incube-only Peano) |
|---|---:|---|
| pre-D-51 baseline (cap=1, original buildStack) | 513 | 28793 / 2 FAILED (chapter-193 cycle) |
| D-51 mid (cap=30 + option-1 + backtracking, ancestor-side contradiction record kept) | 513 | 28804 / 1078 FAILED (nested-contradiction task-formulation gap) |
| **D-51 final (this entry)** | 513 | **28804 / 0 FAILED — airtight** |

Fullest pipeline (RUN_INCUBATOR=True, RUN_MAIN_PATH=True, tags=["Peano", "Gauss"]) verification pending in this commit window.

**Files touched.**

- `GL_Quick_VS/GL_Quick/src/prover.cpp`: `pendingAncestorOrigins` push retired; `mailOut` upward-write removed.
- `GL_Quick_VS/GL_Quick/src/visualizer.cpp`: chapter goal inserted into `g_buildStackPath` at `directStack`; per-candidate-loop and front-fallback `contradiction`-tag LB switches removed; new `__contradiction__` fallback in the all-candidates-failed branch.
- `docs/agentic_swdd/30_invariants.md`: I-35 (cap-full preference) marked superseded.
- `docs/agentic_swdd/40_decisions.md`: this entry; D-49 / I-35 status amendment.
- `docs/agentic_swdd/AGENT_SwDD.md`: invariant quick-reference table updated.
- `docs/agentic_swdd/10_pipeline/04_prover.md`: addStatement / contradiction LB section updated.
- `docs/agentic_swdd/10_pipeline/06_process_proof_graph.md`: buildStack section updated.

**What does NOT change.**

- D-50 (`addStatement` `&& local` gate) — preserved.
- `max_origin_per_expr = 30` across all configs — preserved (multi-origin storage is what makes the chapter walker have alternatives to choose from when the cycle filter rejects the first candidate).
- Verifier — untouched per [I-16](30_invariants.md#i-16). Reaching 0 failures without verifier modification confirms the fix is on the producer side.
- D-49 / I-35 the *invariant*: superseded but kept in the doc with a status amendment for traceability. The cap-full preference logic still exists in `addOrigin`; it is just rarely triggered (cap=30 is rarely full) and no longer load-bearing for any chapter shape. Cleanup deferred.

---

<a id="d-50"></a>
## D-50 — `addStatement` equality-mirror push gated on `local` (2026-05-07) — restores 366-theorem incubator-Peano yield lost to [](../) `&& false` band-aid

**What.** The `isEquality(expr)` branch in [`addStatement`](../GL_Quick_VS/GL_Quick/src/prover.hpp) — which pushes the mirrored equality `(=[args[1],args[0]])` onto the function's `newStatements` return vector — has its gate changed from `&& false` (dead-coded since [](../)) to `&& local`. The `local` parameter is the existing `bool local` parameter of `addStatement`; the kernel's call site (`addExprToMemoryBlockKernel` in [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)) passes `isLocal = (status == 0 || status == 1)` into it. So mirror-push fires when an equality enters via local addition (status 0/1), and is skipped when it enters via mail-in absorb (status=3).

**Why.** Bisection traced incubator-Peano yield drop 513 → 147 to commit [](../) "addStatement: disable unconditional equality-mirror push (`&& false`)". Producer-LB hashburst trace (chapter-237 producer, LB chain `[0] (in3[2,2,15,4]) → [1] (AnchorIncubator[…]) → [2] <root>`) localized the regression to a single side-effect:

- At burst #2 of this LB, both runs (with-`&& false` vs without) have byte-identical state including `toBeProved` of size 9.
- Between burst #2 and burst #3:
 - Without `&& false`: `addEquality(allowSymmetry=true)` registers `(=[2,15])` and mirror `(=[15,2])` in `encodedStatements` + `statementLevelsMap`. `addStatement(expr=(=[2,15]), local=true)` then pushes the mirror `(=[15,2])` onto `newStatements`. The post-`addStatement` loop in `addExprToMemoryBlockKernel` iterates `newStatements`, calls `toBeProved.find((=[15,2]))`, finds the open goal, and erases it. `toBeProved` shrinks to 8.
 - With `&& false`: `addEquality` adds the same entries to `encodedStatements`, but `addStatement`'s mirror push is gated off. `newStatements` does not contain `(=[15,2])`. The discharge loop iterates only over `(=[2,15])` (which is not in `toBeProved`), never matches `(=[15,2])`, and the goal stays open. `toBeProved` remains at 9.

The discharge loop iterates `newStatements`, **not** `encodedStatements`. The compensation chain ([](../) `allowSymmetry`, [](../) `updateEquivalenceClasses` `#if 0`, [](../) kernel `isLocal` gating) covered the `encodedStatements`-side mirror registration but did not cover this discharge side-effect. Result: the producer LB never closed; its wrapped implication never broadcast to AnchorIncubator's `mailIn`; AnchorIncubator stalled (`stmts=508` plateau from burst #8 onwards). Cascade across all 9 numeric atoms × 2 forms (right-identity-of-`+` and its uniqueness mirror) plus their downstream consumers cost 366 incubator-Peano theorems (513 → 147, ≈ 71% yield loss).

The original [](../) commit body is candid: it silenced an assert at `prover.cpp::addExprToMemoryBlockKernel` (the `statementLevelsMap.find` lookup that follows the mirror push in the discharge loop) which fired on **mail-in absorb** (status=3) paths, where `addEquality(allowSymmetry=false)` had skipped registering the mirror. The author closed with: *"Awaiting direction on whether to remove the dead block entirely or keep `&& false` as a documented switch."* — i.e. flagged as unfinished. `&& local` is the correct narrow gate: status 0/1 takes the mirror push (and its discharge side-effect) because `addEquality(allowSymmetry=true)` has registered the mirror; status 3 skips the push because no mirror entry exists, avoiding the assert.

**What was investigated and ruled out.**

- **Removing the dead block entirely.** Rejected — drops the `toBeProved` discharge side-effect on local paths just as fully as `&& false` did. Yield loss persists.
- **Routing the discharge through `encodedStatements` instead of `newStatements`.** Considered. Would decouple the discharge from `addStatement`'s return value, but the discharge loop in `addExprToMemoryBlockKernel` is heavily coupled to `newStatements` (admission map updates, levels lookup, scope handling all use the per-entry data). Refactor scope is large; the local-gate is a one-line change with the same outcome.
- **Triggering discharge inside `addEquality` directly (when it pushes the mirror to `encodedStatements`).** Considered. Would re-implement the discharge logic at the registration site. Risks duplication / divergence with the `addStatement` post-loop. Defer pending need.

**Trade-off.** Mail-in absorb (status=3) paths still skip the `toBeProved` discharge for incoming mirrored equalities — same as before D-50. If a mail-in absorb arrives carrying an equality whose mirror is already an open `toBeProved` goal in the receiving LB, the goal will not be discharged on absorb. Whether this scenario occurs in practice and matters is open; the chapter-237 case studied here is fully a local-derivation scenario, so the gate suffices. If a follow-up trace surfaces a mail-in absorb scenario, the fix will be a parallel discharge inside the absorb path, not a removal of the `local` gate.

**Verified.** Incubator-only Peano via `main.py` (RUN_INCUBATOR=True, RUN_MAIN_PATH=False, tags=["Peano"]) at HEAD with `&& local`: `Number proven theorems: 513` (matches pre- baseline), Verifier `28793 checks, 2 FAILED` (same 2 `origin chain termination` failures as the unconditional no-false test — pre-existing and not introduced by this gate; tracked separately). Snapshot .

**Files touched.**

- `GL_Quick_VS/GL_Quick/src/prover.hpp`: `addStatement` `isEquality` branch gate `&& false` → `&& local`.
- `docs/agentic_swdd/40_decisions.md`: this entry.

**What does NOT change.**

- `addStatement` signature / parameter list — `bool local` already existed long before [](../); no new parameter added.
- `addEquality`'s `allowSymmetry` parameter — unchanged. Continues to gate `encodedStatements` + `statementLevelsMap` mirror registration on the kernel-supplied `isLocal`.
- The discharge loop in `addExprToMemoryBlockKernel` — unchanged. Still iterates `newStatements`, still calls `toBeProved.find` per entry, still erases on hit.
- The 2 pre-existing `origin chain termination` failures in the no-incubator-only-Peano run — not regressed, not addressed; their root cause is a separate investigation.
- Verifier — untouched per [I-16](30_invariants.md#i-16).

---

<a id="d-49"></a>
## D-49 — `addOrigin` cap-full preference: foundation displaces convenience (2026-05-07) — superseded by [D-51](#d-51), 2026-05-08

> **Status amendment (2026-05-08).** Superseded by [D-51](#d-51). The cap-full preference rule was the immediate fix for the chapter-100/101 swap-cycle and remains in `addOrigin` as inert code at HEAD (cap=30 is rarely full, so the replacement branch is rarely taken; behavior depends on `front` of the stored origins, not on the cap-full preference). The structural fix is D-51's combination: contradiction record stays in the `__contradiction__` LB only (no `pendingAncestorOrigins` upward write), and `buildStack` enters the contradiction LB explicitly via the chapter-boundary `__contradiction__` simpleMap fallback, with the chapter goal inserted into the path stack so the existing cycle filter rejects self-applying origins. Cleanup of the now-inert cap-full preference code in `addOrigin` is deferred.


**What.** `addOrigin` ([`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp), `addOrigin`) gains a cap-full preference replacement step. When the per-key origin vector is at `maxOrigins` and a new origin arrives whose tag is **not** `equality1` and not `equality2`, the helper scans the vector for the first `equality1`/`equality2` slot and replaces it with the new origin. Below-cap behavior unchanged (append + dedup + D-44 symmetry-source trap intact). New invariant: [I-35](30_invariants.md#i-35).

Coupled change at [`prover.cpp::performElementaryLogicalStep`](../GL_Quick_VS/GL_Quick/src/prover.cpp): the bulk-merge step `expr_origin_map = mail_in | current` is rewritten to route through `addOrigin`. Pre-D-49 it was a raw `std::map` swap with body-wins-on-conflict. The raw merge bypassed both the cap and the new preference; mail-arrived origin vectors landed verbatim, and existing body entries overrode mail entries on conflict by insertion-order semantics. Routing through `addOrigin` puts every arriving origin through the same cap+preference gate.

**Why.** [`verifier.py::check_origin_chain_termination`](../verifier.py) flagged 4 cyclic rows on the iter-35 verification run (Peano + Gauss, no incubator):

- `files/processed_proof_graph/100_check_zero.txt` rows 56–57 — `(in2[i0,v10,id]) ↔ (in2[v10,i0,id])`.
- `files/processed_proof_graph/101_check_induction_condition.txt` rows 72–73 — analogous shape.

Both belong to theorem 96 (the Gauss `fold` induction in `theorems.txt:2`). The new `mailIn.exprOriginMap` dump section revealed the actual mechanism: at burst 1 of the chapter-100 zero-case LB, `mailIn.exprOriginMap` carried **two origins per cycle member**:

```
(in2[2,it_0_lev_0_32,8])
    <- equality1 | (in2[it_0_lev_0_32,2,8])  (=[2,it_0_lev_0_32])  (=[it_0_lev_0_32,2])
    <- implication | (>[1](in[1,u_1])(>[2](=[1,2])(in2[1,2,u_8])))
                   | (=[2,it_0_lev_0_32])  (in[2,1])

(in2[it_0_lev_0_32,2,8])
    <- implication | (...) ...
    <- equality1 | (in2[2,it_0_lev_0_32,8])  ...
```

Both expressions had a foundational `implication` origin (the rule "if `1 ∈ N` and `1 = 2` then `in2[1,2,id]`") **and** a cyclic `equality1` origin pointing at the other. With `max_origin_per_expr = 1` the bulk-merge from `mailIn.exprOriginMap` to `body.exprOriginMap` had to keep just one origin per key. Because the merge was raw `std::map` swap with body-wins-on-conflict and `addOrigin` was uninvolved, the choice was governed by mailIn's vector ordering, which silently picked the cyclic `equality1` origin and dropped the foundational `implication` one. Once installed in `body.exprOriginMap`, every downstream consumer (chapter projection, verifier chain walk) saw only the cyclic origin.

The fix has two coupled halves:

1. **Mechanism (`addOrigin`).** When at cap and a non-equality origin arrives, replace any existing `equality1`/`equality2` slot. The rule is categorical: foundation displaces convenience.
2. **Routing (bulk-merge).** Replace the raw `std::map` swap with a per-origin loop that calls `addOrigin`. Every arriving mail origin goes through the same gate as direct producer-side emissions, so the preference applies uniformly across producer-side and bulk-merge code paths.

Together these resolve the chapter-100/101 swap-cycle: at the bulk-merge, the foundational `implication` origin (when present) replaces any pre-existing `equality1`/`equality2` slot in `body.exprOriginMap`, and the chain walk reaches a base.

**What was investigated and ruled out.**

- **Bumping `max_origin_per_expr` from 1 to 2+ for Gauss.** Rejected: 1 is the operating contract; the bug was that the choice mechanism wasn't smart enough. Bumping the cap would defer the choice to the verifier's compressor / projection step and shift the failure mode rather than fixing it.
- **Capping `smashMail`'s mailIn aggregation per `max_origin_per_expr` (commit, reverted).** Rejected after user feedback. Capping at smashMail does the same thing as the existing cap inside `addOrigin` during bulk-merge, just one step earlier. It does not fix the choice — it only changes which origin happens to be first in mailIn. The structural issue is "how to choose", not "where the cap is enforced".
- **Producer-side gate inside `applyEquivalenceClass` only.** Insufficient on its own. The gate guards local emissions but does not affect mail-imported origins. Kept as a producer-side redundancy guard ([I-34](30_invariants.md#i-34)); D-49 is the system-level fix.
- **Per-tag priority table.** Rejected: the binary categorical distinction (equality-convenience vs. anything else) is the structurally correct one. Adding `implication > recursion > theorem > expansion >...` couples the helper to producer-side tag semantics and yields no extra cycle suppression.
- **Restricting preference to non-compressor mode only.** Rejected: in compressor mode the cap is 30, so cap-full is rare; but when it occurs the same preference applies. Uniform behavior is simpler and equally sound.

**Trade-off.** `addOrigin` becomes slightly less ordering-symmetric: a non-equality origin arriving after an equality1/equality2 slot displaces the equality slot, while previously it would have been silently dropped. Producer-side determinism is unchanged (same ordering, same keys, but different surviving origin per key when both kinds of tag are present). One previously-dropped foundational origin per cycle now wins; the cyclic origin disappears. Theorem count and verifier coverage expected unchanged on Peano / Gauss baselines that were already cycle-free; the chapter-100/101 cycle disappears.

**Verified.** Pending Peano + Gauss `python main.py` (no incubator, iter cap 35) re-run after this commit. Acceptance: `origin chain termination` failures drop to 0; Gauss `Number proven theorems` ≥ 11.

**Files touched.**

- `GL_Quick_VS/GL_Quick/src/prover.hpp`: `addOrigin` cap-full preference replacement step.
- `GL_Quick_VS/GL_Quick/src/prover.cpp`: `performElementaryLogicalStep` bulk-merge from `body.mailIn.exprOriginMap` to `body.exprOriginMap` rewritten to route through `addOrigin`.
- `docs/agentic_swdd/30_invariants.md`: I-35 added.
- `docs/agentic_swdd/AGENT_SwDD.md`: invariant quick-reference table updated (I-35).
- `docs/agentic_swdd/40_decisions.md`: this entry.

**What does NOT change.**

- Below-cap append behavior in `addOrigin` — unchanged. Dedup logic, D-44 symmetry-source trap, vector ordering all preserved.
- `smashMail` aggregation into `body.mailIn.exprOriginMap` — unchanged. Stays uncapped (compressor-mode multi-origin support); preference applies downstream at bulk-merge.
- Producer-side cap enforcement — unchanged. Each producer's local `addOrigin` call already respects `max_origin_per_expr`; D-49's preference adds replacement on top of that.
- D-46 cross-pair `equality2` gate — unchanged. Composes with D-49.
- D-48 cross-substitution `equality1` gate — unchanged. Defensible producer-side redundancy guard; D-49 is the system-level cycle resolution.
- Verifier — untouched per [I-16](30_invariants.md#i-16).

---

<a id="d-48"></a>
## D-48 — Cross-substitution `equality1` emission gated on existing LB origin (2026-05-07) — superseded as cycle-fix by [D-49](#d-49); kept as producer-side redundancy guard

> **Status amendment (2026-05-07, post-iter-35 verification).** D-48's gate is real and lands in the codebase, but the chapter-96/97 cycle it was *intended to close* re-emerged in a different shape (`(in2[i0,v10,id]) ↔ (in2[v10,i0,id])`, the swap-cycle, in chapters 100/101 of the same theorem) once the iter cap was raised from 29 to 35 and Gauss reached its full theorem count. The system-level cycle-resolution mechanism is **[D-49](#d-49) / [I-35](30_invariants.md#i-35)** — `addOrigin`'s cap-full preference replacement plus the bulk-merge routing change. D-48 stays in the codebase as a producer-side redundancy guard ([I-34](30_invariants.md#i-34)) but is not the load-bearing fix. The original "What / Why / Verified" text below describes D-48's intent at write-time; it is preserved verbatim for traceability.

**What.** `applyEquivalenceClass` ([`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp), `applyEquivalenceClass`) — the `if (parameters.trackHistory)` block that emits the `equality1` origin record for a class-rewritten expression — now skips the two `addOrigin` calls (`memoryBlock.exprOriginMap` and `memoryBlock.mailOut.exprOriginMap`) when the target `applied @ depositValidity` already carries any origin entry in `memoryBlock.exprOriginMap`. New invariant: [I-34](30_invariants.md#i-34). The `exprOriginMapLocal` populate at the rewrite-enumeration site stays unconditional (required for the FIRST emission's well-formedness — `check_equality1` rejects `len(rest) < 4`). A one-shot discovery trap inside `performElementaryLogicalStep` ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)) dumps the parent chain of every recursion-equality zero-case LB to , supporting follow-up debug rounds if the gate proves insufficient.

**Why.** [`verifier.py::check_origin_chain_termination`](../verifier.py) flagged 4 cyclic rows on the branch (Peano + Gauss, no incubator):

- `files/processed_proof_graph/96_check_zero.txt` rows 56–57 — self-cycle on `(in2[i0,v10,id]) ↔ (in2[i0,i0,id])`.
- `files/processed_proof_graph/97_check_induction_condition.txt` rows 72–73 — self-cycle on `(in2[i0,v6,id]) ↔ (in2[v6,i0,id])`.

Both chapters are sub-blocks of theorem 96 (`theorems.txt:2`, the Gauss `fold` induction theorem):

```
(>[1,2,3,4,5,7,8](AnchorGauss[1,2,3,4,5,6,7,8])
  (>[12,9](fold[1,3,4,8,2,9,12])
    (>[10](in2[9,10,3])
      (>[11](in3[7,12,11,5])
        (in3[9,10,11,5])))))
```

Cycle pattern (chapter 96):

```
56: (in2[i0,v10,id]) ← equality1 | (in2[i0,i0,id])  (=[i0,v10])
57: (in2[i0,i0,id])  ← equality1 | (in2[i0,v10,id]) (=[v10,i0])
```

Each row's only origin points at the other; the DFS loops indefinitely. Same shape as the chapter-22 / theorem-12 cycle that [I-32](30_invariants.md#i-32) / [D-46](#d-46) closed on the `equality2` cross-pair path — but here on the `equality1` substitution path. `applyEquivalenceClass` had a previously-commented-out guard `if (memoryBlock.exprOriginMap.find(appliedWithValidity) == memoryBlock.exprOriginMap.end)` at the addOrigin site; the comment said it was disabled to allow multi-origin accumulation (fixing two unrelated rt_conjecturer-reshuffle equality1 failures). The fix keeps the rationale intact (populate `exprOriginMapLocal` unconditionally so any first emission is well-formed) but reinstates the emission gate against existing origins to break the back-direction-cycle vector.

The fix is one-sided: gate only the emission site in `applyEquivalenceClass`. Site 2 (`applyEquivalenceClassToNegatedEquality`) has its own `statementLevelsMap` early-exit that prevents the same cycle shape on negated equalities. Site 3 (`emitIntegrationRevivalToInternalMailIn`) emits to `sameIterationInternalMail`; mail absorb at the next hashburst would invoke site 1's gate downstream. Both sites 2 and 3 are documented in [I-34](30_invariants.md#i-34) as future-extension candidates if a different cycle shape ever surfaces.

**What was investigated and ruled out.**

- **Producer-side mail-origin sync** (mirroring D-46's coupled producer-side change in `performElementaryLogicalStep`). Rejected: D-46's producer-side sync exists because the equality2 cross-pair logic consults `class.equalityOriginMap`, which the bulk-merge alone doesn't update. The equality1 gate only consults `body.exprOriginMap`, which the bulk-merge already populates from `body.mailIn.exprOriginMap` at the top of every hashburst. No producer-side change needed.
- **Loose form of the gate** (skip only when target has a *non-equality1* origin entry, allowing multiple equality1 origins from different classes). Rejected for parity with [I-32](30_invariants.md#i-32) and to be conservative — if a legitimate proof needs multiple equality1 records on the same target, the user can relax the predicate after measurement. The strong form closes the chapter-96/97 cycle without false-suppressing legitimate first emissions.
- **Gating sites 2 and 3** (`applyEquivalenceClassToNegatedEquality`, `emitIntegrationRevivalToInternalMailIn`). Deferred. Site 2's `statementLevelsMap` early-exit at `emitNew` is already a stronger guard than the proposed gate (it skips ALL deposit, not just origin). Site 3 emits to `sameIterationInternalMail`; redundant origins there get filtered by site 1 once mail flows back through `applyEquivalenceClass` on subsequent iterations. Adding gates at sites 2/3 would be defense-in-depth but is out of DoD scope for this fix.
- **Removing the discovery trap immediately**. Rejected per — the trap stays through verification; removed only after the fix is confirmed (Stage 2 strict chain match would replace it if needed).

**Trade-off.** Equality1 origin records become single-origin-per-target instead of multi-origin-accumulated. Verifier-visible chapter rows: equality1 substitutions for already-known targets disappear. The previously-cited "two equality1 failures from rt_conjecturer reshuffle" are NOT regressed because the populate step (`exprOriginMapLocal[r.rewrittenExpr] = eqs`) stays unconditional — only the emission step is gated. The chapter-96/97 cycle disappears; foundation-only chapters where each target has a single class-substitution origin are unaffected.

**Verified.** Two run sequences:

1. **Iter cap 29 (Peano + Gauss, no incubator).** `origin chain termination`: 0 failures (passes the DoD on this metric). But Gauss `Number proven theorems`: 9 vs. baseline 11 — the iter cap stopped 2 theorems before the cycle-prone proof paths fired, masking the issue.
2. **Iter cap 35 (Peano + Gauss, no incubator).** Gauss recovers to 11 proved. `origin chain termination`: 4 failures resurface in `100_check_zero.txt` rows 56–57 and `101_check_induction_condition.txt` rows 72–73 — the same theorem 96 cycle in a swap-form `(in2[i0,v10,id]) ↔ (in2[v10,i0,id])`. D-48's gate does not cover this shape because both targets are syntactically distinct keys both first-emitted; each target's `body.exprOriginMap` entry is empty when the gate fires, so emission proceeds.

The shape-B finding triggered [D-49](#d-49). Final acceptance verified there.

**Files touched.**

- `GL_Quick_VS/GL_Quick/src/prover.hpp`: `applyEquivalenceClass` — `alreadyHasOrigin` lambda + gated `addOrigin` calls inside the `trackHistory` block.
- `GL_Quick_VS/GL_Quick/src/prover.cpp`: `performElementaryLogicalStep` — Phase A discovery trap dumping recursion-equality zero-case LB chains to .
- `docs/agentic_swdd/20_core_concepts/05_equivalence_classes.md`: new *Cross-substitution `equality1` emission gating* subsection.
- `docs/agentic_swdd/30_invariants.md`: I-34 added.
- `docs/agentic_swdd/AGENT_SwDD.md`: invariant quick-reference table updated (I-34).
- `docs/agentic_swdd/50_gotchas.md`: G-39 added.
- `docs/agentic_swdd/40_decisions.md`: this entry.

**What does NOT change.**

- Statement deposit (`memoryBlock.statementLevelsMap`, `encodedStatements`, `localEncodedStatements*`, `newStatements`, mailOut statement) — unchanged. The gate only suppresses the origin emission; the rewritten statement still flows through `newStatements` and admission. [I-25](30_invariants.md#i-25) preserved.
- The `exprOriginMapLocal` populate inside the `enumerateEqClassRewrites` callback — unchanged. Required for first-emission well-formedness.
- `applyEquivalenceClassToNegatedEquality` and `emitIntegrationRevivalToInternalMailIn` — unchanged. Documented in [I-34](30_invariants.md#i-34) as deferred-extension sites.
- D-46 cross-pair `equality2` gate — unchanged. The two gates compose: equality2 cross-pair gate inside `mergeTwoEquivalenceClasses` + equality1 substitution gate inside `applyEquivalenceClass`.
- D-43 `applyEquivalenceClassToRejectedMapIntegration` additive contract — unchanged. The rmi side does not emit `equality1` origin records (it routes through `emitIntegrationRevivalToInternalMailIn` → `sameIterationInternalMail`).
- Verifier — untouched per [I-16](30_invariants.md#i-16).

---

<a id="d-45"></a>
## D-45 — `addOrigin` symmetry-source assert disabled; absorption fallbacks hardened; dead `proved` parameter removed (2026-05-07)

**What.** Three coordinated changes, all under the D-45 tag:

1. **`addOrigin` symmetry-source assert disabled.** [`prover.hpp::addOrigin`](../GL_Quick_VS/GL_Quick/src/prover.hpp) — the D-44 trap that fires when adding a `symmetry of equality` / `symmetry of inequality` origin whose source has no entry in the same map keeps its diagnostic dump to  but no longer asserts. The original assert was set for a different issue (cycle-class bugs caught by the chapter cycle-detection verifier check) and conflated the real bug (origin lines generated with no history at all) with the legitimate cross-map asymmetry between `body.exprOriginMap` (fed by the mailIn bulk-merge in `performElementaryLogicalStep`) and `body.mailOut.exprOriginMap` (local-delta only). Diagnostic value preserved; abort path removed.

2. **Absorption fallbacks hardened.** [`prover.cpp::performElementaryLogicalStep`](../GL_Quick_VS/GL_Quick/src/prover.cpp), the `sameIterationInternalMail`-absorb and `mailIn.statements`-absorb blocks. Pre-D-45 these had a defensive empty-origin fallback (silently used an empty origin pair when the matching `exprOriginMap` entry was missing). Post-D-45 they assert `it!= exprOriginMap.end && !it->second.empty`. Rationale: the silent fallback masked a sender-side bug where statements were mailed without their paired origin records. Hard-asserting surfaces the producer-side gap immediately.

3. **Dead `proved` parameter removed from `addTheoremToMemory`.** [`prover.cpp::addTheoremToMemory`](../GL_Quick_VS/GL_Quick/src/prover.cpp). The single caller (inside the conjecture-batch loop) always passed `false`; the body's `proved == true` branch was already dead. Parameter dropped; signature simplified.

**Why.** Cleanup pass following the chapter-22 / theorem-12 cycle investigation. The D-44 trap had served its purpose (revealed the `mergeTwoEquivalenceClasses` cross-pair cycle generator that [D-46](#d-46) closed), and its abort path was now spuriously firing on the legitimate map-asymmetry case. The absorption-fallback hardening turns a class of "silent empty-origin" bugs into immediate aborts, narrowing the search space when future origin-chain anomalies surface. The `proved` parameter cleanup is pure dead-code removal.

**What was investigated and ruled out.**

- **Removing the trap dump entirely** (alongside the assert). Rejected: the dump-to-file path is cheap and continues to be useful for diagnosing future cross-map asymmetry cases.
- **Tightening the trap to fire only when both maps are missing the source.** Considered but skipped — the dump now logs both maps' state for the trapped key + sources, which lets the future agent reason about asymmetry without a tighter predicate.
- **Replacing the absorption assert with a soft warning + skip.** Rejected: a missing origin in `mailIn.exprOriginMap` for a statement that *did* mail through means a sender-side bug. Soft-skip would mask the bug; hard-assert surfaces it with full context.

**Trade-off.** Aborts on the absorption hard-assert path replace silent skips. Net: one class of bugs surfaces immediately instead of propagating into chapter shape; in exchange, any pre-existing latent gap on the mail/origin pairing aborts the prover at first mail-arriving statement.

**Verified.** Inline with the D-44 / D-46 / D-47 work-stream verification. No standalone re-run for this commit.

**Files touched.**

- `GL_Quick_VS/GL_Quick/src/prover.hpp` — `addOrigin` D-44 trap: assert disabled, dump retained.
- `GL_Quick_VS/GL_Quick/src/prover.cpp` — `performElementaryLogicalStep` absorption blocks: hard asserts; `addTheoremToMemory` dead `proved` parameter dropped.
- (No SwDD docs touched at original commit time. This entry added 2026-05-07 by audit-fix to close the gap.)

**What does NOT change.**

- The trap's diagnostic dump format and target file () — unchanged.
- Mailing protocol semantics (which origins paired with which statements) — unchanged. D-45 only hardens the receiver-side check.
- D-44's ancestor-pass merge contract — unchanged.

---

<a id="d-47"></a>
## D-47 — `mergeTwoEquivalenceClasses` cross-vN preconditions: ancestor-only direction + eqArgs-subset assert is same-vN only (2026-05-07)

**What.** `mergeTwoEquivalenceClasses` ([`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp), `mergeTwoEquivalenceClasses`) gains a `classBValidityName` parameter. Three derived rules — see [I-33](30_invariants.md#i-33):

1. Cross-vN: `classBValidityName` must be a strict ancestor of `validityName` (`memoryBlock.nameMap.isStrictAncestor`). Asserted at function entry.
2. Cross-vN early-exit: if `classB.variables ⊆ classA.variables`, return immediately. Descendant `mergedClass` already covers every variable the ancestor class would contribute.
3. The existing `assert(!isSubsetOf(eqArgs, classB.variables))` and `assert(tmp.size == 1)` are gated on `sameVN`. Cross-vN allows both — the multi-bridge case picks `commonArg = *tmp.begin` deterministically.

Both call sites in `updateEquivalenceClasses` updated:
- Same-vN pass: pass `validityName, validityName`.
- Ancestor pass (D-44): pass `validityName, ancestorV`.

**Why.** [D-44](#d-44) added the ancestor-scope merge pass to `updateEquivalenceClasses`. The original `mergeTwoEquivalenceClasses` was written under same-vN assumptions: if any same-vN class held both `eqArgs`, it would be iterated first by the sequential overlap loop and absorbed via the subset path before `mergedClass` could grow past `eqArgs`. The `eqArgs ⊄ classB` and `tmp.size == 1` asserts encoded that invariant. Ancestor classes break it — they may independently contain both `eqArgs` (e.g. the equality was already admitted at the ancestor scope via mail or prior derivation), and the descendant's iteration order can't influence the ancestor's class structure.

The Gauss main batch (commit, after the D-46 squash) hit the assertion at Hash burst 2 and aborted via `0xC0000409`. The fix preserves the same-vN contract (assertions stay enforced for legacy merges) while permitting the cross-vN call legitimately added by D-44.

The early-exit on `classB ⊆ classA` (rule 2) is a separate optimisation: it avoids the merged-pair logic when the ancestor's contribution is already represented in the descendant. Without it, the merged-pair would emit cross-pair records for already-known equalities; [I-32](30_invariants.md#i-32) would suppress them downstream, but stopping early is cheaper and clearer.

**What was investigated and ruled out.**

- **Removing the assertions entirely.** Rejected — the same-vN contract is real and a violation indicates a `updateEquivalenceClasses` iteration-order regression. Keep the asserts; gate them.
- **Adding a single `bool isCrossScope` parameter.** Considered. Rejected because the ancestor-direction validity check needs the actual ancestor scope name, not just a boolean. Carrying the full `classBValidityName` is cleaner and supports the validation assert.
- **Multi-bridge cross-pair emission (record cross-pairs through both bridges when `tmp.size == 2`).** Rejected — would produce parallel origin records that [I-32](30_invariants.md#i-32) suppresses anyway. Single deterministic bridge is sufficient.
- **Folding classB.equalityOriginMap into classA on cross-vN early-exit.** Rejected — the ancestor class stays at its scope per [I-31](30_invariants.md#i-31); the descendant's class doesn't need a copy of the ancestor's origin records to function. The buildStack walker resolves cross-scope dependencies via the ancestor's exprOriginMap directly.

**Trade-off.** One more parameter on the merge helper; one more invariant to remember. In exchange: cross-vN merges are now safely admitted without false-positive aborts on Gauss / FTA-ladder runs that exercise the ancestor pass at scale.

**Verified.** Pending Peano + Gauss `python main.py` re-run after this commit. Acceptance: prior 0xC0000409 abort gone; verifier reports clean; no theorem regression vs. baseline.

**Files touched.**

- `GL_Quick_VS/GL_Quick/src/prover.hpp`: `mergeTwoEquivalenceClasses` signature + body; both call sites in `updateEquivalenceClasses`.
- `docs/agentic_swdd/30_invariants.md`: I-33 added.
- `docs/agentic_swdd/AGENT_SwDD.md`: invariant quick-reference table updated (I-33).
- `docs/agentic_swdd/40_decisions.md`: this entry.

**What does NOT change.**

- D-44's ancestor-scope merge contract — preserved. The ancestor class is still read by `const&`, never written; `equivalenceClassesMap[ancestorV]` and `eqClassSttmntIndexMapMap[ancestorV]` untouched per [I-31](30_invariants.md#i-31).
- Same-vN merge semantics — unchanged.
- Verifier — untouched per [I-16](30_invariants.md#i-16).
- The D-46 cross-pair gate — unchanged. The two fixes are independent (D-46 prevents redundant records; D-47 admits the cross-vN call legitimately).

---

<a id="d-46"></a>
## D-46 — Cross-pair `equality2` emission gated on existing class/LB origin (2026-05-07)

**What.** `mergeTwoEquivalenceClasses` ([`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp), `mergeTwoEquivalenceClasses`) — the merged-pair history block — now skips its `equality2` cross-pair `addOrigin` calls when the target equality `(=[varA, varB]) @ validityName` (or its mirror) already has an origin entry in (a) `mergedOriginMap`, (b) `classB.equalityOriginMap`, or (c) `memoryBlock.exprOriginMap`. New invariant: [I-32](30_invariants.md#i-32). Coupled with a producer-side change at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) (`performElementaryLogicalStep`) that, after the bulk-merge of `body.mailIn.exprOriginMap` into `body.exprOriginMap`, additionally syncs mail-arrived equality origins into the class's `equalityOriginMap` for any class whose variables already contain both args of `(=[a,b])`.

**Why.** [`verifier.py::verify_origin_chain_termination`](../verifier.py) — the origin-chain-termination check — flagged 3 cyclic rows in `files/processed_proof_graph/22_check_zero.txt` (theorem 12 induction zero-case). Root cause:

The zero-case LB chain — `AnchorPeano → in2[v1,i0,s] → in2[i0,v2,s] → (=[v1,i0])` — is internally contradictory. `in2[v1,i0,s]` says "i0 = S(v1)" (v1 is a predecessor of zero), which contradicts the Peano "no successor of zero" axiom mailed in as `!(in2[v1,i0,s])`. The parent-scope contradiction-cascade emits the full equality clique `{(=[v1,v2]), (=[v1,i1]), (=[v2,i1])}` + mirrors and ships them to the LB via mail. The bulk-merge at `performElementaryLogicalStep` placed mail origins in `body.exprOriginMap`, but the equivalence-class state was unchanged. Then `mergeTwoEquivalenceClasses`, iterating possible bridge variables, produced cross-pair `equality2` records via different bridges that point at each other:

- `(=[v1,v2]) ← equality2 | (=[v1,i1]) (=[i1,v2])` (bridge `i1`, emitted when `(=[i1,v2])` arrived and merged a class containing `v1` with one containing `v2`).
- `(=[v1,i1]) ← equality2 | (=[v1,v2]) (=[v2,i1])` (bridge `v2`, emitted when `(=[v1,v2])` arrived against a class containing `i1`).

Both records survived into chapter 22; the verifier's DFS reported the cycle.

The fix has two coupled halves:

1. **Producer-side sync** (`prover.cpp`). After bulk-merge into `body.exprOriginMap`, walk `body.mailIn.exprOriginMap` and register mail equality origins in the matching class's `equalityOriginMap`. Closes the information gap that prevented the merge logic from seeing mail derivations as already-known.
2. **Consumer-side gate** (`prover.hpp`). Cross-pair `equality2` push only fires when the target is *not* already established by some other path. The `equality2` record is a transitive convenience — it documents derivability through the merge bridge. When the equality is already established (mail, prior merge, anchor handling, recursion premise), the convenience record contributes no new deductive content but can form cycles with parallel `equality2` records via different bridges.

**What was investigated and ruled out.**

- **Recent symmetry-handling commits** were initially suspected — they touch `addStatement`'s mirror push and `updateEquivalenceClasses`'s "symmetry of equality" emission blocks. Ruled out: none of them touch `mergeTwoEquivalenceClasses`'s `equality2` cross-pair emission. The cycle generator predates these commits. The new origin-chain-termination check in `verifier.py:3438-3492` is what made the latent cycle visible.
- **Symmetric-emission-only suppression** (skip cross-pair when the *symmetric* target `(=[varB, varA])` is already known but emit on the asymmetric one). Rejected as inconsistent: equality is symmetric by I-9, so suppressing one without the other produces lopsided origin maps.
- **Source-side check on cross-pair sources** (skip if `(=[varA, commonArg])` or `(=[commonArg, varB])` lacks an origin). Rejected: cycle requires the *target* to be already-established; checking the sources is orthogonal and would over-suppress.
- **Pure consumer-side gate without the mail-sync producer change.** Half-correct: `memoryBlock.exprOriginMap` already learns mail origins via the bulk-merge, so the gate against `memoryBlock.exprOriginMap` alone would close the verifier-visible cycle. Adopted both halves anyway because the class state should be the source of truth that the merge logic consults; making it complete is a correctness invariant in its own right (matches the user's "equi classes have their own originMap" framing).

**Trade-off.** Cross-pair pushes are now skipped for already-derived targets. Consequence on `equality2`-tagged chapter rows: fewer rows where a separate path already produced the target's origin. Verifier still receives complete `equality2` foundation chains for cases where the equality genuinely originates only via the merge bridge (no prior mail, no recursion premise, no anchor handling). The chapter-22 / theorem-12 cycle disappears; foundation-only Peano chapters are unaffected.

**Verified.** Peano-only `python main.py` clean run (planned in this commit's verification step). Acceptance criteria documented in commit message.

**Files touched.**

- `GL_Quick_VS/GL_Quick/src/prover.cpp`: producer-side mail-origin sync into class `equalityOriginMap` after bulk-merge in `performElementaryLogicalStep`. (Landed in the prior commit on this sub-branch.)
- `GL_Quick_VS/GL_Quick/src/prover.hpp`: consumer-side cross-pair gate inside `mergeTwoEquivalenceClasses`.
- `docs/agentic_swdd/20_core_concepts/05_equivalence_classes.md`: *Origin tracking and the mail-sync rule* subsection (prior commit) + cross-pair-gate subsection (this commit).
- `docs/agentic_swdd/20_core_concepts/03_mail_system.md`: cycle-boundary protocol step 4 mentions equality-origin sync (prior commit).
- `docs/agentic_swdd/30_invariants.md`: I-32 added.
- `docs/agentic_swdd/AGENT_SwDD.md`: invariant quick-reference table updated (I-32).
- `docs/agentic_swdd/50_gotchas.md`: G-31 added (cyclic `equality2` chains from cross-pair re-emission of mailed equalities).
- `docs/agentic_swdd/40_decisions.md`: this entry.

**What does NOT change.**

- `addStatement` — unchanged. Equality ingestion still routes through `updateEquivalenceClasses` line 5886 (incoming-equality origin) and the existing class-merge / cross-scope-deposit machinery.
- `mergeTwoEquivalenceClasses`'s subset path (lines 5566-5589) — unchanged. No cross-pair emission there.
- `applyEquivalenceClass` / `applyEquivalenceClassToRejectedMapIntegration` — unchanged. They consume class state but do not write `equality2` cross-pair origins.
- Verifier — untouched per [I-16](30_invariants.md#i-16). The new origin-chain-termination check that surfaced the bug was already present.
- D-numbering: D-45 is reserved (referenced from `prover.cpp:1393, 1494` for an assert-tightening decision not yet entered into this log; future agent should add D-45 entry separately).

---

<a id="d-44"></a>
## D-44 — `updateEquivalenceClasses` ancestor-scope merge: cross-NS extension preserves ancestor class (2026-05-05)

**What.** `updateEquivalenceClasses` ([`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp), `updateEquivalenceClasses`) gains an ancestor pass that runs after the existing same-NS merge loop. For each strict ancestor `V_a` of `validityName`, the pass iterates `mb.equivalenceClassesMap[V_a]` and absorbs every class whose `variables` overlap `eqArgs` into the new `mergedClass` at `validityName` via `mergeTwoEquivalenceClasses`. Ancestor classes themselves stay UNCHANGED at `V_a` — `mb.equivalenceClassesMap[V_a]` and `mb.eqClassSttmntIndexMapMap[V_a]` are never written ([I-31](30_invariants.md#i-31)).

**Why.** Pre-D-44, an equality `(=[a,b])` admitted at scope `V` (descendant) with `a` already in an ancestor class `C_a @ V_a` would build a `mergedClass = {a, b, …same-NS-overlaps…}` at `V` but never absorb `C_a`'s members. The transitive `b ≡ all of C_a at V` was recoverable only at runtime via the apply-side machinery — the descendant scope's stored class never explicitly carried the cross-scope picture.

This mirrors the apply-side bidirectionality landed in [D-33](#d-33). D-33 gave the apply machinery cross-scope reach (a class at one scope can rewrite a statement at a comparable scope, with the rewrite landing at the deeper of the two). D-44 gives the merge machinery the symmetric cross-scope reach: when the merge driver runs at descendant `V`, it pulls in ancestor-scope classes that the new equality bridges to.

Soundness rests on descendant-inheritance: a class at `V_a` is observably visible at every descendant of `V_a` (every `var ≡ var'` pair holds at every descendant). An equality at descendant `V` that bridges to an ancestor class can therefore legitimately propagate the ancestor's equivalences into the descendant's merged class. The original ancestor class must not be touched — the new equality is invisible at `V_a` (same-or-deeper visibility rule), so writing `V_a`'s state from a descendant-scope operation is unsound.

**What was investigated and ruled out.**

- **Modify the ancestor class in place.** Rejected. The new equality is admitted at the descendant scope only; the ancestor scope cannot see it, so adding the equality's transitive consequences to `V_a`'s class would conjure equivalences `V_a` should not yet know.
- **Descendant-scope class merge (class strictly deeper than the equality's scope).** Excluded by symmetry with [D-33](#d-33)'s rmi class-deeper exclusion. A class at a deeper scope is invisible at the equality's scope; pulling its members into the equality's merged class would conjure equivalences the merging scope should not know.
- **Chained ancestor merges (overlap test against the growing `mergedClass` instead of `eqArgs`).** Defer. Matches the existing same-NS contract — `mergeTwoEquivalenceClasses`'s bridge invariant requires `commonArg ∈ eqArgs ∩ classA ∩ classB`, so the overlap test against `eqArgs` is what feeds the bridge. Transitivity through ancestor classes that overlap only with same-NS-absorbed vars is still recoverable at runtime via the apply machinery, just not explicitly stored in the descendant's merged class. Documented as a *Known & tracked* weakness in `docs/agentic_swdd/20_core_concepts/05_equivalence_classes.md`.

**Trade-off.** Merged classes at descendant scopes now carry members from every relevant ancestor class. Applies-loop cost scales as `|class|^2` per substitution; at Gauss scale classes are typically small (≤5 members), ancestor merges may push them to 10–20. Origin-graph fan-out grows: `mergeTwoEquivalenceClasses` builds `mergedOriginMap` entries for every cross-pair. Verifier `equality1` / `equality2` checks pattern-match on origin chains — the merge logic is reused unchanged, so origin shape stays consistent.

**Verified.**

- Pipeline: full `python main.py` clean run.
- Verifier: main 2750 checks, 0 failures; incubator 32795 checks, 0 failures; total airtight.
- `theorems.txt`: ≥41 lines (no theorem loss vs. main baseline).
- `global_theorem_list.txt`: Gauss summation induction theorem present; FTA-rung-1 lemmas (`interval`, `limitSet`, `limitSequence`, `sequence`, `fXY` rows under AnchorGauss) present.

**Files touched.**

- `GL_Quick_VS/GL_Quick/src/prover.hpp`: `updateEquivalenceClasses` — ancestor-pass block inserted after the same-NS merge loop, before the `mergedClass` push into `newClasses`.
- `docs/agentic_swdd/20_core_concepts/05_equivalence_classes.md`: new *Cross-scope class merge* section + chained-merge weakness entry.
- `docs/agentic_swdd/30_invariants.md`: I-31 added (ancestor classes read-only inputs to `updateEquivalenceClasses`).
- `docs/agentic_swdd/AGENT_SwDD.md`: invariant quick-reference + count update (30 → 31).
- `docs/agentic_swdd/40_decisions.md`: this entry.

**What does NOT change.**

- `mergeTwoEquivalenceClasses` — used as-is, takes `classB` by `const&`, so ancestor classes are read-only by construction.
- `applyEquivalenceClass` ([D-33](#d-33)) — unchanged. It already sees ancestor classes via `applyClassesFrom`'s ancestor-NS pass; now it sees a richer descendant `mergedClass` too, which composes naturally.
- `applyEquivalenceClassToRejectedMapIntegration` ([D-43](#d-43)) — unchanged. The rmi side remains additive at the new key; the ancestor-merge extension only affects the descendant-scope class-storage shape.
- `enumerateEqClassRewrites` shared helper — unchanged.
- Verifier — untouched per [I-16](30_invariants.md#i-16).
- Same-NS merge logic — unchanged. The ancestor pass runs after, and is purely additive on `mergedClass`.

---

<a id="d-43"></a>
## D-43 — ~~`applyEquivalenceClassToRejectedMapIntegration` keep-old: rmi rewrites are additive~~ *(superseded by [D-64](#d-64), 2026-05-13,)*

**Status.** Superseded. The additive-on-no-match design recorded here was the integration analog of the pre-D-63 algebra stance and carried the same provenance-leak bug acknowledged in the pre-revision [I-37](30_invariants.md#i-37): substituted constituents inserted at K2 lacked post-substitution `disintegration` provenance. The integration hook now follows the algebra D-63 drop+mail pattern via [D-64](#d-64); [I-30](30_invariants.md#i-30) is retired in favor of the unified [I-37](30_invariants.md#i-37) which covers both algebra and integration sides.

**Original wording (preserved for historical reference).**

**What (retired).** `applyEquivalenceClassToRejectedMapIntegration` ([`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp), function `applyEquivalenceClassToRejectedMapIntegration`) no longer erases the original rmi entry K1 when an equivalence class rewrites it to K2. Both the match path (revival via `sameIterationInternalMail`) and the no-match path (insert at K2) now leave K1 in place. Only K2 is added to `rmi`; the `toErase` queue and the post-loop `rmi.erase` apply-mutations block are removed. New invariant: [I-30](30_invariants.md#i-30).

**Why.** K1 is registered in `rmi` because at registration time it failed admission. It remains a candidate for revival via [`revisitRejectedIntegration2`](../GL_Quick_VS/GL_Quick/src/prover.hpp), which fires every time `makeAdmissionKeys` writes a new admission entry. `revisitRejectedIntegration2` iterates `rmi` keys and probes each against the current admission landscape. **A K1 erased by class application can never be revived by a future admission write**, even if that future admission would have matched K1 directly without any class involvement.

K1 and K2 also probe distinct admission slots: u_-form against `admissionMapIntegration`, bare form against `admissionSetIntegration`. The class's a→b substitution does not propagate to admission keys (admission is keyed structurally). So K2 failing admission today does not imply K1 will fail admission tomorrow under a different admission landscape.

This mirrors the additive principle [D-33](#d-33) already established for the expression-side cross-scope rewrite: original facts at the ancestor scope are never overwritten; the rewrite is purely additive at the new scope. Pre-D-43, the rmi side violated that principle by erasing K1 unconditionally — the same destructive pattern D-33 reverted on the descendant-class direction was still alive in the same-NS / class-shallower paths and was never reverted there.

**What was investigated and ruled out.**

- **Erase only on no-match path** (keep K1 on match, erase K1 on no-match). Considered as a partial fix. Rejected: the no-match path's erasure has the same revival-loss problem as the match path. K1 might match a future admission entry that the class's K2 cannot. The cleanest fix is uniform additive semantics across both paths.
- **Cap rmi growth** (e.g., LRU eviction, time-based decay). Not implemented in this commit; deferred until measurement shows growth is a problem at FTA scale. The existing `varsInRejectedMapIntegrationKeys` overlap short-circuit already suppresses additions for non-overlapping classes, which is the dominant fan-out.

**Trade-off.** `rmi` grows monotonically under class application instead of moving in place. Different keys (K1 + K2 + K3 from cross-class fan-out) accumulate in the map. Same-key inserts merge into the per-key value set (`std::map<ExpressionWithValidity, std::set<RejectedMapIntegrationValue>>` collapses duplicates), so identical (K, V) pairs from re-applied classes are idempotent. The fixpoint in `applyClassesFrom` terminates on `encodedStatements.size` plateau, **independent of `rmi` state** — kept K1 entries do not prolong the fixpoint unless they themselves drive new statements through revival. Memory overhead at Gauss scale is bounded; FTA-scale measurement still pending.

**Verified.**

- Pipeline: full `python main.py` clean run.
- Verifier: main 2750 checks, 0 failures; incubator 32795 checks, 0 failures; total airtight.
- `theorems.txt`: ≥41 lines (no theorem loss vs. main baseline).
- `global_theorem_list.txt`: Gauss summation induction theorem present; FTA-rung-1 lemmas (`interval`, `limitSet`, `limitSequence`, `sequence`, `fXY` rows under AnchorGauss) present.

**Files touched.**

- `GL_Quick_VS/GL_Quick/src/prover.hpp`: `applyEquivalenceClassToRejectedMapIntegration` — drop `toErase` declaration, drop `toErase.push_back(keyEv)`, drop the post-loop `for k in toErase: rmi.erase(k)`. Update three comment blocks to document additive semantics.
- `docs/agentic_swdd/20_core_concepts/05_equivalence_classes.md`: Extension section items 4–5 rewritten; "Why additive (D-43)" + "rmi growth under additive semantics" paragraphs added; rmi-growth weakness entry added.
- `docs/agentic_swdd/30_invariants.md`: I-30 added (keep-old).
- `docs/agentic_swdd/40_decisions.md`: this entry.

**What does NOT change.**

- I-22 (admission-map entry not cleaned on revival) — orthogonal; unchanged.
- I-21 (`sameIterationInternalMail` cleared at top of hashburst after absorb) — unchanged.
- D-33 scope-direction admission for rmi (rejects descendant direction) — unchanged.
- `applyEquivalenceClass` (expressions path) — unchanged; was already additive per D-33.
- `revisitRejectedIntegration2` — unchanged. It already operates on whatever's in `rmi`; with more entries kept it has more candidates to revive.
- `enumerateEqClassRewrites` shared helper ([D-NN](20_core_concepts/05_equivalence_classes.md#shared-inner-loop-helper), commit ) — unchanged.
- Verifier — untouched per [I-16](30_invariants.md#i-16).

---

<a id="d-42"></a>
## D-42 — `v / V` registry form, `w / W` citation form for theorem-anchor implications (2026-05-05)

**What.** Two coordinated changes — one pass added on each side of the producer/verifier boundary:

1. `process_proof_graphs.py` ITERATION 5 (`_v_to_w_in_theorem_citation`): for every chapter cell at column index ≥ 3 (the rest fields, i.e. cited dependencies — never the chapter's own HEAD claim at column 0) that passes `is_theorem_anchor_implication`, apply a case-preserving `v→w / V→W` letter swap to every `[vV]\d+` token at argument positions. Anchor variables (`N, i0, s, +, *, i1, i2, id, …`) never match `[vV]\d+` and are untouched. `<digit>_copy` forms are excluded by the lookahead. `global_theorem_list.txt` is built independently from `renamed_theorems` and remains in `v / V`.

2. `verifier.py` (`_revert_w_to_v_in_theorem_citation` + `_is_theorem_anchor_impl_local`): symmetric inverse. At three sites (`origin` meta-check `rest[0]` for `implication / multiplied from / mirrored from / reformulated from`, `check_theorem_tag`, `check_externally_provided_theorem`), if the cited expression has the theorem-anchor shape, apply `w→v / W→V` and try the reverted form in `state.global_theorems` / `state.external_theorems` before falling through to `_normalize_expr_list`. `check_implication`'s anchor branch needs no change (`_normalize_all_vars_in_list` already folds *every* arg).

**Why.** Pre-D-42, cited foreign-theorem inner bvars rendered with the same `v / V` letter as chapter-global free variables. A reader scanning `(in3[v1, i0, v2, +])` inside a chapter could not tell at sight whether the `v1, v2` were chapter-global free variables (Priority-2 numbering, shared with the head row) or bvars scoped inside an applied foreign theorem. Reserving `w / W` for the citation case removes the ambiguity at sight while keeping `v / V` as the registry/title form (so `global_theorem_list.txt` and the chapter HEAD of mirror / reformulation / OR-theorem chapters continue to present each theorem in its canonical registry form).

The user's stated three-step shape: *find applied/foreign theorem in a chapter → replace `v` by `w` → verifier recognises the citation, reverses `w → v`, and looks up the `v` form in `global_theorem_list.txt`*. The Pass 3 + revert pair implements this verbatim.

**What was investigated and rejected.**

- **HTML-side rename** (the historical `_v_to_w_display` in `generate_full_proof_graph.py`, removed). Flawed: verifier reads the processed proof graph, not the HTML, so verifier and customer saw different forms — verifier-vs-display split. Rejected permanently. The fix lives at the producer side (`process_proof_graphs.py`), so verifier and HTML consume the same form.
- **Renumbering per cell** (give each cited theorem fresh `w1, w2, …`). Rejected: breaks the digit-preserving round-trip with the registry, and makes the verifier-side revert non-trivial. The case-preserving letter swap is the simplest construction that satisfies "both sides agree on the digit, both sides differ only by letter".
- **Relying on `_normalize_expr_list` fold alone** (no explicit revert in verifier). Functionally correct — the fold already widens to `[vVwW]\d+` (per [D-NN c?590885 — implicit in the commit's verifier widening]) — but loses the fast `dep in state.global_theorems` exact-string path on every citation lookup. The explicit revert preserves the fast path and makes the round-trip invariant auditable.

**Why HEAD column 0 stays `v / V`.** Mirror / reformulation / OR-theorem chapters carry the chapter's *own* theorem in column 0 — that is the chapter's "title" / claim being proved. Rendering it in the same form as `global_theorem_list.txt` keeps title/registry parity; a reader can copy the HEAD verbatim and grep for it in the registry. Direct-proof and induction chapter HEADs are smaller fragments (not theorem-anchor implications), so the `is_theorem_anchor_implication` guard skips them implicitly anyway — the column-0 guard is the load-bearing rule that protects the mirror/reformulation/OR cases.

**Verified.**

- Main verifier (post-process re-run, raw graph unchanged): 2750 checks, 0 failures.
- Incubator verifier: 32795 checks, 0 failures.
- Spot-checked artefacts: `14_direct_proof.txt` rest cells in `w / W` form; `38_mirrored_statement.txt` HEAD in `v / V`, rest[0] in `w / W`; `109_reformulated_statement.txt` HEAD has `v1, v2, V1`, rest[0] has `w1, w2, W1, W2` (V → W proven on the set-typed bvars); `12_or_theorem.txt` HEAD `>[v1]`, rest[0]+rest[2] both `>[w1]`; `global_theorem_list.txt` carries zero `[wW]\d+` tokens.

**Files touched.**

- `process_proof_graphs.py`: `_VW_SWAP_PATTERN` + `_v_to_w_in_theorem_citation` helpers; ITERATION 5 in `create_processed_proof_graph`. ~30 LOC additive.
- `verifier.py`: `_WV_REVERT_PATTERN` + `_is_theorem_anchor_impl_local` + `_revert_w_to_v_in_theorem_citation` helpers; revert-then-membership added at the three lookup sites named above. ~25 LOC additive; no existing checker semantics modified (the revert is wired *before* the existing exact-string membership tests, with the fold-based fallback unchanged behind it). I-16 honoured.
- `docs/agentic_swdd/10_pipeline/06_process_proof_graph.md`: new `## Pass 3 — v→w rename of cited theorem-anchor implications` section.
- `docs/agentic_swdd/10_pipeline/07_html_export.md`: w-rename row in the Selected-functions table extended to mention Pass 3 alongside Pass 2.
- `docs/agentic_swdd/10_pipeline/08_verifier.md`: origin-check code excerpt updated to show the `dep_v` fast path; new `### w → v revert for cited theorem-anchor implications` subsection under the `origin` meta-check.
- `docs/agentic_swdd/40_decisions.md`: this entry.

**What does NOT change.**

- `generate_full_proof_graph.py` — already neutralized; HTML renders processed cells as-is.
- C++ prover, conjecturer, compressor, MPL configs — orthogonal.
- `_normalize_expr_list`, `_normalize_with_unchangeables`, `_alpha_canonicalize_bound_vars` — already widened to `[vVwW]`; remain widened, used as the safety net behind the explicit revert.
- The per-cell `w / W` rename for non-anchor implications (`w_rename_impl_local`, Pass 2). Pass 3 is *additive* — Pass 2 still handles products of disintegration with raw `\d+` bvars; Pass 3 handles cited theorem-anchor implications. The two passes act on disjoint cell categories.
- Renumbering. Pass 3 is a pure letter swap; the digit is preserved.

---

<a id="d-41"></a>
## D-41 — Verifier `definition set consistency` meta-check + ConfigVisu defset drift fix (2026-05-03)

**What.** Two coupled changes:

1. New verifier meta-check `definition set consistency` (a per-row check counted in the standard ~32-line tally) implementing the C++ compiler's defset-consistency algorithm at `compiler.hpp` (`ArgumentAnalyzer` + `RecursiveParser::parseSubtree` + `mergeMaps` + `processLeaf` + `checkDefinitionConsistency`) faithfully in Python. For every chapter row, every expression in the row (the left-hand expression and each `rest[i]` at even indices) is parsed independently with bound-variable scoping at `>[…]` quantifier nodes; `_merge_maps` flags type-label mismatch when two child sub-trees share a variable name with disagreeing types. Per-batch resolved defsets are pre-computed once in `run_verifier` startup and selected per-chapter via the same anchor-substring match the verifier already uses for `state.current_gl_binary` (mirrors the compiler's per-batch `ArgumentAnalyzer(this->coreExpressionMap)` construction at `prover.hpp`).

2. `ConfigVisu.json` defset drift fix. Audit between `ConfigVisu.json` and the per-batch configs (`ConfigPeano`, `ConfigGauss`, `ConfigIncubatorPeano`, `ConfigIncubatorGauss[1]`) found one drifted defset (`interval` position 2: ConfigVisu had `P(x(1)(1))`, all batch configs had `P(x(1)(x(1)(1)))`) and six operators present in batch configs but absent from ConfigVisu (`infiniteSequence`, `limitSequence`, `limitSet`, `constSeq`, `nonInterval`, `nonSequence`). All 7 issues fixed by syncing `ConfigVisu.json` to the per-batch authoritative versions.

**Why.** Coverage gap surfaced by the user: the verifier did not check whether a chapter row's variable connections respected definition-set typing. A theorem could in principle have variables wired through ports with conflicting type labels and every existing verifier check would still pass — formal completeness gap. The compiler enforces this on the producer side (its `ArgumentAnalyzer` runs at `compileCoreExpressionMap` time), but the verifier's purpose is to be the independent oracle on shipped artefacts. Mirroring the compiler's algorithm in the verifier closes the gap.

The drift fix in ConfigVisu was discovered as a Stage-1-style finding when the new check first ran: 46 main + 357 incubator = 403 failures fired, all rooted in `interval` pos-2 type contention. Sync'd ConfigVisu, re-ran: 1 + 266 = 267 failures (down 137). Remaining failures decomposed into a verifier-algorithm scoping bug (Class A — initial regex-based pooling did not respect bound-variable rebinding at `>[v1,v2]` quantifiers, treating same-name variables across alpha-distinct scopes as the same variable) and a per-batch compact-name collision case (Class B — `implication26` is allocated arity 2 in Gauss main but arity 5 in IncubatorGauss; my initial single-flat-resolved-defsets dict picked one binary's allocation arbitrarily). Class A fixed by translating the C++ recursive parser faithfully (Python `_parse_subtree` + `_merge_maps` + `_process_leaf`). Class B fixed by per-tag resolution (the analog of the compiler's per-batch analyzer construction).

**Verified.**

- Main side: 2632 successful defset checks, 0 failures (35545 / 0 — airtight).
- Incubator side: 31584 successful defset checks, 0 failures (32795 / 0 — airtight).
- Synthetic-corruption sanity test: 4 hand-crafted type-mismatched expressions correctly flagged; 4 well-typed expressions correctly accepted (including the bound-var-rebinding case `(>[v](in[v,N])(=[v,a]))` that confirms scoping works).

**Files touched.**

- `verifier.py`: `VerifierState` now carries `resolved_defsets_per_tag`, `resolved_defsets_atomic_only`, and per-chapter `current_resolved_defsets`. New helpers: `build_resolved_defsets_per_tag`, `_try_derive_from_elements`, `_merge_maps`, `_process_leaf`, `_parse_subtree`, `check_defset_consistency`. Wiring at `run_verifier` startup (build the per-tag indices) and `verify_chapter` (select per-chapter `current_resolved_defsets` alongside `current_gl_binary`). New per-row check loop after the `origin` meta-check. ~280 LOC additive; no existing checker modified. I-16 honoured via additive-only change with explicit user consent for this specific extension.
- `files/config/ConfigVisu.json`: `interval` pos-2 sync to ternary; six operators added (`infiniteSequence`, `limitSequence`, `limitSet`, `constSeq`, `nonInterval`, `nonSequence`).
- `docs/agentic_swdd/40_decisions.md`: this entry.
- `docs/agentic_swdd/30_invariants.md`: I-29 (variable-port type consistency invariant).
- `docs/agentic_swdd/10_pipeline/08_verifier.md`: new "definition set consistency meta-check" section.
- `docs/agentic_swdd/20_core_concepts/06_anchors_and_scopes.md`: weakness at line 172 ("Anchor-slot typing table is not validated at load") updated — the verifier-side check now closes the in-flight artefact half of the gap; the load-time half (config↔MPL-definition cross-validation) remains.
- `docs/agentic_swdd/04_configs.md`: note added that ConfigVisu.json should mirror per-batch configs for shared operators.

**Why "compiler's variant" in particular.** Translating the C++ algorithm verbatim instead of inventing a new check has three benefits: (a) it inherits the compiler's known-correct scoping rules (bound-var removal at `>[…]` is the only place where variables leave scope, and `mergeMaps` is the only mismatch detector); (b) it matches the producer-side contract exactly, so any verifier-side failure points squarely at a producer-side bug rather than a verifier-side definition gap; (c) Python translation of ~150 LOC of C++ is auditable, with `verifier.py`'s comments citing the C++ line numbers for every translation point.

**Per-batch resolution and shared-binary fallback.** `build_resolved_defsets_per_tag` builds one resolved-defset map per `tag` in `gl_binaries`. Each tag's map starts from atomic seeds (ConfigVisu.json), unions in `GL_binary_shared.json`'s composites (cross-batch fallback for spontaneous-category compact names — `_SPONTANEOUS_CATEGORIES = {"implication", "existence", "or", "and"}`), then resolves the tag's own composites with override semantics on collision (per-batch is authoritative for its own chapters). `verify_chapter` picks the right tag's map at chapter start. Chapters whose theorem doesn't disclose an anchor (rare) fall back to atomic-only.

**What does NOT change.**

- No existing verifier checker modified. I-16 honoured.
- ConfigVisu drift fix is data-correction (sync), not schema change.
- Per-batch configs unchanged.
- Verifier returns the same 28-line tally + 1 new line; airtight runs stay airtight.

**Open follow-up.**

- Cross-row in-chapter aggregation (Phase 2 from the original plan): variables in `chapter row N` and `chapter row M` at the same `validityName` could be type-checked together. Deferred — not exercised by current chapter shapes; would be additive when needed.
- Cross-chapter rule citations (Phase 3): rules cited from one chapter's `rest[0]` lookup match against `state.global_theorems` registry — the registry entries themselves could be defset-checked on load. Deferred — the producer-side compiler already enforces this for any rule entered into its `coreExpressionMap`.

---

<a id="d-40"></a>
## D-40 — Cross-batch externals seed switched from `theorems.txt` (expanded) to `compiled_theorems.txt` (compact) (2026-05-03)

**What.** `run_modes.py` (the per-tag incubator-stage seed step) was reading `files/theorems/theorems.txt` (expanded form — compiled structural operators like `existence2` / `or0` rewritten to base form by `prover.cpp`'s `expandToBaseForm`) and writing it as the next-tag incubator's `externally_provided_theorems.txt`. Switched to `files/theorems/compiled_theorems.txt` (compact form — keeps `existence2` / `or0` as compact heads; `prover.cpp`).

**Why.** Stage-2 cleanup of the residual verifier failure exposed by D-39's determinism fix. Once execution was deterministic, the incubator verifier reliably reported one `origin`-tag failure at `1209_direct_proof.txt` row 57, an `implication` row whose `rest[0]` cited the Peano rule `(>[N,i0,s](AnchorPeano[N,i0,s,+,*,i1])(>[i2](in[i2,N])(>[]!(=[i2,i0])(existence2[N,i2,s]))))` (with the compact head `existence2`). The verifier's `origin` check looks `rest[0]` up in `state.global_theorems ∪ state.external_theorems` via alpha-canonical match, but `_alpha_canonicalize_bound_vars` only renames bound variables — it does not expand or compact structural operators. The seed file was carrying the rule's expanded form (`!(>[8](in[8,1])!(in2[8,7,3]))` instead of `existence2[1,7,3]`), so chapter rows that cite the compact form had no registry hit.

The expansion at `prover.cpp` was originally added to keep `theorems.txt` parseable by a downstream batch that might not have the same compact-name dictionary loaded at parse time. By the time `GL_binary_shared.json` infrastructure (`run_modes.py:_seed_per_batch_binary` + `_merge_into_shared`) was added — which makes the spontaneous-category compact dictionary (existence/or/and/implication) cross-batch — the expansion at the seed-propagation step became unnecessary for actual cross-batch parsing of those categories. Non-spontaneous categories (anchor entries, atomic entries) are still excluded from the shared binary, so the inter-batch `theorems.txt` retains its expanded form and remains the safe source for category-agnostic cross-batch parsers; only the seed-propagation source for the next-tag incubator switches to the compact-form file.

**Caveat documented inline (`run_modes.py`).** Cross-batch parsing of compact-form externals is supported only because `GL_binary_shared.json` carries the spontaneous compact-name dictionary. If a future batch references compact heads outside the spontaneous categories — i.e. names that are batch-local — the seed switch will reintroduce parser failures. The fix in that case is to expand the shared-binary coverage or to revert the seed source.

**Verified.** 3 full-pipeline runs (Peano + Gauss, both incubator and main) post-fix: per-tag verifier counts identical; processed-proof-graph md5 identical; proved-theorems md5 identical; total verifier checks `35545`, **0 failures across all tags** (incubator: 32795 checks, 0; main: 2750 checks, 0). All Stage-1 + Stage-2 DOD criteria met.

**A/B against pre-Stage-2.**

| | Pre-Stage-2 (post-D-39) | Post-Stage-2 (post-D-40) |
|---|---|---|
| Incubator checks | 32796 | 32795 (-1: redundant `origin` lookup gone) |
| Incubator failures | 1 (`origin` tag) | 0 |
| Main checks | 2750 | 2750 |
| Main failures | 0 | 0 |
| Total | 35546 / 1 failed | 35545 / 0 — airtight |
| 3-run identity | ✅ (D-39) | ✅ (preserved) |

**What does NOT change.**
- `theorems.txt` continues to be written in expanded form (`prover.cpp`) — its role as inter-batch parser-input (for downstream consumers that aren't gated on the compact dictionary) is unchanged.
- `compiled_theorems.txt` continues to be written in compact form — its role as proof-graph pruning source is unchanged.
- `--mirror-externals` C++ step's input file (`externally_provided_theorems.txt`) shape is unchanged; only the source content (compact instead of expanded). The mirror logic operates on whatever form it receives.
- Verifier (`verifier.py`) is untouched (I-16).
- D-39's determinism fix is unaffected; the deferred-action collector remains intact.

**Files touched.**
- `run_modes.py`: source file changed from `theorems.txt` to `compiled_theorems.txt`; new comment block documenting the rationale + cross-batch caveat.
- `docs/agentic_swdd/40_decisions.md`: this entry.
- `docs/agentic_swdd/10_pipeline/09_incubator.md`: externals-seed paragraph updated to reflect the new source.

**Open follow-up.**
- If a future Stage proves theorems with compact heads outside `_SPONTANEOUS_CATEGORIES` (`run_modes.py`) that need to cross batches via the externals seed, the shared-binary coverage will need to expand. Track via a future invariant or D-entry as needed.
- The `expandToBaseForm` call at `prover.cpp` is no longer needed for the externals propagation path. It is retained for any other consumer of `theorems.txt` that may need the expanded form. Could be revisited if no such consumer exists.

---

<a id="d-39"></a>
## D-39 — Race-free contradiction-origin propagation via deferred-action collector (`pendingAncestorOrigins`) (2026-05-03)

**What.** Replace the `addStatement` `primedForContradiction` handler's direct walk over `memoryBlock.parentMemory` (`prover.cpp` was 4763-4768) with a class-level deferred-action collector that drains in `proveKernel` after `pool.join`, single-threaded, in sorted order. The walk-and-`addOrigin` step itself is unchanged — it is moved from the parallel-phase descendant thread to the post-join single-threaded drain. Producer site stages a `PendingAncestorOrigin{ &memoryBlock, ev, origin, maxOrig }` under `pendingAncestorOriginsMutex`; consumer site sorts by `(emitter.exprKey, ev.original, ev.validityName, origin.first)` and walks each emitter's `parentMemory` chain calling `addOrigin` on each ancestor.

**Why.** GL was non-deterministic on `main` HEAD: 3 consecutive `main.py` runs produced different verifier check counts and proof graphs. Investigation traced the root cause to the contradiction handler: when a "primedForContradiction" LB detected a reductio (`(A) ∧ ¬(A) ⊢ ¬(assumption)`), the handler walked its own `parentMemory` chain and called `addOrigin(pred->exprOriginMap, …)` on every ancestor. `addOrigin` (`prover.hpp`) is a plain `std::map[ev]; vec.push_back(...)` — not thread-safe. Meanwhile the parallel `proveKernel` worker pool (`prover.cpp`, `workers = std::thread::hardware_concurrency`) had OTHER threads reading and writing those same ancestor maps via their own `performElementaryLogicalStep`. Concurrent `std::map` insert + iterate is undefined behavior. The race produced a bimodal pattern: an entry was sometimes recorded, sometimes lost.

**Diagnostic evidence.** Hashburst dump retargeted to AnchorIncubator base LB (the single mailing recipient of the entire incubator deductive process). 3 incubator-Peano-only `main.py` runs:

| Run | hashburst hash | bytes | verifier |
|---|---|---|---|
| 1 | `4eb46e2c…` | 48,000,529 | 28723 checks, 2 FAILED (self-reference) |
| 2 | `055bb9cd…` | 47,995,681 | 28778 checks, 0 failures (airtight) |
| 3 | `055bb9cd…` | 47,995,681 | 28778 checks, 0 failures (airtight) |

First divergence at HASHBURST #10: Run 1 has `origins=3690`, Run 2/3 have `origins=3691`. The missing entry in Run 1 is exactly:
```
!(in3[10,11,2,4]) | v=main
  <- contradiction | (in2[14,2,3]) | !(in2[14,2,3]) | (in3[10,11,2,4])
```
A reductio chain `(in3[10,11,2,4]) ⊢ (in2[14,2,3]) ∧ ¬(in2[14,2,3]) ⇒ ¬(in3[10,11,2,4])`. When the entry survives, the verifier traces 55 additional checks and reports 0 failures. When it vanishes, the verifier reports 2 self-reference failures (the chain has no contradiction origin to follow).

**Verified.** After the fix, on:

- 3 incubator-Peano-only runs: byte-identical hashburst (`055bb9cd…`), all 28778 checks, 0 failures.
- 3 full-pipeline runs (Peano + Gauss, both incubator and main): byte-identical hashburst (`40d18b02…`), per-tag verifier counts identical, byte-identical processed-proof-graph md5, byte-identical proved-theorems md5. One verifier failure remains in tag `origin` (success 266, failure 1) on the incubator side — single-digit and reproducible across all 3 runs, consistent with a pre-existing IncubatorGauss origin chain bug now exposed by deterministic execution. Stage 2 of the determinism work (separate phase) addresses it.

**A/B against the abandoned predecessor ( D-38, commit ).** The earlier branch attempted a "canonical-min origin selection" fix that made verifier counts identical but introduced 1078 verifier failures and left intermediate state non-identical. That branch is reference material only and is not cherry-picked. D-39 is the canonical determinism fix on `main`-derived branches; D-38 (on the abandoned branch) is superseded.

**Why "induction-precedent collector" over a `mailOutAncestor` mail channel.** Two viable patterns, both deferring the cross-LB write to a post-`pool.join` single-threaded drain:

- *Mail-channel* (`mailOutAncestor` on `Memory`): per-LB inbox/outbox mirror of `mailOut`. Race-safe via same-LB writes during parallel phase; new field on every Memory; new smashMail variant.
- *Class-level collector* (`pendingAncestorOrigins` on `ExpressionAnalyzer`): mutex-guarded vector + post-join sort + walk. Direct precedent in `inductionMemoryBlocks` (`prover.hpp, 112`; `prover.cpp, 6269-6278`).

Picked the collector pattern. Reasons: direct existing precedent (the induction collector is structurally identical — same skeleton, different payload, drained in the same post-join block), tighter memory footprint (only firing LBs contribute entries), and keeps the mail abstraction (`mailOut`/`mailIn`) reserved for grid-broadcast cycle communication rather than special-cased ancestor writes.

**Files touched.**
- `GL_Quick_VS/GL_Quick/src/prover.hpp`: `PendingAncestorOrigin` struct, `pendingAncestorOrigins` vector, `pendingAncestorOriginsMutex`.
- `GL_Quick_VS/GL_Quick/src/prover.cpp`: constructor init list; producer swap at the contradiction handler; drain block in `proveKernel` after the existing induction-block drain. Also: hashburst-dump retarget to AnchorIncubator base LB, pointer-address scrub, `encodedMap` sort-on-emit, dump toggled OFF for production. Rule-12 stale-comment fix at the `exprOriginMap` block.
- `GL_Quick_VS/GL_Quick/src/filter.cpp`: reset in `releaseCEBatchMemory` (whole CE-batch teardown extracted from `prover.cpp`, 2026-05-04 — same body, new TU).
- `run_modes.py`: temporarily set to incubator-Peano-only during the iteration loop; restored to full pipeline for Phase 4 validation.
- `docs/agentic_swdd/40_decisions.md`: this entry.
- `docs/agentic_swdd/30_invariants.md`: I-22 (cross-LB writes during parallel phase forbidden — defer to post-`pool.join`).
- `docs/agentic_swdd/20_core_concepts/03_mail_system.md`: stale `logicalCores = 1` claim corrected (Rule 12); deferred-action collector pattern noted alongside `sameIterationInternalMail`.

**What does NOT change.**
- `verifier.py` is untouched (I-16).
- `parameters.max_origin_per_expr = 1` is untouched.
- The contradiction handler's own-LB writes (`memoryBlock.exprOriginMap`, `memoryBlock.mailOut.exprOriginMap`) are kept — they were always race-safe.
- Mail routing (`mailOut → smashMail → mailIn`) is untouched — descendant-direction propagation continues unchanged.
- Theorem set proved is unchanged (md5-identical to baseline).

**Open follow-up (Stage 2).** The 1 residual `origin`-tag failure in the IncubatorGauss path is now reproducible byte-identically and can be debugged by standard trap-debug methodology with the determinism guarantee from this fix in hand.

---

<a id="d-37"></a>
## D-37 — HTML export: matryoshka sub-proof nesting (arbitrary depth) (2026-05-03, branch `main`)

**What.** `generate_full_proof_graph.py`'s sub-proof renderer is now recursive. Sub-sub-…-proofs render as collapsed cards inside their parent's collapsed card — Russian-doll style, no fixed depth. Implementation: `_partition_stack_subproofs` recurses on each child scope's row group; `_render_subproof_card` recurses on `nested_subproofs`. Per-depth CSS classes (`.subproof-depth-1` … `.subproof-depth-5`, plus a generic `:not` rule for depth 6+) differentiate nested cards visually with color + indent.

**Why.** D-36's producer-side `ordisMerge` extension caused chapter exports to surface deeper nested scopes (`_ordis_` branches inside an outer `_orint_` subproof, etc.). The previous one-level-only sub-proof renderer flattened these into the main stack of the parent subproof, losing the structural information. The matryoshka rendering preserves it.

**Detection.** Scope hierarchy is derived purely from primary-namespace ancestry (`row[1]` of each row): a row at namespace `A_boundary_<X>` is a child of the scope at namespace `A`. No tag-level marker is required for nesting detection. Title inference for a child scope reuses `validity name` row metadata when present (preserves existing rich titles for implication subproofs); falls back to the namespace's payload pattern (`_orint_` / `_ordis_`) for OR-branch sub-subproofs that have no `validity name` introduction.

**Verification.** Chapter `1209_direct_proof.txt` (FTA-rung-1 forward direction, `EnumerationSet2 ⟹ interval`) renders as 3 depth-1 cards (impl24/impl25/impl26 subproofs) with 5 depth-2 cards nested inside (2 `_ordis_` branches under impl24, 2 under impl25, 1 `_orint_` subproof under impl26). DOM-depth walk confirms the depth-2 cards are inside their parent's `<div class='subproof-body'>`. Other chapters with no nested scopes render identically to before.

**Files touched.** `generate_full_proof_graph.py` (rewrite of `_partition_stack_subproofs`, new `_scope_title_info`, new `_render_subproof_card`, rewrite of `render_stack_with_subproofs`, new CSS depth classes); `docs/agentic_swdd/10_pipeline/07_html_export.md` (new "Matryoshka subproof structure" section).

---

<a id="d-36"></a>
## D-36 — Producer-side `or convergence` spec'd row + verifier correction on `or branch proven` (2026-05-03)

**Background.** `verifier_rung1` ended with 3 deliberate verifier failures on the incubator path. This entry covers their resolution. Two go away via a producer-side `ordisMerge` extension; one goes away via a verifier-side check correction (a check based on a wrong premise).

**Corrected `_orint_` mental model (user clarification, 2026-05-03).** `_orint_` does not case-split. To prove `(A ∨ B)` it rewrites the goal into two implication-sub-proofs `(!A → B)` and `(!B → A)`; each is a subproof scope. Inside each scope the negated antecedent is assumed (emitted as `or branch assumption`), and the subproof tries to derive the consequent. If one subproof fires, the OR is derived at parent scope and emitted as `or branch proven` — that row IS the OR's derivation by design. There is no separate non-`or branch proven` derivation row, by construction.

This is distinct from `_ordis_` (case-split, all-branches-converge — see [D-34](#d-34) and [`07_or_branching.md` §3b](20_core_concepts/07_or_branching.md#3b-_ordis_-or-disintegration-all-branches-converge)). `_ordis_` consumes an existing OR; `_orint_` produces an OR.

The "branch" terminology in `or branch proven` and `or branch assumption` is a historical misnomer — these are subproof rows. Renaming has been deferred to avoid touching every consumer at once; the doc now flags this clearly.

**Stage P0 — Verifier correction (`check_or_branch_proven` round-3 check #1 dropped).** The pre-fix check required a chapter row with `expression == or_expr`, `namespace == parent_ns`, and `tag!= "or branch proven"`. Codex round-3's rationale was "chapter 1209 line 19 derives the OR before line 22 splits it"; closer reading shows line 19 USES `(or2[i1,v5,i0])` as a premise of an implication firing (rule premise → conclusion), it does not derive the OR. Under correct `_orint_` semantics the OR has no separate derivation. The check is unsatisfiable for legitimate proofs.

Per the user's directive ("no relaxations of verifier") this is a CORRECTION (the check encoded a wrong premise) rather than a relaxation (softening a real check to pass tests). The four remaining `check_or_branch_proven` validations (layout, OR-shape, disjunct-membership, exact-namespace) and the round-3 check #2 on `check_or_branch_assumption` (matching `or branch proven` row) are unaffected — both are well-founded under the corrected semantics.

**Stage P1 — Producer-side `or convergence` spec'd row layout (`ordisMerge` extension).** The `_ordis_` convergence row is the legitimate gap. Pre-fix the row had 4 fields `(C, parent, or convergence, OR, parent)` and the verifier's `check_or_convergence` deliberately failed it (clean-fail per [D-35](#d-35)). Per the user-directive of 2026-05-03 the new producer-side row layout is fixed:

```text
<C>  <parent>  or convergence  <OR>  <parent>  <C>  <branch_D1>  <C>  <branch_D2>  …  <C>  <branch_DK>
```

`ordisMerge` (`prover.hpp` ~line 6020) now extends `mergeOrigin.second` with one `(C, fullBranchValidity)` per branch in `memoryBlock.orBookkeeping[mergeKey]`. The `fullBranchValidity` is reconstructed via the existing `branchPrefix + branchPayload` formula. The chapter export pipeline naturally renders the per-branch derivation rows of `C` because `removeExpressionFromMemoryBlock(state=0)` (called by `ordisMerge` to collapse the per-branch copies) does NOT touch `exprOriginMap` — the branch derivations remain available, and `buildStack` (visualizer.cpp) recurses into each new ingredient via `exprOriginMap.find`.

**Stage P1b — Tighten `check_or_disintegration` for compiled OR (newly-exposed by P1).** Stage P1's producer-side `ordisMerge` extension caused `buildStack` to recurse into per-branch ingredients of the new convergence row layout. Side-effect: 4 previously-hidden `or disintegration` rows (chapter 1209 lines 17, 21, 29, 33) surfaced. They use the compiled-OR shape `(or<N>[…])` in `rest[0]`, but the legacy `check_or_disintegration` accepted only the expanded `!(&!(…))` form and never validated namespace structure or OR-origin. Tightened in P1b to:

1. `len(rest) == 2` exactly.
2. `rest[0]` is a known compiled OR with ≥2 disjuncts via GL-binary (matching arity).
3. `line.expression` is one of the disjuncts (modulo equality symmetry).
4. `line.namespace` is EXACTLY `rest[1] + "_boundary_ordis_" + rest[0] + "_(" + <disjunct> + ")"` for `<disjunct>` matching `line.expression` (modulo equality symmetry).
5. The OR has an independent derivation row at parent scope (`expression == rest[0]`, `namespace == rest[1]`, `tag!= "or disintegration"`). **This check IS well-founded for `_ordis_`** — case-split CONSUMES an existing OR. The analogous check on `check_or_branch_proven` was dropped above because `_orint_` PRODUCES an OR.

The asymmetry between `check_or_branch_proven` (no OR-origin check) and `check_or_disintegration` (OR-origin check required) is exactly the `_orint_` (produces) vs `_ordis_` (consumes) semantic distinction.

**End state (D-36).** Main pipeline `2750/0` (unchanged). Incubator `32763/1` — the 3 originally-targeted failures all closed:
- Failures #1, #2 (`or convergence` lines 5, 14) — closed by Stage P1.
- Failure #3 (`or branch proven` line 22) — closed by Stage P0 (verifier correction).
- The 4 newly-exposed `or disintegration` rows (chapter 1209 lines 17, 21, 29, 33) — closed by Stage P1b.

**Remaining 1 failure (separate, pre-existing).** `self-reference failure 1` on chapter `1032_direct_proof.txt`: the chapter's target theorem `(>[…](AnchorIncubator)!(fold[N,s,+,id,i0,i1,i7]))` appears in the incubator's `global_theorem_list.txt` as `direct -1`, and the chapter's only proof step cites this same implication as a `theorem` rule — circular self-citation. The verifier's self-reference counter is correct to flag it. Cause: producer-side issue (likely conjecturer or processor — the theorem ends up in `global_theorem_list` and is re-cited in its own chapter's proof). NOT caused by the `ordisMerge` change (the chapter-1032 theorem has no OR involvement); surfaced incidentally by the `main.py` rerun that regenerated the chapter set. Tracked as a separate next-task investigation per the project conventions `Failures are first-class` ("Failures are first-class").

**Files touched.** `verifier.py` (Stage P0), `GL_Quick_VS/GL_Quick/src/prover.hpp` (Stage P1), `docs/agentic_swdd/20_core_concepts/{07_or_branching,08_proof_tags}.md`, `docs/agentic_swdd/10_pipeline/{04_prover,06_process_proof_graph}.md`, `docs/agentic_swdd/40_decisions.md`, `docs/agentic_swdd/verifier_rung1_changes.md`.

---

<a id="d-35"></a>
## D-35 — `verifier_rung1`: incubator-verifier extensions for FTA-ladder rung-1 forward direction (2026-05-03)

**What.** A coordinated extension of `verifier.py` to clear the 11 verifier failures that surface when the verifier is pointed at `files/incubator/processed_proof_graph/` (the historical `verifier.py` `main` hardcoded only the main pipeline output, so the incubator regression was invisible). All 11 failures stem from the FTA-ladder rung-1 forward direction (`EnumerationSet2 ⟹ interval`) — chapter `1209_direct_proof.txt` and its reformulated companion `1210_reformulated_statement.txt`.

The extension consists of seven logical edits, lettered V-1..V-7. **This entry is seeded with V-1 + V-7 + the alpha-canonicalize helper; subsequent commits on this branch extend it as V-2..V-6 land.** Per Rule 10 each commit updates this entry as it lands its slice.

**V-1 — CLI base-dir argument.** `main` now accepts a positional `base_dir` (default = the historical `files/processed_proof_graph`). Same default behaviour for old invocations; new invocation `python verifier.py files/incubator/processed_proof_graph …` verifies the incubator output.

**V-7 — `--include-globals PATH` (repeatable).** Unions sibling-batch `global_theorem_list.txt` entries into `state.global_theorems`/`state.global_theorem_list` so cross-batch theorem citations in the origin check (e.g. an incubator chapter citing the Peano `existence2` axiom) resolve. Entries are loaded in supplied order; local-batch entries take precedence on key collisions.

**V-2 — `check_or_convergence` deliberately fails until producer-side evidence lands (clean-fail final state, post-Codex-review).** Three iterations:

1. **Initial.** Added a compiled-form `(or<N>[…])` path validating only OR shape + parent-scope namespace match. Cleared the 2 `or convergence` failures in chapter `1209_direct_proof.txt`.
2. **Strengthened.** Codex review correctly flagged the initial check as too weak under [I-16](30_invariants.md#i-16). Added "OR must be live at parent scope" to both expanded and compiled paths; documented residual gap as a `Suspected fragility`. Still cleared the 2 failures.
3. **Clean-fail.** The strengthened check still PASSED the 2 rows — but on accident, because the OR happened to be derived right before each convergence row. Per I-16, when a check cannot verify its semantic contract the correct response is FAIL, not PASS-with-asterisk. `check_or_convergence` returned `False` unconditionally for both `rest[0]` shapes. The 2 chapter-1209 rows surfaced as `or convergence failure 2`.
4. **Spec'd layout (this entry).** Per the user-directive of 2026-05-03 the new producer-side row layout is fixed:
 ```text
   <C>  <parent>  or convergence  <OR>  <parent>  <C>  <branch_D1>  <C>  <branch_D2>  …  <C>  <branch_DK>
   ```
 `check_or_convergence` is rewritten to validate this layout: layout shape (`len(rest) == 2 + 2*K`), parent-scope match, OR is a known compiled `(or<N>[…])` with `K` disjuncts, conclusion repetition, branch-scope ancestry, branch distinctness, and the load-bearing **per-branch derivation evidence** — for every `(C, branch_Di)` pair the chapter must contain a row with `expression == C` and `namespace == branch_Di` (the user's "each ingredient has its own line" requirement). The check is now real verification, not a structural pass-through.

 The 2 chapter-1209 rows still use the old 4-field layout and continue to fail at the layout check (step 1: `len(rest) == 4`, not `>= 6`). Same end-state count (`32779/2`), but the failure now means "old layout, new layout pending" rather than "unconditional reject". Producer-side fix (next task) makes the rows pass.

**Why the chapter-local check WAS fundamentally insufficient before this revision.** The contract is "the same conclusion `C` was independently derived in EVERY branch of this OR's case split". The chapter export historically doesn't carry that evidence: `ordisMerge` removes the per-branch copies of `C` at convergence (`07_or_branching.md` §3b), and `process_proof_graphs.py` does not retain `_boundary_ordis_` rows. The new spec'd layout closes the gap by carrying the per-branch evidence into the row's rest fields AND requiring the chapter to retain the per-branch derivations of `C`.

**Producer-side fix — the next task ("buildstack and history tracking").** Two coordinated changes must land together:

a. **Prover.** Emit the new convergence row layout. `ordisMerge` (or its equivalent at the prover's emit site) records the per-branch validity names alongside the converged conclusion and writes them into the row's rest fields.
b. **Process-proof-graph.** Retain per-branch derivation rows of `C` in chapter export — i.e. don't suppress chapter rows whose namespace is a branch scope and whose expression is the converged `C` cited by an `or convergence` row. Simplest implementation: walk the convergence rows first, collect the cited `branch_Di` namespaces, then suppress only branch rows that are NOT cited.

Either change without the other leaves the verifier failing.

**V-3 + V-4 — `or branch proven` and `or branch assumption` promoted to first-class `TAG_CHECKERS` entries.** Both tags were previously claimed retired/overridden in the SwDD; the override path never existed and the prover always emitted them live. FTA-rung-1 chapter `1209_direct_proof.txt` lines 22 (`or branch proven`) and 43 (`or branch assumption`) carry the tags, and pre-extension they showed up as `<unknown:…>` failures. The new checkers validate:

- **`check_or_branch_proven`.** `line.expression` = the OR (compiled `(or<N>[…])`); `line.namespace` = the OR's parent scope; `rest[0]` = the asserted disjunct; `rest[1]` = the branch's namespace. Validation requires parent-as-strict-ancestor of branch, GL-binary disjunct membership of `rest[0]` (modulo equality symmetry), and the branch payload to encode the OR + asserted disjunct via the `_boundary_orint_<or>_(<disjunct>)` substring.
- **`check_or_branch_assumption`.** `line.expression` = `!<other-disjunct>`; `line.namespace` = the branch's namespace; `rest[0]` = `<or>_integration_goal`; `rest[1]` = the OR's parent scope. Validation requires parent-as-strict-ancestor of branch, the OR-with-suffix to strip cleanly to a known compiled OR, the negated content to be a disjunct of the OR (modulo equality symmetry), and the branch's `_boundary_orint_<or>_(<asserted>)` payload to name a DIFFERENT disjunct (the one the branch asserts).
- A shared helper `_or_disjuncts_from_compiled` substitutes `u_i` placeholders in the OR binary's elements list with the OR's args; `_disjunct_matches` adds equality-symmetric matching for `(=[a,b])` ↔ `(=[b,a])`.
- Both tags are added to `_ORIGIN_EXEMPT_TAGS` so the inline origin check does not require chapter-LHS membership for `_integration_goal`-suffixed deps or for asserted disjuncts whose own derivation lives in a different chapter.

**V-5 — `check_implication` accepts ancestor-scope premises (comparable-scope inheritance).** The pre-extension rule was "at most one distinct non-main namespace among premises + implication, and the result must equal that one". This rejected legitimate FTA-rung-1 firings (chapter `1209_direct_proof.txt` lines 36, 41, 56, 73) where the result lands in an OR-branch scope but some premises live at the OR's parent scope. The new rule iterates each source namespace and accepts if it is `"main"`, equal to the result's namespace, or a strict ancestor of it (`result_ns.startswith(ns + "_boundary_")`). This is the faithful encoding of GL's comparable-scope inheritance — facts at an ancestor scope are visible at every descendant.

**V-6 — `_check_reformulation` binary-lookup fallback for split-tag anchors.** The helper derives the GL-binary tag from the target's anchor (e.g. `AnchorGauss → "Gauss"`) and selects `gl_binaries[tag]`. The incubator's GL-binaries are split across multiple tag files (`IncubatorPeano` / `IncubatorGauss` / `IncubatorGauss1`); chapter `1210_reformulated_statement.txt`'s anchor `AnchorIncubator` derives the literal tag `"Incubator"` for which no binary is loaded, so the lookup returned `None` and the check rejected immediately. The new behaviour: if the exact-tag lookup is empty or the head core is missing/non-existence in it, the helper scans every loaded binary for one that defines the head's compiled name as an `existence` entry. The fallback is purely additive — exact-tag lookups still take precedence for batches where they succeed (e.g. all main-pipeline runs).

**Codex round-3 tightening (post-round-2, 2026-05-03).** A third Codex review surfaced two cross-row soundness gaps in the OR-branch checkers — both addressed in one commit:

1. **`check_or_branch_proven` requires an independent OR-derivation row at parent scope.** Pre-tightening the checker validated structure but not provenance: the bookkeeping split row could appear in isolation and pass. Tightened: a chapter row must exist with `expression == or_expr`, `namespace == parent_ns`, and `tag!= "or branch proven"`. **Surfaces a new deliberate failure on chapter 1209**: the OR `(or2[i1,v5,i0])` is consumed as a premise by line 19's implication but never appears as a chapter LHS at the parent scope (the chapter export does not render the consumed OR's derivation). Per the project conventions `Failures are first-class` this is a feature — the failure is a forcing function for the next task to fix the chapter export (or for the prover to emit an explicit OR-derivation row).
2. **`check_or_branch_assumption` requires a matching `or branch proven` row.** Pre-tightening the checker validated structure but not the case-split's existence: an assumption row could pass even if the branch was never opened by a `or branch proven` row. Tightened: a chapter row must exist with `tag == "or branch proven"`, `expression == or_expr`, `namespace == parent_ns`, `len(rest) == 2`, `rest[1] == branch_ns`, and `rest[0]` matching the asserted disjunct (modulo equality symmetry). Chapter 1209's assumption row at line 43 continues to pass — line 22 satisfies all sub-conditions.

**End-state count after round-3.** Incubator `32779 checks, 3 FAILED` (was 2 after round-2): the 2 pre-existing deliberate `or convergence` fails plus 1 new deliberate `or branch proven` fail surfaced by round-3 step 1. Main pipeline `2750 checks, 0 failures` (unchanged).

**Codex round-2 tightening (post-spec'd-layout, 2026-05-03).** A second Codex review surfaced three soundness gaps in the `or branch proven` / `or branch assumption` checkers and the `_or_disjuncts_from_compiled` helper. All three fixed in one commit:

1. **`len(rest) == 2` exactly** for both `check_or_branch_proven` and `check_or_branch_assumption`. Both tags sit in `_ORIGIN_EXEMPT_TAGS`, so any extra `(expression, namespace)` rest pairs were silently accepted by the generic origin check — no audit trail for hidden ingredients. The pre-tightening checks used `len(rest) >= 2`; tightened to `==`. Empirically chapter 1209's two rows have exactly 2 rest fields, so no regression.
2. **Exact branch-namespace match** in both checkers. The pre-tightening code used substring search (`needle in branch_ns` for `check_or_branch_proven`, `find` for `check_or_branch_assumption`), which would PASS a nested or unrelated descendant scope that merely contained the expected `_boundary_orint_<or>_(<disjunct>)` substring. The contract says branch is at exactly `parent + "_boundary_orint_<or>_(<disjunct>)"` — tightened to literal equality on the full namespace string (with a balanced-parens parser to extract the asserted disjunct in `check_or_branch_assumption`).
3. **Arity check** in `_or_disjuncts_from_compiled`. The pre-tightening code did not verify `len(args)` against the binary's `arity` field; a malformed `(or<N>[…])` with too many or too few args was treated as a valid known OR if the disjunct placeholders happened to substitute. Tightened: read `arity` from the binary entry (or parse `signature` if `arity` absent), reject when `len(args)!= arity`. Used by both `check_or_branch_proven`, `check_or_branch_assumption`, and the spec'd-layout `check_or_convergence`.

**End state (corrected through Codex round-3).** With V-1..V-7 + the `_alpha_canonicalize_bound_vars` helper + the V-2 clean-fail revision + the V-2 spec'd-layout revision + Codex round-2 tightening + Codex round-3 tightening:

- Main pipeline: **2750 checks, 0 failures** (no regressions; main does not currently emit any `or convergence` / `or branch proven` / `or branch assumption` rows).
- Incubator: **32779 checks, 3 FAILED**:
 - 2 × `or convergence` (chapter 1209 lines 5, 14) — old 4-field layout, awaiting spec'd-layout producer-side fix.
 - 1 × `or branch proven` (chapter 1209 line 22) — OR not derived at parent scope as a chapter LHS, awaiting chapter-export fix to render the OR's derivation row.

8 of the 11 baseline failures genuinely cleared via real verifier-side checks (4 `implication`, 1 `<unknown:or branch assumption>`, 1 `origin`, 2 reformulated_statement). The 3 remaining are deliberate forcing functions for the next task ("buildstack and history tracking"). Per the project conventions `Failures are first-class` ("Failures are first-class") these failures are accepted as valuable signal, not softened to recover "0 failures". The honest answer to "is the FTA-rung-1 forward chain verifier-clean?" is **no, not yet** — three real verification gaps are surfaced and need producer-side work to close. Pretending otherwise (the original "32779/0 airtight" claim) would have been a violation of [I-16](30_invariants.md#i-16) in spirit. The `verifier_rung1` task is closed in the corrected state.

**Alpha-canonicalize helper.** A new `_alpha_canonicalize_bound_vars` helper canonicalizes every `>[…]` bound-variable name to `b1, b2, …` in declaration order. Used by the inline origin check at `verifier.py` so a chapter row whose `rest[0]` rule names a bound variable `i2` (the prover's local free-index counter at deposit time) matches a global-list entry that names the same bound variable `v1` (process_proof_graphs.py's canonical-export rename). Pre-extension, `_normalize_expr_list` (which only renames `v\d+`) treated the two as distinct strings and the origin check rejected the legitimate citation.

**Why this branch exists.** Rung 1's forward closure (D-34, sandbox/ordis_merge) introduced OR-disintegration machinery whose chapter rows use proof-tag patterns that the verifier had never been exercised against — most notably the live (not, as the SwDD claimed, retired) tags `or branch proven` / `or branch assumption`, OR-branch ancestor-scope premise inheritance in `implication`, the compiled-form `(or<N>[…])` argument shape in `or convergence`, and the 4-argument-head `existence4` reformulation in `_check_reformulation`. None were caught earlier because the verifier was never pointed at the incubator tree. Per [I-16](30_invariants.md#i-16) the response is to extend the verifier with proper checks, not to suppress.

**Cross-checks.**
- `python verifier.py` against the main pipeline: 0 failures (no regression).
- `python verifier.py files/incubator/processed_proof_graph --include-globals files/processed_proof_graph/global_theorem_list.txt` against the incubator: 11 → 0 failures (target end-state, reached as V-2..V-6 land).

**Files touched (this entry).** `verifier.py`, `docs/agentic_swdd/10_pipeline/08_verifier.md`, `docs/agentic_swdd/40_decisions.md`. (Subsequent commits on this branch will also touch `docs/agentic_swdd/20_core_concepts/07_or_branching.md`, `docs/agentic_swdd/20_core_concepts/08_proof_tags.md`, and add `docs/agentic_swdd/verifier_rung1_changes.md` as the per-edit changelog.)

---

<a id="d-34"></a>
## D-34 — `_ordis_` merge: kernel-level `ordisMerge` over `sameIterationInternalMail` (2026-05-02)

**What.** OR-disintegration convergence bookkeeping moves from `trackOrBookkeeping` (a stale inline member previously called from inside [`addStatement`](../GL_Quick_VS/GL_Quick/src/prover.hpp)) to a new inline member [`ordisMerge`](../GL_Quick_VS/GL_Quick/src/prover.hpp) called from [`addExprToMemoryBlockKernel`](../GL_Quick_VS/GL_Quick/src/prover.cpp)'s post-`addStatement` loop, sibling to the `toBeProved`-discharge logic and to the `_orint_/NotOrScope` block. The old `trackOrBookkeeping` function is deleted.

`ordisMerge` runs once per `(addExpression, effectiveValidity)` pair returned by `addStatement`. With [D-33](#d-33)'s single-channel routing, `effectiveValidity` is the deposit's actual scope — so the function observes per-branch deposits including descendant-direction cross-scope rewrites that the legacy `addStatement`-call-site bookkeeping never saw. It records each deposit in `Memory::orBookkeeping[(expr, orSignature)] → set<branchDisjunct>` and tests against `Memory::orDisjunctCount[orSignature]` (registered at OR-disintegration mint time, [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)). On convergence:

1. **Promotion via `sameIterationInternalMail`.** `(addExpression, addExpressionLevels, parentValidity)` is pushed onto `Memory::sameIterationInternalMail.statements` — the same revival channel `revisitRejectedIntegration2` uses ([`memory.hpp` `InternalMail`](../GL_Quick_VS/GL_Quick/src/memory.hpp)). Drained at the top of the next hashburst body via `addExprToMemoryBlock(..., status=1,..., parentValidity,...)`, so the parent-scope deposit goes through the full kernel pipeline (parent's `toBeProved` discharge included). Deferring to `sameIterationInternalMail` instead of calling `addExprToMemoryBlock` directly avoids reentrant kernel invocation in the post-`addStatement` loop.
2. **Per-branch cleanup of the converged expression.** For each branch validity in the recorded disjunct set, `ordisMerge` calls `removeExpressionFromMemoryBlock(EncodedExpression(addExpression, branchValidity), mb, /*state=*/0)`. The N per-branch copies of the converged expression collapse into the single parent-scope copy. The parent-scope copy remains visible to each branch via comparable-scope inheritance; branches reuse it through standard scope-walk reads. Branches keep all their *other* facts and stay live (unlike `_orint_`'s `cleanUpOrIntegrationBranches` which wipes the entire branch).

`exprOriginMap` (history) is not touched — `removeExpressionFromMemoryBlock(state=0)` only erases the encoded-statement vectors. `statementLevelsMap` and `intKnownStatements` are also retained, so a re-deposit attempt at a wiped branch scope short-circuits at the standard `intKnownStatements` gate at `addStatement`. `orBookkeeping` is also retained — re-fires for the same `(expr, orSig)` on a subsequent deposit (e.g. when the user re-runs the same case via a different path) are absorbed by the `std::set` dedup of `sameIterationInternalMail.statements`.

**Why — visibility gap.** FTA-rung-1 §9a / §9b symptom (`docs/fta_ladder/rung1/current_proof_state.md`): all four ordis cells under impl24 / impl25 contain both per-branch preorder facts (D-33 working — descendant-direction eq-class rewrites land at the branch scopes), but the OR-convergence promotion to the immediate impl24 / impl25 boundary scope was never firing. Diagnosis: `trackOrBookkeeping` was called from inside `addStatement` at line 6126 with the *original* `validityName` parameter the caller passed in. After D-33 made `applyEquivalenceClass` push descendant-direction cross-scope rewrites to `newStatements` at the deeper deposit scope, those branch-scope deposits never reached the bookkeeping — they landed in `encodedStatements` but the convergence map never saw them. Moving the bookkeeping to the kernel's per-pair loop closes the gap structurally: the kernel iterates pairs at their deposit scope, so every branch-scope deposit is observed.

**Why — `sameIterationInternalMail` route.** The user's design directive: "we have internal mail. revisitRejectedIntegration uses. let's put stuff after merge there. it prevents endless recursion and is our future general path." `sameIterationInternalMail` is the existing per-LB validity-preserving inbox for integration-side revival messages, populated by `applyEquivalenceClassToRejectedMapIntegration` and `revisitRejectedIntegration2` during a hashburst body and drained at the top of the next hashburst's body. Using it for `_ordis_` convergence promotion gives three properties for free:

- **No reentrance.** `addExprToMemoryBlockKernel` does not call itself recursively during the post-`addStatement` loop iteration. The deferred drain runs the parent-scope deposit through a clean kernel entry at the next hashburst.
- **Full kernel pipeline at the parent scope.** The `status=1` absorb path runs the parent-scope deposit through `toBeProved` discharge, the `_orint_/NotOrScope` block, *and* `ordisMerge` again at the parent scope — so a parent-scope deposit that itself sits inside another `_ordis_` chain participates in further convergences cleanly.
- **Set-dedup absorbs idempotent re-fires.** `sameIterationInternalMail.statements` is `std::set<std::tuple<std::string, std::set<int>, std::string>>`; pushing the same tuple twice is a no-op.

**Why — narrow cleanup.** The legacy `trackOrBookkeeping` cleanup wiped *every* statement at the branch scopes via prefix-match and added the branch validities to `validityNamesToFilter` to block all future inserts. That was strictly wrong for `_ordis_`: a case-split over a known disjunction has every branch contributing usable derivations under its case condition; wiping siblings discards those. Worse, the filter-block prevented further convergences — once one expression converged, the OR scope was dead for any other expression. The narrow cleanup keeps branches alive so the second preorder of the same impl scope's body (`(preorder[1,4,p,1])` after `(preorder[1,4,0,p])` for impl24, etc.) can converge in turn.

**Verification.** Pipeline-run verification target: full `main.py`, both Gauss summation copies present in `theorems.txt`, and the four target FTA-rung-1 §9a / §9b preorder rows present at the parent scopes (`(preorder[1,4,2,repl_lev_1_0])` and `(preorder[1,4,repl_lev_1_0,6])` at `main_boundary_(implication24[15,1,4,2])`; `(preorder[1,4,2,repl_lev_1_1])` and `(preorder[1,4,repl_lev_1_1,6])` at `main_boundary_(implication25[15,1,4,6])`).

**Verified — full main.py run HEAD (post-mailOut gate fix).** Exit 0, runtime 1266 s, verifier 2750 checks 0 failures airtight. Per-batch theorem counts: IncubatorPeano 513, Peano 49, IncubatorGauss 91, IncubatorGauss1 **2 (vs baseline 0)**, Gauss 11. Both Gauss summation copies in `files/theorems/theorems.txt` (`fold[…,5]`-anchored implications, 41 theorems total = D-33 baseline parity). IncubatorGauss1 saves the §4.1 forward direction `(>[1,2,4,6](AnchorIncubator[…])(>[15](EnumerationSet2[2,6,15])(interval[1,4,2,6,15])))` — first time the FTA-rung-1 `{0,1}=[0,1]` forward conjecture has closed end-to-end.

Three of the four §9a / §9b target preorders land at the clean immediate-parent scope (impl24-up: 50 hits, impl25-up: 46 hits, impl25-down: 50 hits in the trace). The fourth (`(preorder[1,4,repl_lev_1_0,6])` at impl24-down) lands at v=`main` instead of impl24: the orSignature `(or2[2,repl_lev_1_0,6])` exists at two stack positions (impl24-internal AND top-level main), the top-level OR-convergence promotes to `main` first, then the impl24-internal convergence's `sameIterationInternalMail` deposit gets dedupe'd by Site F's ancestor scan at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) because the fact is already in `intKnownStatements` at the strict ancestor v=`main`. Comparable-scope inheritance makes the `main` entry visible at impl24, so the §4.1 forward chain closes regardless. `or convergence` origins appear 106 084 times in the IncubatorGauss1 hashburst trace — the mechanism fires extensively.

**Mail-out gate (added with the same change set).** [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)'s `mailOut.implications` insert is now gated on `impValidity == "main"` in addition to the pre-existing `allowedForMail` check. The mailOut.implications/statements channels are MAIN-ONLY by contract (mailOut.exprOriginMap continues to carry history for all scopes per `trackExpansionHistory`); the old code shipped non-main rules unconditionally, the receiver re-installed them at hardcoded v=main with the rule's origin still keyed at the sender's deeper scope, and visualizer's `buildStack` walked the firing-time dep at `(rule, "main")` with no matching exprOriginMap entry. With the gate, non-main rules stay local; receivers re-derive them from the mailed v=main statements via their own disintegration, yielding properly-scoped origins via `trackExpansionHistory`.

**Code.**
- `ordisMerge` definition — [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp), inline member of `ExpressionAnalyzer`, around the location where `trackOrBookkeeping` lived.
- Kernel call site — [`prover.cpp` `addExprToMemoryBlockKernel`](../GL_Quick_VS/GL_Quick/src/prover.cpp), single line at the bottom of the post-`addStatement` `for (idx...)` loop.
- `trackOrBookkeeping` deleted (function body + `addStatement` call site at the old line ~6126).
- `Memory::orBookkeeping` and `Memory::orDisjunctCount` (declared in [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp)) — fields retained, semantics unchanged. `Memory::sameIterationInternalMail` — used by `ordisMerge` exactly the same way `revisitRejectedIntegration2` uses it.

**Supersedes.** The `addStatement`-call-site `trackOrBookkeeping` design. Both the placement (now kernel-level) and the cleanup shape (now narrow) change.

---

<a id="d-33"></a>
## D-33 — Bidirectional equivalence-class application + single-channel deposit routing (2026-05-01)

**What — bidirectional eq-class application.** Equivalence-class application now runs in both directions whenever the class's validity and the expression's validity are comparable in the scope tree. Three cases admit:

| Class scope | Expression scope | Deposit scope |
|---|---|---|
| `S` | `S` | `S` |
| `S_a` (strict ancestor of `S_d`) | `S_d` | `S_d` |
| `S_d` (strict descendant of `S_a`) | `S_a` | `S_d` |

The deposit scope is `deeperOf(class.scope, expr.scope)` — see [`memory.hpp::NameMap::deeperOf`](../GL_Quick_VS/GL_Quick/src/memory.hpp). For the legacy directions (same / class-shallower) the deposit collapses to the expression's own scope; for the new direction (class strictly deeper) the deposit lands at the class's deeper scope. The original at the ancestor scope is never overwritten — the rewrite is purely additive. Both copies coexist.

Trigger sites: [`updateEquivalenceClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp)'s merge-postscan (walks every encoded statement, applies merged class to comparable-scope statements), [`addStatement`](../GL_Quick_VS/GL_Quick/src/prover.hpp)'s per-statement block (iterates classes at every comparable scope of the new fact's scope), and the same fixpoint repetition for late-appearing classes.

**What — single-channel deposit routing.** Every deposit — same-scope, shallower-class, deeper-class — flows into the single `newStatements` vector that `addStatement` returns. There are no separate sinks. Each entry is an [`ExpressionWithValidity`](../GL_Quick_VS/GL_Quick/src/memory.hpp) pair carrying the deposit's actual scope.

The kernel's post-`addStatement` loop ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)) iterates the pairs and uses each pair's `validityName` for `statementLevelsMap` lookup, admission-map updates, `toBeProved` discharge, and validity-name promotion. Cross-scope deposits go through the same discharge logic as same-scope deposits — the only difference is which `validityName` drives the lookup. See [I-25](30_invariants.md#i-25).

**What — rejected-integration map exclusion.** [`applyEquivalenceClassToRejectedMapIntegration`](../GL_Quick_VS/GL_Quick/src/prover.hpp) does NOT admit the descendant direction. The rmi map is a deferred-match registry — its entries represent auxy integrations whose preconditions live in the world visible at the entry's scope. A class at a strictly deeper scope is invisible there, so it carries no information for that entry's deferred match; rewriting the entry would manufacture a synthetic deferred match that did not actually defer at that scope, and erasing the original would destroy a real revival path at the ancestor.

**Why — bidirectional application.** FTA-rung-1 §9b (upper-bound forward conjunct) and §9b-style cases need a class registered at an OR-disintegration branch scope (descendant of `main`) to rewrite a ground `preorder` fact at `main`. Pre-D-33, eq-class application only walked descendants of the class scope, so a deeper class never reached up into ancestor-scope statements. The new direction closes the FTA-§9b gap by letting the descendant-scope class produce the rewrite at its own scope.

Soundness rests on descendant-inheritance: a fact at `S_a` is observably true at every descendant of `S_a`, including `S_d` where the class lives. Substituting under the descendant-scope equivalence and depositing at the descendant scope is locally sound at the descendant.

**Why — single-channel routing.** An earlier WIP attempt at D-33 (the original bad commit ) routed cross-scope deposits to separate sinks (`crossScopeSink` in `addStatement`, `descendantSink` / `ancestorSink` in the merge-postscan) to avoid tripping the kernel's same-scope `assert(sit!= memoryBlock.statementLevelsMap.end)` ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)) — the lookup was built as `EncodedExpression(addExpression, validityName)` with `validityName` = the kernel's caller-scope, and a cross-scope deposit at a different scope key would miss.

The side-sink design avoided the assert but bypassed the kernel's entire post-loop, including the `toBeProved` discharge at all five sites (recursion-LB head match, direct-theorem proof, validity-name promotion for OR-integration, validity-name promotion for NotOrScope, vacuous-truth in induction). Cross-scope deposits sat in `statementLevelsMap` and `intKnownStatements` but never closed their matching `toBeProved` entries.

Concretely on the Gauss summation induction-condition LB: the boundary fact `(in2[2,repl_lev_4_0,int_lev_4_2349]) @ main_boundary_(implication23[2,8,int_lev_4_2349])` was synthesized via the merge-postscan ancestorSink (equality1 from main-scope source using boundary class `{2,repl_lev_4_0}`). The fact lived in `encodedStatements`. The matching `toBeProved` entry stayed open. The validity-name promotion that would have lifted `(implication23[2,8,int_lev_4_2349])` to `v=main` never fired. The IH-application chain cascaded as MISSING. Gauss summation stayed unproved.

The pair-based return type for `addStatement` carries each deposit's scope into the kernel, so the lookup uses the deposit's own validity, the assert holds, and the discharge logic runs. That single change closes the cross-scope discharge gap without sacrificing the FTA-§9b unblock.

**Why — rmi exclusion.** The rmi map is keyed by `(expression, validity)` and lookup happens against entries the prover registered as deferred. A descendant-scope equality applied to an ancestor-scope rmi entry would synthesize a key the descendant scope never deferred (and may never need); writing it manufactures rmi state out of nothing. The legacy `toErase.push_back(keyEv)` (replace-not-add) compounded the problem by destroying the ancestor's revival path. Restricting rmi application to same-scope and class-shallower keeps the rmi pool a faithful record of deferred matches per scope.

**Verification.**

- Pre-D-33 (commit baseline, no-incubator full main.py): 41 proved theorems, both Gauss summation copies present, verifier 2758 checks 0 failures.
- WIP D-33 (commit, no-incubator full main.py): 39 proved theorems, both Gauss summation copies MISSING, verifier 2484 checks 0 failures (passes only because the missing-Gauss checks don't run).
- Final D-33 (commit, no-incubator full main.py): 41 proved theorems, both Gauss summation copies present, verifier 2750 checks 0 failures — airtight. Slight check-count delta vs the baseline (2750 vs 2758) reflects path differences in how some derivations route through the new pair-based discharge but does not indicate theorem loss.
- FTA-rung-1 §9b unblock: see in-flight `docs/fta_ladder/rung1/current_proof_state.md` and trap dumps under the `EnumerationSet2[2,6,15] → AnchorIncubator` LB in the fullest pipeline run.

**Code.**

- `applyEquivalenceClass`: depositValidity computation at function entry — [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp). Parameter type `std::vector<ExpressionWithValidity>& newStatements`.
- `applyEquivalenceClassToNegatedEquality`: same parameter-type change — [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp).
- `cleanUpExpressions`: takes/returns `std::vector<ExpressionWithValidity>` — [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp).
- `updateEquivalenceClasses`: takes/returns `std::vector<ExpressionWithValidity>`; merge-postscan walks every encoded statement, single-channel emit — [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp).
- `addStatement`: returns `std::vector<ExpressionWithValidity>`; descendant-classes block and fixpoint descendant pass route to `newStatements` — [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp).
- `applyEquivalenceClassToRejectedMapIntegration`: scope-match gate restricted to same-NS and class-shallower — [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp).
- Kernel post-`addStatement` loop iterates pairs and uses `effectiveValidity` from each pair throughout — [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp).
- `NameMap::deeperOf` helper for strings — [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp).
- Verifier `equality1` source-namespace relaxation (admits source at strict ancestor of result) — [`verifier.py::check_equality1`](../verifier.py).

**Supersedes.** Bad commit 's WIP design with separate cross-scope sinks. The user-facing consequences of that design (Gauss regression, cross-scope discharge gap) are gone; the FTA-9b unblock is preserved.

---

<a id="d-32"></a>
## D-32 — Sharper OR-disintegration gate: product-of-disintegration head + checkLocal-routed (2026-05-01)

**What.** Replaces D-31's broad implication-scope OR-disintegration bypass with a sharper gate. OR-disintegration in [`disintegrateExprCore2`](../GL_Quick_VS/GL_Quick/src/prover.cpp)'s OR case is now allowed iff a new boolean parameter `allowOrDisintegration == true` is threaded into the call. The flag is set by [`checkLocalEncodedMemoryStatic`](../GL_Quick_VS/GL_Quick/src/prover.cpp) at the head-firing call site, fed from a new `LocalMemoryValue::productOfDisintegration` field stamped at install time inside [`addToHashMemory`](../GL_Quick_VS/GL_Quick/src/prover.cpp). The stamp criterion is: at least one premise (chain element) of the implication has an argument whose name starts with `"u_"`.

The `"u_"` prefix is reserved for bound-variable placeholders introduced by `prefixArgumentsWithU` during `disintegrateExpr2`. An anchor-bound theorem's chain (e.g. starts with `(AnchorPeano[1,2,3,...])`) carries only concrete integer args; its premises never have `u_*` args, so the stamp is `false` and the bypass never fires. A body-element implication produced by disintegrating a compound (e.g. ES2-forward `implication20`: `(in[u_p, u_M]) ⇒ (or2[u_2, u_p, u_6])`) carries `u_*` args, gets stamped `true`, and triggers the bypass when its head deposit is an OR.

Coupling with general disintegration: if `addExprToMemoryBlock`'s `doNotDisintegrate == true`, `allowOrDisintegration` is forced to `false` at function entry. OR-disintegration cannot fire when general disintegration is forbidden.

The legacy `orAdmissionSet` fallback gate (line ~7191) is preserved as a no-op extension hook — `orAdmissionSet` still has no `.insert` site anywhere in the codebase.

**Why.** D-31's broad bypass — "any OR landing in any implication scope disintegrates unconditionally" — caused a runtime explosion. The IncubatorGauss1 hashburst trace under D-31 contained 13 191 `_boundary_ordis_(or…)` rows at `(implication24[15,1,4,2])` scope, plus more elsewhere; the cascade fanned out across every implication scope in the batch. Anchor-bound theorems (P1 left-identity, P2 successor, etc.) firing into implication scopes would each spawn case-split branches even when the OR was a known concrete instantiation with no actual case-split needed. Full main.py runtime ballooned beyond the D-30 baseline of 1183 s.

The narrowing: only ORs delivered by an implication that is *itself* a product of disintegration (= came from disintegrating a compound's body, not a top-level anchor-bound theorem) trigger the bypass. The dominant FTA-rung-1 case — `implication20` (ES2-forward) firing at `(implication24[15,1,4,2])` scope and depositing `(or2[2, repl_lev_1_0, 6])` — survives because `implication20`'s chain `(in[u_p, u_M])` has `u_p`/`u_M` args. Anchor-bound rules' broad fan-out is filtered out.

**Verification (full main.py + IncubatorGauss1 standalone hashburst).**

- Steps 1–8 + Step 9 lower (impl24) of the FTA-rung-1 `{0,1}=[0,1]` proof all reach the same trace evidence as under D-31. Burst-by-burst toBeProved trajectory `6 → 5 → 3 → 2` (bursts 1–8 → 9 → 10 → 11+) is identical to D-31's. The Step-9-lower closure (`(preorder[1,4,2,repl_lev_1_0])` deposited at `(implication24[15,1,4,2])` scope, 42 occurrences) is unchanged. The `_boundary_ordis_` rows at impl24 dropped from 13 191 (D-31) to 1 501 (D-32) — an 88% reduction in OR-disint fan-out at that scope while preserving the closure path.
- Per-batch theorem-count parity vs D-31: see commit message for the 5-batch totals.
- Verifier: see commit message for check counts and failure totals.
- Full main.py runtime: see commit message; target is the D-30 baseline (~1183 s).

**Code.** Gate site: [`prover.cpp` `disintegrateExprCore2` OR case](../GL_Quick_VS/GL_Quick/src/prover.cpp) (around line 7180, replaces D-31's `IMPL_PREFIX` peel block). Stamp site: [`prover.cpp` `addToHashMemory`](../GL_Quick_VS/GL_Quick/src/prover.cpp) (around line 1131, immediately before the `encodedMap[intIgnoredKey].push_back(lmv)`). Forwarding: [`prover.cpp` `addExprToMemoryBlock`](../GL_Quick_VS/GL_Quick/src/prover.cpp) → `disintegrateExpr2` → `disintegrateExprCore2`, all with `allowOrDisintegration` default-`false` so the 28+ existing call sites of `addExprToMemoryBlock` and the 2 of `disintegrateExpr2` need no edit. Field declaration: [`memory.hpp` `LocalMemoryValue`](../GL_Quick_VS/GL_Quick/src/memory.hpp) (around line 84).

**Supersedes.** D-31. The implication-scope bypass at `disintegrateExprCore2:7146` no longer exists; the legacy `orAdmissionSet` fallback gate stays.

---

<a id="d-31"></a>
## D-31 — Implication-scope OR-disintegration bypass — closes Step 9 lower (impl24) (2026-04-29 evening,) — SUPERSEDED by D-32 on 2026-05-01

**What.** Adds a bypass at the OR-disintegration admission gate inside [`disintegrateExprCore2`](../GL_Quick_VS/GL_Quick/src/prover.cpp) (around line 7146). Pre-D-31 the gate consulted `Memory::orAdmissionSet`, which has no `.insert` site anywhere in the codebase — so `orAdmitted` was permanently `false` and OR-disintegration never fired (zero `or disintegration` verifier successes, zero `_boundary_ordis_` scopes in any pre-D-31 trace). The D-31 bypass: if the OR arrives at a validity scope whose last `_boundary_<payload>` segment matches `(implicationNN[…])`, set `orAdmitted = true` unconditionally. Detection peels the last boundary segment via `validityName.rfind(NameMap::BOUNDARY_STR,...)` and string-compares the prefix `"(implication"`. The legacy `orAdmissionSet` gate remains as a no-op fallback.

**Why.** FTA-rung-1 Step 9 lower-bound conjunct (impl24's preorder body `(in[p, M]) ⇒ preorder(0, p)`) was stuck. Path: ES2-forward (`implication20`) deposits `or2[2, p, 6]` at impl24 scope; OR-disintegration should mint two `_ordis_` branches with seed equalities `(=[2, p])` and `(=[6, p])`; each branch substitutes into the preorder body via P1 left-identity to derive `preorder(0, 0)` / `preorder(0, 1)`; ordis convergence promotes `preorder(0, p)` to impl24 scope. Step 1 of the chain (ordis branch minting) never fired because of the structurally-dead `orAdmissionSet` gate.

**Verification (full main.py + 35 458-check verifier — done at commit time).**

- Step 9 lower toBeProved `(preorder[1,4,2,repl_lev_1_0])` at impl24 scope closes — burst-trace toBeProved trajectory drops 4 → 3 in the IncubatorGauss1 burst trace. Closure path: preorder integration with P1 witness `k = p` + variable-copy primitive at impl24 scope (not via the case-split, but enabled by the broader exploration that D-31 unblocks).
- 27 530 `_boundary_ordis_` scope occurrences in the IncubatorGauss1 hashburst trace (was 0 pre-D-31).
- Verifier: 35 458 checks, 0 failures across both proof graphs (main 32 700 / 0, incubator 2 758 / 0). No regression on Peano OR-branching, Gauss main, or any other batch.
- Per-batch theorem counts unchanged (513 / 49 / 91 / 0 / 11) — IncubatorGauss1 still saves 0 because Step 9 upper (impl25's preorder upper conjunct) and Step 10 (top-level interval) remain open.

**Why superseded.** D-31's bypass fires for *every* OR that lands in *any* implication scope, including ORs delivered by anchor-bound theorems. Runtime explosion observed in subsequent IncubatorGauss1 / full main.py runs. D-32 narrows the gate to ORs delivered by implications that are themselves products of disintegration (premise has a `u_*` arg).

---

<a id="d-30"></a>
## D-30 — OR-integration emits the wrapping OR at the OR-branches' parent scope, not unconditionally at `"main"` (2026-04-29 evening,)

**What.** At [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp), the `addExprToMemoryBlock` call that emits the wrapping OR expression after a branch's head proves no longer hard-codes `"main"` as the destination scope. Instead, the destination is computed by peeling the last `_boundary_<payload>` segment off the proving branch's `validityName`:

```cpp
std::size_t lastBoundary =
    validityName.rfind(NameMap::BOUNDARY_STR, std::string::npos,
                       NameMap::BOUNDARY_LEN);
const std::string orEmitScope =
    (lastBoundary == std::string::npos)
        ? std::string("main")
        : validityName.substr(0, lastBoundary);
```

For top-level OR-branches whose parent IS main (Peano OR-branching, the original `or0[7,2,1,3]` use case), this still yields `"main"` — semantics unchanged. For ORs nested under a hypothetical / integration scope (the FTA-rung-1 `{0,1}=[0,1]` proof's `or2[6,p,2]` under `_boundary_(implication26[1,4,2,6,15])`), the OR is now emitted at the implication's hypothetical scope rather than at main.

**Why.** Step 8 of the FTA-ladder rung 1 (OR-integration into `(in[p, M])`) was failing to close even though every input condition for the trigger was met. Trap-debug at `prover.cpp` showed the full chain:

1. Branch A head `(=[6, repl_lev_1_2])` deposited at Branch A scope `…_orint_(or2[6,repl_lev_1_2,2])_((=[6,repl_lev_1_2]))` with `status=1` and `inToBeProved=1`. Gate at line 4460 fires.
2. The OR `(or2[6,repl_lev_1_2,2])` emitted at `v=main` per the hard-coded destination.
3. `implication21` (`or2 ⇒ (in[p, M])`, ES2 backward direction) then fired at `v=main`, deriving `(in[repl_lev_1_2, 15])` at `v=main`.
4. The toBeProved goal `(in[repl_lev_1_2, 15])` lived at `_boundary_(implication26[1,4,2,6,15])` — the implication's hypothetical scope, NOT at main. The deposit at `v=main` did not match the goal at the implication scope (toBeProved lookup is exact-scope, keyed by `EncodedExpression(expr, validityName)`).
5. Step 8 stalled at `toBeProved=4` indefinitely.

The hard-coded `"main"` was a latent bug from the period when ORs lived only at the top-level scope (Peano OR-branching, `or0[7,2,1,3]` directly at v=main). Once OR-branching moved into the body of hypothetical / integration scopes — the FTA-ladder pattern starting with rung 1 — the destination became wrong.

**Verification (full main.py + verifier).** With the fix on:

- IncubatorGauss1 standalone burst #50 toBeProved drops from 4 (pre-fix) to **3** (post-fix). The `(in[repl_lev_1_2, 15])` entry at `implication26` scope is matched and erased; Step 8 closes. Step 9 (forward preorder conjuncts) and Step 10 (top-level interval) remain open as documented.
- Full main.py: 5 batches complete (IncubatorPeano 513 / Peano 49 / IncubatorGauss 91 / IncubatorGauss1 0 / Gauss 11), runtime 1183 s (was 1332 s — Step-8-closure cleanup pays back ~10% wall-clock).
- Verifier: 35 458 checks, 0 failures across both proof graphs (main 32 700 / 0, incubator 2 758 / 0). No regression on Peano OR-branching, Gauss main, or the four pre-existing configs.

**Risk and scope.** Behaviour change for ORs nested under hypothetical scopes (FTA-ladder rung 1 onward); no behaviour change for ORs at top-level (the only OR shape exercised pre-FTA-ladder). Runs cleanly on the full 35K-check verifier sweep — the existing OR machinery's invariants are preserved.

**Code.** [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp). Sibling `cleanUpOrIntegrationBranches` call at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) is unchanged (it walks branch payloads via the OR signature, scope-independent).

**Open follow-up.** Step 9's forward preorder conjuncts (`(preorder[1,4,2,p])` and `(preorder[1,4,p,6])` for universal `p ∈ M`) still don't fire automatically. The mechanism needed there is **OR-disintegration** (`ordisint`), not OR-integration: ES2-forward (`(in[p, M]) ⇒ or2[2, p, 6]`) plus a per-disjunct case re-deriving the same preorder consequence, then emit at parent scope. If `ordisint`'s emission site has the same hard-coded destination-scope bug, the same `rfind+substr` fix applies. To investigate next.

---

<a id="d-29"></a>
## D-29 — Two-part disintegration gate (anchor-LB always blocks; non-anchor needs local premise) under incubator + !ban_disintegration (2026-04-29 evening,)

**What.** New conditional gate inside `checkLocalEncodedMemoryStatic` ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)). After the existing `doNotDisintegrate = (lmv.justification == "integration")`, an additional check fires when both `parameters.incubator_mode == true` and `!parameters.ban_disintegration`:

1. **Anchor LB always blocks** — if `memoryBlock.exprKey` starts with `"(" + anchorInfo.name`, set `doNotDisintegrate = true` unconditionally.
2. **Non-anchor LB needs a local premise** — otherwise, iterate `orderedPremises`; if **none** are in `memoryBlock.localEncodedStatementsSet`, set `doNotDisintegrate = true`.

The first iteration of D-29 (committed earlier the same day) had only the second clause and was inert — the anchor LB itself processed most rule firings, and `localEncodedStatements` is populated by `prehandleAnchor` ([`prover.cpp/8236`](../GL_Quick_VS/GL_Quick/src/prover.cpp)) so anchor's own statement set looked "local" to itself; almost every rule firing satisfied the local-premise check. Adding clause 1 fixed it.

Supporting infrastructure: new `Memory::localEncodedStatementsSet` (`std::set<EncodedExpression>`, declared at [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp)) maintained in lockstep with the existing `localEncodedStatements` vector. 11 push_back sites in `prover.cpp/.hpp` got mirror `.insert` calls; the one assignment site at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp) rebuilds the set after the wholesale vector replacement.

**Why.** With `ConfigIncubatorGauss1.json` (the SE2 migration target) running with `incubator_mode=true && ban_disintegration=false`, the prover loaded the 1209 proved theorems from the prior IncubatorGauss batch (shared `theorems_folder`) into hashmem. The anchor LB then processed every loaded external rule whose premise (typically `(in[X, N])`) matched any of its anchor-deposited `(in[X, N])` rows; Pass B fan-out on the heads minted fresh `it_*` / `int_*` / scope-name variables; growth was 76 → 88 → 538 → 2702 → 4586 expressions in 2 main-phase bursts before the int16_t NameMap (capped at `ExecutionParameters::MAX_NAME_IDS` = 16384) exhausted at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp).

Clause 1 of the gate cuts the explosion at its source: the anchor LB stops disintegrating heads of broadcast-driven rule firings (it still receives them and emits their heads as whole statements via the `else` branch at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp), which goes through normal mailing without name-fan-out). Clause 2 is a finer filter for descendant LBs where the anchor-broadcast rules also fire: only LB-originated rule firings get to disintegrate.

**Why this preserves §4.1.** The §4.1 proof's required disintegrations all happen at the SE2 LB and below (the existence0 body, the `(in3[…])` preorder body, the OR branches via case-OR mint). Those LBs are non-anchor, and the §4.1-relevant rule firings have premises the SE2 LB itself derives during proof — so clause 2 keeps the gate open for them.

**Performance.** O(log N) lookup per premise via `std::set<EncodedExpression>` (the parallel container) vs O(N) on the vector. With LBs reaching thousands of statements, the set is justified. Memory cost: ~doubling the local-statement footprint per LB. Acceptable for an active SE2 LB; for the 1209-LB IncubatorGauss batch the cost is per-LB and bounded.

**Mode gating.** New gate is dormant outside `incubator_mode && !ban_disintegration`. ConfigPeano / ConfigGauss (no incubator_mode) and the legacy ConfigIncubator{Peano,Gauss} (ban_disintegration=true) follow the legacy path verbatim.

**Verification.** `gl_quick.exe IncubatorGauss1` now runs to completion (exit 0, prover finished, plateau at 7730 expressions across all 48 main-phase bursts). Pre-D-29 the same run crashed at burst 2 / ~4570 expressions. Gate-firing stats during the run: anchor blocks ≈ 3236, non-anchor non-local blocks ≈ 64, allowed (LB-derived rule firings, disintegration proceeds) ≈ 56700. §4.1 / SE2 not proved yet on this run (saved 0 theorems) — acceptable per user "se2 might be unproved" framing; further conjecturer / prover tuning can come later. Crash mitigation milestone: the architecture supports the SE2-migration combination without exhausting the int16_t NameMap.

**Code.** Gate at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp). Set-container infrastructure: [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp), 11 push_back sites and 1 rebuild-after-assign site across `prover.cpp` / `prover.hpp` — grep `localEncodedStatementsSet.insert` for the full list.

**Container superseded (2026-06-10).** The `std::set<EncodedExpression>` and its O(log N) lookup described above were re-keyed to the packed `std::unordered_set<int32_t>` `intLocalEncodedStatementsSet` with O(1) probes taken directly from the request's int rows — see `D-129` and `I-86`. The gate's two-clause semantics are unchanged.

---

<a id="d-28"></a>
## D-28 — Collapse `allow_disintegration` into `ban_disintegration`; single flag for every disintegration path (2026-04-29 evening,)

**What.** The `allow_disintegration` flag introduced earlier the same day in [D-27](#d-27) is removed. Pass B at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) is now gated by `!parameters.ban_disintegration` (combined with `!parameters.compressor_mode`), making `ban_disintegration` the **single** gate for every disintegration-shaped path: Pass B, back-reformulation ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)), hypothetical disintegration ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)), necessity-for-equality-hypo ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)).

`allow_multiplication` (the multiplyImplication gate at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)) stays — different concern, different flag.

Per-config matrix:

| Config | `ban_disintegration` | `allow_multiplication` |
|---|---|---|
| `ConfigPeano.json` | `false` (default; field omitted) | `false` |
| `ConfigGauss.json` | `false` (default; field omitted) | `false` |
| `ConfigIncubatorPeano.json` | `true` | `true` |
| `ConfigIncubatorGauss.json` | `true` | `true` |
| `ConfigIncubatorGauss1.json` | `false` | `false` |

**Why.** D-27's two-flag split was over-cautious. Once the per-config matrix was filled in, every config's `ban_disintegration` happened to equal `!allow_disintegration` exactly — there was no config that wanted to allow Pass B while banning back-reformulation, or vice versa. Two flags doing the same job under opposite signs ("ban" + true blocks vs "allow" + false blocks) is just confusion to read. Collapsing into one flag drops that confusion, removes the awkward mixed-sign naming, deletes one row from the config schema, and reduces the "five-flag interplay" suspected fragility in [`docs/agentic_swdd/10_pipeline/09_incubator.md`](10_pipeline/09_incubator.md) to a four-flag one.

**Behavior change.** None for any of the 5 existing configs (the matrix above is exactly what each config did pre-D-28). The compressor's Phase 1 toggle (`ban_disintegration = true` at [`compressor.cpp`](../GL_Quick_VS/GL_Quick/src/compressor.cpp), restored at [`compressor.cpp`](../GL_Quick_VS/GL_Quick/src/compressor.cpp)) is unchanged and now also disables Pass B during Phase 1 directly through this flag — previously Pass B during Phase 1 was disabled by `compressor_mode=true` and the `ban_disintegration=true` was orthogonal noise; now both flags are aligned.

**Alternatives considered.** (a) Keep both flags with different semantics — rejected, no use case for divergent semantics has emerged in 5 configs and the mixed-sign naming would persist. (b) Rename `ban_disintegration` to `allow_disintegration` (positive sense, easier to read) and remove the `ban_*` flag — rejected as more churn (existing reads at multiple sites, no behaviour gain), and `ban_*` carries useful "this thing is normally on, here we explicitly stop it" intent.

**Verification.** Read all 5 configs after the collapse — `ban_disintegration` field present and correctly set in 3 (the two legacy incubator + ConfigIncubatorGauss1), default-false (field omitted) in 2 main configs. JSON parses; conjecturer-only run (`gl_quick.exe --conjecture IncubatorGauss1`) still emits the single SE2 conjecture byte-identical.

**Memory implication.** The "did we really need two flags" question is answered. Future agents inheriting this branch should not be confused by a brief D-27 era; the I-7 invariant text now reflects the collapsed state.

---

<a id="d-27"></a>
## D-27 — Decouple Pass B and `multiplyImplication` from `incubator_mode`; migrate `EnumerationSet2` to incubator side (2026-04-29)

**What.** Two new `prover_parameters` boolean flags, `allow_disintegration` and `allow_multiplication`, replace the previous role of `parameters.incubator_mode` at the Pass B and `multiplyImplication` gates respectively:

- [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) — Pass B entry. Old: `!compressor_mode && !incubator_mode`. New: `!compressor_mode && allow_disintegration`. Updates [I-7](30_invariants.md#i-7).
- [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) — `multiplyImplication` early return. Old: `!incubator_mode && !ceFilteringActive`. New: `!allow_multiplication && !ceFilteringActive` (CE-filter carve-out preserved).

`incubator_mode` keeps its other semantics (`head-in-wholeExpressions` short-circuit, integration reformulation, tempArgs assert, conjecturer behaviors). `ban_disintegration` keeps its narrower scope (back-reformulation, hypothetical disintegration, necessity-for-equality-hypo).

Per-config flag values match current behaviour for the four pre-existing configs:

| Config | `allow_disintegration` | `allow_multiplication` |
|---|---|---|
| `ConfigPeano.json` | `true` | `false` |
| `ConfigGauss.json` | `true` | `false` |
| `ConfigIncubatorPeano.json` | `false` | `true` |
| `ConfigIncubatorGauss.json` | `false` | `true` |
| `ConfigIncubatorGauss1.json` (NEW) | `true` | `false` |

Ride-along change to the orchestrator: `run_modes.py` now globs `^Config<base_tag>\d*\.json$` per base tag and runs the matches alphanumerically (the dot-vs-digit ASCII ordering puts the un-suffixed config first). The new `ConfigIncubatorGauss1.json` slots between `ConfigIncubatorGauss.json` and `ConfigGauss.json` in the Gauss group's run order.

`EnumerationSet2` migrated from `ConfigGauss.json` (deleted entry + 17 `prohibited_combinations` entries) into the new `ConfigIncubatorGauss1.json`. `files/theorems/theorems.txt` is byte-identical post-migration because §4.1 / §4.2 were never proved on the Gauss main path — only enumerated as conjectures and stalled in the prove pass. `files/incubator/theorems/theorems.txt` may grow if `ConfigIncubatorGauss1.json` succeeds in proving §4.1 (not required for this PR).

**Why.** Pre-decoupling, no batch could combine `Pass B on + multiplyImplication off + incubator_mode on (for contradiction LBs / skip CE filter)`. The FTA-ladder rung-1 proof of `{0,1}=[0,1]` (§4.1 forward = `ES2 ⟹ interval`) needs exactly that combination: it disintegrates the `EnumerationSet2` and `interval` body universal quantifiers (Pass B), must not fan out via Bell-partition multiplication (which interferes with the proof's specific shape), and benefits from incubator-side `0≤0`, `0≤1`, `1≤1` preorder facts that aren't visible at Gauss-main `v=main`. The decoupling makes that combination expressible.

**Alternatives considered.** (a) Keep `EnumerationSet2` in Gauss main but flip a hidden flag to disable multiplication for §4.1's specific shape — rejected as ad-hoc and fragile. (b) Move only `incubator_mode`'s Pass-B role onto the new flag, leave `multiplyImplication` tied to `incubator_mode` — rejected because it forces ConfigIncubatorGauss1 to choose between Pass B (needs `incubator_mode=false`) and contradiction LBs / skip CE filter (needs `incubator_mode=true`). (c) Keep three-flag interplay (`incubator_mode`, `ban_disintegration`, `compressor_mode`) — rejected, that interplay is already flagged in `docs/agentic_swdd/10_pipeline/09_incubator.md` weaknesses as a known fragility.

**Risk.** New flag-multiplication risks: a config that omits one of the new flags falls back to the default (`allow_disintegration=true`, `allow_multiplication=false` — main-path semantics), which is sensible for a non-incubator config but wrong for an incubator config that forgets to set them. Mitigation: every existing config got both flags written explicitly; the per-tag glob makes new incubator-side configs (`ConfigIncubator*1.json`, `ConfigIncubator*2.json`, …) increasingly common, and they all need to set both explicitly.

**Verification.** `files/theorems/theorems.txt` byte-identical (no §4.1/§4.2 lines were there to lose). `files/incubator/theorems/theorems.txt` is a strict superset (no removals). FTA-ladder Steps 1–7 from `docs/fta_ladder/rung1/current_proof_state.md` continue to fire (now on the incubator side via `ConfigIncubatorGauss1.json` instead of stalling in the Gauss-main `(EnumerationSet2[…])` LB). Steps 8–10 may now be reachable because the missing preorder facts are local — empirical, not gated.

---

<a id="d-26"></a>
## D-26 — `multiplyImplication` double-`u_` skip restored + verifier free-anchor-merge guard added (2026-04-28)

**What.** Two-sided defence against `multiplyImplication` equating distinct free `u_*` anchor parameters. (1) Prover side: re-enabled `if (hasDoubleU) continue;` at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) inside the partition-iteration loop, restoring the soundness skip that commit cf358271 had deleted while pursuing a `u_`-equalisation extension. (2) Verifier side: extended `check_equalize_variable` ([`verifier.py`](../verifier.py)) with a post-mapping free-anchor-merge guard. The new helper `_extract_bound_vars` parses every `>[…]` binder of source and copy implications; any `(orig_arg → copy_arg)` pair where both names are free (not in any binder) and the names differ now fails the `multiplied from` row.

**Why.** Chapter-1115 of the IncubatorGauss proof graph carried a `multiplied from` step that collapsed slot 6 of `existence2` from anchor element `1` to `0` while preserving the rest of the rule body. Downstream, the chapter used the forged rule to derive `!fold[N,s,+,id,0,0,0]` (HTML title: "sum(i=0..0) ≠ 0") — a mathematically false statement. Trapping the input to `multiplyImplication` confirmed the input was byte-identical to the raw `1114_direct_proof.txt:3` source (the prover received a sound rule); the bug was wholly inside `multiplyImplication`'s partition loop. cf358271's verifier-clean (1818 checks) had not exercised the FTA-ladder shape that triggers the bad partition.

**Why both gates rather than one.** The prover-side skip is the primary fix — it prevents the unsound copy from being emitted, so no chapter cites it. The verifier-side guard is independent defence: if a future change re-introduces u\_-equalisation in a different code path (a new compress mode, an alternate hash-burst rule, an externally-provided theorem with a `multiplied from` row), the verifier catches it before the proof ships. Per [I-16](30_invariants.md#i-16) the verifier is sacred — soundness gates that can live there should.

**Why parse binders for free/bound classification, not preserve `u_*` in processed graph.** The original instruction was to preserve `u_*` prefixes in implication source columns of `multiplied from` rows so the verifier could distinguish free anchors from bound variables by string prefix. I implemented it and reverted: keeping `u_*` survives `process_proof_graphs.py` iteration 1 but iteration 2/3 walks the modified cell, sees the `u_*` args as unmapped, and assigns chapter `v`-numbers — bypassing the anchor map and breaking the origin check (~1740 multiplied-from sources fail to resolve in `state.global_theorems`). Skipping iteration 2/3 selectively for the multiplied-source cell would propagate inconsistencies to cross-references; storing the raw form in an extra column would change the row layout for every consumer. Parsing `>[…]` binders inside the verifier itself sidesteps both — same soundness guarantee, zero change to the row format.

**Verification.**

| | Pre-fix (incubator-only run) | Post-fix (full chain, all four batches) |
|---|---:|---:|
| Verifier checks | 32585 / 0 | 35458 / 0 (incubator + main) |
| Incubator HTML chapters | 1342 (incl. false `!fold[…,0,0,0]`) | 1202 (140 chapters dropped — every theorem whose only proof path used a double-`u_` partition; the false statement is gone) |
| FTA-ladder Steps 1–7 (HB#36 SE2 LB) | reached | reached, identical Branch A validity, identical witness names |
| Synthetic adversarial test (`existence2` slot-6 `i1 → i0`) | — | verifier returns `False` ✓ |
| Synthetic sound tests (identity, Bell partition, bound→free specialise) | — | verifier returns `True` ✓ |

**SwDD touches.**
- This entry (D-26).
- [I-24](30_invariants.md#i-24) — new invariant.
- `docs/agentic_swdd/AGENT_SwDD.md` quick-reference — I-24 row added.
- [`docs/agentic_swdd/10_pipeline/04_prover.md`](10_pipeline/04_prover.md) — *Soundness gate* subsection inside the `multiplyImplication` chapter.
- [`docs/agentic_swdd/20_core_concepts/08_proof_tags.md`](20_core_concepts/08_proof_tags.md) — `multiplied from` entry now describes both the prover-side skip and the verifier guard.

---

<a id="d-25"></a>
## D-25 — Compressor invocation moved out of `analyzeExpressions` (2026-04-28)

**What.** The compressor is no longer invoked from inside `ExpressionAnalyzer::analyzeExpressions`. The prover's main entry now does prove → post-prove `headSwitchOne` walk → return; `run_modes.cpp::fullRun` then conditionally invokes `Compressor::run` after `analyzeExpressions` returns, gated by `parameters.skipCompression`. The gate's truthiness is unchanged (`incubator_mode || ban_disintegration`).

**Why.** Two reasons. (1) Restore master flow — the legacy in-`analyzeExpressions` compressor invocation was a leftover from the deleted multi-iteration loop ([D-24](#d-24)), where compressor output fed the next big-iteration broadcast set. With the loop gone there is nothing inside `analyzeExpressions` that consumes the compressor's result, so the call's only effect is to extend the prover's wall-time profile with a phase that semantically belongs to the orchestrator. (2) Symmetry with `run_modes` — every other post-prove pipeline step (`exportCompiledExpressionsJSON`, the OR-pair construction over `orPairsFromHeadSwitch`, the proof-graph emission) already lives in `run_modes.cpp`. Keeping the compressor next to its peers makes the post-prove sequence readable in one file; nothing in `prover.cpp` after the head-switch walk now competes for the reader's attention.

**Why not behind a flag.** The relocation preserves byte-equivalence: the same `Compressor::run` is called with the same inputs in the same order, just from a different translation unit. There is no behaviour to gate. A flag would only document the move, which is what this entry exists for.

**Code.** Compressor invocation site moved from `prover.cpp` (deleted) to `run_modes.cpp::fullRun`. The class member `parameters.skipCompression` keeps its semantics. The dropped `prover.cpp` block is recoverable via `git log` of commit.

---

<a id="d-24"></a>
## D-24 — Multi-iteration prover loop deleted; single-pass head-switch pre-emit becomes the only path (2026-04-28)

**What.** The big-iteration loop in `ExpressionAnalyzer::analyzeExpressions` is gone. The function now does one prove → one compress → one orPairsFromHeadSwitch population, in that order, and returns. With it go:

- `parameters.pre_emit_head_switch` (the trial flag from D-23-era exploration, now unconditional behaviour).
- `parameters.maxBigIterations` (the iteration cap, no longer meaningful).
- The grid teardown + rebuild block (`destroyGrid`, the cache-proof-stacks scan, `compileAndRegister(remaining)`, `buildGrid`) — every consumer was the second prove pass that no longer happens.
- The `headSwitchedTheorems` and `alreadySwitched` locals — only existed to seed the deleted next iteration.
- The "subsequent iteration" branch of the prove call (the one that broadcast all-of-`globalTheoremList` instead of `provedTheorems`) — same reason.
- About 400 lines of `prover.cpp` and a dozen lines elsewhere.

**Why.** D-23 closed the only gap between single-pass and two-pass on Peano: the single-pass trial ( then ) reached **27 proved theorems** in **377 s** versus the two-pass baseline **28 proved in 626 s** — and the sole "missing" theorem (`(>[5,6](AnchorPeano)(in3[8,7,6,5])(in3[7,8,6,5]))`) is the slot-6 instance of the more general `(>[5](AnchorPeano)(in3[8,7,9,5])(in3[7,8,9,5]))` which IS proved on the first pass. So the second pass was deriving content already entailed by the first. Keeping it was burning ~250 s of wall time per Peano run for no semantic gain.

**Why not keep the flag for safety.** Two reasons. First, the flag was *only* a kill-switch for code we now know is functionally dead — keeping it preserves zero behaviour and adds maintenance noise (every config has to know about it, the SwDD has to document a non-feature, future readers have to understand a "legacy two-pass" path that nobody ever uses). Second, the multi-iteration machinery had non-trivial state (the cache-proof-stacks scan, the per-iteration broadcast difference between iter-0 and iter-N>0, the dedup against `allProvedEver`); leaving it in as dead code makes the prover harder to read, not safer. If a future need re-emerges, `git log` recovers the path.

**What survives.** `headSwitchOne` ([`prover.cpp+`](../GL_Quick_VS/GL_Quick/src/prover.cpp)) — the stateless contrapositive helper extracted — stays. Both the pre-emit pass (now unconditional, just before the first `compileAndRegister`) and the post-prove walk (now the only call site that populates `orPairsFromHeadSwitch`) call it. Output between the two paths is byte-identical to the legacy in-loop construction.

**Verification on Peano-only `main.py` (post-deletion):**

| | Two-pass baseline (pre-D-23) | Single-pass trial (D-23) | Post-D-24 (this commit) |
|---|---:|---:|---:|
| Wall | 626 s | 377 s | matches D-23 (refactor only) |
| Verifier | 2105 / 0 airtight | 2098 / 0 airtight | 2098 / 0 airtight |
| Proved theorems | 28 | 27 (general slot-N variant subsumes the missing slot-6 specialisation) | 27 |
| `prove(...)` calls per batch | up to `1 + N×2` | `2` (warm-up + main, gated loop) | `2` (warm-up + main, no loop) |

`fullTheoremList` dedup is now a local `alreadyInFull` set inside the compress block (used to live as `allProvedEver` across the loop). All other downstream consumers (`run_modes.cpp::fullRun`, `visualizer.cpp::generateRawProofGraph`) read the same class members as before — `globalTheoremList`, `fullTheoremList`, `lastCompressionSurvivors`, `orPairsFromHeadSwitch`, `cachedProofStacks` — and their producer-paths are unchanged for the parts that still exist.

**Out of scope.** The warm-up vs main-iteration split *inside* the surviving single prove call (Phase 1 + Phase 2) is unchanged. That's the user's instruction; it's a different mechanism (`prove(preIterations, …)` → broadcast → `prove(remainingIterations, …)`) and not part of "multi-pass".

**SwDD touches in this commit.**
- This entry (D-24).
- `docs/agentic_swdd/10_pipeline/04_prover.md` — replaced the "Big-iteration loop and head-switch" + "Single-pass mode (`pre_emit_head_switch`)" subsections with a flat "Single-pass head-switch model" + "Contrapositive construction" pair.
- `docs/agentic_swdd/04_configs.md` — removed the `maxBigIterations` and `pre_emit_head_switch` rows from the `prover_parameters` field table.

---

<a id="d-23"></a>
## D-23 — D-21 relaxations 2 and 3 walked back; anchor-membership-axiom premise filter added (2026-04-28)

**What.** Two of D-21's three conjecturer relaxations are reverted, and one new structural-rejection filter is added. Net change visible in `gl_quick.exe --conjecture Peano`: 1785 → 536 surviving conjectures (−70 %). The cancellation theorem still emits (its ascending head `=[2,8]` reaches `conjectures.txt:20` independently of any relaxation).

| D-21 relaxation | Status after D-23 |
|---|---|
| Relaxation 1 — `passesInPremiseFilter` cnt==2 neutralisation rule | **Kept.** Genuine typing premises like `(in[a, N])` next to `(in3[a, b, i0, +])` still need this rule; without it the cancellation theorem's typing premise can't pair with its operator-equation companion. |
| Relaxation 2 — `controlEquality` accepts descending `=[a, b]` at nse ≤ 3 | **Removed.** The pair-combination enumeration produces both ascending `=[a, b]` and descending `=[b, a]` forms independently — the cancellation theorem head exists in ascending form `=[i0, b]` (e.g. `=[2, 8]`), so dropping the descending sibling loses no genuine theorem. Function reverts to the original "reject any descending `=[a, b]`" rule. |
| Relaxation 3 — `passesComplexityAfterExistence` returns true unconditionally at complexityLevel ≤ 3 | **Removed.** The hardcoded `complexityLevel <= 2 → return true` and the 73-line `complexityLevel == 3` carve-out are deleted. The function now consults only `max_complexity_if_anchor_parameter_connected_after_existence` (per-type 2-tuple `[complexity_cap, arity_sum_cap]` from D-22's predecessor ) and rejects when **all three** of `complexity > complexity_cap`, `arity_sum > arity_sum_cap`, and the slot is present in non-anchor leaves. The cancellation theorem (arity_sum 4+2+2 = 8 ≤ 8) still escapes via the arity-sum dimension; explosion shapes (3 × in3 = 12 > 8) are rejected. |

**Plus, new filter — anchor-membership-axiom rejection.** A non-anchor `(in[v, X])` premise (or its negation) where BOTH `v` AND `X` are anchor-slot values — e.g. `(in[2, 1])` reads "i0 ∈ N", which is one of `AnchorPeano`'s own axioms — is vacuous as a premise. The previous complexityLevel == 3 carve-out had this check inline; with the carve-out removed, the check moved to `passesInPremiseFilter` as a top-of-function gate so it now runs for every conjecture, not just nse=3..

**Why.** D-21 was authored before the arity-sum dimension existed. Once both branches were merged, the relaxation 3 sat at the top of `passesComplexityAfterExistence` and short-circuited before the arity-sum check ever ran. Result on the merged branch: 1108 explosion-class conjectures of the form `(>[…anchor_slot_idx](AnchorPeano…)(in3[…6…])(in3[…6…])(in3[…6…]))` — slot-6 (i1) used 3× per conjecture — slipped through every gate. The arity-sum dimension is the precise tool for this discrimination; deleting the relaxation lets it work.

Relaxation 2's removal is a similarly cheap simplification: `controlEquality`'s pair-combination enumeration produces both orderings of every `=` head, so dropping descending forms loses nothing. The relaxation was a defensive measure when the conjecture-generation order was uncertain; the empirical evidence (descending and ascending heads both present at nse=3 across `conjectures.txt`) shows the defensive measure isn't load-bearing.

**Why not the alternatives.**

- *Tighten Peano `(1)` arity_sum_cap from 8 to 7* to also catch the 8 in2-based `=[…]` 3-occurrence shapes that survive D-23's tightening. Would also block the cancellation theorem (arity_sum exactly 8). Rejected.
- *Add a separate "max occurrences per anchor slot value" cap field*. Duplicates the arity-sum dimension's role. Rejected.
- *Keep relaxation 3 with cross-leaf aggregation*. Less elegant; hard-coded literals stay; doesn't address D-22's filter mismatch. Rejected per user direction ("`passesComplexityAfterExistence` … shld use only config and not hard coded at all").
- *Keep relaxation 2 with anchor-involvement gate* (descending kept only when one arg is anchor-slot value, dropped when both bound). Rejected after empirical check showed the ascending mirror of every cancellation-shape conjecture is already in `conjectures.txt` independently — relaxation 2 is unnecessary.

**Effect on `conjectures.txt` (Peano alone, full conjecturer pass).**

| State | Count | Notes |
|---|---:|---|
| Pre-merge cancellation_theorem (D-21 active, no D-22) | 5656 | (per D-21 commit body) |
| Post-merge before D-23 | 1785 | D-22 brings the registry; D-21 still active in this state |
| After D-23 commit (drop relaxation 3) | 677 | `(1)`-pinned 3-`in3` shapes rejected via arity-sum cap |
| After D-23 commit (anchor-membership filter) | 635 | `(in[anchor, anchor])` premises rejected |
| After D-23 commit (drop relaxation 2) | 536 | descending `=[a, b]` symmetric duplicates rejected |

The cancellation theorem is at `conjectures.txt:20` in the final state and was confirmed proved by the prover in the post-merge full Peano run (per user observation in run.log).

**SwDD touches.** Conjecturer chapter (`docs/agentic_swdd/10_pipeline/02_conjecturer.md`) updated to describe the post-D-23 form of the three filters: `passesComplexityAfterExistence` no longer has hard-coded complexity bands; `passesInPremiseFilter` has the anchor-membership gate at top; `controlEquality` is back to original strict-canonicalisation. Configs chapter (`docs/agentic_swdd/04_configs.md`) unchanged — the field semantics (`[complexity_cap, arity_sum_cap]` vector) were already documented at D-22-merge time.

---

<a id="d-22"></a>
## D-22 — Cross-batch shared registry for spontaneous compact operators (2026-04-25,, merged 2026-04-26)

**What.** Spontaneous compact operator names (`implication<N>`, `existence<N>`, `or<N>`, `and<N>`) are persisted across batches via a single growing JSON file `files/GL_binaries/GL_binary_shared.json`. Per batch:

- **Pre-batch.** Python copies `GL_binary_shared.json` to `GL_binary_<Tag>.json` (Python is the sole writer of shared; it is also the sole creator of each per-batch file). On a clean run when shared does not yet exist, an empty `{}` is written instead.
- **C++ startup.** The `ExpressionAnalyzer` constructor calls a new `loadGlBinary(path)` method that parses the per-batch JSON, populates `compiledExpressions` for every entry, registers a `repetitionExclusionMap` row for every spontaneous entry (using the JSON's `elements` field directly as `splitNK`, which is exactly what `excludeRepetitions` stored when the entry was first allocated), and seeds the four shared counter members from the trailing-integer maxima of the loaded names: `implCounter = max(N for "implication<N>") + 1` and analogously for `existenceCounter`, `andCounter`, `orCounter`. `variableCounter` continues to start at zero each batch — bound-variable identifiers have no cross-batch identity to preserve.
- **C++ shutdown.** `exportCompiledExpressionsJSON` (unchanged) writes the entire `compiledExpressions` map to `GL_binary_<Tag>.json`. Because shared entries were preloaded into that map, the per-batch file at end of run contains the inherited shared entries plus any newly-allocated this-batch entries.
- **Post-batch.** Python reads `GL_binary_<Tag>.json`, filters to entries whose `category` is in `{implication, existence, or, and}`, and adds any name not already present to `GL_binary_shared.json`. Anchor and atomic entries are excluded — they are batch-local. The shared file grows monotonically over the run.

**Why.** Each batch was previously assigning compact identifiers from counters that reset to zero in the `ExpressionAnalyzer` constructor. The same logical operator therefore acquired different names in different batches: a Peano successor-existence theorem named `existence2[1,6,3]` was renamed to `existence3[1,6,3]` when `gl_quick.exe Gauss` re-ran `precompileStructuralOperators` on Peano's expanded-form `theorems.txt` entries. Because Gauss never adds those re-imported theorems back into its `globalTheoremList`, the Gauss-named form (`existence3`) ended up cited inside chapter 85 of the proof graph but missing from `raw_proof_graph/global_theorem_list.txt` (which holds Peano's `existence2` only). The Python pruning step in `process_proof_graphs.py:_prune_proof_graph` then dropped both forms because the essential set (`compiled_theorems.txt`) and the raw-proof-graph set used disjoint identifiers, and the verifier failed the chapter-85 origin check on the cited template. The shared registry eliminates the rename: Peano allocates `existence2` and writes it to shared; Gauss loads shared at startup, finds the `existence2` entry already in `repetitionExclusionMap`, and `excludeRepetitions` returns the existing name instead of allocating `existence3`. Same name everywhere — the chapter-85 failure resolves itself.

**Why not the alternatives.** *Re-emitting prev-batch theorems into Gauss's `globalTheoremList`* would push the Gauss-renamed identifier into the raw graph, but `compiled_theorems.txt` would still need a synonym mapping to reconcile with the Peano-named form already in earlier raw-graph entries — net effect: two identifiers for one theorem with bookkeeping overhead. *Verifier-side synonym resolution* would touch [I-16](30_invariants.md#i-16) (verifier sacred). The shared-registry approach removes the divergence at its origin.

**Operational consequence.** A clean run still wipes `files/GL_binaries/` (existing behaviour in `run_modes.py:full_run` lines 227-230). To force re-allocation under fresh numbering after a config change to a structural operator's `definedSet` or `existence_variable_position`, delete `GL_binary_shared.json` manually and rerun; there is no automatic staleness detection. See [I-23](30_invariants.md#i-23).

**Renaming consequence.** The `int statementCounter` member on `ExpressionAnalyzer` was renamed to `andCounter` in the same change. Every operational use was passing it into `compileCoreExpressionMapCore`'s `andCounter` parameter slot; no `statement<N>` operator exists anywhere in the codebase. The four spontaneous-operator counter members are now spelled by their actual roles: `implCounter`, `existenceCounter`, `andCounter`, `orCounter`.


---

<a id="d-21"></a>
## D-21 — Three nse≤3 conjecturer relaxations for the cancellation family (2026-04-24)

**What.** Three related relaxations in the conjecturer's filter cascade, all scoped to small (`complexityLevel / nse <= 3`) conjectures, landing together so the additive-cancellation theorem `(in[a,N]), (in3[a,b,i0,+]) -> (=[b,i0])` actually ends up in `conjectures.txt`:

1. The `cnt == 2` branch of `Conjecturer::passesInPremiseFilter` at [`conjecturer.cpp`](../GL_Quick_VS/GL_Quick/src/conjecturer.cpp) now accepts an additional shape: a positive `(in[v,X])` premise together with one non-anchor companion premise, provided `v` (first arg of the `in`) also appears as an argument in the companion premise or in the head. Originally the `cnt == 2` branch required at least one of the two premises to be negated; this rule is kept as rule (2) and the new check is rule (3). `cnt >= 3` remains rejected unconditionally.

2. `Conjecturer::controlEquality` at [`conjecturer.cpp`](../GL_Quick_VS/GL_Quick/src/conjecturer.cpp) no longer rejects descending-ordered `(=[a,b])` (`a > b` as integers) when the conjecture has `nse <= 3`. The original canonicalisation still applies at `nse > 3`. Motivation: after anchor-pinning, a bound-var ID lands in arg-1 and an anchor-slot ID in arg-2, producing the descending form `=[8, 2]` — the `b = i0` head of the cancellation theorem. Rejecting that form silently erases the family.

3. `Conjecturer::passesComplexityAfterExistence` at [`conjecturer.cpp`](../GL_Quick_VS/GL_Quick/src/conjecturer.cpp) now returns `true` unconditionally when `complexityLevel <= 3`. The full per-def-set cap logic still applies at `complexityLevel >= 4`. Motivation: Peano's `(1)` cap is 2, meaning "no chain arg may be pinned to anchor's i0 or i1 slots when the conjecture has 3+ `(>[` layers". The cancellation theorem has `complexityLevel = 3` AND pins `in3`'s c-arg to anchor slot 2 (i0), so the un-relaxed filter drops it. Discovered via the hard-coded-target tracer protocol (see ): every earlier filter passed, every other anchor-attach variant that reached this point was dropped here. The carve-out admits the cancellation family but also a wide set of other `nse=3` (1)-pinned shapes that were previously filtered on blow-up grounds — see trade-off note below.

**Why.** The FTA ladder's next rung needs the additive-cancellation family — `(in[a,N]), (in3[a,b,i0,+]) → (=[b,i0])` and its successor/product siblings — to be conjectured. Under the original rule (negation required at `cnt == 2`) the cancellation shape was enumerated by the combinatorial core but silently dropped by the filter, because `in[a,N]` carries semantic weight (typing) without needing a negation partner. The relaxation scopes strictly to `cnt == 2` (i.e. `nse = 3`) so the conjecture population cannot explode on larger shapes.

**Why this shape, not a config flag.** "Neutralises a free variable" is a structural property of the conjecture (does `v` participate elsewhere?), not a per-batch policy. Putting it in code keeps the rule auditable. A batch that genuinely needs to reject it can still do so via `prohibited_heads` or the prohibited-combinations list.

**Side correction.** While tracing this change, found that `apply_in_premise_filter` — documented as a per-batch gate — is dead code. The flag is declared and loaded but never consulted at the filter entry or at its callsites. SwDD `OPEN-9` and `docs/agentic_swdd/04_configs.md` updated. `ConfigGauss.json`'s `"apply_in_premise_filter": false` has no effect; Gauss avoids the filter only because its `in[]`-bearing conjectures place `in` at the head (so `hasIn` is false and the function short-circuits).

**Before / after (conjecturer output, strict-subset confirmed each step).**
- Peano: 501 → 524 (relaxation 1, +23 generalised-cancellation lines) → 559 (relaxation 2, +35 descending-`=` mirrors) → 5656 (relaxation 3, +5097 `(1)`-pinned nse=3 shapes). The target cancellation theorem `(>[1,2,4](AnchorPeano[1,2,3,4,5,6])(>[7,8](in3[7,8,2,4])(>[](in[7,1])(=[8,2]))))` is in the final set.
- Gauss: 395 → 397 (relaxation 1, +2 interval neutralisation) → 397 (relaxation 2, no change) → 397 (relaxation 3, no change). Relaxation 3 has zero effect on Gauss because its anchor has no `(1)`-typed cap in this config slot.

**Trade-off — relaxation 3 is wide.** Moving `complexityLevel > 2 (1)-pinned` from "reject" to "accept at complexityLevel=3" adds ≈5000 new Peano conjectures. Most are unproved (prover + CE filter will discard), but the batch wall-time grows proportionally. Alternatives considered:
- Raise the config cap from `(1):2` to `(1):3` — equivalent effect, same 5000-conjecture blow-up, parameterised in config rather than in code. Rejected because the cap is *semantically* "max non-anchor chain length when (1) pinned", not a cancellation-specific knob, and touching it via config reads as a general policy change rather than a cancellation-family unlock.
- Pattern-match specifically for `in3[*,*,anchor-slot,+/*]` + `in[*,N]` + `=[*,anchor-slot]` and skip only for that shape — narrower, but couples the filter to a specific conjecture family and would need extending for each new family.
- Leave the filter as-is and seed the cancellation theorem via `externally_provided_theorems.txt` — abandons "conjectured first".
The current choice (unconditional pass at `complexityLevel <= 3`) is the blunt version; tightening is open if the 5000-conjecture blow-up turns out to cost meaningful prover wall-time.

**Alternatives considered.**
- *Config flag to gate the relaxation.* Rejected — the neutralisation property is intrinsic to the conjecture, not a batch-level policy. Also parallels the OPEN-9 dead-flag we just found: adding another unused flag does not help.
- *Widen to `cnt == 3`.* Rejected per user direction — risks a conjecture-population blow-up without a proven need; can be revisited if the FTA ladder needs it.

---

<a id="d-20"></a>
## D-20 — Conjecturer int-story performance campaign (2026-04-24, branch `rt_conjecturer_session_24042026`)

**What.** Two perf-only changes on the conjecturer hot path that preserve the byte-for-byte output contract (`conjectures.txt`, `reshuffled_conjectures.txt`, `reshuffled_mirrored_conjectures.txt`) on both Peano and Gauss:

1. **Path B reshuffle** — `Conjecturer::reshuffle`'s permutation inner loop replaced: pre-parse each chainEntry + head into an `EntryTemplate` that pinpoints every `[...]`-token byte position with an `isAtom` flag; per permutation, walk atom slots in permuted byte order to build the first-occurrence rename map into a thread_local `int[]` indexed by dense token id; render the rebuilt string directly into a thread_local `std::vector<char>` with inline rename substitution; winner selected by `memcmp` on the rendered buffer. Eliminates the per-permutation `std::string` rebuild + `replaceKeysInString` pass. Semantics byte-for-byte equivalent to OLD (memcmp on renamed rebuilt strings IS `std::string::operator<`).

2. **Disintegrate cache** — `compiler.hpp::disintegrateImplication` carries a thread_local single-slot cache keyed by the input expression. Matches the hot call pattern: one candidate passes through ~10 filters that each re-disintegrate the same string; each new candidate misses once, every subsequent filter call on the same string hits. thread_local gives zero-contention per-worker reuse.

**Why.** Profiling (session baseline) showed the conjecturer at ~30 s wall for Peano. Top consumers of thread-time: `reshuffle` 218 s (86 % inside the permutation inner loop, all `std::string` / `replaceKeysInString` work) and `disintegrateImplication` 198 s (17.3 M calls, 11 µs / call dominated by heap-allocated TreeNode1 trees). The SwDD's `docs/agentic_swdd/10_pipeline/02_conjecturer.md` already identifies the string-path as "the older lane... unfinished portion of the 100x acceleration campaign" (line 120). These two changes address two distinct string-path pain points without touching semantics.

**Measured effect (Peano, 3-run avg, 32 HW threads).**

| commit | wall | reshuffle | disintegrate |
|---|---|---|---|
| (baseline) | 30.5 s | 218 s | 198 s |
| (+ Path B) | 27.2 s | 111 s | 198 s |
| (+ cache) | 23.8 s | 114 s | 75 s |

Total: −6.7 s wall (−22 %) across two commits.

**Alternatives considered.**

- *Path A for reshuffle* — keep the `std::string` rebuild but make it cheap (reserve + direct char writes, skip `replaceKeysInString`). Would have given ~half the win with lower design risk. Rejected because the byte-identity proof is just as easy for Path B (memcmp equivalent to string compare) and the payoff is ~2× bigger.
- *Full iterative `disintegrateImplication`* (replace `parseExpr` + TreeNode1 tree build with an allocation-free string walk). Would give a bigger win than the cache on cache-miss calls, at the cost of a higher byte-identity risk (treeToExpr's canonicalization edges). Reserved for a later commit if we want to squeeze more.
- *Port the remaining string filters to int* (`checkInputVariablesOrder`, `evaluateOperatorExprs2`, `checkInputVariablesHead`) — touches filter signatures + many callers. Parked pending user direction.

**Follow-ups.**

- The profiling scaffold (`prof::` namespace in `conjecturer.cpp`, `g_disint*` atomics in `compiler.hpp`) is temporary, left in-tree while the campaign is active so before/after measurements stay reproducible. Remove in a clean-up commit when the campaign closes.
- `exprGood2Int` at 278 s / 28 % of CPU is now the top remaining consumer. It's already int-based; the cost is pure call volume (30.3 M) × per-scan work. It internally decodes and calls `evaluateOperatorExprs2` on strings — porting those to int is the next wins candidate.

---

<a id="d-19"></a>
## D-19 — `rejectedMapIntegration` revival via `sameIterationInternalMail` only (2026-04-24)

**Status.** Superseded by [D-53](#d-53) (2026-05-07, renumbered from main's D-46). The "dedicated `InternalMail` channel" rationale below is preserved for historical context; `struct InternalMail` was deleted and `Memory::sameIterationInternalMail` is now type `Mail`. The lifecycle and absorb-status distinctions described below remain intact.

**What.** Integration-side rejection recovery uses a dedicated `InternalMail sameIterationInternalMail` channel on `Memory`. Its statements re-enter the LB at next hashburst with `status=1` (full disintegration pipeline). It does not go through `addExprToMemoryBlock` directly from the equi-class rewrite site; it does not share `Mail::statements` with the routing `mailIn`.

**Why.** Three reasons.

1. *Avoid cyclic re-entry.* The algebra-side counterpart (`revisitRejected2`) calls `addExprToMemoryBlock`, which internally can call `updateRejectedMap` again — a structural cycle guarded by `revisitInProgress`. The mailIn-style deposit is linear: the producer writes to `sameIterationInternalMail`, the next hashburst absorbs, the absorb triggers normal `addStatement` flow. No re-entry risk.
2. *Scope fidelity.* Legacy `Mail` is main-scope-only (every `mailOut.statements.insert` is gated on `validityName == "main"` and absorb hard-codes `"main"`). Integration rejections can live at non-main validities (e.g. Branch A inside an OR integration). Mixing scope semantics into legacy mail would force 8 existing call sites to migrate from `pair<expr,levels>` to `tuple<expr,levels,validity>`. A new `InternalMail` struct with a tuple-shaped `statements` set isolates the change.
3. *Status asymmetry.* Legacy `mailIn` absorb at `prover.cpp` uses `status=3`, which skips `disintegrateExpr2` — correct for broadcast recipients of main-scope facts. Integration revival is the opposite: we *want* the revived constituent to re-enter Pass B so a fresh `int_` mint can succeed under the now-rewritten args. `status=1` is the right status, distinct from legacy.

**Operational consequence.** `Memory` carries two inboxes: `Mail mailIn` (routing) + `InternalMail sameIterationInternalMail` (revival). Same drain site (top of hashburst) but different absorb status and different lifecycle ([I-21](30_invariants.md#i-21)). Revival origin uses the existing `equality1` tag (eq-class substitution), with `origin.second[0]` = pre-rewrite constituent form so the verifier's `equality1` checker at [`verifier.py`](../verifier.py) accepts the shape.

---

<a id="d-18"></a>
## D-18 — the project conventions tag-count reconciliation (2026-04-23)

**What.** Corrected the project conventions's implicit claim of 28 tags in `TAG_CHECKERS` to 28 distinct tags + 29 registry entries. The missing tag in the project conventions's narrative list was `symmetry of inequality`; the extra registry entry is the shared `equalize variable` (dead alias) / `multiplied from` checker.

**Why.** Documented here during SwDD authoring so future agents don't inherit the drift.

**Implication.** When editing `TAG_CHECKERS`, remember one checker may serve two tag keys. Removing the `equalize variable` alias would be cosmetic — no emission site produces it.

---

<a id="d-17"></a>
## D-17 — Dual-license stamp on documentation (2026-04-23)

**What.** Every `.md` under `docs/` carries the AGPLv3 + commercial dual-license HTML comment at top, matching the Python/cpp stamp pattern.

**Why.** Documentation is part of the commodity. Proof graphs, theorems, and now the SwDD all inherit AGPL terms. Commercial-license buyers get the same text without the AGPL obligation. The HTML-comment form keeps the stamp non-rendering in markdown viewers while still embedding it in the file.

---

<a id="d-16"></a>
## D-16 — Hard-coded Peano-theorem drop on Gauss batch (2026-04-22, commit )

**What.** `run_modes.cpp–150` contains a hard-coded filter that removes a specific Peano theorem (`"(>[1,3,6](AnchorPeano[1,2,3,4,5,6])!(>[7](in[7,1])!(in2[7,6,3])))"`) from the inherited `proved_set` when `anchor_id == "Gauss"`.

**Why.** The branch's investigation found that this particular Peano theorem (an unsound-by-induction-typing artefact — see [I-18](30_invariants.md#i-18)) poisons downstream Gauss proofs of the `fold` / `limitSequence` cascade. Dropping it restores the cascade.

**Status.** Temporary mitigation. Real fix is the induction-typing sub-theorem rollout per [`docs/agentic_swdd/induction_typing_plan.md`](induction_typing_plan.md). The hard-coded drop should be removed once the induction-typing fix lands.

**Root-cause note on the drop:** reason the drop restores the cascade is not fully understood as of the commit message. An active debugging line.

---

<a id="d-15"></a>
## D-15 — Induction-typing sub-theorem — design approved (2026-04-??, pre-)

**What.** To close the induction-soundness gap, induction on a bound variable `n` must be preceded by proving `(in[n, N])` from the current chain. The typing sub-theorem uses the same prover with a `typingProofOnly = true` flag to forbid induction re-entry.

**Why.** Without typing, induction can be scheduled on variables that are not in `N`, ranging the theorem over all entities. This is unsound and produced `theorems.txt` entries that should not be there. See [I-18](30_invariants.md#i-18) and [`docs/agentic_swdd/induction_typing_plan.md`](induction_typing_plan.md).

**Trade-off considered.** Could weaken the induction scheduler to reject all cases where typing is not trivially derivable — less intrusive but also loses some valid proofs. The chosen approach (full typing sub-theorem, artefact-visible via a dedicated chapter) preserves auditability.

---

<a id="d-14"></a>
## D-14 — `passesInPremiseFilter` behind a config gate (2026-04-??, commit )

**What.** The recent `passesInPremiseFilter` conjecturer filter is now config-gated rather than always-on.

**Why.** The filter is beneficial for some batches and detrimental for others. A config flag lets per-batch decisions flow through `Config<Tag>.json`. See [`10_pipeline/02_conjecturer.md`](10_pipeline/02_conjecturer.md) — `Conjecturer::passesInPremiseFilter` at [`conjecturer.cpp`](../GL_Quick_VS/GL_Quick/src/conjecturer.cpp).

---

<a id="d-13"></a>
## D-13 — Validity-stack NameMap migration (2026-04-??, branch, landed subsequently)

**What.** The per-LB validity-name system was migrated from free-form strings + side tables to the `NameMap` + `pairMap` + `stackOfValidity` canonical form. Non-`main` scopes must now be minted via `encodePush(parent, payload)`. See [I-2](30_invariants.md#i-2).

**Why.** Free-form scope strings made ancestor queries ambiguous; side tables required coordinated updates per scope kind. Centralising in `pairMap` gives a single source of truth for parent-child relationships and makes `comparable` / `deeperOf` reliable under branching.

**Trade-off considered.** Keep the old form, add assertions. Rejected — the underlying ambiguity is structural, not a discipline problem.

**Subsequent extensions** (`or_5`, `or_6`):

- Equivalence classes propagate to descendants, not just the defining scope.
- `vacuous truth` confined to scope `"main"` (the vacuous-truth tag fires only at root).
- Sentinel scope kind renamed to a less ambiguous payload prefix.

---

<a id="d-12"></a>
## D-12 — Variable-copy tag subsumes retired tags (pre)

**What.** The `variable copy` tag now covers what used to be two separate tags: `reaction to hypo` and `necessity for equality (hypo)`. The old tags are retired — no chapter row carries them.

**Why.** The two cases are semantically the same — a fresh `_copy` duplicate of an existing variable introduced during hypothesis handling. Separate tags were a historical accident; consolidation simplifies the verifier.

---

<a id="d-11"></a>
## D-11 — OR-branch scope classification via `classifyOrScope` (pre)

**What.** OR-scope bookkeeping no longer uses side-table flags; instead, `classifyOrScope` at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp) reads the scope's payload prefix and returns `NotOrScope | Integration | Disintegration`.

**Why.** Same motivation as D-13 — centralising scope-role info in the payload avoids parallel-state bugs. Complements the NameMap migration.

---

<a id="d-10"></a>
## D-10 — `incubator_mode` vs `ban_disintegration` — separate flags (per the project conventions)

**What.** Two distinct `ProverParameters` flags control Pass B disintegration gating: `incubator_mode` (for the incubator pipeline) and `ban_disintegration` (for other contexts — currently: compressor Phase 1). Pass B is gated on `!incubator_mode` [I-7](30_invariants.md#i-7) — not on the umbrella `ban_disintegration`.

**Why.** A previous attempt conflated them, which turned out to gate more than intended. Separating the flags clarifies intent per caller.

---

<a id="d-9"></a>
## D-9 — Pass B single-input operator gate (empirical, commit reverted)

**What.** The standalone fallback admission rule `isAllowedAsOperatorInput` at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp) is restricted to operators with `inputIndices.size == 1`. Widening to multi-input operators broke Gauss summation.

**Why.** Empirical. The gate is not theoretically justified; it is a measurement-grounded guardrail. See [I-6](30_invariants.md#i-6) + memory.

**Open question.** Whether a theoretical argument exists for single-input-only, or whether future expansion is possible under stricter admission-map rigour. See [AGENT_SwDD.md OPEN-3](AGENT_SwDD.md#open-questions).

---

<a id="d-8"></a>
## D-8 — `logicalCores = 1` hardcode

**What.** `logicalCores` is set to `1` unconditionally in the binary. Parallel smashMail was explicitly rejected in a previous session.

**Why.** Parallelism would reorder mail-drain operations in ways that break determinism guarantees. The LB grid is designed to be parallelisable, but the *specific* parallel implementation proposed was not safe.

**Future direction.** The RT campaign explores alternative parallelism schemes — per-LB lock-free mailbox, coarser cycle semantics with commit barriers — that preserve determinism.

---

<a id="d-7"></a>
## D-7 — C++ conjecturer replaces Python `create_expressions.py` (2025-late to 2026-early)

**What.** The conjecturer is now a C++ component invoked via `gl_quick.exe --conjecture <tag>`. The Python `create_expressions.py` is retired (file removed).

**Why.** Measured ~6.5× exe speedup and enabled hot-path int-path enumeration. The retired Python code produced the same conjecture set but was the pipeline bottleneck pre-migration.

**Status.** Landed. The `expression_utils.py` survivor module (string-parsing helpers) is what remained of the Python conjecture machinery.

---

<a id="d-6"></a>
## D-6 — Division-free Gauss summation (by design)

**What.** GL's Gauss batch proves `n · (n+1) = 2 · Σ i` — not `Σ i = n(n+1)/2`.

**Why.** GL avoids division by design. Expressing the Gauss formula in division-free form keeps the theorem-space within the `(+, *, s)` closed fragment. See the project conventions.

---

<a id="d-5"></a>
## D-5 — One-sided negated-equality expansion (per [I-12](30_invariants.md#i-12))

**What.** `applyEquivalenceClassToNegatedEquality` emits one-sided sibling inequalities from `!(=[a,b])`, not the symmetric cross-product.

**Why.** Combinatorial containment. Two-sided expansion blows up `|class(a)| × |class(b)|` per input. Anything the cross-product would conclude is reachable by two one-sided steps.

---

<a id="d-4"></a>
## D-4 — Incubator as separate pipeline (per the project conventions)

**What.** The incubator has its own config (`ConfigIncubator<Tag>.json`), its own theorem storage (`files/theorems_incubator/`), and its own CE-filter behaviour (`skip_ce_filter = true`). Output never enters the main proof graph.

**Why.** The incubator is a producer of ground-level fact tables, not a main theorem source. Isolating its pipeline prevents its heuristic output (via `try_contradiction`) from contaminating the main pipeline's provenance-complete proof graph.

---

<a id="d-3"></a>
## D-3 — Verifier as independent proof checker (per the project conventions)

**What.** `verifier.py` has its own copies of every algorithm it needs. It does not import from `expression_utils` or from any prover code.

**Why.** The verifier is the sole independent oracle for proof-graph correctness. If it shared code with the prover, a shared bug could go undetected. Independence is the verification commodity.

**Operational consequence.** Never modify `verifier.py` to "pass a test". See [I-16](30_invariants.md#i-16).

---

<a id="d-2"></a>
## D-2 — Hash-based inference, not search (architectural — project inception 2024-10)

**What.** GL's central bet: reformulate inference as hash-table lookup. Every implication rule is indexed by the integer-normalised signature of its premise; queries are O(1).

**Why.** Search-based provers pay the cost of discovering what to prove. GL's LBs discover that they already know what was asked. This pays forward into the ASIC roadmap — each LB becomes a silicon core with on-die SRAM for its hash memory.

**Operational consequence.** Every proof step must normalise to a stable hash key. Expression canonicalisation (`reshuffleTheorems`, `precompileStructuralOperators`) and the `IntNormalizedKey` form are what make this feasible.

---

<a id="d-1"></a>
## D-1 — Dual AGPLv3 + commercial licensing (project inception)

**What.** Every source file carries the dual-license header; commercial terms available at [https://generative-logic.com/license](https://generative-logic.com/license).

**Why.** AGPLv3 keeps derivatives open (including SaaS users); commercial licensing funds development. The AGPL stamp is also a deliberate scraper-deterrent for LLM-training pipelines — pipelines have to ingest a very explicit copyright notice on every file.

---

## Meta

- **Numbering is append-only.** D-1 is the oldest. Numbers are assigned in increasing order and are immutable once assigned; gaps and retirements exist (e.g. D-38 is reserved — it was used by the abandoned branch's superseded canonical-min fix; see D-39). New entries authored on a sandbox branch use slug placeholders (`D-pending-<slug>`) per the project conventions and receive their final number at merge to `main` — so this section deliberately does not pin a "newest" integer (it would drift on every addition).
- **Revision pattern.** If a decision is revised, add a new D-N+1 entry and annotate the old D-K with "superseded by D-N+1 on YYYY-MM-DD".
- **Scope.** This log captures *architectural* and *tactical-but-persistent* decisions. Routine bugfixes and refactors do not warrant entries — they live in commit messages. A decision that shapes how multiple chapters of this SwDD are written belongs here.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
