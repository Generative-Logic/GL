<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Per-structure memory measurement `[DRAFT]`

> Status: Living. Permanent, off-by-default instrumentation layer that answers
> "which statified structure holds the memory" the way
> [`rt_measurement.md`](rt_measurement.md) answers "which section holds the
> time". Source-of-truth design document — every claim here should match
> `src/infra/mem_tracker.hpp` / `mem_tracker.cpp` and the three `MEM_MEASUREMENT`
> call sites.

---

## Purpose

The static memory hierarchy is sized by one program-start reservation per pool
([I-95](../30_invariants.md#i-95)), and pool exhaustion is an assert naming the
config knob — never a fallback. Sizing that reservation, and every argument
about what an FTA-scale run will need, rests on knowing which structures
actually hold the bytes. Before this layer the only figures available were
pool-level (`blocksInUse`) and per-LB (`blocksHeld`), both of which stop at
"a block is in use" and say nothing about which of the ~140 statified
containers filled it.

The layer attributes bytes to structures at the granularity the deload format
already uses: the `LbMemory::ContainerTag`. `main.cpp` writes two views of one
snapshot per batch — `.rt/_memory_<anchor>.log` for grepping and quoting, and
`.rt/_memory_<anchor>.html` for browsing the hierarchy.

---

## Compile-time gate and tunables

The gate sits beside `RT_MEASUREMENT` at the top of
[`parameters.hpp`](../../GL_Quick_VS/GL_Quick/src/parameters.hpp); the tunables
are the `MemMeasurementParameters` struct near `RTMeasurementParameters`.

```cpp
#define MEM_MEASUREMENT 0                              // master gate

static constexpr int MEM_TAG_SPACE     = 1024;         // per-tag table size
static constexpr int MEM_MAX_SLOTS     = 64;           // worker accumulator rows
static constexpr int MEM_MIN_PERCENTAGE = 0;           // table filter (0 = every row)
```

| Symbol | Default | What it controls |
|---|---|---|
| `MEM_MEASUREMENT` | `0` | Master gate. `0` removes all three call sites at compile time. The tracker itself compiles unconditionally so the unit tests exercise it in a normal build — the `RT_MEASUREMENT` convention. |
| `MEM_TAG_SPACE` | `1024` | Size of the per-tag byte table. The tag IS the index, so this is sized by the highest `ContainerTag` plus room, never by the count of live containers. Two parallel tables of this size hold content and derived-index bytes; slack has no tag and lives in its own counter. Appending a tag past the ceiling trips a Rule-19 assert. |
| `MEM_MAX_SLOTS` | `64` | Worker accumulator rows. Must be at least the run's `logicalCores`; an out-of-range slot asserts. Static storage is `MEM_MAX_SLOTS * MEM_TAG_SPACE * 8` bytes. |
| `MEM_MIN_PERCENTAGE` | `0` | Structure rows below this share of the peak are folded into one trailing line. |

---

## What is measured, and against which denominator

Three families, deliberately kept apart because they do not share a
denominator.

**1. Per-LB deload-enrolled containers — logical content bytes.** Everything
`LbMemory::visitContainers` reaches ([I-123](../30_invariants.md#i-123)): the
direct `LbMemory` tags, the four `HashMemory` bands at 51 / 151 / 251 / 351,
both `ColdMail` instances at 455 / 505, the changed-classes buffer at 655, the
name caches at 705, the deloadable mail-out at 755. The figure is
`sizeof(value_type) * size` — byte-for-byte what the deload directory records
for that tag, so a row here is directly comparable to the `.deload/` image.

**2. Derived hash indexes — measured per container.** The `PagedHashIndex` slot
array inside every cold map is rebuilt on reload and never deloaded
([I-117](../30_invariants.md#i-117)), so it carries no `ContainerTag` and family
1 cannot see it. `HashMap::indexBytes` exposes it, and the FIRST facet of each
container — `KeysView` for a POD key, `LengthsView` for a byte key — forwards
that accessor. Because only the first facet has it, a visitor walking every
facet adds a container's index exactly once, attributed to the tag that names
the container. This deliberately needs **no second enumeration**:
`visitContainers` keeps its role as the sole container walk
([I-123](../30_invariants.md#i-123)) and there is no parallel member list to
drift.

**2b. Block and page slack — the residual.** Whatever an LB's arena physically
holds beyond content and index is slack: the unused tail of every pinned 256 KiB
block and of every touched 8 KiB page. It belongs to no container, so it is a
single counter rather than a row — `blocksHeld * blockBytes` minus content
minus index. Every LB with any content pins a whole block
([G-52](../50_gotchas.md)), so at incubator scale this is expected to dominate.

**3. Process-wide singletons — physical block bytes.** The four pool footprints
(`staticMemory` / `persistentMemory` / `mailMemory` / `lbMemory`) and the two
scratch registries' summed peak. These are printed in their own section and are
**never** folded into the percentages: a pool figure counts whole blocks, a
container figure counts content, and adding them would double-count and
overstate. The pool section is context for the container table, not part of it.

---

## Where the samples land, and why

**Per LB — the end of `performElemPhase3`.** A cold container reads as empty
when its arena is deloaded, behind a residency assert
([I-111](../30_invariants.md#i-111), cookbook pitfall 12), so only a resident LB
can be polled at all. The end of phase 3 is the last point at which the burst's
LB is still claimed and loaded. The sample is placed immediately BEFORE the
Rule-14 hashburst EXIT trap, which is untouched.

Each worker writes only its own slot row (`coreId`), so concurrent samples never
share a destination and no atomic is needed. An LB idle across the iteration
contributes nothing — which is the intended reading, not a gap: it is deloaded,
and its content is on SSD rather than in RAM.

**Per iteration — the head and the phase-3 seam of `proveKernel`.**
`resetIteration` runs at the single-threaded head, before workers spawn, so
the fold counts each active LB exactly once. `commitIterationSample` runs
immediately after `steward->endPhaseWindow` closes the phase-3 window and
BEFORE the steward's discharge and deload work further down, so the pool
figures it reads are the iteration's high-water rather than the post-eviction
remainder.

**Per batch — `main.cpp`.** Two files from one snapshot and one tree builder,
so they cannot disagree: `dumpMemAggregate` writes `.rt/_memory_<anchor>.log`
(greppable, quotable in a commit message) and `dumpMemHtml` writes
`.rt/_memory_<anchor>.html` — a collapsible instance → container → facet tree
carrying content, index, total, share of peak and share of content per row. The
HTML is self-contained: no stylesheet, script, font or network reference, so it
opens offline.

---

## The two denominators — "peak" and "content"

Every percentage in this chapter is against one of two numbers. They are not
interchangeable and the tables label which is which.

**Peak** is content plus derived index plus slack at the fullest single sample.
Because slack is *defined* as physical minus content minus index, the three add
back up to physical, so:

> **peak = the whole physical block footprint of the logic blocks that were
> active in the peak iteration.**

On the FTA shortcut that is 1 007 157 248 B = **exactly 3 842 blocks** of the
256 KiB grant unit — a whole number, which is the arithmetic confirming the
identity: a non-integral result would mean slack had drifted away from its
definition. It is the honest "how much RAM did the working set pin" figure, and
it is what the `%peak` column divides by in both output files.

**Content** is family 1 alone: `sizeof(value_type) * size` summed over every
deload-enrolled container — the element bytes actually stored inside those
blocks, byte-for-byte what the deload directory records. On the same sample,
296.33 MiB.

The relationship is therefore block occupancy:

| | MiB | share of peak |
|---|---|---|
| peak (3 842 pinned blocks) | 960.50 | 100% |
| content (elements inside them) | 296.33 | **30.85%** |
| `<derived indexes + block/page padding>` | 664.17 | 69.15% |

**The active working set's blocks are 31% full.** That is what the 69% overhead
row means, and it is why both denominators are needed: against `peak` every real
container looks tiny (`originStrings` is 4.69%) because it is competing with
two-thirds of empty and index space, so `peak` answers "what does the machine
cost" while `% of content` ranks containers against each other and answers
"what is the data".

Neither denominator is the pool. `staticMemory` held 1 365.75 MiB = 5 463 blocks
at that instant, so the peak iteration's active LBs owned 70% of resident
blocks; the other 30% belonged to LBs resident but idle, which contribute
nothing to either figure (they are unpollable — see the sampling section).

---

## The percentage is a share of one instant

Each commit compares the iteration's grand total (families 1 and 2) against the
running high-water. On a new high the WHOLE vector is copied into the peak
snapshot. The reported percentages are therefore each structure's share of one
real moment — the fullest the machine ever got — not a collage of per-structure
maxima that never coexisted.

This is the deliberate difference from RT's column. RT sums seconds, and a
cumulative sum is meaningful because time is a rate. Bytes are a level: summing
a level over samples measures residency-duration, not size, and taking each
structure's own maximum describes no instant at all. The peak instant is the
figure that answers "what filled the pool".

---

## File format

```
Per-structure static memory at the peak end-of-burst sample

Iteration samples committed : 412
Peak sample                 : #237
Peak attributed bytes       : 1873141760  (1786.35 MiB)
Display threshold           : 0 % of peak

Structure                                                    |        bytes |    MiB |  %peak
-------------------------------------------------------------+--------------+--------+-------
<derived indexes + block/page padding>                       |    901775360 | 860.00 | 48.14
overallHashMemory.encodedMap                                 |    312475648 | 298.00 | 16.68
...
-------------------------------------------------------------+--------------+--------+-------

Process-wide footprint at the same sample (PHYSICAL block
bytes — a different, strictly larger denominator ...)

  pool: staticMemory (Main, deloadable per-LB store)          ...
```

Rows are one per structure, facets of the same container folded together — the
five `encodedMap` columns are one `encodedMap` row — sorted by descending bytes.

---

## First measurement — the FTA shortcut, 2026-08-22

One `main.py --shortcut` run with the gate at 1, 45 iteration samples, peak at
sample #5. Raw table archived at .

| | bytes | MiB |
|---|---|---|
| Peak attributed (families 1 + 2) | 1 007 157 248 | 960.50 |
| — of which the overhead row | 696 433 989 | 664.17 |
| — of which logical container content | 310 723 259 | 296.33 |
| `staticMemory` pool at the same sample | 1 432 092 672 | 1365.75 |

**Measured split of the peak** (second run, `MEM_MEASUREMENT` at 1, same 45
samples, same peak at #5 — content is unchanged, the former single overhead row
is now resolved):

| | bytes | MiB | %peak |
|---|---|---|---|
| container content | 310 723 259 | 296.33 | 30.85 |
| derived hash indexes | 133 349 376 | 127.17 | 13.24 |
| block + page slack | 563 084 613 | 537.00 | 55.91 |

**Slack dominates and indexes are an order below it**, which settles the
question the single row could not answer. 537 MiB of the 960.50 MiB pinned is
empty space inside blocks and pages. But 127 MiB of derived index is not
negligible either: it is 43% of all container content, and it never appears in
a `.deload/` image.

**The sharpest result is inside the index column.** The three per-burst
`HashMemory` instances carry bucket arrays wildly out of proportion to their
contents:

| instance | content MiB | index MiB | ratio |
|---|---|---|---|
| `workingMemory` | 1.85 | 9.06 | 4.9x |
| `localHashMemory` | 1.05 | 8.28 | 7.9x |
| `localHashMemoryDelta` | 0.21 | 4.81 | 22.9x |
| all three | 3.11 | 22.15 | 7.1x |

That is **17% of all derived index for 1% of all content** — a near-empty map,
once per LB, each still paying a full bucket array. `overallHashMemory` by
contrast is 51.03 content against 23.36 index, a sane ratio. The per-burst
instances are emptied every step by design, so their indexes are sized for
traffic that has already gone.

Per container the index is now visible where the first run could not show it:
`originStrings` 49.09 content + 13.34 index, `exprOriginMap` 31.38 + 13.45 —
provenance carries 26.8 MiB of index on top of its content.

Instance-level roll-up at the peak:

| structure | content | index | total | %peak |
|---|---|---|---|---|
| `LbMemory` (direct members) | 207.98 | 63.23 | 271.21 | 28.24 |
| `overallHashMemory` | 51.03 | 23.36 | 74.39 | 7.74 |
| `mailOut` | 28.92 | 13.50 | 42.42 | 4.42 |
| `workingMemory` | 1.85 | 9.06 | 10.91 | 1.14 |
| `localHashMemory` | 1.05 | 8.28 | 9.33 | 0.97 |
| `eqClassNameCaches` | 4.27 | 2.25 | 6.52 | 0.68 |
| `localHashMemoryDelta` | 0.21 | 4.81 | 5.02 | 0.52 |
| `sameInternalMail` | 1.03 | 2.59 | 3.62 | 0.38 |
| `nextInternalMail` | 0.00 | 0.08 | 0.08 | 0.01 |

Raw artifacts archived at  / `.html`.

Within the 296.33 MiB of logical content the top ten structures are 69.1% and
the shape is string-heavy:

| Structure | MiB | % of logical | % of peak |
|---|---|---|---|
| `originStrings.bytes` | 45.03 | 15.20 | 4.69 |
| `intEncodedStatements` | 40.88 | 13.79 | 4.26 |
| `intLocalEncodedStatements` | 20.97 | 7.08 | 2.18 |
| `overallHashMemory.normalizedEncodedSubkeys` | 18.86 | 6.37 | 1.96 |
| `mailOut.strings` | 17.23 | 5.81 | 1.79 |
| `exprOriginMap.blobPool` | 15.01 | 5.06 | 1.56 |
| `intLocalEncodedStatementsDelta` | 12.54 | 4.23 | 1.31 |
| `valueStrings.bytes` | 12.07 | 4.07 | 1.26 |
| `intExternalStatements` | 11.61 | 3.92 | 1.21 |
| `overallHashMemory.encodedMap` | 10.56 | 3.57 | 1.10 |

Two readings worth recording:

- **`originStrings` is the single largest container** at 49.09 MiB (16.57% of
 content), well clear of the runner-up — `overallHashMemory`'s largest
 container, `normalizedEncodedSubkeys`, is 18.86 MiB. Adding all four
 `exprOriginMap` facets (31.38 MiB) puts **provenance at 80.47 MiB, 27.16% of
 logical content**, and origin history is process documentation that is never a
 proof input ([I-44](../30_invariants.md#i-44), Rule 16). The forthcoming ASIC
 / fast-PC build runs without an origin map at all, and this is the first
 measurement of what that saves.

 **Compare at the right level.** Provenance as a whole is 1.58× a single
 `overallHashMemory` (51.03 MiB) and 1.49× all four `HashMemory` instances
 together (54.14 MiB). The origin interner ALONE is not: at 49.09 MiB it is
 1.94 MiB *smaller* than `overallHashMemory`. `originStrings` is one container
 and `overallHashMemory` is an instance holding seventeen, so a claim that one
 "outweighs the hash engine" has to name which aggregate it means — for the
 interner alone the ordering runs the other way.
- **The four statement columns** (`intEncodedStatements`,
 `intLocalEncodedStatements`, `intLocalEncodedStatementsDelta`,
 `intExternalStatements`) are 86.00 MiB, 29.0% of logical content — the
 expected shape, since `IntEncodedExpr` is a flat 352-byte row (I-84).

### Roll-ups the per-container table does not show directly

**The hash engine — all four `HashMemory` instances: 54.14 MiB, 18.27% of
logical content, 5.64% of peak.** `overallHashMemory` is 51.03 MiB of that;
`workingMemory` 1.85, `localHashMemory` 1.05, `localHashMemoryDelta` 0.21 — the
three per-burst instances are noise beside the persistent one. Rolled up by
container across all four:

| Container | MiB | % of content |
|---|---|---|
| `normalizedEncodedSubkeys` | 19.86 | 6.70 |
| `encodedMap` | 11.42 | 3.85 |
| `normalizedEncodedKeys` | 9.55 | 3.22 |
| `remainingArgsNormalizedEncodedMap` | 8.48 | 2.86 |
| `rejectedMap` | 2.27 | 0.76 |
| `rejectedMapIntegration` | 1.65 | 0.56 |
| `originals` | 0.70 | 0.24 |
| the other ten containers together | 0.20 | 0.07 |

The three owner-set maps are 37.89 MiB — **70% of the engine's persistent state,
and 3.3× the `encodedMap` rule registry they protect**. The whole admission and
rejection complex is negligible by comparison (`admissionMap` 0.05 MiB; every
`*Ordis*` / `varsIn*` / `productsOfRecursionIds` container together under 0.05
MiB), which is worth knowing before anyone optimizes it.

**`NameMap` — 14.26 MiB, 4.81% of logical content, 1.48% of peak**, spread over
`nameStrings.bytes` 10.23, `validityNodes` 2.66, `nameStrings.lengths` 1.33,
`subStrings` 0.04. The validity-payload table being essentially free is
[I-104](../30_invariants.md#i-104) working as designed: scope structure lives in
the flat parent-pointer forest, not in per-scope strings.

**`intKnownStatements` — 1.21 MiB, 0.41% of logical content**, and the cheapest
thing in the machine relative to its role. Its two facets cross-check the
registry's contracts exactly:

- keys 1 015 936 B / 4 = **253 984 rows**, the packed `(originalId, validityId)`
 int32 key of [I-85](../30_invariants.md#i-85);
- values 253 984 B over those rows = **1.00 byte per row** — `StatementFlags`
 is a single byte, so membership plus the `local` / `fullyDisintegrated`
 payload costs one byte per known statement;
- `intStatementLevelsMap.keys` is byte-identical at 1 015 936 → the same 253 984
 rows, which is [I-182](../30_invariants.md#i-182) ("a levels row is never
 empty") holding exactly, measured rather than asserted. The levels map's three
 facets total 2.35 MiB.

For scale: 253 984 registry rows are carried by 121 768 `IntEncodedExpr` rows in
`intEncodedStatements` (42.86 MiB at a flat 352 B each, `I-84`). The registry
that decides what is KNOWN costs 1.21 MiB; the columns that store what the
statements ARE cost 35× that.

**Peak attributed is 960.50 MiB against a 1365.75 MiB Main pool** at the same
instant. The gap is not unaccounted memory: attribution covers only the LBs
active in the peak iteration, while the pool counts every resident LB including
those idle but not yet evicted. Reading the two as if they shared a denominator
is the mistake this chapter's three-family split exists to prevent.

---

## Adding a container

When a new `ContainerTag` is appended, add its row to `kLbMemoryTags` (or its
band's facet table) in `mem_tracker.cpp`. Nothing breaks if you forget: the tag
still contributes its bytes and renders as `tag<N>`, which is a visible prompt
rather than a silent omission. The same is true of a new facet inside an
existing band, which renders as `<instance>.offset<N>`.

---

## Relationship to the neighbours

- **RT measurement** ([`rt_measurement.md`](rt_measurement.md)) — the sibling.
 Same folder, same gate convention, same output directory, same
 never-a-proof-input discipline. They share no code and no file.
- **The hashburst dump** (Rule 14) — untouched. The memory sample sits before
 the EXIT trap and changes no line of it.
- **`Rule 16` / [I-106](../30_invariants.md#i-106)** — nothing here is a proof
 input, and no deload or steward decision reads a counter. The steward's
 deload-set decisions still read quiesced barrier counts and the grant ledger
 only.

---

## Weaknesses

### Suspected fragility

- **`ReverseArgsIndex` is still invisible.** The cold-map indexes are measured
 now, but the reverse membership side-index
 ([I-154](../30_invariants.md#i-154)) is not a `HashMap` and exposes no size
 accessor, so its bytes fall into slack. It is one container on one map, so the
 error is bounded and small, but slack is therefore a slight over-count and
 index a slight under-count.
- **Slack does not separate block padding from page padding.** Both are the
 residual. Splitting them needs a page-count accessor on `LbArena`; the
 distinction matters because the two have different fixes (block size versus
 letting small facets share a page).
- **Idle LBs are invisible by construction.** The table describes RAM residency,
 not total proof state. An LB deloaded across the peak iteration contributes
 nothing, so the figure is smaller than the sum of all `.deload/` images. That
 is the intended reading, but a reader looking for "how big is the proof" will
 be misled unless they read this paragraph.
- **`MEM_MAX_SLOTS` is a hard cap.** A run with more `logicalCores` than 64
 asserts rather than wrapping. Widen the constant; the storage is static.
- **The dump allocates.** `dumpMemAggregate` builds `std::string` rows and an
 `ostringstream`. It runs once at end of batch from `main.cpp`, outside the
 statified `standardProcessing` tree, so Rule 28 is not engaged — but it is not
 heap-free and must never be called from a burst.

### Not exercised by tests

- The three prover call sites are source-level annotations, exercised by hand
 (flip the gate, run `main.py`, read `.rt/_memory_<anchor>.log`). The tracker
 itself ships positive and negative unit tests in
 `src/tests/test_mem_tracker.cpp`.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
