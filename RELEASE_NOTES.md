# Release Notes

## v0.8.1 — 2026-05-12

**Theme: hardening and audit infrastructure following FTA-ladder rung 1.**

v0.8.0 closed the forward direction of FTA-ladder rung 1 — the proof that
`{0, 1} = [0, 1]`, the foundational case-differentiation milestone on the
roadmap toward the Fundamental Theorem of Arithmetic. v0.8.1 is the
cleanup release that follows: round-trip soundness fixes, a complete
overhaul of the proof-graph visualization layer, the first in-tree unit-
test suites, and the prover's hot path split into separate translation
units.

### Next

The next workstream is **ASIC 0.1 — the runtime and memory optimization campaign**: static memory allocation of the entire prover state (logic-block pool, hash memory, mail buffers, name map). ASIC 0.1 is part of the branching campaign / FTA ladder and is gated to land before the FTA push resumes — current allocation patterns make FTA-scale proofs infeasible at present consumption.

### Soundness & determinism

- **Race-free contradiction-origin propagation.** Cross-batch determinism
  restored end-to-end; zero verifier failures on the Peano main and Gauss
  main baselines.
- **Mail-channel unification.** The internal-mail and main-mail delivery
  channels collapsed into a single struct that carries scope in its
  payload. Eliminates a long-standing class of cross-pair origin drift
  visible in equality-rule firing order.
- **Equivalence-class equality flow tightened.** Per-target origin
  gating on equality-class application, cross-pair equality control,
  symmetry-emission discipline. Restores 366 incubator-Peano theorems
  that an earlier band-aid had silently shed.
- **Build-stack rewrite.** The contradiction record now stays in the
  `__contradiction__` logic block; the chapter-goal target lives in the
  path stack. Fixes the chapter-193 origin-chain-termination cycle
  surfaced in the post-rung-1 audit.
- **Algebra equivalence-class hook on the admission map.** Walks the
  admission map, rewrites entries via the per-class substitution map,
  and fires the revisit pipeline on every new key — provenance-free
  and additive: the original entry is never replaced, the rejection-side
  map is never written.
- **OR-disintegration K-implication provenance.** Provenance recording
  on OR-branch K-implications plus a cross-scope firing fix; resolves
  a class of missing-origin failures on OR-derived theorems.
- **`_orint_` goal-flow row carries the sub-implication.** Previously
  recorded the bare disjunct, which broke proof-graph reconstruction
  when the goal flow re-entered an OR branch.
- **Ancestor-scope merge in `updateEquivalenceClasses`.** Closes a
  mail-front origin-selection bug visible in late-chapter equality
  classes.

### Verifier

- **Origin-chain cycle detection.** The verifier now flags any
  proof-graph row whose origin chain re-enters itself. This is what
  surfaced the cycles repaired in this release; cycle-freeness is now
  an enforced invariant, not an aspiration.
- **Definition-set consistency meta-check.** The verifier now cross-
  checks the variable-port typing of every connection site against the
  declared definition set, closing the formal-completeness gap on
  cross-port typing drift.
- **Strict-equality → namespace-comparable** on the local-encoded-memory
  check. Brings the runtime gate into line with the documented hash-
  engine locality semantics.

### Modularization

- **Hash engine extracted** from `prover.cpp` into a new translation
  unit `memory.cpp`. No semantic change; cleaner build dependencies.
- **CE filter extracted** from `prover.{cpp,hpp}` into a new translation
  unit `filter.{cpp,hpp}`.
- **Shared `enumerateEqClassRewrites` helper** between the two
  equivalence-class application sites that previously carried near-
  duplicate code.

### Testing

- **First in-tree unit-test suite for the conjecturer.** Six batches —
  premise filter, max-distinct, reshuffle / reform, worker, legacy
  fixtures, OR-pair. Wired into every MSBuild target.
- **First in-tree unit-test suite for the verifier.** Python coverage
  of the deeper-of rule and the equality-set checks.
- **Prover-style doxygen pass** across the conjecturer, verifier,
  prover, memory, compiler, filter, and compressor modules. ~80 vacuous
  legacy test assertions replaced with concrete checks.

### Tooling

- **Leftover-debug finder.** New `find_debug_dumps.py` script scans
  Git-tracked C++ source for `.debug/` ofstream blocks, debugger
  breakpoint-target scaffolding (`int test = 0; test++;` and the
  `breakpoint_here` variant), and explicit `// DEBUG / TRAP / DUMP /
  HACK / XXX` markers. Console output, `// TODO` comments, and the
  sacred hashburst dump are not flagged.
- **Release-pipeline trap gate.** `release_public.py` runs the finder
  with `--fail-on-traps` as Step 0; releases abort if any leftover
  `.debug/` trap block or breakpoint scaffolding remains.
- **Snapshot tool overwrite-on-collision.** No more `GL_snapshot (N)`
  rotation; existing target and zip are deleted before the new
  snapshot lands.

### Source cleanup

Pure investigation residue removed; algorithm semantics unchanged.
Where a dump preceded an assert, the dump goes and the assert stays.
−310 lines of C++ across five files.

- **`visualizer.cpp`** — `checkZeroStack` theorem-12 TRAP infrastructure
  and `buildStack` no-origin dump removed.
- **`prover.hpp`** — `name_overflow_dump`, `incube_gauss_assert_trap`
  plus the `g_excludeRepetitionsBreadcrumb` infrastructure that fed
  it, and the orphaned `debugBody` parameter.
- **`conjecturer.{cpp,hpp}`** — leftover `Conjecturer::dumpDebug`
  function and its permanently-disabled `CONJ_DEBUG_DUMP` macro.
- **`prover.{cpp,hpp}`** — six debugger breakpoint-target blocks left
  behind from past investigations into specific anchor / interval LBs.

### Visualization

- **Major HTML proof-graph overhaul — readability first.** Chapter
  rendering rewritten end-to-end: 40px click-back-home GL symbol in
  the paragraph footer, external citation pipeline, namespace tags
  rendered unclickable, and a consistent within-chapter link-target
  rule. Published proof graphs now read as documents rather than as
  raw dependency dumps.
- **OR-branching readability.** Disjunctive branches render as a
  cohesive sub-chapter rather than a flat row sequence: formatter
  glue installed at the parent level, K-implication provenance
  threaded through cross-scope firings, and HTML links inside a
  branch resolve to the branch's own chapter rather than to the
  enclosing one.
- **Origin-chain cycles in proof graphs eliminated.** Multiple sites
  that previously produced cycles in the origin chain are now gated:
  equality1 emission on existing target origin, equality-mirror push
  gated on `local`, contradiction-record placement in the
  `__contradiction__` LB, and buildStack chapter-row lifting to the
  closest-to-main ancestor with origin. Proof graphs at v0.8.1 are
  cycle-free across Peano main, Gauss main, and Peano incubator.
- **Variable typing in the processed proof graph.** Digit arguments
  render as `v<N>` / `V<N>` (theorem-anchor scope), bound variables as
  `w<N>` / `W<N>` per implication.
- **`EnumerationSet2`** rendered readably.

### Documentation

- **Agentic SwDD introduced.** Long-form Software Design Document at
  `docs/AGENT_SwDD.md` — dense, cross-linked, sized for an AI assistant
  working on a code change. Maps the ten pipeline stages, names every
  load-bearing invariant, lists weaknesses and gotchas, and walks one
  theorem end-to-end through the pipeline. Meant to simplify tasks for
  AI agents: paste an error message plus the relevant chapter into the
  assistant and it has enough context to proceed.
- README framing updated: dropped the reverse-direction "next frontier"
  claim, added the incubator-side ground-fact list.
- Architecture documentation tightened across every chapter that
  changed in this release window.

### Verification baseline

**All verifier checks airtight** on Peano main + Gauss main + Peano
incubator on the deepest verification run that landed inside this
release window. The Gauss / fold theorem (both forward and mirror
variants) is back in `proved_theorems.txt` after the temporary sort
workaround below.

### Temporary workarounds

- **Local MSVC-introsort port at the rule-firing sort sites.** A new
  in-tree header (`msvc_sort.hpp`) provides a byte-faithful
  reimplementation of MSVC STL's `std::sort` and is wired into the
  four sorts in `generateEncodedRequests*` (the rule-firing hot path).
  This is a **temporary measure to unblock this bug-fix release**: it
  restores the Gauss / fold theorem proof that an earlier
  comparator-widening change had inadvertently broken, and it
  guarantees byte-identical sort output across Windows and Linux hosts
  regardless of which standard library is in use. Trade-off: about
  10 % slower than `std::stable_sort` would be at the same sites. The
  intended end state is `std::stable_sort` (emergence-order tie
  resolution); migration is gated on the upcoming runtime / memory
  campaign restructuring the rule-firing layer so that the prover is
  no longer tie-order-sensitive. Revisit explicitly once that work
  lands.

---

GL is dual-licensed under AGPLv3 and a commercial license — see
[`https://generative-logic.com/license`](https://generative-logic.com/license).
Paper: [`arxiv.org/abs/2508.00017v4`](https://arxiv.org/abs/2508.00017v4).
