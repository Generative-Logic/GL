# Release Notes

## v0.9.1 — 2026-07-12

**The non-prover frame now has a complete cross-language timing ledger.** Every full run records Python orchestration, processed-proof generation, main and incubator HTML generation, both verifier passes, native save/export work, and raw-proof stack construction in one hierarchical JSON-lines report. The WSL direct-copy runner pairs each measurement with a SHA-256 manifest of every proof artifact, making sequential performance work measurable without relaxing byte identity. The first measured culprit was seven obsolete two-second filesystem waits between the conjecturer and native batch run; subprocess completion already closes every output handle, so the fixed fourteen-second delay is removed.

**SSD paging engine rebuilt: half the memory, the same proofs, near-baseline runtime.**

The static-memory prover can now run entire pipelines in half its previous memory reservation while streaming logic blocks to and from SSD continuously — with proof output byte-for-byte identical and total runtime within a few percent of the full-memory baseline. The paging engine was rebuilt end to end:

- **Raw memory images.** An evicted logic block is written as a verbatim image of its memory pages into one preallocated extent file (a database-style layout: per-block slabs, in-place overwrite, no per-eviction file churn) and reloads by reading straight back — no serialization, no index rebuilding, an order-of-magnitude faster round trip. The canonical element-stream format remains for the archival discharge images, where cross-run reproducibility matters.
- **A working-set pager that knows the future.** Because the prover's sweep order is deterministic, eviction is next-use-optimal: the pager keeps the blocks about to be processed resident, evicts the ones needed last, and maintains a free reserve so a worker thread never waits on disk it didn't cause. Prefetch and eviction run on a small pool of dedicated I/O threads, overlapped with proving.
- **No calculation cap per logic block.** Main-path hash bursts run their assigned work to completion instead of stopping at a fixed per-block calculation quota, so a large block is partitioned rather than truncated.
- **Expression-based logic-block splitting can reach unbounded granularity.** Heavy work is partitioned by expression structure rather than confined to a fixed number of rule buckets; continued expression-based refinement can make the parallel work units arbitrarily fine.
- **Quiescent blocks skip their turn.** A logic block that provably received no new work since its last burst is skipped entirely — validated by a shadow mode that runs every skipped burst anyway and asserts it produced nothing — so converged regions of a proof stop consuming sweep time and paging traffic.
- **Memory pressure is diagnosable.** If the pool is ever genuinely exhausted, the engine prints a full census of where every block sits before stopping, and per-phase telemetry reports the paging stream's volume and throughput throughout the run.

Eviction images are intentionally exempt from the byte-reproducibility contract that governs archival images — they never outlive a run. Everything the proofs depend on remains deterministic.

**Partial fix of the incubator slowdown** — one hot request-generation lookup restored to a single probe; proof output byte-for-byte identical.

**Mail memory is now measured per run and outgoing staging deloads with each logic block.** Every `main.py` run writes a paired memory report with the physical high-water, reservation, and attribution for all four static pools. Grids with no initially dormant logic blocks recycle delivered mail history and its global id table after each burst, while grids that need dormant catch-up retain the complete window. Each producer's outgoing mailbox and private id table now live in its deloadable logic-block arena instead of the always-on mail pool. The Windows confirmation preserved all seven theorem totals and completed 131,886 verifier checks with no failures; the largest measured mail-pool reduction was 417 MiB (92.1%) in the first Peano incubator batch.

**Next release: v0.10.0 — incubator theorems that require branching.**

## v0.9.0 — 2026-07-06

**Theme: MPU 0.1 — all per-logic-block prover memory moves off the malloc heap onto a fixed static pool.**

The prover's working memory is now statically allocated. One reservation is taken at program start — the software model of a fixed-SRAM accelerator — and every per-logic-block container (encoded statements, the string interners, equivalence classes, the hash inference engine, the goal registry, the cross-block mail log, and the logic-block objects themselves) is carved from it, with cold logic blocks streamed to and from SSD on demand. This removes the per-proof heap growth that made the next milestone — the Fundamental Theorem of Arithmetic — infeasible. Proof output is byte-for-byte identical to the prior heap-based engine; only where the memory lives has changed.

### Memory architecture

- **The burst kernel's compute paths are fully heap-free.** A compiler-verified sweep (a CodeQL call-tree inventory over the built translation units, regenerated after every change) now shows zero heap-using functions on the production paths of the prover's three-phase kernel: every transient working structure — string scratch, admission and rejection state, equivalence-class processing, origin records, the disintegration and integration working forms, and the cross-block mail absorb — runs on the static substrate. The only heap remaining inside the kernel is the debug trace dump (off the compute path), and the compiled-definition config tables stay heap-resident data behind zero-allocation read fences (ROM on silicon). Proof output remained byte-for-byte identical at every step of the conversion, and the pipeline runs measurably faster with the kernel's allocator traffic gone.

- **All prover working memory is static.** Per-logic-block containers no longer call the system allocator; they pack their elements into pages drawn from the one program-start reservation, and pool exhaustion is a loud, deterministic stop rather than unbounded growth.
- **Deterministic by construction.** Containers address storage through per-block virtual offsets that are a pure function of allocation order; physical block identity and grant order are unobservable, so every deload image and proof artifact stays byte-identical across runs and hosts.
- **On-demand SSD paging.** A background custodian deloads idle logic blocks to disk and reloads them on touch, keeping only the working set resident — so memory tracks the active proof frontier, not the whole run.
- **One unified memory substrate.** The transient calculation memory (per-worker scratch strings and staging-record pages) had run as a separate "hot" model alongside the cold per-block store — its own accounting, its blocks retained for the whole run. That second model is removed: every arena now draws from the one cold substrate, and scratch memory is returned to the pool the instant each worker finishes rather than held. Proof output stays byte-for-byte identical.
- **A parallel-phase cross-block write is eliminated.** While setting up its recursion budget, a recursion step could reach across logic blocks to update its parent fold block's tables from a worker thread; with many recursion steps sharing one parent, those updates could race and sporadically abort a Gauss-batch proof with a cold-index corruption. The cross-block update is now deferred to the single-threaded phase that runs after the workers join — the same discipline every other cross-block effect already follows — so the parallel phase writes only block-local state. Proof output is unchanged.
- **The last per-logic-block bookkeeping moves off the heap.** Each logic block's arena still kept its own block and page bookkeeping — which physical block backs each byte offset, which physical page backs each virtual id — on the malloc heap; this was the one piece the static-memory work above left behind. It is now static too: a small-buffer directory that holds a typical small arena's tables inline at no extra pool cost and spills only a large arena onto the pool, an intrusive page free-list with no side container, and a compaction scratch drawn from a pool independent of the one it reclaims. Proof output is byte-for-byte identical; only where the bookkeeping lives has changed. The single remaining deliberate heap user is the global pool's own recycle queue — one process-wide root, outside any logic block.

### Documentation

- **MPU 0.1 hardware booklet.** A new standalone page (`docs/MPU/index.html`) maps the static-memory prover onto silicon for a hardware and computer-architecture audience — the logic block as a core, the static memory hierarchy as on-die SRAM with on-demand SSD paging, the cross-block mail system as an on-chip network, and logic-block split as the parallelism lever — together with an order-of-magnitude estimate of what building a Mathematical Processing Unit (MPU) would cost at one, a hundred, a million, and a billion units.
- **User SwDD accuracy pass.** The user-facing prover reference (`docs/user_swdd/`) was reviewed page by page against the current engine and corrected: the static-memory chapter, the pull-model mail system, the anchor and proof-method catalogues, and the silicon-target naming now match the shipped code. Several concept diagrams whose in-figure captions overflowed their frames were also repaired.

### Build & portability

- **Undefined-behaviour and build-robustness pass.** A compiler-sanitizer sweep found and removed the engine's sole undefined behaviour — a left-shift of a negative sentinel id while packing two 32-bit ids into a 64-bit map key — by routing every such pack through one well-defined helper that produces bit-for-bit identical keys, so proof output is unchanged. Alongside it: a stale unit test that broke the Linux build and a non-deterministic test that compared struct padding are both fixed, and the release tooling now aborts loudly if the Linux binary fails to build rather than silently shipping without it.

## v0.8.2

**Theme: integration-side equivalence-class symmetry, cross-batch operator-naming consistency, end-of-burst memory reclamation, and release-quality infrastructure.**

A post-v0.8.1 release that closes three independent soundness / completeness gaps and lands the release-build infrastructure follow-ups. The integration-side equivalence-class machinery (admission template and rejection buffer) is now symmetric with the algebra side, retiring a deliberate provenance-free direct-insert path and enabling the temporary local introsort port from v0.8.1 to be replaced by the standard cross-host `std::stable_sort` at the four rule-firing hot-path sites. The spontaneous-operator namespace is unified across every batch in a run, closing a cross-batch naming divergence between the incubator and main proof stages and a latent allocator cross-category collision. When an implication subproof closes, its per-implication state across the scope subtree is now physically reclaimed at end-of-burst rather than left as filtered orphans, lowering memory pressure on proofs with deeply nested OR branching. The release also tightens the release pipeline itself: a `VERSION` file as the user-visible identity anchor and a `SHA256SUMS` integrity manifest.

### Soundness & determinism

- **The "mirrored from" proof step is gone — both directions are proved for
  real.** A theorem's reverse direction used to be asserted from its forward
  proof (recorded as a separate "mirrored statement" row) rather than derived.
  That shortcut is removed: the reverse direction now enters the prover as its
  own conjecture and is proved explicitly, so every theorem in a proof graph is
  backed by a genuine derivation in the direction it claims.

- **Admission/rejection equivalence-class keys unified to one canonical
  representative.** Both the algebra- and integration-side admission and
  rejection key hooks now collapse an equivalence class to a single
  canonical representative and rewrite the key once, replacing the older
  enumerate-and-add path. A completeness gap this surfaced — the Gauss
  summation theorem's successor-case induction step no longer closing — is
  fixed: when a rewritten compound is re-mailed for derivation, its stale
  per-block deduplication state is reset so it re-derives correctly instead
  of being silently skipped.
- **Equivalence-class disintegration dedup gates on full disintegration.**
  The optional disintegration-dedup gate — which suppresses re-expanding a
  compound when an equivalence-class variant of it is already known — now
  keys on whether that variant was *fully disintegrated* (every existential
  inside it landed an admitted witness) rather than on mere presence or
  locality. This stops a variant whose witness was rejected from suppressing
  its admissible canonical twin; the two had mutually blocked the Gauss
  summation theorem's successor-case induction step. With the new gate the
  fold proof closes. The per-statement "seen" record carries a small
  full-disintegration flag for this.
- **Runtime-measurement snapshot writes its first table immediately.** The
  RT-measurement file writer's one-write-per-second throttle no longer
  suppresses the *first* snapshot (it was seeded so the initial write looked
  zero-seconds-old and was wrongly throttled). A tracker whose triggered
  lifetime is under a second now produces its file; this also fixes a
  measurement unit test that previously passed only when a stale file
  happened to linger, unblocking the C++ test gate on the Linux/g++ build.

- **Integration-side equivalence-class machinery now symmetric with
  the algebra side.** When a new equality forms, every map the prover
  uses to gate integration-side admission and rejection is rewritten
  under the canonical form. The admission template gains the canonical
  form via an additive insert (the original key stays for backward
  compatibility; multiple canonical forms accumulate as further
  equalities form). The rejection buffer drops any entry whose key
  contains a non-canonical class member and re-routes the rewritten
  compound through the standard rejection-and-revival pipeline, so
  the canonical-form rejection record has full provenance recorded by
  the normal production path. The asymmetric direct-insert path that
  previously rewrote the rejection buffer in place — and which had a
  known provenance gap — is retired.
- **Integration admission templates remain reusable.** The one
  deliberate asymmetry with the algebra side: when an integration
  admission fires (a witness is admitted via the template), the
  template is not erased. A single template can admit multiple
  distinct witnesses over the proof lifetime. The algebra-side
  single-use semantics, where consuming a head erases the template,
  does not apply here.
- **`std::stable_sort` restored at the four rule-firing-hot-path
  sites.** Previously these sites ran a local re-implementation of
  MSVC's `std::sort` to keep the Gauss summation theorem proving
  under a specific tie-order. With the integration-side admission
  machinery now symmetric, the fold proof no longer depends on
  introsort tie-order: it goes through cleanly under `stable_sort`'s
  emergence-order ties. The deterministic, cross-host
  `stable_sort` contract is restored; the local introsort port and
  its 250-line header are removed.
- **Cross-batch operator namespace unified.** Every batch's
  spontaneous compact-operator allocations contribute to a single
  shared registry that every later batch reads. Previously the
  incubator stage's allocations were strictly batch-local, which
  meant the same logical operator could carry one name in the
  incubator-side proof graph and a different name in the main proof
  graph — visible to the verifier's operator-registry consistency
  meta-check as cross-tag divergence. The check now reports airtight.
- **Allocator key is category-aware.** The spontaneous-operator
  allocator's lookup table now keys on `(body, category)` rather
  than body alone. Closes a latent cross-category collision in which
  two batches could allocate the same body shape under different
  categories (one as an implication, the other as an existence) and
  the receiving batch would compile its existence shape using the
  implication's name — silently routing a subsequent disintegration
  down the wrong rule path and dropping downstream proofs. The
  collision was dormant before this release because the cross-batch
  registry did not span the incubator boundary; closing that boundary
  required closing the allocator gap first.
- **Per-implication state reclaimed on closure.** When an implication
  subproof completes, the prover physically reclaims the per-implication
  state at its scope and all descendant sub-scopes at the end of the
  hash burst — previously this state lingered as inert orphans behind a
  scope filter. Theorem set unchanged; lowers memory pressure on
  proofs with deeply nested OR branching.
- **Unified universal-quantifier construction for implications.**
  Every implication — theorems included — now binds its variables
  under one rule, replacing a separate, narrower rule that previously
  applied only to theorems. The proof verifier's implication check is
  correspondingly unified into a single path. The change is
  representational: the set of proved theorems and every proof are
  unchanged.
- **Mail-broadcast implication compaction made deterministic.** The
  ASIC-preparation step that also compiles every broadcast implication
  into a compact named form was being driven from the parallel proof
  phase, so its name allocation raced: a run proved exactly the same
  theorems, but the compact-operator registry and the artifacts derived
  from it differed from one run to the next. The compaction is now
  deferred to a single ordered pass after the parallel phase, so
  identical inputs always yield identical compact names — byte-for-byte
  run-to-run reproducibility is restored, with no change to which
  theorems are proved.
- **Algebra admission-map updates deferred out of the hash burst.** The
  per-burst admission-map writes — a new admission template, its pending
  status, and the revival of any rejected fact the template now admits —
  are applied once from a single point after each hash burst's fixpoint
  loop instead of inline during it. A behaviour-preserving step toward the
  static-memory (ASIC) prover build, where every container mutation needs
  one well-defined drain point; the set of proved theorems and every proof
  are unchanged.

### Pipeline

- **Verifier-side Python unit-test harness reordered.** The in-tree
  Python test harness that exercises every verifier checker now runs
  after the pipeline produces its output rather than before. Its
  fixtures bootstrap from the per-batch operator registries on disk,
  so a pre-pipeline invocation against an empty registry would crash
  on absent-tag lookups before any real work happened.
- **Incubator anchor split (AI3 / AI8) and rung-1 rebase.** The
  14-slot `AnchorIncubator` (i0..i8 successor chain) is split into a
  9-slot `AnchorIncubator3` (i0..i3) alongside the renamed
  `AnchorIncubator8` (the original 14-slot anchor). The Peano- and
  Gauss-incubator groups each gain an AI3 sibling and the
  FTA-rung-1 batch is rebased onto AI3, so the incubator now runs
  five batches (`IncubatorPeano1` AI8, `IncubatorPeano2` AI3,
  `IncubatorGauss1` AI8, `IncubatorGauss2` AI3 mirror,
  `IncubatorGauss3` AI3 rung-1). The rung-1 batch no longer
  saturates the int16_t name-id cap because its smaller anchor only
  generates the i0..i3 simple facts it actually needs. The
  cross-anchor implication generator emits one bridge per distinct
  incubator anchor in a tag-group (so both `AnchorIncubator8 →
  AnchorPeano` and `AnchorIncubator3 → AnchorPeano` are now
  attached), and the incubator-to-simple-facts converter accepts the
  new digit-suffixed anchor names and rejects implication-bodied
  heads that the CE filter could not match anyway.
- **Hash-burst evaluation simplified to a single pass.** The inner
  hash-burst request evaluation was an iterated fixpoint loop. After the
  equivalence-class and admission reshuffles in this release line, no
  derived statement becomes newly available mid-evaluation, so the loop
  always reached its fixed point in a single pass and the iteration was
  redundant. It is now one pass, with the trap that captures a block's
  mid-pass closure moved to fire right after the pass. Proof output is
  unchanged — verified bit-identical per-burst proof state and verifier
  results against the prior build.
- **Request-generation validity prune.** Hash-request generation now drops a
  growing request as soon as its validity scope cannot match any rule the
  candidate key advertises, rather than carrying it to the rule-firing gate
  that would reject it anyway. The check runs on cached integer scope ids and
  is skipped for the root (`"main"`) scope, so it is free on the non-branching
  bulk of a run and does measurable work only inside branching proofs. Proof
  output is unchanged — bit-identical theorems and verifier results against the
  prior build; the headroom it opens is groundwork for the deeper
  case-splitting of the FTA ladder.
- **Hash burst now runs split across CPU cores, adaptively.** A heavy logic
  block's hash burst is spread over many cores through one flat work-stealing
  pool, and each block self-tunes its split — starting unsplit, escalating to
  the full split only when a burst saturates the per-part work cap (re-running
  that burst at the full split in the same cycle), and coarsening back when it
  runs light, so the split's setup cost is paid only where it earns it. Output
  stays deterministic run-to-run and the theorem set is unchanged; the split is
  a performance / memory lever and groundwork for the ASIC 0.1 static-allocation
  cut.
- **Clearer artifact naming — `.mpl` definitions, conjectures vs theorems.** MPL
  definition files now carry the `.mpl` extension that names the language they hold
  (`files/definitions/*.mpl`). The proof-pipeline artifacts move to standard
  mathematical vocabulary: the conjecturer's candidate output is now `conjectures.txt`
  (was `theorems.txt`) and the prover's proved output is `theorems.txt` (was
  `proved_theorems.txt`). A pure naming change — proofs, theorem counts, and verifier
  checks are unchanged.

### Tooling

- **Release identity.** New `VERSION` file at the repo root, surfaced
  by `python main.py --version`. The release script copies it into
  the release tree as the user-visible version anchor.
- **SHA256SUMS manifest.** `release_public.py` now writes a
  `sha256sum`-compatible `SHA256SUMS` file alongside the shipped
  binaries plus the release-identity files (`VERSION`, `README.md`,
  `RELEASE_NOTES.md`, `LICENSE`); a user can verify integrity with
  `sha256sum -c SHA256SUMS`.
- **Verifier unit tests run standalone at startup.** The Python verifier
  test suite now loads a small frozen fixture committed under
  `tests/fixtures/` instead of the pipeline's regenerated binaries, so it
  no longer depends on a prior proof run and executes up front alongside
  the C++ unit-test gate — a checker regression now aborts before any
  proof work rather than after the full pipeline.

### ASIC 0.1 mail-channel unification

- **Statement storage unified onto the int16 registry.** Every logic block used to store each statement twice — a heap-allocated string form and a flat int16 form maintained in lockstep at every write. The string vectors are removed: the int16 rows are now the only stored statement form, with the string view reconstructed on demand at the few boundaries that genuinely need text (diagnostic dump, mail, visualization). This removes the fattest per-statement storage ahead of the ASIC 0.1 static-allocation cut and eliminates a standing dual-write hazard; proved theorems, proof graphs, and all verifier checks are unchanged.
- **The last string/int mirror containers are gone.** The remaining lockstep string halves — the closed-scope mail filter, the axed-variable set, and the recursion-product set — are dropped in favor of their integer halves, and the string-keyed statement-registration record folds into the int16 statement registry as two per-row membership bits, so one packed-key map now carries both the registration and the duplicate-suppression records. Proved theorems, proof graphs, and all verifier checks are unchanged.
- **The statement indexes are re-keyed to packed integer keys.** The per-block local-statement membership set and the per-statement level index — the last string-keyed statement containers — now key on the same packed integer pair the statement registry uses, turning the disintegration locality gate and every level lookup into a constant-time integer probe with no string reconstruction in the hot path. Proved theorems, proof graphs, and all verifier checks are unchanged.
- **The goal registry is re-keyed to packed integer keys.** Each logic block's open-goal table (`toBeProved`) — the last string-keyed per-block registry on the goal side — now keys on the same packed integer pair as the statement containers, so goal discharge lookups, the block-deactivation survey, and the early-exit probe run as constant-time integer checks instead of re-parsing expression text; a dead, never-written tags field was dropped from the goal values in the same change. Proved theorems, proof graphs, and all verifier checks are unchanged.
- **Mail subsystem unified onto a single channel.** The legacy implication tuple channel is retired; implications now travel exclusively as the compact `(implication<N>[…])` form on the statements channel and are recovered receiver-side by a dedicated external-mail absorb path. Prepares the mail layout for the upcoming ASIC 0.1 static-allocation cut.
- **Same-burst absorb-fixpoint preserved.** Rules arriving in a logic block's inbox at burst N are installed and fire in burst N's own hashburst (pre-fixpoint absorb), eliminating an additional round of cross-burst latency that had pushed contradiction logic blocks past per-block name-id limits on incubator-Peano runs.
- **Anchor predicates are never rewritten by equivalence-class substitution.** A class containing an anchor-slot member can no longer enumerate `Anchor*[…]` variants on either the statement-rewrite path or the admission-map value loop; the original concrete anchor is preserved unchanged.
- **Induction-theorem and mirror-theorem registrations are canonical at emission.** The `globalTheoremList` entry registered for an induction theorem, and the body bound-variable order of a `createReshuffledMirrored` output, now match the binary-canonical reconstruction that chapter rows cite — closing two `origin` meta-check shapes that had silently misrouted citations between the registry and the chapter.
- **Per-step prover pipeline reorganized for predictable per-step cost.** Equivalence-class application and goal discharge move out of the statement-insertion path into single explicit per-step stages, and the per-block internal mail splits into a same-iteration revival channel and a next-iteration deferral channel — making a logic block's per-step work predictable ahead of the ASIC 0.1 static-allocation cut and tightening the parent→child mail-direction contract. The same pass closed a Peano-main missing-theorem regression and a cluster of origin-chain-termination check failures.
- **Integration admission registration drained post-fixpoint.** The one integration-side admission write a hash burst performed inside its fixpoint loop — registering a fresh integration template — is now staged as it is discovered and replayed once after the loop, joining the algebra side's per-burst admission drain. The burst's inner loop no longer mutates any admission container, giving the upcoming ASIC 0.1 static-allocation cut a single well-defined integration-admission drain point. Proved theorems and proofs are unchanged — the Gauss fold theorem and the full division-free summation batch prove identically.

### Verification baseline

- Main: 46 proved theorems including both Gauss / fold variants.
- Incubator: 1423 proved theorems.
- Verifier: all checks airtight end-to-end across Peano main, Gauss
  main, Incubator Peano, and Incubator Gauss.

### Documentation

- **User SwDD Section 1 review pass.** First systematic maintainer
  review of the user-facing concept document (`docs/user_swdd/`).
  The overview, pipeline, and deductive-neighbourhood pages were
  corrected: the pipeline is now described as eight stages rather
  than ten (definition compilation is part of the prover's startup,
  not a separately orchestrated step; the post-conjecturer
  normalisation / reshuffle write is a runtime workaround inside the
  conjecturer's emit logic, not a true stage); cross-batch data flow
  is shown as two parallel tracks (incubator + main) with selective
  cross-track exchange; orchestration is correctly credited to
  `run_modes.full_run()` in Python rather than to `main.py`. The
  branching milestone is updated to FTA-ladder rung 1
  (`{0, 1} = [0, 1]`); the `or0[7,2,1,3]` theorem is retained as the
  natural worked example for explaining OR disintegration. Matching
  corrections landed in the agent-oriented markdown SwDD
  (`docs/agentic_swdd/`).

### Next

The next workstream remains **ASIC 0.1 — the runtime and memory
optimization campaign**: static memory allocation of the entire
prover state (logic-block pool, hash memory, mail buffers, name
map). ASIC 0.1 is the gating release before the FTA push resumes —
current allocation patterns make FTA-scale proofs infeasible at
present consumption.

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

## Earlier releases

Pre-v0.8.1 release history is not maintained in this file. The
release-candidate orphan branches on origin (`sandbox/release_candidate*`)
plus `git log` against the source repo cover the per-version commit
trail. The headline milestones map to the *Three pillars* section of
the [README](README.md): Peano (algebra, 2025-09), Gauss (logical
transformations, 2026-02), and Branching 0.1 (case differentiation,
2026-05). Per-release detail starts being tracked here from v0.8.1
onwards.

---

GL is dual-licensed under AGPLv3 and a commercial license — see
[`https://generative-logic.com/license`](https://generative-logic.com/license).
Paper: [`arxiv.org/abs/2508.00017v4`](https://arxiv.org/abs/2508.00017v4).
