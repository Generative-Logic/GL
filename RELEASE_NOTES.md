# Release Notes

## v0.11.0 — 2026-09-04

**Theme: the lemmas leading up to FTA, and FTA itself.**

**Processor by default, GPU by one flag.** `python main.py` runs every batch on the processor. `python main.py --GPU` runs Phase 2 of every batch on the GPU with byte-identical proofs, several times faster; a missing device or driver stops the run instead of silently falling back. Linux builds gain `make USE_CUDA=1`. The MPU booklet moves to 0.2 (GPU Edition) and records the measured GPU route.

**Lean 4 checking inside every run.** With a Lean toolchain installed, the full run and the shortcut export their proofs to Lean and kernel-check them, next to the HTML: `files/full_proof_graph/lean_export/` and `files/shortcut/full_proof_graph/lean_export/`. Every proof page links its Lean twin. The FTA shortlist's export depends only on its externals snapshot. The run's export folders are the release's Lean package; the previously tracked index-pinned corpora are retired. Divisibility, strict order, and their negations render as symbols in the proof graph.

**CUDA Phase 2 for the complete pipeline.** The GPU backend, previously scoped to the FTA shortcut, now owns Phase 2 for every batch of the standard pipeline: the processor route is the default of every run, `python main.py --GPU` selects the CUDA route for the whole run (a batch configuration may pin one batch with `use_gpu`, and `--phase2-backend` stays as the explicit override), a CUDA-selected batch also runs its counterexample filter through the device route, and full-run device images stream as deterministic fixed-capacity chunks sized by named audited capacity profiles. Proof outputs are bit-for-bit identical across backends: the shortcut and the complete pipeline reproduce the processor reference's theorems and verifier reports exactly on both routes, and the mixed GPU pipeline finishes measurably faster than the processor baseline.

**Divisibility is closed under addition (C6), and the mail relay that proves it.** The ladder's sum-closure lemma — d | a and d | b imply d | (a+b) — required teaching the prover to relay witness-bearing existence facts from parent scopes to the descendants that need them: mail-arrived premises are now sendable onward, relayed existence compacts disintegrate at the receiver with correctly-stamped witnesses, and a statement dropped by canonicalization still ships its provenance history so every exported proof step stays fully cited. A follow-on regression hunt — driven by a complete step-by-step trace of the affected Gauss proof against a reference run — hardened the admission machinery so that a name proven equal to a recursion product counts as one, keeping demand-driven inference stable no matter which equivalent name a relayed fact happens to carry. The shortcut ladder stands at 52 of 52 lemmas proved and the full pipeline's verifier reports every check passing.

**Independent Lean 4 kernel validation for Peano, Gauss, and the FTA shortlist.** A new downstream exporter translates existing processed proof graphs into typed, hash-pinned neutral certificates and then into an ordinary-predicate Lean theory. The tracked project checks the complete acyclic Peano corpus (65 theorems; the known cyclic sources 24 and 25 remain explicitly excluded, never assumed), all 29 Gauss main theorems, and the 42 requested FTA-shortlist theorems plus four internal supports. FTA consumes 17 Peano and two Gauss results as explicit external-certificate dependencies; four registry-era naming or premise-order differences are accepted only after structural definition equality and theorem-isomorphism checks, never as backward reformulations. In total, the pinned `lake build` accepts 140 public theorems and 6,173 named source-row facts with no admission shortcut. The Lean project, generated modules, manifests, certificates, exporter, and tests are tracked; only the rebuildable `.lake/` cache is ignored. This checker began as a standalone command; the Lean paragraph above folds it into every run.

**Totality of the order relation (shortlist lemma A15), and or-theorems from a single direction.** The prover reaches the next milestone on the shortlist: for any two natural numbers, one is less-than-or-equal to the other. The proof runs through the new sequenced case-analysis machinery — case branches open on demand and are released one at a time — and its or-shaped formulation is now licensed from a single proved direction: classically, "not A implies B" already *is* "A or B", so the former requirement that both mirror directions be proved separately is retired. The full standard pipeline and the shortcut pipeline both verify airtight with this campaign's lemmas in the pool.

**Robustness: every big run starts clean, and cross-run artifacts are registry-independent.** Both run modes now begin from an empty operator registry and empty fact tables, making each run a pure function of the tracked inputs; the shortcut's external corpus is kept in expanded base form, so nothing in it can be silently reinterpreted when the registry is rebuilt. Several latent soundness and provenance defects surfaced and were fixed on the way: a disjunct's sign could be lost when an or-theorem's registered form was rebuilt (producing a semantically flipped disjunction), the base-form theorem writer emitted malformed double negations for a class of existence statements, structurally identical or-operators could be minted under two different names, and induction-proved theorems circulated in a partially bound form whose compact citation the proof-graph walker could not resolve. All four are closed, with the external verifier airtight across the full pipeline, the incubator graph, and the shortcut graph.

**Shortcut mode: proving a curated lemma list against the corpus.** A new run mode (`python main.py --shortcut`) drives the road to FTA: the prover receives the previously proved Peano and Gauss corpus as external theorems plus a hand-maintained list of true conjectures — the FTA lemma shortlist — and assembles the proofs autonomously, in an order of its own making. The native prover learned to take an arbitrary conjecture-list file and an arbitrary external-theorems file on the command line; a dedicated configuration routes every output — proved theorems, the processed proof graph, the browsable HTML proof graph, and the verifier report — into its own `files/shortcut/` tree, leaving the main pipeline's artifacts untouched. The post-proof compressor is skipped in this mode so every granular lemma stays available to the lemmas that build on it. The first shortlist lemma (0 ≤ n, in witness form) is proved end to end with the verifier airtight.

**Trichotomy of the natural order (shortlist lemma A16), and negated definitions as case splits.** For any two natural numbers, either the first is strictly smaller, they are equal, or the second is strictly smaller — the next shortlist milestone, proved autonomously. The enabling capability is a new inference door: the negation of any operator defined as a conjunction is recognized as the disjunction it classically is (De Morgan), and is consumed by the existing case-analysis machinery at runtime — its mutual-exclusion rules fire immediately when standing facts decide the case, and a genuine case split opens when they do not. No new operator is compiled for this; the negated statement itself serves as the case-split identity. On the way, the hypothesis-exploration path was hardened: hypothetically assumed constituents now travel the same internal channel as every other statement, closing a latent one-sided-registration defect in negated equalities, and carry a dedicated internal-only provenance mark that never reaches the proof graph. The proof checker learned the new negated-conjunction shapes as strictly additive checks; both pipelines verify airtight, the previously proved theorem pool is fully preserved, and the full standard run reproduces byte-identically across independent runs.

**Below the successor (shortlist lemma A17), and complete n-way or-theorems.** The next shortlist milestone closes in both directions: a number is strictly below a successor exactly when it is at most the predecessor's value. The forward direction is the first shortlist proof to run a predecessor case split on a witness the prover minted itself, and it closes fully flat — the case-analysis rules decide the split from a derived not-zero fact, with no branch scopes opened. Two supporting lemmas join the pool as first-class library rows: "adding a nonzero amount changes the number" and the arithmetic core of stepping a successor across a sum, the latter deliberately packaged so downstream proofs stay within the engine's inference-width budgets. Separately, or-theorem assembly is now n-ary: a proved chain of negated hypotheses folds into one disjunction with every branch included — the trichotomy theorem, previously exported with a residual hypothesis, now ships as the full three-way disjunction — with the standard pipeline reproducing its previous results byte-identically.

**Case analysis no longer grinds inside dead branches.** When a case-split branch's assumed case is refuted, the engine already retires the branch at the end of the burst — but the equivalence-class machinery could still spend the bulk of that burst rewriting statements inside the doomed branch before the retirement ran, which surfaced as an eight-fold runtime blow-up on one shortlist batch. The equality engine now consults the same refutation evidence up front: changed classes living at or below a refuted branch are skipped and the branch is staged for the standing retirement, collapsing the affected batch back to baseline. Independently, the per-step changed-class delta is now consumed once per distinct class state rather than once per update event, removing redundant registry sweeps in update-heavy steps, and the hot paged-memory access primitives are force-inlined. Both pipelines verify airtight with the theorem pool unchanged and run-to-run outputs byte-identical.

**Case analysis no longer grinds inside finished branches either.** A case-split branch that has reached every goal on its chain is now frozen: its facts stay on record for the eventual convergence and the proof export, but it stops feeding the request generator, and a logic block that has nothing left to prove opens no case splits at all. Every proof and every verifier check is unchanged; the full pipeline runs about six percent faster.

**The FTA lemma campaign is under way, under its own anchor.** The shortcut batch now runs under a dedicated anchor — a same-slot twin of the Gauss anchor — with its own batch identity and operator registry. The previously proved corpus keeps its original Peano and Gauss anchors and is activated by two cross-anchor connection theorems that the prover proves itself at the start of each run, a first: anchor predicates proved as theorem heads, with the anchor-handling verifier checks passing throughout. On this foundation the opening stretch of the FTA shortlist is proved — the order-relation groundwork from 0 ≤ a through "less-or-equal and unequal implies strictly-less" — including a strict-order operator introduced by definition (less-or-equal and unequal), with the browsable proof graph and an airtight verifier report on every step.

**Cancellation of multiplication (shortlist lemma B8), demand-keyed case splits, and formulation-collision copies.** The prover closes the next shortlist milestone: if c ≥ 1 and a·c = b·c then a = b — the first shortlist row whose proof needs a case split that no standing fact decides. Three capabilities land together. Case-analysis rules gain single-exclusion forms for three-way-or-wider splits: refuting ONE case yields the remaining cases as a live disjunction, the step that turns a lone assumed-for-contradiction fact into a genuine case split. Parked case splits and hungry rules now find each other through a keyed rendezvous: a rule matched everywhere except one compound ground premise registers a demand under that premise's text, a parked split files under each of its cases' texts, and whichever side arrives second wakes the other — a split opens exactly when something can consume it, in either arrival order. And a conjecture that states an equality of two compound terms by fusing both result binders into one name now deposits a fresh-copy equality for the fused name at theorem load, so pool rules written with two distinct result variables can match the fused facts; the deposit is scoped to the conjecture's own premise chain, leaving unrelated conjectures untouched. Under the hood, statement bookkeeping moved to a one-door contract — a statement is known exactly when it carries a non-empty derivation-level set — retiring two legacy flags, and a latent provenance defect (branch-scope origin rows not shipped alongside main-scope contradiction records) was exposed by the new proof's graph export and closed. Both pipelines verify airtight end to end; the standing theorem pool is preserved byte-identically.

**Circular theorem citations are now impossible — and the Lean corpus is complete.** The proof-graph exporter could let two chapters justify each other: inside any single chapter a previously proved theorem is bedrock, so a mirrored pair of theorems could each ground itself in the other while every per-chapter check passed — exactly one such pair existed in the Peano batch, and one of the two shipped in the theorem pool. Two layers close the gap for good. The external verifier gains a global check that builds the directed "theorem A uses theorem B" graph over all proven theorems and fails every theorem lying on a cycle; on the pre-fix artifacts it flagged exactly the known pair and nothing else. And the chapter exporter now consults a running cross-chapter usage graph before accepting a proven-theorem citation, rejecting any citation that would close a cycle in favor of a genuine derivation — the surviving mirror now carries a real proof from the definitions, and its circular twin, never independently proved, left the graph. With the cycle gone, the Lean corpus is complete: the full Peano theorem list exports with an empty exclusion list, and the whole tracked project — Peano, Gauss, and the FTA shortlist on one manifest generation, 141 public theorems — passes the pinned kernel build with no admission shortcut. Two latent exporter defects surfaced by the rebuild (a negated case in reconstructed disjunction definitions, and stale certificate hash pins) were fixed on the way. All pipelines verify airtight, outputs reproduce byte-identically across independent runs, and the theorem pool is preserved.

**Divisibility opens Part C — with no new operator at all.** The divisibility stretch of the FTA shortlist begins, and its first design decision is a reduction: d | n is not a new operator but the existing "preorder" definition instantiated with multiplication in place of addition — the same witness form that already carries less-or-equal (∃k: d·k = n versus ∃k: a+k = b) — so the whole order-theoretic machinery applies to divisibility for free, and the prepared dedicated divides definition was deleted as redundant. The first two rows are proved: 1 divides everything, and every number divides itself, the latter deliberately stated as "a = b implies a | b" — the engine's native idiom keeps distinct names related by equalities rather than collapsing them into repeated-argument diagonals, and the supporting multiplicative-identity lemma ("a = b implies 1·a = b", the product analog of the addition base case) is stated the same way and proved by the engine itself. Alongside, the one deliberately over-premised pool row — kept until now as a designed prove-but-don't-register regression sentinel — was retired, making this the first campaign run in which every pool conjecture is proved and registered with the verifier airtight. The third divisibility row — everything divides zero — follows in its literal witness form on the next run, and the fourth — zero divides only zero — closes right after, keeping the ladder fully proved. Transitivity of divisibility follows as the first multi-premise divisibility row, with the engine finding the witness-product route on its own.

**Parallel search as an exact partition, and derivation levels that tell the truth.** The machinery that fans a heavy block's proof search across all cores was re-founded twice over. Each part now owns a disjoint subtree of the one search instead of re-deriving every candidate containing its assigned prefix — the split becomes an exact partition of the unsplit enumeration, collapsing the heaviest shortcut batches (roughly 220 down to 130 seconds end to end) — and when a mid-burst result dooms the remainder of a burst, every part now stops at one deterministically chosen line in its own stream, so the early exit saves the work without costing reproducibility. Three correctness campaigns land alongside. Goals proved by contradiction now retire their search machinery on success, eliminating blocks that previously stayed live to the end of the batch. Global registration of a proved theorem is now gated on its derivation citing every level of the definition it rests on — goal closure itself stays level-free — and the audit exposed a long-standing leak in which equality rewriting dropped the justifying pair's derivation levels; its fix recovered a silently lost theorem (1·a = a). Admission keys extracted from anchored rules now retain their anchor context, eliminating a spurious case branch. Finally, the Lean 4 kernel validation became an official release artifact: the release now regenerates the Peano, Gauss, and FTA exports in dependency order, byte-checks them, and runs the pinned kernel build, growing the kernel-checked corpus to 146 public theorems.

**A five-fold faster shortcut, and anchor copies confined to their purpose.** A day-long runtime campaign collapses the FTA-shortcut wall from over ten minutes to about two. Statements already known at an ancestor scope are no longer re-registered inside case-split branches — one shared ancestor-known predicate now guards every statement-registration door, refusals still deliver the case-split release signal, and an end-of-burst sweep retires late-arriving duplicates; the scope-asymmetry that had made one branch copy load-bearing (negated-equality expansion) was fixed rather than exempted. The disproof twin blocks are skipped for the all-true shortlist pool. And the anchor-copy machinery — fresh x-named copies of anchor numerals that let rules bind anchor values as elements — is now confined to where it serves proofs: induction sub-blocks arm the containment filter without holding the copied anchor themselves, so the copies no longer breed derived facts and rule twins there (over half of one monster block's rule registry was such breeding), with one deliberate exception for grids whose theorem head is about an anchor numeral, where the historical machinery demonstrably carries the proof. Closing that leak unmasked a long-masked provenance gap — case-analysis inference rules traveled to descendant blocks without their history rows, previously papered over by the leak's local re-derivations — now closed with paired history shipping, so every block can explain every rule it fires. Both pipelines verify airtight; the standard pipeline's theorem output is byte-identical to the pre-campaign baseline and reproduces byte-identically across independent runs.

**Difference closure of divisibility (shortlist lemma C8), and demand keys for witnesses in waiting.** The divisibility ladder reaches its hardest row yet: if d divides a, a + b = c, and d divides c, then d divides b — the first row whose proof must extract a witness from an order fact rather than construct one from the premises. Three gaps fell in sequence. The pool gains the order-introduction row (a + k = c implies a ≤ c) and a difference-transport row that folds distributivity and cancellation into one premise-rich rule. Case-split demand now treats an anchor constant as bound — a rule wanting "1 ≤ d" no longer fails the demand filter merely because the numeral 1 has no premise to bind it — which lets the d = 0 versus 1 ≤ d split open and the order chain fire inside the branch. And the deepest gap: a witness minted from an order existence sits at an operator's input slot (∃k: a + k = b), a position for which no consumable demand key could exist — the admission language marked output slots only, and the input-slot keys that did exist served case-split evidence invisible to general admission. Premise-rich rules now install consumable input-slot demand keys: when a rule's other premises match standing facts, the instantiated key demands exactly the witness shape that would otherwise park forever, and the admission probe consults the deposit's ancestor scopes, connecting a demand established at the main scope to a witness minted inside a case-split branch. The obvious alternative — admitting every two-input-operator witness unconditionally — was tried first and reproduced the historical runtime explosion; the demand-driven route proves the row with the pool fully proved (56 of 56), the shortcut verifier airtight, and a full standard-pipeline run whose verifier report is category-for-category identical to the previous full-run baseline.

**Divisibility antisymmetry closes Part C — by deriving the case split instead of running it.** The divisibility ladder completes, and its hardest row — if a divides b and b divides a, then a = b — arrives through new machinery: the theorem needs a case split on b = 0 versus 1 ≤ b, so the pool proves the two guarded variants as ordinary flat rows and a post-prove constructor merges them, licensed by a proved or-theorem stating that the two cases exhaust — classical or-elimination, with the merged theorem registered as a derived row citing both variants and its license, validated end to end by a dedicated verifier check and rendered as a first-class chapter. A new registration class ships alongside: a closure whose derivation turns out not to need one of its stated premises is now recorded with a full browsable chapter but never circulated — previously such sound-but-weaker closures vanished silently — and exactly these recorded rows are what the merge may consume as variants. The independently kernel-checked Lean corpus grows to the full 79-theorem shortlist, merged antisymmetry included, with no admission shortcut. And a registration-order race the new recording class exposed — the same theorem closing through two routes in one iteration, with a worker-scheduling artifact deciding which registration won — is fixed at the root: every registration drain now orders on proof state alone, and two full pipeline runs reproduce byte-identically, verifier airtight.

**One spelling per fact: the canonical statement door.** The prover's statement registry is now kept canonical: every deposited statement is rewritten at a single admission door to its equivalence-class canonical spelling, a duplicate whose canonical form is already on record is turned away at arrival, and the equality engine becomes the sole source of alternative spellings — it materializes a copy at a logic block exactly when that block does not yet know it, with no level condition standing in its way. Settled theorem heads are no longer re-deposited alongside the standing broadcast that already re-derives them with honest level bookkeeping, and the redesign surfaced and closed a series of latent bookkeeping gaps at the seams it touched: parked case-split cohorts losing their level provenance, per-step equality-class deltas discarded before first use, and provenance caps displacing established history rows. The full pipeline reproduces the previous theorem pool — the Gauss fold-composition lemmas included — with the verifier airtight on every batch and run-to-run outputs byte-identical.

## v0.10.0 — 2026-07-31

**Theme: Rung 2 of FTA Ladder — the first theorems requiring genuinely nested case analysis.**

**FTA ladder rung 2 proved: {0, 1, 2} = [0, 2].** The prover now closes the first theorem requiring genuinely nested case analysis — proving that the three-element enumerated set equals the interval [0, 2], with the false sibling conjecture ({0, 1} = [0, 2]) disproved outright in the same batch. The proof exercises a set of new engine capabilities end to end: contiguous nested disjunctions flatten into one atomic case-split cohort on both the assumption and the goal side; every disjunction now contributes its mutual-exclusion inference rules at any case-split depth; reductio blocks can prove a conjecture by refuting its negation, powered by a conjecturer template that emits injectivity contrapositives; and the request engine grants a deeper witness-variable budget inside case-split branches only — wide enough for two-level predecessor descent, narrow enough that batch runtime stays under the previous baseline. The result is reproduced byte-identically across independent runs, with the full verifier suite airtight. The 32-bit scope-name migration underlying this work raises the prover's naming capacity about thirtyfold.

**Disjunction introduction and sound firing scopes.** Every implication whose premise is a disjunction now also installs one introduction rule per case — a single true case states the whole disjunction — closing the gap that kept an enumerated set's concrete memberships underivable and unblocking the reductio path toward the rung 2.1 disproof ({0, 1, 2} ≠ [0, 1], closed below). Exercising the new rules surfaced a latent inference-scope leak: a rule installed inside a case-split branch could deposit its conclusion outside that branch when fired on outer facts. The firing engine now places every conclusion at the deepest scope among its rule and premises, the proof checker's scope discipline that had guarded this contract all along stays unchanged, and the verifier accepts the new introduction shape as an additional expansion form — strictly additive, every previously passing proof still passes.

**FTA ladder rung 2.1 closed: {0, 1, 2} ≠ [0, 1], and case analysis learned to discard dead cases.** The false sibling of rung 2 is now disproved by reductio, closing the rung with the result reproduced byte-identically across independent runs. The proof forced a family of engine capabilities into existence. A case-split branch whose assumed case is refuted — everything it derives holds only ex falso — is now retired outright: its state is deleted, its cohort's convergence bookkeeping shrinks to the surviving cases, and the exported case-analysis row accounts for every case honestly, citing each retired case's refutation as a reductio ingredient (the verifier checks both forms). On the affected batch this removed two thirds of the resident statements and two thirds of the batch's prover time. Vacuous truth is now governed end to end, in the sense of vacuity detection from model checking: a logic block that proves its own premise set inconsistent retracts the theorems it already published, theorem pairs proving contradictory heads from identical premises certify their shared premise set unsatisfiable, and a dependency-directed retraction pass retires anything whose proof leans on a vacuous theorem — classified into a dedicated artifact, excluded from the deliverable pool, with the honest proof chapters kept for audit. Anchor rules can now fire on variable-copy arguments per batch, the enabler for the antisymmetry route that closes the disproof.

**A refuted conjecture now takes its mirror down with it.** Every conjecture's reverse direction rides the prove pool as a conjecture of its own, but a finite-model counterexample against the forward direction never witnesses against the reverse — so false reverse directions routinely survived counterexample filtering and burned prover time. The conjecturer now records each operator-only conjecture and its pool mirror as an explicit pair, and when the counterexample filter refutes either member the other is discarded in the same pass. The heuristic is deliberately one-sided and openly documented: discarding never fabricates a proof, a config flag turns it off, and the full pipeline reproduces the identical theorem set byte-for-byte about 23% faster.

**Proved conjectures now stand down their machinery immediately.** An instrumented audit of logic-block lifecycles showed that a proved conjecture's contradiction blocks kept running for a few bursts after the proof (they retired only when the broadcast theorem re-fired on the original premises), and that blocks whose last goal closed after the batch's final proof stayed active to batch end. Both gaps are closed: proving a theorem now retires its contradiction blocks in the same step, and the global deactivation sweep runs every iteration instead of only on proof events. The proof output is unchanged byte-for-byte; the Peano main prover runs about 15% faster. The same audit confirmed induction auxiliaries were already retired promptly, and that the remaining batch cost is genuine proof-search work.

**Logic-block splitting is now a per-batch config switch.** The parallel work-partitioning lever is enabled only for the batches where it pays off; the third Gauss incubator batch is the first to run split.

**Candidate refresh.** Proof-graph pages now render negated case-analysis theorems in the standard "from … follows …" template, with proper inequality signs and set braces at any arity. An external review prompted three small hardenings: the incubator's vacuity retraction reads its own proof graph, the statement-cleanup bookkeeping contract is pinned and regression-tested, and the shipped Windows DLLs joined the release's integrity manifest.

**Next releases: v0.11.x — the lemmas leading up to FTA, and FTA itself.**

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
