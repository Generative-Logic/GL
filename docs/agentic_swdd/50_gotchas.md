<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Gotchas `[DRAFT]`

> Things that keep biting us. Each entry describes a concrete failure mode — how it manifests, the root cause, how to detect it, and how to fix. Distinct from [`30_invariants.md`](30_invariants.md): invariants are the *rules*; gotchas are the *stories of what happens when a rule is forgotten or a subtle convention is missed*.
>
> Sections are grouped by concern — workflow, build, code-pattern, domain-specific. Add a new entry whenever a bug you spend > 30 minutes diagnosing turns out to be a recurrence of a known class.

---

## Workflow gotchas

<a id="g-47"></a>
### G-47 — `compilation` row's `rest[0]` rejected when input form is alpha-equivalent to but binder-permuted from the binary's stored body


**Manifestation.** Verifier reports `compilation success 0, failure N` on chapters whose post-processed output emits `compilation` rows (typically `check_induction_condition` chapters for associativity / distributivity proofs). Reading the failing row, `line.expression` is `(implication<N>[])` and `rest[0]` is a structurally-valid expanded implication. The binary entry for `implication<N>` exists, category is `implication`, but the verifier's `_try_expand` rebuild does not match. Both forms are alpha-equivalent under a swap of two co-bound variables in a `>[w_i,w_j]` group plus a parallel swap inside the in-body args.

**Cause.** `compileImplicationToCompact` dedups inputs by alpha-equivalence (the same `implication<N>` name covers every binder-permuted variant of the same theorem). The binary's `elements` only records the first-seen body. A later input whose binder list `>[w_i,w_j]` is paired with an inner premise that reads `in3[w2,w_j,w_i,+]` instead of `in3[w2,w_i,w_j,+]` (i.e. the body's first-appearance order is `w_j` then `w_i`, but the binder names them in the opposite order) is alpha-equivalent to the binary's form yet not structurally identical. The verifier's `_normalize_with_unchangeables` rewrites both binder content and arg content to a fresh `v<k>` scheme — but only when the binder list's first-appearance order matches the body's. A binder-order mismatch produces normalized strings that differ in `>[v_k,v_l]` vs `>[v_l,v_k]`.

The pre-fix `prover.cpp::proveKernel` deferred-compaction drain passed the raw `original` (from `pendingCompactionQueue`) into the `compilation` origin row, exposing this mismatch whenever a second compaction touched an already-registered `implication<N>` with a binder-permuted body.

**Detect.** Open the failing chapter row; compare its `rest[0]` body to the binary's `elements` reconstruction. If they're alpha-equivalent but the inner `>[w_i,w_j]` group has its body args in the order opposite to the binder declaration, this is the gotcha. Cross-check with `(implication<N>[])`'s `expansion` row in the same chapter — that row's `line.expression` is in the canonical-binary form (`expansion` is reconstructed from the binary on the producer side too) and will differ from the `compilation` row's `rest[0]`.

**Fix.** Reconstruct from the binary at the compaction-emit site: `this->reconstructImplicationFullBind(elements[:-1], elements[-1])` after looking up `compiledExpressions[compactCore]`. Cite the result as `rest[0]`. See [I-52](30_invariants.md#i-52) and [D-81](40_decisions.md#d-81).

**Second emit site — `broadcastTheorems` (2026-05-22).** The same non-occurrence-order `rest[0]` surfaced on the FULL cross-batch run (3 `compilation` failures: `108_direct_proof` impl1100, `97_check_zero` + `98_check_induction_condition` impl1151) — a peano-only run hides it (no Gauss→Peano broadcast). Root cause: `broadcastTheorems` nested its `reconstructImplication` (FullBind) call *inside* the "head is an uncompiled existence" check, so a non-existence-head external was relayed as the raw `thOriginal` with its binders untouched. **Fix — at this site, FullBind the *theorem itself* (`finalTheorem`), not the compilation row.** Pull the reconstruction out of the existence gate so every relayed theorem is rebuilt in occurrence order; the existing `rest[0] = finalTheorem` citation is then already canonical. **Do NOT reconstruct a binary form into `rest[0]` here** (the drain's fix): `broadcastTheorems` only records a broadcast origin for `finalTheorem`, so a separate reconstructed string has no origin entry and `visualizer.cpp::buildStack` aborts with `"no origin found"` when it walks the compilation row.

**Related.** Similar binder-permutation alpha-equivalence can affect `implication`-tag rows' `rest[0]` (cited rule body) when the global theorem registry stores a binder-permuted form of the cited rule. The `_alpha_canonicalize_bound_vars` membership check used by the `origin` meta-tag does not try binder permutations, so origin failures can surface at chapters whose row-1 cited implication is alpha-equivalent to but binder-permuted from the registered form. Deferred pending direction on whether to canonicalize the global-theorem registration too. (The `mirrored statement` arm of this issue is gone — the prover no longer emits `reshuffledMirrored` registry rows; see D-112.)

---

<a id="g-46"></a>
### G-46 — A non-empty level set in a `Mail::statements` deposit silently swallows every theorem derived from the recovered rule


**Manifestation.** A hypothesis-LB or contradiction-LB target shows the rule installed (in `overallHashMemory.originals`), the dep present (in `encodedStatements` at v=main), the head derived (in `encodedStatements` at v=main, with the right origin record citing the rule + deps) — yet `theorems.txt` is missing the corresponding theorem. (Historically `toBeProved` also never decreased; since the closure/registration decoupling — [D-278](40_decisions.md#d-278) — the goal closes anyway and only the registration is refused.) No assert. No verifier failure. No `toBeProved` residue noise.

The cross-branch surface symptom (pre-fix): 690 incube theorems vs 's 1035, with the diff concentrated in addition-by-2 uniqueness `(>[15](in3[2,N,15,4])(=[15,N]))` for N ∈ {6, 7, 9, 10, 11, 12, 13, 14} and their downstream back-reformulations and contradictions. N=2 base case proved on both branches; cascade breaks at the FIRST rung that needs a rule arriving via mail.

**Cause.** The rule arriving via `Mail::statements` carries a `std::set<int>` level set in its tuple. The receiver's `addExprToMemoryBlock(..., status=3, levels=<from mail>,...)` installs the recovered rule in `overallHashMemory` with that level set written into `LocalMemoryValue::levels`. At every firing, the derived statement's `intStatementLevelsMap` entry is `union(rule.levels, premise[i].levels)`. The registration verdict `allLevelsInvolved` at `prover.hpp::dischargeToBeProved` checks `levels.size == memoryBlock.level + 1` (primary) or `== memoryBlock.level` with no 0 (alternate). A non-empty deposited set pollutes the union with one extra integer, making `size > level + 1`, turning the verdict false. The rule fires, the head is derived, the goal closes, but the promotion to `globalTheoremList` never happens.

The specific past offender was a `for (int i = 0; i <= kySize; ++i) compactLevels.insert(i);` in `prover.cpp::proveKernel`'s D-76 deferred-compaction drain, fixed in [D-77](40_decisions.md#d-77). The retired `Mail::implications` tuple channel always used `std::set<int>` at every insert site — that was the right answer all along.

**Detect.**

1. Trap-from-within the producer LB (one rung above where the cascade visibly breaks). Compare per-burst dumps with the REF baseline.
2. If `(target_head) | v=main | levels={…}` appears in the producer LB's `statementLevelsMap` with size > `memoryBlock.level + 1`, the gate is silently swallowing the discharge.
3. Walk back to the rule whose firing produced the head (origin record in `exprOriginMap`). Check the rule's installation level set — look up its compact form `(implication<N>[])` in the LB's `statementLevelsMap`, OR check the chain entry's `LocalMemoryValue::levels` in `overallHashMemory.encodedMap`. If the set contains any integer ≥ `memoryBlock.level + 1`, the source is a non-empty mail deposit.

**Fix.** Pin `Mail::statements` level deposits to `std::set<int>` per [I-51](30_invariants.md#i-51). The receiver computes the right effective level set on its own; the mail-side levels are never additive into the derived-statement union.

**Related.** [I-51](30_invariants.md#i-51), [D-77](40_decisions.md#d-77), [D-76](40_decisions.md#d-76), [`02_glossary.md::levels`](02_glossary.md#levels), [`02_glossary.md::intStatementLevelsMap`](02_glossary.md#intstatementlevelsmap), [`20_core_concepts/02_hash_engine.md`](20_core_concepts/02_hash_engine.md), [`20_core_concepts/03_mail_system.md`](20_core_concepts/03_mail_system.md).

---

<a id="g-49"></a>
### G-49 — Contradiction LBs are extremely sensitive to mail-delivery latency


**Manifestation.** `NAME ID OVERFLOW: varId=<N> MAX=16384 …` followed by `Assertion failed: varId < ExecutionParameters::MAX_NAME_IDS, file prover.hpp` (the assert in `makeIntNormalizedKeyWithMap`). Process exits with Windows abort `0xC0000409`. The LB whose `nameMap` overflows is a `__contradiction__(=[a,b])` directly under an anchor LB; the `nameMap.idToName` tail is dominated by combinatorial `(in3[X,Y,Z,N])` expressions covering digit-by-digit substitutions across the anchor's slots × arithmetic operators (`N ∈ {4,5}` for `+` and `*`).

**Cause.** A `__contradiction__(=[a,b])` LB assumes `(=[a,b])` as fuel and propagates `a↔b` across every in-scope expression via the equivalence-class machinery. With ~9 digits and 2 arithmetic operators in scope and a few dozen existence/typing rules absorbed at recent bursts, each unchecked hashburst generates a combinatorial set of `(in3[…])` substitutions. The only pruner is a discharge premise — a raw inequality whose chain with the assumed equality reaches X∧¬X. The discharge premise is proven in a peer LB (typically `__contradiction__(=[c,d])` for some other inequality) and reaches this LB via mail. If the mail-delivery chain adds even one extra burst between the premise being proved and it firing locally, that extra burst of combinatorial substitution can exhaust the per-LB `nameMap.nextId` (`MAX_NAME_IDS = 16384`) before discharge fires.

**Detect.** Run with the hashburst dump retargeted to the overflowing LB (`hashburst_dump.cpp::isTargetLB` chain-match to `__contradiction__(=[a,b])` under its anchor) and inspect per-burst `nameMap.idToName` tail growth + `mailIn.statements` content. Compare to a baseline (= main HEAD) run to identify when the discharge premise arrives via the regular mail path. (The `rs_trap.txt` `NMHIGH` watermark dumps cited in earlier investigation notes were removed in the branch-cleanup pass; reinstate a targeted name-id watermark trap if a fresh investigation needs that signal.)

**Fix.** Remove latency from the rule-delivery chain. On the fix was to move the mail-absorption block back to its pre-fixpoint position ([D-79](40_decisions.md#d-79)) so the compact `(implication<N>[…])` arriving at burst K is absorbed at the top of burst K and the recovered rule fires in burst K's hashburst (same-burst absorb-fixpoint). The residual one PK of latency from the D-76 deferred-compaction broadcast remains; whether it stays survivable at FTA scale is open.

**Related.** [D-79](40_decisions.md#d-79), [D-78](40_decisions.md#d-78), [D-76](40_decisions.md#d-76), [I-21](30_invariants.md#i-21). See `reconstruction.md` for the per-burst diagnostic trace.

---

<a id="g-50"></a>
### G-50 — D-76's deferred compact-form broadcast adds one PK of latency at the receiver


**Manifestation.** A theorem proved by a contradiction LB at proving-PK#K reaches its descendants' `mailIn.statements` only at PK#K+1 (one PK after the proving PK). On main HEAD the equivalent rule arrived at descendants' `mailIn.implications` at the same proving-PK#K (the inline `Mail::implications` push happened during the parallel phase; smashMail consolidated at end of PK; receivers absorbed at PK#K's mailIn).

**Cause.** D-76 ([2026-05-17](40_decisions.md#d-76)) deferred the compile-and-deposit step out of the parallel proving phase into a single-threaded drain after `pool.join` (to fix an [I-28](30_invariants.md#i-28) data race on `implCounter` / `compiledExpressions` / `repetitionExclusionMap`). The drain runs inside the proving PK but emits the compact form into mailbox slots *after* the workers have finished; the next `smashMail` consolidation makes the compact form visible in receivers' `mailIn.statements` at the *following* PK. The latency is structural — as long as compact-form compilation is deferred for I-28 safety, the one-PK delay is the cost.

**Detect.** Track the per-burst arrival of `(implication<N>[…])` items in a receiver's `mailIn.statements` and the producing PK. On the contradiction LB receives `(implication46[])` at burst 4 (matching producing-PK#3); on main HEAD the equivalent implication tuple arrives at burst 3 via the deleted `mailIn.implications` channel.

**Fix (open).** None applied. Reverting D-76 reintroduces the parallel race. Inline single-threading inside `updateGlobalDirect` would serialise the parallel proving phase. A second-batch compile-and-deposit *during* the parallel phase, with a private per-thread `implCounter` and a post-join coalesce, could in principle reach same-PK delivery — not attempted; deferred to FTA-scale survivability assessment.

**Related.** [D-76](40_decisions.md#d-76), [G-49](#g-49), [I-28](30_invariants.md#i-28).

---

<a id="g-41"></a>
### G-41 — chapter row counts shift after `buildStack` lifting


**Manifestation.** After enabling the `buildStack` lifting from `D-56`, verifier check counts on a clean run differ from the pre-lifting baseline. The difference is not a regression — there are still zero failures — but the totals do not match.

**Cause.** Pre-lifting, `buildStack` emitted a chapter row for every visited `(expr, validity)` pair using `proved.validityName` verbatim, even when the origin was fetched via the main-fallback. Two recursive calls with the same expression at different deep validities each emitted their own row (the `covered` dedup keyed on the un-shifted pair). Post-lifting, both calls lift to the same `(expr, lifted_v)` and `covered` deduplicates to a single row. Chapters lose the redundant boundary-scope rows and the verifier sees a smaller, cleaner check set.

**Detect.** Verifier reports `K checks, 0 failures` where `K` is consistently smaller than the pre-lifting baseline. The "smaller" is concentrated in chapters that used to have rows at deep scopes; chapters that already lived purely at `main` are unchanged. Spot-check `files/raw_proof_graph/193_check_induction_condition.txt` — the falsified line-102-style rows should be absent or relocated to a shallower ancestor.

**Fix.** Accept the new baseline. Re-run twice to confirm determinism of the new count. Update any external baseline that referenced the pre-lifting total (CI assertions, regression-document numbers, etc.).

**Related.** [D-56](40_decisions.md#d-56), [I-39](30_invariants.md#i-39).

---

<a id="g-1"></a>
### G-1 — `gl_quick.exe` must be killed before rebuild

**Manifestation.** MSBuild fails with `LNK1104: cannot open file 'gl_quick.exe'`. The build aborts with the file still held open by a running or stuck process.

**Cause.** Windows locks the executable while any process has it open — including zombie processes from an interrupted run and sometimes even VS Code's debugger.

**Fix.**

```bash
taskkill //F //IM gl_quick.exe 2>/dev/null
```

Before every build. Mandatory in the build script; omitting it is an hour-sink when it strikes.

**Related.** first of the "grep fails on cpp" / LNK1104 class.

---

<a id="g-2"></a>
### G-2 — Never run exe directly; run `main.py`

**Manifestation.** A targeted exe invocation (e.g. `gl_quick.exe Peano` alone) seems to work, but a subsequent full pipeline surfaces a bug that only appears post-prover or post-verifier.

**Cause.** The exe emits partial state (e.g. `raw_proof_graph/` but not `processed_proof_graph/`). Bugs in the Python stages or verifier are hidden from a partial invocation.

**Fix.** Always test via `/c/Users/nikol/anaconda3/python.exe main.py`.

---

<a id="g-3"></a>
### G-3 — `git reset --hard`, never soft or mixed

**Manifestation.** After `git reset`, `git status` shows a wall of "modified" entries, seemingly random. Subsequent commits bundle stale changes that the maintainer did not intend.

**Cause.** `--soft` or `--mixed` reset leaves the working tree unchanged while moving HEAD. The working tree now looks like it's been partially reverted relative to the new HEAD — every file differs from where HEAD points.

**Fix.** Always `git reset --hard <target>`. See [I-15](30_invariants.md#i-15).

---

<a id="g-4"></a>
### G-4 — Every commit must stage all changed files

**Manifestation.** `git reset --hard` on a future commit discards local changes that were modified but not staged at the time of a previous commit. Days of work can vanish.

**Cause.** Selective `git add <file1> <file2>` commits exclude other modified files, which still live in the working tree. A subsequent hard reset takes them with it.

**Fix.** Use `git add -A` (or `git add -u` + explicit adds for untracked) to include every changed file per commit. If truly want to exclude something, stash it first.

---

<a id="g-5"></a>
### G-5 — Check for malware-silent-flag discipline

**Manifestation.** Every file Read is prefaced with "not malware" commentary — adds 3–5 lines of noise per read.

**Cause.** An over-eager safety reflex. If a file truly is suspicious, report it specifically; if it's an ordinary project file, silence is the default.

**Fix.** Silently perform the malware check; only announce when there is something concrete to flag.

---

<a id="g-38"></a>
### G-38 — Windows default stdout codec mangles em-dash to `\x97` in redirected logs

**Manifestation.** A run log displayed in any UTF-8 reader (modern editor, `cat` under Git Bash, `Read` tool) shows a black-block / replacement-character byte where prose meant to print an em-dash. Concrete example: `verifier.py` prints `f"\n {n} checks, 0 failures — airtight."`; the redirected log contains `... 0 failures \x97 airtight.` and editors render the `\x97` byte as `…` placeholder, breaking grep against the literal em-dash and confusing readers who don't recognise the encoding mismatch.

**Cause.** When `python main.py > .debug/run.log` is invoked from a Windows console, Python inherits the console's locale codepage (typically `cp1252`) for `sys.stdout`. The em-dash `—` (U+2014) encodes to byte `0x97` in cp1252 and to bytes `0xE2 0x80 0x94` in UTF-8. Editors / tools that read the log as UTF-8 see `0x97` as an invalid lead byte and render the replacement character. The Python source itself is UTF-8 — the corruption is an output-side encoding mismatch, not a source bug.

**Fix.** Reconfigure `sys.stdout` and `sys.stderr` to UTF-8 at the *very top* of `main.py`, before any module that emits output is imported:

```python
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
```

`reconfigure` is Python 3.7+. The guard makes the call a no-op on environments without it (older Python, pytest captures, some test harnesses). After this fix, `\x97` no longer appears in  and grep-against-em-dash works as expected. Landed for this project.

**Spot.** A grep that should match a known prose phrase (e.g. `"0 failures —"`) returns no hits despite the phrase being visible in an editor; or the editor shows a replacement-character / black-block where the em-dash should be. Inspect the byte stream with `hexdump` or `python -c "open('run.log','rb').read.find(b'\\x97')"` to confirm.

---

## Debugging gotchas

<a id="g-6"></a>
### G-6 — Always trap-debug, never reason through code alone

**Manifestation.** An agent is asked "why does X fail?" and responds with a hypothesis derived by reading the code. The hypothesis is wrong. An hour of further reading deepens the wrong hypothesis. Eventually, adding a single `std::cout` reveals a state the agent never considered.

**Cause.** Reasoning through code without live state is unreliable in a system as layered as GL. The prover's state is governed by timing (which cycle), scope (which validityName), and admission order (which levels) — all of which are invisible in source.

**Fix.** Default to instrumenting with prints/dumps. Do not change logic until the dump confirms the hypothesis.— flagged as *most important* across the memory files.

---

<a id="g-7"></a>
### G-7 — Filter debug output to a single LB first

**Manifestation.** A prover trap fires across hundreds of LBs per cycle; the log is unreadable.

**Cause.** Default print-every-LB-every-step. No way to pick out the interesting LB by eye.

**Fix.** Add an explicit guard:

```cpp
if (body.exprKey == "(my specific expression)") {
    std::cout << "...";
}
```

Redirect to a file (`> .debug/my_trap.log 2>&1`), then Read the file.

---

<a id="g-8"></a>
### G-8 — LB identification needs full parent chain, not just `exprKey`

**Manifestation.** Debug trap fires on the "wrong" LB — same `exprKey` as the intended target, but a different role (e.g. a compressor Phase 1 LB with the same key string as a main-pipeline target LB).

**Cause.** `exprKey` is not unique across LBs. A child LB in the main tree can share `exprKey` with a fresh LB in a compressor pool. Identification by `exprKey` alone matches both.

**Fix.** Match the full parent chain (walk `parentMemory` up to root) + `exprKey` equality per generation.

---

<a id="g-9"></a>
### G-9 — Trap layered strategy for hash-request generation

**Manifestation.** A conjecture that should fire a specific rule in a specific LB never seems to produce the emission. Logging every hash-request generation dumps gigabytes.

**Cause.** Hash-request generation is the hottest path. Filtering by coincidence (expression contains "X") misses it if the encoder substitutes names.

**Fix.** Five-batch layered trap strategy:

1. Trap at the request-build site — log when the builder considers the target.
2. Trap at the lookup site — log when the key is queried.
3. Trap at the fire site — log when a hit fires.
4. Correlate via `mbPtr` (the `Memory*`) — not `exprKey`.
5. Resolve `nameId` vs `originalId` — names translate through `NameMap`; a filter on `originalId` catches renamings.

---

<a id="g-10"></a>
### G-10 — Compare chapter-vs-originmap for divergence

**Manifestation.** Two runs of the same pipeline produce different results for a specific chapter, with the inputs byte-identical.

**Cause.** Some state carries over between runs unexpectedly (stale in-memory caches, file-system ordering, simple-facts table order). The chapter's generated content differs, but the origin map's content differs subtly differently.

**Fix.** Run chapter-vs-originmap bottom-up walk to localise first divergence. Companion Python script in `.debug/`. Start with the chapter rows; for each row, check that its citations exist in the origin map with the expected shape. First row where they diverge is the target.

---

## Code-pattern gotchas

<a id="g-11"></a>
### G-11 — Precompile structural operators before every theorem-load

**Manifestation.** An assert fires in disintegration with an exprKey visibly containing `!(&` or `!(>` at string level. Or: silent CE-filter divergence — the same theorem admitted to one LB matches its hash target, to another LB does not.

**Cause.** Raw `!(&...)` / `!(>...)` expressions reach `addTheoremToMemory` / `disintegrateImplication` / `addToHashMemory` without being rewritten into compiled `or<N>` / `existence<N>` names. The hash engine keys on string form, so two LBs with the same semantic content but one carrying raw and another carrying compiled names produce divergent matches.

**Fix.** Every theorem-load path must call `precompileStructuralOperators(thm)` first. See [I-1](30_invariants.md#i-1).

**Historical.** User flagged this as a recurrent bug — has fired multiple times.

---

<a id="g-42"></a>
### G-42 — `repetitionExclusionMap` key must include category, not just elements

**Manifestation.** Under cross-batch sharing of spontaneous compact operators (approach A — every batch including incubator merges into `GL_binary_shared.json`), main-batch theorems that depend on existence-introduction or or-disintegration silently stop proving. Concretely: on a clean run after the prior agent dropped the `tag.startswith("Incubator")` early-return in `_merge_into_shared`, **Gauss + FTA-rung-1 theorems vanished** while the verifier still reported zero failures on the surviving proofs.

**Cause.** `repetitionExclusionMap` in `prover.hpp` was keyed by elements vector only (`std::vector<std::string>`). The lookup in `excludeRepetitions` is category-blind: it ignores the `category` parameter and matches on splitNK alone. Cross-category elements collisions exist between incubator and main allocations — specifically, IncubatorPeano's `implication23` and Peano's `existence2` share elements `[(in[1,u_1]), (in2[1,u_2,u_3])]`. Under approach A, IncubatorPeano's `implication23` lands in `shared.json` first; Peano boots, `loadGlBinary` registers it in `repetitionExclusionMap`; Peano's compiler computes splitNK for `∃x∈N: in2[x,b,c]` and the lookup HITS — returns `implication23`. The body is now treated as an implication-category operator. Disintegration dispatches by `compiledExpressions["implication23"].category == "implication"` → applies implication-disintegration to an existence-shape body. Every proof downstream that needed existence introduction (`or0` chain, predecessor reasoning, rung-1) fails.

**Detect.**
- Run `python verifier.py` on a clean post-`main.py` state. The verifier's `operator registry consistency` check is the closest existing detector — it flags name collisions but does NOT flag same-name-same-category-different-elements vs same-elements-different-category. The check that actually catches the cross-category-elements case requires walking the elements space across all loaded binaries; build it ad-hoc with the snippet at the bottom of.
- The visible symptom is **theorem loss** in `files/theorems/theorems.txt` vs a known-good baseline. Per that's the only regression authority — verifier check counts alone do not catch this.

**Fix.** Make the map key `(elements, category)`:

```cpp
std::map<std::pair<std::vector<std::string>, std::string>,
         std::tuple<...>> repetitionExclusionMap;

// lookup:    repetitionExclusionMap.find(std::make_pair(splitNK, category));
// insertion: repetitionExclusionMap[std::make_pair(splitNK, category)] = ...;
```

Touches one declaration + two access sites in `prover.hpp` (lookup + insertion in `excludeRepetitions`) and one access site in `visualizer.cpp::loadGlBinary`. Within a single batch this is a no-op — there are no within-batch cross-category elements collisions in current binaries. Across batches it makes shared-registry loading safe.

**Why this is a gotcha rather than just an invariant.** The bug is dormant until two independent allocators contribute to the same `loadGlBinary` source. Prior to D-54's reversal, no codepath surfaced it — the verifier had no I-23 check, incubator allocations were segregated, and within-batch collisions don't occur. The forcing function was approach A. A future change that introduces another allocator (e.g. a second main-style batch type) or relaxes within-batch ordering could re-surface it.

**Related.** [I-23](30_invariants.md#i-23), [D-60](40_decisions.md#d-60).

---

<a id="g-45"></a>
### G-45 — implication compilation must be injective; a bug breaks it

> **🔴 CONFIRMED ROOT CAUSE (2026-05-17, follow-up session) — DATA RACE FROM PARALLEL-PHASE INVOCATION, superseding the sub-key-conflation "Cause" below (kept un-deleted per the annotate-don't-erase convention).** The non-injectivity + run-to-run drift is **not** a single-threaded key flaw. `compileImplicationToCompact` is called from the parallel proof phase: `addExprToMemoryBlock(... int coreId...)` (per-worker) → `updateGlobalDirect(theorem, coreId)` (`prover.cpp`) → `compileCoreExpressionMapCore` / `excludeRepetitions`, all mutating shared `implCounter` (plain `int`) + `compiledExpressions` + `repetitionExclusionMap` (`std::map`) with no lock. Proven by torn/fused lines in the saved compile-trace (concurrent `ofstream` writes) and clobbered shared `keyNeg` values (811, 1830). Two fresh runs: `theorems.txt` identical (42), distinct `implication<N>` 1359 vs 1361, verifier 10120 vs 10122; 16 distinct-input→one-name collisions within a single run. Violates [I-28](30_invariants.md#i-28). The negation-overlooked hypothesis is disproved (`makeNormalizedEncodedKey` emits a correct per-constituent negation bit; 0 pure-negation-drop collisions). Detect: torn lines / impossible `keyNeg` in a per-call compaction trace; or run twice and diff distinct `implication<N>` on an identical `theorems.txt`. **FIX APPLIED (Option A):** the broadcast sites enqueue via `recordPendingCompaction`; a single-threaded post-`pool.join` sorted drain performs the compile + deposit (I-28-compliant) — deterministic and injective. Diagnostic trap removed; `compileimpltocompact_negation_is_injective` unit guard added. **VERIFIED:** two fresh fixed-build `main.py` runs byte-identical — `theorems.txt` (42, no proof loss), `GL_binary_shared.json`, `hashburst_trace.txt`, verifier counts (1397 / 10125 / 109704, 0 failures); distinct `implication<N>` stable at 1364 (was non-deterministic 1359 / 1361). See the FIX APPLIED note in [D-76](40_decisions.md#d-76).

**Manifestation.** On, feeding *every* mail implication through `compileImplicationToCompact` → `compileCoreExpressionMapCore` → `excludeRepetitions` produced (a) a non-deterministic compiled-implication registry — same proven-theorem set (incubator 1035/183/2, main 42 identical across two runs) but 1370 vs 1371 `compiledExpressions`, 7 differing `(implication<N>[])` names, divergent proof-graph chapters and verifier counts (`operator registry consistency` 1391 vs 1392) — and (b) a ~22 GB incubator OOM/freeze from combinatorial registry growth.

**Cause.** Implication compilation is *designed* to be **injective**: a structural canonicalization mapping each distinct implication to a stable `implication<N>` (and structurally-identical implications to the *same* name — correct dedup). `excludeRepetitions` implements that canonical identity via `giveAllSortedCombinations(encodedTail)` × permutations into `repetitionExclusionMap[(splitNK, category)]`, with cache reuse (caller rolls back `implCounter--` on hit) intended to make the key permutation-invariant. **A bug breaks that intended injectivity**: on an identical proven-theorem set the compiled-name set/count differs run-to-run (1370 vs 1371, 7 names), so the compact form is not, in practice, the pure deterministic function of structure it is designed to be — distinct implications can collapse to one name (or the count varies) depending on invocation order. The exact injectivity-breaking defect is **not yet isolated** (branch held); candidate surfaces: the `makeNormalizedEncodedKey` / `splitNK` canonical-key computation, the combo/permutation registration, or order-dependent state leaking into the key or the wrapper's inputs (`stripUPrefixAST` / upstream broadcast form). Same *class* as [G-42](#g-42)/[D-60](40_decisions.md#d-60) (compact-name identity) — but there the keying was wrong by construction; here it is *meant* to be sound and a defect breaks it. A shared name makes a `compilation` row / GL-binary expansion wrong for one original, so this is a soundness bug, not mere flakiness.

**Detect.** Two full `main.py` runs; diff `compiledExpressions (N)` count and the `implication<N> | sig=(implication<N>[])` set in , and the verifier `operator registry consistency` count. A run-to-run delta on an identical `theorems.txt` is the signature.

**Fix.** Open — held for user architectural direction (the project conventions banner / Rule 8). The task is to **find and fix the bug that breaks the intended injective compilation** so the same implication always yields the same `implication<N>` regardless of invocation order. A determinism-only patch (content-address the name from the key) is **wrong** — it would mask the injectivity bug, not restore correct injective compilation. Open sub-question: whether the defect is latent in the pre-existing definition-subexpression path or only manifests at full-theorem mail scale.

**Related.** [D-76](40_decisions.md#d-76), [G-42](#g-42), [D-60](40_decisions.md#d-60), [I-23](30_invariants.md#i-23).

---

<a id="g-12"></a>
### G-12 — NameMap decode returns a reference — copy before nested mint

**Manifestation.** Mysterious string corruption in a scope-name or payload, during a code path that involves nested encoding. Intermittent crashes not reproducible every run.

**Cause.** `NameMap::decode` and `idToSub[id]` return references into `std::vector<std::string>`. Any nested call that could trigger a `push_back` may reallocate the vector, dangling the reference.

**Fix.**

```cpp
std::string payload = nameMap.decode(scopeId);  // COPY — local variable
doSomethingThatMayMintAnotherScope();             // fine
useStringBasedOn(payload);                        // safe
```

See [I-3](30_invariants.md#i-3).

---

<a id="g-13"></a>
### G-13 — `ChunkPool` must use static `char[]`, never malloc

**Manifestation.** A refactor replaces a static buffer with `std::vector<char>` or `new char[...]`. Hot-path timings develop tail-latency spikes.

**Cause.** Hot-path allocations contend with mimalloc's caches. The `ChunkPool` pattern pre-allocates a static buffer and hands out chunks with zero heap interaction after startup; replacing with dynamic allocation re-introduces the contention.

**Fix.** Always a `static char[]` of fixed size. User has requested this pattern four times — a chronic regression-risk. See [I-13](30_invariants.md#i-13).

---

<a id="g-14"></a>
### G-14 — Don't disable mechanisms when asked to filter them

**Manifestation.** User asks "filter out this specific case in Pass B". Agent disables all of Pass B.

**Cause.** A "filter X" request is asking for a targeted guard, not for removal of the surrounding mechanism. Disabling the whole mechanism produces broad behaviour change that was never requested.

**Fix.** Read the request literally. If "filter X" — add a guard that filters X specifically, leaving the rest of the path unchanged.

---

<a id="g-15"></a>
### G-15 — Don't change what wasn't asked

**Manifestation.** A diff touches ten files when only two were in scope. Unrelated lines are "cleaned up" alongside the intended change.

**Cause.** Agents default to making improvements they notice. This complicates review, hides the real change, and can introduce regressions from the incidental edits.

**Fix.** Change only what is literally asked. If a collateral improvement is clearly warranted, open it as a separate commit with its own message.

---

<a id="g-16"></a>
### G-16 — Pass B single-input gate is empirical — don't widen

**Manifestation.** Someone refactors `isAllowedAsOperatorInput` to handle multi-input operators. Next Gauss run is 10× slower. Memory pressure climbs.

**Cause.** Widening admission breaks the RT-explosion guardrail. The single-input check is empirical — it was broken and reverted. See [I-6](30_invariants.md#i-6), [D-9](40_decisions.md#d-9-pass-b-single-input-operator-gate-empirical-commit-463f402-reverted).

**Fix.** Leave the `inputIndices.size == 1` check alone.

---

<a id="g-17"></a>
### G-17 — MPL has no spaces

**Manifestation.** A test expression like `(in2[a, b, c])` fails to parse. Debug output mentions an unknown name `a_with_trailing_space`.

**Cause.** MPL uses bare commas. The parser treats whitespace as part of the next token.

**Fix.** Never insert spaces in MPL expressions. See [Glossary — MPL](02_glossary.md#mpl).

---

<a id="g-18"></a>
### G-18 — Grep can fail on C++ files due to non-ASCII header

**Manifestation.** A `grep` against a source file returns empty, but the pattern is definitely there. Visual inspection confirms.

**Cause.** Every C++ file in `GL_Quick_VS/GL_Quick/src/` carries the AGPL+commercial header with `haftungsbeschränkt` (non-ASCII `ä`). Ripgrep occasionally chokes on the encoding in specific configurations.

**Fix.** Fall back to `Read` with line offsets when grep unexpectedly empty.

---

## Domain-specific gotchas

<a id="g-19"></a>
### G-19 — `incubator_mode` vs `ban_disintegration` — not the same

**Manifestation.** A refactor substitutes `ban_disintegration` where `incubator_mode` was checked. Incubator runs start disintegrating when they shouldn't, emitting tag rows the incubator-mode verifier isn't expecting.

**Cause.** The two flags have overlapping effect (both disable Pass B in certain paths) but are set from different places. Incubator config sets `incubator_mode = true`; compressor sets `ban_disintegration = true` during Phase 1.

**Fix.** Use the right flag for the right intent. See [I-7](30_invariants.md#i-7), [D-10](40_decisions.md#d-10-incubator_mode-vs-ban_disintegration--separate-flags-per-claudemd).

---

<a id="g-20"></a>
### G-20 — Trivial equality forbidden in head only

**Manifestation.** Conjecturer emits `(=[v1,v1])` heads. CE filter passes them (`v=v` is trivially true). Prover "proves" them in zero steps. Global theorem list gets bloated with tautologies.

**Cause.** The `controlEquality` filter's head guard drops trivial equality, but a regression in the filter implementation admits them.

**Fix.** Preserve the guard — reject any proposed head where both arg-positions of `(=[...])` resolve to the same variable. See [I-8](30_invariants.md#i-8).

---

<a id="g-21"></a>
### G-21 — Induction on untyped variables is silently unsound

**Manifestation.** A Peano `theorems.txt` entry is semantically false — e.g. a universally-quantified claim that should fail for entities not in `N` but is marked proved by induction.

**Cause.** The historical induction scheduler did not check that the induction variable is in `N`. For bound variables that only appear in negations, bare equalities, or existence heads, the typing is not derivable from the chain — the theorem's effective universe widens from `N` to "everything".

**Fix.** Implement the induction-typing sub-theorem per [`docs/agentic_swdd/induction_typing_plan.md`](induction_typing_plan.md). See [I-18](30_invariants.md#i-18), [D-15](40_decisions.md#d-15-induction-typing-sub-theorem--design-approved-2026-04--pre-c21386e). In progress.

---

<a id="g-22"></a>
### G-22 — CE filter j-copy requirement

**Manifestation.** A new fact file is written by hand, using only `i`-prefixed constants (no `j`-copies). CE filtering mysteriously underfires — some conjectures that should be refuted slip through.

**Cause.** The request generator at stump length 0 (`generateEncodedRequestsStatic`) needs *distinct* fact entries for its rule-firing pattern. Without j-copies (parallel constants), rules that require two different-looking arguments to match never fire on the table.

**Fix.** Use the j-copy pattern: every constant appears as `i<k>` and `j<k>`. `incubator_to_simple_facts.py` implements this automatically when regenerating facts.

---

<a id="g-23"></a>
### G-23 — Chapter v-numbering drift

**Manifestation.** Verifier fails `task formulation` or `theorem` checks with "expected v1, got v3" messages.

**Cause.** The chapter's `repl_map` assigned `v1` to a variable that appears first in chapter lines, but the global theorem list's `v1` refers to a different variable (assigned by the theorem's left-to-right scan).

**Fix.** Chapter v-numbering must be seeded from the theorem expression's left-to-right scan (Priority 2 of [`process_proof_graphs.py`](10_pipeline/06_process_proof_graph.md) renaming). See [I-10](30_invariants.md#i-10).

---

<a id="g-24"></a>
### G-24 — Pipeline cross-contamination (incubator ⟷ main)

**Manifestation.** A Gauss incubator config change affects Peano CE behaviour.

**Cause.** Unclear. Suspected shared state via the simple-facts fact-table loader or a configuration side-effect. Not yet root-caused.

**Status.** Open.

---

## Verifier gotchas

<a id="g-25"></a>
### G-25 — Verifier failures are real bugs

**Manifestation.** An agent looks at a verifier failure, traces to `verifier.py`, and proposes loosening the check. The user rejects.

**Cause.** Misreading the failure as "verifier is wrong" when the verifier is the authoritative oracle. Weakening a check hides the real bug.

**Fix.** Always treat a verifier failure as a bug in the proof-graph producer (prover / processor / compiler). Never modify `verifier.py` without explicit consent. See [I-16](30_invariants.md#i-16).

---

<a id="g-26"></a>
### G-26 — `theorems.txt` is authoritative, not `global_theorem_list.txt`

**Manifestation.** Two runs produce different `global_theorem_list.txt` — method labels shift (`direct` vs `induction`) or column-3 references differ. Panic ensues; "regressions".

**Cause.** `global_theorem_list.txt` is encoding-sensitive to how the processor renames theorems. Method-label shifts are not theorem losses — they reflect processor's interpretation, which can drift.

**Fix.** The regression-claim source of truth is `files/theorems/theorems.txt` — the set of theorems the prover actually produced. Use `diff` on this file to detect real losses.

---

<a id="g-37"></a>
### G-37 — Incubator verifier needs `--include-globals` for cross-batch theorem refs

**Manifestation.** `python verifier.py files/incubator/processed_proof_graph` reports an `origin failure 1` (and possibly `implication` failures whose `rest[0]` rules can't be resolved). The proof graph itself is sound; the verifier just can't find the cited rule in the loaded registry.

**Cause.** Incubator chapters legitimately cite rules from sibling batches (a rung-1 chapter referencing a Peano-batch axiom such as `existence2`). `run_verifier(base_dir)` loads only `base_dir/global_theorem_list.txt` plus `base_dir/external_theorems.txt`. The Peano rule lives in `files/processed_proof_graph/global_theorem_list.txt` — outside `base_dir`. The inline origin check at the bottom of `verify_chapter` doesn't know about it and rejects.

**Fix.** Pass the sibling list explicitly: `python verifier.py files/incubator/processed_proof_graph --include-globals files/processed_proof_graph/global_theorem_list.txt`. The flag is repeatable for multi-batch unions. Added by [D-35](40_decisions.md#d-35).

---

<a id="g-38"></a>
### G-38 — `_normalize_expr_list` only renames `v\d+`; cross-batch comparisons need `_alpha_canonicalize_bound_vars`

**Manifestation.** Even with `--include-globals` in place (G-37), an `origin` check still rejects a row whose `rest[0]` rule is "obviously" in the global list. Manual inspection shows the chapter's rule names a bound variable `i2` while the global list stores the same rule as `v1`.

**Cause.** `_normalize_expr_list` (verifier.py) renames only `v\d+` patterns. The chapter's rule was emitted with the prover's local free-index counter (`i2`), the global-list version was rewritten by `process_proof_graphs.py`'s canonical-export rename (`v1`). Both forms describe the same alpha-equivalent rule; `_normalize_expr_list` treats them as distinct strings.

**Fix.** Use `_alpha_canonicalize_bound_vars` (verifier.py, added by [D-35](40_decisions.md#d-35)) for cross-batch comparisons. It walks every `>[…]` binder and renames each bound name to `b1, b2, …` in declaration order. The inline origin check now compares both via `_normalize_expr_list` (existing semantics) and via `_alpha_canonicalize_bound_vars` (cross-batch alpha-equivalence).

**Don't.** Do not extend `_normalize_expr_list` itself to cover `i\d+` patterns — `i0` and `i1` are anchor slot names (free, not bound) in many expressions; renaming them would corrupt anchor citations. The bound-var-aware walker is the correct primitive.

---

<a id="g-39"></a>
### G-39 — `current_gl_binary` silently goes to None for `AnchorIncubator` chapters

**Manifestation.** A reformulation check or other GL-binary lookup intermittently relies on `state.current_gl_binary` and gets None when the chapter is from the incubator. `binaries_for_chapter` falls back to all loaded binaries, which works for most queries but masks the underlying issue.

**Cause.** `verify_chapter` sets `state.current_gl_binary` by literal substring match: `f'Anchor{tag}' in thm_expr` for each loaded `tag`. Loaded incubator binaries are tagged `IncubatorPeano` / `IncubatorGauss` / `IncubatorGauss1`, but the chapter's anchor is `AnchorIncubator` (without the `Peano` / `Gauss` / `Gauss1` suffix). `'AnchorIncubatorPeano' in 'AnchorIncubator…'` is False; no tag matches, and `current_gl_binary` stays None.

**Fix (current).** `binaries_for_chapter` returns all loaded binaries when `current_gl_binary` is None, so most lookups succeed via the fallback. `_check_reformulation` was updated by [D-35](40_decisions.md#d-35) to also fall back across all binaries when its anchor-derived tag has no entry. New checkers that rely on `state.current_gl_binary` directly should use `state.binaries_for_chapter` instead.

**Cleaner future fix.** Either (a) load a `GL_binary_Incubator.json` that unions the three suffix-named binaries at load time, or (b) widen the substring match to a prefix match (`anchor.startswith(f'Anchor{tag}')` reversed: `tag in chapter_anchor_suffix`). Not done yet.

---

## Multi-agent gotchas

<a id="g-27"></a>
### G-27 — The tested state must equal the committed state

**Manifestation.** Agent tests a fix locally, sees it works, reverts the fix to "leave the file clean", then commits with a message claiming validation. Future reader pulls the branch, re-tests, doesn't see the fix — because it was reverted.

**Cause.** Conflating "I want the diff minimal" with "the state I tested".

**Fix.** Never revert a verification-enabling edit before committing. The commit message's validation claim is fiction otherwise.

---

<a id="g-28"></a>
### G-28 — After a narrow task, stop

**Manifestation.** Agent finishes a bug fix, then volunteers additional cherry-picks, analyses, or related refactors the user didn't ask for.

**Cause.** Helpfulness reflex overrides scope discipline.

**Fix.** After "done + committed + pushed" on a narrow task, stop. Wait for the next task.

---

## Structural gotchas

<a id="g-44"></a>
### G-44 — `reformulateTheorem`'s peeling-layer trigger must not read a reconstructed binder

**Manifestation.** After the implication binder was unified ([D-75](40_decisions.md#d-75)), `reformulated statement` rows silently vanish from `global_theorem_list.txt` and the proved-theorem stream — a whole-row delta, not a `>[...]`-only delta — even though the change was supposed to be representation-only.

**Cause.** `reformulateTheorem` ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)) decides whether a Definition is the reformulation "peeling layer" by reconstructing the theorem, re-disintegrating it, and testing the *reconstructed* last-link `>[...]` cardinality: `boundVars.size == 1 && boundVars[0] == expectedArg`. That test only worked because the old sparse binder bound a variable iff it occurred ≥2×, so the last link's binder happened to be exactly `{set-argument}`. Under the unified rule the last link binds every non-`u_` variable, so `boundVars.size` is almost always > 1 and the trigger stops firing. The binder was being used as **control flow**, not representation — the one place in the codebase where that was true.

**Detection.** Diff a post-change `main.py` log against the pre-change reference; if any `reformulated statement` row is absent (rather than merely differing inside `>[...]`), this trigger is the cause.

**Fix.** Compute the peeling-layer predicate directly from the original chain instead of from the reconstructed binder: the target Definition's set-argument is the single non-`u_` arg that occurs ≥2× across (all premises + head) and appears in no other premise (so its first occurrence in `[others…, targetDef, head]` order was the targetDef link). This reproduces the old predicate exactly, so the same reformulations fire; the inner negated-existence still binds exactly the peeled set-argument (byte-identical); only the emitted theorem's outer `>[...]` widen, which is the intended DoD-covered change.

**Don't.** Do not "fix" by carving `reformulateTheorem` out of the unification (keeping a sparse binder there) — the user directed it be reworked, not exempted. Do not accept the vanished rows as DoD-covered: `>[...]`-only deltas are covered, missing rows are not. Any future control-flow that reads a *reconstructed* binder's shape is suspect for the same reason — derive the intent from the chain.

**Code.** Peeling-layer predicate at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) (`reformulateTheorem`). See [I-4](30_invariants.md#i-4), [D-75](40_decisions.md#d-75).

---

<a id="g-29"></a>
### G-29 — Hashmem dumps should show template rows, not instantiations

**Manifestation.** Asked to report on hashmem contents, agent dumps every entry — hundreds of `int_lev_*` / `repl_lev_*` rows that are instantiations of the same underlying u_-template. Unreadable.

**Cause.** Default dump is exhaustive.

**Fix.** When reporting hashmem contents, show only the `u_`-prefixed template rows. Hide the instantiations unless explicitly asked.

---

<a id="g-30"></a>
### G-30 — Anchor-change sizing tension

**Manifestation.** Expanding `AnchorIncubator` with more slots (e.g. going from 14 to 16 args) — conjecturer runs out of memory during `createMapAnchor`.

**Cause.** `createMapAnchor` materialises permutation tables whose size grows multiplicatively with typed-arg count + `max_values_*` config. Per the project conventions, `right > 3` causes RAM explosion.

**Fix.** Keep `max_values_for_uncomb_def_sets + max_values_for_def_sets ≤ 3`.

---

<a id="g-31"></a>
### G-31 — Adding `cleanAdmissionMap` to integration revival

> **Retired on this branch by [D-299](40_decisions.md#d-299) (2026-08-22).** `cleanAdmissionMap` is deleted, so there is nothing left to symmetrize onto the integration side. The standing prohibition — no consumption-driven erase of an admission entry, on either side — moved to [I-200](30_invariants.md#i-200). The rest of this entry describes the retired failure mode.

**Manifestation.** A second revival of the same marker-template fails because the admission entry was silently removed by a previous revival. Verifier still passes (each derivation sound); some previously-provable theorems stop proving.

**Cause.** Someone "symmetrized" the integration revival to match algebra's `revisitRejected2`, which calls `cleanAdmissionMap` after admission use. Integration intentionally does NOT. The algebra revisit can afford cleanup because its `addExprToMemoryBlock` path re-populates admission maps via downstream integration-prep; integration revival is mailIn-only — once the template is gone, nothing re-creates it for a future matching rejection.

**Fix.** Do not call `cleanAdmissionMap` in `revisitRejectedIntegration2` or in the match-branch of `applyEquivalenceClassToRejectedMapIntegration`. See [I-22](30_invariants.md#i-22). The `rejectedMapIntegration` entry itself IS erased after revival; the admission-map entry stays live.

---

<a id="g-33"></a>
### G-33 — `applyEquivalenceClass` emits malformed `equality1` origin on re-derivation

**Manifestation.** Verifier reports `equality1 — success N, failure K` with K ≥ 1, and failing rows have `rest=[source_expr, source_ns]` (length 2) — zero equalities appended. `check_equality1` at `verifier.py` rejects for `len(rest) < 4`.

**Cause.** In `applyEquivalenceClass` at `prover.hpp`, `exprOriginMapLocal[newExpr]` was populated only in non-compressor mode when `newExpr` was NOT already in `memoryBlock.exprOriginMap`:

```cpp
if (parameters.trackHistory) {
    if (parameters.compressor_mode || memoryBlock.exprOriginMap.find(newExprEnc) == end()) {
        exprOriginMapLocal[newExpr] = eqs;   // eqs = setEqualities
    }
}
```

The downstream origin emission at `prover.hpp` always fires (its own "if not-in-origin-map" guard is commented out to allow multi-origin accumulation). On subsequent class passes — ancestor-NS loop at `prover.hpp`, fixpoint re-iter at `:6107` — that re-derive the same `newExpr`, the population step is skipped but the emission still pushes `expr2.original` as `rest[0]` and then finds `exprOriginMapLocal[newExpr]` absent → zero equalities appended → malformed `rest=[source, source_ns]`.

Latent since the multi-origin refactor. Usually hidden because most test theorems don't re-derive the same expression across class passes. Exposed by the branch's conjecturer reshuffle (pin-anchor + contiguous-arg normalization) which generates expression shapes that trigger the re-derivation path.

**Fix.** Remove the gate. Always populate `exprOriginMapLocal` in `trackHistory` mode. Landed on `rt_conjecturer2` in the follow-up commit to the `rejectedMapIntegration` merge.

**Spot.** Any `equality1 failure ≥ 1` in verifier output on a branch where it was airtight before. Confirm via monkey-patch trace (see [debug trap patterns](10_pipeline/04_prover.md#debug-trap-patterns-for-pass-b-admission-revival-paths)) — failing rows print `rest=['(…)', 'main']` (length 2).

---

<a id="g-32"></a>
### G-32 — `rejectedMapIntegration` key shape: bare concrete, not u_-prefixed

**Manifestation.** `revisitRejectedIntegration2` is called on new admission-map inserts but always reports `found=no`. Or `applyEquivalenceClassToRejectedMapIntegration` processes entries but admission probes never match. Revival mechanism fires but nothing gets revived.

**Cause.** Three maps with three different key shapes:

| Map | Key shape | Where |
|---|---|---|
| `admissionMapIntegration` | `u_`-prefixed on all non-marker args | `prover.hpp` — built from `le.signature` |
| `admissionSetIntegration` | bare concrete, `repl_` preserved | `prover.cpp` — `removeUPrefixFromArguments(mappedElement)` |
| `rejectedMapIntegration` | bare concrete (matches Pass B's `markedExpr`) | `prover.cpp` — `makeMarkedExpr(removedU, var)` |

A lookup using the wrong form silently misses. `isAdmittedIntegration` at `prover.hpp` bridges admission-map ↔ Pass-B keys by u_-prefixing the bare marker form on lookup — the integration revival path must do the same transform.

**Fix.** When revisiting from an admission insert at `prover.hpp`, STRIP `u_` off `keyString` via `removeUPrefixFromArguments` before looking up `rejectedMapIntegration`. When probing admission from `applyEquivalenceClassToRejectedMapIntegration`'s rewritten key, ADD `u_` to all non-marker args before looking up `admissionMapIntegration` (but look up `admissionSetIntegration` with the bare form). See the admission-probe section inside `applyEquivalenceClassToRejectedMapIntegration` and the revisit call site at `prover.hpp`.

---

<a id="g-34"></a>
### G-34 — `disintegrateImplication` thread_local cache staleness

**Manifestation.** A refactor changes the chain's shape (e.g. you want a fourth tuple element, or change `std::get<1>` from `vector<string>` to `vector<int>`). The cache stores the OLD shape. On a hit, callers receive the old shape and either crash or silently misbehave. Symptoms: byte-identity regression in `reshuffled_conjectures.txt`, random segfaults at tuple access sites, or filters that pass when they shouldn't.

**Cause.** `compiler.hpp::disintegrateImplication` has a single-slot `thread_local` cache keyed by the input expression string. The cache's stored chain has a fixed type (`ChainT = std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>>`). If the function's output shape changes, the `thread_local` storage is initialized with the OLD type at thread start and the compiler will not catch the divergence because the cached values are memcpy-compatible until you `std::get<>` them.

Equally: if the thread_local is initialized while the key field is mutable, re-entry during initialization can use half-populated cache state (shouldn't happen with the current code — initialization is bound to the thread's first call — but a future refactor that adds re-entrant calls to the cached function would trip this).

**Detection.** Run the byte-identity harness (diff the three output files against ) on every commit that touches `disintegrateImplication`, `parseExpr`, or any filter that consumes its output shape.

**Fix.** When changing the chain's output shape, also change the cache's `ChainT` typedef in the same commit and add a build-level static_assert if the shape is non-trivial (e.g. `static_assert(std::tuple_size<ChainT::value_type>::value == 3,...)`). When adding a new field, prefer a new typed member over changing the tuple — the cache is less fragile against additive changes.

**Related.** See [D-20](40_decisions.md#d-20) for the cache's origin rationale. The cache will be factored out as part of the int-story campaign's follow-up port (once `disintegrateImplication` itself moves to an allocation-free iterative walker, the cache may become redundant).

---

<a id="g-35"></a>
### G-35 — Inline `std::atomic` in compiler.hpp — ODR shelter

**Manifestation.** Two TUs include compiler.hpp, one version of the atomic counter updates, the other's reads always return zero. The profiling report shows obviously-wrong numbers (e.g. `disintegrate calls=0` despite observable filter activity).

**Cause.** Plain `std::atomic<uint64_t> g_x{0};` at file scope in a header violates ODR when the header is included in multiple TUs. C++17 `inline` variables get a single shared definition across all TUs, but the declaration MUST be `inline`, not plain.

**Detection.** Report numbers that look "off by a factor" (one TU's activity missing). Unit tests that simulate multi-TU inclusion (add a second consumer, verify counter increments are visible from both).

**Fix.** Always `inline std::atomic<T> g_x{0};` in headers. Confirmed (`compiler.hpp`'s `g_disintCalls` / `g_disintNs` / `g_disintCacheHits`). Temporary profiling globals should be obvious when removing them — `#ifdef GL_PROFILING`-gating is an option but has not been adopted yet in this project.

---

<a id="g-37"></a>
### G-37 — `multiplyImplication` must skip partitions that merge two distinct free `u_*` parameters

**Manifestation.** A `multiplied from` chapter row in which the source and copy implications differ in a single argument slot of one expression — the differing slot's source value is one anchor element (e.g. `u_6` / displayed `1`), the copy value is a *different* anchor element (e.g. `u_2` / displayed `0`). Downstream chapters cite the forged copy as a hash rule and "prove" theorems that are mathematically false. Concrete incident: chapter-1115 of the IncubatorGauss proof graph derived `!fold[N,s,+,id,0,0,0]` (HTML title *"sum(i=0..0) ≠ 0"*) by feeding the bad copy into a contradiction step.

**Cause.** `multiplyImplication` ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)) collects every `(1)`-typed argument across the rule body — both bound and free `u_*` — into `oneTypedVars` and enumerates Bell partitions over that set. The substitution loop picks each equivalence class's representative with `u_*` preference, then replaces every other class member with the representative throughout the rule string. Bound→bound and bound→free merges are sound (Bell partition / specialisation respectively), but a partition class containing two distinct free `u_*` rewrites a free slot of the rule body and emits a logically stronger rule than the source — provable conclusions then include theorems the source rule never authorised.

The original code skipped such partitions via `if (hasDoubleU) continue;`. Commit cf358271 (Session 2026-03-25, *"u_ equalization, remainingArgs recompute, integration instruction pass-through"*) deleted the skip, leaving the `hasDoubleU` flag computed and discarded with a trailing comment *"u_ equalization allowed in general (no double-u skip)"*. The unsoundness sat dormant on Peano and Gauss verifier-clean — those proof graphs do not exercise the partition shape — until the FTA-ladder branch (sandbox/or_6) hit incubator chapters where two free anchors land in the same `(1)`-typed positions of `existence2` / `interval`.

**Detection.** A `multiplied from` chapter whose `data-text` for source and copy differ only in slots that are filled with anchor elements (e.g. `i0` / `i1`); the copy uniformly identifies one anchor with another. The verifier's `check_equalize_variable` ([`verifier.py`](../verifier.py)) now contains a free-anchor-merge guard that catches the row directly — see [I-24](30_invariants.md#i-24).

**Fix.** Re-enable the skip — `if (hasDoubleU) continue;` after the `hasDoubleU` computation block. Independently, keep the verifier's `_extract_bound_vars`-based guard live so any future code path that bypasses the prover-side gate gets caught before the chapter ships. See [I-24](30_invariants.md#i-24) and [D-26](40_decisions.md#d-26).

**Don't.** Do not "fix" symptoms by adjusting the substitution direction (representative preference order), by adding compensation downstream, or by widening `oneTypedVars` to include only bound vars (which loses sound bound→free specialisation). The minimal fix is the skip; widening or narrowing the partition set would break orthogonal code paths that this gate does not need to touch.

---

<a id="g-36"></a>
### G-36 — `_extract_args` returns only the outermost `[…]` group

**Manifestation.** A verifier trace-back filter rejects a citation edge that visibly carries the variable being traced. The walker stops, returns a structural-failure verdict, and the chapter is flagged even though the proof is structurally sound.

**Cause.** `_extract_args(expr)` ([`verifier.py`](../../verifier.py)) uses `re.search(r'\[([^\]]*)\]', expr)` which matches **only the first** `[…]` group. For a compound source expression like `!(>[v6](in[v6,N])!(in2[v6,i1_copy,s]))`, the outermost group is `[v6]`; the `i1_copy` nested deeper inside `(in2[v6,i1_copy,s])` is invisible. Filters of the form `lambda src: b in _extract_args(src)` therefore mis-conclude that the source does not carry `b`.

**Detection.** A `variable copy` (or other tag) that traces back through compound expressions: trace-back termination at a row that obviously cites the copied variable. The verifier reports a `variable copy` failure with no other apparent cause. Add a temporary print at the trace-back filter to confirm `_extract_args(src)` is dropping the nested arg.

**Fix.** For trace-back filters that need to detect a variable anywhere in a compound expression, use `_extract_all_args` ([`verifier.py`](../../verifier.py)) — added for `check_variable_copy`. The plain `_extract_args` stays correct semantics for the 30+ atom-level callers that genuinely want the outermost group; do not widen it globally. If a new tag's trace-back encounters this issue, add the recursive helper to that filter site only.

---

<a id="g-38"></a>
### G-38 — Cyclic `equality2` origin chains from cross-pair re-emission of mailed equalities

**Manifestation.** `verifier.py`'s origin-chain-termination check (DFS at [`verifier.py::check_origin_chain_termination`](../verifier.py)) reports cyclic origin chains within a clique of `equality2`-tagged rows in a chapter. Two or more rows of the form `(=[a,b]) equality2 (=[a,c]) (=[c,b])` and `(=[a,c]) equality2 (=[a,b]) (=[b,c])` mutually depend on each other, with no foundational chain breaking out of the cycle.

Historical case: 3 cyclic rows in `files/processed_proof_graph/22_check_zero.txt` (theorem 12 induction zero-case), verifier output:

```
(=[v1,v2]) ns=main
(=[v1,i1]) ns=main
```

reported as cyclic.

**Cause.** When a clique of equalities arrives at an LB via mail (typically from a parent-scope contradiction-cascade — e.g. the Peano "no successor of zero" axiom firing against a zero-case recursion premise), the LB's bulk-merge places mail origins in `body.exprOriginMap`. Without the mail-sync into `EquivalenceClass.equalityOriginMap` ([I-32](30_invariants.md#i-32)) and the cross-pair gate in `mergeTwoEquivalenceClasses`, subsequent class merges iterate every possible bridge variable (`commonArg`) and emit `equality2` cross-pair origin records via DIFFERENT bridges that point at each other. The records survive into the chapter; the verifier's DFS finds the cycle.

**Detection.**

- New rows of the form `equality2 | (=[a,c]) (=[c,b])` AND `equality2 | (=[a,b]) (=[b,c])` for the same clique members in the same chapter.
- The same equality having BOTH a non-`equality2` origin (e.g. `recursion`, `merging origin`, mail-derived) AND an `equality2` origin in the chapter — the second is the redundant cross-pair record that the gate now suppresses.
- Verifier failure under `origin chain termination` tag with cyclic-node lines pointing at clique equalities.

**Fix.** [I-32](30_invariants.md#i-32). The cross-pair `equality2` push in `mergeTwoEquivalenceClasses` ([`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp)) must check `mergedOriginMap`, `classB.equalityOriginMap`, and `memoryBlock.exprOriginMap` for an existing origin entry on the target equality before pushing. Coupled with the producer-side mail-sync at [`prover.cpp::performElementaryLogicalStep`](../GL_Quick_VS/GL_Quick/src/prover.cpp) so that mail-arrived origins are visible to the class state. See [D-46](40_decisions.md#d-46).

**Don't.** Do not weaken the verifier check ([I-16](30_invariants.md#i-16)). Do not skip the gate selectively (the three sources cover three distinct cycle paths; dropping any re-opens that path). Do not blame recent symmetry-handling commits — the cycle generator predates them; the new origin-chain-termination check just exposed it.

---

<a id="g-40"></a>
### G-40 — Cyclic `equality1` origin chains from cross-substitution re-emission of equivalence-class peers

**Manifestation.** `verifier.py`'s `check_origin_chain_termination` reports cyclic origin chains within a *pair* of `equality1`-tagged rows in a chapter. Two shapes have been observed; both involve `applyEquivalenceClass` substituting equivalence-class peers and emitting back-direction `equality1` origins on top of foundational ones:

```text
shape A (chapter-96 self-cycle, original):
(in2[i0,v10,id]) ← equality1 | (in2[i0,i0,id])  (=[i0,v10])
(in2[i0,i0,id])  ← equality1 | (in2[i0,v10,id]) (=[v10,i0])

shape B (chapter-100 swap-cycle, post-gate):
(in2[i0,v10,id]) ← equality1 | (in2[v10,i0,id])  (=[i0,v10]) (=[v10,i0])
(in2[v10,i0,id]) ← equality1 | (in2[i0,v10,id])  (=[i0,v10]) (=[v10,i0])
```

Historical cases:

- **Shape A.** 4 cyclic rows in `files/processed_proof_graph/96_check_zero.txt` (rows 56–57) and `97_check_induction_condition.txt` (rows 72–73). Theorem 96 — Gauss `fold` induction zero-case + step.
- **Shape B.** Same theorem, same chapter pair (re-numbered 100/101 after iter cap bumped 29→35). The producer-side gate from [I-34](30_invariants.md#i-34) / [D-48](40_decisions.md#d-48) suppresses shape A within a single LB but does **not** suppress shape B because the two targets are syntactically distinct keys both first-emitted (each target's exprOriginMap entry is empty when its own emission fires). System-level resolution required.

Sibling of [G-38](#g-38) (cyclic `equality2`); same shape category, different emission path. G-38 is on the *cross-pair transitivity* path (`mergeTwoEquivalenceClasses`); G-40 is on the *argument-substitution* path (`applyEquivalenceClass`) with mail-bulk-merge concentrating the cycle vector at the receiver.

**Cause.** When two members of the same equivalence class (e.g. `{i0, v10}` after successor-uniqueness derives `(=[v10,i0])`) both appear in admissible expressions, `applyEquivalenceClass` fires the substitution loop in both directions, and **multiple producers** (different LBs, different broadcasts) emit `equality1` origin records for the same target across the system. Each producer respects `max_origin_per_expr = 1` locally, but `smashMail` aggregates per-core mail slots into `body.mailIn.exprOriginMap` uncapped. The receiver LB ends up with multiple origins per cycle member — typically a **foundational** `implication` origin AND a cyclic `equality1` origin pointing at the swap peer.

The pre-D-49 bulk-merge from `mailIn.exprOriginMap` to `body.exprOriginMap` was a raw `std::map` swap with body-wins-on-conflict. With cap=1, only the first-pushed origin per key survived, which by mailIn vector order was the cyclic `equality1` — the foundational `implication` was silently dropped. Once installed in `body.exprOriginMap`, every downstream consumer (chapter projection, verifier chain walk) saw only the cyclic origin.

**Detection.**

- New rows where a pair `(P[…X…]) equality1 (P[…Y…]) (=[X,Y])` and `(P[…Y…]) equality1 (P[…X…]) (=[Y,X])` both exist in the same chapter, with no third (non-`equality1`) origin row for either expression.
- Verifier failure under `origin chain termination` with a cyclic-node set of exactly two expressions related by an equivalence-class substitution.
- Inspection of `body.mailIn.exprOriginMap` (added to the hashburst trap dump): two origins per cycle key — at least one foundational (`implication`, `recursion`,...) AND one cyclic `equality1`. With pre-D-49 raw bulk-merge, only the equality1 survives in `body.exprOriginMap`.

**Fix.** [I-35](30_invariants.md#i-35) / [D-49](40_decisions.md#d-49). `addOrigin` gains a cap-full preference replacement: when at cap and the new origin is non-equality, scan for an `equality1`/`equality2` slot and replace it. The bulk-merge from `body.mailIn.exprOriginMap` to `body.exprOriginMap` at `performElementaryLogicalStep` is rewritten to route through `addOrigin` so the preference applies during mail intake. Foundation displaces convenience.

[I-34](30_invariants.md#i-34) / [D-48](40_decisions.md#d-48) (the producer-side `applyEquivalenceClass` gate) is kept as a redundancy guard but is not the cycle-resolution mechanism: it only suppresses redundant emissions inside the LB that ran the substitution; it does not affect the receiver-side cycle vector that `smashMail` aggregates from multiple producers.

**Don't.** Do not skip the `exprOriginMapLocal` populate at the rewrite-enumeration site (would make any first emission malformed — `check_equality1` rejects `len(rest) < 4`). Do not bump `max_origin_per_expr` from 1 as a substitute for D-49 — that defers the choice to the verifier's projection step and shifts the failure mode rather than fixing it. Do not relax I-35's preference to "first-non-equality wins" — replacement, not insertion order, is the structurally correct rule.

---

<a id="g-43"></a>
### G-43 — Radical wipe inflates the NameMap because iter-var counters are monotonic


**Status (after [D-73](40_decisions.md#d-73)).** Symptom no longer reproducible on the recorded workload — the full pipeline (IncubatorPeano → Peano → Compressor → IncubatorGauss → IncubatorGauss1 → Gauss) now completes with `107,055 verifier checks, 0 failures`. Trap-sampled rung-1 `pi_lev_*` mint count fell from 681 (no fix) to 165 (both closure-time filter sites restored). The underlying mechanism — monotonic iter-var counters surviving the wipe — is unchanged; the synchronous filter inserts at the two closure call sites bounce the post-wipe re-derivation cascade at the kernel and rule-fire entry points before fresh iter-vars are minted, so the counter no longer climbs unboundedly on the recorded workload. The gotcha remains documented because heavier workloads (rung-2 onwards, broader OR fan-out) may resurface the inflation pattern; the resolution options below are still open.

**Symptom (historical, pre-[D-73](40_decisions.md#d-73)).** On with the radical subtree wipe ([D-72](40_decisions.md#d-72)) active and no synchronous filter inserts at closure time, the IncubatorGauss1 batch asserted out at `prover.hpp::makeIntNormalizedKey` with `NAME ID OVERFLOW: varId=26190 MAX=16384 arg[1]=pi_lev_1_2810 isUnchangeable=0`. The reported `arg` happens to be a `pi_lev_*` name, which initially looked like a `pi_` runaway — it was not.

**Trap evidence (rung-1 `(EnumerationSet2[2,6,15])` LB hashburst trace plus a temporary `[RUNG1-PI-MINT]` cerr at the `pi_lev_*` mint site in `prepareIntegration`).**

- 681 rung-1 `pi_lev_*` mints captured across the run; the `startIntPi` counter went `0 → 2808`.
- The trace's expanded `nameMap.idToName` dump shows **only 2 unique `pi_lev_*` identifiers** in the entire NameMap (`pi_lev_1_1`, `pi_lev_1_3`). The other 2800+ pi names are created as transient strings inside `prepareIntegration` (line ~4881), used to rewrite changeable args, then immediately substituted to `marker` by `makeMarkedExprLambda` (`prover.hpp` ~4239-4241) before any path that would encode them into the NameMap. Pi vars never settle anywhere.
- The actual NameMap growth at the rung-1 LB is in EXPRESSION names containing iteration variables. `nameMap.idToName` grew `103 → 26183` entries across the run; substring composition of those 26183 names:
 ```
    62322  it_
    59169  repl_
    29346  int_
      467  u_
       42  pi_   (2 unique * 21 dump fires)
  ```
- The `it_*_lev_1_*` counter reached **3400** (1546 unique iter-var identifiers); `int_lev_1_*` reached **3401**. Each new iter var spawns a fresh family of expressions (e.g. `(in3[it_1_lev_1_1534, 9, it_1_lev_1_1566, 4])`, `(preorder[1,4,it_1_lev_1_1534,it_1_lev_1_1566])`, `!(fold[…,it_1_lev_1_1534,it_1_lev_1_1566])`) — each becomes a new `nameMap.idToName` entry. 1546 distinct iter vars × the combinatorial product across expression positions = the 26000-entry NameMap.

**Cause.** The radical wipe is mechanically correct — it drops all per-LB state at descendant scopes when an impl closes. But the prover's `it_*` / `int_*` / `repl_*` / `pi_*` iteration-variable counters live on `Memory` (`startInt`, `startIntRepl`, `startIntPi`) and grow monotonically; they are not reset by the wipe. So when the next burst re-derives the same facts at descendant scopes (the wipe took them, but the producer mail or the kernel's re-firing brings them back), each re-derivation step mints **fresh** iter-var names with higher counter values. Each fresh name = fresh expression = fresh `nameMap.idToName` entry, even if the underlying logical content is identical to a pre-wipe statement. The post-wipe re-derivation pattern is a fan-out of structurally identical expressions distinguished only by counter values.

In baseline (no wipe), iter-var counters stabilise in the hundreds because facts persist across bursts and are not re-derived. With the wipe, the counter climbs unboundedly across bursts.

**Detection.**

- Crash assertion at `prover.hpp::makeIntNormalizedKey` or its peer (`varId < ExecutionParameters::MAX_NAME_IDS`).
- `nameMap.idToName.size` reported in the hashburst trace climbing across bursts past ~16k.
- Iter-var counter (`it_*_lev_*_X` or `int_lev_*_X`) maximum value growing across bursts; expression family duplicates with only the counter differing.

**Don't.** Do not "fix" by reducing the wipe scope — the radical wipe is the user-approved design ([D-72](40_decisions.md#d-72)). Do not pre-allocate skip-lists of structures to preserve from the wipe without explicit user approval (such carve-outs were tried for `integrationPrepared` / `integrationPreparedMarker` / `expandedImplications` and were rejected as guess-and-edit; see ).

**Resolution.** Open. The two structural options are:

1. **Break the monotonicity assumption.** Reset `startInt` / `startIntRepl` / `startIntPi` at end of burst when the LB has just had its subtree wiped. Soundness implications unproven — post-wipe iter vars may collide with surviving statements at ancestor scopes.
2. **Constrain the wipe.** Identify which structures, when wiped, force the re-derivation cycle. Earlier attempts at this surfaced `integrationPrepared` / `integrationPreparedMarker` as gates whose absence allowed re-prepares with fresh iter vars; trap-confirmed evidence is required before any such carve-out lands.

The hashburst trace expansion in [D-74](40_decisions.md#d-74) and the `nameMap.idToName` dump it produces are the diagnostic baseline for the next round of work.

**Code.** No fix yet. Diagnostic: every container is now in the hashburst trace via `hashburst_dump.cpp::writeNameMap` + the extended per-container writers; awaiting user direction.

---

<a id="g-51"></a>
## G-51 — tier-1 deload frees memory only BETWEEN iterations; active LBs stay resident for the whole iteration

**Symptom.** Expecting "RAM proportional to thread count" from the statification deload and measuring instead a peak close to "all active LBs resident" — the same order as before deload existed.

**Mechanism.** `proveKernel` runs barriered sweeps (phase 1 for ALL active LBs → join → phase 2 → join → phase 3 → join → post-join drains), and the deload sweep is deliberately the kernel's last act ([D-155](40_decisions.md#d-155)). Every LB active this iteration is therefore resident from its phase-1 reload to the end-of-kernel deload. The tier-1 win is real but different: LBs IDLE across iterations (discharged, waiting on mail) stay on SSD, and the between-iteration footprint drops to near zero.

**Spot.** Memory profile sawtooths per iteration instead of staying flat-low.

**Fix direction.** Per-phase deload (3 SSD round-trips per LB per iteration at sweep barriers) or per-LB pipelining (restructuring the barriered kernel) are LATER-tier optimizations, each a Rule-8 architecture decision. Do not silently "optimize" tier 1 toward them.

---

<a id="g-52"></a>
## G-52 — every LB with any statified content holds at least one whole block

**Symptom.** The static pool exhausts (assert naming `static_pool_bytes`) at an LB count far below `static_pool_bytes / typical-LB-bytes` intuition; `peakBlocksInUse` ≈ live LB count even though most LBs hold a handful of statements.

**Mechanism.** The grant unit between the global manager and an LB is a whole block (`static_block_bytes`, default 256 KiB). An LB whose statified containers hold even one element still pins a full block (the arena's first allocation acquires a block). Incubator batches run thousands of small LBs; the CE filter adds one single-use clone per thread plus the template. Pool capacity in LB terms is `static_pool_bytes / static_block_bytes` (default 8192), not a byte budget.

**Spot.** `peakBlocksInUse == totalBlocks` in the exhaustion report while the deload files are tiny.

**Fix direction.** Lower `static_block_bytes` (more, smaller blocks — in `parameters.hpp`, all batches) or raise `static_pool_bytes`. Do NOT add a sub-block sharing layer without user approval — cross-LB block sharing breaks the one-owner-per-block model the deload/release lifecycle relies on.

---

<a id="g-53"></a>
## G-53 — cold HashMemory rides the LB arena but sits outside `LbMemory`'s lifecycle enumeration

**Symptom.** `LbArena::freePage` asserts "out-of-range vid" with a garbage or stale `vid` on a COLD arena whose page table is empty (`pageTableSize=0`); OR the static pool exhausts during the chapter export of a *small* batch (not the prover). Three faces of one root cause.

**Mechanism.** All four `HashMemory` instances bind their `encodedMap` (a `TypedColdBlobMap` on arena pages) to `lbMemory.manager` — the two transient ones (`localHashMemoryDelta` / `workingMemory`) moved off the former per-LB hot arena onto it, retiring `hotHashMemoryArena` ([D-176](40_decisions.md#d-176)) — yet are declared as `Memory` members OUTSIDE `LbMemory`. So `LbMemory::visitContainers` — the single enumeration the deload dump, load, AND release all drive — never sees them, and each arena-lifecycle operation must thread all four by hand. Three things break when one is forgotten:
1. **Release.** `releaseStaticBlocks` releasing only `LbMemory::visitContainers` then `manager.releaseAll` leaves the `encodedMap` `PagedVector`s holding page vids into the wiped arena; the next reload's `clear` walks a stale directory and `freePage`s garbage (an ASCII-looking `vid` is blob/key bytes read as a vid).
2. **Export reload.** The chapter export (`buildStack` / the `generateRawProofGraph` driver) `ensureLoadedForRead`s every LB across the proof-graph walk, but nothing released them, so resident blocks climb monotonically to pool exhaustion — looks like a footprint problem, is not.
3. **Teardown.** The four `HashMemory` instances, declared before `lbMemory`, destruct AFTER its arena, so `~PagedVector` `freePage`s on a dead arena — on EVERY `Memory` destruction (CE-filter clone LBs, grid children, the root at process exit).

**Spot.** `[TRAP] freePage OOR vid=… pageTableSize=0 mode=COLD` (the diagnostic that found all three). The crash phase distinguishes the face: during a reload (face 1), during the export with `blocksInUse` climbing (face 2), or at teardown / inside the CE filter / process exit (face 3).

**Fix direction (current).** Face 1: `releaseStaticBlocks` releases the cold containers via `HashMemory::visitContainers(base)` before `releaseAll`. Face 2: `generateRawProofGraph` releases each theorem's reloaded LBs (the `g_exportReloadSink` that `Memory::reloadFromImage` populates only during the export, plus `releaseStaticBlocksDispatch`, which routes Raw-kind LBs to the walk-free raw release — the raw rebind needs the container bookkeeping, and an extent-raw LB has no named file set) at chapter end; the on-disk image stays for revisits. Face 3: an explicit `~Memory` empties all four `encodedMap`s before any member destructs, and the now-live `destroyGrid` (a member, called after the export in `run_modes::fullRun`) wipes the grid at batch end. ALL THREE retire when `HashMemory` folds INTO `LbMemory` — the single enumeration then covers it and the arena destructs last — the stated Part-C endgame.

---

<a id="g-54"></a>
## G-54 — intermittent `ColdHashSet::indexPlace: duplicate id` abort under heavy reload pressure

**Symptom.** A `gl_quick.exe <tag>` run (observed: Gauss; also recorded 2026-06-25 on `IncubatorPeano1`) aborts in a parallel hash burst on `assert(buckets_.at(i)!= id && "ColdHashSet::indexPlace: duplicate id")` — the cold-index minting an id already present. **Intermittent:** one abort in three full-pipeline runs of identical logic; a re-run usually completes byte-identically with the verifier clean.

**Mechanism (suspected — not yet caught at its origin).** Minting is single-threaded ([I-83](30_invariants.md#i-83)) and `PagedHashIndex::reset` memsets every page to zero, so a stale (non-zeroed) page is ruled out — the dup means the throw-away index and its key store disagree on the id set, i.e. a count/index DESYNC reached `indexInsert`. It is reload-sequence dependent: it surfaces in the heaviest batch under deload/reload churn, so WHICH LBs reload WHEN (steward timing) decides whether it fires. The slab-pool LB-body store (`lbMemory`) and the cold index (`staticMemory`) are DISJOINT reservations, so the LB-body statification cannot corrupt the index by overlap — the cutover only perturbs timing; it is not the cause (two full runs post-cutover were byte-identical + verifier 0).

**Spot.** The original `indexPlace` dup-assert firing during a phase-2 burst (`active_bodies` large) in the heaviest batch. To localize the ORIGIN, four consistency asserts are armed in `cold_hash_map.hpp` (`HashMap`): `rebuildIndex` end (`occupiedBuckets == count`), `indexInsert` entry (`id == count`), `mint` (post-insert `lookup(k) == id` round-trip), `resetToFresh` (fully empty). Whichever fires pins the desync to its operation; if the dup recurs with NONE firing, the stale id entered between rebuilds → external corruption or a count-rollback path not yet asserted.

**Fix direction.** Catch it under the armed asserts (intermittent — needs the desync to occur during an instrumented run; stressing deload by shrinking `static_pool_bytes` raises the hit rate), then fix the operation the firing assert names. Pre-existing and DEFERRED (user, 2026-06-26) — the LB-body campaign proceeds; the asserts ride along to catch the origin whenever it next fires.

---

<a id="g-55"></a>
## G-55 — mail-inbox writers must `ensureArena` before ANY insert; origins-only blobs skip `insertStatement`'s lazy bind

**Symptom.** After the mail-system statification, `gl_quick.exe Peano` AV'd (`0xC0000005`, no assert) at Phase-2 hash-burst 1, inside `MailLog::pull` → `Codec<Mail>::deserializeInto`, decoding a VALID blob (`start+len` within the pool). Non-deterministic multi-thread, deterministic single-thread; the small incubator batches never hit it.

**Cause.** `RoutingColdMail` binds its self-owned mail arena LAZILY via `ensureArena`. `insertStatement` calls it; the origins path — `addMailOriginRecord(inbox.origins_, …)` called directly by `deserializeInto` — does NOT. `deserializeInto` implicitly relied on the statements loop to bind the arena first. An **origins-only blob** (`stCount == 0`: origins present, no statements) runs zero `insertStatement`s, so `addMailOriginRecord` writes through an UNBOUND arena → AV. Peano emits origins-only commits at scale; the incubator's first pulled blob always had a statement, which bound the arena and masked the bug.

**Spot.** Value-markers in `deserializeInto` printed `stCount=0`, `keyCount=31`, crash at the first origin insert. The two sibling inbox writers, `mergeBatchIntoMailIn` ([`mail_log.hpp`](../GL_Quick_VS/GL_Quick/src/mail_log.hpp)) and `addRoutingMailOrigin` ([`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp)), both `ensureArena` first — `deserializeInto` was the one that forgot.

**Fix.** `Codec<Mail>::deserializeInto` calls `inbox.ensureArena` at the top, before either loop.

**Rule.** EVERY mail-inbox writer `ensureArena`s before its first insert — never assume a prior insert bound it. Origins-only blobs (statements empty, origins non-empty) are a real committed shape.

---

<a id="g-56"></a>
## G-56 — a worker-pool entry point that never publishes `g_currentCoreId` collapses every thread onto the shared reserved scratch slot


**Symptom.** A parallel phase-2 hash burst aborts on `LbArena::popTo`'s "forwards past the cursor" guard. The frame that trips it resolved its per-slot arena with `g_currentCoreId == -1` (the reserved slot, `scratchArenas.slotCount - 1`), while running on a worker-pool thread that should own a slot of its own. Concurrent sibling scopes rewind the SAME reserved arena, so one scope's `popTo` targets a cursor another scope already moved.

**Mechanism.** Every `g_currentCoreId`-resolved per-slot consumer reads the thread-local `g_currentCoreId`; when the thread never published it, the resolution falls back to the single reserved slot (`slotCount - 1`). That reserved slot is a SINGLE-THREADED contingency (setup / drains, where `g_currentCoreId == -1` legitimately). Two or more pool threads landing on it simultaneously each open/rewind scopes on one arena — a data race on the cursor, surfaced deterministically by `popTo`'s guard. The phase-1, phase-3, and phase-2 FINALIZE workers all publish their slot (`g_currentCoreId = static_cast<int>(cid)` / `tIdx`); the phase-2 EXECUTOR worker did not. It was LATENT for as long as the pool existed and harmless while NO `g_currentCoreId`-resolved consumer sat on the executor path (the request-generation scratch takes the `coreId` PARAMETER directly, never the thread-local). It was exposed the moment a consumer landed there — `allowedForMail`'s `C5` string-scratch scope, reached from `performElem2` → `checkLocalEncodedMemoryStatic` → `allowedForMail`.

**Second instance — the CE filter worker pool.** `filterConjecturesWithCE`'s workers publish their slot only inside `performElemPhase1` (the burst), yet `addConjectureForCEFiltering` runs FIRST and already reaches `g_currentCoreId`-resolved scratch (`addToHashMemory` → `makeNormalizedKeysForAdmission` → `insertRemainingArgsNormKey`, and `disintegrateExpr2`). It was latent until those installers were statified onto the per-slot arenas; then every worker's FIRST conjecture ran on the reserved slot and the concurrent workers raced its byte cursor — here surfacing as a `resolve`-of-unallocated-offset abort inside the `insertRemainingArgsNormKey` RMW (`off > cursor` with `gen == 0`, a straight-line grow-only frame that only a concurrent rewind can invert) rather than a `popTo` abort. Same one-line fix: publish `g_currentCoreId = static_cast<int>(coreId)` at the worker-lambda top, before `addConjectureForCEFiltering`.

**Spot.** A `popTo` forwards-past-cursor abort whose frame carries `core=-1` inside a worker-pool context (phase-1/2/3 sweep), with every pool thread sharing the reserved slot. Contrast with the reserved slot's legitimate single-threaded users (setup, grid build, compressor, drains) where `-1` is correct.

**Fix.** Publish the worker's slot at the top of the pool's worker lambda, BEFORE any code that resolves a per-slot arena from `g_currentCoreId` runs on that thread — `g_currentCoreId = static_cast<int>(cid);`, mirroring the finalize worker. One line; changes only which slot the consumer resolves to, not any proof output (arena grant order is proof-output-invisible, [I-107](30_invariants.md#i-107)).

**Rule.** EVERY worker-pool entry point MUST publish `g_currentCoreId` before the first per-slot-arena resolution on that thread. A pool thread that leaves it at `-1` silently shares the single reserved slot with every sibling — safe only while nothing on that thread's path reads `g_currentCoreId`, a precondition the next consumer added to the path will break without warning.

---

<a id="g-59"></a>
## G-59 — the `(int64_t)id << 32` map-key pack idiom is undefined behaviour for a negative id


**Symptom.** UBSan (`-fsanitize=undefined`) reports `runtime error: left shift of negative value -1` in `Codec<LbStatePairKey>::encode` (`typed_cold_map.hpp`) — the only UB in the whole unit-test suite under a full non-halting sweep. Silent on MSVC and on a normal (non-sanitizer) g++ build, so it never affected a proof run.

**Mechanism.** The engine's `(int32_t, int32_t) -> int64_t` map keys were packed as `(static_cast<int64_t>(hi) << 32) | static_cast<uint32_t>(lo)`. Left-shifting a *signed* negative value is undefined behaviour in C++17, and these keys legitimately carry a `-1` sentinel in `hi` (e.g. a no-ancestor / eternal-root validity in `expandedImplications`, keyed `(maybeAncestor, descendant)`). g++ happens to emit the two's-complement bit pattern, so `decode(encode(k)) == k` still holds and no run was ever wrong — the UB is latent, but real and compiler-dependent. The same idiom sat at ~5 production sites (`packLbStateKey`, `packOriginKey`, both `packMappingKey` lambdas, the codec) plus test-local packers. Only the `hi` half was UB — the `lo` half already cast through `uint32_t`.

**Spot.** A `left shift of negative value` UBSan report, or any `(static_cast<int64_t>(x) << 32)` on a value that can be a negative id/sentinel.

**Fix.** One shared `constexpr` primitive `packInt32Pair(int32_t hi, int32_t lo)` (`typed_cold_map.hpp`) does the whole pack in `uint64_t` (`(uint64_t)(uint32_t)hi << 32 | (uint32_t)lo`, cast to `int64_t`), well-defined for every input and **byte-identical** to the former form on a two's-complement target — so every stored key value is preserved. All production packers route through it; the `decode` twins already used the unsigned form. Unit-tested (`test_typed_cold_map.cpp` `pack_int32_pair`, corners incl. negatives).

**Rule.** Never `(int64_t)signed << 32` to build a key — widen through `uint32_t`/`uint64_t` first (route through `packInt32Pair`). Applies wherever two 32-bit ids fold into one 64-bit key.

---

<a id="g-60"></a>
## G-60 — a unit test that `memcmp`s a padded struct fails non-deterministically


**Symptom.** `sealed_pages.record_payload_address_stable` fails intermittently (~⅓ of runs with asserts, ~⅔ without) on g++, always passes on MSVC — a long-standing flaky unit-test crash mistaken for a static-memory / `SealedPageSet` bug. The failing check is a `std::memcmp` in `test_sealed_pages.cpp`.

**Mechanism.** The record type `ChainP { int32_t a; int64_t b; }` has 4 bytes of alignment padding (offsets 4-7). The test compared a stored record against a freshly-built `expected` local with `memcmp` over `sizeof(ChainP)`. Aggregate init (`ChainP{a, b}`) leaves the padding bytes unspecified; g++ at `-O3` leaves stack garbage there (varies run-to-run → non-deterministic), MSVC happens to zero it (always passes). A byte dump showed the diff was *only* at the padding offsets — the record fields `a`/`b` and the payload address were always correct, so `SealedPageSet` was never at fault.

**Spot.** A non-deterministic, g++-only unit-test failure whose check is a `memcmp` over a struct with mixed-alignment members (hence padding).

**Fix.** Compare the fields, not the raw bytes (`ASSERT_EQ(r->a,...); ASSERT_EQ(r->b,...)`).

**Rule.** A value-equality check on a struct compares fields, never `memcmp` — padding bytes are unspecified and non-deterministic.

---

<a id="g-58"></a>
## G-58 — a statified single-concat RMW asserts when a key's cold run outgrows one arena block


**Symptom.** A cold-blob-map RMW insert aborts on `total <= ExecutionParameters::kMaxAdmissionRunBytes` (262144, one block). The first to fire is `insertRemainingArgsNormKey` on a Gauss-scale incubator batch (`IncubatorGauss1`); Peano stays well under one block so it never surfaces there.

**Mechanism.** These RMWs peek the existing run into gArena, splice the new record at its sorted position, and build the WHOLE new run as one contiguous `gArena.alloc(total)` buffer for one `assignRun`. An arena alloc may not straddle a block (`LbArena::alloc` asserts `bytes <= blockBytes`), so the single-concat path caps a key's run at one block. The retired heap forms (`std::vector<NormKey>` / `std::set`) had no such ceiling, so a run that legitimately grows past 256 KB — reachable at Gauss scale — crashed only after statification.

**Spot.** A `kMaxAdmissionRunBytes` abort naming a `remaining-args NormKey` / `admission` run, reached from the hash-rule install path, on a batch large enough to pack one `argSet`'s run past a block. Peano never reaches it.

**Fix.** Split the write: keep the one-block concat for `total <= kMaxAdmissionRunBytes` (the common case, unchanged and byte-identical), and for a larger run open the key's run empty (`assignRun(k, nullptr, nullptr, 0)`) then append each blob in sorted order through `appendBlobToRun`. The map's `BlobCsrValueStore` pool straddles pages, so the run there is unbounded, and `appendBlobToRun` lands each blob byte-identical to the concat's `assignRun` (test `test_cold_hash_map.cpp`, `viaAppend` == `viaAssign`). Applied to all four such RMWs: `insertRemainingArgsNormKey` (`prover.hpp`) — where the Gauss batch actually overflowed — and the three siblings `insertAdmissionBlobSorted` / `insertRejectedBlobSorted` / `insertRejectedIntegrationBlobSorted` (`memory.hpp`), widened pre-emptively since they share the exact pattern and would assert the same way at scale.

**Rule.** Every statified single-concat RMW that can grow a key's run without bound needs the widening branch. All four current such RMWs (the `insertRemainingArgsNormKey` + the three `insert*BlobSorted`) now carry it; any new single-concat cold-blob RMW must add the same `total > kMaxAdmissionRunBytes` widening branch, not just the raw ceiling assert.

---

<a id="g-57"></a>
## G-57 — scratch-slot fallbacks are PER-REGISTRY; a slot derived from one registry must never index the other


**Symptom.** `gl_quick.exe` aborts (0xC0000409) on the `ScratchArenaRegistry::forSlot` range assert ("scratch arena slot out of range — coreId beyond the logicalCores the registry was initialized with") on a SINGLE-THREADED path — `g_currentCoreId == -1`, the offending slot exactly equal to the gen registry's slot count. First surfaced in the batch-7 gate: the first main-thread `addToHashMemory` install of an incubator batch, immediately after CE filtering.

**Mechanism.** The two scratch registries deliberately differ by one slot: `initScratchArenas(logicalCores + 1)` gives the STRING registry a DEDICATED reserved single-threaded slot (index `logicalCores`), while `initGenScratchArenas(logicalCores)` gives the GEN registry none — its single-threaded fallback convention is `slotCount - 1` (= `logicalCores - 1`, shared with the last worker slot, safe because `-1` contexts run only in single-threaded phases with every pool joined). A site that computes ONE slot value from the string registry's fallback (`scratchArenas.slotCount - 1` = `logicalCores`) and indexes BOTH registries with it hands the gen registry an index one past its last slot — a deterministic assert on the first `-1`-context call. Worker contexts (`g_currentCoreId >= 0`, ids `0..logicalCores-1`) are in range for both registries, which is why the bug hides until a single-threaded caller reaches the site.

**Spot.** A `forSlot` range assert with the offending slot == the gen registry's `slotCount` and the thread's `g_currentCoreId == -1`; the site resolves arenas from BOTH registries but computes only one slot value.

**Fix.** Per-registry slot derivation: `strSlot = (coreId >= 0)? coreId: scratchArenas.slotCount - 1` for the string arena and `genSlot = (coreId >= 0)? coreId: genScratchArenas.slotCount - 1` for the gen arena — each fallback from its OWN registry's `slotCount`. This is NOT the [G-56](#g-56--a-worker-pool-entry-point-that-never-publishes-g_currentcoreid-collapses-every-thread-onto-the-shared-reserved-scratch-slot) clamp anti-pattern: workers keep their own disjoint `coreId` slot on both registries; only the genuinely single-threaded `-1` context takes a fallback, where no worker runs concurrently. Slot identity reaches no proof output ([I-107](30_invariants.md#i-107)).

**Rule.** Scratch-slot fallbacks are PER-REGISTRY — never index one registry with a slot derived from another's `slotCount`. A site that touches both registries computes both slots. (The cookbooks' slot-handle idiom lines carry the same rule.)

---

<a id="g-61"></a>
## G-61 — a paren-anchored subexpression extractor silently strips a leading negation


**Symptom.** An induction-hypothesis rule silently INVERTS a negated premise. For the OR-companion theorem, `createAuxyImplication` installed `(in[u_rec,u_1]) (existence3[u_1,u_rec,u_3]) ⇒ (=[u_rec,u_2])` — "anything with a predecessor is 0", mathematically false — where the source theorem's premise is the NEGATED `!(existence3[…])`. The unsound rule is latent while its frozen premises are unsatisfiable and proves false statements the moment they are not.

**Mechanism.** `extractSubstringsForAuxy` captured paren-to-paren windows `(X[...])` only; a `!` immediately preceding the `(` sat outside the window, so every negated leaf premise entered the auxiliary chain positive. Nothing crashes — the rule installs and sits in `overallHashMemory.originals` looking legitimate.

**Spot.** Dump the recursion sub-LB (sacred dump) and compare each auxiliary rule's premises against the source theorem's chain: a premise that lost its `!` is this gotcha. Any lexical extractor whose window starts at a fixed opening byte is suspect for the same class of drop.

**Fix.** The extractor includes a `!` immediately preceding the matched `(` (lexical twin of `!?\(([^>(\[]+\[[^\]]*\])\)`); the regex-oracle unit test carries negated forms and pins the companion shape verbatim (`test_memory.cpp`, `extract_substrings_for_auxy_matches_regex_oracle`).

---

<a id="g-62"></a>
## G-62 — `patterns_to_exclude` never sees negated-premise variants


**Symptom.** A `patterns_to_exclude` regex written to bar a conjecture shape has no effect on rows carrying a negated premise: the barred shape's `!(...)`-premise variants appear in `conjectures.txt` even though the regex matches them.

**Mechanism.** `generateNegatedPremiseVariants` runs AFTER the string-lane filter cascade (`patternInConjecture` included) and derives one negated-premise variant per negatable premise from each *surviving* positive row. The variants are emitted without re-entering the filters, so a pattern only ever screens positive base forms. Conversely, a pattern that matches only mixed-polarity text (e.g. one negated plus one positive `=`) is dead config — no candidate carrying that text ever reaches `patternInConjecture`.

**Spot.** Emitted rows whose only difference from a pattern-barred shape is a `!` on one premise; or a pattern whose removal changes nothing in the pool.

**Fix.** Bar the positive base form — the variants die with it. Verify a new pattern empirically by diffing the pool with and without it (`--conjecture <Tag>` twice).

---

## See also

- [`30_invariants.md`](30_invariants.md) — the rules that, when forgotten, produce the gotchas above.
- [`40_decisions.md`](40_decisions.md) — why the rules exist.
- — the `feedback_*` files are the living history of gotcha discoveries.

<a id="g-63"></a>
## G-63 — a definition body with `(!(...))`-wrapped negation children fast-fails the definition parser with no message

**Symptom.** `gl_quick.exe` dies with exit 0xC0000409 (stack-buffer fast-fail, Release, no assert text) between the flushed `[INIT] initAnchor done.` print and `Loading proved theorems...` — i.e. inside `ExpressionAnalyzer::compileCoreExpressionMap`'s per-definition `ArgumentAnalyzer` pass — after adding a new operator whose `.mpl` definition wraps a negation child in extra parentheses, e.g. `(&(!(=[n,m]))...)`.

**Cause.** Canonical MPL writes negation children bare — `!(=[n,m])`, as every shipped definition does (`nonInterval.mpl` is the reference shape). The recursive definition parser treats `(` as the start of an operator-named node; `(` followed by `!` has no name and overruns a stack buffer.

**Fix.** Write definition-body children as `(name[args])` or bare `!(...)` only. Diagnosis shortcut when the crash window matches: diff the new definition against `nonInterval.mpl`'s negation shape before instrumenting anything.

<a id="g-64"></a>
## G-64 — the same OR structure minted under two compact names breaks registry matching across artifacts


**Symptom.** A single verifier `origin` failure on a chapter whose `implication` row cites an or-carrying rule: the cited rule IS a proved, registered theorem, but its registry row names the or-compact `or<N>` while the citation names `or<M>` — alpha normalization bridges variable names, never operator names, so the match rightly fails. Looks like an unproved-theorem use; is a naming split.

**Cause.** `constructOrTheorem`'s registration dedup was a stub (`found = true; // for now, assume unique`) that never consulted the registry, so every constructing batch minted a fresh `or<N>` for the same disjunct structure (an I-23 violation). Cross-batch theorem files carry the expanded base form; the reloading batch's compile-side dedup (`repetitionExclusionMap`, one slot per `(elements, category)` key) then resolves the structure to ONE of the duplicate names — not necessarily the one the producing batch wrote into its `global_theorem_list.txt`. Reachable only once incubator batches construct ORs at all (D-255); the D-55 both-directions gate had masked it.

**Fix.** Mint-side dedup mirrors the load side: `constructOrTheorem` builds the u_-canonical elements first, scans `compiledExpressions` for an `or`-category entry with the identical ordered element list, and reuses the existing name on a hit (no counter bump, no re-registration). Diagnosis shortcut: when an origin failure names an or-carrying citation, grep the producing batch's `global_theorem_list.txt` for the same structure under a DIFFERENT `or<N>` before assuming a provenance hole.

<a id="g-65"></a>
## G-65 — the base-form writer emitted `!!(...)` for negated existence head elements; latent because final-batch rows were never reloaded


**Symptom.** `gl_quick.exe` dies with 0xC0000005 right after `Loaded 0 proved theorems.` when a batch loads external theorems whose rows contain a double negation `!!(...)` — malformed MPL that no parser path normalizes (OPEN-MPL-1: no universal double-negation canonicalizer). The rows came from `files/theorems/theorems.txt` itself.

**Cause.** `expandToBaseForm`'s existence branch negated the head element with a blind `"!" +` prefix; an existence whose registered head element already carries `!` (the Gauss `limitSet`/`interval` shapes) produced `!!(...)` — 6 such rows in every generated pool, `main`'s included. Same polarity-blindness class as the or-element bug ([I-175](30_invariants.md#i-175)) — third instance of the family. Latent for months because the affected rows are Gauss-batch rows and Gauss is the final batch: nothing downstream ever reloaded them until the shortcut corpus was refreshed from the base pool.

**Fix.** The head negation cancels a leading `!` (matching the or-branch's `negate` and the outer negation block in the same function). Diagnosis shortcut: on a load-path access violation over inter-batch theorem files, `grep -c '!!'` the input file first.

<a id="g-66"></a>
## G-66 — a nondeterministic sort key is latent until a new consumer makes tied groups observable


**Symptom.** Same-input full runs diverge in one registration observable (a theorem's method, a chapter's producer, a count total) while EVERY per-burst content observable stays byte-identical across runs — the signature that the proof engine is deterministic and only a post-join drain's processing order raced.

**Cause.** A single-threaded drain sorted its records on a key that includes a scheduling artifact — here `coreId`, the worker slot that sealed the record, in `updateGlobalDirectLess`. The key was harmless for years because tied groups (same theorem, different records) had an order-insensitive outcome: the sink's string dedup dropped the loser and both records carried the same effect. The proved-not-broadcast tier changed the loser's effect (a refused record now REGISTERS something different), and the latent key became an observable coin flip. The general trap: "order-insensitive" claims about a nondeterministically-ordered tie are load-bearing assumptions that silently expire when any consumer starts distinguishing tied records.

**Fix.** Sort keys in single-threaded drains must be pure functions of proof state (bytes, verdicts, producer chains via `compareProducerChains`) — never worker slots, allocation addresses, or arrival order. When auditing: for every `std::sort` in a post-join drain, ask what breaks ties and whether two records that tie on the deterministic prefix can carry DIFFERENT effects at the sink.

<a id="g-68"></a>
## G-68 — a disintegration recognizer that read another function's side effect was silently dead for four weeks


**Symptom.** A known negated existence `!(existence<N>[args])` registered flat and produced no `left → !right` / `right → !left` rule; nothing failed — theorem counts, the verifier and the artifacts were all unchanged, because the only consumer in the artifacts (the chapter of `¬∃m(m+1=n) → n=0`) closes by direct contradiction against the derived positive existence, never through the rules.

**Root cause.** The negated-existence block in `disintegrateExprCore2` recognized its input by probing the prepared instruction for the positive existence entity (`innerHit`). `disintegrateExpr2` fills that instruction through `prepareIntegrationCore`, whose polarity guard (, 2026-08-04) returns without committing anything for a `!(` non-atomic input — the guard's comment assumed negated existences are intercepted upstream in `prepareIntegration`, which is true for the integration wrapper but not for the disintegration path. From that commit on the probe could never succeed. A trapped full run: 19 negated existences admitted to disintegration, every one with an empty instruction, the block fired 0 times.

**Fix.** The recognizer resolves the inner existence through the registry (`compiledEntity` on the inner core, category `existence`, two elements) — the or route's pattern — and the compact route (D-310) produces the rules.

**Lesson.** A recognizer that depends on a side effect of another function (an entity committed by `prepareIntegrationCore`) needs either the source of truth itself (the registry) or a tripwire assert that fires when the side effect is missing; a silent `if` on someone else's state is a latent no-op waiting for the next guard. And "no theorem lost" is not evidence a mechanism runs — trap it.


<a id="g-67"></a>
## G-67 — a statement registered and converged in one step lost its history in mail; the failing LB was not the producer


**Symptom.** `buildStack: no origin found` in a contradiction LB for a statement at an `_ordis_` branch scope, cited by an `or convergence` row whose scope the failing LB never minted; the walker resolves the cohort's first branch and fails on the last one; every converged row of the cohort fails the same way (96 on IncubatorGauss3).

**Root cause.** The producer (the parent LB) derived the expression at the last-released branch and converged it in the same step: `ordisMerge` step 2 erased the branch row from `intLocalEncodedStatementsDelta` before `fillMailOut` walked it, so the branch's `equality1` row never shipped while the convergence row citing it did. The producer's own map held the origin all along — the hole is in what reaches descendants ([D-311](40_decisions.md#d-311)).

**Diagnosis lesson.** A first trapped run instrumented every local writer of the failing LB and stayed completely silent while the crash reproduced — the rows had come by mail. When a walker fails on a scope the failing LB never minted (its `orBookkeeping` empty at every burst, its `toBeProved` never holding the cited goal), trap the ancestor that owns the scope, and trap the mail-out selector, not only the registration doors. The dump's `mailIn` sections cannot show it: the inbox is absorbed and cleared before the entry dump ([I-101](30_invariants.md#i-101)).

**Fix.** Ship first, drop second at every delta-dropping site — [D-286](40_decisions.md#d-286) for the canonicalization prune, [D-311](40_decisions.md#d-311) for the convergence removal; the sanctioned writers are listed in [I-64](30_invariants.md#i-64).

<a id="g-69"></a>
## G-69 — under `allow_multiplication` a rule's install is its multiplication's copies: a rule whose multiplication emits no copy installs NOTHING, and a multiplied rule installs one text per surviving partition (2026-08-28, )

**Symptom.** A removal (or any code that assumes "this expanded implication is in hash memory under its own text") asserts `ruleInterner.lookup(text) == 0` on an incubator batch (`allow_multiplication` is `true` in `IncubatorPeano1/2` and `IncubatorGauss1/2`, `false` in `Peano` / `Gauss` / `IncubatorGauss3`) although the door recorded the expansion. Observed: `(>[]!(=[u_10,u_6])!(=[u_2,u_2]))` at `main` in the contradiction LB `!(=[10,2])` of IncubatorPeano2, hashburst 4.

**Mechanism.** `addToHashMemory` loops over `multiplyImplication`'s copies and installs each copy under its own `ruleInterner` id (`originalImplicationId` of every LMV = the copy's id; the owner pair of every index entry likewise). With multiplication off the one copy is the text itself. With it on, the discrete partition's copy is `deduplicateBoundVarsScratch(text)`, and the per-copy trivial-head skip (`(=[x,x])` heads) applies to that copy too; every other partition that would equate two `u_` variables is skipped (`hasDoubleU`). A rule with a trivial-equality head whose one-typed variables are all `u_` therefore has ZERO copies — no chain, no owner, no LMV — while the door's index writes (`expandedImplications`, `compactExpansions`, `expansionCarrierCount`) do not depend on the copy count.

**Spot.** `RuleIndexOp::dropSet` empty after a removal enumeration; `hm.originals.count` unchanged by an install; the unit tests `remove_rule_with_multiplied_copies_drops_every_copy` and `zero_copy_rule_removal_is_a_no_op`.

**Fix direction.** Never look a rule up by its recorded text; run the install's own enumeration (`RuleIndexOp::Remove` stages every copy's pair, the transient probe reads the first copy's chain owner). The install/removal symmetry is the contract — zero copies in, zero out. Whether a zero-copy expansion should be recorded at all (it costs one carrier-index record and is never fireable) is an open design question, not a bug.

<a id="g-71"></a>
## G-71 — An `equality2` line can be the ONLY well-founded origin of a class-pair equality

**Symptom.** `buildStack` hits its 5M-call tripwire; every class-pair equality of a collapsed class carries exactly `max_origin_per_expr` same-tag `implication` rows and no `equality2` line; a least-fixpoint pass says the chapter goal is not derivable.

**Cause.** Under the canonical door a rule can re-derive a class equality from canonical premises whose only origin cites that equality (circular); the transitivity line written at the class merge is the one non-circular row, and a tag-based cap policy that treats `equality2` as disposable removes it ([D-325](40_decisions.md#d-325)).

**Rule.** Origin rows are kept in insertion order at the cap; tags say nothing about foundational value. Diagnose derivability with a least-fixpoint pass over `exprOriginMap` before touching the walker.

<a id="g-70"></a>
## G-70 — An `exprOriginMap` row no longer implies a registered statement

**Symptom.** A copy-split statement (`(P[…,X_copy,…])`) never appears at `main` although rule firings derived it (origin rows exist) and its canonical base is registered and multiplied; a lemma whose proof needs the copy (collision-pattern-exact rule keys) is lost together with every theorem citing it.

**Cause.** The canonical door keeps the raw producer line of a deposit it folds and ends (Rule 16 documentation), so origin rows exist for texts that were never registered. Any prover gate that reads `exprOriginMap` as "known" is now wrong — the `applyEquivalenceClass` product commit did exactly that ([D-321](40_decisions.md#d-321)).

**Rule.** Presence in the statement registry is the only "known" test (I-85); the origin map is process documentation (I-44). Diagnose with the Rule-14 dump: `encodedStatements` versus `exprOriginMap` for the same key.

<a id="g-73"></a>
## G-73 — the canonical door's normal-name walk is bracket-flat: a normal-name member nested inside a compound argument is NOT rewritten

**Symptom.** Two spellings that differ only in a name nested inside a compound argument — `(in2[(s[b]),7])` versus `(in2[(s[a]),7])` under the class `{a, b}` — both survive the canonical door as distinct registered statements; a dedup or equality built on "the door canonicalizes member tokens at any depth" silently misses them (this bit the or-uniqueness gate's first test).

**Cause.** `canonicalFormAtScope`'s normal-name token source is `collectExprTokens` ([`str_ops.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/str_ops.hpp)) — a FLAT bracket tokenizer: it scans to the next `[`, takes everything to the next `]` as one comma-split token run, and resumes AFTER that `]`. On a nested compound arg it yields mangled fragments (`(s[b` from the example) and never the inner name, and the text after the inner `]` up to the outer one is skipped entirely. Only the two special tiers (`int_lev_*` / `it_*_lev_*`) are depth-independent — they come from the occurrence scan (`buildSpecialTokenScanView`), not the bracket walk.

**Rule.** Anything comparing spellings "under the door's canonicalization" must use the door's own walk on both sides (the or-uniqueness gate does exactly this) — never assume depth-complete normal-name rewriting. Whether the flat walk is a deliberate scope bound or a latent gap in I-217 is a maintainer question; do not widen `collectExprTokens` without Rule-8 approval (every consumer's byte behavior shifts).

<a id="g-72"></a>
## G-72 — the rule-owner removal assert fires intermittently on CUDA full runs; its context diagnostic is PERMANENT until root-caused (2026-09-01, )

**Status (2026-09-04, release 13, ): CLOSED and de-instrumented.** The defect is root-caused and fixed ([D-336](40_decisions.md#d-336), [D-337](40_decisions.md#d-337), [I-222](30_invariants.md#i-222)); on the maintainer's order every G-72-specific instrument described below is removed: the `G72_LEDGER` switch and `g72Ledger` writer, `dumpRuleKeyFamily` and its test, the `[RULE-OWNER-REMOVE-*]` context prints, and the remove policy's diagnostic-only fields. The two `removeOwnerFromRun` asserts stay exactly as they were. The abort-time minidump handler in `main.cpp` is general crash plumbing (every assert, every batch) and stays. The paragraphs below are the history of the hunt.

**Symptom.** `removeOwnerFromRun: the key is absent` (the Rule-19 assert in `prover.hpp::removeOwnerFromRun`) aborts a full mixed-CUDA `main.py` run intermittently — observed once in the Peano main prover around hash burst 9 on this branch, and earlier during 's full-run acceptance. Identical-semantics runs pass either side of a firing, and a passing CUDA run's artifacts are byte-identical to the processor reference.

**Reading.** The removal re-enumerates the install's own keys ([I-210](30_invariants.md#i-210) owner lists), so a lookup miss means the remove enumeration diverged from the install. Because proof state is provably identical across backends on passing runs, an intermittent miss implies a timing-sensitive divergence — a latent race or an unpinned order in the owner-list bookkeeping — most likely a defect in the shared machinery that the CUDA route's compressed Phase 2 timing exposes and processor scheduling never hits. No processor-route firing has ever been observed.

**Standing instrumentation (maintainer directive, 2026-09-01).** The full-context diagnostic — `[RULE-OWNER-REMOVE-FAILURE]` (index name, key length, owner) before the retained assert, and `[RULE-OWNER-REMOVE-CONTEXT]` (hash memory, rule text, scope, full `parentMemory` LB chain per Rule 12) at the edge-removal first-visit miss — is PERMANENT observability, not a Rule-30 temporary trap. It stays in the tree, through squashes, until the divergence is root-caused and fixed. The assert itself is never weakened (Rule 19, [I-19](30_invariants.md#i-19)).

**On a firing.** Capture the stderr context lines and the run log; the named rule, scope, and LB chain are the trap-debug entry point (retarget the Rule-14 hashburst dump at that LB chain next).
**Instrument (2026-09-02, ).** On the removal miss, `removeRuleKeyEntries` now writes the rule's KEY FAMILY to  (`dumpRuleKeyFamily`, Rule 31): the owner, the failing whole key and edge key (raw `NameId` runs), every `normalizedEncodedKeys` / `remainingArgsOwners` entry that still lists the owner, and every edge carrying the failing NormKey under another argument set — the data that tells whether the install's and the removal's key enumerations diverged. A full-memory crash dump cannot answer that (two 44 GB dumps were analysed; only the removal chain came out, via a raw-stack return-address scan). Chain of the two 2026-09-02 firings: `applyEquiClasses → applyEquivalenceClassToCompactImplications → removeCompactExpansion → removeRuleFromHashMemory → removeRuleKeyEntries → removeRemainingArgsOwner → removeOwnerFromRun`, both on existence-implication compacts `!(=[u_X,u_2]) → existence3[u_1,u_X,u_3]` in the Peano batch.
**Trap campaign (2026-09-03,, ledger `docs/G72/g72_crash_ledger.md`).** Reading the section: the removal's key enumeration is a pure function of the rule text and the LB's own interners, so a single-threaded install / removal divergence is excluded; the defect is cross-thread or a lost entry. Six `#if CRASH_TRAPS` traps (per-slot scratch handover detector, foreign-claim detector, flush read-back, per-LB install ledger of every remaining-args edge checked at removal, owner-less erase audit) stayed silent through two shortcut and two full runs (production and RT-instrumented builds, CUDA); the only lines were the lazy `ruleStagings` construction touching every gen slot from one worker (address takes, benign). Two latent doors found on paper and closed under [D-337](40_decisions.md#d-337): the gen-scratch registry had no reserved slot (its `-1` fallback was worker `logicalCores - 1`'s arena and staging pool), and the claim handshake's `WorkerOwned` early return was not phase-gated. Neither was observed in use in the crashing configuration (resident-only steward).
**ROOT-CAUSED (2026-09-03, first firing after the traps were removed, Peano main, hash burst 10).** The permanent key-family dump showed the failing edge key as `[-842150451 × 10, 4, 42]` — `0xCDCDCDCD`, the arena poison. `remArgsEdgeKeyInto` built the `(argSet, NormKey)` edge key on the slot's gen-scratch arena and returned only its length; its four callers read the key back at a cursor mark captured BEFORE the allocation. `LbArena::alloc` aligns and pads to the next block when the request would straddle the current block, so whenever the edge build wrapped, the mark addressed the poisoned tail of the previous block: the install staged (or the removal looked up) garbage instead of the edge — whole key and owner present, edge absent. The wrap depends on where the slot's cursor sits inside its block, i.e. on which LBs the slot processed before (scheduling), hence the intermittency and the CUDA / RT sensitivity. Fixed under [D-336](40_decisions.md#d-336) (caller-owned stack buffer, `kMaxRemArgsEdgeKeyBytes`), general rule [I-222](30_invariants.md#i-222). The `[RULE-OWNER-*]` context prints, the key-family dump and the `G72_LEDGER` gate stay as permanent observability of the owner-list machinery.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
