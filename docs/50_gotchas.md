<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Gotchas `[DRAFT]`

> Things that keep biting us. Each entry describes a concrete failure mode — how it manifests, the root cause, how to detect it, and how to fix. Distinct from [`30_invariants.md`](30_invariants.md): invariants are the *rules*; gotchas are the *stories of what happens when a rule is forgotten or a subtle convention is missed*.
>
> Sections are grouped by concern — workflow, build, code-pattern, domain-specific. Add a new entry whenever a bug you spend > 30 minutes diagnosing turns out to be a recurrence of a known class.

---

## Workflow gotchas

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

**Manifestation.** A Peano `proved_theorems.txt` entry is semantically false — e.g. a universally-quantified claim that should fail for entities not in `N` but is marked proved by induction.

**Cause.** The historical induction scheduler did not check that the induction variable is in `N`. For bound variables that only appear in negations, bare equalities, or existence heads, the typing is not derivable from the chain — the theorem's effective universe widens from `N` to "everything".

**Fix.** Implement the induction-typing sub-theorem per [`docs/induction_typing_plan.md`](induction_typing_plan.md). See [I-18](30_invariants.md#i-18), [D-15](40_decisions.md#d-15-induction-typing-sub-theorem--design-approved-2026-04--pre-c21386e). In progress.

---

<a id="g-22"></a>
### G-22 — CE filter j-copy requirement

**Manifestation.** A new fact file is written by hand, using only `i`-prefixed constants (no `j`-copies). CE filtering mysteriously underfires — some conjectures that should be refuted slip through.

**Cause.** `generateEncodedRequestsStaticCE` needs *distinct* fact entries for its rule-firing pattern. Without j-copies (parallel constants), rules that require two different-looking arguments to match never fire on the table.

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
### G-26 — `proved_theorems.txt` is authoritative, not `global_theorem_list.txt`

**Manifestation.** Two runs produce different `global_theorem_list.txt` — method labels shift (`direct` vs `induction`) or column-3 references differ. Panic ensues; "regressions".

**Cause.** `global_theorem_list.txt` is encoding-sensitive to how the processor renames theorems. Method-label shifts are not theorem losses — they reflect processor's interpretation, which can drift.

**Fix.** The regression-claim source of truth is `files/theorems/proved_theorems.txt` — the set of theorems the prover actually produced. Use `diff` on this file to detect real losses.

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

**Manifestation.** A refactor changes the chain's shape (e.g. you want a fourth tuple element, or change `std::get<1>` from `vector<string>` to `vector<int>`). The cache stores the OLD shape. On a hit, callers receive the old shape and either crash or silently misbehave. Symptoms: byte-identity regression in `reshuffled_theorems.txt`, random segfaults at tuple access sites, or filters that pass when they shouldn't.

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

**Cause.** `_extract_args(expr)` ([`verifier.py`](../verifier.py)) uses `re.search(r'\[([^\]]*)\]', expr)` which matches **only the first** `[…]` group. For a compound source expression like `!(>[v6](in[v6,N])!(in2[v6,i1_copy,s]))`, the outermost group is `[v6]`; the `i1_copy` nested deeper inside `(in2[v6,i1_copy,s])` is invisible. Filters of the form `lambda src: b in _extract_args(src)` therefore mis-conclude that the source does not carry `b`.

**Detection.** A `variable copy` (or other tag) that traces back through compound expressions: trace-back termination at a row that obviously cites the copied variable. The verifier reports a `variable copy` failure with no other apparent cause. Add a temporary print at the trace-back filter to confirm `_extract_args(src)` is dropping the nested arg.

**Fix.** For trace-back filters that need to detect a variable anywhere in a compound expression, use `_extract_all_args` ([`verifier.py`](../verifier.py)) — added for `check_variable_copy`. The plain `_extract_args` stays correct semantics for the 30+ atom-level callers that genuinely want the outermost group; do not widen it globally. If a new tag's trace-back encounters this issue, add the recursive helper to that filter site only.

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

## See also

- [`30_invariants.md`](30_invariants.md) — the rules that, when forgotten, produce the gotchas above.
- [`40_decisions.md`](40_decisions.md) — why the rules exist.
- — the `feedback_*` files are the living history of gotcha discoveries.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
