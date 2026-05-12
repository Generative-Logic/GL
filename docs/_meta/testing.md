<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# In-tree unit-test harness `[DRAFT]`

> Status: Living. The harness is a doxygen-+-tests companion to the
> main code base; it gates every `main.py` invocation.

---

## Purpose

The harness exists for two reasons:

1. **Gate every `main.py` run on a fast pre-flight check.** Before any pipeline
 work or proof run, `main.py` runs `gl_quick.exe --unit-tests` and aborts on
 non-zero exit. Test runtime is sub-second today; the budget ceiling is 30 s.
 Regressions in the most-cited invariants surface in the first second of a
 `main.py` run instead of buried in a 10–60-minute prover invocation.

2. **Pin the per-function doxygen contracts.** The doxygen pass on
 `prover.cpp/.hpp`, `memory.cpp/.hpp`, `compiler.cpp/.hpp`,
 `filter.cpp/.hpp`, and `compressor.cpp/.hpp` documents many invariants
 (`I-2`, `I-3`, `I-9`, `I-12`, `I-13`, `I-22`, `I-26`, `I-27`, `I-28`,
 `I-30`–`I-34`, etc.). Tests in `src/tests/` exercise the testable subset
 directly so a future regression that breaks a contract trips the suite.

---

## File layout

```
GL_Quick_VS/GL_Quick/src/tests/
    test_harness.hpp        TEST(...) macro + ASSERT_* macros + registry
                            declarations
    test_harness.cpp        Registry storage + runAllTests() driver
    test_memory.cpp         NameMap, ChunkPool, KeyArena, Memory, HashMemory,
                            Mail, EquivalenceClass, IntNormalizedKey,
                            LocalMemoryValue, ExpressionWithValidity, plus
                            ExpressionAnalyzer-instance tests for
                            addOrigin's cap-full preference, negate
                            double-cancel, smoothenExpr idempotence, and
                            listLastRemovedArgs.
    test_compiler.cpp       generateBinarySequencesAsLists, generateAllPermutations,
                            getArgs (including the shallow-not-bracket-balanced
                            pin), extractExpression*,
                            makeAnchorSignature, findAnchorKey,
                            expressionIsSimple.
    test_filter.cpp         ContradictionItem default + parameterized
                            construction + int-to-bool conversion.
    test_compressor.cpp     CompressorNode default + populated; std::stable_sort
                            determinism (OPEN-13).
    test_conjecturer.cpp    Free helpers in conj:: namespace (repetitionsExist,
                            findMinMaxNumbers, findAllIds, defSetsEqual,
                            reorderNumbers, treeToStr*, shiftTogether,
                            connectExpressionSets, ...); int-path encode/decode
                            round-trip + IntConjBuf / IntDefSetMap / IntConnMap
                            layout; createMap + createMapAnchor structural
                            invariants; the full filter cascade (controlEquality,
                            checkInputVariablesOrder + 13 sub-helpers,
                            passesInPremiseFilter D-23 + 3 cnt-shape rules,
                            passesComplexityAfterExistence post-D-23 3-condition
                            rule, passesMaxDistinctAnchorValuesPerType, ...);
                            reshuffle pipeline + createReshuffledMirrored;
                            reformulation/negation paths; worker drivers (string
                            + int); generateOrConjectures smoke; Conjecturer
                            ctor smoke for Peano + Gauss; direct guards for
                            I-8 / I-9 / I-10 / I-11.

tests/
    __init__.py                       package marker
    test_harness.py                   TESTS registry + @register decorator +
                                      fixtures (make_state_*, make_proof_line,
                                      set_chapter_context, assert_failure,
                                      assert_pass, assert_chapter_meta_*) +
                                      run_all_tests() + entry point
    test_verifier_implication.py      implication / theorem / task formulation
    test_verifier_equality.py         equality1 / equality2 /
                                      symmetry of (in)equality
    test_verifier_structural.py       disintegration / expansion / recursion /
                                      anchor handling
    test_verifier_mirror.py           mirrored from / reformulated from /
                                      incubator back reformulation / externally
                                      provided theorem / variable copy /
                                      multiplied from
    test_verifier_contradiction.py    contradiction / vacuous truth
    test_verifier_or.py               or theorem / or disintegration /
                                      or convergence / or branch proven /
                                      or branch assumption
    test_verifier_integration.py      expansion for integration / premise
                                      element / validity name / reformulation
                                      for integration {and, >[bound], >[]}
    test_verifier_meta.py             chapter-level meta-checks (theorem goal
                                      reached, self-reference, anchor handling
                                      uniqueness/trace, contradiction trace,
                                      vacuous truth trace, origin, definition
                                      set consistency [D-41], origin chain
                                      termination [cycle])
    test_verifier_positive.py         positive sanity tests — well-formed
                                      inputs that MUST pass each checker
                                      (fixture-rig meta-check)
```

Conjecturer suite uses `conj::testing::Friend` (a friend class declared in
`conjecturer.hpp`'s private section, defined inside `test_conjecturer.cpp`)
to reach private methods (filters, int-path encode/decode, `createMap*`
statics, etc.) without widening the public API surface.

Test-side build integration:

- `GL_Quick_VS/GL_Quick/Makefile` — `SRCS` glob extended to include
 `src/tests/*.cpp`; the build rule's `mkdir -p $(@D)` lets object
 files drop into `build/tests/` cleanly.
- `GL_Quick_VS/GL_Quick/GL_Quick.vcxproj` — explicit `<ClCompile>` and
 `<ClInclude>` entries for every test file. Pure additions; no
 removals from the existing item-group.
- `GL_Quick_VS/GL_Quick/GL_Quick.vcxproj.filters` — new `Tests`
 filter group for IDE Solution-Explorer ergonomics.

---

## Macros

Hand-rolled, no external dependencies.

```cpp
TEST(suite, name) { ... body ... }

ASSERT_TRUE(x);
ASSERT_FALSE(x);
ASSERT_EQ(a, b);
ASSERT_NE(a, b);
ASSERT_LT(a, b);
ASSERT_GE(a, b);
ASSERT_THROW(stmt, ex_type);
```

Each `ASSERT_*` macro reports `[ASSERT] <file>:<line> <expr>` to stderr on
miss, then throws an internal sentinel exception caught by `runAllTests`. The
runner aborts on first failure with exit code 1.

The registry uses a Meyers singleton (function-local static
`std::vector<TestEntry>`) so static-initialisation-order issues across
translation units do not affect ordering.

---

## Python verifier-test layer

The Python side mirrors the C++ harness shape — a module-level
registry, an `@register` decorator, fixtures, and a `run_all_tests`
runner — without any external test framework.

```python
@register
def test_implication_premise_core_mismatch():
    state = make_state_minimal()
    line = make_proof_line(
        "(in[(s[a]),N])", "main",
        "implication",
        "(>[v1](in[v1,N])(in[(s[v1]),N]))", "main",
        "(=[a,N])", "main",   # wrong core: '=' instead of 'in'
    )
    assert_failure(check_implication, line, [line], state)
```

The decorator appends each test to a module-level `TESTS` list under
its qualified name (`tests.test_verifier_<group>.<fn>`). The runner
walks `TESTS`, catches `AssertionError`, counts failures, prints
`[verifier-tests] FAIL <name>: <msg>` per failure, and returns 0 on
all-pass or 1 on any failure (mirroring `gl_quick.exe --unit-tests`).
The cross-module registry is unified via a `sys.modules.setdefault`
alias at the top of `test_harness.py` so script-mode invocation
(`python tests/test_harness.py`) and package-mode invocation
(`python -m tests.test_harness`) populate the same `TESTS` list as
sibling test files.

### Fixtures

- `make_state_minimal` — empty `VerifierState`; used when a checker
 doesn't consult the global registry.
- `make_state_with_binaries(tags=("Peano",))` — loads
 `files/GL_binaries/*.json` and `files/config/ConfigVisu.json` via
 the real loaders (`load_gl_binaries`, `load_output_indices`,
 `load_input_indices`, `load_definition_sets`,
 `build_resolved_defsets_per_tag`); sets `current_gl_binary` /
 `current_resolved_defsets` from the first tag. Read-only fields
 are shared by reference across tests via a cached prototype; tests
 must not mutate `gl_binaries` / indices / defsets.
- `make_state_with_globals(thms)` / `make_state_with_externals(thms)`
 — fresh state plus a custom `global_theorems` / `external_theorems`.
- `make_proof_line(expr, ns, tag, *rest)` — terse `ProofLine` builder.
- `set_chapter_context(state, thm, chapter_type)` — mirror of
 `verify_chapter`'s transient context setup; lets per-tag tests
 dispatch checkers without going through `verify_chapter`.
- `assert_failure(checker, line, chapter, state)` /
 `assert_pass(...)` / `assert_chapter_meta_fail(...)` /
 `assert_chapter_meta_pass(...)` — assertion wrappers.

### Test inventory

- ~265 per-tag failure tests across 30 `TAG_CHECKERS` entries.
- ~53 chapter-level meta-check failure tests (theorem goal reached,
 self-reference, anchor handling uniqueness/trace, contradiction
 trace, vacuous truth trace, origin, definition set consistency,
 origin chain termination).
- ~22 positive sanity tests — well-formed inputs that MUST pass each
 checker, used as a fixture-rig meta-check. If the rig were broken,
 the failure tests would trivially "pass" without the verifier doing
 anything; the positives prove the rig is intact.

Total: ~340 tests, runtime <5 s.

---

## Entry points

- `gl_quick.exe --unit-tests`
 Direct invocation of the C++ harness. Returns 0 on all-pass, 1 on
 first failure.

- `python tests/test_harness.py`
 Direct invocation of the Python verifier-side harness. Imports every
 sibling `test_verifier_<group>.py` and runs all registered tests.
 Each `test_verifier_<group>.py` is also standalone-runnable via
 `python tests/test_verifier_<group>.py` for development (its
 `if __name__ == "__main__"` footer calls the same runner with only
 that group's tests registered).

- `main.py` prelude — double gate (C++ then Python)
 Runs `gl_quick.exe --unit-tests` via `subprocess.run`, then
 `python tests/test_harness.py` via `subprocess.run`, both before
 `run_modes.full_run`. Each gate is skipped silently when its
 artefact is missing (binary not built yet, harness file absent).
 Non-zero exit from either aborts `main.py` with that gate's exit code.

Direct `gl_quick.exe Peano` / `gl_quick.exe Gauss` invocations do NOT
trigger either gate — the harnesses are opt-in via flag/script entry
for those so direct debugging runs are unaffected.

---

## Performance budget

- Hard ceiling: **30 s** for the full suite.
- Per-test budget: 100 µs–500 ms (the harness prints per-test elapsed
 ms so a regression names itself).
- No real config-file reads except via `ExpressionAnalyzer{ "Peano" }`
 used in a few prover-side tests; that ctor loads the Peano JSON
 config in ~5–10 ms. All other tests use synthetic in-memory data.

Current measured runtime:
- C++ harness: 502 tests in well under 1 s
 (127 prover/memory/compiler/filter/compressor + 375 conjecturer).
 Conjecturer suite is the bulk; most tests are pure-helper invocations
 that complete in single-digit microseconds, plus ~80 tests that
 construct a `Conjecturer{"Peano"}` or `Conjecturer{"Gauss"}` (each
 ctor loads the per-batch JSON config in ~5-10 ms) and dominate the
 cumulative cost.
- Python verifier harness: ~340 tests in ~1–3 s (dominated by the
 one-shot config / GL-binaries load inside `make_state_with_binaries`'s
 prototype; the prototype is shared by reference across all tests).

---

## Sacred boundaries

- `verifier.py` — `[I-16](../30_invariants.md#i-16)` sacred. The
 Python verifier-test layer is **strictly black-box**: tests import
 `verifier.py` symbols (`ProofLine`, `VerifierState`, `TAG_CHECKERS`,
 `verify_chapter`, every `check_*`) and call them as-is, with no
 monkey-patching, no logic overrides, no source modifications. The
 verifier remains the user-controlled checker; the test layer
 exercises its rejection paths, not the other way around.
- `prover.cpp::performElementaryLogicalStep` hashburst dump —
 sacred infrastructure. The harness does NOT instrument the dump,
 add parallel traps, or modify the dump format / output path /
 call sites / lambda ordering / target-LB chain match.

---

## Adding a new test

### C++ side

1. Pick a test file by module: `test_memory.cpp`, `test_compiler.cpp`,
 `test_filter.cpp`, `test_compressor.cpp`, or
 `tests/test_<new_module>.cpp` for a new file.
2. If a new file: add a `<ClCompile>` entry to `GL_Quick.vcxproj` and
 the `Tests` filter in `GL_Quick.vcxproj.filters`. The `Makefile`
 glob picks it up automatically.
3. Write the test using `TEST(suite, name) {... }`. Pick a `suite`
 name that matches the module under test (e.g. `memory`, `compiler`,
 `prover`).
4. Build, run `gl_quick.exe --unit-tests`, confirm the new entry
 appears in the registered count.
5. Commit with `git add -A` (every commit a complete snapshot).

### Python verifier side

1. Pick the test file by tag group: `test_verifier_implication.py`,
 `test_verifier_equality.py`, `test_verifier_structural.py`,
 `test_verifier_mirror.py`, `test_verifier_contradiction.py`,
 `test_verifier_or.py`, `test_verifier_integration.py`,
 `test_verifier_meta.py`, `test_verifier_positive.py`, or
 `tests/test_verifier_<new_group>.py` for a new file. Stay black-box
 — no `verifier.py` source edits (I-16).
2. If a new file: the `sys.path` shim + `from tests.test_harness import
 register,...` is the same as in every existing file; copy-paste
 the header and standalone-runnable footer.
3. Write the test as a `@register`-decorated function. Use the
 fixtures (`make_state_*`, `make_proof_line`, `set_chapter_context`)
 and assertion helpers (`assert_failure`, `assert_pass`,
 `assert_chapter_meta_fail`, `assert_chapter_meta_pass`).
4. Run `python tests/test_harness.py`, confirm the new entry is
 counted in the final `M/M` summary.
5. Commit with `git add -A`.

---

## See also

- [I-13](../30_invariants.md#i-13) — `ChunkPool` static `char[]`
 requirement (tested indirectly by `keyarena_store_and_release`
 because the same chunk-allocator contract applies to the sibling
 `KeyArena`).
- [I-19](../30_invariants.md#i-19) — asserts are first-class
 (the harness's `ASSERT_*` macros mirror this stance for test-side
 checks).

---

## End-to-end run findings (2026-05-10)

Final closure ran `python main.py` end-to-end on a fresh worktree
checked out from `origin/main`. Result:

- **Prover stages all completed cleanly.** IncubatorPeano +0,
 Peano +26, IncubatorGauss +0, IncubatorGauss1 +0, Gauss +35
 (shared total 61). `files/theorems/proved_theorems.txt` carries
 40 theorems; `files/raw_proof_graph/` carries 212 chapters.
 This matches the expected `origin/main` baseline shape;
 `files/theorems/proved_theorems.txt` is the regression-claim
 source of truth.

- **HTML-generation stage crashes — pre-existing ordering bug in
 `run_modes.full_run`.** The crash:

 ```
  Traceback (most recent call last):
    File "...\run_modes.py", line 495, in full_run
      generate_full_proof_graph.generate_proof_graph_pages(
        configuration_visu, proc_dir=incubator_proc_dir,
        out_dir=incubator_full_dir, sibling_graphs=incubator_siblings)
    File "...\generate_full_proof_graph.py", line 2249, in read_theorem_list
      with open(map_path, "r", encoding="utf-8") as f:
  FileNotFoundError: [Errno 2] No such file or directory:
    'files\\processed_proof_graph\\global_theorem_list.txt'
  ```

 The bug: `run_modes.full_run` generates the **incubator** HTML
 pages FIRST and passes `main_proc_dir` as a sibling-graph entry,
 but the **main**-batch's `process_proof_graphs.py` step has not
 yet run, so `main_proc_dir/global_theorem_list.txt` does not
 exist. On a fresh checkout the read fails immediately. On
 re-runs in a directory with stale state from previous runs, the
 sibling read succeeds (loading stale theorem-list content) and
 the bug is masked.

 This is a finding from reading existing code, recorded here
 rather than fixed because the fix is outside the scope of this
 documentation pass and would change pipeline orchestration
 (architectural; needs separate review). A future pass can pick
 this up. Suggested remediation: either reorder so
 `process_proof_graphs.create_processed_proof_graph` runs before
 HTML generation for the incubator, or pass an empty
 `sibling_graphs` list on first run.

- **Verifier not run for the main-batch processed_proof_graph.**
 Because the main-batch's `processed_proof_graph/` was never
 created (the HTML stage crashed BEFORE reaching the main
 process-proof-graph step at `run_modes.py:502`), `verifier.py`
 has no main-batch directory to walk. The verifier reports
 `ERROR: directory not found`. The incubator's
 `processed_proof_graph/` IS present (212 chapters) but the
 verifier needs both directories to produce the cross-anchor
 audit.

- **Conclusion.** The doxygen + unit-test harness + per-phase tests
 landed cleanly. 51/51 unit tests pass in 17 ms, well inside the
 30-second budget. MSBuild Release|x64 builds with only the
 pre-existing C4267 warnings. The end-to-end `main.py` regression
 check is blocked at the HTML stage by the pre-existing ordering
 bug above; the prover stages themselves match the `origin/main`
 baseline.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
