<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Release 13 preparation ledger

Living task document (Rule 32). Branch created 2026-09-04 from `main` tip
. Nothing on this branch reaches `main` without the maintainer's
explicit approval; the candidate branch is
published only by `release_public.py` from `main` afterwards.

## Objective and acceptance

Prepare release 13 (`v0.11.0`, the unreleased stub at the top of
`RELEASE_NOTES.md`). Acceptance: every work item below is done with its
evidence recorded, the unit-test gates pass, the full pipeline and the
shortcut pipeline verify with zero failures on every route named in item 7,
the Lean corpus kernel-checks, and the maintainer approves the squash.

## Hard constraints and maintainer decisions

- Work only on this branch. No squash, no merge, no candidate branch without
 the maintainer's word.
- Rule 24: every run goes through `main.py`. Rule 25: every build is a full
 rebuild. Rule 29: long runs get one completion watcher, no polling.
- Lean policy (maintainer, 2026-09-04): GL must run on a customer PC without
 Lean. When the Lean toolchain is present the run exports and kernel-checks;
 when it is absent the run prints one notice and produces no Lean export.
 No crash, no fallback proof.
- GPU policy (maintainer, 2026-09-04): the processor route is the default for
 every run. `--GPU` selects CUDA for the whole run; if the user chose `--GPU`
 and anything the CUDA route needs is missing (device, driver, runtime), the
 run asserts. Whatever runtime component the CUDA route needs on a customer
 PC ships with the release like the other binaries.
- Agents run with `--GPU` (new the project conventions / AGENTS.md rule, lands with item 6).
- Relative paths in shell commands from the repository root; no `cd` inside
 compound commands (the shell's working directory persists between calls).
- Runs strictly one at a time, on any host combination (maintainer,
 2026-09-04, after the agent overlapped the WSL chain with the Windows
 shortcut pair; the WSL chain was stopped during its copy and re-run
 afterwards).

## Work items (the maintainer's list)

| # | Item | Status |
|---|---|---|
| 1 | HTML visualiser: `preorder[N,*,a,b]` renders as `a | b`, `strictOrder[N,+,a,b]` as `a < b`, negations as `∤` / `≮` / `≰`, in captions and titles of the shortcut graph | coded + unit-tested; HTML regeneration check pending |
| 2 | Lean export still works on the current graphs (three exports, contract tests, `lake build`, FTA compile) | exporter runs, but all three tracked selections pin the previous graph generation (see evidence); needs the re-pin = item 3's live builder |
| 3 | Lean export runs inside the normal run for the non-incubator batches (Peano + Gauss in the full run, FTA in the shortcut run) when Lean is installed | DONE standalone: main graph 64 + 30 theorems green (9.9 s); FTA 79 theorems / 2,782 rows green against the externals snapshot alone (10.2 s); pipeline-run gate under item 7 |
| 4 | Every HTML proof page links to a Lean page of the same proof in the same visual style; the Lean pages travel with the proof-graph folder to the website | DONE: 94 twin pages in the main graph, 79 in the shortcut graph; checked in the browser |
| 5 | README third-party notices gain the Lean line | done |
| 6 | Processor route is the default for every run; `--GPU` selects CUDA; the project conventions and AGENTS.md rule: agents run with `--GPU` | coded (Rule 33, D-338, docs, user SwDD); rebuild + unit tests running; run gates under item 7 |
| 7 | Processor runs on Windows and on Ubuntu reproduce the CUDA run (full and shortcut) | open |
| 8 | Ubuntu can use the GPU: Makefile CUDA build, the Windows-only gates replaced by a CUDA build flag, install commands for the missing software | approved; the maintainer installed the WSL software (2026-09-04) |
| 10 | Remove the permanent G-72 trap instrumentation (the intermittent `removeOwnerFromRun` assert was root-caused and fixed on `main`, D-336 / D-337; the instruments kept "permanently" for it are obsolete): the `G72_LEDGER`-gated ledger writer and its  output, the key-family dump on the first firing, the `[TRAP-*]` lines. Rule-14 hashburst dump untouched. Rebuild, unit tests, shortcut and full gates after removal | coded 2026-09-04: `G72_LEDGER` + `g72Ledger` + 7 calls, `dumpRuleKeyFamily` + `test_rule_key_family.cpp` (project entries too), the `[RULE-OWNER-REMOVE-*]` prints, `RuleIndexOp`'s diagnostic-only fields and their setters removed; both asserts untouched; `main.cpp`'s abort minidump handler kept (general). G-72 gotcha carries the closure. Gates: Windows rebuild clean + 1624/1624 (the two dump tests gone); WSL CUDA build 1624/1624, full run 141,136 / 0 and shortcut 9,217 / 0 on the post-removal source, artifacts identical to the pre-removal references |
| 9 | MPU booklet becomes **MPU 0.2 (GPU Edition)**, everywhere the booklet is named: the main runtime consumer, phase 2 of the hashburst (the engine itself), is now demonstrated to split across independent threads — from 32 logical cores to GPU threads, ten times faster so far, with the campaign just started. Plain, standard, concise English; no flourish | added 2026-09-04 (maintainer) |

### Item 1 — math symbols

`generate_full_proof_graph.py::_htmlify_readable` rewrites
`(preorder[N,+,a,b])` to `a ≤ b` with a regex that ignores the operation
slot, so `preorder[N,*,a,b]` (divisibility) also shows `≤`; `strictOrder`
has no rule and shows raw MPL; a leading `!` is not consumed, so negations
show `!a ≤ b`. Plan: one symbol table keyed by (operator, operation slot,
negated) → `≤ / ≰`, `| / ∤`, `< / ≮`, and for `strictOrder[N,*,…]` the
proper-divisor form `a | b, a ≠ b`. Same table for the `<title>` caption.
Regression: a Python test over the four shapes plus negations.

### Item 3 — Lean export inside the run (design)

Today every export is driven by a tracked selection file that pins the
theorem-list hash, the source range, the expected coverage and, for Gauss
and FTA, the dependency certificates by hash and the cited theorems by
source index. A normal run produces a fresh graph, so those pins cannot be
tracked ahead of time. Design:

- A live selection is derived from the run itself: the source range is the
 batch's contiguous block in `global_theorem_list.txt` (Peano rows, then
 Gauss rows), the theorem-list hash is computed, `expected_coverage` is
 absent (recorded instead of asserted — a pinned release corpus keeps the
 assert), exclusions are empty.
- Gauss depends on the Peano certificate just written; the required Peano
 theorems are the ones Gauss actually cites, resolved by content (the
 builder already resolves citations alpha-equivalently).
- FTA (shortcut run) depends on the Peano and Gauss certificates of the
 last full run on disk under `files/lean_export/`; cited external theorems
 are found by content, and the four known adaptations (premise
 permutation, definition-identical head aliases) are found by the same
 structural search the validator already performs, instead of being
 declared by index.
- Outputs are run artifacts under `files/lean_export/` (gitignored like
 `full_proof_graph/`): the certificates, one Lake project, the manifests.
 The tracked `lean_export/` + `proof_export/generated/` release package is
 still produced by `release_public.py` from the same code path.
- Stage order in `full_run`: processed graph → Lean export → HTML (so the
 Lean pages exist) → verifier → kernel check. Same in `shortcut_run`.
- Lean toolchain absent: one notice line, no export, no Lean pages, no link.
 Toolchain present: export, `lake build`, FTA compile; a kernel failure
 fails the run.

### Item 4 — Lean pages in the HTML export (design)

Per theorem chapter `chapter<N>.html` a companion `chapter<N>_lean.html`
with the same stylesheet, navigation bar and footer: the theorem's Lean
namespace (`peano_source_NNN` and its private helpers) sliced from the
generated module, one row fact per proof step with the step's source line,
and links to the raw module and certificate copied into
`full_proof_graph/lean/`. The chapter navigation gets a `Lean 4 proof`
link; the index gets one per theorem. Because the pages live inside
`full_proof_graph/`, the website copy (`proofs_gauss/`) carries them.

### Item 6 — processor default (design)

`parameters.hpp` `use_gpu` defaults false; the explicit `use_gpu: true` in
`ConfigIncubatorPeano1.json` goes; `main.py --GPU` passes
`--phase2-backend cuda` to every batch of the run, so a `--GPU` run equals
today's default run exactly (every batch CUDA). `--phase2-backend` stays as
the explicit oracle override. Shortcut default becomes the processor route.
Docs: `04_configs.md`, D-329 follow-up decision, prover chapter, user SwDD
`parallel.html`, README, release notes.

### Item 8 — Ubuntu GPU (design)

The CUDA code is gated by `_WIN32` in four places (`prover.cpp` includes and
the CUDA block, `tests/test_phase2_projection.cpp` three gates); the `.cu`
files are Visual Studio-only. Plan: a build flag `GL_CUDA` defined by the
Visual Studio project and by the Makefile under `USE_CUDA=1` (opt-in, so
the macOS CI build stays CUDA-free); the four gates switch from `_WIN32` to
`GL_CUDA`; the Makefile compiles the two `.cu` files with `nvcc`
(`-arch=sm_89`, the same targets as the project) and links `cudart`. A
`USE_CUDA=0` build reaches the same processor route as today's Linux build.

Missing software on the Ubuntu host (user-run, needs sudo — the agent never
changes system settings):

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-3 libmimalloc-dev build-essential
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
```

Facts behind the list: WSL Ubuntu 26.04, driver 596.08 (CUDA 13.2 driver
API), no `nvcc`, no `/usr/local/cuda`, no elan, `apt` only offers
`nvidia-cuda-toolkit` 12.4, Python 3.14 with `regex` present, g++ 15.2.
The WSL-Ubuntu repository ships driver-free toolkits (never install the
`cuda` or `cuda-drivers` meta-packages under WSL). CUDA 13.x supports GCC 15
as host compiler. 13.3 matches the Windows project; a 13.3-built binary
already runs on this driver on Windows.

## Proposals (need the maintainer's yes before any work)

- **P1 README build prerequisites for CUDA.** The Visual Studio project
 imports the CUDA 13.3 build customisation, so the solution does not build
 without the toolkit, and the README says nothing about CUDA. Add a "GPU
 (optional)" section: Windows toolkit 13.3 to build, Linux `USE_CUDA=1`,
 `--GPU` at runtime, what ships in the release for the CUDA route.
- **P2 FTA proofs on the website.** The site has Gauss and Incubator proof
 graphs only; the release theme is the FTA shortlist. Add a third "Proofs"
 entry (`proofs_fta/` from `files/shortcut/full_proof_graph/`). Separate
 repository and a live deploy — only on approval.
- **P3 Refresh the shortcut externals snapshot from the release full run.**
 The SwDD records that the FTA export is pinned one Peano generation behind
 and that one cited Peano theorem (`0 + m = n → n = m`) is no longer proved
 by the current Peano batch. With the live export (item 3) that drift is a
 hard failure, so the snapshot has to be refreshed now; a lemma lost in the
 refreshed shortcut run is a real regression to fix, not to hide.
- **P4 Release notes.** One section for this window: processor default and
 `--GPU`, Lean export inside the run with browsable Lean pages, the math
 symbols, the Linux CUDA build. Version stays `v0.11.0`.
- **P5 README text refresh.** The sentences "Lean export is not part of
 `main.py`" become false after item 3; the status table and the reference
 timings (now processor by default) need the gate numbers.
- **P6 AGENTS.md sync.** AGENTS.md is two lines behind the project conventions (verifier
 tally, the `expandToBaseForm` symbol); sync while adding the `--GPU` rule.
- **P7 Fresh Ubuntu clone.** The WSL clone `~/gl` no longer exists; items 7
 and 8 need a fresh copy and build (rsync of this worktree). No approval
 needed, noted for the time it costs.

## Current measured state

| Gate | Value | Source |
|---|---|---|
| `main` tip | | this branch's base |
| Full run, CUDA | 141,136 checks / 0 failures, ~287 s wall | G-72 ledger (2026-09-03) |
| Shortcut, CUDA | 9,217 / 0, prover ~42 s | G-72 ledger |
| Tracked Lean corpus | `lake build` 7 jobs OK in 15 s (cached objects) | measured 2026-09-04 |
| Shortlist | 79 theorems in `global_theorem_list.txt`; 65 conjecture rows | on disk |
| Website clone | `gl_webpage` = `origin/main` | fetched 2026-09-04 |

## Final state (2026-09-04, FROZEN)

Every item (1–10) and every decision is implemented and gated; nothing is merged. The maintainer answered
the five open decisions with "agreed to all. finish"; the closing work is
recorded under *Maintainer decisions of 2026-09-04 (third message)* below.
What stays with the maintainer: the push of the website clone (`proofs-fta`
folder and the nav entry, prepared and uncommitted in
), the squash to `main` (GL squash
playbook, `swdd_renumber.py` for the three `D-pending` entries), and running
`release_public.py` from `main` afterwards. Durable architecture lives in the
agent SwDD (`10_pipeline/10_external_proof_export.md`, `40_decisions.md`
D-pending entries, `50_gotchas.md` G-72 closure), the user SwDD, and the MPU
0.2 booklet. This document is frozen.

## Item 7 run matrix (same results on both routes, both hosts)

| Run | Host / route | Verifier | Wall | Lean inside the run | Artifacts vs reference |
|---|---|---|---|---|---|
| full 1 | Windows, `--GPU` (all 7 batches CUDA by override) | 124,224 main + 16,912 incubator = 141,136 / 0 | 287.5 s | peano_live_64 (140 chapters, 3,314 rows) + gauss_live_30 (34 / 1,550), `lake build` green; 94 twin pages | REFERENCE (`win_gpu_full`) |
| full 2 | Windows, processor (bare `main.py`, all 7 batches `phase2_backend=cpu`) | 141,136 / 0 | 515.4 s | same 64 + 30 theorems, `lake build` green | IDENTICAL: 6 theorem files, 206 + 1,466 + 174 HTML/Lean files byte-equal (CRLF-normalized) |
| full 3a | WSL Ubuntu, processor (`make`, 1616/1616 unit tests — the 10 CUDA tests are compiled out) | CRASH: `gl_quick IncubatorPeano1` SIGSEGV around prover iteration 11 (1,143 active LBs, `backend=cpu`); no core (dumps were disabled) | — | — | intermittent, see 3b |
| full 3b | WSL Ubuntu, processor, same flags plus `-g`, core dumps armed | 141,136 / 0 | 475.0 s | Lean export inside the run (elan toolchain fetched on first use), `lake build` green | vs `win_gpu_full`: 6 theorem files, 1,466 incubator files, 174 shortcut files and 201 of 206 main files identical (CRLF-normalized), including every generated Lean module and `lean_manifest.json`; the 5 remaining files (2 selections, 2 certificates, `lean_gauss_manifest.json`) differ ONLY in byte-hash pins of inputs whose line endings differ between the hosts (`theorem_list_sha256`, `selection_sha256`, `gl_binary_sha256`, per-chapter `source_sha256`) — every theorem, row and definition is equal |
| full 3c | WSL Ubuntu, processor, `-g` binary, core dumps armed, repeated | 141,136 / 0 | 466.9 s | green | no core, no crash: the single SIGSEGV of 3a did not recur in the armed re-runs (unexplained, intermittent; recorded as an open risk) |
| full 4 | WSL Ubuntu, CUDA (`make USE_CUDA=1`, 1624/1624 unit tests, `main.py --GPU`, every batch `phase2_backend=cuda`) | 141,136 / 0 | 257.5 s (fastest of the four) | Lean export + `lake build` green inside the run | vs `win_gpu_full`: identical except the same 5 hash-pin files |
| shortcut 4 | WSL Ubuntu, CUDA (`--shortcut --GPU`) | 9,217 / 0 | 48.8 s | fta_live_79 green | vs `win_gpu_shortcut`: identical except the 2 hash-pin files |
| conclusion | Windows CPU, Windows GPU, Ubuntu CPU, Ubuntu GPU — full and shortcut | every run 141,136 / 0 and 9,217 / 0 | | every run exported and kernel-checked its Lean corpus | proof output identical across all four routes; the only cross-host delta is the byte-hash pin of line-ending-dependent inputs inside the Lean selection/certificate files (decision for the maintainer: keep raw-byte pins, or hash CRLF-normalized bytes) |
| shortcut 3 | WSL Ubuntu, processor (`--shortcut`) | 9,217 / 0 | 170.8 s | fta_live_79 green, `lake build` on Linux | vs `win_gpu_shortcut`: 2 theorem files and 172 of 174 files identical incl. `FTA.lean` and its manifest; the 2 others (`selection_fta.json`, `certificates/fta/certificate.json`) differ only in line-ending byte-hash pins |
| shortcut 1 | Windows, `--shortcut --GPU` | 9,217 / 0 | 70.9 s | fta_live_79 (109 chapters, 2,782 rows) green, 79 twin pages | REFERENCE (`win_gpu_shortcut`) |
| shortcut 2 | Windows, `--shortcut` (processor) | 9,217 / 0 | 205.3 s | same, green | IDENTICAL: 2 theorem files, 174 HTML/Lean files |
| shortcut 3 / 4 | WSL processor / WSL CUDA | pending | | | |

## Evidence log (append-only)

- 2026-09-04 — survey done, branch created, ledger written. No code changed.
- 2026-09-04 — CUDA runtime shipping question (item 6 policy). `dumpbin
 /dependents gl_quick.exe` lists no CUDA DLL: the project links the static
 CUDA runtime (the Visual Studio CUDA integration default), which loads
 `nvcuda.dll` and `nvcudart_hybrid64.dll` at runtime — both are installed by
 the NVIDIA display driver (`System32` and the driver store), never by us.
 The release therefore ships nothing extra for the CUDA route; a customer
 PC needs an NVIDIA driver whose CUDA driver API is 13.x or newer. Without
 it, `--GPU` asserts at the startup device contract. The import table holds
 only `mimalloc.dll` (already shipped) and the Windows CRT.
- 2026-09-04 — item 2: the tracked Lean selections cannot be rebuilt from
 the graphs on disk (2026-09-03 generation). Peano export asserts at the
 theorem-list hash pin; by content, 34 of 64 Peano rows, 6 of 29 Gauss rows
 and 7 of 79 FTA rows (plus two method labels) sit at different indices
 than the tracked certificates record. The tracked package is still
 internally consistent (`lake build` passes on it). Conclusion: item 2 is
 satisfied only through a re-pin against the current generation, which is
 the manual form of item 3's live selection — so item 3's builder is written
 first and used as the re-pin tool.
- 2026-09-04 — item 1 coded: `_ORDER_SYMBOLS` + `_apply_order_symbols` in
 `generate_full_proof_graph.py`, applied in `_htmlify_readable` and the
 `<title>` caption; eight unit tests in `tests/test_html_readable_symbols.py`
 pass; SwDD `07_html_export.md` updated.
- 2026-09-04 — item 3 coded: `proof_export/live.py` (live selections from
 the anchor blocks of the theorem list, computed hash, coverage recorded,
 same-list imports by content, external citations by alpha match then by the
 validator's structural search with inferred head aliases; per-mode output
 folder `<full_proof_graph>/lean_export/` with the Lake skeleton copied
 from the tracked project; `lake build` + FTA compile with a log; toolchain
 absent → one notice); `certificate.py` gained `theorem_citations`,
 `binary_entries_with_overrides`, the non-asserting
 `theorem_adaptation_solutions`, and an optional `expected_coverage`;
 `lean.py` classifies corpora by id prefix (`corpus_kind`) and asserts the
 tracked counts only for tracked ids; `run_modes` calls the export after
 the processed graph and before the HTML in both modes. 12 unit tests in
 `tests/test_proof_export_live.py`. First live run on the 2026-09-03 main
 graph: Peano 64 and Gauss 30 theorems certified (the tracked corpus had
 29 Gauss theorems), Lean modules rendered, `lake build` FAILED on Peano
 source 60's induction-condition chapter: an `expansion` row turns a
 negated existence (`¬ existence3`) into a compact implication
 (`implication1247`, the I-209 existence-implication compact) and the
 writer replays it as a one-head unfold, which does not close
 (`¬¬∀ x, N x → ¬ succ x v1` versus `∀ w1, succ w1 v1 → ¬ N w1`). A real
 writer gap on the current generation — item 2's answer is "no, not until
 this shape is rendered".
- 2026-09-04 — item 8 coded: `GL_CUDA` build flag (Visual Studio project
 defines it; Makefile `USE_CUDA=1` defines it, compiles `src/gpu/*.cu` and
 `src/tests/*.cu` with nvcc for `sm_89` + `sm_120`, links `cudart`); the
 five `_WIN32` CUDA gates in `prover.cpp` and `test_phase2_projection.cpp`
 switched to `GL_CUDA`; a CUDA selection on a non-CUDA build asserts at
 prover construction. WSL now has CUDA 13.3 (`/usr/local/cuda`), elan and
 mimalloc (maintainer-installed).
- 2026-09-04 — item 3 first green: after the writer case and the Peano
 renderer receiving the certificate definitions, the live main-graph
 export passes — `peano_live_64` (64 theorems, 3,107 rows), `gauss_live_30`
 (30 theorems, 1,550 rows), `lake build` green, 9.9 s end to end
 (commit `release13 items 3, 8, 9`). FTA export next, via the shortcut run.
- 2026-09-04 — first shortcut run through the hook (`--GPU`, 9,217 checks
 expected): the FTA export asserted — the citation pre-pass counted an
 integration-goal template dependency (`(AnchorPeano[...])_integration_goal`)
 as an external theorem; `theorem_citations` now yields ordinary-scope
 citations only (scoped templates are resolved by the chapter certificate
 from the chapter's own rows). Second finding: the HTML generator wipes its
 output folder wholesale and deleted the run's `lean_export/`; the wipe now
 keeps that child. Both fixed; FTA export re-run pending.
- 2026-09-04 — FTA export, second blocker: shortcut source 17 cites
 `∀ w1 ∈ N, or2[w1,i0,N,s]`, Peano proves `∀ v1 ∈ N, or1[N,v1,s,i0]` —
 the same disjunction with permuted parameters and swapped disjuncts (the
 current Peano generation's `or1` differs in parameter order from the
 generation the tracked adaptation was written against), which the
 identical-elements alias check cannot express. Design (agent, flagged to
 the maintainer): a third resolution stage compares registry-independent
 base forms (`certificate.base_form_equivalent`; details in
 `10_external_proof_export.md` and D-339); the
 declared adaptation carries `base_form: true`; Lean interface unchanged.
 Unit test reproduces the or2/or1 case.
- 2026-09-04 — FTA export, third finding: after the base-form stage (and a
 deterministic choice among duplicate statements of one corpus — Peano
 proves `∀ v1 ∈ N, or1[N,v1,s,i0]` twice, sources 61 and 62), shortcut
 source 41 cites `w1 + 1 = w2 → 1 + w1 = w2`, which the current Peano batch
 does not prove — it is an external of the shortcut's tracked snapshot.
 The maintainer's ruling (see the decisions block above): that is by
 design; the shortcut is independent of the main path. An externals
 refresh written at this point was removed again unrun. Consequence for
 the export: the FTA certificate's dependency must be the externals
 snapshot itself, not the full run's certificates.
- 2026-09-04 — externals-only FTA export implemented ("go, go"):
 `live.external_theorem_lists` turns the tracked snapshot into two
 certificate-shaped theorem lists; `export_shortcut_graph` depends on them
 alone; `certificate.base_form` unfolds exactly the spontaneous compacts
 (as `expandToBaseForm`) and cancels double negation; the FTA Lean project
 is self-contained (`lean.write_lean` layout without
 `definition_isolation`; `lake build` is the check); an unresolvable
 citation from an `or theorem` row is dropped like the chapter builder does
 (D-217). The externals refresh was removed; its test file was deleted
 through `git rm` (plain `rm` is policy-blocked for the agent). Docs: SwDD
 live section, D-339, READMEs.
- 2026-09-04 — externals-only FTA export, certificate stage green: every
 citation of the 79 shortlist theorems resolves against the two externals
 lists (alpha or base form; the or-theorem companion drop applied). Two
 Lean-writer gaps surfaced on the current shortcut generation, both fixed
 in `lean.py`: (1) a `compound_project` row whose source is guarded by a
 witness the projected conjunct never mentions lost that guard
 (`inherited_witnesses` now derives from the ordered guard set); (2) a
 contradiction row with BOTH contradictory facts inside the contradiction
 scope (the renderer only knew scoped-versus-main) — now enters the scope
 once and instantiates every scoped fact there; (3) a `compound_project`
 from a De Morgan disjunction `¬(¬A ∧ ¬B)` whose conclusion is the
 implication compact `implication<N>[…]` (= `¬A → B`) rather than the
 spelled-out implication — the compact is unfolded first and its
 instantiated definition is the implication the classical projection
 proves (seven such rows in the current shortlist graph); (4) an
 induction step whose helper assumes the current value's membership
 (`in[v1,N]`) received the theorem's own typing premise (about the
 theorem's variable, not `induction_m`) — the step now derives
 `N induction_m` from the successor edge through the anchor's
 successor-function closure; (5) the or-elimination renderer hardcoded the
 anchor projection paths, the closure compact's name and the or's disjunct
 order from the old registry — all now read off the corpus' definitions
 (`_left_associated_projection`). First self-contained FTA `lake build`
 reached Lean with 4 errors in 2,782 rows, all in these shapes.
- 2026-09-04 — FTA export GREEN (commit `release13 item 3 (FTA)`; its
 message says 2,835 rows — the correct live figure is 2,782 rows, 109
 chapters, 79 theorems): every citation resolved against 36 Peano-anchored
 and 24 Gauss-anchored externals, self-contained `lake build` green,
 10.2 s end to end. Shortcut HTML regenerated with 79 Lean twin pages.
- 2026-09-04 — item 7/8 finding: the Linux PROCESSOR-ONLY build (`make`,
 `USE_CUDA=0`) does not link — `src/gpu/phase2_sealing.cpp` calls
 `CudaPhase2EvaluationBuffer::download*`, defined only in the `.cu`
 translation unit. Broken on `main` since the GPU full-run squash (the
 Linux build gate was not run since; the shipped Linux binary would have
 been missing again, as the old `tail -3` masking once let happen). Fix:
 the sealing unit is CUDA-route code and is now compiled under `GL_CUDA`
 only. The WSL processor phase is re-run after the fix. Second finding on
 the re-run: the processor-only binary's unit tests aborted — the
 backend-default test constructs an analyzer with an explicit CUDA route,
 which the new construction-time assert refuses on a non-CUDA build (the
 intended contract); that part of the test is now under `GL_CUDA`. Windows
 rebuild + 1626/1626 after each fix.
- 2026-09-04 — P1 done: README "GPU (optional)" (driver, `--GPU`, what a
 missing piece does, Windows toolkit 13.3, Linux/WSL toolkit commands and
 `make USE_CUDA=1`, macOS) and "Lean 4 (optional)" sections.
- 2026-09-04 — item 4 coded: `lean_pages.py` (export view keyed by source
 index, corpus selection by theorem-list hash, per-theorem slice of the
 generated module, highlighting, page and link renderers) hooked into
 `generate_full_proof_graph.py` (index summary, `[Lean 4]` index links,
 `Lean 4 proof` chapter navigation, `chapter<N>_lean.html` twins); 9 unit
 tests in `tests/test_lean_pages.py`; SwDD `07_html_export.md` section.
- 2026-09-04 — item 8 build gates: Windows full rebuild with `GL_CUDA`
 clean, 1626/1626 unit tests; WSL snapshot (`~/gl_r13`, rsync copy per the
 snapshot script) built with `make USE_CUDA=1 USE_MIMALLOC=1` in 2 min 7 s
 (nvcc 13.3, g++ 15.2), 1626/1626 unit tests on Linux. The Linux CUDA run
 gates follow under item 7.
- 2026-09-04 — item 9 done in the booklet (title, brand, stamp, thesis key
 block "What 0.2 adds", §2, §3, §5, floorplan label, verdict, license line),
 README (booklet name, status row, MPU paragraph), SwDD navigation,
 the project conventions / AGENTS.md Rule 20,.
- 2026-09-04 — item 6 coded: `use_gpu` default false, incubator pin removed,
 `main.py --GPU`, shortcut default processor, C++ default test inverted,
 Rule 33 in the project conventions + AGENTS.md (AGENTS.md's two stale lines synced),
 D-338, I-211 clause 1, `04_configs.md`,
 prover chapter, GPU cookbook, user SwDD `parallel.html`, README quick
 start, release-notes CUDA bullet, run rule.

## Maintainer decisions of 2026-09-04 (second message)

- Lean export output lives inside each mode's proof-graph folder:
 `files/full_proof_graph/lean_export/` (Peano + Gauss) and
 `files/shortcut/full_proof_graph/lean_export/` (FTA). The incubator graph
 gets none: its `incubator back reformulation` rows are unsound by design
 and every external export rejects them.
- Items 3, 4 and 8: approved as designed above ("just make it run
 automatically").
- P1 accepted and broadened: the README gets everything a customer must
 know and do to use an NVIDIA GPU (driver, `--GPU`, Windows toolchain,
 Linux toolchain and build flag, what a missing piece does).
- P2 accepted: the website gets a `proofs-fta` folder (third "Proofs" entry).
- P3 as first read was WRONG (maintainer, 2026-09-04, third message): the
 FTA shortcut is 100 % independent of the main path. Its only theorem input
 is the tracked externals snapshot, which does not have to match the latest
 main run. No externals refresh; the FTA Lean export never reads the full
 run's export — its dependency is the externals list itself, each cited
 external resolved by content and entering Lean as a proposition parameter
 in a self-contained Lean project.
- The Ubuntu host is never a clone: the project is snapshotted and copied
 (`.scripts/wsl_snapshot.sh` / `snapshot_project.py`), as for every
 earlier release gate. P7 is void.

## Maintainer decisions of 2026-09-04 (third message: "agreed to all. finish")

1. **Tracked Lean release package retired.** The release ships the two run
 exports (`files/full_proof_graph/lean_export/`,
 `files/shortcut/full_proof_graph/lean_export/`).
 `release_public.py::check_live_lean_exports` requires both with a green
 `kernel_check.log`, re-runs `lake build` in each, runs
 `tests.test_proof_export_lean`, and the copy is byte-checked
 (`LEAN_EXPORT_RESULT_PATHS` = the two folders). `snapshot_project.py`
 overlays the same two folders and skips every `.lake/` directory
 snapshot-wide. Removed from git: `proof_export/generated/`, the seven
 selection files, `lean_export/GLExport/Generated/`, the root
 `GLExport.lean`, the three manifests. Kept: the Lake skeleton
 `proof_export/live.py` copies. Two historical certificates moved to
 `tests/fixtures/lean_certificates/` (`lean_full_65.json`,
 `lean_gauss_main_29.json`) for the renderer tests. The FTA and
 module-scanning tests read the live exports (skipped without one); their
 row counts are derived from the live certificates
 (`_expected_rows`) instead of literals pinned to one generation.
 Evidence: `tests.test_proof_export_lean` 52/52, all Python suites 90/90,
 `.scripts/tests` 7/7, `release_public.py --dry-run` green through the
 Lean gate.
2. **Release notes (P4)** written into `RELEASE_NOTES.md` v0.11.0 as two
 paragraphs (processor default / `--GPU` / Linux CUDA / MPU 0.2; Lean in
 every run / Lean twins / FTA on externals / package retired / math
 notation); the stale "not yet invoked by main.py" sentence amended.
3. **Cross-host hash pins** stay raw-byte pins (no code change); the
 Windows/Ubuntu difference is confined to line-ending-dependent inputs and
 recorded in the evidence log.
4. **Ubuntu processor SIGSEGV** (one occurrence, three armed clean re-runs)
 stays recorded as an open risk; no action.
5. **`main.cpp` abort minidump handler** stays as general crash plumbing.

Found on the way and fixed: the release's trap-dump gate
(`find_debug_dumps.py --fail-on-traps`) failed on two permanent Rule-31
sinks (`infra/diagnostics_log.hpp`, the `#if RT_MEASUREMENT` hit log in
`prover.cpp`); the finder now suppresses `#if RT_MEASUREMENT` /
`#if MEM_MEASUREMENT` blocks and the listed permanent sink files
(`.scripts/tests/test_find_debug_dumps.py`, 4 tests). Gate: 0 traps.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
