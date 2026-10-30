# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------------
#
# This software is also available under a commercial license. For details,
# see: https://generative-logic.com/license
#
# Contributions to this project must be made under the terms of the
# Contributor License Agreement (CLA). See the project's CONTRIBUTING.md file.


import expression_utils
import fta_shortcut
import generate_full_proof_graph
import json
import re
import sys
import time
import shutil
import os
import tempfile
from configuration_reader import configuration_reader
from pathlib import Path
import subprocess
import process_proof_graphs
import verifier
from proof_export import live as lean_live
import compilation_state
from incubator_to_simple_facts import convert_incubator_theorems
from frame_timing import StageTimer

# wherever this file lives, assume the project root is its parent folder
PROJECT_ROOT = Path(__file__).resolve().parent

# Cross-platform native-binary lookup. Windows builds end with `.exe`
# (MSBuild output); Linux/macOS builds (e.g. via the Makefile in
# GL_Quick_VS/GL_Quick/) drop the extension. Try the platform's
# canonical name first, then fall back to the other so a developer
# who built one and Python is invoked from the other layout still
# works.
def _find_gl_quick_exe() -> Path:
    base = PROJECT_ROOT / 'GL_Quick_VS' / 'GL_Quick'
    if sys.platform.startswith('win'):
        candidates = [base / 'gl_quick.exe', base / 'gl_quick']
    else:
        candidates = [base / 'gl_quick', base / 'gl_quick.exe']
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Native executable not found. Looked for: "
        + ", ".join(str(c) for c in candidates)
    )


def run_gl_quick(
        anchor_id: str = "", extra_args=None, phase2_backend: str = None):
    assert phase2_backend is None or phase2_backend in ("cpu", "cuda"), \
        "phase2 backend override must be None, cpu, or cuda"
    exe_path = _find_gl_quick_exe()
    cmd = [str(exe_path)]
    if anchor_id:
        cmd.append(anchor_id)
    # Optional batch-path flags after the tag (shortcut mode:
    # --conjectures-file / --externals-file, see run_modes.cpp::fullRun).
    if extra_args:
        cmd.extend(str(arg) for arg in extra_args)
    if phase2_backend is not None:
        cmd.extend(["--phase2-backend", phase2_backend])

    # Inherit parent's stdout/stderr -> prints live instead of at the end
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


_SPONTANEOUS_CATEGORIES = {"implication", "existence", "or", "and"}


def _seed_per_batch_binary(tag: str) -> None:
    """Pre-batch: copy GL_binary_shared.json to GL_binary_<tag>.json so the C++
    prover loads cross-batch persistent compact-operator names at startup.
    On a clean run when the shared file does not exist yet, write an empty
    object so the C++ loader sees a valid (but empty) JSON file. Python is
    the sole writer of the shared file and the sole creator of the per-batch
    file; the C++ prover only reads at startup and overwrites this same path
    at end of run via ``exportCompiledExpressionsJSON``.
    """
    bin_dir = PROJECT_ROOT / "files" / "GL_binaries"
    bin_dir.mkdir(parents=True, exist_ok=True)
    shared = bin_dir / "GL_binary_shared.json"
    target = bin_dir / f"GL_binary_{tag}.json"
    if shared.exists():
        shutil.copyfile(shared, target)
    else:
        target.write_text("{}\n", encoding="utf-8")
    compilation_state.seed_state(PROJECT_ROOT, tag)


def _merge_into_shared(tag: str) -> None:
    """Post-batch: read GL_binary_<tag>.json and add any new entries with a
    spontaneous category (``implication`` / ``existence`` / ``or`` / ``and``)
    into GL_binary_shared.json. Atomic entries are excluded by the category
    check; everything else with a spontaneous category is merged.

    Applies to EVERY tag — incubator and main alike. Incubator allocations
    safely contribute to shared and every batch sees the same compiled-
    expression space, because ``ExpressionAnalyzer::repetitionExclusionMap``
    is keyed by ``(elements, category)`` — without that an incubator-allocated
    implication with elements E and a main-batch existence with the same
    elements E would collide on lookup, causing the main batch to compile its
    existence body as an implication — see [D-61] and
    [D-60].

    Why no name-prefix filter. Restricting the merge to names matching
    ``^(existence|implication|or|and)\\d+$`` so anchor entries like
    ``AnchorIncubator`` would not leak into shared would be too
    aggressive: hand-defined entities such as ``preorder`` / ``sequence`` /
    ``interval`` / ``limitSet`` / ``fXY`` / ``fold`` / ``constSeq`` etc. are
    transitive dependencies of spontaneous-allocator operators (e.g.
    ``implication20`` carries ``(preorder[u_2,u_3,u_4,1])`` in its elements),
    and Peano's MPL does not define them. With those entities filtered out
    of shared, Peano boots without ``preorder`` in ``compiledExpressions``
    and the prover asserts at ``prover.hpp::resolveCoreOrCrash`` when an
    implication walk reaches ``preorder``. Category-only is the correct
    filter; anchor entries are inert in batches whose theorems do not name
    them (Peano references ``AnchorPeano``, never ``AnchorIncubator``).

    Compilation migration runs before this merge: retired numbered entries
    and their references have already been replaced. Surviving entries are
    not overwritten, so every retained name keeps its original definition.
    The compilation-state sidecar preserves monotone issued-number counters.
    """
    bin_dir = PROJECT_ROOT / "files" / "GL_binaries"
    shared = bin_dir / "GL_binary_shared.json"

    per_batch = bin_dir / f"GL_binary_{tag}.json"
    if not per_batch.exists():
        return

    try:
        with open(per_batch, "r", encoding="utf-8") as f:
            batch_entries = json.load(f) or {}
    except (json.JSONDecodeError, OSError) as e:
        print(f"[merge_into_shared] failed to read {per_batch}: {e}")
        return

    shared_entries = {}
    if shared.exists():
        try:
            with open(shared, "r", encoding="utf-8") as f:
                shared_entries = json.load(f) or {}
        except (json.JSONDecodeError, OSError):
            shared_entries = {}

    added = 0
    for name, entry in batch_entries.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("category") not in _SPONTANEOUS_CATEGORIES:
            continue
        if name in shared_entries:
            continue
        shared_entries[name] = entry
        added += 1

    bin_dir.mkdir(parents=True, exist_ok=True)
    with open(shared, "w", encoding="utf-8") as f:
        json.dump(shared_entries, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[merge_into_shared] {tag}: +{added} new entries "
          f"(shared total: {len(shared_entries)})")


def empty_simple_facts(dir_path: str = "files/simple_facts") -> None:
    """Remove all files/subdirectories inside files/simple_facts (leaves the folder)."""
    d = Path(dir_path)
    if d.exists():
        for p in d.iterdir():
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
    else:
        d.mkdir(parents=True, exist_ok=True)


def generate_anchor_connection(current_tag: str, prev_tag: str, theorems_dir=None):
    """
    Connects current tag's anchor (A) to previous tag's anchor (B) as A -> B.
    (e.g., AnchorGauss -> AnchorPeano)
    Appends the result to conjectures.txt.
    """
    # 1. Load Configurations
    config_current = configuration_reader(PROJECT_ROOT / "files" / "config" / f"Config{current_tag}.json")
    config_prev = configuration_reader(PROJECT_ROOT / "files" / "config" / f"Config{prev_tag}.json")

    # 2. Get Anchor Expressions (Normalized: e.g. (Anchor[1,2,3]))
    anchor_name_current = expression_utils.get_anchor_name(config_current)
    anchor_name_prev = expression_utils.get_anchor_name(config_prev)

    if not anchor_name_current or not anchor_name_prev:
        print(f"Warning: Could not find anchor names for {current_tag} or {prev_tag}.")
        return

    expr_A = config_current[anchor_name_current].short_mpl_normalized  # Current (e.g. Gauss)
    expr_B = config_prev[anchor_name_prev].short_mpl_normalized  # Previous (e.g. Peano)

    # 3. Get Arguments
    args_A = expression_utils.get_args(expr_A)
    args_B = expression_utils.get_args(expr_B)

    # 4. Assert Subset: Args(B) <= Args(A)
    # We require the previous anchor's arguments (target) to be a subset of the current anchor's (source)
    # so that the implication quantification (over A) covers all variables in B.
    if not set(args_B).issubset(set(args_A)):
        print(f"Abort: Arguments of {prev_tag} ({args_B}) are not a subset of {current_tag} ({args_A}).")
        return

    # 5. Create Implication A -> B (Current -> Previous)
    # e.g., (> [1,2,3] (AnchorGauss[1,2,3]) (AnchorPeano[1]))
    # Every slot of A binds at its premise, used by B or not (I-4).
    quantified_vars = args_A

    implication = f"(>[{','.join(quantified_vars)}]{expr_A}{expr_B})"

    # 6. Append to conjectures.txt
    if theorems_dir is None:
        theorems_dir = PROJECT_ROOT / "files" / "theorems"
    theorems_path = theorems_dir / "conjectures.txt"
    try:
        with open(theorems_path, "a", encoding="utf-8") as f:
            f.write(implication + "\n")
        print(f"Attached anchor implication between {current_tag} and {prev_tag}: {implication}")
    except IOError as e:
        print(f"Error appending to conjectures.txt: {e}")

def empty_raw_proof_graph(dir_path: str = "files/raw_proof_graph") -> None:
    """Remove all files/subdirectories inside files/raw_proof_graph (leaves the folder)."""
    d = PROJECT_ROOT / dir_path
    if d.exists():
        for p in d.iterdir():
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
    else:
        d.mkdir(parents=True, exist_ok=True)

def _rebuild_compressed_externals(theorems_dir: Path):
    """Rebuild compressed_external_theorems.txt = originals + mirrored variants."""
    ext_thm_path = theorems_dir / "externally_provided_theorems.txt"
    if not ext_thm_path.exists():
        ext_thm_path.touch()

    comp_ext_path = theorems_dir / "compressed_external_theorems.txt"
    comp_ext_path.write_text("")

    # Delegate to C++ --mirror-externals mode
    rel_dir = theorems_dir.relative_to(PROJECT_ROOT)
    exe_path = _find_gl_quick_exe()
    subprocess.run([str(exe_path), "--mirror-externals", str(rel_dir).replace("\\", "/")],
                   cwd=PROJECT_ROOT, check=True)


def _setup_theorem_folder(theorems_dir: Path):
    """Setup a theorem folder: clear proved, keep externals, rebuild compressed."""
    theorems_dir.mkdir(parents=True, exist_ok=True)

    for fname in [
        "theorems.txt",
        "compressed_out_theorems.txt",
        "vacuous_theorems.txt",
    ]:
        fpath = theorems_dir / fname
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write("")

    _rebuild_compressed_externals(theorems_dir)


def _discover_configs_for_tag(base_tag: str) -> list:
    """Return every <suffix> for which files/config/Config<suffix>.json
    exists, where suffix matches `<base_tag>\\d*`. Sorted alphanumerically.

    Default Python string sort puts ConfigPeano.json before ConfigPeano1.json
    because '.' (ASCII 46) < '1' (ASCII 49). So a tag's "base" config (no
    digit suffix) always runs first in the per-tag sequence.
    """
    pattern = re.compile(rf"^Config({re.escape(base_tag)}\d*)\.json$")
    config_dir = PROJECT_ROOT / "files" / "config"
    out = []
    if config_dir.exists():
        for f in os.listdir(config_dir):
            m = pattern.match(f)
            if m:
                out.append(m.group(1))
    return sorted(out)


def _filter_vacuous_tainted(theorems_dir: Path, raw_dir: Path) -> None:
    """@brief Taint closure of the vacuity classification: retire theorems
    whose proofs cite a vacuous theorem, directly or transitively.

    @details
    The C++ side seeds ``vacuous_theorems.txt`` with the two certificates
    (``retracted producer`` — the producer LB derived its own main-scope
    contradiction; ``premise contradiction`` — same-chain theorems with
    contradictory heads). A vacuous theorem may have been broadcast as a
    rule and used by other proofs before its retirement, so this pass —
    running after every batch, before the next batch reads the pool —
    builds the citation graph over the batch family's RAW proof chapters
    (producer-form namespace, so seeds and citations share one string
    space; rows tagged ``theorem`` / ``externally provided theorem`` cite
    other theorems by statement), computes the transitive closure from the
    seeds, and rewrites ``theorems.txt`` / ``compiled_theorems.txt``
    (line-paired by the ``saveProvedTheoremsFiltered`` contract) without
    the tainted theorems. Tainted theorems are appended to
    ``vacuous_theorems.txt`` with the ``tainted`` label; their chapters
    are KEPT — the derivations are honest and auditable, they merely rest
    on a vacuous fact — only the deliverable pool shrinks (dependency-
    directed retraction in the Truth-Maintenance-System sense over the
    proof graph GL already ships).

    A missing seed file, empty seed set, missing manifest, or zero
    tainted theorems is a defined no-op. A seed string re-proved
    legitimately in a later batch would be conservatively retired too —
    over-removal loses a true theorem but never ships a vacuous one.

    @param theorems_dir  The batch family's theorem folder (holds the
                         seed artifact and the paired pool files).
    @param raw_dir       The batch family's raw proof-graph folder (the
                         citation source; accumulates across the family's
                         batches, so cross-batch citations are covered).
    @return None.
    @invariant Deterministic: seeds and chapters are read in file order;
               the closure is order-independent; the pool rewrite
               preserves surviving line order.
    """
    vac_path = theorems_dir / "vacuous_theorems.txt"
    if not vac_path.is_file():
        return
    seeds = set()
    with open(vac_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip():
                seeds.add(line.split("\t")[0])
    if not seeds:
        return

    _theorems, ordered = verifier.load_global_theorem_list(str(raw_dir))
    if not ordered:
        return
    chapter_files = sorted(
        (fn for fn in os.listdir(raw_dir)
         if fn.endswith(".txt") and fn != "global_theorem_list.txt"),
        key=verifier.chapter_sort_key)
    ch_map = verifier.build_chapter_theorem_map(chapter_files, ordered)

    cites = {}
    for fn, (thm_expr, _type, _ref) in ch_map.items():
        edges = cites.setdefault(thm_expr, set())
        with open(raw_dir / fn, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3 and parts[2] in (
                        "theorem", "externally provided theorem"):
                    edges.add(parts[0])

    tainted = set()
    changed = True
    while changed:
        changed = False
        for thm, edges in cites.items():
            if thm in tainted or thm in seeds:
                continue
            if edges & (seeds | tainted):
                tainted.add(thm)
                changed = True
    if not tainted:
        return

    compiled_path = theorems_dir / "compiled_theorems.txt"
    theorems_path = theorems_dir / "theorems.txt"
    if not (compiled_path.is_file() and theorems_path.is_file()):
        return
    with open(compiled_path, encoding="utf-8") as f:
        compiled = [l.rstrip("\n") for l in f if l.strip()]
    with open(theorems_path, encoding="utf-8") as f:
        raw_lines = [l.rstrip("\n") for l in f if l.strip()]
    assert len(compiled) == len(raw_lines), (
        "theorems.txt / compiled_theorems.txt line pairing broken")
    keep_compiled, keep_raw, removed = [], [], []
    for c_line, r_line in zip(compiled, raw_lines):
        if c_line in tainted:
            removed.append(c_line)
        else:
            keep_compiled.append(c_line)
            keep_raw.append(r_line)
    if not removed:
        return
    with open(compiled_path, "w", encoding="utf-8") as f:
        f.write("".join(l + "\n" for l in keep_compiled))
    with open(theorems_path, "w", encoding="utf-8") as f:
        f.write("".join(l + "\n" for l in keep_raw))
    with open(vac_path, "a", encoding="utf-8") as f:
        for thm in removed:
            f.write(f"{thm}\ttainted\n")
    for thm in removed:
        print(f"[VACUOUS-TAINT] {thm}")


def _supplied_conjectures(tag: str, config) -> list:
    """@brief Read a batch's supplied conjectures.

    @details
    The rows of the configuration's ``external_conjectures_file``, blank lines
    dropped, each a complete parenthesized expression. A batch that disables
    generation must supply at least one row. Read by the full run's
    preparation (``_prepare_batch_conjectures``).

    @param tag Configuration suffix, for the failure message.
    @param config Loaded batch configuration with its input policy.
    @return The supplied rows in file order (empty without an input file).
    @invariant Every row is parenthesized; disabling generation requires a
               nonempty input (asserted).
    """
    external = []
    if config.external_conjectures_file:
        source = PROJECT_ROOT / config.external_conjectures_file
        assert source.is_file(), f"External conjectures file is missing: {source}"
        external = [line.strip() for line in source.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
        assert all(line.startswith("(") and line.endswith(")") for line in external)
    assert config.generate_conjectures or external, (
        f"{tag}: generation is disabled but external conjectures are empty")
    return external


def _prepare_batch_conjectures(tag: str, config, theorems_dir: Path) -> None:
    """@brief Prepare generated and externally supplied proof obligations.

    @details
    An empty external input contributes no conjectures. A batch that disables
    generation must supply a nonempty file. Supplied rows enter only the
    conjecture file; earlier proved theorems and external premises are untouched.
    Generated mirror and OR metadata are empty for a supplied-only batch.

    @param tag Configuration suffix passed to the native conjecturer.
    @param config Loaded batch configuration with its input policy.
    @param theorems_dir Destination folder for this batch's proof obligations.
    @return None.
    @invariant Disabling generation requires nonempty external conjectures.
    """
    external = _supplied_conjectures(tag, config)
    theorems_dir.mkdir(parents=True, exist_ok=True)
    destination = theorems_dir / "conjectures.txt"
    if config.generate_conjectures:
        print(f"Conjecture creation started for {tag}.")
        with StageTimer("native.conjecturer", parent="python.pipeline", batch=tag, excluded=True):
            subprocess.run([str(_find_gl_quick_exe()), "--conjecture", tag],
                           cwd=PROJECT_ROOT, check=True)
        print(f"Conjecture creation finished for {tag}.")
    else:
        source = PROJECT_ROOT / config.external_conjectures_file
        assert destination.resolve() != source.resolve(), "External input must remain durable"
        destination.write_text("", encoding="utf-8")
        for name in ("or_pairs.txt", "mirror_pairs.txt"):
            (theorems_dir / name).write_text("", encoding="utf-8")
    if external:
        with destination.open("a", encoding="utf-8") as output:
            output.write("\n".join(external) + "\n")
        print(f"Loaded {len(external)} external conjectures for {tag}.")


def _run_batch(tag: str, prev_tags: list, theorems_dir: Path = None,
               simple_facts_fn=None, add_cross_anchor: bool = True,
               phase2_backend: str = None):
    """Run one batch: supplied/generated goals, anchor connections, and prover.

    `add_cross_anchor` defaults True for the historic single-batch-per-tag
    behaviour. The orchestrator passes False for the 2nd, 3rd, … batches
    of the same tag — the cross-anchor implication only needs to be added
    once per tag (it's identical across that tag's batches), and
    re-emitting it pollutes the next batch's conjectures.txt.

    ``phase2_backend`` is an optional whole-run oracle override. ``None``
    leaves ownership to the batch's ``use_gpu`` parameter; ``cpu`` or ``cuda``
    explicitly selects that backend for comparison runs.
    """
    config_path = PROJECT_ROOT / "files" / "config" / f"Config{tag}.json"
    if not config_path.exists():
        print(f"Warning: Configuration file {config_path} not found. Skipping.")
        return

    config = configuration_reader(config_path)

    # A. Create Simple Facts (if provided)
    if simple_facts_fn:
        with StageTimer(
            "python.simple_facts", parent="python.pipeline", batch=tag
        ):
            for n in config.parameters.simple_facts_parameters:
                simple_facts_fn(n)

    # B. Prepare this batch's proof obligations.
    target_dir = theorems_dir or PROJECT_ROOT / (config.theorems_folder or "files/theorems")
    _prepare_batch_conjectures(tag, config, target_dir)

    # C. Connect to Previous Anchors (Current -> Prev) — only on first batch per tag
    if add_cross_anchor:
        with StageTimer(
            "python.anchor_connections", parent="python.pipeline", batch=tag
        ):
            for prev_tag in prev_tags:
                generate_anchor_connection(tag, prev_tag, theorems_dir=theorems_dir)

    # D. Run Native Prover. The conjecturer subprocess has exited and closed
    # every output handle before subprocess.run returns, so its files are
    # immediately available to the next process without a fixed delay.
    print(f"Running GL_Quick for {tag}...")
    # Cross-batch persistent naming: hand the shared spontaneous-operator
    # registry to the C++ prover by copying GL_binary_shared.json into the
    # per-batch file the prover reads on startup. The prover overwrites this
    # same path at end of run with the inherited entries plus any new ones
    # it allocated.
    with StageTimer("python.binary_seed", parent="python.pipeline", batch=tag):
        _seed_per_batch_binary(tag)
    if tag == "FTA":
        with StageTimer("python.fta_input_capture", parent="python.pipeline", batch=tag):
            fta_shortcut.capture(PROJECT_ROOT)
    os.environ["GL_FRAME_BATCH"] = tag
    with StageTimer(f"native.{tag}", parent="python.pipeline", batch=tag):
        run_gl_quick(tag, phase2_backend=phase2_backend)
    # Fold any newly-allocated spontaneous entries from this batch back into
    # the shared registry so the next batch starts with them.
    with StageTimer("python.binary_merge", parent="python.pipeline", batch=tag):
        migration = compilation_state.migrate_batch(PROJECT_ROOT, tag)
        print(f"[compilation] {tag}: {migration}")
        _merge_into_shared(tag)

    # Vacuity taint closure: retire pool theorems whose proofs cite a
    # vacuous theorem before the next batch reads the pool. The batch's
    # config names its folders (the same keys the native side reads).
    with StageTimer("python.vacuity_filter", parent="python.pipeline", batch=tag):
        cfg = configuration_reader(
            PROJECT_ROOT / "files" / "config" / f"Config{tag}.json")
        t_dir = (theorems_dir if theorems_dir is not None
                 else PROJECT_ROOT / "files" / "theorems")
        raw_dir = PROJECT_ROOT / cfg.raw_proof_graph_folder
        _filter_vacuous_tainted(t_dir, raw_dir)




CLEAN_RUN = True
RUN_INCUBATOR = True  # full run: incubator + main (submatch cap=40000, fixed-100 splits)

RUN_MAIN_PATH = True  # full pipeline: incubator + main (determinism study complete)

SIMPLE_FACTS_MAP = {}


def full_run(phase2_backend: str = None, fresh_lean: bool = False) -> int:
    """@brief Run the unified pipeline and return the verifier's failure count.

    @details
    For each tag: Incubator{Tag} -> {Tag}. Incubator produces ground-level
    facts for CE filtering. Main path produces proof graph theorems, then FTA
    (weak and strong induction) consumes its supplied conjectures and the
    preceding proved pool. Use
    --no-clean to keep previous results. The finalization verifies the main
    and incubator graphs; their combined failure count is returned so that
    ``main.py`` can end a run with failures in a failed process status.

    @param phase2_backend ``None`` for per-batch ``use_gpu`` selection (the
           processor route unless a config pins CUDA), or an explicit
           ``cpu``/``cuda`` override for the whole run (``main.py --GPU``
           passes ``cuda``).
    @param fresh_lean Whether the Lean export removes Lake's build folder
           ``.lake`` before the kernel check (``main.py --fresh-lean``); by
           default Lake rebuilds only what changed since the last run.
    @return The verifier failures of the main and incubator graphs together.
    """
    tags = ["Peano", "Gauss"]  # full pipeline

    theorems_dir = PROJECT_ROOT / "files" / "theorems"
    incubator_theorems_dir = PROJECT_ROOT / "files" / "incubator" / "theorems"

    incubator_dir = PROJECT_ROOT / "files" / "incubator"
    incubator_raw_dir = incubator_dir / "raw_proof_graph"
    incubator_proc_dir = incubator_dir / "processed_proof_graph"
    incubator_full_dir = incubator_dir / "full_proof_graph"

    # 1. Global Setup
    if CLEAN_RUN:
        with StageTimer("python.global_setup", parent="python.pipeline"):
            empty_simple_facts()
            empty_raw_proof_graph()

            gl_binaries_dir = PROJECT_ROOT / "files" / "GL_binaries"
            if gl_binaries_dir.exists():
                shutil.rmtree(gl_binaries_dir)
            gl_binaries_dir.mkdir(parents=True, exist_ok=True)

            _setup_theorem_folder(theorems_dir)
            if RUN_INCUBATOR:
                # Empty incubator externals at start (main externals untouched)
                ext_path = incubator_theorems_dir / "externally_provided_theorems.txt"
                ext_path.parent.mkdir(parents=True, exist_ok=True)
                ext_path.write_text("")
                _setup_theorem_folder(incubator_theorems_dir)
                empty_raw_proof_graph(str(incubator_raw_dir.relative_to(PROJECT_ROOT)))
    else:
        print("Skipping cleanup (--no-clean).")

    # 2. Run stages: for each tag, incubator first, then main
    for i, tag in enumerate(tags):
        prev_tags = tags[:i]

        # Per-tag glob: discover every Config<Tag><digits>?.json (and the
        # incubator twins). Sorted alphanumerically; the base config (no
        # digit suffix) sorts first. This is what allows
        # ConfigIncubatorGauss.json to run before ConfigIncubatorGauss1.json.
        with StageTimer(
            "python.config_discovery", parent="python.pipeline", batch=tag
        ):
            incub_suffixes = _discover_configs_for_tag(f"Incubator{tag}")
            main_suffixes = _discover_configs_for_tag(tag)

        assert incub_suffixes, f"No ConfigIncubator{tag}*.json found in files/config/"
        assert main_suffixes,  f"No Config{tag}*.json found in files/config/"

        if RUN_INCUBATOR:
            # Provide main proved theorems as incubator externals — once per
            # tag, before the FIRST incubator batch. Subsequent incubator
            # batches under the same tag share the same theorems folder, so
            # they see the prior batch's proved theorems via theorems.txt.
            #
            # The source is compiled_theorems.txt (compact form — keeps
            # existence2 / or0 as compact heads), not theorems.txt
            # (expanded — compiled operators like existence2 / or0 expanded to
            # base form). Reason: chapter rows cite rules in
            # compact form; the verifier's `origin` check looks them up by
            # alpha-canonical match against the externals registry; expanded
            # form misses on compact-form citations. Cross-batch parsing of
            # compact-form externals is supported because GL_binary_shared.json
            # carries the spontaneous compact-name dictionary across batches
            # (see _seed_per_batch_binary / _merge_into_shared above).
            if prev_tags:
                main_compiled = theorems_dir / "compiled_theorems.txt"
                incubator_ext = incubator_theorems_dir / "externally_provided_theorems.txt"
                with open(main_compiled, "r", encoding="utf-8") as src:
                    content = src.read()
                if content.strip():
                    with open(incubator_ext, "w", encoding="utf-8") as dst:
                        dst.write(content)
                    _rebuild_compressed_externals(incubator_theorems_dir)

            # Incubator stage(s) — alphanumeric order. Cross-anchor is
            # attached for the FIRST batch encountered for each distinct
            # anchor in the group (so e.g. AI8 -> Peano lands on the first
            # AI8 batch and AI3 -> Peano lands on the first AI3 batch).
            # Once a batch proves its cross-anchor implication it goes into
            # theorems.txt and subsequent same-anchor batches inherit
            # it via the shared theorems folder.
            incub_bridged_anchors: set[str] = set()
            for j, incub_tag in enumerate(incub_suffixes):
                incub_cfg = configuration_reader(
                    PROJECT_ROOT / "files" / "config" / f"Config{incub_tag}.json")
                incub_anchor = expression_utils.get_anchor_name(incub_cfg)
                needs_bridge = bool(prev_tags) and incub_anchor not in incub_bridged_anchors
                incub_bridged_anchors.add(incub_anchor)
                print(f"\n=== Incubator {tag}: {incub_tag} ===")
                _run_batch(incub_tag, prev_tags=prev_tags,
                           theorems_dir=incubator_theorems_dir,
                           add_cross_anchor=needs_bridge,
                           phase2_backend=phase2_backend)

        else:
            print(f"\n=== Skipping Incubator {tag} ===")

        # Write the incubator's processed theorem list, which
        # convert_incubator_theorems reads; the incubator book itself is
        # arranged once, in finalization (_process_incubator_worker).
        if RUN_INCUBATOR:
            visu_config_path = PROJECT_ROOT / "files" / "config" / "ConfigVisu.json"
            configuration_visu = configuration_reader(visu_config_path)
            with StageTimer(
                "python.incubator_theorem_list",
                parent="python.pipeline",
                batch=tag,
            ):
                process_proof_graphs.create_processed_theorem_list(
                    configuration_visu, raw_dir=incubator_raw_dir,
                    proc_dir=incubator_proc_dir,
                    theorems_dir=incubator_theorems_dir)

        # Convert incubator proved theorems to simple facts (always, even if incubator skipped)
        with StageTimer(
            "python.incubator_conversion", parent="python.pipeline", batch=tag
        ):
            convert_incubator_theorems(tag, theorems_dir=incubator_theorems_dir)

        if not RUN_MAIN_PATH:
            print(f"\n=== Skipping Main {tag} ===")
            continue

        # Main stage(s) — alphanumeric order. Cross-anchor only on the
        # first batch (j == 0).
        for j, main_tag in enumerate(main_suffixes):
            print(f"\n=== {main_tag} ===")
            _run_batch(main_tag, prev_tags=prev_tags,
                       simple_facts_fn=SIMPLE_FACTS_MAP.get(tag),
                       add_cross_anchor=(j == 0),
                       phase2_backend=phase2_backend)

    # FTA uses the preceding proved pool and supplied conjectures directly.
    # Its input already contains the Peano and Gauss anchor connections.
    if RUN_MAIN_PATH:
        print("\n=== FTA ===")
        _run_batch("FTA", prev_tags=tags, theorems_dir=theorems_dir,
                   add_cross_anchor=False, phase2_backend=phase2_backend)

    # 3. Finalization
    visu_config_path = PROJECT_ROOT / "files" / "config" / "ConfigVisu.json"
    configuration_visu = configuration_reader(visu_config_path)

    all_success, all_failure = _finish_full_graphs(
        configuration_visu, RUN_MAIN_PATH, RUN_INCUBATOR, fresh_lean)

    if all_failure == 0:
        art = (
            "\n"
            "                    *  .  *\n"
            "                   . *\\|/* .\n"
            "                *   * \\|/ *   *\n"
            "                 .  */|\\*  .\n"
            "                *  . /|\\ .  *\n"
            "                   . *|* .\n"
            "                     |||\n"
            "                     |||\n"
            "            All proof graphs verified.\n"
            f"        {all_success} checks, 0 failures — airtight.\n"
        )
        print(art)
    return all_failure


def _finalize_incubator_worker(result_path: str, include_main: bool) -> None:
    """@brief Render and verify the completed incubator graph in its own process.

    @details
    Reads completed processed graphs and writes only incubator HTML, its timing
    records, and the requested result file. Separate process state isolates the
    renderer's configuration and link maps from simultaneous main rendering.

    @param result_path JSON destination for the verifier success/failure totals.
    @param include_main Whether main chapters are available as sibling links.
    @return None.
    @invariant All processed inputs are complete before this worker starts.
    @see _finish_full_graphs
    """
    with StageTimer("python.finalization.incubator", parent="python.finalization", batch="incubator"):
        proc_dir = PROJECT_ROOT / "files/incubator/processed_proof_graph"
        out_dir = PROJECT_ROOT / "files/incubator/full_proof_graph"
        config = configuration_reader(PROJECT_ROOT / "files/config/ConfigVisu.json")
        siblings = ([{"proc_dir": PROJECT_ROOT / "files/processed_proof_graph",
                      "out_dir": PROJECT_ROOT / "files/full_proof_graph"}]
                    if include_main else [])
        print("\n--- Generating Incubator Proof Graph HTML ---")
        with StageTimer("python.html.incubator", parent="python.finalization.incubator", batch="incubator"):
            generate_full_proof_graph.generate_proof_graph_pages(
                config, proc_dir=proc_dir, out_dir=out_dir, sibling_graphs=siblings)
        print("\n--- Running Incubator Proof Graph Verifier ---")
        with StageTimer("python.verifier.incubator", parent="python.finalization.incubator", batch="incubator"):
            state = verifier.run_verifier(str(proc_dir))
            verifier.print_report(state)
            success, failure = verifier.get_totals(state)
        Path(result_path).write_text(json.dumps({"success": success, "failure": failure}), encoding="utf-8")


def _process_incubator_worker() -> None:
    """@brief Arrange the incubator book once, in its own process.

    @details
    Runs the complete incubator processing (pruning, renaming and the book
    arrangement) on the raw incubator graph after the last batch. The runs
    between the incubator and main batches wrote only the theorem list
    (``process_proof_graphs.create_processed_theorem_list``).

    @return None.
    @see _finish_full_graphs
    """
    incubator_dir = PROJECT_ROOT / "files" / "incubator"
    config = configuration_reader(PROJECT_ROOT / "files/config/ConfigVisu.json")
    with StageTimer("python.processed_graph.incubator", parent="python.finalization", batch="incubator"):
        process_proof_graphs.create_processed_proof_graph(
            config, raw_dir=incubator_dir / "raw_proof_graph",
            proc_dir=incubator_dir / "processed_proof_graph",
            theorems_dir=incubator_dir / "theorems")


def _start_finalization_worker(prefix: str, statement: str, arguments: list[str]):
    """@brief Start one finalization step in its own Python process.

    @details
    The child runs ``statement`` with the result path and ``arguments`` as
    ``sys.argv[1:]``, inherits the environment (and with it the frame-timing
    ledger) and writes its output to a log retained under ``.debug/``.

    @param prefix File-name prefix of the retained log.
    @param statement Python statement the child executes.
    @param arguments Further string arguments after the result path.
    @return The running process, its open log file and its result path.
    @see _join_finalization_worker
    """
    worker_log = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=prefix, suffix=".log",
        dir=PROJECT_ROOT / ".debug", delete=False)
    result_path = Path(worker_log.name).with_suffix(".json")
    worker_environment = dict(os.environ)
    worker_environment["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen([sys.executable, "-c", statement, str(result_path), *arguments],
                               cwd=PROJECT_ROOT, env=worker_environment,
                               stdout=worker_log, stderr=subprocess.STDOUT)
    return process, worker_log, result_path


def _join_finalization_worker(worker) -> None:
    """@brief Wait for a finalization process, print its log and require success.

    @param worker The process, log file and result path of
           ``_start_finalization_worker``.
    @return None.
    @invariant A failed child asserts with its retained log's path.
    """
    process, worker_log, _ = worker
    return_code = process.wait()
    worker_log.close()
    print(Path(worker_log.name).read_text(encoding="utf-8"), end="")
    assert return_code == 0, f"Incubator finalization failed: {worker_log.name}"


def _finish_full_graphs(config, run_main: bool, run_incubator: bool,
                        fresh_lean: bool = False) -> tuple[int, int]:
    """@brief Finish independent main and incubator outputs concurrently.

    @details
    A child arranges the incubator book while the parent processes the main
    graph; both sibling registries exist once the parent has joined it. A second
    child then renders and verifies the incubator while the parent writes the
    Lean export, starts its kernel check in the background, renders main HTML,
    verifies main and finally waits for the kernel check, whose failure fails
    the run. The parent joins every child and the kernel check on every exit
    path and retains the child logs. Failed child execution asserts with its log.

    @param config Fixed visualization configuration for main output.
    @param run_main Whether the main graph is enabled.
    @param run_incubator Whether the incubator graph is enabled.
    @param fresh_lean Whether the Lean export removes Lake's build folder first.
    @return Combined verifier success and failure counts.
    @invariant Main HTML follows the Lean export and both processed registries;
               output trees and renderer state are separate.
    @see _process_incubator_worker, _finalize_incubator_worker
    """
    main_proc = PROJECT_ROOT / "files/processed_proof_graph"
    main_out = PROJECT_ROOT / "files/full_proof_graph"
    success = failure = 0
    with StageTimer("python.finalization", parent="python.pipeline"):
        processing = rendering = kernel = None
        try:
            if run_incubator:
                processing = _start_finalization_worker(
                    "incubator_processing_",
                    "import run_modes; run_modes._process_incubator_worker()", [])
            if run_main:
                print("\n--- Generating Proof Graph ---")
                with StageTimer("python.processed_graph.main", parent="python.finalization", batch="main"):
                    process_proof_graphs.create_processed_proof_graph(config)
            if processing is not None:
                worker, processing = processing, None
                _join_finalization_worker(worker)
                rendering = _start_finalization_worker(
                    "incubator_finalize_",
                    "import sys, run_modes; run_modes._finalize_incubator_worker(sys.argv[1], sys.argv[2] == '1')",
                    ["1" if run_main else "0"])
            if run_main:
                with StageTimer("python.finalization.main", parent="python.finalization", batch="main"):
                    with StageTimer("python.lean_export.main", parent="python.finalization.main", batch="main"):
                        export_dir = lean_live.write_main_export(main_proc, main_out,
                                                                 PROJECT_ROOT / "files/config",
                                                                 PROJECT_ROOT / "files/GL_binaries",
                                                                 fresh_lean=fresh_lean)
                        if export_dir is not None:
                            kernel = lean_live.start_kernel_check(lean_live.find_lake(), export_dir)
                    siblings = ([{"proc_dir": PROJECT_ROOT / "files/incubator/processed_proof_graph",
                                  "out_dir": PROJECT_ROOT / "files/incubator/full_proof_graph"}]
                                if run_incubator else [])
                    with StageTimer("python.html.main", parent="python.finalization.main", batch="main"):
                        generate_full_proof_graph.generate_proof_graph_pages(
                            config, sibling_graphs=siblings,
                            lean_kernel_check_running=kernel is not None)
                    print("\n--- Running Proof Graph Verifier ---")
                    with StageTimer("python.verifier.main", parent="python.finalization.main", batch="main"):
                        state = verifier.run_verifier(str(main_proc))
                        verifier.print_report(state)
                        success, failure = verifier.get_totals(state)
                    if kernel is not None:
                        check, kernel = kernel, None
                        with StageTimer("python.lean_kernel_wait.main", parent="python.finalization.main",
                                        batch="main"):
                            lean_live.finish_kernel_check(check)
        finally:
            if kernel is not None:
                lean_live.finish_kernel_check(kernel)
            for worker in (processing, rendering):
                if worker is not None:
                    _join_finalization_worker(worker)
        if rendering is not None:
            result = json.loads(rendering[2].read_text(encoding="utf-8"))
            assert set(result) == {"success", "failure"}
            assert all(type(value) is int and value >= 0 for value in result.values())
            success += result["success"]
            failure += result["failure"]
    return success, failure


def shortcut_run(phase2_backend: str = "cpu"):
    """@brief Replay the full pipeline's saved FTA input in the shortcut tree.

    @details
    The full run captures its prepared FTA registry, theorem channels and prior
    proof chapters. Replay validates that capture, restores those bytes, then
    runs the same native FTA batch and the graph processing. A full run is
    required after changing configurations, definitions or FTA's supplied
    conjectures. The shortcut skips earlier computation and connected Lean
    certification.

    @param phase2_backend Explicit cpu or cuda Phase 2 implementation.
    @return The shortcut graph's verifier failure count. Missing, stale or
            invalid input asserts before cleanup.
    @see fta_shortcut.capture, fta_shortcut.restore
    """
    assert phase2_backend in ("cpu", "cuda"), \
        "phase2 backend must be exactly cpu or cuda"
    tag = "FTA"
    shortcut_dir = PROJECT_ROOT / "files" / "shortcut"
    shortcut_theorems_dir = shortcut_dir / "theorems"
    shortcut_raw_dir = shortcut_dir / "raw_proof_graph"
    shortcut_proc_dir = shortcut_dir / "processed_proof_graph"
    shortcut_full_dir = shortcut_dir / "full_proof_graph"

    with StageTimer("python.global_setup", parent="python.pipeline", batch=tag):
        fta_shortcut.restore(PROJECT_ROOT)

    # 3. Prover run (no conjecturer — conjectures.txt is the input).
    with StageTimer("python.binary_seed", parent="python.pipeline", batch=tag):
        _seed_per_batch_binary(tag)
    os.environ["GL_FRAME_BATCH"] = tag
    with StageTimer(f"native.{tag}", parent="python.pipeline", batch=tag):
        run_gl_quick(tag, extra_args=[
            "--shortcut",
            "--conjectures-file",
            "files/shortcut/theorems/replay_conjectures.txt",
            "--externals-file",
            "files/shortcut/theorems/compressed_external_theorems.txt",
        ], phase2_backend=phase2_backend)
    with StageTimer("python.binary_merge", parent="python.pipeline", batch=tag):
        migration = compilation_state.migrate_batch(PROJECT_ROOT, tag, shortcut=True)
        print(f"[compilation] {tag}: {migration}")
        _merge_into_shared(tag)

    # 4. Vacuity taint closure over the shortcut pool.
    with StageTimer("python.vacuity_filter", parent="python.pipeline", batch=tag):
        _filter_vacuous_tainted(shortcut_theorems_dir, shortcut_raw_dir)

    # 5. Processed proof graph + HTML + verifier, all inside files/shortcut/.
    visu_config_path = PROJECT_ROOT / "files" / "config" / "ConfigVisu.json"
    configuration_visu = configuration_reader(visu_config_path)
    print("\n--- Generating Shortcut Proof Graph ---")
    with StageTimer(
        "python.processed_graph.shortcut", parent="python.pipeline", batch=tag
    ):
        process_proof_graphs.create_processed_proof_graph(
            configuration_visu, raw_dir=shortcut_raw_dir,
            proc_dir=shortcut_proc_dir,
            theorems_dir=shortcut_theorems_dir)
    # The connected Lean certificate is produced by the full run.
    lean_live.skip_export(shortcut_full_dir,
                         "Development shortcut: run the full pipeline for Lean certification.")
    with StageTimer("python.html.shortcut", parent="python.pipeline", batch=tag):
        generate_full_proof_graph.generate_proof_graph_pages(
            configuration_visu, proc_dir=shortcut_proc_dir,
            out_dir=shortcut_full_dir)

    print("\n--- Running Shortcut Proof Graph Verifier ---")
    with StageTimer("python.verifier.shortcut", parent="python.pipeline", batch=tag):
        shortcut_state = verifier.run_verifier(str(shortcut_proc_dir))
        verifier.print_report(shortcut_state)
        success_count, failure_count = verifier.get_totals(shortcut_state)

    if failure_count == 0:
        art = (
            "\n"
            "                    *  .  *\n"
            "                   . *\\|/* .\n"
            "                *   * \\|/ *   *\n"
            "                 .  */|\\*  .\n"
            "                *  . /|\\ .  *\n"
            "                   . *|* .\n"
            "                     |||\n"
            "                     |||\n"
            "            All proof graphs verified.\n"
            f"        {success_count} checks, 0 failures — airtight.\n"
        )
        print(art)
    else:
        print(f"\nShortcut verification: {success_count} checks, "
              f"{failure_count} failures.")
    return failure_count
