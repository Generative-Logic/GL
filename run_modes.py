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
import generate_full_proof_graph
import json
import re
import sys
import time
import shutil
import os
from configuration_reader import configuration_reader
from pathlib import Path
import subprocess
import process_proof_graphs
import verifier
from proof_export import live as lean_live
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

    The shared file grows monotonically over the run; entries already present
    are not overwritten so a name allocated by an earlier batch keeps its
    original definition.
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
    quantified_vars = args_B

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


def _run_batch(tag: str, prev_tags: list, theorems_dir: Path = None,
               simple_facts_fn=None, add_cross_anchor: bool = True,
               phase2_backend: str = None):
    """Run a single batch: conjecture generation + anchor connections + prover.

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

    # B. Create Expressions (C++ conjecturer)
    start_time = time.time()
    print(f"Conjecture creation started for {tag}.")
    exe_path = _find_gl_quick_exe()
    with StageTimer(
        "native.conjecturer", parent="python.pipeline", batch=tag, excluded=True
    ):
        subprocess.run(
            [str(exe_path), "--conjecture", tag], cwd=PROJECT_ROOT, check=True
        )
    print(f"Conjecture creation finished for {tag}.")
    end_time = time.time()
    print(f"Conjecture creation runtime: {end_time - start_time:.5f} seconds")

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
    os.environ["GL_FRAME_BATCH"] = tag
    with StageTimer(f"native.{tag}", parent="python.pipeline", batch=tag):
        run_gl_quick(tag, phase2_backend=phase2_backend)
    # Fold any newly-allocated spontaneous entries from this batch back into
    # the shared registry so the next batch starts with them.
    with StageTimer("python.binary_merge", parent="python.pipeline", batch=tag):
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


def full_run(phase2_backend: str = None):
    
    """
    Unified pipeline. For each tag: Incubator{Tag} -> {Tag}.
    Incubator produces ground-level facts for CE filtering.
    Main path produces proof graph theorems.
    Use --no-clean to keep previous results (e.g. run only Gauss after Peano).

    ``phase2_backend`` is ``None`` for per-batch ``use_gpu`` selection (the
    processor route unless a config pins CUDA), or an explicit ``cpu``/``cuda``
    override for the whole run (``main.py --GPU`` passes ``cuda``).
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

        # Process incubator proof graph so global_theorem_list.txt exists
        # before convert_incubator_theorems reads it
        if RUN_INCUBATOR:
            visu_config_path = PROJECT_ROOT / "files" / "config" / "ConfigVisu.json"
            configuration_visu = configuration_reader(visu_config_path)
            with StageTimer(
                "python.processed_graph.incubator",
                parent="python.pipeline",
                batch=tag,
            ):
                process_proof_graphs.create_processed_proof_graph(
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

    # 3. Finalization
    visu_config_path = PROJECT_ROOT / "files" / "config" / "ConfigVisu.json"
    configuration_visu = configuration_reader(visu_config_path)

    # Cross-batch HTML link plumbing: the incubator HTML can deep-link
    # into the main pipeline's HTML (and vice versa) so an external-
    # theorem citation in an incubator chapter opens the corresponding
    # main-pipeline chapter in a new browser tab. Each side passes the
    # other as a sibling_graphs entry; out_dirs are used to compute
    # relative URLs from one HTML tree to the other.
    main_proc_dir = PROJECT_ROOT / "files" / "processed_proof_graph"
    main_full_dir = PROJECT_ROOT / "files" / "full_proof_graph"

    incubator_siblings = []
    if RUN_MAIN_PATH:
        incubator_siblings.append({"proc_dir": main_proc_dir, "out_dir": main_full_dir})
    main_siblings = []
    if RUN_INCUBATOR:
        main_siblings.append({"proc_dir": incubator_proc_dir, "out_dir": incubator_full_dir})

    # 3a. Main proof graph — must run first so files/processed_proof_graph/
    # global_theorem_list.txt exists before the incubator HTML pass
    # (Section 3b) references it via `incubator_siblings`. Otherwise
    # generate_full_proof_graph.read_theorem_list raises FileNotFoundError
    # when the sibling read hits a not-yet-created path on a fresh worktree.
    if RUN_MAIN_PATH:
        print("\n--- Generating Proof Graph ---")
        with StageTimer(
            "python.processed_graph.main", parent="python.pipeline", batch="main"
        ):
            process_proof_graphs.create_processed_proof_graph(configuration_visu)
        # Lean export of the Peano + Gauss theorems into
        # files/full_proof_graph/lean_export/ (skipped with one notice when
        # no Lean toolchain is installed); runs before the HTML so the
        # chapter pages can link their Lean twins.
        with StageTimer(
            "python.lean_export.main", parent="python.pipeline", batch="main"
        ):
            lean_live.export_main_graph(
                main_proc_dir, main_full_dir,
                PROJECT_ROOT / "files" / "config",
                PROJECT_ROOT / "files" / "GL_binaries")
        with StageTimer(
            "python.html.main", parent="python.pipeline", batch="main"
        ):
            generate_full_proof_graph.generate_proof_graph_pages(
                configuration_visu, sibling_graphs=main_siblings)

    # 3b. Incubator HTML (incubator processed proof graph already built
    # in the tag loop; the main sibling's processed_proof_graph is now
    # built by Section 3a above).
    if RUN_INCUBATOR:
        print("\n--- Generating Incubator Proof Graph HTML ---")
        with StageTimer(
            "python.html.incubator", parent="python.pipeline", batch="incubator"
        ):
            generate_full_proof_graph.generate_proof_graph_pages(
                configuration_visu, proc_dir=incubator_proc_dir,
                out_dir=incubator_full_dir,
                sibling_graphs=incubator_siblings)

    # 4. Verification
    all_success = 0
    all_failure = 0

    if RUN_INCUBATOR:
        print("\n--- Running Incubator Proof Graph Verifier ---")
        with StageTimer(
            "python.verifier.incubator", parent="python.pipeline", batch="incubator"
        ):
            incubator_state = verifier.run_verifier(str(incubator_proc_dir))
            verifier.print_report(incubator_state)
            s, f = verifier.get_totals(incubator_state)
        all_success += s
        all_failure += f

    if RUN_MAIN_PATH:
        print("\n--- Running Proof Graph Verifier ---")
        with StageTimer(
            "python.verifier.main", parent="python.pipeline", batch="main"
        ):
            main_state = verifier.run_verifier(
                str(PROJECT_ROOT / "files" / "processed_proof_graph")
            )
            verifier.print_report(main_state)
            s, f = verifier.get_totals(main_state)
        all_success += s
        all_failure += f

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


def shortcut_run(phase2_backend: str = "cpu"):
    """@brief FTA-shortcut pipeline (``main.py --shortcut``): prove the
    hand-maintained conjecture list against the corpus supplied as external
    theorems, entirely inside ``files/shortcut/``.

    @details
    Implements D-248 (see ``docs/agentic_swdd/40_decisions.md``
    and ``docs/fta_ladder/fta_shortlist.md``): the prover receives the
    previously proved Peano + Gauss corpus externally plus a list of true
    conjectures, and assembles the proofs autonomously. Stages, mirroring
    ``full_run`` / ``_run_batch`` for the single ``Shortcut`` batch:

    1. Preconditions (assert, never fall back): the externals snapshot
       ``files/shortcut/theorems/externally_provided_theorems.txt`` in
       EXPANDED BASE FORM (rows of the main ``theorems.txt``, refreshed
       manually after a full run — base form is registry-independent, so
       the run needs no prior spontaneous registry) and the
       hand-maintained ``conjectures.txt``.
    2. Clean start: under ``CLEAN_RUN``, wipe ``files/GL_binaries`` and
       ``files/simple_facts`` (every big run starts clean/empty), then
       clean the shortcut tree: wipe ``raw_proof_graph``, truncate the
       theorems-folder run products, keep the two durable inputs,
       re-mirror the externals
       (``_setup_theorem_folder`` -> ``--mirror-externals``).
    3. Run the prover: seed ``GL_binary_FTA.json`` from shared (an empty
       object on the clean start), invoke
       ``gl_quick.exe Shortcut --conjectures-file ... --externals-file ...``
       (``ConfigFTA.json`` routes all outputs into ``files/shortcut/``
       and sets ``skip_compression``), merge new spontaneous names back into
       shared. No conjecturer step — ``conjectures.txt`` is the input.
    4. Vacuity taint closure over the shortcut pool.
    5. Processed proof graph + HTML export + verifier report, all under
       ``files/shortcut/``.

    ``full_run`` is untouched by this mode; the main ``files/theorems/``
    pool is never written.

    @param phase2_backend Explicit ``cpu`` or ``cuda`` Phase 2 implementation
        (default ``cpu`` — the processor route is the default of every run;
        ``main.py --GPU`` passes ``cuda``).
    @return None. Raises on any precondition or subprocess failure.
    """
    assert phase2_backend in ("cpu", "cuda"), \
        "phase2 backend must be exactly cpu or cuda"
    tag = "FTA"
    shortcut_dir = PROJECT_ROOT / "files" / "shortcut"
    shortcut_theorems_dir = shortcut_dir / "theorems"
    shortcut_raw_dir = shortcut_dir / "raw_proof_graph"
    shortcut_proc_dir = shortcut_dir / "processed_proof_graph"
    shortcut_full_dir = shortcut_dir / "full_proof_graph"

    # 1. Preconditions — every input the mode cannot regenerate itself. The
    # externals snapshot is a tracked input in its own right: the shortcut is
    # independent of the main path and never reads its theorem pool.
    externals_path = shortcut_theorems_dir / "externally_provided_theorems.txt"
    assert externals_path.exists() and externals_path.read_text(
        encoding="utf-8").strip(), (
        f"{externals_path} missing or empty — snapshot it from a full run: "
        "copy files/theorems/theorems.txt (base form) there.")
    conjectures_path = shortcut_theorems_dir / "conjectures.txt"
    assert conjectures_path.exists() and conjectures_path.read_text(
        encoding="utf-8").strip(), (
        f"{conjectures_path} missing or empty — it is the hand-maintained "
        "conjecture list (docs/fta_ladder/fta_shortlist.md).")
    # 2. Clean start. The corpus is BASE FORM (registry-independent), so the
    # run needs no pre-existing spontaneous registry and no CE fact tables —
    # like full_run, every big run starts from a clean/empty position: wipe
    # files/GL_binaries and files/simple_facts, then clean the shortcut tree
    # (keeping the two durable inputs conjectures.txt and
    # externally_provided_theorems.txt) and rebuild
    # compressed_external_theorems.txt via --mirror-externals.
    with StageTimer("python.global_setup", parent="python.pipeline", batch=tag):
        if CLEAN_RUN:
            empty_simple_facts()
            gl_binaries_dir = PROJECT_ROOT / "files" / "GL_binaries"
            if gl_binaries_dir.exists():
                shutil.rmtree(gl_binaries_dir)
            gl_binaries_dir.mkdir(parents=True, exist_ok=True)
        empty_raw_proof_graph(str(shortcut_raw_dir.relative_to(PROJECT_ROOT)))
        _setup_theorem_folder(shortcut_theorems_dir)

    # 3. Prover run (no conjecturer — conjectures.txt is the input).
    with StageTimer("python.binary_seed", parent="python.pipeline", batch=tag):
        _seed_per_batch_binary(tag)
    os.environ["GL_FRAME_BATCH"] = tag
    with StageTimer(f"native.{tag}", parent="python.pipeline", batch=tag):
        run_gl_quick(tag, extra_args=[
            "--conjectures-file",
            "files/shortcut/theorems/conjectures.txt",
            "--externals-file",
            "files/shortcut/theorems/compressed_external_theorems.txt",
        ], phase2_backend=phase2_backend)
    with StageTimer("python.binary_merge", parent="python.pipeline", batch=tag):
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
    # Lean export of the FTA theorems into files/shortcut/full_proof_graph/
    # lean_export/: independent of the main path — its only dependency is the
    # externals snapshot (skipped with one notice without a Lean toolchain).
    with StageTimer("python.lean_export.shortcut", parent="python.pipeline", batch=tag):
        lean_live.export_shortcut_graph(
            shortcut_proc_dir, shortcut_full_dir, externals_path,
            PROJECT_ROOT / "files" / "config",
            PROJECT_ROOT / "files" / "GL_binaries")
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
