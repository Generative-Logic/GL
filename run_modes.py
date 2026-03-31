# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025 Generative Logic UG (haftungsbeschränkt)
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
import time
import shutil
import os
from configuration_reader import configuration_reader
from pathlib import Path
import subprocess
import process_proof_graphs
import verifier
from incubator_to_simple_facts import convert_incubator_theorems

# wherever this file lives, assume the project root is its parent folder
PROJECT_ROOT = Path(__file__).resolve().parent


def run_gl_quick(anchor_id: str = ""):
    exe_path = PROJECT_ROOT / 'GL_Quick_VS' / 'GL_Quick' / 'gl_quick.exe'
    if not exe_path.exists():
        raise FileNotFoundError(f"Native executable not found at: {exe_path}")

    cmd = [str(exe_path)]
    if anchor_id:
        cmd.append(anchor_id)

    # Inherit parent's stdout/stderr -> prints live instead of at the end
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


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
    Appends the result to theorems.txt.
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

    # 6. Append to theorems.txt
    if theorems_dir is None:
        theorems_dir = PROJECT_ROOT / "files" / "theorems"
    theorems_path = theorems_dir / "theorems.txt"
    try:
        with open(theorems_path, "a", encoding="utf-8") as f:
            f.write(implication + "\n")
        print(f"Attached anchor implication between {current_tag} and {prev_tag}: {implication}")
    except IOError as e:
        print(f"Error appending to theorems.txt: {e}")

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
    exe_path = PROJECT_ROOT / 'GL_Quick_VS' / 'GL_Quick' / 'gl_quick.exe'
    subprocess.run([str(exe_path), "--mirror-externals", str(rel_dir).replace("\\", "/")],
                   cwd=PROJECT_ROOT, check=True)


def _setup_theorem_folder(theorems_dir: Path):
    """Setup a theorem folder: clear proved, keep externals, rebuild compressed."""
    theorems_dir.mkdir(parents=True, exist_ok=True)

    for fname in [
        "proved_theorems.txt",
        "compressed_out_theorems.txt",
    ]:
        fpath = theorems_dir / fname
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write("")

    _rebuild_compressed_externals(theorems_dir)


def _run_batch(tag: str, prev_tags: list, theorems_dir: Path = None,
               simple_facts_fn=None):
    """Run a single batch: conjecture generation + anchor connections + prover."""
    config_path = PROJECT_ROOT / "files" / "config" / f"Config{tag}.json"
    if not config_path.exists():
        print(f"Warning: Configuration file {config_path} not found. Skipping.")
        return

    config = configuration_reader(config_path)

    # A. Create Simple Facts (if provided)
    if simple_facts_fn:
        for n in config.parameters.simple_facts_parameters:
            simple_facts_fn(n)

    # B. Create Expressions (C++ conjecturer)
    start_time = time.time()
    print(f"Conjecture creation started for {tag}.")
    exe_path = PROJECT_ROOT / 'GL_Quick_VS' / 'GL_Quick' / 'gl_quick.exe'
    subprocess.run([str(exe_path), "--conjecture", tag], cwd=PROJECT_ROOT, check=True)
    print(f"Conjecture creation finished for {tag}.")
    end_time = time.time()
    print(f"Conjecture creation runtime: {end_time - start_time:.5f} seconds")

    # C. Connect to Previous Anchors (Current -> Prev)
    for prev_tag in prev_tags:
        generate_anchor_connection(tag, prev_tag, theorems_dir=theorems_dir)

    # D. Run Native Prover (brief pause to let filesystem flush txt files)
    time.sleep(2)
    print(f"Running GL_Quick for {tag}...")
    run_gl_quick(tag)




CLEAN_RUN = True
RUN_INCUBATOR = True

RUN_MAIN_PATH = True

SIMPLE_FACTS_MAP = {}


def full_run():
    
    """
    Unified pipeline. For each tag: Incubator{Tag} -> {Tag}.
    Incubator produces ground-level facts for CE filtering.
    Main path produces proof graph theorems.
    Use --no-clean to keep previous results (e.g. run only Gauss after Peano).
    """
    tags = ["Peano", "Gauss"]

    theorems_dir = PROJECT_ROOT / "files" / "theorems"
    incubator_theorems_dir = PROJECT_ROOT / "files" / "incubator" / "theorems"

    incubator_dir = PROJECT_ROOT / "files" / "incubator"
    incubator_raw_dir = incubator_dir / "raw_proof_graph"
    incubator_proc_dir = incubator_dir / "processed_proof_graph"
    incubator_full_dir = incubator_dir / "full_proof_graph"

    # 1. Global Setup
    if CLEAN_RUN:
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
        incubator_tag = f"Incubator{tag}"

        incubator_config = PROJECT_ROOT / "files" / "config" / f"Config{incubator_tag}.json"
        assert incubator_config.exists(), f"Incubator config not found: {incubator_config}"

        if RUN_INCUBATOR:
            # Provide main proved theorems as incubator externals
            if prev_tags:
                main_proved = theorems_dir / "proved_theorems.txt"
                incubator_ext = incubator_theorems_dir / "externally_provided_theorems.txt"
                with open(main_proved, "r", encoding="utf-8") as src:
                    content = src.read()
                if content.strip():
                    with open(incubator_ext, "w", encoding="utf-8") as dst:
                        dst.write(content)
                    _rebuild_compressed_externals(incubator_theorems_dir)

            # Incubator stage
            print(f"\n=== Incubator {tag} ===")
            _run_batch(incubator_tag, prev_tags=prev_tags,
                       theorems_dir=incubator_theorems_dir)

        else:
            print(f"\n=== Skipping Incubator {tag} ===")

        # Process incubator proof graph so global_theorem_list.txt exists
        # before convert_incubator_theorems reads it
        if RUN_INCUBATOR:
            visu_config_path = PROJECT_ROOT / "files" / "config" / "ConfigVisu.json"
            configuration_visu = configuration_reader(visu_config_path)
            process_proof_graphs.create_processed_proof_graph(
                configuration_visu, raw_dir=incubator_raw_dir, proc_dir=incubator_proc_dir,
                theorems_dir=incubator_theorems_dir)

        # Convert incubator proved theorems to simple facts (always, even if incubator skipped)
        convert_incubator_theorems(tag, theorems_dir=incubator_theorems_dir)

        if not RUN_MAIN_PATH:
            print(f"\n=== Skipping Main {tag} ===")
            continue

        # Main stage
        print(f"\n=== {tag} ===")
        _run_batch(tag, prev_tags=prev_tags,
                   simple_facts_fn=SIMPLE_FACTS_MAP.get(tag))

    # 3. Finalization
    visu_config_path = PROJECT_ROOT / "files" / "config" / "ConfigVisu.json"
    configuration_visu = configuration_reader(visu_config_path)

    # 3a. Incubator HTML (processed proof graph already built in the tag loop)
    if RUN_INCUBATOR:
        print("\n--- Generating Incubator Proof Graph HTML ---")
        generate_full_proof_graph.generate_proof_graph_pages(
            configuration_visu, proc_dir=incubator_proc_dir, out_dir=incubator_full_dir)

    # 3b. Main proof graph
    if RUN_MAIN_PATH:
        print("\n--- Generating Proof Graph ---")
        process_proof_graphs.create_processed_proof_graph(configuration_visu)
        generate_full_proof_graph.generate_proof_graph_pages(configuration_visu)

    # 4. Verification
    all_success = 0
    all_failure = 0

    if RUN_INCUBATOR:
        print("\n--- Running Incubator Proof Graph Verifier ---")
        incubator_state = verifier.run_verifier(str(incubator_proc_dir))
        verifier.print_report(incubator_state)
        s, f = verifier.get_totals(incubator_state)
        all_success += s
        all_failure += f

    if RUN_MAIN_PATH:
        print("\n--- Running Proof Graph Verifier ---")
        main_state = verifier.run_verifier(str(PROJECT_ROOT / "files" / "processed_proof_graph"))
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
