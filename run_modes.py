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


import create_expressions
from simple_facts_peano import make_simple_facts_peano
from simple_facts_gauss import make_simple_facts_gauss
import test_create_expressions
import generate_full_proof_graph
import time
import shutil
import os
from configuration_reader import configuration_reader
from pathlib import Path
import subprocess
import process_proof_graphs

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


def generate_anchor_connection(current_tag: str, prev_tag: str):
    """
    Connects current tag's anchor (A) to previous tag's anchor (B) as A -> B.
    (e.g., AnchorGauss -> AnchorPeano)
    Appends the result to theorems.txt.
    """
    # 1. Load Configurations
    config_current = configuration_reader(PROJECT_ROOT / "files" / "config" / f"Config{current_tag}.json")
    config_prev = configuration_reader(PROJECT_ROOT / "files" / "config" / f"Config{prev_tag}.json")

    # 2. Get Anchor Expressions (Normalized: e.g. (Anchor[1,2,3]))
    anchor_name_current = create_expressions.get_anchor_name(config_current)
    anchor_name_prev = create_expressions.get_anchor_name(config_prev)

    if not anchor_name_current or not anchor_name_prev:
        print(f"Warning: Could not find anchor names for {current_tag} or {prev_tag}.")
        return

    expr_A = config_current[anchor_name_current].short_mpl_normalized  # Current (e.g. Gauss)
    expr_B = config_prev[anchor_name_prev].short_mpl_normalized  # Previous (e.g. Peano)

    # 3. Get Arguments
    args_A = create_expressions.get_args(expr_A)
    args_B = create_expressions.get_args(expr_B)

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
    theorems_path = PROJECT_ROOT / "files" / "theorems" / "theorems.txt"
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

def full_run():
    """
    """




    # 1. Global Setup (Run Once)
    empty_simple_facts()
    empty_raw_proof_graph()  # <--- NEW: Clear proof graph only once at the start

    # Empty theorem output files at the very start
    theorems_dir = PROJECT_ROOT / "files" / "theorems"
    theorems_dir.mkdir(parents=True, exist_ok=True)

    for fname in [
        "proved_theorems.txt",          # cross-batch propagation (essential only)
        "compressed_out_theorems.txt",   # diagnostic: compressor's rejected set
    ]:
        fpath = theorems_dir / fname
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write("")

    # proved_theorems.txt is rewritten after each batch with the globally compressed set
    tags = ["Peano", "Gauss"]

    # 2. Iterative Execution
    for i, tag in enumerate(tags):
        print(f"\n--- Starting run for tag: {tag} ---")

        config_path = PROJECT_ROOT / "files" / "config" / f"Config{tag}.json"
        if not config_path.exists():
            print(f"Warning: Configuration file {config_path} not found. Skipping.")
            continue

        config = configuration_reader(config_path)

        # A. Create Simple Facts
        if tag == "Peano":
            for n in config.parameters.simple_facts_parameters:
                make_simple_facts_peano(n)
        elif tag == "Gauss":
            for n in config.parameters.simple_facts_parameters:
                make_simple_facts_gauss(n)
        else:
            print(f"No simple facts generator mapped for tag: {tag}")

        # B. Create Expressions
        start_time = time.time()
        print(f"Conjecture creation started for {tag}.")
        create_expressions.create_expressions_parallel(config)
        print(f"Conjecture creation finished for {tag}.")
        end_time = time.time()
        print(f"Conjecture creation runtime: {end_time - start_time:.5f} seconds")

        # C. Connect to Previous Anchors (Current -> Prev)
        # Iterate through all tags strictly before the current one
        previous_tags = tags[:i]
        for prev_tag in previous_tags:
            generate_anchor_connection(tag, prev_tag)

        # D. Run Tests (Commented out)
        #print(f"Running tests for {tag}...")
        #test_create_expressions.test1(tag)
        #test_create_expressions.test2(tag)

        # E. Run Native Prover
        print(f"Running GL_Quick for {tag}...")
        run_gl_quick(tag)




    # 3. Finalization
    print("\n--- Generating Proof Graph ---")
    visu_config_path = PROJECT_ROOT / "files" / "config" / "ConfigVisu.json"
    configuration_visu = configuration_reader(visu_config_path)
    process_proof_graphs.create_processed_proof_graph(configuration_visu)
    generate_full_proof_graph.generate_proof_graph_pages(configuration_visu)


