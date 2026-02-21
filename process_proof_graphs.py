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


import re
import copy
import shutil
from pathlib import Path

from configuration_reader import configuration_reader
import create_expressions

# Assume the project root is the folder containing this file
PROJECT_ROOT = Path(__file__).resolve().parent


def get_all_args(expr: str) -> list[str]:
    """Extracts all arguments from an expression."""
    if not expr:
        return []
    pattern = r'(?<=[\[,])([^,\[\]]+)(?=[\],])'
    return list(dict.fromkeys(re.findall(pattern, expr)))


def replace_keys_in_string(big_string: str, replacement_map: dict[str, str]) -> str:
    """Replaces argument keys safely using regex lookarounds."""
    if not replacement_map:
        return big_string
    escaped_keys = [re.escape(k) for k in replacement_map.keys()]
    pattern = r'(?<=[\[,])(' + '|'.join(escaped_keys) + r')(?=[\],])'
    regex = re.compile(pattern)
    return regex.sub(lambda m: replacement_map.get(m.group(1), m.group(1)), big_string)


def get_anchor_mapping_from_expr(expr_str: str, config: configuration_reader) -> dict:
    """Finds Anchor in theorem string and maps its args to config's short_mpl_raw."""
    repl_map = {}
    if not expr_str:
        return repl_map

    for match in re.finditer(r'\(Anchor([A-Za-z0-9_]+)\[(.*?)\]\)', expr_str):
        anchor_name = "Anchor" + match.group(1)
        stack_args = get_all_args(match.group(0))

        if anchor_name in config:
            short_mpl_raw = config[anchor_name].short_mpl_raw
            config_args = get_all_args(short_mpl_raw)
            for s_arg, c_arg in zip(stack_args, config_args):
                if s_arg not in repl_map:
                    repl_map[s_arg] = c_arg

    return repl_map


def create_processed_proof_graph(config: configuration_reader):
    create_expressions.set_configuration(config)

    raw_dir = PROJECT_ROOT / "files" / "raw_proof_graph"
    proc_dir = PROJECT_ROOT / "files" / "processed_proof_graph"

    if proc_dir.exists():
        shutil.rmtree(proc_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Load raw global theorem list
    raw_theorems = []
    global_list_path = raw_dir / "global_theorem_list.txt"
    if global_list_path.exists():
        with open(global_list_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n\r").split("\t")
                if len(parts) >= 3:
                    raw_theorems.append(parts)

    raw_thm_to_idx = {thm[0]: i for i, thm in enumerate(raw_theorems)}

    # Build exact map from filename -> raw_thm_expr to fix index misalignment
    fname_to_raw_thm = {}
    file_idx = 0
    for parts in raw_theorems:
        thm_expr = parts[0]
        method = parts[1].lower() if len(parts) > 1 else ""
        if method == "induction":
            fname_to_raw_thm[f"{file_idx}_check_zero.txt"] = thm_expr
            fname_to_raw_thm[f"{file_idx + 1}_check_induction_condition.txt"] = thm_expr
            file_idx += 2
        elif method == "direct":
            fname_to_raw_thm[f"{file_idx}_direct_proof.txt"] = thm_expr
            file_idx += 1
        elif method == "debug":
            fname_to_raw_thm[f"{file_idx}_debug.txt"] = thm_expr
            file_idx += 1
        elif method == "mirrored statement":
            fname_to_raw_thm[f"{file_idx}_mirrored_statement.txt"] = thm_expr
            file_idx += 1
        else:
            safe = re.sub(r"[^A-Za-z0-9._\-+]+", "_", method)[:64] or "unknown"
            fname_to_raw_thm[f"{file_idx}_unknown_{safe}.txt"] = thm_expr
            file_idx += 1

    # Load all raw stacks into RAM
    raw_stacks = {}
    for file in raw_dir.glob("*.txt"):
        if file.name in ("global_theorem_list.txt", "compiled_expressions.txt"):
            continue
        stack = []
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n\r")
                stack.append(line.split("\t") if line else [])
        raw_stacks[file.name] = stack

    ram_stacks = copy.deepcopy(raw_stacks)

    # -------------------------------------------------------------------------
    # ITERATION 1: Handle 'u_' variables in unanchored implications
    # -------------------------------------------------------------------------
    for fname, stack in ram_stacks.items():
        raw_stack = raw_stacks[fname]
        for r_idx, row in enumerate(stack):
            for c_idx, cell in enumerate(row):
                if c_idx == 2 or cell == "main":
                    continue

                orig_cell = raw_stack[r_idx][c_idx]
                is_impl = orig_cell.startswith("(>[")
                has_anchor = "(Anchor" in orig_cell

                # Strictly ignore anchored implications here
                if is_impl and not has_anchor:
                    args = get_all_args(cell)
                    has_u = any(a.startswith("u_") for a in args)

                    if has_u:
                        repl_map = {}
                        for a in args:
                            if a.startswith("u_"):
                                repl_map[a] = a[2:]  # Remove 'u_'
                            else:
                                repl_map[a] = "c_" + a  # Add 'c_'
                        ram_stacks[fname][r_idx][c_idx] = replace_keys_in_string(cell, repl_map)

    # -------------------------------------------------------------------------
    # ITERATION 2: Build a universal replacement map per stack
    # -------------------------------------------------------------------------
    stack_repl_maps = {}
    for fname, stack in ram_stacks.items():
        raw_stack = raw_stacks[fname]
        repl_map = {}

        # Priority 1: Extract vars from the exact global theorem proved by this stack
        raw_thm_expr = fname_to_raw_thm.get(fname)
        if raw_thm_expr:
            repl_map.update(get_anchor_mapping_from_expr(raw_thm_expr, config))

        # Priority 2: Iterate over remaining args
        var_counter = 1
        for r_idx, row in enumerate(stack):
            for c_idx, cell in enumerate(row):
                if c_idx == 2 or cell == "main":
                    continue

                orig_cell = raw_stack[r_idx][c_idx]
                is_impl = orig_cell.startswith("(>[")
                has_anchor = "(Anchor" in orig_cell

                # IGNORE anchored implications entirely
                if is_impl and has_anchor:
                    continue

                # ALL other expressions (including Iteration 1's modified unanchored implications) are processed!
                args = get_all_args(cell)
                for a in args:
                    if a not in repl_map:
                        repl_map[a] = f"v{var_counter}"
                        var_counter += 1

        stack_repl_maps[fname] = repl_map

    # Generate the processed version of the global theorem list
    renamed_theorems = copy.deepcopy(raw_theorems)
    for i, thm_row in enumerate(renamed_theorems):
        raw_thm_expr = thm_row[0]

        thm_repl_map = get_anchor_mapping_from_expr(raw_thm_expr, config)
        var_c = 1
        for a in get_all_args(raw_thm_expr):
            if a not in thm_repl_map:
                thm_repl_map[a] = f"v{var_c}"
                var_c += 1

        renamed_theorems[i][0] = replace_keys_in_string(raw_thm_expr, thm_repl_map)

    # -------------------------------------------------------------------------
    # ITERATION 3: Apply the replacement maps
    # -------------------------------------------------------------------------
    for fname, stack in ram_stacks.items():
        repl_map = stack_repl_maps[fname]
        raw_stack = raw_stacks[fname]

        for r_idx, row in enumerate(stack):
            for c_idx, cell in enumerate(row):
                if c_idx == 2 or cell == "main":
                    continue

                orig_cell = raw_stack[r_idx][c_idx]
                is_impl = orig_cell.startswith("(>[")
                has_anchor = "(Anchor" in orig_cell

                # DO NOT apply map to anchored implications
                if is_impl and has_anchor:
                    continue

                # Apply map to the current cell (Iter 1 mods are safely mapped here)
                ram_stacks[fname][r_idx][c_idx] = replace_keys_in_string(cell, repl_map)

    # -------------------------------------------------------------------------
    # ITERATION 4: Deal with anchored implications across the global context
    # -------------------------------------------------------------------------
    for fname, stack in ram_stacks.items():
        raw_stack = raw_stacks[fname]
        for r_idx, row in enumerate(stack):
            for c_idx, cell in enumerate(row):
                if c_idx == 2 or cell == "main":
                    continue

                orig_cell = raw_stack[r_idx][c_idx]
                is_impl = orig_cell.startswith("(>[")
                has_anchor = "(Anchor" in orig_cell

                # Strictly replace the anchored implications with their global renamed version
                if is_impl and has_anchor:
                    if orig_cell in raw_thm_to_idx:
                        global_idx = raw_thm_to_idx[orig_cell]
                        ram_stacks[fname][r_idx][c_idx] = renamed_theorems[global_idx][0]

    # -------------------------------------------------------------------------
    # Final Save
    # -------------------------------------------------------------------------
    for fname, stack in ram_stacks.items():
        with open(proc_dir / fname, "w", encoding="utf-8") as f:
            for row in stack:
                f.write("\t".join(row) + "\n")

    with open(proc_dir / "global_theorem_list.txt", "w", encoding="utf-8") as f:
        for thm in renamed_theorems:
            f.write("\t".join(thm) + "\n")

    comp_expr = raw_dir / "compiled_expressions.txt"
    if comp_expr.exists():
        shutil.copy(comp_expr, proc_dir / "compiled_expressions.txt")

    print(f"Success: Processed proof graphs have been written to {proc_dir}")