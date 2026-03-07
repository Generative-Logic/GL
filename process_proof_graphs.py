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


def _find_matching_paren(expr: str, start: int) -> int:
    """Return the index of the ')' matching '(' at position *start*."""
    depth = 0
    for i in range(start, len(expr)):
        if expr[i] == '(':
            depth += 1
        elif expr[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    return -1


def disintegrate_implication_full(expr: str) -> tuple[list[str], str]:
    """Peel implication layers and return (premises, head)."""
    premises: list[str] = []

    while expr.startswith('(>['):
        try:
            bracket_close = expr.index(']', 3)
        except ValueError:
            break

        premise_start = bracket_close + 1
        if premise_start >= len(expr) or expr[premise_start] != '(':
            break

        premise_end = _find_matching_paren(expr, premise_start)
        if premise_end < 0:
            break

        body_start = premise_end + 1
        if body_start >= len(expr) or expr[body_start] != '(':
            break

        body_end = _find_matching_paren(expr, body_start)
        if body_end < 0:
            break

        premises.append(expr[premise_start:premise_end + 1])
        expr = expr[body_start:body_end + 1]

    return premises, expr


def is_theorem_anchor_implication(expr: str) -> bool:
    """True only for theorem implications whose first premise is an Anchor expression."""
    if not expr or not expr.startswith("(>["):
        return False

    premises, _ = disintegrate_implication_full(expr)
    if not premises:
        return False

    return premises[0].startswith("(Anchor")


def strip_anchor_vars_from_outer_implication(expr: str) -> str:
    """For anchored theorem implications, clear the outer >[...] binder list."""
    if not is_theorem_anchor_implication(expr):
        return expr

    try:
        bracket_close = expr.index(']', 3)
    except ValueError:
        return expr

    return '(>[]' + expr[bracket_close + 1:]


def normalize_anchor_implications(raw_theorems, raw_stacks):
    """No-op: anchor implications are kept with their original >[...] binder lists."""
    return raw_theorems, raw_stacks


def _build_fname_list(theorems):
    """
    Given a list of theorem entries [expr, method, var, ...],
    return a parallel list of (fname_list) for each entry.
    """
    result = []
    idx = 0
    for parts in theorems:
        method = parts[1].lower() if len(parts) > 1 else ""
        if method == "induction":
            fnames = [f"{idx}_check_zero.txt", f"{idx + 1}_check_induction_condition.txt"]
            idx += 2
        elif method == "direct":
            fnames = [f"{idx}_direct_proof.txt"]
            idx += 1
        elif method == "debug":
            fnames = [f"{idx}_debug.txt"]
            idx += 1
        elif method == "mirrored statement":
            fnames = [f"{idx}_mirrored_statement.txt"]
            idx += 1
        elif method == "reformulated statement":
            fnames = [f"{idx}_reformulated_statement.txt"]
            idx += 1
        else:
            safe = re.sub(r"[^A-Za-z0-9._\-+]+", "_", method)[:64] or "unknown"
            fnames = [f"{idx}_unknown_{safe}.txt"]
            idx += 1
        result.append(fnames)
    return result


def _prune_proof_graph(raw_theorems, raw_stacks):
    """
    Remove theorems that did NOT survive pruning AND are not needed
    by any surviving theorem's proof chain.

    Returns (filtered_theorems, filtered_stacks) with re-indexed filenames.
    """
    proved_file = PROJECT_ROOT / "files" / "theorems" / "proved_theorems.txt"
    essential = set()
    if proved_file.exists():
        with open(proved_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    essential.add(line)

    if not essential:
        return raw_theorems, raw_stacks

    all_thm_exprs = set(t[0] for t in raw_theorems)

    # Map each theorem → its old stack filenames
    fname_lists = _build_fname_list(raw_theorems)
    expr_to_old_fnames: dict[str, list[str]] = {}
    for i, parts in enumerate(raw_theorems):
        expr_to_old_fnames.setdefault(parts[0], []).extend(fname_lists[i])

    # Precompute var-field deps (mirrored/reformulated → source theorem)
    var_deps: dict[str, set[str]] = {}
    for parts in raw_theorems:
        expr = parts[0]
        method = parts[1].lower() if len(parts) > 1 else ""
        var_field = parts[2] if len(parts) > 2 else ""
        if method in ("mirrored statement", "reformulated statement") and var_field in all_thm_exprs:
            var_deps.setdefault(expr, set()).add(var_field)

    # ---- Collect dependencies from proof stacks (exact match) ----
    def get_deps(expr):
        deps = set()
        for fname in expr_to_old_fnames.get(expr, []):
            for row in raw_stacks.get(fname, []):
                for cell in row:
                    if cell in all_thm_exprs:
                        deps.add(cell)
        # Var-field deps
        deps.update(var_deps.get(expr, set()))
        deps.discard(expr)
        return deps

    # ---- BFS from essential theorems ----
    needed = set()
    queue = list(essential & all_thm_exprs)
    visited = set(queue)
    while queue:
        thm = queue.pop(0)
        needed.add(thm)
        for dep in get_deps(thm):
            if dep not in visited:
                visited.add(dep)
                queue.append(dep)

    # ---- Filter theorems and re-index stack files ----
    filtered_theorems = [t for t in raw_theorems if t[0] in needed]
    new_fname_lists = _build_fname_list(filtered_theorems)

    filtered_stacks = {}
    for i, parts in enumerate(filtered_theorems):
        old_fnames = expr_to_old_fnames.get(parts[0], [])
        new_fnames = new_fname_lists[i]
        for old_fn, new_fn in zip(old_fnames, new_fnames):
            if old_fn in raw_stacks:
                filtered_stacks[new_fn] = raw_stacks[old_fn]

    removed = len(raw_theorems) - len(filtered_theorems)
    print(f"Proof graph pruning: kept {len(filtered_theorems)}, removed {removed} unused theorems.")
    return filtered_theorems, filtered_stacks


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

    # Load all raw stacks into RAM
    raw_stacks = {}
    for file in raw_dir.glob("*.txt"):
        if file.name in ("global_theorem_list.txt",):
            continue
        stack = []
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n\r")
                stack.append(line.split("\t") if line else [])
        raw_stacks[file.name] = stack

    # -------------------------------------------------------------------------
    # STEP 0: Prune proof graph — remove unused non-essential theorems
    # -------------------------------------------------------------------------
    raw_theorems, raw_stacks = _prune_proof_graph(raw_theorems, raw_stacks)

    # -------------------------------------------------------------------------
    # STEP 1: Normalize anchored theorem implications before local renaming
    # -------------------------------------------------------------------------
    raw_theorems, raw_stacks = normalize_anchor_implications(raw_theorems, raw_stacks)

    # Rebuild derived indices from (possibly filtered) raw_theorems
    raw_thm_to_idx = {thm[0]: i for i, thm in enumerate(raw_theorems)}

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
        elif method == "reformulated statement":
            fname_to_raw_thm[f"{file_idx}_reformulated_statement.txt"] = thm_expr
            file_idx += 1
        else:
            safe = re.sub(r"[^A-Za-z0-9._\-+]+", "_", method)[:64] or "unknown"
            fname_to_raw_thm[f"{file_idx}_unknown_{safe}.txt"] = thm_expr
            file_idx += 1

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
                is_theorem_impl = is_theorem_anchor_implication(orig_cell)

                # Rename implication-local variables for every non-theorem implication.
                # Theorem implications are handled later via the global theorem list.
                if is_impl and not is_theorem_impl:
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

        # Priority 2: Seed v-numbering from the theorem expression itself
        #             (left-to-right scan — same order used for the global theorem list)
        var_counter = 1
        if raw_thm_expr:
            for a in get_all_args(raw_thm_expr):
                if a not in repl_map:
                    repl_map[a] = f"v{var_counter}"
                    var_counter += 1

        # Priority 3: Iterate over remaining args in chapter lines
        for r_idx, row in enumerate(stack):
            for c_idx, cell in enumerate(row):
                if c_idx == 2 or cell == "main":
                    continue

                orig_cell = raw_stack[r_idx][c_idx]
                is_impl = orig_cell.startswith("(>[")
                is_theorem_impl = is_theorem_anchor_implication(orig_cell)

                # Ignore theorem implications here. They are replaced wholesale later
                # so they stay aligned with the globally renamed theorem list.
                if is_impl and is_theorem_impl:
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

        # Also rename the induction variable (field 2) for induction theorems
        method = thm_row[1].lower() if len(thm_row) > 1 else ""
        if method == "induction" and len(thm_row) > 2:
            raw_ind_var = thm_row[2]
            renamed_theorems[i][2] = thm_repl_map.get(raw_ind_var, raw_ind_var)

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
                is_theorem_impl = is_theorem_anchor_implication(orig_cell)

                # Do not apply the local chapter map to theorem implications.
                if is_impl and is_theorem_impl:
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
                is_theorem_impl = is_theorem_anchor_implication(orig_cell)

                # Replace theorem implications with their global renamed version.
                if is_impl and is_theorem_impl:
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

    print(f"Success: Processed proof graphs have been written to {proc_dir}")