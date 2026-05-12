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


import re
import copy
import shutil
from pathlib import Path

from configuration_reader import configuration_reader
import expression_utils
from expression_utils import replace_keys_in_string

# Assume the project root is the folder containing this file
PROJECT_ROOT = Path(__file__).resolve().parent


def get_all_args(expr: str) -> list[str]:
    """Extracts all arguments from an expression."""
    if not expr:
        return []
    pattern = r'(?<=[\[,])([^,\[\]]+)(?=[\],])'
    return list(dict.fromkeys(re.findall(pattern, expr)))


# Matches a single non-meta sub-expression of the form (opname[a1,a2,...]).
# Skips '&' and '>' meta-shapes (their first char isn't a letter).
_OP_SUBEXPR_RE = re.compile(r'\(([A-Za-z][A-Za-z0-9_]*)\[([^\[\]]+)\]\)')


def classify_args_by_defset(expr: str, config) -> dict[str, str]:
    """For every arg in expr, infer 'digit' / 'set' / 'unknown' from the
    operator + position context.

    Walks every (opName[args]) sub-expression, looks up
    config[opName].definition_sets[1-based position]. The defset text is
    something like '(1)' (digit / element of N) or 'P(...)' (set / power
    set of something). Anchor* sub-expressions are skipped — their args
    are concrete pre-renamed names handled by the anchor-mapping step.

    First-observed-type wins per arg. Cross-type usage is silently tolerated
    (a raw bound-variable index can legitimately appear in different
    implication scopes within the same chapter and end up in slots of
    incompatible defset; the per-chapter merge in create_processed_proof_graph
    treats this as "first wins" because there is no globally-correct case
    for a variable used both ways and the rendering is purely visual).

    Returns: {arg: 'digit' | 'set' | 'unknown'}.
    """
    if not expr:
        return {}
    out: dict[str, str] = {}
    for m in _OP_SUBEXPR_RE.finditer(expr):
        op_name = m.group(1)
        if op_name.startswith("Anchor"):
            continue
        if op_name not in config:
            continue
        defsets = config[op_name].definition_sets  # {pos_str: (text, _, _)}
        op_args = [a.strip() for a in m.group(2).split(",")]
        for idx, arg in enumerate(op_args):
            pos_key = str(idx + 1)
            if pos_key not in defsets:
                continue
            defset_text = defsets[pos_key][0]
            if defset_text == "(1)":
                tag = "digit"
            elif defset_text.startswith("P("):
                tag = "set"
            else:
                tag = "unknown"
            prior = out.get(arg)
            if prior is None or prior == "unknown":
                out[arg] = tag
            # else: prior in {'digit','set'} and (tag in {'digit','set','unknown'})
            # — keep prior. First-observed wins, including across digit/set
            # conflicts. See docstring.
    return out


def _mint_typed_var(arg: str, arg_type: dict[str, str], counters: dict[str, int]) -> str:
    """Pick the case (v / V) based on arg_type and bump the matching counter.

    counters is a mutable dict with keys 'digit' and 'set'; both start at 1.
    'unknown' falls back to 'digit' (lowercase v) since most undeducible
    args turn out to be digits in practice.
    """
    tag = arg_type.get(arg, "unknown")
    if tag == "set":
        name = f"V{counters['set']}"
        counters["set"] += 1
    else:
        name = f"v{counters['digit']}"
        counters["digit"] += 1
    return name


def w_rename_impl_local(raw_cell: str, current_cell: str, config) -> str:
    """Per-cell rename of bound vars whose RAW form is `\\d+` to w<N> /
    W<N>. Bvars whose raw form is anything else (`repl_lev_*`,
    `it_0_lev_*`, …) keep their Pass-1 v/V name with the chapter-
    global counter.

    Walks `raw_cell` and `current_cell` in parallel through their
    non-anchor `(>[bvars]…)` brackets. For each bvar position, if the
    RAW bvar matches `^\\d+$`, the corresponding current bvar (a
    v-name) gets w-renamed; otherwise it stays.

    Counter resets per cell-call (so context is local to one
    implication tree). Within a call the counter is monotonic — same
    raw `\\d+` appearing in two nested `>[…]` of the same cell maps
    to the same w-name (preserves shadow when the prover wrote it).
    """
    if '(>[' not in current_cell:
        return current_cell

    raw_lists = _collect_non_anchor_bvar_lists(raw_cell)
    cur_lists = _collect_non_anchor_bvar_lists(current_cell)

    if len(raw_lists) != len(cur_lists):
        return current_cell

    arg_types = classify_args_by_defset(current_cell, config)
    digit_n = 1
    set_n = 1
    rmap: dict[str, str] = {}
    for raw_bvars, cur_bvars in zip(raw_lists, cur_lists):
        if len(raw_bvars) != len(cur_bvars):
            continue
        for rb, cb_name in zip(raw_bvars, cur_bvars):
            if not re.fullmatch(r'\d+', rb):
                continue
            if cb_name in rmap:
                continue
            t = arg_types.get(cb_name, 'unknown')
            if t == 'set':
                rmap[cb_name] = f'W{set_n}'
                set_n += 1
            else:
                rmap[cb_name] = f'w{digit_n}'
                digit_n += 1

    if not rmap:
        return current_cell
    return replace_keys_in_string(current_cell, rmap)


def _collect_non_anchor_bvar_lists(cell: str) -> list[list[str]]:
    """Walk `cell` and return the bvar list of every non-anchor
    `(>[bvars]…)` in order of appearance (skips anchor implications)."""
    out: list[list[str]] = []
    i = 0
    while i < len(cell):
        if cell.startswith('(>[', i):
            cb = cell.find(']', i + 3)
            if cb < 0:
                i += 1
                continue
            impl_end = _find_matching_paren(cell, i)
            if impl_end < 0:
                i += 1
                continue
            inner = cell[cb + 1:impl_end]
            if not inner.startswith('(Anchor'):
                bvars = [v.strip() for v in cell[i + 3:cb].split(',') if v.strip()]
                out.append(bvars)
            i = cb + 1
        else:
            i += 1
    return out


_VW_SWAP_PATTERN = re.compile(r'(?<=[\[,])([vV])(\d+)(?=[\],])')


def _v_to_w_in_theorem_citation(expr: str) -> str:
    """Case-preserving v->w / V->W swap on every [vV]\\d+ token at argument
    positions of `expr`. Caller must guarantee `expr` is a theorem-anchor
    implication being cited — never the chapter's own HEAD claim, never the
    global theorem registry.

    Anchor slot names (N, i0, s, +, *, i1, i2, id, ...) never match
    [vV]\\d+, so they are left alone. `<digit>_copy` forms (e.g. `v1_copy`)
    are also left alone — the lookahead requires `,` or `]` immediately
    after the digits, which `_copy` violates.

    Counterpart: verifier.py defines _revert_w_to_v_in_theorem_citation as
    the symmetric inverse, used at every chapter-cell-to-registry lookup
    site so the chapter's w/W form maps back to the registry's v/V form
    by exact string equality.
    """
    return _VW_SWAP_PATTERN.sub(
        lambda m: ('w' if m.group(1) == 'v' else 'W') + m.group(2),
        expr,
    )


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


def rename_external_theorem(raw_expr: str, config) -> str:
    """Rename a raw external-theorem expression to v/V form using the
    visualizer config's anchor definitions.

    Externals are theorems pulled in from a sibling batch (e.g. an incubator
    chapter cites a Peano-batch theorem). They arrive in *raw numeric* form
    where every argument is a digit-string placeholder
    (`(>[1,2,3](AnchorPeano[1,2,3,4,5,6])(>[7](in[7,1])(=[7,2])))`). To
    render them consistently with internally-proven theorems and to keep
    HTML cross-citation matching robust, we rename:

    - **Anchor-slot positions** (positions 1..arity of the first premise's
      Anchor expression) get the anchor's canonical slot names from
      `config[anchor_name].short_mpl_raw` — e.g. `1->N, 2->i0, 3->s,
      4->+, 5->*, 6->i1` for AnchorPeano.
    - **Non-anchor bound variables** (the inner `>[...]` binders' names)
      get typed names via `_mint_typed_var` — `v1, v2, ...` for digit-
      typed positions, `V1, V2, ...` for set-typed. The case-preserving
      v->w/V->W swap in iteration 5 of `create_processed_proof_graph`
      then converts these to `w<N>/W<N>` for citation form, matching how
      internal-theorem citations are emitted.

    Returns the renamed expression. If the expression is not a theorem-
    anchor implication, has no recognizable external anchor, or the anchor
    is not declared in `config.external_anchors`, returns the input
    unchanged so the caller's downstream fallback (chapter-local repl_map)
    still applies.

    See ConfigVisu.json's `external_anchors` field. Default behaviour when
    the field is empty (no externals declared): no-op.
    """
    if not is_theorem_anchor_implication(raw_expr):
        return raw_expr

    premises, _ = disintegrate_implication_full(raw_expr)
    if not premises:
        return raw_expr

    anchor_match = re.match(r'\(Anchor([A-Za-z0-9_]+)\[', premises[0])
    if not anchor_match:
        return raw_expr
    anchor_name = "Anchor" + anchor_match.group(1)

    declared = list(getattr(config, "external_anchors", []) or [])
    if anchor_name not in declared:
        return raw_expr
    if anchor_name not in config:
        return raw_expr

    anchor_short_mpl = config[anchor_name].short_mpl_raw
    anchor_slot_names = get_all_args(anchor_short_mpl)
    raw_anchor_args = get_all_args(premises[0])

    repl_map: dict[str, str] = {}
    for raw_arg, slot_name in zip(raw_anchor_args, anchor_slot_names):
        repl_map.setdefault(raw_arg, slot_name)

    arg_types = classify_args_by_defset(raw_expr, config)
    counters = {"digit": 1, "set": 1}
    for a in get_all_args(raw_expr):
        if a not in repl_map:
            repl_map[a] = _mint_typed_var(a, arg_types, counters)

    return replace_keys_in_string(raw_expr, repl_map)


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
            fnames = [
                f"{idx}_induction_typing.txt",
                f"{idx + 1}_check_zero.txt",
                f"{idx + 2}_check_induction_condition.txt",
            ]
            idx += 3
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
        elif method == "incubator back reformulation":
            fnames = [f"{idx}_back_reformulated_statement.txt"]
            idx += 1
        elif method == "or theorem":
            fnames = [f"{idx}_or_theorem.txt"]
            idx += 1
        else:
            safe = re.sub(r"[^A-Za-z0-9._\-+]+", "_", method)[:64] or "unknown"
            fnames = [f"{idx}_unknown_{safe}.txt"]
            idx += 1
        result.append(fnames)
    return result


def _prune_proof_graph(raw_theorems, raw_stacks, theorems_dir=None):
    """
    Remove theorems that did NOT survive pruning AND are not needed
    by any surviving theorem's proof chain.

    Returns (filtered_theorems, filtered_stacks) with re-indexed filenames.
    """
    if theorems_dir is None:
        theorems_dir = PROJECT_ROOT / "files" / "theorems"
    proved_file = theorems_dir / "compiled_proved_theorems.txt"
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
    if not queue:
        return raw_theorems, raw_stacks
    visited = set(queue)
    while queue:
        thm = queue.pop(0)
        needed.add(thm)
        for dep in get_deps(thm):
            if dep not in visited:
                visited.add(dep)
                queue.append(dep)

    # ---- Remove external theorems and their mirrored variants ----
    ext_file = theorems_dir / "compressed_external_theorems.txt"
    external_theorems = set()
    if ext_file.exists():
        with open(ext_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    external_theorems.add(line)

    if external_theorems:
        mirrored_of_external = set()
        for parts in raw_theorems:
            expr = parts[0]
            method = parts[1].lower() if len(parts) > 1 else ""
            ref = parts[2] if len(parts) > 2 else ""
            if method == "mirrored statement" and ref in external_theorems:
                mirrored_of_external.add(expr)

        to_remove = (external_theorems | mirrored_of_external) & all_thm_exprs
        needed -= to_remove

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


def create_processed_proof_graph(config: configuration_reader,
                                 raw_dir=None, proc_dir=None,
                                 theorems_dir=None):
    expression_utils.set_configuration(config)

    if raw_dir is None:
        raw_dir = PROJECT_ROOT / "files" / "raw_proof_graph"
    if proc_dir is None:
        proc_dir = PROJECT_ROOT / "files" / "processed_proof_graph"
    if theorems_dir is None:
        theorems_dir = PROJECT_ROOT / "files" / "theorems"

    if proc_dir.exists():
        # Remove contents but keep the directory itself — on Windows, the
        # top-level dir can briefly hold a handle (indexer / explorer) that
        # blocks os.rmdir even when the directory is empty.
        for child in proc_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
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
    theorems_dir = raw_dir.parent / "theorems"
    raw_theorems, raw_stacks = _prune_proof_graph(raw_theorems, raw_stacks, theorems_dir)

    # -------------------------------------------------------------------------
    # STEP 1: Normalize anchored theorem implications before local renaming
    # -------------------------------------------------------------------------
    raw_theorems, raw_stacks = normalize_anchor_implications(raw_theorems, raw_stacks)

    # Rebuild derived indices from (possibly filtered) raw_theorems
    raw_thm_to_idx = {thm[0]: i for i, thm in enumerate(raw_theorems)}

    # -------------------------------------------------------------------------
    # External-theorem rename map. Loaded once here so iteration 4 can use it
    # in the same fallback branch where internal-theorem rename via
    # raw_thm_to_idx fails (externals aren't in that index because they
    # come from a sibling batch's compiled output).
    #
    # The renamed form uses v/V for non-anchor bound vars; iteration 5's
    # v->w swap then converts to citation-form w/W, matching how internally-
    # proven theorems are emitted into chapter cells.
    # -------------------------------------------------------------------------
    raw_external_set: set[str] = set()
    ext_file_for_rename = theorems_dir / "compressed_external_theorems.txt" if theorems_dir else None
    if ext_file_for_rename and ext_file_for_rename.exists():
        with open(ext_file_for_rename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_external_set.add(line)
    external_renames: dict[str, str] = {}
    for raw_ext in raw_external_set:
        renamed_ext = rename_external_theorem(raw_ext, config)
        if renamed_ext != raw_ext:
            external_renames[raw_ext] = renamed_ext

    fname_to_raw_thm = {}
    file_idx = 0
    for parts in raw_theorems:
        thm_expr = parts[0]
        method = parts[1].lower() if len(parts) > 1 else ""
        if method == "induction":
            fname_to_raw_thm[f"{file_idx}_induction_typing.txt"] = thm_expr
            fname_to_raw_thm[f"{file_idx + 1}_check_zero.txt"] = thm_expr
            fname_to_raw_thm[f"{file_idx + 2}_check_induction_condition.txt"] = thm_expr
            file_idx += 3
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
        elif method == "incubator back reformulation":
            fname_to_raw_thm[f"{file_idx}_back_reformulated_statement.txt"] = thm_expr
            file_idx += 1
        elif method == "or theorem":
            fname_to_raw_thm[f"{file_idx}_or_theorem.txt"] = thm_expr
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
        # Build per-chapter type classification ONCE, from the union of the
        # theorem expression and every chapter cell, so digit/set typing is
        # stable across Priority 2 (theorem scan) and Priority 3 (chapter
        # scan). Independent counters per the "v for digit, V for set" scheme.
        chapter_arg_type: dict[str, str] = {}
        if raw_thm_expr:
            chapter_arg_type.update(classify_args_by_defset(raw_thm_expr, config))
        for r_idx_pre, row_pre in enumerate(stack):
            for c_idx_pre, cell_pre in enumerate(row_pre):
                if c_idx_pre == 2 or cell_pre == "main":
                    continue
                cell_types = classify_args_by_defset(cell_pre, config)
                for a_pre, t_pre in cell_types.items():
                    prior = chapter_arg_type.get(a_pre)
                    if prior is None or prior == "unknown":
                        chapter_arg_type[a_pre] = t_pre
                    # else: keep prior. First-observed wins (raw bound-vars
                    # can legitimately appear in cross-type slots across
                    # implication scopes within the same chapter; visual
                    # rendering picks one and lives with the imperfection).

        counters = {"digit": 1, "set": 1}
        if raw_thm_expr:
            for a in get_all_args(raw_thm_expr):
                if a not in repl_map:
                    repl_map[a] = _mint_typed_var(a, chapter_arg_type, counters)

        # Priority 2.5: Anchor handling — assign N_copy names to x-prefixed vars
        for row in raw_stack:
            if len(row) >= 3 and row[2] == "anchor handling":
                target_expr = row[0]
                anc_match = re.match(r'\(Anchor([A-Za-z0-9_]+)\[', target_expr)
                if anc_match:
                    anchor_name = "Anchor" + anc_match.group(1)
                    if anchor_name in config:
                        short_mpl_raw = config[anchor_name].short_mpl_raw
                        # Raw findall preserves positional ordering (no dedup)
                        target_args = re.findall(r'(?<=[\[,])([^,\[\]]+)(?=[\],])', target_expr)
                        config_args = re.findall(r'(?<=[\[,])([^,\[\]]+)(?=[\],])', short_mpl_raw)
                        for t_arg, c_arg in zip(target_args, config_args):
                            if t_arg.startswith("x") and t_arg not in repl_map:
                                m = re.match(r'^i(\d+)$', c_arg)
                                if m:
                                    repl_map[t_arg] = f"{m.group(1)}_copy"

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
                        repl_map[a] = _mint_typed_var(a, chapter_arg_type, counters)

        # Priority 4: Derive _copy variable names from their base variable
        for a in list(repl_map.keys()):
            if a.endswith("_copy"):
                base = a[:-5]
                if base in repl_map:
                    repl_map[a] = repl_map[base] + "_copy"

        stack_repl_maps[fname] = repl_map

    # Generate the processed version of the global theorem list
    renamed_theorems = copy.deepcopy(raw_theorems)
    for i, thm_row in enumerate(renamed_theorems):
        raw_thm_expr = thm_row[0]

        thm_repl_map = get_anchor_mapping_from_expr(raw_thm_expr, config)
        thm_arg_type = classify_args_by_defset(raw_thm_expr, config)
        thm_counters = {"digit": 1, "set": 1}
        for a in get_all_args(raw_thm_expr):
            if a not in thm_repl_map:
                thm_repl_map[a] = _mint_typed_var(a, thm_arg_type, thm_counters)

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
    # ITERATION 4½: w/W rename of bound vars whose RAW form is `\d+`,
    # per-cell counter starting at 1. Bvars whose raw form is anything
    # else (`repl_lev_*`, `it_0_lev_*`, …) keep their Pass-1 v-name
    # with the chapter-global counter. Free vars unchanged. Applies
    # to every cell whose raw form contains `(>[`, except theorem-
    # anchor-implications (replaced wholesale in ITER 4).
    # -------------------------------------------------------------------------
    for fname, stack in ram_stacks.items():
        raw_stack = raw_stacks[fname]
        for r_idx, row in enumerate(stack):
            for c_idx, cell in enumerate(row):
                if c_idx == 2 or cell == "main":
                    continue
                orig_cell = raw_stack[r_idx][c_idx]
                if '(>[' not in orig_cell:
                    continue
                is_impl = orig_cell.startswith("(>[")
                is_theorem_impl = is_theorem_anchor_implication(orig_cell)
                if is_impl and is_theorem_impl:
                    continue
                ram_stacks[fname][r_idx][c_idx] = w_rename_impl_local(
                    orig_cell, ram_stacks[fname][r_idx][c_idx], config)

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
                    elif orig_cell in external_renames:
                        # External-theorem citation: use the v/V renamed form
                        # built from ConfigVisu.external_anchors. Iteration 5
                        # below converts v/V -> w/W as for internal cites.
                        ram_stacks[fname][r_idx][c_idx] = external_renames[orig_cell]
                    else:
                        # Fallback: rename with chapter's own map (e.g. back-reformulated source)
                        repl_map = stack_repl_maps.get(fname, {})
                        if repl_map:
                            ram_stacks[fname][r_idx][c_idx] = replace_keys_in_string(cell, repl_map)

    # -------------------------------------------------------------------------
    # ITERATION 5: v->w / V->W on cited theorem-anchor implications.
    # Applies ONLY to cells at column index >= 3 (the rest fields, i.e. the
    # cited dependencies). Column 0 (the chapter's own HEAD claim) keeps
    # v/V — that is the "title" form, kept consistent with the global
    # theorem registry. global_theorem_list.txt is built independently from
    # renamed_theorems and stays in v/V.
    #
    # The case-preserving letter swap preserves the digit (v3 -> w3,
    # V5 -> W5), so a chapter row's cited theorem-anchor implication and
    # its registry counterpart share the same digit numbering. The
    # verifier's _revert_w_to_v_in_theorem_citation undoes this swap at
    # every registry-lookup site, restoring exact-string equality with
    # the v/V form stored in state.global_theorems / state.external_theorems.
    # -------------------------------------------------------------------------
    for fname, stack in ram_stacks.items():
        for r_idx, row in enumerate(stack):
            for c_idx, cell in enumerate(row):
                if c_idx < 3:                       # 0=head, 1=ns, 2=tag — skip
                    continue
                if cell == "main":                  # namespace cell — skip
                    continue
                if not is_theorem_anchor_implication(cell):
                    continue
                ram_stacks[fname][r_idx][c_idx] = _v_to_w_in_theorem_citation(cell)

    # -------------------------------------------------------------------------
    # Fix tags for external theorems
    # -------------------------------------------------------------------------
    # Reuse the raw_external_set loaded earlier (just before iteration 4)
    # so we don't re-read compressed_external_theorems.txt twice.
    external_set = set(raw_external_set)

    # Collect renamed versions of external theorems.
    # Three sources:
    #   (a) v/V rename forms produced by external_renames (precomputed,
    #       used by iteration 4 for the chapter-cell substitution).
    #   (b) w/W forms produced by iteration 5's swap from those v/V cells.
    #   (c) any other renames that landed via the fallback path.
    renamed_external = set(external_renames.values())
    # Also collect w/W forms by re-applying the iteration-5 swap to each
    # external_renames value — these are what end up in chapter cells.
    renamed_external.update(_v_to_w_in_theorem_citation(v) for v in external_renames.values())
    for i, thm_row in enumerate(renamed_theorems):
        raw_expr = raw_theorems[i][0] if i < len(raw_theorems) else ""
        if raw_expr in external_set:
            renamed_external.add(thm_row[0])

    # Also collect renamed forms of external theorems from chapter content
    # (after iteration 4, these may have been renamed via fallback chapter maps)
    for fname, stack in ram_stacks.items():
        raw_stack = raw_stacks[fname]
        for r_idx, row in enumerate(stack):
            for c_idx, cell in enumerate(row):
                if c_idx == 2 or cell == "main":
                    continue
                orig_cell = raw_stack[r_idx][c_idx]
                if orig_cell in external_set and cell != orig_cell:
                    renamed_external.add(cell)

    all_external = external_set | renamed_external

    if all_external:
        # Fix tags: any line referencing an external theorem as "theorem" or "broadcast"
        for fname, stack in ram_stacks.items():
            for r_idx, row in enumerate(stack):
                if len(row) >= 3:
                    expr = row[0]
                    tag = row[2]
                    if expr in all_external and tag in ("theorem", "broadcast"):
                        ram_stacks[fname][r_idx][2] = "externally provided theorem"

    # Write renamed external theorems for verifier consumption
    processed_ext_path = proc_dir / "external_theorems.txt"
    with open(processed_ext_path, "w", encoding="utf-8") as f:
        for expr in all_external:
            f.write(expr + "\n")

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