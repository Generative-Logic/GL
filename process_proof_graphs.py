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
from functools import cache

from configuration_reader import configuration_reader
import expression_utils
from expression_utils import replace_keys_in_string
from proof_convergence import process_to_fixed_point
from proof_export.corpora import INCUBATOR_ANCHORS
from proof_export.mpl import Atom, Implication, parse_mpl
from proof_hierarchy import order_main_hierarchies

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

    Recursion-variable preservation: `rec` (and `rec<N>` for nested
    inductions) is the prover's free recursion-step variable and must
    not be renamed — it is the conceptual placeholder for "the
    predecessor witness" in induction-step rows (e.g. `(in2[rec, v1, s])`).
    Renaming it to a generic `v<N>` loses the visual cue that the row
    is the induction-step hypothesis. Counters are not bumped for
    preserved names.
    """
    if arg == "rec" or re.fullmatch(r"rec\d+", arg):
        return arg
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


def theorem_rename_map(raw_expr: str, config) -> dict[str, str]:
    """
    @brief Map every token of one closed theorem expression to its
    theorem-local display name.
    @details
    Anchor slots take the anchor's configured names
    (`get_anchor_mapping_from_expr`); every other token, in order of first
    appearance, takes a typed name from `_mint_typed_var` (`v<N>` for a
    number, `V<N>` for a set or relation, counted separately). The map
    depends on the expression alone. It names the global theorem list, and a
    chapter cell holding a theorem implication that is not in that list
    verbatim — for example a compact implication's canonical reconstruction of
    a proved theorem in a different variable numbering. Such a theorem binds
    all its variables, so the chapter's own map, whose names belong to the
    chapter's variables, never applies to it.
    @param raw_expr Raw closed theorem expression (anchor implication).
    @param config Configuration reader with anchors and definition sets.
    @return Token-to-name replacement map for `replace_keys_in_string`.
    """

    repl_map = get_anchor_mapping_from_expr(raw_expr, config)
    arg_types = classify_args_by_defset(raw_expr, config)
    counters = {"digit": 1, "set": 1}
    for a in get_all_args(raw_expr):
        if a not in repl_map:
            repl_map[a] = _mint_typed_var(a, arg_types, counters)
    return repl_map


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
        if method in ("induction", "strong induction"):
            fnames = [
                f"{idx}_induction_typing.txt",
                f"{idx + 1}_check_zero.txt",
                f"{idx + 2}_check_induction_condition.txt",
            ]
            idx += 3
        elif method == "single chapter proof":
            fnames = [f"{idx}_single_chapter_proof.txt"]
            idx += 1
        elif method == "debug":
            fnames = [f"{idx}_debug.txt"]
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
        elif method == "or elimination":
            fnames = [f"{idx}_or_elimination.txt"]
            idx += 1
        elif method == "proved not broadcast":
            fnames = [f"{idx}_proved_not_broadcast.txt"]
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
    proved_file = theorems_dir / "compiled_theorems.txt"
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

    # Precompute var-field deps (reformulated → source theorem)
    var_deps: dict[str, set[str]] = {}
    for parts in raw_theorems:
        expr = parts[0]
        method = parts[1].lower() if len(parts) > 1 else ""
        var_field = parts[2] if len(parts) > 2 else ""
        if method == "reformulated statement" and var_field in all_thm_exprs:
            var_deps.setdefault(expr, set()).add(var_field)
        # An or-elimination row depends on its variant citation by contract
        # (its other citations sit in the chapter cells and are collected by
        # get_deps' exact-match scan like any raw theorem expression).
        if method == "or elimination" and var_field in all_thm_exprs:
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

    # ---- Remove external theorems (imported, not proved here) ----
    ext_file = theorems_dir / "compressed_external_theorems.txt"
    external_theorems = set()
    if ext_file.exists():
        with open(ext_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    external_theorems.add(line)

    if external_theorems:
        to_remove = external_theorems & all_thm_exprs
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


SCOPE_BOUNDARY = "_boundary_"
INDUCTION_SCOPE_PREFIX = "is_"
INT_SCOPE_PREFIX = "int_scope_"
# Payload prefixes of the prover's search fences the processed graph deletes:
# induction scopes (D-519) and int scopes
# (D-578).
STRIPPED_SCOPE_PREFIXES = (INDUCTION_SCOPE_PREFIX, INT_SCOPE_PREFIX)
# Raw-only tags that hand a statement from such a scope to an enclosing one.
SCOPE_EXIT_TAGS = ("induction scope", "int scope")
IDENTITY_TAGS = ("zoom in",) + SCOPE_EXIT_TAGS


def strip_induction_scope_segments(scope: str) -> str:
    """
    @brief Delete every induction-scope and int-scope segment from one
    validity name.

    @details An induction scope `<parent>_boundary_is_<chain>_<names>` is the
    prover's search fence around one admission of an induction hypothesis; it
    carries no assumption (D-519). An int scope
    `<parent>_boundary_int_scope_<premise>_<witness>` is the fence around one
    premise-bound integration (D-578) and carries no
    assumption either. The processed graph shows the rows of both at the
    nearest enclosing scope. The name is split at `_boundary_` (no payload
    contains it, I-139); every segment whose payload starts with `is_` or
    `int_scope_` is removed and every other segment keeps its place, so an OR
    branch or subproof opened inside such a scope moves up with it.

    @param scope A validity name `main(_boundary_<payload>)*`.
    @return The name without induction-scope and int-scope segments.
    @invariant Pure; a name without such segments is returned unchanged.
    @see strip_induction_scopes
    """
    parts = scope.split(SCOPE_BOUNDARY)
    kept = [parts[0]] + [part for part in parts[1:]
                         if not part.startswith(STRIPPED_SCOPE_PREFIXES)]
    return SCOPE_BOUNDARY.join(kept)


def _chapter_scope_cells(row):
    """
    @brief List the positions of a chapter row's validity names.

    @details A row is `expression, validity, tag, (dependency, validity)*`:
    its own validity sits at position 1 and each dependency's at an even
    position from 4 on.

    @param row One chapter row (a list of cells).
    @return The positions holding validity names.
    """
    positions = [1] if len(row) >= 2 else []
    positions.extend(range(4, len(row), 2))
    return positions


def _reachable_rows(rows, keys):
    """
    @brief Find the rows a chapter's goal row reaches through its citations.

    @details Starts at the first non-empty row and follows each dependency
    `(expression, validity)` cell to the row with that key, depth first.

    @param rows Chapter rows; empty rows are skipped.
    @param keys The key of each row, `None` for an empty or removed row.
    @return The set of reached row positions.
    """
    by_key = {}
    for index, key in enumerate(keys):
        if key is not None and key not in by_key:
            by_key[key] = index
    start = next((i for i, key in enumerate(keys) if key is not None), None)
    reached = set()
    if start is None:
        return reached
    pending = [start]
    while pending:
        index = pending.pop()
        if index in reached:
            continue
        reached.add(index)
        row = rows[index]
        for j in range(3, len(row) - 1, 2):
            target = by_key.get((row[j], row[j + 1]))
            if target is not None and target not in reached:
                pending.append(target)
    return reached


def strip_induction_scopes(stack):
    """
    @brief Remove the induction scopes and int scopes from one raw chapter.

    @details The raw graph keeps the prover's induction scopes (D-519) and
    int scopes (D-578); the
    processed graph deletes them. Every validity name loses its
    `_boundary_is_...` and `_boundary_int_scope_...` segments. A row that
    thereby cites itself — a `zoom in` of a source into such a scope, the
    `induction scope` row depositing a goal reached in a leaf at the scope's
    root, or the `int scope` row promoting a statement of a finished int scope
    to its parent — is an identity: its place is taken by the row it cites
    (following identities to the first real derivation), which leaves its own
    place, so the goal row stays first. Of rows with one key after the merge
    the first stays, and rows that only the dropped rows reached are dropped
    with them. A chapter without such scopes is returned unchanged.

    @param stack Raw chapter rows, each a list of cells; empty rows allowed.
    @return The chapter rows without induction scopes and int scopes.
    @invariant No `_boundary_is_` or `_boundary_int_scope_` segment and no
               `induction scope` or `int scope` row remains; the citations form
               no cycle; each key names one row.
    @see strip_induction_scope_segments
    """
    markers = tuple(SCOPE_BOUNDARY + prefix for prefix in STRIPPED_SCOPE_PREFIXES)
    if not any(marker in row[c] for row in stack for c in _chapter_scope_cells(row) for marker in markers) \
            and not any(len(row) > 2 and row[2] in SCOPE_EXIT_TAGS for row in stack):
        return stack
    raw_keys = [(row[0], row[1]) if len(row) >= 2 else None for row in stack]
    reachable_before = _reachable_rows(stack, raw_keys)
    rows = [list(row) for row in stack]
    for row in rows:
        for c in _chapter_scope_cells(row):
            row[c] = strip_induction_scope_segments(row[c])
    raw_index = {}
    for index, key in enumerate(raw_keys):
        if key is not None:
            raw_index.setdefault(key, index)

    def is_identity(index):
        row = rows[index]
        return (len(row) == 5 and row[2] in IDENTITY_TAGS
                and (row[3], row[4]) == (row[0], row[1]))

    for index, row in enumerate(rows):
        if len(row) > 2 and row[2] in SCOPE_EXIT_TAGS:
            assert is_identity(index), "an induction scope or int scope row cites its statement at the leaf"
    keys = [(row[0], row[1]) if len(row) >= 2 else None for row in rows]
    removed = set()
    for index in range(len(rows)):
        if index in removed or keys[index] is None or not is_identity(index):
            continue
        source = index
        visited = {index}
        while is_identity(source):
            raw_dependency = (stack[source][3], stack[source][4])
            assert raw_dependency in raw_index, "an identity row's source is in its chapter"
            source = raw_index[raw_dependency]
            assert source not in visited, "identity rows form no cycle"
            visited.add(source)
        for dropped in visited:
            if dropped != index:
                removed.add(dropped)
        rows[index] = list(rows[source])
        keys[index] = keys[source]
    for index in removed:
        keys[index] = None
    seen = set()
    for index, key in enumerate(keys):
        if key is None:
            continue
        if key in seen:
            keys[index] = None
            continue
        seen.add(key)
    reachable_after = _reachable_rows(rows, keys)
    for index in reachable_before:
        if keys[index] is not None and index not in reachable_after:
            keys[index] = None
    result = [rows[index] for index in range(len(rows))
              if keys[index] is not None or not stack[index]]
    for row in result:
        for c in _chapter_scope_cells(row):
            for marker in markers:
                assert marker not in row[c], "no induction scope or int scope survives processing"
        assert not (len(row) > 2 and row[2] in SCOPE_EXIT_TAGS), \
            "no induction scope or int scope row survives processing"
    assert len({(row[0], row[1]) for row in result if len(row) >= 2}) \
        == sum(1 for row in result if len(row) >= 2), \
        "each key names one row after induction-scope and int-scope removal"
    assert _has_citation_cycle(stack) or not _has_citation_cycle(result), \
        "induction-scope and int-scope removal creates no citation cycle"
    return result


def _has_citation_cycle(rows):
    """
    @brief Whether a chapter's citations form a cycle.

    @details Each row cites the rows keyed by its dependency cells; a cycle
    is a row reaching itself. Depth first with an explicit frame stack, so
    long chapters need no recursion.

    @param rows Chapter rows; empty rows are skipped, the first row of a key
                stands for it.
    @return True iff some row reaches itself through citations.
    """
    by_key = {}
    for i, row in enumerate(rows):
        if len(row) >= 2:
            by_key.setdefault((row[0], row[1]), i)
    state = {}
    for root in by_key.values():
        if root in state:
            continue
        state[root] = 1
        frames = [(root, 3)]
        while frames:
            i, j = frames[-1]
            row = rows[i]
            if j >= len(row) - 1:
                state[i] = 2
                frames.pop()
                continue
            frames[-1] = (i, j + 2)
            target = by_key.get((row[j], row[j + 1]))
            if target is None or state.get(target) == 2:
                continue
            if state.get(target) == 1:
                return True
            state[target] = 1
            frames.append((target, 3))
    return False


def _prepare_processed_graph(config: configuration_reader, raw_dir, proc_dir, theorems_dir):
    """
    @brief Load, prune and rename a raw proof graph for processing.

    @details Empties the processed folder, loads the raw theorem list and
    chapters, prunes theorems nothing needs, normalises anchored theorem
    implications, deletes the prover's induction scopes and int scopes, applies the variable
    renaming iterations and the external-theorem tags, and writes the processed
    `external_theorems.txt`. Arranging the book and writing its chapters and
    theorem list is left to the caller.

    @param config Visualizer configuration of the graph.
    @param raw_dir Raw proof-graph folder, or None for the main graph.
    @param proc_dir Processed folder, or None for the main graph.
    @param theorems_dir Theorem folder, or None for the main graph.
    @return The renamed theorem rows, the renamed chapters by file name, the
            external theorem set, and the processed folder.
    @see create_processed_proof_graph, create_processed_theorem_list
    """
    expression_utils.set_configuration(config)
    # These results depend only on the expression and this fixed configuration.
    # Caches die with the call; chapter-specific replacement maps remain local.
    arguments_for = cache(get_all_args)
    types_for = cache(lambda cell: classify_args_by_defset(cell, config))
    is_theorem = cache(is_theorem_anchor_implication)
    rename_local = cache(lambda raw, current: w_rename_impl_local(raw, current, config))
    citation_form = cache(_v_to_w_in_theorem_citation)

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

    # -------------------------------------------------------------------------
    # STEP 1b: Delete the prover's induction scopes (D-519) and int scopes
    # (D-578) from every validity name before any
    # renaming reads a scope cell.
    # -------------------------------------------------------------------------
    raw_stacks = {name: strip_induction_scopes(stack)
                  for name, stack in raw_stacks.items()}

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
    # The exe's load-time compiled twins of the externals (written by
    # run_modes.cpp::fullRun). Base-form externals compile against the
    # run's live registry at load; chapter citations carry those compiled
    # forms, so the external registry must hold both shapes for the
    # verifier's textual origin match.
    precompiled_ext_file = (theorems_dir / "precompiled_external_theorems.txt"
                            if theorems_dir else None)
    if precompiled_ext_file and precompiled_ext_file.exists():
        with open(precompiled_ext_file, "r", encoding="utf-8") as f:
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
        if method in ("induction", "strong induction"):
            fname_to_raw_thm[f"{file_idx}_induction_typing.txt"] = thm_expr
            fname_to_raw_thm[f"{file_idx + 1}_check_zero.txt"] = thm_expr
            fname_to_raw_thm[f"{file_idx + 2}_check_induction_condition.txt"] = thm_expr
            file_idx += 3
        elif method == "single chapter proof":
            fname_to_raw_thm[f"{file_idx}_single_chapter_proof.txt"] = thm_expr
            file_idx += 1
        elif method == "debug":
            fname_to_raw_thm[f"{file_idx}_debug.txt"] = thm_expr
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
        elif method == "or elimination":
            fname_to_raw_thm[f"{file_idx}_or_elimination.txt"] = thm_expr
            file_idx += 1
        elif method == "proved not broadcast":
            fname_to_raw_thm[f"{file_idx}_proved_not_broadcast.txt"] = thm_expr
            file_idx += 1
        else:
            safe = re.sub(r"[^A-Za-z0-9._\-+]+", "_", method)[:64] or "unknown"
            fname_to_raw_thm[f"{file_idx}_unknown_{safe}.txt"] = thm_expr
            file_idx += 1

    ram_stacks = {name: [row.copy() for row in stack]
                  for name, stack in raw_stacks.items()}

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
                is_theorem_impl = is_theorem(orig_cell)

                # Rename implication-local variables for every non-theorem implication.
                # Theorem implications are handled later via the global theorem list.
                if is_impl and not is_theorem_impl:
                    args = arguments_for(cell)
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
            chapter_arg_type.update(types_for(raw_thm_expr))
        for r_idx_pre, row_pre in enumerate(stack):
            for c_idx_pre, cell_pre in enumerate(row_pre):
                if c_idx_pre == 2 or cell_pre == "main":
                    continue
                cell_types = types_for(cell_pre)
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
            for a in arguments_for(raw_thm_expr):
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
                is_theorem_impl = is_theorem(orig_cell)

                # Ignore theorem implications here. They are replaced wholesale later
                # so they stay aligned with the globally renamed theorem list.
                if is_impl and is_theorem_impl:
                    continue

                # ALL other expressions (including Iteration 1's modified unanchored implications) are processed!
                args = arguments_for(cell)
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
    renamed_theorems = [row.copy() for row in raw_theorems]
    for i, thm_row in enumerate(renamed_theorems):
        raw_thm_expr = thm_row[0]

        thm_repl_map = theorem_rename_map(raw_thm_expr, config)
        renamed_theorems[i][0] = replace_keys_in_string(raw_thm_expr, thm_repl_map)

        # Also rename the induction variable (field 2) for induction theorems
        method = thm_row[1].lower() if len(thm_row) > 1 else ""
        if method in ("induction", "strong induction") and len(thm_row) > 2:
            raw_ind_var = thm_row[2]
            renamed_theorems[i][2] = thm_repl_map.get(raw_ind_var, raw_ind_var)

    # -------------------------------------------------------------------------
    # ITERATION 3: Apply the replacement maps
    # -------------------------------------------------------------------------
    for fname, stack in ram_stacks.items():
        repl_map = stack_repl_maps[fname]
        raw_stack = raw_stacks[fname]
        replace_cell = cache(lambda cell: replace_keys_in_string(cell, repl_map))

        for r_idx, row in enumerate(stack):
            for c_idx, cell in enumerate(row):
                if c_idx == 2 or cell == "main":
                    continue

                orig_cell = raw_stack[r_idx][c_idx]
                is_impl = orig_cell.startswith("(>[")
                is_theorem_impl = is_theorem(orig_cell)

                # Do not apply the local chapter map to theorem implications.
                if is_impl and is_theorem_impl:
                    continue

                # Apply map to the current cell (Iter 1 mods are safely mapped here)
                ram_stacks[fname][r_idx][c_idx] = replace_cell(cell)

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
                is_theorem_impl = is_theorem(orig_cell)
                if is_impl and is_theorem_impl:
                    continue
                ram_stacks[fname][r_idx][c_idx] = rename_local(
                    orig_cell, ram_stacks[fname][r_idx][c_idx])

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
                is_theorem_impl = is_theorem(orig_cell)

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
                        # A theorem implication outside the theorem list (an
                        # or theorem's source implication, a compact's
                        # reconstruction in another numbering) binds all its
                        # variables: it takes its own theorem-local names.
                        ram_stacks[fname][r_idx][c_idx] = replace_keys_in_string(
                            cell, theorem_rename_map(cell, config))

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
            # A zoom in dependency names the exact parent row, so it keeps
            # the same theorem spelling as the result rather than a citation alias.
            if len(row) >= 3 and row[2] == "zoom in":
                continue
            for c_idx, cell in enumerate(row):
                if c_idx < 3:                       # 0=head, 1=ns, 2=tag — skip
                    continue
                if cell == "main":                  # namespace cell — skip
                    continue
                if not is_theorem(cell):
                    continue
                ram_stacks[fname][r_idx][c_idx] = citation_form(cell)

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

    # Write renamed external theorems for verifier consumption.
    # sorted() because all_external is a set; iterating it directly would emit
    # rows in per-process hash order, making the file differ run to run.
    processed_ext_path = proc_dir / "external_theorems.txt"
    with open(processed_ext_path, "w", encoding="utf-8") as f:
        for expr in sorted(all_external):
            f.write(expr + "\n")

    return renamed_theorems, ram_stacks, all_external, proc_dir


def _write_theorem_list(proc_dir, theorems):
    """
    @brief Write a processed graph's `global_theorem_list.txt`.

    @details One tab-separated line per theorem row (statement, method,
    induction variable), in the given order.

    @param proc_dir Processed proof-graph folder.
    @param theorems Theorem rows in book order.
    @return None.
    """
    with open(proc_dir / "global_theorem_list.txt", "w", encoding="utf-8") as f:
        for thm in theorems:
            f.write("\t".join(thm) + "\n")


def create_processed_proof_graph(config: configuration_reader,
                                 raw_dir=None, proc_dir=None,
                                 theorems_dir=None):
    """
    @brief Process a raw proof graph into its arranged proof book.

    @details Prepares the graph (`_prepare_processed_graph`), arranges the book
    (`process_to_fixed_point`, then the reader hierarchies of
    `order_main_hierarchies`), and writes every chapter and the theorem list.

    @param config Visualizer configuration of the graph.
    @param raw_dir Raw proof-graph folder, or None for the main graph.
    @param proc_dir Processed folder, or None for the main graph.
    @param theorems_dir Theorem folder, or None for the main graph.
    @return None.
    @see create_processed_theorem_list
    """
    renamed_theorems, ram_stacks, all_external, proc_dir = _prepare_processed_graph(
        config, raw_dir, proc_dir, theorems_dir)

    renamed_theorems, ram_stacks = process_to_fixed_point(
        renamed_theorems, ram_stacks, _build_fname_list, all_external,
        PROJECT_ROOT / "files" / "GL_binaries")

    ram_stacks = order_main_hierarchies(
        renamed_theorems, ram_stacks, _build_fname_list(renamed_theorems), all_external)

    for fname, stack in ram_stacks.items():
        with open(proc_dir / fname, "w", encoding="utf-8") as f:
            for row in stack:
                f.write("\t".join(row) + "\n")

    _write_theorem_list(proc_dir, renamed_theorems)

    print(f"Success: Processed proof graphs have been written to {proc_dir}")


def create_processed_theorem_list(config: configuration_reader, raw_dir, proc_dir, theorems_dir):
    """
    @brief Write only the renamed theorem list of the incubator book.

    @details Between the incubator and the main batches the run needs only the
    incubator's processed theorem list: the simple-fact conversion reads
    nothing else. The incubator arrangement (`process_incubator_book`) permutes
    theorems and rows but changes no theorem row, so the prepared list holds the
    lines of the arranged list in the prover's order. The book itself is
    arranged once, after the last batch (`create_processed_proof_graph`).
    No chapter is written.

    @param config Visualizer configuration of the graph.
    @param raw_dir Incubator raw proof-graph folder.
    @param proc_dir Incubator processed folder.
    @param theorems_dir Incubator theorem folder.
    @return None.
    @invariant Every theorem is anchored on an incubator anchor (asserted).
    @see create_processed_proof_graph
    """
    renamed_theorems, _, _, proc_dir = _prepare_processed_graph(config, raw_dir, proc_dir, theorems_dir)
    for theorem in renamed_theorems:
        expression = parse_mpl(theorem[0])
        assert isinstance(expression, Implication) and isinstance(expression.premise, Atom)
        assert expression.premise.head in INCUBATOR_ANCHORS, \
            f"the theorem list alone serves the incubator book only: {theorem[0]}"
    _write_theorem_list(proc_dir, renamed_theorems)
    print(f"Success: Incubator theorem list has been written to {proc_dir}")
