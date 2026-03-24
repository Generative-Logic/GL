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

import os
import re
import sys
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProofLine:
    """One tab-separated line from a chapter file."""
    expression: str
    namespace: str
    tag: str
    rest: List[str]
    raw: str
    line_no: int


@dataclass
class TagCounter:
    success: int = 0
    failure: int = 0

    def record(self, passed: bool):
        if passed:
            self.success += 1
        else:
            self.failure += 1


@dataclass
class VerifierState:
    tag_counters: Dict[str, TagCounter] = field(default_factory=dict)
    goal_reached: TagCounter = field(default_factory=TagCounter)

    # Global theorem list: expression → { "type": ..., "ref": ... }
    global_theorems: Dict[str, dict] = field(default_factory=dict)
    # Ordered list of (expression, type, ref) for chapter mapping
    global_theorem_list: List[Tuple[str, str, str]] = field(default_factory=list)

    # Output indices: core_name → 0-based index of output argument
    output_indices: Dict[str, int] = field(default_factory=dict)

    # Input indices: core_name → list of 0-based indices of input arguments
    input_indices: Dict[str, List[int]] = field(default_factory=dict)

    # GL binaries: tag → { name → { "category", "elements", "signature", "definedSet", ... } }
    gl_binaries: Dict[str, dict] = field(default_factory=dict)

    # Definition sets: core_name → { 1-based-position-str → [defset_str, bool] }
    definition_sets: Dict[str, dict] = field(default_factory=dict)

    # Transient: set per-chapter before dispatching line checkers
    current_chapter_thm: Optional[Tuple[str, str, str]] = None
    current_chapter_type: str = ""

    # Definition sets: core_name → { "1-based-position" → [def_set_str, bool] }
    definition_sets: Dict[str, Dict[str, list]] = field(default_factory=dict)

    # External theorems (raw + renamed forms)
    external_theorems: set = field(default_factory=set)

    def counter_for(self, tag: str) -> TagCounter:
        if tag not in self.tag_counters:
            self.tag_counters[tag] = TagCounter()
        return self.tag_counters[tag]


# ---------------------------------------------------------------------------
#  Parsing
# ---------------------------------------------------------------------------

def parse_chapter_file(filepath: str) -> List[ProofLine]:
    lines: List[ProofLine] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, raw_line in enumerate(f, start=1):
            raw_line = raw_line.rstrip("\r\n")
            if not raw_line.strip():
                continue
            parts = raw_line.split("\t")
            lines.append(ProofLine(
                expression=parts[0] if parts else "",
                namespace=parts[1] if len(parts) > 1 else "",
                tag=parts[2] if len(parts) > 2 else "<malformed>",
                rest=parts[3:] if len(parts) > 3 else [],
                raw=raw_line,
                line_no=i,
            ))
    return lines


def load_global_theorem_list(base_dir: str) -> Tuple[Dict[str, dict],
                                                      List[Tuple[str, str, str]]]:
    path = os.path.join(base_dir, "global_theorem_list.txt")
    theorems: Dict[str, dict] = {}
    ordered: List[Tuple[str, str, str]] = []
    if not os.path.isfile(path):
        return theorems, ordered
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            expr = parts[0]
            thm_type = parts[1] if len(parts) > 1 else ""
            ref = parts[2] if len(parts) > 2 else ""
            theorems[expr] = {"type": thm_type, "ref": ref}
            ordered.append((expr, thm_type, ref))
    return theorems, ordered


# ---------------------------------------------------------------------------
#  Build chapter → theorem mapping
#
#  Walk global_theorem_list in order, consume chapter files sequentially.
#  "induction" theorems consume 2 chapters (check_zero + check_induction_condition).
#  All other types consume 1 chapter.
# ---------------------------------------------------------------------------

def build_chapter_theorem_map(
    chapter_files: List[str],
    theorem_list: List[Tuple[str, str, str]],
) -> Dict[str, Tuple[str, str, str]]:
    """Returns chapter_filename → (theorem_expr, theorem_type, theorem_ref)."""
    mapping: Dict[str, Tuple[str, str, str]] = {}
    ci = 0
    for thm_expr, thm_type, thm_ref in theorem_list:
        if ci >= len(chapter_files):
            break
        if thm_type == "induction":
            mapping[chapter_files[ci]] = (thm_expr, thm_type, thm_ref)
            if ci + 1 < len(chapter_files):
                mapping[chapter_files[ci + 1]] = (thm_expr, thm_type, thm_ref)
            ci += 2
        else:
            mapping[chapter_files[ci]] = (thm_expr, thm_type, thm_ref)
            ci += 1
    return mapping


# ---------------------------------------------------------------------------
#  Theorem goal reached check
# ---------------------------------------------------------------------------

def _find_matching_paren(expr: str, start: int) -> int:
    """Return index of ')' matching '(' at position *start*."""
    depth = 0
    for i in range(start, len(expr)):
        if expr[i] == '(':
            depth += 1
        elif expr[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    return -1


def disintegrate_implication_head(expr: str) -> str:
    """
    Peel off all ``(>[vars](premise)(body))`` layers and return the
    innermost conclusion as-is. No renaming.
    """
    while expr.startswith('(>['):
        bracket_close = expr.index(']', 3)
        p_start = bracket_close + 1
        if p_start >= len(expr) or expr[p_start] != '(':
            break
        p_end = _find_matching_paren(expr, p_start)
        if p_end < 0:
            break
        b_start = p_end + 1
        if b_start >= len(expr):
            break
        if expr[b_start] == '!':
            # Negated body: !(...)
            inner_start = b_start + 1
            if inner_start >= len(expr) or expr[inner_start] != '(':
                break
            inner_end = _find_matching_paren(expr, inner_start)
            if inner_end < 0:
                break
            expr = expr[b_start:inner_end + 1]
            return expr
        if expr[b_start] != '(':
            break
        b_end = _find_matching_paren(expr, b_start)
        if b_end < 0:
            break
        expr = expr[b_start:b_end + 1]

    return expr


# ---------------------------------------------------------------------------
#  Mirroring helpers  (independent copy — no imports from create_expressions)
# ---------------------------------------------------------------------------

def _extract_args(expr: str) -> List[str]:
    """Extract arguments from an expression like ``(in3[a,b,c,+])`` → [a,b,c,+]."""
    m = re.search(r'\[([^\]]*)\]', expr)
    if not m:
        return []
    return [a for a in m.group(1).split(',') if a]


def _extract_core_name(expr: str) -> str:
    """Extract core name: ``(in3[a,b,c,+])`` → ``in3``."""
    m = re.match(r'\((\w+)\[', expr)
    return m.group(1) if m else ""


def _trace_back_to(expr: str,
                   expr_to_lines: Dict[str, List["ProofLine"]],
                   is_target,
                   should_follow=None,
                   visited: set = None) -> bool:
    """Generic recursive trace-back through proof chain.

    Finds *expr* on the LHS of chapter lines, then:
    1. If any such line satisfies *is_target(line)* → True
    2. Otherwise follows rest-field expressions that pass *should_follow(src)*
       (all if *should_follow* is None) and recurses.
    """
    if visited is None:
        visited = set()
    if expr in visited:
        return False
    visited.add(expr)
    for ch_line in expr_to_lines.get(expr, []):
        if is_target(ch_line):
            return True
        for i in range(0, len(ch_line.rest), 2):
            src = ch_line.rest[i]
            if should_follow is not None and not should_follow(src):
                continue
            if _trace_back_to(src, expr_to_lines, is_target, should_follow, visited):
                return True
    return False


def disintegrate_implication_full(expr: str) -> Tuple[List[str], str]:
    """
    Peel off all ``(>[vars](premise)(body))`` layers.
    Returns (premises_list, head) where premises_list is list of premise expressions.
    """
    premises: List[str] = []
    while expr.startswith('(>['):
        bracket_close = expr.index(']', 3)
        p_start = bracket_close + 1
        if p_start >= len(expr) or expr[p_start] != '(':
            break
        p_end = _find_matching_paren(expr, p_start)
        if p_end < 0:
            break
        premise = expr[p_start:p_end + 1]
        b_start = p_end + 1
        if b_start >= len(expr):
            break
        if expr[b_start] == '!':
            # Negated body: !(...)
            inner_start = b_start + 1
            if inner_start >= len(expr) or expr[inner_start] != '(':
                break
            inner_end = _find_matching_paren(expr, inner_start)
            if inner_end < 0:
                break
            premises.append(premise)
            expr = expr[b_start:inner_end + 1]
            return premises, expr
        if expr[b_start] != '(':
            break
        b_end = _find_matching_paren(expr, b_start)
        if b_end < 0:
            break
        premises.append(premise)
        expr = expr[b_start:b_end + 1]

    return premises, expr


def _replace_arg_safe(expr: str, old: str, new: str) -> str:
    """Replace an argument in expression, argument-level safe (bracket-delimited)."""
    pattern = r'(?<=[\[,])' + re.escape(old) + r'(?=[\],])'
    return re.sub(pattern, new, expr)


def _normalize_expr_list(exprs: List[str]) -> List[str]:
    """
    Rename v-variables across a list of expressions by order of first
    appearance in expression arguments (not bound vars).
    """
    seen: Dict[str, str] = {}
    counter = [0]

    def _repl(m: re.Match) -> str:
        v = m.group(0)
        if v not in seen:
            counter[0] += 1
            seen[v] = f'v{counter[0]}'
        return seen[v]

    result: List[str] = []
    for expr in exprs:
        result.append(re.sub(r'v\d+', _repl, expr))
    return result


def _check_mirror(source_expr: str, target_expr: str,
                   output_indices: Dict[str, int]) -> bool:
    """
    Verify that target_expr is a valid mirror of source_expr.

    1. Disintegrate both into premises + head.
    2. From source: find premise with same output var as head → swap.
    3. Try all permutations of non-anchor source premises.
    4. For each permutation: normalize expression lists (ignoring bound vars),
       compare to normalized target expression list.
    """
    src_premises, src_head = disintegrate_implication_full(source_expr)
    tgt_premises, tgt_head = disintegrate_implication_full(target_expr)

    if len(src_premises) < 2 or len(src_premises) != len(tgt_premises):
        return False

    # Normalize target: [anchor, premises..., head] by expression arg order
    tgt_list = tgt_premises + [tgt_head]
    tgt_norm = _normalize_expr_list(tgt_list)

    # Find output var of source head
    head_core = _extract_core_name(src_head)
    head_args = _extract_args(src_head)
    if head_core not in output_indices:
        return False
    out_idx = output_indices[head_core]
    if out_idx >= len(head_args):
        return False
    head_out_var = head_args[out_idx]

    # Find non-anchor source premise with same output var
    swap_idx = -1
    for i in range(1, len(src_premises)):
        prem_core = _extract_core_name(src_premises[i])
        prem_args = _extract_args(src_premises[i])
        if prem_core in output_indices:
            p_out_idx = output_indices[prem_core]
            if p_out_idx < len(prem_args) and prem_args[p_out_idx] == head_out_var:
                swap_idx = i
                break

    if swap_idx < 0:
        return False

    # After swap: new head = old premise, old head takes the premise's slot
    new_head = src_premises[swap_idx]
    src_non_anchor: List[str] = []
    for i in range(1, len(src_premises)):
        if i == swap_idx:
            src_non_anchor.append(src_head)
        else:
            src_non_anchor.append(src_premises[i])

    # Try all permutations of non-anchor premises
    for perm in itertools.permutations(src_non_anchor):
        candidate_list = [src_premises[0]] + list(perm) + [new_head]
        candidate_norm = _normalize_expr_list(candidate_list)
        if candidate_norm == tgt_norm:
            return True

    return False


def _check_reformulation(source_expr: str, target_expr: str,
                          gl_binaries: Dict[str, dict]) -> bool:
    """
    Verify that target_expr is a valid reformulation of source_expr.

    1. Disintegrate both into premises + head.
    2. Target head must be an existence expression (per GL_binary).
    3. Determine anchor tag → select correct GL_binary.
    4. Expand existence head into left + right using GL_binary elements.
    5. New bound var = first unused v-index in the target theorem.
    6. Check left expression's new-bound-var position == definedSet of that expression.
    7. Left expression joins target premises, right expression is new head.
    8. Permute non-anchor premises, normalize, compare to normalized source.
    """
    src_premises, src_head = disintegrate_implication_full(source_expr)
    tgt_premises, tgt_head = disintegrate_implication_full(target_expr)

    if not tgt_premises:
        return False

    # Determine anchor tag from first premise (e.g. AnchorGauss → Gauss)
    anchor_core = _extract_core_name(tgt_premises[0])
    if not anchor_core.startswith("Anchor"):
        return False
    tag = anchor_core[len("Anchor"):]  # e.g. "Gauss"

    # Select GL binary
    binary = gl_binaries.get(tag)
    if binary is None:
        return False

    # Target head must be an existence expression
    head_core = _extract_core_name(tgt_head)
    if head_core not in binary:
        return False
    head_spec = binary[head_core]
    if head_spec.get("category") != "existence":
        return False

    elements = head_spec.get("elements", [])
    signature = head_spec.get("signature", "")
    if len(elements) != 2 or not signature:
        return False

    # Build substitution map: u_i → actual arg
    sig_args = _extract_args(signature)
    head_args = _extract_args(tgt_head)
    if len(sig_args) != len(head_args):
        return False
    subst: Dict[str, str] = {}
    for s_arg, h_arg in zip(sig_args, head_args):
        subst[s_arg] = h_arg

    # Find unused v-index for the new bound variable
    all_v_indices = set()
    for m in re.finditer(r'v(\d+)', target_expr):
        all_v_indices.add(int(m.group(1)))
    new_v_idx = 1
    while new_v_idx in all_v_indices:
        new_v_idx += 1
    new_bound_var = f"v{new_v_idx}"

    # Expand elements: substitute args + replace placeholder "1" with new bound var
    def _apply_subst(element_expr: str) -> str:
        result = element_expr
        # Replace "1" placeholder with new bound var (in argument positions)
        # Use the same lookaround-based replacement as process_proof_graphs
        full_map = dict(subst)
        full_map["1"] = new_bound_var
        escaped_keys = [re.escape(k) for k in full_map]
        pattern = r'(?<=[\[,])(' + '|'.join(escaped_keys) + r')(?=[\],])'
        return re.compile(pattern).sub(lambda m: full_map.get(m.group(1), m.group(1)), result)

    left_expr = _apply_subst(elements[0])
    right_expr = _apply_subst(elements[1])

    # Check definedSet: left expression's new bound var must be at the definedSet position
    left_core = _extract_core_name(left_expr)
    if left_core in binary:
        defined_set_uvar = binary[left_core].get("definedSet", "")
        if defined_set_uvar:
            left_sig = binary[left_core].get("signature", "")
            left_sig_args = _extract_args(left_sig)
            if defined_set_uvar in left_sig_args:
                ds_idx = left_sig_args.index(defined_set_uvar)
                left_actual_args = _extract_args(left_expr)
                if ds_idx >= len(left_actual_args) or left_actual_args[ds_idx] != new_bound_var:
                    return False

    # Build expanded target: original premises + left_expr as extra premise, right_expr as head
    expanded_non_anchor = []
    for i in range(1, len(tgt_premises)):
        expanded_non_anchor.append(tgt_premises[i])
    expanded_non_anchor.append(left_expr)

    # Source non-anchor premises
    src_non_anchor = src_premises[1:]

    if len(expanded_non_anchor) != len(src_non_anchor):
        return False

    # Normalize source: [anchor, non_anchor..., head]
    src_list = src_premises + [src_head]
    src_norm = _normalize_expr_list(src_list)

    # Try all permutations of expanded non-anchor premises
    for perm in itertools.permutations(expanded_non_anchor):
        candidate_list = [tgt_premises[0]] + list(perm) + [right_expr]
        candidate_norm = _normalize_expr_list(candidate_list)
        if candidate_norm == src_norm:
            return True

    return False


def load_gl_binaries(binaries_dir: str) -> Dict[str, dict]:
    """
    Load GL binary JSON files. Returns tag → binary_dict.
    E.g. "Peano" → contents of GL_binary_Peano.json
    """
    import json
    result: Dict[str, dict] = {}
    if not os.path.isdir(binaries_dir):
        return result
    for fname in os.listdir(binaries_dir):
        if fname.startswith("GL_binary_") and fname.endswith(".json"):
            tag = fname[len("GL_binary_"):-len(".json")]
            with open(os.path.join(binaries_dir, fname), "r", encoding="utf-8") as f:
                result[tag] = json.load(f)
    return result


def load_output_indices(config_dir: str) -> Dict[str, int]:
    """
    Load output indices from ConfigVisu.json.
    Returns core_name → 0-based index of the output argument.
    """
    indices: Dict[str, int] = {}
    config_path = os.path.join(config_dir, "ConfigVisu.json")
    if not os.path.isfile(config_path):
        return indices

    import json
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    for name, spec in config.items():
        if not isinstance(spec, dict):
            continue
        out_args = spec.get("output_args", [])
        if not out_args:
            continue
        short_mpl = spec.get("short_mpl", "")
        m = re.search(r'\[([^\]]*)\]', short_mpl)
        if not m:
            continue
        ordered_args = [a.strip() for a in m.group(1).split(',') if a.strip()]
        for out_name in out_args:
            if out_name in ordered_args:
                indices[name] = ordered_args.index(out_name)
                break

    return indices


def load_definition_sets(config_dir: str) -> Dict[str, Dict[str, list]]:
    """
    Load definition_sets from ConfigVisu.json.
    Returns core_name → { "1-based-position-str" → [def_set_str, bool] }.
    """
    import json
    result: Dict[str, Dict[str, list]] = {}
    config_path = os.path.join(config_dir, "ConfigVisu.json")
    if not os.path.isfile(config_path):
        return result
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    for name, spec in config.items():
        if not isinstance(spec, dict):
            continue
        ds = spec.get("definition_sets", {})
        if ds:
            result[name] = ds
    return result


def load_input_indices(config_dir: str) -> Dict[str, List[int]]:
    """
    Load input indices from ConfigVisu.json.
    Returns core_name → list of 0-based indices of input arguments.
    """
    import json
    result: Dict[str, List[int]] = {}
    config_path = os.path.join(config_dir, "ConfigVisu.json")
    if not os.path.isfile(config_path):
        return result
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    for name, spec in config.items():
        if not isinstance(spec, dict):
            continue
        in_args = spec.get("input_args", [])
        if not in_args:
            continue
        short_mpl = spec.get("short_mpl", "")
        m = re.search(r'\[([^\]]*)\]', short_mpl)
        if not m:
            continue
        ordered = [a.strip() for a in m.group(1).split(',') if a.strip()]
        indices = []
        for in_name in in_args:
            if in_name in ordered:
                indices.append(ordered.index(in_name))
        if indices:
            result[name] = indices
    return result


def _find_digit_args(all_exprs: List[str],
                     anchor_expr: str,
                     input_indices: Dict[str, List[int]],
                     output_indices: Dict[str, int]) -> set:
    """
    Replicate C++ findDigitArgs: collect all input args from all expressions,
    remove anchor args, subtract output args.
    """
    all_input_args: set = set()
    for expr in all_exprs:
        core = _extract_core_name(expr)
        if core in input_indices:
            args = _extract_args(expr)
            for idx in input_indices[core]:
                if idx < len(args):
                    all_input_args.add(args[idx])

    # Remove anchor args
    for a in _extract_args(anchor_expr):
        all_input_args.discard(a)

    # Subtract output args
    all_output_args: set = set()
    for expr in all_exprs:
        core = _extract_core_name(expr)
        if core in output_indices:
            args = _extract_args(expr)
            out_idx = output_indices[core]
            if out_idx < len(args):
                all_output_args.add(args[out_idx])

    return all_input_args - all_output_args


def _find_immutable_args(chain_exprs: List[str],
                         digit_args: set,
                         ind_var: str,
                         input_indices: Dict[str, List[int]],
                         output_indices: Dict[str, int]) -> set:
    """
    Replicate C++ findImmutableArgs: start from digit args minus ind_var,
    propagate through outputs where all inputs are immutable.
    """
    immutables = set(digit_args)
    immutables.discard(ind_var)

    changed = True
    while changed:
        changed = False
        for expr in chain_exprs:
            core = _extract_core_name(expr)
            if core not in input_indices or core not in output_indices:
                continue
            args = _extract_args(expr)
            # Check all inputs are immutable
            all_immutable = True
            for idx in input_indices[core]:
                if idx < len(args) and args[idx] not in immutables:
                    all_immutable = False
                    break
            if all_immutable:
                out_idx = output_indices[core]
                if out_idx < len(args) and args[out_idx] not in immutables:
                    immutables.add(args[out_idx])
                    changed = True

    return immutables


# ---------------------------------------------------------------------------
#  Theorem goal reached dispatcher
# ---------------------------------------------------------------------------

def check_theorem_goal_reached(
    chapter_file: str,
    chapter_type: str,
    lines: List[ProofLine],
    chapter_thm: Optional[Tuple[str, str, str]],
    output_indices: Dict[str, int],
    gl_binaries: Dict[str, dict],
) -> bool:
    if not lines or chapter_thm is None:
        return False

    thm_expr, thm_type, _ = chapter_thm
    first = lines[0]

    # reformulated_statement: expand existence head, permute, compare
    if chapter_type == "reformulated_statement":
        if first.namespace != "main" or first.tag != "reformulated from":
            return False
        if len(first.rest) < 1:
            return False
        source_expr = first.rest[0]
        return _check_reformulation(source_expr, first.expression, gl_binaries)

    # mirrored_statement: mirror the source theorem and compare
    if chapter_type == "mirrored_statement":
        if first.namespace != "main" or first.tag != "mirrored from":
            return False
        if len(first.rest) < 1:
            return False
        source_expr = first.rest[0]
        return _check_mirror(source_expr, first.expression, output_indices)

    # back_reformulated_statement: single line with tag "incubator back reformulation"
    if chapter_type == "back_reformulated_statement":
        if first.namespace != "main" or first.tag != "incubator back reformulation":
            return False
        if len(first.rest) < 1:
            return False
        return True

    # direct proof, check_zero, check_induction_condition:
    # extract head from theorem, compare with first line
    if chapter_type in ("direct_proof", "check_zero", "check_induction_condition"):
        if first.namespace != "main":
            return False
        head = disintegrate_implication_head(thm_expr)
        return first.expression == head

    return False


# ---------------------------------------------------------------------------
#  Empty checker stubs — one per proof tag
#
#  Each returns True (success) or False (failure).
#  Currently every checker returns False (= not yet implemented).
# ---------------------------------------------------------------------------

def _normalize_implication(expr: str) -> str:
    """
    Normalize v-variables in an implication by order of first appearance
    in expression arguments (not bound var lists in >[...]).
    Two-pass: build map from expression args, then apply to everything.
    """
    seen: Dict[str, str] = {}
    counter = [0]
    i = 0
    while i < len(expr):
        if expr[i:i+2] == '>[':
            i = expr.index(']', i + 2) + 1
        elif expr[i] == '[' and i > 0 and expr[i-1] != '>':
            j = expr.index(']', i + 1)
            for m in re.finditer(r'v\d+', expr[i:j+1]):
                v = m.group(0)
                if v not in seen:
                    counter[0] += 1
                    seen[v] = f'v{counter[0]}'
            i = j + 1
        else:
            i += 1

    def _repl(m: re.Match) -> str:
        return seen.get(m.group(0), m.group(0))

    return re.sub(r'v\d+', _repl, expr)


def _reconstruct_implication(chain_exprs: List[str], head: str) -> str:
    """
    Reconstruct an implication from premises + head.
    Only v-variables are candidates for binding.
    Multi-occurrence v-vars become bound at their first appearance.
    """
    all_exprs = chain_exprs + [head]
    counter: Dict[str, int] = {}
    for e in all_exprs:
        for a in _extract_args(e):
            if not re.match(r'v\d+$', a):
                continue
            counter[a] = counter.get(a, 0) + 1
    multi = {a for a, c in counter.items() if c > 1}

    placed: set = set()
    when: List[List[str]] = [[] for _ in range(len(chain_exprs))]
    for idx, e in enumerate(all_exprs):
        for a in _extract_args(e):
            if a in multi and a not in placed:
                placed.add(a)
                if idx < len(chain_exprs):
                    when[idx].append(a)

    result = head
    for i in range(len(chain_exprs) - 1, -1, -1):
        bound_str = ','.join(when[i])
        result = f'(>[{bound_str}]{chain_exprs[i]}{result})'
    return result


def _collect_bound_vars(expr: str) -> set:
    """Collect all variables listed inside >[...] brackets."""
    result = set()
    i = 0
    while i < len(expr):
        if expr[i:i+2] == '>[':
            j = expr.index(']', i + 2)
            for a in expr[i+2:j].split(','):
                a = a.strip()
                if a:
                    result.add(a)
            i = j + 1
        else:
            i += 1
    return result


def _collect_all_expr_vars(expr: str) -> set:
    """Collect all variables from expression argument brackets (not >[...])."""
    result = set()
    i = 0
    while i < len(expr):
        if expr[i:i+2] == '>[':
            i = expr.index(']', i + 2) + 1
        elif expr[i] == '[' and i > 0 and expr[i-1] != '>':
            j = expr.index(']', i + 1)
            for a in expr[i+1:j].split(','):
                a = a.strip()
                if a:
                    result.add(a)
            i = j + 1
        else:
            i += 1
    return result


def _normalize_all_vars_in_list(exprs: List[str]) -> List[str]:
    """Rename ALL variables across a list of expressions by order of first
    appearance in expression arguments. Consistent across the whole list."""
    seen: Dict[str, str] = {}
    counter = [0]
    for expr in exprs:
        for m in re.finditer(r'(?<=[\[,])([^,\[\]]+)(?=[\],])', expr):
            a = m.group(1)
            if a not in seen:
                counter[0] += 1
                seen[a] = f'v{counter[0]}'
    if not seen:
        return list(exprs)
    escaped = [re.escape(k) for k in seen]
    pattern = re.compile(r'(?<=[\[,])(' + '|'.join(escaped) + r')(?=[\],])')
    return [pattern.sub(lambda m: seen[m.group(1)], e) for e in exprs]


def check_implication(line: ProofLine, chapter: List[ProofLine],
                      state: VerifierState) -> bool:
    """
    IMPLICATION checker.

    Namespace rules:
      - Implication (rest[0]) must have namespace "main" (rest[1]).
      - Among premise namespaces, at most one non-main namespace kind.
      - Result namespace must equal that non-main kind, or "main" if all
        premises are "main".

    Structural check — two cases:
      Disintegrate the reference implication. If the first premise is an
      Anchor expression (theorem-level): all variables are changeable.
      Disintegrate into flat premise+head lists, normalize ALL vars by
      first appearance, permutate actual premises, compare.
      Otherwise (disintegration-level): unchangeables = all_expr_vars -
      bound_vars. Normalize, rebuild, compare via >[...] structure.
    """
    if len(line.rest) < 2:
        return False

    impl = line.rest[0]
    impl_ns = line.rest[1]

    # Collect premise namespaces
    premise_nss = [line.rest[i] for i in range(3, len(line.rest), 2)]
    # All non-main namespaces across implication + premises
    all_nss = [impl_ns] + premise_nss
    non_main = set(ns for ns in all_nss if ns != "main")

    # At most one non-main namespace kind among implication + premises
    if len(non_main) > 1:
        return False

    # Result namespace: must be the non-main kind from premises/impl,
    # OR the result can be in a non-main validity context when all
    # sources are "main" (a main implication firing inside a non-main scope)
    if non_main:
        if line.namespace != list(non_main)[0]:
            return False

    # --- Structural check ---
    result_expr = line.expression
    premises = [line.rest[i] for i in range(2, len(line.rest), 2)]

    ref_premises, ref_head = disintegrate_implication_full(impl)

    if ref_premises and ref_premises[0].startswith("(Anchor"):
        # Anchor (theorem-level): all vars changeable.
        # Compare flat disintegrated lists with all vars normalized.
        ref_chain = ref_premises + [ref_head]
        norm_ref = _normalize_all_vars_in_list(ref_chain)

        for perm in itertools.permutations(premises):
            act_chain = list(perm) + [result_expr]
            norm_act = _normalize_all_vars_in_list(act_chain)
            if norm_act == norm_ref:
                return True
    else:
        # Non-anchor: substitution-based matching.
        # Unchangeables = vars not bound in >[...]. Changeables = bound vars.
        # Find a permutation of actual premises where core names align,
        # unchangeables match at every position, and changeables map
        # consistently (well-defined function).
        all_vars = _collect_all_expr_vars(impl)
        bound_vars = _collect_bound_vars(impl)
        unchangeables = all_vars - bound_vars

        ref_chain = ref_premises + [ref_head]

        for perm in itertools.permutations(premises):
            act_chain = list(perm) + [result_expr]
            if len(ref_chain) != len(act_chain):
                continue
            changeable_map: Dict[str, str] = {}
            ok = True
            for ref_e, act_e in zip(ref_chain, act_chain):
                if _extract_core_name(ref_e) != _extract_core_name(act_e):
                    ok = False
                    break
                ra = _extract_args(ref_e)
                aa = _extract_args(act_e)
                if len(ra) != len(aa):
                    ok = False
                    break
                for rv, av in zip(ra, aa):
                    if rv in unchangeables:
                        if rv != av:
                            ok = False
                            break
                    elif rv in bound_vars:
                        if rv in changeable_map:
                            if changeable_map[rv] != av:
                                ok = False
                                break
                        else:
                            changeable_map[rv] = av
                    else:
                        if rv != av:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                return True

    return False


def _normalize_with_unchangeables(expr: str, unchangeables: set) -> str:
    """
    Normalize variables by order of first appearance in expression arguments
    (skipping >[...] brackets). Variables in unchangeables are kept as-is.
    Two-pass: build map from expression args, then apply everywhere.
    """
    seen: Dict[str, str] = {}
    counter = [0]
    i = 0
    while i < len(expr):
        if expr[i:i+2] == '>[':
            i = expr.index(']', i + 2) + 1
        elif expr[i] == '[' and i > 0 and expr[i-1] != '>':
            j = expr.index(']', i + 1)
            for a in expr[i+1:j].split(','):
                a = a.strip()
                if a and a not in unchangeables and a not in seen:
                    counter[0] += 1
                    seen[a] = f'v{counter[0]}'
            i = j + 1
        else:
            i += 1

    if not seen:
        return expr
    escaped = [re.escape(k) for k in seen]
    pattern = r'(?<=[\[,])(' + '|'.join(escaped) + r')(?=[\],])'
    return re.compile(pattern).sub(
        lambda m: seen.get(m.group(1), m.group(1)), expr)


def _build_and_from_elements(elements: List[str]) -> str:
    """Build nested (&...) from elements."""
    result = elements[0]
    for e in elements[1:]:
        result = f'(&{result}{e})'
    return result


def _build_implication_from_elements(elements: List[str],
                                      unchangeables: set) -> str:
    """Build implication from elements: premises = all but last, head = last."""
    premises = elements[:-1]
    head = elements[-1]
    all_exprs = premises + [head]
    bindable: set = set()
    for e in all_exprs:
        for a in _extract_args(e):
            if a not in unchangeables:
                bindable.add(a)

    placed: set = set()
    when: List[List[str]] = [[] for _ in range(len(premises))]
    for idx, e in enumerate(all_exprs):
        for a in _extract_args(e):
            if a in bindable and a not in placed:
                placed.add(a)
                if idx < len(premises):
                    when[idx].append(a)

    result = head
    for i in range(len(premises) - 1, -1, -1):
        bound_str = ','.join(when[i])
        result = f'(>[{bound_str}]{premises[i]}{result})'
    return result


def _build_existence_from_elements(elements: List[str]) -> str:
    """Build existence: !(>[1](el1)!(el2))."""
    return f'!(>[1]{elements[0]}!{elements[1]})'


def _build_existence_empty_binding(elements: List[str]) -> str:
    """Build existence with empty binding: (>[](el1)(el2))."""
    return f'(>[]{elements[0]}{elements[1]})'


def _parse_existence_expansion(expr: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse an expanded existence expression of the form:
        !(>[vars](left_expr)!(right_expr))

    Returns (bound_vars_csv, left_expr, right_expr) on success, else None.
    """
    if not expr.startswith('!(') or not expr.endswith(')'):
        return None

    inner = expr[1:]
    if not inner.startswith('(>['):
        return None

    try:
        bracket_close = inner.index(']', 3)
    except ValueError:
        return None

    left_start = bracket_close + 1
    if left_start >= len(inner) or inner[left_start] != '(':
        return None

    left_end = _find_matching_paren(inner, left_start)
    if left_end < 0:
        return None

    neg_idx = left_end + 1
    if neg_idx >= len(inner) or inner[neg_idx] != '!':
        return None

    right_start = neg_idx + 1
    if right_start >= len(inner) or inner[right_start] != '(':
        return None

    right_end = _find_matching_paren(inner, right_start)
    if right_end < 0:
        return None

    # The implication itself must close immediately after the negated right side.
    if right_end + 1 != len(inner) - 1 or inner[-1] != ')':
        return None

    bound_vars = inner[3:bracket_close]
    left_expr = inner[left_start:left_end + 1]
    right_expr = inner[right_start:right_end + 1]
    return bound_vars, left_expr, right_expr


def _check_existence_disintegration(line: ProofLine,
                                    chapter: List[ProofLine],
                                    compound: str,
                                    compound_ns: str,
                                    compact: str,
                                    entry: dict) -> bool:
    """
    Validate disintegration from an existence expansion.

    Allowed structure:
      1. The expanded existence origin may emit one or two disintegration lines.
         More than two is invalid.
      2. Any emitted child expressions must match actual existence members.
      3. The expanded existence expression itself must match the GL binary
         instantiation modulo bound-variable renaming.
      4. The current disintegration line must match one expected child modulo
         bound-variable renaming.
    """
    siblings = [
        ch_line for ch_line in chapter
        if (ch_line.tag == 'disintegration'
            and ch_line.namespace == compound_ns
            and len(ch_line.rest) >= 2
            and ch_line.rest[0] == compound
            and ch_line.rest[1] == compound_ns)
    ]

    if len(siblings) < 1 or len(siblings) > 2:
        return False

    parsed_compound = _parse_existence_expansion(compound)
    if parsed_compound is None:
        return False

    _, actual_left, actual_right = parsed_compound

    sig_args = _extract_args(entry.get('signature', ''))
    actual_args = _extract_args(compact)
    elements = entry.get('elements', [])

    if len(sig_args) != len(actual_args) or len(elements) != 2:
        return False

    subst = dict(zip(sig_args, actual_args))
    expected_children = [_replace_arg_safe_multi(elem, subst)
                         for elem in elements]
    expected_compound = _build_existence_from_elements(expected_children)

    unchangeables = set(actual_args)

    compound_norm = _normalize_with_unchangeables(compound, unchangeables)
    expected_compound_norm = _normalize_with_unchangeables(
        expected_compound, unchangeables)
    if compound_norm != expected_compound_norm:
        return False

    actual_member_norms = {
        _normalize_with_unchangeables(actual_left, unchangeables),
        _normalize_with_unchangeables(actual_right, unchangeables),
    }
    expected_child_norms = {
        _normalize_with_unchangeables(expr, unchangeables)
        for expr in expected_children
    }
    if actual_member_norms != expected_child_norms:
        return False

    sibling_norms = {
        _normalize_with_unchangeables(ch_line.expression, unchangeables)
        for ch_line in siblings
    }
    if not sibling_norms.issubset(expected_child_norms):
        return False

    line_norm = _normalize_with_unchangeables(line.expression, unchangeables)
    return line_norm in expected_child_norms


def _try_expand(binary_entry: dict, right_expr: str,
                left_expr: str) -> bool:
    """
    Try to expand right_expr using a GL binary entry and match against left_expr.
    """
    sig_args = _extract_args(binary_entry['signature'])
    actual_args = _extract_args(right_expr)
    if len(sig_args) != len(actual_args):
        return False

    subst = dict(zip(sig_args, actual_args))
    elements = [_replace_arg_safe_multi(e, subst)
                for e in binary_entry['elements']]

    cat = binary_entry['category']
    unch = set(actual_args)

    if cat == 'and':
        built = _build_and_from_elements(elements)
    elif cat == 'implication':
        built = _build_implication_from_elements(elements, unch)
    elif cat == 'existence':
        built = _build_existence_from_elements(elements)
    else:
        return False

    if (_normalize_with_unchangeables(built, unch)
            == _normalize_with_unchangeables(left_expr, unch)):
        return True

    # For existence, also try the empty-binding form (>[](el1)(el2))
    if cat == 'existence':
        built_empty = _build_existence_empty_binding(elements)
        if (_normalize_with_unchangeables(built_empty, unch)
                == _normalize_with_unchangeables(left_expr, unch)):
            return True

    return False


def _replace_arg_safe_multi(expr: str, subst: Dict[str, str]) -> str:
    """Apply multiple argument-level replacements."""
    if not subst:
        return expr
    pattern = r'(?<=[\[,])(' + '|'.join(re.escape(k) for k in subst) + r')(?=[\],])'
    return re.compile(pattern).sub(
        lambda m: subst.get(m.group(1), m.group(1)), expr)


def check_expansion(line: ProofLine, chapter: List[ProofLine],
                    state: VerifierState) -> bool:
    """
    EXPANSION checker.

    1. Namespaces must match (left == right).
    2. Right expression must exist as a left-side expression in the chapter.
    3. Take expression from right side (rest[0]), find it in GL binary.
       Build compound expression from elements (substituting signature args).
       Normalize both sides keeping right-side args as unchangeables.
       Match.
    """
    if len(line.rest) < 2:
        return False

    right_expr = line.rest[0]
    right_ns = line.rest[1]

    # Namespaces must match
    if line.namespace != right_ns:
        return False

    # Right expression must exist as a left-side expression in chapter
    if not any(ch_line.expression == right_expr for ch_line in chapter):
        return False

    core = _extract_core_name(right_expr)

    for binary in state.gl_binaries.values():
        if core in binary:
            if _try_expand(binary[core], right_expr, line.expression):
                return True

    return False


def check_disintegration(line: ProofLine, chapter: List[ProofLine],
                         state: VerifierState) -> bool:
    """
    DISINTEGRATION checker.

    1. Namespaces must match.
    2. Find chapter line where compound (rest[0]) is left side with tag "expansion".
    3. Take right side of that expansion line (compact name).
    4. Look up in GL binary.
       - "and": instantiated target must be one of the instantiated elements.
       - "existence": there must be exactly two disintegration children for the
         same expanded origin, and the expanded existence must match the
         instantiated GL binary modulo bound-variable renaming.
    """
    if len(line.rest) < 2:
        return False

    compound = line.rest[0]
    compound_ns = line.rest[1]

    if line.namespace != compound_ns:
        return False

    # Find expansion line where compound is left side
    for ch_line in chapter:
        if (ch_line.expression == compound
                and ch_line.namespace == compound_ns
                and ch_line.tag == "expansion"
                and len(ch_line.rest) >= 2
                and ch_line.rest[1] == compound_ns):
            compact = ch_line.rest[0]
            core = _extract_core_name(compact)

            for binary in state.gl_binaries.values():
                if core not in binary:
                    continue
                entry = binary[core]
                category = entry.get('category')

                if category == 'and':
                    sig_args = _extract_args(entry.get('signature', ''))
                    actual_args = _extract_args(compact)
                    if len(sig_args) != len(actual_args):
                        continue

                    subst = dict(zip(sig_args, actual_args))
                    renamed_elems = [_replace_arg_safe_multi(e, subst)
                                     for e in entry.get('elements', [])]

                    if line.expression in renamed_elems:
                        return True

                elif category == 'existence':
                    if _check_existence_disintegration(
                            line, chapter, compound, compound_ns, compact, entry):
                        return True

    return False


def check_task_formulation(line: ProofLine, chapter: List[ProofLine],
                           state: VerifierState) -> bool:
    """
    TASK FORMULATION checker.

    The expression must be a premise of the chapter's theorem and
    namespace must be "main".

    For contradiction proofs (head starts with '!'), the un-negated head
    (cleanOp) is also valid — it is seeded into the contradiction LB as
    the hypothesis to be disproved.
    """
    if line.namespace != "main":
        return False
    if state.current_chapter_thm is None:
        return False
    thm_expr = state.current_chapter_thm[0]
    premises, head = disintegrate_implication_full(thm_expr)
    if line.expression in premises:
        return True
    # Contradiction seed: head is !(...), cleanOp is (...)
    if head.startswith('!') and line.expression == head[1:]:
        return True
    return False


def check_equality1(line: ProofLine, chapter: List[ProofLine],
                    state: VerifierState) -> bool:
    """
    EQUALITY1: argument substitution.

    Result and source have the same core and arity. Each differing
    argument position (a→b) must be justified by an equality (=[a,b])
    in rest. All namespaces must be equal.
    """
    # rest layout: source, ns, eq1, ns, [eq2, ns, ...]
    if len(line.rest) < 4:
        return False

    source_expr, source_ns = line.rest[0], line.rest[1]
    if line.namespace != source_ns:
        return False

    # Collect equalities
    eq_set: set = set()
    i = 2
    while i + 1 < len(line.rest):
        eq_expr, eq_ns = line.rest[i], line.rest[i + 1]
        if eq_ns != line.namespace:
            return False
        ea = _extract_args(eq_expr)
        if len(ea) == 2:
            eq_set.add((ea[0], ea[1]))
        i += 2

    # Same core and arity
    result_core = _extract_core_name(line.expression)
    source_core = _extract_core_name(source_expr)
    result_args = _extract_args(line.expression)
    source_args = _extract_args(source_expr)

    if result_core != source_core or len(result_args) != len(source_args):
        return False

    # Each differing position must be covered by an equality (source→result)
    for ra, sa in zip(result_args, source_args):
        if ra != sa:
            if (sa, ra) not in eq_set:
                return False

    return True


def check_equality2(line: ProofLine, chapter: List[ProofLine],
                    state: VerifierState) -> bool:
    """
    EQUALITY2: transitivity.

    (=[a,c]) from (=[a,b]) and (=[b,c]). All namespaces must be equal.
    """
    if len(line.rest) < 4:
        return False

    eq1_expr, eq1_ns = line.rest[0], line.rest[1]
    eq2_expr, eq2_ns = line.rest[2], line.rest[3]

    if line.namespace != eq1_ns or line.namespace != eq2_ns:
        return False

    result_args = _extract_args(line.expression)
    eq1_args = _extract_args(eq1_expr)
    eq2_args = _extract_args(eq2_expr)

    if len(result_args) != 2 or len(eq1_args) != 2 or len(eq2_args) != 2:
        return False

    # (=[a,c]) from (=[a,b]) and (=[b,c])
    return (result_args[0] == eq1_args[0]
            and result_args[1] == eq2_args[1]
            and eq1_args[1] == eq2_args[0])


def check_symmetry_of_equality(line: ProofLine, chapter: List[ProofLine],
                               state: VerifierState) -> bool:
    """
    SYMMETRY OF EQUALITY: (=[a,b]) from (=[b,a]). Namespaces must be equal.
    """
    if len(line.rest) < 2:
        return False

    source_expr, source_ns = line.rest[0], line.rest[1]
    if line.namespace != source_ns:
        return False

    result_args = _extract_args(line.expression)
    source_args = _extract_args(source_expr)

    if len(result_args) != 2 or len(source_args) != 2:
        return False

    return result_args[0] == source_args[1] and result_args[1] == source_args[0]


def check_recursion(line: ProofLine, chapter: List[ProofLine],
                    state: VerifierState) -> bool:
    """
    RECURSION checker.

    check_zero case:
      Expression must be (=[ind_var, i0]) where ind_var is the induction
      variable from the theorem. Namespace must be "main".
    """
    if line.namespace != "main":
        return False
    if state.current_chapter_thm is None:
        return False

    _, thm_type, ind_var = state.current_chapter_thm

    if state.current_chapter_type == "check_zero":
        # Must be equality (=[ind_var, i0])
        if not line.expression.startswith("(=["):
            return False
        args = _extract_args(line.expression)
        if len(args) != 2:
            return False
        return args[0] == ind_var and args[1] == "i0"

    if state.current_chapter_type == "check_induction_condition":
        # Either (in2[*, ind_var, s]) or an implication (>[...)
        if line.expression.startswith("(in2["):
            args = _extract_args(line.expression)
            if len(args) < 3:
                return False
            return args[1] == ind_var and args[2] == "s"
        if line.expression.startswith("(>["):
            # Find x from the companion in2[x, ind_var, s] recursion line
            x_var = None
            for ch_line in chapter:
                if (ch_line.tag == "recursion"
                        and ch_line.expression.startswith("(in2[")):
                    in2_args = _extract_args(ch_line.expression)
                    if (len(in2_args) >= 3
                            and in2_args[1] == ind_var
                            and in2_args[2] == "s"):
                        x_var = in2_args[0]
                        break
            if x_var is None:
                return False

            # Take theorem, cut off anchor
            thm_expr = state.current_chapter_thm[0]
            thm_premises, thm_head = disintegrate_implication_full(thm_expr)
            if len(thm_premises) < 2:
                return False

            # Untouchable vars: anchor args + x_var + immutable args
            # (replicates C++ createAuxyImplication logic)
            anchor_args = set(_extract_args(thm_premises[0]))

            # All expressions for digit/immutable analysis (premises + head)
            all_thm_exprs = list(thm_premises) + [thm_head]
            digit_args = _find_digit_args(
                all_thm_exprs, thm_premises[0],
                state.input_indices, state.output_indices)
            immutables = _find_immutable_args(
                all_thm_exprs, digit_args, ind_var,
                state.input_indices, state.output_indices)

            untouchables = set(anchor_args)
            untouchables.add(x_var)
            untouchables.update(immutables)

            # Non-anchor premises + head with ind_var → x_var
            chain_exprs = [_replace_arg_safe(e, ind_var, x_var)
                           for e in thm_premises[1:]]
            head_expr = _replace_arg_safe(thm_head, ind_var, x_var)

            # Reconstruct implication (C++ reconstructImplication logic):
            # bound vars = vars in >1 expression, excluding untouchables,
            # placed at first expression where they appear
            all_exprs = chain_exprs + [head_expr]
            counter: Dict[str, int] = {}
            for expr in all_exprs:
                for a in _extract_args(expr):
                    if a in untouchables:
                        continue
                    counter[a] = counter.get(a, 0) + 1
            multi_vars = {a for a, c in counter.items() if c > 1}

            placed: set = set()
            when_removed: List[List[str]] = [[] for _ in range(len(chain_exprs))]
            for idx, expr in enumerate(all_exprs):
                for a in _extract_args(expr):
                    if a in multi_vars and a not in placed:
                        placed.add(a)
                        if idx < len(chain_exprs):
                            when_removed[idx].append(a)

            # Build implication from inside out
            result = head_expr
            for i in range(len(chain_exprs) - 1, -1, -1):
                bound_str = ','.join(when_removed[i])
                result = f'(>[{bound_str}]{chain_exprs[i]}{result})'

            # Normalize both and compare
            return (_normalize_expr_list([result])
                    == _normalize_expr_list([line.expression]))
        return False

    # Other cases not yet implemented
    return False


def check_theorem_tag(line: ProofLine, chapter: List[ProofLine],
                  state: VerifierState) -> bool:
    """
    THEOREM checker.

    The expression must exist in the global theorem list and
    namespace must be "main".

    Compares by normalization: variable renaming inside contradiction LBs
    may produce v-numbering different from the global theorem list.
    """
    if line.namespace != "main":
        return False
    if line.expression in state.global_theorems:
        return True
    # Fallback: normalize and compare
    norm_line = _normalize_expr_list([line.expression])
    for gt_expr in state.global_theorems:
        if _normalize_expr_list([gt_expr]) == norm_line:
            return True
    return False


_INTEGRATION_GOAL_SUFFIX = "_integration_goal"


def _strip_integration_goal(s: str) -> str:
    """Strip the ``_integration_goal`` postfix if present."""
    if s.endswith(_INTEGRATION_GOAL_SUFFIX):
        return s[:-len(_INTEGRATION_GOAL_SUFFIX)]
    return s


def _flatten_and(expr: str) -> List[str]:
    """
    Flatten a left-nested AND expression into its ordered element list.

    ``(&(&(&A B)C)D)`` → ``[A, B, C, D]``

    The C++ builds ANDs as: ``current = elem[0]; for i in 1..n:
    current = "(&" + current + elem[i] + ")"`` — so the leftmost
    leaf is elem[0] and each successive right child is the next element.
    """
    elements: List[str] = []
    while expr.startswith('(&'):
        inner = expr[2:-1]          # strip outer "(& " and ")"
        # Find end of the first (left) sub-expression
        depth = 0
        split = -1
        for i, c in enumerate(inner):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    split = i + 1
                    break
        if split < 0:
            break
        left = inner[:split]
        right = inner[split:]
        elements.append(right)      # right child = next element
        expr = left                 # recurse into left child
    elements.append(expr)           # leftmost leaf = first element
    elements.reverse()
    return elements


def _normalize_existence_bound_vars(expr: str) -> Optional[str]:
    """
    Normalize bound-variable names in an existence expansion:
        !(>[vars](left)!(right))

    Only bound variables are renamed; signature variables stay unchanged.
    """
    parsed = _parse_existence_expansion(expr)
    if parsed is None:
        return None

    bound_csv, _, _ = parsed
    bounds = [b for b in bound_csv.split(',') if b]
    result = expr
    for idx, bound in enumerate(bounds, start=1):
        result = _replace_arg_safe(result, bound, f'bv{idx}')
    return result



def _infer_missing_existence_bound_var(premises: List[str], head: str) -> Optional[str]:
    """
    Infer the missing existential variable from a reformulated implication.

    The existential variable is the argument shared by both premises but not
    present in the compiled head.
    """
    if len(premises) != 2:
        return None

    head_args = set(_extract_args(head))
    shared = set(_extract_args(premises[0])) & set(_extract_args(premises[1]))
    candidates = [arg for arg in shared if arg not in head_args]

    if len(candidates) != 1:
        return None
    return candidates[0]



def _reformulation_implication_to_existence(expr: str) -> Optional[Tuple[str, str]]:
    """
    Convert a reformulated existence implication into existence form.

    Example:
      (>[v3](in[v3,N])(>[](in3[v1,v3,v2,+])(preorder[N,+,v1,v2])))
    becomes:
      !(>[v3](in[v3,N])!(in3[v1,v3,v2,+]))

    Returns (converted_existence_expr, head_expr).
    """
    premises, head = disintegrate_implication_full(expr)
    if len(premises) != 2:
        return None

    m = re.match(r'^\(>\[([^\]]*)\]', expr)
    if m is None:
        return None

    bound_csv = m.group(1)
    if not bound_csv:
        inferred = _infer_missing_existence_bound_var(premises, head)
        if inferred is None:
            return None
        bound_csv = inferred

    existence_expr = f'!(>[{bound_csv}]{premises[0]}!{premises[1]})'
    return existence_expr, head



def _reformulation_integration_preamble(line: ProofLine,
                                        chapter: List[ProofLine],
                                        state: VerifierState):
    """
    Shared preamble for all three reformulation-for-integration tags.

    Returns (entry, compact, right_expr) on success, None on failure.
    ``entry`` is the GL binary entry, ``compact`` is the head expression,
    ``right_expr`` is the expansion-for-integration expression.
    """
    if len(line.rest) < 2:
        return None

    right_expr = line.rest[0]
    right_ns = line.rest[1]

    if line.namespace != right_ns:
        return None

    expansion_origin = None
    for ch_line in chapter:
        if (ch_line.expression == right_expr
                and ch_line.namespace == right_ns
                and ch_line.tag == 'expansion for integration'
                and len(ch_line.rest) >= 2):
            expansion_origin = ch_line
            break

    if expansion_origin is None:
        return None

    compact = _strip_integration_goal(expansion_origin.rest[0])
    core = _extract_core_name(compact)

    entry = None
    for binary in state.gl_binaries.values():
        if core in binary:
            entry = binary[core]
            break

    if entry is None:
        return None

    _, proof_head = disintegrate_implication_full(line.expression)
    if compact != proof_head:
        return None

    return entry, compact, right_expr



def _check_reformulation_integration_and(line, compact, right_expr):
    """AND category: rebuild implication chain from expanded AND form."""
    right_expr_clean = _strip_integration_goal(right_expr)
    if not right_expr_clean.startswith('(&'):
        return False

    elements = _flatten_and(right_expr_clean)
    if not elements:
        return False

    expected = compact
    for elem in reversed(elements):
        expected = f'(>[]{elem}{expected})'

    return line.expression == expected


def _check_reformulation_integration_existence(line, entry, compact):
    """Existence category: convert implication to existence form, compare."""
    converted = _reformulation_implication_to_existence(line.expression)
    if converted is None:
        return False

    left_as_existence, converted_head = converted
    if converted_head != compact:
        return False

    sig_args = _extract_args(entry.get('signature', ''))
    actual_args = _extract_args(compact)
    elements = entry.get('elements', [])

    if len(sig_args) != len(actual_args) or len(elements) != 2:
        return False

    subst = dict(zip(sig_args, actual_args))
    expected_children = [_replace_arg_safe_multi(elem, subst)
                         for elem in elements]
    expected_existence = _build_existence_from_elements(expected_children)

    left_norm = _normalize_existence_bound_vars(left_as_existence)
    expected_norm = _normalize_existence_bound_vars(expected_existence)
    if left_norm is None or expected_norm is None:
        return False

    return left_norm == expected_norm


def check_reformulation_for_integration_and(line: ProofLine,
                                            chapter: List[ProofLine],
                                            state: VerifierState) -> bool:
    """REFORMULATION FOR INTEGRATION AND — AND category only."""
    preamble = _reformulation_integration_preamble(line, chapter, state)
    if preamble is None:
        return False
    entry, compact, right_expr = preamble
    if entry.get('category') != 'and':
        return False
    return _check_reformulation_integration_and(line, compact, right_expr)


def check_reformulation_for_integration_bound(line: ProofLine,
                                              chapter: List[ProofLine],
                                              state: VerifierState) -> bool:
    """
    REFORMULATION FOR INTEGRATION >[bound] — existence with pi_lev_ bound var.

    The outermost >[...] must be non-empty (has bound var).
    The expression has the negated existence form.
    """
    preamble = _reformulation_integration_preamble(line, chapter, state)
    if preamble is None:
        return False
    entry, compact, right_expr = preamble
    if entry.get('category') != 'existence':
        return False

    # Verify outermost >[...] has a bound variable
    m = re.match(r'^\(>\[([^\]]*)\]', line.expression)
    if m is None or not m.group(1):
        return False

    return _check_reformulation_integration_existence(line, entry, compact)


def check_reformulation_for_integration_empty(line: ProofLine,
                                              chapter: List[ProofLine],
                                              state: VerifierState) -> bool:
    """
    REFORMULATION FOR INTEGRATION >[] — existence with occupied (non-pi_) bound var.

    The outermost >[...] must be empty (>[]). The bound var was stripped
    because it was occupied. Need to infer it for validation.
    """
    preamble = _reformulation_integration_preamble(line, chapter, state)
    if preamble is None:
        return False
    entry, compact, right_expr = preamble
    if entry.get('category') != 'existence':
        return False

    # Verify outermost >[...] is empty
    m = re.match(r'^\(>\[([^\]]*)\]', line.expression)
    if m is None:
        return False
    if m.group(1):
        return False  # not empty — wrong tag

    return _check_reformulation_integration_existence(line, entry, compact)


def check_expansion_for_integration(line: ProofLine,
                                    chapter: List[ProofLine],
                                    state: VerifierState) -> bool:
    """
    EXPANSION FOR INTEGRATION checker.

    Same as expansion but right-side expression does not need to be
    present as a left-side expression in the chapter.
    1. Namespaces must match.
    2. Structural expansion check via GL binary (after stripping
       ``_integration_goal`` postfix).
    """
    if len(line.rest) < 2:
        return False

    right_expr = line.rest[0]
    right_ns = line.rest[1]

    if line.namespace != right_ns:
        return False

    # Strip _integration_goal postfix for structural checks
    left_clean = _strip_integration_goal(line.expression)
    right_clean = _strip_integration_goal(right_expr)

    core = _extract_core_name(right_clean)

    for binary in state.gl_binaries.values():
        if core in binary:
            if _try_expand(binary[core], right_clean, left_clean):
                return True

    return False


def check_premise_element(line: ProofLine, chapter: List[ProofLine],
                          state: VerifierState) -> bool:
    """
    PREMISE ELEMENT checker.

    1. rest[0] is the origin implication (may carry ``_integration_goal``
       postfix). Disintegrate the *clean* form and verify that
       line.expression is one of its premises.
    2. Find a chapter line where expression == rest[0] (the origin
       implication) and tag == "expansion for integration".
    3. That expansion line's rest[0] (after stripping
       ``_integration_goal``) must equal line.namespace
       (the validityName of this premise element).
    """
    if len(line.rest) < 1:
        return False

    origin_impl = line.rest[0]

    # Step 1: strip postfix before disintegrating
    origin_impl_clean = _strip_integration_goal(origin_impl)
    premises, _ = disintegrate_implication_full(origin_impl_clean)
    if line.expression not in premises:
        return False

    # Step 2 + 3: match on raw origin_impl (with postfix), but
    # strip postfix from ch_line.rest[0] before comparing with namespace
    for ch_line in chapter:
        if (ch_line.expression == origin_impl
                and ch_line.tag == "expansion for integration"
                and len(ch_line.rest) >= 1
                and _strip_integration_goal(ch_line.rest[0]) == line.namespace):
            return True

    return False


def check_validity_name(line: ProofLine, chapter: List[ProofLine],
                        state: VerifierState) -> bool:
    """
    VALIDITY NAME checker.

    Find a chapter line where the implication name (line.expression) is on
    the right side with tag "expansion for integration". Disintegrate the
    left side of that line. The head must equal rest[0].
    """
    if len(line.rest) < 1:
        return False

    target_expr = line.rest[0]
    impl_name = line.expression

    for ch_line in chapter:
        if (ch_line.tag == "expansion for integration"
                and len(ch_line.rest) >= 1
                and _strip_integration_goal(ch_line.rest[0]) == impl_name):
            head = disintegrate_implication_head(
                _strip_integration_goal(ch_line.expression))
            return head == target_expr

    return False


def check_anchor_handling(line: ProofLine, chapter: List[ProofLine],
                          state: VerifierState) -> bool:
    """
    ANCHOR HANDLING checker.

    Both sides are anchor expressions with the same anchor tag.
    Differing arguments must have "(1)" definition set.
    Both namespaces must be "main".
    The right-side (origin) anchor must exist in the chapter as a
    task formulation line.
    """
    if line.namespace != "main":
        return False
    if len(line.rest) < 2:
        return False

    origin_expr = line.rest[0]
    origin_ns = line.rest[1]
    if origin_ns != "main":
        return False

    # Both must be the same anchor type
    target_core = _extract_core_name(line.expression)
    origin_core = _extract_core_name(origin_expr)
    if target_core != origin_core:
        return False
    if not target_core.startswith("Anchor"):
        return False

    # Extract args and compare
    target_args = _extract_args(line.expression)
    origin_args = _extract_args(origin_expr)
    if len(target_args) != len(origin_args):
        return False

    # Differing positions must have "(1)" definition set
    ds = state.definition_sets.get(target_core, {})
    for i, (ta, oa) in enumerate(zip(target_args, origin_args)):
        if ta != oa:
            # 1-based position key
            pos_key = str(i + 1)
            pos_ds = ds.get(pos_key, [])
            if not pos_ds or pos_ds[0] != "(1)":
                return False

    # Origin anchor must exist as task formulation in chapter
    for ch_line in chapter:
        if (ch_line.expression == origin_expr
                and ch_line.namespace == "main"
                and ch_line.tag == "task formulation"):
            return True

    return False


def check_mirrored_from(line: ProofLine, chapter: List[ProofLine],
                        state: VerifierState) -> bool:
    """
    MIRRORED FROM checker.

    rest[0] is the source theorem. Verify mirroring via _check_mirror.
    Namespace must be "main".
    """
    if line.namespace != "main":
        return False
    if len(line.rest) < 1:
        return False
    source_expr = line.rest[0]
    return _check_mirror(source_expr, line.expression, state.output_indices)


def check_reformulated_from(line: ProofLine, chapter: List[ProofLine],
                            state: VerifierState) -> bool:
    """
    REFORMULATED FROM checker.

    rest[0] is the source theorem. Verify reformulation via _check_reformulation.
    Namespace must be "main".
    """
    if line.namespace != "main":
        return False
    if len(line.rest) < 1:
        return False
    source_expr = line.rest[0]
    return _check_reformulation(source_expr, line.expression, state.gl_binaries)


def check_incubator_back_reformulation(line: ProofLine, chapter: List[ProofLine],
                                        state: VerifierState) -> bool:
    """
    INCUBATOR BACK REFORMULATION checker (stub).

    rest[0] is the source reformulated theorem (Anchor -> op -> =[x,a]).
    Full verification deferred — incubator proof graphs not yet verified.
    """
    if line.namespace != "main":
        return False
    if len(line.rest) < 1:
        return False
    return True


def check_externally_provided_theorem(line: ProofLine, chapter: List[ProofLine],
                                       state: VerifierState) -> bool:
    """
    EXTERNALLY PROVIDED THEOREM checker.

    Verifies that the expression is in processed_proof_graph/external_theorems.txt
    (which includes raw + renamed forms and mirrored variants). Falls back to checking if
    the expression is a valid mirror of a known external theorem
    using the verifier's own _check_mirror logic.
    Namespace must be "main".
    """
    if line.namespace != "main":
        return False
    # Direct membership (covers originals + mirrors written by Python)
    if line.expression in state.external_theorems:
        return True
    # Fallback: check if this expression mirrors a known external theorem
    for ext in state.external_theorems:
        if _check_mirror(line.expression, ext, state.output_indices):
            return True
    return False


def check_necessity_for_equality_hypo(line: ProofLine,
                                      chapter: List[ProofLine],
                                      state: VerifierState) -> bool:
    """
    NECESSITY FOR EQUALITY (HYPO) checker.

    1. Expression must be equality (=[a,b]).
    2. 'a' must appear 2+ times in the right-side expression (rest[0]) args.
    3. For every line where 'b' appears in expression args (excluding equalities
       containing 'b'): recursively trace 'b' backwards through origin
       expressions until reaching the equality under scrutiny. If any path
       cannot reach it → fail.
    4. Namespaces must match.
    """
    if len(line.rest) < 2:
        return False
    if not line.expression.startswith("(=["):
        return False

    target_expr, target_ns = line.rest[0], line.rest[1]
    if line.namespace != target_ns:
        return False

    eq_args = _extract_args(line.expression)
    if len(eq_args) != 2:
        return False

    a, b = eq_args[0], eq_args[1]

    # 'a' must appear 2+ times in right-side expression args
    target_args = _extract_args(target_expr)
    if target_args.count(a) < 2:
        return False

    # 'b' must not appear in right-side expression args
    if b in target_args:
        return False

    # Build expression → lines map for left-side lookup
    expr_to_lines: Dict[str, List[ProofLine]] = {}
    for ch_line in chapter:
        expr_to_lines.setdefault(ch_line.expression, []).append(ch_line)

    the_equality = line.expression  # e.g. (=[v1,v3])
    is_target = lambda cl: cl.expression == the_equality
    should_follow = lambda src: b in _extract_args(src)

    # Check every line where b appears in expression args
    for ch_line in chapter:
        if b not in _extract_args(ch_line.expression):
            continue
        # Skip equalities containing b (they are part of the propagation chain)
        if ch_line.expression.startswith("(=["):
            continue
        # This expression must trace back to the_equality
        if not _trace_back_to(ch_line.expression, expr_to_lines,
                              is_target, should_follow):
            return False

    return True


def check_equalize_variable(line: ProofLine, chapter: List[ProofLine],
                            state: VerifierState) -> bool:
    """Verify equalize variable: origin and copy must have consistent arg mapping."""
    if len(line.rest) < 2:
        return False
    origin_expr = line.rest[0]
    copy_expr = line.expression
    origin_premises, origin_head = disintegrate_implication_full(origin_expr)
    copy_premises, copy_head = disintegrate_implication_full(copy_expr)
    if len(origin_premises) != len(copy_premises):
        return False
    pairs = list(zip(origin_premises, copy_premises))
    pairs.append((origin_head, copy_head))
    mapping: Dict[str, str] = {}
    for orig_elem, copy_elem in pairs:
        if _extract_core_name(orig_elem) != _extract_core_name(copy_elem):
            return False
        orig_args = _extract_args(orig_elem)
        copy_args = _extract_args(copy_elem)
        if len(orig_args) != len(copy_args):
            return False
        for oa, ca in zip(orig_args, copy_args):
            if oa in mapping:
                if mapping[oa] != ca:
                    return False
            else:
                mapping[oa] = ca
    return True


def check_contradiction(line: ProofLine, chapter: List[ProofLine],
                        state: VerifierState) -> bool:
    """
    CONTRADICTION checker.

    A contradiction line records that !(cleanOp) was proved because both
    expr and negate(expr) were derived in the contradiction LB.
    rest layout: expr, ns, negate(expr), ns, cleanOp, ns

    Checks:
    1. line.expression == "!" + cleanOp
    2. expr and negate(expr) are negations of each other
    3. Both expr and negate(expr) exist as expressions in the chapter
    4. cleanOp exists as a task formulation in the chapter
    """
    if line.namespace != "main":
        return False
    if len(line.rest) < 6:
        return False

    expr = line.rest[0]
    expr_ns = line.rest[1]
    neg_expr = line.rest[2]
    neg_ns = line.rest[3]
    clean_op = line.rest[4]
    clean_ns = line.rest[5]

    # All namespaces must be "main"
    if expr_ns != "main" or neg_ns != "main" or clean_ns != "main":
        return False

    # line.expression must be "!" + cleanOp
    if line.expression != "!" + clean_op:
        return False

    # expr and neg_expr must be negations of each other
    if neg_expr == "!" + expr:
        pass
    elif expr == "!" + neg_expr:
        pass
    else:
        return False

    # cleanOp must appear as task formulation in the chapter
    for ch in chapter:
        if ch.expression == clean_op and ch.tag == "task formulation":
            return True

    return False


# Tag name → checker function
TAG_CHECKERS = {
    "implication":                      check_implication,
    "expansion":                        check_expansion,
    "disintegration":                   check_disintegration,
    "task formulation":                 check_task_formulation,
    "equality1":                        check_equality1,
    "equality2":                        check_equality2,
    "symmetry of equality":             check_symmetry_of_equality,
    "recursion":                        check_recursion,
    "theorem":                          check_theorem_tag,
    "reformulation for integration and": check_reformulation_for_integration_and,
    "reformulation for integration >[bound]": check_reformulation_for_integration_bound,
    "reformulation for integration >[]": check_reformulation_for_integration_empty,
    "expansion for integration":        check_expansion_for_integration,
    "premise element":                  check_premise_element,
    "validity name":                    check_validity_name,
    "anchor handling":                  check_anchor_handling,
    "mirrored from":                    check_mirrored_from,
    "reformulated from":                check_reformulated_from,
    "necessity for equality (hypo)":    check_necessity_for_equality_hypo,
    "externally provided theorem":      check_externally_provided_theorem,
    "incubator back reformulation":     check_incubator_back_reformulation,
    "equalize variable":                check_equalize_variable,
    "multiplied from":                  check_equalize_variable,
    "contradiction":                    check_contradiction,
}


# ---------------------------------------------------------------------------
#  Chapter-level verification
# ---------------------------------------------------------------------------

def verify_chapter(chapter_file: str, lines: List[ProofLine],
                   chapter_type: str, state: VerifierState,
                   chapter_thm: Optional[Tuple[str, str, str]]):
    """Verify a single chapter.  Updates state counters in-place."""

    # 1. Theorem goal reached
    goal_ok = check_theorem_goal_reached(
        chapter_file, chapter_type, lines, chapter_thm,
        state.output_indices, state.gl_binaries)
    state.goal_reached.record(goal_ok)

    # Set transient chapter context for line checkers
    state.current_chapter_thm = chapter_thm
    state.current_chapter_type = chapter_type

    # 2. Self-reference check: theorem must not appear as its own justification
    if chapter_thm is not None:
        thm_expr = chapter_thm[0]
        for line in lines:
            if line.tag == "theorem" and line.expression == thm_expr:
                state.counter_for("self-reference").record(False)

    # 3. Anchor handling uniqueness: at most one per chapter
    anchor_handling_count = sum(1 for line in lines if line.tag == "anchor handling")
    if anchor_handling_count > 1:
        state.counter_for("anchor handling uniqueness").record(False)

    # Build expr → lines map once for chapter-level trace checks
    _ch_expr_to_lines: Dict[str, List[ProofLine]] = {}
    for ch_line in lines:
        _ch_expr_to_lines.setdefault(ch_line.expression, []).append(ch_line)

    # 4. Anchor handling trace: every _copy var in rest fields must trace
    #    back to the anchor handling line
    anchor_line = None
    for line in lines:
        if line.tag == "anchor handling":
            anchor_line = line
            break

    if anchor_line is not None:
        anchor_args = _extract_args(anchor_line.expression)
        copy_vars = [a for a in anchor_args if a.endswith("_copy")]

        if copy_vars:
            _anc_target = lambda cl: cl.tag == "anchor handling"

            for cv in copy_vars:
                _cv_re = re.compile(
                    r'(?<=[\[,])' + re.escape(cv) + r'(?=[\],])')
                _cv_follow = lambda src, _r=_cv_re: bool(_r.search(src))

                for ch_line in lines:
                    if ch_line.tag == "anchor handling":
                        continue
                    for i in range(0, len(ch_line.rest), 2):
                        src = ch_line.rest[i]
                        if _cv_follow(src):
                            ok = _trace_back_to(src, _ch_expr_to_lines,
                                                _anc_target, _cv_follow)
                            state.counter_for("anchor handling trace").record(ok)

    # 5. Contradiction trace: at least one of the two contradicting expressions
    #    must trace back to the seed (task formulation) through all ingredients
    for line in lines:
        if line.tag != "contradiction":
            continue
        if len(line.rest) < 6:
            state.counter_for("contradiction trace").record(False)
            continue

        seed_expr = line.rest[4]
        _seed_target = (lambda cl, _s=seed_expr:
                        cl.tag == "task formulation" and cl.expression == _s)

        ok = (_trace_back_to(line.rest[0], _ch_expr_to_lines, _seed_target)
              or _trace_back_to(line.rest[2], _ch_expr_to_lines, _seed_target))
        state.counter_for("contradiction trace").record(ok)

    # 6. Dispatch every line to its tag checker
    for line in lines:
        tag = line.tag
        if tag in TAG_CHECKERS:
            passed = TAG_CHECKERS[tag](line, lines, state)
            state.counter_for(tag).record(passed)
        else:
            state.counter_for(f"<unknown:{tag}>").record(False)

    # 4. General origin check: every expression referenced as a dependency
    #    (in rest fields) must appear as a left-side expression (have its
    #    own derivation line). Exceptions:
    #    - integration-goal postfixed exprs
    #    - incubator back reformulation deps (external theorem references)
    #    - contradiction ingredient deps (may not be fully traced)
    _ORIGIN_EXEMPT_TAGS = {
        "incubator back reformulation",
        "contradiction",
    }
    originated = {line.expression for line in lines}
    for line in lines:
        if line.tag in _ORIGIN_EXEMPT_TAGS:
            continue
        for i in range(0, len(line.rest), 2):
            dep = line.rest[i]
            if dep in originated:
                continue
            if dep.endswith("_integration_goal"):
                continue
            # rest[0] global theorem check: implication rules and multiplied-from sources
            if i == 0 and line.tag in ("implication", "multiplied from", "mirrored from", "reformulated from"):
                norm_dep = _normalize_expr_list([dep])
                found = dep in state.global_theorems or dep in state.external_theorems or any(
                    _normalize_expr_list([gt]) == norm_dep
                    for gt in (state.global_theorems.keys() | state.external_theorems)
                )
                state.counter_for("origin").record(found)
                continue
            state.counter_for("origin").record(False)


# ---------------------------------------------------------------------------
#  Main driver
# ---------------------------------------------------------------------------

def chapter_sort_key(filename: str) -> Tuple[int, str]:
    base = filename.replace(".txt", "")
    parts = base.split("_", 1)
    try:
        return (int(parts[0]), parts[1] if len(parts) > 1 else "")
    except ValueError:
        return (999999, base)


def run_verifier(base_dir: str) -> VerifierState:
    state = VerifierState()

    # Load global theorem list
    state.global_theorems, state.global_theorem_list = \
        load_global_theorem_list(base_dir)

    # Load output indices from config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "files", "config")
    state.output_indices = load_output_indices(config_dir)
    state.input_indices = load_input_indices(config_dir)
    state.definition_sets = load_definition_sets(config_dir)

    # Load GL binaries — look next to the processed proof graph first,
    # then fall back to the default location
    binaries_dir = os.path.join(os.path.dirname(base_dir), "GL_binaries")
    if not os.path.isdir(binaries_dir):
        binaries_dir = os.path.join(script_dir, "files", "GL_binaries")
    state.gl_binaries = load_gl_binaries(binaries_dir)

    # Load external theorems
    ext_file = os.path.join(base_dir, "external_theorems.txt")
    if os.path.exists(ext_file):
        with open(ext_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    state.external_theorems.add(line)

    # Enumerate chapter files
    _exclude = {"global_theorem_list.txt", "external_theorems.txt"}
    chapter_files = sorted(
        [f for f in os.listdir(base_dir)
         if f.endswith(".txt") and f not in _exclude],
        key=chapter_sort_key,
    )

    # Build chapter → theorem mapping
    chapter_thm_map = build_chapter_theorem_map(
        chapter_files, state.global_theorem_list)

    # Process every chapter
    for cf in chapter_files:
        filepath = os.path.join(base_dir, cf)
        lines = parse_chapter_file(filepath)
        base = cf.replace(".txt", "")
        parts = base.split("_", 1)
        chapter_type = parts[1] if len(parts) > 1 else "unknown"
        chapter_thm = chapter_thm_map.get(cf)

        verify_chapter(cf, lines, chapter_type, state, chapter_thm)

    return state


def get_totals(state: VerifierState) -> tuple[int, int]:
    g = state.goal_reached
    total_success = g.success
    total_failure = g.failure
    for tag in TAG_CHECKERS:
        ctr = state.tag_counters.get(tag, TagCounter())
        total_success += ctr.success
        total_failure += ctr.failure
    for tag, ctr in state.tag_counters.items():
        if tag not in TAG_CHECKERS:
            total_failure += ctr.failure
    return total_success, total_failure


def print_report(state: VerifierState):
    W = 44
    g = state.goal_reached
    print(f"{'theorem goal reached':<{W}s}success {g.success}, failure {g.failure}")
    for tag in TAG_CHECKERS:
        ctr = state.tag_counters.get(tag, TagCounter())
        print(f"{tag:<{W}s}success {ctr.success}, failure {ctr.failure}")
    for tag, ctr in state.tag_counters.items():
        if tag not in TAG_CHECKERS:
            print(f"{tag:<{W}s}success {ctr.success}, failure {ctr.failure}")
    total_success, total_failure = get_totals(state)
    if total_failure == 0:
        print(f"\n          {total_success} checks, 0 failures — airtight.")
    else:
        print(f"\nVerifier: {total_success + total_failure} checks, {total_failure} FAILED.")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "files", "processed_proof_graph")

    if not os.path.isdir(base_dir):
        print(f"ERROR: directory not found: {base_dir}", file=sys.stderr)
        sys.exit(1)

    state = run_verifier(base_dir)
    print_report(state)


if __name__ == "__main__":
    main()
