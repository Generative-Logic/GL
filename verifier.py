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

import argparse
import os
import re
import sys
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional


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
    current_gl_binary: Optional[dict] = None  # the GL binary for the current chapter's batch

    # Definition sets: core_name → { "1-based-position" → [def_set_str, bool] }
    definition_sets: Dict[str, Dict[str, list]] = field(default_factory=dict)

    # Resolved defset index — PER-TAG. tag → { core_name → { pos_str → type_label_str } }
    # Mirrors the compiler's per-batch ArgumentAnalyzer(this->coreExpressionMap)
    # construction at prover.hpp:3556 — each batch resolves composites against
    # its own binary, so the same compact name (e.g. `implication26`) can mean
    # different things in different batches without colliding. Per-chapter the
    # right tag is selected the same way `current_gl_binary` is (anchor substring
    # match on the chapter's theorem expression). See D-41.
    resolved_defsets_per_tag: Dict[str, Dict[str, Dict[str, str]]] = field(default_factory=dict)
    # Atomic-only fallback (no batch-specific composites) — used for chapters
    # whose theorem doesn't disclose an anchor and the per-tag selection
    # therefore fails. Composites unresolved → unknown-skip per check.
    resolved_defsets_atomic_only: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # Per-chapter pointer set by verify_chapter; check_defset_consistency reads.
    current_resolved_defsets: Optional[Dict[str, Dict[str, str]]] = None

    # External theorems (raw + renamed forms)
    external_theorems: set = field(default_factory=set)

    def binaries_for_chapter(self) -> list:
        """Return the GL binary(ies) relevant to the current chapter."""
        if self.current_gl_binary is not None:
            return [self.current_gl_binary]
        return list(self.gl_binaries.values())

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
    """Returns chapter_filename → (theorem_expr, theorem_type, theorem_ref).

    For `induction` rows the chapter triple is, in order:
      <i>_induction_typing.txt, <i+1>_check_zero.txt, <i+2>_check_induction_condition.txt
    — typing first because it is the gate that establishes `(in[ind_var,N])`,
    which both subsequent chapters rely on. Structural invariant enforced:
    each induction row must have exactly three consecutive chapters at the
    expected filename suffixes. Violation raises AssertionError — the
    verifier's "airtight or it fails loudly" stance.
    """
    mapping: Dict[str, Tuple[str, str, str]] = {}
    ci = 0
    for thm_expr, thm_type, thm_ref in theorem_list:
        if ci >= len(chapter_files):
            break
        if thm_type == "induction":
            # Structural invariant: three consecutive chapters at the right
            # suffixes, in the order typing → check_zero → check_induction_condition.
            assert ci + 2 < len(chapter_files), (
                f"induction theorem {thm_expr!r} missing three chapters "
                f"(available: {chapter_files[ci:ci+3]})"
            )
            assert chapter_files[ci].endswith("_induction_typing.txt"), (
                f"induction chapter at index {ci} must be *_induction_typing.txt, "
                f"got {chapter_files[ci]!r}"
            )
            assert chapter_files[ci + 1].endswith("_check_zero.txt"), (
                f"induction chapter at index {ci + 1} must be *_check_zero.txt, "
                f"got {chapter_files[ci + 1]!r}"
            )
            assert chapter_files[ci + 2].endswith("_check_induction_condition.txt"), (
                f"induction chapter at index {ci + 2} must be "
                f"*_check_induction_condition.txt, got {chapter_files[ci + 2]!r}"
            )
            mapping[chapter_files[ci]]     = (thm_expr, thm_type, thm_ref)
            mapping[chapter_files[ci + 1]] = (thm_expr, thm_type, thm_ref)
            mapping[chapter_files[ci + 2]] = (thm_expr, thm_type, thm_ref)
            ci += 3
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
        if p_start >= len(expr):
            break
        # Premise can be negated: !(...)
        if expr[p_start] == '!' and p_start + 1 < len(expr) and expr[p_start + 1] == '(':
            p_end = _find_matching_paren(expr, p_start + 1)
        elif expr[p_start] == '(':
            p_end = _find_matching_paren(expr, p_start)
        else:
            break
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


def _extract_all_args(expr: str) -> List[str]:
    """Extract arguments from every ``[..]`` group in *expr*, including nested
    operator subexpressions like ``!(>[v6](in[v6,N])!(in2[v6,i1_copy,s]))``.

    Used by trace-back filters that need to detect whether a copy variable is
    present anywhere in a compound source expression — ``_extract_args`` only
    looks at the outermost bracket group and therefore misses nested args.
    """
    out: List[str] = []
    for group in re.findall(r'\[([^\]]*)\]', expr):
        for a in group.split(','):
            if a:
                out.append(a)
    return out


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
        if p_start >= len(expr):
            break
        # Premise can be negated: !(...)
        if expr[p_start] == '!' and p_start + 1 < len(expr) and expr[p_start + 1] == '(':
            inner_end = _find_matching_paren(expr, p_start + 1)
            if inner_end < 0:
                break
            p_end = inner_end
            premise = expr[p_start:p_end + 1]
        elif expr[p_start] == '(':
            p_end = _find_matching_paren(expr, p_start)
            if p_end < 0:
                break
            premise = expr[p_start:p_end + 1]
        else:
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


def _alpha_canonicalize_bound_vars(expr: str) -> str:
    """
    Canonicalize every name introduced inside a `>[...]` binder of `expr` to
    `b1, b2, ...` in declaration order. Names not appearing in any binder
    (anchor slots, operator constants, integer literals like `i0`/`i1` that
    are anchor-slot names, etc.) are left untouched.

    Used by the origin check to compare a chapter row's `rest[0]` implication
    rule against the global theorem registry across batches: an incubator
    chapter may name a Peano-batch rule's bound variable `i2` (its local
    free-index counter), while the registry stored the same rule with the
    bound variable renamed to `v1` by `process_proof_graphs.py`. Both forms
    must canonicalize to the same string.

    Limitation: the rename is applied uniformly across `expr`. Two disjoint
    `>[…]` binders that happen to use the same name would collapse, but
    well-formed GL implications never do that within a single rule
    expression.
    """
    bound_order: List[str] = []
    seen_bound: Set[str] = set()
    for m in re.finditer(r'>\[([^\]]*)\]', expr):
        names = m.group(1)
        if not names:
            continue
        for n in names.split(','):
            n = n.strip()
            if n and n not in seen_bound:
                seen_bound.add(n)
                bound_order.append(n)
    if not bound_order:
        return expr
    rename = {n: f'b{i + 1}' for i, n in enumerate(bound_order)}
    pattern = re.compile(r'(?<=[\[,])('
                         + '|'.join(re.escape(n) for n in bound_order)
                         + r')(?=[\],])')
    return pattern.sub(lambda m: rename[m.group(0)], expr)



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

    # Select GL binary. Try the exact anchor-derived tag first. If absent
    # (D-35 — the incubator runs `AnchorIncubator` but the loaded binaries
    # are tagged `IncubatorPeano` / `IncubatorGauss` / `IncubatorGauss1`,
    # so the literal tag `Incubator` has no entry), fall back to scanning
    # every loaded binary for one that defines the head's compiled name as
    # an `existence` entry.
    head_core = _extract_core_name(tgt_head)
    binary = gl_binaries.get(tag)
    if binary is None or head_core not in binary \
            or binary[head_core].get("category") != "existence":
        binary = None
        for candidate in gl_binaries.values():
            entry = candidate.get(head_core)
            if entry is not None and entry.get("category") == "existence":
                binary = candidate
                break
        if binary is None:
            return False

    # Target head must be an existence expression
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

    # or_theorem: single line with tag "or theorem"
    if chapter_type == "or_theorem":
        if first.namespace != "main" or first.tag != "or theorem":
            return False
        if len(first.rest) < 2:
            return False
        return True

    # direct proof, check_zero, check_induction_condition:
    # extract head from theorem, compare with first line
    if chapter_type in ("direct_proof", "check_zero", "check_induction_condition"):
        if first.namespace != "main":
            return False
        head = disintegrate_implication_head(thm_expr)
        return first.expression == head

    # induction_typing: third induction chapter proving (in[ind_var, anchor_args[0]])
    # — the typing sub-proof gating induction promotion in the prover.
    # Chapter's first line's expression must equal that typing goal.
    # See docs/induction_typing_plan.md.
    if chapter_type == "induction_typing":
        if first.namespace != "main":
            return False
        # ind_var is the 3rd field of the induction row in global_theorem_list.
        _, _, ind_var = chapter_thm
        # anchor_args[0] = N: find the first `(Anchor...[...])` occurrence in
        # the theorem expression and take its first argument.
        anchor_match = re.search(r"\(Anchor[A-Za-z0-9_]*\[([^\]]*)\]\)", thm_expr)
        if not anchor_match:
            return False
        anchor_args_raw = anchor_match.group(1)
        if not anchor_args_raw:
            return False
        n_name = anchor_args_raw.split(",", 1)[0]
        typing_goal = f"(in[{ind_var},{n_name}])"
        return first.expression == typing_goal

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

    Namespace rule (D-35 — comparable-scope premise inheritance).
      Every premise namespace and the implication's namespace must be one of:
        - `"main"`,
        - the result's namespace, or
        - a strict ancestor of the result's namespace
          (`result_ns.startswith(ns + "_boundary_")`).
      A premise drawn from a deeper-than-result scope is rejected; a premise
      from an ancestor scope is accepted (faithful encoding of GL's
      comparable-scope inheritance — facts at an ancestor scope are visible
      at every descendant). Pre-D-35 the rule was strict "at most one
      distinct non-main namespace, equal to the result's", which rejected
      legitimate FTA-rung-1 implication firings that mixed an OR-branch
      scope (result) with the OR's parent scope (some premises).

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
    result_ns = line.namespace

    # Comparable-scope rule: every source ns must be main, the result ns,
    # or a strict ancestor of it.
    for ns in [impl_ns] + premise_nss:
        if ns == "main":
            continue
        if ns == result_ns:
            continue
        if result_ns.startswith(ns + "_boundary_"):
            continue
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


def _build_or_from_elements(elements: List[str]) -> str:
    """Build nested !(&!(...) !(...)) OR from elements (same as expandSignature OR case)."""
    if len(elements) == 1:
        return elements[0]
    current = f'!(&!{elements[0]}!{elements[1]})'
    for e in elements[2:]:
        current = f'!(&{current}!{e})'
    return current


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
    elif cat == 'or':
        built = _build_or_from_elements(elements)
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


def _build_implication_fullbind(premises: List[str], head: str,
                               unchangeables: set) -> str:
    """Build implication binding ALL variables except unchangeables.

    Mirrors C++ reconstructImplicationFullBind: every variable that is not in
    *unchangeables* is bound in >[...] at its first-appearance level.
    """
    all_exprs = premises + [head]
    placed: set = set()
    when: List[List[str]] = [[] for _ in range(len(premises))]
    for idx, e in enumerate(all_exprs):
        for a in _extract_args(e):
            if a not in unchangeables and a not in placed:
                placed.add(a)
                if idx < len(premises):
                    when[idx].append(a)

    result = head
    for i in range(len(premises) - 1, -1, -1):
        bound_str = ','.join(when[i])
        result = f'(>[{bound_str}]{premises[i]}{result})'
    return result


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
    4. Negated existence: !(existence_name[args]) expands to two implications
       (left→!right) and (right→!left) via FullBind.
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

    # --- Negated existence expansion: !(name[args]) ---
    if right_expr.startswith('!'):
        inner_expr = right_expr[1:]  # strip leading !
        core = _extract_core_name(inner_expr)
        for binary in state.binaries_for_chapter():
            if core not in binary:
                continue
            entry = binary[core]
            if entry.get('category') != 'existence':
                continue
            sig_args = _extract_args(entry.get('signature', ''))
            actual_args = _extract_args(inner_expr)
            if len(sig_args) != len(actual_args):
                continue
            subst = dict(zip(sig_args, actual_args))
            elements = [_replace_arg_safe_multi(e, subst)
                        for e in entry['elements']]
            if len(elements) != 2:
                continue
            left, right = elements[0], elements[1]
            unch = set(actual_args)
            # Two implications from !(>[bound](left)!(right)):
            #   impl1: left -> !(right)
            #   impl2: right -> !(left)
            impl1 = _build_implication_fullbind([left], '!' + right, unch)
            impl2 = _build_implication_fullbind([right], '!' + left, unch)
            target = line.expression
            if (_normalize_with_unchangeables(target, unch)
                    == _normalize_with_unchangeables(impl1, unch)):
                return True
            if (_normalize_with_unchangeables(target, unch)
                    == _normalize_with_unchangeables(impl2, unch)):
                return True
        return False

    core = _extract_core_name(right_expr)

    for binary in state.binaries_for_chapter():
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

            for binary in state.binaries_for_chapter():
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


def _ns_matches_or_strict_prefix(src_ns: str, tgt_ns: str) -> bool:
    """
    True iff ``src_ns`` is the same namespace as ``tgt_ns`` OR ``src_ns`` is
    a strict byte-level prefix of ``tgt_ns``. Mirrors the prover's extension
    that lets an equivalence class at an ancestor validity apply to an
    expression at a strict-descendant validity. Byte-level prefix is safe
    here because validity names use the ``_boundary_`` separator between
    stacked scope payloads, so sibling scopes cannot alias via prefix.
    """
    if src_ns == tgt_ns:
        return True
    return len(src_ns) < len(tgt_ns) and tgt_ns.startswith(src_ns)


def check_equality1(line: ProofLine, chapter: List[ProofLine],
                    state: VerifierState) -> bool:
    """
    EQUALITY1: argument substitution.

    Result and source have the same core and arity. Each differing
    argument position (a→b) must be justified by an equality (=[a,b])
    in rest. The source expression may live either in the same namespace
    as the result OR in a strict byte-level prefix (ancestor validity);
    each justifying equality may live either in the same namespace OR in
    a strict byte-level prefix (ancestor validity). The source-NS widening
    mirrors D-33: a class registered at a strict-descendant validity may
    rewrite an ancestor-scope fact, depositing the result at the
    descendant scope. Sound under descendant-inheritance — facts at
    ancestor scope are observably true at every descendant.
    """
    # rest layout: source, ns, eq1, ns, [eq2, ns, ...]
    if len(line.rest) < 4:
        return False

    source_expr, source_ns = line.rest[0], line.rest[1]
    if not _ns_matches_or_strict_prefix(source_ns, line.namespace):
        return False

    # Collect equalities
    eq_set: set = set()
    i = 2
    while i + 1 < len(line.rest):
        eq_expr, eq_ns = line.rest[i], line.rest[i + 1]
        if not _ns_matches_or_strict_prefix(eq_ns, line.namespace):
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

    (=[a,c]) from (=[a,b]) and (=[b,c]). Each source equality may live
    either in the same namespace as the result OR in a strict byte-level
    prefix (ancestor validity) — mirrors the prover's extended
    equivalence-class application.
    """
    if len(line.rest) < 4:
        return False

    eq1_expr, eq1_ns = line.rest[0], line.rest[1]
    eq2_expr, eq2_ns = line.rest[2], line.rest[3]

    if not _ns_matches_or_strict_prefix(eq1_ns, line.namespace):
        return False
    if not _ns_matches_or_strict_prefix(eq2_ns, line.namespace):
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


def check_symmetry_of_inequality(line: ProofLine, chapter: List[ProofLine],
                                  state: VerifierState) -> bool:
    """
    SYMMETRY OF INEQUALITY: !(=[a,b]) from !(=[b,a]). Namespaces must be equal.
    Emitted by the prover (prover.cpp:~3906) when a negated equality is
    mirrored inside addStatement. The checker mirrors
    check_symmetry_of_equality but strips the leading '!' on both sides.
    """
    if len(line.rest) < 2:
        return False

    source_expr, source_ns = line.rest[0], line.rest[1]
    if line.namespace != source_ns:
        return False

    # Both result and source must be negated equalities `!(=[a,b])`.
    if not line.expression.startswith("!(=[") or not source_expr.startswith("!(=["):
        return False

    result_inner = line.expression[1:]  # drop leading '!'
    source_inner = source_expr[1:]

    result_args = _extract_args(result_inner)
    source_args = _extract_args(source_inner)

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
    for binary in state.binaries_for_chapter():
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

    for binary in state.binaries_for_chapter():
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
    3. The premise element's namespace must be a scope rooted in the
       implication signature. After the NameMap migration, integration
       scopes are minted via encodePush(parent, cleanSig) — canonical
       form is ``<parent>_boundary_<cleanSig>``. So line.namespace must
       end with ``_boundary_<cleanSig>`` where cleanSig is the stripped
       form of ch_line.rest[0]. The equality-to-cleanSig form is kept
       as a backward-compat fallback.
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
    # strip postfix from ch_line.rest[0] before matching against namespace
    for ch_line in chapter:
        if (ch_line.expression == origin_impl
                and ch_line.tag == "expansion for integration"
                and len(ch_line.rest) >= 1):
            clean_sig = _strip_integration_goal(ch_line.rest[0])
            if (line.namespace == clean_sig
                    or line.namespace.endswith("_boundary_" + clean_sig)):
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


def check_or_theorem(line: ProofLine, chapter: List[ProofLine],
                     state: VerifierState) -> bool:
    """
    OR THEOREM checker.

    rest[0] is the existence theorem, rest[2] is the companion.
    The line's expression should be the OR form: !(&(!a)(!b)).
    Namespace must be "main".
    """
    if line.namespace != "main":
        return False
    if len(line.rest) < 2:
        return False
    return True


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


def check_variable_copy(line: ProofLine,
                        chapter: List[ProofLine],
                        state: VerifierState) -> bool:
    """
    VARIABLE COPY checker.

    GL may introduce a free axiom (=[Y, Y_copy]) in any scope, where
    Y_copy = Y + "_copy" is a freshly manufactured suffixed name. This
    is a conservative extension — Y_copy never appears anywhere in the
    model except as a second name for Y — so the axiom is always sound.

    The three C++ emission sites (checkNecessityForEquality,
    disintegrateExprHypothetically, reactToHypo) all emit this shape with
    an empty origin vector; they differ only in *when* the prover chose to
    introduce the copy. The verifier enforces the contract uniformly.

    Checks:
    1. Expression is (=[a, b]) with b == a + "_copy".
    2. Origin vector is empty (dead-end axiom — no rest fields).
    3. Every non-equality chapter line whose expression mentions b in its
       args can reach a 'variable copy' declaration for (=[a, b]) via an
       origin-graph walk. This enforces that b cannot be smuggled into the
       proof graph without an explicit declaration.
    """
    if not line.expression.startswith("(=["):
        return False
    eq_args = _extract_args(line.expression)
    if len(eq_args) != 2:
        return False
    a, b = eq_args[0], eq_args[1]
    if b != a + "_copy":
        return False
    if line.rest:                        # dead-end: no origin refs
        return False

    # Build expression → lines map for trace-back
    expr_to_lines: Dict[str, List[ProofLine]] = {}
    for ch_line in chapter:
        expr_to_lines.setdefault(ch_line.expression, []).append(ch_line)

    the_equality = line.expression
    is_target = lambda cl: (cl.expression == the_equality
                            and cl.tag == "variable copy")
    # Compound sources (e.g. negated implications) hide the copy var inside
    # nested ``[..]`` groups; use the recursive extractor so the trace-back
    # walker keeps following edges that still carry b.
    should_follow = lambda src: b in _extract_all_args(src)

    # Every occurrence of b in a non-equality chapter expression must
    # trace back to a variable copy declaration for (=[a, b]).
    for ch_line in chapter:
        if b not in _extract_args(ch_line.expression):
            continue
        if ch_line.expression.startswith("(=["):
            continue            # skip equalities, they propagate b
        if not _trace_back_to(ch_line.expression, expr_to_lines,
                              is_target, should_follow):
            return False
    return True


def _extract_bound_vars(expr: str) -> Set[str]:
    """Collect every variable name that appears inside any ``>[...]`` binder
    clause of an implication expression — at any nesting depth. Used to
    classify args as bound (locally quantified) vs free (anchor parameters
    or external constants) for the multiplied-from soundness check.
    """
    bound: Set[str] = set()
    for m in re.finditer(r'>\[([^\]]*)\]', expr):
        for name in m.group(1).split(','):
            name = name.strip()
            if name:
                bound.add(name)
    return bound


def check_equalize_variable(line: ProofLine, chapter: List[ProofLine],
                            state: VerifierState) -> bool:
    """Verify equalize variable: origin and copy must have consistent arg mapping.

    Soundness gate: a `multiplied from` step may identify bound variables
    (Bell-partition equalisation), and may rewrite a bound variable to a
    free anchor parameter, but it must NOT identify two distinct free anchor
    parameters with each other. Doing so silently rewrites a free slot of the
    rule body and emits a logically stronger rule than the source — see
    `prover.cpp:multiplyImplication` and the chapter-1115 incubator bug.

    Free vs bound classification: a name is *bound* if it appears inside any
    ``>[...]`` binder clause of the implication; otherwise it is free.
    """
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
    # Free-anchor-merge guard: reject any (orig_arg → copy_arg) where both
    # sides are free (not bound by any ``>[...]`` clause) and the names
    # differ. Bound→bound and bound→free remappings are sound; free→free
    # with different names is the chapter-1115 multiplyImplication bug.
    origin_bound = _extract_bound_vars(origin_expr)
    copy_bound = _extract_bound_vars(copy_expr)
    for orig_arg, copy_arg in mapping.items():
        if orig_arg == copy_arg:
            continue
        if orig_arg in origin_bound:
            continue
        if copy_arg in copy_bound:
            continue
        return False
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


# ---------------------------------------------------------------------------
#  OR disintegration / convergence helpers
# ---------------------------------------------------------------------------

def _parse_or_disjuncts(expanded_or: str) -> List[str]:
    """
    Parse an expanded OR expression like !(&!(&!(=[a,x])!(=[b,x]))!(=[c,x]))
    into a list of disjuncts: ['(=[a,x])', '(=[b,x])', '(=[c,x])'].

    Structure is right-associative nested !(&<left><right>) where:
    - left child starting with !(&  => recurse
    - left child starting with !    => single disjunct (strip leading !)
    - right child always !<disjunct> => single disjunct (strip leading !)
    """
    s = expanded_or.strip()
    if len(s) < 4 or s[:3] != '!(&':
        return [s]  # single expression, not an OR

    # Strip outer !(&...) -> inner
    inner = s[3:-1]

    # Find boundary between left and right child by tracking paren depth
    depth = 0
    seen_open = False
    for i, ch in enumerate(inner):
        if ch in ('(', '['):
            depth += 1
            seen_open = True
        elif ch in (')', ']'):
            depth -= 1
        if seen_open and depth == 0:
            left = inner[:i + 1]
            right = inner[i + 1:]
            # Left: recurse if !(&...), else strip ! to get disjunct
            if left.startswith('!(&'):
                disjuncts = _parse_or_disjuncts(left)
            elif left.startswith('!'):
                disjuncts = [left[1:]]
            else:
                disjuncts = [left]
            # Right: always !<disjunct>
            if right.startswith('!'):
                disjuncts.append(right[1:])
            else:
                disjuncts.append(right)
            return disjuncts

    return [s]  # fallback


def check_or_disintegration(line: ProofLine, chapter: List[ProofLine],
                            state: VerifierState) -> bool:
    """
    OR DISINTEGRATION checker.

    Records that an `_ordis_` (case-split) branch was opened with one
    specific disjunct asserted. Sibling to `or branch proven` but for
    `_ordis_` (consume an existing OR) rather than `_orint_` (produce
    an OR via sub-implications).

    Row layout (exactly two rest fields):
      line.expression = the asserted disjunct
      line.namespace  = parent + `_boundary_ordis_<or>_(<disjunct>)`
      rest[0]         = the compiled OR `(or<N>[…])`
      rest[1]         = parent (the OR's parent scope)

    Validation (D-36, mirrors `check_or_branch_proven` round-2
    structure + the OR-origin check that IS well-founded for `_ordis_`):

      1. `len(rest) == 2` exactly. The tag is in `_ORIGIN_EXEMPT_TAGS`,
         so any extra rest pairs would be silently accepted by the
         generic origin check; reject up front so the row's contract
         stays auditable.
      2. `rest[0]` is a known compiled OR `(or<N>[…])` with ≥2
         disjuncts after `u_i` substitution against the OR's args (and
         matching arity per the binary's `signature`).
      3. `line.expression` is one of those disjuncts (modulo equality
         symmetry).
      4. `line.namespace` is EXACTLY `rest[1] + "_boundary_ordis_" +
         rest[0] + "_(" + <disjunct> + ")"` for `<disjunct>` matching
         `line.expression` (modulo equality symmetry). No substring
         search.
      5. The OR has an independent derivation row at parent scope
         (`rest[1]`). A chapter row exists with `expression == rest[0]`,
         `namespace == rest[1]`, and `tag != "or disintegration"` —
         i.e. the OR was actually derived (via `implication`,
         `expansion`, `theorem`, …) before being case-split. This check
         IS well-founded for `_ordis_`: case-split CONSUMES an existing
         OR, so the OR must be derived first. (The analogous check on
         `check_or_branch_proven` was dropped per D-36 because `_orint_`
         PRODUCES an OR — no separate derivation exists by design.)

    Pre-D-36 the checker accepted only the expanded `!(&!(…))` form in
    `rest[0]` and never validated namespace structure or OR-origin.
    """
    if len(line.rest) != 2:
        return False
    asserted = line.expression
    or_expr = line.rest[0]
    parent_ns = line.rest[1]
    branch_ns = line.namespace

    disjuncts = _or_disjuncts_from_compiled(or_expr, state.binaries_for_chapter())
    if disjuncts is None:
        return False
    if not _disjunct_matches(asserted, disjuncts):
        return False

    # OR has an independent derivation at parent scope (well-founded
    # for `_ordis_` — case-split consumes an existing OR).
    or_origin_exists = any(
        ch_line.expression == or_expr
        and ch_line.namespace == parent_ns
        and ch_line.tag != "or disintegration"
        for ch_line in chapter
    )
    if not or_origin_exists:
        return False

    candidates = [asserted]
    if asserted.startswith('(=[') and asserted.endswith('])'):
        eq_args = asserted[3:-2].split(',')
        if len(eq_args) == 2:
            candidates.append('(=[' + eq_args[1] + ',' + eq_args[0] + '])')

    for variant in candidates:
        expected = (parent_ns
                    + "_boundary_ordis_"
                    + or_expr
                    + "_(" + variant + ")")
        if branch_ns == expected:
            return True
    return False


def check_or_convergence(line: ProofLine, chapter: List[ProofLine],
                         state: VerifierState) -> bool:
    """
    OR CONVERGENCE checker — validates the post-clean-fail spec for the
    new producer-side row layout (user-directive, 2026-05-03 thread).

    Row layout (compiled-form OR only).
        <C>  <parent>  or convergence  <OR>  <parent>
                                              <C>  <branch_D1>
                                              <C>  <branch_D2>
                                              …
                                              <C>  <branch_DK>

    Where K = number of disjuncts of the OR. Concretely:
      - line.expression  = C (the converged conclusion)
      - line.namespace   = parent (the OR's parent scope)
      - rest[0]          = OR (compiled (or<N>[…]))
      - rest[1]          = parent
      - rest[2*i + 2]    = C (must equal line.expression for every i)
      - rest[2*i + 3]    = branch_Di (the i-th branch's namespace)

    Validation contract. For the row to PASS, all of:

      1. Layout: len(rest) == 2 + 2*K; len(rest) is even; len(rest) >= 6
         (so K >= 2).
      2. Parent-scope match: line.namespace == rest[1].
      3. OR is real: rest[0] is a compiled (or<N>[…]) with a known
         GL-binary entry of category "or" and >= 2 elements; the
         disjunct count K matches the number of (C, branch_Di) pairs
         in the rest fields.
      4. Conclusion repetition: rest[2*i + 2] == line.expression for
         every i in [0, K).
      5. Branch-scope ancestry: each branch_Di is a strict descendant
         of parent (branch_Di.startswith(parent + "_boundary_")).
      6. Branch distinctness: the K branch_Di values are pairwise
         distinct (no branch cited twice).
      7. Per-branch derivation evidence: for every (C, branch_Di) pair
         in rest, a chapter row exists with expression == C and
         namespace == branch_Di under any tag — proves C was derived
         at that branch scope. This is the "each ingredient has its
         own line" check (user-directive).

    Status as of D-35 (clean-fail) → spec'd-layout (this revision).
    The producer side does not yet emit this layout. The
    currently-emitted convergence rows in chapter `1209_direct_proof.txt`
    use the old 4-field layout (C, parent, or convergence, OR, parent)
    and continue to fail at the layout check (step 1: len(rest) == 4,
    not >= 6). The deliberate-fail outcome is preserved; the failure
    message changes from "unconditional" to "layout mismatch" once the
    producer-side fix lands and the new layout shows up.

    Re-enabling cleanly. Two coordinated producer-side changes must
    land together (the "buildstack + history tracking" follow-on task):
      a. Prover emits the new convergence row layout above.
      b. process_proof_graphs.py retains per-branch derivations of C
         in chapter export (so the step-7 chapter-row lookup succeeds).
    Either change without the other leaves the verifier failing.
    """
    # 1. Layout shape
    if len(line.rest) < 6 or (len(line.rest) % 2) != 0:
        return False

    or_expr = line.rest[0]
    parent_ns = line.rest[1]

    # 2. Parent-scope match
    if line.namespace != parent_ns:
        return False

    # 3. OR is real; count disjuncts via GL-binary lookup
    disjuncts = _or_disjuncts_from_compiled(
        or_expr, state.binaries_for_chapter())
    if disjuncts is None or len(disjuncts) < 2:
        return False
    k = len(disjuncts)
    if len(line.rest) != 2 + 2 * k:
        return False

    # 4 + 5 + 6: per-pair structural checks
    branch_nss: List[str] = []
    conclusion = line.expression
    for i in range(k):
        c_field = line.rest[2 + 2 * i]
        b_field = line.rest[2 + 2 * i + 1]
        # 4. Conclusion repetition
        if c_field != conclusion:
            return False
        # 5. Branch-scope ancestry
        if not b_field.startswith(parent_ns + "_boundary_"):
            return False
        branch_nss.append(b_field)
    # 6. Branch distinctness
    if len(set(branch_nss)) != k:
        return False

    # 7. Per-branch derivation evidence (each ingredient has its own line)
    for branch_ns in branch_nss:
        if not any(ch_line.expression == conclusion
                   and ch_line.namespace == branch_ns
                   for ch_line in chapter):
            return False

    return True


def _or_disjuncts_from_compiled(
        or_expr: str,
        binaries: List[dict]) -> Optional[List[str]]:
    """
    Given a compiled OR expression `(or<N>[arg1,arg2,…])`, look it up in the
    supplied list of GL-binary dicts (each binary is `{name → entry}`) and
    return its disjuncts after substituting `u_i` placeholders with the OR's
    args in order.

    Returns None if any of:
      - the input does not match the `(or<N>[…])` shape;
      - no loaded binary contains an entry for `or<N>`;
      - the entry's `category` is not `"or"`;
      - the entry has fewer than 2 elements;
      - the OR's argument count does not match the binary's `arity`
        (or, when `arity` is absent, the placeholder count parsed from
        `signature`). A mismatched arity is treated as a malformed
        OR expression and rejected.
    """
    m = re.match(r'^\(or(\d+)\[([^\]]*)\]\)$', or_expr)
    if not m:
        return None
    core = f'or{m.group(1)}'
    raw = m.group(2)
    args = raw.split(',') if raw else []
    for binary in binaries:
        entry = binary.get(core)
        if entry is None:
            continue
        if entry.get('category') != 'or':
            continue
        # Reject mismatched arity. Prefer the explicit `arity` field; fall
        # back to counting placeholders in `signature` if `arity` is absent.
        expected = entry.get('arity')
        if expected is None:
            sig = entry.get('signature', '')
            sig_match = re.match(r'^\(or\d+\[([^\]]*)\]\)$', sig)
            if sig_match:
                sig_args = sig_match.group(1)
                expected = len(sig_args.split(',')) if sig_args else 0
        if expected is not None and len(args) != expected:
            continue
        elements = entry.get('elements', [])
        if len(elements) < 2:
            return None
        subbed: List[str] = []
        for elem in elements:
            substituted = elem
            for i, a in enumerate(args, start=1):
                substituted = _replace_arg_safe(substituted, f'u_{i}', a)
            subbed.append(substituted)
        return subbed
    return None


def _disjunct_matches(candidate: str, disjuncts: List[str]) -> bool:
    """
    True iff `candidate` matches one of `disjuncts` directly OR via equality
    symmetry: an equality `(=[a,b])` is treated as the same disjunct as
    `(=[b,a])`. Other expression cores require literal equality.
    """
    if candidate in disjuncts:
        return True
    if candidate.startswith('(=[') and candidate.endswith('])'):
        args = candidate[3:-2].split(',')
        if len(args) == 2:
            swapped = '(=[' + args[1] + ',' + args[0] + '])'
            if swapped in disjuncts:
                return True
    return False


def check_or_branch_proven(line: ProofLine, chapter: List[ProofLine],
                           state: VerifierState) -> bool:
    """
    OR BRANCH PROVEN checker (D-35).

    Emitted by the prover (`prover.cpp` `or branch proven` site, see
    `docs/20_core_concepts/07_or_branching.md`) when an OR is case-split and
    a per-branch scope is opened with one disjunct asserted.

    Row layout (exactly two rest fields):
      line.expression = the compiled OR `(or<N>[…])`
      line.namespace  = parent (the OR's parent scope)
      rest[0]         = the asserted disjunct
      rest[1]         = parent + `_boundary_orint_<or>_(<disjunct>)`

    Validation:
      1. `len(rest) == 2` exactly. The tag is in `_ORIGIN_EXEMPT_TAGS`,
         so any extra rest pairs would be silently accepted by the
         generic origin check; reject them up front so the row's
         contract stays auditable.
      2. `line.expression` is a known compiled OR with ≥2 disjuncts
         after `u_i` substitution against the OR's args (and matching
         arity per the binary's `signature`).
      3. `rest[0]` is one of those disjuncts (modulo equality symmetry).
      4. `rest[1]` is EXACTLY `parent + "_boundary_orint_" + or_expr +
         "_(" + <disjunct> + ")"` for `<disjunct>` matching `rest[0]`
         (modulo equality symmetry). No substring search — the row's
         claim is "this immediate-child subproof", not "some descendant
         that contains the substring".

    Note (D-36, 2026-05-03): An earlier Codex round-3 step required a
    non-`or branch proven` derivation row for the OR at parent scope.
    That check was based on a wrong mental model of `_orint_` (treated
    it as case-split with a separately-derived OR). Correct semantics:
    `_orint_` rewrites the OR goal into two sub-implications-to-prove
    (`!A → B`) and (`!B → A`); when one fires, the `or branch proven`
    row IS the OR's derivation by design — no separate derivation row
    exists. The check was dropped per D-36 as a correction (not a
    relaxation).
    """
    if len(line.rest) != 2:
        return False
    or_expr = line.expression
    asserted = line.rest[0]
    branch_ns = line.rest[1]
    parent_ns = line.namespace

    disjuncts = _or_disjuncts_from_compiled(or_expr, state.binaries_for_chapter())
    if disjuncts is None:
        return False
    if not _disjunct_matches(asserted, disjuncts):
        return False

    candidates = [asserted]
    if asserted.startswith('(=[') and asserted.endswith('])'):
        eq_args = asserted[3:-2].split(',')
        if len(eq_args) == 2:
            candidates.append('(=[' + eq_args[1] + ',' + eq_args[0] + '])')

    for variant in candidates:
        expected = (parent_ns
                    + "_boundary_orint_"
                    + or_expr
                    + "_(" + variant + ")")
        if branch_ns == expected:
            return True
    return False


def check_or_branch_assumption(line: ProofLine, chapter: List[ProofLine],
                               state: VerifierState) -> bool:
    """
    OR BRANCH ASSUMPTION checker (D-35).

    Emitted by the prover (`prover.hpp` `or branch assumption` site, see
    `docs/20_core_concepts/07_or_branching.md`) for each "other" disjunct of
    an OR that has been case-split: in the branch where disjunct D_i is
    asserted, the negation `!D_j` of every D_j (j ≠ i) is seeded as a
    branch-local assumption.

    Row layout (exactly two rest fields):
      line.expression = the negated other-disjunct (e.g. `!(=[i0,v5])`)
      line.namespace  = parent + `_boundary_orint_<or>_(<asserted>)`
      rest[0]         = `<or>_integration_goal` (the OR-with-suffix marker)
      rest[1]         = parent (the OR's parent scope)

    Validation:
      1. `len(rest) == 2` exactly. The tag is in `_ORIGIN_EXEMPT_TAGS`,
         so any extra rest pairs would be silently accepted by the
         generic origin check; reject them up front so the row's
         contract stays auditable.
      2. `rest[0]` ends with `_integration_goal`; stripping the suffix
         yields a compiled OR `(or<N>[…])` with a known GL-binary entry
         (and matching arity).
      3. `line.expression` starts with `!`; stripping it yields a
         disjunct of the OR (modulo equality symmetry).
      4. `line.namespace` is EXACTLY `parent + "_boundary_orint_" +
         or_expr + "_(" + <asserted> + ")"` for some disjunct
         `<asserted>` of the OR. No substring search — the row's claim
         is "this immediate-child branch", not "some descendant that
         contains the substring".
      5. The asserted disjunct (extracted from `line.namespace`'s
         payload) is DIFFERENT from the negated one (modulo equality
         symmetry) — the row asserts a disjunct's negation only in
         branches where ANOTHER disjunct is asserted.
      6. **A matching `or branch proven` row exists.** Some chapter row
         with `tag == "or branch proven"`, `expression == or_expr`,
         `namespace == parent_ns`, `len(rest) == 2`, `rest[1] ==
         branch_ns`, and `rest[0]` matching the asserted disjunct
         (modulo equality symmetry). Without this check the assumption
         row could pass structurally even when the corresponding
         case-split was never opened (Codex round-3 finding).
    """
    if len(line.rest) != 2:
        return False

    parent_ns = line.rest[1]
    branch_ns = line.namespace

    suffix = "_integration_goal"
    raw = line.rest[0]
    if not raw.endswith(suffix):
        return False
    or_expr = raw[:-len(suffix)]

    if not line.expression.startswith("!"):
        return False
    negated = line.expression[1:]

    disjuncts = _or_disjuncts_from_compiled(or_expr, state.binaries_for_chapter())
    if disjuncts is None:
        return False
    if not _disjunct_matches(negated, disjuncts):
        return False

    # Strict structural check: branch_ns must be EXACTLY
    # parent_ns + _boundary_orint_<or>_(<disjunct>).
    expected_prefix = (parent_ns
                       + "_boundary_orint_"
                       + or_expr
                       + "_(")
    if not branch_ns.startswith(expected_prefix):
        return False
    after = branch_ns[len(expected_prefix):]
    # Parse one balanced parens group (the asserted disjunct, which itself
    # starts with `(` since GL expressions are paren-wrapped). After that
    # group, exactly one `)` closes the wrapper, and nothing else may
    # follow.
    if not after.startswith('('):
        return False
    depth = 1
    end = -1
    for j in range(1, len(after)):
        if after[j] == '(':
            depth += 1
        elif after[j] == ')':
            depth -= 1
            if depth == 0:
                end = j
                break
    if end < 0:
        return False
    asserted = after[:end + 1]
    if after[end + 1:] != ')':
        return False

    if not _disjunct_matches(asserted, disjuncts):
        return False
    if asserted == negated:
        return False
    if negated.startswith('(=[') and negated.endswith('])'):
        eq_args = negated[3:-2].split(',')
        if len(eq_args) == 2:
            swapped = '(=[' + eq_args[1] + ',' + eq_args[0] + '])'
            if asserted == swapped:
                return False

    # 6. Matching `or branch proven` row must exist (Codex round-3).
    matching_proven = any(
        ch_line.tag == "or branch proven"
        and ch_line.expression == or_expr
        and ch_line.namespace == parent_ns
        and len(ch_line.rest) == 2
        and ch_line.rest[1] == branch_ns
        and _disjunct_matches(ch_line.rest[0], [asserted])
        for ch_line in chapter
    )
    if not matching_proven:
        return False

    return True


def check_vacuous_truth(line: ProofLine, chapter: List[ProofLine],
                        state: VerifierState) -> bool:
    """
    VACUOUS TRUTH checker (format + negation shape only).

    In an induction step, if both expr and negate(expr) exist as derived
    statements, the step is vacuously true (self-contradictory premises).
    rest layout: expr, ns, negate(expr), ns, lb_exprKey, ns

    Traceability of at least one ingredient back to lb_exprKey is checked
    separately in verify_chapter (counter "vacuous truth trace"), matching
    the pattern used by "contradiction trace".
    """
    if line.namespace != "main":
        return False
    if len(line.rest) < 6:
        return False

    expr = line.rest[0]
    ns1 = line.rest[1]
    neg_expr = line.rest[2]
    ns2 = line.rest[3]
    ns3 = line.rest[5]

    if ns1 != "main" or ns2 != "main" or ns3 != "main":
        return False

    # Negation shape
    if neg_expr == "!" + expr:
        return True
    if expr == "!" + neg_expr:
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
    "symmetry of inequality":           check_symmetry_of_inequality,
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
    "variable copy":                    check_variable_copy,
    "externally provided theorem":      check_externally_provided_theorem,
    "incubator back reformulation":     check_incubator_back_reformulation,
    "equalize variable":                check_equalize_variable,
    "multiplied from":                  check_equalize_variable,
    "contradiction":                    check_contradiction,
    "or disintegration":                check_or_disintegration,
    "or convergence":                   check_or_convergence,
    "or branch proven":                 check_or_branch_proven,
    "or branch assumption":             check_or_branch_assumption,
    "vacuous truth":                    check_vacuous_truth,
    "or theorem":                       check_or_theorem,
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

    # Detect which GL binary applies to this chapter from the theorem's anchor
    state.current_gl_binary = None
    state.current_resolved_defsets = None
    if chapter_thm is not None:
        thm_expr = chapter_thm[0]
        for tag, binary in state.gl_binaries.items():
            anchor_name = f'Anchor{tag}'
            if anchor_name in thm_expr:
                state.current_gl_binary = binary
                state.current_resolved_defsets = state.resolved_defsets_per_tag.get(tag)
                break
    if state.current_resolved_defsets is None:
        state.current_resolved_defsets = state.resolved_defsets_atomic_only

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

    # 5b. Vacuous truth trace: at least one of the two contradicting
    #     ingredients must trace back to the LB's exprKey (3rd ingredient
    #     of the vacuous-truth origin line). This is the mirror of the
    #     contradiction-trace check: it proves the contradiction is rooted
    #     in the LB's own work, not in two inherited anchor-level facts.
    for line in lines:
        if line.tag != "vacuous truth":
            continue
        if len(line.rest) < 6:
            state.counter_for("vacuous truth trace").record(False)
            continue

        lb_key = line.rest[4]
        _lb_target = (lambda cl, _k=lb_key: cl.expression == _k)

        ok = (_trace_back_to(line.rest[0], _ch_expr_to_lines, _lb_target)
              or _trace_back_to(line.rest[2], _ch_expr_to_lines, _lb_target))
        state.counter_for("vacuous truth trace").record(ok)

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
        "or disintegration",
        "or convergence",
        "or branch proven",
        "or branch assumption",
        "or theorem",
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
                # Cross-batch refs may carry alpha-equivalent but textually
                # distinct bound-var names (e.g. incubator chapter row uses
                # `i2` where the Peano global list stored `v1`).
                # Compare modulo alpha-renaming of every >[...] bound variable.
                alpha_dep = _alpha_canonicalize_bound_vars(dep)
                registry = state.global_theorems.keys() | state.external_theorems
                found = (
                    dep in state.global_theorems
                    or dep in state.external_theorems
                    or any(_normalize_expr_list([gt]) == norm_dep
                           for gt in registry)
                    or any(_alpha_canonicalize_bound_vars(gt) == alpha_dep
                           for gt in registry)
                )
                state.counter_for("origin").record(found)
                continue
            state.counter_for("origin").record(False)

    # 5. Definition-set consistency meta-check (D-41).
    #    For every chapter row, every variable that appears at multiple
    #    operator-call positions across the row's expressions (left-hand
    #    expression + each rest[i] expression) must connect to ports with
    #    identical type labels per state.resolved_defsets. Pure formal
    #    completeness — failures literally impossible if the producer
    #    (compiler/conjecturer/prover) honoured its own typing rules; if any
    #    failure surfaces, that is a producer bug.
    for line in lines:
        state.counter_for("definition set consistency").record(
            check_defset_consistency(line, state))


# ---------------------------------------------------------------------------
#  Definition-set consistency meta-check (D-41) — helpers
#
#  Faithful Python translation of the C++ reference at
#  GL_Quick_VS/GL_Quick/src/compiler.hpp:1246-1538 (`ArgumentAnalyzer` +
#  `RecursiveParser::parseSubtree` + `mergeMaps` + `processLeaf` +
#  `checkDefinitionConsistency`).
#
#  Algorithm: recursive parse of each MPL expression independently,
#  accumulating per-node `remainingArgsDefs` (a `var → type_label` map).
#  At quantifier nodes (`>[v1,v2]<L><R>`) the bound variables are
#  REMOVED from the merged child map, so their names can be reused at
#  outer scopes without collision. `_merge_maps` is the consistency
#  check: shared variable names across child sub-trees must have
#  identical type labels (else mismatch). Each expression in a chapter
#  row is checked independently — no cross-expression pooling, matching
#  how the compiler invokes `analyze()` once per expression at compile
#  time.
# ---------------------------------------------------------------------------

# Regex matching a leaf operator-call: `(<head>[<args>])` where the bracket
# contents have no nested brackets. Head matches `[\w=]+` so the `=`
# operator (which is a non-word char) is captured alongside word-named
# operators (`in`, `in2`, `in3`, `fXY`, `existence2`, `or0`, `implication26`,
# etc.). Argument tokens may include `*`, `+`, `s`, `id` (anchor slot
# names) — those are word chars and split cleanly on `,`.
_LEAF_OP_CALL_RE = re.compile(r'\(([\w=]+)\[([^\[\]]*)\]\)')

# Used to extract head + args inside a leaf label (no surrounding parens).
_LEAF_LABEL_RE = re.compile(r'([\w=]+)\[([^\[\]]*)\]')


class _DefsetMismatch(Exception):
    """Raised by _merge_maps when two child sub-trees disagree on the type of
    a shared variable, or by _process_leaf when the same variable appears at
    two same-leaf positions with different types. Caught by
    check_defset_consistency to record a per-row failure."""
    pass


def _merge_maps(a: Dict[str, str], b: Dict[str, str]) -> Dict[str, str]:
    """Mirror of `ArgumentAnalyzer::mergeMaps` (compiler.hpp:1250-1264).
    Union the two maps; on shared key, types must match else raise."""
    out = dict(a)
    for k, v in b.items():
        existing = out.get(k)
        if existing is None:
            out[k] = v
        elif existing != v:
            raise _DefsetMismatch
    return out


def _process_leaf(label: str,
                  resolved_defsets: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """Mirror of `RecursiveParser::processLeaf` (compiler.hpp:1409-1426).
    Look up the operator's defsets and emit `var → type_label`. The C++
    silently overwrites on same-arg-twice; we tighten with a mismatch
    raise (catches e.g. `(in[v,v])` style — pos 1 is (1), pos 2 is P(1))."""
    m = _LEAF_LABEL_RE.match(label)
    if not m:
        return {}
    head = m.group(1)
    args = [a for a in m.group(2).split(",") if a]
    ds = resolved_defsets.get(head)
    if ds is None:
        return {}
    out: Dict[str, str] = {}
    for i, arg in enumerate(args, start=1):
        t = ds.get(str(i))
        if t is None:
            continue
        existing = out.get(arg)
        if existing is None:
            out[arg] = t
        elif existing != t:
            raise _DefsetMismatch
    return out


def _parse_subtree(s: str, idx_box: List[int],
                   resolved_defsets: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """Mirror of `RecursiveParser::parseSubtree` (compiler.hpp:1281-1407).
    `idx_box` is a 1-element mutable list carrying the parse cursor. Returns
    the node's `remainingArgsDefs` (variables visible at this node). Bound
    variables introduced at `>[…]` quantifiers are removed from the map
    after merge — matching the C++ scoping rule."""
    idx = idx_box[0]
    n = len(s)
    if idx >= n:
        return {}

    if s[idx] == "(":
        idx += 1
        if idx < n and s[idx] == ">":
            # Implication: (>[bound_list] <left> <right>)
            idx += 1
            close = s.find("]", idx)
            assert close != -1, "Missing ']' in implication"
            bound_str = s[idx + 1:close]
            bound_vars = [v for v in bound_str.split(",") if v]
            idx = close + 1
            idx_box[0] = idx
            left_map = _parse_subtree(s, idx_box, resolved_defsets)
            right_map = _parse_subtree(s, idx_box, resolved_defsets)
            idx = idx_box[0]
            combined = _merge_maps(left_map, right_map)
            for v in bound_vars:
                combined.pop(v, None)
            if idx < n and s[idx] == ")":
                idx += 1
            idx_box[0] = idx
            return combined
        if idx < n and s[idx] == "&":
            idx += 1
            idx_box[0] = idx
            left_map = _parse_subtree(s, idx_box, resolved_defsets)
            right_map = _parse_subtree(s, idx_box, resolved_defsets)
            idx = idx_box[0]
            combined = _merge_maps(left_map, right_map)
            if idx < n and s[idx] == ")":
                idx += 1
            idx_box[0] = idx
            return combined
        if idx < n and s[idx] == "|":
            # OR — treated like conjunction for scoping (no bound vars).
            idx += 1
            idx_box[0] = idx
            left_map = _parse_subtree(s, idx_box, resolved_defsets)
            right_map = _parse_subtree(s, idx_box, resolved_defsets)
            idx = idx_box[0]
            combined = _merge_maps(left_map, right_map)
            if idx < n and s[idx] == ")":
                idx += 1
            idx_box[0] = idx
            return combined
        # Leaf operator call: (<head>[<args>])
        close = s.find(")", idx)
        assert close != -1, "Missing ')' for leaf"
        label = s[idx:close]
        idx = close + 1
        leaf_map = _process_leaf(label, resolved_defsets)
        idx_box[0] = idx
        return leaf_map

    if idx + 1 < n and s[idx] == "!" and s[idx + 1] == "(":
        idx += 2
        if idx < n and s[idx] == ">":
            # !(>[bound_list] <left> <right>)
            idx += 1
            close = s.find("]", idx)
            assert close != -1, "Missing ']' in !>"
            bound_str = s[idx + 1:close]
            bound_vars = [v for v in bound_str.split(",") if v]
            idx = close + 1
            idx_box[0] = idx
            left_map = _parse_subtree(s, idx_box, resolved_defsets)
            right_map = _parse_subtree(s, idx_box, resolved_defsets)
            idx = idx_box[0]
            combined = _merge_maps(left_map, right_map)
            for v in bound_vars:
                combined.pop(v, None)
            if idx < n and s[idx] == ")":
                idx += 1
            idx_box[0] = idx
            return combined
        if idx < n and s[idx] == "&":
            idx += 1
            idx_box[0] = idx
            left_map = _parse_subtree(s, idx_box, resolved_defsets)
            right_map = _parse_subtree(s, idx_box, resolved_defsets)
            idx = idx_box[0]
            combined = _merge_maps(left_map, right_map)
            if idx < n and s[idx] == ")":
                idx += 1
            idx_box[0] = idx
            return combined
        if idx < n and s[idx] == "|":
            idx += 1
            idx_box[0] = idx
            left_map = _parse_subtree(s, idx_box, resolved_defsets)
            right_map = _parse_subtree(s, idx_box, resolved_defsets)
            idx = idx_box[0]
            combined = _merge_maps(left_map, right_map)
            if idx < n and s[idx] == ")":
                idx += 1
            idx_box[0] = idx
            return combined
        # Negated leaf: !(<head>[<args>])
        close = s.find(")", idx)
        assert close != -1, "Missing ')' for negated leaf"
        label = s[idx:close]
        idx = close + 1
        leaf_map = _process_leaf(label, resolved_defsets)
        idx_box[0] = idx
        return leaf_map

    # Defensive — unrecognised shape.
    return {}


def build_resolved_defsets_per_tag(
        definition_sets: Dict[str, Dict[str, list]],
        gl_binaries: Dict[str, dict]
) -> Tuple[Dict[str, Dict[str, Dict[str, str]]], Dict[str, Dict[str, str]]]:
    """Build per-tag resolved defset indices used by check_defset_consistency.

    Mirrors the compiler's `ArgumentAnalyzer(this->coreExpressionMap)`
    pattern at `prover.hpp:3556`: each batch's analyzer is constructed
    with its own coreExpressionMap, so per-batch compact-name allocations
    (e.g. `implication26` is arity 2 in Gauss but arity 5 in IncubatorGauss)
    don't cross-contaminate. The verifier mirrors that by building one
    resolved-defset map per tag, selected per-chapter via anchor-substring
    match (same convention as `current_gl_binary` at verifier.py:3167-3174).

    GL_binary_shared.json (when present, keyed as `shared`) acts as a
    cross-batch fallback inside every tag's resolved map — for ops in
    `_SPONTANEOUS_CATEGORIES` the shared dictionary is the canonical
    cross-batch source. Per-tag entries override shared on collision
    (per-batch is authoritative for its own chapters).

    Inputs:
      - definition_sets: ConfigVisu.json's per-atomic-operator
        `{position_str: [type_label, combinable]}` map.
      - gl_binaries: tag → { op_name → { "category", "elements",
        "signature", "arity", "definedSet" } }.

    Returns:
      - per_tag: { tag → { core_name → { pos_str → type_label_str } } }
      - atomic_only: { core_name → { pos_str → type_label_str } } — used as
        fallback for chapters whose theorem doesn't disclose an anchor.
    """
    atomic_seeds: Dict[str, Dict[str, str]] = {
        core_name: {pos: entry[0] for pos, entry in ds.items()}
        for core_name, ds in definition_sets.items()
    }

    shared_binary = gl_binaries.get("shared", {}) or {}

    per_tag: Dict[str, Dict[str, Dict[str, str]]] = {}
    for tag in gl_binaries.keys():
        # Build a per-tag "operator superset": shared first (so per-tag
        # overrides apply on collision), then this tag's own binary.
        merged_binary: Dict[str, dict] = {}
        for op_name, spec in shared_binary.items():
            merged_binary[op_name] = spec
        if tag != "shared":
            for op_name, spec in gl_binaries[tag].items():
                merged_binary[op_name] = spec  # per-tag wins on collision

        # Resolve composites in merged_binary against atomic_seeds + ones
        # already resolved this pass. Fixed-point iteration.
        resolved: Dict[str, Dict[str, str]] = dict(atomic_seeds)
        composites: Dict[str, dict] = {}
        for op_name, spec in merged_binary.items():
            if op_name in resolved:
                continue  # atomic baseline wins for known atomics
            if not isinstance(spec, dict):
                continue
            elements = spec.get("elements") or []
            if not any(e for e in elements if e):
                continue
            composites[op_name] = spec

        while True:
            progress = False
            for op_name, spec in list(composites.items()):
                if op_name in resolved:
                    composites.pop(op_name, None)
                    continue
                derived = _try_derive_from_elements(spec, resolved)
                if derived is not None:
                    resolved[op_name] = derived
                    composites.pop(op_name, None)
                    progress = True
            if not progress:
                break

        per_tag[tag] = resolved

    return per_tag, atomic_seeds


def _try_derive_from_elements(spec: dict,
                              resolved: Dict[str, Dict[str, str]]
                              ) -> Optional[Dict[str, str]]:
    """Walk a composite spec's elements to derive its signature args' types.

    Returns the per-position type-label dict on success, or None if the spec
    references an inner operator whose own defsets are not yet resolved.
    """
    signature = spec.get("signature", "")
    elements = spec.get("elements") or []

    # Local accumulator: variable name (u_K, integer literal) → type label
    var_types: Dict[str, str] = {}

    for elem in elements:
        if not elem:
            continue
        m = _LEAF_OP_CALL_RE.search(elem)
        if not m:
            continue  # element is not a leaf-call shape; skip
        head = m.group(1)
        args_str = m.group(2)
        args = [a for a in args_str.split(",") if a]
        inner_ds = resolved.get(head)
        if inner_ds is None:
            return None  # inner op not yet resolved; defer
        for i, arg in enumerate(args, start=1):
            t = inner_ds.get(str(i))
            if t is None:
                continue
            existing = var_types.get(arg)
            if existing is not None and existing != t:
                # Internal contradiction in the producer-emitted spec.
                # Surface as unresolved (caller will leave it absent from
                # the output map; the consistency check then skips it).
                return None
            var_types[arg] = t

    # Extract signature args (u-prefixed in normal case) and look them up.
    sig_match = _LEAF_OP_CALL_RE.search(signature)
    if sig_match is None:
        return None
    sig_args = [a for a in sig_match.group(2).split(",") if a]
    out: Dict[str, str] = {}
    for i, sa in enumerate(sig_args, start=1):
        t = var_types.get(sa)
        if t is None:
            return None  # signature arg never appeared in elements
        out[str(i)] = t
    return out


def check_defset_consistency(line: ProofLine, state: VerifierState) -> bool:
    """Per-row defset consistency check, faithful to the C++ compiler's
    reference algorithm at compiler.hpp:1246-1538.

    Each expression in the row (left-hand expression + each rest[i] expr at
    even indices) is parsed independently with `_parse_subtree`, which
    accumulates per-node `remainingArgsDefs` and removes bound variables at
    `>[…]` quantifiers. `_merge_maps` is the actual mismatch detector — a
    `_DefsetMismatch` raised anywhere during parse is caught and returns
    False. Returns True if every expression parses without raising.

    Uses `state.current_resolved_defsets` (set per-chapter by `verify_chapter`
    based on the chapter's tag) — falls back to the atomic-only map when no
    tag is selected (chapters whose theorem doesn't disclose an anchor).
    """
    resolved = state.current_resolved_defsets
    if resolved is None:
        resolved = state.resolved_defsets_atomic_only

    exprs: List[str] = [line.expression]
    for i in range(0, len(line.rest), 2):
        exprs.append(line.rest[i])

    for expr in exprs:
        if not expr:
            continue
        s = "".join(c for c in expr if c not in " \t\n\r")
        if not s:
            continue
        try:
            _parse_subtree(s, [0], resolved)
        except _DefsetMismatch:
            return False
        except (AssertionError, IndexError):
            # Parse-shape anomaly — treat as non-failure (the producer-side
            # checks would have already caught syntactic problems). Surface
            # via the unknown-op skip convention.
            continue
    return True


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


def run_verifier(base_dir: str,
                 extra_global_lists: Optional[List[str]] = None) -> VerifierState:
    state = VerifierState()

    # Load global theorem list
    state.global_theorems, state.global_theorem_list = \
        load_global_theorem_list(base_dir)

    # Union sibling-batch theorem lists into globals (cross-batch references
    # such as Peano axioms cited from incubator chapters). Local entries take
    # precedence on key collisions; the ordered list is appended in the order
    # the extra paths are supplied.
    if extra_global_lists:
        for extra_path in extra_global_lists:
            extra_dir = os.path.dirname(os.path.abspath(extra_path))
            extra_thms, extra_ordered = load_global_theorem_list(extra_dir)
            for expr, info in extra_thms.items():
                if expr not in state.global_theorems:
                    state.global_theorems[expr] = info
            for entry in extra_ordered:
                if entry[0] not in {e[0] for e in state.global_theorem_list}:
                    state.global_theorem_list.append(entry)

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

    # Per-tag defset indices — atomic ops from definition_sets, composite ops
    # resolved iteratively from each batch's gl_binary.elements (D-41).
    # Mirrors the compiler's per-batch ArgumentAnalyzer construction at
    # prover.hpp:3556. Selected per-chapter in verify_chapter the same way
    # current_gl_binary is.
    state.resolved_defsets_per_tag, state.resolved_defsets_atomic_only = \
        build_resolved_defsets_per_tag(state.definition_sets, state.gl_binaries)

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
    default_base = os.path.join(script_dir, "files", "processed_proof_graph")

    parser = argparse.ArgumentParser(
        description="Verify a GL processed proof graph.")
    parser.add_argument(
        "base_dir",
        nargs="?",
        default=default_base,
        help=("Directory containing chapter files + global_theorem_list.txt. "
              f"Default: {default_base}"))
    parser.add_argument(
        "--include-globals",
        action="append",
        default=[],
        metavar="PATH",
        help=("Additional global_theorem_list.txt to union into the "
              "theorem registry (for cross-batch references, e.g. "
              "incubator chapters citing Peano axioms). Repeatable."))
    args = parser.parse_args()

    if not os.path.isdir(args.base_dir):
        print(f"ERROR: directory not found: {args.base_dir}", file=sys.stderr)
        sys.exit(1)

    state = run_verifier(args.base_dir, extra_global_lists=args.include_globals)
    print_report(state)


if __name__ == "__main__":
    main()
