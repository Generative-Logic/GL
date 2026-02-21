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

# !/usr/bin/env python3
"""
generate_full_proof_graph.py

Generates a set of HTML proof pages:
- An index page with a Table of Contents
- One HTML file per theorem (with navigation links)

Each theorem tuple is now (theorem_name:str, method:str, var_name:str).
Default output directory: full_proof_graph
"""
import copy
import os
import html

import re
import shutil

import create_expressions
from typing import Dict

from configuration_reader import configuration_reader
from parameters import debug

import visu_helpers
from visu_helpers import format_mirroring, expand_expr
from pathlib import Path

# wherever this file lives, assume the project root is its parent folder
PROJECT_ROOT = Path(__file__).resolve().parent

# global mapping from displayed theorem title → its chapter file
theorem_to_file = {}
# alpha-normalized theorem shape -> file (only unique matches)
theorem_shape_to_file = {}


def _alpha_normalize_theorem_expr(expr: str) -> str:
    """Normalize theorem expressions up to variable renaming (alpha-equivalence).
    We replace every bracket-argument token with a stable placeholder by first occurrence.
    This lets instantiated theorem schemas (e.g. broadcasted versions) match the generic theorem page.
    """
    if not expr or not isinstance(expr, str):
        return expr
    s = expr.strip()
    if not (s.startswith('(>') or s.startswith('!(')):
        return s

    token_pattern = re.compile(r'(?<=[\[,])([^,\[\]]+)(?=[\],])')
    mapping = {}
    next_idx = 0

    def repl(m):
        nonlocal next_idx
        tok = m.group(1)
        base = _strip_u_prefixes(tok)
        if base not in mapping:
            mapping[base] = f"@{next_idx}"
            next_idx += 1
        return mapping[base]

    return token_pattern.sub(repl, s)


def _resolve_theorem_target(expr: str):
    target = theorem_to_file.get(expr)
    if target:
        return target
    shape = _alpha_normalize_theorem_expr(expr)
    return theorem_shape_to_file.get(shape)


# Utility function to wrap clickable substrings starting with '(' or '!(' and ending at space or end-of-string
def wrap_clickable(text):
    def repl(m):
        s = m.group(0)  # the raw token, e.g. "(v7*v8)=(v8*v7)"
        esc = html.escape(s, quote=True)
        target = _resolve_theorem_target(s)
        # span for the normal “expand on right-click”
        span = f'<span class="clickable" data-text="{esc}">{esc}</span>'
        # if it exactly matches one of our theorems, link it
        return f'<a href="{target}" class="theorem-link">{span}</a>' if target else span

    pattern = r"!?\([^ ]*?\)(?= |$)"
    return re.sub(pattern, repl, text)


def _format_validity_tag(validity: str) -> str:
    if not validity:
        return ""
    raw = validity.strip()
    esc = html.escape(raw)
    if raw.startswith("(") and raw.endswith(")"):
        return f' <span class="validity-tag">{esc}</span>'
    return f' <span class="validity-tag">({esc})</span>'


def format_stack_entries(stack, prefix='', cursor_index=None, reverse_entries=True, external_anchor_map=None, goal_key_norm=None):
    """
    Convert a proof-stack (list of [key, validity, explanation, ing1, val1, ...]) into HTML.

    New Format Structure:
    0: Result Expression (Key)
    1: Result Validity
    2: Explanation (Method/Justification)
    3, 5, 7...: Ingredient Expressions
    4, 6, 8...: Ingredient Validities
    """
    # Reverse so the earliest step is first in the output (legacy behavior)
    rev = list(stack)[::-1] if reverse_entries else list(stack)

    # Map each key (normalized) to a unique anchor ID
    key_map = {}
    external_anchor_map = external_anchor_map or {}
    for idx, entry in enumerate(rev):
        if not entry:
            continue
        key = entry[0]
        norm = re.sub(r'\s+', '', key).lower()
        key_map[norm] = f"{prefix}-entry{idx}" if prefix else f"entry{idx}"

    lines = []
    total = len(rev)

    highlight_idx = None
    if goal_key_norm:
        for i, entry in enumerate(rev):
            if entry and _norm_expr(entry[0]) == goal_key_norm:
                highlight_idx = i
                break
    else:
        if total > 0:
            highlight_idx = total - 1

    for idx, entry in enumerate(rev):
        if 'theorem' in entry:
            continue
        if not entry:
            continue

        key_expr = entry[0]
        key_validity = entry[1] if len(entry) > 1 else ""
        explanation = entry[2] if len(entry) > 2 else ""

        norm_key = re.sub(r'\s+', '', key_expr).lower()
        anchor = key_map.get(norm_key, "")

        if highlight_idx is not None and idx == highlight_idx:
            first, *rest = key_expr.split(' ', 1)
            first_html = wrap_clickable(first)

            if first_html.startswith('<a '):
                first_html = first_html.replace(
                    '<span class="clickable"',
                    '<span class="clickable" style="color:red !important; font-weight:bold !important;"',
                    1
                )
            else:
                first_html = f'<span class="clickable" style="color:red; font-weight:bold">{first_html}</span>'

            if rest:
                rest_html = wrap_clickable(rest[0])
                key_html = f"{first_html} {rest_html}"
            else:
                key_html = first_html
        else:
            key_html = wrap_clickable(key_expr)

        if key_validity:
            key_html += _format_validity_tag(key_validity)

        parts = [f"<span id='{anchor}'>{key_html}</span>"]

        if explanation:
            parts.append(f"<b>{html.escape(explanation)}</b>")

        if explanation == "validity name":
            if len(entry) > 3:
                rhs_html = wrap_clickable(entry[3])
                if len(entry) > 4 and entry[4]:
                    rhs_html += _format_validity_tag(entry[4])
                parts.append(rhs_html)

            line_html = "&nbsp;&nbsp;".join(parts)
            if cursor_index is not None and idx == cursor_index:
                line_html = (
                    f"<div style='background-color:#fffa8b; padding:4px; "
                    f"border-radius:4px'>{line_html}</div>"
                )
            lines.append(line_html)
            continue

        for i in range(3, len(entry), 2):
            ing_expr = entry[i]
            ing_val = entry[i + 1] if i + 1 < len(entry) else ""

            norm_ref = re.sub(r'\s+', '', ing_expr).lower()
            ref_html = wrap_clickable(ing_expr)
            # --- NEW: Highlight integration target ---
            is_integration_target = (explanation == "expansion for integration" and i == 3)
            if is_integration_target:
                # 1. Strip away the clickable <span> and data attributes to get plain text
                clean_text = re.sub(r'<[^>]+>', '', ref_html)

                # 2. Re-wrap it in a single, perfectly unclickable magenta span with quotes
                ref_html = f'<span style="color:magenta !important; font-weight:bold !important;">"{clean_text}"</span>'
            # -----------------------------------------

            if norm_ref in key_map:
                linked_ref = f"<a href='#{key_map[norm_ref]}' style='text-decoration:none'>{ref_html}</a>"
            elif norm_ref in external_anchor_map:
                linked_ref = f"<a href='#{external_anchor_map[norm_ref]}' style='text-decoration:none'>{ref_html}</a>"
            else:
                linked_ref = ref_html

            if ing_val:
                linked_ref += _format_validity_tag(ing_val)

            parts.append(linked_ref)

        line_html = "&nbsp;&nbsp;".join(parts)
        if cursor_index is not None and idx == cursor_index:
            line_html = (
                f"<div style='background-color:#fffa8b; padding:4px; "
                f"border-radius:4px'>{line_html}</div>"
            )

        lines.append(line_html)

        if len(entry) > 2 and entry[2] == 'implication':
            if len(entry) > 3:
                helper_list = [entry[0], "implication", entry[3]]
                for k in range(5, len(entry), 2):
                    helper_list.append(entry[k])

                impl_text = visu_helpers.format_implication(helper_list)
                if impl_text:
                    impl_html = (
                        f"<div class='implication' style='margin-left:20px; color:#888888; "
                        f"font-weight:bold; font-size:1.3em;'>{html.escape(impl_text)}</div>"
                    )
                    lines.append(impl_html)

        if len(entry) > 2 and entry[2] == 'mirrored from':
            if len(entry) > 3:
                helper_list = [entry[0], "mirrored from", entry[3]]
                mirrored_text = format_mirroring(helper_list)
                mirrored_html = (
                    f"<div class='mirrored' style='margin-left:20px; color:#888888; "
                    f"font-weight:bold; font-size:1.3em;'>{html.escape(mirrored_text)}</div>"
                )
                lines.append(mirrored_html)

    return "<br/><br/><br/>".join(lines)

def extract_args(s: str) -> list[str]:
    # same pattern as before
    pattern = r'(?<=[\[,])([^,\[\]]+)(?=[\],])'
    all_subs = re.findall(pattern, s)
    # remove duplicates while preserving order
    return list(dict.fromkeys(all_subs))



def _strip_u_prefixes(token: str) -> str:
    out = token
    while out.startswith("u_"):
        out = out[2:]
    return out


def _looks_like_internal_var_token(token: str) -> bool:
    """
    Internal proof-engine variable-ish names we want to rename to v<number>.
    We intentionally do NOT rename theorem/operator symbols like implication26, in2, and0, id, s, etc.
    """
    if not token:
        return False

    base = _strip_u_prefixes(token)

    if base.isdigit():
        return True
    if re.fullmatch(r"x\d+", base):
        return True
    if re.fullmatch(r"(?:int|repl)_lev_\d+_\d+", base):
        return True
    if re.fullmatch(r"it_\d+_lev_\d+_\d+", base):
        return True
    if re.fullmatch(r"it_lev_\d+_\d+", base):
        return True
    if base == "rec":
        return True

    return False


def _normalize_local_expr_vars(expr: str):
    """
    Lightweight local normalization:
    Leaves u_ variables COMPLETELY ALONE (u_30 stays u_30).
    Only normalizes raw non-prefixed variables (7 -> v7, x7 -> vx7).
    """
    args = extract_args(expr)
    replacement_map = {}

    for arg in args:
        if arg.startswith("u_"):
            continue  # Do not touch u_ variables at all

        if arg.isdigit():
            replacement_map[arg] = "v" + arg
        elif arg.startswith("x") and arg[1:].isdigit():
            replacement_map[arg] = "v" + arg

    return replace_keys_in_string(expr, replacement_map)


def rename_expr_peano(expr: str):
    return _normalize_local_expr_vars(expr)


def rename_expr_gauss(expr: str):
    return _normalize_local_expr_vars(expr)


def rename_expr(expr: str):
    # generic fallback
    return _normalize_local_expr_vars(expr)


def disintegrate_implication(expr_for_desintegration, chain):
    head = ""

    root = create_expressions.parse_expr(expr_for_desintegration)

    node = root
    while True:
        if node is not None:
            if node.value[0] == ">":
                chain.append((create_expressions.tree_to_expr(node.left), create_expressions.get_args(node.value),
                              node.left.arguments))
                node = node.right
            else:
                head = create_expressions.tree_to_expr(node)
                break
        else:
            break

    return head


# Global cache to store compiled regex patterns keyed by a sorted tuple of keys.
_regex_cache = {}


def _get_compiled_regex(keys) -> re.Pattern:
    # Create a key for caching: a sorted tuple of keys ensures consistency.
    key_tuple = tuple(sorted(keys))
    if key_tuple not in _regex_cache:
        # Escape all keys to handle special regex characters.
        escaped_keys = [re.escape(key) for key in key_tuple]
        # Build the regex pattern with lookbehind and lookahead for context.
        pattern = r'(?<=[\[,])(' + '|'.join(escaped_keys) + r')(?=[\],])'
        _regex_cache[key_tuple] = re.compile(pattern)
    return _regex_cache[key_tuple]


def replace_keys_in_string(big_string: str, replacement_map: Dict[str, str]) -> str:
    """
    Replaces keys in big_string based on replacement_map, but only if they occur in a context where they are
    immediately preceded by '[' or ',' and immediately followed by ']' or ','.

    Args:
        big_string (str): The original string containing keys to be replaced.
        replacement_map (Dict[str, str]): A dictionary mapping keys to their replacement values.

    Returns:
        str: The modified string with specified keys replaced.
    """
    if not replacement_map:
        return big_string  # No replacements needed

    # Get the precompiled regex from the cache.
    regex = _get_compiled_regex(replacement_map.keys())

    # Use a lambda for direct substitution.
    return regex.sub(lambda m: replacement_map.get(m.group(1), m.group(1)), big_string)


def extract_natural_numbers_expression(expression: str) -> str:
    """
    Extracts the substring starting at "(NaturalNumbers[" up to and including
    the first ']' that follows. Returns "" if not found or malformed.
    """
    needle = "(NaturalNumbers["
    start = expression.find(needle)
    if start == -1:
        return ""
    end = expression.find("]", start + len(needle))
    if end == -1:
        return ""
    return expression[start:end + 1]


def find_one_arg_name(zero_arg: str, s_arg: str, expr: str) -> str:
    """
    Find the middle argument name in a pattern like:
        (in2[<zero_arg>, <NAME>, <s_arg>])
    where NAME is [A-Za-z0-9_]+. Returns the first match or "" if none.
    """
    pattern = (
            r"\(in2\[\s*"
            + re.escape(zero_arg)
            + r"\s*,\s*([A-Za-z0-9_]+)\s*,\s*"
            + re.escape(s_arg)
            + r"\s*\]\)"
    )
    m = re.search(pattern, expr)
    return m.group(1) if m else ""


def find_identity_arg_name(n_arg: str, expr: str) -> str:
    """Extract id from (identity[<N>, <id>])."""
    pattern = (
        r"\(identity\[\s*"
        + re.escape(n_arg)
        + r"\s*,\s*([A-Za-z0-9_]+)\s*\]\)"
    )
    m = re.search(pattern, expr)
    return m.group(1) if m else ""


def _make_prefixed_token(token: str, mode: str) -> str:
    if not token:
        return token
    if mode == 'as_is':
        return token
    if mode == 'vprefix':
        return token if token.startswith('v') else f"v{token}"
    raise ValueError(f"Unknown prefix mode: {mode}")



def infer_anchor_kind_from_expr(expr: str) -> str:
    if not expr:
        return ""
    if "(AnchorGauss[" in expr:
        return "gauss"
    if "(AnchorPeano[" in expr:
        return "peano"
    return ""


def infer_anchor_kind_from_theorem(theorem_expr: str) -> str:
    kind = infer_anchor_kind_from_expr(theorem_expr)
    if kind:
        return kind

    temp_chain = []
    try:
        disintegrate_implication(theorem_expr, temp_chain)
    except Exception:
        return ""
    for element in temp_chain:
        k = infer_anchor_kind_from_expr(element[0])
        if k:
            return k
    return ""


def build_anchor_symbol_replacement_map(anchor_expr: str, prefix_mode: str = 'as_is', anchor_kind: str = 'auto') -> dict[str, str]:
    """Build replacement map for AnchorPeano / AnchorGauss symbols."""
    replacement_map: dict[str, str] = {}

    detected_kind = infer_anchor_kind_from_expr(anchor_expr)
    if anchor_kind == 'auto':
        anchor_kind = detected_kind

    if anchor_kind not in ('peano', 'gauss'):
        return replacement_map

    if not (anchor_expr.startswith('(AnchorPeano[') or anchor_expr.startswith('(AnchorGauss[')):
        return replacement_map

    expanded_anchor = expand_expr(anchor_expr)
    nn_expr = extract_natural_numbers_expression(expanded_anchor)
    if not nn_expr:
        return replacement_map

    args = create_expressions.get_args(nn_expr)
    if len(args) < 5:
        return replacement_map

    def put_symbol(token: str, symbol: str, include_vx_alias: bool = False):
        if not token:
            return
        key = _make_prefixed_token(token, prefix_mode)
        replacement_map[key] = symbol
        if include_vx_alias and key.startswith('v') and len(key) > 1:
            replacement_map['vx' + key[1:]] = symbol

    # NaturalNumbers[N,i0,s,+,*]
    put_symbol(args[0], 'N')
    put_symbol(args[1], '0', include_vx_alias=True)
    put_symbol(args[2], 's')
    put_symbol(args[3], '+')
    put_symbol(args[4], '*')

    zero_arg_name = args[1]
    s_arg_name = args[2]
    one_arg_name = find_one_arg_name(zero_arg_name, s_arg_name, expanded_anchor)
    put_symbol(one_arg_name, '1', include_vx_alias=True)

    if anchor_kind == 'gauss':
        two_arg_name = find_one_arg_name(one_arg_name, s_arg_name, expanded_anchor) if one_arg_name else ''
        put_symbol(two_arg_name, '2', include_vx_alias=True)

        id_arg_name = find_identity_arg_name(args[0], expanded_anchor)
        put_symbol(id_arg_name, 'id')

    return replacement_map


def _extract_anchor_subexpressions(expr: str) -> list[str]:
    """Return all nested (AnchorPeano[...]) / (AnchorGauss[...]) subexpressions found in expr."""
    if not expr:
        return []

    starts = ["(AnchorPeano[", "(AnchorGauss["]
    out = []
    i = 0
    n = len(expr)
    while i < n:
        start = -1
        start_token = ""
        for token in starts:
            j = expr.find(token, i)
            if j != -1 and (start == -1 or j < start):
                start = j
                start_token = token
        if start == -1:
            break

        # Scan until the matching ')' of the anchor expression.
        bracket_depth = 0
        k = start
        end = -1
        while k < n:
            ch = expr[k]
            if ch == '[':
                bracket_depth += 1
            elif ch == ']':
                bracket_depth = max(0, bracket_depth - 1)
            elif ch == ')' and bracket_depth == 0:
                end = k
                break
            k += 1

        if end != -1:
            out.append(expr[start:end + 1])
            i = end + 1
        else:
            # malformed; avoid infinite loop
            i = start + len(start_token)

    # preserve order, remove duplicates
    return list(dict.fromkeys(out))


def _contains_u_prefixed_token(expr: str) -> bool:
    for tok in extract_args(expr):
        if tok.startswith('u_'):
            return True
    return False


def _extract_arg_tokens_in_order(s: str) -> list[str]:
    """Return all bracket/comma-delimited argument tokens in order (with duplicates)."""
    return re.findall(r'(?<=[\[,])([^,\[\]]+)(?=[\],])', s)


def _build_local_w_replacement_map_for_unanchored_implication(norm_expr: str, raw_expr: str) -> dict[str, str]:
    """
    Only renames the standard v<number> variables to w<number>.
    Leaves u_ variables entirely untouched.
    """
    norm_tokens = _extract_arg_tokens_in_order(norm_expr)

    if not norm_tokens:
        return {}

    has_u = any(tok.startswith('u_') for tok in norm_tokens)
    if not has_u:
        return {}

    replacement_map: dict[str, str] = {}
    next_num = 1

    for tok in norm_tokens:
        if not tok.startswith('u_') and (re.fullmatch(r"v\d+", tok) or re.fullmatch(r"vx\d+", tok)):
            if tok not in replacement_map:
                replacement_map[tok] = f"w{next_num}"
                next_num += 1

    return replacement_map


def _collect_anchor_maps_from_expr(expr: str, prefix_mode: str, default_kind: str = "") -> dict[str, str]:
    out = {}
    if not expr:
        return out

    for anchor_expr in _extract_anchor_subexpressions(expr):
        k = infer_anchor_kind_from_expr(anchor_expr) or default_kind
        out.update(build_anchor_symbol_replacement_map(anchor_expr, prefix_mode=prefix_mode, anchor_kind=k or 'auto'))

    return out


def _infer_row_anchor_kind(row: list[str], default_kind: str = "") -> str:
    if not row:
        return default_kind
    expr_indices = [0] + list(range(3, len(row), 2))
    for idx in expr_indices:
        if idx < len(row):
            k = infer_anchor_kind_from_expr(row[idx])
            if k:
                return k
    return default_kind



def rename_theorem(theorem: str):
    """
    Renaming is now fully handled by process_proof_graphs.py.
    Just return the theorem exactly as it is.
    """
    return theorem


def clean_stack(stack: list[list[str]]):
    """
    Cleans a stack of strings by normalizing internal variable keys.
    Leaves u_ variables completely untouched.
    """
    # Avoid mutating while iterating
    stack[:] = [entry for entry in stack if not (len(entry) > 2 and entry[2] == "anchor handling")]

    token_pattern = re.compile(r'(?<=[\[,])([^,\[\]]+)(?=[\],])')

    def expr_cols(row):
        return [0] + list(range(3, len(row), 2))

    def validity_cols(row):
        cols = []
        if len(row) > 1:
            cols.append(1)
        cols.extend(range(4, len(row), 2))
        return cols

    def all_norm_cols(row):
        return expr_cols(row) + validity_cols(row)

    # Gather already-existing v<number> indices so numbering doesn't collide.
    max_index = -1
    for row in stack:
        if not row:
            continue
        for idx in all_norm_cols(row):
            if idx >= len(row):
                continue
            for token in token_pattern.findall(row[idx]):
                m = re.fullmatch(r'v(\d+)', token)
                if m:
                    max_index = max(max_index, int(m.group(1)))

    next_idx = max_index + 1 if max_index >= 0 else 0
    replacement_map: dict[str, str] = {}

    for row in stack:
        if not row:
            continue
        for idx in all_norm_cols(row):
            if idx >= len(row):
                continue
            for token in token_pattern.findall(row[idx]):
                if token in replacement_map:
                    continue
                if token.startswith("u_"):
                    continue  # LEAVE u_ variables ALONE!

                if _looks_like_internal_var_token(token):
                    if token.isdigit():
                        replacement_map[token] = f"v{token}"
                    elif token.startswith("x") and token[1:].isdigit():
                        replacement_map[token] = f"v{token[1:]}"
                    else:
                        replacement_map[token] = f"v{next_idx}"
                        next_idx += 1

    # Apply replacements to all normalized columns (expr + validity)
    for i, row in enumerate(stack):
        if not row:
            continue
        for j in all_norm_cols(row):
            if j < len(row):
                stack[i][j] = replace_keys_in_string(row[j], replacement_map)


def rename_stack(stack: list[list[str]], theorem: str):
    """
    Renaming is now fully handled by process_proof_graphs.py.
    We do absolutely nothing here to preserve the processed variables.
    """
    pass


def read_stack(file_path: str, proof_part: str):
    """
    Load a raw stack written by generate_raw_proof_graph.

    - File is resolved under PROJECT_ROOT / 'files/processed_proof_graph'
    - Filename is the same sanitizer as in generate_raw_proof_graph:
        f"{_safe(theorem)}__{proof_part}.txt"
    - Each line corresponds to one list[str], items separated by tabs.
    - Empty line => empty list.
    """
    # same sanitizer used when writing

    stack = []
    with open(file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            # remove newline only (preserve all other characters)
            line = raw_line.rstrip("\n")
            # handle Windows CR if present
            if line.endswith("\r"):
                line = line[:-1]
            if line == "":
                stack.append([])  # empty list was written as a blank line
            else:
                stack.append(line.split("\t"))  # items were joined with tabs
    return stack


def _norm_expr(expr: str) -> str:
    return re.sub(r"\s+", "", expr or "").lower()


def _row_validity_values(row: list[str]) -> list[str]:
    vals = []
    if not row:
        return vals
    if len(row) > 1 and row[1]:
        vals.append(row[1])
    for i in range(4, len(row), 2):
        if row[i]:
            vals.append(row[i])
    return vals


def _row_contains_rhs_expr(row: list[str], expr: str) -> bool:
    target = _norm_expr(expr)
    for i in range(3, len(row), 2):
        if _norm_expr(row[i]) == target:
            return True
    return False


def _partition_stack_subproofs(stack: list[list[str]]):
    """
    Split one stack into main stack + subproof sections.

    Subproofs are declared by rows with explanation == "validity name":
      [implication_expr, ..., "validity name", subproof_goal_expr, implication_expr]
    """
    subproofs = []
    seen_impl = set()

    for row_idx, row in enumerate(stack):
        if not row or len(row) < 4:
            continue
        if len(row) > 2 and row[2] == "validity name":
            implication_expr = row[0]
            subproof_goal_expr = row[3]
            impl_norm = _norm_expr(implication_expr)
            if impl_norm in seen_impl:
                continue
            seen_impl.add(impl_norm)
            subproofs.append({
                "implication_expr": implication_expr,
                "implication_norm": impl_norm,
                "goal_expr": subproof_goal_expr,
                "goal_norm": _norm_expr(subproof_goal_expr),
                "namespace_expr": row[4] if len(row) > 4 else implication_expr,
                "namespace_norm": _norm_expr(row[4] if len(row) > 4 else implication_expr),
                "validity_row_idx": row_idx,
                "seed_expansion_idx": None,
                "member_indices": set(),
            })

    if not subproofs:
        return [copy.deepcopy(r) for r in stack], []

    for sp in subproofs:
        for row_idx, row in enumerate(stack):
            if not row or len(row) < 4:
                continue
            if row[2] == "expansion for integration" and _row_contains_rhs_expr(row, sp["implication_expr"]):
                sp["seed_expansion_idx"] = row_idx
                break

    assigned_to_any_subproof = set()

    for sp in subproofs:
        impl_norm = sp["implication_norm"]
        namespace_norm = sp["namespace_norm"]
        goal_norm = sp["goal_norm"]

        for row_idx, row in enumerate(stack):
            if not row:
                continue

            belongs = False

            if len(row) > 2 and row[2] == "validity name" and _norm_expr(row[0]) == impl_norm:
                belongs = True

            if not belongs:
                for v in _row_validity_values(row):
                    if _norm_expr(v) == namespace_norm:
                        belongs = True
                        break

            if not belongs and _norm_expr(row[0]) == goal_norm:
                belongs = True

            if not belongs and sp["seed_expansion_idx"] == row_idx:
                belongs = True

            if belongs:
                sp["member_indices"].add(row_idx)
                assigned_to_any_subproof.add(row_idx)

        ordered_indices = sorted(sp["member_indices"])
        # Hide the declaration row from the subproof body; we show it in the subproof header/meta.
        ordered_indices = [i for i in ordered_indices if not (len(stack[i]) > 2 and stack[i][2] == "validity name")]

        if sp["seed_expansion_idx"] in ordered_indices:
            ordered_indices.remove(sp["seed_expansion_idx"])

        display_indices = []
        if sp["seed_expansion_idx"] is not None:
            display_indices.append(sp["seed_expansion_idx"])
        display_indices.extend(ordered_indices)

        # Put the subproof goal-alias row at the end (and highlight it there), mirroring the main-stack visual logic.
        goal_pos = None
        for k, row_idx in enumerate(display_indices):
            if _norm_expr(stack[row_idx][0]) == goal_norm:
                goal_pos = k
                break
        if goal_pos is not None:
            goal_idx = display_indices.pop(goal_pos)
            display_indices.append(goal_idx)

        sp["display_stack"] = [copy.deepcopy(stack[i]) for i in display_indices]

    main_stack = [copy.deepcopy(row) for i, row in enumerate(stack) if i not in assigned_to_any_subproof]

    subproofs.sort(key=lambda sp: sp["validity_row_idx"])
    return main_stack, subproofs


def render_stack_with_subproofs(stack: list[list[str]], prefix: str = "") -> str:
    main_stack, subproofs = _partition_stack_subproofs(stack)

    blocks = []

    subproof_anchor_map = {}
    for j, sp in enumerate(subproofs, start=1):
        anchor_id = f"{prefix}subproof{j}" if prefix else f"subproof{j}"
        sp["anchor_id"] = anchor_id
        subproof_anchor_map[sp["implication_norm"]] = anchor_id

    blocks.append("<div class='proof-section main-proof-section'>")
    blocks.append("<div class='proof-section-title'>Main stack</div>")
    if main_stack:
        blocks.append(format_stack_entries(main_stack, prefix=f"{prefix}m", external_anchor_map=subproof_anchor_map))
    else:
        blocks.append("<div class='proof-empty'>No main-stack entries.</div>")
    blocks.append("</div>")

    if subproofs:
        blocks.append("<div class='proof-section subproofs-section'>")
        blocks.append("<div class='proof-section-title'>Subproofs</div>")

        for j, sp in enumerate(subproofs, start=1):
            title_expr_html = wrap_clickable(sp["implication_expr"])
            goal_expr = html.escape(sp["goal_expr"])

            blocks.append("<div class='subproof-card'>")
            blocks.append(
                f"<div class='subproof-title'><span id='{sp['anchor_id']}'>{title_expr_html}</span> <span class='subproof-label'>subproof</span></div>"
            )
            blocks.append(
                f"<div class='subproof-meta'>Goal alias: <span class='subproof-goal'>{goal_expr}</span>"
                f"{_format_validity_tag(sp['namespace_expr'])}</div>"
            )

            if sp["display_stack"]:
                blocks.append(
                    format_stack_entries(
                        sp["display_stack"],
                        prefix=f"{prefix}sp{j}",
                        reverse_entries=False,
                        goal_key_norm=sp["goal_norm"]
                    )
                )
            else:
                blocks.append("<div class='proof-empty'>No subproof steps detected.</div>")

            blocks.append("</div>")

        blocks.append("</div>")

    return "".join(blocks)

# Proof step functions returning HTML for subchapters
def check_zero(theorem, file_path, induction_var, prefix=''):
    stack = read_stack(file_path, "check_zero")
    rename_stack(stack, theorem)
    return render_stack_with_subproofs(stack, prefix)


def check_induction_condition(theorem, file_path, induction_var, prefix=''):
    stack = read_stack(file_path, "check_induction_condition")
    rename_stack(stack, theorem)
    return render_stack_with_subproofs(stack, prefix)


def direct(theorem, file_path, prefix=''):
    stack = read_stack(file_path, "direct")
    rename_stack(stack, theorem)
    return render_stack_with_subproofs(stack, prefix)


def split_at_plus(s: str) -> tuple[str, str]:
    left, sep, right = s.partition("+")
    if sep != "+":
        raise ValueError("String does not contain '+'")
    return left, right


def debugging(path_plus_end, file_path, prefix=''):
    stack = read_stack(file_path, "debugging")
    return render_stack_with_subproofs(stack, prefix)


def mirrored(theorem, file_path, prefix=''):
    # Actually read the beautifully processed file instead of hardcoding a fake stack!
    stack = read_stack(file_path, "mirrored statement")
    rename_stack(stack, theorem)
    return render_stack_with_subproofs(stack, prefix)


def read_theorem_list(map_path: str | Path | None = None):
    """
    Reads PROJECT_ROOT/files/processed_proof_graph/global_theorem_list.txt
    and returns a list of (theorem, method, var) tuples.

    Each line in the file must be tab-separated: theorem \t method \t var
    Blank or malformed lines are ignored.
    """
    if map_path is None:
        map_path = Path(PROJECT_ROOT) / "files" / "processed_proof_graph" / "global_theorem_list.txt"
    else:
        map_path = Path(map_path)

    theorem_list = []
    with open(map_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            theorem, method, var = parts[0], parts[1], parts[2]
            theorem_list.append((theorem, method, var))
    return theorem_list


def makes_file_path_map(theorem_list, base_dir=None):
    """
    Build a map from theorem name -> list of indexed stack file paths.

    Indexing matches generate_raw_proof_graph/find_ends:
      - induction  -> two files:  <i>_check_zero.txt, <i+1>_check_induction_condition.txt
      - direct     -> one file:   <i>_direct_proof.txt
      - debug      -> one file:   <i>_debug.txt
      - mirrored statement (if present) -> one file: <i>_mirrored_statement.txt
      - unknown method -> one file: <i>_unknown_<sanitized>.txt

    Args:
        theorem_list: list[tuple[str, str, str]] like [(theorem, method, var), ...]
        base_dir: optional Path/str; default PROJECT_ROOT/files/raw_proof_graph

    Returns:
        dict[str, list[Path]]
    """
    if base_dir is None:
        base_dir = Path(PROJECT_ROOT) / "files" / "processed_proof_graph"
    else:
        base_dir = Path(base_dir)

    result = {}
    idx = 0  # global file index (0-based)

    for name, method, var in theorem_list:
        m = (method or "").lower()
        files = []

        if m == "induction":
            files.append(base_dir / f"{idx}_check_zero.txt")
            files.append(base_dir / f"{idx + 1}_check_induction_condition.txt")
            idx += 2
        elif m == "direct":
            files.append(base_dir / f"{idx}_direct_proof.txt")
            idx += 1
        elif m == "debug":
            files.append(base_dir / f"{idx}_debug.txt")
            idx += 1
        elif m == "mirrored statement":
            files.append(base_dir / f"{idx}_mirrored_statement.txt")
            idx += 1
        else:
            safe = re.sub(r"[^A-Za-z0-9._\-+]+", "_", m)[:64] or "unknown"
            files.append(base_dir / f"{idx}_unknown_{safe}.txt")
            idx += 1

        # accumulate (support multiple entries of the same theorem name)
        if name in result:
            result[name].extend(files)
        else:
            result[name] = files

    return result


def generate_proof_graph_pages(config: configuration_reader):
    global theorem_to_file, theorem_shape_to_file
    create_expressions.set_configuration(config)

    out_dir = PROJECT_ROOT / "files/full_proof_graph"

    # 2. convert to Path and ensure the directory exists (or will be deleted)
    # out_dir = Path(out_dir)

    # if the directory already exists, delete it and everything inside
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    theorem_list = read_theorem_list()
    file_path_map = makes_file_path_map(theorem_list)

    # JavaScript for left-click/right-click expansion; popup named “Expression” with deep-navy styling
    popup_script = """
    <script>
    function processText(input) {
      const indentChar = "  ";
      let indent = 0, output = "", token = "";
      for (const char of input) {
        if (char === "(") {
          if (token.trim()) { output += indentChar.repeat(indent) + token.trim() + "\\n"; token = ""; }
          output += indentChar.repeat(indent) + "(\\n"; indent++;
        } else if (char === ")") {
          if (token.trim()) { output += indentChar.repeat(indent) + token.trim() + "\\n"; token = ""; }
          indent = Math.max(0, indent - 1); 
          output += indentChar.repeat(indent) + ")" + "\\n";
        } else {
          token += char;
        }
      }
      if (token.trim()) output += indentChar.repeat(indent) + token.trim() + "\\n";
      return output;
    }

    document.addEventListener('DOMContentLoaded', function() {
      // Inject a reusable dialog into the body
      document.body.insertAdjacentHTML('beforeend', `
        <dialog id="expr-modal" style="border:1px solid #d0d7de; border-radius:6px; background:#f6f8fa; color:#000080; padding:1em; max-width:80%; max-height:80vh; overflow:auto; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
          <pre id="expr-content" style="white-space:pre-wrap; font-family:monospace; margin:0;"></pre>
        </dialog>
      `);
      
      const modal = document.getElementById('expr-modal');
      const content = document.getElementById('expr-content');

      // Close modal when clicking anywhere outside the box
      modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.close();
      });

      document.body.addEventListener('contextmenu', function(e) {
        const el = e.target.closest('.clickable');
        if (!el) return;
        e.preventDefault();
        
        // Populate and open the native modal
        content.textContent = processText(el.getAttribute('data-text'));
        modal.showModal();
      });
    });
    </script>
    """

    # Common CSS for chapter pages (original colors preserved)
    common_style = """
    <style>
      body { font-family: Arial, sans-serif; margin: 2rem; color: #000080; padding-bottom: 200em;}
      nav a { text-decoration: none; color: #0366d6; margin-right: 1em; }
      /* enforce same color & no underline for all links, visited or not */
      a, a:link, a:visited {
        color: #0366d6;
        text-decoration: none;
      }
      .var-highlight { font-style: italic; margin-bottom: 1em; display: block; }
      .step-output { margin: 1em 0; padding: 0.5em; background: #f6f8fa; border-radius: 4px; }
      .clickable    { cursor: pointer; }   /* no color overridden here */
      .goal-highlight { color: red; font-weight: bold; }
       /* only kill underlines on our autogenerated theorem links */
      a.theorem-link,
      a.theorem-link .clickable {
      text-decoration: none;
      /* force link to inherit whatever color its parent has (e.g. the red inline span) */
      color: inherit;
     }
     /* make inter-page theorem links lightblue */
     a.theorem-link[href$=".html"] .clickable {
     color: #0366d6 !important;
     }
     /* 2) but inside our .mirrored block revert to the red you use for in-page jumps */
     .mirrored a.theorem-link[href$=".html"] .clickable {
     color: #d73a49 !important;
     }
     .validity-tag {
       color: #22863a !important;
       font-size: 0.78em;
       margin-left: 0.45em;
       white-space: nowrap;
       display: inline-block;
       font-weight: normal;
       line-height: 1;
       vertical-align: baseline;
     }

     .proof-section { margin-top: 0.8em; }
     .proof-section-title {
       font-size: 1.05em;
       font-weight: 700;
       color: #586069;
       margin: 0.2em 0 0.8em 0;
       text-transform: uppercase;
       letter-spacing: 0.03em;
     }
     .main-proof-section { margin-bottom: 1.2em; }
     .subproofs-section { border-top: 1px solid #d0d7de; padding-top: 0.8em; }
     .subproof-card {
       margin: 0.8em 0 1.1em 0;
       padding: 0.7em 0.8em;
       background: #fbfdff;
       border: 1px solid #d8dee4;
       border-left: 4px solid #9ecbff;
       border-radius: 6px;
     }
     .subproof-title {
       font-weight: 700;
       margin-bottom: 0.35em;
       color: #24292f;
     }
     .subproof-label {
       color: #57606a;
       font-weight: 600;
       margin-left: 0.15em;
     }
     .subproof-meta {
       margin-bottom: 0.55em;
       color: #57606a;
       font-size: 0.95em;
     }
     .subproof-goal { color: #0366d6; }
     .proof-empty { color:#6a737d; font-style: italic; }

    </style>
    """

    # --- Index page ---
    index_head = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Proof Graph – Index</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; }}
    ul {{ list-style: none; padding: 0; }}
    ul ul {{ padding-left: 1.5rem; font-size: 0.9em; }}
    li {{ margin-bottom: 0.5em; }}
    a {{ text-decoration: none; color: #0366d6; }}
  </style>
  {popup_script}
</head>
<body>
  <h1>Proof Graph</h1>
  <h2>Table of Contents</h2>
  <ul>"""
    index_tail = """  </ul>
</body>
</html>"""

    toc = []
    # build mapping from each displayed theorem title → its chapter file
    theorem_to_file = {
        rename_theorem(name): f"chapter{idx}.html"
        for idx, (name, *_) in enumerate(theorem_list, start=1)
    }

    # Fuzzy theorem-link map for instantiated/broadcast theorem expressions (alpha-equivalent match).
    _shape_buckets = {}
    for idx, (name, *_) in enumerate(theorem_list, start=1):
        disp = rename_theorem(name)
        shape = _alpha_normalize_theorem_expr(disp)
        if not shape:
            continue
        _shape_buckets.setdefault(shape, set()).add(f"chapter{idx}.html")
    theorem_shape_to_file = {
        shape: next(iter(files))
        for shape, files in _shape_buckets.items()
        if len(files) == 1
    }

    for idx, (name, method, _) in enumerate(theorem_list, start=1):
        filename = f"chapter{idx}.html"
        toc.append(f"    <li>{idx}. <a href='{filename}'>{html.escape(rename_theorem(name))}</a>")
        # force a new line and style it
        if not debug:
            toc.append(
                f"    <div style=\"margin-left:20px; color:#888888; font-weight:bold; font-size:1.3em;\">"
                f"{html.escape(visu_helpers.make_readable_title(rename_theorem(name)))}</div>"
            )

        if method.lower() == "induction":
            toc.append("      <ul>")
            toc.append(f"        <li>{idx}.1. <a href='{filename}#sub1'>Check for 0</a></li>")
            toc.append(f"        <li>{idx}.2. <a href='{filename}#sub2'>Check induction condition</a></li>")
            toc.append("      </ul>")
        toc.append("    </li>")

    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join([index_head] + toc + [index_tail]))

    # --- Chapter pages ---
    for idx, (name, method, var) in enumerate(theorem_list, start=1):
        filename = f"chapter{idx}.html"
        prev_link = f"<a href='chapter{idx - 1}.html'>Previous</a>" if idx > 1 else ""
        next_link = f"<a href='chapter{idx + 1}.html'>Next</a>" if idx < len(theorem_list) else ""
        nav_links = ' '.join(link for link in (prev_link, next_link) if link)

        head = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(rename_theorem(name))}</title>
  {f'<title>{html.escape(visu_helpers.make_readable_title(rename_theorem(name)))}</title>' if not debug else ''}
  {common_style}
  {popup_script}
</head>
<body>
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <nav>
      <a href="index.html">Index</a> {nav_links}
    </nav>
    <div>
      <span style="margin-right:1em;"><span style="color:red; font-weight:bold;">■</span> Goal of the proof</span>
      <span style="margin-right:1em;"><span style="color:lightblue;">■</span> Has justification link</span>
      <span style="margin-right:1em;"><span style="color:#888888;">■</span> Readable version</span>
      <span style="margin-right:1em;"><span style="color:#22863a;">(namespace)</span></span>
      <span style="margin-right:1em;"><span style="color:magenta; font-weight:bold;">"integration target"</span></span>
      <span>Right-click to expand</span>
    </div>
  </div>
  <h1>Chapter {idx}: {html.escape(rename_theorem(name))}</h1>
  {f'''<div style="margin-left:20px; color:#888888; font-weight:bold; font-size:3em;">
    {html.escape(visu_helpers.make_readable_title(rename_theorem(name)))}
  </div><br><br>''' if not debug else ''}
"""

        body = [head]
        if method.lower() == "induction":
            body.extend([
                f"  <span class=\"var-highlight\">Induction variable: v{html.escape(var)}</span>",
                "  <h2 id=\"sub1\">Check for 0</h2>",
                f"  <div class=\"step-output\">{check_zero(name, file_path_map[name][0], var, f'c{idx}s1')}</div>",
                "  <h2 id=\"sub2\">Check induction condition</h2>",
                f"  <div class=\"step-output\">{check_induction_condition(name, file_path_map[name][1], var, f'c{idx}s2')}</div>",
            ])

        elif method.lower() == "direct":
            body.extend([
                "  <h2>Direct Proof</h2>",
                "  <div class='step-output'>", direct(name, file_path_map[name][0]), "  </div>",
            ])
        elif method.lower() == "debug":
            body.extend([
                "  <h2>Debugging</h2>",
                "  <div class='step-output'>", debugging(name, file_path_map[name][0]), "  </div>",
            ])
        elif method.lower() == "mirrored statement":
            body.extend([
                "  <h2>Mirrored</h2>",
                "  <div class='step-output'>",
                # Pass the file path from the map instead of the raw 'var' string
                mirrored(name, file_path_map[name][0]),
                "  </div>",
            ])
        body.append("</body>")
        body.append("</html>")

        with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as f:
            f.write("\n".join(body))