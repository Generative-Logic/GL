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
import json
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

# Tag descriptions for the proof graph reference page and right-click popups
TAG_DESCRIPTIONS = {
    "implication": (
        "Hash-table inference",
        "A premise was matched against a universally quantified implication rule "
        "stored in hash memory, and the conclusion was emitted. "
        "The first dependency is the rule itself; the remaining dependencies are "
        "the expressions that matched the rule's premises."
    ),
    "expansion": (
        "Named expression expanded",
        "A named compound expression (e.g. NaturalNumbers, fXY) was expanded "
        "into its compiled definition structure from the GL binary. "
        "The dependency is the named expression that was expanded."
    ),
    "disintegration": (
        "Compound expression decomposed",
        "A conjunction (&amp;) or existence node was broken apart into its "
        "constituent sub-expressions. For conjunctions, each element is extracted; "
        "for existence nodes, the left element (with a fresh bound variable) and "
        "the right element are produced."
    ),
    "task formulation": (
        "Proof premise (root assumption)",
        "The starting assumption for the theorem under proof. In a direct proof, "
        "this is the premise of the implication being proved. In an induction proof, "
        "this includes the base case or induction hypothesis."
    ),
    "equality1": (
        "Argument substitution via equality",
        "An expression's argument was replaced using an equality fact. "
        "If (=[a,b]) is known and f(...,a,...) exists, then f(...,b,...) is derived."
    ),
    "equality2": (
        "Transitivity of equality",
        "From (=[a,b]) and (=[b,c]), the equality (=[a,c]) is derived. "
        "This is the standard transitivity rule for the equality relation."
    ),
    "symmetry of equality": (
        "Symmetry of equality",
        "From (=[a,b]), the symmetric equality (=[b,a]) is derived. "
        "This is the standard symmetry rule for the equality relation."
    ),
    "recursion": (
        "Induction hypothesis",
        "Introduction of the induction hypothesis. In check_zero chapters, the induction "
        "variable is set to i0 (zero). In check_induction_condition chapters, the "
        "successor step is applied to the induction variable."
    ),
    "theorem": (
        "Previously proved theorem",
        "A theorem that was proved in an earlier chapter is used as an inference "
        "rule. The dependency links to the chapter where the theorem was originally proved."
    ),
    "reformulation for integration": (
        "Reformulated for reverse-disintegration",
        "An expression was reformulated into a form suitable for integration "
        "(reverse disintegration). This prepares the expression structure so "
        "that it can be reassembled into a compound expression."
    ),
    "expansion for integration": (
        "Expanded for reverse-disintegration",
        "A named expression was expanded specifically in preparation for integration. "
        "The expanded form provides the structural template needed for the "
        "reverse-disintegration step."
    ),
    "premise element": (
        "Implication premise consumed",
        "A premise of an implication was matched during the integration process. "
        "This marks one of the conditions that needed to be satisfied before "
        "the implication's conclusion could be assembled."
    ),
    "validity name": (
        "Implication scope identifier",
        "Ties an expression to the specific implication whose scope it belongs to "
        "during the integration process. Ensures that premises and conclusions "
        "are matched within the correct logical context."
    ),
    "anchor handling": (
        "Anchor variable substitution",
        "An anchor variable was substituted with a concrete value from the "
        "definition set. This binds abstract anchor parameters to specific "
        "elements (e.g., replacing argument position 1 with an element of N)."
    ),
    "mirrored from": (
        "Mirror of source theorem",
        "This theorem is a mirrored variant of another theorem — the output-variable "
        "premise and the head (conclusion) are swapped. The dependency links to "
        "the original source theorem."
    ),
    "reformulated from": (
        "Reformulation of source theorem",
        "This theorem is a reformulation of another theorem — the head (conclusion) "
        "has been rewritten using an existence-node expansion from the GL binary. "
        "The dependency links to the original source theorem."
    ),
    "necessity for equality (hypo)": (
        "Equality hypothesis introduction",
        "An equality is introduced as a hypothesis for collapsing duplicate "
        "variables. When multiple variables must be identified as equal, this "
        "tag marks the equality assumption that enables the collapse."
    ),
    "externally provided theorem": (
        "External theorem (not proved by GL)",
        "A theorem injected from the externally_provided_theorems.txt file. "
        "This theorem was not proved by GL's own deduction engine but is "
        "accepted as a given fact for use in downstream proofs."
    ),
    "incubator back reformulation": (
        "Back-reformulated operator theorem",
        "An operator-equality theorem (e.g., a+b=c) was back-reformulated from "
        "the implication form into a direct operator statement. The dependency "
        "links to the source proof."
    ),
    "contradiction": (
        "Proved by contradiction",
        "Both an expression and its negation were derived within the same "
        "logic block, establishing a contradiction. The theorem is proved "
        "because the negation of the conclusion led to an inconsistency."
    ),
    "origin": (
        "Provenance tracking",
        "Tracks the origin of an expression in a contradiction proof chain. "
        "Links an expression to its source derivation so the full "
        "dependency chain can be reconstructed."
    ),
    "multiplied from": (
        "Partition-based variable equalization",
        "A theorem produced by the multiplyImplication algorithm. Bell partitions "
        "of bound variables generate copies where variable groups are set equal, "
        "enabling cross-expression equalization."
    ),
    "equalize variable": (
        "Variable equalization step",
        "A step within a multiplied proof where specific variables are identified "
        "as equal. This is part of the partition-based equalization process "
        "that collapses variables across expressions."
    ),
}


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


# Display v-variables as w-variables for applied theorems (avoids confusion with chapter's own v-variables)
def _v_to_w_display(expr: str) -> str:
    """Replace v-prefixed variables with w-prefixed ones for display only."""
    args = extract_args(expr)
    v_args = [a for a in args if re.fullmatch(r'v\d+', a) or re.fullmatch(r'vx\d+', a)]
    if not v_args:
        return expr
    rmap = {a: 'w' + a[1:] for a in v_args}
    return replace_keys_in_string(expr, rmap)


def _strip_i_prefix(expr: str) -> str:
    """Strip 'i' prefix from anchor element names (i0->0, i1->1) for display.
    Works on both bracketed args and readable math text."""
    return re.sub(r'\bi(\d+)\b', r'\1', expr)


# Extract top-level constituent expression names for GL binary display
_EXPR_NAME_RE = re.compile(r'\(([A-Za-z_][A-Za-z0-9_]*)\[')

def _extract_expr_parts(expr: str) -> str:
    """Extract unique expression names (e.g. 'in2', 'in3', 'AnchorPeano') from an expression.
    Returns comma-separated string of names that exist in the GL binary map."""
    names = list(dict.fromkeys(_EXPR_NAME_RE.findall(expr)))  # unique, order-preserving
    return ','.join(names) if names else ''


def _space_commas(s: str) -> str:
    """Add spaces after commas inside [...] brackets for display readability."""
    return re.sub(r',(?=[^\s])', ', ', s)


# Utility function to wrap clickable substrings starting with '(' or '!(' and ending at space or end-of-string
def wrap_clickable(text):
    def repl(m):
        s = m.group(0)  # the raw token, e.g. "(v7*v8)=(v8*v7)"
        esc = html.escape(s, quote=True)
        parts = html.escape(_extract_expr_parts(s), quote=True)
        parts_attr = f' data-parts="{parts}"' if parts else ''
        target = _resolve_theorem_target(s)
        if target:
            # Display with w-variables, keep original v-expression for link target and data-text
            display = html.escape(_strip_i_prefix(_v_to_w_display(s)), quote=True)
            span = f'<span class="clickable" data-text="{esc}"{parts_attr}>{display}</span>'
            return f'<a href="{target}" class="theorem-link">{span}</a>'
        # span for the normal "expand on right-click"
        display_esc = html.escape(_strip_i_prefix(s), quote=True)
        span = f'<span class="clickable" data-text="{esc}"{parts_attr}>{display_esc}</span>'
        return span

    pattern = r"!?\([^ ]*?\)(?= |$)"
    result = re.sub(pattern, repl, text)
    # Strip i-prefix from plain text segments (outside HTML tags)
    result = re.sub(r'(?<![<"\w])i(\d+)(?!["\w>])', r'\1', result)
    # Wrap _integration_goal suffix as a right-clickable label
    result = result.replace(
        '_integration_goal',
        '<span class="integration-goal-label">_integration_goal</span>')
    return result


def _htmlify_readable(text):
    """Convert plain-text readable title to HTML with mathematical notation."""
    h = html.escape(text)
    # (preorder[N,+,a,b]) -> a < b
    h = re.sub(
        r'\(preorder\[([^,]+),([^,]+),([^,]+),([^\]]+)\]\)',
        lambda m: f'{m.group(3)} &lt; {m.group(4)}',
        h)
    # (interval[N,+,start,end,set]) -> set = [start,end]
    h = re.sub(
        r'\(interval\[([^,]+),([^,]+),([^,]+),([^,]+),([^\]]+)\]\)',
        lambda m: f'{m.group(5)} = [{m.group(3)},{m.group(4)}]',
        h)
    # (fXY[a,B,C]) -> a: B -> C
    h = re.sub(
        r'\(fXY\[([^,]+),([^,]+),([^\]]+)\]\)',
        lambda m: f'{m.group(1)}: {m.group(2)} \u2192 {m.group(3)}',
        h)
    # (sequence[N,+,c,a,b]) -> b is a sequence (b^i)_{i in [c,a]}
    h = re.sub(
        r'\(sequence\[([^,]+),([^,]+),([^,]+),([^,]+),([^\]]+)\]\)',
        lambda m: f'{m.group(5)} is a sequence ({m.group(5)}<sup>i</sup>)<sub>i\u2208[{m.group(3)},{m.group(4)}]</sub>',
        h)
    # sum(i=start..end) -> vertical sigma with bounds above/below, summing i
    def _fmt_sum(m):
        lo, hi, fn = m.group(1), m.group(2), m.group(3)
        body = f'{fn}(i)' if fn else 'i'
        sigma = (
            '<span class="sm" style="display:inline-flex;flex-direction:column;'
            'align-items:center;vertical-align:middle;margin:0 2px;line-height:1">'
            f'<span style="font-size:0.6em">{hi}</span>'
            '<span style="font-size:1.8em;line-height:0.8;margin-bottom:8px">\u2211</span>'
            f'<span style="font-size:0.6em;margin-top:8px">i={lo}</span>'
            '</span>'
        )
        return f'{sigma} {body}'
    h = re.sub(r'sum\(i=([^.]+)\.\.([^)]+)\)(?: (\w+)\(i\))?', _fmt_sum, h)
    # enlarge parentheses that directly wrap a sigma block
    h = re.sub(
        r'\(([^()]*?<span class="sm".*?</span>[^()]*?)\)',
        lambda m: (
            '<span style="font-size:2em;vertical-align:middle;font-weight:normal">(</span>'
            + m.group(1) +
            '<span style="font-size:2em;vertical-align:middle;font-weight:normal">)</span>'
        ), h)
    # Space around '=' in text segments only (skip HTML tags/attributes)
    parts = re.split(r'(<[^>]+>)', h)
    h = ''.join(
        re.sub(r'(?<!=)\s*=\s*(?!=)', ' = ', p) if not p.startswith('<') else p
        for p in parts
    )
    # vN → v<sub>N</sub> in text segments only
    parts = re.split(r'(<[^>]+>)', h)
    h = ''.join(
        re.sub(r'v(\d+)', r'v<sub>\1</sub>', p) if not p.startswith('<') else p
        for p in parts
    )
    # Add breathing room around scaffolding keywords
    for kw in ['RULE:', 'IMPLIES:', 'from', 'follows', 'and', 'mirrored from',
               'reformulated from', 'back-reformulated from', 'is a sequence']:
        h = h.replace(kw, f'&ensp;{kw}&ensp;')
    return h


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
    visible_count = sum(1 for e in rev if e and 'theorem' not in e)
    step_num = 0

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
        step_num += 1
        step_badge = f"<span class='step-badge'>({step_num}/{visible_count})</span>"

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
                    '<span class="clickable" style="color:#EF9F27 !important; font-weight:bold !important;"',
                    1
                )
            else:
                first_html = f'<span class="clickable" style="color:#EF9F27; font-weight:bold">{first_html}</span>'

            if rest:
                rest_html = wrap_clickable(rest[0])
                key_html = f"{first_html} {rest_html}"
            else:
                key_html = first_html
        else:
            key_html = wrap_clickable(key_expr)

        # Expression diff highlight for equality1 steps
        if explanation == "equality1" and len(entry) > 3:
            key_html = _diff_highlight_html(key_html, key_expr, entry[3])

        if key_validity:
            key_html += _format_validity_tag(key_validity)

        # Collect dependency anchor IDs for hover-highlighting
        dep_ids = []
        for di in range(3, len(entry), 2):
            d_expr = entry[di]
            d_norm = re.sub(r'\s+', '', d_expr).lower()
            if d_norm in key_map:
                dep_ids.append(key_map[d_norm])
            elif d_norm in external_anchor_map:
                dep_ids.append(external_anchor_map[d_norm])
        deps_attr = f" data-deps=\"{' '.join(dep_ids)}\"" if dep_ids else ""
        parts = [f"<span id='{anchor}'{deps_attr}>{key_html}</span>"]

        if explanation:
            tag_key = explanation.lower().strip()
            # Normalize sub-variants of "reformulation for integration"
            if tag_key.startswith("reformulation for integration"):
                tag_anchor = "reformulation-for-integration"
            else:
                tag_anchor = tag_key.replace(" ", "-").replace("(", "").replace(")", "")
            escaped = html.escape(explanation)
            parts.append(
                f"<a href='tags.html#{tag_anchor}' class='proof-tag' "
                f"data-tag='{html.escape(tag_key, quote=True)}'>"
                f"<b>{escaped}</b></a>"
            )

        if explanation == "validity name":
            if len(entry) > 3:
                rhs_html = wrap_clickable(entry[3])
                if len(entry) > 4 and entry[4]:
                    rhs_html += _format_validity_tag(entry[4])
                parts.append(rhs_html)

            content_html = "&nbsp;&nbsp;".join(parts)
            if cursor_index is not None and idx == cursor_index:
                content_html = (
                    f"<span style='background-color:#3A3520; padding:4px; "
                    f"border-radius:4px'>{content_html}</span>"
                )
            line_html = f"<div class='proof-line'>{step_badge}<span class='proof-line-content'>{content_html}</span></div>"
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

                # 2. Re-wrap in magenta span, right-clickable for integration goal explanation
                ref_html = f'<span class="integration-goal-label" style="color:#E879F9 !important; font-weight:bold !important;">"{clean_text}"</span>'
            # -----------------------------------------

            if is_integration_target:
                linked_ref = ref_html
            elif norm_ref in key_map:
                linked_ref = f"<a href='#{key_map[norm_ref]}' style='text-decoration:none'>{ref_html}</a>"
            elif norm_ref in external_anchor_map:
                linked_ref = f"<a href='#{external_anchor_map[norm_ref]}' style='text-decoration:none'>{ref_html}</a>"
            else:
                linked_ref = ref_html

            if ing_val:
                linked_ref += _format_validity_tag(ing_val)

            parts.append(linked_ref)

        content_html = "&nbsp;&nbsp;".join(parts)
        if cursor_index is not None and idx == cursor_index:
            content_html = (
                f"<span style='background-color:#3A3520; padding:4px; "
                f"border-radius:4px'>{content_html}</span>"
            )

        line_html = f"<div class='proof-line'>{step_badge}<span class='proof-line-content'>{content_html}</span></div>"
        lines.append(line_html)

        if len(entry) > 2 and entry[2] == 'implication':
            if len(entry) > 3:
                helper_list = [entry[0], "implication", entry[3]]
                for k in range(5, len(entry), 2):
                    helper_list.append(entry[k])

                impl_text = visu_helpers.format_implication(helper_list)
                if impl_text:
                    impl_html = (
                        f"<div class='implication readable-grey'>"
                        f"{_htmlify_readable(_strip_i_prefix(impl_text))}</div>"
                    )
                    lines.append(impl_html)

        if len(entry) > 2 and entry[2] == 'mirrored from':
            if len(entry) > 3:
                helper_list = [entry[0], "mirrored from", entry[3]]
                mirrored_text = format_mirroring(helper_list)
                mirrored_html = (
                    f"<div class='mirrored readable-grey'>"
                    f"{_htmlify_readable(_strip_i_prefix(mirrored_text))}</div>"
                )
                lines.append(mirrored_html)

        if len(entry) > 2 and entry[2] == 'reformulated from':
            if len(entry) > 3:
                helper_list = [entry[0], "reformulated from", entry[3]]
                reformulated_text = visu_helpers.format_reformulation(helper_list)
                reformulated_html = (
                    f"<div class='reformulated readable-grey'>"
                    f"{_htmlify_readable(_strip_i_prefix(reformulated_text))}</div>"
                )
                lines.append(reformulated_html)

        if len(entry) > 2 and entry[2] == 'incubator back reformulation':
            if len(entry) > 3:
                source_readable = visu_helpers.make_readable_title(entry[3])
                back_ref_readable = visu_helpers.make_readable_title(entry[0])
                back_ref_text = f"{back_ref_readable} back-reformulated from {source_readable}"
                back_ref_html = (
                    f"<div class='readable-grey'>"
                    f"{_htmlify_readable(_strip_i_prefix(back_ref_text))}</div>"
                )
                lines.append(back_ref_html)

    return "\n".join(lines)

def extract_args(s: str) -> list[str]:
    # same pattern as before
    pattern = r'(?<=[\[,])([^,\[\]]+)(?=[\],])'
    all_subs = re.findall(pattern, s)
    # remove duplicates while preserving order
    return list(dict.fromkeys(all_subs))


def _diff_highlight_html(result_html: str, result_expr: str, source_expr: str) -> str:
    """Highlight arguments in result_html that differ from source_expr."""
    # Extract bracket structure: name[arg1,arg2,...] from both
    r_match = re.match(r'^!?\((\w+)\[([^\]]+)\]\)$', result_expr.strip())
    s_match = re.match(r'^!?\((\w+)\[([^\]]+)\]\)$', source_expr.strip())
    if not r_match or not s_match:
        return result_html
    if r_match.group(1) != s_match.group(1):
        return result_html  # different expression name
    r_args = r_match.group(2).split(',')
    s_args = s_match.group(2).split(',')
    if len(r_args) != len(s_args):
        return result_html
    # Find changed args
    changed = {r_args[i] for i in range(len(r_args)) if r_args[i] != s_args[i]}
    if not changed:
        return result_html
    # Wrap changed arg text in highlight spans — text segments only (skip HTML tags/attributes)
    # Search for both raw form (v2, i4) and i-stripped form (4) since display strips i-prefix
    for arg in changed:
        variants = [html.escape(arg)]
        if re.match(r'^i\d+$', arg):
            variants.append(html.escape(arg[1:]))  # stripped form: i4 -> 4
        for escaped in variants:
            pat = re.compile(r'(?<=[,\[])(' + re.escape(escaped) + r')(?=[,\]])')
            parts = re.split(r'(<[^>]+>)', result_html)
            result_html = ''.join(
                pat.sub(r'<span class="arg-changed">\1</span>', p) if not p.startswith('<') else p
                for p in parts
            )
    return result_html



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


def _strip_integration_goal(expr: str) -> str:
    s = (expr or "").strip()
    # Strip any accidental quotation marks so the suffix can be correctly removed
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    suffix = "_integration_goal"
    return s[:-len(suffix)] if s.endswith(suffix) else s

def _structural_match(expr1: str, expr2: str) -> bool:
    """Check if two expressions match structurally, ignoring exact 'v' variable numbers."""
    def norm_vars(s: str) -> str:
        return re.sub(r'v\d+', 'vX', s)
    return norm_vars(expr1) == norm_vars(expr2)

def _row_contains_rhs_expr_integration_aware(row: list[str], expr: str) -> bool:
    target = _norm_expr(_strip_integration_goal(expr))
    for i in range(3, len(row), 2):
        cell_expr = _norm_expr(_strip_integration_goal(row[i]))
        # Check for exact match or structural match (v12 vs v17)
        if cell_expr == target or _structural_match(cell_expr, target):
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
            if row[2] == "expansion for integration" and _row_contains_rhs_expr_integration_aware(row, sp["implication_expr"]):
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

            # Keep the declaration row (the implication / validity-name line)
            # on the main stack. The subproof card already uses it as header/meta.
            # Only the actual local proof steps belong to the subproof body.
            if len(row) > 2 and row[2] == "validity name" and _norm_expr(row[0]) == impl_norm:
                continue

            belongs = False

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

    # Build main stack key map so subproofs can link back to main stack entries
    main_prefix_str = f"{prefix}m"
    main_rev = list(main_stack)[::-1]
    main_key_map = {}
    for midx, mentry in enumerate(main_rev):
        if not mentry:
            continue
        mnorm = re.sub(r'\s+', '', mentry[0]).lower()
        main_key_map[mnorm] = f"{main_prefix_str}-entry{midx}"

    blocks.append("<div class='proof-section main-proof-section'>")
    blocks.append("<div class='proof-section-title'>Main stack</div>")
    if main_stack:
        blocks.append(format_stack_entries(main_stack, prefix=f"{prefix}m", external_anchor_map=subproof_anchor_map))
    else:
        blocks.append("<div class='proof-empty'>No main-stack entries.</div>")
    blocks.append("</div>")

    # Merge main key map with subproof anchors for cross-referencing inside subproofs
    subproof_external_map = {**main_key_map, **subproof_anchor_map}

    if subproofs:
        blocks.append("<div class='proof-section subproofs-section'>")
        blocks.append("<div class='proof-section-title'>Subproofs</div>")

        for j, sp in enumerate(subproofs, start=1):
            title_expr_html = wrap_clickable(sp["implication_expr"])
            goal_expr = html.escape(sp["goal_expr"])

            blocks.append("<div class='subproof-card collapsed'>")
            blocks.append(
                f"<div class='subproof-title'><span class='subproof-toggle'>\u25BC</span>"
                f"<span id='{sp['anchor_id']}'>{title_expr_html}</span> <span class='subproof-label'>subproof</span></div>"
            )
            blocks.append("<div class='subproof-body'>")
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
                        goal_key_norm=sp["goal_norm"],
                        external_anchor_map=subproof_external_map
                    )
                )
            else:
                blocks.append("<div class='proof-empty'>No subproof steps detected.</div>")

            blocks.append("</div>")  # close subproof-body
            blocks.append("</div>")  # close subproof-card

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


def reformulated(theorem, file_path, prefix=''):
    stack = read_stack(file_path, "reformulated statement")
    rename_stack(stack, theorem)
    return render_stack_with_subproofs(stack, prefix)


def back_reformulated(theorem, file_path, prefix=''):
    stack = read_stack(file_path, "incubator back reformulation")
    rename_stack(stack, theorem)
    return render_stack_with_subproofs(stack, prefix)


def _compute_chapter_stats(*file_paths):
    """Compute stats across one or more proof stack files."""
    steps = 0
    tags = set()
    theorems_cited = 0
    for fp in file_paths:
        stack = read_stack(fp, None)
        for entry in stack:
            if not entry:
                continue
            tag = entry[2] if len(entry) > 2 else ""
            if 'theorem' in entry and tag != "theorem":
                continue
            steps += 1
            if tag:
                tags.add(tag)
            if tag == "theorem":
                theorems_cited += 1
    parts = [f"{steps} step{'s' if steps != 1 else ''}"]
    parts.append(f"{len(tags)} reasoning rule{'s' if len(tags) != 1 else ''}")
    if theorems_cited:
        parts.append(f"{theorems_cited} theorem{'s' if theorems_cited != 1 else ''} cited")
    return ", ".join(parts)


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
        elif m == "reformulated statement":
            files.append(base_dir / f"{idx}_reformulated_statement.txt")
            idx += 1
        elif m == "incubator back reformulation":
            files.append(base_dir / f"{idx}_back_reformulated_statement.txt")
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


# ---------------------------------------------------------------------------
# GL Binary Map — read per-tag JSON from files/GL_binaries/
# ---------------------------------------------------------------------------

def _gl_split_elements(elements_raw):
    """Split elements string into individual elements, handling negation prefix."""
    elements = []
    current = ''
    depth = 0
    for ch in elements_raw:
        if ch == '(':
            depth += 1
            current += ch
        elif ch == ')':
            depth -= 1
            current += ch
            if depth == 0:
                elements.append(current.strip())
                current = ''
        elif depth == 0 and ch == '!':
            current += ch
        elif depth == 0 and ch == ' ':
            current = ''
        else:
            current += ch
    return elements


def _gl_get_bracket_tokens(expr):
    """Get all tokens from bracket args in order of first appearance."""
    tokens = []
    for m in re.finditer(r'\[([^\]]*)\]', expr):
        for tok in m.group(1).split(','):
            tok = tok.strip()
            if tok and tok not in tokens:
                tokens.append(tok)
    return tokens


def _gl_rename_vars(expr, var_map):
    """Replace variable tokens inside brackets using var_map."""
    def _repl(m):
        args = m.group(1).split(',')
        return '[' + ','.join(var_map.get(a.strip(), a.strip()) for a in args) + ']'
    return re.sub(r'\[([^\]]*)\]', _repl, expr)


def _gl_make_conjunction(elements):
    """Build nested right-associated conjunction: (&e1(&e2 e3))"""
    if len(elements) == 0:
        return ''
    if len(elements) == 1:
        return elements[0]
    if len(elements) == 2:
        return '(&' + elements[0] + elements[1] + ')'
    return '(&' + elements[0] + _gl_make_conjunction(elements[1:]) + ')'


def build_gl_binary_map(gl_binaries_dir):
    """Read per-tag JSON files from GL_binaries/ and build gl_binary_map for popup display."""
    gl_binary_map = {}
    gl_binaries_path = Path(gl_binaries_dir)
    if not gl_binaries_path.exists():
        return gl_binary_map

    for json_file in sorted(gl_binaries_path.glob("GL_binary_*.json")):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for core, entry in data.items():
            category = entry.get('category', '')
            signature = entry.get('signature', '')
            elements_list = entry.get('elements', [])

            if category == 'atomic' or not elements_list:
                continue

            # Parse signature args -> u_ to x mapping
            sig_args = []
            m = re.search(r'\[([^\]]*)\]', signature)
            if m:
                sig_args = [a.strip() for a in m.group(1).split(',')]
            u_to_x = {}
            xi = 1
            for arg in sig_args:
                if arg.startswith('u_') and arg not in u_to_x:
                    u_to_x[arg] = f'x{xi}'
                    xi += 1

            # Elements are already a list from JSON
            elements = elements_list

            # Find bound vars (non-u_ tokens) across all elements -> y mapping
            bound_to_y = {}
            yi = 1
            for elem in elements:
                for tok in _gl_get_bracket_tokens(elem):
                    if tok not in u_to_x and tok not in bound_to_y:
                        bound_to_y[tok] = f'y{yi}'
                        yi += 1

            # Combined variable mapping
            var_map = {**u_to_x, **bound_to_y}

            # Rename signature and elements
            renamed_sig = _gl_rename_vars(signature, var_map)
            renamed_elems = [_gl_rename_vars(e, var_map) for e in elements]

            # Collect bound var names (yN) for quantifiers
            if category == 'existence':
                qvars = []
                for tok in _gl_get_bracket_tokens(elements[0]):
                    if tok in bound_to_y and bound_to_y[tok] not in qvars:
                        qvars.append(bound_to_y[tok])
            else:
                qvars = list(bound_to_y.values())

            bound_str = ','.join(qvars)

            # Reconstruct MPL based on category
            if category == 'and':
                mpl = _gl_make_conjunction(renamed_elems)
            elif category == 'existence':
                first = renamed_elems[0]
                second = renamed_elems[1] if len(renamed_elems) > 1 else ''
                if second.startswith('!'):
                    neg_second = second[1:]
                else:
                    neg_second = '!' + second
                mpl = '!(>[' + bound_str + ']' + first + neg_second + ')'
            elif category == 'implication':
                premises = renamed_elems[:-1]
                conclusion = renamed_elems[-1]
                premises_conj = _gl_make_conjunction(premises)
                mpl = '(>[' + bound_str + ']' + premises_conj + conclusion + ')'
            else:
                mpl = _gl_make_conjunction(renamed_elems)

            gl_binary_map[core] = {
                'signature': renamed_sig,
                'mpl': mpl
            }

    return gl_binary_map


def _generate_tags_page(out_dir, common_style):
    """Generate a tags.html reference page listing all proof tags with descriptions."""
    rows = []
    for tag_key, (short_desc, long_desc) in TAG_DESCRIPTIONS.items():
        anchor = tag_key.replace(" ", "-").replace("(", "").replace(")", "")
        rows.append(
            f'<tr id="{html.escape(anchor)}">'
            f'<td style="white-space:nowrap; vertical-align:top; padding:0.6em 1.2em 0.6em 0;">'
            f'<b>{html.escape(tag_key)}</b></td>'
            f'<td style="vertical-align:top; padding:0.6em 1.2em 0.6em 0; color:#5DCAA5;">'
            f'{html.escape(short_desc)}</td>'
            f'<td style="vertical-align:top; padding:0.6em 0;">'
            f'{long_desc}</td></tr>'
        )
    table_html = "\n".join(rows)

    tags_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Reasoning Rules Reference</title>
  <link rel="icon" type="image/png" href="favicon.png">
  {common_style}
  <style>
    tr:target {{ background: #3A3520; }}
    tr:target td {{ padding-top: 0.8em; padding-bottom: 0.8em; }}
  </style>
</head>
<body>
  <nav><a href="index.html">Index</a></nav>
  <h1>Reasoning Rules Reference</h1>
  <p style="color:#8B8FA5; margin-bottom:1.5em;">
    Each step in a proof chapter is annotated with a reasoning rule that describes how the
    expression was derived. Click a rule name in any chapter to jump here; right-click
    for a quick popup.
  </p>
  <table style="border-collapse:collapse; width:100%;">
    <thead>
      <tr style="border-bottom:2px solid #3A3D4A;">
        <th style="text-align:left; padding:0.5em 1.2em 0.5em 0;">Reasoning rule</th>
        <th style="text-align:left; padding:0.5em 1.2em 0.5em 0; color:#5DCAA5;">Short</th>
        <th style="text-align:left; padding:0.5em 0;">Description</th>
      </tr>
    </thead>
    <tbody>
      {table_html}
    </tbody>
  </table>
</body>
</html>"""

    with open(os.path.join(out_dir, "tags.html"), "w", encoding="utf-8") as f:
        f.write(tags_page)


def generate_proof_graph_pages(config: configuration_reader,
                               proc_dir=None, out_dir=None):
    global theorem_to_file, theorem_shape_to_file
    create_expressions.set_configuration(config)

    if proc_dir is None:
        proc_dir = PROJECT_ROOT / "files" / "processed_proof_graph"
    if out_dir is None:
        out_dir = PROJECT_ROOT / "files/full_proof_graph"

    # if the directory already exists, delete it and everything inside
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    # Copy favicon logo into output directory
    favicon_src = PROJECT_ROOT / "small_logo_amber_bl_bg_2.png"
    if favicon_src.exists():
        shutil.copy2(favicon_src, Path(out_dir) / "favicon.png")

    theorem_list = read_theorem_list(proc_dir / "global_theorem_list.txt")
    file_path_map = makes_file_path_map(theorem_list, base_dir=proc_dir)

    # Build "used by" reverse-dependency map: chapter_idx -> list of citing chapter_idxs
    used_by = {}  # keyed by 1-based chapter index
    for citing_idx, (citing_name, _, _) in enumerate(theorem_list, start=1):
        for fp in file_path_map.get(citing_name, []):
            if not os.path.exists(fp):
                continue
            stack = read_stack(fp, None)
            for entry in stack:
                if not entry or len(entry) < 3 or entry[2] != "theorem":
                    continue
                # entry is a marker row: the theorem expression is in entry[0]
                # find which chapter it belongs to
                cited_disp = rename_theorem(entry[0])
                for src_idx, (src_name, _, _) in enumerate(theorem_list, start=1):
                    if src_idx == citing_idx:
                        continue
                    if rename_theorem(src_name) == cited_disp:
                        used_by.setdefault(src_idx, set()).add(citing_idx)
                        break

    # Build GL binary map from per-tag JSON files (check next to proc_dir first, then default)
    gl_binaries_dir = Path(proc_dir).parent / "GL_binaries"
    if not gl_binaries_dir.exists():
        gl_binaries_dir = PROJECT_ROOT / "files" / "GL_binaries"
    gl_binary_map = build_gl_binary_map(gl_binaries_dir)
    gl_binary_json = json.dumps(gl_binary_map, ensure_ascii=False)

    # JavaScript for left-click/right-click expansion; popup named "Expression" with deep-navy styling
    popup_script = """
    <script>
    const GL_BINARY_MAP = """ + gl_binary_json + """;

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

    function escapeHtml(s) {
      return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }

    document.addEventListener('DOMContentLoaded', function() {
      // Inject a reusable dialog into the body
      document.body.insertAdjacentHTML('beforeend', `
        <dialog id="expr-modal" style="border:1px solid #3A3D4A; border-radius:6px; background:#262938; color:#F0E8DC; padding:1em; max-width:80%; max-height:80vh; overflow:auto; box-shadow: 0 4px 12px rgba(0,0,0,0.4);">
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

        const rawText = el.getAttribute('data-text');
        let output = processText(rawText);
        let glBinary = '';

        // GL binary section: show definition for each constituent expression
        const parts = el.getAttribute('data-parts');
        if (parts) {
            const names = parts.split(',').filter(Boolean);
            const shown = [];
            for (const name of names) {
                const entry = GL_BINARY_MAP[name];
                if (entry && !shown.includes(name)) {
                    shown.push(name);
                    if (shown.length === 1) {
                        glBinary += '\\n\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\n\\n';
                        glBinary += '<b style="font-size:1.15em">GL binary:</b>\\n\\n';
                    } else {
                        glBinary += '\\n';
                    }
                    glBinary += escapeHtml(entry.signature + ' :=\\n\\n');
                    glBinary += escapeHtml(processText(entry.mpl));
                }
            }
        }

        content.innerHTML = escapeHtml(output) + (glBinary ? glBinary : '');
        modal.showModal();
      });

      // --- Tag popup (right-click on proof tags) ---
      const TAG_INFO = """ + json.dumps(
        {k: {"short": v[0], "long": v[1]} for k, v in TAG_DESCRIPTIONS.items()},
        ensure_ascii=False) + """;

      // Tag popup dialog
      document.body.insertAdjacentHTML('beforeend', `
        <dialog id="tag-modal" style="border:1px solid #3A3D4A; border-radius:6px; background:#262938; color:#F0E8DC; padding:1.2em 1.5em; max-width:500px; box-shadow: 0 4px 12px rgba(0,0,0,0.4);">
          <div id="tag-content"></div>
        </dialog>
      `);
      const tagModal = document.getElementById('tag-modal');
      const tagContent = document.getElementById('tag-content');
      tagModal.addEventListener('click', (e) => {
        if (e.target === tagModal) tagModal.close();
      });

      document.body.addEventListener('contextmenu', function(e) {
        const tagEl = e.target.closest('.proof-tag');
        if (!tagEl) return;
        e.preventDefault();
        e.stopPropagation();
        const tagKey = tagEl.getAttribute('data-tag') || '';
        // Normalize sub-variants
        const lookupKey = tagKey.startsWith('reformulation for integration') ? 'reformulation for integration' : tagKey;
        const info = TAG_INFO[lookupKey];
        if (info) {
          tagContent.innerHTML =
            '<b style="font-size:1.15em; color:#5DCAA5;">' + escapeHtml(tagKey) + '</b>' +
            '<div style="color:#8B8FA5; margin:0.3em 0 0.6em 0; font-size:0.92em;">' + escapeHtml(info.short) + '</div>' +
            '<div style="line-height:1.5;">' + info.long + '</div>';
        } else {
          tagContent.innerHTML = '<b>' + escapeHtml(tagKey) + '</b>';
        }
        tagModal.showModal();
      });

      // --- Integration goal popup (right-click on _integration_goal labels) ---
      document.body.addEventListener('contextmenu', function(e) {
        const igEl = e.target.closest('.integration-goal-label');
        if (!igEl) return;
        e.preventDefault();
        e.stopPropagation();
        tagContent.innerHTML =
          '<b style="font-size:1.15em; color:#E879F9;">Integration goal</b>' +
          '<div style="color:#8B8FA5; margin:0.3em 0 0.6em 0; font-size:0.92em;">Operational target, not a derived result</div>' +
          '<div style="line-height:1.5;">' +
          'An integration goal does not need to be justified &mdash; it only must make sense structurally. ' +
          'GL can, for various reasons, decide to try to integrate a compound logical structure. ' +
          'This expression is a <em>goal for the operation</em>, not a result of it.<br><br>' +
          'GL could equally well try to integrate the Schr\\u00F6dinger equation and it would still be ' +
          'a mathematically correct operation &mdash; although it would make zero sense in the current axiomatic context.' +
          '</div>';
        tagModal.showModal();
      });

      // --- Collapsible subproofs ---
      document.querySelectorAll('.subproof-title').forEach(function(title) {
        title.style.cursor = 'pointer';
        title.addEventListener('click', function() {
          title.closest('.subproof-card').classList.toggle('collapsed');
        });
      });

      // --- Hover-highlight dependency chain ---
      function collectAncestors(id, visited) {
        if (visited.has(id)) return;
        visited.add(id);
        var el = document.getElementById(id);
        if (!el) return;
        var deps = (el.getAttribute('data-deps') || '').split(' ').filter(Boolean);
        for (var i = 0; i < deps.length; i++) collectAncestors(deps[i], visited);
      }
      document.body.addEventListener('mouseover', function(e) {
        var link = e.target.closest('a[href^="#"]');
        if (!link) return;
        link.classList.add('dep-glow');
        var targetId = link.getAttribute('href').slice(1);
        var targetEl = document.getElementById(targetId);
        if (targetEl) targetEl.classList.add('dep-glow');
        var srcLine = link.closest('span[id]');
        if (srcLine) srcLine.classList.add('dep-glow');
        var ancestors = new Set();
        collectAncestors(targetId, ancestors);
        ancestors.forEach(function(id) {
          var el = document.getElementById(id);
          if (el) el.classList.add('dep-glow');
        });
      });
      document.body.addEventListener('mouseout', function(e) {
        var link = e.target.closest('a[href^="#"]');
        if (!link) return;
        document.querySelectorAll('.dep-glow').forEach(function(el) {
          el.classList.remove('dep-glow');
        });
      });
    });
    </script>
    """

    # Common CSS for chapter pages — dark theme matching generative-logic.com
    common_style = """
    <style>
      body { font-family: Arial, sans-serif; margin: 2rem; background: #181B27; color: #F0E8DC; padding-bottom: 200em;}
      nav a { text-decoration: none; color: #5DCAA5; margin-right: 1em; }
      /* enforce same color & no underline for all links, visited or not */
      a, a:link, a:visited {
        color: #5DCAA5;
        text-decoration: none;
      }
      .var-highlight { font-style: italic; margin-bottom: 1em; display: block; }
      .step-output { margin: 1em 0; padding: 0.5em; background: #262938; border-radius: 4px; }
      .clickable    { cursor: pointer; }   /* no color overridden here */
      .integration-goal-label { cursor: pointer; }
      .proof-line { display: flex; align-items: flex-start; margin-bottom: 1.8em; }
      .step-badge { color: #6B6F82; font-size: 0.75em; white-space: nowrap; min-width: 5em; flex-shrink: 0; text-align: right; margin-right: 0.7em; padding-top: 0.15em; }
      .proof-line-content { flex: 1; min-width: 0; }
      .readable-grey { margin-left: calc(5em + 0.7em); color: #8B8FA5; font-weight: bold; font-size: 1.3em; }
      .chapter-stats { margin-top: 2em; padding: 0.6em 1em; background: #262938; border-top: 1px solid #3A3D4A; color: #8B8FA5; font-size: 0.9em; }
      .dep-glow { background: linear-gradient(90deg, rgba(232,121,249,0.15), rgba(239,159,39,0.12) 60%, transparent); border-left: 2px solid rgba(232,121,249,0.6); padding-left: 6px; border-radius: 4px; box-shadow: 0 0 8px rgba(239,159,39,0.15), inset 0 0 12px rgba(232,121,249,0.06); transition: all 0.2s ease; }
      .arg-changed { color: #F97316; text-decoration: underline; text-decoration-style: wavy; }
      .subproof-toggle { cursor: pointer; user-select: none; margin-right: 0.3em; display: inline-block; transition: transform 0.2s; }
      .subproof-card.collapsed .subproof-body { display: none; }
      .subproof-card.collapsed .subproof-toggle { transform: rotate(-90deg); }
      .goal-highlight { color: #EF9F27; font-weight: bold; }
       /* only kill underlines on our autogenerated theorem links */
      a.theorem-link,
      a.theorem-link .clickable {
      text-decoration: none;
      /* force link to inherit whatever color its parent has (e.g. the gold inline span) */
      color: inherit;
     }
     /* make inter-page theorem links mint green */
     a.theorem-link[href$=".html"] .clickable {
     color: #5DCAA5 !important;
     }
     /* 2) but inside our .mirrored block use amber */
     .mirrored a.theorem-link[href$=".html"] .clickable {
     color: #EF9F27 !important;
     }
     .validity-tag {
       color: #E8D44D !important;
       font-size: 0.78em;
       margin-left: 0.45em;
       white-space: nowrap;
       display: inline-block;
       font-weight: normal;
       line-height: 1;
       vertical-align: baseline;
     }
     a.proof-tag, a.proof-tag:link, a.proof-tag:visited {
       color: #F0E8DC;
       text-decoration: none;
       cursor: pointer;
       border-bottom: 1px dotted #5DCAA5;
     }
     a.proof-tag:hover {
       color: #5DCAA5;
     }

     .proof-section { margin-top: 0.8em; }
     .proof-section-title {
       font-size: 1.05em;
       font-weight: 700;
       color: #8B8FA5;
       margin: 0.2em 0 0.8em 0;
       text-transform: uppercase;
       letter-spacing: 0.03em;
     }
     .main-proof-section { margin-bottom: 1.2em; }
     .subproofs-section { border-top: 1px solid #3A3D4A; padding-top: 0.8em; }
     .subproof-card {
       margin: 0.8em 0 1.1em 0;
       padding: 0.7em 0.8em;
       background: #262938;
       border: 1px solid #3A3D4A;
       border-left: 4px solid #5DCAA5;
       border-radius: 6px;
     }
     .subproof-title {
       font-weight: 700;
       margin-bottom: 0.35em;
       color: #F0E8DC;
     }
     .subproof-label {
       color: #8B8FA5;
       font-weight: 600;
       margin-left: 0.15em;
     }
     .subproof-meta {
       margin-bottom: 0.55em;
       color: #8B8FA5;
       font-size: 0.95em;
     }
     .subproof-goal { color: #5DCAA5; }
     .proof-empty { color:#6B6F82; font-style: italic; }

    </style>
    """

    # --- Index page ---
    index_head = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Proof Graph – Index</title>
  <link rel="icon" type="image/png" href="favicon.png">
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; background: #181B27; color: #F0E8DC; }}
    ul {{ list-style: none; padding: 0; }}
    ul ul {{ padding-left: 1.5rem; font-size: 0.9em; }}
    li {{ margin-bottom: 0.5em; }}
    a, a:link, a:visited {{ text-decoration: none; color: #5DCAA5; }}
    .clickable {{ cursor: pointer; }}
  </style>
  {popup_script}
</head>
<body>
  <nav><a href="tags.html">Reasoning rules</a></nav>
  <h1>Proof Graph</h1>
  <h2>Table of Contents</h2>
  <input type="text" id="theorem-search" placeholder="Filter theorems..."
         style="width:100%; max-width:600px; padding:0.5em; margin-bottom:1em; background:#262938; color:#F0E8DC;
                border:1px solid #3A3D4A; border-radius:4px; font-size:1em; outline:none;">
  <ul id="theorem-list">"""
    index_tail = """  </ul>
  <script>
  document.getElementById('theorem-search').addEventListener('input', function() {
    const q = this.value.toLowerCase();
    document.querySelectorAll('#theorem-list > li').forEach(function(li) {
      li.style.display = li.textContent.toLowerCase().includes(q) ? '' : 'none';
    });
  });
  </script>
</body>
</html>"""

    toc = []
    # build mapping from each displayed theorem title → its chapter file
    theorem_to_file = {
        rename_theorem(name): f"chapter{idx}.html"
        for idx, (name, *_) in enumerate(theorem_list, start=1)
    }

    chapter_theorem_list_idx = [(name, method, var) for name, method, var in theorem_list]

    # Fuzzy theorem-link map for instantiated/broadcast theorem expressions (alpha-equivalent match).
    _shape_buckets = {}
    for idx, (name, *_) in enumerate(chapter_theorem_list_idx, start=1):
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

    for idx, (name, method, _) in enumerate(chapter_theorem_list_idx, start=1):
        filename = f"chapter{idx}.html"
        theorem_display = rename_theorem(name)
        theorem_esc = html.escape(theorem_display, quote=True)
        display_stripped = html.escape(_strip_i_prefix(theorem_display), quote=True)
        theorem_span = f'<span class="clickable" data-text="{theorem_esc}">{display_stripped}</span>'
        toc.append(f"    <li>{idx}. <a href='{filename}' style='text-decoration:none'>{theorem_span}</a>")
        # force a new line and style it
        if not debug:
            toc.append(
                f"    <div style=\"margin-left:20px; color:#8B8FA5; font-weight:bold; font-size:1.3em;\">"
                f"{_htmlify_readable(_strip_i_prefix(visu_helpers.make_readable_title(rename_theorem(name))))}</div>"
            )

        if method.lower() == "induction":
            toc.append("      <ul>")
            toc.append(f"        <li>{idx}.1. <a href='{filename}#sub1'>Check for 0</a></li>")
            toc.append(f"        <li>{idx}.2. <a href='{filename}#sub2'>Check induction condition</a></li>")
            toc.append("      </ul>")
        citing = sorted(used_by.get(idx, set()))
        if citing:
            links = ", ".join(f"<a href='chapter{c}.html' style='color:#EF9F27; font-weight:bold;'>{c}</a>" for c in citing)
            toc.append(f"    <div style='margin-left:20px; color:#6B6F82; font-size:0.85em;'>Used by: {links}</div>")
        toc.append("    </li>")

    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join([index_head] + toc + [index_tail]))

    # --- Tags reference page ---
    _generate_tags_page(out_dir, common_style)

    # --- Chapter pages ---
    chapter_theorem_list = [(name, method, var) for name, method, var in theorem_list]
    for idx, (name, method, var) in enumerate(chapter_theorem_list, start=1):
        filename = f"chapter{idx}.html"
        prev_link = f"<a href='chapter{idx - 1}.html'>Previous</a>" if idx > 1 else ""
        next_link = f"<a href='chapter{idx + 1}.html'>Next</a>" if idx < len(chapter_theorem_list) else ""
        nav_links = ' '.join(link for link in (prev_link, next_link) if link)

        head = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(_strip_i_prefix(rename_theorem(name)))}</title>
  {f'<title>{html.escape(_strip_i_prefix(visu_helpers.make_readable_title(rename_theorem(name))))}</title>' if not debug else ''}
  <link rel="icon" type="image/png" href="favicon.png">
  {common_style}
  {popup_script}
</head>
<body>
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <nav>
      <a href="index.html">Index</a> <a href="tags.html">Reasoning rules</a> {nav_links}
    </nav>
    <div>
      <span style="margin-right:1em; color:#EF9F27; font-weight:bold;">Goal of the proof</span>
      <span style="margin-right:1em; color:#5DCAA5;">Has justification link</span>
      <span style="margin-right:1em; color:#8B8FA5; font-weight:bold; font-size:1.3em;">Readable version</span>
      <span style="margin-right:1em; color:#E8D44D;">Namespace</span>
      <span style="margin-right:1em; color:#E879F9; font-weight:bold;">Integration goal</span>
      <span style="margin-right:1em; border-bottom:1px dotted #5DCAA5;"><b>Reasoning rule</b></span>
      <span>Right-click to expand</span>
    </div>
  </div>
  <h1>Chapter {idx}: <span class="clickable" data-text="{html.escape(rename_theorem(name), quote=True)}">{html.escape(_strip_i_prefix(rename_theorem(name)))}</span></h1>
  {f'''<div style="margin-left:20px; color:#8B8FA5; font-weight:bold; font-size:3em;">
    {_htmlify_readable(_strip_i_prefix(visu_helpers.make_readable_title(rename_theorem(name))))}
  </div><br><br>''' if not debug else ''}
"""

        body = [head]
        if method.lower() == "induction":
            body.extend([
                f"  <span class=\"var-highlight\">Induction variable: {html.escape(var)}</span>",
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
        elif method.lower() == "reformulated statement":
            body.extend([
                "  <h2>Reformulated</h2>",
                "  <div class='step-output'>",
                reformulated(name, file_path_map[name][0]),
                "  </div>",
            ])
        elif method.lower() == "incubator back reformulation":
            body.extend([
                "  <h2>Back-Reformulated</h2>",
                "  <div class='step-output'>",
                back_reformulated(name, file_path_map[name][0]),
                "  </div>",
            ])
        # Chapter statistics footer
        stats_text = _compute_chapter_stats(*file_path_map[name])
        body.append(f"  <div class='chapter-stats'>{stats_text}</div>")

        body.append("</body>")
        body.append("</html>")

        with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as f:
            f.write("\n".join(body))