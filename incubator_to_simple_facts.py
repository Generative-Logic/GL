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

"""
Optimized simple facts generator v2.

Key insight: j-copies are needed by generateEncodedRequests because it picks
facts at strictly increasing indices — it can't reuse the same fact string.
j-copies provide distinct string names for the same value, enabling the
request builder to construct multi-premise requests.

Strategy:
  - For facts with ALL DISTINCT i-value args: emit only the i-form
    (no j-copy needed — already has distinct names)
  - For facts with REPEATED i-value args: emit j-copy variants that break
    the repetition (need at least one variant with all-distinct names)
  - Always include j0/j1 variants for anchor-matching (max_j=2)

This produces fewer facts than the old "all j-copies for everything" approach
while retaining the structural diversity the CE engine needs.
"""

import re
import os
import itertools
from pathlib import Path
from configuration_reader import configuration_reader
import expression_utils

PROJECT_ROOT = Path(__file__).resolve().parent


def _get_operator_names(config):
    ops = set()
    for name in config:
        try:
            if config[name].output_args and config[name].input_args:
                ops.add(name)
        except (AttributeError, TypeError):
            pass
    return frozenset(ops)


def _get_anchor_arg_map(config):
    anchor_name = expression_utils.get_anchor_name(config)
    short_mpl = config[anchor_name].short_mpl_raw
    args = expression_utils.get_args(short_mpl)
    return {str(i + 1): arg for i, arg in enumerate(args)}


def _get_value_names(config):
    anchor_name = expression_utils.get_anchor_name(config)
    short_mpl = config[anchor_name].short_mpl_raw
    args = expression_utils.get_args(short_mpl)
    return {a for a in args if re.match(r'i\d+$', a)}


def _extract_head(theorem_str):
    # Anchor name may carry a trailing digit suffix after the AI3/AI8 split
    # (`AnchorIncubator3`, `AnchorIncubator8`) — accept digits after the first
    # alphabetic character so AI3 and AI8 theorems are recognised.
    m = re.search(r'\(Anchor[A-Za-z]\w*\[[^\]]*\]\)', theorem_str)
    if not m:
        return None, False
    rest = theorem_str[m.end():-1]
    is_neg = rest.startswith('!')
    if is_neg:
        rest = rest[1:]
    return rest, is_neg


def _replace_args(expr, arg_map):
    def replacer(match):
        content = match.group(1)
        args = content.split(',')
        new_args = [arg_map.get(a.strip(), a.strip()) for a in args]
        return '[' + ','.join(new_args) + ']'
    return re.sub(r'\[([^\]]+)\]', replacer, expr)


def _max_element_in_fact(fact):
    expr = fact[1:] if fact.startswith('!') else fact
    match = re.match(r'\(([^\[]+)\[([^\]]+)\]\)', expr)
    if not match:
        return -1
    max_val = -1
    for arg in match.group(2).split(','):
        m = re.match(r'i(\d+)$', arg)
        if m:
            max_val = max(max_val, int(m.group(1)))
    return max_val


def _make_anchor_with_copies(anchor_expr, value_names):
    val_to_num = {}
    for v in value_names:
        mt = re.match(r'i(\d+)$', v)
        if mt:
            val_to_num[v] = int(mt.group(1))

    def replacer(match):
        content = match.group(1)
        args = content.split(',')
        new_args = []
        for a in args:
            if a in val_to_num:
                new_args.append(f'j{val_to_num[a]}')
            else:
                new_args.append(a)
        return '[' + ','.join(new_args) + ']'

    return re.sub(r'\[([^\]]+)\]', replacer, anchor_expr)


def _expand_with_copies(raw_facts, value_names, operator_names, max_j,
                        variable_kinds=None):
    """
    Smart expansion: only generate j-copies where needed.

    For each fact:
      - Parse value positions and their element numbers
      - Check if any value arg repeats within the fact
      - If no repeats: emit only the all-i form (already distinct)
      - If repeats: generate j-copy combos that break repetitions
      - Always generate j0/j1 combos for anchor matching (up to max_j)

    Distinct check: all value arg NAMES must be distinct (operators only).
    """
    val_to_num = {}
    for v in value_names:
        mt = re.match(r'i(\d+)$', v)
        if mt:
            val_to_num[v] = int(mt.group(1))

    expanded = []
    seen = set()

    def add(atom):
        if atom not in seen:
            seen.add(atom)
            expanded.append(atom)

    for fact in raw_facts:
        is_neg = fact.startswith('!')
        expr = fact[1:] if is_neg else fact

        match = re.match(r'\(([^\[]+)\[([^\]]+)\]\)', expr)
        if not match:
            add(fact)
            continue

        name = match.group(1)
        args = match.group(2).split(',')
        is_operator = name in operator_names

        # Find value positions
        value_positions = []
        for i, arg in enumerate(args):
            if arg in val_to_num:
                value_positions.append((i, val_to_num[arg]))

        if not value_positions:
            add(fact)
            continue

        # Check if any value nums repeat
        value_nums = [vnum for _, vnum in value_positions]
        has_repeat = len(set(value_nums)) < len(value_nums)

        # Determine which positions need j-copies
        # If no repeat: only generate j-copies for elements that are in the
        # main anchor (j0, j1) — needed for anchor matching variety
        # If repeat: all elements can get j-copies to break repetition
        # All elements get j-copies — needed for cross-fact variable sharing
        # in generateEncodedRequests (j2 in P1 must match j2 in P2)
        kinds = variable_kinds if variable_kinds else ['i', 'j', 'k']
        kinds_per_pos = [kinds for _ in value_positions]

        for combo in itertools.product(*kinds_per_pos):
            new_args = list(args)
            for idx, (pos, vnum) in enumerate(value_positions):
                new_args[pos] = f'{combo[idx]}{vnum}'

            kinds_list = list(combo)

            # At least one 'i' for operators
            if is_operator and 'i' not in kinds_list:
                continue
            # Max copy count per non-i kind
            skip = False
            for k in kinds:
                if k != 'i' and kinds_list.count(k) > max_j:
                    skip = True
                    break
            if skip:
                continue

            # Distinct check: all value arg names must be unique (operators)
            if is_operator:
                value_vars = [new_args[pos] for pos, _ in value_positions]
                if len(set(value_vars)) != len(value_vars):
                    continue

            atom = f'({name}[{",".join(new_args)}])'
            if is_neg:
                atom = '!' + atom
            add(atom)

    return expanded


def convert_incubator_theorems(tag, config_path=None, theorems_dir=None,
                                out_dir=None):
    if config_path is None:
        # After the AI3/AI8 anchor split, the unsuffixed `ConfigIncubator<tag>.json`
        # no longer exists — each tag has at least the AI8 big batch
        # `ConfigIncubator<tag>1.json` and may have an AI3 mirror at suffix 2
        # plus a rung-1 batch at suffix 3 (Gauss). The AI8 batch is loaded here
        # because its 14-slot anchor argument list (N, i0, s, +, *, i1, i2,
        # id, i3, i4, i5, i6, i7, i8) is a strict superset of the AI3 9-slot
        # list, so its arg-index -> slot-name map covers theorems from every
        # incubator batch of the tag — AI3 theorems only ever reference
        # indices 1..9 and those agree with the AI8 mapping by construction.
        config_path = PROJECT_ROOT / "files" / "config" / f"ConfigIncubator{tag}1.json"
    if theorems_dir is None:
        theorems_dir = PROJECT_ROOT / "files" / "theorems_incubator"
    if out_dir is None:
        out_dir = PROJECT_ROOT / "files" / "simple_facts"

    config = configuration_reader(config_path)
    arg_map = _get_anchor_arg_map(config)
    value_names = _get_value_names(config)

    # Read from global theorem list (has method column) instead of theorems.txt
    # Only use 'direct' and 'incubator back reformulation' methods
    _ALLOWED_METHODS = {"direct", "incubator back reformulation"}
    gtl_path = PROJECT_ROOT / "files" / "incubator" / "processed_proof_graph" / "global_theorem_list.txt"
    if not gtl_path.exists():
        print(f"Warning: {gtl_path} not found.")
        return []

    raw_facts = []
    raw_seen = set()
    with open(gtl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            method = parts[1]
            if method not in _ALLOWED_METHODS:
                continue
            theorem = parts[0]
            head, is_neg = _extract_head(theorem)
            if head is None:
                continue
            # CE-filter consumes only atomic-head facts of the form
            # `(name[args])` or `!(name[args])`. Drop implications (`(>[`)
            # and conjunctions (`(&...`) — and their negations — since they
            # cannot match the byte-direct check in the prover's CE rule.
            if head.startswith('(>') or head.startswith('(&'):
                continue
            symbolic = _replace_args(head, arg_map)
            if is_neg:
                symbolic = '!' + symbolic
            if symbolic not in raw_seen:
                raw_seen.add(symbolic)
                raw_facts.append(symbolic)

    main_config_path = PROJECT_ROOT / "files" / "config" / f"Config{tag}.json"
    main_config = configuration_reader(main_config_path)
    operator_names = _get_operator_names(main_config)
    main_anchor_name = expression_utils.get_anchor_name(main_config)
    max_j = 2
    variable_kinds = config.parameters.fact_variable_kinds or None

    main_anchor_expr = main_config[main_anchor_name].short_mpl_raw
    anchor_with_copies = _make_anchor_with_copies(main_anchor_expr, value_names)
    # The i-version (no j-rewrite) is needed alongside the j-version so the
    # CE-filter conjecture rules can match their chain[0] anchor element.
    # The conjecturer's CE-filter `addConjectureForCEFiltering` substitutes
    # slot 2 -> "i0", slot 6 -> "i1" etc. (see `filter.cpp::addConjectureForCEFiltering`'s
    # replacementMap), producing chain[0] = (AnchorPeano[N,i0,s,+,*,i1]).
    # Pre-this change, simple_facts only carried the j-version
    # (AnchorPeano[N,j0,s,+,*,j1]) — string-mismatched against the chain
    # element, so the rule never fired and survivors that depend on the
    # anchor as a chain premise stayed unfiltered. The i-version below
    # closes that gap. See peano_filter_design_decisions.md D-rtf-2.
    anchor_i_version = main_anchor_expr

    sfp = config.parameters.simple_facts_parameters
    if not sfp:
        sfp = [None]

    os.makedirs(out_dir, exist_ok=True)
    out_paths = []

    for n in sfp:
        if n is not None:
            filtered = [f for f in raw_facts if _max_element_in_fact(f) <= n]
        else:
            filtered = raw_facts

        expanded = _expand_with_copies(filtered, value_names, operator_names, max_j,
                                       variable_kinds=variable_kinds)

        # Peano S0 axiom: ∀a ∈ N. ¬(s(a) = 0)  — "0 is not a successor".
        # Emitted as direct universal statements so the CE-filter
        # contradiction check at `prover.cpp::addExprToMemoryBlock` can
        # byte-direct-match against the negated-universal head produced by
        # `addConjectureForCEFiltering` (see D-rtf-2 trap output).
        #
        # The conjecturer's replacementMap names slot 7 → "i2" and slot
        # 8 → "id"; slots ≥ 9 retain their bare-digit form. To cover the
        # bound-variable names the Peano main conjecturer's CE-filter
        # rules use, we emit the S0 axiom in each variant. Soundness:
        # the axiom is an MPL-level Peano axiom (NaturalNumbers.txt line
        # 8–11) and holds in every Peano model, including the finite
        # `{i0..i_max, j0..j_max}` model the CE filter operates in.
        #
        # Emitted only for the Peano main pipeline (`tag == "Peano"`).
        # Gauss main has its own conjecturer surface and its own
        # replacementMap; injecting Peano-bound-var-form axioms there
        # would be no-ops at best and confusing at worst.
        # See peano_filter_design_decisions.md D-rtf-5.
        if tag == "Peano":
            zero_names = ["i0", "j0"]
            slot7_names = ["i2"]          # main conjecturer slot 7 → "i2"
            slot8_names = ["id"]          # main conjecturer slot 8 → "id"
            extra_slot_names = [str(d) for d in range(9, 14)]
            bvar_names = slot7_names + slot8_names + extra_slot_names
            seen_axiom = set(expanded)
            for zname in zero_names:
                for bv in bvar_names:
                    if bv == zname:
                        continue
                    axiom = f"(>[{bv}](in[{bv},N])!(in2[{bv},{zname},s]))"
                    if axiom not in seen_axiom:
                        seen_axiom.add(axiom)
                        expanded.append(axiom)

        # Reflexive equality atoms. The incubator's conjecturer treats
        # `(=[x, x])` as a tautology and skips it, so simple_facts has
        # zero positive same-arg equality atoms. CE-filter rules whose
        # chain references a positive equality with a bound variable
        # (e.g. `(=[i0, i2])` where i2 is bound) cannot fire because no
        # atom of shape `(=[i0, X])` matches when X would be i0 — the
        # would-be witness `(=[i0, i0])` is missing. Emitting the
        # reflexive atom for every value name (i_k and j_k) closes that
        # gap. Soundness: `(=[x, x])` is a logical tautology.
        # See peano_filter_design_decisions.md D-rtf-6.
        if tag == "Peano":
            seen_refl = set(expanded)
            digit_max = n if n is not None else max(
                (int(re.match(r'[ij](\d+)', name).group(1))
                 for name in (value_names or [])
                 if re.match(r'[ij](\d+)', name)),
                default=8,
            )
            for d in range(digit_max + 1):
                for kind in ("i", "j"):
                    atom = f"(=[{kind}{d},{kind}{d}])"
                    if atom not in seen_refl:
                        seen_refl.add(atom)
                        expanded.append(atom)

        # Equality-mirror expansion. The CE filter's contradiction check at
        # `prover.cpp::addExprToMemoryBlock` is string-direction-sensitive
        # for `(=[a, b])` — it only matches the negation against the exact
        # text in the LB's `intKnownStatements`. The incubator's conjecturer
        # enumerates only the canonical direction (e.g. `!(=[i0, i1])` but
        # not `!(=[i1, i0])`), and `_expand_with_copies` does not treat `=`
        # as an "operator" (no `output_args` for `=` in main config) so it
        # never adds the symmetric copy. Result: survivors whose head is
        # `(=[i1, i_k])` (constant on the LEFT, bound on the RIGHT) cannot
        # be refuted because `!(=[i1, i_k])` is missing from the LB.
        # This block computes the symmetric closure over `=` and `!=` facts
        # in the expanded set — adds `(=[b, a])` whenever `(=[a, b])` is
        # present and similarly for the negated form. Soundness: `=` is
        # symmetric in standard logic, so the mirror is a logical
        # consequence of the original — no new information, just the same
        # information in the form the CE-filter probe needs.
        # See peano_filter_design_decisions.md D-rtf-3.
        seen_mirror = set(expanded)
        mirrors = []
        for fact in expanded:
            m = re.match(r"^(!?)\(=\[([^,\]]+),([^,\]]+)\]\)$", fact)
            if not m:
                continue
            neg, a, b = m.group(1), m.group(2), m.group(3)
            if a == b:
                continue  # `(=[x, x])` is trivially equal to itself; no mirror needed
            mirror = f"{neg}(=[{b},{a}])"
            if mirror not in seen_mirror:
                seen_mirror.add(mirror)
                mirrors.append(mirror)
        expanded = expanded + mirrors

        suffix = f"_{n}" if n is not None else ""
        out_path = os.path.join(out_dir, f"simple_facts_{tag.lower()}{suffix}.txt")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(anchor_with_copies + "\n")
            if anchor_i_version != anchor_with_copies:
                f.write(anchor_i_version + "\n")
            for fact in expanded:
                f.write(fact + "\n")

        print(f"Facts multiplier applied.")
        out_paths.append(out_path)

    return out_paths


if __name__ == "__main__":
    convert_incubator_theorems("Peano")
