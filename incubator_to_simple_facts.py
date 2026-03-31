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
    m = re.search(r'\(Anchor[A-Za-z]+\[[^\]]*\]\)', theorem_str)
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
        config_path = PROJECT_ROOT / "files" / "config" / f"ConfigIncubator{tag}.json"
    if theorems_dir is None:
        theorems_dir = PROJECT_ROOT / "files" / "theorems_incubator"
    if out_dir is None:
        out_dir = PROJECT_ROOT / "files" / "simple_facts"

    config = configuration_reader(config_path)
    arg_map = _get_anchor_arg_map(config)
    value_names = _get_value_names(config)

    # Read from global theorem list (has method column) instead of proved_theorems.txt
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

        suffix = f"_{n}" if n is not None else ""
        out_path = os.path.join(out_dir, f"simple_facts_{tag.lower()}{suffix}.txt")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(anchor_with_copies + "\n")
            for fact in expanded:
                f.write(fact + "\n")

        print(f"Facts multiplier applied.")
        out_paths.append(out_path)

    return out_paths


if __name__ == "__main__":
    convert_incubator_theorems("Peano")
