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

"""
Lightweight utility functions extracted from create_expressions.py.
Conjecture generation is now handled by the C++ conjecturer (gl_quick.exe --conjecture).
This module retains only the string-parsing utilities needed by visualization,
proof graph processing, and pipeline orchestration.
"""

import re
from typing import Dict
from configuration_reader import configuration_reader

# ---------------------------------------------------------------------------
# Module-level state (replaces create_expressions._CONFIGURATION)
# ---------------------------------------------------------------------------

_CONFIGURATION = configuration_reader()


def set_configuration(config: configuration_reader):
    global _CONFIGURATION
    _CONFIGURATION = config


def get_configuration_data():
    return _CONFIGURATION.data


def get_anchor_name(config: configuration_reader):
    # 1. Explicit anchor_name field in config JSON
    anchor_name = getattr(config, "anchor_name", None)
    if anchor_name and anchor_name in config.data:
        return anchor_name

    # 2. Try "Anchor" + anchor_id
    if getattr(config, "anchor_id", None):
        candidate = "Anchor" + config.anchor_id
        if candidate in config.data:
            return candidate

    assert False


# ---------------------------------------------------------------------------
# Expression parsing utilities
# ---------------------------------------------------------------------------

def extract_between_brackets(s, start_index=0):
    start = s.find('[', start_index)
    end = s.find(']', start)

    if start != -1 and end != -1:
        return s[start + 1:end]
    return None


def get_args(expr):
    sub_expr = extract_between_brackets(expr, 0)

    if sub_expr == "":
        return []
    else:
        return sub_expr.split(',')


def extract_expression(s: str) -> str:
    index = s.find('[')
    if index != -1:
        if s[0] == "(":
            return s[1:index]
        else:
            return s[0:index]
    return ""


def extract_expression_from_negation(s: str) -> str:
    start_index = s.find("!(")
    end_index = s.find("[")
    if start_index != -1 and end_index != -1 and start_index < end_index:
        return s[start_index + 2:end_index]
    return ""


# ---------------------------------------------------------------------------
# Tree / parse_expr / tree_to_expr
# ---------------------------------------------------------------------------

class TreeNode1:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.arguments = set()


def parse_expr(tree_str):
    tree_str = tree_str.replace("\n", "")
    tree_str = tree_str.replace(" ", "")
    tree_str = tree_str.replace("\t", "")
    index = 0

    def parse_subtree(s):
        nonlocal index

        if not s:
            raise RuntimeError("Input 's' cannot be empty. Execution terminated.")

        node = TreeNode1("")
        node_label = ""

        if s[index] == '(':
            index = index + 1
            if s[index] == '>':
                index = index + 1
                node_label = node_label + '>'
                args_to_remove = get_args(s[index:])
                index = s[index:].find(']') + index + 1
                node_label = node_label + "[" + ",".join(args_to_remove) + "]"
                node.left = parse_subtree(s)
                node.right = parse_subtree(s)
                if node.left is not None:
                    node.arguments.update(node.left.arguments)
                if node.right is not None:
                    node.arguments.update(node.right.arguments)
                node.arguments.difference_update(get_args(node_label))

            elif s[index] == '&':
                index = index + 1
                node_label = node_label + '&'
                node.left = parse_subtree(s)
                node.right = parse_subtree(s)
                if node.left is not None:
                    node.arguments.update(node.left.arguments)
                if node.right is not None:
                    node.arguments.update(node.right.arguments)

            else:
                end_index = s.find(')', index)
                if end_index == -1:
                    raise RuntimeError("No closing ')'.")
                node_label = s[index:end_index]
                index = end_index
                node.arguments.update(get_args(node_label))

        elif s[index:index + 2] == "!(":
            index = index + 2
            if s[index] == '>':
                index = index + 1
                node_label = node_label + "!>"
                args_to_remove = get_args(s[index:])
                index = s[index:].find(']') + index + 1
                node_label = node_label + "[" + ",".join(args_to_remove) + "]"
                node.left = parse_subtree(s)
                node.right = parse_subtree(s)
                if node.left is not None:
                    node.arguments.update(node.left.arguments)
                if node.right is not None:
                    node.arguments.update(node.right.arguments)
                node.arguments.difference_update(get_args(node_label))

            elif s[index] == '&':
                index = index + 1
                node_label = node_label + "!&"
                node.left = parse_subtree(s)
                node.right = parse_subtree(s)
                if node.left is not None:
                    node.arguments.update(node.left.arguments)
                if node.right is not None:
                    node.arguments.update(node.right.arguments)

            else:
                end_index = s.find(')', index)
                if end_index == -1:
                    raise RuntimeError("No closing ')'.")
                node_label = s[index:end_index]
                node_label = "!(" + node_label + ")"
                index = end_index
                node.arguments.update(get_args(node_label))
        elif s[index] == ")":
            index -= 1

        index = index + 1
        node.value = node_label
        if node.value == "":
            node = None
        return node

    root = parse_subtree(tree_str)
    return root


def tree_to_expr(root):
    local_expr = ""

    def node_to_str(node):
        nonlocal local_expr

        if node.value[0] == ">":
            local_expr = local_expr + "(" + node.value
        elif node.value == "&":
            local_expr = local_expr + "(&"
        elif node.value[0:2] == "!>":
            local_expr = local_expr + "!(" + node.value[1:]
        elif node.value == "!&":
            local_expr = local_expr + "!(&"
        elif node.value[0] == '!':
            local_expr = local_expr + "!(" + node.value[2:-1]
        else:
            local_expr = local_expr + "(" + node.value

        if node.left is not None:
            node_to_str(node.left)
        if node.right is not None:
            node_to_str(node.right)
        local_expr = local_expr + ")"

    node_to_str(root)
    return local_expr


# ---------------------------------------------------------------------------
# disintegrate_implication
# ---------------------------------------------------------------------------

def disintegrate_implication(expr_for_desintegration, chain):
    head = ""

    root = parse_expr(expr_for_desintegration)

    node = root
    while True:
        if node is not None:
            if node.value[0] == ">":
                chain.append((tree_to_expr(node.left), get_args(node.value), node.left.arguments))
                node = node.right
            else:
                head = tree_to_expr(node)
                break
        else:
            break

    return head


# ---------------------------------------------------------------------------
# replace_keys_in_string (regex-based)
# ---------------------------------------------------------------------------

_regex_cache = {}


def _get_compiled_regex(keys) -> re.Pattern:
    key_tuple = tuple(sorted(keys))
    if key_tuple not in _regex_cache:
        escaped_keys = [re.escape(key) for key in key_tuple]
        pattern = r'(?<=[\[,])(' + '|'.join(escaped_keys) + r')(?=[\],])'
        _regex_cache[key_tuple] = re.compile(pattern)
    return _regex_cache[key_tuple]


def replace_keys_in_string(big_string: str, replacement_map: Dict[str, str]) -> str:
    if not replacement_map:
        return big_string

    regex = _get_compiled_regex(replacement_map.keys())
    return regex.sub(lambda m: replacement_map.get(m.group(1), m.group(1)), big_string)
