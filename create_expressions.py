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


import time
import multiprocessing

import itertools
import copy

from typing import Tuple


from parameters import *
import re
from itertools import permutations
from typing import Dict
from typing import List, Set

from itertools import product

from typing import Any

from pathlib import Path

import os
import shutil


# wherever this file lives, assume the project root is its parent folder
PROJECT_ROOT = Path(__file__).resolve().parent

identity = {i: i for i in range(1, 1000000 + 1)}

_ALL_PERMUTATIONS = {}
_MAPPINGS_MAP = {}
_BINARY_SEQS_MAP = {}
_ANCHOR = ("", 0, {}, "")

OPERATORS = ["in3", "in2"]

DEFINITIONS_FOLDER = PROJECT_ROOT / 'files/definitions'

core_expression_map = {#"inN":  (1, {"1": "(1)", "N": "P(1)"}, "(x(1)P(1))", "(in[1,N])", 1, "inN[1]", False),
                       "in":   (2, {"1": "(1)", "2": "P(1)"}, "(x(1)P(1))", "(in[1,2])", 0, "in[1,2]", False),
                       "s":    (2, {"1": "(1)", "2": "(1)", "s": "P(x(1)(1))"}, "(x(1)(x(2)P(x(1)(2))))",
                                "(in2[1,2,s])", 1, "s[1,2]", False),
                       "=":    (2, {"1": "(1)", "2": "(1)"}, "(x(1)(1))", "(=[1,2])", 1, "=[1,2]", True),
                       "fXY":  (3, {"1": "P(x(1)(1))", "2": "P(1)", "3": "P(1)"}, "(xP(x(1)(1))(xP(1)P(1)))",
                                DEFINITIONS_FOLDER/"fXY.txt", 1, "fXY[f,X,Y]", False),
                       "fXYZ": (4, {"1": "P(x(1)(x(1)(1)))", "2": "P(1)", "3": "P(1)", "4": "P(1)"},
                                "(xP(x(1)(x(1)(1)))(xP(1)(xP(1)P(1))))",
                                DEFINITIONS_FOLDER/"fXYZ.txt", 1, "fXY[f,X,Y,Z]",
                                False),
                       "in2":  (3, {"1": "(1)", "2": "(1)", "3": "P(x(1)(1))"}, "(x(1)(x(1)P(x(1)(1))))",
                                "(in2[1,2,3])", 0, "in2[1,2,3]", False),
                       "in3":  (4, {"1": "(1)", "2": "(1)", "3": "(1)", "4": "P(x(1)(x(1)(1)))"},
                                "(x(1)(x(1)(x(1)P(x(1)(x(1)(1))))))", "(in3[1,2,3,4])", 0, "in3[1,2,3,4]", True),
                       "NaturalNumbers": (6, {"1": "P(1)", "2": "(1)", "3": "(1)", "4": "P(x(1)(1))",
                                              "5": "P(x(1)(x(1)(1)))", "6": "P(x(1)(x(1)(1)))"}, "",
                                          DEFINITIONS_FOLDER/"NaturalNumbers.txt", 1,
                                          "NaturalNumbers[N,i0,i1,s,+,*]", False)}

expression_def_set_map = {
                            #"(in[1,2])": (core_expression_map["in"], 1, "(in["),
                            "(in2[1,2,3])": (core_expression_map["in2"], 2, "(in2["),
                            "(in3[1,2,3,4])": (core_expression_map["in3"], 5, "(in3[")
                            #"(=[1,2])": (core_expression_map["="], 1, "=")
                        }

anchor = ("(NaturalNumbers[1,2,3,4,5,6])", core_expression_map["NaturalNumbers"][0],
          core_expression_map["NaturalNumbers"][1], "NaturalNumbers")

exclude_as_first_expression = {"(inN[1])"}
# expression_def_set_map["fXY[1,2,3]"] = core_expression_map["fXY"]
# expression_def_set_map["fXYZ[1,2,3,4]"] = core_expression_map["fXYZ"]

depth_counter = 0


# Function to read tree description from a .txt file
def read_tree_from_file(file_path):
    with open(file_path, 'r') as file:
        tree_str = file.read().strip()
    return tree_str


def mapping_good(mapping: dict):
    result = 1

    reversed_mapping = {}
    for key in mapping:
        value = mapping[key]
        if value in reversed_mapping:
            reversed_mapping[value] = min(reversed_mapping[value], key)
        else:
            reversed_mapping[value] = key

    for value in reversed_mapping:
        if reversed_mapping[value] != value:
            result = 0

    return result

def mapping_good2(mapping: dict, size_first: int):
    result = 1

    for arg in range(1, size_first + 1):
        if mapping[arg] != arg:
            result = 0
            break

    for arg in range(size_first + 1, len(mapping) + 1):
        for arg2 in range(arg + 1, len(mapping) + 1):
            if mapping[arg] == mapping[arg2]:
                result = 0
                break

    return result


def generate_binary_sequences_as_lists(n):
    """
    Generate all possible binary sequences of length n as lists of integers.

    Parameters:
    n (int): The length of the binary sequences.

    Returns:
    list: A list of binary sequences, where each sequence is a list of integers.
    """
    """"
    if n <= 0:
        return []

    lst = [
        sequence
        for sequence in [[int(bit) for bit in bin(i)[2:].zfill(n)] for i in range(2 ** n)]
        if not all(bit == 1 for bit in sequence)
    ]
    """
    if n == 0:
        return []

    lst = [[int(bit) for bit in bin(i)[2:].zfill(n)] for i in range(2 ** n)]

    return lst


def generate_mappings(m, n):
    # Create a range of numbers from 1 to m
    domain = range(1, m + 1)
    # Create a range of numbers from 1 to n
    codomain = range(1, n + 1)
    # Generate all possible mappings using the Cartesian product
    all_mappings = list(itertools.product(codomain, repeat=m))
    # Convert each mapping to a dictionary format for better readability
    mappings_as_dicts = [dict(zip(domain, mapping)) for mapping in all_mappings]

    filtered_mappings = []
    for mapping in mappings_as_dicts:
        if mapping_good(mapping):
            filtered_mappings.append(mapping)

    return filtered_mappings


def generate_definition_mappings(m, n):
    # Create a range of numbers from 1 to m
    domain = range(1, m + 1)
    # Create a range of numbers from 1 to n
    codomain = range(1, n + 1)
    # Generate all possible mappings using the Cartesian product
    all_mappings = list(itertools.product(codomain, repeat=m))
    # Convert each mapping to a dictionary format for better readability
    mappings_as_dicts = [dict(zip(domain, mapping)) for mapping in all_mappings]

    return mappings_as_dicts


class TreeNode1:
    def __init__(self, value, number_leafs):
        self.value = value
        self.number_leafs = number_leafs
        self.left = None
        self.right = None
        self.arguments = set()


class TreeNode2:
    def __init__(self, value, mp, fe, nl):
        self.value = value
        self.map = mp
        self.full_expression = fe
        self.number_leafs = nl
        self.left = None
        self.right = None


def repetitions_exist(s: str) -> bool:
    """
    Finds all substrings that:
      - start with '('
      - end with ')'
      - do NOT contain '(' or ')' in between
    Checks whether any such substring is repeated in the input string.

    Returns:
      True if there is at least one repeated parenthesized substring (no nesting),
      False otherwise.
    """
    # This regex will match:
    #    '(' + (zero or more characters that are NOT '(' or ')') + ')'
    # Example matches: "(abc)", "(>[])", "([1,2,3])"
    # It will NOT match something like "(abc(def))" because it contains bracket inside.
    substrings = re.findall(r"\([^()]*\)", s)

    # If any parenthesized substring (with no extra parens inside) appears more than once,
    # we'll detect it by comparing the list to its set.

    return len(substrings) != len(set(substrings))



def compare_trees(root1: TreeNode1, root2: TreeNode1) -> bool:
    """
    Compares two trees (root1 and root2) for equality.
    Returns True if they are the same structure, False otherwise.
    """

    # 1. Both nodes are None => same
    if root1 is None and root2 is None:
        return True

    # 2. One is None and the other is not => different
    if root1 is None or root2 is None:
        return False

    # 3. Compare current node data
    if root1.value != root2.value:
        return False
    if root1.number_leafs != root2.number_leafs:
        return False
    if root1.arguments != root2.arguments:
        return False

    # 4. Recursively compare children
    return (compare_trees(root1.left, root2.left) and
            compare_trees(root1.right, root2.right))




# Function to parse the binary tree description
def parse_expr_new(expr):
    expr = ''.join(expr.split())
    index = 0
    label_stack = []
    node_stack = []

    while True:
        if expr[index] == '(':
            index = index + 1
            if expr[index] == '>':
                index = index + 1
                node_label = '>'
                args_to_remove = get_args(expr[index:])
                index = expr[index:].find(']') + index + 1
                node_label = node_label + "[" + ",".join(args_to_remove) + "]"
                label_stack.append(node_label)
            elif expr[index] == '&':
                index = index + 1
                node_label = '&'
                label_stack.append(node_label)
            else:
                end_index = expr.find(')', index)
                if end_index == -1:
                    raise RuntimeError("No closing ')'.")
                node_label = expr[index:end_index]
                index = end_index + 1
                subexpr = extract_expression(node_label)
                node = TreeNode1("", 0)
                node.arguments.update(get_args(node_label))
                node.number_leafs = core_expression_map[subexpr][0]
                node.value = node_label
                node_stack.append(node)
        elif expr[index] == '!(':
            index = index + 1
            if expr[index] == '>':
                index = index + 1
                node_label = '!>'
                args_to_remove = get_args(expr[index:])
                index = expr[index:].find(']') + index + 1
                node_label = node_label + "[" + ",".join(args_to_remove) + "]"
                label_stack.append(node_label)
            elif expr[index] == '&':
                index = index + 1
                node_label = '!&'
                label_stack.append(node_label)
            else:
                end_index = expr.find(')', index)
                if end_index == -1:
                    raise RuntimeError("No closing ')'.")
                node_label = expr[index:end_index]
                node_label = "!(" + node_label + ")"
                index = end_index + 1
                subexpr = extract_expression(node_label)
                node = TreeNode1("", 0)
                node.arguments.update(get_args(node_label))
                node.number_leafs = core_expression_map[subexpr][0]
                node.value = node_label
                node_stack.append(node)
        else:
            assert expr[index] == ")"
            index = index + 1
            node = TreeNode1("", 0)
            node.right = node_stack.pop()
            node.left = node_stack.pop()
            node.number_leafs = node.left.number_leafs + node.right.number_leafs
            node.arguments.update(node.left.arguments)
            node.arguments.update(node.right.arguments)
            node.value = label_stack.pop()
            node.arguments.difference_update(get_args(node.value))

            node_stack.append(node)

            if index == len(expr):
                break

    root = node_stack.pop()
    root_number_leafs = root.number_leafs

    """
    root_old, root_number_leafs_old = parse_expr_old(expr)
    if not compare_trees(root, root_old):
        test = 0
    """

    return root, root_number_leafs


# Function to parse the binary tree description
def parse_expr(tree_str):
    tree_str = tree_str.replace("\n", "")  # Remove spaces
    tree_str = tree_str.replace(" ", "")  # Remove spaces
    tree_str = tree_str.replace("\t", "")  # Remove spaces
    index = 0

    # Recursively parse the tree string
    def parse_subtree(s):
        nonlocal index

        if not s:
            raise RuntimeError("Input 's' cannot be empty. Execution terminated.")

        node = TreeNode1("", 0)
        node_label = ""
        node_number_leafs = 0

        if s[index] == '(':
            index = index + 1
            if s[index] == '>':
                index = index + 1
                node_label = node_label + '>'
                args_to_remove = get_args(s[index:])
                index = s[index:].find(']') + index + 1
                node_label = node_label + "[" + ",".join(args_to_remove) + "]"
                node.left, left_node_number_leafs = parse_subtree(s)  # Process the left child
                node.right, right_node_number_leafs = parse_subtree(s)  # Process the right child
                node_number_leafs = left_node_number_leafs + right_node_number_leafs
                if node.left is not None:
                    node.arguments.update(node.left.arguments)
                if node.right is not  None:
                    node.arguments.update(node.right.arguments)
                node.arguments.difference_update(get_args(node_label))

            elif s[index] == '&':
                index = index + 1
                node_label = node_label + '&'
                node.left, left_node_number_leafs = parse_subtree(s)  # Process the left child
                node.right, right_node_number_leafs = parse_subtree(s)  # Process the right child
                node_number_leafs = left_node_number_leafs + right_node_number_leafs
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
                expr = extract_expression(node_label)
                node_number_leafs = core_expression_map[expr][0]
                node.arguments.update(get_args(node_label))



        elif s[index:index + 2] == "!(":
            index = index + 2
            if s[index] == '>':
                index = index + 1
                node_label = node_label + "!>"
                args_to_remove = get_args(s[index:])
                index = s[index:].find(']') + index + 1
                node_label = node_label + "[" + ",".join(args_to_remove) + "]"
                node.left, left_node_number_leafs = parse_subtree(s)  # Process the left child
                node.right, right_node_number_leafs = parse_subtree(s)  # Process the right child
                node_number_leafs = left_node_number_leafs + right_node_number_leafs
                if node.left is not None:
                    node.arguments.update(node.left.arguments)
                if node.right is not  None:
                    node.arguments.update(node.right.arguments)
                node.arguments.difference_update(get_args(node_label))

            elif s[index] == '&':
                index = index + 1
                node_label = node_label + "!&"
                node.left, left_node_number_leafs = parse_subtree(s)  # Process the left child
                node.right, right_node_number_leafs = parse_subtree(s)  # Process the right child
                node_number_leafs = left_node_number_leafs + right_node_number_leafs
                if node.left is not None:
                    node.arguments.update(node.left.arguments)
                if node.right is not  None:
                    node.arguments.update(node.right.arguments)

            else:
                end_index = s.find(')', index)
                if end_index == -1:
                    raise RuntimeError("No closing ')'.")
                node_label = s[index:end_index]
                node_label = "!(" + node_label + ")"
                index = end_index
                expr = extract_expression_from_negation(node_label)
                node_number_leafs = core_expression_map[expr][0]
                node.arguments.update(get_args(node_label))
        elif s[index] == ")":
            index -= 1

        index = index + 1
        node.value = node_label
        node.number_leafs = node_number_leafs
        if node.value == "":
            node = None
        return node, node_number_leafs

    root, root_number_leafs = parse_subtree(tree_str)
    return root, root_number_leafs



def modify_integers_in_string_by_subtraction(s, num_to_subtract):
    # Find all occurrences of square-bracketed sections with integers separated by commas
    bracketed_sections = re.findall(r'\[([\d,]+)]', s)

    # Iterate through each section found within the brackets
    for section in bracketed_sections:
        # Split the section by commas to get individual integers
        integers = section.split(',')

        # Subtract the number from each integer
        modified_integers = [str(int(i) - num_to_subtract) for i in integers]

        # Join the modified integers back with commas
        new_section = ','.join(modified_integers)

        # Replace the old section with the new section in the string
        s = s.replace(f'[{section}]', f'[{new_section}]', 1)

    return s


def modify_integers_in_string_by_mapping(s, mp):
    # Find all occurrences of square-bracketed sections with integers separated by commas
    bracketed_sections = re.findall(r'\[([\d,]+)]', s)

    # Iterate through each section found within the brackets
    for section in bracketed_sections:
        # Split the section by commas to get individual integers
        integers = section.split(',')

        # Subtract the number from each integer
        modified_integers = [mp[i] if i in mp else i for i in integers]

        # Join the modified integers back with commas
        new_section = ','.join(modified_integers)

        # Replace the old section with the new section in the string
        s = s.replace(f'[{section}]', f'[{new_section}]', 1)

    return s


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


def find_min_max_numbers(s: str):
    # Use regex to find all numbers inside parentheses
    numbers = re.findall(r'\((\d+)\)', s)

    # Convert the found strings to integers
    numbers = [int(num) for num in numbers]

    # If no numbers are found, return None for min and max
    if not numbers:
        return None, None

    # Find the minimum and maximum values
    min_number = min(numbers)
    max_number = max(numbers)

    return min_number, max_number


def parse_def_set(s: str):
    index = 0

    # Recursively parse the tree string
    def parse_subexpr():
        nonlocal index
        node_leaf_ids_list = []

        if not s:
            raise RuntimeError("Input 's' cannot be empty. Execution terminated.")

        node = TreeNode1("", 0)
        node_label = ""

        if s[index] == '(':
            index = index + 1
            if s[index] == 'x':
                index = index + 1
                node_label = node_label + 'x'
                node.left, left_node_leaf_ids_list = parse_subexpr()  # Process the left child
                node.right, right_node_leaf_ids_list = parse_subexpr()  # Process the right child
                node_leaf_ids_list.extend(left_node_leaf_ids_list)
                node_leaf_ids_list.extend(right_node_leaf_ids_list)
            else:
                end_index = s.find(')', index)
                if end_index == -1:
                    raise RuntimeError("No closing ')'.")
                node_label = s[index:end_index]
                node_leaf_ids_list.append(int(node_label))
                index = end_index

        elif s[index:index + 2] == "P(":
            index = index + 2
            if s[index] == 'x':
                index = index + 1
                node_label = node_label + "P(x)"
                node.left, left_node_leaf_ids_list = parse_subexpr()  # Process the left child
                node.right, right_node_leaf_ids_list = parse_subexpr()  # Process the right child
                node_leaf_ids_list.extend(left_node_leaf_ids_list)
                node_leaf_ids_list.extend(right_node_leaf_ids_list)
            else:
                end_index = s.find(')', index)
                if end_index == -1:
                    raise RuntimeError("No closing ')'.")
                node_label = s[index:end_index]
                node_leaf_ids_list.append(int(node_label))
                node_label = "P(" + node_label + ")"
                index = end_index

        index = index + 1
        node.value = node_label
        return node, node_leaf_ids_list

    root, root_leaf_ids_list = parse_subexpr()
    return root, root_leaf_ids_list


def tree_to_str_reorder(root):
    local_def_set = ""
    counter = 1

    def node_to_str(node):
        nonlocal local_def_set
        nonlocal counter

        if node.value == "x":
            local_def_set = local_def_set + "(x"
        elif node.value == "P(x)":
            local_def_set = local_def_set + "P(x"
        elif node.value[0] == 'P':
            temp_int = int(counter)
            local_def_set = local_def_set + "P(" + str(temp_int)
            counter = counter + 1
        else:
            temp_int = int(counter)
            local_def_set = local_def_set + "(" + str(temp_int)
            counter = counter + 1

        if node.left is not None:
            node_to_str(node.left)
        if node.right is not None:
            node_to_str(node.right)
        local_def_set = local_def_set + ")"

    node_to_str(root)
    return local_def_set


def tree_to_str(root, offset, substitution_map):
    local_def_set = ""

    def node_to_str(node):
        nonlocal local_def_set

        if node.value == "x":
            local_def_set = local_def_set + "(x"
        elif node.value == "P(x)":
            local_def_set = local_def_set + "P(x"
        elif node.value[0] == 'P':
            contents = re.findall(r'\((.*?)\)', node.value)
            temp_int = int(contents[0]) - offset
            if temp_int in substitution_map:
                temp_int = substitution_map[temp_int]
            local_def_set = local_def_set + "P(" + str(temp_int)
        else:
            temp_int = int(node.value) - offset
            if temp_int in substitution_map:
                temp_int = substitution_map[temp_int]
            local_def_set = local_def_set + "(" + str(temp_int)

        if node.left is not None:
            node_to_str(node.left)
        if node.right is not None:
            node_to_str(node.right)
        local_def_set = local_def_set + ")"

    node_to_str(root)
    return local_def_set


def update_replacement_map(replacement_map, num1, num2):
    connected = set()
    connected.add(num1)
    connected.add(num2)

    new_connected = set()
    stay = 1
    while stay:
        for arg in connected:
            if arg in replacement_map:
                if replacement_map[arg] not in connected:
                    new_connected.add(replacement_map[arg])
        if len(new_connected) > 0:
            connected.update(new_connected)
            new_connected.clear()
        else:
            stay = 0

    min_val = min(connected)

    for arg in connected:
        if arg != min_val:
            replacement_map[arg] = min_val


def def_sets_equal(def_set1, def_set2):
    temp1, root_leaf_ids_list1 = reorder_numbers(def_set1)
    temp2, root_leaf_ids_list2 = reorder_numbers(def_set2)

    replacement_map = {}

    if temp1 == temp2:
        for leaf_id1 in root_leaf_ids_list1:
            for leaf_id2 in root_leaf_ids_list2:
                update_replacement_map(replacement_map, leaf_id1, leaf_id2)

    return temp1 == temp2, replacement_map


def subtract_and_replace_numbers(s: str, subtract_value: int, m: dict):
    root, root_leaf_ids_list = parse_def_set(s)

    new_str = tree_to_str(root, subtract_value, m)

    return new_str, root_leaf_ids_list


def reorder_numbers(s: str):
    root, root_leaf_ids_list = parse_def_set(s)

    new_str = tree_to_str_reorder(root)

    return new_str, root_leaf_ids_list


def find_all_ids(s):
    # Use regular expressions to find all integer numbers in the string
    numbers = re.findall(r'\((\d+)\)', s)

    # Convert them to integers and filter those greater than n
    bigger_numbers = set([int(num) for num in numbers])

    return bigger_numbers


def shift_together(arg_def_set_map):
    id_set = set()
    for arg in arg_def_set_map:
        id_set.update(find_all_ids(arg_def_set_map[arg]))
    id_list = list(id_set)

    replacement_map = {}
    for ind in range(0, len(id_list)):
        replacement_map[id_list[ind]] = ind + 1

    for arg in arg_def_set_map:
        arg_def_set_map[arg], ids_list = subtract_and_replace_numbers(arg_def_set_map[arg], 0, replacement_map)


def connect_expression_sets(set1, set2, connection_type, is_definition, args_to_remove, after_grooming):
    replacement_map = {}
    success = 1
    removed_args = []

    global_max = 0

    for tple in set1:
        local_min, local_max = find_min_max_numbers(tple[1])
        global_max = max(global_max, local_max)

    temp_set = set()
    for tple in set2:
        val, root_leaf_ids_list = subtract_and_replace_numbers(tple[1], -global_max, identity)
        temp_set.add((tple[0], val))

    temp_set.update(set1)

    bool_set = set()
    for tple1 in temp_set:
        bool_set.add(tple1)
        for tple2 in temp_set:
            if tple1[0] == tple2[0] and tple1 != tple2 and tple2 not in bool_set:
                bool_set.add(tple2)
                equality, small_replacement_map = def_sets_equal(tple1[1], tple2[1])
                if equality:
                    for key in small_replacement_map:
                        update_replacement_map(replacement_map, key, small_replacement_map[key])
                else:
                    success = 0

    temp_set2 = set()
    for tple in temp_set:
        val, root_leaf_ids_list = subtract_and_replace_numbers(tple[1], 0, replacement_map)
        temp_set2.add((tple[0], val))

    common_map = {}
    for tple in temp_set2:
        common_map[tple[0]] = tple[1]
    if connection_type == ">" and success == 1:
        for tple1 in set1:
            for tple2 in set2:
                if tple1[0] == tple2[0] and tple1[0] in common_map and tple1[0] in args_to_remove:
                    del common_map[tple1[0]]
                    removed_args.append(tple1[0])
                    if tple1[1][0] == 'P' and is_definition == 0:
                        success = 0

    """
    if connection_type == "&" and success == 1:
        arg_set1 = set()
        arg_set2 = set()
        for tple1 in set1:
            arg_set1.add(tple1[0])
        for tple2 in set2:
            arg_set2.add(tple2[0])
        if arg_set1 != arg_set2:
            success = 0
    """

    if not after_grooming:
        if set(args_to_remove) != set(removed_args):
            success = 0
    shift_together(common_map)
    common_set = set()
    for key in common_map:
        common_set.add((key, common_map[key]))

    return common_map, common_set, success, removed_args


def extract_between_brackets(s, start_index=0):
    start = s.find('[', start_index)
    end = s.find(']', start)

    if start != -1 and end != -1:
        return s[start + 1:end]
    return None


def get_args(expr):
    sub_expr = extract_between_brackets(expr, 0)

    if sub_expr is None:
        test = 0

    if sub_expr == "":
        return []
    else:
        return sub_expr.split(',')


def extract_expression(s: str) -> str:
    # Find the index of the first occurrence of '['
    index = s.find('[')

    # If '[' is found, return the substring from index 0 to the found index
    if index != -1:
        if s[0] == "(":
            return s[1:index]
        else:
            return s[0:index]

    # If '[' is not found, return the entire string
    return ""


def extract_expression_from_negation(s: str) -> str:
    # Find the index of the first occurrence of "!("
    start_index = s.find("!(")

    # Find the index of the first occurrence of "["
    end_index = s.find("[")

    # Ensure both "!(" and "[" are found and that "!(" appears before "["
    if start_index != -1 and end_index != -1 and start_index < end_index:
        # Extract the substring, excluding "!(" and "["
        return s[start_index + 2:end_index]

    # If the conditions are not met, return an empty string or an appropriate message
    return ""




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


def find_position_surrounded(text, substring):
    sub_len = len(substring)
    pos = text.find(substring)

    while pos != -1:
        # We only consider this a valid position if:
        # - There's a preceding '[' or ',' (unless pos=0 which automatically fails)
        # - AND there's a trailing ']' or ',' (unless we're at the end)
        if (pos > 0 and text[pos - 1] in ['[', ',']) \
           and (pos + sub_len < len(text) and text[pos + sub_len] in [']', ',']):
            return pos

        pos = text.find(substring, pos + 1)

    return -1



def sort_list_according_to_occurrence(lst, text):
    tple_list = []

    for arg in lst:
        pos = find_position_surrounded(text, arg)
        if pos == -1:
            raise ValueError(f"Argument '{arg}' not found in the expression")
        tple_list.append((arg, pos))

    # Sort the list of tuples based on the position
    tple_list.sort(key=lambda x: x[1])
    return [item[0] for item in tple_list]





def find_ordered_integers(int_strings, big_string):
    # Filter only those integer strings that appear in big_string
    found_integers = [s for s in int_strings if s in big_string]

    # Sort found integers by their first appearance in big_string
    found_integers.sort(key=lambda x: big_string.index(x))

    return found_integers


def replace_integer_in_string(big_string, target_int, replacement_int):
    # Use regex to find the target integer as a whole word (to avoid partial matches)
    target_pattern = r'\b{}\b'.format(re.escape(target_int))  # escape in case of special characters
    result_string = re.sub(target_pattern, replacement_int, big_string)
    return result_string


def rename_args(expr, args, first_int_to_use):
    new_expr = expr
    repl_int = first_int_to_use

    ordered_integers = find_ordered_integers(args, expr)
    for arg in ordered_integers:
        new_expr = replace_integer_in_string(new_expr, arg, str(repl_int))
        repl_int = repl_int + 1

    next_int_to_use = repl_int

    return new_expr, next_int_to_use


def subtract_number_from_ints(expr: str, number: int, numbers_to_replace: Set[int], replace_all: bool) -> str:
    """
    Subtracts a specified number from integers in a string based on a set of keys.

    Args:
        expr (str): The input string containing integers enclosed in brackets and separated by commas.
        number (int): The number to subtract from each matching integer.
        numbers_to_replace (List[int]): A list of integers to be replaced. If empty, all integers are replaced.
        replace_all (bool): replace all

    Returns:
        str: The modified string with specified integers subtracted by the given number.
    """
    if not expr:
        return expr  # Early return if the input string is empty





    pattern = r'(?<=[\[,])(\d+)(?=[\],])'

    # Precompile the regex for better performance
    regex = re.compile(pattern)

    def replacer(match: re.Match) -> str:
        # Extract the matched number as a string
        old_number_str = match.group(1)
        old_number = int(old_number_str)

        # Determine if the number should be replaced
        if replace_all or old_number in numbers_to_replace:
            new_number = old_number - number
            return str(new_number)

        # Return the original number if no replacement is needed
        return old_number_str

    # Perform the substitution using the compiled regex and the replacer function
    result = regex.sub(replacer, expr)
    return result


def clean_expr(expr: str):
    # Regular expression to match substrings starting with >[ and ending with ], without internal [ or ]
    pattern = r">\[[^\[\]]*\]"

    # Removing matching substrings
    output_string = re.sub(pattern, ">[]", expr)

    return output_string


def order_by_pattern(input_str: str, arg_set: Set[str]) -> List[str]:
    """
    Orders the keys in arg_set based on their first occurrence in the input_str
    within specific contexts: surrounded by '[' and ']', ',', and combinations thereof.

    Args:
        input_str (str): The original string containing keys.
        arg_set (Set[str]): A set of keys to search for and order.

    Returns:
        List[str]: A list of keys from arg_set ordered by their first occurrence.
    """
    # Step 1: Clean the input string using the provided clean_expr function
    cleaned_expr = clean_expr(input_str)

    if not arg_set:
        return []  # No keys to process

    # Step 2: Escape all keys to handle special regex characters
    escaped_keys = [re.escape(key) for key in arg_set]

    # Step 3: Create a single regex pattern to match any key in the specified contexts
    # The pattern uses a capturing group to identify which key was matched
    # Contexts:
    # 1. Surrounded by '[' and ']'
    # 2. Surrounded by ',' and ','
    # 3. Surrounded by ',' and ']'
    # 4. Surrounded by '[' and ','
    # The pattern ensures that only exact matches in these contexts are captured
    pattern = r'(?<=[\[,])(' + '|'.join(escaped_keys) + r')(?=[\],])'

    # Step 4: Precompile the regex for better performance
    regex = re.compile(pattern)

    # Step 5: Initialize a dictionary to store the first occurrence index of each key
    first_occurrence: dict = {}

    # Step 6: Iterate over all matches in the cleaned_expr
    for match in regex.finditer(cleaned_expr):
        key = match.group(1)
        # Record the first occurrence index if not already recorded
        if key not in first_occurrence:
            first_occurrence[key] = match.start(1)
            # Early exit if all keys have been found
            if len(first_occurrence) == len(arg_set):
                break

    # Step 7: Sort the keys based on their first occurrence index
    # Only include keys that were found
    ordered_keys = sorted(first_occurrence.items(), key=lambda x: x[1])

    # Step 8: Extract and return the ordered list of keys
    return [key for key, _ in ordered_keys]



def find_arg_map(expr: str):
    expr = expr.replace("\n", "")  # Remove spaces
    expr = expr.replace(" ", "")  # Remove spaces
    expr = expr.replace("\t", "")  # Remove spaces
    index = 0


    def find_arg_map_core():
        nonlocal index
        node_set = set()
        node_map = {}

        if expr[index] == '(':
            index = index + 1
            if expr[index] == '>':
                index = index + 1
                args_to_remove = get_args(expr[index:])
                index = expr[index:].find(']') + index + 1
                left_map = find_arg_map_core()

                right_map = find_arg_map_core()


                node_map = {**left_map, **right_map}
                for arg in args_to_remove:
                    if arg not in node_map:
                        test = 0
                    del node_map[arg]


            elif expr[index] == '&':
                index = index + 1
                left_map = find_arg_map_core()
                right_map = find_arg_map_core()

                node_map = {**left_map, **right_map}
            else:
                end_index = expr.find(')', index)
                if end_index == -1:
                    raise RuntimeError("No closing ')'.")
                node_label = expr[index:end_index]
                array_args = get_args(node_label)
                temp_expr = extract_expression(node_label)

                expr_map = core_expression_map[temp_expr][1]
                for i in range(len(array_args)):
                    node_set.add((array_args[i], expr_map[str(i + 1)]))
                    node_map[array_args[i]] = expr_map[str(i + 1)]
                for arg in expr_map:
                    if not arg.isdigit():
                        node_set.add((arg, expr_map[arg]))
                        node_map[arg] = expr_map[arg]

                index = end_index
        elif expr[index:index + 2] == "!(":
            index = index + 2
            if expr[index] == '>':
                index = index + 1
                args_to_remove = get_args(expr[index:])
                index = expr[index:].find(']') + index + 1
                left_map = find_arg_map_core()


                right_map = find_arg_map_core()



                node_map = {**left_map, **right_map}
                for arg in args_to_remove:
                    del node_map[arg]





            elif expr[index] == '&':
                index = index + 1
                left_map, left_full_expression = find_arg_map_core()
                right_map, right_full_expression = find_arg_map_core()

                node_map = {**left_map, **right_map}
            else:
                end_index = expr.find(')', index)
                if end_index == -1:
                    raise RuntimeError("No closing ')'.")
                node_label = expr[index:end_index]
                temp_expr = extract_expression(node_label)
                node_label = "!(" + node_label + ")"
                array_args = get_args(node_label)
                expr_map = core_expression_map[temp_expr][1]
                for i in range(len(array_args)):
                    node_set.add((array_args[i], expr_map[str(i + 1)]))
                    node_map[array_args[i]] = expr_map[str(i + 1)]
                for arg in expr_map:
                    if not arg.isdigit():
                        node_set.add((arg, expr_map[arg]))
                        node_map[arg] = expr_map[arg]

                index = end_index
        elif expr[index] == ")":
            index -= 1

        index = index + 1
        return node_map

    root_map = find_arg_map_core()

    return root_map


def rename_variables_in_expr(expr:str, deep: bool):
    expr = expr.replace("\n", "")  # Remove spaces
    expr = expr.replace(" ", "")  # Remove spaces
    expr = expr.replace("\t", "")  # Remove spaces
    index = 0
    numbers_to_replace = set()
    first_int_to_use = 10000
    unchanged_first_int_to_use = first_int_to_use
    replacement_map = {}

    def create_replacement_map(subexpr, args_to_remove, nums_to_replace):
        replacement_map1 = {}
        nonlocal first_int_to_use
        nonlocal unchanged_first_int_to_use

        ordered_args1 = order_by_pattern(subexpr, args_to_remove)

        for arg in ordered_args1:
            while True:
                if str(first_int_to_use - (unchanged_first_int_to_use - 1)) not in original_arg_map:
                    replacement_map1[arg] = str(first_int_to_use)
                    nums_to_replace.add(first_int_to_use)
                    first_int_to_use += 1
                    break
                else:
                    first_int_to_use += 1

        return replacement_map1, ordered_args1

    def rename_variables_in_subexpr():
        full_expression = ""
        nonlocal index
        nonlocal first_int_to_use
        nonlocal original_arg_map
        node_set = set()
        node_map = {}
        node_success = 0
        nonlocal replacement_map

        if expr[index] == '(':
            index = index + 1
            if expr[index] == '>':
                index = index + 1
                args_to_remove = get_args(expr[index:])
                index = expr[index:].find(']') + index + 1
                left_map, left_full_expression = rename_variables_in_subexpr()

                replacement_map2, ordered_ints2 = \
                    create_replacement_map(left_full_expression, args_to_remove, numbers_to_replace)
                replacement_map.update(replacement_map2)

                right_map, right_full_expression = rename_variables_in_subexpr()

                left_full_expression = \
                    replace_keys_in_string(left_full_expression, replacement_map2)
                right_full_expression = \
                    replace_keys_in_string(right_full_expression, replacement_map2)

                node_map = {**left_map, **right_map}
                for arg in args_to_remove:
                    del node_map[arg]

                renamed_args_to_remove = [replacement_map2[str(item)] for item in ordered_ints2]

                temp_list = ["(>", "[", ",".join(renamed_args_to_remove), "]", left_full_expression,
                             right_full_expression, ")"]
                full_expression = "".join(temp_list)

            elif expr[index] == '&':
                index = index + 1
                left_map, left_full_expression = rename_variables_in_subexpr()
                right_map, right_full_expression = rename_variables_in_subexpr()

                node_map = {**left_map, **right_map}

                if node_success:
                    full_expression = "(&" + left_full_expression + right_full_expression + ")"

            else:
                end_index = expr.find(')', index)
                if end_index == -1:
                    raise RuntimeError("No closing ')'.")
                node_label = expr[index:end_index]
                array_args = get_args(node_label)
                temp_expr = extract_expression(node_label)

                expr_map = core_expression_map[temp_expr][1]
                for i in range(len(array_args)):
                    node_set.add((array_args[i], expr_map[str(i + 1)]))
                    node_map[array_args[i]] = expr_map[str(i + 1)]
                for arg in expr_map:
                    if not arg.isdigit():
                        node_set.add((arg, expr_map[arg]))
                        node_map[arg] = expr_map[arg]

                index = end_index
                full_expression = "".join(["(", node_label, ")"])
        elif expr[index:index + 2] == "!(":
            index = index + 2
            if expr[index] == '>':
                index = index + 1
                args_to_remove = get_args(expr[index:])
                index = expr[index:].find(']') + index + 1
                left_map, left_full_expression = rename_variables_in_subexpr()

                replacement_map2, ordered_ints2 = \
                    create_replacement_map(left_full_expression, args_to_remove, numbers_to_replace)
                replacement_map.update(replacement_map2)

                right_map, right_full_expression = rename_variables_in_subexpr()

                left_full_expression = \
                    replace_keys_in_string(left_full_expression, replacement_map2)
                right_full_expression = \
                    replace_keys_in_string(right_full_expression, replacement_map2)

                node_map = {**left_map, **right_map}
                for arg in args_to_remove:
                    del node_map[arg]

                renamed_args_to_remove = [replacement_map2[str(item)] for item in ordered_ints2]

                temp_list = ["!(>", "[", ",".join(renamed_args_to_remove), "]", left_full_expression,
                             right_full_expression, ")"]
                full_expression = "".join(temp_list)



            elif expr[index] == '&':
                index = index + 1
                left_map, left_full_expression = rename_variables_in_subexpr()
                right_map, right_full_expression = rename_variables_in_subexpr()

                node_map = {**left_map, **right_map}

                if node_success:
                    full_expression = "!(&" + left_full_expression + right_full_expression + ")"

            else:
                end_index = expr.find(')', index)
                if end_index == -1:
                    raise RuntimeError("No closing ')'.")
                node_label = expr[index:end_index]
                temp_expr = extract_expression(node_label)
                node_label = "!(" + node_label + ")"
                array_args = get_args(node_label)
                expr_map = core_expression_map[temp_expr][1]
                for i in range(len(array_args)):
                    node_set.add((array_args[i], expr_map[str(i + 1)]))
                    node_map[array_args[i]] = expr_map[str(i + 1)]
                for arg in expr_map:
                    if not arg.isdigit():
                        node_set.add((arg, expr_map[arg]))
                        node_map[arg] = expr_map[arg]

                index = end_index
                full_expression = node_label
        elif expr[index] == ")":
            index -= 1

        index = index + 1
        return node_map, full_expression

    if not deep:
        original_arg_map = find_arg_map(expr)
    else:
        original_arg_map = {}

    root_map, renamed_expr = rename_variables_in_subexpr()

    if deep:
        args = set(root_map.keys())
        replacement_map3, ordered_ints = create_replacement_map(renamed_expr, args, numbers_to_replace)
        replacement_map.update(replacement_map3)
        renamed_expr = replace_keys_in_string(renamed_expr, replacement_map3)
        renamed_expr =\
            subtract_number_from_ints(renamed_expr, unchanged_first_int_to_use - 1, numbers_to_replace, False)

        temp_map = {}
        for arg in root_map:
            if arg in replacement_map3:
                temp_map[str(int(replacement_map3[arg]) - (unchanged_first_int_to_use - 1))] = root_map[arg]
            else:
                temp_map[arg] = root_map[arg]
        root_map = temp_map
    else:
        renamed_expr =\
            subtract_number_from_ints(renamed_expr, unchanged_first_int_to_use - 1, numbers_to_replace, False)

    for arg in replacement_map:
        replacement_map[arg] = str(int(replacement_map[arg]) - (unchanged_first_int_to_use - 1))

    return renamed_expr, root_map, replacement_map

# Function to parse the binary tree description
def parse_tree(tree_str, definition_type, replace, after_grooming):
    tree_str = tree_str.replace("\n", "")  # Remove spaces
    tree_str = tree_str.replace(" ", "")  # Remove spaces
    tree_str = tree_str.replace("\t", "")  # Remove spaces
    index = 0
    arg_set = set()
    arg_list = []
    first_int_to_use = 10000
    unchanged_first_int_to_use = first_int_to_use
    temp_str = tree_str[:]
    numbers_to_replace = set()

    # Recursively parse the tree string
    def parse_subtree(s):
        nonlocal index
        node_set = set()
        node_map = {}
        node_success = 0
        node_number_leafs = 0
        nonlocal arg_set
        nonlocal arg_list
        nonlocal first_int_to_use
        global core_expression_map

        node = TreeNode2("", "", "", 0)
        node_label = ""
        full_expression = ""

        if s[index] == '(':
            index = index + 1
            if s[index] == '>':
                index = index + 1
                node_label = node_label + '>'
                args_to_remove = get_args(s[index:])
                index = s[index:].find(']') + index + 1
                node_label = node_label + "[" + ",".join(args_to_remove) + "]"
                replacement_map = {}
                for arg in args_to_remove:
                    replacement_map[arg] = str(first_int_to_use)
                    numbers_to_replace.add(first_int_to_use)
                    first_int_to_use = first_int_to_use + 1

                renamed_args_to_remove = [replacement_map.get(item, item) for item in args_to_remove]
                node.left, left_set, left_success, full_expression_left, left_node_number_leafs = parse_subtree(
                    s)  # Process the left child
                if replace:
                    full_expression_left = \
                        replace_keys_in_string(full_expression_left, replacement_map)
                node.right, right_set, right_success, full_expression_right, right_node_number_leafs = parse_subtree(
                    s)  # Process the right child
                if replace:
                    full_expression_right = \
                        replace_keys_in_string(full_expression_right, replacement_map)
                node_number_leafs = left_node_number_leafs + right_node_number_leafs
                if left_success and right_success:
                    node_map, node_set, node_success, removed_args = connect_expression_sets(left_set, right_set, '>',
                                                                                             definition_type == 1,
                                                                                             args_to_remove,
                                                                                             after_grooming)
                    if replace:
                        full_expression = "(>" + "[" + ",".join(
                            renamed_args_to_remove) + "]" + full_expression_left + full_expression_right + ")"
                    else:
                        full_expression = "(>" + "[" + ",".join(
                            args_to_remove) + "]" + full_expression_left + full_expression_right + ")"
                    # if len(removed_args) == 0:
                    # node_success = 0
                    if full_expression_left == full_expression_right:
                        node_success = 0
            elif s[index] == '&':
                index = index + 1
                node_label = node_label + '&'
                node.left, left_set, left_success, full_expression_left, left_node_number_leafs = parse_subtree(
                    s)  # Process the left child
                node.right, right_set, right_success, full_expression_right, right_node_number_leafs = parse_subtree(
                    s)  # Process the right child
                node_number_leafs = left_node_number_leafs + right_node_number_leafs
                if left_success and right_success:
                    node_map, node_set, node_success, removed_args = connect_expression_sets(left_set,
                                                                                             right_set,
                                                                                             '&',
                                                                                             definition_type == 1,
                                                                                             [],
                                                                                             after_grooming)
                    full_expression = "(&" + full_expression_left + full_expression_right + ")"
            else:
                end_index = s.find(')', index)
                if end_index == -1:
                    raise RuntimeError("No closing ')'.")
                node_label = s[index:end_index]
                array_args = get_args(node_label)
                for arg in array_args:
                    if arg not in arg_set:
                        arg_set.add(arg)
                        arg_list.append(arg)
                expr = extract_expression(node_label)
                node_number_leafs = core_expression_map[expr][0]

                expr_map = core_expression_map[expr][1]
                for i in range(len(array_args)):
                    node_set.add((array_args[i], expr_map[str(i + 1)]))
                    node_map[array_args[i]] = expr_map[str(i + 1)]
                for arg in expr_map:
                    if not arg.isdigit():
                        node_set.add((arg, expr_map[arg]))
                        node_map[arg] = expr_map[arg]

                node_success = 1
                index = end_index
                full_expression = "(" + node_label + ")"

        elif s[index:index + 2] == "!(":
            index = index + 2
            if s[index] == '>':
                index = index + 1
                node_label = node_label + '>'
                args_to_remove = get_args(s[index:])
                index = s[index:].find(']') + index + 1
                node_label = node_label + "[" + ",".join(args_to_remove) + "]"
                replacement_map = {}
                for arg in args_to_remove:
                    replacement_map[arg] = str(first_int_to_use)
                    numbers_to_replace.add(first_int_to_use)
                    first_int_to_use = first_int_to_use + 1

                renamed_args_to_remove = [replacement_map.get(item, item) for item in args_to_remove]
                node.left, left_set, left_success, full_expression_left, left_node_number_leafs = parse_subtree(
                    s)  # Process the left child
                if replace:
                    full_expression_left = \
                        replace_keys_in_string(full_expression_left, replacement_map)
                node.right, right_set, right_success, full_expression_right, right_node_number_leafs = parse_subtree(
                    s)  # Process the right child
                if replace:
                    full_expression_right = \
                        replace_keys_in_string(full_expression_right, replacement_map)
                node_number_leafs = left_node_number_leafs + right_node_number_leafs
                if left_success and right_success:
                    node_map, node_set, node_success, removed_args = connect_expression_sets(left_set, right_set, '>',
                                                                                             definition_type == 1,
                                                                                             args_to_remove,
                                                                                             after_grooming)
                    if replace:
                        full_expression = "!(>" + "[" + ",".join(
                            renamed_args_to_remove) + "]" + full_expression_left + full_expression_right + ")"
                    else:
                        full_expression = "!(>" + "[" + ",".join(
                            args_to_remove) + "]" + full_expression_left + full_expression_right + ")"

                    if len(removed_args) == 0:
                        node_success = 0
                    if full_expression_left == full_expression_right:
                        node_success = 0
            elif s[index] == '&':
                index = index + 1
                node_label = node_label + "!&"
                node.left, left_set, left_success, full_expression_left, left_node_number_leafs = parse_subtree(
                    s)  # Process the left child
                node.right, right_set, right_success, full_expression_right, right_node_number_leafs = parse_subtree(
                    s)  # Process the right child
                node_number_leafs = left_node_number_leafs + right_node_number_leafs
                if left_success and right_success:
                    node_map, node_set, node_success, removed_args = connect_expression_sets(left_set, right_set, '&',
                                                                                             definition_type == 1, [],
                                                                                             after_grooming)
                    full_expression = "!(&" + full_expression_left + full_expression_right + ")"
            else:
                end_index = s.find(')', index)
                if end_index == -1:
                    raise RuntimeError("No closing ')'.")
                node_label = s[index:end_index]
                expr = extract_expression(node_label)
                node_label = "!(" + node_label + ")"
                array_args = get_args(node_label)
                for arg in array_args:
                    if arg not in arg_set:
                        arg_set.add(arg)
                        arg_list.append(arg)
                node_number_leafs = core_expression_map[expr][0]

                expr_map = core_expression_map[expr][1]
                for i in range(len(array_args)):
                    node_set.add((array_args[i], expr_map[str(i + 1)]))
                    node_map[array_args[i]] = expr_map[str(i + 1)]
                for arg in expr_map:
                    if not arg.isdigit():
                        node_set.add((arg, expr_map[arg]))
                        node_map[arg] = expr_map[arg]

                node_success = 1
                index = end_index
                full_expression = node_label

        index = index + 1
        node.value = node_label
        node.map = str(node_map)
        node.full_expression = full_expression
        node.number_leafs = node_number_leafs
        return node, node_set, node_success, full_expression, node_number_leafs

    root, root_set, root_success, root_full_expression, root_number_leafs = parse_subtree(temp_str)

    if replace and root_success:
        root_full_expression = subtract_number_from_ints(root_full_expression, unchanged_first_int_to_use - 1,
                                                         numbers_to_replace, False)

    root_map = {}
    for tple in root_set:
        root_map[tple[0]] = tple[1]

    return root, root_map, root_success, arg_list, root_number_leafs, root_full_expression


def connect_expressions(expr1: str,
                        expr2: str,
                        map1: dict,
                        map2: dict,
                        substitution_map: dict,
                        binary_list: list,
                        connect_to_anchor: bool):

    def check_maps(atr: [],
                    mp1: dict,
                    mp2: dict):
        check_passed = True

        for arg in mp1:
            if arg in mp2:
                if mp1[arg] != mp2[arg]:
                    return False

        if not connect_to_anchor:
            for arg in atr:
                if mp1[arg][0] == "P":
                    return False

        return check_passed

    shift_num = max({int(arg) for arg in map1.keys()})

    left_map = {}
    removable_args = set()
    for arg in map1.keys():
        new_arg = substitution_map[arg]
        left_map[new_arg] = map1[arg]

    right_map = {}
    for arg in map2.keys():
        if str(int(arg) + shift_num) not in substitution_map:
            test = 0
        new_arg = substitution_map[str(int(arg) + shift_num)]
        right_map[new_arg] = map2[arg]
        if new_arg in map1.keys():
            removable_args.add(new_arg)


    removable_args_list = list(removable_args)
    sorted_list = sort_list_according_to_occurrence(removable_args_list, expr1)
    args_to_remove = []
    for index in range(len(binary_list)):
        if binary_list[index]:
            if index not in range(len(sorted_list)):
                test = 0
            args_to_remove.append(sorted_list[index][0])

    """
    connected_map, connected_set, success, removed_args = (
        connect_expression_sets(set1, set2, ">", 0, args_to_remove, 0))
    """


    connected_map = {**left_map, **right_map}
    for arg in args_to_remove:
        del connected_map[arg]
    success = True


    if not check_maps(args_to_remove, left_map, right_map):
        success = False

    """
    new_expr2, root2 = subtract_and_replace_numbers_in_expr(expr2, -shift_num, identity)
    new_expr1, root1 = subtract_and_replace_numbers_in_expr(expr1, 0, substitution_map)
    new_expr2, root2 = subtract_and_replace_numbers_in_expr(new_expr2, 0, substitution_map)
    """

    new_expr2 = subtract_number_from_ints(expr2, -shift_num, set(), True)
    new_expr1 = replace_keys_in_string(expr1, substitution_map)
    new_expr2 = replace_keys_in_string(new_expr2, substitution_map)

    connected_expr = "(>[" + ",".join(args_to_remove) + "]"
    connected_expr += new_expr1
    connected_expr += new_expr2
    connected_expr += ")"

    if new_expr1 == new_expr2:
        success = False

    """"
    if not connect_to_definition:
        new_expr1, unused_arg_map1 = rename_variables_in_expr(new_expr1, True)
        connected_expr, connected_map = rename_variables_in_expr(connected_expr, True)
    else:
        new_expr2, unused_arg_map2 = rename_variables_in_expr(new_expr2, True)
    """

    return success, connected_expr, connected_map


def get_number_removable_args(mapping: dict):

    return len(mapping.keys()) - len(set(mapping.values()))







def expr_good(expr: str):
    good = False

    if expr[:3] == "(>[" and expr[:4] != "(>[]":
        good = True

    if repetitions_exist(expr):
        good = False

    return good

def expr_good3(expr: str, to_exclude):
    good = True

    for subexpr in to_exclude:
        if subexpr in expr:
            good = False

    return good


def check_def_sets(arg_map: dict):
    check_positive = True
    counter_map = {}

    for arg in arg_map:
        if arg_map[arg] in counter_map:
            counter_map[arg_map[arg]] += 1
        else:
            counter_map[arg_map[arg]] = 1

    for def_set in counter_map:
        if counter_map[def_set] > max_values_for_def_sets[def_set]:
            return False

    return check_positive

def check_complexity_level_for_def_sets(arg_map: dict, complexity_level: int):
    check_positive = True
    def_sets = set()

    for arg in arg_map:
        def_sets.add(arg_map[arg])

    for def_set in def_sets:
        if max_complexity_if_anchor_parameter_connected[def_set] < complexity_level:
            return False

    return check_positive


def evaluate_operator_exprs2(operator_exprs2: List[str], anchor_attached: bool):
    arg_map = {}
    evaluation_positive = True

    arg_lists_list = []
    for operator_expr in operator_exprs2:
        args = get_args(operator_expr)
        arg_lists_list.append(args)

    position = 0
    for arg_list in arg_lists_list:
        for arg_ind in range(len(arg_list) - 1):
            arg = arg_list[arg_ind]
            if arg in arg_map:
                if arg_ind == len(arg_list) - 2:
                    arg_map[arg][1].add(position)
                else:
                    arg_map[arg][0].add(position)
            else:
                arg_map[arg] = []
                arg_map[arg].append(set())
                arg_map[arg].append(set())
                if arg_ind == len(arg_list) - 2:
                    arg_map[arg][1].add(position)
                else:
                    arg_map[arg][0].add(position)
        position += 1

    for arg in arg_map:
        if len(arg_map[arg][0]) > 0 and len(arg_map[arg][1]) == 0:
            continue
        elif len(arg_map[arg][0]) == 0 and len(arg_map[arg][1]) <= 2:
            if len(operator_exprs2) - 1 in arg_map[arg][1]:
                if len(operator_exprs2) == max_number_simple_expressions or anchor_attached:
                    if len(arg_map[arg][1]) < 2:
                        evaluation_positive = False
                    else:
                        continue
                else:
                    continue
            else:
                evaluation_positive = False
        elif len(arg_map[arg][0]) == 1 and len(arg_map[arg][1]) == 1:
            continue
        else:
            evaluation_positive = False
    return evaluation_positive

def extract_operator_expressions(operators: List[str], expr2: str) -> List[str]:
    """
    Extracts substrings from expr2 that start with any of the specified operators
    followed by "[" and end with "]". The substrings are returned in the order
    they appear in expr2.

    Args:
        operators (List[str]): A list of operator strings.
        expr2 (str): The expression string to search within.

    Returns:
        List[str]: A list of matching substrings in order of occurrence.
    """

    if not operators:
        return []

    # Sort operators by length in descending order to handle overlapping operators
    sorted_ops = sorted(operators, key=lambda op: -len(op))

    # Escape operators to safely include them in the regex pattern
    escaped_ops = [re.escape(op) for op in sorted_ops]

    # Create a regex pattern that matches any operator followed by "[" and then any non-"]" characters until "]"
    pattern = r'(' + '|'.join(escaped_ops) + r')\[[^\]]*\]'

    # Compile the regex for better performance if the function is called multiple times
    regex = re.compile(pattern)

    # Find all non-overlapping matches in the expression
    matches = regex.finditer(expr2)

    # Extract the matched substrings
    result = [match.group(0) for match in matches]

    return result

def expr_good2(expr: str,
               number_simple_expressions,
               connected_map):


    def evaluate_operator_exprs(operator_exprs2: List[str], there_are_free_args):
        evaluation_positive = True
        global OPERATORS

        last_expr = operator_exprs2[len(operator_exprs2) - 1]
        last_expr_args = get_args(last_expr)
        last_arg = last_expr_args[len(last_expr_args) - 2]

        occurrence_counter = 0
        for ind in range(len(operator_exprs2) - 1):
            args = get_args(operator_exprs2[ind])
            if last_arg in args:
                occurrence_counter += 1
                if args[len(args) - 2] != last_arg:
                    evaluation_positive = False



        if (len(operator_exprs2) >= operator_threshold or
                (len(operator_exprs2) == (operator_threshold - 1) and there_are_free_args)):
            for last_expr_arg_ind in range(len(last_expr_args) - 2):
                occurrence_counter = 0
                for operator_expr_ind in range(len(operator_exprs2) - 1):
                    args = get_args(operator_exprs2[operator_expr_ind])
                    if last_expr_args[last_expr_arg_ind] in args:
                        occurrence_counter += 1
                        if args[len(args) - 2] != last_expr_args[last_expr_arg_ind]:
                            evaluation_positive = False
                if occurrence_counter > 1:
                    evaluation_positive = False

        return evaluation_positive





    good = True
    global OPERATORS

    #expr = "(>[b,c,d](in3[b,c,d,+])(>[a,e](in3[a,d,e,*])(>[f](in3[a,b,f,*])(>[g](in3[a,c,g,*])(in3[f,g,e,+])))))"
    #expr = "(>[1,2,3,4](in3[1,2,3,4])(>[5,6,7](in2[5,6,7])(>[8](NaturalNumbers[9,10,7,4,8])(>[](in3[1,3,6,8])(in3[6,2,5,8])))))"
    #connected_map = {}

    #expr = '(>[4,5](in2[4,5,2])(in3[5,4,1,3]))'

    #expr = '(>[b,c,d](in3[b,c,d,+])(>[a,e](in3[a,d,e,+])(>[f](in3[a,b,f,+])(in3[f,c,e,+]))))'

    if repetitions_exist(expr):
        good = False
        return good

    if not numbers_good(expr):
        good = False
        return good


    size_args = len({arg for arg in connected_map if connected_map[arg][:1] != 'P' })
    if (number_simple_expressions == max_number_simple_expressions and
            not check_def_sets(connected_map)):
        good = False
        return good


    operator_exprs = extract_operator_expressions(OPERATORS, expr)
    good = evaluate_operator_exprs(operator_exprs, size_args > 0) and good
    good = evaluate_operator_exprs2(operator_exprs, False) and good

    return good

def expression_is_simple(expr: str):
    result = True

    if expr[0:2] == "(>" or expr[0:3] == "!(>":
        result = False


    return result

def numbers_good(expr: str):
    passed = True

    for element in expression_def_set_map:
        number = expr.count(expression_def_set_map[element][2])
        if number > expression_def_set_map[element][1]:
            passed = False
            break

    return passed


def generate_all_permutations(n):
    if n < 0:
        return {}  # Return an map if n < 0

    # Generate permutations for all i from 0 to n-1
    all_permutations = {0: []}
    for i in range(1, n + 1):
        all_permutations[i] = [list(p) for p in permutations(range(i))]
    return all_permutations

def disintegrate_implication(expr_for_desintegration, chain):
    head = ""

    root, root_number_leafs = parse_expr(expr_for_desintegration)

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

def extract_key_value(expr2: str):
    root, root_number_leafs = parse_expr(expr2)
    value = ""

    node = root
    while True:
        if node is not None:
            if node.value[0] == ">":
                node = node.right
            else:
                value = tree_to_expr(node)
                break
        else:
            break

    if value == "":
        key = expr2
    else:
        parts = expr2.rsplit(value, 1)
        key = ''.join(parts)

    return key, value

def sort_first_two_args(expr_for_arg_sorting):
    def sort_first_two_in_brackets(input_string, prefix):
        """
        Finds all instances of "(prefix[" - it will be followed "])" later -
        in the input string, sorts the first two
        elements inside the brackets lexicographically, and replaces the original
        substring with the sorted version.

        Args:
            input_string (str): The string to process.
            prefix (str): The variable prefix to search for (e.g., '+', 'foo').

        Returns:
            str: The processed string with sorted elements.
        """
        # Escape the prefix to handle any special regex characters
        escaped_prefix = re.escape(prefix)

        # Compile the regex pattern dynamically based on the prefix
        # Removed redundant escape before ']'
        pattern = re.compile(r'\(' + escaped_prefix + r'\[(.*?)]')


        def replace_match(match):
            # Extract the content inside the brackets
            content = match.group(1)
            # Split the content by comma and strip any surrounding whitespace
            items = [item.strip() for item in content.split(',')]

            if len(items) >= 2:
                # Sort the first two elements as integers if needed
                if int(items[0]) > int(items[1]):
                    items[0], items[1] = items[1], items[0]
            # No else needed; if fewer than two items, no sorting required

            # Join the items back into a string separated by commas
            new_content = ','.join(items)
            # Return the modified substring with "(prefix[" and "]"
            return f'({prefix}[{new_content}]'

        # Use re.sub with the replacement function to process all matches
        processed_string = pattern.sub(replace_match, input_string)
        return processed_string

    complexity = count_operator_occurrences_regex(expr_for_arg_sorting)
    if complexity <= max_complexity_for_commutative_law:
        return expr_for_arg_sorting

    sorted_output = expr_for_arg_sorting[:]

    for core_expr in core_expression_map:
        if core_expression_map[core_expr][6]:
            sorted_output = sort_first_two_in_brackets(sorted_output, core_expr)

    return sorted_output

def reshuffle(expr: str, perms: [], deep: bool):


    def create_last_occurrence_map(chn: [], perm: []):
        lst_occrrnce_mp = {}

        assert len(chn) == len(perm)

        all_removed_args = set()
        for ind in range(len(chn)):
            subexpr, removed_args, remaining_args = chn[perm[ind]]
            all_removed_args.update(removed_args)

        for ind in range(len(chn)):
            subexpr, removed_args, remaining_args = chn[perm[ind]]

            for arg in all_removed_args:
                if arg in remaining_args:
                    if arg not in lst_occrrnce_mp:
                        lst_occrrnce_mp[arg] = ind
                    else:
                        lst_occrrnce_mp[arg] = min(lst_occrrnce_mp[arg], ind)

        return lst_occrrnce_mp

    def build_removed_args_lists(lst_occrrnce_mp: {}, size_list: int):
        rmvd_args_lsts = [[] for _ in range(size_list)]

        for arg in lst_occrrnce_mp:
            rmvd_args_lsts[lst_occrrnce_mp[arg]].append(arg)

        return rmvd_args_lsts

    def make_reshuffled_expression(perm: [], rmvd_args_lst: []):
        nonlocal  head

        rshffld = head[:]

        for ind in range(len(perm) - 1, -1, -1):
            substr = "[" + ",".join(rmvd_args_lst[ind]) + "]"
            rshffld = "(>" + substr + chain[perm[ind]][0] + rshffld + ")"

        return rshffld





    #expr = "(>[b,c,d](+[b,c,d])(>[a,e](*[a,d,e])(>[f](*[a,b,f])(>[g](*[a,c,g])(+[f,g,e])))))"

    chain = []
    head = ""

    min_reshuffled, min_arg_map, min_replacement_map = rename_variables_in_expr(expr, deep)

    head = disintegrate_implication(expr, chain)
    for permutation in perms[len(chain)]:
        last_occurrence_map = create_last_occurrence_map(chain, permutation)
        removed_args_lists = build_removed_args_lists(last_occurrence_map, len(chain))
        reshuffled = make_reshuffled_expression(permutation, removed_args_lists)
        reshuffled, arg_map, replacement_map = rename_variables_in_expr(reshuffled, deep)

        if min_reshuffled > reshuffled:
            min_reshuffled = reshuffled
            min_arg_map = copy.copy(arg_map)
            min_replacement_map = copy.copy(replacement_map)


    #min_reshuffled = sort_first_two_args(min_reshuffled)

    return min_reshuffled, min_arg_map, min_replacement_map


def init_pool(mappings_map, binary_seqs_map, all_permutations):
    global _MAPPINGS_MAP
    global _BINARY_SEQS_MAP
    global _ALL_PERMUTATIONS
    global _ANCHOR

    _MAPPINGS_MAP = mappings_map
    _BINARY_SEQS_MAP = binary_seqs_map
    _ALL_PERMUTATIONS = all_permutations
    _ANCHOR = anchor

def count_operator_occurrences_regex(s: str) -> int:
    """
    Count occurrences of the constant substring "(>[" in the input string `s`
    using a regex lookahead to include overlapping matches.

    :param s: The string where the search is performed.
    :return: The count of occurrences of "(>["
    """
    # Define the regex pattern using a lookahead.
    # The lookahead '(?=(\(\>\[))' checks for the occurrence of literal "(>["
    # We escape '(' and '[' as they are special characters in regex.
    pattern = r"(?=(\(\>\[))"
    matches = re.findall(pattern, s)
    return len(matches)

def stays_output_variable(full_expr: str, output_variable: str):
    core_expr = extract_expression(full_expr)
    args = get_args(full_expr)

    if core_expr == "in":
        if args[0] == output_variable:
            return True
    if core_expr == "in2":
        if args[1] == output_variable:
            return True
    if core_expr == "in3":
        if args[2] == output_variable:
            return True

    return False

def prioritize_anchor(chain: list[str], anchor: str) -> None:
    """
    Modify `chain` in place by finding the first element that contains `anchor`
    and moving it to index 0. If no element contains `anchor`, `chain` is left unchanged.

    :param chain: List of strings to search through.
    :param anchor: Substring to look for.
    """
    for i, s in enumerate(chain):
        if anchor in s:
            # remove it from its current position and insert at front
            chain.insert(0, chain.pop(i))
            break


def create_reshuffled_mirrored(expr: str, perms, anchor_first = False):
    temp_chain = []

    #expr = '(>[1,2,3](NaturalNumbers[1,2,3,5,6])(>[4](in[4,1])(in2[4,2,3])))'

    head = disintegrate_implication(expr, temp_chain)

    head_args = get_args(head)
    head_expr = extract_expression(head)

    output_variable = ""
    if head_expr == "in":
        output_variable = head_args[0]

    if head_expr == "in2":
        output_variable = head_args[1]

    if head_expr == "in3":
        output_variable = head_args[2]

    assert output_variable != ""

    alternative = ""
    chain = []
    for element in temp_chain:
        if stays_output_variable(element[0], output_variable):
            alternative = element[0]
        else:
            chain.append(element[0])

    if anchor_first:
        prioritize_anchor(chain, anchor[3])

    if alternative == "":
        return ""
    assert alternative != ""

    chain.append(head)
    chain.append(alternative)

    args_to_remove = set()
    for element in temp_chain:
        args_to_remove.update(element[1])

    args_chain = []
    for element in chain:
        args_chain.append(set(get_args(element)))

    how_to_remove = [[] for _ in range(len(chain) - 1)]

    for arg_to_remove in args_to_remove:
        for index in range(len(chain)):
            if arg_to_remove in args_chain[index]:


                how_to_remove[index].append(arg_to_remove)
                break

    new_expr = chain[len(chain) - 1]
    for ind in range(len(chain) - 2, -1, -1):
        substr = "[" + ",".join(how_to_remove[ind]) + "]"
        new_expr = "(>" + substr + chain[ind] + new_expr + ")"

    if anchor_first:
        reshuffled_expr = new_expr
    else:
        reshuffled_expr, reshuffled_map, replacement_map = reshuffle(new_expr, perms, True)

    return reshuffled_expr





def contains_excluded_expression1(s: str) -> bool:
    """
    Return True if `s` contains a substring of the form:
        (in2[<integer>,3,4])
    where <integer> is one or more digits.
    """
    return bool(pattern_to_exclude1.search(s))

def single_thread_calculation(statement: str,
                              growing_theorem: str,
                              number_simple_expressions_statement: int,
                              number_simple_expressions_growing_theorem: int,
                              args_statement: {},
                              args_growing_theorem: {}):
    def make_all_connection_maps(args_map1: Dict[str, Any], args_map2: Dict[str, Any]) -> List[Dict[str, Any]]:
        def union_of_dicts(dicts):
            """
            Merge a sequence of dictionaries into one.
            In case of key conflicts, later dictionaries overwrite earlier ones.
            """
            result = {}
            for d in dicts:
                result.update(d)
            return result

        def create_union_maps(list_of_lists):
            """
            Given a list of lists of dictionaries, produce all merged maps.
            """
            results = []
            for selection in product(*list_of_lists):
                merged = union_of_dicts(selection)
                results.append(merged)
            return results

        mappings_list = []
        src_map: Dict[str, set] = {}
        dst_map: Dict[str, set] = {}

        # Deterministically compute shift number by extracting digits from keys
        shift_num = max(
            int(re.search(r"\d+", arg).group())
            for arg in args_map2.keys()
            if re.search(r"\d+", arg)
        )  # CHANGED: use regex to handle non-integer literal keys

        # Build source and destination groupings
        for arg, val in args_map1.items():
            src_map.setdefault(val, set()).add(arg)
        for arg, val in args_map2.items():
            dst_map.setdefault(val, set()).add(arg)

        # Process destination-only sets (sorted lexicographically for determinism)
        for def_set in sorted(dst_map):  # CHANGED: removed key=int to avoid ValueError
            if def_set not in src_map:
                args = sorted(dst_map[def_set])  # CHANGED: lexicographic sort
                mappings_list.append([
                    {element: element for element in args}
                ])

        # Process overlapping and source-only sets
        for def_set in sorted(src_map):  # CHANGED: removed key=int
            if def_set in dst_map:
                dst_args = sorted(dst_map[def_set])  # CHANGED: lexicographic sort
                src_args = sorted(src_map[def_set])  # CHANGED: lexicographic sort
                # Apply shift to source args
                shifted_src = [
                    str(int(re.search(r"\d+", x).group()) + shift_num) if re.search(r"\d+", x) else x
                    for x in src_args
                ]  # CHANGED: safe extraction with regex
                args = dst_args + shifted_src

                mappings = _MAPPINGS_MAP[
                    len(dst_args) + len(src_args)
                    ][(len(dst_args), len(src_args))]

                block = []
                for mapping in mappings:
                    temp_map = {
                        args[i]: args[mapping[i + 1] - 1]
                        for i in range(len(args))
                    }
                    block.append(temp_map)
                mappings_list.append(block)
            else:
                src_args = sorted(src_map[def_set])  # CHANGED: lexicographic sort
                shifted_src = [
                    str(int(re.search(r"\d+", x).group()) + shift_num) if re.search(r"\d+", x) else x
                    for x in src_args
                ]
                mappings_list.append([
                    {element: element for element in shifted_src}
                ])

        # Combine all blocks into full connection maps
        all_connection_maps = create_union_maps(mappings_list)
        return all_connection_maps

    list_args = [key for key in args_statement.keys() if args_statement[key][0] != "P"]
    list_args.extend([arg for arg in args_growing_theorem.keys() if args_growing_theorem[arg][0] != "P"])
    number_digits_both = len(list_args)
    number_sets_both = (len(args_statement) + len(args_growing_theorem)) - number_digits_both

    connected_list = []
    connected_list2 = []
    reshuffled_list = []
    reshuffled_mirrored_list = []

    if max(number_digits_both, number_sets_both) > max_size_mapping_def_set:
        return connected_list, connected_list2, reshuffled_list, reshuffled_mirrored_list

    connection_maps = make_all_connection_maps(args_growing_theorem, args_statement)



    for connection_map in connection_maps:
        number_removable_args = get_number_removable_args(connection_map)

        for binary_list in _BINARY_SEQS_MAP[number_removable_args]:
            if all(x == 0 for x in binary_list):
                continue

            success, connected_expr, connected_map = \
                connect_expressions(statement,
                                    growing_theorem,
                                    args_statement,
                                    args_growing_theorem,
                                    connection_map,
                                    binary_list,
                                    False)

            if success:
                number_simple_expressions = (number_simple_expressions_statement +
                                             number_simple_expressions_growing_theorem)







                if not expr_good2(connected_expr, number_simple_expressions, connected_map):

                    """
                    reshuffled_expr, reshuffled_map, replacement_map = reshuffle(connected_expr, _ALL_PERMUTATIONS,True)
                    if reshuffled_test_expr2 == reshuffled_expr:
                        test = 0
                        expr_good2(connected_expr, number_simple_expressions, connected_map)
                    """

                    continue



                reshuffled_expr, reshuffled_map, replacement_map = reshuffle(connected_expr, _ALL_PERMUTATIONS, True)


                test_expr2 = "(>[n,m](in3[n,1,m,+])(in2[n,m,s]))"

                reshuffled_test_expr2, reshuffled_test_map2, replacement_test_map2 = \
                    reshuffle(test_expr2, _ALL_PERMUTATIONS, True)

                if reshuffled_test_expr2 == reshuffled_expr:
                    test = 0





                """
                if reshuffled_test_expr2 == reshuffled_expr:
                    test = 0
                """

                connected_list.append((reshuffled_expr, reshuffled_map))

                complexity_level = count_operator_occurrences_regex(reshuffled_expr) + 1

                if (check_def_sets(reshuffled_map) and len(reshuffled_map) <= max_number_args_expr and
                        check_complexity_level_for_def_sets(reshuffled_map, complexity_level)):
                    connection_maps2 = make_all_connection_maps(reshuffled_map, _ANCHOR[2])

                    for connection_map2 in connection_maps2:

                        if connection_map2 == {'1': '1', '10': '5', '11': '4', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '9': '3'}:
                            test = 2

                        to_continue = False
                        for ky in connection_map2:

                            mp = _ANCHOR[2]
                            if int(ky) > len(mp):
                                if connection_map2[ky] == ky:
                                    to_continue = True
                        if to_continue:
                            continue

                        binary_list2 = [1 for _ in range(len(reshuffled_map))]
                        success2, connected_expr2, connected_map2 = \
                            connect_expressions(_ANCHOR[0],
                                                reshuffled_expr,
                                                _ANCHOR[2],
                                                reshuffled_map,
                                                connection_map2,
                                                binary_list2,
                                                True)

                        if success2:
                            #connected_expr2 = sort_first_two_args(connected_expr2)

                            test_expr = "(>[0,s,+](NaturalNumbers[N,0,s,+,*])(>[1](in2[0,1,s])(>[n,m](in3[n,1,m,+])(in2[n,m,s]))))"

                            operator_exprs = extract_operator_expressions(OPERATORS, connected_expr2)

                            #connected_expr2 = '(>[3,4,5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[](in2[7,3,4])(in3[3,8,9,5]))))'

                            if (not check_input_variables_theorem(connected_expr2) or
                                    not check_theorem_complexity_per_operator(connected_expr2) or
                                    not check_input_variables_order(connected_expr2, _ALL_PERMUTATIONS) or
                                    contains_excluded_expression1(connected_expr2)):
                                continue


                            if not evaluate_operator_exprs2(operator_exprs, True):
                                continue



                            connected_list2.append(connected_expr2)

                            reshuffled_expr2, reshuffled_map2, replacement_map2 = \
                                reshuffle(connected_expr2, _ALL_PERMUTATIONS, True)

                            """
                            reshuffled_test_expr, reshuffled_test_map, replacement_test_map = \
                                reshuffle(test_expr, _ALL_PERMUTATIONS, True)

                            if reshuffled_expr2 == reshuffled_test_expr:
                                test = 0
                            """


                            reshuffled_list.append(reshuffled_expr2)

                            reshuffled_mirrored = \
                                create_reshuffled_mirrored(connected_expr2, _ALL_PERMUTATIONS)

                            """
                            test_resh_m = create_reshuffled_mirrored(test_expr, _ALL_PERMUTATIONS)

                            if reshuffled_test_expr == reshuffled_expr:
                                test = 0

                            if test_resh_m == reshuffled_mirrored:
                                test = 0
                            """


                            reshuffled_mirrored_list.append(reshuffled_mirrored)
                            """
                            print("")
                            print(connected_expr2)
                            print(reshuffled_expr2)
                            print(reshuffled_mirrored)
                            print("")
                            """


    test = 0
    return connected_list, connected_list2, reshuffled_list, reshuffled_mirrored_list


def create_map(N: int) -> Dict[int, Dict[Tuple[int, int], List[Dict[int, int]]]]:
    """
    Creates a nested map structure based on the given specifications.

    Parameters:
    - N (int): An integer greater than or equal to 2.

    Returns:
    - Dict[int, Dict[Tuple[int, int], List[Dict[int, int]]]]: The nested map structure.
    """
    if N < 2:
        raise ValueError("N must be at least 2.")

    outer_map = {}

    for n in range(2, N + 1):
        M = {}
        for p in range(1, n):
            q = n - p
            S = list(range(p + 1, n + 1))  # Elements greater than p
            L = []

            # Generate all possible subsets T of S
            for r in range(len(S) + 1):
                for T in itertools.combinations(S, r):
                    # For subset T, generate all injective mappings to {1, ..., p}
                    if len(T) > p:
                        continue  # Cannot injectively map more elements than available targets
                    targets = list(range(1, p + 1))
                    for mapping in itertools.permutations(targets, len(T)):
                        Q = {}
                        # Assign Q(i) = i for 1 <= i <= p
                        for i in range(1, p + 1):
                            Q[i] = i
                        # Assign Q(i) for i in T
                        for idx, s in enumerate(T):
                            Q[s] = mapping[idx]
                        # Assign Q(i) = i for i not in T and i > p
                        for s in S:
                            if s not in T:
                                Q[s] = s
                        L.append(Q)
            M[(p, q)] = L
        outer_map[n] = M

    return outer_map



def find_entry_args(args_list: list[list[str]], args: list[str]):
    entry_args = set()

    for index  in range(0, len(args) - 1):
        arg = args[index]
        found = False
        for index in range(len(args_list)):
            if arg == args_list[index][-1]:
                found = True
                entry_args.update(find_entry_args(args_list, args_list[index]))

        if not found:
            entry_args.add(arg)

    return entry_args

def check_theorem_complexity_per_operator(theorem: str):
    result = True

    temp_chain = []
    head = disintegrate_implication(theorem, temp_chain)
    chain = []
    for element in temp_chain:
        chain.append(element[0])
    chain.append(head)


    for element in chain:
        core_expr = extract_expression(element)

        if core_expr in OPERATORS:


            if max_values_for_operators[core_expr] < len(chain):
                result = False

    return result

def check_input_variables_theorem(theorem: str):
    temp_chain = []
    head = disintegrate_implication(theorem, temp_chain)
    chain = []
    for element in temp_chain:
        chain.append(element[0])
    chain.append(head)

    zero_arg = ''
    one_arg = ''
    for element in chain:
        if anchor[3] in element:
            zero_arg = get_args(element)[1]
            one_arg = get_args(element)[2]
            chain.remove(element)

    last_expr = chain[len(chain) - 1]
    core_expr = extract_expression(last_expr)
    if core_expr not in OPERATORS:
        result = False
    else:
        args_list = []
        for element in chain:
            core_expr = extract_expression(element)
            assert core_expr in OPERATORS
            args = get_args(element)
            args_list.append(args[:len(args) - 1])

        last_expr_args = args_list[len(args_list) - 1]
        output_var = last_expr_args[len(last_expr_args) - 1]

        second_last_args = []
        for index, args in enumerate(args_list):
            if output_var in args:
                second_last_args = get_args(chain[index])
                second_last_args.pop()
                break

        entry_args = find_entry_args(args_list, last_expr_args)
        if zero_arg in entry_args:
            entry_args.remove(zero_arg)
        if one_arg in entry_args:
            entry_args.remove(one_arg)
        entry_args_second = find_entry_args(args_list, second_last_args)
        if zero_arg in entry_args_second:
            entry_args_second.remove(zero_arg)
        if one_arg in entry_args_second:
            entry_args_second.remove(one_arg)

        result = entry_args == entry_args_second

    return result

def find_digit_args_old(expr: str):
    def find_substrings(s: str, mapping: dict) -> set:
        """
        Given a string s and a mapping (dict) where keys are strings,
        find for each key all substrings in s that start with key followed by "[" and
        continue until the closing "]". The matching substring is expected to be preceded by "(",
        but the returned substring does not include that "(".

        For example, if a key is "in3", the function will capture a substring like
        "(in3[...]" but return "in3[...]" without the leading "(".

        Args:
            s (str): The input string.
            mapping (dict): A dictionary with string keys.

        Returns:
            set: A set of substrings matching key + "[...]" (without the preceding "(") for each key in mapping.
        """
        result = set()
        for key in mapping.keys():
            # Use a lookbehind assertion to ensure that the match is preceded by "(" but not include it in the result.
            pattern = r"(?<=\()" + re.escape(key) + r"\[[^]]*\]"
            matches = re.findall(pattern, s)
            result.update(matches)
        return result

    def map_arguments_with_position(s: str) -> dict:
        """
        Given a string in the format "keyA[arg1, arg2, ...]", this function extracts the
        arguments from inside the square brackets and returns a dictionary mapping each
        argument to its position in the list (starting with 1).

        Args:
            s (str): The input string.

        Returns:
            dict: A dictionary where keys are the arguments (as strings) and values are their positions.

        Raises:
            ValueError: If the string does not contain a valid bracketed argument list.
        """
        # Use a regex to extract the content inside the first pair of square brackets.
        match = re.search(r"\[([^]]+)]", s)
        if not match:
            raise ValueError("The input string does not contain a valid bracketed argument list.")

        # Get the arguments as a string, split by commas, and strip whitespace.
        args_str = match.group(1)
        args2 = [arg2.strip() for arg2 in args_str.split(',')]

        # Build and return a dictionary mapping each argument to its position (starting at 1).
        return {arg2: str(idx + 1) for idx, arg2 in enumerate(args2)}

    subexprs = find_substrings(expr, core_expression_map)

    zero_arg_name = ""
    for subexpr in subexprs:
        if anchor[3] in subexpr:
            anchor_args = get_args(subexpr)
            zero_arg_name = anchor_args[1]

    digit_args = set()
    for ky in core_expression_map:
        for subexpr in subexprs:
            core_expr = extract_expression(subexpr)
            if ky == core_expr:
                arg_map = map_arguments_with_position(subexpr)
                for arg in arg_map:
                    if core_expression_map[ky][1][arg_map[arg]][0] != "P":
                        digit_args.add(arg)

    for operator in OPERATORS:
        for subexpr in subexprs:
            core_expr = extract_expression(subexpr)
            if operator == core_expr:
                args3 = get_args(subexpr)
                assert len(args3) - 2 >= 0
                output = args3[len(args3) - 2]
                if output in digit_args:
                    digit_args.remove(output)


    if zero_arg_name in digit_args:
        digit_args.remove(zero_arg_name)
    return digit_args

def find_digit_args(theorem: str):
    temp_chain = []
    head = disintegrate_implication(theorem, temp_chain)
    chain = []
    for element in temp_chain:
        chain.append(element[0])
    chain.append(head)

    all_args = set()
    for element in chain:
        core_expression = extract_expression(element)

        if core_expression in OPERATORS:
            all_args.update(get_args(element))

    for element in chain:
        if anchor[3] in element:

            for arg in get_args(element):
                if arg in all_args:
                    all_args.remove(arg)

    for element in chain:
        core_expression = extract_expression(element)

        if core_expression in OPERATORS:

            args = get_args(element)
            for index in range(len(args) - 2, len(args)):
                if args[index] in all_args:
                    all_args.remove(args[index])

    return all_args



def get_left_right(chain: list[str], expression: str, digits: set[str]):
    left = set()
    right = set()

    core_expression = extract_expression(expression)
    if not core_expression == 'in3':
        test = 0
    assert core_expression == 'in3'

    args = get_args(expression)
    if args[0] in digits:
        left.add(args[0])
    else:
        for element in chain:
            core_expression_element = extract_expression(element)
            if core_expression_element == 'in3':
                args_element = get_args(element)

                if args_element[2] == args[0]:
                    left_element, right_element = get_left_right(chain, element, digits)
                    left.update(left_element)
                    left.update(right_element)

    if args[1] in digits:
        right.add(args[1])
    else:
        for element in chain:
            core_expression_element = extract_expression(element)
            if core_expression_element == 'in3':
                args_element = get_args(element)

                if args_element[2] == args[1]:
                    left_element, right_element = get_left_right(chain, element, digits)
                    right.update(left_element)
                    right.update(right_element)


    return left, right

def get_right_chain(chain: list[str], head: str):
    right_chain = [head]

    head_args = get_args(head)
    head_inputs = head_args[:len(head_args) - 2]

    for expression in chain:
        core_expr = extract_expression(expression)
        if core_expr not in OPERATORS:
            continue

        expr_args = get_args(expression)

        expr_output = expr_args[len(expr_args) - 2]

        if expr_output in head_inputs:
            right_chain.extend(get_right_chain(chain, expression))

    return right_chain

def get_left_right_chains(chain: list[str]):
    no_anchor_chain = chain[1:]
    right_chain = get_right_chain(no_anchor_chain, chain[-1])

    left_chain = [x for x in no_anchor_chain if x not in right_chain]

    return left_chain, right_chain

def check_input_variable_position(chain: list[str], digits: set[str]):
    result = True

    order_map = {}

    for expression in chain:
        core_expression = extract_expression(expression)

        if core_expression == 'in3':
            args = get_args(expression)

            for arg_ind in range(2):

                if args[arg_ind] in digits:
                    key = (args[3], args[arg_ind])

                    if key not in order_map:
                        order_map[key] = arg_ind
                    else:
                        if order_map[key] != arg_ind:
                            return False




    return result

def remove_outputs(chain: list[str]):
    removed = []
    replacement_map = {}

    for expr in chain:
        core_expr = extract_expression(expr)
        args = get_args(expr)

        if core_expr in OPERATORS:
            op_name = args[len(args) - 2]
            replacement_map[op_name] = ''

    removed = ''.join(chain)
    removed = replace_keys_in_string(removed, replacement_map)

    return removed

def check_tautology(left_chain: list[str], right_chain: list[str]):
    left_removed = remove_outputs(left_chain)
    right_removed = remove_outputs(right_chain)

    result = left_removed != right_removed

    return result

def check_functions(chain: list[str]):
    removed_set = set()
    result = True

    for expr in chain:
        core_expr = extract_expression(expr)

        if core_expr == 'in2':
            args = get_args(expr)
            output = args[len(args) - 2]

            replacement_map = {output: ''}
            removed = replace_keys_in_string(expr, replacement_map)

            if removed in removed_set:
                result = False
                break
            else:
                removed_set.add(removed)

    return result


def only_one_operator(chain: list[str]):
    result = True

    no_anchor = chain[1:]

    head = no_anchor[-1]
    head_args = get_args(head)

    head_key = (head_args[len(head_args) - 1], extract_expression(head))

    for expr in no_anchor:
        expr_args = get_args(expr)

        expr_key = (expr_args[len(expr_args) - 1], extract_expression(expr))

        if head_key != expr_key:
            result = False

    return result


def check_input_variables_order(theorem: str, permutation):
    result = True

    temp_chain = []
    head = disintegrate_implication(theorem, temp_chain)
    chain = []
    for element in temp_chain:
        chain.append(element[0])
    chain.append(head)

    digits = find_digit_args(theorem)

    order_map = {}
    order_set = set()

    for expression in chain:
        core_expression = extract_expression(expression)

        if core_expression == 'in3':
            args = get_args(expression)

            if args[0] in digits and args[1] in digits:
                order_map[(args[3], frozenset({args[0], args[1]}))] = [args[0], args[1]]

                key = frozenset({args[0], args[1]})

                if key in order_set:
                    result = False
                else:
                    order_set.add(key)



    for expression in chain:
        core_expression = extract_expression(expression)

        if core_expression == 'in3':
            args = get_args(expression)

            left, right = get_left_right(chain, expression, digits)

            operator = args[3]
            for left_arg in left:
                for right_arg in right:
                    key = (operator, frozenset({left_arg, right_arg}))
                    if key in order_map:
                        if order_map[key] != [left_arg, right_arg]:
                            result = False

    left_chain, right_chain = get_left_right_chains(chain)

    if result:
        result = (check_input_variable_position(left_chain, digits) and
                  check_input_variable_position(right_chain, digits))

    if result:
        result = check_tertiaries(left_chain, right_chain)


    if only_one_operator(chain):
        reshuffled = reshuffle(theorem,permutation, True)
        reshuffled_mirrored = create_reshuffled_mirrored(theorem, permutation)

        if reshuffled[0] == reshuffled_mirrored:
            result = True

    result = result and check_tautology(left_chain, right_chain)

    result = result and check_functions(chain)

    return result

def get_tertiaries(chain: list[str]):
    tertiaries = set()

    for expr in chain:
        if extract_expression(expr) == 'in3':
            args = get_args(expr)
            tertiaries.add(args[-1])

    return tertiaries

def check_tertiaries(left_chain: list[str], right_chain: list[str]):
    result = True

    left_tertiaries = get_tertiaries(left_chain)
    right_tertiaries = get_tertiaries(right_chain)

    if left_tertiaries and right_tertiaries:
        if not left_tertiaries & right_tertiaries:
            result = False

    return result


def create_expressions_parallel():
    expression_map = expression_def_set_map.copy()
    result_expr_set = set()
    reshuffled_expr_set = set()
    reshuffled_mirrored_expr_set = set()
    control_set = set()


    mappings_map = create_map(max_size_mapping_def_set)

    all_permutations = generate_all_permutations(max_number_simple_expressions + 1)

    binary_seqs_map = {}
    for num in range(0, max_size_binary_list):
        binary_seqs_map[num] = generate_binary_sequences_as_lists(num)

    expr_list = list(expression_map.keys())

    last_visited_map = {i: -1 for i in range(len(expr_list))}

    growing_theorems = [expr for expr in expr_list]
    growing_theorems_set = set(growing_theorems)

    expr_leafs_args_map = {}
    for expr in expr_list:
        expr_leafs_args_map[expr] = \
            (expression_map[expr][0][1], 1)

    cpu_count = multiprocessing.cpu_count()
    #cpu_count = 1
    with (multiprocessing.Pool(
        processes=cpu_count,           # Number of worker processes
        initializer=init_pool, # Initializer function
        initargs=(mappings_map, binary_seqs_map, all_permutations,)    # Arguments for the initializer
    ) as pool):

    #with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:





        start_time = time.time()
        created = 1
        counter = 0
        counter2 = 0

        while created:
            created = 0

            input_list_statements = []
            input_list_growing_theorems = []
            input_list_args_statement = []
            input_list_args_growing_theorem = []
            input_list_number_simple_expressions_statement = []
            input_list_number_simple_expressions_growing_theorem = []

            for expr_index in range(len(expr_list)):
                statement = expr_list[expr_index]
                args_statement = expr_leafs_args_map[statement][0]
                number_simple_expressions_statement = expr_leafs_args_map[statement][1]

                for growing_theorem_index in range(last_visited_map[expr_index] + 1, len(growing_theorems)):
                    growing_theorem = growing_theorems[growing_theorem_index]
                    args_growing_theorem = expr_leafs_args_map[growing_theorem][0]
                    number_simple_expressions_growing_theorem = expr_leafs_args_map[growing_theorem][1]

                    input_list_statements.append(statement)
                    input_list_growing_theorems.append(growing_theorem)
                    input_list_args_statement.append(args_statement)
                    input_list_args_growing_theorem.append(args_growing_theorem)
                    input_list_number_simple_expressions_statement.append(number_simple_expressions_statement)
                    (input_list_number_simple_expressions_growing_theorem.
                     append(number_simple_expressions_growing_theorem))

                    last_visited_map[expr_index] = growing_theorem_index

                    args = zip(input_list_statements,
                               input_list_growing_theorems,
                               input_list_number_simple_expressions_statement,
                               input_list_number_simple_expressions_growing_theorem,

                               input_list_args_statement,
                               input_list_args_growing_theorem)
                    if growing_theorem_index == len(growing_theorems) - 1:
                        results = pool.starmap(single_thread_calculation, args)

                        for ind in range(len(results)):
                            for entry in results[ind][0]:
                                connected_expr = entry[0]
                                connected_map = entry[1]

                                number_simple_expressions = (input_list_number_simple_expressions_statement[ind] +
                                                             input_list_number_simple_expressions_growing_theorem[ind])

                                if connected_expr not in expr_leafs_args_map:
                                    expr_leafs_args_map[connected_expr] = \
                                        (connected_map, number_simple_expressions)
                                    counter += 1


                                non_digits = {arg for arg in connected_map if arg.isdigit()}
                                if (connected_expr not in growing_theorems_set and
                                        len(non_digits) != 0 and
                                        number_simple_expressions <
                                        max_number_simple_expressions):
                                    growing_theorems.append(connected_expr)
                                    growing_theorems_set.add(connected_expr)
                                    created = 1



                            for entry_index, entry in enumerate(results[ind][1]):
                                connected_expr = entry

                                if expr_good(connected_expr):
                                    if connected_expr not in result_expr_set:

                                        reshuffled = results[ind][2][entry_index]


                                        reshuffled_mirrored = results[ind][3][entry_index]

                                        if reshuffled not in control_set and reshuffled_mirrored not in control_set:
                                            #assert reshuffled_mirrored not in control_set
                                            if reshuffled == '(>[1,2,3,4](in3[1,2,3,4])(>[5](NaturalNumbers[6,7,2,5,4,8])(in2[1,3,5])))' or reshuffled_mirrored == '(>[1,2,3](NaturalNumbers[6,7,1,2,3,8])(>[4,5](in2[4,5,2])(in3[4,1,5,3])))':
                                                test = 0

                                            if reshuffled_mirrored == '(>[1,2,3,4](in3[1,2,3,4])(>[5](NaturalNumbers[6,7,2,5,4,8])(in2[1,3,5])))' or reshuffled == '(>[1,2,3](NaturalNumbers[6,7,1,2,3,8])(>[4,5](in2[4,5,2])(in3[4,1,5,3])))':
                                                test = 0

                                            result_expr_set.add(connected_expr)
                                            reshuffled_expr_set.add(reshuffled)
                                            reshuffled_mirrored_expr_set.add(reshuffled_mirrored)
                                            control_set.add(reshuffled)
                                            control_set.add(reshuffled_mirrored)




                                        """
                                        print("")
                                        print(connected_expr)
                                        print(reshuffled)
                                        print(reshuffled_mirrored)
                                        print("")
                                        """



                        input_list_statements = []
                        input_list_growing_theorems = []
                        input_list_args_statement = []
                        input_list_args_growing_theorem = []
                        input_list_number_simple_expressions_statement = []
                        input_list_number_simple_expressions_growing_theorem = []


        end_time = time.time()





    sorted_list = list(result_expr_set)
    reshuffled_sorted_list = list(reshuffled_expr_set)
    reshuffled_mirrored_sorted_list = list(reshuffled_mirrored_expr_set)
    sorted_list.sort()
    reshuffled_sorted_list.sort()

    theorems_folder = PROJECT_ROOT / 'files/theorems'

    # if the directory already exists, delete it and everything inside
    if os.path.isdir(theorems_folder):
        shutil.rmtree(theorems_folder)

    os.makedirs(theorems_folder, exist_ok=True)

    filename = str(theorems_folder / "theorems.txt")
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            for expr in sorted_list:
                file.write(expr + "\n")
        print(f"Successfully wrote to {filename}")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")
    filename = str(theorems_folder /"reshuffled_theorems.txt")
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            for expr in reshuffled_sorted_list:
                file.write(expr + "\n")
        print(f"Successfully wrote to {filename}")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")
    filename = str(theorems_folder /"reshuffled_mirrored_theorems.txt")
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            for expr in reshuffled_mirrored_sorted_list:
                file.write(expr + "\n")
        print(f"Successfully wrote to {filename}")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")
    print("Size: " + str(len(result_expr_set)))
    print(f"Main loop runtime: {end_time - start_time:.5f} seconds")
    return result_expr_set



