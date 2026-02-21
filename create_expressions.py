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

import copy

from typing import Tuple





from itertools import permutations
from typing import Dict
from typing import List, Set

from itertools import product

from typing import Any

from pathlib import Path

import os
import shutil
from configuration_reader import configuration_reader
from configuration_reader import ExpressionDescription

from typing import Iterable
import re




# wherever this file lives, assume the project root is its parent folder
PROJECT_ROOT = Path(__file__).resolve().parent

identity = {i: i for i in range(1, 1000000 + 1)}

_ALL_PERMUTATIONS = {}
_MAPPINGS_MAP = {}
_MAPPINGS_MAP_ANCHOR = {}
_BINARY_SEQS_MAP = {}
_ANCHOR = ExpressionDescription()

_CONFIGURATION = configuration_reader()


_OPERATORS =  []
_RELATIONS = []

def set_configuration(config: configuration_reader):
    global _CONFIGURATION

    _CONFIGURATION = config

def set_operators():
    global _OPERATORS

    _OPERATORS = [ky for ky in _CONFIGURATION.data if
                 _CONFIGURATION.data[ky].input_args and _CONFIGURATION.data[ky].output_args]

def get_configuration_data():
    return _CONFIGURATION.data

def get_anchor_name(config: configuration_reader):
    # 1. Priority: Try to find the specific anchor matching the config ID
    # (e.g. "AnchorGauss" if loaded from "ConfigGauss.json")
    if getattr(config, "anchor_id", None):
        candidate = "Anchor" + config.anchor_id
        if candidate in config.data:
            return candidate

    assert False






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




def generate_binary_sequences_as_lists(n):
    """
    Generate all possible binary sequences of length n as lists of integers.

    Parameters:
    n (int): The length of the binary sequences.

    Returns:
    list: A list of binary sequences, where each sequence is a list of integers.
    """

    if n == 0:
        return [[]]

    lst = [[int(bit) for bit in bin(i)[2:].zfill(n)] for i in range(2 ** n)]

    return lst







class TreeNode1:
    def __init__(self, value):
        self.value = value
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

        node = TreeNode1("")
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
                node.left = parse_subtree(s)  # Process the left child
                node.right = parse_subtree(s)  # Process the right child
                if node.left is not None:
                    node.arguments.update(node.left.arguments)
                if node.right is not  None:
                    node.arguments.update(node.right.arguments)
                node.arguments.difference_update(get_args(node_label))

            elif s[index] == '&':
                index = index + 1
                node_label = node_label + '&'
                node.left = parse_subtree(s)  # Process the left child
                node.right = parse_subtree(s)  # Process the right child
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
                node.arguments.update(get_args(node_label))



        elif s[index:index + 2] == "!(":
            index = index + 2
            if s[index] == '>':
                index = index + 1
                node_label = node_label + "!>"
                args_to_remove = get_args(s[index:])
                index = s[index:].find(']') + index + 1
                node_label = node_label + "[" + ",".join(args_to_remove) + "]"
                node.left = parse_subtree(s)  # Process the left child
                node.right = parse_subtree(s)  # Process the right child
                if node.left is not None:
                    node.arguments.update(node.left.arguments)
                if node.right is not  None:
                    node.arguments.update(node.right.arguments)
                node.arguments.difference_update(get_args(node_label))

            elif s[index] == '&':
                index = index + 1
                node_label = node_label + "!&"
                node.left = parse_subtree(s)  # Process the left child
                node.right = parse_subtree(s)  # Process the right child
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

        node = TreeNode1("")
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

                expr_map = _CONFIGURATION[temp_expr].definition_sets
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
                expr_map = _CONFIGURATION[temp_expr].definition_sets
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

                expr_map = _CONFIGURATION[temp_expr].definition_sets
                for i in range(len(array_args)):
                    if str(i + 1) not in expr_map:
                        test = 0

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
                expr_map = _CONFIGURATION[temp_expr].definition_sets
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
                if mp1[arg][0] != mp2[arg][0]:
                    return False

        if not connect_to_anchor:
            for arg in atr:
                #if mp1[arg][0][0] == "P" and mp1[arg][1]:
                if mp1[arg][0][0] == "P" and not (mp1[arg][1] and mp2[arg][1]):
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
        new_arg = substitution_map[str(int(arg) + shift_num)]
        right_map[new_arg] = map2[arg]
        if new_arg in map1.keys():
            removable_args.add(new_arg)


    removable_args_list = list(removable_args)
    sorted_list = sort_list_according_to_occurrence(removable_args_list, expr1)
    args_to_remove = []

    for index in range(len(binary_list)):
        if binary_list[index]:

            if index >= len(sorted_list):
                test = 0
            args_to_remove.append(sorted_list[index])

    connected_map = {**left_map, **right_map}
    for arg in args_to_remove:
        del connected_map[arg]
    success = True


    if not check_maps(args_to_remove, left_map, right_map):
        success = False

    new_expr2 = subtract_number_from_ints(expr2, -shift_num, set(), True)
    new_expr1 = replace_keys_in_string(expr1, substitution_map)
    new_expr2 = replace_keys_in_string(new_expr2, substitution_map)

    connected_expr = "(>[" + ",".join(args_to_remove) + "]"
    connected_expr += new_expr1
    connected_expr += new_expr2
    connected_expr += ")"

    if new_expr1 == new_expr2:
        success = False


    return success, connected_expr, connected_map


def get_number_removable_args(mapping: dict):

    #return len(mapping.keys()) - len(set(mapping.values()))

    sub_map = {k: v for k, v in mapping.items() if v != k}
    num = len(set(sub_map.values()))

    return num

def expr_good(expr: str):
    good = False

    if expr[:3] == "(>[" and expr[:4] != "(>[]":
        good = True

    if repetitions_exist(expr):
        good = False

    return good


def check_def_sets(arg_map: dict):
    check_positive = True
    counter_map = {}

    for arg in arg_map:
        if not arg_map[arg][1]:
            continue

        if arg_map[arg][0] in counter_map:
            counter_map[arg_map[arg][0]] += 1
        else:
            counter_map[arg_map[arg][0]] = 1

    for def_set in counter_map:



        if counter_map[def_set] > _CONFIGURATION.parameters.max_values_for_def_sets[def_set]:
            return False

    counter_map = {}

    for arg in arg_map:
        if arg_map[arg][1]:
            continue

        if arg_map[arg][0] in counter_map:
            counter_map[arg_map[arg][0]] += 1
        else:
            counter_map[arg_map[arg][0]] = 1

    for def_set in counter_map:

        if counter_map[def_set] > _CONFIGURATION.parameters.max_values_for_uncomb_def_sets[def_set]:
            return False

    return check_positive

def check_complexity_level_for_def_sets(arg_map: dict, complexity_level: int):
    check_positive = True
    def_sets = set()

    for arg in arg_map:
        def_sets.add(arg_map[arg][0])

    for def_set in def_sets:
        if _CONFIGURATION.parameters.max_complexity_if_anchor_parameter_connected[def_set] < complexity_level:
            return False

    return check_positive


def qualified_for_equality(expr: str) -> bool:
    #expr = '(>[9,10](fold[1,3,4,8,2,9,10])(>[11](fold[21,23,24,28,22,10,11])(=[9,11])))'

    # 1. Disintegrate the expression into a chain of elements
    temp_chain = []
    head = disintegrate_implication(expr, temp_chain)

    # Combine antecedents and head into a single list
    full_chain = [element[0] for element in temp_chain]
    full_chain.append(head)

    # Identify the specific anchor name from the current configuration
    anchor_name = get_anchor_name(_CONFIGURATION)

    # 2. Filter the chain to exclude the anchor expression if it exists
    chain = [e for e in full_chain if extract_expression(e) != anchor_name]

    # Detect if an anchor was present (length of full chain > filtered chain)
    anchor_present = (len(full_chain) > len(chain))

    # 3. Original Logic: strict check for exactly 3 elements (Op1, Op2, Equality)
    if len(chain) != 3:
        return False

    e1, e2, e3 = chain[0], chain[1], chain[2]

    # Condition: last is equality
    if extract_expression(e3) != "=":
        return False

    # Condition: first two expressions have to be operators
    core1 = extract_expression(e1)
    core2 = extract_expression(e2)

    if core1 not in _OPERATORS or core2 not in _OPERATORS:
        return False

    # Condition: non-atomic (check that short mpl and full_mpl are not equal in config)
    desc1 = _CONFIGURATION.data[core1]
    desc2 = _CONFIGURATION.data[core2]

    if desc1.short_mpl_normalized == desc1.full_mpl:
        return False
    if desc2.short_mpl_normalized == desc2.full_mpl:
        return False

    # Condition: first two are identical (implies same operator core)
    if core1 != core2:
        return False

    args1 = get_args(e1)
    args2 = get_args(e2)

    # Retrieve the index of the output argument for this operator
    if not desc1.indices_output_args:
        return False
    out_idx = desc1.indices_output_args[0]

    # Check argument counts match
    assert (len(args1) == len(args2))

    # Condition: identical except output arg, and output arg has to be diff
    for i in range(len(args1)):
        if i == out_idx:
            # Output arguments must be different (e.g., s(x)=a, s(x)=b -> a!=b)
            if args1[i] == args2[i]:
                return False
        else:
            # Non-output arguments: Enforce equality ONLY if anchor is present
            if anchor_present:
                if args1[i] != args2[i]:
                    return False

    # Condition: those two output args are compared by equality
    out1 = args1[out_idx]
    out2 = args2[out_idx]
    args3 = get_args(e3)

    # The equality arguments must be exactly the two different outputs
    if set(args3) != {out1, out2}:
        return False

    return True

def evaluate_operator_exprs2(expression: str, anchor_attached: bool):


    arg_map = {}
    evaluation_positive = True

    temp_chain = []
    head = disintegrate_implication(expression, temp_chain)
    chain = []
    for element in temp_chain:
        chain.append(element[0])
    chain.append(head)

    op_exprs = []
    for elem in chain:
        if extract_expression(elem) in _OPERATORS:
            op_exprs.append(elem)

    head_is_op = False
    if extract_expression(head) in _OPERATORS:
        head_is_op = True

    arg_lists_list = []
    for operator_expr in op_exprs:
        args = get_args(operator_expr)
        arg_lists_list.append(args)

    rel_args = set()
    rel_args_list = []
    rel_core_exprs = []
    for elem in chain:
        if extract_expression(elem) in _RELATIONS:
            core_expr = extract_expression(elem)
            rel_core_exprs.append(core_expr)
            args = get_args(elem)
            lst = []
            for input_index in _CONFIGURATION.data[core_expr].indices_input_args:
                rel_args.add(args[input_index])
                lst.append(args[input_index])
            rel_args_list.append(lst)

    position = 0
    for expr_ind, arg_list in enumerate(arg_lists_list):
        core_expr = extract_expression(op_exprs[expr_ind])
        for arg_ind in range(len(arg_list)):
            arg = arg_list[arg_ind]
            if arg in arg_map:
                if arg_ind == _CONFIGURATION.data[core_expr].indices_output_args[0]:
                    arg_map[arg][1].add(position)
                if arg_ind in _CONFIGURATION.data[core_expr].indices_input_args:
                    arg_map[arg][0].add(position)
            else:
                if arg_ind == _CONFIGURATION.data[core_expr].indices_output_args[0]:
                    arg_map[arg] = []
                    arg_map[arg].append(set())
                    arg_map[arg].append(set())
                    arg_map[arg][1].add(position)
                if arg_ind in _CONFIGURATION.data[core_expr].indices_input_args:
                    arg_map[arg] = []
                    arg_map[arg].append(set())
                    arg_map[arg].append(set())
                    arg_map[arg][0].add(position)
        position += 1

    for arg in arg_map:
        if len(arg_map[arg][0]) > 0 and len(arg_map[arg][1]) == 0:
            continue
        elif len(arg_map[arg][0]) == 0 and len(arg_map[arg][1]) <= 2:
            if len(op_exprs) - 1 in arg_map[arg][1]:
                if len(op_exprs) == _CONFIGURATION.parameters.max_number_simple_expressions or anchor_attached:
                    if len(arg_map[arg][1]) < 2 and head_is_op:
                        evaluation_positive = False
                    else:
                        continue
                else:
                    continue
            else:
                if head_is_op:
                    evaluation_positive = False
                else:
                    continue
        elif len(arg_map[arg][0]) == 1 and len(arg_map[arg][1]) == 1:
            if not arg_map[arg][0].issubset(arg_map[arg][1]):
                if arg in rel_args:
                    evaluation_positive = False
            else:
                continue
        else:
            evaluation_positive = False

    for index, rel_args in enumerate(rel_args_list):
        core_expr = rel_core_exprs[index]
        if core_expr == "=":
            counter = 0

            for arg in rel_args:
                if arg in arg_map:
                    if arg_map[arg][1]:
                        counter += 1

            if counter >= 2:
                if not qualified_for_equality(expression):
                    evaluation_positive = False
                else:
                    test = 0


    num_end_operators = 0
    if anchor_attached:
        positions = set()

        for arg in arg_map:
            if not arg_map[arg][0] and arg_map[arg][1]:
                positions.update(arg_map[arg][1])

        num_end_operators = len(positions)
        if len(positions) > 2:
            evaluation_positive = False


    if num_end_operators > 1:
        for rel_args in rel_args_list:
            assert rel_args[0] in arg_map and rel_args[1] in arg_map

            if not ((arg_map[rel_args[0]][1] and arg_map[rel_args[1]][1] and not arg_map[rel_args[0]][0] and not arg_map[rel_args[1]][0]) or
                    (arg_map[rel_args[0]][0] and arg_map[rel_args[1]][0] and not arg_map[rel_args[0]][1] and not arg_map[rel_args[1]][1])):
                evaluation_positive = False



    if num_end_operators == 2:
        end_operator_indices = []
        for index, op_expr in enumerate(op_exprs):
            core_expr = extract_expression(op_expr)
            args = get_args(op_expr)
            output_arg = args[_CONFIGURATION.data[core_expr].indices_output_args[0]]
            if not arg_map[output_arg][0] and arg_map[output_arg][1]:
                end_operator_indices.append(index)

        input_args_list = []
        output_args_list = []
        for element in op_exprs:
            core_expr = extract_expression(element)
            args = get_args(element)

            input_args = [args[ind] for ind in _CONFIGURATION.data[core_expr].indices_input_args]
            input_args_list.append(input_args)

            output_args = [args[ind] for ind in _CONFIGURATION.data[core_expr].indices_output_args]
            output_args_list.append(output_args)

        entry_args = find_entry_args2(input_args_list, output_args_list, end_operator_indices[0], set())
        entry_args_second = find_entry_args2(input_args_list, output_args_list, end_operator_indices[1], set())

        for rel_args in rel_args_list:
            if rel_args[0] in entry_args:
                if rel_args[1] not in entry_args_second:
                    evaluation_positive = False

            if rel_args[1] in entry_args:
                if rel_args[0] not in entry_args_second:
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
    result = ["(" + match.group(0) + ")" for match in matches]

    return result

def check_prohibited_combinations(expression: str):
    result = True

    temp_chain = []
    head = disintegrate_implication(expression, temp_chain)
    chain = []
    for element in temp_chain:
        chain.append(element[0])
    chain.append(head)

    core_expressions = set()
    for element in chain:
        core_expr = extract_expression(element)
        core_expressions.add(core_expr)

    for prohibited in _CONFIGURATION.prohibited_combinations:
        if prohibited.issubset(core_expressions):
            result = False
            break

    return result



def expr_good2(expr: str,
               number_simple_expressions,
               connected_map):


    def evaluate_operator_exprs(operator_exprs2: List[str], there_are_free_args):
        evaluation_positive = True
        global _OPERATORS

        if not (operator_exprs2):
            return evaluation_positive
        last_expr = operator_exprs2[len(operator_exprs2) - 1]
        last_expr_args = get_args(last_expr)

        last_expr_core = extract_expression(last_expr)
        last_arg = last_expr_args[_CONFIGURATION.data[last_expr_core].indices_output_args[0]]

        occurrence_counter = 0
        for ind in range(len(operator_exprs2) - 1):
            args = get_args(operator_exprs2[ind])
            if last_arg in args:
                occurrence_counter += 1

                occ_ind = args.index(last_arg)
                core_expr = extract_expression(operator_exprs2[ind])
                if occ_ind != _CONFIGURATION.data[core_expr].indices_output_args[0]:
                    evaluation_positive = False

        if (len(operator_exprs2) >= _CONFIGURATION.parameters.operator_threshold or
                (len(operator_exprs2) == (_CONFIGURATION.parameters.operator_threshold - 1) and there_are_free_args)):
            for last_expr_arg_ind in _CONFIGURATION.data[last_expr_core].indices_input_args:
                occurrence_counter = 0
                for operator_expr_ind in range(len(operator_exprs2) - 1):
                    core_expr = extract_expression(operator_exprs2[operator_expr_ind])
                    args = get_args(operator_exprs2[operator_expr_ind])
                    if last_expr_args[last_expr_arg_ind] in args:
                        occurrence_counter += 1
                        #if args[len(args) - 2] != last_expr_args[last_expr_arg_ind]:
                        if args[_CONFIGURATION.data[core_expr].indices_output_args[0]] != last_expr_args[last_expr_arg_ind]:
                            evaluation_positive = False
                if occurrence_counter > 1:
                    evaluation_positive = False

        return evaluation_positive

    good = True
    global _OPERATORS

    if repetitions_exist(expr):
        good = False
        return good

    if not numbers_good(expr):
        good = False
        return good


    size_args = len({arg for arg in connected_map if connected_map[arg][0][:1] != 'P' and connected_map[arg][1] })
    if (number_simple_expressions == _CONFIGURATION.parameters.max_number_simple_expressions and
            not check_def_sets(connected_map)):
        good = False
        return good


    operator_exprs = extract_operator_expressions(_OPERATORS, expr)
    good = evaluate_operator_exprs(operator_exprs, size_args > 0) and good
    good = evaluate_operator_exprs2(expr, False) and good
    good = check_prohibited_combinations(expr) and good

    return good

def numbers_good(expr: str):
    passed = True

    for element in _CONFIGURATION:
        number = expr.count(_CONFIGURATION[element].handle)
        if number > _CONFIGURATION[element].max_count_per_conjecture:
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



def init_pool(mappings_map, binary_seqs_map, all_permutations, mappings_map_anchor, config):
    global _MAPPINGS_MAP
    global _MAPPINGS_MAP_ANCHOR
    global _BINARY_SEQS_MAP
    global _ALL_PERMUTATIONS
    global _ANCHOR
    global _CONFIGURATION
    global _OPERATORS
    global _RELATIONS

    _MAPPINGS_MAP = mappings_map
    _MAPPINGS_MAP_ANCHOR = mappings_map_anchor
    _BINARY_SEQS_MAP = binary_seqs_map
    _ALL_PERMUTATIONS = all_permutations
    _CONFIGURATION = config
    anchor_name = get_anchor_name(config)
    _ANCHOR = _CONFIGURATION[anchor_name]
    _OPERATORS = [ky for ky in _CONFIGURATION.data if
                 _CONFIGURATION.data[ky].input_args and _CONFIGURATION.data[ky].output_args]
    _RELATIONS = [ky for ky in _CONFIGURATION.data if
                  len(_CONFIGURATION.data[ky].input_args) == 2 and not _CONFIGURATION.data[ky].output_args]

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

    if core_expr in _OPERATORS:
        if args[_CONFIGURATION.data[core_expr].indices_output_args[0]] == output_variable:
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

    if head_expr in _OPERATORS:
        output_variable = head_args[_CONFIGURATION.data[head_expr].indices_output_args[0]]
    else:
        return ""

    assert output_variable != ""

    alternative = ""
    chain = []
    for element in temp_chain:
        if stays_output_variable(element[0], output_variable):
            alternative = element[0]
        else:
            chain.append(element[0])

    if anchor_first:
        anchor_name = get_anchor_name(_CONFIGURATION)
        prioritize_anchor(chain, _CONFIGURATION[anchor_name].handle)

    if alternative == "":
        return ""

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


def pattern_in_conjecture(configuration, conjecture: str) -> bool:
    """
    Return True iff any exclusion pattern matches anywhere in `conjecture`.

    Expects `configuration.patterns_to_exclude` to be an iterable of compiled
    regex objects (as provided by configuration_reader). If it's missing, falls
    back to compiling `configuration.patterns_to_exclude_raw` on the fly.
    """
    pats: Iterable[re.Pattern] = getattr(configuration, "patterns_to_exclude", None)
    if pats is None:
        raw = getattr(configuration, "patterns_to_exclude_raw", []) or []
        pats = [re.compile(s) for s in raw if isinstance(s, str) and s.strip()]

    return any(p.search(conjecture) for p in pats)

def check_def_sets_prior_to_connection(args_statement: {}, args_growing_theorem: {}):
    check_positive = True
    counter_map = {}

    for arg in args_statement:
        if not args_statement[arg][1]:
            continue
        if args_statement[arg][0] in counter_map:
            counter_map[args_statement[arg][0]] += 1
        else:
            counter_map[args_statement[arg][0]] = 1

    for arg in args_growing_theorem:
        if not args_growing_theorem[arg][1]:
            continue
        if args_growing_theorem[arg][0] in counter_map:
            counter_map[args_growing_theorem[arg][0]] += 1
        else:
            counter_map[args_growing_theorem[arg][0]] = 1

    for def_set in counter_map:

        if counter_map[def_set] > _CONFIGURATION.parameters.max_values_for_def_sets_prior_connection[def_set]:
            check_positive = False

    return check_positive


def prohibited_heads_good(conjecture: str) -> bool:
    """
    Returns False if the head (conclusion) of the conjecture matches
    any expression in the prohibited_heads list.
    """
    # Check if list exists and is not empty
    if not getattr(_CONFIGURATION, "prohibited_heads", None):
        return True

    temp_chain = []
    # Extract the conclusion (head) of the implication chain
    head = disintegrate_implication(conjecture, temp_chain)

    # Extract the core name (e.g. 'limitSet' from '(limitSet[...])')
    head_core = extract_expression(head)

    # Check against the prohibited list
    if head_core in _CONFIGURATION.prohibited_heads:
        return False

    return True

def count_arguments_filter(conjecture: str) -> bool:
    """
    Disintegrates a conjecture and checks if any constituting expression
    contains duplicate arguments.

    Args:
        conjecture (str): The MPL conjecture string.

    Returns:
        bool: False if any expression has the same argument more than once, True otherwise.
    """
    # 1. Disintegrate the conjecture
    chain = []
    # disintegrate_implication populates 'chain' with antecedents and returns the conclusion (head)
    head = disintegrate_implication(conjecture, chain)

    # 2. Collect all constituent expressions
    # The chain contains tuples where the first element is the expression string
    all_expressions = [item[0] for item in chain]
    all_expressions.append(head)

    # 3. Check each expression for duplicate arguments
    for expr in all_expressions:
        args = get_args(expr)

        # If the count of arguments differs from the count of unique arguments, duplicates exist
        if len(args) != len(set(args)):
            return False

    return True

def single_thread_calculation(statement: str,
                              growing_theorem: str,
                              number_simple_expressions_statement: int,
                              number_simple_expressions_growing_theorem: int,
                              args_statement: {},
                              args_growing_theorem: {}):
    def make_all_connection_maps(args_map1: Dict[str, Any],
                                 args_map2: Dict[str, Any],
                                 with_anchor: bool,
                                 mappings_map: Dict[int, Dict[Tuple[int, int], List[Dict[int, int]]]]) -> List[Dict[str, Any]]:
        def union_of_dicts(dicts, sn):
            """
            Merge a sequence of dictionaries into one.
            In case of key conflicts, later dictionaries overwrite earlier ones.
            """
            result = {}
            for d in dicts:
                result.update(d)

            for arg in args_map1:
                shifted_arg = str(int(arg) + sn)
                if shifted_arg not in result:
                    result[shifted_arg] = shifted_arg

            for arg in args_map2:
                if arg not in result:
                    result[arg] = arg

            return result

        def create_union_maps(list_of_lists, sn):
            """
            Given a list of lists of dictionaries, produce all merged maps.
            """
            results = []
            for selection in product(*list_of_lists):
                merged = union_of_dicts(selection, sn)
                results.append(merged)
            return results

        mappings_list = []
        src_map: Dict[str, set] = {}
        dst_map: Dict[str, set] = {}

        mapping_size = len(args_map1) + len(args_map2)

        # Deterministically compute shift number by extracting digits from keys
        shift_num = max(
            int(re.search(r"\d+", arg).group())
            for arg in args_map2.keys()
            if re.search(r"\d+", arg)
        )  # CHANGED: use regex to handle non-integer literal keys
        shift_num = max({int(arg) for arg in args_map2.keys()})

        # Build source and destination groupings
        for arg, val in args_map1.items():
            if not with_anchor:
                if val[1]:
                    src_map.setdefault(val[0], set()).add(arg)
            else:
                src_map.setdefault(val[0], set()).add(arg)
        for arg, val in args_map2.items():
            if not with_anchor:
                if val[1]:
                    dst_map.setdefault(val[0], set()).add(arg)
            else:
                    dst_map.setdefault(val[0], set()).add(arg)

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



                mappings = mappings_map[
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
        all_connection_maps = create_union_maps(mappings_list, shift_num)
        return all_connection_maps

    list_args = [key for key in args_statement.keys() if args_statement[key][0][0] != "P" and args_statement[key][1]]
    list_args.extend([arg for arg in args_growing_theorem.keys() if args_growing_theorem[arg][0][0] != "P" and args_growing_theorem[arg][1]])
    list_set_args = [key for key in args_statement.keys() if args_statement[key][0][0] == "P" and args_statement[key][1]]
    list_set_args.extend([arg for arg in args_growing_theorem.keys() if args_growing_theorem[arg][0][0] == "P" and args_growing_theorem[arg][1]])
    number_digits_both = len(list_args)
    number_sets_both = len(list_set_args)


    connected_list = []
    connected_list2 = []
    reshuffled_list = []
    reshuffled_mirrored_list = []

    #test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    growing_theorem = "(>[11](limitSet[1,4,12,9,11])(interval[1,4,2,9,11]))"
    statement = "(interval[1,4,2,10,12])"
    number_simple_expressions_growing_theorem = 2
    number_simple_expressions_statement = 1
    args_growing_theorem = {"1": ("P(1)", False), "4": ("P(x(1)(x(1)(1)))", False), "12": ("P(1)",True), "9": ("(1)",True), "2": ("(1)",True)}
    args_statement = {"1": ("P(1)", False), "4": ("P(x(1)(x(1)(1)))", False), "12": ("P(1)",True), "10": ("(1)",True), "2": ("(1)",True)}
    """

    """
    growing_theorem = "(interval[1,4,2,9,11])"
    statement = "(limitSet[1,4,12,9,11])"
    number_simple_expressions_growing_theorem = 1
    number_simple_expressions_statement = 1
    args_growing_theorem = {"1": ("P(1)", False), "4": ("P(x(1)(x(1)(1)))", False), "11": ("P(1)",True), "9": ("(1)",True), "2": ("(1)",True)}
    args_statement = {"1": ("P(1)", False), "4": ("P(x(1)(x(1)(1)))", False), "12": ("P(1)",True), "9": ("(1)",True), "11": ("P(1)",True)}
    """
    """
    growing_theorem = "(>[12](interval[1,4,2,10,12])(>[11](limitSet[1,4,12,9,11])(interval[1,4,2,9,11])))"
    statement = "(in2[9,10,3])"
    number_simple_expressions_growing_theorem = 3
    number_simple_expressions_statement = 1
    args_growing_theorem = {"1": ("P(1)", False), "4": ("P(x(1)(x(1)(1)))", False), "9": ("(1)",True), "2": ("(1)",True), "10": ("(1)",True)}
    args_statement = { "9": ("(1)",True), "10": ("(1)",True), "3": ("P(x(1)(1))", False)}
    """

    """
    growing_theorem = "(>[7](fold[1,2,3,4,5,6,7])(=[6,8]))"
    statement = "(fold[1,2,3,4,5,6,7])"
    number_simple_expressions_growing_theorem = 2
    number_simple_expressions_statement = 1
    args_growing_theorem = {"1": ("P(1)", False), "2": ("P(x(1)(1))", False), "3": ("P(x(1)(x(1)(1)))", False),
      "4": ("P(x(1)(1))", False), "5": ("(1)", False), "6": ("(1)", True), "8": ("(1)", True)}
    args_statement = { "1": ("P(1)", False), "2": ("P(x(1)(1))", False), "3": ("P(x(1)(x(1)(1)))", False),
      "4": ("P(x(1)(1))", False), "5": ("(1)", False), "6": ("(1)", True), "7": ("(1)", True)}
    """

    """
    growing_theorem = "(>[1](fXY[1,2,3])(sequence[4,5,6,7,1]))"
    statement = "(interval[1,2,3,4,5])"
    number_simple_expressions_growing_theorem = 2
    number_simple_expressions_statement = 1
    args_growing_theorem = {"2": ("P(1)", True), "3": ("P(1)", False), "4": ("P(1)", False), "5": ("P(x(1)(x(1)(1)))", False), "6": ("(1)", False), "7": ("(1)", True)}
    args_statement = {"1": ("P(1)", False), "2": ("P(x(1)(x(1)(1)))", False), "3": ("(1)", False), "4": ("(1)", True), "5": ("P(1)", True)}
    """


    #if max(number_digits_both, number_sets_both) > configuration.parameters.max_size_mapping_def_set:
        #return connected_list, connected_list2, reshuffled_list, reshuffled_mirrored_list


    if not check_def_sets_prior_to_connection(args_statement, args_growing_theorem):
        return connected_list, connected_list2, reshuffled_list, reshuffled_mirrored_list

    if not check_conjecture_complexity_per_operator(growing_theorem, statement):
        return connected_list, connected_list2, reshuffled_list, reshuffled_mirrored_list

    connection_maps = make_all_connection_maps(args_growing_theorem, args_statement, False, _MAPPINGS_MAP)

    for connection_map in connection_maps:

        """
        connection_map = {'1': '1', '10': '10', '12': '12', '13': '13', '14': '2', '16': '16', '2': '2', '21': '21', '24': '12', '4': '4'}
        """

        """
        connection_map = {'10': '10', '11': '11', '12': '12', '14': '14', '19': '9', '20': '10', '3': '3', '9': '9'}
        """

        #test!!!!!!!!!!!!!!!!!!!!!!!!!!

        """
        connection_map = {'1': '1', '11': '11', '12': '12', '13': '13', '14': '14', '16': '16', '21': '21', '23': '11', '4': '4', '9': '9'}
        """

        """
        connection_map = {'1': '1', '10': '10', '11': '11', '12': '12', '13': '6', '15': '7', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9'}
        """

        """
        connection_map = {'1': '1', '10': '10', '11': '11', '12': '4', '2': '2', '3': '3', '4': '4', '5': '5', '7': '5', '8': '8', '9': '9'}
        """


        number_removable_args = get_number_removable_args(connection_map)



        for binary_list in _BINARY_SEQS_MAP[number_removable_args]:

            """
            binary_list = [1, 1]
            #test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            """




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
                    continue


                if not only_in_head_good(connected_expr):
                    continue

                if not prohibited_heads_good(connected_expr):
                    continue


                reshuffled_expr, reshuffled_map, replacement_map = reshuffle(connected_expr, _ALL_PERMUTATIONS, True)

                connected_list.append((reshuffled_expr, reshuffled_map))

                complexity_level = count_operator_occurrences_regex(reshuffled_expr) + 1

                number_combinable_args = len({arg for arg in reshuffled_map if reshuffled_map[arg][1]})
                if (check_def_sets(reshuffled_map) and number_combinable_args <= _CONFIGURATION.parameters.max_number_args_expr and
                        check_complexity_level_for_def_sets(reshuffled_map, complexity_level)):
                    connection_maps2 = make_all_connection_maps(reshuffled_map,
                                                                _ANCHOR.definition_sets,
                                                                True,
                                                                _MAPPINGS_MAP_ANCHOR)


                    for connection_map2 in connection_maps2:

                        to_continue = False
                        for ky in connection_map2:

                            mp = _ANCHOR.definition_sets
                            if int(ky) > len(mp):
                                if connection_map2[ky] == ky:
                                    to_continue = True
                        if to_continue:
                            continue




                        number_removable_args = get_number_removable_args(connection_map2)
                        binary_list2 = [1 for _ in range(number_removable_args)]

                        success2, connected_expr2, connected_map2 = \
                            connect_expressions(_ANCHOR.short_mpl_normalized,
                                                reshuffled_expr,
                                                _ANCHOR.definition_sets,
                                                reshuffled_map,
                                                connection_map2,
                                                binary_list2,
                                                True)


                        if success2:

                            if ((not check_input_variables_theorem_operator_head(connected_expr2) or
                                    not check_input_variables_order(connected_expr2, _ALL_PERMUTATIONS) or
                                    pattern_in_conjecture(_CONFIGURATION, connected_expr2))):
                                continue


                            if not evaluate_operator_exprs2(connected_expr2, True):
                                continue

                            if not control_equality(connected_expr2):
                                continue

                            connected_list2.append(connected_expr2)

                            reshuffled_expr2, reshuffled_map2, replacement_map2 = \
                                reshuffle(connected_expr2, _ALL_PERMUTATIONS, True)

                            reshuffled_list.append(reshuffled_expr2)

                            reshuffled_mirrored = \
                                create_reshuffled_mirrored(connected_expr2, _ALL_PERMUTATIONS)

                            #if reshuffled_mirrored:
                            reshuffled_mirrored_list.append(reshuffled_mirrored)

    return connected_list, connected_list2, reshuffled_list, reshuffled_mirrored_list


# Literal pattern: "(=[<digits>,<digits>])"
_PATTERN = re.compile(r"\(=\[\d+,\d+]\)")

def control_equality(conjecture: str) -> bool:
    result = True

    m = _PATTERN.search(conjecture)

    if m:
        args = get_args(m.group(0))
        if int(args[0]) > int(args[1]):
            result = False

    result = count_arguments_filter(conjecture)

    return result

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

from typing import Dict, Tuple, List
import itertools

def create_map_anchor(
    left_part_max_size: int,
    right_part_max_size: int
) -> Dict[int, Dict[Tuple[int, int], List[Dict[int, int]]]]:
    """
    Creates nested map structures for all totals n with 2 <= n <= N,
    where N = left_part_max_size + right_part_max_size, but only for
    splits (p, q) with p + q = n such that:
        p <= left_part_max_size and q <= right_part_max_size.

    For each valid (p, q) on universe {1, ..., n}:
      - The left part is {1, ..., p} (fixed points: Q(i) = i for i <= p).
      - The right part is {p+1, ..., n}.
      - For any subset T ⊆ right part, elements of T can map arbitrarily
        (non-injectively; many-to-one allowed) into {1, ..., p}.
      - Elements in right part not in T stay fixed (Q(i) = i).

    Returns:
      Dict[n, Dict[(p, q), List[Q]]], where Q is a dict representing a map
      {1, ..., n} -> {1, ..., n}.
    """
    if left_part_max_size < 1 or right_part_max_size < 1:
        raise ValueError("Both left_part_max_size and right_part_max_size must be at least 1.")

    N = left_part_max_size + right_part_max_size
    outer_map: Dict[int, Dict[Tuple[int, int], List[Dict[int, int]]]] = {}

    for n in range(2, N + 1):
        M: Dict[Tuple[int, int], List[Dict[int, int]]] = {}

        # Consider all splits p + q = n
        for p in range(1, n):
            q = n - p

            # Respect the thresholds
            if p > left_part_max_size or q > right_part_max_size:
                continue

            S = list(range(p + 1, n + 1))  # right part {p+1, ..., n}
            L: List[Dict[int, int]] = []

            # All subsets T of S
            for r in range(len(S) + 1):
                for T in itertools.combinations(S, r):
                    # All (possibly non-injective) assignments T -> {1, ..., p}
                    targets = list(range(1, p + 1))
                    for assignment in itertools.product(targets, repeat=len(T)):
                        # Start as identity on {1, ..., n}
                        Q: Dict[int, int] = {i: i for i in range(1, n + 1)}
                        # Override images for elements in T
                        for s, t in zip(T, assignment):
                            Q[s] = t
                        L.append(Q)

            M[(p, q)] = L

        if M:
            outer_map[n] = M

    return outer_map




def find_entry_args2(input_args_list: list[list[str]],
                     output_args_list: list[list[str]],
                     index: int,
                     already_visited):
    entry_args = set()

    if index in already_visited:
        return entry_args

    already_visited.add(index)

    assert len(output_args_list) == len(input_args_list)

    for input_arg in input_args_list[index]:
        found = False

        for index2 in range(len(output_args_list)):
            if input_arg in output_args_list[index2]:
                assert index2 != len(output_args_list)

                found = True
                entry_args.update(find_entry_args2(input_args_list, output_args_list, index2, already_visited))

                break

        if not found:
            entry_args.add(input_arg)

    return entry_args

def only_in_head_good(conjecture: str):
    result = True

    for handle in _CONFIGURATION.only_in_head_raw:
        if handle in conjecture:
            if conjecture.count(handle) == 1:
                temp_chain = []
                head = disintegrate_implication(conjecture, temp_chain)

                if handle not in head:
                    result = False
                    break

            else:
                result = False
                break

    return result


def check_conjecture_complexity_per_operator(conjecture: str, new_expression: str):
    result = True

    temp_chain = []
    head = disintegrate_implication(conjecture, temp_chain)
    chain = []
    for element in temp_chain:
        chain.append(element[0])
    chain.append(head)
    chain.append(new_expression)


    for element in chain:
        core_expr = extract_expression(element)



        if _CONFIGURATION.data[core_expr].max_size_expression < len(chain):
            result = False

    return result

def check_input_variables_theorem_operator_head(theorem: str):
    temp_chain = []
    head = disintegrate_implication(theorem, temp_chain)
    chain = []
    for element in temp_chain:
        chain.append(element[0])
    chain.append(head)

    anchor_name = get_anchor_name(_CONFIGURATION)
    for element in chain:
        if _CONFIGURATION[anchor_name].handle in element:
            chain.remove(element)

    last_expr = chain[len(chain) - 1]
    core_expr = extract_expression(last_expr)
    if core_expr not in _OPERATORS:
        result = True
    else:
        args_list = []
        for element in chain:
            core_expr = extract_expression(element)
            args = get_args(element)
            args_list.append(args[:len(args) - 1])

        input_args_list = []
        output_args_list = []
        for element in chain:
            core_expr = extract_expression(element)
            args = get_args(element)

            input_args = [args[ind] for ind in _CONFIGURATION.data[core_expr].indices_input_args]
            input_args_list.append(input_args)

            output_args = [args[ind] for ind in _CONFIGURATION.data[core_expr].indices_output_args]
            output_args_list.append(output_args)


        assert len(output_args_list[len(chain) - 1]) == 1
        output_var = output_args_list[len(chain) - 1][0]

        second_last_index = -1
        for index, args in enumerate(output_args_list):
            if output_var in output_args_list[index]:

                if index == len(chain) - 1:
                    return False


                second_last_index = index

                break

        assert second_last_index >= 0

        entry_args = find_entry_args2(input_args_list, output_args_list, len(chain) - 1, set())
        entry_args_second = find_entry_args2(input_args_list, output_args_list, second_last_index, set())

        anchor_name = get_anchor_name(_CONFIGURATION)
        entry_args = entry_args - _CONFIGURATION[anchor_name].definition_sets.keys()
        entry_args_second = entry_args_second - _CONFIGURATION[anchor_name].definition_sets.keys()

        result = entry_args == entry_args_second

    return result





def find_digit_args(theorem: str):
    temp_chain = []
    head = disintegrate_implication(theorem, temp_chain)
    chain = []
    for element in temp_chain:
        chain.append(element[0])
    chain.append(head)


    all_input_args = set()
    for element in chain:
        core_expression = extract_expression(element)

        args = get_args(element)
        all_input_args.update([args[index] for index in _CONFIGURATION.data[core_expression].indices_input_args])

    anchor_name = get_anchor_name(_CONFIGURATION)
    for element in chain:
        if _CONFIGURATION[anchor_name].handle in element:
            args = get_args(element)
            all_input_args.difference_update(args)


    all_output_args = set()
    for element in chain:
        core_expression = extract_expression(element)

        args = get_args(element)
        all_output_args.update([args[index] for index in _CONFIGURATION.data[core_expression].indices_output_args])


    all_input_args = all_input_args - all_output_args
    return all_input_args




def get_left_right(chain: list[str], expression: str, digits: set[str], counter, already_visited):
    left = set()
    right = set()



    if expression in already_visited:
        return left, right

    already_visited.add(expression)

    core_expression = extract_expression(expression)

    assert core_expression in _OPERATORS
    assert len(_CONFIGURATION.data[core_expression].indices_input_args) == 2

    args = get_args(expression)
    if args[0] in digits:
        left.add(args[0])
    else:
        for element in chain:
            #if element == expression:
                #continue

            core_expression_element = extract_expression(element)
            if (core_expression_element in _OPERATORS and
                    len(_CONFIGURATION.data[core_expression_element].indices_input_args) == 2):
                args_element = get_args(element)

                if (args_element[_CONFIGURATION.data[core_expression_element].indices_output_args[0]] ==
                        args[_CONFIGURATION.data[core_expression].indices_input_args[0]]):
                    left_element, right_element = get_left_right(chain, element, digits, counter + 1, already_visited)
                    left.update(left_element)
                    left.update(right_element)

    if args[1] in digits:
        right.add(args[1])
    else:
        for element in chain:
            #if element == expression:
                #continue

            core_expression_element = extract_expression(element)
            if (core_expression_element in _OPERATORS and
                    len(_CONFIGURATION.data[core_expression_element].indices_input_args) == 2):
                args_element = get_args(element)

                if (args_element[_CONFIGURATION.data[core_expression_element].indices_output_args[0]] ==
                        args[_CONFIGURATION.data[core_expression].indices_input_args[1]]):
                    left_element, right_element = get_left_right(chain, element, digits, counter + 1, already_visited)
                    right.update(left_element)
                    right.update(right_element)


    return left, right

def get_right_chain(chain: list[str], head: str, already_visited: Set[str]):
    right_chain = []

    if head in already_visited:
        return right_chain

    already_visited.add(head)

    right_chain.append(head)

    head_args = get_args(head)

    head_core_expr = extract_expression(head)

    head_inputs = [head_args[ind] for ind in _CONFIGURATION.data[head_core_expr].indices_input_args]

    for expression in chain:
        core_expr = extract_expression(expression)
        if core_expr not in _OPERATORS:
            continue



        expr_args = get_args(expression)
        core_expr = extract_expression(expression)


        expr_output = expr_args[_CONFIGURATION.data[core_expr].indices_output_args[0]]

        if expr_output in head_inputs:
            right_chain.extend(get_right_chain(chain, expression, already_visited))

    return right_chain

def get_left_right_chains(chain: list[str]):
    no_anchor_chain = chain[1:]
    right_chain = get_right_chain(no_anchor_chain, chain[-1], set())

    left_chain = [x for x in no_anchor_chain if x not in right_chain]

    return left_chain, right_chain

def get_operator_id(expr: str):
    expr_id = ""

    core_expr = extract_expression(expr)
    if core_expr in _OPERATORS:
        repl_map = {}

        args = get_args(expr)

        for ind in _CONFIGURATION.data[core_expr].indices_input_args:
            repl_map[args[ind]] = ""

        for ind in _CONFIGURATION.data[core_expr].indices_output_args:
            repl_map[args[ind]] = ""

        expr_id = replace_keys_in_string(expr, repl_map)

    return expr_id

def check_input_variable_position(chain: list[str], digits: set[str]):
    result = True

    order_map = {}

    for expression in chain:
        core_expression = extract_expression(expression)

        if core_expression in _OPERATORS and len(_CONFIGURATION.data[core_expression].indices_input_args) == 2:
            args = get_args(expression)

            for arg_ind in range(2):

                arg = args[_CONFIGURATION.data[core_expression].indices_input_args[arg_ind]]
                if arg in digits:
                    key = (get_operator_id(expression), arg)

                    if key not in order_map:
                        order_map[key] = arg_ind
                    else:
                        if order_map[key] != arg_ind:
                            return False

    return result

def remove_outputs(chain: list[str]):
    replacement_map = {}

    for expr in chain:
        core_expr = extract_expression(expr)

        args = get_args(expr)
        output_args = [args[index] for index in _CONFIGURATION.data[core_expr].indices_output_args]
        for arg in output_args:
            replacement_map[arg] = ''

    removed = ''.join(chain)
    removed = replace_keys_in_string(removed, replacement_map)

    return removed

def check_tautology(left_chain: list[str], right_chain: list[str]):
    left_removed = remove_outputs(left_chain)
    right_removed = remove_outputs(right_chain)

    result = left_removed != right_removed

    if not result:
        test = 0

    return result

def check_functions(chain: list[str]):
    removed_set = set()
    result = True

    for expr in chain:
        core_expr = extract_expression(expr)

        #if core_expr == 'in2':
        if core_expr in _OPERATORS and len(_CONFIGURATION.data[core_expr].indices_input_args) == 1:
            args = get_args(expr)
            #output = args[len(args) - 2]
            output = args[_CONFIGURATION.data[core_expr].indices_output_args[0]]

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
    #head_args = get_args(head)

    if extract_expression(head) not in _OPERATORS:
        return False

    #head_key = (head_args[len(head_args) - 1], extract_expression(head))
    head_key = get_operator_id(head)

    for expr in no_anchor:
        #expr_args = get_args(expr)

        #expr_key = (expr_args[len(expr_args) - 1], extract_expression(expr))
        expr_key = get_operator_id(expr)

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

        if core_expression in _OPERATORS and len(_CONFIGURATION.data[core_expression].indices_input_args) == 2:
            args = get_args(expression)

            arg_left = args[_CONFIGURATION.data[core_expression].indices_input_args[0]]
            arg_right = args[_CONFIGURATION.data[core_expression].indices_input_args[1]]

            if arg_left in digits and arg_right in digits:
                order_map[(get_operator_id(expression), frozenset({arg_left, arg_right}))] = [arg_left, arg_right]

                key = frozenset({arg_left, arg_right})

                if key in order_set:
                    result = False
                else:
                    order_set.add(key)



    for expression in chain:
        core_expression = extract_expression(expression)

        if core_expression in _OPERATORS and len(_CONFIGURATION.data[core_expression].indices_input_args) == 2:
            args = get_args(expression)

            left, right = get_left_right(chain, expression, digits, 0, set())

            operator = get_operator_id(expression)
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


            assert reshuffled_mirrored # reshuffled_mirrored != ""
            result = True

    result = result and check_tautology(left_chain, right_chain)

    if not qualified_for_equality(theorem):
        result = result and check_functions(chain)

    return result


def get_tertiaries(chain: list[str]):
    tertiaries = set()

    for expr in chain:
        core_expression = extract_expression(expr)
        if core_expression in _OPERATORS and len(_CONFIGURATION.data[core_expression].indices_input_args) == 2:
            tertiaries.add(get_operator_id(expr))

    return tertiaries

def check_tertiaries(left_chain: list[str], right_chain: list[str]):
    result = True

    left_tertiaries = get_tertiaries(left_chain)
    right_tertiaries = get_tertiaries(right_chain)

    if left_tertiaries and right_tertiaries:
        if not left_tertiaries & right_tertiaries:
            result = False

    return result

def determine_left_side_boundary(config: configuration_reader):
    sets = config.data[get_anchor_name(config)].definition_sets

    counter_map = {}
    for arg in sets:
        st = sets[arg][0]

        if st in counter_map:
            counter_map[st] += 1
        else:
            counter_map[st] = 1

    boundary = max(counter_map.values(), default=None)

    return boundary

def determine_right_side_boundary(config: configuration_reader):
    boundary = -1
    for def_set in config.parameters.max_values_for_def_sets:
        cand_bound = (config.parameters.max_values_for_uncomb_def_sets[def_set] +
                      config.parameters.max_values_for_def_sets[def_set])

        if cand_bound > boundary:
            boundary = cand_bound

    return boundary



def create_expressions_parallel(config: configuration_reader):
    result_expr_set = set()
    reshuffled_expr_set = set()
    reshuffled_mirrored_expr_set = set()
    control_set = set()





    mappings_map = create_map(config.parameters.max_size_mapping_def_set)

    left_side_boundary = determine_left_side_boundary(config)
    right_side_boundary = determine_right_side_boundary(config)
    mappings_map_anchor = create_map_anchor(left_side_boundary,
                                            right_side_boundary)

    all_permutations = generate_all_permutations(config.parameters.max_number_simple_expressions + 1)

    binary_seqs_map = {}
    for num in range(0, config.parameters.max_size_binary_list):
        binary_seqs_map[num] = generate_binary_sequences_as_lists(num)

    expr_list = list(config[ky].short_mpl_normalized for ky in config.keys() if config[ky].max_count_per_conjecture > 0)

    last_visited_map = {i: -1 for i in range(len(expr_list))}

    growing_theorems = [expr for expr in expr_list]
    growing_theorems_set = set(growing_theorems)


    expr_leafs_args_map = {}
    for expr in expr_list:
        core_expr = extract_expression(expr)
        expr_leafs_args_map[expr] = \
            (config[core_expr].definition_sets, 1)

    cpu_count = multiprocessing.cpu_count()
    #cpu_count = 1
    with (multiprocessing.Pool(
        processes=cpu_count,           # Number of worker processes
        initializer=init_pool, # Initializer function
        initargs=(mappings_map, binary_seqs_map, all_permutations, mappings_map_anchor, config)    # Arguments for the initializer
    ) as pool):

    #with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:





        start_time = time.time()
        created = 1
        counter = 0
        counter2 = 0

        while created:
            created = 0



            for expr_index in range(len(expr_list)):
                statement = expr_list[expr_index]
                args_statement = expr_leafs_args_map[statement][0]
                nse_statement = expr_leafs_args_map[statement][1]

                start = last_visited_map[expr_index] + 1
                end = len(growing_theorems)  # snapshot the current tail for this pass
                if start >= end:
                    continue

                # Build inputs for [start, end)
                input_list_statements = []
                input_list_growing_theorems = []
                input_list_args_statement = []
                input_list_args_growing_theorem = []
                input_list_number_simple_expressions_statement = []
                input_list_number_simple_expressions_growing_theorem = []

                for gti in range(start, end):
                    gt = growing_theorems[gti]
                    args_gt, nse_gt = expr_leafs_args_map[gt][0], expr_leafs_args_map[gt][1]


                    input_list_statements.append(statement)
                    input_list_growing_theorems.append(gt)
                    input_list_args_statement.append(args_statement)
                    input_list_args_growing_theorem.append(args_gt)
                    input_list_number_simple_expressions_statement.append(nse_statement)
                    input_list_number_simple_expressions_growing_theorem.append(nse_gt)

                # One dispatch per statement per snapshot
                args = list(zip(
                    input_list_statements,
                    input_list_growing_theorems,
                    input_list_number_simple_expressions_statement,
                    input_list_number_simple_expressions_growing_theorem,
                    input_list_args_statement,
                    input_list_args_growing_theorem,
                ))
                results = pool.starmap(single_thread_calculation, args)

                # process `results` exactly as you already do...
                # (no change to the body that consumes `results`)

                # Mark the processed window as visited *after* successful dispatch
                last_visited_map[expr_index] = end - 1

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
                                        config.parameters.max_number_simple_expressions):
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


                                            result_expr_set.add(connected_expr)
                                            reshuffled_expr_set.add(reshuffled)
                                            if reshuffled_mirrored:
                                                reshuffled_mirrored_expr_set.add(reshuffled_mirrored)
                                            control_set.add(reshuffled)
                                            if reshuffled_mirrored:
                                                control_set.add(reshuffled_mirrored)







        end_time = time.time()

    sorted_list = list(result_expr_set)
    reshuffled_sorted_list = list(reshuffled_expr_set)
    reshuffled_mirrored_sorted_list = list(reshuffled_mirrored_expr_set)
    sorted_list.sort()
    reshuffled_sorted_list.sort()

    theorems_folder = PROJECT_ROOT / 'files/theorems'

    # MODIFIED LOGIC: Clean everything EXCEPT proved_theorems.txt
    if os.path.isdir(theorems_folder):
        for item in os.listdir(theorems_folder):
            if item == "proved_theorems.txt":
                continue

            item_path = os.path.join(theorems_folder, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Failed to delete {item_path}. Reason: {e}")
    else:
        os.makedirs(theorems_folder, exist_ok=True)

    # Write new theorems (overwriting old ones for this batch, which is intended)
    filename = str(theorems_folder / "theorems.txt")
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            for expr in sorted_list:
                file.write(expr + "\n")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")

    filename = str(theorems_folder / "reshuffled_theorems.txt")
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            for expr in reshuffled_sorted_list:
                file.write(expr + "\n")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")

    filename = str(theorems_folder / "reshuffled_mirrored_theorems.txt")
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            for expr in reshuffled_mirrored_sorted_list:
                file.write(expr + "\n")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")

    print("Number conjectures: " + str(len(result_expr_set)))
    return result_expr_set




