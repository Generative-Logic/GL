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

from create_expressions import *
import regex
from collections import deque

from pympler import asizeof

import copy
import bisect
import time










import faulthandler
faulthandler.enable()






_ALL_PERMUTATIONS_ANA = {}
_ALL_BINARIES_ANA = {}
_ALL_MAPPINGS_ANA = {}
_LIST_OF_MEMORY_BLOCKS = []

counter_decouple = 0

class LocalMemoryValue:
    def __init__(self, value="", levels=frozenset(),
                 original_implication="", key=None, remaining_args=None):
        self.value = value
        self.levels = levels
        self.original_implication = original_implication
        self.key = key or []
        self.remaining_args = remaining_args or set()

    def __eq__(self, other):
        if not isinstance(other, LocalMemoryValue):
            return NotImplemented
        return (
            self.value == other.value and
            self.levels == other.levels and
            self.original_implication == other.original_implication and
            tuple(self.key) == tuple(other.key) and
            frozenset(self.remaining_args) == frozenset(other.remaining_args)
        )

    def __hash__(self):
        # Turn unhashable attributes into hashable ones
        return hash((
            self.value,
            self.levels,
            self.original_implication,
            tuple(self.key),
            frozenset(self.remaining_args),
        ))


class LocalMemory:
    def __init__(self):
        self.map = {}
        self.remaining_args_tuples_map = {}
        self.remaining_args_normalized_map = {}
        self.max_occurrence_per_key_map = {}
        self.max_key_length = 0
        self.min_max = {}
        self.normalized_keys = set()
        self.normalized_subkeys = set()
        self.admission_map = {}
        self.rejected_map = {}
        self.admission_status_map = {}
        self.products_of_recursion = set()

class Mail:
    def __init__(self):
        self.statements = set()
        self.implications = set()
        self.expr_origin_map = {}

class EquivalenceClass:
    def __init__(self):
        self.variables = set()
        self.equality_levels_map = {}
        self.equality_origin_map = {}

class BodyOfProves:
    def __init__(self):
        self.simple_map = {}
        self.binary_seqs_map = {}
        for num in range(1, 8 + 1):
            self.binary_seqs_map[num] = generate_binary_sequences_as_lists(num)
        self.start_int = 0
        self.to_be_proved = {}
        self.statements = []
        self.encoded_statements = []
        self.statement_levels_map = {}
        self.expr_key = ""
        self.parent_body_of_proves = None
        self.level = -1
        self.local_memory = LocalMemory()
        self.equivalence_classes = []
        self.local_statements = []
        self.local_encoded_statements = []
        self.local_statements_delta = []
        self.mail_in = Mail()
        self.mail_out = Mail()
        self.whole_expressions = set()
        self.eq_class_sttmnt_index_map = {}
        self.is_active = True
        self.is_part_of_recursion = False
        self.delta_number_statements = 0
        self.expr_origin_map = {}
        self.recursion_counter = 0

    def __getstate__(self):
        st = self.__dict__.copy()
        # zero out the big field so pickle won’t recurse into it
        st['simple_map'] = None
        return st

    def __setstate__(self, state):
        # worker will get simple_map == None
        self.__dict__.update(state)

class DependencyItem:
    def __init__(self):
        self.auxies = set()
        self.expr = ""
        self.all_levels_involved = False

class Dependencies:
    def __init__(self):
        self.auxy_original_map = {}
        self.original_auxy_map = {}
        self.original_induction_variable_map = {}
        self.auxy_index = 0
        self.original_index = 0


ARGUMENT_PATTERN = re.compile(r'it_(\d+)_lev_(\d+)_(\d+)')
class EncodedExpression:
    def __init__(self, expression: str):
        core_expr = extract_expression(expression)
        self.name = core_expr

        self.negation = False
        if expression.startswith('!'):
            self.negation = True

        args = get_args(expression)
        self.arguments = tuple([self._parse_argument(arg) for arg in args])

        self.original = expression


    @staticmethod
    def _parse_argument(arg: str):
        match = ARGUMENT_PATTERN.fullmatch(arg)
        if match:
            iteration = int(match.group(1))
            lev = int(match.group(2))
            id = match.group(3)
            return iteration + 1, lev + 1, id
        return 0, 0, arg


global_body_of_proves = BodyOfProves()
max_num_leafs_per_key = 0
global_dependencies = Dependencies()
global_theorem_list = []



def generate_all_mappings(n, m):
    """
    Assumes n, m > 0.
    Returns a dictionary mapping each pair (i, j) with 1 <= i < n and 1 <= j < m
    to a list of functions (represented as lists) from [0, ..., i-1] to [0, ..., j-1].

    Each function is represented as a list of length i, where each element is in the range [0, j).
    """
    function_map = {}
    for i in range(1, n):  # i will be 1, 2, ..., n-1
        for j in range(1, m):  # j will be 1, 2, ..., m-1
            # Generate all functions from [0, i-1] to [0, j-1]
            functions = [list(prod) for prod in itertools.product(range(j), repeat=i)]
            function_map[(i, j)] = functions
    return function_map


def smoothen_expr(expr):
    def smoothen_one_and(expr2):
        local_expr = ""
        found_and = False

        root, number_leafs = parse_expr(expr2)

        def node_to_str(node):
            nonlocal local_expr
            nonlocal found_and

            if node.value[0] == ">":
                if node.left.value == "&":
                    found_and = True

                    left_left_expr = tree_to_expr(node.left.left)
                    left_right_expr = tree_to_expr(node.left.right)
                    right_expr = tree_to_expr(node.right)

                    temp_expr = "(" + ">[]" + left_right_expr + right_expr + ")"
                    local_expr += "(" + node.value + left_left_expr + temp_expr
                else:
                    local_expr = local_expr + "(" + node.value
                    node_to_str(node.left)
                    node_to_str(node.right)
            elif node.value == "&":
                local_expr = local_expr + "(&"
                node_to_str(node.left)
                node_to_str(node.right)
            elif node.value[0:2] == "!>":
                local_expr = local_expr + "!(" + node.value[1:]
                node_to_str(node.left)
                node_to_str(node.right)
            elif node.value == "!&":
                local_expr = local_expr + "!(&"
                node_to_str(node.left)
                node_to_str(node.right)
            elif node.value[0] == '!':
                local_expr = local_expr + "!(" + node.value[2:-1]
            else:
                local_expr = local_expr + "(" + node.value

            local_expr = local_expr + ")"

        node_to_str(root)
        return local_expr, found_and

    found = True
    smoothened_expr = expr[:]
    while found:
        smoothened_expr, found = smoothen_one_and(smoothened_expr)

    return smoothened_expr


# Function to parse the binary tree description
def groom_expr(tree_str):
    index = 0

    # Recursively parse the tree string
    def groom_subexpr(s, side: int):
        nonlocal index

        subexpr_list = []

        if s[index] == '(':
            index = index + 1
            if s[index] == '>':
                index = index + 1

                args_to_remove = get_args(s[index:])
                index = s[index:].find(']') + index + 1

                left_subexpr_list = groom_subexpr(s, 0)  # Process the left child

                right_subexpr_list = groom_subexpr(s, side)  # Process the right child

                for left_expr in left_subexpr_list:
                    for right_expr in right_subexpr_list:
                        subexpr_list.append("(>" + "[" + ",".join(
                            args_to_remove) + "]" + left_expr + right_expr + ")")

            elif s[index] == '&':
                index = index + 1

                left_subexpr_list = groom_subexpr(s, side)  # Process the left child

                right_subexpr_list = groom_subexpr(s, side)  # Process the right child

                if side:
                    subexpr_list = left_subexpr_list + right_subexpr_list
                else:
                    subexpr_list.append("(&" + left_subexpr_list[0] + right_subexpr_list[0] + ")")

            else:
                end_index = s.find(')', index)
                if end_index == -1:
                    raise RuntimeError("No closing ')'.")
                node_label = s[index:end_index]
                index = end_index
                subexpr_list.append("(" + node_label + ")")

        elif s[index:index + 2] == "!(":
            index = index + 2
            if s[index] == '>':
                index = index + 1

                args_to_remove = get_args(s[index:])
                index = s[index:].find(']') + index + 1

                left_subexpr_list = groom_subexpr(s, 0)  # Process the left child

                right_subexpr_list = groom_subexpr(s, 0)  # Process the right child

                subexpr_list.append("!(>" + "[" + ",".join(
                    args_to_remove) + "]" + left_subexpr_list[0] + right_subexpr_list[0] + ")")

            elif s[index] == '&':
                index = index + 1

                left_subexpr_list = groom_subexpr(s, 0)  # Process the left child

                right_subexpr_list = groom_subexpr(s, 0)  # Process the right child

                if side:
                    subexpr_list = left_subexpr_list + right_subexpr_list
                else:
                    subexpr_list.append("!(&" + left_subexpr_list[0] + right_subexpr_list[0] + ")")

            else:
                end_index = s.find(')', index)
                if end_index == -1:
                    raise RuntimeError("No closing ')'.")
                node_label = s[index:end_index]
                node_label = "!(" + node_label + ")"
                index = end_index
                subexpr_list.append(node_label)

        index = index + 1
        return subexpr_list

    expr_list = groom_subexpr(tree_str, 1)

    return expr_list


def list_last_removed_args(expr: str):
    pattern = r">\[[^\[\]]*\]"
    match = re.search(pattern, expr)
    if match:
        # Extract the content between ">[" and "]"
        return match.group()[2:-1].split(',')
    return []

def rename_last_removed(expr: str, start_int: int, iteration: int, level: int):
    assert expr[0:2] == "(>" or expr[0:3] == "!(>"

    args = list_last_removed_args(expr)
    ordered_args = order_by_pattern(expr, set(args))

    replacement_map = {}
    assert len(ordered_args) == 1
    arg = ordered_args[0]
    new_var = "it_" + str(iteration) + "_lev_" + str(level) + "_" + str(start_int)
    replacement_map[arg] = "it_" + str(iteration) + "_lev_" + str(level) + "_" + str(start_int)
    start_int += 1

    expr = \
        replace_keys_in_string(expr, replacement_map)

    return expr, start_int, new_var




def expand_expr(expr: str):
    expanded_expr = expr[:]
    magic_string = "@19023847@"

    for core_expr in core_expression_map:
        if core_expression_map[core_expr][4] == 1:
            index = expr.find(r"(" + core_expr + r"[")
            if index != -1:
                index += 1
                replacing_args = get_args(expr[index:])
                args_to_be_replaced = get_args(core_expression_map[core_expr][5])

                replacement_map = {}
                for ind in range(len(args_to_be_replaced)):
                    replacement_map[args_to_be_replaced[ind]] = replacing_args[ind] + magic_string

                expanded_expr = replace_keys_in_string(core_expression_map[core_expr][3][:], replacement_map)
                expanded_expr = expanded_expr.replace(magic_string, "")

    return expanded_expr



def access_body_of_proves(theorem_key: list[str], body_of_proves_1: BodyOfProves):

    memory_block = body_of_proves_1
    for index in range(len(theorem_key)):
        if theorem_key[index] in memory_block.simple_map:
            memory_block = memory_block.simple_map[theorem_key[index]]
        else:
            return None

    return memory_block



def get_all_args(expression_list: list[str]):
    all_args = set()

    for expression in expression_list:
        all_args.update(get_args(expression))

    return all_args

def find_integers(string):
    integer = r'-?\d+'  # Pattern to match integers (including negative numbers)
    pattern = r'(?<=[\[,])(' + integer + r')(?=[\],])'

    matches = re.findall(pattern, string)  # Find all matches
    result_set = {str(match) for match in matches}  # Convert to a set of strings

    return result_set

def perform_back_replacement(expr2: str, back_replacement_map2: {}):
    ints = find_integers(expr2)
    max_int = max({int(number) for number in ints})
    values = {value for key, value in back_replacement_map2.items() if key != value}
    repl_int = max_int + 1

    temp_map = {}
    for number in ints:
        if number in values:
            temp_map[number] = str(repl_int)
            repl_int += 1

    temp_expr = replace_keys_in_string(expr2, temp_map)
    back_replaced_expr = replace_keys_in_string(temp_expr, back_replacement_map2)



    return back_replaced_expr

def check_local_memory(expression_list: list[str], memory_block: BodyOfProves, iteration: int, tple):
    def replace_u_substrings(s):
        """
        Replace substrings in the string s that have the structure 'u_' followed by any string,
        provided these substrings are immediately preceded by '[' or ',' and immediately followed by ']' or ','.

        For example:
          "[u_hello,u_world]"  -->  "[hello,world]"

        Parameters:
          s (str): The input string.

        Returns:
          str: The modified string with the 'u_' prefix removed.
        """
        # Pattern explanation:
        #   (?<=[\[,])   - Positive lookbehind: the match must be preceded by '[' or ','
        #   u_           - Literal "u_"
        #   (.+?)        - Non-greedy capture of any characters (at least one character)
        #   (?=[\],])    - Positive lookahead: the match must be followed by ']' or ','
        pattern = r'(?<=[\[,])u_(.+?)(?=[\],])'

        # Replace the entire match with the captured group (i.e., the part after "u_")
        return re.sub(pattern, r'\1', s)

    pure = True
    for expression in expression_list:
        args = set(get_args(expression))

        args = {arg for arg in args if extract_max_iteration_number(arg) is not None}

        if not args.issubset(memory_block.local_memory.products_of_recursion):
            pure = False
            break

    combined_expression = ""
    combined_levels = set()
    for subexpr in expression_list:
        combined_expression += subexpr
        combined_levels.update(memory_block.statement_levels_map[subexpr])

    all_args = get_all_args(expression_list)
    if len(memory_block.local_memory.map) != 0:

        for st in sorted(list(memory_block.local_memory.remaining_args_normalized_map.keys())):
            if st.issubset(all_args):
                if tple in memory_block.local_memory.remaining_args_normalized_map[st]:
                    replacement_map = {arg: "u_" + arg for arg in st}
                    replaced_expr = replace_keys_in_string(combined_expression, replacement_map)

                    normalized, norm_map = normalize_variables(replaced_expr)
                    bck_rplcmnt_mp = {value: key for key, value in norm_map.items()}

                    if normalized in memory_block.local_memory.map:
                        vlue_set = memory_block.local_memory.map[normalized]
                        for local_memory_value in sorted(vlue_set, key=lambda item: item.value):


                            rpl_expr1 = replace_keys_in_string(local_memory_value.value, bck_rplcmnt_mp)
                            rpl_expr2 = replace_u_substrings(rpl_expr1)

                            if "marker" not in local_memory_value.value:
                                temp_levels = copy.copy(combined_levels)
                                temp_levels.update(local_memory_value.levels)

                                add_expr_to_memory_block(rpl_expr2,
                                                         memory_block,
                                                         iteration,
                                                         1,
                                                         temp_levels,
                                                         ["implication", local_memory_value.original_implication] +
                                                         sorted(expression_list))

                            else:

                                if not pure:
                                    continue


                                replaced_key =\
                                    [replace_u_substrings(replace_keys_in_string(element, bck_rplcmnt_mp))
                                     for element in local_memory_value.key]

                                remaining_args = set(local_memory_value.remaining_args)

                                if rpl_expr2 in memory_block.local_memory.admission_map:
                                    memory_block.local_memory.admission_map[rpl_expr2].add((tuple(replaced_key),
                                                                                           frozenset(remaining_args),
                                                                                        standard_max_admission_depth,
                                                                                        standard_max_secondary_number,
                                                                                            False))
                                else:
                                    memory_block.local_memory.admission_map[rpl_expr2] = \
                                        {(tuple(replaced_key),
                                          frozenset(remaining_args),
                                          standard_max_admission_depth,
                                          standard_max_secondary_number,
                                          False)}

                                memory_block.local_memory.admission_status_map[rpl_expr2] = False

                                revisit_rejected(rpl_expr2, memory_block)

    return



import re
from functools import lru_cache
from typing import Dict, Tuple

_ALLOWED_REGEX_PATTERN_NORM_VARS = re.compile(r'(?<=[\[,])([A-Za-z0-9_]+)(?=[],])')

@lru_cache(maxsize=6000000)
def _normalize_variables_cached(expression: str, ignore_u: bool) -> Tuple[str, tuple[tuple[str, str], ...]]:
    mapping: Dict[str, str] = {}

    def replacement(match: re.Match) -> str:
        token = match.group(1)
        if ignore_u and token.startswith("u_"):
            return token
        if token not in mapping:
            mapping[token] = str(len(mapping) + 1)
        return mapping[token]

    normalized = _ALLOWED_REGEX_PATTERN_NORM_VARS.sub(replacement, expression)
    mapping_tuple: tuple[tuple[str, str], ...] = tuple((k, v) for k, v in mapping.items())
    return normalized, mapping_tuple

def normalize_variables(expression: str, ignore_u: bool = True) -> Tuple[str, Dict[str, str]]:
    normalized, mapping_tuple = _normalize_variables_cached(expression, ignore_u)
    return normalized, dict(mapping_tuple)




def get_global_key(memory_block: BodyOfProves):
    global_key = []

    current_memory_block = memory_block
    while True:
        if current_memory_block.expr_key != "":
            global_key.append(current_memory_block.expr_key)
        if current_memory_block.parent_body_of_proves is not None:
            current_memory_block = current_memory_block.parent_body_of_proves
        else:
            break

    global_key.reverse()

    return global_key

def reconstruct_implication(key: list[str], value: str):
    #key = ["(NaturalNumbers[1,2,3,4,5])", "[6,7,8](in3[6,7,8,4])","(in3[6,9,10,4])","(in3[6,7,9,5])"]
    #value = "(in2[10,8,3])"
    #implication = "(>[3,4,5](NaturalNumbers[1,2,3,4,5])(>[6,7,8](in3[6,7,8,4])(>[9,10](in3[6,9,10,4])(>[](in3[6,7,9,5])(in2[10,8,3])))))"


    implication = value
    chain = key + [value]
    args_chain = [get_args(element) for element in chain]

    removed_args = set()
    counter_map = {}
    for args in args_chain:
        for arg in args:
            if arg[:2] == "u_":
                continue

            if arg in counter_map:
                counter_map[arg] += 1
            else:
                counter_map[arg] = 1

            if counter_map[arg] > 1:
                removed_args.add(arg)

    when_removed = [[] for _ in range(len(key))]
    for arg in removed_args:
        for index, args in enumerate(args_chain):
            if arg in args:

                when_removed[index].append(arg)
                break

    for index in reversed(range(len(key))):
        implication = "(>[" + ",".join(sorted(when_removed[index], key=int)) + "]" + key[index] + implication + ")"

    return implication


def calc_num_diff_args(input_key: list[str], binary_list: list[int]):
    extr_expressions = []
    diff_args = set()

    for index, element in enumerate(input_key):
        if binary_list[index]:
            if element[0] != "!":
                extr_expr = extract_expression(element)
            else:
                extr_expr = "!" + extract_expression_from_negation(element)

            extr_expressions.append(extr_expr)

            diff_args.update(get_args(element))

    return tuple(sorted(extr_expressions)), len(diff_args)

def normalize_subkey(input_key: list[str]):
    subkey = "".join(input_key)
    subkey, mp = normalize_variables(subkey, False)

    return subkey

def update_admission_map(hash_memory: LocalMemory,
                         key: list[str],
                         remaining_args: frozenset[str],
                         max_admission_depth: int,
                         max_secondary_number: int,
                         part_of_recursion: bool):
    replacement_map = {}
    for arg in remaining_args:
        replacement_map["u_" + arg] = arg

    replaced_key = [replace_keys_in_string(element, replacement_map) for element in key]

    for element in replaced_key:
        core_expr = extract_expression(element)
        if core_expr not in OPERATORS:
            continue

        args = set(get_args(element))

        without_remaining = args - remaining_args

        if len(without_remaining) == 1:
            removing_map = {next(iter(without_remaining)): "marker"}
            removed = replace_keys_in_string(element, removing_map)

            if removed in hash_memory.admission_map:
                hash_memory.admission_map[removed].add((tuple(replaced_key),
                                                        remaining_args,
                                                        max_admission_depth,
                                                        max_secondary_number,
                                                        part_of_recursion))
                hash_memory.admission_status_map[removed] = (part_of_recursion or
                                                             hash_memory.admission_status_map[removed])
            else:
                hash_memory.admission_map[removed] = set()
                hash_memory.admission_map[removed].add((tuple(replaced_key),
                                                        remaining_args,
                                                        max_admission_depth,
                                                        max_secondary_number,
                                                        part_of_recursion))
                hash_memory.admission_status_map[removed] = part_of_recursion

    return

def update_admission_map2(hash_memory: LocalMemory, marked_expr: str, var: str):
    parts = marked_expr.split("marker")

    assert len(parts) == 2

    # take a one-time snapshot of the entries
    original_keys = list(hash_memory.admission_map[marked_expr])

    for key, remaining_args, max_admission_depth, max_secondary_number, part_of_recursion in original_keys:
        key = list(key)

        for element in key:
            if element.startswith(parts[0]) and element.endswith(parts[1]):
                var_for_repl = element.removeprefix(parts[0]).removesuffix(parts[1])
                replacement_map = {var_for_repl: var}

                repl_key = [replace_keys_in_string(element, replacement_map) for element in key]
                new_rem_args = remaining_args.union([var])

                update_admission_map(hash_memory,
                                     repl_key,
                                     new_rem_args,
                                     max_admission_depth,
                                     max_secondary_number,
                                     part_of_recursion)

    return

def get_remaining_args(key: list[str]):
    remaining_args = set()
    for element in key:
        args = get_args(element)
        for arg in args:
            if arg.startswith('u_'):
                remaining_args.add(arg)

    return frozenset(remaining_args)

def implication_is_qualified(key: list[str], value: str):
    counter_key = 0
    result = False

    for element in key:
        if element.startswith('(in3') or element.startswith('(in3'):
            counter_key += 1

    value_cond = value.startswith('(in3[') or value.startswith('(in3[')

    if counter_key > min_num_operators_key and value_cond:
        result = True

    return result

def make_normalized_keys_for_admission(key: list[str], hash_memory: LocalMemory, value: str):



    if not implication_is_qualified(key, value):
        return

    for index in range(len(key)):
        if extract_expression(key[index]) not in OPERATORS:
            continue

        binary = [1] * len(key)
        binary[index] = 0

        subkey = [key[index2] for index2 in range(len(key)) if binary[index2]]

        subkey_args = set()
        for element in subkey:
            subkey_args.update(get_args(element))

        args = get_args(key[index])
        output_arg = args[len(args) - 2]
        if output_arg not in subkey_args:
            replacement_map = {output_arg: "marker"}
            replaced = replace_keys_in_string(key[index], replacement_map)

            make_normalized_subkeys(subkey, hash_memory)

            variants = create_variants(subkey, replaced)

            for ky, vlue, mp in variants:
                remaining_args = get_remaining_args(subkey)

                local_memory_value = LocalMemoryValue()
                local_memory_value.value = vlue

                repl_key = [replace_keys_in_string(element, dict(mp)) for element in key]

                local_memory_value.key = repl_key
                local_memory_value.remaining_args = remaining_args
                if ky in hash_memory.map:
                    hash_memory.map[ky].add(local_memory_value)
                else:
                    hash_memory.map[ky] = set()
                    hash_memory.map[ky].add(local_memory_value)

                normalized_key, norm_mp = normalize_variables(ky, False)
                hash_memory.normalized_keys.add(normalized_key)

                if frozenset(remaining_args) in hash_memory.remaining_args_normalized_map:
                    hash_memory.remaining_args_normalized_map[remaining_args].add(normalized_key)
                else:
                    hash_memory.remaining_args_normalized_map[remaining_args] = {normalized_key}

    return

def make_normalized_subkeys(key: list[str], hash_memory: LocalMemory):
    permuts = _ALL_PERMUTATIONS_ANA[len(key)]

    ids = []
    for expression in key:
        ids.append(extract_expression(expression))

    for permut in permuts:
        temp_list = []
        for index in permut:
            temp_list.append(key[index])

        for index in range(0,len(key)):
            to_break = False
            for index2 in range(0, index):
                if ids[permut[index2]] > ids[permut[index2 + 1]]:
                    to_break = True
                    break
            if to_break:
                break

            subkey_variant = "".join(temp_list[:index + 1])
            subkey_variant, mp = normalize_variables(subkey_variant, False)
            hash_memory.normalized_subkeys.add(subkey_variant)

    return

def add_to_hash_memory(key: list[str],
                       value: str,
                       remaining_args: frozenset[str],
                       hash_memory: LocalMemory,
                       levels: set[int],
                       original_implication: str,
                       max_admission_depth: int,
                       max_secondary_number: int,
                       part_of_recursion: bool):


    make_normalized_keys_for_admission(key, hash_memory, value)

    update_admission_map(hash_memory,
                         key + [value],
                         remaining_args,
                         max_admission_depth,
                         max_secondary_number,
                         part_of_recursion)

    full_tuple, num = calc_num_diff_args(key, [1] * len(key))

    variants = create_variants(key, value)

    make_normalized_subkeys(key, hash_memory)

    for ky, vlue, mp in variants:
        local_memory_value = LocalMemoryValue()
        local_memory_value.value = vlue
        local_memory_value.levels = frozenset(levels)
        local_memory_value.original_implication = original_implication
        local_memory_value.remaining_args = remaining_args
        if ky in hash_memory.map:
            hash_memory.map[ky].add(local_memory_value)
        else:
            hash_memory.map[ky] = set()
            hash_memory.map[ky].add(local_memory_value)

        normalized_key, norm_mp = normalize_variables(ky, False)
        hash_memory.normalized_keys.add(normalized_key)

        if remaining_args in hash_memory.remaining_args_normalized_map:
            hash_memory.remaining_args_normalized_map[remaining_args].add(normalized_key)
        else:
            hash_memory.remaining_args_normalized_map[remaining_args] = {normalized_key}



    core_expression_counter_map = {}
    for sttmnt in key:
        core_expression = extract_expression(sttmnt)
        if core_expression in core_expression_counter_map:
            core_expression_counter_map[core_expression] += 1
        else:
            core_expression_counter_map[core_expression] = 1

    for sttmnt in core_expression_counter_map:
        if sttmnt in hash_memory.max_occurrence_per_key_map:
            new_val = max(core_expression_counter_map[sttmnt], hash_memory.max_occurrence_per_key_map[sttmnt])
            hash_memory.max_occurrence_per_key_map[sttmnt] = new_val
        else:
            hash_memory.max_occurrence_per_key_map[sttmnt] =\
                core_expression_counter_map[sttmnt]

    if remaining_args in hash_memory.remaining_args_tuples_map:
        hash_memory.remaining_args_tuples_map[remaining_args].add(full_tuple)
    else:
        hash_memory.remaining_args_tuples_map[remaining_args] = {full_tuple}

    hash_memory.max_key_length = max(len(key), hash_memory.max_key_length)








_pattern = re.compile(r'it_(\d+)_lev_\d+_')

def extract_max_iteration_number(s: str) -> int:
    """
    Finds all occurrences of the pattern "it_<int1>_lev_<int2>_" in the string,
    extracts the int1 values, and returns the maximum one.

    Parameters:
        s (str): The input string.

    Returns:
        int: The maximum int1 integer found, or None if no matches exist.
    """


    # Using a generator expression to avoid building an intermediate list.
    return max((int(match.group(1)) for match in _pattern.finditer(s)), default=None)

"""
_pattern_counting = re.compile(r'it_')

def count_pattern_occurrences(input_string):
    return input_string.count('it_')

"""
_pattern_counting = re.compile(r'it_\d+_lev_\d+_\d+')



def count_pattern_occurrences(input_string, local_memory: LocalMemory):
    matches = re.findall(_pattern_counting, input_string)

    counter = 0
    for match in matches:
        if match not in local_memory.products_of_recursion:
            counter += 1


    return counter


def send_mail(memory_block: BodyOfProves, mail: Mail):
    #print("memory_block.expr_key: " + memory_block.expr_key)
    if mail.statements or mail.implications:

        #assert mail.expr_origin_map

        for key in memory_block.simple_map:
            subsequent = memory_block.simple_map[key]
            for mail_statement in mail.statements:
                subsequent.mail_in.statements.add(mail_statement[:])

            for impl_chain, impl_head, impl_remaining_args_key, impl_levels, or_impl in mail.implications:
                subsequent.mail_in.implications.add((copy.copy(impl_chain),
                                                     copy.copy(impl_head),
                                                     copy.copy(impl_remaining_args_key),
                                                     copy.copy(impl_levels),
                                                     copy.copy(or_impl)))

            subsequent.mail_in.expr_origin_map =\
                mail.expr_origin_map | subsequent.mail_in.expr_origin_map



            send_mail(subsequent, mail)


def filter_statements(statements: list[str],
                      local_memory: LocalMemory):
    filtered = []

    for sttment in statements:
        normed = normalize_subkey([sttment])
        if normed not in local_memory.normalized_subkeys:
            continue

        filter_statements_itr = extract_max_iteration_number(sttment)
        if filter_statements_itr is not None:
            if filter_statements_itr > max_iteration_number_variable:
                continue


        filtered.append(sttment)

    return filtered

def perform_elementary_logical_step(body_of_proves: BodyOfProves):
    def pre_evaluate_key(key: list):
        """
        Evaluate the key (a list of statements) by counting the occurrences of each core expression.
        Returns a tuple (local_good, global_good) after comparing the counts against maximum allowed
        occurrences in local and global memory maps.
        """






        local_good = True

        secondary_counter = 0
        for key_expression in key:
            secondary_counter += count_pattern_occurrences(key_expression, body_of_proves.local_memory)

        if secondary_counter > max_number_secondary_variables:
            return False

        normalized_subkey = normalize_subkey(key)

        # Check local memory restrictions for binary analysis
        if normalized_subkey not in body_of_proves.local_memory.normalized_subkeys:
            local_good = False

        return local_good



    import copy

    def make_mandatory_statement_lists1(local_memory: LocalMemory,local_statements: list[str]):
        temp_list = sorted(filter_statements(local_statements, local_memory))

        return [[element] for element in temp_list]

    def make_mandatory_statement_lists2(local_memory: LocalMemory,
                                        first_layer: list[str],
                                        second_layer: list[str]):
        mandatory_statement_lists = []

        if first_layer and second_layer:
            filtered_first_layer = sorted(filter_statements(first_layer, local_memory))
            filtered_second_layer = sorted(filter_statements(second_layer, local_memory))

            mandatory_statement_lists = []

            for filtered_first in filtered_first_layer:
                for filtered_second in filtered_second_layer:
                    if filtered_first != filtered_second:
                        statement_list = [filtered_first, filtered_second]

                        if pre_evaluate_key(statement_list):
                            mandatory_statement_lists.append(statement_list)

        return mandatory_statement_lists

    def sort_expressions(strings):
        """
        Sorts a list of strings lexicographically (in ascending order)
        using the extracted expressions from each string as the key.

        The extraction is applied only once per element.
        """
        # Decorate: create a list of tuples (extracted key, original string)
        decorated = [(extract_expression(s), s) for s in strings]
        # Sort the decorated list based on the extracted expression
        decorated.sort(key=lambda pair: pair[0])
        # Undecorate: extract the sorted original strings
        return [original for _, original in decorated]


    def merge_insert_sorted(list_a, values_list_a, list_b, values_list_b):
        """
        Merges two lists of strings by inserting elements from list_b into list_a.

        - list_a is already sorted in ascending lexicographical order
          according to the keys in values_list_a.
        - Each element in list_b has a corresponding key in values_list_b.

        The function returns a new list containing the elements of list_a and list_b,
        where the elements from list_b have been inserted into the correct positions
        such that the resulting list is sorted lexicographically by their keys.

        Parameters:
            list_a (List[str]): A list of strings sorted by values_list_a.
            values_list_a (List[str]): A list of sort keys corresponding to list_a.
            list_b (List[str]): A list of strings that will be merged into list_a.
            values_list_b (List[str]): A list of sort keys corresponding to list_b.

        Returns:
            List[str]: A new list containing all elements from list_a and list_b in
                       ascending lexicographical order based on their keys.
        """
        # Create copies to avoid modifying the original lists
        merged_list = list(list_a)
        merged_keys = list(values_list_a)

        # Insert each element of list_b into the merged list at the proper position.
        for element, key in zip(list_b, values_list_b):
            # Find the insertion index where key would fit in the sorted merged_keys.
            index = bisect.bisect_left(merged_keys, key)
            merged_keys.insert(index, key)
            merged_list.insert(index, element)

        return merged_list


    def generate_requests(local_memory: LocalMemory,
                          mandatory_statement_lists: list[list[str]],
                          all_statements: list[str]):
        """
        Iteratively generates requests by combining subsets of statements from
        body_of_proves.statements (using their indices) with each statement from
        body_of_proves.local_statements. A request (a list of statements) is added to the output
        list if pre_evaluate_key returns True for either the local or global condition.

        Branch-and-bound pruning is applied: a branch is extended only if at least one combination
        in that branch succeeds the test.

        Returns:
            A list of tuples (rqst, local_good, global_good, sttmnt_tple) where 'rqst' is a valid request.
        """
        rqsts = []

        filtered_all_statements = sort_expressions(filter_statements(all_statements, local_memory))


        values_fas = [extract_expression(filtered) for filtered in filtered_all_statements]
        values_msl = [[extract_expression(sttmnt) for sttmnt in sttmnt_lst] for sttmnt_lst in mandatory_statement_lists]


        # Process each local statement individually.
        for mandatory_statement_list in mandatory_statement_lists:
            rqst = mandatory_statement_list
            local_good = pre_evaluate_key(rqst)
            normalized_req = normalize_subkey(rqst)
            if local_good and normalized_req in local_memory.normalized_keys:
                rqsts.append((copy.deepcopy(rqst), local_good, normalized_req))

        # Use a stack for iterative backtracking over indices from body_of_proves.statements.
        # Instead of a viability mask (list of booleans), we'll use a set of viable indices.
        initial_viable_set = set(range(len(mandatory_statement_lists)))
        stack = deque()
        # Each element is a tuple: (next_start_index, current_indices_subset, viable_set)
        stack.append((0, [], initial_viable_set))

        max_key_len = local_memory.max_key_length

        size_one_mandatory_statement_list = len(mandatory_statement_lists[0])

        while stack:
            start, indices_subset, viable_set = stack.pop()
            if len(indices_subset) < max_key_len - size_one_mandatory_statement_list:
                for i in range(start, len(filtered_all_statements)):
                    new_indices_subset = indices_subset + [i]
                    # Build the base request using the selected indices.
                    base_request = [filtered_all_statements[idx] for idx in new_indices_subset]
                    base_request_values = [values_fas[idx] for idx in new_indices_subset]

                    # Copy the current viable set for this branch.
                    new_viable_set = viable_set.copy()
                    once_successful = False
                    # Iterate only over the indices that are still viable.
                    for j in list(new_viable_set):
                        msl = mandatory_statement_lists[j]
                        to_continue2 = False
                        for sttmnt in msl:
                            if sttmnt in base_request:
                                to_continue2 = True
                        if to_continue2:
                            continue

                        #rqst_old = base_request + msl
                        rqst = merge_insert_sorted(base_request,
                                                   base_request_values,
                                                   msl,
                                                   values_msl[j])

                        local_good = pre_evaluate_key(rqst)
                        if local_good:
                            normalized_req = normalize_subkey(rqst)
                            if normalized_req in local_memory.normalized_keys:

                                rqsts.append((copy.deepcopy(rqst), local_good, normalized_req))
                            once_successful = True
                        else:
                            # Remove this index from the viable set for this branch.
                            new_viable_set.remove(j)

                    # Only extend this branch if at least one combination succeeded.
                    if once_successful:
                        stack.append((i + 1, new_indices_subset, new_viable_set))

        return sorted(rqsts)




    if not body_of_proves.is_active:
        return body_of_proves

    #print(body_of_proves.expr_key)
    #print(len(body_of_proves.statements))

    for chain, head, remaining_args_key, levels, or_impl in body_of_proves.mail_in.implications:


        add_to_hash_memory(list(chain),
                           head,
                           remaining_args_key,
                           body_of_proves.local_memory,
                           levels,
                           or_impl,
                           standard_max_admission_depth,
                           standard_max_secondary_number,
                           False)

    body_of_proves.expr_origin_map =\
        body_of_proves.mail_in.expr_origin_map | body_of_proves.expr_origin_map

    working_hash_memory = LocalMemory()
    for chain, head, remaining_args_key, levels,or_impl in body_of_proves.mail_in.implications:
        add_to_hash_memory(list(chain),
                           head,
                           remaining_args_key,
                           working_hash_memory,
                           levels,
                           or_impl,
                           standard_max_admission_depth,
                           standard_max_secondary_number,
                           False)


    working_hash_memory_requests = []
    if len(working_hash_memory.map) > 0:

        mandatory_statement_lists1 =\
            make_mandatory_statement_lists1(working_hash_memory,
                                            body_of_proves.local_statements)

        if mandatory_statement_lists1:
            working_hash_memory_requests = generate_requests(working_hash_memory,
                                                             mandatory_statement_lists1,
                                                             body_of_proves.statements)


    for statement, levels in body_of_proves.mail_in.statements:
        if statement not in body_of_proves.statement_levels_map:

            origin = None
            if track_history:

                origin = body_of_proves.mail_in.expr_origin_map[statement]
            add_expr_to_memory_block(statement,
                                     body_of_proves,
                                     -1,
                                     3,
                                     set(levels),
                                     origin)

    mandatory_new_local_lists1 = \
        make_mandatory_statement_lists1(body_of_proves.local_memory,
                                        body_of_proves.local_statements_delta)

    new_local_requests = []
    if mandatory_new_local_lists1:
        new_local_requests = generate_requests(body_of_proves.local_memory,
                                                mandatory_new_local_lists1,
                                                body_of_proves.statements)

    mail_statements = [element[0] for element in body_of_proves.mail_in.statements]
    mandatory_local_mail_lists2 = (
        make_mandatory_statement_lists2(body_of_proves.local_memory, body_of_proves.local_statements, mail_statements))

    local_mail_requests = []
    if mandatory_local_mail_lists2:
        local_mail_requests = generate_requests(body_of_proves.local_memory,
                                                mandatory_local_mail_lists2,
                                                body_of_proves.statements)


    requests = working_hash_memory_requests + new_local_requests + local_mail_requests

    body_of_proves.mail_in.statements.clear()
    body_of_proves.mail_in.implications.clear()
    body_of_proves.mail_in.expr_origin_map.clear()

    body_of_proves.local_statements_delta = []

    for request in requests:
        combined_expression = ""
        combined_levels = set()



        to_continue = False
        for subexpr in request[0]:
            combined_expression += subexpr
            if subexpr in body_of_proves.statement_levels_map:
                combined_levels.update(body_of_proves.statement_levels_map[subexpr])
            else:
                #it can happen that subexpr was removed during update_equivalence_classes
                to_continue = True

        if to_continue:
            continue


        itr = extract_max_iteration_number(combined_expression)
        if itr is None:
            itr = -1

        check_local_memory(request[0], body_of_proves, itr + 1, request[2])

    send_mail(body_of_proves, body_of_proves.mail_out)
    body_of_proves.mail_out.statements.clear()
    body_of_proves.mail_out.implications.clear()
    body_of_proves.mail_out.expr_origin_map.clear()




    """
    mb = global_body_of_proves.simple_map['(NaturalNumbers[1,2,3,4,5])']
    mb = mb.simple_map['(in3[2,6,7,4])']
    mb = mb.simple_map['(in2[rec,6,3])']
    """

    for expr in sorted(body_of_proves.simple_map):
        # if body_of_proves.simple_map[expr].is_active:
        perform_elementary_logical_step(body_of_proves.simple_map[expr])




    return None

def prove():

    for iteration in range(0, max_iteration_number_proof):
        print("Hash burst: " + str(iteration))
        perform_elementary_logical_step(global_body_of_proves)

        """
        cur, peak = tracemalloc.get_traced_memory()
        print(f"Python heap now: {cur / 2 ** 20:,.1f} MB  |  peak: {peak / 2 ** 20:,.1f} MB")

        for idx, stat in enumerate(tracemalloc.take_snapshot().statistics("lineno")[:10], 1):
            frame = stat.traceback[0]
            print(f"{idx:>2}. {frame.filename}:{frame.lineno} "
                  f"{stat.size / 2 ** 20:6.1f} MB  "
                  f"— {linecache.getline(frame.filename, frame.lineno).strip()}")
        """




# Precompile the regex for form1 using a DEFINE block.
# The DEFINE block declares a recursive group "bal" that matches either "(" or "!(",
# followed by any content (or nested "bal") and a closing ")".
_FORM1_REGEX = regex.compile(r'''
    (?(DEFINE)
        (?P<bal> !?\( (?: [^()]+ | (?&bal) )* \) )
    )
    ^(!?\()         # Outer opening, optionally with "!"
    >\[[^]]*\]     # A literal ">" and a bracketed part (ignored)
    (?P<b>(?&bal))  # First balanced group (capturing b) including its original brackets
    (?P<c>(?&bal))  # Second balanced group (capturing c) including its original brackets
    \)$            # Outer closing
    ''', regex.VERBOSE)

# Pattern for form2: if the input does not follow form1, we expect a balanced structure.
_FORM2_REGEX = regex.compile(r'^!?\(.*\)$')

def extract_values_regex(s: str):
    s = s.strip()
    m = _FORM1_REGEX.match(s)
    if m:
        # Return the captured groups exactly as found.
        return m.group("b"), m.group("c")
    if _FORM2_REGEX.match(s):
        return s  # Form2: return unchanged.
    raise ValueError("Input string does not match expected formats.")







def revisit_rejected(marked_expr: str, memory_block: BodyOfProves):
    rm = memory_block.local_memory.rejected_map

    if marked_expr in rm:
        rejected_exprs = rm[marked_expr]

        rejected_exprs_copy = copy.deepcopy(rejected_exprs)
        for rejected_expr, expanded_expr, iteration in rejected_exprs_copy:
            involved_levels = memory_block.statement_levels_map[expanded_expr]

            implications, statements, memory_block.start_int = disintegrate_expr(expanded_expr,
                                                                                memory_block.start_int,
                                                                                iteration,
                                                                                memory_block.level,
                                                                                memory_block.local_memory)
            local_origin = None
            if track_history:
                local_origin = ["disintegration", expanded_expr]

            for implication in implications:
                ky, vlue = extract_key_value(implication)
                remaining_args_key = extract_difference(ky)

                remaining_args_impl = extract_difference(implication)
                replacement_map = {remaining_arg: "u_" + remaining_arg for remaining_arg in remaining_args_impl}
                replaced_impl = replace_keys_in_string(implication, replacement_map)

                temp_chain = []
                head = disintegrate_implication(replaced_impl, temp_chain)
                chain = []
                for element in temp_chain:
                    chain.append(element[0])

                add_to_hash_memory(chain,
                                   head,
                                   frozenset(remaining_args_key),
                                   memory_block.local_memory,
                                   involved_levels,
                                   implication,
                                   standard_max_admission_depth,
                                   standard_max_secondary_number,
                                   False)
                memory_block.mail_out.implications.add((tuple(copy.deepcopy(chain)),
                                                        copy.deepcopy(head),
                                                        frozenset(copy.deepcopy(remaining_args_key)),
                                                        frozenset(involved_levels),
                                                        implication))


                if track_history:
                    if implication not in memory_block.expr_origin_map:
                        memory_block.expr_origin_map[implication] = local_origin
                        memory_block.mail_out.expr_origin_map[implication] = local_origin

            for statement in statements:

                add_expr_to_memory_block(statement,
                                         memory_block,
                                         iteration,
                                         0,
                                         involved_levels,
                                         local_origin)

        memory_block.local_memory.rejected_map.pop(marked_expr)


    return



def make_marked_expr(expr: str, var:str):
    replacement_map = {var: "marker"}

    marked_expr = replace_keys_in_string(expr, replacement_map)

    return marked_expr

def is_admitted(hash_memory: LocalMemory, expr: str, var: str, marked_expr:str):
    result = False

    core_expr = extract_expression(expr)
    if core_expr in OPERATORS:
        parts = expr.split(var)

        assert len(parts) == 2




        if marked_expr in hash_memory.admission_map:
            tuples = list(hash_memory.admission_map[marked_expr])


            for key, remaining_args, max_admission_depth, max_secondary_number, part_of_recursion in tuples:
                mn = extract_max_iteration_number(var)

                cnt = count_pattern_occurrences(expr, hash_memory)


                if mn <= max_admission_depth and cnt <= max_secondary_number:
                    result = True

                    assert marked_expr in hash_memory.admission_status_map
                    if hash_memory.admission_status_map[marked_expr]:
                        hash_memory.products_of_recursion.add(var)

                    key = list(key)

                    for element in key:
                        if element.startswith(parts[0]) and element.endswith(parts[1]):
                            var_for_repl = element.removeprefix(parts[0]).removesuffix(parts[1])
                            replacement_map = {var_for_repl: var}

                            repl_key = [replace_keys_in_string(element, replacement_map) for element in key]
                            new_rem_args = remaining_args.union([var])

                            update_admission_map(hash_memory,
                                                 repl_key,
                                                 new_rem_args,
                                                 max_admission_depth,
                                                 max_secondary_number,
                                                 part_of_recursion)



    return result

def update_rejected_map(expr:str,
                        marked_expr: str,
                        expanded_expr: str,
                        hash_memory: LocalMemory,
                        iteration: int):
    if marked_expr in hash_memory.rejected_map:
        hash_memory.rejected_map[marked_expr].add((expr, expanded_expr, iteration))
    else:
        hash_memory.rejected_map[marked_expr] = set()
        hash_memory.rejected_map[marked_expr].add((expr,expanded_expr, iteration))

def disintegrate_expr(expr: str,
                      start_int: int,
                      iteration: int,
                      level: int,
                      hash_memory: LocalMemory):
    implications = set()
    statements = set()

    def disintegrate_expr_node(expr2: str):
        nonlocal start_int
        nonlocal iteration
        nonlocal level


        subexprs_after_grooming = groom_expr(expr2)
        renamed_subexprs_after_grooming = []
        for subexpr in subexprs_after_grooming:
            subexpr = smoothen_expr(subexpr)
            renamed_subexprs_after_grooming.append(subexpr)

        for subexpr in renamed_subexprs_after_grooming:
            if subexpr[:2] == "(>":
                implications.add(subexpr)

            elif subexpr[:3] == "!(>":
                renamed_subdef, start_int, new_var = rename_last_removed(subexpr, start_int, iteration, level)

                left_expr, right_expr = extract_values_regex(renamed_subdef)
                if right_expr[0] == "!":
                    right_expr = right_expr[1:len(right_expr)]
                else:
                    right_expr = "!" + right_expr

                left_marked = make_marked_expr(left_expr, new_var)
                right_marked= make_marked_expr(right_expr, new_var)

                if (is_admitted(hash_memory, left_expr, new_var, left_marked)
                        or is_admitted(hash_memory, right_expr, new_var, right_marked)):
                    disintegrate_expr_node(left_expr)
                    disintegrate_expr_node(right_expr)
                else:
                    update_rejected_map(renamed_subdef,
                                        left_marked,
                                        expr,
                                        hash_memory,
                                        iteration)

                    update_rejected_map(renamed_subdef,
                                        right_marked,
                                        expr,
                                        hash_memory,
                                        iteration)

                    start_int -= 1

            elif subexpr[:2] == "!(&":
                RuntimeError("Should not happen.")
            else:
                statements.add(subexpr)

    disintegrate_expr_node(expr)

    return implications, statements, start_int




# Precompile the regex pattern used in filtering iterations and reducing equivalence classes.
FILTER_PATTERN = re.compile(r'it_(\d+)_lev_\d+_\d+')


def filter_iterations(expr2: str, eq_class: EquivalenceClass) -> bool:
    # Build a set of eq_class strings that exactly match the FILTER_PATTERN.
    eq_class_matches = {s for s in eq_class.variables if FILTER_PATTERN.fullmatch(s)}



    if not eq_class_matches:
        return True  # Nothing to compare against.

    # Determine the lexicographically smallest candidate in eq_class_matches.
    min_candidate = min(eq_class_matches)

    # Iterate over matches in expr2 using finditer for early exit.
    for m in FILTER_PATTERN.finditer(expr2):
        candidate = m.group(0)
        # If candidate is in eq_class and isn't the minimal element, fail early.
        if candidate in eq_class_matches and candidate != min_candidate:
            return False
        """
        if candidate in eq_class_matches and len(eq_class_matches) < len(eq_class.variables):
            args = set(get_args(expr2))
            diff = non_matches - args
            if len(diff) == 0:
                return False
        """




    return True


def reduce_eq_class(eq_class: set) -> set:
    """
    Returns a new set that keeps all members of eq_class that do not match FILTER_PATTERN,
    plus the lexicographically smallest member that does match (if any).
    """
    matching_members = {s for s in eq_class if FILTER_PATTERN.fullmatch(s)}
    non_matching_members = eq_class - matching_members


    if non_matching_members:
        return non_matching_members
    else:
        return {min(matching_members)}




def apply_equivalence_class(clss: EquivalenceClass,
                            expr2: str,
                            memory_block,
                            levels: set[int],
                            new_statements: list[str]) -> None:
    expr_levels_map = {}
    expr_origin_map = {}

    # Cache the argument list once.
    args_expr = get_args(expr2)
    # Compute indices where the argument is in the equivalence class.
    indices = [i for i, arg in enumerate(args_expr) if arg in clss.variables]
    # Precompute the reduced equivalence class list.
    eq_list = list(reduce_eq_class(clss.variables))

    # Precompute the base expression and wrapping tokens.
    if expr2.startswith("("):
        base_expr = extract_expression(expr2)
        wrap_left, wrap_right = "(", ")"
    else:
        assert expr2.startswith("!(")
        base_expr = extract_expression_from_negation(expr2)
        wrap_left, wrap_right = "!(", ")"

    if indices:
        # Retrieve mappings from the global _ALL_MAPPINGS_ANA using a local lookup.
        mappings = _ALL_MAPPINGS_ANA.get((len(indices), len(eq_list)), [])

        for mapping in mappings:
            set_equalities = set()

            new_levels = set(levels)
            # Create a shallow copy of the argument list.
            temp_list = args_expr[:]
            # Substitute the arguments at the found indices according to the mapping.
            for i, index in enumerate(indices):
                fs = frozenset({temp_list[index], eq_list[mapping[i]]})
                if len(fs) > 1:
                    new_levels.update(clss.equality_levels_map[fs])

                    if track_history:
                        set_equalities.add("(=[" + temp_list[index] + "," + eq_list[mapping[i]] + "])")

                temp_list[index] = eq_list[mapping[i]]
            # Build the new expression using precomputed base and wrapping.
            new_expr = f"{wrap_left}{base_expr}[{','.join(temp_list)}]{wrap_right}"
            expr_levels_map[new_expr] = new_levels

            if track_history:
                if new_expr not in memory_block.expr_origin_map:
                    expr_origin_map[new_expr] = sorted(list(set_equalities))

    # Only process new expressions not already in memory_block.
    new_exprs = set(expr_levels_map.keys()) - set(memory_block.statement_levels_map.keys())
    for applied in new_exprs:


        mn = extract_max_iteration_number(applied)
        if mn is not None and mn > max_iteration_number_variable:
            continue
        if count_pattern_occurrences(applied, memory_block.local_memory) > max_number_secondary_variables:
            continue
        if not applied in memory_block.statement_levels_map and applied not in memory_block.expr_origin_map  and max(expr_levels_map[applied]) == memory_block.level:
            memory_block.statement_levels_map[applied] = expr_levels_map[applied]
            memory_block.statements.append(applied)
            #memory_block.encoded_statements.append(EncodedExpression(applied))
            memory_block.local_statements.append(applied)
            #memory_block.local_encoded_statements.append(EncodedExpression(applied))
            memory_block.local_statements_delta.append(applied)
            new_statements.append(applied)

            memory_block.mail_out.statements.add((applied, frozenset(expr_levels_map[applied])))



            if track_history:
                if applied not in memory_block.expr_origin_map:
                    origin = ["equality1"] + [expr2] + expr_origin_map[applied]
                    memory_block.expr_origin_map[applied] = origin
                    memory_block.mail_out.expr_origin_map[applied] = origin






def merge_two_equivalence_classes(class_a: EquivalenceClass,
                                  class_b: EquivalenceClass,
                                  eq_args: set[str],
                                  levels: set[int],
                                  memory_block: BodyOfProves):



    if class_a.variables.issubset(class_b.variables):
        class_a.variables = class_b.variables | class_a.variables
        class_a.equality_levels_map = class_b.equality_levels_map | class_a.equality_levels_map
        class_a.equality_origin_map = class_b.equality_origin_map | class_a.equality_origin_map
        return



    assert not eq_args.issubset(class_b.variables)

    common_args = (eq_args & class_a.variables) & class_b.variables
    assert len(common_args) == 1
    common_arg = next(iter(common_args))

    merged_map = {}
    merged_origin_map = {}
    for var_a in class_a.variables:
        if var_a == common_arg:
            continue
        for var_b in class_b.variables:
            if var_b == common_arg:
                continue

            new_levels = copy.copy(levels)
            fs_aa = frozenset({common_arg, var_a})
            fs_bb = frozenset({common_arg, var_b})

            if fs_aa in class_a.equality_levels_map:
                new_levels.update(class_a.equality_levels_map[frozenset({common_arg, var_a})])

            if fs_bb in class_b.equality_levels_map:
                new_levels.update(class_b.equality_levels_map[frozenset({common_arg, var_b})])

            merged_map[frozenset({var_a, var_b})] = new_levels

            if track_history:
                if var_a != common_arg and var_b != common_arg:
                    # "(=[a,d])" := ["equality2", "(=[b,c])", "(=[a,b])", "(=[c,d])"]

                    eq1 = "(=[" + var_a + "," + var_b + "])"
                    or1 = ["equality2",
                             "(=[" + var_a + "," + common_arg + "])",
                             "(=[" + common_arg + "," + var_b + "])"]
                    eq2 = "(=[" + var_b + "," + var_a + "])"
                    or2 = ["equality2",
                             "(=[" + var_b + "," + common_arg + "])",
                             "(=[" + common_arg + "," + var_a + "])"]

                    if eq1 not in memory_block.expr_origin_map:
                        merged_origin_map[eq1] = or1
                        memory_block.expr_origin_map[eq1] = or1
                        memory_block.mail_out.expr_origin_map[eq1] = or1

                        merged_origin_map[eq2] = or2
                        memory_block.expr_origin_map[eq2] = or2
                        memory_block.mail_out.expr_origin_map[eq2] = or2



    class_a.variables = class_b.variables | class_a.variables

    class_a.equality_levels_map = class_b.equality_levels_map | class_a.equality_levels_map
    class_a.equality_levels_map = merged_map | class_a.equality_levels_map

    class_a.equality_origin_map = class_b.equality_origin_map | class_a.equality_origin_map
    class_a.equality_origin_map = merged_origin_map | class_a.equality_origin_map

def clean_up_expressions(mb: BodyOfProves, new_statements: list[str]):
    # Rebuild local_statements and statements by filtering out those that fail the filter.

    new_local = []
    new_local_encoded = []
    for eq_clss in mb.equivalence_classes:
        for index in range(len(mb.local_statements)):
            if filter_iterations(mb.local_statements[index], eq_clss):
                new_local.append(mb.local_statements[index])

                if index >= len(mb.local_encoded_statements):
                    test = 0

                #new_local_encoded.append(mb.local_encoded_statements[index])

        new_local_delta = [s for s in mb.local_statements_delta if filter_iterations(s, eq_clss)]
        for s in set(mb.local_statements) - set(new_local):
            mb.statement_levels_map.pop(s, None)
        mb.local_statements = new_local
        mb.local_statements_delta = new_local_delta
        mb.local_encoded_statements = new_local_encoded

        new_stmts = []
        new_encoded_statements = []
        for index in range(len(mb.statements)):
            if filter_iterations(mb.statements[index], eq_clss):
                new_stmts.append(mb.statements[index])
                #new_encoded_statements.append(mb.encoded_statements[index])

        for s in set(mb.statements) - set(new_stmts):
            mb.statement_levels_map.pop(s, None)
        mb.statements = new_stmts
        mb.encoded_statements = new_encoded_statements

        new_statements = [s for s in new_statements if filter_iterations(s, eq_clss)]


    return new_statements





def update_equivalence_classes(mb: BodyOfProves,
                               eqlty: str,
                               levels: set[int],
                               origin: list[str],
                               new_statements: list[str]):
    """
    Updates the equivalence classes in the memory block based on an equality expression.
    Merges overlapping classes and then filters out any statements that no longer satisfy the
    equivalence class conditions.
    """
    args_list = get_args(eqlty)
    eq_args = set(args_list)
    mirrored = "(=[" + args_list[1] + "," + args_list[0] + "])"
    merged_class = EquivalenceClass()
    merged_class.variables.update(eq_args)
    merged_class.equality_levels_map[frozenset(eq_args)] = copy.copy(levels)
    merged_class.equality_origin_map[eqlty] = origin
    mirrored_origin = ["symmetry of equality", eqlty]
    merged_class.equality_origin_map[mirrored] = mirrored_origin


    if track_history:


        if eqlty not in mb.expr_origin_map:
            mb.expr_origin_map[eqlty] = origin
            mb.mail_out.expr_origin_map[eqlty] = origin

        if mirrored not in mb.expr_origin_map:
            mb.expr_origin_map[mirrored] = mirrored_origin
            mb.mail_out.expr_origin_map[mirrored] = mirrored_origin

    new_classes = []

    # Merge all equivalence classes that overlap with eq_args.
    for eq_clss in mb.equivalence_classes:
        if eq_args & eq_clss.variables:
            merge_two_equivalence_classes(merged_class, eq_clss, eq_args, levels, mb)
            mb.eq_class_sttmnt_index_map.pop(frozenset(eq_clss.variables), None)
        else:
            new_classes.append(eq_clss)

    mb.eq_class_sttmnt_index_map[frozenset(merged_class.variables)] = 0
    new_classes.append(merged_class)
    mb.equivalence_classes = new_classes

    for index in range(0, len(mb.statements)):
        apply_equivalence_class(merged_class,
                                mb.statements[index],
                                mb,
                                mb.statement_levels_map[mb.statements[index]], new_statements)

    mb.eq_class_sttmnt_index_map[frozenset(merged_class.variables)] = len(mb.statements)




    new_statements = clean_up_expressions(mb, new_statements)



    mb.expr_origin_map = merged_class.equality_origin_map | mb.expr_origin_map

    return new_statements


def is_equality(cand: str) -> bool:
    """Determines whether the expression is considered an equality expression."""
    return cand.startswith("(=[")



def add_statement(expr: str,
                  memory_block,
                  local: bool,
                  levels: set[int],
                  origin: list[str]) -> list[str]:
    """
    Adds an expression to the memory block if it passes various filters.
    Equivalence classes are updated as needed, and new equivalent expressions are generated.
    """
    new_statements = []



    max_iteration = extract_max_iteration_number(expr)
    if max_iteration is not None and max_iteration > max_iteration_number_variable:
        return new_statements

    if count_pattern_occurrences(expr, memory_block.local_memory) > max_number_secondary_variables:
        return new_statements

    if not is_equality(expr):
        if expr not in memory_block.statement_levels_map and expr not in memory_block.statement_levels_map:
            # Check if any equivalence class filters out this expression.
            if not any(not filter_iterations(expr, ec) for ec in memory_block.equivalence_classes):
                memory_block.statement_levels_map[expr] = copy.copy(levels)
                if track_history:
                    if expr not in memory_block.expr_origin_map:
                        memory_block.expr_origin_map[expr] = origin
                        assert expr not in memory_block.mail_out.expr_origin_map
                        memory_block.mail_out.expr_origin_map[expr] = origin
                memory_block.statements.append(expr)
                #memory_block.encoded_statements.append(EncodedExpression(expr))
                if local:


                    memory_block.local_statements.append(expr)
                    #memory_block.local_encoded_statements.append(EncodedExpression(expr))
                    memory_block.local_statements_delta.append(expr)
                    new_statements.append(expr)
                    memory_block.mail_out.statements.add((expr, frozenset(levels)))


                    if track_history:
                        if expr not in memory_block.mail_out.expr_origin_map:
                            memory_block.mail_out.expr_origin_map[expr] = origin
        # Apply each equivalence class to generate new statements.
        for eq_class in memory_block.equivalence_classes:
            apply_equivalence_class(eq_class, expr, memory_block, levels, new_statements)

    else:
        new_statements = update_equivalence_classes(memory_block, expr, levels, origin, new_statements)



    # Iteratively apply equivalence classes to any newly generated statements.
    old_size = len(memory_block.statements)
    while True:
        for equivalence_class in memory_block.equivalence_classes:
            start_index = memory_block.eq_class_sttmnt_index_map[frozenset(equivalence_class.variables)]



            for index in range(start_index, len(memory_block.statements)):
                apply_equivalence_class(equivalence_class,
                                        memory_block.statements[index],
                                        memory_block,
                                        memory_block.statement_levels_map[memory_block.statements[index]], new_statements)



            memory_block.eq_class_sttmnt_index_map[frozenset(equivalence_class.variables)] =\
                len(memory_block.statements)
        if old_size == len(memory_block.statements):
            break
        old_size = len(memory_block.statements)

    if is_equality(expr):
        new_statements = clean_up_expressions(memory_block, new_statements)

    return new_statements

def find_immutable_args(theorem: str, digit) -> set[str]:
    immutables = set()

    temp_chain = []
    disintegrate_implication(theorem, temp_chain)
    chain = []
    for element in temp_chain:
        chain.append(element[0])

    digits = find_digit_args(theorem)

    immutables.update(digits)
    immutables.remove(digit)

    found = True
    while found:
        found = False
        for expression in chain:
            core_expr = extract_expression(expression)
            if core_expr in OPERATORS:
                args = get_args(expression)

                cand = args[len(args) - 2]

                inputs = {args[ind] for ind in range(0, len(args) - 2)}

                if inputs.issubset(immutables) and cand not in immutables:
                    immutables.add(cand)
                    found = True



    return immutables


def add_theorem_to_memory(expr: str,
                       memory: BodyOfProves,
                       iteration: int,
                       proved: bool,
                       dependency_table: Dependencies):
    global _LIST_OF_MEMORY_BLOCKS
    temp_chain = []
    head = disintegrate_implication(expr, temp_chain)
    chain = []
    for element in temp_chain:
        chain.append(element[0])

    memory_block = memory
    for index, element in enumerate(chain):
        if element in memory_block.simple_map:
            memory_block = memory_block.simple_map[element]
        else:
            memory_block.simple_map[element] = BodyOfProves()
            _LIST_OF_MEMORY_BLOCKS.append(memory_block.simple_map[element])
            memory_block.simple_map[element].parent_body_of_proves = memory_block
            memory_block = memory_block.simple_map[element]
            memory_block.level = index
            memory_block.expr_key = element[:]
            add_expr_to_memory_block(element,
                                     memory_block,
                                     iteration,
                                     0,
                                     {memory_block.level},
                                     ["task formulation"])


        if index == len(chain) - 1:
            if proved:
                add_expr_to_memory_block(head, memory_block, iteration, 1, set(), [])
            else:

                digits = find_digit_args(expr)



                for digit_arg in sorted(digits):
                    immutables = find_immutable_args(expr, digit_arg)

                    rec_arg = 'rec' + str(memory_block.recursion_counter)
                    memory_block.recursion_counter += 1


                    auxy_implication, auxy_chain, auxy_head, remaining_args_key, zero_arg_name, s_name =\
                        create_auxy_implication(expr, digit_arg, 'rec', immutables)

                    temp_expr = "(in2[" + rec_arg+ "," + digit_arg + "," + s_name + "])"
                    temp_expr2 = "(in2[" + 'rec' + "," + digit_arg + "," + s_name + "])"
                    assert temp_expr not in memory_block.simple_map
                    memory_block.simple_map[temp_expr] = BodyOfProves()
                    _LIST_OF_MEMORY_BLOCKS.append(memory_block.simple_map[temp_expr])
                    temp_memory_block = memory_block.simple_map[temp_expr]
                    temp_memory_block.level = index + 1
                    temp_memory_block.parent_body_of_proves = memory_block
                    temp_memory_block.expr_key = temp_expr[:]
                    temp_memory_block.is_part_of_recursion = True
                    temp_memory_block.expr_origin_map[auxy_implication] = ["recursion"]
                    temp_memory_block.mail_out.expr_origin_map[auxy_implication] = ["recursion"]


                    add_to_hash_memory(auxy_chain,
                                       auxy_head,
                                       frozenset(remaining_args_key),
                                       temp_memory_block.local_memory,
                                       {temp_memory_block.level},
                                       auxy_implication,
                                       induction_max_admission_depth,
                                       induction_max_secondary_number,
                                       True)

                    #marked_expr = '(in3[rec,7,marker,4])'
                    #if marked_expr in temp_memory_block.local_memory.admission_map:
                        #update_admission_map2(temp_memory_block.local_memory, marked_expr, "it_1")


                    add_expr_to_memory_block(temp_expr2,
                                             temp_memory_block,
                                             iteration,
                                             0,
                                             {temp_memory_block.level},
                                             ["recursion"])

                    add_expr_to_memory_block(head,
                                             temp_memory_block,
                                             iteration,
                                             2,
                                             set(),
                                             [],
                                             dependency_table.auxy_index)




                    dependency_table.original_auxy_map[dependency_table.original_index] = DependencyItem()
                    dependency_table.original_auxy_map[dependency_table.original_index].auxies.add(
                        dependency_table.auxy_index)
                    dependency_table.original_auxy_map[dependency_table.original_index].expr = expr
                    dependency_table.auxy_original_map[dependency_table.auxy_index] = \
                        dependency_table.original_index
                    dependency_table.auxy_index += 1

                    temp_expr3 = "(=[s(" + rec_arg + ")," + zero_arg_name + "])"
                    temp_expr4 = "(=[" + digit_arg + "," + zero_arg_name + "])"
                    assert temp_expr3 not in memory_block.simple_map
                    memory_block.simple_map[temp_expr3] = BodyOfProves()
                    _LIST_OF_MEMORY_BLOCKS.append(memory_block.simple_map[temp_expr3])
                    temp_memory_block2 = memory_block.simple_map[temp_expr3]
                    temp_memory_block2.level = index + 1
                    temp_memory_block2.parent_body_of_proves = memory_block
                    temp_memory_block2.expr_key = temp_expr4[:]
                    temp_memory_block2.is_part_of_recursion = True
                    add_expr_to_memory_block(temp_expr4,
                                             temp_memory_block2,
                                             iteration,
                                             0,
                                             {temp_memory_block2.level},
                                             ["recursion"])


                    add_expr_to_memory_block(head,
                                             temp_memory_block2,
                                             iteration,
                                             2,
                                             set(),
                                             [],
                                             dependency_table.auxy_index)
                    


                    temp_memory_block2.is_active = False




                    dependency_table.auxy_original_map[dependency_table.auxy_index] = \
                        dependency_table.original_index
                    dependency_table.original_auxy_map[dependency_table.original_index].auxies.add(
                        dependency_table.auxy_index)
                    dependency_table.original_induction_variable_map[dependency_table.original_index] = digit_arg
                    dependency_table.auxy_index += 1
                    dependency_table.original_index += 1

                add_expr_to_memory_block(head, memory_block, iteration, 2, set(), [])


    return


def extract_difference(s):
    """
    Given a string s, this function extracts tokens from two sets:
      1. Tokens inside patterns >[ ... ]
      2. Tokens inside patterns [ ... ] not preceded by '>'

    It then returns a set of tokens (as strings) that appear only in the second set.
    """
    # 1. Extract tokens from patterns like >[ ... ]
    first_set = set()
    pattern1 = r'>\[([^\]]*)\]'  # This captures what's between ">[" and "]"
    for match in re.findall(pattern1, s):
        tokens = [token.strip() for token in match.split(',') if token.strip()]
        first_set.update(tokens)

    # 2. Extract tokens from patterns like [ ... ] that are NOT preceded by ">"
    second_set = set()
    pattern2 = r'(?<!>)\[(.*?)\]'  # capture inside brackets not preceded by ">"
    for match in re.findall(pattern2, s):
        tokens = [token.strip() for token in match.split(',') if token.strip()]
        second_set.update(tokens)

    # 3. Compute the difference: tokens in second_set not in first_set.
    difference = second_set - first_set

    return difference

def update_global_direct(theorem: str):
    temp_chain = []
    vlue = disintegrate_implication(theorem, temp_chain)
    ky = []
    for element in temp_chain:
        ky.append(element[0])

    global_body_of_proves.mail_out.implications.add((tuple(copy.deepcopy(ky)),
                                                     copy.deepcopy(vlue),
                                                     frozenset(),
                                                     frozenset(),
                                                     theorem))

    if track_history:
        global_body_of_proves.mail_out.expr_origin_map[theorem] = ["theorem"]

    global_theorem_list.append((theorem, "direct", "-1"))
    print(theorem)

    reshuffled_mirrored = \
        create_reshuffled_mirrored(theorem,
                                   _ALL_PERMUTATIONS_ANA,
                                   True)
    if reshuffled_mirrored:
        global_theorem_list.append((reshuffled_mirrored,
                                    "mirrored statement",
                                    theorem))
        # print(reshuffled_mirrored)
        if track_history:
            global_body_of_proves.mail_out.expr_origin_map[reshuffled_mirrored] = ["theorem"]

        temp_chain = []
        vlue = disintegrate_implication(reshuffled_mirrored, temp_chain)
        ky = []
        for element in temp_chain:
            ky.append(element[0])
        memory_block_m = access_body_of_proves(ky, global_body_of_proves)
        if memory_block_m:
            origin = ["implication", reshuffled_mirrored]
            origin.extend(ky)
            add_statement(vlue, memory_block_m, False, set(range(len(ky) + 1)), origin)

        global_body_of_proves.mail_out.implications.add((tuple(copy.deepcopy(ky)),
                                                         copy.deepcopy(vlue),
                                                         frozenset(),
                                                         frozenset(),
                                                         reshuffled_mirrored))

    send_mail(global_body_of_proves, global_body_of_proves.mail_out)
    global_body_of_proves.mail_out.statements.clear()
    global_body_of_proves.mail_out.implications.clear()
    global_body_of_proves.expr_origin_map.clear()

def update_global(auxy_index, all_levels_involved: bool):
    global global_dependencies
    global global_body_of_proves

    original_index = global_dependencies.auxy_original_map[auxy_index]
    global_dependencies.original_auxy_map[original_index].auxies.remove(auxy_index)
    global_dependencies.original_auxy_map[original_index].all_levels_involved =\
        all_levels_involved | global_dependencies.original_auxy_map[original_index].all_levels_involved



    if (len(global_dependencies.original_auxy_map[original_index].auxies) == 0 and
            global_dependencies.original_auxy_map[original_index].all_levels_involved):
        temp_chain = []
        vlue = disintegrate_implication(global_dependencies.original_auxy_map[original_index].expr, temp_chain)
        ky = []
        for element in temp_chain:
            ky.append(element[0])
        memory_block = access_body_of_proves(ky, global_body_of_proves)


        if vlue in memory_block.to_be_proved:
            assert len(memory_block.to_be_proved[vlue]) == 0
            memory_block.to_be_proved.pop(vlue)

            origin = ["implication", global_dependencies.original_auxy_map[original_index].expr]
            origin.extend(ky)
            add_statement(vlue, memory_block, False, set(range(len(ky) + 1)), origin)

            global_body_of_proves.mail_out.implications.add((tuple(copy.deepcopy(ky)),
                                                         copy.deepcopy(vlue),
                                                         frozenset(),
                                                         frozenset(),
                                                         global_dependencies.original_auxy_map[original_index].expr))

            if track_history:
                global_body_of_proves.mail_out.expr_origin_map[global_dependencies.original_auxy_map[original_index].expr] =\
                    ["theorem"]

            ind_var = global_dependencies.original_induction_variable_map[original_index]
            global_theorem_list.append((global_dependencies.original_auxy_map[original_index].expr, "induction", ind_var))

            print(global_dependencies.original_auxy_map[original_index].expr)


            reshuffled_mirrored =\
                create_reshuffled_mirrored(global_dependencies.original_auxy_map[original_index].expr,
                                               _ALL_PERMUTATIONS_ANA,
                                               True)
            if reshuffled_mirrored:
                global_theorem_list.append((reshuffled_mirrored,
                                            "mirrored statement",
                                            global_dependencies.original_auxy_map[original_index].expr))
                #print(reshuffled_mirrored)
                if track_history:
                    global_body_of_proves.mail_out.expr_origin_map[reshuffled_mirrored] =  ["theorem"]

                temp_chain_m = []
                vlue_m = disintegrate_implication(reshuffled_mirrored, temp_chain_m)
                ky_m = []
                for element in temp_chain_m:
                    ky_m.append(element[0])
                memory_block_m = access_body_of_proves(ky_m, global_body_of_proves)
                if memory_block_m:
                    origin = ["implication", reshuffled_mirrored]
                    origin.extend(ky_m)
                    add_statement(vlue_m, memory_block_m, False, set(range(len(ky_m) + 1)), origin)


                global_body_of_proves.mail_out.implications.add((tuple(copy.deepcopy(ky_m)),
                                                                 copy.deepcopy(vlue_m),
                                                                 frozenset(),
                                                                 frozenset(),
                                                                 reshuffled_mirrored))



            send_mail(global_body_of_proves, global_body_of_proves.mail_out)
            global_body_of_proves.mail_out.statements.clear()
            global_body_of_proves.mail_out.implications.clear()
            global_body_of_proves.expr_origin_map.clear()

def create_variants(chain: list[str], head: str):
    variants = set()

    permuts = _ALL_PERMUTATIONS_ANA[len(chain)]
    ids = []
    for expression in chain:
        ids.append(extract_expression(expression))


    for permutation in permuts:
        temp_list = []
        for index in permutation:
            temp_list.append(chain[index])

        to_continue = False
        for index in range(len(chain) - 1):
            if ids[permutation[index]] > ids[permutation[index + 1]]:
                to_continue = True
                break
        if to_continue:
            continue

        key_variant = "".join(temp_list)
        key_variant, mp = normalize_variables(key_variant)
        value_variant = replace_keys_in_string(head, mp)
        variants.add((key_variant, value_variant, frozenset(mp.items())))

    return variants

_is_proved_pattern = r'it_\d+_lev_\d+_'
_is_proved_pattern2 = r'c\d+'
def is_proved(s):
    """
    Checks if the input string contains a substring matching the pattern:
    "it_<int1>_lev_<int2>_"

    Parameters:
        s (str): The string to check.

    Returns:
        bool: True if the pattern is found, False otherwise.
    """

    match1 = re.search(_is_proved_pattern, s)
    match2 = re.search(_is_proved_pattern2, s)
    return not bool(match1) and not bool(match2)

def find_zero_arg_name(memory_block: BodyOfProves):
    current_memory_block = memory_block
    while True:
        if current_memory_block.expr_key.startswith("(NaturalNumbers["):
            zero_arg = get_args(current_memory_block.expr_key)[1]
            break
        elif current_memory_block.parent_body_of_proves is not None:
            current_memory_block = current_memory_block.parent_body_of_proves
        else:
            assert False

    return zero_arg

def update_admission_map3(expr: str,
                          memory_block: BodyOfProves,
                          max_admission_depth,
                          max_secondary_number,
                          part_of_recursion):
    digit_args = find_digit_args(expr)
    temp_mb = memory_block
    while True:
        expr_key_args = set(get_args(temp_mb.expr_key))
        if digit_args & expr_key_args:
            args = get_args(expr)
            update_admission_map(temp_mb.local_memory,
                                 [expr],
                                 frozenset(digit_args).union([args[len(args) - 1]]),
                                 max_admission_depth,
                                 max_secondary_number,
                                 part_of_recursion)
            break
        else:
            if temp_mb.parent_body_of_proves:
                temp_mb = temp_mb.parent_body_of_proves
            else:
                break


def extract_items(groups: list[str]) -> Set[str]:
    """Flatten a list of comma-separated strings into a set of stripped items."""
    items = set()
    for grp in groups:
        for item in grp.split(','):
            item = item.strip()
            if item:
                items.add(item)
    return items


def check_theorem(s: str, ancr: str) -> bool:
    """
    1. Extract all substrings inside >[ ... ] → first set.
    2. Extract all substrings inside [ ... ] not preceded by '>' → second set.
    3. Extract all substrings inside ancr[ ... ] → third set.
    4. Remove third set from second set, then check that remainder is a subset of first set.
    Returns True if (second \ third) ⊆ first, else False.
    """
    # 1) >[ ... ]
    first_groups = re.findall(r'>\[(.*?)]', s)

    # 2) [ ... ] not preceded by '>'
    second_groups = re.findall(r'(?<!>)\[(.*?)]', s)

    # 3) ancr[ ... ]
    #    (this will also be found in second_groups, but we separate it explicitly)
    third_groups = re.findall(rf'{re.escape(ancr)}\[(.*?)]', s)

    set1 = extract_items(first_groups)
    set2 = extract_items(second_groups)
    set3 = extract_items(third_groups)

    remainder = set2 - set3

    return remainder.issubset(set1) and remainder

def add_expr_to_memory_block(expr: str,
                             memory_block: BodyOfProves,
                             iteration: int,
                             status: int,
                             involved_levels: set[int],
                             origin: list[str],
                             auxy_index = -1):
    """
        status (int): An integer representing the statement's status.
                      Valid values are:
                        0 - Local statement.
                        1 - Proved statement.
                        2 - Statement pending proof.
                        3 - Non-local statement.
    """

    if expr in memory_block.whole_expressions:
        return




    if anchor[3] in expr and status == 0 and False:
        anchor_args = get_args(expr)
        copy_counter = 0
        replacement_map = {}
        for anchor_arg in anchor_args:
            if core_expression_map[anchor[3]][1][anchor_arg] == "(1)":
                replacement_map[anchor_arg] = "c" + str(copy_counter)
                copy_counter += 1

        replaced_anchor = replace_keys_in_string(expr, replacement_map)

        origin = ["copy", expr]
        add_statement(replaced_anchor, memory_block, True, {memory_block.level}, origin)
        memory_block.mail_out.statements.add((replaced_anchor,frozenset({memory_block.level})))
        memory_block.mail_out.expr_origin_map[replaced_anchor] = origin









    if status != 2:
        if is_equality(expr):
            args = get_args(expr)
            mirrored = "(=[" + args[1] + "," + args[0] + "])"
            memory_block.whole_expressions.add(mirrored)

            if status == 1:
                memory_block.mail_out.statements.add((expr, frozenset(involved_levels)))
                memory_block.mail_out.statements.add((mirrored, frozenset(involved_levels)))

                if track_history:
                    if expr not in memory_block.mail_out.expr_origin_map:
                        memory_block.mail_out.expr_origin_map[expr] = origin

                    if mirrored not in memory_block.mail_out.expr_origin_map:

                        mirrored_origin = ["symmetry of equality", expr]
                        memory_block.mail_out.expr_origin_map[mirrored] = mirrored_origin


        memory_block.whole_expressions.add(expr)



    if status == 2:
        if expr in memory_block.to_be_proved:
            if auxy_index >= 0:
                memory_block.to_be_proved[expr].add(auxy_index)
        else:
            if auxy_index >= 0:
                memory_block.to_be_proved[expr] = {auxy_index}
            else:
                memory_block.to_be_proved[expr] = set()

        core_expr = extract_expression(expr)
        if core_expr in OPERATORS:
            update_admission_map3(expr,
                                  memory_block,
                                  induction_max_admission_depth,
                                  induction_max_secondary_number,
                                  True)


    else:
        if track_history:
            if expr not in memory_block.expr_origin_map:
                memory_block.expr_origin_map[expr] = origin
                memory_block.mail_out.expr_origin_map[expr] = origin




        is_simple = expression_is_simple(expr)
        new_statements = []
        if is_simple:
            is_local = (status == 0 or status == 1)


            new_statements = add_statement(expr, memory_block, is_local, involved_levels, origin)


        for index in range(len(sorted(new_statements))):
            add_expression = new_statements[index]



            add_expression_levels = memory_block.statement_levels_map[add_expression]
            all_levels_involved = len(add_expression_levels) == (memory_block.level + 1)

            if memory_block.is_part_of_recursion:
                if add_expression in memory_block.to_be_proved:
                    for auxy_index in memory_block.to_be_proved[add_expression]:

                        update_global(auxy_index, all_levels_involved)

                    memory_block.to_be_proved.pop(add_expression)

                    memory_block.is_active = False

                    if memory_block.expr_key.startswith("(in2[rec"):
                        args = get_args(memory_block.expr_key)
                        zero_arg_name = find_zero_arg_name(memory_block)
                        eq_expr = "(=[s(" + args[0] + ")," + zero_arg_name + "])"
                        eq_mb = memory_block.parent_body_of_proves.simple_map[eq_expr]
                        eq_mb.is_active = True





            if (not memory_block.is_part_of_recursion and
                    is_proved(new_statements[index]) and all_levels_involved and status != 0):
                global_key = get_global_key(memory_block)
                full_theorem = reconstruct_implication(global_key, add_expression)

                #if check_theorem(full_theorem, anchor[3]):
                if add_expression in memory_block.to_be_proved:


                    update_global_direct(full_theorem)



                    memory_block.to_be_proved.pop(add_expression)


        expanded_expr = expand_expr(expr)
        if track_history:
            if expanded_expr not in memory_block.expr_origin_map:
                memory_block.expr_origin_map[expanded_expr] = ["expansion", expr]
                memory_block.mail_out.expr_origin_map[expanded_expr] = ["expansion", expr]



        if expanded_expr == expr and is_simple:
            return

        if expanded_expr not in memory_block.statement_levels_map:
            memory_block.statement_levels_map[expanded_expr] = involved_levels

        if status != 3:
            implications, statements, memory_block.start_int = disintegrate_expr(expanded_expr,
                                                                                memory_block.start_int,
                                                                                iteration,
                                                                                memory_block.level,
                                                                                 memory_block.local_memory)
            local_origin = None
            if track_history:
                local_origin = ["disintegration", expanded_expr]

            for implication in implications:
                ky, vlue = extract_key_value(implication)
                remaining_args_key = extract_difference(ky)

                remaining_args_impl = extract_difference(implication)
                replacement_map = {remaining_arg: "u_" + remaining_arg for remaining_arg in remaining_args_impl}
                replaced_impl = replace_keys_in_string(implication, replacement_map)

                temp_chain = []
                head = disintegrate_implication(replaced_impl, temp_chain)
                chain = []
                for element in temp_chain:
                    chain.append(element[0])

                add_to_hash_memory(chain,
                                   head,
                                   frozenset(remaining_args_key),
                                   memory_block.local_memory,
                                   involved_levels,
                                   implication,
                                   standard_max_admission_depth,
                                   standard_max_secondary_number,
                                   False)
                memory_block.mail_out.implications.add((tuple(copy.deepcopy(chain)),
                                                        copy.deepcopy(head),
                                                        frozenset(copy.deepcopy(remaining_args_key)),
                                                        frozenset(involved_levels),
                                                        implication))


                if track_history:
                    if implication not in memory_block.expr_origin_map:
                        memory_block.expr_origin_map[implication] = local_origin
                        memory_block.mail_out.expr_origin_map[implication] = local_origin

            for statement in statements:

                add_expr_to_memory_block(statement,
                                         memory_block,
                                         iteration,
                                         0,
                                         involved_levels,
                                         local_origin)



    return








def create_auxy_implication(expr: str, arg, rec_arg, digit_args: set[str]):
    def extract_substrings(expr2: str) -> list:
        """
        Extracts substrings of the form "something[...]" from the input string expr2,
        where "something" is the text immediately following a "(" (and not including it)
        up to the literal "[". Substrings where "something" is ">" are ignored.

        The substrings are returned in the order of their occurrence and now include the surrounding "()".

        For example:
          Input:
            expr2 = "(>[1,2,3,4](in3[1,2,3,4])(>[5,6](in3[5,2,6,4])" \
                    "(>[7](NaturalNumbers[9,7,10,11,4])(>[8](in3[1,2,8,4])" \
                    "(>[](in3[3,2,5,4])(in3[6,8,7,4]))))))"

          Output:
            [
              "(in3[1,2,3,4])",
              "(in3[5,2,6,4])",
              "(NaturalNumbers[9,7,10,11,4])",
              "(in3[1,2,8,4])",
              "(in3[6,8,7,4])"
            ]
        """
        # The regex breakdown:
        #   \(            : Match a literal "(".
        #   (?!>)         : Ensure that the character following "(" is not ">".
        #   [^(\[]+       : Match one or more characters that are neither "(" nor "[".
        #   \[            : Match a literal "[".
        #   [^\]]*        : Match zero or more characters that are not "]".
        #   \]            : Match a literal "]".
        #   \)            : Match a literal ")".
        pattern = r"(\((?!>)[^(\[]+\[[^\]]*\]\))"
        matches = re.findall(pattern, expr2)
        return matches




    untouchables = {rec_arg}.union(digit_args)
    chain = []
    zero_arg_name = ""
    s_name = ""

    repl_map = {arg: rec_arg}
    repl_expr = replace_keys_in_string(expr, repl_map)

    subexprs = extract_substrings(repl_expr)
    for subexpr in subexprs:
        if anchor[3] in subexpr:
            anchor_args = get_args(subexpr)
            untouchables.update(anchor_args)
            zero_arg_name = anchor_args[1]
            s_name = anchor_args[3]
        else:
            chain.append(subexpr)

    remaining_args_key = set()
    for element_index in range(len(chain) - 1):
        element = chain[element_index]
        element_args = get_args(element)
        for elemnt_arg in element_args:
            if elemnt_arg in untouchables:
                remaining_args_key.add(elemnt_arg)

    repl_map2 = {}
    for untouchable in untouchables:
        repl_map2[untouchable] = "u_" + untouchable
    for index in range(len(chain)):
        chain[index] = replace_keys_in_string(chain[index], repl_map2)

    assert len(chain) > 0
    head = chain.pop()

    implication = reconstruct_implication(chain, head)


    return implication, chain, head, remaining_args_key, zero_arg_name, s_name

def build_stack(memory_block: BodyOfProves,
                proved: str,
                stack: list[tuple[str,list[str]]],
                covered: set[str]):

    origin = memory_block.expr_origin_map[proved]

    stack.append([proved] + origin)




    for ingredient in origin[1:]:
        if ingredient not in covered:
            covered.add(ingredient)

            build_stack(memory_block, ingredient, stack, covered)

    return

def sort_by_values_desc(strings: List[str], values: List[int]) -> List[str]:
    """
    Return a new list of strings, ordered by descending integer values.
    Raises ValueError if the two lists differ in length.
    """
    if len(strings) != len(values):
        raise ValueError(f"got {len(strings)} strings but {len(values)} values")
    # Pair each integer with its string, sort by integer descending, then extract strings
    paired: List[Tuple[int, str]] = sorted(zip(values, strings), reverse=True)
    return [s for _, s in paired]

def find_ends(path: list[str]):
    global global_theorem_list
    global global_body_of_proves

    memory_block = global_body_of_proves
    for elt in path:
        memory_block = memory_block.simple_map[elt]



    all_exprs = set(memory_block.expr_origin_map)


    ends = all_exprs

    stack_sizes = []
    for end in ends:
        stack = []
        build_stack(memory_block, end, stack, set())
        stack_sizes.append(len(stack))

    ends = sort_by_values_desc(list(ends), stack_sizes)


    global_theorem_list = []
    for end in ends:
        global_theorem_list.append((";".join(path) + "+" + end, "debug", "-1"))

    return


def analyze_expressions(theorems):
    global global_body_of_proves
    global max_num_leafs_per_key
    global global_dependencies
    global _ALL_PERMUTATIONS_ANA
    global _ALL_BINARIES_ANA
    global _ALL_MAPPINGS_ANA
    global _LIST_OF_MEMORY_BLOCKS

    binary_seqs_map = {}
    for num in range(0, size_all_binaries_ana + 1):
        binary_seqs_map[num] = generate_binary_sequences_as_lists(num)
    _ALL_BINARIES_ANA = binary_seqs_map
    _ALL_MAPPINGS_ANA = generate_all_mappings(max_size_def_set_mapping, max_size_target_set_mapping)


    max_num_leafs_per_key = 0

    _ALL_PERMUTATIONS_ANA = generate_all_permutations(size_all_permutations_ana)
    dependency_table = Dependencies()
    global_dependencies = dependency_table



    _LIST_OF_MEMORY_BLOCKS.append(global_body_of_proves)
    for theorem in theorems:
        add_theorem_to_memory(theorem, global_body_of_proves, 0, False, dependency_table)

    print("Deep size:", asizeof.asizeof(global_body_of_proves), "bytes")


    start_time2 = time.time()
    prove()
    end_time2 = time.time()
    print("Deep size:", asizeof.asizeof(global_body_of_proves), "bytes")

    print(f"Runtime prove(): {end_time2 - start_time2:.5f} seconds")



    temp_mem = global_body_of_proves

    #mb = temp_mem.simple_map["(NaturalNumbers[1,2,3,4,5])"].simple_map["(in3[2,6,7,4])"].simple_map["(in2[rec,6,3])"]
    #stck = []
    #covered = set()
    #build_stack(mb, "(in3[6,2,7,4])", stck, covered)

    return