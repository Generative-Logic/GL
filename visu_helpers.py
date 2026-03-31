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

import expression_utils
from expression_utils import disintegrate_implication
import regex
import re
import copy

test_expr = '(>[5,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[7,10,11,6])(>[12](in3[8,10,12,6])(>[13](in3[9,10,13,6])(in3[11,12,13,5]))))))'

def get_output_arg(expr: str) -> str:
    """Helper to extract the output variable based on expression type."""
    args = expression_utils.get_args(expr)
    if not args:
        return ''
    if expr.startswith('(fold['):
        return args[-1]
    if expr.startswith('(in[') or expr.startswith('(in2[') or expr.startswith('(in3['):
        return args[-2]
    return ''

def fully_resolve_markers(base_str: str, chain_subset: list[str]) -> str:
    """Recursively replaces nested marker variables to build a single mathematical equation."""
    res = base_str
    for el in reversed(chain_subset):
        if el.startswith('(in') or el.startswith('(fold['):
            out = get_output_arg(el)
            if out and ('marker' + out) in res:
                res = res.replace('marker' + out, rewrite_expression(el))
    return res.replace('marker', '')

def rewrite_expression(expression: str):
    rewritten = ''

    assert (expression.startswith('(in') or
            expression.startswith('(in2') or
            expression.startswith('(in3') or
            expression.startswith('(fold['))

    args = expression_utils.get_args(expression)

    if expression.startswith('(in['):
        rewritten = 'marker' + args[-2]

    if expression.startswith('(in2'):
        rewritten = args[-1] + '(' + 'marker' + args[-3] + ')'

    if expression.startswith('(in3'):
        rewritten = '(marker' + args[-4] + ' ' + args[-1] + ' ' + 'marker' + args[-3] + ')'

    if expression.startswith('(fold['):
        # fold[N,s,+,f,n,m,p]  ->  sum(i=n..m) if f is id
        # Drop the `= p_name` assignment so the sum can dynamically embed into other operators
        _, _, _, f_name, n_name, m_name, p_name = args
        term = '' if f_name == 'id' else f' {f_name}(i)'
        # Removed the outer '(' and ')' here:
        rewritten = 'sum(i=' + 'marker' + n_name + '..' + 'marker' + m_name + ')' + term

    return rewritten


def rewrite_expression2(expression: str):
    rewritten = ''

    assert (expression.startswith('(in[') or
            expression.startswith('(in2') or
            expression.startswith('(in3') or
            expression.startswith('(=[') or
            expression.startswith('(fold['))

    args = expression_utils.get_args(expression)

    if expression.startswith('(in['):
        rewritten = 'marker' + args[-2] + ' \u2208 ' + args[-1]

    if expression.startswith('(in2['):
        rewritten = args[-1] + '(' + 'marker' + args[-3] + ') = ' + 'marker' + args[-2]

    if expression.startswith('(in3['):
        rewritten = 'marker' + args[-4] + ' ' + args[-1] + ' ' + 'marker' + args[-3] + ' = ' + args[2]

    if expression.startswith('(=['):
        rewritten = 'marker' + args[-2] + ' = ' + 'marker' + args[-1]

    if expression.startswith('(fold['):
        # fold[N,s,+,f,n,m,p]  ->  sum(i=n..m)=p if f is id
        _, _, _, f_name, n_name, m_name, p_name = args
        term = '' if f_name == 'id' else f' {f_name}(i)'
        rewritten = 'sum(i=' + 'marker' + n_name + '..' + 'marker' + m_name + ')' + term + ' = ' + 'marker' + p_name

    return rewritten

def make_readable_simple_implication_title(chain: list[str]):
    readable = ''
    head = chain[-1]
    head_output = get_output_arg(head)

    for index, element in enumerate(chain[:-1]):
        output = get_output_arg(element)
        if output == head_output and output != '':
            readable = rewrite_expression(element) + '=' + head_output + '=' + rewrite_expression(head)
            break

    if not readable:
        readable = rewrite_expression2(head)

    readable = fully_resolve_markers(readable, chain[:-1])
    return readable


def make_readable_equality(chain: list[str]):
    eq_args = expression_utils.get_args(chain[-1])
    left_output = eq_args[0]
    right_output = eq_args[1]

    left_operator = ''
    right_operator = ''
    for element in chain[:-1]:
        if element.startswith('(in2') or element.startswith('(in3') or element.startswith('(fold['):
            output = get_output_arg(element)
            if output == left_output:
                left_operator = element
            if output == right_output:
                right_operator = element

    assert left_operator and right_operator

    # Integrate the equality mathematically
    integrated_equality = fully_resolve_markers(rewrite_expression2(chain[-1]), chain[:-1])

    if left_operator != right_operator:
        left_idx = chain.index(left_operator)
        right_idx = chain.index(right_operator)

        resolved_left = fully_resolve_markers(rewrite_expression2(left_operator), chain[:left_idx])
        resolved_right = fully_resolve_markers(rewrite_expression2(right_operator), chain[:right_idx])

        readable = 'from ' + resolved_left + ' and ' + resolved_right + ' follows ' + integrated_equality
    else:
        non_empty = left_operator if left_operator else right_operator
        non_empty_idx = chain.index(non_empty)
        resolved_non_empty = fully_resolve_markers(rewrite_expression2(non_empty), chain[:non_empty_idx])

        readable = 'from ' + resolved_non_empty + ' follows ' + integrated_equality

    return readable



def list_last_removed_args(expr: str):
    pattern = r">\[[^\[\]]*\]"
    match = re.search(pattern, expr)
    if match:
        # Extract the content between ">[" and "]"
        return match.group()[2:-1].split(',')
    return []

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

def is_relation_expression(expr: str) -> bool:
    return expr.startswith('(in[') or expr.startswith('(in2[') or expr.startswith('(in3[')


def is_rewritable_atom(expr: str) -> bool:
    e = expr[1:] if expr.startswith('!') else expr
    return (
        e.startswith('(in[')
        or e.startswith('(in2[')
        or e.startswith('(in3[')
        or e.startswith('(=[')
        or e.startswith('(fold[')
    )


def safe_rewrite_atom(expr: str) -> str:
    try:
        if is_rewritable_atom(expr):
            if expr.startswith('!'):
                inner = rewrite_expression2(expr[1:]).replace('marker', '')
                return inner.replace(' = ', ' \u2260 ').replace(' \u2208 ', ' \u2209 ')
            return rewrite_expression2(expr).replace('marker', '')
    except Exception:
        pass
    return expr


def make_readable_generic_chain(chain: list[str]) -> str:
    if not chain:
        return ''

    rendered = [safe_rewrite_atom(expr) for expr in chain]
    if len(rendered) == 1:
        return rendered[0]

    return 'from ' + ', '.join(rendered[:-1]) + ' follows ' + rendered[-1]


def make_readable_element(chain: list[str]):
    head = chain[-1]
    readable = ''

    assert len(chain) == 2

    element_arg = expression_utils.get_args(head)[-2]

    assert element_arg in expression_utils.get_args(chain[0])

    readable = 'from ' + rewrite_expression2(chain[0]) +  ' follows ' + rewrite_expression2(chain[-1])

    readable = readable.replace('marker', '')

    return readable


def make_readable_existence(chain: list[str]):
    head = chain[-1]

    try:
        left_expr, right_expr = extract_values_regex(head)
    except ValueError:
        return safe_rewrite_atom(head)

    left_text = safe_rewrite_atom(left_expr)

    # If the second expression has a '!', strip it. Otherwise, prepend 'not '
    if right_expr.startswith('!'):
        right_text = safe_rewrite_atom(right_expr[1:])
    else:
        right_text = 'not ' + safe_rewrite_atom(right_expr)

    # Build the final existence string
    existence_str = f"exists {left_text} with {right_text}"
    existence_str = existence_str.replace('marker', '')

    # If it's part of a chain with premises, prepend the "from ... follows"
    if len(chain) > 1:
        premises = [safe_rewrite_atom(expr) for expr in chain[:-1]]
        return 'from ' + ', '.join(premises) + ' follows ' + existence_str

    return existence_str

def make_readable_from_chain(chain: list[str]):
    if not chain:
        return ''

    head = chain[-1]
    readable = ''

    try:
        # Now natively targets fold directly as a viable sequence
        if head.startswith('(in2') or head.startswith('(in3') or head.startswith('(fold['):
            readable = make_readable_simple_implication(chain)
        elif head.startswith('(=['):
            readable = make_readable_equality(chain)
        elif head.startswith('(in['):
            readable = make_readable_element(chain)
        elif head.startswith('!(>['):
            readable = make_readable_existence(chain)
        elif head.startswith('!'):
            readable = safe_rewrite_atom(head)
    except Exception:
        readable = ''

    if not readable:
        readable = make_readable_generic_chain(chain)

    return readable

def make_readable_from_chain_title(chain: list[str]):
    if not chain:
        return ''

    head = chain[-1]
    readable = ''

    try:
        if head.startswith('(in2') or head.startswith('(in3'):
            readable = make_readable_simple_implication_title(chain)
        elif head.startswith('(fold['):
            readable = safe_rewrite_atom(head)
        elif head.startswith('!(>['):
            readable = make_readable_existence(chain)
        elif head.startswith('!'):
            readable = safe_rewrite_atom(head)
    except Exception:
        readable = ''

    if not readable:
        readable = make_readable_generic_chain(chain)

    return readable


def make_readable_simple_implication(chain: list[str]):
    readable = ''
    args_list = []

    for element in chain:
        assert element.startswith('(in')
        args_list.append(expression_utils.get_args(element))

    head_output = args_list[-1][-2]

    left_head = ''
    left_head_index = -1
    for index, args in enumerate(args_list):
        assert args != args_list[-1]
        output = args[-2]
        if output == head_output:
            # FIXED: Removed the '+ '=' + head_output' to prevent internal variable leaks like '=v8='
            readable = rewrite_expression(chain[index]) + '=' + rewrite_expression(chain[-1])
            left_head = chain[index]
            left_head_index = index
            break

    for_list = []
    head = chain[-1]
    head_args = expression_utils.get_args(head)
    left_head_args = expression_utils.get_args(left_head)
    input_args = []

    for arg_index in range(0, len(left_head_args) - 2):
        input_args.append(left_head_args[arg_index])
    for arg_index in range(0, len(head_args) - 2):
        input_args.append(head_args[arg_index])

    for input_arg in input_args:
        for index, args in enumerate(args_list):
            if index == left_head_index or index == len(chain) - 1:
                continue

            output_arg = args[-2]
            if output_arg == input_arg:
                for_list.append(chain[index])

    for_list2 = [rewrite_expression2(expression) for expression in for_list]

    if for_list2:
        readable = 'from ' + ', '.join(for_list2) + ' follows ' + readable

    readable = readable.replace('marker', '')

    return readable

def expand_expr(expr: str):
    expanded_expr = expr[:]
    magic_string = "@19023847@"

    for core_expr in expression_utils.get_configuration_data():
        index = expr.find(r"(" + core_expr + r"[")
        if index != -1:
            index += 1
            replacing_args = expression_utils.get_args(expr[index:])
            args_to_be_replaced = expression_utils.get_args(
                expression_utils.get_configuration_data()[core_expr].short_mpl_raw)

            replacement_map = {}
            for ind in range(len(args_to_be_replaced)):
                replacement_map[args_to_be_replaced[ind]] = replacing_args[ind] + magic_string

            expanded_expr = (
                expression_utils.replace_keys_in_string(
                    expression_utils.get_configuration_data()[core_expr].full_mpl[:], replacement_map))
            expanded_expr = expanded_expr.replace(magic_string, "")

    return expanded_expr

def make_readable(expression: str):
    readable = expression

    temp_chain = []
    head = disintegrate_implication(expression, temp_chain)
    chain = []
    for element in temp_chain:
        chain.append(element[0])
    chain.append(head)
    if "Anchor" in chain[0]:
        chain = chain[1:]

    readable = make_readable_from_chain(chain)

    return readable

def make_readable_title(expression: str):
    readable = expression

    temp_chain = []
    head = disintegrate_implication(expression, temp_chain)
    chain = []
    for element in temp_chain:
        chain.append(element[0])
    chain.append(head)
    if "Anchor" in chain[0]:
        chain = chain[1:]

    readable = make_readable_from_chain_title(chain)

    return readable

def format_implication(sublist):
    rule = make_readable(sublist[2])

    application = ''
    if rule == sublist[2]:
        rule = ''
    else:
        temp_list = sublist[3:]
        temp_list.extend([copy.copy(sublist[0])])
        if "Anchor" in temp_list[0]:
            temp_list = temp_list[1:]
        application = make_readable_from_chain(temp_list)

    output = 'RULE:  ' + rule + ' IMPLIES:  '  + application

    return output

def format_mirroring(sublist):
    mirrored = make_readable_title(sublist[0])
    original = make_readable_title(sublist[2])

    output = mirrored + ' mirrored from ' + original

    return output

def format_reformulation(sublist):
    reformulated = make_readable_title(sublist[0])
    original = make_readable_title(sublist[2])

    output = reformulated + ' reformulated from ' + original

    return output