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

import create_expressions
from analyze_expressions import *


from create_expressions import disintegrate_implication

test_expr = '(>[5,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[7,10,11,6])(>[12](in3[8,10,12,6])(>[13](in3[9,10,13,6])(in3[11,12,13,5]))))))'

def rewrite_expression(expression: str):
    rewritten = ''


    assert (expression.startswith('(in') or
            expression.startswith('(in2') or
            expression.startswith('(in3'))

    args = get_args(expression)

    if expression.startswith('(in'):
        rewritten = 'marker' + args[-2]

    if expression.startswith('(in2'):
        rewritten = args[-1] + '(' + 'marker' + args[-3]  + ')'

    if expression.startswith('(in3'):
        rewritten = '(marker' + args[-4] + args[-1] + 'marker' + args[-3] + ')'


    return rewritten

def rewrite_expression2(expression: str):
    rewritten = ''


    assert (expression.startswith('(in[') or
            expression.startswith('(in2') or
            expression.startswith('(in3') or
            expression.startswith('(=['))

    args = get_args(expression)

    if expression.startswith('(in['):
        rewritten = 'marker' + args[-2] + ' in ' + args[-1]

    if expression.startswith('(in2['):
        rewritten = args[-1] + '(' + 'marker' + args[-3]  + ')=' + 'marker' + args[-2]

    if expression.startswith('(in3['):
        rewritten = 'marker' + args[-4] + args[-1] + 'marker' + args[-3] + '=' + args[2]

    if expression.startswith('(=['):
        rewritten = 'marker' + args[-2] + '=' 'marker' + args[-1]

    return rewritten

def make_readable_simple_implication_title(chain: list[str]):
    readable = ''
    args_list = []

    for element in chain:
        assert element.startswith('(in')
        args_list.append(get_args(element))

    head_output = args_list[-1][-2]

    for index, args in enumerate(args_list):
        assert args != args_list[-1]
        output = args[-2]
        if output == head_output:
            readable = rewrite_expression(chain[index]) + '=' +  rewrite_expression(chain[-1])
            break

    for index, args in enumerate(args_list):
        output = args[-2]

        if 'marker' + output in readable:
            rewritten = rewrite_expression(chain[index])
            readable = readable.replace('marker' + output, rewritten)

    readable = readable.replace('marker', '')

    return readable

def make_readable_equality(chain: list[str]):
    eq_args = get_args(chain[-1])
    left_output = eq_args[0]
    right_output = eq_args[1]

    left_operator = ''
    right_operator = ''
    for element in chain:
        if element.startswith('(in2') or element.startswith('(in3'):
            args = get_args(element)

            if left_output in args:
                left_operator = element

            if right_output in args:
                right_operator = element

    assert left_operator and right_operator

    if left_operator != right_operator:

        rewritten_left = rewrite_expression2(left_operator)
        rewritten_right = rewrite_expression2(right_operator)
        rewritten_equality = rewrite_expression2(chain[-1])

        readable = 'from ' + rewritten_left + ' and ' + rewritten_right + ' follows ' + rewritten_equality

    else:
        if left_operator:
            non_empty = left_operator
        else:
            non_empty = right_operator

        args = get_args(non_empty)
        assert left_output in args and right_output in args

        rewritten_non_empty = rewrite_expression2(non_empty)
        rewritten_equality = rewrite_expression2(chain[-1])

        readable = 'from ' + rewritten_non_empty + ' follows ' + rewritten_equality

    readable = readable.replace('marker', '')

    return readable

def make_readable_element(chain: list[str]):
    head = chain[-1]
    readable = ''

    assert len(chain) == 2

    element_arg = get_args(head)[-2]

    assert element_arg in get_args(chain[0])

    readable = 'from ' + rewrite_expression2(chain[0]) +  ' follows ' + rewrite_expression2(chain[-1])

    readable = readable.replace('marker', '')

    return readable

def make_readable_existence(chain: list[str]):
    readable = ''

    last_removed = list_last_removed_args(chain[-1])
    left_expr, right_expr = extract_values_regex(chain[-1])
    right_expr = right_expr[1:]

    readable = 'from ' + rewrite_expression2(chain[0])
    for index in range(1, len(chain) - 1):
        readable += ' and ' + rewrite_expression2(chain[index])

    readable += ' follows existence of ' + rewrite_expression2(left_expr) + ' with ' + rewrite_expression2(right_expr)

    readable = readable.replace('marker', '')

    return readable

def make_readable_from_chain(chain: list[str]):
    head = chain[-1]
    readable = ''

    if head.startswith('(in2') or head.startswith('(in3'):
        readable = make_readable_simple_implication(chain)

    if head.startswith('(=['):
        readable = make_readable_equality(chain)

    if head.startswith('(in['):
        readable = make_readable_element(chain)

    if head.startswith('!(>['):
        readable = make_readable_existence(chain)

    return readable

def make_readable_from_chain_title(chain: list[str]):
    head = chain[-1]
    readable = ''

    if head.startswith('(in2') or head.startswith('(in3'):
        readable = make_readable_simple_implication_title(chain)


    return readable

def make_readable_simple_implication(chain: list[str]):
    readable = ''
    args_list = []

    for element in chain:
        assert element.startswith('(in')
        args_list.append(get_args(element))

    head_output = args_list[-1][-2]

    left_head = ''
    left_head_index = -1
    for index, args in enumerate(args_list):
        assert args != args_list[-1]
        output = args[-2]
        if output == head_output:
            readable = rewrite_expression(chain[index]) + '=' + head_output + '=' +  rewrite_expression(chain[-1])
            left_head = chain[index]
            left_head_index = index
            break


    for_list = []
    head = chain[-1]
    head_args = get_args(head)
    left_head_args = get_args(left_head)
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
    #for_list2.append(rewrite_expression2(left_head))

    if for_list2:
        readable = 'from ' + ','.join(for_list2) + ' follows ' + readable

    readable = readable.replace('marker', '')

    return readable


def make_readable(expression: str):
    readable = expression

    temp_chain = []
    head = disintegrate_implication(expression, temp_chain)
    chain = []
    for element in temp_chain:
        chain.append(element[0])
    chain.append(head)
    if create_expressions.anchor[3] in chain[0]:
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
    if create_expressions.anchor[3] in chain[0]:
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
        if create_expressions.anchor[3] in temp_list[0]:
            temp_list = temp_list[1:]
        application = make_readable_from_chain(temp_list)

    output = 'RULE:  ' + rule + ' IMPLIES:  '  + application

    return output

def format_mirroring(sublist):
    mirrored = make_readable_title(sublist[0])
    original = make_readable_title(sublist[2])

    output = mirrored + ' mirrored from ' + original

    return output