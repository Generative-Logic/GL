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
from typing import Set
import parameters
import analyze_expressions

from pathlib import Path


# wherever this file lives, assume the project root is its parent folder
PROJECT_ROOT = Path(__file__).resolve().parent


_ALL_PERMUTATIONS_TEST = {}

to_be_created =\
    [
        '(>[+](NaturalNumbers[N,0,1,s,+,*])(>[a,b,c](in3[a,b,c,+])(in3[b,a,c,+])))',
        '(>[*](NaturalNumbers[N,0,1,s,+,*])(>[a,b,c](in3[a,b,c,*])(in3[b,a,c,*])))',
        '(>[+,0](NaturalNumbers[N,0,1,s,+,*])(>[a,c](in3[a,0,c,+])(in3[0,a,c,+])))',
        '(>[*,0](NaturalNumbers[N,0,1,s,+,*])(>[a,c](in3[a,0,c,*])(in3[0,a,c,*])))',
        '(>[1,s,+](NaturalNumbers[N,0,1,s,+,*])(>[n,m](in3[n,1,m,+])(in2[n,m,s])))',
        '(>[s,+](NaturalNumbers[N,0,1,s,+,*])(>[a,c](in2[a,c,s])(>[b,d](in3[c,b,d,+])(>[e](in3[a,b,e,+])(in2[e,d,s])))))',
        '(>[+](NaturalNumbers[N,0,1,s,+,*])(>[b,c,d](in3[b,c,d,+])(>[a,e](in3[a,d,e,+])(>[f](in3[a,b,f,+])(in3[f,c,e,+])))))',
        '(>[*](NaturalNumbers[N,0,1,s,+,*])(>[b,c,d](in3[b,c,d,*])(>[a,e](in3[a,d,e,*])(>[f](in3[a,b,f,*])(in3[f,c,e,*])))))',
        '(>[+,*](NaturalNumbers[N,0,1,s,+,*])(>[b,c,d](in3[b,c,d,+])(>[a,e](in3[a,d,e,*])(>[f](in3[a,b,f,*])(>[g](in3[a,c,g,*])(in3[f,g,e,+]))))))',
        '(>[+,*](NaturalNumbers[N,0,1,s,+,*])(>[a,b,d](in3[a,b,d,+])(>[c,e](in3[d,c,e,*])(>[f](in3[a,c,f,*])(>[g](in3[b,c,g,*])(in3[f,g,e,+]))))))',
        '(>[+](NaturalNumbers[N,0,1,s,+,*])(>[a,b,d](in3[a,b,d,+])(>[c,e](in3[d,c,e,+])(>[f](in3[a,c,f,+])(in3[f,b,e,+])))))',
        '(>[s,+,*](NaturalNumbers[N,0,1,s,+,*])(>[a,c](in2[a,c,s])(>[b,d](in3[c,b,d,*])(>[e](in3[a,b,e,*])(in3[e,b,d,+])))))'
    ]

def read_theorems(file_name: str) -> Set[str]:
    """
    Reads a file line by line and returns a set of theorem strings.

    Parameters:
    file_name (str): Path to the input file containing one theorem per line.

    Returns:
    Set[str]: A set where each element is a line from the file.

    Example:
    --------
    >>> # Given a file 'theorems.txt' with lines:
    >>> # Pythagorean theorem
    >>> # Fundamental theorem of calculus
    >>> thrms = read_theorems('theorems.txt')
    >>> thrms
    {'Pythagorean theorem', 'Fundamental theorem of calculus'}
    """
    theorems: Set[str] = set()
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            # Strip newline and any surrounding whitespace
            theorem = line.rstrip('\n').strip()
            if theorem:
                theorems.add(theorem)
    return theorems

def test1():
    theorems_folder = PROJECT_ROOT / 'files/theorems'
    file_name = str(theorems_folder / "theorems.txt")

    theorem_set = read_theorems(file_name)

    all_permutations = create_expressions.generate_all_permutations(parameters.max_number_simple_expressions + 1)

    found = []
    not_found = []

    for theorem in to_be_created:
        reshuffled = create_expressions.reshuffle(theorem, all_permutations, True)[0]
        reshuffled_mirrored = create_expressions.create_reshuffled_mirrored(theorem, all_permutations)

        if reshuffled in theorem_set or reshuffled_mirrored in theorem_set:
            found.append(theorem)
        else:
            not_found.append(theorem)

    print("test_create_expressions.test1")
    print("\n found \n")

    for theorem in found:
        print(theorem)

    print("\n not found \n")

    for theorem in not_found:
        print(theorem)

    print()

def test2():
    theorems_folder = PROJECT_ROOT / 'files/theorems'
    file_name = str(theorems_folder / "theorems.txt")
    theorem_set = read_theorems(file_name)

    all_permutations = create_expressions.generate_all_permutations(parameters.max_number_simple_expressions + 1)

    set_of_interest = set()

    for theorem in to_be_created:
        reshuffled = create_expressions.reshuffle(theorem, all_permutations, True)[0]
        set_of_interest.add(reshuffled)

        reshuffled_mirrored = create_expressions.create_reshuffled_mirrored(theorem, all_permutations)
        set_of_interest.add(reshuffled_mirrored)

    list_of_interest = []

    counter = 0
    for theorem in theorem_set:
        reshuffled = create_expressions.reshuffle(theorem, all_permutations, True)[0]

        if reshuffled in set_of_interest:
            list_of_interest.append(theorem)


        if counter % 100 == 0:
            print(counter)

        counter += 1

    theorems_folder = PROJECT_ROOT / 'files/theorems'
    file_name = str(theorems_folder / "theorems.txt")
    try:
        with open(file_name, 'w', encoding='utf-8') as file:
            for expr in list_of_interest:
                file.write(expr + "\n")
        print(f"Successfully wrote to {file_name}")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")

def test3():
    theorems_folder = PROJECT_ROOT / 'files/theorems'
    file_name = str(theorems_folder / "checked_theorems.txt")
    theorem_set = read_theorems(file_name)

    tmp_lst = list(theorem_set)
    tmp_lst.sort()

    if '(>[4,5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in2[10,11,4])(>[](in2[10,7,4])(in3[11,8,9,5])))))' in theorem_set:
        test = 0

    more_lines = [
        '(>[4,5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in2[10,11,4])(>[](in2[10,7,4])(in3[11,8,9,5])))))',
        '(>[4,5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in2[10,11,4])(>[](in2[10,8,4])(in3[7,11,9,5])))))',
        '(>[4,5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in2[10,11,4])(>[](in3[7,8,10,5])(in2[9,11,4])))))',
        '(>[4,5,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in2[10,11,4])(>[](in3[8,10,7,6])(in3[8,11,9,6])))))',
        '(>[4,5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10](in2[7,10,4])(>[](in2[7,8,4])(in3[7,10,9,5])))))',
        '(>[4,5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10](in2[8,10,4])(>[](in2[8,7,4])(in3[10,8,9,5])))))',
        '(>[4,5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10](in2[8,10,4])(>[11](in2[9,11,4])(in3[7,10,11,5])))))',
        '(>[5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11,12](in3[10,11,12,5])(>[](in3[7,8,10,5])(in3[9,11,12,5])))))',
        '(>[5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11,12](in3[10,11,12,5])(>[](in3[7,8,11,5])(in3[10,9,12,5])))))',
        '(>[5,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11,12](in3[10,11,12,6])(>[](in3[7,8,10,5])(in3[9,11,12,6])))))',
        '(>[5,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11,12](in3[10,11,12,6])(>[](in3[7,8,11,5])(in3[10,9,12,6])))))',
        '(>[5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[10,8,11,5])(>[](in3[7,8,10,5])(in3[9,8,11,5])))))',
        '(>[5,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[10,8,11,6])(>[](in3[7,8,10,5])(in3[9,8,11,6])))))',
        '(>[5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[7,10,11,5])(>[](in3[7,8,10,5])(in3[7,9,11,5])))))',
        '(>[5,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[7,10,11,6])(>[](in3[7,8,10,5])(in3[7,9,11,6])))))',
        '(>[5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[8,10,11,5])(>[](in3[7,8,10,5])(in3[8,9,11,5])))))',
        '(>[5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[8,10,11,5])(>[](in3[8,10,7,5])(in3[11,8,9,5])))))',
        '(>[5,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[8,10,11,6])(>[](in3[7,8,10,5])(in3[8,9,11,6])))))',
        '(>[5,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[8,10,11,6])(>[](in3[8,10,7,6])(in3[11,8,9,5])))))',
        '(>[4,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(>[10,11](in2[10,11,4])(>[](in2[10,7,4])(in3[11,8,9,6])))))',
        '(>[4,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(>[10,11](in2[10,11,4])(>[](in2[10,8,4])(in3[7,11,9,6])))))',
        '(>[4,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(>[10,11](in2[10,11,4])(>[](in3[7,8,10,6])(in2[9,11,4])))))',
        '(>[4,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(>[10](in2[7,10,4])(>[](in2[7,8,4])(in3[7,10,9,6])))))',
        '(>[4,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(>[10](in2[8,10,4])(>[](in2[8,7,4])(in3[10,8,9,6])))))',
        '(>[5,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(>[10,11,12](in3[10,11,12,5])(>[](in3[7,8,10,6])(in3[9,11,12,5])))))',
        '(>[5,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(>[10,11,12](in3[10,11,12,5])(>[](in3[7,8,11,6])(in3[10,9,12,5])))))',
        '(>[6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(>[10,11,12](in3[10,11,12,6])(>[](in3[7,8,10,6])(in3[9,11,12,6])))))',
        '(>[6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(>[10,11,12](in3[10,11,12,6])(>[](in3[7,8,11,6])(in3[10,9,12,6])))))',
        '(>[5,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(>[10,11](in3[10,8,11,5])(>[](in3[7,8,10,6])(in3[9,8,11,5])))))',
        '(>[6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(>[10,11](in3[10,8,11,6])(>[](in3[7,8,10,6])(in3[9,8,11,6])))))',
        '(>[5,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(>[10,11](in3[7,10,11,5])(>[](in3[7,8,10,6])(in3[7,9,11,5])))))',
        '(>[6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(>[10,11](in3[7,10,11,6])(>[](in3[7,8,10,6])(in3[7,9,11,6])))))',
        '(>[5,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(>[10,11](in3[8,10,11,5])(>[](in3[7,8,10,6])(in3[8,9,11,5])))))',
        '(>[5,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(>[10,11](in3[8,10,11,5])(>[](in3[8,10,7,5])(in3[11,8,9,6])))))',
        '(>[6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(>[10,11](in3[8,10,11,6])(>[](in3[7,8,10,6])(in3[8,9,11,6])))))',
        '(>[6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(>[10,11](in3[8,10,11,6])(>[](in3[8,10,7,6])(in3[11,8,9,6])))))'
    ]

    #tmp_lst = [x for x in tmp_lst if x not in more_lines]


    analyze_expressions.analyze_expressions(tmp_lst)


    for theorem in analyze_expressions.global_theorem_list:
        print(theorem)

    """
    analyze_expressions.global_body_of_proves = analyze_expressions.BodyOfProves()

    theorem_list = []
    for tuple in analyze_expressions.global_theorem_list:
        theorem_list.append(tuple[0])

    analyze_expressions.analyze_expressions(theorem_list)
    """

def test4():
    global _ALL_PERMUTATIONS_TEST

    theorems_folder = PROJECT_ROOT / 'files/theorems'
    file_name = str(theorems_folder / "theorems.txt")
    theorem_set = read_theorems(file_name)
    checked = set()

    _ALL_PERMUTATIONS_TEST = create_expressions.generate_all_permutations(parameters.size_all_permutations_ana)


    for theorem in theorem_set:
        #theorem = '(>[4,5](NaturalNumbers[1,2,3,4,5])(>[6,7,8](in3[6,7,8,5])(>[9,10](in3[6,9,10,5])(>[11,12](in3[6,11,12,5])(>[](in3[7,9,11,4])(in3[8,10,12,4]))))))'
        #theorem = '(>[3,4,5](NaturalNumbers[1,2,3,4,5])(>[6,7,8](in3[6,7,8,4])(>[9,10,11](in3[9,10,11,5])(>[12](in3[6,9,12,5])(>[](in2[10,7,3])(in3[8,11,12,5]))))))'
        #theorem = '(>[2,3,4](NaturalNumbers[1,2,3,4,5])(>[6,7](in2[6,7,3])(>[8](in2[2,8,3])(in3[6,8,7,4]))))'
        #theorem = '(>[4,5](NaturalNumbers[1,2,3,4,5])(>[6,7,8](in3[6,7,8,5])(>[9,10](in3[6,9,10,5])(>[11,12](in3[6,11,12,5])(>[](in3[7,9,11,4])(in3[10,8,12,4]))))))'
        #theorem = '(>[4](NaturalNumbers[1,2,3,4,5])(>[6,7,8](in3[6,7,8,4])(in3[7,6,8,4])))'
        #theorem = '(>[4,5](NaturalNumbers[1,2,3,4,5])(>[6,7,8](in3[6,7,8,4])(>[9,10,11](in3[9,10,11,5])(>[12](in3[7,10,12,5])(>[](in3[10,9,6,4])(in3[12,11,8,4]))))))'
        #theorem = '(>[4,5](NaturalNumbers[1,2,3,4,5])(>[6,7,8](in3[6,7,8,4])(>[9,10,11](in3[9,10,11,4])(>[12](in3[6,10,12,5])(>[](in3[9,10,7,5])(in3[11,12,8,4]))))))'
        #theorem = '(>[5,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(>[10,11](in3[7,10,11,5])(>[12,13](in3[12,7,13,6])(>[](in3[8,12,10,6])(in3[9,13,11,5]))))))'
        #theorem = '(>[6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(>[10,11](in3[7,10,11,6])(>[12](in3[8,12,10,6])(in3[9,12,11,6])))))'
        #theorem = "(>[2,5](NaturalNumbers[1,2,3,4,5,6])(>[7,8](in3[2,7,8,5])(in3[7,2,8,5])))"
        #theorem = '(>[5,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[8,10,11,6])(>[](in3[8,10,7,6])(in3[11,8,9,5])))))'
        #theorem = '(>[5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(in3[8,7,9,5])))'
        #theorem = '(>[6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(>[10,11](in3[8,10,11,6])(>[](in3[8,10,7,6])(in3[11,8,9,6])))))'
        #theorem = '(>[4,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(>[10](in2[8,10,4])(>[](in2[8,7,4])(in3[10,8,9,6])))))'
        #theorem = '(>[4,5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10](in2[8,10,4])(>[](in2[8,7,4])(in3[8,10,9,5])))))'
        if create_expressions.check_input_variables_theorem(theorem):
            if create_expressions.check_theorem_complexity_per_operator(theorem):
                if create_expressions.check_input_variables_order(theorem, _ALL_PERMUTATIONS_TEST):
                    checked.add(theorem)

    checked = sorted(checked)

    theorems_folder = PROJECT_ROOT / 'files/theorems'
    file_name = str(theorems_folder / "theorems.txt")
    try:
        with open(file_name, 'w', encoding='utf-8') as file:
            for expr in checked:
                file.write(expr + "\n")
        print(f"Successfully wrote to {file_name}")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")



