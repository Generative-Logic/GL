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



from pathlib import Path
from configuration_reader import configuration_reader



# wherever this file lives, assume the project root is its parent folder
PROJECT_ROOT = Path(__file__).resolve().parent



peano_config_path = PROJECT_ROOT / "files" / "config" / "ConfigPeano.json"

configuration_peano = configuration_reader(peano_config_path)



gauss_config_path = PROJECT_ROOT / "files" / "config" / "ConfigGauss.json"

configuration_gauss = configuration_reader(gauss_config_path)

# 0: Peano
# 1. Gauss
test_variant = 0

_ALL_PERMUTATIONS_TEST = {}

to_be_created_peano =\
    [
        '(>[+](AnchorPeano[N,0,s,+,*,1])(>[a,b,c](in3[a,b,c,+])(in3[b,a,c,+])))',
        '(>[*](AnchorPeano[N,0,s,+,*,1])(>[a,b,c](in3[a,b,c,*])(in3[b,a,c,*])))',
        '(>[+,0](AnchorPeano[N,0,s,+,*,1])(>[a,c](in3[a,0,c,+])(in3[0,a,c,+])))',
        '(>[*,0](AnchorPeano[N,0,s,+,*,1])(>[a,c](in3[a,0,c,*])(in3[0,a,c,*])))',
        '(>[1,s,+](AnchorPeano[N,0,s,+,*,1])(>[n,m](in3[n,1,m,+])(in2[n,m,s])))',
        '(>[s,+](AnchorPeano[N,0,s,+,*,1])(>[a,c](in2[a,c,s])(>[b,d](in3[c,b,d,+])(>[e](in3[a,b,e,+])(in2[e,d,s])))))',
        '(>[+](AnchorPeano[N,0,s,+,*,1])(>[b,c,d](in3[b,c,d,+])(>[a,e](in3[a,d,e,+])(>[f](in3[a,b,f,+])(in3[f,c,e,+])))))',
        '(>[*](AnchorPeano[N,0,s,+,*,1])(>[b,c,d](in3[b,c,d,*])(>[a,e](in3[a,d,e,*])(>[f](in3[a,b,f,*])(in3[f,c,e,*])))))',
        '(>[+,*](AnchorPeano[N,0,s,+,*,1])(>[b,c,d](in3[b,c,d,+])(>[a,e](in3[a,d,e,*])(>[f](in3[a,b,f,*])(>[g](in3[a,c,g,*])(in3[f,g,e,+]))))))',
        '(>[+,*](AnchorPeano[N,0,s,+,*,1])(>[a,b,d](in3[a,b,d,+])(>[c,e](in3[d,c,e,*])(>[f](in3[a,c,f,*])(>[g](in3[b,c,g,*])(in3[f,g,e,+]))))))',
        '(>[+](AnchorPeano[N,0,s,+,*,1])(>[a,b,d](in3[a,b,d,+])(>[c,e](in3[d,c,e,+])(>[f](in3[a,c,f,+])(in3[f,b,e,+])))))',
        '(>[s,+,*](AnchorPeano[N,0,s,+,*,1])(>[a,c](in2[a,c,s])(>[b,d](in3[c,b,d,*])(>[e](in3[a,b,e,*])(in3[e,b,d,+])))))'
    ]

to_be_created_gauss =\
    [
        #"(>[1,2,3,4,5,6](AnchorGauss[1,2,3,4,5,6,7,8])(AnchorPeano[1,2,3,4,5,6]))",
        "(>[1,3,4](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(preorder[1,4,9,10])))",
        "(>[1,4](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10](preorder[1,4,9,10])(>[11](preorder[1,4,10,11])(preorder[1,4,9,11]))))",
        "(>[1,2,3,4](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10,11](limitSet[1,4,9,10,11])(>[12](in2[10,12,3])(>[](interval[1,4,2,12,9])(interval[1,4,2,10,11])))))",
        "(>[1,2,3,4](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10,11](limitSequence[1,4,9,10,11])(>[12](in2[9,12,3])(>[](sequence[1,4,2,12,10])(sequence[1,4,2,9,11])))))",
        "(>[1,2,4](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10](fXY[9,10,1])(>[11](interval[1,4,2,11,10])(sequence[1,4,2,11,9]))))",
        "(>[1,2,4](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10](interval[1,4,2,9,10])(in[9,10])))",
        "(>[1,2,3,4,5,7,8](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10](fold[1,3,4,8,2,9,10])(>[11](in3[7,10,11,5])(>[12](in2[9,12,3])(in3[9,12,11,5])))))"
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

def test1(test_variant: str):

    config = None
    to_be_created = None
    if test_variant == "Peano":
        config = configuration_peano
        to_be_created = to_be_created_peano
    elif test_variant == "Gauss":
        config = configuration_gauss
        to_be_created = to_be_created_gauss

    assert config is not None
    create_expressions.set_configuration(config)
    create_expressions.set_operators()

    theorems_folder = PROJECT_ROOT / 'files/theorems'
    file_name = str(theorems_folder / "reshuffled_theorems.txt")

    theorem_set = read_theorems(file_name)



    all_permutations = create_expressions.generate_all_permutations(config.parameters.max_number_simple_expressions + 1)

    found = []
    not_found = []

    for theorem in to_be_created:
        reshuffled = create_expressions.reshuffle(theorem, all_permutations, True)[0]
        reshuffled_mirrored = create_expressions.create_reshuffled_mirrored(theorem, all_permutations)

        if reshuffled in theorem_set or (reshuffled_mirrored in theorem_set and reshuffled_mirrored != ""):
            found.append(theorem)
        else:
            not_found.append(theorem)
            create_expressions.create_reshuffled_mirrored(theorem, all_permutations)

    print("test_create_expressions.test1")
    print("\n found \n")

    for theorem in found:
        print(theorem)

    print("\n not found \n")

    for theorem in not_found:
        print(theorem)

    print()


def test2(test_variant: str):
    config = None
    to_be_created = None
    if test_variant == "Peano":
        config = configuration_peano
        to_be_created = to_be_created_peano
    elif test_variant == "Gauss":
        config = configuration_gauss
        to_be_created = to_be_created_gauss

    assert config is not None
    create_expressions.set_configuration(config)
    create_expressions.set_operators()

    theorems_folder = PROJECT_ROOT / 'files/theorems'
    file_name = str(theorems_folder / "theorems.txt")
    theorem_set = read_theorems(file_name)

    all_permutations = create_expressions.generate_all_permutations(config.parameters.max_number_simple_expressions + 1)

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

    file_name = str(theorems_folder / "created_theorems.txt")
    try:
        with open(file_name, 'w', encoding='utf-8') as file:
            for expr in list_of_interest:
                file.write(expr + "\n")
        print(f"Successfully wrote to {file_name}")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")








