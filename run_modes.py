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

from test_functions import *

import analyze_expressions
import create_expressions

import generate_full_proof_graph
import parameters

import time

import multiprocessing

import test_create_expressions

from pathlib import Path


# wherever this file lives, assume the project root is its parent folder
PROJECT_ROOT = Path(__file__).resolve().parent



def quick_run():
    expr_of_interest1 = "(>[5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(in3[8,7,9,5])))"
    expr_of_interest2 = '(>[3,4,5](NaturalNumbers[1,2,3,4,5,6])(>[7,8](in2[7,8,4])(in3[7,3,8,5])))'
    expr_of_interest3 = '(>[5,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[7,10,11,6])(>[12](in3[8,10,12,6])(>[13](in3[9,10,13,6])(in3[11,12,13,5]))))))'
    expr_of_interest4 = '(>[5,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(>[10,11](in3[7,10,11,6])(>[12,13](in3[7,12,13,6])(>[](in3[8,10,12,5])(in3[9,11,13,5]))))))'
    expr_of_interest5 = '(>[4,5,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in2[10,11,4])(>[](in3[10,8,7,6])(in3[11,8,9,6])))))'
    expr_of_interest6 = "(>[6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(in3[8,7,9,6])))"
    expr_of_interest7 = '(>[5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[7,10,11,5])(>[12](in3[9,10,12,5])(in3[11,8,12,5])))))'
    expr_of_interest8 = "(>[6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(>[10,11](in3[7,10,11,6])(>[12](in3[8,12,10,6])(in3[9,12,11,6])))))"
    expr_of_interest9 = "(>[4,5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10](in2[7,10,4])(>[11](in2[9,11,4])(in3[10,8,11,5])))))"
    expr_of_interest10 = "(>[5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[7,10,11,5])(>[12](in3[8,12,10,5])(in3[9,12,11,5])))))"
    expr_of_interest11 = "(>[2,5](NaturalNumbers[1,2,3,4,5,6])(>[7,8](in3[2,7,8,5])(in3[7,2,8,5])))"
    expr_of_interest12 = "(>[2,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8](in3[2,7,8,6])(in3[7,2,8,6])))"
    expr_of_interest13 = '(>[3,4,5](NaturalNumbers[1,2,3,4,5,6])(>[7,8](in2[7,8,4])(in3[3,7,8,5])))'



    expression_list = [expr_of_interest1,
                       expr_of_interest2,
                       expr_of_interest3,
                       expr_of_interest4,
                       expr_of_interest5,
                       expr_of_interest6,
                       expr_of_interest7,
                       expr_of_interest8,
                       expr_of_interest9,
                       expr_of_interest10,
                       expr_of_interest11,
                       expr_of_interest12,
                       expr_of_interest13]

    analyze_expressions.analyze_expressions(expression_list)


    if parameters.debug:
        expr_lst = ['(NaturalNumbers[1,2,3,4,5,6])','(in3[7,8,9,5])','(in2[rec0,7,4])']
        analyze_expressions.find_ends(expr_lst)
    generate_full_proof_graph.generate_proof_graph_pages(analyze_expressions.global_theorem_list)

def full_run():
    create_expressions.create_expressions_parallel()

    theorems_folder = PROJECT_ROOT / 'files/theorems'
    file_name = str(theorems_folder / "theorems.txt")
    theorem_set = test_create_expressions.read_theorems(file_name)

    tmp_lst = list(theorem_set)
    tmp_lst.sort()

    analyze_expressions.analyze_expressions(tmp_lst)

    if parameters.debug:
        expr_lst = ['(NaturalNumbers[1,2,3,4,5,6])','(in3[7,8,9,5])','(in2[rec0,7,4])']
        analyze_expressions.find_ends(expr_lst)
    generate_full_proof_graph.generate_proof_graph_pages(analyze_expressions.global_theorem_list)