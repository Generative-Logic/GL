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

import os

import time

import multiprocessing

import run_modes










def modify_core_expression_map():
    for expr in create_expressions.core_expression_map:
        temp_str = create_expressions.core_expression_map[expr][3]
        if os.path.isfile(str(create_expressions.core_expression_map[expr][3])):
            definition = create_expressions.read_tree_from_file(str(create_expressions.core_expression_map[expr][3]))
            definition = definition.replace("\n", "")  # Remove spaces
            definition = definition.replace(" ", "")  # Remove spaces
            definition = definition.replace("\t", "")  # Remove spaces
            create_expressions.core_expression_map[expr] = (create_expressions.core_expression_map[expr][0], create_expressions.core_expression_map[expr][1],
                                         create_expressions.core_expression_map[expr][2], definition, create_expressions.core_expression_map[expr][4],
                                         create_expressions.core_expression_map[expr][5], create_expressions.core_expression_map[expr][6])






def main():
    modify_core_expression_map()

    start_time = time.time()


    run_modes.full_run()

    end_time = time.time()

    print(f"Overall runtime: {end_time - start_time:.5f} seconds")


if __name__ == "__main__":

    multiprocessing.freeze_support()
    main()