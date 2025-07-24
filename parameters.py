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

import re

max_number_simple_expressions = 5
max_size_mapping_def_set = 7
max_number_args_expr = 3
shifting_parameter = 10000
default_number_leafs_last_added_expr = 10000
max_number_args_to_connect_with_definitions = 0
size_all_permutations_ana = 7
size_all_binaries_ana = 10
operator_threshold = 5
max_iteration_number_proof = 17
max_number_statements_per_key = 10
max_values_for_def_sets = {"(1)": 1, "P(x(1)(1))": 1, "P(x(1)(x(1)(1)))": 2, "P(1)": 1}
max_complexity_if_anchor_parameter_connected = {"(1)": 2, "P(x(1)(1))": 10, "P(x(1)(x(1)(1)))": 10, "P(1)": 10}
max_size_binary_list = 8
max_size_def_set_mapping = 5
max_size_target_set_mapping = 12
max_iteration_number_variable = 1
max_number_secondary_variables = 2
max_complexity_for_commutative_law = 2
track_history = True
standard_max_admission_depth = 0
induction_max_admission_depth = 1
standard_max_secondary_number = 1
induction_max_secondary_number = 2
min_num_operators_key = 2
max_values_for_operators = {'in2': 5, 'in3': 6}
# compile a regex that matches "(in2[<any-integer>+,3,4])"
pattern_to_exclude1 = re.compile(r'\(in2\[\d+,3,4]\)')

quick_run_mode = True
debug = False

