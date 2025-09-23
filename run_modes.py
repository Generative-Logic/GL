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

import generate_full_proof_graph
import time






from pathlib import Path

import subprocess


# wherever this file lives, assume the project root is its parent folder
PROJECT_ROOT = Path(__file__).resolve().parent






def run_gl_quick():
    exe_path = PROJECT_ROOT / 'GL_Quick_VS' / 'GL_Quick' / 'gl_quick.exe'
    if not exe_path.exists():
        raise FileNotFoundError(exe_path)

    # Inherit parent's stdout/stderr → prints live instead of at the end
    subprocess.run([str(exe_path)], cwd=PROJECT_ROOT, check=True)






def full_run():
    start_time = time.time()
    print("Conjecture creation started.")
    create_expressions.create_expressions_parallel()
    print("Conjecture creation finished.")
    end_time = time.time()
    print(f"Conjecture creation runtime: {end_time - start_time:.5f} seconds")

    run_gl_quick()


    generate_full_proof_graph.generate_proof_graph_pages()