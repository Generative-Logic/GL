/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025 Generative Logic UG(haftungsbeschränkt)

 This program is free software : you can redistribute it and /or modify
 it under the terms of the GNU Affero General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.If not, see < https://www.gnu.org/licenses/>.

 ------------------------------------------------------------------------------

 This software is also available under a commercial license.For details,
 see: https://generative-logic.com/license

 Contributions to this project must be made under the terms of the
 Contributor License Agreement(CLA).See the project's CONTRIBUTING.md file.*/

#include <iostream>
#include <chrono>
#include "run_modes.hpp"

int main(int argc, char* argv[]) {
    auto start = std::chrono::high_resolution_clock::now();

    // Grab anchor id if provided (e.g., "Peano" or "Gauss")
    std::string anchor_id = (argc > 1) ? std::string(argv[1]) : "Gauss";

    // run_modes::quickRun();
    run_modes::fullRun(anchor_id);  // <-- pass it through

    auto end = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(end - start).count();
    std::cout << "Runtime of the executable (counter example filter + prover): "
        << secs << " seconds" << std::endl;
    return 0;
}
