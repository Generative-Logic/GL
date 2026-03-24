/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025 Generative Logic UG(haftungsbeschr�nkt)

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
#ifdef USE_MIMALLOC
#include <mimalloc.h>
#endif
#include "run_modes.hpp"

int main(int argc, char* argv[]) {
#ifdef USE_MIMALLOC
    int v = mi_version();  // ensure mimalloc override DLL is loaded
    std::cout << "mimalloc version: " << v << std::endl;
#endif
    auto start = std::chrono::high_resolution_clock::now();

    // Grab anchor id if provided (e.g., "Peano" or "Gauss")
    // Default to IncubatorPeano for MSVS F5 debugging; full_run passes "Peano"/"Gauss" via argv
    std::string anchor_id = (argc > 1) ? std::string(argv[1]) : "Gauss";

    // run_modes::quickRun();
    run_modes::fullRun(anchor_id);  // <-- pass it through

    // Return memory to OS (mimalloc retains free pages by default)
#ifdef USE_MIMALLOC
    mi_collect(true);
#endif

    auto end = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(end - start).count();
    std::cout << "Runtime of the executable (counter example filter + prover): "
        << secs << " seconds" << std::endl;
    return 0;
}
