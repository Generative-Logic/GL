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
#include <string>
#include <cstring>
#ifdef USE_MIMALLOC
#include <mimalloc.h>
#endif
#include "run_modes.hpp"
#include "conjecturer.hpp"
#include "prover.hpp"

int main(int argc, char* argv[]) {
#ifdef USE_MIMALLOC
    int v = mi_version();  // ensure mimalloc override DLL is loaded
    std::cout << "mimalloc version: " << v << std::endl;
#endif
    auto start = std::chrono::high_resolution_clock::now();

    // --mirror-externals <theoremsDir>: rebuild compressed_external_theorems.txt
    if (argc >= 3 && std::strcmp(argv[1], "--mirror-externals") == 0) {
        namespace fs = std::filesystem;
        const fs::path projectRoot = fs::path(argv[0]).parent_path().parent_path().parent_path();
        const fs::path theoremsDir = projectRoot / argv[2];
        const fs::path extPath = theoremsDir / "externally_provided_theorems.txt";
        const fs::path compPath = theoremsDir / "compressed_external_theorems.txt";

        // Read externals
        std::vector<std::string> extTheorems;
        if (fs::exists(extPath)) {
            std::ifstream in(extPath);
            std::string line;
            while (std::getline(in, line)) {
                // trim
                while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' '))
                    line.pop_back();
                if (!line.empty()) extTheorems.push_back(line);
            }
        }

        if (extTheorems.empty()) {
            // Write empty file
            std::ofstream out(compPath, std::ios::trunc);
            std::cout << "External theorems: 0 originals + 0 mirrored variants.\n";
            return 0;
        }

        // Group by anchor tag
        std::map<std::string, std::vector<std::string>> anchorGroups;
        std::regex anchorRe(R"(\(Anchor([A-Za-z0-9_]+)\[)");
        for (auto& thm : extTheorems) {
            std::smatch m;
            std::string tag;
            if (std::regex_search(thm, m, anchorRe)) tag = m[1].str();
            anchorGroups[tag].push_back(thm);
        }

        std::set<std::string> extSet(extTheorems.begin(), extTheorems.end());
        std::vector<std::string> extWithMirrors = extTheorems;

        for (auto& [tag, group] : anchorGroups) {
            if (tag.empty()) continue;
            auto coreMap = ce::modifyCoreExpressionMap(tag);
            if (coreMap.empty()) {
                std::cout << "Warning: No config for anchor " << tag << ", skipping mirrors.\n";
                continue;
            }
            std::string anchorName = ce::findAnchorKey(coreMap);

            for (auto& thm : group) {
                std::string mirrored = ce::createReshuffledMirrored(thm, anchorName, true, coreMap);
                if (!mirrored.empty() && extSet.find(mirrored) == extSet.end()) {
                    extWithMirrors.push_back(mirrored);
                    extSet.insert(mirrored);
                }
            }
        }

        // Write output
        std::ofstream out(compPath, std::ios::trunc);
        for (auto& thm : extWithMirrors) out << thm << "\n";

        std::cout << "External theorems: " << extTheorems.size() << " originals + "
                  << (extWithMirrors.size() - extTheorems.size()) << " mirrored variants.\n";

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Mirror-externals runtime: "
            << std::chrono::duration<double>(end - start).count() << " seconds\n";
        return 0;
    }

    // --conjecture <anchorId>: run C++ conjecture generation only
    if (argc >= 3 && std::strcmp(argv[1], "--conjecture") == 0) {
        std::string anchorId = argv[2];
        std::cout << "Running C++ conjecturer for: " << anchorId << "\n";
        conj::Conjecturer c(anchorId);
        c.run();
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Conjecture generation runtime: "
            << std::chrono::duration<double>(end - start).count() << " seconds\n";
        return 0;
    }

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
