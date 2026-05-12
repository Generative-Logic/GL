/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025-2026 Generative Logic UG(haftungsbeschr�nkt)

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
#include <fstream>
#ifdef USE_MIMALLOC
#include <mimalloc.h>
#endif
#include "run_modes.hpp"
#include "conjecturer.hpp"
#include "prover.hpp"
#include "tests/test_harness.hpp"

int main(int argc, char* argv[]) {
    // Whether this invocation is the --unit-tests gate. Suppresses the
    // mimalloc banner below so the console-summary contract documented
    // in `docs/_meta/testing.md` ("one line on success: N/N Unit tests
    // passed") holds. The gate still loads mimalloc via the static
    // link; only the version printout is silenced.
    const bool isUnitTests =
        (argc >= 2 && std::strcmp(argv[1], "--unit-tests") == 0);

#ifdef USE_MIMALLOC
    int v = mi_version();  // ensure mimalloc override DLL is loaded
    if (!isUnitTests) {
        std::cout << "mimalloc version: " << v << std::endl;
    }
#endif
    auto start = std::chrono::high_resolution_clock::now();

    // Note: .debug/hashburst_trace.txt is NOT truncated here. It is
    // truncated inside performElementaryLogicalStep on the first burst
    // where the target-LB guard fires (prover.cpp:2384,
    // burstCount == 1 ? std::ios::trunc : std::ios::app). That keeps the
    // trace from being wiped by sibling gl_quick.exe invocations in the
    // same main.py session whose batches do not match the target LB —
    // e.g. the IncubatorGauss1 trace would otherwise be erased by the
    // following Gauss-main batch.

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

    // --unit-tests: run the in-tree unit-test harness and exit. main.py
    // invokes this before run_modes.full_run() so a regression aborts the
    // pipeline before any pipeline work or proof run. Direct invocations
    // such as `gl_quick.exe Peano` do NOT trigger this gate; the harness
    // is opt-in via flag for them.
    //
    // Working-directory hardening: Linux/macOS users `make` from
    // `GL_Quick_VS/GL_Quick/` and naturally invoke `./gl_quick --unit-tests`
    // from there. The ExpressionAnalyzer constructor's config search is
    // anchored on the current working directory; running from the build
    // dir would fail to locate `files/config/Config*.json` and abort with
    // a missing-anchor assert. Resolve project root from argv[0] (the
    // binary lives at `<project>/GL_Quick_VS/GL_Quick/gl_quick(.exe)`,
    // three parent_path() walks up to the project root) and chdir there
    // before running tests. This matches the path resolution used by
    // the --mirror-externals branch above.
    if (argc >= 2 && std::strcmp(argv[1], "--unit-tests") == 0) {
        namespace fs = std::filesystem;
        std::error_code ec;
        const fs::path projectRoot =
            fs::path(argv[0]).parent_path().parent_path().parent_path();
        if (!projectRoot.empty()) {
            fs::current_path(projectRoot, ec);
            if (ec) {
                std::cerr << "[unit-tests] warning: chdir to project root '"
                          << projectRoot.string() << "' failed: "
                          << ec.message() << "\n";
            }
        }
        return gl::tests::runAllTests();
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
