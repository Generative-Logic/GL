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
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <ctime>
#include <exception>
#include <mutex>
#include <string>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <cassert>
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX  // keep windows.h from clobbering std::min / std::max
#  endif
#  include <windows.h>
#  include <crtdbg.h>
#  include <dbghelp.h>
#  pragma comment(lib, "dbghelp.lib")
#endif
#ifdef USE_MIMALLOC
#include <mimalloc.h>
#endif
#include "project_paths.hpp"
#include "run_modes.hpp"
#include "conjecturer.hpp"
#include "prover.hpp"
#include "infra/phase2_trace.hpp"
#include "infra/rt_tracker.hpp"
#include "infra/mem_tracker.hpp"
#include "infra/heap_profile.hpp"
#include "memory_infra/global_memory_manager.hpp"
#include "memory_infra/heap_allocator.hpp"
#include "tests/test_harness.hpp"

namespace {

#ifdef _WIN32
// Enable explicitly when a full process image is needed for crash diagnosis.
constexpr bool kWriteAbortFullMemoryDump = false;

/// @brief Write a full-memory minidump of the current process to
///        `.debug/crashdumps/` on `SIGABRT` (an assert firing).
///
/// @details
/// The LocalDumps registry policy captures dumps only for crashes that
/// traverse Windows Error Reporting; the release CRT's `assert -> abort`
/// terminates without reaching that hook (no `.dmp` despite
/// `_CALL_REPORTFAULT`). This handler runs ON the aborting thread, so
/// `MiniDumpWriteDump` records the exact faulting stack. It is
/// re-entrancy-guarded, best-effort by design (a diagnostic writer must
/// never mask the abort — the process still dies with the original
/// signal). The handler is installed only when
/// `kWriteAbortFullMemoryDump` is enabled.
///
/// @param signalNumber The signal that fired (always `SIGABRT` here;
///                     required by the `signal` handler signature).
void writeAbortMinidump(int signalNumber) {
    (void)signalNumber;
    static volatile long alreadyDumping = 0;
    if (InterlockedExchange(&alreadyDumping, 1) != 0) return;
    // Flush every open stream first so buffered diagnostics (Rule-31 trap
    // files, the run logs) survive the abort.
    std::fflush(nullptr);
    CreateDirectoryA(".debug", nullptr);
    CreateDirectoryA(".debug\\crashdumps", nullptr);
    char path[MAX_PATH];
    std::snprintf(path, sizeof(path),
                  ".debug\\crashdumps\\gl_quick_abort_%lu.dmp",
                  static_cast<unsigned long>(GetCurrentProcessId()));
    const HANDLE file = CreateFileA(path, GENERIC_WRITE, 0, nullptr,
                                    CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL,
                                    nullptr);
    if (file != INVALID_HANDLE_VALUE) {
        MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), file,
                          MiniDumpWithFullMemory, nullptr, nullptr, nullptr);
        CloseHandle(file);
    }
}
#endif

} // namespace

int main(int argc, char* argv[]) {
    // Before the first large allocation: arenas that place blocks above
    // 64 MiB, so freed blocks go back to the operating system
    // (D-427).
    gl::configureHeapAllocator();
#ifdef _WIN32
    // Assertion failures terminate without a full process dump unless the
    // diagnostic switch is explicitly enabled. Suppress the abort dialog in
    // both modes.
    _set_abort_behavior(
        kWriteAbortFullMemoryDump ? _CALL_REPORTFAULT : 0,
        _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    if (kWriteAbortFullMemoryDump) {
        std::signal(SIGABRT, writeAbortMinidump);
    }
#endif
    // Whether this invocation is the --unit-tests gate. Suppresses the
    // mimalloc banner below so the console-summary contract documented
    // in `docs/agentic_swdd/_meta/testing.md` ("one line on success: N/N Unit tests
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

#if MEM_MEASUREMENT
    // Opt-in allocation-site sampling for the heap census: GL_HEAP_PROFILE_KIB
    // is the sampling interval in KiB (see docs/agentic_swdd/_meta/
    // memory_measurement.md). Unset: live-heap counting only.
    {
#ifdef _WIN32
        char text[32]{};
        std::size_t required = 0;
        const errno_t status = getenv_s(&required, text, sizeof(text), "GL_HEAP_PROFILE_KIB");
        assert(status == 0 && "GL_HEAP_PROFILE_KIB must be a short integer");
        const char* value = required == 0 ? nullptr : text;
#else
        const char* value = std::getenv("GL_HEAP_PROFILE_KIB");
#endif
        // The unit-test gate enables sampling inside its own heap-profile test.
        if (value != nullptr && !isUnitTests) {
            char* end = nullptr;
            const long long kib = std::strtoll(value, &end, 10);
            assert(end != value && *end == '\0' && kib > 0
                && "GL_HEAP_PROFILE_KIB must be a positive integer");
            gl::heap_profile::enable(static_cast<int64_t>(kib) * 1024);
        }
    }
#endif
    auto start = std::chrono::high_resolution_clock::now();

    // Note: .debug/hashburst_trace.txt is NOT truncated here. It is
    // truncated inside the ENTRY hashburst-dump trap
    // (`hashburst_dump::dumpEntry`, reached from `performElemPhase1`) on the
    // first ENTRY for the target LB (entryCount == 1 ? std::ios::trunc :
    // std::ios::app). That keeps the
    // trace from being wiped by sibling gl_quick.exe invocations in the
    // same main.py session whose batches do not match the target LB —
    // e.g. the IncubatorGauss1 trace would otherwise be erased by the
    // following Gauss-main batch.

    // --mirror-externals <theoremsDir>: rebuild compressed_external_theorems.txt
    if (argc >= 3 && std::strcmp(argv[1], "--mirror-externals") == 0) {
        namespace fs = std::filesystem;
        const fs::path& projectRoot = gl::projectRoot();
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

    // --ce-only <anchorId>: run only the counterexample filter on the
    // existing conjectures.txt, write filtered_conjectures.txt, exit. No main
    // prover, no compression, no proof graph. Fast iteration loop used to
    // tighten the Peano CE filter — see
    // `peano_filter_design_decisions.md` D-rtf-1.
    if (argc >= 3 && std::strcmp(argv[1], "--ce-only") == 0) {
        std::string anchorId = argv[2];
        run_modes::ceOnlyRun(anchorId);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "CE-only runtime: "
            << std::chrono::duration<double>(end - start).count() << " seconds\n";
        return 0;
    }

    // --unit-tests: run the in-tree unit-test harness and exit. main.py
    // invokes this before run_modes.full_run() so a regression aborts the
    // pipeline before any pipeline work or proof run. Direct invocations
    // such as `gl_quick.exe Peano` do NOT trigger this gate; the harness
    // is opt-in via flag for them.
    //
    // `--unit-tests` alone or with `--phase2-backend cuda` runs every
    // compiled test. `--phase2-backend cpu` is the processor-route gate
    // main.py runs without `--GPU`: it leaves out the CUDA-route tests, so a
    // processor run needs no CUDA device. Anything else after the flag is a
    // caller bug.
    //
    // Unit fixtures use relative paths inside the deployed project tree.
    if (isUnitTests) {
        bool includeCudaRoute = true;
        if (argc > 2) {
            assert(argc == 4 && std::strcmp(argv[2], "--phase2-backend") == 0
                   && "--unit-tests accepts only --phase2-backend cpu|cuda");
            const bool cpu = std::strcmp(argv[3], "cpu") == 0;
            assert((cpu || std::strcmp(argv[3], "cuda") == 0)
                   && "--phase2-backend requires cpu or cuda");
            includeCudaRoute = !cpu;
        }
        std::error_code ec;
        std::filesystem::current_path(gl::projectRoot(), ec);
        assert(!ec && "chdir to runtime project root failed");
        return gl::tests::runAllTests(includeCudaRoute);
    }

    // Grab anchor id if provided (e.g., "Peano" or "Gauss")
    // Default to IncubatorPeano for MSVS F5 debugging; full_run passes "Peano"/"Gauss" via argv
    std::string anchor_id = (argc > 1) ? std::string(argv[1]) : "Gauss";
    gl::Phase2TraceScope phase2BatchTrace(anchor_id.c_str());

    // Optional batch-path file overrides (shortcut mode):
    //   gl_quick.exe <Tag> [--conjectures-file <path>] [--externals-file <path>]
    // --conjectures-file replaces the conjecture-list file, --externals-file
    // the file external theorems are loaded from. Relative paths resolve
    // against the project root inside fullRun. Anything else after the tag
    // is a caller bug and asserts.
    std::string conjecturesFile;
    std::string externalsFile;
    std::string phase2Backend;
    bool shortcutMode = false;
    bool phase2BackendSeen = false;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shortcut") == 0) {
            assert(!shortcutMode && "--shortcut must be supplied at most once");
            shortcutMode = true;
            continue;
        }
        const bool isConjectures =
            std::strcmp(argv[i], "--conjectures-file") == 0;
        const bool isExternals =
            std::strcmp(argv[i], "--externals-file") == 0;
        const bool isPhase2Backend =
            std::strcmp(argv[i], "--phase2-backend") == 0;
        assert((isConjectures || isExternals || isPhase2Backend)
               && "unknown argument after the batch tag");
        assert(i + 1 < argc && "batch-path flag missing its <path> value");
        if (isConjectures) {
            conjecturesFile = argv[++i];
        } else if (isExternals) {
            externalsFile = argv[++i];
        } else {
            assert(!phase2BackendSeen
                   && "--phase2-backend must be supplied at most once");
            phase2BackendSeen = true;
            phase2Backend = argv[++i];
            assert((phase2Backend == "cpu" || phase2Backend == "cuda")
                   && "--phase2-backend requires cpu or cuda");
        }
    }

    // run_modes::quickRun();
    run_modes::fullRun(
        anchor_id, conjecturesFile, externalsFile, phase2Backend, shortcutMode);

#if RT_MEASUREMENT
    // Cross-burst RT aggregate for this batch process (one batch per
    // gl_quick invocation). Trigger-independent — every burst's sections
    // are folded in at tracker destruction; this writes the summary once
    // at end of batch.
    gl::rt_tracker::dumpRtAggregate(".rt/_aggregate_" + anchor_id + ".log");
#endif

#if MEM_MEASUREMENT
    // Per-structure memory table for this batch process. The peak snapshot is
    // whichever iteration held the most attributed bytes; percentages are that
    // structure's share of that one instant.
    gl::mem_tracker::dumpMemAggregate(".rt/_memory_" + anchor_id + ".log");
    gl::mem_tracker::dumpMemHtml(".rt/_memory_" + anchor_id + ".html");
#endif

    // Return memory to OS (mimalloc retains free pages by default)
#ifdef USE_MIMALLOC
    mi_collect(true);
#endif

    auto end = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(end - start).count();
    std::cout << "Runtime of the executable (counter example filter + prover): "
        << secs << " seconds" << std::endl;

    // Two-level page-directory spill telemetry for this batch (observability;
    // a forced-small-page run reports a non-zero promotion count).
    gl::staticMemory().reportPageStats();
    return 0;
}
