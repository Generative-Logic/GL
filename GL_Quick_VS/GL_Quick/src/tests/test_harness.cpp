/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025-2026 Generative Logic UG(haftungsbeschränkt)

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

/// @file
/// @brief Registry storage and `runAllTests` driver for the unit-test harness.
///
/// @details
/// The registry is a Meyers singleton — local-static `std::vector<TestEntry>`
/// inside `registry()`. C++ guarantees first-call construction is
/// thread-safe and that the storage outlives the static initializers
/// that push into it. That guarantee matters here because `TEST(...)`
/// expands to a static initializer at namespace scope; using a
/// file-scope global would risk static-initialization-order fiasco
/// when a test object is constructed in a different translation unit
/// than the registry.
///
/// `runAllTests()` is invoked by `main.cpp` when `argv[1] == "--unit-tests"`.
/// It iterates the registry in registration order, catches the
/// dedicated `FailExc`, and aborts on the first failure. Per-test
/// elapsed time is printed so a regression names itself.

#include "test_harness.hpp"

#include "../memory_infra/global_memory_manager.hpp"
#include "../memory_infra/scratch_arena.hpp"
#include "../parameters.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <streambuf>
#include <string>
#include <system_error>
#include <thread>

namespace gl {
namespace tests {

    namespace {

        /// Sentinel thrown by `fail()` and caught by `runAllTests()`.
        /// Local to this translation unit so test bodies cannot catch
        /// it accidentally with `catch (...)`.
        struct FailExc {};

        /// Meyers singleton — function-local static avoids any cross-TU
        /// static-initialization-order issue with the `TEST(...)` macro
        /// register-on-init pattern.
        std::vector<TestEntry>& registry() {
            static std::vector<TestEntry> r;
            return r;
        }

    }  // anonymous namespace

    void registerTest(const char* suite, const char* name, TestFn fn) {
        registry().push_back({suite, name, fn});
    }

    [[noreturn]] void fail(const char* file, int line, const char* expr) {
        std::fprintf(stderr, "[ASSERT] %s:%d  %s\n", file, line, expr);
        std::fflush(stderr);
        throw FailExc{};
    }

    int runAllTests() {
        namespace fs = std::filesystem;
        const std::vector<TestEntry>& reg = registry();

        // Statification: tests that grow `Memory`'s statified containers
        // need the process-wide pool. Init with the ProverParameters
        // DEFAULT triple — the sole source of truth now that the sizing is
        // config-independent (D-139), so a test
        // constructing an ExpressionAnalyzer (whose ctor re-inits from the
        // same struct defaults) hits the idempotent-same-config path, never
        // the mismatch assert.
        {
            const ProverParameters defaults;
            initStaticMemory(StaticMemoryConfig{
                defaults.static_pool_bytes,
                defaults.static_block_bytes,
                defaults.static_page_bytes });
            // The persistent (second) pool backs Memory::intToBeProved; every
            // test that constructs a Memory and inserts a goal touches it, so
            // it must be initialized here too (PoolKind::Persistent).
            initPersistentMemory(StaticMemoryConfig{
                defaults.static_persistent_pool_bytes,
                defaults.static_persistent_block_bytes,
                defaults.static_page_bytes,
                PoolKind::Persistent });
            // The mail (third) pool backs the cross-LB pull-model mail log;
            // every test that constructs an ExpressionAnalyzer touches it, so
            // it must be initialized here too (PoolKind::Mail).
            initMailMemory(StaticMemoryConfig{
                defaults.static_mail_pool_bytes,
                defaults.static_mail_block_bytes,
                defaults.static_page_bytes,
                PoolKind::Mail });
            // The LB-body (fourth) pool backs the LB object store; every test
            // that constructs an ExpressionAnalyzer touches it (its ctor inits
            // the pool), so pre-init here too (PoolKind::Lb) to keep that ctor on
            // the idempotent-same-config path.
            initLbMemory(StaticMemoryConfig{
                defaults.static_lb_pool_bytes,
                defaults.static_lb_block_bytes,
                defaults.static_page_bytes,
                PoolKind::Lb });
            // The two scratch-arena registries (string-tier + request-gen) are
            // normally initialized by the ExpressionAnalyzer constructor; a test
            // that never constructs one — e.g. the standalone NameMap rig, whose
            // statified encodePush now builds its canonical scope name on the
            // string-scratch arena — still needs them. Pre-init here with the
            // SAME slot counts the constructor uses (logicalCores + 1 /
            // logicalCores) so a later ExpressionAnalyzer ctor stays on the
            // idempotent-same-shape path.
            const unsigned logicalCores =
                std::max(1u, std::thread::hardware_concurrency());
            initScratchArenas(logicalCores + 1);
            initGenScratchArenas(logicalCores);
        }

        // Per-test detail goes to `.debug/unit_tests.log` so the console
        // shows only the final summary (one line on success) plus any
        // failure information on miss. Cwd is the project root via
        // main.cpp's --unit-tests dispatch (chdir to projectRoot derived
        // from argv[0]), so the relative path lands inside .debug/.
        std::error_code ec;
        fs::create_directories(".debug", ec);
        std::ofstream log(".debug/unit_tests.log",
                          std::ios::out | std::ios::trunc);
        if (log) {
            log << "[unit-tests] " << reg.size() << " tests registered\n";
            log.flush();
        }

        // Redirect std::cout and std::clog into the log file for the
        // duration of the test run. Tests construct heavy objects
        // (notably ExpressionAnalyzer) whose initialisation chatter
        // would otherwise flood the console. std::cerr stays attached
        // to the real stderr so genuine errors thrown out of a test
        // body are visible. fail()'s assertion-line print goes through
        // stderr for the same reason.
        std::streambuf* const oldCoutBuf =
            log ? std::cout.rdbuf(log.rdbuf()) : nullptr;
        std::streambuf* const oldClogBuf =
            log ? std::clog.rdbuf(log.rdbuf()) : nullptr;

        std::size_t passed = 0;
        const auto wallStart = std::chrono::steady_clock::now();

        for (const TestEntry& t : reg) {
            const auto testStart = std::chrono::steady_clock::now();
            bool failed = false;
            const char* failKind = nullptr;
            std::string failWhat;

            // Pre-test flushed START marker so we can identify the
            // aborting test if a non-throwing failure (assert(), SIGABRT,
            // SIGSEGV) kills the harness — buffered [PASS] lines for the
            // last completed test would otherwise be lost in the abort.
            if (log) {
                log << "[START] " << t.suite << "." << t.name << "\n";
                log.flush();
            }

            try {
                t.fn();
            } catch (const FailExc&) {
                failed = true;
                failKind = "assert";
            } catch (const std::exception& ex) {
                failed = true;
                failKind = "std::exception";
                failWhat = ex.what();
            } catch (...) {
                failed = true;
                failKind = "unknown";
            }

            const auto testEnd = std::chrono::steady_clock::now();
            const long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     testEnd - testStart).count();

            if (failed) {
                // Failure mirrors to BOTH the console (via the C-level
                // stdout, which is independent of the std::cout rdbuf
                // swap above) and the log so the regression names
                // itself even when the run is otherwise silent.
                std::string suffix;
                if (!failWhat.empty()) {
                    suffix = std::string(": ") + failWhat;
                }
                std::printf("[FAIL] %s.%s  (%lld ms)  [%s%s]\n",
                            t.suite, t.name, ms, failKind, suffix.c_str());
                std::printf("[unit-tests] aborting on first failure\n");
                std::fflush(stdout);
                if (log) {
                    log << "[FAIL] " << t.suite << "." << t.name
                        << "  (" << ms << " ms)  ["
                        << failKind << suffix << "]\n";
                    log << "[unit-tests] aborting on first failure\n";
                    log.close();
                }
                if (oldCoutBuf) std::cout.rdbuf(oldCoutBuf);
                if (oldClogBuf) std::clog.rdbuf(oldClogBuf);
                return 1;
            }

            // Pass: log-only. The log file keeps the per-test detail
            // available for offline diff.
            if (log) {
                log << "[PASS] " << t.suite << "." << t.name
                    << "  (" << ms << " ms)\n";
            }
            ++passed;
        }

        const auto wallEnd = std::chrono::steady_clock::now();
        const long long totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      wallEnd - wallStart).count();
        if (log) {
            log << "[unit-tests] " << passed << "/" << reg.size()
                << " passed in " << totalMs << " ms\n";
            log.close();
        }

        // Restore std::cout / std::clog so the summary lands on the
        // real console rather than back in the log.
        if (oldCoutBuf) std::cout.rdbuf(oldCoutBuf);
        if (oldClogBuf) std::clog.rdbuf(oldClogBuf);

        // Single console summary on success.
        std::printf("%zu/%zu Unit tests passed\n", passed, reg.size());
        std::fflush(stdout);
        return 0;
    }

}  // namespace tests
}  // namespace gl
