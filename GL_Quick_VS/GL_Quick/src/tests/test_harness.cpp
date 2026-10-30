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
/// @brief Registry storage, the `runAllTests` driver and the parallel worker
/// split of the unit-test harness.
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
/// The suite runs in parallel worker processes. The gate process gives each
/// worker its own project tree under `.debug/unit_test_workers/w<k>/` (hard
/// links of the executable, its DLLs, `files/config` and `files/definitions`),
/// so relative and project-root writers of different workers never meet;
/// starts `gl_quick --unit-tests` there concurrently, each with its own
/// environment carrying `GL_UNIT_TEST_WORKER=<k>/<K>`;
/// waits for all workers and merges their logs in registration order. The
/// split is fixed: in the CUDA-route gate the CUDA-route tests run on worker 0
/// and every other test round-robin on workers 1..K-1 in registration order;
/// the processor-route gate deals all tests round-robin. A worker iterates its
/// share, catches the dedicated `FailExc`, and aborts on its first failure.
/// A single selected test (`GL_UNIT_TEST_ONLY`), `GL_UNIT_TEST_WORKERS=1` and
/// a requested external peak capture (`GL_PEAK_TEST_CAPTURE`, which calls a
/// tool of the live project tree) run in the gate process itself. Per-test
/// elapsed time is logged so a regression names itself. The processor-route
/// gate leaves out the `CUDA_TEST` entries and names each of them in the log
/// as not selected.

#include "test_harness.hpp"

#include "../memory_infra/global_memory_manager.hpp"
#include "../memory_infra/scratch_arena.hpp"
#include "../memory.hpp"
#include "../parameters.hpp"
#include "../project_paths.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <streambuf>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <spawn.h>
#include <sys/wait.h>
extern char** environ;
#endif

namespace gl {
namespace tests {

    namespace {

        /// Sentinel thrown by `fail()` and caught by `runAllTests()`.
        /// Local to this translation unit so test bodies cannot catch
        /// it accidentally with `catch (...)`.
        struct FailExc {};
        bool selectedCudaRoute = true;

        /// Meyers singleton — function-local static avoids any cross-TU
        /// static-initialization-order issue with the `TEST(...)` macro
        /// register-on-init pattern.
        std::vector<TestEntry>& registry() {
            static std::vector<TestEntry> r;
            return r;
        }

        /// Directory of the per-worker project trees, below the live root.
        constexpr const char* kWorkerTrees = ".debug/unit_test_workers";

#ifdef _WIN32
        constexpr const char* kExecutableName = "gl_quick.exe";
        using WorkerProcess = HANDLE;
#else
        constexpr const char* kExecutableName = "gl_quick";
        using WorkerProcess = pid_t;
#endif

        std::string qualifiedName(const TestEntry& test) {
            return std::string(test.suite) + "." + test.name;
        }

        /// Worker count of the gate: `GL_UNIT_TEST_WORKERS` when set,
        /// otherwise the machine default.
        unsigned configuredWorkerCount() {
            const auto requested = testEnvironment("GL_UNIT_TEST_WORKERS");
            if (!requested) return defaultWorkerCount(std::thread::hardware_concurrency());
            const unsigned long count = std::stoul(*requested);
            assert(count >= 1 && count <= 256 && "GL_UNIT_TEST_WORKERS must be 1..256");
            return static_cast<unsigned>(count);
        }

        /// The `<k>/<K>` slot of a worker process.
        std::pair<unsigned, unsigned> parseWorkerSlot(const std::string& text) {
            const std::size_t slash = text.find('/');
            assert(slash != std::string::npos && "GL_UNIT_TEST_WORKER must read <k>/<K>");
            const unsigned worker = static_cast<unsigned>(std::stoul(text.substr(0, slash)));
            const unsigned count = static_cast<unsigned>(std::stoul(text.substr(slash + 1)));
            assert(count >= 2 && worker < count && "GL_UNIT_TEST_WORKER slot out of range");
            return { worker, count };
        }

        /// One worker's project tree: hard links of the executable, its DLLs
        /// and the configuration and definition inputs, plus an empty
        /// `files/GL_binaries`.
        void prepareWorkerTree(const std::filesystem::path& liveRoot,
                               const std::filesystem::path& tree) {
            namespace fs = std::filesystem;
            const fs::path liveBinaries = liveRoot / "GL_Quick_VS" / "GL_Quick";
            const fs::path binaries = tree / "GL_Quick_VS" / "GL_Quick";
            fs::create_directories(binaries);
            fs::create_hard_link(liveBinaries / kExecutableName, binaries / kExecutableName);
            for (const auto& entry : fs::directory_iterator(liveBinaries)) {
                if (entry.path().extension() == ".dll")
                    fs::create_hard_link(entry.path(), binaries / entry.path().filename());
            }
            fs::create_directories(tree / "files" / "GL_binaries");
            const auto links = fs::copy_options::recursive | fs::copy_options::create_hard_links;
            fs::copy(liveRoot / "files" / "config", tree / "files" / "config", links);
            fs::copy(liveRoot / "files" / "definitions", tree / "files" / "definitions", links);
        }

#ifdef _WIN32
        using WorkerEnvironment = std::wstring;
#else
        using WorkerEnvironment = std::vector<std::string>;
#endif

        /// The environment of one worker: this process's environment without
        /// a worker slot, followed by `GL_UNIT_TEST_WORKER=<k>/<K>`. Each
        /// worker gets its own copy, so the workers can start concurrently.
        WorkerEnvironment workerEnvironment(unsigned worker, unsigned count) {
            const std::string slot = "GL_UNIT_TEST_WORKER=" + std::to_string(worker) + "/"
                + std::to_string(count);
#ifdef _WIN32
            wchar_t* const strings = GetEnvironmentStringsW();
            assert(strings != nullptr && "the process environment must be readable");
            std::wstring block;
            for (const wchar_t* entry = strings; *entry; entry += std::wcslen(entry) + 1) {
                if (std::wcsncmp(entry, L"GL_UNIT_TEST_WORKER=", 20) == 0) continue;
                block.append(entry);
                block.push_back(L'\0');
            }
            FreeEnvironmentStringsW(strings);
            block.append(slot.begin(), slot.end());
            block.push_back(L'\0');
            block.push_back(L'\0');
            return block;
#else
            std::vector<std::string> entries;
            for (char** entry = environ; *entry; ++entry) {
                if (std::strncmp(*entry, "GL_UNIT_TEST_WORKER=", 20) == 0) continue;
                entries.emplace_back(*entry);
            }
            entries.push_back(slot);
            return entries;
#endif
        }

        /// Start one worker process of the gate with its own environment; it
        /// inherits this process's standard handles and console. Safe to call
        /// from several threads at once.
        WorkerProcess startWorker(const std::filesystem::path& executable,
                                  bool includeCudaRoute, WorkerEnvironment& environment) {
            const char* backend = includeCudaRoute ? "cuda" : "cpu";
#ifdef _WIN32
            const std::wstring application = executable.wstring();
            std::wstring commandLine = L"\"" + application + L"\" --unit-tests --phase2-backend "
                + std::wstring(backend, backend + std::char_traits<char>::length(backend));
            STARTUPINFOW startup{};
            startup.cb = sizeof(startup);
            startup.dwFlags = STARTF_USESTDHANDLES;
            startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
            startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
            startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
            PROCESS_INFORMATION process{};
            const BOOL started = CreateProcessW(application.c_str(), commandLine.data(), nullptr,
                nullptr, TRUE, CREATE_UNICODE_ENVIRONMENT, environment.data(), nullptr,
                &startup, &process);
            assert(started && "unit-test worker must start");
            CloseHandle(process.hThread);
            return process.hProcess;
#else
            std::string binary = executable.string();
            std::string flag = "--unit-tests";
            std::string backendFlag = "--phase2-backend";
            std::string backendName = backend;
            char* arguments[] = { binary.data(), flag.data(), backendFlag.data(), backendName.data(), nullptr };
            std::vector<char*> envp;
            for (std::string& entry : environment) envp.push_back(entry.data());
            envp.push_back(nullptr);
            pid_t process = 0;
            const int started = posix_spawn(&process, binary.c_str(), nullptr, nullptr, arguments, envp.data());
            assert(started == 0 && "unit-test worker must start");
            return process;
#endif
        }

        /// Wait for one worker process; its exit code.
        int waitWorker(WorkerProcess process) {
#ifdef _WIN32
            const DWORD waited = WaitForSingleObject(process, INFINITE);
            assert(waited == WAIT_OBJECT_0 && "unit-test worker wait failed");
            DWORD exitCode = 0;
            const BOOL read = GetExitCodeProcess(process, &exitCode);
            assert(read && "unit-test worker exit code must be readable");
            CloseHandle(process);
            return static_cast<int>(exitCode);
#else
            int status = 0;
            const pid_t waited = waitpid(process, &status, 0);
            assert(waited == process && "unit-test worker wait failed");
            if (WIFEXITED(status)) return WEXITSTATUS(status);
            return 128 + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
#endif
        }

        /// The per-test blocks of one worker log, keyed by test name: every
        /// line from a test's `[START]` line up to the next test or summary.
        std::map<std::string, std::string> readWorkerBlocks(const std::filesystem::path& log) {
            std::map<std::string, std::string> blocks;
            std::ifstream in(log);
            assert(in.is_open() && "unit-test worker log must exist");
            std::string line;
            std::string current;
            while (std::getline(in, line)) {
                if (line.rfind("[START] ", 0) == 0) {
                    current = line.substr(8);
                    blocks[current] += line + "\n";
                } else if (line.rfind("[unit-tests] ", 0) == 0) {
                    current.clear();
                } else if (!current.empty()) {
                    blocks[current] += line + "\n";
                }
            }
            return blocks;
        }

        /// Whether a log block holds a line starting with @p prefix.
        bool blockHasLine(const std::string& block, const std::string& prefix) {
            return block.rfind(prefix, 0) == 0 || block.find("\n" + prefix) != std::string::npos;
        }

        /// The console summary of a passing gate.
        void printPassSummary(std::size_t passed, std::size_t selected, std::size_t notSelected,
                              bool single, unsigned workers) {
            const std::string where = workers > 1
                ? " in " + std::to_string(workers) + " worker processes" : std::string();
            if (notSelected == 0) {
                std::printf("%zu/%zu Unit tests passed%s\n", passed, selected, where.c_str());
            } else if (single) {
                std::printf("%zu/%zu Unit tests passed (filtered selection; %zu other tests not selected)\n",
                            passed, selected, notSelected);
            } else {
                std::printf("%zu/%zu Unit tests passed%s (processor route; %zu CUDA-route tests not selected)\n",
                            passed, selected, where.c_str(), notSelected);
            }
            std::fflush(stdout);
        }

        /// The gate process: run the selected tests in @p workers worker
        /// processes, each in its own project tree, and merge their logs.
        int runInWorkers(const std::vector<const TestEntry*>& selectedTests,
                         std::size_t notSelected, bool includeCudaRoute, unsigned workers) {
            namespace fs = std::filesystem;
            const auto wallStart = std::chrono::steady_clock::now();
            const fs::path liveRoot = projectRoot();
            const fs::path treeRoot = liveRoot / kWorkerTrees;
            fs::remove_all(treeRoot);
            std::vector<fs::path> trees;
            for (unsigned worker = 0; worker < workers; ++worker) {
                trees.push_back(treeRoot / ("w" + std::to_string(worker)));
                prepareWorkerTree(liveRoot, trees.back());
            }
            // Start the workers concurrently: one process start costs about
            // 0.3 s on the development host, so sequential starts would delay
            // the last worker by seconds.
            std::vector<WorkerEnvironment> environments;
            for (unsigned worker = 0; worker < workers; ++worker)
                environments.push_back(workerEnvironment(worker, workers));
            std::vector<WorkerProcess> processes(workers);
            std::vector<std::thread> starters;
            for (unsigned worker = 0; worker < workers; ++worker) {
                starters.emplace_back([&, worker] {
                    processes[worker] = startWorker(
                        trees[worker] / "GL_Quick_VS" / "GL_Quick" / kExecutableName,
                        includeCudaRoute, environments[worker]);
                });
            }
            for (std::thread& starter : starters) starter.join();
            std::vector<int> exitCodes;
            for (WorkerProcess process : processes) exitCodes.push_back(waitWorker(process));

            std::map<std::string, std::string> blocks;
            for (const fs::path& tree : trees) {
                for (auto& [name, block] : readWorkerBlocks(tree / ".debug" / "unit_tests.log"))
                    blocks.emplace(name, std::move(block));
            }
            fs::create_directories(".debug");
            std::ofstream log(".debug/unit_tests.log", std::ios::out | std::ios::trunc);
            assert(log.is_open() && "unit-test log must open");
            log << "[unit-tests] " << registry().size() << " tests registered, "
                << selectedTests.size() << " selected ("
                << (includeCudaRoute ? "CUDA-route gate" : "processor-route gate")
                << ", " << workers << " worker processes)\n";
            std::size_t passed = 0;
            std::vector<std::string> failures;
            for (const TestEntry& test : registry()) {
                if (test.cudaRoute && !includeCudaRoute)
                    log << "[NOT SELECTED] " << qualifiedName(test) << "  (CUDA route)\n";
            }
            std::size_t nonCudaOrdinal = 0;
            for (const TestEntry* test : selectedTests) {
                const std::string name = qualifiedName(*test);
                const unsigned worker = assignedWorker(test->cudaRoute, nonCudaOrdinal, workers, includeCudaRoute);
                if (!test->cudaRoute) ++nonCudaOrdinal;
                const auto it = blocks.find(name);
                if (it == blocks.end()) continue;
                log << it->second;
                if (blockHasLine(it->second, "[PASS] ")) {
                    ++passed;
                } else if (!blockHasLine(it->second, "[FAIL] ")) {
                    failures.push_back(name + " (worker " + std::to_string(worker)
                        + " ended inside this test, exit code "
                        + std::to_string(exitCodes[worker]) + ")");
                } else {
                    failures.push_back(name + " (worker " + std::to_string(worker) + ")");
                }
            }
            const long long totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - wallStart).count();
            log << "[unit-tests] " << passed << "/" << selectedTests.size()
                << " passed in " << totalMs << " ms (" << workers << " worker processes)\n";
            log.close();

            const bool allExited = std::all_of(exitCodes.begin(), exitCodes.end(),
                                               [](int code) { return code == 0; });
            if (allExited && failures.empty()) {
                assert(passed == selectedTests.size()
                    && "every worker passed, so every selected test has a passing block");
                printPassSummary(passed, selectedTests.size(), notSelected, false, workers);
                return 0;
            }
            for (const std::string& failure : failures)
                std::printf("[FAIL] %s\n", failure.c_str());
            for (unsigned worker = 0; worker < workers; ++worker) {
                if (exitCodes[worker] != 0)
                    std::printf("[unit-tests] worker %u exited with code %d (log %s)\n", worker,
                                exitCodes[worker], (trees[worker] / ".debug" / "unit_tests.log").string().c_str());
            }
            std::printf("[unit-tests] %zu/%zu passed; %zu not run after a worker failure\n",
                        passed, selectedTests.size(),
                        selectedTests.size() - passed - failures.size());
            std::fflush(stdout);
            return 1;
        }

    }  // anonymous namespace

    void registerTest(const char* suite, const char* name, TestFn fn, bool cudaRoute) {
        registry().push_back({suite, name, fn, cudaRoute});
    }

    [[noreturn]] void fail(const char* file, int line, const char* expr) {
        std::fprintf(stderr, "[ASSERT] %s:%d  %s\n", file, line, expr);
        std::fflush(stderr);
        throw FailExc{};
    }

    std::optional<std::string> testEnvironment(const char* name) {
#ifdef _WIN32
        char* value = nullptr;
        std::size_t length = 0;
        ASSERT_EQ(_dupenv_s(&value, &length, name), 0);
        std::optional<std::string> result;
        if (value != nullptr) {
            result = value;
            std::free(value);
        }
        return result;
#else
        const char* value = std::getenv(name);
        return value == nullptr ? std::nullopt : std::optional<std::string>(value);
#endif
    }

    bool includeCudaRouteForChildTests() { return selectedCudaRoute; }

    unsigned assignedWorker(bool cudaRoute, std::size_t nonCudaOrdinal, unsigned workerCount,
                            bool cudaWorker) {
        assert(workerCount >= 1 && "a gate has at least one worker");
        assert((cudaWorker || !cudaRoute) && "a CUDA-route test runs only in the CUDA-route gate");
        if (workerCount == 1) return 0;
        if (!cudaWorker) return static_cast<unsigned>(nonCudaOrdinal % workerCount);
        if (cudaRoute) return 0;
        return 1 + static_cast<unsigned>(nonCudaOrdinal % (workerCount - 1));
    }

    unsigned defaultWorkerCount(unsigned hardwareThreads) {
        return (std::max)(1u, (std::min)(12u, hardwareThreads / 2));
    }

    int runAllTests(bool includeCudaRoute) {
        selectedCudaRoute = includeCudaRoute;
        namespace fs = std::filesystem;
        const std::vector<TestEntry>& reg = registry();
        const auto only = testEnvironment("GL_UNIT_TEST_ONLY");
        // A single selected test runs in this process even inside a worker
        // (the isolated export child inherits its worker's environment).
        const auto workerSlot = only ? std::nullopt : testEnvironment("GL_UNIT_TEST_WORKER");
        std::vector<const TestEntry*> selectedTests;
        for (const TestEntry& t : reg) {
            if ((!only || qualifiedName(t) == *only) && (includeCudaRoute || !t.cudaRoute))
                selectedTests.push_back(&t);
        }
        ASSERT_TRUE(!selectedTests.empty());
        const std::size_t notSelected = reg.size() - selectedTests.size();

        if (!only && !workerSlot && !testEnvironment("GL_PEAK_TEST_CAPTURE")) {
            const unsigned workers = configuredWorkerCount();
            if (workers > 1)
                return runInWorkers(selectedTests, notSelected, includeCudaRoute, workers);
        }
        unsigned ownWorker = 0;
        unsigned workerCount = 1;
        if (workerSlot) std::tie(ownWorker, workerCount) = parseWorkerSlot(*workerSlot);

        // Statification: tests that grow `Memory`'s statified containers
        // need the process-wide pool. Init with the ProverParameters
        // DEFAULT triple — the sole source of truth now that the sizing is
        // config-independent (D-139), so a test
        // constructing a Prover (whose ctor re-inits from the
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
            // every test that constructs a Prover touches it, so
            // it must be initialized here too (PoolKind::Mail).
            initMailMemory(StaticMemoryConfig{
                defaults.static_mail_pool_bytes,
                defaults.static_mail_block_bytes,
                defaults.static_page_bytes,
                PoolKind::Mail });
            // The LB-body (fourth) pool backs the LB object store; every test
            // that constructs a Prover touches it (its ctor inits
            // the pool), so pre-init here too (PoolKind::Lb) to keep that ctor on
            // the idempotent-same-config path.
            initLbMemory(StaticMemoryConfig{
                defaults.static_lb_pool_bytes,
                defaults.static_lb_block_bytes,
                defaults.static_page_bytes,
                PoolKind::Lb });
            // The two scratch-arena registries (string-tier + request-gen) are
            // normally initialized by the Prover constructor; a test
            // that never constructs one — e.g. the standalone NameMap rig, whose
            // statified encodePush now builds its canonical scope name on the
            // string-scratch arena — still needs them. Pre-init here with the
            // SAME slot counts the constructor uses (logicalCores + 1 /
            // logicalCores) so a later Prover ctor stays on the
            // idempotent-same-shape path.
            const unsigned logicalCores =
                (std::max)(1u, std::thread::hardware_concurrency());
            initScratchArenas(logicalCores + 1);
            initGenScratchArenas(logicalCores + 1);   // same shape as the analyzer's init (reserved slot)
            initCompactionScratchArenas(logicalCores + 1);   // same shape as the analyzer's init
            initBatchByteSharing();   // same shape as the analyzer's init
        }

        // Per-test detail goes to `.debug/unit_tests.log` so the console
        // shows only the final summary (one line on success) plus any
        // failure information on miss. Cwd is the project root via
        // main.cpp's --unit-tests dispatch (chdir to projectRoot derived
        // from argv[0]), so the relative path lands inside .debug/ — for a
        // worker, inside its own project tree.
        std::error_code ec;
        fs::create_directories(".debug", ec);
        std::ofstream log(".debug/unit_tests.log",
                          std::ios::out | std::ios::trunc);
        if (log) {
            log << "[unit-tests] " << reg.size() << " tests registered, "
                << selectedTests.size() << " selected ("
                << (includeCudaRoute ? "CUDA-route gate" : "processor-route gate");
            if (workerSlot) log << ", worker " << ownWorker << "/" << workerCount;
            log << ")\n";
            log.flush();
        }

        // Redirect std::cout and std::clog into the log file for the
        // duration of the test run. Tests construct heavy objects
        // (notably Prover) whose initialisation chatter
        // would otherwise flood the console. std::cerr stays attached
        // to the real stderr so genuine errors thrown out of a test
        // body are visible. fail()'s assertion-line print goes through
        // stderr for the same reason.
        std::streambuf* const oldCoutBuf =
            log ? std::cout.rdbuf(log.rdbuf()) : nullptr;
        std::streambuf* const oldClogBuf =
            log ? std::clog.rdbuf(log.rdbuf()) : nullptr;

        std::size_t passed = 0;
        std::size_t assigned = 0;
        const auto wallStart = std::chrono::steady_clock::now();

        if (!workerSlot) {
            for (const TestEntry& t : reg) {
                if (t.cudaRoute && !includeCudaRoute && log) {
                    log << "[NOT SELECTED] " << t.suite << "." << t.name
                        << "  (CUDA route)\n";
                }
            }
        }
        std::size_t nonCudaOrdinal = 0;
        for (const TestEntry* entry : selectedTests) {
            const TestEntry& t = *entry;
            const unsigned worker = assignedWorker(t.cudaRoute, nonCudaOrdinal, workerCount, includeCudaRoute);
            if (!t.cudaRoute) ++nonCudaOrdinal;
            if (worker != ownWorker) continue;
            ++assigned;
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
                // swap above) and the log so the regression names itself
                // even when the run is otherwise silent.
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
            log << "[unit-tests] " << passed << "/" << assigned
                << " passed in " << totalMs << " ms\n";
            log.close();
        }

        // Restore std::cout / std::clog so the summary lands on the
        // real console rather than back in the log.
        if (oldCoutBuf) std::cout.rdbuf(oldCoutBuf);
        if (oldClogBuf) std::clog.rdbuf(oldClogBuf);

        // A worker reports through its log; the gate prints the summary.
        if (!workerSlot)
            printPassSummary(passed, selectedTests.size(), notSelected, only.has_value(), 1);
        return 0;
    }

}  // namespace tests
}  // namespace gl
