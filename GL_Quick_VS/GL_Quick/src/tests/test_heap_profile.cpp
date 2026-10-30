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
/// @brief Unit tests for the heap census (`gl::heap_profile`).
///
/// @details
/// Covers the thread-sharded live counter, the allocation and free hooks'
/// counting, the replaced global operators (measurement builds with mimalloc),
/// the site table (recording, forgetting with backward-shift deletion under
/// colliding probes, snapshot order), the site label, symbolization, and one
/// end-to-end pass: sampling enabled, a large block allocated from a probe
/// function, the census hook, and the report naming the probe.

#include "test_harness.hpp"

#include "../infra/heap_profile.hpp"
#include "../parameters.hpp"

#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

#ifdef _MSC_VER
#define GL_TEST_NOINLINE __declspec(noinline)
#else
#define GL_TEST_NOINLINE __attribute__((noinline))
#endif

    // Distinct fake frames for the site-table tests.
    void* frame(uintptr_t v) { return reinterpret_cast<void*>(v); }

} // namespace

// The symbolization probes are exported C functions, so `symbolName` names
// them in every build, including the shipped binaries: Windows resolves them
// from the executable's export table when no PDB is present (the release
// strips PDBs), Linux and macOS through `dladdr`, which sees only the dynamic
// symbol table (the Makefile links with `-rdynamic`). A function with internal
// linkage, or one named only in a PDB, stays unnamed in a shipped binary. The
// names must not contain `heap_profile`: `siteLabel` treats such frames as
// allocator frames.
#ifdef _MSC_VER
#define GL_TEST_EXPORT __declspec(dllexport)
#else
#define GL_TEST_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

    // Distinct constants keep identical-code folding from merging the probes.
    GL_TEST_EXPORT GL_TEST_NOINLINE int64_t heapProfileSymbolProbe(int64_t x) {
        return x * 7919 + 104729;
    }

    GL_TEST_EXPORT GL_TEST_NOINLINE char* heapProfileAllocationProbe(std::size_t bytes) {
        char* block = new char[bytes];
        block[0] = 1;
        return block;
    }

} // extern "C"

namespace {

    // Internal linkage: absent from every export table and dynamic symbol
    // table, so only a PDB can name it.
    GL_TEST_NOINLINE int64_t internalSymbolProbe(int64_t x) {
        return x * 104723 + 7907;
    }

} // namespace

TEST(heap_profile, live_counter_sums_thread_shards) {
    const int64_t before = gl::heap_profile::liveBytes();
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([] { gl::heap_profile::addLiveBytes(1000); });
    for (std::thread& t : threads) t.join();
    threads.clear();
    threads.shrink_to_fit();
    const int64_t after = gl::heap_profile::liveBytes();
    gl::heap_profile::addLiveBytes(-4000);
    ASSERT_EQ(after - before, int64_t(4000));
}

TEST(heap_profile, hooks_count_allocations_and_frees) {
    const int64_t before = gl::heap_profile::liveBytes();
    int dummy = 0;
    gl::heap_profile::onAllocate(&dummy, 123);
    ASSERT_EQ(gl::heap_profile::liveBytes() - before, int64_t(123));
    gl::heap_profile::onFree(&dummy, 123);
    ASSERT_EQ(gl::heap_profile::liveBytes(), before);
}

TEST(heap_profile, replaced_operators_count_live_bytes) {
#if HEAP_MEASUREMENT && defined(USE_MIMALLOC)
    ASSERT_TRUE(gl::heap_profile::countsNewDelete());
    const int64_t before = gl::heap_profile::liveBytes();
    char* volatile block = new char[1 << 20];
    block[0] = 7;
    ASSERT_TRUE(gl::heap_profile::liveBytes() - before >= int64_t(1 << 20));
    delete[] block;
    ASSERT_EQ(gl::heap_profile::liveBytes(), before);
#else
    ASSERT_TRUE(!gl::heap_profile::countsNewDelete());
#endif
}

TEST(heap_profile, site_profile_records_and_forgets) {
    gl::heap_profile::SiteProfile profile;
    profile.init(64, 256, 1024);
    ASSERT_TRUE(profile.initialized());
    void* stackA[3] = { frame(0x1000), frame(0x2000), frame(0x3000) };
    void* stackB[2] = { frame(0x4000), frame(0x5000) };
    int blocks[3] = {};
    profile.record(&blocks[0], 100, stackA, 3, 11);
    profile.record(&blocks[1], 50, stackA, 3, 11);
    profile.record(&blocks[2], 70, stackB, 2, 22);
    ASSERT_EQ(profile.sampledLiveBytes(), int64_t(220));
    ASSERT_EQ(profile.liveSamples(), int64_t(3));

    gl::heap_profile::SiteBytes rows[4];
    ASSERT_EQ(profile.snapshot(rows, 4), 2);
    ASSERT_EQ(rows[0].bytes, int64_t(150));
    ASSERT_EQ(rows[0].count, int64_t(2));
    ASSERT_EQ(profile.site(rows[0].site).depth, 3);
    ASSERT_EQ(profile.site(rows[0].site).frames[1], stackA[1]);
    ASSERT_EQ(rows[1].bytes, int64_t(70));

    ASSERT_TRUE(profile.mayHold(&blocks[0]));
    ASSERT_TRUE(profile.forget(&blocks[0]));
    ASSERT_TRUE(!profile.forget(&blocks[0]));          // already gone
    int other = 0;
    ASSERT_TRUE(!profile.forget(&other));               // never sampled
    ASSERT_TRUE(profile.forget(&blocks[1]));
    ASSERT_TRUE(profile.forget(&blocks[2]));
    ASSERT_EQ(profile.sampledLiveBytes(), int64_t(0));
    ASSERT_EQ(profile.snapshot(rows, 4), 0);
    ASSERT_TRUE(!profile.mayHold(&blocks[0]));
}

TEST(heap_profile, site_profile_deletion_keeps_colliding_probe_chains) {
    // 16 sample slots, 12 live samples: probe chains collide and wrap, and
    // every removal must keep the remaining samples findable.
    gl::heap_profile::SiteProfile profile;
    profile.init(16, 16, 64);
    void* stack[1] = { frame(0x9000) };
    char blocks[12] = {};
    for (int i = 0; i < 12; ++i) profile.record(&blocks[i], i + 1, stack, 1, 5);
    ASSERT_EQ(profile.sampledLiveBytes(), int64_t(78));
    for (int i = 0; i < 12; i += 3) ASSERT_TRUE(profile.forget(&blocks[i]));
    for (int i = 0; i < 12; ++i) {
        if (i % 3 == 0) ASSERT_TRUE(!profile.forget(&blocks[i]));
        else ASSERT_TRUE(profile.forget(&blocks[i]));
    }
    ASSERT_EQ(profile.sampledLiveBytes(), int64_t(0));
    ASSERT_EQ(profile.liveSamples(), int64_t(0));
}

TEST(heap_profile, site_label_skips_allocator_frames_and_names_container) {
    const std::vector<std::string> names = {
        "gl::heap_profile::onAllocate", "operator new", "mi_new",
        "std::_Default_allocate_traits::_Allocate",
        "std::vector<int,std::allocator<int> >::_Emplace_reallocate<int const &>",
        "gl::Memory::addStatement", "gl::Prover::performElemPhase1",
        "gl::Prover::proveKernel", "gl::Prover::prove",
        "gl::run_modes::fullRun", "main" };
    ASSERT_EQ(gl::heap_profile::siteLabel(names),
        std::string("gl::Memory::addStatement < gl::Prover::performElemPhase1"
                    " < gl::Prover::proveKernel < gl::Prover::prove"
                    " < gl::run_modes::fullRun"
                    " [std::vector<int,std::allocator<int> >::_Emplace_reallocate<int const &>]"));
    const std::vector<std::string> bare = { "operator new", "gl::f" };
    ASSERT_EQ(gl::heap_profile::siteLabel(bare), std::string("gl::f"));
    const std::string longName(200, 'x');
    const std::string cutLabel = gl::heap_profile::siteLabel({ longName });
    ASSERT_EQ(cutLabel.size(), std::size_t(110));
}

TEST(heap_profile, symbol_name_resolves_a_function) {
    ASSERT_EQ(heapProfileSymbolProbe(1), int64_t(112648));
    const std::string name = gl::heap_profile::symbolName(
        reinterpret_cast<void*>(&heapProfileSymbolProbe));
    ASSERT_TRUE(name.find("heapProfileSymbolProbe") != std::string::npos);
}

TEST(heap_profile, symbol_name_never_lends_another_functions_name) {
    ASSERT_EQ(internalSymbolProbe(1), int64_t(112630));
    // Inside a function that no export or dynamic table lists, the nearest
    // listed symbol before it must not lend it its name.
    const std::string inner = gl::heap_profile::symbolName(
        reinterpret_cast<char*>(&internalSymbolProbe) + 1);
    ASSERT_TRUE(inner.empty() || inner.find("internalSymbolProbe") != std::string::npos);
    // 4 KiB past the start of the few-byte probe the address belongs to
    // another function, or to none.
    const std::string past = gl::heap_profile::symbolName(
        reinterpret_cast<char*>(&heapProfileSymbolProbe) + 4096);
    ASSERT_TRUE(past.find("heapProfileSymbolProbe") == std::string::npos);
}

TEST(heap_profile, sampling_names_the_allocating_function) {
#if HEAP_MEASUREMENT && defined(USE_MIMALLOC)
    if (!gl::heap_profile::enabled()) gl::heap_profile::enable(64 * 1024);
    char* block = heapProfileAllocationProbe(std::size_t(256) << 20);
    gl::heap_profile::noteCensusSample(gl::heap_profile::liveBytes(), true);
    std::ostringstream diagnostics;
    std::ostringstream markdown;
    gl::heap_profile::writeReport(diagnostics, markdown);
    delete[] block;
    const std::string report = markdown.str();
    ASSERT_TRUE(report.find("Heap live by allocation site at the heap peak") != std::string::npos);
    ASSERT_TRUE(report.find("heapProfileAllocationProbe") != std::string::npos);
    ASSERT_TRUE(diagnostics.str().find("[heap-site] snapshot=barrier") != std::string::npos);
#else
    // Without the replaced operators no allocation reaches the sampler.
    ASSERT_TRUE(!gl::heap_profile::countsNewDelete());
#endif
}
