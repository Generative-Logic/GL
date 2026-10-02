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

#include "phase2_trace.hpp"
#include <cassert>
#include <cstdlib>
#include <cstring>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#else
#include <sys/resource.h>
#endif
#ifdef GL_CUDA
#include <nvtx3/nvToolsExt.h>
#endif

namespace gl {
    namespace { thread_local uint32_t traceDepth = 0; }


    /// @brief Read cumulative current-thread CPU usage without observing CUDA.
    /// @details The operating system accounts user and system execution
    /// separately: `GetThreadTimes` on Windows, the Mach `thread_info` basic
    /// record on macOS (which has no `RUSAGE_THREAD`), `getrusage` with
    /// `RUSAGE_THREAD` on Linux. Windows cycle counts are an independent
    /// diagnostic measure; other supported hosts report their absence
    /// explicitly.
    /// @return Current thread CPU counters; every operating-system call must succeed.
    Phase2ThreadCpu readPhase2ThreadCpu() {
        Phase2ThreadCpu result;
#ifdef _WIN32
        FILETIME creation{}, exit{}, system{}, user{};
        const auto status = GetThreadTimes(GetCurrentThread(), &creation, &exit, &system, &user);
        assert(status != 0);
        const auto nanoseconds = [](const FILETIME& value) {
            return ((uint64_t(value.dwHighDateTime) << 32) | value.dwLowDateTime) * 100;
        };
        result.userNanoseconds = nanoseconds(user);
        result.systemNanoseconds = nanoseconds(system);
        ULONG64 cycles = 0;
        const auto cycleStatus = QueryThreadCycleTime(GetCurrentThread(), &cycles);
        assert(cycleStatus != 0);
        result.cycles = cycles;
        result.cyclesAvailable = true;
#elif defined(__APPLE__)
        // mach_thread_self returns a send right that must be released.
        const mach_port_t thread = mach_thread_self();
        thread_basic_info_data_t info{};
        mach_msg_type_number_t count = THREAD_BASIC_INFO_COUNT;
        const kern_return_t status = thread_info(
            thread, THREAD_BASIC_INFO, reinterpret_cast<thread_info_t>(&info), &count);
        const kern_return_t released = mach_port_deallocate(mach_task_self(), thread);
        assert(status == KERN_SUCCESS && released == KERN_SUCCESS);
        result.userNanoseconds = uint64_t(info.user_time.seconds) * 1000000000
            + uint64_t(info.user_time.microseconds) * 1000;
        result.systemNanoseconds = uint64_t(info.system_time.seconds) * 1000000000
            + uint64_t(info.system_time.microseconds) * 1000;
#else
        rusage usage{};
        const auto status = getrusage(RUSAGE_THREAD, &usage);
        assert(status == 0);
        result.userNanoseconds = uint64_t(usage.ru_utime.tv_sec) * 1000000000
            + uint64_t(usage.ru_utime.tv_usec) * 1000;
        result.systemNanoseconds = uint64_t(usage.ru_stime.tv_sec) * 1000000000
            + uint64_t(usage.ru_stime.tv_usec) * 1000;
#endif
        return result;
    }

    /// @brief Read the immutable process opt-in for Phase 2 observations.
    /// @details Accepted values are absent, 0 and 1; malformed settings assert.
    /// The cache avoids repeated environment reads at operation boundaries.
    /// @return True exactly for GL_PHASE2_RT=1.
    bool phase2TraceEnabled() {
        static const bool enabled = [] {
#ifdef _WIN32
            char setting[2]{};
            std::size_t required = 0;
            const auto status = getenv_s(&required, setting, sizeof(setting), "GL_PHASE2_RT");
            assert(status == 0);
            const char* value = required == 0 ? nullptr : setting;
#else
            const char* value = std::getenv("GL_PHASE2_RT");
#endif
            assert(value == nullptr || std::strcmp(value, "0") == 0
                || std::strcmp(value, "1") == 0);
            return value != nullptr && std::strcmp(value, "1") == 0;
        }();
        return enabled;
    }

    /// @brief Return current-thread observation nesting.
    /// @details The counter is independent of prover and CUDA execution state.
    /// @return The number of enabled ranges not yet closed.
    uint32_t phase2TraceDepth() { return traceDepth; }

    /// @brief Begin an optional host range associated with CUDA launches.
    /// @details NVTX consumes the supplied text at entry. The numeric payload
    /// connects ranges with the separately recorded task and full-chain census.
    /// @param label Nonempty operation name.
    /// @param payload Numeric observation identity.
    /// @param enabled Whether to open the observation range.
    /// @param measureCpu Whether to collect current-thread CPU counters.
    /// @return An object owning this thread's range until finish or destruction.
    Phase2TraceScope::Phase2TraceScope(const char* label, uint64_t payload, bool enabled, bool measureCpu) {
        assert(label != nullptr && label[0] != '\0');
        if (enabled) {
            assert(traceDepth < UINT32_MAX);
            depth_ = ++traceDepth;
            measureCpu_ = measureCpu;
            if (measureCpu_) cpuStart_ = readPhase2ThreadCpu();
#ifdef GL_CUDA
            nvtxEventAttributes_t event{};
            event.version = NVTX_VERSION;
            event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            event.messageType = NVTX_MESSAGE_TYPE_ASCII;
            event.message.ascii = label;
            event.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
            event.payload.ullValue = payload;
            nvtxRangePushEx(&event);
#else
            (void)payload;
#endif
        }
    }

    /// @brief Close an observation still owned by this scope.
    /// @details Explicit finish leaves no range for the destructor to close.
    /// @return No value.
    Phase2TraceScope::~Phase2TraceScope() { finish(); }

    /// @brief Close the innermost enabled range on its creating thread.
    /// @details Disabled and already-closed scopes are defined no-ops; an
    /// out-of-order active close asserts before modifying the nesting counter.
    /// @return No value.
    void Phase2TraceScope::finish() {
        if (depth_ != 0) {
            assert(traceDepth == depth_);

            if (measureCpu_) {
                const auto end = readPhase2ThreadCpu();
                assert(end.userNanoseconds >= cpuStart_.userNanoseconds);
                assert(end.systemNanoseconds >= cpuStart_.systemNanoseconds);
                assert(end.cyclesAvailable == cpuStart_.cyclesAvailable);
                assert(end.cycles >= cpuStart_.cycles);
#ifdef GL_CUDA
                const auto mark = [](const char* label, uint64_t value) {
                    nvtxEventAttributes_t event{};
                    event.version = NVTX_VERSION;
                    event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
                    event.messageType = NVTX_MESSAGE_TYPE_ASCII;
                    event.message.ascii = label;
                    event.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
                    event.payload.ullValue = value;
                    nvtxMarkEx(&event);
                };
                mark("GL.Phase2.ThreadCpu.UserNs", end.userNanoseconds - cpuStart_.userNanoseconds);
                mark("GL.Phase2.ThreadCpu.SystemNs", end.systemNanoseconds - cpuStart_.systemNanoseconds);
                if (end.cyclesAvailable)
                    mark("GL.Phase2.ThreadCpu.Cycles", end.cycles - cpuStart_.cycles);
#endif
            }

#ifdef GL_CUDA
            nvtxRangePop();
#endif
            --traceDepth;
            depth_ = 0;
        }
    }
}
