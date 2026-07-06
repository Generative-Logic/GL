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

#include "scratch_arena.hpp"

namespace gl {

    // `ScratchArena` is `LbArena` on the cold grant path (see
    // scratch_arena.hpp); the bump core lives in lb_arena.cpp. Only the
    // per-slot registry and its process-wide binding are defined here.

    /// @brief Create `slotCount` cold scratch arenas bound to `global`.
    ///
    /// @details
    /// Idempotent for an identical shape (run modes construct several
    /// `ExpressionAnalyzer`s per process); asserts on a mismatching re-init.
    /// Each slot is default-constructed unbound and then `bind`-ed cold (no
    /// reserve cap; blocks are acquired lazily and released per worker task).
    ///
    /// @param global    Pool the arenas draw blocks from.
    /// @param slotCount Number of worker slots (`logicalCores`).
    void ScratchArenaRegistry::init(GlobalMemoryManager* global,
                                    unsigned slotCount) {
        assert(global != nullptr && global->initialized());
        assert(slotCount >= 1);
        if (initialized()) {
            assert(global == global_
                && slotCount == static_cast<unsigned>(arenas_.size())
                && "ScratchArenaRegistry re-init with a different shape — one "
                   "process, one scratch-arena shape");
            return;
        }
        global_ = global;
        for (unsigned i = 0; i < slotCount; ++i) {
            arenas_.emplace_back();
            arenas_.back().bind(global);
        }
    }

    /// @brief The process-wide scratch-arena registry.
    ///
    /// @details
    /// Production binding for the prover's worker slots. Unit tests build
    /// private `ScratchArenaRegistry` / `ScratchArena` instances over private
    /// pools instead.
    ///
    /// @return The singleton (constructed on first use; `init` it via
    ///         `initScratchArenas` before any scratch allocation).
    ScratchArenaRegistry& scratchArenas() {
        static ScratchArenaRegistry instance;
        return instance;
    }

    /// @brief Initialize the process-wide registry against the process-wide
    ///        pool.
    ///
    /// @details
    /// Thin forwarder to `scratchArenas().init(&staticMemory(), ...)`; carries
    /// the same idempotent-same-shape / assert-on-mismatch contract. Called
    /// by the `ExpressionAnalyzer` constructor after `initStaticMemory` and
    /// after `logicalCores` is known.
    ///
    /// @param slotCount Number of worker slots (`logicalCores`).
    void initScratchArenas(unsigned slotCount) {
        scratchArenas().init(&staticMemory(), slotCount);
    }

    /// @brief The process-wide request-generation scratch-arena registry.
    ///
    /// @details
    /// A second per-slot registry beside `scratchArenas()` — see the header for
    /// why the request generators need scratch isolated from the string arena.
    ///
    /// @return The singleton (constructed on first use; `init` it via
    ///         `initGenScratchArenas` before any request-generation scratch).
    ScratchArenaRegistry& genScratchArenas() {
        static ScratchArenaRegistry instance;
        return instance;
    }

    /// @brief Initialize the process-wide request-generation registry against
    ///        the process-wide pool.
    ///
    /// @details
    /// Thin forwarder to `genScratchArenas().init(&staticMemory(), ...)`; same
    /// idempotent-same-shape / assert-on-mismatch contract as
    /// `initScratchArenas`. Called by the `ExpressionAnalyzer` constructor right
    /// after `initScratchArenas`.
    ///
    /// @param slotCount Number of worker slots (`logicalCores`).
    void initGenScratchArenas(unsigned slotCount) {
        genScratchArenas().init(&staticMemory(), slotCount);
    }

}
