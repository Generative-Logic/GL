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
#pragma once
#include "../memory.hpp"
#include "../memory_infra/temporary_witness_store.hpp"
namespace gl {
namespace tests { struct RuntimeSupportFixture; }
class EquivalenceProcessor;
class Prover;
class WatchProcessor;

        /// @brief Canonical spellings computed during one watch maintenance
        ///        pass, reused while the block's canonical classes are
        ///        unchanged.
        ///
        /// @details
        /// Keys pack (original id, scope id). A value of 0 records that the
        /// expression is already canonical at the scope; any other value is
        /// the canonical text's id in `texts`. `version` is the
        /// `canonicalUses.version` the entries were computed under: every
        /// complete-view change increments it, and the next lookup then
        /// starts from an empty cache. Both containers ride the page tier of
        /// a generation scratch arena.
        /// @invariant Every stored form equals `canonicalFormAtScope` of its
        ///            key under the classes of `version`.
        /// @see canonicalFormCached; canonicalTextCached; canonicalIdCached.
        class CanonicalFormCache {
private:
friend class EquivalenceProcessor;
friend class Prover;
friend class WatchProcessor;
friend struct tests::RuntimeSupportFixture;

            DirtyState dirty = DirtyState::Clean;
            ColdHashSet<BytesKeyStore> texts;
            TypedColdMap<int64_t, int32_t> forms;
            int32_t version = -1;

public:

            /// @brief Bind an empty cache to a generation scratch arena.
            /// @param arena Page-tier scratch arena that outlives the cache.
            /// @return (constructor)
            explicit CanonicalFormCache(ScratchArena& arena)
                : texts(&arena, &dirty), forms(&arena, &dirty) {}
            CanonicalFormCache(const CanonicalFormCache&) = delete;
            CanonicalFormCache& operator=(const CanonicalFormCache&) = delete;
        };
}
