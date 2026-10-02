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
#include "typed_cold_map.hpp"

namespace gl {
    /// @brief Stable references for an ordered implication chain and its proof.
    /// @details Scope uses NameMap; the other columns use ruleInterner. Levels
    /// are stored in the matching key blob, so every support variant is distinct.
    /// Premise stores the newline-separated ordered atoms; premiseCount is exact.
    struct UnaryRuleInfo {
        int32_t premise, head, proof, scope;
        uint32_t local;
        int32_t premiseCount = 1;
    };

    /// @brief Per-block producer-chain roots, unary-consumer joins and pending work.
    /// @details All logical state is pool-backed and canonically deloaded.
    /// Source keys contain parent owner, copy owner, scope and levels. Closure
    /// keys contain canonical endpoints, scope and levels. Separate root activity
    /// prevents a derived cycle from surviving removal of its last source.
    struct UnaryCompositionStore {
        ColdHashSet<BytesKeyStore> sources;
        ColdHashSet<BytesKeyStore>::LengthsView sourceLengths{&sources};
        ColdHashSet<BytesKeyStore>::BytesView sourceBytes{&sources};
        PagedVector<UnaryRuleInfo> sourceInfo;
        PagedVector<uint8_t> sourceActive;
        ColdHashSet<BytesKeyStore> rules;
        ColdHashSet<BytesKeyStore>::LengthsView ruleLengths{&rules};
        ColdHashSet<BytesKeyStore>::BytesView ruleBytes{&rules};
        PagedVector<UnaryRuleInfo> ruleInfo;
        PagedVector<int32_t> pending;
        ColdHashSet<PodKeyStore<int64_t>> generated;
        ColdHashSet<PodKeyStore<int64_t>>::KeysView generatedKeys{&generated};
        TypedColdSetMap<int64_t, int32_t> heads, premises;
        TypedColdSetMap<int64_t, int32_t>::KeysView headKeys{&heads.inner()}, premiseKeys{&premises.inner()};
        TypedColdSetMap<int64_t, int32_t>::RunStartsView headStarts{&heads.inner()}, premiseStarts{&premises.inner()};
        TypedColdSetMap<int64_t, int32_t>::RunValuesView headValues{&heads.inner()}, premiseValues{&premises.inner()};
        PagedVector<uint8_t> rebuild;
        ColdHashSet<BytesKeyStore> admissionTransfers;
        ColdHashSet<BytesKeyStore>::LengthsView transferLengths{&admissionTransfers};
        ColdHashSet<BytesKeyStore>::BytesView transferBytes{&admissionTransfers};
        TypedColdSetMap<int32_t, int64_t> consumerDemands;
        TypedColdSetMap<int32_t, int64_t>::KeysView consumerDemandKeys{&consumerDemands.inner()};
        TypedColdSetMap<int32_t, int64_t>::RunStartsView consumerDemandStarts{&consumerDemands.inner()};
        TypedColdSetMap<int32_t, int64_t>::RunValuesView consumerDemandValues{&consumerDemands.inner()};
        ColdHashSet<PodKeyStore<int64_t>> observedDemands;
        ColdHashSet<PodKeyStore<int64_t>>::KeysView observedDemandKeys{&observedDemands};
        // These guards are call-local: a claimed block cannot deload mid-drain.
        bool publishing = false;
        int64_t publishingOwner = 0;
        bool rebuilding = false;
        // Key-set epochs of the integration demands and of `rules` read at the
        // start of the last admission-transfer walk. That walk minted a receipt
        // for every demand/rule pair it could match, so while both epochs stand
        // a repeated walk finds only receipted pairs and is skipped. Derived,
        // never deloaded; a reload bumps both epochs and forces a walk.
        bool transferWalked = false;
        uint32_t transferDemandEpoch = 0;
        uint32_t transferRuleEpoch = 0;

        /// @brief Bind every column to the same owning arena and dirty state.
        /// @details Construction is lazy and allocates no blocks.
        /// @param arena Owning block arena.
        /// @param dirty Owning aggregate dirty state.
        /// @return An empty composition store.
        UnaryCompositionStore(LbArena* arena, DirtyState* dirty)
            : sources(arena, dirty), sourceInfo(arena, dirty), sourceActive(arena, dirty),
              rules(arena, dirty), ruleInfo(arena, dirty), pending(arena, dirty),
              generated(arena, dirty), heads(arena, dirty), premises(arena, dirty), rebuild(arena, dirty),
              admissionTransfers(arena, dirty), consumerDemands(arena, dirty), observedDemands(arena, dirty) {}

        /// @brief Enumerate mutable canonical facets in fixed tag order.
        /// @details Every pending event and root-support bit survives deload.
        /// @param base First of twenty-two reserved tags.
        /// @param visit Synchronous container visitor.
        /// @return Void.
        template<class Visitor> void visitContainers(uint32_t base, Visitor&& visit) {
            visit(base, sourceLengths); visit(base+1, sourceBytes);
            visit(base+2, sourceInfo); visit(base+3, sourceActive);
            visit(base+4, ruleLengths); visit(base+5, ruleBytes);
            visit(base+6, ruleInfo); visit(base+7, pending); visit(base+8, generatedKeys);
            visit(base+9, headKeys); visit(base+10, headStarts); visit(base+11, headValues);
            visit(base+12, premiseKeys); visit(base+13, premiseStarts); visit(base+14, premiseValues);
            visit(base+15, rebuild);
            visit(base+16, transferLengths); visit(base+17, transferBytes);
            visit(base+18, consumerDemandKeys); visit(base+19, consumerDemandStarts);
            visit(base+20, consumerDemandValues); visit(base+21, observedDemandKeys);
        }

        /// @brief Enumerate read-only canonical facets in fixed tag order.
        /// @details Tags match the mutable visitor exactly.
        /// @param base First of twenty-two reserved tags.
        /// @param visit Synchronous read-only container visitor.
        /// @return Void.
        template<class Visitor> void visitContainers(uint32_t base, Visitor&& visit) const {
            visit(base, sourceLengths); visit(base+1, sourceBytes);
            visit(base+2, sourceInfo); visit(base+3, sourceActive);
            visit(base+4, ruleLengths); visit(base+5, ruleBytes);
            visit(base+6, ruleInfo); visit(base+7, pending); visit(base+8, generatedKeys);
            visit(base+9, headKeys); visit(base+10, headStarts); visit(base+11, headValues);
            visit(base+12, premiseKeys); visit(base+13, premiseStarts); visit(base+14, premiseValues);
            visit(base+15, rebuild);
            visit(base+16, transferLengths); visit(base+17, transferBytes);
            visit(base+18, consumerDemandKeys); visit(base+19, consumerDemandStarts);
            visit(base+20, consumerDemandValues); visit(base+21, observedDemandKeys);
        }

        /// @brief Account for every retained source, rule and index byte.
        /// @details Includes pending events and retired root identities.
        /// @return Total pool-backed live bytes.
        int64_t liveBytes() const {
            return sources.liveBytes()+sourceInfo.liveBytes()+sourceActive.liveBytes()
                +rules.liveBytes()+ruleInfo.liveBytes()+pending.liveBytes()+generated.liveBytes()
                +heads.liveBytes()+premises.liveBytes()+rebuild.liveBytes()+admissionTransfers.liveBytes()
                +consumerDemands.liveBytes()+observedDemands.liveBytes();
        }
    };
}
