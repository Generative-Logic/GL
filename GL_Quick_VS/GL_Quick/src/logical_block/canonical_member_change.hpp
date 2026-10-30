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
namespace gl {
//#pragma optimize("", off)

        /// One actual complete-view change, separate from multiplication events.
        /// Flag bits: 1 normal representative moved (or the member is new at
        /// the scope), 2 member new at the scope, 4 its class's bytes changed,
        /// 8 special representative moved, 16 locality of its rewrite equality
        /// changed (`canonicalRecordLocal`).
        struct CanonicalMemberChange { NameId scope, member; int32_t flags; };

        /// @brief Bit of a member record's low half that holds the locality of
        ///        the member's rewrite equality.
        ///
        /// @details
        /// A `canonicalMemberRepresentatives` value packs the normal
        /// representative into its high half and the special representative
        /// into its low half. Name ids stay below `MAX_NAME_IDS`, far under
        /// bit 30, so bit 30 of the low half is free to hold the locality.
        constexpr uint32_t kCanonicalRecordLocalBit = uint32_t(1) << 30;

        /// @brief Pack one member's recorded canonical state at a scope.
        ///
        /// @details
        /// The record is what the complete-view rebuild
        /// (`Canonicalizer::rebuildCanonicalScope`) knows about a member at one
        /// scope: the class's normal representative (the token every occurrence
        /// rewrites to), its special representative (the cleanup's choice) and
        /// whether the equality that licenses the member's rewrite is a local
        /// statement of this logic block (`Canonicalizer::appliedEqualityIsLocal`).
        /// Readers use `canonicalRecordNormal`, `canonicalRecordSpecial` and
        /// `canonicalRecordLocal`; none decodes the halves by hand.
        ///
        /// @param normal Normal representative id, 0 when the class has no strong member.
        /// @param special Special representative id, 0 when the class has no special member.
        /// @param local Whether the member's rewrite equality is a local statement here.
        /// @return The packed record.
        /// @invariant Both ids lie below `kCanonicalRecordLocalBit` (asserted).
        /// @see canonicalRecordNormal, canonicalRecordSpecial, canonicalRecordLocal.
        inline int64_t packCanonicalMemberRecord(NameId normal, NameId special, bool local) {
            assert(normal >= 0 && static_cast<uint32_t>(normal) < kCanonicalRecordLocalBit);
            assert(special >= 0 && static_cast<uint32_t>(special) < kCanonicalRecordLocalBit);
            const uint32_t low = static_cast<uint32_t>(special) | (local ? kCanonicalRecordLocalBit : 0u);
            return static_cast<int64_t>((static_cast<uint64_t>(static_cast<uint32_t>(normal)) << 32) | low);
        }

        /// @brief The normal representative of a packed member record.
        ///
        /// @details
        /// The high half of the record written by `packCanonicalMemberRecord`:
        /// the token every occurrence of the member rewrites to at the scope.
        ///
        /// @param record Packed member record.
        /// @return The normal representative id, 0 when the class has no strong member.
        /// @see packCanonicalMemberRecord.
        inline NameId canonicalRecordNormal(int64_t record) {
            return static_cast<NameId>(static_cast<uint64_t>(record) >> 32);
        }

        /// @brief The special representative of a packed member record.
        ///
        /// @details
        /// The low half of the record without the locality bit: the special
        /// member the cleanup verdicts compare against.
        ///
        /// @param record Packed member record.
        /// @return The special representative id, 0 when the class has no special member.
        /// @see packCanonicalMemberRecord.
        inline NameId canonicalRecordSpecial(int64_t record) {
            return static_cast<NameId>(static_cast<uint32_t>(record) & ~kCanonicalRecordLocalBit);
        }

        /// @brief The recorded locality of a member's rewrite equality.
        ///
        /// @details
        /// Bit 30 of the record's low half: whether the equality licensing the
        /// rewrite of the member to its normal representative was a local
        /// statement of this logic block when the scope's view was last
        /// rebuilt. False for a member that is its own representative.
        ///
        /// @param record Packed member record.
        /// @return The recorded locality.
        /// @see packCanonicalMemberRecord, Canonicalizer::appliedEqualityIsLocal.
        inline bool canonicalRecordLocal(int64_t record) {
            return (static_cast<uint32_t>(record) & kCanonicalRecordLocalBit) != 0;
        }
}
