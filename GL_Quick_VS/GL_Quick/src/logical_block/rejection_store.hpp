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
#include "rejection_record.hpp"
namespace gl {
namespace tests { struct RuntimeSupportFixture; }
class Prover;

        /// @brief Transient backing for `disintegrateExpr2`'s `pendingRejections` /
        ///        `pendingRejectionsIntegration` buffers, off the heap.
        ///
        /// @details
        /// Replaces `std::map<std::string, std::vector<PendingRejection>>` (and the
        /// integration twin) with an APPEND-ONLY flat list of @ref RejRec on a
        /// per-slot `genScratchArenas` arena: a `ColdHashSet<BytesKeyStore>` string
        /// interner plus three `PagedVector`s (the records, the sibling string ids,
        /// the level ints). No `std::map`/`std::vector`/`std::set`/`std::string` node
        /// heap. The records append in Pass-B iteration order, which IS the former
        /// `std::map` lex-var-then-insertion drain order (Pass B walks
        /// `forEachVarSorted`, and one var's rejections are pushed consecutively), so
        /// the drain walks the list in order — no per-var keying, no sort.
        ///
        /// @invariant Lives within a single `disintegrateExpr2` call; spans it
        ///            returns point into the `arena` it was constructed on.
        class RejectionStore {
private:
friend class Prover;
friend struct tests::RuntimeSupportFixture;

            LbArena* arena;
            DirtyState dirty = DirtyState::Clean;
            ColdHashSet<BytesKeyStore> strings;
            PagedVector<RejRec> recs;
            PagedVector<int32_t> sibIds;
            PagedVector<int32_t> levs;

public:

            explicit RejectionStore(LbArena* a)
                : arena(a), strings(a, &dirty), recs(a, &dirty),
                  sibIds(a, &dirty), levs(a, &dirty) {}
            RejectionStore(const RejectionStore&) = delete;
            RejectionStore& operator=(const RejectionStore&) = delete;

            /// @brief Append one rejection record.
            /// @param var Witness var. @param s1 removedU / concreteConstituent.
            /// @param s2 markedExpr. @param s3 compactExpr (empty for integration).
            /// @param siblings Sibling spans. @param sibN Sibling count.
            /// @param levels Sorted level ints. @param levN Level count.
            void addRejection(const StrSpan& var, const StrSpan& s1, const StrSpan& s2,
                              const StrSpan& s3, const StrSpan* siblings, int32_t sibN,
                              const int32_t* levels, int32_t levN) {
                RejRec rec;
                rec.varId = strings.mint(var);
                rec.s1Id = strings.mint(s1);
                rec.s2Id = strings.mint(s2);
                rec.s3Id = s3.empty() ? 0 : strings.mint(s3);
                rec.sibStart = sibIds.size();
                rec.sibCount = sibN;
                for (int32_t i = 0; i < sibN; ++i) sibIds.push_back(strings.mint(siblings[i]));
                rec.levStart = levs.size();
                rec.levCount = levN;
                for (int32_t i = 0; i < levN; ++i) levs.push_back(levels[i]);
                recs.push_back(rec);
            }
            /// @brief Number of buffered rejections. @return Record count.
            int32_t count() const { return recs.size(); }
            /// @brief The @p i-th record in append order. @return Record reference.
            const RejRec& recAt(int32_t i) const { return recs[i]; }
            /// @brief Decode a string id (0 -> empty span). @return The span.
            StrSpan decode(int32_t id) const { return id == 0 ? StrSpan() : strings.keyAt(id); }
            /// @brief The @p k-th sibling of @p rec. @return Sibling span.
            StrSpan sibAt(const RejRec& rec, int32_t k) const {
                return strings.keyAt(sibIds[rec.sibStart + k]);
            }
            /// @brief The @p k-th level of @p rec. @return Level int.
            int32_t levAt(const RejRec& rec, int32_t k) const { return levs[rec.levStart + k]; }
        };
}
