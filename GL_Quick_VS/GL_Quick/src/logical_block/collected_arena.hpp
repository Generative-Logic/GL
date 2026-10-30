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
#include "collected_record.hpp"
namespace gl {
namespace tests { struct RuntimeSupportFixture; }
class CaseProcessor;
class DisintegrationProcessor;
class Prover;

        /// @brief Transient append-only backing for `disintegrateExpr2`'s
        ///        `collected` keyed accumulator, off the malloc heap.
        ///
        /// @details
        /// Replaces the former
        /// `std::map<std::string, std::pair<std::set<ExpressionWithValidity>,
        /// std::set<std::string>>>` with a flat `PagedVector<CollectedRec>` on a
        /// per-slot `genScratchArenas` arena (released per task) — no
        /// `std::map`/`std::set` node heap, no per-element `std::string`. Records
        /// are appended raw because neither order nor duplicates are observable:
        /// every read funnels into the final `std::set`s (which re-sort) and
        /// `addToFinal` is idempotent via `finalStringStatements`, so no
        /// dedup/sort is needed. Strings are copied into the arena once on
        /// insert (@ref dup); the spans stay valid for the whole
        /// `disintegrateExpr2` call (the scratch arena never compacts mid-use).
        ///
        /// @invariant Lives within a single `disintegrateExpr2` call; every span
        ///            it returns points into the `arena` it was constructed on.
        class CollectedArena {
private:
friend class CaseProcessor;
friend class DisintegrationProcessor;
friend class Prover;
friend struct tests::RuntimeSupportFixture;

            LbArena* arena;
            DirtyState dirty = DirtyState::Clean;
            ColdHashSet<BytesKeyStore> strings;   ///< byte interner (page tier)
            PagedVector<CollectedRec> recs;       ///< id records (page tier)
            TypedColdSet<int32_t> orSourceKeyIds;
            TypedColdSet<int32_t> unconditionalOrdisKeyIds;

public:

            /// @brief Bind to the per-slot scratch arena. The interner and the
            ///        records both ride its PAGE tier (ColdHashSet / PagedVector);
            ///        nothing uses the byte-bump tier, so there is no interference.
            /// @param a The arena; outlives this accumulator.
            explicit CollectedArena(LbArena* a)
                : arena(a), strings(a, &dirty), recs(a, &dirty),
                  orSourceKeyIds(a, &dirty), unconditionalOrdisKeyIds(a, &dirty) {}
            CollectedArena(const CollectedArena&) = delete;
            CollectedArena& operator=(const CollectedArena&) = delete;

            /// @brief Append an implication under @p key.
            /// @param key Statement key. @param original Implication expression.
            /// @param validity Implication validity name.
            void insertImpl(const StrSpan& key, const StrSpan& original,
                            const StrSpan& validity) {
                recs.push_back(CollectedRec{ strings.mint(key), strings.mint(original),
                                             strings.mint(validity), 0 });
            }
            /// @brief Append a child statement under @p key.
            /// @param key Statement key. @param child Child statement bytes.
            void insertChild(const StrSpan& key, const StrSpan& child) {
                recs.push_back(CollectedRec{ strings.mint(key), strings.mint(child), 0, 1 });
            }
            /// @brief Record that @p key exists even with no impls/children.
            /// @param key Statement key (the `collected[K];` ensure-exists).
            void ensureKey(const StrSpan& key) {
                recs.push_back(CollectedRec{ strings.mint(key), 0, 0, 2 });
            }

            /// @brief Mark a collected positive OR for final source admission.
            /// @details Intent follows the key through the existing admission
            /// closure; collecting it alone does not create a live source.
            /// @param key Ground OR expression in the collection's key space.
            /// @return Void.
            void markOrSource(StrSpan key) {
                ensureKey(key);
                orSourceKeyIds.mint(strings.lookup(key));
            }

            /// @brief Test whether an admitted collection key has OR intent.
            /// @details Missing and ordinary statement keys are defined misses.
            /// @param key Exact collected expression bytes.
            /// @return True exactly when the key was marked as an OR source.
            bool isOrSource(StrSpan key) const {
                return orSourceKeyIds.contains(strings.lookup(key));
            }

            /// @brief Mark a collected OR as reached by unfolding a conjecture
            ///        premise.
            /// @details The publication walk marks an OR it consumes with the
            /// conjecture-premise permission; the deposit door stages each
            /// marked OR published at main for the unconditional-ordis mail
            /// flag, so every descendant with proof work opens its case split
            /// (D-528).
            /// The key must already be collected (`markOrSource` precedes this
            /// call in the same walk).
            /// @param key Ground OR expression in the collection's key space.
            /// @return Void.
            void markUnconditionalOrdis(StrSpan key) {
                const int32_t id = strings.lookup(key);
                assert(id != 0
                    && "markUnconditionalOrdis: the OR is collected before it is marked");
                unconditionalOrdisKeyIds.mint(id);
            }

            /// @brief Test whether a collection key was reached by unfolding a
            ///        conjecture premise.
            /// @details Missing and unmarked keys are defined misses.
            /// @param key Exact collected expression bytes.
            /// @return True exactly when `markUnconditionalOrdis` marked the key.
            bool isUnconditionalOrdis(StrSpan key) const {
                return unconditionalOrdisKeyIds.contains(strings.lookup(key));
            }
            /// @brief Invoke @p fn(original, validity) for each impl under @p key.
            /// @param key Statement key. @param fn Callback `(StrSpan, StrSpan)`.
            template <typename Fn> void forImpls(const StrSpan& key, Fn fn) const {
                const int32_t qid = strings.lookup(key);
                if (qid == 0) return;
                const int32_t n = recs.size();
                for (int32_t i = 0; i < n; ++i) {
                    const CollectedRec& r = recs[i];
                    if (r.kind == 0 && r.keyId == qid)
                        fn(strings.decode(r.aId), strings.decode(r.bId));
                }
            }
            /// @brief Invoke @p fn(child) for each child under @p key.
            /// @param key Statement key. @param fn Callback `(StrSpan)`.
            template <typename Fn> void forChildren(const StrSpan& key, Fn fn) const {
                const int32_t qid = strings.lookup(key);
                if (qid == 0) return;
                const int32_t n = recs.size();
                for (int32_t i = 0; i < n; ++i) {
                    const CollectedRec& r = recs[i];
                    if (r.kind == 1 && r.keyId == qid) fn(strings.decode(r.aId));
                }
            }
            /// @brief Invoke @p fn(key) once per DISTINCT key (any kind).
            /// @param fn Callback `(StrSpan)`; keys arrive in first-seen order.
            template <typename Fn> void forEachKey(Fn fn) const {
                const int32_t n = recs.size();
                for (int32_t i = 0; i < n; ++i) {
                    bool seen = false;
                    for (int32_t j = 0; j < i; ++j)
                        if (recs[j].keyId == recs[i].keyId) { seen = true; break; }
                    if (!seen) fn(strings.decode(recs[i].keyId));
                }
            }
        };
}
