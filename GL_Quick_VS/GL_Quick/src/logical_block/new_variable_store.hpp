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
class CaseProcessor;
class DisintegrationProcessor;
class Prover;

        /// @brief Transient backing for `disintegrateExpr2`'s `newVarMap`
        ///        (witness variable → its ordered defining elements), off the heap.
        ///
        /// @details
        /// Replaces the former `std::map<std::string, std::vector<std::string>>`
        /// with a `ColdMultiMap<BytesKeyStore,int32_t>` (var key → ordered run of
        /// element-string ids) over a `ColdHashSet<BytesKeyStore>` element interner,
        /// both PAGE-tier on a per-slot `genScratchArenas` arena (released per task) —
        /// no `std::map`/`std::vector` node heap. Each witness `newVar` is minted with
        /// a fresh `startInt`, so it is brand-new and its elements append
        /// consecutively (`appendToTail`'s current-last-or-new contract holds).
        /// `forEachVarSorted` reproduces the former `std::map` lex key order that
        /// Pass B's admission cascade depends on, sorting the key ids by decoded
        /// bytes on the arena's BYTE-BUMP tier (independent of the page tier, no heap).
        ///
        /// @invariant Lives within a single `disintegrateExpr2` call; spans it
        ///            returns point into the `arena` it was constructed on.
        class NewVarStore {
private:
friend class CaseProcessor;
friend class DisintegrationProcessor;
friend class Prover;
friend struct tests::RuntimeSupportFixture;

            LbArena* arena;
            DirtyState dirty = DirtyState::Clean;
            ColdHashSet<BytesKeyStore> elems;            ///< element-string interner
            ColdMultiMap<BytesKeyStore, int32_t> runs;   ///< var -> ordered elem-id run

public:

            explicit NewVarStore(LbArena* a)
                : arena(a), elems(a, &dirty), runs(a, &dirty) {}
            NewVarStore(const NewVarStore&) = delete;
            NewVarStore& operator=(const NewVarStore&) = delete;

            /// @brief Append one defining element to @p var's run.
            /// @param var  Witness variable name. @param elem Element expression.
            void addElement(const StrSpan& var, const StrSpan& elem) {
                runs.appendToTail(var, elems.mint(elem));
            }
            /// @brief Whether @p var is a key. @return `true` if present.
            bool hasVar(const StrSpan& var) const { return runs.lookup(var) != 0; }
            /// @brief @p var's key id, or 0. @return Id; 0 when absent.
            int32_t lookupVar(const StrSpan& var) const { return runs.lookup(var); }
            /// @brief Whether no var has been added. @return `count() == 0`.
            bool empty() const { return runs.count() == 0; }
            /// @brief Number of vars. @return Var count (ids 1..count()).
            int32_t varCount() const { return runs.count(); }
            /// @brief Decode a var id to its name. @param id Var id. @return Var span.
            StrSpan varAt(int32_t id) const { return runs.keyAt(id); }
            /// @brief Number of elements in @p id's run. @return Element count.
            int32_t elemCount(int32_t id) const { return runs.runLen(id); }
            /// @brief Decode the @p j-th element of var @p id. @return Element span.
            StrSpan elemAt(int32_t id, int32_t j) const {
                return elems.keyAt(runs.valueAt(id, j));
            }
            /// @brief Invoke @p fn(varId) for each var in DECODED-LEX key order.
            /// @param fn Callback `(int32_t varId)`.
            template <typename Fn> void forEachVarSorted(Fn fn) const {
                const int32_t n = runs.count();
                if (n == 0) return;
                ScratchScope sc(*arena);
                int32_t* ids = reinterpret_cast<int32_t*>(arena->resolve(arena->alloc(n * 4, 4)));
                for (int32_t i = 0; i < n; ++i) ids[i] = i + 1;
                std::sort(ids, ids + n, [&](int32_t a, int32_t b) {
                    return compareSpans(runs.keyAt(a), runs.keyAt(b)) < 0;
                });
                for (int32_t i = 0; i < n; ++i) fn(ids[i]);
            }
        };
}
