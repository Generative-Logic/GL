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
class EqualityPropagation;
class Prover;






        /// @brief The scan-and-lookup token source of
        ///        `forEachNonCanonicalMemberTokenCore`.
        ///
        /// @details
        /// Walks the two special sections of a `SpecialTokenScanView` with its
        /// sequential `TokenCursor`s and the normal-name tokens a caller source
        /// yields, looking each token up in the NameMap (non-minting) just
        /// before it is visited — the registry filter's per-token order. After
        /// a stop the normal source keeps yielding but no further token is
        /// looked up or visited.
        ///
        /// @tparam NormalSource Callable `(Fn) -> void` invoking `Fn(StrSpan)`
        ///                      per bracket token; an empty source yields none.
        template <class NormalSource>
        class ScannedTokenRuns {
private:
friend class EqualityPropagation;
friend class Prover;
friend struct tests::RuntimeSupportFixture;

            const SpecialTokenScanView& scan;   ///< The statement's token scan.
            NormalSource& normalSource;         ///< Bracket-token source.
            Memory& mb;                         ///< Owning LB (NameMap).

public:
/// @brief Bind token scans and a normal-token visitor for one traversal.
/// @details References retain the original aggregate's binding order and lifetime.
/// @param scan Existing special-token view.
/// @param normalSource Visitor for normal tokens.
/// @param mb Block owning name identities.
/// @return Constructed token traversal view.
ScannedTokenRuns(const SpecialTokenScanView& scan, NormalSource& normalSource, Memory& mb)
 : scan(scan), normalSource(normalSource), mb(mb) {}

            /// @brief Visit the `force_lev_*` / `int_<depth>_lev_*` section.
            /// @param fn Callable `(StrSpan token, NameId id) -> bool`; true stops.
            /// @return True iff @p fn stopped the run.
            template <class Fn>
            bool forEachInt(Fn&& fn) const { return walk(scan.intTokens(), fn); }

            /// @brief Visit the `it_*_lev_*` section.
            /// @param fn Callable `(StrSpan token, NameId id) -> bool`; true stops.
            /// @return True iff @p fn stopped the run.
            template <class Fn>
            bool forEachIt(Fn&& fn) const { return walk(scan.itTokens(), fn); }

            /// @brief Visit the normal source's tokens.
            /// @param fn Callable `(StrSpan token, NameId id) -> bool`; true stops.
            /// @return True iff @p fn stopped the run.
            template <class Fn>
            bool forEachNormal(Fn&& fn) const {
                bool stopped = false;
                normalSource([&](StrSpan token) {
                    if (stopped) return;
                    if (fn(token, mb.nameMap.lookup(token))) stopped = true;
                });
                return stopped;
            }

private:
            /// @brief Walk one scan section, looking each token up first.
            /// @param run Cursor over the section.
            /// @param fn  Callable `(StrSpan token, NameId id) -> bool`; true stops.
            /// @return True iff @p fn stopped the run.
            template <class Fn>
            bool walk(SpecialTokenScanView::TokenCursor run, Fn& fn) const {
                StrSpan token;
                while (run.next(token))
                    if (fn(token, mb.nameMap.lookup(token))) return true;
                return false;
            }
        };
}
