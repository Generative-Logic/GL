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
#include "resolved_token.hpp"
namespace gl {
/// @brief A statement's member-walk tokens, each looked up once — the
        ///        pre-resolved token source of `forEachNonCanonicalMemberTokenCore`.
        ///
        /// @details The three runs of the per-class member walk in its fixed
        /// order: the scanned `force_lev_*` / `int_<depth>_lev_*` section, the scanned `it_*_lev_*`
        /// section, then every bracket token (`collectExprTokens`, duplicates
        /// kept). The arrays live on the caller's scratch arena. Each
        /// `forEach*` visits its run in order and stops at the first truthy
        /// visitor return — the token-source interface the core walk reads.
        struct ResolvedStatementTokens {
            const ResolvedToken* intRefs = nullptr;
            int32_t intCount = 0;
            const ResolvedToken* itRefs = nullptr;
            int32_t itCount = 0;
            const ResolvedToken* normalRefs = nullptr;
            int32_t normalCount = 0;

            /// @brief Visit the `force_lev_*` / `int_<depth>_lev_*` run.
            /// @details Retains token order and stops on the first accepted visit.
            /// @param fn Callable `(StrSpan token, NameId id) -> bool`; true stops.
            /// @return True iff @p fn stopped the run.
            template <class Fn>
            bool forEachInt(Fn&& fn) const {
                for (int32_t i = 0; i < intCount; ++i)
                    if (fn(intRefs[i].token, intRefs[i].id)) return true;
                return false;
            }

            /// @brief Visit the `it_*_lev_*` run.
            /// @details Retains token order and stops on the first accepted visit.
            /// @param fn Callable `(StrSpan token, NameId id) -> bool`; true stops.
            /// @return True iff @p fn stopped the run.
            template <class Fn>
            bool forEachIt(Fn&& fn) const {
                for (int32_t i = 0; i < itCount; ++i)
                    if (fn(itRefs[i].token, itRefs[i].id)) return true;
                return false;
            }

            /// @brief Visit the bracket-token run.
            /// @details Retains repeated tokens and stops on the first accepted visit.
            /// @param fn Callable `(StrSpan token, NameId id) -> bool`; true stops.
            /// @return True iff @p fn stopped the run.
            template <class Fn>
            bool forEachNormal(Fn&& fn) const {
                for (int32_t i = 0; i < normalCount; ++i)
                    if (fn(normalRefs[i].token, normalRefs[i].id)) return true;
                return false;
            }
        };
}
