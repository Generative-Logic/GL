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
/// @brief Identify anchor blocks and their history-relay role.
/// @details Owns this algorithm boundary; logical blocks retain their physical stores.
class LogicBlockIdentity {
public:
/// @brief Bind LogicBlockIdentity inputs.
/// @details Binds the existing inputs without performing proof work or moving storage.
/// @param parameters Existing parameters interface or settings.
/// @param anchorInfo Existing anchorInfo interface or settings.
/// @return None; construction binds the supplied references.
LogicBlockIdentity(const ProverParameters& parameters, const ce::AnchorInfo& anchorInfo);

public:
/// @brief True iff `lb` is an anchor LB — its `exprKey` starts with
        ///        `"(" + anchorInfo.name`.
        ///
        /// @details
        /// The one shared anchor-LB predicate: the zero-allocation prefix test
        /// on `exprKeyView()` (a `memcmp` after the leading `(`), byte-identical
        /// to `exprKey().rfind("(" + anchorInfo.name, 0) == 0`. Used by the D-29
        /// disintegration gate, by `prehandleAnchor`'s subtree walk, and by
        /// the incubator mail-edge pruning. The root sentinel (empty key) is
        /// never an anchor LB.
        ///
        /// @param lb The LB to classify.
        /// @return True iff the key carries the anchor prefix.
        /// @invariant Pure; reads only `anchorInfo.name` and the LB's key.
bool isAnchorLb(const Memory& lb) const {
            const std::string& anchorName = this->anchorInfo.name;
            const StrSpan key = lb.exprKeyView();
            return key.len > static_cast<int32_t>(anchorName.size())
                && key.len > 0 && key.ptr[0] == '('
                && std::memcmp(key.ptr + 1, anchorName.data(),
                               anchorName.size()) == 0;
        }

public:
/// @brief True iff `mb` relays the origin rows it absorbs from routing
        ///        mail into its own `mailOut` — the anchor LB of an incubator
        ///        grid.
        ///
        /// @details
        /// In incubator mode the root sentinel's log has one reader, the anchor
        /// LB (I-220). Its STATEMENTS (the compact
        /// rule carriers) stay there by design, but their HISTORY — the
        /// compacts' `compilation` rows, the `theorem` rows — is what every
        /// descendant's chapter cites (a twin's chapter walks theorem →
        /// `expansion` → compact → `compilation`). So the anchor LB copies every
        /// origin row of a status-3 (routing-mail) absorb into its own
        /// `mailOut.exprOriginMap`; its descendants receive those rows one
        /// cycle later — exactly the rows the root's log used to deliver to
        /// them. History only (Rule 16 / I-44): no statement is relayed, and a
        /// receiver dedups repeated lines. Outside incubator mode every LB
        /// reads the root's log itself and nothing is relayed.
        ///
        /// @param mb The absorbing LB.
        /// @return True iff `mb` is the anchor LB of an incubator grid.
        /// @invariant Pure; reads `parameters.incubator_mode` and the LB's key.
        /// @see isAnchorLb, collectMailAncestors, standardProcessing (the
        ///      routing-mail origin merge).
bool relaysAbsorbedHistory(const Memory& mb) const {
            return parameters.incubator_mode && this->isAnchorLb(mb);
        }
private:
const ProverParameters& parameters;
const ce::AnchorInfo& anchorInfo;
};
}

namespace gl {
/// @brief Bind LogicBlockIdentity inputs.
/// @details Binds the existing inputs without performing proof work or moving storage.
/// @param parameters Existing parameters interface or settings.
/// @param anchorInfo Existing anchorInfo interface or settings.
/// @return None; construction binds the supplied references.
inline LogicBlockIdentity::LogicBlockIdentity(const ProverParameters& parameters, const ce::AnchorInfo& anchorInfo) : parameters(parameters), anchorInfo(anchorInfo) {}
} // namespace gl
