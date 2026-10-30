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
#include "statement_arrivals.hpp"
#include "demand_built_rules.hpp"

namespace gl {

/// @brief Queue current-state reconciliation for a relevant known-fact key.
/// @details An unbuilt index will read current known state during bootstrap.
/// A built index only watches signed cores that still have active OR sources.
/// @param mb Owning block and its source/index state.
/// @param key Original expression and validity whose known membership changed.
/// @return Void.
void StatementArrivals::stageOrFactChange(Memory& mb, int64_t key) {
    stageDeferredOrEvent(mb, key);
    if (!mb.orImplicationMatches.built) return;
    const NameId original = static_cast<NameId>(uint64_t(key) >> 32);
    const StrSpan text = mb.nameMap.decodeView(original);
    const bool negated = text.len > 0 && text.ptr[0] == '!';
    const NameId core = mb.nameMap.lookup(extractExpressionUniversalSpan(text));
    assert(core != 0);
    const int64_t signedCore = packInt32Pair(core, negated);
    if (!mb.orImplicationMatches.postings().containsKey(
            orPostingKey(OrPostingKind::SourceCore, signedCore))) return;
    mb.orImplications.pendingFacts.push_back(key);
}

/// @brief Stage a known-row change that is an event of a demand-built OR
///        head (D-597).
/// @details A row shaped like a registered OR head's refutation template,
/// or one a waiting event names as its missing premise, is staged; any other
/// row is no event.
/// @param mb Owning block.
/// @param key Packed `(original, scope)` of the changed row.
/// @return Void.
void StatementArrivals::stageDeferredOrEvent(Memory& mb, int64_t key) {
    const DemandBuiltRuleRegistry& registry = DemandBuiltRuleRegistry::instance();
    if (!registry.hasOrRules()) return;
    const NameId original = static_cast<NameId>(uint64_t(key) >> 32);
    if (registry.hasOrTrigger(mb.nameMap.decodeView(original), DemandBuiltTrigger::Refutation)
        || (!mb.deferredOrWaiting.empty() && mb.deferredOrWaiting.lookup(static_cast<int32_t>(original)) != 0)) {
        mb.pendingDeferredOrEvents.push_back(key);
    }
}
}
