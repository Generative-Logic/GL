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
#include "../proof/worker_context.hpp"
#include "derivation_tree.hpp"
#include <cassert>
#include <cstdint>

namespace gl {

/// @brief The strong-shot history gate: a strong step block's induction
///        hypothesis never fires on its own products
///        (D-600).
///
/// @details
/// A strong shot is a firing of one of a strong step block's induction
/// implications, the strong hypothesis (`Memory::recursionAuxyImplId`) or
/// the multiplication catcher (`Memory::recursionProductImplId`). The block's
/// history records its conclusion as an `implication` row whose first
/// dependency is the implication's text. A statement derives from a strong
/// shot when it is such a conclusion or when the derivation tree buildStack
/// cites for it (`DerivationTree::walk`) reaches one: a forced witness of a
/// shot's conclusion, an object built over it, a class rewrite of a
/// conclusion and every fact derived from those alike. The phase-2 finalize
/// (`HashRuleEvaluator::applyFiringRecords`) drops a strong shot when one of
/// its premises derives from a strong shot, so the hypothesis never shoots
/// on what its own shots produced, directly or through any number of further
/// derivations.
///
/// @invariant Mints nothing and writes no block state; all scratch rides the
///            worker slot's arenas and is released on return.
/// @see I-348, DerivationTree::walk,
///      Memory::isInductionHypothesisRule.
class StrongShotHistory {
public:
    /// @brief The origin-interner ids of a block's induction implication
    ///        texts.
    ///
    /// @details One id per induction implication (the hypothesis, then the
    /// multiplication catcher); 0 where the block has no such implication or
    /// its text was never cited by a history row.
    struct RuleTexts {
        int32_t ids[2] = { 0, 0 };
    };

    /// @brief Look up the origin-interner ids of the block's induction
    ///        implication texts.
    ///
    /// @details
    /// Decodes each installed induction implication from the block's rule
    /// interner and probes the origin interner for the same text, without
    /// minting. A history row cites a fired rule by this text (the firing
    /// capture seals `ruleInterner.decodeView(originalImplicationId)` as the
    /// row's first dependency), so a row applied an induction implication
    /// exactly when its first dependency carries one of these ids.
    ///
    /// @param mb Owning block (resident).
    /// @return The two ids; an entry is 0 when the block has no such
    ///         implication or no row cites it.
    /// @invariant Mints nothing.
    /// @see isShotConclusion.
    static RuleTexts inductionRuleTexts(Memory& mb) {
        RuleTexts out;
        const int32_t implications[2] = { mb.recursionAuxyImplId, mb.recursionProductImplId };
        for (int k = 0; k < 2; ++k) {
            if (implications[k] == 0) continue;
            out.ids[k] = mb.originInterner.lookup(mb.ruleInterner.decodeView(implications[k]));
        }
        return out;
    }

    /// @brief Whether a located statement is the conclusion of a strong
    ///        shot.
    ///
    /// @details
    /// True when one of the statement's history rows (`DerivationTree::forEachRow`,
    /// body run or class lines alike) is an `implication` row whose first
    /// dependency, the applied rule, carries the text of one of the block's
    /// induction implications.
    ///
    /// @param mb     Owning block (resident).
    /// @param node   The statement's history location (`DerivationTree::locate`).
    /// @param rules  The block's induction implication texts (`inductionRuleTexts`).
    /// @param gArena Container-tier scratch arena for record peeks.
    /// @return True iff a row of @p node applied an induction implication.
    /// @invariant Mints nothing; an `implication` row without its rule asserts.
    /// @see derivesFromShot.
    static bool isShotConclusion(Memory& mb, const DerivationTree::Node& node, const RuleTexts& rules,
                                 ScratchArena& gArena) {
        bool shot = false;
        DerivationTree::forEachRow(mb, node, gArena, [&](uint8_t tag, const auto& depAt, int32_t depN) {
            if (static_cast<OriginTag>(tag) != OriginTag::implication) return false;
            assert(depN > 0 && "StrongShotHistory: an implication row cites its rule first");
            const int32_t rule = static_cast<int32_t>(static_cast<uint64_t>(depAt(0)) >> 32);
            shot = rule == rules.ids[0] || rule == rules.ids[1];
            return shot;
        });
        return shot;
    }

    /// @brief Whether a statement is a strong shot's conclusion or derives
    ///        from one.
    ///
    /// @details
    /// Checks the statement itself at its exact scope, then walks its
    /// derivation tree (`DerivationTree::walk`, no end points) and stops at
    /// the first ancestor that is a strong shot's conclusion. The tree
    /// follows class rewrites, forced disintegrations and every other
    /// recorded derivation, so a product of a shot is found at any depth.
    ///
    /// @param mb    Owning block (resident, single owner).
    /// @param expr  The statement (caller-stable bytes).
    /// @param scope The statement's scope (caller-stable bytes).
    /// @param rules The block's induction implication texts (`inductionRuleTexts`).
    /// @return True iff the statement or an ancestor in its derivation tree
    ///         is a strong shot's conclusion.
    /// @invariant Mints nothing; a statement without history at its scope
    ///            asserts (Rule 16).
    /// @see isShotConclusion, firesOnOwnProduct.
    static bool derivesFromShot(Memory& mb, StrSpan expr, StrSpan scope, const RuleTexts& rules) {
        const unsigned slot = WorkerContext::coreId() >= 0
            ? static_cast<unsigned>(WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
        ScratchArena& gArena = genScratchArenas().forSlot(slot);
        ScratchScope frame(gArena);
        DerivationTree::Node node;
        const bool located = DerivationTree::locate(mb, expr, scope, true, gArena, node);
        assert(located && "StrongShotHistory: a premise holds history at its scope (Rule 16)");
        (void)located;
        if (isShotConclusion(mb, node, rules, gArena)) return true;
        return DerivationTree::walk(mb, expr, scope,
            [&](StrSpan ancestor, StrSpan ancestorScope) {
                ScratchScope visitFrame(gArena);
                DerivationTree::Node at;
                const bool found = DerivationTree::locate(mb, ancestor, ancestorScope, true, gArena, at);
                assert(found && "StrongShotHistory: a visited ancestor holds history at its scope");
                (void)found;
                return isShotConclusion(mb, at, rules, gArena);
            },
            [](StrSpan, StrSpan) { return false; });
    }

    /// @brief Whether a strong shot fires on one of its own products.
    ///
    /// @details
    /// The record is a head firing of one of a strong step block's induction
    /// implications. Its premises are the dependencies of its `implication`
    /// origin after the rule; the shot fires on its own products when one of
    /// them derives from a strong shot (`derivesFromShot`).
    ///
    /// @param mb     Owning block (resident, single owner): a strong step block.
    /// @param record A head firing record with `recursionHead` set.
    /// @return True iff a premise of @p record derives from a strong shot.
    /// @invariant Mints nothing; asserts the record is a strong shot.
    /// @see HashRuleEvaluator::applyFiringRecords, derivesFromShot.
    static bool firesOnOwnProduct(Memory& mb, const FiringRecord& record) {
        assert(mb.strongStepBlock && !record.isMarker && record.recursionHead
            && "StrongShotHistory: a strong shot of a strong step block");
        assert(record.originDeps.size() > 0 && "StrongShotHistory: a shot cites its rule first");
        const RuleTexts rules = inductionRuleTexts(mb);
        for (int32_t i = 1; i < record.originDeps.size(); ++i)
            if (derivesFromShot(mb, StrSpan(record.originDeps[i].original),
                    StrSpan(record.originDeps[i].validityName), rules))
                return true;
        return false;
    }
};

} // namespace gl
