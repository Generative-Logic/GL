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

#include <algorithm>
#include <cstring>
#include <set>
#include <string>

namespace gl {

/// @brief Visit every deeper premise list of an integration root — the flat
///        variants of its definition tree (D-352).
///
/// @details
/// Walk over the read-only entity table @p table (children resolved by
/// first-match signature scan), the root always dissolved, per element:
/// - no matching entity (atomic operators, negated elements), or an entity
///   of category `atomic` / `implication` / `or`: a LEAF — the element
///   verbatim as a premise;
/// - category `existence` / `and` whose core is NOT a configured definition
///   (@p configured; the generated `and<N>` / `existence<N>` products):
///   always DISSOLVED — its elements are walked in place, never a choice;
/// - category `existence` / `and` whose core IS a configured definition: a
///   CHOICE — INTACT (the element as a premise) or EXTENDED (its direct
///   elements walked in place; compounds exposed there choose again).
/// Every choice combination is enumerated depth-first, intact before
/// extended at every choice: immutable stack-local frames resume the
/// parent's remaining elements when a child's run ends, so a nested choice
/// exists only under an extended parent and sibling choices cross-multiply.
/// A premise identical to one already on the current list is not added
/// again: two branches of the tree can reach the same expression (in
/// `hasPrimeFactorization`, the binder premise `sequence[…,k,P]` of the
/// inner existence and the first conjunct of the bundle `primeFactorSeq`),
/// and a rule that names one premise twice can never fire, since a match
/// uses each statement once. The construction uses that one premise for
/// both leaves; the verifier (`_flat_unfold_variants`) and the Lean export
/// (`build_flat_integration_plan`) collapse the same leaves.
/// A premise list whose sorted multiset equals the root's own element list
/// (the one-level rule) or an already visited list is skipped (dedup over
/// the sorted `\x01`-joined premises in a `ColdHashSet` on the gen-scratch
/// page tier). The sink runs synchronously once per new list, with spans
/// into @p table that stay valid for the call.
///
/// Caller: the compile-time integration set
/// (`IntegrationRuleCompiler::compileIntegrationImplications`), whose flat
/// compacts the run time installs (D-503).
///
/// 0% heap: stack `StrSpan` runs (caps `MAX_ADMISSION_KEY_ELEMENTS` /
/// `MAX_INSTRUCTION_ELEMENTS`, asserted), the dedup set and its key bytes on
/// the per-slot gen-scratch arena (byte-bump rewound and pages released at
/// exit).
///
/// @tparam Sink Callable `void(const StrSpan* premises, int32_t premiseCount)`.
/// @param table      The read-only entity table holding the root and every
///                   entity of its definition tree.
/// @param rootEntity Index of the root entity; category `existence` or `and`
///                   (asserted).
/// @param configured The configured definition names.
/// @param sink       Receives each new premise list.
/// @return Void.
/// @invariant Reads @p table only; the visiting order is a pure function of
///            the table and @p configured.
/// @see forEachOrIntroductionRule, IntegrationRuleCompiler::compileIntegrationFamily.
template <class Sink>
inline void forEachFlatIntegrationVariant(const WorkInstruction& table, int32_t rootEntity,
    const std::set<std::string, std::less<>>& configured, Sink&& sink) {
    assert(rootEntity >= 0 && rootEntity < table.entityCount()
        && "forEachFlatIntegrationVariant: root entity out of range");
    {
        const StrSpan rootCategory = table.category(rootEntity);
        assert((equalSpans(rootCategory, StrSpan("existence", 9))
                || equalSpans(rootCategory, StrSpan("and", 3)))
            && "forEachFlatIntegrationVariant: root must be existence or and");
    }

    const unsigned genSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& genArena = genScratchArenas().forSlot(genSlot);
    ScratchScope genScope(genArena);
    DirtyState seenDirty = DirtyState::Clean;
    ColdHashSet<BytesKeyStore> seen(&genArena, &seenDirty);

    // Dedup key of a premise list: the premises sorted by compareSpans,
    // each followed by '\x01' (never an MPL byte), on the gen byte-bump
    // tier — order-free identity, so two variants with the same premise
    // multiset collapse.
    const auto premiseKey = [&](const StrSpan* prem, int32_t n) -> StrSpan {
        assert(n > 0 && "forEachFlatIntegrationVariant: empty premise list");
        StrSpan sorted[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
        for (int32_t i = 0; i < n; ++i) sorted[i] = prem[i];
        std::sort(sorted, sorted + n, [](const StrSpan& a, const StrSpan& b) {
            return compareSpans(a, b) < 0;
        });
        int32_t total = 0;
        for (int32_t i = 0; i < n; ++i) total += sorted[i].len + 1;
        char* buf = reinterpret_cast<char*>(genArena.resolve(genArena.alloc(total, 1)));
        int32_t at = 0;
        for (int32_t i = 0; i < n; ++i) {
            if (sorted[i].len > 0) {
                std::memcpy(buf + at, sorted[i].ptr, static_cast<std::size_t>(sorted[i].len));
                at += sorted[i].len;
            }
            buf[at++] = '\x01';
        }
        assert(at == total);
        return StrSpan(buf, total);
    };

    // Seed with the root's own element list, a repeated element named once
    // as on every variant: the one-level rule is not a flat variant.
    {
        StrSpan oneLevel[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
        const int32_t ec = table.elemCount(rootEntity);
        assert(ec > 0 && ec <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
            && "forEachFlatIntegrationVariant: root element count exceeds cap");
        int32_t oneLevelN = 0;
        for (int32_t j = 0; j < ec; ++j) {
            const StrSpan elem = table.elemAt(rootEntity, j);
            bool onList = false;
            for (int32_t i = 0; i < oneLevelN && !onList; ++i) onList = equalSpans(oneLevel[i], elem);
            if (!onList) oneLevel[oneLevelN++] = elem;
        }
        seen.mint(premiseKey(oneLevel, oneLevelN));
    }

    enum class Kind { Leaf, Dissolve, Choice };
    const auto classify = [&](StrSpan elem, int32_t& child) -> Kind {
        child = -1;
        for (int32_t c = 0; c < table.entityCount(); ++c) {
            if (equalSpans(table.signature(c), elem)) { child = c; break; }
        }
        if (child < 0) return Kind::Leaf;
        const StrSpan cat = table.category(child);
        const bool container = equalSpans(cat, StrSpan("existence", 9))
            || equalSpans(cat, StrSpan("and", 3));
        if (!container) return Kind::Leaf;
        const StrSpan core = extractExpressionSpan(elem);
        const bool fromConfig = configured.find(
                std::string_view(core.ptr, static_cast<std::size_t>(core.len)))
            != configured.end();
        return fromConfig ? Kind::Choice : Kind::Dissolve;
    };

    StrSpan prem[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
    int32_t premN = 0;
    struct Frame { int32_t entity; int32_t next; const Frame* parent; };

    const auto emitVariant = [&]() {
        const StrSpan key = premiseKey(prem, premN);
        if (seen.lookup(key) != 0) return;
        seen.mint(key);
        sink(static_cast<const StrSpan*>(prem), premN);
    };

    const auto walk = [&](int32_t entity, int32_t k, int32_t depth, const Frame* parent,
                          const auto& self) -> void {
        if (k == table.elemCount(entity)) {
            if (depth == 0) { emitVariant(); return; }
            assert(parent != nullptr);
            self(parent->entity, parent->next, depth - 1, parent->parent, self);
            return;
        }
        const StrSpan elem = table.elemAt(entity, k);
        int32_t child = -1;
        const Kind kind = classify(elem, child);
        if (kind == Kind::Leaf || kind == Kind::Choice) {
            bool onList = false;
            for (int32_t i = 0; i < premN && !onList; ++i) onList = equalSpans(prem[i], elem);
            if (onList) {
                self(entity, k + 1, depth, parent, self);
            } else {
                assert(premN < ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                    && "forEachFlatIntegrationVariant: premise count exceeds cap");
                prem[premN++] = elem;
                self(entity, k + 1, depth, parent, self);
                --premN;
            }
        }
        if (kind == Kind::Dissolve || kind == Kind::Choice) {
            assert(depth < ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                && "forEachFlatIntegrationVariant: nesting depth exceeds cap");
            assert(child != entity && "forEachFlatIntegrationVariant: cyclic instruction");
            for (const Frame* ancestor = parent; ancestor; ancestor = ancestor->parent) {
                assert(ancestor->entity != child
                    && "forEachFlatIntegrationVariant: cyclic instruction");
            }
            // A resumed sibling may explore at the same depth while this
            // invocation still owns alternative child choices. Its
            // continuation must not overwrite this path's frame.
            const Frame continuation{ entity, k + 1, parent };
            self(child, 0, depth + 1, &continuation, self);
        }
    };
    walk(rootEntity, 0, 0, nullptr, walk);
    assert(premN == 0);
    seen.release();
}

} // namespace gl
