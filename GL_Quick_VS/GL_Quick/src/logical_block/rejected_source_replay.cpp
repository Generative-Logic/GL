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
#include "../logical_block/statement_registry_operations.hpp"
#include "rejected_source_replay.hpp"
namespace gl {
/// @brief Decide at a rule's installation whether it is demand-built, and
///        register it when it is.
///
/// @details
/// The block's anchor arguments (`ScopeSyntax::anchorArgumentSpans`) bound
/// the sets and operators the criterion admits and fix the anchor slots of
/// a theorem rule's anchor premise; `analyzeDemandBuiltRule` applies the
/// criterion to the rule text and the compiled definitions. A demand-built
/// rule is parsed into the process-wide registry, once per rule text; the
/// caller then records that the block holds it instead of installing it
/// (D-596, D-597).
///
/// @param mb      The installing block.
/// @param rule    The expanded rule text.
/// @param carrier The statement whose expansion produced it.
/// @return True when the rule is demand-built and must not be installed.
/// @invariant Mints nothing in the block; allocates nothing.
/// @see analyzeDemandBuiltRule, CompactRuleProcessor::installExpandedImplication.
bool RejectedSourceReplay::deferDemandBuiltRule(Memory& mb, StrSpan rule, StrSpan carrier) {
    if (rule.len < 4 || std::memcmp(rule.ptr, "(>[", 3) != 0) return false;
    StrSpan anchors[ExecutionParameters::MAX_ARITY];
    const int32_t anchorN = gl::ScopeSyntax::anchorArgumentSpans(mb, StrSpan(anchorInfo.name),
        anchors, ExecutionParameters::MAX_ARITY);
    DemandBuiltRule parsed;
    if (!analyzeDemandBuiltRule(rule, carrier, definitions, StrSpan(anchorInfo.name), anchors, anchorN, parsed))
        return false;
    DemandBuiltRuleRegistry::instance().add(parsed);
    return true;
}

/// @brief Build the instances of the demand-built rules a demand key names.
///
/// @details
/// Every registered rule whose witness template matches @p key
/// (`matchDemandBuiltWitness`) is built for the matched values, in rule-text
/// order, by `buildDemandedInstance`. Called at every key gain that passes a
/// revival seam with a visible demand, and by `buildWaitingInstances`.
///
/// @param mb           The owning block.
/// @param key          The demand key in bare marker form.
/// @param validityName The demand's scope.
/// @return Nothing; writes the internal mail.
/// @invariant Heap-free; the matched order is deterministic.
/// @see buildDemandedInstance, buildWaitingInstances.
void RejectedSourceReplay::buildDemandedInstances(Memory& mb, StrSpan key, StrSpan validityName) {
    const DemandBuiltRuleRegistry& registry = DemandBuiltRuleRegistry::instance();
    const int32_t ruleN = registry.count();
    int32_t matched[kDemandBuiltRuleCap];
    StrSpan values[kDemandBuiltRuleCap][kDemandBuiltBoundCap];
    int32_t matchN = 0;
    for (int32_t i = 0; i < ruleN; ++i)
        if (matchDemandBuiltWitness(registry.at(i), key, values[matchN])) matched[matchN++] = i;
    int32_t order[kDemandBuiltRuleCap];
    for (int32_t m = 0; m < matchN; ++m) order[m] = m;
    std::sort(order, order + matchN, [&](int32_t left, int32_t right) {
        return compareSpans(registry.at(matched[left]).textSpan(), registry.at(matched[right]).textSpan()) < 0;
    });
    for (int32_t m = 0; m < matchN; ++m) {
        int32_t missingPremise = -1;
        buildDemandedInstance(mb, registry.at(matched[order[m]]), values[order[m]], validityName,
            mb.sameIterationInternalMail, missingPremise);
    }
}

/// @brief Build one demand-built rule's instance for matched values, exactly
///        as the rule's firing would.
///
/// @details
/// Nothing is built unless, on the demand scope's chain, the block holds the
/// rule (its installed-rule index) and knows every instantiated typing
/// premise; nor for a repeated premise instance (the hash engine never uses
/// one statement for two premises), a premise beyond the generation cap, or
/// a head already known. Otherwise the head instance enters the internal
/// mail as the firing record would deposit it: at the deepest of the rule's
/// scope and the premises' scopes (each the shallowest scope that knows it),
/// with the union of their levels and the carrier's, witness generation one
/// above the premises', the `implication` history row (the rule, then the
/// premises in expression order) and the incubator disintegration gate
/// (D-29).
///
/// @param mb             The owning block.
/// @param rule           The demand-built rule.
/// @param values         Each bound variable's value.
/// @param validityName   The demand's scope.
/// @param output         The internal-mail channel the instance enters.
/// @param missingPremise Receives the first premise not known on the chain
///                       when the result is `Untyped`.
/// @return The outcome.
/// @invariant Heap-free (stack buffers and the slot's scratch arena).
/// @see HashRuleEvaluator::checkLocalEncodedMemoryStatic, the firing record.
DemandBuildResult RejectedSourceReplay::buildDemandedInstance(Memory& mb, const DemandBuiltRule& rule,
    const StrSpan* values, StrSpan validityName, ColdMail& output, int32_t& missingPremise) {
    const NameId target = mb.nameMap.lookup(validityName);
    assert(target != 0 && "buildDemandedInstance: the demand scope is minted");
    const int32_t ancN = mb.nameMap.ancLen(target);
    // The shallowest scope on the demand scope's chain (root first) that
    // knows a statement, 0 when none does.
    const auto shallowestKnown = [&](StrSpan text) -> NameId {
        const NameId id = mb.nameMap.lookup(text);
        if (id == 0) return 0;
        for (int32_t k = 0; k < ancN; ++k) {
            const NameId anc = mb.nameMap.ancAt(target, k);
            if (mb.intKnownStatements.find(StatementKey{ id, anc }) != nullptr) return anc;
        }
        return 0;
    };
    NameId ruleScope = 0;
    const int32_t ruleTextId = mb.lbStateInterner.lookup(rule.textSpan());
    for (int32_t k = 0; ruleTextId != 0 && k < ancN && ruleScope == 0; ++k) {
        const NameId anc = mb.nameMap.ancAt(target, k);
        const int32_t scopeId = mb.lbStateInterner.lookup(mb.nameMap.decodeView(anc));
        if (scopeId != 0 && mb.expandedImplications.lookup(LbStatePairKey{ ruleTextId, scopeId }) != 0)
            ruleScope = anc;
    }
    if (ruleScope == 0) return DemandBuildResult::NoRule;

    const int32_t premiseN = rule.premiseCount;
    char premiseText[kDemandBuiltPremiseCap][kDemandBuiltTextCap];
    StrSpan premises[kDemandBuiltPremiseCap];
    NameId premiseScopes[kDemandBuiltPremiseCap];
    int premiseIteration = -1;
    for (int32_t p = 0; p < premiseN; ++p) {
        premises[p] = StrSpan(premiseText[p],
            instantiateDemandBuiltPart(rule, rule.premises[p], values, premiseText[p], kDemandBuiltTextCap));
        for (int32_t q = 0; q < p; ++q)
            if (equalSpans(premises[q], premises[p])) return DemandBuildResult::Diagonal;
    }
    for (int32_t p = 0; p < premiseN; ++p) {
        premiseScopes[p] = shallowestKnown(premises[p]);
        if (premiseScopes[p] == 0) {
            missingPremise = p;
            return DemandBuildResult::Untyped;
        }
        premiseIteration = std::max(premiseIteration, admissionPolicy.extractMaxIterationNumber(premises[p]));
    }
    if (premiseIteration > parameters.maxIterationNumberVariable) return DemandBuildResult::BeyondGeneration;
    NameId firing = ruleScope;
    for (int32_t p = 0; p < premiseN; ++p)
        if (mb.nameMap.ancLen(premiseScopes[p]) > mb.nameMap.ancLen(firing)) firing = premiseScopes[p];
    char headText[kDemandBuiltTextCap];
    const StrSpan head(headText,
        instantiateDemandBuiltPart(rule, rule.head, values, headText, kDemandBuiltTextCap));
    const NameId headId = mb.nameMap.lookup(head);
    if (!parameters.compressor_mode && headId != 0
        && gl::StatementKnowledge::ancestorKnown(mb, headId, firing, true))
        return DemandBuildResult::Known;

    // Levels: the premises' and the rule's runs (the rule's are its
    // carrier's), the non-derived tier excluded, sorted and unique.
    int levels[(kDemandBuiltPremiseCap + 1) * 64];
    int levelN = 0;
    const auto addLevels = [&](NameId id, NameId scope) {
        const int32_t run = mb.intStatementLevelsMap.lookup(packStatementKey(id, scope));
        if (run == 0) return;
        for (int32_t j = 0; j < mb.intStatementLevelsMap.runLen(run); ++j) {
            const int level = mb.intStatementLevelsMap.valueAt(run, j);
            if (level < 0) continue;
            assert(levelN < static_cast<int>(sizeof levels / sizeof levels[0])
                && "buildDemandedInstance: level run exceeds its capacity");
            levels[levelN++] = level;
        }
    };
    for (int32_t p = 0; p < premiseN; ++p) addLevels(mb.nameMap.lookup(premises[p]), premiseScopes[p]);
    const NameId carrierId = mb.nameMap.lookup(rule.carrierSpan());
    assert(carrierId != 0 && "buildDemandedInstance: the block knows the rule's carrier");
    addLevels(carrierId, ruleScope);
    std::sort(levels, levels + levelN);
    levelN = static_cast<int>(std::unique(levels, levels + levelN) - levels);

    // The incubator disintegration gate (D-29): an anchor block never
    // disintegrates a fired head, another block only when a matched premise
    // is its own derivation.
    bool doNotDisintegrate = false;
    if (parameters.incubator_mode && !parameters.ban_disintegration) {
        const StrSpan blockKey = mb.exprKeyView();
        const StrSpan anchorName(anchorInfo.name);
        bool anchorBlock = blockKey.len > anchorName.len + 1 && blockKey.ptr[0] == '('
            && std::memcmp(blockKey.ptr + 1, anchorName.ptr, static_cast<std::size_t>(anchorName.len)) == 0;
        bool localPremise = false;
        for (int32_t p = 0; p < premiseN && !localPremise; ++p)
            localPremise = mb.intLocalEncodedStatementsSet.contains(
                packStatementKey(mb.nameMap.lookup(premises[p]), premiseScopes[p]));
        doNotDisintegrate = anchorBlock || !localPremise;
    }

    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope frame(strings);
    // Scope names copied before the mail doors mint (I-3).
    const auto scopeName = [&](NameId scope) {
        const StrSpan view = mb.nameMap.decodeView(scope);
        return ScratchString::copyFrom(strings, view.ptr, view.len);
    };
    const ScratchString firingName = scopeName(firing);
    const ScratchString ruleScopeName = scopeName(ruleScope);
    OriginDep deps[kDemandBuiltPremiseCap + 1];
    deps[0] = OriginDep{ rule.textSpan(), StrSpan(ruleScopeName) };
    for (int32_t p = 0; p < premiseN; ++p) {
        const ScratchString name = scopeName(premiseScopes[p]);
        deps[1 + p] = OriginDep{ premises[p], StrSpan(name) };
    }
    std::sort(deps + 1, deps + 1 + premiseN, [](const OriginDep& a, const OriginDep& b) {
        const int expression = compareSpans(a.original, b.original);
        return expression != 0 ? expression < 0 : compareSpans(a.validity, b.validity) < 0;
    });
    insertInternalStatement(output, mb.nameMap, head, StrSpan(firingName), levels, levelN);
    if (parameters.trackHistory) {
        const int maxOrigins = parameters.compressor_mode
            ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr;
        addInternalMailOrigin(output, mb.originInterner, head, StrSpan(firingName),
            OriginTag::implication, deps, 1 + premiseN, maxOrigins);
    }
    setInternalDisintegrationSignal(output, mb.nameMap, head, StrSpan(firingName),
        doNotDisintegrate, false, premiseIteration + 1, false);
    return DemandBuildResult::Built;
}

/// @brief Offer every stored demand of a demand-built rule's shape to the
///        build again, once per burst.
///
/// @details
/// A demand filed before the block held the rule, or before an input was
/// typed, built nothing when it was filed; the eager rule would fire in the
/// burst the late rule or typing arrived. At the end of every burst's
/// algebra admission drain, each stored algebra and integration demand key
/// that matches a registered witness template and has a visible demand is
/// offered to `buildDemandedInstances`; a known instance builds nothing.
///
/// @param mb The owning block.
/// @return Nothing; writes the internal mail.
/// @invariant Heap-free; keys are visited in map order.
/// @see AdmissionMemory::drainAdmissionKeysAlgebra.
void RejectedSourceReplay::buildWaitingInstances(Memory& mb) {
    const DemandBuiltRuleRegistry& registry = DemandBuiltRuleRegistry::instance();
    const int32_t ruleN = registry.count();
    if (ruleN == 0) return;
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
    ScratchArena& strings = scratchArenas().forSlot(slot);
    const auto scan = [&](const auto& map, bool integration) {
        const int32_t count = map.count();
        for (int32_t id = 1; id <= count; ++id) {
            StrSpan keyView;
            StrSpan scopeView;
            decodeTemplateKeyView(map.keyAt(id), mb.templateInterner, mb.nameMap, keyView, scopeView);
            ScratchScope frame(strings);
            // Copies before the build mints (I-3); an integration key spells
            // its fixed arguments u_, the revival seams take the bare form.
            const ScratchString key = integration
                ? gl::DefinitionExpansion::removeUPrefixScratch(strings, keyView)
                : ScratchString::copyFrom(strings, keyView.ptr, keyView.len);
            bool shaped = false;
            StrSpan values[kDemandBuiltBoundCap];
            for (int32_t i = 0; i < ruleN && !shaped; ++i)
                shaped = matchDemandBuiltWitness(registry.at(i), StrSpan(key), values);
            if (!shaped) continue;
            const ScratchString scope = ScratchString::copyFrom(strings, scopeView.ptr, scopeView.len);
            if (!hasPotentialParkedDemand(mb, StrSpan(key), StrSpan(scope), integration)) continue;
            buildDemandedInstances(mb, StrSpan(key), StrSpan(scope));
        }
    };
    scan(mb.overallHashMemory.admissionMap, false);
    scan(mb.overallHashMemory.admissionMapIntegration, true);
    // Stored case demands of a deferred OR head's disjuncts: a demand filed
    // before the block held the rule, or before a typing, built nothing then
    // (D-597).
    if (!registry.hasOrRules()) return;
    const auto& ordis2 = mb.overallHashMemory.admissionMapOrdis2;
    const int32_t demandN = ordis2.count();
    for (int32_t id = 1; id <= demandN; ++id) {
        StrSpan keyView;
        StrSpan scopeView;
        decodeTemplateKeyView(ordis2.keyAt(id), mb.templateInterner, mb.nameMap, keyView, scopeView);
        if (!registry.hasOrTrigger(keyView, DemandBuiltTrigger::CaseDemand)) continue;
        buildDemandedOrCaseInstances(mb, keyView, scopeView);
    }
}

/// @brief Build the instances of the OR heads the staged events name
///        (D-597).
///
/// @details
/// The staged rows move to a scratch list first, so that the events a
/// premise arrival re-stages from `Memory::deferredOrWaiting` join the same
/// pass. A row neither known nor a goal ends its event. A known row that
/// matches a registered OR head's refutation template, or a goal that
/// matches its goal template, builds that instance into @p output, in
/// rule-text then template order; an instance whose premise is not known on
/// the row's chain waits under that premise's text, which its arrival
/// re-stages.
///
/// @param mb     The owning block.
/// @param output The running call's internal-mail output channel.
/// @return Nothing; writes @p output and the waiting index.
/// @invariant Heap-free (the slot's scratch arenas); the pending list is
///            empty on return.
/// @see StatementArrivals::stageDeferredOrEvent, buildDemandedInstance.
void RejectedSourceReplay::drainDeferredOrEvents(Memory& mb, ColdMail& output) {
    if (mb.pendingDeferredOrEvents.size() == 0) return;
    RT_SCOPE_HERE("SP_DEFERRED_OR_DRAIN");
    const DemandBuiltRuleRegistry& registry = DemandBuiltRuleRegistry::instance();
    int32_t order[kDemandBuiltRuleCap];
    const int32_t orN = registry.orderedOrRules(order);
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
    ScratchArena& arena = genScratchArenas().forSlot(slot);
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope frame(arena);
    DirtyState dirty = DirtyState::Clean;
    PagedVector<int64_t> keys(&arena, &dirty);
    for (int32_t i = 0; i < mb.pendingDeferredOrEvents.size(); ++i) keys.push_back(mb.pendingDeferredOrEvents[i]);
    mb.pendingDeferredOrEvents.clear();
    for (int32_t i = 0; i < keys.size(); ++i) {
        const int64_t key = keys[i];
        const NameId original = static_cast<NameId>(uint64_t(key) >> 32);
        const NameId validity = static_cast<NameId>(uint64_t(key) & 0xffffffffu);
        const bool known = mb.intKnownStatements.find(StatementKey{ original, validity }) != nullptr;
        const bool goal = mb.intToBeProved.lookup(key) != 0;
        if (!known && !goal) continue;
        // The events that waited for this row as their missing premise.
        const int32_t waitingRow = mb.deferredOrWaiting.empty()
            ? 0 : mb.deferredOrWaiting.lookup(static_cast<int32_t>(original));
        if (waitingRow != 0) {
            for (int32_t j = 0; j < mb.deferredOrWaiting.runLen(waitingRow); ++j)
                keys.push_back(mb.deferredOrWaiting.valueAt(waitingRow, j));
            mb.deferredOrWaiting.eraseSet(static_cast<int32_t>(original));
        }
        ScratchScope textFrame(strings);
        // Copies before the build mints (I-3).
        const StrSpan textView = mb.nameMap.decodeView(original);
        const ScratchString text = ScratchString::copyFrom(strings, textView.ptr, textView.len);
        const StrSpan scopeView = mb.nameMap.decodeView(validity);
        const ScratchString scope = ScratchString::copyFrom(strings, scopeView.ptr, scopeView.len);
        for (int32_t m = 0; m < orN; ++m) {
            const DemandBuiltRule& rule = registry.at(order[m]);
            for (int32_t t = 0; t < rule.triggerCount; ++t) {
                const DemandBuiltTrigger trigger = rule.triggerKinds[t];
                if (!(trigger == DemandBuiltTrigger::Refutation && known)
                    && !(trigger == DemandBuiltTrigger::Goal && goal)) continue;
                StrSpan values[kDemandBuiltBoundCap];
                if (!matchDemandBuiltTemplate(rule, t, StrSpan(text), values)) continue;
                int32_t missingPremise = -1;
                const DemandBuildResult result =
                    buildDemandedInstance(mb, rule, values, StrSpan(scope), output, missingPremise);
                if (result != DemandBuildResult::Untyped) continue;
                char premiseText[kDemandBuiltTextCap];
                const int32_t premiseLength = instantiateDemandBuiltPart(rule, rule.premises[missingPremise],
                    values, premiseText, kDemandBuiltTextCap);
                const NameId premiseId = mb.nameMap.encode(StrSpan(premiseText, premiseLength));
                mb.deferredOrWaiting.insertSorted(static_cast<int32_t>(premiseId), key);
            }
        }
    }
}

/// @brief Stage every known row that is an event of a newly held OR-headed
///        demand-built rule (D-597).
///
/// @details
/// Every known row with the shape of one of the rule's refutation templates
/// and every goal with the shape of its goal template
/// (`demandBuiltHasTrigger`) is staged for the next `drainDeferredOrEvents`,
/// which matches and builds. An operator-totality rule stages nothing.
///
/// @param mb       The owning block.
/// @param ruleText The rule text just recorded as held.
/// @return Nothing; appends to `Memory::pendingDeferredOrEvents`.
/// @invariant The rule is registered (asserted); mints nothing.
/// @see CompactRuleProcessor::installExpandedImplication.
void RejectedSourceReplay::stageExistingDeferredOrEvents(Memory& mb, StrSpan ruleText) {
    const DemandBuiltRule* rule = DemandBuiltRuleRegistry::instance().find(ruleText);
    assert(rule != nullptr && "stageExistingDeferredOrEvents: the deferred rule is registered");
    if (rule->kind != DemandBuiltKind::OrHead) return;
    const int32_t knownN = mb.intKnownStatements.count();
    for (int32_t id = 1; id <= knownN; ++id) {
        const StatementKey known = mb.intKnownStatements.decodeKey(id);
        if (!demandBuiltHasTrigger(*rule, mb.nameMap.decodeView(known.orig), DemandBuiltTrigger::Refutation))
            continue;
        mb.pendingDeferredOrEvents.push_back(packStatementKey(known.orig, known.validity));
    }
    const int32_t goalN = mb.intToBeProved.count();
    for (int32_t id = 1; id <= goalN; ++id) {
        const StatementKey goal = mb.intToBeProved.decodeKey(id);
        if (!demandBuiltHasTrigger(*rule, mb.nameMap.decodeView(goal.orig), DemandBuiltTrigger::Goal)) continue;
        mb.pendingDeferredOrEvents.push_back(packStatementKey(goal.orig, goal.validity));
    }
}

/// @brief Build the instances of the OR heads a case demand names
///        (D-597).
///
/// @details
/// A case-split demand for a disjunct (`admissionMapOrdis2`, ground
/// disjunct text) opens an OR only when that OR is deposited; a deferred OR
/// must therefore exist where the demand is. Every registered OR head whose
/// case-demand template matches @p groundText is built for the matched
/// values through `buildDemandedInstance`, in rule-text then template order,
/// into the running absorb's output channel when called inside an absorb
/// (`Memory::replayMailDuringAbsorb`, whose input clear would drop a
/// same-iteration deposit), else into the same-iteration channel. A refused
/// build ends here: the stored demand is offered again once per burst by
/// `buildWaitingInstances`.
///
/// @param mb           The owning block.
/// @param groundText   The demanded disjunct, ground text.
/// @param validityName The demand's scope.
/// @return Nothing; writes the internal mail.
/// @invariant Heap-free (the slot's scratch arena); copies its inputs before
///            the build mints (I-3).
/// @see revisitRejectedOrdis2, buildWaitingInstances.
void RejectedSourceReplay::buildDemandedOrCaseInstances(Memory& mb, StrSpan groundText, StrSpan validityName) {
    const DemandBuiltRuleRegistry& registry = DemandBuiltRuleRegistry::instance();
    if (!registry.hasOrTrigger(groundText, DemandBuiltTrigger::CaseDemand)) return;
    ColdMail& output = mb.replayMailDuringAbsorb != nullptr
        ? *mb.replayMailDuringAbsorb : mb.sameIterationInternalMail;
    int32_t order[kDemandBuiltRuleCap];
    const int32_t orN = registry.orderedOrRules(order);
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope frame(strings);
    const ScratchString text = ScratchString::copyFrom(strings, groundText.ptr, groundText.len);
    const ScratchString scope = ScratchString::copyFrom(strings, validityName.ptr, validityName.len);
    for (int32_t m = 0; m < orN; ++m) {
        const DemandBuiltRule& rule = registry.at(order[m]);
        for (int32_t t = 0; t < rule.triggerCount; ++t) {
            if (rule.triggerKinds[t] != DemandBuiltTrigger::CaseDemand) continue;
            StrSpan values[kDemandBuiltBoundCap];
            if (!matchDemandBuiltTemplate(rule, t, StrSpan(text), values)) continue;
            int32_t missingPremise = -1;
            buildDemandedInstance(mb, rule, values, StrSpan(scope), output, missingPremise);
        }
    }
}

/// @brief Bind inputs for rejected source replay.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param admissionPolicy Borrowed AdmissionPolicy interface or state; must outlive this component.
/// @param anchorInfo Borrowed anchor description of the batch; must outlive this component.
/// @param canonicalTemplates Borrowed CanonicalTemplateMatcher interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param rejectedSources Borrowed RejectedSourceMemory interface or state; must outlive this component.
/// @param admissionMemory Borrowed AdmissionMemory interface or state; must outlive this component.
/// @param caseLedger Borrowed CaseLedger interface or state; must outlive this component.
/// @param proofHistory Borrowed ProofHistory interface or state; must outlive this component.
/// @param watchProcessor Borrowed WatchProcessor interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
RejectedSourceReplay::RejectedSourceReplay(AdmissionPolicy& admissionPolicy, const ce::AnchorInfo& anchorInfo, CanonicalTemplateMatcher& canonicalTemplates, Canonicalizer& canonicalizer, const CompiledDefinitionRegistry& definitions, const ProverParameters& parameters, RejectedSourceMemory& rejectedSources, AdmissionMemory& admissionMemory, CaseLedger& caseLedger, ProofHistory& proofHistory, WatchProcessor& watchProcessor) : admissionPolicy(admissionPolicy), anchorInfo(anchorInfo), canonicalTemplates(canonicalTemplates), canonicalizer(canonicalizer), definitions(definitions), parameters(parameters), rejectedSources(rejectedSources), admissionMemory(admissionMemory), caseLedger(caseLedger), proofHistory(proofHistory), watchProcessor(watchProcessor) {}

/// @brief Wake the `rejectedMap` cohorts an admission key gain can
        ///        satisfy — the exact-scope cohort (erased, revived at the
        ///        key's scope) and every cohort parked at a strict ANCESTOR
        ///        scope (kept, released into the key's scope).
        ///
        /// @details
        /// Two paths on one key gain, both gated by the same two revival
        /// budgets (secondary-variable count and iteration depth of
        /// `markedExpr`):
        /// 1. **Exact scope.** The cohort parked under
        ///    `(markedExpr, validityName)` is snapshotted, erased, and every
        ///    record mailed on `sameIterationInternalMail` at `validityName`
        ///    through the `equality1` self-source revival door — unchanged.
        /// 2. **Ancestor scopes.** @ref releaseAncestorRejected2 — a park at
        ///    a strict ancestor of `validityName` holds a product that is
        ///    valid at `validityName` too; the demand lives only here, so the
        ///    product is released INTO `validityName` and the ancestor's park
        ///    is left in place for the next scope that asks
        ///    (`D-341`).
        ///
        /// @param markedExpr   The admission template that gained a key
        ///                     (bare marker form).
        /// @param memoryBlock  The owning LB.
        /// @param validityName The key's validity scope.
        /// @return (void)
        /// @invariant Probes never mint (I-89); the `revisitInProgress` guard
        ///            covers the exact path only — the ancestor path is
        ///            mail-only and cannot re-enter.
        /// @see releaseAncestorRejected2, revisitRejectedIntegration2 — the
        ///      integration twin; drainAdmissionKeysAlgebra,
        ///      updateAdmissionMapRecursion, applyEquivalenceClassToAdmissionMap
        ///      and the D-243 park-side probe — the seams.
void RejectedSourceReplay::revisitRejected2(StrSpan key, Memory& mb, StrSpan validityName) {
    if (queueCanonicalWake(mb,key,validityName,1)) return;
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
    ScratchArena& arena = genScratchArenas().forSlot(slot);
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope frame(arena);
    ScratchScope stringFrame(strings);
    const ScratchString ownedKey = ScratchString::copyFrom(strings, key.ptr, key.len);
    const ScratchString ownedValidity = ScratchString::copyFrom(strings, validityName.ptr, validityName.len);
    key = StrSpan(ownedKey); validityName = StrSpan(ownedValidity);
    if (!hasPotentialParkedDemand(mb, key, validityName, false)) return;
    buildDemandedInstances(mb, key, validityName);
    StrSpan scopes[ExecutionParameters::MAX_SCOPE_DEPTH];
    scopes[0] = validityName; const int32_t count = 1;
    for (int32_t i = 0; i < count; ++i) {
        DirtyState dirty = DirtyState::Clean;
        PagedVector<int64_t> matches(&arena, &dirty);
        canonicalTemplates.matchingCanonicalTemplates(mb, mb.overallHashMemory.rejectedMap,
            mb.lbMemory.canonicalUses.algebraParkMatches, key, mb.nameMap.lookup(scopes[i]),
            mb.nameMap.lookup(validityName), matches, arena);
        for (int32_t k = 0; k < matches.size(); ++k) {
            const auto records = snapshotRejectedRun(mb.overallHashMemory.rejectedMap, matches[k], arena);
            for (int32_t r = 0; r < records.count; ++r)
                queueParkedExistence(mb, key, records.views[r], scopes[i], validityName, false);
        }
    }
    releaseAncestorRejected2(key, mb, validityName);
}

/// @brief Record an unchanged expression inherited into a descendant scope.
///
/// @details The zoom in row has exactly one dependency: the same expression
/// at its source scope. Parent history is neither read nor copied. The row is
/// written to the internal mail, for this block, and to the outgoing mail:
/// the released expression never rides the delta `fillMailOut` walks, while
/// what is derived from it reaches the descendant blocks, which must resolve
/// its citation (I-64 exception). The normal history writer retains its
/// duplicate and capacity policy; this operation returns no decision to the
/// reasoning algorithm.
///
/// @param mb Owning logic block, internal-mail and outgoing-mail history sink.
/// @param expression Released expression, unchanged by the transfer.
/// @param sourceValidity Exact source scope of the ancestor park.
/// @param targetValidity Exact descendant scope receiving the expression.
/// @return (void)
/// @invariant Only history is written; source derivations and release state are untouched.
/// @see releaseParkedProductIntoScope, addInternalMailOrigin.
void RejectedSourceReplay::recordZoomInOrigin(Memory& mb,
    StrSpan expression, StrSpan sourceValidity, StrSpan targetValidity) {
    const OriginDep source[1] = {{ expression, sourceValidity }};
    const int maxOrigins = parameters.compressor_mode
        ? parameters.compressor_max_origins_per_expr
        : parameters.max_origin_per_expr;
    addInternalMailOrigin(mb.sameIterationInternalMail, mb.originInterner,
        expression, targetValidity, OriginTag::zoomIn, source, 1, maxOrigins);
    mb.addMailOutOrigin(expression, targetValidity, OriginTag::zoomIn, source, 1, maxOrigins);
}

/// @brief Mail one parked product (with its sibling cohort) into a
        ///        descendant scope, carrying its park-scope provenance.
        ///
        /// @details
        /// The per-record door of the ancestor-park release, shared by the
        /// algebra and integration twins. Known-skip first: a product already
        /// known at `targetValidity` or at one of its ancestors
        /// (`ancestorKnown`, self included) is not released — the park stays
        /// for a later scope, and re-mailing would only feed the Site F
        /// refusal. Otherwise the product and every sibling are deposited on
        /// `sameIterationInternalMail` at `targetValidity` with the caller's
        /// level run. Optional history maintenance records one zoom in row
        /// per expression at the target scope, citing that same expression at
        /// parkValidity. Parent history is neither read nor copied, and never
        /// decides whether the product or siblings are released.
        ///
        /// @param mb             The owning LB.
        /// @param product        The parked concrete constituent.
        /// @param siblings       Its sibling cohort (positional).
        /// @param siblingCount   Sibling count.
        /// @param levels         Ascending-unique level run for every deposit;
        ///                       may be null only when `levelCount` is 0.
        /// @param levelCount     Level count.
        /// @param parkValidity   The scope the cohort is parked at (the
        ///                       provenance source).
        /// @param targetValidity The demand's scope (the deposit scope); must
        ///                       be interned.
        /// @param gArena         Caller scratch arena; unchanged by history recording.
        /// @return True iff the cohort was mailed; false on the known-skip.
        /// @invariant Parent parks are retained; release has no history prerequisite.
        /// @see releaseAncestorRejected2, releaseAncestorRejectedIntegration2
        ///      — the callers; recordZoomInOrigin;
        ///      ancestorKnown.
bool RejectedSourceReplay::releaseParkedProductIntoScope(Memory& mb,
    StrSpan product,
    const StrSpan* siblings, int32_t siblingCount,
    const int* levels, int32_t levelCount,
    StrSpan parkValidity,
    StrSpan targetValidity,
    ScratchArena& gArena) {
    (void)gArena;
    // Known-skip: the product is already visible at the target scope (known
    // there or at one of its ancestors) — this demand needs nothing and the
    // park stays for a later scope. Non-minting: an un-interned product text
    // cannot be known anywhere.
    const NameId targetVid = mb.nameMap.lookup(targetValidity);
    assert(targetVid != 0
        && "releaseParkedProductIntoScope: the demand scope is interned (its key was minted)");
    const NameId productId = mb.nameMap.lookup(product);
    if (productId != 0
        && gl::StatementKnowledge::ancestorKnown(mb, productId, targetVid, /*includeSelf=*/true)) {
        return false;
    }

    insertInternalStatement(mb.sameIterationInternalMail, mb.nameMap,
        product, targetValidity, levels, levelCount);
    for (int32_t i = 0; i < siblingCount; ++i) {
        insertInternalStatement(mb.sameIterationInternalMail, mb.nameMap,
            siblings[i], targetValidity, levels, levelCount);
    }

    mb.mutatedThisBurst = true;
    if (parameters.trackHistory) {
        recordZoomInOrigin(mb, product, parkValidity, targetValidity);
        for (int32_t i = 0; i < siblingCount; ++i) {
            recordZoomInOrigin(mb, siblings[i], parkValidity, targetValidity);
        }
    }

    return true;
}

/// @brief Release into `validityName` every `rejectedMap` cohort
        ///        parked under `markedExpr` at a strict ancestor scope of
        ///        `validityName`; the ancestor parks stay.
        ///
        /// @details
        /// Walks the owned ancestor copies from @ref copyStrictAncestorSpans
        /// (root-first, deterministic), probes `rejectedMap` at each
        /// `(markedExpr, ancestor)` key without minting, snapshots a hit's
        /// run onto gen-scratch and hands every record to
        /// @ref releaseParkedProductIntoScope after the exact-scope revival's
        /// per-record witness-depth gate (`maxIterationNumberVariable` on the
        /// marker-slot witness). Nothing is erased: the park at the ancestor
        /// remains the supply for any other descendant scope — or the
        /// ancestor itself — that later demands the same template. The
        /// caller (@ref revisitRejected2) has already applied the key-level
        /// budget gates.
        ///
        /// @param markedExpr   The admission template that gained a key
        ///                     (bare marker form).
        /// @param memoryBlock  The owning LB.
        /// @param validityName The key's scope — the release target.
        /// @return (void)
        /// @invariant Never writes `rejectedMap` (I-37's writer set is
        ///            unchanged); probes never mint (I-89).
        /// @see revisitRejected2 — the caller; releaseParkedProductIntoScope —
        ///      the per-record door; copyStrictAncestorSpans.
void RejectedSourceReplay::releaseAncestorRejected2(StrSpan key, Memory& mb, StrSpan validityName) {
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
    ScratchArena& arena = genScratchArenas().forSlot(slot);
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope frame(arena);
    ScratchScope stringFrame(strings);
    const ScratchString ownedKey = ScratchString::copyFrom(strings, key.ptr, key.len);
    const ScratchString ownedValidity = ScratchString::copyFrom(strings, validityName.ptr, validityName.len);
    key = StrSpan(ownedKey); validityName = StrSpan(ownedValidity);
    StrSpan scopes[ExecutionParameters::MAX_SCOPE_DEPTH];
    const int32_t count = gl::ScopeSyntax::copyStrictAncestorSpans(mb, validityName, scopes, strings);
    for (int32_t i = 0; i < count; ++i) {
        DirtyState dirty = DirtyState::Clean;
        PagedVector<int64_t> matches(&arena, &dirty);
        canonicalTemplates.matchingCanonicalTemplates(mb, mb.overallHashMemory.rejectedMap,
            mb.lbMemory.canonicalUses.algebraParkMatches, key, mb.nameMap.lookup(scopes[i]),
            mb.nameMap.lookup(validityName), matches, arena);
        for (int32_t k = 0; k < matches.size(); ++k) {
            const auto records = snapshotRejectedRun(mb.overallHashMemory.rejectedMap, matches[k], arena);
            for (int32_t r = 0; r < records.count; ++r)
                queueParkedExistence(mb, key, records.views[r], scopes[i], validityName, false);
        }
    }

}

/// @brief Full per-LB registry reset for a PARKED or statement about
        ///        to be re-deposited — the ordis revival's "un-know".
        ///
        /// @details
        /// The local-statement sibling of
        /// @ref resetResentExpressionRegistries: the parked or statement was
        /// consumed flat at its first arrival and is a KNOWN (and local)
        /// statement of the LB, so a plain re-deposit would be dropped by
        /// the Site F ancestor dedup before the or consumption re-runs.
        /// This helper erases the statement's rows from every dedup
        /// registry INCLUDING the local statement vectors (which
        /// `resetResentExpressionRegistries`'s non-local contract excludes):
        /// `intStatementLevelsMap`, `intKnownStatements` (the known bit's
        /// row — the documented, maintainer-approved exception to the
        /// OR-only flag discipline of I-85, precedented by the CE
        /// teardown's full paired reset per I-58),
        /// `intEncodedStatements`, `intLocalEncodedStatements`, and
        /// `intLocalEncodedStatementsDelta`. The
        /// `intLocalEncodedStatementsSet` membership is left in place: the
        /// re-absorbed statement re-registers as local, and the set is a
        /// monotone locality index whose transient over-approximation is
        /// harmless (the row-bearing registries are the gates). History
        /// (`exprOriginMap`) is NOT touched (I-44, Rule 16).
        ///
        /// @param mb           Owning LB.
        /// @param original     The parked or statement's clean text.
        /// @param validityName The cohort parent scope.
        /// @return (void)
        /// @invariant Called only by @ref revisitRejectedOrdis, immediately
        ///            before the mail re-deposit of the same statement.
        /// @see resetResentExpressionRegistries — the non-local sibling.
void RejectedSourceReplay::resetParkedOrStatementRegistries(Memory& mb,
    StrSpan original,
    StrSpan validityName) {
    // encode (not lookup) — mirrors resetResentExpressionRegistries: the
    // statement and scope are both interned already (the statement was
    // deposited and the cohort parked there), so these are id fetches.
    const NameId origId = mb.nameMap.encode(original);
    const NameId valId = mb.nameMap.encode(validityName);
    mb.intStatementLevelsMap.eraseSet(packStatementKey(origId, valId));
    const bool wasKnown =
        mb.intKnownStatements.find(StatementKey{ origId, valId }) != nullptr;
    gl::StatementRegistryOperations::eraseKnownStatementKey(mb, origId, valId);
    watchProcessor.clearNegationWatchFor(mb, origId, valId, wasKnown);
    // Stored rows are canonical-pipeline encodings, so the id pair is the
    // full identity; erase back to front to keep indices valid. The local
    // vectors are included — the parked statement is a LOCAL statement (the
    // non-local resetResentExpressionRegistries contract excludes them).
    for (int32_t i = mb.intEncodedStatements.size(); i-- > 0; ) {
        if (mb.intEncodedStatements[i].originalId == origId
            && mb.intEncodedStatements[i].validityId == valId) {
            mb.intEncodedStatements.erase(i);
        }
    }
    for (int32_t i = mb.intLocalEncodedStatements.size(); i-- > 0; ) {
        if (mb.intLocalEncodedStatements[i].originalId == origId
            && mb.intLocalEncodedStatements[i].validityId == valId) {
            mb.intLocalEncodedStatements.erase(i);
        }
    }
    for (int32_t i = mb.intLocalEncodedStatementsDelta.size(); i-- > 0; ) {
        if (mb.intLocalEncodedStatementsDelta[i].originalId == origId
            && mb.intLocalEncodedStatementsDelta[i].validityId == valId) {
            mb.intLocalEncodedStatementsDelta.erase(i);
        }
    }
    // The external staging is a view of the registry for this burst's
    // request generation: a row whose registry entry is gone must leave
    // it too, or request generation keeps a mandatory ingredient that
    // can no longer be a premise.
    for (int32_t i = mb.intExternalStatements.size(); i-- > 0; ) {
        if (mb.intExternalStatements[i].originalId == origId
            && mb.intExternalStatements[i].validityId == valId) {
            mb.intExternalStatements.erase(i);
        }
    }
    // The revived statement must re-enter the door as the representative:
    // drop its processed-or row (idempotent — a second park key of the
    // same or finds it already cleared).
    caseLedger.clearProcessedOr(mb, origId, valId);
}

/// @brief Revive parked or-cohorts whose product template just
        ///        gained an admission key — the `rejectedMapOrdis` mirror of
        ///        @ref revisitRejected2.
        ///
        /// @details
        /// Hooked at every seam where the algebra `admissionMap` gains keys
        /// (the drain step after `revisitRejected2`, the Writer-B inline
        /// insert, the recursion propagation insert, and the equi-class
        /// admission re-key), firing for TAGGED and untagged gains alike —
        /// the cohort probe reads both. Mechanics mirror `revisitRejected2`:
        /// non-minting template probe (a never-interned template was never
        /// parked — a defined miss), a DEDICATED re-entrancy guard
        /// (`ordisRevisitInProgress`), copy-before-erase snapshot of the
        /// parked run onto gen-scratch, whole-cohort `eraseBlobIf`, then a
        /// per-record MAIL re-deposit of the parked or statement on
        /// `sameIterationInternalMail` — the absorb re-runs the FULL or
        /// consumption (fresh probe over all standing demands, standing
        /// starter tie rule, K rules deduped, fresh-cohort guard against
        /// double-open; a revival may legitimately re-park, the map entry
        /// already erased). Because the parked statement is a KNOWN local
        /// statement, each re-deposit is preceded by
        /// @ref resetParkedOrStatementRegistries — the maintainer-approved
        /// un-know that lets the deposit pass the Site F ancestor dedup.
        ///
        /// @param markedExpr   The admission template that gained a key
        ///                     (bare marker form).
        /// @param memoryBlock  The owning LB.
        /// @param validityName The key's validity scope (exact-validity
        ///                     matching; widening deferred by design).
        /// @return (void)
        /// @invariant Never inserts into `rejectedMapOrdis`; re-park happens
        ///            only through the re-consumption's own park path.
        /// @see revisitRejected2 — the discipline template;
        ///      drainAdmissionKeysAlgebra, updateAdmissionMap,
        ///      updateAdmissionMapRecursion,
        ///      applyEquivalenceClassToAdmissionMap — the four seams.
void RejectedSourceReplay::revisitRejectedOrdis(StrSpan markedExpr,
    Memory& memoryBlock,
    StrSpan validityName, int64_t selectedKey) {
    if (selectedKey==0 && queueCanonicalWake(memoryBlock,markedExpr,validityName,4)) return;
    auto& rmo = memoryBlock.overallHashMemory.rejectedMapOrdis;

    // Non-minting probe — a never-interned template was never parked; a
    // key without a parked run is a defined miss.
    if (selectedKey == 0) {
        const unsigned slot = gl::WorkerContext::coreId() >= 0 ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : genScratchArenas().slotCount() - 1;
        ScratchArena& arena = genScratchArenas().forSlot(slot);
        ScratchArena& strings = scratchArenas().forSlot(slot);
        ScratchScope frame(arena), textFrame(strings);
        const ScratchString text = ScratchString::copyFrom(strings, markedExpr.ptr, markedExpr.len);
        const ScratchString scope = ScratchString::copyFrom(strings, validityName.ptr, validityName.len);
        DirtyState dirty = DirtyState::Clean;
        PagedVector<int64_t> matches(&arena, &dirty);
        const NameId scopeId = memoryBlock.nameMap.lookup(validityName);
        assert(scopeId != 0);
        canonicalTemplates.matchingCanonicalTemplates(memoryBlock, memoryBlock.overallHashMemory.rejectedMapOrdis,
            memoryBlock.lbMemory.canonicalUses.ordisParkMatches, StrSpan(text), scopeId, scopeId, matches, arena);
        for (int32_t i = 0; i < matches.size(); ++i)
            revisitRejectedOrdis(StrSpan(text), memoryBlock, StrSpan(scope), matches[i]);
        return;
    }
    const int64_t revisitPk = selectedKey;
    if (rmo.lookup(revisitPk) == 0) {
        return;
    }

    // DEDICATED re-entrancy guard (never shared with revisitInProgress — a
    // concurrent general revisit of the same packed key must not swallow an
    // ordis wake).
    if (memoryBlock.overallHashMemory.ordisRevisitInProgress.contains(revisitPk)) {
        return;
    }
    memoryBlock.overallHashMemory.ordisRevisitInProgress.mint(revisitPk);

    // Snapshot the parked cohort as VERBATIM blob copies on the gen scratch
    // arena — copy BEFORE erase (the eraseBlobIf restructures the blob
    // pool; the revisitRejected2 discipline). Erasing the whole key first
    // also makes a legitimate re-park by the re-consumption below start
    // from a clean slate.
    const unsigned ordisSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(ordisSlot);
    const ArenaOffset mark = gArena.cursor();
    const RejectedOrdisRunSnapshot snap =
        snapshotRejectedOrdisRun(rmo, revisitPk, gArena);
    rmo.eraseBlobIf([revisitPk](int64_t k) { return k == revisitPk; });

    for (int32_t r = 0; r < snap.count; ++r) {
        const RejectedOrdisValueBlobView& v = snap.views[r];
        // decodeView is I-3-safe across the loop: the sinks below mint
        // nameMap + originInterner, never valueInterner, and the walked
        // blob bytes are gen-arena copies.
        const StrSpan parkedStmt =
            memoryBlock.valueInterner.decodeView(v.orStatementId());
        const int32_t lvN = v.levelCount();
        int lvRun[256];
        assert(lvN <= 256 && "parked ordis level run exceeds lvRun");
        for (int32_t j = 0; j < lvN; ++j) lvRun[j] = v.levelAt(j);

        // The un-know (maintainer-approved): the parked statement is a
        // known local statement, so without this reset the Site F ancestor
        // dedup would drop the re-deposit before the or consumption
        // re-runs.
        resetParkedOrStatementRegistries(memoryBlock, parkedStmt,
                                         validityName);

        // Re-deposit; the absorb re-runs the full or consumption (fresh
        // probe, standing starter tie rule, K rules deduped, fresh-cohort
        // guard). pre == post self-source equality1 mirrors the
        // revisitRejected2 revival door; the statement's foundation origin
        // already sits in exprOriginMap and D-49's cap-full preference
        // protects it.
        memoryBlock.mutatedThisBurst = true;
        insertInternalStatement(memoryBlock.sameIterationInternalMail,
            memoryBlock.nameMap, parkedStmt, validityName, lvRun, lvN);
        if (parameters.trackHistory) {
            OriginDep dep[1] = { { parkedStmt, validityName } };
            addInternalMailOrigin(memoryBlock.sameIterationInternalMail,
                memoryBlock.originInterner, parkedStmt, validityName,
                OriginTag::equality1, dep, 1,
                (parameters.compressor_mode
                     ? parameters.compressor_max_origins_per_expr
                     : parameters.max_origin_per_expr));
        }
    }

    gArena.popTo(mark);

    memoryBlock.overallHashMemory.ordisRevisitInProgress.erase(revisitPk);
}

/// @brief Wake the or heads filed in `rejectedMapOrdis2` under one
        ///        ground-text key — the ordis2 pair's rendezvous, fired by
        ///        the demand drain (D-267).
        ///
        /// @details
        /// Twin of @ref revisitRejectedOrdis pointed at the SECOND park
        /// index, whose keys are eligible disjuncts' clean GROUND texts —
        /// the demand map's own key language, so the wake is a plain key
        /// rendezvous with no product-template translation. It is called by
        /// the two demand writers, `drainAdmissionKeysOrdis2` and
        /// `fileHypothesisOrDemand` (the induction hypothesis's missing
        /// expression, D-575), and by
        /// `drainCanonicalWakes` for a wake deferred during canonical
        /// maintenance. The algebra key-gain seams keep waking only the old map — different key
        /// language; a cross-probe would double-wake operator-application
        /// disjuncts whose old-map product template IS the disjunct text.
        /// Mechanics mirror the twin: non-minting template probe (a
        /// never-interned text was never filed — a defined miss), the
        /// DEDICATED `ordis2RevisitInProgress` guard, copy-before-erase
        /// snapshot onto gen-scratch, whole-key `eraseBlobIf`, per-record
        /// @ref resetParkedOrStatementRegistries un-know + mail re-deposit
        /// on `sameIterationInternalMail`, or on the running absorb's output
        /// channel (`Memory::replayMailDuringAbsorb`) when the wake comes
        /// from inside an absorb, whose own input clear would drop it (the
        /// `queueParkedExistence` routing); the absorb re-runs the FULL or
        /// consumption, whose route-(c) probe reads the freshly-drained
        /// demand and opens while preserving it for other sources. Entries
        /// under other keys and in `rejectedMapOrdis` stay; the cohort
        /// bootstrap guard makes stale wakes idempotent. A route-(c) probe
        /// removes a satisfied demand, and scope wipe cleans at discharge.
        ///
        /// @param groundText   The demand's ground compound premise text
        ///                     (polarity verbatim, I-175).
        /// @param memoryBlock  The owning LB.
        /// @param validityName The demand key's validity scope
        ///                     (exact-validity matching, like the twin).
        /// @return (void)
        /// @invariant Never inserts into `rejectedMapOrdis2`; re-park
        ///            happens only through the re-consumption's own park
        ///            path.
        /// @see revisitRejectedOrdis — the discipline template;
        ///      drainAdmissionKeysOrdis2, AdmissionMemory::fileHypothesisOrDemand,
        ///      drainCanonicalWakes — the calling seams.
void RejectedSourceReplay::revisitRejectedOrdis2(StrSpan groundText,
    Memory& memoryBlock,
    StrSpan validityName, int64_t selectedKey) {
    if (selectedKey==0 && queueCanonicalWake(memoryBlock,groundText,validityName,8)) return;
    auto& rmo2 = memoryBlock.overallHashMemory.rejectedMapOrdis2;

    // Non-minting probe — a never-interned ground text was never filed; a
    // key without a parked run is a defined miss (a demand with no supply
    // waits in admissionMapOrdis2 for a later deposit's route-(c) probe).
    if (selectedKey == 0) {
        const unsigned slot = gl::WorkerContext::coreId() >= 0 ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : genScratchArenas().slotCount() - 1;
        ScratchArena& arena = genScratchArenas().forSlot(slot);
        ScratchArena& strings = scratchArenas().forSlot(slot);
        ScratchScope frame(arena), textFrame(strings);
        const ScratchString text = ScratchString::copyFrom(strings, groundText.ptr, groundText.len);
        const ScratchString scope = ScratchString::copyFrom(strings, validityName.ptr, validityName.len);
        DirtyState dirty = DirtyState::Clean;
        PagedVector<int64_t> matches(&arena, &dirty);
        const NameId scopeId = memoryBlock.nameMap.lookup(validityName);
        assert(scopeId != 0);
        canonicalTemplates.matchingCanonicalTemplates(memoryBlock, memoryBlock.overallHashMemory.rejectedMapOrdis2,
            memoryBlock.lbMemory.canonicalUses.ordis2ParkMatches, StrSpan(text), scopeId, scopeId, matches, arena);
        for (int32_t i = 0; i < matches.size(); ++i)
            revisitRejectedOrdis2(StrSpan(text), memoryBlock, StrSpan(scope), matches[i]);
        // A deferred OR with this disjunct is built where the demand is
        // (D-597).
        buildDemandedOrCaseInstances(memoryBlock, StrSpan(text), StrSpan(scope));
        return;
    }
    const int64_t revisitPk = selectedKey;
    if (rmo2.lookup(revisitPk) == 0) {
        return;
    }

    // DEDICATED re-entrancy guard (never shared with ordisRevisitInProgress
    // — a concurrent old-map wake of the same packed key must not swallow
    // an ordis2 wake).
    if (memoryBlock.overallHashMemory.ordis2RevisitInProgress.contains(revisitPk)) {
        return;
    }
    memoryBlock.overallHashMemory.ordis2RevisitInProgress.mint(revisitPk);

    // Snapshot the filed cohort as VERBATIM blob copies on the gen scratch
    // arena — copy BEFORE erase (the eraseBlobIf restructures the blob
    // pool; the revisitRejectedOrdis discipline). Erasing the whole key
    // first also makes a legitimate re-park by the re-consumption below
    // start from a clean slate; the or's entries under OTHER disjunct keys
    // (and in rejectedMapOrdis) stay — a stale later wake re-deposits an
    // already-open or, which the cohort bootstrap guard turns into a no-op,
    // and scope wipe clears the rest at discharge.
    const unsigned ordisSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(ordisSlot);
    const ArenaOffset mark = gArena.cursor();
    const RejectedOrdisRunSnapshot snap =
        snapshotRejectedOrdisRun(rmo2, revisitPk, gArena);
    rmo2.eraseBlobIf([revisitPk](int64_t k) { return k == revisitPk; });
    // Inside an absorb the same-iteration channel may be the one being
    // drained, and its clear would drop the re-deposit: replay through the
    // absorb's output channel, as queueParkedExistence does.
    ColdMail& replayMail = memoryBlock.replayMailDuringAbsorb != nullptr
        ? *memoryBlock.replayMailDuringAbsorb : memoryBlock.sameIterationInternalMail;

    for (int32_t r = 0; r < snap.count; ++r) {
        const RejectedOrdisValueBlobView& v = snap.views[r];
        // decodeView is I-3-safe across the loop: the sinks below mint
        // nameMap + originInterner, never valueInterner, and the walked
        // blob bytes are gen-arena copies.
        const StrSpan parkedStmt =
            memoryBlock.valueInterner.decodeView(v.orStatementId());
        const int32_t lvN = v.levelCount();
        int lvRun[256];
        assert(lvN <= 256 && "filed ordis2 level run exceeds lvRun");
        for (int32_t j = 0; j < lvN; ++j) lvRun[j] = v.levelAt(j);

        // The un-know (the I-178 exception, same as the old-map wake): the
        // parked statement is a known local statement, so without this
        // reset the Site F ancestor dedup would drop the re-deposit before
        // the or consumption re-runs.
        resetParkedOrStatementRegistries(memoryBlock, parkedStmt,
                                         validityName);

        // Re-deposit; the absorb re-runs the full or consumption, whose
        // route-(c) probe now reads the freshly-drained demand entry and
        // opens (consuming it). pre == post self-source equality1 mirrors
        // the revisitRejectedOrdis revival door.
        memoryBlock.mutatedThisBurst = true;
        insertInternalStatement(replayMail,
            memoryBlock.nameMap, parkedStmt, validityName, lvRun, lvN);
        if (parameters.trackHistory) {
            OriginDep dep[1] = { { parkedStmt, validityName } };
            addInternalMailOrigin(replayMail,
                memoryBlock.originInterner, parkedStmt, validityName,
                OriginTag::equality1, dep, 1,
                (parameters.compressor_mode
                     ? parameters.compressor_max_origins_per_expr
                     : parameters.max_origin_per_expr));
        }
    }

    gArena.popTo(mark);

    memoryBlock.overallHashMemory.ordis2RevisitInProgress.erase(revisitPk);
    // The shared demand stays available for every queued source.
    // A route-(c) probe removes it only after the fact becomes known.
}

// Integration-side counterpart to revisitRejected2. Walks
// rejectedMapIntegration[markedKey, validityName] and, for each stored
// entry, emits the constituent + siblings onto sameIterationInternalMail so the next
// hashburst re-runs the full disintegration pipeline at the constituent's
// original validity. Unlike revisitRejected2:
//   * no addExprToMemoryBlock call (mailIn-only revival, linear),
//   * no revisitInProgress guard (no re-entry through addStatement).
// The rejectedMapIntegration entry itself IS erased — the rejection is
// resolved (the constituents have been handed to the revival channel).
// A cohort parked at a strict ANCESTOR of validityName is released into
// validityName and KEPT (releaseAncestorRejectedIntegration2) — the same
// two-path shape as revisitRejected2.
void RejectedSourceReplay::revisitRejectedIntegration2(StrSpan key, Memory& mb, StrSpan validityName) {
    if (queueCanonicalWake(mb,key,validityName,2)) return;
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
    ScratchArena& arena = genScratchArenas().forSlot(slot);
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope frame(arena);
    ScratchScope stringFrame(strings);
    const ScratchString ownedKey = ScratchString::copyFrom(strings, key.ptr, key.len);
    const ScratchString ownedValidity = ScratchString::copyFrom(strings, validityName.ptr, validityName.len);
    key = StrSpan(ownedKey); validityName = StrSpan(ownedValidity);
    if (!hasPotentialParkedDemand(mb, key, validityName, true)) return;
    buildDemandedInstances(mb, key, validityName);
    StrSpan scopes[ExecutionParameters::MAX_SCOPE_DEPTH];
    scopes[0] = validityName; const int32_t count = 1;
    for (int32_t i = 0; i < count; ++i) {
        DirtyState dirty = DirtyState::Clean;
        PagedVector<int64_t> matches(&arena, &dirty);
        canonicalTemplates.matchingCanonicalTemplates(mb, mb.overallHashMemory.rejectedMapIntegration,
            mb.lbMemory.canonicalUses.integrationParkMatches, key, mb.nameMap.lookup(scopes[i]),
            mb.nameMap.lookup(validityName), matches, arena);
        for (int32_t k = 0; k < matches.size(); ++k) {
            const auto records = snapshotRejectedIntegrationRun(mb.overallHashMemory.rejectedMapIntegration, matches[k], arena);
            for (int32_t r = 0; r < records.count; ++r)
                queueParkedExistence(mb, key, records.views[r], scopes[i], validityName, true);
        }
    }
    releaseAncestorRejectedIntegration2(key, mb, validityName);
}

/// @brief Release into `validityName` every `rejectedMapIntegration`
        ///        cohort parked under `markedKey` at a strict ancestor scope
        ///        of `validityName`; the ancestor parks stay.
        ///
        /// @details
        /// The integration twin of @ref releaseAncestorRejected2: owned
        /// ancestor copies (@ref copyStrictAncestorSpans), a non-minting
        /// `rejectedMapIntegration` probe per `(markedKey, ancestor)`, a
        /// gen-scratch snapshot of each hit, and one
        /// @ref releaseParkedProductIntoScope call per record — constituent
        /// plus siblings, with the compound's level run looked up at the PARK
        /// scope (where the compound is known), exactly as the exact-scope
        /// revival looks it up at its own scope. No budget gates (the
        /// integration revival has none) and nothing erased.
        ///
        /// @param markedKey    The bare marker-form integration template that
        ///                     gained a key.
        /// @param memoryBlock  The owning LB.
        /// @param validityName The key's scope — the release target.
        /// @return (void)
        /// @invariant Never writes `rejectedMapIntegration` (I-37); probes
        ///            never mint (I-89); I-22 untouched (no admission entry
        ///            is consumed).
        /// @see revisitRejectedIntegration2 — the caller;
        ///      releaseParkedProductIntoScope — the per-record door.
void RejectedSourceReplay::releaseAncestorRejectedIntegration2(StrSpan key, Memory& mb, StrSpan validityName) {
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
    ScratchArena& arena = genScratchArenas().forSlot(slot);
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope frame(arena);
    ScratchScope stringFrame(strings);
    const ScratchString ownedKey = ScratchString::copyFrom(strings, key.ptr, key.len);
    const ScratchString ownedValidity = ScratchString::copyFrom(strings, validityName.ptr, validityName.len);
    key = StrSpan(ownedKey); validityName = StrSpan(ownedValidity);
    StrSpan scopes[ExecutionParameters::MAX_SCOPE_DEPTH];
    const int32_t count = gl::ScopeSyntax::copyStrictAncestorSpans(mb, validityName, scopes, strings);
    for (int32_t i = 0; i < count; ++i) {
        DirtyState dirty = DirtyState::Clean;
        PagedVector<int64_t> matches(&arena, &dirty);
        canonicalTemplates.matchingCanonicalTemplates(mb, mb.overallHashMemory.rejectedMapIntegration,
            mb.lbMemory.canonicalUses.integrationParkMatches, key, mb.nameMap.lookup(scopes[i]),
            mb.nameMap.lookup(validityName), matches, arena);
        for (int32_t k = 0; k < matches.size(); ++k) {
            const auto records = snapshotRejectedIntegrationRun(mb.overallHashMemory.rejectedMapIntegration, matches[k], arena);
            for (int32_t r = 0; r < records.count; ++r)
                queueParkedExistence(mb, key, records.views[r], scopes[i], validityName, true);
        }
    }

}

/// @brief Wake the parked integration sources a newly filed double-marker
///        key admits (D-492).
///
/// @details
/// The double-marker twin of the revival after a new
/// `admissionMapIntegration` key. Candidates are the
/// `rejectedMapIntegration` parks with the key's predicate whose scope is
/// comparable to @p validityName. A park at that scope or below is revisited
/// at its own scope; a park at a strict ancestor is revisited at the key's
/// scope, which releases it into the key's scope
/// (`releaseAncestorRejectedIntegration2`). A candidate is revisited only when
/// the route-5 probe (`AdmissionMemory::isAdmittedIntegrationDouble`) admits
/// its marked text at the revisit scope. Revisits run in decoded
/// (template, scope) order, each distinct pair once; the park texts are copied
/// before any probe, since the class-aware match mints templates.
///
/// @param mb           Owning block (single-threaded seam).
/// @param keyTemplate  The new key's template (`u_` form with `marker1` /
///                     `marker2`); only its predicate is read.
/// @param validityName The key's scope.
/// @return (void)
/// @invariant Writes no admission or park record itself; every wake goes
///            through `revisitRejectedIntegration2`.
/// @see AdmissionMemory::updateAdmissionMapIntegrationDouble,
///      revisitRejectedIntegration2.
void RejectedSourceReplay::revisitRejectedIntegrationDouble(Memory& mb, StrSpan keyTemplate, StrSpan validityName) {
    const TypedColdBlobMap<int64_t, RejectedMapIntegrationValue>& parks = mb.overallHashMemory.rejectedMapIntegration;
    if (parks.empty()) return;
    const NameId keyScope = mb.nameMap.lookup(validityName);
    assert(keyScope != 0 && "revisitRejectedIntegrationDouble: the key's scope is registered");
    int32_t predicate = 0;
    while (predicate < keyTemplate.len && keyTemplate.ptr[predicate] != '[') ++predicate;
    assert(predicate < keyTemplate.len && "revisitRejectedIntegrationDouble: a key without arguments");
    const unsigned genSlot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
    const unsigned stringSlot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
    ScratchArena& arena = genScratchArenas().forSlot(genSlot);
    ScratchArena& strings = scratchArenas().forSlot(stringSlot);
    ScratchScope frame(arena), stringFrame(strings);
    DirtyState dirty = DirtyState::Clean;
    // One wake: a park template and the scope its revisit runs at.
    struct Wake { NameId text; NameId scope; };
    PagedVector<Wake> wakes(&arena, &dirty);
    ColdHashSet<PodKeyStore<int64_t>> pairs(&arena, &dirty);
    for (int32_t id = 1; id <= parks.count(); ++id) {
        const StatementKey park = Codec<StatementKey>::decode(parks.keyAt(id));
        NameId revisitScope = 0;
        if (park.validity == keyScope || mb.nameMap.isStrictAncestor(keyScope, park.validity))
            revisitScope = park.validity;
        else if (mb.nameMap.isStrictAncestor(park.validity, keyScope))
            revisitScope = keyScope;
        else
            continue;
        const StrSpan text = mb.templateInterner.decodeView(park.orig);
        if (text.len <= predicate || text.ptr[predicate] != '['
            || std::memcmp(text.ptr, keyTemplate.ptr, static_cast<std::size_t>(predicate)) != 0)
            continue;
        const int32_t before = pairs.count();
        pairs.mint(packInt32Pair(park.orig, revisitScope));
        if (pairs.count() != before) wakes.push_back(Wake{ park.orig, revisitScope });
    }
    const auto less = [&](int32_t a, int32_t b) {
        const int c = compareSpans(mb.templateInterner.decodeView(wakes[a].text),
            mb.templateInterner.decodeView(wakes[b].text));
        return c != 0 ? c < 0 : compareSpans(mb.nameMap.decodeView(wakes[a].scope),
            mb.nameMap.decodeView(wakes[b].scope)) < 0;
    };
    ChunkSortedOrdinals<decltype(less)> order(arena, wakes.size(), 0, less);
    for (int32_t i = 0; i < wakes.size(); ++i) {
        ScratchScope wakeFrame(strings);
        const Wake wake = wakes[order.next()];
        const StrSpan textView = mb.templateInterner.decodeView(wake.text);
        const ScratchString text = ScratchString::copyFrom(strings, textView.ptr, textView.len);
        const StrSpan scopeView = mb.nameMap.decodeView(wake.scope);
        const ScratchString scope = ScratchString::copyFrom(strings, scopeView.ptr, scopeView.len);
        if (!admissionMemory.isAdmittedIntegrationDouble(mb, StrSpan(text), StrSpan(scope))) continue;
        revisitRejectedIntegration2(StrSpan(text), mb, StrSpan(scope));
    }
}

/// @brief Wake the integration parks in the OR cases a scope that gained a
///        route-3 demand key serves (D-572).
///
/// @details
/// A route-3 demand is visible at its scope and at the OR cases it serves
/// (`AdmissionMemory::ancestorDemandOwner`), so a source parked in such a case
/// before the demand arrived can now be admitted there. Candidates are the
/// `rejectedMapIntegration` parks at a scope whose serving owner is
/// @p validityName and whose template has the key's predicate; each distinct
/// (template, park scope) pair whose route-3 probe at the park scope
/// (`AdmissionMemory::isAdmittedIntegration`, no commit) succeeds is revisited
/// at the park scope through `revisitRejectedIntegration2`, in decoded
/// (template, scope) order. The key's own scope and its ancestors are the
/// existing `revisitRejectedIntegration2` call's at the filing site. Park texts
/// and scopes are copied before any probe, since the class-aware match mints
/// templates.
///
/// @param mb           Owning block (single-threaded seam).
/// @param bareKey      The new key's bare marker template (no `u_` prefixes);
///                     only its predicate is read.
/// @param validityName The key's scope, registered.
/// @return (void)
/// @invariant Writes no admission or park record itself; every wake goes
///            through `revisitRejectedIntegration2`.
/// @see IntegrationProcessor::prepareIntegrationCore2,
///      EquivalenceAdmissionRewriter::applyEquivalenceClassToAdmissionMapIntegration
///      — the filing sites; revisitRejectedIntegrationDouble — the route-5 twin.
void RejectedSourceReplay::wakeDescendantIntegrationParks(Memory& mb, StrSpan bareKey, StrSpan validityName) {
    const TypedColdBlobMap<int64_t, RejectedMapIntegrationValue>& parks = mb.overallHashMemory.rejectedMapIntegration;
    if (parks.empty()) return;
    const NameId keyScope = mb.nameMap.lookup(validityName);
    assert(keyScope != 0 && "wakeDescendantIntegrationParks: the key's scope is registered");
    int32_t predicate = 0;
    while (predicate < bareKey.len && bareKey.ptr[predicate] != '[') ++predicate;
    assert(predicate < bareKey.len && "wakeDescendantIntegrationParks: a key without arguments");
    const unsigned genSlot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
    const unsigned stringSlot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
    ScratchArena& arena = genScratchArenas().forSlot(genSlot);
    ScratchArena& strings = scratchArenas().forSlot(stringSlot);
    ScratchScope frame(arena), stringFrame(strings);
    // The key's predicate with its bracket, owned: the probes below mint
    // templates.
    const ScratchString head = ScratchString::copyFrom(strings, bareKey.ptr, predicate + 1);
    DirtyState dirty = DirtyState::Clean;
    // One wake: a park template and the park scope its revisit runs at.
    struct Wake { NameId text; NameId scope; };
    PagedVector<Wake> wakes(&arena, &dirty);
    ColdHashSet<PodKeyStore<int64_t>> pairs(&arena, &dirty);
    for (int32_t id = 1; id <= parks.count(); ++id) {
        const StatementKey park = Codec<StatementKey>::decode(parks.keyAt(id));
        if (admissionMemory.ancestorDemandOwner(mb, park.validity) != keyScope) continue;
        const StrSpan text = mb.templateInterner.decodeView(park.orig);
        if (text.len <= predicate
            || std::memcmp(text.ptr, StrSpan(head).ptr, static_cast<std::size_t>(predicate + 1)) != 0)
            continue;
        const int32_t before = pairs.count();
        pairs.mint(packInt32Pair(park.orig, park.validity));
        if (pairs.count() != before) wakes.push_back(Wake{ park.orig, park.validity });
    }
    const auto less = [&](int32_t a, int32_t b) {
        const int c = compareSpans(mb.templateInterner.decodeView(wakes[a].text),
            mb.templateInterner.decodeView(wakes[b].text));
        return c != 0 ? c < 0 : compareSpans(mb.nameMap.decodeView(wakes[a].scope),
            mb.nameMap.decodeView(wakes[b].scope)) < 0;
    };
    ChunkSortedOrdinals<decltype(less)> order(arena, wakes.size(), 0, less);
    const StrSpan marker("marker", 6);
    // During a canonical maintenance pass the revisit only queues the pair;
    // the drain probes it against the settled demands.
    const bool deferring = mb.lbMemory.canonicalUses.deferWakes;
    for (int32_t i = 0; i < wakes.size(); ++i) {
        ScratchScope wakeFrame(strings);
        const Wake wake = wakes[order.next()];
        const StrSpan textView = mb.templateInterner.decodeView(wake.text);
        const ScratchString text = ScratchString::copyFrom(strings, textView.ptr, textView.len);
        const StrSpan scopeView = mb.nameMap.decodeView(wake.scope);
        const ScratchString scope = ScratchString::copyFrom(strings, scopeView.ptr, scopeView.len);
        if (!deferring
            && !admissionMemory.isAdmittedIntegration(mb, StrSpan(text), marker, StrSpan(text), StrSpan(scope), false))
            continue;
        revisitRejectedIntegration2(StrSpan(text), mb, StrSpan(scope));
    }
}

/// @brief Wake the parked integration sources a newly filed wildcard key
///        admits (D-586).
///
/// @details
/// The wildcard twin of the revival after a new `admissionMapIntegration` key.
/// A wildcard key spells the other open argument of its part `pi_`, so its
/// text names no park and the exact revival finds none. Candidates are the
/// `rejectedMapIntegration` parks with the key's predicate owned at the key's
/// scope or at a strict ancestor of it; each distinct park template whose
/// route-3 probe at the key's scope (`AdmissionMemory::isAdmittedIntegration`,
/// no commit, the wildcard variants included) succeeds is revisited at the
/// key's scope through `revisitRejectedIntegration2`, which also releases an
/// ancestor's park into the scope. During a canonical maintenance pass the
/// revisit only queues, as in `wakeDescendantIntegrationParks`. Revisits run in
/// decoded template order; the park texts are copied before any probe, since
/// the class-aware match mints templates. The parks in the OR cases the scope
/// serves remain `wakeDescendantIntegrationParks`'s.
///
/// @param mb           Owning block (single-threaded seam).
/// @param bareKey      The new key's bare template (no `u_` prefixes); only its
///                     predicate is read.
/// @param validityName The key's scope, registered.
/// @return (void)
/// @invariant Writes no admission or park record itself; every wake goes
///            through `revisitRejectedIntegration2`.
/// @see IntegrationProcessor::prepareIntegrationCore2 — the filing site;
///      revisitRejectedIntegrationDouble — the route-5 twin.
void RejectedSourceReplay::wakeWildcardIntegrationParks(Memory& mb, StrSpan bareKey, StrSpan validityName) {
    const TypedColdBlobMap<int64_t, RejectedMapIntegrationValue>& parks = mb.overallHashMemory.rejectedMapIntegration;
    if (parks.empty()) return;
    const NameId keyScope = mb.nameMap.lookup(validityName);
    assert(keyScope != 0 && "wakeWildcardIntegrationParks: the key's scope is registered");
    int32_t predicate = 0;
    while (predicate < bareKey.len && bareKey.ptr[predicate] != '[') ++predicate;
    assert(predicate < bareKey.len && "wakeWildcardIntegrationParks: a key without arguments");
    const unsigned genSlot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
    const unsigned stringSlot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
    ScratchArena& arena = genScratchArenas().forSlot(genSlot);
    ScratchArena& strings = scratchArenas().forSlot(stringSlot);
    ScratchScope frame(arena), stringFrame(strings);
    // The key's predicate with its bracket and the key's scope, owned: the
    // probes below mint templates.
    const ScratchString head = ScratchString::copyFrom(strings, bareKey.ptr, predicate + 1);
    const ScratchString scope = ScratchString::copyFrom(strings, validityName.ptr, validityName.len);
    DirtyState dirty = DirtyState::Clean;
    PagedVector<NameId> texts(&arena, &dirty);
    ColdHashSet<PodKeyStore<int64_t>> seen(&arena, &dirty);
    for (int32_t id = 1; id <= parks.count(); ++id) {
        const StatementKey park = Codec<StatementKey>::decode(parks.keyAt(id));
        if (park.validity != keyScope && !mb.nameMap.isStrictAncestor(park.validity, keyScope)) continue;
        const StrSpan text = mb.templateInterner.decodeView(park.orig);
        if (text.len <= predicate
            || std::memcmp(text.ptr, StrSpan(head).ptr, static_cast<std::size_t>(predicate + 1)) != 0)
            continue;
        const int32_t before = seen.count();
        seen.mint(static_cast<int64_t>(park.orig));
        if (seen.count() != before) texts.push_back(park.orig);
    }
    const auto less = [&](int32_t a, int32_t b) {
        return compareSpans(mb.templateInterner.decodeView(texts[a]),
            mb.templateInterner.decodeView(texts[b])) < 0;
    };
    ChunkSortedOrdinals<decltype(less)> order(arena, texts.size(), 0, less);
    const StrSpan marker("marker", 6);
    const bool deferring = mb.lbMemory.canonicalUses.deferWakes;
    for (int32_t i = 0; i < texts.size(); ++i) {
        ScratchScope wakeFrame(strings);
        const StrSpan textView = mb.templateInterner.decodeView(texts[order.next()]);
        const ScratchString text = ScratchString::copyFrom(strings, textView.ptr, textView.len);
        if (!deferring
            && !admissionMemory.isAdmittedIntegration(mb, StrSpan(text), marker, StrSpan(text), StrSpan(scope), false))
            continue;
        revisitRejectedIntegration2(StrSpan(text), mb, StrSpan(scope));
    }
}
}
