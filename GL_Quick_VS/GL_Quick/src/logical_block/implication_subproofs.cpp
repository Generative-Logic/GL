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
#include "implication_subproofs.hpp"
namespace gl {
/// @brief Bind inputs for implication subproofs.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param canonicalTemplates Shared canonical matcher used by parked-subproof maintenance.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param definitions Read-only compiled implication templates and their formal inputs.
/// @param integrationBuilder Borrowed IntegrationInstructionBuilder interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param equivalenceQueries Borrowed EquivalenceQueries interface or state; must outlive this component.
/// @param statementMemory Borrowed StatementMemory interface or state; must outlive this component.
/// @param rejectedSourceReplay Borrowed RejectedSourceReplay interface (the replay of a promoted trial's source); must outlive this component.
/// @return None; construction binds the supplied references.
ImplicationSubproofs::ImplicationSubproofs(CanonicalTemplateMatcher& canonicalTemplates, Canonicalizer& canonicalizer, const CompiledDefinitionRegistry& definitions, IntegrationInstructionBuilder& integrationBuilder, const ProverParameters& parameters, EquivalenceQueries& equivalenceQueries, StatementMemory& statementMemory, RejectedSourceReplay& rejectedSourceReplay) : canonicalTemplates(canonicalTemplates), canonicalizer(canonicalizer), definitions(definitions), integrationBuilder(integrationBuilder), parameters(parameters), equivalenceQueries(equivalenceQueries), statementMemory(statementMemory), rejectedSourceReplay(rejectedSourceReplay) {}

/// @brief Test whether distinct implication inputs coincide without an equality premise.
///
/// @details The compiled signature defines the input positions independently of
/// their actual names. Only a direct positive equality premise between the same
/// two formal inputs permits their identification, in either orientation. The
/// conclusion, nested compounds and bound premise variables grant no permission.
/// Complete equality classes at the supplied scope include ancestors but never
/// sibling or descendant facts; weak representatives do not affect membership.
///
/// @param signature Clean numbered implication instance whose inputs are fixed.
/// @param scope Registered scope at which to compare the actual input objects.
/// @param body Resident logic block supplying the complete equality view.
/// @return True exactly when an unlicensed pair of distinct input positions is equal.
/// @invariant This search predicate derives no statement and reads no proof history.
/// @see retireEqualInputImplicationSubproofs; openImplicationSubproof.
bool ImplicationSubproofs::hasForbiddenEqualInputs(StrSpan signature, NameId scope, Memory& body) {
    assert(scope != 0 && startsWithSpan(signature, "(implication", 12));
    StrSpan actual[ExecutionParameters::MAX_ARITY];
    const int32_t count = getArgsSpans(signature, actual, ExecutionParameters::MAX_ARITY);
    if (count < 2) return false;

    canonicalizer.initializeCanonicalScope(body, scope);
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
    ScratchArena& arena = genScratchArenas().forSlot(slot);
    ScratchScope frame(arena);
    NameId names[ExecutionParameters::MAX_ARITY]{};
    int32_t classes[ExecutionParameters::MAX_ARITY]{};
    for (int32_t i = 0; i < count; ++i) names[i] = body.nameMap.lookup(actual[i]);
    const int32_t bucket = body.canonicalEquivalenceClassesMap.lookup(scope);
    if (bucket != 0) {
        const int32_t classCount = body.canonicalEquivalenceClassesMap.runLen(bucket);
        for (int32_t c = 0; c < classCount; ++c) {
            ScratchScope classFrame(arena);
            int32_t length = 0;
            const char* bytes = body.canonicalEquivalenceClassesMap.peekRecordBytes(bucket, c, length, arena);
            const EquivalenceClassView view{bytes, length};
            for (int32_t i = 0; i < count; ++i)
                if (names[i] != 0 && classHasMember(view, names[i])) classes[i] = c + 1;
        }
    }

    const LogicalEntity* definition = nullptr;
    StrSpan formal[ExecutionParameters::MAX_ARITY];
    for (int32_t a = 0; a < count; ++a) {
        for (int32_t b = a + 1; b < count; ++b) {
            if (!equalSpans(actual[a], actual[b])
                && (classes[a] == 0 || classes[a] != classes[b])) continue;
            if (definition == nullptr) {
                definition = definitions.compiledEntity(extractExpressionSpan(signature));
                assert(definition != nullptr && definition->category == "implication");
                assert(definition->elements.size() >= 2);
                const int32_t formalCount = getArgsSpans(StrSpan(definition->signature), formal,
                    ExecutionParameters::MAX_ARITY);
                assert(formalCount == count && definition->arity == count);
            }
            if (equalSpans(formal[a], formal[b])) continue;
            bool allowed = false;
            for (std::size_t p = 0; p + 1 < definition->elements.size() && !allowed; ++p) {
                const StrSpan premise(definition->elements[p]);
                if (!startsWithSpan(premise, "(=[", 3)) continue;
                StrSpan pair[2];
                const int32_t pairCount = getArgsSpans(premise, pair, 2);
                assert(pairCount == 2);
                allowed = (equalSpans(pair[0], formal[a]) && equalSpans(pair[1], formal[b]))
                    || (equalSpans(pair[0], formal[b]) && equalSpans(pair[1], formal[a]));
            }
            if (!allowed) return true;
        }
    }
    return false;
}

/// @brief Test whether an implication instance names a built-on-listed object.
///
/// @details An argument names a built-on-listed object (I-293) when it is in
/// `Memory::builtOnListedWitnesses` and every member of its complete equality
/// class at the scope is in the registry too. A class member outside the
/// registry names the same object without the listed construction, so the
/// verdict does not depend on which member the class-canonical choice
/// (`compareWitnessNames`, I-88) puts into the instance (I-217). An argument
/// in no class stands for itself. Complete equality classes at the supplied
/// scope include ancestors but never sibling or descendant facts.
///
/// @param signature Clean numbered implication instance whose inputs are fixed.
/// @param scope Registered scope at which to compare the actual input objects.
/// @param body Resident logic block supplying the registry and the complete equality view.
/// @return True exactly when some argument's whole class is built-on-listed.
/// @invariant This search predicate derives no statement and reads no proof history.
/// @see openImplicationSubproof; hasForbiddenEqualInputs.
bool ImplicationSubproofs::hasBuiltOnListedArgument(StrSpan signature, NameId scope, Memory& body) {
    assert(scope != 0 && "hasBuiltOnListedArgument: expected a registered scope");
    StrSpan actual[ExecutionParameters::MAX_ARITY];
    const int32_t count = getArgsSpans(signature, actual, ExecutionParameters::MAX_ARITY);
    bool anyListed = false;
    for (int32_t i = 0; i < count && !anyListed; ++i)
        anyListed = body.builtOnListedWitnesses.lookup(actual[i]) != 0;
    if (!anyListed) return false;

    canonicalizer.initializeCanonicalScope(body, scope);
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
    ScratchArena& arena = genScratchArenas().forSlot(slot);
    ScratchScope frame(arena);
    const int32_t bucket = body.canonicalEquivalenceClassesMap.lookup(scope);
    for (int32_t i = 0; i < count; ++i) {
        if (body.builtOnListedWitnesses.lookup(actual[i]) == 0) continue;
        // Every class member is interned at class commit (I-88), so a listed
        // name the NameMap never interned is in no class and stands for itself.
        const NameId name = body.nameMap.lookup(actual[i]);
        bool unlistedMember = false;
        if (name != 0 && bucket != 0) {
            const int32_t classCount = body.canonicalEquivalenceClassesMap.runLen(bucket);
            for (int32_t c = 0; c < classCount && !unlistedMember; ++c) {
                ScratchScope classFrame(arena);
                int32_t length = 0;
                const char* bytes = body.canonicalEquivalenceClassesMap.peekRecordBytes(bucket, c, length, arena);
                const EquivalenceClassView view{bytes, length};
                if (!classHasMember(view, name)) continue;
                for (int32_t m = 0; m < view.memberCount() && !unlistedMember; ++m)
                    unlistedMember = body.builtOnListedWitnesses.lookup(
                        body.nameMap.decodeView(view.memberId(m))) == 0;
            }
        }
        if (!unlistedMember) return true;
    }
    return false;
}

/// @brief Retire implication subproofs whose distinct inputs become equal without permission.
///
/// @details Only induction-step blocks (recursionAuxyImplId != 0) are checked;
/// ordinary, contradiction, typing and induction-zero blocks are unchanged.
/// Both bare and goal-qualified numbered implication subproofs are selected
/// through their prepared scope identities. The input predicate reads
/// each subproof's own complete equality view. A hit queues that scope and its
/// descendants for the ordinary deferred wipe; history and the prepared key
/// survive, so the same candidate cannot reopen. This abandons search and never
/// records a contradiction, a proved implication or a parent goal closure.
///
/// @param body Resident logic block whose open implication subproofs are checked.
/// @return Nothing; offending scope roots are added to pendingWipeScopes.
/// @invariant Closed scopes and descendants of an already queued root are skipped.
/// @see hasForbiddenEqualInputs; retireSettledImplicationSubproofs.
void ImplicationSubproofs::retireEqualInputImplicationSubproofs(Memory& body) {
    if (body.recursionAuxyImplId == 0) return;
    const int32_t count = body.integrationPrepared.count();
    if (count == 0) return;
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
    ScratchArena& arena = scratchArenas().forSlot(slot);
    ScratchScope frame(arena);
    for (int32_t i = 1; i <= count; ++i) {
        ScratchScope candidateFrame(arena);
        StrSpan payloadView, parentView;
        decodeTemplateKeyView(body.integrationPrepared.decode(i), body.templateInterner, body.nameMap,
            payloadView, parentView);
        const StrSpan bare = gl::ScopeSyntax::stripSubproofPrefixView(payloadView);
        if (!startsWithSpan(bare, "(implication", 12)) continue;
        const ScratchString signature = ScratchString::copyFrom(arena, bare.ptr, bare.len);
        const int32_t length = parentView.len + static_cast<int32_t>(NameMap::BOUNDARY_LEN) + payloadView.len;
        char* scopeText = arena.allocBytes(length);
        std::memcpy(scopeText, parentView.ptr, parentView.len);
        std::memcpy(scopeText + parentView.len, NameMap::BOUNDARY_STR, NameMap::BOUNDARY_LEN);
        std::memcpy(scopeText + parentView.len + NameMap::BOUNDARY_LEN, payloadView.ptr, payloadView.len);
        const NameId scope = body.nameMap.lookup(StrSpan(scopeText, length));
        if (scope == 0) continue;
        bool closed = false;
        for (NameId current = scope; current != 0 && !closed; current = body.nameMap.parentOf(current))
            closed = body.intValidityNamesToFilter.contains(current) || body.pendingWipeScopes.contains(current);
        if (closed) continue;
        if (hasForbiddenEqualInputs(StrSpan(signature), scope, body)) {
            RT_SCOPE_HERE("RETIRE_EQUAL_INPUT_SUBPROOF_HIT");
            body.pendingWipeScopes.mint(scope);
        }
    }
}

/// @brief Retire the open induction scopes whose admitted names all equal
///        names their root holds.
///
/// @details An induction scope carries no assumption (D-519): its admission's
/// facts hold at its root, and a hypothesis instance on names the root
/// already holds is the root's in-place advance. A scope whose admitted
/// names, the one or two names its scope name carries after the chain
/// (`ScopeSyntax::inductionScopeNames`), each share a class of the scope's
/// complete equality view with a root name adds nothing the root does not
/// hold, so it joins the wipe drain (D-545). A
/// root name is an ordinary name (a number, `rec`, a copy) or a witness of an
/// ancestor block (its level below the block's). A scope of a settled tree,
/// and a scope that is wiped or queued itself or through an ancestor, is no
/// candidate, and so is an in-place scope, a one-name scope named by a root
/// name (D-546): it carries the path its aliases
/// duplicate. Nothing keeps a later admission from opening the scope again.
/// Only a block whose hypothesis ate (`LbMemory::inductionEats` non-empty)
/// scans its NameMap; candidates come in NameMap id order. Heap-free (Rule
/// 28): the payload names and the strong chain are copied onto the string
/// tier, the class views ride the gen-scratch tier.
///
/// @param body Resident logic block whose induction scopes are swept.
/// @return Nothing; retired scopes are added to pendingWipeScopes.
/// @invariant Deterministic: NameMap id order, one pass.
/// @see ScopeSyntax::inductionScopeNames; retireEqualInputImplicationSubproofs;
///      Memory::settleInductionTrees.
void ImplicationSubproofs::retireAliasInductionScopes(Memory& body) {
    if (body.lbMemory.inductionEats.count() == 0) return;
    const unsigned gSlot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(gSlot);
    ScratchScope gFrame(gArena);
    const unsigned sSlot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
    ScratchArena& sArena = scratchArenas().forSlot(sSlot);
    ScratchScope sFrame(sArena);
    // The strong step's chains (sum and product,
    // D-566): a scope one of them opened carries
    // two names.
    StrSpan strongChains[2];
    int32_t strongChainN = 0;
    for (const NameId chainId : { body.inductionDoubleChainId, body.inductionProductChainId }) {
        if (chainId == 0) continue;
        const StrSpan view = body.nameMap.decodeView(chainId);
        strongChains[strongChainN++] = StrSpan(ScratchString::copyFrom(sArena, view.ptr, view.len));
    }
    const auto rootName = [&](StrSpan name) {
        const WitnessName witness = classifyWitnessName(name);
        return witness.kind == WitnessNameKind::Ordinary || witness.level < body.level;
    };
    const int32_t nameHighWater = body.nameMap.nameCount();
    for (NameId id = 1; id <= nameHighWater; ++id) {
        if (body.nameMap.parentOf(id) == 0 || !gl::ScopeSyntax::isInductionScope(body.nameMap, id)) continue;
        bool closed = false;
        for (NameId scope = id; scope != 0 && !closed; scope = body.nameMap.parentOf(scope))
            closed = body.intValidityNamesToFilter.contains(scope) || body.pendingWipeScopes.contains(scope);
        if (closed || body.inductionTreeSettled(id)) continue;
        ScratchScope candidateFrame(sArena);
        const StrSpan payloadView = body.nameMap.decodeSubView(body.nameMap.stackBack(id));
        const StrSpan payload(ScratchString::copyFrom(sArena, payloadView.ptr, payloadView.len));
        bool strongPayload = false;
        for (int32_t k = 0; k < strongChainN && !strongPayload; ++k)
            strongPayload = gl::ScopeSyntax::inductionPayloadNamesChain(payload, &strongChains[k], 1);
        const int32_t nameN = strongPayload ? 2 : 1;
        StrSpan names[2];
        gl::ScopeSyntax::inductionScopeNames(payload, nameN, names);
        // A one-name scope named by a root name is an in-place eat: it carries
        // the path its aliases duplicate and is never wiped
        // (D-546).
        if (nameN == 1 && rootName(names[0])) continue;
        bool alias = true;
        for (int32_t k = 0; k < nameN && alias; ++k) {
            if (rootName(names[k])) continue;
            alias = false;
            const NameId nameId = body.nameMap.lookup(names[k]);
            if (nameId == 0) break;
            canonicalizer.initializeCanonicalScope(body, id);
            const int32_t bucket = body.canonicalEquivalenceClassesMap.lookup(id);
            if (bucket == 0) break;
            const int32_t classCount = body.canonicalEquivalenceClassesMap.runLen(bucket);
            for (int32_t c = 0; c < classCount; ++c) {
                ScratchScope classFrame(gArena);
                int32_t length = 0;
                const char* bytes = body.canonicalEquivalenceClassesMap.peekRecordBytes(bucket, c, length, gArena);
                const EquivalenceClassView view{bytes, length};
                if (!classHasMember(view, nameId)) continue;
                for (int32_t m = 0; m < view.memberCount() && !alias; ++m) {
                    const NameId member = view.memberId(m);
                    alias = member != nameId && rootName(body.nameMap.decodeView(member));
                }
                break;
            }
        }
        if (alias) {
            RT_SCOPE_HERE("RETIRE_ALIAS_INDUCTION_SCOPE_HIT");
            body.pendingWipeScopes.mint(id);
        }
    }
}

/// @brief End-of-burst retirement of settled implication subproofs —
///        a subproof whose implication is already known or refuted at
///        its parent is wiped, and of two subproofs whose implications
///        became class-equal the one with more statements survives.
///
/// @details
/// Candidates are the OPEN implication subproofs with a bare payload:
/// `prepareIntegrationCore2` Case A scopes opened by an arriving
/// statement (admission and marker replays) or nested below another
/// scope. The prepared key at the parent names the payload; the
/// scope is `<parent>_boundary_<payload>`, found by a non-minting
/// lookup. A scope in `intValidityNamesToFilter` (already wiped) or
/// queued on `pendingWipeScopes`, itself or through an ancestor, is
/// not open. Goal-driven subproofs (`<goal>_subproof_` payloads)
/// follow their goal's lifecycle and are not candidates: disproof
/// cleanup relies on each goal owning its own copy.
///
/// Three retirements, all queued on `pendingWipeScopes` so this
/// burst's wipe drain removes them:
/// 1. Known part: the payload is a known statement at the parent
///    scope, exactly or through the parent's complete classes
///    (`compactKnownUpToEquality`) — the subproof can add nothing.
/// 2. Refuted: outside a primed contradiction block, the negated
///    payload is known at the parent scope or an ancestor, literally
///    or through the parent's complete classes
///    (`elementKnownAtOrAbove`) — written there by the subproof's
///    reductio (`prepareNegatedCompoundContradiction`, primed at
///    opening) or derived otherwise. The implication is false at the
///    parent, so the subproof can never close. A refutation at a scope
///    retires the implication's subproofs at that scope and below it,
///    never above it (D-532). A primed
///    block keeps its subproofs: its parent is meant to be
///    inconsistent.
/// 3. Collapse: open candidates under one parent whose payloads have
///    the same canonical form at the parent (`canonicalFormAtScope`)
///    are one case; the KEEPER is the one with the most statements in
///    its scope and nested scopes (`scopeSubtreeStatementCount`),
///    ties to the byte-lex smaller payload — deterministic. Every
///    other one is wiped.
/// The prepared key of a wiped subproof stays, so admission never
/// reopens it, and the wipe adds the scope to
/// `intValidityNamesToFilter`, so it is never queued again. Wipes
/// never touch proof history (I-44). The remaining work runs under
/// the survivor, whose result every equal compact matches through the
/// classes (I-254).
///
/// Runs in `standardProcessing` after the OR freeze and release
/// drains, immediately before the `pendingWipeScopes` drain, so no
/// other sweep sees a queued subproof. Heap-free (Rule 28): the
/// candidate runs ride the gen-scratch tier and the payload, scope
/// name and canonical texts ride the string tier, reclaimed at exit.
/// Mints nothing into the NameMap or the template interner.
///
/// @param body The LB whose open subproofs are swept (containers
///             resident — the drain runs inside the LB's own burst).
/// @return (void)
/// @invariant Candidates process in `integrationPrepared` insertion
///            order; the keeper of a collapse group is a pure function
///            of container content.
/// @invariant A scope is queued at most once over the LB's lifetime.
/// @see retireDuplicateOrCohorts — the OR-cohort twin;
///      compactKnownUpToEquality; elementKnownAtOrAbove;
///      scopeSubtreeStatementCount; retireSubproofReductios — the
///      reductio scopes of the retired subproofs.
void ImplicationSubproofs::retireSettledImplicationSubproofs(Memory& body) {
    const int32_t keyN = body.integrationPrepared.count();
    if (keyN == 0) return;

    const unsigned gSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(gSlot);
    const ArenaOffset gMark = gArena.cursor();
    const unsigned sSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : scratchArenas().slotCount() - 1;
    ScratchArena& sArena = scratchArenas().forSlot(sSlot);
    ScratchScope sweepStrings(sArena);

    // A scope is open unless it or an ancestor is wiped (the filter holds
    // every wiped scope) or already queued for this burst's wipe.
    const auto closedOrQueued = [&](NameId scope) {
        for (NameId s = scope; s != 0; s = body.nameMap.parentOf(s))
            if (body.intValidityNamesToFilter.contains(s)
                || body.pendingWipeScopes.contains(s)) return true;
        return false;
    };

    struct Candidate { NameId scope; NameId parent; StrSpan payload; StrSpan parentName; StrSpan canonical; };
    DirtyState dirty = DirtyState::Clean;
    PagedVector<Candidate> open(&gArena, &dirty);
    for (int32_t id = 1; id <= keyN; ++id) {
        const int64_t key = body.integrationPrepared.decode(id);
        StrSpan payloadView, parentView;
        decodeTemplateKeyView(key, body.templateInterner, body.nameMap, payloadView, parentView);
        // Bare implication payloads only: goal-driven payloads start with
        // their goal, and every other prepared template is no subproof.
        if (!startsWithSpan(payloadView, "(implication", 12)) continue;
        const int32_t boundaryLen = static_cast<int32_t>(NameMap::BOUNDARY_LEN);
        const int32_t nameLen = parentView.len + boundaryLen + payloadView.len;
        char* nameBuf = sArena.allocBytes(nameLen);
        std::memcpy(nameBuf, parentView.ptr, static_cast<std::size_t>(parentView.len));
        std::memcpy(nameBuf + parentView.len, NameMap::BOUNDARY_STR, NameMap::BOUNDARY_LEN);
        std::memcpy(nameBuf + parentView.len + boundaryLen, payloadView.ptr,
                    static_cast<std::size_t>(payloadView.len));
        // A prepared implication template whose scope was never pushed is an
        // integration expression, not a Case A subproof.
        const NameId scope = body.nameMap.lookup(StrSpan(nameBuf, nameLen));
        if (scope == 0 || closedOrQueued(scope)) continue;
        const StatementKey sk = Codec<StatementKey>::decode(key);
        const ScratchString payload = ScratchString::copyFrom(sArena, payloadView.ptr, payloadView.len);
        const ScratchString parentName = ScratchString::copyFrom(sArena, parentView.ptr, parentView.len);
        open.push_back(Candidate{ scope, sk.validity, StrSpan(payload), StrSpan(parentName), StrSpan() });
    }
    if (open.empty()) { gArena.popTo(gMark); return; }

    // 1. Known part: the implication is already known at the parent scope.
    // 2. Refuted: its negation is known at the parent or an ancestor.
    PagedVector<int32_t> alive(&gArena, &dirty);
    for (int32_t i = 0; i < open.size(); ++i) {
        const Candidate& c = open[i];
        if (equivalenceQueries.compactKnownUpToEquality(c.payload, c.parentName, body)) {
            body.pendingWipeScopes.mint(c.scope);
            continue;
        }
        if (!body.primedForContradiction) {
            ScratchScope negationScope(sArena);
            const StrSpan negation = gl::DefinitionExpansion::negateScratch(sArena, c.payload);
            if (equivalenceQueries.elementKnownAtOrAbove(negation, c.parent, body)) {
                RT_SCOPE_HERE("RETIRE_REFUTED_IMPLICATION_SUBPROOF_HIT");
                body.pendingWipeScopes.mint(c.scope);
                continue;
            }
        }
        alive.push_back(i);
    }
    if (alive.size() < 2) { gArena.popTo(gMark); return; }

    // 3. Collapse: equal canonical payloads under one parent form one group.
    for (int32_t a = 0; a < alive.size(); ++a) {
        Candidate c = open[alive[a]];
        int lvRun[256];
        int32_t lvN = 0;
        const CanonicalForm form = canonicalizer.canonicalFormAtScope(c.payload, c.parentName, body,
                                                        sArena, lvRun, lvN, 256);
        c.canonical = form.text;
        open.setAt(alive[a], c);
    }
    const int32_t aliveN = alive.size();
    int32_t* order = reinterpret_cast<int32_t*>(gArena.resolve(gArena.alloc(
        aliveN * static_cast<int32_t>(sizeof(int32_t)), static_cast<int32_t>(alignof(int32_t)))));
    for (int32_t a = 0; a < aliveN; ++a) order[a] = alive[a];
    std::sort(order, order + aliveN, [&](int32_t x, int32_t y) {
        const Candidate& l = open[x];
        const Candidate& r = open[y];
        if (l.parent != r.parent) return l.parent < r.parent;
        const int c = compareSpans(l.canonical, r.canonical);
        if (c != 0) return c < 0;
        return compareSpans(l.payload, r.payload) < 0;
    });
    for (int32_t start = 0; start < aliveN;) {
        int32_t end = start + 1;
        while (end < aliveN && open[order[end]].parent == open[order[start]].parent
               && equalSpans(open[order[end]].canonical, open[order[start]].canonical))
            ++end;
        if (end - start >= 2) {
            // Keeper: most statements in its subtree; ties to the byte-lex
            // smaller payload, which the sort already put first.
            int32_t keep = start;
            int32_t keepScore = scopeSubtreeStatementCount(body, open[order[start]].scope);
            for (int32_t k = start + 1; k < end; ++k) {
                const int32_t score = scopeSubtreeStatementCount(body, open[order[k]].scope);
                if (score > keepScore) { keep = k; keepScore = score; }
            }
            for (int32_t k = start; k < end; ++k)
                if (k != keep) body.pendingWipeScopes.mint(open[order[k]].scope);
        }
        start = end;
    }
    gArena.popTo(gMark);
}

/// @brief Queue the reductio scope of every closed or queued bare implication
///        subproof for the wipe drain.
///
/// @details
/// A bare numbered implication subproof opened outside a primed contradiction
/// block has its reductio beside it: `contradiction_!(<payload>)` pushed on
/// the same parent (`openImplicationSubproof`, step 5). The reductio serves
/// only the subproof's retirement, so it ends with the subproof
/// (D-532): once the subproof scope is closed
/// (in `intValidityNamesToFilter` — proved, retired or wiped with an
/// ancestor) or queued on `pendingWipeScopes` in this burst, its reductio
/// scope, when it exists and is still open, is queued too. A reductio that
/// discharged its refutation already queued itself, and one below a closed or
/// queued ancestor goes with that ancestor.
///
/// Candidates come from `integrationPrepared` in insertion order: bare
/// payloads starting `(implication`, whose subproof scope
/// `<parent>_boundary_<payload>` exists. Both scope names are found by
/// non-minting lookups. A primed contradiction block is skipped: its subproofs
/// get no reductio.
///
/// Runs in `standardProcessing` after every other retirement of the burst and
/// immediately before the `pendingWipeScopes` drain, so it sees every scope
/// queued in this burst. Heap-free (Rule 28): the scope names ride the string
/// tier, reclaimed per key. Mints nothing into the NameMap or the template
/// interner; reads no proof history.
///
/// @param body The LB whose reductio scopes are swept (containers resident —
///             the drain runs inside the LB's own burst).
/// @return (void)
/// @invariant A reductio scope is queued at most once: the wipe adds it to
///            `intValidityNamesToFilter`.
/// @see openImplicationSubproof; prepareNegatedCompoundContradiction;
///      retireSettledImplicationSubproofs;
///      GoalProcessor::dischargeContradictionScopes.
void ImplicationSubproofs::retireSubproofReductios(Memory& body) {
    if (body.primedForContradiction) return;
    const int32_t keyN = body.integrationPrepared.count();
    if (keyN == 0) return;

    const unsigned sSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : scratchArenas().slotCount() - 1;
    ScratchArena& sArena = scratchArenas().forSlot(sSlot);
    ScratchScope sweepStrings(sArena);

    // A scope is open unless it or an ancestor is wiped or already queued.
    const auto closedOrQueued = [&](NameId scope) {
        for (NameId s = scope; s != 0; s = body.nameMap.parentOf(s))
            if (body.intValidityNamesToFilter.contains(s)
                || body.pendingWipeScopes.contains(s)) return true;
        return false;
    };
    const StrSpan reductioPrefix("contradiction_!", 15);
    const int32_t boundaryLen = static_cast<int32_t>(NameMap::BOUNDARY_LEN);
    for (int32_t id = 1; id <= keyN; ++id) {
        ScratchScope keyStrings(sArena);
        StrSpan payloadView, parentView;
        decodeTemplateKeyView(body.integrationPrepared.decode(id), body.templateInterner,
            body.nameMap, payloadView, parentView);
        if (!startsWithSpan(payloadView, "(implication", 12)) continue;
        const int32_t subproofLen = parentView.len + boundaryLen + payloadView.len;
        char* subproofName = sArena.allocBytes(subproofLen);
        std::memcpy(subproofName, parentView.ptr, static_cast<std::size_t>(parentView.len));
        std::memcpy(subproofName + parentView.len, NameMap::BOUNDARY_STR, NameMap::BOUNDARY_LEN);
        std::memcpy(subproofName + parentView.len + boundaryLen, payloadView.ptr,
                    static_cast<std::size_t>(payloadView.len));
        const NameId subproof = body.nameMap.lookup(StrSpan(subproofName, subproofLen));
        if (subproof == 0 || !closedOrQueued(subproof)) continue;
        const int32_t reductioLen = subproofLen + reductioPrefix.len;
        char* reductioName = sArena.allocBytes(reductioLen);
        std::memcpy(reductioName, subproofName,
                    static_cast<std::size_t>(parentView.len + boundaryLen));
        std::memcpy(reductioName + parentView.len + boundaryLen, reductioPrefix.ptr,
                    static_cast<std::size_t>(reductioPrefix.len));
        std::memcpy(reductioName + parentView.len + boundaryLen + reductioPrefix.len,
                    payloadView.ptr, static_cast<std::size_t>(payloadView.len));
        const NameId reductio = body.nameMap.lookup(StrSpan(reductioName, reductioLen));
        if (reductio == 0 || closedOrQueued(reductio)) continue;
        RT_SCOPE_HERE("RETIRE_SUBPROOF_REDUCTIO_HIT");
        body.pendingWipeScopes.mint(reductio);
    }
}

/// @brief Count runtime statements in one scope and every scope nested below
///        it.
///
/// @details The collapse census of `retireSettledImplicationSubproofs`: one
/// pass over `intEncodedStatements`, counting each row whose validity is the
/// scope itself or a strict descendant in the validity forest. Reads no proof
/// history and mints nothing.
///
/// @param body  Owning resident block.
/// @param scope Minted validity id of the subtree root.
/// @return Number of runtime statements in the subtree.
/// @see retireSettledImplicationSubproofs; orCohortStatementCount.
int32_t ImplicationSubproofs::scopeSubtreeStatementCount(const Memory& body,
    NameId scope) const {
    assert(scope != 0 && "scopeSubtreeStatementCount: expected a minted scope");
    int32_t count = 0;
    for (int32_t i = 0; i < body.intEncodedStatements.size(); ++i) {
        const NameId validity = body.intEncodedStatements[i].validityId;
        if (validity == scope || body.nameMap.isStrictAncestor(scope, validity)) ++count;
    }
    return count;
}
}
