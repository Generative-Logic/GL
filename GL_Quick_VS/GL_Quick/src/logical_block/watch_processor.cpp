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
#include "../logical_block/statement_arrivals.hpp"
#include "watch_processor.hpp"
namespace gl {
/// @brief Bind inputs for watch processor.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param canonicalTemplates Borrowed CanonicalTemplateMatcher interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param ceFilteringActive Borrowed bool interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param caseLedger Borrowed CaseLedger interface or state; must outlive this component.
/// @param caseProcessor Borrowed CaseProcessor interface or state; must outlive this component.
/// @param proofHistory Borrowed ProofHistory interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
WatchProcessor::WatchProcessor(CanonicalTemplateMatcher& canonicalTemplates, Canonicalizer& canonicalizer, const bool& ceFilteringActive, const ProverParameters& parameters, CaseLedger& caseLedger, CaseProcessor& caseProcessor, ProofHistory& proofHistory) : canonicalTemplates(canonicalTemplates), canonicalizer(canonicalizer), ceFilteringActive(ceFilteringActive), parameters(parameters), caseLedger(caseLedger), caseProcessor(caseProcessor), proofHistory(proofHistory) {}

/// @brief Compose a canonical form and its proof bridge at a visible scope.
///
/// @details
/// One `canonicalFormAtScope` call reads the maintained complete target-scope
/// bucket and writes the rewritten text on caller-owned string scratch.
/// Applied equalities contribute levels and justifiers at their original
/// citation scopes; the read does not build or merge class state.
/// OR matching opts into equality views; ordinary watch behavior is unchanged.
///
/// @param mb      Owning LB.
/// @param origId  The statement's original id.
/// @param scopeId The scope whose visible classes apply.
/// @param outArena String scratch owning the returned spans.
/// @param out Canonical text and the applied equality bridge.
/// @param lvRun Ascending level buffer, extended by applied equalities.
/// @param lvN Current level count, updated in place.
/// @param lvCap Level-buffer capacity.
/// @param rewriteEqualities Include standalone equality and inequality views.
/// @param collectProofEvidence Include supporting levels and citation scopes.
///        False preserves equality dependency text and locality for lookup only;
///        lvRun must be null, lvN and lvCap zero, and eqScope entries are empty.
/// @return Void.
void WatchProcessor::canonicalWatchFormAt(Memory& mb, NameId origId,
    NameId scopeId, ScratchArena& outArena, WatchBridge& out, int* lvRun,
    int32_t& lvN, int32_t lvCap, bool rewriteEqualities,
    bool collectProofEvidence) {
    assert(origId != 0 && scopeId != 0
        && "canonicalWatchFormAt: both ids must be minted");
    out.changed = false;
    out.localEquality = false;
    out.eqN = 0;
    const StrSpan origView = mb.nameMap.decodeView(origId);
    const ScratchString original = ScratchString::copyFrom(outArena, origView.ptr, origView.len);
    const StrSpan scopeView = mb.nameMap.decodeView(scopeId);
    const ScratchString validity = ScratchString::copyFrom(outArena, scopeView.ptr, scopeView.len);
    const CanonicalForm canonical = canonicalizer.canonicalFormAtScope(StrSpan(original), StrSpan(validity),
        mb, outArena, lvRun, lvN, lvCap, rewriteEqualities,
        true, collectProofEvidence);
    out.changed = canonical.changed;
    const bool cite = parameters.trackHistory && !ceFilteringActive;
    for (int32_t i = 0; i < canonical.eqN; ++i) {
        assert(out.eqN < WatchBridge::CAP);
        out.eqText[out.eqN] = canonical.eqJust[i];
        out.localEquality = out.localEquality || canonical.eqLocal[i];
        out.eqScope[out.eqN] = StrSpan{};
        if (collectProofEvidence) {
            StrSpan scope = StrSpan(validity);
            if (cite) scope = proofHistory.findEqualityCiteScope(mb, canonical.eqJust[i], StrSpan(validity));
            out.eqScope[out.eqN] = StrSpan(ScratchString::copyFrom(outArena, scope.ptr, scope.len));
        }
        ++out.eqN;
    }
    const StrSpan cur = canonical.text;
    out.text = cur;
}

/// @brief Return the canonical identifier without constructing proof metadata.
///
/// @details Uses the same scope-visible chooser and substitution walk as proof
/// canonicalization. Only the resulting expression can mint a NameMap entry;
/// supporting levels, equality justifiers and citation scopes are not requested.
/// @param mb Owning block with initialized visible canonical classes.
/// @param origId Minted original expression identifier.
/// @param scopeId Minted effective scope identifier.
/// @return Original identifier or the identifier of its canonical text.
NameId WatchProcessor::canonicalWatchIdAt(Memory& mb, NameId origId,
    NameId scopeId) {
    assert(origId != 0 && scopeId != 0);
    const unsigned slot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : scratchArenas().slotCount() - 1;
    ScratchArena& strArena = scratchArenas().forSlot(slot);
    ScratchScope canonScope(strArena);
    int32_t levelCount = 0;
    const CanonicalForm canonical = canonicalizer.canonicalFormAtScope(mb.nameMap.decodeView(origId),
        mb.nameMap.decodeView(scopeId), mb, strArena, nullptr, levelCount, 0, false, false);
    if (!canonical.changed) return origId;
    return mb.nameMap.encode(canonical.text);
}

/// @brief The token a class member is rewritten to by canonical forms at a
///        scope.
/// @details Reads `canonicalMemberRepresentatives`, which the complete-view
/// rebuild writes for every member of every class of the scope's view. No
/// entry (the token is in no class there) and a zero normal representative
/// (every member weak there) both leave the token unchanged, as
/// `canonicalFormAtScope` does.
/// @param mb Owning LB.
/// @param member Minted token id.
/// @param scopeId Minted scope id.
/// @return The replacing token, or @p member when nothing replaces it.
NameId WatchProcessor::canonicalRepresentativeAt(const Memory& mb,
    NameId member, NameId scopeId) {
    assert(member != 0 && scopeId != 0);
    const int64_t* selected = mb.lbMemory.canonicalMemberRepresentatives.find(
        packStatementKey(member, scopeId));
    if (selected == nullptr) return member;
    const NameId normal = canonicalRecordNormal(*selected);
    return normal == 0 ? member : normal;
}

/// @brief Whether an expression has one canonical spelling at two scopes.
/// @details Every bracket token (the tokens `canonicalFormAtScope` may
/// replace) must have the same representative at both scopes; a token never
/// interned is in no class. Reads only.
/// @param mb Owning LB.
/// @param origId Minted expression id.
/// @param fromScope First scope.
/// @param toScope Second scope.
/// @return True when the representatives agree for every bracket token.
bool WatchProcessor::sameCanonicalSpelling(const Memory& mb, NameId origId,
    NameId fromScope, NameId toScope) {
    assert(origId != 0 && fromScope != 0 && toScope != 0);
    if (fromScope == toScope) return true;
    bool same = true;
    collectExprTokens(mb.nameMap.decodeView(origId), [&](StrSpan token) {
        if (!same) return;
        const NameId member = mb.nameMap.lookup(token);
        if (member == 0) return;
        same = canonicalRepresentativeAt(mb, member, fromScope)
            == canonicalRepresentativeAt(mb, member, toScope);
    });
    return same;
}

/// @brief Canonical form of an expression at a scope through a pass cache.
/// @details A new `canonicalUses.version` empties the cache. A miss whose
/// representatives agree with @p anchorScope takes the anchor's form;
/// otherwise one text-only `canonicalFormAtScope` call computes it and a
/// changed text is interned in the cache's own table (never the NameMap).
/// @param mb Owning LB.
/// @param cache The pass's cache.
/// @param origId Minted expression id.
/// @param anchorScope The scope whose form is likely shared.
/// @param scopeId The scope whose classes apply.
/// @return 0 when already canonical, else the text id in `cache.texts`.
int32_t WatchProcessor::canonicalFormCached(Memory& mb,
    CanonicalFormCache& cache, NameId origId, NameId anchorScope, NameId scopeId) {
    assert(origId != 0 && anchorScope != 0 && scopeId != 0);
    const int32_t version = mb.lbMemory.canonicalUses.version;
    if (cache.version != version) {
        cache.forms.resetToFresh();
        cache.texts.resetToFresh();
        cache.version = version;
    }
    const int64_t key = packInt32Pair(origId, scopeId);
    if (const int32_t* stored = cache.forms.find(key)) return *stored;
    int32_t form = 0;
    if (anchorScope != scopeId && sameCanonicalSpelling(mb, origId, anchorScope, scopeId)) {
        form = canonicalFormCached(mb, cache, origId, anchorScope, anchorScope);
    } else {
        const unsigned slot = (gl::WorkerContext::coreId() >= 0)
            ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
        ScratchArena& strings = scratchArenas().forSlot(slot);
        ScratchScope frame(strings);
        int32_t levelCount = 0;
        const CanonicalForm canonical = canonicalizer.canonicalFormAtScope(mb.nameMap.decodeView(origId),
            mb.nameMap.decodeView(scopeId), mb, strings, nullptr, levelCount, 0, false, false);
        if (canonical.changed) form = cache.texts.mint(canonical.text);
    }
    cache.forms.insert(key, form);
    return form;
}

/// @brief Canonical text of an expression at a scope through a pass cache.
/// @details The text (the expression itself when already canonical) is
/// copied onto @p outArena; the span aliases neither the NameMap nor the cache.
/// @param mb Owning LB.
/// @param cache The pass's cache.
/// @param origId Minted expression id.
/// @param anchorScope The scope whose form is likely shared.
/// @param scopeId The scope whose classes apply.
/// @param outArena String-tier arena receiving the copy.
/// @return The canonical text on @p outArena.
StrSpan WatchProcessor::canonicalTextCached(Memory& mb, CanonicalFormCache& cache,
    NameId origId, NameId anchorScope, NameId scopeId, ScratchArena& outArena) {
    const int32_t form = canonicalFormCached(mb, cache, origId, anchorScope, scopeId);
    const StrSpan text = form == 0 ? mb.nameMap.decodeView(origId) : cache.texts.view(form);
    return StrSpan(ScratchString::copyFrom(outArena, text.ptr, text.len));
}

/// @brief Canonical id of an expression at a scope through a pass cache.
/// @details A changed form is encoded in the NameMap from the cache's own
/// table, so the name and the moment of its first mint match
/// `canonicalWatchIdAt` for the same call sequence.
/// @param mb Owning LB.
/// @param cache The pass's cache.
/// @param origId Minted expression id.
/// @param anchorScope The scope whose form is likely shared.
/// @param scopeId The scope whose classes apply.
/// @return @p origId when nothing applies, else the canonical id.
NameId WatchProcessor::canonicalIdCached(Memory& mb, CanonicalFormCache& cache,
    NameId origId, NameId anchorScope, NameId scopeId) {
    const int32_t form = canonicalFormCached(mb, cache, origId, anchorScope, scopeId);
    if (form == 0) return origId;
    return mb.nameMap.encode(cache.texts.view(form));
}

/// @brief Registration-door hook: stage a freshly admitted row for
        ///        the next watch drain.
        ///
        /// @details
        /// Called next to `recordNegationWatchFor` at the three registration
        /// seams; both polarities are staged (a positive arrival probes the
        /// negation ledger, a negated one probes the positive rows). Rows
        /// at phase-3-ignored scopes and non-watch shapes are defined
        /// no-ops. Nothing is canonicalized here — the drain does that once,
        /// single-threaded.
        ///
        /// @param mb The owning LB.
        /// @param ie The row just registered (ids minted).
        /// @return (void)
        /// @see materializeWatchHits.
void WatchProcessor::stageWatchArrival(Memory& mb, const IntEncodedExpr& ie) {
    gl::StatementArrivals::stageOrFactChange(mb, packStatementKey(ie.originalId, ie.validityId));
    gl::StatementArrivals::stageSubproofArrival(mb, packStatementKey(ie.originalId, ie.validityId));
    gl::StatementArrivals::stageContradictionArrival(mb, packStatementKey(ie.originalId, ie.validityId));
    if (!watchShaped(mb, ie)) return;
    if (mb.phase3IgnoresScope(ie.validityId)) return;
    mb.pendingWatchArrivals.push_back(
        packStatementKey(ie.originalId, ie.validityId));
}

/// @brief Class-merge hook: import the ancestor-scope negated rows a
        ///        new class at @p scopeId can rewrite into @p scopeId's
        ///        negation run.
        ///
        /// @details
        /// A class formed at a scope is visible there and below but not at
        /// its ancestors, so an ancestor's negated row that names a class
        /// member has, at @p scopeId, a canonical spelling its own run does
        /// not carry. Every such row (member postings over known statements:
        /// negated, watch-shaped, at a strict ancestor, an argument among
        /// the members) is recorded in the run `(scopeId, core)` with its
        /// canonical id under the classes visible at @p scopeId, and staged
        /// for a re-check. The mirror of `applyEquiClasses`' "descendant
        /// classes rewrite ancestor facts" pass, as ledger entries instead
        /// of statement rows.
        /// Positive statements and negated equalities are excluded before
        /// candidate collection and sorting; eligible sources retain the
        /// same decoded expression/scope order.
        ///
        /// Sources already recorded at @p scopeId are dropped before the
        /// sort. A source whose tokens have the same representatives at its
        /// own scope and at @p scopeId takes its own scope's canonical id
        /// from @p cache, computed once for every target of a refresh.
        ///
        /// @param mb      The owning LB.
        /// @param scopeId The scope whose class bucket just gained a class.
        /// @param members The merged class's member ids.
        /// @param cache   The refresh's canonical forms; `nullptr` uses a
        ///                cache local to this call.
        /// @return (void)
        /// @see importNegationIntoDescendantClassScopes; recanonicalizeWatchLedgers.
void WatchProcessor::importAncestorNegationsForClass(Memory& mb,
    NameId scopeId, const PagedVector<NameId>& members, CanonicalFormCache* cache) {
    RT_SCOPE_HERE("CANONICAL_ANCESTOR_NEGATIONS");
    assert(scopeId != 0);
    if (scopeId == NameMap::MAIN_ID || members.empty()) return;
    const unsigned slot = gl::WorkerContext::coreId() >= 0 ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& arena = genScratchArenas().forSlot(slot);
    ScratchScope frame(arena);
    DirtyState dirty = DirtyState::Clean;
    ensureCanonicalStatementUses(mb, arena);
    ensureCanonicalWatchUses(mb);
    TypedColdSet<int64_t> selected(&arena, &dirty);
    for (NameId source = mb.nameMap.parentOf(scopeId); source != 0; source = mb.nameMap.parentOf(source))
        for (int32_t m = 0; m < members.size(); ++m)
            mb.lbMemory.canonicalUses.statements.forEach(source, mb.nameMap.decodeView(members[m]),
                [&](int64_t key) {
                    const auto original = Codec<StatementKey>::decode(key).orig;
                    const StrSpan text = mb.nameMap.decodeView(original);
                    if (text.empty() || text.ptr[0] != '!' || canonicalizer.isNegatedEquality(text)) return;
                    selected.mint(key);
                });
    // Existing entries were recanonicalized at this refresh boundary; they are
    // dropped before the sort. An entry recorded by this loop itself is caught
    // by the second presence test below.
    PagedVector<int64_t> candidates(&arena, &dirty);
    for (int32_t i = 1; i <= selected.count(); ++i) {
        const int64_t key = selected.keyAt(i);
        if (mb.lbMemory.canonicalUses.watches.contains(
                packStatementKey(Codec<StatementKey>::decode(key).orig, scopeId))) continue;
        candidates.push_back(key);
    }
    const auto less = [&](int32_t a, int32_t b) {
        const auto left = Codec<StatementKey>::decode(candidates[a]);
        const auto right = Codec<StatementKey>::decode(candidates[b]);
        const int textOrder = compareSpans(mb.nameMap.decodeView(left.orig), mb.nameMap.decodeView(right.orig));
        return textOrder != 0 ? textOrder < 0
            : compareSpans(mb.nameMap.decodeView(left.validity), mb.nameMap.decodeView(right.validity)) < 0;
    };
    ChunkSortedOrdinals<decltype(less)> order(arena, candidates.size(), 0, less);
    std::optional<CanonicalFormCache> ownForms;
    if (cache == nullptr) cache = &ownForms.emplace(arena);
    for (int32_t i = 0; i < candidates.size(); ++i) {
        const auto source = Codec<StatementKey>::decode(candidates[order.next()]);
        // Pending new arrivals still take the normal import path below.
        if (mb.lbMemory.canonicalUses.watches.contains(packStatementKey(source.orig, scopeId))) continue;
        const StrSpan text = mb.nameMap.decodeView(source.orig);
        assert(!text.empty() && text.ptr[0] == '!' && !canonicalizer.isNegatedEquality(text));
        ScratchArena& strings=scratchArenas().forSlot(slot);
        ScratchScope textFrame(strings);
        const ScratchString expression=ScratchString::copyFrom(strings,text.ptr,text.len);
        const StrSpan scopeView=mb.nameMap.decodeView(source.validity);
        const ScratchString sourceScope=ScratchString::copyFrom(strings,scopeView.ptr,scopeView.len);
        const IntEncodedExpr row=encodeExpression(StrSpan(expression),StrSpan(sourceScope),mb.nameMap);
        if (row.isAnchor || !watchShaped(mb, row)) continue;
        const NameId canonical = canonicalIdCached(mb, *cache, row.originalId, source.validity, scopeId);
        const int64_t key = watchKey(scopeId, row.nameId);
        const int64_t value = packInt32Pair(canonical, row.originalId);
        mb.negationLedger.insertSorted(key, value, [](int64_t a, int64_t b) { return a < b; });
        const int64_t dependencyKey=packStatementKey(row.originalId,scopeId);
        canonicalTemplates.indexCanonicalText(mb.lbMemory.canonicalUses.watches,dependencyKey,scopeId,mb.nameMap.decodeView(row.originalId));
        canonicalTemplates.indexCanonicalText(mb.lbMemory.canonicalUses.watches,dependencyKey,scopeId,mb.nameMap.decodeView(canonical));
        mb.pendingWatchRechecks.push_back(packInt32Pair(scopeId, row.originalId));
    }
}

/// @brief Drain-time hook: import a newly registered negated row
        ///        into every descendant scope whose classes can rewrite it.
        ///
        /// @details
        /// The registration-time twin of `importAncestorNegationsForClass`:
        /// the class buckets at strict descendants of the row's scope are
        /// scanned for a member among the row's arguments; a hit records
        /// the row in that scope's run and stages a re-check there. The
        /// canonical id at each such scope comes from @p cache, shared with
        /// the row's own scope when the representatives agree.
        ///
        /// @param mb    The owning LB.
        /// @param ie    The negated row (ids minted, watch-shaped).
        /// @param cache The drain's canonical forms; `nullptr` uses a cache
        ///              local to this call.
        /// @return (void)
        /// @see materializeWatchHits.
void WatchProcessor::importNegationIntoDescendantClassScopes(Memory& mb,
    const IntEncodedExpr& ie, CanonicalFormCache* cache) {
    assert(ie.negation && "importNegationIntoDescendantClassScopes: negated rows only");
    const int32_t vCount = mb.equivalenceClassesMap.count();
    if (vCount == 0) return;
    const unsigned gSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(gSlot);
    std::optional<CanonicalFormCache> ownForms;
    if (cache == nullptr) cache = &ownForms.emplace(gArena);
    for (int32_t kid = 1; kid <= vCount; ++kid) {
        const NameId vid = mb.equivalenceClassesMap.keyAt(kid);
        if (!mb.nameMap.isStrictAncestor(ie.validityId, vid)) continue;
        if (mb.phase3IgnoresScope(vid)) continue;
        if(mb.equivalenceClassesMap.runLen(kid)==0) continue;
        canonicalizer.initializeCanonicalScope(mb,vid);
        const int32_t bucketId = mb.canonicalEquivalenceClassesMap.lookup(vid);
        const int32_t cc = (bucketId == 0) ? 0
            : mb.canonicalEquivalenceClassesMap.runLen(bucketId);
        bool named = false;
        for (int32_t ci = 0; ci < cc && !named; ++ci) {
            ScratchScope viewScope(gArena);
            int32_t blen = 0;
            const char* bp = mb.canonicalEquivalenceClassesMap.peekRecordBytes(
                bucketId, ci, blen, gArena);
            const EquivalenceClassView view{ bp, blen };
            const int32_t mc = view.memberCount();
            for (int32_t m = 0; m < mc && !named; ++m) {
                const NameId member = view.memberId(m);
                for (int32_t a = 0; a < ie.arity; ++a)
                    if (ie.argId[a] == member) { named = true; break; }
            }
        }
        if (!named) continue;
        const NameId canonical = canonicalIdCached(mb, *cache, ie.originalId, ie.validityId, vid);
        if (watchRunContainsValue(mb.negationLedger, watchKey(vid, ie.nameId),
                packInt32Pair(canonical, ie.originalId))) continue;
        recordWatchCanonical(mb.negationLedger, mb, ie.originalId, vid, ie.nameId, canonical);
        mb.pendingWatchRechecks.push_back(packInt32Pair(vid, ie.originalId));
    }
}

/// @brief The registry commit shared by the equivalence-product drain
        ///        and the watch twins: one derived local row, no door.
        ///
        /// @details
        /// Levels row (asserted non-empty and well-shaped), known entry
        /// (local), local-set membership, the three encoded lists and the
        /// negation ledger — byte for byte the block `drainStagedEquiProducts`
        /// committed inline before. Writes no history: the caller records
        /// the row's own origin. A row already carrying a levels row asserts
        /// (the callers probe `known` first).
        ///
        /// @param mb      The owning LB.
        /// @param text    The row's text.
        /// @param scopeId Its scope id (minted).
        /// @param lv      Ascending level run.
        /// @param lvN     Its length (> 0).
        /// @return The committed row (id form).
        /// @see drainStagedEquiProducts; writeWatchTwinCanonical.
IntEncodedExpr WatchProcessor::commitDerivedLocalRow(Memory& mb,
    StrSpan text, NameId scopeId, const int* lv, int32_t lvN) {
    assert(lvN > 0 && ((lvN == 1 && lv[0] == -1) || lv[0] >= 0)
        && "derived-row commit: invalid statement level run");
    const unsigned slot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : scratchArenas().slotCount() - 1;
    ScratchArena& strArena = scratchArenas().forSlot(slot);
    ScratchScope scope(strArena);
    const StrSpan tView = mb.nameMap.decodeView(scopeId);
    const ScratchString tName =
        ScratchString::copyFrom(strArena, tView.ptr, tView.len);
    const NameId origId = mb.nameMap.encode(text);
    const int64_t pk = packStatementKey(origId, scopeId);
    assert(mb.intStatementLevelsMap.lookup(pk) == 0
        && "derived-row commit: the row is already known");
    mb.intStatementLevelsMap.assignSetRange(pk, lv, lv + lvN);
    gl::StatementRegistryOperations::publishKnownStatementKey(mb, pk, /*local=*/true);
    mb.intLocalEncodedStatementsSet.mint(pk);
    const IntEncodedExpr ie = encodeExpression(text, StrSpan(tName), mb.nameMap);
    this->recordNegationWatchFor(mb, ie);
    mb.intEncodedStatements.push_back(ie);
    mb.intLocalEncodedStatements.push_back(ie);
    mb.intLocalEncodedStatementsDelta.push_back(ie);
    return ie;
}

/// @brief Materialize a selected canonical fact with its equality proof bridge.
/// @details Known twins and unchanged forms require no new row. A newly
/// materialized twin is local, carries all applied equality levels, and enters
/// the ordinary negation and OR fact notifications. OR callers can include
/// equality and inequality refuters while other callers preserve their spelling.
/// @param mb Owning block and known registries.
/// @param srcOrigId Known source expression.
/// @param srcScopeId Scope where the source is known.
/// @param atScopeId Comparable scope where the twin is required.
/// @param rewriteEqualities Include standalone equality and inequality views.
/// @return True when a new canonical row is materialized.
bool WatchProcessor::writeWatchTwinCanonical(Memory& mb, NameId srcOrigId,
    NameId srcScopeId, NameId atScopeId, bool rewriteEqualities) {
    assert(srcOrigId != 0 && srcScopeId != 0 && atScopeId != 0
        && "writeWatchTwinCanonical: ids must be minted");
    const unsigned slot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : scratchArenas().slotCount() - 1;
    ScratchArena& strArena = scratchArenas().forSlot(slot);
    ScratchScope scope(strArena);
    // Levels: the source row's own run seeds the union.
    int lv[256];
    int32_t lvN = 0;
    {
        const int32_t lvId = mb.intStatementLevelsMap.lookup(
            packStatementKey(srcOrigId, srcScopeId));
        assert(lvId != 0 && mb.intStatementLevelsMap.runLen(lvId) > 0
            && "writeWatchTwinCanonical: known source without a levels row");
        lvN = coldIntRunAt(mb.intStatementLevelsMap, lvId, lv, 256);
    }
    WatchBridge br;
    canonicalWatchFormAt(mb, srcOrigId, atScopeId, strArena, br, lv, lvN, 256,
        rewriteEqualities);
    if (!br.changed) return false;   // the source is the canonical spelling
    const NameId twinId = mb.nameMap.encode(br.text);
    if (gl::StatementKnowledge::ancestorKnown(mb, twinId, atScopeId, /*includeSelf=*/true))
        return false;                // exact twin already present (I-187)
    if (lvN == 0) { lv[0] = -1; lvN = 1; }   // all-non-derived union re-enters as {-1}
    const IntEncodedExpr twin =
        commitDerivedLocalRow(mb, br.text, atScopeId, lv, lvN);
    const StrSpan atView = mb.nameMap.decodeView(atScopeId);
    const ScratchString atName =
        ScratchString::copyFrom(strArena, atView.ptr, atView.len);
    if (parameters.trackHistory && !ceFilteringActive
        && !proofHistory.originRowExists(mb, br.text, StrSpan(atName))) {
        const StrSpan srcView = mb.nameMap.decodeView(srcOrigId);
        const ScratchString srcText =
            ScratchString::copyFrom(strArena, srcView.ptr, srcView.len);
        const StrSpan srcScopeView = mb.nameMap.decodeView(srcScopeId);
        const ScratchString srcScope =
            ScratchString::copyFrom(strArena, srcScopeView.ptr, srcScopeView.len);
        OriginDep deps[1 + WatchBridge::CAP];
        deps[0] = OriginDep{ StrSpan(srcText), StrSpan(srcScope) };
        for (int32_t k = 0; k < br.eqN; ++k)
            deps[1 + k] = OriginDep{ br.eqText[k], br.eqScope[k] };
        const int cap = parameters.compressor_mode
            ? parameters.compressor_max_origins_per_expr
            : parameters.max_origin_per_expr;
        addOriginEncoded(mb.exprOriginMap, mb.originInterner, br.text,
            StrSpan(atName), OriginTag::equality1, deps, 1 + br.eqN, cap);
    }
    (void)twin;
    caseProcessor.ordisMerge(br.text, StrSpan(atName), lv, lvN, mb);
    return true;
}

/// @brief Materialize a statement's LITERAL spelling from its
        ///        canonical twin — a goal keeps the conjecture's text, a
        ///        branch assumption's negation keeps the scope name's text —
        ///        so the exact probes fire without re-spelling anything.
        ///
        /// @details
        /// Precondition: the literal's canonical form (under the classes
        /// visible at @p scope) is known at @p scope or an ancestor — a
        /// watch hit guarantees it (the peer row or its canonical twin).
        /// The bridge cites that canonical row and, per applied pair, the
        /// MIRROR equality `(=[canonical,member])` (co-registered with every
        /// equality, I-180). Defined no-ops: the literal is already
        /// canonical; the literal text is already known at the scope or an
        /// ancestor.
        ///
        /// @param mb            The owning LB.
        /// @param literalOrigId The literal text's id (minted).
        /// @param scope         The scope the literal row is deposited at.
        /// @return `true` when the literal row was written.
        /// @see writeWatchTwinCanonical; materializeWatchHits.
bool WatchProcessor::writeLiteralTwin(Memory& mb, NameId goalOrigId,
    NameId goalScope) {
    assert(goalOrigId != 0 && goalScope != 0
        && "writeLiteralTwin: ids must be minted");
    if (gl::StatementKnowledge::ancestorKnown(mb, goalOrigId, goalScope, /*includeSelf=*/true))
        return false;                // the literal text is already known
    const unsigned slot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : scratchArenas().slotCount() - 1;
    ScratchArena& strArena = scratchArenas().forSlot(slot);
    ScratchScope scope(strArena);
    int lv[256];
    int32_t lvN = 0;
    WatchBridge br;
    canonicalWatchFormAt(mb, goalOrigId, goalScope, strArena, br, lv, lvN, 256);
    if (!br.changed) return false;   // the goal IS canonical: its twin is the goal text
    // The canonical row: known at the goal scope or an ancestor (the hit's
    // peer or its canonical twin) — the source of the literal.
    const NameId canonId = mb.nameMap.lookup(br.text);
    assert(canonId != 0 && "writeLiteralTwin: canonical text never interned");
    NameId srcScope = 0;
    for (NameId s = goalScope; s != 0; s = mb.nameMap.parentOf(s)) {
        if (mb.intKnownStatements.find(StatementKey{ canonId, s }) != nullptr) {
            srcScope = s; break;
        }
        if (s == NameMap::MAIN_ID) break;
    }
    assert(srcScope != 0
        && "writeLiteralTwin: the literal's canonical form is not known at its scope or an ancestor");
    {
        const int32_t lvId = mb.intStatementLevelsMap.lookup(
            packStatementKey(canonId, srcScope));
        assert(lvId != 0 && mb.intStatementLevelsMap.runLen(lvId) > 0
            && "writeLiteralTwin: canonical row without a levels row");
        int srcLv[256];
        const int32_t srcN = coldIntRunAt(mb.intStatementLevelsMap, lvId, srcLv, 256);
        for (int32_t j = 0; j < srcN; ++j) {
            if (srcLv[j] < 0) continue;
            lvN = insertLevelSorted(lv, lvN, srcLv[j], 256);
        }
        // Strip a leading non-derived marker the union may still carry.
        int32_t lead = 0;
        while (lead < lvN && lv[lead] < 0) ++lead;
        if (lead > 0) { for (int32_t j = lead; j < lvN; ++j) lv[j - lead] = lv[j]; lvN -= lead; }
        if (lvN == 0) { lv[0] = -1; lvN = 1; }
    }
    const StrSpan goalView = mb.nameMap.decodeView(goalOrigId);
    const ScratchString goalText =
        ScratchString::copyFrom(strArena, goalView.ptr, goalView.len);
    const StrSpan gView = mb.nameMap.decodeView(goalScope);
    const ScratchString gName = ScratchString::copyFrom(strArena, gView.ptr, gView.len);
    const IntEncodedExpr twin =
        commitDerivedLocalRow(mb, StrSpan(goalText), goalScope, lv, lvN);
    (void)twin;
    if (parameters.trackHistory && !ceFilteringActive
        && !proofHistory.originRowExists(mb, StrSpan(goalText), StrSpan(gName))) {
        const StrSpan sView = mb.nameMap.decodeView(srcScope);
        const ScratchString srcScopeName =
            ScratchString::copyFrom(strArena, sView.ptr, sView.len);
        OriginDep deps[1 + WatchBridge::CAP];
        deps[0] = OriginDep{ br.text, StrSpan(srcScopeName) };
        for (int32_t k = 0; k < br.eqN; ++k) {
            // Mirror `(=[member,canonical])` -> `(=[canonical,member])`:
            // the literal is reached from the canonical row, so each
            // differing position cites the reverse orientation
            // (co-registered with every equality, I-180). Member names
            // carry no comma or bracket, so the split is exact.
            const StrSpan e = br.eqText[k];
            assert(e.len > 5 && e.ptr[0] == '(' && e.ptr[1] == '=' && e.ptr[2] == '['
                && e.ptr[e.len - 2] == ']' && e.ptr[e.len - 1] == ')'
                && "writeLiteralTwin: justifier is not an equality text");
            int32_t comma = -1;
            for (int32_t i = 3; i < e.len - 2; ++i) if (e.ptr[i] == ',') { comma = i; break; }
            assert(comma > 3 && "writeLiteralTwin: equality without two arguments");
            const StrSpan a(e.ptr + 3, comma - 3);
            const StrSpan b(e.ptr + comma + 1, e.len - 2 - (comma + 1));
            char* buf = strArena.allocBytes(e.len);
            int32_t at = 0;
            buf[at++] = '('; buf[at++] = '='; buf[at++] = '[';
            std::memcpy(buf + at, b.ptr, static_cast<std::size_t>(b.len)); at += b.len;
            buf[at++] = ',';
            std::memcpy(buf + at, a.ptr, static_cast<std::size_t>(a.len)); at += a.len;
            buf[at++] = ']'; buf[at++] = ')';
            assert(at == e.len);
            const StrSpan mirror(buf, e.len);
            const StrSpan cite = proofHistory.findEqualityCiteScope(mb, mirror, br.eqScope[k]);
            const ScratchString citeOwned =
                ScratchString::copyFrom(strArena, cite.ptr, cite.len);
            deps[1 + k] = OriginDep{ mirror, StrSpan(citeOwned) };
        }
        const int cap = parameters.compressor_mode
            ? parameters.compressor_max_origins_per_expr
            : parameters.max_origin_per_expr;
        addOriginEncoded(mb.exprOriginMap, mb.originInterner, StrSpan(goalText),
            StrSpan(gName), OriginTag::equality1, deps, 1 + br.eqN, cap);
    }
    caseProcessor.ordisMerge(StrSpan(goalText), StrSpan(gName), lv, lvN, mb);
    return true;
}

/// @brief Book a branch conclusion justified by its visible equalities.
///
/// @details
/// Retains the actual source and both canonicalization bridges without adding
/// a target statement. The source and target must have the same canonical form
/// at the live branch. Saved support and combined levels survive branch freeze;
/// ordinary convergence publishes the parent conclusion and exact dependencies.
///
/// @param mb The resident owning logical block.
/// @param targetId The fixed parent conclusion's expression identifier.
/// @param branchId The live OR case scope.
/// @param sourceId The actual supporting expression identifier.
/// @param sourceScope Its known scope, at the case or an ancestor.
/// @return Nothing; duplicate support preserves the first complete record.
/// @invariant No target statement or sibling equality context is created.
/// @see ordisMerge, materializeWatchHits, orBranchGoalsResolved.
void WatchProcessor::bookOrEqualitySupport(Memory& mb, NameId targetId,
    NameId branchId, NameId sourceId, NameId sourceScope) {
    const StatementKey target{ targetId, branchId };
    if (mb.lbMemory.orSupportFacts.lookup(target) != 0) return;
    assert(mb.nameMap.ancContains(branchId, sourceScope));
    assert(mb.intKnownStatements.find(StatementKey{ sourceId, sourceScope }) != nullptr);
    assert(mb.frozenOrBranches.lookup(branchId) == 0);
    assert(mb.intValidityNamesToFilter.lookup(branchId) == 0);
    StrSpan signature, disjunct;
    assert(gl::ScopeSyntax::classifyOrScopeView(mb.nameMap, branchId, signature, disjunct)
        == OrScopeKind::Disintegration);
    const unsigned slot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
    ScratchArena& arena = scratchArenas().forSlot(slot);
    ScratchScope scratch(arena);
    WatchBridge source, conclusion;
    int levels[256];
    int32_t levelCount = 0;
    canonicalWatchFormAt(mb, sourceId, branchId, arena, source, levels, levelCount, 256);
    canonicalWatchFormAt(mb, targetId, branchId, arena, conclusion, levels, levelCount, 256);
    assert(equalSpans(source.text, conclusion.text)
        && "OR contribution requires equal branch-local canonical forms");
    assert(source.eqN + conclusion.eqN > 0);
    const int32_t sourceLevels = mb.intStatementLevelsMap.lookup(
        StatementKey{ sourceId, sourceScope });
    assert(sourceLevels != 0);
    for (int32_t j = 0; j < mb.intStatementLevelsMap.runLen(sourceLevels); ++j) {
        const int level = mb.intStatementLevelsMap.valueAt(sourceLevels, j);
        if (level >= 0) levelCount = insertLevelSorted(levels, levelCount, level, 256);
    }
    if (levelCount == 0) levels[levelCount++] = -1;
    mb.lbMemory.closedOrGoalLevels.assignSet(target, levels, levelCount);
    mb.lbMemory.orSupportFacts.upsert(target, packStatementKey(sourceId, sourceScope));
    constexpr int capacity = ExecutionParameters::MAX_ARITY;
    StrSpan sourceArgs[capacity], targetArgs[capacity];
    const int32_t arity = getArgsSpans(mb.nameMap.decodeView(sourceId), sourceArgs, capacity);
    assert(arity == getArgsSpans(mb.nameMap.decodeView(targetId), targetArgs, capacity));
    StrSpan changedArgs[2 * capacity];
    int32_t changedCount = 0;
    for (int32_t j = 0; j < arity; ++j) if (!equalSpans(sourceArgs[j], targetArgs[j])) {
        changedArgs[changedCount++] = StrSpan(ScratchString::copyFrom(arena, sourceArgs[j].ptr, sourceArgs[j].len));
        changedArgs[changedCount++] = StrSpan(ScratchString::copyFrom(arena, targetArgs[j].ptr, targetArgs[j].len));
    }
    assert(changedCount > 0);
    const auto appendBridge = [&](const WatchBridge& bridge) {
        for (int32_t j = 0; j < bridge.eqN; ++j) {
            StrSpan pair[2];
            assert(getArgsSpans(bridge.eqText[j], pair, 2) == 2);
            bool relevant = false;
            for (int32_t k = 0; k < changedCount; ++k)
                relevant = relevant || equalSpans(pair[0], changedArgs[k]);
            if (!relevant) continue;
            const NameId equality = mb.nameMap.encode(bridge.eqText[j]);
            const NameId scope = mb.nameMap.encode(bridge.eqScope[j]);
            assert(mb.nameMap.ancContains(branchId, scope));
            mb.lbMemory.orSupportEqualities.insertSorted(target,
                packStatementKey(equality, scope), [&](int64_t a, int64_t b) {
                    const StatementKey x = Codec<StatementKey>::decode(a);
                    const StatementKey y = Codec<StatementKey>::decode(b);
                    const int order = compareSpans(mb.nameMap.decodeView(x.orig), mb.nameMap.decodeView(y.orig));
                    return order != 0 ? order < 0
                        : compareSpans(mb.nameMap.decodeView(x.validity), mb.nameMap.decodeView(y.validity)) < 0;
                });
        }
    };
    appendBridge(source);
    appendBridge(conclusion);
    const StrSpan tv = mb.nameMap.decodeView(targetId);
    const ScratchString text = ScratchString::copyFrom(arena, tv.ptr, tv.len);
    const StrSpan bv = mb.nameMap.decodeView(branchId);
    const ScratchString branch = ScratchString::copyFrom(arena, bv.ptr, bv.len);
    caseProcessor.ordisMerge(StrSpan(text), StrSpan(branch), levels, levelCount, mb);
}

/// @brief The per-step watch drain: turn every canonical
        ///        contradiction / goal pair into the exact pair the sweeps
        ///        fire on.
        ///
        /// @details
        /// Runs once per `standardProcessing` after `applyEquiClasses` and
        /// its cleanup and before `dischargeContradiction` /
        /// `dischargeToBeProved`, single-threaded (I-66, I-28). Four
        /// sources of hits: (a) a POSITIVE arrival at S probes the negation
        /// runs at S and every ancestor for `gl::DefinitionExpansion::negate(canonical_S(arrival))`;
        /// (b) a NEGATED arrival (or a re-checked entry) at S first imports
        /// itself into the descendant class scopes, then — when its
        /// positive core is used as a premise, so the positive rows are
        /// multiplied — probes `intKnownStatements` for its canonical
        /// positive at S and every ancestor, else files a batch request;
        /// (c) every goal (few) does the same with its own sign; (d) one
        /// sweep of `intEncodedStatements` serves every batch request (rows
        /// filtered by core, sign and scope before any canonicalization).
        /// Ordinary hits materialize the canonical twin of each side at the
        /// probing scope (`writeWatchTwinCanonical`; goals also
        /// `writeLiteralTwin`). (f) Every live `_ordis_` branch whose
        /// assumption's negation is not known exactly but is canonically
        /// equal (under the classes visible at the branch) to a negation
        /// entry at the branch or above gets that entry's canonical twin
        /// and the literal negation deposited at the branch, so the
        /// existing exact refutation probes (`refutedOrBranchAtOrAbove`,
        /// `ordisMerge`) retire it. Both pending lists are cleared at the
        /// end. Existing ancestor goals are also compared at every live OR
        /// case. These hits call `bookOrEqualitySupport` instead of writing
        /// either twin: actual facts and local equalities enter bookkeeping.
        /// With `skip_eq_classes` there are no classes and the exact
        /// sweeps already see every pair: the lists are just cleared.
        ///
        /// @param mb     The owning LB.
        /// @param coreId The worker slot (scratch arenas).
        /// @return (void)
        /// @invariant Never discharges, never writes another LB; every row
        ///            it writes is a derived local row with a bridge.
        /// @see stageWatchArrival; recanonicalizeWatchLedgers.
void WatchProcessor::materializeWatchHits(Memory& mb, int coreId) {
    if (parameters.skip_eq_classes) {
        // No classes: canonical == original and the exact sweeps already see
        // every pair (incubator mode, OPEN-CFG-1).
        mb.pendingWatchArrivals.clear();
        mb.pendingWatchRechecks.clear();
        return;
    }
    if (mb.pendingWatchArrivals.size() == 0 && mb.pendingWatchRechecks.size() == 0
        && mb.intToBeProved.count() == 0) return;
    ensureGoalLedgerCurrent(mb);

    const unsigned gSlot = (coreId >= 0)
        ? static_cast<unsigned>(coreId) : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(gSlot);
    const ArenaOffset gMark = gArena.cursor();
    const unsigned sSlot = (coreId >= 0)
        ? static_cast<unsigned>(coreId) : scratchArenas().slotCount() - 1;
    ScratchArena& sArena = scratchArenas().forSlot(sSlot);

    // kind 0: contradiction — `a` the positive side, `b` the negated side.
    // kind 1: goal — `a` the peer row, `b` the goal.
    struct Hit { NameId aOrig, aScope, bOrig, bScope, atScope; uint8_t kind; };
    struct Req { NameId scope, core, canonId, entryOrig, entryScope; uint8_t sign, kind; };
    DirtyState dirty = DirtyState::Clean;
    PagedVector<Hit> hits(&gArena, &dirty);
    PagedVector<Req> batch(&gArena, &dirty);
    // Canonical forms of this drain. The probes and the sweep change no
    // class; a scope first initialized by an arrival import renews it.
    CanonicalFormCache forms(gArena);

    // The scope a ledger entry's row lives at: the shallowest scope at or
    // above `runScope` where it is known (an imported entry's row sits at
    // an ancestor of its run scope; an erased row has none — dead entry).
    const auto entryRowScope = [&](NameId origId, NameId runScope) -> NameId {
        NameId found = 0;
        for (NameId s = runScope; s != 0; s = mb.nameMap.parentOf(s)) {
            if (mb.intKnownStatements.find(StatementKey{ origId, s }) != nullptr)
                found = s;
            if (s == NameMap::MAIN_ID) break;
        }
        return found;
    };
    const auto knownAtOrAbove = [&](NameId origId, NameId scope) -> NameId {
        for (NameId s = scope; s != 0; s = mb.nameMap.parentOf(s)) {
            if (mb.intKnownStatements.find(StatementKey{ origId, s }) != nullptr)
                return s;
            if (s == NameMap::MAIN_ID) break;
        }
        return 0;
    };

    // (a) a positive row at S against the negation runs at S and above.
    const auto probePositive = [&](NameId origId, NameId S, NameId core) {
        // No negation run for this core anywhere on the scope chain: no
        // entry can match, so the canonicalization is skipped.
        bool anyRun = false;
        for (NameId T = S; T != 0 && !anyRun; T = mb.nameMap.parentOf(T)) {
            anyRun = mb.negationLedger.lookup(watchKey(T, core)) != 0;
            if (T == NameMap::MAIN_ID) break;
        }
        if (!anyRun) return;
        ScratchScope sc(sArena);
        const StrSpan negText = gl::DefinitionExpansion::negateScratch(sArena,
            canonicalTextCached(mb, forms, origId, S, S, sArena));
        const NameId negId = mb.nameMap.lookup(negText);
        if (negId == 0) return;   // never interned: no entry can carry it
        for (NameId T = S; T != 0; T = mb.nameMap.parentOf(T)) {
            NameId entryOrig = 0;
            if (watchRunHasCanonical(mb.negationLedger, T, core, negId, &entryOrig)) {
                const NameId eScope = entryRowScope(entryOrig, T);
                if (eScope != 0)
                    hits.push_back(Hit{ origId, S, entryOrig, eScope, S, 0 });
            }
            if (T == NameMap::MAIN_ID) break;
        }
    };
    // (b)/(c) an entry (negated row or goal) against the rows of the
    // opposite / same polarity at S and above. An entry copied into a
    // descendant class scope usually keeps the spelling of its own scope;
    // the cache then computes that form once for every copy.
    const auto probeEntry = [&](NameId entryOrig, NameId entryScope, NameId S,
                                NameId core, uint8_t kind) {
        ScratchScope sc(sArena);
        const StrSpan text = canonicalTextCached(mb, forms, entryOrig, entryScope, S, sArena);
        StrSpan peerText = text;
        uint8_t sign = (peerText.len > 0 && peerText.ptr[0] == '!') ? 1 : 0;
        if (kind == 0) { peerText = gl::DefinitionExpansion::negateScratch(sArena, text); sign = sign ? 0 : 1; }
        if (usedAsPremiseCore(mb, core, sign)) {
            const NameId peerId = mb.nameMap.lookup(peerText);
            if (peerId == 0) return;
            const NameId at = knownAtOrAbove(peerId, S);
            if (at == 0) return;
            hits.push_back(Hit{ peerId, at, entryOrig, entryScope, S, kind });
            return;
        }
        // Not multiplied: the batch sweep canonicalizes the candidate rows.
        batch.push_back(Req{ S, core, mb.nameMap.encode(peerText), entryOrig,
                             entryScope, sign, kind });
    };

    // Arrivals.
    for (int32_t i = 0; i < mb.pendingWatchArrivals.size(); ++i) {
        const StatementKey sk = Codec<StatementKey>::decode(mb.pendingWatchArrivals[i]);
        if (mb.intKnownStatements.find(StatementKey{ sk.orig, sk.validity }) == nullptr)
            continue;   // erased since registration
        ScratchScope sc(sArena);
        const StrSpan tView = mb.nameMap.decodeView(sk.orig);
        const ScratchString text = ScratchString::copyFrom(sArena, tView.ptr, tView.len);
        const StrSpan vView = mb.nameMap.decodeView(sk.validity);
        const ScratchString vName = ScratchString::copyFrom(sArena, vView.ptr, vView.len);
        const IntEncodedExpr ie = encodeExpression(StrSpan(text), StrSpan(vName), mb.nameMap);
        if (ie.negation) {
            importNegationIntoDescendantClassScopes(mb, ie, &forms);
            probeEntry(sk.orig, sk.validity, sk.validity, ie.nameId, 0);
        } else {
            probePositive(sk.orig, sk.validity, ie.nameId);
        }
    }
    // Re-checks (moved or imported negation entries). The ledger keys every
    // entry by its core name, read here without re-encoding the row.
    for (int32_t i = 0; i < mb.pendingWatchRechecks.size(); ++i) {
        const int64_t v = mb.pendingWatchRechecks[i];
        const NameId runScope = static_cast<NameId>(static_cast<uint64_t>(v) >> 32);
        const NameId entryOrig = static_cast<NameId>(static_cast<uint64_t>(v) & 0xffffffffu);
        const NameId eScope = entryRowScope(entryOrig, runScope);
        if (eScope == 0) continue;   // dead entry
        const NameId core = mb.nameMap.lookup(
            extractExpressionUniversalSpan(mb.nameMap.decodeView(entryOrig)));
        assert(core != 0 && "watch re-check: a recorded negation's core is minted");
        probeEntry(entryOrig, eScope, runScope, core, 0);
    }
    // Goals — every drain, every goal (few).
    {
        const int32_t keyN = mb.goalLedger.count();
        for (int32_t id = 1; id <= keyN; ++id) {
            const int64_t key = mb.goalLedger.decodeKey(id);
            const NameId G = static_cast<NameId>(static_cast<uint64_t>(key) >> 32);
            const NameId core = static_cast<NameId>(static_cast<uint64_t>(key) & 0xffffffffu);
            if (mb.phase3IgnoresScope(G)) continue;
            const int32_t n = mb.goalLedger.runLen(id);
            for (int32_t j = 0; j < n; ++j) {
                const NameId g = gl::CaseLedger::ledgerOriginalId(mb.goalLedger.valueAt(id, j));
                probeEntry(g, G, G, core, 1);
                for (int32_t bi = 1; bi <= mb.orLiveBranches.count(); ++bi) {
                    const NameId branch = mb.orLiveBranches.decode(bi);
                    if (!mb.nameMap.isStrictAncestor(G, branch)) continue;
                    // Only the branch's innermost goal scope counts for its
                    // resolution (D-549).
                    if (gl::StatementKnowledge::innermostGoalScope(mb, branch) != G) continue;
                    if (mb.frozenOrBranches.lookup(branch) != 0
                        || mb.pendingDeadOrBranches.lookup(branch) != 0
                        || mb.intValidityNamesToFilter.lookup(branch) != 0
                        || mb.phase3IgnoresScope(branch)) continue;
                    if (gl::StatementKnowledge::ancestorKnown(mb, g, branch, true)
                        || mb.lbMemory.orSupportFacts.lookup(StatementKey{ g, branch }) != 0) continue;
                    probeEntry(g, G, branch, core, 2);
                }
            }
        }
    }
    // (d) one sweep for every batch request. The requests are grouped by
    // (core, sign), each group in batch order, so a row visits only the
    // requests it can match and hits keep their row-major, batch order.
    if (batch.size() > 0) {
        const int32_t reqN = batch.size();
        const auto groupOf = [&](int32_t q) -> int64_t {
            return packInt32Pair(batch[q].core, batch[q].sign);
        };
        int32_t* byGroup = reinterpret_cast<int32_t*>(gArena.resolve(gArena.alloc(
            reqN * static_cast<int32_t>(sizeof(int32_t)), static_cast<int32_t>(alignof(int32_t)))));
        for (int32_t q = 0; q < reqN; ++q) byGroup[q] = q;
        std::sort(byGroup, byGroup + reqN, [&](int32_t a, int32_t b) {
            const int64_t left = groupOf(a), right = groupOf(b);
            return left != right ? left < right : a < b;
        });
        TypedColdMap<int64_t, int64_t> groups(&gArena, &dirty);
        for (int32_t first = 0; first < reqN; ) {
            int32_t end = first;
            while (end < reqN && groupOf(byGroup[end]) == groupOf(byGroup[first])) ++end;
            groups.insert(groupOf(byGroup[first]), packInt32Pair(first, end - first));
            first = end;
        }
        const int32_t rowN = mb.intEncodedStatements.size();
        for (int32_t r = 0; r < rowN; ++r) {
            const IntEncodedExpr& row = mb.intEncodedStatements[r];
            if (row.isAnchor) continue;
            const int64_t* group = groups.find(packInt32Pair(row.nameId, row.negation ? 1 : 0));
            if (group == nullptr) continue;
            const int32_t first = static_cast<int32_t>(static_cast<uint64_t>(*group) >> 32);
            const int32_t end = first + static_cast<int32_t>(static_cast<uint64_t>(*group) & 0xffffffffu);
            for (int32_t k = first; k < end; ++k) {
                const Req& req = batch[byGroup[k]];
                if (row.validityId != req.scope
                    && !mb.nameMap.isStrictAncestor(row.validityId, req.scope)) continue;
                if (row.originalId == req.entryOrig && row.validityId == req.entryScope) continue;
                if (canonicalIdCached(mb, forms, row.originalId, row.validityId, req.scope)
                        != req.canonId) continue;
                hits.push_back(Hit{ row.originalId, row.validityId, req.entryOrig,
                                    req.entryScope, req.scope, req.kind });
            }
        }
    }
    // (e) twins.
    for (int32_t h = 0; h < hits.size(); ++h) {
        const Hit hit = hits[h];
        if (!mb.isActive) break;   // a sweep-independent discharge already closed the LB
        if (hit.kind == 2) {
            bookOrEqualitySupport(mb, hit.bOrig, hit.atScope, hit.aOrig, hit.aScope);
        } else if (hit.kind == 0) {
            writeWatchTwinCanonical(mb, hit.aOrig, hit.aScope, hit.atScope);
            writeWatchTwinCanonical(mb, hit.bOrig, hit.bScope, hit.atScope);
        } else {
            writeWatchTwinCanonical(mb, hit.aOrig, hit.aScope, hit.atScope);
            writeLiteralTwin(mb, hit.bOrig, hit.bScope);
        }
    }
    // (f) live or-branches: an assumption refuted only up to canonical
    // equality gets its exact negation at the branch.
    {
        const int32_t liveN = mb.orLiveBranches.count();
        for (int32_t li = 1; li <= liveN && mb.isActive; ++li) {
            const NameId B = mb.orLiveBranches.decode(li);
            if (mb.intValidityNamesToFilter.lookup(B) != 0) continue;
            if (mb.frozenOrBranches.lookup(B) != 0) continue;
            if (mb.pendingDeadOrBranches.lookup(B) != 0) continue;
            if (mb.phase3IgnoresScope(B)) continue;
            StrSpan orSigV, bodyV;
            if (gl::ScopeSyntax::classifyOrScopeView(mb.nameMap, B, orSigV, bodyV)
                    != OrScopeKind::Disintegration) continue;
            assert(bodyV.len >= 2 && bodyV.ptr[0] == '(' && bodyV.ptr[bodyV.len - 1] == ')'
                && "watch drain: ordis branch payload must be a wrapped (disjunct)");
            ScratchScope sc(sArena);
            const StrSpan assumptionV(bodyV.ptr + 1, bodyV.len - 2);
            const ScratchString assumption =
                ScratchString::copyFrom(sArena, assumptionV.ptr, assumptionV.len);
            const StrSpan negText = gl::DefinitionExpansion::negateScratch(sArena, StrSpan(assumption));
            {
                const NameId exact = mb.nameMap.lookup(negText);
                if (exact != 0 && knownAtOrAbove(exact, B) != 0) continue;   // exact probes see it
            }
            const StrSpan bView = mb.nameMap.decodeView(B);
            const ScratchString bName = ScratchString::copyFrom(sArena, bView.ptr, bView.len);
            const IntEncodedExpr negIe =
                encodeExpression(negText, StrSpan(bName), mb.nameMap);
            if (!watchShaped(mb, negIe)) continue;
            // `negIe.originalId` is now minted: the literal negation's id.
            const NameId canon = canonicalWatchIdAt(mb, negIe.originalId, B);
            NameId entryOrig = 0;
            NameId entryRun = 0;
            for (NameId T = B; T != 0; T = mb.nameMap.parentOf(T)) {
                if (watchRunHasCanonical(mb.negationLedger, T, negIe.nameId, canon, &entryOrig)) {
                    entryRun = T; break;
                }
                if (T == NameMap::MAIN_ID) break;
            }
            if (entryRun == 0) continue;
            const NameId eScope = entryRowScope(entryOrig, entryRun);
            if (eScope == 0) continue;   // dead entry
            writeWatchTwinCanonical(mb, entryOrig, eScope, B);
            writeLiteralTwin(mb, negIe.originalId, B);
        }
    }
    mb.pendingWatchArrivals.clear();
    mb.pendingWatchRechecks.clear();
    gArena.popTo(gMark);
}

/// @brief Record a statement in a watch ledger.
        ///
        /// @details
        /// Inserts `packInt32Pair(canonicalWatchIdAt(origId, scopeId),
        /// origId)` into the run of `watchKey(scopeId, coreId)`; a
        /// duplicate insert is a defined no-op (sorted-unique run), so the
        /// registration doors may record without a presence probe.
        ///
        /// @param ledger  `mb.negationLedger` or `mb.goalLedger`.
        /// @param mb      Owning LB.
        /// @param origId  The statement's original id (minted).
        /// @param scopeId Its scope id (minted).
        /// @param coreId  Its unsigned core name id.
        /// @return (void)
        /// @see clearWatch; watchRunHasCanonical.
void WatchProcessor::recordWatch(WatchLedger& ledger, Memory& mb,
    NameId origId, NameId scopeId, NameId coreId) {
    assert(origId != 0 && scopeId != 0 && coreId != 0
        && "recordWatch: every id must be minted by the door path");
    recordWatchCanonical(ledger, mb, origId, scopeId, coreId,
        canonicalWatchIdAt(mb, origId, scopeId));
}

/// @brief Record a statement whose canonical id is already known in a watch
///        ledger.
/// @details Inserts `packInt32Pair(canonId, origId)` into the sorted-unique
/// run of `watchKey(scopeId, coreId)` (a duplicate is a no-op) and, for the
/// negation ledger, indexes the original and canonical texts.
/// @param ledger `mb.negationLedger` or `mb.goalLedger`.
/// @param mb Owning LB.
/// @param origId The statement's original id.
/// @param scopeId Its scope id.
/// @param coreId Its unsigned core name id.
/// @param canonId Its canonical id at @p scopeId.
/// @return (void)
void WatchProcessor::recordWatchCanonical(WatchLedger& ledger, Memory& mb,
    NameId origId, NameId scopeId, NameId coreId, NameId canonId) {
    assert(origId != 0 && scopeId != 0 && coreId != 0 && canonId != 0
        && "recordWatchCanonical: every id must be minted");
    ledger.insertSorted(watchKey(scopeId, coreId),
        packInt32Pair(canonId, origId),
        [](int64_t a, int64_t b) { return a < b; });
    if (&ledger == &mb.negationLedger) {
        const int64_t key = packStatementKey(origId,scopeId);
        canonicalTemplates.indexCanonicalText(mb.lbMemory.canonicalUses.watches,key,scopeId,mb.nameMap.decodeView(origId));
        canonicalTemplates.indexCanonicalText(mb.lbMemory.canonicalUses.watches,key,scopeId,mb.nameMap.decodeView(canonId));
    }
}

/// @brief Whether a watch run holds an exact packed entry.
/// @details Binary search over the ascending, sorted-unique run.
/// @param ledger `mb.negationLedger` or `mb.goalLedger`.
/// @param key `watchKey(scope, core)`.
/// @param value `packInt32Pair(canonicalId, originalId)`.
/// @return True when the run of @p key contains @p value.
bool WatchProcessor::watchRunContainsValue(const WatchLedger& ledger,
    int64_t key, int64_t value) {
    const int32_t run = ledger.lookup(key);
    if (run == 0) return false;
    int32_t lo = 0, hi = ledger.runLen(run);
    while (lo < hi) {
        const int32_t mid = lo + (hi - lo) / 2;
        if (ledger.valueAt(run, mid) < value) lo = mid + 1; else hi = mid;
    }
    return lo < ledger.runLen(run) && ledger.valueAt(run, lo) == value;
}

/// @brief Remove a statement's entry from a watch ledger.
        ///
        /// @details
        /// Drops every run value whose original half is @p origId (one at
        /// most: the run is unique per original). The rebuilt run rides the
        /// gen-scratch arena and is written back with `assignSet`.
        ///
        /// @param ledger  `mb.negationLedger` or `mb.goalLedger`.
        /// @param mb      Owning LB.
        /// @param origId  The statement's original id.
        /// @param scopeId Its scope id.
        /// @param coreId  Its unsigned core name id.
        /// @return `true` when an entry was removed; `false` when the run
        ///         held none (a caller that knows the row was recorded
        ///         asserts on the result).
        /// @see recordWatch.
bool WatchProcessor::clearWatch(WatchLedger& ledger, Memory& mb,
    NameId origId, NameId scopeId, NameId coreId) {
    assert(origId != 0 && scopeId != 0 && coreId != 0
        && "clearWatch: every id must be minted");
    const int64_t key = watchKey(scopeId, coreId);
    const int32_t row = ledger.lookup(key);
    if (row == 0) return false;   // no run at this (scope, core)
    const int32_t n = ledger.runLen(row);

    const unsigned slot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(slot);
    const ArenaOffset mark = gArena.cursor();
    int64_t* keep = reinterpret_cast<int64_t*>(gArena.resolve(gArena.alloc(
        (n > 0 ? n : 1) * static_cast<int32_t>(sizeof(int64_t)),
        static_cast<int32_t>(alignof(int64_t)))));
    int32_t keepN = 0;
    for (int32_t j = 0; j < n; ++j) {
        const int64_t v = ledger.valueAt(row, j);
        if (gl::CaseLedger::ledgerOriginalId(v) != origId) keep[keepN++] = v;
    }
    const bool removed = keepN != n;
    if (removed) ledger.assignSet(key, keep, keepN);
    if (removed && &ledger == &mb.negationLedger)
        mb.lbMemory.canonicalUses.watches.erase(packStatementKey(origId,scopeId),gArena);
    gArena.popTo(mark);
    return removed;
}

/// @brief Probe a watch ledger for an entry with a given canonical
        ///        id.
        ///
        /// @details
        /// Binary search over the run of `watchKey(scopeId, coreId)` on the
        /// packed value (canonical id in the high half), so a hit costs
        /// log(run). Non-minting; a scope or core never interned has no
        /// run.
        ///
        /// @param ledger  `mb.negationLedger` or `mb.goalLedger`.
        /// @param scopeId The scope probed.
        /// @param coreId  The unsigned core name id probed.
        /// @param canonId The canonical id looked for.
        /// @param origOut Receives the matching entry's original id on a
        ///                hit (the first of the contiguous block).
        /// @return `true` on a hit.
        /// @see recordWatch; canonicalWatchIdAt.
bool WatchProcessor::watchRunHasCanonical(const WatchLedger& ledger,
    NameId scopeId, NameId coreId, NameId canonId, NameId* origOut) {
    assert(origOut != nullptr && "watchRunHasCanonical: origOut required");
    if (scopeId == 0 || coreId == 0 || canonId == 0) return false;
    const int32_t row = ledger.lookup(watchKey(scopeId, coreId));
    if (row == 0) return false;
    const int32_t n = ledger.runLen(row);
    // Lower bound of the contiguous block for canonId (original half 0 is
    // below every minted original).
    const int64_t probe = packInt32Pair(canonId, 0);
    int32_t lo = 0, hi = n;
    while (lo < hi) {
        const int32_t mid = lo + (hi - lo) / 2;
        if (ledger.valueAt(row, mid) < probe) lo = mid + 1; else hi = mid;
    }
    if (lo >= n) return false;
    const int64_t v = ledger.valueAt(row, lo);
    if (gl::CaseLedger::ledgerCanonicalId(v) != canonId) return false;
    *origOut = gl::CaseLedger::ledgerOriginalId(v);
    return true;
}

/// @brief Registration-door hook: record a freshly known negated
        ///        row in `negationLedger`.
        ///
        /// @details
        /// Called right after every `intKnownStatements` upsert of a
        /// statement (`addStatement`, `drainStagedEquiProducts`, the
        /// status-4 fact load); a positive row, an equality shape or an
        /// anchor is a defined no-op, so the callers need no shape probe.
        ///
        /// @param mb The owning LB.
        /// @param ie The row just registered (ids minted).
        /// @return (void)
        /// @invariant After the call `negationLedger` mirrors
        ///            `intKnownStatements` for the watch-shaped negated rows.
        /// @see clearNegationWatchFor.
void WatchProcessor::recordNegationWatchFor(Memory& mb,
    const IntEncodedExpr& ie) {
    if (!ie.negation || !watchShaped(mb, ie)) return;   // not watched by shape
    recordWatch(mb.negationLedger, mb, ie.originalId, ie.validityId, ie.nameId);
}

/// @brief Registry-erase hook: drop a statement's `negationLedger`
        ///        entry when its known row is erased.
        ///
        /// @details
        /// Called by the two un-know seams (`resetResentExpressionRegistries`,
        /// `resetParkedOrStatementRegistries`) after their
        /// `intKnownStatements` erase with the pre-erase membership. A
        /// statement that was never known had no entry (defined no-op); a
        /// known watch-shaped negated statement MUST have one — a miss
        /// asserts (the ledger would be out of sync with the registry).
        /// Whole-scope teardowns (`wipeSubtree`, discharge) erase the
        /// ledgers by scope themselves.
        ///
        /// @param mb       The owning LB.
        /// @param origId   The statement's original id.
        /// @param valId    Its scope id.
        /// @param wasKnown Whether `intKnownStatements` held the row before
        ///                 the erase.
        /// @return (void)
        /// @see recordNegationWatchFor.
void WatchProcessor::clearNegationWatchFor(Memory& mb, NameId origId,
    NameId valId, bool wasKnown) {
    if (!wasKnown) return;   // never registered: nothing was recorded
    assert(origId != 0 && valId != 0
        && "clearNegationWatchFor: both ids must be minted");
    // Owned scratch copies: encodeExpression re-interns already-minted
    // names only, but a decodeView span must never be held across a NameMap
    // call (I-3).
    const unsigned slot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : scratchArenas().slotCount() - 1;
    ScratchArena& strArena = scratchArenas().forSlot(slot);
    ScratchScope scope(strArena);
    const StrSpan tView = mb.nameMap.decodeView(origId);
    const ScratchString text = ScratchString::copyFrom(strArena, tView.ptr, tView.len);
    const StrSpan vView = mb.nameMap.decodeView(valId);
    const ScratchString vName = ScratchString::copyFrom(strArena, vView.ptr, vView.len);
    const IntEncodedExpr ie =
        encodeExpression(StrSpan(text), StrSpan(vName), mb.nameMap);
    if (!ie.negation || !watchShaped(mb, ie)) return;
    const bool removed =
        clearWatch(mb.negationLedger, mb, origId, valId, ie.nameId);
    assert(removed
        && "clearNegationWatchFor: known negated row without a ledger entry");
    (void)removed;
}

/// @brief Bring the DERIVED `goalLedger` up to date with
        ///        `intToBeProved`.
        ///
        /// @details
        /// The goal ledger is not mirrored at the goal doors: it is rebuilt
        /// from the goal registry whenever the registry's key-set epoch or
        /// its row count moved (every insert bumps the epoch, every erase
        /// changes the count) or a class merge / scope wipe / discharge
        /// marked it stale (`Memory::goalLedgerStale`). Goals are few per
        /// block, so the rebuild is a handful of `canonicalWatchIdAt`
        /// calls, and a test that seeds the registry directly needs no
        /// door. Equality-shaped and anchor goals stay out (`watchShaped`).
        ///
        /// @param mb The owning LB (registry resident).
        /// @return (void)
        /// @invariant After the call every watch-shaped goal row has exactly
        ///            one `goalLedger` entry whose canonical half is
        ///            `canonicalWatchIdAt(original, scope)`.
        /// @see recanonicalizeWatchLedgers; watchRunHasCanonical.
void WatchProcessor::ensureGoalLedgerCurrent(Memory& mb) {
    const uint32_t epoch = mb.intToBeProved.inner().insertEpoch();
    const int32_t count = mb.intToBeProved.count();
    if (!mb.goalLedgerStale && epoch == mb.goalLedgerEpoch
        && count == mb.goalLedgerCount) return;   // current: defined no-op
    mb.goalLedger.resetToFresh();
    for (int32_t id = 1; id <= count; ++id) {
        const StatementKey sk =
            Codec<StatementKey>::decode(mb.intToBeProved.keyAt(id));
        // Owned scratch copies across the NameMap calls below (I-3).
        const unsigned slot = (gl::WorkerContext::coreId() >= 0)
            ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : scratchArenas().slotCount() - 1;
        ScratchArena& strArena = scratchArenas().forSlot(slot);
        ScratchScope scope(strArena);
        const StrSpan tView = mb.nameMap.decodeView(sk.orig);
        const ScratchString text =
            ScratchString::copyFrom(strArena, tView.ptr, tView.len);
        const StrSpan vView = mb.nameMap.decodeView(sk.validity);
        const ScratchString vName =
            ScratchString::copyFrom(strArena, vView.ptr, vView.len);
        const IntEncodedExpr ie =
            encodeExpression(StrSpan(text), StrSpan(vName), mb.nameMap);
        if (!watchShaped(mb, ie)) continue;
        recordWatch(mb.goalLedger, mb, sk.orig, sk.validity, ie.nameId);
    }
    mb.goalLedgerEpoch = epoch;
    mb.goalLedgerCount = count;
    mb.goalLedgerStale = false;
}

/// @brief Re-derive the canonical halves of the negation ledger's
        ///        entries a class merge at @p scopeId can change, and mark the
        ///        derived goal ledger stale.
        ///
        /// @details
        /// Exact-scope dependency postings select original/canonical token
        /// users once across all cores. Live indexed records map directly to
        /// their ledger run; no full-ledger scan or per-row membership scan is
        /// needed for an exact-scope refresh. Run order and decoded lexical
        /// expression order preserve canonical identifier minting. Each touched
        /// run is rebuilt once and moved entries are staged for a re-check.
        /// Goal dependencies synchronize even when no representative changed.
        ///
        /// @param mb Owning block.
        /// @param scopeId Scope whose class bucket changed.
        /// @param members Members with changed normal representatives, including new members.
        /// @param exactScope True limits work to this scope; false includes descendants.
        /// @return The number of entries whose canonical id changed.
        /// @invariant After the call every entry's canonical half equals
        ///            `canonicalWatchIdAt(original, run scope)`.
        /// @see recanonicalizeProcessedOrLedger; importAncestorNegationsForClass;
        ///      updateEquivalenceClasses.
int32_t WatchProcessor::recanonicalizeWatchLedgers(Memory& mb,
    NameId scopeId, const PagedVector<NameId>& members, bool exactScope) {
    RT_SCOPE_HERE("CANONICAL_WATCH_LEDGERS");
    assert(scopeId != 0
        && "recanonicalizeWatchLedgers: the scope id must be minted");
    // Goal keys share the same effective-scope dependency rule.
    const unsigned slot0 = gl::WorkerContext::coreId() >= 0 ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& goalArena = genScratchArenas().forSlot(slot0);
    auto& goalUses = mb.lbMemory.canonicalUses.goals;
    goalUses.synchronize(mb.intToBeProved, goalArena, [&](int64_t key) {
        const auto goal = Codec<StatementKey>::decode(key);
        canonicalTemplates.indexCanonicalText(goalUses, key, goal.validity, mb.nameMap.decodeView(goal.orig));
    });
    const auto invalidateGoalsAt = [&](NameId scope) {
        for (int32_t i = 0; i < members.size(); ++i)
            goalUses.forEach(scope, mb.nameMap.decodeView(members[i]), [&](int64_t) { mb.goalLedgerStale = true; });
    };
    invalidateGoalsAt(scopeId);
    if (!exactScope) {
        for (int32_t i=1;i<=mb.canonicalEquivalenceClassesMap.count();++i) {
            const NameId scope=mb.canonicalEquivalenceClassesMap.keyAt(i);
            if (mb.nameMap.isStrictAncestor(scopeId,scope)) invalidateGoalsAt(scope);
        }
    }
    const int32_t memberN = members.size();
    if (memberN == 0) return 0;
    const unsigned slot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(slot);
    const ArenaOffset mark = gArena.cursor();
    DirtyState dirty = DirtyState::Clean;

    // Select each affected logical record once, independently of core count.
    // The index is maintained by every negation-ledger writer and scope wipe.
    ensureCanonicalWatchUses(mb);
    auto& watchUses = mb.lbMemory.canonicalUses.watches;
    TypedColdSet<int64_t> selected(&gArena, &dirty);
    const auto selectScope = [&](NameId scope) {
        for (int32_t m = 0; m < memberN; ++m)
            watchUses.forEach(scope, mb.nameMap.decodeView(members[m]),
                [&](int64_t key) { selected.mint(key); });
    };
    selectScope(scopeId);
    if (!exactScope) {
        TypedColdSet<NameId> descendants(&gArena, &dirty);
        for (int32_t id = 1; id <= mb.negationLedger.count(); ++id) {
            const NameId scope = static_cast<NameId>(uint64_t(mb.negationLedger.decodeKey(id)) >> 32);
            if (mb.nameMap.isStrictAncestor(scopeId, scope)) descendants.mint(scope);
        }
        for (int32_t i = 1; i <= descendants.count(); ++i) selectScope(descendants.keyAt(i));
    }
    struct SelectedRow { int32_t keyId; NameId scope; NameId original; };
    PagedVector<SelectedRow> rows(&gArena, &dirty);
    for (int32_t i = 1; i <= selected.count(); ++i) {
        const int64_t key = selected.keyAt(i);
        assert(watchUses.contains(key));
        const auto row = Codec<StatementKey>::decode(key);
        const StrSpan text = mb.nameMap.decodeView(row.orig);
        assert(!text.empty() && text[0] == '!' && !canonicalizer.isNegatedEquality(text));
        const NameId core = mb.nameMap.lookup(extractExpressionUniversalSpan(text));
        assert(core != 0);
        const int32_t run = mb.negationLedger.lookup(watchKey(row.validity, core));
        assert(run != 0 && mb.negationLedger.runLen(run) != 0);
        rows.push_back(SelectedRow{run, row.validity, row.orig});
    }
    // Reproduce the prior ledger-key order and decoded lexical row order,
    // including the order in which canonical text identifiers are minted.
    const auto less = [&](int32_t a, int32_t b) {
        const auto left = rows[a], right = rows[b];
        return left.keyId != right.keyId ? left.keyId < right.keyId
            : compareSpans(mb.nameMap.decodeView(left.original), mb.nameMap.decodeView(right.original)) < 0;
    };
    ChunkSortedOrdinals<decltype(less)> order(gArena, rows.size(), 0, less);
    struct RunRef { int32_t keyId; NameId scope; };
    struct Change { int32_t runIdx; NameId orig; NameId canon; };
    PagedVector<RunRef> runs(&gArena, &dirty);
    PagedVector<Change> changes(&gArena, &dirty);
    for (int32_t i = 0; i < rows.size(); ++i) {
        const auto row = rows[order.next()];
        if (runs.empty() || runs[runs.size() - 1].keyId != row.keyId)
            runs.push_back(RunRef{row.keyId, row.scope});
        changes.push_back(Change{runs.size() - 1, row.original,
            canonicalWatchIdAt(mb, row.original, row.scope)});
    }
    int32_t changedTotal = 0;
    if (changes.size() > 0) {
        // (3) One rebuild per touched run: the run's values are replaced
        //     through a sorted (orig -> canon) lookup of its change group.
        const int32_t cN = changes.size();
        Change* cs = reinterpret_cast<Change*>(gArena.resolve(gArena.alloc(
            cN * static_cast<int32_t>(sizeof(Change)), static_cast<int32_t>(alignof(Change)))));
        for (int32_t i = 0; i < cN; ++i) cs[i] = changes[i];
        std::sort(cs, cs + cN, [](const Change& a, const Change& b) {
            return a.runIdx != b.runIdx ? a.runIdx < b.runIdx : a.orig < b.orig; });
        for (int32_t g = 0; g < cN; ) {
            int32_t h = g;
            while (h < cN && cs[h].runIdx == cs[g].runIdx) ++h;
            const RunRef& ref = runs[cs[g].runIdx];
            const int32_t n = mb.negationLedger.runLen(ref.keyId);
            const ArenaOffset runMark = gArena.cursor();
            int64_t* next = reinterpret_cast<int64_t*>(gArena.resolve(gArena.alloc(
                n * static_cast<int32_t>(sizeof(int64_t)), static_cast<int32_t>(alignof(int64_t)))));
            bool changed = false;
            for (int32_t j = 0; j < n; ++j) {
                const int64_t v = mb.negationLedger.valueAt(ref.keyId, j);
                const NameId orig = gl::CaseLedger::ledgerOriginalId(v);
                // binary search the group [g, h) by orig
                int32_t lo = g, hi = h;
                while (lo < hi) {
                    const int32_t mid = lo + (hi - lo) / 2;
                    if (cs[mid].orig < orig) lo = mid + 1; else hi = mid;
                }
                if (lo < h && cs[lo].orig == orig && cs[lo].canon != gl::CaseLedger::ledgerCanonicalId(v)) {
                    next[j] = packInt32Pair(cs[lo].canon, orig);
                    const int64_t dependencyKey = packStatementKey(orig,ref.scope);
                    mb.lbMemory.canonicalUses.watches.erase(dependencyKey,gArena);
                    canonicalTemplates.indexCanonicalText(mb.lbMemory.canonicalUses.watches,dependencyKey,ref.scope,mb.nameMap.decodeView(orig));
                    canonicalTemplates.indexCanonicalText(mb.lbMemory.canonicalUses.watches,dependencyKey,ref.scope,mb.nameMap.decodeView(cs[lo].canon));
                    changed = true; ++changedTotal;
                    mb.pendingWatchRechecks.push_back(packInt32Pair(ref.scope, orig));
                } else {
                    next[j] = v;
                }
            }
            if (changed) {
                std::sort(next, next + n);
                const int32_t unique = static_cast<int32_t>(std::unique(next, next + n) - next);
                mb.negationLedger.assignSet(mb.negationLedger.decodeKey(ref.keyId), next, unique);
            }
            gArena.popTo(runMark);
            g = h;
        }
    }
    gArena.popTo(mark);
    return changedTotal;
}
}
