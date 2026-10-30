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
#include "equivalence_queries.hpp"
namespace gl {
/// @brief Bind inputs for equivalence queries.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param canonicalTemplates Borrowed CanonicalTemplateMatcher interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
EquivalenceQueries::EquivalenceQueries(CanonicalTemplateMatcher& canonicalTemplates, Canonicalizer& canonicalizer) : canonicalTemplates(canonicalTemplates), canonicalizer(canonicalizer) {}

/// @brief Check for a fully disintegrated equivalent expression at one scope.
        /// @details Forwards to the span implementation, whose identity uses the
        /// complete target-scope classes. Ancestor and sibling receipt ownership
        /// does not satisfy this scope's disintegration gate.
        /// @param expr Candidate expression, owned by the caller.
        /// @param validityName Exact disintegration scope.
        /// @param memoryBlock Owning block and complete class views.
        /// @return Whether an equivalent known row is fully disintegrated here.
        /// @see checkForEquivalence(StrSpan, StrSpan, Memory&).
bool EquivalenceQueries::checkForEquivalence(const std::string& expr,
    const std::string& validityName,
    Memory& memoryBlock) {
    return checkForEquivalence(StrSpan(expr), StrSpan(validityName), memoryBlock);
}

/// @brief Test the canonical identity of fully disintegrated known rows.
        /// @details An already completed literal row is a direct hit. Otherwise
        /// the shared posting index compares both sides at the exact target scope
        /// and reads the live fullyDisintegrated bit from each matching original
        /// record. Classes come from the complete view; no Cartesian expression
        /// family is materialized. Existing application multiplication is separate.
        /// @param expr Candidate expression on caller-owned storage.
        /// @param validityName Registered scope of this disintegration attempt.
        /// @param memoryBlock Owning block, classes and known-statement store.
        /// @return Whether a matching row is fully disintegrated at this scope.
        /// @invariant Uses arena-backed derived indexes; creates no proof row.
        /// @see matchingCanonicalTemplates, CanonicalTemplateIndex.
bool EquivalenceQueries::checkForEquivalence(StrSpan expr,
    StrSpan validityName,
    Memory& memoryBlock) {
    FtaBlowupAudit* audit=ftaBlowupAudit;
    const auto auditStarted=audit ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const NameId scope=memoryBlock.nameMap.lookup(validityName);
    assert(scope!=0 && "checkForEquivalence: expected a registered scope");
    const unsigned slot=gl::WorkerContext::coreId()>=0 ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount()-1;
    ScratchArena& arena=genScratchArenas().forSlot(slot);
    ScratchScope frame(arena);
    DirtyState dirty=DirtyState::Clean;
    canonicalizer.initializeCanonicalScope(memoryBlock,scope);
    const StatementFlags* direct=lookupStatementFlags(memoryBlock.intKnownStatements,memoryBlock.nameMap,expr,validityName);
    const bool directMatch=direct!=nullptr && direct->fullyDisintegrated;
    uint64_t checkedVariants=0;
    uint64_t snapshotMembers=0;
    int32_t argN=0;
    int32_t possLen[ExecutionParameters::MAX_ARITY];
    if(audit) {
        StrSpan args[ExecutionParameters::MAX_ARITY];
        argN=getArgsSpans(expr,args,ExecutionParameters::MAX_ARITY);
        for(int32_t i=0;i<argN;++i) possLen[i]=1;
        const int32_t bucket=memoryBlock.canonicalEquivalenceClassesMap.lookup(scope);
        if(bucket!=0) for(int32_t i=0;i<memoryBlock.canonicalEquivalenceClassesMap.runLen(bucket);++i) {
            ScratchScope classFrame(arena);
            int32_t length=0;
            const char* bytes=memoryBlock.canonicalEquivalenceClassesMap.peekRecordBytes(bucket,i,length,arena);
            const EquivalenceClassView cls{bytes,length};
            snapshotMembers+=cls.memberCount();
            for(int32_t j=0;j<argN;++j)
                if(classHasMember(cls,memoryBlock.nameMap.lookup(args[j]))) possLen[j]=cls.memberCount();
        }
    }
    const auto auditPrepared = audit ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    const auto recordAudit = [&](bool matched) {
        if (!audit) return;
        const uint64_t elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - auditStarted).count();
        const uint64_t setup = std::chrono::duration_cast<std::chrono::nanoseconds>(
            auditPrepared - auditStarted).count();
        ++audit->calls; audit->matches += matched; audit->directMatches += directMatch;
        audit->variants += checkedVariants;
        if (directMatch) audit->directVariants += checkedVariants;
        audit->setupNs += setup; audit->probesNs += elapsed - setup;
        audit->snapshotMembers += snapshotMembers;
        if (checkedVariants >= 10000 || elapsed >= 5000000) {
            char line[32768];
            int used = std::snprintf(line, sizeof(line),
                "[FTA-CFE] burst=%d ns=%llu setup_ns=%llu variants=%llu matched=%d direct=%d status=%d allowed=%d lengths=",
                audit->burst, elapsed, setup, checkedVariants, matched, directMatch, audit->status, audit->disintegrationAllowed);
            assert(used > 0 && used < sizeof(line));
            for (int32_t i = 0; i < argN; ++i) {
                const int count = std::snprintf(line + used, sizeof(line) - used,
                    "%s%d", i ? "," : "", possLen[i]);
                assert(count > 0 && count < sizeof(line) - used); used += count;
            }
            const int count = std::snprintf(line + used, sizeof(line) - used,
                " expr=%.*s scope=%.*s\n", expr.len, expr.ptr, validityName.len, validityName.ptr);
            assert(count > 0 && count < sizeof(line) - used); used += count;
            FILE* out = nullptr;
#ifdef _WIN32
        const errno_t openResult = fopen_s(&out, ".debug/traps.log", "a");
        assert(openResult == 0);
#else
        out = std::fopen(".debug/traps.log", "a");
#endif
        assert(out);
            assert(std::fwrite(line, 1, used, out) == used); std::fclose(out);
        }
    };
    if(directMatch) { checkedVariants=1; recordAudit(true); return true; }
    const int32_t bucket=memoryBlock.canonicalEquivalenceClassesMap.lookup(scope);
    if(bucket==0 || memoryBlock.canonicalEquivalenceClassesMap.runLen(bucket)==0) {
        recordAudit(false); return false;
    }
    PagedVector<int64_t> matches(&arena,&dirty);
    canonicalTemplates.matchingCanonicalTemplates(memoryBlock,memoryBlock.intKnownStatements,
        memoryBlock.lbMemory.canonicalUses.statementMatches,expr,scope,scope,matches,arena,true);
    for(int32_t i=0;i<matches.size();++i) {
        ++checkedVariants;
        const StatementFlags* record=memoryBlock.intKnownStatements.find(Codec<StatementKey>::decode(matches[i]));
        assert(record!=nullptr);
        if(record->fullyDisintegrated) { recordAudit(true); return true; }
    }
    recordAudit(false);
    return false;
}

/// @brief Test whether a compact is a known statement at one scope, up to
///        that scope's complete equivalence classes.
///
/// @details The known-part check of `prepareIntegrationCore2` Case A: an
/// implication compact that is already a known statement needs no subproof.
/// A literal row at the exact scope is a direct hit. Otherwise the shared
/// statement posting index compares the compact with every known statement
/// owned by that exact scope after both are normalized through the scope's
/// complete class view (I-254), so a statement that names another member of a
/// class counts. Unlike `checkForEquivalence`, the disintegration state of the
/// matching row is irrelevant: a known statement is known whether or not it
/// was disintegrated.
///
/// @param compact Candidate compact statement on caller-owned storage.
/// @param validityName Exact scope that owns the matching statement; it must
///        already be registered in the block's name map.
/// @param memoryBlock Owning block, its classes and known-statement store.
/// @return Whether the compact, or a class variant of it at this scope, is a
///         known statement owned by that scope.
/// @invariant Creates no proof row; only arena-backed derived comparison
///            indexes are synchronized.
/// @see checkForEquivalence, matchingCanonicalTemplates.
bool EquivalenceQueries::compactKnownUpToEquality(StrSpan compact,
    StrSpan validityName,
    Memory& memoryBlock) {
    if (lookupStatementFlags(memoryBlock.intKnownStatements, memoryBlock.nameMap,
            compact, validityName) != nullptr)
        return true;
    const NameId scope = memoryBlock.nameMap.lookup(validityName);
    assert(scope != 0 && "compactKnownUpToEquality: expected a registered scope");
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
    ScratchArena& arena = genScratchArenas().forSlot(slot);
    ScratchScope frame(arena);
    DirtyState dirty = DirtyState::Clean;
    PagedVector<int64_t> matches(&arena, &dirty);
    canonicalTemplates.matchingCanonicalTemplates(memoryBlock, memoryBlock.intKnownStatements,
        memoryBlock.lbMemory.canonicalUses.statementMatches, compact, scope, scope,
        matches, arena, true);
    return matches.size() > 0;
}
}
