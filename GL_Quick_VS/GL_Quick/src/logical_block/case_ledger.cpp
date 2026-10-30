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
#include "../compiler/compact_expression_syntax.hpp"
#include "case_ledger.hpp"
namespace gl {
/// @brief Bind inputs for case ledger.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param canonicalTemplates Borrowed CanonicalTemplateMatcher interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
CaseLedger::CaseLedger(CanonicalTemplateMatcher& canonicalTemplates, Canonicalizer& canonicalizer, const CompiledDefinitionRegistry& definitions) : canonicalTemplates(canonicalTemplates), canonicalizer(canonicalizer), definitions(definitions) {}

/// @brief Orient the validated digit pair for OR processing identity.
///
/// @details Swaps argument positions, never all occurrences of their values.
/// Other slots, including a repeated value in another slot, remain unchanged.
/// The returned text uses caller-owned string scratch only when changed.
/// @param expression Positive numbered OR instance.
/// @param strings Caller-owned string arena.
/// @return Input text or its lexicographically smaller validated orientation.
/// @invariant No proof expression, definition or interner is changed.
/// @see compileOrSymmetry; canonicalOrIdAtScope.
StrSpan CaseLedger::canonicalOrProcessingForm(StrSpan expression,
    ScratchArena& strings) const {
    assert(gl::CompactExpressionSyntax::isOrCompactSpan(expression));
    const auto* entity = this->definitions.compiledEntity(extractExpressionUniversalSpan(expression));
    if (entity == nullptr || entity->symmetricDigitSlots[0] == 0) return expression;
    StrSpan arguments[ExecutionParameters::MAX_ARITY];
    const int32_t count = getArgsSpans(expression, arguments, ExecutionParameters::MAX_ARITY);
    const int32_t a = entity->symmetricDigitSlots[0] - 1;
    const int32_t b = entity->symmetricDigitSlots[1] - 1;
    assert(a >= 0 && a < b && b < count);
    if (compareSpans(arguments[a], arguments[b]) <= 0) return expression;
    char* result = strings.allocBytes(expression.len);
    int32_t at = 0;
    const auto append = [&](const char* bytes, int32_t length) {
        std::memcpy(result + at, bytes, length);
        at += length;
    };
    append(expression.ptr, static_cast<int32_t>(arguments[a].ptr - expression.ptr));
    append(arguments[b].ptr, arguments[b].len);
    const char* middle = arguments[a].ptr + arguments[a].len;
    append(middle, static_cast<int32_t>(arguments[b].ptr - middle));
    append(arguments[a].ptr, arguments[a].len);
    const char* suffix = arguments[b].ptr + arguments[b].len;
    append(suffix, static_cast<int32_t>(expression.ptr + expression.len - suffix));
    assert(at == expression.len);
    return StrSpan(result, at);
}

/// @brief Compute an OR processing id under the scope's equality classes.
///
/// @details Applies the existing equality chooser, then the compiled symmetric
/// slot orientation. Only the resulting processing identity is interned.
/// @param mb Owning block.
/// @param origId Minted original OR id.
/// @param validityName Scope whose existing equality bucket applies.
/// @return Minted processing id; original proof text is retained separately.
/// @invariant No proof history or known statement is rewritten.
/// @see canonicalOrProcessingForm; recordProcessedOr.
NameId CaseLedger::canonicalOrIdAtScope(Memory& mb, NameId origId,
    StrSpan validityName) {
    assert(origId != 0
        && "canonicalOrIdAtScope: the original id must be minted");
    const StrSpan text = mb.nameMap.decodeView(origId);
    assert(gl::CompactExpressionSyntax::isOrCompactSpan(text)
        && "canonicalOrIdAtScope: ledger rows hold or compacts only");
    const unsigned slot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : scratchArenas().slotCount() - 1;
    ScratchArena& strArena = scratchArenas().forSlot(slot);
    ScratchScope canonScope(strArena);
    int lvBuf[256];
    int32_t lvN = 0;
    const CanonicalForm cf = canonicalizer.canonicalFormAtScope(
        text, validityName, mb, strArena, lvBuf, lvN, 256);
    const StrSpan key = canonicalOrProcessingForm(cf.changed ? cf.text : text, strArena);
    return mb.nameMap.encode(key);
}

/// @brief Test whether an equivalent OR already received processing.
        ///
        /// @details The input has the deposit door's equality-canonical spelling.
        /// Its compiled digit-pair orientation is looked up without minting and
        /// compared with each processing id in the exact scope's ledger. Original
        /// ids also match verbatim retries. Other scopes remain independent.
        /// @param mb Owning resident block.
        /// @param expr Equality-canonical positive numbered OR instance.
        /// @param validityName Exact scope being tested.
        /// @return Whether a recorded representative supplies OR processing.
        /// @invariant Does not mint or modify the ledger or proof state.
        /// @see canonicalOrProcessingForm; recordProcessedOr.
bool CaseLedger::orEquiRepresentativeRecorded(Memory& mb,
    StrSpan expr,
    StrSpan validityName) {
    assert(gl::CompactExpressionSyntax::isOrCompactSpan(expr)
        && "orEquiRepresentativeRecorded: caller must gate on the or-compact shape");

    const NameId vid = mb.nameMap.lookup(validityName);
    if (vid == 0) return false;   // fresh scope: nothing recorded there yet
    const int32_t row = mb.processedOrLedger.lookup(vid);
    if (row == 0) return false;
    const NameId exprId = mb.nameMap.lookup(expr);
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope stringScope(strings);
    const NameId processingId = mb.nameMap.lookup(canonicalOrProcessingForm(expr, strings));
    if (exprId == 0 && processingId == 0) return false;
    const int32_t n = mb.processedOrLedger.runLen(row);
    for (int32_t j = 0; j < n; ++j) {
        const int64_t v = mb.processedOrLedger.valueAt(row, j);
        if (ledgerCanonicalId(v) == processingId || ledgerOriginalId(v) == exprId)
            return true;
    }
    return false;
}

/// @brief Record an or compact as fully processed at a scope — the
        ///        ledger insert paired with the willDisintegrate branch.
        ///
        /// @details
        /// Inserts `packInt32Pair(canonicalOrIdAtScope(origId), origId)`
        /// into the scope's `processedOrLedger` run (sorted-unique, numeric
        /// order — the run is never observable). Called by real nested OR consumption and the
        /// completed deposit; repeated insertion is idempotent. The
        /// canonical half is computed under the classes present now and
        /// kept current by the class-change seam afterwards.
        ///
        /// @param mb     Owning LB.
        /// @param origId The or statement's NameMap id (door-canonical text).
        /// @param valId  The scope's NameMap id.
        /// @return (void)
        /// @see orEquiRepresentativeRecorded; clearProcessedOr;
        ///      recanonicalizeProcessedOrLedger.
void CaseLedger::recordProcessedOr(Memory& mb, NameId origId,
    NameId valId) {
    assert(origId != 0 && valId != 0
        && "recordProcessedOr: both ids must be minted by the door path");
    // The scope-name view is read by the canonicalization only, before the
    // possible mint inside — no span outlives the mint (I-3).
    const NameId canonId =
        canonicalOrIdAtScope(mb, origId, mb.nameMap.decodeView(valId));
    mb.processedOrLedger.insertSorted(valId, packInt32Pair(canonId, origId),
        [](int64_t a, int64_t b) { return a < b; });
    const unsigned slot = gl::WorkerContext::coreId() >= 0 ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    indexCanonicalOrMarker(mb, origId, canonId, valId, genScratchArenas().forSlot(slot));

}

/// @brief Drop one or statement's row from the processed-or ledger
        ///        — the clear paired with every path that intentionally
        ///        re-runs an or consumption.
        ///
        /// @details
        /// Rebuilds the scope's run without the row whose ORIGINAL half is
        /// @p origId (`assignSet`; the run is small — the distinct
        /// fully-processed ors of one scope). Idempotent by contract:
        /// several park keys of one or each wake once, and each wake must
        /// leave the ledger cleared — an absent row is the defined
        /// already-cleared state, never a failure. Callers:
        /// `resetParkedOrStatementRegistries` (the revival un-know), the
        /// two ordis park hooks' re-mail loops (the mailed rewritten
        /// statement must re-enter the door as the representative, or the
        /// gate would swallow the canonical re-file), and the D-331 drain's
        /// loser retirement.
        ///
        /// @param mb     Owning LB.
        /// @param origId The or statement's original NameMap id.
        /// @param valId  The scope's NameMap id.
        /// @return (void)
        /// @see recordProcessedOr; orEquiRepresentativeRecorded.
void CaseLedger::clearProcessedOr(Memory& mb, NameId origId,
    NameId valId) {
    assert(origId != 0 && valId != 0
        && "clearProcessedOr: both ids must be minted");
    const int32_t row = mb.processedOrLedger.lookup(valId);
    if (row == 0) return;   // defined already-cleared state
    const int32_t n = mb.processedOrLedger.runLen(row);

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
        const int64_t v = mb.processedOrLedger.valueAt(row, j);
        if (ledgerOriginalId(v) != origId) keep[keepN++] = v;
    }
    if (keepN != n) {
        mb.processedOrLedger.assignSet(valId, keep, keepN);
    }
    gArena.popTo(mark);
    mb.lbMemory.canonicalUses.processedOr.erase(packStatementKey(origId, valId), gArena);

}

/// @brief Re-key one scope's processed-or ledger rows to the
        ///        scope's current classes — the class-change seam of I-219.
        ///
        /// @details
        /// For every row of @p valId's run, recomputes the canonical half
        /// from the original half (`canonicalOrIdAtScope`) and, when any
        /// half changed, replaces the run with the re-sorted pairs
        /// (`assignSet`; the original halves are untouched, so no two rows
        /// collide — two originals that now share a canonical form stay two
        /// rows, which is exactly what the D-331 drain groups). Called from
        /// `updateEquivalenceClasses` right after the merged class is
        /// stored and pushed to `changedClassesThisStep`, the one place a
        /// scope's class bucket changes; the cost is the run length times
        /// one canonicalization per class change, instead of one
        /// canonicalization per record per or deposit at the gate. A scope
        /// with no rows is the defined no-op. Heap-free: the rebuilt run
        /// rides a gen-scratch mark, the canonical texts a string-tier
        /// scope, both reclaimed on return; the scope name is copied into
        /// the arena before the first mint (I-3).
        ///
        /// @param mb    Owning LB.
        /// @param valId The scope whose class bucket just changed.
        /// @return (void)
        /// @invariant After the call every row's canonical half equals
        ///            `canonicalOrIdAtScope(originalHalf)` under the
        ///            scope's classes.
        /// @see recordProcessedOr; orEquiRepresentativeRecorded;
        ///      updateEquivalenceClasses.
void CaseLedger::recanonicalizeProcessedOrLedger(Memory& mb,
    NameId valId, const PagedVector<NameId>* members) {
    RT_SCOPE_HERE("CANONICAL_PROCESSED_OR_LEDGER");
    assert(valId != 0
        && "recanonicalizeProcessedOrLedger: the scope id must be minted");
    const int32_t row = mb.processedOrLedger.lookup(valId);
    if (row == 0) return;   // no or processed at this scope: defined no-op
    const int32_t n = mb.processedOrLedger.runLen(row);
    if (n == 0) return;     // every row cleared: defined no-op

    const unsigned slot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(slot);
    const ArenaOffset mark = gArena.cursor();
    DirtyState selectionDirty = DirtyState::Clean;
    TypedColdSet<NameId> selected(&gArena, &selectionDirty);
    if (members != nullptr) {
        ensureCanonicalOrMarkerUses(mb, gArena);
        for (int32_t m = 0; m < members->size(); ++m)
            mb.lbMemory.canonicalUses.processedOr.forEach(valId, mb.nameMap.decodeView((*members)[m]),
                [&](int64_t key) { selected.mint(Codec<StatementKey>::decode(key).orig); });
        if (selected.empty()) return;
    }

    int64_t* next = reinterpret_cast<int64_t*>(gArena.resolve(gArena.alloc(
        n * static_cast<int32_t>(sizeof(int64_t)),
        static_cast<int32_t>(alignof(int64_t)))));
    const unsigned sSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : scratchArenas().slotCount() - 1;
    ScratchArena& strArena = scratchArenas().forSlot(sSlot);
    ScratchScope nameScope(strArena);
    // Arena-held copy of the scope name: the per-row mint below may
    // invalidate a bare NameMap view (I-3).
    const StrSpan vView = mb.nameMap.decodeView(valId);
    const ScratchString vHold =
        ScratchString::copyFrom(strArena, vView.ptr, vView.len);
    const StrSpan vName(vHold);
    bool changed = false;
    for (int32_t j = 0; j < n; ++j) {
        const int64_t v = mb.processedOrLedger.valueAt(row, j);
        const NameId origId = ledgerOriginalId(v);
        if (members != nullptr && selected.lookup(origId) == 0) { next[j] = v; continue; }
        const NameId canonId = canonicalOrIdAtScope(mb, origId, vName);
        next[j] = packInt32Pair(canonId, origId);
        if (canonId != ledgerCanonicalId(v)) {
            changed = true;
            indexCanonicalOrMarker(mb, origId, canonId, valId, gArena);
        }
    }
    if (changed) {
        std::sort(next, next + n);
        mb.processedOrLedger.assignSet(valId, next, n);
    }
    gArena.popTo(mark);
}
}
