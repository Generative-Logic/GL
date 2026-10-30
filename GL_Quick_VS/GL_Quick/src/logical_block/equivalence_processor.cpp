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
#include "equivalence_processor.hpp"
namespace gl {
/// @brief Bind inputs for equivalence processor.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param canonicalTemplates Borrowed CanonicalTemplateMatcher interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param rejectedSources Borrowed RejectedSourceMemory interface or state; must outlive this component.
/// @param caseLedger Borrowed CaseLedger interface or state; must outlive this component.
/// @param caseProcessor Borrowed CaseProcessor interface or state; must outlive this component.
/// @param compactRuleProcessor Borrowed CompactRuleProcessor interface or state; must outlive this component.
/// @param equalityPropagation Borrowed EqualityPropagation interface or state; must outlive this component.
/// @param equivalenceAdmissionRewriter Borrowed EquivalenceAdmissionRewriter interface or state; must outlive this component.
/// @param rejectedSourceReplay Borrowed RejectedSourceReplay interface or state; must outlive this component.
/// @param watchProcessor Borrowed WatchProcessor interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
EquivalenceProcessor::EquivalenceProcessor(CanonicalTemplateMatcher& canonicalTemplates, Canonicalizer& canonicalizer, const ProverParameters& parameters, RejectedSourceMemory& rejectedSources, CaseLedger& caseLedger, CaseProcessor& caseProcessor, CompactRuleProcessor& compactRuleProcessor, EqualityPropagation& equalityPropagation, EquivalenceAdmissionRewriter& equivalenceAdmissionRewriter, RejectedSourceReplay& rejectedSourceReplay, WatchProcessor& watchProcessor) : canonicalTemplates(canonicalTemplates), canonicalizer(canonicalizer), parameters(parameters), rejectedSources(rejectedSources), caseLedger(caseLedger), caseProcessor(caseProcessor), compactRuleProcessor(compactRuleProcessor), equalityPropagation(equalityPropagation), equivalenceAdmissionRewriter(equivalenceAdmissionRewriter), rejectedSourceReplay(rejectedSourceReplay), watchProcessor(watchProcessor) {}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)

/// @brief Install an equality `(=[a,b])` into an LB.
///
/// @details
/// - Asserts `args[0] != args[1]` per I-9. A firing assert here means an
///   upstream produced a trivial reflexive equality, which the
///   conjecturer / prover should never emit (per I-8 trivial equality
///   is forbidden in heads).
/// - Searches `memoryBlock.equivalenceClassesMap` for an existing class
///   containing either `args[0]` or `args[1]`; if found, merges the
///   other arg into the class via `mergeTwoEquivalenceClasses` per I-33.
/// - Otherwise creates a fresh
///   `EquivalenceClass{ {args[0], args[1]}, ... }` and seeds
///   `equalityLevelsMap` / `equalityOriginMap`.
/// - Emits the canonical mirror `(=[args[1], args[0]])` as a separate
///   statement so the verifier sees both directions; the per-pair
///   `args[0] != args[1]` assert keeps this from looping.
///
/// @param expr         Equality expression text.
/// @param memoryBlock  Owning LB.
/// @param local        True if the deposit is local-only (no propagation
///                     to the overall hash memory).
/// @param levels       Ascending-unique level run — LB levels at which the
///                     equality is admissible (caller-owned stack/arena
///                     memory, never a cold-column pointer).
/// @param levelCount   The level count (`>= 0`; 0 with null = empty).
/// @param ...          Routing parameters.
/// @invariant I-9 — mirror guard.
/// @invariant I-33 — class-merge cross-vN preconditions.
/// @see `addNegatedEquality` — negated counterpart.
void EquivalenceProcessor::addEquality(StrSpan expr,
    Memory& memoryBlock,
    bool local,
    const int* levels, int32_t levelCount,
    const TransientOrigin& origin,
    StrSpan validityName) {
    // Callers arrive through the addStatement door, which pins the level-run
    // invariant: non-empty, and either exactly {-1} or all-values-≥0.
    assert(levelCount > 0
        && ((levelCount == 1 && levels[0] == -1) || levels[0] >= 0)
        && "addEquality: invalid statement level run");

    const NameId phase3Scope = memoryBlock.nameMap.lookup(validityName);
    if (phase3Scope != 0 && memoryBlock.phase3IgnoresScope(phase3Scope))
        return;

    const StatementFlags* eqRow = lookupStatementFlags(
        memoryBlock.intKnownStatements, memoryBlock.nameMap, expr, validityName);
    if (eqRow == nullptr)
    {
        // 1. Register the original equality
        const int64_t pkEq = packStatementKey(
            memoryBlock.nameMap.encode(expr),
            memoryBlock.nameMap.encode(validityName));
        memoryBlock.intStatementLevelsMap.assignSetRange(
            pkEq, levels, levels + levelCount);
        gl::StatementRegistryOperations::publishKnownStatementKey(memoryBlock, pkEq, local);

        if (parameters.trackHistory && origin.present) {
            assert(origin.tag != OriginTag::COUNT);
            addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner, expr, StrSpan(validityName), origin.tag, origin.deps, origin.depN, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            // mailOut.exprOriginMap addOrigin is done in `fillMailOut`.
        }

        { IntEncodedExpr ie = encodeExpression(expr, StrSpan(validityName), memoryBlock.nameMap);
          memoryBlock.intEncodedStatements.push_back(ie);

        if (local) {
            memoryBlock.intLocalEncodedStatementsSet.mint(pkEq);
            memoryBlock.intLocalEncodedStatements.push_back(ie);
            memoryBlock.intLocalEncodedStatementsDelta.push_back(ie);
            // mailOut.statements + mailOut.exprOriginMap are written in
            // `fillMailOut`.
        }
        } // close IntEncodedExpr ie scope

        // 2. Register the MIRRORED equality
        StrSpan argSpans[ExecutionParameters::MAX_ARITY];
        const int32_t argN = getArgsSpans(expr, argSpans,
                                          ExecutionParameters::MAX_ARITY);
        assert(argN == 2);

        if (!equalSpans(argSpans[0], argSpans[1])) {
        // Boundary string: the mirror is registered (encodeExpression /
        // intKnownStatements) and origin-tracked. Built "(=[a1,a0])" on the
        // per-slot string-scratch arena; every consumer (lookupStatementFlags /
        // nameMap.encode / encodeExpression / addOriginEncoded) mints into its
        // own cold pool, not this arena, and argSpans alias the caller-stable
        // expr, so the mirror span stays live for the whole block (I-3).
        const unsigned mSlot = (gl::WorkerContext::coreId() >= 0)
            ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : scratchArenas().slotCount() - 1;
        ScratchArena& mArena = scratchArenas().forSlot(mSlot);
        ScratchScope mScope(mArena);
        const int32_t mirrorLen = 3 + argSpans[1].len + 1 + argSpans[0].len + 2;
        char* mBuf = mArena.allocBytes(mirrorLen);
        int32_t mAt = 0;
        mBuf[mAt++] = '('; mBuf[mAt++] = '='; mBuf[mAt++] = '[';
        std::memcpy(mBuf + mAt, argSpans[1].ptr,
                    static_cast<std::size_t>(argSpans[1].len));
        mAt += argSpans[1].len;
        mBuf[mAt++] = ',';
        std::memcpy(mBuf + mAt, argSpans[0].ptr,
                    static_cast<std::size_t>(argSpans[0].len));
        mAt += argSpans[0].len;
        mBuf[mAt++] = ']'; mBuf[mAt++] = ')';
        assert(mAt == mirrorLen);
        const StrSpan mirrored(mBuf, mirrorLen);
        const StrSpan validitySpan(validityName);

        const StatementFlags* mirrorRow = lookupStatementFlags(
            memoryBlock.intKnownStatements, memoryBlock.nameMap, mirrored, validitySpan);
        assert(mirrorRow == nullptr);
        (void)mirrorRow;

        const int64_t pkEqM = packStatementKey(
            memoryBlock.nameMap.encode(mirrored),
            memoryBlock.nameMap.encode(validitySpan));
        memoryBlock.intStatementLevelsMap.assignSetRange(
            pkEqM, levels, levels + levelCount);
        gl::StatementRegistryOperations::publishKnownStatementKey(memoryBlock, pkEqM, local);

        if (parameters.trackHistory) {
            // symmetry-of-equality mirror origin: one antecedent = the
            // original equality (expr, validityName), spanned in place.
            const OriginDep mirroredDeps[1] = {
                { expr, StrSpan(validityName) } };
            addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner, mirrored, validitySpan, OriginTag::symmetryOfEquality, mirroredDeps, 1, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            // mailOut.exprOriginMap addOrigin for the mirror is done in
            // `fillMailOut`.
        }

        { IntEncodedExpr ieM = encodeExpression(mirrored, validitySpan, memoryBlock.nameMap);
          memoryBlock.intEncodedStatements.push_back(ieM);

        if (local) {
            memoryBlock.intLocalEncodedStatementsSet.mint(pkEqM);
            memoryBlock.intLocalEncodedStatements.push_back(ieM);
            memoryBlock.intLocalEncodedStatementsDelta.push_back(ieM);

            // mailOut.statements for the mirror are written in
            // `fillMailOut`. The mirror's intLocalEncodedStatementsDelta
            // push above ensures fillMailOut picks it up.
        }
        } // close IntEncodedExpr ieM scope
        } // end if args[0] != args[1]
    }
}

//#pragma optimize("", off)

/// @brief Install a negated equality `!(=[a,b])` into an LB.
///
/// @details
/// Per I-12, `!(=[a,b])` expands **one-sidedly** via equivalence classes
/// — never as a symmetric cross-product. The expansion picks the
/// deeper-scoped variable as the substitution target so the resulting
/// negated expression sits at the union scope of the original and the
/// class's scope.
///
/// Concrete: if `b` is in a class `C = {b, c}` at scope `S`, then
/// `!(=[a, b])` produces `!(=[a, c])` at the deeper of (current scope,
/// S). It does NOT also produce `!(=[c, a])` — that would be the
/// symmetric counterpart, which the one-sided rule forbids.
///
/// @param expr         Negated equality expression text.
/// @param memoryBlock  Owning LB.
/// @param local        True if the deposit is local-only (no propagation
///                     to the overall hash memory).
/// @param levels       Ascending-unique level run — LB levels at which the
///                     negated equality is admissible (caller-owned).
/// @param levelCount   The level count (`>= 0`; 0 with null = empty).
/// @param ...          Routing parameters.
/// @invariant I-12 — one-sided expansion via equivalence classes; no
///            symmetric cross-product.
/// @see `addEquality` — positive counterpart.
/// @see `applyEquivalenceClassToNegatedEquality` — class-driven sibling
///      rewrite.
void EquivalenceProcessor::addNegatedEquality(StrSpan expr,
    Memory& memoryBlock,
    bool local,
    const int* levels, int32_t levelCount,
    const TransientOrigin& origin,
    StrSpan validityName) {
    // Mirror of addEquality for negated form !(=[a,b]).
    // Pair invariant: original and mirror are registered together or not at all.
    // Sole gateway — any external path that registers a negated equality
    // without passing through here breaks the invariant and will trip the
    // mirror-absence assert below.
    // Callers arrive through the addStatement door, which pins the level-run
    // invariant: non-empty, and either exactly {-1} or all-values-≥0.
    assert(levelCount > 0
        && ((levelCount == 1 && levels[0] == -1) || levels[0] >= 0)
        && "addNegatedEquality: invalid statement level run");

    const NameId phase3Scope = memoryBlock.nameMap.lookup(validityName);
    if (phase3Scope != 0 && memoryBlock.phase3IgnoresScope(phase3Scope))
        return;

    const StatementFlags* negRow = lookupStatementFlags(
        memoryBlock.intKnownStatements, memoryBlock.nameMap, expr, validityName);
    if (negRow == nullptr)
    {
        // 1. Register the original negated equality
        const int64_t pkNeg = packStatementKey(
            memoryBlock.nameMap.encode(expr),
            memoryBlock.nameMap.encode(validityName));
        memoryBlock.intStatementLevelsMap.assignSetRange(
            pkNeg, levels, levels + levelCount);
        gl::StatementRegistryOperations::publishKnownStatementKey(memoryBlock, pkNeg, local);

        if (parameters.trackHistory && origin.present) {
            assert(origin.tag != OriginTag::COUNT);
            addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner, expr, StrSpan(validityName), origin.tag, origin.deps, origin.depN, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            // mailOut.exprOriginMap addOrigin is done in `fillMailOut`.
        }

        { IntEncodedExpr ie = encodeExpression(expr, StrSpan(validityName), memoryBlock.nameMap);
          memoryBlock.intEncodedStatements.push_back(ie);

        if (local) {
            memoryBlock.intLocalEncodedStatementsSet.mint(pkNeg);
            memoryBlock.intLocalEncodedStatements.push_back(ie);
            memoryBlock.intLocalEncodedStatementsDelta.push_back(ie);
            // mailOut.statements + mailOut.exprOriginMap are written in
            // `fillMailOut`.
        }
        } // close IntEncodedExpr ie scope

        // 2. Register the MIRRORED negated equality
        StrSpan argSpans[ExecutionParameters::MAX_ARITY];
        const int32_t argN = getArgsSpans(expr, argSpans,
                                          ExecutionParameters::MAX_ARITY);
        assert(argN == 2);

        if (!equalSpans(argSpans[0], argSpans[1])) {
        // Boundary string: the mirror is registered (encodeExpression /
        // intKnownStatements) and origin-tracked. Built "!(=[a1,a0])" on the
        // per-slot string-scratch arena; every consumer (lookupStatementFlags /
        // nameMap.encode / encodeExpression / addOriginEncoded) mints into its
        // own cold pool, not this arena, and argSpans alias the caller-stable
        // expr, so the mirror span stays live for the whole block (I-3).
        const unsigned mSlot = (gl::WorkerContext::coreId() >= 0)
            ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : scratchArenas().slotCount() - 1;
        ScratchArena& mArena = scratchArenas().forSlot(mSlot);
        ScratchScope mScope(mArena);
        const int32_t mirrorLen = 4 + argSpans[1].len + 1 + argSpans[0].len + 2;
        char* mBuf = mArena.allocBytes(mirrorLen);
        int32_t mAt = 0;
        mBuf[mAt++] = '!'; mBuf[mAt++] = '('; mBuf[mAt++] = '='; mBuf[mAt++] = '[';
        std::memcpy(mBuf + mAt, argSpans[1].ptr,
                    static_cast<std::size_t>(argSpans[1].len));
        mAt += argSpans[1].len;
        mBuf[mAt++] = ',';
        std::memcpy(mBuf + mAt, argSpans[0].ptr,
                    static_cast<std::size_t>(argSpans[0].len));
        mAt += argSpans[0].len;
        mBuf[mAt++] = ']'; mBuf[mAt++] = ')';
        assert(mAt == mirrorLen);
        const StrSpan mirrored(mBuf, mirrorLen);
        const StrSpan validitySpan(validityName);

        const StatementFlags* negMirrorRow = lookupStatementFlags(
            memoryBlock.intKnownStatements, memoryBlock.nameMap, mirrored, validitySpan);
        assert(negMirrorRow == nullptr);
        (void)negMirrorRow;

        const int64_t pkNegM = packStatementKey(
            memoryBlock.nameMap.encode(mirrored),
            memoryBlock.nameMap.encode(validitySpan));
        memoryBlock.intStatementLevelsMap.assignSetRange(
            pkNegM, levels, levels + levelCount);
        gl::StatementRegistryOperations::publishKnownStatementKey(memoryBlock, pkNegM, local);

        if (parameters.trackHistory) {
            // symmetry-of-inequality mirror origin: one antecedent = the
            // original inequality (expr, validityName), spanned in place.
            const OriginDep mirroredDeps[1] = {
                { expr, StrSpan(validityName) } };
            addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner, mirrored, validitySpan, OriginTag::symmetryOfInequality, mirroredDeps, 1, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            // mailOut.exprOriginMap addOrigin for the mirror is done in
            // `fillMailOut`.
        }

        { IntEncodedExpr ieM = encodeExpression(mirrored, validitySpan, memoryBlock.nameMap);
          memoryBlock.intEncodedStatements.push_back(ieM);

        if (local) {
            memoryBlock.intLocalEncodedStatementsSet.mint(pkNegM);
            memoryBlock.intLocalEncodedStatements.push_back(ieM);
            memoryBlock.intLocalEncodedStatementsDelta.push_back(ieM);

            // mailOut.statements for the mirror are written in
            // `fillMailOut`. The mirror's intLocalEncodedStatementsDelta
            // push above ensures fillMailOut picks it up.
        }
        } // close IntEncodedExpr ieM scope
        } // end if args[0] != args[1]
    }
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)
void EquivalenceProcessor::updateWeakVariables(Memory& mb, StrSpan validityName, StrSpan eqlty) {
    // 1. Identify variables involved in the equality (arity-bounded spans).
    StrSpan eqVarSpans[ExecutionParameters::MAX_ARITY];
    const int32_t eqVarN = getArgsSpans(eqlty, eqVarSpans,
                                        ExecutionParameters::MAX_ARITY);
    if (eqVarN == 0) return;
    // Non-minting id probes: a variable never interned cannot be a class
    // member (members are interned by rebuildIntMirror).
    NameId eqVarIds[ExecutionParameters::MAX_ARITY];
    for (int32_t i = 0; i < eqVarN; ++i)
        eqVarIds[i] = mb.nameMap.lookup(eqVarSpans[i]);

    // 2-4. Locate the target class at this validity via a zero-copy view walk
    // (no heap EquivalenceClass decode), then record its weak members inline.
    // Steps 3-4 mint only NameMap / intWeakVariables / the name-kind cache,
    // never equivalenceClassesMap, so the peeked blob stays valid across them.
    // The class blob is peeked onto the per-slot gen-scratch byte-bump tier via
    // the ScratchArena& peekBlobContiguous overload (contiguous -> pool pointer,
    // straddle -> fresh arena copy), each iteration reclaimed by its own
    // ScratchScope. This completes the equi-class page-straddle-buffer sweep
    // (the cleanUpExpressions / updateEquivalenceClasses /
    // applyEquivalenceClassToNegatedEquality siblings already retired their
    // std::vector<char> buffers via this overload).
    const NameId weakValId = mb.nameMap.lookup(validityName);
    const int32_t bucketId = mb.equivalenceClassesMap.lookup(weakValId);
    assert(weakValId != 0 && bucketId != 0
        && "updateWeakVariables: expected a bucket at this validity");
    const std::size_t classCount = static_cast<std::size_t>(
        mb.equivalenceClassesMap.runLen(bucketId));
    const unsigned wvSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& wvArena = genScratchArenas().forSlot(wvSlot);

    bool found = false;
    for (std::size_t ci = 0; ci < classCount && !found; ++ci) {
        ScratchScope clsScope(wvArena);
        int32_t clsLen = 0;
        const char* clsPtr = mb.equivalenceClassesMap.inner().peekBlobContiguous(
            bucketId, static_cast<int32_t>(ci), clsLen, wvArena);
        const EquivalenceClassView ec{ clsPtr, clsLen };

        // First class containing any equality variable is the target — same
        // selection as the former decoded-vector scan (blob-run order).
        bool isTarget = false;
        for (int32_t i = 0; i < eqVarN; ++i) {
            if (eqVarIds[i] != 0 && classHasMember(ec, eqVarIds[i])) {
                isTarget = true;
                break;
            }
        }
        if (!isTarget) continue;
        found = true;

        // 3. Preferred special member: forced, integration by numeric depth,
        // then algebraic, with decoded lexical ties.
        const NameId canonicalId = canonicalizer.firstSpecialMemberId(ec, mb);
        if (canonicalId == 0) return;  // no special-tier members -> nothing weak

        // 4. Weak = special witness AND not the preferred special member. Dual-write via the
        // packed twin (D-134). Write site, minting allowed.
        const NameId weakValidityId = mb.nameMap.encode(validityName);
        for (int32_t k = 0; k < ec.memberCount(); ++k) {
            const NameId id = ec.memberId(k);
            if (id == canonicalId) continue;
            const NameKind kind = mb.eqClassNameCaches.kindOf(id, mb.nameMap);
            if (kind == NameKind::Normal) continue;
            mb.intWeakVariables.mint(packStatementKey(id, weakValidityId));
        }
    }
    if (!found) assert(false);
}
}
