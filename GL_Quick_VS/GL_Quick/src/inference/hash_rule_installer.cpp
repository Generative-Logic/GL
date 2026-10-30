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
#include "hash_rule_installer.hpp"
#include "../logical_block/demand_built_rules.hpp"
#include <optional>
namespace gl {
/// @brief Bind inputs for hash rule installer.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param admissionMarkers Borrowed AdmissionMarkerBuilder interface or state; must outlive this component.
/// @param allPermutationsAna Borrowed std::map interface or state; must outlive this component.
/// @param hashRuleClassifier Borrowed HashRuleClassifier interface or state; must outlive this component.
/// @param implicationMultiplier Borrowed ImplicationMultiplier interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param unaryMemory Borrowed UnaryRuleMemory interface or state; must outlive this component.
/// @param admissionMemory Borrowed AdmissionMemory interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
HashRuleInstaller::HashRuleInstaller(AdmissionMarkerBuilder& admissionMarkers, const std::map<int, std::vector<std::vector<int> > >& allPermutationsAna, HashRuleClassifier& hashRuleClassifier, ImplicationMultiplier& implicationMultiplier, const ProverParameters& parameters, UnaryRuleMemory& unaryMemory, AdmissionMemory& admissionMemory) : admissionMarkers(admissionMarkers), allPermutationsAna(allPermutationsAna), hashRuleClassifier(hashRuleClassifier), implicationMultiplier(implicationMultiplier), parameters(parameters), unaryMemory(unaryMemory), admissionMemory(admissionMemory) {}

/// @brief Install one or more `LocalMemoryValue` records into a `HashMemory`,
    /// keyed on the canonical normalization of `key`.
    ///
    /// @details
    /// Path 1 of the hash-engine rule installation: head-implication LMV inserts.
    /// Each call corresponds to one head-implication that the prover wants to
    /// register with the hash engine for later firing.
    ///
    /// The body unfolds in three logical phases:
    ///
    /// 1. **multiplyImplication.** The producer implication is expanded via
    ///    `multiplyImplication(originalImplication)` into one or more copies that
    ///    differ in how their bound variables and remaining-args are partitioned.
    ///    Copy 0 reuses the caller-supplied `key`/`value`; copies 1..N synthesize
    ///    their own from `multiplyImplication`'s output. Provenance for the
    ///    multiplied copies is recorded in `mb.exprOriginMap` (and the mail-out
    ///    mirror) under the `"multiplied from"` tag, capped per `max_origin_per_expr`
    ///    (`compressor_max_origins_per_expr` in compressor mode).
    /// 2. **Per-copy normalization + dedupe.** For each copy, build the
    ///    `IntNormalizedKey` from the NameId-encoded key bytes; consult
    ///    `normalizedEncodedKeys` and `normalizedEncodedSubkeys*` for fast
    ///    rejection; if novel, allocate on the gen scratch arena byte-bump tier so the key
    ///    pointer stays stable for the rest of the LB's lifetime.
    /// 3. **LMV emit + admission.** Construct a `LocalMemoryValue` carrying
    ///    the value, levels, justification, originalImplication, key,
    ///    remainingArgs, validityName. Set `productOfDisintegration` per
    ///    [D-32](../../docs/agentic_swdd/40_decisions.md#d-32) when the producing
    ///    implication has at least one `u_`-prefixed premise arg (this gates
    ///    later OR-disintegration on the head-firing path). Push into
    ///    `targetIntMemory.encodedMap[normKey]`. When
    ///    `performAdmissionMapUpdate == true`, also call into the admission
    ///    side via `updateAdmissionMap3` so the `(expression, scope)` pair
    ///    is registered in `targetIntMemory.admissionMap`.
    ///
    /// @param key                     Pre-built normalization key (string form;
    ///                                NameId mirror is computed inside).
    /// @param value                   Head expression text (what the rule produces).
    /// @param remainingArgs           Argument names not consumed by the key
    ///                                that must travel with the head.
    /// @param mb                      Owning `Memory` (used for `exprOriginMap`,
    ///                                `mailOut`, `ruleInterner`).
    /// @param secondTarget            Optional second instance receiving every
    ///                                staged record of the same build (the
    ///                                incubator's `workingMemory` beside
    ///                                `overallHashMemory`); install-only,
    ///                                never the first target.
    /// @param targetIntMemory         The `HashMemory` to install into. Usually
    ///                                `mb.localHashMemory` for fresh installs;
    ///                                deltas go to `mb.localHashMemoryDelta`.
    /// @param levels                  Ascending-unique level run — LB levels at
    ///                                which the rule is admissible.
    /// @param levelCount              The level count (`>= 0`).
    /// @param originalImplication     Full producer implication text, kept for
    ///                                provenance and for `multiplyImplication`.
    /// @param maxAdmissionDepth       Standard depth budget passed to the
    ///                                admission map.
    /// @param maxSecondaryNumber      Standard secondary-iteration budget.
    /// @param admissionFlags          Flag byte of the admission entries the
    ///                                install files (0, 1 or 3):
    ///                                `kAdmissionRecursionFlag` for a
    ///                                recursion-owned producer, with
    ///                                `kAdmissionInductionHypothesisFlag` for
    ///                                the step block's induction hypothesis
    ///                                (D-519).
    /// @param minNumOperatorsKey      Minimum-operator threshold used by some
    ///                                downstream filters; carried through.
    /// @param justification           Proof tag string emitted on hit
    ///                                (e.g. `"hash"`, `"head"`, `"equality1"`).
    /// @param performAdmissionMapUpdate If true, call `updateAdmissionMap3`
    ///                                in lockstep with the LMV install.
    /// @param originalImplicationClean De-renamed form used as the antecedent
    ///                                in `"multiplied from"` origin records.
    /// @param validityName            Scope at which the rule fires. `"main"`
    ///                                or a `_boundary_`-bearing form minted via
    ///                                `NameMap::encodePush` per
    ///                                [I-2](../../docs/agentic_swdd/30_invariants.md#i-2).
    /// @param goalNearMiss            The rule is a root goal's flat
    ///                                integration rule: its admission build
    ///                                adds the near-miss demand variants
    ///                                (D-574);
    ///                                a removal replays the same value.
    /// @param weakHypothesisInstall   The rule is a weak step block's induction
    ///                                hypothesis: its admission build may file
    ///                                the two-marker start of a chain that
    ///                                cannot start eating
    ///                                (`AdmissionMemory::updateAdmissionMap`,
    ///                                D-586).
    ///
    /// @pre  `mb.lbMemory.manager` is resident — the cold `encodedMap` and the
    ///       two `normalizedEncoded*` owner maps install onto it. Keys are owning
    ///       (`NormKey`); the former keyArena-backed `IntNormalizedKey`
    ///       persistence is retired.
    /// @post `targetIntMemory.encodedMap` and the related `normalizedEncoded*`
    ///       sets are extended with the new LMV(s). Origin entries are added
    ///       under the `"multiplied from"` tag for each non-zero copy.
    /// @invariant [I-2](../../docs/agentic_swdd/30_invariants.md#i-2) — non-`"main"`
    ///            validityName must be minted via `NameMap::encodePush`.
    /// @invariant [I-3](../../docs/agentic_swdd/30_invariants.md#i-3) — when this function
    ///            calls `nm.decode()` / reads `idToSub[]`, the result is
    ///            copied before any nested mint (re-entrance through
    ///            `multiplyImplication` may grow the underlying vectors).
    /// @invariant [I-25](../../docs/agentic_swdd/30_invariants.md#i-25) — cross-scope
    ///            deposits keep `validityName` end-to-end via
    ///            `ExpressionWithValidity` pairs.
    /// @see `inference/hash_rule_installer.hpp::checkLocalEncodedMemoryStatic` — consumer of the
    ///      LMVs installed here.
    /// @see `inference/hash_rule_installer.hpp::updateAdmissionMap3` — admission-side counterpart.
    /// @see `multiplyImplication` (`inference/hash_rule_installer.cpp`) — generator of the per-copy
    ///      key/value pairs.
void HashRuleInstaller::addToHashMemory(const StrSpan* keyRun, int32_t keyN,
        StrSpan valueSpan,
        const StrSpan* remRun, int32_t remN,
        Memory& mb,
        HashMemory& targetIntMemory,
        const int* levels, int32_t levelCount,
        StrSpan originalImplicationSpan,
        int maxAdmissionDepth,
        int maxSecondaryNumber,
        uint8_t admissionFlags,
        int minNumOperatorsKey,
        StrSpan justificationSpan,
        bool performAdmissionMapUpdate,
        StrSpan originalImplicationCleanSpan,
        StrSpan validityNameSpan,
        const RuleIndexOp& op,
        HashMemory* secondTarget,
        bool goalNearMiss,
        bool weakHypothesisInstall) {

        // 0% heap: the caller feeds spans over its own stable bytes and this
        // install body reads them directly (key = keyRun, value = valueSpan,
        // remainingArgs = the sorted-unique remRun, original / justification /
        // clean / validity = their spans). Byte-identical: the ordered chain +
        // the sorted-unique remaining-arg run are preserved.

        // A demand-built rule's head is built only for a demand on its
        // witness (D-596): a rule citing
        // that head as a premise would never see it.
        for (int32_t k = 0; k < keyN; ++k)
            assert(!DemandBuiltRuleRegistry::instance().citesHead(keyRun[k])
                && "addToHashMemory: a rule cites a demand-built rule's head as a premise");

        // A rule installed from non-derived material (the {-1} statement
        // level tier, e.g. a mail-carried non-derived implication) has no
        // level constraints — that is the empty-set convention (I-51). Rule
        // level sets therefore stay empty-or-real; the {-1} tier never
        // enters hash memory.
        if (levelCount == 1 && levels[0] == -1) {
            levels = nullptr;
            levelCount = 0;
        }
        assert((levelCount == 0 || levels[0] >= 0)
            && "rule install level run mixes the non-derived tier with real levels");
        // A removal leaves owner-less entries behind for the end-of-apply
        // compaction (eraseOwnerlessEntries) — flag the instance.
        assert((!op.isRemove() || (op.visited != nullptr && op.dropSet != nullptr))
            && "addToHashMemory: a removal without its visited set or drop set");
        if (op.isRemove()) targetIntMemory.ownerlessPending = true;
        if (op.isRemove() && &targetIntMemory == &mb.overallHashMemory)
            unaryMemory.forgetUnaryImplication(mb, originalImplicationSpan, validityNameSpan);

        // Own multiplyImplication loop — fully independent from addToHashMemory.
        // The span form emits each copy onto mulStrArena (string tier) as a
        // CopyRef index on mulGenArena (page tier, I-124); each copy is read as a
        // zero-copy copySpan over mulStrArena (0% heap). mulStrScope holds the
        // copies live across the whole loop.
        // Per-registry slot derivation: both registries carry logicalCores
        // worker slots plus one DEDICATED reserved single-threaded slot (index
        // logicalCores), and each fallback still reads its OWN slotCount().
        // Workers (gl::WorkerContext::coreId() >= 0) use their own coreId on both —
        // disjoint arenas, in range for both registries.
        const unsigned mulStrSlot = (gl::WorkerContext::coreId() >= 0)
            ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : scratchArenas().slotCount() - 1;
        const unsigned mulGenSlot = (gl::WorkerContext::coreId() >= 0)
            ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : genScratchArenas().slotCount() - 1;
        ScratchArena& mulStrArena = scratchArenas().forSlot(mulStrSlot);
        ScratchScope mulStrScope(mulStrArena);
        ScratchArena& mulGenArena = genScratchArenas().forSlot(mulGenSlot);
        DirtyState mulDirty = DirtyState::Clean;
        PagedVector<CopyRef> copies(&mulGenArena, &mulDirty);
        // The rule-index staging serving this instance on this slot (an
        // install only; a removal uses the direct doors): the install stages
        // every index write into it; the window's closer flushes it. The
        // install is a staging window of its own: a nested deposit into the
        // same LB (a copy equality, an integration seed) shares the bindings
        // and must not close them from under this install — the outermost
        // window closes all four instances.
        RuleIndexStaging* stagingPtr = op.isRemove()
            ? nullptr : &ruleStagings().acquire(mulGenSlot, &targetIntMemory);
        RuleIndexStaging& staging = stagingPtr != nullptr
            ? *stagingPtr : ruleStagings().at(mulGenSlot, 0);   // never read on the remove path
        // A second target (the incubator's workingMemory beside
        // overallHashMemory) receives every staged record of the same build:
        // the copies, permutations and keys are computed once and written to
        // both stagings (D-333).
        assert((secondTarget == nullptr || (!op.isRemove() && secondTarget != &targetIntMemory))
            && "addToHashMemory: a second target is an install-only twin of a different instance");
        RuleIndexStaging* const staging2Ptr = (secondTarget != nullptr)
            ? &ruleStagings().acquire(mulGenSlot, secondTarget) : nullptr;
        ++mb.ruleStagingArmDepth;
        {
            RT_SCOPE_HERE("HM_MULTIPLY");
            implicationMultiplier.multiplyImplication(originalImplicationSpan, mulStrArena, copies);
            RT_NOTE_ITERATIONS_HERE(copies.size());   // copies produced
        }

        for (int32_t c = 0; c < copies.size(); ++c) {
            RT_SCOPE_HERE("HM_COPY");
            const CopyRef cr = copies[c];
            const StrSpan copySpan(
                reinterpret_cast<const char*>(mulStrArena.resolve(cr.off)), cr.len);
            // curKey is a StrSpan run; curOrigImpl == copySpan (mulStrArena,
            // stable for this copy iteration).
            StrSpan curKeyRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
            int32_t curKeyN = 0;
            StrSpan curValueSpan;

            if (c == 0) {
                for (int32_t i = 0; i < keyN; ++i) {
                    assert(curKeyN < ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                        && "addToHashMemory: curKey element count exceeds cap");
                    curKeyRun[curKeyN++] = keyRun[i];
                }
                curValueSpan = valueSpan;
            } else {
                // Track history for multiplied copies (L3 span-record door; drops
                // the EWV / OriginLine build).
                if (parameters.trackHistory) {
                    RT_SCOPE_HERE("HM_COPY_HISTORY");
                    const OriginDep mulDeps[1] = {
                        { originalImplicationCleanSpan, validityNameSpan } };
                    const int mulCap = (parameters.compressor_mode
                        ? parameters.compressor_max_origins_per_expr
                        : parameters.max_origin_per_expr);
                    addOriginEncoded(mb.exprOriginMap, mb.originInterner, copySpan,
                        validityNameSpan, OriginTag::multipliedFrom, mulDeps, 1, mulCap);
                    // Direct mailOut.exprOriginMap write — DOCUMENTED EXCEPTION
                    // to the fillMailOut-sole-writer policy. The multiplied
                    // implication installs as a *rule* in overallHashMemory
                    // (and never lands in intEncodedStatements /
                    // intLocalEncodedStatementsDelta), so `fillMailOut` cannot
                    // propagate the "multiplied from" history line via the
                    // delta-driven copy. See I-64 Caveats.
                    mb.addMailOutOrigin(copySpan, validityNameSpan,
                        OriginTag::multipliedFrom, mulDeps, 1, mulCap);
                }

                // Row 238: span twin — curKey from each triple's key (get<0>),
                // curValue = head. Spans slice copySpan (mulStrArena, stable).
                RT_SCOPE_HERE("HM_COPY_DISINT");
                ce::disintegrateImplicationSpans(copySpan, curValueSpan,
                    [&curKeyRun, &curKeyN](StrSpan keySpan, const StrSpan*, int32_t) {
                        assert(curKeyN < ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                            && "addToHashMemory: curKey element count exceeds cap");
                        curKeyRun[curKeyN++] = keySpan;
                    });
            }

            // One encode per premise per copy: every whole-key permutation,
            // every subkey prefix and every admission derivative below reads
            // this memo. The lazy fill keeps the NameMap first-touch order
            // (a premise is first read where it used to be first encoded).
            KeyEncodeMemo encMemo;
            encMemo.bind(curKeyRun, curKeyN);

            // The rule's owner identity in this HashMemory instance: the copy
            // text's ruleInterner id + the install scope's NameMap id, recorded
            // on every index entry the install below mints (whole keys, subkeys,
            // remaining-args edges, the originals chain) so a later removal of
            // this rule deletes exactly its owner from each. Minted here, before
            // the chain ids, so the owner exists for the originals insert.
            int32_t copyImplId = 0;
            NameId ownerVid = 0;
            RuleOwner owner = 0;
            RuleOwner ruleOwner = 0;
            {
                RT_SCOPE_HERE("HM_COPY_OWNER_MINT");
                copyImplId = mb.ruleInterner.encode(copySpan);
                ownerVid = mb.nameMap.encode(validityNameSpan);
                owner = packRuleOwner(copyImplId, ownerVid);
                ruleOwner = packRuleOwner(
                    mb.ruleInterner.encode(originalImplicationSpan), ownerVid);
            }
            // The copy's owning rules (copyOwners, keyed by the packed copy
            // owner): two rules can multiply into one copy text, so the
            // install lists this rule on the copy and a removal delists it —
            // the copy's index entries and LMVs are touched only when no
            // other rule owns the copy any more. A removal then stages the
            // copy's LMVs for the batched compaction: the pair every LMV
            // record of this copy carries (originalImplicationId = the copy's
            // id). The drop set follows the multiplication exactly — several
            // copies, several pairs; no copy (the trivial-head skip), nothing
            // staged, nothing installed.
            char copyKey[sizeof(RuleOwner)];
            std::memcpy(copyKey, &owner, sizeof(RuleOwner));
            {
            RT_SCOPE_HERE("HM_COPY_OWNERS");
            if (!op.isRemove()) {
                gl::HashRuleIndex::stageOwnerToRun(staging.copyOwners,
                                StrSpan(copyKey, static_cast<int32_t>(sizeof(RuleOwner))),
                                ruleOwner);
                if (staging2Ptr != nullptr) {
                    gl::HashRuleIndex::stageOwnerToRun(staging2Ptr->copyOwners,
                                    StrSpan(copyKey, static_cast<int32_t>(sizeof(RuleOwner))),
                                    ruleOwner);
                }
            } else {
                if (gl::HashRuleIndex::removeOwnerFromRun(targetIntMemory.copyOwners,
                                       StrSpan(copyKey, static_cast<int32_t>(sizeof(RuleOwner))),
                                       ruleOwner, mulGenArena) > 0) {
                    continue;   // another rule still owns this copy
                }
                op.dropSet->mint(owner);
            }
            } // RT_SCOPE HM_COPY_OWNERS

            if (!op.isRemove() && &targetIntMemory == &mb.overallHashMemory
                && curKeyN > 0 && equalSpans(justificationSpan, StrSpan("implication", 11))) {
                unaryMemory.rememberUnaryImplication(mb, curKeyRun, curKeyN, curValueSpan, copySpan,
                    originalImplicationSpan, validityNameSpan, levels, levelCount, secondTarget == nullptr);
            }

            // --- Shared members (path-independent, needed for integration) ---
            // implication = curKey + curValue as a StrSpan run.
            StrSpan implRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1];
            int32_t implRunN = 0;
            for (int32_t i = 0; i < curKeyN; ++i) implRun[implRunN++] = curKeyRun[i];
            implRun[implRunN++] = curValueSpan;
            // Positional value-id mint (byte-identical to encodeValueVector, I-84);
            // the originals chain gains this rule as an owner via the raw
            // encodeIdVecKeyInto door (no owning IdVecKey).
            {
                RT_SCOPE_HERE("HM_ORIGINALS_CHAIN");
                int32_t implIds[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1];
                for (int32_t i = 0; i < implRunN; ++i)
                    implIds[i] = mb.ruleInterner.encode(implRun[i]);
                char ivKeyBuf[4 * (ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 2)];
                const int32_t ivLen = encodeIdVecKeyInto(implIds, implRunN,
                    ivKeyBuf, static_cast<int32_t>(sizeof(ivKeyBuf)));
                const StrSpan ivKey(ivKeyBuf, ivLen);
                if (!op.isRemove()) {
                    gl::HashRuleIndex::stageOwnerToRun(staging.originals, ivKey, owner);
                    if (staging2Ptr != nullptr) gl::HashRuleIndex::stageOwnerToRun(staging2Ptr->originals, ivKey, owner);
                } else if (op.firstVisit(RuleIndexOp::Chain, ivKey, owner, mulGenArena)) {
                    gl::HashRuleIndex::removeOwnerFromRun(targetIntMemory.originals, ivKey, owner,
                                       mulGenArena);
                }
            }
            for (int32_t tgtIx = 0; tgtIx < 2; ++tgtIx) {
                // Each target's own triggers, the second target after the
                // first (a second target is an install-only twin).
                HashMemory* const tgtPtr = (tgtIx == 0) ? &targetIntMemory : secondTarget;
                if (tgtPtr == nullptr) continue;
                HashMemory& trigTarget = *tgtPtr;
                // Decoded-lex INDEX over the triggers, sorted by decodeTemplateKeyView
                // (template then validity, compareSpans) — reproduces the former
                // std::sort(triggerRows) order (I-84). Per row copyFrom template +
                // validity to the string tier (MANDATORY: makeAdmissionKeys mints
                // templateInterner, the SAME interner the trigger decodes from, I-3).
                RT_SCOPE_HERE("HM_TRIGGERS");
                auto& trig = trigTarget.triggersForAdmissionSetIntegration;
                const int32_t trigN = trig.count();
                RT_NOTE_ITERATIONS_HERE(trigN);
                if (trigN > 0 && !op.isRemove()) {   // admission keys are install-only
                    const unsigned tgGenSlot = (gl::WorkerContext::coreId() >= 0)
                        ? static_cast<unsigned>(gl::WorkerContext::coreId())
                        : genScratchArenas().slotCount() - 1;
                    ScratchArena& tgArena = genScratchArenas().forSlot(tgGenSlot);
                    ScratchScope tgScope(tgArena);
                    int32_t* tidx = reinterpret_cast<int32_t*>(
                        tgArena.resolve(tgArena.alloc(trigN * 4, 4)));
                    for (int32_t i = 0; i < trigN; ++i) tidx[i] = i + 1;
                    std::sort(tidx, tidx + trigN, [&](int32_t a, int32_t b) {
                        StrSpan ka, va, kb, vb;
                        decodeTemplateKeyView(trig.keyAt(a), mb.templateInterner, mb.nameMap, ka, va);
                        decodeTemplateKeyView(trig.keyAt(b), mb.templateInterner, mb.nameMap, kb, vb);
                        const int ck = compareSpans(ka, kb);
                        if (ck != 0) return ck < 0;
                        return compareSpans(va, vb) < 0;
                    });
                    const unsigned tgStrSlot = (gl::WorkerContext::coreId() >= 0)
                        ? static_cast<unsigned>(gl::WorkerContext::coreId())
                        : scratchArenas().slotCount() - 1;
                    ScratchArena& tgStrArena = scratchArenas().forSlot(tgStrSlot);
                    for (int32_t i = 0; i < trigN; ++i) {
                        ScratchScope trScope(tgStrArena);
                        StrSpan tKey, tVal;
                        const int64_t triggerKey = trig.keyAt(tidx[i]);
                        decodeTemplateKeyView(triggerKey, mb.templateInterner, mb.nameMap, tKey, tVal);
                        const ScratchString tKeyC = ScratchString::copyFrom(tgStrArena, tKey.ptr, tKey.len);
                        const ScratchString tValC = ScratchString::copyFrom(tgStrArena, tVal.ptr, tVal.len);
                        admissionMemory.makeAdmissionKeys(implRun, implRunN, StrSpan(tKeyC),
                            trigTarget, StrSpan(tValC), mb);
                    }
                }
            }

            {
            RT_SCOPE_HERE("HM_ADMISSION_NORMKEYS");
            admissionMarkers.makeNormalizedKeysForAdmission(curKeyRun, curKeyN, targetIntMemory,
                mb.nameMap, mb.ruleInterner, curValueSpan, minNumOperatorsKey,
                copySpan, validityNameSpan, &mb, op, &encMemo, stagingPtr,
                equalSpans(justificationSpan, StrSpan("integration", 11)), goalNearMiss);
            if (secondTarget != nullptr) {
                admissionMarkers.makeNormalizedKeysForAdmission(curKeyRun, curKeyN, *secondTarget,
                    mb.nameMap, mb.ruleInterner, curValueSpan, minNumOperatorsKey,
                    copySpan, validityNameSpan, &mb, op, &encMemo, staging2Ptr,
                    equalSpans(justificationSpan, StrSpan("integration", 11)), goalNearMiss);
            }
            }

            // Remaining args: c==0 = the caller's sorted-unique remRun; c!=0 =
            // getRemainingArgs over curKeyRun (u_ vars may have been equalized),
            // also sorted-unique. 0% heap.
            const StrSpan* curRemRun;
            int32_t curRemN;
            StrSpan remScratch[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
            if (c == 0) {
                curRemRun = remRun; curRemN = remN;
            } else {
                RT_SCOPE_HERE("HM_REMARGS_MINT");
                curRemN = gl::HashKeyBuilder::getRemainingArgs(curKeyRun, curKeyN, remScratch,
                    ExecutionParameters::MAX_ADMISSION_REM_ARGS);
                curRemRun = remScratch;
            }

            if (performAdmissionMapUpdate && !op.isRemove())
            {
                RT_SCOPE_HERE("HM_UPDATE_ADMISSION_MAP");
                // PREMISES ONLY (curKeyRun), never the head: the
                // premise-completion demand asks for a missing premise
                // instance; the head is the rule's product and is never a
                // demand (D-379 — the head-marker
                // bug filed `!(in3[q,d,t,marker])` demands on the `×` slot
                // of the negated-existence expansion rules). The stored
                // admission key list is therefore the premise chain, and
                // the `isAdmitted` re-key replays premises only. The demand
                // is filed at the installing rule's own scope: a rule that
                // lives in an OR case branch demands only inside that
                // branch (the route-(a) probe reads the exact scope plus
                // strict ancestors), never at `main`.
                admissionMemory.updateAdmissionMap(mb,
                    curKeyRun, curKeyN,
                    curRemRun, curRemN,
                    maxAdmissionDepth,
                    maxSecondaryNumber,
                    admissionFlags,
                    validityNameSpan,
                    false,
                    weakHypothesisInstall);
            }

            // --- Inner: add this single key/value to int hash memory ---
            const int32_t n = curKeyN;
            NameMap& nm = mb.nameMap;

            // Premise patterns of this copy (the request statement filter,
            // premisePatternAdmits): every premise a rule of the copy reads in
            // encodedMap is owned by the copy at the install scope. The memo
            // already holds the encode of each such premise, so the reads here
            // mint nothing. An install also reopens a scope a subtree wipe
            // closed; a removal deletes the copy's owners.
            const auto premisePatternsOfCopy = [&](bool everyPremiseEncoded) {
                RT_SCOPE_HERE("HM_PREMISE_PATTERNS");
                if (!op.isRemove()) {
                    gl::HashRuleIndex::reopenPremisePatternScope(targetIntMemory, ownerVid);
                    if (secondTarget != nullptr)
                        gl::HashRuleIndex::reopenPremisePatternScope(*secondTarget, ownerVid);
                }
                for (int32_t i = 0; i < n; ++i) {
                    if (!encMemo.filled[i]) {
                        assert(!everyPremiseEncoded
                            && "addToHashMemory: a premise of an installed copy was never encoded");
                        continue;
                    }
                    const IntEncodedExpr& premise = encMemo.at(i, nm);
                    if (op.isRemove()) {
                        gl::HashRuleIndex::removePremisePattern(targetIntMemory, op, premise,
                            ownerVid, copyImplId, mulGenArena);
                        continue;
                    }
                    gl::HashRuleIndex::stagePremisePattern(staging.premisePatterns, premise,
                        ownerVid, copyImplId);
                    if (staging2Ptr != nullptr) {
                        gl::HashRuleIndex::stagePremisePattern(staging2Ptr->premisePatterns,
                            premise, ownerVid, copyImplId);
                    }
                }
            };

            // intRemainingArgs: mint NameMap in curRemRun (sorted-lex) order ==
            // the former set-lex mint order (I-84), then a sorted-ASCENDING copy
            // for the Int16SetKey == the former std::set<NameId> iteration.
            NameId intRemArgs[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
            NameId intRemArgsSorted[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
            {
            RT_SCOPE_HERE("HM_REMARGS_MINT");
            for (int32_t i = 0; i < curRemN; ++i) intRemArgs[i] = nm.encode(curRemRun[i]);
            if (curRemN > 0)
                std::memcpy(intRemArgsSorted, intRemArgs,
                    static_cast<std::size_t>(curRemN) * sizeof(NameId));
            std::sort(intRemArgsSorted, intRemArgsSorted + curRemN);
            }

            std::map<int, std::vector<std::vector<int>>>::const_iterator pit =
                this->allPermutationsAna.find(static_cast<int>(n));
            if (pit == this->allPermutationsAna.end()) {
                // The copy has more premises than the permutation table holds:
                // it gets no head or subkey record, but its admission-marker
                // variants (makeNormalizedKeysForAdmission above) are already
                // in encodedMap. Their premises are exactly the ones the memo
                // encoded, and those carry the copy's premise patterns.
                premisePatternsOfCopy(false);
                continue;
            }
            // The table entry for n only gates the install; its sorted
            // permutations are walked directly below.

            // idsRun[i] = the core expression span of curKeyRun[i] (the permutation
            // sort gate; compareSpans == the former ids[i] std::string compare).
            StrSpan idsRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
            for (int32_t i = 0; i < n; ++i)
                idsRun[i] = extractExpressionSpan(curKeyRun[i]);

            NameId bufIgnored[ExecutionParameters::MAX_KEY_SLOTS];
            NameId bufNotIgnored[ExecutionParameters::MAX_KEY_SLOTS];

            const unsigned genSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& genArena = genScratchArenas().forSlot(genSlot);
            const unsigned strSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : scratchArenas().slotCount() - 1;
            ScratchArena& strArena = scratchArenas().forSlot(strSlot);

            // Batch the remaining-args insert across the permutation loop: the
            // key (intRemArgsSorted/curRemN) is loop-invariant, so accumulate each
            // permutation's NormKey on the gen-scratch byte-bump tier and do ONE
            // RMW after the loop instead of one-per-permutation
            // (D-199). The blobs survive the loop's self-framing helpers
            // (appendLmvIdsRecord popTo's only its own frame).
            const ArenaOffset remArgsBatchMark = genArena.cursor();
            DirtyState remArgsBatchDirty = DirtyState::Clean;
            PagedVector<gl::RemArgsBatchBlob> remArgsBatch(&genArena, &remArgsBatchDirty);

            // Both shape detectors read only curKeyRun / curValueSpan, which
            // do not vary across the installed permutations — one detection
            // per multiplied copy, not one per permutation (at keyN >= 2 the
            // subset-exclusion registry scan would otherwise repeat up to
            // n! times).
            bool copyOrIntro = false;
            bool copySubsetExclusion = false;
            {
                RT_SCOPE_HERE("HM_SHAPE_DETECT");
                copyOrIntro = hashRuleClassifier.isOrIntroInstall(curKeyRun, curKeyN, curValueSpan);
                copySubsetExclusion =
                    hashRuleClassifier.isSubsetExclusionInstall(curKeyRun, curKeyN, curValueSpan);
            }

            {
            RT_SCOPE_HERE("HM_WHOLEKEY_RECORDS");
            // The permutations of the table whose operator names never
            // decrease, in table order, without walking the other n! ones.
            int64_t orderingsVisited = 0;
            gl::HashKeyBuilder::forEachSortedOrdering(idsRun, n, [&](const int32_t* permutation) {
                ++orderingsVisited;

                ScratchScope permScope(strArena);

                // Span-native int encode (no heap EncodedExpression -> no
                // parseArgument): each key element becomes an IntEncodedExpr via
                // the encodeExpression span twin, then makeIntNormalizedKeyFromEncoded*
                // (byte-identical outBuf + reverseMap to the EncodedExpression form).
                IntEncodedExpr intEncoded[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                const NameId intEncCount = static_cast<NameId>(n);
                assert(intEncCount <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                    && "addToHashMemory: key element count exceeds cap");
                {
                RT_SCOPE_HERE("HM_WK_ENCODE");
                for (int32_t k = 0; k < n; ++k) {
                    intEncoded[k] = encMemo.at(permutation[k], nm);
                }
                }

                NameId reverseMap[ExecutionParameters::MAX_KEY_SLOTS];
                std::memset(reverseMap, 0, sizeof(reverseMap));
                NameId numNormVars = 0;
                NameId lenIgnored = 0;
                NameId lenNotIgnored = 0;
                {
                RT_SCOPE_HERE("HM_WK_NORMKEY");
                lenIgnored = gl::HashKeyBuilder::makeIntNormalizedKeyFromEncodedWithMap(
                    intEncoded, intEncCount, true, bufIgnored,
                    ExecutionParameters::MAX_KEY_SLOTS, reverseMap, numNormVars);

                lenNotIgnored = gl::HashKeyBuilder::makeIntNormalizedKeyFromEncoded(
                    intEncoded, intEncCount, false, bufNotIgnored,
                    ExecutionParameters::MAX_KEY_SLOTS);
                }

                // Remove policy: this permutation's whole key and remaining-args
                // edge lose the owner (first visit only); no LMV, no batch.
                if (op.isRemove()) {
                    gl::HashRuleIndex::removeRuleKeyEntries(targetIntMemory, op,
                        static_cast<NameId>(n), bufNotIgnored, lenNotIgnored,
                        intRemArgsSorted, curRemN, owner, genArena);
                    return;
                }

                // Rename run { normalized-var name -> decimal id } from reverseMap.
                // decodeView keys: no NameMap mint falls between here and the last
                // replaceKeysScratch use (the mints below are ruleInterner), so
                // decodeView is I-3-safe. Byte-identical to the former lex-ordered
                // std::map (replaceKeysScratch is greedy-longest, order-independent).
                int32_t valueId = 0;
                bool isMarker = false;
                {
                RT_SCOPE_HERE("HM_WK_VALUE_VARIANT");
                StrReplacement mp2Pairs[ExecutionParameters::MAX_KEY_SLOTS];
                int32_t mp2PairsN = 0;
                for (NameId id = 1; id <= numNormVars; ++id) {
                    assert(mp2PairsN < ExecutionParameters::MAX_KEY_SLOTS
                        && "addToHashMemory: rename pair count exceeds cap");
                    char decTmp[12];
                    const int32_t decN = writeDecimalDigits(decTmp, id);
                    char* decBuf = strArena.allocBytes(decN);
                    std::memcpy(decBuf, decTmp, static_cast<std::size_t>(decN));
                    mp2Pairs[mp2PairsN].key = nm.decodeView(reverseMap[id]);
                    mp2Pairs[mp2PairsN].value = StrSpan(decBuf, decN);
                    ++mp2PairsN;
                }
                const ScratchString valueVariantS =
                    replaceKeysScratch(strArena, curValueSpan, mp2Pairs, mp2PairsN);

                // LMV fields (id-run door). ruleInterner mint ORDER: valueVariant,
                // then each remaining arg (the copy text itself was minted with
                // the owner above). keyIds empty (the head install carries none).
                valueId = mb.ruleInterner.encode(StrSpan(valueVariantS));
                isMarker =
                    containsSpan(StrSpan(valueVariantS), StrSpan("marker", 6));
                } // RT_SCOPE HM_WK_VALUE_VARIANT
                const int32_t originalImplId = copyImplId;
                int32_t remIds[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                {
                RT_SCOPE_HERE("HM_WK_LMV_FIELDS");
                for (int32_t i = 0; i < curRemN; ++i)
                    remIds[i] = mb.ruleInterner.encode(curRemRun[i]);
                }
                const RuleJustification just = ruleJustificationFromString(justificationSpan);

                // D-32: product-of-disintegration — true iff at least one premise
                // (chain element) has an arg starting with "u_" (bound-variable
                // placeholder). Consumed by checkLocalEncodedMemoryStatic to gate
                // OR-disintegration on the head-firing path.
                bool productOf = false;
                {
                RT_SCOPE_HERE("HM_WK_LMV_FIELDS");
                for (int32_t i = 0; i < curKeyN && !productOf; ++i) {
                    StrSpan pArgs[ExecutionParameters::MAX_ARITY];
                    const int32_t pArgsN = getArgsSpans(curKeyRun[i], pArgs,
                                                        ExecutionParameters::MAX_ARITY);
                    for (int32_t a = 0; a < pArgsN; ++a) {
                        if (pArgs[a].len >= 2 && pArgs[a].ptr[0] == 'u' && pArgs[a].ptr[1] == '_') {
                            productOf = true; break;
                        }
                    }
                }
                }

                // D-241: may heads fired
                // by this rule be disintegrated. False for integration
                // instructions and for or-intro-shaped rules (their heads are
                // integration instructions consumed flat); the firing site
                // reads this as the rule-intrinsic half of
                // FiringRecord::doNotDisintegrate.
                const bool disintegrationAllowed =
                    just != RuleJustification::integration
                    && !copyOrIntro;

                // I-184: shape-detected
                // subset-exclusion rules fire PARK-FIRST — the firing site
                // strips the route-(b) open signal from their heads, so the
                // fired reduced or parks in rejectedMapOrdis until the
                // compound-demand map opens it.
                const bool subsetExclusion = copySubsetExclusion;

                // encodedMap HEAD record + owner record + remaining-args index,
                // all via the raw / id-run doors (no owning NormKey / LMV). The
                // levels run is caller-owned ascending-unique (I-136) == the former
                // std::set<int> serialize order.
                {
                RT_SCOPE_HERE("HM_LMV_APPEND");
                stageLmvIdsRecord(staging.lmv,
                    static_cast<NameId>(n), bufIgnored, lenIgnored,
                    valueId, isMarker, nullptr, 0, remIds, curRemN,
                    originalImplId, ownerVid, genArena,
                    levels, levelCount, just, productOf, disintegrationAllowed,
                    /*ordisOnly=*/false, subsetExclusion);
                if (staging2Ptr != nullptr) {
                    stageLmvIdsRecord(staging2Ptr->lmv,
                        static_cast<NameId>(n), bufIgnored, lenIgnored,
                        valueId, isMarker, nullptr, 0, remIds, curRemN,
                        originalImplId, ownerVid, genArena,
                        levels, levelCount, just, productOf, disintegrationAllowed,
                        /*ordisOnly=*/false, subsetExclusion);
                }
                }
                {
                RT_SCOPE_HERE("HM_WHOLEKEY_OWNER");
                gl::HashRuleIndex::stageWholeKeyOwner(staging.wholeKeys,
                    static_cast<NameId>(n), bufNotIgnored, lenNotIgnored, owner);
                if (staging2Ptr != nullptr) {
                    gl::HashRuleIndex::stageWholeKeyOwner(staging2Ptr->wholeKeys,
                        static_cast<NameId>(n), bufNotIgnored, lenNotIgnored, owner);
                }
                }
                // Accumulate this permutation's NormKey (Codec<NormKey> bytes:
                // int16 numberExpressions ++ int16 length ++ data) for the one
                // batched insert after the loop.
                {
                    const int32_t nkBlobLen = (lenNotIgnored + 2)
                        * static_cast<int32_t>(sizeof(NameId));
                    const ArenaOffset nkBlobOff = genArena.alloc(
                        nkBlobLen, static_cast<int32_t>(alignof(NameId)));
                    NameId* pnk =
                        reinterpret_cast<NameId*>(genArena.resolve(nkBlobOff));
                    pnk[0] = static_cast<NameId>(n);
                    pnk[1] = lenNotIgnored;
                    std::memcpy(pnk + 2, bufNotIgnored,
                        static_cast<std::size_t>(lenNotIgnored) * sizeof(NameId));
                    remArgsBatch.push_back(gl::RemArgsBatchBlob{ nkBlobOff, nkBlobLen });
                }
            });
            RT_NOTE_ITERATIONS_HERE(orderingsVisited);   // sorted orderings installed
            } // RT_SCOPE HM_WHOLEKEY_RECORDS
            {
            RT_SCOPE_HERE("HM_REMARGS_BATCH");
            // ONE RMW for the whole permutation batch into the invariant key.
            gl::HashRuleIndex::stageRemainingArgsNormKeyBatch(
                staging.remArgs,
                staging.remArgsOwners,
                intRemArgsSorted, curRemN, remArgsBatch, owner, genArena);
            if (staging2Ptr != nullptr) {
                gl::HashRuleIndex::stageRemainingArgsNormKeyBatch(
                    staging2Ptr->remArgs,
                    staging2Ptr->remArgsOwners,
                    intRemArgsSorted, curRemN, remArgsBatch, owner, genArena);
            }
            genArena.popTo(remArgsBatchMark);
            } // RT_SCOPE HM_REMARGS_BATCH

            // --- makeNormalizedSubkeys equivalent ---
            {
            RT_SCOPE_HERE("HM_SUBKEYS");
            // Every operator-name-sorted prefix of the premise permutations,
            // each built and staged once, in the order the permutation table
            // first reaches it (forEachSortedPrefix). The prefix's premises
            // come from the per-copy memo and its normalized key extends the
            // parent prefix's key by one premise (`appendExprToIntNormalizedKey`):
            // after a prefix of length L the first `st.length()` slots ARE
            // its subkey (the resumable builder's contract); leaving the prefix
            // restores the parent's checkpoint. subEncoded[0..index] is the
            // prefix in key order — what the signature record reads. A prefix
            // the table reaches again would stage the same contribution, which
            // the flush's sorted-unique merge drops.
            IntEncodedExpr subEncoded[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
            NameId subBuf[ExecutionParameters::MAX_KEY_SLOTS];
            gl::HashKeyBuilder st;
            std::optional<gl::HashKeyBuilder::Checkpoint> parentKey[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
            int64_t prefixesVisited = 0;
            gl::HashKeyBuilder::forEachSortedPrefix(idsRun, n,
                [&](int32_t index, const int32_t* permut) {
                    ++prefixesVisited;
                    parentKey[index].emplace(st.checkpoint());

                    const NameId subEncCount = static_cast<NameId>(index + 1);
                    assert(subEncCount <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                        && "addToHashMemory subkey: element count exceeds cap");
                    {
                    RT_SCOPE_HERE("HM_SK_ENCODE");
                    subEncoded[index] = encMemo.at(permut[index], nm);
                    }

                    NameId subLen = 0;
                    {
                    RT_SCOPE_HERE("HM_SK_NORMKEY");
                    st.append(subEncoded[index], subBuf, ExecutionParameters::MAX_KEY_SLOTS);
                    subLen = st.length();
                    }

                    // A subkey below kSubkeyUCheckMinElements has an empty
                    // persistent record; from there on its
                    // u_-signature record gains the owner with the signature's
                    // index — via the raw-key door (no owning NormKey). Under
                    // the remove policy only long subkeys lose owner pairs.
                    if (op.isRemove()) {
                        gl::HashRuleIndex::removeRuleSubkeyEntry(targetIntMemory, op,
                            static_cast<NameId>(index + 1), subBuf, subLen,
                            owner, genArena);
                    } else if (subEncCount < gl::HashRuleIndex::kSubkeyUCheckMinElements) {
                        RT_SCOPE_HERE("HM_SK_WRITE_SHORT");
                        gl::HashRuleIndex::stageEmptySubkeyRecord(staging.subkeys,
                            static_cast<NameId>(index + 1), subBuf, subLen);
                        if (staging2Ptr != nullptr) {
                            gl::HashRuleIndex::stageEmptySubkeyRecord(staging2Ptr->subkeys,
                                static_cast<NameId>(index + 1), subBuf, subLen);
                        }
                    } else {
                        RT_SCOPE_HERE("HM_SK_WRITE_MERGE");
                        gl::HashRuleIndex::stageSubkeySignatures(staging.subkeys,
                            static_cast<NameId>(index + 1), subBuf, subLen,
                            subEncoded, subEncCount, owner);
                        if (staging2Ptr != nullptr) {
                            gl::HashRuleIndex::stageSubkeySignatures(staging2Ptr->subkeys,
                                static_cast<NameId>(index + 1), subBuf, subLen,
                                subEncoded, subEncCount, owner);
                        }
                    }
                },
                [&](int32_t index) { st.restore(*parentKey[index]); });
            RT_NOTE_ITERATIONS_HERE(prefixesVisited);   // sorted prefixes staged
            } // RT_SCOPE HM_SUBKEYS
            premisePatternsOfCopy(true);
            if (!op.isRemove()) {
                targetIntMemory.maxKeyLength =
                    std::max(static_cast<NameId>(n), targetIntMemory.maxKeyLength);
                if (secondTarget != nullptr) {
                    secondTarget->maxKeyLength =
                        std::max(static_cast<NameId>(n), secondTarget->maxKeyLength);
                }
            }
        }

        // The outermost window of this LB closes here (LB seeding, the CE
        // facts, the compressor, a standalone door call): every instance the
        // window's nested rounds staged into is flushed and released. Inside an
        // outer window (the absorbs, the apply, an enclosing install) that
        // window's closer flushes every install of the window at once.
        --mb.ruleStagingArmDepth;
        if (mb.ruleStagingArmDepth == 0) {
            gl::HashRuleIndex::closeRuleIndexStaging(mb);
        }
    }

/// @brief Install a goal-driven marked-head implication: its present
///        premises imply the operator premise with `marker` in the output
///        slot (D-574).
///
/// @details
/// The goal-driven rule search (`IntegrationProcessor::forEachGoalDemandVariant`)
/// hands one qualifying operator premise of a rule whose head it bound to a
/// goal's missing expression: @p key holds the rule's premises that do not
/// name the operator premise's output, then the operator premise last. The
/// implication is installed as one admission-marker variant family
/// (`AdmissionMarkerBuilder::installAdmissionMarkerVariants`: every element
/// but the last is the key, @p markedArg the marked argument) into the
/// three instances of the integration route — overall, local and delta, so
/// the new rule meets every visible statement once — with justification
/// `goalDemand`, which the marker branch of the firing site exempts from
/// the purity gate. A firing instantiates the head into the ordinary algebra
/// demand of the output (`drainAdmissionKeysAlgebra`), which wakes an
/// existence parked under the same key.
///
/// The install is a staging window of its own; inside an enclosing window
/// that window's closer flushes it.
///
/// @param mb        The owning LB (single-threaded seam).
/// @param key       The present premises, then the operator premise
///                  (`u_` form); spans stable across the call.
/// @param keyN      Their count; at least two, at most
///                  `MAX_ADMISSION_KEY_ELEMENTS + 1`.
/// @param markedArg The operator premise's output argument: once in the
///                  operator premise and in no present premise (asserted).
/// @param identity  The implication's identity text; with @p validity the
///                  owner of every index entry and LMV of the install.
/// @param validity  The install scope, an existing scope name.
/// @return Nothing.
/// @invariant Writes no originals chain; mints only through the marker
///            door; 0% heap.
/// @see AdmissionMarkerBuilder::installAdmissionMarkerVariants,
///      IntegrationProcessor::serveGoalNearMiss.
void HashRuleInstaller::installGoalDemandVariant(Memory& mb, const StrSpan* key, int32_t keyN,
        StrSpan markedArg, StrSpan identity, StrSpan validity) {
    constexpr int32_t kKeyCap = ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1;
    assert(keyN >= 2 && keyN <= kKeyCap
        && "installGoalDemandVariant: an implication needs present premises and its operator premise");
    const int32_t index = keyN - 1;
    {
        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int32_t argN = getArgsSpans(key[index], args, ExecutionParameters::MAX_ARITY);
        int32_t occurrences = 0;
        for (int32_t a = 0; a < argN; ++a)
            if (equalSpans(args[a], markedArg)) ++occurrences;
        assert(occurrences == 1
            && "installGoalDemandVariant: the output occurs once in the operator premise");
        (void)occurrences;
        for (int32_t i = 0; i < index; ++i) {
            const int32_t n = getArgsSpans(key[i], args, ExecutionParameters::MAX_ARITY);
            for (int32_t a = 0; a < n; ++a)
                assert(!equalSpans(args[a], markedArg)
                    && "installGoalDemandVariant: a present premise names the output");
        }
    }
    // The key is every element but the operator premise, in run order, so
    // the run itself is the subkey.
    int binary[kKeyCap];
    for (int32_t i = 0; i < keyN; ++i) binary[i] = (i == index) ? 0 : 1;
    const unsigned slot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    const NameId ownerVid = mb.nameMap.encode(validity);
    const RuleOwner owner = packRuleOwner(mb.ruleInterner.encode(identity), ownerVid);
    ++mb.ruleStagingArmDepth;
    HashMemory* const targets[3] = {
        &mb.overallHashMemory, &mb.localHashMemory, &mb.localHashMemoryDelta };
    for (HashMemory* target : targets) {
        KeyEncodeMemo memo;
        memo.bind(key, keyN);
        RuleIndexStaging& staging = ruleStagings().acquire(slot, target);
        admissionMarkers.installAdmissionMarkerVariants(key, static_cast<std::size_t>(keyN),
            static_cast<std::size_t>(index), markedArg, binary, key, index,
            *target, mb.nameMap, mb.ruleInterner, identity, ownerVid, owner,
            /*ordisOnly=*/false, /*ordis2Demand=*/false, RuleIndexOp{}, memo, staging,
            RuleJustification::goalDemand);
    }
    --mb.ruleStagingArmDepth;
    if (mb.ruleStagingArmDepth == 0) {
        gl::HashRuleIndex::closeRuleIndexStaging(mb);
    }
}

/// @brief Collect the `u_`-prefixed "remaining" argument names of an
///        expression, with the `u_` prefix stripped, as zero-copy slices.
///
/// @details Scans every `[...]` argument block NOT immediately preceded by `>`
///          (a `>[...]` block is the forbidden zone for `u_` and is only
///          asserted clean). Each comma-separated, trimmed token beginning with
///          `u_` yields one output slice `token[2:]` — a `StrSpan` into the
///          caller's `str` bytes, so the slices live exactly as long as `str`.
///          Duplicates are emitted in scan order; the caller dedups (typically
///          by inserting into a `std::set<std::string>`), which is why this
///          routine owns no container and is 0% heap. The lexical scan is the
///          byte-exact twin of the former `std::sregex_iterator` over
///          `\[([^\]]*)\]` (see `extract_remaining_args_matches_regex_oracle`),
///          and the forbidden-zone / additional-`u_` invariants are asserted
///          exactly as before.
///
/// @param str Expression to scan (borrowed; slices point into it).
/// @param out Caller-provided array receiving the `u_`-stripped slices.
/// @param cap Capacity of `out`; overflow asserts (Rule 19 tripwire).
/// @return Number of slices written to `out[0..return)` (duplicates possible).
///
/// @invariant No token inside a `>[...]` block starts with `u_`; a `u_` token
///            contains no further `u_` substring — both asserted.
/// @see The caller `addExprToMemoryBlock` builds the deduped `std::set` for
///      `addToHashMemory`.
int32_t HashRuleInstaller::extractRemainingArgs(StrSpan str, StrSpan* out, int32_t cap) {
    int32_t count = 0;
    const StrSpan s = str;
    const int32_t n = s.len;

    // Byte-exact trim twin of ce::trimCopy: the IDENTICAL
    // std::isspace(static_cast<unsigned char>(c)) predicate on both ends, so it
    // matches trimCopy byte-for-byte under any locale (MPL is space-free — this
    // trims nothing in practice — but byte-fidelity keeps the twin honest).
    const auto trimSpan = [](StrSpan t) -> StrSpan {
        int32_t i = 0;
        while (i < t.len && std::isspace(static_cast<unsigned char>(t.ptr[i]))) ++i;
        int32_t j = t.len;
        while (j > i && std::isspace(static_cast<unsigned char>(t.ptr[j - 1]))) --j;
        return StrSpan(t.ptr + i, j - i);
    };

    // 1. Check >[...] blocks (Strict Forbidden Zone for "u_"). Lexical twin of
    //    the unanchored iterator over >\[([^\]]*)\]: an attempt needs "> [" at
    //    pos, the group runs to the FIRST ']' at/after pos+2, resume after ']';
    //    a failed start advances one byte; no ']' at/after an attempt means no
    //    match here NOR at any later attempt (\] can never match), so terminate.
    {
        int32_t pos = 0;
        while (pos + 1 < n) {
            if (s.ptr[pos] != '>' || s.ptr[pos + 1] != '[') { ++pos; continue; }
            int32_t close = pos + 2;
            while (close < n && s.ptr[close] != ']') ++close;
            if (close >= n) break;   // no ']' here nor at any later attempt
            // group = [pos+2, close): comma-split, trim, assert no u_.
            int32_t start = pos + 2;
            for (int32_t p = pos + 2; p <= close; ++p) {
                if (p == close || s.ptr[p] == ',') {
                    const StrSpan token = trimSpan(StrSpan(s.ptr + start, p - start));
                    if (token.len > 0) {
                        const bool hasUPrefix = token.len >= 2
                            && token.ptr[0] == 'u' && token.ptr[1] == '_';
                        assert(!hasUPrefix && "Violation: Argument starting with 'u_' found inside >[...] block");
                        (void)hasUPrefix;
                    }
                    start = p + 1;
                }
            }
            pos = close + 1;
        }
    }

    // 2. Collect "u_" args from [...] blocks (NOT preceded by '>'). Lexical twin
    //    of the unanchored iterator over \[([^\]]*)\]: an attempt needs '[' at
    //    pos, the group runs to the first ']', resume after ']' ALWAYS, but the
    //    group is SKIPPED (not the consume) when the '[' is preceded by '>' —
    //    the old loop's `continue`; a naive skip-and-rescan of the interior
    //    would diverge (the "[a>[b]c]" trap).
    {
        int32_t pos = 0;
        while (pos < n) {
            if (s.ptr[pos] != '[') { ++pos; continue; }
            int32_t close = pos + 1;
            while (close < n && s.ptr[close] != ']') ++close;
            if (close >= n) break;   // no ']' here nor at any later attempt
            const bool skip = (pos > 0 && s.ptr[pos - 1] == '>');
            if (!skip) {
                int32_t start = pos + 1;
                for (int32_t p = pos + 1; p <= close; ++p) {
                    if (p == close || s.ptr[p] == ',') {
                        const StrSpan token = trimSpan(StrSpan(s.ptr + start, p - start));
                        if (token.len >= 2 && token.ptr[0] == 'u' && token.ptr[1] == '_') {
                            // uCount = non-overlapping "u_" occurrences, step +2
                            // (replicating the old `p = find("u_", p); p += 2`).
                            int uCount = 0;
                            int32_t q = 0;
                            while (q + 1 < token.len) {
                                if (token.ptr[q] == 'u' && token.ptr[q + 1] == '_') {
                                    ++uCount; q += 2;
                                } else {
                                    ++q;
                                }
                            }
                            assert(uCount == 1 && "Argument starting with 'u_' must not contain additional 'u_' substrings");
                            (void)uCount;
                            // Emit the "u_"-stripped slice (a view into `str`;
                            // token == "u_" emits the empty slice, the substr(2)
                            // twin). The caller dedups/sorts (e.g. into a set).
                            assert(count < cap && "extractRemainingArgs: out capacity exceeded");
                            out[count++] = StrSpan(token.ptr + 2, token.len - 2);
                        }
                        start = p + 1;
                    }
                }
            }
            pos = close + 1;
        }
    }

    return count;
}
}
