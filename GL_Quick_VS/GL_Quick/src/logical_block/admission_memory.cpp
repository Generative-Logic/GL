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
#include "admission_memory.hpp"
namespace gl {
/// @brief Bind inputs for admission memory.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param admissionPolicy Borrowed AdmissionPolicy interface or state; must outlive this component.
/// @param anchorInfo Borrowed ce::AnchorInfo interface or state; must outlive this component.
/// @param canonicalTemplates Borrowed CanonicalTemplateMatcher interface or state; must outlive this component.
/// @param deferredAncestorAdmissionsMutex Borrowed std::mutex interface or state; must outlive this component.
/// @param deferredAncestorPages Borrowed std::optional interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param scopeTypes Borrowed ScopeTypeResolver interface or state; must outlive this component.
/// @param steward Borrowed std::optional interface or state; must outlive this component.
/// @param integrationProcessor Borrowed IntegrationProcessor interface or state; must outlive this component.
/// @param rejectedSourceReplay Borrowed RejectedSourceReplay interface or state; must outlive this component.
/// @param equivalenceQueries Borrowed EquivalenceQueries interface (the known-statement probe of the gates); must outlive this component.
/// @return None; construction binds the supplied references.
AdmissionMemory::AdmissionMemory(AdmissionPolicy& admissionPolicy, const ce::AnchorInfo& anchorInfo, CanonicalTemplateMatcher& canonicalTemplates, std::mutex& deferredAncestorAdmissionsMutex, std::optional<SealedPageSet>& deferredAncestorPages, const CompiledDefinitionRegistry& definitions, const ProverParameters& parameters, ScopeTypeResolver& scopeTypes, std::unique_ptr<MemorySteward>& steward, IntegrationProcessor& integrationProcessor, RejectedSourceReplay& rejectedSourceReplay, EquivalenceQueries& equivalenceQueries) : admissionPolicy(admissionPolicy), anchorInfo(anchorInfo), canonicalTemplates(canonicalTemplates), deferredAncestorAdmissionsMutex(deferredAncestorAdmissionsMutex), deferredAncestorPages(deferredAncestorPages), definitions(definitions), parameters(parameters), scopeTypes(scopeTypes), steward(steward), integrationProcessor(integrationProcessor), rejectedSourceReplay(rejectedSourceReplay), equivalenceQueries(equivalenceQueries) {}

/// @brief Replays the per-burst staged algebra-`admissionMap` writes
    ///        recorded on `Memory::admissionKeysAlgebra`, once, after the
    ///        hashburst fixpoint loop.
    ///
    /// @details
    /// Companion drain to the record append in
    /// `checkLocalEncodedMemoryStatic`'s marker branch. For each
    /// `AdmissionKeyAlgebraRecord` on `memoryBlock.admissionKeysAlgebra`, in
    /// firing order, it replays the four writes the marker branch used to
    /// perform inline mid-burst:
    ///   1. insert `record.value` into `overallHashMemory.admissionMap` under
    ///      `record.key` (fresh per-record lookup; create the set if absent);
    ///   2. set `admissionStatusMap[record.key] = false`;
    ///   3. add every non-`marker` argument of `record.key.original` to
    ///      `varsInAdmissionMapKeys`;
    ///   4. fire `revisitRejected2`, depositing any revival cohort on
    ///      `sameIterationInternalMail`.
    /// Called once from `performElemPhase2`, immediately before phase 3's
    /// post-burst `standardProcessing`, so the revival cohorts are present for
    /// that absorb and `admissionMap` is current before its `isAdmitted` reads
    /// it.
    ///
    /// @param memoryBlock The LB whose `admissionKeysAlgebra` buffer is
    ///        replayed and whose `overallHashMemory` admission containers
    ///        receive the writes. The buffer is left intact; the burst-start
    ///        clear in `performElem2` empties it.
    /// @return (void)
    /// @invariant Drains in append (firing) order.
    /// @invariant FROZEN interner-touch sequence per record: (1)
    ///            `templateInterner.encode(key.original)` inside
    ///            `mintTemplateKey`, (2) `nameMap.encode(key.validityName)`
    ///            inside `mintTemplateKey`, (3) `valueInterner.encode` per
    ///            staged key element in RULE order inside `stagedToArenaBlob`,
    ///            (4) `valueInterner.encode` per staged remainingArg in
    ///            presorted LEX order inside `stagedToArenaBlob` (the same
    ///            positions the retained `stagedToIdValue` oracle touches),
    ///            (5)
    ///            `templateInterner.encode` per non-`"marker"` argument of
    ///            `key.original` in `getArgs` order, (6) `revisitRejected2`'s
    ///            interior mints. Same bytes at the same positions →
    ///            identical ids → identical deload streams.
    /// @see `memory.cpp::checkLocalEncodedMemoryStatic` — the producer.
    /// @see `AdmissionKeyAlgebraRecord` — the staged record type.
void AdmissionMemory::drainAdmissionKeysAlgebra(Memory& memoryBlock) {
        static const bool traceEnabled = [] {
#ifdef _WIN32
            size_t required = 0;
            const errno_t result = getenv_s(&required, nullptr, 0, "GL_FTA_BLOWUP_TRACE");
            assert(result == 0);
            return required != 0;
#else
            return std::getenv("GL_FTA_BLOWUP_TRACE") != nullptr;
#endif
        }();
        const bool traceTarget = traceEnabled && hashburst_dump::isTargetLB(memoryBlock);
        // This drain worker's slot + gen-scratch arena (the house idiom):
        // hosts each record's serialized admission blob + the RMW splice's
        // verbatim run copies, reclaimed per record by the popTo below.
        const unsigned drainSlot = (gl::WorkerContext::coreId() >= 0)
            ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : genScratchArenas().slotCount() - 1;
        ScratchArena& drainArena = genScratchArenas().forSlot(drainSlot);
        for (const AdmissionKeyAlgebraRecord& rec : memoryBlock.admissionKeysAlgebra) {
            // The staged record carries the template as SEALED VIEWS (the
            // parallel producer must not mint — I-68/I-83); this drain is
            // the single-threaded write site that interns the sealed views
            // through the span doors — no materialization. The spans alias
            // the record's page set, sealed (stable) until the post-drain
            // sweep, so they stay valid across the whole record replay.
            const StrSpan keyOriginal(rec.key.original);
            const StrSpan keyValidity(rec.key.validityName);
            if (traceTarget) {
                // permanent instrument: GL_FTA_BLOWUP_TRACE (AGENTS.md Rule 36), off unless the variable is set
                std::ofstream trace(".debug/traps.log", std::ios::app);
                assert(trace.is_open());
                trace << "[FTA-DEMAND-DRAIN] key=" << keyOriginal.toStdString()
                      << " scope=" << keyValidity.toStdString() << " ordis=" << rec.value.ordisOnly << "\n";
            }
            const int64_t recPk = mintTemplateKey(memoryBlock.templateInterner,
                memoryBlock.nameMap, keyOriginal, keyValidity);

            // (1) admissionMap insert — this drain is the single-threaded
            // intern point for the staged value (I-68/I-83). Fast path: the
            // staged strings encode STRAIGHT into the serialized blob
            // (stagedToArenaBlob — the stagedToIdValue interner-touch order
            // preserved byte-for-byte) and the blob splices into the
            // canonical run with no heap value and no heap set.
            {
                const ArenaOffset recMark = drainArena.cursor();
                const StrSpan blob = stagedToArenaBlob(
                    rec.value, memoryBlock.valueInterner, drainArena);
                insertAdmissionBlobSorted(
                    memoryBlock.overallHashMemory.admissionMap, recPk,
                    blob.ptr, blob.len, memoryBlock.valueInterner, drainArena);
                drainArena.popTo(recMark);
            }

            // (3) admissionStatusMap. The status is the template's
            // recursion ownership and merges like updateAdmissionMap's
            // write: a recursion-owned record sets it, an ordinary record
            // keeps an existing recursion status instead of clearing it
            // (D-413; an overwrite with
            // 0 silenced updateAdmissionMapRecursion for the induction
            // hypothesis template once an ordinary rule demanded the same
            // template).
            const uint8_t* existingStatus = memoryBlock.overallHashMemory.admissionStatusMap.find(recPk);
            const bool recursionStatus = rec.value.flag
                || (existingStatus != nullptr && *existingStatus != 0);
            memoryBlock.overallHashMemory.admissionStatusMap.upsert(recPk,
                static_cast<uint8_t>(recursionStatus ? 1 : 0));


            // (4) vars-in-keys cache for the algebra equi-class hook. The
            // args are zero-copy span slices, encoded per-arg in
            // ce::getArgs left-to-right order (getArgsSpans is the
            // documented content+order twin).
            StrSpan argSpans[ExecutionParameters::MAX_ARITY];
            const int argN = getArgsSpans(keyOriginal, argSpans,
                ExecutionParameters::MAX_ARITY);
            for (int i = 0; i < argN; ++i) {
                const StrSpan a = argSpans[i];
                if (!equalSpans(a, StrSpan("marker", 6))) {
                    memoryBlock.overallHashMemory.varsInAdmissionMapKeys.mint(
                        memoryBlock.templateInterner.encode(a));
                }
            }

            // (5) Revival — deposits the rejection cohort on
            // sameIterationInternalMail (spans over the sealed record).
            // Ordis-only key gains must not probe the GENERAL rejectedMap
            // (they are cohort-opening demand evidence; their revival seam
            // is revisitRejectedOrdis).
            if (!rec.value.ordisOnly) {
                rejectedSourceReplay.revisitRejected2(keyOriginal, memoryBlock, keyValidity);
            }
            // (6) Ordis revival — parked cohorts wake on EVERY key gain,
            // tagged or untagged (the cohort probe reads both).
            rejectedSourceReplay.revisitRejectedOrdis(keyOriginal, memoryBlock, keyValidity);
        }
        // (7) Waiting demands of demand-built rules whose rule or input
        // typing arrived after they were filed
        // (D-596).
        rejectedSourceReplay.buildWaitingInstances(memoryBlock);
    }

/// @brief Replays the per-burst staged integration-template registrations
    ///        recorded on `Memory::deferredIntegrationPreps`, once, after the
    ///        hashburst fixpoint loop.
    ///
    /// @details
    /// Integration-side companion of `drainAdmissionKeysAlgebra`. For each
    /// `DeferredIntegrationPrep` on `memoryBlock.deferredIntegrationPreps`, in
    /// firing order, it replays the `prepareIntegration` call that
    /// `checkLocalEncodedMemoryStatic`'s marker branch deferred — registering the
    /// integration admission template (`admissionMapIntegration` plus the
    /// `varsInAdmissionMapIntegrationKeys` cache, via `prepareIntegrationCore2`).
    /// A record whose missing premise has a uniquely definable open argument
    /// already named by a statement known at the firing scope or an ancestor
    /// is not replayed (`IntegrationProcessor::knownVersionSettlesMarkedPremise`,
    /// D-568).
    /// The sealed views feed the `prepareIntegration` span core directly —
    /// no set rebuild and no string materialization at this boundary; the
    /// core's two compiled-definition edge strings are built inside it, past
    /// its early-return gates. Called once from `performElemPhase2`,
    /// immediately before phase 3's post-burst `standardProcessing`, so the
    /// templates are registered before that absorb's
    /// `updateAdmissionMapIntegration` cascade consumes them.
    ///
    /// @param memoryBlock The LB whose `deferredIntegrationPreps` buffer is
    ///        replayed; its `overallHashMemory` integration-admission containers
    ///        receive the writes. The buffer is left intact; the burst-start
    ///        clear in `performElem2` empties it.
    /// @return (void)
    /// @invariant Drains in append (firing) order.
    /// @see `memory.cpp::checkLocalEncodedMemoryStatic` — the producer.
    /// @see `DeferredIntegrationPrep` — the staged record type.
void AdmissionMemory::drainDeferredIntegrationPreps(Memory& memoryBlock) {
        for (const DeferredIntegrationPrep& rec : memoryBlock.deferredIntegrationPreps) {
            // Sealed views pass straight through the prepareIntegration span
            // core — no set rebuild, no materialization at this boundary
            // (the core's two compiled-definition edge strings live inside
            // it, past the gates). The ascending-strict assert pins the
            // producer contract the pass-through relies on: the record's
            // args are the former set as a sorted-unique sealed span
            // (FiringRecord.markerArgsSorted, forwarded verbatim by
            // applyFiringRecords). A firing assert is a gift (Rule 19).
            const int32_t n = rec.unchangeableArgsSorted.size();
            assert(n <= ExecutionParameters::MAX_ARITY);
            StrSpan args[ExecutionParameters::MAX_ARITY];
            for (int32_t i = 0; i < n; ++i) {
                args[i] = StrSpan(rec.unchangeableArgsSorted[i]);
                assert((i == 0 || compareSpans(args[i - 1], args[i]) < 0)
                    && "deferred integration args not sorted-unique — producer contract broken");
            }
            // A missing premise whose open argument is uniquely definable is
            // not integrated when a statement known at the firing scope or an
            // ancestor already names that object: the integration could only
            // rebuild it (D-568).
            if (integrationProcessor.knownVersionSettlesMarkedPremise(StrSpan(rec.expression), args, n,
                    memoryBlock, StrSpan(rec.validityName)))
                continue;
            // Marker replays are statement-driven — no owning goal, so no
            // goal-carrying payload (the replay's mints nest under scopes the
            // admission machinery already owns, or stay bare on main).
            integrationProcessor.prepareIntegration(StrSpan(rec.expression), args, n,
                               memoryBlock, StrSpan(rec.validityName), StrSpan());
        }
    }

/// @brief Drain the per-burst ordis2 demand records into
        ///        `admissionMapOrdis2` and wake the or heads filed in
        ///        `rejectedMapOrdis2` under the same keys
        ///        (D-267).
        ///
        /// @details
        /// Third sibling of `drainAdmissionKeysAlgebra` /
        /// `drainDeferredIntegrationPreps` at the same post-fixpoint seam
        /// (`performElemPhase2`, fixed call order after the integration
        /// drain). Per record, a FROZEN interner-touch sequence:
        /// (1) `mintTemplateKey(expression, validity)` — the packed demand
        /// key; (2) `insertAdmissionOrdis2IdsBlob` — the canonical RMW into
        /// `admissionMapOrdis2` (re-demands dedup); (3)
        /// `revisitRejectedOrdis2(expression, ...)` — the pair's key
        /// rendezvous: `rejectedMapOrdis2` is keyed by disjunct clean
        /// GROUND texts, the demand's own key language, so an or filed
        /// under a byte-equal text is un-known and re-deposited on
        /// `sameIterationInternalMail`; the post-burst absorb re-runs the
        /// opener, whose route (c) reads the fresh demand entry and opens
        /// (consuming it). Never a mid-burst map write — the records were
        /// staged in canonical firing order by `applyFiringRecords`
        /// (D-117). The OLD `rejectedMapOrdis` is never probed here
        /// (different key language — the 3X mis-design this pair fixes).
        /// A record of a goal near-miss (`Ordis2DemandRecord::goalNearMiss`)
        /// is no OR demand: it goes, in the same order, to
        /// `IntegrationProcessor::serveGoalNearMiss`, the goal-driven rule
        /// search, and touches neither map
        /// (D-574).
        ///
        /// @param memoryBlock The LB whose `admissionKeysOrdis2` buffer is
        ///        replayed; its `overallHashMemory.admissionMapOrdis2` and
        ///        `rejectedMapOrdis2` receive the writes. The caller
        ///        (`performElemPhase2`) clears the buffer post-drain.
        /// @return (void)
        /// @invariant Drains in append (canonical firing) order;
        ///            single-threaded seam only (I-83).
        /// @see `Ordis2DemandRecord` — the staged record type.
        /// @see `consumeOrLeavesCohort` — route (c), the opener-side reader.
        /// @see `IntegrationProcessor::serveGoalNearMiss` — the near-miss
        ///      reader.
void AdmissionMemory::drainAdmissionKeysOrdis2(Memory& memoryBlock) {
        const unsigned slot = (gl::WorkerContext::coreId() >= 0)
            ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : genScratchArenas().slotCount() - 1;
        ScratchArena& gArena = genScratchArenas().forSlot(slot);

        for (const Ordis2DemandRecord& rec : memoryBlock.admissionKeysOrdis2) {
            // A goal near-miss is no OR demand: its missing expression goes
            // to the goal-driven rule search, which installs the marked-head
            // implications of the rules that can conclude it
            // (D-574).
            if (rec.goalNearMiss) {
                integrationProcessor.serveGoalNearMiss(memoryBlock, StrSpan(rec.expression),
                    StrSpan(rec.validityName));
                continue;
            }
            // FROZEN interner-touch sequence per record (the
            // drainAdmissionKeysAlgebra discipline): (1) mintTemplateKey
            // (templateInterner + NameMap), (2) the RMW blob insert
            // (valueInterner untouched — the value carries rule-interner
            // ids minted at install), (3) the ordis2 wake.
            const ArenaOffset mark = gArena.cursor();
            const int64_t pk = mintTemplateKey(memoryBlock.templateInterner,
                memoryBlock.nameMap, StrSpan(rec.expression),
                StrSpan(rec.validityName));
            insertAdmissionOrdis2IdsBlob(
                memoryBlock.overallHashMemory.admissionMapOrdis2, pk,
                rec.sourceImplId, rec.levels.begin(), rec.levels.size(),
                memoryBlock.ruleInterner, gArena);
            gArena.popTo(mark);

            // The pair's rendezvous (D-267):
            // rejectedMapOrdis2 is keyed by disjunct clean GROUND texts —
            // the demand's own key language — so this keyed probe IS the
            // wake. A parked or filed under a text byte-equal to the
            // demand is un-known and re-deposited; the absorb re-runs the
            // opener, whose route (c) reads the fresh entry above and
            // opens (consuming it). A probe miss (no filed head yet, or a
            // never-parked text) is a defined no-op — the demand entry
            // stands and a later deposit's route-(c) probe covers the
            // demand-first order. The OLD rejectedMapOrdis is deliberately
            // NOT probed here (different key language; a cross-probe would
            // double-wake operator-application disjuncts).
            rejectedSourceReplay.revisitRejectedOrdis2(StrSpan(rec.expression), memoryBlock,
                                        StrSpan(rec.validityName));
        }
    }

/// @brief File one missing expression of the induction hypothesis as a
        ///        ground OR demand at its scope.
        ///
        /// @details
        /// GL sees a missing expression in the induction hypothesis: at `main`
        /// a fully specified premise of the installed hypothesis, in an
        /// induction scope a premise the eats up to that scope left fully
        /// specified. When the premise is not known at the scope or an
        /// ancestor, it is a reason to open an OR one of whose disjuncts it is
        /// (D-575). The reason is written in the
        /// ordis2 demand language: the one record a firing's demand would
        /// stage, replayed like `drainAdmissionKeysOrdis2` replays it —
        /// `mintTemplateKey`, the canonical RMW insert into
        /// `admissionMapOrdis2` with the hypothesis rule as source and no
        /// levels, then the `revisitRejectedOrdis2` wake of an OR already
        /// parked under the key. A later OR arrival meets the demand through
        /// the route-(c) probe of `consumeOrLeavesCohort`, which opens the
        /// cohort with this disjunct as its first case. A premise outside the
        /// park-key language (`ordis2KeyEligible`: compiled, non-atomic, at
        /// least four arguments) can be no disjunct the pair rendezvous meets
        /// and files nothing; a known premise is no missing expression and
        /// files nothing (the D-358 rule of the firing producers).
        ///
        /// @param mb           The step block.
        /// @param element      The fully specified premise, ground text.
        /// @param validityName The scope the hypothesis misses it at (interned).
        /// @param sourceImplId The hypothesis rule (`isInductionHypothesisRule`).
        /// @return (void)
        /// @invariant Single-threaded per block (a worker's claim or the grid
        ///            build); writes `admissionMapOrdis2` and, through the
        ///            wake, the internal mail, nothing else.
        /// @see drainAdmissionKeysOrdis2, CaseProcessor::consumeOrLeavesCohort,
        ///      RejectedSourceReplay::revisitRejectedOrdis2.
void AdmissionMemory::fileHypothesisOrDemand(Memory& mb, StrSpan element, StrSpan validityName,
        int32_t sourceImplId) {
        assert(mb.isInductionHypothesisRule(sourceImplId)
            && "fileHypothesisOrDemand: the demand's source is the block's induction hypothesis");
        assert(!element.empty() && "fileHypothesisOrDemand: an empty premise");
        if (!admissionPolicy.ordis2KeyEligible(element)) return;
        const NameId scopeId = mb.nameMap.lookup(validityName);
        assert(scopeId != 0 && "fileHypothesisOrDemand: the demand scope is interned");
        // Missing: not known at the scope or an ancestor. Non-minting — an
        // uninterned premise is known nowhere.
        const NameId elementId = mb.nameMap.lookup(element);
        if (elementId != 0 && gl::StatementKnowledge::ancestorKnown(mb, elementId, scopeId, true)) return;

        const unsigned genSlot = (gl::WorkerContext::coreId() >= 0)
            ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : genScratchArenas().slotCount() - 1;
        const unsigned stringSlot = (gl::WorkerContext::coreId() >= 0)
            ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : scratchArenas().slotCount() - 1;
        ScratchArena& gArena = genScratchArenas().forSlot(genSlot);
        ScratchArena& strings = scratchArenas().forSlot(stringSlot);
        ScratchScope stringFrame(strings);
        // Owned copies: the key mint below grows the NameMap and the template
        // interner, so a caller's decodeView spans would dangle (I-3).
        const ScratchString text = ScratchString::copyFrom(strings, element.ptr, element.len);
        const ScratchString scope = ScratchString::copyFrom(strings, validityName.ptr, validityName.len);
        // The frozen interner-touch sequence of drainAdmissionKeysOrdis2:
        // (1) the packed demand key, (2) the canonical RMW insert, (3) the wake.
        const ArenaOffset mark = gArena.cursor();
        const int64_t pk = mintTemplateKey(mb.templateInterner, mb.nameMap, StrSpan(text), StrSpan(scope));
        insertAdmissionOrdis2IdsBlob(mb.overallHashMemory.admissionMapOrdis2, pk,
            sourceImplId, nullptr, 0, mb.ruleInterner, gArena);
        gArena.popTo(mark);
        rejectedSourceReplay.revisitRejectedOrdis2(StrSpan(text), mb, StrSpan(scope));
    }

/// @brief File the missing expressions of a freshly installed induction
        ///        hypothesis at `main`.
        ///
        /// @details
        /// At `main` no eat has happened: the hypothesis's missing expressions
        /// are its premises with only fixed arguments, the `u_` names of its
        /// instruction form (D-575). Each one is
        /// written as a ground statement (`groundHypothesisPremisesScratch`)
        /// and offered to @ref fileHypothesisOrDemand at `main`. The step
        /// block is new, so no OR is parked there yet.
        ///
        /// @param mb           The step block, just after the install.
        /// @param premises     The hypothesis premises, instruction form.
        /// @param premiseN     Number of premises.
        /// @param sourceImplId The installed hypothesis rule.
        /// @return (void)
        /// @invariant Called at the grid build, once per installed rule.
        /// @see fileHypothesisOrDemand, ScopeSyntax::groundHypothesisPremisesScratch.
void AdmissionMemory::fileHypothesisDemandsAtMain(Memory& mb, const StrSpan* premises, int32_t premiseN,
        int32_t sourceImplId) {
        const unsigned slot = (gl::WorkerContext::coreId() >= 0)
            ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : scratchArenas().slotCount() - 1;
        ScratchArena& strings = scratchArenas().forSlot(slot);
        ScratchScope frame(strings);
        ScratchString ground[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
        const int32_t groundN = gl::ScopeSyntax::groundHypothesisPremisesScratch(strings,
            premises, premiseN, nullptr, 0, ground, ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS);
        for (int32_t i = 0; i < groundN; ++i)
            fileHypothesisOrDemand(mb, StrSpan(ground[i]), StrSpan("main", 4), sourceImplId);
    }

/// @brief Demand the successor of a forced witness of a strong step block at
        ///        the scope that minted it.
        ///
        /// @details
        /// A forced witness of the strong step block (the induction head's or
        /// a seed's forced disintegration) whose definition set is `(1)` is a
        /// number the induction apparatus works with, such as the length of
        /// the hypothesis's factorization; extending what the hypothesis gave
        /// needs its successor. The successor function is the third argument
        /// of the block's recursion hypothesis `(in2[rec,n,s])`. The demand
        /// `(in2[w,marker,s])` is written through `updateAdmissionMap` at
        /// @p validityName, the scope the witness was minted at, so only
        /// disintegrations at that scope or below it read it
        /// (`isAdmittedIncludingAncestors`). The chain handed over is
        /// `(in2[w,f,s])` with fixed arguments w and s and a free number f:
        /// `updateAdmissionMap` puts the marker in place of the one argument
        /// outside the fixed run, as for every regular chain, and the
        /// canonical renaming at a scope with equivalence classes accepts
        /// only fixed and numbered arguments. The `revisitRejected2` wake then
        /// replays a successor existence already parked under the key
        /// (D-571).
        ///
        /// @param mb           The strong step block.
        /// @param witness      The forced witness's permanent name.
        /// @param validityName The scope the witness was minted at.
        /// @return (void)
        /// @invariant Single-threaded per block (a worker's claim). Asserts the
        ///            strong step block and its in2 recursion hypothesis.
        /// @see updateAdmissionMap, RejectedSourceReplay::revisitRejected2.
void AdmissionMemory::fileForcedSuccessorDemand(Memory& mb, StrSpan witness, StrSpan validityName) {
        assert(mb.inductionDoubleKey != 0
            && "fileForcedSuccessorDemand: only a strong step block demands a forced witness's successor");
        assert(mb.recursionHypothesisId != 0
            && "fileForcedSuccessorDemand: the step block names its recursion hypothesis");
        assert(witness.len > 0 && "fileForcedSuccessorDemand: an empty witness");
        const unsigned slot = (gl::WorkerContext::coreId() >= 0)
            ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : scratchArenas().slotCount() - 1;
        ScratchArena& strings = scratchArenas().forSlot(slot);
        ScratchScope frame(strings);
        // The successor function: the third argument of (in2[rec,n,s]).
        const StrSpan hypothesis = mb.nameMap.decodeView(mb.recursionHypothesisId);
        assert(equalSpans(extractExpressionSpan(hypothesis), StrSpan("in2", 3))
            && "fileForcedSuccessorDemand: the recursion hypothesis is a successor statement");
        StrSpan hypothesisArgs[ExecutionParameters::MAX_ARITY];
        const int32_t hypothesisArgN = getArgsSpans(hypothesis, hypothesisArgs,
            ExecutionParameters::MAX_ARITY);
        assert(hypothesisArgN == 3 && "fileForcedSuccessorDemand: in2 has three arguments");
        // Owned copies: updateAdmissionMap mints the NameMap, which would
        // leave a decodeView span or a caller's span dangling (I-3).
        const ScratchString successor = ScratchString::copyFrom(strings, hypothesisArgs[2].ptr,
            hypothesisArgs[2].len);
        const ScratchString name = ScratchString::copyFrom(strings, witness.ptr, witness.len);
        const ScratchString scope = ScratchString::copyFrom(strings, validityName.ptr, validityName.len);
        const StrSpan w(name);
        const StrSpan s(successor);
        assert(!equalSpans(w, s) && "fileForcedSuccessorDemand: the witness is no successor function");
        // (in2[<witness>,marker,<successor>])
        const int32_t length = 5 + w.len + 8 + s.len + 2;
        char* bytes = strings.allocBytes(length);
        int32_t at = 0;
        std::memcpy(bytes + at, "(in2[", 5);
        at += 5;
        std::memcpy(bytes + at, w.ptr, static_cast<std::size_t>(w.len));
        at += w.len;
        std::memcpy(bytes + at, ",marker,", 8);
        at += 8;
        std::memcpy(bytes + at, s.ptr, static_cast<std::size_t>(s.len));
        at += s.len;
        std::memcpy(bytes + at, "])", 2);
        at += 2;
        assert(at == length && "fileForcedSuccessorDemand: the demand text fills its buffer exactly");
        const ScratchString demand = ScratchString::wrap(strings, bytes, length);
        // The chain (in2[<witness>,<free>,<successor>]): a free number where
        // the demand has its marker; updateAdmissionMap marks it.
        const StrSpan freeArgument = equalSpans(s, StrSpan("1", 1)) ? StrSpan("2", 1) : StrSpan("1", 1);
        assert(!equalSpans(freeArgument, w) && !equalSpans(freeArgument, s)
            && "fileForcedSuccessorDemand: the free argument is neither fixed argument");
        const int32_t premiseLength = 5 + w.len + 1 + freeArgument.len + 1 + s.len + 2;
        char* premiseBytes = strings.allocBytes(premiseLength);
        at = 0;
        std::memcpy(premiseBytes + at, "(in2[", 5);
        at += 5;
        std::memcpy(premiseBytes + at, w.ptr, static_cast<std::size_t>(w.len));
        at += w.len;
        premiseBytes[at++] = ',';
        std::memcpy(premiseBytes + at, freeArgument.ptr, static_cast<std::size_t>(freeArgument.len));
        at += freeArgument.len;
        premiseBytes[at++] = ',';
        std::memcpy(premiseBytes + at, s.ptr, static_cast<std::size_t>(s.len));
        at += s.len;
        std::memcpy(premiseBytes + at, "])", 2);
        at += 2;
        assert(at == premiseLength && "fileForcedSuccessorDemand: the chain text fills its buffer exactly");
        const ScratchString premise = ScratchString::wrap(strings, premiseBytes, premiseLength);
        // The fixed arguments as updateAdmissionMap's sorted, distinct run.
        const bool witnessFirst = compareSpans(w, s) < 0;
        const StrSpan fixed[2] = { witnessFirst ? w : s, witnessFirst ? s : w };
        const StrSpan chain[1] = { StrSpan(premise) };
        // An ordinary value (flag 0) with the recursion status: the successor
        // it admits is a product of recursion, pure for the marker gate, and
        // the value still admits for originals with witness arguments such as
        // the successor existence of a forced witness.
        updateAdmissionMap(mb, chain, 1, fixed, 2, parameters.maxIterationNumberVariable,
            parameters.standardMaxSecondaryNumber, 0, StrSpan(scope), true);
        rejectedSourceReplay.revisitRejected2(StrSpan(demand), mb, StrSpan(scope));
    }

/// @brief Demand, in a base-case block's released successor case, the
        ///        partner's values of the functions the second twin premise
        ///        names.
        ///
        /// @details
        /// A base-case block of a twin conjecture refutes the successor case
        /// l = s(q) of its partner l (D-588).
        /// The refutation needs the value at l of a function the second twin
        /// premise is about, Q's last entry `(in2[l,x,Q])` for the full I5,
        /// whose existence is known at `main` but parked under
        /// `(in2[l,marker,Q])` while nothing demands it. Every premise of the
        /// block that holds l is read (`ancestorPremiseKeyView`); each of its
        /// arguments X that is neither l nor an anchor argument and whose
        /// definition set there is the one `in2` gives its function slot
        /// (`ScopeTypeResolver::definitionSetOf`) is demanded at
        /// @p scopeId: `updateAdmissionMap` with the chain `(in2[l,f,X])`,
        /// fixed arguments l and X and a free number f, as
        /// `fileForcedSuccessorDemand` writes its successor demand, then the
        /// `revisitRejected2` wake at that scope, which replays a source
        /// parked at an ancestor into it. The block records that it filed
        /// (`Memory::twinValueDemandsFiled`).
        ///
        /// @param mb      The base-case block (single-threaded seam).
        /// @param scopeId The released successor case (`Memory::twinCaseScopeId`,
        ///                asserted).
        /// @return (void)
        /// @invariant Filed once per block (asserted); a premise of the block
        ///            holds the partner (asserted).
        /// @see fileForcedSuccessorDemand, updateAdmissionMap,
        ///      RejectedSourceReplay::revisitRejected2,
        ///      StatementMemory::addStatement.
void AdmissionMemory::fileTwinValueDemands(Memory& mb, NameId scopeId) {
        assert(mb.twinExperiment == TwinExperiment::successorCase && mb.twinPartnerId != 0
            && !mb.twinValueDemandsFiled
            && "fileTwinValueDemands: a base-case block files its partner's value demands once");
        assert(mb.twinCaseScopeId != 0 && scopeId == mb.twinCaseScopeId
            && "fileTwinValueDemands: the demands are filed in the released successor case");
        mb.twinValueDemandsFiled = true;
        const unsigned slot = (gl::WorkerContext::coreId() >= 0)
            ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : scratchArenas().slotCount() - 1;
        ScratchArena& strings = scratchArenas().forSlot(slot);
        ScratchScope frame(strings);
        // Owned copies: updateAdmissionMap mints the NameMap (I-3). The
        // premise and anchor views read the skeleton interner, which no
        // prover phase mints into (I-97).
        const StrSpan scopeView = mb.nameMap.decodeView(scopeId);
        const ScratchString scope = ScratchString::copyFrom(strings, scopeView.ptr, scopeView.len);
        const StrSpan partnerView = mb.nameMap.decodeView(mb.twinPartnerId);
        const ScratchString partnerName = ScratchString::copyFrom(strings, partnerView.ptr, partnerView.len);
        const StrSpan partner(partnerName);
        StrSpan anchors[ExecutionParameters::MAX_ARITY];
        const int32_t anchorN = gl::ScopeSyntax::anchorArgumentSpans(mb, StrSpan(anchorInfo.name), anchors,
            ExecutionParameters::MAX_ARITY);
        int32_t holders = 0;
        for (int32_t level = 1; level < mb.level; ++level) {
            const StrSpan premise = ancestorPremiseKeyView(mb, level);
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argN = getArgsSpans(premise, args, ExecutionParameters::MAX_ARITY);
            bool holdsPartner = false;
            for (int32_t a = 0; a < argN; ++a) holdsPartner = holdsPartner || equalSpans(args[a], partner);
            if (!holdsPartner) continue;
            ++holders;
            for (int32_t a = 0; a < argN; ++a) {
                const StrSpan x = args[a];
                if (equalSpans(x, partner)) continue;
                bool skip = false;
                for (int32_t b = 0; b < a && !skip; ++b) skip = equalSpans(args[b], x);
                for (int32_t k = 0; k < anchorN && !skip; ++k) skip = equalSpans(anchors[k], x);
                if (skip) continue;
                ScratchScope demandFrame(strings);
                // The chain (in2[l,f,X]): a free number where the demand has
                // its marker; updateAdmissionMap marks it.
                const StrSpan freeArgument = (equalSpans(x, StrSpan("1", 1)) || equalSpans(partner, StrSpan("1", 1)))
                    ? ((equalSpans(x, StrSpan("2", 1)) || equalSpans(partner, StrSpan("2", 1)))
                        ? StrSpan("3", 1) : StrSpan("2", 1))
                    : StrSpan("1", 1);
                assert(!equalSpans(freeArgument, x) && !equalSpans(freeArgument, partner)
                    && "fileTwinValueDemands: the free argument is neither fixed argument");
                const int32_t premiseLength = 5 + partner.len + 1 + freeArgument.len + 1 + x.len + 2;
                char* premiseBytes = strings.allocBytes(premiseLength);
                int32_t at = 0;
                std::memcpy(premiseBytes + at, "(in2[", 5); at += 5;
                std::memcpy(premiseBytes + at, partner.ptr, static_cast<std::size_t>(partner.len)); at += partner.len;
                premiseBytes[at++] = ',';
                std::memcpy(premiseBytes + at, freeArgument.ptr, static_cast<std::size_t>(freeArgument.len));
                at += freeArgument.len;
                premiseBytes[at++] = ',';
                std::memcpy(premiseBytes + at, x.ptr, static_cast<std::size_t>(x.len)); at += x.len;
                std::memcpy(premiseBytes + at, "])", 2); at += 2;
                assert(at == premiseLength && "fileTwinValueDemands: the chain text fills its buffer exactly");
                const ScratchString chainElement = ScratchString::wrap(strings, premiseBytes, premiseLength);
                // Only a function of the kind `in2` applies: X's definition
                // set in the premise is that of in2's function slot.
                if (!equalSpans(scopeTypes.definitionSetOf(premise, x),
                                scopeTypes.definitionSetOf(StrSpan(chainElement), x))) continue;
                // (in2[<partner>,marker,<X>])
                const int32_t demandLength = 5 + partner.len + 8 + x.len + 2;
                char* demandBytes = strings.allocBytes(demandLength);
                at = 0;
                std::memcpy(demandBytes + at, "(in2[", 5); at += 5;
                std::memcpy(demandBytes + at, partner.ptr, static_cast<std::size_t>(partner.len)); at += partner.len;
                std::memcpy(demandBytes + at, ",marker,", 8); at += 8;
                std::memcpy(demandBytes + at, x.ptr, static_cast<std::size_t>(x.len)); at += x.len;
                std::memcpy(demandBytes + at, "])", 2); at += 2;
                assert(at == demandLength && "fileTwinValueDemands: the demand text fills its buffer exactly");
                const ScratchString demand = ScratchString::wrap(strings, demandBytes, demandLength);
                // The fixed arguments as updateAdmissionMap's sorted, distinct run.
                const bool partnerFirst = compareSpans(partner, x) < 0;
                const StrSpan fixed[2] = { partnerFirst ? partner : x, partnerFirst ? x : partner };
                const StrSpan chain[1] = { StrSpan(chainElement) };
                updateAdmissionMap(mb, chain, 1, fixed, 2, parameters.maxIterationNumberVariable,
                    parameters.standardMaxSecondaryNumber, 0, StrSpan(scope));
                rejectedSourceReplay.revisitRejected2(StrSpan(demand), mb, StrSpan(scope));
            }
        }
        assert(holders > 0 && "fileTwinValueDemands: a premise of the block holds the partner");
    }

/// @brief Whether every argument of a premise is one of the fixed arguments.
        ///
        /// @details
        /// An admission chain stores its fixed arguments as a sorted,
        /// duplicate-free run (`remainingArgs`); a premise whose arguments all
        /// lie in it names no variable any more — the fully bound premise
        /// `updateAdmissionMap` prepares for integration and the hypothesis may
        /// miss (D-575).
        ///
        /// @param element     The premise, `(...)` or `!(...)`.
        /// @param fixedSorted The fixed arguments, `compareSpans`-ascending
        ///                    and unique.
        /// @param fixedN      Number of fixed arguments.
        /// @return `true` iff each argument of @p element is in the run.
        /// @invariant Pure; allocates nothing.
        /// @see updateAdmissionMap, fileHypothesisOrDemand.
bool AdmissionMemory::fullyBoundElement(StrSpan element, const StrSpan* fixedSorted, int32_t fixedN) {
        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int32_t argN = getArgsSpans(element, args, ExecutionParameters::MAX_ARITY);
        for (int32_t a = 0; a < argN; ++a) {
            if (!std::binary_search(fixedSorted, fixedSorted + fixedN, args[a],
                    [](const StrSpan& x, const StrSpan& y) { return compareSpans(x, y) < 0; }))
                return false;
        }
        return true;
    }



/// @brief Element-wise byte-lexicographic compare of two
        ///        `SealedSpan<SealedString>` runs — the `std::vector`/`std::set`
        ///        `operator<` twin over sealed strings.
        ///
        /// @details
        /// Three-way (`<0` / `0` / `>0`) comparison identical to
        /// `std::lexicographical_compare` over the decoded `std::string`
        /// elements: element `k` compared by `compareSpans` (byte-lex, ==
        /// `std::string::compare`), shorter-run-is-prefix on a common prefix
        /// (`std::vector`/`std::set` `operator<` semantics). One helper serves
        /// both the `key` (vector) and `remainingArgs` (set) runs — both are
        /// element-wise lex.
        ///
        /// @param a Left run.
        /// @param b Right run.
        /// @return Negative when `a < b`, 0 when equal, positive when `a > b`.
        /// @see deferredAncestorAdmissionLess.
int AdmissionMemory::sealedStringRunLexCompare(
    const SealedSpan<SealedString>& a, const SealedSpan<SealedString>& b) {
    const int32_t na = a.size();
    const int32_t nb = b.size();
    const int32_t common = na < nb ? na : nb;
    for (int32_t k = 0; k < common; ++k) {
        const int c = compareSpans(StrSpan(a[k]), StrSpan(b[k]));
        if (c != 0) return c;
    }
    return na < nb ? -1 : (na > nb ? 1 : 0);
}

/// @brief Deterministic replay order of two deferred ancestor admission
        ///        records — the byte-twin of the retired drain sort's 6-field
        ///        chain.
        ///
        /// @details
        /// Three-way compare returning at the first non-zero field, in order:
        ///  1. `ancestor->exprKeyView()` (byte-lex; `exprKeyView` bytes ==
        ///     `exprKey()`, skeletonInterner never deloads). exprKey is NOT
        ///     unique across recursion sub-LBs, so the chain continues.
        ///  2. `key` run (`sealedStringRunLexCompare`; the `std::vector<std::string>`
        ///     twin).
        ///  3. `remainingArgs` run (`sealedStringRunLexCompare`; the
        ///     `std::set<std::string>` twin).
        ///  4. `partOfRecursion` (false < true).
        ///  5. `maxAdmissionDepth`.
        ///  6. `maxSecondaryNumber`.
        /// Two fully-equal records tie (returns 0) → `std::sort` leaves them in
        /// input (append) order, exactly as the retired code did (identical
        /// records replay to the same admission state; not a regression).
        ///
        /// @param a Left record.
        /// @param b Right record.
        /// @return Negative when `a` sorts before `b`, 0 when equal, positive
        ///         otherwise.
        /// @invariant Byte-identical replay order to the retired heap sort
        ///            (I-107).
        /// @see drainDeferredAncestorAdmissions, sealedStringRunLexCompare,
        ///      Memory::exprKeyView.
int AdmissionMemory::deferredAncestorAdmissionLess(
    const DeferredAncestorAdmission& a, const DeferredAncestorAdmission& b) const {
    const int c0 = compareSpans(a.ancestor->exprKeyView(), b.ancestor->exprKeyView());
    if (c0 != 0) return c0;
    const int c1 = sealedStringRunLexCompare(a.key, b.key);
    if (c1 != 0) return c1;
    const int c2 = sealedStringRunLexCompare(a.remainingArgs, b.remainingArgs);
    if (c2 != 0) return c2;
    if (a.partOfRecursion != b.partOfRecursion)
        return a.partOfRecursion ? 1 : -1;   // false < true
    if (a.maxAdmissionDepth != b.maxAdmissionDepth)
        return a.maxAdmissionDepth < b.maxAdmissionDepth ? -1 : 1;
    if (a.maxSecondaryNumber != b.maxSecondaryNumber)
        return a.maxSecondaryNumber < b.maxSecondaryNumber ? -1 : 1;
    return 0;
}

/// @brief Replay every deferred cross-LB admission seed single-threaded, in
        ///        `proveKernel`'s post-join drain (I-28 detect-and-defer trial).
        ///
        /// @details
        /// `updateAdmissionMap3`, reached from a status-2 goal registration inside a
        /// parallel phase-1/phase-3 worker, may climb `parentMemory` to a strict
        /// ancestor (the fold/recursion block that owns the goal's numeral) and seed
        /// that ancestor's admission map. Writing the ancestor from the child's
        /// worker is a cross-LB write forbidden during the parallel phase
        /// ([I-28](../../docs/agentic_swdd/30_invariants.md#i-28)): sibling recursion
        /// children sharing one fold ancestor would mint its cold maps concurrently.
        /// Instead each worker seals the resolved write's strings and
        /// `appendRecord`s it on the shared `deferredAncestorPages`
        /// `SealedPageSet` (under `deferredAncestorAdmissionsMutex`); this drain,
        /// run once after `pool.join()` alongside `activateZeroCondition` /
        /// `updateGlobal`, replays them.
        ///
        /// Replay order is made deterministic by an index sort under
        /// `deferredAncestorAdmissionLess` (the retired 6-field chain: exprKey,
        /// key, remainingArgs, partOfRecursion, maxAdmissionDepth,
        /// maxSecondaryNumber) before applying — grant/timing order must not leak
        /// into proof output (I-107). A `dischargedForever` ancestor is skipped
        /// (a permanently-deactivated LB has no next step and must not be
        /// resurrected — I-102 / I-112); a still-live ancestor is `ensureLoaded` (the
        /// working-set pager may have evicted it during the parallel phase) and then
        /// receives the inline `updateAdmissionMap` call (this drain runs on the main
        /// thread, where `g_inParallelWorkerPhase` is false, so the recursive
        /// re-entry applies inline rather than re-deferring). The page set is
        /// sealed, `freePages`'d, and the `std::optional` reset after the drain.
        ///
        /// @return (void)
        /// @invariant [I-28](../../docs/agentic_swdd/30_invariants.md#i-28) — the
        ///            single-threaded post-join drain is the sole writer of a
        ///            deferred ancestor admission.
        /// @see `updateAdmissionMap3` — the producer; `g_inParallelWorkerPhase` — the
        ///      detect gate; `deferredAncestorAdmissionLess` — the replay order.
void AdmissionMemory::drainDeferredAncestorAdmissions() {
    // proveKernel emplaces the page set before phase-1 and this is its sole,
    // unconditionally-reached post-join call site — it is always engaged here.
    assert(deferredAncestorPages
        && "drainDeferredAncestorAdmissions: proveKernel must emplace the page set");
    SealedPageSet& ps = *deferredAncestorPages;
    if (ps.recordCount() == 0) { deferredAncestorPages.reset(); return; }

    // The drain runs single-threaded post-join (gl::WorkerContext::coreId() == -1 -> the
    // reserved last gen slot). Gather record pointers, then an index sort on the
    // deterministic 6-field chain (I-107). The refs / idx ride the gen-scratch
    // byte-bump + page tiers; the sealed strings ride deferredAncestorPages
    // (never crossed, I-124).
    const unsigned slot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(slot);
    const ArenaOffset mark = gArena.cursor();
    DirtyState d = DirtyState::Clean;
    PagedVector<const DeferredAncestorAdmission*> refs(&gArena, &d);
    ps.forEachRecord<DeferredAncestorAdmission>(
        [&](const DeferredAncestorAdmission& r) { refs.push_back(&r); });

    const int32_t n = static_cast<int32_t>(refs.size());
    int32_t* idx = reinterpret_cast<int32_t*>(
        gArena.resolve(gArena.alloc(n * sizeof(int32_t), alignof(int32_t))));
    for (int32_t i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx, idx + n, [&](int32_t a, int32_t b) {
        return deferredAncestorAdmissionLess(*refs[static_cast<size_t>(a)],
                                             *refs[static_cast<size_t>(b)]) < 0;
    });

    for (int32_t oi = 0; oi < n; ++oi) {
        const DeferredAncestorAdmission& r = *refs[static_cast<size_t>(idx[oi])];
        // A permanently-deactivated ancestor has no next step to consume the
        // seed and must not be resurrected (I-102 / I-112) — skip, mirroring the
        // dischargedForever guard on the proven-head deposit drain.
        if (r.ancestor->dischargedForever) continue;
        // SEAM DOOR (D-196): the pager (a phase window or
        // the barrier seam window) may have evicted the ancestor — the
        // uniform handshake reloads it claim-correctly and holds WorkerOwned
        // across the admission write below; released after.
        steward->claimAndLoadForWork(*r.ancestor, /*phase=*/4,
                                     lbdeload::kDeloadDirectory);
        // SealedSpan<SealedString> -> StrSpan[] stack runs for the C2 signature.
        // The sealed bytes stay valid until freePages() below (updateAdmissionMap
        // mints the ANCESTOR's interners, never this page set; I-3).
        StrSpan keyRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
        const int32_t kN = r.key.size();
        assert(kN <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS);
        for (int32_t i = 0; i < kN; ++i) keyRun[i] = StrSpan(r.key[i]);
        StrSpan remRun[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
        const int32_t rN = r.remainingArgs.size();
        assert(rN <= ExecutionParameters::MAX_ADMISSION_REM_ARGS);
        for (int32_t i = 0; i < rN; ++i) remRun[i] = StrSpan(r.remainingArgs[i]);
        this->updateAdmissionMap(*r.ancestor, keyRun, kN, remRun, rN,
            r.maxAdmissionDepth, r.maxSecondaryNumber, r.partOfRecursion,
            StrSpan("main", 4));
        // WAKE DOOR 3 (D-194): this replays a cross-LB
        // admission seed into the ancestor's admission map (rule state, not a
        // mailbox) — it can enable a fresh firing next burst, so keep the
        // ancestor swept.
        r.ancestor->hasWork = true;
        r.ancestor->stewardClaim.store(
            static_cast<uint8_t>(Memory::StewardClaim::Idle),
            std::memory_order_release);
    }
    gArena.popTo(mark);
    ps.seal();
    ps.freePages();
    deferredAncestorPages.reset();
}

/// @brief Propagate a newly added statement through the
        ///        recursion-owned admission chains that demand it.
        ///
        /// @details
        /// Called at the deposit door (`addExprToMemoryBlock`) for every added
        /// statement row (D-413).
        /// Participants: an operator (either polarity), which binds only its
        /// output, and a positive config-backed compound that is not a
        /// two-input relation, which binds any argument; relations, atomic
        /// cores and config-less compounds return at once. The door marks the
        /// operator's output, or each distinct argument of the compound, and
        /// matches the marked form against the `admissionMap` templates. Only
        /// a template whose `admissionStatusMap` status is 1 (recursion-owned)
        /// propagates: each chain element whose bindable free argument marks
        /// to the same form is re-keyed with the statement's argument. In the
        /// re-keyed chain, an operator with only its output free and a compound
        /// with exactly one free argument gain a derived admission entry (that
        /// argument marked) with the source status; for a positive element the
        /// marker form's integration is prepared as well
        /// (`prepareIntegration`), so the chain's missing argument is
        /// constructed, not only admitted. An operator's unmatched output
        /// probe leaves the default-false status entry of the historical probe
        /// (D-172); an unmatched compound probe leaves nothing.
        ///
        /// A recursion-owned template propagates only when @p inductionAllowed
        /// holds. The door clears it for a product of a disintegration whose
        /// original expression has an argument that is not a number
        /// (`hasOnlyNumberArguments`); the probes and the default-false status
        /// entries above are unaffected
        /// (I-285).
        ///
        /// An in-place feed of the induction hypothesis's own chain (a value with the
        /// induction-hypothesis flag) advances the chain in place at the
        /// statement's scope like any other chain. In an unsettled tree it is also
        /// an eat like a witness admission: under `isAdmitted`'s eat gates (the
        /// chain filed at the scope's eating position, the slot open at the scope,
        /// no backward readmission) it records the eaten slot at the statement's
        /// scope, opens the induction scope
        /// `<scope>_boundary_is_<chain with marker>_<name>` named by the existing
        /// name it fills, and files a copy of the advanced chain's demands there
        /// with their parked-source wakes. The integration preparations stay at
        /// the statement's scope, whose constructions the new scope sees. A
        /// refused eat opens nothing (D-520, D-546).
        ///
        /// @param expression   The added statement (span over a caller-stable
        ///                     buffer; copied on entry).
        /// @param mb           The owning LB.
        /// @param validityName The statement's scope (copied on entry).
        /// @param inductionAllowed Whether a recursion-owned template may
        ///                     propagate this statement; forwarded unchanged
        ///                     to the per-match call.
        /// @param selectedKey  0 on the door call; a matched template's packed
        ///                     key on the per-match recursive call.
        /// @param boundSlot    -1 on the door call; on the per-match call, the
        ///                     slot of the statement argument the matched
        ///                     template marks.
        /// @invariant Heap-free; mutates `admissionMap` inside its value loop,
        ///            so the loop walks a verbatim snapshot
        ///            (`snapshotAdmissionRun`).
        /// @see updateAdmissionMap (the install-time marker branch),
        ///      prepareIntegration, revisitRejected2, revisitRejectedOrdis.
void AdmissionMemory::updateAdmissionMapRecursion(StrSpan expression,
    Memory& mb,
    StrSpan validityName, bool inductionAllowed, int64_t selectedKey, int32_t boundSlot) {
    assert((selectedKey == 0) == (boundSlot < 0)
        && "updateAdmissionMapRecursion: a matched template carries its marked argument slot");
    const unsigned canonicalSlot = gl::WorkerContext::coreId() >= 0 ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : scratchArenas().slotCount() - 1;
    ScratchArena& canonicalStrings = scratchArenas().forSlot(canonicalSlot);
    ScratchScope canonicalFrame(canonicalStrings);
    const ScratchString ownedValidity = ScratchString::copyFrom(canonicalStrings,validityName.ptr,validityName.len);
    validityName=StrSpan(ownedValidity);
    const NameId canonicalScope=mb.nameMap.lookup(validityName);
    assert(canonicalScope!=0);
    const ScratchString originalInput = ScratchString::copyFrom(canonicalStrings, expression.ptr, expression.len);
    expression = StrSpan(originalInput);
    // 1. Recursion participants (D-413):
    // an operator (either polarity) binds only its output; a positive
    // config-backed compound that is not a relation binds any argument.
    // Relations (two configured inputs, no output: preorder, strictOrder),
    // atomic cores (`in`, `=`), config-less compounds and negated compounds
    // are defined non-participants. The core name is a StrSpan slice
    // (extractExpressionSpan is the byte-exact twin of ce::extractExpression)
    // keying `operators` (std::set<std::string, std::less<>>, transparent)
    // and the compiled-definition layer through the coreConfig /
    // compiledEntity reader fences, so no std::string is materialized
    // (I-137).
    enum class Participant { none, operatorElement, compoundElement };
    const auto participantOf = [this](StrSpan element) -> Participant {
        const StrSpan core = extractExpressionSpan(element);
        if (definitions.operatorNames().find(std::string_view(core.ptr,
                static_cast<std::size_t>(core.len))) != definitions.operatorNames().end())
            return Participant::operatorElement;
        if (element.len == 0 || element.ptr[0] == '!') return Participant::none;
        const LogicalEntity* entity = definitions.compiledEntity(core);
        if (entity == nullptr || equalSpans(StrSpan(entity->category), StrSpan("atomic", 6)))
            return Participant::none;
        const ce::CoreExpressionConfig* cfg = definitions.coreConfig(core);
        if (cfg == nullptr) return Participant::none;
        if (cfg->inputIndices.size() == 2 && cfg->outputIndices.empty()) return Participant::none;
        return Participant::compoundElement;
    };
    const auto outputSlotOf = [this](StrSpan element) -> int32_t {
        const ce::CoreExpressionConfig* cfg = definitions.coreConfig(extractExpressionSpan(element));
        assert(cfg && "updateAdmissionMapRecursion: operator core must be compiled");
        assert(!cfg->outputIndices.empty());
        return cfg->outputIndices[0];
    };
    const Participant arriving = participantOf(expression);
    if (arriving == Participant::none) {
        return;
    }

    // 2. The statement arg-parse rides the span twin (getArgsSpans ==
    // ce::getArgs, byte-identical); every argument is a StrSpan slice of
    // `expression` (a caller-stable buffer disjoint from the value / template
    // / NameMap interners minted below, so it survives every mint — I-3). The
    // bound argument feeds the replacementMap substitution + newRemainingArgs
    // membership/encode as spans, no heap std::string (row 41/50,
    // I-138 L9 Item 2).
    StrSpan argSpans[ExecutionParameters::MAX_ARITY];
    const int32_t argN = getArgsSpans(expression, argSpans,
                                      ExecutionParameters::MAX_ARITY);

    // The marker forms ride the per-slot BYTE-scratch arena, disjoint from the
    // template / NameMap / value interners minted below; nested callee scratch
    // scopes rewind LIFO above this function's scopes.
    const unsigned strSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : scratchArenas().slotCount() - 1;
    ScratchArena& strArena = scratchArenas().forSlot(strSlot);
    ScratchScope markedScope(strArena);

    // 3. The door: an operator statement is tried with its output marked; a
    // compound statement with every distinct argument marked, since its
    // demand marks the one argument its premise lacks, wherever it stands.
    // Each matching template is walked by the per-match call with that
    // argument's slot. The historical EWV map created a default-false status
    // entry on the operator's output probe and the dump's status section
    // counts it — the cold map reproduces that byte-for-byte: an unmatched
    // output probe mints the template and inserts the default-false entry
    // (D-172). A compound probe is a pure lookup.
    if (selectedKey == 0) {
        const bool arrivingOperator = arriving == Participant::operatorElement;
        const int32_t outputSlot = arrivingOperator ? outputSlotOf(expression) : -1;
        assert(!arrivingOperator || (outputSlot >= 0 && outputSlot < argN));
        const unsigned slot = gl::WorkerContext::coreId() >= 0 ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : genScratchArenas().slotCount() - 1;
        ScratchArena& arena=genScratchArenas().forSlot(slot);
        for (int32_t probe = 0; probe < argN; ++probe) {
            if (arrivingOperator && probe != outputSlot) continue;
            bool repeated = false;
            for (int32_t prior = 0; prior < probe && !repeated; ++prior)
                repeated = equalSpans(argSpans[prior], argSpans[probe]);
            if (repeated && !arrivingOperator) continue;
            ScratchScope probeStrings(strArena);
            StrReplacement probePair[1];
            probePair[0].key = argSpans[probe];
            probePair[0].value = StrSpan("marker", 6);
            const ScratchString probeOriginal = replaceKeysScratch(strArena, expression, probePair, 1);
            const ScratchString probeMarked = canonicalTemplates.canonicalAdmissionText(mb, StrSpan(probeOriginal), canonicalScope, strArena);
            ScratchScope frame(arena);
            DirtyState dirty=DirtyState::Clean;
            PagedVector<int64_t> matches(&arena,&dirty);
            canonicalTemplates.matchingCanonicalTemplates(mb,mb.overallHashMemory.admissionMap,mb.lbMemory.canonicalUses.algebraMatches,
                StrSpan(probeMarked),canonicalScope,canonicalScope,matches,arena);
            for (int32_t i=0;i<matches.size();++i)
                updateAdmissionMapRecursion(expression,mb,validityName,inductionAllowed,matches[i],probe);
            if (matches.empty() && arrivingOperator) {
                const int64_t missPk=mintTemplateKey(mb.templateInterner,mb.nameMap,StrSpan(probeMarked),validityName);
                if (mb.overallHashMemory.admissionStatusMap.find(missPk) == nullptr) {
                    mb.overallHashMemory.admissionStatusMap.upsert(missPk, static_cast<uint8_t>(0));
                }
            }
        }
        return;
    }

    // 4. The per-match walk: `selectedKey` is a template the door matched with
    // argument `boundSlot` marked. The marker-form template key rides the
    // per-slot BYTE-scratch arena — replaceKeysScratch is the byte-exact twin
    // of ce::replaceKeysInString (greedy-longest, token-boundary). The
    // ScratchScope spans to function end because the convergence compare
    // (testExpr vs markedExpr) reads it deep in the value loop; the arena is
    // disjoint from the template / NameMap / value interners minted below,
    // and nested callee scratch scopes rewind LIFO above it, so markedExpr
    // stays valid to its last use.
    assert(boundSlot < argN);
    const StrSpan rawBoundArg = argSpans[boundSlot];
    const StrSpan boundArg = StrSpan(canonicalTemplates.canonicalAdmissionText(mb, rawBoundArg, canonicalScope, canonicalStrings));
    StrReplacement markerPair[1];
    markerPair[0].key = rawBoundArg;
    markerPair[0].value = StrSpan("marker", 6);
    const ScratchString originalMarked = replaceKeysScratch(strArena, expression, markerPair, 1);
    const ScratchString markedExpr = canonicalTemplates.canonicalAdmissionText(mb, StrSpan(originalMarked), canonicalScope, strArena);
    const int64_t markedPk=selectedKey;

    {
        const uint8_t* stProbe =
            mb.overallHashMemory.admissionStatusMap.find(markedPk);
        if (stProbe == nullptr) {
            mb.overallHashMemory.admissionStatusMap.upsert(
                markedPk, static_cast<uint8_t>(0));
            return;
        }
        if (*stProbe == 0) {
            return;
        }
        // A recursion-owned template serves a disintegration product only
        // when the disintegrated original has numbers as its only arguments
        // (I-285).
        if (!inductionAllowed) {
            return;
        }
        // Nor does it accept a construction of the induction hypothesis'
        // own head: no derived demand, enrollment, preparation or revisit
        // (I-286).
        if (mb.recursionHeadWitnesses.lookup(rawBoundArg) != 0) {
            return;
        }
        // Nor does a compound element bind a construction over constructed
        // objects (I-292); an operator binds
        // only its output, which stays accepted.
        if (arriving == Participant::compoundElement
            && mb.hiddenArgumentWitnesses.lookup(rawBoundArg) != 0) {
            return;
        }
    }

    // 4. Lookup in Admission Map
    const int32_t admPropId = mb.overallHashMemory.admissionMap.lookup(markedPk);


    // Guard: Return if not found
    if (admPropId == 0) {
        return;
    }

    // 5. Iterate over Admission Values (Snapshot). The cold run is kept sorted
    // by DecodedAdmissionValueLess, so the snapshot order matches the former
    // heap set. Copy the run's blobs verbatim onto gen-scratch BEFORE the loop:
    // insertAdmissionIdsBlob below mutates admissionMap mid-loop, so a live peek
    // would dangle; the blob copies survive it (snapshotAdmissionRun, 09b
    // pitfall 6, I-99). admissionRecordsAt stays the retained oracle.
    const unsigned admSnapSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& admSnapArena = genScratchArenas().forSlot(admSnapSlot);
    const ArenaOffset admSnapMark = admSnapArena.cursor();
    const AdmissionRunSnapshot admPropSnap = snapshotAdmissionRun(
        mb.overallHashMemory.admissionMap, markedPk, admSnapArena);

    // The per-value and per-match gen-tier index runs share admSnapArena's
    // per-slot genScratchArena instance: they sit ABOVE the snapshot on the
    // byte-bump tier, so the LIFO scopes reclaim them without disturbing it.
    ScratchArena& genArena = admSnapArena;

    for (int32_t admValIx = 0; admValIx < admPropSnap.count; ++admValIx) {
        const AdmissionValueBlobView& val = admPropSnap.views[admValIx];
        // Ordis-only values never seed recursion propagation — they are
        // cohort-opening demand evidence, invisible to general admission;
        // the derived keys this walk inserts stay untagged by construction.
        if (val.ordisByte() != 0) {
            continue;
        }
        // The value loop MINTS valueInterner (encode + insertAdmissionIdsBlob
        // grow its cold pages), so the value's decoded key elements AND its
        // remaining-arg members are COPIED ONCE onto the string arena and held
        // for the whole match walk — a raw decodeView span would dangle at the
        // first later mint from an EARLIER matching keyElem's insert (I-3; the
        // S9 keyVec/remVec precedent). The span index arrays ride the gen tier
        // ABOVE the admission snapshot; valStrScope/valGenScope reclaim them per
        // value.
        ScratchScope valStrScope(strArena);
        ScratchScope valGenScope(genArena);

        const int32_t valKeyN = val.keyCount();
        StrSpan* valKeySpans = (valKeyN == 0) ? nullptr
            : reinterpret_cast<StrSpan*>(genArena.resolve(genArena.alloc(
                  valKeyN * static_cast<int32_t>(sizeof(StrSpan)),
                  static_cast<int32_t>(alignof(StrSpan)))));
        for (int32_t e = 0; e < valKeyN; ++e) {
            const StrSpan dv = mb.valueInterner.decodeView(val.keyId(e));
            valKeySpans[e] = StrSpan(ScratchString::copyFrom(strArena,dv.ptr,dv.len));
        }

        // valRemaining: the former std::set<std::string> becomes a sorted-unique
        // StrSpan membership run (members copyFrom on the string arena, index on
        // the gen tier). std::set == sorted-unique, so sort + adjacent-dedup by
        // compareSpans reproduces the set content AND order EXACTLY; the two
        // membership probes become std::binary_search(compareSpans).
        const int32_t valRemRawN = val.remCount();
        StrSpan* valRemaining = (valRemRawN == 0) ? nullptr
            : reinterpret_cast<StrSpan*>(genArena.resolve(genArena.alloc(
                  valRemRawN * static_cast<int32_t>(sizeof(StrSpan)),
                  static_cast<int32_t>(alignof(StrSpan)))));
        for (int32_t r = 0; r < valRemRawN; ++r) {
            const StrSpan dv = mb.valueInterner.decodeView(val.remId(r));
            valRemaining[r] = StrSpan(ScratchString::copyFrom(strArena,dv.ptr,dv.len));
        }
        bool excludedAnchor = false;
        const int32_t valRemN = canonicalTemplates.canonicalAdmissionChain(mb, valKeySpans, valKeyN,
            valRemaining, valRemRawN, canonicalScope, strArena, valKeySpans, valRemaining, excludedAnchor);


        for (int32_t keIx = 0; keIx < valKeyN; ++keIx) {
            const StrSpan keyElem = valKeySpans[keIx];

            // --- Flattened Logic Start ---

            // Skip non-participants (the participantOf classes above).
            const Participant kRole = participantOf(keyElem);
            if (kRole == Participant::none) {
                continue;
            }

            StrSpan kArgs[ExecutionParameters::MAX_ARITY];
            const int32_t kArgsN = getArgsSpans(keyElem, kArgs,
                                                ExecutionParameters::MAX_ARITY);
            const int32_t kOutputSlot = kRole == Participant::operatorElement ? outputSlotOf(keyElem) : -1;
            assert(kRole != Participant::operatorElement || (kOutputSlot >= 0 && kOutputSlot < kArgsN));

            // Per-keyElem string window: the testExpr candidates and (on a
            // match) the newKey rewrites + newMarkedExpr live here, freed at
            // keyElem end.
            ScratchScope keScope(strArena);

            // Check for Match: the element's free argument (not in remaining
            // args — binary_search over the sorted-unique run == the former
            // std::set membership; an operator's output only) whose marker
            // form is the statement's marked form. Single-key marker map ->
            // StrReplacement[1]; replaceKeysScratch is the byte-exact twin of
            // ce::replaceKeysInString (greedy-longest, token-boundary), so each
            // testExpr is byte-identical. kBoundArg is a slice of keyElem (a
            // stable string-arena copy).
            bool matched = false;
            StrSpan kBoundArg;
            for (int32_t a = 0; a < kArgsN && !matched; ++a) {
                if (kRole == Participant::operatorElement && a != kOutputSlot) continue;
                bool repeated = false;
                for (int32_t prior = 0; prior < a && !repeated; ++prior)
                    repeated = equalSpans(kArgs[prior], kArgs[a]);
                if (repeated && kRole != Participant::operatorElement) continue;
                if (std::binary_search(valRemaining, valRemaining + valRemN, kArgs[a],
                        [](const StrSpan& x, const StrSpan& y) {
                            return compareSpans(x, y) < 0;
                        })) {
                    continue;
                }
                ScratchScope testScope(strArena);
                StrReplacement kMarkerPair[1];
                kMarkerPair[0].key = kArgs[a];
                kMarkerPair[0].value = StrSpan("marker", 6);
                const ScratchString testExpr =
                    replaceKeysScratch(strArena, keyElem, kMarkerPair, 1);
                if (equalSpans(StrSpan(testExpr), StrSpan(markedExpr))) {
                    matched = true;
                    kBoundArg = kArgs[a];
                }
            }

            if (!matched) {
                continue;
            }

            // --- Match Found ---

            // The chain advances in place at the statement's scope. An in-place
            // feed of the induction hypothesis's chain in an unsettled tree is
            // also an eat: under isAdmitted's eat gates it records the eaten
            // slot here and opens its own induction scope, named by the
            // existing name it fills, where a copy of the advanced chain's
            // demands is filed with their wakes. The integrations stay
            // prepared here; the scope sees their constructions
            // (D-520, D-546).
            StrSpan demandScopes[2] = { validityName, StrSpan() };
            int32_t demandScopeN = 1;
            if (val.inductionHypothesis() && !mb.inductionTreeSettled(canonicalScope)) {
                const int32_t eatSlot = Codec<StatementKey>::decode(markedPk).orig;
                bool eats = gl::ScopeSyntax::inductionEatingPosition(mb.nameMap, canonicalScope) == canonicalScope
                    && mb.inductionSlotOpen(canonicalScope, eatSlot);
                assert(valKeyN <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                    && "updateAdmissionMapRecursion: a hypothesis chain exceeds the key cap");
                StrSpan markedChain[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                if (eats) {
                    const StrReplacement toMarker[1] = { { kBoundArg, StrSpan("marker", 6) } };
                    for (int32_t k = 0; k < valKeyN; ++k)
                        markedChain[k] = StrSpan(replaceKeysScratch(strArena, valKeySpans[k], toMarker, 1));
                    eats = !(gl::ScopeSyntax::atOrBelowInductionScope(mb.nameMap, canonicalScope)
                        && gl::ScopeSyntax::inductionChainOnPath(mb.nameMap, canonicalScope, markedChain, valKeyN));
                }
                if (eats) {
                    mb.recordInductionEat(canonicalScope, eatSlot);
                    const StrSpan names[1] = { boundArg };
                    const ScratchString payload = gl::ScopeSyntax::inductionScopePayloadScratch(
                        strArena, markedChain, valKeyN, names, 1);
                    const NameId inPlaceScope = mb.nameMap.encodePush(canonicalScope, StrSpan(payload));
                    const StrSpan scopeView = mb.nameMap.decodeView(inPlaceScope);
                    demandScopes[demandScopeN++] = StrSpan(ScratchString::copyFrom(strArena, scopeView.ptr, scopeView.len));
                }
            }

            mb.overallHashMemory.productsOfRecursionIds.mint(mb.nameMap.encode(boundArg));

            // 6. Construct New Admission Entry.
            // replacementMap {kBoundArg -> boundArg} as a stack
            // StrReplacement[1]; replaceKeysScratch is the byte-exact twin of
            // ce::replaceKeysInString (single key, greedy-longest, token-boundary).
            // newKey is a StrSpan run over per-element rewrites on the string
            // arena; each element is valueInterner.encode'd at the insert below.
            StrReplacement replPair[1];
            replPair[0].key = kBoundArg;
            replPair[0].value = boundArg;

            // Match-level gen-tier window: the newKey / newRemaining span arrays.
            ScratchScope matchGenScope(genArena);
            StrSpan* newKeySpans = (valKeyN == 0) ? nullptr
                : reinterpret_cast<StrSpan*>(genArena.resolve(genArena.alloc(
                      valKeyN * static_cast<int32_t>(sizeof(StrSpan)),
                      static_cast<int32_t>(alignof(StrSpan)))));
            for (int32_t k = 0; k < valKeyN; ++k)
                newKeySpans[k] = StrSpan(
                    replaceKeysScratch(strArena, valKeySpans[k], replPair, 1));
            const int32_t newKeyN = valKeyN;

            // newRemainingArgs = distinct(valRemaining ∪ {boundArg}) as a
            // sorted StrSpan run on the byte-bump gen-scratch tier (the
            // createAuxyImplication untouchables recipe, row 41). Spans alias the
            // valRemaining string-arena copies + boundArg (a canonical copy on
            // the function-level string frame), both alive to the match's end and
            // disjoint from the value interner minted below (I-3). Membership rides
            // std::binary_search(compareSpans); the sorted-encode is a direct
            // in-order valueInterner.encode loop at the insert site below,
            // byte-identical to the former std::set encodeValueSetSorted (the
            // run is sorted-unique).
            const int32_t newRemCap = valRemN + 1;
            StrSpan* newRemaining = reinterpret_cast<StrSpan*>(genArena.resolve(
                genArena.alloc(
                    newRemCap * static_cast<int32_t>(sizeof(StrSpan)),
                    static_cast<int32_t>(alignof(StrSpan)))));
            int32_t newRemainingN = 0;
            for (int32_t r = 0; r < valRemN; ++r)
                newRemaining[newRemainingN++] = valRemaining[r];
            newRemaining[newRemainingN++] = boundArg;
            std::sort(newRemaining, newRemaining + newRemainingN,
                [](const StrSpan& a, const StrSpan& b) {
                    return compareSpans(a, b) < 0;
                });
            {
                int32_t w = 0;
                for (int32_t r = 0; r < newRemainingN; ++r) {
                    if (w == 0
                        || !equalSpans(newRemaining[w - 1], newRemaining[r])) {
                        newRemaining[w++] = newRemaining[r];
                    }
                }
                newRemainingN = w;
            }

            // An in-place eat of the induction hypothesis: the advanced chain's
            // premises with only fixed arguments are the candidates for the
            // hypothesis's missing expressions in the eat's own scope, which
            // publishes nothing. Both strong forms share every premise but
            // their atomic bound premise, so the block's first hypothesis rule
            // is the source (D-575).
            if (demandScopeN == 2) {
                for (int32_t k = 0; k < newKeyN; ++k) {
                    if (!fullyBoundElement(newKeySpans[k], newRemaining, newRemainingN)
                        || !admissionPolicy.ordis2KeyEligible(newKeySpans[k])) continue;
                    assert(mb.recursionAuxyImplId != 0
                        && "updateAdmissionMapRecursion: a hypothesis chain lives in a step block");
                    fileHypothesisOrDemand(mb, newKeySpans[k], demandScopes[1], mb.recursionAuxyImplId);
                }
            }

            // 7. Validate New Key Candidates. A derived key is the demand for
            // the one argument a participant still lacks (binary_search over
            // the sorted StrSpan run == the former std::set membership): an
            // operator whose output is outside the new remaining args while
            // every other argument is inside; a compound element (the
            // participantOf classes above) with exactly one distinct argument
            // outside.
            for (int32_t nkIx = 0; nkIx < newKeyN; ++nkIx) {
                const StrSpan newKeyElem = newKeySpans[nkIx];

                // A config-backed compound premise that is not an operator,
                // relations included and in either polarity, left with every
                // argument bound is one concrete statement: its integration is
                // prepared for exactly that instance, as updateAdmissionMap
                // prepares a fully bound premise of a recursion-owned rule at
                // install (I-294). The
                // arriving element and a premise already known at the scope or
                // an ancestor need none; a negated premise's reductio waits for
                // the chain's positive premises (prepareHypothesisPremise).
                if (nkIx != keIx) {
                    const StrSpan boundCore = extractExpressionSpan(newKeyElem);
                    const bool boundOperator = definitions.operatorNames().find(std::string_view(boundCore.ptr,
                        static_cast<std::size_t>(boundCore.len))) != definitions.operatorNames().end();
                    const LogicalEntity* boundEntity =
                        (boundOperator || boundCore.len == 0) ? nullptr : definitions.compiledEntity(boundCore);
                    if (boundEntity != nullptr
                        && !equalSpans(StrSpan(boundEntity->category), StrSpan("atomic", 6))
                        && definitions.coreConfig(boundCore) != nullptr) {
                        StrSpan boundArgs[ExecutionParameters::MAX_ARITY];
                        const int32_t boundArgN = getArgsSpans(newKeyElem, boundArgs,
                                                               ExecutionParameters::MAX_ARITY);
                        bool fullyBound = true;
                        for (int32_t a = 0; a < boundArgN && fullyBound; ++a)
                            fullyBound = std::binary_search(newRemaining, newRemaining + newRemainingN,
                                boundArgs[a], [](const StrSpan& x, const StrSpan& y) {
                                    return compareSpans(x, y) < 0;
                                });
                        if (fullyBound) {
                            const NameId boundId = mb.nameMap.lookup(newKeyElem);
                            if (boundId == 0
                                || !gl::StatementKnowledge::ancestorKnown(mb, boundId, canonicalScope, true)) {
                                RT_SCOPE_HERE("UAMR_PREPARE_BOUND_INTEGRATION");
                                integrationProcessor.prepareHypothesisPremise(newKeyElem, newKeySpans, newKeyN,
                                    newRemaining, newRemainingN, mb, validityName);
                            }
                            continue;
                        }
                    }
                }

                const Participant nRole = participantOf(newKeyElem);
                if (nRole == Participant::none) {
                    continue;
                }

                StrSpan newKArgs[ExecutionParameters::MAX_ARITY];
                const int32_t newKArgsN = getArgsSpans(newKeyElem, newKArgs,
                                                       ExecutionParameters::MAX_ARITY);
                const auto inNewRemaining = [&](const StrSpan& arg) -> bool {
                    return std::binary_search(newRemaining, newRemaining + newRemainingN, arg,
                        [](const StrSpan& x, const StrSpan& y) {
                            return compareSpans(x, y) < 0;
                        });
                };
                StrSpan newBoundArg;
                bool demanded = false;
                if (nRole == Participant::operatorElement) {
                    const int32_t newOutIdx = outputSlotOf(newKeyElem);
                    assert(newOutIdx >= 0 && newOutIdx < newKArgsN);
                    newBoundArg = newKArgs[newOutIdx];
                    demanded = !inNewRemaining(newBoundArg);
                    for (int32_t i = 0; i < newKArgsN && demanded; ++i)
                        if (i != newOutIdx && !inNewRemaining(newKArgs[i])) demanded = false;
                } else {
                    int32_t freeN = 0;
                    for (int32_t a = 0; a < newKArgsN; ++a) {
                        bool repeated = false;
                        for (int32_t prior = 0; prior < a && !repeated; ++prior)
                            repeated = equalSpans(newKArgs[prior], newKArgs[a]);
                        if (repeated || inNewRemaining(newKArgs[a])) continue;
                        if (freeN == 0) newBoundArg = newKArgs[a];
                        ++freeN;
                    }
                    demanded = freeN == 1;
                }

                if (!demanded) {
                    continue;
                }

                // 8. Insert New Rule. Single-key marker map -> StrReplacement[1]
                // (replaceKeysScratch byte-exact twin of ce::replaceKeysInString).
                StrReplacement newMarkerPair[1];
                newMarkerPair[0].key = newBoundArg;
                newMarkerPair[0].value = StrSpan("marker", 6);
                const ScratchString newMarkedExpr =
                    replaceKeysScratch(strArena, newKeyElem, newMarkerPair, 1);

                // depth/sec/flag come from the matched snapshot record 'val'
                // (read-side, stays). The two SEQUENCED mints stay ORDERED:
                // encodeValueVector(newKey) FIRST, then the sorted-unique
                // newRemaining — a direct in-order valueInterner.encode loop is
                // encodeValueSpanSetSorted's byte-identical twin (no scratch
                // dedup set), and no owning value object is built.
                const ArenaOffset bMark = genArena.cursor();
                const int32_t bKeyCount = newKeyN;
                int32_t* bKeyIds = (bKeyCount > 0)
                    ? reinterpret_cast<int32_t*>(genArena.resolve(
                          genArena.alloc(bKeyCount * sizeof(int32_t), alignof(int32_t))))
                    : nullptr;
                for (int32_t j = 0; j < bKeyCount; ++j)
                    bKeyIds[j] = mb.valueInterner.encode(newKeySpans[j]);
                int32_t* bRemIds = (newRemainingN > 0)
                    ? reinterpret_cast<int32_t*>(genArena.resolve(
                          genArena.alloc(newRemainingN * sizeof(int32_t), alignof(int32_t))))
                    : nullptr;
                for (int32_t j = 0; j < newRemainingN; ++j)
                    bRemIds[j] = mb.valueInterner.encode(newRemaining[j]);

                // The demand at the statement's scope first, then its copy in
                // the in-place eat's scope.
                int64_t newMarkedPks[2] = { 0, 0 };
                for (int32_t d = 0; d < demandScopeN; ++d) {
                    const int64_t newMarkedPk = mintTemplateKey(mb.templateInterner,
                        mb.nameMap, StrSpan(newMarkedExpr), demandScopes[d]);
                    newMarkedPks[d] = newMarkedPk;
                    const int32_t previousDemandId = mb.overallHashMemory.admissionMap.lookup(newMarkedPk);
                    const int32_t previousDemandCount = previousDemandId == 0 ? 0 : mb.overallHashMemory.admissionMap.runLen(previousDemandId);
                    // The whole flag byte carries over: an advanced copy of the
                    // induction hypothesis's chain stays the hypothesis's
                    // (D-519).
                    insertAdmissionIdsBlob(mb.overallHashMemory.admissionMap,
                                           newMarkedPk, val.depth(),
                                           val.sec(), val.flagByte(),
                                           bKeyIds, bKeyCount, bRemIds, newRemainingN,
                                           mb.valueInterner, genArena);
                    if (mb.overallHashMemory.admissionMap.runLen(mb.overallHashMemory.admissionMap.lookup(newMarkedPk)) != previousDemandCount)
                        mb.lbMemory.canonicalUses.algebraDemands.changed(newMarkedPk);
                }
                genArena.popTo(bMark);
                const uint8_t srcStatusVal =
                    mb.overallHashMemory.admissionStatusMap.findOr(
                        markedPk, static_cast<uint8_t>(0));
                for (int32_t d = 0; d < demandScopeN; ++d)
                    mb.overallHashMemory.admissionStatusMap.upsert(
                        newMarkedPks[d], srcStatusVal);

                // Populate vars-in-keys cache for the algebra equi-class hook.
                StrSpan nmArgs[ExecutionParameters::MAX_ARITY];
                const int32_t nmArgsN = getArgsSpans(StrSpan(newMarkedExpr), nmArgs,
                                                     ExecutionParameters::MAX_ARITY);
                for (int32_t a = 0; a < nmArgsN; ++a) {
                    if (!equalSpans(nmArgs[a], StrSpan("marker", 6))) {
                        mb.overallHashMemory.varsInAdmissionMapKeys.mint(mb.templateInterner.encode(nmArgs[a]));
                    }
                }

                // Integration demand for the derived key
                // (D-413): the induction
                // chain needs its one missing argument constructed, not only
                // admitted, so the marker form's integration is prepared as
                // updateAdmissionMap's marker branch prepares it at install.
                // It runs after the entry and its status exist, so a statement
                // the preparation deposits finds the key. Positive elements
                // only: a negated marker form would take the contradiction
                // route and seed a statement carrying the marker.
                if (newKeyElem.ptr[0] != '!') {
                    RT_SCOPE_HERE("UAMR_PREPARE_INTEGRATION");
                    integrationProcessor.prepareIntegration(StrSpan(newMarkedExpr), newRemaining, newRemainingN,
                        mb, validityName, StrSpan());
                }

                // Each demand wakes the sources parked for it, at its scope or
                // an ancestor; an ordis cohort matching the derived key wakes by
                // mail.
                for (int32_t d = 0; d < demandScopeN; ++d) {
                    rejectedSourceReplay.revisitRejected2(StrSpan(newMarkedExpr), mb, demandScopes[d]);
                    rejectedSourceReplay.revisitRejectedOrdis(StrSpan(newMarkedExpr), mb, demandScopes[d]);
                }
            }
        }
    }
    admSnapArena.popTo(admSnapMark);
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)
void AdmissionMemory::makeAdmissionKeys(const std::vector<std::string>& implication,
    const std::string& expression,
    HashMemory& localMemory,
    const std::string& validityName,
    Memory& mb) {
    // Vector forwarder onto the span-run body (the additive twin) — build a
    // stack StrSpan run over the caller-stable vector strings, byte-identical.
    StrSpan implRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1];
    const int32_t implFwdN = static_cast<int32_t>(implication.size());
    assert(implFwdN <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1
        && "makeAdmissionKeys forwarder: implication length exceeds cap");
    for (int32_t i = 0; i < implFwdN; ++i) implRun[i] = StrSpan(implication[i]);
    makeAdmissionKeys(implRun, implFwdN, StrSpan(expression), localMemory,
                      StrSpan(validityName), mb);
}

/// @brief Span-run overload of @ref makeAdmissionKeys — the span-native
        ///        body; the `const std::vector<std::string>&` form forwards.
        ///
        /// @details
        /// Byte-identical to the vector form (same arg validation, same
        /// `mintTemplateKey` / `admissionSetIntegration` / `revisitRejected-
        /// Integration2` interner-touch sequence on the same bytes), but reads the
        /// implication chain as a caller-owned `StrSpan` run and takes the
        /// expression / validity as spans, so a span-native caller
        /// (`addToHashMemory`, `prepareIntegrationCore2`) reaches it 0% heap. The
        /// vector form builds a stack `StrSpan` run and delegates here.
        ///
        /// @param impl         The implication-chain elements (premise order,
        ///                     head last); cap `MAX_ADMISSION_KEY_ELEMENTS + 1`.
        /// @param implN        Number of chain elements in @p impl.
        /// @param expr         The candidate expression.
        /// @param localMemory  HashMemory to register the trigger / admission keys.
        /// @param validityName Validity scope.
        /// @param mb           Owning LB.
        /// @return (void)
        /// @see makeAdmissionKeys(const std::vector<std::string>&, …) — the twin
        ///      oracle; mintTemplateKey, revisitRejectedIntegration2.
void AdmissionMemory::makeAdmissionKeys(const StrSpan* impl, int32_t implN,
    StrSpan expr,
    HashMemory& localMemory,
    StrSpan validityName,
    Memory& mb) {
    // 1. Check expression arguments. Args are zero-copy slices of `expr`
    //    (a caller-stable buffer) — no owning vector.
    //    Must have at least one "repl_" and all others must be "u_", "repl_", or "marker".
    StrSpan exprArgs[ExecutionParameters::MAX_ARITY];
    const int32_t exprArgsN = getArgsSpans(expr, exprArgs,
                                           ExecutionParameters::MAX_ARITY);
    bool hasRepl = false;

    for (int32_t i = 0; i < exprArgsN; ++i) {
        const StrSpan arg = exprArgs[i];
        if (startsWithSpan(arg, "repl_", 5)) {
            hasRepl = true;
        }
        else if (startsWithSpan(arg, "u_", 2)) {
            continue;
        }
        else if (equalSpans(arg, StrSpan("marker", 6))) {
            continue;
        }
        else {
            // Argument is not allowed (must be u_, repl_, or marker)
            return;
        }
    }

    if (!hasRepl) {
        return;
    }

    {
        const int64_t triggerKey = mintTemplateKey(mb.templateInterner, mb.nameMap, expr, validityName);
        localMemory.triggersForAdmissionSetIntegration.mint(triggerKey);
    }

    // 2. Validate Head vs Expression
    if (implN == 0) return;
    const StrSpan head = impl[implN - 1];

    // A. Check Core Expression Match — compare the two extract spans directly.
    if (!equalSpans(extractExpressionUniversalSpan(head),
                    extractExpressionUniversalSpan(expr))) {
        return;
    }

    // B. Check Negation Status Match
    //    Assuming canonical form where negation always puts '!' at the start
    bool headIsNegated = (head.len > 0 && head.ptr[0] == '!');
    bool exprIsNegated = (expr.len > 0 && expr.ptr[0] == '!');

    if (headIsNegated != exprIsNegated) {
        return;
    }

    // C. Check Arguments and 1:1 "u_" relation. Head args are slices of `head`
    //    (a caller-stable buffer).
    StrSpan headArgs[ExecutionParameters::MAX_ARITY];
    const int32_t headArgsN = getArgsSpans(head, headArgs,
                                           ExecutionParameters::MAX_ARITY);
    assert(headArgsN == exprArgsN);

    // The former std::map<std::string,std::string> rename map becomes a
    // StrReplacement stack run built directly in head-arg order, keys/values
    // spans into head/expression. On a REPEATED head arg the map's
    // last-write-wins is reproduced by overwriting the matching pair's value.
    // The pair ORDER is not observable — replaceKeysScratch is greedy-longest
    // token-boundary, order-independent — so this first-seen order is
    // byte-identical to the former sorted-map iteration.
    StrReplacement replPairs[ExecutionParameters::MAX_KEY_SLOTS];
    int32_t replPairsN = 0;
    for (int32_t i = 0; i < headArgsN; ++i) {
        const StrSpan hArg = headArgs[i];
        const StrSpan eArg = exprArgs[i];

        const bool hIsU = startsWithSpan(hArg, "u_", 2);
        const bool eIsU = startsWithSpan(eArg, "u_", 2);

        // "Important: head and expression must have all 'u_' args in the same position."
        if (hIsU || eIsU) {
            if (!equalSpans(hArg, eArg)) {
                return; // Violation of 1:1 relation
            }
        }
        else {
            // Map non-u head arg to expression arg, last-write-wins on a repeat.
            int32_t slot = -1;
            for (int32_t p = 0; p < replPairsN; ++p)
                if (equalSpans(replPairs[p].key, hArg)) { slot = p; break; }
            if (slot >= 0) {
                replPairs[slot].value = eArg;
            }
            else {
                assert(replPairsN < ExecutionParameters::MAX_KEY_SLOTS
                    && "makeAdmissionKeys: rename pair count exceeds cap");
                replPairs[replPairsN].key = hArg;
                replPairs[replPairsN].value = eArg;
                ++replPairsN;
            }
        }
    }

    // 3. Apply Mapping to Implication Body (Head excluded). mappedElement and
    //    admissionKey are fresh bytes on the per-slot string-scratch arena
    //    (replaceKeysScratch is the byte-exact twin of ce::replaceKeysInString),
    //    framed per element by elemScope. The interner mints below touch
    //    templateInterner / NameMap / originInterner — never this arena — so the
    //    held spans stay valid across the door calls (I-3).
    const unsigned akSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : scratchArenas().slotCount() - 1;
    ScratchArena& strArena = scratchArenas().forSlot(akSlot);
    for (int32_t i = 0; i + 1 < implN; ++i) {
        ScratchScope elemScope(strArena);
        const ScratchString mappedElement =
            replaceKeysScratch(strArena, impl[i], replPairs, replPairsN);

        // Analyze args of the mapped element — slices of mappedElement.
        StrSpan outArgs[ExecutionParameters::MAX_ARITY];
        const int32_t outArgsN = getArgsSpans(StrSpan(mappedElement), outArgs,
                                              ExecutionParameters::MAX_ARITY);
        int markerCount = 0;
        bool argsValid = true;

        for (int32_t a = 0; a < outArgsN; ++a) {
            const StrSpan arg = outArgs[a];
            if (equalSpans(arg, StrSpan("marker", 6))) {
                markerCount++;
            }
            else if (startsWithSpan(arg, "repl_", 5)) {
                continue;
            }
            else if (startsWithSpan(arg, "u_", 2)) {
                continue;
            }
            else {
                argsValid = false;
                break;
            }
        }

        // Admission Condition: Exactly one "marker" and valid args
        if (argsValid && markerCount == 1) {
            const ScratchString admissionKey = gl::DefinitionExpansion::removeUPrefixScratch(
                strArena, StrSpan(mappedElement));
            {
                const int64_t setKey = mintTemplateKey(mb.templateInterner, mb.nameMap,
                                                       StrSpan(admissionKey), validityName);
                localMemory.admissionSetIntegration.mint(setKey);
            }

            // New admission-set-integration key — revive any prior
            // integration-side rejection keyed at this marker form.
            // `mb.overallHashMemory.rejectedMapIntegration` is the fixed
            // lookup target; firing here is safe regardless of whether
            // `localMemory` is overallHashMemory or a local/delta mirror
            // (the overall-memory insert happens first in the enclosing
            // addToHashMemory 3-way fan-out, so by any subsequent call the
            // admission key is visible in overall memory).
            rejectedSourceReplay.revisitRejectedIntegration2(StrSpan(admissionKey), mb, validityName);
        }

        // OR integration trigger: if premise is a compound OR expression with
        // only repl_/u_/marker args (no bound pi_lev_ vars), trigger integration.
        // This enables the prover to construct OR expressions from their elements.
        if (argsValid) {
            const LogicalEntity* le = this->definitions.compiledEntity(
                extractExpressionUniversalSpan(StrSpan(mappedElement)));
            if (le && equalSpans(StrSpan(le->category), StrSpan("or", 2))) {
                // TODO: call prepareIntegration for the OR expression (with u_ intact)
            }
        }
    }
}
}
