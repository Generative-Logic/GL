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
#include "../compiler/compact_expression_syntax.hpp"
#include "statement_memory.hpp"
#include "demand_built_rules.hpp"
namespace gl {
/// @brief Bind inputs for statement memory.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param admissionPolicy Borrowed AdmissionPolicy interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param admissionMemory Borrowed AdmissionMemory interface or state; must outlive this component.
/// @param caseLedger Borrowed CaseLedger interface or state; must outlive this component.
/// @param caseProcessor Borrowed CaseProcessor interface or state; must outlive this component.
/// @param compactRuleProcessor Borrowed CompactRuleProcessor interface or state; must outlive this component.
/// @param disintegrationProcessor Borrowed DisintegrationProcessor interface or state; must outlive this component.
/// @param equalityPropagation Borrowed EqualityPropagation interface or state; must outlive this component.
/// @param equivalenceProcessor Borrowed EquivalenceProcessor interface or state; must outlive this component.
/// @param equivalenceQueries Borrowed EquivalenceQueries interface or state; must outlive this component.
/// @param integrationProcessor Dis value used by this operation.
/// @param orImplicationProcessor Borrowed OrImplicationProcessor interface or state; must outlive this component.
/// @param proofHistory Borrowed ProofHistory interface or state; must outlive this component.
/// @param unaryIntegrationProcessor Borrowed UnaryIntegrationProcessor interface or state; must outlive this component.
/// @param watchProcessor Borrowed WatchProcessor interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
StatementMemory::StatementMemory(AdmissionPolicy& admissionPolicy, Canonicalizer& canonicalizer, const CompiledDefinitionRegistry& definitions, const ProverParameters& parameters, AdmissionMemory& admissionMemory, CaseLedger& caseLedger, CaseProcessor& caseProcessor, CompactRuleProcessor& compactRuleProcessor, DisintegrationProcessor& disintegrationProcessor, EqualityPropagation& equalityPropagation, EquivalenceProcessor& equivalenceProcessor, EquivalenceQueries& equivalenceQueries, IntegrationProcessor& integrationProcessor, OrImplicationProcessor& orImplicationProcessor, ProofHistory& proofHistory, UnaryIntegrationProcessor& unaryIntegrationProcessor, WatchProcessor& watchProcessor) : admissionPolicy(admissionPolicy), canonicalizer(canonicalizer), definitions(definitions), parameters(parameters), admissionMemory(admissionMemory), caseLedger(caseLedger), caseProcessor(caseProcessor), compactRuleProcessor(compactRuleProcessor), disintegrationProcessor(disintegrationProcessor), equalityPropagation(equalityPropagation), equivalenceProcessor(equivalenceProcessor), equivalenceQueries(equivalenceQueries), integrationProcessor(integrationProcessor), orImplicationProcessor(orImplicationProcessor), proofHistory(proofHistory), unaryIntegrationProcessor(unaryIntegrationProcessor), watchProcessor(watchProcessor) {}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)
/// @brief Public entry to install an expression into an LB — the main
/// kernel of the prover.
///
/// @details
/// Every expression that lands in any LB goes through this function:
/// fresh statements from the prover loop, mail-in deposits from the
/// hash burst, head emissions from `checkLocalEncodedMemoryStatic`,
/// and disintegration products from `disintegrateExpr2`. The function
/// dispatches by expression shape:
///
/// 1. **Atomic predicate** (e.g. `(p[a])`) — installs as a statement
///    via the kernel; updates `intEncodedStatements`,
///    `intEncodedStatements`, `intLocalEncodedStatementsSet`, and
///    `intKnownStatements`.
/// 2. **Equality** (e.g. `(=[a,b])`) — routes through `addEquality`
///    so the equivalence-class machinery can absorb it
///    (with mirror per [I-9](../../docs/agentic_swdd/30_invariants.md#i-9)).
/// 3. **Negated equality** (e.g. `!(=[a,b])`) — routes through
///    `addNegatedEquality` for the one-sided expansion per
///    [I-12](../../docs/agentic_swdd/30_invariants.md#i-12).
/// 4. **Implication** (e.g. `(>[v](p[v])(q[v]))`) — installed as a
///    hash rule via `addToHashMemory` (after disintegration when
///    appropriate).
///
/// Anchor slot typing is NOT dispatched here: the x-prefixed anchor
/// variants are installed grid-wide by the `prehandleAnchor` pre-pass
/// before the iteration loop runs.
///
/// The `status` mode selector controls the deposit path; the full
/// table lives in [`docs/agentic_swdd/10_pipeline/04_prover.md`](../../docs/agentic_swdd/10_pipeline/04_prover.md)
/// (`addExprToMemoryBlock — status parameter reference`). Statuses
/// 0/1 disintegrate via `disintegrateExpr2`. **status 3** (external-mail
/// absorb, ASIC 0.1 reshuffle) is admitted to `disintegrateExpr2` ONLY
/// when the expression is a compact implication `(implication<N>...)`
/// (gate at the call site, OUTSIDE disintegrateExpr2); a plain mail
/// statement is absorbed as a statement, never disintegrated. Recovered
/// implications install through the usual persistent 3-way
/// `addToHashMemory` path.
///
/// **Per [I-27](../../docs/agentic_swdd/30_invariants.md#i-27)**, ancestor-scan
/// dedupe runs at this entry (Site F / Site H): if the expression
/// already exists in any ancestor scope, the deposit is skipped.
/// This is what keeps cross-LB writes during the parallel phase
/// from causing duplicate-deposit storms
/// ([I-28](../../docs/agentic_swdd/30_invariants.md#i-28)).
///
/// @param expr           Canonical MPL expression text.
/// @param memoryBlock    Target LB.
/// @param iteration      Witness generation carried by the arrival.
/// @param status         Deposit route selecting the processing contract.
/// @param involvedLevels Ascending-unique level run — LB levels at which
///                       the expression is admissible (caller-owned per
///                       the levels-run chain contract).
/// @param involvedLevelCount The level count (`>= 0`).
/// @param origin         Recorded derivation of the deposit.
/// @param coreId         Worker slot; -1 selects the setup slot.
/// @param auxyIndex      Auxiliary goal registration index.
/// @param validityName   Scope in which the expression is offered.
/// @param doNotDisintegrate Suppress disintegration for this offer.
/// @param allowOrDisintegration Permit rule-product OR cohort opening.
/// @param registeredOut  Optional encoded registration result.
/// @param recursionHead  Force-disintegrate the induction implication's head.
/// @param convergenceMailOut Explicit convergence output during input absorption;
///                       nullptr selects same-iteration mail for ordinary callers.
/// @param replayKinds    Queued construction replay bits (bit 8 selects the
///                       flag-5 relay route).
/// @param replayIntegrationDepth Original assigned depth of a replayed source,
///                       or -1.
/// @param integrationPremise The offer is a premise of an implication being
///                       integrated: an OR compact opens its case split, a
///                       negated conjunction splits into its complemented
///                       conjuncts (D-460).
/// @param conjecturePremise The offer is a conjecture's non-anchor premise
///                       seeded into its block: every OR it unfolds into,
///                       through existence unfolding, opens its case split
///                       with the rule-product permission where this block has
///                       proof work (D-495), and an OR published at main is
///                       staged for the flag-6 unconditional-ordis mail to
///                       every descendant (D-528).
///                       At status 3 the offer is such an OR arriving with
///                       flag 6. A cohort opened under it releases its cases
///                       with force_ witnesses
///                       (D-527).
/// @return Nothing; writes the expression through the ordinary deposit doors.
/// @invariant [I-9](../../docs/agentic_swdd/30_invariants.md#i-9) — addEquality
///            mirror guard.
/// @invariant [I-12](../../docs/agentic_swdd/30_invariants.md#i-12) — !(=[a,b])
///            one-sided expansion.
/// @invariant [I-27](../../docs/agentic_swdd/30_invariants.md#i-27) — Site F / H
///            ancestor-scan dedupe at entry.
/// @invariant [I-28](../../docs/agentic_swdd/30_invariants.md#i-28) — cross-LB
///            writes deferred to post-`pool.join()` collectors.
/// @see `disintegrateExpr2` — structural disintegration.
/// @see `addToHashMemory` (memory.cpp) — implication branch.
/// @see `addEquality`, `addNegatedEquality` — shape-specific branches.
void StatementMemory::addExprToMemoryBlock(StrSpan expr,
    Memory& memoryBlock,
    int iteration,
    int status,
    const int* involvedLevels, int32_t involvedLevelCount,
    const TransientOrigin& origin,
    int coreId,
    int auxyIndex,
    StrSpan validityName,
    bool doNotDisintegrate,
    bool allowOrDisintegration,
    IntEncodedExpr* registeredOut,
    bool recursionHead,
    ColdMail* convergenceMailOut, int32_t replayKinds, int32_t replayIntegrationDepth,
    bool integrationPremise, bool conjecturePremise) {

    // Two-route or-cohort opening (admission-based ordis; restores the D-32
    // gate READ the sequenced branch had overridden): route (b) is the
    // threaded product-of-disintegration signal (the FiringRecord /
    // disintegration-signal plumbing fed from D-32's
    // productOfDisintegration stamp) — such heads open unconditionally, as
    // in rungs 1+2; route (a) is the demand probe at the or arm, enabled by
    // allowOrProbe for every real deposit through this door. The
    // rule-intrinsic doNotDisintegrate (D-241 intro-fired suppression,
    // integration-justified rules) and the D-29 firing-context clauses block
    // both routes. The hypothetical-disintegration path calls
    // disintegrateExpr2 directly (bypassing this door) with both defaults
    // false, so no cohort ever mints — and no park ever writes — at a
    // sentinel scope.
    if ((replayKinds & 8) != 0) status = 5;
    allowOrDisintegration = allowOrDisintegration && !doNotDisintegrate;
    // An integration premise (D-460) obeys the
    // same rule-intrinsic suppression.
    integrationPremise = integrationPremise && !doNotDisintegrate;
    // A conjecture's premise (D-495,
    // D-528) likewise: seeded locally (status 0),
    // or an OR it unfolded into arriving with mail flag 6 (status 3).
    conjecturePremise = conjecturePremise && !doNotDisintegrate;
    assert((!conjecturePremise || status == 0 || status == 3)
        && "addExprToMemoryBlock: a conjecture premise is seeded locally or arrives with flag 6");
    bool allowOrProbe = !doNotDisintegrate;

    // Atomic RT attribution: the door's own row carries the gate / glue
    // self-time; every heavyweight sub-call below opens its own nested row.
    RT_SCOPE_HERE("DOOR_ADD_EXPR");

    // --- Static (int-based) early checks ---
    // Fast path: duplicate + validity filter with just nm.encode (no regex, no vector alloc)
    NameId origId = 0;
    NameId valId = 0;
    {
        RT_SCOPE_HERE("DOOR_ENTRY_GATES");
        origId = memoryBlock.nameMap.encode(expr);
        valId = memoryBlock.nameMap.encode(validityName);
        if (memoryBlock.phase3IgnoresScope(valId)) return;
        canonicalizer.initializeCanonicalScope(memoryBlock, valId);

        // An integration trial scope, the usefulness scope and the scopes
        // below them take part only in firings, equality classes and
        // structural expansion (D-481, D-521): no
        // OR case opens there and no goal is ever registered there. Nothing
        // disintegrates at or below the usefulness scope.
        if (gl::ScopeSyntax::atOrBelowRestrictedScope(memoryBlock.nameMap, valId)) {
            assert(status != 2
                && "addExprToMemoryBlock: a goal never enters a restricted scope");
            allowOrDisintegration = false;
            integrationPremise = false;
            conjecturePremise = false;
            allowOrProbe = false;
            if (gl::ScopeSyntax::atOrBelowUsefulnessScope(memoryBlock.nameMap, valId))
                doNotDisintegrate = true;
        }
        // An induction scope registers no goal of its own: a goal reached
        // inside the tree closes at the root (D-519).
        assert(!(status == 2 && gl::ScopeSyntax::isInductionScope(memoryBlock.nameMap, valId))
            && "addExprToMemoryBlock: a goal never enters an induction scope itself");
        // Goals live only at main, implication and integration subproofs and
        // OR-integration scopes: the innermost goal set of an OR case relies
        // on it (D-549, I-330).
        assert(!(status == 2 && !gl::ScopeSyntax::canHoldGoal(memoryBlock.nameMap, valId))
            && "addExprToMemoryBlock: a goal enters only main, a subproof or an OR-integration scope");
        // The induction hypothesis fired in a leaf of an induction tree: this
        // burst's cleanup settles the tree (D-520).
        if (recursionHead && gl::ScopeSyntax::isInductionScope(memoryBlock.nameMap, valId))
            memoryBlock.recordInductionFiring(valId);
        // Any scope the hypothesis shot at: an ancestor's integration
        // instruction is updated below its scope only after a shot
        // (D-572).
        if (recursionHead) memoryBlock.lbMemory.hypothesisShots.mint(valId);

        // Site F — dup suppression via ancestor scan. A known statement at any
        // ancestor scope (including self) means the same fact already holds at a
        // strictly weaker set of assumptions, so the child-scope insertion is
        // redundant. ancestorsOf[valId] includes valId itself and every strict
        // prefix scope registered via encodePush. For a flat root (e.g. legacy
        // integration cleanSignature-as-validity) the list is just {valId},
        // degenerating to the old strict-equality check — safe under the
        // MAIN_ID transition guard. The scan is the shared `ancestorKnown`
        // predicate (one definition of the contract across every door); a
        // refusal still carries the or-branch resolution signal (I-174) — the
        // branch derived the expression even though the deposit is redundant.
        if (!parameters.compressor_mode && replayKinds == 0
            && gl::StatementKnowledge::ancestorKnown(memoryBlock, origId, valId, /*includeSelf=*/true)) {
            caseProcessor.stageOrReleaseForRefusedDeposit(memoryBlock, origId, valId);
            return;
        }

        // Site H — ancestor-scan the int validity blacklist. A blacklist entry at
        // any ancestor scope filters every descendant scope (deeper scopes inherit
        // the filter because they carry strictly more assumptions).
        for (int32_t ancK = 0, ancN = memoryBlock.nameMap.ancLen(valId);
             ancK < ancN; ++ancK) {
            const NameId anc = memoryBlock.nameMap.ancAt(valId, ancK);
            if (memoryBlock.intValidityNamesToFilter.contains(anc)) return;
        }
    }

    // The canonical door: only the class-canonical form of a deposit is
    // admitted. For every status except a goal (2) and every shape except
    // the two equality shapes (goal closure is literal and neither shape is
    // ever multiplied) and anchors (a scope identity, I-53), the text is
    // rewritten under the maintained complete class view at its scope.
    // Raw scope identities remain separate from the deposited text. A
    // changed deposit: (1) keeps the RAW text's producer history line
    // (Rule 16 — the canonical form's bridge below terminates on it) and,
    // for a local status, ships that line to mailOut so a receiver's
    // chapter walk terminates on it too (the D-286 ship — the raw form
    // never becomes a delta row, so fillMailOut would never carry it);
    // (2) ends the deposit when the canonical form is already known at
    // this scope or a strict ancestor — the same Site-F refusal as for the
    // raw text, releasing the or-branch resolution signal for BOTH ids
    // (I-174); (3) otherwise proceeds with the canonical text, its origin
    // replaced by ONE `equality1` bridge (deps[0] = raw form at this scope,
    // then each applied `(=[member,canonical])` cited where its row lives)
    // when the canonical form has no history row yet (the I-34 gate), and
    // its level run = the deposit's levels ∪ the applied pairs' class
    // levels. `applyEquiClasses` still multiplies the registered canonical
    // form into its orbit, so a literal non-equality goal closes from the
    // product in the same standardProcessing call. The canonical text and
    // every span the rest of the door reads from it ride the string-tier
    // arena under this function-level scope; every nested scope the
    // deposit paths open rewinds above it (LIFO).
    static constexpr int32_t kDoorLevelCap = 256;
    assert(involvedLevelCount <= kDoorLevelCap
        && "addExprToMemoryBlock: level run exceeds kDoorLevelCap");
    int doorLevels[kDoorLevelCap];
    int32_t doorLevelN = involvedLevelCount;
    for (int32_t li = 0; li < involvedLevelCount; ++li) doorLevels[li] = involvedLevels[li];
    OriginDep bridgeDeps[1 + ExecutionParameters::MAX_ARITY];
    TransientOrigin doorOrigin = origin;
    // A deposit the door REWRITES under an equality of this LB's own is this
    // LB's class product — an equality1 rewrite performed here — so it flips
    // to a LOCAL derivation whatever the arrival status: the canonical form
    // registers local (visible to the main-goal discharge's delta and to
    // fillMailOut) and the raw form's producer line ships to mailOut with it,
    // exactly the history an applyEquivalenceClass product leaves behind. A
    // rewrite licensed only by mailed equalities stays the sender's
    // knowledge and keeps the arrival's locality. A mailed fact that
    // canonicalizes onto a main goal's literal text under a local hypothesis
    // would otherwise sit in the registry as the sender's non-local row,
    // never discharge the goal, and Site-F-refuse the local derivation of
    // the same text.
    bool rewriteIsLocal = false;
    const unsigned doorStrSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : scratchArenas().slotCount() - 1;
    ScratchArena& doorStrArena = scratchArenas().forSlot(doorStrSlot);
    ScratchScope doorStrScope(doorStrArena);
    if (status != 2 && !parameters.skip_eq_classes
        && !canonicalizer.isEquality(expr) && !canonicalizer.isNegatedEquality(expr)) {
        RT_SCOPE_HERE("DOOR_CANONICALIZE");
        const CanonicalForm cf = canonicalizer.canonicalFormAtScope(
            expr, validityName, memoryBlock, doorStrArena,
            doorLevels, doorLevelN, kDoorLevelCap);
        if (cf.changed) {
            for (int32_t k = 0; k < cf.eqN; ++k) {
                if (cf.eqLocal[k]) rewriteIsLocal = true;
            }
            // Known canonical form -> the deposit ends here: no row, no
            // history line for the new raw spelling (nothing will cite
            // it), both ids carry the or-branch resolution signal.
            const NameId canonId = memoryBlock.nameMap.lookup(cf.text);
            if (!parameters.compressor_mode && canonId != 0
                && replayKinds == 0
                && gl::StatementKnowledge::ancestorKnown(memoryBlock, canonId, valId, /*includeSelf=*/true)) {
                caseProcessor.stageOrReleaseForRefusedDeposit(memoryBlock, origId, valId);
                caseProcessor.stageOrReleaseForRefusedDeposit(memoryBlock, canonId, valId);
                return;
            }
            const int originCap = parameters.compressor_mode
                ? parameters.compressor_max_origins_per_expr
                : parameters.max_origin_per_expr;
            if (parameters.trackHistory) {
                // A rewritten deposit without a producer line would leave
                // the bridge citing a form no chapter can resolve
                // (buildStack: no origin found) — surface the site here.
                assert(origin.present
                    && "addExprToMemoryBlock: canonical door rewrote a deposit that carries no origin line");
                assert(origin.tag != OriginTag::COUNT);
                addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner,
                    expr, validityName, origin.tag, origin.deps, origin.depN, originCap);
                // A canonical form that registers local mails; the raw form
                // never becomes a delta row, so its line ships from here (the
                // D-286 ship) whenever the registered row will be local.
                if (status == 0 || status == 1 || rewriteIsLocal) {
                    proofHistory.copyOriginRowsToMailOut(memoryBlock, expr, validityName, originCap);
                }
                if (!proofHistory.originRowExists(memoryBlock, cf.text, validityName)) {
                    const int bridgeN = 1 + cf.eqN;
                    assert(bridgeN <= 1 + ExecutionParameters::MAX_ARITY
                        && "addExprToMemoryBlock: bridge dep run exceeds cap");
                    bridgeDeps[0] = OriginDep{ expr, validityName };
                    for (int32_t bk = 0; bk < cf.eqN; ++bk) {
                        // The cite scope may be a NameMap decode; the mints
                        // below (canonical text, its argument ids) precede
                        // the bridge write, so copy it onto the string tier
                        // (I-3).
                        const StrSpan cite = proofHistory.findEqualityCiteScope(
                            memoryBlock, cf.eqJust[bk], validityName);
                        const ScratchString citeHold =
                            ScratchString::copyFrom(doorStrArena, cite.ptr, cite.len);
                        bridgeDeps[1 + bk] = OriginDep{ cf.eqJust[bk], StrSpan(citeHold) };
                    }
                    doorOrigin = TransientOrigin{
                        true, OriginTag::equality1, bridgeDeps, bridgeN };
                } else {
                    // Canonical target already documented — no second line
                    // (the I-34 gate); the door registers with no origin write.
                    doorOrigin = TransientOrigin{};
                }
            }
            expr = cf.text;
            origId = memoryBlock.nameMap.encode(expr);
        }
    }
    involvedLevels = doorLevels;
    involvedLevelCount = doorLevelN;

    // Passed fast checks — now do full encoding for axed var check + downstream.
    // Span twin: parse + encode in one pass off the stable expr/validityName
    // buffers, no intermediate EncodedExpression heap.
    IntEncodedExpr ie =
        encodeExpression(expr, validityName, memoryBlock.nameMap);
    if (registeredOut != nullptr) *registeredOut = ie;

    // Axed-variable containment: a deposit carrying an x-copy name in any
    // argument slot is dropped — the copies exist to serve as anchor-premise
    // completions, never to breed derived x-facts. Flag-gated exception
    // (D-234, parameters.axed_anchor_exception,
    // default off): a POSITIVE anchor-category statement passes — an
    // anchor-bridge rule firing on the x-copied batch anchor derives an
    // external anchor's x-form (e.g. (AnchorPeano[1,x2,3,4,5,x6]) from
    // (AnchorIncubator3[1,x2,3,4,5,x6,x7,8,9])), and that form must be a
    // live statement so external-anchor-premise rules can bind element
    // arguments at anchor slot values. Off by default because a live
    // x-anchor opens the whole external anchor's rule universe at
    // x-arguments and explodes the batch runtime. Negated anchor forms stay
    // filtered even when enabled: a derived negated x-anchor at main would
    // pair against the positive form as a spurious premise-inconsistency
    // (D-236).
    if (!(parameters.axed_anchor_exception && ie.isAnchor && !ie.negation)) {
        for (NameId a = 0; a < ie.arity; ++a) {
            if (memoryBlock.intAxedVariables.contains(ie.argFullId[a])) return;
        }
    }


    if (status == 4)
    {
        // A levels row is never empty and never mixes the non-derived tier
        // {-1} with real levels (the addStatement door rule; status 4
        // bypasses that door, so the invariant is pinned here too).
        assert(involvedLevelCount > 0
            && ((involvedLevelCount == 1 && involvedLevels[0] == -1)
                || involvedLevels[0] >= 0)
            && "status-4 fact load: invalid statement level run");
        memoryBlock.intLocalEncodedStatementsSet.mint(
            packStatementKey(ie.originalId, ie.validityId));
        memoryBlock.intEncodedStatements.push_back(ie);
        memoryBlock.intLocalEncodedStatements.push_back(ie);
        memoryBlock.intLocalEncodedStatementsDelta.push_back(ie);
        memoryBlock.intStatementLevelsMap.assignSetRange(
            packStatementKey(ie.originalId, ie.validityId),
            involvedLevels, involvedLevels + involvedLevelCount);
        upsertStatementKey(memoryBlock.intKnownStatements,
            packStatementKey(ie.originalId, ie.validityId),
            /*local=*/true);
        watchProcessor.recordNegationWatchFor(memoryBlock, ie);
        watchProcessor.stageWatchArrival(memoryBlock, ie);

        return;
    }

    // Past the span-native fast paths (Site F/H dedupe, axed-var scan, status-4).
    // The deeper deposit paths are span-native now: disintegrateExpr2 /
    // checkForEquivalence / checkNecessityForEquality / prepareIntegration /
    // updateAdmissionMap3 all take StrSpan, and the DisintProducts channels are
    // consumed directly -- no expr / validity std::string is materialized here.

    const NameId classValidityId = memoryBlock.nameMap.encode(validityName); // write site
    if (memoryBlock.equivalenceClassesMap.lookup(classValidityId) == 0)
    {
        // Open an empty bucket for this validity (the cold-map form of the
        // former `map[id] = {}`) so a never-filled scope still appears as a
        // 0-class row in the dump, exactly as the std::map seeding did.
        memoryBlock.assignClassesById(classValidityId, {});
        // eqClassSttmntIndexMapMap is flattened to a byte-key map
        // (bytes(validityId ++ memberIds) -> waterline); there is no
        // per-validity inner map to pre-create — entries install on first
        // upsert (the prior empty-map seeding + its assert had no flat analog).
    }


    // CE-filter contradiction detection is no longer gated here. A CE LB's
    // fired heads now enter `intEncodedStatements` like any derived statement (the
    // path below), so both the post-burst `dischargeContradiction` sweep and the
    // mid-burst read-only `burstDeactivates` predicate (prover.hpp) see them and
    // refute the conjecture — handled uniformly with the incubator and
    // vacuous-truth reactions.


    if (status == 2) {
        // Invariant: a toBeProved entry whose body args contain
        // it_/int_ names is class-rewritable by sanitizeToBeProved,
        // and class-rewritable goals never participate in induction
        // discharge. Therefore they must NOT carry an auxy index.
        // Auxies are reserved for induction-discharge bookkeeping
        // and the class-rewrite path bypasses that bookkeeping by
        // construction. Mixing the two means a sanitize pass would
        // rewrite a TBP body while a parallel auxy expects the
        // pre-rename form, breaking the induction discharge contract.
        {
            bool hasItInt = false;
            StrSpan assertArgs[ExecutionParameters::MAX_ARITY];
            const int assertArgN = getArgsSpans(expr, assertArgs, ExecutionParameters::MAX_ARITY);
            for (int ai = 0; ai < assertArgN; ++ai) {
                const StrSpan& a = assertArgs[ai];
                if (isDisintegrationWitnessShape(a) || isItLevShape(a)) {
                    hasItInt = true;
                    break;
                }
            }
            assert((!hasItInt || auxyIndex < 0) &&
                   "toBeProved entry with it_/int_ args must have empty auxies "
                   "(auxies are for induction discharge; class-rewrite path is "
                   "disjoint from induction).");
        }

        // Packed goal key — origId/valId were encoded at function entry, so
        // the goal write mints nothing new.
        const int64_t pkTBP = packStatementKey(origId, valId);

        const int32_t tbpExistingId = memoryBlock.intToBeProved.lookup(pkTBP);
        if (tbpExistingId != 0) {
            if (auxyIndex >= 0) {
                memoryBlock.intToBeProved.insertSorted(pkTBP, auxyIndex);
            }
        }
        else {
            if (auxyIndex >= 0)
                memoryBlock.intToBeProved.insertSorted(pkTBP, auxyIndex);
            else
                memoryBlock.intToBeProved.assignSet(pkTBP, nullptr, 0);

            {
                RT_SCOPE_HERE("DOOR_GOAL_CHECK_NECESSITY");
                // A new goal is visited against every chain.
                integrationProcessor.checkNecessityForEquality(expr, memoryBlock, validityName, 1);
            }
            // A goal shaped like a deferred OR head is an event of its
            // demand-side build (D-597).
            if (DemandBuiltRuleRegistry::instance().hasOrTrigger(expr, DemandBuiltTrigger::Goal)) {
                memoryBlock.pendingDeferredOrEvents.push_back(pkTBP);
            }
        }

        StrSpan argSpans[ExecutionParameters::MAX_ARITY];
        const int argSpanN = getArgsSpans(expr, argSpans, ExecutionParameters::MAX_ARITY);
        // Sorted-unique stack run over argSpans (ascending == the former std::set
        // order, the prepareIntegration span core's precondition), 0% heap.
        StrSpan unchRun[ExecutionParameters::MAX_ARITY];
        int32_t unchN = 0;
        for (int ai = 0; ai < argSpanN; ++ai) {
            const StrSpan v = argSpans[ai];
            int32_t lo = 0, hi = unchN; bool dup = false;
            while (lo < hi) {
                const int32_t mid = (lo + hi) / 2;
                const int c = compareSpans(unchRun[mid], v);
                if (c == 0) { dup = true; break; }
                if (c < 0) lo = mid + 1; else hi = mid;
            }
            if (dup) continue;
            assert(unchN < ExecutionParameters::MAX_ARITY
                && "prepareIntegration: unchangeable args exceed MAX_ARITY");
            for (int32_t z = unchN; z > lo; --z) unchRun[z] = unchRun[z - 1];
            unchRun[lo] = v; ++unchN;
        }
        {
            RT_SCOPE_HERE("DOOR_PREPARE_INTEGRATION");
            // The goal door: the prepared expression is the goal itself, so a
            // positive `=` goal is typed on the spot (I-267).
            integrationProcessor.prepareIntegration(expr, unchRun, unchN, memoryBlock, validityName, expr,
                               /*typedEqualityGoal=*/true);
        }
        const ce::CoreExpressionConfig* cfg = definitions.coreConfig(extractExpressionSpan(expr));

        if (cfg != nullptr && !cfg->inputIndices.empty()) {
            RT_SCOPE_HERE("DOOR_UPDATE_ADMISSION3");
            admissionMemory.updateAdmissionMap3(expr,
                memoryBlock,
                parameters.inductionMaxAdmissionDepth,
                parameters.inductionMaxSecondaryNumber,
                true);
        }
        return;
    }
    else {
        if (parameters.trackHistory && doorOrigin.present) {
            RT_SCOPE_HERE("DOOR_ORIGIN_WRITE");
            assert(doorOrigin.tag != OriginTag::COUNT);
            addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner, expr, validityName, doorOrigin.tag, doorOrigin.deps, doorOrigin.depN, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }

        // Statuses 0/1/3/5 reach here (status 2 returned above, status 4
        // returned earlier). Status 0/1 (local derivations) always
        // disintegrate. status 3 (external-mail / non-local absorb) is
        // admitted to disintegrateExpr2 ONLY for the RULE-CARRIER
        // shapes: an implication carrier (a compact "(implication<N>...)"
        // or a configured implication definition such as "(primeseq[...])")
        // and a negated compact existence "!(existence<N>...)" (a universal in
        // De-Morgan clothing — disintegrating it installs its universal
        // rule and mints NO witnesses; witness minting belongs to the
        // positive existence opening). A plain mail statement/fact must
        // NOT be disintegrated under status=3: disintegrating it mints
        // fresh it_/int_ names, which (a) overflows the static
        // MAX_NAME_IDS cap (prover.hpp assert) and (b) drives the
        // non-convergence runaway. Status 5 (flag-5 relay arrival,
        // D-284) IS admitted with witness minting: the
        // sender's disintegration parked this compact's products for lack
        // of admission demand, and this receiver may hold the demand. The
        // mint volume stays bounded — only parked compacts are relayed,
        // products remain demand-gated (Pass B parks undemanded ones
        // here exactly like local derivations), and a status-5
        // disintegration never re-stages a relay (no cascade). Without the negated-existence
        // admission, a descendant LB receives a universal premise (e.g.
        // an induction step sub-LB inheriting "7 has no predecessor")
        // as a bare statement whose rule can then never meet the
        // descendant-local facts it must refute — the ancestor's rule
        // copy cannot see child-local typings, mail flows only
        // parent->children (I-57). The guard is OUTSIDE
        // disintegrateExpr2 (at this call site), not at the
        // in-disintegrate variable-admission point.
        {
            // disintegrateExpr2's bool output, consumed after the stmts loop to
            // flag expr's intKnownStatements entry for cFE.
            bool fullDisintegrationHappened = false;

            // Implication rule carrier = a compact "(implication<N>[...])"
            // (the compileImplicationToCompact form) or a configured
            // definition whose whole body is one implication ("primeseq"):
            // that public name IS its rule's compact form, so an inherited
            // instance is the only carrier its rule has
            // (D-467).
            const bool isCompactImplication = gl::CompactExpressionSyntax::isCompactImplicationSpan(expr)
                || isConfiguredImplicationCarrier(expr);

            // Negated compact existence = literal "!(existence" followed by
            // a digit — the second rule-carrier shape admitted at status 3.
            const bool isNegatedExistence =
                expr.len > 11 && startsWithSpan(expr, "!(existence", 11)
                && expr.ptr[11] >= '0' && expr.ptr[11] <= '9';

            // Or-uniqueness gate: an or<N> deposit whose processing identity already
            // has a fully-processed representative at this scope stays a
            // PASSIVE statement — registered below and multiplied by
            // applyEquiClasses like any statement, but never disintegrated
            // (no K/subset-exclusion compacts, no cohort, no ordis park),
            // on every entry path (local and the flag-5 relay alike).
            // Ordered before checkForEquivalence so a suppressed OR skips
            // the general completed-expression lookup.
            const bool orCompactDeposit = gl::CompactExpressionSyntax::isOrCompactSpan(expr);
            const bool inheritedOrEligible = status == 3 && orCompactDeposit
                && caseProcessor.hasOrProofWork(memoryBlock);
            const bool statusAllowsDisintegration = status != 3
                || isCompactImplication || isNegatedExistence
                || inheritedOrEligible;
            bool orEquiDuplicate = false;
            if (orCompactDeposit && !doNotDisintegrate) {
                RT_SCOPE_HERE("DOOR_OR_EQUI_GATE");
                orEquiDuplicate =
                    caseLedger.orEquiRepresentativeRecorded(memoryBlock, expr, validityName);
            }
            // Suppress disintegration when an equivalence-class variant of
            // this expression is already a registered statement.
            bool cfeEq = false;
            if (!doNotDisintegrate && !orEquiDuplicate) {
                RT_SCOPE_HERE("DOOR_CHECK_EQUIVALENCE");
                if (ftaBlowupAudit) {
                    ftaBlowupAudit->status = status;
                    ftaBlowupAudit->disintegrationAllowed =
                        statusAllowsDisintegration;
                }
                cfeEq = equivalenceQueries.checkForEquivalence(expr, validityName, memoryBlock);
            }
            const bool willDisintegrate =
                !doNotDisintegrate && !orEquiDuplicate && (replayKinds != 0 || !cfeEq)
                && statusAllowsDisintegration;
            // Hoist the two return channels ABOVE the if/else so both branches
            // fill ONE `out` (per-slot arena, freed per task); consumed directly
            // below via forEachSorted (no std::set<EWV> edge-materialization).
            const unsigned deSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            DisintProducts out(&genScratchArenas().forSlot(deSlot));
            if (willDisintegrate)
            {
                // Ordinary inherited ORs require recipient proof work.
                // Other status-3 shapes remain limited to the two existing
                // rule carriers; local and flag-5 deposits retain their doors.
                assert(isCompactImplication || isNegatedExistence
                    || inheritedOrEligible
                    || status == 0 || status == 1 || status == 5);
                {
                    RT_SCOPE_HERE("DOOR_DISINTEGRATE");
                    fullDisintegrationHappened = disintegrationProcessor.disintegrateExpr2(expr,
                            memoryBlock,
                            iteration,
                            status == 0 || recursionHead,
                            validityName,
                            out,
                            /*trackHistoryLocal=*/true,
                            allowOrDisintegration,
                            involvedLevels, involvedLevelCount,
                            allowOrProbe, replayKinds, status == 5, replayIntegrationDepth,
                            recursionHead, integrationPremise, conjecturePremise);
                }
                // First representative: record the or's full processing so
                // later equi variants at this scope stay passive (the
                // or-uniqueness gate above).
                if (orCompactDeposit) {
                    caseLedger.recordProcessedOr(memoryBlock, origId, valId);
                }
                // Uniform admission — no not-self-returned exemption: a fully
                // disintegrated deposit (e.g. a broadcast theorem compact
                // decomposed into its rule) is OFFERED to the one admission
                // door like any statement. The channel dedups on the
                // composite key, so a self-returned expr makes this a no-op;
                // a not-self-returned expr joins the stmts loop below and is
                // admitted (row + levels) or refused by the door's gates —
                // never row-stamped outside admission. A PARTIALLY
                // disintegrated deposit is deliberately NOT offered: it must
                // stay re-processable (a row would Site-F-block the retry of
                // its unwitnessed existences).
                if (fullDisintegrationHappened) {
                    out.statements.append(expr, validityName);
                }
                // Flag-5 relay staging (D-284): the highest
                // uncovered existence groups' compacts await a flag-5 mail
                // deposit so a demand-holding descendant can disintegrate
                // them. LOCAL deposits only — a mailed carrier (status 3/5)
                // never re-stages (delivery already reached all descendants);
                // main scope only (I-26). The NameMap mint is single-threaded
                // per LB (worker claim / post-join drain).
                if ((status == 0 || status == 1)
                    && equalSpans(validityName, StrSpan("main", 4))) {
                    out.relayCompacts.forEachSorted(
                        [&](StrSpan rc, StrSpan rcValidity) {
                            (void)rcValidity;
                            memoryBlock.stagePendingRelay(
                                memoryBlock.nameMap.encode(rc), iteration,
                                involvedLevels, involvedLevelCount);
                        });
                    // Unconditional-ordis staging
                    // (D-528): an OR reached by
                    // unfolding a conjecture premise ships with flag 6, so
                    // every descendant with proof work opens its case split.
                    // Only the premise's own seeding (status 0) marks such ORs;
                    // a receiver never re-stages one.
                    out.unconditionalOrdis.forEachSorted(
                        [&](StrSpan orStmt, StrSpan orValidity) {
                            assert(equalSpans(orValidity, StrSpan("main", 4))
                                && "addExprToMemoryBlock: an unconditional-ordis OR is published at main");
                            memoryBlock.stageUnconditionalOrdis(
                                memoryBlock.nameMap.encode(orStmt));
                        });
                }
            }
            else
            {
                out.statements.append(expr, validityName);
            }

            {
            RT_SCOPE_HERE("DOOR_IMPLICATIONS_WALK");
            out.implications.forEachSortedWithSource([&](StrSpan impStr, StrSpan impValidity,
                                                         StrSpan carrier)
            {
                RT_SCOPE_HERE("DOOR_INSTALL_IMPLICATION");
                // The per-implication install — the canonical gate, the
                // status-selected instances, the D-274 row, both indexes —
                // lives in installExpandedImplication. The mail-out contract
                // stays MAIN-ONLY (I-26): a disintegration-recovered
                // implication is neither re-broadcast nor re-compacted here
                // (a locally derived one carries its compact form in
                // `expressions`, a mail-recovered one IS the statement being
                // disintegrated; recordPendingCompaction here would fan out).
                compactRuleProcessor.installExpandedImplication(memoryBlock, impStr, impValidity, carrier,
                    status, iteration, involvedLevels, involvedLevelCount,
                    expr, validityName);
            });
            } // RT_SCOPE DOOR_IMPLICATIONS_WALK

            if (out.implications.count() > 0)
            {
                RT_SCOPE_HERE("DOOR_GOALS_CHECK_NECESSITY");
                integrationProcessor.rescanGoalsAgainstNewChains(memoryBlock);
            }

            // Process statements — each carries its own validity name (normal
            // stmts have parent validity, OR branches have branch validity).
            RT_SCOPE_HERE("DOOR_STATEMENTS_WALK");
            out.statements.forEachSorted([&](StrSpan evOrig, StrSpan evValidity)
            {
                RT_SCOPE_HERE("DOOR_STATEMENT_ROW");
                // A replay does not change an already-known source carrier's locality.
                // Its newly materialized products still use ordinary local admission.
                if (replayKinds != 0 && equalSpans(evOrig, expr)
                    && equalSpans(evValidity, validityName)
                    && gl::StatementKnowledge::ancestorKnown(memoryBlock, origId, valId, true)) return;
                const unsigned foSlot = gl::WorkerContext::coreId() >= 0
                    ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
                ScratchArena& foArena = genScratchArenas().forSlot(foSlot);
                ScratchArena& foStrArena = scratchArenas().forSlot(foSlot);
                ScratchScope foScope(foArena);
                ScratchScope foStrScope(foStrArena);
                OriginDep frontDeps[64];
                TransientOrigin frontTO{};
                if (parameters.trackHistory) {
                    int64_t pkEv = 0;
                    const bool evHasKey = lookupOriginKey(memoryBlock.originInterner,
                        evOrig, evValidity, pkEv);
                    assert(evHasKey);
                    (void)evHasKey;
                    const int32_t oidOrigin = memoryBlock.exprOriginMap.lookup(pkEv);
                    assert(oidOrigin != 0 && memoryBlock.exprOriginMap.runLen(oidOrigin) > 0);
                    int32_t foBlen = 0;
                    const char* foBytes = memoryBlock.exprOriginMap.peekRecordBytes(
                        oidOrigin, 0, foBlen, foArena);
                    const IdOriginBlobView foView = viewIdOriginBlob(foBytes, foBlen);
                    int frontDepN = 0;
                    for (int32_t d = 0; d < foView.depN; ++d) {
                        assert(frontDepN < 64 && "front origin chain exceeds 64");
                        StrSpan eSpan, vSpan;
                        decodeOriginKeyCopy(idOriginDepAt(foView, d),
                            memoryBlock.originInterner, eSpan, vSpan, foStrArena);
                        frontDeps[frontDepN++] = { eSpan, vSpan };
                    }
                    frontTO = TransientOrigin{
                        true, static_cast<OriginTag>(foView.tag), frontDeps, frontDepN };
                }

                // `addStatement` is the single statement-add door.
                // It dispatches internally to `addEquality` /
                // `addNegatedEquality` for equality-shaped inputs and
                // sets the expression's `registered` membership bit.
                {
                    // Locality: local statuses register everything local.
                    // A flag-5 relay arrival (status 5) splits per statement
                    // (D-284): the CARRIER itself registers
                    // non-local — only local delta rows reach fillMailOut, so
                    // it is structurally never re-forwarded (delivery already
                    // reached every descendant) — while its disintegration
                    // products are THIS LB's own derivations (witnesses
                    // minted here) and register local, mailable and visible
                    // to the local-delta request batches.
                    const bool isCarrier = (status == 5)
                        && equalSpans(evOrig, expr)
                        && equalSpans(evValidity, validityName);
                    const bool isLocal = (status == 0 || status == 1)
                        || (status == 5 && !isCarrier)
                        || rewriteIsLocal;

                    // Id-form out-param on the per-slot gen-scratch arena —
                    // addStatement is void now (PagedVector is non-movable, cannot
                    // be returned). Page-tier buffer; coexists with addStatement's
                    // own gen-scratch usage on this slot (each tracks its own pages).
                    const unsigned addedSlot = (gl::WorkerContext::coreId() >= 0)
                        ? static_cast<unsigned>(gl::WorkerContext::coreId())
                        : genScratchArenas().slotCount() - 1;
                    ScratchArena& addedArena = genScratchArenas().forSlot(addedSlot);
                    DirtyState addedDirty = DirtyState::Clean;
                    PagedVector<IntEncodedExpr> added(&addedArena, &addedDirty);
                    {
                        RT_SCOPE_HERE("DOOR_ADD_STATEMENT");
                        addStatement(evOrig, memoryBlock, isLocal,
                            involvedLevels, involvedLevelCount, frontTO,
                            evValidity, added, out.orSources.contains(evOrig, evValidity), convergenceMailOut);
                    }

                    // Consume `added` in emission order and decode each row at
                    // the consumption edge. Its (original, validity) is copied into a per-row
                    // string-scratch ScratchString (byte-identical to nameMap.decode,
                    // on the string tier) rather than a heap std::string; the string
                    // tier is disjoint from the NameMap / interner cold tables the
                    // consumers mint into, so the StrSpan handed to each consumer
                    // survives those mints where a raw decodeView span would dangle
                    // (I-3 / I-138). The byte-bump merge window preserves the
                    // caller's scratch lifetime; each string window reclaims
                    // its two copies after the row's consumers finish.
                    ScratchScope addedMergeScope(addedArena);
                    // An induction admission serves a product of this
                    // disintegration only when the disintegrated original has
                    // numbers as its only arguments (I-285), none of them
                    // repeated (D-530); the
                    // deposit itself and a deposit that is not disintegrated
                    // propagate as before.
                    const bool inductionAllowed = !willDisintegrate
                        || (equalSpans(evOrig, expr) && equalSpans(evValidity, validityName))
                        || (hasOnlyNumberArguments(expr) && !hasRepeatedArgument(expr));
                    // In a block whose weak hypothesis install split identical
                    // goals by premise, the statement fill reads which
                    // decimal variables and forced-witness levels the arriving
                    // expression names (D-579)
                    // and whether its free arguments are all numbers, the gate
                    // of an int scope's start (D-578).
                    uint64_t arrivingVariables = 0;
                    uint64_t arrivingLevels = 0;
                    bool arrivingNumbersOnly = true;
                    // A copy bound to the second twin premise counts `rec2`,
                    // the name of the twin partner's predecessor, as its `rec`
                    // (D-589).
                    bool arrivingTwinNumbersOnly = true;
                    if (memoryBlock.premiseSplitFiled) {
                        signatureTokenSets(expr, arrivingVariables, arrivingLevels);
                        arrivingNumbersOnly = hasOnlyNumberFreeArguments(expr);
                        if (memoryBlock.twinPredecessorId != 0)
                            arrivingTwinNumbersOnly = hasOnlyNumberFreeArguments(expr,
                                memoryBlock.nameMap.decodeView(memoryBlock.twinPredecessorId));
                    }
                    // A product of a forced disintegration (a status-0 seed's
                    // or a hypothesis shot's) joins the block's forced-product
                    // list, whatever names it carries and whether or not it
                    // was known before, together with every row it
                    // registered; the disintegrated statement itself does not
                    // (D-522).
                    const bool forcedProduct = willDisintegrate && (status == 0 || recursionHead)
                        && !(equalSpans(evOrig, expr) && equalSpans(evValidity, validityName));
                    if (forcedProduct) {
                        const NameId productId = memoryBlock.nameMap.lookup(evOrig);
                        const NameId productScope = memoryBlock.nameMap.lookup(evValidity);
                        assert(productId != 0 && productScope != 0
                            && "addExprToMemoryBlock: addStatement encodes every product");
                        memoryBlock.lbMemory.forcedProducts.mint(
                            packStatementKey(productId, productScope));
                    }
                    const unsigned rowStrSlot = (gl::WorkerContext::coreId() >= 0)
                        ? static_cast<unsigned>(gl::WorkerContext::coreId())
                        : scratchArenas().slotCount() - 1;
                    ScratchArena& rowStrArena = scratchArenas().forSlot(rowStrSlot);
                    for (int32_t idx = 0; idx < added.size(); ++idx) {
                        const IntEncodedExpr& addedRow = added[idx];
                        if (forcedProduct)
                            memoryBlock.lbMemory.forcedProducts.mint(
                                packStatementKey(addedRow.originalId, addedRow.validityId));
                        ScratchScope rowStrScope(rowStrArena);
                        const StrSpan addOrigView =
                            memoryBlock.nameMap.decodeView(addedRow.originalId);
                        const ScratchString addExpression = ScratchString::copyFrom(
                            rowStrArena, addOrigView.ptr, addOrigView.len);
                        const StrSpan addValidView =
                            memoryBlock.nameMap.decodeView(addedRow.validityId);
                        const ScratchString effectiveValidity = ScratchString::copyFrom(
                            rowStrArena, addValidView.ptr, addValidView.len);
                        {
                            RT_SCOPE_HERE("DOOR_ADMISSION_INTEGRATION");
                            admissionMemory.updateAdmissionMapIntegration(StrSpan(addExpression), memoryBlock,
                                                          StrSpan(effectiveValidity), arrivingVariables, arrivingLevels,
                                                          arrivingNumbersOnly, arrivingTwinNumbersOnly);
                        }
                        // A row at or below the usefulness scope opens no
                        // induction admission (D-521).
                        if (!gl::ScopeSyntax::atOrBelowUsefulnessScope(memoryBlock.nameMap, addedRow.validityId)) {
                            RT_SCOPE_HERE("DOOR_ADMISSION_RECURSION");
                            admissionMemory.updateAdmissionMapRecursion(StrSpan(addExpression), memoryBlock,
                                                        StrSpan(effectiveValidity), inductionAllowed);
                        }

                        const int32_t addLvlsId = lookupStatementLevels(
                            memoryBlock.intStatementLevelsMap, memoryBlock.nameMap,
                            StrSpan(addExpression), StrSpan(effectiveValidity));
                        assert(addLvlsId
                               && memoryBlock.intStatementLevelsMap.runLen(addLvlsId) > 0
                               && "addStatement post-loop intStatementLevelsMap invariant violated");
                        int lvRun[256];
                        const int32_t lvN = coldIntRunAt(
                            memoryBlock.intStatementLevelsMap, addLvlsId,
                            lvRun, 256);

                        {
                            RT_SCOPE_HERE("DOOR_ORDIS_MERGE");
                            caseProcessor.ordisMerge(StrSpan(addExpression), StrSpan(effectiveValidity),
                                       lvRun, lvN, memoryBlock, convergenceMailOut);
                        }
                    }
                }

                // The disintegrateExpr2 `stmts` output is NOT staged into
                // intExternalStatements, which holds the raw mailIn
                // expressions 1:1 (filled in the absorb drain loop), not
                // the recovered/decomposed forms. The recovered stmts are
                // still installed into the LB by the addStatement loop
                // above.
            });

            // expr entered disintegrateExpr2 and came back fully disintegrated:
            // OR the `fullyDisintegrated` bit onto expr's registry row so cFE
            // suppresses future equivalence-class variants of expr (a
            // rejecting / partial twin stays unflagged and never suppresses —
            // the Gauss-fold mirror-bug fix). Bookkeeping on top of admission,
            // nothing more: the row exists iff the admission door created one
            // (expr rode the stmts loop above via the uniform-admission
            // offer); the marker never creates a row and never grants `known`
            // — a statement the door refused stays exactly as the door left
            // it.
            if (fullDisintegrationHappened) {
                const NameId markOrigId = memoryBlock.nameMap.encode(expr);
                const NameId markValId = memoryBlock.nameMap.encode(validityName);
                if (memoryBlock.intKnownStatements.find(
                        StatementKey{ markOrigId, markValId }) != nullptr) {
                    upsertStatementKey(memoryBlock.intKnownStatements,
                        packStatementKey(markOrigId, markValId),
                        /*local=*/false, /*fullyDisintegrated=*/true);
                }
            }

            // The induction hypothesis's missing expressions in the induction
            // scopes this disintegration's eats opened, filed now that its
            // products are in: a premise an eat published into its scope is
            // known there and files nothing
            // (D-575).
            out.hypothesisDemands.forEachSortedWithSource(
                [&](StrSpan premise, StrSpan scope, StrSpan source) {
                    admissionMemory.fileHypothesisOrDemand(memoryBlock, premise, scope,
                        memoryBlock.ruleInterner.lookup(source));
                });

            // The successor of every forced witness this disintegration
            // minted typed `(1)` in a strong step block, demanded at its
            // minting scope (D-571).
            out.successorDemands.forEachSorted([&](StrSpan witness, StrSpan scope) {
                admissionMemory.fileForcedSuccessorDemand(memoryBlock, witness, scope);
            });

        }
    }
}






/// @brief End-of-burst ancestor-known sweep: drop statement-LIST rows at
///        non-main scopes whose text a strict ancestor knows
///        (I-187).
///
/// @details
/// Detection first, mutation second: one read-only pass over
/// `intEncodedStatements` collects the packed `(originalId, validityId)`
/// keys of every non-main row whose text `ancestorKnown` reports at a
/// strict ancestor; the hits then drop in decoded-lex
/// `(original, validity)` order (I-84) through
/// `gl::StatementRegistryOperations::removeExpressionFromMemoryBlock(state=0)` — the statement lists only,
/// registry rows surviving as tombstones. The decodeView spans handed to
/// the removal door stay valid because the door only does non-minting
/// lookups and `PagedVector` erases — no NameMap mint (I-3). Scratch
/// (hit list on the page tier, sort index on the byte-bump tier) rides
/// the per-slot gen-scratch arena, reclaimed before return. Full
/// contract at the declaration.
///
/// @param body Owning LB; statement lists and waterlines mutated in place
///             through the removal door.
/// @invariant Registry rows are never erased here (I-58 / I-85).
/// @see sweepAncestorKnownRows (decl) — placement and tombstone rationale.
void StatementMemory::sweepAncestorKnownRows(Memory& body) {
    if (parameters.compressor_mode) return;
    const unsigned slot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(slot);
    const ArenaOffset sweepMark = gArena.cursor();
    {
        DirtyState hitsDirty = DirtyState::Clean;
        PagedVector<int64_t> hits(&gArena, &hitsDirty);
        {
            RT_SCOPE_HERE("SWEEP_SCAN");
            for (int32_t i = 0; i < body.intEncodedStatements.size(); ++i) {
                const NameId origId = body.intEncodedStatements[i].originalId;
                const NameId valId = body.intEncodedStatements[i].validityId;
                if (valId == NameMap::MAIN_ID) continue;
                if (gl::StatementKnowledge::ancestorKnown(body, origId, valId, /*includeSelf=*/false)) {
                    hits.push_back(packStatementKey(origId, valId));
                }
            }
        }
        const int32_t hitN = hits.size();
        if (hitN > 0) {
            RT_SCOPE_HERE("SWEEP_REMOVE");
            RT_NOTE_ITERATIONS_HERE(hitN);
            // One stable compaction pass per statement vector over the hit-key
            // set — the bulk twin of the per-hit `removeExpressionFromMemoryBlock`
            // (state 0) sequence: that sequence erased every row whose
            // (text, scope) pair is a hit from the same three vectors and
            // repaired the class waterlines after each hit; the repairs compose
            // (a waterline drops by the number of erased positions below it),
            // so one repair over all erased positions of the original
            // `intEncodedStatements` is identical, and the survivors keep their
            // order either way. The hit order of the retired per-hit walk
            // (text-lexicographic) had no observable effect and is not
            // reproduced.
            ColdHashSet<PodKeyStore<int64_t>> hitSet(&gArena, &hitsDirty);
            hitSet.resetToFresh();
            for (int32_t k = 0; k < hitN; ++k) hitSet.mint(hits[k]);
            const auto isHit = [&](const IntEncodedExpr& row) {
                return hitSet.lookup(packStatementKey(row.originalId,
                                                      row.validityId)) != 0;
            };
            body.intLocalEncodedStatements.eraseIf(isHit, [](int32_t) {});
            body.intLocalEncodedStatementsDelta.eraseIf(isHit, [](int32_t) {});
            PagedVector<int32_t> erasedPos(&gArena, &hitsDirty);
            const int32_t erasedN = body.intEncodedStatements.eraseIf(
                isHit, [&](int32_t pos) { erasedPos.push_back(pos); });
            if (erasedN > 0) gl::StatementRegistryOperations::repairEqClassWaterlines(body, erasedPos, erasedN);
            // Quiescence (D-194): removals can cancel a genuine addition in
            // the count diff — flag the mutation directly.
            body.mutatedThisBurst = true;
        }
    }
    gArena.popTo(sweepMark);
}
/// @brief Whether a span is a positive instance of a configured
        ///        definition whose compiled category is `implication`.
        ///
        /// @details
        /// A configured definition whose whole body is one implication
        /// (`primeseq`) is itself the compact form of its rule: the public
        /// name replaces any numbered equivalent, so the block that expands
        /// it has no separate `implication<N>` compact to mail. An inherited
        /// statement of this shape is therefore a rule carrier like
        /// `(implication<N>[…])`, and the deposit door expands it
        /// (D-467). "Configured" is
        /// `expressionsFromConfig` membership, filled before generated
        /// compacts extend `coreExpressionMap`; numbered compacts are matched
        /// by `isCompactImplicationSpan`. Zero allocation: the name is a span
        /// of the input, both probes are transparent lookups.
        ///
        /// @param s The deposited expression.
        /// @return `true` exactly for `(name[…])` with `name` configured and
        ///         compiled as category `implication`; an atomic configured
        ///         operator has no compiled entity and is no carrier.
        /// @see `isCompactImplicationSpan`, `addExprToMemoryBlock`.
        bool StatementMemory::isConfiguredImplicationCarrier(StrSpan s) const {
            return definitions.isConfiguredImplication(s);
        }

}
