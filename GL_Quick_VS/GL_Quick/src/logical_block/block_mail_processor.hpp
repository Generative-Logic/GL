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
#include "../project_paths.hpp"
#include "../infra/hashburst_dump.hpp"
#include "../memory.hpp"
#include "../filter.hpp"
#include "../compilation_state.hpp"
#include "../compiler/expression_parser.hpp"
#include "../compiler/compilation_normalizer.hpp"
#include "../compiler/compilation_persistence.hpp"
#include "../compiler/compiled_definition_registry.hpp"
#include "../inference/hash_key_builder.hpp"
#include "../inference/implication_multiplier.hpp"
#include "../inference/binding_partitions.hpp"
#include "../inference/hash_rule_index.hpp"
#include "../inference/hash_rule_classifier.hpp"
#include "../inference/hash_request_generator.hpp"
#include "../inference/hash_rule_evaluator.hpp"
#include "../inference/burst_sink.hpp"
#include "../logical_block/logic_block_identity.hpp"
#include "../logical_block/statement_knowledge.hpp"
#include "../logical_block/deferred_ancestor_admission.hpp"
#include "../logical_block/update_global_single_chapter_record.hpp"
#include "../logical_block/or_implication_refuter.hpp"
#include "../logical_block/or_implication_selection.hpp"
#include "../logical_block/watch_bridge.hpp"
#include "../logical_block/canonical_form_cache.hpp"
#include "../logical_block/compact_canon_pair.hpp"
#include "../logical_block/numeric_less.hpp"
#include "../logical_block/hypo_scope_parse.hpp"
#include "../logical_block/implication_subproof_outcome.hpp"
#include "../logical_block/collected_record.hpp"
#include "../logical_block/collected_arena.hpp"
#include "../logical_block/new_variable_store.hpp"
#include "../logical_block/witness_meta_store.hpp"
#include "../logical_block/rejection_record.hpp"
#include "../logical_block/rejection_store.hpp"
#include "../logical_block/scanned_token_runs.hpp"
#include "../logical_block/equivalence_span_pair.hpp"
#include "../logical_block/equivalence_rewrite.hpp"
#include "../logical_block/admission_policy.hpp"
#include "../logical_block/scope_type_resolver.hpp"
#include "../logical_block/canonicalizer.hpp"
#include "../logical_block/canonical_template_matcher.hpp"
#include "../logical_block/rejected_source_memory.hpp"
#include "../logical_block/integration_instruction_builder.hpp"
#include "../logical_block/unary_rule_memory.hpp"
#include "../logical_block/admission_marker_builder.hpp"
#include "../compiler/existence_rule_compiler.hpp"
#include "../compiler/implication_builder.hpp"
#include "../compiler/or_rule_compiler.hpp"
#include "../compiler/structural_compiler.hpp"
#include "../compiler/supporting_rule_compiler.hpp"
#include "../compiler/compilation_reuse.hpp"
#include "../compiler/definition_expansion.hpp"
#include "../proof/worker_context.hpp"
#include "../infra/rt_tracker.hpp"
#include "../infra/hashburst_dump.hpp"
#include "../memory_infra/steward.hpp"
#include "../memory_infra/lb_store.hpp"
#include "../mail_log.hpp"
#include "../simple_map_store.hpp"
#include "../memory_infra/or_implication_match_index.hpp"
#include "../memory_infra/temporary_witness_store.hpp"
#include "phase13_timing_slot.hpp"
#include "watch_ledger.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_set>
#include <utility>
#include <vector>
namespace gl {
class ProofHistory;
namespace tests { struct BlockMailProcessorFixture; }
/// @brief Block Mail Processor operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class BlockMailProcessor {
friend struct tests::BlockMailProcessorFixture;
public:
/// @brief Bind inputs for block mail processor.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param blockIdentity Borrowed LogicBlockIdentity interface or state; must outlive this component.
/// @param body Borrowed Memory interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param mailLog Borrowed MailLog interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param permanentBodies Borrowed std::vector interface or state; must outlive this component.
/// @param simpleMapStore Borrowed SimpleMapStore interface or state; must outlive this component.
/// @param structuralCompiler Borrowed StructuralCompiler interface or state; must outlive this component.
/// @param proofHistory Borrowed ProofHistory interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
BlockMailProcessor(LogicBlockIdentity& blockIdentity, Memory& body, const CompiledDefinitionRegistry& definitions, MailLog& mailLog, const ProverParameters& parameters, std::vector<Memory*>& permanentBodies, SimpleMapStore& simpleMapStore, StructuralCompiler& structuralCompiler, ProofHistory& proofHistory);

public:

inline void fillMailOut(Memory& memoryBlock);

private:
// Turn optimizations OFF for just this section
//#pragma optimize("", off)
/// @brief Test a circulating expression against the witness-class mail boundary.
///
/// @details Any ordinary integration witness refuses statement or compact-rule
/// mail, regardless of depth or owning block. Forced and algebraic names do not
/// refuse it; the caller still enforces validity and routing requirements.
/// Proof-history and internal replay channels do not use this restriction.
/// @param expression Complete expression text, including prefixed arguments.
/// @param body Owning block, retained in the shared call interface.
/// @return True exactly when no ordinary integration witness occurs.
/// @invariant Pure lexical policy; no permission store or origin-map read.
bool allowedForMail(StrSpan expression, const Memory& body);

public:
/// @brief Compile a batch of established theorems to compact rule
///        carriers and store the batch ONCE in `store`'s mail log.
///
/// @details
/// Load-time (and between-warm-up-and-main) seeding of an LB grid with
/// theorems proved by earlier batches or supplied externally. Every
/// theorem is disintegrated, rebuilt in first-occurrence binder order
/// (I-4), compiled to its `(implication<N>[...])` compact, and deposited
/// as a main-scope mail statement with an empty level set (I-51) plus
/// its `originTag` / `compilation` origin rows. The batch is committed
/// into `store`'s `MailLog` chain — every LB that lists `store` among
/// its mail ancestors pulls it on its next phase-1 walk — and
/// self-injected into `store.mailIn` (an LB never pulls its own log),
/// with `store.hasWork` raised (wake door, I-153). Single-threaded,
/// before any pull.
///
/// Which LBs receive the batch is decided by `store` together with
/// the mail-edge registration (`collectMailAncestors`): the root
/// sentinel's log reaches every LB outside incubator mode and only the
/// anchor LB inside it (D-332);
/// the anchor LB's log reaches the anchor LB and every LB below it.
///
/// @param provedTheorems The theorems to seed (raw MPL, structural
///        operators already precompiled by the caller).
/// @param originTag Origin tag of the seed rows (`"broadcast"` for
///        theorems proved by earlier batches, `"externally provided
///        theorem"` for the hand-written pool).
/// @param store The LB whose log stores the batch and whose inbox is
///        self-injected: the root sentinel (`body`) or, for the
///        externals of an incubator batch, the anchor LB.
/// @param nestedStore Shape routing (incubator mode, the prior-theorems
///        batch): when non-null, a theorem that `anchorOnlyRule`
///        rejects — a nested rule whose premise LB lies below the
///        anchor — is stored in `nestedStore`'s log (the anchor LB's,
///        read by every LB below it) instead of `store`'s. Null = no
///        split (every theorem goes to `store`).
/// @invariant The batch's level sets are empty (I-51); only
///            `statements` + `exprOriginMap` are stored (D-137 rule 3).
///            Compacts are compiled in list order regardless of the
///            split (the `implication<N>` numbering is observable).
/// @see collectMailAncestors, incubatorAnchorLb, anchorOnlyRule,
///      MailLog::commit, mergeBatchIntoMailIn.
void broadcastTheorems(const std::vector<std::string>& provedTheorems,
                               const std::string& originTag,
                               Memory& store,
                               Memory* nestedStore);

public:
/// @brief Collect the mail ancestors an LB registers with `MailLog`
///        (the logs it pulls from), nearest first.
///
/// @details
/// Outside incubator mode this is the whole `parentMemory` chain up to
/// and including the root sentinel (I-57: an LB pulls from every direct
/// ancestor). In incubator mode the walk STOPS at the anchor LB: an LB
/// strictly below the anchor LB lists its ancestors up to and including
/// the anchor LB and never the root sentinel, so everything the root's
/// log carries — the in-run compact rule broadcasts, their
/// `"theorem"`-origin rows, the load-time batch of theorems proved by
/// earlier incubator batches — reaches only the anchor LB
/// (D-332). The anchor LB itself
/// keeps its root edge; the root sentinel has no ancestors.
///
/// @param lb  The LB being registered.
/// @param out Receives the ancestor list, nearest first (cleared first).
/// @invariant In incubator mode every non-root LB has the anchor LB on
///            its chain (every incubator conjecture starts with the
///            anchor premise); a chain without it asserts.
/// @see isAnchorLb, MailLog::registerLb, broadcastTheorems.
void collectMailAncestors(const Memory* lb,
                                  std::vector<const Memory*>& out) const;

public:
/// @brief The anchor LB of an incubator grid — the root sentinel's one
///        child whose key carries the anchor prefix.
///
/// @details
/// Every incubator conjecture has the anchor as its first premise, so the
/// root sentinel has exactly one anchor child and every other LB hangs
/// below it. The walk over the root's children asserts exactly one
/// anchor LB (Rule 19). The result is the `store` of the externals seed
/// batch in incubator mode (`broadcastTheorems`).
///
/// @return The anchor LB (never null).
/// @invariant Callable only in incubator mode, after the grid is built.
/// @see isAnchorLb, broadcastTheorems, collectMailAncestors.
Memory* incubatorAnchorLb() const;

public:
/// @brief True iff an incubator-derived theorem's rule can only fire
///        at the anchor LB, so its compact need not travel below it.
///
/// @details
/// The shape rule of D-332
/// (maintainer-directed). Anchor-only:
/// - a single-premise theorem `anchor → head` whose head is not a
///   positive anchor;
/// - a two-premise theorem whose second premise is an operator
///   application and whose head is an equality connecting one of that
///   operator's bound (non-anchor) arguments to an anchor digit — an
///   anchor slot value (the slots an equality can mention are the
///   numerals), e.g. `anchor → (in2[10,x,3] → (=[x,11]))`: the
///   operator premise is derivable at the anchor LB, so the rule
///   fires there.
/// Positive anchor-to-anchor bridges are NOT anchor-only: copied
/// source anchors exist in descendant LBs, so those LBs must receive
/// the bridge rule and derive the copied target by ordinary inference.
/// Every other nested shape (`anchor → (A → B)` with a premise that is
/// local only in a premise LB below the anchor, three or more premises)
/// is NOT anchor-only: its compact is stored in the anchor LB's log so
/// the premise LBs below the anchor receive it. Heap parse
/// (`ce::disintegrateImplication`), single-threaded seams only (load,
/// the post-join compaction drain).
///
/// @param theorem The theorem text (raw MPL, structural operators
///        precompiled).
/// @return True iff the rule is anchor-only.
/// @invariant The theorem's first premise is an anchor predicate —
///            this batch's, or an earlier incubator batch's for a
///            loaded theorem (bridged by the cross-anchor conjecture
///            `current anchor → previous anchor`); anything else
///            asserts.
/// @see broadcastTheorems, the deferred-compaction drain in
///      `proveKernel`.
bool anchorOnlyRule(const std::string& theorem) const;
private:

LogicBlockIdentity& blockIdentity;
Memory& body;
const CompiledDefinitionRegistry& definitions;
MailLog& mailLog;
const ProverParameters& parameters;
std::vector<Memory*>& permanentBodies;
SimpleMapStore& simpleMapStore;
StructuralCompiler& structuralCompiler;
ProofHistory& proofHistory;
};
}
#include "../logical_block/proof_history.hpp"
namespace gl {


inline void BlockMailProcessor::fillMailOut(Memory& memoryBlock) {
            const int maxOrigins = parameters.compressor_mode
                ? parameters.compressor_max_origins_per_expr
                : parameters.max_origin_per_expr;

            // Flag-5 relay bookkeeping for this pass: delta rows that shipped
            // with the relay flag, and pending compacts direct-deposited
            // below; both erase from the staging maps after the loops.
            // Arena-backed (a mass firing burst can stage hundreds of relays
            // at one LB in one pass — every undemanded fired existence head
            // is a relay candidate, not just the goal chain's).
            const unsigned frSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& frArena = genScratchArenas().forSlot(frSlot);
            DirtyState frDirty = DirtyState::Clean;
            PagedVector<int32_t> relayShippedKeys(&frArena, &frDirty);
            PagedVector<NameId> ordisShippedKeys(&frArena, &frDirty);

            RT_NOTE_ITERATIONS_HERE(
                static_cast<int>(memoryBlock.intLocalEncodedStatementsDelta.size()));
            for (int32_t di = 0;
                 di < memoryBlock.intLocalEncodedStatementsDelta.size();
                 ++di) {
                RT_SCOPE_HERE("FMO_DELTA_ROW");
                const IntEncodedExpr& ie =
                    memoryBlock.intLocalEncodedStatementsDelta[di];

                // Decoded value — zero-copy decodeView spans over the delta
                // row's ids. Nothing in this loop body mints into NameMap
                // (allowedForMail is non-minting; the mail/origin inserts write
                // mailOut by id / pkEv), so the spans stay valid (no I-3 copy).
                const StrSpan origView =
                    memoryBlock.nameMap.decodeView(ie.originalId);
                const StrSpan valView =
                    memoryBlock.nameMap.decodeView(ie.validityId);

                // Split-gate (I-26): the statement insert is MAIN-ONLY and
                // gated on `allowedForMail` (every mailOut.statements.insert
                // sits behind the same pair). The origin copy is
                // UNCONDITIONAL AND ALL-SCOPES — addOrigin writes to
                // `mailOut.exprOriginMap` carry neither gate; the receiver
                // must always be able to walk past a filtered dep. The
                // allowedForMail case: `(in3[2,2,int_0_lev_0_7,4])` is a
                // producer-local single-`force_lev_*` / `int_<depth>_lev_*` atomic that fails
                // `allowedForMail` but is cited as the source of an
                // `equality1` rewrite shipped via this LB's delta. The
                // scope case: a main-scope refutation record (the
                // ordisMerge branch-death deposit) cites branch-scope
                // statements as its contradiction antecedents — the record
                // rides the statements channel at main, so the cited
                // branch-scope keys' own origin rows must ride too, or
                // every receiver that renders a chapter through the record
                // crashes `buildStack` on the missing dep.

                // "main" is pre-registered as NameMap::MAIN_ID by the NameMap
                // constructor, so the scope gate is a pure id compare.
                if (ie.validityId == NameMap::MAIN_ID
                    && allowedForMail(origView, memoryBlock)) {
                    // Copy levels — every delta entry has a level-index
                    // entry by contract; packed probe via the delta row's ids.
                    const int32_t mailLvId = memoryBlock.intStatementLevelsMap.lookup(
                        packStatementKey(ie.originalId, ie.validityId));
                    assert(mailLvId != 0
                           && memoryBlock.intStatementLevelsMap.runLen(mailLvId) > 0
                           && "intLocalEncodedStatementsDelta / intStatementLevelsMap invariant violated");
                    // mailOut mints both spans into its private deloadable
                    // interner; the commit seam translates those ids to global.
                    int lvRun[256];
                    const int32_t lvN = coldIntRunAt(
                        memoryBlock.intStatementLevelsMap, mailLvId, lvRun, 256);
                    memoryBlock.insertMailOutStatement(origView, valView,
                        lvRun, lvN);

                    // Flag-5 relay: a shipped delta statement staged in
                    // pendingRelay carries the relay flag + generation stamp
                    // on its deposit (D-284); erased below.
                    const int32_t prId = memoryBlock.pendingRelayIter.lookup(
                        static_cast<int32_t>(ie.originalId));
                    if (prId != 0) {
                        memoryBlock.setMailOutStatementFlag(origView, valView,
                            kMailStatementFlagExternalDisintegrate,
                            memoryBlock.pendingRelayIter.valueAt(prId));
                        relayShippedKeys.push_back(
                            static_cast<int32_t>(ie.originalId));
                    }
                    // Unconditional ordis
                    // (D-528): an OR reached
                    // by unfolding a conjecture premise ships with flag 6;
                    // every descendant with proof work opens its case
                    // split. Erased below.
                    if (memoryBlock.lbMemory.pendingUnconditionalOrdis.lookup(ie.originalId) != 0) {
                        memoryBlock.setMailOutStatementFlag(origView, valView,
                            kMailStatementFlagUnconditionalOrdis, -1);
                        ordisShippedKeys.push_back(ie.originalId);
                    }
                    // From premise (D-522): a
                    // product of this block's forced disintegrations ships
                    // marked, so every receiver lists it among its own
                    // forced products.
                    if (memoryBlock.lbMemory.forcedProducts.lookup(
                            packStatementKey(ie.originalId, ie.validityId)) != 0)
                        memoryBlock.setMailOutStatementFlag(origView, valView,
                            kMailStatementFlagNone, -1, true);
                }

                // Origin copy — unconditional (ungated addOrigin to
                // mailOut.exprOriginMap).
                {
                    RT_SCOPE_HERE("FMO_COPY_ORIGIN_ROWS");
                    proofHistory.copyOriginRowsToMailOut(memoryBlock, origView, valView,
                        maxOrigins);
                }
            }

            // Flag-5 relay, direct-deposit half: staged compacts that did NOT
            // ship through the delta this pass (unregistered nested compacts,
            // or registered ones whose gate qualified only later). Walk a
            // decoded-lex snapshot (determinism), gate each on the shape gate,
            // deposit with the staged level run + flag + origin mirror, and
            // erase on success; refused candidates stay staged and re-try
            // next pass.
            if (memoryBlock.pendingRelayIter.count() > relayShippedKeys.size()) {
                RT_SCOPE_HERE("FMO_RELAY_DIRECT_DEPOSIT");
                const int32_t deltaShippedN = relayShippedKeys.size();
                PagedVector<int32_t> prKeys(&frArena, &frDirty);
                const int32_t total = memoryBlock.pendingRelayIter.count();
                for (int32_t id = 1; id <= total; ++id) {
                    const int32_t key = memoryBlock.pendingRelayIter.decodeKey(id);
                    bool shippedNow = false;
                    for (int32_t s = 0; s < deltaShippedN; ++s)
                        if (relayShippedKeys[s] == key) { shippedNow = true; break; }
                    if (!shippedNow) prKeys.push_back(key);
                }
                // Decoded-lex order over the expression bytes (decodeView is
                // non-minting; deposits mint only the private mail interner).
                // Sort an index run on the byte-bump tier (the PagedVector's
                // pages are not contiguous).
                const int32_t prN = prKeys.size();
                ScratchScope prScope(frArena);
                int32_t* prOrder = (prN > 0)
                    ? reinterpret_cast<int32_t*>(frArena.resolve(
                          frArena.alloc(prN * 4, 4)))
                    : nullptr;
                for (int32_t i = 0; i < prN; ++i) prOrder[i] = prKeys[i];
                std::sort(prOrder, prOrder + prN, [&](int32_t a, int32_t b) {
                    return compareSpans(
                        memoryBlock.nameMap.decodeView(static_cast<NameId>(a)),
                        memoryBlock.nameMap.decodeView(static_cast<NameId>(b))) < 0;
                });
                for (int32_t i = 0; i < prN; ++i) {
                    const StrSpan rcView = memoryBlock.nameMap.decodeView(
                        static_cast<NameId>(prOrder[i]));
                    if (!allowedForMail(rcView, memoryBlock)) continue;
                    int32_t iterOf = 0;
                    {
                        const int32_t vid = memoryBlock.pendingRelayIter.lookup(
                            prOrder[i]);
                        assert(vid != 0);
                        iterOf = memoryBlock.pendingRelayIter.valueAt(vid);
                    }
                    int lvRun[256];
                    int32_t lvN = 0;
                    const int32_t lvId = memoryBlock.pendingRelayLevels.lookup(
                        prOrder[i]);
                    if (lvId != 0) {
                        lvN = memoryBlock.pendingRelayLevels.runLen(lvId);
                        assert(lvN <= 256);
                        for (int32_t j = 0; j < lvN; ++j)
                            lvRun[j] = memoryBlock.pendingRelayLevels.valueAt(
                                lvId, j);
                    }
                    memoryBlock.insertMailOutStatement(rcView,
                        StrSpan("main", 4), lvRun, lvN);
                    memoryBlock.setMailOutStatementFlag(rcView,
                        StrSpan("main", 4),
                        kMailStatementFlagExternalDisintegrate, iterOf);
                    proofHistory.copyOriginRowsToMailOut(memoryBlock, rcView,
                        StrSpan("main", 4), maxOrigins);
                    relayShippedKeys.push_back(prOrder[i]);
                }
            }
            for (int32_t s = 0; s < relayShippedKeys.size(); ++s)
                memoryBlock.erasePendingRelay(
                    static_cast<NameId>(relayShippedKeys[s]));
            for (int32_t s = 0; s < ordisShippedKeys.size(); ++s)
                memoryBlock.eraseUnconditionalOrdis(ordisShippedKeys[s]);
        }
}
