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
#include "../logical_block/admission_memory.hpp"
#include "../logical_block/integration_processor.hpp"
#include "../logical_block/implication_subproofs.hpp"
#include "../logical_block/rejected_source_replay.hpp"
#include "../inference/hash_rule_installer.hpp"
#include "../logical_block/compact_rule_processor.hpp"
#include "../logical_block/statement_memory.hpp"
#include "../logical_block/equivalence_processor.hpp"
#include "../logical_block/equality_propagation.hpp"
#include "../logical_block/equivalence_admission_rewriter.hpp"
#include "../logical_block/watch_processor.hpp"
#include "../logical_block/case_ledger.hpp"
#include "../logical_block/case_processor.hpp"
#include "../logical_block/disintegration_processor.hpp"
#include "../logical_block/or_implication_processor.hpp"
#include "../logical_block/proof_history.hpp"
#include "../logical_block/unary_integration_processor.hpp"
#include "../logical_block/goal_processor.hpp"
#include "../logical_block/scope_lifecycle.hpp"
#include "../logical_block/logic_block_operations.hpp"
#include "../logical_block/block_mail_processor.hpp"
#include "../logical_block/anchor_scope_queries.hpp"
#include "../logical_block/equivalence_queries.hpp"
#include "../logical_block/block_key_reader.hpp"
#include "../logical_block/phase13_timing_slot.hpp"
#include "../logical_block/watch_ledger.hpp"
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
#include "../proof/or_candidate.hpp"
#include "../infra/rt_tracker.hpp"
#include "../infra/hashburst_dump.hpp"
#include "../memory_infra/steward.hpp"
#include "../memory_infra/lb_store.hpp"
#include "../mail_log.hpp"
#include "../simple_map_store.hpp"
#include "../memory_infra/or_implication_match_index.hpp"
#include "../memory_infra/temporary_witness_store.hpp"
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
class BlockMailProcessor;
class InductionUsefulness;
class OrTheoremBuilder;
class TheoremPublisher;
namespace tests { struct ProofGridBuilderFixture; }
/// @brief Proof Grid Builder operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class ProofGridBuilder {
friend struct tests::ProofGridBuilderFixture;
public:
/// @brief Bind the proof grid builder operation inputs.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param admissionMemory Borrowed AdmissionMemory interface or state; must outlive this component.
/// @param anchorInfo Borrowed ce::AnchorInfo interface or state; must outlive this component.
/// @param anchorScopeQueries Borrowed AnchorScopeQueries interface or state; must outlive this component.
/// @param blockIdentity Borrowed LogicBlockIdentity interface or state; must outlive this component.
/// @param body Borrowed Memory interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param hashRuleInstaller Borrowed HashRuleInstaller interface or state; must outlive this component.
/// @param inductionUsefulness Borrowed InductionUsefulness interface or state; must outlive this component.
/// @param integrationProcessor Borrowed IntegrationProcessor interface or state; must outlive this component.
/// @param lbStore Borrowed LbStore interface or state; must outlive this component.
/// @param mailLog Borrowed MailLog interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param permanentBodies Borrowed std::vector interface or state; must outlive this component.
/// @param scopeTypes Borrowed ScopeTypeResolver interface or state; must outlive this component.
/// @param simpleMapStore Borrowed SimpleMapStore interface or state; must outlive this component.
/// @param statementMemory Borrowed StatementMemory interface or state; must outlive this component.
/// @param theoremPublisher Borrowed TheoremPublisher interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
ProofGridBuilder(AdmissionMemory& admissionMemory, const ce::AnchorInfo& anchorInfo, AnchorScopeQueries& anchorScopeQueries, LogicBlockIdentity& blockIdentity, Memory& body, const CompiledDefinitionRegistry& definitions, HashRuleInstaller& hashRuleInstaller, InductionUsefulness& inductionUsefulness, IntegrationProcessor& integrationProcessor, LbStore& lbStore, MailLog& mailLog, const ProverParameters& parameters, std::vector<Memory*>& permanentBodies, ScopeTypeResolver& scopeTypes, SimpleMapStore& simpleMapStore, StatementMemory& statementMemory, TheoremPublisher& theoremPublisher);

public:
// … no other prove() overloads …
/// @brief The arguments a weak induction hypothesis on @p digit keeps fixed.
///
/// @details
/// Starts from the conjecture's digit arguments without @p digit and without
/// @p freeDigits, then closes forward over the premise chain: an operator
/// premise whose configured inputs are all fixed fixes its configured outputs.
/// The head takes no part. A twin partner of @p digit is passed in
/// @p freeDigits, so the hypothesis is universal in it and in every output it
/// alone would have fixed (D-587).
///
/// @param theorem    The canonical compiled conjecture.
/// @param digit      The induction variable.
/// @param freeDigits Digit arguments the hypothesis leaves universal.
/// @return The fixed arguments, ascending.
/// @invariant Reads the compiled configuration only; writes nothing.
/// @see createAuxyImplication, twinInductionPartners.
std::set<std::string>
            findImmutableArgs(const std::string& theorem,
                const std::string& digit,
                const std::set<std::string>& freeDigits = std::set<std::string>());

/// @brief The partner of each induction variable of a conjecture with twin
///        premises (D-587).
///
/// @details
/// Two non-anchor premises are twins when they are the only two premises of
/// one predicate and polarity, no other predicate repeats, and the conjecture
/// is symmetric in them: exchanging the two premises gives the same lemma
/// after normalization (`canonicalConjecture`: first-occurrence numbering,
/// every variable bound at its first premise). An equality head, positive or
/// negated, reads unordered, so the exchanged lemma may also match with the
/// head's two arguments swapped.
///
/// A digit variable k of the first twin has the partner l when the second
/// twin holds l at the same argument slot, l is a digit variable, l differs
/// from k, k does not occur in the second twin and l does not occur in the
/// first. Exchanging the twins maps an induction on l onto the induction on
/// k, so the grid builds triads on k only and its weak hypothesis leaves l
/// universal. FTA2's full I5 `primeFactorSeq(k,P,n) ∧ primeFactorSeq(l,Q,n)
/// → P = Q` gives k ↦ l; the equal-length form, with k in both premises,
/// gives nothing.
///
/// @param compiled A conjecture after
///                 `StructuralCompiler::precompileStructuralOperators`
///                 (every premise and the head a simple, possibly negated,
///                 atom; every argument bound).
/// @param digits   The conjecture's digit arguments (`findDigitArgs`).
/// @param digitN   Number of @p digits.
/// @return Map from each first-twin induction variable to its partner; empty
///         when the conjecture has no twin premises or the twins hold no
///         partner pair.
/// @invariant Pure: reads the compiled configuration only, writes nothing.
///            Twins of one predicate have one arity (asserted).
/// @see addTheoremToMemory, findImmutableArgs, canonicalConjecture.
std::map<std::string, std::string> twinInductionPartners(const std::string& compiled,
            const StrSpan* digits, int32_t digitN) const;

public:
/// @brief The canonical spelling of a compiled conjecture: its variables
///        renumbered in first-occurrence order and each bound at its first
///        premise (D-493).
///
/// @details
/// A compiled implication compact stores its chain in this form —
/// first-occurrence numbering over the compiled premises and the head
/// (`CompilationNormalizer::makeNormalizedEncodedKey`) — and expands to it
/// with full binding (`ImplicationBuilder::reconstructImplicationFullBind`).
/// A supplied conjecture may number its bound variables in another order, or
/// leave a numbering gap where a bound variable disappeared into a compiled
/// existence; the theorem it proves would then be registered under a spelling
/// its compact never expands to, and a chapter citing the compact could not
/// reach the theorem's `theorem` history row. Generated conjectures are
/// canonical already, so for them the call is the identity.
///
/// @param compiled A conjecture after
///                 `StructuralCompiler::precompileStructuralOperators`:
///                 every premise and the head a simple, possibly negated,
///                 atom; every argument a bound variable.
/// @return The canonical spelling.
/// @invariant The result is alpha-equivalent to @p compiled: the same premise
///            and head shapes under a bijective renaming. Every chain element
///            is simple and every argument is bound (asserted).
/// @see prepareBatch — the caller; addTheoremToMemory.
std::string canonicalConjecture(const std::string& compiled) const;

/// @brief Compile conjectures and prepare their logical-block grid.
/// @details Preserves mirror generation, registration, anchor preprocessing and startup mail order. The conjecture vector and raw-to-compiled map remain caller-owned at their original lifetimes.
/// @param filteredConjectures Surviving conjectures, extended with distinct mirrors.
/// @param rawToCompiled Caller-owned mapping for changed compiled forms.
/// @param structuralCompiler Compiler operation for nested structural forms.
/// @param orTheoremBuilder Mirror construction operation.
/// @param globalDependencies Existing global dependency store.
/// @param blockMailProcessor Ancestor-routing operation.
/// @param dormantLogicBlocksAtGridBuild Dormant-block observation count.
/// @param rollingMailHistoryEnabled Existing retention-policy result.
/// @return No value.
void prepareBatch(std::vector<std::string>& filteredConjectures, std::unordered_map<std::string, std::string>& rawToCompiled, StructuralCompiler& structuralCompiler, OrTheoremBuilder& orTheoremBuilder, Dependencies& globalDependencies, BlockMailProcessor& blockMailProcessor, int& dormantLogicBlocksAtGridBuild, bool& rollingMailHistoryEnabled);

private:

std::vector<std::string> extractSubstringsForAuxy(const std::string& expr2);

private:

std::tuple<
            std::string,
            std::vector<std::string>,
            std::string,
            std::set<std::string>,
            std::string,
            std::string
        >
            createAuxyImplication(const std::string& expr,
                const std::string& arg,
                const std::string& recArg,
                const std::set<std::string>& digitArgs,
                const std::string& anchorName);

/// @brief Build the strong-induction hypothesis of a conjecture for one
///        induction variable.
///
/// @details
/// The weak hypothesis (`createAuxyImplication`) is the conjecture at the
/// step block's fixed predecessor `rec`, with every other digit argument
/// fixed. The strong hypothesis is the conjecture at every smaller value:
/// a fresh universal variable `m` replaces the induction variable `n`
/// everywhere, and two leading premises bound it, written with atomic
/// expressions only — `(in3[m,x,n,+])` (some `x` with `m+x=n`) and
/// `!(=[m,n])` — so `m<n` (D-500).
/// Only `n` and the anchor arguments are fixed (`u_` prefix); every other
/// variable, digit arguments included, is universal, because the step uses
/// the hypothesis with changed arguments (G3 applies it at `(b,r)` for the
/// goal `(a,b)`). `n` occurs only in the two bounding premises.
///
/// The fresh names `m` and `x` are the two decimal names after the largest
/// decimal argument of the conjecture, so they collide with no conjecture
/// variable. The anchor is dropped from the chain, as in the weak builder;
/// the remaining-arguments key is formed by the same rule.
///
/// The product form is the strong step's multiplication catcher
/// (D-566): its bound premise is
/// `(in3[x,m,n,*])` (some `x` with `x·m=n`, `m` the second factor), the rest
/// unchanged. In a step block `n` is a successor, so `x·m=n` forces `x≥1`,
/// hence `m≤n`, and with `m≠n` again `m<n`.
///
/// @param expr       The conjecture (an implication whose first premise is the
///                   anchor).
/// @param arg        The induction variable `n` (a digit argument of @p expr).
/// @param anchorName The batch anchor's name, for example `AnchorFTA`.
/// @param product    Build the multiplication catcher instead of the sum form.
/// @return The tuple (hypothesis implication, its premise chain, its head,
///         the remaining-arguments key, the anchor's zero name, the anchor's
///         successor name) — the same shape as `createAuxyImplication`.
/// @invariant The hypothesis names `n` only in its first two premises; its
///            fixed names are exactly `n` and the anchor arguments.
/// @see createAuxyImplication, AnchorScopeQueries::findPlusArgNameFromAnchor,
///      addTheoremToMemory.
std::tuple<
            std::string,
            std::vector<std::string>,
            std::string,
            std::set<std::string>,
            std::string,
            std::string
        >
            createStrongAuxyImplication(const std::string& expr,
                const std::string& arg,
                const std::string& anchorName,
                bool product = false);

public:
/// @brief The lemma digits a strong step block's induction demand refuses
///        (D-502).
///
/// @details
/// A digit is a number at a slot whose configured definition set is `(1)`,
/// an element of N: the conjecture's own element variables and the anchor's
/// constants. The demand serves only the lemma's input values (maintainer,
/// 2026-10-08): a digit is allowed when it fills a configured input slot
/// (`CoreExpressionConfig::inputIndices`) of a non-anchor premise or of the
/// head; every other digit is refused. `rec` is never a lemma digit and is
/// refused by `hasOnlyInductionDigits` itself; sets and operators are no
/// digits and pass. A premise or head without a configuration contributes
/// nothing. For G3 (`coprime(a,b) -> bez1(a,b)`) the inputs are a and b,
/// and zero (a parameter of `coprime`), one (a parameter of `bez1`) and two
/// (in the anchor only) are refused.
///
/// @param premises The conjecture's premise chain, the anchor premise
///                 included (compiled form).
/// @param head     The conjecture's head.
/// @param out      Receives the refused digits, ascending.
/// @param cap      Capacity of @p out (asserted, never truncated).
/// @return Number of refused digits written.
/// @invariant Reads configurations only; mints nothing.
/// @see hasOnlyInductionDigits, addTheoremToMemory,
///      AdmissionMemory::isAdmittedIntegrationDouble.
int32_t inductionParameterDigits(const std::vector<std::string>& premises,
            const std::string& head, int32_t* out, int32_t cap) const;

/// @brief Whether a lemma admits a strong triad on one of its induction
///        variables (D-594).
///
/// @details
/// The strong hypothesis states the lemma at every smaller value m with
/// every other variable universal; it serves the step only where what the
/// premises demand of the induction variable carries over to m. Every
/// non-anchor premise that contains @p variable must therefore have one of
/// three shapes:
/// - a property: every other argument is an argument of the anchor premise;
/// - a relation: exactly one other non-anchor argument, itself one of
///   @p inductionVariables;
/// - a set operator: every other non-anchor argument sits at a slot the
///   element's configuration lists as uniquely definable
///   (`CoreExpressionConfig::uniquelyDefinableIndices`) and types as a set
///   of numbers (definition set `P(1)`), as the divisor set of a number.
///
/// In addition no element of the lemma, premises and head, may be an
/// operator whose configured output slot is typed other than `P(1)`.
/// Premises that do not contain @p variable are free; an element without a
/// configuration is neither an operator nor a set operator.
///
/// @param premises               The conjecture's premise chain, the anchor
///                               premise included (compiled form).
/// @param head                   The conjecture's head.
/// @param variable               The induction variable of the strong triad.
/// @param inductionVariables     Every induction variable of the lemma
///                               (`ScopeTypes::findDigitArgs`).
/// @param inductionVariableCount Number of @p inductionVariables.
/// @return True iff the strong triad on @p variable may be built.
/// @invariant Reads configurations only; mints nothing; no heap.
/// @see addTheoremToMemory, inductionParameterDigits,
///      ScopeTypes::findDigitArgs.
bool strongInductionEligible(const std::vector<std::string>& premises,
            const std::string& head, StrSpan variable,
            const StrSpan* inductionVariables, int32_t inductionVariableCount) const;

/// @brief Whether a lemma's premises stay the same when two of its strong
///        candidates are exchanged (D-595).
///
/// @details
/// The qualification of a lemma with two strong candidates (the induction
/// variables `strongInductionEligible` admits): their strong triads are
/// built only when the premises cannot tell the two apart, as coprime(a, b)
/// cannot; a shot that exchanges them, as Euclid's step sets (a, b) to
/// (b, a mod b), then lands in the premises again. Every non-anchor premise
/// is renamed by exchanging @p first and @p second, and the renamed
/// premises must be the original ones again, counted with multiplicity.
/// A renamed premise matches an original when both agree in negation, core
/// and every argument, except that a relation its configuration flags
/// symmetric (`CoreExpressionConfig::symmetric`) matches with its two
/// inputs in either order. The head is not read. Premises are flat
/// elements, as `strongInductionEligible` reads them.
///
/// @param premises The conjecture's premise chain, the anchor premise
///                 included (compiled form).
/// @param first    One strong candidate.
/// @param second   The other strong candidate, distinct from @p first.
/// @return True iff the premises are symmetric in @p first and @p second.
/// @invariant Reads configurations only; mints nothing; no heap.
/// @see strongInductionEligible, addTheoremToMemory.
bool strongPremiseSymmetric(const std::vector<std::string>& premises,
            StrSpan first, StrSpan second) const;

public:

void addTheoremToMemory(const std::string& expr,
            Memory& memory,
            int iteration,
            bool proved,
            Dependencies& dependencyTable);

public:
/// @brief Whether a head-switched mirror is the SAME statement as
///        its source up to a renaming of bound variables — the
///        normalize-recognize-skip test of the mirror pre-emit.
///
/// @details
/// A conjecture of the totality shape — symmetric binder prefix,
/// identically-guarded element binders — head-switches into an
/// alpha-variant of itself (rename 9↔10 and the two `(in[·,1])`
/// binder blocks exchange); scheduling that mirror as its own
/// conjecture grid proves the identical statement twice. The test:
///
/// 1. Disintegrate both strings into (chain, head); equal chain
///    lengths required.
/// 2. Derive the candidate variable bijection π from the
///    token-aligned pairing of the two heads: the delimiter
///    skeletons must match byte-for-byte, and every differing token
///    pair must be two BOUND variables, accumulated into a
///    consistent bijection (identity pairs of bound variables are
///    recorded too; unconstrained bound variables complete as
///    identity).
/// 3. Verify π globally: the π-image of the source's chain links
///    (premise plus bound-var list, token-boundary substitution)
///    must equal the mirror's links as a MULTISET. Multiset — not
///    sequence — because the renaming legitimately exchanges
///    identically-shaped binder blocks; both strings are
///    well-formed theorems, so each side's own binder order is
///    scope-legal and multiset equality under π is statement
///    equality.
///
/// A structural mismatch anywhere (head skeletons differ, a
/// non-bound token differs, π inconsistent, chain multisets differ)
/// returns false — the mirror is a genuinely different statement
/// (the Peano or0 parents) and stays scheduled.
///
/// @param original The source conjecture (compiled form).
/// @param mirror   Its `headSwitchOne` output (compiled form).
/// @return True iff the mirror is an alpha-variant of the source.
/// @invariant Pure — no analyzer state is read beyond
///            `coreExpressionMap` (the disintegration walk) and no
///            state is written.
/// @see headSwitchOne — the mirror producer; the pre-emit block in
///      `analyzeExpressions` — the sole production caller.
bool mirrorIsAlphaVariant(const std::string& original,
                                  const std::string& mirror) const;

private:

void prefillIntegrationMapsRecursive(Memory* mb);

public:
//#pragma optimize("", off)

/// @brief Grid-wide axed-anchor pre-pass: walk the LB tree once; outside
///        recursion subtrees every LB under the batch anchor mints its
///        axed x-names into `intAxedVariables` AND registers its
///        x-substituted anchor expression as a `{-1}`-tier statement with
///        an `anchor handling` history line; recursion subtrees take one
///        of two containment modes (set-only, or the anchor-numeral
///        exception).
///
/// @details
/// For each LB whose ancestor chain reaches the batch anchor LB, the pass
/// collects every argument cited by the chain's exprKeys (the trace) and
/// x-copies exactly the anchor's `(1)`-typed slots whose value appears in
/// that trace: slot value `6` becomes `x6`, minted into the LB's
/// `intAxedVariables`. The x-copy is the anchor-premise completion for
/// rules whose element arguments collide with anchor slot values (the
/// element-vs-anchor-slot collision, I-36 family); the minted axed set
/// makes the containment checks — the `addExprToMemoryBlock` prologue,
/// the `addStatement` door, and the orbit-commit refusal — drop every
/// DERIVED x-citing deposit, so the copies never breed x-facts. The
/// statement write is direct — deliberately bypassing those doors.
///
/// Recursion subtrees (D-272,
/// I-188): a block-#1 root whose goal (the
/// theorem head, queued verbatim at creation) cites a `(1)`-typed anchor
/// slot value as a combinable argument (`goalCitesSlotValueAsArgument`)
/// puts its whole subtree in the ANCHOR-NUMERAL EXCEPTION mode
/// — untouched by this pass, so the ancestor's axed-anchor statement
/// arriving by mail registers at the inert door and the historical
/// premise-completion machinery runs there (anchor-numeral theorems such
/// as unit-product need it). Every other recursion subtree — block-#2
/// `_induction_` side-chains always included — is SET-ONLY: axed names
/// minted, no statement, the mailed ancestor form refused at the armed
/// door (a descendant's trace-accumulated axed set is a superset of its
/// ancestors'), keeping those induction sub-blocks entirely x-free.
///
/// @param mb Subtree root to process; the sole production caller passes
///           the grid root once per grid build (from `buildGrid`).
/// @param axedMode Threaded subtree mode: 0 outside recursion subtrees,
///                 1 set-only containment, 2 anchor-numeral exception
///                 (see the mode comment in the body); callers pass 0.
/// @invariant Registration is level-tier `{-1}` (definitional, not
///            derived) and dedup-guarded by the `intKnownStatements`
///            lookup, so re-walks never double-register.
/// @see `addExprToMemoryBlock` — the axed-containment door; SwDD
///      `20_core_concepts/06_anchors_and_scopes.md`.
void prehandleAnchor(Memory* mb, int axedMode = 0);

/// @brief Whether a goal cites a `(1)`-typed anchor slot value as a
///        combinable argument: the trigger of the anchor-numeral
///        exception.
///
/// @details
/// The goal is expanded to base form (`DefinitionExpansion::expandToBaseForm`
/// unfolds every compiled `and` / `or` / `implication` / `existence` name),
/// so every remaining atom `(name[args])` names a configured expression.
/// An argument equal to a slot value counts only when its position is
/// combinable in that expression's `definition_sets`, the flag the
/// conjecturer generalises over. A fixed definition parameter (the flag
/// off, as `nontrivialFactor`'s lower bound `i2` or `prime`'s `i1` and
/// `i2`) belongs to the definition and is no value the theorem is about,
/// so it never triggers the exception (D-548).
/// Quantifier lists `(>[...]` are no atoms and are skipped.
///
/// @param goal The goal text as queued (compiled or base form).
/// @param slotValues The batch anchor's `(1)`-typed slot values.
/// @return True when some atom cites a slot value at a combinable position.
/// @invariant Every atom of the base form names a configured expression
///            whose `definition_sets` covers each argument position
///            (asserted).
/// @see prehandleAnchor (the caller); I-188.
bool goalCitesSlotValueAsArgument(const std::string& goal,
    const std::set<std::string>& slotValues) const;

public:
/// @brief Tear down the LB grid at batch end — delete every child Memory and
///        reset the root's grid links. See the declaration for the full
///        contract.
///
/// @details
/// The four `HashMemory` instances per LB are arena-backed (every `encodedMap`
/// on the cold `lbMemory.manager`) yet declared before that arena, so their
/// `~PagedVector` would otherwise `freePage` on a dead arena. Empty each LB's
/// `encodedMap` while the arena is alive — children before `delete`, the root in
/// place — then drop the grid links. A deload-released LB is already empty, so
/// the release is residency-gated.
void destroyGrid();
private:

AdmissionMemory& admissionMemory;
const ce::AnchorInfo& anchorInfo;
AnchorScopeQueries& anchorScopeQueries;
LogicBlockIdentity& blockIdentity;
Memory& body;
const CompiledDefinitionRegistry& definitions;
HashRuleInstaller& hashRuleInstaller;
InductionUsefulness& inductionUsefulness;
IntegrationProcessor& integrationProcessor;
LbStore& lbStore;
MailLog& mailLog;
const ProverParameters& parameters;
std::vector<Memory*>& permanentBodies;
ScopeTypeResolver& scopeTypes;
SimpleMapStore& simpleMapStore;
StatementMemory& statementMemory;
TheoremPublisher& theoremPublisher;
};
}
#include "../proof/theorem_publisher.hpp"
#include "../proof/induction_usefulness.hpp"
namespace gl {
}
