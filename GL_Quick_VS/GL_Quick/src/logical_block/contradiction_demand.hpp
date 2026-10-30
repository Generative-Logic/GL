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
#include "../memory.hpp"
#include "../compiler/compiled_definition_registry.hpp"
#include "../compiler/definition_expansion.hpp"
#include "../memory_infra/originals_head_index.hpp"

namespace gl {
class IntegrationProcessor;
class ImplicationSubproofs;
namespace tests { struct ContradictionDemandFixture; }

/// @brief The contradiction-scope demand: an open negated-compound
///        contradiction scope asks for the one missing compound of a rule
///        whose head negates a fact it sees (D-560).
///
/// @details
/// A `contradiction_` scope assumes the positive compact of a negated
/// compound and closes when a statement there meets its negation
/// (`GoalProcessor::dischargeContradictionScopes`). Detection is passive: a
/// rule whose conclusion would give that contradiction fires only when all
/// its premises are known. This component closes the gap the way the
/// necessity check (CNFE) closes it for goals: for every fact F the scope
/// sees, the rules whose head is the negation of F (CNFE's head match, the
/// shared `IntegrationProcessor` head-match functions) are candidates, and a
/// candidate whose one unknown premise is a demandable compound, gated by
/// the scope's own statements, gets that compound integrated at the scope
/// (`ImplicationSubproofs::deferContradictionDemand` decides,
/// `IntegrationProcessor::prepareIntegration` integrates).
///
/// The work is change-driven, in one drain per `standardProcessing`:
/// a newly registered scope matches every fact it sees against every rule
/// chain; afterwards a newly known fact is matched against the chains the
/// scopes have already seen, and a newly installed chain against the facts
/// in the derived fact index (`ContradictionFactIndex`), so every (scope,
/// chain, fact) triple is decided once. Nothing runs while no contradiction
/// scope is registered.
///
/// @see ImplicationSubproofs::deferContradictionDemand,
///      ImplicationSubproofs::prepareNegatedCompoundContradiction,
///      IntegrationProcessor::checkNecessityForEquality, ContradictionFactIndex.
class ContradictionDemand {
friend struct tests::ContradictionDemandFixture;
public:
/// @brief Bind inputs for the contradiction-scope demand.
/// @details Construction stores explicit references; block storage and all
/// invocation lifetimes remain with their owners.
/// @param definitions Read-only compiled definitions (operator configs).
/// @param parameters Borrowed ProverParameters; must outlive this component.
/// @param integrationProcessor The shared head match and the integration
///        door; must outlive this component.
/// @param implicationSubproofs The candidate verdict and the park machinery;
///        must outlive this component.
/// @return None; construction binds the supplied references.
ContradictionDemand(const CompiledDefinitionRegistry& definitions, const ProverParameters& parameters,
            IntegrationProcessor& integrationProcessor, ImplicationSubproofs& implicationSubproofs);

/// @brief Serve every registered contradiction scope of a block: decide the
///        new (scope, chain, fact) triples and integrate the ready demands.
///
/// @details
/// Runs in step 4b of `standardProcessing`, after the implication-gate drain
/// (whose revival of parked candidates stages ready demands) and the trial
/// drain; single-threaded block work. With no registered scope it only
/// empties its arrival queue and frees the fact index; under
/// `ban_disintegration` (the compressor) it empties both queues and does
/// nothing, like CNFE. Otherwise, over the participating scopes (registered,
/// not Phase-3 ignored, not filtered for wiping) in decoded order:
/// 1. The fact index is rebuilt when unbuilt: every known statement some
///    participating scope sees is filed under its sign-flipped masked keys.
/// 2. Each newly registered scope matches every fact it sees against the
///    chains `1..count` and becomes bootstrapped.
/// 3. Each staged arrival still known is filed and matched against the
///    chains `1..mark` for every earlier bootstrapped scope that sees it.
/// 4. Each chain above the mark is matched, through its masked key, against
///    the filed facts every earlier bootstrapped scope sees.
/// 5. The mark becomes the chain count at entry, and the staged ready
///    demands are integrated in decoded (compound, scope) order, each once.
/// Work the integrations cause (facts, chains, scopes) waits for the next
/// drain.
///
/// @param mb Owning block.
/// @return Nothing.
/// @invariant Consumes `pendingContradictionArrivals` and
///            `pendingContradictionDemands`; opens scopes only through
///            `prepareIntegration`; zero heap.
/// @see ImplicationSubproofs::deferContradictionDemand,
///      ImplicationSubproofs::drainSubproofArrivals, ContradictionFactIndex.
void drainContradictionDemands(Memory& mb);

/// @brief Visit the masked keys under which the fact index files a fact:
///        the keys of the rule heads that negate it.
///
/// @details
/// A head negating F carries F's operator, the opposite sign, and F's
/// arguments (`u_`-prefixed) except at its one constant, or at its two
/// constants beside at least one other argument, each at a `(1)` slot of the
/// operator's config (`IntegrationProcessor::maskedKeyOfChain`). So F is
/// filed under its sign-flipped `maskedHeadKey` masked at each `(1)` slot,
/// and at each pair of `(1)` slots when it has more than two arguments. An
/// operator without a config yields no key.
///
/// @tparam Sink Callable `void(StrSpan key)`; the key's buffer is reused by
///         the next call.
/// @param fact    Clean fact text (`(X[args])` or `!(X[args])`).
/// @param strings String-scratch arena for the `u_`-prefixed copy; the
///                caller's frame keeps it.
/// @param sink    The visitor.
/// @return Nothing.
/// @invariant Zero heap; mints nothing.
/// @see IntegrationProcessor::maskedKeyOfChain, maskedHeadKey,
///      ContradictionFactIndex::fileUnder.
template <typename Sink>
void forEachFactKey(StrSpan fact, ScratchArena& strings, Sink&& sink) const {
            const ScratchString generic = gl::DefinitionExpansion::prefixArgumentsWithUScratch(strings, fact);
            const StrSpan genericView(generic);
            const StrSpan name = extractExpressionSpan(genericView);
            const ce::CoreExpressionConfig* cfg = definitions.coreConfig(name);
            if (cfg == nullptr) return;
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argN = getArgsSpans(genericView, args, ExecutionParameters::MAX_ARITY);
            bool numeral[ExecutionParameters::MAX_ARITY];
            for (int32_t i = 0; i < argN; ++i) {
                const std::pair<std::string, bool>* ds =
                    gl::CompiledDefinitionRegistry::findDefinitionSet(cfg, i + 1);
                numeral[i] = ds != nullptr && equalSpans(StrSpan(ds->first), StrSpan("(1)", 3));
            }
            // The negating head carries the opposite sign.
            const bool headNegated = !(fact.len > 0 && fact.ptr[0] == '!');
            char keyBuf[kMaxMaskedHeadKeyBytes];
            for (int32_t i = 0; i < argN; ++i) {
                if (!numeral[i]) continue;
                sink(maskedHeadKey(name, headNegated, args, argN, i, keyBuf, kMaxMaskedHeadKeyBytes));
            }
            if (argN <= 2) return;
            for (int32_t i = 0; i < argN; ++i) {
                if (!numeral[i]) continue;
                for (int32_t j = i + 1; j < argN; ++j) {
                    if (!numeral[j]) continue;
                    sink(maskedHeadKey(name, headNegated, args, argN, i, keyBuf,
                                       kMaxMaskedHeadKeyBytes, j));
                }
            }
        }

private:
/// @brief Whether a registered contradiction scope takes part in a drain.
///
/// @details
/// A scope under a Phase-3-ignored OR root, and a scope filtered for the
/// end-of-burst wipe or below a filtered scope (a discharged scope and its
/// subtree), are closing: they are neither served nor bootstrapped, and a
/// demand staged for them is not run.
///
/// @param mb      Owning block.
/// @param scopeId A registered contradiction scope.
/// @return True iff the scope is not Phase-3 ignored and neither it nor an
///         ancestor is filtered.
/// @invariant Pure.
/// @see drainContradictionDemands.
bool participates(const Memory& mb, NameId scopeId) const;

/// @brief File one known fact in the fact index under every key of
///        `forEachFactKey`; a fact already filed is a defined repeat.
///
/// @param mb   Owning block.
/// @param fact Packed `(original, scope)` key of a known statement.
/// @return Nothing.
/// @invariant Mints only into the derived index.
/// @see forEachFactKey, ContradictionFactIndex.
void fileFact(Memory& mb, int64_t fact);

/// @brief Match one fact against the rule chains of an id range for one
///        contradiction scope and decide every candidate.
///
/// @details
/// The fact's negation with every argument `u_`-prefixed is the head-match
/// input; the chains whose head matches it (`collectHeadMatchChains`,
/// ascending) are decided in turn (`decideCandidate`).
///
/// @param mb         Owning block; its head index is built.
/// @param scopeId    The contradiction scope; it sees the fact.
/// @param fact       Packed `(original, scope)` key of a known statement.
/// @param firstChain Smallest chain id to match; at least 1.
/// @param lastChain  Largest chain id to match; below @p firstChain the range
///                   is empty.
/// @return Nothing.
/// @invariant Zero heap.
/// @see IntegrationProcessor::collectHeadMatchChains, decideCandidate.
void matchFact(Memory& mb, NameId scopeId, int64_t fact, int32_t firstChain, int32_t lastChain);

/// @brief Decide one (scope, chain, fact) candidate.
///
/// @details
/// A chain without a demandable compound premise can never demand and is
/// left at once. Otherwise each head constant is replaced by the input's
/// argument at its slot (`headMatchConstantSlots`); a premise that keeps a
/// bound argument, as CNFE's premise loop (D-534), or that reads
/// `!(=[Y,Y])` and can never hold (D-535), leaves the candidate out of
/// scope. The clean premises go to `deferContradictionDemand`, and a ready
/// compound is staged.
///
/// @param mb      Owning block.
/// @param scopeId The contradiction scope.
/// @param chainId A chain whose head matches @p input.
/// @param input   The fact's negation with every argument `u_`-prefixed.
/// @return Nothing.
/// @invariant Zero heap; a park mints its key and record.
/// @see ImplicationSubproofs::deferContradictionDemand.
void decideCandidate(Memory& mb, NameId scopeId, int32_t chainId, StrSpan input);

/// @brief The head-match input of a fact: its negation with every argument
///        `u_`-prefixed, on the string-scratch arena.
///
/// @param fact    Clean fact text.
/// @param strings String-scratch arena; the caller's frame keeps the result.
/// @return The input text.
/// @invariant Zero heap; mints nothing.
static StrSpan negatedInput(StrSpan fact, ScratchArena& strings);

const CompiledDefinitionRegistry& definitions;
const ProverParameters& parameters;
IntegrationProcessor& integrationProcessor;
ImplicationSubproofs& implicationSubproofs;
};
}
