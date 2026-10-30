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
#include "../compiler/definition_expansion.hpp"
#include "../compiler/compact_expression_syntax.hpp"
#include "disintegration_processor.hpp"
#include "or_introduction_rules.hpp"
#include "disintegration_shots.hpp"
namespace gl {
/// @brief Bind inputs for disintegration processor.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param admissionPolicy Borrowed AdmissionPolicy interface or state; must outlive this component.
/// @param canonicalTemplates Borrowed CanonicalTemplateMatcher interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param ceFilteringActive Borrowed bool interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param existenceRuleCompiler Borrowed ExistenceRuleCompiler interface or state; must outlive this component.
/// @param hashRuleClassifier Borrowed HashRuleClassifier interface or state; must outlive this component.
/// @param integrationBuilder Borrowed IntegrationInstructionBuilder interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param rejectedSources Borrowed RejectedSourceMemory interface or state; must outlive this component.
/// @param admissionMemory Borrowed AdmissionMemory interface or state; must outlive this component.
/// @param caseLedger Borrowed CaseLedger interface or state; must outlive this component.
/// @param caseProcessor Borrowed CaseProcessor interface or state; must outlive this component.
/// @param rejectedSourceReplay Borrowed RejectedSourceReplay interface or state; must outlive this component.
/// @param statementMemory Borrowed StatementMemory interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
DisintegrationProcessor::DisintegrationProcessor(AdmissionPolicy& admissionPolicy, CanonicalTemplateMatcher& canonicalTemplates, Canonicalizer& canonicalizer, const bool& ceFilteringActive, const CompiledDefinitionRegistry& definitions, ExistenceRuleCompiler& existenceRuleCompiler, HashRuleClassifier& hashRuleClassifier, IntegrationInstructionBuilder& integrationBuilder, const ProverParameters& parameters, RejectedSourceMemory& rejectedSources, AdmissionMemory& admissionMemory, CaseLedger& caseLedger, CaseProcessor& caseProcessor, RejectedSourceReplay& rejectedSourceReplay, StatementMemory& statementMemory) : admissionPolicy(admissionPolicy), canonicalTemplates(canonicalTemplates), canonicalizer(canonicalizer), ceFilteringActive(ceFilteringActive), definitions(definitions), existenceRuleCompiler(existenceRuleCompiler), hashRuleClassifier(hashRuleClassifier), integrationBuilder(integrationBuilder), parameters(parameters), rejectedSources(rejectedSources), admissionMemory(admissionMemory), caseLedger(caseLedger), caseProcessor(caseProcessor), rejectedSourceReplay(rejectedSourceReplay), statementMemory(statementMemory) {}

/// @brief Publish prepared auxiliary compacts through ordinary disintegration.
/// @details Only template substitution and checked provenance occur here. Each
/// contrapositive is a registered compact whose normal expansion installs its
/// own rule. Collected traversal markers break reciprocal activation cycles.
/// @param instance Fixed-argument source-family instance.
/// @param ownerKey Current collected statement key.
/// @param memoryBlock Resident target logic block.
/// @param collected Scratch products of this disintegration traversal.
/// @param validityName Exact source scope.
/// @param trackHistoryLocal Whether publication history is enabled.
/// @param iteration Source disintegration generation.
/// @return Void.
/// @invariant No registry mint, grouping or contraposition construction occurs here.
void DisintegrationProcessor::collectContrapositionProducts(StrSpan instance,
    StrSpan ownerKey, Memory& memoryBlock, CollectedArena& collected,
    StrSpan validityName, bool trackHistoryLocal, int iteration) {
    const LogicalEntity* owner = this->definitions.compiledEntity(extractExpressionUniversalSpan(instance));
    assert(owner != nullptr);
    if (owner->contrapositives.empty() && owner->supportingImplications.empty()) return;
    const unsigned stringSlot = gl::WorkerContext::coreId() >= 0 ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : scratchArenas().slotCount() - 1;
    const unsigned containerSlot = gl::WorkerContext::coreId() >= 0 ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& arena = scratchArenas().forSlot(stringSlot);
    ScratchArena& containers = genScratchArenas().forSlot(containerSlot);
    ScratchScope frame(arena);
    StrSpan formal[ExecutionParameters::MAX_ARITY], actual[ExecutionParameters::MAX_ARITY];
    const int32_t count = getArgsSpans(StrSpan(owner->signature), formal, ExecutionParameters::MAX_ARITY);
    assert(count == getArgsSpans(instance, actual, ExecutionParameters::MAX_ARITY));
    StrReplacement ownerArguments[ExecutionParameters::MAX_ARITY];
    for (int32_t i = 0; i < count; ++i) ownerArguments[i] = {formal[i], actual[i]};
    const auto instantiate = [&](const std::string& compactTemplate,
                                  ScratchString& compact, ScratchString& expanded) {
        compact = replaceKeysScratch(arena, StrSpan(compactTemplate), ownerArguments, count);
        const LogicalEntity* entity = this->definitions.compiledEntity(extractExpressionUniversalSpan(StrSpan(compact)));
        assert(entity && entity->category == "implication" && entity->elements.size() >= 2
            && entity->elements.size() <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS);
        StrSpan signatureArguments[ExecutionParameters::MAX_ARITY], instanceArguments[ExecutionParameters::MAX_ARITY];
        const int32_t arity = getArgsSpans(StrSpan(entity->signature), signatureArguments, ExecutionParameters::MAX_ARITY);
        assert(arity == getArgsSpans(StrSpan(compact), instanceArguments, ExecutionParameters::MAX_ARITY));
        StrReplacement replacements[ExecutionParameters::MAX_ARITY];
        for (int32_t i = 0; i < arity; ++i) replacements[i] = {signatureArguments[i], instanceArguments[i]};
        StrSpan elements[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
        const int32_t elementCount = static_cast<int32_t>(entity->elements.size());
        for (int32_t i = 0; i < elementCount; ++i)
            elements[i] = replaceKeysScratch(arena, StrSpan(entity->elements[i]), replacements, arity);
        expanded = gl::ImplicationBuilder::reconstructImplicationFullBindScratch(arena, elements, elementCount - 1,
            elements[elementCount - 1]);
    };
    const auto started = [&](StrSpan key) {
        const int32_t id = collected.strings.lookup(key);
        if (id == 0) return false;
        for (int32_t i = 0; i < collected.recs.size(); ++i)
            if (collected.recs[i].keyId == id && collected.recs[i].kind == 2) return true;
        return false;
    };
    const int cap = parameters.compressor_mode ? parameters.compressor_max_origins_per_expr
        : parameters.max_origin_per_expr;
    const auto recordHistory = [&](StrSpan expression, OriginTag tag,
                                   const OriginDep* dependencies, int32_t n) {
        if (!trackHistoryLocal || !parameters.trackHistory) return;
        addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner,
            expression, validityName, tag, dependencies, n, cap);
        memoryBlock.addMailOutOrigin(expression, validityName, tag, dependencies, n, cap);
    };
    const auto publish = [&](StrSpan compact, StrSpan clean, StrSpan expanded) {
        const OriginDep source{expanded, validityName};
        recordHistory(clean, OriginTag::compilation, &source, 1);
        collected.insertChild(ownerKey, clean);
        WorkInstruction instruction(&containers);
        NewVarStore witnesses(&containers);
        integrationBuilder.prepareIntegrationCore(compact, instruction, memoryBlock, clean, true);
        disintegrateExprCore2(compact, instruction, memoryBlock, iteration,
            collected, witnesses, validityName, trackHistoryLocal, false,
            nullptr, 0, false);
        assert(witnesses.varCount() == 0 && "auxiliary implication expansion minted a witness");
    };
    if (!owner->supportingImplications.empty()) {
        ScratchString sourceCompact, sourceExpanded;
        instantiate(owner->signature, sourceCompact, sourceExpanded);
        const OriginDep ownerDependency{ownerKey, validityName};
        recordHistory(StrSpan(sourceExpanded), OriginTag::expansion, &ownerDependency, 1);
        const bool composed = !owner->composedSupportingImplications.empty();
        assert(!composed || owner->composedSupportingImplications.size() == owner->supportingImplications.size());
        for (std::size_t i = 0; i < owner->supportingImplications.size(); ++i) {
            ScratchScope helperFrame(arena);
            ScratchString compact, expanded;
            instantiate(composed ? owner->composedSupportingImplications[i]
                : owner->supportingImplications[i], compact, expanded);
            const ScratchString carrier = gl::DefinitionExpansion::removeUPrefixScratch(arena, StrSpan(compact));
            if (started(StrSpan(carrier))) continue;
            const OriginDep source{StrSpan(sourceExpanded), validityName};
            if (composed) {
                ScratchString helperCompact, helperExpanded;
                instantiate(owner->supportingImplications[i], helperCompact, helperExpanded);
                recordHistory(StrSpan(helperExpanded), OriginTag::expansion, &source, 1);
                const OriginDep dependencies[2] = {
                    {StrSpan(helperExpanded), validityName}, {StrSpan(sourceExpanded), validityName}};
                recordHistory(StrSpan(expanded), OriginTag::implicationComposition, dependencies, 2);
            } else {
                recordHistory(StrSpan(expanded), OriginTag::expansion, &source, 1);
            }
            publish(StrSpan(compact), StrSpan(carrier), StrSpan(expanded));
        }
    }
    for (const auto& record : owner->contrapositives) {
        ScratchScope recordFrame(arena);
        assert(!record.sources.empty());
        ScratchString sourceCompact, sourceHistory, reverseCompact, reverseHistory;
        instantiate(record.sourceRule, sourceCompact, sourceHistory);
        instantiate(record.reverseRule, reverseCompact, reverseHistory);
        const ScratchString sourceCarrier = gl::DefinitionExpansion::removeUPrefixScratch(arena, StrSpan(sourceCompact));
        const ScratchString reverseCarrier = gl::DefinitionExpansion::removeUPrefixScratch(arena, StrSpan(reverseCompact));
        if (record.sources.size() > 1 && !started(StrSpan(sourceCarrier))) {
            assert(record.sources.size() <= ExecutionParameters::kMaxOriginDeps);
            OriginDep dependencies[ExecutionParameters::kMaxOriginDeps];
            for (std::size_t i = 0; i < record.sources.size(); ++i) {
                ScratchString compact, history;
                instantiate(record.sources[i], compact, history);
                dependencies[i] = {StrSpan(history), validityName};
            }
            recordHistory(StrSpan(sourceHistory), OriginTag::implicationGrouping,
                dependencies, static_cast<int32_t>(record.sources.size()));
            publish(StrSpan(sourceCompact), StrSpan(sourceCarrier), StrSpan(sourceHistory));
        }
        if (started(StrSpan(reverseCarrier))) continue;
        const OriginDep source{StrSpan(sourceHistory), validityName};
        recordHistory(StrSpan(reverseHistory), OriginTag::contraposition, &source, 1);
        publish(StrSpan(reverseCompact), StrSpan(reverseCarrier), StrSpan(reverseHistory));
    }
}

/// @brief Publish prepared refutation compacts with exact structural history.
/// @details Every template is instantiated without changing the shared registry.
/// Structural expansion, compilation and ordinary rule expansion remain distinct.
/// @param mb Owning resident block.
/// @param conjunction Fully fixed conjunction instance.
/// @param scope Exact deepest supporting scope.
/// @param dependencies Sole encountered OR citation.
/// @param dependencyCount Exactly one supporting citation.
/// @param levels Sorted nonempty supporting levels.
/// @param levelCount Number of levels.
/// @param output Current internal-mail output channel.
/// @return Void.
void DisintegrationProcessor::publishConjunctionRefutation(Memory& mb, StrSpan conjunction, NameId scope,
    const OriginDep* dependencies, int32_t dependencyCount, const int* levels, int32_t levelCount, ColdMail& output) {
    assert(dependencyCount == 1 && levelCount > 0);
    const auto* entity = this->definitions.compiledEntity(extractExpressionUniversalSpan(conjunction));
    assert(entity != nullptr && entity->category == "and" && entity->conjunctionRefutationsPrepared);
    const unsigned slot = gl::WorkerContext::coreId() >= 0 ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope frame(strings);
    const StrSpan scopeView = mb.nameMap.decodeView(scope);
    const ScratchString validity = ScratchString::copyFrom(strings, scopeView.ptr, scopeView.len);
    const ScratchString fixed = gl::DefinitionExpansion::prefixArgumentsWithUScratch(strings, conjunction);
    StrSpan formal[ExecutionParameters::MAX_ARITY], actual[ExecutionParameters::MAX_ARITY];
    const int32_t arity = getArgsSpans(StrSpan(entity->signature), formal, ExecutionParameters::MAX_ARITY);
    assert(getArgsSpans(StrSpan(fixed), actual, ExecutionParameters::MAX_ARITY) == arity);
    StrReplacement mapping[ExecutionParameters::MAX_ARITY];
    for (int32_t i = 0; i < arity; ++i) mapping[i] = {formal[i], actual[i]};
    const int cap = parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr;
    for (const auto& row : entity->conjunctionRefutations) {
        const ScratchString compactU = replaceKeysScratch(strings, StrSpan(row.second), mapping, arity);
        const ScratchString compact = gl::DefinitionExpansion::removeUPrefixScratch(strings, StrSpan(compactU));
        const auto* rule = this->definitions.compiledEntity(extractExpressionUniversalSpan(StrSpan(compactU)));
        assert(rule != nullptr && rule->category == "implication" && rule->elements.size() == 2);
        StrSpan ruleFormal[ExecutionParameters::MAX_ARITY], ruleActual[ExecutionParameters::MAX_ARITY];
        const int32_t ruleArity = getArgsSpans(StrSpan(rule->signature), ruleFormal, ExecutionParameters::MAX_ARITY);
        assert(getArgsSpans(StrSpan(compactU), ruleActual, ExecutionParameters::MAX_ARITY) == ruleArity);
        StrReplacement substitutions[ExecutionParameters::MAX_ARITY];
        for (int32_t i = 0; i < ruleArity; ++i) substitutions[i] = {ruleFormal[i], ruleActual[i]};
        const ScratchString premise = replaceKeysScratch(strings, StrSpan(rule->elements[0]), substitutions, ruleArity);
        const ScratchString head = replaceKeysScratch(strings, StrSpan(rule->elements[1]), substitutions, ruleArity);
        const StrSpan elements[2] = {StrSpan(premise), StrSpan(head)};
        const ScratchString expanded = gl::DefinitionExpansion::expandSignature(StrSpan("implication", 11), StrSpan(compactU), elements, 2, strings);
        if (parameters.trackHistory && !ceFilteringActive) {
            addOriginEncoded(mb.exprOriginMap, mb.originInterner, StrSpan(expanded), StrSpan(validity),
                OriginTag::expansion, dependencies, dependencyCount, cap);
            mb.addMailOutOrigin(StrSpan(expanded), StrSpan(validity), OriginTag::expansion, dependencies, dependencyCount, cap);
            const OriginDep dependency{StrSpan(expanded), StrSpan(validity)};
            addOriginEncoded(mb.exprOriginMap, mb.originInterner, StrSpan(compact), StrSpan(validity),
                OriginTag::compilation, &dependency, 1, cap);
            mb.addMailOutOrigin(StrSpan(compact), StrSpan(validity), OriginTag::compilation, &dependency, 1, cap);
            addInternalMailOrigin(output, mb.originInterner, StrSpan(compact), StrSpan(validity),
                OriginTag::compilation, &dependency, 1, cap);
        }
        insertInternalStatement(output, mb.nameMap, StrSpan(compact), StrSpan(validity), levels, levelCount);
        setInternalDisintegrationSignal(output, mb.nameMap, StrSpan(compact), StrSpan(validity), false, false, 0);
        mb.mutatedThisBurst = true;
    }
}

/// @brief Publish fixed direct refutations for an admitted OR's AND alternatives.
/// @details The known OR fixes every argument locally. No reverse-rule search,
/// descendant replay or equality index is created. Ordinary compact lifecycle
/// handles later canonical changes and scope cleanup.
/// @param mb Owning block with the admitted OR and its supporting levels.
/// @param statement Exact positive OR source.
/// @param output Current processing phase's output mail.
/// @return Void.
void DisintegrationProcessor::activateDirectOrRefutations(Memory& mb,
    const IntEncodedExpr& statement, ColdMail& output) {
    if (parameters.ban_disintegration || mb.phase3IgnoresScope(statement.validityId)) return;
    const unsigned slot = gl::WorkerContext::coreId() >= 0 ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : scratchArenas().slotCount() - 1;
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope frame(strings);
    const StrSpan textView = mb.nameMap.decodeView(statement.originalId);
    const ScratchString text = ScratchString::copyFrom(strings, textView.ptr, textView.len);
    const StrSpan scopeView = mb.nameMap.decodeView(statement.validityId);
    const ScratchString scope = ScratchString::copyFrom(strings, scopeView.ptr, scopeView.len);
    const auto* entity = this->definitions.compiledEntity(extractExpressionUniversalSpan(StrSpan(text)));
    assert(!statement.negation && entity && entity->category == "or");
    const int32_t levelRow = mb.intStatementLevelsMap.lookup(
        packStatementKey(statement.originalId, statement.validityId));
    assert(levelRow != 0);
    const int32_t levelCount = mb.intStatementLevelsMap.runLen(levelRow);
    assert(levelCount > 0 && levelCount <= 256);
    int levels[256];
    for (int32_t i = 0; i < levelCount; ++i) levels[i] = mb.intStatementLevelsMap.valueAt(levelRow, i);
    StrSpan leaves[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
    const int32_t count = hashRuleClassifier.flattenRegistryOrLeavesScratch(*entity, StrSpan(text), leaves,
        ExecutionParameters::MAX_INSTRUCTION_ELEMENTS, strings);
    const OriginDep source{StrSpan(text), StrSpan(scope)};
    for (int32_t i = 0; i < count; ++i) {
        const auto* part = this->definitions.compiledEntity(extractExpressionUniversalSpan(leaves[i]));
        if (leaves[i].ptr[0] != '(' || part == nullptr || part->category != "and") continue;
        publishConjunctionRefutation(mb, leaves[i], statement.validityId, &source, 1,
            levels, levelCount, output);
    }
}





// Turn optimizations OFF for just this section
//#pragma optimize("", off)






// In GL_Quick_VS/GL_Quick/src/analyze_expressions.cpp

/// @brief Flatten one compiled OR entity into its ordered atomic leaves in the
///        current working instruction.
///
/// @details
/// The instantiated `WorkInstruction` already contains the outer OR and every
/// recursively prepared child entity. This function walks that read-only graph
/// without re-parsing or re-substituting strings: a child whose exact signature
/// resolves to another `or` entity is expanded immediately in the same call;
/// any non-OR child is appended as one leaf. Element order is preserved across
/// both left- and right-nested shapes. Duplicate entity records must be
/// structurally identical, recursion-path cycles assert, and capacity overflow
/// asserts before a write. The local recursive lambda is stack-only and the
/// function mints and allocates nothing.
///
/// @param instructions The instantiated, read-only instruction graph.
/// @param rootEntity The entity index of the outer OR.
/// @param out Caller array receiving ordered leaf spans.
/// @param cap Capacity of @p out.
/// @return Number of leaves written; always at least two.
/// @invariant Every returned span aliases `instructions`; the outer entity and
///            every recursively entered entity are category `or` with at least
///            two elements.
/// @see DisintegrationProcessor::disintegrateExprCore2.
int32_t DisintegrationProcessor::flattenOrLeaves(
    const WorkInstruction& instructions,
    int32_t rootEntity,
    StrSpan* out,
    int32_t cap) const {
    assert(rootEntity >= 0 && rootEntity < instructions.entityCount());
    assert(out != nullptr);
    assert(cap >= 2);
    assert(cap <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS);
    assert(equalSpans(instructions.category(rootEntity), StrSpan("or", 2)));

    StrSpan recursionPath[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
    int32_t recursionDepth = 0;
    int32_t outCount = 0;

    const auto walk = [&](int32_t entityIndex, const auto& self) -> void {
        assert(entityIndex >= 0 && entityIndex < instructions.entityCount());
        assert(equalSpans(instructions.category(entityIndex), StrSpan("or", 2)));
        const StrSpan signature = instructions.signature(entityIndex);
        for (int32_t i = 0; i < recursionDepth; ++i) {
            assert(!equalSpans(recursionPath[i], signature)
                && "flattenOrLeaves: cyclic OR instruction");
        }
        assert(recursionDepth < ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
            && "flattenOrLeaves: OR depth exceeds cap");
        recursionPath[recursionDepth++] = signature;

        const int32_t elementCount = instructions.elemCount(entityIndex);
        assert(elementCount >= 2
            && "flattenOrLeaves: OR entity has fewer than two elements");
        for (int32_t i = 0; i < elementCount; ++i) {
            const StrSpan element = instructions.elemAt(entityIndex, i);
            int32_t childEntity = -1;
            for (int32_t candidate = 0;
                 candidate < instructions.entityCount(); ++candidate) {
                if (!equalSpans(instructions.signature(candidate), element)) continue;
                if (childEntity < 0) {
                    childEntity = candidate;
                    continue;
                }

                assert(equalSpans(instructions.category(childEntity),
                                  instructions.category(candidate))
                    && instructions.elemCount(childEntity)
                        == instructions.elemCount(candidate)
                    && "flattenOrLeaves: duplicate signature differs");
                for (int32_t j = 0; j < instructions.elemCount(childEntity); ++j) {
                    assert(equalSpans(instructions.elemAt(childEntity, j),
                                      instructions.elemAt(candidate, j))
                        && "flattenOrLeaves: duplicate signature differs");
                }
            }

            if (childEntity >= 0
                && equalSpans(instructions.category(childEntity), StrSpan("or", 2))) {
                self(childEntity, self);
            }
            else {
                assert(outCount < cap
                    && "flattenOrLeaves: leaf count exceeds cap");
                out[outCount++] = element;
            }
        }

        --recursionDepth;
    };

    walk(rootEntity, walk);
    assert(recursionDepth == 0);
    assert(outCount >= 2);
    return outCount;
}


void DisintegrationProcessor::disintegrateExprCore2(StrSpan expr,
    WorkInstruction& instructions, Memory& memoryBlock, int iteration,
    CollectedArena& collected, NewVarStore& newVarMap, StrSpan validityName,
    bool trackHistoryLocal, bool allowOrDisintegration,
    const int* orSeedLevels, int32_t orSeedLevelCount, bool allowOrProbe,
    StrSpan parentWitness, WitnessMetaStore* witnessMeta, bool integrationPremise,
    bool conjecturePremise) {
    DisintInstructionView view(instructions);
    disintegrateExprCoreView(expr, view, memoryBlock, iteration, collected,
        newVarMap, validityName, trackHistoryLocal, allowOrDisintegration,
        orSeedLevels, orSeedLevelCount, allowOrProbe, parentWitness, witnessMeta,
        integrationPremise, conjecturePremise);
}

/// @brief Expand one expression through an instruction substitution view.
/// @details The traversal, witness order, history and OR decisions are unchanged.
/// Witness recursion creates a read-only substitution view; instruction extension
/// and concrete OR consumers materialize only their current view.
/// @param expr Expression at this recursion node.
/// @param instructions Current instruction view.
/// @param memoryBlock Owning logic block.
/// @param iteration Firing generation.
/// @param collected Expansion products and history carriers.
/// @param newVarMap Temporary witnesses and their defining elements.
/// @param validityName Current validity scope.
/// @param trackHistoryLocal Whether history may be emitted.
/// @param allowOrDisintegration Existing unconditional OR signal.
/// @param orSeedLevels Caller-owned source levels.
/// @param orSeedLevelCount Number of source levels.
/// @param allowOrProbe Whether demand-based OR processing is enabled.
/// @param parentWitness Enclosing witness token, or the root empty token.
/// @param witnessMeta Current discovery or publication metadata.
/// @param integrationPremise The expression is the deposited premise of an
///        implication being integrated; recursive calls keep the default false,
///        so only the root reacts (D-460).
/// @param conjecturePremise The expression is, or is reached by existence
///        unfolding from, a conjecture's non-anchor premise seeded into its
///        block; an OR reached this way opens with the rule-product permission
///        where the block has proof work (D-495) and is marked for the flag-6
///        unconditional-ordis mail (D-528).
///        Existence recursion keeps it; conjunction and implication recursion
///        drop it.
/// @return Void.
/// @invariant Child substitution views cannot mutate their parent instruction.
/// @see DisintInstructionView, disintegrateExprCore2.
void DisintegrationProcessor::disintegrateExprCoreView(StrSpan expr,
    DisintInstructionView& instructions,
    Memory& memoryBlock,
    int iteration,
    CollectedArena& collected,
    NewVarStore& newVarMap,
    StrSpan validityName,
    bool trackHistoryLocal,
    bool allowOrDisintegration,
    const int* orSeedLevels,
    int32_t orSeedLevelCount,
    bool allowOrProbe,
    StrSpan parentWitness,
    WitnessMetaStore* witnessMeta,
    bool integrationPremise,
    bool conjecturePremise) {
    // Capture startInt at the start of core() as reference
    int referenceStartInt = memoryBlock.startInt;

    if (iteration == -1)
    {
        iteration = 0;
    }

    // Per-slot string scratch for this level's leaf calculation strings; a
    // ScratchScope frees them when this invocation returns. collected (the
    // separate genScratchArenas page tier) holds interned
    // copies, so nothing this level builds needs to outlive it. Recursion nests
    // scopes by stack discipline -- an inner level allocates above this mark and
    // rewinds to its own, never touching these spans.
    const unsigned coreSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : scratchArenas().slotCount() - 1;
    ScratchArena& sArena = scratchArenas().forSlot(coreSlot);
    ScratchScope coreScope(sArena);

    const ScratchString addedExpression = gl::DefinitionExpansion::addMissingUScratch(sArena, StrSpan(expr));
    const StrSpan addedSpan(addedExpression);

    // 1. Find the instruction corresponding to the current expression — an index
    // scan over the WorkInstruction's span accessors (hit < 0 == the former
    // instructions.end() sentinel).
    int32_t hit = -1;
    for (int32_t i = 0; i < instructions.entityCount(); ++i) {
        ScratchScope signatureFrame(sArena);
        if (equalSpans(instructions.signature(i, sArena), addedSpan)) { hit = i; break; }
    }

    // Determine the current statement signature for the map key
    ScratchString currentStatement;
    if (hit < 0) {
        // ---- Negated existence expansion ----
        // existence category encodes !(>[bound](left)!(right)), so
        //   !(existence<N>[args]) = (>[bound](left)!(right)):
        // from left derive !right, and contrapositively from right derive
        // !left. The inner existence is resolved through the REGISTRY —
        // never through the prepared instruction: prepareIntegrationCore
        // commits no entity for a negated compound input (its polarity
        // guard), so the instruction is empty here.
        if (addedSpan.len >= 2 && addedSpan.ptr[0] == '!' && addedSpan.ptr[1] == '(') {
            const StrSpan innerExpr(addedSpan.ptr + 1, addedSpan.len - 1);
            const LogicalEntity* exLe =
                this->definitions.compiledEntity(extractExpressionUniversalSpan(innerExpr));
            if (exLe != nullptr && exLe->category == "existence"
                && exLe->elements.size() == 2) {
                // The two rules — left -> !right and right -> !left — come
                // from the existence entity's compiled `implications` list
                // (D-310): one compact instance
                // `(implication<N>[args])` per rule, u_p -> the inner
                // instance's argument p positionally. Each instance is a
                // product STATEMENT of the negated existence exactly like an
                // `and` element: a child of its key, registered at this
                // scope, and disintegrated through the implication branch of
                // this function, which expands it into the very rule the
                // on-the-spot construction produced (the compact's elements
                // through reconstructImplicationFullBindScratch — the bound
                // variable is the registry's placeholder, as for every mailed
                // compact) and writes the rule's `expansion` origin citing
                // the compact. The compact itself takes the `expansion`
                // origin the rule carried, citing the negated existence.
                assert(static_cast<int32_t>(exLe->implications.size())
                        == existenceRuleCompiler.existenceImplicationCount(*exLe)
                    && "negated-existence expansion: existence-implication list incomplete (I-209)");
                // A numbered element (a generated `and<N>`, a nested
                // `existence<N>`) leaves the list empty — the named-only rule:
                // the negated existence is then a stored statement only and
                // falls through to the plain registration path below.
                if (!exLe->implications.empty()) {
                StrSpan sigArgs[ExecutionParameters::MAX_ARITY];
                const int32_t sigN = getArgsSpans(StrSpan(exLe->signature), sigArgs,
                    ExecutionParameters::MAX_ARITY);
                StrSpan instArgs[ExecutionParameters::MAX_ARITY];
                const int32_t instN = getArgsSpans(innerExpr, instArgs,
                    ExecutionParameters::MAX_ARITY);
                assert(sigN == instN
                    && "negated-existence expansion: instance arity differs from the compiled existence");
                StrReplacement pairs[ExecutionParameters::MAX_ARITY];
                for (int32_t p = 0; p < sigN; ++p) {
                    pairs[p].key = sigArgs[p];
                    pairs[p].value = instArgs[p];
                }
                currentStatement = gl::DefinitionExpansion::removeUPrefixScratch(sArena, addedSpan);
                const int maxOrig = parameters.compressor_mode
                    ? parameters.compressor_max_origins_per_expr
                    : parameters.max_origin_per_expr;
                // Expansion antecedent: the negated existence, u_-stripped
                // (chapter rows surface u_-stripped expressions), a named
                // local so the span stays live across every door call; the
                // doors mint into originInterner, never sArena (I-3).
                const ScratchString expClean = gl::DefinitionExpansion::removeUPrefixScratch(sArena, StrSpan(expr));
                const OriginDep expDeps[1] = { { StrSpan(expClean), StrSpan(validityName) } };
                // The instruction's marked goal is re-set by every
                // prepareIntegrationCore call; copy it onto the arena first so
                // the span never aliases the interner it is minted back into (I-3).
                const StrSpan goalView = instructions.markedGoal();
                const ScratchString goalCopy =
                    ScratchString::copyFrom(sArena, goalView.ptr, goalView.len);
                for (const std::string& tmpl : exLe->implications) {
                    // Body direction only (D-382):
                    // the compact whose premise is the bound variable's bare
                    // typing predicate — the typing direction
                    // `in[k,S] → !body(k)` — is not instantiated; it would
                    // deposit the negated body once per typed term and no
                    // rule ever consumes those rows. The body direction
                    // `body(k) → !in[k,S]` fires where the body is known and
                    // refutes the witness's typing. Registry read only.
                    if (existenceRuleCompiler.isTypingDirectionCompact(StrSpan(tmpl))) continue;
                    // Per-instance scope: the instance bytes live through the
                    // recursion (the WorkInstruction copies what it keeps).
                    ScratchScope instScope(sArena);
                    const ScratchString inst =
                        replaceKeysScratch(sArena, StrSpan(tmpl), pairs, sigN);
                    const ScratchString instClean =
                        gl::DefinitionExpansion::removeUPrefixScratch(sArena, StrSpan(inst));
                    collected.insertChild(StrSpan(currentStatement), StrSpan(instClean));
                    if (parameters.trackHistory && trackHistoryLocal) {
                        addOriginEncoded(memoryBlock.exprOriginMap,
                            memoryBlock.originInterner, StrSpan(instClean),
                            StrSpan(validityName), OriginTag::expansion,
                            expDeps, 1, maxOrig);
                        // Paired `mailOut.exprOriginMap` write (D-274): the
                        // compact registers as a statement and rides the
                        // delta; a receiver resolving its row needs the
                        // antecedent chain shipped alongside.
                        memoryBlock.addMailOutOrigin(StrSpan(instClean),
                            StrSpan(validityName), OriginTag::expansion,
                            expDeps, 1, maxOrig);
                    }
                    integrationBuilder.prepareIntegrationCore(StrSpan(inst), instructions.materialize(sArena), memoryBlock,
                        StrSpan(goalCopy));
                    disintegrateExprCoreView(StrSpan(inst), instructions, memoryBlock,
                        iteration, collected, newVarMap, validityName,
                        trackHistoryLocal, allowOrDisintegration,
                        orSeedLevels, orSeedLevelCount, allowOrProbe,
                        parentWitness, witnessMeta);
                }

                return;
                }
            }
        }


        // A named conjunction's negation supplies its flat exclusion
        // implications. Numbered conjunctions stay ordinary negative facts;
        // neither kind registers an OR source. A negated conjunction that is
        // an integration premise (named or numbered) is additionally split
        // into cases as the De Morgan OR it denotes: its complemented
        // conjuncts form one forced cohort rooted at the negated conjunction
        // itself (D-460). Every other negated
        // conjunction enters no case processing.
        const StrSpan negCore = extractExpressionUniversalSpan(addedSpan);
        const LogicalEntity* negLe = this->definitions.compiledEntity(negCore);
        const bool negatedConjunction = addedSpan.len >= 2 && addedSpan.ptr[0] == '!'
            && addedSpan.ptr[1] == '(' && negLe != nullptr && negLe->category == "and";
        const bool namedConjunction = negatedConjunction
            && !isNumberedCompilationName(negCore, StrSpan(negLe->category));
        if (namedConjunction || (negatedConjunction && integrationPremise)) {
            const StrSpan innerInst(addedSpan.ptr + 1, addedSpan.len - 1);
            StrSpan signatureArgs[ExecutionParameters::MAX_ARITY];
            const int32_t signatureCount = getArgsSpans(StrSpan(negLe->signature),
                signatureArgs, ExecutionParameters::MAX_ARITY);
            StrSpan instanceArgs[ExecutionParameters::MAX_ARITY];
            const int32_t instanceCount = getArgsSpans(innerInst, instanceArgs,
                ExecutionParameters::MAX_ARITY);
            assert(signatureCount == instanceCount
                && "Named negated conjunction: instance arity differs from its definition");
            StrReplacement substitutions[ExecutionParameters::MAX_ARITY];
            for (int32_t i = 0; i < signatureCount; ++i) {
                substitutions[i].key = signatureArgs[i];
                substitutions[i].value = instanceArgs[i];
            }
            StrSpan leaves[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
            int32_t leafCount = 0;
            for (const std::string& element : negLe->elements) {
                assert(leafCount < ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                    && "Named negated conjunction: element count exceeds capacity");
                const ScratchString conjunct = replaceKeysScratch(sArena,
                    StrSpan(element), substitutions, signatureCount);
                leaves[leafCount++] = gl::DefinitionExpansion::negateScratch(sArena, StrSpan(conjunct));
            }
            assert(leafCount >= 2 && "A conjunction has at least two elements");
            currentStatement = gl::DefinitionExpansion::removeUPrefixScratch(sArena, addedSpan);
            collected.ensureKey(StrSpan(currentStatement));
            if (namedConjunction) {
                const ScratchString expanded = gl::DefinitionExpansion::expandSignature(StrSpan("or", 2),
                    addedSpan, leaves, leafCount, sArena);
                const int maxOrigins = parameters.compressor_mode
                    ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr;
                if (parameters.trackHistory && trackHistoryLocal) {
                    const ScratchString original = gl::DefinitionExpansion::removeUPrefixScratch(sArena, expr);
                    const OriginDep dependency{StrSpan(original), validityName};
                    addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner,
                        StrSpan(expanded), validityName, OriginTag::expansion,
                        &dependency, 1, maxOrigins);
                    memoryBlock.addMailOutOrigin(StrSpan(expanded), validityName,
                        OriginTag::expansion, &dependency, 1, maxOrigins);
                }
                const OriginDep dependency{StrSpan(expanded), validityName};
                for (int32_t i = 0; i < leafCount; ++i) {
                    StrSpan premises[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                    int32_t premiseCount = 0;
                    for (int32_t j = 0; j < leafCount; ++j) {
                        if (j != i) premises[premiseCount++] = gl::DefinitionExpansion::negateScratch(sArena, leaves[j]);
                    }
                    const ScratchString implication = gl::ImplicationBuilder::reconstructImplicationFullBindScratch(
                        sArena, premises, premiseCount, leaves[i]);
                    collected.insertImpl(StrSpan(currentStatement), StrSpan(implication), validityName);
                    if (parameters.trackHistory && trackHistoryLocal) {
                        const ScratchString clean = gl::DefinitionExpansion::removeUPrefixScratch(sArena, StrSpan(implication));
                        addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner,
                            StrSpan(clean), validityName, OriginTag::disintegration,
                            &dependency, 1, maxOrigins);
                        memoryBlock.addMailOutOrigin(StrSpan(clean), validityName,
                            OriginTag::disintegration, &dependency, 1, maxOrigins);
                    }
                }
            }
            if (integrationPremise) {
                // The complemented conjuncts are the cohort's ordered
                // leaves; the negated conjunction is its source signature
                // and opens with route-(b) permission. No exclusion source
                // is registered for it.
                caseProcessor.consumeOrLeavesCohort(expr, addedSpan, leaves, leafCount,
                    StrSpan(currentStatement), memoryBlock, collected, validityName,
                    trackHistoryLocal, /*allowOrDisintegration=*/true,
                    orSeedLevels, orSeedLevelCount, allowOrProbe,
                    /*negatedConjunction=*/true, /*forcedAssumptions=*/false, sArena,
                    instructions.materialize(sArena), iteration, newVarMap,
                    parentWitness, witnessMeta);
            }
            return;
        }

        currentStatement = gl::DefinitionExpansion::removeUPrefixScratch(sArena, addedSpan);
        collected.ensureKey(StrSpan(currentStatement)); // Ensure entry exists
        return;
    }

    // Read the hit entity's fields via the WorkInstruction span accessors (no
    // LogicalEntity materialized); the spans alias the read-only `instructions`
    // interner, stable for the whole call.
    const StrSpan entCategory = instructions.category(hit);
    const StrSpan entSignature = instructions.signature(hit, sArena);
    StrSpan entElems[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
    const int32_t entElemN = instructions.elemCount(hit);
    assert(entElemN <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
        && "disintegrateExprCore2: element count exceeds cap");
    for (int32_t j = 0; j < entElemN; ++j) entElems[j] = instructions.elemAt(hit, j, sArena);
    currentStatement = gl::DefinitionExpansion::removeUPrefixScratch(sArena, entSignature);
    collected.ensureKey(StrSpan(currentStatement)); // Ensure entry exists

    // --- LAMBDA: History Tracking ---
    // Handles "expansion" origin for the entity, and optionally "disintegration"
    // origin for its children — span form (category / signature / elements run),
    // 0% heap via the expandSignature span overload.
    auto trackExpansionHistory = [&](StrSpan thCategory, StrSpan thSignature,
        const StrSpan* thElems, int32_t thElemN, bool trackChildren)
        {
            if (!parameters.trackHistory) return;
            // Hypothetical disintegration path: suppress all history writes.
            // disintegrateExprHypothetically uses disintegrateExpr2 only to
            // discover structure; its products are throw-away and must never
            // feed exprOriginMap / mailOut.exprOriginMap (buildStack followed
            // such an orphan origin and crashed).
            if (!trackHistoryLocal) return;
            RT_SCOPE_HERE("CORE_EXPANSION_HISTORY");

            // 1. Record Expansion Origin.
            // L3 span-record doors. The "expansion" antecedent is u_-stripped
            // onto sArena (removeUPrefixScratch, the ScratchString twin of the
            // heap removeUPrefixFromArguments); expandedSignature is a stable
            // local. The doors mint into originInterner, never sArena (I-3).
            const int expCap = (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr);
            const ScratchString exprClean = gl::DefinitionExpansion::removeUPrefixScratch(sArena, StrSpan(expr));
            const OriginDep expDeps[1] = { { StrSpan(exprClean), StrSpan(validityName) } };

            const ScratchString expandedSignature = gl::DefinitionExpansion::expandSignature(
                thCategory, thSignature, thElems, thElemN, sArena);

            {
                RT_SCOPE_HERE("CORE_HISTORY_LOCAL");
                addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner, StrSpan(expandedSignature), StrSpan(validityName), OriginTag::expansion, expDeps, 1, expCap);
            }
            // Paired `mailOut.exprOriginMap` write. The expansion conjunction
            // is NEVER added to `intEncodedStatements` /
            // `intLocalEncodedStatementsDelta`, so `fillMailOut`'s delta loop
            // cannot ship it. Without this direct write, the receiver's
            // `buildStack` (chapter export) walks from a child's
            // `disintegration | <conjunction>` origin to the conjunction,
            // looks it up in `exprOriginMap`, finds nothing, and asserts.
            {
                RT_SCOPE_HERE("CORE_HISTORY_MAIL");
                memoryBlock.addMailOutOrigin(StrSpan(expandedSignature),
                    StrSpan(validityName), OriginTag::expansion, expDeps, 1, expCap);
            }

            // 2. Record Disintegration Origin for Children (if requested)
            if (trackChildren) {
                const OriginDep disDeps[1] = { { StrSpan(expandedSignature), StrSpan(validityName) } };
                for (int32_t te = 0; te < thElemN; ++te) {
                    const StrSpan elem = thElems[te];
                    const ScratchString elemClean = gl::DefinitionExpansion::removeUPrefixScratch(sArena, elem);

                    {
                        RT_SCOPE_HERE("CORE_HISTORY_LOCAL");
                        addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner, StrSpan(elemClean), StrSpan(validityName), OriginTag::disintegration, disDeps, 1, expCap);
                    }
                    // Paired `mailOut.exprOriginMap` write. Even though each
                    // `elem` eventually enters `intLocalEncodedStatementsDelta`
                    // via recursive `disintegrateExprCore2`, some intermediate
                    // disintegration products do not directly hit the delta in
                    // this step (they are processed later in the recursion), so
                    // a delta-only path misses them.
                    {
                        RT_SCOPE_HERE("CORE_HISTORY_MAIL");
                        memoryBlock.addMailOutOrigin(StrSpan(elemClean),
                            StrSpan(validityName), OriginTag::disintegration,
                            disDeps, 1, expCap);
                    }
                }
            }
        };
    // --------------------------------

    if (equalSpans(entCategory, StrSpan("implication", 11))) {
        assert(entElemN > 0 && "Implication category cannot be empty!");

        const int32_t keyCount = entElemN - 1;
        StrSpan keySpans[64];
        assert(keyCount <= 64 && "implication premise chain exceeds 64");
        for (int32_t ki = 0; ki < keyCount; ++ki) keySpans[ki] = entElems[ki];

        // Add implication specifically to THIS statement's entry (first set)
        collected.insertImpl(StrSpan(currentStatement),
            StrSpan(gl::ImplicationBuilder::reconstructImplicationFullBindScratch(sArena, keySpans,
                keyCount, entElems[entElemN - 1])),
            StrSpan(validityName));

        // D-237: a disjunction-shaped premise element also
        // yields its per-leaf INTRO implications — `(>[bound](D_k)(orPremise))`,
        // one per flattened leaf — so a single true disjunct can state the
        // disjunction as a fact and fire this implication's rule. The K
        // mutual-exclusion rules (D-211) are the elimination half of an OR's
        // flat consumption; these are the introduction half. The leaf and the
        // or head are fed in RAW element form — never u_-stripped — so u_
        // arguments stay free and only the changeable arguments bind
        // (reconstructImplicationFullBindScratch drops u_ args from the
        // binder), preserving each argument's u_ status exactly as it appears
        // in the or premise.
        for (int32_t ki = 0; ki < keyCount; ++ki) {
            // The instruction graph carries no entities for an implication's
            // elements (prepareIntegrationCore recurses only into
            // existence/and/or), so the or premise resolves through the
            // compiled-definition map inside the shared builder; a premise
            // that is not a compiled or visits nothing.
            //
            // History dep, loop-invariant: the intro rule unfolds from the
            // compact implication's compiled definition (its or premise's
            // structure), so the dep is the compact statement itself — the
            // same chapter-capable antecedent trackExpansionHistory cites for
            // the main rule's expansion record.
            ScratchString introDepClean;
            bool introDepReady = false;
            const int introCap = parameters.compressor_mode
                ? parameters.compressor_max_origins_per_expr
                : parameters.max_origin_per_expr;
            forEachOrIntroductionRule(this->definitions, entElems[ki], sArena,
                [&](StrSpan, const StrSpan*, int32_t, StrSpan introStr) {
                    collected.insertImpl(StrSpan(currentStatement), introStr, StrSpan(validityName));
                    if (parameters.trackHistory && trackHistoryLocal) {
                        if (!introDepReady) {
                            introDepClean = gl::DefinitionExpansion::removeUPrefixScratch(sArena, StrSpan(expr));
                            introDepReady = true;
                        }
                        const OriginDep introDeps[1] = {
                            { StrSpan(introDepClean), StrSpan(validityName) } };
                        // KEY u_-stripped like every chapter-facing rule form; the
                        // u_-prefixed form lives only in hash memory.
                        const ScratchString introClean =
                            gl::DefinitionExpansion::removeUPrefixScratch(sArena, introStr);
                        addOriginEncoded(memoryBlock.exprOriginMap,
                            memoryBlock.originInterner, StrSpan(introClean),
                            StrSpan(validityName), OriginTag::expansion,
                            introDeps, 1, introCap);
                    }
                });
        }

        // Use Lambda (Children not tracked for implication in original code)
        trackExpansionHistory(entCategory, entSignature, entElems, entElemN, false);
    }
    else if (equalSpans(entCategory, StrSpan("and", 3))) {
        // Insert elements into the second set, removing the "u_" prefix first
        for (int32_t i = 0; i < entElemN; ++i) {
            collected.insertChild(StrSpan(currentStatement),
                StrSpan(gl::DefinitionExpansion::removeUPrefixScratch(sArena, entElems[i])));
        }

        // Recurse on all elements
        for (int32_t i = 0; i < entElemN; ++i) {
            disintegrateExprCoreView(entElems[i], instructions, memoryBlock, iteration, collected, newVarMap, validityName, trackHistoryLocal, allowOrDisintegration, orSeedLevels, orSeedLevelCount, allowOrProbe, parentWitness, witnessMeta);
        }

        // Use Lambda (Children tracked for AND)
        trackExpansionHistory(entCategory, entSignature, entElems, entElemN, true);
    }
    else if (equalSpans(entCategory, StrSpan("existence", 9))
        || equalSpans(entCategory, StrSpan("counterexample", 14)))
    {
        StrSpan removedArgs[ExecutionParameters::MAX_KEY_SLOTS];
        const int32_t removedN = gl::DefinitionExpansion::listLastRemovedArgsLE(entCategory, entElems,
            entElemN, removedArgs, ExecutionParameters::MAX_KEY_SLOTS);
        if (removedN == 0) {
            for (int32_t i = 0; i < entElemN; ++i) {
                disintegrateExprCoreView(entElems[i], instructions, memoryBlock, iteration, collected, newVarMap, validityName, trackHistoryLocal, allowOrDisintegration, orSeedLevels, orSeedLevelCount, allowOrProbe, parentWitness, witnessMeta, /*integrationPremise=*/false, conjecturePremise);
            }
            return;
        }

        assert(removedN == 1);

        // There is no status=3 existence-disintegration ban HERE.
        // status=3 admission is governed solely OUTSIDE
        // disintegrateExpr2 — only compact "(implication<N>...)" forms
        // are admitted (see the addExprToMemoryBlock call-site guard).

        // boundVar aliases removedArgs[0] -> entElems -> the read-only
        // `instructions` interner, stable for the whole call (0% heap).
        const StrSpan boundVar = removedArgs[0];

        // The disintegration packages on this path
        // (I-333): the binders of enclosing
        // packages this existence fills, and its own package's binder 0 when
        // that package has shots. Each witness minted below fills them for
        // its path; the shots attached to them become its constituents after
        // its own elements.
        assert(witnessMeta != nullptr && "Existence expansion requires witness metadata");
        struct ShotFill {
            ShotFrame* frame;
            int32_t binder;
        };
        constexpr int32_t kMaxShotFills = 16;
        ShotFill shotFills[kMaxShotFills];
        int32_t shotFillN = 0;
        ShotFrame ownShotFrame;
        if (equalSpans(entCategory, StrSpan("existence", 9))) {
            const LogicalEntity* own =
                this->definitions.compiledEntity(extractExpressionUniversalSpan(entSignature));
            assert(own != nullptr && own->category == "existence"
                && "disintegrateExprCore2: an existence instance names a compiled existence");
            assert(own->shotsPrepared
                && "disintegrateExprCore2: every existence carries its disintegration package (I-333)");
            for (ShotFrame* frame = witnessMeta->shotFrame; frame != nullptr; frame = frame->outer) {
                const int32_t binder = matchShotBinder(*frame, entSignature, sArena);
                if (binder < 0) continue;
                assert(shotFillN < kMaxShotFills && "disintegrateExprCore2: package frames exceed kMaxShotFills");
                shotFills[shotFillN++] = ShotFill{ frame, binder };
            }
            if (!own->shots.empty()) {
                bindShotFrame(ownShotFrame, *own, entSignature, witnessMeta->shotFrame);
                assert(shotFillN < kMaxShotFills && "disintegrateExprCore2: package frames exceed kMaxShotFills");
                shotFills[shotFillN++] = ShotFill{ &ownShotFrame, 0 };
            }
        }

        // The history of a published shot head
        // (I-333): the shot is repeated on the
        // path's actual products — the universal's rule exactly as this walk
        // recorded its expansion from the compact, and the instantiated
        // premises — and the head takes the implication row a rule firing
        // writes: the rule first, then the premises in expression order.
        const auto recordShotHistory = [&](const ShotFrame& frame, const DisintegrationShot& shot,
                                           int32_t binder, StrSpan head, WorkInstruction& concrete) {
            ScratchScope historyScope(sArena);
            // Instruction signatures spell every witness u_<token>, the
            // path's own witness included.
            const ScratchString compact = instantiateShotText(StrSpan(shot.universal), frame, -1, sArena);
            int32_t ruleEntity = -1;
            for (int32_t e = 0; e < concrete.entityCount() && ruleEntity < 0; ++e)
                if (equalSpans(concrete.signature(e), StrSpan(compact))) ruleEntity = e;
            assert(ruleEntity >= 0 && equalSpans(concrete.category(ruleEntity), StrSpan("implication", 11))
                && "disintegrateExprCore2: a shot's universal is an implication entity of its path");
            StrSpan ruleElements[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
            const int32_t ruleElementN = concrete.elemCount(ruleEntity);
            assert(ruleElementN >= 2 && ruleElementN <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                && "disintegrateExprCore2: a shot's universal has premises and a head");
            for (int32_t j = 0; j < ruleElementN; ++j) ruleElements[j] = concrete.elemAt(ruleEntity, j);
            const ScratchString rule = gl::DefinitionExpansion::expandSignature(StrSpan("implication", 11),
                concrete.signature(ruleEntity), ruleElements, ruleElementN, sArena);
            const int32_t premiseN = static_cast<int32_t>(shot.premises.size());
            assert(premiseN > 0 && premiseN <= kMaxShotPremises
                && "disintegrateExprCore2: a shot names its premises");
            StrSpan premises[kMaxShotPremises];
            for (int32_t p = 0; p < premiseN; ++p)
                premises[p] = StrSpan(instantiateShotText(StrSpan(shot.premises[static_cast<std::size_t>(p)]),
                    frame, binder, sArena));
            constexpr int32_t kRepeatCapacity = 64;
            UniversalShot repeated[kRepeatCapacity];
            const int32_t repeatedN = shootUniversal(StrSpan(rule), premises, premiseN, sArena,
                repeated, kRepeatCapacity);
            const ScratchString headClean = gl::DefinitionExpansion::removeUPrefixScratch(sArena, head);
            bool fired = false;
            for (int32_t k = 0; k < repeatedN && !fired; ++k) {
                const ScratchString firedClean =
                    gl::DefinitionExpansion::removeUPrefixScratch(sArena, StrSpan(repeated[k].head));
                fired = equalSpans(StrSpan(firedClean), StrSpan(headClean));
            }
            assert(fired && "disintegrateExprCore2: a shot repeats on the actual products of its path");
            const int cap = parameters.compressor_mode
                ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr;
            OriginDep deps[1 + kMaxShotPremises];
            deps[0] = OriginDep{ StrSpan(rule), validityName };
            for (int32_t p = 0; p < premiseN; ++p) {
                const ScratchString premiseClean = gl::DefinitionExpansion::removeUPrefixScratch(sArena, premises[p]);
                deps[1 + p] = OriginDep{ StrSpan(premiseClean), validityName };
            }
            std::sort(deps + 1, deps + 1 + premiseN, [](const OriginDep& a, const OriginDep& b) {
                const int expression = compareSpans(a.original, b.original);
                return expression != 0 ? expression < 0 : compareSpans(a.validity, b.validity) < 0;
            });
            addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner, StrSpan(headClean),
                validityName, OriginTag::implication, deps, 1 + premiseN, cap);
            memoryBlock.addMailOutOrigin(StrSpan(headClean), validityName, OriginTag::implication,
                deps, 1 + premiseN, cap);
        };

        // Each witness path sees an independent substitution over the same
        // instruction. Only fields reached by recursion are instantiated.
        auto processPath = [&](StrSpan newVar, const DisintInstructionView& base) {
            RT_SCOPE_HERE("CORE_EXISTENCE_PATH");
            // ppScope holds renamedElemSpans for the whole path (freed per call,
            // mirroring the former heap renamedElements RAII). The nested
            // rnStrScope below marks ABOVE these, so they survive the `renamed`
            // build's rewind (09c pitfall 4).
            ScratchScope ppScope(sArena);

            // Renamed elements (the recursion targets) as a stack StrSpan run on
            // the string tier — each is a replaceKeysScratch (byte-exact twin of
            // replaceKeysToString).
            const StrReplacement rnPair{ boundVar, newVar };
            StrSpan renamedElemSpans[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
            assert(entElemN <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                && "disintegrateExprCore2 processPath: element count exceeds cap");
            for (int32_t i = 0; i < entElemN; ++i)
                renamedElemSpans[i] = StrSpan(replaceKeysScratch(sArena, entElems[i], &rnPair, 1));

            // The shot heads of this witness (I-333):
            // every shot attached to a binder it fills, instantiated on this
            // path, without repeating an element or an earlier head. They are
            // its constituents after its own elements, so the positions of
            // its own elements stay those of the definition.
            StrSpan shotHeads[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
            int32_t shotFillOf[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
            const DisintegrationShot* shotOf[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
            int32_t shotN = 0;
            for (int32_t k = 0; k < shotFillN; ++k) {
                for (const DisintegrationShot& shot : shotFills[k].frame->owner->shots) {
                    if (shot.binder != shotFills[k].binder) continue;
                    const ScratchString head = instantiateShotText(StrSpan(shot.head), *shotFills[k].frame,
                        shotFills[k].binder, sArena);
                    bool known = false;
                    for (int32_t i = 0; i < entElemN && !known; ++i)
                        known = equalSpans(renamedElemSpans[i], StrSpan(head));
                    for (int32_t i = 0; i < shotN && !known; ++i)
                        known = equalSpans(shotHeads[i], StrSpan(head));
                    if (known) continue;
                    assert(entElemN + shotN < ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                        && "disintegrateExprCore2 processPath: elements and shot heads exceed the cap");
                    shotHeads[shotN] = StrSpan(head);
                    shotFillOf[shotN] = k;
                    shotOf[shotN] = &shot;
                    ++shotN;
                }
            }

            // CRUCIAL: Map the new variable to its immediate defining elements.
            // newVar is brand-new (fresh startInt), so its elements append
            // consecutively (appendToTail's current-last-or-new contract).
            if (!witnessMeta->publishing) {
                for (int32_t i = 0; i < entElemN; ++i)
                    newVarMap.addElement(newVar, renamedElemSpans[i]);
                for (int32_t i = 0; i < shotN; ++i)
                    newVarMap.addElement(newVar, shotHeads[i]);
            }

            assert(!startsWithSpan(boundVar, "u_", 2));
            char uNewVar[256];
            assert(newVar.len + 3 <= static_cast<int32_t>(sizeof(uNewVar)));
            uNewVar[0] = 'u'; uNewVar[1] = '_';
            if (newVar.len > 0) std::memcpy(uNewVar + 2, newVar.ptr, static_cast<size_t>(newVar.len));
            const int32_t uNewVarLen = 2 + newVar.len;
            const StrReplacement u2Pair{ boundVar, StrSpan(uNewVar, uNewVarLen) };

            DisintInstructionView renamed(base, u2Pair);

            // Recurse using the renamed instruction. A nested existence
            // reached through this path records THIS witness as its enclosing
            // witness (the relay-selection parent chain) and the element it
            // lies in (D-529).
            const int32_t savedParentElement = witnessMeta->currentParentElement;
            for (int32_t i = 0; i < entElemN; ++i) {
                RT_SCOPE_HERE("CORE_EXISTENCE_RECURSE");
                witnessMeta->currentParentElement = i;
                disintegrateExprCoreView(renamedElemSpans[i], renamed, memoryBlock, iteration, collected, newVarMap, validityName, trackHistoryLocal, allowOrDisintegration, orSeedLevels, orSeedLevelCount, allowOrProbe, newVar, witnessMeta, /*integrationPremise=*/false, conjecturePremise);
            }
            // The shot heads unfold in the same walk: each extends this
            // path's instruction with its own definition tree (the I-209
            // door's pattern) and is disintegrated as a constituent of this
            // witness. A published head first takes its history. Heads are
            // consequences, not the conjecture's premise: no premise flags.
            if (shotN > 0) {
                RT_SCOPE_HERE("CORE_EXISTENCE_SHOTS");
                const StrSpan goalView = renamed.markedGoal();
                const ScratchString goalCopy = ScratchString::copyFrom(sArena, goalView.ptr, goalView.len);
                WorkInstruction& concrete = renamed.materialize(sArena);
                for (int32_t i = 0; i < shotN; ++i) {
                    ScratchScope headScope(sArena);
                    const ScratchString headU = gl::DefinitionExpansion::addMissingUScratch(sArena, shotHeads[i]);
                    integrationBuilder.prepareIntegrationCore(StrSpan(headU), concrete, memoryBlock,
                        StrSpan(goalCopy), true);
                    if (witnessMeta->publishing && parameters.trackHistory && trackHistoryLocal)
                        recordShotHistory(*shotFills[shotFillOf[i]].frame, *shotOf[i],
                            shotFills[shotFillOf[i]].binder, shotHeads[i], concrete);
                    witnessMeta->currentParentElement = entElemN + i;
                    disintegrateExprCoreView(shotHeads[i], renamed, memoryBlock, iteration, collected, newVarMap,
                        validityName, trackHistoryLocal, allowOrDisintegration, orSeedLevels, orSeedLevelCount,
                        allowOrProbe, newVar, witnessMeta, /*integrationPremise=*/false, /*conjecturePremise=*/false);
                }
            }
            witnessMeta->currentParentElement = savedParentElement;
            };

        // Check if at least one element is an operator expression (transparent
        // operators set -> zero-allocation string_view probe).
        bool hasOperator = false;
        for (int32_t i = 0; i < entElemN; ++i) {
            const StrSpan core = extractExpressionSpan(entElems[i]);
            if (this->definitions.operatorNames().find(std::string_view(core.ptr, core.len))
                != this->definitions.operatorNames().end()) {
                hasOperator = true;
                break;
            }
        }

        assert(witnessMeta != nullptr && "Existence expansion requires witness metadata");
        const auto visitWitness = [&](TemporaryWitnessKind kind) {
            if (kind == TemporaryWitnessKind::Integration && witnessMeta->forceIntegration)
                kind = TemporaryWitnessKind::Forced;
            int32_t witnessId;
            if (witnessMeta->publishing) {
                witnessId = witnessMeta->nextSlot++;
                const TemporaryWitnessSlot recorded = witnessMeta->slot(witnessId);
                assert(recorded.kind == kind && recorded.generation == iteration);
                // An unadmitted subtree is skipped, and so is a witness on
                // trial outside its own trial pass
                // (D-481).
                if (recorded.permanentId == 0 || !witnessMeta->publishes(witnessId)) {
                    witnessMeta->nextSlot = recorded.subtreeEnd;
                    return;
                }
            } else {
                const int32_t parentId = parentWitness.empty()
                    ? 0 : witnessMeta->lookup(parentWitness);
                assert(parentWitness.empty() || parentId != 0);
                assert((parentId == 0 || witnessMeta->currentParentElement >= 0)
                    && "disintegrateExprCore2: a nested witness knows its parent element");
                witnessId = witnessMeta->append(kind, iteration, parentId,
                    witnessMeta->count(), StrSpan(currentStatement), witnessMeta->integrationDepth,
                    parentId == 0 ? -1 : witnessMeta->currentParentElement);
            }
            const StrSpan variableView = witnessMeta->publishing
                ? witnessMeta->permanent(witnessId) : witnessMeta->token(witnessId);
            const ScratchString newVar = ScratchString::copyFrom(
                sArena, variableView.ptr, variableView.len);

            if (parameters.trackHistory && trackHistoryLocal) {
                assert(witnessMeta->publishing);
                ScratchScope historyScope(sArena);
                const StrReplacement replacement{boundVar, StrSpan(newVar)};
                StrSpan elements[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                for (int32_t i = 0; i < entElemN; ++i)
                    elements[i] = replaceKeysScratch(sArena, entElems[i], &replacement, 1);
                trackExpansionHistory(entCategory, entSignature, elements, entElemN, true);
            }
            // This witness fills its binders for its path
            // (I-333).
            for (int32_t k = 0; k < shotFillN; ++k)
                shotFills[k].frame->binderToken[shotFills[k].binder] = StrSpan(newVar);
            ShotFrame* const enclosingFrame = witnessMeta->shotFrame;
            if (ownShotFrame.owner != nullptr) witnessMeta->shotFrame = &ownShotFrame;
            processPath(StrSpan(newVar), instructions);
            witnessMeta->shotFrame = enclosingFrame;
            for (int32_t k = 0; k < shotFillN; ++k)
                shotFills[k].frame->binderToken[shotFills[k].binder] = StrSpan();
            if (witnessMeta->publishing) {
                assert(witnessMeta->nextSlot == witnessMeta->slot(witnessId).subtreeEnd);
            } else {
                witnessMeta->closeSubtree(witnessId);
            }
        };
        // Counterexamples have one integration-kind witness, even when an
        // immediate defining fact happens to be an operator application.
        if (hasOperator && !equalSpans(entCategory, StrSpan("counterexample", 14)))
            visitWitness(TemporaryWitnessKind::Algebra);
        visitWitness(TemporaryWitnessKind::Integration);
    }
    else if (equalSpans(entCategory, StrSpan("or", 2))) {
        // An enclosing existence has not passed admission during this walk.
        // Keep only scratch intent; no OR registration, history, opening or
        // parking is permitted until its full witness chain is admitted.
        if (witnessMeta != nullptr && !witnessMeta->publishing) {
            assert(witnessMeta != nullptr);
            collected.recs.push_back(CollectedRec{
                collected.strings.mint(StrSpan(currentStatement)),
                collected.strings.mint(entSignature),
                parentWitness.empty() ? 0 : collected.strings.mint(parentWitness), 3 });
            return;
        }
        // OR disintegration: flatten every contiguous nested OR now, in this
        // call, then treat the ordered atomic leaves as one cohort. The outer
        // signature remains the cohort/provenance identity; no intermediate
        // OR-valued branch is created for a later hashburst. The finalized
        // statement activates on-demand exclusion. consumeOrLeavesCohort
        // handles case history, depth, admission and parking.
        const StrSpan cleanOr(currentStatement);
        const LogicalEntity* orEntity = this->definitions.compiledEntity(extractExpressionUniversalSpan(cleanOr));
        if (allowOrProbe && gl::CompactExpressionSyntax::isOrCompactSpan(cleanOr) && orEntity != nullptr
            && orEntity->symmetricDigitSlots[0] != 0) {
            const NameId originalId = memoryBlock.nameMap.encode(cleanOr);
            const NameId scopeId = memoryBlock.nameMap.encode(validityName);
            const NameId processingId = caseLedger.canonicalOrIdAtScope(memoryBlock, originalId, validityName);
            if (caseLedger.orEquiRepresentativeRecorded(memoryBlock,
                    memoryBlock.nameMap.decodeView(processingId), validityName)) return;
            caseLedger.recordProcessedOr(memoryBlock, originalId, scopeId);
        }
        collected.markOrSource(StrSpan(currentStatement));
        // An OR reached by unfolding a conjecture premise also travels to every
        // descendant with the unconditional-ordis mail flag
        // (D-528).
        if (conjecturePremise) collected.markUnconditionalOrdis(StrSpan(currentStatement));
        StrSpan orLeaves[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
        WorkInstruction& concreteInstruction = instructions.materialize(sArena);
        const int32_t orLeafN = flattenOrLeaves(
            concreteInstruction, hit, orLeaves,
            ExecutionParameters::MAX_INSTRUCTION_ELEMENTS);
        // An OR integration premise opens its case split like route (b)
        // (D-460), and so does an OR reached by existence unfolding from a
        // conjecture's premise where its block has proof work
        // (D-495). A twin block's case
        // experiment opens its partner's zero-or-successor OR at main the
        // same way, with one case to release
        // (D-588).
        const int32_t experimentLeaf = allowOrProbe
            ? caseProcessor.twinExperimentLeaf(memoryBlock, orLeaves, orLeafN, validityName, sArena) : -1;
        caseProcessor.consumeOrLeavesCohort(expr, entSignature, orLeaves, orLeafN,
            StrSpan(currentStatement), memoryBlock, collected, validityName,
            trackHistoryLocal,
            allowOrDisintegration || integrationPremise || conjecturePremise || experimentLeaf >= 0,
            orSeedLevels, orSeedLevelCount, allowOrProbe,
            /*negatedConjunction=*/false, /*forcedAssumptions=*/conjecturePremise, sArena,
            concreteInstruction, iteration, newVarMap, parentWitness, witnessMeta, experimentLeaf);
    }
    else {
        // Default fallback
        // currentStatement was already inserted at top
    }
    if (equalSpans(entCategory, StrSpan("implication", 11))
        || equalSpans(entCategory, StrSpan("and", 3)))
        collectContrapositionProducts(entSignature, StrSpan(currentStatement),
            memoryBlock, collected, validityName, trackHistoryLocal, iteration);
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)

/// @brief Expand a source temporarily and publish only admitted constructions.
/// @details Admission preserves its ordinary routes and commits successful updates
/// after permanent substitution. Ordinary integration requires an eligible depth
/// and a source without algebra-witness inputs. The one exception (I-268,
/// D-492): a source with exactly one distinct
/// algebra-witness input whose slot has a constituent with a visible
/// double-marker predicate is admitted through route 5 only. Unresolved eligible
/// sources park for ordinary mail replay; no temporary name or speculative
/// witness history enters persistent state.
/// @param expr Concrete source expression.
/// @param memoryBlock Owning logic block.
/// @param iteration Firing generation; -1 retains the ordinary zero-generation normalization.
/// @param forceDeep Select integration constructions on integration-only enclosing paths.
/// @param validityName Target scope.
/// @param out Accepted statements, rules, OR sources and relay carriers.
/// @param trackHistoryLocal Whether this call materializes history.
/// @param allowOrDisintegration Original unconditional OR-opening signal.
/// @param orSeedLevels Caller-owned source level run.
/// @param orSeedLevelCount Number of source levels.
/// @param allowOrProbe Whether ordinary OR demand processing is enabled.
/// @param replayKinds Explicit queued construction bits and inherited-carrier bit.
/// @param replayIntegrationDepth Original assigned depth for replay, or -1 for a new expansion.
/// @param inheritedSource Preserve a relay carrier's no-cascade status in source parks.
/// @param recursionHead The source was fired by the installed induction implication
/// (I-229); requires @p forceDeep. Every published witness enters
/// `Memory::recursionHeadWitnesses` (I-286).
/// @param integrationPremise The source is the deposited premise of an
/// implication being integrated (D-460).
/// @param conjecturePremise The source is a conjecture's non-anchor premise
/// seeded into its block; the publication walk opens the ORs reached through
/// existence unfolding where the block has proof work (D-495) and reports
/// those published at the deposit scope in `out.unconditionalOrdis`
/// (D-528).
/// @return Whether every relevant existence source is covered.
/// @invariant An inner witness's admission carries an enclosing witness along only
/// when that witness fills a level of the matched integration instruction at or
/// above the key, or the bound variable of an existence beside it; otherwise the
/// inner witness waits for its enclosing witnesses (D-482).
/// @see TemporaryWitnessStore, queueParkedExistence, I-252.
bool DisintegrationProcessor::disintegrateExpr2(StrSpan expr, Memory& memoryBlock,
    int iteration, bool forceDeep, StrSpan validityName, DisintProducts& out,
    bool trackHistoryLocal, bool allowOrDisintegration,
    const int* orSeedLevels, int32_t orSeedLevelCount, bool allowOrProbe,
    int32_t replayKinds, bool inheritedSource, int32_t replayIntegrationDepth,
    bool recursionHead, bool integrationPremise, bool conjecturePremise) {
    assert((!recursionHead || forceDeep)
        && "disintegrateExpr2: an induction head is always force-disintegrated (D-356)");
    canonicalizer.initializeCanonicalScope(memoryBlock, memoryBlock.nameMap.encode(validityName));
    const int savedStartInt = memoryBlock.startInt;
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
    ScratchArena& arena = genScratchArenas().forSlot(slot);
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope frame(strings);
    CollectedArena discovered(&arena);
    NewVarStore definitions(&arena);
    assert(replayIntegrationDepth >= -1);
    const int32_t integrationDepth = forceDeep ? 0
        : replayIntegrationDepth >= 0 ? replayIntegrationDepth : assignedIntegrationDepth(expr);
    if (replayIntegrationDepth >= 0 && (replayKinds & 2) != 0)
        assert(integrationDepth <= parameters.maxIntegrationDepth);
    WitnessMetaStore witnesses(arena, integrationDepth, forceDeep);
    WorkInstruction instruction(&arena);
    const ScratchString prefixed = gl::DefinitionExpansion::prefixArgumentsWithUScratch(strings, expr);
    integrationBuilder.prepareIntegrationCore(StrSpan(prefixed), instruction, memoryBlock, expr, true);
    disintegrateExprCore2(StrSpan(prefixed), instruction, memoryBlock, iteration,
        discovered, definitions, validityName, false, false, nullptr, 0, false,
        StrSpan(), &witnesses);
    assert(memoryBlock.startInt == savedStartInt);
    assert(witnesses.count() == definitions.varCount());
    const int32_t count = witnesses.count();
    DirtyState dirty = DirtyState::Clean;
    ColdHashSet<BytesKeyStore> selectedStatements(&arena, &dirty);
    ColdHashSet<BytesKeyStore> algebraParks(&arena, &dirty);
    ColdHashSet<BytesKeyStore> integrationParks(&arena, &dirty);
    uint8_t* disabled = count == 0 ? nullptr : reinterpret_cast<uint8_t*>(
        arena.resolve(arena.alloc(count + 1, 1)));
    if (count != 0) std::memset(disabled, 0, count + 1);
    // Integration slots admitted through route 5 only: exactly one distinct
    // algebra-witness input and a constituent with a visible double-marker
    // predicate (the I-268 exception, D-492).
    // Nothing else selects them: no cascade, counterexample, trial or relay.
    uint8_t* doubleOnly = count == 0 ? nullptr : reinterpret_cast<uint8_t*>(
        arena.resolve(arena.alloc(count + 1, 1)));
    if (count != 0) std::memset(doubleOnly, 0, count + 1);

    const auto substitute = [&](StrSpan raw) -> ScratchString {
        const ScratchString clean = gl::DefinitionExpansion::removeUPrefixScratch(strings, raw);
        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int32_t n = getArgsSpans(StrSpan(clean), args, ExecutionParameters::MAX_ARITY);
        StrReplacement pairs[ExecutionParameters::MAX_ARITY];
        int32_t used = 0;
        for (int32_t i = 0; i < n; ++i) {
            const int32_t id = witnesses.lookup(args[i]);
            if (id == 0 || witnesses.slot(id).permanentId == 0) continue;
            bool present = false;
            for (int32_t j = 0; j < used; ++j)
                if (equalSpans(pairs[j].key, args[i])) present = true;
            if (!present) pairs[used++] = {args[i], witnesses.permanent(id)};
        }
        return replaceKeysScratch(strings, StrSpan(clean), pairs, used);
    };
    const auto temporary = [&](StrSpan text) {
        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int32_t n = getArgsSpans(text, args, ExecutionParameters::MAX_ARITY);
        for (int32_t i = 0; i < n; ++i)
            if (witnesses.lookup(args[i]) != 0) return true;
        return false;
    };
    const auto kindBit = [&](int32_t id) {
        return witnesses.slot(id).kind == TemporaryWitnessKind::Algebra ? 1 : 2;
    };
    // The distinct algebra-witness arguments of a source: temporary algebra
    // slots of this expansion and permanent `it_` names (I-268).
    const auto algebraInputCount = [&](StrSpan source) {
        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int32_t n = getArgsSpans(source, args, ExecutionParameters::MAX_ARITY);
        StrSpan inputs[ExecutionParameters::MAX_ARITY];
        int32_t inputN = 0;
        for (int32_t i = 0; i < n; ++i) {
            StrSpan argument = args[i];
            if (startsWithSpan(argument, "u_", 2))
                argument = StrSpan(argument.ptr + 2, argument.len - 2);
            const int32_t input = witnesses.lookup(argument);
            const bool algebra = input != 0
                ? witnesses.slot(input).kind == TemporaryWitnessKind::Algebra
                : classifyWitnessName(argument, false).kind == WitnessNameKind::Algebra;
            if (!algebra) continue;
            bool repeated = false;
            for (int32_t k = 0; k < inputN && !repeated; ++k) repeated = equalSpans(inputs[k], argument);
            if (!repeated) inputs[inputN++] = argument;
        }
        return inputN;
    };
    const NameId scopeId = memoryBlock.nameMap.lookup(validityName);
    // Whether a slot has a constituent whose predicate has a double-marker
    // key at this scope or a strict ancestor — the condition under which one
    // algebra-witness input leaves the slot to route 5
    // (D-492). An unregistered scope has no
    // keys.
    const auto doubleEligible = [&](int32_t id) {
        if (scopeId == 0) return false;
        const int32_t varId = definitions.lookupVar(witnesses.token(id));
        assert(varId != 0);
        for (int32_t j = 0; j < definitions.elemCount(varId); ++j)
            if (AdmissionMemory::doubleMarkerPredicateVisible(memoryBlock, definitions.elemAt(varId, j), scopeId))
                return true;
        return false;
    };
    // Inside an integration trial scope or the usefulness scope and below
    // them nothing mints a witness (D-481, D-521).
    const bool atTrialScope = scopeId != 0
        && gl::ScopeSyntax::atOrBelowRestrictedScope(memoryBlock.nameMap, scopeId);
    // Disable codes: 1 kind not replayed, 2 completed at this scope, 3 over
    // the integration depth or with an algebra input (one algebra input with
    // a visible double-marker predicate leaves the slot to route 5 instead:
    // `doubleOnly`), 4 inherited from a
    // parent disabled for 2 or 3, 5 inside an integration trial or below a
    // witness on trial (covered: the expansion stays structural, nothing
    // parks), 6 an open integration trial already holds the source
    // (covered), 7 the witness opens an integration trial (materialized,
    // published only at its trial scope; no commit, receipt or park here).
    for (int32_t id = 1; id <= count; ++id) {
        if (atTrialScope) {
            disabled[id] = 5;
            continue;
        }
        const TemporaryWitnessSlot value = witnesses.slot(id);
        if (value.parent != 0 && disabled[value.parent])
            disabled[id] = disabled[value.parent] == 1 ? 1 : 4;
        if (value.kind == TemporaryWitnessKind::Integration) {
            const int32_t algebraInputs = algebraInputCount(witnesses.source(id));
            if (integrationDepth > parameters.maxIntegrationDepth)
                disabled[id] = 3;
            else if (algebraInputs == 1 && doubleEligible(id))
                doubleOnly[id] = 1;
            else if (algebraInputs != 0)
                disabled[id] = 3;
        }
        if (value.parent == 0 && replayKinds != 0 && (replayKinds & kindBit(id)) == 0)
            disabled[id] = 1;
        if (!disabled[id] && !forceDeep && scopeId != 0) {
            const StrSpan source = witnesses.source(id);
            if (!temporary(source)) {
                const NameId sourceId = rejectedSources.canonicalExistenceId(memoryBlock, source, scopeId);
                const int32_t* receipt = memoryBlock.existenceReplays.find(packOriginKey(sourceId, scopeId));
                if (receipt && (*receipt & kindBit(id)) != 0) disabled[id] = 2;
            }
        }
    }

    // Integration trials (D-481): a top-level
    // integration witness whose every admitting route — its own and those of
    // the slots below it, which coadmit it — is a gated instruction is
    // materialized here and published at its own trial scope
    // `<scope>_boundary_integration_<witness>`, where it only fires and joins
    // equality classes until its gate elements are known.
    struct PendingTrial {
        int32_t slot;
        int64_t key;
        IntegrationTrialRec record;
    };
    PagedVector<PendingTrial> trials(&arena, &dirty);
    uint8_t* trialSlots = nullptr;
    // Slots whose route waits for an enclosing slot (D-482);
    // set by the admission pass.
    uint8_t* blocked = nullptr;
    const bool trialsAllowed = !atTrialScope && !forceDeep && scopeId != 0 && !ceFilteringActive
        && !containsSpan(validityName, StrSpan("_hypo_", 6));
    // A recursion-owned admission value serves this expansion only when
    // the original has numbers as its only arguments
    // (I-285) and repeats none of them
    // (D-530).
    const bool originalAdmitsInduction = hasOnlyNumberArguments(expr) && !hasRepeatedArgument(expr);
    // The strong step block's induction demand serves this expansion only
    // when the original's arguments are the lemma's input values, sets and
    // operators (D-502), none of them repeated
    // (D-530), and its inputs are the lemma's induction variables
    // (D-601).
    assert((memoryBlock.inductionDoubleKey == 0 || memoryBlock.inductionVariableDigitCount > 0)
        && "disintegrateExpr2: a strong step block records the lemma's induction variables");
    const bool inductionDigitsAllowed = hasOnlyInductionDigits(expr,
        memoryBlock.inductionParameterDigits, memoryBlock.inductionParameterDigitCount)
        && !hasRepeatedArgument(expr)
        && (memoryBlock.inductionDoubleKey == 0
            || admissionPolicy.inductionInputsAllowed(expr, memoryBlock.inductionVariableDigits,
                memoryBlock.inductionVariableDigitCount));
    // The admission metrics of one constituent of a slot: its generation,
    // the pattern occurrences, and one secondary count per algebra slot
    // among its arguments.
    const auto admissionMetrics = [&](int32_t id, StrSpan candidate) {
        WitnessAdmissionMetrics metrics{witnesses.slot(id).generation,
            admissionPolicy.countPatternOccurrences(candidate, memoryBlock.overallHashMemory, memoryBlock.nameMap)};
        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int32_t n = getArgsSpans(candidate, args, ExecutionParameters::MAX_ARITY);
        for (int32_t i = 0; i < n; ++i) {
            const int32_t temporaryId = witnesses.lookup(args[i]);
            if (temporaryId != 0 && witnesses.slot(temporaryId).kind == TemporaryWitnessKind::Algebra)
                ++metrics.secondaryCount;
        }
        return metrics;
    };
    // The admission route of one constituent of a slot, probed without a
    // commit: 1 the ancestors' admission and 2 an operator input for an
    // algebra slot, 3 an integration instruction, 4 the integration
    // admission set and 5 a double-marker demand at the scope or a strict
    // ancestor otherwise, route 5 alone for a `doubleOnly` slot
    // (D-492; the strong step's induction demand only for an original of
    // lemma input values, sets and operators,
    // D-502); 0 none, or a constituent of
    // an algebra slot that is not an operator fact. A route-5 answer reports
    // the key it admitted through in `doubleHit`: the strong step's induction
    // demand opens an induction scope (D-519). Route 3 counts only an
    // instruction the witness-link gate leaves open for the route's `linked`
    // arguments (D-529) and its witness history
    // (D-522).
    const auto probeRoute = [&](int32_t id, StrSpan candidate, StrSpan variable, StrSpan marker,
                                WitnessAdmissionMetrics& metrics, DoubleMarkerHit* doubleHit,
                                const StrSpan* linked, int32_t linkedN,
                                const WitnessHistoryProbe* history) {
        if (witnesses.slot(id).kind == TemporaryWitnessKind::Algebra) {
            const StrSpan core = extractExpressionSpan(candidate);
            if (this->definitions.operatorNames().find(std::string_view(core.ptr, core.len))
                == this->definitions.operatorNames().end()) return 0;
            if (admissionMemory.isAdmittedIncludingAncestors(memoryBlock, candidate, variable,
                    marker, validityName, &metrics, false, originalAdmitsInduction)) return 1;
            if (admissionPolicy.isAllowedAsOperatorInput(memoryBlock, candidate, variable, &metrics)) return 2;
            return 0;
        }
        if (doubleOnly[id])
            return admissionMemory.isAdmittedIntegrationDouble(memoryBlock, marker, validityName,
                inductionDigitsAllowed, doubleHit) ? 5 : 0;
        if (admissionMemory.isAdmittedIntegration(memoryBlock, candidate, variable, marker, validityName, false,
                0, linked, linkedN, history))
            return 3;
        if (canonicalTemplates.canonicalTemplatePresent(memoryBlock, memoryBlock.overallHashMemory.admissionSetIntegration,
                memoryBlock.lbMemory.canonicalUses.integrationSetMatches, marker, validityName)) return 4;
        if (admissionMemory.isAdmittedIntegrationDouble(memoryBlock, marker, validityName,
                inductionDigitsAllowed, doubleHit)) return 5;
        return 0;
    };
    // The constituents of the slots from `inner` up to its ancestor `outer`
    // in their current spelling (a slot not yet materialized keeps its
    // token), allocated in the caller's string frame.
    const auto pathFacts = [&](int32_t inner, int32_t outer, StrSpan* facts) {
        int32_t factN = 0;
        int32_t x = inner;
        while (true) {
            const int32_t var = definitions.lookupVar(witnesses.token(x));
            assert(var != 0);
            for (int32_t j = 0; j < definitions.elemCount(var); ++j) {
                assert(factN < ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                    && "disintegrateExpr2: coadmission facts exceed the probe table");
                const ScratchString fact = substitute(definitions.elemAt(var, j));
                facts[factN++] = StrSpan(fact);
            }
            if (x == outer) break;
            x = witnesses.slot(x).parent;
            assert(x != 0 && "disintegrateExpr2: a descendant lies below its enclosing witness");
        }
        return factN;
    };
    // The level of the instruction matched by slot d's constituent `marker`
    // (route 3) that the slots from d's parent up to `top` fill
    // (D-482): none when one of them fills no level,
    // else the fill of `top`. Every slot is read in its current spelling, as
    // its facts are: a trial's record pass reads its minted names. An
    // instruction the witness-link gate refuses for the route's `linked`
    // arguments (D-529) or its witness history
    // (D-522) has no place.
    const auto enclosingFill = [&](int32_t d, int32_t top, StrSpan marker,
                                   const StrSpan* linked, int32_t linkedN,
                                   const WitnessHistoryProbe* history) {
        const StrSpan descendant = witnesses.slot(d).permanentId != 0
            ? witnesses.permanent(d) : witnesses.token(d);
        for (int32_t a = witnesses.slot(d).parent; ; a = witnesses.slot(a).parent) {
            assert(a != 0 && "disintegrateExpr2: a descendant lies below its enclosing witness");
            ScratchScope pathFrame(strings);
            StrSpan facts[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
            const int32_t factN = pathFacts(d, a, facts);
            const StrSpan enclosing = witnesses.slot(a).permanentId != 0
                ? witnesses.permanent(a) : witnesses.token(a);
            const int32_t fill = admissionMemory.integrationEnclosingFill(memoryBlock, marker, validityName,
                enclosing, descendant, facts, factN, linked, linkedN, history);
            if (fill == AdmissionMemory::kEnclosingFillNone || a == top) return fill;
        }
    };
    // Whether the slots from d's parent up to `top` are each linked to d
    // through the constituents of the path (route 5,
    // D-501): a
    // double-marker key carries no instruction, so an enclosing slot comes
    // along when the facts of d and of the enclosing slots connect it to d.
    // Every slot is read in its current spelling, as its facts are.
    const auto doubleEnclosingLinked = [&](int32_t d, int32_t top) {
        ScratchScope pathFrame(strings);
        StrSpan facts[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
        const int32_t factN = pathFacts(d, top, facts);
        StrSpan enclosing[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
        int32_t enclosingN = 0;
        for (int32_t a = witnesses.slot(d).parent; ; a = witnesses.slot(a).parent) {
            assert(a != 0 && "disintegrateExpr2: a descendant lies below its enclosing witness");
            assert(enclosingN < ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                && "disintegrateExpr2: enclosing witnesses exceed the probe table");
            enclosing[enclosingN++] = witnesses.slot(a).permanentId != 0
                ? witnesses.permanent(a) : witnesses.token(a);
            if (a == top) break;
        }
        const StrSpan descendant = witnesses.slot(d).permanentId != 0
            ? witnesses.permanent(d) : witnesses.token(d);
        return AdmissionMemory::doubleMarkerEnclosingLinked(descendant, enclosing, enclosingN, facts, factN);
    };
    // The witness-link gate (D-529), only in an
    // induction step block: the arguments that the existences a route admits
    // link their witnesses to — slot `id` through its constituent `j`, and
    // each enclosing slot up to `top` through its element that contains the
    // slot below it. For every such slot, the arguments of its other parts
    // with the slot's own definition set (anchor constants excluded,
    // `AdmissionMemory::witnessLinkArguments`); a slot of this expansion not
    // admitted yet (a temporary token) is a new name, never on a path, and is
    // skipped. The spans point into
    // substituted texts allocated in the caller's string frame, which must
    // outlive every use (the caller holds the frame through probe and commit).
    constexpr int32_t kMaxLinked = ExecutionParameters::MAX_PATH_VARIABLES;
    const bool witnessLinkGate = memoryBlock.recursionAuxyImplId != 0;
    const auto linkedArguments = [&](int32_t id, int32_t j, int32_t top, StrSpan* out) {
        int32_t n = 0;
        if (!witnessLinkGate) return n;
        int32_t s = id;
        int32_t through = j;
        while (true) {
            const int32_t varId = definitions.lookupVar(witnesses.token(s));
            assert(varId != 0);
            assert(through >= 0 && through < definitions.elemCount(varId)
                && "disintegrateExpr2: a route goes through one element of each slot it admits");
            const StrSpan witness = witnesses.slot(s).permanentId != 0
                ? witnesses.permanent(s) : witnesses.token(s);
            for (int32_t k = 0; k < definitions.elemCount(varId); ++k) {
                if (k == through) continue;
                const ScratchString part = substitute(definitions.elemAt(varId, k));
                StrSpan found[ExecutionParameters::MAX_ARITY];
                const int32_t foundN = admissionMemory.witnessLinkArguments(memoryBlock, StrSpan(part),
                    witness, found, 0, ExecutionParameters::MAX_ARITY);
                for (int32_t f = 0; f < foundN; ++f) {
                    bool skip = witnesses.lookup(found[f]) != 0;
                    for (int32_t q = 0; q < n && !skip; ++q) skip = equalSpans(out[q], found[f]);
                    if (skip) continue;
                    assert(n < kMaxLinked && "disintegrateExpr2: linked argument count exceeds MAX_PATH_VARIABLES");
                    out[n++] = found[f];
                }
            }
            if (s == top) break;
            through = witnesses.slot(s).parentElement;
            s = witnesses.slot(s).parent;
            assert(s != 0 && "disintegrateExpr2: a slot lies below its enclosing slot");
        }
        return n;
    };
    // The witness-history probe of a route (D-522),
    // only in an induction step block: the existence being disintegrated is
    // the root of the derivation tree, and every slot the route admits — slot
    // `id` and each enclosing slot up to `top` — that is not an operator
    // output contributes its definition set, typed in its existence. A route
    // whose slots are all operator outputs gets an inactive probe.
    const auto historyProbe = [&](int32_t id, int32_t top, WitnessHistoryProbe& probe) {
        probe.start = expr;
        probe.startScope = validityName;
        probe.labelN = 0;
        if (!witnessLinkGate) return;
        int32_t s = id;
        while (true) {
            const int32_t varId = definitions.lookupVar(witnesses.token(s));
            assert(varId != 0);
            const StrSpan witness = witnesses.slot(s).permanentId != 0
                ? witnesses.permanent(s) : witnesses.token(s);
            bool output = false;
            StrSpan label;
            for (int32_t k = 0; k < definitions.elemCount(varId) && !output; ++k) {
                ScratchScope partScope(strings);
                const ScratchString part = substitute(definitions.elemAt(varId, k));
                output = admissionPolicy.isOperatorOutput(StrSpan(part), witness);
                if (label.len == 0) label = admissionMemory.witnessDefinitionSet(StrSpan(part), witness);
            }
            if (!output) probe.addLabel(label);
            if (s == top) break;
            s = witnesses.slot(s).parent;
            assert(s != 0 && "disintegrateExpr2: a slot lies below its enclosing slot");
        }
    };
    // The topmost enclosing slot of `id` not admitted yet, or `id` itself:
    // the top of the names an admission of `id` brings along.
    const auto admissionTop = [&](int32_t id) {
        int32_t top = id;
        for (int32_t a = witnesses.slot(id).parent; a != 0 && witnesses.slot(a).permanentId == 0;
             a = witnesses.slot(a).parent)
            top = a;
        return top;
    };
    // Whether slot id's route may admit it now: the slots enclosing it that
    // are not admitted yet must each fill a level of the instruction its
    // constituent `marker` matched (route 3, D-482) or
    // be linked to it through the facts of the path (route 5,
    // D-501), and
    // come along; an algebra or admission-set route admits a slot only below
    // admitted slots.
    const auto enclosingAdmitted = [&](int32_t id, int route, StrSpan marker,
                                       const StrSpan* linked, int32_t linkedN,
                                       const WitnessHistoryProbe* history) {
        const int32_t top = admissionTop(id);
        if (top == id) return true;
        if (route == 5) return doubleEnclosingLinked(id, top);
        if (route != 3) return false;
        return enclosingFill(id, top, marker, linked, linkedN, history) != AdmissionMemory::kEnclosingFillNone;
    };
    // Whether a slot was prepared from an exactly-one-variable negative
    // implication: the counterexample exception admits it, without a route
    // of its own, as soon as its enclosing slot is admitted.
    const auto counterexampleSlot = [&](int32_t id) {
        const StrSpan source = witnesses.source(id);
        if (!startsWithSpan(source, "!(", 2)) return false;
        const LogicalEntity* entity = this->definitions.compiledEntity(extractExpressionUniversalSpan(source));
        assert(entity != nullptr);
        if (entity->category != "implication") return false;
        assert(witnesses.slot(id).kind == TemporaryWitnessKind::Integration);
        return true;
    };
    // The new names one admission of slot d creates, marked in `newNames`
    // (D-481): d, its enclosing slots, the slots
    // the shared-expression cascade selects with them, and the counterexample
    // slots below them — the names on their admitted other sides. A dry run
    // of selectWitness, the cascade and selectCounterexamples over this
    // expansion's discovered statements, kept within top's subtree. A slot
    // counts as new while it is unmaterialized or was minted for the trial
    // being recorded (`minted`).
    uint8_t* newNames = count == 0 ? nullptr
        : reinterpret_cast<uint8_t*>(arena.resolve(arena.alloc(count + 1, 1)));
    const auto admissionNames = [&](int32_t d, int32_t top, const uint8_t* minted) {
        std::memset(newNames, 0, count + 1);
        const int32_t end = witnesses.slot(top).subtreeEnd;
        const auto isNew = [&](int32_t s) {
            return witnesses.slot(s).permanentId == 0 || (minted != nullptr && minted[s] != 0);
        };
        DirtyState simDirty = DirtyState::Clean;
        ColdHashSet<BytesKeyStore> simStatements(&arena, &simDirty);
        const auto include = [&](StrSpan text, const auto& self) -> void {
            const int32_t before = simStatements.count();
            simStatements.mint(text);
            if (simStatements.count() == before) return;
            discovered.forChildren(text, [&](StrSpan child) { self(child, self); });
        };
        const auto mark = [&](int32_t s, const auto& self) -> void {
            if (newNames[s] || !isNew(s)) return;
            const int32_t parent = witnesses.slot(s).parent;
            if (parent != 0) self(parent, self);
            newNames[s] = 1;
            const int32_t varId = definitions.lookupVar(witnesses.token(s));
            assert(varId != 0);
            for (int32_t j = 0; j < definitions.elemCount(varId); ++j) {
                ScratchScope elementFrame(strings);
                const ScratchString clean = gl::DefinitionExpansion::removeUPrefixScratch(strings,
                    definitions.elemAt(varId, j));
                include(StrSpan(clean), include);
            }
        };
        mark(d, mark);
        bool grew = true;
        while (grew) {
            grew = false;
            const int32_t passCount = simStatements.count();
            for (int32_t i = 1; i <= passCount; ++i) {
                StrSpan args[ExecutionParameters::MAX_ARITY];
                const int32_t n = getArgsSpans(simStatements.keyAt(i), args, ExecutionParameters::MAX_ARITY);
                bool hasSelected = false;
                int32_t candidates[ExecutionParameters::MAX_ARITY];
                int32_t pending = 0;
                for (int32_t j = 0; j < n; ++j) {
                    const int32_t s = witnesses.lookup(args[j]);
                    if (s == 0 || disabled[s]) continue;
                    if (newNames[s] || !isNew(s)) hasSelected = true;
                    else if (!doubleOnly[s]) candidates[pending++] = s;
                }
                if (!hasSelected) continue;
                for (int32_t j = 0; j < pending; ++j) {
                    if (newNames[candidates[j]]) continue;
                    mark(candidates[j], mark);
                    grew = true;
                }
            }
            for (int32_t s = top + 1; s < end; ++s) {
                if (newNames[s] || disabled[s] || doubleOnly[s] || !isNew(s)) continue;
                const int32_t parent = witnesses.slot(s).parent;
                if (parent == 0 || !newNames[parent] || !counterexampleSlot(s)) continue;
                mark(s, mark);
                grew = true;
            }
        }
        for (int32_t s = 1; s <= count; ++s)
            if (s < top || s >= end) newNames[s] = 0;
    };
    // Whether every admitting route of a top-level witness is a gated
    // instruction (route 3). Its routes are those of its own constituents
    // and the instruction routes of the slots below it whose enclosing slots
    // up to it each have a place in the matched instruction
    // (D-482). Every new name such an admission creates
    // — the matched slot, its enclosing slots and the names the cascade
    // selects on their admitted other sides — takes on the gates of its place
    // in the instruction (D-481); the route is
    // gated when some instruction leaves a gate of one of them unknown. A
    // slot below without a place, an algebra slot below and the admission
    // set of a slot below admit nothing beyond that slot; the witness's own
    // admission set (route 4) or double-marker demand (route 5) and an
    // ungated instruction admit as usual. A double-marker demand of a slot
    // below, route-5-only slots included, admits the witness ungated when
    // the slots up to it are linked to that slot
    // (D-501), and
    // nothing beyond that slot otherwise. The
    // probe (no record) marks the new names of the gated routes in
    // `routeNames`; the record pass (the names minted, flagged in `minted`)
    // files the gates into the record.
    const auto gatedRoutes = [&](int32_t id, IntegrationTrialRec* record,
                                 uint8_t* routeNames, const uint8_t* minted) {
        const int32_t subtreeEnd = witnesses.slot(id).subtreeEnd;
        bool gated = false;
        for (int32_t d = id; d < subtreeEnd; ++d) {
            if (d != id && disabled[d]) continue;
            if (witnesses.slot(d).kind == TemporaryWitnessKind::Algebra) {
                assert(d != id && "disintegrateExpr2: a trial witness is an integration witness");
                continue;
            }
            assert((!doubleOnly[d] || d != id) && "disintegrateExpr2: a trial witness is not a route-5-only slot");
            const StrSpan variable = witnesses.slot(d).permanentId != 0
                ? witnesses.permanent(d) : witnesses.token(d);
            const int32_t varId = definitions.lookupVar(witnesses.token(d));
            assert(varId != 0);
            for (int32_t j = 0; j < definitions.elemCount(varId); ++j) {
                ScratchScope probeFrame(strings);
                ScratchScope routeFrame(arena);
                const ScratchString candidate = substitute(definitions.elemAt(varId, j));
                const ScratchString marker = gl::DefinitionExpansion::makeMarkedExprScratch(strings, StrSpan(candidate), variable);
                if (!doubleOnly[d] && canonicalTemplates.canonicalTemplatePresent(memoryBlock,
                        memoryBlock.overallHashMemory.admissionSetIntegration,
                        memoryBlock.lbMemory.canonicalUses.integrationSetMatches, StrSpan(marker), validityName)) {
                    assert((record == nullptr || d != id)
                        && "disintegrateExpr2: a trial witness has no admission-set route");
                    if (d == id) return false;
                    continue;
                }
                if (admissionMemory.isAdmittedIntegrationDouble(memoryBlock, StrSpan(marker), validityName,
                        inductionDigitsAllowed)) {
                    assert((record == nullptr || d != id)
                        && "disintegrateExpr2: a trial witness has no double-marker route");
                    if (d == id || doubleEnclosingLinked(d, id)) return false;
                    continue;
                }
                if (doubleOnly[d]) continue;
                StrSpan linked[kMaxLinked];
                const int32_t linkedN = linkedArguments(d, j, id, linked);
                WitnessHistoryProbe history;
                historyProbe(d, id, history);
                if (!admissionMemory.isAdmittedIntegration(memoryBlock, StrSpan(candidate), variable,
                        StrSpan(marker), validityName, false, 0, linked, linkedN, &history)) continue;
                if (d != id && enclosingFill(d, id, StrSpan(marker), linked, linkedN, &history)
                        == AdmissionMemory::kEnclosingFillNone) continue;
                admissionNames(d, id, minted);
                int32_t factCapacity = 0;
                for (int32_t s = id; s < subtreeEnd; ++s)
                    if (newNames[s]) factCapacity += definitions.elemCount(definitions.lookupVar(witnesses.token(s)));
                constexpr int32_t spanBytes = static_cast<int32_t>(sizeof(StrSpan));
                constexpr int32_t spanAlign = static_cast<int32_t>(alignof(StrSpan));
                StrSpan* others = reinterpret_cast<StrSpan*>(arena.resolve(arena.alloc(count * spanBytes, spanAlign)));
                StrSpan* facts = reinterpret_cast<StrSpan*>(arena.resolve(
                    arena.alloc((factCapacity + 1) * spanBytes, spanAlign)));
                int32_t otherN = 0;
                int32_t factN = 0;
                for (int32_t s = id; s < subtreeEnd; ++s) {
                    if (!newNames[s]) continue;
                    assert((minted == nullptr || minted[s] != 0)
                        && "disintegrateExpr2: a trial records the gates of the names it minted");
                    if (s != d) others[otherN++] = witnesses.slot(s).permanentId != 0
                        ? witnesses.permanent(s) : witnesses.token(s);
                    const int32_t sVar = definitions.lookupVar(witnesses.token(s));
                    for (int32_t k = 0; k < definitions.elemCount(sVar); ++k) {
                        const ScratchString fact = substitute(definitions.elemAt(sVar, k));
                        facts[factN++] = StrSpan(fact);
                    }
                }
                assert(newNames[d] && factN <= factCapacity);
                const bool routeGated = admissionMemory.integrationTrialGates(memoryBlock, StrSpan(marker),
                    validityName, variable, others, otherN, facts, factN, record, linked, linkedN, &history);
                assert((record == nullptr || routeGated)
                    && "disintegrateExpr2: a trial's routes stay gated while it opens");
                if (!routeGated) return false;
                gated = true;
                if (routeNames != nullptr)
                    for (int32_t s = id; s < subtreeEnd; ++s)
                        if (newNames[s]) routeNames[s] = 1;
            }
        }
        return gated;
    };
    // A passed trial whose recorded witnesses this expansion admits under
    // their names, checked against its recorded expressions after publication.
    struct PendingPromotion {
        int32_t slot;
        IntegrationTrialRec record;
    };
    PagedVector<PendingPromotion> promotions(&arena, &dirty);
    uint8_t* routeNameFlags = count == 0 ? nullptr
        : reinterpret_cast<uint8_t*>(arena.resolve(arena.alloc(count + 1, 1)));
    // Decide the trials before any ordinary admission, so a witness on trial
    // never enters the parent through a descendant. One trial per source and
    // scope: an open record covers the slot. A passed record is the scope's
    // receipt: its recorded witnesses enter here under the names the trial
    // minted, and the ordinary pass admits them through their routes. Only
    // the opening of a new trial depends on `trialsAllowed`.
    const auto openIntegrationTrials = [&]() {
        if (atTrialScope || scopeId == 0) return;
        for (int32_t id = 1; id <= count; ++id) {
            const TemporaryWitnessSlot value = witnesses.slot(id);
            if (disabled[id] || doubleOnly[id] || value.parent != 0
                || value.kind != TemporaryWitnessKind::Integration || value.permanentId != 0) continue;
            const StrSpan source = witnesses.source(id);
            assert(!temporary(source));
            const int64_t key = packOriginKey(
                rejectedSources.canonicalExistenceId(memoryBlock, source, scopeId), scopeId);
            bool pending = false;
            for (int32_t t = 0; t < trials.size() && !pending; ++t) pending = trials[t].key == key;
            const IntegrationTrialRec* existing = memoryBlock.lbMemory.integrationTrials.find(key);
            if (pending || (existing != nullptr && existing->status == IntegrationTrialRec::kOpen)) {
                disabled[id] = 6;
                for (int32_t d = id + 1; d < value.subtreeEnd; ++d) disabled[d] = 5;
                continue;
            }
            if (existing != nullptr) {
                bool promoted = false;
                for (int32_t p = 0; p < promotions.size() && !promoted; ++p)
                    promoted = promotions[p].record.trialScope == existing->trialScope;
                if (promoted) {
                    // A second copy of the promoted source: covered by the first.
                    disabled[id] = 6;
                    for (int32_t d = id + 1; d < value.subtreeEnd; ++d) disabled[d] = 5;
                    continue;
                }
                assert(existing->status == IntegrationTrialRec::kPassed && existing->witnessCount > 0
                    && "disintegrateExpr2: a passed trial lists the witnesses it admits");
                const IntegrationTrialRec record = *existing;
                for (int32_t i = 0; i < record.witnessCount; ++i) {
                    const int64_t* entry = memoryBlock.lbMemory.integrationTrialContents.find(
                        packInt32Pair(record.trialScope, i));
                    assert(entry != nullptr && "disintegrateExpr2: a passed trial's witness list is complete");
                    const int32_t s = id + static_cast<int32_t>(static_cast<uint64_t>(*entry) >> 32);
                    const NameId recorded = static_cast<NameId>(static_cast<uint32_t>(*entry));
                    assert(s >= id && s < value.subtreeEnd && !disabled[s] && witnesses.slot(s).permanentId == 0
                        && "disintegrateExpr2: a recorded witness is a slot of the promoted source");
                    witnesses.materializeAs(s, memoryBlock.nameMap.decodeView(recorded));
                }
                promotions.push_back(PendingPromotion{ id, record });
                continue;
            }
            if (!trialsAllowed) continue;
            std::memset(routeNameFlags, 0, count + 1);
            if (!gatedRoutes(id, nullptr, routeNameFlags, nullptr)) continue;
            assert(routeNameFlags[id] && "disintegrateExpr2: a trial mints its top witness");
            PendingTrial trial{};
            trial.slot = id;
            trial.key = key;
            trial.record.source = memoryBlock.nameMap.encode(source);
            trial.record.status = IntegrationTrialRec::kOpen;
            trial.record.generation = value.generation;
            trial.record.integrationDepth = integrationDepth;
            trial.record.flags = kindBit(id) | (allowOrDisintegration ? 4 : 0) | (inheritedSource ? 8 : 0);
            // Every new name of the gated admissions, parents first.
            for (int32_t s = id; s < value.subtreeEnd; ++s)
                if (routeNameFlags[s]) witnesses.materialize(s, memoryBlock.level, memoryBlock.startInt);
            const bool gated = gatedRoutes(id, &trial.record, nullptr, routeNameFlags);
            assert(gated && trial.record.groupCount > 0 && "disintegrateExpr2: a trial records its gates");
            (void)gated;
            trials.push_back(trial);
            if (trialSlots == nullptr) {
                trialSlots = reinterpret_cast<uint8_t*>(arena.resolve(arena.alloc(count + 1, 1)));
                std::memset(trialSlots, 0, count + 1);
            }
            trialSlots[id] = 1;
            disabled[id] = 7;
            for (int32_t d = id + 1; d < value.subtreeEnd; ++d) disabled[d] = 5;
        }
    };
    // The scratch closure selects products; it never publishes them.
    const auto includeStatement = [&](StrSpan text, const auto& self) -> void {
        const int32_t before = selectedStatements.count();
        selectedStatements.mint(text);
        if (selectedStatements.count() == before) return;
        discovered.forChildren(text, [&](StrSpan child) { self(child, self); });
    };
    const auto selectWitness = [&](int32_t id, const auto& self) -> void {
        assert(!disabled[id]);
        const TemporaryWitnessSlot value = witnesses.slot(id);
        if (value.permanentId != 0) return;
        if (value.parent != 0) self(value.parent, self);
        witnesses.materialize(id, memoryBlock.level, memoryBlock.startInt);
        const int32_t varId = definitions.lookupVar(witnesses.token(id));
        assert(varId != 0);
        for (int32_t j = 0; j < definitions.elemCount(varId); ++j) {
            ScratchScope elementFrame(strings);
            const ScratchString clean = gl::DefinitionExpansion::removeUPrefixScratch(strings, definitions.elemAt(varId, j));
            includeStatement(StrSpan(clean), includeStatement);
        }
    };

    discovered.forEachKey([&](StrSpan text) {
        if (!temporary(text)) includeStatement(text, includeStatement);
    });
    const auto selectCounterexamples = [&]() {
        bool changed = false;
        for (int32_t id = 1; id <= count; ++id) {
            const TemporaryWitnessSlot value = witnesses.slot(id);
            if (disabled[id] || doubleOnly[id] || value.permanentId != 0) continue;
            if (value.parent != 0 && witnesses.slot(value.parent).permanentId == 0) continue;
            if (!counterexampleSlot(id)) continue;
            // This slot was prepared from an exactly-one-variable negative
            // implication. The exception admits only it, not its descendants.
            selectWitness(id, selectWitness);
            changed = true;
        }
        return changed;
    };
    // Induction admissions (D-519): a witness admitted
    // through an induction key eats a premise of the induction hypothesis and
    // opens its own induction scope; the subtree of the names the admission
    // brings along is published there, not here. One pass per (subtree top,
    // scope), in admission order. A pass remembers the slot it caught: when
    // one subtree holds several catches, each scope gets its own copy of the
    // path down to its catch and leaves the other catches out
    // (D-566).
    struct InductionPass {
        int32_t slot;
        NameId scope;
        int32_t caught;
    };
    PagedVector<InductionPass> inductionPasses(&arena, &dirty);
    uint8_t* inductionSlots = nullptr;
    const auto addInductionPass = [&](int32_t top, NameId scope, int32_t caught) {
        assert(top > 0 && top <= count && scope != 0 && caught >= top && caught < witnesses.slot(top).subtreeEnd);
        for (int32_t p = 0; p < inductionPasses.size(); ++p)
            if (inductionPasses[p].slot == top && inductionPasses[p].scope == scope) return;
        inductionPasses.push_back(InductionPass{ top, scope, caught });
    };
    // The induction scope a route-5 admission through one of the strong step's
    // demands opens below this scope: that demand's chain and the two names of
    // the constituent at the key's marker positions, marker1 first. Both names
    // are admitted names; enclosing witnesses brought along stay unnamed.
    const auto openStrongInductionScope = [&](StrSpan constituent, const DoubleMarkerHit& hit) {
        assert((hit.key == memoryBlock.inductionDoubleKey || hit.key == memoryBlock.inductionProductKey)
            && "disintegrateExpr2: a strong induction scope opens through a strong demand");
        const NameId chainId = hit.key == memoryBlock.inductionDoubleKey
            ? memoryBlock.inductionDoubleChainId : memoryBlock.inductionProductChainId;
        assert(chainId != 0 && scopeId != 0);
        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int32_t n = getArgsSpans(constituent, args, ExecutionParameters::MAX_ARITY);
        assert(hit.markerAt >= 0 && hit.markerAt < n && hit.otherAt >= 0 && hit.otherAt < n
            && hit.markerAt != hit.otherAt && "disintegrateExpr2: a route-5 hit names two positions");
        const int32_t first = hit.markerAt < hit.otherAt ? hit.markerAt : hit.otherAt;
        const int32_t second = hit.markerAt < hit.otherAt ? hit.otherAt : hit.markerAt;
        const StrSpan names[2] = { args[first], args[second] };
        for (int32_t k = 0; k < 2; ++k)
            assert(witnesses.lookup(names[k]) == 0
                && "disintegrateExpr2: an induction scope names admitted names only");
        // The caught smaller value and its distance are the strong step's
        // sizes, as `rec` is the weak step's: keys holding them are connected
        // to the induction (D-584).
        if (memoryBlock.strongStepBlock)
            for (int32_t k = 0; k < 2; ++k) admissionMemory.recordSizeMarker(memoryBlock, names[k]);
        ScratchScope scopeFrame(strings);
        const StrSpan chainView = memoryBlock.nameMap.decodeView(chainId);
        const ScratchString chain = ScratchString::copyFrom(strings, chainView.ptr, chainView.len);
        const StrSpan chainSpan(chain);
        const ScratchString payload = gl::ScopeSyntax::inductionScopePayloadScratch(
            strings, &chainSpan, 1, names, 2);
        // The first eat at this scope fixes its slot: one order per path
        // (D-519).
        memoryBlock.recordInductionEat(scopeId, Memory::kStrongInductionSlot);
        const NameId opened = memoryBlock.nameMap.encodePush(scopeId, StrSpan(payload));
        // The catch fills both markers. Every premise of the demand's chain
        // left with only fixed arguments and the two names is a candidate for
        // the hypothesis's missing expression in the new scope; the statement
        // door files it once this disintegration's products are in. The
        // source is the rule whose demand caught (D-575).
        StrSpan premises[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
        const int32_t premiseN = gl::ScopeSyntax::splitChainElements(chainSpan, premises,
            ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS);
        ScratchString ground[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
        const int32_t groundN = gl::ScopeSyntax::groundHypothesisPremisesScratch(strings,
            premises, premiseN, names, 2, ground, ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS);
        const StrSpan openedName = memoryBlock.nameMap.decodeView(opened);
        for (int32_t i = 0; i < groundN; ++i) {
            if (!admissionPolicy.ordis2KeyEligible(StrSpan(ground[i]))) continue;
            const int32_t sourceImplId = hit.key == memoryBlock.inductionDoubleKey
                ? memoryBlock.recursionAuxyImplId : memoryBlock.recursionProductImplId;
            assert(memoryBlock.isInductionHypothesisRule(sourceImplId)
                && "disintegrateExpr2: a strong demand belongs to an installed hypothesis rule");
            out.hypothesisDemands.append(StrSpan(ground[i]), openedName,
                memoryBlock.ruleInterner.decodeView(sourceImplId));
        }
        return opened;
    };
    if (forceDeep) {
        // Inside an integration trial not even a forced witness is minted,
        // e.g. under an induction head fired there.
        for (int32_t id = 1; id <= count && !atTrialScope; ++id) {
            const TemporaryWitnessSlot value = witnesses.slot(id);
            if (value.kind == TemporaryWitnessKind::Forced
                && (value.parent == 0 || witnesses.slot(value.parent).permanentId != 0))
                selectWitness(id, selectWitness);
        }
    } else if (!parameters.compressor_mode && !parameters.ban_disintegration) {
        selectCounterexamples();
        openIntegrationTrials();
        // Preserve the previous admission visitation order without allocating
        // those names. Slot records, not this sorting text, determine their kind.
        const auto admissionLess = [&](int32_t a, int32_t b) {
            char left[96], right[96];
            const auto name = [&](int32_t id, char* bytes) {
                const auto value = witnesses.slot(id + 1);
                assert(savedStartInt <= INT_MAX - id);
                return value.kind == TemporaryWitnessKind::Algebra
                    ? std::snprintf(bytes, 96, "it_%d_lev_%d_%d", value.generation,
                        memoryBlock.level, savedStartInt + id)
                    : std::snprintf(bytes, 96, "int_%d_lev_%d_%d", value.integrationDepth,
                        memoryBlock.level, savedStartInt + id);
            };
            const int ln = name(a, left), rn = name(b, right);
            return compareSpans(StrSpan(left, ln), StrSpan(right, rn)) < 0;
        };
        ChunkSortedOrdinals<decltype(admissionLess)> order(arena, count, 0, admissionLess);
        int32_t* visit = count == 0 ? nullptr
            : reinterpret_cast<int32_t*>(arena.resolve(arena.alloc(count * 4, 4)));
        for (int32_t k = 0; k < count; ++k) visit[k] = order.next() + 1;
        if (count != 0) {
            blocked = reinterpret_cast<uint8_t*>(arena.resolve(arena.alloc(count + 1, 1)));
            std::memset(blocked, 0, count + 1);
        }
        // Admit slot id through its first route: 0 no route, 1 admitted, 2 a
        // route found while an enclosing slot neither is admitted nor fills a
        // level of it (D-482).
        const auto admitSlot = [&](int32_t id) {
            const TemporaryWitnessSlot value = witnesses.slot(id);
            const int32_t varId = definitions.lookupVar(witnesses.token(id));
            assert(varId != 0);
            // The top of the names this admission brings along: the slot
            // itself or its topmost enclosing slot not admitted before.
            const int32_t top = admissionTop(id);
            for (int32_t j = 0; j < definitions.elemCount(varId); ++j) {
                ScratchScope probeFrame(strings);
                const ScratchString candidate = substitute(definitions.elemAt(varId, j));
                const StrSpan variable = value.permanentId != 0
                    ? witnesses.permanent(id) : witnesses.token(id);
                const ScratchString marker = gl::DefinitionExpansion::makeMarkedExprScratch(strings, StrSpan(candidate), variable);
                StrSpan linked[kMaxLinked];
                const int32_t linkedN = linkedArguments(id, j, top, linked);
                WitnessHistoryProbe history;
                historyProbe(id, top, history);
                WitnessAdmissionMetrics metrics = admissionMetrics(id, StrSpan(candidate));
                DoubleMarkerHit doubleHit;
                const int route = probeRoute(id, StrSpan(candidate), variable, StrSpan(marker), metrics,
                    &doubleHit, linked, linkedN, &history);
                if (route == 0) continue;
                if (!enclosingAdmitted(id, route, StrSpan(marker), linked, linkedN, &history)) return 2;
                selectWitness(id, selectWitness);
                const ScratchString permanentCandidate = substitute(definitions.elemAt(varId, j));
                const ScratchString permanentMarker = gl::DefinitionExpansion::makeMarkedExprScratch(strings,
                    StrSpan(permanentCandidate), witnesses.permanent(id));
                if (route == 1) {
                    InductionScopeSink opened;
                    opened.hypothesisDemands = &out.hypothesisDemands;
                    const bool accepted = admissionMemory.isAdmittedIncludingAncestors(memoryBlock,
                        StrSpan(permanentCandidate), witnesses.permanent(id),
                        StrSpan(permanentMarker), validityName, &metrics, true,
                        originalAdmitsInduction, &opened);
                    assert(accepted);
                    for (int32_t s = 0; s < opened.count; ++s) addInductionPass(top, opened.scopes[s], id);
                } else if (route == 3) {
                    // A first fill of a premise-bound copy opens the copy's
                    // int scope: the subtree of the names this admission brings
                    // along is published there, unless an instruction that
                    // opens none accepted the witness here as well
                    // (D-578).
                    InductionScopeSink opened;
                    const bool accepted = admissionMemory.isAdmittedIntegration(memoryBlock,
                        StrSpan(permanentCandidate), witnesses.permanent(id),
                        StrSpan(permanentMarker), validityName, true, 0, linked, linkedN, &history, &opened);
                    assert(accepted);
                    if (!opened.publishedHere)
                        for (int32_t s = 0; s < opened.count; ++s) addInductionPass(top, opened.scopes[s], id);
                } else if (route == 5 && (doubleHit.key == memoryBlock.inductionDoubleKey
                        || doubleHit.key == memoryBlock.inductionProductKey)) {
                    assert(doubleHit.key != 0 && "disintegrateExpr2: a route-5 admission names its key");
                    addInductionPass(top, openStrongInductionScope(StrSpan(permanentCandidate), doubleHit), id);
                }
                return 1;
            }
            return 0;
        };
        for (int32_t k = 0; k < count; ++k) {
            const int32_t id = visit[k];
            if (disabled[id]) continue;
            blocked[id] = admitSlot(id) == 2 ? 1 : 0;
        }
        // A blocked slot follows once its enclosing slots are admitted: probe
        // the blocked slots again until a pass admits none.
        const auto retryBlocked = [&]() {
            bool admitted = false;
            bool progress = true;
            while (progress) {
                progress = false;
                for (int32_t k = 0; k < count; ++k) {
                    const int32_t id = visit[k];
                    if (!blocked[id]) continue;
                    const int result = admitSlot(id);
                    blocked[id] = result == 2 ? 1 : 0;
                    if (result == 1) progress = admitted = true;
                }
            }
            return admitted;
        };
        bool grew;
        do {
            grew = selectCounterexamples();
            if (retryBlocked()) grew = true;
            const int32_t passCount = selectedStatements.count();
            for (int32_t i = 1; i <= passCount; ++i) {
                StrSpan args[ExecutionParameters::MAX_ARITY];
                const int32_t n = getArgsSpans(selectedStatements.keyAt(i), args, ExecutionParameters::MAX_ARITY);
                bool hasSelected = false;
                int32_t candidates[ExecutionParameters::MAX_ARITY];
                int32_t pending = 0;
                for (int32_t j = 0; j < n; ++j) {
                    const int32_t id = witnesses.lookup(args[j]);
                    if (id == 0 || disabled[id]) continue;
                    if (witnesses.slot(id).permanentId != 0) hasSelected = true;
                    else if (!doubleOnly[id]) candidates[pending++] = id;
                }
                if (!hasSelected) continue;
                for (int32_t j = 0; j < pending; ++j) {
                    if (witnesses.slot(candidates[j]).permanentId != 0) continue;
                    selectWitness(candidates[j], selectWitness);
                    grew = true;
                }
            }
        } while (grew);
    }

    // Reconstruct only selected paths using permanent names. This is the
    // existing expansion/history/OR machinery, now behind admission.
    // Slots published at their own scope and skipped here: the witnesses on
    // trial and the subtree tops of the induction admissions. No such slot
    // lies inside another one's subtree.
    // A catch inside the subtree of an earlier pass's top is published from
    // that top: the main pass skips the whole subtree, so the catch's scope
    // gets the path down to it as a copy (D-566).
    // An earlier top never lies inside a later one: a later admission's top
    // lies below an admitted slot.
    for (int32_t p = 0; p < inductionPasses.size(); ++p) {
        const int32_t inner = inductionPasses[p].slot;
        for (int32_t q = 0; q < p; ++q) {
            const int32_t outer = inductionPasses[q].slot;
            if (inner == outer) break;
            assert(!(outer > inner && outer < witnesses.slot(inner).subtreeEnd)
                && "disintegrateExpr2: an earlier induction pass top lies inside a later one");
            if (inner > outer && inner < witnesses.slot(outer).subtreeEnd) {
                InductionPass merged = inductionPasses[p];
                merged.slot = outer;
                inductionPasses.setAt(p, merged);
                break;
            }
        }
    }
    if (inductionPasses.size() != 0) {
        inductionSlots = reinterpret_cast<uint8_t*>(arena.resolve(arena.alloc(count + 1, 1)));
        std::memset(inductionSlots, 0, count + 1);
        for (int32_t p = 0; p < inductionPasses.size(); ++p) inductionSlots[inductionPasses[p].slot] = 1;
    }
    uint8_t* ownScopeSlots = trialSlots;
    if (inductionSlots != nullptr) {
        ownScopeSlots = reinterpret_cast<uint8_t*>(arena.resolve(arena.alloc(count + 1, 1)));
        for (int32_t s = 0; s <= count; ++s)
            ownScopeSlots[s] = static_cast<uint8_t>(
                (trialSlots != nullptr && trialSlots[s] != 0) || inductionSlots[s] != 0 ? 1 : 0);
        for (int32_t s = 1; s <= count; ++s) {
            if (!ownScopeSlots[s]) continue;
            for (int32_t t = s + 1; t < witnesses.slot(s).subtreeEnd; ++t)
                assert(!ownScopeSlots[t]
                    && "disintegrateExpr2: an own-scope slot lies inside another one's subtree");
        }
    }
    CollectedArena published(&arena);
    witnesses.publishing = true;
    witnesses.nextSlot = 1;
    witnesses.setPublicationFilter(ownScopeSlots, 0);
    disintegrateExprCore2(StrSpan(prefixed), instruction, memoryBlock, iteration,
        published, definitions, validityName, trackHistoryLocal,
        allowOrDisintegration, orSeedLevels, orSeedLevelCount, allowOrProbe,
        StrSpan(), &witnesses, integrationPremise, conjecturePremise);
    assert(witnesses.nextSlot == count + 1);
    ColdHashSet<BytesKeyStore> parentProducts(&arena, &dirty);
    ColdHashSet<BytesKeyStore> parentRules(&arena, &dirty);
    published.forEachKey([&](StrSpan text) {
        assert(!containsSpan(text, StrSpan("__gl_temporary_witness_", 23)));
        out.statements.append(text, validityName);
        if (published.isOrSource(text)) out.orSources.append(text, validityName);
        if (published.isUnconditionalOrdis(text)) out.unconditionalOrdis.append(text, validityName);
        published.forImpls(text, [&](StrSpan rule, StrSpan validity) {
            assert(!containsSpan(rule, StrSpan("__gl_temporary_witness_", 23)));
            out.implications.append(rule, validity, text);
            if (promotions.size() != 0) parentRules.mint(rule);
        });
        if (ownScopeSlots != nullptr || promotions.size() != 0) parentProducts.mint(text);
    });
    // A forced witness of a strong step block whose definition set is `(1)`
    // demands its successor at the scope that minted it; the statement door
    // files the demand once the products are in
    // (D-571). A forced slot publishes at this
    // scope: no trial or induction scope holds one.
    if (memoryBlock.inductionDoubleKey != 0) {
        for (int32_t id = 1; id <= count; ++id) {
            const TemporaryWitnessSlot value = witnesses.slot(id);
            if (value.kind != TemporaryWitnessKind::Forced || value.permanentId == 0) continue;
            for (int32_t s = id; s != 0; s = witnesses.slot(s).parent)
                assert((ownScopeSlots == nullptr || ownScopeSlots[s] == 0)
                    && "disintegrateExpr2: a forced witness publishes at the expansion's scope");
            const StrSpan witness = witnesses.permanent(id);
            const int32_t varId = definitions.lookupVar(witnesses.token(id));
            assert(varId != 0);
            StrSpan label;
            for (int32_t k = 0; k < definitions.elemCount(varId) && label.len == 0; ++k) {
                ScratchScope partScope(strings);
                const ScratchString part = substitute(definitions.elemAt(varId, k));
                label = admissionMemory.witnessDefinitionSet(StrSpan(part), witness);
            }
            if (equalSpans(label, StrSpan("(1)", 3)))
                out.successorDemands.append(witness, validityName);
        }
    }
    // A promoted trial admits here exactly what it listed: every recorded
    // statement and rule is published at this scope. A promoted source whose
    // admission opened its own scopes (an int scope's first fill,
    // D-578) is checked against each of those scopes
    // below, after their passes.
    for (int32_t p = 0; p < promotions.size(); ++p) {
        if (inductionSlots != nullptr && inductionSlots[promotions[p].slot]) continue;
        const IntegrationTrialRec record = promotions[p].record;
        for (int32_t i = record.witnessCount; i < record.witnessCount + record.expressionCount; ++i) {
            const int64_t* entry = memoryBlock.lbMemory.integrationTrialContents.find(
                packInt32Pair(record.trialScope, i));
            assert(entry != nullptr && "disintegrateExpr2: a passed trial's expression list is complete");
            const int32_t kind = static_cast<int32_t>(static_cast<uint64_t>(*entry) >> 32);
            const StrSpan text = memoryBlock.nameMap.decodeView(static_cast<NameId>(static_cast<uint32_t>(*entry)));
            assert((kind == kIntegrationTrialStatement ? parentProducts.lookup(text) != 0
                    : kind == kIntegrationTrialRule && parentRules.lookup(text) != 0)
                && "disintegrateExpr2: a promoted trial publishes its recorded expressions at the parent");
        }
    }
    // Each witness on trial: mint its trial scope, record the source's zoom
    // into it, publish the witness's subtree there with
    // OR handling off, and file the trial record. The statements every pass
    // shares (the source and everything outside witness subtrees) stay at
    // this scope.
    for (int32_t t = 0; t < trials.size(); ++t) {
        PendingTrial trial = trials[t];
        ScratchScope trialFrame(strings);
        // `integration_<name1>_<name2>...`: the witnesses minted for the
        // trial, in slot order.
        const int32_t trialEnd = witnesses.slot(trial.slot).subtreeEnd;
        const int32_t prefixLen = static_cast<int32_t>(sizeof(ScopeSyntax::kIntegrationTrialScopePrefix) - 1);
        int32_t payloadLen = prefixLen - 1;
        for (int32_t s = trial.slot; s < trialEnd; ++s)
            if (witnesses.slot(s).permanentId != 0) payloadLen += 1 + witnesses.permanent(s).len;
        char* payload = strings.allocBytes(payloadLen);
        std::memcpy(payload, ScopeSyntax::kIntegrationTrialScopePrefix, static_cast<size_t>(prefixLen));
        int32_t payloadAt = prefixLen;
        for (int32_t s = trial.slot; s < trialEnd; ++s) {
            if (witnesses.slot(s).permanentId == 0) continue;
            if (payloadAt != prefixLen) payload[payloadAt++] = '_';
            const StrSpan name = witnesses.permanent(s);
            std::memcpy(payload + payloadAt, name.ptr, static_cast<size_t>(name.len));
            payloadAt += name.len;
        }
        assert(payloadAt == payloadLen && payloadAt > prefixLen
            && "disintegrateExpr2: a trial scope names the witnesses minted for it");
        const int32_t namesBefore = memoryBlock.nameMap.nameCount();
        const NameId trialScope = memoryBlock.nameMap.encodePush(scopeId, StrSpan(payload, payloadLen));
        assert(memoryBlock.nameMap.nameCount() != namesBefore
            && "disintegrateExpr2: fresh witnesses name a fresh trial scope");
        canonicalizer.initializeCanonicalScope(memoryBlock, trialScope);
        const StrSpan trialView = memoryBlock.nameMap.decodeView(trialScope);
        const ScratchString trialValidity = ScratchString::copyFrom(strings, trialView.ptr, trialView.len);
        if (parameters.trackHistory && trackHistoryLocal) {
            const int cap = parameters.compressor_mode
                ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr;
            const OriginDep zoom[1] = { { expr, validityName } };
            addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner, expr,
                StrSpan(trialValidity), OriginTag::zoomIn, zoom, 1, cap);
            memoryBlock.addMailOutOrigin(expr, StrSpan(trialValidity), OriginTag::zoomIn, zoom, 1, cap);
        }
        CollectedArena trialPublished(&arena);
        witnesses.nextSlot = 1;
        witnesses.setPublicationFilter(ownScopeSlots, trial.slot);
        disintegrateExprCore2(StrSpan(prefixed), instruction, memoryBlock, iteration,
            trialPublished, definitions, StrSpan(trialValidity), trackHistoryLocal,
            false, orSeedLevels, orSeedLevelCount, false, StrSpan(), &witnesses, false);
        assert(witnesses.nextSlot == count + 1);
        // The record lists what the trial admits on success: its witnesses
        // (slot ordinal below the trial slot, name), then the statements and
        // rules published here and not at the parent.
        auto& contents = memoryBlock.lbMemory.integrationTrialContents;
        assert(contents.find(packInt32Pair(trialScope, 0)) == nullptr
            && "disintegrateExpr2: a fresh trial scope lists nothing yet");
        int32_t listed = 0;
        for (int32_t s = trial.slot; s < trialEnd; ++s)
            if (witnesses.slot(s).permanentId != 0)
                contents.insert(packInt32Pair(trialScope, listed++),
                    packInt32Pair(s - trial.slot, memoryBlock.nameMap.encode(witnesses.permanent(s))));
        trial.record.witnessCount = listed;
        trialPublished.forEachKey([&](StrSpan text) {
            if (parentProducts.lookup(text) != 0) return;
            assert(!containsSpan(text, StrSpan("__gl_temporary_witness_", 23)));
            out.statements.append(text, StrSpan(trialValidity));
            if (trialPublished.isOrSource(text)) out.orSources.append(text, StrSpan(trialValidity));
            contents.insert(packInt32Pair(trialScope, listed++),
                packInt32Pair(kIntegrationTrialStatement, memoryBlock.nameMap.encode(text)));
            trialPublished.forImpls(text, [&](StrSpan rule, StrSpan validity) {
                assert(!containsSpan(rule, StrSpan("__gl_temporary_witness_", 23)));
                out.implications.append(rule, validity, text);
                contents.insert(packInt32Pair(trialScope, listed++),
                    packInt32Pair(kIntegrationTrialRule, memoryBlock.nameMap.encode(rule)));
            });
        });
        trial.record.expressionCount = listed - trial.record.witnessCount;
        trial.record.trialScope = trialScope;
        assert(memoryBlock.lbMemory.integrationTrials.find(trial.key) == nullptr
            && "disintegrateExpr2: one trial per source and scope");
        memoryBlock.lbMemory.integrationTrials.insert(trial.key, trial.record);
    }
    // Each induction admission, and each first fill of a premise-bound
    // integration copy: record the source's zoom into its induction or int
    // scope and publish the subtree of the names it brought along there, with
    // this expansion's OR handling. The statements every pass shares stay at
    // this scope (D-519, D-578).
    // The other catches of a pass's subtree are left out of its scope; its
    // own path and its own catch's subtree always stay
    // (D-566).
    uint8_t* passExcluded = inductionPasses.size() == 0 ? nullptr
        : reinterpret_cast<uint8_t*>(arena.resolve(arena.alloc(count + 1, 1)));
    for (int32_t p = 0; p < inductionPasses.size(); ++p) {
        const InductionPass pass = inductionPasses[p];
        std::memset(passExcluded, 0, count + 1);
        for (int32_t q = 0; q < inductionPasses.size(); ++q) {
            const InductionPass other = inductionPasses[q];
            if (other.slot != pass.slot || other.caught == pass.caught) continue;
            for (int32_t s = other.caught; s < witnesses.slot(other.caught).subtreeEnd; ++s) passExcluded[s] = 1;
        }
        for (int32_t s = pass.caught; s != 0; s = witnesses.slot(s).parent) passExcluded[s] = 0;
        for (int32_t s = pass.caught; s < witnesses.slot(pass.caught).subtreeEnd; ++s) passExcluded[s] = 0;
        ScratchScope passFrame(strings);
        canonicalizer.initializeCanonicalScope(memoryBlock, pass.scope);
        const StrSpan passView = memoryBlock.nameMap.decodeView(pass.scope);
        const ScratchString passValidity = ScratchString::copyFrom(strings, passView.ptr, passView.len);
        if (parameters.trackHistory && trackHistoryLocal) {
            const int cap = parameters.compressor_mode
                ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr;
            const OriginDep zoom[1] = { { expr, validityName } };
            addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner, expr,
                StrSpan(passValidity), OriginTag::zoomIn, zoom, 1, cap);
            memoryBlock.addMailOutOrigin(expr, StrSpan(passValidity), OriginTag::zoomIn, zoom, 1, cap);
        }
        CollectedArena passPublished(&arena);
        witnesses.nextSlot = 1;
        witnesses.setPublicationFilter(ownScopeSlots, pass.slot);
        witnesses.setPassExclusions(passExcluded);
        disintegrateExprCore2(StrSpan(prefixed), instruction, memoryBlock, iteration,
            passPublished, definitions, StrSpan(passValidity), trackHistoryLocal,
            allowOrDisintegration, orSeedLevels, orSeedLevelCount, allowOrProbe, StrSpan(), &witnesses,
            integrationPremise, conjecturePremise);
        witnesses.setPassExclusions(nullptr);
        assert(witnesses.nextSlot == count + 1);
        ColdHashSet<BytesKeyStore> passProducts(&arena, &dirty);
        ColdHashSet<BytesKeyStore> passRules(&arena, &dirty);
        passPublished.forEachKey([&](StrSpan text) {
            if (promotions.size() != 0) passProducts.mint(text);
            if (parentProducts.lookup(text) != 0) return;
            assert(!containsSpan(text, StrSpan("__gl_temporary_witness_", 23)));
            out.statements.append(text, StrSpan(passValidity));
            if (passPublished.isOrSource(text)) out.orSources.append(text, StrSpan(passValidity));
            passPublished.forImpls(text, [&](StrSpan rule, StrSpan validity) {
                assert(!containsSpan(rule, StrSpan("__gl_temporary_witness_", 23)));
                out.implications.append(rule, validity, text);
                if (promotions.size() != 0) passRules.mint(rule);
            });
        });
        // A promoted trial whose admission opened this scope admits here
        // exactly what it listed (D-578).
        for (int32_t q = 0; q < promotions.size(); ++q) {
            if (promotions[q].slot != pass.slot) continue;
            const IntegrationTrialRec record = promotions[q].record;
            for (int32_t i = record.witnessCount; i < record.witnessCount + record.expressionCount; ++i) {
                const int64_t* entry = memoryBlock.lbMemory.integrationTrialContents.find(
                    packInt32Pair(record.trialScope, i));
                assert(entry != nullptr && "disintegrateExpr2: a passed trial's expression list is complete");
                const int32_t kind = static_cast<int32_t>(static_cast<uint64_t>(*entry) >> 32);
                const StrSpan text = memoryBlock.nameMap.decodeView(static_cast<NameId>(static_cast<uint32_t>(*entry)));
                assert((kind == kIntegrationTrialStatement
                        ? parentProducts.lookup(text) != 0 || passProducts.lookup(text) != 0
                        : kind == kIntegrationTrialRule
                            && (parentRules.lookup(text) != 0 || passRules.lookup(text) != 0))
                    && "disintegrateExpr2: a promoted trial publishes its recorded expressions in the scope it opened");
            }
        }
    }
    witnesses.setPublicationFilter(nullptr, 0);
    // The induction hypothesis' own constructions: no recursion-owned
    // template accepts them (I-286).
    if (recursionHead)
        for (int32_t id = 1; id <= count; ++id)
            if (witnesses.slot(id).permanentId != 0)
                memoryBlock.recursionHeadWitnesses.mint(witnesses.permanent(id));
    // A recursion block's constructions over constructed objects: no
    // recursion-owned compound binding accepts them
    // (I-292).
    if (memoryBlock.isPartOfRecursion)
        for (int32_t id = 1; id <= count; ++id)
            if (witnesses.slot(id).permanentId != 0
                && !hasOnlyNumberArguments(witnesses.source(id)))
                memoryBlock.hiddenArgumentWitnesses.mint(witnesses.permanent(id));
    // A recursion block's constructions over a witness of either registry
    // above: no implication subproof naming one of them is opened
    // (I-293). Runs after both registries, so a
    // nested witness sees the entries of the witnesses enclosing it; a
    // source names an enclosing witness by its temporary token.
    if (memoryBlock.isPartOfRecursion)
        for (int32_t id = 1; id <= count; ++id) {
            if (witnesses.slot(id).permanentId == 0) continue;
            StrSpan sourceArgs[ExecutionParameters::MAX_ARITY];
            const int32_t sourceArgCount = getArgsSpans(witnesses.source(id), sourceArgs,
                ExecutionParameters::MAX_ARITY);
            for (int32_t a = 0; a < sourceArgCount; ++a) {
                StrSpan name = sourceArgs[a];
                if (startsWithSpan(name, "u_", 2))
                    name = StrSpan(name.ptr + 2, name.len - 2);
                const int32_t enclosing = witnesses.lookup(name);
                if (enclosing != 0) {
                    assert(witnesses.slot(enclosing).permanentId != 0
                        && "disintegrateExpr2: a published witness's source names an unpublished witness");
                    name = witnesses.permanent(enclosing);
                }
                if (memoryBlock.recursionHeadWitnesses.lookup(name) != 0
                    || memoryBlock.hiddenArgumentWitnesses.lookup(name) != 0) {
                    memoryBlock.builtOnListedWitnesses.mint(witnesses.permanent(id));
                    break;
                }
            }
        }
    if (forceDeep) return count == 0;

    int levels[256];
    int32_t levelCount = 0;
    for (int32_t i = 0; i < orSeedLevelCount; ++i)
        if (orSeedLevels[i] >= 0) levelCount = insertLevelSorted(levels, levelCount, orSeedLevels[i], 256);
    const int32_t storedLevels = lookupStatementLevels(memoryBlock.intStatementLevelsMap,
        memoryBlock.nameMap, expr, validityName);
    if (storedLevels != 0)
        for (int32_t i = 0; i < memoryBlock.intStatementLevelsMap.runLen(storedLevels); ++i) {
            const int level = memoryBlock.intStatementLevelsMap.valueAt(storedLevels, i);
            if (level >= 0) levelCount = insertLevelSorted(levels, levelCount, level, 256);
        }

    const auto highestUnresolved = [&](int32_t id) {
        while (witnesses.slot(id).parent != 0
            && witnesses.slot(witnesses.slot(id).parent).permanentId == 0)
            id = witnesses.slot(id).parent;
        return id;
    };
    for (int32_t id = 1; id <= count; ++id) {
        if (disabled[id]) continue;
        // A blocked slot's demand is present already; it waits for its
        // enclosing slots, whose own parks hold the source.
        if (blocked != nullptr && blocked[id]) continue;
        ScratchScope sourceFrame(strings);
        const TemporaryWitnessSlot value = witnesses.slot(id);
        if (value.permanentId != 0) {
            const ScratchString source = substitute(witnesses.source(id));
            assert(!temporary(StrSpan(source)));
            const NameId scope = memoryBlock.nameMap.lookup(validityName);
            assert(scope != 0);
            const int64_t key = packOriginKey(rejectedSources.canonicalExistenceId(memoryBlock, StrSpan(source), scope), scope);
            const int32_t* old = memoryBlock.existenceReplays.find(key);
            const int32_t state = old ? *old : 0;
            rejectedSources.writeExistenceState(memoryBlock, key, state | kindBit(id));
            continue;
        }
        const int32_t root = highestUnresolved(id);
        const ScratchString source = substitute(witnesses.source(root));
        assert(!temporary(StrSpan(source)));
        const int32_t flags = kindBit(root) | (allowOrDisintegration ? 4 : 0)
            | (inheritedSource ? 8 : 0);
        const int32_t varId = definitions.lookupVar(witnesses.token(id));
        for (int32_t j = 0; j < definitions.elemCount(varId); ++j) {
            ScratchScope elementFrame(strings);
            const ScratchString candidate = substitute(definitions.elemAt(varId, j));
            const ScratchString key = gl::DefinitionExpansion::makeMarkedExprScratch(strings, StrSpan(candidate), witnesses.token(id));
            if (temporary(StrSpan(key))) continue;
            const StrSpan core = extractExpressionSpan(StrSpan(candidate));
            if (value.kind == TemporaryWitnessKind::Algebra) {
                if (this->definitions.operatorNames().find(std::string_view(core.ptr, core.len)) == this->definitions.operatorNames().end()) continue;
                rejectedSources.updateRejectedMap(StrSpan(key), StrSpan(source), value.generation, flags,
                    levels, levelCount, memoryBlock.overallHashMemory, memoryBlock, validityName,
                    nullptr, 0, integrationDepth);
                algebraParks.mint(StrSpan(key));
            } else if (!equalSpans(core, StrSpan("in", 2))) {
                rejectedSources.updateRejectedMapIntegration(StrSpan(key), StrSpan(source), value.generation, flags,
                    levels, levelCount, memoryBlock.overallHashMemory, memoryBlock, validityName,
                    nullptr, 0, integrationDepth);
                // The instructions under the key that the witness-link gate
                // refuses for this route — through its linked arguments or
                // its witness history — record the refusal, so the rendezvous
                // below does not replay the source into them again
                // (D-529, D-522); in a block that split identical goals by
                // premise, so do the copies whose forbidden sets the source
                // hits (D-579).
                StrSpan linked[kMaxLinked];
                const int32_t linkedN = linkedArguments(id, j, root, linked);
                WitnessHistoryProbe history;
                historyProbe(id, root, history);
                if (linkedN > 0 || history.active() || memoryBlock.premiseSplitFiled)
                    admissionMemory.markWitnessLinkRefusals(memoryBlock, StrSpan(key), validityName,
                        linked, linkedN, StrSpan(source), &history);
                integrationParks.mint(StrSpan(key));
            }
        }
    }

    if (memoryBlock.startInt != savedStartInt)
        rejectedSources.retireCompletedExistences(memoryBlock, memoryBlock.nameMap.encode(validityName));

    // Whichever side arrives second completes the rendezvous, including a
    // demand already present in a descendant when an ancestor parks its source.
    for (int32_t i = 1; i <= algebraParks.count(); ++i)
        rejectedSourceReplay.revisitRejected2(algebraParks.keyAt(i), memoryBlock, validityName);
    for (int32_t i = 1; i <= integrationParks.count(); ++i)
        rejectedSourceReplay.revisitRejectedIntegration2(integrationParks.keyAt(i), memoryBlock, validityName);
    rejectedSourceReplay.wakeParkedExistencesAtExistingDemands(memoryBlock, algebraParks, integrationParks);

    ColdHashSet<BytesKeyStore> groups(&arena, &dirty);
    ColdHashSet<PodKeyStore<int32_t>> covered(&arena, &dirty);
    int32_t* groupOf = count == 0 ? nullptr : reinterpret_cast<int32_t*>(
        arena.resolve(arena.alloc(count * 4, 4)));
    for (int32_t id = 1; id <= count; ++id) {
        // Source identity groups the alternative constructions; outer temporary
        // identities still separate distinct enclosing expansions.
        groupOf[id - 1] = disabled[id] == 1 ? 0 : groups.mint(witnesses.source(id));
        if (groupOf[id - 1] != 0
            && (witnesses.slot(id).permanentId != 0 || disabled[id] == 2 || disabled[id] == 5
                || disabled[id] == 6))
            covered.mint(groupOf[id - 1]);
    }
    const bool full = groups.count() == covered.count();
    if (!full) {
        for (int32_t id = 1; id <= count; ++id) {
            if (disabled[id] != 0 || doubleOnly[id] || covered.contains(groupOf[id - 1])) continue;
            const int32_t parent = witnesses.slot(id).parent;
            if (parent != 0 && !covered.contains(groupOf[parent - 1])) continue;
            ScratchScope relayFrame(strings);
            const ScratchString source = substitute(witnesses.source(highestUnresolved(id)));
            assert(!temporary(StrSpan(source)));
            out.relayCompacts.append(StrSpan(source), validityName);
        }
    }
    return full;
}

/// @brief Probe what integrating an expression would disintegrate into,
        ///        inside its sentinel and hypothesis scopes.
        ///
        /// @details
        /// Disintegrates @p expr into a fresh sentinel scope
        /// `product_of_hypo_disintegration_of_integration_goal_<n>_<expr>`
        /// (`<n>` = the name map's name count at entry, so a goal prepared
        /// again at the same scope gets a new scope) without
        /// history, and when some argument has the `(1)` definition set, opens
        /// the hypothesis scope `_var<i>_<x>..._hypo_<expr>` for the typed
        /// variables. A marker expression is left alone. The sentinel serves
        /// only this dry run: its name is minted here (asserted) and it is
        /// queued for the step's scope wipe right after the dry run
        /// (D-496), so no later class change
        /// visits it.
        ///
        /// @param expr         Expression being prepared (caller-stable span).
        /// @param memoryBlock  Owning block.
        /// @param validityName Preparation scope.
        /// @return (void)
        /// @see prepareIntegration — the caller.
void DisintegrationProcessor::disintegrateExprHypothetically(StrSpan expr, Memory& memoryBlock, StrSpan validityName) {
    if (parameters.ban_disintegration) return;

    // Function-level string-tier scope holding the sentinel / hypo payloads and
    // the copyFrom'd sentinelValidity / newValidityName (0% heap); the per-tv
    // (=[Y,Y_copy]) build nests above it.
    const unsigned hypoStrSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : scratchArenas().slotCount() - 1;
    ScratchArena& hypoStrArena = scratchArenas().forSlot(hypoStrSlot);
    ScratchScope hypoStrScope(hypoStrArena);

    // 1. Extract input variables (D16: spans over the stable `expr`).
    StrSpan inputVars[ExecutionParameters::MAX_ARITY];
    const int32_t inputVarN = getArgsSpans(expr, inputVars,
                                           ExecutionParameters::MAX_ARITY);

    // A marker goal is an integration demand, not a hypothesis: `marker`, and
    // `marker1` / `marker2` of the weak hypothesis's two-marker start
    // (D-586).
    for (int32_t i = 0; i < inputVarN; ++i) {
        if (equalSpans(inputVars[i], StrSpan("marker", 6))
            || equalSpans(inputVars[i], StrSpan("marker1", 7))
            || equalSpans(inputVars[i], StrSpan("marker2", 7))) {
            return;
        }
    }

    // 2. Call disintegrateExpr2 FIRST
    // We only need the structural products — no history writes.
    //
    // Dual defense:
    //   (a) trackHistoryLocal=false suppresses all writes inside the
    //       trackExpansionHistory lambda, the negated-existence-expansion
    //       block, and the "or disintegration" addOrigin (primary fix).
    //   (b) We hand disintegrateExpr2 a SENTINEL validity so that *if*
    //       anything ever leaks into exprOriginMap / mailOut despite (a),
    //       the leaked entries are trivially identifiable and assert-caught
    //       in buildStack. The sentinel embeds the real validityName for
    //       traceability.
    //
    // IMPORTANT: the rest of this function must keep using the original
    // `validityName` (for intStatementLevelsMap / equivalenceClassesMap lookups
    // and for constructing `newValidityName`). The sentinel stays local.
    // Sentinel payload embeds the integration goal (expr) so any leaked
    // deposits at sentinel scope are self-identifying. The
    // "product_of_hypo_disintegration_of_integration_goal_" marker lets
    // cross-scope consumers (buildStack, Site A descendant iteration, proof-
    // graph rendering) exclude sentinel-scope orphans without pattern
    // guessing or substring heuristics on random tokens.
    NameId parentValId = memoryBlock.nameMap.encode(validityName);
    // Sentinel payload = ScopeSyntax::kHypotheticalSentinelPrefix + "<n>_" +
    // expr, built explicit-length on the string tier (freed after encodePush);
    // encodePush takes a span. <n> is the name map's name count at entry:
    // names only grow, so every dry run gets its own scope, also when the same
    // goal is prepared again at the same scope (the goal rescans of
    // checkNecessityForEquality do that), and the scope can be wiped right
    // after use (D-496).
    NameId sentinelValId;
    {
        ScratchScope spScope(hypoStrArena);
        const int32_t spLen = static_cast<int32_t>(sizeof(gl::ScopeSyntax::kHypotheticalSentinelPrefix) - 1);
        char number[16];
        const int numberLen = std::snprintf(number, sizeof(number), "%d_",
            static_cast<int>(memoryBlock.nameMap.nameCount()));
        assert(numberLen > 0 && numberLen < static_cast<int>(sizeof(number)));
        const int32_t sLen = spLen + numberLen + expr.len;
        char* sb = hypoStrArena.allocBytes(sLen);
        std::memcpy(sb, gl::ScopeSyntax::kHypotheticalSentinelPrefix, static_cast<size_t>(spLen));
        std::memcpy(sb + spLen, number, static_cast<size_t>(numberLen));
        if (expr.len > 0) std::memcpy(sb + spLen + numberLen, expr.ptr, static_cast<size_t>(expr.len));
        const int32_t namesBefore = memoryBlock.nameMap.nameCount();
        sentinelValId = memoryBlock.nameMap.encodePush(parentValId, StrSpan(sb, sLen));
        assert(memoryBlock.nameMap.nameCount() != namesBefore
            && "disintegrateExprHypothetically: a sentinel scope serves exactly one dry run");
        canonicalizer.initializeCanonicalScope(memoryBlock, sentinelValId);
    }
    // copyFrom: disintegrateExpr2 below mints NameMap heavily, so a raw decodeView
    // would dangle (I-3); the copy rides the string tier (a different arena).
    const ScratchString sentinelValidity = ScratchString::copyFrom(hypoStrArena,
        memoryBlock.nameMap.decodeView(sentinelValId).ptr,
        memoryBlock.nameMap.decodeView(sentinelValId).len);
    const unsigned deSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    DisintProducts out(&genScratchArenas().forSlot(deSlot));
    disintegrateExpr2(StrSpan(expr), memoryBlock, -1, true, StrSpan(sentinelValidity), out,
        /*trackHistoryLocal=*/false);
    // The dry run was the sentinel's only use: it is wiped at the end of this
    // step (D-496). History survives a wipe
    // (I-121), so the wipe hides nothing from the chapter export. The
    // hypothesis demands of the dry run's eats are not filed: their scopes lie
    // below the sentinel and leave with it (D-575).
    memoryBlock.pendingWipeScopes.mint(sentinelValId);
    memoryBlock.intValidityNamesToFilter.mint(sentinelValId);

    // Only look at new statements (the out.statements channel). forEachSorted walks
    // in EWV::operator< order, byte-identical to the former std::set order; count()
    // is the deduped entry count == the former newStatementsEv.size().
    if (out.statements.count() == 1)
    {
        return;
    }
    // D19(b) user zero-heap order: iterate the disintegrateExpr2 EWV set directly
    // rather than materializing a std::set<std::string> of the originals. Byte
    // contract: ExpressionWithValidity::operator< is (original, validityName)
    // lexicographic (mail_types.hpp), so reading `.original` in set order yields
    // the former std::set<std::string> order EXACTLY, with equal originals
    // ADJACENT. Step 3's first-match goto = lex-smallest matching original in both
    // forms; step 6 is order-immaterial (count>1 break); step 7 keys on the
    // CONSTANT newValidityName, so the first visit of each distinct original
    // registers (identical push_back + mint) and each adjacent duplicate hits the
    // now-registered probe and continues — the same registration sequence as the
    // deduped set. The transient EncodedExpression stays deleted (D25 doors).

    // 3. Search for defSets for each input variable in the new statements (D20).
    // targetVars is a first-seen-order StrSpan run over `expr` (spans into
    // inputVars); the defSet value stays a std::string (a compiled-def config value).
    StrSpan targetVars[ExecutionParameters::MAX_ARITY];
    int32_t targetVarN = 0;

    for (int32_t vi = 0; vi < inputVarN; ++vi) {
        const StrSpan var = inputVars[vi];
        bool foundDef = false;
        // foundDefSetVal is a span into the never-deloaded config (coreConfig,
        // I-137), stable across the walk; a first-match guard reproduces the
        // former `goto def_found` (the loop is a pure search, so full iteration
        // with a first-match guard is byte-identical).
        StrSpan foundDefSetVal;

        out.statements.forEachSorted([&](StrSpan stmt, StrSpan /*v*/) {
            if (foundDef) return;
            const ce::CoreExpressionConfig* cfg =
                this->definitions.coreConfig(extractExpressionSpan(stmt));

            // "search for at least one expression which has entry in coreExpressionMap"
            if (cfg != nullptr) {
                StrSpan sArgs[ExecutionParameters::MAX_ARITY];
                const int32_t sArgN = getArgsSpans(stmt, sArgs,
                                                   ExecutionParameters::MAX_ARITY);

                // Check if 'var' is an argument in this statement
                for (int32_t k = 0; k < sArgN; ++k) {
                    if (equalSpans(sArgs[k], var)) {
                        // "there read defSet"
                        const std::pair<std::string, bool>* ds =
                            gl::CompiledDefinitionRegistry::findDefinitionSet(cfg, k + 1);
                        if (ds != nullptr) {
                            foundDefSetVal = StrSpan(ds->first);
                            foundDef = true;
                            return; // first match (== the former goto def_found)
                        }
                    }
                }
            }
        });

        // "assert u found def set for each var"
        assert(foundDef);

        // "if there is at least one eligible var proceed"
        // Eligible means defSet == "(1)" based on context.
        // Deduplicate: if expr has the same variable in multiple argument
        // positions (e.g. (f[v,v,w])), inputVars contains duplicates.
        // Each variable must appear in newValidityName only once.
        if (equalSpans(foundDefSetVal, StrSpan("(1)", 3))) {
            bool present = false;
            for (int32_t t = 0; t < targetVarN; ++t)
                if (equalSpans(targetVars[t], var)) { present = true; break; }
            if (!present) {
                targetVars[targetVarN++] = var;
            }
        }
    }

    // "otherwise return"
    if (targetVarN == 0) {
        return;
    }

    // 4. Create a new validity name
    // Format: old + "_var0_x_var1_y_hypo_expr"
    // Example: main__var0_x_var1_y_hypo_(in[x,1])
    // hypoPayload = concat over targetVars of ("_var<i>_" + targetVars[i]) then
    // "_hypo_" + expr, built explicit-length on the string tier (%d ==
    // std::to_string for non-negative i, freed after encodePush); encodePush
    // takes a span.
    NameId hypoParentId = memoryBlock.nameMap.encode(validityName);
    NameId hypoNewValidityId;
    {
        ScratchScope hpScope(hypoStrArena);
        int32_t hpLen = 0;
        char di[16];
        for (int32_t i = 0; i < targetVarN; ++i) {
            const int dn = std::snprintf(di, sizeof(di), "%d", i);
            hpLen += 4 + dn + 1 + targetVars[i].len;   // "_var" + <i> + "_" + var
        }
        static const char HYPO_MID[] = "_hypo_";
        const int32_t hmLen = static_cast<int32_t>(sizeof(HYPO_MID) - 1);
        hpLen += hmLen + expr.len;
        char* hb = hypoStrArena.allocBytes(hpLen);
        int32_t hat = 0;
        for (int32_t i = 0; i < targetVarN; ++i) {
            std::memcpy(hb + hat, "_var", 4); hat += 4;
            const int dn = std::snprintf(di, sizeof(di), "%d", i);
            std::memcpy(hb + hat, di, static_cast<size_t>(dn)); hat += dn;
            hb[hat++] = '_';
            if (targetVars[i].len > 0) {
                std::memcpy(hb + hat, targetVars[i].ptr, static_cast<size_t>(targetVars[i].len));
                hat += targetVars[i].len;
            }
        }
        std::memcpy(hb + hat, HYPO_MID, static_cast<size_t>(hmLen)); hat += hmLen;
        if (expr.len > 0) { std::memcpy(hb + hat, expr.ptr, static_cast<size_t>(expr.len)); hat += expr.len; }
        assert(hat == hpLen);
        hypoNewValidityId = memoryBlock.nameMap.encodePush(hypoParentId, StrSpan(hb, hpLen));
        canonicalizer.initializeCanonicalScope(memoryBlock, hypoNewValidityId);
    }
    // copyFrom: step 6's addExprToMemoryBlock + step 7's encode/nameMap.encode
    // mint NameMap, so a raw decodeView would dangle (I-3); the copy rides the
    // string tier (a different arena) and survives every mint.
    const ScratchString newValidityName = ScratchString::copyFrom(hypoStrArena,
        memoryBlock.nameMap.decodeView(hypoNewValidityId).ptr,
        memoryBlock.nameMap.decodeView(hypoNewValidityId).len);

    // 5. Find levels — caller-owned stack run; the sorted-unique union
    // (origin levels + the LB's own level) via insertLevelSorted covers
    // both the miss-branch singleton and the unconditional insert. The
    // filtered read keeps the {-1} non-derived tier out of the union.
    int lvRun[256];
    int32_t lvN = 0;
    const int32_t originLvlsId = lookupStatementLevels(
        memoryBlock.intStatementLevelsMap, memoryBlock.nameMap, expr, validityName);

    if (originLvlsId) {
        lvN = coldIntRunNonNegAt(memoryBlock.intStatementLevelsMap, originLvlsId,
                                 lvRun, 256);
    }
    lvN = insertLevelSorted(lvRun, lvN, memoryBlock.level, 256);

    // 6. Check Duplication and Add Equalities
    bool anyDuplicationFound = false;

    for (int32_t tv = 0; tv < targetVarN; ++tv) {
        const StrSpan var = targetVars[tv];
        bool foundDuplicationForVar = false;

        // Check duplication in newStatements (D22) — first-match guard (the loop
        // is order-immaterial: it only sets a bool on any count>1).
        out.statements.forEachSorted([&](StrSpan e, StrSpan /*v*/) {
            if (foundDuplicationForVar) return;
            StrSpan eArgs[ExecutionParameters::MAX_ARITY];
            const int32_t eArgN = getArgsSpans(e, eArgs,
                                               ExecutionParameters::MAX_ARITY);
            int count = 0;
            for (int32_t a = 0; a < eArgN; ++a) {
                if (equalSpans(eArgs[a], var)) count++;
            }
            if (count > 1) foundDuplicationForVar = true;
        });

        if (foundDuplicationForVar) {
            anyDuplicationFound = true;

            // Check if equality exists in validityName (not "main")
            bool hasExistingEquality = false;
            {
                const NameId varId = memoryBlock.nameMap.lookup(var);
                // Zero-copy view walk over the scope's class run — the former
                // decodeClassesAt heap snapshot. The scan is a pure read consumed
                // BEFORE the addExprToMemoryBlock mint below, so the peeked views
                // stay valid. Bucket resolution mirrors decodeClassesAt.
                const NameId dupValId = memoryBlock.nameMap.lookup(validityName);
                if(dupValId!=0) canonicalizer.initializeCanonicalScope(memoryBlock,dupValId);
                const int32_t bucketId = (dupValId == 0)
                    ? 0 : memoryBlock.canonicalEquivalenceClassesMap.lookup(dupValId);
                if (bucketId != 0) {
                    const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                        ? static_cast<unsigned>(gl::WorkerContext::coreId())
                        : genScratchArenas().slotCount() - 1;
                    ScratchArena& gArena = genScratchArenas().forSlot(slot);
                    const int32_t cc = memoryBlock.canonicalEquivalenceClassesMap.runLen(bucketId);
                    for (int32_t ci = 0; ci < cc; ++ci) {
                        ScratchScope viewScope(gArena);
                        int32_t blen = 0;
                        const char* bp = memoryBlock.canonicalEquivalenceClassesMap.peekRecordBytes(
                            bucketId, ci, blen, gArena);
                        const EquivalenceClassView view{ bp, blen };
                        if (varId != 0 && view.memberCount() > 1
                            && classHasMember(view, varId)) {
                            hasExistingEquality = true;
                            break;
                        }
                    }
                }
            }

            if (!hasExistingEquality) {
                // Build "(=[Y,Y_copy])" on the string arena then u_-strip (D23, the
                // reactToHypo precedent): "(=[" + Y + "," + Y + "_copy])".
                const unsigned hSlot = (gl::WorkerContext::coreId() >= 0)
                    ? static_cast<unsigned>(gl::WorkerContext::coreId())
                    : scratchArenas().slotCount() - 1;
                ScratchArena& hArena = scratchArenas().forSlot(hSlot);
                ScratchScope hScope(hArena);
                const StrSpan Y = var;
                const int32_t elen = 11 + 2 * Y.len;  // 3 + Y + 1 + Y + 7 ("_copy])")
                char* eb = hArena.allocBytes(elen);
                int32_t eat = 0;
                eb[eat++] = '('; eb[eat++] = '='; eb[eat++] = '[';
                if (Y.len > 0) std::memcpy(eb + eat, Y.ptr, static_cast<size_t>(Y.len));
                eat += Y.len;
                eb[eat++] = ',';
                if (Y.len > 0) std::memcpy(eb + eat, Y.ptr, static_cast<size_t>(Y.len));
                eat += Y.len;
                const char eSfx[] = "_copy])";
                std::memcpy(eb + eat, eSfx, 7); eat += 7;
                assert(eat == elen);
                const ScratchString equalityExpr = ScratchString::wrap(hArena, eb, elen);
                const ScratchString finalExpr =
                    gl::DefinitionExpansion::removeUPrefixScratch(hArena, StrSpan(equalityExpr));

                // Dead-end axiom — see "variable copy" tag
                const TransientOrigin origin{ true, OriginTag::variableCopy, nullptr, 0 };

                statementMemory.addExprToMemoryBlock(StrSpan(finalExpr), memoryBlock, -1, 0, lvRun, lvN, origin, -1, -1, validityName, false);
            }
        }
    }

    // 7. Deposit NEW STATEMENTS with NEW validity name on internal mail

    // The constituents ride sameIterationInternalMail — the absorb runs each
    // through the full kernel pipeline (addStatement shape dispatch with the
    // equality / negated-equality gateways' mirror pair registration, Site F
    // ancestor dedup, equivalence classes), the same channel the or-branch
    // seeds ([I-176](../../../docs/agentic_swdd/30_invariants.md#i-176)) and
    // the ordis revival use. No direct registration here: a direct upsert
    // cannot maintain the gateways' mirror pair invariant. stmtSpan aliases
    // out.statements' storage and newVldSpan the owned decode copy
    // `newValidityName`; the deposit's two flat encodes write only the names
    // table, so neither span moves (I-3).
    out.statements.forEachSorted([&](StrSpan stmtSpan, StrSpan /*v*/) {
        const StrSpan newVldSpan(newValidityName);
        memoryBlock.mutatedThisBurst = true;
        insertInternalStatement(memoryBlock.sameIterationInternalMail,
            memoryBlock.nameMap, stmtSpan, newVldSpan, lvRun, lvN);
        if (parameters.trackHistory) {
            // Terminal `hypothesis` line — one tag, zero dependencies, at the
            // hypo validity ONLY. Hypothesis constituents steer proof
            // direction but are not part of any proof: no walk continues
            // through this line, and it must never be written at the parent
            // or main scope (the 2026-04-10 orphan-origin buildStack abort).
            addInternalMailOrigin(memoryBlock.sameIterationInternalMail,
                memoryBlock.originInterner, stmtSpan, newVldSpan,
                OriginTag::hypothesis, nullptr, 0,
                (parameters.compressor_mode
                     ? parameters.compressor_max_origins_per_expr
                     : parameters.max_origin_per_expr));
        }
    });
}
}
