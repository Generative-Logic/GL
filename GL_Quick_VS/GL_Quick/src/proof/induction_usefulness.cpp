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
#include "induction_usefulness.hpp"
#include "../compiler/definition_expansion.hpp"
#include "../logical_block/scope_syntax.hpp"
#include "../logical_block/statement_memory.hpp"
#include <algorithm>
#include <cstdio>

namespace gl {

namespace {

/// @brief The plain arguments of a flat expression as owned strings.
///
/// @details Grid-build helper over `getArgsSpans`; the expression may be
/// negated (the leading `!` is transparent to the argument list).
///
/// @param expression A flat expression.
/// @return Its arguments in order.
std::vector<std::string> argumentsOf(const std::string& expression) {
    StrSpan spans[ExecutionParameters::MAX_ARITY];
    const int32_t count = getArgsSpans(StrSpan(expression), spans, ExecutionParameters::MAX_ARITY);
    std::vector<std::string> out;
    out.reserve(static_cast<std::size_t>(count));
    for (int32_t i = 0; i < count; ++i) out.push_back(spans[i].toStdString());
    return out;
}

/// @brief The core name of a flat expression as an owned string.
///
/// @details The leading `!` of a negated expression is not part of the core.
///
/// @param expression A flat expression.
/// @return Its core name, e.g. `fold`.
std::string coreOf(const std::string& expression) {
    return extractExpressionSpan(StrSpan(expression)).toStdString();
}

/// @brief The usefulness scope's full validity name as a span.
///
/// @details Over the static `ScopeSyntax::kUsefulnessScopeName`.
///
/// @return `main_boundary_is_useful`.
StrSpan usefulnessScopeName() {
    return StrSpan(ScopeSyntax::kUsefulnessScopeName,
        static_cast<int32_t>(sizeof(ScopeSyntax::kUsefulnessScopeName) - 1));
}

} // anonymous namespace

/// @brief Bind the test to the prover's canonicalizer, definitions and
///        proof burst counter.
///
/// @details The component owns no state; it borrows the three
/// collaborators for the prover's lifetime. The burst counter is the
/// prover's own (`Prover::proofBurstNumber`, 1-based, warm-up included): a
/// test still pending after burst @ref kTestBursts has failed.
///
/// @param canonicalizer    Class-canonical forms at a scope.
/// @param definitions      Compiled definitions and their configuration.
/// @param proofBurstNumber The number of the batch's current proof burst.
/// @return None; construction binds the references.
InductionUsefulness::InductionUsefulness(Canonicalizer& canonicalizer,
                                         const CompiledDefinitionRegistry& definitions,
                                         const int& proofBurstNumber)
    : canonicalizer(canonicalizer), definitions(definitions), proofBurstNumber(proofBurstNumber) {}

/// @brief Return a flat expression with one argument replaced.
///
/// @details Rebuilds `(<core>[a0,...,an])` with argument @p slot set to
/// @p value; the expression must be positive and flat.
///
/// @param expression A positive flat expression.
/// @param slot       The argument index to replace (asserted in range).
/// @param value      The replacement argument.
/// @return The rebuilt expression.
std::string InductionUsefulness::withArgumentAt(const std::string& expression, int32_t slot,
                                                const std::string& value) {
    assert(!expression.empty() && expression[0] == '('
        && "withArgumentAt: a positive flat expression");
    const std::vector<std::string> arguments = argumentsOf(expression);
    assert(slot >= 0 && slot < static_cast<int32_t>(arguments.size())
        && "withArgumentAt: slot outside the argument list");
    std::string out = "(" + coreOf(expression) + "[";
    for (std::size_t i = 0; i < arguments.size(); ++i) {
        if (i > 0) out += ",";
        out += (static_cast<int32_t>(i) == slot) ? value : arguments[i];
    }
    out += "])";
    return out;
}

/// @brief Decide whether a weak step block's hypothesis qualifies for the
///        usefulness test, and build the test when it does.
///
/// @details
/// The hypothesis qualifies when its head is a positive operator
/// expression with exactly one output argument, that output is a free
/// variable (no `u_` prefix) and is listed in the operator's
/// `uniquely_definable`, every other head argument is fixed or named in
/// a hypothesis premise, and the conjecture has exactly one positive
/// premise with the head's operator. On success @p out receives the
/// plain-form premises and head (output replaced by the test variable
/// `test_level_<blockLevel>_0`), the key (that premise with its output
/// written as `marker`) and the open slot. A hypothesis that does not
/// qualify is a defined result: most do not.
///
/// @param conjectureChain The conjecture's premises, anchor included, in
///                        order (plain form).
/// @param anchorName      The batch anchor's name (`AnchorFTA`, ...).
/// @param hypothesisChain The hypothesis premises (`u_` form), as built by
///                        `ProofGridBuilder::createAuxyImplication`.
/// @param hypothesisHead  The hypothesis head (`u_` form).
/// @param blockLevel      The step block's level (the test variable's level).
/// @param out             Receives the test on success.
/// @return `true` iff the hypothesis qualifies.
/// @see arm.
bool InductionUsefulness::qualify(const std::vector<std::string>& conjectureChain,
                                  const std::string& anchorName,
                                  const std::vector<std::string>& hypothesisChain,
                                  const std::string& hypothesisHead,
                                  int blockLevel,
                                  Test& out) const {
    assert(!hypothesisHead.empty() && "qualify: a hypothesis without a head");
    if (hypothesisHead[0] == '!') return false;
    const std::string core = coreOf(hypothesisHead);
    if (definitions.operatorNames().find(std::string_view(core)) == definitions.operatorNames().end())
        return false;
    const ce::CoreExpressionConfig* cfg = definitions.coreConfig(StrSpan(core));
    assert(cfg != nullptr && "qualify: an operator without its configuration");
    if (cfg->outputIndices.size() != 1) return false;
    const int32_t openSlot = cfg->outputIndices[0];
    if (std::find(cfg->uniquelyDefinableIndices.begin(), cfg->uniquelyDefinableIndices.end(), openSlot)
        == cfg->uniquelyDefinableIndices.end()) return false;

    const std::vector<std::string> headArguments = argumentsOf(hypothesisHead);
    assert(openSlot < static_cast<int32_t>(headArguments.size())
        && "qualify: the output slot lies outside the head's arguments");
    if (startsWith(headArguments[openSlot], "u_", 2)) return false;

    // Every other head argument is fixed or named by a hypothesis premise.
    for (std::size_t k = 0; k < headArguments.size(); ++k) {
        if (static_cast<int32_t>(k) == openSlot) continue;
        const std::string& argument = headArguments[k];
        if (startsWith(argument, "u_", 2)) continue;
        bool named = false;
        for (const std::string& premise : hypothesisChain) {
            const std::vector<std::string> premiseArguments = argumentsOf(premise);
            named = named || std::find(premiseArguments.begin(), premiseArguments.end(), argument)
                != premiseArguments.end();
        }
        if (!named) return false;
    }

    // The one conjecture premise with the head's operator.
    const std::string* sameCore = nullptr;
    int sameCoreCount = 0;
    bool anchorSeen = false;
    for (const std::string& premise : conjectureChain) {
        if (coreOf(premise) == anchorName) {
            anchorSeen = true;
            continue;
        }
        if (premise[0] == '!' || coreOf(premise) != core) continue;
        sameCore = &premise;
        ++sameCoreCount;
    }
    assert(anchorSeen && "qualify: the conjecture has no anchor premise");
    (void)anchorSeen;
    if (sameCoreCount != 1) return false;
    assert(argumentsOf(*sameCore).size() == headArguments.size()
        && "qualify: one operator with two arities");

    char name[64];
    const int written = std::snprintf(name, sizeof(name), "test_level_%d_0", blockLevel);
    assert(written > 0 && written < static_cast<int>(sizeof(name)) && "qualify: test variable name overflow");

    out = Test{};
    for (const std::string& premise : hypothesisChain)
        out.premises.push_back(DefinitionExpansion::removeUPrefixFromArguments(premise));
    out.testVariable = std::string(name, static_cast<std::size_t>(written));
    out.head = withArgumentAt(DefinitionExpansion::removeUPrefixFromArguments(hypothesisHead),
        openSlot, out.testVariable);
    out.key = withArgumentAt(*sameCore, openSlot, "marker");
    out.openSlot = openSlot;
    return true;
}

/// @brief Record a planted test on its step block and mark it pending.
///
/// @details Called by the grid build right after the test's premises and
/// head entered the step block at `main_boundary_is_useful` through the
/// statement door. Stores the scope, head and key ids and the open slot
/// on the block (`Memory::usefulnessScopeId`, `usefulnessHeadId`,
/// `usefulnessKeyId`, `usefulnessOpenSlot`) and sets
/// `Memory::usefulnessTest` to `Pending`.
///
/// @param stepBlock The weak step block (resident, at grid build).
/// @param test      The planted test.
/// @return None.
/// @invariant The block carries no test yet, and the head registered at
///            the scope (both asserted).
/// @see qualify, settle.
void InductionUsefulness::arm(Memory& stepBlock, const Test& test) {
    assert(stepBlock.usefulnessTest == Memory::UsefulnessTest::None
        && "arm: the step block already carries a usefulness test");
    assert(test.openSlot >= 0 && "arm: an unqualified test");
    const NameId scope = stepBlock.nameMap.lookup(usefulnessScopeName());
    assert(scope != 0 && stepBlock.nameMap.parentOf(scope) == NameMap::MAIN_ID
        && "arm: the usefulness scope is not a child of main");
    const NameId head = stepBlock.nameMap.lookup(StrSpan(test.head));
    assert(head != 0 && stepBlock.intKnownStatements.find(StatementKey{ head, scope }) != nullptr
        && "arm: the planted head did not register at the usefulness scope");
    stepBlock.usefulnessScopeId = scope;
    stepBlock.usefulnessHeadId = head;
    stepBlock.usefulnessKeyId = stepBlock.nameMap.encode(StrSpan(test.key));
    stepBlock.usefulnessOpenSlot = test.openSlot;
    stepBlock.usefulnessTest = Memory::UsefulnessTest::Pending;
}

/// @brief Whether the key catches the planted head under the usefulness
///        scope's classes.
///
/// @details Reads the head and the key through the classes at
/// `main_boundary_is_useful` (text-only `canonicalFormAtScope`, which
/// mints nothing) and compares them: the same operator and every argument
/// equal except the open slot. Heap-free; the canonical texts ride the
/// per-slot string scratch and are released on return.
///
/// @param stepBlock A step block with a pending test (asserted).
/// @return `true` iff the key catches the head: the step is useless.
/// @see settle.
bool InductionUsefulness::caught(Memory& stepBlock) const {
    assert(stepBlock.usefulnessTest == Memory::UsefulnessTest::Pending
        && "caught: the block has no pending usefulness test");
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope frame(strings);
    // canonicalFormAtScope mints nothing, so the decoded spans stay valid.
    const StrSpan scope = stepBlock.nameMap.decodeView(stepBlock.usefulnessScopeId);
    int32_t noLevels = 0;
    const CanonicalForm head = canonicalizer.canonicalFormAtScope(
        stepBlock.nameMap.decodeView(stepBlock.usefulnessHeadId), scope, stepBlock, strings,
        nullptr, noLevels, 0, false, false, false);
    const CanonicalForm key = canonicalizer.canonicalFormAtScope(
        stepBlock.nameMap.decodeView(stepBlock.usefulnessKeyId), scope, stepBlock, strings,
        nullptr, noLevels, 0, false, false, false);
    assert(equalSpans(extractExpressionSpan(head.text), extractExpressionSpan(key.text))
        && "caught: the key and the head carry different operators");
    StrSpan headArguments[ExecutionParameters::MAX_ARITY];
    StrSpan keyArguments[ExecutionParameters::MAX_ARITY];
    const int32_t headCount = getArgsSpans(head.text, headArguments, ExecutionParameters::MAX_ARITY);
    const int32_t keyCount = getArgsSpans(key.text, keyArguments, ExecutionParameters::MAX_ARITY);
    assert(headCount == keyCount && stepBlock.usefulnessOpenSlot < headCount
        && "caught: the key and the head differ in arity");
    for (int32_t k = 0; k < headCount; ++k) {
        if (k == stepBlock.usefulnessOpenSlot) continue;
        if (!equalSpans(headArguments[k], keyArguments[k])) return false;
    }
    return true;
}

/// @brief Settle a pending usefulness test at the end of a standard
///        processing call.
///
/// @details Does nothing for a block without a pending test. A caught
/// test becomes `Caught` (the barrier switches the block off,
/// `ScopeLifecycle::switchOffUselessStep`); a test still uncaught after
/// burst @ref kTestBursts of the batch's proof becomes `Done`. In both cases
/// the scope is queued on `Memory::pendingWipeScopes`, so the same call's
/// scope cleanup wipes it. Heap-free.
///
/// @param stepBlock The block whose standard processing call ends.
/// @return None.
/// @see caught, arm.
void InductionUsefulness::settle(Memory& stepBlock) const {
    if (stepBlock.usefulnessTest != Memory::UsefulnessTest::Pending) return;
    if (caught(stepBlock)) {
        stepBlock.usefulnessTest = Memory::UsefulnessTest::Caught;
    } else if (proofBurstNumber > kTestBursts) {
        stepBlock.usefulnessTest = Memory::UsefulnessTest::Done;
    } else {
        return;
    }
    if (stepBlock.pendingWipeScopes.lookup(stepBlock.usefulnessScopeId) == 0)
        stepBlock.pendingWipeScopes.mint(stepBlock.usefulnessScopeId);
}

} // namespace gl
