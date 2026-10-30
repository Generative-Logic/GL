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

/// @file
/// @brief Unit tests for the resumable request gates — `resetRequestGateState`,
///        `foldExprIntoRequestGates`, `foldPrefixIntoRequestGates` and
///        `requestGateStateAccepts`, the core the grow search folds one premise
///        at a time instead of re-scanning every candidate.
///
/// @details
/// The oracle is `requestGatesPass` over the whole candidate. The verdict must
/// match on every input: the gates decide whether a request is ever probed
/// against an owner-set map, so a disagreement changes which rules fire.
///
/// Four tests:
///
/// 1. A deterministic corpus of premises covering hypotheses, anchors, several
///    validity scopes and iteration arguments; every candidate is folded as
///    prefix-plus-one and compared with the whole-candidate verdict. The test
///    also asserts both outcomes occur, so a gate that accepted everything
///    could not pass silently.
/// 2. The production pattern — one conflict-free prefix folded once, then many
///    different last premises with a summary rewind in between. A forgotten
///    rollback shows here: the second position would inherit the first's
///    secondary variables.
/// 3. Named gate cases, one per branch: secondary cap, hypothesis-key length
///    cap, hypothesis scope agreement, the main-scope anchor exemption, and the
///    key-length cap.
/// 4. The `_orint_` widening branch of the verdict, both the accepting shape
///    (one shared `_orint_` scope) and the two refusing ones (mixed scopes, and
///    a single scope that is not an `_orint_` branch).
///
/// The corpus comes from a fixed-seed linear congruential sequence, so the
/// tests are byte-reproducible across hosts and runs.

#include <cstddef>
#include <utility>
#include "test_harness.hpp"
#include "prover_fixture.hpp"
#include <thread>
#include "hash_request_generator_fixture.hpp"

#include "../memory.hpp"
#include "../prover.hpp"

#include <cstdint>

namespace {

    // Deterministic 32-bit source. No <random>, no clock seed. Returns the
    // HIGH bits: a linear congruential generator's low bits have a period of a
    // few values, so drawing small ranges from them collapses the corpus — the
    // first version of this file did exactly that and produced 600 candidates
    // that every gate accepted.
    uint32_t nextGateRandom(uint32_t& s) {
        s = s * 1664525u + 1013904223u;
        return s >> 16;
    }

    // A gate-relevant premise. Only isHypo, isAnchor, validityId, arity,
    // argIteration and argFullId reach the gates.
    gl::IntEncodedExpr gateExpr(gl::NameId validityId, int32_t isHypo,
                                int32_t isAnchor, int32_t arity,
                                const gl::NameId* secondaryIds) {
        gl::IntEncodedExpr e{};
        e.nameId = 7;
        e.validityId = validityId;
        e.isHypo = isHypo;
        e.isAnchor = isAnchor;
        e.arity = arity;
        for (int32_t a = 0; a < arity; ++a) {
            e.argIteration[a] = 0;          // > -1 makes the argument a secondary
            e.argFullId[a] = secondaryIds[a];
            e.argId[a] = secondaryIds[a];
        }
        return e;
    }

    // Whole-candidate verdict vs prefix-fold + one appended premise + verdict,
    // which is exactly what the grow search does per position.
    bool gatesMatchOnSplit(gl::tests::ProverFixture& ea, const gl::Memory& m,
                           const gl::IntEncodedExpr* const* exprs,
                           gl::NameId count, gl::NameId mainValidityId,
                           bool& verdictOut) {
        const bool oracle =
            gl::tests::HashRequestGeneratorFixture::requestGatesPass(ea.requestGenerator, exprs, count, m, mainValidityId);

        const gl::TypedColdSet<gl::NameId>& prodRecIds =
            ((((m).overallHashMemory)).productsOfRecursionIds);
        gl::RequestGateState st;
        gl::tests::HashRequestGeneratorFixture::resetRequestGateState(st);

        bool folded = true;
        for (gl::NameId k = 0; k + 1 < count && folded; ++k)
            folded = gl::tests::HashRequestGeneratorFixture::foldExprIntoRequestGates(
                *exprs[k], st, prodRecIds, mainValidityId);
        if (folded)
            folded = gl::tests::HashRequestGeneratorFixture::foldExprIntoRequestGates(
                *exprs[count - 1], st, prodRecIds, mainValidityId);

        const bool split = folded
            && gl::tests::HashRequestGeneratorFixture::requestGateStateAccepts(ea.requestGenerator, st, exprs, count, m, mainValidityId);

        verdictOut = oracle;
        return split == oracle;
    }

}  // namespace

// A corpus that reaches every gate: the folded verdict must equal the
// whole-candidate verdict on all of it, and both verdicts must actually occur.
TEST(request_gates, fold_split_matches_whole_candidate_verdict) {
    gl::tests::ProverFixture ea("Peano");
    ea.parameters.maxLenHypoKey = 2;
    ea.parameters.maxNumberSecondaryVariables = 2;
    ea.parameters.maxNumberSecondaryVariablesOrint = 2;

    gl::Memory m;
    ((((m).overallHashMemory)).maxKeyLength) = 6;
    const gl::NameId mainVid = gl::NameMap::MAIN_ID;
    const gl::NameId hypoA = ((m).nameMap).encodePush(mainVid, "hypo_a");
    const gl::NameId hypoB = ((m).nameMap).encodePush(mainVid, "hypo_b");
    const gl::NameId scopes[3] = { mainVid, hypoA, hypoB };

    uint32_t seed = 424242u;
    gl::IntEncodedExpr pool[48];
    for (int p = 0; p < 48; ++p) {
        const int32_t arity = static_cast<int32_t>(nextGateRandom(seed) % 4u);
        gl::NameId args[4];
        for (int32_t a = 0; a < arity; ++a)
            args[a] = static_cast<gl::NameId>(300u + nextGateRandom(seed) % 4u);
        // Each draw is sequenced into its own local: argument evaluation order
        // is unspecified in C++, and this corpus must be identical on every
        // compiler.
        const gl::NameId vid = scopes[nextGateRandom(seed) % 3u];
        const int32_t isHypo = static_cast<int32_t>(nextGateRandom(seed) % 2u);
        const int32_t isAnchor = static_cast<int32_t>(nextGateRandom(seed) % 2u);
        pool[p] = gateExpr(vid, isHypo, isAnchor, arity, args);
    }

    int accepted = 0;
    int refused = 0;
    for (int trial = 0; trial < 600; ++trial) {
        const gl::NameId count = static_cast<gl::NameId>(
            1u + nextGateRandom(seed) % 5u);
        const gl::IntEncodedExpr* ptrs[gl::ExecutionParameters::MAX_EXPRESSIONS];
        for (gl::NameId k = 0; k < count; ++k)
            ptrs[k] = &pool[nextGateRandom(seed) % 48u];

        bool verdict = false;
        ASSERT_TRUE(gatesMatchOnSplit(ea, m, ptrs, count, mainVid, verdict));
        if (verdict) ++accepted; else ++refused;
    }
    // Neither side may be empty, or the agreement above would be vacuous.
    ASSERT_TRUE(accepted > 0);
    ASSERT_TRUE(refused > 0);
}

// The production pattern: one prefix folded once, many last premises, a summary
// rewind in between. This is the only place a forgotten rollback shows.
TEST(request_gates, rewind_reuses_one_prefix_across_positions) {
    gl::tests::ProverFixture ea("Peano");
    ea.parameters.maxLenHypoKey = 3;
    ea.parameters.maxNumberSecondaryVariables = 3;
    ea.parameters.maxNumberSecondaryVariablesOrint = 3;

    gl::Memory m;
    ((((m).overallHashMemory)).maxKeyLength) = 6;
    const gl::NameId mainVid = gl::NameMap::MAIN_ID;
    const gl::NameId hypoA = ((m).nameMap).encodePush(mainVid, "hypo_a");
    const gl::TypedColdSet<gl::NameId>& prodRecIds =
        ((((m).overallHashMemory)).productsOfRecursionIds);

    // A conflict-free prefix: no premise is a hypothesis, so folding it can
    // never refuse and foldPrefixIntoRequestGates' assert holds.
    uint32_t seed = 909090u;
    gl::IntEncodedExpr prefixPool[4];
    for (int p = 0; p < 4; ++p) {
        gl::NameId args[2] = {
            static_cast<gl::NameId>(400u + nextGateRandom(seed) % 3u),
            static_cast<gl::NameId>(400u + nextGateRandom(seed) % 3u) };
        prefixPool[p] = gateExpr(mainVid, 0, 0, 2, args);
    }

    gl::IntEncodedExpr lastPool[24];
    for (int p = 0; p < 24; ++p) {
        const int32_t arity = static_cast<int32_t>(nextGateRandom(seed) % 3u);
        gl::NameId args[2];
        for (int32_t a = 0; a < arity; ++a)
            args[a] = static_cast<gl::NameId>(400u + nextGateRandom(seed) % 5u);
        const gl::NameId vid = (nextGateRandom(seed) % 2u) ? mainVid : hypoA;
        const int32_t isHypo = static_cast<int32_t>(nextGateRandom(seed) % 2u);
        const int32_t isAnchor = static_cast<int32_t>(nextGateRandom(seed) % 2u);
        lastPool[p] = gateExpr(vid, isHypo, isAnchor, arity, args);
    }

    for (gl::NameId prefixCount = 0; prefixCount <= 3; ++prefixCount) {
        const gl::IntEncodedExpr* ptrs[gl::ExecutionParameters::MAX_EXPRESSIONS];
        for (gl::NameId k = 0; k < prefixCount; ++k) ptrs[k] = &prefixPool[k];

        gl::RequestGateState st;
        gl::tests::HashRequestGeneratorFixture::foldPrefixIntoRequestGates(
            ptrs, prefixCount, prodRecIds, mainVid, st);
        const int32_t pHypoFound = st.hypoFound;
        const gl::NameId pHypoVid = st.hypoValidityId;
        const int32_t pScopes = st.nonExemptScopes;
        const gl::NameId pScopeVid = st.nonExemptValidityId;
        const int32_t pSecondary = st.secondaryCount;

        for (int last = 0; last < 24; ++last) {
            st.hypoFound = pHypoFound;
            st.hypoValidityId = pHypoVid;
            st.nonExemptScopes = pScopes;
            st.nonExemptValidityId = pScopeVid;
            st.secondaryCount = pSecondary;

            const bool folded = gl::tests::HashRequestGeneratorFixture::foldExprIntoRequestGates(
                lastPool[last], st, prodRecIds, mainVid);
            ptrs[prefixCount] = &lastPool[last];
            const gl::NameId count = static_cast<gl::NameId>(prefixCount + 1);
            const bool split = folded
                && gl::tests::HashRequestGeneratorFixture::requestGateStateAccepts(ea.requestGenerator, st, ptrs, count, m, mainVid);

            ASSERT_EQ(split, gl::tests::HashRequestGeneratorFixture::requestGatesPass(ea.requestGenerator, ptrs, count, m, mainVid));
        }
    }
}

// One named case per gate branch, so a regression names itself.
TEST(request_gates, named_gate_branches) {
    gl::tests::ProverFixture ea("Peano");
    ea.parameters.maxLenHypoKey = 2;
    ea.parameters.maxNumberSecondaryVariables = 2;
    ea.parameters.maxNumberSecondaryVariablesOrint = 2;

    gl::Memory m;
    ((((m).overallHashMemory)).maxKeyLength) = 3;
    const gl::NameId mainVid = gl::NameMap::MAIN_ID;
    const gl::NameId hypoA = ((m).nameMap).encodePush(mainVid, "hypo_a");
    const gl::NameId hypoB = ((m).nameMap).encodePush(mainVid, "hypo_b");

    const gl::NameId twoArgs[2] = { 501, 502 };
    const gl::NameId thirdArg[1] = { 503 };
    const gl::NameId noArgs[1] = { 0 };

    bool verdict = false;

    // Two distinct secondaries: exactly at the cap, accepted.
    const gl::IntEncodedExpr plain = gateExpr(mainVid, 0, 0, 2, twoArgs);
    const gl::IntEncodedExpr* atCap[1] = { &plain };
    ASSERT_TRUE(gatesMatchOnSplit(ea, m, atCap, 1, mainVid, verdict));
    ASSERT_TRUE(verdict);

    // A third distinct secondary crosses the cap, and no widening applies.
    const gl::IntEncodedExpr third = gateExpr(mainVid, 0, 0, 1, thirdArg);
    const gl::IntEncodedExpr* overCap[2] = { &plain, &third };
    ASSERT_TRUE(gatesMatchOnSplit(ea, m, overCap, 2, mainVid, verdict));
    ASSERT_FALSE(verdict);

    // The SAME secondary in both premises still counts once — the dedup must
    // survive the fold split.
    const gl::IntEncodedExpr* repeated[2] = { &plain, &plain };
    ASSERT_TRUE(gatesMatchOnSplit(ea, m, repeated, 2, mainVid, verdict));
    ASSERT_TRUE(verdict);

    // Two hypotheses at one scope: accepted at the hypothesis-key length cap.
    const gl::IntEncodedExpr hypo1 = gateExpr(hypoA, 1, 0, 0, noArgs);
    const gl::IntEncodedExpr* hypoPair[2] = { &hypo1, &hypo1 };
    ASSERT_TRUE(gatesMatchOnSplit(ea, m, hypoPair, 2, mainVid, verdict));
    ASSERT_TRUE(verdict);

    // Two hypotheses at different scopes: refused by the fold itself.
    const gl::IntEncodedExpr hypo2 = gateExpr(hypoB, 1, 0, 0, noArgs);
    const gl::IntEncodedExpr* clash[2] = { &hypo1, &hypo2 };
    ASSERT_TRUE(gatesMatchOnSplit(ea, m, clash, 2, mainVid, verdict));
    ASSERT_FALSE(verdict);

    // A non-hypothesis premise at a third scope alongside a hypothesis:
    // refused by the scope agreement, whichever side the hypothesis is on.
    const gl::IntEncodedExpr otherScope = gateExpr(hypoB, 0, 0, 0, noArgs);
    const gl::IntEncodedExpr* strayScope[2] = { &hypo1, &otherScope };
    ASSERT_TRUE(gatesMatchOnSplit(ea, m, strayScope, 2, mainVid, verdict));
    ASSERT_FALSE(verdict);
    const gl::IntEncodedExpr* strayScopeFirst[2] = { &otherScope, &hypo1 };
    ASSERT_TRUE(gatesMatchOnSplit(ea, m, strayScopeFirst, 2, mainVid, verdict));
    ASSERT_FALSE(verdict);

    // A main-scope anchor rides along with a hypothesis scope.
    const gl::IntEncodedExpr anchor = gateExpr(mainVid, 0, 1, 0, noArgs);
    const gl::IntEncodedExpr* withAnchor[2] = { &hypo1, &anchor };
    ASSERT_TRUE(gatesMatchOnSplit(ea, m, withAnchor, 2, mainVid, verdict));
    ASSERT_TRUE(verdict);

    // A main-scope NON-anchor does not — the exemption is anchors only.
    const gl::IntEncodedExpr mainPlain = gateExpr(mainVid, 0, 0, 0, noArgs);
    const gl::IntEncodedExpr* withPlain[2] = { &hypo1, &mainPlain };
    ASSERT_TRUE(gatesMatchOnSplit(ea, m, withPlain, 2, mainVid, verdict));
    ASSERT_FALSE(verdict);

    // Three hypothesis premises exceed maxLenHypoKey.
    const gl::IntEncodedExpr* hypoTriple[3] = { &hypo1, &hypo1, &hypo1 };
    ASSERT_TRUE(gatesMatchOnSplit(ea, m, hypoTriple, 3, mainVid, verdict));
    ASSERT_FALSE(verdict);

    // The key-length cap: maxKeyLength is 3, so four premises are refused.
    const gl::IntEncodedExpr* four[4] = { &anchor, &anchor, &anchor, &anchor };
    ASSERT_TRUE(gatesMatchOnSplit(ea, m, four, 4, mainVid, verdict));
    ASSERT_FALSE(verdict);
    const gl::IntEncodedExpr* three[3] = { &anchor, &anchor, &anchor };
    ASSERT_TRUE(gatesMatchOnSplit(ea, m, three, 3, mainVid, verdict));
    ASSERT_TRUE(verdict);
}

// The scoped widening: a wider secondary budget, but only when every premise
// sits at one shared scope AND that scope is an _orint_ branch.
TEST(request_gates, orint_widening_branch) {
    gl::tests::ProverFixture ea("Peano");
    ea.parameters.maxLenHypoKey = 4;
    ea.parameters.maxNumberSecondaryVariables = 2;
    ea.parameters.maxNumberSecondaryVariablesOrint = 4;

    gl::Memory m;
    ((((m).overallHashMemory)).maxKeyLength) = 4;
    const gl::NameId mainVid = gl::NameMap::MAIN_ID;
    const gl::NameId orintVid = ((m).nameMap).encodePush(mainVid, "_orint_1");
    const gl::NameId plainVid = ((m).nameMap).encodePush(mainVid, "branch_1");

    const gl::NameId argsA[2] = { 601, 602 };
    const gl::NameId argsB[2] = { 603, 604 };

    bool verdict = false;

    // Four distinct secondaries, both premises in one _orint_ scope: over the
    // standard cap, under the widened one, accepted.
    const gl::IntEncodedExpr orintA = gateExpr(orintVid, 0, 0, 2, argsA);
    const gl::IntEncodedExpr orintB = gateExpr(orintVid, 0, 0, 2, argsB);
    const gl::IntEncodedExpr* shared[2] = { &orintA, &orintB };
    ASSERT_TRUE(gatesMatchOnSplit(ea, m, shared, 2, mainVid, verdict));
    ASSERT_TRUE(verdict);

    // Same four secondaries, but the premises straddle two scopes: refused.
    const gl::IntEncodedExpr mainB = gateExpr(mainVid, 0, 0, 2, argsB);
    const gl::IntEncodedExpr* mixed[2] = { &orintA, &mainB };
    ASSERT_TRUE(gatesMatchOnSplit(ea, m, mixed, 2, mainVid, verdict));
    ASSERT_FALSE(verdict);

    // One shared scope, but not an _orint_ branch: refused.
    const gl::IntEncodedExpr plainA = gateExpr(plainVid, 0, 0, 2, argsA);
    const gl::IntEncodedExpr plainB = gateExpr(plainVid, 0, 0, 2, argsB);
    const gl::IntEncodedExpr* notOrint[2] = { &plainA, &plainB };
    ASSERT_TRUE(gatesMatchOnSplit(ea, m, notOrint, 2, mainVid, verdict));
    ASSERT_FALSE(verdict);
}

TEST(request_gates, component_counters_preserve_reset_and_thread_boundaries) {
    gl::tests::ProverFixture analyzer(std::string("Peano"));
    gl::LogicBlockIdentity identity(analyzer.parameters, analyzer.anchorInfo);
    ASSERT_FALSE(identity.isAnchorLb(analyzer.body));
    ASSERT_FALSE(identity.relaysAbsorbedHistory(analyzer.body));
    gl::HashRequestGenerator generator(analyzer.parameters);
    gl::HashRuleEvaluator evaluator(analyzer.parameters, analyzer.definitions,
        analyzer.admissionPolicy, identity);
    using Generator = gl::HashRequestGenerator;
    using Fixture = gl::tests::HashRequestGeneratorFixture;
    Fixture::setGrowthMatchCount(17);
    ASSERT_EQ(Generator::growthMatchCount(), 17);
    Generator::resetGrowthMatchCount();
    ASSERT_EQ(Generator::growthMatchCount(), 0);
    Fixture::seedDepth(2, 9);
    ASSERT_EQ(Generator::gpuGrowAttemptsByDepth(2), 9u);
    ASSERT_EQ(Generator::gpuGrowFrontierByDepth(2), 9u);
    ASSERT_EQ(Generator::gpuGrowSubkeysByDepth(2), 9u);
    ASSERT_EQ(Generator::gpuGrowRequestsByDepth(2), 9u);
    ASSERT_EQ(Generator::gpuProducerAttemptsByDepth(2), 9u);
    ASSERT_EQ(Generator::gpuProducerSurvivorsByDepth(2), 9u);
    Generator::resetDepthCounters(2);
    ASSERT_EQ(Generator::gpuGrowAttemptsByDepth(2), 0u);
    ASSERT_EQ(Generator::gpuGrowFrontierByDepth(2), 0u);
    ASSERT_EQ(Generator::gpuGrowSubkeysByDepth(2), 0u);
    ASSERT_EQ(Generator::gpuGrowRequestsByDepth(2), 0u);
    ASSERT_EQ(Generator::gpuProducerAttemptsByDepth(2), 0u);
    ASSERT_EQ(Generator::gpuProducerSurvivorsByDepth(2), 0u);
    Fixture::seedFilters(generator, 11);
    ASSERT_EQ(generator.gpuRequestFilterCallsValue(), 11u);
    ASSERT_EQ(generator.gpuProducerFilterCallsValue(), 11u);
    ASSERT_EQ(generator.gpuFilterInputStatementsValue(), 11u);
    ASSERT_EQ(generator.gpuFilterOutputStatementsValue(), 11u);
    ASSERT_EQ(generator.gpuFilterMaximumInputStatementsValue(), 11u);
    ASSERT_EQ(generator.gpuFilterMaximumOutputStatementsValue(), 11u);
    generator.resetFilterCounters();
    ASSERT_EQ(generator.gpuRequestFilterCallsValue(), 0u);
    ASSERT_EQ(generator.gpuProducerFilterCallsValue(), 0u);
    ASSERT_EQ(generator.gpuFilterInputStatementsValue(), 0u);
    ASSERT_EQ(generator.gpuFilterOutputStatementsValue(), 0u);
    ASSERT_EQ(generator.gpuFilterMaximumInputStatementsValue(), 0u);
    ASSERT_EQ(generator.gpuFilterMaximumOutputStatementsValue(), 0u);
    gl::HashRuleEvaluator::resetEvaluationUsage();
    gl::HashRuleEvaluator::noteRequest();
    gl::HashRuleEvaluator::noteDependencyPassRequest();
    ASSERT_EQ(gl::HashRuleEvaluator::evaluationUsage().requests, 1u);
    ASSERT_EQ(gl::HashRuleEvaluator::evaluationUsage().dependencyPassRequests, 1u);
    Fixture::setGrowthMatchCount(23);
    bool workerFresh = false;
    std::thread worker([&]() {
        workerFresh = Generator::growthMatchCount() == 0
            && gl::HashRuleEvaluator::evaluationUsage().requests == 0;
        Fixture::setGrowthMatchCount(41);
        gl::HashRuleEvaluator::noteRequest();
    });
    worker.join();
    ASSERT_TRUE(workerFresh);
    ASSERT_EQ(Generator::growthMatchCount(), 23);
    ASSERT_EQ(gl::HashRuleEvaluator::evaluationUsage().requests, 1u);
    Generator::resetGrowthMatchCount();
    gl::HashRuleEvaluator::resetEvaluationUsage();
    ASSERT_EQ(gl::HashRuleEvaluator::evaluationUsage().requests, 0u);
    ASSERT_EQ(gl::HashRuleEvaluator::evaluationUsage().dependencyPassRequests, 0u);
}
