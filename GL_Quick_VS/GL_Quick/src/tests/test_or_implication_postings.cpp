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
#include "../logical_block/statement_arrivals.hpp"
#include "../memory.hpp"
#include <cstddef>
#include <cstring>
#include <utility>
#include "test_harness.hpp"
#include "prover_fixture.hpp"
#include "or_implication_processor_fixture.hpp"

#include "canonical_test_helpers.hpp"
#include "../memory_infra/or_implication_postings.hpp"
#include "../memory_infra/or_implications.hpp"
#include "../memory_infra/or_implication_match_index.hpp"
#include "../prover.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <vector>

namespace {
    int32_t stageReleasedSourceCohort(gl::tests::ProverFixture& analyzer,
        const std::string& source, const std::vector<std::string>& leaves,
        const std::string& scope = "main") {
        auto& memory = analyzer.body;
        ((memory).isActive) = true;
        ((memory).primedForContradiction) = true;
        ((memory).level) = 0;
        const int32_t cohort = gl::mintOrCohortId(((memory).lbStateInterner),
            ((memory).lbStateInterner).encode(gl::StrSpan(scope)),
            ((memory).lbStateInterner).encode(gl::StrSpan(source)));
        ((memory).orDisjunctCount).insert(cohort, static_cast<int>(leaves.size()));
        for (const auto& leaf : leaves) {
            const std::string wrapped = "(" + leaf + ")";
            ((memory).orPendingBranches).insertSorted(cohort,
                ((memory).lbStateInterner).encode(gl::StrSpan(wrapped)),
                gl::DecodedIdLess{&((memory).lbStateInterner)});
        }
        ((memory).orPendingLevels).insertSorted(cohort, 0,
            [](int32_t a, int32_t b) { return a < b; });
        ((memory).pendingOrReleases).mint(cohort);
        return cohort;
    }

    bool releasedSourceMainFact(gl::tests::ProverFixture& analyzer,
        const std::string& text) {
        auto& memory = analyzer.body;
        const gl::NameId original = ((memory).nameMap).lookup(text);
        if (original != 0 && ((memory).intKnownStatements).find(
                gl::StatementKey{original, gl::NameMap::MAIN_ID}) != nullptr) return true;
        for (const gl::ColdMail* mail : {&((memory).sameIterationInternalMail),
                                      &((memory).nextIterationInternalMail)}) {
            for (const auto& row : gl::decodeInternalMailStatements(*mail, ((memory).nameMap)))
                if (row.first.original == text && row.first.validityName == "main") return true;
        }
        return false;
    }
}

namespace {
    struct PostingRig {
        gl::GlobalMemoryManager manager;
        struct Init {
            Init(gl::GlobalMemoryManager& m) {
                m.init(gl::StaticMemoryConfig{8 << 20, 1 << 18});
            }
        } init{manager};
        gl::LbArena arena{&manager};
        gl::OrImplicationPostings index{&arena};
    };

    struct SourceRig {
        gl::GlobalMemoryManager manager;
        struct Init {
            Init(gl::GlobalMemoryManager& m) {
                m.init(gl::StaticMemoryConfig{8 << 20, 1 << 18});
            }
        } init{manager};
        gl::DirtyState dirty = gl::DirtyState::Clean;
        gl::LbArena arena{&manager};
        gl::ScratchArena scratch{&manager};
        gl::OrImplications store{&arena, &dirty};
        gl::OrImplicationMatchIndex matching{&arena};
    };

    void publishTestCache(SourceRig& rig, int64_t key, gl::OrCanonicalCache value,
        std::initializer_list<gl::NameId> tokens = {}) {
        rig.matching.publishCache(key, value, [&](auto&& append) {
            for (gl::NameId token : tokens) append(token);
        });
    }

    std::set<int64_t> postingValues(const gl::OrImplicationMatchIndex& index,
        gl::OrPostingKind kind, int64_t key) {
        std::set<int64_t> result;
        index.postings().forEach(gl::orPostingKey(kind, key),
            [&](int64_t value) { result.insert(value); });
        return result;
    }

    std::vector<int64_t> values(const gl::OrImplicationPostings& index,
                                int64_t key) {
        std::vector<int64_t> result;
        index.forEach(key, [&](int64_t value) { result.push_back(value); });
        std::sort(result.begin(), result.end());
        return result;
    }

    void putOrClasses(gl::Memory& mb, gl::NameId scope,
        const std::vector<std::set<std::string>>& classes, bool initialize = true) {
        std::vector<char> bytes;
        std::vector<int32_t> lengths;
        for (const auto& members : classes) {
            gl::EquivalenceClass cls;
            cls.setMembersFromNames(members, ((mb).nameMap));
            const auto blob = gl::serializeEquivalenceClass(cls);
            bytes.insert(bytes.end(), blob.begin(), blob.end());
            lengths.push_back(static_cast<int32_t>(blob.size()));
        }
        ((mb).equivalenceClassesMap).inner().assignRun(scope, bytes.data(),
            lengths.data(), static_cast<int32_t>(lengths.size()));
        if(initialize) seedFixtureCanonicalClasses(mb);
    }

    std::string orViewText(gl::Memory& mb, const gl::OrImplicationMatchIndex& index,
        gl::OrViewIdentity identity) {
        const int32_t* id = index.viewIds.find(identity);
        assert(id != nullptr && *id >= 0);
        const auto* cache = index.caches().find(index.views[*id].cacheKey);
        assert(cache != nullptr && cache->templateId != 0);
        return ((mb).nameMap).decode(index.views[*id].canonicalId);
    }

    const gl::OrCanonicalCache& orViewCache(const gl::OrImplicationMatchIndex& index,
        gl::OrViewIdentity identity) {
        const int32_t* id = index.viewIds.find(identity);
        assert(id != nullptr && *id >= 0);
        const auto* cache = index.caches().find(index.views[*id].cacheKey);
        assert(cache != nullptr && cache->templateId != 0);
        return *cache;
    }

    // Deposit through the canonical door at main with the given mail status
    // (0 local, 3 mailed), level 0.
    void depositAtMain(gl::tests::ProverFixture& analyzer, const char* expr, int status) {
        const int lv[1] = { 0 };
        const gl::TransientOrigin origin{ true, gl::OriginTag::taskFormulation, nullptr, 0 };
        analyzer.statementMemory.addExprToMemoryBlock(
            gl::StrSpan(expr, static_cast<int32_t>(std::strlen(expr))),
            analyzer.body, 0, status, lv, 1, origin, -1, -1,
            gl::StrSpan("main", 4), /*doNotDisintegrate=*/true,
            /*allowOrDisintegration=*/false, nullptr);
    }
}

namespace {
    std::string symmetricSourceOperator(gl::tests::ProverFixture& analyzer) {
        return analyzer.orRuleCompiler.findOrMintOrOperator(
            {"(in2[u_1,u_2,u_3])", "(=[u_1,u_2])", "(in2[u_2,u_1,u_3])"}, 3);
    }

    int64_t putFiringFact(gl::tests::ProverFixture& analyzer, const std::string& text,
        const std::string& scope = "main", bool source = false, bool local = true,
        int level = 0) {
        gl::DirtyState dirty = gl::DirtyState::Clean;
        gl::PagedVector<gl::IntEncodedExpr> rows(&gl::genScratchArenas().forSlot(0), &dirty);
        const gl::TransientOrigin origin{true, gl::OriginTag::taskFormulation, nullptr, 0};
        analyzer.statementMemory.addStatement(gl::StrSpan(text), analyzer.body, local, &level, 1,
            origin, gl::StrSpan(scope), rows, source);
        return gl::packInt32Pair(((analyzer.body).nameMap).lookup(text),
            ((analyzer.body).nameMap).lookup(scope));
    }
    std::vector<gl::ExpressionWithValidity> firingHeads(gl::tests::ProverFixture& analyzer,
        const gl::ColdMail& mail) {
        const auto decoded = gl::makeHeapMail(mail, ((analyzer.body).nameMap),
            ((analyzer.body).originInterner));
        std::vector<gl::ExpressionWithValidity> heads;
        for (const auto& item : decoded.statements) heads.push_back(std::get<0>(item));
        return heads;
    }
}

TEST(or_implication_firing, binary_both_arrival_orders_retire_without_hash_rules) {
    for (bool factFirst : {false, true}) {
        gl::tests::ProverFixture analyzer("Peano");
        analyzer.parameters.trackHistory = true;
        const auto name = analyzer.orRuleCompiler.findOrMintOrOperator(
            {"(in[u_1,u_2])", "(in[u_2,u_1])"}, 2);
        const std::string source = "(" + name + "[a,N])";
        if (factFirst) putFiringFact(analyzer, "!(in[a,N])");
        const auto key = putFiringFact(analyzer, source, "main", true);
        analyzer.orImplicationProcessor.drainOrImplications(analyzer.body, ((analyzer.body).sameIterationInternalMail));
        if (!factFirst) {
            ASSERT_TRUE(((analyzer.body).sameIterationInternalMail).statementsEmpty());
            putFiringFact(analyzer, "!(in[a,N])");
            analyzer.orImplicationProcessor.drainOrImplications(analyzer.body, ((analyzer.body).sameIterationInternalMail));
        }
        const auto heads = firingHeads(analyzer, ((analyzer.body).sameIterationInternalMail));
        ASSERT_EQ(heads.size(), size_t{1});
        ASSERT_EQ(heads[0].original, std::string("(in[N,a])"));
        ASSERT_EQ(heads[0].validityName, std::string("main"));
        ASSERT_TRUE(((analyzer.body).orImplications).find(key) == nullptr);
        ASSERT_EQ(((((analyzer.body).overallHashMemory)).originals).count(), 0);
        ASSERT_TRUE(!((analyzer.body).sameIterationInternalMail).exprOriginMapEmpty());
    }
}

TEST(or_implication_firing, residual_continues_with_disintegration_suppressed) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = true;
    const auto name = analyzer.orRuleCompiler.findOrMintOrOperator(
        {"(in[u_1,u_4])", "(in[u_2,u_4])", "(in[u_3,u_4])"}, 4);
    analyzer.orRuleCompiler.preMintReducedOrs();
    const auto key = putFiringFact(analyzer, "(" + name + "[a,b,c,N])", "main", true);
    putFiringFact(analyzer, "!(in[a,N])");
    auto& mb = analyzer.body;
    analyzer.orImplicationProcessor.drainOrImplications(mb, ((mb).sameIterationInternalMail));
    const auto heads = firingHeads(analyzer, ((mb).sameIterationInternalMail));
    ASSERT_EQ(heads.size(), size_t{1});
    ASSERT_TRUE(((mb).orImplications).find(key) == nullptr);
    bool dnd, aod, rh, continuation;
    int32_t iteration;
    const auto mailKey = gl::packInt32Pair(((mb).nameMap).lookup(heads[0].original), gl::NameMap::MAIN_ID);
    ((mb).sameIterationInternalMail).getDisintegrationSignal(mailKey, dnd, aod, iteration, &rh, &continuation);
    ASSERT_TRUE(continuation && !aod);
    ((mb).sameIterationInternalMail).setDisintegrationSignal(mailKey, true, false, iteration);
    ((mb).intToBeProved).assignSet(gl::packInt32Pair(((mb).nameMap).encode("(in[unrelated,N])"),
        gl::NameMap::MAIN_ID), nullptr, 0);
    analyzer.logicBlockOperations.standardProcessing(mb, nullptr, ((mb).sameIterationInternalMail),
        ((mb).nextIterationInternalMail), -1);
    ASSERT_TRUE(((mb).orImplications).find(mailKey) != nullptr);
    ASSERT_TRUE(!analyzer.orImplicationProcessor.activateOrImplicationContinuation(mb,
        gl::StrSpan("(or_missing[a])", 15), gl::NameMap::MAIN_ID));
    ((mb).sameIterationInternalMail).clear();
    putFiringFact(analyzer, "!(in[b,N])");
    analyzer.orImplicationProcessor.drainOrImplications(mb, ((mb).sameIterationInternalMail));
    const auto final = firingHeads(analyzer, ((mb).sameIterationInternalMail));
    ASSERT_EQ(final.size(), size_t{1});
    ASSERT_EQ(final[0].original, std::string("(in[c,N])"));
    ASSERT_EQ(((((mb).overallHashMemory)).originals).count(), 0);
}

TEST(or_implication_firing, sibling_scopes_stay_separate_and_parent_survives) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = false;
    auto& mb = analyzer.body;
    const auto left = ((mb).nameMap).encodePush(gl::NameMap::MAIN_ID, "left");
    const auto right = ((mb).nameMap).encodePush(gl::NameMap::MAIN_ID, "right");
    const auto name = analyzer.orRuleCompiler.findOrMintOrOperator(
        {"(in[u_1,u_4])", "(in[u_2,u_4])", "(in[u_3,u_4])"}, 4);
    analyzer.orRuleCompiler.preMintReducedOrs();
    const auto key = putFiringFact(analyzer, "(" + name + "[a,b,c,N])", "main", true);
    putFiringFact(analyzer, "!(in[a,N])", ((mb).nameMap).decode(left));
    putFiringFact(analyzer, "!(in[b,N])", ((mb).nameMap).decode(right));
    analyzer.orImplicationProcessor.drainOrImplications(mb, ((mb).sameIterationInternalMail));
    const auto heads = firingHeads(analyzer, ((mb).sameIterationInternalMail));
    ASSERT_EQ(heads.size(), size_t{2});
    ASSERT_TRUE(((mb).orImplications).find(key) != nullptr);
    for (const auto& head : heads) {
        ASSERT_TRUE(head.original != "(in[c,N])");
        ASSERT_TRUE(head.validityName != "main");
    }
    ((mb).sameIterationInternalMail).clear();
    putFiringFact(analyzer, "!(in[a,N])");
    putFiringFact(analyzer, "!(in[b,N])");
    analyzer.orImplicationProcessor.drainOrImplications(mb, ((mb).sameIterationInternalMail));
    ASSERT_TRUE(((mb).orImplications).find(key) == nullptr);
    const auto last = firingHeads(analyzer, ((mb).sameIterationInternalMail));
    ASSERT_EQ(last.size(), size_t{1});
    ASSERT_EQ(last[0].original, std::string("(in[c,N])"));
    ASSERT_EQ(last[0].validityName, std::string("main"));
}

TEST(or_implication_firing, external_source_needs_local_fact_and_all_refuted_yields_opposite) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = false;
    const auto name = analyzer.orRuleCompiler.findOrMintOrOperator(
        {"(in[u_1,u_2])", "(in[u_2,u_1])"}, 2);
    const auto key = putFiringFact(analyzer, "(" + name + "[a,N])", "main", true, false);
    putFiringFact(analyzer, "!(in[a,N])", "main", false, false);
    auto& mb = analyzer.body;
    analyzer.orImplicationProcessor.drainOrImplications(mb, ((mb).sameIterationInternalMail));
    ASSERT_TRUE(((mb).sameIterationInternalMail).statementsEmpty());
    ASSERT_TRUE(((mb).orImplications).find(key) != nullptr);
    putFiringFact(analyzer, "!(in[N,a])");
    analyzer.orImplicationProcessor.drainOrImplications(mb, ((mb).sameIterationInternalMail));
    const auto heads = firingHeads(analyzer, ((mb).sameIterationInternalMail));
    ASSERT_EQ(heads.size(), size_t{1});
    ASSERT_EQ(heads[0].original, std::string("(in[a,N])"));
    ASSERT_TRUE(((mb).orImplications).find(key) == nullptr);
}

TEST(or_implication_firing, equality_only_arrival_rewrites_whole_source_and_records_bridge) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = true;
    const auto name = analyzer.orRuleCompiler.findOrMintOrOperator(
        {"(in[u_1,u_2])", "(in[u_2,u_1])"}, 2);
    putFiringFact(analyzer, "(" + name + "[b,N])", "main", true);
    putFiringFact(analyzer, "!(in[a,N])");
    auto& mb = analyzer.body;
    analyzer.orImplicationProcessor.drainOrImplications(mb, ((mb).sameIterationInternalMail));
    ASSERT_TRUE(((mb).sameIterationInternalMail).statementsEmpty());
    putFiringFact(analyzer, "(=[a,b])");
    analyzer.orImplicationProcessor.drainOrImplications(mb, ((mb).sameIterationInternalMail));
    const auto heads = firingHeads(analyzer, ((mb).sameIterationInternalMail));
    ASSERT_EQ(heads.size(), size_t{1});
    ASSERT_EQ(heads[0].original, std::string("(in[N,a])"));
    const auto history = gl::decodeOriginMapSorted(((mb).exprOriginMap), ((mb).originInterner));
    bool bridge = false;
    for (const auto& entry : history)
        if (entry.first.first == "(" + name + "[a,N])")
            for (const auto& row : entry.second)
                if (row.first == "equality1") bridge = true;
    ASSERT_TRUE(bridge);
    ASSERT_EQ(((((mb).overallHashMemory)).originals).count(), 0);
}

TEST(or_implication_firing, canonical_twin_rechecks_previously_nonlocal_source) {
  for (bool descendant : {false, true}) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = false;
    auto& mb = analyzer.body;
    const auto first = analyzer.orRuleCompiler.findOrMintOrOperator(
        {"(in[u_1,u_3])", "(in[u_2,u_3])"}, 3);
    const auto second = analyzer.orRuleCompiler.findOrMintOrOperator(
        {"(in[u_1,u_3])", "(in[u_3,u_2])"}, 3);
    ASSERT_TRUE(first != second);
    const std::string earlier = "(" + std::min(first, second) + "[a,c,N])";
    const std::string later = "(" + std::max(first, second) + "[a,c,N])";
    const auto earlierKey = putFiringFact(analyzer, earlier, "main", true, false);
    const auto laterKey = putFiringFact(analyzer, later, "main", true, true);
    const auto scope = descendant
        ? ((mb).nameMap).encodePush(gl::NameMap::MAIN_ID, "child") : gl::NameMap::MAIN_ID;
    const std::string validity = ((mb).nameMap).decode(scope);
    putFiringFact(analyzer, "!(in[b,N])", validity, false, false);
    putFiringFact(analyzer, "(=[a,b])", validity, false, false);
    analyzer.orImplicationProcessor.drainOrImplications(mb, ((mb).sameIterationInternalMail));
    const auto heads = firingHeads(analyzer, ((mb).sameIterationInternalMail));
    const auto* canonical = ((mb).intKnownStatements).find(
        gl::StatementKey{((mb).nameMap).lookup("!(in[a,N])"), scope});
    ASSERT_TRUE(canonical != nullptr && canonical->local);
    ASSERT_EQ(heads.size(), size_t{2});
    ASSERT_TRUE(std::find(heads.begin(), heads.end(),
        gl::ExpressionWithValidity{"(in[c,N])", validity}) != heads.end());
    ASSERT_TRUE(std::find(heads.begin(), heads.end(),
        gl::ExpressionWithValidity{"(in[N,c])", validity}) != heads.end());
    ASSERT_EQ(((mb).orImplications).find(earlierKey) != nullptr, descendant);
    ASSERT_EQ(((mb).orImplications).find(laterKey) != nullptr, descendant);
    ASSERT_EQ(((((mb).overallHashMemory)).originals).count(), 0);
    ((mb).sameIterationInternalMail).clear();
    analyzer.orImplicationProcessor.drainOrImplications(mb, ((mb).sameIterationInternalMail));
    ASSERT_TRUE(((mb).sameIterationInternalMail).statementsEmpty());
  }
}

TEST(or_implication_firing, equality_refuters_use_the_same_normal_form_as_or_leaves) {
    for (bool negativeLeaf : {false, true}) {
        gl::tests::ProverFixture analyzer("Peano");
        analyzer.parameters.trackHistory = true;
        auto& mb = analyzer.body;
        const std::string leaf = negativeLeaf ? "!(=[u_1,u_2])" : "(=[u_1,u_2])";
        const auto name = analyzer.orRuleCompiler.findOrMintOrOperator(
            {leaf, "(in[u_3,u_4])"}, 4);
        putFiringFact(analyzer, "(" + name + "[a,c,d,N])", "main", true);
        putFiringFact(analyzer, negativeLeaf ? "(=[b,c])" : "!(=[b,c])");
        analyzer.orImplicationProcessor.drainOrImplications(mb, ((mb).sameIterationInternalMail));
        ASSERT_TRUE(((mb).sameIterationInternalMail).statementsEmpty());
        putFiringFact(analyzer, "(=[a,b])");
        analyzer.orImplicationProcessor.drainOrImplications(mb, ((mb).sameIterationInternalMail));
        const auto heads = firingHeads(analyzer, ((mb).sameIterationInternalMail));
        ASSERT_EQ(heads.size(), size_t{1});
        ASSERT_EQ(heads[0].original, std::string("(in[d,N])"));
        ASSERT_EQ(heads[0].validityName, std::string("main"));
        ASSERT_EQ(((((mb).overallHashMemory)).originals).count(), 0);
        const auto history = gl::decodeOriginMapSorted(((mb).exprOriginMap), ((mb).originInterner));
        bool refuterBridge = false;
        for (const auto& entry : history) {
            if (entry.first.first != (negativeLeaf ? "(=[a,a])" : "!(=[a,c])")) continue;
            for (const auto& row : entry.second)
                if (row.first == "equality1") refuterBridge = true;
        }
        ASSERT_TRUE(refuterBridge);
    }
}

TEST(or_implication_matching, shared_chooser_initial_classes_and_signed_views) {
    gl::tests::ProverFixture ana("Peano");
    ana.parameters.trackHistory = false;
    auto& mb = ana.body;
    SourceRig rig;
    auto& index = rig.matching;
    const auto main = gl::NameMap::MAIN_ID;
    putOrClasses(mb, main, {{"a", "b"}});
    const auto raw = ((mb).nameMap).encode("!(p[b])");
    const auto fact = ((mb).nameMap).encode("!(p[a])");
    const auto positive = ((mb).nameMap).encode("(p[b])");
    const auto source = ((mb).nameMap).encode("(orFixture[b])");
    const gl::OrViewIdentity watch = gl::OrViewIdentity::sourceLeaf(gl::packInt32Pair(source, main), main, 0);
    const gl::OrViewIdentity factView = gl::OrViewIdentity::knownFact(gl::packInt32Pair(fact, main), main);
    const gl::OrViewIdentity positiveView = gl::OrViewIdentity::knownFact(gl::packInt32Pair(positive, main), main);
    const int32_t before = ((mb).intKnownStatements).count();
    const int32_t watcher = gl::tests::OrImplicationProcessorFixture::bindOrCanonicalView(ana.orImplicationProcessor, mb, index, watch, raw);
    const int32_t known = gl::tests::OrImplicationProcessorFixture::bindOrCanonicalView(ana.orImplicationProcessor, mb, index, factView, fact);
    gl::tests::OrImplicationProcessorFixture::bindOrCanonicalView(ana.orImplicationProcessor, mb, index, positiveView, positive);
    const gl::OrViewIdentity duplicate = gl::OrViewIdentity::sourceLeaf(watch.owner, main, 1);
    gl::tests::OrImplicationProcessorFixture::bindOrCanonicalView(ana.orImplicationProcessor, mb, index, duplicate, raw);
    // !(p[b]) and !(p[a]) share the entry of !(p[a]); (p[b]) has its own.
    ASSERT_EQ(index.caches().count(), 2);
    ASSERT_EQ(index.liveViews, 4);
    ASSERT_EQ(gl::tests::OrImplicationProcessorFixture::refreshOrCanonicalViews(ana.orImplicationProcessor, mb, index), 0);
    ASSERT_EQ(orViewText(mb, index, watch), std::string("!(p[a])"));
    ASSERT_EQ(orViewText(mb, index, factView), std::string("!(p[a])"));
    ASSERT_EQ(orViewText(mb, index, positiveView), std::string("(p[a])"));
    const auto* cache = index.caches().find(index.views[watcher].cacheKey);
    const auto* knownCache = index.caches().find(index.views[known].cacheKey);
    ASSERT_EQ(cache->templateId, knownCache->templateId);
    ASSERT_EQ(((mb).intKnownStatements).count(), before);
    ASSERT_TRUE(index.classScopesBuilt);
}

TEST(or_implication_matching, equality_rekeys_shared_cache_once_and_respects_tokens) {
    gl::tests::ProverFixture ana("Peano");
    ana.parameters.trackHistory = false;
    auto& mb = ana.body;
    SourceRig rig;
    auto& index = rig.matching;
    const auto main = gl::NameMap::MAIN_ID;
    const auto raw = ((mb).nameMap).encode("!(p[b])");
    const auto other = ((mb).nameMap).encode("!(p[bb])");
    const auto predicateOnly = ((mb).nameMap).encode("!(b[z])");
    const gl::OrViewIdentity watch = gl::OrViewIdentity::sourceLeaf(gl::packInt32Pair(raw, main), main, 0);
    gl::tests::OrImplicationProcessorFixture::bindOrCanonicalView(ana.orImplicationProcessor, mb, index, watch, raw);
    gl::tests::OrImplicationProcessorFixture::bindOrCanonicalView(ana.orImplicationProcessor, mb, index, {watch.owner, main, 1}, raw);
    gl::tests::OrImplicationProcessorFixture::bindOrCanonicalView(ana.orImplicationProcessor, mb, index, {gl::packInt32Pair(other, main), main, -1}, other);
    gl::tests::OrImplicationProcessorFixture::bindOrCanonicalView(ana.orImplicationProcessor, mb, index, {gl::packInt32Pair(predicateOnly, main), main, -1}, predicateOnly);
    putOrClasses(mb, main, {{"a", "b"}, {"x", "y"}});
    for (const char* token : {"a", "b", "b", "x", "y"})
        gl::tests::OrImplicationProcessorFixture::stageOrClassMember(ana.orImplicationProcessor, mb, index, ((mb).nameMap).encode(token), main);
    ASSERT_EQ(index.dirtyCaches.count(), 1);
    // A newly attached owner finds the entry of its current form; the drain
    // then moves the dirty entry's users into that same entry.
    gl::tests::OrImplicationProcessorFixture::bindOrCanonicalView(ana.orImplicationProcessor, mb, index, {watch.owner, main, 2}, raw);
    ASSERT_EQ(gl::tests::OrImplicationProcessorFixture::refreshOrCanonicalViews(ana.orImplicationProcessor, mb, index), 1);
    ASSERT_EQ(orViewText(mb, index, watch), std::string("!(p[a])"));
    ASSERT_EQ(orViewText(mb, index, {gl::packInt32Pair(other, main), main, -1}),
        std::string("!(p[bb])"));
    ASSERT_EQ(orViewText(mb, index, {gl::packInt32Pair(predicateOnly, main), main, -1}),
        std::string("!(b[z])"));
    ASSERT_EQ(gl::tests::OrImplicationProcessorFixture::refreshOrCanonicalViews(ana.orImplicationProcessor, mb, index), 0);
}

TEST(or_implication_matching, descendant_class_projects_both_sides_without_parent_rewrite) {
    gl::tests::ProverFixture ana("Peano");
    ana.parameters.trackHistory = false;
    auto& mb = ana.body;
    SourceRig rig;
    auto& index = rig.matching;
    const auto main = gl::NameMap::MAIN_ID;
    const auto child = ((mb).nameMap).encodePush(main, "or_match_child");
    const auto sibling = ((mb).nameMap).encodePush(main, "or_match_sibling");
    const auto raw = ((mb).nameMap).encode("!(p[b])");
    const auto fact = ((mb).nameMap).encode("!(p[a])");
    const gl::OrViewIdentity watch = gl::OrViewIdentity::sourceLeaf(gl::packInt32Pair(raw, main), main, 0);
    const gl::OrViewIdentity known = gl::OrViewIdentity::knownFact(gl::packInt32Pair(fact, main), main);
    gl::tests::OrImplicationProcessorFixture::bindOrCanonicalView(ana.orImplicationProcessor, mb, index, watch, raw);
    gl::tests::OrImplicationProcessorFixture::bindOrCanonicalView(ana.orImplicationProcessor, mb, index, known, fact);
    gl::tests::OrImplicationProcessorFixture::bindOrCanonicalView(ana.orImplicationProcessor, mb, index, {watch.owner, sibling, 0}, raw);
    putOrClasses(mb, child, {{"a", "b"}});
    gl::tests::OrImplicationProcessorFixture::stageOrClassMember(ana.orImplicationProcessor, mb, index, ((mb).nameMap).encode("a"), child);
    gl::tests::OrImplicationProcessorFixture::stageOrClassMember(ana.orImplicationProcessor, mb, index, ((mb).nameMap).encode("b"), child);
    ASSERT_EQ(gl::tests::OrImplicationProcessorFixture::refreshOrCanonicalViews(ana.orImplicationProcessor, mb, index), 0);
    ASSERT_EQ(orViewText(mb, index, watch), std::string("!(p[b])"));
    ASSERT_EQ(orViewText(mb, index, {watch.owner, child, 0}), std::string("!(p[a])"));
    ASSERT_EQ(orViewText(mb, index, {known.owner, child, -1}), std::string("!(p[a])"));
    ASSERT_EQ(orViewText(mb, index, {watch.owner, sibling, 0}), std::string("!(p[b])"));
    ASSERT_EQ(index.liveViews, 5);
}

TEST(or_implication_matching, old_descendant_classes_and_intermediate_dependency_wakeup) {
    gl::tests::ProverFixture ana("Peano");
    ana.parameters.trackHistory = false;
    auto& mb = ana.body;
    auto& index = ((mb).orImplicationMatches);
    const auto main = gl::NameMap::MAIN_ID;
    const auto child = ((mb).nameMap).encodePush(main, "or_match_child");
    const auto grandchild = ((mb).nameMap).encodePush(child, "or_match_grandchild");
    putOrClasses(mb, main, {{"c", "d"}});
    putOrClasses(mb, child, {{"b", "c"}});
    putOrClasses(mb, grandchild, {{"a", "b"}});
    const auto raw = ((mb).nameMap).encode("!(p[d])");
    const gl::OrViewIdentity watch = gl::OrViewIdentity::sourceLeaf(gl::packInt32Pair(raw, main), main, 0);
    gl::tests::OrImplicationProcessorFixture::bindOrCanonicalView(ana.orImplicationProcessor, mb, index, watch, raw);
    gl::tests::OrImplicationProcessorFixture::refreshOrCanonicalViews(ana.orImplicationProcessor, mb, index);
    ASSERT_EQ(orViewText(mb, index, watch), std::string("!(p[c])"));
    ASSERT_EQ(orViewText(mb, index, {watch.owner, child, 0}), std::string("!(p[b])"));
    ASSERT_EQ(orViewText(mb, index, {watch.owner, grandchild, 0}), std::string("!(p[a])"));
    // A consolidated bridge has no intermediate token; the maintained-view
    // refresh must invalidate descendants without creating application scopes.
    index.built = true;
    gl::EquivalenceClass extended;
    extended.setMembersFromNames({"0", "b", "c"}, ((mb).nameMap));
    mb.assignClassesById(child, {extended});
    ana.equivalenceProcessor.refreshCanonicalClasses(mb, child);
    std::vector<int64_t> applicationScopes;
    index.postings().forEach(gl::orPostingKey(gl::OrPostingKind::ClassScopes,
        ((mb).nameMap).lookup("d")), [&](int64_t scope) { applicationScopes.push_back(scope); });
    ASSERT_TRUE(applicationScopes == std::vector<int64_t>{main});
    for (const char* token : {"0", "b", "c"})
        gl::tests::OrImplicationProcessorFixture::stageOrClassMember(ana.orImplicationProcessor, mb, index, ((mb).nameMap).encode(token), child);
    ASSERT_EQ(gl::tests::OrImplicationProcessorFixture::refreshOrCanonicalViews(ana.orImplicationProcessor, mb, index), 2);
    ASSERT_EQ(orViewText(mb, index, watch), std::string("!(p[c])"));
    ASSERT_EQ(orViewText(mb, index, {watch.owner, grandchild, 0}), std::string("!(p[0])"));
}

TEST(or_implication_matching, class_change_without_moved_representative_keeps_cache_clean) {
    gl::tests::ProverFixture ana("Peano");
    ana.parameters.trackHistory = false;
    auto& mb = ana.body;
    auto& index = ((mb).orImplicationMatches);
    const auto main = gl::NameMap::MAIN_ID;
    putOrClasses(mb, main, {{"a", "b"}});
    const auto raw = ((mb).nameMap).encode("!(p[b])");
    const gl::OrViewIdentity watch = gl::OrViewIdentity::sourceLeaf(gl::packInt32Pair(raw, main), main, 0);
    gl::tests::OrImplicationProcessorFixture::bindOrCanonicalView(ana.orImplicationProcessor, mb, index, watch, raw);
    ASSERT_EQ(gl::tests::OrImplicationProcessorFixture::refreshOrCanonicalViews(ana.orImplicationProcessor, mb, index), 0);
    ASSERT_EQ(orViewText(mb, index, watch), std::string("!(p[a])"));
    index.built = true;
    // A larger member joins: the class changes, the representative a stays.
    gl::EquivalenceClass joined;
    joined.setMembersFromNames({"a", "b", "z"}, ((mb).nameMap));
    mb.assignClassesById(main, {joined});
    ana.equivalenceProcessor.refreshCanonicalClasses(mb, main);
    ASSERT_EQ(index.dirtyCaches.count(), 0);
    ASSERT_EQ(gl::tests::OrImplicationProcessorFixture::refreshOrCanonicalViews(ana.orImplicationProcessor, mb, index), 0);
    ASSERT_EQ(orViewText(mb, index, watch), std::string("!(p[a])"));
    // A smaller member joins: the representative of a and b moves to 0.
    gl::EquivalenceClass moved;
    moved.setMembersFromNames({"0", "a", "b", "z"}, ((mb).nameMap));
    mb.assignClassesById(main, {moved});
    ana.equivalenceProcessor.refreshCanonicalClasses(mb, main);
    ASSERT_EQ(index.dirtyCaches.count(), 1);
    ASSERT_EQ(gl::tests::OrImplicationProcessorFixture::refreshOrCanonicalViews(ana.orImplicationProcessor, mb, index), 1);
    ASSERT_EQ(orViewText(mb, index, watch), std::string("!(p[0])"));
}

TEST(or_implication_matching, lookup_canonicalization_preserves_dependencies_and_locality) {
    gl::tests::ProverFixture ana("Peano");
    ana.parameters.trackHistory = true;
    auto& mb = ana.body;
    const auto main = mb.nameMap.encode("main");
    depositAtMain(ana, "(=[b,a])", 0);
    depositAtMain(ana, "(=[e,b])", 3);
    const auto child = mb.nameMap.encodePush(main, "lookup_child");
    ana.canonicalizer.initializeCanonicalScope(mb, child);
    gl::ScratchArena arena{&gl::staticMemory()};
    for (const auto scope : {main, child}) {
        for (const char* text : {"!(p[e,b])", "(=[e,z])", "!(=[e,z])",
                                 "(p[a])", "(AnchorProbe[e])"}) {
            gl::ScratchScope window(arena);
            const auto original = mb.nameMap.encode(text);
            gl::WatchBridge full, lookup;
            int levels[256] = {7};
            int32_t count = 1;
            ana.watchProcessor.canonicalWatchFormAt(mb, original, scope,
                arena, full, levels, count, 256, true);
            int32_t emptyCount = 0;
            ana.watchProcessor.canonicalWatchFormAt(mb, original, scope,
                arena, lookup, nullptr, emptyCount, 0, true, false);
            ASSERT_EQ(lookup.changed, full.changed);
            ASSERT_EQ(lookup.text.toStdString(), full.text.toStdString());
            ASSERT_EQ(lookup.eqN, full.eqN);
            ASSERT_EQ(lookup.localEquality, full.localEquality);
            ASSERT_EQ(emptyCount, 0);
            for (int32_t i = 0; i < full.eqN; ++i) {
                ASSERT_EQ(lookup.eqText[i].toStdString(), full.eqText[i].toStdString());
                ASSERT_TRUE(full.eqScope[i].len > 0);
                ASSERT_EQ(lookup.eqScope[i].len, 0);
            }
            if (full.changed) {
                ASSERT_TRUE(full.eqN > 0);
                ASSERT_TRUE(full.localEquality);
                ASSERT_TRUE(count > 1);
            } else {
                ASSERT_EQ(full.eqN, 0);
                ASSERT_EQ(count, 1);
                ASSERT_EQ(levels[0], 7);
            }
        }
    }
}

// The representative stays, but a mailed direct equality makes the member's
// rewrite non-local: the rebuild records the new locality and the OR entry is
// dirtied through the spelling's rewritten token e (change bit 16), so the
// spelling is matched again and a firing reads its new locality.
TEST(or_implication_matching, changed_rewrite_locality_marks_cache_stale) {
    gl::tests::ProverFixture ana("Peano");
    ana.parameters.trackHistory = true;
    auto& mb = ana.body;
    auto& index = ((mb).orImplicationMatches);
    const auto main = ((mb).nameMap).encode("main");
    depositAtMain(ana, "(=[b,a])", 0);   // local equality -> class {a,b}
    depositAtMain(ana, "(=[e,b])", 3);   // mailed: e joins through b; e -> a is local through (=[b,a])
    const auto raw = ((mb).nameMap).encode("!(p[e])");
    const gl::OrViewIdentity watch = gl::OrViewIdentity::sourceLeaf(gl::packInt32Pair(raw, main), main, 0);
    gl::tests::OrImplicationProcessorFixture::bindOrCanonicalView(ana.orImplicationProcessor, mb, index, watch, raw);
    ASSERT_EQ(orViewText(mb, index, watch), std::string("!(p[a])"));
    const auto canonical = ((mb).nameMap).lookup("!(p[a])");
    ASSERT_TRUE(gl::tests::OrImplicationProcessorFixture::rewriteLocalAt(ana.orImplicationProcessor, mb, raw, main, canonical));
    const int64_t eKey = gl::packStatementKey(((mb).nameMap).lookup("e"), main);
    ASSERT_TRUE(gl::canonicalRecordLocal(*mb.lbMemory.canonicalMemberRepresentatives.find(eKey)));
    index.built = true;
    index.changedViews.resetToFresh();
    depositAtMain(ana, "(=[e,a])", 3);   // mailed direct row: e -> a turns non-local
    ana.equivalenceProcessor.refreshCanonicalClasses(mb, main);
    ASSERT_FALSE(gl::canonicalRecordLocal(*mb.lbMemory.canonicalMemberRepresentatives.find(eKey)));
    ASSERT_EQ(index.dirtyCaches.count(), 1);
    ASSERT_EQ(gl::tests::OrImplicationProcessorFixture::refreshOrCanonicalViews(ana.orImplicationProcessor, mb, index), 1);
    ASSERT_TRUE(index.changedViews.contains(watch));
    ASSERT_EQ(orViewText(mb, index, watch), std::string("!(p[a])"));
    ASSERT_FALSE(gl::tests::OrImplicationProcessorFixture::rewriteLocalAt(ana.orImplicationProcessor, mb, raw, main, canonical));
}

TEST(or_implication_matching, erased_cache_has_no_token_wakeup_or_stale_projection) {
    gl::tests::ProverFixture ana("Peano");
    ana.parameters.trackHistory = false;
    auto& mb = ana.body;
    SourceRig rig;
    auto& index = rig.matching;
    const auto main = gl::NameMap::MAIN_ID;
    const auto child = ((mb).nameMap).encodePush(main, "or_match_child");
    const auto raw = ((mb).nameMap).encode("!(p[b])");
    const gl::OrViewIdentity watch = gl::OrViewIdentity::sourceLeaf(gl::packInt32Pair(raw, main), main, 0);
    const int32_t first = gl::tests::OrImplicationProcessorFixture::bindOrCanonicalView(ana.orImplicationProcessor, mb, index, watch, raw);
    const int32_t second = gl::tests::OrImplicationProcessorFixture::bindOrCanonicalView(ana.orImplicationProcessor, mb, index, {watch.owner, main, 1}, raw);
    putOrClasses(mb, child, {{"a", "b"}});
    gl::tests::OrImplicationProcessorFixture::stageOrClassMember(ana.orImplicationProcessor, mb, index, ((mb).nameMap).encode("b"), child);
    index.eraseView(first, gl::genScratchArenas().forSlot(0));
    ASSERT_TRUE(index.postings().containsKey(gl::orPostingKey(
        gl::OrPostingKind::TokenCaches, ((mb).nameMap).lookup("b"))));
    index.eraseView(second, gl::genScratchArenas().forSlot(0));
    ASSERT_FALSE(index.postings().containsKey(gl::orPostingKey(
        gl::OrPostingKind::TokenCaches, ((mb).nameMap).lookup("b"))));
    ASSERT_EQ(gl::tests::OrImplicationProcessorFixture::refreshOrCanonicalViews(ana.orImplicationProcessor, mb, index), 0);
    ASSERT_EQ(index.liveViews, 0);
    gl::tests::OrImplicationProcessorFixture::bindOrCanonicalView(ana.orImplicationProcessor, mb, index, watch, raw);
    gl::tests::OrImplicationProcessorFixture::refreshOrCanonicalViews(ana.orImplicationProcessor, mb, index);
    ASSERT_EQ(index.deadCaches, 0);
    ASSERT_EQ(index.liveViews, 2);
    index.clear();
    ASSERT_EQ(index.liveBytes(), int64_t{0});
}

TEST(or_implication_matching, retired_cache_recreated_before_dirty_drain) {
    gl::tests::ProverFixture ana("Peano");
    ana.parameters.trackHistory = false;
    auto& mb = ana.body;
    SourceRig rig;
    auto& index = rig.matching;
    const auto main = gl::NameMap::MAIN_ID;
    const auto raw = ((mb).nameMap).encode("!(p[b])");
    const gl::OrViewIdentity watch = gl::OrViewIdentity::sourceLeaf(gl::packInt32Pair(raw, main), main, 0);
    const int32_t id = gl::tests::OrImplicationProcessorFixture::bindOrCanonicalView(ana.orImplicationProcessor, mb, index, watch, raw);
    putOrClasses(mb, main, {{"a", "b"}});
    gl::tests::OrImplicationProcessorFixture::stageOrClassMember(ana.orImplicationProcessor, mb, index, ((mb).nameMap).encode("b"), main);
    ASSERT_EQ(index.dirtyCaches.count(), 1);
    index.eraseView(id, gl::genScratchArenas().forSlot(0));
    gl::tests::OrImplicationProcessorFixture::bindOrCanonicalView(ana.orImplicationProcessor, mb, index, watch, raw);
    ASSERT_EQ(orViewText(mb, index, watch), std::string("!(p[a])"));
    // The retired entry of !(p[b]) stays dead; its queued update finds nothing.
    ASSERT_EQ(gl::tests::OrImplicationProcessorFixture::refreshOrCanonicalViews(ana.orImplicationProcessor, mb, index), 0);
    ASSERT_EQ(index.liveViews, 1);
    ASSERT_EQ(index.deadCaches, 1);
    ASSERT_FALSE(index.cacheLive(gl::packInt32Pair(raw, main)));
}

// Premise-born OR marks (D-528): only a marked,
// collected OR answers true; an unmarked OR source and a missing key are
// defined misses.
TEST(or_implications_admission, unconditional_ordis_mark_is_per_collected_key) {
    auto& arena = gl::genScratchArenas().forSlot(0);
    gl::CollectedArena collected(&arena);
    const std::string marked = "(or97[a,N])";
    const std::string plain = "(or98[a,N])";
    collected.markOrSource(gl::StrSpan(marked));
    collected.markOrSource(gl::StrSpan(plain));
    ASSERT_FALSE(collected.isUnconditionalOrdis(gl::StrSpan(marked)));
    collected.markUnconditionalOrdis(gl::StrSpan(marked));
    ASSERT_TRUE(collected.isUnconditionalOrdis(gl::StrSpan(marked)));
    ASSERT_TRUE(collected.isOrSource(gl::StrSpan(marked)));
    ASSERT_FALSE(collected.isUnconditionalOrdis(gl::StrSpan(plain)));
    const std::string missing = "(in[a,N])";
    ASSERT_FALSE(collected.isUnconditionalOrdis(gl::StrSpan(missing)));
}

TEST(or_implications_admission, intent_survives_collection_without_registering_a_source) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = false;
    auto& arena = gl::genScratchArenas().forSlot(0);
    gl::CollectedArena collected(&arena);
    const std::string ordinary = "(in[a,N])";
    const std::string name = analyzer.orRuleCompiler.findOrMintOrOperator(
        {"(in[u_1,u_2])", "!(in[u_2,u_1])"}, 2);
    analyzer.orRuleCompiler.preMintReducedOrs();
    const std::string source = "(" + name + "[a,N])";
    collected.ensureKey(gl::StrSpan(ordinary));
    ASSERT_FALSE(collected.isOrSource(gl::StrSpan(ordinary)));
    ASSERT_FALSE(collected.isOrSource(gl::StrSpan(source)));
    collected.markOrSource(gl::StrSpan(source));
    ASSERT_TRUE(collected.isOrSource(gl::StrSpan(source)));
    gl::DisintProducts products(&arena);
    analyzer.disintegrationProcessor.disintegrateExpr2(gl::StrSpan(source), analyzer.body, -1,
        true, gl::StrSpan("main", 4), products, false, false, nullptr, 0, false);
    ASSERT_TRUE(products.orSources.contains(gl::StrSpan(source), gl::StrSpan("main", 4)));
    ASSERT_FALSE(products.orSources.contains(gl::StrSpan(ordinary), gl::StrSpan("main", 4)));
    ASSERT_EQ(((analyzer.body).orImplications).sources.count(), 0);
}

TEST(or_implications_admission, finalized_nonlocal_or_keeps_polarity_order_and_levels) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = false;
    auto& mb = analyzer.body;
    const std::string name = analyzer.orRuleCompiler.findOrMintOrOperator(
        {"(in[u_1,u_2])", "!(in[u_2,u_1])"}, 2);
    const std::string source = "(" + name + "[a,N])";
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0), &dirty);
    const int levels[] = {3, 5};
    analyzer.statementMemory.addStatement(gl::StrSpan(source), mb, false, levels, 2,
        gl::TransientOrigin{}, gl::StrSpan("main", 4), out, true);
    ASSERT_EQ(out.size(), 0);
    ASSERT_EQ(((mb).orImplications).sources.count(), 1);
    const int64_t key = gl::packInt32Pair(((mb).nameMap).lookup(source), gl::NameMap::MAIN_ID);
    const auto* row = ((mb).orImplications).find(key);
    ASSERT_TRUE(row != nullptr);
    ASSERT_EQ(row->leafCount, 2);
    ASSERT_EQ(row->flags & gl::OrImplications::Local, uint32_t{0});
    ASSERT_EQ(((mb).nameMap).decode(((mb).orImplications).leaves[row->leafStart].literalId),
        std::string("(in[a,N])"));
    ASSERT_EQ(((mb).nameMap).decode(((mb).orImplications).leaves[row->leafStart + 1].complementId),
        std::string("(in[N,a])"));
    ASSERT_EQ(((mb).orImplications).levels[row->levelStart], 3);
    ASSERT_EQ(((mb).orImplications).levels[row->levelStart + 1], 5);
    const auto encoded = gl::encodeExpression(gl::StrSpan(source), gl::StrSpan("main", 4), ((mb).nameMap));
    analyzer.orImplicationProcessor.registerOrImplicationSource(mb, encoded);
    ASSERT_EQ(((mb).orImplications).sources.count(), 1);
    ASSERT_EQ(((mb).orImplications).pendingSources.size(), 1);
}

TEST(or_implications_admission, refused_or_has_no_source_and_fact_changes_use_signed_cores) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = false;
    auto& mb = analyzer.body;
    const std::string name = analyzer.orRuleCompiler.findOrMintOrOperator(
        {"(in[u_1,u_2])", "!(in[u_2,u_1])"}, 2);
    const std::string source = "(" + name + "[a,N])";
    ((mb).intAxedVariables).mint(((mb).nameMap).encode("a"));
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0), &dirty);
    const int levels[] = {0};
    analyzer.statementMemory.addStatement(gl::StrSpan(source), mb, false, levels, 1,
        gl::TransientOrigin{}, gl::StrSpan("main", 4), out, true);
    ASSERT_EQ(((mb).orImplications).sources.count(), 0);
    const auto core = ((mb).nameMap).encode("in");
    const auto fact = gl::encodeExpression(gl::StrSpan("!(in[b,N])", 10),
        gl::StrSpan("main", 4), ((mb).nameMap));
    const auto positive = gl::encodeExpression(gl::StrSpan("(in[b,N])", 9),
        gl::StrSpan("main", 4), ((mb).nameMap));
    const int64_t key = gl::packInt32Pair(fact.originalId, fact.validityId);
    gl::StatementArrivals::stageOrFactChange(mb, key);
    ASSERT_EQ(((mb).orImplications).pendingFacts.size(), 0);
    ((mb).orImplicationMatches).watchSourceCore(1, gl::packInt32Pair(core, 1));
    ((mb).orImplicationMatches).built = true;
    gl::StatementArrivals::stageOrFactChange(mb, key);
    gl::StatementArrivals::stageOrFactChange(mb, gl::packInt32Pair(positive.originalId, positive.validityId));
    ASSERT_EQ(((mb).orImplications).pendingFacts.size(), 1);
    ASSERT_EQ(((mb).orImplications).pendingFacts[0], key);
}

TEST(or_implication_matching, bootstrap_arrival_and_complete_source_retirement) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = false;
    auto& mb = analyzer.body;
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0), &dirty);
    const int levels[] = {0};
    const std::string main = "main";
    analyzer.statementMemory.addStatement(gl::StrSpan("!(in[a,N])", 10), mb, true, levels, 1,
        gl::TransientOrigin{}, gl::StrSpan(main), out);
    const std::string name = analyzer.orRuleCompiler.findOrMintOrOperator(
        {"(in[u_1,u_2])", "(in[u_2,u_1])"}, 2);
    const std::string source = "(" + name + "[a,N])";
    analyzer.statementMemory.addStatement(gl::StrSpan(source), mb, true, levels, 1,
        gl::TransientOrigin{}, gl::StrSpan(main), out, true);
    const int64_t key = gl::packInt32Pair(((mb).nameMap).lookup(source), gl::NameMap::MAIN_ID);
    gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
    ASSERT_TRUE(((mb).orImplicationMatches).built);
    ASSERT_EQ(((mb).orImplicationMatches).initializedCores.count(), 1);
    ASSERT_TRUE(((mb).orImplicationMatches).evaluations.contains({key, gl::NameMap::MAIN_ID}));
    ASSERT_EQ(((mb).orImplicationMatches).liveViews, 3);
    ASSERT_EQ(((mb).orImplicationMatches).caches().count(), 2);
    ((mb).orImplicationMatches).evaluations.resetToFresh();
    analyzer.statementMemory.addStatement(gl::StrSpan("(in[b,N])", 9), mb, true, levels, 1,
        gl::TransientOrigin{}, gl::StrSpan(main), out);
    ASSERT_EQ(((mb).orImplications).pendingFacts.size(), 0);
    analyzer.statementMemory.addStatement(gl::StrSpan("!(in[N,a])", 10), mb, true, levels, 1,
        gl::TransientOrigin{}, gl::StrSpan(main), out);
    ASSERT_EQ(((mb).orImplications).pendingFacts.size(), 1);
    gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
    ASSERT_EQ(((mb).orImplicationMatches).liveViews, 4);
    ASSERT_TRUE(((mb).orImplicationMatches).evaluations.contains({key, gl::NameMap::MAIN_ID}));
    const int32_t knownCount = ((mb).intKnownStatements).count();
    gl::tests::OrImplicationProcessorFixture::retireOrImplicationSource(analyzer.orImplicationProcessor, mb, key);
    ASSERT_TRUE(((mb).orImplications).find(key) == nullptr);
    ASSERT_EQ(((mb).orImplicationMatches).liveViews, 0);
    ASSERT_EQ(((mb).orImplicationMatches).initializedCores.count(), 0);
    ASSERT_EQ(((mb).intKnownStatements).count(), knownCount);
    gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
    ASSERT_EQ(((mb).orImplications).sources.count(), 0);
    ASSERT_EQ(((mb).orImplicationMatches).liveBytes(), int64_t{0});
}

TEST(or_implication_matching, sibling_fact_scopes_queue_separate_evaluations) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = false;
    auto& mb = analyzer.body;
    const auto main = gl::NameMap::MAIN_ID;
    const auto left = ((mb).nameMap).encodePush(main, "or_match_left");
    const auto right = ((mb).nameMap).encodePush(main, "or_match_right");
    const std::string leftName = ((mb).nameMap).decode(left);
    const std::string rightName = ((mb).nameMap).decode(right);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0), &dirty);
    const int levels[] = {0};
    const std::string name = analyzer.orRuleCompiler.findOrMintOrOperator(
        {"(in[u_1,u_2])", "(in[u_2,u_1])"}, 2);
    const std::string source = "(" + name + "[a,N])";
    analyzer.statementMemory.addStatement(gl::StrSpan(source), mb, true, levels, 1,
        gl::TransientOrigin{}, gl::StrSpan("main", 4), out, true);
    gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
    ASSERT_TRUE(((mb).orImplicationMatches).evaluations.empty());
    analyzer.statementMemory.addStatement(gl::StrSpan("!(in[a,N])", 10), mb, true, levels, 1,
        gl::TransientOrigin{}, gl::StrSpan(leftName), out);
    analyzer.statementMemory.addStatement(gl::StrSpan("!(in[N,a])", 10), mb, true, levels, 1,
        gl::TransientOrigin{}, gl::StrSpan(rightName), out);
    gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
    const int64_t key = gl::packInt32Pair(((mb).nameMap).lookup(source), main);
    ASSERT_EQ(((mb).orImplicationMatches).evaluations.count(), 2);
    ASSERT_TRUE(((mb).orImplicationMatches).evaluations.contains({key, left}));
    ASSERT_TRUE(((mb).orImplicationMatches).evaluations.contains({key, right}));
    ASSERT_FALSE(((mb).orImplicationMatches).evaluations.contains({key, main}));
    mb.wipeSubtree(left);
    gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
    ASSERT_EQ(((mb).orImplicationMatches).evaluations.count(), 1);
    ASSERT_TRUE(((mb).orImplicationMatches).evaluations.contains({key, right}));
}

TEST(or_implication_matching, queued_equality_wakes_unused_fact_and_source) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = false;
    auto& mb = analyzer.body;
    const auto main = gl::NameMap::MAIN_ID;
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0), &dirty);
    const int levels[] = {0};
    const std::string name = analyzer.orRuleCompiler.findOrMintOrOperator(
        {"(in[u_1,u_2])", "(in[u_2,u_1])"}, 2);
    const std::string source = "(" + name + "[b,N])";
    analyzer.statementMemory.addStatement(gl::StrSpan(source), mb, true, levels, 1,
        gl::TransientOrigin{}, gl::StrSpan("main", 4), out, true);
    analyzer.statementMemory.addStatement(gl::StrSpan("!(in[a,N])", 10), mb, false, levels, 1,
        gl::TransientOrigin{}, gl::StrSpan("main", 4), out);
    gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
    ASSERT_TRUE(((mb).orImplicationMatches).evaluations.empty());
    putOrClasses(mb, main, {{"a", "b"}}, false);
    analyzer.equivalenceProcessor.refreshCanonicalClasses(mb,main);
    for (const char* token : {"a", "b"})
        ((mb).orImplications).pendingClasses.push_back(gl::packInt32Pair(((mb).nameMap).encode(token), main));
    gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
    const int64_t key = gl::packInt32Pair(((mb).nameMap).lookup(source), main);
    ASSERT_TRUE(((mb).orImplicationMatches).evaluations.contains({key, main}));
    ASSERT_TRUE(((mb).orImplications).pendingClasses.empty());
}

TEST(or_implication_matching, whole_or_canonicalization_covers_both_equality_leaf_polarities) {
    for (bool negative : {false, true}) {
        gl::tests::ProverFixture analyzer("Peano");
        analyzer.parameters.trackHistory = false;
        auto& mb = analyzer.body;
        const auto main = gl::NameMap::MAIN_ID;
        const auto child = ((mb).nameMap).encodePush(main, "or_equality_child");
        gl::DirtyState dirty = gl::DirtyState::Clean;
        gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0), &dirty);
        const int levels[] = {0};
        const std::string equality = negative ? "!(=[u_1,u_2])" : "(=[u_1,u_2])";
        const std::string name = analyzer.orRuleCompiler.findOrMintOrOperator({"(in[u_1,u_2])", equality}, 2);
        const std::string source = "(" + name + "[b,a])";
        analyzer.statementMemory.addStatement(gl::StrSpan(source), mb, true, levels, 1,
            gl::TransientOrigin{}, gl::StrSpan("main", 4), out, true);
        putOrClasses(mb, child, {{"a", "b"}});
        gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
        const int64_t key = gl::packInt32Pair(((mb).nameMap).lookup(source), main);
        const auto atMain = gl::OrViewIdentity::sourceLeaf(key, main, 1);
        const auto atChild = atMain.atScope(child);
        ASSERT_EQ(orViewText(mb, ((mb).orImplicationMatches), atMain),
            negative ? std::string("(=[b,a])") : std::string("!(=[b,a])"));
        ASSERT_EQ(orViewText(mb, ((mb).orImplicationMatches), atChild),
            negative ? std::string("(=[a,a])") : std::string("!(=[a,a])"));
        ASSERT_EQ(((mb).orImplicationMatches).caches().count(), 2);
        const int64_t cache = gl::tests::OrImplicationProcessorFixture::ensureOrCanonicalCache(analyzer.orImplicationProcessor, mb, ((mb).orImplicationMatches),
            ((mb).nameMap).lookup(source), child);
        const int32_t first = *((mb).orImplicationMatches).viewIds.find({key, child, 0});
        const int32_t second = *((mb).orImplicationMatches).viewIds.find(atChild);
        ASSERT_EQ(((mb).orImplicationMatches).views[first].cacheKey, cache);
        ASSERT_EQ(((mb).orImplicationMatches).views[second].cacheKey, cache);
        putOrClasses(mb, child, {{"0", "a", "b"}}, false);
        analyzer.equivalenceProcessor.refreshCanonicalClasses(mb,child);
        for (const char* token : {"0", "a", "b"})
            ((mb).orImplications).pendingClasses.push_back(gl::packInt32Pair(((mb).nameMap).encode(token), child));
        gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
        ASSERT_EQ(orViewText(mb, ((mb).orImplicationMatches), atChild),
            negative ? std::string("(=[0,0])") : std::string("!(=[0,0])"));
        ASSERT_EQ(orViewText(mb, ((mb).orImplicationMatches), atMain),
            negative ? std::string("(=[b,a])") : std::string("!(=[b,a])"));
    }
}

TEST(or_implication_matching, one_or_cache_serves_leaf_views_and_an_independent_fact_view) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = false;
    auto& mb = analyzer.body;
    const auto main = gl::NameMap::MAIN_ID;
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0), &dirty);
    const int levels[] = {0};
    const std::string firstName = analyzer.orRuleCompiler.findOrMintOrOperator(
        {"(in[u_1,u_2])", "(=[u_1,u_2])"}, 2);
    const std::string secondName = analyzer.orRuleCompiler.findOrMintOrOperator(
        {"!(" + firstName + "[u_1,u_2])", "(in[u_2,u_1])"}, 2);
    const std::string firstSource = "(" + firstName + "[b,a])";
    const std::string secondSource = "(" + secondName + "[b,a])";
    for (const std::string* source : {&firstSource, &secondSource})
        analyzer.statementMemory.addStatement(gl::StrSpan(*source), mb, true, levels, 1,
            gl::TransientOrigin{}, gl::StrSpan("main", 4), out, true);
    gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
    const int64_t firstKey = gl::packInt32Pair(((mb).nameMap).lookup(firstSource), main);
    const gl::OrViewIdentity fact = gl::OrViewIdentity::knownFact(firstKey, main);
    const int32_t factId = *((mb).orImplicationMatches).viewIds.find(fact);
    const int32_t leafId = *((mb).orImplicationMatches).viewIds.find({firstKey, main, 1});
    ASSERT_EQ(((mb).orImplicationMatches).views[factId].cacheKey,
        ((mb).orImplicationMatches).views[leafId].cacheKey);
    ASSERT_TRUE(((mb).orImplicationMatches).views[factId].templateId
        != ((mb).orImplicationMatches).views[leafId].templateId);
    putOrClasses(mb, main, {{"a", "b"}}, false);
    analyzer.equivalenceProcessor.refreshCanonicalClasses(mb,main);
    for (const char* token : {"a", "b"})
        ((mb).orImplications).pendingClasses.push_back(gl::packInt32Pair(((mb).nameMap).encode(token), main));
    gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
    ASSERT_EQ(orViewText(mb, ((mb).orImplicationMatches), fact), "(" + firstName + "[a,a])");
    ASSERT_EQ(orViewText(mb, ((mb).orImplicationMatches), {firstKey, main, 1}),
        std::string("!(=[a,a])"));
    gl::tests::OrImplicationProcessorFixture::retireOrImplicationSource(analyzer.orImplicationProcessor, mb, firstKey);
    ASSERT_TRUE(((mb).orImplicationMatches).views[factId].live != 0);
    ASSERT_EQ(*((mb).orImplicationMatches).viewIds.find(fact), factId);
    gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
    ASSERT_EQ(orViewText(mb, ((mb).orImplicationMatches), fact), "(" + firstName + "[a,a])");
}

TEST(or_implication_matching, domains_and_shared_buckets_remain_independent) {
    SourceRig rig;
    auto& index = rig.matching;
    const int64_t cache = gl::packInt32Pair(22, 1);
    publishTestCache(rig, cache, gl::OrCanonicalCache{22, 41, 0});
    const gl::OrViewIdentity fact = gl::OrViewIdentity::knownFact(gl::packInt32Pair(21, 1), 1);
    const gl::OrViewIdentity first = gl::OrViewIdentity::sourceLeaf(gl::packInt32Pair(31, 1), 1, 0);
    const gl::OrViewIdentity second = gl::OrViewIdentity::sourceLeaf(first.owner, 1, 1);
    const int32_t f = index.bindView(fact, cache, 21);
    const int32_t a = index.bindView(first, cache, 23);
    const int32_t b = index.bindView(second, cache, 23);
    ASSERT_EQ(index.bindView(first, cache, 23), a);
    ASSERT_EQ(index.liveViews, 3);
    // Watch views queue themselves; the fact carrier queues its entry's fact role.
    ASSERT_EQ(index.changedViews.count(), 2);
    ASSERT_EQ(index.changedFacts.count(), 1);
    ASSERT_TRUE(index.changedFacts.contains(cache));
    const auto watch = gl::orPostingKey(gl::OrPostingKind::WatchKey,
        gl::packInt32Pair(41, 1));
    const auto factKey = gl::orPostingKey(gl::OrPostingKind::FactKey,
        gl::packInt32Pair(41, 1));
    std::set<int64_t> hit;
    index.postings().forEach(watch, [&](int64_t id) { hit.insert(id); });
    ASSERT_TRUE(hit == (std::set<int64_t>{a, b}));
    ASSERT_TRUE(postingValues(index, gl::OrPostingKind::FactKey, gl::packInt32Pair(41, 1))
        == (std::set<int64_t>{cache}));
    ASSERT_EQ(index.caches().find(cache)->factCarriers, 1);
    index.eraseView(a, rig.scratch);
    ASSERT_TRUE(index.postings().containsKey(watch));
    ASSERT_TRUE(index.postings().containsKey(gl::orPostingKey(
        gl::OrPostingKind::WatchScopes, 41)));
    index.eraseView(b, rig.scratch);
    ASSERT_FALSE(index.postings().containsKey(watch));
    ASSERT_FALSE(index.postings().containsKey(gl::orPostingKey(
        gl::OrPostingKind::WatchScopes, 41)));
    ASSERT_TRUE(index.postings().containsKey(factKey));
    ASSERT_EQ(index.bindView(first, cache, 23), b);
    ASSERT_EQ(index.liveViews, 2);
    index.eraseView(b, rig.scratch);
    index.eraseView(f, rig.scratch);
    ASSERT_FALSE(index.postings().containsKey(factKey));
    ASSERT_FALSE(index.postings().containsKey(gl::orPostingKey(
        gl::OrPostingKind::FactScopes, 41)));
    ASSERT_FALSE(index.cacheLive(cache));
    index.maintain();
    ASSERT_EQ(index.views.size(), 0);
    ASSERT_EQ(index.viewIds.count(), 0);
    index.clear();
    ASSERT_EQ(index.liveBytes(), int64_t{0});
    ASSERT_FALSE(index.built);
}

TEST(or_implication_matching, rekey_collision_and_identity_churn) {
    SourceRig rig;
    auto& index = rig.matching;
    const int64_t oldCache = gl::packInt32Pair(21, 1);
    const int64_t otherCache = gl::packInt32Pair(22, 1);
    publishTestCache(rig, oldCache, gl::OrCanonicalCache{21, 41, 0}, {7});
    publishTestCache(rig, otherCache, gl::OrCanonicalCache{22, 42, 0}, {8});
    const gl::OrViewIdentity keep = gl::OrViewIdentity::sourceLeaf(gl::packInt32Pair(31, 1), 1, 0);
    const int32_t keepId = index.bindView(keep, otherCache, 22);
    const gl::OrViewIdentity move = gl::OrViewIdentity::sourceLeaf(gl::packInt32Pair(32, 1), 1, 0);
    const int32_t moveId = index.bindView(move, oldCache, 21);
    const gl::OrViewIdentity fact = gl::OrViewIdentity::knownFact(gl::packInt32Pair(21, 1), 1);
    const int32_t factId = index.bindView(fact, oldCache, 21);
    index.changedViews.resetToFresh();
    index.changedFacts.resetToFresh();
    // The form of the old entry moved to the other entry's: its users merge.
    index.moveView(moveId, otherCache);
    index.moveView(factId, otherCache);
    index.retireCache(oldCache, rig.scratch);
    ASSERT_FALSE(index.postings().containsKey(gl::orPostingKey(
        gl::OrPostingKind::WatchKey, gl::packInt32Pair(41, 1))));
    ASSERT_TRUE(postingValues(index, gl::OrPostingKind::WatchKey, gl::packInt32Pair(42, 1))
        == (std::set<int64_t>{keepId, moveId}));
    ASSERT_TRUE(postingValues(index, gl::OrPostingKind::CacheUsers, otherCache)
        == (std::set<int64_t>{keepId, moveId, factId}));
    ASSERT_FALSE(index.postings().containsKey(gl::orPostingKey(
        gl::OrPostingKind::FactKey, gl::packInt32Pair(41, 1))));
    ASSERT_TRUE(postingValues(index, gl::OrPostingKind::FactKey, gl::packInt32Pair(42, 1))
        == (std::set<int64_t>{otherCache}));
    ASSERT_FALSE(index.postings().containsKey(gl::orPostingKey(gl::OrPostingKind::TokenCaches, 7)));
    ASSERT_FALSE(index.cacheLive(oldCache));
    ASSERT_EQ(index.views[factId].canonicalId, 22);
    ASSERT_EQ(index.views[factId].templateId, 42);
    ASSERT_EQ(index.views[moveId].cacheKey, otherCache);
    ASSERT_TRUE(index.changedViews.contains(move));
    ASSERT_TRUE(index.changedFacts.contains(otherCache));
    ASSERT_EQ(index.caches().find(otherCache)->factCarriers, 1);
    index.eraseView(moveId, rig.scratch);
    for (int32_t i = 0; i < 200; ++i) {
        const gl::OrViewIdentity transient = gl::OrViewIdentity::sourceLeaf(gl::packInt32Pair(1000 + i, 1), 1, 0);
        const int32_t id = index.bindView(transient, otherCache, 22);
        ASSERT_EQ(id, moveId);
        index.eraseView(id, rig.scratch);
    }
    index.maintain();
    ASSERT_EQ(index.viewIds.count(), 2);
    ASSERT_EQ(*index.viewIds.find(keep), keepId);
    ASSERT_EQ(*index.viewIds.find(fact), factId);
    ASSERT_EQ(index.liveViews, 2);
}

TEST(or_implication_postings, duplicate_and_missing_edges) {
    PostingRig rig;
    ASSERT_EQ(rig.index.size(), 0);
    ASSERT_TRUE(values(rig.index, 11).empty());
    ASSERT_TRUE(!rig.index.containsKey(11));
    ASSERT_TRUE(!rig.index.erase(11, 37));
    ASSERT_TRUE(rig.index.insert(11, 37));
    ASSERT_TRUE(rig.index.containsKey(11));
    ASSERT_TRUE(!rig.index.insert(11, 37));
    ASSERT_TRUE(rig.index.insert(12, 37));
    ASSERT_EQ(rig.index.size(), 2);
    ASSERT_TRUE(values(rig.index, 11) == std::vector<int64_t>{37});
    ASSERT_TRUE(rig.index.erase(11, 37));
    ASSERT_TRUE(!rig.index.containsKey(11));
    ASSERT_TRUE(!rig.index.erase(11, 37));
    ASSERT_TRUE(values(rig.index, 11).empty());
    ASSERT_TRUE(values(rig.index, 12) == std::vector<int64_t>{37});
    ASSERT_TRUE(rig.index.insert(11, 37));
    ASSERT_EQ(rig.index.size(), 2);
}

TEST(or_implication_postings, immediate_head_middle_tail_unlink) {
    PostingRig rig;
    for (int value = 0; value < 5; ++value)
        ASSERT_TRUE(rig.index.insert(7, value));
    ASSERT_TRUE(rig.index.erase(7, 4));
    ASSERT_TRUE(rig.index.erase(7, 2));
    ASSERT_TRUE(rig.index.erase(7, 0));
    ASSERT_TRUE(values(rig.index, 7) == (std::vector<int64_t>{1, 3}));
    rig.index.maintain();
    ASSERT_TRUE(values(rig.index, 7) == (std::vector<int64_t>{1, 3}));
    ASSERT_TRUE(rig.index.erase(7, 3));
    ASSERT_TRUE(rig.index.erase(7, 1));
    ASSERT_TRUE(values(rig.index, 7).empty());
    rig.index.maintain();
    ASSERT_EQ(rig.index.size(), 0);
    ASSERT_EQ(rig.index.liveBytes(), int64_t{0});
}

TEST(or_implication_postings, churn_matches_sets_and_bounds_retained_storage) {
    PostingRig rig;
    std::map<int64_t, std::set<int64_t>> expected;
    for (int64_t value = 0; value < 256; ++value) {
        rig.index.insert(value % 17, value);
        expected[value % 17].insert(value);
    }
    const int64_t initialBytes = rig.index.liveBytes();
    for (int64_t round = 0; round < 40; ++round) {
        for (int64_t value = round * 192; value < (round + 1) * 192; ++value) {
            const int64_t key = value % 17;
            ASSERT_TRUE(rig.index.erase(key, value));
            expected[key].erase(value);
        }
        rig.index.maintain();
        for (int64_t value = 256 + round * 192;
             value < 256 + (round + 1) * 192; ++value) {
            const int64_t key = value % 17;
            ASSERT_TRUE(rig.index.insert(key, value));
            expected[key].insert(value);
        }
        ASSERT_EQ(rig.index.size(), 256);
        for (const auto& entry : expected) {
            const std::vector<int64_t> ordered(entry.second.begin(), entry.second.end());
            ASSERT_TRUE(values(rig.index, entry.first) == ordered);
        }
        ASSERT_TRUE(rig.index.liveBytes() <= initialBytes * 3);
    }
    rig.index.clear();
    ASSERT_EQ(rig.index.liveBytes(), int64_t{0});
    ASSERT_TRUE(rig.index.insert(-17, int64_t{1} << 40));
    ASSERT_TRUE(values(rig.index, -17) == std::vector<int64_t>{int64_t{1} << 40});
}

TEST(or_implications_storage, source_identity_leaf_order_and_retirement) {
    SourceRig rig;
    const gl::OrImplicationLeaf leaves[] = {{19, 20}, {3, 4}, {11, 12}};
    const int levels[] = {2, 5};
    const int64_t key = gl::packInt32Pair(101, 1);
    ASSERT_TRUE(rig.store.find(key) == nullptr);
    ASSERT_TRUE(!rig.store.retire(key));
    ASSERT_TRUE(rig.store.insert(key, 101, leaves, 3, levels, 2, true));
    ASSERT_TRUE(!rig.store.insert(key, 101, leaves, 3, levels, 2, false));
    const gl::OrImplicationSource source = *rig.store.find(key);
    ASSERT_EQ(source.leafCount, 3);
    ASSERT_EQ(source.levelCount, 2);
    ASSERT_TRUE((source.flags & gl::OrImplications::Local) != 0);
    ASSERT_EQ(rig.store.leaves[source.leafStart].literalId, 19);
    ASSERT_EQ(rig.store.leaves[source.leafStart + 1].literalId, 3);
    ASSERT_EQ(rig.store.levels[source.levelStart + 1], 5);
    ASSERT_EQ(rig.store.pendingSources.size(), 1);
    ASSERT_TRUE(rig.store.retire(key));
    ASSERT_TRUE(rig.store.find(key) == nullptr);
    ASSERT_TRUE(!rig.store.retire(key));
    ASSERT_TRUE(!rig.store.insert(key, 101, leaves, 3, levels, 2, false));
    rig.store.compactRetired(rig.scratch);
    ASSERT_EQ(rig.store.sources.count(), 0);
    ASSERT_EQ(rig.store.leaves.size(), 0);
    ASSERT_EQ(rig.store.levels.size(), 0);
    rig.store.clear();
    ASSERT_EQ(rig.store.liveBytes(), int64_t{0});
}

TEST(or_implications_storage, payload_compaction_preserves_survivors) {
    SourceRig rig;
    const int levels[] = {7};
    for (int32_t i = 1; i <= 4; ++i) {
        const gl::OrImplicationLeaf leaves[] = {{i * 10, i * 10 + 1},
                                               {i * 10 + 2, i * 10 + 3}};
        ASSERT_TRUE(rig.store.insert(gl::packInt32Pair(100 + i, 1),
                                    100 + i, leaves, 2, levels, 1, false));
    }
    rig.store.compactRetired(rig.scratch);
    ASSERT_EQ(rig.store.leaves.size(), 8);
    for (int32_t i = 1; i <= 3; ++i)
        ASSERT_TRUE(rig.store.retire(gl::packInt32Pair(100 + i, 1)));
    rig.store.compactRetired(rig.scratch);
    ASSERT_EQ(rig.store.sources.count(), 1);
    ASSERT_EQ(rig.store.leaves.size(), 2);
    ASSERT_EQ(rig.store.levels.size(), 1);
    const gl::OrImplicationSource kept = *rig.store.find(gl::packInt32Pair(104, 1));
    ASSERT_EQ(kept.leafStart, 0);
    ASSERT_EQ(kept.levelStart, 0);
    ASSERT_EQ(rig.store.leaves[kept.leafStart].literalId, 40);
    ASSERT_EQ(rig.store.leaves[kept.leafStart + 1].complementId, 43);
    ASSERT_EQ(rig.store.levels[kept.levelStart], 7);
}

TEST(or_implications_symmetry, sources_keep_first_arrival_and_exact_scope) {
    for (bool reverse : {false, true}) {
        gl::tests::ProverFixture analyzer("Peano");
        const std::string name = symmetricSourceOperator(analyzer);
        auto& mb = analyzer.body;
        const std::string a = "(" + name + "[a,b,f])";
        const std::string b = "(" + name + "[b,a,f])";
        const auto child = ((mb).nameMap).encodePush(gl::NameMap::MAIN_ID, "symmetric_child");
        const int64_t separate = putFiringFact(analyzer, a, ((mb).nameMap).decode(child), true);
        const int64_t first = putFiringFact(analyzer, reverse ? b : a, "main", true);
        const int64_t second = putFiringFact(analyzer, reverse ? a : b, "main", true);
        const int32_t history = ((mb).exprOriginMap).count();
        gl::tests::OrImplicationProcessorFixture::retireDuplicateOrSources(analyzer.orImplicationProcessor, mb);
        ASSERT_TRUE(((mb).orImplications).find(first) != nullptr);
        ASSERT_TRUE(((mb).orImplications).find(second) == nullptr);
        ASSERT_TRUE(((mb).orImplications).find(separate) != nullptr);
        ASSERT_TRUE(((mb).intKnownStatements).find(gl::StatementKey{
            ((mb).nameMap).lookup(a), gl::NameMap::MAIN_ID}) != nullptr);
        ASSERT_TRUE(((mb).intKnownStatements).find(gl::StatementKey{
            ((mb).nameMap).lookup(b), gl::NameMap::MAIN_ID}) != nullptr);
        ASSERT_EQ(((mb).exprOriginMap).count(), history);
        gl::tests::OrImplicationProcessorFixture::retireDuplicateOrSources(analyzer.orImplicationProcessor, mb);
        ASSERT_TRUE(((mb).orImplications).find(first) != nullptr);
    }
}

TEST(or_implications_symmetry, residual_continuation_cannot_restore_duplicate_work) {
    gl::tests::ProverFixture analyzer("Peano");
    const std::string name = symmetricSourceOperator(analyzer);
    auto& mb = analyzer.body;
    const std::string a = "(" + name + "[a,b,f])";
    const std::string b = "(" + name + "[b,a,f])";
    const int64_t first = putFiringFact(analyzer, a, "main", true);
    const int64_t second = putFiringFact(analyzer, b);
    ASSERT_TRUE(analyzer.orImplicationProcessor.activateOrImplicationContinuation(mb, gl::StrSpan(b),
        gl::NameMap::MAIN_ID, ((mb).nameMap).lookup(b)));
    gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
    ASSERT_TRUE(((mb).orImplications).find(first) != nullptr);
    ASSERT_TRUE(((mb).orImplications).find(second) == nullptr);
}

TEST(or_implications_symmetry, equality_collision_keeps_larger_cohort_and_history) {
    for (bool tied : {false, true}) {
        gl::tests::ProverFixture analyzer("Peano");
        const std::string name = symmetricSourceOperator(analyzer);
        auto& mb = analyzer.body;
        const std::string originals[] = {"(" + name + "[c,a,f])", "(" + name + "[a,b,f])"};
        int64_t keys[2];
        int32_t cohorts[2];
        gl::NameId branches[2];
        for (int i = 0; i < 2; ++i) {
            keys[i] = putFiringFact(analyzer, originals[i], "main", true);
            const auto original = ((mb).nameMap).lookup(originals[i]);
            analyzer.caseLedger.recordProcessedOr(mb, original, gl::NameMap::MAIN_ID);
            cohorts[i] = gl::mintOrCohortId(((mb).lbStateInterner),
                ((mb).lbStateInterner).encode(gl::StrSpan("main", 4)),
                ((mb).lbStateInterner).encode(gl::StrSpan(originals[i])));
            ((mb).orDisjunctCount).insert(cohorts[i], 3);
            const std::string branch = "main_boundary_ordis_" + originals[i] + "_((in2[a,b,f]))";
            branches[i] = ((mb).nameMap).encode(branch);
            ((mb).orLiveBranches).mint(branches[i]);
            const int count = i == 0 && !tied ? 2 : 1;
            for (int j = 0; j < count; ++j)
                ((mb).intEncodedStatements).push_back(gl::encodeExpression(gl::StrSpan("(in[a,N])", 9),
                    gl::StrSpan(branch), ((mb).nameMap)));
            ASSERT_EQ(analyzer.caseProcessor.orCohortStatementCount(mb, gl::StrSpan("main", 4),
                gl::StrSpan(originals[i]), gl::OrScopeKind::Disintegration), count);
        }
        putOrClasses(mb, gl::NameMap::MAIN_ID, {{"b", "c"}});
        analyzer.caseLedger.recanonicalizeProcessedOrLedger(mb, gl::NameMap::MAIN_ID);
        const int32_t history = ((mb).exprOriginMap).count();
        gl::tests::OrImplicationProcessorFixture::retireDuplicateOrSources(analyzer.orImplicationProcessor, mb);
        analyzer.caseProcessor.retireDuplicateOrCohorts(mb);
        const int winner = tied ? 1 : 0;
        ASSERT_TRUE(((mb).orImplications).find(keys[winner]) != nullptr);
        ASSERT_TRUE(((mb).orImplications).find(keys[1 - winner]) == nullptr);
        ASSERT_NE(((mb).orDisjunctCount).lookup(cohorts[winner]), 0);
        ASSERT_EQ(((mb).orDisjunctCount).lookup(cohorts[1 - winner]), 0);
        ASSERT_NE(((mb).pendingWipeScopes).lookup(branches[1 - winner]), 0);
        ASSERT_EQ(((mb).exprOriginMap).count(), history);
    }
}

TEST(or_implications_storage, clone_and_ordered_deload_facets) {
    SourceRig original;
    SourceRig copy;
    const gl::OrImplicationLeaf leaves[] = {{2, 3}, {5, 6}};
    const int64_t key = gl::packInt32Pair(101, 7);
    ASSERT_TRUE(original.store.insert(key, 99, leaves, 2, nullptr, 0, false));
    original.store.pendingFacts.push_back(gl::packInt32Pair(3, 1));
    original.store.pendingClasses.push_back(gl::packInt32Pair(55, 7));
    copy.store.copyFrom(original.store);
    ASSERT_TRUE(copy.store.find(key) != nullptr);
    ASSERT_EQ(copy.store.find(key)->canonicalId, 99);
    ASSERT_EQ(copy.store.pendingFacts[0], original.store.pendingFacts[0]);
    ASSERT_EQ(copy.store.pendingClasses[0], original.store.pendingClasses[0]);
    ASSERT_EQ(copy.store.pendingSources[0], key);
    ASSERT_TRUE(original.store.retire(key));
    ASSERT_TRUE(copy.store.find(key) != nullptr);

    std::vector<uint32_t> writableTags;
    copy.store.visitContainers(900, [&](uint32_t tag, auto&) {
        writableTags.push_back(tag);
    });
    std::vector<uint32_t> constTags;
    const gl::OrImplications& frozen = copy.store;
    frozen.visitContainers(900, [&](uint32_t tag, const auto&) {
        constTags.push_back(tag);
    });
    ASSERT_TRUE(writableTags == constTags);
    ASSERT_EQ(writableTags.size(), static_cast<size_t>(gl::OrImplications::kFacetCount));
    for (uint32_t i = 0; i < gl::OrImplications::kFacetCount; ++i)
        ASSERT_EQ(writableTags[i], 900 + i);
}

TEST(or_implications_storage, block_scope_wipe_and_fact_clone_keep_ancestor_source) {
    gl::tests::ProverFixture analyzer(std::string("Peano"));
    analyzer.parameters.trackHistory = false;
    gl::Memory memory;
    const gl::NameId main = ((memory).nameMap).encode("main");
    const gl::NameId child = ((memory).nameMap).encodePush(main, "or_source_child");
    const gl::NameId grandchild = ((memory).nameMap).encodePush(child, "nested");
    const gl::NameId sibling = ((memory).nameMap).encodePush(main, "or_source_sibling");
    const gl::NameId original = ((memory).nameMap).encode("(in[a,N])");
    const gl::NameId opposite = ((memory).nameMap).encode("!(in[a,N])");
    const gl::OrImplicationLeaf leaves[] = {{original, opposite}, {opposite, original}};
    const gl::NameId scopes[] = {main, child, grandchild, sibling};
    for (const gl::NameId scope : scopes) {
        const int64_t key = gl::packInt32Pair(original, scope);
        ASSERT_TRUE(((memory).orImplications).insert(key, original, leaves, 2, nullptr, 0, false));
        ((memory).orImplications).pendingFacts.push_back(gl::packInt32Pair(opposite, scope));
        ((memory).orImplications).pendingClasses.push_back(gl::packInt32Pair(original, scope));
        gl::tests::OrImplicationProcessorFixture::bindOrCanonicalView(analyzer.orImplicationProcessor, memory, ((memory).orImplicationMatches),
            {key, scope, 0}, opposite);
    }
    ((memory).orImplicationMatches).built = true;
    gl::LbStore cloneStore{&gl::lbMemory(), sizeof(gl::Memory), alignof(gl::Memory)};
    gl::Memory* copy = memory.cloneFactsTemplate(cloneStore);
    ASSERT_EQ(((*(copy)).orImplicationMatches).liveBytes(), int64_t{0});
    ASSERT_FALSE(((*(copy)).orImplicationMatches).built);
    memory.wipeSubtree(child);
    ASSERT_EQ(((memory).orImplicationMatches).liveBytes(), int64_t{0});
    ASSERT_FALSE(((memory).orImplicationMatches).built);
    ASSERT_TRUE(((memory).orImplications).find(gl::packInt32Pair(original, main)) != nullptr);
    ASSERT_TRUE(((memory).orImplications).find(gl::packInt32Pair(original, sibling)) != nullptr);
    ASSERT_TRUE(((memory).orImplications).find(gl::packInt32Pair(original, child)) == nullptr);
    ASSERT_TRUE(((memory).orImplications).find(gl::packInt32Pair(original, grandchild)) == nullptr);
    ASSERT_EQ(((memory).orImplications).pendingSources.size(), 2);
    ASSERT_EQ(((memory).orImplications).pendingFacts.size(), 2);
    ASSERT_EQ(((memory).orImplications).pendingClasses.size(), 2);
    ASSERT_EQ(((*(copy)).orImplications).sources.count(), 4);
    ASSERT_TRUE(((*(copy)).orImplications).find(gl::packInt32Pair(original, child)) != nullptr);
    cloneStore.destroy(copy);
}

TEST(or_implication_matching, explicit_roles_keep_fact_leaf_and_evaluation_separate) {
    const auto fact = gl::OrViewIdentity::knownFact(11, 1);
    const auto leaf = gl::OrViewIdentity::sourceLeaf(11, 1, 0);
    ASSERT_TRUE(fact.isFact());
    ASSERT_FALSE(leaf.isFact());
    ASSERT_TRUE(fact.atScope(2).isFact());
    ASSERT_FALSE(leaf.atScope(2).isFact());
    ASSERT_EQ(leaf.atScope(2).owner, leaf.owner);
    ASSERT_EQ(leaf.atScope(2).leaf, 0);
    ASSERT_EQ(leaf.atScope(2).scope, 2);
    SourceRig rig;
    rig.matching.evaluations.mint({11, 1});
    rig.matching.evaluations.mint({11, 1});
    ASSERT_EQ(rig.matching.evaluations.count(), 1);
    ASSERT_EQ(rig.matching.evaluations.keyAt(1).reserved, 0);
    ASSERT_TRUE(rig.matching.viewIds.empty());
}

TEST(or_implication_matching, cache_publication_rekeys_tokens_and_last_user_cleans_all_links) {
    SourceRig rig;
    auto& index = rig.matching;
    const int64_t key = gl::packInt32Pair(21, 1);
    publishTestCache(rig, key, {21, 41, 0}, {7, 8, 7});
    const int32_t first = index.bindView(gl::OrViewIdentity::knownFact(key, 1), key, 21);
    const int32_t second = index.bindView(gl::OrViewIdentity::sourceLeaf(key, 1, 0), key, 21);
    index.markCacheDirty(key);
    index.markCacheDirty(key);
    ASSERT_EQ(index.dirtyCaches.count(), 1);
    ASSERT_EQ(index.caches().find(key)->dirty, 1);
    // An unchanged form settles in place and queues both roles again.
    index.changedViews.resetToFresh();
    index.changedFacts.resetToFresh();
    index.settleCache(key);
    ASSERT_EQ(index.caches().find(key)->dirty, 0);
    ASSERT_TRUE(index.changedViews.contains(gl::OrViewIdentity::sourceLeaf(key, 1, 0)));
    ASSERT_TRUE(index.changedFacts.contains(key));
    // A moved form moves both users to the entry of the new form.
    const int64_t moved = gl::packInt32Pair(22, 1);
    publishTestCache(rig, moved, {22, 42, 0}, {8, 9});
    index.moveView(first, moved);
    index.moveView(second, moved);
    index.retireCache(key, rig.scratch);
    ASSERT_FALSE(index.postings().containsKey(gl::orPostingKey(gl::OrPostingKind::TokenCaches, 7)));
    ASSERT_TRUE(index.postings().containsKey(gl::orPostingKey(gl::OrPostingKind::TokenCaches, 9)));
    ASSERT_TRUE(postingValues(index, gl::OrPostingKind::TokenCaches, 8) == (std::set<int64_t>{moved}));
    ASSERT_FALSE(index.postings().containsKey(gl::orPostingKey(gl::OrPostingKind::WatchKey, gl::packInt32Pair(41, 1))));
    ASSERT_EQ(index.views[first].canonicalId, 22);
    ASSERT_EQ(index.views[second].canonicalId, 22);
    ASSERT_EQ(index.caches().find(key)->templateId, 0);
    ASSERT_EQ(index.deadCaches, 1);
    index.eraseView(first, rig.scratch);
    ASSERT_EQ(index.caches().find(moved)->templateId, 42);
    index.eraseView(second, rig.scratch);
    ASSERT_EQ(index.caches().find(moved)->templateId, 0);
    ASSERT_FALSE(index.postings().containsKey(gl::orPostingKey(gl::OrPostingKind::CacheScope, 1)));
    ASSERT_FALSE(index.postings().containsKey(gl::orPostingKey(gl::OrPostingKind::TokenCaches, 9)));
    publishTestCache(rig, key, {21, 43, 0}, {10});
    ASSERT_EQ(index.deadCaches, 1);
    ASSERT_FALSE(index.postings().containsKey(gl::orPostingKey(gl::OrPostingKind::TokenCaches, 8)));
}

// A view's rewritten tokens dirty its entry at the entry's scope and project
// that view alone below it; canonical tokens reach every user of the entry.
TEST(or_implication_matching, view_tokens_dirty_their_entry_and_project_only_their_view) {
    gl::tests::ProverFixture ana("Peano");
    ana.parameters.trackHistory = false;
    auto& mb = ana.body;
    SourceRig rig;
    auto& index = rig.matching;
    const auto main = gl::NameMap::MAIN_ID;
    const auto child = ((mb).nameMap).encodePush(main, "or_view_token_child");
    putOrClasses(mb, main, {{"a", "b"}});
    const auto canonical = ((mb).nameMap).encode("!(p[a,N])");
    const auto spelled = ((mb).nameMap).encode("!(p[b,N])");
    const gl::OrViewIdentity canonicalFact = gl::OrViewIdentity::knownFact(gl::packInt32Pair(canonical, main), main);
    const gl::OrViewIdentity spelledFact = gl::OrViewIdentity::knownFact(gl::packInt32Pair(spelled, main), main);
    const int32_t c = gl::tests::OrImplicationProcessorFixture::bindOrCanonicalView(ana.orImplicationProcessor, mb, index, canonicalFact, canonical);
    const int32_t s = gl::tests::OrImplicationProcessorFixture::bindOrCanonicalView(ana.orImplicationProcessor, mb, index, spelledFact, spelled);
    const int64_t entry = gl::packInt32Pair(canonical, main);
    ASSERT_EQ(index.views[c].cacheKey, entry);
    ASSERT_EQ(index.views[s].cacheKey, entry);
    // b is rewritten only in the spelled carrier; N and a belong to the form.
    const auto b = ((mb).nameMap).lookup("b");
    ASSERT_TRUE(postingValues(index, gl::OrPostingKind::ViewTokens, s) == (std::set<int64_t>{b}));
    ASSERT_FALSE(index.postings().containsKey(gl::orPostingKey(gl::OrPostingKind::ViewTokens, c)));
    ASSERT_TRUE(postingValues(index, gl::OrPostingKind::TokenViews, b) == (std::set<int64_t>{s}));
    ASSERT_FALSE(index.postings().containsKey(gl::orPostingKey(gl::OrPostingKind::TokenCaches, b)));
    index.markTokenUsersDirty(b, child);
    ASSERT_TRUE(index.dirtyCaches.empty());
    index.markTokenUsersDirty(b, main);
    ASSERT_TRUE(index.dirtyCaches.contains(entry));
    index.dirtyCaches.resetToFresh();
    index.markTokenUsersDirty(((mb).nameMap).lookup("N"), main);
    ASSERT_TRUE(index.dirtyCaches.contains(entry));
    // A class below on b projects only the spelled carrier; on a, both.
    gl::tests::OrImplicationProcessorFixture::stageOrClassMember(ana.orImplicationProcessor, mb, index, b, child, false);
    ASSERT_EQ(index.projections.count(), 1);
    ASSERT_TRUE(index.projections.contains(spelledFact.atScope(child)));
    gl::tests::OrImplicationProcessorFixture::stageOrClassMember(ana.orImplicationProcessor, mb, index, ((mb).nameMap).lookup("a"), child, false);
    ASSERT_EQ(index.projections.count(), 2);
    ASSERT_TRUE(index.projections.contains(canonicalFact.atScope(child)));
    index.eraseViewTokens(s, rig.scratch);
    ASSERT_FALSE(index.postings().containsKey(gl::orPostingKey(gl::OrPostingKind::TokenViews, b)));
    ASSERT_FALSE(index.postings().containsKey(gl::orPostingKey(gl::OrPostingKind::ViewTokens, s)));
}

// Two stored spellings become one canonical form when an equality arrives:
// their entries merge, one fact posting remains, and selection takes the
// carriers in the order local first, then the smaller spelling, with the
// rewrite locality computed at selection.
TEST(or_implication_matching, merged_spellings_share_an_entry_and_select_local_then_smaller) {
    for (bool localEquality : {false, true}) {
        gl::tests::ProverFixture analyzer("Peano");
        analyzer.parameters.trackHistory = false;
        auto& mb = analyzer.body;
        auto& index = ((mb).orImplicationMatches);
        const auto main = gl::NameMap::MAIN_ID;
        const auto name = analyzer.orRuleCompiler.findOrMintOrOperator(
            {"(in[u_1,u_2])", "(in[u_2,u_1])"}, 2);
        const int64_t plain = putFiringFact(analyzer, "!(in[a,N])", "main", false, false);
        const int64_t spelled = putFiringFact(analyzer, "!(in[b,N])", "main", false, false);
        const int64_t key = putFiringFact(analyzer, "(" + name + "[a,N])", "main", true, true);
        gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
        const int64_t entry = gl::packInt32Pair(((mb).nameMap).lookup("!(in[a,N])"), main);
        const int64_t other = gl::packInt32Pair(((mb).nameMap).lookup("!(in[b,N])"), main);
        ASSERT_TRUE(index.cacheLive(entry));
        ASSERT_TRUE(index.cacheLive(other));
        putFiringFact(analyzer, "(=[a,b])", "main", false, localEquality);
        gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
        ASSERT_FALSE(index.cacheLive(other));
        ASSERT_EQ(index.caches().find(entry)->factCarriers, 2);
        const int32_t shape = ((mb).templateInterner).lookup(gl::StrSpan("!(in[a,N])", 10));
        ASSERT_TRUE(postingValues(index, gl::OrPostingKind::FactKey, gl::packInt32Pair(shape, main))
            == (std::set<int64_t>{entry}));
        // The expected refuter by the rule: the first local carrier in spelling
        // order, else the smallest spelling.
        int64_t expected = 0;
        bool expectedLocal = false;
        for (const int64_t owner : {plain, spelled}) {
            const auto original = static_cast<gl::NameId>(uint64_t(owner) >> 32);
            const bool local = ((mb).intKnownStatements).find({original, main})->local
                || (original != static_cast<gl::NameId>(uint64_t(entry) >> 32)
                    && gl::tests::OrImplicationProcessorFixture::rewriteLocalAt(analyzer.orImplicationProcessor,
                        mb, original, main, static_cast<gl::NameId>(uint64_t(entry) >> 32)));
            if (expected == 0) expected = owner;
            if (local) { expected = owner; expectedLocal = true; break; }
        }
        if (!localEquality) {
            ASSERT_EQ(expected, plain);
            ASSERT_FALSE(expectedLocal);
        } else {
            ASSERT_TRUE(expectedLocal);
        }
        gl::OrImplicationSelection selection;
        ASSERT_TRUE(gl::tests::OrImplicationProcessorFixture::selectOrImplication(analyzer.orImplicationProcessor, mb, key, main, selection));
        ASSERT_EQ(selection.choices[0].owner, expected);
        ASSERT_EQ(selection.choices[0].local, expectedLocal);
        ASSERT_EQ(selection.choices[0].canonical, static_cast<gl::NameId>(uint64_t(entry) >> 32));
    }
}

// A fact carrier leaving keeps its entry and fact posting for the others;
// the last carrier takes the fact role and the entry with it.
TEST(or_implication_matching, entry_survives_until_its_last_fact_carrier_leaves) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = false;
    auto& mb = analyzer.body;
    auto& index = ((mb).orImplicationMatches);
    const auto main = gl::NameMap::MAIN_ID;
    const auto name = analyzer.orRuleCompiler.findOrMintOrOperator(
        {"(in[u_1,u_2])", "(in[u_2,u_1])"}, 2);
    putFiringFact(analyzer, "!(in[a,N])", "main", false, false);
    putFiringFact(analyzer, "!(in[b,N])", "main", false, false);
    putFiringFact(analyzer, "(" + name + "[c,N])", "main", true, true);
    putFiringFact(analyzer, "(=[a,b])", "main", false, false);
    gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
    const int64_t entry = gl::packInt32Pair(((mb).nameMap).lookup("!(in[a,N])"), main);
    ASSERT_EQ(index.caches().find(entry)->factCarriers, 2);
    const int32_t shape = ((mb).templateInterner).lookup(gl::StrSpan("!(in[a,N])", 10));
    const auto factKey = gl::orPostingKey(gl::OrPostingKind::FactKey, gl::packInt32Pair(shape, main));
    gl::StatementRegistryOperations::eraseKnownStatementKey(mb, ((mb).nameMap).lookup("!(in[b,N])"), main);
    gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
    ASSERT_EQ(index.caches().find(entry)->factCarriers, 1);
    ASSERT_TRUE(index.postings().containsKey(factKey));
    gl::StatementRegistryOperations::eraseKnownStatementKey(mb, ((mb).nameMap).lookup("!(in[a,N])"), main);
    gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
    ASSERT_FALSE(index.cacheLive(entry));
    ASSERT_FALSE(index.postings().containsKey(factKey));
}

// The text-only lookup form is the text the evidence canonicalization yields.
TEST(or_implication_matching, text_only_form_matches_the_evidence_form) {
    gl::tests::ProverFixture ana("Peano");
    ana.parameters.trackHistory = true;
    auto& mb = ana.body;
    const auto main = mb.nameMap.encode("main");
    depositAtMain(ana, "(=[b,a])", 0);
    depositAtMain(ana, "(=[e,b])", 3);
    const auto child = mb.nameMap.encodePush(main, "text_only_child");
    ana.canonicalizer.initializeCanonicalScope(mb, child);
    gl::ScratchArena arena{&gl::staticMemory()};
    for (const auto scope : {main, child}) {
        for (const char* text : {"!(p[e,b])", "(=[e,z])", "!(=[e,z])", "(p[a])", "(AnchorProbe[e])"}) {
            gl::ScratchScope window(arena);
            const auto original = mb.nameMap.encode(text);
            gl::WatchBridge full;
            int32_t emptyCount = 0;
            ana.watchProcessor.canonicalWatchFormAt(mb, original, scope,
                arena, full, nullptr, emptyCount, 0, true, false);
            const std::string expected = full.text.toStdString();
            const auto form = gl::tests::OrImplicationProcessorFixture::canonicalOrFormId(ana.orImplicationProcessor,
                mb, original, scope);
            ASSERT_EQ(mb.nameMap.decode(form), expected);
            ASSERT_EQ(form == original, !full.changed);
        }
    }
}

TEST(or_implication_matching, fact_removal_preserves_source_role_and_source_retirement_unlinks_pairs) {
    SourceRig rig;
    auto& index = rig.matching;
    const int64_t owner = gl::packInt32Pair(21, 1);
    const int64_t other = gl::packInt32Pair(22, 1);
    const int64_t core = gl::packInt32Pair(30, 1);
    publishTestCache(rig, owner, {21, 41, 0}, {7});
    const int32_t leaf = index.bindView(gl::OrViewIdentity::sourceLeaf(owner, 1, 0), owner, 21);
    index.bindView(gl::OrViewIdentity::knownFact(owner, 1), owner, 21);
    index.watchSourceCore(owner, core);
    index.watchSourceCore(owner, core);
    index.watchSourceCore(other, core);
    index.recordFactCore(owner, core);
    index.initializedCores.mint(core);
    index.recordClassScope(7, 1);
    index.recordClassScope(7, 1);
    ASSERT_TRUE(index.postings().containsKey(gl::orPostingKey(gl::OrPostingKind::ClassScopes, 7)));
    index.eraseFact(owner, core, rig.scratch);
    index.eraseFact(owner, core, rig.scratch);
    ASSERT_TRUE(index.views[leaf].live != 0);
    ASSERT_FALSE(index.postings().containsKey(gl::orPostingKey(gl::OrPostingKind::FactCore, core)));
    index.retireSource(owner, rig.scratch);
    ASSERT_TRUE(index.postings().containsKey(gl::orPostingKey(gl::OrPostingKind::SourceCore, core)));
    ASSERT_FALSE(index.postings().containsKey(gl::orPostingKey(gl::OrPostingKind::OwnerCores, owner)));
    index.retireSource(other, rig.scratch);
    ASSERT_FALSE(index.postings().containsKey(gl::orPostingKey(gl::OrPostingKind::SourceCore, core)));
    ASSERT_FALSE(index.initializedCores.contains(core));
}

TEST(or_implication_matching, canonical_publication_streams_entry_and_rewritten_view_tokens) {
    gl::tests::ProverFixture analyzer("Peano");
    auto& mb = analyzer.body;
    auto& index = ((mb).orImplicationMatches);
    const auto original = ((mb).nameMap).encode("!(in[a,N])");
    const auto canonical = ((mb).nameMap).encode("!(in[b,N])");
    const int32_t shape = ((mb).templateInterner).encode(((mb).nameMap).decodeView(canonical));
    const auto key = gl::packInt32Pair(canonical, gl::NameMap::MAIN_ID);
    gl::tests::OrImplicationProcessorFixture::publishOrCanonicalCache(analyzer.orImplicationProcessor, mb, index, key,
        {canonical, shape, 0});
    const auto b = ((mb).nameMap).lookup("b");
    const auto n = ((mb).nameMap).lookup("N");
    ASSERT_TRUE(postingValues(index, gl::OrPostingKind::CacheTokens, key) == (std::set<int64_t>{b, n}));
    ASSERT_TRUE(postingValues(index, gl::OrPostingKind::CacheScope, gl::NameMap::MAIN_ID) == (std::set<int64_t>{key}));
    const gl::OrViewIdentity fact = gl::OrViewIdentity::knownFact(gl::packInt32Pair(original, gl::NameMap::MAIN_ID),
        gl::NameMap::MAIN_ID);
    const int32_t view = index.bindView(fact, key, original);
    gl::tests::OrImplicationProcessorFixture::postOrViewTokens(analyzer.orImplicationProcessor, mb, index, view, original, canonical);
    const auto a = ((mb).nameMap).lookup("a");
    ASSERT_TRUE(postingValues(index, gl::OrPostingKind::ViewTokens, view) == (std::set<int64_t>{a}));
    ASSERT_FALSE(index.postings().containsKey(gl::orPostingKey(gl::OrPostingKind::TokenViews, n)));
    // A canonical spelling adds no token.
    const gl::OrViewIdentity canonicalFact = gl::OrViewIdentity::knownFact(gl::packInt32Pair(canonical, gl::NameMap::MAIN_ID),
        gl::NameMap::MAIN_ID);
    const int32_t same = index.bindView(canonicalFact, key, canonical);
    gl::tests::OrImplicationProcessorFixture::postOrViewTokens(analyzer.orImplicationProcessor, mb, index, same, canonical, canonical);
    ASSERT_FALSE(index.postings().containsKey(gl::orPostingKey(gl::OrPostingKind::ViewTokens, same)));
    ASSERT_FALSE(index.postings().containsKey(
        gl::orPostingKey(gl::OrPostingKind::TokenCaches, ((mb).nameMap).encode("unrelated"))));
}

TEST(or_implication_firing, selection_has_no_firing_effects_and_materialization_consumes_its_choices) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = true;
    auto& mb = analyzer.body;
    const auto name = analyzer.orRuleCompiler.findOrMintOrOperator(
        {"(in[u_1,u_2])", "(in[u_2,u_1])"}, 2);
    const auto key = putFiringFact(analyzer, "(" + name + "[a,N])", "main", true);
    gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
    gl::OrImplicationSelection selection;
    ASSERT_FALSE(gl::tests::OrImplicationProcessorFixture::selectOrImplication(analyzer.orImplicationProcessor, mb, 0, gl::NameMap::MAIN_ID, selection));
    ASSERT_FALSE(gl::tests::OrImplicationProcessorFixture::selectOrImplication(analyzer.orImplicationProcessor, mb, key, gl::NameMap::MAIN_ID, selection));
    putFiringFact(analyzer, "!(in[a,N])");
    gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
    const int32_t knownBefore = ((mb).intKnownStatements).count();
    const int32_t originsBefore = ((mb).exprOriginMap).count();
    ASSERT_TRUE(gl::tests::OrImplicationProcessorFixture::selectOrImplication(analyzer.orImplicationProcessor, mb, key, gl::NameMap::MAIN_ID, selection));
    ASSERT_EQ(selection.sourceKey, key);
    ASSERT_EQ(selection.refuted, 1);
    ASSERT_EQ(selection.survivors, 1);
    ASSERT_EQ(((mb).intKnownStatements).count(), knownBefore);
    ASSERT_EQ(((mb).exprOriginMap).count(), originsBefore);
    ASSERT_TRUE(((mb).sameIterationInternalMail).statementsEmpty());
    ASSERT_TRUE(((mb).orImplications).find(key) != nullptr);
    ASSERT_TRUE(gl::tests::OrImplicationProcessorFixture::materializeOrImplication(analyzer.orImplicationProcessor, mb, selection, ((mb).sameIterationInternalMail)));
    const auto heads = firingHeads(analyzer, ((mb).sameIterationInternalMail));
    ASSERT_EQ(heads.size(), size_t{1});
    ASSERT_EQ(heads[0].original, std::string("(in[N,a])"));
    ASSERT_TRUE(((mb).orImplications).find(key) == nullptr);
    ASSERT_TRUE(!((mb).sameIterationInternalMail).exprOriginMapEmpty());
}

TEST(or_implication_firing, materialization_of_a_known_head_retires_without_duplicate_mail) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = false;
    auto& mb = analyzer.body;
    const auto name = analyzer.orRuleCompiler.findOrMintOrOperator(
        {"(in[u_1,u_2])", "(in[u_2,u_1])"}, 2);
    const auto key = putFiringFact(analyzer, "(" + name + "[a,N])", "main", true);
    putFiringFact(analyzer, "!(in[a,N])");
    putFiringFact(analyzer, "(in[N,a])");
    gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
    gl::OrImplicationSelection selection;
    ASSERT_TRUE(gl::tests::OrImplicationProcessorFixture::selectOrImplication(analyzer.orImplicationProcessor, mb, key, gl::NameMap::MAIN_ID, selection));
    ASSERT_TRUE(gl::tests::OrImplicationProcessorFixture::materializeOrImplication(analyzer.orImplicationProcessor, mb, selection, ((mb).sameIterationInternalMail)));
    ASSERT_TRUE(((mb).sameIterationInternalMail).statementsEmpty());
    ASSERT_TRUE(((mb).orImplications).find(key) == nullptr);
}

TEST(or_implication_firing, materialization_keeps_the_existing_hypothetical_premise_gate) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = false;
    auto& mb = analyzer.body;
    const auto scope = ((mb).nameMap).encodePush(gl::NameMap::MAIN_ID, "_hypo_scope_");
    const auto name = analyzer.orRuleCompiler.findOrMintOrOperator(
        {"(in[u_1,u_4])", "(in[u_2,u_4])", "(in[u_3,u_4])"}, 4);
    analyzer.orRuleCompiler.preMintReducedOrs();
    const auto key = putFiringFact(analyzer, "(" + name + "[a,b,c,N])", "main", true);
    putFiringFact(analyzer, "!(in[a,N])");
    putFiringFact(analyzer, "!(in[b,N])", ((mb).nameMap).decode(scope));
    gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
    gl::OrImplicationSelection selection;
    ASSERT_TRUE(gl::tests::OrImplicationProcessorFixture::selectOrImplication(analyzer.orImplicationProcessor, mb, key, scope, selection));
    ASSERT_FALSE(gl::tests::OrImplicationProcessorFixture::materializeOrImplication(analyzer.orImplicationProcessor, mb, selection, ((mb).sameIterationInternalMail)));
    ASSERT_TRUE(((mb).sameIterationInternalMail).statementsEmpty());
    ASSERT_TRUE(((mb).orImplications).find(key) != nullptr);
}

TEST(or_implication_matching, known_row_mutations_keep_their_signed_core_notifications) {
    gl::tests::ProverFixture analyzer("Peano");
    auto& mb = analyzer.body;
    const auto core = ((mb).nameMap).encode("in");
    const auto original = ((mb).nameMap).encode("!(in[a,N])");
    const auto positive = ((mb).nameMap).encode("(in[a,N])");
    const auto key = gl::packInt32Pair(original, gl::NameMap::MAIN_ID);
    const auto positiveKey = gl::packInt32Pair(positive, gl::NameMap::MAIN_ID);
    const int level[] = {0};
    ((mb).intStatementLevelsMap).assignSetRange(key, level, level + 1);
    ((mb).intStatementLevelsMap).assignSetRange(positiveKey, level, level + 1);
    gl::StatementRegistryOperations::publishKnownStatementKey(mb, key, true);
    ASSERT_EQ(((mb).orImplications).pendingFacts.size(), 0);
    ((mb).orImplicationMatches).watchSourceCore(key, gl::packInt32Pair(core, 1));
    ((mb).orImplicationMatches).built = true;
    gl::StatementRegistryOperations::publishKnownStatementKey(mb, key, false);
    gl::StatementRegistryOperations::publishKnownStatementKey(mb, key, true);
    ASSERT_TRUE(((mb).intKnownStatements).find({original, gl::NameMap::MAIN_ID})->local);
    ASSERT_EQ(((mb).orImplications).pendingFacts.size(), 2);
    gl::StatementRegistryOperations::eraseKnownStatementKey(mb, original, gl::NameMap::MAIN_ID);
    gl::StatementRegistryOperations::eraseKnownStatementKey(mb, original, gl::NameMap::MAIN_ID);
    ASSERT_TRUE(((mb).intKnownStatements).find({original, gl::NameMap::MAIN_ID}) == nullptr);
    ASSERT_EQ(((mb).orImplications).pendingFacts.size(), 4);
    gl::StatementRegistryOperations::publishKnownStatementKey(mb, positiveKey, false);
    gl::StatementRegistryOperations::publishKnownStatementKey(mb, positiveKey, true);
    ASSERT_FALSE(((mb).intKnownStatements).find({positive, gl::NameMap::MAIN_ID})->local);
    ASSERT_EQ(((mb).orImplications).pendingFacts.size(), 4);
    for (int32_t i = 0; i < 4; ++i) ASSERT_EQ(((mb).orImplications).pendingFacts[i], key);
}

TEST(or_source_release, d2_original_cohort_keeps_the_opposite_case_and_late_residual) {
    gl::tests::ProverFixture analyzer("FTA");
    analyzer.parameters.trackHistory = true;
    // This fixture isolates case release and exclusion from structural disintegration.
    analyzer.parameters.ban_disintegration = true;
    const std::vector<std::string> leaves = {
        "(strictOrder[1,4,q1,q2])", "(=[q1,q2])", "(strictOrder[1,4,q2,q1])"};
    const auto name = analyzer.orRuleCompiler.findOrMintOrOperator({
        "(strictOrder[u_1,u_2,u_3,u_4])", "(=[u_3,u_4])",
        "(strictOrder[u_1,u_2,u_4,u_3])"}, 4);
    analyzer.orRuleCompiler.preMintReducedOrs();
    const std::string source = "(" + name + "[1,4,q1,q2])";
    auto& mb = analyzer.body;
    const auto key = putFiringFact(analyzer, source, "main", true);
    const auto original = ((mb).nameMap).lookup(source);
    const int32_t cohort = stageReleasedSourceCohort(analyzer, source, leaves);
    gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
    ASSERT_TRUE(((mb).orImplications).find(key) != nullptr);
    analyzer.caseProcessor.drainPendingOrReleases(mb);
    const auto first = firingHeads(analyzer, ((mb).sameIterationInternalMail));
    ASSERT_EQ(first.size(), size_t{1});
    ASSERT_EQ(first[0].original, leaves[0]);
    ASSERT_TRUE(((mb).orImplications).find(key) != nullptr);
    ASSERT_TRUE(((mb).intKnownStatements).find({original, gl::NameMap::MAIN_ID}) != nullptr);
    ASSERT_EQ(*((mb).orDisjunctCount).find(cohort), 3);
    ASSERT_EQ(((mb).orPendingBranches).runLen(((mb).orPendingBranches).lookup(cohort)), 2);
    ASSERT_FALSE(gl::decodeInternalMailOrigins(((mb).sameIterationInternalMail),
        ((mb).originInterner)).empty());
    ((mb).sameIterationInternalMail).clear();
    putFiringFact(analyzer, "!" + leaves[0]);
    analyzer.orImplicationProcessor.drainOrImplications(mb, ((mb).sameIterationInternalMail));
    const auto residual = firingHeads(analyzer, ((mb).sameIterationInternalMail));
    ASSERT_EQ(residual.size(), size_t{1});
    ASSERT_EQ(residual[0].validityName, std::string("main"));
    ASSERT_TRUE(((mb).orImplications).find(key) == nullptr);
    const int levels[] = {0};
    analyzer.caseProcessor.ordisMerge(gl::StrSpan(leaves[0]), gl::StrSpan(first[0].validityName),
        levels, 1, mb);
    analyzer.caseProcessor.drainDeadOrBranches(mb);
    ASSERT_EQ(*((mb).orDisjunctCount).find(cohort), 2);
    analyzer.caseProcessor.drainPendingOrReleases(mb);
    const auto next = firingHeads(analyzer, ((mb).sameIterationInternalMail));
    ASSERT_EQ(next.size(), size_t{2});
    bool released = false, retained = false;
    for (const auto& row : next) {
        if (row.original == leaves[2] && row.validityName.find(source) != std::string::npos) released = true;
        if (row.original == residual[0].original && row.validityName == "main") retained = true;
    }
    ASSERT_TRUE(released && retained);
}

TEST(or_source_release, aliases_reconcile_and_continuation_remains_available) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = false;
    const auto name = symmetricSourceOperator(analyzer);
    analyzer.orRuleCompiler.preMintReducedOrs();
    auto& mb = analyzer.body;
    const std::string source = "(" + name + "[a,b,s])";
    const std::string reverse = "(" + name + "[b,a,s])";
    const auto one = putFiringFact(analyzer, source, "main", true);
    const auto two = putFiringFact(analyzer, reverse, "main", true);
    ASSERT_TRUE(((mb).orImplications).find(one) != nullptr && ((mb).orImplications).find(two) != nullptr);
    const auto cohort = stageReleasedSourceCohort(analyzer, reverse,
        {"(in2[b,a,s])", "(=[b,a])", "(in2[a,b,s])"});
    analyzer.caseProcessor.drainPendingOrReleases(mb);
    ASSERT_TRUE(((mb).orImplications).find(one) != nullptr && ((mb).orImplications).find(two) != nullptr);
    gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
    ASSERT_EQ(((mb).orImplications).sources.count(), 1);
    ASSERT_TRUE(analyzer.orImplicationProcessor.activateOrImplicationContinuation(mb,
        gl::StrSpan(source), gl::NameMap::MAIN_ID, ((mb).nameMap).lookup(source)));
    const auto statement = gl::encodeExpression(gl::StrSpan(reverse), gl::StrSpan("main", 4), ((mb).nameMap));
    analyzer.orImplicationProcessor.registerOrImplicationSource(mb, statement);
    gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
    ASSERT_EQ(((mb).orImplications).sources.count(), 1);
    ASSERT_EQ(*((mb).orDisjunctCount).find(cohort), 3);
    ASSERT_EQ(((mb).orPendingBranches).runLen(((mb).orPendingBranches).lookup(cohort)), 2);
}

TEST(or_source_release, equality_reconciles_aliases_without_retiring_the_remaining_source) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = false;
    const auto name = symmetricSourceOperator(analyzer);
    auto& mb = analyzer.body;
    const std::string original = "(" + name + "[z,b,s])";
    const std::string canonical = "(" + name + "[a,b,s])";
    const auto originalKey = putFiringFact(analyzer, original, "main", true);
    stageReleasedSourceCohort(analyzer, original,
        {"(in2[z,b,s])", "(=[z,b])", "(in2[b,z,s])"});
    analyzer.caseProcessor.drainPendingOrReleases(mb);
    const auto key = putFiringFact(analyzer, canonical, "main", true);
    ASSERT_TRUE(((mb).orImplications).find(key) != nullptr);
    putFiringFact(analyzer, "(=[a,z])");
    gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
    ASSERT_EQ(((mb).orImplications).sources.count(), 1);
    ASSERT_TRUE(((mb).orImplications).find(key) != nullptr || ((mb).orImplications).find(originalKey) != nullptr);
    ASSERT_TRUE(((mb).intKnownStatements).find({((mb).nameMap).lookup(canonical), gl::NameMap::MAIN_ID}) != nullptr);
}

TEST(or_source_release, clone_and_parent_wipe_preserve_exact_scope_ownership) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = false;
    auto& mb = analyzer.body;
    const auto name = symmetricSourceOperator(analyzer);
    const std::string source = "(" + name + "[a,b,s])";
    const auto child = ((mb).nameMap).encodePush(gl::NameMap::MAIN_ID, "child");
    const auto grandchild = ((mb).nameMap).encodePush(child, "grandchild");
    const auto sibling = ((mb).nameMap).encodePush(gl::NameMap::MAIN_ID, "sibling");
    const auto childKey = putFiringFact(analyzer, source, ((mb).nameMap).decode(child), true);
    const auto siblingKey = putFiringFact(analyzer, source, ((mb).nameMap).decode(sibling), true);
    const auto parentKey = putFiringFact(analyzer, source, "main", true);
    stageReleasedSourceCohort(analyzer, source,
        {"(in2[a,b,s])", "(=[a,b])", "(in2[b,a,s])"}, ((mb).nameMap).decode(child));
    analyzer.caseProcessor.drainPendingOrReleases(mb);
    ASSERT_TRUE(((mb).orImplications).find(childKey) != nullptr);
    ASSERT_TRUE(((mb).orImplications).find(siblingKey) != nullptr);
    ASSERT_TRUE(((mb).orImplications).find(parentKey) != nullptr);
    gl::LbStore store{&gl::lbMemory(), sizeof(gl::Memory), alignof(gl::Memory)};
    gl::Memory* copy = mb.cloneFactsTemplate(store);
    ASSERT_TRUE(((*(copy)).orImplications).find(childKey) != nullptr);
    mb.wipeSubtree(grandchild);
    ASSERT_TRUE(((mb).orImplications).find(childKey) != nullptr);
    mb.wipeSubtree(child);
    ASSERT_TRUE(((mb).orImplications).find(childKey) == nullptr);
    ASSERT_TRUE(((mb).orImplications).find(siblingKey) != nullptr);
    ASSERT_TRUE(((mb).orImplications).find(parentKey) != nullptr);
    ASSERT_TRUE(((*(copy)).orImplications).find(childKey) != nullptr);
    store.destroy(copy);
}

TEST(or_source_release, original_cohort_still_derives_last_literal) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = true;
    const auto name = analyzer.orRuleCompiler.findOrMintOrOperator(
        {"(in[u_1,u_3])", "(in[u_2,u_3])"}, 3);
    const std::vector<std::string> leaves = {"(in[a,N])", "(in[b,N])"};
    const std::string source = "(" + name + "[a,b,N])";
    putFiringFact(analyzer, source, "main", true);
    auto& mb = analyzer.body;
    stageReleasedSourceCohort(analyzer, source, leaves);
    analyzer.caseProcessor.drainPendingOrReleases(mb);
    putFiringFact(analyzer, "!" + leaves[0]);
    for (int i = 0; i < 3 && !releasedSourceMainFact(analyzer, leaves[1]); ++i) {
        analyzer.logicBlockOperations.standardProcessing(mb, nullptr, ((mb).sameIterationInternalMail),
            ((mb).nextIterationInternalMail), -1);
        analyzer.logicBlockOperations.standardProcessing(mb, nullptr, ((mb).nextIterationInternalMail),
            ((mb).sameIterationInternalMail), -1);
        analyzer.caseProcessor.drainDeadOrBranches(mb);
        analyzer.caseProcessor.drainPendingOrReleases(mb);
    }
    ASSERT_TRUE(releasedSourceMainFact(analyzer, leaves[1]));
}

TEST(or_source_release, late_exclusion_survives_both_case_orders_and_retires_only_at_source_scope) {
    for (int firstCase : {0, 1}) {
        gl::tests::ProverFixture analyzer("Peano");
        analyzer.parameters.trackHistory = true;
        const auto name = analyzer.orRuleCompiler.findOrMintOrOperator(
            {"(in[u_1,u_3])", "(in[u_2,u_3])"}, 3);
        analyzer.orRuleCompiler.preMintReducedOrs();
        const std::vector<std::string> leaves = {"(in[a,N])", "(in[b,N])"};
        const std::string source = "(" + name + "[a,b,N])";
        auto& mb = analyzer.body;
        const auto sourceKey = putFiringFact(analyzer, source, "main", true);
        const auto cohort = stageReleasedSourceCohort(analyzer, source, leaves);
        const std::string selected = "(" + leaves[firstCase] + ")";
        ((mb).orStarterPick).upsert(cohort, ((mb).lbStateInterner).encode(gl::StrSpan(selected)));
        analyzer.caseProcessor.drainPendingOrReleases(mb);
        const auto released = firingHeads(analyzer, ((mb).sameIterationInternalMail));
        ASSERT_EQ(released.size(), size_t{1});
        ASSERT_EQ(released[0].original, leaves[firstCase]);
        const std::string childScope = released[0].validityName;
        const auto sibling = ((mb).nameMap).encodePush(gl::NameMap::MAIN_ID, "other_case");
        ((mb).sameIterationInternalMail).clear();
        putFiringFact(analyzer, "!" + leaves[firstCase], childScope);
        gl::tests::OrImplicationProcessorFixture::prepareOrImplicationMatches(analyzer.orImplicationProcessor, mb);
        gl::OrImplicationSelection selection;
        ASSERT_FALSE(gl::tests::OrImplicationProcessorFixture::selectOrImplication(analyzer.orImplicationProcessor, mb, sourceKey, gl::NameMap::MAIN_ID, selection));
        ASSERT_FALSE(gl::tests::OrImplicationProcessorFixture::selectOrImplication(analyzer.orImplicationProcessor, mb, sourceKey, sibling, selection));
        analyzer.orImplicationProcessor.drainOrImplications(mb, ((mb).sameIterationInternalMail));
        const auto descendant = firingHeads(analyzer, ((mb).sameIterationInternalMail));
        ASSERT_EQ(descendant.size(), size_t{1});
        ASSERT_EQ(descendant[0].original, leaves[1 - firstCase]);
        ASSERT_EQ(descendant[0].validityName, childScope);
        ASSERT_TRUE(((mb).orImplications).find(sourceKey) != nullptr);
        ASSERT_FALSE(gl::decodeInternalMailOrigins(((mb).sameIterationInternalMail), ((mb).originInterner)).empty());
        ASSERT_EQ(*((mb).orDisjunctCount).find(cohort), 2);
        ASSERT_EQ(((mb).orPendingBranches).runLen(((mb).orPendingBranches).lookup(cohort)), 1);
        ((mb).sameIterationInternalMail).clear();
        putFiringFact(analyzer, "!" + leaves[firstCase]);
        analyzer.orImplicationProcessor.drainOrImplications(mb, ((mb).sameIterationInternalMail));
        bool parentResult = false;
        for (const auto& row : firingHeads(analyzer, ((mb).sameIterationInternalMail)))
            if (row.original == leaves[1 - firstCase] && row.validityName == "main") parentResult = true;
        ASSERT_TRUE(parentResult);
        ASSERT_TRUE(((mb).orImplications).find(sourceKey) == nullptr);
    }
}

TEST(or_source_release, residual_emitted_before_release_remains_available) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = true;
    const auto name = analyzer.orRuleCompiler.findOrMintOrOperator(
        {"(in[u_1,u_4])", "(in[u_2,u_4])", "(in[u_3,u_4])"}, 4);
    analyzer.orRuleCompiler.preMintReducedOrs();
    const std::string source = "(" + name + "[a,b,c,N])";
    const std::vector<std::string> leaves = {"(in[a,N])", "(in[b,N])", "(in[c,N])"};
    putFiringFact(analyzer, source, "main", true);
    putFiringFact(analyzer, "!" + leaves[0]);
    auto& mb = analyzer.body;
    analyzer.orImplicationProcessor.drainOrImplications(mb, ((mb).sameIterationInternalMail));
    const auto before = firingHeads(analyzer, ((mb).sameIterationInternalMail));
    ASSERT_EQ(before.size(), size_t{1});
    const std::string residual = before[0].original;
    stageReleasedSourceCohort(analyzer, source, leaves);
    analyzer.caseProcessor.drainPendingOrReleases(mb);
    bool retained = false;
    for (const auto& row : firingHeads(analyzer, ((mb).sameIterationInternalMail)))
        if (row.original == residual && row.validityName == "main") retained = true;
    ASSERT_TRUE(retained);
    ASSERT_TRUE(((mb).orImplications).find(gl::packStatementKey(
        ((mb).nameMap).lookup(source), gl::NameMap::MAIN_ID)) == nullptr);
}

TEST(or_source_release, all_refuted_cohort_must_discharge_parent_contradiction) {
    for (bool branched : {false, true}) {
        gl::Memory parent;
        gl::tests::ProverFixture analyzer("Peano");
        analyzer.parameters.trackHistory = true;
        const auto name = analyzer.orRuleCompiler.findOrMintOrOperator(
            {"(in[u_1,u_3])", "(in[u_2,u_3])"}, 3);
        const std::vector<std::string> leaves = {"(in[a,N])", "(in[b,N])"};
        const std::string source = "(" + name + "[a,b,N])";
        putFiringFact(analyzer, source, "main", true);
        auto& mb = analyzer.body;
        mb.setExprKey("__contradiction__" + source);
        ((mb).parentMemory) = &parent;
        ((mb).isActive) = true;
        ((mb).primedForContradiction) = true;
        ((mb).level) = 0;
        int32_t cohort = 0;
        if (branched) {
            cohort = stageReleasedSourceCohort(analyzer, source, leaves);
            analyzer.caseProcessor.drainPendingOrReleases(mb);
            ASSERT_TRUE(((mb).orImplications).find(gl::packInt32Pair(
                ((mb).nameMap).lookup(source), gl::NameMap::MAIN_ID)) != nullptr);
        }
        ((mb).level) = 1;
        for (const auto& leaf : leaves) putFiringFact(analyzer, "!" + leaf);
        for (int step = 0; step < 4 && ((mb).isActive); ++step) {
            analyzer.logicBlockOperations.standardProcessing(mb, nullptr, ((mb).sameIterationInternalMail),
                ((mb).nextIterationInternalMail), -1);
            analyzer.logicBlockOperations.standardProcessing(mb, nullptr, ((mb).nextIterationInternalMail),
                ((mb).sameIterationInternalMail), -1);
            analyzer.caseProcessor.drainDeadOrBranches(mb);
            analyzer.caseProcessor.drainPendingOrReleases(mb);
        }
        bool mainCollision = false;
        for (const auto& leaf : leaves) {
            const gl::NameId positive = ((mb).nameMap).lookup(leaf);
            const gl::NameId negative = ((mb).nameMap).lookup("!" + leaf);
            mainCollision |= positive != 0 && negative != 0
                && ((mb).intKnownStatements).find({positive, gl::NameMap::MAIN_ID}) != nullptr
                && ((mb).intKnownStatements).find({negative, gl::NameMap::MAIN_ID}) != nullptr;
        }
        std::cout << "[OR-SOURCE-RELEASE] branched=" << branched
            << " parent_active=" << ((mb).isActive)
            << " main_collision=" << mainCollision
            << " remaining_cohorts=" << ((mb).orDisjunctCount).count() << '\n';
        ASSERT_TRUE(mainCollision);
        ASSERT_FALSE(((mb).isActive));
        gl::LbArena dischargeScratch(&gl::staticMemory());
        mb.dischargeStatementContent(dischargeScratch);
        if (branched) ASSERT_TRUE(((mb).orDisjunctCount).lookup(cohort) == 0);
        ASSERT_TRUE(((mb).orImplications).sources.empty());
        ASSERT_TRUE(((mb).orPendingBranches).empty());
        ASSERT_TRUE(((mb).orLiveBranches).empty());
    }
}
