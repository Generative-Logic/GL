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
/// @brief Unit tests for the integration root plumbing of the finished-task
///        wipe (I-284).
///
/// @details
/// Covers the task context register (`IntegrationTaskScope`), task and gate
/// registration with root classification, the sharing rule
/// (`noteIntegrationReach`, `recordIntegrationOwner`), the removability walk
/// (`integrationTaskWipeable`), the finish test
/// (`integrationRootKnownAtOrAbove`), the staging doors, the sweep and the
/// derivative selection. `in2[1,2,3]` declares slot 1 as its output in the
/// Peano configuration, which makes `(in2[a,marker,b])` an output root.

#include "test_harness.hpp"

#include "../prover.hpp"
#include "canonical_test_helpers.hpp"

#include <cstring>
#include <string>

namespace {
    using EA = gl::ExpressionAnalyzer;
    using Rec = gl::IntegrationTaskRec;

    /// @brief Span over a string literal (tests only).
    gl::StrSpan sp(const char* s) {
        return gl::StrSpan(s, static_cast<int32_t>(std::strlen(s)));
    }

    /// @brief The packed template key of @p text at @p scope.
    int64_t key(gl::Memory& m, const char* text, const char* scope = "main") {
        return gl::mintTemplateKey(m.templateInterner, m.nameMap, std::string(text),
                                   std::string(scope));
    }

    /// @brief Make @p text a known row at @p scope (the registry write only).
    int64_t makeKnown(gl::Memory& m, const char* text, gl::NameId scope) {
        const int64_t row = gl::packStatementKey(m.nameMap.encode(sp(text)), scope);
        gl::upsertStatementKey(m.intKnownStatements, row, /*local=*/true);
        return row;
    }

    /// @brief Make @p text a known row at @p scope marked as the product of an
    ///        integration (D-444).
    int64_t makeIntegrated(gl::Memory& m, const char* text, gl::NameId scope) {
        const int64_t row = makeKnown(m, text, scope);
        gl::markStatementIntegrated(m.intKnownStatements, m.nameMap.lookup(sp(text)), scope);
        return row;
    }

    /// @brief Register @p text at main as a task below @p parent.
    int64_t task(EA& ea, gl::Memory& m, const char* text, int64_t parent) {
        const int64_t k = key(m, text);
        ea.registerIntegrationTask(m, k, sp(text), parent);
        return k;
    }

    /// @brief Whether @p k is registered and marked shared.
    bool shared(const gl::Memory& m, int64_t k) {
        const Rec* r = m.lbMemory.integrationTasks.find(k);
        return r != nullptr && (r->flags & Rec::kShared) != 0;
    }

    /// @brief One existence entity with one element and a marked goal.
    void buildEntry(gl::WorkInstruction& wi, const char* signature, const char* goal) {
        const int32_t mark = wi.elemMark();
        wi.addElement(sp("(in[u_1,u_2])"));
        wi.commitEntity(sp("existence"), sp(signature), sp(""), 2, mark);
        wi.setMarkedGoal(sp(goal));
    }

    /// @brief File @p wi under @p pk of @p map through the production
    ///        read-modify-write.
    void file(gl::Memory& m, gl::TypedColdBlobMap<int64_t, gl::IntegrationEntry>& map,
              int64_t pk, const gl::WorkInstruction& wi, gl::ScratchArena& arena) {
        gl::ArenaIntegrationMap entries(&arena, &m.valueInterner);
        entries.buildFromCold(map, pk);
        entries.findOrEmplaceWork(wi, m.valueInterner);
        entries.writeToCold(map, pk);
    }
} // namespace

// The context register: a scope sets the task and restores the previous one.
TEST(integration_root, task_scope_sets_and_restores_context) {
    gl::Memory m;
    ASSERT_EQ(m.integrationTaskContext, int64_t{0});
    {
        gl::IntegrationTaskScope outer(m, 11);
        ASSERT_EQ(m.integrationTaskContext, int64_t{11});
        {
            gl::IntegrationTaskScope inner(m, 0);
            ASSERT_EQ(m.integrationTaskContext, int64_t{0});
        }
        ASSERT_EQ(m.integrationTaskContext, int64_t{11});
    }
    ASSERT_EQ(m.integrationTaskContext, int64_t{0});
}

// A task prepared without a context is a root, classified by its marker; a
// task prepared on behalf of another inherits the root and is never
// classified. A classified root stages a recheck of its scope.
TEST(integration_root, register_classifies_roots_and_inherits_children) {
    EA ea("Peano");
    gl::Memory m;
    const int64_t output = task(ea, m, "(in2[a,marker,b])", 0);
    const Rec* r = m.lbMemory.integrationTasks.find(output);
    ASSERT_TRUE(r != nullptr);
    ASSERT_TRUE((r->flags & Rec::kOutputRoot) != 0);
    ASSERT_EQ(r->outputSlot, 1);
    ASSERT_EQ(r->root, output);
    ASSERT_EQ(r->parent, int64_t{0});
    ASSERT_EQ(m.pendingRootRechecks.size(), 1);
    ASSERT_EQ(m.pendingRootRechecks[0], gl::NameMap::MAIN_ID);

    const int64_t exact = task(ea, m, "(in2[a,c,b])", 0);
    ASSERT_TRUE((m.lbMemory.integrationTasks.find(exact)->flags & Rec::kExactRoot) != 0);
    ASSERT_EQ(m.pendingRootRechecks.size(), 2);

    // A marker at an input slot, or a negated text, never finishes.
    const int64_t input = task(ea, m, "(in2[marker,c,b])", 0);
    ASSERT_EQ(m.lbMemory.integrationTasks.find(input)->flags, 0u);
    const int64_t negated = task(ea, m, "!(in2[a,marker,b])", 0);
    ASSERT_EQ(m.lbMemory.integrationTasks.find(negated)->flags, 0u);
    ASSERT_EQ(m.pendingRootRechecks.size(), 2);

    const int64_t child = key(m, "(in2[marker,d,e])");
    ASSERT_EQ(ea.registerIntegrationTask(m, child, sp("(in2[marker,d,e])"), output), output);
    const Rec* c = m.lbMemory.integrationTasks.find(child);
    ASSERT_EQ(c->parent, output);
    ASSERT_EQ(c->root, output);
    ASSERT_EQ(c->flags, 0u);
}

// The sharing rule: a requester of the same root changes nothing, another
// root or an independent requester marks the owning node, an unowned
// derivative stays unowned, and re-registration counts as a reach.
TEST(integration_root, reach_marks_other_roots_and_independent_requesters) {
    EA ea("Peano");
    gl::Memory m;
    const int64_t a = task(ea, m, "(in2[a,c,b])", 0);
    const int64_t a1 = task(ea, m, "(in2[marker,c,b])", a);
    const int64_t b = task(ea, m, "(in2[a,d,b])", 0);
    const int64_t c = task(ea, m, "(in2[a,e,b])", 0);
    const int64_t c1 = task(ea, m, "(in2[marker,e,b])", c);
    const int64_t d = task(ea, m, "(in2[a,f,b])", 0);
    const int64_t d1 = task(ea, m, "(in2[marker,f,b])", d);

    EA::noteIntegrationReach(m, a1, a);
    ASSERT_FALSE(shared(m, a1));
    EA::noteIntegrationReach(m, a1, b);
    ASSERT_TRUE(shared(m, a1));
    EA::noteIntegrationReach(m, c1, 0);
    ASSERT_TRUE(shared(m, c1));
    EA::noteIntegrationReach(m, 0, b);
    ASSERT_FALSE(shared(m, b));

    ASSERT_EQ(ea.registerIntegrationTask(m, d1, sp("(in2[marker,f,b])"), d), d);
    ASSERT_FALSE(shared(m, d1));
    ASSERT_EQ(ea.registerIntegrationTask(m, d1, sp("(in2[marker,f,b])"), b), d);
    ASSERT_TRUE(shared(m, d1));
}

// Owner maps: the first requester of a new key owns it, a later requester of
// another root reaches it, a key that existed before or an independent
// requester leaves it unowned.
TEST(integration_root, record_owner_first_owner_then_reach) {
    EA ea("Peano");
    gl::Memory m;
    const int64_t a = task(ea, m, "(in2[a,c,b])", 0);
    const int64_t a1 = task(ea, m, "(in2[marker,c,b])", a);
    const int64_t b = task(ea, m, "(in2[a,d,b])", 0);
    auto& owners = m.lbMemory.algebraKeyOwners;

    EA::recordIntegrationOwner(m, owners, 77, a1, false);
    ASSERT_EQ(owners.findOr(77, int64_t{0}), a1);
    EA::recordIntegrationOwner(m, owners, 77, a, true);
    ASSERT_FALSE(shared(m, a1));
    EA::recordIntegrationOwner(m, owners, 77, b, true);
    ASSERT_TRUE(shared(m, a1));
    ASSERT_EQ(owners.findOr(77, int64_t{0}), a1);

    EA::recordIntegrationOwner(m, owners, 78, a1, true);
    ASSERT_TRUE(owners.find(78) == nullptr);
    EA::recordIntegrationOwner(m, owners, 79, 0, false);
    ASSERT_TRUE(owners.find(79) == nullptr);
}

// A gate opened by a task is a node below it; a gate opened without a task
// stays independent.
TEST(integration_root, gate_node_below_its_task) {
    EA ea("Peano");
    gl::Memory m;
    const int64_t a = task(ea, m, "(in2[a,c,b])", 0);
    const int64_t gate = key(m, "(implication90[1,2])");
    ASSERT_EQ(ea.registerIntegrationGate(m, gate, 0), int64_t{0});
    ASSERT_TRUE(m.lbMemory.integrationTasks.find(gate) == nullptr);
    ASSERT_EQ(ea.registerIntegrationGate(m, gate, a), gate);
    const Rec* g = m.lbMemory.integrationTasks.find(gate);
    ASSERT_TRUE(g != nullptr);
    ASSERT_EQ(g->parent, a);
    ASSERT_EQ(g->root, a);
    ASSERT_EQ(g->flags, 0u);
}

// Removability walks the parent chain: a shared node strictly below the root
// protects itself and its subtree, the root's own mark is ignored, another
// root's node and an independent derivative are never removable.
TEST(integration_root, wipeable_walks_to_the_root_ignoring_its_mark) {
    EA ea("Peano");
    gl::Memory m;
    const int64_t r = task(ea, m, "(in2[a,c,b])", 0);
    const int64_t t = task(ea, m, "(in2[marker,c,b])", r);
    const int64_t n = key(m, "(implication90[1,2])");
    ea.registerIntegrationGate(m, n, t);
    const int64_t s = key(m, "(implication91[1,2])");
    ea.registerIntegrationGate(m, s, t);
    const int64_t x = key(m, "(implication92[1,2])");
    ea.registerIntegrationGate(m, x, s);
    const int64_t other = task(ea, m, "(in2[a,d,b])", 0);
    EA::noteIntegrationReach(m, s, other);

    ASSERT_TRUE(EA::integrationTaskWipeable(m, r, r));
    ASSERT_TRUE(EA::integrationTaskWipeable(m, t, r));
    ASSERT_TRUE(EA::integrationTaskWipeable(m, n, r));
    ASSERT_FALSE(EA::integrationTaskWipeable(m, s, r));
    ASSERT_FALSE(EA::integrationTaskWipeable(m, x, r));
    ASSERT_FALSE(EA::integrationTaskWipeable(m, n, other));
    ASSERT_FALSE(EA::integrationTaskWipeable(m, 0, r));

    EA::noteIntegrationReach(m, r, other);
    ASSERT_TRUE(shared(m, r));
    ASSERT_TRUE(EA::integrationTaskWipeable(m, n, r));
}

// The finish test: an exact root through a literal integrated row at the
// scope or an ancestor or through class representatives; an output root
// through any integrated value at its output slot; misses for rows another
// rule built (not integrated), descendant-only rows, other arguments and
// other operators (D-444).
TEST(integration_root, root_known_exact_and_output) {
    EA ea("Peano");
    gl::Memory m;
    const gl::NameId child = m.nameMap.encodePush(gl::NameMap::MAIN_ID, sp("(implication7[a])"));

    ASSERT_FALSE(ea.integrationRootKnownAtOrAbove(sp("(in2[a,c,b])"), -1, gl::NameMap::MAIN_ID, m));
    makeKnown(m, "(in2[a,c,b])", gl::NameMap::MAIN_ID);
    ASSERT_FALSE(ea.integrationRootKnownAtOrAbove(sp("(in2[a,c,b])"), -1, gl::NameMap::MAIN_ID, m));
    ASSERT_FALSE(ea.integrationRootKnownAtOrAbove(sp("(in2[a,marker,b])"), 1, gl::NameMap::MAIN_ID, m));
    makeIntegrated(m, "(in2[a,c,b])", gl::NameMap::MAIN_ID);
    ASSERT_TRUE(ea.integrationRootKnownAtOrAbove(sp("(in2[a,c,b])"), -1, gl::NameMap::MAIN_ID, m));
    ASSERT_TRUE(ea.integrationRootKnownAtOrAbove(sp("(in2[a,c,b])"), -1, child, m));
    makeIntegrated(m, "(in2[a,g,b])", child);
    ASSERT_FALSE(ea.integrationRootKnownAtOrAbove(sp("(in2[a,g,b])"), -1, gl::NameMap::MAIN_ID, m));

    ASSERT_FALSE(ea.integrationRootKnownAtOrAbove(sp("(in2[h,marker,b])"), 1, gl::NameMap::MAIN_ID, m));
    ASSERT_TRUE(ea.integrationRootKnownAtOrAbove(sp("(in2[a,marker,b])"), 1, gl::NameMap::MAIN_ID, m));
    ASSERT_TRUE(ea.integrationRootKnownAtOrAbove(sp("(in2[a,marker,b])"), 1, child, m));
    ASSERT_FALSE(ea.integrationRootKnownAtOrAbove(sp("(in3[a,marker,b,c])"), 1, gl::NameMap::MAIN_ID, m));

    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "p", "q" }, m.nameMap);
    m.assignClassesById(gl::NameMap::MAIN_ID, { cls });
    seedFixtureCanonicalClasses(m);
    makeKnown(m, "(in2[f,y,q])", gl::NameMap::MAIN_ID);
    ASSERT_FALSE(ea.integrationRootKnownAtOrAbove(sp("(in2[f,marker,p])"), 1, gl::NameMap::MAIN_ID, m));
    makeIntegrated(m, "(in2[f,y,q])", gl::NameMap::MAIN_ID);
    ASSERT_TRUE(ea.integrationRootKnownAtOrAbove(sp("(in2[f,marker,p])"), 1, gl::NameMap::MAIN_ID, m));
    ASSERT_TRUE(ea.integrationRootKnownAtOrAbove(sp("(in2[f,y,p])"), -1, gl::NameMap::MAIN_ID, m));
    ASSERT_FALSE(ea.integrationRootKnownAtOrAbove(sp("(in2[f,z,p])"), -1, gl::NameMap::MAIN_ID, m));
}

// The staging doors write only while the block has an integration task.
TEST(integration_root, stage_only_while_tasks_exist) {
    EA ea("Peano");
    gl::Memory m;
    ea.stageRootArrival(m, 5);
    ea.stageRootRecheck(m, gl::NameMap::MAIN_ID);
    ASSERT_EQ(m.pendingRootArrivals.size(), 0);
    ASSERT_EQ(m.pendingRootRechecks.size(), 0);

    task(ea, m, "(in2[marker,c,b])", 0);   // an unclassified root stages nothing itself
    ASSERT_EQ(m.pendingRootRechecks.size(), 0);
    ea.stageRootArrival(m, 5);
    ea.stageRootRecheck(m, gl::NameMap::MAIN_ID);
    ASSERT_EQ(m.pendingRootArrivals.size(), 1);
    ASSERT_EQ(m.pendingRootArrivals[0], int64_t{5});
    ASSERT_EQ(m.pendingRootRechecks.size(), 1);
}

// The sweep tests the roots an arrival or recheck reached, records each
// finished root once and consumes both queues.
TEST(integration_root, sweep_finishes_roots_on_arrival_and_recheck) {
    EA ea("Peano");
    gl::Memory m;
    const int64_t output = task(ea, m, "(in2[a,marker,b])", 0);
    const int64_t exact = task(ea, m, "(in2[c,d,e])", 0);
    ea.sweepFinishedIntegrationRoots(m);
    ASSERT_FALSE(m.lbMemory.finishedIntegrationRoots.contains(output));
    ASSERT_FALSE(m.lbMemory.finishedIntegrationRoots.contains(exact));
    ASSERT_EQ(m.pendingRootRechecks.size(), 0);

    // A row of another operator reaches no root.
    ea.stageRootArrival(m, makeKnown(m, "(in3[a,z,b,c])", gl::NameMap::MAIN_ID));
    ea.sweepFinishedIntegrationRoots(m);
    ASSERT_FALSE(m.lbMemory.finishedIntegrationRoots.contains(output));

    // A value another rule built does not finish the output root.
    ea.stageRootArrival(m, makeKnown(m, "(in2[a,z,b])", gl::NameMap::MAIN_ID));
    ea.sweepFinishedIntegrationRoots(m);
    ASSERT_FALSE(m.lbMemory.finishedIntegrationRoots.contains(output));

    ea.stageRootArrival(m, makeIntegrated(m, "(in2[a,z,b])", gl::NameMap::MAIN_ID));
    ea.sweepFinishedIntegrationRoots(m);
    ASSERT_TRUE(m.lbMemory.finishedIntegrationRoots.contains(output));
    ASSERT_TRUE(EA::integrationTaskRetired(m, output));
    ASSERT_FALSE(m.lbMemory.finishedIntegrationRoots.contains(exact));
    ASSERT_FALSE(EA::integrationTaskRetired(m, exact));
    ASSERT_EQ(m.pendingRootArrivals.size(), 0);

    // A known row without an arrival waits for a recheck of its scope.
    makeIntegrated(m, "(in2[c,d,e])", gl::NameMap::MAIN_ID);
    ea.sweepFinishedIntegrationRoots(m);
    ASSERT_FALSE(m.lbMemory.finishedIntegrationRoots.contains(exact));
    ea.stageRootRecheck(m, gl::NameMap::MAIN_ID);
    ea.sweepFinishedIntegrationRoots(m);
    ASSERT_TRUE(m.lbMemory.finishedIntegrationRoots.contains(exact));
    ASSERT_EQ(m.lbMemory.finishedIntegrationRoots.count(), 2);
}

// The selection follows the owners: removable nodes, top-level owned scopes,
// rules, copies, present algebra, set and trigger keys, and the entries and
// parks whose record names a removable task; a shared node protects its own
// derivatives.
TEST(integration_root, select_derivatives_follows_owners_and_shares) {
    EA ea("Peano");
    gl::Memory m;
    gl::ScratchArena& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
    gl::ScratchScope frame(arena);
    const int64_t r = task(ea, m, "(in2[a,c,b])", 0);
    const int64_t t = task(ea, m, "(in2[marker,c,b])", r);
    const int64_t g = key(m, "(implication90[1,2])");
    ea.registerIntegrationGate(m, g, t);
    const int64_t s = key(m, "(implication91[1,2])");
    ea.registerIntegrationGate(m, s, t);
    const int64_t other = task(ea, m, "(in2[a,d,b])", 0);
    EA::noteIntegrationReach(m, s, other);

    const gl::NameId s1 = m.nameMap.encodePush(gl::NameMap::MAIN_ID, sp("p1"));
    const gl::NameId s2 = m.nameMap.encodePush(s1, sp("p2"));
    const gl::NameId s3 = m.nameMap.encodePush(gl::NameMap::MAIN_ID, sp("p3"));
    const gl::NameId s4 = m.nameMap.encodePush(gl::NameMap::MAIN_ID, sp("p4"));
    m.lbMemory.integrationScopeOwners.insert(s1, g);
    m.lbMemory.integrationScopeOwners.insert(s2, g);
    m.lbMemory.integrationScopeOwners.insert(s3, s);
    m.lbMemory.integrationScopeOwners.insert(s4, other);
    m.lbMemory.integrationRuleOwners.insert(1001, g);
    m.lbMemory.integrationRuleOwners.insert(1002, 0);
    m.lbMemory.integrationRuleOwners.insert(1003, s);
    m.lbMemory.integrationCopyOwners.insert(2001, t);
    const int64_t setPresent = key(m, "(in2[k,marker,l])");
    const int64_t setGone = key(m, "(in2[k,marker,n])");
    m.overallHashMemory.admissionSetIntegration.mint(setPresent);
    m.lbMemory.admissionSetOwners.insert(setPresent, g);
    m.lbMemory.admissionSetOwners.insert(setGone, g);
    const int64_t trigger = key(m, "(in2[k,l,marker])");
    m.overallHashMemory.triggersForAdmissionSetIntegration.mint(trigger);
    m.lbMemory.admissionTriggerOwners.insert(trigger, t);
    const int64_t algebra = key(m, "(in3[6,marker,7,4])");
    gl::AdmissionMapValue admission;
    admission.standardMaxAdmissionDepth = 3;
    admission.standardMaxSecondaryNumber = 2;
    admission.key = { m.valueInterner.encode("(in3[8,9,10,11])") };
    gl::insertAdmissionValue(m.overallHashMemory.admissionMap, algebra, admission,
                             m.valueInterner, arena);
    m.lbMemory.algebraKeyOwners.insert(algebra, t);

    const int64_t entryKey = key(m, "(in2[u,marker,v])");
    gl::WorkInstruction own(&arena);
    buildEntry(own, "(existence90[u_1,u_2])", "(in2[u,marker,v])");
    own.ownerTask = g;
    file(m, m.overallHashMemory.admissionMapIntegration, entryKey, own, arena);
    gl::WorkInstruction protectedEntry(&arena);
    buildEntry(protectedEntry, "(existence91[u_1,u_2])", "(in2[u,marker,v])");
    protectedEntry.ownerTask = s;
    file(m, m.overallHashMemory.admissionMapIntegration, entryKey, protectedEntry, arena);
    const int64_t parkKey = key(m, "(in2[w,x,y])");
    gl::WorkInstruction parked(&arena);
    buildEntry(parked, "(existence92[u_1,u_2])", "(in2[w,marker,y])");
    parked.ownerTask = t;
    file(m, m.overallHashMemory.rejectedMapSubproof, parkKey, parked, arena);

    gl::DirtyState dirty = gl::DirtyState::Clean;
    EA::FinishedRootSelection selection(&arena, &dirty);
    EA::selectFinishedRootDerivatives(m, r, selection);

    ASSERT_EQ(selection.nodes.size(), 3);
    ASSERT_EQ(selection.sharedNodes, 1);
    ASSERT_EQ(selection.scopes.size(), 1);
    ASSERT_EQ(selection.scopes[0], s1);
    ASSERT_EQ(selection.rules.size(), 1);
    ASSERT_EQ(selection.rules[0], int64_t{1001});
    ASSERT_EQ(selection.copies.size(), 1);
    ASSERT_EQ(selection.setKeys.size(), 1);
    ASSERT_EQ(selection.setKeys[0], setPresent);
    ASSERT_EQ(selection.triggers.size(), 1);
    ASSERT_EQ(selection.algebraKeys.size(), 1);
    ASSERT_EQ(selection.entries, 1);
    ASSERT_EQ(selection.entryKeys.size(), 1);
    ASSERT_EQ(selection.entryKeys[0], entryKey);
    ASSERT_EQ(selection.parks, 1);
    ASSERT_EQ(selection.parkKeys.size(), 1);

    // Another root selects nothing of this root's derivatives.
    EA::FinishedRootSelection none(&arena, &dirty);
    EA::selectFinishedRootDerivatives(m, other, none);
    ASSERT_EQ(none.nodes.size(), 1);
    ASSERT_EQ(none.scopes.size(), 1);
    ASSERT_EQ(none.scopes[0], s4);
    ASSERT_EQ(none.rules.size(), 0);
    ASSERT_EQ(none.entries, 0);
}

// The marker-firing drains run on behalf of the record's task: a replayed
// preparation registers its task below that task (a root without one), and a
// replayed algebra permission is owned by it.
TEST(integration_root, drains_carry_the_record_task) {
    EA ea("Peano");
    gl::Memory m;
    gl::SealedPageSet pages;
    pages.bind(&gl::staticMemory());
    const auto s = [&pages](const std::string& v) {
        return gl::SealedString::copyFrom(pages, v.data(), static_cast<int32_t>(v.size()));
    };
    const int64_t owner = task(ea, m, "(in2[a,c,b])", 0);
    const gl::SealedString argsArr[3] = { s("4"), s("a"), s("b") };
    const auto args = gl::SealedSpan<gl::SealedString>::copyFrom(pages, argsArr, 3);
    gl::DeferredIntegrationPrep owned{ s("(in3[a,marker,b,4])"), args, s("main") };
    owned.ownerTask = owner;
    m.deferredIntegrationPreps.push_back(owned);
    const gl::SealedString argsArr2[3] = { s("4"), s("c"), s("d") };
    const auto args2 = gl::SealedSpan<gl::SealedString>::copyFrom(pages, argsArr2, 3);
    m.deferredIntegrationPreps.push_back(
        gl::DeferredIntegrationPrep{ s("(in3[c,marker,d,4])"), args2, s("main") });
    ea.drainDeferredIntegrationPreps(m);
    ASSERT_EQ(m.integrationTaskContext, int64_t{0});
    const Rec* child = m.lbMemory.integrationTasks.find(key(m, "(in3[a,marker,b,4])"));
    ASSERT_TRUE(child != nullptr);
    ASSERT_EQ(child->parent, owner);
    ASSERT_EQ(child->root, owner);
    const int64_t rootKey = key(m, "(in3[c,marker,d,4])");
    const Rec* root = m.lbMemory.integrationTasks.find(rootKey);
    ASSERT_TRUE(root != nullptr);
    ASSERT_EQ(root->root, rootKey);

    gl::StagedAdmissionValue value;
    const gl::SealedString valueKeyArr[2] = { s("10"), s("3") };
    value.key = gl::SealedSpan<gl::SealedString>::copyFrom(pages, valueKeyArr, 2);
    gl::AdmissionKeyAlgebraRecord algebra{
        gl::SealedExpressionWithValidity{ s("(in2[10,marker,3])"), s("main") }, value };
    algebra.ownerTask = owner;
    m.admissionKeysAlgebra.push_back(algebra);
    ea.drainAdmissionKeysAlgebra(m);
    ASSERT_EQ(m.lbMemory.algebraKeyOwners.findOr(key(m, "(in2[10,marker,3])"), int64_t{0}), owner);

    m.deferredIntegrationPreps.clear();
    m.admissionKeysAlgebra.clear();
    pages.seal();
    pages.freePages();
}

// A scope wipe removes the ownership records at the closed scopes only.
TEST(integration_root, scope_wipe_erases_closed_ownership_only) {
    EA ea("Peano");
    gl::Memory m;
    const gl::NameId childId = m.nameMap.encodePush(gl::NameMap::MAIN_ID, sp("(implication7[a])"));
    const std::string childName = m.nameMap.decode(childId);
    const int64_t mainRoot = task(ea, m, "(in2[a,c,b])", 0);
    const int64_t childRoot = key(m, "(in2[a,c,b])", childName.c_str());
    ea.registerIntegrationTask(m, childRoot, sp("(in2[a,c,b])"), 0);
    const gl::NameId inner = m.nameMap.encodePush(childId, sp("(implication8[a])"));
    m.lbMemory.integrationScopeOwners.insert(inner, childRoot);
    m.lbMemory.integrationRuleOwners.insert(gl::packRuleOwner(9, childId), childRoot);
    m.lbMemory.integrationRuleOwners.insert(gl::packRuleOwner(9, gl::NameMap::MAIN_ID), mainRoot);
    m.lbMemory.finishedIntegrationRoots.mint(childRoot);

    m.wipeSubtrees(&childId, 1);

    ASSERT_TRUE(m.lbMemory.integrationTasks.find(childRoot) == nullptr);
    ASSERT_TRUE(m.lbMemory.integrationTasks.find(mainRoot) != nullptr);
    ASSERT_TRUE(m.lbMemory.integrationScopeOwners.find(inner) == nullptr);
    ASSERT_TRUE(m.lbMemory.integrationRuleOwners.find(gl::packRuleOwner(9, childId)) == nullptr);
    ASSERT_TRUE(m.lbMemory.integrationRuleOwners.find(gl::packRuleOwner(9, gl::NameMap::MAIN_ID)) != nullptr);
    ASSERT_FALSE(m.lbMemory.finishedIntegrationRoots.contains(childRoot));
}

namespace {
    /// @brief Mark @p k retired, as the wipe does.
    void retire(gl::Memory& m, int64_t k) {
        const int32_t id = m.lbMemory.integrationTasks.lookup(k);
        Rec rec = m.lbMemory.integrationTasks.valueAt(id);
        rec.flags |= Rec::kRetired;
        m.lbMemory.integrationTasks.setValueAt(id, rec);
    }
} // namespace

// A retired key registered again — its gate erased by a disproof — is a new
// node below the new requester; a retired root prepared again without a task
// is a root again, classified, out of the finished set and rechecked.
TEST(integration_root, retired_key_registered_again_is_a_new_node) {
    EA ea("Peano");
    gl::Memory m;
    const int64_t r = task(ea, m, "(in2[a,c,b])", 0);
    const int64_t c = task(ea, m, "(in2[marker,c,b])", r);
    const int64_t live = task(ea, m, "(in2[a,d,b])", 0);
    ASSERT_FALSE(EA::integrationTaskRetired(m, 0));
    ASSERT_FALSE(EA::integrationTaskRetired(m, c));
    retire(m, r);
    retire(m, c);
    m.lbMemory.finishedIntegrationRoots.mint(r);
    ASSERT_TRUE(EA::integrationTaskRetired(m, r));
    ASSERT_TRUE(EA::integrationTaskRetired(m, c));

    ASSERT_EQ(ea.registerIntegrationTask(m, c, sp("(in2[marker,c,b])"), live), live);
    const Rec* revived = m.lbMemory.integrationTasks.find(c);
    ASSERT_EQ(revived->parent, live);
    ASSERT_EQ(revived->root, live);
    ASSERT_EQ(revived->flags, 0u);

    const int32_t rechecks = m.pendingRootRechecks.size();
    ASSERT_EQ(ea.registerIntegrationTask(m, r, sp("(in2[a,c,b])"), 0), r);
    const Rec* root = m.lbMemory.integrationTasks.find(r);
    ASSERT_EQ(root->parent, int64_t{0});
    ASSERT_EQ(root->flags, Rec::kExactRoot);
    ASSERT_FALSE(m.lbMemory.finishedIntegrationRoots.contains(r));
    ASSERT_EQ(m.pendingRootRechecks.size(), rechecks + 1);
}

// The wipe removes what the selection names: the removable nodes retire, the
// retired records leave their entry and park keys (an emptied key goes), the
// algebra, admission-set and trigger keys go with their owners, the top-level
// scope is queued, and no owner record names a retired node; a shared node
// keeps its derivatives.
TEST(integration_root, wipe_removes_the_selected_derivatives) {
    EA ea("Peano");
    gl::Memory m;
    gl::ScratchArena& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
    gl::ScratchScope frame(arena);
    const int64_t r = task(ea, m, "(in2[a,c,b])", 0);
    const int64_t t = task(ea, m, "(in2[marker,c,b])", r);
    const int64_t g = key(m, "(implication90[1,2])");
    ea.registerIntegrationGate(m, g, t);
    const int64_t s = key(m, "(implication91[1,2])");
    ea.registerIntegrationGate(m, s, t);
    const int64_t other = task(ea, m, "(in2[a,d,b])", 0);
    EA::noteIntegrationReach(m, s, other);

    const gl::NameId s1 = m.nameMap.encodePush(gl::NameMap::MAIN_ID, sp("p1"));
    const gl::NameId s2 = m.nameMap.encodePush(s1, sp("p2"));
    const gl::NameId s3 = m.nameMap.encodePush(gl::NameMap::MAIN_ID, sp("p3"));
    m.lbMemory.integrationScopeOwners.insert(s1, g);
    m.lbMemory.integrationScopeOwners.insert(s2, g);
    m.lbMemory.integrationScopeOwners.insert(s3, s);
    m.lbMemory.integrationCopyOwners.insert(2001, t);
    const int64_t setKey = key(m, "(in2[k,marker,l])");
    m.overallHashMemory.admissionSetIntegration.mint(setKey);
    m.localHashMemory.admissionSetIntegration.mint(setKey);
    m.lbMemory.admissionSetOwners.insert(setKey, g);
    const int64_t setGone = key(m, "(in2[k,marker,n])");
    m.lbMemory.admissionSetOwners.insert(setGone, g);
    const int64_t trigger = key(m, "(in2[k,l,marker])");
    m.overallHashMemory.triggersForAdmissionSetIntegration.mint(trigger);
    m.lbMemory.admissionTriggerOwners.insert(trigger, t);
    const int64_t algebra = key(m, "(in3[6,marker,7,4])");
    gl::AdmissionMapValue admission;
    admission.standardMaxAdmissionDepth = 3;
    admission.standardMaxSecondaryNumber = 2;
    admission.key = { m.valueInterner.encode("(in3[8,9,10,11])") };
    gl::insertAdmissionValue(m.overallHashMemory.admissionMap, algebra, admission,
                             m.valueInterner, arena);
    m.overallHashMemory.admissionStatusMap.upsert(algebra, uint8_t(0));
    m.lbMemory.algebraKeyOwners.insert(algebra, t);

    const int64_t entryKey = key(m, "(in2[u,marker,v])");
    gl::WorkInstruction own(&arena);
    buildEntry(own, "(existence90[u_1,u_2])", "(in2[u,marker,v])");
    own.ownerTask = g;
    file(m, m.overallHashMemory.admissionMapIntegration, entryKey, own, arena);
    gl::WorkInstruction kept(&arena);
    buildEntry(kept, "(existence91[u_1,u_2])", "(in2[u,marker,v])");
    kept.ownerTask = s;
    file(m, m.overallHashMemory.admissionMapIntegration, entryKey, kept, arena);
    const int64_t parkKey = key(m, "(in2[w,x,y])");
    gl::WorkInstruction parked(&arena);
    buildEntry(parked, "(existence92[u_1,u_2])", "(in2[w,marker,y])");
    parked.ownerTask = t;
    file(m, m.overallHashMemory.rejectedMapSubproof, parkKey, parked, arena);

    gl::DirtyState dirty = gl::DirtyState::Clean;
    EA::FinishedRootSelection selection(&arena, &dirty);
    EA::selectFinishedRootDerivatives(m, r, selection);
    ea.wipeFinishedRootDerivatives(m, r, selection);

    ASSERT_TRUE(EA::integrationTaskRetired(m, r));
    ASSERT_TRUE(EA::integrationTaskRetired(m, t));
    ASSERT_TRUE(EA::integrationTaskRetired(m, g));
    ASSERT_FALSE(EA::integrationTaskRetired(m, s));
    ASSERT_FALSE(EA::integrationTaskRetired(m, other));

    const gl::IntegrationEntryMap entries = gl::admissionIntegrationRecordsAt(
        m.overallHashMemory.admissionMapIntegration, entryKey, m.valueInterner);
    ASSERT_EQ(entries.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(entries.begin()->first.ownerTask, s);
    ASSERT_TRUE(m.overallHashMemory.rejectedMapSubproof.lookup(parkKey) == 0);
    ASSERT_TRUE(m.overallHashMemory.admissionMap.lookup(algebra) == 0);
    ASSERT_TRUE(m.overallHashMemory.admissionStatusMap.find(algebra) == nullptr);
    ASSERT_FALSE(m.overallHashMemory.admissionSetIntegration.contains(setKey));
    ASSERT_FALSE(m.localHashMemory.admissionSetIntegration.contains(setKey));
    ASSERT_FALSE(m.overallHashMemory.triggersForAdmissionSetIntegration.contains(trigger));
    ASSERT_TRUE(m.pendingWipeScopes.contains(s1));
    ASSERT_FALSE(m.pendingWipeScopes.contains(s3));

    ASSERT_TRUE(m.lbMemory.integrationScopeOwners.find(s1) == nullptr);
    ASSERT_TRUE(m.lbMemory.integrationScopeOwners.find(s2) == nullptr);
    ASSERT_EQ(m.lbMemory.integrationScopeOwners.findOr(s3, int64_t{0}), s);
    ASSERT_TRUE(m.lbMemory.integrationCopyOwners.find(2001) == nullptr);
    ASSERT_TRUE(m.lbMemory.admissionSetOwners.find(setKey) == nullptr);
    ASSERT_TRUE(m.lbMemory.admissionSetOwners.find(setGone) == nullptr);
    ASSERT_TRUE(m.lbMemory.admissionTriggerOwners.find(trigger) == nullptr);
    ASSERT_TRUE(m.lbMemory.algebraKeyOwners.find(algebra) == nullptr);
}

// A rule a node installs the way the integration Case B installs it is
// removed exactly from the overall, local and delta hash memories by the wipe
// of its finished root: its rule values leave every encoded map and its owner
// records go.
TEST(integration_root, wipe_removes_an_installed_integration_rule_exactly) {
    EA ea("Peano");
    ea.compiledExpressions["existence90"] = gl::LogicalEntity("existence",
        { "(in2[u_1,x,u_2])" }, "(existence90[u_1,u_2])", 2);
    gl::Memory m;
    gl::ScratchArena& strings = gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);
    gl::ScratchScope stringFrame(strings);
    const int64_t r = task(ea, m, "(in2[a,c,b])", 0);
    const int64_t g = key(m, "(existence90[1,2])");
    ea.registerIntegrationGate(m, g, r);
    const gl::StrSpan chain[1] = { sp("(in2[u_1,x,u_2])") };
    const gl::StrSpan head = sp("(existence90[u_1,u_2])");
    const gl::StrSpan rem[2] = { sp("1"), sp("2") };
    gl::ScratchString history, hash;
    ea.buildIntegrationInstructionScratch(strings, chain, 1, head, history, hash);
    gl::HashMemory* const targets[3] = {
        &m.overallHashMemory, &m.localHashMemory, &m.localHashMemoryDelta };
    {
        gl::IntegrationTaskScope scope(m, g);
        for (gl::HashMemory* target : targets)
            ea.addToHashMemory(chain, 1, head, rem, 2, m, *target, nullptr, 0,
                gl::StrSpan(hash), 0, 0, false, 0, sp("integration"), false,
                gl::StrSpan(history), sp("main"));
    }
    const gl::RuleOwner rule = gl::packRuleOwner(
        m.ruleInterner.lookup(gl::StrSpan(hash)), gl::NameMap::MAIN_ID);
    ASSERT_EQ(m.lbMemory.integrationRuleOwners.findOr(rule, int64_t{0}), g);
    for (gl::HashMemory* target : targets)
        ASSERT_TRUE(target->encodedMap.inner().blobCount() > 0);

    gl::ScratchArena& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
    gl::ScratchScope frame(arena);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    EA::FinishedRootSelection selection(&arena, &dirty);
    EA::selectFinishedRootDerivatives(m, r, selection);
    ASSERT_EQ(selection.rules.size(), 1);
    ASSERT_TRUE(selection.copies.size() > 0);
    ea.wipeFinishedRootDerivatives(m, r, selection);

    for (gl::HashMemory* target : targets)
        ASSERT_EQ(target->encodedMap.inner().blobCount(), 0);
    ASSERT_TRUE(m.lbMemory.integrationRuleOwners.find(rule) == nullptr);
    for (int32_t id = 1; id <= m.lbMemory.integrationCopyOwners.count(); ++id) {
        const int64_t copy = m.lbMemory.integrationCopyOwners.keyAt(id);
        if (m.lbMemory.integrationCopyOwners.lookup(copy) == 0) continue;
        ASSERT_TRUE(m.lbMemory.integrationCopyOwners.valueAt(id) != g);
    }
    ASSERT_TRUE(EA::integrationTaskRetired(m, g));
}

// The removal replays the integration install, not the install door: a
// four-premise rule whose compound slot `and91` binds four arguments gets an
// ordis2 demand variant on the door route (four premises suffice) but not on
// the integration route (six are needed). The wipe leaves every index of
// every instance empty.
TEST(integration_root, wipe_replays_the_integration_install_of_a_rule) {
    EA ea("Peano");
    ea.compiledExpressions["and91"] = gl::LogicalEntity("and",
        { "(in2[u_1,u_2,u_3])", "(in2[u_2,u_3,u_4])" }, "(and91[u_1,u_2,u_3,u_4])", 4);
    ea.compiledExpressions["existence92"] = gl::LogicalEntity("existence",
        { "(in2[u_1,x,u_4])" }, "(existence92[u_1,u_4])", 2);
    gl::ScratchArena& strings = gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);
    gl::ScratchScope stringFrame(strings);
    const gl::StrSpan chainIn[4] = { sp("(and91[u_1,u_2,u_3,u_4])"), sp("(in2[u_1,u_2,u_3])"),
                                     sp("(in2[u_2,u_3,u_4])"), sp("(in2[u_1,u_3,u_4])") };
    gl::ScratchString history, hash;
    ea.buildIntegrationInstructionScratch(strings, chainIn, 4, sp("(existence92[u_1,u_4])"),
        history, hash);
    gl::StrSpan chain[gl::ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
    int32_t chainN = 0;
    gl::StrSpan head;
    gl::StrSpan rem[gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS];
    int32_t remN = 0;
    ea.decomposeImplicationForInstall(gl::StrSpan(hash), strings, chain, chainN, head, rem, remN);
    ASSERT_EQ(chainN, 4);
    const auto indexSize = [](const gl::HashMemory& hm) {
        return hm.encodedMap.inner().blobCount() + hm.normalizedEncodedKeys.count()
            + hm.remainingArgsOwners.count() + hm.originals.count();
    };

    // Control: the door route installs more index keys for the same rule.
    gl::Memory door;
    ea.addToHashMemory(chain, chainN, head, rem, remN, door, door.localHashMemory, nullptr, 0,
        gl::StrSpan(hash), ea.parameters.maxIterationNumberVariable,
        ea.parameters.standardMaxSecondaryNumber, false, ea.parameters.minNumOperatorsKey,
        sp("implication"), false, gl::StrSpan(history), sp("main"));

    gl::Memory m;
    const int64_t r = task(ea, m, "(in2[a,c,b])", 0);
    const int64_t g = key(m, "(existence92[1,4])");
    ea.registerIntegrationGate(m, g, r);
    gl::HashMemory* const targets[3] = {
        &m.overallHashMemory, &m.localHashMemory, &m.localHashMemoryDelta };
    {
        gl::IntegrationTaskScope scope(m, g);
        for (gl::HashMemory* target : targets)
            ea.addToHashMemory(chain, chainN, head, rem, remN, m, *target, nullptr, 0,
                gl::StrSpan(hash), 0, 0, false, 0, sp("integration"), false,
                gl::StrSpan(history), sp("main"));
    }
    ASSERT_TRUE(indexSize(door.localHashMemory) > indexSize(m.localHashMemory));

    gl::ScratchArena& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
    gl::ScratchScope frame(arena);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    EA::FinishedRootSelection selection(&arena, &dirty);
    EA::selectFinishedRootDerivatives(m, r, selection);
    ASSERT_EQ(selection.rules.size(), 1);
    ea.wipeFinishedRootDerivatives(m, r, selection);
    for (gl::HashMemory* target : targets) {
        ASSERT_EQ(indexSize(*target), 0);
        ASSERT_EQ(target->remainingArgsNormalizedEncodedMap.count(), 0);
    }
}
