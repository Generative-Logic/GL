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
/// @brief Byte-identity unit tests for
///        `gl::ExpressionAnalyzer::reconstructImplicationFullBind` after its T3
///        statification (interior heap -> stack `StrSpan` arrays).
///
/// @details
/// The rewrite (`I-129`) replaced the function's interior
/// `std::vector` / `std::set<std::string>` scratch with fixed stack arrays of
/// `StrSpan` filled by `getArgsSpans`, sorted-array + `binary_search` membership,
/// and a direct return-string build. Output bytes must be unchanged. These cases
/// pin representative implication chains: empty key (pass-through), `u_` filtering,
/// and multi-premise nesting with the left-most-occurrence binder placement.

#include "test_harness.hpp"

#include "../prover.hpp"

TEST(reconstruct_implication, full_bind_byte_identity) {
    gl::ExpressionAnalyzer ea("Peano");

    // Empty key: nothing to bind, the value passes through unchanged.
    ASSERT_TRUE(ea.reconstructImplicationFullBind({}, "(p[x])")
                == std::string("(p[x])"));

    // Single premise: only non-`u_` args bind, so `x` binds and `u_y` is skipped.
    ASSERT_TRUE(ea.reconstructImplicationFullBind({ "(a[x,u_y])" }, "(b[x])")
                == std::string("(>[x](a[x,u_y])(b[x]))"));

    // Two premises: each bound var is placed at its left-most occurrence (`x` at
    // the first premise, `y` at the second); the chain nests right-to-left.
    ASSERT_TRUE(ea.reconstructImplicationFullBind({ "(a[x])", "(b[y])" }, "(c[x,y])")
                == std::string("(>[x](a[x])(>[y](b[y])(c[x,y])))"));

    // The thin forwarder `reconstructImplication` is the same single-binder rule,
    // so it produces the identical string.
    ASSERT_TRUE(ea.reconstructImplication({ "(a[x])", "(b[y])" }, "(c[x,y])")
                == std::string("(>[x](a[x])(>[y](b[y])(c[x,y])))"));
}

// Byte-identity of the StrSpan overloads added for the absorb-door origin deposits:
// ValueInterner::encode(StrSpan) and mintOriginKey(StrSpan, StrSpan) must produce the
// same id / packed key as their std::string forms (they intern the same bytes).
TEST(absorb_door_span_overloads, origin_mint_byte_identical) {
    gl::ExpressionAnalyzer ea("Peano");  // inits the static memory pool
    gl::Memory m;

    const std::string s = "(in3[i0,i1,v1,+])";
    ASSERT_TRUE(m.valueInterner.encode(gl::StrSpan(s)) == m.valueInterner.encode(s));
    const std::string t = "(=[a,b])";
    ASSERT_TRUE(m.valueInterner.encode(t) == m.valueInterner.encode(gl::StrSpan(t)));

    const std::string e = "(in2[i0,v1,s])";
    const std::string v = "main";
    ASSERT_TRUE(gl::mintOriginKey(m.originInterner, e, v)
                == gl::mintOriginKey(m.originInterner, gl::StrSpan(e), gl::StrSpan(v)));
}

// prefixArgumentsWithU statified: every argument is `u_`-prefixed (new bytes
// built on the per-slot scratch arena), all other bytes pass through. The
// reference outputs are exactly what the former heap vector<string>+map version
// produced. The test thread leaves g_currentCoreId at -1, so the reserved
// scratch slot is exercised.
TEST(absorb_door_span_overloads, prefix_arguments_with_u) {
    gl::ExpressionAnalyzer ea("Peano");  // inits the scratch arenas (logicalCores+1)

    // No args: plain pass-through.
    ASSERT_TRUE(ea.prefixArgumentsWithU("(zero[])") == std::string("(zero[])"));
    // Every arg prefixed.
    ASSERT_TRUE(ea.prefixArgumentsWithU("(in3[a,b,c])")
                == std::string("(in3[u_a,u_b,u_c])"));
    ASSERT_TRUE(ea.prefixArgumentsWithU("(=[x,y])")
                == std::string("(=[u_x,u_y])"));
    // Already-`u_` args are prefixed again (the original prefixes every arg).
    ASSERT_TRUE(ea.prefixArgumentsWithU("(p[u_x])")
                == std::string("(p[u_u_x])"));
    // Repeated arg: the source is scanned token-by-token, each replaced once.
    ASSERT_TRUE(ea.prefixArgumentsWithU("(=[x,x])")
                == std::string("(=[u_x,u_x])"));
}

// Test-local heap oracle for the retired std::set<std::string> body of
// listLastRemovedArgsLE -- the byte-identity reference for the span-out-param form.
static std::vector<std::string> listLastRemovedArgsLEOracle(const gl::LogicalEntity& le) {
    std::set<std::string> uniqueArgs;
    for (const std::string& element : le.elements) {
        gl::StrSpan args[gl::ExecutionParameters::MAX_ARITY];
        const int32_t argsN = gl::getArgsSpans(gl::StrSpan(element), args,
                                               gl::ExecutionParameters::MAX_ARITY);
        for (int32_t a = 0; a < argsN; ++a) {
            if (!(args[a].len >= 2 && args[a].ptr[0] == 'u' && args[a].ptr[1] == '_')) {
                uniqueArgs.insert(args[a].toStdString());
            }
        }
    }
    return std::vector<std::string>(uniqueArgs.begin(), uniqueArgs.end());
}

// listLastRemovedArgsLE statified: the former sorted std::vector<std::string>
// return (from a std::set) became a sorted-unique StrSpan out-param + count. The
// out spans alias le.elements; the sort/dedup reproduce the std::set order.
TEST(prover_span_twins, list_last_removed_args_le_matches_heap) {
    gl::ExpressionAnalyzer ea("Peano");

    // duplicate arg across elements (a, b) + u_-prefixed args (skipped) + distinct.
    gl::LogicalEntity le;
    le.category = "existence";
    le.elements = { "(P[a,u_x,b])", "(Q[b,a])", "(R[u_y,c])" };

    const std::vector<std::string> oracle = listLastRemovedArgsLEOracle(le);
    gl::StrSpan out[gl::ExecutionParameters::MAX_KEY_SLOTS];
    const int32_t n = ea.listLastRemovedArgsLE(
        le, out, gl::ExecutionParameters::MAX_KEY_SLOTS);

    ASSERT_EQ(static_cast<std::size_t>(n), oracle.size());
    for (int32_t i = 0; i < n; ++i)
        ASSERT_EQ(gl::StrSpan(out[i]).toStdString(), oracle[i]);

    // Span-run entry point directly (the LE form forwards to it).
    gl::StrSpan elemRun[gl::ExecutionParameters::MAX_INSTRUCTION_ELEMENTS + 1];
    const int32_t elemN = static_cast<int32_t>(le.elements.size());
    for (int32_t i = 0; i < elemN; ++i) elemRun[i] = gl::StrSpan(le.elements[i]);
    gl::StrSpan outSpan[gl::ExecutionParameters::MAX_KEY_SLOTS];
    const int32_t nSpan = ea.listLastRemovedArgsLE(
        gl::StrSpan("existence", 9), elemRun, elemN, outSpan,
        gl::ExecutionParameters::MAX_KEY_SLOTS);
    ASSERT_EQ(nSpan, n);
    for (int32_t i = 0; i < nSpan; ++i)
        ASSERT_EQ(gl::StrSpan(outSpan[i]).toStdString(), oracle[i]);
}

// getGlobalKey statified: the former std::vector<std::string> return (walk to
// the root sentinel, deepest-first, then std::reverse) became a caller-fill
// StrSpan out-param + count. The out spans alias the skeletonInterner (via
// exprKeyView); the reverse is done in place on the span array. Byte-identical
// to the retained vector oracle across a multi-level chain, a single-key chain,
// an intermediate LB with an empty exprKey (skipped, like the vector form), and
// the root sentinel alone (empty key).
TEST(prover_span_twins, get_global_key_span_matches_vector) {
    gl::ExpressionAnalyzer ea("Peano");  // inits the static memory pool

    gl::Memory root;                       // root sentinel: empty exprKey
    gl::Memory child;
    child.setExprKey("(=[a,b])");
    child.parentMemory = &root;
    gl::Memory middle;                     // empty exprKey -> skipped by both
    middle.parentMemory = &child;
    gl::Memory grand;
    grand.setExprKey("(in2[x,N])");
    grand.parentMemory = &middle;

    const gl::Memory* nodes[] = { &grand, &child, &root };
    for (const gl::Memory* mb : nodes) {
        const std::vector<std::string> oracle = ea.getGlobalKey(*mb);
        gl::StrSpan out[64];
        const int32_t n = ea.getGlobalKey(*mb, out, 64);
        ASSERT_EQ(static_cast<std::size_t>(n), oracle.size());
        for (int32_t i = 0; i < n; ++i)
            ASSERT_EQ(gl::StrSpan(out[i]).toStdString(), oracle[i]);
    }
}

// dischargeToBeProved Part B: the proven-direct theorems ride a shared
// SealedPageSet as UpdateGlobalDirectRec records. The drain's index-sort on
// updateGlobalDirectLess orders by (theorem bytes, level verdict TRUE first,
// producer chain) — every key a pure function of proof state. The dispatch
// coreId is deliberately ABSENT from the order (a scheduling race outcome:
// ordering same-theorem records by it let the worker race pick the
// registration method), so the coreIds here are adversarial noise — within
// each theorem the true-verdict record carries the LARGER coreId and must
// still drain first. Appended scrambled; all (theorem, verdict) pairs
// distinct so the expected order is total without producers.
TEST(prover_span_twins, update_global_direct_sealed_drain_matches_sorted) {
    gl::GlobalMemoryManager m;
    m.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::SealedPageSet ps;
    ps.bind(&m);

    struct Row { std::string thm; int coreId; bool verdict; };
    const std::vector<Row> input = {
        { "(=[b,a])", 2, false }, { "(=[a,b])", 9, true },
        { "(zero[])", 3, true },  { "(=[b,a])", 7, true },
        { "(=[a,b])", 0, false },
    };
    for (const auto& r : input) {
        ps.appendRecord(gl::ExpressionAnalyzer::UpdateGlobalDirectRec{
            gl::SealedString::copyFrom(ps, r.thm.data(),
                static_cast<int32_t>(r.thm.size())),
            r.coreId, nullptr, r.verdict });
    }

    // Gather + index-sort via updateGlobalDirectLess (the drain's replay order).
    std::vector<const gl::ExpressionAnalyzer::UpdateGlobalDirectRec*> refs;
    ps.forEachRecord<gl::ExpressionAnalyzer::UpdateGlobalDirectRec>(
        [&](const gl::ExpressionAnalyzer::UpdateGlobalDirectRec& r) {
            refs.push_back(&r);
        });
    std::vector<int> idx(refs.size());
    for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return gl::ExpressionAnalyzer::updateGlobalDirectLess(
                   *refs[static_cast<std::size_t>(a)],
                   *refs[static_cast<std::size_t>(b)]) < 0;
    });

    // Expected: theorem bytes ascending; within a theorem, verdict true
    // first — the record's coreId never consulted.
    const std::vector<std::pair<std::string, bool>> oracle = {
        { "(=[a,b])", true }, { "(=[a,b])", false },
        { "(=[b,a])", true }, { "(=[b,a])", false },
        { "(zero[])", true },
    };

    ASSERT_EQ(idx.size(), oracle.size());
    for (std::size_t i = 0; i < idx.size(); ++i) {
        const auto& r = *refs[static_cast<std::size_t>(idx[i])];
        ASSERT_EQ(gl::StrSpan(r.theorem).toStdString(), oracle[i].first);
        ASSERT_EQ(r.allLevelsInvolved, oracle[i].second);
    }
    ps.seal();
    ps.freePages();
}

// Verbatim retained heap oracle for the former std::string-returning expandSignature
// (moved to tests -- a prover-only heap form is not kept in production). Reproduces
// the AND / existence / implication / OR build + u_/it_/int_/c_ renaming byte-for-byte;
// the implication case delegates to the surviving member
// reconstructImplicationFullBindScratch. The single-threaded test thread leaves
// g_currentCoreId at -1, so the scratch slot is the reserved slotCount()-1.
static std::string expandSignatureOracle(gl::ExpressionAnalyzer& ea,
                                         const gl::LogicalEntity& le) {
    const std::string& category = le.category;
    const std::string& signature = le.signature;
    const std::vector<std::string>& elements = le.elements;

    gl::StrSpan sigArgsVec[gl::ExecutionParameters::MAX_ARITY];
    const int32_t sigArgsN = gl::getArgsSpans(gl::StrSpan(signature), sigArgsVec,
                                              gl::ExecutionParameters::MAX_ARITY);
    std::set<std::string> sigArgs;
    for (int32_t i = 0; i < sigArgsN; ++i) sigArgs.insert(sigArgsVec[i].toStdString());
    for (const auto& arg : sigArgs) {
        bool startsWithU = (arg.size() >= 2 && arg[0] == 'u' && arg[1] == '_');
        assert(startsWithU && "Signature arguments must start with 'u_' as per assumption.");
    }

    std::string result;
    if (category == "and") {
        if (elements.empty()) result = signature;
        else if (elements.size() == 1) result = elements[0];
        else {
            std::string current = elements[0];
            for (size_t i = 1; i < elements.size(); ++i)
                current = "(&" + current + elements[i] + ")";
            result = current;
        }
    }
    else if (category == "existence") {
        if (elements.size() < 2) result = signature;
        else {
            std::string body = elements[0];
            std::string head = elements[1];
            std::string negatedHead;
            if (!head.empty() && head[0] == '!') negatedHead = head.substr(1);
            else negatedHead = "!" + head;
            gl::StrSpan bodyArgs[gl::ExecutionParameters::MAX_ARITY];
            const int32_t bodyArgsN = gl::getArgsSpans(gl::StrSpan(body), bodyArgs,
                                                       gl::ExecutionParameters::MAX_ARITY);
            std::set<std::string> boundVars;
            for (int32_t i = 0; i < bodyArgsN; ++i) {
                std::string arg = bodyArgs[i].toStdString();
                if (sigArgs.find(arg) == sigArgs.end()) boundVars.insert(arg);
            }
            std::string varsStr;
            for (const auto& v : boundVars) {
                if (!varsStr.empty()) varsStr += ",";
                varsStr += v;
            }
            result = "!(>[" + varsStr + "]" + body + negatedHead + ")";
        }
    }
    else if (category == "implication") {
        if (elements.empty()) result = signature;
        else {
            std::vector<std::string> chain = elements;
            std::string head = chain.back();
            chain.pop_back();
            const int32_t chainN = static_cast<int32_t>(chain.size());
            assert(chainN <= 64 && "expandSignature: implication chain exceeds 64");
            gl::StrSpan chainSpans[64];
            for (int32_t i = 0; i < chainN; ++i) chainSpans[i] = gl::StrSpan(chain[i]);
            gl::ScratchArena& a =
                gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);
            result = ea.reconstructImplicationFullBindScratch(
                a, chainSpans, chainN, gl::StrSpan(head)).toStdString();
        }
    }
    else if (category == "or") {
        if (elements.empty()) result = signature;
        else if (elements.size() == 1) result = elements[0];
        else {
            // Elements carry true disjunct polarity; the conjunct form is the
            // element's negation with double-negation cancellation. The
            // or-so-far is itself a disjunct of the next level, so it too
            // enters negated (its !(&…) form cancels to the bare (&…)).
            const auto neg = [](const std::string& e) {
                return (!e.empty() && e[0] == '!') ? e.substr(1) : "!" + e;
            };
            std::string current = "!(&" + neg(elements[0]) + neg(elements[1]) + ")";
            for (size_t i = 2; i < elements.size(); ++i)
                current = "!(&" + neg(current) + neg(elements[i]) + ")";
            result = current;
        }
    }
    else {
        assert(false && "Forbidden category encountered in expandSignature.");
        return signature;
    }

    if (category == "implication") return result;

    gl::ScratchArena& esArena =
        gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);
    gl::ScratchScope esScope(esArena);
    gl::StrReplacement pairs[512];
    int32_t pairN = 0;
    int esLevel = 0, esId = 0;
    gl::collectExprTokens(gl::StrSpan(result), [&](const gl::StrSpan& t) {
        if (t.len >= 2 && t.ptr[0] == 'u' && t.ptr[1] == '_') {
            assert(pairN < 512 && "expandSignature rename pairs exceed 512");
            pairs[pairN++] = gl::StrReplacement{ t, gl::StrSpan(t.ptr + 2, t.len - 2) };
        }
        else if (gl::matchItLevId(t, esLevel, esId) || gl::matchIntLevId(t, esLevel, esId)) {
            // it_ / int_ tokens are kept as-is.
        }
        else {
            assert(pairN < 512 && "expandSignature rename pairs exceed 512");
            char* cval = esArena.allocBytes(2 + t.len);
            cval[0] = 'c'; cval[1] = '_';
            std::memcpy(cval + 2, t.ptr, static_cast<size_t>(t.len));
            pairs[pairN++] = gl::StrReplacement{ t, gl::StrSpan(cval, 2 + t.len) };
        }
    });
    return gl::replaceKeysToString(gl::StrSpan(result), pairs, pairN);
}

// expandSignature statified: the former std::string return built from
// std::set/std::vector/std::string scratch became a 0% heap arena builder onto a
// caller ScratchArena (no internal ScratchScope). Byte-identical to the retained
// heap oracle across every category + rename branch.
TEST(prover_span_twins, expand_signature_scratch_matches_heap) {
    gl::ExpressionAnalyzer ea("Peano");  // inits the global scratch arenas
    gl::ScratchArena& arena =
        gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);

    std::vector<gl::LogicalEntity> cases;
    cases.push_back(gl::LogicalEntity(
        "and", { "(P[u_a,x])", "(Q[u_b])", "(R[u_a,y])" }, "(sig[u_a,u_b])", 2));
    cases.push_back(gl::LogicalEntity(
        "existence", { "(body[u_a,z,w])", "(head[u_a])" }, "(sig[u_a])", 1));
    cases.push_back(gl::LogicalEntity(
        "existence", { "(body[u_a,z])", "!(head[u_a])" }, "(sig[u_a])", 1));
    cases.push_back(gl::LogicalEntity(
        "implication", { "(a[u_x])", "(b[u_y])", "(c[u_x,u_y])" }, "(sig[u_x,u_y])", 2));
    cases.push_back(gl::LogicalEntity(
        "or", { "(d0[u_a])", "(d1[u_b])", "(d2[u_c])" }, "(sig[u_a])", 1));
    cases.push_back(gl::LogicalEntity(
        "or", { "(=[u_a,u_b])", "!(=[u_c,u_d])" }, "(sig[u_a,u_b,u_c,u_d])", 4));
    // rename branches: u_ stripped, int_lev kept-or-c_ (both paths agree), plain -> c_.
    cases.push_back(gl::LogicalEntity(
        "and", { "(P[u_a,int_lev_0,plain])", "(Q[u_b])" }, "(sig[u_a,u_b])", 2));

    for (const gl::LogicalEntity& le : cases) {
        const std::string oracle = expandSignatureOracle(ea, le);
        gl::ScratchScope sc(arena);
        const gl::ScratchString twin = ea.expandSignature(le, arena);
        ASSERT_EQ(gl::StrSpan(twin).toStdString(), oracle);

        // Span-run entry point directly (the LE form forwards to it).
        gl::StrSpan elemRun[gl::ExecutionParameters::MAX_INSTRUCTION_ELEMENTS + 1];
        const int32_t elemN = static_cast<int32_t>(le.elements.size());
        for (int32_t i = 0; i < elemN; ++i) elemRun[i] = gl::StrSpan(le.elements[i]);
        const gl::ScratchString twinSpan = ea.expandSignature(
            gl::StrSpan(le.category), gl::StrSpan(le.signature), elemRun, elemN, arena);
        ASSERT_EQ(gl::StrSpan(twinSpan).toStdString(), oracle);
    }
}

// Mixed-polarity or: a registered element may carry its true (negated) sign;
// the expansion negates with double-negation cancellation, so a negated
// disjunct contributes its bare positive core. A positive-only store would
// flip the disjunct's sign on round-trip and mint a FALSE or compound
// ((a=b) OR (c=d) instead of (a=b) OR !(c=d)).
TEST(prover_span_twins, expand_signature_or_negated_disjunct_cancels) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::ScratchArena& arena =
        gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);
    const gl::LogicalEntity le(
        "or", { "(=[u_a,u_b])", "!(=[u_c,u_d])" }, "(sig[u_a,u_b,u_c,u_d])", 4);
    gl::ScratchScope sc(arena);
    const gl::ScratchString twin = ea.expandSignature(le, arena);
    ASSERT_EQ(gl::StrSpan(twin).toStdString(),
              std::string("!(&!(=[a,b])(=[c,d]))"));
    ASSERT_EQ(expandSignatureOracle(ea, le),
              std::string("!(&!(=[a,b])(=[c,d]))"));
}

// k >= 3 or-expansion nests the or-so-far NEGATED: its !(&…) form cancels
// to the bare positive (&…) conjunct, so the nest reads (D1 v D2) v D3.
// The former builder inserted the or-so-far un-negated — semantically
// NOT(D1 v D2) v D3 — byte-mirrored in the verifier's
// _build_or_from_elements, so the prover-vs-verifier compare PASSED on
// the wrong form (the D-260 latent defect). Literal-bytes assertions so
// a shared oracle/production error cannot mask again.
TEST(prover_span_twins, expand_signature_or_three_disjuncts_nests_negated) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::ScratchArena& arena =
        gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);
    const gl::LogicalEntity le(
        "or", { "(d0[u_a])", "(d1[u_b])", "(d2[u_c])" }, "(sig[u_a])", 1);
    gl::ScratchScope sc(arena);
    const gl::ScratchString twin = ea.expandSignature(le, arena);
    ASSERT_EQ(gl::StrSpan(twin).toStdString(),
              std::string("!(&(&!(d0[a])!(d1[b]))!(d2[c]))"));
    ASSERT_EQ(expandSignatureOracle(ea, le),
              std::string("!(&(&!(d0[a])!(d1[b]))!(d2[c]))"));

    // A negated third disjunct still cancels to its bare positive core.
    const gl::LogicalEntity leNeg(
        "or", { "(d0[u_a])", "(d1[u_b])", "!(d2[u_c])" }, "(sig[u_a])", 1);
    gl::ScratchScope sc2(arena);
    const gl::ScratchString twinNeg = ea.expandSignature(leNeg, arena);
    ASSERT_EQ(gl::StrSpan(twinNeg).toStdString(),
              std::string("!(&(&!(d0[a])!(d1[b]))(d2[c]))"));
}

// Test-local heap oracle for the retired std::set<std::string> body of
// getRemainingArgs -- the byte-identity reference for the span-out-param form.
static std::vector<std::string> getRemainingArgsOracle(const std::vector<std::string>& key) {
    std::set<std::string> argsWithoutUPrefix;
    for (std::size_t i = 0; i < key.size(); ++i) {
        gl::StrSpan sp[gl::ExecutionParameters::MAX_ARITY];
        const int32_t n = gl::getArgsSpans(gl::StrSpan(key[i]), sp,
                                           gl::ExecutionParameters::MAX_ARITY);
        for (int32_t j = 0; j < n; ++j) {
            const std::string arg = sp[j].toStdString();
            if (arg.size() >= 2 && arg[0] == 'u' && arg[1] == '_') {
                int uCount = 0;
                std::size_t pos = 0;
                while ((pos = arg.find("u_", pos)) != std::string::npos) { ++uCount; pos += 2; }
                assert(uCount == 1 && "Argument starting with 'u_' must not contain additional 'u_' substrings");
                argsWithoutUPrefix.insert(arg.substr(2));
            }
        }
    }
    return std::vector<std::string>(argsWithoutUPrefix.begin(), argsWithoutUPrefix.end());
}

// getRemainingArgs statified: the former sorted std::set<std::string> return became
// a sorted-unique StrSpan out-param + count. The out spans alias key; the sort/dedup
// reproduce the std::set order (so the downstream NameMap mint sequence is unchanged).
TEST(prover_span_twins, get_remaining_args_matches_heap) {
    gl::ExpressionAnalyzer ea("Peano");

    // u_ args stripped, non-u_ args (x, c) ignored, duplicate stripped tail (a).
    const std::vector<std::string> key = {
        "(P[u_a,x,u_b])", "(Q[u_a,c])", "(R[u_d])"
    };
    const std::vector<std::string> oracle = getRemainingArgsOracle(key);
    gl::StrSpan out[gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS];
    const int32_t n = ea.getRemainingArgs(
        key, out, gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS);

    ASSERT_EQ(static_cast<std::size_t>(n), oracle.size());
    for (int32_t i = 0; i < n; ++i)
        ASSERT_EQ(gl::StrSpan(out[i]).toStdString(), oracle[i]);
}

// Verbatim retained heap oracle for the former std::set<std::string>-returning
// findDigitArgs (moved to tests; a prover-only heap form is not kept in
// production). Uses the same ExpressionAnalyzer config state (operators /
// coreExpressionMap / anchorInfo) the twin reads, so the differential comparison is
// meaningful. The dropped-from-the-twin coreExpressionMap parameter is read here as
// ea.coreExpressionMap (the value every caller passed).
static std::set<std::string> findDigitArgsOracle(gl::ExpressionAnalyzer& ea,
        const std::string& theorem, const ce::AnchorInfo& anchor) {
    std::vector<std::string> chain;
    gl::StrSpan headSpan;
    ce::disintegrateImplicationSpans(gl::StrSpan(theorem), headSpan,
        [&chain](gl::StrSpan keySpan, const gl::StrSpan*, int32_t) {
            chain.push_back(keySpan.toStdString());
        });
    chain.push_back(headSpan.toStdString());

    for (const auto& element : chain) {
        bool contains = element.find(ea.anchorInfo.name) != std::string::npos;
        if (ea.isEquality(element) || contains) continue;
        std::string stripped = element;
        if (stripped.size() >= 2 && stripped[0] == '!' && stripped[1] == '(')
            stripped = stripped.substr(1, stripped.size() - 2) + ")";
        if (ea.isEquality(stripped)) continue;
        std::string coreExpr = gl::extractExpressionSpan(gl::StrSpan(stripped)).toStdString();
        if (ea.operators.find(coreExpr) == ea.operators.end()
            && ea.coreExpressionMap.find(coreExpr) == ea.coreExpressionMap.end()) {
            if (coreExpr.find("existence") == 0) continue;
            return {};
        }
    }

    std::set<std::string> allInputArgs;
    for (const auto& element : chain) {
        std::string coreExpr = gl::extractExpressionUniversalSpan(gl::StrSpan(element)).toStdString();
        auto it = ea.coreExpressionMap.find(coreExpr);
        if (it != ea.coreExpressionMap.end()) {
            const auto& cfg = it->second;
            if (!cfg.inputIndices.empty()) {
                gl::StrSpan args[gl::ExecutionParameters::MAX_ARITY];
                const int32_t argsN = gl::getArgsSpans(gl::StrSpan(element), args,
                                                       gl::ExecutionParameters::MAX_ARITY);
                for (int idx : cfg.inputIndices)
                    if (idx >= 0 && idx < argsN) allInputArgs.insert(args[idx].toStdString());
            }
        }
    }
    for (const auto& element : chain) {
        if (element.find(anchor.name) != std::string::npos) {
            gl::StrSpan args[gl::ExecutionParameters::MAX_ARITY];
            const int32_t argsN = gl::getArgsSpans(gl::StrSpan(element), args,
                                                   gl::ExecutionParameters::MAX_ARITY);
            for (int32_t a = 0; a < argsN; ++a) allInputArgs.erase(args[a].toStdString());
        }
    }
    std::set<std::string> allOutputArgs;
    for (const auto& element : chain) {
        std::string coreExpr = gl::extractExpressionUniversalSpan(gl::StrSpan(element)).toStdString();
        auto it = ea.coreExpressionMap.find(coreExpr);
        if (it != ea.coreExpressionMap.end()) {
            const auto& cfg = it->second;
            if (!cfg.outputIndices.empty()) {
                gl::StrSpan args[gl::ExecutionParameters::MAX_ARITY];
                const int32_t argsN = gl::getArgsSpans(gl::StrSpan(element), args,
                                                       gl::ExecutionParameters::MAX_ARITY);
                for (int idx : cfg.outputIndices)
                    if (idx >= 0 && idx < argsN) allOutputArgs.insert(args[idx].toStdString());
            }
        }
    }
    for (const auto& outArg : allOutputArgs) allInputArgs.erase(outArg);
    return allInputArgs;
}

// findDigitArgs statified: the former std::set<std::string> return became a
// sorted-unique StrSpan out-param + count, and the unused coreExpressionMap param
// was dropped (reads route through coreConfig). Byte-identical (content AND
// compareSpans order) to the retained heap oracle across the branch battery.
TEST(prover_span_twins, find_digit_args_matches_heap) {
    gl::ExpressionAnalyzer ea("Peano");
    const std::vector<std::string> cases = {
        "(>[1,2,3](AnchorPeano[1,2,3,4,5,6])(in2[1,2,3]))",  // anchor + operator
        "(>[1,2](AnchorPeano[1,2,3,4,5,6])(=[1,2]))",         // anchor + equality
        "(AnchorPeano[1,2,3,4,5,6])",                          // anchor alone
        "(>[1,2]!(in[1,2])(in[2,1]))",                        // negated premise (strip path)
        "(>[1,2,3](in2[1,2,3])(in[1,2]))",                    // chained operators
    };
    for (const std::string& theorem : cases) {
        const std::set<std::string> oracle = findDigitArgsOracle(ea, theorem, ea.anchorInfo);
        gl::StrSpan out[gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS];
        const int32_t n = ea.findDigitArgs(theorem, ea.anchorInfo, out,
            gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS);
        ASSERT_EQ(static_cast<std::size_t>(n), oracle.size());
        // compareSpans order == std::set order, so out[i] matches oracle's i-th.
        std::set<std::string>::const_iterator it = oracle.begin();
        for (int32_t i = 0; i < n; ++i, ++it)
            ASSERT_EQ(gl::StrSpan(out[i]).toStdString(), *it);
    }
}

// findDigitArgs — the new StrSpan overload is the real body; the std::string
// form delegates to it. Byte-twin (Rule 18): the two spellings yield an
// identical out run (same count, same spans in compareSpans order) across the
// branch battery.
TEST(prover_span_twins, find_digit_args_span_matches_string) {
    gl::ExpressionAnalyzer ea("Peano");
    const std::vector<std::string> cases = {
        "(>[1,2,3](AnchorPeano[1,2,3,4,5,6])(in2[1,2,3]))",  // anchor + operator
        "(>[1,2](AnchorPeano[1,2,3,4,5,6])(=[1,2]))",         // anchor + equality
        "(AnchorPeano[1,2,3,4,5,6])",                          // anchor alone
        "(>[1,2]!(in[1,2])(in[2,1]))",                        // negated premise (strip path)
        "(>[1,2,3](in2[1,2,3])(in[1,2]))",                    // chained operators
    };
    for (const std::string& theorem : cases) {
        gl::StrSpan outStr[gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS];
        const int32_t nStr = ea.findDigitArgs(theorem, ea.anchorInfo, outStr,
            gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS);
        gl::StrSpan outSpan[gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS];
        const int32_t nSpan = ea.findDigitArgs(gl::StrSpan(theorem), ea.anchorInfo,
            outSpan, gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS);
        ASSERT_EQ(nStr, nSpan);
        for (int32_t i = 0; i < nStr; ++i)
            ASSERT_EQ(gl::StrSpan(outStr[i]).toStdString(),
                      gl::StrSpan(outSpan[i]).toStdString());
    }
}
