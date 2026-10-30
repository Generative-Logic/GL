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
#include "implication_multiplier.hpp"
#include "binding_partitions.hpp"
#include "../proof/worker_context.hpp"
#include <cctype>
#include <unordered_set>
namespace gl {
/// @brief Bind definition readers and the existing multiplication settings.
/// @details References outlive the algorithm; construction performs no expansion.
/// @param definitions Configuration and compiled-definition readers.
/// @param parameters Existing multiplication flags and partition limit.
/// @param ceFilteringActive Whether the counterexample phase is currently active.
/// @return Constructed implication multiplier.
ImplicationMultiplier::ImplicationMultiplier(const CompiledDefinitionRegistry& definitions, const ProverParameters& parameters, const bool& ceFilteringActive)
 : definitions(definitions), parameters(parameters), ceFilteringActive(ceFilteringActive) {}

/// @brief Expand an implication into all variable-renamed copies needed
/// to cover every meaningful binding partition.
///
/// @details
/// Used by `addToHashMemory` (Path 1) to produce the per-copy
/// `(key, value)` pairs that get installed in the hash engine. For an
/// implication `(>[v0,v1](p[v0])(q[v0,v1]))`, multiplyImplication
/// generates the family of permutations / partition variants whose
/// renormalized keys differ. The first copy is the input as-is; later
/// copies carry origin records under the `"multiplied from"` tag so the
/// verifier can chain the multiplications back to the original.
///
/// **Per [I-24](../../docs/agentic_swdd/30_invariants.md#i-24)**, multiplyImplication
/// must NOT equate two distinct free `u_*` anchor parameters — they are
/// the pinned anchor slots and cannot be unified by partition renaming.
/// The function asserts on attempts to do so; a firing assert here means
/// the input implication had a malformed anchor binding.
///
/// @param implication Source implication text.
/// @return Vector of all multiplied copies. Always non-empty (at least
///         the input itself). Order is deterministic.
/// @invariant [I-24](../../docs/agentic_swdd/30_invariants.md#i-24) — no
///            free-anchor-param equation.
/// @see `addToHashMemory` — primary consumer (Path 1 head LMV inserts).
std::vector<std::string> ImplicationMultiplier::multiplyImplication(
    const std::string& implication) {
    // Local word-boundary replacement — verbatim copy of the deleted file-static
    // replaceArgInString (won't match v1 inside v10, nor "7" inside "u_7").
    const auto replaceArgHeap = [](const std::string& str,
                                   const std::string& oldArg,
                                   const std::string& newArg) -> std::string {
        auto isWordChar = [](char c) -> bool {
            return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
        };
        std::string result;
        result.reserve(str.size());
        std::size_t i = 0;
        while (i < str.size()) {
            bool atBoundary = (i == 0) || !isWordChar(str[i - 1]);
            if (atBoundary && str.compare(i, oldArg.size(), oldArg) == 0) {
                std::size_t after = i + oldArg.size();
                bool endBoundary =
                    (after >= str.size()) || !isWordChar(str[after]);
                if (endBoundary) {
                    result += newArg;
                    i = after;
                    continue;
                }
            }
            result += str[i];
            ++i;
        }
        return result;
    };

    // Gated by parameters.allow_multiplication (was parameters.incubator_mode
    // pre-2026-04-29). CE filter still multiplies regardless to keep the
    // filter's reachability fan-out unchanged.
    if (!parameters.allow_multiplication && !ceFilteringActive) return {implication};

    // 1. Parse implication to chain + head (row 238 span twin). Collect all
    //    bound vars (get<1>) into allBoundVars (a set, order-independent) and
    //    all premises (get<0>, skip Anchor) into allExprs in triple order, then
    //    the head — byte-identical to the former two heap loops.
    std::set<std::string> allBoundVars;
    std::vector<std::string> allExprs;
    StrSpan headSpan;
    ce::disintegrateImplicationSpans(StrSpan(implication), headSpan,
        [&allBoundVars, &allExprs](StrSpan keySpan, const StrSpan* bvSpans, int32_t bvN) {
            for (int32_t k = 0; k < bvN; ++k) {
                if (!bvSpans[k].empty()) allBoundVars.insert(bvSpans[k].toStdString());
            }
            std::string expr = keySpan.toStdString();
            if (expr.find("(Anchor") == std::string::npos)
                allExprs.push_back(std::move(expr));
        });
    std::string head = headSpan.toStdString();
    if (head.find("(Anchor") == std::string::npos)
        allExprs.push_back(head);

    // 4. Collect ALL unique variables (bound + u_) appearing in expressions
    std::set<std::string> allVars;
    allVars.insert(allBoundVars.begin(), allBoundVars.end());
    for (const std::string& expr : allExprs) {
        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int32_t argsN = getArgsSpans(StrSpan(expr), args,
                                           ExecutionParameters::MAX_ARITY);
        for (int32_t a = 0; a < argsN; ++a) {
            allVars.insert(args[a].toStdString());
        }
    }

    // 5. Determine which vars are (1)-typed (both bound and u_)
    std::vector<std::string> oneTypedVars;
    for (const std::string& var : allVars) {
        bool isOneTyped = false;
        for (const std::string& expr : allExprs) {
            const ce::CoreExpressionConfig* cfg =
                this->definitions.coreConfig(extractExpressionSpan(StrSpan(expr)));
            if (cfg == nullptr) continue;

            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argsN = getArgsSpans(StrSpan(expr), args,
                                               ExecutionParameters::MAX_ARITY);
            for (int32_t i = 0; i < argsN; ++i) {
                if (equalSpans(args[i], StrSpan(var))) {
                    std::string posKey = std::to_string(i + 1);
                    auto dsIt = cfg->definitionSets.find(posKey);
                    if (dsIt != cfg->definitionSets.end() &&
                        dsIt->second.first == "(1)") {
                        isOneTyped = true;
                        break;
                    }
                }
            }
            if (isOneTyped) break;
        }
        if (isOneTyped) oneTypedVars.push_back(var);
    }

    std::sort(oneTypedVars.begin(), oneTypedVars.end(), numericLess);

    if (oneTypedVars.size() <= 1) return {implication};

    // 6. Generate set partitions (equivalence relations on oneTypedVars) as an
    //    arena CSR of element indices — heap-free (row 252). Each partition can
    //    compose multiple equalization groups simultaneously.
    const unsigned gspSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gspArena = genScratchArenas().forSlot(gspSlot);
    DirtyState gspDirty = DirtyState::Clean;
    PagedVector<int32_t> partMember(&gspArena, &gspDirty);
    PagedVector<int32_t> partClassStart(&gspArena, &gspDirty);
    PagedVector<int32_t> partPartStart(&gspArena, &gspDirty);
    BindingPartitions::generateSetPartitions(oneTypedVars, parameters.max_partition_size, gspArena,
        partMember, partClassStart, partPartStart);

    // 7. For each partition, substitute and deduplicate.
    //    u_ variables always win as representative — non-u_ replaced by u_.
    std::set<std::string> seen;
    std::vector<std::string> result;

    const int32_t numParts = partPartStart.size() - 1;
    for (int32_t pIdx = 0; pIdx < numParts; ++pIdx) {
        const int32_t cBegin = partPartStart[pIdx];
        const int32_t cEnd = partPartStart[pIdx + 1];

        // Skip partitions where any equivalence class contains two different u_ vars.
        // Two distinct free anchor parameters must not be equated by partition-based
        // multiplication: doing so silently rewrites a free slot of the rule body
        // (chapter-1115 bug — `u_6` overwritten by `u_2` inside `existence2`).
        bool hasDoubleU = false;
        for (int32_t c = cBegin; c < cEnd && !hasDoubleU; ++c) {
            int uCount = 0;
            for (int32_t m = partClassStart[c]; m < partClassStart[c + 1]; ++m) {
                const std::string& v = oneTypedVars[static_cast<std::size_t>(partMember[m])];
                if (v.size() >= 2 && v[0] == 'u' && v[1] == '_') uCount++;
            }
            if (uCount >= 2) hasDoubleU = true;
        }
        if (hasDoubleU) continue;

        std::string copy = implication;

        for (int32_t c = cBegin; c < cEnd; ++c) {
            const int32_t mBegin = partClassStart[c];
            const int32_t mEnd = partClassStart[c + 1];
            if (mEnd - mBegin <= 1) continue;

            // Pick representative: prefer u_ variable, else first element.
            const std::string* rep =
                &oneTypedVars[static_cast<std::size_t>(partMember[mBegin])];
            for (int32_t m = mBegin; m < mEnd; ++m) {
                const std::string& v = oneTypedVars[static_cast<std::size_t>(partMember[m])];
                if (v.size() >= 2 && v[0] == 'u' && v[1] == '_') { rep = &v; break; }
            }

            // Replace all non-representative vars with the representative.
            for (int32_t m = mBegin; m < mEnd; ++m) {
                const std::string& v = oneTypedVars[static_cast<std::size_t>(partMember[m])];
                if (v == *rep) continue;
                copy = replaceArgHeap(copy, v, *rep);
            }
        }

        {
            // Boundary copy in this still-heap caller: build the deduplicated
            // implication on the string tier, materialize back to `copy` (reused
            // downstream by disintegrateImplicationSpans + the `seen`/`result`
            // std::string containers). The span never outlives dbvScope.
            const unsigned dbvSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : scratchArenas().slotCount() - 1;
            ScratchArena& dbvArena = scratchArenas().forSlot(dbvSlot);
            ScratchScope dbvScope(dbvArena);
            copy = deduplicateBoundVarsScratch(StrSpan(copy), dbvArena).toStdString();
        }

        // Skip copies whose head is a trivial equality (=[x,x]).
        {
            StrSpan copyHeadSpan;
            ce::disintegrateImplicationSpans(StrSpan(copy), copyHeadSpan,
                [](StrSpan, const StrSpan*, int32_t) {});   // head only
            std::string copyHead = copyHeadSpan.toStdString();
            if (equalSpans(extractExpressionSpan(StrSpan(copyHead)), StrSpan("=", 1))) {
                StrSpan eqArgs[ExecutionParameters::MAX_ARITY];
                const int32_t eqArgsN = getArgsSpans(StrSpan(copyHead), eqArgs,
                                                     ExecutionParameters::MAX_ARITY);
                if (eqArgsN == 2 && equalSpans(eqArgs[0], eqArgs[1])) continue;
            }
        }

        if (seen.find(copy) == seen.end()) {
            seen.insert(copy);
            result.push_back(copy);
        }
    }

    return result;
}

/// @brief Span form of @ref multiplyImplication — emit each surviving
        ///        multiplied copy as a `CopyRef` into a caller-owned index, 0%
        ///        heap.
        ///
        /// @details
        /// The production path (the `std::vector<std::string>` overload is the
        /// retained heap oracle). Byte-for-byte identical emitted copy SEQUENCE
        /// (bytes + order) to the oracle: same `disintegrateImplicationSpans`
        /// var/expr collection, the `numericLessSpan` sort (the `numericLess`
        /// twin), the `generateSetPartitions` StrSpan-run overload, the iterated
        /// `replaceArgSpanScratch` word-boundary substitution (the
        /// `replaceArgInString` twin), `deduplicateBoundVarsScratch`, the
        /// trivial-equality-head skip, and the content-dedup. Vars/exprs are
        /// span/offset references into the caller-stable @p implication; the
        /// dedup rides a gen-scratch `ColdHashSet`; each surviving copy is built
        /// on @p outStrArena's string tier and emitted as a `CopyRef` (the
        /// dropped partitions' scratch is reclaimed per partition). The two
        /// early-outs (`!allow_multiplication` / `<= 1` one-typed var) emit a
        /// single copy of @p implication.
        ///
        /// @param implication Source implication text (caller-stable for the call).
        /// @param outStrArena Caller string-tier arena receiving the surviving
        ///                    copies' bytes; the caller keeps it live (a
        ///                    `ScratchScope`) across its read of @p out.
        /// @param out         Caller-owned page-tier index; receives one `CopyRef`
        ///                    per emitted copy in emission order.
        /// @invariant [I-24](../../docs/agentic_swdd/30_invariants.md#i-24) — no
        ///            free-anchor-param equation (the `hasDoubleU` guard).
        /// @invariant Emitted copy sequence byte-identical to the heap oracle
        ///            (twin test `multiply_implication_span_matches_heap`).
        /// @see multiplyImplication(const std::string&) — the retained heap oracle;
        ///      replaceArgSpanScratch, numericLessSpan, generateSetPartitions.
/// @return Nothing.
void ImplicationMultiplier::multiplyImplication(StrSpan implication,
    ScratchArena& outStrArena, PagedVector<CopyRef>& out) {
    // Emit one copy = a verbatim slice of `s` onto outStrArena; record a CopyRef.
    const auto emitCopy = [&](StrSpan s) {
        assert(s.len > 0 && "multiplyImplication: empty copy");
        const ArenaOffset off = outStrArena.alloc(s.len, 1);
        std::memcpy(outStrArena.resolve(off), s.ptr,
                    static_cast<std::size_t>(s.len));
        out.push_back(CopyRef{ off, s.len });
    };

    if (!parameters.allow_multiplication && !ceFilteringActive) {
        emitCopy(implication);
        return;
    }

    // Per-slot scratch arenas: genArena hosts the page-tier dedup/index/CSR
    // containers + the byte-bump sortedVars run; outStrArena (string tier) holds
    // the emitted copies + the per-partition transient build (09b pitfall 2 /
    // I-124). A var/expr reference is an (offset,len) slice into the
    // caller-stable `implication` (row 62) — no bytes copied.
    const unsigned slot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& genArena = genScratchArenas().forSlot(slot);
    DirtyState d = DirtyState::Clean;

    struct VarRef { int32_t off; int32_t len; };
    const auto refSpan = [&](VarRef r) -> StrSpan {
        return StrSpan(implication.ptr + r.off, r.len);
    };
    const auto makeRef = [&](StrSpan s) -> VarRef {
        return VarRef{ static_cast<int32_t>(s.ptr - implication.ptr), s.len };
    };

    // 1. Parse to chain + head; collect unique vars (allVars) via a dedup
    //    ColdHashSet (order-independent — the numericLessSpan sort below washes
    //    out the mint order), and every non-Anchor expr (triple keys + head).
    ColdHashSet<BytesKeyStore> allVarsSet(&genArena, &d);
    PagedVector<VarRef> allVars(&genArena, &d);
    PagedVector<VarRef> allExprs(&genArena, &d);
    const auto addVar = [&](StrSpan v) {
        if (v.empty()) return;
        const int32_t before = allVarsSet.count();
        allVarsSet.mint(v);
        if (allVarsSet.count() > before) allVars.push_back(makeRef(v));
    };
    StrSpan headSpan;
    ce::disintegrateImplicationSpans(implication, headSpan,
        [&](StrSpan keySpan, const StrSpan* bvSpans, int32_t bvN) {
            for (int32_t k = 0; k < bvN; ++k) addVar(bvSpans[k]);
            if (!containsSpan(keySpan, StrSpan("(Anchor", 7)))
                allExprs.push_back(makeRef(keySpan));
        });
    if (!containsSpan(headSpan, StrSpan("(Anchor", 7)))
        allExprs.push_back(makeRef(headSpan));

    for (int32_t e = 0; e < allExprs.size(); ++e) {
        const StrSpan exprSpan = refSpan(allExprs[e]);
        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int32_t argsN = getArgsSpans(exprSpan, args,
                                           ExecutionParameters::MAX_ARITY);
        for (int32_t a = 0; a < argsN; ++a) addVar(args[a]);
    }

    // 2. Which vars are (1)-typed (appear at a (1)-typed slot of some expr). The
    //    position key is the decimal of (i+1); definitionSets is a non-transparent
    //    std::map, so scan it comparing keys as spans (heap-free, byte-identical
    //    to the former std::to_string(i+1) + find).
    PagedVector<VarRef> oneTypedVars(&genArena, &d);
    for (int32_t vi = 0; vi < allVars.size(); ++vi) {
        const StrSpan var = refSpan(allVars[vi]);
        bool isOneTyped = false;
        for (int32_t e = 0; e < allExprs.size() && !isOneTyped; ++e) {
            const StrSpan exprSpan = refSpan(allExprs[e]);
            const ce::CoreExpressionConfig* cfg =
                this->definitions.coreConfig(extractExpressionSpan(exprSpan));
            if (cfg == nullptr) continue;
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argsN = getArgsSpans(exprSpan, args,
                                               ExecutionParameters::MAX_ARITY);
            for (int32_t i = 0; i < argsN; ++i) {
                if (!equalSpans(args[i], var)) continue;
                char posBuf[12];
                const int32_t posN = writeDecimalDigits(posBuf, i + 1);
                const StrSpan posSpan(posBuf, posN);
                for (const auto& kv : cfg->definitionSets) {
                    if (equalSpans(StrSpan(kv.first), posSpan)) {
                        if (kv.second.first == "(1)") isOneTyped = true;
                        break;
                    }
                }
                if (isOneTyped) break;
            }
        }
        if (isOneTyped) oneTypedVars.push_back(allVars[vi]);
    }

    const int32_t oneN = oneTypedVars.size();
    if (oneN <= 1) { emitCopy(implication); return; }

    // 3. Sort oneTypedVars by numericLessSpan (the numericLess twin) into a
    //    contiguous StrSpan run for the generateSetPartitions StrSpan overload.
    int32_t* order = reinterpret_cast<int32_t*>(genArena.resolve(genArena.alloc(
        oneN * static_cast<int32_t>(sizeof(int32_t)),
        static_cast<int32_t>(alignof(int32_t)))));
    for (int32_t i = 0; i < oneN; ++i) order[i] = i;
    std::sort(order, order + oneN, [&](int32_t a, int32_t b) {
        return numericLessSpan(refSpan(oneTypedVars[a]),
                               refSpan(oneTypedVars[b]));
    });
    StrSpan* sortedVars = reinterpret_cast<StrSpan*>(genArena.resolve(genArena.alloc(
        oneN * static_cast<int32_t>(sizeof(StrSpan)),
        static_cast<int32_t>(alignof(StrSpan)))));
    for (int32_t i = 0; i < oneN; ++i)
        sortedVars[i] = refSpan(oneTypedVars[order[i]]);

    // 4. Set partitions (Bell) of sortedVars as an arena CSR of element indices.
    PagedVector<int32_t> partMember(&genArena, &d);
    PagedVector<int32_t> partClassStart(&genArena, &d);
    PagedVector<int32_t> partPartStart(&genArena, &d);
    BindingPartitions::generateSetPartitions(sortedVars, oneN, parameters.max_partition_size,
        genArena, partMember, partClassStart, partPartStart);

    // 5. For each partition: substitute non-representatives, dedup bound vars,
    //    skip trivial-equality heads, content-dedup, emit survivors. Fresh copies
    //    ride outStrArena (string tier); the `seen` set rides genArena (page
    //    tier, a different arena — I-3-safe). Dropped partitions reclaim their
    //    per-partition scratch via popTo.
    ColdHashSet<BytesKeyStore> seen(&genArena, &d);
    const int32_t numParts = partPartStart.size() - 1;
    for (int32_t pIdx = 0; pIdx < numParts; ++pIdx) {
        const int32_t cBegin = partPartStart[pIdx];
        const int32_t cEnd = partPartStart[pIdx + 1];

        // Skip partitions equating two distinct free u_ anchor params (I-24).
        bool hasDoubleU = false;
        for (int32_t c = cBegin; c < cEnd && !hasDoubleU; ++c) {
            int uCount = 0;
            for (int32_t m = partClassStart[c]; m < partClassStart[c + 1]; ++m) {
                const StrSpan v = sortedVars[partMember[m]];
                if (v.len >= 2 && v.ptr[0] == 'u' && v.ptr[1] == '_') uCount++;
            }
            if (uCount >= 2) hasDoubleU = true;
        }
        if (hasDoubleU) continue;

        const ArenaOffset partMark = outStrArena.cursor();
        ScratchString cur = copyScratch(outStrArena, implication);

        for (int32_t c = cBegin; c < cEnd; ++c) {
            const int32_t mBegin = partClassStart[c];
            const int32_t mEnd = partClassStart[c + 1];
            if (mEnd - mBegin <= 1) continue;

            // Representative: prefer a u_ member, else the first element.
            StrSpan rep = sortedVars[partMember[mBegin]];
            for (int32_t m = mBegin; m < mEnd; ++m) {
                const StrSpan v = sortedVars[partMember[m]];
                if (v.len >= 2 && v.ptr[0] == 'u' && v.ptr[1] == '_') {
                    rep = v; break;
                }
            }
            for (int32_t m = mBegin; m < mEnd; ++m) {
                const StrSpan v = sortedVars[partMember[m]];
                if (equalSpans(v, rep)) continue;
                cur = replaceArgSpanScratch(StrSpan(cur), v, rep, outStrArena);
            }
        }

        cur = deduplicateBoundVarsScratch(StrSpan(cur), outStrArena);

        // Skip copies whose head is a trivial equality (=[x,x]).
        {
            StrSpan curHeadSpan;
            ce::disintegrateImplicationSpans(StrSpan(cur), curHeadSpan,
                [](StrSpan, const StrSpan*, int32_t) {});   // head only
            if (equalSpans(extractExpressionSpan(curHeadSpan), StrSpan("=", 1))) {
                StrSpan eqArgs[ExecutionParameters::MAX_ARITY];
                const int32_t eqArgsN = getArgsSpans(curHeadSpan, eqArgs,
                    ExecutionParameters::MAX_ARITY);
                if (eqArgsN == 2 && equalSpans(eqArgs[0], eqArgs[1])) {
                    outStrArena.popTo(partMark);
                    continue;
                }
            }
        }

        // Content-dedup via the seen ColdHashSet: fresh iff the count grew.
        const int32_t before = seen.count();
        seen.mint(StrSpan(cur));
        if (seen.count() > before) {
            emitCopy(StrSpan(cur));         // survivor persists on outStrArena
        } else {
            outStrArena.popTo(partMark);    // duplicate: reclaim partition scratch
        }
    }
}

/// @brief Compare argument labels in numeric binding order.
/// @details Numeric labels use their values and other labels retain string order.
/// @param a First argument label.
/// @param b Second argument label.
/// @return Whether a precedes b in the binding order.
bool ImplicationMultiplier::numericLess(const std::string& a, const std::string& b) {
    bool aDigit = !a.empty() && std::all_of(a.begin(), a.end(),
                      [](char c) { return std::isdigit(static_cast<unsigned char>(c)); });
    bool bDigit = !b.empty() && std::all_of(b.begin(), b.end(),
                      [](char c) { return std::isdigit(static_cast<unsigned char>(c)); });
    if (aDigit && bDigit) return std::stoi(a) < std::stoi(b);
    return a < b;
}
}
