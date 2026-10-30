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
#include "../memory_infra/str_ops.hpp"
#include "../memory.hpp"
#include "../proof/worker_context.hpp"

namespace gl {
/// @brief Definition expansion and bound/free argument transformations.
/// @details Borrows read-only definition tables and carries no prover state.
class DefinitionExpansion {
public:

/// @brief Expand a `LogicalEntity`'s signature into its full expression form
///        (the AND / existence / implication / OR De-Morgan build followed by
///        the `u_`/`it_`/`int_`/`c_` variable renaming) onto a caller arena — 0%
///        heap.
///
/// @details
/// The arena-builder twin of the former heap `expandSignature`. Every string is
/// built with an EXPLICIT length on @p out's byte-bump (string) tier
/// (`assert(at == n)` per the fresh-string doctrine), and the former
/// `std::set<std::string>` / `std::vector<std::string>` scratch became fixed
/// stack `StrSpan` arrays sorted by `compareSpans` (byte-identical to the
/// `std::set` iteration order). There is NO local `ScratchScope`: the returned
/// `ScratchString`, the intermediate build bytes, and the `c_` rename values all
/// ride @p out, held live until the CALLER's scope reclaims it (09c pitfall 4).
///
/// Per category: AND / OR build the left-nested `(& …)` / `!(& …)` forms
/// incrementally; EXISTENCE builds `!(>[boundVars]body negatedHead)` where the
/// bound vars are the body args NOT in the signature (sorted-unique); IMPLICATION
/// returns `reconstructImplicationFullBindScratch(out, …)` DIRECTLY (no renaming,
/// matching the heap's early return). The final renaming collects tokens via
/// `collectExprTokens` (every variable token is inside a `[...]`) and runs
/// `replaceKeysScratch` — the byte-exact arena twin of the former
/// `replaceKeysToString`.
///
/// @param le  The `LogicalEntity` to expand (category one of and/existence/
///            implication/or; signature args asserted `u_`-prefixed).
/// @param out Caller string-tier scratch arena receiving the result AND all
///            intermediate build bytes; must outlive the caller's use of the
///            returned span.
/// @return The expanded, renamed signature as a `ScratchString` on @p out.
/// @invariant Allocates on the heap NOTHING; builds only on @p out; mints no
///            interner (the callers' downstream doors mint, never @p out — I-3).
/// @see reconstructImplicationFullBindScratch, replaceKeysScratch, collectExprTokens.
static ScratchString expandSignature(const LogicalEntity& le, ScratchArena& out);

/// @brief Span-run overload of @ref expandSignature — the span-native
        ///        body; the `const LogicalEntity&` form forwards.
        ///
        /// @details
        /// Byte-identical to the `LogicalEntity` form (same AND / existence /
        /// implication / OR De-Morgan build and `u_`/`it_`/`int_`/`c_` renaming on
        /// the same bytes), but reads the category / signature / elements as spans
        /// / a caller-owned `StrSpan` run instead of a `LogicalEntity`, so a
        /// span-native caller (`disintegrateExprCore2`'s history tracker) reaches
        /// it 0% heap. The `LogicalEntity` form builds a stack `StrSpan` run over
        /// `le.elements` and delegates here.
        ///
        /// @param category The category (and/existence/implication/or), or an
        ///                 expansion-local counterexample whose last element
        ///                 is the negation of the original implication head.
        /// @param signature The signature span (args asserted `u_`-prefixed).
        /// @param elements  The entity elements as a `StrSpan` run; cap
        ///                  `MAX_INSTRUCTION_ELEMENTS + 1`.
        /// @param elemN     The element count.
        /// @param out       Caller string-tier scratch arena receiving the result
        ///                  AND all intermediate bytes; must outlive the use.
        /// @return The expanded, renamed signature as a `ScratchString` on @p out.
        /// @invariant Allocates nothing on the heap; builds only on @p out.
        /// @see expandSignature(const LogicalEntity&, ScratchArena&) — the twin
        ///      oracle; reconstructImplicationFullBindScratch, replaceKeysScratch.
static ScratchString expandSignature(StrSpan category, StrSpan signature,
            const StrSpan* elements, int32_t elemN, ScratchArena& out);

/// @brief Collect the distinct non-`u_` argument names of an existence
///        `LogicalEntity` (the "last removed" bound-var candidates) into a
///        caller span buffer, sorted-unique — 0% heap.
///
/// @details
/// The span twin of the former `std::set<std::string>` collector: over every
/// `le.elements[i]`, `getArgsSpans` slices the flat `[...]` args and each arg
/// that does NOT begin `u_` is gathered as a `StrSpan` INTO `le.elements`
/// (input lifetime). The gather buffer is then `compareSpans`-sorted and
/// adjacent-deduplicated, reproducing the former `std::set<std::string>`
/// iteration order byte-for-byte (`compareSpans` == `std::string::compare`),
/// and the distinct run is copied into @p out. No arena is reached — every
/// output span aliases @p le, which must outlive the caller's use.
///
/// @param le  Existence `LogicalEntity` (asserted `category == "existence"`).
/// @param out Caller buffer receiving the distinct sorted argument spans.
/// @param cap Capacity of @p out (loud Rule-19 assert on overflow).
/// @return The number of distinct non-`u_` arguments written to @p out.
/// @invariant Allocates nothing; output spans alias @p le.elements.
/// @see getArgsSpans, compareSpans, equalSpans.
static int32_t listLastRemovedArgsLE(const LogicalEntity& le,
                                      StrSpan* out, int32_t cap);

/// @brief Span-run overload of @ref listLastRemovedArgsLE — the
        ///        span-native body; the `const LogicalEntity&` form forwards.
        ///
        /// @details
        /// Byte-identical to the `LogicalEntity` form (same non-`u_` gather over
        /// the elements, same `compareSpans` sort + adjacent-dedup into the
        /// `std::set<std::string>` order), but reads the category / elements as
        /// spans / a caller-owned `StrSpan` run, so a span-native caller
        /// (`disintegrateExprCore2`) reaches it 0% heap. Output spans alias the
        /// caller's @p elements (input lifetime). The `LogicalEntity` form builds
        /// a stack `StrSpan` run over `le.elements` and delegates here.
        ///
        /// @param category Witness category (`existence` or the expansion-local
        ///                 `counterexample` instruction).
        /// @param elements The entity elements as a `StrSpan` run; cap
        ///                 `MAX_INSTRUCTION_ELEMENTS + 1`.
        /// @param elemN    The element count.
        /// @param out      Caller buffer receiving the distinct sorted arg spans.
        /// @param cap      Capacity of @p out (loud Rule-19 assert on overflow).
        /// @return The number of distinct non-`u_` arguments written to @p out.
        /// @invariant Allocates nothing; output spans alias @p elements.
        /// @see listLastRemovedArgsLE(const LogicalEntity&, StrSpan*, int32_t) —
        ///      the twin oracle; getArgsSpans, compareSpans.
static int32_t listLastRemovedArgsLE(StrSpan category,
            const StrSpan* elements, int32_t elemN, StrSpan* out, int32_t cap);

/// @brief Scratch-arena twin of @ref makeMarkedExpr — replace @p var with
        ///        the literal "marker" in @p expr, result on @p arena.
        ///
        /// @details
        /// Byte-identical to @ref makeMarkedExpr: one `var -> "marker"`
        /// `StrReplacement` through `replaceKeysScratch` instead of
        /// `replaceKeysToString` ("marker" is a static literal slice, so no arena is
        /// needed for the value). Result is a `ScratchString` on @p arena.
        ///
        /// @param arena Per-slot scratch arena receiving the result.
        /// @param expr  Canonical MPL expression span.
        /// @param var   Variable name to mark.
        /// @return The marked expression as a `ScratchString` on @p arena.
        /// @see makeMarkedExpr, replaceKeysScratch
static inline ScratchString makeMarkedExprScratch(ScratchArena& arena,
            const StrSpan& expr, const StrSpan& var);

/// @brief Rename an admission chain using its retained argument set.
/// @details Preserves first occurrence, fixed names and deterministic replacement order.
/// @param chain Ordered input expressions.
/// @param remainingArgs Argument names retained by the chain.
/// @return Renamed expressions in the original chain order.
static inline std::vector<std::string> renamingChain(const std::vector<std::string>& chain, const std::set<std::string>& remainingArgs) {
            // 0. Early check: If chain has NO "u_" args, return it unchanged.
            bool hasU = false;
            for (const std::string& expr : chain) {
                std::vector<std::string> args = ce::getArgs(expr);
                for (const std::string& arg : args) {
                    if (startsWith(arg, "u_", 2)) {
                        hasU = true;
                        break;
                    }
                }
                if (hasU) break;
            }

            if (!hasU) {
                return chain;
            }

            // --- Proceed with renaming logic ---

            std::set<std::string> uArgsInChainStr;
            std::set<int> uArgsInChain;
            std::set<int> changeableArgs;

            // 1. Identify all arguments in the chain
            for (const std::string& expr : chain) {
                std::vector<std::string> args = ce::getArgs(expr);
                for (const std::string& arg : args) {
                    if (startsWith(arg, "u_", 2)) {
                        std::string val = arg.substr(2);
                        uArgsInChainStr.insert(val);
                        if (!val.empty() && std::all_of(val.begin(), val.end(),
                            [](unsigned char c) { return std::isdigit(c); })) {
                            uArgsInChain.insert(std::stoi(val));
                        }
                    }
                    else {
                        // Assert that the changeable argument is an integer
                        bool isInt = !arg.empty() && std::all_of(arg.begin(), arg.end(),
                            [](unsigned char c) { return std::isdigit(c); });
                        assert(isInt && "Changeable argument in chain must be a valid integer");

                        changeableArgs.insert(std::stoi(arg));
                    }
                }
            }

            // 2. Assert that remainingArgs is a SUBSET of uArgsInChain
            bool isSubset = std::includes(uArgsInChainStr.begin(), uArgsInChainStr.end(),
                remainingArgs.begin(), remainingArgs.end());
            assert(isSubset && "remainingArgs must be a subset of the 'u_' arguments present in the chain");

            // 3. Check for overlap between changeable args and 'u_' args
            bool overlap = false;
            for (int cArg : changeableArgs) {
                if (uArgsInChain.count(cArg)) {
                    overlap = true;
                    break;
                }
            }

            // 4. Build Replacement Map
            std::map<std::string, std::string> replacementMap;

            // 4a. Always map u_X -> X
            for (const std::string& uArg : uArgsInChainStr) {
                replacementMap["u_" + uArg] = uArg;
            }

            // 4b. Handle changeable arguments only if there is an overlap
            if (overlap) {
                // Fallback to remapping logic: Assign new free integers
                std::set<int> blockedInts = uArgsInChain; // Block all IDs used by u_ args (since they map to X)
                int nextFree = 1;

                for (int originalVal : changeableArgs) {
                    // Find next free integer
                    while (blockedInts.count(nextFree)) {
                        nextFree++;
                    }
                    replacementMap[std::to_string(originalVal)] = std::to_string(nextFree);
                    blockedInts.insert(nextFree);
                }
            }
            // Else: If no overlap, changeable args keep their original names (not added to map)

            // 5. Apply renaming
            std::vector<std::string> renamedChain;
            renamedChain.reserve(chain.size());
            for (const std::string& expr : chain) {
                renamedChain.push_back(ce::replaceKeysInString(expr, replacementMap));
            }

            return renamedChain;
        }
/// @brief Span-run twin of `renamingChain` — reproduces the renamed
        ///        admission-key chain byte-for-byte onto a scratch arena,
        ///        allocating no heap and minting into no interner.
        ///
        /// @details
        /// The zero-heap replacement for `renamingChain`. It reads the
        /// `chain` key elements as `StrSpan`s over stable caller input and
        /// writes the renamed elements as `StrSpan`s (either aliasing the
        /// input on the no-rename fast path, or `ScratchString`s freshly built
        /// on `arena`) into `outChain`.
        ///
        /// Four sub-steps mirror the oracle exactly:
        ///  1. Early no-`u_` path — if no key element carries a `u_`-prefixed
        ///     argument, every `outChain[i]` ALIASES `chain[i]` (no copy) and
        ///     the count returns unchanged.
        ///  2. Identify the chain's arguments into three sorted-unique working
        ///     sets: the `u_`-stripped string values (`uStrVals`), the subset
        ///     of those that are all-digit as ints (`uIntVals`), and the
        ///     non-`u_` "changeable" integer arguments (`changeInts`). The
        ///     `isInt` precondition on a changeable argument is kept as a loud
        ///     assert (Rule 19), byte-identical to the oracle.
        ///  3. Assert `remainingArgs` is a subset of `uStrVals` (the oracle's
        ///     `std::includes`), and detect overlap between changeable ints and
        ///     `u_` ints.
        ///  4. Build the substitution pairs — the `u_X -> X` pairs (KEY is
        ///     FRESH bytes `"u_"+value` on `arena`, the VALUE is a slice of
        ///     that fresh key), and on overlap the `originalVal -> nextFree`
        ///     integer remap (both sides FRESH decimal bytes on `arena`, the
        ///     `nextFree` monotone free-integer walk identical to the oracle).
        ///  5. Apply `replaceKeysScratch` (the greedy-longest twin of
        ///     `ce::replaceKeysInString`) per element.
        ///
        /// The FRESH-vs-slice split is the sole byte-sensitive point: only the
        /// `"u_"+value` keys and the `nextFree`/`originalVal` decimals are
        /// fresh bytes; every value is either a slice of a fresh key or a slice
        /// of stable input. The renamed spans stay live until the caller's
        /// `ScratchScope` rewinds `arena` (Pitfall 4 — held as `outChain`
        /// spans over `arena` allocations, never across a rewind).
        ///
        /// @param arena        Scratch (string-tier) arena receiving the fresh
        ///                     key bytes and the `replaceKeysScratch` results;
        ///                     the caller owns its `ScratchScope`.
        /// @param chain        The key chain elements as spans over stable input.
        /// @param chainN       Element count of `chain`.
        /// @param remainingArgs Sorted-unique remaining-argument run (the former
        ///                     `std::set<std::string>` iteration order).
        /// @param remN         Element count of `remainingArgs`.
        /// @param outChain     Caller buffer receiving the renamed spans; filled
        ///                     positionally.
        /// @param outCap       Capacity of `outChain`.
        /// @return The renamed element count (always `== chainN`).
        /// @invariant Byte-identical to `renamingChain` element-for-element
        ///            (I-84 mint-neutral — this function mints into no interner).
        /// @see renamingChain (the retained byte-oracle), replaceKeysScratch.
static int32_t renamingChainScratch(ScratchArena& arena,
            const StrSpan* chain, int32_t chainN,
            const StrSpan* remainingArgs, int32_t remN,
            StrSpan* outChain, int32_t outCap) {
            assert(chainN <= outCap
                && "renamingChainScratch: chain element count exceeds caller capacity");

            constexpr int32_t kDistinctCap = ExecutionParameters::MAX_ADMISSION_REM_ARGS;

            // 0. Early check: chain with NO "u_" args returns unchanged (aliased).
            bool hasU = false;
            for (int32_t i = 0; i < chainN && !hasU; ++i) {
                StrSpan args[ExecutionParameters::MAX_ARITY];
                const int32_t argN = getArgsSpans(chain[i], args,
                                                  ExecutionParameters::MAX_ARITY);
                for (int32_t a = 0; a < argN; ++a) {
                    if (args[a].len >= 2 && args[a].ptr[0] == 'u'
                        && args[a].ptr[1] == '_') { hasU = true; break; }
                }
            }
            if (!hasU) {
                for (int32_t i = 0; i < chainN; ++i) outChain[i] = chain[i];
                return chainN;
            }

            // Sorted-unique working sets (twins of the std::set / std::set<int>).
            StrSpan uStrVals[kDistinctCap];   int32_t uStrN = 0;    // u_-stripped strings
            int     uIntVals[kDistinctCap];   int32_t uIntN = 0;    // digit u_ values
            int     changeInts[kDistinctCap]; int32_t changeN = 0;  // non-u_ ints

            const auto isAllDigits = [](const StrSpan& s) -> bool {
                if (s.len == 0) return false;
                for (int32_t k = 0; k < s.len; ++k)
                    if (!std::isdigit(static_cast<unsigned char>(s.ptr[k]))) return false;
                return true;
            };
            const auto insSpan = [&](StrSpan* arr, int32_t& n, StrSpan v) {
                int32_t lo = 0, hi = n;
                while (lo < hi) {
                    const int32_t mid = (lo + hi) / 2;
                    const int c = compareSpans(arr[mid], v);
                    if (c == 0) return;
                    if (c < 0) lo = mid + 1; else hi = mid;
                }
                assert(n < kDistinctCap
                    && "renamingChainScratch: distinct u_ string count exceeds cap");
                for (int32_t k = n; k > lo; --k) arr[k] = arr[k - 1];
                arr[lo] = v; ++n;
            };
            const auto insInt = [&](int* arr, int32_t& n, int v) {
                int32_t lo = 0, hi = n;
                while (lo < hi) {
                    const int32_t mid = (lo + hi) / 2;
                    if (arr[mid] == v) return;
                    if (arr[mid] < v) lo = mid + 1; else hi = mid;
                }
                assert(n < kDistinctCap
                    && "renamingChainScratch: distinct int count exceeds cap");
                for (int32_t k = n; k > lo; --k) arr[k] = arr[k - 1];
                arr[lo] = v; ++n;
            };
            const auto hasInt = [](const int* arr, int32_t n, int v) -> bool {
                int32_t lo = 0, hi = n;
                while (lo < hi) {
                    const int32_t mid = (lo + hi) / 2;
                    if (arr[mid] == v) return true;
                    if (arr[mid] < v) lo = mid + 1; else hi = mid;
                }
                return false;
            };
            const auto formatIntOnArena =
                [&](int v, char*& outp, int32_t& outl) {
                char tmp[16]; int32_t t = 0;
                if (v == 0) { tmp[t++] = '0'; }
                else { int vv = v; while (vv > 0) { tmp[t++] = static_cast<char>('0' + vv % 10); vv /= 10; } }
                char* b = arena.allocBytes(t);
                for (int32_t k = 0; k < t; ++k) b[k] = tmp[t - 1 - k];
                outp = b; outl = t;
            };

            // 1. Identify all arguments in the chain.
            for (int32_t i = 0; i < chainN; ++i) {
                StrSpan args[ExecutionParameters::MAX_ARITY];
                const int32_t argN = getArgsSpans(chain[i], args,
                                                  ExecutionParameters::MAX_ARITY);
                for (int32_t a = 0; a < argN; ++a) {
                    const StrSpan& arg = args[a];
                    if (arg.len >= 2 && arg.ptr[0] == 'u' && arg.ptr[1] == '_') {
                        const StrSpan val(arg.ptr + 2, arg.len - 2);
                        insSpan(uStrVals, uStrN, val);
                        if (isAllDigits(val)) {
                            int iv = 0;
                            const bool ok = parseDigitRunToInt(val.ptr, val.len, iv);
                            assert(ok && "renamingChainScratch: u_ digit value overflow");
                            insInt(uIntVals, uIntN, iv);
                        }
                    } else {
                        // Assert that the changeable argument is an integer.
                        const bool isInt = isAllDigits(arg);
                        assert(isInt && "Changeable argument in chain must be a valid integer");
                        int iv = 0;
                        const bool ok = parseDigitRunToInt(arg.ptr, arg.len, iv);
                        assert(ok && "renamingChainScratch: changeable arg overflow");
                        insInt(changeInts, changeN, iv);
                    }
                }
            }

            // 2. remainingArgs must be a SUBSET of uStrVals.
            for (int32_t r = 0; r < remN; ++r) {
                bool found = false;
                int32_t lo = 0, hi = uStrN;
                while (lo < hi) {
                    const int32_t mid = (lo + hi) / 2;
                    const int c = compareSpans(uStrVals[mid], remainingArgs[r]);
                    if (c == 0) { found = true; break; }
                    if (c < 0) lo = mid + 1; else hi = mid;
                }
                assert(found && "remainingArgs must be a subset of the 'u_' arguments present in the chain");
            }

            // 3. Overlap between changeable args and u_ ints.
            bool overlap = false;
            for (int32_t c = 0; c < changeN && !overlap; ++c)
                if (hasInt(uIntVals, uIntN, changeInts[c])) overlap = true;

            // 4. Build the substitution pairs.
            constexpr int32_t kPairsCap = 2 * kDistinctCap;
            StrReplacement pairs[kPairsCap]; int32_t pairN = 0;

            // 4a. Always map u_X -> X (fresh key bytes; value is a slice of the key).
            for (int32_t k = 0; k < uStrN; ++k) {
                const StrSpan& uArg = uStrVals[k];
                char* keyBuf = arena.allocBytes(2 + uArg.len);
                keyBuf[0] = 'u'; keyBuf[1] = '_';
                if (uArg.len > 0)
                    std::memcpy(keyBuf + 2, uArg.ptr, static_cast<size_t>(uArg.len));
                assert(pairN < kPairsCap
                    && "renamingChainScratch: replacement pair count exceeds cap");
                pairs[pairN].key = StrSpan(keyBuf, 2 + uArg.len);
                pairs[pairN].value = StrSpan(keyBuf + 2, uArg.len);
                ++pairN;
            }

            // 4b. On overlap, assign fresh free integers to the changeable args.
            if (overlap) {
                int blocked[kPairsCap]; int32_t blockedN = 0;
                for (int32_t k = 0; k < uIntN; ++k) blocked[blockedN++] = uIntVals[k];
                int nextFree = 1;
                for (int32_t c = 0; c < changeN; ++c) {
                    while (hasInt(blocked, blockedN, nextFree)) ++nextFree;
                    char* kb; int32_t kl; formatIntOnArena(changeInts[c], kb, kl);
                    char* vb; int32_t vl; formatIntOnArena(nextFree, vb, vl);
                    assert(pairN < kPairsCap
                        && "renamingChainScratch: replacement pair count exceeds cap");
                    pairs[pairN].key = StrSpan(kb, kl);
                    pairs[pairN].value = StrSpan(vb, vl);
                    ++pairN;
                    // blockedInts.insert(nextFree) — nextFree is guaranteed absent.
                    int32_t ip = 0;
                    while (ip < blockedN && blocked[ip] < nextFree) ++ip;
                    assert(blockedN < kPairsCap
                        && "renamingChainScratch: blocked-int count exceeds cap");
                    for (int32_t z = blockedN; z > ip; --z) blocked[z] = blocked[z - 1];
                    blocked[ip] = nextFree; ++blockedN;
                }
            }

            // 5. Apply renaming.
            for (int32_t i = 0; i < chainN; ++i)
                outChain[i] = StrSpan(replaceKeysScratch(arena, chain[i], pairs, pairN));

            return chainN;
        }

/// @brief Scratch twin of @ref negate — a leading `!` toggles by slice or
        ///        arena prefix, with no heap `std::string`.
        ///
        /// @details
        /// Byte-for-byte the oracle: a leading `'!'` returns the `substr(1)` SLICE
        /// of the input (no arena touch); otherwise a fresh `"!"+expr` is built on
        /// @p arena (the `prefixBang` idiom). The empty input yields `"!"`. The
        /// returned span aliases either the INPUT (slice case) or @p arena (prefix
        /// case) — the caller keeps both alive across its use (I-116 stack-only).
        ///
        /// @param arena Scratch arena receiving the `"!"+expr` bytes (prefix case only).
        /// @param expr  The expression span.
        /// @return The negated form as a span (input slice or arena bytes).
        /// @invariant Byte-identical to `negate`; mints into no interner.
        /// @see negate (the retained heap oracle — note it retains live production
        ///      callers beyond Core2 Case OR).
static StrSpan negateScratch(ScratchArena& arena, StrSpan expr) {
            if (expr.len > 0 && expr.ptr[0] == '!')
                return StrSpan(expr.ptr + 1, expr.len - 1);
            char* b = arena.allocBytes(1 + expr.len);
            b[0] = '!';
            if (expr.len > 0)
                std::memcpy(b + 1, expr.ptr, static_cast<size_t>(expr.len));
            return StrSpan(b, 1 + expr.len);
        }

    /// @brief Bind the definition tables used by expression expansion.
    /// @details Both tables outlive this view; expansion preserves their contents.
    /// @param configurations Configured source definitions.
    /// @param definitions Registered compiled definitions.
    /// @return Constructed definition-expansion view.
    DefinitionExpansion(const ce::CoreExpressionMap& configurations,
        const CompiledExpressionMap& definitions);

/// @brief Expand a configured predicate without capturing its arguments.
///
/// @details Preserves expression spelling, binding and the existing transformation order.
/// @param expr Expression text.
/// @return Transformed expression text.
std::string expandExpr(const std::string& expr);

/// @brief Expand compiled logical names to their underlying expression form.
///
/// @details Preserves expression spelling, binding and the existing transformation order.
/// @param expr Expression text.
/// @return Transformed expression text.
std::string expandToBaseForm(const std::string& expr) const;

/// @brief Mark the existing free argument names as unchangeable.
///
/// @details Preserves expression spelling, binding and the existing transformation order.
/// @param expr Expression text.
/// @return Transformed expression text.
static std::string prefixArgumentsWithU(const std::string& expr);

/// @brief Remove unchangeable prefixes through token-aware substitution.
///
/// @details Preserves expression spelling, binding and the existing transformation order.
/// @param expr Expression text.
/// @return Transformed expression text.
static std::string removeUPrefixFromArguments(const std::string& expr);

/// @brief Add the existing unchangeable prefix to eligible expression arguments.
///
/// @details Preserves expression spelling, binding and the existing transformation order.
/// @param expr Expression text.
/// @return Transformed expression text.
static std::string addMissingU(const std::string& expr);

/// @brief Replace unchangeable-name occurrences using the existing compiler spelling.
///
/// @details Preserves expression spelling, binding and the existing transformation order.
/// @param s Expression text.
/// @return Transformed expression text.
static std::string replaceUSubstrings(const std::string& s);

/// @brief Remove unchangeable prefixes from expression-tree argument tokens.
///
/// @details Preserves expression spelling, binding and the existing transformation order.
/// @param expr Expression text.
/// @return Transformed expression text.
static std::string stripUPrefixAST(const std::string& expr);

/// @brief Read the two defining sides of a compound expression.
///
/// @details Preserves expression spelling, binding and the existing transformation order.
/// @param s Compound expression text.
/// @param coreExpressionMap Existing configuration view retained by this interface.
/// @return Left and right defining expressions.
static std::pair<std::string, std::string> extractValues(const std::string& s, const ce::CoreExpressionMap& coreExpressionMap);

/// @brief Append the conjuncts of a two-child conjunction chain in leaf order.
///
/// @details
/// A conjunction has exactly two children, so a longer conjunction nests,
/// to the left `(&(&A B)C)` or to the right `(&A(&B C))`. A positive `&`
/// expression is split into its two children (`extractValues`) and each is
/// collected recursively; any other expression — an atom, an implication, a
/// negated conjunction `!(&..)` — is one conjunct. Both nestings of the same
/// conjuncts therefore yield the same ordered list (associativity of `&`).
///
/// @param expression        A conjunct or a positive conjunction; never empty.
/// @param coreExpressionMap Configuration view forwarded to `extractValues`.
/// @param conjuncts         Receives the conjuncts in left-to-right leaf order
///                          (appended).
/// @return Nothing.
/// @see extractValues; StructuralCompiler::compileExpression — the negated
///      conjunction (or) branch that collects its disjuncts through this.
static void collectConjuncts(const std::string& expression, const ce::CoreExpressionMap& coreExpressionMap,
                             std::vector<std::string>& conjuncts);

/// @brief Toggle one leading expression negation.
///
/// @details Preserves expression spelling, binding and the existing transformation order.
/// @param expr Expression text.
/// @return Transformed expression text.
static std::string negate(std::string expr);




/// @brief Scratch twin of @ref prefixArgumentsWithU — `u_`-prefix EVERY argument
///        of @p expr onto the scratch tier, with no heap `std::string`.
///
/// @details
/// Prefixes ALL arguments (contrast @ref prefixNonMarkerArgumentsWithUScratch,
/// which SKIPS the literal `"marker"`). Each argument yields one substitution
/// pair whose value `"u_"+arg` is FRESH bytes bump-allocated on @p arena (the
/// `prefixArgumentsWithU` idiom); the substitution runs through
/// `replaceKeysScratch` (the byte-exact greedy-longest twin of
/// `ce::replaceKeysInString`). A no-argument input contributes zero pairs and
/// degrades to a plain pass-through copy — byte-identical to the oracle's `return
/// expr`. An already-`u_` argument is prefixed AGAIN (`u_u_arg`), exactly as the
/// string oracle does — NOT special-cased. The result rides @p arena (the caller
/// owns the `ScratchScope`) and survives a later interner mint.
///
/// @param arena Scratch (string-tier) arena receiving the fresh value bytes and
///              the substituted result.
/// @param expr  Expression whose arguments are prefixed, as a span over stable input.
/// @return The substituted text as a `ScratchString` on @p arena.
/// @invariant Byte-identical to `prefixArgumentsWithU`; mints into no interner.
/// @see prefixArgumentsWithU (the retained heap oracle),
///      prefixNonMarkerArgumentsWithUScratch (the marker-skipping sibling),
///      replaceKeysScratch, checkNecessityForEquality (the consumer).
static ScratchString prefixArgumentsWithUScratch(ScratchArena& arena, StrSpan expr);


/// @brief `u_`-prefix every NON-`"marker"` argument of `expr` onto the scratch
///        tier — the faithful twin of `isAdmittedIntegration`'s inline
///        `{arg != "marker" -> u_+arg}` rename.
///
/// @details
/// Unlike `prefixArgumentsWithU`, which prefixes ALL arguments, this helper
/// skips any argument equal to the literal `"marker"`, leaving it verbatim.
/// Each non-marker argument yields one substitution pair whose value
/// `"u_"+arg` is FRESH bytes bump-allocated on `arena`; the substitution runs
/// through `replaceKeysScratch` (the byte-exact greedy-longest twin of
/// `ce::replaceKeysInString`). A no-argument or all-`"marker"` input
/// contributes zero pairs and degrades to a plain pass-through copy. The
/// result rides `arena` (the caller owns the `ScratchScope`) and survives a
/// later interner mint (the interner writes its own cold pool, not this arena).
///
/// @param arena Scratch (string-tier) arena receiving the fresh value bytes
///              and the substituted result.
/// @param expr  Expression whose arguments are prefixed, as a span over stable
///              input.
/// @return The substituted text as a `ScratchString` on `arena`.
/// @see prefixArgumentsWithU (contrast — prefixes ALL args),
///      replaceKeysScratch, isAdmittedIntegration.
static ScratchString prefixNonMarkerArgumentsWithUScratch(ScratchArena& arena, StrSpan expr);


        /// @brief Scratch-arena twin of @ref removeUPrefixFromArguments — strip a
        ///        leading `u_` from every bracketed argument, result on @p arena.
        ///
        /// @details
        /// Byte-identical to @ref removeUPrefixFromArguments: the same
        /// `getArgsSpans` + per-argument `u_`-prefix test feeding `StrReplacement`
        /// pairs whose values are the `arg[2:]` slices (no new bytes), then
        /// `replaceKeysScratch` in place of `replaceKeysToString`. Only the result
        /// type differs — a `ScratchString` on @p arena rather than a heap
        /// `std::string` — so a caller inside a `disintegrateExpr2` burst keeps the
        /// bytes off the malloc heap.
        ///
        /// @param arena Per-slot scratch arena receiving the result.
        /// @param expr  Canonical MPL expression span.
        /// @return The `u_`-stripped expression as a `ScratchString` on @p arena.
        /// @see removeUPrefixFromArguments, replaceKeysScratch
static ScratchString removeUPrefixScratch(ScratchArena& arena, const StrSpan& expr) {
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int argN = getArgsSpans(expr, args, ExecutionParameters::MAX_ARITY);
            StrReplacement pairs[ExecutionParameters::MAX_ARITY];
            int pairN = 0;
            for (int ai = 0; ai < argN; ++ai) {
                const StrSpan& arg = args[ai];
                if (arg.len >= 2 && arg.ptr[0] == 'u' && arg.ptr[1] == '_') {
                    pairs[pairN].key = arg;
                    pairs[pairN].value = StrSpan(arg.ptr + 2, arg.len - 2);
                    ++pairN;
                }
            }
            return replaceKeysScratch(arena, expr, pairs, pairN);
        }


        /// @brief Scratch-arena twin of @ref addMissingU — add a `u_` prefix to
        ///        every bracketed argument that lacks one (and is not `marker`),
        ///        result on @p arena.
        ///
        /// @details
        /// Byte-identical to @ref addMissingU. The original builds a
        /// `std::map<std::string,std::string>` of `arg -> "u_"+arg` and runs
        /// `ce::replaceKeysInString`; this twin builds `StrReplacement` pairs over
        /// `getArgsSpans` (each `"u_"+arg` value is NEW bytes built on @p arena)
        /// and runs `replaceKeysScratch` — the proven byte-exact twin of
        /// `ce::replaceKeysInString` (greedy-longest, token-boundary). The
        /// duplicate pairs a repeated argument would create are harmless (identical
        /// value), so the map's dedup is unnecessary.
        ///
        /// @param arena Per-slot scratch arena receiving the result and the
        ///              `"u_"+arg` replacement values.
        /// @param expr  Canonical MPL expression span.
        /// @return The `u_`-completed expression as a `ScratchString` on @p arena.
        /// @see addMissingU, replaceKeysScratch
static ScratchString addMissingUScratch(ScratchArena& arena, const StrSpan& expr) {
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int argN = getArgsSpans(expr, args, ExecutionParameters::MAX_ARITY);
            StrReplacement pairs[ExecutionParameters::MAX_ARITY];
            int pairN = 0;
            for (int ai = 0; ai < argN; ++ai) {
                const StrSpan& arg = args[ai];
                const bool startsU = arg.len >= 2 && arg.ptr[0] == 'u' && arg.ptr[1] == '_';
                const bool isMarker = arg.len == 6 && std::memcmp(arg.ptr, "marker", 6) == 0;
                if (!startsU && !isMarker) {
                    char* uval = arena.allocBytes(2 + arg.len);
                    uval[0] = 'u';
                    uval[1] = '_';
                    std::memcpy(uval + 2, arg.ptr, static_cast<size_t>(arg.len));
                    pairs[pairN].key = arg;
                    pairs[pairN].value = StrSpan(uval, 2 + arg.len);
                    ++pairN;
                }
            }
            return replaceKeysScratch(arena, expr, pairs, pairN);
        }


        /// @brief Scratch twin of @ref stripUPrefixAST — strip the `u_` prefix
        ///        from every bracketed token of a compound expression, on @p arena.
        ///
        /// @details
        /// The oracle parses @p expr, walks the tree collecting every node's
        /// `getArgs` tokens into a distinct set, keeps the `u_`-prefixed ones as a
        /// `renameMap` (`u_X -> X`), and applies `ce::replaceKeysInString`. This
        /// twin replaces the tree walk with `collectExprTokens` (the registered
        /// flat `[...]` scan proven to yield the same token SET — every variable
        /// token lives inside some `[...]`), dedups the `u_` tokens by a linear
        /// `equalSpans` scan, maps each to its `u_`-stripped INPUT SLICE, and runs
        /// `replaceKeysScratch` (the greedy-longest twin of
        /// `ce::replaceKeysInString`). Byte-identical: same distinct `u_` token
        /// set, same `(key, value)` pairs, and greedy-longest is key-order-
        /// independent (dup tokens collapse identically).
        ///
        /// @param arena Scratch arena receiving the result.
        /// @param expr  The compound expression span.
        /// @return The `u_`-stripped expression as a `ScratchString` on @p arena.
        /// @invariant Byte-identical to `stripUPrefixAST`; mints into no interner.
        /// @see stripUPrefixAST (the retained heap oracle), collectExprTokens,
        ///      replaceKeysScratch.
static ScratchString stripUPrefixASTScratch(ScratchArena& arena, StrSpan expr) {
            StrSpan uToks[ExecutionParameters::MAX_EXPR_TOKENS];
            int32_t n = 0;
            collectExprTokens(expr, [&](const StrSpan& tok) {
                if (!(tok.len >= 2 && tok.ptr[0] == 'u' && tok.ptr[1] == '_')) return;
                for (int32_t k = 0; k < n; ++k)
                    if (equalSpans(uToks[k], tok)) return;
                assert(n < ExecutionParameters::MAX_EXPR_TOKENS
                    && "stripUPrefixASTScratch: distinct u_ token count exceeds MAX_EXPR_TOKENS");
                uToks[n++] = tok;
            });
            StrReplacement pairs[ExecutionParameters::MAX_EXPR_TOKENS];
            for (int32_t k = 0; k < n; ++k) {
                pairs[k].key = uToks[k];
                pairs[k].value = StrSpan(uToks[k].ptr + 2, uToks[k].len - 2);
            }
            return replaceKeysScratch(arena, expr, pairs, n);
        }

private:
    /// @brief Find one configured expression without allocating a lookup key.
    /// @details The returned row is borrowed from the stable configuration map.
    /// @param core Expression name.
    /// @return Configuration row, or null for the defined no-match case.
    const ce::CoreExpressionConfig* coreConfig(const StrSpan& core) const;
    const ce::CoreExpressionMap& coreExpressionMap;
    const CompiledExpressionMap& compiledExpressions;
};
} // namespace gl

namespace gl {
/// @brief Scratch-arena twin of @ref makeMarkedExpr — replace @p var with
        ///        the literal "marker" in @p expr, result on @p arena.
        ///
        /// @details
        /// Byte-identical to @ref makeMarkedExpr: one `var -> "marker"`
        /// `StrReplacement` through `replaceKeysScratch` instead of
        /// `replaceKeysToString` ("marker" is a static literal slice, so no arena is
        /// needed for the value). Result is a `ScratchString` on @p arena.
        ///
        /// @param arena Per-slot scratch arena receiving the result.
        /// @param expr  Canonical MPL expression span.
        /// @param var   Variable name to mark.
        /// @return The marked expression as a `ScratchString` on @p arena.
        /// @see makeMarkedExpr, replaceKeysScratch
inline ScratchString DefinitionExpansion::makeMarkedExprScratch(ScratchArena& arena,
            const StrSpan& expr, const StrSpan& var) {
            const StrReplacement pairs[1] = { { var, StrSpan("marker", 6) } };
            return replaceKeysScratch(arena, expr, pairs, 1);
        }
}
