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

namespace gl {
/// @brief Build normalized integer keys and resume their prefix construction.
/// @details Owns only the original fixed scratch arrays and counters. Output buffers
/// remain caller-owned; all construction operations retain their inline boundary.
class HashKeyBuilder {
public:
    /// @brief An opaque checkpoint of the three prefix counters.
    /// @details The scratch arrays are not copied. Restore only while this builder
    /// and its output buffer still contain the checkpoint's prefix mappings.
    class Checkpoint {
        friend class HashKeyBuilder;
        NameId variables_;
        NameId next_;
        NameId length_;
        /// @brief Capture the existing prefix counters.
        /// @details Only the owning key builder constructs checkpoints.
        /// @param variables Number of assigned variable slots.
        /// @param next Next normalized variable identifier.
        /// @param length Number of output slots already written.
        /// @return Constructed checkpoint.
        Checkpoint(NameId variables, NameId next, NameId length)
            : variables_(variables), next_(next), length_(length) {}
    };

    /// @brief Start an empty incremental key.
    /// @details Initializes the three counters without clearing the scratch arrays.
    /// @return Constructed key builder.
    HashKeyBuilder() { reset(); }

    /// @brief Start another empty key using the same scratch arrays.
    /// @details Previous array contents become unreachable through the zero variable count.
    /// @return None.
    void reset() { state_.nVars = 0; state_.nextNormId = 1; state_.pos = 0; }

    /// @brief Append one expression to the current normalized key.
    /// @details Uses the existing first-occurrence identifier assignment and slot format.
    /// @param expression Encoded expression to append.
    /// @param output Caller-owned buffer containing the current prefix.
    /// @param capacity Buffer capacity in NameId slots.
    /// @return None.
    /// @invariant Appending allocates no heap or arena storage.
    void append(const IntEncodedExpr& expression, NameId* output, NameId capacity) {
        appendExprToIntNormalizedKey(expression, state_, output, capacity);
    }

    /// @brief Build a complete prefix and retain its variable assignments.
    /// @details Resets the counters and appends the supplied expressions in order.
    /// @param expressions Ordered expression pointers.
    /// @param count Number of expressions.
    /// @param output Caller-owned output buffer.
    /// @param capacity Buffer capacity in NameId slots.
    /// @return None.
    void buildPrefix(const IntEncodedExpr* const* expressions, NameId count,
                     NameId* output, NameId capacity) {
        buildIntNormalizedKeyPrefix(expressions, count, output, capacity, state_);
    }

    /// @brief Read the number of written key slots.
    /// @details This is the length of the prefix in the caller's output buffer.
    /// @return Written slot count.
    NameId length() const { return state_.pos; }

    /// @brief Capture the current prefix counters without copying its arrays.
    /// @details The checkpoint belongs to this builder and the current output prefix.
    /// @return Opaque prefix checkpoint.
    Checkpoint checkpoint() const { return Checkpoint(state_.nVars, state_.nextNormId, state_.pos); }

    /// @brief Restore the counters before appending another candidate to a prefix.
    /// @details Retains the existing prefix mappings in the scratch arrays and buffer.
    /// @param prefix Checkpoint of this builder's still-present prefix mappings.
    /// @return None.
    void restore(const Checkpoint& prefix) {
        state_.nVars = prefix.variables_;
        state_.nextNormId = prefix.next_;
        state_.pos = prefix.length_;
    }

    /// @brief Visit every operator-name-sorted ordering of a premise list,
    ///        in the lexicographic order of the orderings.
    ///
    /// @details
    /// An ordering is a permutation of the premise indices `0..n-1` whose
    /// operator names never decrease (`compareSpans`; equal names in any
    /// order). These are exactly the permutations of the permutation table
    /// (`generateAllPermutations`, lexicographic) that a whole-key install
    /// keeps, in the same order, without enumerating the other `n!`
    /// permutations: a depth-first walk takes the unused indices in ascending
    /// order and drops a branch as soon as the next name is smaller than the
    /// previous one. Zero premises have no ordering, as the table's empty
    /// entry for zero holds none. Stack storage only.
    ///
    /// @tparam Visit Callable `void(const int32_t* ordering)`; the ordering
    ///         holds `n` indices and is valid during the call.
    /// @param names Operator name of each premise, by premise index.
    /// @param n     Number of premises (`<= MAX_ADMISSION_KEY_ELEMENTS`).
    /// @param visit Consumer of each sorted ordering.
    /// @return Nothing.
    /// @see forEachSortedPrefix, generateAllPermutations
    template <typename Visit>
    static void forEachSortedOrdering(const StrSpan* names, int32_t n, Visit visit) {
        assert(n >= 0 && n <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
            && "forEachSortedOrdering: premise count exceeds the key cap");
        if (n == 0) return;
        forEachSortedPrefix(names, n,
            [&](int32_t depth, const int32_t* ordering) {
                if (depth + 1 == n) visit(ordering);
            },
            [](int32_t) {});
    }

    /// @brief Walk the operator-name-sorted prefixes of a premise list in
    ///        preorder, each distinct prefix once.
    ///
    /// @details
    /// A prefix is a sequence of distinct premise indices whose operator
    /// names never decrease. The walk takes the unused indices in ascending
    /// order and calls @p enter when it extends a prefix by one premise and
    /// @p leave when it backtracks over that premise. The preorder is the
    /// order in which a scan of the lexicographic permutation table meets
    /// each sorted prefix for the first time (every permutation's prefixes
    /// up to the first decreasing name): a prefix comes before its
    /// extensions, and of two prefixes that first differ at one position the
    /// one with the smaller index there comes first, as its lexicographically
    /// smallest permutation does. The table scan meets a prefix once per
    /// permutation that starts with it; the walk meets it once. Stack
    /// storage only.
    ///
    /// @tparam Enter Callable `void(int32_t depth, const int32_t* prefix)`;
    ///         the prefix holds `depth + 1` indices, the new one last.
    /// @tparam Leave Callable `void(int32_t depth)`, called after the subtree
    ///         of the premise entered at @p depth.
    /// @param names Operator name of each premise, by premise index.
    /// @param n     Number of premises (`<= MAX_ADMISSION_KEY_ELEMENTS`).
    /// @param enter Called on each new prefix.
    /// @param leave Called when the walk leaves a prefix.
    /// @return Nothing.
    /// @see forEachSortedOrdering, checkpoint, restore
    template <typename Enter, typename Leave>
    static void forEachSortedPrefix(const StrSpan* names, int32_t n, Enter enter, Leave leave) {
        constexpr int32_t kCap = ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS;
        assert(n >= 0 && n <= kCap
            && "forEachSortedPrefix: premise count exceeds the key cap");
        int32_t prefix[kCap];
        bool used[kCap] = {};
        // next[d] = the next candidate index to try at depth d.
        int32_t next[kCap + 1];
        int32_t depth = 0;
        next[0] = 0;
        while (depth >= 0) {
            int32_t candidate = next[depth];
            while (candidate < n
                   && (used[candidate]
                       || (depth > 0 && compareSpans(names[prefix[depth - 1]], names[candidate]) > 0)))
                ++candidate;
            if (candidate == n) {
                // Every candidate at this depth is done: backtrack.
                --depth;
                if (depth >= 0) {
                    leave(depth);
                    used[prefix[depth]] = false;
                }
                continue;
            }
            next[depth] = candidate + 1;
            prefix[depth] = candidate;
            used[candidate] = true;
            enter(depth, prefix);
            if (depth + 1 < n) {
                ++depth;
                next[depth] = 0;
            } else {
                leave(depth);
                used[candidate] = false;
            }
        }
    }

/// @brief Build an `IntNormalizedKey` from a pre-encoded
        /// `IntEncodedExpr` pointer array — no `nm.encode()` calls,
        /// reads `NameId` fields directly.
        ///
        /// @details
        /// Hot-path key builder used by the static request pipeline.
        /// Layout matches `IntNormalizedKey`'s documented format:
        ///
        ///   [expressionID] [negation] [varID, changeable] x arity
        ///
        /// Always uses `ignoreU=false` semantics: all args are
        /// normalized to sequential ids in first-appearance order with
        /// `changeable=0`. Unchangeable args are distinguished from
        /// changeable by their `u_`-prefixed `argId` field (set in
        /// `encodeExpression`), not by an `ignoreU` flag here.
        ///
        /// @param exprs        Array of `IntEncodedExpr*` to fold into the
        ///                     key, in source order.
        /// @param count        Length of `exprs`.
        /// @param outBuf       Caller-owned `NameId` buffer for the
        ///                     key bytes. Must be at least `bufCapacity`
        ///                     wide.
        /// @param bufCapacity  Capacity of `outBuf`. Must be ≥ the worst-
        ///                     case key length for `count` expressions.
        /// @return Number of `NameId` slots written into `outBuf`.
        /// @pre  Each `exprs[i]` is a valid `IntEncodedExpr` minted by
        ///       `encodeExpression`.
        /// @post Caller wraps the returned bytes in an `IntNormalizedKey`
        ///       — typically copying through the gen scratch arena byte-bump store to
        ///       guarantee pointer stability.
        /// @see [`IntNormalizedKey`](memory.hpp#intnormalizedkey) — output
        ///      type.
        /// @see `makeIntNormalizedKey` — string-input counterpart.
        /// @see `buildIntNormalizedKeyPrefix` + `appendExprToIntNormalizedKey` —
        ///      the resumable form the grow search uses; same bytes, built one
        ///      premise at a time.
static NameId makeIntNormalizedKeyFromEncoded(
            const IntEncodedExpr* const* exprs, NameId count,
            NameId* outBuf, NameId bufCapacity) {

            NormKeyBuildState state;
            buildIntNormalizedKeyPrefix(exprs, count, outBuf, bufCapacity, state);
            return state.pos;
        }

/// @brief Build a normalized integer key and its reverse argument mapping.
///
/// @details Uses argument modes and uses caller-owned output storage.
/// @param exprs Ordered encoded expressions.
/// @param count Number of expressions.
/// @param unchangeableArgIds Arguments retaining their fixed-argument mode.
/// @param outBuf Caller-owned normalized-key buffer.
/// @param bufCapacity Output buffer capacity in NameId slots.
/// @param reverseMap Receives the original argument identifier for each normalized slot.
/// @param numNormVars Receives the number of normalized variable slots.
/// @return Number of written output slots.
/// Build IntNormalizedKey from IntEncodedExpr with ignoreU=true semantics
        /// and a set of arg IDs to additionally mark as unchangeable.
        /// Produces reverseMap[normId] = argFullId for back-substitution on hit.
        /// unchangeableArgIds = remaining-args candidate set (already NameId encoded).
static NameId makeIntNormalizedKeyFromEncodedWithMap(
            const IntEncodedExpr* const* exprs, NameId count,
            const std::set<NameId>& unchangeableArgIds,
            NameId* outBuf, NameId bufCapacity,
            NameId* reverseMap, NameId& numNormVars) {

            NameId varIds[ExecutionParameters::MAX_KEY_SLOTS];
            NameId normIds[ExecutionParameters::MAX_KEY_SLOTS];
            NameId nVars = 0;
            NameId nextNormId = 1;
            NameId pos = 0;
            numNormVars = 0;

            for (NameId i = 0; i < count; ++i) {
                const IntEncodedExpr& e = *exprs[i];
                assert(pos + 2 <= bufCapacity);
                outBuf[pos++] = e.nameId;
                outBuf[pos++] = e.negation;

                for (NameId j = 0; j < e.arity; ++j) {
                    assert(pos + 2 <= bufCapacity);
                    // ignoreU=true: use argFullId (raw, not u_-prefixed) for ALL args
                    NameId rawId = e.argFullId[j];
                    bool isUnch = (e.argUnchangeable[j] != 0) ||
                                  (unchangeableArgIds.find(rawId) != unchangeableArgIds.end());

                    if (isUnch) {
                        // Unchangeable: write raw ID as-is, changeable=1
                        outBuf[pos++] = rawId;
                        outBuf[pos++] = 1;
                    } else {
                        // Changeable: normalize to sequential ID
                        NameId normId = 0;
                        for (NameId v = 0; v < nVars; ++v) {
                            if (varIds[v] == rawId) { normId = normIds[v]; break; }
                        }
                        if (normId == 0) {
                            normId = nextNormId++;
                            varIds[nVars] = rawId;
                            normIds[nVars] = normId;
                            reverseMap[normId] = rawId;
                            ++nVars;
                        }
                        outBuf[pos++] = normId;
                        outBuf[pos++] = 0;
                    }
                }
            }
            numNormVars = nVars;
            return pos;
        }

/// @brief Read the remaining argument spans in their established order.
///
/// @details Uses argument modes and uses caller-owned output storage.
/// @param exprs Ordered encoded expressions.
/// @param count Number of expressions.
/// @param ignoreU Whether argument normalization ignores the unchangeable flag.
/// @param outBuf Caller-owned normalized-key buffer.
/// @param bufCapacity Output buffer capacity in NameId slots.
/// @return Number of written output slots.
/// Build IntNormalizedKey from pre-encoded IntEncodedExpr array.
        /// No string ops, no NameMap::encode calls — all IDs already resolved.
static NameId makeIntNormalizedKeyFromEncoded(
            const IntEncodedExpr* exprs, NameId count,
            bool ignoreU,
            NameId* outBuf, NameId bufCapacity) {

            NameId varIds[ExecutionParameters::MAX_KEY_SLOTS];
            NameId normIds[ExecutionParameters::MAX_KEY_SLOTS];
            NameId nVars = 0;
            NameId nextNormId = 1;
            NameId pos = 0;

            for (NameId ei = 0; ei < count; ++ei) {
                const IntEncodedExpr& expr = exprs[ei];
                assert(pos + 2 <= bufCapacity);
                outBuf[pos++] = expr.nameId;
                outBuf[pos++] = expr.negation;

                for (NameId j = 0; j < expr.arity; ++j) {
                    assert(pos + 2 <= bufCapacity);
                    bool isUnchangeable = (expr.argUnchangeable[j] != 0);
                    // argId has u_-prefixed ID for unchangeable args (for ignoreU=false normalization)
                    // argFullId has raw ID (for ignoreU=true literal passthrough)
                    NameId varId = (ignoreU && isUnchangeable) ? expr.argFullId[j] : expr.argId[j];

                    if (ignoreU && isUnchangeable) {
                        outBuf[pos++] = varId;
                        outBuf[pos++] = 1;
                    }
                    else {
                        NameId normId = 0;
                        for (NameId v = 0; v < nVars; ++v) {
                            if (varIds[v] == varId) { normId = normIds[v]; break; }
                        }
                        if (normId == 0) {
                            normId = nextNormId++;
                            varIds[nVars] = varId;
                            normIds[nVars] = normId;
                            ++nVars;
                        }
                        outBuf[pos++] = normId;
                        outBuf[pos++] = 0;
                    }
                }
            }
            return pos;
        }

/// @brief Build a normalized integer key and its reverse argument mapping.
///
/// @details Uses argument modes and uses caller-owned output storage.
/// @param exprs Ordered encoded expressions.
/// @param count Number of expressions.
/// @param ignoreU Whether argument normalization ignores the unchangeable flag.
/// @param outBuf Caller-owned normalized-key buffer.
/// @param bufCapacity Output buffer capacity in NameId slots.
/// @param reverseMap Receives the original argument identifier for each normalized slot.
/// @param numNormVars Receives the number of normalized variable slots.
/// @return Number of written output slots.
/// WithMap variant: also produces reverseMap[normId] = original varId.
static NameId makeIntNormalizedKeyFromEncodedWithMap(
            const IntEncodedExpr* exprs, NameId count,
            bool ignoreU,
            NameId* outBuf, NameId bufCapacity,
            NameId* reverseMap, NameId& numNormVars) {

            NameId varIdsL[ExecutionParameters::MAX_KEY_SLOTS];
            NameId normIdsL[ExecutionParameters::MAX_KEY_SLOTS];
            NameId nVars = 0;
            NameId nextNormId = 1;
            NameId pos = 0;

            for (NameId ei = 0; ei < count; ++ei) {
                const IntEncodedExpr& expr = exprs[ei];
                assert(pos + 2 <= bufCapacity);
                outBuf[pos++] = expr.nameId;
                outBuf[pos++] = expr.negation;

                for (NameId j = 0; j < expr.arity; ++j) {
                    assert(pos + 2 <= bufCapacity);
                    bool isUnchangeable = (expr.argUnchangeable[j] != 0);
                    NameId varId = (ignoreU && isUnchangeable) ? expr.argFullId[j] : expr.argId[j];

                    if (ignoreU && isUnchangeable) {
                        outBuf[pos++] = varId;
                        outBuf[pos++] = 1;
                    }
                    else {
                        NameId normId = 0;
                        for (NameId v = 0; v < nVars; ++v) {
                            if (varIdsL[v] == varId) { normId = normIdsL[v]; break; }
                        }
                        if (normId == 0) {
                            normId = nextNormId++;
                            varIdsL[nVars] = varId;
                            normIdsL[nVars] = normId;
                            ++nVars;
                            reverseMap[normId] = varId;
                        }
                        outBuf[pos++] = normId;
                        outBuf[pos++] = 0;
                    }
                }
            }
            numNormVars = static_cast<NameId>(nextNormId - 1);
            return pos;
        }

/// @brief Collect the distinct `u_`-stripped argument tails of an
        ///        admission key into a caller span buffer, sorted-unique — 0% heap.
        ///
        /// @details
        /// The span twin of the former `std::set<std::string>` collector: over
        /// every `key[i]`, `getArgsSpans` slices the flat `[...]` args; each arg
        /// that begins `u_` (asserted to contain exactly one `u_`) contributes
        /// its `u_`-stripped tail `StrSpan(arg.ptr+2, arg.len-2)` — a slice of
        /// `key[i]` (input lifetime). The gather buffer is `compareSpans`-sorted
        /// and adjacent-deduplicated, reproducing the former
        /// `std::set<std::string>` iteration order byte-for-byte, and the distinct
        /// run is copied into @p out. No arena is reached; @p key is a
        /// caller-owned container, so the function constructs nothing owning.
        ///
        /// @param key Admission-key elements.
        /// @param out Caller buffer receiving the distinct sorted stripped tails.
        /// @param cap Capacity of @p out (loud Rule-19 assert on overflow).
        /// @return The number of distinct stripped tails written to @p out.
        /// @invariant Allocates nothing; output spans alias @p key.
        /// @see getArgsSpans, compareSpans, equalSpans.
static int32_t getRemainingArgs(const std::vector<std::string>& key,
                                        StrSpan* out, int32_t cap) {
            // Gather each u_-stripped tail span (slice of key[i], input lifetime).
            StrSpan buf[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
            int32_t bufN = 0;
            for (std::size_t i = 0; i < key.size(); ++i) {
                StrSpan sp[ExecutionParameters::MAX_ARITY];
                const int32_t n = getArgsSpans(StrSpan(key[i]), sp,
                                               ExecutionParameters::MAX_ARITY);
                for (int32_t j = 0; j < n; ++j) {
                    // Only process arguments starting with "u_"
                    if (sp[j].len >= 2 && sp[j].ptr[0] == 'u' && sp[j].ptr[1] == '_') {
                        // Count "u_" occurrences to ensure exactly one (at the start).
                        int uCount = 0;
                        for (int32_t p = 0; p + 1 < sp[j].len; ++p)
                            if (sp[j].ptr[p] == 'u' && sp[j].ptr[p + 1] == '_') ++uCount;
                        assert(uCount == 1 && "Argument starting with 'u_' must not contain additional 'u_' substrings");

                        assert(bufN < ExecutionParameters::MAX_ADMISSION_REM_ARGS
                            && "getRemainingArgs: stripped-arg count exceeds cap");
                        buf[bufN++] = StrSpan(sp[j].ptr + 2, sp[j].len - 2);  // strip u_
                    }
                }
            }

            // Sort + adjacent-dedup -> the former std::set<std::string> order.
            std::sort(buf, buf + bufN, [](const StrSpan& a, const StrSpan& b) {
                return compareSpans(a, b) < 0;
            });
            int32_t outN = 0;
            for (int32_t i = 0; i < bufN; ++i) {
                if (i == 0 || !equalSpans(buf[i], buf[i - 1])) {
                    assert(outN < cap
                        && "getRemainingArgs: distinct arg count exceeds out cap");
                    out[outN++] = buf[i];
                }
            }
            return outN;
        }

/// @brief Span-run overload of @ref getRemainingArgs — @p key as a
        ///        `StrSpan` run, no owning `std::vector<std::string>`.
        ///
        /// @details Byte-identical to the `std::vector<std::string>` form: reads
        /// each element only through `getArgsSpans`, gathers the `u_`-stripped
        /// tails (slices of @p key, input lifetime), `compareSpans`-sorts and
        /// adjacent-dedups into @p out. Allocates nothing.
        ///
        /// @param key  Admission-key element spans.
        /// @param keyN The element count.
        /// @param out  Caller buffer receiving the distinct sorted stripped tails.
        /// @param cap  Capacity of @p out (loud Rule-19 assert on overflow).
        /// @return The number of distinct stripped tails written to @p out.
        /// @see getRemainingArgs(const std::vector<std::string>&, StrSpan*, int32_t).
static int32_t getRemainingArgs(const StrSpan* key, int32_t keyN,
                                        StrSpan* out, int32_t cap) {
            StrSpan buf[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
            int32_t bufN = 0;
            for (int32_t i = 0; i < keyN; ++i) {
                StrSpan sp[ExecutionParameters::MAX_ARITY];
                const int32_t n = getArgsSpans(key[i], sp,
                                               ExecutionParameters::MAX_ARITY);
                for (int32_t j = 0; j < n; ++j) {
                    if (sp[j].len >= 2 && sp[j].ptr[0] == 'u' && sp[j].ptr[1] == '_') {
                        int uCount = 0;
                        for (int32_t p = 0; p + 1 < sp[j].len; ++p)
                            if (sp[j].ptr[p] == 'u' && sp[j].ptr[p + 1] == '_') ++uCount;
                        assert(uCount == 1 && "Argument starting with 'u_' must not contain additional 'u_' substrings");
                        assert(bufN < ExecutionParameters::MAX_ADMISSION_REM_ARGS
                            && "getRemainingArgs: stripped-arg count exceeds cap");
                        buf[bufN++] = StrSpan(sp[j].ptr + 2, sp[j].len - 2);
                    }
                }
            }
            std::sort(buf, buf + bufN, [](const StrSpan& a, const StrSpan& b) {
                return compareSpans(a, b) < 0;
            });
            int32_t outN = 0;
            for (int32_t i = 0; i < bufN; ++i) {
                if (i == 0 || !equalSpans(buf[i], buf[i - 1])) {
                    assert(outN < cap
                        && "getRemainingArgs: distinct arg count exceeds out cap");
                    out[outN++] = buf[i];
                }
            }
            return outN;
        }

private:
struct NormKeyBuildState {
            NameId varIds[ExecutionParameters::MAX_KEY_SLOTS];
            NameId normIds[ExecutionParameters::MAX_KEY_SLOTS];
            NameId nVars;
            NameId nextNormId;
            NameId pos;
        };
NormKeyBuildState state_;

/// @brief Fold one expression onto the end of an in-progress normalized
        ///        key, minting slots only for arguments the state has not seen.
        ///
        /// @details
        /// The single per-expression body of the whole key language. Every key
        /// this class builds from `IntEncodedExpr` records goes through here, so
        /// the whole-candidate builder and the grow search's resume path cannot
        /// drift apart.
        ///
        /// Writes `[nameId] [negation]` and then `[slot] [0]` per argument, which
        /// is `IntNormalizedKey`'s documented layout. Uses `ignoreU=false`
        /// semantics throughout: every argument is renumbered, and unchangeable
        /// arguments are told apart by their `u_`-prefixed `argId` (set in
        /// `encodeExpression`), not by a flag here.
        ///
        /// @param e            Expression to fold in.
        /// @param state        Running build state; updated in place.
        /// @param outBuf       Caller-owned key buffer, already holding
        ///                     `state.pos` slots.
        /// @param bufCapacity  Capacity of @p outBuf in `NameId` slots.
        /// @pre  @p state was produced by `buildIntNormalizedKeyPrefix` or by an
        ///       earlier call to this function on the same @p outBuf.
        /// @post `state.pos` advanced by `2 + 2 * e.arity`; `state.nVars` and
        ///       `state.nextNormId` advanced by the number of newly seen
        ///       arguments.
        /// @invariant Allocates nothing; the state's arrays are the only scratch.
        /// @see `NormKeyBuildState` — the record this advances.
static void appendExprToIntNormalizedKey(const IntEncodedExpr& e,
                                                        NormKeyBuildState& state,
                                                        NameId* outBuf,
                                                        NameId bufCapacity) {
            assert(state.pos + 2 <= bufCapacity);
            assert(state.nextNormId == state.nVars + 1
                && "normalized-key slots are minted one per newly seen argument");
            outBuf[state.pos++] = e.nameId;
            outBuf[state.pos++] = e.negation;

            for (NameId j = 0; j < e.arity; ++j) {
                assert(state.pos + 2 <= bufCapacity);
                // ignoreU=false: normalize ALL args (u_-prefixed argId distinguishes unchangeable)
                const NameId varId = e.argId[j];
                NameId normId = 0;
                for (NameId v = 0; v < state.nVars; ++v) {
                    if (state.varIds[v] == varId) { normId = state.normIds[v]; break; }
                }
                if (normId == 0) {
                    normId = state.nextNormId++;
                    state.varIds[state.nVars] = varId;
                    state.normIds[state.nVars] = normId;
                    ++state.nVars;
                }
                outBuf[state.pos++] = normId;
                outBuf[state.pos++] = 0;
            }
        }

/// @brief Build the normalized key of a premise prefix and leave the
        ///        build resumable.
        ///
        /// @details
        /// Identical in output to `makeIntNormalizedKeyFromEncoded` over the same
        /// premises — it is the same loop — but it hands back the state instead
        /// of discarding it, so a caller that will append one premise at a time
        /// can pay for the prefix once. The grow search calls this once per stack
        /// pop and `appendExprToIntNormalizedKey` once per position, rewinding
        /// `nVars` / `nextNormId` / `pos` to the values recorded here in between.
        ///
        /// @param exprs        Premise pointers, in key order.
        /// @param count        Prefix length; `0` is legal and yields an empty key.
        /// @param outBuf       Caller-owned key buffer.
        /// @param bufCapacity  Capacity of @p outBuf in `NameId` slots.
        /// @param state        Out-parameter; overwritten, never read on entry.
        /// @post `outBuf[0 .. state.pos)` is byte-identical to what
        ///       `makeIntNormalizedKeyFromEncoded(exprs, count, ...)` writes.
        /// @invariant Allocates nothing.
        /// @see `appendExprToIntNormalizedKey` — the resume step.
static void buildIntNormalizedKeyPrefix(const IntEncodedExpr* const* exprs,
                                                       NameId count,
                                                       NameId* outBuf,
                                                       NameId bufCapacity,
                                                       NormKeyBuildState& state) {
            state.nVars = 0;
            state.nextNormId = 1;
            state.pos = 0;
            for (NameId i = 0; i < count; ++i)
                appendExprToIntNormalizedKey(*exprs[i], state, outBuf, bufCapacity);
        }
};
} // namespace gl
