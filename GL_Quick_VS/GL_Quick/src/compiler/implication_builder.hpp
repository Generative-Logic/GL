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

/// @brief Build implication syntax with the shared binder-placement rule.
/// @details Composes through definition interfaces without retaining a prover.
class ImplicationBuilder {
public:
/// @brief Build an implication with the shared free-variable binding rule.
///
/// @details Preserves expression order, binding and compiler name allocation.
/// @param key Ordered premise expressions.
/// @param value Consequent expression.
/// @return The result of the documented operation.
static std::string reconstructImplication(const std::vector<std::string>& key,
            const std::string& value) {
            return gl::ImplicationBuilder::reconstructImplicationFullBind(key, value);
        }
/// @brief Build a nested implication with binders at each first premise occurrence.
///
/// @details Preserves expression order, binding and compiler name allocation.
/// @param key Ordered premise expressions.
/// @param value Consequent expression.
/// @return The result of the documented operation.
static std::string reconstructImplicationFullBind(const std::vector<std::string>& key, const std::string& value) {
            // I-129: the former interior heap (chain / argsChain
            // vector-of-vectors, the removedArgs / placed std::set, whenRemoved, the per-link
            // joined / tmp strings) is fixed stack StrSpan arrays. Every argument name is a
            // substring of the stable inputs key / value, so the spans (from the zero-allocation
            // getArgsSpans) stay valid for the call; the sets become a sorted StrSpan array +
            // binary_search (membership byte-identical to std::set via compareSpans) plus a
            // parallel bool. The returned implication is the one std::string, built directly.
            const std::size_t keyCount = key.size();
            const std::size_t chainCount = keyCount + 1;  // key ++ [value]
            constexpr int kChainCap = 64;  // safe bound: the measured max chain is 14 (a 13-premise implication); 64 leaves wide margin
            constexpr int kArity = ExecutionParameters::MAX_ARITY;
            constexpr int kAllArgs = kChainCap * kArity;
            assert(chainCount <= static_cast<std::size_t>(kChainCap)
                && "reconstructImplicationFullBind: chain exceeds cap (64)");

            StrSpan argsChain[kChainCap][kArity];
            int argCount[kChainCap];
            for (std::size_t i = 0; i < chainCount; ++i) {
                const StrSpan src = (i < keyCount) ? StrSpan(key[i]) : StrSpan(value);
                argCount[i] = getArgsSpans(src, argsChain[i], kArity);
            }

            // removedArgs: sorted-unique set of the non-u_ args across the whole chain.
            StrSpan removedArgs[kAllArgs];
            int removedCount = 0;
            for (std::size_t i = 0; i < chainCount; ++i) {
                for (int j = 0; j < argCount[i]; ++j) {
                    const StrSpan arg = argsChain[i][j];
                    if (arg.len >= 2 && arg.ptr[0] == 'u' && arg.ptr[1] == '_') continue;
                    removedArgs[removedCount++] = arg;
                }
            }
            std::sort(removedArgs, removedArgs + removedCount,
                [](const StrSpan& a, const StrSpan& b) { return compareSpans(a, b) < 0; });
            removedCount = static_cast<int>(std::unique(removedArgs, removedArgs + removedCount,
                [](const StrSpan& a, const StrSpan& b) { return compareSpans(a, b) == 0; })
                - removedArgs);

            // placed: a parallel bool over removedArgs IS the std::set placed (placed only ever
            // holds removedArgs members), so first-occurrence dedup is O(1) by bsearch index.
            bool placed[kAllArgs];
            for (int i = 0; i < removedCount; ++i) placed[i] = false;

            // whenRemoved[idx] = bound vars first placed at premise idx (idx < keyCount).
            StrSpan whenRemoved[kChainCap][kArity];
            int whenRemovedCount[kChainCap];
            for (std::size_t i = 0; i < keyCount; ++i) whenRemovedCount[i] = 0;

            for (std::size_t idx = 0; idx < chainCount; ++idx) {
                for (int j = 0; j < argCount[idx]; ++j) {
                    const StrSpan arg = argsChain[idx][j];
                    const StrSpan* lo = std::lower_bound(removedArgs, removedArgs + removedCount,
                        arg, [](const StrSpan& a, const StrSpan& b) { return compareSpans(a, b) < 0; });
                    if (lo == removedArgs + removedCount || compareSpans(*lo, arg) != 0) continue;
                    const int ri = static_cast<int>(lo - removedArgs);
                    if (placed[ri]) continue;
                    placed[ri] = true;
                    if (idx < keyCount) whenRemoved[idx][whenRemovedCount[idx]++] = arg;
                }
            }

            std::string implication = value;
            for (std::size_t i = keyCount; i-- > 0; ) {
                int joinedLen = 0;
                for (int j = 0; j < whenRemovedCount[i]; ++j) {
                    if (j > 0) ++joinedLen;
                    joinedLen += whenRemoved[i][j].len;
                }
                std::string tmp;
                tmp.reserve(static_cast<std::size_t>(5 + joinedLen) + key[i].size() + implication.size());
                tmp += "(>[";
                for (int j = 0; j < whenRemovedCount[i]; ++j) {
                    if (j > 0) tmp.push_back(',');
                    tmp.append(whenRemoved[i][j].ptr, static_cast<std::size_t>(whenRemoved[i][j].len));
                }
                tmp += "]";
                tmp += key[i];
                tmp += implication;
                tmp += ")";
                implication.swap(tmp);
            }

            return implication;
        }
/// @brief Scratch-arena twin of @ref reconstructImplicationFullBind —
        ///        build the nested fully-bound implication on @p arena, not the
        ///        heap.
        ///
        /// @details
        /// The premise/binder analysis is identical to
        /// @ref reconstructImplicationFullBind (the same fixed stack `StrSpan`
        /// arrays via `getArgsSpans`, the sorted-unique `removedArgs`, the
        /// left-most-occurrence `whenRemoved` placement). Only the final assembly
        /// differs: where the original folds a growing `std::string`
        /// right-to-left, this twin computes the exact output length and fills one
        /// `arena` allocation in a single left-to-right pass — each premise's
        /// `(>[binders]premise` prefix in order, then the consequent, then one `)`
        /// per premise. The byte sequence is identical. The premise chain arrives
        /// as a `StrSpan` array so the caller's premise vector is also off the
        /// heap.
        ///
        /// @param arena    Per-slot scratch arena receiving the result.
        /// @param key      Premise spans (each a canonical MPL expression).
        /// @param keyCount Number of premises.
        /// @param value    Consequent span.
        /// @return The fully-bound `(>[...]...)` implication as a `ScratchString`.
        /// @invariant `keyCount + 1 <= 64` (the measured-max chain cap, asserted).
        /// @see reconstructImplicationFullBind
static ScratchString reconstructImplicationFullBindScratch(ScratchArena& arena,
            const StrSpan* key, int keyCount, const StrSpan& value) {
            const std::size_t keyN = static_cast<std::size_t>(keyCount);
            const std::size_t chainCount = keyN + 1;  // key ++ [value]
            constexpr int kChainCap = 64;
            constexpr int kArity = ExecutionParameters::MAX_ARITY;
            constexpr int kAllArgs = kChainCap * kArity;
            assert(chainCount <= static_cast<std::size_t>(kChainCap)
                && "reconstructImplicationFullBindScratch: chain exceeds cap (64)");

            StrSpan argsChain[kChainCap][kArity];
            int argCount[kChainCap];
            for (std::size_t i = 0; i < chainCount; ++i) {
                const StrSpan src = (i < keyN) ? key[i] : value;
                argCount[i] = getArgsSpans(src, argsChain[i], kArity);
            }

            StrSpan removedArgs[kAllArgs];
            int removedCount = 0;
            for (std::size_t i = 0; i < chainCount; ++i) {
                for (int j = 0; j < argCount[i]; ++j) {
                    const StrSpan arg = argsChain[i][j];
                    if (arg.len >= 2 && arg.ptr[0] == 'u' && arg.ptr[1] == '_') continue;
                    removedArgs[removedCount++] = arg;
                }
            }
            std::sort(removedArgs, removedArgs + removedCount,
                [](const StrSpan& a, const StrSpan& b) { return compareSpans(a, b) < 0; });
            removedCount = static_cast<int>(std::unique(removedArgs, removedArgs + removedCount,
                [](const StrSpan& a, const StrSpan& b) { return compareSpans(a, b) == 0; })
                - removedArgs);

            bool placed[kAllArgs];
            for (int i = 0; i < removedCount; ++i) placed[i] = false;

            StrSpan whenRemoved[kChainCap][kArity];
            int whenRemovedCount[kChainCap];
            for (std::size_t i = 0; i < keyN; ++i) whenRemovedCount[i] = 0;

            for (std::size_t idx = 0; idx < chainCount; ++idx) {
                for (int j = 0; j < argCount[idx]; ++j) {
                    const StrSpan arg = argsChain[idx][j];
                    const StrSpan* lo = std::lower_bound(removedArgs, removedArgs + removedCount,
                        arg, [](const StrSpan& a, const StrSpan& b) { return compareSpans(a, b) < 0; });
                    if (lo == removedArgs + removedCount || compareSpans(*lo, arg) != 0) continue;
                    const int ri = static_cast<int>(lo - removedArgs);
                    if (placed[ri]) continue;
                    placed[ri] = true;
                    if (idx < keyN) whenRemoved[idx][whenRemovedCount[idx]++] = arg;
                }
            }

            // One-pass exact-length build: prefix_0 .. prefix_{keyN-1} value )^keyN,
            // where prefix_i = "(>[" + join(whenRemoved[i], ",") + "]" + key[i].
            // Byte-identical to the original's right-to-left std::string fold.
            int32_t total = 0;
            for (std::size_t i = 0; i < keyN; ++i) {
                total += 3;  // "(>["
                for (int j = 0; j < whenRemovedCount[i]; ++j) {
                    if (j > 0) ++total;  // ','
                    total += whenRemoved[i][j].len;
                }
                total += 1;  // "]"
                total += key[i].len;
            }
            total += value.len;
            total += static_cast<int32_t>(keyN);  // one ')' per premise
            if (total == 0) return ScratchString();  // empty value, no premises

            char* buf = arena.allocBytes(total);
            int32_t at = 0;
            for (std::size_t i = 0; i < keyN; ++i) {
                buf[at++] = '('; buf[at++] = '>'; buf[at++] = '[';
                for (int j = 0; j < whenRemovedCount[i]; ++j) {
                    if (j > 0) buf[at++] = ',';
                    std::memcpy(buf + at, whenRemoved[i][j].ptr,
                        static_cast<size_t>(whenRemoved[i][j].len));
                    at += whenRemoved[i][j].len;
                }
                buf[at++] = ']';
                std::memcpy(buf + at, key[i].ptr, static_cast<size_t>(key[i].len));
                at += key[i].len;
            }
            std::memcpy(buf + at, value.ptr, static_cast<size_t>(value.len));
            at += value.len;
            for (std::size_t i = 0; i < keyN; ++i) buf[at++] = ')';
            assert(at == total);
            return ScratchString::wrap(arena, buf, total);
        }

private:
};
} // namespace gl
