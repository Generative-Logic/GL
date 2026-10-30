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
#include "block_key_reader.hpp"
namespace gl {
/// @brief Bind inputs for block key reader.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @return None; construction binds the supplied references.
BlockKeyReader::BlockKeyReader() {}

/// @brief Compute the *global key* for a memory block — the canonical
/// vector identifying its position in the LB tree.
///
/// @details
/// The global key is a sequence of `exprKey` values from the LB up to
/// the root sentinel, in deepest-first order. Used as a `std::map` key
/// in cross-LB tables (e.g. proof-stack caches) where a `Memory*`
/// pointer alone would not survive teardown / re-creation of the LB.
///
/// @param memoryBlock LB to compute the key for.
/// @return Vector of `exprKey` values from this LB up to the root,
///         deepest first. Returns an empty vector for the root sentinel
///         itself (its `exprKey` is empty by definition).
/// @see The full-LB-chain trace requirement for any debug trap that
///      identifies a target LB; this function returns exactly the
///      chain such traps must match.
std::vector<std::string> BlockKeyReader::getGlobalKey(const Memory& memoryBlock) {
    std::vector<std::string> globalKey;
    const Memory* current = &memoryBlock;
    while (true) {
        if (!current->exprKey().empty()) {
            globalKey.push_back(current->exprKey());
        }
        if (current->parentMemory != NULL) {
            current = current->parentMemory;
        }
        else {
            break;
        }
    }
    std::reverse(globalKey.begin(), globalKey.end());
    return globalKey;
}

/// @brief Compute the *global key* for a memory block into a
/// caller-owned span array — the 0%-heap production form.
///
/// @details
/// Walks `parentMemory` to the root sentinel, writing each non-empty
/// `exprKeyView()` span (a zero-copy view over the process-wide
/// `skeletonInterner`, never deloaded and frozen during the parallel
/// phase, so no walk mints it — I-97) into `out` in deepest-first
/// order, then reverses the filled range in place. The spans alias
/// the interner's cold bytes and inherit its lifetime; they stay
/// valid until the next mint into `skeletonInterner`, which does not
/// happen on the discharge path. Byte-identical to the retained
/// `std::vector<std::string>` oracle above (`exprKeyView` bytes ==
/// `exprKey`). Body in `logical_block/block_key_reader.cpp`.
///
/// @param memoryBlock LB to compute the key for.
/// @param out         Caller-owned span array of capacity `cap`;
///                    filled `[0, n)` with the reversed key spans.
/// @param cap         Capacity of `out`; a loud Rule-19 assert fires
///                    if the LB chain is deeper than `cap`
///                    (STOP-widen at FTA scale).
/// @return The number of spans written (`n <= cap`).
/// @see The `std::vector<std::string>` oracle overload above.
int32_t BlockKeyReader::getGlobalKey(const Memory& memoryBlock,
                                         StrSpan* out, int32_t cap) {
    int32_t n = 0;
    const Memory* current = &memoryBlock;
    while (true) {
        const StrSpan k = current->exprKeyView();
        if (k.len != 0) {
            assert(n < cap && "getGlobalKey: global key exceeds caller capacity");
            out[n++] = k;
        }
        if (current->parentMemory != NULL) {
            current = current->parentMemory;
        }
        else {
            break;
        }
    }
    std::reverse(out, out + n);
    return n;
}
}
