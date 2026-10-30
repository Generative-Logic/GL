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
/// @brief Out-of-line bodies for the larger `ce::` helpers declared in
/// `compiler.hpp` plus the recursive expression parser `parseExpr`.
///
/// @details
/// `compiler.hpp` is mostly inline templates and short helpers that the
/// compiler folds into hot-path callers. Three functions are kept
/// out-of-line because their bodies are larger than the call-site cost
/// of an indirect call:
///
/// - `generateBinarySequencesAsLists(n)` — produces 2ⁿ binary masks.
/// - `generateAllPermutations(n)`        — produces all permutations of
///   `0..n-1` grouped by fixed-point count.
/// - `parseExpr(s)` — recursive descent parser turning a canonical MPL
///   string into a `TreeNode1` tree.
///
/// Each of these has its full @brief / @details / @return contract
/// documented at the matching declaration site in `compiler.hpp`. The
/// brief reminders below point back at that documentation.

#include "compiler.hpp"
#include "compiler/expression_parser.hpp"

namespace ce {

    /// See compiler.hpp declaration for full documentation. Body
    /// pre-reserves `2^n` rows; iterates `mask` from `0` to `(1<<n) - 1`;
    /// each row is the big-endian bit list of `mask`.
    std::vector<std::vector<int>> generateBinarySequencesAsLists(int n) {
        std::vector<std::vector<int>> out;

        if (n < 0) {
            return out;
        }
        if (n == 0) {
            out.emplace_back();
            return out;
        }

        const std::size_t total = static_cast<std::size_t>(1ULL) << n;
        out.reserve(total);

        for (std::size_t mask = 0; mask < total; ++mask) {
            std::vector<int> row;
            row.reserve(static_cast<std::size_t>(n));
            for (int i = n - 1; i >= 0; --i) {
                int bit = static_cast<int>((mask >> i) & 1ULL);
                row.push_back(bit);
            }
            out.push_back(std::move(row));
        }
        return out;
    }

    /// See compiler.hpp declaration for full documentation. Body uses
    /// `std::next_permutation` over `0..i-1` for each `i ∈ [1..n]`,
    /// emitting permutations in lexicographic order.
    std::map<int, std::vector<std::vector<int>>> generateAllPermutations(int n) {
        std::map<int, std::vector<std::vector<int>>> allPermutations;

        if (n < 0) {
            return allPermutations;
        }

        allPermutations[0] = std::vector<std::vector<int>>();

        for (int i = 1; i <= n; ++i) {
            std::vector<int> current;
            current.reserve(static_cast<std::size_t>(i));
            for (int k = 0; k < i; ++k) {
                current.push_back(k);
            }

            std::vector<std::vector<int>> perms;
            do {
                perms.push_back(current);
            } while (std::next_permutation(current.begin(), current.end()));

            allPermutations[i] = perms;
        }

        return allPermutations;
    }

    /// @brief Recursive-descent parser turning a canonical MPL expression
    /// into a `TreeNode1` tree.
    ///
    /// @details
    /// Strips whitespace from the input (canonical MPL never carries
    /// whitespace; whitespace in the input usually means an upstream
    /// bug — this parser is defensive about it because some definition
    /// files have legacy whitespace). Then walks the string
    /// dispatching on the leading character:
    ///
    /// - `(>` …               quantifier node — args inside the `>[...]`
    ///   bound-variable list become the node label, then two child trees
    ///   (premise + body) are parsed recursively.
    /// - `(&` …               conjunction node — two child trees, no
    ///   bound vars.
    /// - `(<predicate>[...])` simple predicate — node label is the parens
    ///   contents.
    /// - `!(>` / `!(&` / `!(<predicate>[...])`
    ///                        negated forms of the same shapes.
    ///
    /// Each node propagates its `arguments` set upward (children's args
    /// minus locally-bound vars). Throws `std::runtime_error` on any
    /// well-formedness violation (unbalanced brackets, missing `]`/`)`).
    ///
    /// @param treeStrIn Canonical MPL expression text. Whitespace tolerated.
    /// @return Heap-allocated `TreeNode1*` rooting the parsed tree, or
    ///         `nullptr` on empty input. Caller frees via `deleteTree`.
    /// @pre  Brackets are balanced; every `[` has a matching `]`, every
    ///       `(` has a matching `)`.
    /// @post On success: returned tree's `arguments` set carries every
    ///       free variable.
    /// @see [`TreeNode1`](compiler.hpp#treenode1), `deleteTree`,
    ///      `nodeToStr`, `treeToExpr`.
    TreeNode1* parseExpr(const std::string& treeStrIn) {
        return gl::ExpressionParser::parse(treeStrIn);
    }

} // namespace ce
