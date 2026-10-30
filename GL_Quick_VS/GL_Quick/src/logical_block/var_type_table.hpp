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
#include "../parameters.hpp"
#include "../memory_infra/str_ops.hpp"
namespace gl {
/// @brief Union-find table of one definition-set inference: every
        ///        distinct variable name is a node, every group carries at
        ///        most one definition-set label.
        ///
        /// @details
        /// The typing rule of `=` (I-267): a slot of any
        /// other operator FIXES its argument's label, while `=` only JOINS its
        /// two arguments. `fix` records a label on a node's group; `join`
        /// merges two groups. A group with two different labels is a type
        /// conflict and asserts (Rule 19) — a theorem or definition that joins
        /// a set with a number is a bug to see at its origin. An unfixed group
        /// resolves to `=`'s configured label, the starting point
        /// (`ScopeTypeResolver::resolvedLabel`).
        ///
        /// Names and labels are `StrSpan`s into caller-stable bytes (the typed
        /// leaves, the compiled registry, the config), so the table owns
        /// nothing. It is a fixed stack array of `MAX_TYPED_VARIABLES` nodes;
        /// the add path asserts at the ceiling. The root of a merge is the
        /// smaller node index, so the grouping depends only on the order of
        /// the typed leaves, never on hashing or allocation.
        ///
        /// @invariant `I-267`; allocates nothing.
        /// @see typeLeafInto, resolvedLabel, findDigitArgs.
        class VarTypeTable {
private:
            StrSpan names[ExecutionParameters::MAX_TYPED_VARIABLES];
            int32_t parent[ExecutionParameters::MAX_TYPED_VARIABLES];
            StrSpan labels[ExecutionParameters::MAX_TYPED_VARIABLES];
            int32_t count = 0;

public:
            /// @brief The node index of a variable name.
            ///
            /// @details A linear scan over the registered names (the table is
            /// small by contract). Absence is a defined answer: the caller
            /// asks whether the inference saw the variable at all.
            ///
            /// @param name The variable name.
            /// @return The node index, or -1 when the name is not registered.
            /// @see add.
            int32_t indexOf(StrSpan name) const {
                for (int32_t i = 0; i < count; ++i)
                    if (equalSpans(names[i], name)) return i;
                return -1;
            }

            /// @brief Register a variable name (idempotent).
            ///
            /// @details A new name becomes its own unfixed group; a known name
            /// returns its existing node. Registration order is the order the
            /// typed leaves mention the names, which also orders the records
            /// built from the table.
            ///
            /// @param name The variable name (caller-stable bytes).
            /// @return The node index of @p name.
            /// @invariant Asserts at `MAX_TYPED_VARIABLES` (never truncates).
            /// @see indexOf.
            int32_t add(StrSpan name) {
                const int32_t known = indexOf(name);
                if (known >= 0) return known;
                assert(count < ExecutionParameters::MAX_TYPED_VARIABLES
                    && "VarTypeTable: variable count exceeds MAX_TYPED_VARIABLES");
                names[count] = name;
                parent[count] = count;
                labels[count] = StrSpan();
                return count++;
            }

            /// @brief The group root of a node (with path halving).
            ///
            /// @details Standard union-find root walk; path halving only
            /// shortens later walks and never changes a group's root.
            ///
            /// @param i A valid node index.
            /// @return The root node index of @p i's group.
            /// @see join.
            int32_t root(int32_t i) {
                assert(i >= 0 && i < count && "VarTypeTable::root: invalid node");
                while (parent[i] != i) {
                    parent[i] = parent[parent[i]];
                    i = parent[i];
                }
                return i;
            }

            /// @brief Fix the definition-set label of a node's group.
            ///
            /// @details An unfixed group takes @p label; a group fixed to the
            /// same label is unchanged; a different label asserts (a type
            /// conflict).
            ///
            /// @param i     A valid node index.
            /// @param label The definition-set label (caller-stable bytes).
            /// @invariant One label per group (asserted).
            /// @see join.
            void fix(int32_t i, StrSpan label) {
                assert(label.len > 0 && "VarTypeTable::fix: empty label");
                const int32_t r = root(i);
                if (labels[r].len == 0) { labels[r] = label; return; }
                assert(equalSpans(labels[r], label)
                    && "VarTypeTable: one variable group carries two definition sets");
            }

            /// @brief Merge the groups of two nodes (what `=` does).
            ///
            /// @details The smaller root index becomes the root of the merged
            /// group and keeps the fixed label if either group had one; two
            /// different fixed labels assert (a type conflict).
            ///
            /// @param a A valid node index.
            /// @param b A valid node index.
            /// @invariant One label per group (asserted).
            /// @see fix.
            void join(int32_t a, int32_t b) {
                int32_t ra = root(a);
                int32_t rb = root(b);
                if (ra == rb) return;
                if (rb < ra) { const int32_t t = ra; ra = rb; rb = t; }
                if (labels[ra].len == 0) labels[ra] = labels[rb];
                else assert((labels[rb].len == 0 || equalSpans(labels[ra], labels[rb]))
                    && "VarTypeTable: equality joins two different definition sets");
                parent[rb] = ra;
            }

            /// @brief The fixed label of a node's group.
            ///
            /// @details Reads the group root's label.
            ///
            /// @param i A valid node index.
            /// @return The label, or an empty span while the group is unfixed.
            /// @see ScopeTypeResolver::resolvedLabel.
            StrSpan labelOf(int32_t i) { return labels[root(i)]; }
        
    /// @brief Number of registered variable names.
    /// @details Reads the existing count without changing the union-find table.
    /// @return Number of names in registration order.
    int32_t size() const { return count; }

    /// @brief Read the registered names without exposing mutable table columns.
    /// @details The caller reads only the first size() entries; spans retain their original lifetime.
    /// @return Read-only pointer to the fixed name array.
    const StrSpan* namesView() const { return names; }
};
}
