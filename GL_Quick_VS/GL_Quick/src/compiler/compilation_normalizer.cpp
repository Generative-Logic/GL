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
#include "compilation_normalizer.hpp"
#include <algorithm>
#include <set>

namespace gl {

/// @brief Build the retained definition argument list from its slot mapping.
///
/// @details Orders arguments by their numeric unchangeable slots and removes exactly one outer unchangeable prefix.
/// @param mp Original-to-normalized argument mapping.
/// @return Ordered bracketed argument list.
std::string CompilationNormalizer::createArgList(const std::map<std::string, std::string>& mp) {
            struct SortableItem {
                int sortValue;   // The integer from the value (e.g., 5 from "u_5")
                std::string key; // The original key
            };

            std::vector<SortableItem> filteredItems;

            for (const auto& pair : mp) {
                const std::string& key = pair.first;
                const std::string& value = pair.second;

                // 1. Check if value has form "u_"
                if (value.size() >= 2 && value[0] == 'u' && value[1] == '_') {

                    std::string suffix = value.substr(2);

                    // 2. Assert that the remaining part of the VALUE (suffix) is an integer
                    bool isInt = !suffix.empty() && std::all_of(suffix.begin(), suffix.end(),
                        [](unsigned char c) { return std::isdigit(c); });

                    // Debug/Safety block
                    assert(isInt && "Value starting with 'u_' must be followed by a valid integer");

                    if (!isInt) {
                        // Fail hard if assertion is ignored in release, though user requested asserts.
                        // Throwing ensures we don't process garbage.
                        throw std::runtime_error("Assertion failed: Value suffix is not an integer: " + value);
                    }

                    // 3. Extract integer from the VALUE for sorting
                    try {
                        int valId = std::stoi(suffix);
                        filteredItems.push_back({ valId, key });
                    }
                    catch (...) {
                        assert(false && "std::stoi failed on verified integer string");
                    }
                }
            }

            // 4. Sort according to integer after corresponding "u_" (from the value)
            std::sort(filteredItems.begin(), filteredItems.end(),
                [](const SortableItem& a, const SortableItem& b) {
                    return a.sortValue < b.sortValue;
                });

            // 5. Create output string "[key1,key2,key3]" with "u_" stripped from keys
            std::string result = "[";
            for (size_t i = 0; i < filteredItems.size(); ++i) {
                std::string currentKey = filteredItems[i].key;

                // Assertions requested by user
                assert(startsWithSpan(StrSpan(currentKey), "u_", 2) && "Argument in output list must start with 'u_'");

                int uCount = 0;
                size_t pos = 0;
                while ((pos = currentKey.find("u_", pos)) != std::string::npos) {
                    uCount++;
                    pos += 2;
                }
                assert(uCount == 1 && "Argument in output list must contain exactly one 'u_' substring");

                // Remove "u_" prefix
                result += currentKey.substr(2);

                if (i < filteredItems.size() - 1) {
                    result += ",";
                }
            }
            result += "]";

            return result;
        }

/// @brief Compose argument renamings through normalized unchangeable slots.
///
/// @details Preserves unchanged ordinary arguments and requires every intermediate unchangeable slot to have a destination.
/// @param a First argument mapping.
/// @param b Second argument mapping.
/// @return Composed argument mapping.
std::map<std::string, std::string> CompilationNormalizer::combineUnchangeablePaths(const std::map<std::string, std::string>& a, const std::map<std::string, std::string>& b) {
            std::map<std::string, std::string> c;

            for (const auto& pair : a) {
                const std::string& x = pair.first;     // Key
                const std::string& ax = pair.second;   // Value in a (a[x])

                // Check if a[x] starts with "u_"
                if (ax.size() >= 2 && ax[0] == 'u' && ax[1] == '_') {

                    // Extract the part after "u_"
                    std::string suffix = ax.substr(2);

                    // Control: Assert that the remainder is an integer
                    bool isInt = !suffix.empty() && std::all_of(suffix.begin(), suffix.end(),
                        [](unsigned char c) { return std::isdigit(c); });

                    assert(isInt && "Value starting with 'u_' must be followed by an integer");

                    // Double check for runtime safety if assertions are disabled
                    if (!isInt) {
                        throw std::runtime_error("Found 'u_' prefix but suffix is not an integer: " + ax);
                    }

                    // Logic: c[x] = b[a[x]]
                    // Since b is const, we must use find() instead of []
                    auto it = b.find(ax);
                    if (it != b.end()) {
                        c[x] = it->second;
                    }
                    else {
                        // If a[x] points to a key that doesn't exist in b,
                        // this usually indicates a broken logic chain.
                        assert(false && "Key found in 'a' mapping does not exist in 'b'");
                    }
                }
                else {
                    // Logic: for other keys c[x] = a[x]
                    c[x] = ax;
                }
            }

            return c;
        }

/// @brief Relate normalized unchangeable slots to retained arguments.
///
/// @details Visits the second mapping in its existing key order and resolves each shared original through the first mapping.
/// @param a Retained mapping from original arguments.
/// @param b New mapping from the same original arguments.
/// @return New normalized slots mapped to retained arguments.
std::map<std::string, std::string> CompilationNormalizer::createReverseUnchangeableMap(const std::map<std::string, std::string>& a, const std::map<std::string, std::string>& b) {
            std::map<std::string, std::string> c;

            // Iterate through b to find keys 'x' where b[x] is "u_..."
            for (const auto& pair : b) {
                const std::string& x = pair.first;       // This is 'x'
                const std::string& b_val = pair.second;  // This is 'b[x]' (e.g., "u_1")

                // 1. Check if value in b starts with "u_"
                if (b_val.size() >= 2 && b_val[0] == 'u' && b_val[1] == '_') {

                    // 2. Validate that the rest is an integer
                    std::string suffix = b_val.substr(2);
                    bool isInt = !suffix.empty() && std::all_of(suffix.begin(), suffix.end(),
                        [](unsigned char ch) { return std::isdigit(ch); });

                    assert(isInt && "Value in 'b' starting with 'u_' must be followed by an integer");

                    if (isInt) {
                        // 3. Find corresponding value in 'a' (a[x])
                        // Since 'a' is const, we use find() instead of operator[]
                        auto itA = a.find(x);

                        if (itA != a.end()) {
                            // Logic: c[b[x]] = a[x]
                            // c key: b_val ("u_1")
                            // c value: itA->second (value from a)
                            c[b_val] = itA->second;
                        }
                        else {
                            // Strict: if 'b' has a mapping for 'x', 'a' should too.
                            assert(false && "Key found in 'b' was missing in 'a'");
                        }
                    }
                }
            }

            return c;
        }

/// @brief Order a body's premises canonically: by operator name, then the
///        smallest normalized form inside each same-name group.
///
/// @details
/// The repetition map keys every definition by this order, so two bodies that
/// differ only in premise order and variable names share one key. Premises are
/// grouped by operator name in string order. Inside the groups the order whose
/// normalized token sequence is lexicographically smallest wins: each premise
/// contributes its polarity and, per argument, its kind (a bound variable
/// before a `u_` one) and its number. Bound and `u_` variables are numbered
/// separately by first appearance, as `makeNormalizedEncodedKey` mode 2 numbers
/// them, and numbers compare numerically. The head follows the premises and
/// takes part in the comparison, so it decides ties the premises leave.
///
/// The minimum is found depth first, one premise slot at a time. At each slot
/// only the candidates with the smallest next token can lead to the minimum,
/// and a branch whose sequence exceeds the best complete one found so far is
/// dropped. Two tied candidates whose new variables differ only in variables
/// that occur in no other element are interchangeable: swapping them, together
/// with those private variables, maps the body onto itself and fixes
/// everything already placed. So among tied candidates with the same new
/// shared variables (each with its number) only the first is expanded. The
/// search visits at most every ordering of each group; it reaches that bound
/// only for premises that stay interchangeable to the end through shared
/// variables.
///
/// Among orderings with the minimal sequence, the first in input index order is
/// returned, so a list already in canonical order comes back unchanged.
///
/// @param tail Premises (every element of a body without a head), any order.
/// @param head Elements that follow the premises unpermuted: the conclusion of
///        an implication, empty otherwise.
/// @return The premises in canonical order; empty for an empty `tail`.
/// @invariant Premises with the same operator name have the same arity
///            (asserted).
/// @see makeNormalizedEncodedKey, CompilationReuse::canonicalReuseKey
std::vector<EncodedExpression> CompilationNormalizer::canonicalCombination(
        const std::vector<EncodedExpression>& tail, const std::vector<EncodedExpression>& head) {
    if (tail.empty()) return {};

    // Each element as its polarity and the dense indices of its variables. A
    // variable is identified by the spelling makeNormalizedEncodedKey keys its
    // mapping by; its kind (bound 0, u_ 1) is recorded once.
    struct Element { int negation; std::vector<int> variables; };
    std::map<std::string, int> variableIndex;
    std::vector<int> kindOf;
    const auto encode = [&](const EncodedExpression& expression) {
        Element element{expression.negation ? 1 : 0, {}};
        for (const auto& argument : expression.arguments) {
            assert(argument.size() >= 2);
            const bool isU = argument[0] == "True";
            const auto inserted = variableIndex.emplace(isU ? "u_" + argument[1] : argument[1],
                                                        static_cast<int>(kindOf.size()));
            if (inserted.second) kindOf.push_back(isU ? 1 : 0);
            element.variables.push_back(inserted.first->second);
        }
        return element;
    };
    std::vector<Element> premises;
    std::vector<Element> conclusion;
    for (const auto& expression : tail) premises.push_back(encode(expression));
    for (const auto& expression : head) conclusion.push_back(encode(expression));

    // The number of elements, premises and head, that contain each variable.
    std::vector<int> occurrences(kindOf.size(), 0);
    const auto count = [&](const Element& element) {
        const std::set<int> distinct(element.variables.begin(), element.variables.end());
        for (const int variable : distinct) ++occurrences[static_cast<std::size_t>(variable)];
    };
    for (const auto& element : premises) count(element);
    for (const auto& element : conclusion) count(element);

    // One slot per premise; the groups follow the string order of their names.
    std::map<std::string, std::vector<int>> groups;
    for (std::size_t index = 0; index < tail.size(); ++index)
        groups[tail[index].name].push_back(static_cast<int>(index));
    std::vector<const std::vector<int>*> slots;
    for (const auto& group : groups)
        for (std::size_t k = 0; k < group.second.size(); ++k) slots.push_back(&group.second);

    std::vector<int> number(kindOf.size(), 0);   // 0: not numbered yet
    int next[2] = {0, 0};                        // last number given, per kind
    std::vector<char> used(tail.size(), 0);
    std::vector<int> order;
    std::vector<std::vector<int>> sequence;
    std::vector<int> bestOrder;
    std::vector<std::vector<int>> bestSequence;

    // The token of an element under the current numbering: its polarity, then
    // per argument its kind and number. A new variable takes the next number of
    // its kind in order of appearance and is listed in `fresh` with it.
    const auto tokenOf = [&](const Element& element, std::vector<int>& token,
                             std::vector<std::pair<int, int>>& fresh) {
        token.assign(1, element.negation);
        fresh.clear();
        int last[2] = {next[0], next[1]};
        for (const int variable : element.variables) {
            const int kind = kindOf[static_cast<std::size_t>(variable)];
            int value = number[static_cast<std::size_t>(variable)];
            if (value == 0) {
                const auto seen = std::find_if(fresh.begin(), fresh.end(),
                    [variable](const std::pair<int, int>& entry) { return entry.first == variable; });
                if (seen != fresh.end()) {
                    value = seen->second;
                } else {
                    value = ++last[kind];
                    fresh.emplace_back(variable, value);
                }
            }
            token.push_back(kind);
            token.push_back(value);
        }
    };
    const auto assign = [&](const std::vector<std::pair<int, int>>& fresh) {
        for (const auto& [variable, value] : fresh) {
            number[static_cast<std::size_t>(variable)] = value;
            int& counter = next[kindOf[static_cast<std::size_t>(variable)]];
            counter = std::max(counter, value);
        }
    };
    const auto release = [&](const std::vector<std::pair<int, int>>& fresh) {
        for (const auto& entry : fresh) number[static_cast<std::size_t>(entry.first)] = 0;
    };

    const auto search = [&](const auto& self, std::size_t depth) -> void {
        if (depth == slots.size()) {
            std::vector<std::vector<int>> complete = sequence;
            const int saved[2] = {next[0], next[1]};
            std::vector<std::vector<std::pair<int, int>>> numbered;
            for (const auto& element : conclusion) {
                std::vector<int> token;
                std::vector<std::pair<int, int>> fresh;
                tokenOf(element, token, fresh);
                assign(fresh);
                numbered.push_back(std::move(fresh));
                complete.push_back(std::move(token));
            }
            for (const auto& fresh : numbered) release(fresh);
            next[0] = saved[0];
            next[1] = saved[1];
            if (bestOrder.empty() || complete < bestSequence) {
                bestSequence = std::move(complete);
                bestOrder = order;
            }
            return;
        }
        struct Candidate { int index; std::vector<int> token; std::vector<std::pair<int, int>> fresh; };
        std::vector<Candidate> candidates;
        for (const int index : *slots[depth]) {
            if (used[static_cast<std::size_t>(index)]) continue;
            Candidate candidate{index, {}, {}};
            tokenOf(premises[static_cast<std::size_t>(index)], candidate.token, candidate.fresh);
            candidates.push_back(std::move(candidate));
        }
        assert(!candidates.empty());
        std::size_t lowest = 0;
        for (std::size_t c = 0; c < candidates.size(); ++c) {
            assert(candidates[c].token.size() == candidates[0].token.size()
                && "canonicalCombination: premises with one operator name differ in arity");
            if (candidates[c].token < candidates[lowest].token) lowest = c;
        }
        const std::vector<int> minimum = candidates[lowest].token;
        // Against the best complete sequence: drop the branch as soon as its
        // sequence is larger; an equal prefix compares the next token.
        if (!bestOrder.empty()) {
            bool equalPrefix = true;
            for (std::size_t d = 0; d < depth && equalPrefix; ++d) {
                if (bestSequence[d] < sequence[d]) return;
                if (sequence[d] < bestSequence[d]) equalPrefix = false;
            }
            if (equalPrefix && bestSequence[depth] < minimum) return;
        }
        // A tied candidate is identified, up to interchangeable premises, by
        // its new variables that occur elsewhere, each with its number.
        std::set<std::vector<std::pair<int, int>>> expanded;
        for (const auto& candidate : candidates) {
            if (candidate.token != minimum) continue;
            std::vector<std::pair<int, int>> shared;
            for (const auto& entry : candidate.fresh)
                if (occurrences[static_cast<std::size_t>(entry.first)] > 1) shared.push_back(entry);
            if (!expanded.insert(std::move(shared)).second) continue;
            const int saved[2] = {next[0], next[1]};
            assign(candidate.fresh);
            used[static_cast<std::size_t>(candidate.index)] = 1;
            order.push_back(candidate.index);
            sequence.push_back(minimum);
            self(self, depth + 1);
            sequence.pop_back();
            order.pop_back();
            used[static_cast<std::size_t>(candidate.index)] = 0;
            release(candidate.fresh);
            next[0] = saved[0];
            next[1] = saved[1];
        }
    };
    search(search, 0);

    assert(bestOrder.size() == tail.size());
    std::vector<EncodedExpression> result;
    result.reserve(tail.size());
    for (const int index : bestOrder) result.push_back(tail[static_cast<std::size_t>(index)]);
    return result;
}

/// @brief Normalize expression arguments in first-occurrence order.
///
/// @details Preserves operator names, polarity and the existing unchangeable-argument modes. The returned mapping records the original-to-normalized argument correspondence.
/// @param lst Expressions in their existing order.
/// @param uStatus Existing unchangeable mode: normalize together, retain literal names, or normalize separately.
/// @return Normalized key and argument mapping.
std::pair<NormalizedKey, std::map<std::string, std::string>> CompilationNormalizer::makeNormalizedEncodedKey(const std::vector<EncodedExpression>& lst, int uStatus) {
            NormalizedKey ky;
            ky.numberExpressions = static_cast<int>(lst.size());

            // Pre-size calculation
            std::size_t total = 0;
            for (std::size_t i = 0; i < lst.size(); ++i) {
                total += 2;
                const std::vector<std::vector<std::string>>& args = lst[i].arguments;
                for (std::size_t j = 0; j < args.size(); ++j) {
                    if (args[j].size() >= 2) ++total;
                }
            }
            ky.data.reserve(total);

            // Map for standard variables (produces "1", "2", "3")
            std::unordered_map<std::string, int> idByVar;
            idByVar.reserve(total);
            int nextId = 0;

            // NEW: Map for "u_" variables when status is 2 (produces "u_1", "u_2")
            std::unordered_map<std::string, int> idByUVar;
            int nextUId = 0;

            std::map<std::string, std::string> coveredVariables;

            for (std::size_t i = 0; i < lst.size(); ++i) {
                const EncodedExpression& expr = lst[i];

                ky.data.emplace_back(expr.name);
                ky.data.emplace_back(expr.negation ? "True" : "False");

                const std::vector<std::vector<std::string>>& args = expr.arguments;
                for (std::size_t j = 0; j < args.size(); ++j) {
                    const std::vector<std::string>& arg = args[j];
                    assert(arg.size() >= 2);

                    std::string original = arg[1];
                    bool isUContext = (arg[0] == "True");
                    if (isUContext)
                    {
						original = "u_" + original;
                    }


                    // CASE 1: Status 1 (Old ignoreU = true)
                    // Keep strictly as literal "u_" + name
                    if (uStatus == 1 && isUContext) {
                        ky.data.emplace_back(std::move(original));
                    }
                    // CASE 2: Status 2 (New requirement)
                    // Normalize into separate "u_1", "u_2" namespace
                    else if (uStatus == 2 && isUContext) {
                        std::unordered_map<std::string, int>::iterator it = idByUVar.find(original);
                        int id;
                        if (it == idByUVar.end()) {
                            id = ++nextUId;
                            idByUVar.insert(std::make_pair(original, id));

                            // Store replacement mapping: "x" -> "u_1"
                            std::string replacement = "u_" + std::to_string(id);
                            coveredVariables.insert(std::make_pair(original, replacement));
                            ky.data.emplace_back(std::move(replacement));
                        }
                        else {
                            id = it->second;
                            ky.data.emplace_back("u_" + std::to_string(id));
                        }
                    }
                    // CASE 3: Standard Normalization
                    // Used for Status 0, or for non-U variables in other statuses
                    else {
                        std::unordered_map<std::string, int>::iterator it = idByVar.find(original);
                        int id;
                        if (it == idByVar.end()) {
                            id = ++nextId;
                            idByVar.insert(std::make_pair(original, id));
                            coveredVariables.insert(std::make_pair(original, std::to_string(id)));
                        }
                        else {
                            id = it->second;
                        }
                        ky.data.emplace_back(std::to_string(id));
                    }
                }
            }
            return std::make_pair(ky, coveredVariables);
        }

} // namespace gl
