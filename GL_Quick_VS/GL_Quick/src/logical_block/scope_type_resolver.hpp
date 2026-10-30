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
#include "../compiler/compiled_definition_registry.hpp"
#include "../compiler/structural_compiler.hpp"
#include "scope_syntax.hpp"
#include "canonicalizer.hpp"
#include "var_type_table.hpp"
namespace gl {
namespace tests { struct ScopeTypeFixture; }
/// @brief Infer variable definition sets and resolve typed goals in logical scopes.
/// @details Preserves leaf order, scope inheritance and configured equality labels.
class ScopeTypeResolver {
friend struct tests::ScopeTypeFixture;
private:
/// @brief Byte ceiling of one variable definition-set record.
        static constexpr int32_t kMaxVarDefSetsRecordBytes = 16384;
public:
/// @brief Bind definition readers and typed-equality compilation.
/// @details Inference tables remain caller-owned; block type records keep their existing stores.
/// @param definitions Read-only configured and compiled definition metadata.
/// @param structuralCompiler Compiler operation for required typed equality entities.
/// @param canonicalizer Existing equality-shape queries.
/// @param anchorInfo Batch anchor used by the original validity scan.
/// @return Constructed scope type resolver.
ScopeTypeResolver(const CompiledDefinitionRegistry& definitions, StructuralCompiler& structuralCompiler, Canonicalizer& canonicalizer, const ce::AnchorInfo& anchorInfo);

private:
/// @brief `=`'s configured definition set — the starting point of every
        ///        definition-set inference.
        ///
        /// @details
        /// The config keeps the hard-coded typing of `=` (`(1)` on both slots)
        /// as the label of an argument group nothing else fixes. Reading it
        /// from the config (slot 1) instead of spelling `(1)` keeps the config
        /// the single source.
        ///
        /// @return A span over the config label (stable: the config is never
        ///         mutated during a burst, I-137).
        /// @invariant Asserts when `=` or its slot 1 has no configured label.
        /// @see resolvedLabel, VarTypeTable.
inline StrSpan equalityStartLabel() const {
            const ce::CoreExpressionConfig* cfg = this->definitions.coreConfig(StrSpan("=", 1));
            assert(cfg != nullptr && "equalityStartLabel: `=` has no config entry");
            const std::pair<std::string, bool>* ds = gl::CompiledDefinitionRegistry::findDefinitionSet(cfg, 1);
            assert(ds != nullptr && "equalityStartLabel: `=` slot 1 has no definition set");
            return StrSpan(ds->first);
        }

private:
/// @brief The definition set a table node resolves to.
        ///
        /// @details
        /// The group's fixed label, or `=`'s configured label when nothing
        /// fixed the group (the starting point — a defined part of the typing
        /// rule, not a failure fallback).
        ///
        /// @param table The inference table.
        /// @param index A valid node index of @p table.
        /// @return The resolved definition-set label.
        /// @see VarTypeTable::labelOf, equalityStartLabel.
inline StrSpan resolvedLabel(VarTypeTable& table, int32_t index) const {
            const StrSpan label = table.labelOf(index);
            return label.len > 0 ? label : equalityStartLabel();
        }

private:
/// @brief Whether a span is one flat operator application
        ///        `(name[a,b,...])` — the only leaf shape the typing reads.
        ///
        /// @details
        /// True iff the span starts with `(`, has one `[...]` window with no
        /// brackets or parentheses inside, and ends right after that window
        /// with `])`. Negation is stripped by the caller.
        ///
        /// @param s The candidate leaf.
        /// @return True for a flat operator application.
        /// @see typeLeafInto.
static bool isFlatOperatorLeaf(StrSpan s) {
            if (s.len < 4 || s.ptr[0] != '(' || s.ptr[s.len - 1] != ')'
                || s.ptr[s.len - 2] != ']') return false;
            int32_t open = -1;
            for (int32_t i = 1; i < s.len - 2; ++i) {
                const char c = s.ptr[i];
                if (c == '[') {
                    if (open >= 0) return false;
                    open = i;
                } else if (c == '(' || c == ')' || c == ']') {
                    return false;
                }
            }
            return open > 1;
        }

private:
/// @brief Type one flat MPL leaf into an inference table.
        ///
        /// @details
        /// The typing rule of `=` (I-267), applied to one
        /// leaf (a leading `!` is stripped — polarity does not change types):
        /// - `(=[a,b])` joins `a` and `b`, fixing neither;
        /// - an operator with a config entry fixes each argument to its
        ///   slot's definition set (a slot without one fixes nothing);
        /// - a compiled compact without a config entry is typed through its
        ///   compiled elements: they are typed into a local table in the
        ///   compact's own variable space, then each formal's label fixes the
        ///   matching actual argument, and formals joined inside the compact
        ///   join their actual arguments.
        /// An operator with neither a config entry nor a compiled entity
        /// asserts. Every argument is registered, fixed or not. The tables are
        /// stack arrays and all names are spans into caller-stable text, so
        /// the routine allocates nothing and is safe on the burst paths.
        ///
        /// @param leaf  One flat operator application, optionally negated.
        /// @param table The inference table receiving the leaf's constraints.
        /// @param depth Compact nesting depth (0 at the caller; asserts at 16).
        /// @invariant `I-267`; allocates nothing.
        /// @see VarTypeTable, resolvedLabel, findDefinitionSet.
inline void typeLeafInto(StrSpan leaf, VarTypeTable& table,
                                 int32_t depth = 0) const {
            assert(depth < 16 && "typeLeafInto: compact nesting exceeds 16");
            StrSpan body = leaf;
            if (body.len >= 2 && body.ptr[0] == '!' && body.ptr[1] == '(')
                body = StrSpan(body.ptr + 1, body.len - 1);
            assert(isFlatOperatorLeaf(body) && "typeLeafInto: leaf is not a flat operator application");

            const StrSpan core = extractExpressionSpan(body);
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argN = getArgsSpans(body, args, ExecutionParameters::MAX_ARITY);

            if (equalSpans(core, StrSpan("=", 1))) {
                assert(argN == 2 && "typeLeafInto: `=` takes two arguments");
                table.join(table.add(args[0]), table.add(args[1]));
                return;
            }

            const ce::CoreExpressionConfig* cfg = this->definitions.coreConfig(core);
            if (cfg != nullptr) {
                for (int32_t i = 0; i < argN; ++i) {
                    const int32_t node = table.add(args[i]);
                    const std::pair<std::string, bool>* ds = gl::CompiledDefinitionRegistry::findDefinitionSet(cfg, i + 1);
                    if (ds != nullptr) table.fix(node, StrSpan(ds->first));
                }
                return;
            }

            const LogicalEntity* le = this->definitions.compiledEntity(core);
            assert(le != nullptr && "typeLeafInto: operator has neither a config entry nor a compiled entity");
            VarTypeTable local;
            for (const std::string& element : le->elements) {
                if (element.empty()) continue;
                typeLeafInto(StrSpan(element), local, depth + 1);
            }
            StrSpan formals[ExecutionParameters::MAX_ARITY];
            const int32_t formalN = getArgsSpans(StrSpan(le->signature), formals,
                                                 ExecutionParameters::MAX_ARITY);
            assert(formalN == argN && "typeLeafInto: compact arity differs from its signature");
            int32_t actualNode[ExecutionParameters::MAX_ARITY];
            int32_t formalNode[ExecutionParameters::MAX_ARITY];
            for (int32_t i = 0; i < argN; ++i) {
                actualNode[i] = table.add(args[i]);
                formalNode[i] = local.indexOf(formals[i]);
                if (formalNode[i] < 0) continue;
                const StrSpan label = local.labelOf(formalNode[i]);
                if (label.len > 0) table.fix(actualNode[i], label);
            }
            for (int32_t i = 0; i < argN; ++i) {
                if (formalNode[i] < 0) continue;
                for (int32_t j = i + 1; j < argN; ++j) {
                    if (formalNode[j] < 0) continue;
                    if (local.root(formalNode[i]) == local.root(formalNode[j]))
                        table.join(actualNode[i], actualNode[j]);
                }
            }
        }

public:
/// @brief The arguments of one flat expression whose definition set equals
        ///        a given argument's (D-529).
        ///
        /// @details
        /// Types @p part on its own by position — a configured operator fixes
        /// each argument to its slot's definition set, a compiled compact
        /// types through its elements, an `=` joins its two arguments
        /// (`typeLeafInto`) — and returns, in argument order and without
        /// duplicates, every argument other than @p witness whose resolved
        /// definition set equals the resolved definition set of @p witness.
        /// The witness-link admission gate reads the other part of a witness's
        /// existence through it: only an argument of the witness's own kind
        /// (a sequence beside a sequence) links the witness to it. A part that
        /// does not mention @p witness links it to nothing and yields 0 — a
        /// defined answer, not a failure.
        ///
        /// @param part    One flat operator application, optionally negated.
        /// @param witness The argument whose definition set selects (a span
        ///                equal to one of @p part's arguments, or absent).
        /// @param out     Caller array receiving spans into @p part.
        /// @param cap     Capacity of @p out (at most `MAX_ARITY` arguments).
        /// @return The number of spans written.
        /// @invariant Allocates and mints nothing (I-267 typing tables are
        ///            stack arrays).
        /// @see typeLeafInto, resolvedLabel, AdmissionMemory::witnessLinkArguments.
inline int32_t sameDefinitionSetArguments(StrSpan part, StrSpan witness,
                                                  StrSpan* out, int32_t cap) const {
            VarTypeTable table;
            typeLeafInto(part, table);
            const int32_t w = table.indexOf(witness);
            if (w < 0) return 0;
            const StrSpan label = resolvedLabel(table, w);
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argN = getArgsSpans(part, args, ExecutionParameters::MAX_ARITY);
            int32_t n = 0;
            for (int32_t i = 0; i < argN; ++i) {
                if (equalSpans(args[i], witness)) continue;
                bool repeated = false;
                for (int32_t k = 0; k < n && !repeated; ++k) repeated = equalSpans(out[k], args[i]);
                if (repeated) continue;
                const int32_t node = table.indexOf(args[i]);
                assert(node >= 0 && "sameDefinitionSetArguments: an argument was not typed");
                if (!equalSpans(resolvedLabel(table, node), label)) continue;
                assert(n < cap && "sameDefinitionSetArguments: argument count exceeds capacity");
                out[n++] = args[i];
            }
            return n;
        }

public:
/// @brief The definition set of one argument of a flat expression
        ///        (D-522).
        ///
        /// @details
        /// Types @p part on its own by position, exactly as
        /// `sameDefinitionSetArguments` does (`typeLeafInto`: a configured
        /// operator fixes each argument to its slot's definition set, a compiled
        /// compact types through its elements, an `=` joins its two arguments),
        /// and returns the resolved definition set of @p argument. The
        /// witness-history check reads the definition set of an admitted
        /// witness from its existence and of an ancestor's argument from the
        /// ancestor's leaf. An argument that @p part does not mention has no
        /// definition set here and yields the empty span — a defined answer,
        /// not a failure.
        ///
        /// @param part     One flat operator application, optionally negated.
        /// @param argument The argument to type.
        /// @return The definition-set label (a span into the configuration,
        ///         stable for the batch), or empty.
        /// @invariant Allocates and mints nothing.
        /// @see sameDefinitionSetArguments, typeLeafInto, resolvedLabel,
        ///      AdmissionMemory::witnessHistoryRefuses.
inline StrSpan definitionSetOf(StrSpan part, StrSpan argument) const {
            VarTypeTable table;
            typeLeafInto(part, table);
            const int32_t node = table.indexOf(argument);
            if (node < 0) return StrSpan();
            return resolvedLabel(table, node);
        }

private:
/// @brief The MPL expression an LB key stands for.
        ///
        /// @details
        /// Strips the `__contradiction__` prefix of a contradiction LB and any
        /// suffix after the leading balanced expression (recursion block #2
        /// appends `_induction_<recN>_`). A key without a leading `(` or `!(`
        /// — the root sentinel, a compressor node, a counterexample clone —
        /// stands for no expression and yields the empty span (a defined
        /// answer: such an LB has no key arguments).
        ///
        /// @param key The LB key (its `exprKeyView`).
        /// @return The leading expression (a sub-span of @p key), or empty.
        /// @invariant Asserts on an unbalanced leading expression.
        /// @see assignLbVarDefinitionSets.
static StrSpan leadingExpressionOfKey(StrSpan key) {
            static const char kContradiction[] = "__contradiction__";
            const std::size_t prefixLen = sizeof(kContradiction) - 1;
            StrSpan k = key;
            if (startsWithSpan(k, kContradiction, prefixLen))
                k = StrSpan(k.ptr + prefixLen, k.len - static_cast<int32_t>(prefixLen));
            const int32_t start = (k.len >= 1 && k.ptr[0] == '!') ? 1 : 0;
            if (k.len <= start || k.ptr[start] != '(') return StrSpan();
            int32_t depth = 0;
            for (int32_t i = start; i < k.len; ++i) {
                if (k.ptr[i] == '(') ++depth;
                else if (k.ptr[i] == ')' && --depth == 0) return StrSpan(k.ptr, i + 1);
            }
            assert(false && "leadingExpressionOfKey: unbalanced key expression");
            return StrSpan();
        }

private:
/// @brief Append one `name:label` pair to a record buffer.
        ///
        /// @details Writes a `;` separator before every pair but the first.
        ///
        /// @param buf   The record buffer.
        /// @param len   The record length so far (updated).
        /// @param name  The variable name.
        /// @param label The definition-set label.
        /// @invariant Asserts at `kMaxVarDefSetsRecordBytes`.
        /// @see buildLbVarDefinitionSetsRecord, assignTheoremVarDefinitionSets.
static void appendVarDefSetPair(char* buf, int32_t& len,
                                        StrSpan name, StrSpan label) {
            const int32_t need = (len > 0 ? 1 : 0) + name.len + 1 + label.len;
            assert(len + need <= kMaxVarDefSetsRecordBytes
                && "appendVarDefSetPair: record exceeds kMaxVarDefSetsRecordBytes");
            if (len > 0) buf[len++] = ';';
            std::memcpy(buf + len, name.ptr, static_cast<std::size_t>(name.len));
            len += name.len;
            buf[len++] = ':';
            std::memcpy(buf + len, label.ptr, static_cast<std::size_t>(label.len));
            len += label.len;
        }

private:
/// @brief Build an LB's variable definition-set record from its key
        ///        chain.
        ///
        /// @details
        /// Types the key expressions from @p lb up to the root sentinel
        /// (`leadingExpressionOfKey`; argument-free keys contribute nothing)
        /// and writes one `name:label` pair per distinct argument of @p lb's
        /// own key, in argument order, with the resolved label. The record
        /// depends only on the chain of keys from the root to @p lb, so every
        /// theorem that shares the LB builds the same record.
        ///
        /// @param lb  The LB (its key and parent chain are set).
        /// @param buf Caller buffer of `kMaxVarDefSetsRecordBytes` bytes.
        /// @return The record length (0 for an argument-free key).
        /// @invariant `I-267`; allocates nothing.
        /// @see assignLbVarDefinitionSets, lbVarDefinitionSetsMatch.
inline int32_t buildLbVarDefinitionSetsRecord(const Memory& lb, char* buf) const {
            StrSpan leaves[ExecutionParameters::MAX_SCOPE_DEPTH];
            int32_t leafN = 0;
            for (const Memory* m = &lb; m != nullptr; m = m->parentMemory) {
                const StrSpan expression = leadingExpressionOfKey(m->exprKeyView());
                if (expression.len == 0) continue;
                assert(leafN < ExecutionParameters::MAX_SCOPE_DEPTH
                    && "buildLbVarDefinitionSetsRecord: LB chain exceeds MAX_SCOPE_DEPTH");
                leaves[leafN++] = expression;
            }
            const StrSpan own = leadingExpressionOfKey(lb.exprKeyView());
            if (own.len == 0) return 0;

            VarTypeTable table;
            for (int32_t i = leafN - 1; i >= 0; --i) typeLeafInto(leaves[i], table);

            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argN = getArgsSpans(own, args, ExecutionParameters::MAX_ARITY);
            int32_t len = 0;
            for (int32_t i = 0; i < argN; ++i) {
                bool repeated = false;
                for (int32_t j = 0; j < i && !repeated; ++j)
                    repeated = equalSpans(args[j], args[i]);
                if (repeated) continue;
                const int32_t node = table.indexOf(args[i]);
                assert(node >= 0 && "buildLbVarDefinitionSetsRecord: key argument was not typed");
                appendVarDefSetPair(buf, len, args[i], resolvedLabel(table, node));
            }
            return len;
        }

public:
/// @brief Record an LB's variable definition sets at its creation.
        ///
        /// @details
        /// Builds the record from the LB's key chain and interns it into the
        /// LB (`Memory::setVarDefSets`). Called right after the key and the
        /// parent are set, before anything is seeded into the LB, on the
        /// single-threaded registration seam.
        ///
        /// @param lb The newly created LB.
        /// @invariant `I-267` — written once per LB.
        /// @see buildLbVarDefinitionSetsRecord, lbVarDefinitionSetsMatch.
inline void assignLbVarDefinitionSets(Memory& lb) const {
            assert(lb.varDefSetsId == 0 && "assignLbVarDefinitionSets: record already set");
            char buf[kMaxVarDefSetsRecordBytes];
            const int32_t len = buildLbVarDefinitionSetsRecord(lb, buf);
            lb.setVarDefSets(StrSpan(buf, len));
        }

public:
/// @brief Whether an existing LB's record equals a fresh build.
        ///
        /// @details
        /// A theorem that reaches an existing LB (a shared chain prefix)
        /// rebuilds the record and compares; the record is a function of the
        /// key chain, so a mismatch means the typing inputs changed under a
        /// live LB.
        ///
        /// @param lb The existing LB.
        /// @return True iff the stored record equals the rebuilt one.
        /// @see assignLbVarDefinitionSets.
inline bool lbVarDefinitionSetsMatch(const Memory& lb) const {
            char buf[kMaxVarDefSetsRecordBytes];
            const int32_t len = buildLbVarDefinitionSetsRecord(lb, buf);
            return equalSpans(lb.varDefSetsView(), StrSpan(buf, len));
        }

public:
/// @brief Record the variable definition sets of a whole theorem on
        ///        an LB whose key holds no arguments (a compressor node).
        ///
        /// @details
        /// Types @p leaves (the theorem's premise chain and head) and writes
        /// one `name:label` pair per variable, in first-mention order. Called
        /// once, on the single-threaded node setup, before the theorem's
        /// premises and head are seeded at the node's `main`.
        ///
        /// @param lb     The LB (its key must stand for no expression).
        /// @param leaves The theorem's flat leaves (premises then head).
        /// @param n      The leaf count.
        /// @invariant `I-267`; the record is written once.
        /// @see lookupLbVarDefinitionSet.
inline void assignTheoremVarDefinitionSets(Memory& lb, const StrSpan* leaves,
                                                   int32_t n) const {
            assert(lb.varDefSetsId == 0 && "assignTheoremVarDefinitionSets: record already set");
            assert(leadingExpressionOfKey(lb.exprKeyView()).len == 0
                && "assignTheoremVarDefinitionSets: LB key has its own arguments");
            VarTypeTable table;
            for (int32_t i = 0; i < n; ++i) typeLeafInto(leaves[i], table);
            char buf[kMaxVarDefSetsRecordBytes];
            int32_t len = 0;
            for (int32_t i = 0; i < table.size(); ++i)
                appendVarDefSetPair(buf, len, table.namesView()[i], resolvedLabel(table, i));
            lb.setVarDefSets(StrSpan(buf, len));
        }

private:
/// @brief The definition set one LB records for a variable.
        ///
        /// @details
        /// Scans the LB's `name:label` record; absence is a defined answer (the
        /// variable is not one of this LB's recorded variables — the caller
        /// walks further up the chain). The returned span points into the
        /// never-deloaded `skeletonInterner()`, which no burst mints into.
        ///
        /// @param lb  The LB.
        /// @param var The variable name.
        /// @return The recorded label, or an empty span when not recorded.
        /// @see Memory::varDefSetsView, assignLbVarDefinitionSets.
static StrSpan lookupLbVarDefinitionSet(const Memory& lb, StrSpan var) {
            const StrSpan record = lb.varDefSetsView();
            int32_t pos = 0;
            while (pos < record.len) {
                int32_t end = pos;
                while (end < record.len && record.ptr[end] != ';') ++end;
                int32_t colon = pos;
                while (colon < end && record.ptr[colon] != ':') ++colon;
                assert(colon < end && "lookupLbVarDefinitionSet: malformed record");
                if (equalSpans(StrSpan(record.ptr + pos, colon - pos), var))
                    return StrSpan(record.ptr + colon + 1, end - colon - 1);
                pos = end + 1;
            }
            return StrSpan();
        }

private:
/// @brief The definition set the LB chain records for a variable.
        ///
        /// @details
        /// Walks @p mb and its parents up to the root sentinel and returns the
        /// first record that holds @p var. The nearest record wins: a deeper LB
        /// is typed from a longer key chain. Absence is a defined answer (the
        /// variable is not an LB variable — a subproof's renamed bound variable,
        /// for example).
        ///
        /// @param mb  The LB the lookup starts at.
        /// @param var The variable name.
        /// @return The recorded label, or an empty span.
        /// @see lookupLbVarDefinitionSet, resolveEqualityGoalLabel.
static StrSpan chainVarLabel(const Memory& mb, StrSpan var) {
            for (const Memory* m = &mb; m != nullptr; m = m->parentMemory) {
                const StrSpan label = lookupLbVarDefinitionSet(*m, var);
                if (label.len > 0) return label;
            }
            return StrSpan();
        }

private:
/// @brief Type a variable of a scope element's own space through the
        ///        element's interface slots.
        ///
        /// @details
        /// An integration scope is opened by an element instance: a Case-A
        /// subproof by `(implicationN[actuals])`, an OR-case branch by the OR
        /// instance `(orN[actuals])`. The compiled element is typed in its own
        /// variable space with the shared typing rule: its formals are seeded
        /// by its config slots (when it has a config entry) and its compiled
        /// elements are typed through `typeLeafInto`. The group of
        /// @p elementVar then resolves:
        /// - to its fixed label;
        /// - else, when the group holds formals, to the type of a formal's
        ///   actual as seen at the parent scope (`scopeVarLabel`); an actual
        ///   that no scope types leaves the group untyped (empty result);
        /// - else (a group of the element's bound variables only) to the
        ///   typing rule's own resolution inside the element, `=`'s starting
        ///   point, exactly as for a theorem variable that only `=` touches.
        ///
        /// @param instance    The element instance `(core[actuals])`.
        /// @param elementVar  A variable of the compiled element's own space
        ///                    (a formal or a bound variable).
        /// @param mb          The LB holding the scopes.
        /// @param parentScope The scope where the instance's actuals live.
        /// @return The definition set, or an empty span when the group's
        ///         formals have untyped actuals.
        /// @invariant Allocates nothing (stack tables, spans into the registry
        ///            and the NameMap). Asserts that the element is compiled.
        /// @see scopeVarLabel, scopeGoalArgLabel, typeLeafInto.
inline StrSpan elementVarLabel(StrSpan instance, StrSpan elementVar,
                                       const Memory& mb, NameId parentScope) const {
            const StrSpan core = extractExpressionSpan(instance);
            const LogicalEntity* le = this->definitions.compiledEntity(core);
            assert(le != nullptr && "elementVarLabel: a scope element is a compiled compact");

            StrSpan formals[ExecutionParameters::MAX_ARITY];
            const int32_t formalN = getArgsSpans(StrSpan(le->signature), formals,
                                                 ExecutionParameters::MAX_ARITY);
            StrSpan actuals[ExecutionParameters::MAX_ARITY];
            const int32_t actualN = getArgsSpans(instance, actuals, ExecutionParameters::MAX_ARITY);
            assert(formalN == actualN && "elementVarLabel: instance arity differs from the element signature");

            VarTypeTable local;
            const ce::CoreExpressionConfig* cfg = this->definitions.coreConfig(core);
            for (int32_t i = 0; i < formalN; ++i) {
                const int32_t node = local.add(formals[i]);
                if (cfg == nullptr) continue;
                const std::pair<std::string, bool>* ds = gl::CompiledDefinitionRegistry::findDefinitionSet(cfg, i + 1);
                if (ds != nullptr) local.fix(node, StrSpan(ds->first));
            }
            for (const std::string& element : le->elements) {
                if (element.empty()) continue;
                typeLeafInto(StrSpan(element), local);
            }

            const int32_t node = local.indexOf(elementVar);
            assert(node >= 0 && "elementVarLabel: the variable does not occur in the element");
            const StrSpan fixedLabel = local.labelOf(node);
            if (fixedLabel.len > 0) return fixedLabel;
            bool groupHasFormal = false;
            for (int32_t i = 0; i < formalN; ++i) {
                if (local.root(local.indexOf(formals[i])) != local.root(node)) continue;
                groupHasFormal = true;
                const StrSpan label = scopeVarLabel(actuals[i], mb, parentScope);
                if (label.len > 0) return label;
            }
            if (groupHasFormal) return StrSpan();
            return resolvedLabel(local, node);
        }

private:
/// @brief The goal of a Case-A integration scope.
        ///
        /// @details A Case-A scope below main holds exactly the one goal its
        /// opening deposited (the renamed head of its implication); the
        /// registry scan finds it by validity id.
        ///
        /// @param mb    The LB.
        /// @param scope The scope's validity id.
        /// @return The goal, or an empty span when the scope holds none.
        /// @invariant Asserts on two goals in one scope. Allocates nothing.
        /// @see scopeVarLabel.
static StrSpan goalAtScope(const Memory& mb, NameId scope) {
            StrSpan goal;
            int32_t hits = 0;
            const int32_t n = mb.intToBeProved.count();
            for (int32_t id = 1; id <= n; ++id) {
                const StatementKey sk = Codec<StatementKey>::decode(mb.intToBeProved.keyAt(id));
                if (sk.validity != scope) continue;
                goal = mb.nameMap.decodeView(sk.orig);
                ++hits;
            }
            assert(hits <= 1 && "goalAtScope: an integration scope holds one goal");
            return goal;
        }

private:
/// @brief The definition set of a variable as seen at an integration
        ///        scope, derived by walking the scopes up to the LB records.
        ///
        /// @details
        /// A scope's variables are the LB variables, the actuals of the
        /// element that opened it (they live in the parent scope), and the
        /// element's renamed bound variables (they occur in the scope's goal).
        /// The variable resolves:
        /// - through the LB records up the parent chain (`chainVarLabel`);
        /// - at main (a root scope) nowhere else: no element opened it;
        /// - as an actual of the scope's element, through the element's slot
        ///   for it (`elementVarLabel`, which climbs to the parent scope);
        /// - as an argument of the scope's goal, position by position through
        ///   the element (`scopeGoalArgLabel`);
        /// - at a scope no element opened (an OR-disintegration branch, for
        ///   example), as seen at the parent scope.
        ///
        /// @param var   The variable name.
        /// @param mb    The LB holding the scopes.
        /// @param scope The scope's validity id.
        /// @return The definition set, or an empty span when no scope types it.
        /// @invariant Allocates nothing. Terminates: every step reaches the
        ///            parent scope or the LB records.
        /// @see elementVarLabel, scopeGoalArgLabel, resolveEqualityGoalLabel.
inline StrSpan scopeVarLabel(StrSpan var, const Memory& mb, NameId scope) const {
            const StrSpan recorded = chainVarLabel(mb, var);
            if (recorded.len > 0) return recorded;
            if (mb.nameMap.stackEmpty(scope)) return StrSpan();
            const NameId parent = mb.nameMap.parentOf(scope);

            StrSpan instance;
            StrSpan orSignature;
            StrSpan orBody;
            const bool orCase = gl::ScopeSyntax::classifyOrScopeView(mb.nameMap, scope, orSignature, orBody)
                == OrScopeKind::Integration;
            if (orCase) {
                instance = orSignature;
            } else {
                const StrSpan payload = gl::ScopeSyntax::stripSubproofPrefixView(
                    mb.nameMap.decodeSubView(mb.nameMap.stackBack(scope)));
                if (!isFlatOperatorLeaf(payload)) return scopeVarLabel(var, mb, parent);
                instance = payload;
            }

            const LogicalEntity* le = this->definitions.compiledEntity(extractExpressionSpan(instance));
            assert(le != nullptr && "scopeVarLabel: a scope element is a compiled compact");
            StrSpan formals[ExecutionParameters::MAX_ARITY];
            const int32_t formalN = getArgsSpans(StrSpan(le->signature), formals,
                                                 ExecutionParameters::MAX_ARITY);
            StrSpan actuals[ExecutionParameters::MAX_ARITY];
            const int32_t actualN = getArgsSpans(instance, actuals, ExecutionParameters::MAX_ARITY);
            assert(formalN == actualN && "scopeVarLabel: instance arity differs from the element signature");
            for (int32_t i = 0; i < actualN; ++i)
                if (equalSpans(actuals[i], var))
                    return elementVarLabel(instance, formals[i], mb, parent);

            const StrSpan goal = orCase ? StrSpan(orBody.ptr + 1, orBody.len - 2)
                                        : goalAtScope(mb, scope);
            if (goal.len == 0) return StrSpan();
            StrSpan goalArgs[ExecutionParameters::MAX_ARITY];
            const int32_t goalN = getArgsSpans(goal, goalArgs, ExecutionParameters::MAX_ARITY);
            for (int32_t p = 0; p < goalN; ++p)
                if (equalSpans(goalArgs[p], var))
                    return scopeGoalArgLabel(goal, p, mb, scope);
            return StrSpan();
        }

private:
/// @brief The definition set of one argument of a goal, through the
        ///        element that opened the goal's scope.
        ///
        /// @details
        /// - The LB records win when they hold the argument.
        /// - At main no element opened the scope: no answer.
        /// - A Case-A goal is its implication's head with the bound variables
        ///   renamed, so goal position @p argIndex is the compiled head's
        ///   argument at the same position, typed inside the element
        ///   (`elementVarLabel`).
        /// - An OR-case goal is one of the OR instance's disjuncts over the
        ///   instance's actuals: an actual is typed through the OR element's
        ///   slot for it. A compiled OR binds no variable of its own, so an
        ///   argument that is no actual has no answer here.
        ///
        /// @param goal     The goal at @p scope.
        /// @param argIndex Zero-based argument position.
        /// @param mb       The LB holding the scopes.
        /// @param scope    The goal's scope.
        /// @return The definition set, or an empty span when nothing types it.
        /// @invariant Allocates nothing. Asserts that a goal below main sits in
        ///            an integration scope and matches its element's head.
        /// @see elementVarLabel, scopeVarLabel, resolveEqualityGoalLabel.
inline StrSpan scopeGoalArgLabel(StrSpan goal, int32_t argIndex,
                                         const Memory& mb, NameId scope) const {
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argN = getArgsSpans(goal, args, ExecutionParameters::MAX_ARITY);
            assert(argIndex >= 0 && argIndex < argN && "scopeGoalArgLabel: position out of range");
            const StrSpan recorded = chainVarLabel(mb, args[argIndex]);
            if (recorded.len > 0) return recorded;
            if (mb.nameMap.stackEmpty(scope)) return StrSpan();
            const NameId parent = mb.nameMap.parentOf(scope);

            StrSpan orSignature;
            StrSpan orBody;
            if (gl::ScopeSyntax::classifyOrScopeView(mb.nameMap, scope, orSignature, orBody)
                    == OrScopeKind::Integration) {
                const LogicalEntity* orLe = this->definitions.compiledEntity(extractExpressionSpan(orSignature));
                assert(orLe != nullptr && "scopeGoalArgLabel: an OR scope element is a compiled compact");
                StrSpan formals[ExecutionParameters::MAX_ARITY];
                const int32_t formalN = getArgsSpans(StrSpan(orLe->signature), formals,
                                                     ExecutionParameters::MAX_ARITY);
                StrSpan actuals[ExecutionParameters::MAX_ARITY];
                const int32_t actualN = getArgsSpans(orSignature, actuals,
                                                     ExecutionParameters::MAX_ARITY);
                assert(formalN == actualN && "scopeGoalArgLabel: OR instance arity differs from its signature");
                for (int32_t i = 0; i < actualN; ++i)
                    if (equalSpans(actuals[i], args[argIndex]))
                        return elementVarLabel(orSignature, formals[i], mb, parent);
                return StrSpan();
            }

            const StrSpan payload = gl::ScopeSyntax::stripSubproofPrefixView(
                mb.nameMap.decodeSubView(mb.nameMap.stackBack(scope)));
            assert(isFlatOperatorLeaf(payload)
                && "scopeGoalArgLabel: a goal below main sits in an integration scope");
            const LogicalEntity* le = this->definitions.compiledEntity(extractExpressionSpan(payload));
            assert(le != nullptr && le->category == "implication" && !le->elements.empty()
                && "scopeGoalArgLabel: a Case-A scope element is a compiled implication");
            const StrSpan head(le->elements.back());
            StrSpan headArgs[ExecutionParameters::MAX_ARITY];
            const int32_t headN = getArgsSpans(head, headArgs, ExecutionParameters::MAX_ARITY);
            assert(equalSpans(extractExpressionSpan(head), extractExpressionSpan(goal)) && headN == argN
                && "scopeGoalArgLabel: a Case-A goal is its implication's renamed head");
            return elementVarLabel(payload, headArgs[argIndex], mb, parent);
        }

public:
/// @brief The definition set of an `=` integration goal, typed on the
        ///        spot.
        ///
        /// @details
        /// The maintainer's rule: an integration goal is fully typed, and the
        /// rest is inferred from interface slots
        /// (I-267):
        /// (a) the LB records up the parent chain type the goal's arguments
        ///     when they are LB variables (a theorem head, a recursion or
        ///     contradiction goal, a compressor node's target head);
        /// (b) otherwise the element that opened the goal's scope types it:
        ///     the implication of a Case-A subproof or the OR of an OR-case
        ///     branch, position by position, iterating up the scopes when the
        ///     element's own arguments need it (`scopeGoalArgLabel`);
        /// (c) otherwise the goal is untyped, which asserts (maintainer's
        ///     choice: no `(1)` fallback).
        /// Step (b) runs only when the records type neither side; when they
        /// type both, the two labels must agree. `=` joins its two sides, so
        /// one typed side types the goal.
        ///
        /// @param goal         The positive `=` goal.
        /// @param mb           The LB holding the goal.
        /// @param validityName The goal's scope.
        /// @return The goal's definition set (never empty).
        /// @invariant Asserts when the goal cannot be typed or its arguments
        ///            disagree. Allocates nothing.
        /// @see chainVarLabel, scopeGoalArgLabel, prepareIntegration.
inline StrSpan resolveEqualityGoalLabel(StrSpan goal, const Memory& mb,
                                                StrSpan validityName) const {
            assert(startsWithSpan(goal, "(=[", 3) && "resolveEqualityGoalLabel: not a positive equality");
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argN = getArgsSpans(goal, args, ExecutionParameters::MAX_ARITY);
            assert(argN == 2 && "resolveEqualityGoalLabel: `=` takes two arguments");

            const StrSpan left = chainVarLabel(mb, args[0]);
            const StrSpan right = chainVarLabel(mb, args[1]);
            if (left.len > 0 && right.len > 0) {
                assert(equalSpans(left, right)
                    && "resolveEqualityGoalLabel: the LB records type the two sides differently");
                return left;
            }
            if (left.len > 0) return left;
            if (right.len > 0) return right;

            const NameId scope = mb.nameMap.lookup(validityName);
            assert(scope != 0 && "resolveEqualityGoalLabel: goal scope is not minted");
            const StrSpan leftInScope = scopeGoalArgLabel(goal, 0, mb, scope);
            if (leftInScope.len > 0) return leftInScope;
            const StrSpan rightInScope = scopeGoalArgLabel(goal, 1, mb, scope);
            if (rightInScope.len > 0) return rightInScope;
            assert(false && "resolveEqualityGoalLabel: `=` integration goal is untyped");
            return StrSpan();
        }

public:
/// @brief The extensionality entities, one per registered non-digit
        ///        definition set; written on the single-threaded registration
        ///        seams, read-only during bursts.





        /// @brief Register the extensionality entity a theorem's `=` head
        ///        needs, before the head is seeded as a goal.
        ///
        /// @details
        /// When the last leaf (the head) is a positive `=` that the theorem
        /// types as something other than `(1)`, compiles its entity
        /// (`ensureTypedEqualityEntity`). Every other head needs nothing.
        ///
        /// @param leaves The theorem's flat leaves: premises, then the head.
        /// @param n      The leaf count (the head is `leaves[n - 1]`).
        /// @see ensureTypedEqualityEntity, typeLeafInto.
void ensureTypedEqualityEntityForTheorem(const StrSpan* leaves, int32_t n) {
            assert(n >= 1 && "ensureTypedEqualityEntityForTheorem: no head");
            const StrSpan head = leaves[n - 1];
            if (!startsWithSpan(head, "(=[", 3)) return;
            VarTypeTable table;
            for (int32_t i = 0; i < n; ++i) typeLeafInto(leaves[i], table);
            StrSpan args[ExecutionParameters::MAX_ARITY];
            getArgsSpans(head, args, ExecutionParameters::MAX_ARITY);
            const StrSpan label = resolvedLabel(table, table.indexOf(args[0]));
            if (!equalSpans(label, StrSpan("(1)", 3))) structuralCompiler.ensureTypedEqualityEntity(label);
        }

public:
/// @brief Collect the "digit" (induction-variable) arguments of an
        ///        admission theorem — its `(1)`-typed config input args that
        ///        occupy only input slots and no anchor element — into a caller
        ///        span buffer, sorted-unique; 0% heap.
        ///
        /// @details
        /// The span twin of the former `std::set<std::string>` collector. The
        /// unused `coreExpressionMap` parameter is dropped: compiled-definition
        /// lookups route through the member `coreConfig(StrSpan)` accessor
        /// (nullptr = defined miss, I-137). The chain is disintegrated into stack
        /// `StrSpan`s over @p theorem; the validity/operator scan reads
        /// `operators` via a transparent `std::string_view` probe and the config
        /// via `coreConfig`; an invalid element returns 0 (the former empty set).
        /// The candidates and the anchor-element args are gathered as `StrSpan`
        /// arrays, then the result = distinct(candidates) minus the anchor args
        /// minus every candidate `occursOutsideInputSlot` reports,
        /// `compareSpans`-sorted (byte-identical to the former
        /// `std::set<std::string>` order). No arena; output spans alias
        /// @p theorem.
        ///
        /// A candidate is an argument at a configured input slot whose
        /// definition set is `(1)` (D-413). An input argument of `=` counts only
        /// when the theorem types it `(1)` (`typeLeafInto` over the whole chain,
        /// I-267): `=` may relate sets or relations, and only digits are
        /// induction variables. A lone `=` goal types to the starting point
        /// `(1)`, so its arguments count as before.
        ///
        /// A candidate is an induction variable only when it is always an input
        /// (D-580): one occurrence at an
        /// output slot, or at a slot its element's configuration lists as
        /// neither input nor output, discards it, and so does any occurrence in
        /// an anchor element. The arguments of a property (one input, no output)
        /// outside its input slot are never candidates; D-565's property rule
        /// is retired.
        ///
        /// @param theorem The admission theorem (span source; must outlive the use).
        /// @param anchor  The anchor whose args are removed in step 4 (the member
        ///                `anchorInfo` still gates the step-2 continue, matching the
        ///                former body's mixed `anchorInfo` / `anchor` use).
        /// @param out     Caller buffer receiving the distinct sorted digit args.
        /// @param cap     Capacity of @p out (loud Rule-19 assert on overflow).
        /// @return The number of distinct digit args written to @p out (0 if the
        ///         theorem is invalid).
        /// @invariant Allocates nothing on the heap; output spans alias @p theorem.
        /// @see coreConfig, occursOutsideInputSlot, disintegrateImplicationSpans,
        ///      extractExpressionSpan, extractExpressionUniversalSpan, compareSpans.
inline int32_t findDigitArgs(StrSpan theorem,
                                     const ce::AnchorInfo& anchor,
                                     StrSpan* out, int32_t cap) {
            // 1. Disintegrate the implication chain: each triple's key + the head,
            //    all zero-copy slices of `theorem`.
            StrSpan chain[64];
            int32_t chainN = 0;
            StrSpan headSpan;
            ce::disintegrateImplicationSpans(theorem, headSpan,
                [&](StrSpan keySpan, const StrSpan*, int32_t) {
                    assert(chainN < 64 && "findDigitArgs: implication chain exceeds 64");
                    chain[chainN++] = keySpan;
                });
            assert(chainN < 64 && "findDigitArgs: implication chain exceeds 64");
            chain[chainN++] = headSpan;

            // 2. Validity/operator scan (return 0 = the former empty set on invalid).
            for (int32_t e = 0; e < chainN; ++e) {
                const StrSpan element = chain[e];
                // Allow equalities (including negated) and anchor-bearing elements.
                if (canonicalizer.isEquality(element)
                    || containsSpan(element, StrSpan(anchorInfo.name))) continue;

                // Strip a leading negation "!(...)" -> "(...)" (a slice; the last
                // byte is the closing ')', so dropping the '!' is byte-identical to
                // the former substr(1,size-2)+")").
                StrSpan stripped = element;
                if (element.len >= 2 && element.ptr[0] == '!' && element.ptr[1] == '(')
                    stripped = StrSpan(element.ptr + 1, element.len - 1);
                if (canonicalizer.isEquality(stripped)) continue;

                // Registered operator or compiled expression, else a compiled
                // existence head, else invalid.
                const StrSpan core = extractExpressionSpan(stripped);
                const bool inOperators = this->definitions.operatorNames().find(
                    std::string_view(core.ptr, static_cast<size_t>(core.len)))
                    != this->definitions.operatorNames().end();
                if (!inOperators && this->definitions.coreConfig(core) == nullptr) {
                    if (core.len >= 9 && std::memcmp(core.ptr, "existence", 9) == 0)
                        continue;  // compiled existence head
                    return 0;      // invalid
                }
            }

            // 3. Collect input args from each element's config inputIndices. An
            //    `=` argument counts only when the theorem types it `(1)`: `=`
            //    relates values of any definition set (I-267)
            //    and only digits are induction variables. The inference runs
            //    once, on the first `=` element, over the whole chain.
            StrSpan inputBuf[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
            int32_t inputN = 0;
            VarTypeTable eqTypes;
            bool eqTyped = false;
            for (int32_t e = 0; e < chainN; ++e) {
                const StrSpan core = extractExpressionUniversalSpan(chain[e]);
                const ce::CoreExpressionConfig* cfg = this->definitions.coreConfig(core);
                if (cfg != nullptr && !cfg->inputIndices.empty()) {
                    StrSpan args[ExecutionParameters::MAX_ARITY];
                    const int32_t argsN = getArgsSpans(chain[e], args,
                                                       ExecutionParameters::MAX_ARITY);
                    const bool isEq = equalSpans(core, StrSpan("=", 1));
                    if (isEq && !eqTyped) {
                        for (int32_t k = 0; k < chainN; ++k) typeLeafInto(chain[k], eqTypes);
                        eqTyped = true;
                    }
                    for (int idx : cfg->inputIndices) {
                        if (idx >= 0 && idx < argsN) {
                            if (isEq && !equalSpans(
                                    resolvedLabel(eqTypes, eqTypes.indexOf(args[idx])),
                                    StrSpan("(1)", 3)))
                                continue;
                            // Any other input counts only when its definition
                            // slot is typed `(1)`: a set- or relation-valued
                            // input (the FTA `sequence` input `seq`) is never
                            // an induction variable
                            // (D-413).
                            if (!isEq) {
                                const std::pair<std::string, bool>* slotDefinition = nullptr;
                                for (const auto& slotEntry : cfg->definitionSets) {
                                    int slot = 0;
                                    for (const char digit : slotEntry.first) {
                                        assert(digit >= '0' && digit <= '9'
                                            && "findDigitArgs: definition slot key must be decimal");
                                        slot = slot * 10 + (digit - '0');
                                    }
                                    if (slot == idx + 1) {
                                        slotDefinition = &slotEntry.second;
                                        break;
                                    }
                                }
                                assert(slotDefinition != nullptr
                                    && "findDigitArgs: every configured input slot has a definition set");
                                if (slotDefinition->first != "(1)")
                                    continue;
                            }
                            assert(inputN < ExecutionParameters::MAX_ADMISSION_REM_ARGS
                                && "findDigitArgs: input-arg count exceeds cap");
                            inputBuf[inputN++] = args[idx];
                        }
                    }
                }
            }

            // 4. Removal set: every arg of an anchor-bearing element (the former
            //    "remove anchor args").
            StrSpan removeBuf[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
            int32_t removeN = 0;
            for (int32_t e = 0; e < chainN; ++e) {
                if (containsSpan(chain[e], StrSpan(anchor.name))) {
                    StrSpan args[ExecutionParameters::MAX_ARITY];
                    const int32_t argsN = getArgsSpans(chain[e], args,
                                                       ExecutionParameters::MAX_ARITY);
                    for (int32_t a = 0; a < argsN; ++a) {
                        assert(removeN < ExecutionParameters::MAX_ADMISSION_REM_ARGS
                            && "findDigitArgs: removal-arg count exceeds cap");
                        removeBuf[removeN++] = args[a];
                    }
                }
            }

            // 5. Result = distinct(inputBuf) minus the anchor args minus every
            //    candidate that occupies a slot other than an input slot of a
            //    configured element anywhere in the chain (an output slot, or a
            //    slot that is neither input nor output), compareSpans-sorted
            //    (byte-identical to the former std::set<std::string> order).
            const auto lessSpan = [](const StrSpan& a, const StrSpan& b) {
                return compareSpans(a, b) < 0;
            };
            std::sort(inputBuf, inputBuf + inputN, lessSpan);
            std::sort(removeBuf, removeBuf + removeN, lessSpan);
            int32_t outN = 0;
            for (int32_t i = 0; i < inputN; ++i) {
                if (i > 0 && equalSpans(inputBuf[i], inputBuf[i - 1])) continue;  // dedup
                if (std::binary_search(removeBuf, removeBuf + removeN,
                                       inputBuf[i], lessSpan)) continue;  // anchor arg
                if (occursOutsideInputSlot(chain, chainN, inputBuf[i])) continue;
                assert(outN < cap && "findDigitArgs: result count exceeds out cap");
                out[outN++] = inputBuf[i];
            }
            return outN;
        }

        /// @brief Whether an induction-variable candidate occupies a slot other
        ///        than an input slot of some configured element of a theorem's
        ///        chain.
        ///
        /// @details
        /// The always-input test of `findDigitArgs`: a candidate qualifies as an
        /// induction variable only when every one of its occurrences in a
        /// configured element (one with a core configuration, `coreConfig`) is
        /// at one of that element's configured input slots. An occurrence at a
        /// configured output slot, or at a slot the configuration lists as
        /// neither input nor output, disqualifies it. Elements without a core
        /// configuration (compiled existence heads) have no input or output
        /// slots and are not read. `=` lists both of its arguments as inputs.
        ///
        /// @param chain     The theorem's premises and head (zero-copy spans).
        /// @param chainN    Number of elements in @p chain.
        /// @param candidate The candidate argument name.
        /// @return True when some configured element holds @p candidate at a
        ///         slot outside its input slots.
        /// @invariant Allocates nothing on the heap.
        /// @see findDigitArgs, coreConfig.
        bool occursOutsideInputSlot(const StrSpan* chain, int32_t chainN, StrSpan candidate) {
            for (int32_t e = 0; e < chainN; ++e) {
                const StrSpan core = extractExpressionUniversalSpan(chain[e]);
                const ce::CoreExpressionConfig* cfg = this->definitions.coreConfig(core);
                if (cfg == nullptr) continue;
                StrSpan args[ExecutionParameters::MAX_ARITY];
                const int32_t argsN = getArgsSpans(chain[e], args,
                                                   ExecutionParameters::MAX_ARITY);
                for (int32_t a = 0; a < argsN; ++a) {
                    if (!equalSpans(args[a], candidate)) continue;
                    bool inputSlot = false;
                    for (int idx : cfg->inputIndices)
                        if (idx == a) { inputSlot = true; break; }
                    if (!inputSlot) return true;
                }
            }
            return false;
        }

public:
/// @brief String-form delegator for `findDigitArgs` — forwards to the
        ///        `StrSpan` implementation.
        ///
        /// @details Preserved so `std::string`-holding callers (and the retained
        /// twin test) reach the collector without an ambiguous overload. Builds a
        /// zero-copy `StrSpan` over @p theorem and forwards; byte-identical to the
        /// span form (which contains the whole body).
        ///
        /// @param theorem The admission theorem (string source; must outlive the use).
        /// @param anchor  The anchor whose args are removed (forwarded).
        /// @param out     Caller buffer receiving the distinct sorted digit args.
        /// @param cap     Capacity of @p out.
        /// @return The number of distinct digit args written to @p out.
        /// @invariant Allocates no heap beyond the caller-owned @p theorem.
        /// @see findDigitArgs(StrSpan, const ce::AnchorInfo&, StrSpan*, int32_t).
inline int32_t findDigitArgs(const std::string& theorem,
                                     const ce::AnchorInfo& anchor,
                                     StrSpan* out, int32_t cap) {
            return findDigitArgs(StrSpan(theorem), anchor, out, cap);
        }
private:
const CompiledDefinitionRegistry& definitions;
StructuralCompiler& structuralCompiler;
Canonicalizer& canonicalizer;
const ce::AnchorInfo& anchorInfo;
};
}

namespace gl {
/// @brief Bind definition readers and typed-equality compilation.
/// @details Inference tables remain caller-owned; block type records keep their existing stores.
/// @param definitions Read-only configured and compiled definition metadata.
/// @param structuralCompiler Compiler operation for required typed equality entities.
/// @param canonicalizer Existing equality-shape queries.
/// @param anchorInfo Batch anchor used by the original validity scan.
/// @return Constructed scope type resolver.
inline ScopeTypeResolver::ScopeTypeResolver(const CompiledDefinitionRegistry& definitions, StructuralCompiler& structuralCompiler, Canonicalizer& canonicalizer, const ce::AnchorInfo& anchorInfo)
 : definitions(definitions), structuralCompiler(structuralCompiler), canonicalizer(canonicalizer), anchorInfo(anchorInfo) {}
} // namespace gl
