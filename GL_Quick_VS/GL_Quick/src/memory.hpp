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

#include "compiler.hpp"

namespace gl {



    /// @brief One element of a request *Instruction*: a typed (category, signature, arity)
    /// description of a single logical entity plus its bound elements.
    ///
    /// @details
    /// Used inside `Instruction::data` (a sequence of `LogicalEntity` values) to describe
    /// a logical pattern the prover wants to match or instantiate. The fields are deliberately
    /// stringly typed because the same struct is keyed on `std::set` / `std::map` for
    /// deduplication and admission-map indexing.
    ///
    /// `category` is one of the high-level conceptual labels (e.g. "expression",
    /// "anchor", "operator"); `signature` is the concrete operator name; `elements`
    /// holds the bound argument names; `arity` is the declared arity; `definedSet`
    /// names the type set the entity belongs to (added to track the typing-set
    /// per the Pass-B / induction-typing fix scaffolding — see I-18).
    ///
    /// `operator<` orders by (category, signature, elements, arity, definedSet) so the
    /// struct works as a stable key in `std::set<LogicalEntity>` and inside compound
    /// containers like `std::map<Instruction, ...>`.
    ///
    /// @see [`Instruction`](#instruction) — owns a `std::vector<LogicalEntity> data`.
    /// @see [`HashMemory::admissionMapIntegration`](#hashmemory) — keyed on `Instruction`,
    ///      so transitively keyed on this type.
    struct LogicalEntity {
        // Member Variables
        std::string category = "Uncategorized";
        std::vector<std::string> elements = {};
        std::string signature = "";
        int arity = 0;
        std::string definedSet = ""; // <--- NEW FIELD

        // 1. Default Constructor
        LogicalEntity() = default;

        // 2. Parameterized Constructor
        LogicalEntity(std::string cat, std::vector<std::string> elems, std::string sig, int ar, std::string ds = "")
            : category(std::move(cat)),
            elements(std::move(elems)),
            signature(std::move(sig)),
            arity(ar),
            definedSet(std::move(ds)) {
        }

        // 3. Helper to add items easily
        void addElement(const std::string& element) {
            elements.push_back(element);
        }

        // 4. Operator < for std::set usage
        bool operator<(const LogicalEntity& rhs) const {
            if (category != rhs.category) return category < rhs.category;
            if (signature != rhs.signature) return signature < rhs.signature;
            if (elements != rhs.elements) return elements < rhs.elements;
            if (arity != rhs.arity) return arity < rhs.arity;
            return definedSet < rhs.definedSet; // Include new field
        }

        bool operator==(const LogicalEntity& rhs) const {
            return category == rhs.category &&
                signature == rhs.signature &&
                elements == rhs.elements &&
                arity == rhs.arity &&
                definedSet == rhs.definedSet; // Include new field
        }
    };

    /// @brief Value record stored in `HashMemory::encodedMap` — the cached effect of
    /// firing a single hash-engine rule.
    ///
    /// @details
    /// One `LocalMemoryValue` (LMV) describes a single (premise-key → head) entry of
    /// the hash engine. Multiple LMVs may share a key when several distinct
    /// implications collapse onto the same normalized premise pattern (e.g. mirror
    /// statements after `addEquality` symmetrization). The fields:
    ///
    /// - `value`            — the head expression (what the rule produces).
    /// - `levels`           — the set of LB levels at which the rule is admissible.
    /// - `originalImplication` — the producer rule's full implication text, kept for
    ///   provenance and origin chaining.
    /// - `justification`    — the proof-tag string emitted on firing
    ///   (`hash`, `head`, `equality1`, etc.).
    /// - `key`              — the human-readable normalized key (string form). The
    ///   int16_t mirror lives in `IntNormalizedKey` and is what the unordered_map
    ///   actually keys on; this string form is retained for diagnostics.
    /// - `remainingArgs`    — argument names that were not consumed by the key match
    ///   and must be carried into the head substitution.
    /// - `validityName`     — the validity scope at which the rule fires. `"main"` is
    ///   the canonical root; non-`"main"` names are minted via `NameMap::encodePush`
    ///   per [I-2](../../docs/30_invariants.md#i-2).
    /// - `productOfDisintegration` — D-32 marker. True when at least one premise of
    ///   the producing implication carries a `u_`-prefixed arg (i.e. the implication
    ///   itself is a product of an earlier disintegration step). Set in
    ///   `addToHashMemory`; consumed by `checkLocalEncodedMemoryStatic` to gate
    ///   OR-disintegration on the head-firing path.
    ///
    /// `operator<` is lexicographic over the user-visible fields so the struct
    /// serves as a key in `std::set<LocalMemoryValue>` (used at a few inspection
    /// sites) without depending on `productOfDisintegration` (which is a runtime
    /// classification of the same content).
    ///
    /// @see [`HashMemory`](#hashmemory) — owns the `encodedMap` keyed on
    ///      `IntNormalizedKey`, valued as `std::vector<LocalMemoryValue>`.
    /// @see `prover.hpp::addToHashMemory` — installs LMVs.
    /// @see `prover.hpp::checkLocalEncodedMemoryStatic` — reads LMVs on hash hit.
    /// @see [D-32 in 40_decisions.md](../../docs/40_decisions.md#d-32) for the
    ///      `productOfDisintegration` rationale.
    struct LocalMemoryValue {
        std::string value;
        std::set<int> levels;
        std::string originalImplication;
        std::string justification;
        std::vector<std::string> key;
        std::set<std::string> remainingArgs;
        std::string validityName;
        // D-32: marks LMVs whose implication is itself a "product of
        // disintegration" — at least one premise has an arg starting
        // with "u_". Set at install time in addToHashMemory; consumed by
        // checkLocalEncodedMemoryStatic to gate OR-disintegration on
        // the head-firing path. Default false (admission/marker LMVs
        // and non-disintegration-product implications).
        bool productOfDisintegration;

        // Constructors
        LocalMemoryValue()
            : value(),
            levels(),
            originalImplication(),
            justification(),
            key(),
            remainingArgs(),
            validityName("main"),
            productOfDisintegration(false) {
        }

        LocalMemoryValue(const std::string& value_,
            const std::set<int>& levels_,
            const std::string& originalImplication_,
            const std::string& justification_,
            const std::vector<std::string>& key_,
            const std::set<std::string>& remainingArgs_,
            const std::string& validityName_ = "main")
            : value(value_),
            levels(levels_),
            originalImplication(originalImplication_),
            justification(justification_),
            key(key_),
            remainingArgs(remainingArgs_),
            validityName(validityName_),
            productOfDisintegration(false) {
        }

        // Ordering so it can be stored in std::set<LocalMemoryValue>
        // Lexicographic over (value, levels, originalImplication, key, remainingArgs, validityName).
        bool operator<(const LocalMemoryValue& rhs) const {
            if (value != rhs.value) {
                return value < rhs.value;
            }
            if (levels != rhs.levels) {
                return levels < rhs.levels;
            }
            if (originalImplication != rhs.originalImplication) {
                return originalImplication < rhs.originalImplication;
            }
            if (justification != rhs.justification) {
                return justification < rhs.justification;
            }
            if (key != rhs.key) {
                return key < rhs.key;
            }
            if (remainingArgs != rhs.remainingArgs) {
                return remainingArgs < rhs.remainingArgs;
            }
            if (validityName != rhs.validityName) {
                return validityName < rhs.validityName;
            }
            return false; // equal in ordering terms
        }
    };



    /// @brief One entry in the *admission map* — pre-computed parameters that bound how
    /// far a hash-engine rule may be applied at a given expression / validity pair.
    ///
    /// @details
    /// The admission map (`HashMemory::admissionMap`, keyed on `ExpressionWithValidity`
    /// → `std::set<AdmissionMapValue>`) caches the work of `updateAdmissionMap3` so
    /// repeated firings of the same rule under the same scope do not have to
    /// re-derive the depth budget. Each `AdmissionMapValue` carries:
    ///
    /// - `key` and `remainingArgs` — the same shapes used inside `LocalMemoryValue`,
    ///   so the admission record can be matched against the rule's own key during
    ///   `checkLocalEncodedMemoryStatic`.
    /// - `standardMaxAdmissionDepth` — the maximum recursive depth at which the rule
    ///   is allowed to fire from this anchor point. Caps blow-up.
    /// - `standardMaxSecondaryNumber` — the maximum secondary-iteration counter the
    ///   rule is allowed to consume.
    /// - `flag` — admission-side discriminator; `true` for marker LMVs (the entries
    ///   that exist only to gate other rules), `false` for ordinary head LMVs.
    ///
    /// The struct is value-equal under lexicographic ordering of all fields, which
    /// lets it live in the `std::set<AdmissionMapValue>` slot of `admissionMap`.
    ///
    /// @see [`HashMemory::admissionMap`](#hashmemory) — owns these values.
    /// @see `prover.hpp::updateAdmissionMap3` — populates new entries.
    /// @see [I-22](../../docs/30_invariants.md#i-22) — `rejectedMapIntegration`
    ///      revival does NOT clean the admission-map entry; the two side-tables
    ///      are intentionally not coupled.
    struct AdmissionMapValue {
        // Fields (camelCase)
        std::vector<std::string> key;
        std::set<std::string> remainingArgs;
        int standardMaxAdmissionDepth;
        int standardMaxSecondaryNumber;
        bool flag;

        // Default constructor
        AdmissionMapValue()
            : key(),
            remainingArgs(),
            standardMaxAdmissionDepth(0),
            standardMaxSecondaryNumber(0),
            flag(false) {
        }

        // Full constructor
        AdmissionMapValue(const std::vector<std::string>& key_,
            const std::set<std::string>& remainingArgs_,
            int standardMaxAdmissionDepth_,
            int standardMaxSecondaryNumber_,
            bool flag_)
            : key(key_),
            remainingArgs(remainingArgs_),
            standardMaxAdmissionDepth(standardMaxAdmissionDepth_),
            standardMaxSecondaryNumber(standardMaxSecondaryNumber_),
            flag(flag_) {
        }

        // Ordering so it can be used in std::set<AdmissionMapValue>
        // Lexicographic over (key, remainingArgs, standardMaxAdmissionDepth, standardMaxSecondaryNumber, flag)
        bool operator<(const AdmissionMapValue& rhs) const {
            if (key != rhs.key) {
                return key < rhs.key; // std::vector<std::string> has lexicographic operator<
            }
            if (remainingArgs != rhs.remainingArgs) {
                return remainingArgs < rhs.remainingArgs; // std::set<std::string> lexicographic
            }
            if (standardMaxAdmissionDepth != rhs.standardMaxAdmissionDepth) {
                return standardMaxAdmissionDepth < rhs.standardMaxAdmissionDepth;
            }
            if (standardMaxSecondaryNumber != rhs.standardMaxSecondaryNumber) {
                return standardMaxSecondaryNumber < rhs.standardMaxSecondaryNumber;
            }
            if (flag != rhs.flag) {
                return flag < rhs.flag; // false < true
            }
            return false; // equal in ordering terms
        }
    };




    /// @brief Legacy string-form of a hash-engine normalized key.
    ///
    /// @details
    /// Predecessor of `IntNormalizedKey`. Each `data` entry is an alternating
    /// expression-name / variable-name string sequence; `numberExpressions` is the
    /// count of expression segments. Read by older paths that have not yet been
    /// migrated to the int16-only static path; new code keys hash memory on
    /// `IntNormalizedKey` instead. The string-form key is kept primarily for
    /// debug printing and for cross-checks during the static-mirror build-up
    /// in `addToHashMemory`.
    ///
    /// `operator<` and `operator==` give lexicographic ordering over
    /// `(numberExpressions, data)` so the struct can be used as a `std::map` key
    /// or a `std::set` element.
    ///
    /// @see [`IntNormalizedKey`](#intnormalizedkey) — int16_t replacement on the
    ///      hot path.
    struct NormalizedKey {
        // Fields (camelCase)
        int numberExpressions;
        std::vector<std::string> data;

        // Constructors
        NormalizedKey()
            : numberExpressions(0), data() {
        }

        NormalizedKey(int numberExpressions_, const std::vector<std::string>& data_)
            : numberExpressions(numberExpressions_), data(data_) {
        }

        // Ordering so it can be used in std::set<> and as a std::map<> key.
        // Lexicographic over (numberExpressions, data).
        bool operator<(const NormalizedKey& rhs) const {
            if (numberExpressions != rhs.numberExpressions) {
                return numberExpressions < rhs.numberExpressions;
            }
            return data < rhs.data; // vector<string> has lexicographic operator<
        }

        // Optional equality (useful if you compare directly)
        bool operator==(const NormalizedKey& rhs) const {
            return numberExpressions == rhs.numberExpressions && data == rhs.data;
        }
    };


    /// @brief One rejection record for the disintegration-side admission map.
    ///
    /// @details
    /// Lives in `HashMemory::rejectedMap`, keyed on the rejected
    /// `ExpressionWithValidity`. Used to remember disintegration-rejected products
    /// across iterations so the same rejection isn't re-emitted on every fold of
    /// the same producer rule. The fields:
    ///
    /// - `renamedExpression` — the expression after the disintegration-rename pass
    ///   (variables substituted to fresh names so identity-collision is
    ///   distinguishable across LBs).
    /// - `expression`        — the original (pre-rename) compact-compound expression
    ///   text (the compound that was being disintegrated when this child was
    ///   rejected), retained for origin chaining, human-readable diagnostics, and
    ///   `statementLevelsMap` lookup at revival time.
    /// - `iteration`         — the prover iteration at which the rejection was
    ///   recorded. Used to expire stale rejections when iteration advances.
    /// - `concreteConstituent` — the rejected child element with the `u_` prefix
    ///   stripped (matches the integration-side `RejectedMapIntegrationValue`
    ///   field). This is the post-disintegration form that revival mails to
    ///   `internalMailIn` as the primary constituent.
    /// - `siblings`            — the OTHER body elements of the parent compound
    ///   (same disintegration cohort, in concrete u_-stripped form). Carried so
    ///   the `internalMailIn` revival path can re-emit the full body together.
    /// - `levels`              — the deductive-depth level set inherited from the
    ///   parent compound. Snapshot at rejection time; used as the mail-deposit
    ///   levels at revival.
    ///
    /// `operator<` orders lexicographically over all fields so the struct fits
    /// directly into `std::set<RejectedMapValue>`.
    ///
    /// @see [`HashMemory::rejectedMap`](#hashmemory).
    /// @see [`RejectedMapIntegrationValue`](#rejectedmapintegrationvalue) — the
    ///      integration-side counterpart with no iteration semantics.
    struct RejectedMapValue {
        // Fields (camelCase)
        std::string renamedExpression;
        std::string expression;
        int iteration;
        std::string concreteConstituent;
        std::vector<std::string> siblings;
        std::set<int> levels;

        // Constructors
        RejectedMapValue()
            : renamedExpression(),
            expression(),
            iteration(0),
            concreteConstituent(),
            siblings(),
            levels() {
        }

        RejectedMapValue(const std::string& renamedExpression_,
            const std::string& expression_,
            int iteration_,
            const std::string& concreteConstituent_,
            const std::vector<std::string>& siblings_,
            const std::set<int>& levels_)
            : renamedExpression(renamedExpression_),
            expression(expression_),
            iteration(iteration_),
            concreteConstituent(concreteConstituent_),
            siblings(siblings_),
            levels(levels_) {
        }

        // Ordering so it can be used in std::set<RejectedMapValue>
        // Lexicographic over (renamedExpression, expression, iteration,
        // concreteConstituent, siblings, levels).
        bool operator<(const RejectedMapValue& rhs) const {
            if (renamedExpression != rhs.renamedExpression) {
                return renamedExpression < rhs.renamedExpression;
            }
            if (expression != rhs.expression) {
                return expression < rhs.expression;
            }
            if (iteration != rhs.iteration) {
                return iteration < rhs.iteration;
            }
            if (concreteConstituent != rhs.concreteConstituent) {
                return concreteConstituent < rhs.concreteConstituent;
            }
            if (siblings != rhs.siblings) {
                return siblings < rhs.siblings;
            }
            return levels < rhs.levels;
        }

        // Optional equality (useful for comparisons; not required for std::set)
        bool operator==(const RejectedMapValue& rhs) const {
            return renamedExpression == rhs.renamedExpression
                && expression == rhs.expression
                && iteration == rhs.iteration
                && concreteConstituent == rhs.concreteConstituent
                && siblings == rhs.siblings
                && levels == rhs.levels;
        }
    };

    /// @brief One rejection record for the integration-side admission map.
    ///
    /// @details
    /// Integration counterpart to `RejectedMapValue`. Stored in
    /// `HashMemory::rejectedMapIntegration`, keyed on a non-`(in[...])` constituent's
    /// marker form (the `int_` variable at the `marker` position). Integration has
    /// no iteration semantics (the `int_` mint pulls only `level + startInt` per
    /// the integration-side counter), so this struct has no iteration field.
    ///
    /// - `concreteConstituent` — the body element with the `u_` prefix stripped and
    ///   its `int_` marker intact; this is what the marker keys on.
    /// - `siblings`            — all OTHER body elements of the compound, including
    ///   the typing `(in[...])` element, in concrete form. Carried so the
    ///   `internalMailIn` revival path can re-emit the full body together.
    /// - `compoundExpression`  — the original compound, kept for origin chaining.
    ///
    /// @see [`HashMemory::rejectedMapIntegration`](#hashmemory).
    /// @see [I-22](../../docs/30_invariants.md#i-22) — revival does NOT clean the
    ///      admission-map entry; revival is additive.
    /// @see [I-30](../../docs/30_invariants.md#i-30) —
    ///      `applyEquivalenceClassToRejectedMapIntegration` is additive (existing
    ///      entries are never erased).
    struct RejectedMapIntegrationValue {
        std::string concreteConstituent;        // body element, u_ stripped, int_ intact
        std::vector<std::string> siblings;      // other body elements (incl. in[]), concrete
        std::string compoundExpression;         // original compound, for origin chaining

        RejectedMapIntegrationValue()
            : concreteConstituent(), siblings(), compoundExpression() {}

        RejectedMapIntegrationValue(const std::string& concreteConstituent_,
            const std::vector<std::string>& siblings_,
            const std::string& compoundExpression_)
            : concreteConstituent(concreteConstituent_),
              siblings(siblings_),
              compoundExpression(compoundExpression_) {}

        bool operator<(const RejectedMapIntegrationValue& rhs) const {
            if (concreteConstituent != rhs.concreteConstituent) return concreteConstituent < rhs.concreteConstituent;
            if (siblings != rhs.siblings) return siblings < rhs.siblings;
            return compoundExpression < rhs.compoundExpression;
        }

        bool operator==(const RejectedMapIntegrationValue& rhs) const {
            return concreteConstituent == rhs.concreteConstituent
                && siblings == rhs.siblings
                && compoundExpression == rhs.compoundExpression;
        }
    };

    /// @brief Pair of `(expression, validityName)` used as a `std::map` / `std::set`
    /// key throughout the prover.
    ///
    /// @details
    /// The validity stack matters for almost every map keyed on an expression — the
    /// same expression text in two different scopes is two distinct facts. This
    /// struct couples the expression text with its scope name so the pair can be
    /// used directly as a key (no manual concatenation, no fragile string
    /// delimiters).
    ///
    /// - `original`     — the expression text (typically already disintegrated and
    ///   placed into pretty form).
    /// - `validityName` — the scope id in canonical string form. `"main"` is the
    ///   root; deeper scopes are produced by `NameMap::encodePush`. Per
    ///   [I-2](../../docs/30_invariants.md#i-2), non-`"main"` scopes are minted
    ///   only via `encodePush` so the `pairMap` cache stays consistent.
    ///
    /// `operator<` is lexicographic over `(original, validityName)`; `operator==`
    /// is the natural equality.
    ///
    /// @see [`HashMemory::admissionMap`](#hashmemory) — keyed on this type.
    /// @see [`Memory::exprOriginMap`](#memory) — keyed on this type for origin
    ///      provenance lookups.
    /// @see [I-25](../../docs/30_invariants.md#i-25) — `addStatement` returns
    ///      `ExpressionWithValidity` pairs; cross-scope deposits ride through
    ///      `newStatements` so the receiver scope is preserved end-to-end.
    struct ExpressionWithValidity {
        std::string original;
        std::string validityName;

        // 1. Default Constructor
        ExpressionWithValidity()
            : original(""), validityName("") {
        }

        // 2. Parameterized Constructor
        ExpressionWithValidity(std::string original_, std::string validityName_)
            : original(std::move(original_)), validityName(std::move(validityName_)) {
        }

        // 3. Equality Operator (useful for comparisons)
        bool operator==(const ExpressionWithValidity& other) const {
            return original == other.original && validityName == other.validityName;
        }

        // 4. Less-than Operator (REQUIRED for std::set or std::map keys)
        bool operator<(const ExpressionWithValidity& other) const {
            if (original != other.original) {
                return original < other.original;
            }
            return validityName < other.validityName;
        }
    };

    /// @brief A typed sequence of `LogicalEntity` items plus a marked goal — one
    /// integration-side request shape.
    ///
    /// @details
    /// Used inside `HashMemory::admissionMapIntegration` (keyed on
    /// `ExpressionWithValidity`, valued as `map<Instruction, set<string>>`) to
    /// record, per scope, which structured request bodies have been admitted and
    /// which sibling expressions still gate them. The `data` vector holds the
    /// per-element `LogicalEntity` description; `markedGoal` is the consequent the
    /// instruction is steering toward. Together they form a stable, deduplicated
    /// description of an integration request.
    ///
    /// `operator<` orders lexicographically over `(data, markedGoal)`; both
    /// underlying types provide their own ordering, so the comparison is
    /// well-defined and stable.
    ///
    /// @see [`LogicalEntity`](#logicalentity) — the element type.
    /// @see [`HashMemory::admissionMapIntegration`](#hashmemory) — owns this map.
    struct Instruction {
        // Members
        std::vector<LogicalEntity> data;
        std::string markedGoal;

        // 1. Default Constructor
        Instruction()
            : data(), markedGoal() {
        }

        // 2. Parameterized Constructor
        Instruction(std::vector<LogicalEntity> insts, std::string goal)
            : data(std::move(insts)), markedGoal(std::move(goal)) {
        }

        // 3. Operator < (Required for std::map key)
        // Uses lexicographical comparison: first compares instructions, then markedGoal.
        bool operator<(const Instruction& rhs) const {
            if (data != rhs.data) {
                return data < rhs.data;
                // This works because std::vector and LogicalEntity already implement operator<
            }
            return markedGoal < rhs.markedGoal;
        }

        // 4. Equality Operator
        bool operator==(const Instruction& rhs) const {
            return data == rhs.data && markedGoal == rhs.markedGoal;
        }
    };

    // ========================================================================
    // Static hot path: int16_t-based hash memory structures
    // ========================================================================

    /// Bidirectional string <-> int16_t dictionary. Simple counter, no intelligence.
    ///
    /// A validityName is a LIFO stack of sub-names. Canonical grammar:
    ///
    ///   validityName := Root ( "_boundary_" Payload )*
    ///   Root         := arbitrary string; MUST NOT contain "_boundary_"
    ///   Payload      := arbitrary string; MUST NOT contain "_boundary_"
    ///
    /// The root is typically the literal "main" (the general validity scope),
    /// but any string free of the "_boundary_" delimiter is accepted as a
    /// root. Each deeper scope is formed by appending "_boundary_" + payload.
    /// Prefix relationship between two ids is precomputed into a sparse
    /// pairMap: verdict(a, b) = -1 if a is a strict prefix of b, +1 if b is
    /// a strict prefix of a, 0 if a == b, diverge (absent from map) if neither.
    struct NameMap {
        // Existing — validity-name dictionary.
        std::unordered_map<std::string, int16_t> nameToId;
        std::vector<std::string> idToName;
        int16_t nextId = 1;  // 0 reserved for invalid

        // Sub-name dictionary — interns each distinct payload.
        std::unordered_map<std::string, int16_t> subToId;
        std::vector<std::string> idToSub;  // slot 0 unused
        int16_t nextSubId = 1;

        // Per-validity-id stack of sub-ids; top at back(); empty for root.
        std::vector<std::vector<int16_t>> stackOfValidity;

        // Per-validity-id ancestor list (includes self at back).
        std::vector<std::vector<int16_t>> ancestorsOf;

        // String-level strict-ancestors mirror of `ancestorsOf`. Keyed by
        // descendant canonical validity-name string, maps to the set of its
        // strict-ancestor validity-name strings (self excluded). Populated
        // eagerly inside encodePush in lock-step with `ancestorsOf`.
        std::unordered_map<std::string, std::set<std::string>> stringAncestorsOf;

        // Sparse prefix verdicts. Key packs ordered pair (a, b); value in {-1, +1}.
        // Self-equality (a == a) not stored. Absence = diverge.
        std::unordered_map<uint32_t, int16_t> pairMap;

        static constexpr const char* BOUNDARY_STR = "_boundary_";
        static constexpr std::size_t BOUNDARY_LEN = 10;  // strlen("_boundary_")

        /// Canonical "main" (general validity scope) is always id 1.
        /// Guaranteed by the NameMap constructor pre-registering "main".
        static constexpr int16_t MAIN_ID = 1;

        NameMap() {
            idToName.push_back("");           // slot 0 = invalid
            idToSub.push_back("");            // slot 0 = invalid
            stackOfValidity.emplace_back();   // slot 0 unused
            ancestorsOf.emplace_back();       // slot 0 unused
            // Pre-register "main" so MAIN_ID == 1 invariant holds before any
            // call site encodes it explicitly. Flat-root registration path,
            // no pairMap entries added.
            encode("main");
            assert(nameToId["main"] == MAIN_ID);
        }

        static uint32_t packKey(int16_t a, int16_t b) {
            return (static_cast<uint32_t>(static_cast<uint16_t>(a)) << 16)
                 |  static_cast<uint32_t>(static_cast<uint16_t>(b));
        }

        /// Encode a validity name into a stable id, registering metadata on first sight.
        /// Any string without "_boundary_" inside it is registered as a root with
        /// an empty stack (canonical root = "main"). A string containing
        /// "_boundary_" is split at the last delimiter: everything before it is
        /// recursively encoded as the parent, and the tail is pushed as payload.
        int16_t encode(const std::string& s) {
            auto it = nameToId.find(s);
            if (it != nameToId.end()) return it->second;

            std::size_t lastPos = s.rfind(BOUNDARY_STR, std::string::npos, BOUNDARY_LEN);
            if (lastPos == std::string::npos) {
                // Empty string or legacy flat string — register as a fresh root.
                int16_t id = nextId++;
                nameToId[s] = id;
                idToName.push_back(s);
                stackOfValidity.emplace_back();
                ancestorsOf.emplace_back();
                ancestorsOf.back().push_back(id);
                return id;
            }

            // Has "_boundary_" — recursively register parent prefix, then push payload.
            std::string parent  = s.substr(0, lastPos);
            std::string payload = s.substr(lastPos + BOUNDARY_LEN);
            int16_t parentId = encode(parent);
            return encodePush(parentId, payload);
        }

        /// Intern a payload string, returning a stable sub-id.
        int16_t encodeSub(const std::string& payload) {
            auto it = subToId.find(payload);
            if (it != subToId.end()) return it->second;
            int16_t id = nextSubId++;
            subToId[payload] = id;
            idToSub.push_back(payload);
            return id;
        }

        /// Push `payload` onto `parentId`'s stack, returning id of the new scope.
        /// Deduplicates via canonical string (parent + "_boundary_" + payload).
        /// Asserts payload does not contain the "_boundary_" delimiter.
        int16_t encodePush(int16_t parentId, const std::string& payload) {
            assert(payload.find(BOUNDARY_STR) == std::string::npos
                   && "payload must not contain '_boundary_'");

            std::string canonical = idToName[parentId];
            canonical.append(BOUNDARY_STR, BOUNDARY_LEN);
            canonical.append(payload);

            auto it = nameToId.find(canonical);
            if (it != nameToId.end()) return it->second;

            int16_t subId = encodeSub(payload);
            int16_t newId = nextId++;
            nameToId[canonical] = newId;
            idToName.push_back(canonical);

            // Derive stack from parent.
            stackOfValidity.push_back(stackOfValidity[parentId]);
            stackOfValidity.back().push_back(subId);

            // Derive ancestors from parent + self.
            ancestorsOf.push_back(ancestorsOf[parentId]);
            ancestorsOf.back().push_back(newId);

            // Mirror into stringAncestorsOf. `canonical` is the descendant's
            // full validity-name string; every entry of `ancestorsOf[newId]`
            // except self (which is `newId` itself, at back) is a strict
            // ancestor — record the string form.
            {
                auto& strAncestors = stringAncestorsOf[canonical];
                const std::vector<int16_t>& anc = ancestorsOf.back();
                for (std::size_t i = 0; i + 1 < anc.size(); ++i) {
                    strAncestors.insert(idToName[static_cast<std::size_t>(anc[i])]);
                }
            }

            // Fill pairMap verdicts against all pre-existing ids.
            for (int16_t x = 1; x < newId; ++x) {
                if (x == parentId) {
                    pairMap[packKey(newId, parentId)] = 1;
                    pairMap[packKey(parentId, newId)] = -1;
                    continue;
                }
                int16_t vpx;
                if (!verdict(parentId, x, vpx)) continue;   // diverge → diverge
                if (vpx == -1) continue;                    // parent < x → diverge
                if (vpx == 1) {
                    // x is strict ancestor of parent → strict ancestor of newId.
                    pairMap[packKey(x, newId)] = -1;
                    pairMap[packKey(newId, x)] = 1;
                }
            }
            return newId;
        }

        const std::string& decode(int16_t id) const {
            return idToName[static_cast<std::size_t>(id)];
        }

        const std::vector<int16_t>& stackOf(int16_t id) const {
            return stackOfValidity[static_cast<std::size_t>(id)];
        }

        /// Split a validity name into its ordered payload list.
        /// The root prefix (everything up to the first "_boundary_") is
        /// discarded — the root has an empty stack. Returns payloads in
        /// bottom-to-top order. Returns {} for a pure root (e.g. "main").
        std::vector<std::string> parse(const std::string& s) const {
            std::vector<std::string> out;
            std::size_t pos = s.find(BOUNDARY_STR, 0, BOUNDARY_LEN);
            if (pos == std::string::npos) return out;   // pure root
            pos += BOUNDARY_LEN;
            while (true) {
                std::size_t next = s.find(BOUNDARY_STR, pos, BOUNDARY_LEN);
                if (next == std::string::npos) {
                    out.push_back(s.substr(pos));
                    break;
                }
                out.push_back(s.substr(pos, next - pos));
                pos = next + BOUNDARY_LEN;
            }
            return out;
        }

        /// Lookup verdict. Returns true + value if strictly comparable or a == b.
        /// Returns false when the pair is divergent (not in map).
        bool verdict(int16_t a, int16_t b, int16_t& out) const {
            if (a == b) { out = 0; return true; }
            auto it = pairMap.find(packKey(a, b));
            if (it == pairMap.end()) return false;
            out = it->second;
            return true;
        }

        bool comparable(int16_t a, int16_t b) const {
            if (a == b) return true;
            return pairMap.find(packKey(a, b)) != pairMap.end();
        }

        /// True iff `maybeAncestor` is a strict ancestor of `descendant` in
        /// the validity hierarchy (same relation as pairMap verdict == -1
        /// when packed (maybeAncestor, descendant), i.e. maybeAncestor is a
        /// strict prefix of descendant). Backed by the eagerly populated
        /// `stringAncestorsOf` cache — no string-level computation here.
        inline bool isStrictAncestor(const std::string& maybeAncestor,
                                     const std::string& descendant) const {
            auto it = stringAncestorsOf.find(descendant);
            if (it == stringAncestorsOf.end()) return false;
            return it->second.count(maybeAncestor) > 0;
        }

        /// Caller guarantees comparable(a, b).
        int16_t deeperOf(int16_t a, int16_t b) const {
            if (a == b) return a;
            int16_t v;
            bool ok = verdict(a, b, v);
            (void)ok;
            assert(ok && "deeperOf called on divergent pair");
            return (v >= 0) ? a : b;
        }

        /// String-level deeperOf — returns whichever of `a`, `b` is the
        /// deeper validity in the scope tree. Caller must have established
        /// that `a` and `b` are comparable (one is on the other's
        /// root-to-leaf path). Used by `applyEquivalenceClass` and
        /// `applyEquivalenceClassToRejectedMapIntegration` to route the
        /// rewritten expression to the deeper of (class scope, expr scope).
        inline std::string deeperOf(const std::string& a,
                                    const std::string& b) const {
            if (a == b) return a;
            if (isStrictAncestor(a, b)) return b;   // a shallower
            if (isStrictAncestor(b, a)) return a;   // b shallower
            assert(false && "deeperOf called on divergent string pair");
            return a;
        }
    };

    /// IntNormalizedKey — int16_t array encoding of a normalized key.
    ///
    /// Data layout (repeats for each expression, up to numberExpressions):
    ///
    ///   data[i+0] = expressionID    (NameMap ID of expression name)
    ///   data[i+1] = negation        (1 = negated, 0 = not negated)
    ///   then for each variable of the expression (arity times):
    ///     data[j+0] = varID
    ///     data[j+1] = changeableStatus  (1 = unchangeable, 0 = changeable)
    ///
    ///   If changeableStatus == 1 (unchangeable): varID is a NameMap ID of the
    ///     actual variable name — preserved literally, not normalized.
    ///   If changeableStatus == 0 (changeable): varID is a sequential normalization
    ///     ID (1, 2, 3, ...) assigned by first-appearance order within the key.
    ///
    /// Next expression follows immediately after the last variable pair.
    struct IntNormalizedKey {
        int16_t numberExpressions;
        int16_t length;
        const int16_t* data;  // non-owning pointer — references KeyArena storage

        IntNormalizedKey() : numberExpressions(0), length(0), data(nullptr) {}

        IntNormalizedKey(int16_t numExpr, const int16_t* arr, int16_t len)
            : numberExpressions(numExpr), length(len), data(arr) {}

        bool operator==(const IntNormalizedKey& rhs) const {
            if (numberExpressions != rhs.numberExpressions) return false;
            if (length != rhs.length) return false;
            return std::memcmp(data, rhs.data, length * sizeof(int16_t)) == 0;
        }

        bool operator<(const IntNormalizedKey& rhs) const {
            if (numberExpressions != rhs.numberExpressions) return numberExpressions < rhs.numberExpressions;
            if (length != rhs.length) return length < rhs.length;
            for (int16_t i = 0; i < length; ++i) {
                if (data[i] != rhs.data[i]) return data[i] < rhs.data[i];
            }
            return false;
        }
    };

    /// @brief FNV-1a 64-bit hasher for `IntNormalizedKey`, suitable for
    /// `std::unordered_map` / `std::unordered_set`.
    ///
    /// @details
    /// FNV offset basis 14695981039346656037 / FNV prime 1099511628211, applied
    /// per-byte over `numberExpressions` first, then over each `int16_t` of `data`.
    /// Determinism is essential: the same key produced on two different LBs (or
    /// across two threads in the parallel hash burst) must hash identically so the
    /// dedupe in `addToHashMemory` and the lookup in `checkLocalEncodedMemoryStatic`
    /// agree.
    ///
    /// @see [`IntNormalizedKey`](#intnormalizedkey) — the type being hashed.
    struct IntNormalizedKeyHash {
        std::size_t operator()(const IntNormalizedKey& k) const {
            std::size_t h = 14695981039346656037ULL;
            h ^= static_cast<std::size_t>(k.numberExpressions);
            h *= 1099511628211ULL;
            for (int16_t i = 0; i < k.length; ++i) {
                h ^= static_cast<std::size_t>(static_cast<uint16_t>(k.data[i]));
                h *= 1099511628211ULL;
            }
            return h;
        }
    };

    /// @brief Bump allocator for persisting `int16_t` key data, all-or-nothing release.
    ///
    /// @details
    /// `IntNormalizedKey::data` is a non-owning pointer; the bytes it points at must
    /// outlive the unordered_map entries that key on it. `KeyArena` is the storage
    /// backing those bytes — chunked, append-only, freed wholesale at LB teardown.
    /// Each `store(src, len)` copies `len` `int16_t`s into the current chunk
    /// (allocating a new chunk of `ExecutionParameters::KEY_ARENA_CHUNK` slots when
    /// the current one fills) and returns a stable pointer.
    ///
    /// Lifetime contract: all returned pointers are valid until `release()` (or
    /// destructor) runs. Because the arena hands out interior pointers into
    /// individual chunks, and chunks are never reallocated, the pointers do not
    /// invalidate when more `store` calls happen — unlike `std::vector` whose
    /// elements move on growth.
    ///
    /// @note `KeyArena` allocates with `new int16_t[capacity]`; it is **not** the
    ///       same object as `ChunkPool` (which lives in `prover.hpp` and uses a
    ///       static `char[]` per [I-13](../../docs/30_invariants.md#i-13)).
    ///       The two arenas are conceptually similar but back different storage
    ///       contracts.
    /// @see [`IntNormalizedKey`](#intnormalizedkey) — its `data` pointer references
    ///      memory owned here.
    struct KeyArena {
        std::vector<int16_t*> blocks;
        int16_t* current = nullptr;
        int32_t used = 0;
        int32_t capacity = 0;

        /// Append `len` int16_t slots from `src` into the arena, allocating a fresh
        /// chunk if the current one is full. Returns a stable pointer to the start
        /// of the copied region. Pointer remains valid until `release()`.
        int16_t* store(const int16_t* src, int16_t len) {
            if (used + len > capacity) {
                capacity = ExecutionParameters::KEY_ARENA_CHUNK;
                current = new int16_t[capacity];
                blocks.push_back(current);
                used = 0;
            }
            int16_t* dst = current + used;
            std::memcpy(dst, src, len * sizeof(int16_t));
            used += len;
            return dst;
        }

        /// Free every chunk and reset the arena to empty. Invalidates every
        /// pointer previously returned by `store`. Idempotent — calling on an
        /// already-released arena is a no-op.
        void release() {
            for (auto* b : blocks) delete[] b;
            blocks.clear();
            current = nullptr;
            used = capacity = 0;
        }

        ~KeyArena() { release(); }
    };

    /// @brief Pre-encoded statement for the static request pipeline. All fields
    /// `int16_t`; no heap allocation; fixed size (176 bytes on x64).
    ///
    /// @details
    /// The static request pipeline operates entirely on `int16_t` IDs minted from
    /// the per-LB `NameMap` so the hot path in `generateEncodedRequests*` never
    /// touches strings. Each `IntEncodedExpr` carries the canonicalized form of
    /// one expression: its name id, negation flag, arity, two flags for hypo /
    /// anchor classification, and the four fixed-capacity per-arg arrays
    /// (`argId`, `argUnchangeable`, `argIteration`, `argLevPlus1`, `argFullId`).
    /// Capacity is `ExecutionParameters::MAX_ARITY` per arg array.
    ///
    /// The struct is purely arithmetic — no constructors, no operator overloads,
    /// no virtual table — so it can be `memcpy`'d directly into the typed arena
    /// (`TypedArena<IntEncodedExpr>`). This is what lets the request emitter copy
    /// expressions in O(1) and key on the resulting pointer for dedupe.
    ///
    /// @see [`StaticRequestEmitter`](#staticrequestemitter) — populates an arena
    ///      of these and tracks dedupe.
    /// @see `prover.hpp::encodeExpression` — canonical EncodedExpression →
    ///      IntEncodedExpr converter.
    struct IntEncodedExpr {
        int16_t nameId;          // NameMap ID of expression name
        int16_t negation;        // 0 or 1
        int16_t arity;           // number of arguments (capped at MAX_ARITY)
        int16_t maxIteration;    // max iteration number across args (-1 if none)
        int16_t originalId;      // NameMap ID of original string
        int16_t validityId;      // NameMap ID of validityName
        int16_t isHypo;          // 1 if validityName contains "_hypo_", else 0
        int16_t isAnchor;        // 1 if name starts with "Anchor", else 0
        int16_t argId[ExecutionParameters::MAX_ARITY];           // NameMap ID of arg name
        int16_t argUnchangeable[ExecutionParameters::MAX_ARITY]; // 1=unchangeable, 0=changeable
        int16_t argIteration[ExecutionParameters::MAX_ARITY];    // iteration number per arg (-1 if none)
        int16_t argLevPlus1[ExecutionParameters::MAX_ARITY];     // level+1 per arg (0 if none)
        int16_t argFullId[ExecutionParameters::MAX_ARITY];       // NameMap ID of full arg string (e.g. "it_0_lev_0_1")
    };

    // encodeExpression is defined after EncodedExpression (forward reference).

    /// @brief Per-type bump allocator. Allocates a fixed pool at construction and
    /// hands out aligned slots; resets per LB without freeing the underlying
    /// buffer.
    ///
    /// @details
    /// Used to back the static-request pipeline's per-thread storage of
    /// `IntEncodedExpr`, `StaticRequest`, `MandatoryPair`, and `int16_t` index
    /// arrays. The contract is:
    ///
    /// - Construction with `cap` slots allocates `cap * sizeof(T)` bytes via
    ///   `::operator new`. `cap == 0` means no buffer (subsequent `alloc` is
    ///   undefined behaviour and trips the assert below).
    /// - `alloc(n)` advances the bump pointer by `n` slots and returns the head
    ///   pointer. Asserts that capacity is not exceeded — a firing assert here
    ///   means the per-thread arena was undersized for the LB.
    /// - `reset()` rewinds `used` to 0; the buffer stays alive. Use between LBs.
    /// - Destructor releases the buffer once.
    ///
    /// Move-only: copy construction and assignment are deleted because two
    /// arenas pointing at the same buffer would double-free on destruction.
    /// Move construction transfers ownership and zeroes the source.
    ///
    /// @tparam T element type. The arena makes no constructor calls, so `T` must
    ///         be trivially default-constructible or fully overwritten by callers
    ///         after `alloc`.
    /// @see [`ThreadArenas`](#threadarenas) — assembles four `TypedArena<T>` for
    ///      one thread of the static-request pipeline.
    /// @see [I-19](../../docs/30_invariants.md#i-19) — assert is first-class; the
    ///      capacity assert here is intentional and must not be weakened.
    template<typename T>
    struct TypedArena {
        T* buf;
        int32_t used;
        int32_t capacity;

        explicit TypedArena(int32_t cap = 0) : used(0), capacity(cap), buf(nullptr) {
            if (cap > 0) buf = static_cast<T*>(::operator new(static_cast<std::size_t>(cap) * sizeof(T)));
        }
        TypedArena(const TypedArena&) = delete;
        TypedArena& operator=(const TypedArena&) = delete;
        TypedArena(TypedArena&& o) noexcept : buf(o.buf), used(o.used), capacity(o.capacity) {
            o.buf = nullptr; o.used = 0; o.capacity = 0;
        }
        TypedArena& operator=(TypedArena&& o) noexcept {
            if (this != &o) {
                if (buf) ::operator delete(buf);
                buf = o.buf; used = o.used; capacity = o.capacity;
                o.buf = nullptr; o.used = 0; o.capacity = 0;
            }
            return *this;
        }
        void reset() { used = 0; }
        T* alloc(int32_t n = 1) {
            assert(used + n <= capacity);
            T* p = buf + used;
            used += n;
            return p;
        }
        ~TypedArena() { if (buf) ::operator delete(buf); }
    };

    /// @brief Index pair used by the static `makeMandatory2` pass.
    ///
    /// @details
    /// `makeMandatory2` filters statement pairs whose hash keys are mutually
    /// mandatory (one's key is a strict subset of the other's, modulo
    /// remainingArgs). Each surviving pair is recorded as a `MandatoryPair`
    /// (`idx1`, `idx2` are indices into the per-LB `intEncodedStatements`
    /// vector). The struct is plain pair arithmetic; no ordering or hashing
    /// because consumers walk the array sequentially.
    ///
    /// @see `memory.cpp::makeMandatoryEncodedStatementLists2Static` — emits these.
    struct MandatoryPair {
        int16_t idx1;
        int16_t idx2;
    };

    /// @brief Fully-static request — no string data, no heap allocations.
    ///
    /// @details
    /// One element of the per-LB request stream consumed by the static pipeline.
    /// Each request bundles up to `ExecutionParameters::MAX_EXPRESSIONS` pointers
    /// into a `TypedArena<IntEncodedExpr>` (so the pointers stay stable through
    /// the arena's lifetime), the count of those pointers, the maximum iteration
    /// counter across the bundle, and the precomputed `IntNormalizedKey` used to
    /// look the request up in the hash memory.
    ///
    /// Pointer stability matters: requests are emplaced into a contiguous
    /// `StaticRequest*` output buffer; if the underlying arena were a
    /// `std::vector<IntEncodedExpr>`, every push could reallocate and invalidate
    /// the pointers. The arena is append-only with stable interior pointers, so
    /// pointer-based identity holds through emission.
    ///
    /// @see [`StaticRequestEmitter`](#staticrequestemitter) — produces these.
    struct StaticRequest {
        const IntEncodedExpr* intExprs[ExecutionParameters::MAX_EXPRESSIONS]; // ptrs into arena
        int16_t count;
        int16_t maxIteration;
        IntNormalizedKey normalizedKey;
    };

    /// @brief Pack `(originalId, validityId)` into a single `int32_t` for fast
    /// `std::unordered_set` / `std::map` lookup.
    ///
    /// @details
    /// Concatenates the two `int16_t` halves into a 32-bit value with
    /// `originalId` in the high 16 bits and `validityId` in the low 16. Cheaper
    /// than hashing a `std::pair<int16_t, int16_t>` and equivalent to
    /// `std::tuple<int16_t, int16_t>` ordering for keys with the same
    /// `originalId`. Used by `Memory::intKnownStatements` for the O(1)
    /// "is this statement already known here" check.
    ///
    /// @param originalId NameMap id of the original (un-normalized) expression
    ///                   string.
    /// @param validityId NameMap id of the scope's canonical validity name.
    /// @return Combined key value. Uniqueness is guaranteed because both inputs
    ///         are distinct ids minted by the same `NameMap`.
    inline int32_t packStatementKey(int16_t originalId, int16_t validityId) {
        return (static_cast<int32_t>(static_cast<uint16_t>(originalId)) << 16)
             | static_cast<int32_t>(static_cast<uint16_t>(validityId));
    }

    /// @brief Dedupe + emit for all `generateEncodedRequests*` variants.
    ///
    /// @details
    /// The static request pipeline produces a stream of `StaticRequest` values
    /// destined for the hash memory lookup. Many of those requests turn out to
    /// duplicate previously emitted ones (same expressions in same order under
    /// the same scope). The emitter keeps a `seen` set of `RequestKey` (the
    /// packed `(originalId, validityId)` tuple per expression) and rejects
    /// duplicates before they ever reach the output buffer.
    ///
    /// `emit` flow:
    /// 1. Build a `RequestKey` from the input expressions' packed IDs.
    /// 2. Insert into `seen`. If the entry already existed, return early.
    /// 3. Otherwise copy each `IntEncodedExpr` into the per-thread arena and
    ///    record pointers into the output buffer slot.
    /// 4. Compute and store `maxIteration` across the bundle.
    /// 5. Increment `outCount`.
    ///
    /// The arena copy is what guarantees pointer stability across the rest of
    /// the request stream — without it, the input expression pointers might
    /// dangle when the caller's container reallocates.
    ///
    /// @see [`StaticRequest`](#staticrequest) — output element type.
    /// @see [`TypedArena`](#typedarena) — arena that backs `intExprs` storage.
    struct StaticRequestEmitter {
        struct RequestKey {
            int32_t packed[ExecutionParameters::MAX_EXPRESSIONS];
            int16_t count;
            bool operator<(const RequestKey& rhs) const {
                if (count != rhs.count) return count < rhs.count;
                return std::memcmp(packed, rhs.packed, count * sizeof(int32_t)) < 0;
            }
        };
        std::set<RequestKey> seen;
        int16_t outCount = 0;
        TypedArena<IntEncodedExpr>& exprArena;
        StaticRequest* outBuf;
        int16_t maxOut;

        StaticRequestEmitter(TypedArena<IntEncodedExpr>& arena, StaticRequest* buf, int16_t max)
            : exprArena(arena), outBuf(buf), maxOut(max) {}

        void emit(const IntEncodedExpr* const* exprs, int16_t count,
                  const IntNormalizedKey& nk) {
            if (outCount >= maxOut) return;
            RequestKey dk;
            dk.count = count;
            for (int16_t i = 0; i < count; ++i)
                dk.packed[i] = packStatementKey(exprs[i]->originalId, exprs[i]->validityId);
            if (!seen.insert(dk).second) return;

            StaticRequest& req = outBuf[outCount];
            req.count = count;
            req.maxIteration = -1;
            for (int16_t i = 0; i < count; ++i) {
                IntEncodedExpr* copy = exprArena.alloc(1);
                *copy = *exprs[i];
                req.intExprs[i] = copy;
                if (copy->maxIteration > req.maxIteration)
                    req.maxIteration = copy->maxIteration;
            }
            req.normalizedKey = nk;
            ++outCount;
        }
    };

    /// @brief One candidate produced by the *grow* phase of the static
    /// `makeMandatory` merge pass.
    ///
    /// @details
    /// The grow phase iterates through pre-encoded statements building partial
    /// match-candidate sets. Each surviving partial candidate is recorded as a
    /// `BaseCandidate` (its statement indices in `allIdx[count]` plus the scope
    /// id). The merge phase then folds these candidates into final
    /// `StaticRequest`s. Plain index storage; no ordering or hashing.
    ///
    /// @see `memory.cpp::makeMandatoryEncodedStatementLists2Static` — grow phase.
    struct BaseCandidate {
        int16_t allIdx[ExecutionParameters::MAX_EXPRESSIONS];
        int16_t count;
        int16_t validityId;
    };

    /// @brief Per-thread bundle of typed arenas backing one thread's slice of
    /// the static request pipeline.
    ///
    /// @details
    /// Four sub-arenas, sized for the typical worst case observed on a Gauss
    /// batch:
    ///
    /// - `requests`     (capacity 4096) — `StaticRequest` storage.
    /// - `encodedExprs` (capacity 2048) — `IntEncodedExpr` copies.
    /// - `pairs`        (capacity 8192) — `MandatoryPair`s.
    /// - `indices`      (capacity 4096) — `int16_t` index arrays.
    ///
    /// `reset()` rewinds all four arenas to zero use; called once per LB-
    /// iteration boundary so storage is reused without going through the
    /// allocator. The destructor releases the underlying buffers.
    ///
    /// @see [`TypedArena`](#typedarena) — the per-arena type.
    /// @see [I-28](../../docs/30_invariants.md#i-28) — cross-LB writes during
    ///      `proveKernel`'s parallel phase are forbidden; per-thread arenas are
    ///      one of the mechanisms that keep that invariant true.
    struct ThreadArenas {
        TypedArena<StaticRequest>   requests;
        TypedArena<IntEncodedExpr>  encodedExprs;
        TypedArena<MandatoryPair>   pairs;
        TypedArena<int16_t>         indices;

        ThreadArenas()
            : requests(4096),
              encodedExprs(2048),
              pairs(8192),
              indices(4096)
        {}

        void reset() {
            requests.reset();
            encodedExprs.reset();
            pairs.reset();
            indices.reset();
        }
    };

    /// @brief FNV-1a 64-bit hasher for `std::set<int16_t>`, used as the key type
    /// in `HashMemory::remainingArgsNormalizedEncodedMap`.
    ///
    /// @details
    /// `std::set` orders its elements, so iteration is deterministic; combined
    /// with the same FNV constants used elsewhere in the file, the hash value
    /// is stable across LBs and across threads. That stability is what allows
    /// the `remainingArgs`-keyed dispatch in `checkLocalEncodedMemoryStatic` to
    /// agree with the inserts performed in `addToHashMemory`.
    ///
    /// @see [`HashMemory::remainingArgsNormalizedEncodedMap`](#hashmemory) — the
    ///      consumer.
    struct SetInt16Hash {
        std::size_t operator()(const std::set<int16_t>& s) const {
            std::size_t h = 14695981039346656037ULL;
            for (int16_t v : s) {
                h ^= static_cast<std::size_t>(static_cast<uint16_t>(v));
                h *= 1099511628211ULL;
            }
            return h;
        }
    };

    /// @brief Central rule store of one *logic block* (LB) — the hash engine's
    /// per-LB state.
    ///
    /// @details
    /// A `Memory` instance owns three `HashMemory` slots (`overallHashMemory`,
    /// `localHashMemory`, `localHashMemoryDelta`); together they index every
    /// rule the prover can fire from this LB. The members fall into three
    /// groups:
    ///
    /// **Hot-path indices (int16_t-keyed)**
    /// - `encodedMap` — primary index. Maps a normalized premise key to the
    ///   stack of `LocalMemoryValue`s that can fire from it.
    /// - `remainingArgsNormalizedEncodedMap` — secondary index by the set of
    ///   carried arg ids; lets `checkLocalEncodedMemoryStatic` skip rules
    ///   whose remaining-args set has no overlap with the candidate.
    /// - `normalizedEncodedKeys`, `normalizedEncodedSubkeys`,
    ///   `normalizedEncodedSubkeysMinusOne`, `normalizedEncodedSubkeysMinusTwo`
    ///   — bloom-filter-style fast-rejection sets keyed at full / 1-short /
    ///   2-short / 3-short subkey lengths.
    /// - `maxKeyLength` — saturating upper bound used to short-circuit lookup
    ///   loops when the candidate exceeds anything stored.
    ///
    /// **Path-independent shared state**
    /// - `originals` — the set of original expression strings ever installed,
    ///   used during dedupe.
    /// - `admissionMap`, `admissionMapIntegration` — depth and instruction
    ///   budgets per `(expression, validity)`. See [`AdmissionMapValue`](#admissionmapvalue).
    /// - `admissionSetIntegration`, `triggersForAdmissionSetIntegration` —
    ///   integration-side admission control.
    /// - `rejectedMap`, `rejectedMapIntegration` — disintegration- and
    ///   integration-side rejection caches respectively. The integration side
    ///   has an extra `varsInRejectedMapIntegrationKeys` cache: a
    ///   monotonically-growing set of non-marker args appearing anywhere in
    ///   `rejectedMapIntegration` keys, used by
    ///   `applyEquivalenceClassToRejectedMapIntegration` to short-circuit
    ///   when a class has no key overlap (was the hot spot on Gauss
    ///   batches with ~10⁵ rmi entries × ~10⁶ class calls).
    /// - `admissionStatusMap` — per-key admission verdict cache.
    /// - `productsOfRecursion`, `productsOfRecursionIds` — string and
    ///   int16_t views of expressions known to be products of a recursion
    ///   block; consulted for OR-disintegration gating.
    /// - `consumedAdmissionKeys`, `revisitInProgress` — re-entrant guards.
    ///
    /// `clear()` wipes every slot; called when a `HashMemory` is reset for
    /// reuse or torn down.
    ///
    /// @see [`Memory`](#memory) — owns three `HashMemory` slots.
    /// @see [`LocalMemoryValue`](#localmemoryvalue) — value type of `encodedMap`.
    /// @see [I-22](../../docs/30_invariants.md#i-22),
    ///      [I-30](../../docs/30_invariants.md#i-30) — invariants on the
    ///      `rejectedMapIntegration` lifecycle.
    /// @see [I-32](../../docs/30_invariants.md#i-32) — cross-pair `equality2`
    ///      emission gated on existing class / LB origin (consumes this state).
    struct HashMemory {
        std::unordered_map<IntNormalizedKey, std::vector<LocalMemoryValue>,
            IntNormalizedKeyHash> encodedMap;
        std::unordered_map<std::set<int16_t>,
            std::unordered_set<IntNormalizedKey, IntNormalizedKeyHash>,
            SetInt16Hash> remainingArgsNormalizedEncodedMap;
        std::unordered_set<IntNormalizedKey, IntNormalizedKeyHash> normalizedEncodedKeys;
        std::unordered_set<IntNormalizedKey, IntNormalizedKeyHash> normalizedEncodedSubkeys;
        std::unordered_set<IntNormalizedKey, IntNormalizedKeyHash> normalizedEncodedSubkeysMinusOne;
        std::unordered_set<IntNormalizedKey, IntNormalizedKeyHash> normalizedEncodedSubkeysMinusTwo;
        int16_t maxKeyLength = 0;
        // --- shared members (path-independent) ---
        std::set<std::vector<std::string>> originals;
        std::map<ExpressionWithValidity, std::set<AdmissionMapValue>> admissionMap;
        std::map<ExpressionWithValidity, std::map<Instruction, std::set<std::string>>> admissionMapIntegration;
        std::set<ExpressionWithValidity> admissionSetIntegration;
        std::set<ExpressionWithValidity> triggersForAdmissionSetIntegration;
        std::map<ExpressionWithValidity, std::set<RejectedMapValue>> rejectedMap;
        // Integration-side rejection buffer — keyed on a non-in[] constituent's
        // marker form. See RejectedMapIntegrationValue for shape + rationale.
        std::map<ExpressionWithValidity, std::set<RejectedMapIntegrationValue>> rejectedMapIntegration;
        // Monotonically-growing cache of non-marker args that appear in any
        // rejectedMapIntegration key. Used by applyEquivalenceClassToRejectedMapIntegration
        // to short-circuit when an eq class has no overlap with any stored
        // key — saves O(|rmi|) walk per class call on batches where rmi is
        // large but most classes are unrelated (observed in Gauss: ~10^5 rmi
        // entries and ~10^6 class calls).
        std::unordered_set<std::string> varsInRejectedMapIntegrationKeys;
        // Symmetric cache for admissionMap, populated at every admissionMap
        // insert. Used by applyEquivalenceClassToAdmissionMap to short-circuit
        // when an eq class has no overlap with any stored admission key.
        std::unordered_set<std::string> varsInAdmissionMapKeys;
        std::map<ExpressionWithValidity, bool> admissionStatusMap;
        std::set<std::string> productsOfRecursion;
        std::unordered_set<int16_t> productsOfRecursionIds;
        std::set<ExpressionWithValidity> consumedAdmissionKeys;
        std::set<ExpressionWithValidity> revisitInProgress;

        HashMemory()
            : encodedMap(), remainingArgsNormalizedEncodedMap(), maxKeyLength(0),
              normalizedEncodedKeys(), normalizedEncodedSubkeys(),
              normalizedEncodedSubkeysMinusOne(), normalizedEncodedSubkeysMinusTwo(),
              originals(), admissionMap(), admissionMapIntegration(),
              admissionSetIntegration(), triggersForAdmissionSetIntegration(),
              rejectedMap(), rejectedMapIntegration(),
              varsInRejectedMapIntegrationKeys(),
              varsInAdmissionMapKeys(),
              admissionStatusMap(), productsOfRecursion(), productsOfRecursionIds(),
              consumedAdmissionKeys(), revisitInProgress()
        {}

        void clear() {
            encodedMap.clear();
            remainingArgsNormalizedEncodedMap.clear();
            normalizedEncodedKeys.clear();
            normalizedEncodedSubkeys.clear();
            normalizedEncodedSubkeysMinusOne.clear();
            normalizedEncodedSubkeysMinusTwo.clear();
            maxKeyLength = 0;
            originals.clear();
            admissionMap.clear();
            admissionMapIntegration.clear();
            admissionSetIntegration.clear();
            triggersForAdmissionSetIntegration.clear();
            rejectedMap.clear();
            rejectedMapIntegration.clear();
            varsInRejectedMapIntegrationKeys.clear();
            varsInAdmissionMapKeys.clear();
            admissionStatusMap.clear();
            productsOfRecursion.clear();
            productsOfRecursionIds.clear();
            consumedAdmissionKeys.clear();
            revisitInProgress.clear();
        }
    };


    // ========================================================================

    /// @brief String-form encoded expression — the legacy expression record kept
    /// alongside `IntEncodedExpr` for paths that have not yet migrated to the
    /// int16-only static pipeline.
    ///
    /// @details
    /// Carries the parsed structure of one expression — name, negation flag,
    /// argument vector (5 fields per argument: `[isUnchangeable, arg, iteration,
    /// levPlus1, argId]`), the maximum iteration counter across all arguments,
    /// the original (raw) string, and the validity scope name.
    ///
    /// Three constructors:
    /// - default — empty record (used for `std::map`/`std::set` insertion).
    /// - parameterized — caller already has the parsed pieces; we normalize
    ///   each raw arg via `parseArgument` and compute `maxIterationNumber`.
    /// - explicit-from-`(original, validityName)` — peels off the leading `!`
    ///   if present, splits the expression name from the args via
    ///   `ce::extractExpression*`, parses each arg.
    ///
    /// `parseArgument` handles two grammar shapes:
    /// 1. The full `it_<iter>_lev_<lev>_<argId>` form (with optional `u_`
    ///    prefix) — produces a 5-tuple with the raw iteration / level + 1 /
    ///    argId broken out.
    /// 2. Anything else — passes through as `[u/False, arg, "-1", "0", arg]`.
    ///
    /// `operator<` orders by `(original, validityName)` so identical text in
    /// different scopes is two distinct entries; `operator==` is a full
    /// field-by-field compare.
    ///
    /// @warning The `arguments` vector's inner shape is positional. Callers
    ///          rely on indices 0..4 having the documented meaning. Adding a
    ///          new positional field requires a coordinated change wherever
    ///          this struct is read.
    /// @see [`IntEncodedExpr`](#intencodedexpr) — int16_t replacement on the
    ///      hot path.
    /// @see `encodeExpression` — converts this into an `IntEncodedExpr`.
    struct EncodedExpression {
        // Fields (CamelCase)
        std::string name;
        bool negation;
        std::vector<std::vector<std::string> > arguments; // each: [isUnchangeable, arg, iteration, levPlus1, argId]
        int maxIterationNumber;
        std::string original;
        std::string validityName;

        // --- Constructors ---
        EncodedExpression()
            : name(),
            negation(false),
            arguments(),
            maxIterationNumber(-1),
            original(),
            validityName() {
        }

        // Build from pre-parsed pieces (you provide name, negation, and raw args; we normalize args like Python)
        EncodedExpression(const std::string& name_,
            bool negation_,
            const std::vector<std::string>& rawArgs,
            const std::string& original_,
            const std::string& validityName_)
            : name(name_),
            negation(negation_),
            arguments(),
            maxIterationNumber(-1),
            original(original_),
            validityName(validityName_) {
            for (std::size_t i = 0; i < rawArgs.size(); ++i) {
                arguments.push_back(parseArgument(rawArgs[i]));
            }
            maxIterationNumber = computeMaxIterationNumber(arguments);
        }

        explicit EncodedExpression(const std::string& original_, const std::string& validityName_)
            : name(),
            negation(false),
            arguments(),
            maxIterationNumber(-1),
            original(original_),
            validityName(validityName_) {
            if (original.size() >= 2 && original[0] == '!' && original[1] == '(') {
                negation = true;
                name = ce::extractExpressionFromNegation(original);
            }
            else {
                name = ce::extractExpression(original);
            }
            const std::vector<std::string> rawArgs = ce::getArgs(original);
            for (std::size_t i = 0; i < rawArgs.size(); ++i) {
                arguments.push_back(parseArgument(rawArgs[i]));
            }
            maxIterationNumber = computeMaxIterationNumber(arguments);
        }

        bool operator<(const EncodedExpression& rhs) const {
            if (original != rhs.original) {
                return original < rhs.original;
            }
            return validityName < rhs.validityName;
        }

        bool operator==(const EncodedExpression& rhs) const {
            return name == rhs.name
                && negation == rhs.negation
                && arguments == rhs.arguments
                && maxIterationNumber == rhs.maxIterationNumber
                && original == rhs.original
                && validityName == rhs.validityName;
        }

        // --- Helpers mirroring Python private methods ---

        // Matches: it_<iter>_lev_<lev>_<argId>
        // Fast path avoids regex; preserves exact output strings used elsewhere.
        static std::vector<std::string> parseArgument(const std::string& argIn) {
            using std::string;
            using std::size_t;

            std::string_view v(argIn);
            bool isU = (v.size() >= 2 && v[0] == 'u' && v[1] == '_');
            if (isU) v.remove_prefix(2); // strip "u_"

            // Quick shape check: "it_" ... "_lev_" ... "_" ...
            if (v.size() >= 7 && v[0] == 'i' && v[1] == 't' && v[2] == '_') {
                size_t p = v.find("_lev_", 3);
                if (p != std::string_view::npos) {
                    size_t p2 = v.find('_', p + 5);
                    if (p2 != std::string_view::npos) {
                        // Pull slices
                        std::string iter(v.substr(3, p - 3));
                        std::string levs(v.substr(p + 5, p2 - (p + 5)));
                        std::string id(v.substr(p2 + 1));

                        // lev + 1
                        int lev = std::atoi(levs.c_str());
                        std::string levPlus1 = std::to_string(lev + 1);

                        std::vector<std::string> out;
                        out.reserve(5);
                        out.emplace_back(isU ? "True" : "False");
                        out.emplace_back(std::string(v)); // arg without "u_"
                        out.emplace_back(std::move(iter));
                        out.emplace_back(std::move(levPlus1));
                        out.emplace_back(std::move(id));
                        return out;
                    }
                }
            }

            // Default branch from Python: ['False/True', arg, '-1', '0', arg]
            std::vector<std::string> out;
            out.reserve(5);
            out.emplace_back(isU ? "True" : "False");
            out.emplace_back(std::string(v));
            out.emplace_back("-1");
            out.emplace_back("0");
            out.emplace_back(std::string(v));
            return out;
        }


        static int computeMaxIterationNumber(const std::vector<std::vector<std::string> >& args) {
            int maxIt = -1;
            for (std::size_t i = 0; i < args.size(); ++i) {
                if (args[i].size() >= 3) {
                    const int it = std::atoi(args[i][2].c_str());
                    if (it > maxIt) maxIt = it;
                }
            }
            return maxIt;
        }
    };

    /// @brief Convert a parsed `EncodedExpression` (string fields) into an
    /// `IntEncodedExpr` (int16_t fields) using a `NameMap` to mint stable ids.
    ///
    /// @details
    /// The string-side `EncodedExpression` is the language-level record; the
    /// int16-side `IntEncodedExpr` is the hot-path record. This function maps
    /// every string field through `nm.encode` (mints a fresh id on first
    /// sight, returns the existing id otherwise) and copies the per-arg flags
    /// directly. Three encoder details worth flagging:
    ///
    /// 1. `argId[i]` stores the *u_-prefixed* version when the argument is
    ///    unchangeable — matching what `makeIntNormalizedKey` does with
    ///    `ignoreU=false`. The non-prefixed full string is in `argFullId[i]`.
    /// 2. Arity is capped at `ExecutionParameters::MAX_ARITY` even if the
    ///    `EncodedExpression` carries more arguments — silent truncation, but
    ///    in practice expressions never exceed the cap (Gauss-tail Anchor at
    ///    arity 6 is the observed maximum).
    /// 3. `isHypo` is derived by substring-search for `_hypo_` in the
    ///    validity name; `isAnchor` by prefix-match `Anchor` on the
    ///    expression name. Both are precomputed here so the hot path can
    ///    branch on a single `int16_t` flag instead of a string scan.
    ///
    /// @pre `ee` was constructed via the canonical pipeline (parser →
    ///      `EncodedExpression(...)` ctor); fields are not partially-populated.
    /// @post Returns an `IntEncodedExpr` whose every id field references `nm`.
    ///       The returned value is trivially copyable; consumers store it
    ///       into `TypedArena<IntEncodedExpr>` or
    ///       `Memory::intEncodedStatements`.
    /// @invariant All non-`"main"` validity names registered through `nm`
    ///            during this call are minted via `NameMap::encode` (which
    ///            either returns an existing id or recurses into `encodePush`
    ///            for `_boundary_`-bearing names). See
    ///            [I-2](../../docs/30_invariants.md#i-2).
    /// @see [`EncodedExpression`](#encodedexpression),
    ///      [`IntEncodedExpr`](#intencodedexpr).
    inline IntEncodedExpr encodeExpression(const EncodedExpression& ee, NameMap& nm) {
        IntEncodedExpr ie;
        ie.nameId = nm.encode(ee.name);
        ie.negation = ee.negation ? 1 : 0;
        ie.arity = static_cast<int16_t>(std::min(ee.arguments.size(),
            static_cast<std::size_t>(ExecutionParameters::MAX_ARITY)));
        ie.maxIteration = static_cast<int16_t>(ee.maxIterationNumber);
        ie.originalId = nm.encode(ee.original);
        ie.validityId = nm.encode(ee.validityName);
        ie.isHypo = (ee.validityName.find("_hypo_") != std::string::npos) ? 1 : 0;
        ie.isAnchor = (ee.name.rfind("Anchor", 0) == 0) ? 1 : 0;

        std::memset(ie.argId, 0, sizeof(ie.argId));
        std::memset(ie.argUnchangeable, 0, sizeof(ie.argUnchangeable));
        std::memset(ie.argIteration, 0, sizeof(ie.argIteration));
        std::memset(ie.argLevPlus1, 0, sizeof(ie.argLevPlus1));
        std::memset(ie.argFullId, 0, sizeof(ie.argFullId));

        for (int16_t i = 0; i < ie.arity; ++i) {
            const std::vector<std::string>& arg = ee.arguments[static_cast<std::size_t>(i)];
            if (arg.size() >= 2) {
                ie.argUnchangeable[i] = (arg[0] == "True") ? 1 : 0;
                ie.argFullId[i] = nm.encode(arg[1]);
                // argId stores u_-prefixed version for unchangeable args,
                // matching what makeIntNormalizedKey does with ignoreU=false
                ie.argId[i] = (arg[0] == "True")
                    ? nm.encode("u_" + arg[1])
                    : nm.encode(arg[1]);
            }
            if (arg.size() >= 3) {
                ie.argIteration[i] = static_cast<int16_t>(std::atoi(arg[2].c_str()));
            } else {
                ie.argIteration[i] = -1;
            }
            if (arg.size() >= 4) {
                ie.argLevPlus1[i] = static_cast<int16_t>(std::atoi(arg[3].c_str()));
            }
        }
        return ie;
    }

    /// @brief Inter-LB message bag — statements + implications + origin map
    /// queued for the receiver to absorb on its next hash burst.
    ///
    /// @details
    /// The mail subsystem is the *only* sanctioned cross-LB write channel
    /// during a parallel hash burst (per
    /// [I-28](../../docs/30_invariants.md#i-28) — "cross-LB writes during
    /// `proveKernel`'s parallel phase forbidden — defer to post-`pool.join()`
    /// collectors"). The flow direction is parent-to-children only; no
    /// upward writes (per [D-51](../../docs/40_decisions.md#d-51)).
    ///
    /// Three slots:
    /// - `statements`     — each element is `(ExpressionWithValidity, levels)`.
    ///   The EWV's `validityName` is `"main"` for routing-channel traffic
    ///   (`mailIn`/`mailOut`) — enforced by `smashMail` / `sendMail`'s
    ///   per-item assert per [I-26](../../docs/30_invariants.md#i-26) —
    ///   and may be non-main for the per-LB integration-revival channel
    ///   `Memory::internalMailIn`. Migrated 2026-05-07 under
    ///   [D-53](../../docs/40_decisions.md#d-53); was previously
    ///   `pair<string, set<int>>` with validity hardcoded to `"main"` at
    ///   the receiver.
    /// - `implications`   — each element is the 5-tuple `(key, value,
    ///   remainingArgs, levels, originalImplication)`. Mail-out
    ///   implications are MAIN-ONLY per
    ///   [I-26](../../docs/30_invariants.md#i-26).
    /// - `exprOriginMap`  — origin-chain mirror, ALL-SCOPES per the same
    ///   invariant. Carries the provenance edges that the verifier needs
    ///   to reconstruct the proof tree on the receiver side.
    ///
    /// @see [I-21](../../docs/30_invariants.md#i-21) — `internalMailIn`
    ///      is cleared at the top of the hashburst (after absorb), not
    ///      at the end.
    /// @see [I-26](../../docs/30_invariants.md#i-26) — channel-membership
    ///      rule for routed traffic.
    struct Mail {
        // statements element: pair<ExpressionWithValidity, levels>. The EWV's
        // validityName is "main" for routing-channel traffic (mailIn/mailOut)
        // — enforced by smashMail / sendMail per-item assert (I-26) — and may
        // be non-main for the per-LB integration-revival channel
        // (Memory::internalMailIn). Migrated 2026-05-07 (D-53, renumbered from main's D-46 on merge into sandbox/incub_fix) — was previously
        // pair<string, set<int>>, with validity hardcoded to "main" at the
        // receiver.
        std::set< std::pair<ExpressionWithValidity, std::set<int> > > statements;
        std::set< std::tuple<std::vector<std::string>, std::string, std::set<std::string>, std::set<int>, std::string> > implications;

        // CHANGED: Tracking multiple origins (Vector of Pairs)
        std::map<ExpressionWithValidity, std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>> exprOriginMap;

        Mail() : statements(), implications(), exprOriginMap() {}
    };

    /// @brief One equivalence class — a set of variables known to be equal
    /// in some scope, together with provenance.
    ///
    /// @details
    /// Equivalence classes are the prover's internal representation of
    /// `(=[a,b])` chains. Whenever `addEquality` fires it either creates a
    /// fresh class with two variables or merges existing classes; whenever
    /// `applyEquivalenceClass` rewrites an expression it routes through this
    /// data structure to look up which variable to substitute.
    ///
    /// Three slots:
    /// - `variables`           — the class members. Membership is symmetric
    ///   and transitive; the order in which members were added is not
    ///   preserved.
    /// - `equalityLevelsMap`   — for each pair of variables, the set of LB
    ///   levels at which the equality is admissible. Used by
    ///   `applyEquivalenceClass` to gate substitution by level.
    /// - `equalityOriginMap`   — origin chains for each `(expression, scope)`
    ///   pair that this class produced. Carries multiple origins per pair
    ///   so the verifier can reconstruct the full provenance tree.
    ///
    /// Class storage lives in `Memory::equivalenceClassesMap` (keyed by an
    /// internal class-id string). Mutations are subject to:
    /// - [I-30](../../docs/30_invariants.md#i-30) —
    ///   `applyEquivalenceClassToRejectedMapIntegration` is additive; the
    ///   original entries are never erased.
    /// - [I-31](../../docs/30_invariants.md#i-31) —
    ///   `updateEquivalenceClasses` ancestor-pass never modifies ancestor-scope
    ///   class state.
    /// - [I-32](../../docs/30_invariants.md#i-32) — cross-pair `equality2`
    ///   emission gated on existing class / LB origin.
    /// - [I-33](../../docs/30_invariants.md#i-33) —
    ///   `mergeTwoEquivalenceClasses` cross-vN preconditions.
    /// - [I-34](../../docs/30_invariants.md#i-34) — cross-substitution
    ///   `equality1` emission gated on existing target origin.
    ///
    /// @see [`Memory::equivalenceClassesMap`](#memory) — owns these.
    /// @see `prover.hpp::applyEquivalenceClass` — consumer.
    /// @see `prover.hpp::mergeTwoEquivalenceClasses` — merger.
    struct EquivalenceClass {
        std::set<std::string> variables;
        std::map<std::set<std::string>, std::set<int> > equalityLevelsMap;

        // CHANGED: Tracking multiple origins
        std::map<ExpressionWithValidity, std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>> equalityOriginMap;

        EquivalenceClass() : variables(), equalityLevelsMap(), equalityOriginMap() {}
        EquivalenceClass(const std::set<std::string>& variables_,
            const std::map<std::set<std::string>, std::set<int> >& equalityLevelsMap_,
            const std::map<ExpressionWithValidity, std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>>& equalityOriginMap_)
            : variables(variables_), equalityLevelsMap(equalityLevelsMap_), equalityOriginMap(equalityOriginMap_) {
        }
    };






    /// @brief One *logic block* (LB) — the prover's primary unit of state.
    ///
    /// @details
    /// A `Memory` instance is one node in the LB tree. The tree's root is the
    /// "main" sentinel (default-constructed, `exprKey.empty()`,
    /// `parentMemory == nullptr`); every non-root LB carries the expression
    /// it owns (`exprKey`) and a back-pointer to its parent (`parentMemory`).
    /// Children are reached via `simpleMap[K] -> Memory*` where `K` is a
    /// routing-key string set at LB creation in `prover.cpp`.
    ///
    /// The `Memory` struct is intentionally large — it bundles every kind of
    /// per-LB state the prover touches:
    ///
    /// **Tree structure**
    /// - `simpleMap`        — routing-key → child LB pointer.
    /// - `parentMemory`     — parent pointer (`nullptr` at the sentinel root).
    /// - `exprKey`          — the expression text owned by this LB. Empty at
    ///   the root sentinel; otherwise carries the disintegrated form. NOTE:
    ///   the `simpleMap` routing key may differ from `exprKey` (the
    ///   recursion-block-#2 site sets `simpleMap[(=[s(recN),zero])] -> child`
    ///   while the child's `exprKey` is the substituted induction-variable
    ///   form `(=[<digitArg>,<zero>])`). Always match a target LB by
    ///   walking the full parent chain to root, never by `exprKey` alone.
    /// - `level`            — depth in the LB tree (`-1` for the root sentinel).
    ///
    /// **Iteration / counter state**
    /// - `startInt`, `startIntRepl`, `startIntPi` — counters for the three
    ///   independent name-mint streams used during disintegration. See
    ///   [I-17](../../docs/30_invariants.md#i-17): the freshness check
    ///   relies on these being a single monotonic stream within their
    ///   respective category.
    /// - `recursionCounter`, `contradictionIndex`, `recursionHypothesis`,
    ///   `isPartOfRecursion`, `primedForContradiction`, `contradictionTheorem`
    ///   — induction- / recursion- / contradiction-LB bookkeeping.
    ///
    /// **Statement state**
    /// - `toBeProved`               — pending obligations on this LB.
    /// - `encodedStatements`        — full string-form statements known here.
    /// - `localEncodedStatements`,
    ///   `localEncodedStatementsSet`,
    ///   `localEncodedStatementsDelta` — local-premise gate inputs (see
    ///   [I-7](../../docs/30_invariants.md#i-7) /
    ///   [D-28](../../docs/40_decisions.md#d-28)). The set mirror is
    ///   maintained in lockstep with the vector for O(log N) membership
    ///   lookups inside `checkLocalEncodedMemoryStatic`.
    /// - `intEncodedStatements`,
    ///   `intLocalEncodedStatements`,
    ///   `intLocalEncodedStatementsDelta` — int16_t mirrors used by the
    ///   static request pipeline.
    /// - `intKnownStatements`       — packed `(originalId, validityId)`
    ///   keys for the O(1) "is this statement already known here" check.
    /// - `statementLevelsMap`       — per-statement set of LB levels at
    ///   which it was deposited.
    /// - `wholeExpressions`         — set of expressions that originated as
    ///   whole assertions (not as derivative products).
    ///
    /// **Hash memory slots** (each is a [`HashMemory`](#hashmemory))
    /// - `overallHashMemory`        — full set of rules visible from this LB.
    /// - `localHashMemory`          — rules installed at this LB.
    /// - `localHashMemoryDelta`     — rules installed since the last burst,
    ///   awaiting absorption.
    ///
    /// **Equivalence classes**
    /// - `equivalenceClassesMap`    — class-id → list of [`EquivalenceClass`](#equivalenceclass).
    /// - `eqClassSttmntIndexMapMap` — class-id → (rmaining-args set →
    ///   statement index) cache used by the cross-pair `equality2` gate
    ///   ([I-32](../../docs/30_invariants.md#i-32)).
    ///
    /// **Mail subsystem**
    /// - `mailIn`, `mailOut`        — routing-channel mail (parent → child;
    ///   [I-26](../../docs/30_invariants.md#i-26) /
    ///   [I-28](../../docs/30_invariants.md#i-28)).
    /// - `internalMailIn`           — per-LB integration-revival channel
    ///   (typed `Mail` since 2026-05-07; see
    ///   [D-53](../../docs/40_decisions.md#d-53)). Cleared at the top of
    ///   the hashburst after absorb per
    ///   [I-21](../../docs/30_invariants.md#i-21).
    ///
    /// **Origin / provenance**
    /// - `exprOriginMap`            — `(expression, scope)` → vector of
    ///   `(tag, list-of-antecedents)` pairs. The ALL-SCOPES half of the
    ///   mail-out contract per
    ///   [I-26](../../docs/30_invariants.md#i-26).
    /// - `integrationStartIntMap`   — per-int_-prefix counter snapshots used
    ///   to keep integration-side renames stable across iterations.
    /// - `integrationPrepared`,
    ///   `integrationPreparedMarker` — integration-side admission tracking.
    /// - `validityNamesToFilter`,
    ///   `intValidityNamesToFilter` — receiver-side mail filter sets.
    /// - `canBeSentSet`,
    ///   `canBeSentMarkerSet`       — sender-side mail-eligibility sets.
    /// - `weakVariables`,
    ///   `axedVariables`,
    ///   `intAxedVariables`         — variable-eligibility classification
    ///   used by `multiplyImplication` and admission.
    ///
    /// **OR-branching state**
    /// - `orAdmissionSet`           — per-LB set of `(disjunct, scope)`s
    ///   that have been admitted; OR fires only when ALL disjuncts in this
    ///   set are present.
    /// - `orBookkeeping`            — convergence tracking: keyed by
    ///   `(derived_expression, or_signature)` → set of branch disjuncts that
    ///   produced it.
    /// - `orDisjunctCount`          — disjunct count per OR expression for
    ///   the convergence check.
    ///
    /// **State flags**
    /// - `isActive`                 — set to `false` by `deactivate*` paths
    ///   when this LB has been retired.
    /// - `deltaNumberStatements`    — count of statements added since the
    ///   last burst boundary; used to short-circuit no-op iterations.
    ///
    /// **Static hot path**
    /// - `nameMap`                  — per-LB `NameMap` (see
    ///   [`NameMap`](#namemap)).
    /// - `keyArena`                 — `KeyArena` backing storage for
    ///   `IntNormalizedKey::data` pointers in this LB.
    ///
    /// The default constructor produces the root sentinel (empty `exprKey`,
    /// `parentMemory == nullptr`, `level == -1`, `isActive == true`,
    /// `recursionCounter == 0`, all containers empty). Tests can rely on
    /// this shape.
    ///
    /// @see [`NameMap`](#namemap), [`HashMemory`](#hashmemory),
    ///      [`Mail`](#mail), [`EquivalenceClass`](#equivalenceclass).
    /// @see The full-LB-chain trace requirement when matching a target
    ///      LB in any debug trap (full parent walk to the root sentinel,
    ///      never `exprKey` alone).
    /// @see The `performElementaryLogicalStep` hashburst dump operating
    ///      on this state is treated as sacred infrastructure: format,
    ///      lambdas, call sites, and target-LB chain match all require
    ///      explicit approval to change.
    struct Memory {
        std::map<std::string, Memory*> simpleMap;
        int startInt;
        int startIntRepl;
        int startIntPi;
        std::map<EncodedExpression, std::tuple<std::set<int>, std::set<std::string>> > toBeProved;
        std::vector<EncodedExpression> encodedStatements;
        std::map<EncodedExpression, std::set<int> > statementLevelsMap;
        std::string exprKey;
        // The concrete "recursion" hypothesis deposited in this LB
        // (tempExpr2 for check_induction_condition, tempExpr4 for
        // check_zero). Empty when the LB is not an induction sub-block.
        // Used by vacuous-truth emission as the 3rd ingredient so the
        // verifier's chapter-local trace can actually reach it (the
        // derivation chains inside the LB use this form, not exprKey).
        std::string recursionHypothesis;
        Memory* parentMemory;
        int level;

        HashMemory overallHashMemory;
        HashMemory localHashMemory;
        HashMemory localHashMemoryDelta;

        std::map<std::string, std::vector<EquivalenceClass>> equivalenceClassesMap;
        std::vector<EncodedExpression> localEncodedStatements;
        // Parallel set keyed by EncodedExpression for O(log N) membership lookups
        // (vector lookup is O(N) and the local-premise gate inside
        // checkLocalEncodedMemoryStatic iterates per disintegration candidate).
        // Maintained in lockstep with localEncodedStatements at every push_back
        // site and the one assignment site at prover.hpp. See I-7 / D-28 for
        // the gate semantics.
        std::set<EncodedExpression> localEncodedStatementsSet;
        std::vector<EncodedExpression> localEncodedStatementsDelta;
        std::map<std::string, int> integrationStartIntMap;

        Mail mailIn;
        Mail mailOut;
        Mail internalMailIn;           // revival inbox for integration-side rejection recovery (typed Mail since 2026-05-07; D-53, renumbered from main's D-46 on merge into sandbox/incub_fix)

        std::set<EncodedExpression> wholeExpressions;
        std::map<std::string, std::map<std::set<std::string>, int>> eqClassSttmntIndexMapMap;
        bool isActive;
        bool isPartOfRecursion;
        int deltaNumberStatements;
        std::map<ExpressionWithValidity, std::vector<std::pair<std::string, std::vector<ExpressionWithValidity> > > > exprOriginMap;
        int recursionCounter;
        int contradictionIndex;
        std::set<ExpressionWithValidity> integrationPrepared;
        std::set<ExpressionWithValidity> integrationPreparedMarker;
        std::set<std::string> validityNamesToFilter;
		std::set<std::string> canBeSentSet;
        std::set<std::string> canBeSentMarkerSet;
        std::set<ExpressionWithValidity> weakVariables;
        std::set<std::string> axedVariables;

        // --- Incubator: contradiction proving ---
        bool primedForContradiction = false;
        std::string contradictionTheorem;

        // --- OR admission: controls which OR expressions fire branch machinery ---
        // Empty = no OR branches fire (implications always fire regardless).
        // Contains ExpressionWithValidity entries (disjuncts with parent validity).
        // OR fires only when ALL its disjuncts are in this set.
        std::set<ExpressionWithValidity> orAdmissionSet;

        // --- OR bookkeeping: convergence tracking ---
        // Key: (derived_expression, or_signature) → set of branch disjuncts that produced it
        // e.g. ("(in2[x,y,z])", "(or3[a,b,c])") → {"(in[7,1])", "(=[7,2])"}
        std::map<std::pair<std::string, std::string>, std::set<std::string>> orBookkeeping;
        // Number of disjuncts per OR expression for convergence check
        std::map<std::string, int> orDisjunctCount;

        // --- Static hot path: int16_t-based hash memory ---
        NameMap nameMap;
        KeyArena keyArena;

        // Pre-encoded int16_t mirrors of encodedStatements / local / delta
        std::vector<IntEncodedExpr> intEncodedStatements;
        std::vector<IntEncodedExpr> intLocalEncodedStatements;
        std::vector<IntEncodedExpr> intLocalEncodedStatementsDelta;

        // Fast O(1) existence check: packStatementKey(originalId, validityId)
        std::unordered_set<int32_t> intKnownStatements;

        // Int mirrors for static addExprToMemoryBlock early checks
        std::unordered_set<int16_t> intValidityNamesToFilter;
        std::unordered_set<int16_t> intAxedVariables;

        Memory()
            : simpleMap(),
            startInt(0),
            startIntRepl(0),
            startIntPi(0),
            toBeProved(),
            encodedStatements(),
            statementLevelsMap(),
            exprKey(),
            recursionHypothesis(),
            parentMemory(nullptr),
            level(-1),
            overallHashMemory(),
            localHashMemory(),
            localHashMemoryDelta(),
            equivalenceClassesMap(),
            localEncodedStatements(),
            localEncodedStatementsDelta(),
			integrationStartIntMap(),
            mailIn(),
            mailOut(),
            internalMailIn(),
            wholeExpressions(),
            eqClassSttmntIndexMapMap(),
            isActive(true),
            isPartOfRecursion(false),
            deltaNumberStatements(0),
            exprOriginMap(),
            recursionCounter(0),
            contradictionIndex(-1),
            integrationPrepared(),
            integrationPreparedMarker(),
            validityNamesToFilter(),
			canBeSentSet(),
			canBeSentMarkerSet(),
			weakVariables(),
			axedVariables(),
			primedForContradiction(false),
			contradictionTheorem(),
            orAdmissionSet(),
            orBookkeeping(),
            orDisjunctCount(),
            nameMap(),
            keyArena(),
            intValidityNamesToFilter(),
            intAxedVariables()
        {
        }


    };




    /// @brief One dependency record — used during the static-pipeline mandatory
    /// merge to track which auxiliary index an expression depends on.
    ///
    /// @details
    /// A `DependencyItem` is one entry in `Dependencies::originalAuxyMap`.
    /// `auxies` is the set of auxiliary indices the expression cares about;
    /// `expr` is the raw expression text; `allLevelsInvolved` is `true` when
    /// the dependency spans every LB level (overrides per-level filtering).
    /// Plain value type, no ordering.
    ///
    /// @see [`Dependencies`](#dependencies) — owns these.
    struct DependencyItem {
        std::set<int> auxies;
        std::string expr;
        bool allLevelsInvolved;

        DependencyItem()
            : auxies(),
            expr(),
            allLevelsInvolved(false) {
        }

        DependencyItem(const std::set<int>& auxies_,
            const std::string& expr_,
            bool allLevelsInvolved_)
            : auxies(auxies_),
            expr(expr_),
            allLevelsInvolved(allLevelsInvolved_) {
        }
    };

    /// @brief Bidirectional auxiliary-index ↔ original-expression map plus
    /// per-original induction-variable bookkeeping.
    ///
    /// @details
    /// Used by the static-pipeline mandatory merge and by induction-block
    /// expansion. The two maps are kept synchronized by the producer:
    ///
    /// - `auxyOriginalMap`               — auxiliary index → original index.
    /// - `originalAuxyMap`               — original index → `DependencyItem`
    ///   (carries the auxiliary-index set, the expression text, and the
    ///   all-levels flag).
    /// - `originalInductionVariableMap`  — original index → `(induction_var,
    ///   typing_set)` pair. Populated only for induction-target originals.
    /// - `auxyIndex`, `originalIndex`    — running counters; the producer
    ///   bumps these on every fresh insertion to keep the indices stable.
    ///
    /// @see [`DependencyItem`](#dependencyitem) — value type of `originalAuxyMap`.
    /// @see [I-18](../../docs/30_invariants.md#i-18) — induction scheduled on
    ///      a bound variable must first prove `(in[n, N])`. The typing set
    ///      lives in `originalInductionVariableMap`'s second tuple slot.
    struct Dependencies {
        std::map<int, int> auxyOriginalMap;
        std::map<int, DependencyItem> originalAuxyMap;
        std::map<int, std::tuple<std::string, std::string>> originalInductionVariableMap;
        int auxyIndex;
        int originalIndex;

        Dependencies()
            : auxyOriginalMap(),
            originalAuxyMap(),
            originalInductionVariableMap(),
            auxyIndex(0),
            originalIndex(0) {
        }

        Dependencies(const std::map<int, int>& auxyOriginalMap_,
            const std::map<int, DependencyItem>& originalAuxyMap_,
            const std::map<int, std::tuple<std::string, std::string>>& originalInductionVariableMap_,
            int auxyIndex_,
            int originalIndex_)
            : auxyOriginalMap(auxyOriginalMap_),
            originalAuxyMap(originalAuxyMap_),
            originalInductionVariableMap(originalInductionVariableMap_),
            auxyIndex(auxyIndex_),
            originalIndex(originalIndex_) {
        }
    };

    /// @brief One *variant* — a paired pair-of-normalized-keys + value text
    /// + mapping-items, used during equivalence-class application to
    /// dedupe rewrite emissions.
    ///
    /// @details
    /// When `applyEquivalenceClass` rewrites an expression, it produces a
    /// `(normalizedIgnoredKey, normalizedNotIgnoredKey, valueVariant,
    /// mappingItems)` tuple. The first two keys distinguish two normalization
    /// modes (with and without ignoring `u_`-prefixed args); the value text
    /// is the rewritten head; `mappingItems` carries the substitution edges
    /// the rewrite performed.
    ///
    /// `operator<` is lexicographic over all four fields so the variant fits
    /// directly into `std::set<VariantItem>` for dedupe.
    ///
    /// @see [I-30](../../docs/30_invariants.md#i-30),
    ///      [I-34](../../docs/30_invariants.md#i-34) — invariants on the
    ///      `applyEquivalenceClass` path that emits these.
    struct VariantItem {
        NormalizedKey normalizedIgnoredKey;
        NormalizedKey normalizedNotIgnoredKey;
        std::string valueVariant;
        std::set<std::pair<std::string, std::string>> mappingItems;

        bool operator<(const VariantItem& rhs) const {
            if (!(normalizedIgnoredKey == rhs.normalizedIgnoredKey)) {
                return normalizedIgnoredKey < rhs.normalizedIgnoredKey;
            }
            if (!(normalizedNotIgnoredKey == rhs.normalizedNotIgnoredKey)) {
                return normalizedNotIgnoredKey < rhs.normalizedNotIgnoredKey;
            }
            if (valueVariant != rhs.valueVariant) {
                return valueVariant < rhs.valueVariant;
            }
            return mappingItems < rhs.mappingItems;
        }
    };

    // ContradictionItem moved to filter.hpp (CE-only type).


    /// @brief Aliases for the two LB-tree-shaped maps the prover passes around.
    ///
    /// @details
    /// `ParentChildrenMap` maps each `Memory*` to the vector of its direct
    /// children — used during deactivation walks and during structural
    /// queries that need O(1) lookup of children by parent. Built once
    /// per iteration from the `simpleMap` edges.
    ///
    /// `PerCoreMailboxes` maps each `Memory*` to a vector of `Mail` slots,
    /// one per logical core. The receiver-side hash burst reads its slot;
    /// the sender-side `smashMail` writes into all recipients' slots in
    /// turn. With `logicalCores == 1` the vector has length 1 and the
    /// per-core dimension is degenerate; the typedef survives for code that
    /// would reactivate multi-core dispatch.
    ///
    /// @see [I-26](../../docs/30_invariants.md#i-26) — mail-out direction.
    /// @see [I-28](../../docs/30_invariants.md#i-28) — cross-LB writes
    ///      forbidden during the parallel phase; `PerCoreMailboxes` is
    ///      one of the per-thread structures that satisfies that.
    using ParentChildrenMap = std::unordered_map<Memory*, std::vector<Memory*>>;

    // One mailbox per logical core for each body
    using PerCoreMailboxes = std::unordered_map<Memory*, std::vector<Mail>>;

} // namespace gl
