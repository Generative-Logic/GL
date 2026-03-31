/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025 Generative Logic UG(haftungsbeschränkt)

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

    struct LocalMemoryValue {
        std::string value;
        std::set<int> levels;
        std::string originalImplication;
        std::string justification;
        std::vector<std::string> key;
        std::set<std::string> remainingArgs;
        std::string validityName;

        // Constructors
        LocalMemoryValue()
            : value(),
            levels(),
            originalImplication(),
            justification(),
            key(),
            remainingArgs(),
            validityName("main") {
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
            validityName(validityName_) {
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


    struct RejectedMapValue {
        // Fields (camelCase)
        std::string renamedExpression;
        std::string expression;
        int iteration;

        // Constructors
        RejectedMapValue()
            : renamedExpression(),
            expression(),
            iteration(0) {
        }

        RejectedMapValue(const std::string& renamedExpression_,
            const std::string& expression_,
            int iteration_)
            : renamedExpression(renamedExpression_),
            expression(expression_),
            iteration(iteration_) {
        }

        // Ordering so it can be used in std::set<RejectedMapValue>
        // Lexicographic over (renamedExpression, markedLeft, expression, iteration)
        bool operator<(const RejectedMapValue& rhs) const {
            if (renamedExpression != rhs.renamedExpression) {
                return renamedExpression < rhs.renamedExpression;
            }
            if (expression != rhs.expression) {
                return expression < rhs.expression;
            }
            return iteration < rhs.iteration;
        }

        // Optional equality (useful for comparisons; not required for std::set)
        bool operator==(const RejectedMapValue& rhs) const {
            return renamedExpression == rhs.renamedExpression
                && expression == rhs.expression
                && iteration == rhs.iteration;
        }
    };

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
    struct NameMap {
        std::unordered_map<std::string, int16_t> nameToId;
        std::vector<std::string> idToName;
        int16_t nextId = 1;  // 0 reserved for invalid

        NameMap() { idToName.push_back(""); } // slot 0 = invalid

        int16_t encode(const std::string& s) {
            auto it = nameToId.find(s);
            if (it != nameToId.end()) return it->second;
            int16_t id = nextId++;
            nameToId[s] = id;
            idToName.push_back(s);
            return id;
        }

        const std::string& decode(int16_t id) const {
            return idToName[static_cast<std::size_t>(id)];
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

    /// Bump allocator for persisting int16_t key data. All-or-nothing release.
    struct KeyArena {
        std::vector<int16_t*> blocks;
        int16_t* current = nullptr;
        int32_t used = 0;
        int32_t capacity = 0;

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

        void release() {
            for (auto* b : blocks) delete[] b;
            blocks.clear();
            current = nullptr;
            used = capacity = 0;
        }

        ~KeyArena() { release(); }
    };

    // ========================================================================
    // IntEncodedExpr — pre-encoded statement for static request pipeline.
    // All fields int16_t. No heap allocation. Fixed-size (176 bytes).
    // ========================================================================
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

    // ========================================================================
    // TypedArena — per-type bump allocator. Reset per LB, no deallocation.
    // ========================================================================
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

    // ========================================================================
    // MandatoryPair — index pair for static makeMandatory2.
    // ========================================================================
    struct MandatoryPair {
        int16_t idx1;
        int16_t idx2;
    };

    // ========================================================================
    // StaticRequest — fully static request: no string data, no heap allocs.
    // intExprs point into TypedArena (stable through vector reallocation).
    // ========================================================================
    struct StaticRequest {
        const IntEncodedExpr* intExprs[ExecutionParameters::MAX_EXPRESSIONS]; // ptrs into arena
        int16_t count;
        int16_t maxIteration;
        IntNormalizedKey normalizedKey;
    };

    // Pack (originalId, validityId) into int32_t for fast set/map lookup.
    inline int32_t packStatementKey(int16_t originalId, int16_t validityId) {
        return (static_cast<int32_t>(static_cast<uint16_t>(originalId)) << 16)
             | static_cast<int32_t>(static_cast<uint16_t>(validityId));
    }

    // ========================================================================
    // StaticRequestEmitter — dedup + emit for all generateEncodedRequests* variants.
    // ========================================================================
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

    // ========================================================================
    // BaseCandidate — grow-phase output for mandatory merge.
    // ========================================================================
    struct BaseCandidate {
        int16_t allIdx[ExecutionParameters::MAX_EXPRESSIONS];
        int16_t count;
        int16_t validityId;
    };

    // ========================================================================
    // ThreadArenas — per-thread typed arenas for static request pipeline.
    // ========================================================================
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

    /// Hash for std::set<int16_t> used as map key in remainingArgsNormalizedEncodedMap
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
              rejectedMap(), admissionStatusMap(), productsOfRecursion(), productsOfRecursionIds(),
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
            admissionStatusMap.clear();
            productsOfRecursion.clear();
            productsOfRecursionIds.clear();
            consumedAdmissionKeys.clear();
            revisitInProgress.clear();
        }
    };


    // ========================================================================
    // Debug dump: decode IntNormalizedKey to string vector matching
    // NormalizedKey.data format, for diff-based validation.
    // Requires arity lookup: exprName → number of arguments.
    // ========================================================================

    inline std::vector<std::string> decodeIntKey(
        const IntNormalizedKey& ik,
        const NameMap& nm,
        const std::map<std::string, LogicalEntity>& compiledExpressions) {

        std::vector<std::string> out;
        int16_t pos = 0;
        for (int16_t e = 0; e < ik.numberExpressions; ++e) {
            if (pos + 2 > ik.length) break;
            const std::string& name = nm.decode(ik.data[pos++]);
            out.push_back(name);
            out.push_back(ik.data[pos++] ? "True" : "False");

            int arity = 0;
            auto it = compiledExpressions.find(name);
            if (it != compiledExpressions.end()) {
                arity = it->second.arity;
            }

            for (int a = 0; a < arity; ++a) {
                if (pos + 2 > ik.length) break;
                int16_t varId = ik.data[pos++];
                int16_t changeStatus = ik.data[pos++];
                if (changeStatus == 1) {
                    out.push_back("u_" + nm.decode(varId));
                } else {
                    out.push_back(std::to_string(varId));
                }
            }
        }
        return out;
    }

    inline std::string decodedKeyToString(int16_t numExpr, const std::vector<std::string>& data) {
        std::string s = std::to_string(numExpr) + ":";
        for (std::size_t i = 0; i < data.size(); ++i) {
            if (i > 0) s += ",";
            s += data[i];
        }
        return s;
    }

    inline void dumpHashMemory(
        const HashMemory& mem,
        const NameMap& nm,
        const std::map<std::string, LogicalEntity>& compiledExpressions,
        int burstIndex,
        const std::string& prefix = ".dump/hash_int_") {

        std::string path = prefix + std::to_string(burstIndex) + ".txt";
        std::ofstream f(path);
        if (!f.is_open()) return;

        // encodedMap
        {
            std::vector<std::string> lines;
            for (const auto& kv : mem.encodedMap) {
                std::vector<std::string> decoded = decodeIntKey(kv.first, nm, compiledExpressions);
                std::string keyStr = decodedKeyToString(kv.first.numberExpressions, decoded);
                for (const auto& lmv : kv.second) {
                    lines.push_back("EM|" + keyStr + "|" + lmv.value + "|" +
                        lmv.justification + "|" + lmv.validityName + "|" +
                        lmv.originalImplication);
                }
            }
            std::sort(lines.begin(), lines.end());
            for (const auto& l : lines) f << l << "\n";
        }
        // normalizedEncodedKeys
        {
            std::vector<std::string> lines;
            for (const auto& ik : mem.normalizedEncodedKeys) {
                std::vector<std::string> decoded = decodeIntKey(ik, nm, compiledExpressions);
                lines.push_back("NK|" + decodedKeyToString(ik.numberExpressions, decoded));
            }
            std::sort(lines.begin(), lines.end());
            for (const auto& l : lines) f << l << "\n";
        }
        // normalizedEncodedSubkeys
        {
            std::vector<std::string> lines;
            for (const auto& ik : mem.normalizedEncodedSubkeys) {
                std::vector<std::string> decoded = decodeIntKey(ik, nm, compiledExpressions);
                lines.push_back("SK|" + decodedKeyToString(ik.numberExpressions, decoded));
            }
            std::sort(lines.begin(), lines.end());
            for (const auto& l : lines) f << l << "\n";
        }
        // normalizedEncodedSubkeysMinusOne
        {
            std::vector<std::string> lines;
            for (const auto& ik : mem.normalizedEncodedSubkeysMinusOne) {
                std::vector<std::string> decoded = decodeIntKey(ik, nm, compiledExpressions);
                lines.push_back("S1|" + decodedKeyToString(ik.numberExpressions, decoded));
            }
            std::sort(lines.begin(), lines.end());
            for (const auto& l : lines) f << l << "\n";
        }
        // normalizedEncodedSubkeysMinusTwo
        {
            std::vector<std::string> lines;
            for (const auto& ik : mem.normalizedEncodedSubkeysMinusTwo) {
                std::vector<std::string> decoded = decodeIntKey(ik, nm, compiledExpressions);
                lines.push_back("S2|" + decodedKeyToString(ik.numberExpressions, decoded));
            }
            std::sort(lines.begin(), lines.end());
            for (const auto& l : lines) f << l << "\n";
        }
        // remainingArgsNormalizedEncodedMap
        {
            std::vector<std::string> lines;
            for (const auto& kv : mem.remainingArgsNormalizedEncodedMap) {
                std::string argsStr;
                for (const auto& a : kv.first) {
                    if (!argsStr.empty()) argsStr += ",";
                    argsStr += nm.decode(a);
                }
                for (const auto& ik : kv.second) {
                    std::vector<std::string> decoded = decodeIntKey(ik, nm, compiledExpressions);
                    lines.push_back("RA|" + argsStr + "|" + decodedKeyToString(ik.numberExpressions, decoded));
                }
            }
            std::sort(lines.begin(), lines.end());
            for (const auto& l : lines) f << l << "\n";
        }
        f << "MAX_KEY_LEN|" << mem.maxKeyLength << "\n";
    }

    // ========================================================================

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

    /// Encode an EncodedExpression into IntEncodedExpr using NameMap.
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

    struct Mail {
        std::set< std::pair<std::string, std::set<int> > > statements;
        std::set< std::tuple<std::vector<std::string>, std::string, std::set<std::string>, std::set<int>, std::string> > implications;

        // CHANGED: Tracking multiple origins (Vector of Pairs)
        std::map<ExpressionWithValidity, std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>> exprOriginMap;

        Mail() : statements(), implications(), exprOriginMap() {}
    };

    

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






    struct Memory {
        std::map<std::string, Memory*> simpleMap;
        int startInt;
        int startIntRepl;
        int startIntPi;
        std::map<EncodedExpression, std::tuple<std::set<int>, std::set<std::string>> > toBeProved;
        std::vector<EncodedExpression> encodedStatements;
        std::map<EncodedExpression, std::set<int> > statementLevelsMap;
        std::string exprKey;
        Memory* parentMemory;
        int level;

        HashMemory overallHashMemory;
        HashMemory localHashMemory;
        HashMemory localHashMemoryDelta;

        std::map<std::string, std::vector<EquivalenceClass>> equivalenceClassesMap;
        std::vector<EncodedExpression> localEncodedStatements;
        std::vector<EncodedExpression> localEncodedStatementsDelta;
        std::map<std::string, int> integrationStartIntMap;

        Mail mailIn;
        Mail mailOut;

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
            nameMap(),
            keyArena(),
            intValidityNamesToFilter(),
            intAxedVariables()
        {
        }


    };




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

    struct ContradictionItem {
        std::string expr;
        bool successful;
        ContradictionItem()
            : expr(),
            successful(false){
        }
        ContradictionItem(const std::string& expr_,
            int successful_)
            : expr(expr_),
            successful(successful_) {
        }
	};

    
    using ParentChildrenMap = std::unordered_map<Memory*, std::vector<Memory*>>;

    // One mailbox per logical core for each body
    using PerCoreMailboxes = std::unordered_map<Memory*, std::vector<Mail>>;

} // namespace gl
