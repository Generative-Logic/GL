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
#include <string>
#include <vector>
#include <map>
#include <set>
#include <deque>
#include <chrono>
#include "create_expressions_shim.hpp"
#include <regex>
#include <cassert>
#include <tuple>
#include <cstdlib> 
#include <algorithm>
#include <functional>
#include <mutex> 
#include "parameters.hpp"

// This is a scaffolded C++ port of GL/analyze_expressions.py focused on "quick mode".
// The goal is readability and structure parity. Algorithmic internals are intentionally minimal
// so the project builds & runs immediately, producing quick_mode.txt.

namespace gl {



    struct LocalMemoryValue {
        std::string value;
        std::set<int> levels;
        std::string originalImplication;
        std::vector<std::string> key;
        std::set<std::string> remainingArgs;

        // Constructors
        LocalMemoryValue()
            : value(),
            levels(),
            originalImplication(),
            key(),
            remainingArgs() {
        }

        LocalMemoryValue(const std::string& value_,
            const std::set<int>& levels_,
            const std::string& originalImplication_,
            const std::vector<std::string>& key_,
            const std::set<std::string>& remainingArgs_)
            : value(value_),
            levels(levels_),
            originalImplication(originalImplication_),
            key(key_),
            remainingArgs(remainingArgs_) {
        }

        // Ordering so it can be stored in std::set<LocalMemoryValue>
        // Lexicographic over (value, levels, originalImplication, key, remainingArgs).
        bool operator<(const LocalMemoryValue& rhs) const {
            if (value != rhs.value) {
                return value < rhs.value;
            }
            if (levels != rhs.levels) {
                return levels < rhs.levels; // std::set<std::string> has lexicographic operator<
            }
            if (originalImplication != rhs.originalImplication) {
                return originalImplication < rhs.originalImplication;
            }
            if (key != rhs.key) {
                return key < rhs.key; // std::vector<std::string> has lexicographic operator<
            }
            if (remainingArgs != rhs.remainingArgs) {
                return remainingArgs < rhs.remainingArgs; // std::set<std::string> lexicographic
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

    struct LocalMemory {
        // Python: encoded_map: {NormalizedKey -> LocalMemoryValue}
        std::map<NormalizedKey, std::set<LocalMemoryValue>> encodedMap;

        // Python: remaining_args_normalized_encoded_map: {set<string> -> set<NormalizedKey>}
        std::map<std::set<std::string>, std::set<NormalizedKey> > remainingArgsNormalizedEncodedMap;

        // Python: max_key_length: int
        int maxKeyLength;

        // Python: normalized_encoded_keys: set<NormalizedKey>
        std::set<NormalizedKey> normalizedEncodedKeys;

        // Python: normalized_encoded_subkeys: set<NormalizedKey>
        std::set<NormalizedKey> normalizedEncodedSubkeys;

        
        std::set<NormalizedKey> normalizedEncodedSubkeysMinusOne;
        std::set<NormalizedKey> normalizedEncodedSubkeysMinusTwo;

        // Python: admission_map: {string -> set<AdmissionMapValue>}
        std::map<std::string, std::set<AdmissionMapValue> > admissionMap;

        // Python: rejected_map: {string -> RejectedMapValue}
        std::map<std::string, std::set<RejectedMapValue>> rejectedMap;

        // Python: admission_status_map: {string -> bool}
        std::map<std::string, bool> admissionStatusMap;

        // Python: products_of_recursion: set<string>
        std::set<std::string> productsOfRecursion;

        // Default constructor (mirrors Python __init__)
        LocalMemory()
            : encodedMap(),
            remainingArgsNormalizedEncodedMap(),
            maxKeyLength(0),
            normalizedEncodedKeys(),
            normalizedEncodedSubkeys(),
            normalizedEncodedSubkeysMinusOne(),
            normalizedEncodedSubkeysMinusTwo(),
            admissionMap(),
            rejectedMap(),
            admissionStatusMap(),
            productsOfRecursion() {
        }

        // Utility: reset to "empty" state (like recreating Python object)
        void clear() {
            encodedMap.clear();
            remainingArgsNormalizedEncodedMap.clear();
            maxKeyLength = 0;
            normalizedEncodedKeys.clear();
            normalizedEncodedSubkeys.clear();
            normalizedEncodedSubkeysMinusOne.clear();
            normalizedEncodedSubkeysMinusTwo.clear();
            admissionMap.clear();
            rejectedMap.clear();
            admissionStatusMap.clear();
            productsOfRecursion.clear();
        }
    };

    struct Mail {
        // (statement, levels)
        std::set< std::pair<std::string, std::set<int> > > statements;

        // (chain, head, remaining_args_key, levels, original_implication)
        std::set<
            std::tuple<
            std::vector<std::string>,  // chain
            std::string,               // head
            std::set<std::string>,     // remaining_args_key
            std::set<int>,             // levels
            std::string                // original implication / or_impl
            >
        > implications;

        // expr -> history (list of origins)
        std::map<std::string, std::vector<std::string> > exprOriginMap;

        Mail()
            : statements(),
            implications(),
            exprOriginMap() {
        }
    };


    struct EquivalenceClass {
        // Fields (camelCase)
        std::set<std::string> variables;                                              // set[str]
        std::map<std::set<std::string>, std::set<int> > equalityLevelsMap;           // set[str] -> set[int]
        std::map<std::string, std::vector<std::string> > equalityOriginMap;          // str -> list[str]

        
        EquivalenceClass()
            : variables(),
            equalityLevelsMap(),
            equalityOriginMap() {
        }

        // Full constructor
        EquivalenceClass(const std::set<std::string>& variables_,
            const std::map<std::set<std::string>, std::set<int> >& equalityLevelsMap_,
            const std::map<std::string, std::vector<std::string> >& equalityOriginMap_)
            : variables(variables_),
            equalityLevelsMap(equalityLevelsMap_),
            equalityOriginMap(equalityOriginMap_) {
        }
    };



    struct EncodedExpression {
        // Fields (CamelCase)
        std::string name;
        bool negation;
        std::vector<std::vector<std::string> > arguments; // each: [isUnchangeable, arg, iteration, levPlus1, argId]
        int maxIterationNumber;
        std::string original;

        // --- Constructors ---
        EncodedExpression()
            : name(),
            negation(false),
            arguments(),
            maxIterationNumber(-1),
            original() {
        }

        // Build from pre-parsed pieces (you provide name, negation, and raw args; we normalize args like Python)
        EncodedExpression(const std::string& name_,
            bool negation_,
            const std::vector<std::string>& rawArgs,
            const std::string& original_)
            : name(name_),
            negation(negation_),
            arguments(),
            maxIterationNumber(-1),
            original(original_) {
            for (std::size_t i = 0; i < rawArgs.size(); ++i) {
                arguments.push_back(parseArgument(rawArgs[i]));
            }
            maxIterationNumber = computeMaxIterationNumber(arguments);
        }

        explicit EncodedExpression(const std::string& original_)
            : name(),
            negation(false),
            arguments(),
            maxIterationNumber(-1),
            original(original_) {
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
            return original < rhs.original;
        }

        bool operator==(const EncodedExpression& rhs) const {
            return name == rhs.name
                && negation == rhs.negation
                && arguments == rhs.arguments
                && maxIterationNumber == rhs.maxIterationNumber
                && original == rhs.original;
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


    struct BodyOfProves {
        std::map<std::string, BodyOfProves*> simpleMap;
        int startInt;
        std::map<std::string, std::set<int> > toBeProved;
        std::vector<EncodedExpression> encodedStatements;
        std::map<std::string, std::set<int> > statementLevelsMap;
        std::string exprKey;
        BodyOfProves* parentBodyOfProves;
        int level;

        LocalMemory localMemory;

        std::vector<EquivalenceClass> equivalenceClasses;
        std::vector<EncodedExpression> localEncodedStatements;
        std::vector<EncodedExpression> localEncodedStatementsDelta;

        Mail mailIn;
        Mail mailOut;

        std::set<std::string> wholeExpressions;
        std::map<std::set<std::string>, int> eqClassSttmntIndexMap;
        bool isActive;
        bool isPartOfRecursion;
        int deltaNumberStatements;
        std::map<std::string, std::vector<std::string> > exprOriginMap;
        int recursionCounter;
        int contradictionIndex;

        BodyOfProves()
            : simpleMap(),
            startInt(0),
            toBeProved(),
            encodedStatements(),
            statementLevelsMap(),
            exprKey(),
            parentBodyOfProves(nullptr),
            level(-1),
            localMemory(),
            equivalenceClasses(),
            localEncodedStatements(),
            localEncodedStatementsDelta(),
            mailIn(),
            mailOut(),
            wholeExpressions(),
            eqClassSttmntIndexMap(),
            isActive(true),
            isPartOfRecursion(false),
            deltaNumberStatements(0),
            exprOriginMap(),
            recursionCounter(0),
            contradictionIndex(-1){
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

    
    using ParentChildrenMap = std::unordered_map<BodyOfProves*, std::vector<BodyOfProves*>>;

    // One mailbox per logical core for each body
    using PerCoreMailboxes = std::unordered_map<BodyOfProves*, std::vector<Mail>>;

    struct ExpressionAnalyzer {
        BodyOfProves body;
        // Root for counter-example filtering memory
        BodyOfProves ceBody;
        std::vector< std::tuple<int, bool, int>> updateGlobalTuples;
        std::vector< std::tuple<std::string, int>> updateGlobalDirectTuples;
        std::vector< std::tuple<std::string, std::string, std::string, std::string >> globalTheoremList;
        std::vector<BodyOfProves*> inductionMemoryBlocks;

        std::map<int, std::vector<std::vector<int> > > allBinariesAna;
        std::map<std::pair<int, int>, std::vector<std::vector<int> > > allMappingsAna;
        int maxNumLeafsPerKey;
        std::map<int, std::vector<std::vector<int> > > allPermutationsAna;
        Dependencies globalDependencies;
		std::vector<ContradictionItem> contradictionTable;

		std::map<std::string, ce::CoreExpressionConfig> coreExpressionMap; // Load and resolve core expressions  
        ce::AnchorInfo anchorInfo;

        std::vector<BodyOfProves*> permanentBodies;
        std::vector<BodyOfProves*> permanentBodiesCE;
        unsigned logicalCores;
        ParentChildrenMap index;
        PerCoreMailboxes boxes;
        ParentChildrenMap indexCE;
        PerCoreMailboxes boxesCE;

        mutable std::mutex dependenciesMutex;  // protects globalDependencies
        mutable std::mutex theoremListMutex;   // protects globalTheoremList
        mutable std::mutex inductionMemoryBlocksMutex;   // protects globalTheoremList
        mutable std::mutex updateGlobalMutex;
        mutable std::mutex updateGlobalDirectMutex;


        ExpressionAnalyzer();

        std::string smoothenExpr(const std::string& expr);
        std::vector<std::string> groomSubexpr(const std::string& s, int side, std::size_t& index);
        std::vector<std::string> groomExpr(const std::string& treeStr);
        std::vector<std::string> listLastRemovedArgs(const std::string& expr);
        std::tuple<std::string, int, std::string>
            renameLastRemoved(const std::string& expr, int startInt, int iteration, int level);
        std::string expandExpr(const std::string& expr);
        std::vector<std::string> getGlobalKey(const BodyOfProves& memoryBlock);
        void addToHashMemory(const std::vector<std::string>& key,
            const std::string& value,
            const std::set<std::string>& remainingArgs,
            LocalMemory& hashMemory,
            const std::set<int>& levels,
            const std::string& originalImplication,
            int maxAdmissionDepth,
            int maxSecondaryNumber,
            bool partOfRecursion);
        // ---- Ordering helpers (function pointers used by std::sort) ----
        static bool lessByName(const EncodedExpression& a, const EncodedExpression& b);
        static bool lessByOriginal(const EncodedExpression& a, const EncodedExpression& b);

        // ---- Tiny utilities ----
        std::vector<EncodedExpression> sortEncodedExpressionsByName(const std::vector<EncodedExpression>& in);
        std::vector<EncodedExpression> mergeInsertSortedEncoded(const std::vector<EncodedExpression>& listA,
            const std::vector<std::string>& valuesA,
            const std::vector<EncodedExpression>& listB,
            const std::vector<std::string>& valuesB);



        // ---- Mandatory lists makers ----
        std::vector< std::vector<EncodedExpression> >
            makeMandatoryEncodedStatementLists1(const LocalMemory& localMemory,
                const std::vector<EncodedExpression>& localStatements);

        std::vector< std::vector<EncodedExpression> >
            makeMandatoryEncodedStatementLists2(BodyOfProves& body,
                const LocalMemory& localMemory,
                const std::vector<EncodedExpression>& firstLayer,
                const std::vector<EncodedExpression>& secondLayer);

        // ---- Requests generator ----
        std::vector< std::tuple< std::vector<EncodedExpression>, bool, NormalizedKey > >
            generateEncodedRequests(BodyOfProves& body,
                const LocalMemory& localMemory,
                const std::vector< std::vector<EncodedExpression> >& mandatoryLists,
                const std::vector<EncodedExpression>& allStatements);



        // New kernels
        BodyOfProves& performElementaryLogicalStep(BodyOfProves& body,
            unsigned coreId,
            const ParentChildrenMap& index,
            PerCoreMailboxes& boxes);

        // Declaration only (no body). Matches the .cpp definition.
        // 0 - Local statement.
        // 1 - Proved statement.
        // 2 - Statement pending proof.
        // 3 - Non - local statement.
        void addExprToMemoryBlock(const std::string& expr,
            BodyOfProves& memoryBlock,
            int iteration,
            int status,
            const std::set<int>& involvedLevels,
            const std::vector<std::string>& origin,
            int coreId,
            int auxyIndex);





        std::string replaceUSubstrings(const std::string& s);
        void checkLocalEncodedMemory(const std::vector<EncodedExpression>& requestKey,
            BodyOfProves& body,
            int nextIteration,
            const NormalizedKey& normalizedReq,
            unsigned coreId);

        // … earlier declarations …

        void activateZeroCondition(BodyOfProves& memoryBlock);

        // Parallel kernel
        void proveKernel(const std::vector<BodyOfProves*>& bodies,
            const ParentChildrenMap& index,
            PerCoreMailboxes& boxes);

        // Keep ONLY this prove()
        void prove(int numberIterations,
            const std::vector<BodyOfProves*>& bodies,
            const ParentChildrenMap& index,
            PerCoreMailboxes& boxes);

        // … no other prove() overloads …


        std::set<std::string>
            findImmutableArgs(const std::string& theorem,
                const std::string& digit);

        BodyOfProves* accessBodyOfProves(const std::vector<std::string>& theoremKey,
            BodyOfProves& bodyOfProves1);
        void updateGlobalDirect(const std::string& theorem, int coreId);

        void updateGlobal(int auxyIndex, bool allLevelsInvolved, int coreId);

        // Deactivate all aux blocks created for a proven theorem
        void deactivateUnnecessary(const std::string& theorem, BodyOfProves& root);



        inline void updateAdmissionMap3(const std::string& expr,
            BodyOfProves& memoryBlock,
            int maxAdmissionDepth,
            int maxSecondaryNumber,
            bool partOfRecursion);

        std::tuple<
            std::string,
            std::vector<std::string>,
            std::string,
            std::set<std::string>,
            std::string,
            std::string
        >
            createAuxyImplication(const std::string& expr,
                const std::string& arg,
                const std::string& recArg,
                const std::set<std::string>& digitArgs,
                const std::string& anchorName);
                

        std::vector<std::string> extractSubstringsForAuxy(const std::string& expr2);

        void addTheoremToMemory(const std::string& expr,
            BodyOfProves& memory,
            int iteration,
            bool proved,
            Dependencies& dependencyTable);

        inline void revisitRejected(const std::string& markedExpr,
            BodyOfProves& memoryBlock);

        void analyzeExpressions(const std::vector<std::string>& theorems);

        void buildStack(BodyOfProves& memoryBlock,
                const std::string& proved,
                std::vector<std::vector<std::string>>& stack,
                std::set<std::string>& covered);

        // Sort strings by corresponding values (descending). Throws std::invalid_argument if sizes differ.
        std::vector<std::string> sortByValuesDesc(const std::vector<std::string>& strings,
            const std::vector<int>& values);

        // Debug: find ends, write indexed stacks + mapping file (like Python find_ends)
        void findEnds(const std::vector<std::string>& path);

        // Build MPL tables for 0..n
    // Returns {multiplications, additions, successions, inequalities}
    // If print is true, prints entries that satisfy print_if (or all if print_if==nullptr).
        std::tuple<
            std::vector<std::string>,
            std::vector<std::string>,
            std::vector<std::string>,
            std::vector<std::string>
        > makeMPLTables(int n,
            bool print = false,
            std::function<bool(const std::string&)> print_if = nullptr);


        void generateRawProofGraph(
            const std::vector<std::tuple<std::string, std::string, std::string, std::string>>& theoremList,
            const std::filesystem::path& outDir = std::filesystem::path("files") / "raw_proof_graph");



        // Build CE blocks for one conjecture: two consecutive LBs under ceBody.
        // LB0 gets +,*,s. LB1 gets inequalities and stores the conjecture via addToHashMemory.
        void addConjectureForCEFiltering(const std::string& conjecture,
            const std::tuple<
            std::vector<std::string>,
            std::vector<std::string>,
            std::vector<std::string>,
            std::vector<std::string>
            >& tables,
            int cIndex);

        // Batch teardown (deletes all CE nodes + shrinks containers + trims heap)
        void releaseCEBatchMemory();

        // Helpers to free/shrink heavy containers
        void destroyParentChildrenMap(ParentChildrenMap& m);
        void destroyMailboxes(PerCoreMailboxes& boxes);





        // Like analyzeExpressions, but for CE filtering
        std::vector<std::string> filterConjecturesWithCE(const std::vector<std::string>& conjectures);


        std::string negate(std::string expr);



        // Build parent->children index for the given roots (walks their reachable subgraph).
        ParentChildrenMap buildParentChildrenMap(const std::vector<BodyOfProves*>& roots) const;

        // Allocate per-core mailboxes sized to 'logicalCores'.
        PerCoreMailboxes buildPerCoreMailboxes(const ParentChildrenMap& index) const;

        // Compress per-core mailboxes into each body's mailIn in parallel.
        void smashMail(PerCoreMailboxes& boxes) const;
        


        

        


        inline std::pair<bool, NormalizedKey>
            preEvaluateEncodedKey(const std::vector<EncodedExpression>& key,
                BodyOfProves& body, const std::set<NormalizedKey>& keySet) {
            // Quick reject: too many secondary variables → no need to normalize at all
            int secondaryCounter = 0;
            for (std::size_t i = 0; i < key.size(); ++i) {
                secondaryCounter += countPatternOccurrencesEncoded(key[i], body.localMemory);
            }
            if (secondaryCounter > parameters::maxNumberSecondaryVariables) {
                return std::make_pair(false, NormalizedKey());
            }

            // Quick reject by length vs. admitted max
            if (static_cast<int>(key.size()) > body.localMemory.maxKeyLength) {
                return std::make_pair(false, NormalizedKey());
            }

            // Build NormalizedKey WITHOUT constructing the (expensive) mapping
            NormalizedKey normalized = makeNormalizedEncodedKeyOnly(key, /*ignoreU=*/false);

            // Check if this normalized subkey is admissible
            const bool ok =
                (keySet.find(normalized) !=
                    keySet.end());

            return std::make_pair(ok, normalized);
        }

        


        inline std::set<std::string> getAllEncodedArgs(const std::vector<EncodedExpression>& expressionList) {
            std::set<std::string> allArgs;
            for (std::size_t i = 0; i < expressionList.size(); ++i) {
                const std::vector<std::vector<std::string> >& args = expressionList[i].arguments;
                for (std::size_t j = 0; j < args.size(); ++j) {
                    if (args[j].size() >= 2) {
                        allArgs.insert(args[j][1]);
                    }
                }
            }
            return allArgs;
        }

        inline void setUnchangeables(std::vector<EncodedExpression>& expressionList,
            const std::set<std::string>& unchangeables) {
            for (std::size_t i = 0; i < expressionList.size(); ++i) {
                std::vector<std::vector<std::string> >& args = expressionList[i].arguments;
                for (std::size_t j = 0; j < args.size(); ++j) {
                    std::vector<std::string>& argument = args[j];
                    if (argument.size() >= 2) {
                        if (unchangeables.find(argument[1]) != unchangeables.end()) {
                            argument[0] = "True";
                        }
                    }
                }
            }
        }

        inline std::set<std::string> extractUnchangeables(const std::vector<EncodedExpression>& expressionList) {
            std::set<std::string> unchangeables;
            for (std::size_t i = 0; i < expressionList.size(); ++i) {
                const std::vector<std::vector<std::string> >& args = expressionList[i].arguments;
                for (std::size_t j = 0; j < args.size(); ++j) {
                    const std::vector<std::string>& argument = args[j];
                    if (argument.size() >= 2 && argument[0] == "True") {
                        unchangeables.insert(argument[1]);
                    }
                }
            }
            return unchangeables;
        }

        inline void resetUnchangeables(std::vector<EncodedExpression>& expressionList,
            const std::set<std::string>& unchangeables) {
            for (std::size_t i = 0; i < expressionList.size(); ++i) {
                std::vector<std::vector<std::string> >& args = expressionList[i].arguments;
                for (std::size_t j = 0; j < args.size(); ++j) {
                    std::vector<std::string>& argument = args[j];
                    if (argument.size() < 2) {
                        continue;
                    }
                    argument[0] = "False";
                    if (unchangeables.find(argument[1]) != unchangeables.end()) {
                        argument[0] = "True";
                    }
                }
            }
        }

        // NEW: build only the NormalizedKey (no mapping). Used by preEvaluateEncodedKey fast path.
        inline NormalizedKey makeNormalizedEncodedKeyOnly(const std::vector<EncodedExpression>& lst,
            bool ignoreU = true) {
            NormalizedKey ky;
            ky.numberExpressions = static_cast<int>(lst.size());

            // Pre-size ky.data: for each expr we push name + neg flag (2) + one entry per arg (>=2 items)
            std::size_t total = 0;
            for (std::size_t i = 0; i < lst.size(); ++i) {
                total += 2;
                const std::vector<std::vector<std::string>>& args = lst[i].arguments;
                for (std::size_t j = 0; j < args.size(); ++j) if (args[j].size() >= 2) ++total;
            }
            ky.data.reserve(total);

            // Map original variable -> numeric id (int); faster than map<string,string>
            std::unordered_map<std::string, int> idByVar;
            idByVar.reserve(total);
            int nextId = 0;

            for (std::size_t i = 0; i < lst.size(); ++i) {
                const EncodedExpression& expr = lst[i];

                ky.data.emplace_back(expr.name);
                ky.data.emplace_back(expr.negation ? "True" : "False");

                const std::vector<std::vector<std::string>>& args = expr.arguments;
                for (std::size_t j = 0; j < args.size(); ++j) {
                    const std::vector<std::string>& arg = args[j];
                    if (arg.size() < 2) continue;

                    const std::string& original = arg[1];
                    if (ignoreU && arg[0] == "True") {
                        // "u_" + original, built with one allocation
                        std::string u; u.reserve(2 + original.size());
                        u.append("u_").append(original);
                        ky.data.emplace_back(std::move(u));
                    }
                    else {
                        int id;
                        std::unordered_map<std::string, int>::iterator it = idByVar.find(original);
                        if (it == idByVar.end()) {
                            id = ++nextId;
                            idByVar.insert(std::make_pair(original, id));
                        }
                        else {
                            id = it->second;
                        }
                        ky.data.emplace_back(std::to_string(id));
                    }
                }
            }
            return ky;
        }

        // REPLACE the old inline with this allocation-aware version.
        // Same signature/return type; safe for all current call sites.
        inline std::pair<NormalizedKey, std::map<std::string, std::string> >
            makeNormalizedEncodedKey(const std::vector<EncodedExpression>& lst, bool ignoreU = true) {
            NormalizedKey ky;
            ky.numberExpressions = static_cast<int>(lst.size());

            // Pre-size as above
            std::size_t total = 0;
            for (std::size_t i = 0; i < lst.size(); ++i) {
                total += 2;
                const std::vector<std::vector<std::string>>& args = lst[i].arguments;
                for (std::size_t j = 0; j < args.size(); ++j) if (args[j].size() >= 2) ++total;
            }
            ky.data.reserve(total);

            // Use an O(1) map for id assignment; only mirror to std::map when a new id appears.
            std::unordered_map<std::string, int> idByVar;
            idByVar.reserve(total);
            std::map<std::string, std::string> coveredVariables; // returned to callers that need replacements
            int nextId = 0;

            for (std::size_t i = 0; i < lst.size(); ++i) {
                const EncodedExpression& expr = lst[i];

                ky.data.emplace_back(expr.name);
                ky.data.emplace_back(expr.negation ? "True" : "False");

                const std::vector<std::vector<std::string>>& args = expr.arguments;
                for (std::size_t j = 0; j < args.size(); ++j) {
                    const std::vector<std::string>& arg = args[j];
                    if (arg.size() < 2) continue;

                    const std::string& original = arg[1];
                    if (ignoreU && arg[0] == "True") {
                        std::string u; u.reserve(2 + original.size());
                        u.append("u_").append(original);
                        ky.data.emplace_back(std::move(u));
                    }
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

        

        // named functor comparator: compares numeric strings like Python's key=int
        struct NumericLess {
            bool operator()(const std::string& a, const std::string& b) const {
                return std::atoi(a.c_str()) < std::atoi(b.c_str());
            }
        };

        // requires ce::getArgs(const std::string&)
        inline std::string reconstructImplication(const std::vector<std::string>& key,
            const std::string& value) {
            std::string implication = value;

            std::vector<std::string> chain = key;
            chain.push_back(value);

            std::vector<std::vector<std::string> > argsChain;
            argsChain.reserve(chain.size());
            for (std::size_t i = 0; i < chain.size(); ++i) {
                argsChain.push_back(ce::getArgs(chain[i]));
            }

            std::set<std::string> removedArgs;
            std::map<std::string, int> counterMap;

            for (std::size_t i = 0; i < argsChain.size(); ++i) {
                const std::vector<std::string>& args = argsChain[i];
                for (std::size_t j = 0; j < args.size(); ++j) {
                    const std::string& arg = args[j];
                    if (arg.size() >= 2 && arg[0] == 'u' && arg[1] == '_') continue;
                    std::map<std::string, int>::iterator it = counterMap.find(arg);
                    if (it == counterMap.end()) {
                        counterMap.insert(std::make_pair(arg, 1));
                    }
                    else {
                        it->second += 1;
                        if (it->second > 1) removedArgs.insert(arg);
                    }
                }
            }

            std::vector<std::vector<std::string> > whenRemoved(key.size());
            for (std::set<std::string>::const_iterator it = removedArgs.begin(); it != removedArgs.end(); ++it) {
                const std::string& arg = *it;
                for (std::size_t idx = 0; idx < argsChain.size(); ++idx) {
                    const std::vector<std::string>& args = argsChain[idx];
                    if (std::find(args.begin(), args.end(), arg) != args.end()) {
                        if (idx < key.size()) whenRemoved[idx].push_back(arg);
                        break;
                    }
                }
            }

            for (std::size_t i = key.size(); i-- > 0; ) {
                std::vector<std::string>& lst = whenRemoved[i];
                std::sort(lst.begin(), lst.end(), NumericLess());

                std::string joined;
                for (std::size_t j = 0; j < lst.size(); ++j) {
                    if (j > 0) joined.push_back(',');
                    joined += lst[j];
                }

                std::string tmp;
                tmp.reserve(5 + joined.size() + key[i].size() + implication.size());
                tmp += "(>[";
                tmp += joined;
                tmp += "]";
                tmp += key[i];
                tmp += implication;
                tmp += ")";
                implication.swap(tmp);
            }

            return implication;
        }


        inline void updateAdmissionMap(LocalMemory& hashMemory,
            const std::vector<std::string>& key,
            const std::set<std::string>& remainingArgs,
            int maxAdmissionDepth,
            int maxSecondaryNumber,
            bool partOfRecursion) {
            std::map<std::string, std::string> replacementMap;
            for (std::set<std::string>::const_iterator it = remainingArgs.begin(); it != remainingArgs.end(); ++it) {
                replacementMap.insert(std::make_pair(std::string("u_") + *it, *it));
            }

            std::vector<std::string> replacedKey;
            replacedKey.reserve(key.size());
            for (std::size_t i = 0; i < key.size(); ++i) {
                replacedKey.push_back(ce::replaceKeysInString(key[i], replacementMap));
            }

            for (std::size_t i = 0; i < replacedKey.size(); ++i) {
                const std::string& element = replacedKey[i];
                const std::string coreExpr = ce::extractExpression(element);
                if (ce::operators.find(coreExpr) == ce::operators.end()){
                    continue;
                }

                const std::vector<std::string> argsVec = ce::getArgs(element);
                std::set<std::string> args(argsVec.begin(), argsVec.end());

                std::set<std::string> withoutRemaining;
                for (std::set<std::string>::const_iterator it = args.begin(); it != args.end(); ++it) {
                    if (remainingArgs.find(*it) == remainingArgs.end()) {
                        withoutRemaining.insert(*it);
                    }
                }

                if (withoutRemaining.size() == 1) {
                    const std::string onlyArg = *withoutRemaining.begin();
                    std::map<std::string, std::string> removingMap;
                    removingMap.insert(std::make_pair(onlyArg, "marker"));

                    const std::string removed = ce::replaceKeysInString(element, removingMap);

                    AdmissionMapValue value(replacedKey,
                        remainingArgs,
                        maxAdmissionDepth,
                        maxSecondaryNumber,
                        partOfRecursion);

                    std::map<std::string, std::set<AdmissionMapValue> >::iterator itAd = hashMemory.admissionMap.find(removed);
                    if (itAd != hashMemory.admissionMap.end()) {
                        itAd->second.insert(value);

                        std::map<std::string, bool>::iterator itSt = hashMemory.admissionStatusMap.find(removed);
                        if (itSt != hashMemory.admissionStatusMap.end()) {
                            itSt->second = partOfRecursion || itSt->second;
                        }
                        else {
                            hashMemory.admissionStatusMap[removed] = partOfRecursion;
                        }
                    }
                    else {
                        std::set<AdmissionMapValue> s;
                        s.insert(value);
                        hashMemory.admissionMap.insert(std::make_pair(removed, s));
                        hashMemory.admissionStatusMap[removed] = partOfRecursion;
                    }
                }
            }
        }

        inline bool startsWithStr(const std::string& s, const std::string& pre) {
            if (s.size() < pre.size()) return false;
            for (std::size_t i = 0; i < pre.size(); ++i) if (s[i] != pre[i]) return false;
            return true;
        }
        inline bool endsWithStr(const std::string& s, const std::string& suf) {
            if (s.size() < suf.size()) return false;
            const std::size_t off = s.size() - suf.size();
            for (std::size_t i = 0; i < suf.size(); ++i) if (s[off + i] != suf[i]) return false;
            return true;
        }

        inline void updateAdmissionMap2(LocalMemory& hashMemory,
            const std::string& markedExpr,
            const std::string& var) {
            const std::string marker = "marker";
            const std::size_t pos = markedExpr.find(marker);
            assert(pos != std::string::npos);
            assert(markedExpr.find(marker, pos + marker.size()) == std::string::npos);

            const std::string part0 = markedExpr.substr(0, pos);
            const std::string part1 = markedExpr.substr(pos + marker.size());

            std::map<std::string, std::set<AdmissionMapValue> >::iterator it =
                hashMemory.admissionMap.find(markedExpr);
            if (it == hashMemory.admissionMap.end()) return;

            // snapshot
            std::vector<AdmissionMapValue> originalKeys;
            originalKeys.reserve(it->second.size());
            for (std::set<AdmissionMapValue>::const_iterator sit = it->second.begin();
                sit != it->second.end(); ++sit) {
                originalKeys.push_back(*sit);
            }

            for (std::size_t k = 0; k < originalKeys.size(); ++k) {
                std::vector<std::string> key = originalKeys[k].key;
                const std::set<std::string> remainingArgs = originalKeys[k].remainingArgs;
                const int maxAdmissionDepth = originalKeys[k].standardMaxAdmissionDepth;
                const int maxSecondaryNumber = originalKeys[k].standardMaxSecondaryNumber;
                const bool partOfRecursion = originalKeys[k].flag;

                for (std::size_t e = 0; e < key.size(); ++e) {
                    const std::string& element = key[e];
                    if (startsWithStr(element, part0) && endsWithStr(element, part1)) {
                        const std::size_t coreLen = element.size() - part0.size() - part1.size();
                        const std::string varForRepl = element.substr(part0.size(), coreLen);

                        std::map<std::string, std::string> replacementMap;
                        replacementMap.insert(std::make_pair(varForRepl, var));

                        std::vector<std::string> replKey;
                        replKey.reserve(key.size());
                        for (std::size_t i = 0; i < key.size(); ++i) {
                            replKey.push_back(ce::replaceKeysInString(key[i], replacementMap));
                        }

                        std::set<std::string> newRemArgs = remainingArgs;
                        newRemArgs.insert(var);

                        updateAdmissionMap(hashMemory,
                            replKey,
                            newRemArgs,
                            maxAdmissionDepth,
                            maxSecondaryNumber,
                            partOfRecursion);
                    }
                }
            }
        }

        inline std::set<std::string> getRemainingArgs(const std::vector<std::string>& key) {
            std::set<std::string> remainingArgs;
            for (std::size_t i = 0; i < key.size(); ++i) {
                const std::vector<std::string> args = ce::getArgs(key[i]);
                for (std::size_t j = 0; j < args.size(); ++j) {
                    const std::string& arg = args[j];
                    if (arg.size() >= 2 && arg[0] == 'u' && arg[1] == '_') {
                        remainingArgs.insert(arg);
                    }
                }
            }
            return remainingArgs;
        }

        inline bool implicationIsQualified(const std::vector<std::string>& key, const std::string& value) {
            int counterKey = 0;
            bool result = false;

            for (std::size_t i = 0; i < key.size(); ++i) {
                const std::string& element = key[i];
                if (element.size() >= 4 && element[0] == '(' && element[1] == 'i' && element[2] == 'n' && element[3] == '3') {
                    counterKey += 1;
                }
            }

            bool valueCond = (value.size() >= 5 &&
                value[0] == '(' && value[1] == 'i' && value[2] == 'n' && value[3] == '3' && value[4] == '[');

            if (counterKey > parameters::minNumOperatorsKey && valueCond) {
                result = true;
            }

            return result;
        }



        inline std::set<VariantItem> createVariants(const std::vector<std::string>& chain,
            const std::string& head) {
            std::set<VariantItem> variants;

            const std::size_t n = chain.size();
            std::map<int, std::vector<std::vector<int> > >::const_iterator pit =
                this->allPermutationsAna.find(static_cast<int>(n));
            if (pit == this->allPermutationsAna.end()) {
                return variants;
            }
            const std::vector<std::vector<int> >& permuts = pit->second;

            std::vector<std::string> ids;
            ids.reserve(n);
            for (std::size_t i = 0; i < n; ++i) {
                ids.push_back(ce::extractExpression(chain[i]));
            }

            for (std::size_t p = 0; p < permuts.size(); ++p) {
                const std::vector<int>& permutation = permuts[p];

                std::vector<std::string> tempList;
                tempList.reserve(n);
                for (std::size_t k = 0; k < permutation.size(); ++k) {
                    const int idx = permutation[k];
                    tempList.push_back(chain[static_cast<std::size_t>(idx)]);
                }

                bool toContinue = false;
                if (n >= 2) {
                    for (std::size_t k = 0; k + 1 < n; ++k) {
                        const int a = permutation[k];
                        const int b = permutation[k + 1];
                        if (ids[static_cast<std::size_t>(a)] > ids[static_cast<std::size_t>(b)]) {
                            toContinue = true;
                            break;
                        }
                    }
                }
                if (toContinue) {
                    continue;
                }

                std::vector<EncodedExpression> tempEncodedList;
                tempEncodedList.reserve(tempList.size());
                for (std::size_t t = 0; t < tempList.size(); ++t) {
                    tempEncodedList.push_back(EncodedExpression(tempList[t]));
                }

                std::pair<NormalizedKey, std::map<std::string, std::string> > pairIgnored =
                    makeNormalizedEncodedKey(tempEncodedList, true);
                std::pair<NormalizedKey, std::map<std::string, std::string> > pairNotIgnored =
                    makeNormalizedEncodedKey(tempEncodedList, false);

                const NormalizedKey& normalizedIgnoredKeyVariant = pairIgnored.first;
                const NormalizedKey& normalizedNotIgnoredKeyVariant = pairNotIgnored.first;

                const std::map<std::string, std::string>& mp2 = pairIgnored.second;

                std::string valueVariant = ce::replaceKeysInString(head, mp2);

                std::set< std::pair<std::string, std::string> > mappingItems;
                for (std::map<std::string, std::string>::const_iterator it = mp2.begin(); it != mp2.end(); ++it) {
                    mappingItems.insert(std::make_pair(it->first, it->second));
                }

                VariantItem item;
                item.normalizedIgnoredKey = normalizedIgnoredKeyVariant;
                item.normalizedNotIgnoredKey = normalizedNotIgnoredKeyVariant;
                item.valueVariant = valueVariant;
                item.mappingItems = mappingItems;

                variants.insert(item);
            }

            return variants;
        }

        inline void makeNormalizedSubkeys(const std::vector<std::string>& key,
            LocalMemory& hashMemory) {
            const std::size_t n = key.size();

            std::map<int, std::vector<std::vector<int> > >::const_iterator pit =
                this->allPermutationsAna.find(static_cast<int>(n));
            if (pit == this->allPermutationsAna.end()) {
                return;
            }
            const std::vector<std::vector<int> >& permuts = pit->second;

            std::vector<std::string> ids;
            ids.reserve(n);
            for (std::size_t i = 0; i < n; ++i) {
                ids.push_back(ce::extractExpression(key[i]));
            }

            for (std::size_t p = 0; p < permuts.size(); ++p) {
                const std::vector<int>& permut = permuts[p];

                std::vector<std::string> tempList;
                tempList.reserve(n);
                for (std::size_t k = 0; k < permut.size(); ++k) {
                    const int idx = permut[k];
                    tempList.push_back(key[static_cast<std::size_t>(idx)]);
                }

                for (std::size_t index = 0; index < n; ++index) {
                    bool toBreak = false;
                    for (std::size_t index2 = 0; index2 < index; ++index2) {
                        const int a = permut[index2];
                        const int b = permut[index2 + 1];
                        if (ids[static_cast<std::size_t>(a)] > ids[static_cast<std::size_t>(b)]) {
                            toBreak = true;
                            break;
                        }
                    }
                    if (toBreak) {
                        break;
                    }

                    std::vector<EncodedExpression> subkeyEncodedList;
                    subkeyEncodedList.reserve(index + 1);
                    for (std::size_t t = 0; t <= index; ++t) {
                        subkeyEncodedList.push_back(EncodedExpression(tempList[t]));
                    }

                    std::pair<NormalizedKey, std::map<std::string, std::string> > res =
                        makeNormalizedEncodedKey(subkeyEncodedList, false);
                    hashMemory.normalizedEncodedSubkeys.insert(res.first);

                    if (subkeyEncodedList.size() == (n - 1))
                    {
                        hashMemory.normalizedEncodedSubkeysMinusOne.insert(res.first);
                    }

                    if (subkeyEncodedList.size() == (n - 2))
                    {
                        hashMemory.normalizedEncodedSubkeysMinusTwo.insert(res.first);
                    }
                }
            }
        }

        inline void makeNormalizedKeysForAdmission(const std::vector<std::string>& key,
            LocalMemory& hashMemory,
            const std::string& value) {
            if (!implicationIsQualified(key, value)) {
                return;
            }

            const std::size_t n = key.size();

            for (std::size_t index = 0; index < n; ++index) {
                const std::string coreExpr = ce::extractExpression(key[index]);


                std::vector<int> binary(n, 1);
                binary[index] = 0;

                std::vector<std::string> subkey;
                subkey.reserve(n ? n - 1 : 0);
                for (std::size_t i = 0; i < n; ++i) {
                    if (binary[i]) {
                        subkey.push_back(key[i]);
                    }
                }

                std::set<std::string> subkeyArgs;
                for (std::size_t i = 0; i < subkey.size(); ++i) {
                    const std::vector<std::string> a = ce::getArgs(subkey[i]);
                    for (std::size_t j = 0; j < a.size(); ++j) {
                        subkeyArgs.insert(a[j]);
                    }
                }

                const std::vector<std::string> args = ce::getArgs(key[index]);
                if (args.size() < 2) {
                    continue;
                }
                const std::string outputArg = args[args.size() - 2];

                if (subkeyArgs.find(outputArg) == subkeyArgs.end()) {
                    std::map<std::string, std::string> replacementMap;
                    replacementMap.insert(std::make_pair(outputArg, "marker"));
                    const std::string replaced = ce::replaceKeysInString(key[index], replacementMap);

                    this->makeNormalizedSubkeys(subkey, hashMemory);

                    const std::set<VariantItem> variants = this->createVariants(subkey, replaced);

                    std::set<std::string> remainingArgs = getRemainingArgs(subkey);

                    for (std::set<VariantItem>::const_iterator vit = variants.begin(); vit != variants.end(); ++vit) {
                        const VariantItem& v = *vit;

                        std::map<std::string, std::string> mpMap;
                        for (std::set< std::pair<std::string, std::string> >::const_iterator it = v.mappingItems.begin();
                            it != v.mappingItems.end(); ++it) {
                            mpMap.insert(*it);
                        }

                        std::vector<std::string> replKey;
                        replKey.reserve(key.size());
                        for (std::size_t i = 0; i < key.size(); ++i) {
                            replKey.push_back(ce::replaceKeysInString(key[i], mpMap));
                        }

                        LocalMemoryValue localMemoryValue;
                        localMemoryValue.value = v.valueVariant;
                        localMemoryValue.key = replKey;
                        localMemoryValue.remainingArgs = remainingArgs;

                        std::map<NormalizedKey, std::set<LocalMemoryValue> >::iterator emIt =
                            hashMemory.encodedMap.find(v.normalizedIgnoredKey);
                        if (emIt != hashMemory.encodedMap.end()) {
                            emIt->second.insert(localMemoryValue);
                        }
                        else {
                            std::set<LocalMemoryValue> bucket;
                            bucket.insert(localMemoryValue);
                            hashMemory.encodedMap.insert(std::make_pair(v.normalizedIgnoredKey, bucket));
                        }

                        hashMemory.normalizedEncodedKeys.insert(v.normalizedNotIgnoredKey);

                        std::map<std::set<std::string>, std::set<NormalizedKey> >::iterator ramIt =
                            hashMemory.remainingArgsNormalizedEncodedMap.find(remainingArgs);
                        if (ramIt != hashMemory.remainingArgsNormalizedEncodedMap.end()) {
                            ramIt->second.insert(v.normalizedNotIgnoredKey);
                        }
                        else {
                            std::set<NormalizedKey> s;
                            s.insert(v.normalizedNotIgnoredKey);
                            hashMemory.remainingArgsNormalizedEncodedMap.insert(std::make_pair(remainingArgs, s));
                        }
                    }
                }
            }
        }

        inline int countPatternOccurrences(const std::string& inputString,
            const LocalMemory& localMemory) {
            static const std::regex patternCounting(R"(it_\d+_lev_\d+_\d+)");
            int counter = 0;

            std::sregex_iterator it(inputString.begin(), inputString.end(), patternCounting);
            std::sregex_iterator end;
            for (; it != end; ++it) {
                const std::string match = it->str();
                if (localMemory.productsOfRecursion.find(match) == localMemory.productsOfRecursion.end()) {
                    ++counter;
                }
            }
            return counter;
        }

        inline int countPatternOccurrencesEncoded(const EncodedExpression& encodedExpression,
            const LocalMemory& localMemory) {
            int counter = 0;
            const std::vector<std::vector<std::string> >& args = encodedExpression.arguments;

            for (std::size_t i = 0; i < args.size(); ++i) {
                const std::vector<std::string>& argument = args[i];
                if (argument.size() >= 3) {
                    const int it = std::atoi(argument[2].c_str());
                    if (it > -1) {
                        if (localMemory.productsOfRecursion.find(argument[1]) == localMemory.productsOfRecursion.end()) {
                            ++counter;
                        }
                    }
                }
            }
            return counter;
        }

        inline std::vector<EncodedExpression>
            filterEncodedStatements(const std::vector<EncodedExpression>& encodedStatements,
                const LocalMemory& localMemory) {
            std::vector<EncodedExpression> filtered;

            for (std::size_t i = 0; i < encodedStatements.size(); ++i) {
                const EncodedExpression& sttment = encodedStatements[i];

                std::vector<EncodedExpression> single;
                single.push_back(sttment);
                std::pair<NormalizedKey, std::map<std::string, std::string> > enc =
                    makeNormalizedEncodedKey(single, false);

                if (localMemory.normalizedEncodedSubkeys.find(enc.first) ==
                    localMemory.normalizedEncodedSubkeys.end()) {
                    continue;
                }

                const int filterStatementsItr = sttment.maxIterationNumber;
                if (filterStatementsItr > parameters::maxIterationNumberVariable) {
                    continue;
                }

                filtered.push_back(sttment);
            }

            return filtered;
        }



        inline std::pair<std::string, std::string>
            extractValues(const std::string& s,
                const std::map<std::string, ce::CoreExpressionConfig>& coreExpressionMap) {
            std::pair<ce::TreeNode1*, int> pr = ce::parseExpr(s, coreExpressionMap);
            ce::TreeNode1* root = pr.first;

            std::string left = ce::treeToExpr(root ? root->left : NULL);
            std::string right = ce::treeToExpr(root ? root->right : NULL);

            ce::deleteTree(root);
            return std::make_pair(left, right);
        }

        inline void updateRejectedMap(const std::string& renamedExpr,
            const std::string& markedExpr,
            const std::string& expr,
            LocalMemory& hashMemory,
            int iteration) {
            RejectedMapValue v(renamedExpr, expr, iteration);
            hashMemory.rejectedMap[markedExpr].insert(v);
        }


        inline std::string makeMarkedExpr(const std::string& expr, const std::string& var) {
            std::map<std::string, std::string> replacementMap;
            replacementMap.insert(std::make_pair(var, "marker"));
            return ce::replaceKeysInString(expr, replacementMap);
        }

        inline int extractMaxIterationNumber(const std::string& s) {
            static const std::regex pattern(R"(it_(\d+)_lev_\d+_)");
            int maxVal = -1;

            std::sregex_iterator it(s.begin(), s.end(), pattern);
            std::sregex_iterator end;
            for (; it != end; ++it) {
                const std::smatch& m = *it;
                const int v = std::atoi(m.str(1).c_str());
                if (v > maxVal) {
                    maxVal = v;
                }
            }
            // Returns -1 if no matches (Python's default=None equivalent).
            return maxVal;
        }

        inline bool isAdmitted(LocalMemory& hashMemory,
            const std::string& expr,
            const std::string& var,
            const std::string& markedExpr) {
            bool result = false;

            const std::string coreExpr = ce::extractExpression(expr);
            if (ce::operators.find(coreExpr) == ce::operators.end()) {
                return result;
            }

            // parts = expr.split(var)  with assert len(parts) == 2
            const std::size_t pos = expr.find(var);
            assert(pos != std::string::npos);
            const std::size_t pos2 = expr.find(var, pos + var.size());
            assert(pos2 == std::string::npos); // exactly one occurrence

            const std::string part0 = expr.substr(0, pos);
            const std::string part1 = expr.substr(pos + var.size());

            std::map<std::string, std::set<AdmissionMapValue> >::iterator mit =
                hashMemory.admissionMap.find(markedExpr);
            if (mit == hashMemory.admissionMap.end()) {
                return result;
            }

            // snapshot set into a vector (mirrors Python list())
            std::vector<AdmissionMapValue> tuples;
            tuples.reserve(mit->second.size());
            for (std::set<AdmissionMapValue>::const_iterator it = mit->second.begin();
                it != mit->second.end(); ++it) {
                tuples.push_back(*it);
            }

            for (std::size_t i = 0; i < tuples.size(); ++i) {
                const std::vector<std::string> keyVec = tuples[i].key;
                const std::set<std::string> remainingArgs = tuples[i].remainingArgs;
                const int maxAdmissionDepth = tuples[i].standardMaxAdmissionDepth;
                const int maxSecondaryNumber = tuples[i].standardMaxSecondaryNumber;
                const bool partOfRecursion = tuples[i].flag;

                const int mn = extractMaxIterationNumber(var);
                const int cnt = countPatternOccurrences(expr, hashMemory);

                if (mn <= maxAdmissionDepth && cnt <= maxSecondaryNumber) {
                    result = true;

                    std::map<std::string, bool>::iterator sit =
                        hashMemory.admissionStatusMap.find(markedExpr);
                    assert(sit != hashMemory.admissionStatusMap.end());
                    if (sit != hashMemory.admissionStatusMap.end() && sit->second) {
                        hashMemory.productsOfRecursion.insert(var);
                    }

                    for (std::size_t e = 0; e < keyVec.size(); ++e) {
                        const std::string& element = keyVec[e];

                        // startswith/endswith checks with part0/part1
                        bool startsOk = false;
                        if (element.size() >= part0.size()) {
                            startsOk = (element.compare(0, part0.size(), part0) == 0);
                        }
                        bool endsOk = false;
                        if (element.size() >= part1.size()) {
                            const std::size_t off = element.size() - part1.size();
                            endsOk = (element.compare(off, part1.size(), part1) == 0);
                        }

                        if (startsOk && endsOk) {
                            const std::size_t coreLen =
                                element.size() - part0.size() - part1.size();
                            const std::string varForRepl =
                                element.substr(part0.size(), coreLen);

                            std::map<std::string, std::string> replacementMap;
                            replacementMap.insert(std::make_pair(varForRepl, var));

                            std::vector<std::string> replKey;
                            replKey.reserve(keyVec.size());
                            for (std::size_t k = 0; k < keyVec.size(); ++k) {
                                replKey.push_back(ce::replaceKeysInString(keyVec[k], replacementMap));
                            }

                            std::set<std::string> newRemArgs = remainingArgs;
                            newRemArgs.insert(var);

                            updateAdmissionMap(hashMemory,
                                replKey,
                                newRemArgs,
                                maxAdmissionDepth,
                                maxSecondaryNumber,
                                partOfRecursion);
                        }
                    }
                }
            }

            return result;
        }

        inline static bool startsWith(const std::string& s, const char* pfx, std::size_t n) {
            if (s.size() < n) return false;
            for (std::size_t i = 0; i < n; ++i) if (s[i] != pfx[i]) return false;
            return true;
        }

        void disintegrateExprNode(const std::string& expr2,
            const std::string& expandedExpr,
            int& startInt,
            int iteration,
            int level,
            LocalMemory& hashMemory,
            std::set<std::string>& implications,
            std::set<std::string>& statements) {
            // 1) groom
            const std::vector<std::string> subexprsAfterGrooming = groomExpr(expr2);

            // 2) smoothen
            std::vector<std::string> renamedSubexprsAfterGrooming;
            renamedSubexprsAfterGrooming.reserve(subexprsAfterGrooming.size());
            for (std::size_t i = 0; i < subexprsAfterGrooming.size(); ++i) {
                const std::string sm = this->smoothenExpr(subexprsAfterGrooming[i]);
                renamedSubexprsAfterGrooming.push_back(sm);
            }

            // 3) iterate
            for (std::size_t i = 0; i < renamedSubexprsAfterGrooming.size(); ++i) {
                const std::string& subexpr = renamedSubexprsAfterGrooming[i];

                if (startsWith(subexpr, "(>", 2)) {
                    implications.insert(subexpr);
                }
                else if (startsWith(subexpr, "!(", 2) && subexpr.size() >= 3 && subexpr[2] == '>') {
                    // rename last removed
                    std::string renamedSubdef;
                    int newStartInt = startInt;
                    std::string newVar;
                    {
                        std::tuple<std::string, int, std::string> t = renameLastRemoved(subexpr, startInt, iteration, level);
                        renamedSubdef = std::get<0>(t);
                        newStartInt = std::get<1>(t);
                        newVar = std::get<2>(t);
                        startInt = newStartInt;
                    }

                    // split into left/right
                    std::pair<std::string, std::string> lr = extractValues(renamedSubdef, this->coreExpressionMap);
                    std::string leftExpr = lr.first;
                    std::string rightExpr = lr.second;

                    // toggle negation on right
                    if (!rightExpr.empty() && rightExpr[0] == '!') {
                        rightExpr = rightExpr.substr(1);
                    }
                    else {
                        rightExpr = std::string("!") + rightExpr;
                    }

                    // mark with newVar
                    const std::string leftMarked = makeMarkedExpr(leftExpr, newVar);
                    const std::string rightMarked = makeMarkedExpr(rightExpr, newVar);

                    const bool admittedLeft = isAdmitted(hashMemory, leftExpr, newVar, leftMarked);
                    const bool admittedRight = isAdmitted(hashMemory, rightExpr, newVar, rightMarked);

                    if (admittedLeft || admittedRight) {
                        // recurse on both
                        this->disintegrateExprNode(leftExpr, expandedExpr, startInt, iteration, level, hashMemory, implications, statements);
                        this->disintegrateExprNode(rightExpr, expandedExpr, startInt, iteration, level, hashMemory, implications, statements);
                    }
                    else {
                        // record rejection and undo startInt increment
                        updateRejectedMap(renamedSubdef, leftMarked, expandedExpr, hashMemory, iteration);
                        updateRejectedMap(renamedSubdef, rightMarked, expandedExpr, hashMemory, iteration);
                        startInt -= 1;
                    }
                }
                else if (startsWith(subexpr, "!(&", 3)) {
                    throw std::runtime_error("Should not happen.");
                }
                else {
                    statements.insert(subexpr);
                }
            }
        }

        std::tuple<std::set<std::string>, std::set<std::string>, int>
            disintegrateExpr(const std::string& expr,
                int startInt,
                int iteration,
                int level,
                LocalMemory& hashMemory) {
            std::set<std::string> implications;
            std::set<std::string> statements;

            this->disintegrateExprNode(expr, expr, startInt, iteration, level, hashMemory, implications, statements);

            return std::make_tuple(implications, statements, startInt);
        }


        inline bool filterIterations(const std::string& expr2, const EquivalenceClass& eqClass) {

            // Precompiled regex: it_<digits>_lev_<digits>_<digits>
            static const std::regex FILTER_PATTERN(R"(it_(\d+)_lev_\d+_\d+)");

            // Build eq_class_matches = { s in eqClass.variables | full match of FILTER_PATTERN }
            std::set<std::string> eqClassMatches;
            for (std::set<std::string>::const_iterator it = eqClass.variables.begin();
                it != eqClass.variables.end(); ++it) {
                if (std::regex_match(*it, FILTER_PATTERN)) {
                    eqClassMatches.insert(*it);
                }
            }

            if (eqClassMatches.empty()) {
                return true; // Nothing to compare against.
            }

            // Lexicographically smallest candidate in eq_class_matches
            const std::string minCandidate = *eqClassMatches.begin();

            // Iterate over matches in expr2
            std::sregex_iterator it2(expr2.begin(), expr2.end(), FILTER_PATTERN);
            std::sregex_iterator end2;
            for (; it2 != end2; ++it2) {
                const std::string candidate = it2->str();
                if (eqClassMatches.find(candidate) != eqClassMatches.end() &&
                    candidate != minCandidate) {
                    return false;
                }
            }

            return true;
        }

        inline std::set<std::string>
            reduceEqClass(const std::set<std::string>& eqClassVariables) {
            std::set<std::string> matchingMembers;
            std::set<std::string> nonMatchingMembers;

            // Precompiled regex: it_<digits>_lev_<digits>_<digits>
            static const std::regex FILTER_PATTERN(R"(it_(\d+)_lev_\d+_\d+)");

            // Partition into matching / non-matching
            for (std::set<std::string>::const_iterator it = eqClassVariables.begin();
                it != eqClassVariables.end(); ++it) {
                const std::string& s = *it;
                if (std::regex_match(s, FILTER_PATTERN)) {
                    matchingMembers.insert(s);
                }
                else {
                    nonMatchingMembers.insert(s);
                }
            }

            // If there are any non-matching members, keep all of them
            if (!nonMatchingMembers.empty()) {
                return nonMatchingMembers;
            }

            // Otherwise, keep only the lexicographically smallest matching member (if any)
            std::set<std::string> out;
            if (!matchingMembers.empty()) {
                out.insert(*matchingMembers.begin()); // std::set is ordered; begin() is the minimum
            }
            return out; // empty if input was empty
        }

        inline void applyEquivalenceClass(const EquivalenceClass& clss,
            const std::string& expr2,
            BodyOfProves& memoryBlock,
            const std::set<int>& levels,
            std::vector<std::string>& newStatements) {

            if (isEquality(expr2)) {
                return; // Skip equalities
			}

            // expr_levels_map: new_expr -> levels
            std::map<std::string, std::set<int> > exprLevelsMap;
            // expr_origin_map (local): new_expr -> sorted equality facts
            std::map<std::string, std::vector<std::string> > exprOriginMapLocal;

            // Cache args once
            const std::vector<std::string> argsExpr = ce::getArgs(expr2);

            // Indices of args that are in the equivalence class
            std::vector<int> indices;
            indices.reserve(argsExpr.size());
            for (std::size_t i = 0; i < argsExpr.size(); ++i) {
                if (clss.variables.find(argsExpr[i]) != clss.variables.end()) {
                    indices.push_back(static_cast<int>(i));
                }
            }

            // Reduced eq class as a list
            const std::set<std::string> reducedSet = this->reduceEqClass(clss.variables);
            std::vector<std::string> eqList;
            eqList.reserve(reducedSet.size());
            for (std::set<std::string>::const_iterator it = reducedSet.begin(); it != reducedSet.end(); ++it) {
                eqList.push_back(*it);
            }

            // Base expression and wrappers
            std::string baseExpr;
            std::string wrapLeft;
            std::string wrapRight;
            if (!expr2.empty() && expr2[0] == '(') {
                baseExpr = ce::extractExpression(expr2);
                wrapLeft = "(";
                wrapRight = ")";
            }
            else {
                // must start with "!("
                assert(expr2.size() >= 2 && expr2[0] == '!' && expr2[1] == '(');
                baseExpr = ce::extractExpressionFromNegation(expr2);
                wrapLeft = "!(";
                wrapRight = ")";
            }

            if (!indices.empty()) {
                // Look up mappings by (len(indices), len(eq_list))
                const std::pair<int, int> key(static_cast<int>(indices.size()), static_cast<int>(eqList.size()));
                std::map< std::pair<int, int>, std::vector<std::vector<int> > >::const_iterator mit =
                    this->allMappingsAna.find(key);
                if (mit != this->allMappingsAna.end()) {
                    const std::vector<std::vector<int> >& mappings = mit->second;

                    for (std::size_t m = 0; m < mappings.size(); ++m) {
                        const std::vector<int>& mapping = mappings[m];

                        std::set<std::string> setEqualities;
                        std::set<int> newLevels = levels; // copy

                        // temp_list = args_expr copy
                        std::vector<std::string> tempList = argsExpr;

                        // Substitute according to mapping
                        for (std::size_t i = 0; i < indices.size() && i < mapping.size(); ++i) {
                            const int idxInArgs = indices[i];
                            const int mappedIdx = mapping[i];
                            assert(mappedIdx >= 0 && mappedIdx < static_cast<int>(eqList.size()));

                            const std::string& fromVar = tempList[static_cast<std::size_t>(idxInArgs)];
                            const std::string& toVar = eqList[static_cast<std::size_t>(mappedIdx)];

                            std::set<std::string> fs;
                            fs.insert(fromVar);
                            fs.insert(toVar);

                            if (fs.size() > 1U) {
                                // new_levels update with equality levels map
                                std::map< std::set<std::string>, std::set<int> >::const_iterator itLev =
                                    clss.equalityLevelsMap.find(fs);
                                if (itLev != clss.equalityLevelsMap.end()) {
                                    const std::set<int>& lv = itLev->second;
                                    for (std::set<int>::const_iterator lit = lv.begin(); lit != lv.end(); ++lit) {
                                        newLevels.insert(*lit);
                                    }
                                }
                                if (parameters::trackHistory) {
                                    setEqualities.insert(std::string("(=[") + fromVar + "," + toVar + "])");
                                }
                            }

                            tempList[static_cast<std::size_t>(idxInArgs)] = toVar;
                        }

                        // Build new expression: wrapLeft + baseExpr + "[" + ",".join(tempList) + "]" + wrapRight
                        std::string joined;
                        for (std::size_t i = 0; i < tempList.size(); ++i) {
                            if (i > 0) joined.push_back(',');
                            joined += tempList[i];
                        }
                        const std::string newExpr = wrapLeft + baseExpr + "[" + joined + "]" + wrapRight;

                        exprLevelsMap[newExpr] = newLevels;

                        if (parameters::trackHistory) {
                            if (memoryBlock.exprOriginMap.find(newExpr) == memoryBlock.exprOriginMap.end()) {
                                // sorted equality list
                                std::vector<std::string> eqs(setEqualities.begin(), setEqualities.end());
                                std::sort(eqs.begin(), eqs.end());
                                exprOriginMapLocal[newExpr] = eqs;
                            }
                        }
                    }
                }
            }

            // new_exprs = exprLevelsMap.keys() \ memoryBlock.statementLevelsMap.keys()
            for (std::map<std::string, std::set<int> >::const_iterator it = exprLevelsMap.begin();
                it != exprLevelsMap.end(); ++it) {
                const std::string& applied = it->first;

                if (memoryBlock.statementLevelsMap.find(applied) != memoryBlock.statementLevelsMap.end()) {
                    continue;
                }

                const int mn = this->extractMaxIterationNumber(applied);
                if (mn != -1 && mn > parameters::maxIterationNumberVariable) {
                    continue;
                }
                if (this->countPatternOccurrences(applied, memoryBlock.localMemory) >
                    parameters::maxNumberSecondaryVariables) {
                    continue;
                }

                // Only if neither present and max(levels) == memoryBlock.level
                if (memoryBlock.statementLevelsMap.find(applied) == memoryBlock.statementLevelsMap.end() &&
                    memoryBlock.exprOriginMap.find(applied) == memoryBlock.exprOriginMap.end()) {

                    // compute max level of this expr
                    int maxLevel = -2147483647; // INT_MIN
                    const std::set<int>& lvls = it->second;
                    for (std::set<int>::const_iterator lit = lvls.begin(); lit != lvls.end(); ++lit) {
                        if (*lit > maxLevel) maxLevel = *lit;
                    }
                    if (maxLevel != memoryBlock.level) {
                        continue;
                    }

                    // Commit to memory block
                    memoryBlock.statementLevelsMap[applied] = lvls;
                    memoryBlock.encodedStatements.push_back(EncodedExpression(applied));
                    memoryBlock.localEncodedStatements.push_back(EncodedExpression(applied));
                    memoryBlock.localEncodedStatementsDelta.push_back(EncodedExpression(applied));
                    newStatements.push_back(applied);

                    // Mail out statement with its levels
                    memoryBlock.mailOut.statements.insert(
                        std::make_pair(applied, lvls)
                    );

                    if (parameters::trackHistory) {
                        if (memoryBlock.exprOriginMap.find(applied) == memoryBlock.exprOriginMap.end()) {
                            std::vector<std::string> origin;
                            origin.push_back("equality1");
                            origin.push_back(expr2);

                            std::map<std::string, std::vector<std::string> >::const_iterator oit =
                                exprOriginMapLocal.find(applied);
                            if (oit != exprOriginMapLocal.end()) {
                                const std::vector<std::string>& tail = oit->second;
                                for (std::size_t i = 0; i < tail.size(); ++i) origin.push_back(tail[i]);
                            }

                            memoryBlock.exprOriginMap[applied] = origin;
                            memoryBlock.mailOut.exprOriginMap[applied] = origin;
                        }
                    }
                }
            }
        }

        static bool isSubsetOf(const std::set<std::string>& a, const std::set<std::string>& b) {
            for (std::set<std::string>::const_iterator it = a.begin(); it != a.end(); ++it) {
                if (b.find(*it) == b.end()) return false;
            }
            return true;
        }

        // tmp := left; overwrite with right's entries (Python dict union: left | right -> right wins)
        static void overwriteLevels(std::map<std::set<std::string>, std::set<int> >& left,
            const std::map<std::set<std::string>, std::set<int> >& right) {
            for (std::map<std::set<std::string>, std::set<int> >::const_iterator it = right.begin();
                it != right.end(); ++it) {
                left[it->first] = it->second;
            }
        }

        static void overwriteOrigins(std::map<std::string, std::vector<std::string> >& left,
            const std::map<std::string, std::vector<std::string> >& right) {
            for (std::map<std::string, std::vector<std::string> >::const_iterator it = right.begin();
                it != right.end(); ++it) {
                left[it->first] = it->second;
            }
        }

        inline void mergeTwoEquivalenceClasses(EquivalenceClass& classA,
            const EquivalenceClass& classB,
            const std::set<std::string>& eqArgs,
            const std::set<int>& levels,
            BodyOfProves& memoryBlock) const {
            // If A ⊆ B: union and right-precedence dict union (B | A), then return.
            if (isSubsetOf(classA.variables, classB.variables)) {
                // variables := B ∪ A
                std::set<std::string> vars = classB.variables;
                for (std::set<std::string>::const_iterator it = classA.variables.begin();
                    it != classA.variables.end(); ++it) {
                    vars.insert(*it);
                }
                classA.variables.swap(vars);

                // equalityLevelsMap := (B | A)  (right operand A has precedence)
                std::map<std::set<std::string>, std::set<int> > newLevels = classB.equalityLevelsMap;
                overwriteLevels(newLevels, classA.equalityLevelsMap);
                classA.equalityLevelsMap.swap(newLevels);

                // equalityOriginMap := (B | A)
                std::map<std::string, std::vector<std::string> > newOrigins = classB.equalityOriginMap;
                overwriteOrigins(newOrigins, classA.equalityOriginMap);
                classA.equalityOriginMap.swap(newOrigins);

                return;
            }

            // assert not eq_args ⊆ classB.variables
            assert(!isSubsetOf(eqArgs, classB.variables));

            // common_arg := ((eq_args ∩ A.variables) ∩ B.variables), must have size 1
            std::set<std::string> tmp;
            for (std::set<std::string>::const_iterator it = eqArgs.begin(); it != eqArgs.end(); ++it) {
                if (classA.variables.find(*it) != classA.variables.end() &&
                    classB.variables.find(*it) != classB.variables.end()) {
                    tmp.insert(*it);
                }
            }
            assert(tmp.size() == 1);
            const std::string commonArg = *tmp.begin();

            // Build merged maps (levels + optional origin history)
            std::map<std::set<std::string>, std::set<int> > mergedMap;
            std::map<std::string, std::vector<std::string> > mergedOriginMap;

            for (std::set<std::string>::const_iterator ita = classA.variables.begin();
                ita != classA.variables.end(); ++ita) {
                const std::string& varA = *ita;
                if (varA == commonArg) continue;

                for (std::set<std::string>::const_iterator itb = classB.variables.begin();
                    itb != classB.variables.end(); ++itb) {
                    const std::string& varB = *itb;
                    if (varB == commonArg) continue;

                    // new_levels = copy(levels)
                    std::set<int> newLevels = levels;

                    // fs_aa = {commonArg, varA}
                    std::set<std::string> fsAA;
                    fsAA.insert(commonArg);
                    fsAA.insert(varA);

                    // fs_bb = {commonArg, varB}
                    std::set<std::string> fsBB;
                    fsBB.insert(commonArg);
                    fsBB.insert(varB);

                    // if fs_aa in classA.levelsMap: update
                    std::map<std::set<std::string>, std::set<int> >::const_iterator la =
                        classA.equalityLevelsMap.find(fsAA);
                    if (la != classA.equalityLevelsMap.end()) {
                        for (std::set<int>::const_iterator itl = la->second.begin();
                            itl != la->second.end(); ++itl) {
                            newLevels.insert(*itl);
                        }
                    }

                    // if fs_bb in classB.levelsMap: update
                    std::map<std::set<std::string>, std::set<int> >::const_iterator lb =
                        classB.equalityLevelsMap.find(fsBB);
                    if (lb != classB.equalityLevelsMap.end()) {
                        for (std::set<int>::const_iterator itl = lb->second.begin();
                            itl != lb->second.end(); ++itl) {
                            newLevels.insert(*itl);
                        }
                    }

                    // merged_map[{varA, varB}] = new_levels
                    std::set<std::string> keyAB;
                    keyAB.insert(varA);
                    keyAB.insert(varB);
                    mergedMap[keyAB] = newLevels;

                    // history
                    if (parameters::trackHistory) {
                        if (varA != commonArg && varB != commonArg) {
                            const std::string eq1 = std::string("(=[") + varA + "," + varB + "])";
                            const std::string eq2 = std::string("(=[") + varB + "," + varA + "])";

                            std::vector<std::string> or1;
                            or1.push_back("equality2");
                            or1.push_back(std::string("(=[") + varA + "," + commonArg + "])");
                            or1.push_back(std::string("(=[") + commonArg + "," + varB + "])");

                            std::vector<std::string> or2;
                            or2.push_back("equality2");
                            or2.push_back(std::string("(=[") + varB + "," + commonArg + "])");
                            or2.push_back(std::string("(=[") + commonArg + "," + varA + "])");

                            if (memoryBlock.exprOriginMap.find(eq1) == memoryBlock.exprOriginMap.end()) {
                                mergedOriginMap[eq1] = or1;
                                memoryBlock.exprOriginMap[eq1] = or1;
                                memoryBlock.mailOut.exprOriginMap[eq1] = or1;

                                mergedOriginMap[eq2] = or2;
                                memoryBlock.exprOriginMap[eq2] = or2;
                                memoryBlock.mailOut.exprOriginMap[eq2] = or2;
                            }
                        }
                    }
                }
            }

            // classA.variables = classB.variables | classA.variables
            for (std::set<std::string>::const_iterator it = classB.variables.begin();
                it != classB.variables.end(); ++it) {
                classA.variables.insert(*it);
            }

            // classA.equalityLevelsMap = classB.equalityLevelsMap | classA.equalityLevelsMap
            {
                std::map<std::set<std::string>, std::set<int> > tmpMap = classB.equalityLevelsMap;
                overwriteLevels(tmpMap, classA.equalityLevelsMap); // right (A) overrides
                classA.equalityLevelsMap.swap(tmpMap);
            }
            // classA.equalityLevelsMap = merged_map | classA.equalityLevelsMap
            {
                std::map<std::set<std::string>, std::set<int> > tmpMap = mergedMap;
                overwriteLevels(tmpMap, classA.equalityLevelsMap); // right (existing A) overrides
                classA.equalityLevelsMap.swap(tmpMap);
            }

            // classA.equalityOriginMap = classB.equalityOriginMap | classA.equalityOriginMap
            {
                std::map<std::string, std::vector<std::string> > tmpMap = classB.equalityOriginMap;
                overwriteOrigins(tmpMap, classA.equalityOriginMap); // right (A) overrides
                classA.equalityOriginMap.swap(tmpMap);
            }
            // classA.equalityOriginMap = merged_origin_map | classA.equality_origin_map
            {
                std::map<std::string, std::vector<std::string> > tmpMap = mergedOriginMap;
                overwriteOrigins(tmpMap, classA.equalityOriginMap); // right (existing A) overrides
                classA.equalityOriginMap.swap(tmpMap);
            }
        }

        inline std::vector<std::string>
            cleanUpExpressions(BodyOfProves& mb,
                const std::vector<std::string>& newStatementsIn) {
            std::vector<std::string> newStatements = newStatementsIn;

            for (std::vector<EquivalenceClass>::const_iterator clsIt = mb.equivalenceClasses.begin();
                clsIt != mb.equivalenceClasses.end(); ++clsIt) {

                const EquivalenceClass& eqClss = *clsIt;

                // ---- local_encoded_statements ----
                std::vector<EncodedExpression> newLocalEncoded;
                newLocalEncoded.reserve(mb.localEncodedStatements.size());
                for (std::size_t i = 0; i < mb.localEncodedStatements.size(); ++i) {
                    const EncodedExpression& ee = mb.localEncodedStatements[i];
                    if (filterIterations(ee.original, eqClss)) {
                        newLocalEncoded.push_back(ee);
                    }
                }

                std::vector<EncodedExpression> newLocalEncodedDelta;
                newLocalEncodedDelta.reserve(mb.localEncodedStatementsDelta.size());
                for (std::size_t i = 0; i < mb.localEncodedStatementsDelta.size(); ++i) {
                    const EncodedExpression& ee = mb.localEncodedStatementsDelta[i];
                    if (filterIterations(ee.original, eqClss)) {
                        newLocalEncodedDelta.push_back(ee);
                    }
                }

                // remove dropped locals from statementLevelsMap
                {
                    std::set<EncodedExpression> oldSet(mb.localEncodedStatements.begin(),
                        mb.localEncodedStatements.end());
                    std::set<EncodedExpression> newSet(newLocalEncoded.begin(),
                        newLocalEncoded.end());
                    for (std::set<EncodedExpression>::const_iterator it = oldSet.begin();
                        it != oldSet.end(); ++it) {
                        if (newSet.find(*it) == newSet.end()) {
                            // keys in map are strings -> erase by original text
                            mb.statementLevelsMap.erase(it->original);
                        }
                    }
                }

                mb.localEncodedStatements = newLocalEncoded;
                mb.localEncodedStatementsDelta = newLocalEncodedDelta;

                // ---- encoded_statements ----
                std::vector<EncodedExpression> newEncodedStatements;
                newEncodedStatements.reserve(mb.encodedStatements.size());
                for (std::size_t i = 0; i < mb.encodedStatements.size(); ++i) {
                    const EncodedExpression& ee = mb.encodedStatements[i];
                    if (filterIterations(ee.original, eqClss)) {
                        newEncodedStatements.push_back(ee);
                    }
                }

                // remove dropped from statementLevelsMap
                {
                    std::set<EncodedExpression> oldSet(mb.encodedStatements.begin(),
                        mb.encodedStatements.end());
                    std::set<EncodedExpression> newSet(newEncodedStatements.begin(),
                        newEncodedStatements.end());
                    for (std::set<EncodedExpression>::const_iterator it = oldSet.begin();
                        it != oldSet.end(); ++it) {
                        if (newSet.find(*it) == newSet.end()) {
                            mb.statementLevelsMap.erase(it->original);
                        }
                    }
                }

                mb.encodedStatements = newEncodedStatements;

                // ---- newStatements (strings) ----
                std::vector<std::string> filtered;
                filtered.reserve(newStatements.size());
                for (std::size_t i = 0; i < newStatements.size(); ++i) {
                    if (filterIterations(newStatements[i], eqClss)) {
                        filtered.push_back(newStatements[i]);
                    }
                }
                newStatements.swap(filtered);
            }

            return newStatements;
        }

        std::vector<std::string>
            updateEquivalenceClasses(BodyOfProves& mb,
                const std::string& eqlty,
                const std::set<int>& levels,
                const std::vector<std::string>& origin,
                const std::vector<std::string>& newStatementsIn) {
            std::vector<std::string> newStatements = newStatementsIn;

            // args_list / eq_args
            const std::vector<std::string> argsList = ce::getArgs(eqlty);
            std::set<std::string> eqArgs;
            for (std::size_t i = 0; i < argsList.size(); ++i) eqArgs.insert(argsList[i]);

            // mirrored "(=[b,a])"
            const std::string mirrored = std::string("(=[") + argsList[1] + "," + argsList[0] + "])";

            // merged_class init
            EquivalenceClass mergedClass;
            mergedClass.variables = eqArgs;
            mergedClass.equalityLevelsMap[eqArgs] = levels;
            mergedClass.equalityOriginMap[eqlty] = origin;
            std::vector<std::string> mirroredOrigin;
            mirroredOrigin.push_back("symmetry of equality");
            mirroredOrigin.push_back(eqlty);
            mergedClass.equalityOriginMap[mirrored] = mirroredOrigin;

            if (parameters::trackHistory) {
                if (mb.exprOriginMap.find(eqlty) == mb.exprOriginMap.end()) {
                    mb.exprOriginMap[eqlty] = origin;
                    mb.mailOut.exprOriginMap[eqlty] = origin;
                }
                if (mb.exprOriginMap.find(mirrored) == mb.exprOriginMap.end()) {
                    mb.exprOriginMap[mirrored] = mirroredOrigin;
                    mb.mailOut.exprOriginMap[mirrored] = mirroredOrigin;
                }
            }

            // Merge all eq-classes that overlap with eq_args
            std::vector<EquivalenceClass> newClasses;
            newClasses.reserve(mb.equivalenceClasses.size() + 1);
            for (std::size_t i = 0; i < mb.equivalenceClasses.size(); ++i) {
                const EquivalenceClass& eqClss = mb.equivalenceClasses[i];

                bool overlaps = false;
                for (std::set<std::string>::const_iterator it = eqArgs.begin(); it != eqArgs.end(); ++it) {
                    if (eqClss.variables.find(*it) != eqClss.variables.end()) { overlaps = true; break; }
                }

                if (overlaps) {
                    this->mergeTwoEquivalenceClasses(mergedClass, eqClss, eqArgs, levels, mb);
                    mb.eqClassSttmntIndexMap.erase(eqClss.variables); // pop(frozenset, None)
                }
                else {
                    newClasses.push_back(eqClss);
                }
            }

            mb.eqClassSttmntIndexMap[mergedClass.variables] = 0;
            newClasses.push_back(mergedClass);
            mb.equivalenceClasses.swap(newClasses);

            // Apply merged class to all encoded statements
            const std::size_t initial_count = mb.encodedStatements.size();   // snapshot
            for (std::size_t index = 0; index < initial_count; ++index) {
                const std::string original = mb.encodedStatements[index].original; // <-- copy!
                const auto lvIt = mb.statementLevelsMap.find(original);
                const std::set<int> emptyLevels;
                const std::set<int>& lvls = (lvIt != mb.statementLevelsMap.end()) ? lvIt->second : emptyLevels;

                this->applyEquivalenceClass(mergedClass, original, mb, lvls, newStatements);
            }


            mb.eqClassSttmntIndexMap[mergedClass.variables] =
                static_cast<int>(mb.encodedStatements.size());

            // Cleanup
            newStatements = this->cleanUpExpressions(mb, newStatements);

            // expr_origin_map = merged_class.equality_origin_map | mb.expr_origin_map
            {
                std::map<std::string, std::vector<std::string> > merged = mergedClass.equalityOriginMap;
                // right-precedence union
                overwriteOrigins(merged, mb.exprOriginMap);
                mb.exprOriginMap.swap(merged);
            }

            return newStatements;
        }

        inline bool isEquality(const std::string& cand) {
            return cand.size() >= 3 &&
                cand[0] == '(' &&
                cand[1] == '=' &&
                cand[2] == '[';
        }

        inline std::vector<std::string>
            addStatement(const std::string& expr,
                BodyOfProves& memoryBlock,
                bool local,
                const std::set<int>& levels,
                const std::vector<std::string>& origin) {
            std::vector<std::string> newStatements;

            const int maxIteration = extractMaxIterationNumber(expr);
            if (maxIteration != -1 && maxIteration > parameters::maxIterationNumberVariable) {
                return newStatements;
            }

            if (countPatternOccurrences(expr, memoryBlock.localMemory) >
                parameters::maxNumberSecondaryVariables) {
                return newStatements;
            }

            if (!isEquality(expr)) {
                // not in memory already?
                if (memoryBlock.statementLevelsMap.find(expr) == memoryBlock.statementLevelsMap.end()) {

                    // Check if any equivalence class filters the expr out
                    bool anyFilteredOut = false;
                    for (std::size_t i = 0; i < memoryBlock.equivalenceClasses.size(); ++i) {
                        if (!filterIterations(expr, memoryBlock.equivalenceClasses[i])) {
                            anyFilteredOut = true; break;
                        }
                    }

                    if (!anyFilteredOut) {
                        memoryBlock.statementLevelsMap[expr] = levels;

                        if (parameters::trackHistory) {
                            if (memoryBlock.exprOriginMap.find(expr) == memoryBlock.exprOriginMap.end()) {
                                memoryBlock.exprOriginMap[expr] = origin;
                                // should not already exist in mailOut
                                memoryBlock.mailOut.exprOriginMap[expr] = origin;
                            }
                        }

                        memoryBlock.encodedStatements.push_back(EncodedExpression(expr));
                        if (local) {
                            memoryBlock.localEncodedStatements.push_back(EncodedExpression(expr));
                            memoryBlock.localEncodedStatementsDelta.push_back(EncodedExpression(expr));
                            newStatements.push_back(expr);

                            memoryBlock.mailOut.statements.insert(std::make_pair(expr, levels));

                            if (parameters::trackHistory) {
                                if (memoryBlock.mailOut.exprOriginMap.find(expr) ==
                                    memoryBlock.mailOut.exprOriginMap.end()) {
                                    memoryBlock.mailOut.exprOriginMap[expr] = origin;
                                }
                            }
                        }
                    }
                }

                // Apply each equivalence class to generate more statements from expr
                for (std::size_t i = 0; i < memoryBlock.equivalenceClasses.size(); ++i) {
                    this->applyEquivalenceClass(memoryBlock.equivalenceClasses[i],
                        expr,
                        memoryBlock,
                        levels,
                        newStatements);
                }
            }
            else {
                newStatements = this->updateEquivalenceClasses(memoryBlock, expr, levels, origin, newStatements);
				newStatements.push_back(expr); // also add the equality itself
            }

            // Iteratively apply equivalence classes to any newly generated statements
            std::size_t oldSize = memoryBlock.encodedStatements.size();
            while (true) {
                for (std::size_t c = 0; c < memoryBlock.equivalenceClasses.size(); ++c) {
                    const EquivalenceClass& eqc = memoryBlock.equivalenceClasses[c];

                    int startIndex = 0;
                    std::map<std::set<std::string>, int>::const_iterator itIdx =
                        memoryBlock.eqClassSttmntIndexMap.find(eqc.variables);
                    if (itIdx != memoryBlock.eqClassSttmntIndexMap.end()) {
                        startIndex = itIdx->second;
                    }

                    for (int idx = startIndex; idx < static_cast<int>(memoryBlock.encodedStatements.size()); ++idx) {
                        const std::string& s = memoryBlock.encodedStatements[static_cast<std::size_t>(idx)].original;

                        std::map<std::string, std::set<int> >::const_iterator lvIt =
                            memoryBlock.statementLevelsMap.find(s);
                        const std::set<int> emptyLevels;
                        const std::set<int>& lvls = (lvIt != memoryBlock.statementLevelsMap.end())
                            ? lvIt->second : emptyLevels;

                        this->applyEquivalenceClass(eqc, s, memoryBlock, lvls, newStatements);
                    }

                    memoryBlock.eqClassSttmntIndexMap[eqc.variables] =
                        static_cast<int>(memoryBlock.encodedStatements.size());
                }

                if (oldSize == memoryBlock.encodedStatements.size()) {
                    break;
                }
                oldSize = memoryBlock.encodedStatements.size();
            }

            if (isEquality(expr)) {
                newStatements = this->cleanUpExpressions(memoryBlock, newStatements);
            }

            return newStatements;
        }



        inline bool isProved(const std::string& s) {
            // Precompiled patterns (equivalent to Python's _is_proved_pattern / _is_proved_pattern2)
            static const std::regex kIsProvedPattern1(R"(it_\d+_lev_\d+_)");
            static const std::regex kIsProvedPattern2(R"(c\d+)");

            const bool hasMatch1 = std::regex_search(s, kIsProvedPattern1);
            const bool hasMatch2 = std::regex_search(s, kIsProvedPattern2);
            return !hasMatch1 && !hasMatch2;
        }

        inline std::string findZeroArgName(BodyOfProves& memoryBlock) {
            BodyOfProves* current = &memoryBlock;
            while (true) {
                if (!current->exprKey.empty() &&
                    current->exprKey.find("(NaturalNumbers[") == 0) { // starts with
                    const std::vector<std::string> args = ce::getArgs(current->exprKey);
                    assert(args.size() > 1);
                    return args[1];
                }
                else if (current->parentBodyOfProves != NULL) {
                    current = current->parentBodyOfProves;
                }
                else {
                    assert(false); // not found up the chain
                }
            }
            // Unreachable, but keeps some compilers happy:
            return std::string();
        }

        inline std::string findSName(BodyOfProves& memoryBlock) {
            BodyOfProves* current = &memoryBlock;
            while (true) {
                if (!current->exprKey.empty() &&
                    current->exprKey.find("(NaturalNumbers[") == 0) { // starts with
                    const std::vector<std::string> args = ce::getArgs(current->exprKey);
                    assert(args.size() > 1);
                    return args[3];
                }
                else if (current->parentBodyOfProves != NULL) {
                    current = current->parentBodyOfProves;
                }
                else {
                    assert(false); // not found up the chain
                }
            }
            // Unreachable, but keeps some compilers happy:
            return std::string();
        }

        // analyze_expressions.hpp  inside struct ExpressionAnalyzer
        inline void sendMail(const BodyOfProves& sender,
            const Mail& mail,
            unsigned coreId,
            const ParentChildrenMap& index,
            PerCoreMailboxes& boxes) const
        {
            if (mail.statements.empty() && mail.implications.empty() && mail.exprOriginMap.empty()) return;

            const unsigned cores = logicalCores;
            assert((0 <= coreId) && (coreId < logicalCores));
            const unsigned slot = coreId;

            auto pit = index.find(const_cast<BodyOfProves*>(&sender));
            if (pit == index.end()) return;
            const std::vector<BodyOfProves*>& children = pit->second;

            for (BodyOfProves* child : children) {
                if (!child) continue;

                auto it = boxes.find(child);
                if (it == boxes.end()) continue;                 // must be preallocated
                std::vector<Mail>& slots = it->second;
                if (slots.size() != cores) continue;             // hard guard

                Mail& dst = slots[slot];
                dst.statements.insert(mail.statements.begin(), mail.statements.end());
                dst.implications.insert(mail.implications.begin(), mail.implications.end());

                std::map<std::string, std::vector<std::string>> merged = mail.exprOriginMap; // rhs wins
                for (const auto& kv : dst.exprOriginMap) merged[kv.first] = kv.second;
                dst.exprOriginMap.swap(merged);
            }
        }







        
    };

    












// Surface API mirroring Python (camelCase):




} // namespace gl
