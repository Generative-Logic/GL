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

#include "memory.hpp"

namespace gl {

    struct ExpressionAnalyzer {
        ProverParameters parameters;
        Memory body;
        // Root for counter-example filtering memory
        Memory ceBody;
        std::vector< std::tuple<int, bool, int>> updateGlobalTuples;
        std::vector< std::tuple<std::string, int>> updateGlobalDirectTuples;
        std::vector< std::tuple<std::string, std::string, std::string, std::string >> globalTheoremList;
        std::vector<Memory*> inductionMemoryBlocks;

        std::map<int, std::vector<std::vector<int> > > allBinariesAna;
        std::map<std::pair<int, int>, std::vector<std::vector<int> > > allMappingsAna;
        int maxNumLeafsPerKey;
        std::map<int, std::vector<std::vector<int> > > allPermutationsAna;
        Dependencies globalDependencies;
		std::vector<ContradictionItem> contradictionTable;
		bool ceFilteringActive = false;

		std::map<std::string, ce::CoreExpressionConfig> coreExpressionMap; // Load and resolve core expressions
        ce::AnchorInfo anchorInfo;
        std::string anchorID_;

        std::vector<Memory*> permanentBodies;
        std::vector<Memory*> permanentBodiesCE;
        unsigned logicalCores;
        ParentChildrenMap index;
        PerCoreMailboxes boxes;
        ParentChildrenMap indexCE;
        PerCoreMailboxes boxesCE;

        int implCounter;
        int existenceCounter;
        int statementCounter;
        int variableCounter;

        

        std::set<std::string> operators;
        std::set<std::string> expressionsFromConfig;

        

        mutable std::mutex dependenciesMutex;  // protects globalDependencies
        mutable std::mutex theoremListMutex;   // protects globalTheoremList
        mutable std::mutex inductionMemoryBlocksMutex;   // protects globalTheoremList
        mutable std::mutex updateGlobalMutex;
        mutable std::mutex updateGlobalDirectMutex;


        ExpressionAnalyzer(std::string anchorID);

        std::vector<std::vector<std::string>> readSimpleFacts() const;

        std::string smoothenExpr(const std::string& expr);
        std::vector<std::string> groomSubexpr(const std::string& s, int side, std::size_t& index);
        std::vector<std::string> groomExpr(const std::string& treeStr);
        std::vector<std::string> listLastRemovedArgs(const std::string& expr);
        std::tuple<std::string, int, std::string>
            renameLastRemoved(const std::string& expr, int startInt);
        std::string expandExpr(const std::string& expr);
        std::vector<std::string> getGlobalKey(const Memory& memoryBlock);

        bool checkNoUArguments(const std::vector<std::string>& key, const std::string& value);

        void addToHashMemory(const std::vector<std::string>& key,
            const std::string& value,
            const std::set<std::string>& remainingArgs,
            Memory& mb,
            HashMemory& targetIntMemory,
            const std::set<int>& levels,
            const std::string& originalImplication,
            int maxAdmissionDepth,
            int maxSecondaryNumber,
            bool partOfRecursion,
            int minNumOperatorsKey,
            const std::string& justification,
            bool performAdmissionMapUpdate,
            const std::string& originalImplicationClean,
            const std::string& validityName = "main");

        void makeNormalizedKeysForAdmission(
            const std::vector<std::string>& key,
            HashMemory& intHashMemory,
            NameMap& nameMap,
            KeyArena& arena,
            const std::string& value,
            int minNumOperatorsKey);

        std::vector<std::vector<EncodedExpression>>
            makeMandatoryEncodedStatementLists1(const HashMemory& intMemory,
                const std::vector<EncodedExpression>& localStatements, NameMap& nm);

        std::vector<std::vector<EncodedExpression>>
            makeMandatoryEncodedStatementLists2(Memory& body,
                const HashMemory& intMemory,
                const std::vector<EncodedExpression>& firstLayer,
                const std::vector<EncodedExpression>& secondLayer);

        void checkLocalEncodedMemory(const std::vector<EncodedExpression>& expressionList,
            Memory& memoryBlock,
            int iteration,
            const IntNormalizedKey& tple,
            unsigned coreId);

        /// Static checkLocal: int pre-hit checks, string post-hit via nm.decode().
        void checkLocalEncodedMemoryStatic(
            const StaticRequest& req,
            Memory& memoryBlock,
            unsigned coreId);

        // --- Static filter+mandatory using IntEncodedExpr arrays ---
        int16_t makeMandatoryEncodedStatementLists1Static(
            const HashMemory& mem,
            const IntEncodedExpr* stmts, int16_t stmtCount,
            int16_t* outIndices, int16_t maxOut);

        int16_t makeMandatoryEncodedStatementLists2Static(
            Memory& body, const HashMemory& mem,
            const IntEncodedExpr* first, int16_t firstCount,
            const IntEncodedExpr* second, int16_t secondCount,
            MandatoryPair* outPairs, int16_t maxOut);

        // --- Static request generation (singles, sizeOne=1) ---
        int16_t generateEncodedRequestsStatic(
                Memory& body,
                const HashMemory& intMemory,
                const int16_t* mandatoryIndices, int16_t mandatoryCount,
                const IntEncodedExpr* mandatorySrcInt,
                TypedArena<IntEncodedExpr>& exprArena,
                StaticRequest* outBuf, int16_t maxOut);

        // --- Static request generation (pairs, sizeOne=2) ---
        int16_t generateEncodedRequestsStaticPairs(
                Memory& body,
                const HashMemory& intMemory,
                const MandatoryPair* pairs, int16_t pairCount,
                const IntEncodedExpr* firstSrcInt,
                const IntEncodedExpr* secondSrcInt,
                TypedArena<IntEncodedExpr>& exprArena,
                StaticRequest* outBuf, int16_t maxOut);

        // --- Static request generation for CE mode (no mandatory) ---
        int16_t generateEncodedRequestsStaticCE(
                Memory& body,
                const HashMemory& intMemory,
                TypedArena<IntEncodedExpr>& exprArena,
                StaticRequest* outBuf, int16_t maxOut);

        // --- Common grow loop for Singles/Pairs (collects base candidates) ---
        void growBaseCandidates(
                Memory& body,
                const HashMemory& intMemory,
                const IntEncodedExpr* allIntStmts,
                const int16_t* filteredIdx, int16_t nFiltered,
                int16_t mainValidityId,
                int targetLen,
                const std::unordered_set<IntNormalizedKey, IntNormalizedKeyHash>& targetSubkeys,
                std::vector<BaseCandidate>& baseCandidates);

        // Incubator: multiply an implication into copies where subsets of
        // (1)-typed bound variables are set equal (Bell partitions).
        std::vector<std::string> multiplyImplication(const std::string& implication);

        // Incubator: back-reformulate operator-equality pattern to clean operator form.
        // Returns true if theorem matches Anchor -> op[...,x,...] -> =[x,a] and fills
        // backReformulated with Anchor -> op[...,a,...].
        bool tryBackReformulateOperatorHead(
            const std::string& theorem,
            std::string& backReformulated);
        // ---- Ordering helpers (function pointers used by std::sort) ----
        static bool lessByName(const EncodedExpression& a, const EncodedExpression& b);
        static bool lessByOriginal(const EncodedExpression& a, const EncodedExpression& b);
        std::map<std::string, LogicalEntity> compiledExpressions;
        std::map<std::vector<std::string>,
                 std::tuple<std::map<std::string, std::string>, 
                            std::string, 
                            std::vector<std::string>>> repetitionExclusionMap;

        // ---- Tiny utilities ----
        std::vector<EncodedExpression> sortEncodedExpressionsByName(const std::vector<EncodedExpression>& in);
        std::vector<EncodedExpression> mergeInsertSortedEncoded(const std::vector<EncodedExpression>& listA,
            const std::vector<std::string>& valuesA,
            const std::vector<EncodedExpression>& listB,
            const std::vector<std::string>& valuesB);




        bool getValidityName(const std::vector<EncodedExpression>& req, std::string& validityName);






        // New kernels
        Memory& performElementaryLogicalStep(Memory& body,
            unsigned coreId,
            const ParentChildrenMap& index,
            PerCoreMailboxes& boxes);

        void reactToHypo(Memory& mb);

        bool checkForEquivalence(const std::string& expr,
            const std::string& validityName,
            Memory& memoryBlock);

        void updateWeakVariables(Memory& mb, const std::string& validityName, const std::string& eqlty);

        // Declaration only (no body). Matches the .cpp definition.
        // 0 - Local statement.
        // 1 - Proved statement.
        // 2 - Statement pending proof.
        // 3 - Non - local statement.
		// 4 - Do not disintegrate statement.
        void addExprToMemoryBlock(const std::string& expr,
            Memory& memoryBlock,
            int iteration,
            int status,
            const std::set<int>& involvedLevels,
            const std::pair<std::string, std::vector<ExpressionWithValidity>>& origin,
            int coreId,
            int auxyIndex,
            std::string validityName,
            bool doNotDisintegrate);


        // Reformulates a theorem by moving set definitions to the end and 
        // applying a specific negation transformation to the innermost one.
        bool reformulateTheorem(const std::string& theorem, std::vector<std::pair<std::string, std::string>>& outTheorems);

        void addEquality(const std::string& expr,
            Memory& memoryBlock,
            bool local,
            const std::set<int>& levels,
            const std::pair<std::string, std::vector<ExpressionWithValidity>>& origin,
            const std::string& validityName);


        void checkNecessityForEquality(const std::string& inputExprStr, Memory& mb, std::string validityName);

        void addExprToMemoryBlockKernel(const std::string& expr,
            Memory& memoryBlock,
            int status,
            const std::set<int>& involvedLevels,
            const std::pair<std::string, std::vector<ExpressionWithValidity>>& origin,
            const std::string& validityName,
            int coreId,
            int iteration);

        bool allowedForMail(const std::string& expression, Memory& body);

        std::set<std::string> extractRemainingArgs(const std::string& str);


        void handleAnchor(const std::string& expr, Memory& memoryBlock, const std::set<int>& involvedLevels);



        std::string replaceUSubstrings(const std::string& s);

        // … earlier declarations …

        void activateZeroCondition(Memory& memoryBlock);

        // Parallel kernel
        void proveKernel(const std::vector<Memory*>& bodies,
            const ParentChildrenMap& index,
            PerCoreMailboxes& boxes);

        void overwriteOrigins(std::map<ExpressionWithValidity, std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>>& left,
            const std::map<ExpressionWithValidity, std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>>& right,
            int maxOrigins);

        // Keep ONLY this prove()
        void prove(int numberIterations,
            const std::vector<Memory*>& bodies,
            const ParentChildrenMap& index,
            PerCoreMailboxes& boxes);

        // … no other prove() overloads …


        std::set<std::string>
            findImmutableArgs(const std::string& theorem,
                const std::string& digit);

        Memory* accessMemory(const std::vector<std::string>& theoremKey,
            Memory& bodyOfProves1);
        void updateGlobalDirect(const std::string& theorem, int coreId);

        void updateGlobal(int auxyIndex, bool allLevelsInvolved, int coreId);

        void deactivateRecursively();

        // Deactivate all aux blocks created for a proven theorem
        void deactivateUnnecessary(const std::string& theorem, Memory& root);



        inline void updateAdmissionMap3(const std::string& expr,
            Memory& memoryBlock,
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
            Memory& memory,
            int iteration,
            bool proved,
            Dependencies& dependencyTable);

        void revisitRejected2(const std::string& markedExpr,
            Memory& memoryBlock,
            std::string validityName);



        void saveFilteredConjectures(const std::vector<std::string>& lines);

        void broadcastTheorems(const std::vector<std::string>& provedTheorems,
                               const std::string& originTag = "broadcast");

        void prefillIntegrationMapsRecursive(Memory* mb);

        void disintegrateExprHypothetically(const std::string& expr, Memory& memoryBlock, std::string validityName);

        void prehandleAnchor(Memory* mb);


        void analyzeExpressions(const std::vector<std::string>& theorems,
                                const std::vector<std::string>& provedTheorems,
                                const std::vector<std::string>& externalTheorems = {});

        void updateAdmissionMapRecursion(const std::string& expression,
            Memory& mb,
            const std::string& validityName);

        void buildStack(Memory& memoryBlock,
                const ExpressionWithValidity& proved,
                std::vector<std::vector<std::string>>& stack,
                std::set<ExpressionWithValidity>& covered);

        // Sort strings by corresponding values (descending). Throws std::invalid_argument if sizes differ.
        std::vector<ExpressionWithValidity> sortByValuesDesc(const std::vector<ExpressionWithValidity>& expressions,
            const std::vector<int>& values);

        // Debug: find ends, write indexed stacks + mapping file (like Python find_ends)
        void findEnds(const std::vector<std::string>& path, const std::filesystem::path& outDirParam);

        // In class ExpressionAnalyzer:
        void exportCompiledExpressionsJSON(const std::filesystem::path& outDir);


        void generateRawProofGraph(
            const std::vector<std::tuple<std::string, std::string, std::string, std::string>>& theoremList,
            const std::filesystem::path& outDir = std::filesystem::path("files") / "raw_proof_graph");



        // Build CE blocks for one conjecture: two consecutive LBs under ceBody.
        // LB0 gets +,*,s. LB1 gets inequalities and stores the conjecture via addToHashMemory.


        void loadFactsForCEFiltering(
            std::vector<std::string> simpleFacts,
            int batchSize);

        void addConjectureForCEFiltering(const std::string& conjecture,
            Memory *mb,
            int cIndex);

        // Batch teardown (deletes all CE nodes + shrinks containers + trims heap)
        void releaseCEBatchMemory();

        // Helpers to free/shrink heavy containers
        void destroyParentChildrenMap(ParentChildrenMap& m);
        void destroyMailboxes(PerCoreMailboxes& boxes);





        // Like analyzeExpressions, but for CE filtering



        std::vector<std::string> filterConjecturesWithCE(
            const std::vector<std::string>& conjectures,
            const std::vector<std::string>& simpleFacts);


        std::string negate(std::string expr);



        // Build parent->children index for the given roots (walks their reachable subgraph).
        ParentChildrenMap buildParentChildrenMap(const std::vector<Memory*>& roots) const;

        // Allocate per-core mailboxes sized to 'logicalCores'.
        PerCoreMailboxes buildPerCoreMailboxes(const ParentChildrenMap& index) const;

        // Compress per-core mailboxes into each body's mailIn in parallel.
        void smashMail(PerCoreMailboxes& boxes) const;
        
        // Appends a newly proved theorem to the persistent file storage.
        void saveProvedTheorems();

        // NEW: Saves the surviving theorems after the Compressor is finished.
        void saveCompressedTheorems(const std::vector<std::string>& compressedTheorems);

        // Saves only the essential (compressor-surviving) theorems to proved_theorems.txt.
        // This is the sole cross-batch propagation path: only essential theorems
        // are inherited by subsequent batches.
        void saveProvedTheoremsFiltered(const std::vector<std::string>& essentialTheorems,
                                       const std::unordered_set<std::string>& externalTheorems = {});


        // Helper to append origins safely up to max limit
        inline void addOrigin(std::map<ExpressionWithValidity, std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>>& map,
            const ExpressionWithValidity& ev,
            const std::pair<std::string, std::vector<ExpressionWithValidity>>& origin,
            int maxOrigins) const {
            auto& vec = map[ev];
            if (vec.size() < static_cast<size_t>(maxOrigins)) {
                if (std::find(vec.begin(), vec.end(), origin) == vec.end()) vec.push_back(origin);
            }
        }

        
//#pragma optimize("", off)

        inline std::pair<bool, IntNormalizedKey>
            preEvaluateEncodedKey(const std::vector<EncodedExpression>& key,
                Memory& body, const std::unordered_set<IntNormalizedKey, IntNormalizedKeyHash>& keySet) {

            // --- Validity Consistency Check ---
            std::string hypoValidity = "";
            bool foundHypo = false;
            for (const auto& expr : key) {
                if (expr.validityName.find("_hypo_") != std::string::npos) {
                    if (foundHypo && hypoValidity != expr.validityName) {
                        return std::make_pair(false, IntNormalizedKey());
                    }
                    hypoValidity = expr.validityName;
                    foundHypo = true;
                }
            }
            if (foundHypo) {
                if (key.size() > parameters.maxLenHypoKey) {
                    return std::make_pair(false, IntNormalizedKey());
                }
                for (const auto& expr : key) {
                    if (expr.validityName == hypoValidity) continue;
                    if (expr.validityName == "main" && expr.name.rfind("Anchor", 0) == 0) continue;
                    return std::make_pair(false, IntNormalizedKey());
                }
            }

            // Quick reject: too many secondary variables
            int secondaryCounter = 0;
            for (std::size_t i = 0; i < key.size(); ++i) {
                secondaryCounter += countPatternOccurrencesEncoded(key[i], body.overallHashMemory);
            }
            if (secondaryCounter > parameters.maxNumberSecondaryVariables) {
                return std::make_pair(false, IntNormalizedKey());
            }

            // Quick reject by length vs. admitted max
            if (static_cast<int>(key.size()) > body.overallHashMemory.maxKeyLength) {
                return std::make_pair(false, IntNormalizedKey());
            }

            // Build IntNormalizedKey on stack buffer — probe without arena allocation
            int16_t buf[ExecutionParameters::MAX_KEY_SLOTS];
            NameMap& nm = body.nameMap;
            int16_t len = makeIntNormalizedKey(key, nm, false, buf, ExecutionParameters::MAX_KEY_SLOTS);

            // Probe with stack-local key (no arena cost for rejects)
            IntNormalizedKey probe(static_cast<int16_t>(key.size()), buf, len);
            if (keySet.find(probe) == keySet.end()) {
                return std::make_pair(false, IntNormalizedKey());
            }

            // Accept: persist into arena only now
            const int16_t* p = body.keyArena.store(buf, len);
            IntNormalizedKey normalized(static_cast<int16_t>(key.size()), p, len);
            return std::make_pair(true, normalized);
        }

        // =================================================================
        // Static pipeline helpers (zero string operations in hot path)
        // =================================================================

        /// Build IntNormalizedKey from pre-encoded IntEncodedExpr pointer array.
        /// No nm.encode() calls — reads int16_t fields directly.
        /// Always uses ignoreU=false semantics: all args normalized, changeable=0.
        /// Unchangeable args are distinguished by their u_-prefixed argId.
        inline int16_t makeIntNormalizedKeyFromEncoded(
            const IntEncodedExpr* const* exprs, int16_t count,
            int16_t* outBuf, int16_t bufCapacity) {

            int16_t varIds[ExecutionParameters::MAX_KEY_SLOTS];
            int16_t normIds[ExecutionParameters::MAX_KEY_SLOTS];
            int16_t nVars = 0;
            int16_t nextNormId = 1;
            int16_t pos = 0;

            for (int16_t i = 0; i < count; ++i) {
                const IntEncodedExpr& e = *exprs[i];
                assert(pos + 2 <= bufCapacity);
                outBuf[pos++] = e.nameId;
                outBuf[pos++] = e.negation;

                for (int16_t j = 0; j < e.arity; ++j) {
                    assert(pos + 2 <= bufCapacity);
                    // ignoreU=false: normalize ALL args (u_-prefixed argId distinguishes unchangeable)
                    int16_t varId = e.argId[j];
                    int16_t normId = 0;
                    for (int16_t v = 0; v < nVars; ++v) {
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
            return pos;
        }

        /// Build IntNormalizedKey from IntEncodedExpr with ignoreU=true semantics
        /// and a set of arg IDs to additionally mark as unchangeable.
        /// Produces reverseMap[normId] = argFullId for back-substitution on hit.
        /// unchangeableArgIds = remaining-args candidate set (already int16_t encoded).
        inline int16_t makeIntNormalizedKeyFromEncodedWithMap(
            const IntEncodedExpr* const* exprs, int16_t count,
            const std::set<int16_t>& unchangeableArgIds,
            int16_t* outBuf, int16_t bufCapacity,
            int16_t* reverseMap, int16_t& numNormVars) {

            int16_t varIds[ExecutionParameters::MAX_KEY_SLOTS];
            int16_t normIds[ExecutionParameters::MAX_KEY_SLOTS];
            int16_t nVars = 0;
            int16_t nextNormId = 1;
            int16_t pos = 0;
            numNormVars = 0;

            for (int16_t i = 0; i < count; ++i) {
                const IntEncodedExpr& e = *exprs[i];
                assert(pos + 2 <= bufCapacity);
                outBuf[pos++] = e.nameId;
                outBuf[pos++] = e.negation;

                for (int16_t j = 0; j < e.arity; ++j) {
                    assert(pos + 2 <= bufCapacity);
                    // ignoreU=true: use argFullId (raw, not u_-prefixed) for ALL args
                    int16_t rawId = e.argFullId[j];
                    bool isUnch = (e.argUnchangeable[j] != 0) ||
                                  (unchangeableArgIds.find(rawId) != unchangeableArgIds.end());

                    if (isUnch) {
                        // Unchangeable: write raw ID as-is, changeable=1
                        outBuf[pos++] = rawId;
                        outBuf[pos++] = 1;
                    } else {
                        // Changeable: normalize to sequential ID
                        int16_t normId = 0;
                        for (int16_t v = 0; v < nVars; ++v) {
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

        /// Count secondary variables from IntEncodedExpr using int16_t set.
        inline int countPatternOccurrencesFromEncoded(
            const IntEncodedExpr& expr,
            const std::unordered_set<int16_t>& prodRecIds) {
            int counter = 0;
            for (int16_t i = 0; i < expr.arity; ++i) {
                if (expr.argIteration[i] > -1) {
                    if (prodRecIds.find(expr.argFullId[i]) == prodRecIds.end()) {
                        ++counter;
                    }
                }
            }
            return counter;
        }

        /// Full pre-evaluate from IntEncodedExpr pointer array. Zero string ops.
        /// mainValidityId = nm.encode("main"), cached by caller.
        inline std::pair<bool, IntNormalizedKey>
            preEvaluateFromEncoded(const IntEncodedExpr* const* exprs, int16_t count,
                Memory& body, int16_t mainValidityId,
                const std::unordered_set<IntNormalizedKey, IntNormalizedKeyHash>& keySet) {

            // Hypo validity check
            int16_t hypoValidityId = -1;
            bool foundHypo = false;
            for (int16_t i = 0; i < count; ++i) {
                if (exprs[i]->isHypo) {
                    if (foundHypo && exprs[i]->validityId != hypoValidityId) {
                        return std::make_pair(false, IntNormalizedKey());
                    }
                    hypoValidityId = exprs[i]->validityId;
                    foundHypo = true;
                }
            }
            if (foundHypo) {
                if (count > parameters.maxLenHypoKey) {
                    return std::make_pair(false, IntNormalizedKey());
                }
                for (int16_t i = 0; i < count; ++i) {
                    if (exprs[i]->validityId == hypoValidityId) continue;
                    if (exprs[i]->validityId == mainValidityId && exprs[i]->isAnchor) continue;
                    return std::make_pair(false, IntNormalizedKey());
                }
            }

            // Secondary variable count
            int secondaryCounter = 0;
            const auto& prodRecIds = body.overallHashMemory.productsOfRecursionIds;
            for (int16_t i = 0; i < count; ++i) {
                secondaryCounter += countPatternOccurrencesFromEncoded(*exprs[i], prodRecIds);
            }
            if (secondaryCounter > parameters.maxNumberSecondaryVariables) {
                return std::make_pair(false, IntNormalizedKey());
            }

            // Length check
            if (count > body.overallHashMemory.maxKeyLength) {
                return std::make_pair(false, IntNormalizedKey());
            }

            // Build IntNormalizedKey on stack
            int16_t buf[ExecutionParameters::MAX_KEY_SLOTS];
            int16_t len = makeIntNormalizedKeyFromEncoded(exprs, count, buf,
                ExecutionParameters::MAX_KEY_SLOTS);

            IntNormalizedKey probe(count, buf, len);
            if (keySet.find(probe) == keySet.end()) {
                return std::make_pair(false, IntNormalizedKey());
            }

            const int16_t* p = body.keyArena.store(buf, len);
            IntNormalizedKey normalized(count, p, len);
            return std::make_pair(true, normalized);
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

        /// Hot path: builds int16_t normalized key into caller's stack buffer.
        /// Returns length written. No heap allocation.
        /// ignoreU: if true, unchangeable vars get their NameMap ID with changeableStatus=1.
        ///          if false, all vars are normalized sequentially with changeableStatus=0.
        inline int16_t makeIntNormalizedKey(
            const std::vector<EncodedExpression>& lst,
            NameMap& nm,
            bool ignoreU,
            int16_t* outBuf,
            int16_t bufCapacity) {

            // Small linear map instead of 8KB memset: keys have few distinct vars
            int16_t varIds[ExecutionParameters::MAX_KEY_SLOTS];
            int16_t normIds[ExecutionParameters::MAX_KEY_SLOTS];
            int16_t nVars = 0;
            int16_t nextNormId = 1;
            int16_t pos = 0;

            for (std::size_t i = 0; i < lst.size(); ++i) {
                const EncodedExpression& expr = lst[i];

                // expressionID
                assert(pos + 2 <= bufCapacity);
                outBuf[pos++] = nm.encode(expr.name);
                // negation
                outBuf[pos++] = expr.negation ? 1 : 0;

                const std::vector<std::vector<std::string>>& args = expr.arguments;
                for (std::size_t j = 0; j < args.size(); ++j) {
                    const std::vector<std::string>& arg = args[j];
                    if (arg.size() < 2) continue;

                    assert(pos + 2 <= bufCapacity);
                    bool isUnchangeable = (arg[0] == "True");
                    int16_t varId = (!ignoreU && isUnchangeable)
                        ? nm.encode("u_" + arg[1])
                        : nm.encode(arg[1]);
                    assert(varId < ExecutionParameters::MAX_NAME_IDS);

                    if (ignoreU && isUnchangeable) {
                        outBuf[pos++] = varId;
                        outBuf[pos++] = 1;
                    }
                    else {
                        // Linear scan for normalization (nVars typically < 20)
                        int16_t normId = 0;
                        for (int16_t v = 0; v < nVars; ++v) {
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

        /// Hot path variant: also produces a reverse map (normId -> original varId)
        /// for back-substitution when decoding heads.
        inline int16_t makeIntNormalizedKeyWithMap(
            const std::vector<EncodedExpression>& lst,
            NameMap& nm,
            bool ignoreU,
            int16_t* outBuf,
            int16_t bufCapacity,
            int16_t* reverseMap,     // reverseMap[normId] = original varId
            int16_t& numNormVars) {

            // Small linear map instead of 8KB memset
            int16_t varIdsL[ExecutionParameters::MAX_KEY_SLOTS];
            int16_t normIdsL[ExecutionParameters::MAX_KEY_SLOTS];
            int16_t nVars = 0;
            int16_t nextNormId = 1;
            int16_t pos = 0;

            for (std::size_t i = 0; i < lst.size(); ++i) {
                const EncodedExpression& expr = lst[i];

                assert(pos + 2 <= bufCapacity);
                outBuf[pos++] = nm.encode(expr.name);
                outBuf[pos++] = expr.negation ? 1 : 0;

                const std::vector<std::vector<std::string>>& args = expr.arguments;
                for (std::size_t j = 0; j < args.size(); ++j) {
                    const std::vector<std::string>& arg = args[j];
                    if (arg.size() < 2) continue;

                    assert(pos + 2 <= bufCapacity);
                    bool isUnchangeable = (arg[0] == "True");
                    int16_t varId = (!ignoreU && isUnchangeable)
                        ? nm.encode("u_" + arg[1])
                        : nm.encode(arg[1]);
                    assert(varId < ExecutionParameters::MAX_NAME_IDS);

                    if (ignoreU && isUnchangeable) {
                        outBuf[pos++] = varId;
                        outBuf[pos++] = 1;
                    }
                    else {
                        int16_t normId = 0;
                        for (int16_t v = 0; v < nVars; ++v) {
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
            numNormVars = static_cast<int16_t>(nextNormId - 1);
            return pos;
        }

        /// Build IntNormalizedKey from pre-encoded IntEncodedExpr array.
        /// No string ops, no NameMap::encode calls — all IDs already resolved.
        inline int16_t makeIntNormalizedKeyFromEncoded(
            const IntEncodedExpr* exprs, int16_t count,
            bool ignoreU,
            int16_t* outBuf, int16_t bufCapacity) {

            int16_t varIds[ExecutionParameters::MAX_KEY_SLOTS];
            int16_t normIds[ExecutionParameters::MAX_KEY_SLOTS];
            int16_t nVars = 0;
            int16_t nextNormId = 1;
            int16_t pos = 0;

            for (int16_t ei = 0; ei < count; ++ei) {
                const IntEncodedExpr& expr = exprs[ei];
                assert(pos + 2 <= bufCapacity);
                outBuf[pos++] = expr.nameId;
                outBuf[pos++] = expr.negation;

                for (int16_t j = 0; j < expr.arity; ++j) {
                    assert(pos + 2 <= bufCapacity);
                    bool isUnchangeable = (expr.argUnchangeable[j] != 0);
                    // argId has u_-prefixed ID for unchangeable args (for ignoreU=false normalization)
                    // argFullId has raw ID (for ignoreU=true literal passthrough)
                    int16_t varId = (ignoreU && isUnchangeable) ? expr.argFullId[j] : expr.argId[j];

                    if (ignoreU && isUnchangeable) {
                        outBuf[pos++] = varId;
                        outBuf[pos++] = 1;
                    }
                    else {
                        int16_t normId = 0;
                        for (int16_t v = 0; v < nVars; ++v) {
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

        /// WithMap variant: also produces reverseMap[normId] = original varId.
        inline int16_t makeIntNormalizedKeyFromEncodedWithMap(
            const IntEncodedExpr* exprs, int16_t count,
            bool ignoreU,
            int16_t* outBuf, int16_t bufCapacity,
            int16_t* reverseMap, int16_t& numNormVars) {

            int16_t varIdsL[ExecutionParameters::MAX_KEY_SLOTS];
            int16_t normIdsL[ExecutionParameters::MAX_KEY_SLOTS];
            int16_t nVars = 0;
            int16_t nextNormId = 1;
            int16_t pos = 0;

            for (int16_t ei = 0; ei < count; ++ei) {
                const IntEncodedExpr& expr = exprs[ei];
                assert(pos + 2 <= bufCapacity);
                outBuf[pos++] = expr.nameId;
                outBuf[pos++] = expr.negation;

                for (int16_t j = 0; j < expr.arity; ++j) {
                    assert(pos + 2 <= bufCapacity);
                    bool isUnchangeable = (expr.argUnchangeable[j] != 0);
                    int16_t varId = (ignoreU && isUnchangeable) ? expr.argFullId[j] : expr.argId[j];

                    if (ignoreU && isUnchangeable) {
                        outBuf[pos++] = varId;
                        outBuf[pos++] = 1;
                    }
                    else {
                        int16_t normId = 0;
                        for (int16_t v = 0; v < nVars; ++v) {
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
            numNormVars = static_cast<int16_t>(nextNormId - 1);
            return pos;
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


        inline std::pair<NormalizedKey, std::map<std::string, std::string>>
            makeNormalizedEncodedKey(const std::vector<EncodedExpression>& lst, int uStatus = 1) {
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
            std::set<std::string> placed;
            for (std::size_t idx = 0; idx < argsChain.size(); ++idx) {
                const std::vector<std::string>& args = argsChain[idx];
                for (std::size_t j = 0; j < args.size(); ++j) {
                    const std::string& arg = args[j];
                    if (removedArgs.count(arg) && placed.find(arg) == placed.end()) {
                        placed.insert(arg);
                        if (idx < key.size()) whenRemoved[idx].push_back(arg);
                    }
                }
            }

            for (std::size_t i = key.size(); i-- > 0; ) {
                std::vector<std::string>& lst = whenRemoved[i];

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

        // Like reconstructImplication, but binds ALL non-u_ variables in >[...].
        // Used exclusively during disintegration, where every non-anchor variable
        // in the implication elements must be universally quantified.
        inline std::string reconstructImplicationFullBind(const std::vector<std::string>& key,
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

            for (std::size_t i = 0; i < argsChain.size(); ++i) {
                const std::vector<std::string>& args = argsChain[i];
                for (std::size_t j = 0; j < args.size(); ++j) {
                    const std::string& arg = args[j];
                    if (arg.size() >= 2 && arg[0] == 'u' && arg[1] == '_') continue;
                    removedArgs.insert(arg);
                }
            }

            std::vector<std::vector<std::string> > whenRemoved(key.size());
            std::set<std::string> placed;
            for (std::size_t idx = 0; idx < argsChain.size(); ++idx) {
                const std::vector<std::string>& args = argsChain[idx];
                for (std::size_t j = 0; j < args.size(); ++j) {
                    const std::string& arg = args[j];
                    if (removedArgs.count(arg) && placed.find(arg) == placed.end()) {
                        placed.insert(arg);
                        if (idx < key.size()) whenRemoved[idx].push_back(arg);
                    }
                }
            }

            for (std::size_t i = key.size(); i-- > 0; ) {
                std::vector<std::string>& lst = whenRemoved[i];

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

        // Wrapper around FullBind for integration.
        // If outermost >[...] has a pi_lev_ bound var (unoccupied): return as-is.
        // Otherwise (occupied bound var): remove the var from >[...], making it empty >[].
        // pi_ format: pi_lev_<digits>_<digits> (e.g. pi_lev_0_1)
        inline std::string reconstructImplicationForIntegration(const std::vector<std::string>& key,
            const std::string& value) {
            std::string result = reconstructImplicationFullBind(key, value);

            // Find outermost >[...] content
            if (result.size() > 3 && result[0] == '(' && result[1] == '>' && result[2] == '[') {
                std::size_t bracketEnd = result.find(']', 3);
                if (bracketEnd != std::string::npos && bracketEnd > 3) {
                    std::string bv = result.substr(3, bracketEnd - 3);

                    // Check if bound var matches pi_lev_ prefix
                    if (bv.size() >= 7 && bv[0] == 'p' && bv[1] == 'i' && bv[2] == '_'
                        && bv[3] == 'l' && bv[4] == 'e' && bv[5] == 'v' && bv[6] == '_') {
                        return result; // unoccupied pi_lev_ bound var — return unchanged
                    }

                    // Occupied bound var — remove from >[...] → >[]
                    result = "(>[]" + result.substr(bracketEnd + 1);
                }
            }

            return result;
        }

        inline std::vector<std::string> renamingChain(const std::vector<std::string>& chain, const std::set<std::string>& remainingArgs) {
            // 0. Early check: If chain has NO "u_" args, return it unchanged.
            bool hasU = false;
            for (const std::string& expr : chain) {
                std::vector<std::string> args = ce::getArgs(expr);
                for (const std::string& arg : args) {
                    if (startsWith(arg, "u_", 2)) {
                        hasU = true;
                        break;
                    }
                }
                if (hasU) break;
            }

            if (!hasU) {
                return chain;
            }

            // --- Proceed with renaming logic ---

            std::set<std::string> uArgsInChainStr;
            std::set<int> uArgsInChain;
            std::set<int> changeableArgs;

            // 1. Identify all arguments in the chain
            for (const std::string& expr : chain) {
                std::vector<std::string> args = ce::getArgs(expr);
                for (const std::string& arg : args) {
                    if (startsWith(arg, "u_", 2)) {
                        std::string val = arg.substr(2);
                        uArgsInChainStr.insert(val);
                        if (!val.empty() && std::all_of(val.begin(), val.end(),
                            [](unsigned char c) { return std::isdigit(c); })) {
                            uArgsInChain.insert(std::stoi(val));
                        }
                    }
                    else {
                        // Assert that the changeable argument is an integer
                        bool isInt = !arg.empty() && std::all_of(arg.begin(), arg.end(),
                            [](unsigned char c) { return std::isdigit(c); });
                        assert(isInt && "Changeable argument in chain must be a valid integer");

                        changeableArgs.insert(std::stoi(arg));
                    }
                }
            }

            // 2. Assert that remainingArgs is a SUBSET of uArgsInChain
            bool isSubset = std::includes(uArgsInChainStr.begin(), uArgsInChainStr.end(),
                remainingArgs.begin(), remainingArgs.end());
            assert(isSubset && "remainingArgs must be a subset of the 'u_' arguments present in the chain");

            // 3. Check for overlap between changeable args and 'u_' args
            bool overlap = false;
            for (int cArg : changeableArgs) {
                if (uArgsInChain.count(cArg)) {
                    overlap = true;
                    break;
                }
            }

            // 4. Build Replacement Map
            std::map<std::string, std::string> replacementMap;

            // 4a. Always map u_X -> X
            for (const std::string& uArg : uArgsInChainStr) {
                replacementMap["u_" + uArg] = uArg;
            }

            // 4b. Handle changeable arguments only if there is an overlap
            if (overlap) {
                // Fallback to remapping logic: Assign new free integers
                std::set<int> blockedInts = uArgsInChain; // Block all IDs used by u_ args (since they map to X)
                int nextFree = 1;

                for (int originalVal : changeableArgs) {
                    // Find next free integer
                    while (blockedInts.count(nextFree)) {
                        nextFree++;
                    }
                    replacementMap[std::to_string(originalVal)] = std::to_string(nextFree);
                    blockedInts.insert(nextFree);
                }
            }
            // Else: If no overlap, changeable args keep their original names (not added to map)

            // 5. Apply renaming
            std::vector<std::string> renamedChain;
            renamedChain.reserve(chain.size());
            for (const std::string& expr : chain) {
                renamedChain.push_back(ce::replaceKeysInString(expr, replacementMap));
            }

            return renamedChain;
        }

        inline void updateAdmissionMap(Memory& mb,
            const std::vector<std::string>& key,
            const std::set<std::string>& remainingArgs,
            int maxAdmissionDepth,
            int maxSecondaryNumber,
            bool partOfRecursion,
            std::string validityName) {

            std::vector<std::string> renamedKey = renamingChain(key, remainingArgs);

            for (std::size_t i = 0; i < renamedKey.size(); ++i) {
                const std::string& element = renamedKey[i];
                const std::string coreExpr = ce::extractExpression(element);
                if (operators.find(coreExpr) == operators.end()) {
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

                    prepareIntegration(removed, remainingArgs, mb, validityName);

                    std::string core = ce::extractExpression(removed);
                    auto itComp = compiledExpressions.find(core);
                    assert(itComp != compiledExpressions.end());
                    if (itComp->second.category != "atomic")
                    {
                        mb.canBeSentMarkerSet.insert(removed);
                    }

                    AdmissionMapValue value(renamedKey,
                        remainingArgs,
                        maxAdmissionDepth,
                        maxSecondaryNumber,
                        partOfRecursion);

                    std::map<ExpressionWithValidity, std::set<AdmissionMapValue> >::iterator itAd = mb.overallHashMemory.admissionMap.find(ExpressionWithValidity(removed, validityName));
                    if (itAd != mb.overallHashMemory.admissionMap.end()) {
                        itAd->second.insert(value);

                        std::map<ExpressionWithValidity, bool>::iterator itSt = mb.overallHashMemory.admissionStatusMap.find(ExpressionWithValidity(removed, validityName));
                        if (itSt != mb.overallHashMemory.admissionStatusMap.end()) {
                            itSt->second = partOfRecursion || itSt->second;
                        }
                        else {
                            mb.overallHashMemory.admissionStatusMap[ExpressionWithValidity(removed, validityName)] = partOfRecursion;
                        }
                    }
                    else {
                        std::set<AdmissionMapValue> s;
                        s.insert(value);
                        mb.overallHashMemory.admissionMap.insert(std::make_pair(ExpressionWithValidity(removed, validityName), s));
                        mb.overallHashMemory.admissionStatusMap[ExpressionWithValidity(removed, validityName)] = partOfRecursion;
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

        inline std::set<std::string> getRemainingArgs(const std::vector<std::string>& key) {
            std::set<std::string> argsWithoutUPrefix;
            for (std::size_t i = 0; i < key.size(); ++i) {
                const std::vector<std::string> args = ce::getArgs(key[i]);
                for (std::size_t j = 0; j < args.size(); ++j) {
                    const std::string& arg = args[j];

                    // Only process arguments starting with "u_"
                    if (startsWith(arg, "u_", 2)) {

                        // Count occurrences of "u_" to ensure it only appears once (at the start)
                        int uCount = 0;
                        std::size_t pos = 0;
                        while ((pos = arg.find("u_", pos)) != std::string::npos) {
                            uCount++;
                            pos += 2;
                        }
                        assert(uCount == 1 && "Argument starting with 'u_' must not contain additional 'u_' substrings");

                        // Remove "u_" prefix and insert
                        argsWithoutUPrefix.insert(arg.substr(2));
                    }
                }
            }
            return argsWithoutUPrefix;
        }

        //#pragma optimize("", off)

        inline bool implicationIsQualified(const std::vector<std::string>& key, const std::string& value, int minNumOperatorsKey) {

            
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

            if (counterKey > minNumOperatorsKey && valueCond) {
                result = true;
            }

            if (key.size() >= parameters.minLenLongKey)
            {
				result = true;
            }

            return result;
        }






        inline int countPatternOccurrences(const std::string& inputString,
            const HashMemory& localMemory) {
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
            const HashMemory& localMemory) {
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
                const HashMemory& intMemory, NameMap& nm) {
            std::vector<EncodedExpression> filtered;

            int16_t buf[ExecutionParameters::MAX_KEY_SLOTS];

            for (std::size_t i = 0; i < encodedStatements.size(); ++i) {
                const EncodedExpression& sttment = encodedStatements[i];

                std::vector<EncodedExpression> single;
                single.push_back(sttment);
                int16_t len = makeIntNormalizedKey(single, nm, false, buf,
                    ExecutionParameters::MAX_KEY_SLOTS);
                IntNormalizedKey ik(1, buf, len);

                if (intMemory.normalizedEncodedSubkeys.find(ik) ==
                    intMemory.normalizedEncodedSubkeys.end()) {
                    continue;
                }

                const int filterStatementsItr = sttment.maxIterationNumber;
                if (filterStatementsItr > parameters.maxIterationNumberVariable) {
                    continue;
                }

                filtered.push_back(sttment);
            }

            return filtered;
        }

        /// Static filter: works on IntEncodedExpr*, returns passing indices.
        /// No heap allocation. outIndices must have room for count entries.
        inline int16_t filterIntEncodedStatements(
            const IntEncodedExpr* stmts, int16_t count,
            const HashMemory& mem,
            int16_t* outIndices, int16_t maxOut) {

            int16_t buf[ExecutionParameters::MAX_KEY_SLOTS];
            int16_t nOut = 0;

            for (int16_t i = 0; i < count && nOut < maxOut; ++i) {
                const IntEncodedExpr& s = stmts[i];

                // Build single-expr IntNormalizedKey on stack
                int16_t pos = 0;
                buf[pos++] = s.nameId;
                buf[pos++] = s.negation;
                for (int16_t j = 0; j < s.arity; ++j) {
                    buf[pos++] = s.argId[j];
                    buf[pos++] = 0; // changeable (ignoreU=false)
                }
                // Normalize: sequential IDs by first appearance
                {
                    int16_t varMap[ExecutionParameters::MAX_KEY_SLOTS];
                    int16_t nV = 0;
                    int16_t nextN = 1;
                    // Start after nameId+negation (pos 2), step by 2 (varId, changeable)
                    for (int16_t p = 2; p < pos; p += 2) {
                        int16_t raw = buf[p];
                        int16_t norm = 0;
                        for (int16_t v = 0; v < nV; ++v) {
                            if (varMap[v * 2] == raw) { norm = varMap[v * 2 + 1]; break; }
                        }
                        if (norm == 0) {
                            norm = nextN++;
                            varMap[nV * 2] = raw;
                            varMap[nV * 2 + 1] = norm;
                            ++nV;
                        }
                        buf[p] = norm;
                    }
                }

                IntNormalizedKey ik(1, buf, pos);
                if (mem.normalizedEncodedSubkeys.find(ik) == mem.normalizedEncodedSubkeys.end())
                    continue;
                if (s.maxIteration > parameters.maxIterationNumberVariable)
                    continue;

                outIndices[nOut++] = i;
            }
            return nOut;
        }

        /// CE filter: accepts statements that appear in subkeys OR full keys.
        inline int16_t filterIntEncodedStatementsCE(
            const IntEncodedExpr* stmts, int16_t count,
            const HashMemory& mem,
            int16_t* outIndices, int16_t maxOut) {

            int16_t buf[ExecutionParameters::MAX_KEY_SLOTS];
            int16_t nOut = 0;

            for (int16_t i = 0; i < count && nOut < maxOut; ++i) {
                const IntEncodedExpr& s = stmts[i];

                // Build single-expr IntNormalizedKey on stack
                int16_t pos = 0;
                buf[pos++] = s.nameId;
                buf[pos++] = s.negation;
                for (int16_t j = 0; j < s.arity; ++j) {
                    buf[pos++] = s.argId[j];
                    buf[pos++] = 0; // changeable
                }
                // Normalize: sequential IDs by first appearance
                {
                    int16_t varMap[ExecutionParameters::MAX_KEY_SLOTS];
                    int16_t nV = 0;
                    int16_t nextN = 1;
                    for (int16_t p = 2; p < pos; p += 2) {
                        int16_t raw = buf[p];
                        int16_t norm = 0;
                        for (int16_t v = 0; v < nV; ++v) {
                            if (varMap[v * 2] == raw) { norm = varMap[v * 2 + 1]; break; }
                        }
                        if (norm == 0) {
                            norm = nextN++;
                            varMap[nV * 2] = raw;
                            varMap[nV * 2 + 1] = norm;
                            ++nV;
                        }
                        buf[p] = norm;
                    }
                }

                IntNormalizedKey ik(1, buf, pos);
                if (mem.normalizedEncodedSubkeys.find(ik) == mem.normalizedEncodedSubkeys.end()
                    && mem.normalizedEncodedKeys.find(ik) == mem.normalizedEncodedKeys.end())
                    continue;
                if (s.maxIteration > parameters.maxIterationNumberVariable)
                    continue;

                outIndices[nOut++] = i;
            }
            return nOut;
        }

        inline std::pair<std::string, std::string>
            extractValues(const std::string& s,
                const std::map<std::string, ce::CoreExpressionConfig>& coreExpressionMap) {
            ce::TreeNode1* root = ce::parseExpr(s);


            std::string left = ce::treeToExpr(root ? root->left : NULL);
            std::string right = ce::treeToExpr(root ? root->right : NULL);

            ce::deleteTree(root);
            return std::make_pair(left, right);
        }

        inline void updateRejectedMap(const std::string& renamedExpr,
            const std::string& markedExpr,
            const std::string& expr,
            HashMemory& hashMemory,
            int iteration,
            std::string validityName) {
            RejectedMapValue v(renamedExpr, expr, iteration);
            hashMemory.rejectedMap[ExpressionWithValidity(markedExpr, validityName)].insert(v);
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

        void removeExpressionFromMemoryBlock(const EncodedExpression& encExpr, Memory& mb, int state);



        void cleanUpIntegrationPreparation(const std::string& expression,
            Memory& memoryBlock);

        void cleanUpIntegrationPreparationCore(const Instruction& instructions,
            Memory& mb);

        inline bool isAdmittedIntegration(Memory& mb,
            const std::string& expr,
            const std::string& var,
            const std::string& markedExpr,
            const std::string validityName) {

            std::string core = ce::extractExpressionUniversal(markedExpr);
            auto it0 = this->compiledExpressions.find(core);
            assert(it0 != this->compiledExpressions.end());
            if (ce::extractExpressionUniversal(expr) == "in") {
                return false;
            }

            // 1. Get all arguments from the expression
            std::vector<std::string> args = ce::getArgs(markedExpr);
            std::map<std::string, std::string> replacementMap;

            // 2. Iterate and rename non-marker arguments
            for (const std::string& arg : args) {
                if (arg != "marker") {
                    if (startsWith(arg, "u_", 2)) {
                        std::cout << "Assertion violation in isAdmittedIntegration. Arg: " << arg << std::endl;
                        assert(false && "Argument in isAdmittedIntegration already starts with u_");
                    }
                    replacementMap[arg] = "u_" + arg;
                }
            }

            // 3. Apply the renaming to the marked expression
            std::string renamedMarkedExpr = ce::replaceKeysInString(markedExpr, replacementMap);


            // 4. Check if the marked expression exists in the integration map
            auto it = mb.overallHashMemory.admissionMapIntegration.find(ExpressionWithValidity(renamedMarkedExpr, validityName));
            if (it == mb.overallHashMemory.admissionMapIntegration.end()) {
                return false;
            }

            // 5. Prepare the replacement map ("marker" -> newVar)
            std::map<std::string, std::string> replacementMap2;
            assert(!startsWith(var, "u_", 2));
            std::string uVar = "u_" + var;
            replacementMap2["marker"] = uVar;

            // 6. Access the set (Reference to the set in the map)
            auto& entryMap = it->second;

            // [FIX] SNAPSHOT: Create a stable copy of the set to iterate over.
            // This prevents new templates added by prepareIntegrationCore2 from being processed in this loop.
            std::vector<std::pair<Instruction, std::set<std::string>>> snapshot(entryMap.begin(), entryMap.end());

            for (const auto& pair : snapshot) {
                const Instruction& instruction = pair.first;
                const std::set<std::string>& appliedVars = pair.second;

                // 1. Check if var is already present in this instruction's history
                if (appliedVars.count(var)) continue;

                // 2. Update the LIVE map (insert var into the set)
                // Using [instruction] is safe and efficient with the map structure
                entryMap[instruction].insert(var);

                // 3. Run logic using the instruction template
                Instruction instructionsCopy = instruction;
                cleanInstruction(instructionsCopy, renamedMarkedExpr);
                prepareIntegrationCore2(instructionsCopy, replacementMap2, mb, validityName, "");
            }

			cleanAdmissionMap(markedExpr, validityName, mb);

            // 8. Return true indicating the key was found and processed
            return true;
        }

        inline std::set<std::string>
            findDigitArgs(const std::string& theorem,
                const ce::AnchorInfo& anchor,
                const std::map<std::string, ce::CoreExpressionConfig>& coreExpressionMap)
        {
            // 1. Disintegrate Implication Chain
            std::vector< std::tuple<
                std::string,
                std::vector<std::string>,
                std::set<std::string>
            > > tempChain;

            const std::string head = ce::disintegrateImplication(theorem, tempChain, coreExpressionMap);

            std::vector<std::string> chain;
            chain.reserve(tempChain.size() + 1);
            for (const auto& t : tempChain) {
                chain.push_back(std::get<0>(t));
            }
            chain.push_back(head);

            for (const auto& element : chain) {
                // Allow equalities
				bool contains = element.find(anchorInfo.name) != std::string::npos;
                if (this->isEquality(element) || contains) {
                    continue;
                }

                // Check if it is a registered operator
                std::string coreExpr = ce::extractExpression(element);
                if (this->operators.find(coreExpr) == this->operators.end()) {
                    return {}; // Return empty set if invalid
                }
            }

            // 2. Collect ALL Input Arguments using Config Indices
            std::set<std::string> allInputArgs;
            for (const auto& element : chain) {
                std::string coreExpr = ce::extractExpression(element);

                auto it = coreExpressionMap.find(coreExpr);
                if (it != coreExpressionMap.end()) {
                    const auto& cfg = it->second;
                    // Only proceed if indices are defined (Python behavior implies existence)
                    if (!cfg.inputIndices.empty()) {
                        std::vector<std::string> args = ce::getArgs(element);
                        for (int idx : cfg.inputIndices) {
                            if (idx >= 0 && idx < static_cast<int>(args.size())) {
                                allInputArgs.insert(args[idx]);
                            }
                        }
                    }
                }
            }

            // 3. Remove Anchor Arguments
            for (const auto& element : chain) {
                // Check if element contains anchor name
                if (element.find(anchor.name) != std::string::npos) {
                    std::vector<std::string> args = ce::getArgs(element);
                    for (const auto& arg : args) {
                        allInputArgs.erase(arg);
                    }
                }
            }

            // 4. Collect ALL Output Arguments
            std::set<std::string> allOutputArgs;
            for (const auto& element : chain) {
                std::string coreExpr = ce::extractExpression(element);
                auto it = coreExpressionMap.find(coreExpr);
                if (it != coreExpressionMap.end()) {
                    const auto& cfg = it->second;
                    if (!cfg.outputIndices.empty()) {
                        std::vector<std::string> args = ce::getArgs(element);
                        for (int idx : cfg.outputIndices) {
                            if (idx >= 0 && idx < static_cast<int>(args.size())) {
                                allOutputArgs.insert(args[idx]);
                            }
                        }
                    }
                }
            }

            // 5. Subtract Outputs from Inputs
            for (const auto& outArg : allOutputArgs) {
                allInputArgs.erase(outArg);
            }

            return allInputArgs;
        }

        void cleanAdmissionMap(const std::string& markedExpr, const std::string& validity, Memory& mb) {           
            // 1. Identify the core expression
            std::string core = ce::extractExpression(markedExpr);

            // 2. Check if it is a registered operator
            if (this->operators.find(core) == this->operators.end()) {
                return;
            }

            // 3. Retrieve configuration to check indices
            auto itComp = this->coreExpressionMap.find(core);
            assert(itComp != this->coreExpressionMap.end());

            const auto& cfg = itComp->second;
            std::vector<std::string> mArgs = ce::getArgs(markedExpr);
            bool markerIsOutput = false;

            // 4. Check if "marker" appears in any output index position
            for (int outIdx : cfg.outputIndices) {
                if (outIdx >= 0 && outIdx < static_cast<int>(mArgs.size())) {
                    if (mArgs[outIdx] == "marker") {
                        markerIsOutput = true;
                        break;
                    }
                }
            }

            // 5. If "marker" is an output variable, perform cleanup
            if (markerIsOutput) {
                ExpressionWithValidity evKey(markedExpr, validity);

                // Mark as consumed so it isn't used again
                mb.overallHashMemory.consumedAdmissionKeys.insert(evKey);

                // Remove from active admission maps
                mb.overallHashMemory.admissionMap.erase(evKey);
                mb.overallHashMemory.admissionStatusMap.erase(evKey);
                mb.overallHashMemory.admissionMapIntegration.erase(evKey);
            }
        }


        inline bool isAdmitted(Memory& mb,
            const std::string& expr,
            const std::string& var,
            const std::string& markedExpr,
            std::string validityName) {
            bool result = false;

            const std::string coreExpr = ce::extractExpression(expr);

            // parts = expr.split(var)  with assert len(parts) == 2
            const std::size_t pos = expr.find(var);
            assert(pos != std::string::npos);
            const std::size_t pos2 = expr.find(var, pos + var.size());
            assert(pos2 == std::string::npos); // exactly one occurrence

            const std::string part0 = expr.substr(0, pos);
            const std::string part1 = expr.substr(pos + var.size());

            std::map<ExpressionWithValidity, std::set<AdmissionMapValue> >::iterator mit =
                mb.overallHashMemory.admissionMap.find(ExpressionWithValidity(markedExpr, validityName));
            if (mit == mb.overallHashMemory.admissionMap.end()) {
                return result;
            }

            assert(mb.overallHashMemory.consumedAdmissionKeys.find(ExpressionWithValidity(markedExpr, validityName)) == mb.overallHashMemory.consumedAdmissionKeys.end());

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
                const int cnt = countPatternOccurrences(expr, mb.overallHashMemory);

                if (mn <= maxAdmissionDepth && cnt <= maxSecondaryNumber) {
                    result = true;

                    std::map<ExpressionWithValidity, bool>::iterator sit =
                        mb.overallHashMemory.admissionStatusMap.find(ExpressionWithValidity(markedExpr, validityName));
                    assert(sit != mb.overallHashMemory.admissionStatusMap.end());
                    if (sit != mb.overallHashMemory.admissionStatusMap.end() && sit->second) {
                        mb.overallHashMemory.productsOfRecursion.insert(var);
                        mb.overallHashMemory.productsOfRecursionIds.insert(mb.nameMap.encode(var));
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

                            updateAdmissionMap(mb,
                                replKey,
                                newRemArgs,
                                maxAdmissionDepth,
                                maxSecondaryNumber,
                                partOfRecursion,
                                validityName);
                        }
                    }
                }
            }

			cleanAdmissionMap(markedExpr, validityName, mb);

            return result;
        }

        inline static bool startsWith(const std::string& s, const char* pfx, std::size_t n) {
            if (s.size() < n) return false;
            for (std::size_t i = 0; i < n; ++i) if (s[i] != pfx[i]) return false;
            return true;
        }

        

        std::vector<std::string> splitNormalizedKey(const std::vector<std::string> normalizedKey)
        {
            std::vector<std::string> result;

            int position = 0;
            while (true)
            {
                const LogicalEntity& le = compiledExpressions[normalizedKey[position]];
                std::string coreExpression = normalizedKey[position];
                position++;

                std::string expression = "(" + coreExpression + "[";
                bool isNegation = normalizedKey[position] == "True";
                position++;

                if (isNegation)
                {
                    expression = "!" + expression;
                }

                int arity = le.arity;
                for (int index = 0; index < arity; index++)
                {
                    expression += normalizedKey[position] + ",";
                    position++;
                }
                expression.pop_back();
                expression += "])";
                result.push_back(expression);

                if (position == normalizedKey.size())
                {
                    break;
                }

            }

            return result;
        }



        // Helper to verify a string is a valid integer (ignoring "u_" prefix if present)
        inline bool isIntegerString(const std::string& s) {
            if (s.empty()) return false;
            size_t start = 0;
            if (s.size() > 2 && s[0] == 'u' && s[1] == '_') {
                start = 2;
            }
            if (start >= s.size()) return false;

            for (size_t i = start; i < s.size(); ++i) {
                if (!std::isdigit(s[i])) return false;
            }
            return true;
        }


        /**
         * @brief Constructs a sorted, formatted string list of arguments from a mapping, stripping "u_" prefixes.
         *
         * Iterates through the provided map \p mp to find entries where the *value* indicates an "unchangeable"
         * argument (starts with "u_" followed by an integer, e.g., "u_1"). It sorts these entries based on
         * the integer value of the target "u_N" string.
         *
         * The output is a bracketed list string (e.g., "[arg1,arg2]") formed from the *keys* corresponding
         * to these values. Before adding a key to the list, the function asserts that the key itself starts
         * with "u_" and strips this prefix (e.g., "u_x" becomes "x").
         *
         * @param mp A map where values are checked for the "u_N" pattern to determine inclusion and order.
         * @return A string representation of the list of keys (stripped of "u_") sorted by their mapped "u_N" values.
         *
         * @note Asserts that any value starting with "u_" is followed by a valid integer.
         * @note Asserts that any key added to the output list starts with "u_" and contains exactly one "u_" substring.
         */
        inline std::string createArgList(const std::map<std::string, std::string>& mp) {
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
                assert(startsWith(currentKey, "u_", 2) && "Argument in output list must start with 'u_'");

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

        std::map<std::string, std::string> combineUnchangeablePaths(
            const std::map<std::string, std::string>& a,
            const std::map<std::string, std::string>& b)
        {
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

        /**
         * @brief Creates a reverse mapping for "unchangeable" identifiers based on two association maps.
         *
         * Iterates through map \p b to find values conforming to the "unchangeable" pattern
         * (starting with "u_" followed by an integer). For each such value (e.g., "u_1"),
         * the function creates a mapping in the result where the key is "u_1" and the value
         * is taken from map \p a using the same key found in \p b.
         *
         * Effectively performs: result[b[k]] = a[k]
         *
         * @param a Map containing the target values to be mapped to the unchangeable keys.
         * @param b Map defining the association between original keys and "u_N" identifiers.
         * @return A new map { "u_N" : value_from_a }.
         *
         * @note Asserts that any value in \p b starting with "u_" is followed by a valid integer.
         * @note Asserts that for every key in \p b mapping to a "u_" value, a corresponding key exists in \p a.
         */
        std::map<std::string, std::string> createReverseUnchangeableMap(
            const std::map<std::string, std::string>& a,
            const std::map<std::string, std::string>& b)
        {
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
                            // Start strict: if 'b' has a mapping for 'x', 'a'  should too..
                            assert(false && "Key found in 'b' was missing in 'a'");
                        }
                    }
                }
            }

            return c;
        }


        /**
         * @brief Normalizes a logical construction to avoid duplicate definitions for structurally identical expressions.
         *
         * Checks if the combination of \p encodedTail and \p encodedHead results in a logical structure
         * (normalized key) that has already been defined. If an isomorphic structure exists (accounting for
         * variable permutations), it returns a call to the existing core expression with mapped arguments.
         * Otherwise, it registers a new logical entity.
         *
         * @param encodedTail The list of expressions forming the body or antecedent of the construction.
         * @param encodedHead The list of expressions forming the head or consequent of the construction.
         * @param newCoreExpressionName The proposed name for the new core expression if creation is necessary (e.g., "implication5").
         * @param category The logical category of the expression (e.g., "implication", "and", "existence").
         * @param force If \c true, bypasses the canonical lookup and forces the registration of \p defaultSignature.
         * @param defaultSignature The explicit signature to enforce when \p force is \c true.
         * @return A string representing the normalized expression call, typically "(CoreName[args...])".
         */
        std::string excludeRepetitions(const std::vector<EncodedExpression>& encodedTail,
                                       const std::vector<EncodedExpression>& encodedHead,
                                       const std::string& newCoreExpressionName,
                                       const std::string& category,
                                       const bool force,
                                       const std::string defaultSignature)
        {
            std::string result;

			std::vector<EncodedExpression> encodedList = encodedTail;
			encodedList.insert(encodedList.end(), encodedHead.begin(), encodedHead.end());
            std::pair<NormalizedKey, std::map<std::string, std::string>> pr2 = makeNormalizedEncodedKey(encodedList, 2);
            std::vector<std::string> splitNK = splitNormalizedKey(pr2.first.data);

            if (force)
            {
                std::map<std::string, std::string> replacementMap;

                std::vector<std::string> argsDefaultSignature = ce::getArgs(defaultSignature);
                // Pass .begin() and .end() iterators
                assert(std::set<std::string>(argsDefaultSignature.begin(), argsDefaultSignature.end()).size() == argsDefaultSignature.size());

				for (int ind = 1; ind <= static_cast<int>(argsDefaultSignature.size()); ind++)
                {
                    std::string value = "u_" + std::to_string(ind);

                    assert(!startsWith(argsDefaultSignature[static_cast<std::size_t>(ind) - 1], "u_", 2));
                    std::string key = "u_" + argsDefaultSignature[static_cast<std::size_t>(ind) - 1];
                    replacementMap[key] = value;
                }
                std::map<std::string, std::string> revUnchMap = createReverseUnchangeableMap(replacementMap, pr2.second);

                std::vector<std::string> renSplitNK;
                for (const auto& elem : splitNK)
                {
                    renSplitNK.push_back(ce::replaceKeysInString(elem, revUnchMap));
                }

                splitNK = renSplitNK;
                pr2.first.data = splitNK;
                pr2.second = replacementMap;
            }

            std::map<std::vector<std::string>,
                std::tuple<std::map<std::string, std::string>,
                std::string,
				std::vector<std::string>>>::const_iterator it = repetitionExclusionMap.find(splitNK);
            if (it != repetitionExclusionMap.end() and !force)
            {
                const std::tuple<std::map<std::string, std::string>,
                    std::string,
                    std::vector<std::string>>&tpl = it->second;
                const std::map<std::string, std::string>& namePermutation = std::get<0>(tpl);
                const std::string& coreExpression = std::get<1>(tpl);
				const std::vector<std::string>& mainNormalizedDefinition = std::get<2>(tpl);

				std::map<std::string, std::string> combinedMap = combineUnchangeablePaths(pr2.second, namePermutation);
				std::string args = createArgList(combinedMap);

				std::string expression = "(" + coreExpression;
				expression += args;
				expression += ")";
				result = expression;
            }
            else
            {
                const std::vector<std::vector<int>>& permuations = allPermutationsAna[encodedTail.size()];

				int numUnchArgs = 0;

                std::vector<std::vector<EncodedExpression>> allSortedCombos = giveAllSortedCombinations(encodedTail);

                for (const auto& combo : allSortedCombos)
                {
                    std::vector<EncodedExpression> combinedAttempt = combo;
                    combinedAttempt.insert(combinedAttempt.end(), encodedHead.begin(), encodedHead.end());
                    
                    std::pair<NormalizedKey, std::map<std::string, std::string>> attemptPr2 = makeNormalizedEncodedKey(combinedAttempt, 2);
					std::vector<std::string> attemptSplitNK = splitNormalizedKey(attemptPr2.first.data);
					std::map<std::string, std::string> reverseUnchMap = createReverseUnchangeableMap(pr2.second, attemptPr2.second);
					numUnchArgs = static_cast<int>(reverseUnchMap.size());

                    std::string tempCoreExpression;
                    if (force)
                    {
                        tempCoreExpression = ce::extractExpressionUniversal(defaultSignature);
                    }
                    else
                    {
                        tempCoreExpression = newCoreExpressionName;
                    }

                    repetitionExclusionMap[attemptSplitNK] = std::make_tuple(reverseUnchMap, tempCoreExpression, splitNK);
                }

                if (!force)
                {
                    std::string signature = "(" + newCoreExpressionName + "[";
                    for (int index = 1; index <= numUnchArgs; index++)
                    {
                        signature += "u_" + std::to_string(index) + ",";
                    }
                    signature.pop_back(); // remove last comma   
                    signature += "])";

                    {
                        std::string preservedDS;
                        auto itPrev = compiledExpressions.find(newCoreExpressionName);
                        if (itPrev != compiledExpressions.end())
                            preservedDS = itPrev->second.definedSet;
                        compiledExpressions[newCoreExpressionName] = LogicalEntity(category, splitNK, signature, numUnchArgs, preservedDS);
                    }

                    std::string args = createArgList(pr2.second);

                    std::string expression = "(" + newCoreExpressionName;
                    expression += args;
                    expression += ")";
                    result = expression;
                }
                else
                {
                    std::string coreExpressionName = ce::extractExpressionUniversal(defaultSignature);
                    std::string signature = "(" + coreExpressionName + "[";
                    for (int index = 1; index <= numUnchArgs; index++)
                    {
                        signature += "u_" + std::to_string(index) + ",";
                    }
                    signature.pop_back(); // remove last comma   
                    signature += "])";

                    {
                        std::string preservedDS;
                        auto itPrev = compiledExpressions.find(coreExpressionName);
                        if (itPrev != compiledExpressions.end())
                            preservedDS = itPrev->second.definedSet;
                        compiledExpressions[coreExpressionName] = LogicalEntity(category, splitNK, signature, numUnchArgs, preservedDS);
                    }
                    result = defaultSignature;
                }


            }

			return result;
        }



        // Helper to expand the results by appending new permutations
        // effectively performing a Cartesian Product: Result = Result x CurrentGroupPermutations
        void extendCombinations(
            std::vector<std::vector<EncodedExpression>>& currentResults,
            const std::vector<std::vector<EncodedExpression>>& groupPermutations)
        {
            if (currentResults.empty()) {
                currentResults = groupPermutations;
                return;
            }

            std::vector<std::vector<EncodedExpression>> newResults;
            // We expect the new size to be current_count * group_perms_count
            newResults.reserve(currentResults.size() * groupPermutations.size());

            for (const auto& existingPath : currentResults) {
                for (const auto& extension : groupPermutations) {
                    std::vector<EncodedExpression> combined = existingPath;
                    combined.insert(combined.end(), extension.begin(), extension.end());
                    newResults.push_back(std::move(combined));
                }
            }
            currentResults = std::move(newResults);
        }

        std::vector<std::vector<EncodedExpression>> giveAllSortedCombinations(const std::vector<EncodedExpression>& input) {
            if (input.empty()) {
                return {};
            }

            // 1. Group elements by Name. 
            // std::map automatically sorts keys (names), satisfying the global sort requirement.
            std::map<std::string, std::vector<EncodedExpression>> groups;
            for (const auto& expr : input) {
                groups[expr.name].push_back(expr);
            }

            std::vector<std::vector<EncodedExpression>> finalCombinations;

            // 2. Iterate through groups in alphabetical order (A, B, C...)
            for (auto& pair : groups) {
                std::vector<EncodedExpression>& cluster = pair.second;

                // Sort by 'original' string (or any unique ID) to prepare for next_permutation
                // This ensures we start at the lexicographically first permutation for this cluster.
                std::sort(cluster.begin(), cluster.end(),
                    [](const EncodedExpression& a, const EncodedExpression& b) {
                        return a.original < b.original;
                    });

                // Generate all permutations for *just this group*.
                // Cap: groups larger than 7 elements (7!=5040) would explode
                // (e.g. 10 same-typed equalities → 10!=3,628,800).  For such
                // groups only the canonical (sorted) ordering is cached.
                std::vector<std::vector<EncodedExpression>> clusterPermutations;
                if (cluster.size() <= 7) {
                    do {
                        clusterPermutations.push_back(cluster);
                    } while (std::next_permutation(cluster.begin(), cluster.end(),
                        [](const EncodedExpression& a, const EncodedExpression& b) {
                            return a.original < b.original;
                        }));
                } else {
                    // Single canonical ordering — already sorted above.
                    clusterPermutations.push_back(cluster);
                }

                // 3. Multiply (Cartesian Product) into the final results
                extendCombinations(finalCombinations, clusterPermutations);
            }

            return finalCombinations;
        }

        std::string compileCoreExpressionMapCore(const std::string& inputExpression, int& implCounter, int& existenceCounter, int& andCounter, int& variableCounter)
        {
            std::string returnExpression;
            std::string expression = inputExpression;
            std::string expandedExpression = expression;

            bool isSimple = ce::expressionIsSimple(expression);

            // Early exit: if expression is simple and already compiled, do not re-compile.
            // This prevents runtime calls (e.g. from reformulateTheorem) from
            // re-expanding already-compiled definitions and creating duplicate
            // implications with new names.
            if (isSimple)
            {
                std::string coreExpression = ce::extractExpressionUniversal(expression);
                if (compiledExpressions.find(coreExpression) != compiledExpressions.end())
                {
                    return expression;
                }
            }

            if (isSimple)
            {
                std::vector<std::string> args = ce::getArgs(expression);
                std::set<std::string> uniqueArgs = std::set<std::string>(args.begin(), args.end());
                if (uniqueArgs.size() < coreExpressionMap[ce::extractExpressionUniversal(expression)].arity)
                {
                    std::string coreExpression = ce::extractExpressionUniversal(expression);

                    expression = coreExpressionMap[coreExpression].signature;
                }
            }

            expandedExpression = expandExpr(expression);

            if (isSimple and expandedExpression == expression)
            {
				std::string coreExpression = ce::extractExpressionUniversal(expression);

                std::map<std::string, LogicalEntity>::const_iterator itLev =
                compiledExpressions.find(coreExpression);
                if (itLev == compiledExpressions.end())
                {
					int numArgs = ce::getArgs(expression).size();
                    std::string signature = "(" + coreExpression + "[";
                    for (int index = 1; index <= numArgs; index++)
                    {
                        signature += "u_" + std::to_string(index) + ",";
                    }
                    signature.pop_back(); // remove last comma   
                    signature += "])";

                    LogicalEntity le("atomic", std::vector<std::string>(1, ""), signature, numArgs);
					compiledExpressions.insert(std::make_pair(coreExpression, le));
                }

                return expression;
            }

            if (isSimple && ce::expressionIsSimple(expandedExpression))
            {
                std::string coreExpression = ce::extractExpressionUniversal(expression);

                std::map<std::string, LogicalEntity>::const_iterator itLev =
                    compiledExpressions.find(coreExpression);
                if (itLev == compiledExpressions.end())
                {
                    // Define replacements: map args to u_1, u_2, ...
                    std::map<std::string, std::string> replacementMap;
                    std::vector<std::string> args = ce::getArgs(expression);
                    for (size_t i = 0; i < args.size(); ++i) {
                        replacementMap[args[i]] = "u_" + std::to_string(i + 1);
                    }

                    LogicalEntity le;
                    le.category = "and";

                    // Apply u_ renamings to signature
                    le.signature = ce::replaceKeysInString(expression, replacementMap);

                    // Apply u_ renamings to the expanded expression and save as element
                    std::string processedElem = ce::replaceKeysInString(expandedExpression, replacementMap);
                    le.elements.push_back(processedElem);

                    compiledExpressions.insert(std::make_pair(coreExpression, le));
                }

                return expression;
            }

            std::string coreExpr = ce::extractExpressionUniversal(expression);

            std::vector<std::string> overallChain;

            // 1) groom
            const std::vector<std::string> subexprsAfterGrooming = groomExpr(expandedExpression);

            // 2) smoothen
            std::vector<std::string> smoothenedSubexprsAfterGrooming;
            smoothenedSubexprsAfterGrooming.reserve(subexprsAfterGrooming.size());
            for (std::size_t i = 0; i < subexprsAfterGrooming.size(); ++i)
            {
                const std::string sm = this->smoothenExpr(subexprsAfterGrooming[i]);
                smoothenedSubexprsAfterGrooming.push_back(sm);
            }

            // 3) iterate
            for (std::size_t i = 0; i < smoothenedSubexprsAfterGrooming.size(); ++i)
            {
                const std::string& subexpr = smoothenedSubexprsAfterGrooming[i];

                if (startsWith(subexpr, "(>", 2))
                {
                    std::string implName;
                    LogicalEntity le;

                    ce::TreeNode1* root = ce::parseExpr(subexpr);                 

                    std::vector< std::tuple<
                        std::string,                    // left expr
                        std::vector<std::string>,       // args of current '>' node
                        std::set<std::string>           // left node's arguments (set)
                    > > tempChain;

                    std::string head =
                        ce::disintegrateImplication(subexpr, tempChain, coreExpressionMap);

                    std::string simpleHead = compileCoreExpressionMapCore(head, implCounter, existenceCounter, andCounter, variableCounter);

                    std::vector<std::string> chain;
                    std::vector<std::string> simpleChain;
                    chain.reserve(tempChain.size());
                    for (std::size_t i = 0; i < tempChain.size(); ++i) {
                        chain.push_back(std::get<0>(tempChain[i]));

                        std::string simpleElement = compileCoreExpressionMapCore(std::get<0>(tempChain[i]), implCounter, existenceCounter, andCounter, variableCounter);
                        simpleChain.push_back(simpleElement);
                    }
                    simpleChain.push_back(simpleHead);

                    std::vector<EncodedExpression> simpleEncodedChain;
                    simpleEncodedChain.reserve(simpleChain.size());
                    for (const std::string& elem : simpleChain)
                    {
                        simpleEncodedChain.push_back(EncodedExpression(elem, "main"));
                    }
                    for (EncodedExpression& ee : simpleEncodedChain)
                    {
                        for (const std::string arg : root->arguments)
                        {
                            for (std::vector<std::string>& argumentDescription : ee.arguments)
                            {
                                if (arg == argumentDescription[1])
                                {
                                    argumentDescription[0] = "True";
                                }
                            }
                        }
                    }

                    if (isSimple and smoothenedSubexprsAfterGrooming.size() == 1)
                    {
                        implName = excludeRepetitions(std::vector<EncodedExpression>(simpleEncodedChain.begin(), simpleEncodedChain.end() - 1),
                            std::vector<EncodedExpression>(simpleEncodedChain.end() - 1, simpleEncodedChain.end()),
                            "",
                            "implication",
                            true,
                            expression);

                        std::string tempName = ce::extractExpressionUniversal(implName);
                        if (implName != tempName)
                        {
							assert(false && "Should not happen");
                        }

                        returnExpression = implName;
                    }
                    else
                    {
                        std::string coreExprImpl = "implication" + std::to_string(implCounter);
                        implCounter++;

                        implName = excludeRepetitions(std::vector<EncodedExpression>(simpleEncodedChain.begin(), simpleEncodedChain.end() - 1),
                            std::vector<EncodedExpression>(simpleEncodedChain.end() - 1, simpleEncodedChain.end()),
                            coreExprImpl,
                            "implication",
                            false,
                            "");


                        std::string tempName = ce::extractExpressionUniversal(implName);
                        if (coreExprImpl != tempName)
                        {
                            implCounter--;
                        }


                        if (smoothenedSubexprsAfterGrooming.size() == 1)
                        {
                            returnExpression = implName;
                        }
                    }



                    ce::deleteTree(root);

                    overallChain.push_back(implName);
                }

                else if (startsWith(subexpr, "!(>", 3))
                {
                    std::string exName;
                    LogicalEntity le;

                    ce::TreeNode1* root = ce::parseExpr(subexpr);

                    std::tuple<std::string, int, std::string> t = renameLastRemoved(subexpr, variableCounter);
                    variableCounter++;
                    std::string renamedSubdef = std::get<0>(t);
                    

                    // split into left/right
                    std::pair<std::string, std::string> lr = extractValues(renamedSubdef, this->coreExpressionMap);
                    std::string leftExpr = lr.first;
                    std::string compiledLeftExpr = compileCoreExpressionMapCore(leftExpr, implCounter, existenceCounter, andCounter, variableCounter);
                    std::string rightExpr = lr.second;

                    // toggle negation on right
                    if (!rightExpr.empty() && rightExpr[0] == '!') {
                        rightExpr = rightExpr.substr(1);
                    }
                    else {
                        rightExpr = std::string("!") + rightExpr;
                    };
                    std::string compiledRightExpr = compileCoreExpressionMapCore(rightExpr, implCounter, existenceCounter, andCounter, variableCounter);


                    std::vector<EncodedExpression> simpleEncodedChain;
                    simpleEncodedChain.reserve(2);

                    simpleEncodedChain.push_back(EncodedExpression(compiledLeftExpr, "main"));
                    simpleEncodedChain.push_back(EncodedExpression(compiledRightExpr, "main"));
                    
                    for (EncodedExpression& ee : simpleEncodedChain)
                    {
                        for (const std::string arg : root->arguments)
                        {
                            for (std::vector<std::string>& argumentDescription : ee.arguments)
                            {
                                if (arg == argumentDescription[1])
                                {
                                    argumentDescription[0] = "True";
                                }
                            }
                        }
                    }

                    if (isSimple and smoothenedSubexprsAfterGrooming.size() == 1)
                    {
                        exName = excludeRepetitions(std::vector<EncodedExpression>(simpleEncodedChain.begin(), simpleEncodedChain.end()),
                            std::vector<EncodedExpression>(simpleEncodedChain.end(), simpleEncodedChain.end()),
                            "",
                            "existence",
                            true,
                            expression);

                        
                        if (exName != expression)
                        {
                            assert(false && "Should not happen");
                        }

                        returnExpression = exName;
                    }
                    else
                    {
                        std::string coreExprEx = "existence" + std::to_string(existenceCounter);
                        existenceCounter++;

                        exName = excludeRepetitions(std::vector<EncodedExpression>(simpleEncodedChain.begin(), simpleEncodedChain.end()),
                            std::vector<EncodedExpression>(simpleEncodedChain.end(), simpleEncodedChain.end()),
                            coreExprEx,
                            "existence",
                            false,
                            "");
                        std::string tempName = ce::extractExpressionUniversal(exName);
                        if (coreExprEx != tempName)
                        {
                            existenceCounter--;
                        }


                        if (smoothenedSubexprsAfterGrooming.size() == 1)
                        {
                            returnExpression = exName;
                        }
                    }
                    


                    ce::deleteTree(root);

                    overallChain.push_back(exName);
                }
                else if (startsWith(subexpr, "!(&", 3))
                {
                    // split into left/right
                    std::pair<std::string, std::string> lr = extractValues(subexpr, this->coreExpressionMap);
                    std::string leftExpr = lr.first;
                    std::string rightExpr = lr.second;

                    std::string notLeftExpr;
                    // toggle negation on left
                    if (!leftExpr.empty() && leftExpr[0] == '!') {
                        notLeftExpr = leftExpr.substr(1);
                    }
                    else {
                        notLeftExpr = std::string("!") + leftExpr;
                    };

					std::string notRightExpr;
                    // toggle negation on right
                    if (!rightExpr.empty() && rightExpr[0] == '!') {
                        notRightExpr = rightExpr.substr(1);
                    }
                    else {
                        notRightExpr = std::string("!") + rightExpr;
                    };

					std::string notLeftRightExpr = "(>[]" + notLeftExpr + "," + rightExpr + ")";
					std::string notRightLeftExpr = "(>[]" + notRightExpr + "," + leftExpr + ")";

                    std::string compiledNotLeftRightExpr = compileCoreExpressionMapCore(notLeftRightExpr, implCounter, existenceCounter, andCounter, variableCounter);
                    //std::string compiledNotRightLeftExpr = compileCoreExpressionMapCore(notRightLeftExpr, implCounter, existenceCounter, andCounter, variableCounter);
                    overallChain.push_back(compiledNotLeftRightExpr);
					//overallChain.push_back(compiledNotRightLeftExpr);
                }
                else if (ce::expressionIsSimple(subexpr))
                {
                    std::string compiledExpr = compileCoreExpressionMapCore(subexpr, implCounter, existenceCounter, andCounter, variableCounter);
                    assert(compiledExpr == subexpr && "Should not happen");
                    overallChain.push_back(compiledExpr);
                }
                else
                {
					assert(false && "Should not happen");
                }

            }

            if (smoothenedSubexprsAfterGrooming.size() > 1)
            {
                std::string andName;
                LogicalEntity le;

                std::vector<EncodedExpression> simpleEncodedChain;
                simpleEncodedChain.reserve(overallChain.size());

                for (const auto& elem : overallChain)
                {
                    simpleEncodedChain.push_back(EncodedExpression(elem, "main"));
                }

                for (EncodedExpression& ee : simpleEncodedChain)
                {
                    for (std::vector<std::string>& argumentDescription : ee.arguments)
                    {
                        argumentDescription[0] = "True";
                    }
                }

                if (isSimple)
                {
                    std::string coreExpression = ce::extractExpressionUniversal(expression);
                    andName = excludeRepetitions(std::vector<EncodedExpression>(simpleEncodedChain.begin(), simpleEncodedChain.end()),
                        std::vector<EncodedExpression>(simpleEncodedChain.end(), simpleEncodedChain.end()),
                        "",
                        "and",
                        true,
                        expression);


                    if (andName != expression)
                    {
                        assert(false && "Should not happen");
                    }
                }
                else
                {
                    ce::TreeNode1* root = ce::parseExpr(expression);

                    std::string coreExprAnd = "and" + std::to_string(andCounter);
                    andCounter++;

                    andName = excludeRepetitions(std::vector<EncodedExpression>(simpleEncodedChain.begin(), simpleEncodedChain.end()),
                        std::vector<EncodedExpression>(simpleEncodedChain.end(), simpleEncodedChain.end()),
                        coreExprAnd,
                        "and",
                        false,
                        "");
                    std::string tempName = ce::extractExpressionUniversal(andName);
                    if (coreExprAnd != tempName)
                    {
                        andCounter--;
                    }

                    ce::deleteTree(root);
                }

                returnExpression = andName;
            }

            if (isSimple)
            {
                returnExpression = inputExpression;
            }

            return returnExpression;
        }

        void checkCompiledCoreExpressionMap() {
            // Local helper struct for recursion to verify no nested implications exist.
            struct DeepChecker {
                const std::map<std::string, LogicalEntity>& definitions;

                DeepChecker(const std::map<std::string, LogicalEntity>& defs) : definitions(defs) {}

                void check(const std::string& expression) const {
                    // 1. Identify the core entity name from the expression string
                    std::string core = ce::extractExpressionUniversal(expression);

                    auto it = definitions.find(core);

                    // Base Case 1: The core MUST be in our compiled map.
                    // If it is missing, we assert failure instead of treating it as a variable/primitive.
                    if (it == definitions.end()) {
                        std::cerr << "Assertion Failed: Core expression not found in definitions." << std::endl;
                        std::cerr << "  Expression: " << expression << std::endl;
                        std::cerr << "  Core: " << core << std::endl;
                        assert(false && "Core expression not found in compiled expressions.");
                    }

                    const LogicalEntity& le = it->second;

                    // 2. Check for Forbidden Category (Implication)
                    // If we encounter an implication while recursing *inside* another implication, assert failure.
                    if (le.category == "implication") {
                        std::cerr << "Assertion Failed: Nested implication detected in definition." << std::endl;
                        std::cerr << "  Expression: " << expression << std::endl;
                        std::cerr << "  The prover does not yet support implications as constituents of other implications." << std::endl;
                        assert(false && "Nested implication detected in compiled expressions.");
                    }

                    // Base Case 2: Atomic entities stop the recursion.
                    if (le.category == "atomic") {
                        return;
                    }

                    // 3. Recursive Step
                    // For categories like "and", "existence", etc., we must deep dive into their elements.
                    for (const std::string& subElement : le.elements) {
                        check(subElement);
                    }
                }
            };

            DeepChecker checker(compiledExpressions);

            // Iterate through all compiled LogicalEntities
            for (auto it = compiledExpressions.begin(); it != compiledExpressions.end(); ++it) {
                const LogicalEntity& le = it->second;

                // We only initiate the deep check if the top-level entity is itself an implication.
                // We want to ensure that NONE of its constituents (body or head) contain an implication.
                if (le.category == "implication") {
                    checker.check(le.elements.back());
                }
            }
        }

        std::string findDefinitionSetRecursive(const std::string& var, const LogicalEntity& le) {
            for (const std::string& elem : le.elements) {
                std::string core = ce::extractExpression(elem);
                std::vector<std::string> args = ce::getArgs(elem);

                for (size_t i = 0; i < args.size(); ++i) {
                    if (args[i] == var) {
                        // Case 1: Element is in coreExpressionMap (atomic/base operator)
                        if (coreExpressionMap.find(core) != coreExpressionMap.end()) {
                            std::string idx = std::to_string(i + 1);
                            const auto& ds = coreExpressionMap.at(core).definitionSets;
                            if (ds.find(idx) != ds.end()) {
                                return ds.at(idx).first;
                            }
                        }
                        // Case 2: Element is in compiledExpressions (recurse into sub-macro)
                        else if (compiledExpressions.find(core) != compiledExpressions.end()) {
                            const LogicalEntity& subLe = compiledExpressions.at(core);
                            std::vector<std::string> sigArgs = ce::getArgs(subLe.signature);
                            if (i < sigArgs.size()) {
                                std::string mappedVar = sigArgs[i];
                                std::string res = findDefinitionSetRecursive(mappedVar, subLe);
                                if (!res.empty()) return res;
                            }
                        }
                    }
                }
            }
            return "";
        }

        void extendCoreExpressionMap() {
            for (const auto& kv : compiledExpressions) {
                const std::string& key = kv.first;
                const LogicalEntity& le = kv.second;

                // Only process if missing from coreExpressionMap
                if (coreExpressionMap.find(key) == coreExpressionMap.end()) {
                    ce::CoreExpressionConfig cfg;
                    cfg.arity = le.arity;

                    // Definition and signature: strip "u_" prefixes
                    std::string cleanSig = removeUPrefixFromArguments(le.signature);
                    cfg.signature = cleanSig;
                    cfg.definition = cleanSig; // As requested: fill definition same as signature

                    // Populate definitionSets by tracing arguments
                    std::vector<std::string> sigArgs = ce::getArgs(le.signature);
                    for (size_t i = 0; i < sigArgs.size(); ++i) {
                        std::string defSet = findDefinitionSetRecursive(sigArgs[i], le);
                        if (!defSet.empty()) {
                            // pair[0] = definitionSet, pair[1] = false
                            cfg.definitionSets[std::to_string(i + 1)] = std::make_pair(defSet, false);
                        }
                    }

                    coreExpressionMap[key] = cfg;
                }
            }
        }

//#pragma optimize("", off)

        void compileCoreExpressionMap()
        {
            // ---------------------------------------------------------
            // 1. Compilation Phase
            // Must run FIRST so 'compiledExpressions' (LogicalEntities) exist.
            // ---------------------------------------------------------
            for (auto coreExpression : this->coreExpressionMap)
            {
                compileCoreExpressionMapCore(coreExpression.second.signature, implCounter, existenceCounter, statementCounter, variableCounter);
            }

            // ---------------------------------------------------------
            // 2. Analysis Phase
            // Updates the existing LogicalEntities with 'definedSet' info.
            // ---------------------------------------------------------
            gl::ArgumentAnalyzer analyzer(this->coreExpressionMap);

            for (const auto& kv : this->coreExpressionMap) {
                const std::string& key = kv.first; // "identity", "interval", etc.
                const ce::CoreExpressionConfig& cfg = kv.second; // Already accessed, no need to look up

                std::string defString;
                if (std::holds_alternative<std::string>(cfg.definition)) {
                    defString = std::get<std::string>(cfg.definition);
                }

                if (!defString.empty()) {
                    // Build the tree
                    gl::AnalysisNode* root = analyzer.analyze(defString);
                    assert(root != nullptr && "Analyzer produced null for non-empty definition.");

                    // Use the instance 'analyzer' to call the static method (safer syntax)
                    analyzer.checkDefinitionConsistency(root, cfg, cfg.signature);

                    std::string setArg;
                    std::string defSet;

                    // Check if this definition describes a Set
                    if (analyzer.isSetDefinition(root, setArg, defSet)) {

                        // 1. Find the index of the set argument in the Config Signature
                        //    We use cfg.signature because 'setArg' matches the variable names used there.
                        std::vector<std::string> cfgArgs = ce::getArgs(cfg.signature);
                        auto itArg = std::find(cfgArgs.begin(), cfgArgs.end(), setArg);

                        // Strict assertion: The variable found in the definition MUST exist in the signature
                        assert(itArg != cfgArgs.end() && "Set argument from definition not found in configuration signature.");

                        int index = static_cast<int>(std::distance(cfgArgs.begin(), itArg));

                        // 2. Retrieve the compiled LogicalEntity using the loop key
                        auto leIt = compiledExpressions.find(key);
                        assert(leIt != compiledExpressions.end() && "LogicalEntity not found in compiled expressions.");

                        // 3. Extract the corresponding argument from the compiled signature
                        //    The names here (e.g., u_1, u_2) might differ from config, but the position (index) is invariant.
                        std::vector<std::string> compiledArgs = ce::getArgs(leIt->second.signature);
                        assert(index >= 0 && index < static_cast<int>(compiledArgs.size()) && "Argument index out of bounds in compiled signature.");

                        // 4. Update the definedSet field
                        leIt->second.definedSet = compiledArgs[index];

                        /*
                        std::cout << ">>> [Set Definition Detected] Key: " << key
                        << " | Variable: " << setArg
                        << " | Index: " << index
                        << " | Mapped to Sig Arg: " << leIt->second.definedSet << std::endl;
                        */

                    }

                    delete root;
                }
            }

            checkCompiledCoreExpressionMap();

            for (const auto& kv : coreExpressionMap) {
                this->expressionsFromConfig.insert(kv.first);
            }


            extendCoreExpressionMap();
        }

        // Turn optimizations OFF for just this section
//#pragma optimize("", off)
        

        void prepareIntegrationCore(const std::string& expression,
            Instruction& instruction,
            Memory& mb,
            std::string goal)
        {
            // 1. Extract core and find corresponding LogicalEntity
            std::string core = ce::extractExpressionUniversal(expression);
            auto it = compiledExpressions.find(core);
            instruction.markedGoal = goal;

            // Safety check: if core not found, we cannot proceed with template instantiation
            if (it == compiledExpressions.end()) {
                assert(false);
            }

            // Use const reference to ensure we do not modify the original template
            const LogicalEntity& le = it->second;

            // 2. Create Replacement Map (Signature -> Expression Arguments)
            std::vector<std::string> sigArgs = ce::getArgs(le.signature);
            std::vector<std::string> exprArgs = ce::getArgs(expression);
            std::map<std::string, std::string> replacementMap;

            // Map u_x in signature to actual arguments in expression
            size_t argLimit = std::min(sigArgs.size(), exprArgs.size());
            for (size_t i = 0; i < argLimit; ++i) 
            {
				replacementMap[sigArgs[i]] = exprArgs[i]; 
            }

            // 3. Special Handling for "existence" category
            // We must identify bound variables (args without "u_") in the original elements
            // and map them to fresh variables.
            if (le.category == "existence") {
                for (const std::string& rawElement : le.elements) {
                    std::vector<std::string> elemArgs = ce::getArgs(rawElement);
                    for (const std::string& arg : elemArgs) {
                        // Check if argument does NOT start with "u_"
                        bool isU = (arg.size() >= 2 && arg[0] == 'u' && arg[1] == '_');

                        if (!isU) {
                            // It is a bound variable. Check if we already assigned a fresh ID for it.
                            if (replacementMap.find(arg) == replacementMap.end()) {
                                // Create fresh variable: pi_lev_<level>_<startInt>
                                std::string freshVar = "pi_lev_" + std::to_string(mb.level) + "_" + std::to_string(mb.startIntPi);

                                // Update mapping
                                replacementMap[arg] = freshVar;

                                // Update mb.startInt as requested
                                mb.startIntPi++;
                            }
                        }
                    }
                }
            }



            //if (le.category != "atomic")
            {
                // 4. Create New LogicalEntity (The Instance)
                LogicalEntity newLe = le; // Copy properties (category, arity, etc.)

                // Replace signature in new entity by the expression itself
                newLe.signature = ce::replaceKeysInString(le.signature, replacementMap);

                // Apply replacement map to all elements
                newLe.elements.clear();
                newLe.elements.reserve(le.elements.size());
                for (const std::string& rawElement : le.elements) {
                    newLe.elements.push_back(ce::replaceKeysInString(rawElement, replacementMap));
                }

                // Add the fully instantiated entity to instructions
                instruction.data.push_back(newLe);

                // 5. Recursion
                // If category is "existence" or "and", recurse on the NEW elements
                if (le.category == "existence" || le.category == "and") {
                    for (const std::string& subExpr : newLe.elements) {
                        prepareIntegrationCore(subExpr, instruction, mb, goal);
                    }
                }
            }
        }

        std::set<std::string> findAllUArgs(const LogicalEntity& le) {
            std::set<std::string> uArgs;

            // Helper lambda to check arguments and strip "u_"
            auto processExpression = [&](const std::string& expr) {
                std::vector<std::string> args = ce::getArgs(expr);
                for (const std::string& arg : args) {
                    // Check for "u_" prefix
                    if (arg.size() >= 2 && arg[0] == 'u' && arg[1] == '_') {
                        // Insert substring starting at index 2 (removes "u_")
                        uArgs.insert(arg.substr(2));
                    }
                }
            };

            // 1. Check the Signature
            processExpression(le.signature);

            // 2. Check all Elements
            for (const std::string& element : le.elements) {
                processExpression(element);
            }

            return uArgs;
        }

        inline std::vector<std::string> renamingChain2(const std::vector<std::string>& chain,
            Memory& mb) {
            std::map<std::string, std::string> replacementMap;

            for (const std::string& expr : chain) {
                std::vector<std::string> args = ce::getArgs(expr);
                for (const std::string& arg : args) {
                    // Check for "u_" prefix
                    if (startsWith(arg, "u_", 2)) {
                        // Remove "u_"
                        replacementMap[arg] = arg.substr(2);
                    }
                    else {
                        // If not already mapped, assign new "repl_..." name using mb.startInt
                        if (replacementMap.find(arg) == replacementMap.end()) {
                            // Pattern: repl_lev_<mb.level>_<mb.startInt>
                            std::string newName = "repl_lev_" + std::to_string(mb.level) + "_" + std::to_string(mb.startIntRepl);

                            // Update startInt immediately after use
                            mb.startIntRepl++;

                            replacementMap[arg] = newName;
                        }
                    }
                }
            }

            std::vector<std::string> renamedChain;
            renamedChain.reserve(chain.size());
            for (const std::string& expr : chain) {
                renamedChain.push_back(ce::replaceKeysInString(expr, replacementMap));
            }

            return renamedChain;
        }

        std::string expandSignatureForIntegration(const std::string& category, const std::vector<std::string>& renamedChain, const std::string& cleanSignature, bool* hasPiBoundVars = nullptr) {
            // 1. Extract arguments from cleanSignature and assert no 'u_' prefix
            std::vector<std::string> sigArgsVec = ce::getArgs(cleanSignature);
            std::set<std::string> sigArgs;
            std::map<std::string, std::string> toUReplacementMap;

            for (const std::string& arg : sigArgsVec) {
                assert((arg.size() < 2 || arg.substr(0, 2) != "u_") && "Signature argument already contains 'u_' prefix!");
                sigArgs.insert("u_" + arg);
                toUReplacementMap[arg] = "u_" + arg;
            }

            // 2. Apply "u_" prefix replacements to the entire renamedChain and the signature
            std::vector<std::string> modifiedChain;
            modifiedChain.reserve(renamedChain.size());
            for (const std::string& elem : renamedChain) {
                modifiedChain.push_back(ce::replaceKeysInString(elem, toUReplacementMap));
            }
            std::string modifiedSignature = ce::replaceKeysInString(cleanSignature, toUReplacementMap);

            std::string result;

            // 3. Build Result String based on Category Logic
            if (category == "and") {
                if (modifiedChain.empty()) result = modifiedSignature;
                else if (modifiedChain.size() == 1) result = modifiedChain[0];
                else {
                    std::string current = modifiedChain[0];
                    for (size_t i = 1; i < modifiedChain.size(); ++i) {
                        current = "(&" + current + modifiedChain[i] + ")";
                    }
                    result = current;
                }
            }
            else if (category == "existence") {
                if (modifiedChain.size() < 2) result = modifiedSignature;
                else {
                    std::string body = modifiedChain[0];
                    std::string head = modifiedChain[1];

                    std::string negatedHead;
                    if (!head.empty() && head[0] == '!') {
                        negatedHead = head.substr(1);
                    }
                    else {
                        negatedHead = "!" + head;
                    }

                    std::vector<std::string> bodyArgs = ce::getArgs(body);
                    std::set<std::string> boundVars;
                    for (const auto& arg : bodyArgs) {
                        if (sigArgs.find(arg) == sigArgs.end()) {
                            boundVars.insert(arg);
                        }
                    }

                    assert(!boundVars.empty() && "Existence must have at least one bound variable after expansion for integration");

                    // Check if bound vars are pi_lev_-prefixed (unoccupied) or not (occupied)
                    // Format: pi_lev_<level>_<counter>, e.g. pi_lev_0_1
                    bool allPi = true;
                    for (const auto& bv : boundVars) {
                        if (bv.size() < 7 || bv[0] != 'p' || bv[1] != 'i' || bv[2] != '_'
                            || bv[3] != 'l' || bv[4] != 'e' || bv[5] != 'v' || bv[6] != '_') {
                            allPi = false;
                            break;
                        }
                    }
                    if (hasPiBoundVars) {
                        *hasPiBoundVars = allPi;
                    }

                    if (allPi) {
                        // Unoccupied bound vars: preserve binding — !(>[bound](body)(head))
                        std::string varsStr;
                        for (const auto& v : boundVars) {
                            if (!varsStr.empty()) varsStr += ",";
                            varsStr += v;
                        }
                        result = "!(>[" + varsStr + "]" + body + negatedHead + ")";
                    } else {
                        // Occupied bound vars: empty binding — (>[](body)(head))
                        result = "(>[]" + body + head + ")";
                    }
                }
            }
            else if (category == "implication") {
                if (modifiedChain.empty()) {
                    result = modifiedSignature;
                }
                else {
                    std::vector<std::string> chain = modifiedChain;
                    std::string head = chain.back();
                    chain.pop_back();
                    result = this->reconstructImplicationFullBind(chain, head);
                }
            }
            else {
                assert(false && "Forbidden category encountered in expandSignatureForIntegration.");
                return cleanSignature;
            }

            // 4. AST Token Parsing to collect all variables in the reconstructed result
            std::set<std::string> tokens;
            ce::TreeNode1* root = ce::parseExpr(result);

            std::vector<ce::TreeNode1*> stack;
            if (root) {
                stack.push_back(root);
            }

            while (!stack.empty()) {
                ce::TreeNode1* curr = stack.back();
                stack.pop_back();

                std::vector<std::string> args = ce::getArgs(curr->value);
                for (const std::string& t : args) {
                    tokens.insert(t);
                }

                if (curr->left) stack.push_back(curr->left);
                if (curr->right) stack.push_back(curr->right);
            }

            ce::deleteTree(root);

            // 5. Final Renaming: Strip "u_" prefix everywhere 
            std::map<std::string, std::string> renameMap;

            for (const std::string& t : tokens) {
                if (t.size() >= 2 && t[0] == 'u' && t[1] == '_') {
                    renameMap[t] = t.substr(2);
                }
            }

            // 6. Apply Renaming to return the final clean signature
            return ce::replaceKeysInString(result, renameMap);
        }

        // Strip u_ prefix from ALL variables in a compound expression using AST parsing.
        // Unlike removeUPrefixFromArguments (which only sees the first [...]),
        // this walks the full parse tree to find all tokens at every nesting level.
        static std::string stripUPrefixAST(const std::string& expr) {
            std::set<std::string> tokens;
            ce::TreeNode1* root = ce::parseExpr(expr);

            std::vector<ce::TreeNode1*> stack;
            if (root) stack.push_back(root);

            while (!stack.empty()) {
                ce::TreeNode1* curr = stack.back();
                stack.pop_back();

                std::vector<std::string> args = ce::getArgs(curr->value);
                for (const std::string& a : args) {
                    if (!a.empty()) tokens.insert(a);
                }

                if (curr->left) stack.push_back(curr->left);
                if (curr->right) stack.push_back(curr->right);
            }

            ce::deleteTree(root);

            std::map<std::string, std::string> renameMap;
            for (const std::string& t : tokens) {
                if (t.size() >= 2 && t[0] == 'u' && t[1] == '_') {
                    renameMap[t] = t.substr(2);
                }
            }

            return ce::replaceKeysInString(expr, renameMap);
        }

        std::pair<std::string, std::string> buildIntegrationInstruction(const std::vector<std::string>& elements, const std::string& signature) {
            // 1. Build the implication: only pi_-prefixed vars get bound (unoccupied existence bound vars)
            std::string implication = this->reconstructImplicationForIntegration(elements, signature);

            // 2. History version: strip u_, and strip non-pi_lev_ bound vars from outermost >[...]
            //    pi_lev_ stays in >[...], occupied (non-pi_lev_) vars get stripped to >[]
            std::string history = stripUPrefixAST(implication);
            if (history.size() > 3 && history[0] == '(' && history[1] == '>' && history[2] == '[') {
                std::size_t bracketEnd = history.find(']', 3);
                if (bracketEnd != std::string::npos && bracketEnd > 3) {
                    std::string bv = history.substr(3, bracketEnd - 3);
                    bool isPiLev = (bv.size() >= 7 && bv[0] == 'p' && bv[1] == 'i' && bv[2] == '_'
                        && bv[3] == 'l' && bv[4] == 'e' && bv[5] == 'v' && bv[6] == '_');
                    if (!isPiLev) {
                        history = "(>[]" + history.substr(bracketEnd + 1);
                    }
                }
            }
            // 3. Return pair: .first = history (pi_lev_ kept, others stripped), .second = u_-preserved (for hash matching)
            return std::make_pair(history, implication);
        }

//#pragma optimize("", off)

        void prepareIntegrationCore2(Instruction& instructions,
            const std::map<std::string, std::string>& replacementMap,
            Memory& mb,
            std::string validityName,
            std::string original)
        {
            //test
            Instruction& instructionsCopy = instructions;

            auto makeMarkedExprLambda = [](const std::string& expression) -> std::string {
                std::vector<std::string> args = ce::getArgs(expression);
                std::map<std::string, std::string> replacementMap;

                int counter = 0;
                for (const std::string& arg : args) {
                    // Check for "u_repl_" prefix (length 7)
                    if (arg.rfind("u_repl_", 0) == 0) {
                        replacementMap[arg] = arg.substr(2); // "u_repl_..." -> "repl_..."
                    }
                    // Check for "pi_" prefix (length 3)
                    else if (arg.rfind("pi_", 0) == 0) {
                        replacementMap[arg] = "marker";
                        counter++;
                    }
                }

                assert(counter <= 1);

                return ce::replaceKeysInString(expression, replacementMap);
                };

            // [EXTENSION] Pre-calculate allowed repetitions based on instructions signatures
            std::set<std::string> allowedRepeatedVars;
            for (const LogicalEntity& le : instructions.data) {
                std::string coreExpr = ce::extractExpression(le.signature);

                // If signature's core expression is part of coreExpressionMap keys
                if (this->expressionsFromConfig.find(coreExpr) != this->expressionsFromConfig.end()) {
                    std::vector<std::string> args = ce::getArgs(le.signature);
                    std::map<std::string, int> counts;

                    for (const std::string& a : args) {
                        counts[a]++;
                    }

                    // Take those args which happen more than once
                    for (const auto& kv : counts) {
                        if (kv.second > 1) {
                            allowedRepeatedVars.insert(kv.first);
                        }
                    }
                }
            }

            // 1. Apply replacement map on all signatures and elements of every instruction
            for (LogicalEntity& le : instructions.data) {
                le.signature = ce::replaceKeysInString(le.signature, replacementMap);
                for (std::string& elem : le.elements) {
                    elem = ce::replaceKeysInString(elem, replacementMap);

                    bool isControl = startsWith(elem, "(implication", 12) ||
                        startsWith(elem, "(existence", 10) ||
                        startsWith(elem, "(and", 4);

                    if (isControl) {
                        std::vector<std::string> args = ce::getArgs(elem);
                        std::set<std::string> uniqueArgs(args.begin(), args.end());

                        // [MODIFIED CHECK]
                        if (uniqueArgs.size() != args.size()) {

                            // Identify which specific variables are repeated in this element
                            std::map<std::string, int> currentCounts;
                            for (const std::string& a : args) {
                                currentCounts[a]++;
                            }

                            bool unauthorizedRepetition = false;
                            for (const auto& kv : currentCounts) {
                                if (kv.second > 1) {
                                    // If a variable is repeated here but WAS NOT repeated in the input expression, fail.
                                    if (allowedRepeatedVars.find(kv.first) == allowedRepeatedVars.end()) {
                                        unauthorizedRepetition = true;
                                        break;
                                    }
                                }
                            }

                            if (unauthorizedRepetition) {
                                return;
                            }
                        }
                    }
                }
            }

            std::set<LogicalEntity> toBeRemoved;

            // 2. Analyze Instructions for Admission/Memory
            for (const LogicalEntity& le : instructions.data) 
            {

                std::vector<std::string> sigArgs = ce::getArgs(le.signature);
                bool allSigArgsAreU = true;
                for (const std::string& a : sigArgs) {
                    if (!(a.size() >= 2 && a[0] == 'u' && a[1] == '_')) {
                        allSigArgsAreU = false;
                        break;
                    }
                }

                // Case A: Implication with all u_ args
                if (le.category == "implication" && allSigArgsAreU) {
					toBeRemoved.insert(le);

                    // [INSERTED CODE START]test
                    bool isControl = startsWith(le.signature, "(implication", 12)
                        //||
                        //startsWith(elem, "(existence", 10) ||
                        //startsWith(elem, "(and", 4)
                        ;

                    if (isControl) {
                        std::vector<std::string> args = ce::getArgs(le.signature);
                        std::set<std::string> uniqueArgs(args.begin(), args.end());
                        if (uniqueArgs.size() != args.size()) {
                            //continue;
                        }
                    }
                    // [INSERTED CODE END]

                    std::string cleanSignature = removeUPrefixFromArguments(le.signature);

                    if (mb.wholeExpressions.find(EncodedExpression(cleanSignature, validityName)) != mb.wholeExpressions.end())
                    {
                        continue;
                    }

                    if (mb.integrationPrepared.find(ExpressionWithValidity(cleanSignature, validityName)) != mb.integrationPrepared.end())
                    {
                        continue;
                    }

                    mb.integrationPrepared.insert(ExpressionWithValidity(cleanSignature, validityName));

					
                    std::vector<std::string> renamedChain = renamingChain2(le.elements, mb);

                    std::string expandedImplication = expandSignatureForIntegration("implication", renamedChain, cleanSignature);
                    if (parameters.trackHistory)
                    {
                        std::pair<std::string, std::vector<ExpressionWithValidity>> originImplicationExpansion;
                        originImplicationExpansion.first = "expansion for integration";
                        originImplicationExpansion.second.push_back(ExpressionWithValidity(cleanSignature + "_integration_goal", validityName));

                        ExpressionWithValidity ev(expandedImplication + "_integration_goal", validityName);

                        // The addOrigin helper replaces the if() check, handling both 
                        // the max origin limits and preventing duplicate paths.
                        addOrigin(mb.exprOriginMap, ev, originImplicationExpansion, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                        addOrigin(mb.mailOut.exprOriginMap, ev, originImplicationExpansion, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                    }

                    for (size_t i = 0; i < renamedChain.size() - 1; ++i) {
                        const std::string& elem = renamedChain[i];

                        std::pair<std::string, std::vector<ExpressionWithValidity>> originPremiseElement;
                        if (parameters.trackHistory)
                        {
                            originPremiseElement.first = "premise element";
                            originPremiseElement.second.push_back(ExpressionWithValidity(expandedImplication + "_integration_goal", validityName));
                            
                            ExpressionWithValidity ev(elem, cleanSignature);

                            addOrigin(mb.exprOriginMap, ev, originPremiseElement, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                            addOrigin(mb.mailOut.exprOriginMap, ev, originPremiseElement, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                        }

                        this->addExprToMemoryBlock(elem, mb, -1, 0, std::set<int>(), originPremiseElement, -1, -1, cleanSignature, false);
                    }

                    this->addExprToMemoryBlock(renamedChain.back(), mb, -1, 2, std::set<int>(), std::pair<std::string, std::vector<ExpressionWithValidity>>(), -1, -1, cleanSignature, false);
                }

				std::set<std::string> unchangeableArguments = findAllUArgs(le);
                // Case B: Existence or And with all u_ args
                if ((le.category == "existence" || le.category == "and") && allSigArgsAreU) {
					toBeRemoved.insert(le);

                    //test
                    bool isControl = startsWith(le.signature, "(implication", 12)
                        //||
                        //startsWith(elem, "(existence", 10) ||
                        //startsWith(elem, "(and", 4)
                        ;

                    if (true) {
                        std::vector<std::string> args = ce::getArgs(le.signature);
                        std::set<std::string> uniqueArgs(args.begin(), args.end());
                        if (uniqueArgs.size() != args.size()) {
                            //continue;
                        }
                    }

                    std::string cleanSignature = removeUPrefixFromArguments(le.signature);
                    if (mb.integrationPrepared.find(ExpressionWithValidity(cleanSignature, validityName)) != mb.integrationPrepared.end())
                    {
                        continue;
                    }
                    mb.integrationPrepared.insert(ExpressionWithValidity(cleanSignature, validityName));

                    if (le.category == "existence")
                    {
                        for (const auto& elem : le.elements) {
                            std::string markedElem = makeMarkedExprLambda(elem);

                            for (const auto& implication : mb.overallHashMemory.originals)
                            {
                                makeAdmissionKeys(implication, markedElem, mb.overallHashMemory, validityName);
                            }
                        }
                    }


                    
                    std::string integrationInstructionForHistory;
                    std::string integrationInstructionForHash;
                    if (parameters.trackHistory)
                    {
                        std::string expandedSignature;
                        bool hasPiBoundVars = false;

                        if (le.category == "existence")
                        {
                            expandedSignature = expandSignatureForIntegration("existence", le.elements, cleanSignature, &hasPiBoundVars);
                        }
                        if (le.category == "and")
                        {
                            expandedSignature = expandSignatureForIntegration("and", le.elements, cleanSignature);
                        }


                        std::pair<std::string, std::vector<ExpressionWithValidity>> originExpansion;
                        originExpansion.first = "expansion for integration";
                        originExpansion.second.push_back(ExpressionWithValidity(cleanSignature + "_integration_goal", validityName));

                        ExpressionWithValidity ev(expandedSignature + "_integration_goal", validityName);

                        addOrigin(mb.exprOriginMap, ev, originExpansion, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                        addOrigin(mb.mailOut.exprOriginMap, ev, originExpansion, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));

                        auto integrationInstructionPair = buildIntegrationInstruction(le.elements, le.signature);
                        integrationInstructionForHistory = integrationInstructionPair.first;
                        integrationInstructionForHash = integrationInstructionPair.second;

                        std::pair<std::string, std::vector<ExpressionWithValidity>> originInstruction;
                        if (le.category == "and") {
                            originInstruction.first = "reformulation for integration and";
                        } else if (le.category == "existence" && hasPiBoundVars) {
                            originInstruction.first = "reformulation for integration >[bound]";
                        } else if (le.category == "existence" && !hasPiBoundVars) {
                            originInstruction.first = "reformulation for integration >[]";
                        }
                        originInstruction.second.push_back(ExpressionWithValidity(expandedSignature + "_integration_goal", validityName));

                        ExpressionWithValidity iiv(integrationInstructionForHistory, validityName);

                        addOrigin(mb.exprOriginMap, iiv, originInstruction, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                        addOrigin(mb.mailOut.exprOriginMap, iiv, originInstruction, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                        // Also record for u_-preserved version (what addToHashMemory/buildStack sees)
                        ExpressionWithValidity iivHash(integrationInstructionForHash, validityName);
                        addOrigin(mb.exprOriginMap, iivHash, originInstruction, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                        addOrigin(mb.mailOut.exprOriginMap, iivHash, originInstruction, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                    }

                    this->addToHashMemory(le.elements,
                        le.signature,
                        unchangeableArguments,
                        mb,
                        mb.overallHashMemory,
                        std::set<int>(),
                        integrationInstructionForHash,
                        0, 0, false, 0,
                        "integration",
                        false,
                        integrationInstructionForHistory,
                        validityName);
                    this->addToHashMemory(le.elements,
                        le.signature,
                        unchangeableArguments,
                        mb,
                        mb.localHashMemory,
                        std::set<int>(),
                        integrationInstructionForHash,
                        0, 0, false, 0,
                        "integration",
                        false,
                        integrationInstructionForHistory,
                        validityName);
                    this->addToHashMemory(le.elements,
                        le.signature,
                        unchangeableArguments,
                        mb,
                        mb.localHashMemoryDelta,
                        std::set<int>(),
                        integrationInstructionForHash,
                        0, 0, false, 0,
                        "integration",
                        false,
                        integrationInstructionForHistory,
                        validityName);
                }

                // Case C: Marker Logic (Admission)

                // C1. Check Signature (Only for "atomic")
                //if (le.category == "atomic") 
                {
                    //test

                    if (true) {
                        std::vector<std::string> args = ce::getArgs(le.signature);
                        std::set<std::string> uniqueArgs(args.begin(), args.end());
                        if (uniqueArgs.size() != args.size()) {
                            continue;
                        }
                    }

                    if (ce::extractExpressionUniversal(le.signature) == ce::extractExpressionUniversal(instructions.markedGoal))
                    {
                        continue;
					}

                    std::vector<std::string> sArgs = ce::getArgs(le.signature);
                    std::string nonUArg;
                    int countNonU = 0;

                    for (const std::string& a : sArgs) {
                        if (!(a.size() >= 2 && a[0] == 'u' && a[1] == '_')) {
                            nonUArg = a;
                            countNonU++;
                        }
                    }

                    if (countNonU == 1 && expressionsFromConfig.count(ce::extractExpressionUniversal(le.signature)) > 0)
                    {
                        std::map<std::string, std::string> markerMap;
                        markerMap[nonUArg] = "marker";
                        std::string keyString = ce::replaceKeysInString(le.signature, markerMap);

                        // Create copy for admission map
                        Instruction instructionCopy = instructions;

                        for (LogicalEntity& inst : instructionCopy.data) {
                            inst.signature = ce::replaceKeysInString(inst.signature, markerMap);
                            for (std::string& elem : inst.elements) {
                                elem = ce::replaceKeysInString(elem, markerMap);
                            }
                        }

                        mb.overallHashMemory.admissionMapIntegration[ExpressionWithValidity(keyString, validityName)][instructionCopy];
                    }

                }

                // C2. Check Elements (Only for "existence" or "and")
                /*
                 if (le.category == "existence" || le.category == "and") {
                    for (const std::string& element : le.elements) {

                        // Exclusion: Do not process if the element itself is an implication expression
                        if (element.size() >= 12 && element.substr(0, 12) == "(implication") {
                            continue;
                        }

                        std::vector<std::string> eArgs = ce::getArgs(element);
                        std::string nonUArg;
                        int countNonU = 0;

                        for (const std::string& a : eArgs) {
                            if (!(a.size() >= 2 && a[0] == 'u' && a[1] == '_')) {
                                nonUArg = a;
                                countNonU++;
                            }
                        }

                        if (countNonU == 1) {
                            std::map<std::string, std::string> markerMap;
                            markerMap[nonUArg] = "marker";
                            std::string keyString = ce::replaceKeysInString(element, markerMap);

                            // Create copy for admission map
                            std::vector<LogicalEntity> instructionCopy = instructions;
                            for (LogicalEntity& inst : instructionCopy) {
                                inst.signature = ce::replaceKeysInString(inst.signature, markerMap);
                                for (std::string& elem : inst.elements) {
                                    elem = ce::replaceKeysInString(elem, markerMap);
                                }
                            }
                            mb.overallHashMemory.admissionMapIntegration[keyString].insert(instructionCopy);
                        }
                    }
                }
                */
            }
        }

        //#pragma optimize("", off)

        void prepareIntegration(const std::string& expression,
            const std::set<std::string>& unchangeableArgs,
            Memory& mb,
            std::string validityName)
        {

            std::vector<std::string> argsCheck = ce::getArgs(expression);
            bool hasMarker = false;
            for (const std::string& arg : argsCheck) {
                if (arg == "marker") {
                    hasMarker = true;
                    break;
                }
            }

            if (!hasMarker) {
                ExpressionWithValidity ev(expression, validityName);
                ExpressionWithValidity eMain(expression, "main");
                if (mb.integrationPrepared.find(ev) != mb.integrationPrepared.end()) {
                    return;
                }

                if (mb.integrationPrepared.find(eMain) != mb.integrationPrepared.end()) {
                    return;
                }
            }

            //return;
            std::string core = ce::extractExpressionUniversal(expression);
            auto it = compiledExpressions.find(core);
            if (it->second.category == "atomic") {
                //return;
            }

			disintegrateExprHypothetically(expression, mb, validityName);


            mb.integrationStartIntMap[expression] = mb.startInt;

            // --- New Logic Start ---
            std::vector<std::string> currentArgs = ce::getArgs(expression);
            std::map<std::string, std::string> markerReplacementMap;

            for (const std::string& arg : currentArgs) {
                // If arg is NOT in unchangeableArgs, it is changeable
                if (unchangeableArgs.find(arg) == unchangeableArgs.end()) {
                    markerReplacementMap[arg] = "marker";
                }
            }

            std::string replacedExpr = ce::replaceKeysInString(expression, markerReplacementMap);

            ExpressionWithValidity rev(replacedExpr, validityName);
            ExpressionWithValidity reMain(replacedExpr, "main");
            if (mb.integrationPreparedMarker.find(rev) != mb.integrationPreparedMarker.end()) {
                return;
            }

            if (mb.integrationPreparedMarker.find(reMain) != mb.integrationPreparedMarker.end()) {
                return;
            }

            mb.integrationPreparedMarker.insert(rev);

            // --- New Logic End ---

            std::vector<std::string> tempArgs = ce::getArgs(expression);
            for (auto it = tempArgs.begin(); it != tempArgs.end();) {
                if (unchangeableArgs.count(*it)) {
                    it = tempArgs.erase(it);
                }
                else {
                    ++it;
                }
            }
            if (!parameters.incubator_mode) {
                assert(tempArgs.size() <= 1);
            }


            Instruction instructions;

            // 1. Separate Unchangeables and Changeables
            std::vector<std::string> args = ce::getArgs(expression);
            std::map<std::string, std::string> unchangeableMap;
            std::map<std::string, std::string> changeableMap;

            for (const std::string& arg : args) {
                // Check if already processed (avoid duplicates from getArgs)
                if (unchangeableMap.count(arg) || changeableMap.count(arg)) {
                    continue;
                }

                if (unchangeableArgs.find(arg) != unchangeableArgs.end()) {
                    // Unchangeable Mapping
                    bool alreadyU = (arg.size() >= 2 && arg[0] == 'u' && arg[1] == '_');

                    if (!alreadyU)
                    {
                        unchangeableMap[arg] = "u_" + arg;
                    }
                }
                else {
                    // Changeable Mapping
                    std::string freshVar = "pi_lev_" + std::to_string(mb.level) + "_" + std::to_string(mb.startIntPi);
                    changeableMap[arg] = freshVar;
                    mb.startIntPi++;
                }
            }

            // 2. Replace Unchangeables in Expression
            std::string expressionWithUnchangeables = ce::replaceKeysInString(expression, unchangeableMap);

            // 3. Generate initial instructions 
            prepareIntegrationCore(expressionWithUnchangeables, instructions, mb, expression);

            // 4. Call Core2 to handle changeables and admission logic
            prepareIntegrationCore2(instructions, changeableMap, mb, validityName, expressionWithUnchangeables);
        }

        void cleanInstruction(Instruction& instruction, const std::string& expression) {
            // 1. Find the LogicalEntity where signature matches the expression
            auto it = std::find_if(instruction.data.begin(), instruction.data.end(),
                [&expression](const LogicalEntity& le) {
                    return le.signature == expression;
                });

            // If not found, exit
            if (it == instruction.data.end()) {
                return;
            }

            // 2. Cache properties needed for recursion before deletion
            // (Iterators are invalidated after erase)
            std::string category = it->category;
            std::vector<std::string> elements = it->elements;

            // 3. Remove the entity from the instruction
            instruction.data.erase(it);

            // 4. Handle recursion for container categories
            if (category == "existence" || category == "and") {
                for (const std::string& element : elements) {

                    // Check if this element is still in use by any *other* LogicalEntity
                    // in the instruction that is also a container ("existence" or "and").
                    bool isUsedElsewhere = false;
                    for (const auto& remainingEntity : instruction.data) {
                        if (remainingEntity.category == "existence" || remainingEntity.category == "and") {
                            const std::vector<std::string>& otherElements = remainingEntity.elements;
                            // Check if the element exists in this entity's elements list
                            if (std::find(otherElements.begin(), otherElements.end(), element) != otherElements.end()) {
                                isUsedElsewhere = true;
                                break;
                            }
                        }
                    }

                    // If the element is not used by any other container, recursively remove it
                    if (!isUsedElsewhere) {
                        cleanInstruction(instruction, element);
                    }
                }
            }
        }

        void updateAdmissionMapIntegration(const std::string& expression,
            Memory& mb,
            const std::string& validityName)
        {
            if (validityName != "main") {
                return;
            }

            std::vector<std::string> args = ce::getArgs(expression);
            std::set<std::string> uniqueArgs(args.begin(), args.end());

            for (const std::string& argName : uniqueArgs) {

                std::map<std::string, std::string> markerReplacementMap;
                markerReplacementMap[argName] = "marker";

                std::string replacedCopy = ce::replaceKeysInString(expression, markerReplacementMap);
                replacedCopy = addMissingU(replacedCopy);

                auto it = mb.overallHashMemory.admissionMapIntegration.find(ExpressionWithValidity(replacedCopy, validityName));
                if (it != mb.overallHashMemory.admissionMapIntegration.end()) {

                    std::map<std::string, std::string> changeableMap;
                    std::string uVar = "u_" + argName;
                    changeableMap["marker"] = uVar;

                    // Get a reference to the set (the value in the map)
                    auto& entryMap = it->second;

                    // [FIX] SNAPSHOT: Iterate over a copy to handle concurrent insertions safely
                    std::vector<std::pair<Instruction, std::set<std::string>>> snapshot(entryMap.begin(), entryMap.end());

                    for (const auto& pair : snapshot) {
                        const Instruction& instruction = pair.first;
                        const std::set<std::string>& appliedVars = pair.second;

                        // 1. Check history
                        if (appliedVars.count(argName)) continue;

                        // 2. Update LIVE map
                        entryMap[instruction].insert(argName);

                        // 3. Run Logic
                        Instruction instructionsCopy = instruction;
                        auto it = mb.overallHashMemory.admissionMapIntegration.find(ExpressionWithValidity(replacedCopy, validityName));
                        cleanInstruction(instructionsCopy, replacedCopy);
                        prepareIntegrationCore2(instructionsCopy, changeableMap, mb, validityName, "");
                    }
                }
            }
        }

        std::string prefixArgumentsWithU(const std::string& expr);

        std::string removeUPrefixFromArguments(const std::string& expr);

        std::string addMissingU(const std::string& expr);

        std::vector<std::string> listLastRemovedArgsLE(const LogicalEntity& le);

        std::string expandSignature(const LogicalEntity& le);

        void disintegrateExprCore2(const std::string& expr,
            std::vector<LogicalEntity>& instructions,
            Memory& memoryBlock,
            int iteration,
            std::map<std::string, std::pair<std::set<std::string>, std::set<std::string>>>& collected,
            std::map<std::string, std::vector<std::string>>& newVarMap,
            std::string validityName);

        std::tuple<std::set<std::string>, std::set<std::string>, int>
            disintegrateExpr2(const std::string& expr,
                Memory& memoryBlock,
                int iteration,
                bool forceDeep,
                std::string validityName);

        void makeAdmissionKeys(const std::vector<std::string>& implication,
            const std::string& expression,
            HashMemory& localMemory,
            std::string validityName);


        




        // Turn optimizations OFF for just this section
//#pragma optimize("", off)

        inline bool filterIterations(const std::string& expr2, const EquivalenceClass& eqClass) {

            // Precompiled regexes
            // Priority 1: int_lev_<digits>_<digits>
            static const std::regex PATTERN_INT(R"(int_lev_\d+_\d+)");
            // Priority 2: it_<digits>_lev_<digits>_<digits>
            static const std::regex PATTERN_IT(R"(it_\d+_lev_\d+_\d+)");

            // Sets to hold matching variables from the equivalence class
            std::set<std::string> itMatches;
            std::set<std::string> intMatches;

            // 1. Analyze the Equivalence Class to find special variables
            for (std::set<std::string>::const_iterator it = eqClass.variables.begin();
                it != eqClass.variables.end(); ++it) {
                const std::string& var = *it;
                if (std::regex_match(var, PATTERN_INT)) {
                    intMatches.insert(var);
                }
                else if (std::regex_match(var, PATTERN_IT)) {
                    itMatches.insert(var);
                }
            }

            // 2. Determine the "Canonical" (Allowed) Variable
            std::string canonicalVar;

            if (!intMatches.empty()) {
                // Rule: int_ has higher priority. If present, pick the minimum int_ variable.
                canonicalVar = *intMatches.begin(); // std::set orders lexicographically
            }
            else if (!itMatches.empty()) {
                // Rule: If no int_, fallback to it_. Pick the minimum it_ variable.
                canonicalVar = *itMatches.begin();
            }
            else {
                // No special variables in this class -> No filtering needed.
                return true;
            }

            // 3. Scan the Expression for "Forbidden" Variables
            // Any variable from intMatches OR itMatches found in expr2 MUST be the canonicalVar.

            // Check for INT matches in expression
            {
                std::sregex_iterator i(expr2.begin(), expr2.end(), PATTERN_INT);
                std::sregex_iterator end;
                for (; i != end; ++i) {
                    const std::string match = i->str();
                    // If the match is part of the equivalence class...
                    if (intMatches.find(match) != intMatches.end()) {
                        // ...it must be the canonical one.
                        if (match != canonicalVar)
                        {
                            return false;
                        }
                            
                    }
                }
            }

            // Check for IT matches in expression
            {
                std::sregex_iterator i(expr2.begin(), expr2.end(), PATTERN_IT);
                std::sregex_iterator end;
                for (; i != end; ++i) {
                    const std::string match = i->str();
                    // If the match is part of the equivalence class...
                    if (itMatches.find(match) != itMatches.end()) {
                        // ...it must be the canonical one.
                        // (If canonicalVar is an int_, any it_ here will fail this check)
                        if (match != canonicalVar) 
                            return false;
                    }
                }
            }

            return true;
        }



        inline std::set<std::string> reduceEqClass(const std::set<std::string>& vars,
            const Memory& mb,
            const std::string& validityName)
        {
            std::set<std::string> result;

            for (const auto& var : vars) {
                // Check if the variable is marked as "weak" (renamed/deprecated)
                // ExpressionWithValidity key(var, validityName);
                if (mb.weakVariables.find(ExpressionWithValidity(var, validityName)) == mb.weakVariables.end()) {
                    result.insert(var);
                }
            }

            // Returns only the "strong" variables. 
            // If all variables are weak, this returns an empty vector.
            return result;
        }

//#pragma optimize("", off)


        inline void applyEquivalenceClass(const EquivalenceClass& clss,
            const EncodedExpression expr2,
            Memory& memoryBlock,
            const std::set<int>& levels,
            std::vector<std::string>& newStatements,
            std::string validityName) {

            if (isEquality(expr2.original)) {
                return; // Skip equalities
			}

            // expr_levels_map: new_expr -> levels
            std::map<std::string, std::set<int> > exprLevelsMap;
            // expr_origin_map (local): new_expr -> sorted equality facts
            std::map<std::string, std::vector<std::string> > exprOriginMapLocal;

            // Cache args once
            const std::vector<std::string> argsExpr = ce::getArgs(expr2.original);

            // Indices of args that are in the equivalence class
            std::vector<int> indices;
            indices.reserve(argsExpr.size());
            for (std::size_t i = 0; i < argsExpr.size(); ++i) {
                if (clss.variables.find(argsExpr[i]) != clss.variables.end()) {
                    indices.push_back(static_cast<int>(i));
                }
            }

            // Reduced eq class as a list
            const std::set<std::string> reducedSet = this->reduceEqClass(clss.variables, memoryBlock, validityName);
            std::vector<std::string> eqList;
            eqList.reserve(reducedSet.size());
            for (std::set<std::string>::const_iterator it = reducedSet.begin(); it != reducedSet.end(); ++it) {
                eqList.push_back(*it);
            }

            // Base expression and wrappers
            std::string baseExpr;
            std::string wrapLeft;
            std::string wrapRight;
            if (!expr2.original.empty() && expr2.original[0] == '(') {
                baseExpr = ce::extractExpression(expr2.original);
                wrapLeft = "(";
                wrapRight = ")";
            }
            else {
                // must start with "!("
                assert(expr2.original.size() >= 2 && expr2.original[0] == '!' && expr2.original[1] == '(');
                baseExpr = ce::extractExpressionFromNegation(expr2.original);
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
                                if (parameters.trackHistory) {
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
						const ExpressionWithValidity newExprEnc(newExpr, expr2.validityName);
                        exprLevelsMap[newExpr] = newLevels;

                        if (parameters.trackHistory) {
                            if (parameters.compressor_mode || memoryBlock.exprOriginMap.find(newExprEnc) == memoryBlock.exprOriginMap.end()) {
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
				EncodedExpression appliedEnc(applied, expr2.validityName);
				ExpressionWithValidity appliedWithValidity(applied, expr2.validityName);

                if (!parameters.compressor_mode && memoryBlock.statementLevelsMap.find(appliedEnc) != memoryBlock.statementLevelsMap.end()) {
                    continue;
                }

                const int mn = this->extractMaxIterationNumber(applied);
                if (mn != -1 && mn > parameters.maxIterationNumberVariable) {
                    continue;
                }
                if (this->countPatternOccurrences(applied, memoryBlock.overallHashMemory) >
                    parameters.maxNumberSecondaryVariables) {
                    continue;
                }

                // Only if neither present and max(levels) == memoryBlock.level
                if (memoryBlock.statementLevelsMap.find(appliedEnc) == memoryBlock.statementLevelsMap.end() &&
                    memoryBlock.exprOriginMap.find(appliedWithValidity) == memoryBlock.exprOriginMap.end()) {

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

                    if (appliedEnc.original == "(in3[2,2,2,4])" and memoryBlock.exprKey == "(AnchorIncubator[1,2,3,4,5,6,7,8,9,10])")
                    {
                        int test = 0;
                        test++;
                    }

                    memoryBlock.statementLevelsMap[appliedEnc] = lvls;
                    memoryBlock.intKnownStatements.insert(packStatementKey(
                        memoryBlock.nameMap.encode(appliedEnc.original),
                        memoryBlock.nameMap.encode(appliedEnc.validityName)));
                    memoryBlock.encodedStatements.push_back(appliedEnc);
                    memoryBlock.localEncodedStatements.push_back(appliedEnc);
                    memoryBlock.localEncodedStatementsDelta.push_back(appliedEnc);
                    { IntEncodedExpr ie = encodeExpression(appliedEnc, memoryBlock.nameMap);
                      memoryBlock.intEncodedStatements.push_back(ie);
                      memoryBlock.intLocalEncodedStatements.push_back(ie);
                      memoryBlock.intLocalEncodedStatementsDelta.push_back(ie); }
                    newStatements.push_back(applied);

                    if (expr2.validityName == "main" && allowedForMail(applied, memoryBlock))
                    {
                        // Mail out statement with its levels
                        memoryBlock.mailOut.statements.insert(
                            std::make_pair(applied, lvls)
                        );
                    }


                }

                if (parameters.trackHistory) {
                    //if (memoryBlock.exprOriginMap.find(appliedWithValidity) == memoryBlock.exprOriginMap.end()) 
                    {
                        std::pair<std::string, std::vector<ExpressionWithValidity>> origin;
                        origin.first = "equality1";
                        origin.second.push_back(ExpressionWithValidity(expr2.original, expr2.validityName));

                        std::map<std::string, std::vector<std::string> >::const_iterator oit =
                            exprOriginMapLocal.find(applied);
                        if (oit != exprOriginMapLocal.end()) {
                            const std::vector<std::string>& tail = oit->second;
                            for (std::size_t i = 0; i < tail.size(); ++i)
                            {
                                origin.second.push_back(ExpressionWithValidity(tail[i], expr2.validityName));
                            }
                        }

                        // Using the helper to append the origin to the vector while respecting limits
                        addOrigin(memoryBlock.exprOriginMap, appliedWithValidity, origin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                        addOrigin(memoryBlock.mailOut.exprOriginMap, appliedWithValidity, origin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));

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



        inline void mergeTwoEquivalenceClasses(EquivalenceClass& classA,
            const EquivalenceClass& classB,
            const std::set<std::string>& eqArgs,
            const std::set<int>& levels,
            Memory& memoryBlock,
            const std::string validityName) {
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
                // The map now stores a vector of origins to allow multiple derivation paths
                std::map<ExpressionWithValidity, std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>> newOrigins = classB.equalityOriginMap;

                // Pass the compression limit so overwriteOrigins knows when to stop merging paths
                overwriteOrigins(newOrigins, classA.equalityOriginMap, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));

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
            std::map<ExpressionWithValidity, std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>> > mergedOriginMap;

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
                    if (parameters.trackHistory) {
                        if (varA != commonArg && varB != commonArg) {
                            const std::string eq1 = std::string("(=[") + varA + "," + varB + "])";
                            const std::string eq2 = std::string("(=[") + varB + "," + varA + "])";

                            std::pair<std::string, std::vector<ExpressionWithValidity>> or1;
                            or1.first = "equality2";
                            or1.second.push_back(ExpressionWithValidity(std::string("(=[") + varA + "," + commonArg + "])", validityName));
                            or1.second.push_back(ExpressionWithValidity(std::string("(=[") + commonArg + "," + varB + "])", validityName));

                            std::pair<std::string, std::vector<ExpressionWithValidity>> or2;
                            or2.first = "equality2";
                            or2.second.push_back(ExpressionWithValidity(std::string("(=[") + varB + "," + commonArg + "])", validityName));
                            or2.second.push_back(ExpressionWithValidity(std::string("(=[") + commonArg + "," + varA + "])", validityName));

							const ExpressionWithValidity eq1Enc(eq1, validityName);

                            // Check if eq1Enc exists; addOrigin handles the vector insertion and limit
                            addOrigin(mergedOriginMap, eq1Enc, or1, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                            addOrigin(memoryBlock.exprOriginMap, eq1Enc, or1, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                            addOrigin(memoryBlock.mailOut.exprOriginMap, eq1Enc, or1, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));

                            const ExpressionWithValidity eq2Enc(eq2, validityName);
                            addOrigin(mergedOriginMap, eq2Enc, or2, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                            addOrigin(memoryBlock.exprOriginMap, eq2Enc, or2, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                            addOrigin(memoryBlock.mailOut.exprOriginMap, eq2Enc, or2, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
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
                // The map now uses a vector of origins to allow the Compressor to track multiple derivation paths
                std::map<ExpressionWithValidity, std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>> tmpMap = classB.equalityOriginMap;

                // Merge origins from Class A into the temporary map, respecting the max_origins limit
                overwriteOrigins(tmpMap, classA.equalityOriginMap, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));

                classA.equalityOriginMap.swap(tmpMap);
            }
            // classA.equalityOriginMap = merged_origin_map | classA.equality_origin_map
            {
                // Update the type to a map of vectors to align with the new Compressor data structure
                std::map<ExpressionWithValidity, std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>> tmpMap = mergedOriginMap;

                // Pass the max_origins limit to overwriteOrigins to manage path merging
                overwriteOrigins(tmpMap, classA.equalityOriginMap, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));

                classA.equalityOriginMap.swap(tmpMap);
            }
        }

        inline std::vector<std::string>
            cleanUpExpressions(Memory& mb,
                const std::vector<std::string>& newStatementsIn,
                std::string validityName) {
            std::vector<std::string> newStatements = newStatementsIn;

            for (std::vector<EquivalenceClass>::const_iterator clsIt = mb.equivalenceClassesMap[validityName].begin();
                clsIt != mb.equivalenceClassesMap[validityName].end(); ++clsIt) {

                const EquivalenceClass& eqClss = *clsIt;

                // ---- local_encoded_statements ----
                std::vector<EncodedExpression> newLocalEncoded;
                newLocalEncoded.reserve(mb.localEncodedStatements.size());
                std::vector<IntEncodedExpr> newIntLocalEncoded;
                newIntLocalEncoded.reserve(mb.intLocalEncodedStatements.size());
                for (std::size_t i = 0; i < mb.localEncodedStatements.size(); ++i) {
                    const EncodedExpression& ee = mb.localEncodedStatements[i];
					if (ee.validityName != validityName)
                    {
                        newLocalEncoded.push_back(ee);
                        newIntLocalEncoded.push_back(mb.intLocalEncodedStatements[i]);
                        continue;
                    }
                    if (filterIterations(ee.original, eqClss)) {
                        newLocalEncoded.push_back(ee);
                        newIntLocalEncoded.push_back(mb.intLocalEncodedStatements[i]);
                    }
                }

                std::vector<EncodedExpression> newLocalEncodedDelta;
                newLocalEncodedDelta.reserve(mb.localEncodedStatementsDelta.size());
                std::vector<IntEncodedExpr> newIntLocalEncodedDelta;
                newIntLocalEncodedDelta.reserve(mb.intLocalEncodedStatementsDelta.size());
                for (std::size_t i = 0; i < mb.localEncodedStatementsDelta.size(); ++i) {
                    const EncodedExpression& ee = mb.localEncodedStatementsDelta[i];
                    if (ee.validityName != validityName)
                    {
                        newLocalEncodedDelta.push_back(ee);
                        newIntLocalEncodedDelta.push_back(mb.intLocalEncodedStatementsDelta[i]);
                        continue;
                    }
                    if (filterIterations(ee.original, eqClss)) {
                        newLocalEncodedDelta.push_back(ee);
                        newIntLocalEncodedDelta.push_back(mb.intLocalEncodedStatementsDelta[i]);
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
                            mb.statementLevelsMap.erase(*it);
                            mb.intKnownStatements.erase(packStatementKey(
                                mb.nameMap.encode(it->original),
                                mb.nameMap.encode(it->validityName)));
                        }
                    }
                }

                mb.localEncodedStatements = newLocalEncoded;
                mb.localEncodedStatementsDelta = newLocalEncodedDelta;
                mb.intLocalEncodedStatements = newIntLocalEncoded;
                mb.intLocalEncodedStatementsDelta = newIntLocalEncodedDelta;

                // ---- encoded_statements ----
                std::vector<EncodedExpression> newEncodedStatements;
                newEncodedStatements.reserve(mb.encodedStatements.size());
                std::vector<IntEncodedExpr> newIntEncodedStatements;
                newIntEncodedStatements.reserve(mb.intEncodedStatements.size());
                for (std::size_t i = 0; i < mb.encodedStatements.size(); ++i) {
                    const EncodedExpression& ee = mb.encodedStatements[i];
                    if (ee.validityName != validityName)
                    {
                        newEncodedStatements.push_back(ee);
                        newIntEncodedStatements.push_back(mb.intEncodedStatements[i]);
                        continue;
                    }
                    if (filterIterations(ee.original, eqClss)) {
                        newEncodedStatements.push_back(ee);
                        newIntEncodedStatements.push_back(mb.intEncodedStatements[i]);
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
                            mb.statementLevelsMap.erase(*it);
                            mb.intKnownStatements.erase(packStatementKey(
                                mb.nameMap.encode(it->original),
                                mb.nameMap.encode(it->validityName)));
                        }
                    }
                }

                mb.encodedStatements = newEncodedStatements;
                mb.intEncodedStatements = newIntEncodedStatements;

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

        //#pragma optimize("", off)

        std::vector<std::string>
            updateEquivalenceClasses(Memory& mb,
                const std::string& eqlty,
                const std::set<int>& levels,
                const std::pair<std::string, std::vector<ExpressionWithValidity>>& origin,
                const std::vector<std::string>& newStatementsIn,
                std::string validityName) {
            std::vector<std::string> newStatements = newStatementsIn;

            // args_list / eq_args
            const std::vector<std::string> argsList = ce::getArgs(eqlty);
            std::set<std::string> eqArgs;
            for (std::size_t i = 0; i < argsList.size(); ++i) eqArgs.insert(argsList[i]);
			const ExpressionWithValidity eqltyEnc(eqlty, validityName);

            // mirrored "(=[b,a])"
            const std::string mirrored = std::string("(=[") + argsList[1] + "," + argsList[0] + "])";
			const ExpressionWithValidity mirroredEnc(mirrored, validityName);

            // merged_class init
            EquivalenceClass mergedClass;
            mergedClass.variables = eqArgs;
            mergedClass.equalityLevelsMap[eqArgs] = levels;
            addOrigin(mergedClass.equalityOriginMap, eqltyEnc, origin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            std::pair<std::string, std::vector<ExpressionWithValidity>> mirroredOrigin;
            mirroredOrigin.first = "symmetry of equality";
            mirroredOrigin.second.push_back(ExpressionWithValidity(eqlty, validityName));
            addOrigin(mergedClass.equalityOriginMap, mirroredEnc, mirroredOrigin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));

            if (parameters.trackHistory) {
				const ExpressionWithValidity eqltyEnc(eqlty, validityName);
                // addOrigin handles the vector push_back, duplicate checking, 
                // and respects parameters.compressor_max_origins_per_expr.
                addOrigin(mb.exprOriginMap, eqltyEnc, origin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                addOrigin(mb.mailOut.exprOriginMap, eqltyEnc, origin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                const ExpressionWithValidity mirroredEnc(mirrored, validityName);
                // addOrigin handles the vector push_back and ensures we don't 
                // exceed parameters.compressor_max_origins_per_expr.
                addOrigin(mb.exprOriginMap, mirroredEnc, mirroredOrigin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                addOrigin(mb.mailOut.exprOriginMap, mirroredEnc, mirroredOrigin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            }

            // Merge all eq-classes that overlap with eq_args
            std::vector<EquivalenceClass> newClasses;
            newClasses.reserve(mb.equivalenceClassesMap[validityName].size() + 1);
            for (std::size_t i = 0; i < mb.equivalenceClassesMap[validityName].size(); ++i) {
                const EquivalenceClass& eqClss = mb.equivalenceClassesMap[validityName][i];

                bool overlaps = false;
                for (std::set<std::string>::const_iterator it = eqArgs.begin(); it != eqArgs.end(); ++it) {
                    if (eqClss.variables.find(*it) != eqClss.variables.end()) { overlaps = true; break; }
                }

                if (overlaps) {
                    this->mergeTwoEquivalenceClasses(mergedClass, eqClss, eqArgs, levels, mb, validityName);
                    mb.eqClassSttmntIndexMapMap[validityName].erase(eqClss.variables); // pop(frozenset, None)
                }
                else {
                    newClasses.push_back(eqClss);
                }
            }

            mb.eqClassSttmntIndexMapMap[validityName][mergedClass.variables] = 0;
            newClasses.push_back(mergedClass);
            mb.equivalenceClassesMap[validityName].swap(newClasses);

            // Apply merged class to all encoded statements
            const std::size_t initial_count = mb.encodedStatements.size();   // snapshot
            for (std::size_t index = 0; index < initial_count; ++index) {
                if (mb.encodedStatements[index].validityName != validityName)
                {
                    continue;
                }

                const std::string original = mb.encodedStatements[index].original; // <-- copy!
                const auto lvIt = mb.statementLevelsMap.find(mb.encodedStatements[index]);
                const std::set<int> emptyLevels;
                const std::set<int>& lvls = (lvIt != mb.statementLevelsMap.end()) ? lvIt->second : emptyLevels;

                this->applyEquivalenceClass(mergedClass, mb.encodedStatements[index], mb, lvls, newStatements, validityName);
            }


            mb.eqClassSttmntIndexMapMap[validityName][mergedClass.variables] =
                static_cast<int>(mb.encodedStatements.size());

            // Cleanup
            newStatements = this->cleanUpExpressions(mb, newStatements, validityName);
			updateWeakVariables(mb, validityName, eqlty);
            

            // expr_origin_map = merged_class.equality_origin_map | mb.expr_origin_map
            {
                // Updated to use a vector of origins
                std::map<ExpressionWithValidity, std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>> merged = mergedClass.equalityOriginMap;

                // right-precedence union: merge mb.exprOriginMap into 'merged'
                // Respects the compressor_max_origins_per_expr limit during the merge
                overwriteOrigins(merged, mb.exprOriginMap, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));

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

        //#pragma optimize("", off)

        inline std::vector<std::string>
            addStatement(const std::string& expr,
                Memory& memoryBlock,
                bool local,
                const std::set<int>& levels,
                const std::pair<std::string, std::vector<ExpressionWithValidity>>& origin,
                std::string validityName) {
            std::vector<std::string> newStatements;
            EncodedExpression encodedExpr(expr, validityName);
			ExpressionWithValidity exprWithValidity(expr, validityName);
            IntEncodedExpr ieStmt = encodeExpression(encodedExpr, memoryBlock.nameMap);
            if (this->isEquality(expr)) {
                std::vector<std::string> args = ce::getArgs(expr);
                assert(args.size() == 2);
                std::string mirrored = "(=[" + args[1] + "," + args[0] + "])";
                newStatements.push_back(mirrored);
            }

            const int maxIteration = extractMaxIterationNumber(expr);
            if (maxIteration != -1 && maxIteration > parameters.maxIterationNumberVariable) {
                return newStatements;
            }

            if (countPatternOccurrences(expr, memoryBlock.overallHashMemory) >
                parameters.maxNumberSecondaryVariables) {
                return newStatements;
            }

            if (!isEquality(expr) || parameters.skip_eq_classes) {
                // not in memory already?
                if (memoryBlock.statementLevelsMap.find(encodedExpr) == memoryBlock.statementLevelsMap.end()) {

                    // Check if any equivalence class filters the expr out
                    bool anyFilteredOut = false;
                    for (std::size_t i = 0; i < memoryBlock.equivalenceClassesMap[validityName].size(); ++i) {
                        if (!filterIterations(expr, memoryBlock.equivalenceClassesMap[validityName][i])) {
                            anyFilteredOut = true; break;
                        }
                    }

                    if (!anyFilteredOut) {
                        //in3[2,2,2,4] finder:
                        if (expr == "(in3[2,2,2,4])" && memoryBlock.exprKey == "(AnchorIncubator[1,2,3,4,5,6,7,8,9,10])")
                        {
                            int test = 0;
                            test++;
                        }
                        memoryBlock.statementLevelsMap[encodedExpr] = levels;
                        memoryBlock.intKnownStatements.insert(packStatementKey(ieStmt.originalId, ieStmt.validityId));

                        if (parameters.trackHistory) {
                            // addOrigin handles existence checks, vector push_back,
                            // and respects the compressor_max_origins_per_expr limit.
                            addOrigin(memoryBlock.exprOriginMap, exprWithValidity, origin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                            addOrigin(memoryBlock.mailOut.exprOriginMap, exprWithValidity, origin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                        }

                        memoryBlock.encodedStatements.push_back(EncodedExpression(expr, validityName));
                        { const IntEncodedExpr& ie = ieStmt;
                          memoryBlock.intEncodedStatements.push_back(ie);

                        if (local) {
                            memoryBlock.localEncodedStatements.push_back(EncodedExpression(expr, validityName));
                            memoryBlock.localEncodedStatementsDelta.push_back(EncodedExpression(expr, validityName));
                            memoryBlock.intLocalEncodedStatements.push_back(ie);
                            memoryBlock.intLocalEncodedStatementsDelta.push_back(ie);
                            newStatements.push_back(expr);

							bool afm = allowedForMail(expr, memoryBlock);
                            if (validityName == "main" && afm)
                            {
                                memoryBlock.mailOut.statements.insert(std::make_pair(expr, levels));
                            }

                            if (parameters.trackHistory) {
                                if (memoryBlock.mailOut.exprOriginMap.find(exprWithValidity) ==
                                    memoryBlock.mailOut.exprOriginMap.end()) {

                                    addOrigin(memoryBlock.mailOut.exprOriginMap, exprWithValidity, origin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                                }
                            }
                        }
                        } // close IntEncodedExpr ie scope
                    }
                }

                // Apply each equivalence class to generate more statements from expr
                if (!parameters.skip_eq_classes) {
                    for (std::size_t i = 0; i < memoryBlock.equivalenceClassesMap[validityName].size(); ++i) {
                        this->applyEquivalenceClass(memoryBlock.equivalenceClassesMap[validityName][i],
                            encodedExpr,
                            memoryBlock,
                            levels,
                            newStatements,
                            validityName);
                    }
                }
            }
            else {
                //addEquality(expr, memoryBlock, local, levels, origin, validityName);

                newStatements = this->updateEquivalenceClasses(memoryBlock, expr, levels, origin, newStatements, validityName);
				newStatements.push_back(expr); // also add the equality itself
            }

            if (!parameters.skip_eq_classes) {
                // Iteratively apply equivalence classes to any newly generated statements
                std::size_t oldSize = memoryBlock.encodedStatements.size();
                while (true) {
                    for (std::size_t c = 0; c < memoryBlock.equivalenceClassesMap[validityName].size(); ++c) {
                        const EquivalenceClass& eqc = memoryBlock.equivalenceClassesMap[validityName][c];

                        int startIndex = 0;
                        std::map<std::set<std::string>, int>::const_iterator itIdx =
                            memoryBlock.eqClassSttmntIndexMapMap[validityName].find(eqc.variables);
                        if (itIdx != memoryBlock.eqClassSttmntIndexMapMap[validityName] .end()) {
                            startIndex = itIdx->second;
                        }

                        for (int idx = startIndex; idx < static_cast<int>(memoryBlock.encodedStatements.size()); ++idx) {
                            if (memoryBlock.encodedStatements[static_cast<std::size_t>(idx)].validityName != validityName)
                            {
                                continue;
                            }

                            const std::string& s = memoryBlock.encodedStatements[static_cast<std::size_t>(idx)].original;

                            std::map<EncodedExpression, std::set<int> >::const_iterator lvIt =
                                memoryBlock.statementLevelsMap.find(memoryBlock.encodedStatements[static_cast<std::size_t>(idx)]);
                            const std::set<int> emptyLevels;
                            const std::set<int>& lvls = (lvIt != memoryBlock.statementLevelsMap.end())
                                ? lvIt->second : emptyLevels;

                            this->applyEquivalenceClass(eqc, memoryBlock.encodedStatements[static_cast<std::size_t>(idx)], memoryBlock, lvls, newStatements, validityName);
                        }

                        memoryBlock.eqClassSttmntIndexMapMap[validityName][eqc.variables] =
                            static_cast<int>(memoryBlock.encodedStatements.size());
                    }

                    if (oldSize == memoryBlock.encodedStatements.size()) {
                        break;
                    }
                    oldSize = memoryBlock.encodedStatements.size();
                }

                if (isEquality(expr)) {
                    newStatements = this->cleanUpExpressions(memoryBlock, newStatements, validityName);
                }
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

        inline std::string extractNaturalNumbersExpression(const std::string& expression)
        {
            const std::string needle = "(NaturalNumbers[";
            const std::size_t start = expression.find(needle);
            if (start == std::string::npos) {
                return {};
            }

            const std::size_t end = expression.find(']', start + needle.size());
            if (end == std::string::npos) {
                return {};
            }

            return expression.substr(start, end - start + 1);
        }


        inline std::string findZeroArgNameFromAnchor(const std::string& anchorExpression)
        {
            std::string fullMPL = expandExpr(anchorExpression);

			std::string naturalNumbersExpression = extractNaturalNumbersExpression(fullMPL);
			std::vector<std::string> args = ce::getArgs(naturalNumbersExpression);

            return args[1];
        }

        inline std::string findSArgNameFromAnchor(const std::string& anchorExpression)
        {
            std::string fullMPL = expandExpr(anchorExpression);

            std::string naturalNumbersExpression = extractNaturalNumbersExpression(fullMPL);
            std::vector<std::string> args = ce::getArgs(naturalNumbersExpression);

            return args[2];
        }

        inline std::string findZeroArgName(Memory& memoryBlock) {
            Memory* current = &memoryBlock;
            while (true) {
                if (!current->exprKey.empty() &&
                    current->exprKey.find("(Anchor") == 0) { // starts with

                    std::string zeroArg = findZeroArgNameFromAnchor(current->exprKey);
                    return zeroArg;
                }
                else if (current->parentMemory != NULL) {
                    current = current->parentMemory;
                }
                else {
                    assert(false); // not found up the chain
                }
            }
            // Unreachable, but keeps some compilers happy:
            return std::string();
        }

        inline std::string findSName(Memory& memoryBlock) {
            Memory* current = &memoryBlock;
            while (true) {
                if (!current->exprKey.empty() &&
                    current->exprKey.find("(Anchor") == 0) { // starts with

                    std::string sArg = findSArgNameFromAnchor(current->exprKey);
                    return sArg;
                }
                else if (current->parentMemory != NULL) {
                    current = current->parentMemory;
                }
                else {
                    assert(false); // not found up the chain
                }
            }
            // Unreachable, but keeps some compilers happy:
            return std::string();
        }


        inline void sendMail(const Memory& sender, const Mail& mail, unsigned coreId, const ParentChildrenMap& index, PerCoreMailboxes& boxes) const {
            if (mail.statements.empty() && mail.implications.empty() && mail.exprOriginMap.empty()) return;
            const unsigned cores = logicalCores;
            const unsigned slot = coreId;
            auto pit = index.find(const_cast<Memory*>(&sender));
            if (pit == index.end()) return;
            const std::vector<Memory*>& children = pit->second;

            for (Memory* child : children) {
                if (!child) continue;
                auto it = boxes.find(child);
                if (it == boxes.end()) continue;
                std::vector<Mail>& slots = it->second;
                if (slots.size() != cores) continue;

                Mail& dst = slots[slot];
                dst.statements.insert(mail.statements.begin(), mail.statements.end());
                dst.implications.insert(mail.implications.begin(), mail.implications.end());

                // Accumulate origins (deterministic — order-independent union)
                for (const auto& kv : mail.exprOriginMap) {
                    auto& existing = dst.exprOriginMap[kv.first];
                    for (const auto& orig : kv.second) {
                        if (std::find(existing.begin(), existing.end(), orig) == existing.end())
                            existing.push_back(orig);
                    }
                }
            }
        }







        
    };

    












// Surface API mirroring Python (camelCase):




} // namespace gl
