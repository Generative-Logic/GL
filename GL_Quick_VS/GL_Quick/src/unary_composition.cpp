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
#include "prover.hpp"
#include "memory_infra/unary_composition.hpp"
#include "infra/rt_tracker.hpp"

namespace gl {
    /// @brief Match a unary head against an existing integration demand.
    /// @details Only the witness becomes u_marker in the matching view. The
    /// one remaining non-u_ variable must occupy a digit port. Signed operators,
    /// fixed arguments and repetitions match exactly; the rule is not modified.
    /// @param head Original rule head.
    /// @param demand Stored integration demand with ordinary marker.
    /// @param mapping Caller array of MAX_ARITY replacements, valid on success.
    /// @param mappingCount Receives the successful replacement count.
    /// @param witness Receives the head witness variable on success.
    /// @return Whether the head satisfies the CNFE matching conditions.
    /// @invariant No fact, rule, witness or admission is created by this predicate.
    /// Successful replacement and witness views borrow the caller's input storage.
    bool ExpressionAnalyzer::matchUnaryAdmissionHead(StrSpan head, StrSpan demand,
        StrReplacement* mapping, int32_t& mappingCount, StrSpan& witness) {
        assert(head.len > 0 && demand.len > 0);
        if ((head.ptr[0]=='!') != (demand.ptr[0]=='!')
            || !equalSpans(extractExpressionUniversalSpan(head),extractExpressionUniversalSpan(demand))) return false;
        StrSpan args[ExecutionParameters::MAX_ARITY], demandArgs[ExecutionParameters::MAX_ARITY];
        const int count=getArgsSpans(head,args,ExecutionParameters::MAX_ARITY);
        const int arity=getArgsSpans(demand,demandArgs,ExecutionParameters::MAX_ARITY);
        if(count!=arity)return false;
        int marker=-1,markerCount=0;
        for(int i=0;i<arity;++i)if(equalSpans(demandArgs[i],StrSpan("marker",6))){marker=i;++markerCount;}
        if(markerCount!=1)return false;
        const ce::CoreExpressionConfig* config=coreConfig(extractExpressionUniversalSpan(head));
        if(config==nullptr)return false;
        witness=args[marker];
        if(startsWithSpan(witness,"u_",2))return false;
        mappingCount=0;
        int variableCount=0,variableSlot=-1;
        for(int i=0;i<arity;++i){
            const StrSpan matching=equalSpans(args[i],witness)?StrSpan("u_marker",8):args[i];
            const StrSpan expected=i==marker?StrSpan("u_marker",8):demandArgs[i];
            if(startsWithSpan(matching,"u_",2)){
                if(!equalSpans(matching,expected))return false;
            }else{++variableCount;variableSlot=i;}
            if(!startsWithSpan(args[i],"u_",2)){
                bool repeated=false;
                for(int j=0;j<mappingCount;++j)if(equalSpans(mapping[j].key,args[i])){
                    repeated=true;
                    if(!equalSpans(mapping[j].value,demandArgs[i]))return false;
                }
                if(!repeated)mapping[mappingCount++]={args[i],demandArgs[i]};
            }
        }
        if(variableCount!=1)return false;
        const auto* type=findDefinitionSet(config,variableSlot+1);
        return type!=nullptr && equalSpans(StrSpan(type->first),StrSpan("(1)",3));
    }

    /// @brief Enroll a consumer whose head can serve an existing demand.
    /// @details The premise index includes only qualified consumers. Exact
    /// demand keys remain attached to each consumer so joins can reject a
    /// demand that disappeared or is incompatible with the resulting scope.
    /// @param mb Resident owning block.
    /// @param ruleId Existing composition node ordinal.
    /// @param demandKey Existing integration admission key.
    /// @return Whether CNFE matching and comparable scopes permit this consumer.
    bool ExpressionAnalyzer::indexUnaryCompositionDemand(Memory& mb,int32_t ruleId,int64_t demandKey){
        auto& store=mb.lbMemory.unaryComposition;
        assert(ruleId>0 && ruleId<=store.ruleInfo.size());
        assert(mb.overallHashMemory.admissionMapIntegration.lookup(demandKey)!=0);
        const UnaryRuleInfo rule=store.ruleInfo[ruleId-1];
        if (rule.premiseCount != 1) return false;
        const StatementKey demand=Codec<StatementKey>::decode(demandKey);
        if(!mb.nameMap.comparable(rule.scope,demand.validity))return false;
        if(mb.phase3IgnoresScope(mb.nameMap.deeperOf(rule.scope,demand.validity)))return false;
        const unsigned slot=g_currentCoreId>=0?static_cast<unsigned>(g_currentCoreId):scratchArenas().slotCount()-1;
        ScratchArena& arena=scratchArenas().forSlot(slot);
        ScratchScope frame(arena);
        const StrSpan view=mb.ruleInterner.decodeView(rule.head);
        const ScratchString head=ScratchString::copyFrom(arena,view.ptr,view.len);
        const StrSpan marked=mb.templateInterner.decodeView(demand.orig);
        StrReplacement mapping[ExecutionParameters::MAX_ARITY];int32_t count=0;StrSpan witness;
        if(!matchUnaryAdmissionHead(head,marked,mapping,count,witness))return false;
        const bool first=store.consumerDemands.lookup(ruleId)==0;
        store.consumerDemands.insertSorted(ruleId,demandKey);
        if(first){
            const StrSpan premiseView=mb.ruleInterner.decodeView(rule.premise);
            const ScratchString premise=ScratchString::copyFrom(arena,premiseView.ptr,premiseView.len);
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int arity=getArgsSpans(premise,args,ExecutionParameters::MAX_ARITY);
            const int name=mb.ruleInterner.encode(extractExpressionUniversalSpan(premise));
            store.premises.insertSorted(packInt32Pair(name,2*arity+(premise[0]=='!'?1:0)),ruleId);
        }
        return true;
    }

    /// @brief Queue existing consumers when an integration demand first appears.
    /// @details Signed head indexing limits candidates. Removing a demand clears
    /// its observation receipt; reintroduction wakes consumers again. A queued
    /// consumer joins every compatible producer, including older installed rules.
    /// @param mb Resident owning block with current integration admission keys.
    /// @return Nothing.
    void ExpressionAnalyzer::wakeUnaryCompositionDemands(Memory& mb){
        auto& store=mb.lbMemory.unaryComposition;
        const auto& demands=mb.overallHashMemory.admissionMapIntegration;
        store.observedDemands.eraseIf([&](int64_t key){return demands.lookup(key)==0;});
        for(int32_t i=1;i<=demands.count();++i){
            const int64_t key=demands.keyAt(i);
            if(store.observedDemands.lookup(key)!=0)continue;
            store.observedDemands.mint(key);
            const StatementKey demand=Codec<StatementKey>::decode(key);
            const StrSpan text=mb.templateInterner.decodeView(demand.orig);
            const int name=mb.ruleInterner.lookup(extractExpressionUniversalSpan(text));
            if(name==0)continue;
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int arity=getArgsSpans(text,args,ExecutionParameters::MAX_ARITY);
            const int family=store.heads.lookup(packInt32Pair(name,2*arity+(text.ptr[0]=='!'?1:0)));
            if(family==0)continue;
            for(int32_t j=0;j<store.heads.runLen(family);++j){
                const int32_t rule=store.heads.valueAt(family,j);
                if(indexUnaryCompositionDemand(mb,rule,key))store.pending.push_back(rule);
            }
        }
    }

    /// @brief Register an actual implication producer and its root support.
    /// @details Canonical endpoints deduplicate logical work independently of
    /// the exact source proof. Each distinct supporting level run remains a node.
    /// @param mb Owning block.
    /// @param premises Complete ordered registered premise run.
    /// @param premiseCount Number of premises within the existing chain limit.
    /// @param head Registered head.
    /// @param proof Exact installed copy expression.
    /// @param parent Original owning rule before multiplication.
    /// @param scope Rule validity.
    /// @param levels Sorted supporting levels.
    /// @param levelCount Number of levels.
    /// @param local Whether this is a local rule installation.
    /// @return Nothing.
    void ExpressionAnalyzer::rememberUnaryImplication(Memory& mb, const StrSpan* premises, int premiseCount,
        StrSpan head, StrSpan proof, StrSpan parent, StrSpan scope,
        const int* levels, int32_t levelCount, bool local) {
        auto& store = mb.lbMemory.unaryComposition;
        if (store.rebuilding) return;
        assert(levelCount >= 0 && levelCount <= 256);
        const unsigned slot = g_currentCoreId >= 0 ? static_cast<unsigned>(g_currentCoreId)
            : scratchArenas().slotCount() - 1;
        ScratchArena& strings = scratchArenas().forSlot(slot);
        ScratchScope frame(strings);
        ScratchString normalizedPremise, normalizedHead;
        normalizeCompositionChain(strings, premises, premiseCount, head, normalizedPremise, normalizedHead);
        const UnaryRuleInfo info{mb.ruleInterner.encode(normalizedPremise),
            mb.ruleInterner.encode(normalizedHead), mb.ruleInterner.encode(proof), mb.nameMap.encode(scope), local ? 1u : 0u,
            premiseCount};
        int32_t key[260];
        key[0] = info.premise; key[1] = info.head; key[2] = info.scope; key[3] = levelCount;
        for (int32_t i = 0; i < levelCount; ++i) key[4+i] = levels[i];
        const StrSpan keyBytes(reinterpret_cast<const char*>(key), (4+levelCount)*4);
        const int32_t parentId = mb.ruleInterner.encode(parent);
        if (!store.publishing || store.publishingOwner != packRuleOwner(parentId, info.scope)) {
            const int32_t canonicalPremiseId = key[0], canonicalHeadId = key[1];
            key[0] = parentId; key[1] = info.proof;
            int32_t source = store.sources.lookup(keyBytes);
            if (source == 0) {
                source = store.sources.mint(keyBytes);
                assert(source == store.sourceInfo.size()+1);
                store.sourceInfo.push_back(info);
                store.sourceActive.push_back(1);
            } else {
                store.sourceActive.setAt(source-1, 1);
                if (local && !store.sourceInfo[source-1].local) {
                    auto promoted = store.sourceInfo[source-1]; promoted.local = 1;
                    store.sourceInfo.setAt(source-1, promoted);
                }
            }
            key[0] = canonicalPremiseId; key[1] = canonicalHeadId;
        }
        if (store.rules.lookup(keyBytes) != 0) return;
        const int32_t rule = store.rules.mint(keyBytes);
        assert(rule == store.ruleInfo.size()+1);
        store.ruleInfo.push_back(info);
        store.pending.push_back(rule);
        const auto signature = [&](StrSpan atom) {
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t arity = getArgsSpans(atom, args, ExecutionParameters::MAX_ARITY);
            return packInt32Pair(mb.ruleInterner.encode(extractExpressionUniversalSpan(atom)),
                2*arity+(atom.ptr[0]=='!' ? 1 : 0));
        };
        store.heads.insertSorted(signature(normalizedHead), rule);
        if (premiseCount == 1) (void)signature(normalizedPremise);
        for(int32_t i=1;i<=mb.overallHashMemory.admissionMapIntegration.count();++i)
            indexUnaryCompositionDemand(mb,rule,mb.overallHashMemory.admissionMapIntegration.keyAt(i));
    }

    /// @brief Retire independently installed roots of one removed owning rule.
    /// @details Generated-rule teardown never feeds roots back into the closure.
    /// @param mb Owning block.
    /// @param parent Owning rule removed from the normal registry.
    /// @param scope Its original validity.
    /// @return Nothing.
    void ExpressionAnalyzer::forgetUnaryImplication(Memory& mb, StrSpan parent, StrSpan scope) {
        auto& store = mb.lbMemory.unaryComposition;
        if (store.rebuilding) return;
        const int32_t parentId = mb.ruleInterner.lookup(parent), scopeId = mb.nameMap.lookup(scope);
        for (int32_t i = 1; i <= store.sources.count(); ++i) {
            const StrSpan bytes = store.sources.keyAt(i);
            int32_t header[4];
            assert(bytes.len >= 16);
            std::memcpy(header, bytes.ptr, 16);
            if (header[0] == parentId && header[2] == scopeId && store.sourceActive[i-1]) {
                store.sourceActive.setAt(i-1, 0);
                if (store.rebuild.empty()) store.rebuild.push_back(1);
            }
        }
    }

    /// @brief Compose producer chains with demanded unary consumers.
    /// @details New rules and newly observed demands enqueue both join roles;
    /// only eligible consumers enter the premise index. New
    /// results enter normal hash-rule installation, then participate in further
    /// joins. Removed roots invalidate the derived closure as a whole; rebuilding
    /// from live independent roots prevents cyclic support from becoming an axiom.
    /// Closure installs share one rule-index staging window. The independent
    /// composition indexes expose each new rule immediately; hash indexes are
    /// flushed once before returning to admission transfer or inference.
    /// @param mb Resident claimed block with flushed rule indexes.
    /// @return Nothing.
    void ExpressionAnalyzer::drainUnaryComposition(Memory& mb) {
        auto& store = mb.lbMemory.unaryComposition;
        assert(!store.publishing && !store.rebuilding);
        const unsigned stringSlot = g_currentCoreId >= 0 ? static_cast<unsigned>(g_currentCoreId)
            : scratchArenas().slotCount()-1;
        const unsigned pageSlot = g_currentCoreId >= 0 ? static_cast<unsigned>(g_currentCoreId)
            : genScratchArenas().slotCount()-1;
        ScratchArena& strings = scratchArenas().forSlot(stringSlot);
        ScratchArena& pages = genScratchArenas().forSlot(pageSlot);
        ScratchScope stringFrame(strings);
        const auto copyRule = [&](int32_t id) {
            const StrSpan view = mb.ruleInterner.decodeView(id);
            return ScratchString::copyFrom(strings, view.ptr, view.len);
        };
        const auto copyScope = [&](int32_t id) {
            const StrSpan view = mb.nameMap.decodeView(id);
            return ScratchString::copyFrom(strings, view.ptr, view.len);
        };
        const auto levelsAt = [&](const auto& map, int32_t id, int* out) {
            const StrSpan key = map.keyAt(id);
            int32_t count;
            assert(key.len >= 16);
            std::memcpy(&count, key.ptr+12, 4);
            assert(count >= 0 && count <= 256 && key.len == 16+4*count);
            std::memcpy(out, key.ptr+16, count*4);
            return count;
        };
        if (!store.rebuild.empty()) {
            store.rebuilding = true;
            LbArena removalArena(&staticMemory());
            DirtyState dirty = DirtyState::Clean;
            ColdHashSet<BytesKeyStore> visited(&removalArena, &dirty);
            ColdHashSet<PodKeyStore<int64_t>> dropped(&removalArena, &dirty);
            HashMemory* instances[] = {&mb.overallHashMemory, &mb.localHashMemory,
                &mb.localHashMemoryDelta, &mb.workingMemory};
            for (int32_t i = 1; i <= store.generated.count(); ++i) {
                ScratchScope row(strings);
                const auto owner = Codec<StatementKey>::decode(store.generated.keyAt(i));
                const ScratchString proof = copyRule(owner.orig), scope = copyScope(owner.validity);
                for (auto* instance : instances)
                    removeRuleFromHashMemory(mb, *instance, proof, scope, visited, dropped, true);
                mb.expandedImplications.erase(LbStatePairKey{
                    mb.lbStateInterner.lookup(proof), mb.lbStateInterner.lookup(scope)});
            }
            for (auto* instance : instances) finishRuleRemovals(*instance, dropped, pages);
            // An independently installed rule may have exactly the same owner
            // text as a generated rule. Restore its active supports after the
            // shared owner was removed, preserving local versus mail routing.
            for (int32_t i = 1; i <= store.sources.count(); ++i) {
                if (!store.sourceActive[i-1]) continue;
                ScratchScope row(strings);
                int32_t parentId;
                const StrSpan key = store.sources.keyAt(i);
                std::memcpy(&parentId, key.ptr, 4);
                const UnaryRuleInfo source = store.sourceInfo[i-1];
                if (store.generated.lookup(packRuleOwner(parentId, source.scope)) == 0) continue;
                const ScratchString proof = copyRule(parentId), scope = copyScope(source.scope);
                StrSpan chain[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS], remaining[ExecutionParameters::MAX_ADMISSION_REM_ARGS], head;
                int chainCount=0, remainingCount=0;
                decomposeImplicationForInstall(proof, strings, chain, chainCount, head, remaining, remainingCount);
                int levels[256]; const int levelCount=levelsAt(store.sources, i, levels);
                for (auto* instance : instances) {
                    if (instance == &mb.workingMemory && source.local) continue;
                    if ((instance == &mb.localHashMemory || instance == &mb.localHashMemoryDelta) && !source.local) continue;
                    addToHashMemory(chain,chainCount,head,remaining,remainingCount,mb,*instance,
                        levels,levelCount,proof,parameters.maxIterationNumberVariable,parameters.standardMaxSecondaryNumber,
                        false,parameters.minNumOperatorsKey,StrSpan("implication",11),true,proof,scope);
                }
                mb.expandedImplications.mint(LbStatePairKey{mb.lbStateInterner.encode(proof),mb.lbStateInterner.encode(scope)});
            }
            store.rules.resetToFresh(); store.ruleInfo.clear(); store.pending.clear();
            store.heads.resetToFresh(); store.premises.resetToFresh(); store.generated.resetToFresh();
            store.admissionTransfers.resetToFresh();
            store.consumerDemands.resetToFresh();store.observedDemands.resetToFresh();
            store.rebuild.clear();
            store.rebuilding = false;
            store.publishing = true;
            for (int32_t i = 1; i <= store.sources.count(); ++i) {
                if (!store.sourceActive[i-1]) continue;
                ScratchScope row(strings);
                const UnaryRuleInfo source = store.sourceInfo[i-1];
                const ScratchString premise = copyRule(source.premise), head = copyRule(source.head),
                    proof = copyRule(source.proof), scope = copyScope(source.scope);
                int levels[256]; const int n = levelsAt(store.sources, i, levels);
                store.publishingOwner = packRuleOwner(source.proof, source.scope);
                StrSpan premiseRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                const int premiseCount=splitByCharSpans(premise,'\n',premiseRun,ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS);
                assert(premiseCount==source.premiseCount);
                rememberUnaryImplication(mb, premiseRun, premiseCount, head, proof, proof, scope, levels, n, source.local != 0);
            }
            store.publishing = false;
            store.publishingOwner = 0;
        }
        wakeUnaryCompositionDemands(mb);
        RT_SCOPE_HERE("UNARY_COMPOSITION");
        assert(mb.ruleStagingArmDepth == 0);
        ScratchScope stagingFrame(pages);
        ++mb.ruleStagingArmDepth;
        const auto signature = [&](int32_t expression) {
            const StrSpan atom = mb.ruleInterner.decodeView(expression);
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t arity = getArgsSpans(atom, args, ExecutionParameters::MAX_ARITY);
            const int32_t name = mb.ruleInterner.lookup(extractExpressionUniversalSpan(atom));
            assert(name != 0);
            return packInt32Pair(name, 2*arity+(atom.ptr[0]=='!' ? 1 : 0));
        };
        const auto compose = [&](int32_t leftId, int32_t rightId) {
            ScratchScope pairFrame(strings);
            const UnaryRuleInfo left = store.ruleInfo[leftId-1], right = store.ruleInfo[rightId-1];
            assert(right.premiseCount==1);
            if (!mb.nameMap.comparable(left.scope, right.scope)) return;
            const NameId scopeId = mb.nameMap.deeperOf(left.scope, right.scope);
            if (mb.phase3IgnoresScope(scopeId)) return;
            const int demandRow=store.consumerDemands.lookup(rightId);
            bool demanded=false;
            if(demandRow!=0)for(int i=0;i<store.consumerDemands.runLen(demandRow);++i){
                const int64_t key=store.consumerDemands.valueAt(demandRow,i);
                const auto demand=Codec<StatementKey>::decode(key);
                if(mb.overallHashMemory.admissionMapIntegration.lookup(key)!=0
                    && mb.nameMap.comparable(scopeId,demand.validity)
                    && !mb.phase3IgnoresScope(mb.nameMap.deeperOf(scopeId,demand.validity))){demanded=true;break;}
            }
            if(!demanded)return;
            const ScratchString a=copyRule(left.premise), b=copyRule(left.head),
                c=copyRule(right.premise), d=copyRule(right.head);
            ScratchString premise, head;
            StrSpan producerPremises[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
            const int producerCount=splitByCharSpans(a,'\n',producerPremises,ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS);
            assert(producerCount==left.premiseCount);
            if (composeImplicationChain(strings,producerPremises,producerCount,b,c,d,premise,head)==UnaryCompositionMatch::incompatible) return;
            if (equalSpans(premise,head)) return;
            int levelRun[256], otherLevels[256];
            int levelCount=levelsAt(store.rules,leftId,levelRun);
            const int otherCount=levelsAt(store.rules,rightId,otherLevels);
            for(int i=0;i<otherCount;++i) levelCount=insertLevelSorted(levelRun,levelCount,otherLevels[i],256);
            int32_t key[260]={mb.ruleInterner.encode(premise),mb.ruleInterner.encode(head),scopeId,levelCount};
            for(int i=0;i<levelCount;++i)key[4+i]=levelRun[i];
            if(store.rules.lookup(StrSpan(reinterpret_cast<const char*>(key),(4+levelCount)*4))!=0)return;

            StrSpan resultPremises[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
            const int resultCount=splitByCharSpans(premise,'\n',resultPremises,ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS);
            assert(resultCount==producerCount);
            const ScratchString resultStorage=reconstructImplicationFullBindScratch(strings,resultPremises,resultCount,head);
            const StrSpan result(resultStorage);
            const ScratchString scope=copyScope(scopeId), leftProof=copyRule(left.proof),
                rightProof=copyRule(right.proof), leftScope=copyScope(left.scope), rightScope=copyScope(right.scope);
            if(parameters.trackHistory) {
                const OriginDep dependencies[2]={{leftProof,leftScope},{rightProof,rightScope}};
                const int originCap=parameters.compressor_mode
                    ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr;
                addOriginEncoded(mb.exprOriginMap,mb.originInterner,result,scope,
                    OriginTag::implicationComposition,dependencies,2,originCap);
                // Installed rules do not enter the statement delta. Their
                // proof rows must ship directly with the derived facts' mail
                // so a recipient can resolve the cited rule (I-64).
                mb.addMailOutOrigin(result,scope,OriginTag::implicationComposition,
                    dependencies,2,originCap);
            }
            store.generated.mint(packRuleOwner(mb.ruleInterner.encode(result),scopeId));
            store.publishing=true;
            store.publishingOwner=packRuleOwner(mb.ruleInterner.encode(result),scopeId);
            HashMemory* instances[]={&mb.overallHashMemory,&mb.localHashMemory,&mb.localHashMemoryDelta};
            StrSpan remaining[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
            StrSpan premiseRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS], installedHead;
            int premiseCount=0, remainingCount=0;
            decomposeImplicationForInstall(result,strings,premiseRun,premiseCount,installedHead,remaining,remainingCount);
            assert(premiseCount==producerCount);
            for(auto* instance:instances)
                addToHashMemory(premiseRun,premiseCount,installedHead,remaining,remainingCount,mb,*instance,
                    levelRun,levelCount,result,parameters.maxIterationNumberVariable,
                    parameters.standardMaxSecondaryNumber,false,parameters.minNumOperatorsKey,
                    StrSpan("implication",11),true,result,scope);
            store.publishing=false;
            store.publishingOwner=0;
            mb.expandedImplications.mint(LbStatePairKey{mb.lbStateInterner.encode(result),mb.lbStateInterner.encode(scope)});
        };
        for(int32_t at=0;at<store.pending.size();++at) {
            const int32_t id=store.pending[at];
            const UnaryRuleInfo info=store.ruleInfo[id-1];
            const int32_t producerKey=store.consumerDemands.lookup(id)!=0
                ?store.heads.lookup(signature(info.premise)):0;
            const int32_t consumerKey=store.premises.lookup(signature(info.head));
            DirtyState dirty=DirtyState::Clean;
            PagedVector<int32_t> producers(&pages,&dirty),consumers(&pages,&dirty);
            if(producerKey) for(int32_t i=0;i<store.heads.runLen(producerKey);++i)
                producers.push_back(store.heads.valueAt(producerKey,i));
            if(consumerKey) for(int32_t i=0;i<store.premises.runLen(consumerKey);++i)
                consumers.push_back(store.premises.valueAt(consumerKey,i));
            for(int32_t i=0;i<producers.size();++i)compose(producers[i],id);
            for(int32_t i=0;i<consumers.size();++i)compose(id,consumers[i]);
        }
        store.pending.clear();
        assert(mb.ruleStagingArmDepth == 1);
        --mb.ruleStagingArmDepth;
        closeRuleIndexStaging(mb);
    }

    /// @brief Transfer unary witness demands and request demanded equality copies.
    /// @details The marked head argument becomes u_marker for matching; exactly
    /// one other non-u_ digit argument remains. A successful substitution must
    /// determine every premise argument and preserve the same marked witness.
    /// Complete producer chains additionally request an existing fixed argument's
    /// alias when a positive equality connects it to the demanded marker. Other
    /// premises remain firing obligations. Copies use the deepest compatible
    /// scope and the ordinary variable-copy mail and provenance contract.
    /// Receipts process each rule/demand pair once and are cleared with closure
    /// invalidation. The original integration instruction is never modified.
    /// Every effect follows a receipt mint, and a walk mints a receipt for each
    /// pair it can match. A walk is therefore skipped while the demand key set
    /// and the rule registry (whose reset also clears the receipts) have gained
    /// no key since the previous walk started: it would visit receipted pairs
    /// only. Erased demands only shrink that pair set.
    /// @param mb Owning block with current rule and instruction registries.
    /// @return Nothing.
    void ExpressionAnalyzer::transferUnaryIntegrationDemands(Memory& mb) {
        if (parameters.ban_disintegration) return;
        auto& store=mb.lbMemory.unaryComposition;
        const uint32_t demandEpoch=mb.overallHashMemory.admissionMapIntegration.inner().insertEpoch();
        const uint32_t ruleEpoch=store.rules.insertEpoch();
        if(store.transferWalked && demandEpoch==store.transferDemandEpoch
            && ruleEpoch==store.transferRuleEpoch)return;
        store.transferWalked=true;
        store.transferDemandEpoch=demandEpoch;
        store.transferRuleEpoch=ruleEpoch;
        const unsigned ss=g_currentCoreId>=0 ? static_cast<unsigned>(g_currentCoreId):scratchArenas().slotCount()-1;
        const unsigned gs=g_currentCoreId>=0 ? static_cast<unsigned>(g_currentCoreId):genScratchArenas().slotCount()-1;
        ScratchArena& strings=scratchArenas().forSlot(ss);
        ScratchArena& pages=genScratchArenas().forSlot(gs);
        RT_SCOPE_HERE("CNFE_ADMISSION_TRANSFER");
        const int32_t demandCount=mb.overallHashMemory.admissionMapIntegration.count();
        for(int32_t demandId=1;demandId<=demandCount;++demandId) {
            ScratchScope demandFrame(strings);
            const int64_t demandKey=mb.overallHashMemory.admissionMapIntegration.keyAt(demandId);
            const StatementKey demand=Codec<StatementKey>::decode(demandKey);
            const StrSpan demandView=mb.templateInterner.decodeView(demand.orig);
            const ScratchString demandText=ScratchString::copyFrom(strings,demandView.ptr,demandView.len);
            StrSpan demandArgs[ExecutionParameters::MAX_ARITY];
            const int32_t arity=getArgsSpans(demandText,demandArgs,ExecutionParameters::MAX_ARITY);
            const StrSpan name=extractExpressionUniversalSpan(demandText);
            const int32_t nameId=mb.ruleInterner.lookup(name);
            if(nameId==0)continue;
            const int64_t signature=packInt32Pair(nameId,2*arity+(demandText[0]=='!' ? 1:0));
            const int32_t family=store.heads.lookup(signature);
            if(family==0)continue;
            const int32_t candidateCount=store.heads.runLen(family);
            for(int32_t candidate=0;candidate<candidateCount;++candidate) {
                ScratchScope candidateFrame(strings);
                const int32_t id=store.heads.valueAt(family,candidate);
                char receipt[12];std::memcpy(receipt,&demandKey,8);std::memcpy(receipt+8,&id,4);
                if(store.admissionTransfers.lookup(StrSpan(receipt,12))!=0)continue;
                store.admissionTransfers.mint(StrSpan(receipt,12));
                const UnaryRuleInfo rule=store.ruleInfo[id-1];
                if(!mb.nameMap.comparable(rule.scope,demand.validity))continue;
                const NameId scopeId=mb.nameMap.deeperOf(rule.scope,demand.validity);
                if(mb.phase3IgnoresScope(scopeId))continue;
                const StrSpan headView=mb.ruleInterner.decodeView(rule.head);
                const ScratchString head=ScratchString::copyFrom(strings,headView.ptr,headView.len);
                StrReplacement mapping[ExecutionParameters::MAX_ARITY];int32_t mappingCount=0;StrSpan witness;
                if(!matchUnaryAdmissionHead(head,demandText,mapping,mappingCount,witness))continue;
                const StrSpan premiseView=mb.ruleInterner.decodeView(rule.premise);
                const ScratchString premise=ScratchString::copyFrom(strings,premiseView.ptr,premiseView.len);
                const StrSpan scopeView=mb.nameMap.decodeView(scopeId);
                const ScratchString scope=ScratchString::copyFrom(strings,scopeView.ptr,scopeView.len);
                StrSpan premiseRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                const int chainCount=splitByCharSpans(premise,'\n',premiseRun,ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS);
                assert(chainCount==rule.premiseCount);
                for(int pi=0;pi<chainCount;++pi) {
                    if(premiseRun[pi].ptr[0]=='!' || !equalSpans(
                        extractExpressionUniversalSpan(premiseRun[pi]),StrSpan("=",1)))continue;
                    const ScratchString equality=replaceKeysScratch(strings,premiseRun[pi],mapping,mappingCount);
                    StrSpan arguments[ExecutionParameters::MAX_ARITY];
                    assert(getArgsSpans(equality,arguments,ExecutionParameters::MAX_ARITY)==2);
                    const bool firstMarker=equalSpans(arguments[0],StrSpan("marker",6));
                    const bool secondMarker=equalSpans(arguments[1],StrSpan("marker",6));
                    if(firstMarker==secondMarker)continue;
                    const StrSpan fixed=arguments[firstMarker?1:0];
                    if(!startsWithSpan(fixed,"u_",2))continue;
                    assert(fixed.len>2);
                    // Fixed source parameters are available at the source
                    // scope; substituted demand parameters at the demand scope.
                    // Their deeper compatible scope contains both contexts.
                    const StrSpan term(fixed.ptr+2,fixed.len-2);
                    initializeCanonicalScope(mb,scopeId);
                    const NameId termId=mb.nameMap.lookup(term);
                    const int bucket=mb.canonicalEquivalenceClassesMap.lookup(scopeId);
                    bool hasEquality=false;
                    if(termId!=0 && bucket!=0) {
                        for(int ci=0;ci<mb.canonicalEquivalenceClassesMap.runLen(bucket);++ci) {
                            ScratchScope viewFrame(pages);
                            int length=0;
                            const char* bytes=mb.canonicalEquivalenceClassesMap.peekRecordBytes(bucket,ci,length,pages);
                            const EquivalenceClassView view{bytes,length};
                            if(classHasMember(view,termId)) {
                                assert(view.memberCount()>1);
                                hasEquality=true;
                                break;
                            }
                        }
                    }
                    if(hasEquality)continue;
                    const int length=11+2*term.len;
                    char* buffer=strings.allocBytes(length);
                    int at=0;
                    std::memcpy(buffer+at,"(=[",3);at+=3;
                    std::memcpy(buffer+at,term.ptr,term.len);at+=term.len;
                    buffer[at++]=',';
                    std::memcpy(buffer+at,term.ptr,term.len);at+=term.len;
                    std::memcpy(buffer+at,"_copy])",7);at+=7;
                    assert(at==length);
                    const ScratchString copy=ScratchString::wrap(strings,buffer,length);
                    const int levels[]={mb.level};
                    insertInternalStatement(mb.sameIterationInternalMail,mb.nameMap,copy,scope,levels,1);
                    if(parameters.trackHistory) {
                        const int cap=parameters.compressor_mode?parameters.compressor_max_origins_per_expr:parameters.max_origin_per_expr;
                        addInternalMailOrigin(mb.sameIterationInternalMail,mb.originInterner,copy,scope,
                            OriginTag::variableCopy,nullptr,0,cap);
                    }
                    mb.mutatedThisBurst=true;
                    mb.hasWork=true;
                }
                if(rule.premiseCount!=1)continue;
                const ScratchString instantiated=replaceKeysScratch(strings,premise,mapping,mappingCount);
                StrSpan premiseArgs[ExecutionParameters::MAX_ARITY];
                const int premiseCount=getArgsSpans(instantiated,premiseArgs,ExecutionParameters::MAX_ARITY);
                bool hasMarker=false,determined=true;
                for(int i=0;i<premiseCount;++i) {
                    const bool isMarker=equalSpans(premiseArgs[i],StrSpan("marker",6));
                    hasMarker|=isMarker;
                    determined&=isMarker || startsWithSpan(premiseArgs[i],"u_",2);
                }
                if(!hasMarker || !determined)continue;
                // The admission key is marked, but its value retains an
                // ordinary rule premise: a numeric free placeholder and the
                // explicit fixed-argument run consumed by witness replay.
                const StrReplacement restoreWitness[1]={{StrSpan("marker",6),witness}};
                const ScratchString bindingPremise=replaceKeysScratch(strings,instantiated,restoreWitness,1);
                const StrSpan bindingChain[1]={bindingPremise};
                StrSpan fixed[ExecutionParameters::MAX_ADMISSION_REM_ARGS];int fixedCount=0;
                const int bindingCount=getArgsSpans(bindingPremise,premiseArgs,ExecutionParameters::MAX_ARITY);
                for(int i=0;i<bindingCount;++i)if(startsWithSpan(premiseArgs[i],"u_",2)) {
                    const StrSpan value(premiseArgs[i].ptr+2,premiseArgs[i].len-2);
                    bool repeated=false;
                    for(int j=0;j<fixedCount;++j)repeated|=equalSpans(fixed[j],value);
                    if(!repeated){assert(fixedCount<ExecutionParameters::MAX_ADMISSION_REM_ARGS);fixed[fixedCount++]=value;}
                }
                std::sort(fixed,fixed+fixedCount,[](StrSpan a,StrSpan b){return compareSpans(a,b)<0;});
                StrSpan normalized[1], normalizedFixed[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                bool excludedAnchor=false;
                const int normalizedFixedCount=canonicalAdmissionChain(mb,bindingChain,1,fixed,fixedCount,
                    scopeId,strings,normalized,normalizedFixed,excludedAnchor);
                const int normalizedCount=getArgsSpans(normalized[0],premiseArgs,ExecutionParameters::MAX_ARITY);
                StrSpan freeWitness;int freeCount=0;
                for(int i=0;i<normalizedCount;++i) {
                    bool isFixed=false;
                    for(int j=0;j<normalizedFixedCount;++j)isFixed|=equalSpans(premiseArgs[i],normalizedFixed[j]);
                    if(isFixed)continue;
                    if(freeCount==0){freeWitness=premiseArgs[i];freeCount=1;}
                    else assert(equalSpans(freeWitness,premiseArgs[i]) && "unary admission has multiple undetermined arguments");
                }
                assert(freeCount==1 && "unary admission lost its witness placeholder");
                const StrReplacement markWitness[1]={{freeWitness,StrSpan("marker",6)}};
                const ScratchString marked=replaceKeysScratch(strings,normalized[0],markWitness,1);
                const int64_t key=mintTemplateKey(mb.templateInterner,mb.nameMap,marked,scope);
                const int32_t oldId=mb.overallHashMemory.admissionMap.lookup(key);
                const int32_t oldCount=oldId==0 ? 0:mb.overallHashMemory.admissionMap.runLen(oldId);
                const int32_t premiseId=mb.valueInterner.encode(normalized[0]);
                int32_t fixedIds[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                for(int i=0;i<normalizedFixedCount;++i)fixedIds[i]=mb.valueInterner.encode(normalizedFixed[i]);
                insertAdmissionIdsBlob(mb.overallHashMemory.admissionMap,key,
                    parameters.maxIterationNumberVariable,parameters.standardMaxSecondaryNumber,false,
                    &premiseId,1,fixedIds,normalizedFixedCount,mb.valueInterner,pages);
                // The permission belongs to the tasks whose integration entries
                // demanded it: the first entry's task owns a new key, every
                // other entry reaches it (I-284). An
                // entry's task is the last field of its record.
                {
                    const ArenaOffset entryMark=pages.cursor();
                    const int32_t entryCount=mb.overallHashMemory.admissionMapIntegration.runLen(demandId);
                    for(int32_t entry=0;entry<entryCount;++entry) {
                        int32_t entryLength=0;
                        const char* entryBytes=mb.overallHashMemory.admissionMapIntegration
                            .peekRecordBytes(demandId,entry,entryLength,pages);
                        assert(entryLength>=static_cast<int32_t>(sizeof(int64_t)));
                        int64_t entryTask;
                        std::memcpy(&entryTask,entryBytes+entryLength-sizeof(int64_t),sizeof(int64_t));
                        recordIntegrationOwner(mb,mb.lbMemory.algebraKeyOwners,key,entryTask,oldId!=0 || entry>0);
                    }
                    pages.popTo(entryMark);
                }
                if(mb.overallHashMemory.admissionMap.runLen(mb.overallHashMemory.admissionMap.lookup(key))==oldCount)continue;
                mb.lbMemory.canonicalUses.algebraDemands.changed(key);
                if(oldId==0){
                    mb.overallHashMemory.admissionStatusMap.upsert(key,0);
                }
                const int n=getArgsSpans(marked,premiseArgs,ExecutionParameters::MAX_ARITY);
                for(int i=0;i<n;++i)if(!equalSpans(premiseArgs[i],StrSpan("marker",6)))
                    mb.overallHashMemory.varsInAdmissionMapKeys.mint(mb.templateInterner.encode(premiseArgs[i]));
                revisitRejected2(marked,mb,scope);
                revisitRejectedOrdis(marked,mb,scope);
            }
        }
    }
}
