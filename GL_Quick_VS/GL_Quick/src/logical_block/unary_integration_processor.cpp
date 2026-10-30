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
#include "unary_integration_processor.hpp"
#include "../memory_infra/unary_composition.hpp"
namespace gl {
/// @brief Bind inputs for unary integration processor.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param canonicalTemplates Borrowed CanonicalTemplateMatcher interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param unaryMemory Borrowed UnaryRuleMemory interface or state; must outlive this component.
/// @param hashRuleInstaller Borrowed HashRuleInstaller interface or state; must outlive this component.
/// @param rejectedSourceReplay Borrowed RejectedSourceReplay interface or state; must outlive this component.
/// @param integrationProcessor Existing shared state or operation interface, borrowed for this component lifetime.
/// @return None; construction binds the supplied references.
UnaryIntegrationProcessor::UnaryIntegrationProcessor(CanonicalTemplateMatcher& canonicalTemplates, Canonicalizer& canonicalizer, const ProverParameters& parameters, UnaryRuleMemory& unaryMemory, HashRuleInstaller& hashRuleInstaller, RejectedSourceReplay& rejectedSourceReplay, IntegrationProcessor& integrationProcessor) : canonicalTemplates(canonicalTemplates), canonicalizer(canonicalizer), parameters(parameters), unaryMemory(unaryMemory), hashRuleInstaller(hashRuleInstaller), rejectedSourceReplay(rejectedSourceReplay) , integrationProcessor(integrationProcessor) {}

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
void UnaryIntegrationProcessor::drainUnaryComposition(Memory& mb) {
        auto& store = mb.lbMemory.unaryComposition;
        assert(!store.publishing && !store.rebuilding);
        const unsigned stringSlot = gl::WorkerContext::coreId() >= 0 ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : scratchArenas().slotCount()-1;
        const unsigned pageSlot = gl::WorkerContext::coreId() >= 0 ? static_cast<unsigned>(gl::WorkerContext::coreId())
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
                    hashRuleInstaller.removeRuleFromHashMemory(mb, *instance, proof, scope, visited, dropped, true);
                mb.expandedImplications.erase(LbStatePairKey{
                    mb.lbStateInterner.lookup(proof), mb.lbStateInterner.lookup(scope)});
            }
            for (auto* instance : instances) gl::HashRuleInstaller::finishRuleRemovals(*instance, dropped, pages);
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
                hashRuleInstaller.decomposeImplicationForInstall(proof, strings, chain, chainCount, head, remaining, remainingCount);
                int levels[256]; const int levelCount=levelsAt(store.sources, i, levels);
                for (auto* instance : instances) {
                    if (instance == &mb.workingMemory && source.local) continue;
                    if ((instance == &mb.localHashMemory || instance == &mb.localHashMemoryDelta) && !source.local) continue;
                    hashRuleInstaller.addToHashMemory(chain,chainCount,head,remaining,remainingCount,mb,*instance,
                        levels,levelCount,proof,parameters.maxIterationNumberVariable,parameters.standardMaxSecondaryNumber,
                        false,parameters.minNumOperatorsKey,StrSpan("implication",11),true,proof,scope);
                }
                mb.expandedImplications.mint(LbStatePairKey{mb.lbStateInterner.encode(proof),mb.lbStateInterner.encode(scope)});
            }
            store.rules.resetToFresh(); store.ruleInfo.clear(); store.pending.clear();
            store.heads.resetToFresh(); store.premises.resetToFresh(); store.generated.resetToFresh();
            store.admissionTransfers.resetToFresh();
            store.consumerDemands.resetToFresh();store.observedDemands.resetToFresh();
            store.consumerGoals.resetToFresh();store.observedGoals.resetToFresh();
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
                unaryMemory.rememberUnaryImplication(mb, premiseRun, premiseCount, head, proof, proof, scope, levels, n, source.local != 0);
            }
            store.publishing = false;
            store.publishingOwner = 0;
        }
        unaryMemory.wakeUnaryCompositionDemands(mb);
        unaryMemory.wakeUnaryCompositionGoals(mb);
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
        int32_t installed = 0;
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
            if(!demanded && !unaryMemory.unaryCompositionGoalOpen(mb,rightId,scopeId))return;
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
            const ScratchString resultStorage=gl::ImplicationBuilder::reconstructImplicationFullBindScratch(strings,resultPremises,resultCount,head);
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
            hashRuleInstaller.decomposeImplicationForInstall(result,strings,premiseRun,premiseCount,installedHead,remaining,remainingCount);
            assert(premiseCount==producerCount);
            for(auto* instance:instances)
                hashRuleInstaller.addToHashMemory(premiseRun,premiseCount,installedHead,remaining,remainingCount,mb,*instance,
                    levelRun,levelCount,result,parameters.maxIterationNumberVariable,
                    parameters.standardMaxSecondaryNumber,false,parameters.minNumOperatorsKey,
                    StrSpan("implication",11),true,result,scope);
            store.publishing=false;
            store.publishingOwner=0;
            mb.expandedImplications.mint(LbStatePairKey{mb.lbStateInterner.encode(result),mb.lbStateInterner.encode(scope)});
            ++installed;
        };
        for(int32_t at=0;at<store.pending.size();++at) {
            const int32_t id=store.pending[at];
            const UnaryRuleInfo info=store.ruleInfo[id-1];
            const bool consumer=store.consumerDemands.lookup(id)!=0 || store.consumerGoals.lookup(id)!=0;
            const int32_t producerKey=consumer?store.heads.lookup(signature(info.premise)):0;
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
        gl::HashRuleIndex::closeRuleIndexStaging(mb);
        // Composed rules can conclude an open goal; CNFE visits them for every
        // goal now, as the door does after its own implication installs.
        if (installed > 0) {
            RT_SCOPE_HERE("UNARY_COMPOSITION_GOAL_RESCAN");
            integrationProcessor.rescanGoalsAgainstNewChains(mb);
        }
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
void UnaryIntegrationProcessor::transferUnaryIntegrationDemands(Memory& mb) {
        if (parameters.ban_disintegration) return;
        auto& store=mb.lbMemory.unaryComposition;
        const uint32_t demandEpoch=mb.overallHashMemory.admissionMapIntegration.inner().insertEpoch();
        const uint32_t ruleEpoch=store.rules.insertEpoch();
        if(store.transferWalked && demandEpoch==store.transferDemandEpoch
            && ruleEpoch==store.transferRuleEpoch)return;
        store.transferWalked=true;
        store.transferDemandEpoch=demandEpoch;
        store.transferRuleEpoch=ruleEpoch;
        const unsigned ss=gl::WorkerContext::coreId()>=0 ? static_cast<unsigned>(gl::WorkerContext::coreId()):scratchArenas().slotCount()-1;
        const unsigned gs=gl::WorkerContext::coreId()>=0 ? static_cast<unsigned>(gl::WorkerContext::coreId()):genScratchArenas().slotCount()-1;
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
                if(!unaryMemory.matchUnaryAdmissionHead(head,demandText,mapping,mappingCount,witness))continue;
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
                    canonicalizer.initializeCanonicalScope(mb,scopeId);
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
                const int normalizedFixedCount=canonicalTemplates.canonicalAdmissionChain(mb,bindingChain,1,fixed,fixedCount,
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
                if(mb.overallHashMemory.admissionMap.runLen(mb.overallHashMemory.admissionMap.lookup(key))==oldCount)continue;
                mb.lbMemory.canonicalUses.algebraDemands.changed(key);
                if(oldId==0){
                    mb.overallHashMemory.admissionStatusMap.upsert(key,0);
                }
                const int n=getArgsSpans(marked,premiseArgs,ExecutionParameters::MAX_ARITY);
                for(int i=0;i<n;++i)if(!equalSpans(premiseArgs[i],StrSpan("marker",6)))
                    mb.overallHashMemory.varsInAdmissionMapKeys.mint(mb.templateInterner.encode(premiseArgs[i]));
                rejectedSourceReplay.revisitRejected2(marked,mb,scope);
                rejectedSourceReplay.revisitRejectedOrdis(marked,mb,scope);
            }
        }
    }
}
