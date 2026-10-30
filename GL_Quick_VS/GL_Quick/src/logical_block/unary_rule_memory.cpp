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
#include "unary_rule_memory.hpp"
#include "../memory_infra/unary_composition.hpp"
#include "../proof/worker_context.hpp"
namespace gl {
/// @brief Bind definitions used by unary-rule demand matching.
/// @details Source and consumer records remain in the supplied logical block.
/// @param definitions Read-only operator configurations and type metadata.
/// @return Constructed unary source-memory interface.
UnaryRuleMemory::UnaryRuleMemory(const CompiledDefinitionRegistry& definitions) : definitions(definitions) {}

/// @brief Match a unary head against an existing integration demand.
        /// @details Only the witness variable becomes u_marker in the matching
        /// view; exactly one non-u_ digit variable remains. All fixed arguments,
        /// repetitions and polarity retain their ordinary classifications.
        /// @param head Rule head with its bound variables unchanged.
        /// @param demand Stored marked integration demand.
        /// @param mapping Caller array of MAX_ARITY replacements, valid on success.
        /// @param mappingCount Receives the successful replacement count.
        /// @param witness Receives the head's witness variable on success.
        /// @return True exactly when the CNFE head conditions match.
        /// @invariant Matching does not publish a fact, rule or admission key;
        /// returned views borrow the caller's head and demand storage.
bool UnaryRuleMemory::matchUnaryAdmissionHead(StrSpan head, StrSpan demand,
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
        const ce::CoreExpressionConfig* config=definitions.coreConfig(extractExpressionUniversalSpan(head));
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
        const auto* type=gl::CompiledDefinitionRegistry::findDefinitionSet(config,variableSlot+1);
        return type!=nullptr && equalSpans(StrSpan(type->first),StrSpan("(1)",3));
    }

/// @brief Index a rule as a consumer for one existing demand.
        /// @details Matching rules enter the consumer-premise index once; exact
        /// demand keys retain the scope needed to test a proposed composition.
        /// @param mb Resident owning block.
        /// @param ruleId Existing composition rule ordinal.
        /// @param demandKey Existing integration admission key.
        /// @return Whether this rule can serve the demand under CNFE matching.
bool UnaryRuleMemory::indexUnaryCompositionDemand(Memory& mb,int32_t ruleId,int64_t demandKey) {
        auto& store=mb.lbMemory.unaryComposition;
        assert(ruleId>0 && ruleId<=store.ruleInfo.size());
        assert(mb.overallHashMemory.admissionMapIntegration.lookup(demandKey)!=0);
        const UnaryRuleInfo rule=store.ruleInfo[ruleId-1];
        if (rule.premiseCount != 1) return false;
        const StatementKey demand=Codec<StatementKey>::decode(demandKey);
        if(!mb.nameMap.comparable(rule.scope,demand.validity))return false;
        if(mb.phase3IgnoresScope(mb.nameMap.deeperOf(rule.scope,demand.validity)))return false;
        const unsigned slot=gl::WorkerContext::coreId()>=0?static_cast<unsigned>(gl::WorkerContext::coreId()):scratchArenas().slotCount()-1;
        ScratchArena& arena=scratchArenas().forSlot(slot);
        ScratchScope frame(arena);
        const StrSpan view=mb.ruleInterner.decodeView(rule.head);
        const ScratchString head=ScratchString::copyFrom(arena,view.ptr,view.len);
        const StrSpan marked=mb.templateInterner.decodeView(demand.orig);
        StrReplacement mapping[ExecutionParameters::MAX_ARITY];int32_t count=0;StrSpan witness;
        if(!matchUnaryAdmissionHead(head,marked,mapping,count,witness))return false;
        const bool first=store.consumerDemands.lookup(ruleId)==0;
        store.consumerDemands.insertSorted(ruleId,demandKey);
        if(first)indexUnaryConsumerPremise(mb,ruleId);
        return true;
    }

/// @brief Wake installed consumer rules for newly observed demands.
        /// @details The signed head index limits candidates. Removed demand
        /// receipts retire so a reintroduced demand creates a fresh wake event.
        /// @param mb Resident owning block with current admission maps.
        /// @return Nothing; eligible existing consumers enter the pending queue.
void UnaryRuleMemory::wakeUnaryCompositionDemands(Memory& mb) {
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

/// @brief Register an actual implication as a composition producer.
        /// @details Canonical endpoint identity is distinct from its exact proof citation.
        /// @param mb Owning block.
        /// @param premises Registered ordered premise run.
        /// @param premiseCount Number of registered premises.
        /// @param head Registered head.
        /// @param proof Exact installed copy expression.
        /// @param parent Exact owning rule before multiplication.
        /// @param scope Original validity scope.
        /// @param levels Sorted supporting levels.
        /// @param levelCount Number of supporting levels.
        /// @param local Whether the installation belongs to the local rule instances.
        /// @return Nothing.
void UnaryRuleMemory::rememberUnaryImplication(Memory& mb, const StrSpan* premises, int premiseCount,
        StrSpan head, StrSpan proof, StrSpan parent, StrSpan scope,
        const int* levels, int32_t levelCount, bool local) {
        auto& store = mb.lbMemory.unaryComposition;
        if (store.rebuilding) return;
        assert(levelCount >= 0 && levelCount <= 256);
        const unsigned slot = gl::WorkerContext::coreId() >= 0 ? static_cast<unsigned>(gl::WorkerContext::coreId())
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
        for(int32_t i=1;i<=mb.intToBeProved.count();++i)
            indexUnaryCompositionGoal(mb,rule,mb.intToBeProved.keyAt(i));
    }

/// @brief Retire the roots owned by a removed implication.
        /// @details The next drain rebuilds derived closure from remaining roots.
        /// @param mb Owning block.
        /// @param parent Removed owning rule.
        /// @param scope Its exact validity.
        /// @return Nothing.
void UnaryRuleMemory::forgetUnaryImplication(Memory& mb, StrSpan parent, StrSpan scope) {
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
/// @brief Enter an enrolled unary consumer into the premise join index.
        /// @details The key is the premise's signed operator signature, the
        /// value the rule id; runs are sorted and unique, so a consumer
        /// enrolled by both a demand and a goal is indexed once.
        /// @param mb Resident owning block.
        /// @param ruleId Existing unary composition rule ordinal.
        /// @return Nothing.
        /// @invariant The rule has exactly one premise.
void UnaryRuleMemory::indexUnaryConsumerPremise(Memory& mb, int32_t ruleId) {
        auto& store=mb.lbMemory.unaryComposition;
        assert(ruleId>0 && ruleId<=store.ruleInfo.size());
        const UnaryRuleInfo rule=store.ruleInfo[ruleId-1];
        assert(rule.premiseCount==1);
        const unsigned slot=gl::WorkerContext::coreId()>=0?static_cast<unsigned>(gl::WorkerContext::coreId()):scratchArenas().slotCount()-1;
        ScratchArena& arena=scratchArenas().forSlot(slot);
        ScratchScope frame(arena);
        const StrSpan premiseView=mb.ruleInterner.decodeView(rule.premise);
        const ScratchString premise=ScratchString::copyFrom(arena,premiseView.ptr,premiseView.len);
        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int arity=getArgsSpans(premise,args,ExecutionParameters::MAX_ARITY);
        const int name=mb.ruleInterner.encode(extractExpressionUniversalSpan(premise));
        store.premises.insertSorted(packInt32Pair(name,2*arity+(premise[0]=='!'?1:0)),ruleId);
    }

/// @brief Match a unary head against an open goal the way CNFE does.
        /// @details Same sign, operator and arity; every fixed `u_` argument
        /// equals the goal's argument at its position; exactly one bound
        /// variable, at a `(1)` definition slot.
        /// @param head Rule head with its bound variables unchanged.
        /// @param goal Goal statement as registered, without `u_` prefixes.
        /// @return Whether the head can establish the goal up to its constant.
        /// @invariant Pure predicate; creates nothing.
        /// @see checkNecessityForEquality.
bool UnaryRuleMemory::matchUnaryGoalHead(StrSpan head, StrSpan goal) {
        assert(head.len > 0 && goal.len > 0);
        if ((head.ptr[0]=='!') != (goal.ptr[0]=='!')
            || !equalSpans(extractExpressionUniversalSpan(head),extractExpressionUniversalSpan(goal))) return false;
        StrSpan args[ExecutionParameters::MAX_ARITY], goalArgs[ExecutionParameters::MAX_ARITY];
        const int count=getArgsSpans(head,args,ExecutionParameters::MAX_ARITY);
        const int arity=getArgsSpans(goal,goalArgs,ExecutionParameters::MAX_ARITY);
        if(count!=arity)return false;
        int variableCount=0,variableSlot=-1;
        for(int i=0;i<arity;++i){
            if(!startsWithSpan(args[i],"u_",2)){++variableCount;variableSlot=i;continue;}
            if(!equalSpans(StrSpan(args[i].ptr+2,args[i].len-2),goalArgs[i]))return false;
        }
        if(variableCount!=1)return false;
        const ce::CoreExpressionConfig* config=definitions.coreConfig(extractExpressionUniversalSpan(head));
        if(config==nullptr)return false;
        const auto* type=gl::CompiledDefinitionRegistry::findDefinitionSet(config,variableSlot+1);
        return type!=nullptr && equalSpans(StrSpan(type->first),StrSpan("(1)",3));
    }

/// @brief Enroll a consumer whose head can establish an open goal.
        /// @details The goal-side twin of `indexUnaryCompositionDemand`
        /// (D-469): the rule's premise enters the join
        /// index once and the goal key stays with the rule for the open-goal
        /// check of each proposed composition.
        /// @param mb Resident owning block.
        /// @param ruleId Existing composition rule ordinal.
        /// @param goalKey Packed key of an open goal in `intToBeProved`.
        /// @return Whether CNFE matching and comparable scopes permit it.
        /// @see matchUnaryGoalHead.
bool UnaryRuleMemory::indexUnaryCompositionGoal(Memory& mb, int32_t ruleId, int64_t goalKey) {
        auto& store=mb.lbMemory.unaryComposition;
        assert(ruleId>0 && ruleId<=store.ruleInfo.size());
        assert(mb.intToBeProved.lookup(goalKey)!=0);
        const UnaryRuleInfo rule=store.ruleInfo[ruleId-1];
        if (rule.premiseCount != 1) return false;
        const StatementKey goal=Codec<StatementKey>::decode(goalKey);
        if(!mb.nameMap.comparable(rule.scope,goal.validity))return false;
        if(mb.phase3IgnoresScope(mb.nameMap.deeperOf(rule.scope,goal.validity)))return false;
        const unsigned slot=gl::WorkerContext::coreId()>=0?static_cast<unsigned>(gl::WorkerContext::coreId()):scratchArenas().slotCount()-1;
        ScratchArena& arena=scratchArenas().forSlot(slot);
        ScratchScope frame(arena);
        const StrSpan headView=mb.ruleInterner.decodeView(rule.head);
        const ScratchString head=ScratchString::copyFrom(arena,headView.ptr,headView.len);
        const StrSpan goalView=mb.nameMap.decodeView(goal.orig);
        const ScratchString text=ScratchString::copyFrom(arena,goalView.ptr,goalView.len);
        if(!matchUnaryGoalHead(head,text))return false;
        const bool first=store.consumerGoals.lookup(ruleId)==0;
        store.consumerGoals.insertSorted(ruleId,goalKey);
        if(first)indexUnaryConsumerPremise(mb,ruleId);
        return true;
    }

/// @brief Wake installed consumer rules for newly observed goals.
        /// @details Goals are walked in sorted order; a removed goal's receipt
        /// retires, so a goal registered again wakes its consumers again.
        /// @param mb Resident owning block with its current goal registry.
        /// @return Nothing; eligible existing consumers enter the pending queue.
void UnaryRuleMemory::wakeUnaryCompositionGoals(Memory& mb) {
        auto& store=mb.lbMemory.unaryComposition;
        const auto& goals=mb.intToBeProved;
        store.observedGoals.eraseIf([&](int64_t key){return goals.lookup(key)==0;});
        const unsigned pageSlot=gl::WorkerContext::coreId()>=0?static_cast<unsigned>(gl::WorkerContext::coreId()):genScratchArenas().slotCount()-1;
        ScratchArena& pages=genScratchArenas().forSlot(pageSlot);
        ScratchScope pageFrame(pages);
        int32_t goalCount=0;
        const int64_t* keys=sortToBeProvedKeys(goals,mb.nameMap,pages,goalCount);
        const unsigned slot=gl::WorkerContext::coreId()>=0?static_cast<unsigned>(gl::WorkerContext::coreId()):scratchArenas().slotCount()-1;
        ScratchArena& strings=scratchArenas().forSlot(slot);
        for(int32_t i=0;i<goalCount;++i){
            const int64_t key=keys[i];
            if(store.observedGoals.lookup(key)!=0)continue;
            store.observedGoals.mint(key);
            ScratchScope frame(strings);
            const StrSpan view=mb.nameMap.decodeView(Codec<StatementKey>::decode(key).orig);
            const ScratchString text=ScratchString::copyFrom(strings,view.ptr,view.len);
            const int name=mb.ruleInterner.lookup(extractExpressionUniversalSpan(text));
            if(name==0)continue;
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int arity=getArgsSpans(text,args,ExecutionParameters::MAX_ARITY);
            const int family=store.heads.lookup(packInt32Pair(name,2*arity+(text[0]=='!'?1:0)));
            if(family==0)continue;
            for(int32_t j=0;j<store.heads.runLen(family);++j){
                const int32_t rule=store.heads.valueAt(family,j);
                if(indexUnaryCompositionGoal(mb,rule,key))store.pending.push_back(rule);
            }
        }
    }

/// @brief Whether a goal-enrolled consumer still serves an open goal.
        /// @details One of its goals must still be registered, with a scope
        /// comparable to the composition scope whose deeper end phase 3 does
        /// not ignore.
        /// @param mb Resident owning block.
        /// @param ruleId Consumer rule ordinal.
        /// @param scopeId Scope of the proposed composition.
        /// @return Whether the composition serves at least one open goal.
bool UnaryRuleMemory::unaryCompositionGoalOpen(Memory& mb, int32_t ruleId, NameId scopeId) {
        const auto& store=mb.lbMemory.unaryComposition;
        const int32_t row=store.consumerGoals.lookup(ruleId);
        if(row==0)return false;
        for(int32_t i=0;i<store.consumerGoals.runLen(row);++i){
            const int64_t key=store.consumerGoals.valueAt(row,i);
            const StatementKey goal=Codec<StatementKey>::decode(key);
            if(mb.intToBeProved.lookup(key)!=0
                && mb.nameMap.comparable(scopeId,goal.validity)
                && !mb.phase3IgnoresScope(mb.nameMap.deeperOf(scopeId,goal.validity)))return true;
        }
        return false;
    }

}
