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
#include "canonicalizer.hpp"
#include "scope_syntax.hpp"
#include "../compiler/definition_expansion.hpp"
namespace gl {
/// @brief Match admission and source templates against complete canonical scope views.
/// @details Owns canonical text, template projection and matching operations without
/// changing source-record ownership or the target-scope comparison contract.
class CanonicalTemplateMatcher {
public:

/// @brief Test an integration instruction's applied values canonically.
        /// @details The selected instruction's own history is compared through
        /// complete target representatives. Original stored values remain available
        /// until ordinary record maintenance coalesces them.
        /// @param mb Owning block with current complete classes.
        /// @param entries Selected admission record's arena snapshot.
        /// @param entry Instruction row within that snapshot.
        /// @param argument Witness argument to test.
        /// @param scope Effective target scope.
        /// @return Whether an equivalent value already occurs in this instruction.
inline bool integrationValueSeenAtScope(Memory& mb,const ArenaIntegrationMap& entries,
                int32_t entry,StrSpan argument,NameId scope) {
            const StrSpan canonical=canonicalArgumentAtScope(mb,argument,scope);
            for(int32_t node=entries.entries[entry].valHead;node>=0;node=entries.valNodes[node].prev) {
                const StrSpan previous=mb.valueInterner.decodeView(entries.valNodes[node].valueId);
                if(equalSpans(canonical,canonicalArgumentAtScope(mb,previous,scope))) return true;
            }
            return false;
        }
/// @brief Bind complete-scope canonicalization operations.
/// @details The peer outlives this matcher; derived indices remain on Memory.
/// @param canonicalizer Scope initialization and complete class-view operations.
/// @return Constructed canonical template matcher.
CanonicalTemplateMatcher(Canonicalizer& canonicalizer);

/// @brief Canonicalize fixed admission arguments without binding placeholders.
        /// @details Mixed chains identify fixed names with u_; already renamed
        /// chains identify them through remainingArgs. Complete scope representatives
        /// are selected before the existing alpha-renamer separates colliding free
        /// names. Fixed metadata absent from the premises stays in the output value.
        /// An Anchor-changing chain retains its original names and reports its
        /// existing substitution exclusion to the caller.
        /// @param mb Owning block and equality evidence.
        /// @param key Ordered premise chain on stable caller storage.
        /// @param keyN Number of premise elements, within the admission capacity.
        /// @param remainingArgs Sorted fixed-name metadata.
        /// @param remN Number of fixed names, within the admission capacity.
        /// @param canonicalScope Exact effective comparison scope.
        /// @param sArena String arena owning every resulting chain span.
        /// @param outChain Output buffer with room for keyN elements.
        /// @param outRemaining Output buffer with room for remN fixed names.
        /// @param excludedAnchor Whether representative substitution would change an Anchor.
        /// @param substitutions Optional complete-component substitutions for a maintenance hook.
        /// @param substitutionN Number of supplied substitutions; zero uses the complete scope view.
        /// @return Number of distinct output fixed names.
        /// @invariant No scoped equality substitutes a free placeholder.
        /// @see renamingChainScratch, updateAdmissionMap.
inline int32_t canonicalAdmissionChain(Memory& mb,
            const StrSpan* key, int32_t keyN, const StrSpan* remainingArgs, int32_t remN,
            NameId canonicalScope, ScratchArena& sArena, StrSpan* outChain,
            StrSpan* outRemaining, bool& excludedAnchor,
            const StrReplacement* substitutions = nullptr, int32_t substitutionN = 0) {
            assert(canonicalScope != 0);
            assert(keyN <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                && remN <= ExecutionParameters::MAX_ADMISSION_REM_ARGS);
            canonicalizer.initializeCanonicalScope(mb, canonicalScope);
            const auto representativeOf = [&](StrSpan name) {
                if (substitutions != nullptr) {
                    for (int32_t i = 0; i < substitutionN; ++i)
                        if (equalSpans(name, substitutions[i].key)) return substitutions[i].value;
                    return name;
                }
                return canonicalArgumentAtScope(mb, name, canonicalScope);
            };
            bool prefixedChain = false;
            for (int32_t i = 0; i < keyN; ++i)
                forEachSubstitutionToken(key[i], [&](StrSpan token) {
                    if (startsWithSpan(token, "u_", 2)) prefixedChain = true;
                });
            const auto fixedArgument = [&](StrSpan token) {
                if (startsWithSpan(token, "u_", 2)) return true;
                if (prefixedChain || equalSpans(token, StrSpan("marker", 6))) return false;
                for (int32_t r = 0; r < remN; ++r)
                    if (equalSpans(token, remainingArgs[r])) return true;
                bool numbered = !token.empty();
                for (int32_t i = 0; i < token.len; ++i)
                    if (token[i] < '0' || token[i] > '9') numbered = false;
                return !numbered;
            };
            StrSpan canonicalKey[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
            StrSpan canonicalRemaining[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
            StrSpan renameRemaining[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
            const StrSpan* renameRemainingArgs = remainingArgs;
            int32_t renameRemainingN = remN;
            excludedAnchor = false;
            for (int32_t i = 0; i < keyN; ++i) {
                StrSpan text = key[i];
                if (!text.empty() && text[0] == '!') text = StrSpan(text.ptr + 1, text.len - 1);
                if (!startsWithSpan(text, "(Anchor", 7)) continue;
                forEachSubstitutionToken(text, [&](StrSpan token) {
                    const bool prefixed = startsWithSpan(token, "u_", 2);
                    const StrSpan name = prefixed ? StrSpan(token.ptr + 2, token.len - 2) : token;
                    if (fixedArgument(token) && !equalSpans(name, representativeOf(name))) excludedAnchor = true;
                });
            }
            const int32_t canonicalBucket = excludedAnchor ? 0 : mb.canonicalEquivalenceClassesMap.lookup(canonicalScope);
            if (!excludedAnchor && (substitutionN != 0
                || (canonicalBucket != 0 && mb.canonicalEquivalenceClassesMap.runLen(canonicalBucket) != 0))) {
                assert(keyN <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS && remN <= ExecutionParameters::MAX_ADMISSION_REM_ARGS);
                // Bare arguments in a mixed u_-prefixed chain are bound
                // variables. In an already renamed chain, remainingArgs
                // identifies the concrete arguments. Keep that distinction
                // while selecting true representatives, then let the existing
                // renamer avoid capturing a bound name with a representative.
                for (int32_t i = 0; i < keyN; ++i) {
                    StrReplacement pairs[ExecutionParameters::MAX_ARITY];
                    int32_t pairN = 0;
                    forEachSubstitutionToken(key[i], [&](StrSpan token) {
                        const bool prefixed = startsWithSpan(token, "u_", 2);
                        if (!fixedArgument(token)) return;
                        for (int32_t p = 0; p < pairN; ++p)
                            if (equalSpans(pairs[p].key, token)) return;
                        const StrSpan original = prefixed ? StrSpan(token.ptr + 2, token.len - 2) : token;
                        const StrSpan representative = representativeOf(original);
                        char* text = sArena.allocBytes(representative.len + 2);
                        text[0] = 'u'; text[1] = '_';
                        std::memcpy(text + 2, representative.ptr, representative.len);
                        assert(pairN < ExecutionParameters::MAX_ARITY);
                        pairs[pairN++] = {token, StrSpan(text, representative.len + 2)};
                    });
                    canonicalKey[i] = StrSpan(replaceKeysScratch(sArena, key[i], pairs, pairN));
                }
                for (int32_t i = 0; i < remN; ++i) {
                    const StrSpan canonical = representativeOf(remainingArgs[i]);
                    canonicalRemaining[i] = StrSpan(ScratchString::copyFrom(sArena, canonical.ptr, canonical.len));
                }
                std::sort(canonicalRemaining, canonicalRemaining + remN,
                    [](StrSpan a, StrSpan b) { return compareSpans(a,b) < 0; });
                remN = static_cast<int32_t>(std::unique(canonicalRemaining, canonicalRemaining + remN,
                    [](StrSpan a, StrSpan b) { return equalSpans(a,b); }) - canonicalRemaining);
                key = canonicalKey; remainingArgs = canonicalRemaining;
                renameRemainingArgs = remainingArgs;
                renameRemainingN = remN;
                if (!prefixedChain) {
                    // A previously renamed chain may retain fixed names
                    // absent from its remaining premises. They still belong
                    // to the admission value, but cannot constrain the local
                    // alpha-renaming of arguments that occur in this chain.
                    renameRemainingN = 0;
                    for (int32_t r = 0; r < remN; ++r) {
                        bool present = false;
                        for (int32_t i = 0; i < keyN && !present; ++i)
                            forEachSubstitutionToken(key[i], [&](StrSpan token) {
                                if (startsWithSpan(token, "u_", 2)
                                    && equalSpans(StrSpan(token.ptr + 2, token.len - 2), remainingArgs[r])) present = true;
                            });
                        if (present) renameRemaining[renameRemainingN++] = remainingArgs[r];
                    }
                    renameRemainingArgs = renameRemaining;
                }
            }
            StrSpan renamedKey[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
            int32_t renamedN = 0;
            {
                RT_SCOPE_HERE("UAM_RENAME_CHAIN");
                renamedN = DefinitionExpansion::renamingChainScratch(sArena, key, keyN,
                    renameRemainingArgs, renameRemainingN, renamedKey,
                    ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS);
            }

            for (int32_t i = 0; i < keyN; ++i) outChain[i] = renamedKey[i];
            for (int32_t i = 0; i < remN; ++i) outRemaining[i] = remainingArgs[i];
            return remN;
        }

/// @brief Index the substitution names of one scope-owned expression.
        /// @details Exact tokens and their one-level instruction prefixes share
        /// dependencies; the consumer still applies its own shape and marker rules.
        /// The private index never mints names into the proof's NameMap.
        /// @param index Derived record-dependency index.
        /// @param key Stable logical record key.
        /// @param scope Effective scope of the record.
        /// @param text Expression or marked template to index.
        /// @return Nothing.
inline void indexCanonicalText(CanonicalUseIndex& index, int64_t key,
                                       NameId scope, StrSpan text) {
            index.remember(key);
            StrSpan literalGoal, body;
            if (gl::ScopeSyntax::splitSubproofPayload(text,literalGoal,body)) text=body;

            if (!text.empty() && !containsSpan(text, StrSpan("[", 1))) index.add(key, scope, text);
            forEachSubstitutionToken(text, [&](StrSpan token) {
                index.add(key, scope, token);
                if (startsWithSpan(token, "u_", 2) && token.len > 2)
                    index.add(key, scope, StrSpan(token.ptr + 2, token.len - 2));
            });
        }

/// @brief Read a bare argument's representative in the complete scope view.
        /// @details The maintained member snapshot provides the same normal chooser
        /// result as the complete class. Names outside any component retain their
        /// spelling; no interner is minted and no proof state changes.
        /// @param mb Owning block with initialized target classes.
        /// @param argument Bare concrete argument.
        /// @param scope Exact canonical scope.
        /// @return Canonical name span, or the original unclassified argument span.
inline StrSpan canonicalArgumentAtScope(Memory& mb,StrSpan argument,NameId scope) {
            assert(scope!=0);
            const NameId member=mb.nameMap.lookup(argument);
            if(member==0) return argument;
            const NameId canonical=canonicalMemberIdAtScope(mb,member,scope);
            return canonical==member ? argument : mb.nameMap.decodeView(canonical);
        }

/// @brief Read a member's representative id in the complete scope view.
        /// @details
        /// The id form of `canonicalArgumentAtScope`: the maintained member
        /// snapshot (`canonicalMemberRepresentatives`) gives the normal chooser
        /// result of the member's complete class at the scope. A member in no
        /// class of the view, or in a class without a strong member, is its own
        /// representative. Two members are in one class of the view exactly when
        /// their representatives are equal, since the classes of one view are
        /// disjoint (I-218). Reads only; no interner is minted.
        /// @param mb Owning block with initialized target classes.
        /// @param member The member's NameMap id (nonzero).
        /// @param scope Exact canonical scope (nonzero).
        /// @return The representative's NameMap id, or @p member itself.
        /// @see canonicalArgumentAtScope, AdmissionMemory::oneWayWitnessAtScope.
inline NameId canonicalMemberIdAtScope(Memory& mb,NameId member,NameId scope) {
            assert(member!=0 && scope!=0);
            const int64_t* representatives=mb.lbMemory.canonicalMemberRepresentatives.find(packInt32Pair(member,scope));
            if(representatives==nullptr) return member;
            const NameId canonical=canonicalRecordNormal(*representatives);
            return canonical==0 ? member : canonical;
        }

/// @brief Normalize an admission expression through the complete target view.
        /// @details Concrete tokens use the ordinary representative chooser. Optional
        /// u_ tokens retain their prefix; markers, protected witness names and Anchor
        /// identities retain their spelling. This changes no bound-variable scheme.
        /// @param mb Owning block with a current canonical view.
        /// @param text Expression or bare argument to normalize.
        /// @param scope Effective comparison scope.
        /// @param arena String arena owning the returned bytes.
        /// @param prefixed Whether rigid u_-prefixed arguments participate.
        /// @param protect Literal witness token to retain, or an empty span.
        /// @return Owned canonical text on the caller's string arena.
inline ScratchString canonicalAdmissionText(Memory& mb, StrSpan text, NameId scope,
                ScratchArena& arena, bool prefixed = false, StrSpan protect = StrSpan()) {
            assert(scope != 0);
            const ScratchString owned = ScratchString::copyFrom(arena, text.ptr, text.len);
            text = StrSpan(owned);
            StrSpan literalGoal, payload;
            if (gl::ScopeSyntax::splitSubproofPayload(text,literalGoal,payload)) {
                const ScratchString normalized = canonicalAdmissionText(mb,payload,scope,arena,prefixed,protect);
                const StrSpan normalizedText(normalized);
                const int32_t prefix = static_cast<int32_t>(payload.ptr-text.ptr);
                char* bytes=arena.allocBytes(prefix+normalizedText.len);
                std::memcpy(bytes,text.ptr,prefix);
                std::memcpy(bytes+prefix,normalizedText.ptr,normalizedText.len);
                return ScratchString::wrap(arena,bytes,prefix+normalizedText.len);
            }

            canonicalizer.initializeCanonicalScope(mb, scope);
            StrSpan body = text;
            if (!body.empty() && body[0] == '!') body = StrSpan(body.ptr + 1, body.len - 1);
            if (startsWithSpan(body, "(Anchor", 7) || startsWithSpan(body, "Anchor", 6)) return owned;
            const int32_t bucket = mb.canonicalEquivalenceClassesMap.lookup(scope);
            if (bucket == 0) return owned;
            StrReplacement pairs[ExecutionParameters::MAX_ARITY];
            int32_t count = 0;
            const bool bare = !containsSpan(text, StrSpan("[", 1));
            const auto token = [&](StrSpan original) {
                if (equalSpans(original, StrSpan("marker", 6)) || equalSpans(original, protect)) return;
                const bool prefix = prefixed && startsWithSpan(original, "u_", 2);
                const StrSpan name = prefix ? StrSpan(original.ptr + 2, original.len - 2) : original;
                for (int32_t i = 0; i < count; ++i) if (equalSpans(pairs[i].key, original)) return;
                StrSpan replacement=canonicalArgumentAtScope(mb,name,scope);
                if(equalSpans(replacement,name)) return;
                if (prefix) {
                    char* buffer = arena.allocBytes(replacement.len + 2);
                    buffer[0] = 'u'; buffer[1] = '_';
                    std::memcpy(buffer + 2, replacement.ptr, replacement.len);
                    replacement = StrSpan(buffer, replacement.len + 2);
                }
                assert(count < ExecutionParameters::MAX_ARITY);
                pairs[count++] = {original, replacement};
            };
            if (bare) token(text);
            else forEachSubstitutionToken(text, token);
            if (count == 0) return owned;
            if (bare) return ScratchString::copyFrom(arena, pairs[0].value.ptr, pairs[0].value.len);
            return replaceKeysScratch(arena, text, pairs, count);
        }

/// @brief Extend template-shape and scope postings for newly stored keys.
        /// @details Existing projections remain in place. New keys are projected
        /// only into previously requested target scopes for their shape and owner.
        /// @tparam Map Authoritative template-keyed map or set.
        /// @param mb Owning block and template interner.
        /// @param map Source records whose original keys remain authoritative.
        /// @param index Derived matching index for this store.
        /// @param arena Caller-owned scratch.
        /// @param statements Read originals from NameMap instead of the template interner.
        /// @return Nothing.
template<class Map>
        inline void ensureCanonicalTemplateIndex(Memory& mb,const Map& map,CanonicalTemplateIndex& index,ScratchArena& arena,bool statements=false) {
            RT_SCOPE_HERE("CANONICAL_MATCH_INDEX_SYNC");
            index.sources.synchronize(map,arena,[&](int64_t key) {
                index.sources.remember(key);
                const StatementKey source=Codec<StatementKey>::decode(key);
                const StrSpan text=statements ? mb.nameMap.decodeView(source.orig) : mb.templateInterner.decodeView(source.orig);
                int32_t length=0;
                while(length<text.len && text[length]!='[') ++length;
                const int32_t label=index.labels.mint(StrSpan(text.ptr,length));
                const int64_t shapeScope=packInt32Pair(label,source.validity);
                index.keysByShape.insert(shapeScope,key);
                index.scopesByShape.insert(label,source.validity);
                index.targetsByShape.forEach(shapeScope,[&](int64_t target) {
                    const int64_t identity=packInt32Pair(source.orig,static_cast<NameId>(target));
                    if(index.forms.lookup(identity)==0) index.enqueueProjection(identity);
                });
            });
        }

/// @brief Find original template keys equivalent at one effective scope.
        /// @details First use projects eligible source templates once. Later source
        /// insertions and equality changes update only dependent projections. Cached
        /// keys never change ancestor ownership, and the authoritative map excludes
        /// retired results. Returned keys are sorted by their original template.
        /// @tparam Map Authoritative admission, source or demand map.
        /// @param mb Owning block and current canonical classes.
        /// @param map Authoritative map queried without mutation.
        /// @param index Derived comparison cache for this map.
        /// @param expression Query template, with the map's ordinary marker/prefix shape.
        /// @param sourceScope Exact original ownership scope permitted by this route.
        /// @param targetScope Scope where both compared forms are normalized.
        /// @param output Empty caller-owned vector receiving existing source keys.
        /// @param arena Caller-owned page-tier scratch.
        /// @param statements Use NameMap originals and preserve statement-specific shape rules.
        /// @return Nothing.
template<class Map>
        inline void matchingCanonicalTemplates(Memory& mb, const Map& map, CanonicalTemplateIndex& index,
                StrSpan expression, NameId sourceScope, NameId targetScope,
                PagedVector<int64_t>& output, ScratchArena& arena, bool statements=false) {
            RT_SCOPE_HERE("CANONICAL_MATCH_QUERY");
            assert(sourceScope == targetScope || mb.nameMap.isStrictAncestor(sourceScope, targetScope));
            const auto originalText = [&](int32_t id) -> StrSpan {
                return statements ? mb.nameMap.decodeView(id) : mb.templateInterner.decodeView(id);
            };
            const unsigned slot = gl::WorkerContext::coreId() >= 0 ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : scratchArenas().slotCount() - 1;
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope queryLifetime(strings);
            const ScratchString queryText = ScratchString::copyFrom(strings, expression.ptr, expression.len);
            expression = StrSpan(queryText);
            canonicalizer.initializeCanonicalScope(mb, targetScope);
            bool relevant = false;
            const int32_t view = mb.canonicalEquivalenceClassesMap.lookup(targetScope);
            if (view != 0) {
                StrSpan literalGoal, dependencyText = expression, payload;
                if (gl::ScopeSyntax::splitSubproofPayload(expression,literalGoal,payload)) dependencyText = payload;
                forEachSubstitutionToken(dependencyText, [&](StrSpan token) {
                    if (!statements && startsWithSpan(token, "u_", 2)) token = StrSpan(token.ptr + 2, token.len - 2);
                    const NameId member = mb.nameMap.lookup(token);
                    if (member != 0 && mb.lbMemory.canonicalMemberRepresentatives.lookup(packInt32Pair(member, targetScope)) != 0)
                        relevant = true;
                });
            }
            if (!relevant) {
                const int32_t text = statements ? mb.nameMap.lookup(expression) : mb.templateInterner.lookup(expression);
                const int64_t key = packInt32Pair(text, sourceScope);
                if (text != 0 && map.lookup(key) != 0) output.push_back(key);
                return;
            }

            const auto enqueue = [&](int64_t key, NameId target) {
                const StatementKey source = Codec<StatementKey>::decode(key);
                if (source.validity != target && !mb.nameMap.isStrictAncestor(source.validity, target)) return;
                const int64_t identity = packInt32Pair(source.orig, target);
                if (index.forms.lookup(identity) == 0) index.enqueueProjection(identity);
            };
            const auto shape = [&](StrSpan text) {
                int32_t length = 0;
                while (length < text.len && text[length] != '[') ++length;
                return index.labels.mint(StrSpan(text.ptr, length));
            };
            ensureCanonicalTemplateIndex(mb,map,index,arena,statements);
            const int64_t shapeScope = packInt32Pair(shape(expression), sourceScope);
            if (index.targetsByShape.insert(shapeScope, targetScope)) {
                index.dirty=DirtyState::Restructured;
                index.keysByShape.forEach(shapeScope, [&](int64_t key) {
                    if (map.lookup(key) != 0) enqueue(key, targetScope);
                });
            }
            DirtyState dirty = DirtyState::Clean;
            PagedVector<int64_t> pending(&arena, &dirty);
            index.pending.forEach(targetScope, [&](int64_t key) { pending.push_back(key); });
            const auto less = [&](int32_t a, int32_t b) {
                return compareSpans(originalText(Codec<StatementKey>::decode(pending[a]).orig),
                    originalText(Codec<StatementKey>::decode(pending[b]).orig)) < 0;
            };
            ChunkSortedOrdinals<decltype(less)> order(arena, pending.size(), 0, less);
            for (int32_t i = 0; i < pending.size(); ++i) {
                RT_SCOPE_HERE("CANONICAL_MATCH_REBUILD");
                RT_NOTE_ITERATIONS_HERE(1);
                ++index.rebuildCount;
                ScratchScope frame(strings);
                const int64_t key = pending[order.next()];
                const StatementKey original = Codec<StatementKey>::decode(key);
                const ScratchString canonical = canonicalAdmissionText(mb,
                    originalText(original.orig), targetScope, strings, !statements);
                const int32_t result = mb.templateInterner.encode(StrSpan(canonical));
                const int32_t* previous = index.forms.find(key);
                if (previous == nullptr || *previous != result) {
                    if (previous != nullptr) index.matches.erase(packInt32Pair(*previous, targetScope), original.orig);
                    index.forms.upsert(key, result);
                    index.matches.insert(packInt32Pair(result, targetScope), original.orig);
                    index.uses.erase(key, arena);
                    indexCanonicalText(index.uses, key, targetScope, originalText(original.orig));
                    // Equal forms have exactly the same token dependencies.
                    if (!equalSpans(originalText(original.orig), StrSpan(canonical)))
                        indexCanonicalText(index.uses, key, targetScope, mb.templateInterner.decodeView(result));
                }
            }
            for (int32_t i = 0; i < pending.size(); ++i) {
                const bool erased = index.pending.erase(targetScope, pending[i]);
                assert(erased);
            }
            if (!pending.empty()) index.dirty = DirtyState::Restructured;
            index.pending.maintain();
            ScratchScope queryFrame(strings);
            const ScratchString canonical = canonicalAdmissionText(mb, expression, targetScope, strings, !statements);
            const int32_t query = mb.templateInterner.lookup(StrSpan(canonical));
            if (query == 0) return;
            TypedColdSet<int64_t> found(&arena, &dirty);
            index.matches.forEach(packInt32Pair(query, targetScope), [&](int64_t original) {
                const int64_t key = packInt32Pair(static_cast<int32_t>(original), sourceScope);
                if (map.lookup(key) != 0) found.mint(key);
            });
            const auto resultLess = [&](int32_t a, int32_t b) {
                return compareSpans(originalText(Codec<StatementKey>::decode(found.keyAt(a + 1)).orig),
                    originalText(Codec<StatementKey>::decode(found.keyAt(b + 1)).orig)) < 0;
            };
            ChunkSortedOrdinals<decltype(resultLess)> resultOrder(arena, found.count(), 0, resultLess);
            for (int32_t i = 0; i < found.count(); ++i) output.push_back(found.keyAt(resultOrder.next() + 1));
        }

/// @brief Test admission identity at its effective scope while retaining ownership.
        /// @details Exact source-owned raw or target-canonical keys establish a
        /// positive answer without projecting other records. Aliases use the same
        /// canonical projection cache as record-valued reads. A missing template
        /// is an ordinary absence; scope eligibility and proof policy are unchanged.
        /// @tparam Map Authoritative marked-template map or set.
        /// @param mb Owning block.
        /// @param map Authoritative records.
        /// @param index Derived comparisons for this record kind.
        /// @param expression Query template.
        /// @param validity Source ownership scope.
        /// @param target Effective comparison scope, or zero for the source scope.
        /// @return Whether at least one original record matches canonically.
template<class Map>
        inline bool canonicalTemplatePresent(Memory& mb, const Map& map, CanonicalTemplateIndex& index,
                StrSpan expression, StrSpan validity, NameId target = 0) {
            return canonicalTemplatePresentIf(mb, map, index, expression, validity, target,
                [](int64_t) { return true; });
        }

/// @brief Test admission identity owned at a scope or at any strict
        ///        ancestor of it, compared at the scope.
        ///
        /// @details
        /// The record visibility of a stored instruction's continuation
        /// (D-572): a record owned at an
        /// ancestor holds below it as well. The scope itself is probed first,
        /// then each strict ancestor in `ScopeSyntax::copyStrictAncestorSpans`
        /// order, every probe `canonicalTemplatePresent` with the owning scope as
        /// source and @p validity as target. An unregistered @p validity is an
        /// ordinary absence, as for `canonicalTemplatePresent`. The query text
        /// and the scope are copied before the first probe, since the
        /// class-aware match mints templates.
        /// @tparam Map Authoritative marked-template map or set.
        /// @param mb Owning block.
        /// @param map Authoritative records.
        /// @param index Derived comparisons for this record kind.
        /// @param expression Query template.
        /// @param validity The comparison scope; owners at it or above count.
        /// @return Whether a record owned at @p validity or at a strict
        ///         ancestor matches canonically at @p validity.
        /// @invariant Adds no record; may extend the derived index only.
        /// @see canonicalTemplatePresent,
        ///      IntegrationProcessor::prepareIntegrationCore2,
        ///      ImplicationSubproofs::openImplicationSubproof.
template<class Map>
        inline bool canonicalTemplatePresentAtOrAbove(Memory& mb, const Map& map, CanonicalTemplateIndex& index,
                StrSpan expression, StrSpan validity) {
            const NameId target = mb.nameMap.lookup(validity);
            if (target == 0) return false;
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(strings);
            const ScratchString text = ScratchString::copyFrom(strings, expression.ptr, expression.len);
            const ScratchString scope = ScratchString::copyFrom(strings, validity.ptr, validity.len);
            StrSpan ancestors[ExecutionParameters::MAX_SCOPE_DEPTH];
            const int32_t ancestorN = gl::ScopeSyntax::copyStrictAncestorSpans(mb, StrSpan(scope), ancestors, strings);
            if (canonicalTemplatePresent(mb, map, index, StrSpan(text), StrSpan(scope), target)) return true;
            for (int32_t a = 0; a < ancestorN; ++a)
                if (canonicalTemplatePresent(mb, map, index, StrSpan(text), ancestors[a], target)) return true;
            return false;
        }

/// @brief Test admission identity at its effective scope, counting only the
        ///        original records a predicate accepts.
        /// @details
        /// The shared core of `canonicalTemplatePresent`: the exact raw key,
        /// then the target-canonical key, then the class-aware matches
        /// (`matchingCanonicalTemplates`, in their sorted order), each counted
        /// only when @p accept returns true for its packed source key. A
        /// rejected exact hit falls through to the later forms, which may name
        /// another record. With an accept-all predicate the answer equals the
        /// former unfiltered test, probe for probe. The predicate is a pure
        /// read; it must not mutate @p map.
        /// @tparam Map Authoritative marked-template map or set.
        /// @tparam Accept Callable `bool(int64_t key)` over packed source keys.
        /// @param mb Owning block.
        /// @param map Authoritative records.
        /// @param index Derived comparisons for this record kind.
        /// @param expression Query template.
        /// @param validity Source ownership scope.
        /// @param target Effective comparison scope, or zero for the source scope.
        /// @param accept Whether a matching record counts.
        /// @return Whether at least one accepted original record matches canonically.
        /// @see canonicalTemplatePresent, matchingCanonicalTemplates,
        ///      AdmissionMemory::isAdmittedIntegrationDouble.
template<class Map, class Accept>
        inline bool canonicalTemplatePresentIf(Memory& mb, const Map& map, CanonicalTemplateIndex& index,
                StrSpan expression, StrSpan validity, NameId target, const Accept& accept) {
            const NameId source = mb.nameMap.lookup(validity);
            if (source == 0) return false;
            if (target == 0) target = source;
            assert(source == target || mb.nameMap.isStrictAncestor(source, target));
            const unsigned slot = gl::WorkerContext::coreId() >= 0 ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(arena), textFrame(strings);
            const ScratchString original = ScratchString::copyFrom(strings, expression.ptr, expression.len);
            canonicalizer.initializeCanonicalScope(mb, target);
            const int32_t raw = mb.templateInterner.lookup(StrSpan(original));
            if (raw != 0 && map.lookup(packInt32Pair(raw, source)) != 0 && accept(packInt32Pair(raw, source)))
                return true;
            const ScratchString canonical = canonicalAdmissionText(mb, StrSpan(original), target, strings, true);
            const int32_t normalized = mb.templateInterner.lookup(StrSpan(canonical));
            if (normalized != 0 && map.lookup(packInt32Pair(normalized, source)) != 0
                && accept(packInt32Pair(normalized, source))) return true;
            DirtyState dirty = DirtyState::Clean;
            PagedVector<int64_t> matches(&arena, &dirty);
            matchingCanonicalTemplates(mb, map, index, StrSpan(canonical), source, target, matches, arena);
            for (int32_t i = 0; i < matches.size(); ++i)
                if (accept(matches[i])) return true;
            return false;
        }

private:
Canonicalizer& canonicalizer;
};
}

namespace gl {
/// @brief Bind complete-scope canonicalization operations.
/// @details The peer outlives this matcher; derived indices remain on Memory.
/// @param canonicalizer Scope initialization and complete class-view operations.
/// @return Constructed canonical template matcher.
inline CanonicalTemplateMatcher::CanonicalTemplateMatcher(Canonicalizer& canonicalizer) : canonicalizer(canonicalizer) {}
} // namespace gl
