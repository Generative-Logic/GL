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
#include "peak_snapshot.hpp"
#include "../memory.hpp"
#include "../compiler.hpp"
#include <array>
#include <cstdio>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

namespace gl { namespace peak_snapshot {
// Opt-in memory measurement: typed heap traversal at the quiescent second-pass snapshot.

template<class T> struct HeapView;
/// @brief Visit a named member through its concrete C++ storage type.
/// @details References and pointers themselves do not confer ownership of targets.
/// @tparam T Member type.
/// @param name Field name.
/// @param value Live member, stable for the whole capture.
/// @return Nothing.
template<class T> void field(const char* name,const T& value) {
    Scope owner(name);HeapView<std::remove_cv_t<T>>::read(value);
}

/// @brief Inline representation of a scalar or a type without owned heap fields.
/// @details Pointer values are recorded but never followed as owning edges.
/// @tparam T Concrete field type.
template<class T> struct HeapView {
    /// @brief Observe the inline field bytes.
    /// @details A containing object supplies padding not represented by its children.
    /// @param value Stable field value.
    /// @return Nothing.
    static void read(const T& value){note(&value,sizeof(value),3);}
};

template<class C,class Traits,class A> struct HeapView<std::basic_string<C,Traits,A>> {
    /// @brief Observe string metadata, live characters, capacity and allocation rounding.
    /// @details Inline small-string storage stays within the string object.
    /// @param value Stable string.
    /// @return Nothing.
    static void read(const std::basic_string<C,Traits,A>& value) {
        note(&value,sizeof(value),0);Scope bytes("characters");
        const auto begin=reinterpret_cast<uintptr_t>(value.data());
        const auto object=reinterpret_cast<uintptr_t>(&value);
        if(begin>=object && begin<object+sizeof(value)) {
            note(value.data(),(value.capacity()+1)*sizeof(C),1);
            note(value.data(),(value.size()+1)*sizeof(C),3);
        } else allocation(value.data(),(value.capacity()+1)*sizeof(C),(value.size()+1)*sizeof(C));
    }
};

template<class T,class A> struct HeapView<std::vector<T,A>> {
    /// @brief Observe a vector's control fields, live elements and spare capacity.
    /// @details Nested element owners are visited without copying the vector.
    /// @param value Stable vector.
    /// @return Nothing.
    static void read(const std::vector<T,A>& value) {
        note(&value,sizeof(value),0);Scope storage("storage");
        allocation(value.data(),value.capacity()*sizeof(T),value.size()*sizeof(T));
        for(const auto& element:value)field("element",element);
    }
};

template<class A,class B> struct HeapView<std::pair<A,B>> {
    /// @brief Observe the two named pair fields and their padding.
    /// @details Owned children retain their own allocation identities.
    /// @param value Stable pair.
    /// @return Nothing.
    static void read(const std::pair<A,B>& value) {
        note(&value,sizeof(value),0);field("first",value.first);field("second",value.second);
    }
};

template<class... T> struct HeapView<std::tuple<T...>> {
    /// @brief Observe tuple elements in declaration order.
    /// @details Labels use each element's ordinal; storage is never copied.
    /// @param value Stable tuple.
    /// @return Nothing.
    static void read(const std::tuple<T...>& value) {
        note(&value,sizeof(value),0);
        std::size_t index=0;
        std::apply([&](const auto&... elements){
            const auto visit=[&](const auto& element){char name[32]{};std::snprintf(name,sizeof(name),"element_%zu",index++);field(name,element);};
            (visit(elements),...);
        },value);
    }
};

template<class T,std::size_t N> struct HeapView<std::array<T,N>> {
    /// @brief Observe inline array elements and alignment bytes.
    /// @details The array does not own a second allocation.
    /// @param value Stable array.
    /// @return Nothing.
    static void read(const std::array<T,N>& value) {
        note(&value,sizeof(value),0);for(const auto& element:value)field("element",element);
    }
};

template<class... T> struct HeapView<std::variant<T...>> {
    /// @brief Observe the variant's active object and remaining inline storage.
    /// @details No inactive alternative is treated as a live object.
    /// @param value Stable non-valueless variant.
    /// @return Nothing.
    static void read(const std::variant<T...>& value) {
        note(&value,sizeof(value),0);assert(!value.valueless_by_exception());
        std::visit([](const auto& element){field("activeValue",element);},value);
    }
};

template<> struct HeapView<std::filesystem::path> {
    /// @brief Observe the path's native character storage.
    /// @details The native string reference aliases the path; no copy is made.
    /// @param value Stable path.
    /// @return Nothing.
    static void read(const std::filesystem::path& value) {
        note(&value,sizeof(value),0);field("native",value.native());
    }
};

#ifdef _MSC_VER
/// @brief Observe Microsoft's stable tree-node storage through its iterator.
/// @details The installed standard-library iterator publicly exposes its node
/// pointer. Exact sizeof(node) is used; no private layout offsets are guessed.
/// @tparam T Ordered map or set type.
/// @param value Stable container.
/// @return Nothing.
template<class T> void treeNodes(const T& value) {
    note(&value,sizeof(value),0);
    {Scope sentinel("sentinel");const auto* node=value.end()._Ptr;allocation(node,sizeof(*node),sizeof(*node));}
    Scope nodes("nodes");
    for(auto it=value.begin();it!=value.end();++it) {
        const auto* node=it._Ptr;
        allocation(node,sizeof(*node),sizeof(*node));field("value",*it);
    }
}
template<class K,class V,class C,class A> struct HeapView<std::map<K,V,C,A>> {
    /// @brief Observe every map node, key and mapped value.
    /// @details Empty maps retain only their sentinel allocation.
    /// @param value Stable map.
    /// @return Nothing.
    static void read(const std::map<K,V,C,A>& value){treeNodes(value);}
};
template<class K,class C,class A> struct HeapView<std::set<K,C,A>> {
    /// @brief Observe every set node and stored value.
    /// @details The container's sentinel is a separate named allocation.
    /// @param value Stable set.
    /// @return Nothing.
    static void read(const std::set<K,C,A>& value){treeNodes(value);}
};
#endif

template<> struct HeapView<ContrapositionTemplate> {
    /// @brief Observe all heap-bearing contraposition-template fields.
    /// @details These are definition data, read without modifying any template.
    /// @param value Stable template.
    /// @return Nothing.
    static void read(const ContrapositionTemplate& value) {
        note(&value,sizeof(value),0);field("sources",value.sources);
        field("sourceRule",value.sourceRule);field("reverseRule",value.reverseRule);
    }
};
template<> struct HeapView<DisintegrationShot> {
    /// @brief Observe all heap-bearing fields of a disintegration-package shot.
    /// @details Definition data, read without modifying the shot.
    /// @param value Stable shot.
    /// @return Nothing.
    static void read(const DisintegrationShot& value) {
        note(&value,sizeof(value),0);field("universal",value.universal);
        field("premises",value.premises);field("head",value.head);field("binder",value.binder);
    }
};
template<> struct HeapView<IntegrationImplication> {
    /// @brief Observe the heap-bearing fields of one compiled integration implication.
    /// @details Definition data, read without modifying the record; the kind
    /// and citation index stay in the inline representation.
    /// @param value Stable integration implication.
    /// @return Nothing.
    static void read(const IntegrationImplication& value) {
        note(&value,sizeof(value),0);field("compact",value.compact);
        field("kind",value.kind);field("cites",value.cites);
    }
};
template<> struct HeapView<LogicalEntity> {
    /// @brief Observe the compiled entity and every owned sequence/string field.
    /// @details Scalar and alignment bytes remain in the inline representation.
    /// @param value Stable compiled entity.
    /// @return Nothing.
    static void read(const LogicalEntity& value) {
        note(&value,sizeof(value),0);
        field("category",value.category);field("elements",value.elements);
        field("signature",value.signature);field("definedSet",value.definedSet);
        field("implications",value.implications);
        field("integrationImplications",value.integrationImplications);
        field("contrapositives",value.contrapositives);
        field("conjunctionRefutations",value.conjunctionRefutations);
        field("supportingImplications",value.supportingImplications);
        field("composedSupportingImplications",value.composedSupportingImplications);
        field("arity",value.arity);field("symmetricDigitSlots",value.symmetricDigitSlots);
        field("contrapositionPrepared",value.contrapositionPrepared);
        field("contrapositionAuxiliary",value.contrapositionAuxiliary);
        field("conjunctionRefutationsPrepared",value.conjunctionRefutationsPrepared);
        field("supportingImplicationsPrepared",value.supportingImplicationsPrepared);
        field("supportingImplicationAuxiliary",value.supportingImplicationAuxiliary);
        field("shotBinders",value.shotBinders);field("shotPlaceholders",value.shotPlaceholders);
        field("shots",value.shots);field("shotsPrepared",value.shotsPrepared);
    }
};
template<> struct HeapView<ce::CoreExpressionConfig> {
    /// @brief Observe the core-definition configuration's owned fields.
    /// @details Both inline and file-backed definition forms retain their real types.
    /// @param value Stable configuration.
    /// @return Nothing.
    static void read(const ce::CoreExpressionConfig& value) {
        note(&value,sizeof(value),0);field("arity",value.arity);field("definition",value.definition);
        field("signature",value.signature);field("definitionSets",value.definitionSets);
        field("inputArgs",value.inputArgs);field("outputArgs",value.outputArgs);
        field("inputIndices",value.inputIndices);field("outputIndices",value.outputIndices);
    }
};
} }
