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
#include "canonical_template_matcher.hpp"
#include <cassert>
#include <cstring>

namespace gl {

/// @brief Whether a statement met by the witness-history walk is a forced
///        product of its block (D-522).
///
/// @details
/// A forced disintegration is the disintegration the prover performs
/// unconditionally on a statement it holds as given at its scope: a status-0
/// seed (task, subproof and case premises, contradiction heads, variable
/// copies) or the conclusion of a hypothesis shot. The deposit door lists
/// every statement such a disintegration produces in
/// `LbMemory::forcedProducts`, and a block lists a statement a parent shipped
/// with the from-premise mail flag. A connection that comes from a forced
/// disintegration does not link a witness to an instruction's path, so the
/// witness-history walk ends at a listed statement.
///
/// A visited statement matches when it is listed at its scope or at an
/// ancestor scope, or when its canonical form at its scope
/// (`CanonicalTemplateMatcher::canonicalAdmissionText`) equals the canonical
/// form, at that scope, of a product listed there or at an ancestor scope: a
/// class rewrite of a forced product is a forced product too. The canonical
/// forms of the listed products are computed at the first canonical query of
/// a scope and kept for the matcher's lifetime, so one matcher serves one
/// walk under one class state.
///
/// The walk can visit a statement at a scope this block never registered: a
/// mailed history row cites the sender's scopes, and the absorb copies the
/// rows into the origin interner only. Such a scope is read as its deepest
/// registered `_boundary_` prefix, the scope as far as this block knows it
/// (its listings and its classes); with no registered prefix it holds no
/// product of the block.
///
/// @invariant Mints nothing in the block; its storage rides the caller's
///            page-tier and string scratch arenas.
/// @see LbMemory::forcedProducts, DerivationTree::walk,
///      AdmissionMemory::witnessHistoryRefuses.
class ForcedProductMatch {
public:
    /// @brief Bind a matcher to one block for one walk.
    ///
    /// @details
    /// The matcher reads the block's forced-product list, its NameMap and its
    /// canonical view; it keeps the canonical forms it computes on @p pages
    /// and builds transient texts on @p strings. Both arenas must outlive the
    /// matcher.
    ///
    /// @param mb        Owning block (resident, single owner).
    /// @param canonical The block's canonical template matcher.
    /// @param pages     Page-tier scratch arena for the kept forms.
    /// @param strings   String scratch arena for transient texts.
    /// @return (constructor)
    ForcedProductMatch(Memory& mb, CanonicalTemplateMatcher& canonical,
                       ScratchArena& pages, ScratchArena& strings)
        : mb_(mb), canonical_(canonical), strings_(strings),
          forms_(&pages, &dirty_), builtScopes_(&pages, &dirty_) {}
    ForcedProductMatch(const ForcedProductMatch&) = delete;
    ForcedProductMatch& operator=(const ForcedProductMatch&) = delete;

    /// @brief Whether a visited statement is a forced product of the block.
    ///
    /// @details
    /// Exact first: the statement listed at @p scope or at one of its
    /// ancestor scopes. Otherwise canonical: the statement's canonical form
    /// at @p scope among the canonical forms, at @p scope, of the products
    /// listed at @p scope or an ancestor scope. An unregistered @p scope is
    /// read as its deepest registered prefix (`knownScope`).
    ///
    /// @param expr  The visited statement.
    /// @param scope The scope the walk visits it at.
    /// @return True iff the statement is a forced product of the block.
    /// @invariant Mints nothing in the block.
    /// @see knownScope, ensureScope, scopeChain.
    bool matches(StrSpan expr, StrSpan scope) {
        const auto& listed = mb_.lbMemory.forcedProducts;
        if (listed.count() == 0) return false;
        const NameId scopeId = knownScope(scope);
        if (scopeId == 0) return false;
        NameId chain[kMaxScopeDepth];
        const int32_t chainN = scopeChain(scopeId, chain);
        const NameId exprId = mb_.nameMap.lookup(expr);
        if (exprId != 0)
            for (int32_t c = 0; c < chainN; ++c)
                if (listed.lookup(packStatementKey(exprId, chain[c])) != 0) return true;
        ensureScope(scopeId, chain, chainN);
        ScratchScope frame(strings_);
        const ScratchString form = canonical_.canonicalAdmissionText(mb_, expr, scopeId, strings_);
        return forms_.lookup(formKey(scopeId, StrSpan(form))) != 0;
    }

private:
    /// @brief Bound on the nesting depth of a scope, root included.
    static constexpr int32_t kMaxScopeDepth = 256;

    /// @brief The scope as far as this block knows it.
    ///
    /// @details
    /// The scope's own id when the block registered it; otherwise the id of
    /// its deepest registered prefix ending before a `_boundary_` separator
    /// (a mailed history row can cite a sender scope the block never
    /// registered); otherwise 0: neither the scope nor any prefix is known,
    /// so no product of the block is listed there and no class of the block
    /// applies.
    ///
    /// @param scope A scope text.
    /// @return The registered id, or 0.
    /// @invariant Mints nothing.
    NameId knownScope(StrSpan scope) const {
        const NameId direct = mb_.nameMap.lookup(scope);
        if (direct != 0) return direct;
        static constexpr char kSep[] = "_boundary_";
        constexpr int32_t sepLen = static_cast<int32_t>(sizeof(kSep) - 1);
        for (int32_t at = scope.len - sepLen; at > 0; --at) {
            if (std::memcmp(scope.ptr + at, kSep, static_cast<std::size_t>(sepLen)) != 0) continue;
            const NameId prefix = mb_.nameMap.lookup(StrSpan(scope.ptr, at));
            if (prefix != 0) return prefix;
        }
        return 0;
    }

    /// @brief The scope and its ancestor scopes, innermost first.
    ///
    /// @details Follows `NameMap::parentOf` from @p scopeId up to the root.
    ///
    /// @param scopeId A registered scope.
    /// @param chain   Out: the scope ids; room for `kMaxScopeDepth`.
    /// @return The number of ids written.
    /// @invariant Asserts the nesting bound.
    int32_t scopeChain(NameId scopeId, NameId* chain) const {
        int32_t n = 0;
        for (NameId s = scopeId; s != 0; s = mb_.nameMap.parentOf(s)) {
            assert(n < kMaxScopeDepth && "ForcedProductMatch: scope nesting exceeds its bound");
            chain[n++] = s;
        }
        return n;
    }

    /// @brief Compute, once per scope, the canonical forms of the products
    ///        visible there.
    ///
    /// @details
    /// A product is visible at @p scopeId when it is listed there or at an
    /// ancestor scope (@p chain). Its canonical form at @p scopeId joins the
    /// kept forms under the scope's key.
    ///
    /// @param scopeId The scope of the canonical query.
    /// @param chain   The scope and its ancestors (`scopeChain`).
    /// @param chainN  The number of ids in @p chain.
    /// @return Nothing.
    void ensureScope(NameId scopeId, const NameId* chain, int32_t chainN) {
        for (int32_t i = 0; i < builtScopes_.size(); ++i)
            if (builtScopes_[i] == scopeId) return;
        builtScopes_.push_back(scopeId);
        const auto& listed = mb_.lbMemory.forcedProducts;
        for (int32_t id = 1; id <= listed.count(); ++id) {
            const StatementKey key = Codec<StatementKey>::decode(listed.keyAt(id));
            bool visible = false;
            for (int32_t c = 0; c < chainN && !visible; ++c) visible = key.validity == chain[c];
            if (!visible) continue;
            ScratchScope frame(strings_);
            const ScratchString form = canonical_.canonicalAdmissionText(mb_,
                mb_.nameMap.decodeView(key.orig), scopeId, strings_);
            forms_.mint(formKey(scopeId, StrSpan(form)));
        }
    }

    /// @brief The key of a canonical form at a scope: the scope id's four
    ///        bytes followed by the form.
    ///
    /// @details The bytes ride the string scratch arena; the caller's frame
    /// releases them.
    ///
    /// @param scopeId The scope.
    /// @param form    The canonical form.
    /// @return The key bytes.
    StrSpan formKey(NameId scopeId, StrSpan form) {
        const int32_t len = static_cast<int32_t>(sizeof(NameId)) + form.len;
        char* bytes = strings_.allocBytes(len);
        std::memcpy(bytes, &scopeId, sizeof(NameId));
        if (form.len > 0) std::memcpy(bytes + sizeof(NameId), form.ptr, static_cast<std::size_t>(form.len));
        return StrSpan(bytes, len);
    }

    Memory& mb_;
    CanonicalTemplateMatcher& canonical_;
    ScratchArena& strings_;
    DirtyState dirty_ = DirtyState::Clean;
    ColdHashSet<BytesKeyStore> forms_;
    PagedVector<NameId> builtScopes_;
};

} // namespace gl
