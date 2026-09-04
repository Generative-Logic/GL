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

#include "../parameters.hpp"
#include "cold_hash_map.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace gl {

    /// @brief Pack two 32-bit ids into one 64-bit key: `hi` in the high 32
    ///        bits, `lo` in the low 32 bits.
    ///
    /// @details
    /// The single well-defined primitive behind every `(int32_t, int32_t) ->
    /// int64_t` map-key packer in the engine (`packLbStateKey`,
    /// `packOriginKey`, the mapping-table keys, and `Codec<LbStatePairKey>`).
    /// Both halves are widened through `uint32_t` before the shift, so the
    /// operation is defined for negative ids too: a plain
    /// `static_cast<int64_t>(hi) << 32` is undefined behaviour when `hi` is
    /// negative, and a `-1` sentinel does reach these keys. The produced bit
    /// pattern is byte-identical to the former signed-shift form on a two's-
    /// complement target, so every previously stored key value is preserved;
    /// the corresponding `decode` twins already used the unsigned form.
    ///
    /// @param hi High 32 bits of the key (any `int32_t`, sentinels included).
    /// @param lo Low 32 bits of the key (any `int32_t`).
    /// @return The packed 64-bit key.
    /// @invariant `packInt32Pair(a, b) == packInt32Pair(c, d)` iff
    ///            `a == c && b == d` (bijective on the `(hi, lo)` pair).
    /// @see `packLbStateKey`, `packOriginKey`, `Codec<LbStatePairKey>`.
    constexpr int64_t packInt32Pair(int32_t hi, int32_t lo) noexcept {
        return static_cast<int64_t>(
            (static_cast<uint64_t>(static_cast<uint32_t>(hi)) << 32)
            | static_cast<uint64_t>(static_cast<uint32_t>(lo)));
    }

    /// @brief Per-type serialization policy that bridges a self-documenting C++
    ///        key/record type to the type-erased cold-map family.
    ///
    /// @details
    /// `Codec<T>` is the single place a composite key or a record value's
    /// canonical byte/scalar form is defined. The primary template is left
    /// UNDEFINED on purpose: every key or record type used with `TypedCold` must
    /// provide an explicit specialization, so a missing codec is a compile error
    /// at the wrapper instantiation, never a silent fallback.
    ///
    /// A KEY codec provides: `using KeyStore` (the cold-map family key store
    /// backing `K` — `BytesKeyStore` for a variable-length key, `PodKeyStore<S>`
    /// for a key packing to a fixed scalar), `using Encoded` (the stored form held
    /// in a caller local so the `StrSpan` a byte key hands the engine never
    /// dangles), `static Encoded encode(const K&)`, `static KeyView view(const
    /// Encoded&)`, `static K decode(KeyDecode)`. The reusable bases
    /// `BytesKeyCodecBase` / `PackedKeyCodecBase<S>` / `IdentityKeyCodec<S>` supply
    /// `KeyStore` / `Encoded` / `view`.
    ///
    /// A RECORD codec (used by a blob-map `TypedCold`) provides
    /// `static std::vector<char> serialize(const Record&)` and
    /// `static Record deserialize(const char*, int32_t)` — a deterministic,
    /// field-ordered round trip. A record codec carries NO comparator: the
    /// decoded-order set comparators the prover's record-set values use stay a
    /// per-call argument at the typed surface, never stored in the codec.
    ///
    /// @tparam T The key or record type the specialization serializes.
    /// @see `BytesKeyCodecBase`, `PackedKeyCodecBase`, `IdentityKeyCodec`,
    ///      `TypedCold`, D-171.
    template <typename T>
    struct Codec;

    /// @brief Reusable base for a variable-length BYTE key — backs `K` on a
    ///        `BytesKeyStore`.
    ///
    /// @details
    /// A concrete byte-key codec inherits this and adds only `encode(const K&) ->
    /// std::string` (the canonical byte layout) and `decode(StrSpan) -> K`. The
    /// stored form is a `std::string` the caller keeps alive while probing, and
    /// `view` spans it — so the engine's `StrSpan` never outlives its backing
    /// bytes.
    ///
    /// @see `Codec`, `BytesKeyStore`.
    struct BytesKeyCodecBase {
        /// @brief Cold-map family key store for a byte key.
        using KeyStore = BytesKeyStore;

        /// @brief Caller-side stored form held alive for the probe's duration.
        using Encoded = std::string;

        /// @brief View the stored bytes as the engine's probe span.
        ///
        /// @param e The encoded bytes (a caller local, alive for the call).
        /// @return A span over `e`.
        static StrSpan view(const std::string& e) { return StrSpan(e); }
    };

    /// @brief Reusable base for a key that packs to a fixed scalar `S` — backs `K`
    ///        on a `PodKeyStore<S>`.
    ///
    /// @details
    /// A concrete packed-key codec inherits this and adds only `encode(const K&)
    /// -> S` (the pack) and `decode(const S&) -> K` (the unpack). The stored form
    /// IS the scalar, so `view` is the identity — no temporary lifetime concern.
    ///
    /// @tparam S The fixed scalar the key packs to (`int32_t`, `int64_t`).
    /// @see `Codec`, `PodKeyStore`.
    template <typename S>
    struct PackedKeyCodecBase {
        /// @brief Cold-map family key store for a packed scalar key.
        using KeyStore = PodKeyStore<S>;

        /// @brief Caller-side stored form — the packed scalar itself.
        using Encoded = S;

        /// @brief Identity view — the scalar is already the engine's probe.
        ///
        /// @param e The packed scalar.
        /// @return `e`.
        static S view(const S& e) { return e; }
    };

    /// @brief Codec for a key that IS a fixed scalar `S` (no composition) — the
    ///        identity over a `PodKeyStore<S>`.
    ///
    /// @details
    /// For maps whose key is a single id (e.g. a validity-scope `int16_t`): the
    /// stored form, the probe, and the decoded key are all the scalar itself.
    /// Inherit it in a one-line `Codec<S>` specialization. Because `K == KeyView`
    /// here, `TypedCold`'s raw-`KeyView` overloads are SFINAE-suppressed for these
    /// instantiations (the typed `(const K&)` surface already takes the scalar).
    ///
    /// @tparam S The scalar key type.
    /// @see `Codec`, `PackedKeyCodecBase`.
    template <typename S>
    struct IdentityKeyCodec {
        /// @brief Cold-map family key store for the scalar key.
        using KeyStore = PodKeyStore<S>;

        /// @brief Caller-side stored form — the scalar itself.
        using Encoded = S;

        /// @brief Encode is the identity.
        ///
        /// @param k The key.
        /// @return `k`.
        static S encode(const S& k) { return k; }

        /// @brief Identity view.
        ///
        /// @param e The scalar.
        /// @return `e`.
        static S view(const S& e) { return e; }

        /// @brief Decode is the identity.
        ///
        /// @param kd The stored scalar.
        /// @return `kd`.
        static S decode(const S& kd) { return kd; }
    };

    /// @brief The ONE typed owning wrapper over the cold-map engine — a
    ///        self-documenting façade keyed by a real type `K` rather than the
    ///        type-erased `BytesKeyStore` / `PodKeyStore` the engine stores.
    ///
    /// @details
    /// `TypedCold` mirrors the engine's own collapse to a single class: just as
    /// `HashMap<KeyStore, ValueStore>` is one class surfaced through four aliases
    /// (`ColdHashSet` / `ColdHashMap` / `ColdSetMap` / `ColdBlobMap`), `TypedCold`
    /// is one class surfaced through the four aliases `TypedColdSet` /
    /// `TypedColdMap` / `TypedColdSetMap` / `TypedColdBlobMap`. It owns exactly one
    /// inner `HashMap` (zero added bytes, zero new deload tags) and re-exports its
    /// facet view types, so `LbMemory` constructs the deload facets against
    /// `&map.inner()` and the on-disk bytes are a pure function of the keys the
    /// codec produces — byte-identical to the hand-rolled layout when the codec
    /// reproduces it.
    ///
    /// Every shape's method surface lives on this one class. Shape-specific
    /// methods (a map's `find`, a set-map's `insertSorted`, a blob map's
    /// `assignRun`) are member templates that instantiate ONLY when called — the
    /// same on-demand member instantiation the engine relies on — so a set never
    /// names a value type and a single-value map never names a run. Naming a
    /// method outside its shape is a compile error exactly where the engine would
    /// raise one.
    ///
    /// The raw `KeyView` overloads (`lookup` / `insertSorted` / `assignSet` /
    /// `assignSetRange` / `setContains` / `eraseSet` taking the engine's packed
    /// scalar) let the packed-int set-map call sites pass a pre-computed
    /// `(id,id)` scalar directly (a first-class shared value). They are
    /// SFINAE-enabled only when `KeyView != K`, so an identity-keyed
    /// instantiation (`K == KeyView`, e.g. a blob map keyed by `int16_t`) keeps a
    /// single unambiguous `(const K&)` surface.
    ///
    /// @tparam K      The self-documenting key type; needs a `Codec<K>`.
    /// @tparam VS     The cold-map value-store policy (`EmptyValueStore` /
    ///                `SingleValueStore<V>` / `SetValueStore<V>` /
    ///                `BlobCsrValueStore`); selects the shape, exactly as the
    ///                engine's second parameter does.
    /// @tparam Record The record value type for a blob map; needs a
    ///                `Codec<Record>` at the call sites (not at the `LbMemory`
    ///                declaration — it is named only inside member templates, so
    ///                the wrapper instantiates with `Record` incomplete).
    ///                Defaulted `void` for the non-blob shapes.
    /// @see `Codec`, `TypedColdSet`, `TypedColdMap`, `TypedColdSetMap`,
    ///      `TypedColdBlobMap`, `HashMap`, D-171.
    template <typename K, typename VS, typename Record = void>
    class TypedCold {
    public:
        /// @brief The wrapped engine type (the type-erased cold container).
        using Inner = HashMap<typename Codec<K>::KeyStore, VS>;

        /// @brief Deload facet for a byte key's lengths tag.
        using LengthsView = typename Inner::LengthsView;
        /// @brief Deload facet for a byte key's content-bytes tag.
        using BytesView = typename Inner::BytesView;
        /// @brief Deload facet for a packed key's single key tag.
        using KeysView = typename Inner::KeysView;
        /// @brief Deload facet for a single-value map's value tag.
        using ValuesView = typename Inner::ValuesView;
        /// @brief Deload facet for a CSR map's run-start column tag.
        using RunStartsView = typename Inner::RunStartsView;
        /// @brief Deload facet for a CSR map's run-value column tag.
        using RunValuesView = typename Inner::RunValuesView;
        /// @brief Deload facet for a blob map's blob-start column tag.
        using BlobStartsView = typename Inner::BlobStartsView;
        /// @brief Deload facet for a blob map's blob-pool tag.
        using BlobPoolView = typename Inner::BlobPoolView;

        /// @brief Deload tags this container contributes (key store + value store).
        static constexpr int kTagCount = Inner::kTagCount;

        /// @brief Bind to the owning LB's arena and the aggregate dirty flag.
        ///
        /// @param arena The LB's bump arena; outlives the container.
        /// @param dirty The aggregate's shared content-change state.
        TypedCold(LbArena* arena, DirtyState* dirty) : inner_(arena, dirty) {}

        TypedCold(const TypedCold&) = delete;
        TypedCold& operator=(const TypedCold&) = delete;

        /// @brief The wrapped engine — facet binding + raw escape hatch.
        ///
        /// @return Reference to the owned engine container.
        Inner& inner() { return inner_; }

        /// @brief Const overload of `inner`.
        ///
        /// @return Const reference to the owned engine container.
        const Inner& inner() const { return inner_; }

        // ---- Lifecycle (all shapes) ----------------------------------------

        /// @brief Wholesale reset to empty — the `destroyGrid` path.
        void resetToFresh() { inner_.resetToFresh(); }

        /// @brief Drop everything including index pages — the deload release path.
        void release() { inner_.release(); }

        /// @brief Cross-arena deep copy — the LB-clone path.
        ///
        /// @param other Source container; resident, this one empty.
        void copyFrom(const TypedCold& other) { inner_.copyFrom(other.inner_); }

        /// @brief Number of keys minted.
        ///
        /// @return Key count.
        int32_t count() const { return inner_.count(); }

        /// @brief Whether no key has been minted.
        ///
        /// @return `true` when `count() == 0`.
        bool empty() const { return inner_.empty(); }

        /// @brief Approximate live byte footprint.
        ///
        /// @return Live bytes the container occupies.
        int64_t liveBytes() const { return inner_.liveBytes(); }

        // ---- Key accessors (all shapes) ------------------------------------

        /// @brief Decode an id to its RAW stored key view — the engine's untyped
        ///        keyAt, kept (a synonym of `decode`) so the sacred hashburst dump
        ///        and raw-scalar survey sites that read the packed key directly
        ///        compile unchanged (Rule 14).
        ///
        /// @param id A minted id; `1 <= id <= count()`.
        /// @return The engine key decode (a scalar for a packed key, a span for a
        ///         byte key).
        typename Inner::KeyDecode keyAt(int32_t id) const {
            return inner_.keyAt(id);
        }

        /// @brief Decode an id to its RAW stored key view — engine `decode`
        ///        synonym of `keyAt` (the sacred dump uses both names).
        ///
        /// @param id A minted id; `1 <= id <= count()`.
        /// @return The engine key decode.
        typename Inner::KeyDecode decode(int32_t id) const {
            return inner_.decode(id);
        }

        /// @brief Decode an id back to its TYPED key.
        ///
        /// @param id A minted id; `1 <= id <= count()`.
        /// @return The decoded typed key.
        K decodeKey(int32_t id) const { return Codec<K>::decode(inner_.keyAt(id)); }

        // ---- SET surface ---------------------------------------------------

        /// @brief Find-or-mint `k`.
        ///
        /// @param k The key to intern.
        /// @return The key's id; >= 1.
        int32_t mint(const K& k) {
            const typename Codec<K>::Encoded e = Codec<K>::encode(k);
            return inner_.mint(Codec<K>::view(e));
        }

        /// @brief Non-minting probe.
        ///
        /// @param k The key to look up.
        /// @return The id, or 0 when `k` was never minted.
        int32_t lookup(const K& k) const {
            const typename Codec<K>::Encoded e = Codec<K>::encode(k);
            return inner_.lookup(Codec<K>::view(e));
        }

        /// @brief Whether `k` is present.
        ///
        /// @param k The key to test.
        /// @return `true` when `k` has been minted.
        bool contains(const K& k) const {
            const typename Codec<K>::Encoded e = Codec<K>::encode(k);
            return inner_.contains(Codec<K>::view(e));
        }

        /// @brief Remove `k` if present (POD-key set + single-value map).
        ///
        /// @param k The key to remove.
        /// @return `true` when a key was removed, `false` on a miss.
        bool erase(const K& k) {
            const typename Codec<K>::Encoded e = Codec<K>::encode(k);
            return inner_.erase(Codec<K>::view(e));
        }

        /// @brief Remove every key the predicate accepts — the scope-wipe driver
        ///        (POD-key set + single-value map).
        ///
        /// @tparam Pred A callable `bool(const K&)`.
        /// @param pred The erase predicate, tested against each decoded key.
        /// @return Number of keys removed.
        template <typename Pred>
        int32_t eraseIf(Pred pred) {
            return inner_.eraseIf([&pred](typename Inner::KeyDecode kd) {
                return pred(Codec<K>::decode(kd));
            });
        }

        // ---- Single-value MAP surface (instantiates on call) ---------------

        /// @brief Look up a key's value.
        ///
        /// @tparam VS2 The value store (defaulted; do not pass).
        /// @param k The key to look up.
        /// @return Pointer to the value (stable while resident), or `nullptr` on a
        ///         miss — a defined query result, not a failure.
        template <typename VS2 = VS>
        const typename VS2::ValueType* find(const K& k) const {
            const typename Codec<K>::Encoded e = Codec<K>::encode(k);
            return inner_.find(Codec<K>::view(e));
        }

        /// @brief Look up a key's value, returning `fallback` on a miss.
        ///
        /// @tparam VS2 The value store (defaulted; do not pass).
        /// @param k        The key to look up.
        /// @param fallback The value to return when `k` is absent.
        /// @return The stored value, or `fallback`.
        template <typename VS2 = VS>
        typename VS2::ValueType findOr(const K& k,
                                       typename VS2::ValueType fallback) const {
            const typename VS2::ValueType* p = find(k);
            return p != nullptr ? *p : fallback;
        }

        /// @brief Insert a NEW key with its value (set-once; asserts `k` is new).
        ///
        /// @tparam VV The value type (deduced).
        /// @param k The key; must not already be present.
        /// @param v The value to store.
        /// @return The key's id; >= 1.
        template <typename VV>
        int32_t insert(const K& k, const VV& v) {
            const typename Codec<K>::Encoded e = Codec<K>::encode(k);
            return inner_.insert(Codec<K>::view(e), v);
        }

        /// @brief Set `k`'s value — in-place overwrite on a hit, set-once insert
        ///        on a miss (the `[k] = v` write door).
        ///
        /// @tparam VS2 The value store (defaulted; do not pass).
        /// @param k The key.
        /// @param v The value to store.
        template <typename VS2 = VS>
        void upsert(const K& k, const typename VS2::ValueType& v) {
            const typename Codec<K>::Encoded e = Codec<K>::encode(k);
            const int32_t id = inner_.lookup(Codec<K>::view(e));
            if (id != 0) inner_.setValueAt(id, v);
            else inner_.insert(Codec<K>::view(e), v);
        }

        /// @brief Overwrite an EXISTING key's value in place WITHOUT escalating
        ///        the dirty state — the parallel-safe twin of `upsert` for a map
        ///        on a never-deloaded pool whose keys are all pre-created.
        ///
        /// @details
        /// Asserts `k` is already present (never inserts — an insert would mutate
        /// structure and the shared dirty flag, defeating the point) and routes
        /// the value write through `HashMap::setValueAtRelaxed`, which leaves the
        /// dirty flag untouched, so concurrent calls on DISJOINT keys are
        /// race-free. Sanctioned ONLY for a map on a NEVER-DELOADED pool — the
        /// pull-model mail cursor, whose every (recipient, ancestor) cell is
        /// pre-created single-threaded at registration and advanced disjointly by
        /// each recipient's worker in the parallel phase-1 pull. On a deloadable
        /// map it would silently skip the rewrite the change needs — a Rule-8/19
        /// violation. See `PagedVector::setAtRelaxed` for the full contract.
        ///
        /// @tparam VS2 The value store (defaulted; do not pass).
        /// @param k The key; MUST already be present.
        /// @param v The new value.
        template <typename VS2 = VS>
        void setValueAtRelaxed(const K& k, const typename VS2::ValueType& v) {
            const typename Codec<K>::Encoded e = Codec<K>::encode(k);
            const int32_t id = inner_.lookup(Codec<K>::view(e));
            assert(id != 0
                && "TypedCold::setValueAtRelaxed on an absent key — cursor "
                   "cells must be pre-created at registration");
            inner_.setValueAtRelaxed(id, v);
        }

        /// @brief Value at a known id (single-value map).
        ///
        /// @tparam VS2 The value store (defaulted; do not pass).
        /// @param id A minted id; `1 <= id <= count()`.
        /// @return Const reference to the value.
        template <typename VS2 = VS>
        const typename VS2::ValueType& valueAt(int32_t id) const {
            return inner_.valueAt(id);
        }

        // ---- SET-MAP surface (instantiates on call) ------------------------

        /// @brief Insert `v` into `k`'s sorted-unique run under `cmp`.
        ///
        /// @tparam VS2 The value store (defaulted; do not pass).
        /// @tparam Cmp A strict-weak-ordering `bool(const V&, const V&)`.
        /// @param k   The key.
        /// @param v   The value to insert.
        /// @param cmp The run's ordering predicate (defaulted to `std::less<V>`).
        /// @return The key's id; >= 1.
        template <typename VS2 = VS,
                  typename Cmp = std::less<typename VS2::ValueType>>
        int32_t insertSorted(const K& k, const typename VS2::ValueType& v,
                             Cmp cmp = Cmp{}) {
            const typename Codec<K>::Encoded e = Codec<K>::encode(k);
            return inner_.insertSorted(Codec<K>::view(e), v, cmp);
        }

        /// @brief Replace `k`'s run with the `m` already-sorted-unique values.
        ///
        /// @tparam VS2 The value store (defaulted; do not pass).
        /// @param k    The key.
        /// @param vals The new run values, pre-sorted + unique.
        /// @param m    Value count; >= 0.
        /// @return The key's id; >= 1.
        template <typename VS2 = VS>
        int32_t assignSet(const K& k, const typename VS2::ValueType* vals,
                          int32_t m) {
            const typename Codec<K>::Encoded e = Codec<K>::encode(k);
            return inner_.assignSet(Codec<K>::view(e), vals, m);
        }

        /// @brief Replace `k`'s run with the sorted-unique values in
        ///        `[first, last)` — the `std::set` / range write door.
        ///
        /// @tparam It A forward iterator over the value type.
        /// @param k     The key.
        /// @param first Range begin.
        /// @param last  Range end.
        /// @return The key's id; >= 1.
        template <typename It>
        int32_t assignSetRange(const K& k, It first, It last) {
            const typename Codec<K>::Encoded e = Codec<K>::encode(k);
            return inner_.assignSetRange(Codec<K>::view(e), first, last);
        }

        /// @brief Whether `k`'s run contains `v`.
        ///
        /// @tparam VS2 The value store (defaulted; do not pass).
        /// @param k The key.
        /// @param v The value to test.
        /// @return `true` when `k` is present and its run contains `v`.
        template <typename VS2 = VS>
        bool setContains(const K& k, const typename VS2::ValueType& v) const {
            const typename Codec<K>::Encoded e = Codec<K>::encode(k);
            return inner_.setContains(Codec<K>::view(e), v);
        }

        /// @brief Remove `k` and its whole run (set-map).
        ///
        /// @param k The key to remove.
        /// @return `true` when a key was removed, `false` on a miss.
        bool eraseSet(const K& k) {
            const typename Codec<K>::Encoded e = Codec<K>::encode(k);
            return inner_.eraseSet(Codec<K>::view(e));
        }

        /// @brief Remove every key the predicate accepts — the run-aware
        ///        scope-wipe driver (set-map).
        ///
        /// @tparam Pred A callable `bool(const K&)`.
        /// @param pred The erase predicate, tested against each decoded key.
        /// @return Number of keys removed.
        template <typename Pred>
        int32_t eraseSetIf(Pred pred) {
            return inner_.eraseSetIf([&pred](typename Inner::KeyDecode kd) {
                return pred(Codec<K>::decode(kd));
            });
        }

        /// @brief Number of values in key `id`'s run (set-map / multimap / blob).
        ///
        /// @param id A minted id; `1 <= id <= count()`.
        /// @return The run length.
        int32_t runLen(int32_t id) const { return inner_.runLen(id); }

        /// @brief Value `j` of key `id`'s run (set-map / multimap).
        ///
        /// @tparam VS2 The value store (defaulted; do not pass).
        /// @param id A minted id; `1 <= id <= count()`.
        /// @param j  Position in `[0, runLen(id))`.
        /// @return Const reference to the value.
        template <typename VS2 = VS>
        const typename VS2::ValueType& valueAt(int32_t id, int32_t j) const {
            return inner_.valueAt(id, j);
        }

        // ---- BLOB-MAP surface (instantiates on call; uses Record) ----------

        /// @brief Replace key `k`'s run with `recs` — serialize each record
        ///        through `Codec<Record>` and replace the whole blob run.
        ///
        /// @tparam R The record type (defaulted to `Record`; do not pass).
        /// @param k    The key.
        /// @param recs The records to store, in run order.
        /// @return The key's id; >= 1.
        template <typename R = Record>
        int32_t assignRun(const K& k, const std::vector<R>& recs) {
            std::vector<char> bytes;
            std::vector<int32_t> lens;
            lens.reserve(recs.size());
            for (const R& r : recs) {
                const std::vector<char> b = Codec<R>::serialize(r);
                lens.push_back(static_cast<int32_t>(b.size()));
                bytes.insert(bytes.end(), b.begin(), b.end());
            }
            const typename Codec<K>::Encoded e = Codec<K>::encode(k);
            return inner_.assignRun(Codec<K>::view(e), bytes.data(), lens.data(),
                                    static_cast<int32_t>(recs.size()));
        }

        /// @brief Append one record to key `k`'s run-end — the RMW insert fast
        ///        path (serialize one record; splice its blob without re-encoding
        ///        the existing run).
        ///
        /// @details
        /// Semantics-identical to `recordsAt(k)` + `push_back(record)` +
        /// `assignRun` (the new record lands at the run-end), but it never
        /// decodes or re-encodes the existing run — the O(run^2)-per-key cost the
        /// encodedMap RMW used to pay. An existing key splices the one new blob at
        /// its run-end (`HashMap::appendBlobToRun`); a brand-new key opens a
        /// one-blob run (`HashMap::assignRun`). A member template, instantiated
        /// only when called.
        ///
        /// @tparam R The record type (defaulted to `Record`; do not pass).
        /// @param k      The key.
        /// @param record The record to append (lands at the run-end).
        template <typename R = Record>
        void appendRecord(const K& k, const R& record) {
            const std::vector<char> b = Codec<R>::serialize(record);
            const typename Codec<K>::Encoded e = Codec<K>::encode(k);
            const int32_t id = inner_.lookup(Codec<K>::view(e));
            if (id != 0) {
                inner_.appendBlobToRun(id, b.data(),
                                       static_cast<int32_t>(b.size()));
            } else {
                const int32_t len = static_cast<int32_t>(b.size());
                inner_.assignRun(Codec<K>::view(e), b.data(), &len, 1);
            }
        }

        /// @brief Decode record `j` of key `id`'s run.
        ///
        /// @tparam R The record type (defaulted to `Record`; do not pass).
        /// @param id A minted id; `1 <= id <= count()`.
        /// @param j  Position in `[0, runLen(id))`.
        /// @return The decoded record.
        template <typename R = Record>
        R recordAt(int32_t id, int32_t j) const {
            std::vector<char> buf;
            inner_.blobAt(id, j, buf);
            return Codec<R>::deserialize(buf.data(),
                                         static_cast<int32_t>(buf.size()));
        }

        /// @brief Decode key `id`'s whole run into a vector of records.
        ///
        /// @tparam R The record type (defaulted to `Record`; do not pass).
        /// @param id A minted id; `1 <= id <= count()`.
        /// @return The decoded records, in run order.
        template <typename R = Record>
        std::vector<R> recordsAt(int32_t id) const {
            const int32_t rl = inner_.runLen(id);
            std::vector<R> out;
            out.reserve(rl);
            for (int32_t j = 0; j < rl; ++j) out.push_back(recordAt<R>(id, j));
            return out;
        }

        /// @brief Contiguous raw bytes of record `j` of key `id`'s run — the
        ///        no-allocation peek the owner-set prune reads through instead of
        ///        the decoding `recordAt`/`recordsAt` (blob map).
        ///
        /// @details
        /// Returns a pointer to the record's canonical blob bytes the caller parses
        /// field-by-field against the `Codec<Record>` layout offsets, WITHOUT
        /// materializing a `Record`. Zero-copy when the blob is single-page (the
        /// common case for the small owner-set records — the bulk request-
        /// generation prune touches only the first byte); a page-straddling blob is
        /// copied into `scratch` once. The decode-on-every-probe cost `recordsAt`
        /// pays is thereby avoided on the hot path. Read-only; burst-safe
        /// ([I-83](../30_invariants.md#i-83)). A member template, instantiated only
        /// when called.
        ///
        /// @tparam R The record type (defaulted to `Record`; do not pass).
        /// @param id      A minted id; `1 <= id <= count()`.
        /// @param j       Position in `[0, runLen(id))`.
        /// @param len     [out] The blob's byte length.
        /// @param scratch Caller-owned reuse buffer used only on a straddle.
        /// @return Pointer to `len` contiguous record bytes (into the pool, or into
        ///         `scratch` on a straddle).
        template <typename R = Record>
        const char* peekRecordBytes(int32_t id, int32_t j, int32_t& len,
                                    std::vector<char>& scratch) const {
            return inner_.peekBlobContiguous(id, j, len, scratch);
        }

        /// @brief Contiguous raw bytes of record `j` of key `id`'s run — the
        ///        arena-backed twin of the `std::vector<char>` `peekRecordBytes`,
        ///        assembling a page-straddling record onto `scratch`'s byte-bump
        ///        tier (blob map).
        ///
        /// @details
        /// Forwards to the inner `HashMap::peekBlobContiguous` arena overload:
        /// zero-copy on the single-page common case, one FRESH (never-rewound)
        /// `scratch` allocation on a straddle, so an outer peek survives a nested
        /// re-entrant peek. The returned bytes are byte-identical to the
        /// `std::vector<char>` overload's. Read-only; burst-safe. A member
        /// template, instantiated only when called.
        ///
        /// @tparam R The record type (defaulted to `Record`; do not pass).
        /// @param id      A minted id; `1 <= id <= count()`.
        /// @param j       Position in `[0, runLen(id))`.
        /// @param len     [out] The blob's byte length.
        /// @param scratch Byte-bump arena the straddle copy is assembled onto.
        /// @return Pointer to `len` contiguous record bytes (into the pool, or
        ///         into `scratch` on a straddle).
        template <typename R = Record>
        const char* peekRecordBytes(int32_t id, int32_t j, int32_t& len,
                                    ScratchArena& scratch) const {
            return inner_.peekBlobContiguous(id, j, len, scratch);
        }

        /// @brief Contiguous raw bytes of record `j` of key `id`'s run — the
        ///        caller-buffer twin of the `std::vector<char>` /
        ///        `ScratchArena&` `peekRecordBytes` overloads (blob map).
        ///
        /// @details
        /// Forwards to the inner `HashMap::peekBlobContiguous` caller-buffer
        /// overload: zero-copy on the single-page common case, one copy into
        /// `buf` on a straddle, a hard `assert` when a straddling record
        /// exceeds `cap` (Rule 19 — a broken write-side size invariant, never
        /// a clamp). The returned bytes are byte-identical to the
        /// `std::vector<char>` overload's. Read-only; burst-safe. A member
        /// template, instantiated only when called.
        ///
        /// @tparam R The record type (defaulted to `Record`; do not pass).
        /// @param id  A minted id; `1 <= id <= count()`.
        /// @param j   Position in `[0, runLen(id))`.
        /// @param len [out] The blob's byte length.
        /// @param buf Caller-owned buffer of at least `cap` bytes; written
        ///            only on a straddle.
        /// @param cap The buffer's capacity.
        /// @return Pointer to `len` contiguous record bytes (into the pool,
        ///         or into `buf` on a straddle).
        /// @see HashMap::peekBlobContiguous — the underlying door and its
        ///      capacity contract.
        template <typename R = Record>
        const char* peekRecordBytes(int32_t id, int32_t j, int32_t& len,
                                    char* buf, int32_t cap) const {
            return inner_.peekBlobContiguous(id, j, len, buf, cap);
        }

        /// @brief Remove every key the predicate accepts — the run-aware
        ///        scope-wipe driver (blob map).
        ///
        /// @tparam Pred A callable `bool(const K&)`.
        /// @param pred The erase predicate, tested against each decoded key.
        /// @return Number of keys removed.
        template <typename Pred>
        int32_t eraseBlobIf(Pred pred) {
            return inner_.eraseBlobIf([&pred](typename Inner::KeyDecode kd) {
                return pred(Codec<K>::decode(kd));
            });
        }

        // ---- Raw packed-scalar overloads (KeyView != K only) ---------------
        // The packed (origId,validityId) / (id,id) scalar is a first-class value
        // shared across the packed-int registries, often pre-computed once per
        // call site; these let those sites pass it directly (byte-identical to the
        // engine). SFINAE-gated on KeyView != K so an identity-keyed instantiation
        // (K == KeyView) keeps one unambiguous (const K&) surface.

        /// @brief Raw-key non-minting probe.
        /// @tparam KV The engine key view (deduced from the packed scalar).
        /// @param k The packed key view.
        /// @return The id, or 0 when absent.
        template <typename KV,
                  std::enable_if_t<std::is_same<KV, typename Inner::KeyView>::value
                      && !std::is_same<KV, K>::value, int> = 0>
        int32_t lookup(KV k) const { return inner_.lookup(k); }

        /// @brief Raw-key sorted-unique insert.
        /// @tparam KV  The engine key view (deduced).
        /// @tparam VS2 The value store (defaulted; do not pass).
        /// @tparam Cmp A strict-weak-ordering callable on the value type.
        /// @param k   The packed key view.
        /// @param v   The value to insert.
        /// @param cmp The run's ordering predicate.
        /// @return The key's id; >= 1.
        template <typename KV, typename VS2 = VS,
                  typename Cmp = std::less<typename VS2::ValueType>,
                  std::enable_if_t<std::is_same<KV, typename Inner::KeyView>::value
                      && !std::is_same<KV, K>::value, int> = 0>
        int32_t insertSorted(KV k, const typename VS2::ValueType& v,
                             Cmp cmp = Cmp{}) {
            return inner_.insertSorted(k, v, cmp);
        }

        /// @brief Raw-key whole-run replace.
        /// @tparam KV  The engine key view (deduced).
        /// @tparam VS2 The value store (defaulted; do not pass).
        /// @param k    The packed key view.
        /// @param vals Pre-sorted-unique values; read only when `m > 0`.
        /// @param m    Value count; >= 0.
        /// @return The key's id; >= 1.
        template <typename KV, typename VS2 = VS,
                  std::enable_if_t<std::is_same<KV, typename Inner::KeyView>::value
                      && !std::is_same<KV, K>::value, int> = 0>
        int32_t assignSet(KV k, const typename VS2::ValueType* vals, int32_t m) {
            return inner_.assignSet(k, vals, m);
        }

        /// @brief Raw-key range replace (sorts + dedups the range).
        /// @tparam KV The engine key view (deduced).
        /// @tparam It A forward iterator over the value type.
        /// @param k     The packed key view.
        /// @param first Range begin.
        /// @param last  Range end.
        /// @return The key's id; >= 1.
        template <typename KV, typename It,
                  std::enable_if_t<std::is_same<KV, typename Inner::KeyView>::value
                      && !std::is_same<KV, K>::value, int> = 0>
        int32_t assignSetRange(KV k, It first, It last) {
            return inner_.assignSetRange(k, first, last);
        }

        /// @brief Raw-key set-membership probe.
        /// @tparam KV  The engine key view (deduced).
        /// @tparam VS2 The value store (defaulted; do not pass).
        /// @param k The packed key view.
        /// @param v The value to test.
        /// @return `true` when `k` is present and its run contains `v`.
        template <typename KV, typename VS2 = VS,
                  std::enable_if_t<std::is_same<KV, typename Inner::KeyView>::value
                      && !std::is_same<KV, K>::value, int> = 0>
        bool setContains(KV k, const typename VS2::ValueType& v) const {
            return inner_.setContains(k, v);
        }

        /// @brief Raw-key whole-run erase.
        /// @tparam KV The engine key view (deduced).
        /// @param k The packed key view.
        /// @return `true` when a key was removed, `false` on a miss.
        template <typename KV,
                  std::enable_if_t<std::is_same<KV, typename Inner::KeyView>::value
                      && !std::is_same<KV, K>::value, int> = 0>
        bool eraseSet(KV k) { return inner_.eraseSet(k); }

    private:
        Inner inner_;
    };

    /// @brief A self-documenting cold SET keyed by `K` — `TypedCold` over
    ///        `EmptyValueStore` (the `ColdHashSet` analog).
    template <typename K>
    using TypedColdSet = TypedCold<K, EmptyValueStore>;

    /// @brief A self-documenting cold single-value MAP keyed by `K` — `TypedCold`
    ///        over `SingleValueStore<V>` (the `ColdHashMap` analog).
    template <typename K, typename V>
    using TypedColdMap = TypedCold<K, SingleValueStore<V>>;

    /// @brief A self-documenting cold SET-MAP (key -> sorted-unique value set)
    ///        keyed by `K` — `TypedCold` over `SetValueStore<V>` (the `ColdSetMap`
    ///        analog).
    template <typename K, typename V>
    using TypedColdSetMap = TypedCold<K, SetValueStore<V>>;

    /// @brief A self-documenting cold BLOB-MAP (key -> run of variable-length
    ///        records) keyed by `K` — `TypedCold` over `BlobCsrValueStore` (the
    ///        `ColdBlobMap` analog), with `Record` (de)serialized through
    ///        `Codec<Record>`.
    template <typename K, typename Record>
    using TypedColdBlobMap = TypedCold<K, BlobCsrValueStore, Record>;

    // ===================================================================
    //  Concrete KEY codecs (self-contained — pure int serialization; the
    //  record VALUE codecs live beside their record types in memory.hpp).
    // ===================================================================

    /// @brief Generic `int16_t` identity key over `PodKeyStore<int16_t>` — kept
    ///        for any true 16-bit key domain (validity-id maps now key on
    ///        `NameId` via `Codec<int32_t>`).
    template <>
    struct Codec<int16_t> : IdentityKeyCodec<int16_t> {};

    /// @brief Bare `NameId` (`int32_t`) key — the identity over
    ///        `PodKeyStore<int32_t>`.
    ///
    /// @details
    /// Serves the containers keyed by a single NameMap id — the validity-id
    /// key of `equivalenceClassesMap` is the production instance. The stored
    /// form, the probe, and the decoded key are all the `int32_t` itself.
    /// (The admission / rejection subsystem's packed `(templateId, validityId)`
    /// keys are `int64_t` since the `NameId` widening — `mintTemplateKey` packs
    /// two `NameId` halves via `packInt32Pair` — so those containers key
    /// through `Codec<int64_t>` below, not this specialization.)
    ///
    /// @see `Codec<int64_t>`, `Codec<StatementKey>`, `IdentityKeyCodec`.
    template <>
    struct Codec<int32_t> : IdentityKeyCodec<int32_t> {};

    /// @brief Pre-packed `(expressionId, validityId)` `int64_t` key — the identity
    ///        over `PodKeyStore<int64_t>`.
    ///
    /// @details
    /// The origin-history map (`Memory::exprOriginMap`) keys on a packed
    /// `(uint32(expressionId) << 32) | uint32(validityId)` scalar that
    /// `mintOriginKey` produces, so the cold container keys on that `int64_t`
    /// directly: the stored form, the probe, and the decoded key are all the
    /// scalar itself. `PodKeyStore<int64_t>` is already a production key store
    /// (the `(high, low)` pair key `Codec<LbStatePairKey>` packs to `int64_t`);
    /// this is the bare-scalar identity twin of it.
    ///
    /// @see `Codec<LbStatePairKey>`, `IdentityKeyCodec`, I-121.
    template <>
    struct Codec<int64_t> : IdentityKeyCodec<int64_t> {};

    /// @brief A 2-field equivalence-class index key — a validity-scope id plus the
    ///        class's sorted member ids — keyed on a `BytesKeyStore`.
    ///
    /// @details
    /// The typed form of the former `eqClassSttmntIndexMapMap` byte key. Replaces
    /// the free `encodeEqClassKey` helper.
    ///
    /// @see `Codec<EqClassKey>`.
    struct EqClassKey {
        /// @brief The class's validity-scope id.
        NameId validity;
        /// @brief The class's sorted member ids.
        std::vector<NameId> members;

        /// @brief Value equality (for tests + decoded-key comparisons).
        ///
        /// @param o The other key.
        /// @return `true` when both fields match.
        bool operator==(const EqClassKey& o) const {
            return validity == o.validity && members == o.members;
        }
    };

    /// @brief Codec for `EqClassKey` — the byte layout `encodeEqClassKey`
    ///        produced: fixed-width `sizeof(NameId)`-byte little-endian `validity`
    ///        then each `member`, no separators.
    ///
    /// @see `EqClassKey`, `BytesKeyCodecBase`, I-117.
    template <>
    struct Codec<EqClassKey> : BytesKeyCodecBase {
        /// @brief Encode to the canonical byte key.
        ///
        /// @param k The key.
        /// @return `sizeof(NameId)*(members.size()+1)` bytes: `validity` then
        ///         each `member`.
        static std::string encode(const EqClassKey& k) {
            std::string key(sizeof(NameId) * (k.members.size() + 1), '\0');
            std::memcpy(&key[0], &k.validity, sizeof(NameId));
            for (std::size_t i = 0; i < k.members.size(); ++i)
                std::memcpy(&key[sizeof(NameId) * (i + 1)], &k.members[i],
                            sizeof(NameId));
            return key;
        }

        /// @brief Decode the byte key back to the typed form.
        ///
        /// @param s The stored bytes (a multiple of `sizeof(NameId)`, at least
        ///          `sizeof(NameId)`).
        /// @return The decoded key.
        static EqClassKey decode(StrSpan s) {
            assert(s.len >= static_cast<int32_t>(sizeof(NameId))
                && (s.len % static_cast<int32_t>(sizeof(NameId))) == 0
                && "Codec<EqClassKey>::decode: malformed key length");
            EqClassKey k{};
            std::memcpy(&k.validity, s.ptr, sizeof(NameId));
            const int32_t n =
                s.len / static_cast<int32_t>(sizeof(NameId)) - 1;
            k.members.resize(static_cast<std::size_t>(n));
            for (int32_t i = 0; i < n; ++i)
                std::memcpy(&k.members[static_cast<std::size_t>(i)],
                            s.ptr + sizeof(NameId) * (i + 1), sizeof(NameId));
            return k;
        }
    };

    /// @brief A 2-field statement-registry key — an original/template id (high
    ///        32 bits) plus a validity id (low 32 bits) — packed to one `int64_t`.
    ///
    /// @details
    /// The typed form of the `packStatementKey` packed key
    /// (`intToBeProved` / `intStatementLevelsMap` / `intKnownStatements` /
    /// the admission-rejected subsystem). Both halves are `NameId` (32-bit), so
    /// the pack is an `int64_t` — the same `packInt32Pair` primitive
    /// `Codec<LbStatePairKey>` uses.
    ///
    /// @see `Codec<StatementKey>`.
    struct StatementKey {
        /// @brief Original / template id (the high 32 bits).
        NameId orig;
        /// @brief Validity-scope id (the low 32 bits).
        NameId validity;

        /// @brief Value equality (for tests + decoded-key comparisons).
        ///
        /// @param o The other key.
        /// @return `true` when both fields match.
        bool operator==(const StatementKey& o) const {
            return orig == o.orig && validity == o.validity;
        }
    };

    /// @brief Codec for `StatementKey` — the scalar `packStatementKey` produced:
    ///        `packInt32Pair(orig, validity)` (`orig` in the high 32 bits,
    ///        `validity` in the low 32).
    ///
    /// @see `StatementKey`, `PackedKeyCodecBase`, `packInt32Pair`.
    template <>
    struct Codec<StatementKey> : PackedKeyCodecBase<int64_t> {
        /// @brief Pack to the canonical `int64_t` key.
        ///
        /// @param k The key.
        /// @return The packed scalar.
        static int64_t encode(const StatementKey& k) {
            return packInt32Pair(k.orig, k.validity);
        }

        /// @brief Unpack the scalar back to the typed form.
        ///
        /// @param s The packed scalar.
        /// @return The decoded key.
        static StatementKey decode(const int64_t& s) {
            return StatementKey{
                static_cast<NameId>(static_cast<uint64_t>(s) >> 32),
                static_cast<NameId>(static_cast<uint32_t>(
                    static_cast<uint64_t>(s) & 0xFFFFFFFFu)) };
        }
    };

    /// @brief A 2-field LB-state-interner pair key — a high id and a low id —
    ///        packed to one `int64_t`.
    ///
    /// @details
    /// The typed form of the `packLbStateKey` packed key (`orBookkeeping` keyed by
    /// `(exprId, cohortId)`, where the cohort id contains parent + signature;
    /// `expandedImplications` keyed by
    /// `(maybeAncestor, descendant)`).
    ///
    /// @see `Codec<LbStatePairKey>`.
    struct LbStatePairKey {
        /// @brief The high 32 bits.
        int32_t high;
        /// @brief The low 32 bits.
        int32_t low;

        /// @brief Value equality (for tests + decoded-key comparisons).
        ///
        /// @param o The other key.
        /// @return `true` when both fields match.
        bool operator==(const LbStatePairKey& o) const {
            return high == o.high && low == o.low;
        }
    };

    /// @brief Codec for `LbStatePairKey` — the scalar `packLbStateKey` produced:
    ///        `(int64(high) << 32) | uint32(low)`.
    ///
    /// @see `LbStatePairKey`, `PackedKeyCodecBase`.
    template <>
    struct Codec<LbStatePairKey> : PackedKeyCodecBase<int64_t> {
        /// @brief Pack to the canonical `int64_t` key.
        ///
        /// @param k The key.
        /// @return The packed scalar.
        static int64_t encode(const LbStatePairKey& k) {
            return packInt32Pair(k.high, k.low);
        }

        /// @brief Unpack the scalar back to the typed form.
        ///
        /// @param s The packed scalar.
        /// @return The decoded key.
        static LbStatePairKey decode(const int64_t& s) {
            return LbStatePairKey{
                static_cast<int32_t>(static_cast<uint64_t>(s) >> 32),
                static_cast<int32_t>(static_cast<uint32_t>(
                    static_cast<uint64_t>(s) & 0xFFFFFFFFu)) };
        }
    };

    /// @brief The owning form of a normalized hash-engine key — the variable-length
    ///        `int16_t` premise key that `encodedMap` and the two owner-set maps
    ///        key on — backed by a `BytesKeyStore`.
    ///
    /// @details
    /// The stored twin of the transient probe `IntNormalizedKey` (which carries a
    /// non-owning `const int16_t* data` into a `KeyArena`/`g_reqKeyArena`). The cold
    /// store owns the key bytes, so the persistent storage migrates from the
    /// `KeyArena` onto this codec's `BytesKeyStore` — the typed key owns its `data`,
    /// the hot path keeps probing through `IntNormalizedKey` via the raw-`StrSpan`
    /// overloads. Identity matches `IntNormalizedKey::operator==`: equal
    /// `numberExpressions` AND equal `data`.
    ///
    /// @see `Codec<NormKey>`, `IntNormalizedKey`.
    struct NormKey {
        /// @brief The leading expression-count field of the normalized key.
        int32_t numberExpressions;
        /// @brief The key's `NameId` payload; `data.size()` is the key length.
        std::vector<NameId> data;

        /// @brief Value equality (for tests + decoded-key comparisons).
        ///
        /// @param o The other key.
        /// @return `true` when both fields match.
        bool operator==(const NormKey& o) const {
            return numberExpressions == o.numberExpressions && data == o.data;
        }
    };

    /// @brief Hash for `NormKey` as a standard unordered-container key.
    ///
    /// @details
    /// FNV-1a over `numberExpressions` then the `data` payload — consistent with
    /// `NormKey::operator==`. Used by `remainingArgsNormalizedEncodedMap`'s inner
    /// set once that migrated off the keyArena-backed `IntNormalizedKey` onto the
    /// owning `NormKey` (the keyArena retirement). Independent of the cold map's
    /// `PagedHashIndex` (which hashes the encoded key bytes directly).
    struct NormKeyHash {
        /// @brief Hash a `NormKey`.
        /// @param k The key.
        /// @return The hash.
        std::size_t operator()(const NormKey& k) const {
            std::size_t h = 1469598103934665603ULL;
            const auto mix = [&h](NameId v) {
                h ^= static_cast<std::size_t>(
                    static_cast<std::uint32_t>(v));
                h *= 1099511628211ULL;
            };
            mix(k.numberExpressions);
            for (const NameId v : k.data) mix(v);
            return h;
        }
    };

    /// @brief Codec for `NormKey` — the byte layout `NameId numberExpressions ++
    ///        NameId length ++ length×NameId data`, all fixed-width little-endian
    ///        (the whole record is uniform `sizeof(NameId)`).
    ///
    /// @details
    /// The `length` field is redundant with the byte count but kept so the decode
    /// can assert the framing; the layout is injective (distinct
    /// `(numberExpressions, data)` produce distinct bytes) and deterministic.
    ///
    /// @see `NormKey`, `BytesKeyCodecBase`.
    template <>
    struct Codec<NormKey> : BytesKeyCodecBase {
        /// @brief Encode to the canonical byte key.
        ///
        /// @param k The key.
        /// @return `sizeof(NameId)*(k.data.size()+2)` bytes: `numberExpressions`,
        ///         `length`, data.
        static std::string encode(const NormKey& k) {
            const int32_t numExpr = k.numberExpressions;
            const int32_t length = static_cast<int32_t>(k.data.size());
            std::string key(sizeof(NameId) * (k.data.size() + 2), '\0');
            std::memcpy(&key[0], &numExpr, sizeof(NameId));
            std::memcpy(&key[sizeof(NameId)], &length, sizeof(NameId));
            for (std::size_t i = 0; i < k.data.size(); ++i)
                std::memcpy(&key[sizeof(NameId) * (i + 2)], &k.data[i],
                            sizeof(NameId));
            return key;
        }

        /// @brief Decode the byte key back to the typed form.
        ///
        /// @param s The stored bytes (a multiple of `sizeof(NameId)`, at least
        ///          `2*sizeof(NameId)`).
        /// @return The decoded key.
        static NormKey decode(StrSpan s) {
            assert(s.len >= 2 * static_cast<int32_t>(sizeof(NameId))
                && (s.len % static_cast<int32_t>(sizeof(NameId))) == 0
                && "Codec<NormKey>::decode: malformed key length");
            NormKey k{};
            std::memcpy(&k.numberExpressions, s.ptr, sizeof(NameId));
            int32_t length = 0;
            std::memcpy(&length, s.ptr + sizeof(NameId), sizeof(NameId));
            const int32_t n =
                s.len / static_cast<int32_t>(sizeof(NameId)) - 2;
            assert(n == length && "Codec<NormKey>::decode: length field mismatch");
            k.data.resize(static_cast<std::size_t>(n));
            for (int32_t i = 0; i < n; ++i)
                std::memcpy(&k.data[static_cast<std::size_t>(i)],
                            s.ptr + sizeof(NameId) * (i + 2), sizeof(NameId));
            return k;
        }

        /// @brief Serialize as a blob-map record — byte-identical to the key
        ///        `encode`, so a NormKey value-blob equals the same NormKey's key
        ///        bytes (the byte-peek membership lever relies on this identity).
        ///
        /// @param k The key.
        /// @return The record bytes.
        static std::vector<char> serialize(const NormKey& k) {
            const std::string s = encode(k);
            return std::vector<char>(s.begin(), s.end());
        }

        /// @brief Deserialize a blob-map record back to a NormKey.
        ///
        /// @param data Blob bytes.
        /// @param n    Blob length.
        /// @return The decoded key.
        static NormKey deserialize(const char* data, int32_t n) {
            return decode(StrSpan(data, n));
        }
    };

    /// @brief An ascending set of `NameId` ids as one byte key — the
    ///        `remainingArgsNormalizedEncodedMap` key (the former
    ///        `std::set<NameId>`) — backed by a `BytesKeyStore`.
    ///
    /// @details
    /// The caller passes `ids` in ascending order (the `std::set<NameId>`
    /// iteration order), so the byte form is the canonical set representation.
    /// The historical `Int16SetKey` name is kept even though the ids are now
    /// `NameId` (32-bit), to bound the migration diff — the remaining-arg ids are
    /// NameMap ids and can exceed the old 16-bit ceiling.
    ///
    /// @see `Codec<Int16SetKey>`.
    struct Int16SetKey {
        /// @brief The set's ids, ascending.
        std::vector<NameId> ids;

        /// @brief Value equality (for tests + decoded-key comparisons).
        ///
        /// @param o The other key.
        /// @return `true` when the id vectors match.
        bool operator==(const Int16SetKey& o) const { return ids == o.ids; }
    };

    /// @brief Codec for `Int16SetKey` — the byte layout `NameId count ++
    ///        count×NameId` ascending, all fixed-width little-endian.
    ///
    /// @see `Int16SetKey`, `BytesKeyCodecBase`.
    template <>
    struct Codec<Int16SetKey> : BytesKeyCodecBase {
        /// @brief Encode to the canonical byte key.
        ///
        /// @param k The key.
        /// @return `sizeof(NameId)*(k.ids.size()+1)` bytes: `count` then each id.
        static std::string encode(const Int16SetKey& k) {
            const NameId count = static_cast<NameId>(k.ids.size());
            std::string key(sizeof(NameId) * (k.ids.size() + 1), '\0');
            std::memcpy(&key[0], &count, sizeof(NameId));
            for (std::size_t i = 0; i < k.ids.size(); ++i)
                std::memcpy(&key[sizeof(NameId) * (i + 1)], &k.ids[i],
                            sizeof(NameId));
            return key;
        }

        /// @brief Decode the byte key back to the typed form.
        ///
        /// @param s The stored bytes (a multiple of `sizeof(NameId)`, at least
        ///          `sizeof(NameId)`).
        /// @return The decoded key.
        static Int16SetKey decode(StrSpan s) {
            assert(s.len >= static_cast<int32_t>(sizeof(NameId))
                && (s.len % static_cast<int32_t>(sizeof(NameId))) == 0
                && "Codec<Int16SetKey>::decode: malformed key length");
            Int16SetKey k{};
            NameId count = 0;
            std::memcpy(&count, s.ptr, sizeof(NameId));
            const int32_t n =
                s.len / static_cast<int32_t>(sizeof(NameId)) - 1;
            assert(n == count && "Codec<Int16SetKey>::decode: count mismatch");
            k.ids.resize(static_cast<std::size_t>(n));
            for (int32_t i = 0; i < n; ++i)
                std::memcpy(&k.ids[static_cast<std::size_t>(i)],
                            s.ptr + sizeof(NameId) * (i + 1), sizeof(NameId));
            return k;
        }
    };

    /// @brief A vector of `int32_t` ids as one byte key — the `originals` element
    ///        (an implication chain as `ruleInterner` id vector) — backed by a
    ///        `BytesKeyStore`.
    ///
    /// @details
    /// `originals` is a `std::set<std::vector<int32_t>>`; each element becomes one
    /// `IdVecKey`. The ids keep their positional (rule) order — they are NOT
    /// re-sorted — so a dump that needs decoded-lex order sorts the decoded vectors
    /// itself.
    ///
    /// @see `Codec<IdVecKey>`.
    struct IdVecKey {
        /// @brief The id vector, in positional order.
        std::vector<int32_t> ids;

        /// @brief Value equality (for tests + decoded-key comparisons).
        ///
        /// @param o The other key.
        /// @return `true` when the id vectors match.
        bool operator==(const IdVecKey& o) const { return ids == o.ids; }
    };

    /// @brief Codec for `IdVecKey` — the byte layout `int32 count ++ count×int32`,
    ///        all fixed-width little-endian.
    ///
    /// @see `IdVecKey`, `BytesKeyCodecBase`.
    template <>
    struct Codec<IdVecKey> : BytesKeyCodecBase {
        /// @brief Encode to the canonical byte key.
        ///
        /// @param k The key.
        /// @return `4*(k.ids.size()+1)` bytes: `count` then each id.
        static std::string encode(const IdVecKey& k) {
            const int32_t count = static_cast<int32_t>(k.ids.size());
            std::string key(sizeof(int32_t) * (k.ids.size() + 1), '\0');
            std::memcpy(&key[0], &count, sizeof(int32_t));
            for (std::size_t i = 0; i < k.ids.size(); ++i)
                std::memcpy(&key[sizeof(int32_t) * (i + 1)], &k.ids[i],
                            sizeof(int32_t));
            return key;
        }

        /// @brief Decode the byte key back to the typed form.
        ///
        /// @param s The stored bytes (a multiple of 4, at least 4).
        /// @return The decoded key.
        static IdVecKey decode(StrSpan s) {
            assert(s.len >= static_cast<int32_t>(sizeof(int32_t))
                && (s.len % static_cast<int32_t>(sizeof(int32_t))) == 0
                && "Codec<IdVecKey>::decode: malformed key length");
            IdVecKey k{};
            int32_t count = 0;
            std::memcpy(&count, s.ptr, sizeof(int32_t));
            const int32_t n =
                s.len / static_cast<int32_t>(sizeof(int32_t)) - 1;
            assert(n == count && "Codec<IdVecKey>::decode: count mismatch");
            k.ids.resize(static_cast<std::size_t>(n));
            for (int32_t i = 0; i < n; ++i)
                std::memcpy(&k.ids[static_cast<std::size_t>(i)],
                            s.ptr + sizeof(int32_t) * (i + 1), sizeof(int32_t));
            return k;
        }
    };

    /// @brief Zero-copy view over an `IdVecKey`'s stored bytes — reads the
    ///        `int32 count ++ count x int32` layout WITHOUT the decoding
    ///        `Codec<IdVecKey>::decode` (which allocates a `std::vector<int32_t>`).
    ///
    /// @details
    /// The cold-blob-view peer of `Codec<IdVecKey>::decode`. The `originals`
    /// rule-registry is a `TypedColdSet<IdVecKey>` whose keys are these blobs;
    /// `checkNecessityForEquality` reads a chain key's ids straight off
    /// `inner().keyAt(id)` through this view (into a stack buffer, before any
    /// mint into the same set), and `sortOriginalChainIndex` compares chain keys
    /// the same way — neither materializes a heap `IdVecKey`. `count()` is the
    /// leading `int32`; `idAt(k)` is the `int32` at byte offset `4 + 4k`
    /// (unaligned-safe `memcpy`). The view aliases the caller's bytes and
    /// inherits their lifetime — valid only while the source key blob is resident
    /// and unmutated (I-116).
    ///
    /// @see Codec<IdVecKey>::decode — the owning form + twin oracle; IdVecKey.
    struct IdVecKeyView {
        const char* p = nullptr;  ///< The stored key bytes.
        int32_t byteLen = 0;      ///< The key byte length.

        /// @brief The id count (the leading `int32` field).
        /// @return The number of ids in the key.
        int32_t count() const {
            assert(p != nullptr
                && byteLen >= static_cast<int32_t>(sizeof(int32_t))
                && "IdVecKeyView::count: truncated key");
            int32_t c = 0;
            std::memcpy(&c, p, sizeof(int32_t));
            return c;
        }

        /// @brief Id @p k, unaligned-safe.
        /// @param k Index in `[0, count())`.
        /// @return The id at position @p k.
        int32_t idAt(int32_t k) const {
            assert(k >= 0 && k < count()
                && "IdVecKeyView::idAt: index out of range");
            int32_t v = 0;
            std::memcpy(&v,
                p + sizeof(int32_t) * (static_cast<std::size_t>(k) + 1),
                sizeof(int32_t));
            return v;
        }
    };

    /// @brief Wrap a stored `IdVecKey` byte span in an @ref IdVecKeyView.
    /// @param s The stored bytes (`inner().keyAt(id)` == `Codec<IdVecKey>::encode`).
    /// @return A zero-copy view aliasing @p s.
    /// @see IdVecKeyView, Codec<IdVecKey>.
    inline IdVecKeyView viewIdVecKey(StrSpan s) {
        return IdVecKeyView{ s.ptr, s.len };
    }

}
