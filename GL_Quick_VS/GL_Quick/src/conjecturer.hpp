
/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025-2026 Generative Logic UG(haftungsbeschraenkt)

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
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <regex>
#include <filesystem>
#include <atomic>
#include <thread>
#include <mutex>
#include <functional>
#include "compiler.hpp"

namespace conj {

// ============================================================================
// Int-path constants
// ============================================================================

/// @brief Hard upper bound on argument count in a conjecture's
///        def-set map.
///
/// @details
/// Sized for the largest expected anchor (Incubator with its 7
/// `(1)`-typed slots) plus chain-introduced bound variables and a
/// safety margin. The fixed-size `IntDefSetMap::argId[]` and sibling
/// arrays use this as their dimension; every writer asserts
/// `count <= MAX_CONJ_ARGS` before adding an entry.
static constexpr int MAX_CONJ_ARGS   = 32;

/// @brief Maximum int16_t words in a flat `IntConjBuf`.
///
/// @details
/// Each conjecture block is `[boundCount, bv0..bvN, nameId, arity,
/// arg0..argN]`; a typical conjecture occupies 30-80 words. Overflow
/// of this cap indicates either a misconfigured `max_size_expression_*`
/// setting or a conjecture that escapes the cascade-filter accounting
/// — treat as a hard error.
static constexpr int MAX_CONJ_BUF    = 256;

/// @brief Cap on connection maps materialised from the cartesian
///        product over `(p, q)` pairs in `makeAllConnectionMapsInt`.
///
/// @details
/// Exceeding this means the candidate generated more bijection
/// permutations than the buffer can hold. Misconfiguration in
/// `max_values_for_def_sets` or `max_size_mapping_def_set` is the
/// typical cause. The hot-path callers stop appending past
/// `MAX_CONN_MAPS` rather than reallocate, so the cap is observable
/// as a silent truncation if it ever fires; keep batch configs
/// sized so it does not.
static constexpr int MAX_CONN_MAPS   = 512;

/// @brief Maximum distinct def-set types in a single conjecture.
///
/// @details
/// One slot per type group (`(1)`, `P(x(1))`, etc.). Sized to cover
/// the union of all per-batch type spectra; the conjecturer never
/// emits more types than this in a single candidate.
static constexpr int MAX_DEFSET_GROUPS = 8;

/// @brief Maximum args belonging to one def-set group inside a
///        single conjecture.
///
/// @details
/// Together with `MAX_DEFSET_GROUPS` bounds the per-conjecture
/// type-position table used by `checkDefSets*`.
static constexpr int MAX_PER_GROUP   = 16;

// ============================================================================
// Int-path structs
// ============================================================================

/// @brief Bidirectional name <-> int16_t map for expression names
///        and def-set texts.
///
/// @details
/// Built once during `Conjecturer::buildNameMap` from the per-batch
/// config; immutable thereafter. Variables (already numeric in
/// conjecture strings) are NOT in this map — they are stored
/// directly as int16_t. ID 0 is reserved for invalid/empty so that
/// an unmatched `lookup` can be distinguished from a valid encoding
/// without a sentinel parameter.
///
/// @invariant `idToName[0] == ""` — the reserved "empty" slot is
///            installed by the default constructor and never
///            overwritten.
/// @invariant `nameToId[s] == id` iff `idToName[id] == s` for every
///            inserted `s`.
struct ConjNameMap {
    /// @brief Forward map: name -> id. Lookup is O(1) average.
    std::unordered_map<std::string, int16_t> nameToId;

    /// @brief Reverse map: id -> name. Indexed directly by id;
    ///        `idToName[0]` is the reserved empty string.
    std::vector<std::string> idToName;

    /// @brief Next id to mint. The constructor seeds it at 1
    ///        because id 0 is reserved.
    int16_t nextId = 1;

    /// @brief Construct with the reserved 0 slot installed.
    ConjNameMap() { idToName.push_back(""); }

    /// @brief Look up `s`, minting a new id on miss. Idempotent.
    /// @param s Name to encode.
    /// @return Stable id for `s` (>= 1).
    int16_t encode(const std::string& s) {
        auto it = nameToId.find(s);
        if (it != nameToId.end()) return it->second;
        int16_t id = nextId++;
        nameToId[s] = id;
        idToName.push_back(s);
        return id;
    }

    /// @brief Recover the string for a previously minted id.
    ///
    /// @details
    /// The returned reference points into `idToName` and is stable
    /// across the map's lifetime as long as no concurrent `encode`
    /// runs (the underlying vector may reallocate on `push_back`).
    /// All `Conjecturer` users observe `nameMap_` only after
    /// construction completes, so this is safe in production.
    ///
    /// @param id A previously minted id, or 0 for the empty slot.
    /// @return Reference into `idToName`.
    const std::string& decode(int16_t id) const {
        return idToName[static_cast<std::size_t>(id)];
    }

    /// @brief Read-only lookup; returns 0 (the reserved invalid id)
    ///        on miss instead of inserting.
    /// @param s Name to look up.
    /// @return Id of `s` if present, else 0.
    int16_t lookup(const std::string& s) const {
        auto it = nameToId.find(s);
        return (it != nameToId.end()) ? it->second : 0;
    }
};

/// @brief Flat int16_t buffer for a conjecture expression.
///
/// @details
/// Layout: repeating blocks of `[boundCount, bv0..bvN, nameId,
/// arity, arg0..argN]`. The last block is the head and always has
/// `boundCount == 0`. Each block represents one quantifier layer
/// with its bound variables and the predicate it scopes. The flat
/// encoding lets the int-path filter cascade walk a candidate
/// without repeatedly parsing the string form.
///
/// @invariant `len <= MAX_CONJ_BUF`. Every writer asserts before
///            adding a block; overflow indicates a misconfigured
///            `max_size_expression_*` cap.
struct IntConjBuf {
    /// @brief Block-formatted payload. Walk via the layout described
    ///        in the struct documentation.
    int16_t data[MAX_CONJ_BUF];

    /// @brief Words used in `data`. Always <= `MAX_CONJ_BUF`.
    int16_t len = 0;
};

/// @brief Argument -> def-set mapping as parallel arrays.
///
/// @details
/// Replaces the string-keyed `DefSetMap` in the hot path so the int
/// lane does not pay for `std::map` lookups inside per-candidate
/// filtering. Each parallel array is indexed by position `i` in
/// `[0, count)`; position `i` of every array describes the same
/// argument.
///
/// @invariant `count <= MAX_CONJ_ARGS`. All four arrays are kept in
///            sync — never write `argId[i]` without writing the
///            companion entries at the same index.
struct IntDefSetMap {
    /// @brief Active entries in the parallel arrays.
    int16_t count = 0;

    /// @brief variable ID (1, 2, 3, ...) of position `i`.
    int16_t argId[MAX_CONJ_ARGS];

    /// @brief `ConjNameMap` id of the def-set text at position `i`.
    int16_t defSetId[MAX_CONJ_ARGS];

    /// @brief 0 or 1 — the "combinable" flag inherited from the
    ///        expression's `definition_sets` config field.
    int16_t combinable[MAX_CONJ_ARGS];

    /// @brief 0 or 1 — the "connectable" flag inherited from the
    ///        expression's `definition_sets` config field.
    int16_t connectable[MAX_CONJ_ARGS];
};

/// @brief Connection map: argId -> target argId. Array-indexed, no
///        `std::map`.
///
/// @details
/// `map[srcArgId] == targetArgId`; an entry of 0 means unmapped.
/// The array dimension is `MAX_CONJ_ARGS * 2` so two distinct
/// candidates' arg-IDs can coexist during a `connectExpressionsInt`
/// merge without collision (the second candidate's IDs are shifted
/// past `MAX_CONJ_ARGS`).
///
/// @invariant `maxArg < MAX_CONJ_ARGS * 2`. Every writer enforces.
struct IntConnMap {
    /// @brief `map[argId] = targetArgId`; 0 = unmapped.
    int16_t map[MAX_CONJ_ARGS * 2];

    /// @brief Highest argId in use. Used to bound iteration.
    int16_t maxArg = 0;
};

// ============================================================================
// Type aliases
// ============================================================================

/// @brief Definition-set tuple: `(text, combinable_flag,
///        connectable_flag)`.
///
/// @details
/// `text` is the normalised def-set text (e.g. `"(1)"`,
/// `"P(x(1))"`); the two flags drive cascade-filter decisions about
/// whether the arg may be combined with others or connected to the
/// anchor.
using DefSetTuple = std::tuple<std::string, bool, bool>;

/// @brief Argument-name -> `DefSetTuple` map.
///
/// @details
/// The string-path twin of `IntDefSetMap`. A candidate carries one
/// of these from the moment the conjecturer recognises its arg
/// structure until the int-path encoding takes over. Keys are arg
/// names as strings (e.g. `"1"`, `"5"`).
using DefSetMap = std::map<std::string, DefSetTuple>;

/// @brief Pre-computed injective-mapping table.
///
/// @details
/// `MappingsMap[size][(left, right)]` is the list of every injective
/// map of arity `size` from `[1..left]` -> `[1..right]`. Built once
/// by `Conjecturer::createMap` (and the `Anchor` twin) and consumed
/// read-only in the hot path by `makeAllConnectionMapsInt`.
using MappingsMap = std::map<int, std::map<std::pair<int,int>, std::vector<std::map<int,int>>>>;

/// @brief Permutations: `chain_length` -> list of permutation
///        vectors.
///
/// @details
/// Used by the reshuffle pipeline to enumerate canonical renamings.
using PermutationsMap = std::map<int, std::vector<std::vector<int>>>;

/// @brief Binary sequences: `length` -> list of `{0, 1}` vectors.
///
/// @details
/// Drives the sign-bit enumeration for negation variants in the
/// conjecturer's combine loop.
using BinarySeqsMap = std::map<int, std::vector<std::vector<int>>>;

// ============================================================================
// Configuration structs (mirrors Python's configuration_reader.py)
// ============================================================================

/// @brief Per-expression configuration record loaded from
///        `Config<Tag>.json`.
///
/// @details
/// One instance per registered expression name (`=`, `in`, `in2`,
/// `in3`, `fold`, ...). Carries the structural shape (arity,
/// definition-set types per position, MPL text), the combinatorial
/// budget (count caps, size caps), and the per-expression policy
/// flags (negation allowed, may-constitute-existence-head,
/// existence-variable position, allowed-for-existence positions).
///
/// @see `Conjecturer::loadConfiguration` (`conjecturer.cpp`) for
///      the JSON parser.
struct ExpressionDescription {
    /// @brief Number of arguments the expression takes.
    int arity = 0;

    /// @brief Per-position definition-set mapping. Key is the
    ///        position string (`"1"`, `"2"`, ...); value is
    ///        `(text, combinable, connectable)`.
    DefSetMap definition_sets;

    /// @brief Full MPL form of the expression definition.
    std::string full_mpl;

    /// @brief Short handle used in the conjecture-emission text.
    std::string handle;

    /// @brief Short MPL form as authored in the config.
    std::string short_mpl_raw;

    /// @brief Short MPL form after normalisation (whitespace and
    ///        arg-renaming canonicalisation).
    std::string short_mpl_normalized;

    /// @brief Cap on copies of this expression in one conjecture.
    int max_count_per_conjecture = 0;

    /// @brief Pre-existence size cap. Applies at current
    ///        `checkComplexityPerOp` call sites.
    int max_size_expression_before_existence = 0;

    /// @brief Post-existence size cap. Applies after
    ///        `reformulateToExistenceHead`, pre-emission.
    int max_size_expression_after_existence = 0;

    /// @brief Minimum size for an expression carrying this head.
    int min_size_expression = 1;

    /// @brief Names of input arguments.
    std::vector<std::string> input_args;

    /// @brief Names of output arguments.
    std::vector<std::string> output_args;

    /// @brief 0-based positions of input args within
    ///        `definition_sets`.
    std::vector<int> indices_input_args;

    /// @brief 0-based positions of output args within
    ///        `definition_sets`.
    std::vector<int> indices_output_args;

    /// @brief May this expression appear negated as `!(...)`?
    bool allow_negation = false;

    /// @brief May this expression head an existence?
    bool allow_to_constitute_existence = false;

    /// @brief Which 1-based position carries the bound variable in
    ///        an existence node. `-1` means unset.
    int existence_variable_position = -1;

    /// @brief 1-based positions eligible to be existentially
    ///        wrapped when this expression is an ungrounded
    ///        operator head.
    std::vector<int> allowed_for_existence;
};

/// @brief Top-level conjecturer parameters loaded from
///        `Config<Tag>.json`'s `parameters` block.
///
/// @details
/// One instance per `Conjecturer` (immutable after construction).
/// Drives every cascade-filter threshold: how many simple
/// expressions per conjecture, how big a single arg-mapping table
/// may be, per-def-set complexity caps pre- and post-existence, and
/// a small handful of policy flags.
///
/// @warning `apply_in_premise_filter` is **dead code** — declared
///          and loaded but never consulted by `passesInPremiseFilter`
///          or its callsites. SwDD `OPEN-9` records the gap. Until
///          either the flag is reconnected or removed, treat its
///          value as informational only.
///
/// @see `Conjecturer::loadConfiguration` (`conjecturer.cpp`) for
///      the JSON loader.
/// @see `Conjecturer::passesInPremiseFilter` (`conjecturer.cpp`)
///      for the `in[...]` cnt-shape rules and the
///      [D-23](../../docs/40_decisions.md#d-23)
///      anchor-membership-axiom rejection.
struct ConfigurationParameters {
    /// @brief Lower bound for `nse` iteration. Setting `1` enables
    ///        the preliminary single-expression-anchor pass
    ///        (`singleExprAnchorConnection`).
    int min_number_simple_expressions = 2;

    /// @brief Upper bound for `nse` iteration. Below 2, the combine
    ///        loop is skipped entirely.
    int max_number_simple_expressions = 0;

    /// @brief Argument to `Conjecturer::createMap(N)` — controls
    ///        bijection-table size.
    int max_size_mapping_def_set = 0;

    /// @brief Cap on argument count for any candidate; consulted by
    ///        `countArgumentsFilter`.
    int max_number_args_expr = 0;

    /// @brief Filter-cascade gate: reject when operator-block count
    ///        exceeds this.
    int operator_threshold = 0;

    /// @brief Per-def-set cap on the number of combinable args
    ///        (consumed by `checkDefSets`).
    std::map<std::string, int> max_values_for_def_sets;

    /// @brief Per-def-set cap on uncombinable args (consumed by
    ///        `checkDefSets`).
    std::map<std::string, int> max_values_for_uncomb_def_sets;

    /// @brief Per-def-set prior-connection cap (combined-candidate
    ///        gate consumed by `checkDefSetsPriorToConnection`).
    std::map<std::string, int> max_values_for_def_sets_prior_connection;

    /// @brief Per-type complexity cap pre-existence reformulation
    ///        (consumed by `checkComplexityLevelForDefSets`).
    std::map<std::string, int> max_complexity_if_anchor_parameter_connected_before_existence;

    /// @brief Per-type after-existence cap as a 2-tuple
    ///        `[complexity, arity_sum]`.
    ///
    /// @details
    /// `first` = complexity-level cap (count of `(>[`); legacy
    /// semantics.
    /// `second` = max sum of non-anchor leaf arities; new dimension
    /// that lets simple/small theorems escape type-pinning
    /// rejection. Reject only when BOTH caps are exceeded AND a slot
    /// of this type appears in non-anchor leaves. Set
    /// `second = 100` (or any value larger than realistic
    /// arity-sums) to disable the new dimension for that type.
    /// JSON shape: `[a, b]`; legacy int values auto-promote to
    /// `[int, 100]`. See SwDD chapter `02_conjecturer.md` section
    /// *passesComplexityAfterExistence* and decision
    /// [D-23](../../docs/40_decisions.md#d-23).
    std::map<std::string, std::pair<int, int>> max_complexity_if_anchor_parameter_connected_after_existence;

    /// @brief Length cap for binary-sequence enumeration in
    ///        negation variants.
    int max_size_binary_list = 0;

    /// @brief Reserved for downstream consumers; unused inside
    ///        `conjecturer.cpp`.
    std::vector<int> simple_facts_parameters;

    /// @brief Reserved for downstream consumers; unused inside
    ///        `conjecturer.cpp` (SwDD `OPEN-CFG-2`).
    std::vector<std::string> fact_variable_kinds;

    /// @brief When true, branch-specific incubator paths in
    ///        `Conjecturer::run` are taken.
    bool incubator_mode = false;

    /// @brief Gate for `passesInPremiseFilter` — the Peano-motivated
    ///        `(in[x, X])` shape restriction.
    ///
    /// @warning Currently **dead code**. The flag is loaded but
    ///          never consulted; `passesInPremiseFilter` always
    ///          runs when `hasIn` is true regardless of this flag.
    ///          See SwDD `OPEN-9` for history. Configs (e.g.
    ///          `ConfigGauss.json`) that set this to `false` rely
    ///          on Gauss conjectures carrying `in[...]` only at the
    ///          head — where `hasIn` short-circuits the function —
    ///          for the restriction to appear inactive.
    ///          Reconnecting the gate is a one-line fix at the top
    ///          of `passesInPremiseFilter`.
    bool apply_in_premise_filter = true;

    /// @brief Per-def-set cap on the number of DISTINCT anchor-slot
    ///        values of that type allowed to appear as arguments in
    ///        non-anchor leaves of a conjecture.
    ///
    /// @details
    /// Example Peano: `"(1)" : 1` means at most 1 distinct
    /// `(1)`-typed anchor slot value in leaves (i0 = 2 OR i1 = 6,
    /// not both) — rejects shapes like `in3[i0, i1, c, +]` while
    /// keeping the cancellation family `in3[a, b, i0, +]`. Absent
    /// keys -> no cap for that type (filter off for that type).
    /// Empty map -> filter fully off. Gauss leaves this empty to
    /// keep double-digit identities alive. Consumed by
    /// `passesMaxDistinctAnchorValuesPerType`.
    std::map<std::string, int> max_distinct_anchor_values_per_type;
};

/// @brief Aggregate configuration record for one conjecturer batch.
///
/// @details
/// Combines the per-expression `ExpressionDescription` map with the
/// top-level `ConfigurationParameters`, plus the regex-based
/// exclusion patterns, the prohibited-combination / prohibited-head
/// block lists, the anchor identity, and the I/O folder paths.
/// Constructed once by `Conjecturer::loadConfiguration`; immutable
/// thereafter.
struct ConfigurationData {
    /// @brief Expression name -> description.
    std::map<std::string, ExpressionDescription> data;

    /// @brief JSON key insertion order; preserved for deterministic
    ///        enumeration in `Conjecturer::run`.
    std::vector<std::string> expressionOrder;

    /// @brief Top-level parameter block.
    ConfigurationParameters parameters;

    /// @brief Raw regex strings for `patternInConjecture` as
    ///        authored in the config.
    std::vector<std::string> patterns_to_exclude_raw;

    /// @brief Compiled regexes consulted by `patternInConjecture`.
    std::vector<std::regex> patterns_to_exclude;

    /// @brief Raw regex strings for the `onlyInHeadGood` filter.
    std::vector<std::string> only_in_head_raw;

    /// @brief Compiled regexes consulted by `onlyInHeadGood`.
    std::vector<std::regex> only_in_head_patterns;

    /// @brief Block-listed expression-combinations consulted by
    ///        `checkProhibitedCombinations`.
    std::vector<std::set<std::string>> prohibited_combinations;

    /// @brief Block-listed expression heads consulted by
    ///        `prohibitedHeadsGood`.
    std::vector<std::string> prohibited_heads;

    /// @brief Override path for the theorem-output folder. Empty
    ///        means "derive from `Conjecturer::projectRoot_`".
    std::string theorems_folder;

    /// @brief Override path for OR-pair / external-theorem inputs.
    std::string background_theorems_folder;

    /// @brief Full anchor name (e.g. `"AnchorPeano"`).
    std::string anchor_name;

    /// @brief Short anchor id (e.g. `"Peano"`); the constructor's
    ///        input.
    std::string anchor_id;

    /// @brief Convenience accessor — matches Python's
    ///        `get_anchor_name`.
    /// @return Copy of `anchor_name`.
    std::string getAnchorName() const;
};

// ============================================================================
// Worker result
// ============================================================================

/// @brief Result bundle returned by every worker driver.
///
/// @details
/// Returned from `singleThreadCalculation*` and
/// `singleExprAnchorConnection*`. `connected_list` carries
/// `(expression, def_set_map)` pairs for intermediate (non-anchor)
/// connections that may seed the next `nse` round.
/// `connected_list2` carries final-stage conjectures already
/// attached to the anchor. `reshuffled_list` is the canonicalised
/// companion for `connected_list2`, and `reshuffled_mirrored_list`
/// carries the mirror variants emitted via
/// `createReshuffledMirrored`.
///
/// @invariant `reshuffled_list.size() == connected_list2.size()`
///            after a complete worker invocation; the mirror list
///            may differ in length when a mirror collapses to its
///            own canonical form (rejected by I-9's distinctness
///            guard).
struct WorkerResult {
    /// @brief Intermediate non-anchor connections.
    std::vector<std::pair<std::string, DefSetMap>> connected_list;

    /// @brief Final conjectures (anchor-attached).
    std::vector<std::string> connected_list2;

    /// @brief Canonical-form companion for `connected_list2`.
    std::vector<std::string> reshuffled_list;

    /// @brief Mirror variants emitted via `createReshuffledMirrored`.
    std::vector<std::string> reshuffled_mirrored_list;
};

// ============================================================================
// RAII guard for TreeNode1
// ============================================================================

/// @brief RAII guard for `ce::TreeNode1*`.
///
/// @details
/// Owns a `TreeNode1*` and calls `ce::deleteTree` on destruction.
/// Move-only (copy explicitly deleted) so a tree is never freed
/// twice. Use `release()` to hand ownership to a caller that
/// intends to free the tree itself. Used inside conjecturer
/// functions that build a short-lived tree from `ce::parseExpr` and
/// need exception-safe cleanup.
struct TreeGuard {
    /// @brief Owned tree pointer; may be null after `release`.
    ce::TreeNode1* root;

    /// @brief Adopt ownership of the tree.
    explicit TreeGuard(ce::TreeNode1* r) : root(r) {}

    /// @brief Delete the owned tree on scope exit.
    ~TreeGuard() { if (root) ce::deleteTree(root); }

    TreeGuard(const TreeGuard&) = delete;
    TreeGuard& operator=(const TreeGuard&) = delete;

    /// @brief Borrow the underlying pointer without releasing
    ///        ownership.
    ce::TreeNode1* get() const { return root; }

    /// @brief Hand ownership of the tree to the caller; the guard
    ///        becomes inert.
    ce::TreeNode1* release() { auto* r = root; root = nullptr; return r; }
};

// ============================================================================
// Free utility functions (no config dependency)
// ============================================================================

/// @brief Check whether any inner-paren leaf token (`(...)` with no
///        nested parens) appears more than once in `s`.
///
/// @details
/// Scans `s` with the regex `\([^()]*\)` and counts distinct
/// inner-paren tokens. Returns `true` when the same leaf token
/// appears at least twice — for example `"(in[1,2])(in[1,2])"`.
/// Used by the structural-uniqueness gate that rejects conjectures
/// whose chain carries duplicate predicates. NOT about repeated
/// arg ids inside a single leaf — `(in[1,1])` is a single token
/// and passes this check.
///
/// @param s Conjecture or sub-expression text.
/// @return `true` if any inner-paren leaf token occurs at least
///         twice; `false` otherwise.
bool repetitionsExist(const std::string& s);

/// @brief Parse a definition-set text into a tree plus the list of
///        argument ids it references.
///
/// @details
/// Wraps `ce::parseExpr` with a definition-set-aware post-process
/// that extracts the integer ids referenced inside the tree. The
/// returned tree is heap-allocated; the caller owns it (use
/// `TreeGuard` for exception-safe cleanup).
///
/// @param s Definition-set text (e.g. `"(1)"`, `"P(x(1))"`).
/// @return Pair `(root, argIds)` where `root` is the parsed tree
///         and `argIds` is the integer ids referenced.
std::pair<ce::TreeNode1*, std::vector<int>> parseDefSet(const std::string& s);

/// @brief Render a `TreeNode1` to its string form, reordering
///        commutative arg lists into canonical (ascending) order.
///
/// @details
/// Used by the def-set canonicalisation pipeline so two def-set
/// texts that differ only in arg-list ordering compare equal after
/// rendering. Symmetric heads (`=`, `&`, `|`) get their args
/// sorted; non-symmetric heads pass through.
///
/// @param root Tree to render. Must be non-null.
/// @return Canonical string form.
std::string treeToStrReorder(const ce::TreeNode1* root);

/// @brief Render a `TreeNode1` to its string form, applying
///        `subMap` as an integer-id substitution and shifting every
///        retained id by `offset`.
///
/// @details
/// Used by the def-set unifier (`defSetsEqual`) and by the
/// connection helpers that need to renumber a def-set's args before
/// merging it with another. Substitution is total: every id in the
/// tree is either remapped via `subMap` or shifted by `offset`
/// (those not in `subMap`).
///
/// @param root   Tree to render. Must be non-null.
/// @param offset Integer shift applied to ids that survive
///               `subMap`.
/// @param subMap Substitution map (`oldId -> newId`).
/// @return String form with substitutions and shift applied.
std::string treeToStr(const ce::TreeNode1* root, int offset, const std::map<int,int>& subMap);

/// @brief Find the minimum and maximum integer arg ids appearing in
///        `s`.
///
/// @details
/// Scans every `[...]`-bounded integer token. When `s` contains no
/// integer ids the function returns `(INT_MAX, INT_MIN)` — callers
/// expecting a non-empty range should check before consuming.
///
/// @param s Expression text.
/// @return Pair `(min, max)`.
std::pair<int,int> findMinMaxNumbers(const std::string& s);

/// @brief Collect every distinct integer arg id in `s`.
///
/// @details
/// Scans every `[...]`-bounded integer token and deduplicates via
/// `std::set` insertion. Order in the returned set is ascending
/// (the `std::set` ordering invariant).
///
/// @param s Expression text.
/// @return Sorted set of ids.
std::set<int> findAllIds(const std::string& s);

/// @brief Merge `(num1, num2)` into a transitive replacement map.
///
/// @details
/// Equates `num1` and `num2` in `repMap` while preserving any
/// existing equivalences. After the call,
/// `repMap[num1] == repMap[num2]` and any chain of prior
/// equivalences reaching either endpoint is collapsed to the same
/// representative. Used by `defSetsEqual` to track the unifier
/// between two definition-set texts.
///
/// @param repMap Mutable replacement map (in/out).
/// @param num1   First endpoint to equate.
/// @param num2   Second endpoint to equate.
void updateReplacementMap(std::map<int,int>& repMap, int num1, int num2);

/// @brief Test whether two definition-set texts are equal up to a
///        consistent renaming of integer arg ids.
///
/// @details
/// Pairs the two trees position-by-position; whenever both sides
/// carry an integer id at the same node, the pair feeds
/// `updateReplacementMap`. The trees are equal up to renaming iff
/// the resulting map is consistent (no two distinct ids in `ds1`
/// equate to the same id in `ds2`). When the result is `true`,
/// the returned map IS the unifier; when `false`, the map is
/// undefined.
///
/// @param ds1 First definition-set text.
/// @param ds2 Second definition-set text.
/// @return Pair `(equal, unifier)`.
std::pair<bool, std::map<int,int>> defSetsEqual(const std::string& ds1, const std::string& ds2);

/// @brief Subtract `subtractValue` from every integer arg id in
///        `s`, applying `m` as an additional substitution map.
///
/// @details
/// First applies `m` to each integer id, then subtracts
/// `subtractValue`. Useful for shifting a def-set's arg-id space
/// down to start from 1 after a connection step has consumed some
/// ids.
///
/// @param s             Expression text.
/// @param subtractValue Constant to subtract from each surviving
///                      id.
/// @param m             Substitution applied before subtraction.
/// @return Pair `(transformed, newIds)` where `newIds` lists the
///         resulting ids in occurrence order.
std::pair<std::string, std::vector<int>> subtractAndReplaceNumbers(const std::string& s, int subtractValue, const std::map<int,int>& m);

/// @brief Canonicalise the integer arg ids in `s` so they form the
///        contiguous sequence `1, 2, 3, ...` in first-occurrence
///        order.
///
/// @details
/// Walks `s` left-to-right, assigning the next free id (starting at
/// 1) the first time each old id is encountered. Subsequent
/// occurrences of the same old id are mapped consistently.
///
/// @param s Expression text with possibly-sparse integer ids.
/// @return Pair `(transformed, oldIdsInOrder)`.
std::pair<std::string, std::vector<int>> reorderNumbers(const std::string& s);

/// @brief Adjust every def-set text in `argDefSetMap` so they share
///        a contiguous integer-id space.
///
/// @details
/// Maps the union of ids across every value in the map to `1..N`
/// in first-occurrence order, then rewrites every value in-place
/// using the substitution. After the call, no two values use
/// disjoint or overlapping id ranges; the global view is
/// contiguous.
///
/// @param argDefSetMap Mutable map (in/out) of arg name -> def-set
///                     text.
void shiftTogether(std::map<std::string,std::string>& argDefSetMap);

/// @brief Connect two expression sets and return the merge result.
///
/// @details
/// Computes the intersection / extension between two `(name, kind)`
/// expression sets given a connection type, a definition flag, an
/// argument-removal set, and a grooming flag. Used by the
/// string-path expression-merge engine.
///
/// @param set1            First expression set.
/// @param set2            Second expression set.
/// @param connectionType  Type label that drives the merge policy.
/// @param isDefinition    Non-zero when a definition (vs ordinary
///                        statement) is being connected.
/// @param argsToRemove    Args to drop from the merged result.
/// @param afterGrooming   Whether the inputs have already been
///                        through the grooming pass.
/// @return Tuple `(common_map, common_set, success,
///         removed_args)` where `success` is `1` on a clean merge
///         and `0` on rejection.
std::tuple<std::map<std::string,std::string>, std::set<std::pair<std::string,std::string>>, int, std::vector<std::string>>
    connectExpressionSets(
        const std::set<std::pair<std::string,std::string>>& set1,
        const std::set<std::pair<std::string,std::string>>& set2,
        const std::string& connectionType,
        int isDefinition,
        const std::set<std::string>& argsToRemove,
        bool afterGrooming);

/// @brief Extract the substring between the first `[` at or after
///        `startIndex` and its matching `]`.
///
/// @details
/// Bracket matching is balanced — handles nested `[...]` correctly.
/// Returns the bracketed content (without the outer brackets); the
/// empty string is returned if no `[` is found at or after
/// `startIndex`.
///
/// @param s          Source text.
/// @param startIndex Position to begin searching from.
/// @return Bracketed content, or empty string on miss.
std::string extractBetweenBrackets(const std::string& s, size_t startIndex = 0);

/// @brief Find the position of `substring` inside `text`, accepting
///        only matches that are themselves enclosed by `[...]`
///        brackets.
///
/// @details
/// Used by argument-position lookups where matching outside a
/// bracket pair (e.g. inside a head name) would yield a false
/// positive.
///
/// @param text      Source text.
/// @param substring Substring to find.
/// @return `-1` on miss, otherwise the 0-based index of the
///         match.
int findPositionSurrounded(const std::string& text, const std::string& substring);

/// @brief Reorder `lst` so its elements appear in the same order as
///        their first occurrence inside `text`.
///
/// @details
/// Stable-sort by first-occurrence index. Elements not appearing
/// in `text` are placed at the end in their original relative
/// order.
///
/// @param lst  Elements to reorder.
/// @param text Source text whose ordering controls the sort.
/// @return Reordered copy of `lst`.
std::vector<std::string> sortListAccordingToOccurrence(const std::vector<std::string>& lst, const std::string& text);

/// @brief Extract integer-ish strings from `intStrings` ordered by
///        their first occurrence in `bigString`.
///
/// @details
/// Returns those entries of `intStrings` that appear at least once
/// in `bigString`, sorted by their first-occurrence position.
/// Entries absent from `bigString` are dropped.
///
/// @param intStrings Candidate integer strings.
/// @param bigString  Source text whose ordering controls the sort.
/// @return Filtered + reordered list.
std::vector<std::string> findOrderedIntegers(const std::vector<std::string>& intStrings, const std::string& bigString);

/// @brief Replace one occurrence of an integer-ish substring inside
///        a bigger string.
///
/// @details
/// Replacement is single-shot (first match only) so callers can
/// perform a controlled rename without disturbing identically-named
/// later occurrences.
///
/// @param bigString      Source text.
/// @param targetInt      Integer (as string) to find.
/// @param replacementInt Replacement integer (as string).
/// @return New string with one occurrence replaced; original
///         unchanged.
std::string replaceIntegerInString(const std::string& bigString, const std::string& targetInt, const std::string& replacementInt);

/// @brief Subtract `number` from every integer arg id in `expr`
///        that satisfies the `numbersToReplace` filter.
///
/// @details
/// When `replaceAll` is `true`, every integer id is shifted by
/// `-number`. When `replaceAll` is `false`, only ids appearing in
/// `numbersToReplace` are shifted. Used by the post-connect
/// renumber step in `connectExpressionsInt`.
///
/// @param expr               Source expression text.
/// @param number             Constant to subtract.
/// @param numbersToReplace   Selective filter (consulted when
///                           `replaceAll == false`).
/// @param replaceAll         Override the filter and shift every
///                           id.
/// @return Shifted expression text.
std::string subtractNumberFromInts(const std::string& expr, int number, const std::set<int>& numbersToReplace, bool replaceAll);

/// @brief Count the distinct non-identity targets in `mapping`.
///
/// @details
/// Filters out identity entries (`key == value`), then returns the
/// size of the residual values-set. So a mapping that is entirely
/// identity returns 0; a mapping `{"a"->"x", "b"->"y"}` returns 2;
/// a mapping `{"a"->"x", "b"->"x"}` returns 1 (the two non-identity
/// entries share one target).
///
/// @param mapping Substitution map.
/// @return Count of distinct non-identity values.
int getNumberRemovableArgs(const std::map<std::string,std::string>& mapping);

/// @brief Reject mappings that are not "self-rooted" — every target
///        must be reachable from a key equal to itself.
///
/// @details
/// Builds the reverse map `value -> min(key)` and accepts the
/// mapping only when every reverse entry has `min(key) == value`.
/// In effect: every entry `(k, v)` must satisfy `k == v`. Used as
/// a pre-acceptance gate before applying the mapping in
/// `connectExpressions`; substitutions that re-name args (i.e.
/// any non-identity entry) are rejected.
///
/// @param mapping Substitution map to validate.
/// @return `true` when every entry is identity (`k == v`); `false`
///         otherwise.
bool mappingGood(const std::map<std::string,std::string>& mapping);

// ============================================================================
// Main class
// ============================================================================

namespace testing { class Friend; }  // forward declaration for the unit-test access shim.

/// @brief Conjecturer — combinatorial enumeration of candidate
///        theorems for one anchor batch.
///
/// @details
/// Constructed once per batch; load the config, run, write output.
/// Equivalent to Python's legacy `create_expressions_parallel(config)`.
/// Public API is deliberately minimal: construction loads the
/// config and seeds every cache; `run()` does everything else.
///
/// Two execution lanes coexist (int-path / string-path); see the
/// SwDD chapter for the full architecture. The int-path was added
/// during the 100x acceleration campaign
/// ([D-20](../../docs/40_decisions.md#d-20)) and is the hot lane.
/// The string-path is retained for final-stage structural checks
/// (pattern matching, mirror generation, reshuffle) where the
/// string form is unavoidable.
///
/// **Thread safety.** The class itself is single-threaded; the
/// hot-path workers (`singleThreadCalculation*`) are driven by a
/// thread pool inside `run()` and read only the immutable post-
/// construction state.
///
/// @see [`docs/10_pipeline/02_conjecturer.md`](../../docs/10_pipeline/02_conjecturer.md)
///      — full architecture, filter-cascade details, reshuffle
///      pipeline, weaknesses.
/// @see [I-8](../../docs/30_invariants.md#i-8),
///      [I-9](../../docs/30_invariants.md#i-9),
///      [I-10](../../docs/30_invariants.md#i-10),
///      [I-11](../../docs/30_invariants.md#i-11) — invariants the
///      conjecturer enforces or relies on.
class Conjecturer {
public:
    /// @brief Construct the conjecturer for one anchor batch.
    ///
    /// @details
    /// Steps performed:
    /// 1. Locate `files/config/Config<anchorId>.json` via
    ///    search-path walk (the same default search list as the
    ///    prover).
    /// 2. Load the JSON via `loadConfiguration`, populating
    ///    `config_`.
    /// 3. Build the `ce::CoreExpressionConfig` adapter
    ///    (`buildCoreExprMapAdapter`) so any shim function
    ///    expecting prover-side compiled expression metadata can
    ///    borrow it.
    /// 4. Build the int-path lookup tables (`buildNameMap`,
    ///    `buildIntExprConfigs`).
    /// 5. Pre-compute the bijection / permutation / binary-sequence
    ///    tables (`createMap`, `createMapAnchor`,
    ///    permutation/binary enumeration) into the immutable
    ///    caches.
    ///
    /// Construction does NOT enumerate conjectures — that is
    /// `run()`'s job. Construction also does not write any output
    /// file.
    ///
    /// @param anchorId Short anchor name (e.g. `"Peano"`,
    ///                 `"Gauss"`, `"IncubatorPeano"`). The
    ///                 `"Anchor"` prefix is prepended internally
    ///                 where the full name is needed.
    /// @pre  `files/config/Config<anchorId>.json` exists and is
    ///       well-formed JSON.
    /// @post `config_`, `anchor_`, `mappingsMap_`,
    ///       `mappingsMapAnchor_`, `binarySeqsMap_`,
    ///       `allPermutations_`, the operator / relation / property
    ///       classifications, the int-path lookup tables, and
    ///       `projectRoot_` are all populated.
    explicit Conjecturer(const std::string& anchorId);

    /// @brief Generate conjectures for the loaded batch and write
    ///        `theorems.txt` plus the canonical-form / mirror-form
    ///        companion files.
    ///
    /// @details
    /// Walks the `nse` axis from
    /// `parameters.min_number_simple_expressions` to
    /// `parameters.max_number_simple_expressions`; for each `nse`
    /// enumerates candidate combinations, runs them through the
    /// filter cascade, reshuffles survivors into canonical form,
    /// generates mirror variants, and emits OR conjectures via
    /// `generateOrConjectures()` (config-derived from
    /// per-expression `allow_to_constitute_existence` flags).
    /// Final write to:
    ///
    /// - `theorems.txt` — raw survivors.
    /// - `reshuffled_theorems.txt` — canonical-form survivors.
    /// - `reshuffled_mirrored_theorems.txt` — mirror variants.
    /// - `or_pairs.txt` — OUTPUT artefact recording the
    ///   `(existence, companion)` pairs emitted this run; opened
    ///   with `std::ios::out` and overwritten each invocation. Not
    ///   an input. Existing content is replaced.
    ///
    /// Equivalent to Python's `create_expressions_parallel(config)`.
    void run();

private:
    /// @brief Test-only access shim. Defined inside
    ///        `test_conjecturer.cpp`; production code never
    ///        references it. The friend declaration grants the
    ///        unit-test suite access to private methods (filters,
    ///        int-path encode/decode, `createMap*` statics, etc.)
    ///        without widening the public surface.
    friend class ::conj::testing::Friend;

    // ---- Immutable after construction (thread-safe to share) ----

    /// @brief Loaded per-batch config record.
    ConfigurationData config_;

    /// @brief Description record for the anchor expression itself
    ///        (looked up from `config_.data` by name).
    ExpressionDescription anchor_;

    /// @brief Pre-computed bijections for non-anchor combine.
    MappingsMap mappingsMap_;

    /// @brief Pre-computed bijections for anchor-attach connections.
    MappingsMap mappingsMapAnchor_;

    /// @brief Pre-computed binary sign sequences for negation
    ///        enumeration in `connectExpressionsInt`.
    BinarySeqsMap binarySeqsMap_;

    /// @brief Pre-computed permutations consumed by reshuffle.
    PermutationsMap allPermutations_;

    /// @brief Names of expressions classified as operators (both
    ///        `input_args` AND `output_args` non-empty).
    std::vector<std::string> operators_;

    /// @brief Names of expressions classified as relations
    ///        (2 `input_args` AND no `output_args`).
    std::vector<std::string> relations_;

    /// @brief Names of expressions classified as properties
    ///        (1 `input_arg` AND no `output_args`).
    std::vector<std::string> properties_;

    /// @brief Adapter for shim functions that need
    ///        `ce::CoreExpressionConfig`. One entry per expression
    ///        name; populated by `buildCoreExprMapAdapter`.
    std::map<std::string, ce::CoreExpressionConfig> coreExprMap_;

    /// @brief Resolved project root (for theorem-file output).
    ///        Computed in the constructor from `argv[0]`.
    std::filesystem::path projectRoot_;

    // ---- Int-path data (immutable after construction) ----

    /// @brief Bidirectional name <-> id map for every expression
    ///        name and def-set text appearing in `config_.data`.
    ///        Populated by `buildNameMap`.
    ConjNameMap nameMap_;

    /// @brief Encoded anchor expression buffer.
    IntConjBuf anchorInt_;

    /// @brief Encoded anchor def-set map.
    IntDefSetMap anchorDefSetsInt_;

    /// @brief nameMap ids of the entries in `operators_`.
    std::vector<int16_t> operatorNameIds_;

    /// @brief nameMap ids of the entries in `relations_`.
    std::vector<int16_t> relationNameIds_;

    /// @brief Compact per-expression config record indexed densely
    ///        by nameId.
    ///
    /// @details
    /// One entry per expression name. Stores arity, count cap,
    /// handle id, size caps before / after existence, min size,
    /// and the input / output argument index lists as fixed-size
    /// int16_t arrays so a hot-path filter can fetch the record
    /// with a single indexed read.
    struct IntExprConfig {
        /// @brief Owning nameMap id.
        int16_t nameId = 0;

        /// @brief Argument count.
        int16_t arity = 0;

        /// @brief Cap on copies per conjecture.
        int16_t maxCountPerConj = 0;

        /// @brief nameMap id of the handle string.
        int16_t handleId = 0;

        /// @brief Size cap pre-existence reformulation.
        int16_t maxSizeExprBeforeEx = 0;

        /// @brief Size cap post-existence reformulation.
        int16_t maxSizeExprAfterEx = 0;

        /// @brief Minimum size cap.
        int16_t minSizeExpr = 1;

        /// @brief 0-based input-arg positions.
        int16_t indicesInputArgs[16];

        /// @brief Active entries in `indicesInputArgs`.
        int16_t numInputArgs = 0;

        /// @brief 0-based output-arg positions.
        int16_t indicesOutputArgs[4];

        /// @brief Active entries in `indicesOutputArgs`.
        int16_t numOutputArgs = 0;
    };

    /// @brief Per-expression config table indexed by nameId.
    ///        Populated by `buildIntExprConfigs`.
    std::vector<IntExprConfig> intExprConfigs_;

    /// @brief Per-defSetId combinable cap from
    ///        `parameters.max_values_for_def_sets`.
    std::vector<int16_t> maxForDefSets_;

    /// @brief Per-defSetId uncombinable cap from
    ///        `parameters.max_values_for_uncomb_def_sets`.
    std::vector<int16_t> maxForUncombDefSets_;

    /// @brief Per-defSetId prior-connection cap.
    std::vector<int16_t> maxForDefSetsPrior_;

    /// @brief Per-defSetId pre-existence complexity cap.
    std::vector<int16_t> maxComplexityAnchorConn_;

    // ---- Configuration loading ----

    /// @brief Locate and parse `Config<anchorId>.json`.
    ///
    /// @details
    /// Builds the full `ConfigurationData` record including the
    /// regex compilation for `patterns_to_exclude` and
    /// `only_in_head_patterns`. Uses `nlohmann::ordered_json` for
    /// the first parse so the JSON key order in `expressionOrder`
    /// survives — downstream enumeration walks expressions in
    /// author order for deterministic conjecture sequences.
    ///
    /// @param anchorId Short anchor name.
    /// @return Populated `ConfigurationData`.
    /// @pre  Config file present on the search path.
    ConfigurationData loadConfiguration(const std::string& anchorId);

    /// @brief Build the `coreExprMap_` adapter shim from
    ///        `config_.data`.
    ///
    /// @details
    /// One-to-one projection: every `ExpressionDescription` produces
    /// a `ce::CoreExpressionConfig` that downstream shim functions
    /// (originally written against the prover's compiled-config
    /// record shape) consume without further translation.
    void buildCoreExprMapAdapter();

    // ---- Pre-computation ----

    /// @brief Pre-compute every injective mapping of arity `2..N`
    ///        from `[1..p]` -> `[1..q]` for every `(p, q)` cross
    ///        product within the size budget.
    ///
    /// @details
    /// Built once during construction; consumed read-only by
    /// `makeAllConnectionMaps` / `makeAllConnectionMapsInt` in the
    /// hot path. Memory footprint scales combinatorially in `N`;
    /// the current production batches use `N <= 5` so the table
    /// fits comfortably in a few MB.
    ///
    /// @param N Maximum arity to enumerate.
    /// @return `MappingsMap[size][(p, q)]` -> list of injections.
    static MappingsMap createMap(int N);

    /// @brief Pre-compute anchor <-> expression argument
    ///        permutation tables.
    ///
    /// @details
    /// Cartesian product over `T \subseteq S` with `targets^|T|`
    /// images per subset. For `AnchorIncubator` with its 7
    /// `(1)`-typed slots `leftMax = 7`; `rightMax` is the max over
    /// def-sets of `(uncomb + comb)` values from the per-batch
    /// config.
    ///
    /// @warning `rightMax > 3` causes RAM explosion: millions of
    ///          permutation dicts materialise. There is no assert
    ///          currently — see SwDD `OPEN-8`. Keep
    ///          `parameters.max_values_for_def_sets` plus
    ///          `parameters.max_values_for_uncomb_def_sets` so that
    ///          their per-type maximum sum stays at most 3. Adding
    ///          the assert is recommended; doing so is a Rule-8
    ///          architectural touch and is currently deferred.
    ///
    /// @param leftMax  Anchor-side slot count.
    /// @param rightMax Candidate-side slot count.
    static MappingsMap createMapAnchor(int leftMax, int rightMax);

    /// @brief Compute the maximum anchor-slot id over the loaded
    ///        config; used as `leftMax` for `createMapAnchor`.
    int determineLeftSideBoundary() const;

    /// @brief Compute the maximum candidate-slot id (sum of `uncomb
    ///        + comb` per type) over the loaded config; used as
    ///        `rightMax` for `createMapAnchor`. Result must stay
    ///        <= 3 to avoid RAM explosion (SwDD `OPEN-8`).
    int determineRightSideBoundary() const;

    // ---- Expression parsing & arg maps ----

    /// @brief Walk `expr` and extract every arg name with its
    ///        definition-set type, combinable flag, and connectable
    ///        flag.
    ///
    /// @details
    /// Pure walk: descends into nested `(>[...])` quantifier blocks,
    /// pulls every `name[arg0, arg1, ...]` token, and looks up the
    /// def-set tuple for each arg position via `coreExprMap_`. The
    /// returned `DefSetMap` is the canonical input to every
    /// string-path filter and to the encoders that produce the
    /// int-path twin.
    ///
    /// @param expr Expression text in numbered-variable form.
    /// @return `DefSetMap` keyed by arg name (as string). Empty
    ///         when `expr` carries no recognised expression.
    DefSetMap findArgMap(const std::string& expr) const;

    /// @brief Legacy reshuffle predecessor: rename bound variables
    ///        in `expr` to canonical `1..N` ids in first-occurrence
    ///        order.
    ///
    /// @details
    /// Largely superseded by `reshuffle` on the `rt_conjecturer*`
    /// branches but retained because some callers still consume
    /// its returned `(text, defSets, renameMap)` triple directly.
    ///
    /// @param expr Source expression.
    /// @param deep Whether to descend into nested existence heads.
    /// @return Tuple `(renamed, defSets, renameMap)`.
    std::tuple<std::string, DefSetMap, std::map<std::string,std::string>>
        renameVariablesInExpr(const std::string& expr, bool deep) const;

    // ---- Expression connection ----

    /// @brief String-path: merge two expressions via a substitution
    ///        map and a binary-sign vector.
    ///
    /// @details
    /// `subMap` carries the bijection between a subset of `map1`'s
    /// args and a subset of `map2`'s args; `binaryList` carries one
    /// sign bit per merge slot to enumerate negation variants.
    /// `connectToAnchor` toggles whether the second expression is
    /// the anchor (uses `mappingsMapAnchor_` instead of
    /// `mappingsMap_`). String-path twin of `connectExpressionsInt`;
    /// the int lane is the hot path and is byte-equivalent in
    /// output.
    ///
    /// @param expr1            First expression.
    /// @param expr2            Second expression (or anchor).
    /// @param map1             First expression's def-set map.
    /// @param map2             Second expression's def-set map.
    /// @param subMap           Bijection between subset args.
    /// @param binaryList       Per-slot sign bits.
    /// @param connectToAnchor  Toggle anchor-attach mode.
    /// @return Tuple `(success, mergedExpr, mergedDefSets)`.
    std::tuple<bool, std::string, DefSetMap>
        connectExpressions(const std::string& expr1, const std::string& expr2,
                          const DefSetMap& map1, const DefSetMap& map2,
                          const std::map<std::string,std::string>& subMap,
                          const std::vector<int>& binaryList, bool connectToAnchor) const;

    /// @brief String-path: enumerate every valid connection map
    ///        between the args of `map1` and `map2`.
    ///
    /// @details
    /// Uses `mappingsMap` (or `mappingsMapAnchor_` when `withAnchor`
    /// is true). One entry in the returned vector becomes the
    /// `subMap` argument of a subsequent `connectExpressions` call.
    ///
    /// @param map1         First def-set map.
    /// @param map2         Second def-set map.
    /// @param withAnchor   Toggle anchor-attach mode.
    /// @param mappingsMap  Pre-computed bijection table.
    /// @return All valid connection maps.
    std::vector<std::map<std::string,std::string>>
        makeAllConnectionMaps(const DefSetMap& map1, const DefSetMap& map2,
                             bool withAnchor, const MappingsMap& mappingsMap) const;

    // ---- Validation filters ----

    /// @brief First-pass structural sanity check on `expr`.
    ///
    /// @details
    /// Cheap gate the conjecturer applies before per-conjecture
    /// filters. Catches malformed expression shapes that would
    /// trip up downstream filters.
    bool exprGood(const std::string& expr) const;

    /// @brief Second-pass structural sanity check, given an `nse`
    ///        depth and the candidate's connected-arg map.
    ///
    /// @details
    /// Final survivor gate at the string-path level — combines
    /// numeric-arg contiguity, per-type combinable / uncombinable
    /// caps, and operator-block accounting. String-path twin of
    /// `exprGood2Int`.
    bool exprGood2(const std::string& expr, int nse, const DefSetMap& connectedMap) const;

    /// @brief Per-expression occurrence-count cap.
    ///
    /// @details
    /// For each registered expression name, counts how many times
    /// its handle appears in `expr`; rejects when any count
    /// exceeds the per-expression `max_count_per_conjecture` cap
    /// from `config_.data`. String-path twin of `numbersGoodInt`.
    /// Despite the name, this is a count-cap check, not an
    /// arg-id contiguity check.
    bool numbersGood(const std::string& expr) const;

    /// @brief Reject when the def-set type assigned to any arg is
    ///        inconsistent across positions, OR when the number of
    ///        combinable / uncombinable args of a given type
    ///        exceeds its config cap. String-path twin of
    ///        `checkDefSetsInt`.
    bool checkDefSets(const DefSetMap& argMap) const;

    /// @brief Reject when any def-set type's complexity exceeds
    ///        the configured pre-existence cap.
    ///
    /// @details
    /// Consults
    /// `parameters.max_complexity_if_anchor_parameter_connected_before_existence`.
    /// String-path twin of `checkComplexityLevelInt`.
    bool checkComplexityLevelForDefSets(const DefSetMap& argMap, int complexityLevel) const;

    /// @brief Pre-equality-emission qualification check on a head
    ///        expression.
    ///
    /// @details
    /// Used before letting `(=[...])` into a candidate's head
    /// position; rejects equalities among un-typed positions and
    /// other ill-formed cases that would slip past the more general
    /// filters.
    bool qualifiedForEquality(const std::string& expr) const;

    /// @brief Validate every operator-headed sub-expression in
    ///        `expression`.
    ///
    /// @details
    /// Skipped when `anchorAttached` is false on the `nse = 1`
    /// path — operator-output binding cannot be established with
    /// only one expression. See SwDD chapter `02_conjecturer.md`
    /// section *Operator head vs relation head*.
    bool evaluateOperatorExprs2(const std::string& expression, bool anchorAttached) const;

    /// @brief Collect operator-headed sub-expressions inside
    ///        `expr`. Pure walk over the expression tree.
    std::vector<std::string> extractOperatorExpressions(const std::string& expr) const;

    /// @brief Reject when `expression` matches an entry on
    ///        `config_.prohibited_combinations` (per-batch block
    ///        list).
    bool checkProhibitedCombinations(const std::string& expression) const;

    /// @brief Reject when the head of `conjecture` is on
    ///        `config_.prohibited_heads`. String-path twin of
    ///        `prohibitedHeadsGoodInt`.
    bool prohibitedHeadsGood(const std::string& conjecture) const;

    /// @brief Reject if the total argument count of `conjecture`
    ///        exceeds `parameters.max_number_args_expr`.
    bool countArgumentsFilter(const std::string& conjecture) const;

    /// @brief Reject when `conjecture` matches any compiled regex
    ///        in `config_.patterns_to_exclude`.
    bool patternInConjecture(const std::string& conjecture) const;

    /// @brief Reject when an `only_in_head` pattern matches outside
    ///        the head position of `conjecture`.
    bool onlyInHeadGood(const std::string& conjecture) const;

    /// @brief Reject if appending `newExpr` to `conjecture` would
    ///        push any per-operator complexity over its config cap.
    ///        String-path twin of `checkComplexityPerOpInt`.
    bool checkConjectureComplexityPerOperator(const std::string& conjecture, const std::string& newExpr) const;

    /// @brief Reject if `conjecture` is below the per-expression
    ///        minimum-size threshold (`min_size_expression`).
    bool checkMinSizeExpression(const std::string& conjecture) const;

    /// @brief Operator-head-specific validity check.
    ///
    /// @details
    /// Skipped on the `nse = 1` path because a single expression
    /// cannot consume an operator's output binding.
    bool checkInputVariablesTheoremOperatorHead(const std::string& theorem) const;

    /// @brief Enforce input-variable ordering across the chain to
    ///        suppress trivially-rearranged conjectures.
    ///
    /// @details
    /// Uses 13 sub-helpers (`findDigitArgs`, `getLeftRight`,
    /// `getRightChain`, `getLeftRightChains`, `getOperatorId`,
    /// `checkInputVariablePosition`, `removeOutputs`,
    /// `checkTautology`, `checkFunctions`, `onlyOneOperator`,
    /// `findEntryArgs2`, `getTertiaries`, `checkTertiaries`) to
    /// walk the chain and verify the ordering invariant.
    ///
    /// @invariant [I-10](../../docs/30_invariants.md#i-10) — bound
    ///            variables appear left-to-right in input-arg
    ///            positions.
    bool checkInputVariablesOrder(const std::string& theorem) const;

    /// @brief Equality-head canonicalisation guard.
    ///
    /// @details
    /// Rejects descending-ordered positive equality `(=[a, b])`
    /// with `stoi(a) > stoi(b)` so that the symmetric pair has
    /// only one orientation in the output set. Forwards to
    /// `countArgumentsFilter` (the AND-combine at the bottom),
    /// which provides the I-8 enforcement via its no-duplicate-
    /// args rule — `(=[x, x])` is rejected because the two args
    /// are identical, not because `controlEquality` itself spots
    /// the trivial form. See SwDD chapter `02_conjecturer.md`
    /// section *controlEquality* for the D-21 -> D-23 history.
    ///
    /// @invariant [I-8](../../docs/30_invariants.md#i-8) — trivial
    ///            equality forbidden in head; enforced via
    ///            `countArgumentsFilter`.
    bool controlEquality(const std::string& conjecture) const;

    /// @brief Pre-connection per-type cap.
    ///
    /// @details
    /// Reject when the union of candidate args from both sides
    /// would exceed the configured prior-connection cap. Consults
    /// `parameters.max_values_for_def_sets_prior_connection`.
    /// String-path twin of `checkDefSetsPriorInt`.
    bool checkDefSetsPriorToConnection(const DefSetMap& argsStatement, const DefSetMap& argsGrowingTheorem) const;

    // ---- Sub-functions for checkInputVariablesOrder ----

    /// @brief Collect every "digit" arg id appearing in `theorem`.
    ///
    /// @details
    /// A digit arg is a bound variable that will appear in the
    /// rendered output as a numeric token (1, 2, ...). The
    /// returned set drives the ordering checks in
    /// `checkInputVariablesOrder`.
    std::set<std::string> findDigitArgs(const std::string& theorem) const;

    /// @brief Walk a chain from a chosen `expression` and partition
    ///        the visited digit args into left-side and right-side
    ///        contributions.
    ///
    /// @param chain      Chain entries.
    /// @param expression Anchor entry to walk from.
    /// @param digits     Digit args to consider.
    /// @param counter    Recursion depth tracker.
    /// @param visited    Mutable set of already-visited entries.
    /// @return Pair `(leftDigits, rightDigits)`.
    std::pair<std::set<std::string>, std::set<std::string>>
        getLeftRight(const std::vector<std::string>& chain,
                     const std::string& expression,
                     const std::set<std::string>& digits, int counter,
                     std::set<std::string>& visited) const;

    /// @brief Compute the right-side sub-chain reachable from
    ///        `head`, marking visited entries to prevent
    ///        re-traversal.
    std::vector<std::string>
        getRightChain(const std::vector<std::string>& chain,
                      const std::string& head, std::set<std::string>& visited) const;

    /// @brief Split `chain` into its left and right subchains
    ///        relative to the conjecture's outer implication.
    std::pair<std::vector<std::string>, std::vector<std::string>>
        getLeftRightChains(const std::vector<std::string>& chain) const;

    /// @brief Build an operator-skeleton string for `expr`.
    ///
    /// @details
    /// Returns the empty string when `expr`'s core expression is not
    /// on the operators list (e.g. `(in[1, 2])` -> `""`). For an
    /// operator-headed expression like `(in2[1, 2, 3])`, returns
    /// the expression with every input/output arg replaced by an
    /// empty string — leaving a "skeleton" that the caller uses to
    /// compare operator usage across chain entries. Despite the
    /// name, this is NOT the bare expression name.
    std::string getOperatorId(const std::string& expr) const;

    /// @brief Check that input-variable positions in `chain`
    ///        respect the canonical left-to-right order over
    ///        `digits`.
    bool checkInputVariablePosition(const std::vector<std::string>& chain,
                                    const std::set<std::string>& digits) const;

    /// @brief Drop output-arg indices from each entry in `chain`.
    /// @return Collapsed chain text with only input args
    ///         preserved.
    std::string removeOutputs(const std::vector<std::string>& chain) const;

    /// @brief Reject a conjecture whose left and right subchains
    ///        are identical modulo bound-var renaming.
    bool checkTautology(const std::vector<std::string>& leftChain, const std::vector<std::string>& rightChain) const;

    /// @brief Reject a chain whose function-position usage violates
    ///        the conjecturer's well-formedness rules.
    bool checkFunctions(const std::vector<std::string>& chain) const;

    /// @brief Test whether `chain` carries exactly one operator
    ///        expression (used as a special-case relaxation gate
    ///        in `checkInputVariablesOrder`).
    bool onlyOneOperator(const std::vector<std::string>& chain) const;

    /// @brief Compute the closure of input args reachable from
    ///        position `index`, threading through output-arg
    ///        bindings.
    ///
    /// @param inputArgsList   Per-position input-arg lists.
    /// @param outputArgsList  Per-position output-arg lists.
    /// @param index           Starting position.
    /// @param visited         Mutable visit-tracking set for cycle
    ///                        prevention.
    /// @return Closure of reachable input args.
    std::set<std::string> findEntryArgs2(
        const std::vector<std::vector<std::string>>& inputArgsList,
        const std::vector<std::vector<std::string>>& outputArgsList,
        int index, std::set<int>& visited) const;

    /// @brief Collect tertiary (third-tier) bound variables from
    ///        `chain` (those not directly cited in the head's
    ///        input positions).
    std::set<std::string> getTertiaries(const std::vector<std::string>& chain) const;

    /// @brief Check tertiary-variable compatibility between left
    ///        and right subchains; backstop for `checkTautology`.
    bool checkTertiaries(const std::vector<std::string>& leftChain, const std::vector<std::string>& rightChain) const;

    // ---- Reshuffling & mirroring ----

    /// @brief Canonicalise a conjecture into its `theorems.txt` /
    ///        `reshuffled_theorems.txt` form.
    ///
    /// @details
    /// Pipeline (per SwDD chapter `02_conjecturer.md` section
    /// *Reshuffle pipeline*):
    /// 1. Flat-walk rename — first-occurrence numbering of the
    ///    arg-id space.
    /// 2. Existence-head pinning — extract the outermost existence
    ///    bv-list before the flat-walk so existence bvs land at
    ///    canonical positions.
    /// 3. Contiguous-arg renumber post-connect — drop holes left
    ///    by `connectExpressionsInt`'s subMap.
    /// 4. Anchor position-0 pin — anchor args are never permuted.
    ///
    /// Each stage's output feeds the next; skipping any stage
    /// produces drift in the canonical form vs the `_mirrored`
    /// companion.
    ///
    /// @param expr Source expression.
    /// @param deep Descend into nested existence heads when true.
    /// @return Tuple `(canonical, defSets, renameMap)`.
    std::tuple<std::string, DefSetMap, std::map<std::string,std::string>>
        reshuffle(const std::string& expr, bool deep) const;

    /// @brief Build the mirror variant of a reshuffled conjecture.
    ///
    /// @details
    /// The mirror swaps left and right of the implication; for
    /// equality heads the mirror is the symmetric orientation.
    /// `anchorFirst` toggles whether the anchor lands at position
    /// 0 of the result (used when the caller has not yet anchor-
    /// pinned).
    ///
    /// @invariant [I-9](../../docs/30_invariants.md#i-9) — a
    ///            mirror survives only if it differs from its
    ///            source after both pass through `reshuffle`.
    std::string createReshuffledMirrored(const std::string& expr, bool anchorFirst = false) const;

    /// @brief Count `(>[` operator-block headers in `s`. Equivalent
    ///        to the conjecture's complexity level. String-path
    ///        twin of `countOperatorOccurrencesInt`.
    int countOperatorOccurrences(const std::string& s) const;

    /// @brief Test whether the output variable of `headExpr`
    ///        survives in `fullExpr` after the conjecture is
    ///        constructed.
    bool staysOutputVariable(const std::string& fullExpr, const std::string& headExpr) const;

    // ---- Worker functions ----

    /// @brief String-path worker: combine `statement` with
    ///        `growingTheorem` and emit `(connected,
    ///        anchor-attached, reshuffled, mirrored)` lists.
    ///
    /// @details
    /// Drives the per-candidate string-path filter cascade. Mirror
    /// of `singleThreadCalculationInt` on the int path. Both paths
    /// must agree on output (see `worker_*` test category for the
    /// int-vs-string identity check on a fixed candidate).
    ///
    /// @param statement           First input expression.
    /// @param growingTheorem      Second input expression (the
    ///                            partial theorem being grown).
    /// @param nseStatement        Number of simple expressions in
    ///                            `statement`.
    /// @param nseGrowingTheorem   Number of simple expressions in
    ///                            `growingTheorem`.
    /// @param argsStatement       `statement`'s def-set map.
    /// @param argsGrowingTheorem  `growingTheorem`'s def-set map.
    /// @return Filled `WorkerResult`.
    WorkerResult singleThreadCalculation(
        const std::string& statement, const std::string& growingTheorem,
        int nseStatement, int nseGrowingTheorem,
        const DefSetMap& argsStatement, const DefSetMap& argsGrowingTheorem) const;

    /// @brief String-path worker: attach a single expression
    ///        directly to the anchor (`nse = 1` path).
    ///
    /// @details
    /// Skips operator-head validity checks — an operator's output
    /// cannot be bound to anything in this configuration. Only
    /// runs when `parameters.min_number_simple_expressions == 1`.
    WorkerResult singleExprAnchorConnection(
        const std::string& expr, const DefSetMap& exprDefSets) const;

    /// @brief Detect an ungrounded-operator-head conjecture and
    ///        rewrite it with the operator's allowed argument
    ///        wrapped in a negated-universal existence head.
    ///
    /// @details
    /// Pass-through if no reformulation applies; the new form is
    /// returned otherwise.
    std::string reformulateOperatorHead(const std::string& conjecture) const;

    /// @brief Per-expression after-existence size cap.
    ///
    /// @details
    /// Returns true iff every leaf expression in the
    /// pre-reformulation `conj` has
    /// `max_size_expression_after_existence >= leafCount`.
    /// Intended to be called on the post-anchor-attach,
    /// pre-existence-reformulation string; a single disintegrate
    /// pass yields the flat chain + head and we check each leaf
    /// once (no descent into any nested structure).
    ///
    /// @param conj      Post-anchor-attach,
    ///                  pre-existence-reformulation string.
    /// @param leafCount Already-available leaf count (typically
    ///                  `nse + 1`).
    bool passesMaxSizeAfterExistence(const std::string& conj, int leafCount) const;

    /// @brief Post-existence per-type 2-tuple cap.
    ///
    /// @details
    /// Mirror of `checkComplexityLevelForDefSets` but keyed by
    /// `max_complexity_if_anchor_parameter_connected_after_existence`.
    /// Operates on the pre-reformulation post-anchor-attach
    /// string; gathers every def-set type that appears on any leaf
    /// via the static per-expression definition, then rejects when,
    /// for some capped type T, all three hold: complexity-level
    /// exceeds T's complexity cap, arity-sum exceeds T's
    /// arity-sum cap, AND a slot of type T appears in non-anchor
    /// leaves. See SwDD chapter `02_conjecturer.md` section
    /// *passesComplexityAfterExistence* (post-D-23 3-condition
    /// rule).
    bool passesComplexityAfterExistence(const std::string& conj) const;

    /// @brief Per-type cap on distinct anchor-slot values appearing
    ///        in non-anchor leaves.
    ///
    /// @details
    /// For each def-set type T present in
    /// `max_distinct_anchor_values_per_type`, counts how many
    /// DISTINCT anchor-slot values of type T appear as args of
    /// non-anchor leaves of `conj`. Returns false (reject) if any
    /// type's count exceeds the configured cap. Empty config map
    /// -> always true (filter off). Walk descends into nested
    /// `!(>[...]...)` existence heads so the cap applies to the
    /// full body, not just the top-level chain.
    bool passesMaxDistinctAnchorValuesPerType(const std::string& conj) const;

    /// @brief `(in[...])`-premise shape filter.
    ///
    /// @details
    /// Top-of-function gate — anchor-membership-axiom rejection
    /// ([D-23](../../docs/40_decisions.md#d-23)). Reject any
    /// `(in[v, X])` premise (positive or negated) where BOTH `v`
    /// AND `X` are anchor-slot values, since the anchor's own
    /// axioms already entail it.
    ///
    /// Cnt-shape rules (post-anchor-membership gate). When `hasIn`
    /// is true, accept iff one of:
    /// 1. `cnt == 1` AND head is an existence form
    ///    `!(>[...]...)`.
    /// 2. `cnt == 2` AND at least one of the two non-anchor
    ///    premises is negated.
    /// 3. `cnt == 2` AND there exists a positive `(in[v, X])`
    ///    premise whose first arg `v` participates elsewhere
    ///    (neutralisation rule).
    ///
    /// `cnt >= 3` is rejected unconditionally.
    ///
    /// @warning `parameters.apply_in_premise_filter` is dead code;
    ///          this function ignores it. SwDD `OPEN-9`.
    bool passesInPremiseFilter(const std::string& conj) const;

    /// @brief Emit one negated-premise variant per negatable
    ///        premise in `conj`.
    ///
    /// @details
    /// For each premise of `conj` whose core expression has
    /// `allow_negation = true` in the config, emit one new
    /// conjecture where that single premise is wrapped in
    /// `!(...)`. The head of the outer implication is never
    /// negated (which implicitly excludes the `!(>[...])`
    /// existence-head form when it sits at head position).
    /// Multiple negatable premises -> one new variant per premise
    /// (never co-negated). The original is NOT included in the
    /// returned list.
    std::vector<std::string> generateNegatedPremiseVariants(const std::string& conj) const;

    /// @brief Existence-head reformulation eligibility test.
    ///
    /// @details
    /// True iff the ungrounded-operator-head filter rejection hits
    /// AND the candidate qualifies for existence reformulation:
    /// head `coreExpr` has non-empty `allowed_for_existence`,
    /// chain contains `(in[x, X])` for `x` at an allowed position,
    /// and `x` occurs nowhere else in chain besides `P_x` and the
    /// head. Caller runs `reformulateToExistenceHead` in lieu of
    /// rejecting the candidate when this returns true.
    bool triggersExistenceReformulation(const std::string& theorem) const;

    /// @brief Rebuild the candidate with the allowed arg wrapped
    ///        in a negated-universal existence head.
    /// @pre `triggersExistenceReformulation(theorem)` was true at
    ///      the same state.
    std::string reformulateToExistenceHead(const std::string& theorem) const;

    // ---- OR theorem conjecture generation ----

    /// @brief Generate `(existence, companion)` pairs for OR-shaped
    ///        conjectures from config flags.
    ///
    /// @details
    /// Replaces the older `generateOrConjectures` path that read
    /// `or_pairs.txt`. The current implementation derives pairs
    /// directly from per-expression
    /// `allow_to_constitute_existence` flags. Pairs are emitted
    /// into `theorems.txt` alongside ordinary conjectures and the
    /// prover treats them via the `or disintegration` /
    /// `or convergence` tags.
    std::vector<std::pair<std::string,std::string>> generateOrConjectures() const;

    // ---- Int-path: encode/decode ----

    /// @brief Build `nameMap_` from every expression name, handle,
    ///        and def-set text appearing in `config_.data`. Called
    ///        once during construction.
    void buildNameMap();

    /// @brief Populate `intExprConfigs_` (indexed by nameId) plus
    ///        the per-defSetId limit arrays from `config_`.
    ///
    /// @details
    /// Walks every `ExpressionDescription` once, projecting it
    /// into the dense int16_t per-expression record. A second
    /// pass over the per-type config maps populates
    /// `maxForDefSets_`, `maxForUncombDefSets_`,
    /// `maxForDefSetsPrior_`, and `maxComplexityAnchorConn_`.
    /// After this returns the int-path hot loop can answer every
    /// per-expression / per-def-set query with a single indexed
    /// read.
    void buildIntExprConfigs();

    /// @brief Serialize an MPL expression string into a flat
    ///        `IntConjBuf`.
    ///
    /// @details
    /// Walks `expr` block by block, emitting `[boundCount,
    /// bv0..bvN, nameId, arity, arg0..argN]` per quantifier layer.
    /// The output buffer is suitable for the int-path filter
    /// cascade — no further string-form parsing is needed once it
    /// is built. Inverse of `decodeExpr`; round-trip is
    /// byte-stable on every conjecture emitted by the conjecturer
    /// (tested via `intpath_*` suite).
    ///
    /// @param expr Expression in numbered-variable form.
    /// @return Encoded buffer with `len <= MAX_CONJ_BUF`.
    IntConjBuf encodeExpr(const std::string& expr) const;

    /// @brief Deserialize a flat `IntConjBuf` back into its MPL
    ///        string form. Inverse of `encodeExpr`.
    ///
    /// @details
    /// Walks the buffer block by block, reconstructing the
    /// `(>[bv0,bv1](nameHandle[arg0,arg1])(...))` shape. The
    /// returned string is byte-equivalent to the input of
    /// `encodeExpr` for any expression the conjecturer emits.
    std::string decodeExpr(const IntConjBuf& buf) const;

    /// @brief Serialize a string-keyed `DefSetMap` into the
    ///        parallel-array `IntDefSetMap`.
    ///
    /// @details
    /// Each `argId` is decoded from the stringified arg name
    /// (e.g. `"5"` -> `5`); each def-set text goes through
    /// `nameMap_.encode`. The combinable / connectable bools are
    /// projected to int16_t.
    IntDefSetMap encodeDefSetMap(const DefSetMap& dsm) const;

    /// @brief Deserialize an `IntDefSetMap` back into a
    ///        string-keyed `DefSetMap`. Inverse of
    ///        `encodeDefSetMap`.
    DefSetMap decodeDefSetMap(const IntDefSetMap& idsm) const;

    // ---- Int-path: connection ----

    /// @brief Int-path: merge two encoded expressions via a
    ///        connection map and binary-sign vector. Hot-path
    ///        twin of `connectExpressions`.
    ///
    /// @details
    /// `subMap` carries the bijection between a subset of `map1`'s
    /// args and a subset of `map2`'s args; `binaryList` carries
    /// one sign bit per merge slot to enumerate negation variants.
    /// `connectToAnchor` toggles whether the second expression is
    /// the anchor (uses `mappingsMapAnchor_` instead of
    /// `mappingsMap_`). Outputs are written into `outExpr` /
    /// `outMap` (caller-owned) to avoid per-call allocation; this
    /// keeps the hot loop allocation-free.
    ///
    /// @return `true` on a successful merge; `false` on rejection.
    bool connectExpressionsInt(
        const IntConjBuf& expr1, const IntConjBuf& expr2,
        const IntDefSetMap& map1, const IntDefSetMap& map2,
        const IntConnMap& subMap,
        const int16_t* binaryList, int binaryLen,
        bool connectToAnchor,
        IntConjBuf& outExpr, IntDefSetMap& outMap) const;

    /// @brief Int-path: enumerate every connection map between two
    ///        encoded def-set maps. Hot-path twin of
    ///        `makeAllConnectionMaps`.
    ///
    /// @details
    /// Walks the precomputed `mappingsMap` (or `mappingsMapAnchor_`
    /// when `withAnchor` is true), generating every valid
    /// bijection between the two arg sets. Output appended to
    /// `outMaps`. Cap on the number of maps written is
    /// `MAX_CONN_MAPS`; misconfiguration that would blow past
    /// this triggers a hard cut.
    void makeAllConnectionMapsInt(
        const IntDefSetMap& argsMap1, const IntDefSetMap& argsMap2,
        bool withAnchor, const MappingsMap& mappingsMap,
        std::vector<IntConnMap>& outMaps) const;

    // ---- Int-path: filters ----

    /// @brief Int-path twin of `repetitionsExist`. Reject when the
    ///        same leaf (same `nameId` AND same arg list) appears
    ///        more than once in `buf`. NOT about repeated arg ids
    ///        inside a single leaf.
    bool repetitionsExistInt(const IntConjBuf& buf) const;

    /// @brief Int-path twin of `numbersGood`. Per-expression
    ///        occurrence-count cap: counts how many times each
    ///        registered nameId appears in `buf`; rejects when
    ///        any count exceeds its `maxCountPerConj`.
    bool numbersGoodInt(const IntConjBuf& buf) const;

    /// @brief Int-path twin of `checkDefSets`. Reject if def-set
    ///        type per-arg counts exceed the per-type
    ///        `max_values_for_def_sets` /
    ///        `max_values_for_uncomb_def_sets` caps.
    bool checkDefSetsInt(const IntDefSetMap& argMap) const;

    /// @brief Int-path twin of `checkComplexityLevelForDefSets`.
    ///        Reject if any def-set type's complexity exceeds the
    ///        configured pre-existence cap (consults
    ///        `maxComplexityAnchorConn_`).
    bool checkComplexityLevelInt(const IntDefSetMap& argMap, int complexityLevel) const;

    /// @brief Int-path twin of `checkDefSetsPriorToConnection`.
    ///        Pre-merge per-type cap: reject when the union of
    ///        args from both sides would exceed the configured
    ///        prior-connection cap.
    bool checkDefSetsPriorInt(const IntDefSetMap& argsStmt, const IntDefSetMap& argsGT) const;

    /// @brief Int-path twin of `exprGood2`. Final survivor gate
    ///        at the int-path level — combines numeric-arg
    ///        contiguity, per-type combinable / uncombinable caps,
    ///        and operator-block accounting.
    bool exprGood2Int(const IntConjBuf& buf, int nse, const IntDefSetMap& connMap) const;

    /// @brief Int-path twin of `onlyInHeadGood`. Reject when a
    ///        "head-only" expression appears outside the head
    ///        position of `buf`.
    bool onlyInHeadGoodInt(const IntConjBuf& buf) const;

    /// @brief Int-path twin of `prohibitedHeadsGood`. Reject when
    ///        the head of `buf` is on `config_.prohibited_heads`.
    bool prohibitedHeadsGoodInt(const IntConjBuf& buf) const;

    /// @brief Count identity entries in `connMap` (entries
    ///        `k -> k`). Int-path twin of
    ///        `getNumberRemovableArgs`.
    int getNumberRemovableArgsInt(const IntConnMap& connMap) const;

    /// @brief Sort `removableArgs` (length `numRemovable`) by
    ///        their first-occurrence position inside `expr`.
    ///        Result written to `sortedOut` (caller-owned, must
    ///        hold `numRemovable` entries).
    void sortByOccurrenceInt(const IntConjBuf& expr, const int16_t* removableArgs, int numRemovable,
                             int16_t* sortedOut) const;

    /// @brief Int-path twin of `countOperatorOccurrences`. Counts
    ///        operator-block headers in `buf`.
    int countOperatorOccurrencesInt(const IntConjBuf& buf) const;

    /// @brief Int-path twin of
    ///        `checkConjectureComplexityPerOperator`. Reject if
    ///        appending `statement` to `growingTheorem` would push
    ///        any per-operator complexity over its config cap.
    bool checkComplexityPerOpInt(const IntConjBuf& growingTheorem, const IntConjBuf& statement) const;

    // ---- Int-path: workers ----

    /// @brief Int-path twin of `singleThreadCalculation`. Hot-path
    ///        per-candidate driver.
    ///
    /// @details
    /// Drives the per-candidate int-path filter cascade. Mirror
    /// of `singleThreadCalculation` on the string path. Both paths
    /// must agree on output (see `worker_*` test category).
    ///
    /// @return Filled `WorkerResult`.
    WorkerResult singleThreadCalculationInt(
        const IntConjBuf& intStatement, const IntConjBuf& intGrowingTheorem,
        int nseStatement, int nseGrowingTheorem,
        const IntDefSetMap& intArgsStatement, const IntDefSetMap& intArgsGrowingTheorem) const;

    /// @brief Int-path twin of `singleExprAnchorConnection`.
    ///        Attach a single encoded expression directly to the
    ///        encoded anchor (`nse = 1` path).
    WorkerResult singleExprAnchorConnectionInt(
        const IntConjBuf& intExpr, const IntDefSetMap& intExprDefSets) const;
};

} // namespace conj
