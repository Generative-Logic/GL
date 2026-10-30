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
#include "../logical_block/statement_registry_operations.hpp"
#include "proof_grid_builder.hpp"
#include "../compiler/definition_expansion.hpp"
#include <string_view>
#include "../infra/diagnostics_log.hpp"
#include "or_theorem_builder.hpp"
#include "../logical_block/block_mail_processor.hpp"
#include "variable_copy_detector.hpp"
namespace gl {
/// @brief Bind the proof grid builder operation inputs.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param anchorInfo Borrowed ce::AnchorInfo interface or state; must outlive this component.
/// @param anchorScopeQueries Borrowed AnchorScopeQueries interface or state; must outlive this component.
/// @param blockIdentity Borrowed LogicBlockIdentity interface or state; must outlive this component.
/// @param body Borrowed Memory interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param hashRuleInstaller Borrowed HashRuleInstaller interface or state; must outlive this component.
/// @param inductionUsefulness Borrowed InductionUsefulness interface or state; must outlive this component.
/// @param integrationProcessor Borrowed IntegrationProcessor interface or state; must outlive this component.
/// @param lbStore Borrowed LbStore interface or state; must outlive this component.
/// @param mailLog Borrowed MailLog interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param permanentBodies Borrowed std::vector interface or state; must outlive this component.
/// @param scopeTypes Borrowed ScopeTypeResolver interface or state; must outlive this component.
/// @param simpleMapStore Borrowed SimpleMapStore interface or state; must outlive this component.
/// @param statementMemory Borrowed StatementMemory interface or state; must outlive this component.
/// @param theoremPublisher Borrowed TheoremPublisher interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
ProofGridBuilder::ProofGridBuilder(AdmissionMemory& admissionMemory, const ce::AnchorInfo& anchorInfo, AnchorScopeQueries& anchorScopeQueries, LogicBlockIdentity& blockIdentity, Memory& body, const CompiledDefinitionRegistry& definitions, HashRuleInstaller& hashRuleInstaller, InductionUsefulness& inductionUsefulness, IntegrationProcessor& integrationProcessor, LbStore& lbStore, MailLog& mailLog, const ProverParameters& parameters, std::vector<Memory*>& permanentBodies, ScopeTypeResolver& scopeTypes, SimpleMapStore& simpleMapStore, StatementMemory& statementMemory, TheoremPublisher& theoremPublisher) : admissionMemory(admissionMemory), anchorInfo(anchorInfo), anchorScopeQueries(anchorScopeQueries), blockIdentity(blockIdentity), body(body), definitions(definitions), hashRuleInstaller(hashRuleInstaller), inductionUsefulness(inductionUsefulness), integrationProcessor(integrationProcessor), lbStore(lbStore), mailLog(mailLog), parameters(parameters), permanentBodies(permanentBodies), scopeTypes(scopeTypes), simpleMapStore(simpleMapStore), statementMemory(statementMemory), theoremPublisher(theoremPublisher) {}



/// @brief The arguments a weak induction hypothesis on @p digit keeps fixed.
///
/// @details
/// Starts from the conjecture's digit arguments without @p digit and without
/// @p freeDigits, then closes forward over the premise chain: an operator
/// premise whose configured inputs are all fixed fixes its configured outputs.
/// The head takes no part. A twin partner of @p digit is passed in
/// @p freeDigits, so the hypothesis is universal in it and in every output it
/// alone would have fixed (D-587).
///
/// @param theorem    The canonical compiled conjecture.
/// @param digit      The induction variable.
/// @param freeDigits Digit arguments the hypothesis leaves universal.
/// @return The fixed arguments, ascending.
/// @invariant Reads the compiled configuration only; writes nothing.
/// @see createAuxyImplication, twinInductionPartners.
std::set<std::string>
ProofGridBuilder::findImmutableArgs(const std::string& theorem,
    const std::string& digit, const std::set<std::string>& freeDigits) {
    std::set<std::string> immutables;

    // Disintegrate implication to a chain of left subexpressions.
    std::vector< std::tuple<
        std::string,                  // leftExpr
        std::vector<std::string>,     // args of current '>'
        std::set<std::string>         // left node arguments (unused here)
    > > tempChain;

    ce::disintegrateImplication(theorem, tempChain, this->definitions.configurations());

    std::vector<std::string> chain;
    chain.reserve(tempChain.size());
    for (std::size_t i = 0; i < tempChain.size(); ++i) {
        chain.push_back(std::get<0>(tempChain[i]));
    }

    // Start with all digit args except the given 'digit'. findDigitArgs is now 0%
    // heap (sorted-unique StrSpan run); insert directly into the still-heap set.
    StrSpan digitBuf[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
    const int32_t digitN = scopeTypes.findDigitArgs(theorem, anchorInfo, digitBuf,
                                         ExecutionParameters::MAX_ADMISSION_REM_ARGS);
    for (int32_t i = 0; i < digitN; ++i)
        immutables.emplace(digitBuf[i].ptr, static_cast<std::size_t>(digitBuf[i].len));
    immutables.erase(digit);
    for (const std::string& freeDigit : freeDigits) immutables.erase(freeDigit);

    // Propagate immutability forward along the chain
    bool found = true;
    while (found) {
        found = false;

        for (std::size_t i = 0; i < chain.size(); ++i) {
            const std::string& expression = chain[i];
            const StrSpan coreSpan = extractExpressionSpan(StrSpan(expression));

            // Use the pre-populated 'operators' attribute for efficiency
            if (this->definitions.operatorNames().find(std::string_view(coreSpan.ptr,
                    static_cast<std::size_t>(coreSpan.len))) != this->definitions.operatorNames().end()) {

                const ce::CoreExpressionConfig* cfg = this->definitions.coreConfig(coreSpan);
                assert(cfg && "findImmutableArgs: operator core must be compiled");
                const std::vector<std::string> args = ce::getArgs(expression);

                // 1. Check if all configured Inputs are already Immutable
                bool inputsAreImmutable = true;
                for (int idx : cfg->inputIndices) {
                    // BLUNT ASSERT: Fail if config index is out of bounds for this expression
                    assert(idx >= 0 && idx < static_cast<int>(args.size()));

                    if (immutables.find(args[idx]) == immutables.end()) {
                        inputsAreImmutable = false;
                        break;
                    }
                }

                // 2. If inputs are immutable, add configured Outputs to immutables
                if (inputsAreImmutable) {
                    for (int idx : cfg->outputIndices) {
                        // BLUNT ASSERT: Fail if config index is out of bounds
                        assert(idx >= 0 && idx < static_cast<int>(args.size()));

                        const std::string& outputArg = args[idx];
                        if (immutables.find(outputArg) == immutables.end()) {
                            immutables.insert(outputArg);
                            found = true;
                        }
                    }
                }
            }
        }
    }

    return immutables;
}


std::vector<std::string> ProofGridBuilder::extractSubstringsForAuxy(const std::string& expr2) {
    // Lexical twin of the unanchored iterator over !?\(([^>(\[]+\[[^\]]*\])\):
    // at each '(' the [^>(\[]+ run is maximal and excludes '>','(','[', so its
    // terminator is the first such byte and the pattern then REQUIRES '[' — if
    // the terminator is '>' or '(' the attempt fails with no shorter-run retry
    // possible ('[' cannot occur inside the run); [^\]]* terminates at the
    // first ']'. One deterministic scan per attempt decides exactly what the
    // engine decides. Leftmost / resume-after-match / failed-start-advances-one;
    // a later match may START inside a failed attempt's window ("((a[b]))"
    // matches at offset 1), which the ++pos rule preserves. A '!' immediately
    // preceding the matched '(' belongs to the subexpression — a NEGATED leaf
    // premise (e.g. the OR-companion theorem's !(existence3[...]) universal)
    // must keep its negation, or the induction-hypothesis rule built from the
    // chain silently inverts that premise into an unsound positive form. The
    // whole match (WITH the leading '!' when present and the surrounding
    // parens) is pushed as a std::string — the createAuxyImplication
    // tuple-return boundary heap. THE RULE: in the burst kernel the only
    // permitted heap is the Rule-14 hashburst_trace.txt dump; this
    // tuple-return boundary heap is an OPEN violation to statify, not a
    // sanctioned island.
    const StrSpan s(expr2);
    const int32_t n = s.len;
    std::vector<std::string> out;
    int32_t pos = 0;
    while (pos < n) {
        if (s.ptr[pos] != '(') { ++pos; continue; }
        int32_t i = pos + 1;
        while (i < n && s.ptr[i] != '>' && s.ptr[i] != '(' && s.ptr[i] != '[') ++i;
        if (i - (pos + 1) < 1) { ++pos; continue; }   // [^>(\[]+ needs >= 1
        if (i >= n || s.ptr[i] != '[') { ++pos; continue; }
        int32_t j = i + 1;
        while (j < n && s.ptr[j] != ']') ++j;
        if (j >= n) { ++pos; continue; }              // no ']'
        if (j + 1 >= n || s.ptr[j + 1] != ')') { ++pos; continue; }
        const int32_t start = (pos > 0 && s.ptr[pos - 1] == '!') ? pos - 1 : pos;
        out.push_back(std::string(s.ptr + start,
            static_cast<std::size_t>(j + 1 - start + 1)));  // s[start .. j+1] inclusive
        pos = j + 2;
    }
    return out;
}


std::tuple<
    std::string,
    std::vector<std::string>,
    std::string,
    std::set<std::string>,
    std::string,
    std::string
>
ProofGridBuilder::createAuxyImplication(const std::string& expr,
    const std::string& arg,
    const std::string& recArg,
    const std::set<std::string>& digitArgs,
    const std::string& anchorName) {
    // untouchables as a sorted-unique StrSpan array (spans into digitArgs /
    // recArg / subexprs, all alive to function end); membership via
    // binary_search(compareSpans); replMap2 reuses the same run. The tuple
    // returns (chain / remainingArgsKey / subexprs) are still heap-allocated
    // (I-138) — an OPEN violation to statify, not a sanctioned "by decision"
    // carve-out. THE RULE: in the burst kernel the only permitted heap is the
    // Rule-14 hashburst_trace.txt dump.
    constexpr int kUntouchCap = 4 * ExecutionParameters::MAX_ARITY;
    StrSpan untouchables[kUntouchCap];
    int32_t untouchN = 0;
    for (const std::string& d : digitArgs) {
        assert(untouchN < kUntouchCap);
        untouchables[untouchN++] = StrSpan(d);
    }
    assert(untouchN < kUntouchCap);
    untouchables[untouchN++] = StrSpan(recArg);

    std::vector<std::string> chain;
    std::string zeroArgName;
    std::string sName;

    // replace arg -> recArg in the whole expression (value is an input slice,
    // so replaceKeysToString needs no arena)
    StrReplacement replPair[1];
    replPair[0].key = StrSpan(arg);
    replPair[0].value = StrSpan(recArg);
    const std::string replExpr = replaceKeysToString(StrSpan(expr), replPair, 1);

    // collect non-top-level subexpressions "(X[...])" with X != '>'
    const std::vector<std::string> subexprs = extractSubstringsForAuxy(replExpr);
    for (std::size_t i = 0; i < subexprs.size(); ++i) {
        const std::string& sub = subexprs[i];
        if (sub.find(anchorName) != std::string::npos) {
            StrSpan anchorArgs[ExecutionParameters::MAX_ARITY];
            const int32_t anchorN = getArgsSpans(
                StrSpan(sub), anchorArgs, ExecutionParameters::MAX_ARITY);
            // add all anchor args to untouchables (spans into the stable sub)
            for (int32_t k = 0; k < anchorN; ++k) {
                assert(untouchN < kUntouchCap);
                untouchables[untouchN++] = anchorArgs[k];
            }

            zeroArgName = anchorScopeQueries.findZeroArgNameFromAnchor(sub);
            sName = anchorScopeQueries.findSArgNameFromAnchor(sub);
        }
        else {
            chain.push_back(sub);
        }
    }

    // Sort + dedup so binary_search membership and the replMap2 walk reproduce
    // the former std::set (ascending, unique) byte-for-byte.
    std::sort(untouchables, untouchables + untouchN,
        [](const StrSpan& a, const StrSpan& b) { return compareSpans(a, b) < 0; });
    {
        int32_t w = 0;
        for (int32_t r = 0; r < untouchN; ++r) {
            if (w == 0 || !equalSpans(untouchables[w - 1], untouchables[r])) {
                untouchables[w++] = untouchables[r];
            }
        }
        untouchN = w;
    }

    // remaining_args_key: args from all but the last element of chain that intersect untouchables
    std::set<std::string> remainingArgsKey;
    if (chain.size() >= 1U) {
        for (std::size_t elementIndex = 0; elementIndex + 1 < chain.size(); ++elementIndex) {
            StrSpan elementArgs[ExecutionParameters::MAX_ARITY];
            const int32_t elementN = getArgsSpans(
                StrSpan(chain[elementIndex]), elementArgs,
                ExecutionParameters::MAX_ARITY);
            for (int32_t a = 0; a < elementN; ++a) {
                if (std::binary_search(untouchables, untouchables + untouchN,
                        elementArgs[a],
                        [](const StrSpan& x, const StrSpan& y) {
                            return compareSpans(x, y) < 0;
                        })) {
                    remainingArgsKey.insert(elementArgs[a].toStdString());
                }
            }
        }
    }

    // prefix all untouchables with "u_" inside chain
    if (untouchN > 0) {
        const unsigned uSlot = (gl::WorkerContext::coreId() >= 0)
            ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : scratchArenas().slotCount() - 1;
        ScratchArena& uArena = scratchArenas().forSlot(uSlot);
        ScratchScope uScope(uArena);

        // Build the "u_"+key run ONCE (fresh bytes on the string-scratch arena
        // under a ScratchScope spanning the whole chain loop), then reuse it
        // across every chain element via replaceKeysScratch.
        StrReplacement replPairs2[kUntouchCap];
        for (int32_t i = 0; i < untouchN; ++i) {
            char* v = uArena.allocBytes(2 + untouchables[i].len);
            v[0] = 'u';
            v[1] = '_';
            if (untouchables[i].len > 0) {
                std::memcpy(v + 2, untouchables[i].ptr,
                    static_cast<size_t>(untouchables[i].len));
            }
            replPairs2[i].key = untouchables[i];
            replPairs2[i].value = StrSpan(v, 2 + untouchables[i].len);
        }
        for (std::size_t i = 0; i < chain.size(); ++i) {
            chain[i] = replaceKeysScratch(
                uArena, StrSpan(chain[i]), replPairs2, untouchN).toStdString();
        }
    }

    // head is last element; chain are the preceding elements
    assert(!chain.empty());
    const std::string head = chain.back();
    chain.pop_back();

    const std::string implication = gl::ImplicationBuilder::reconstructImplication(chain, head);

    return std::make_tuple(implication, chain, head, remainingArgsKey, zeroArgName, sName);
}

/// @brief Build the strong-induction hypothesis of a conjecture for one
///        induction variable.
///
/// @details
/// The weak hypothesis (`createAuxyImplication`) is the conjecture at the
/// step block's fixed predecessor `rec`, with every other digit argument
/// fixed. The strong hypothesis is the conjecture at every smaller value:
/// a fresh universal variable `m` replaces the induction variable `n`
/// everywhere, and two leading premises bound it, written with atomic
/// expressions only — `(in3[m,x,n,+])` (some `x` with `m+x=n`) and
/// `!(=[m,n])` — so `m<n` (D-500).
/// Only `n` and the anchor arguments are fixed (`u_` prefix); every other
/// variable, digit arguments included, is universal, because the step uses
/// the hypothesis with changed arguments (G3 applies it at `(b,r)` for the
/// goal `(a,b)`). `n` occurs only in the two bounding premises.
///
/// The fresh names `m` and `x` are the two decimal names after the largest
/// decimal argument of the conjecture, so they collide with no conjecture
/// variable. The anchor is dropped from the chain, as in the weak builder;
/// the remaining-arguments key is formed by the same rule.
///
/// The product form is the strong step's multiplication catcher
/// (D-566): its bound premise is
/// `(in3[x,m,n,*])` (some `x` with `x·m=n`, `m` the second factor), the rest
/// unchanged. In a step block `n` is a successor, so `x·m=n` forces `x≥1`,
/// hence `m≤n`, and with `m≠n` again `m<n`.
///
/// @param expr       The conjecture (an implication whose first premise is the
///                   anchor).
/// @param arg        The induction variable `n` (a digit argument of @p expr).
/// @param anchorName The batch anchor's name, for example `AnchorFTA`.
/// @param product    Build the multiplication catcher instead of the sum form.
/// @return The tuple (hypothesis implication, its premise chain, its head,
///         the remaining-arguments key, the anchor's zero name, the anchor's
///         successor name) — the same shape as `createAuxyImplication`.
/// @invariant The hypothesis names `n` only in its first two premises; its
///            fixed names are exactly `n` and the anchor arguments.
/// @see createAuxyImplication, AnchorScopeQueries::findPlusArgNameFromAnchor,
///      addTheoremToMemory.
std::tuple<
    std::string,
    std::vector<std::string>,
    std::string,
    std::set<std::string>,
    std::string,
    std::string
>
ProofGridBuilder::createStrongAuxyImplication(const std::string& expr,
    const std::string& arg,
    const std::string& anchorName,
    bool product) {
    // Fresh universal names after the conjecture's largest decimal argument.
    long long maxDecimal = 0;
    bool argFound = false;
    for (const std::string& sub : extractSubstringsForAuxy(expr)) {
        for (const std::string& name : ce::getArgs(sub)) {
            if (name == arg) argFound = true;
            if (name.empty()) continue;
            bool decimal = true;
            for (const char c : name) {
                if (c < '0' || c > '9') { decimal = false; break; }
            }
            if (decimal) maxDecimal = std::max(maxDecimal, std::stoll(name));
        }
    }
    assert(argFound && "createStrongAuxyImplication: the induction variable occurs in no expression");
    const std::string boundName = std::to_string(maxDecimal + 1);
    const std::string offsetName = std::to_string(maxDecimal + 2);

    // n -> m in the whole conjecture.
    StrReplacement replPair[1];
    replPair[0].key = StrSpan(arg);
    replPair[0].value = StrSpan(boundName);
    const std::string replExpr = replaceKeysToString(StrSpan(expr), replPair, 1);

    std::set<std::string> untouchables;
    untouchables.insert(arg);
    std::vector<std::string> conjectureChain;
    std::string zeroArgName;
    std::string sName;
    std::string plusName;
    std::string timesName;
    for (const std::string& sub : extractSubstringsForAuxy(replExpr)) {
        if (sub.find(anchorName) != std::string::npos) {
            for (const std::string& anchorArg : ce::getArgs(sub)) untouchables.insert(anchorArg);
            zeroArgName = anchorScopeQueries.findZeroArgNameFromAnchor(sub);
            sName = anchorScopeQueries.findSArgNameFromAnchor(sub);
            plusName = anchorScopeQueries.findPlusArgNameFromAnchor(sub);
            timesName = anchorScopeQueries.findTimesArgNameFromAnchor(sub);
        } else {
            conjectureChain.push_back(sub);
        }
    }
    assert(!plusName.empty() && "createStrongAuxyImplication: the conjecture names no anchor");
    assert(!conjectureChain.empty() && "createStrongAuxyImplication: the conjecture has no head");

    // m<n as two atomic premises, ahead of the conjecture's own premises:
    // m+x=n, or x·m=n for the multiplication catcher.
    std::vector<std::string> chain;
    chain.reserve(conjectureChain.size() + 2);
    chain.push_back(product
        ? "(in3[" + offsetName + "," + boundName + "," + arg + "," + timesName + "])"
        : "(in3[" + boundName + "," + offsetName + "," + arg + "," + plusName + "])");
    chain.push_back("!(=[" + boundName + "," + arg + "])");
    for (const std::string& element : conjectureChain) chain.push_back(element);

    // remaining_args_key: fixed names among the premises (all but the head).
    std::set<std::string> remainingArgsKey;
    for (std::size_t elementIndex = 0; elementIndex + 1 < chain.size(); ++elementIndex) {
        for (const std::string& name : ce::getArgs(chain[elementIndex])) {
            if (untouchables.count(name) != 0) remainingArgsKey.insert(name);
        }
    }

    // Prefix the fixed names with "u_" in every element.
    std::vector<std::string> prefixedValues;
    prefixedValues.reserve(untouchables.size());
    for (const std::string& name : untouchables) prefixedValues.push_back("u_" + name);
    std::vector<StrReplacement> prefixPairs(untouchables.size());
    {
        std::size_t i = 0;
        for (const std::string& name : untouchables) {
            prefixPairs[i].key = StrSpan(name);
            prefixPairs[i].value = StrSpan(prefixedValues[i]);
            ++i;
        }
    }
    for (std::string& element : chain) {
        element = replaceKeysToString(StrSpan(element), prefixPairs.data(),
            static_cast<int32_t>(prefixPairs.size()));
    }

    const std::string head = chain.back();
    chain.pop_back();
    assert(chain.size() >= 2 && "createStrongAuxyImplication: the bounding premises lead the chain");

    const std::string implication = gl::ImplicationBuilder::reconstructImplication(chain, head);
    return std::make_tuple(implication, chain, head, remainingArgsKey, zeroArgName, sName);
}

/// @brief The lemma digits a strong step block's induction demand refuses
///        (D-502).
///
/// @details
/// A digit is a number at a slot whose configured definition set is `(1)`,
/// an element of N: the conjecture's own element variables and the anchor's
/// constants. The demand serves only the lemma's input values (maintainer,
/// 2026-10-08): a digit is allowed when it fills a configured input slot
/// (`CoreExpressionConfig::inputIndices`) of a non-anchor premise or of the
/// head; every other digit is refused. `rec` is never a lemma digit and is
/// refused by `hasOnlyInductionDigits` itself; sets and operators are no
/// digits and pass. A premise or head without a configuration contributes
/// nothing. For G3 (`coprime(a,b) -> bez1(a,b)`) the inputs are a and b,
/// and zero (a parameter of `coprime`), one (a parameter of `bez1`) and two
/// (in the anchor only) are refused.
///
/// @param premises The conjecture's premise chain, the anchor premise
///                 included (compiled form).
/// @param head     The conjecture's head.
/// @param out      Receives the refused digits, ascending.
/// @param cap      Capacity of @p out (asserted, never truncated).
/// @return Number of refused digits written.
/// @invariant Reads configurations only; mints nothing; no heap.
/// @see hasOnlyInductionDigits, addTheoremToMemory,
///      AdmissionMemory::isAdmittedIntegrationDouble.
int32_t ProofGridBuilder::inductionParameterDigits(const std::vector<std::string>& premises,
            const std::string& head, int32_t* out, int32_t cap) const {
    assert(cap >= 0 && (cap == 0 || out != nullptr)
        && "inductionParameterDigits: an output without storage");
    int32_t elements[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
    int32_t elementCount = 0;
    int32_t inputs[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
    int32_t inputCount = 0;
    const auto add = [](int32_t* values, int32_t& count, int32_t value) {
        for (int32_t i = 0; i < count; ++i)
            if (values[i] == value) return;
        assert(count < ExecutionParameters::MAX_ADMISSION_REM_ARGS
            && "inductionParameterDigits: digit count exceeds cap");
        values[count++] = value;
    };
    const int32_t leafCount = static_cast<int32_t>(premises.size()) + 1;
    for (int32_t e = 0; e < leafCount; ++e) {
        const StrSpan leaf = e + 1 < leafCount ? StrSpan(premises[static_cast<std::size_t>(e)]) : StrSpan(head);
        const StrSpan core = extractExpressionUniversalSpan(leaf);
        const ce::CoreExpressionConfig* config = this->definitions.coreConfig(core);
        if (config == nullptr) continue;
        const bool anchor = equalSpans(core, StrSpan(this->anchorInfo.name));
        StrSpan arguments[ExecutionParameters::MAX_ARITY];
        const int32_t count = getArgsSpans(leaf, arguments, ExecutionParameters::MAX_ARITY);
        for (int32_t i = 0; i < count; ++i) {
            int value = 0;
            const bool number = parseDigitRunToInt(arguments[i].ptr, arguments[i].len, value);
            assert(number && "inductionParameterDigits: a compiled lemma names its arguments by numbers");
            (void)number;
            const std::pair<std::string, bool>* definition = nullptr;
            for (const auto& entry : config->definitionSets) {
                int slot = 0;
                const bool decimal = parseDigitRunToInt(entry.first.data(),
                    static_cast<int32_t>(entry.first.size()), slot);
                assert(decimal && "inductionParameterDigits: definition slot key must be decimal");
                (void)decimal;
                if (slot == i + 1) {
                    definition = &entry.second;
                    break;
                }
            }
            assert(definition != nullptr && "inductionParameterDigits: every configured slot has a definition set");
            if (definition->first == "(1)") add(elements, elementCount, value);
            if (anchor) continue;
            for (const int input : config->inputIndices)
                if (input == i) add(inputs, inputCount, value);
        }
    }
    std::sort(elements, elements + elementCount);
    int32_t written = 0;
    for (int32_t i = 0; i < elementCount; ++i) {
        bool input = false;
        for (int32_t k = 0; k < inputCount && !input; ++k) input = inputs[k] == elements[i];
        if (input) continue;
        assert(written < cap && "inductionParameterDigits: refused digits exceed the output capacity");
        out[written++] = elements[i];
    }
    return written;
}

/// @brief Whether a lemma admits a strong triad on one of its induction
///        variables (D-594).
///
/// @details
/// The strong hypothesis states the lemma at every smaller value m with
/// every other variable universal; it serves the step only where what the
/// premises demand of the induction variable carries over to m. Every
/// non-anchor premise that contains @p variable must therefore have one of
/// three shapes:
/// - a property: every other argument is an argument of the anchor premise;
/// - a relation: exactly one other non-anchor argument, itself one of
///   @p inductionVariables;
/// - a set operator: every other non-anchor argument sits at a slot the
///   element's configuration lists as uniquely definable
///   (`CoreExpressionConfig::uniquelyDefinableIndices`) and types as a set
///   of numbers (definition set `P(1)`), as the divisor set of a number.
///
/// In addition no element of the lemma, premises and head, may be an
/// operator whose configured output slot is typed other than `P(1)`.
/// Premises that do not contain @p variable are free; an element without a
/// configuration is neither an operator nor a set operator.
///
/// @param premises               The conjecture's premise chain, the anchor
///                               premise included (compiled form).
/// @param head                   The conjecture's head.
/// @param variable               The induction variable of the strong triad.
/// @param inductionVariables     Every induction variable of the lemma
///                               (`ScopeTypes::findDigitArgs`).
/// @param inductionVariableCount Number of @p inductionVariables.
/// @return True iff the strong triad on @p variable may be built.
/// @invariant Reads configurations only; mints nothing; no heap.
/// @see addTheoremToMemory, inductionParameterDigits,
///      ScopeTypes::findDigitArgs.
bool ProofGridBuilder::strongInductionEligible(const std::vector<std::string>& premises,
            const std::string& head, StrSpan variable,
            const StrSpan* inductionVariables, int32_t inductionVariableCount) const {
    assert(inductionVariableCount >= 0 && (inductionVariableCount == 0 || inductionVariables != nullptr)
        && "strongInductionEligible: induction variables without storage");
    // The definition set a configuration gives the argument at position slot.
    const auto definitionSetIs = [](const ce::CoreExpressionConfig& config, int32_t slot, const char* pattern) {
        for (const auto& entry : config.definitionSets) {
            int key = 0;
            const bool decimal = parseDigitRunToInt(entry.first.data(),
                static_cast<int32_t>(entry.first.size()), key);
            assert(decimal && "strongInductionEligible: definition slot key must be decimal");
            (void)decimal;
            if (key == slot + 1) return entry.second.first == pattern;
        }
        assert(false && "strongInductionEligible: every configured slot has a definition set");
        return false;
    };
    // The arguments of the anchor premise: the values every lemma shares.
    StrSpan anchorArgs[ExecutionParameters::MAX_ARITY];
    int32_t anchorN = 0;
    for (const std::string& premise : premises)
        if (equalSpans(extractExpressionUniversalSpan(StrSpan(premise)), StrSpan(this->anchorInfo.name)))
            anchorN = getArgsSpans(StrSpan(premise), anchorArgs, ExecutionParameters::MAX_ARITY);
    assert(anchorN > 0 && "strongInductionEligible: the lemma has its anchor premise");
    const auto isAnchorArg = [&](StrSpan argument) {
        for (int32_t i = 0; i < anchorN; ++i)
            if (equalSpans(anchorArgs[i], argument)) return true;
        return false;
    };
    const auto isInductionVariable = [&](StrSpan argument) {
        for (int32_t i = 0; i < inductionVariableCount; ++i)
            if (equalSpans(inductionVariables[i], argument)) return true;
        return false;
    };
    const int32_t leafCount = static_cast<int32_t>(premises.size()) + 1;
    for (int32_t e = 0; e < leafCount; ++e) {
        const bool isHead = e + 1 == leafCount;
        const StrSpan leaf = isHead ? StrSpan(head) : StrSpan(premises[static_cast<std::size_t>(e)]);
        const StrSpan core = extractExpressionUniversalSpan(leaf);
        if (equalSpans(core, StrSpan(this->anchorInfo.name))) continue;
        const ce::CoreExpressionConfig* config = this->definitions.coreConfig(core);
        // No operator whose output is not a set of numbers.
        if (config != nullptr)
            for (const int output : config->outputIndices)
                if (!definitionSetIs(*config, output, "P(1)")) return false;
        if (isHead) continue;
        StrSpan arguments[ExecutionParameters::MAX_ARITY];
        const int32_t count = getArgsSpans(leaf, arguments, ExecutionParameters::MAX_ARITY);
        bool holdsVariable = false;
        for (int32_t i = 0; i < count && !holdsVariable; ++i) holdsVariable = equalSpans(arguments[i], variable);
        if (!holdsVariable) continue;
        // The other non-anchor arguments decide the shape.
        StrSpan other;
        int32_t distinctOthers = 0;
        bool setOperator = config != nullptr;
        for (int32_t i = 0; i < count; ++i) {
            if (equalSpans(arguments[i], variable) || isAnchorArg(arguments[i])) continue;
            if (distinctOthers == 0) {
                other = arguments[i];
                distinctOthers = 1;
            } else if (!equalSpans(arguments[i], other)) {
                distinctOthers = 2;
            }
            if (!setOperator) continue;
            bool uniquelyDefined = false;
            for (const int index : config->uniquelyDefinableIndices) uniquelyDefined = uniquelyDefined || index == i;
            setOperator = uniquelyDefined && definitionSetIs(*config, i, "P(1)");
        }
        if (distinctOthers == 0) continue;                                   // property
        if (distinctOthers == 1 && isInductionVariable(other)) continue;     // relation
        if (setOperator) continue;                                           // set operator
        return false;
    }
    return true;
}

/// @brief Whether a lemma's premises stay the same when two of its strong
///        candidates are exchanged (D-595).
///
/// @details
/// The qualification of a lemma with two strong candidates (the induction
/// variables `strongInductionEligible` admits): their strong triads are
/// built only when the premises cannot tell the two apart, as coprime(a, b)
/// cannot; a shot that exchanges them, as Euclid's step sets (a, b) to
/// (b, a mod b), then lands in the premises again. Every non-anchor premise
/// is renamed by exchanging @p first and @p second, and the renamed
/// premises must be the original ones again, counted with multiplicity.
/// A renamed premise matches an original when both agree in negation, core
/// and every argument, except that a relation its configuration flags
/// symmetric (`CoreExpressionConfig::symmetric`) matches with its two
/// inputs in either order. The head is not read. Premises are flat
/// elements, as `strongInductionEligible` reads them.
///
/// @param premises The conjecture's premise chain, the anchor premise
///                 included (compiled form).
/// @param first    One strong candidate.
/// @param second   The other strong candidate, distinct from @p first.
/// @return True iff the premises are symmetric in @p first and @p second.
/// @invariant Reads configurations only; mints nothing; no heap.
/// @see strongInductionEligible, addTheoremToMemory.
bool ProofGridBuilder::strongPremiseSymmetric(const std::vector<std::string>& premises,
            StrSpan first, StrSpan second) const {
    assert(!equalSpans(first, second) && "strongPremiseSymmetric: two distinct candidates");
    constexpr int32_t kMaxPremises = 64;
    StrSpan own[kMaxPremises];
    int32_t ownN = 0;
    for (const std::string& premise : premises) {
        if (equalSpans(extractExpressionUniversalSpan(StrSpan(premise)), StrSpan(this->anchorInfo.name))) continue;
        assert(ownN < kMaxPremises && "strongPremiseSymmetric: premise chain exceeds 64");
        own[ownN++] = StrSpan(premise);
    }
    // Whether premise a, its candidates exchanged, is premise b.
    const auto renamedMatches = [&](StrSpan a, StrSpan b) {
        if ((a.len > 0 && a.ptr[0] == '!') != (b.len > 0 && b.ptr[0] == '!')) return false;
        const StrSpan core = extractExpressionUniversalSpan(a);
        if (!equalSpans(core, extractExpressionUniversalSpan(b))) return false;
        StrSpan renamed[ExecutionParameters::MAX_ARITY];
        StrSpan target[ExecutionParameters::MAX_ARITY];
        const int32_t n = getArgsSpans(a, renamed, ExecutionParameters::MAX_ARITY);
        if (n != getArgsSpans(b, target, ExecutionParameters::MAX_ARITY)) return false;
        for (int32_t i = 0; i < n; ++i) {
            if (equalSpans(renamed[i], first)) renamed[i] = second;
            else if (equalSpans(renamed[i], second)) renamed[i] = first;
        }
        const ce::CoreExpressionConfig* config = this->definitions.coreConfig(core);
        if (config != nullptr && config->symmetric) {
            assert(config->inputIndices.size() == 2 && "strongPremiseSymmetric: a symmetric relation has two inputs");
            const int32_t x = config->inputIndices[0];
            const int32_t y = config->inputIndices[1];
            assert(x < n && y < n && "strongPremiseSymmetric: relation inputs inside the argument list");
            if (compareSpans(renamed[x], renamed[y]) > 0) std::swap(renamed[x], renamed[y]);
            if (compareSpans(target[x], target[y]) > 0) std::swap(target[x], target[y]);
        }
        for (int32_t i = 0; i < n; ++i)
            if (!equalSpans(renamed[i], target[i])) return false;
        return true;
    };
    bool matched[kMaxPremises] = {};
    for (int32_t i = 0; i < ownN; ++i) {
        bool found = false;
        for (int32_t j = 0; j < ownN && !found; ++j) {
            if (matched[j] || !renamedMatches(own[i], own[j])) continue;
            matched[j] = true;
            found = true;
        }
        if (!found) return false;
    }
    return true;
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)

// Member of gl::Prover
void ProofGridBuilder::addTheoremToMemory(const std::string& expr,
                                            Memory& memory,
                                            int iteration,
                                            bool proved,
                                            Dependencies& dependencyTable) {
    // Initial compound deposits can install many rules before the next reader.
    // Use the same staging window as premise seeding in the compressor.
    const auto seedExpression = [&](auto&& expression, Memory& block, auto&&... args) {
        ++block.ruleStagingArmDepth;
        statementMemory.addExprToMemoryBlock(std::forward<decltype(expression)>(expression),
            block, std::forward<decltype(args)>(args)...);
        --block.ruleStagingArmDepth;
        if (block.ruleStagingArmDepth == 0) gl::HashRuleIndex::closeRuleIndexStaging(block);
    };
    // The dead-end variable-copy axiom (=[v,v_copy]) at a premise block
    // (variableCopy origin, no dependencies), shared by the output-collision,
    // antisymmetry and defined-set triggers (D-265, D-235,
    // D-599). LB-creation paired mailOut write — the
    // same timing exception as the premise deposit: buildGrid's startup
    // commit needs mailOut populated so every descendant receives the axiom
    // and its history line on its first phase-1 pull. Family sibling rows
    // sharing the block re-run the deposit; the door dedups.
    const auto depositVariableCopy = [&](Memory& block, const std::string& variable) {
        const std::string copyEquality = "(=[" + variable + "," + variable + "_copy])";
        const TransientOrigin copyOrigin{ true, OriginTag::variableCopy, nullptr, 0 };
        const int lvRunCopy[1] = { block.level };
        seedExpression(StrSpan(copyEquality), block, iteration, 0, lvRunCopy, 1,
            copyOrigin, -1, -1, StrSpan("main", 4), false);
        if (parameters.trackHistory) {
            const int maxOriginsLocal = parameters.compressor_mode
                ? parameters.compressor_max_origins_per_expr
                : parameters.max_origin_per_expr;
            ExpressionWithValidity ev(copyEquality, "main");
            block.insertMailOutStatement(StrSpan(ev.original), StrSpan(ev.validityName), lvRunCopy, 1);
            block.addMailOutOrigin(StrSpan(ev.original), StrSpan(ev.validityName), copyOrigin.tag,
                copyOrigin.deps, copyOrigin.depN, maxOriginsLocal);
        }
    };
    // Initial registration accepts conjectures only. Supplied theorems enter
    // through broadcast and mail absorption as already available rule fuel.
    assert(!proved);
    (void)proved;

    // 1) Disintegrate implication -> chain (left nodes) + head (rightmost)
    std::vector< std::tuple<
        std::string,                  // left expression
        std::vector<std::string>,     // node args
        std::set<std::string>         // left-node arguments
    > > tempChain;
    const std::string head = ce::disintegrateImplication(expr, tempChain, this->definitions.configurations());

    std::vector<std::string> chain;
    chain.reserve(tempChain.size());
    for (std::size_t i = 0; i < tempChain.size(); ++i) chain.push_back(std::get<0>(tempChain[i]));

    // A set- or relation-typed `=` head integrates by extensionality
    // (I-267): compile its entity on this single-threaded
    // seam, before the head is seeded as a goal below.
    {
        StrSpan leaves[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1];
        int32_t leafN = 0;
        for (const std::string& premise : chain) {
            assert(leafN < ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                && "addTheoremToMemory: premise chain exceeds MAX_ADMISSION_KEY_ELEMENTS");
            leaves[leafN++] = StrSpan(premise);
        }
        leaves[leafN++] = StrSpan(head);
        scopeTypes.ensureTypedEqualityEntityForTheorem(leaves, leafN);
    }

    // Output-collision conjecture: two same-operator premises fuse their
    // result binders into one shared output name (the MPL encoding of an
    // equality between compound terms). Detect before the walk; the
    // deposit lands during the walk at the SECOND colliding premise LB —
    // mail flows only ancestor-to-descendant, and every consumer of the
    // copy (deeper premise LBs, contradiction twins under the innermost
    // premise, recursion auxiliaries) sits at or below it, while foreign
    // conjectures sharing only the first colliding premise as a chain
    // prefix never see it.
    std::string collisionCopyVar;
    std::size_t collisionDepositIndex = 0;
    const bool haveOutputCollision = gl::VariableCopyDetector::detectOutputCollisionCopyVar(
        chain, this->definitions.configurations(), collisionCopyVar, collisionDepositIndex);

    // 2) Walk/build the memory path for the chain
    Memory* memoryBlock = &memory;
    bool markedGoalPending = false;
    for (std::size_t index = 0; index < chain.size(); ++index) {
        const std::string& element = chain[index];
        bool isNewBody = false;

        Memory* existingChild = simpleMapStore.findChild(memoryBlock, element);
        if (existingChild != NULL) {
            // A shared chain prefix: the record is a function of the key chain
            // (I-267), so the rebuilt record must agree.
            assert(scopeTypes.lbVarDefinitionSetsMatch(*existingChild)
                && "addTheoremToMemory: shared LB's variable definition sets changed");
            memoryBlock = existingChild;
        } else {
            Memory* child = lbStore.create<Memory>();
            simpleMapStore.linkChild(memoryBlock, element, child);

            child->parentMemory = memoryBlock;
            child->level = static_cast<int>(index);
            child->setExprKey(element);
            scopeTypes.assignLbVarDefinitionSets(*child);

            this->permanentBodies.push_back(child);

            memoryBlock = child;
            isNewBody = true;

            const int lvRun[1] = { child->level };
            const TransientOrigin origin{ true, OriginTag::taskFormulation, nullptr, 0 };
            // A conjecture's last non-anchor premise is seeded into its goal
            // block with the goal pending (D-495).
            // Every non-anchor premise carries the conjecture-premise
            // permission: the ORs it unfolds into open their case split where
            // its block has proof work, and travel to every descendant with
            // the unconditional-ordis mail flag
            // (D-528). The anchor block is shared
            // by every conjecture and takes neither the mark nor the
            // permission.
            const bool nonAnchorPremise =
                !equalSpans(extractExpressionUniversalSpan(StrSpan(element)),
                            StrSpan(this->anchorInfo.name));
            const bool goalBlockPremise = index == chain.size() - 1 && nonAnchorPremise;
            if (goalBlockPremise) {
                assert(!child->goalPending
                    && "addTheoremToMemory: a fresh block has no pending goal");
                child->goalPending = true;
                markedGoalPending = true;
            }
            seedExpression(element, *child, iteration, 0, lvRun, 1, origin, -1, -1, StrSpan("main", 4), false,
                /*allowOrDisintegration=*/false, /*registeredOut=*/nullptr,
                /*recursionHead=*/false, /*convergenceMailOut=*/nullptr,
                /*replayKinds=*/0, /*replayIntegrationDepth=*/-1,
                /*integrationPremise=*/false, /*conjecturePremise=*/nonAnchorPremise);

            // LB-creation paired mailOut write — documented exception to the
            // "fillMailOut sole writer" invariant. fillMailOut runs only
            // during prover steps (step 1's pre-burst standardProcessing);
            // buildGrid()'s startup commit needs mailOut already populated so
            // the LB's task-formulation entry lands in its log and reaches every
            // descendant on that descendant's first phase-1 pull. Without this
            // pair, a descendant's exprOriginMap would miss the dep history at
            // its step 1.
            if (parameters.trackHistory) {
                const int maxOriginsLocal = parameters.compressor_mode
                    ? parameters.compressor_max_origins_per_expr
                    : parameters.max_origin_per_expr;
                ExpressionWithValidity ev(element, "main");
                child->insertMailOutStatement(StrSpan(ev.original),
                    StrSpan(ev.validityName), lvRun, 1);
                child->addMailOutOrigin(StrSpan(ev.original),
                    StrSpan(ev.validityName), origin.tag, origin.deps,
                    origin.depN, maxOriginsLocal);
            }
        }

        // Output-collision copy axiom (=[r,r_copy]) at the second
        // colliding premise LB.
        if (haveOutputCollision && index == collisionDepositIndex)
            depositVariableCopy(*memoryBlock, collisionCopyVar);

        // 3) At the last node: set up recursion/auxiliaries.
        // (D-45: the historical proved=true branch was dead — single caller
        // always passes proved=false. Guarded by assert at function entry.)
        if (index == chain.size() - 1) {
            {
                // digits / immutables. findDigitArgs is now 0% heap (sorted-unique
                // StrSpan run); materialize the std::set in this still-heap caller.
                StrSpan digitBuf[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                const int32_t digitN = scopeTypes.findDigitArgs(expr, this->anchorInfo, digitBuf,
                                                     ExecutionParameters::MAX_ADMISSION_REM_ARGS);
                std::set<std::string> digits;
                for (int32_t i = 0; i < digitN; ++i)
                    digits.emplace(digitBuf[i].ptr, static_cast<std::size_t>(digitBuf[i].len));
                // Twin premises: the induction runs on the first twin's
                // variable only, and its weak hypothesis leaves the partner
                // universal (D-587).
                const std::map<std::string, std::string> twinPartners =
                    twinInductionPartners(expr, digitBuf, digitN);
                std::set<std::string> partnerDigits;
                for (const auto& twin : twinPartners) {
                    partnerDigits.insert(twin.second);
                    std::ostringstream line;
                    line << "[twin induction] induction=" << twin.first
                         << " partner=" << twin.second
                         << " conjecture=" << expr << '\n';
                    diagnosticsLog() << line.str() << std::flush;
                }
                // The strong candidates: the induction variables whose
                // premises carry over to a smaller value
                // (D-594). A single candidate keeps
                // its strong triad; two keep theirs only when the premises
                // are symmetric in them; three or more keep none
                // (D-595).
                std::set<std::string> strongCandidates;
                if (parameters.strong_induction)
                    for (const std::string& digit : digits)
                        if (partnerDigits.count(digit) == 0
                            && strongInductionEligible(chain, head, StrSpan(digit), digitBuf, digitN))
                            strongCandidates.insert(digit);
                const bool strongCandidatesQualify = strongCandidates.size() == 1
                    || (strongCandidates.size() == 2
                        && strongPremiseSymmetric(chain, StrSpan(*strongCandidates.begin()),
                            StrSpan(*strongCandidates.rbegin())));
                // One triad per digit argument and enabled induction kind,
                // weak before strong (D-500). Both
                // kinds share the step and zero block shapes; they differ
                // only in the hypothesis the step block installs.
                for (std::set<std::string>::const_iterator dit = digits.begin(); dit != digits.end(); ++dit)
                for (int inductionKind = 0; inductionKind < 2; ++inductionKind) {
                    const bool strongInduction = inductionKind == 1;
                    if (strongInduction ? !parameters.strong_induction : !parameters.weak_induction) continue;
                    if (partnerDigits.count(*dit) != 0) continue;
                    if (strongInduction && strongCandidates.count(*dit) == 0) {
                        std::ostringstream line;
                        line << "[strong induction] skipped variable=" << *dit
                             << " conjecture=" << expr << '\n';
                        diagnosticsLog() << line.str() << std::flush;
                        continue;
                    }
                    if (strongInduction && !strongCandidatesQualify) {
                        std::ostringstream line;
                        line << "[strong induction] refused variable=" << *dit
                             << " candidates=" << strongCandidates.size()
                             << " conjecture=" << expr << '\n';
                        diagnosticsLog() << line.str() << std::flush;
                        continue;
                    }
                    const std::string digitArg = *dit;
                    const std::map<std::string, std::string>::const_iterator twin = twinPartners.find(digitArg);
                    const std::string partner = twin == twinPartners.end() ? std::string() : twin->second;
                    std::set<std::string> freeDigits;
                    if (!partner.empty()) freeDigits.insert(partner);

                    // recX label for equality block, generic 'rec' for the auxy implication chain
                    const std::string recArg = std::string("rec") + std::to_string(memoryBlock->recursionCounter);


                    std::string auxyImplication;
                    std::vector<std::string> auxyChain;
                    std::string auxyHead;
                    std::set<std::string> remainingArgsKey;
                    std::string zeroArgName;
                    std::string sName;

                    // Weak: createAuxyImplication(expr, digitArg, "rec", immutables, anchorName).
                    // Strong: every smaller value of digitArg, other variables universal.
                    {
                        std::tuple<
                            std::string,
                            std::vector<std::string>,
                            std::string,
                            std::set<std::string>,
                            std::string,
                            std::string
                        > tpl = strongInduction
                            ? createStrongAuxyImplication(expr, digitArg, this->anchorInfo.name)
                            : createAuxyImplication(expr, digitArg, "rec",
                                  this->findImmutableArgs(expr, digitArg, freeDigits), this->anchorInfo.name);

                        auxyImplication  = std::get<0>(tpl);
                        auxyChain        = std::get<1>(tpl);
                        auxyHead         = std::get<2>(tpl);
                        remainingArgsKey = std::get<3>(tpl);
                        zeroArgName      = std::get<4>(tpl);
                        sName            = std::get<5>(tpl);
                    }

                    // --- Recursion block #1 ----------------------------------------------------
                    const std::string tempExpr  = std::string("(in2[") + recArg + "," + digitArg + "," + sName + "])";
                    const std::string tempExpr2 = std::string("(in2[") + "rec"   + "," + digitArg + "," + sName + "])";



                    // ensure unique
                    if (simpleMapStore.findChild(memoryBlock, tempExpr) == nullptr) {
                        Memory* tempMb = lbStore.create<Memory>();
                        simpleMapStore.linkChild(memoryBlock, tempExpr, tempMb);

                        tempMb->level = static_cast<int>(index + 1);
                        tempMb->parentMemory = memoryBlock;
                        tempMb->setExprKey(tempExpr);
                        scopeTypes.assignLbVarDefinitionSets(*tempMb);
                        tempMb->recursionHypothesisId =
                            tempMb->nameMap.encode(tempExpr2);
                        tempMb->isPartOfRecursion = true;
                        // Before the install: its integration keys are held
                        // to the induction already
                        // (D-583,
                        // D-584).
                        tempMb->weakStepBlock = !strongInduction;
                        tempMb->strongStepBlock = strongInduction;
                        assert(!zeroArgName.empty() && "step block: the anchor names the induction's zero");
                        tempMb->inductionZeroId = tempMb->nameMap.encode(zeroArgName);
                        // The weak hypothesis leaves the partner universal
                        // (D-587); the install
                        // arms the zero case of the partner's
                        // zero-or-successor OR
                        // (D-588).
                        if (!strongInduction && !partner.empty()) {
                            tempMb->twinPartnerId = tempMb->nameMap.encode(partner);
                            tempMb->twinZeroId = tempMb->nameMap.encode(zeroArgName);
                            tempMb->twinSuccessorId = tempMb->nameMap.encode(sName);
                        }


                        // history tag for auxy implication
                        // L3 span-record door. "recursion" has empty antecedents
                        // (P1); the KEY `auxyImplication` is a stable std::string
                        // local, "main" a static literal.
                        addOriginEncoded(tempMb->exprOriginMap, tempMb->originInterner, StrSpan(auxyImplication), StrSpan("main", 4), OriginTag::recursion, nullptr, 0, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));

                        this->permanentBodies.push_back(tempMb);

                        const int lvRun[1] = { tempMb->level };

                        StrSpan auxyChainRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                        int32_t auxyChainRunN = 0;
                        for (const std::string& s : auxyChain) {
                            assert(auxyChainRunN < ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                                && "addToHashMemory auxyChain run exceeds cap");
                            auxyChainRun[auxyChainRunN++] = StrSpan(s);
                        }
                        StrSpan auxyRemRun[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                        int32_t auxyRemRunN = 0;
                        for (const std::string& s : remainingArgsKey) {
                            assert(auxyRemRunN < ExecutionParameters::MAX_ADMISSION_REM_ARGS
                                && "addToHashMemory auxyRem run exceeds cap");
                            auxyRemRun[auxyRemRunN++] = StrSpan(s);
                        }
                        // The hypothesis's own chain: recursion-owned and
                        // "admitted by induction hypothesis", the one flag
                        // that opens induction scopes
                        // (D-519).
                        hashRuleInstaller.addToHashMemory(auxyChainRun, auxyChainRunN, StrSpan(auxyHead),
                            auxyRemRun, auxyRemRunN,
                            *tempMb, tempMb->overallHashMemory, lvRun, 1,
                            StrSpan(auxyImplication),
                            parameters.inductionMaxAdmissionDepth,
                            parameters.inductionMaxSecondaryNumber,
                            static_cast<uint8_t>(kAdmissionRecursionFlag | kAdmissionInductionHypothesisFlag),
                            parameters.minNumOperatorsKey,
                            StrSpan("implication", 11), true, StrSpan(auxyImplication), StrSpan("main", 4),
                            RuleIndexOp{}, nullptr, false, !strongInduction);

                        // The installed induction implication's ruleInterner id
                        // — the firing site marks this rule's fired head as the
                        // induction hypothesis (FiringRecord::recursionHead).
                        // Non-minting: addToHashMemory minted it just above.
                        tempMb->recursionAuxyImplId =
                            tempMb->ruleInterner.lookup(StrSpan(auxyImplication));
                        assert(tempMb->recursionAuxyImplId != 0
                            && "recursion block #1: auxy implication not in ruleInterner after install");
                        // At main nothing is eaten yet: the hypothesis's premises
                        // with only fixed arguments are its missing expressions
                        // there, each a reason to open an OR it is a disjunct of
                        // (D-575).
                        admissionMemory.fileHypothesisDemandsAtMain(*tempMb, auxyChainRun, auxyChainRunN,
                            tempMb->recursionAuxyImplId);

                        // add (in2[rec,...]) to temp block
                        {
                            const int lvRun2[1] = { tempMb->level };
							const TransientOrigin or1{ true, OriginTag::recursion, nullptr, 0 };
                            seedExpression(tempExpr2, *tempMb, iteration, 0, lvRun2, 1, or1, -1, -1, StrSpan("main", 4), false);
                        }

                        // queue head as "to be proved" with auxy index
                        seedExpression(head,
                                                   *tempMb,
                                                   iteration,
                                                   2,
                                                   nullptr, 0,
                                                   TransientOrigin{},
							                       -1,
                                                   dependencyTable.auxyIndex,
                                                   StrSpan("main", 4),
                                                   false);

                        // A strong step block demands a value below n: the
                        // hypothesis's bound premise in3[m,x,n,+] (m and x open,
                        // n and + fixed; atomics only) is filed as a
                        // double-marker integration demand, so an existence
                        // offering a witness r with r+x=n (G3: D1's division of
                        // a by n) can be admitted and the hypothesis fires at r
                        // (D-500, D-492). It is the
                        // block's induction demand: it admits only for an
                        // original whose arguments are the lemma's input
                        // values, sets and operators; the block records the
                        // lemma digits it refuses
                        // (D-502).
                        if (strongInduction) {
                            assert(!auxyChain.empty() && startsWithSpan(StrSpan(auxyChain[0]), "(in3[", 5)
                                && "strong step block: the hypothesis does not open with its bound premise");
                            const bool filed = admissionMemory.updateAdmissionMapIntegrationDouble(*tempMb,
                                StrSpan(auxyChain[0]), StrSpan("main", 4), InductionDemand::sum);
                            assert(filed && "strong step block: its bound premise files no double-marker demand");
                            tempMb->inductionParameterDigitCount = inductionParameterDigits(chain, head,
                                tempMb->inductionParameterDigits, ExecutionParameters::MAX_ARITY);
                            // The lemma's induction variables: the demand admits an
                            // original only when its inputs are among them
                            // (D-601).
                            assert(digitN <= ExecutionParameters::MAX_ARITY
                                && "strong step block: more induction variables than its record holds");
                            for (int32_t v = 0; v < digitN; ++v) {
                                int value = 0;
                                const bool decimal = parseDigitRunToInt(digitBuf[v].ptr, digitBuf[v].len, value);
                                assert(decimal && "strong step block: an induction variable is a number");
                                (void)decimal;
                                tempMb->inductionVariableDigits[tempMb->inductionVariableDigitCount++] = value;
                            }
                            // The chain every induction scope this demand
                            // opens carries in its name, with the demand's two
                            // markers (D-519).
                            const unsigned chainSlot = gl::WorkerContext::coreId() >= 0
                                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                                : scratchArenas().slotCount() - 1;
                            ScratchArena& chainStrings = scratchArenas().forSlot(chainSlot);
                            ScratchScope chainFrame(chainStrings);
                            const ScratchString markedChain = gl::ScopeSyntax::inductionDoubleChainScratch(
                                chainStrings, auxyChainRun, auxyChainRunN);
                            tempMb->inductionDoubleChainId = tempMb->nameMap.encode(StrSpan(markedChain));

                            // The multiplication catcher
                            // (D-566): x·m = n,
                            // m ≠ n and the lemma's premises at m give the
                            // lemma at m. Installed like the sum form, it is
                            // the block's second induction implication and
                            // files the block's second induction demand.
                            const auto productTuple = createStrongAuxyImplication(expr, digitArg,
                                this->anchorInfo.name, true);
                            const std::string& productImplication = std::get<0>(productTuple);
                            const std::vector<std::string>& productChain = std::get<1>(productTuple);
                            const std::string& productHead = std::get<2>(productTuple);
                            const std::set<std::string>& productRemaining = std::get<3>(productTuple);
                            addOriginEncoded(tempMb->exprOriginMap, tempMb->originInterner, StrSpan(productImplication),
                                StrSpan("main", 4), OriginTag::recursion, nullptr, 0,
                                (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                            StrSpan productChainRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                            int32_t productChainRunN = 0;
                            for (const std::string& s : productChain) {
                                assert(productChainRunN < ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                                    && "addToHashMemory product chain run exceeds cap");
                                productChainRun[productChainRunN++] = StrSpan(s);
                            }
                            StrSpan productRemRun[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                            int32_t productRemRunN = 0;
                            for (const std::string& s : productRemaining) {
                                assert(productRemRunN < ExecutionParameters::MAX_ADMISSION_REM_ARGS
                                    && "addToHashMemory product remaining run exceeds cap");
                                productRemRun[productRemRunN++] = StrSpan(s);
                            }
                            hashRuleInstaller.addToHashMemory(productChainRun, productChainRunN, StrSpan(productHead),
                                productRemRun, productRemRunN,
                                *tempMb, tempMb->overallHashMemory, lvRun, 1,
                                StrSpan(productImplication),
                                parameters.inductionMaxAdmissionDepth,
                                parameters.inductionMaxSecondaryNumber,
                                static_cast<uint8_t>(kAdmissionRecursionFlag | kAdmissionInductionHypothesisFlag),
                                parameters.minNumOperatorsKey,
                                StrSpan("implication", 11), true, StrSpan(productImplication), StrSpan("main", 4));
                            tempMb->recursionProductImplId = tempMb->ruleInterner.lookup(StrSpan(productImplication));
                            assert(tempMb->recursionProductImplId != 0
                                && tempMb->recursionProductImplId != tempMb->recursionAuxyImplId
                                && "strong step block: the multiplication catcher is a rule of its own");
                            admissionMemory.fileHypothesisDemandsAtMain(*tempMb, productChainRun, productChainRunN,
                                tempMb->recursionProductImplId);
                            const bool productFiled = admissionMemory.updateAdmissionMapIntegrationDouble(*tempMb,
                                StrSpan(productChain[0]), StrSpan("main", 4), InductionDemand::product);
                            assert(productFiled
                                && "strong step block: the multiplication catcher files no double-marker demand");
                            const ScratchString productMarkedChain = gl::ScopeSyntax::inductionDoubleChainScratch(
                                chainStrings, productChainRun, productChainRunN);
                            tempMb->inductionProductChainId = tempMb->nameMap.encode(StrSpan(productMarkedChain));
                        }

                        // A weak step whose hypothesis head can only repeat the
                        // conjecture premise with its operator is tested: the
                        // hypothesis premises and head are planted at
                        // main_boundary_is_useful, where main's rules fire on
                        // them; a caught step is switched off at a barrier
                        // (D-521).
                        if (!strongInduction) {
                            InductionUsefulness::Test usefulnessTest;
                            if (inductionUsefulness.qualify(chain, this->anchorInfo.name, auxyChain, auxyHead,
                                    tempMb->level, usefulnessTest)) {
                                const int lvRunTest[1] = { tempMb->level };
                                const TransientOrigin testOrigin{ true, OriginTag::recursion, nullptr, 0 };
                                const StrSpan testScope(gl::ScopeSyntax::kUsefulnessScopeName,
                                    static_cast<int32_t>(sizeof(gl::ScopeSyntax::kUsefulnessScopeName) - 1));
                                for (const std::string& premise : usefulnessTest.premises)
                                    seedExpression(StrSpan(premise), *tempMb, iteration, 0, lvRunTest, 1,
                                        testOrigin, -1, -1, testScope, true);
                                seedExpression(StrSpan(usefulnessTest.head), *tempMb, iteration, 0, lvRunTest, 1,
                                    testOrigin, -1, -1, testScope, true);
                                InductionUsefulness::arm(*tempMb, usefulnessTest);
                                std::ostringstream line;
                                line << "[STEP-USEFULNESS] variable=" << digitArg
                                     << " step=" << tempMb->exprKey()
                                     << " head=" << usefulnessTest.head
                                     << " key=" << usefulnessTest.key
                                     << " conjecture=" << expr << '\n';
                                diagnosticsLog() << line.str() << std::flush;
                            }
                        }

                        // update dependency table (original entry collects this auxy)
                        dependencyTable.originalAuxyMap[dependencyTable.originalIndex] = DependencyItem();
                        dependencyTable.originalAuxyMap[dependencyTable.originalIndex].auxies.insert(dependencyTable.auxyIndex);
                        dependencyTable.originalAuxyMap[dependencyTable.originalIndex].expr = expr;
                        dependencyTable.auxyOriginalMap[dependencyTable.auxyIndex] = dependencyTable.originalIndex;
                        // The step-win seam resolves a closed step block to its
                        // triad through this record
                        // (D-517).
                        dependencyTable.stepBlockOriginal[tempMb] = dependencyTable.originalIndex;
                        dependencyTable.auxyIndex += 1;
                    }

                    // --- Recursion block #2 (equality side-chain) -----------------------------
                    const std::string tempExpr3 = std::string("(=[s(") + recArg + ")," + zeroArgName + "])";
                    const std::string tempExpr4 = std::string("(=[") + digitArg + "," + zeroArgName + "])";

                    if (simpleMapStore.findChild(memoryBlock, tempExpr3) == nullptr) {
                        Memory* tempMb2 = lbStore.create<Memory>();
                        simpleMapStore.linkChild(memoryBlock, tempExpr3, tempMb2);

                        tempMb2->level = static_cast<int>(index + 1);
                        tempMb2->parentMemory = memoryBlock;
                        // _induction_<recN>_ marker: recursion-block-#2
                        // instances across DIFFERENT induction triads share
                        // the same substituted equality (distinct routing
                        // keys (=[s(recN),zero]), identical exprKey after
                        // substitution — Rule 12), making coexisting LBs
                        // chain-identical; the full chain is the LB identity
                        // and the deload file identity. The suffix carries
                        // the per-triad recArg so every instance is disjoint.
                        // Parsers are unaffected (getArgs reads only the
                        // first [...] window); the routing key
                        // (tempExpr3) and the recursion hypothesis keep the
                        // unmarked form.
                        tempMb2->setExprKey(
                            tempExpr4 + "_induction_" + recArg + "_");
                        scopeTypes.assignLbVarDefinitionSets(*tempMb2);
                        tempMb2->recursionHypothesisId =
                            tempMb2->nameMap.encode(tempExpr4);
                        tempMb2->isPartOfRecursion = true;
                        // A zero block releases the successor case of the
                        // partner's zero-or-successor OR
                        // (D-588).
                        if (!partner.empty()) {
                            tempMb2->twinPartnerId = tempMb2->nameMap.encode(partner);
                            tempMb2->twinZeroId = tempMb2->nameMap.encode(zeroArgName);
                            tempMb2->twinSuccessorId = tempMb2->nameMap.encode(sName);
                            tempMb2->twinExperiment = TwinExperiment::successorCase;
                        }

                        // Register toBeProved (status=2) BEFORE adding expression (status=0).
                        // When head == tempExpr4 (e.g. companion theorem where head is (=[7,2])),
                        // the status=0 add puts it in intKnownStatements, which would cause
                        // the status=2 call to return early without creating the toBeProved entry.
                        seedExpression(head,
                                                   *tempMb2,
                                                   iteration,
                                                   2,
                                                   nullptr, 0,
                                                   TransientOrigin{},
                                                   -1,
                                                   dependencyTable.auxyIndex,
                                                   StrSpan("main", 4),
                                                   false);

                        {
                            const int lvRun4[1] = { tempMb2->level };
							const TransientOrigin or2{ true, OriginTag::recursion, nullptr, 0 };
                            seedExpression(tempExpr4, *tempMb2, iteration, 0, lvRun4, 1, or2, -1, -1, StrSpan("main", 4), false);
                        }

                        tempMb2->isActive = false;

                        this->permanentBodies.push_back(tempMb2);

                        dependencyTable.auxyOriginalMap[dependencyTable.auxyIndex] = dependencyTable.originalIndex;
                        dependencyTable.originalAuxyMap[dependencyTable.originalIndex].auxies.insert(dependencyTable.auxyIndex);
                        dependencyTable.originalInductionVariableMap[dependencyTable.originalIndex] = std::make_tuple(digitArg, std::to_string(memoryBlock->recursionCounter));
                        if (strongInduction)
                            dependencyTable.strongInductionOriginals.insert(dependencyTable.originalIndex);
                        dependencyTable.auxyIndex += 1;
                        dependencyTable.originalIndex += 1;
                        memoryBlock->recursionCounter += 1;
                    }
                }

                // finally mark head as "to be proved" at the original block too
                const StatementFlags* headRow = lookupStatementFlags(
                    memoryBlock->intKnownStatements, memoryBlock->nameMap, head, "main");
                if (headRow != nullptr) {
                    // Head already derived (e.g. from anchor disintegration) — register directly
                    std::string fullTheorem = gl::ImplicationBuilder::reconstructImplication(chain, head);
                    if (theoremPublisher.appendGlobalTheorem(fullTheorem, "single chapter proof", "-1",
                            "-1", memoryBlock)) {
                        std::cout << fullTheorem << std::endl;
                    }
                } else {
                    seedExpression(head, *memoryBlock, iteration, 2, nullptr, 0, TransientOrigin{}, -1, -1, StrSpan("main", 4), false);
                }
                // The goal step is done: the goal is registered, or the
                // theorem was published directly.
                if (markedGoalPending) {
                    assert(memoryBlock->goalPending
                        && "addTheoremToMemory: the goal block's mark survives until its goal step");
                    memoryBlock->goalPending = false;
                }
            }

            // Antisymmetry-shaped conjecture: the variable-copy axiom
            // (=[X,X_copy]) at this innermost LB, so the equivalence class
            // can generate the all-distinct statement variants the exact
            // request match needs; the induction triads above and the
            // contradiction LBs below receive it by mail (D-235).
            {
                std::string copyVar;
                if (gl::VariableCopyDetector::detectAntisymmetryCopyVar(chain, head, copyVar))
                    depositVariableCopy(*memoryBlock, copyVar);
            }
            // Defined-set conjecture: the head is a property of a set D that
            // a premise defines from one number n, so the proof about D's
            // members needs a second name of n; the copy axiom (=[n,n_copy])
            // at this innermost LB supplies it
            // (D-599).
            {
                std::string copyVar;
                if (gl::VariableCopyDetector::detectDefinedSetCopyVar(chain, head,
                        this->definitions.configurations(), copyVar))
                    depositVariableCopy(*memoryBlock, copyVar);
            }

            // --- Incubator: create contradiction LB ---
            if (parameters.try_contradiction && !proved) {
                // Trap: reformulated operator conjecture?
                // head = =[x,a], one non-anchor operator premise, equality concerns output var.
                // Create __contradiction__(op[a]) under the ANCHOR (predecessor), seeded with
                // clean operator expression. Any contradiction in that LB -> !(op[a]).
                bool reformulatedOperator = false;
                if (parameters.incubator_mode) {
                    std::string eqExpr = ce::extractExpression(head);
                    if (eqExpr == "=" && chain.size() >= 2) {
                        const std::string& lastPremise = chain.back();
                        std::string opName = ce::extractExpression(lastPremise);
                        const ce::CoreExpressionConfig* cfg = this->definitions.coreConfig(StrSpan(opName));
                        if (cfg != nullptr && !cfg->outputIndices.empty()) {
                            int outIdx = cfg->outputIndices[0];
                            std::vector<std::string> opArgs = ce::getArgs(lastPremise);
                            std::vector<std::string> eqArgs = ce::getArgs(head);
                            if (eqArgs.size() == 2 && outIdx < static_cast<int>(opArgs.size())) {
                                std::string x = opArgs[outIdx];
                                std::string a_out;
                                if (eqArgs[0] == x) a_out = eqArgs[1];
                                else if (eqArgs[1] == x) a_out = eqArgs[0];

                                if (!a_out.empty()) {
                                    // Back-reformulate: clean operator with original output var
                                    opArgs[outIdx] = a_out;
                                    std::string cleanOp = "(" + opName + "[";
                                    for (size_t ci = 0; ci < opArgs.size(); ++ci) {
                                        if (ci > 0) cleanOp += ",";
                                        cleanOp += opArgs[ci];
                                    }
                                    cleanOp += "])";

                                    std::vector<std::string> outerChain(chain.begin(), chain.end() - 1);
                                    std::string negCleanOp = "!" + cleanOp;
                                    std::string contrTheorem = gl::ImplicationBuilder::reconstructImplication(outerChain, negCleanOp);

                                    // Create __contradiction__(op[a]) under the predecessor (Anchor LB)
                                    Memory* predecessor = memoryBlock->parentMemory;
                                    const std::string contradictionKey = "__contradiction__" + cleanOp;
                                    if (predecessor != nullptr &&
                                        simpleMapStore.findChild(predecessor, contradictionKey) == nullptr) {

                                        Memory* contradictionLB = lbStore.create<Memory>();
                                        simpleMapStore.linkChild(predecessor, contradictionKey, contradictionLB);

                                        contradictionLB->parentMemory = predecessor;
                                        contradictionLB->level = static_cast<int>(index);
                                        contradictionLB->setExprKey(contradictionKey);
                                        scopeTypes.assignLbVarDefinitionSets(*contradictionLB);
                                        contradictionLB->primedForContradiction = true;
                                        contradictionLB->contradictionTheoremId =
                                            contradictionLB->nameMap.encode(contrTheorem);

                                        this->permanentBodies.push_back(contradictionLB);

                                        // Seed with clean operator expression as fuel
                                        const int lvRun[1] = { contradictionLB->level };
                                        const TransientOrigin origin{ true, OriginTag::taskFormulation, nullptr, 0 };
                                        seedExpression(cleanOp, *contradictionLB, iteration, 0, lvRun, 1, origin, -1, -1, StrSpan("main", 4), false);

                                        // LB-creation paired mailOut write — see comment at the
                                        // chain-walk LB-creation site (~line 4618) for rationale.
                                        // Documented exception to fillMailOut sole-writer invariant.
                                        if (parameters.trackHistory) {
                                            const int maxOriginsLocal = parameters.compressor_mode
                                                ? parameters.compressor_max_origins_per_expr
                                                : parameters.max_origin_per_expr;
                                            ExpressionWithValidity ev(cleanOp, "main");
                                            contradictionLB->insertMailOutStatement(
                                                StrSpan(ev.original),
                                                StrSpan(ev.validityName), lvRun, 1);
                                            contradictionLB->addMailOutOrigin(
                                                StrSpan(ev.original),
                                                StrSpan(ev.validityName), origin.tag,
                                                origin.deps, origin.depN,
                                                maxOriginsLocal);
                                        }
                                    }
                                    reformulatedOperator = true;
                                }
                            }
                        }
                    }
                }

                if (!reformulatedOperator) {
                    const std::string contradictionKey = "__contradiction__" + head;
                    if (simpleMapStore.findChild(memoryBlock, contradictionKey) == nullptr) {
                        Memory* contradictionLB = lbStore.create<Memory>();
                        simpleMapStore.linkChild(memoryBlock, contradictionKey, contradictionLB);

                        contradictionLB->parentMemory = memoryBlock;
                        contradictionLB->level = static_cast<int>(index + 1);
                        contradictionLB->setExprKey(contradictionKey);
                        scopeTypes.assignLbVarDefinitionSets(*contradictionLB);
                        contradictionLB->primedForContradiction = true;
                        contradictionLB->contradictionTheoremId =
                            contradictionLB->nameMap.encode(
                                gl::ImplicationBuilder::reconstructImplication(chain, gl::DefinitionExpansion::negate(head)));

                        this->permanentBodies.push_back(contradictionLB);

                        // Add the head expression as fuel (status 0)
                        const int lvRun[1] = { contradictionLB->level };
                        const TransientOrigin origin{ true, OriginTag::taskFormulation, nullptr, 0 };
                        seedExpression(head, *contradictionLB, iteration, 0, lvRun, 1, origin, -1, -1, StrSpan("main", 4), false);

                        // LB-creation paired mailOut write — see comment at the
                        // chain-walk LB-creation site (~line 4618) for rationale.
                        // Documented exception to fillMailOut sole-writer invariant.
                        if (parameters.trackHistory) {
                            const int maxOriginsLocal = parameters.compressor_mode
                                ? parameters.compressor_max_origins_per_expr
                                : parameters.max_origin_per_expr;
                            ExpressionWithValidity ev(head, "main");
                            contradictionLB->insertMailOutStatement(
                                StrSpan(ev.original), StrSpan(ev.validityName),
                                lvRun, 1);
                            contradictionLB->addMailOutOrigin(
                                StrSpan(ev.original), StrSpan(ev.validityName),
                                origin.tag, origin.deps, origin.depN,
                                maxOriginsLocal);
                        }

                    }
                }
            }

            // --- Incubator: complementary contradiction LB (reductio) ---
            // Assumes the NEGATION of the head; a contradiction then proves
            // the conjecture itself. Dual of the try_contradiction LB above,
            // which assumes the head verbatim and can only ever disprove it.
            if (parameters.try_contradiction_negated_head && !proved) {
                const std::string negHead = gl::DefinitionExpansion::negate(head);
                const std::string contradictionKey = "__contradiction__" + negHead;
                if (simpleMapStore.findChild(memoryBlock, contradictionKey) == nullptr) {
                    Memory* contradictionLB = lbStore.create<Memory>();
                    simpleMapStore.linkChild(memoryBlock, contradictionKey, contradictionLB);

                    contradictionLB->parentMemory = memoryBlock;
                    contradictionLB->level = static_cast<int>(index + 1);
                    contradictionLB->setExprKey(contradictionKey);
                    scopeTypes.assignLbVarDefinitionSets(*contradictionLB);
                    contradictionLB->primedForContradiction = true;
                    contradictionLB->contradictionTheoremId =
                        contradictionLB->nameMap.encode(
                            gl::ImplicationBuilder::reconstructImplication(chain, head));

                    this->permanentBodies.push_back(contradictionLB);

                    // Add the negated head as fuel (status 0)
                    const int lvRun[1] = { contradictionLB->level };
                    const TransientOrigin origin{ true, OriginTag::taskFormulation, nullptr, 0 };
                    seedExpression(negHead, *contradictionLB, iteration, 0, lvRun, 1, origin, -1, -1, StrSpan("main", 4), false);

                    // LB-creation paired mailOut write — see comment at the
                    // chain-walk LB-creation site for rationale. Documented
                    // exception to fillMailOut sole-writer invariant.
                    if (parameters.trackHistory) {
                        const int maxOriginsLocal = parameters.compressor_mode
                            ? parameters.compressor_max_origins_per_expr
                            : parameters.max_origin_per_expr;
                        ExpressionWithValidity ev(negHead, "main");
                        contradictionLB->insertMailOutStatement(
                            StrSpan(ev.original), StrSpan(ev.validityName),
                            lvRun, 1);
                        contradictionLB->addMailOutOrigin(
                            StrSpan(ev.original), StrSpan(ev.validityName),
                            origin.tag, origin.deps, origin.depN,
                            maxOriginsLocal);
                    }
                }
            }

        }
    }
}

/// @brief Whether a head-switched mirror is the SAME statement as
///        its source up to a renaming of bound variables — the
///        normalize-recognize-skip test of the mirror pre-emit.
///
/// @details
/// A conjecture of the totality shape — symmetric binder prefix,
/// identically-guarded element binders — head-switches into an
/// alpha-variant of itself (rename 9↔10 and the two `(in[·,1])`
/// binder blocks exchange); scheduling that mirror as its own
/// conjecture grid proves the identical statement twice. The test:
///
/// 1. Disintegrate both strings into (chain, head); equal chain
///    lengths required.
/// 2. Derive the candidate variable bijection π from the
///    token-aligned pairing of the two heads: the delimiter
///    skeletons must match byte-for-byte, and every differing token
///    pair must be two BOUND variables, accumulated into a
///    consistent bijection (identity pairs of bound variables are
///    recorded too; unconstrained bound variables complete as
///    identity).
/// 3. Verify π globally: the π-image of the source's chain links
///    (premise plus bound-var list, token-boundary substitution)
///    must equal the mirror's links as a MULTISET. Multiset — not
///    sequence — because the renaming legitimately exchanges
///    identically-shaped binder blocks; both strings are
///    well-formed theorems, so each side's own binder order is
///    scope-legal and multiset equality under π is statement
///    equality.
///
/// A structural mismatch anywhere (head skeletons differ, a
/// non-bound token differs, π inconsistent, chain multisets differ)
/// returns false — the mirror is a genuinely different statement
/// (the Peano or0 parents) and stays scheduled.
///
/// @param original The source conjecture (compiled form).
/// @param mirror   Its `headSwitchOne` output (compiled form).
/// @return True iff the mirror is an alpha-variant of the source.
/// @invariant Pure — no analyzer state is read beyond
///            `coreExpressionMap` (the disintegration walk) and no
///            state is written.
/// @see headSwitchOne — the mirror producer; the pre-emit block in
///      `analyzeExpressions` — the sole production caller.
bool ProofGridBuilder::mirrorIsAlphaVariant(const std::string& original,
                                              const std::string& mirror) const {
    // 1. Disintegrate both sides.
    std::vector<std::tuple<std::string, std::vector<std::string>,
                           std::set<std::string>>> chainO, chainM;
    const std::string headO =
        ce::disintegrateImplication(original, chainO, this->definitions.configurations());
    const std::string headM =
        ce::disintegrateImplication(mirror, chainM, this->definitions.configurations());
    if (chainO.size() != chainM.size()) return false;

    // Bound-variable universe of the source (every binder-list entry).
    std::set<std::string> bound;
    for (const auto& link : chainO)
        for (const std::string& v : std::get<1>(link)) bound.insert(v);

    // 2. Candidate bijection from the token-aligned head pairing.
    const auto isTokByte = [](char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
            || (c >= '0' && c <= '9') || c == '_';
    };
    std::map<std::string, std::string> pi;     // source var -> mirror var
    std::map<std::string, std::string> piInv;
    {
        std::size_t i = 0, j = 0;
        while (i < headO.size() || j < headM.size()) {
            const bool ta = i < headO.size() && isTokByte(headO[i]);
            const bool tb = j < headM.size() && isTokByte(headM[j]);
            if (ta != tb) return false;
            if (!ta) {
                if (i >= headO.size() || j >= headM.size()
                    || headO[i] != headM[j]) return false;
                ++i; ++j;
                continue;
            }
            const std::size_t i0 = i, j0 = j;
            while (i < headO.size() && isTokByte(headO[i])) ++i;
            while (j < headM.size() && isTokByte(headM[j])) ++j;
            const std::string tokO = headO.substr(i0, i - i0);
            const std::string tokM = headM.substr(j0, j - j0);
            const bool boundO = bound.count(tokO) != 0;
            const bool boundM = bound.count(tokM) != 0;
            if (tokO != tokM && (!boundO || !boundM)) return false;
            if (boundO != boundM) return false;
            if (boundO) {
                const auto it = pi.find(tokO);
                if (it != pi.end()) {
                    if (it->second != tokM) return false;
                } else {
                    if (piInv.count(tokM)) return false;
                    pi[tokO] = tokM;
                    piInv[tokM] = tokO;
                }
            }
        }
    }
    // Complete π to a permutation: untouched bound variables map to
    // themselves (must not collide with an existing image).
    for (const std::string& v : bound) {
        if (pi.count(v)) continue;
        if (piInv.count(v)) return false;
        pi[v] = v;
        piInv[v] = v;
    }

    // 3. Global verification: π-image of the source links == mirror links
    //    as a multiset (premise + ordered bound-var list per link).
    const char SEP = '\x01';
    std::vector<std::string> imgO, keysM;
    imgO.reserve(chainO.size());
    keysM.reserve(chainM.size());
    for (const auto& link : chainO) {
        std::string key = ce::replaceKeysInString(std::get<0>(link), pi);
        for (const std::string& v : std::get<1>(link)) {
            key += SEP;
            const auto it = pi.find(v);
            key += (it != pi.end()) ? it->second : v;
        }
        imgO.push_back(std::move(key));
    }
    for (const auto& link : chainM) {
        std::string key = std::get<0>(link);
        for (const std::string& v : std::get<1>(link)) {
            key += SEP;
            key += v;
        }
        keysM.push_back(std::move(key));
    }
    std::sort(imgO.begin(), imgO.end());
    std::sort(keysM.begin(), keysM.end());
    return imgO == keysM;
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)
void ProofGridBuilder::prefillIntegrationMapsRecursive(Memory* mb) {
    if (mb == nullptr) return;
    if (mb->isPartOfRecursion)
    {
        return;
    }

    // Collect expression keys from the current block and all its predecessors
    std::vector<std::string> exprKeys;
    Memory* current = mb;
    while (current != nullptr) {
        if (!current->exprKey().empty()) {
            exprKeys.push_back(current->exprKey());
        }
        current = current->parentMemory;
    }

    // Process all collected expressions for the current block *mb
    for (const std::string& expr : exprKeys) {
        const StrSpan coreSpan = extractExpressionSpan(StrSpan(expr));
        if (this->definitions.operatorNames().find(std::string_view(coreSpan.ptr,
                static_cast<std::size_t>(coreSpan.len))) != this->definitions.operatorNames().end()) {
            const ce::CoreExpressionConfig* cfg = this->definitions.coreConfig(coreSpan);
            assert(cfg && "prefillIntegrationMapsRecursive: operator core must be in coreExpressionMap");

            const LogicalEntity* le = this->definitions.compiledEntity(coreSpan);
            assert(le && "prefillIntegrationMapsRecursive: operator core must be compiled");

            if (!equalSpans(StrSpan(le->category), StrSpan("atomic", 6))) {
                assert(!cfg->outputIndices.empty());

                std::vector<std::string> args = ce::getArgs(expr);
                // Assuming the operator has one output variable as implied by context
                int outIdx = cfg->outputIndices[0];

                assert(outIdx >= 0 && outIdx < static_cast<int>(args.size()));

                std::string outputArg = args[outIdx];

                std::map<std::string, std::string> replacementMap;
                replacementMap[outputArg] = "marker";
                std::string rplExpr2 = ce::replaceKeysInString(expr, replacementMap);

                std::vector<std::string> tempArgs = ce::getArgs(rplExpr2);
                std::set<std::string> argSet(tempArgs.begin(), tempArgs.end());

                // Per instructions: erase marker from argSet and call prepareIntegration
                assert(argSet.find("marker") != argSet.end());

                argSet.erase("marker");
                StrSpan unchRun[ExecutionParameters::MAX_ARITY];
                int32_t unchN = 0;
                for (const std::string& a : argSet) {
                    assert(unchN < ExecutionParameters::MAX_ARITY
                        && "prepareIntegration: unchangeable args exceed MAX_ARITY");
                    unchRun[unchN++] = StrSpan(a);
                }
                integrationProcessor.prepareIntegration(StrSpan(rplExpr2), unchRun, unchN, *mb, StrSpan("main", 4), StrSpan());
            }
        }
    }

    // Recurse into children
    simpleMapStore.forEachChild(mb, [&](const gl::StrSpan&, Memory* child) {
        prefillIntegrationMapsRecursive(child);
    });
}

//#pragma optimize("", off)

/// @brief Whether a goal cites a `(1)`-typed anchor slot value as a
///        combinable argument: the trigger of the anchor-numeral
///        exception.
///
/// @details
/// The goal is expanded to base form (`DefinitionExpansion::expandToBaseForm`
/// unfolds every compiled `and` / `or` / `implication` / `existence` name),
/// so every remaining atom `(name[args])` names a configured expression.
/// An argument equal to a slot value counts only when its position is
/// combinable in that expression's `definition_sets`, the flag the
/// conjecturer generalises over. A fixed definition parameter (the flag
/// off, as `nontrivialFactor`'s lower bound `i2` or `prime`'s `i1` and
/// `i2`) belongs to the definition and is no value the theorem is about,
/// so it never triggers the exception (D-548).
/// Quantifier lists `(>[...]` are no atoms and are skipped.
///
/// @param goal The goal text as queued (compiled or base form).
/// @param slotValues The batch anchor's `(1)`-typed slot values.
/// @return True when some atom cites a slot value at a combinable position.
/// @invariant Every atom of the base form names a configured expression
///            whose `definition_sets` covers each argument position
///            (asserted).
/// @see prehandleAnchor (the caller); I-188.
bool ProofGridBuilder::goalCitesSlotValueAsArgument(const std::string& goal,
    const std::set<std::string>& slotValues) const {
    const DefinitionExpansion expansion(definitions.configurations(), definitions.entries());
    const std::string base = expansion.expandToBaseForm(goal);
    const StrSpan text(base);
    for (int32_t open = 0; open < text.len; ++open) {
        if (text.ptr[open] != '[') continue;
        int32_t start = open;
        while (start > 0 && text.ptr[start - 1] != '(') --start;
        assert(start > 0 && "goalCitesSlotValueAsArgument: an argument list without its expression");
        int32_t close = open + 1;
        while (close < text.len && text.ptr[close] != ']') ++close;
        assert(close < text.len && "goalCitesSlotValueAsArgument: an unclosed argument list");
        const StrSpan name(text.ptr + start, open - start);
        if (equalSpans(name, StrSpan(">", 1))) {
            open = close;
            continue;
        }
        assert(definitions.configuredNames().count(std::string_view(name.ptr, static_cast<std::size_t>(name.len))) != 0
            && "goalCitesSlotValueAsArgument: a base-form atom names no configured expression");
        const ce::CoreExpressionConfig* config = definitions.coreConfig(name);
        assert(config != nullptr && "goalCitesSlotValueAsArgument: a configured expression without its configuration");
        int32_t position = 0;
        int32_t from = open + 1;
        for (int32_t at = open + 1; at <= close; ++at) {
            if (at < close && text.ptr[at] != ',') continue;
            const StrSpan argument(text.ptr + from, at - from);
            ++position;
            from = at + 1;
            bool slotValue = false;
            for (const std::string& value : slotValues)
                slotValue = slotValue || equalSpans(argument, StrSpan(value));
            if (!slotValue) continue;
            const std::pair<std::string, bool>* definitionSet =
                CompiledDefinitionRegistry::findDefinitionSet(config, position);
            assert(definitionSet != nullptr
                && "goalCitesSlotValueAsArgument: a configured position without a definition set");
            if (definitionSet->second) return true;
        }
        open = close;
    }
    return false;
}

/// @brief Grid-wide axed-anchor pre-pass: walk the LB tree once; outside
///        recursion subtrees every LB under the batch anchor mints its
///        axed x-names into `intAxedVariables` AND registers its
///        x-substituted anchor expression as a `{-1}`-tier statement with
///        an `anchor handling` history line; recursion subtrees take one
///        of two containment modes (set-only, or the anchor-numeral
///        exception).
///
/// @details
/// For each LB whose ancestor chain reaches the batch anchor LB, the pass
/// collects every argument cited by the chain's exprKeys (the trace) and
/// x-copies exactly the anchor's `(1)`-typed slots whose value appears in
/// that trace: slot value `6` becomes `x6`, minted into the LB's
/// `intAxedVariables`. The x-copy is the anchor-premise completion for
/// rules whose element arguments collide with anchor slot values (the
/// element-vs-anchor-slot collision, I-36 family); the minted axed set
/// makes the containment checks — the `addExprToMemoryBlock` prologue,
/// the `addStatement` door, and the orbit-commit refusal — drop every
/// DERIVED x-citing deposit, so the copies never breed x-facts. The
/// statement write is direct — deliberately bypassing those doors.
///
/// Recursion subtrees (D-272,
/// I-188): a block-#1 root whose goal (the
/// theorem head, queued verbatim at creation) cites a `(1)`-typed anchor
/// slot value as a combinable argument (`goalCitesSlotValueAsArgument`)
/// puts its whole subtree in the ANCHOR-NUMERAL EXCEPTION mode
/// — untouched by this pass, so the ancestor's axed-anchor statement
/// arriving by mail registers at the inert door and the historical
/// premise-completion machinery runs there (anchor-numeral theorems such
/// as unit-product need it). Every other recursion subtree — block-#2
/// `_induction_` side-chains always included — is SET-ONLY: axed names
/// minted, no statement, the mailed ancestor form refused at the armed
/// door (a descendant's trace-accumulated axed set is a superset of its
/// ancestors'), keeping those induction sub-blocks entirely x-free.
///
/// @param mb Subtree root to process; the sole production caller passes
///           the grid root once per grid build (from `buildGrid`).
/// @param axedMode Threaded subtree mode: 0 outside recursion subtrees,
///                 1 set-only containment, 2 anchor-numeral exception
///                 (see the mode comment in the body); callers pass 0.
/// @invariant Registration is level-tier `{-1}` (definitional, not
///            derived) and dedup-guarded by the `intKnownStatements`
///            lookup, so re-walks never double-register.
/// @see `addExprToMemoryBlock` — the axed-containment door; SwDD
///      `20_core_concepts/06_anchors_and_scopes.md`.
void ProofGridBuilder::prehandleAnchor(Memory* mb, int axedMode) {
    if (mb == nullptr) return;

    // Recursion-subtree mode, threaded down the walk (only subtree roots
    // carry `isPartOfRecursion`):
    //   0 — outside recursion subtrees: mint + register (full treatment).
    //   1 — set-only containment: mint the axed names so the door refuses
    //       every x-citing deposit (the mailed ancestor-form axed anchor
    //       included); no statement. The subtree stays entirely x-free.
    //   2 — anchor-numeral exception: the subtree is untouched — no mint,
    //       no registration — so the ancestor's axed-anchor statement
    //       arriving by mail registers at the inert door and the
    //       historical premise-completion machinery runs inside this
    //       grid's induction blocks. Chosen at a block-#1 recursion root
    //       whose goal (the theorem head verbatim) cites a `(1)`-typed
    //       anchor slot value as a combinable argument, never as a fixed
    //       definition parameter; block-#2 roots (`_induction_` equality
    //       side-chains) always take mode 1 — their synthetic
    //       `(=[digitArg,zero])` goal cites the zero numeral by
    //       construction, not because the theorem is about numerals.
    int mode = axedMode;
    const bool isSubtreeRoot = (axedMode == 0) && mb->isPartOfRecursion;
    if (isSubtreeRoot) mode = 1;

    const bool isAnchorLB = blockIdentity.isAnchorLb(*mb);

    // Skip the body work for the Anchor LB itself; still recurse below.
    // Mode-2 descendants skip it too — their whole subtree is untouched.
    if (!isAnchorLB && mode != 2) {

        // 1. Trace the hierarchy to find the specific Anchor Key
        std::string anchorExprKey;
        Memory* current = mb;
        std::set<std::string> traceVariables; // Collect variables from trace

        while (current != nullptr) {
            if (!current->exprKey().empty()) {
                // Check if this ancestor is the Anchor LB
                if (blockIdentity.isAnchorLb(*current)) {
                    anchorExprKey = current->exprKey();
                    break;
                }
                // Collect variables from the current trace element's key
                std::vector<std::string> kArgs = ce::getArgs(current->exprKey());
                traceVariables.insert(kArgs.begin(), kArgs.end());
            }
            current = current->parentMemory;
        }

        // 2. If we found an anchor ancestor, proceed
        if (!anchorExprKey.empty()) {
            std::vector<std::string> args = ce::getArgs(anchorExprKey);
            std::map<std::string, std::string> replacementMap;

            // 3. Collect the (1)-typed slot VALUES and build the
            //    replacement map (trace-guarded). Minting is deferred until
            //    the subtree mode is final: a mode-2 root must stay
            //    entirely untouched.
            std::set<std::string> slotValues;
            for (const auto& [slot, pattern] : this->anchorInfo.definitionSets) {
                if (pattern == "(1)") {
                    try {
                        int index = std::stoi(slot) - 1;
                        if (index >= 0 && index < static_cast<int>(args.size())) {
                            std::string originalVar = args[index];
                            slotValues.insert(originalVar);

                            // Check if the variable exists in the trace
                            if (traceVariables.find(originalVar) != traceVariables.end()) {
                                // Only apply x-prefix if not already present
                                if (originalVar.rfind("x", 0) != 0) {
                                    replacementMap[originalVar] = "x" + originalVar;
                                }
                            }
                        }
                    }
                    catch (...) {
                        // Ignore malformed slots
                    }
                }
            }

            // 3b. Anchor-numeral exception: at a block-#1 recursion root,
            //     scan the LB's own goals (the theorem head, queued
            //     verbatim at recursion-block creation) for a (1)-typed
            //     slot value cited as a combinable argument. A hit
            //     upgrades the subtree to mode 2 (untouched).
            if (isSubtreeRoot
                && mb->exprKey().find("_induction_") == std::string::npos
                && !slotValues.empty()) {
                for (const DecodedToBeProvedRow& row :
                     decodeToBeProvedSorted(mb->intToBeProved, mb->nameMap)) {
                    if (goalCitesSlotValueAsArgument(row.original, slotValues)) { mode = 2; break; }
                }
            }

            // 3c. Mint the axed names (modes 0 and 1 — the armed door).
            if (mode != 2) {
                for (const auto& [originalVar, xVar] : replacementMap) {
                    mb->intAxedVariables.mint(mb->nameMap.encode(xVar));
                }
            }

            // 4. Create and Add the Anchor Expression using the map —
            //    mode 0 only (D-272): mode-1
            //    subtrees mint the axed set above but never hold the
            //    axed-anchor statement, so no x-citing firing can exist
            //    there; mode-2 subtrees receive the ancestor's form by
            //    mail instead.
            if (!replacementMap.empty() && mode == 0) {
                std::string replacedAnchor = ce::replaceKeysInString(anchorExprKey, replacementMap);

                EncodedExpression enc(replacedAnchor, "main");

                // Add only if not already present
                const StatementFlags* anchorRow = lookupStatementFlags(
                    mb->intKnownStatements, mb->nameMap,
                    enc.original, enc.validityName);
                if (anchorRow == nullptr) {

                    // Anchors are definitional, not derived — the {-1}
                    // non-derived tier, transparent to level accounting.
                    const int lv0[1] = { -1 };

                    { IntEncodedExpr ie = encodeExpression(enc, mb->nameMap);
                      mb->intEncodedStatements.push_back(ie);
                      mb->intLocalEncodedStatements.push_back(ie);
                      mb->intLocalEncodedStatementsDelta.push_back(ie);
                      mb->intLocalEncodedStatementsSet.mint(
                          packStatementKey(ie.originalId, ie.validityId));
                      mb->intStatementLevelsMap.assignSetRange(
                          packStatementKey(ie.originalId, ie.validityId),
                          lv0, lv0 + 1); }
                    gl::StatementRegistryOperations::publishKnownStatementKey(*mb, packStatementKey(
                        mb->nameMap.encode(enc.original),
                        mb->nameMap.encode(enc.validityName)),
                        /*local=*/true);

                    // L3 span-record door. The "anchor handling" antecedent
                    // (the original anchor expression `anchorExprKey`) and the
                    // KEY (`replacedAnchor`) are in-hand std::string locals;
                    // "main" is a static literal.
                    const StrSpan pMain("main", 4);
                    const OriginDep pDeps[1] = { { StrSpan(anchorExprKey), pMain } };
                    addOriginEncoded(mb->exprOriginMap, mb->originInterner, StrSpan(replacedAnchor), pMain, OriginTag::anchorHandling, pDeps, 1, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                    // The history line is written locally on the same LB
                    // that registers the statement, so buildStack resolves
                    // any firing that cites the axed anchor as an
                    // antecedent. The statement also rides the Delta into
                    // mailOut, but every descendant's axed set is a superset
                    // of this LB's (the trace accumulates down the chain), so
                    // the mailed copy is refused at descendant doors rather
                    // than re-registered — recursion subtrees in particular
                    // never hold it (D-272).
                }
            }
        }
    }

    // 5. Recurse into children, threading the subtree mode
    simpleMapStore.forEachChild(mb, [&](const gl::StrSpan&, Memory* child) {
        prehandleAnchor(child, mode);
    });
}

/// @brief Tear down the LB grid at batch end — delete every child Memory and
///        reset the root's grid links. See the declaration for the full
///        contract.
///
/// @details
/// The four `HashMemory` instances per LB are arena-backed (every `encodedMap`
/// on the cold `lbMemory.manager`) yet declared before that arena, so their
/// `~PagedVector` would otherwise `freePage` on a dead arena. Empty each LB's
/// `encodedMap` while the arena is alive — children before `delete`, the root in
/// place — then drop the grid links. A deload-released LB is already empty, so
/// the release is residency-gated.
void ProofGridBuilder::destroyGrid() {
    // 1) Delete every child Memory, emptying its arena-backed encodedMaps
    //    first so ~Memory does not freePage on an already-released arena.
    std::unordered_set<Memory*> seen;
    seen.insert(&this->body);  // never delete the root
    std::vector<Memory*> stack;
    simpleMapStore.forEachChild(&this->body, [&](const gl::StrSpan&, Memory* child) {
        if (child) stack.push_back(child);
    });
    while (!stack.empty()) {
        Memory* node = stack.back(); stack.pop_back();
        if (!node || !seen.insert(node).second) continue;
        simpleMapStore.forEachChild(node, [&](const gl::StrSpan&, Memory* child) {
            if (child) stack.push_back(child);
        });
        if (node->lbMemory.manager.resident()) {
            node->lbMemory.existenceLookups.clear();
            node->lbMemory.contradictionFacts.clear();
            node->overallHashMemory.releaseAllCold();
            node->localHashMemory.releaseAllCold();
            node->localHashMemoryDelta.releaseAllCold();
            node->workingMemory.releaseAllCold();
        }
        lbStore.destroy(node);
    }
    // 2) Empty the root's arena-backed cold HashMemory containers in place (its
    //    arena destructs before them at end of process — same hazard).
    if (this->body.lbMemory.manager.resident()) {
        this->body.lbMemory.existenceLookups.clear();
        this->body.lbMemory.contradictionFacts.clear();
        this->body.overallHashMemory.releaseAllCold();
        this->body.localHashMemory.releaseAllCold();
        this->body.localHashMemoryDelta.releaseAllCold();
        this->body.workingMemory.releaseAllCold();
    }
    // 3) Drop child links and reset the grid LB list to the root.
    simpleMapStore.clear();
    permanentBodies.clear();
    permanentBodies.push_back(&this->body);
    // New mail system (D-137): drop the per-LB pull logs and
    // cursors. The Memory* keys were just freed above, so clearing here keeps a
    // stale pointer-keyed entry from aliasing a reused address next batch.
    this->mailLog.clear();
}
/// @brief The canonical spelling of a compiled conjecture: its variables
///        renumbered in first-occurrence order and each bound at its first
///        premise (D-493).
///
/// @details
/// A compiled implication compact stores its chain in this form —
/// first-occurrence numbering over the compiled premises and the head
/// (`CompilationNormalizer::makeNormalizedEncodedKey`) — and expands to it
/// with full binding (`ImplicationBuilder::reconstructImplicationFullBind`).
/// A supplied conjecture may number its bound variables in another order, or
/// leave a numbering gap where a bound variable disappeared into a compiled
/// existence; the theorem it proves would then be registered under a spelling
/// its compact never expands to, and a chapter citing the compact could not
/// reach the theorem's `theorem` history row. Generated conjectures are
/// canonical already, so for them the call is the identity.
///
/// @param compiled A conjecture after
///                 `StructuralCompiler::precompileStructuralOperators`:
///                 every premise and the head a simple, possibly negated,
///                 atom; every argument a bound variable.
/// @return The canonical spelling.
/// @invariant The result is alpha-equivalent to @p compiled: the same premise
///            and head shapes under a bijective renaming. Every chain element
///            is simple and every argument is bound (asserted).
/// @see prepareBatch — the caller; addTheoremToMemory.
std::string ProofGridBuilder::canonicalConjecture(const std::string& compiled) const {
    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> links;
    const std::string head = ce::disintegrateImplication(compiled, links, definitions.configurations());
    std::vector<std::string> premises;
    premises.reserve(links.size());
    for (const auto& link : links) premises.push_back(std::get<0>(link));

    // Every binder list of the compiled text names bound variables only.
    std::set<std::string> bound;
    for (std::size_t at = compiled.find(">["); at != std::string::npos; at = compiled.find(">[", at + 2)) {
        const std::size_t close = compiled.find(']', at + 2);
        assert(close != std::string::npos && "canonicalConjecture: an unterminated binder list");
        const std::string list = compiled.substr(at + 2, close - at - 2);
        for (std::size_t from = 0; from < list.size(); ) {
            const std::size_t comma = list.find(',', from);
            const std::size_t to = comma == std::string::npos ? list.size() : comma;
            if (to > from) bound.insert(list.substr(from, to - from));
            from = to + 1;
        }
    }

    std::map<std::string, std::string> renaming;
    const auto number = [&](const std::string& element) {
        assert(ce::expressionIsSimple(element) && "canonicalConjecture: a chain element is not a simple atom");
        for (const std::string& argument : ce::getArgs(element)) {
            assert(bound.count(argument) != 0 && "canonicalConjecture: a conjecture argument is not bound");
            if (renaming.find(argument) == renaming.end())
                renaming.emplace(argument, std::to_string(renaming.size() + 1));
        }
    };
    for (const std::string& premise : premises) number(premise);
    number(head);
    for (std::string& premise : premises) premise = ce::replaceKeysInString(premise, renaming);
    return ImplicationBuilder::reconstructImplicationFullBind(premises, ce::replaceKeysInString(head, renaming));
}

/// @brief The partner of each induction variable of a conjecture with twin
///        premises (D-587).
///
/// @details
/// Two non-anchor premises are twins when they are the only two premises of
/// one predicate and polarity, no other predicate repeats, and the conjecture
/// is symmetric in them: exchanging the two premises gives the same lemma
/// after normalization (`canonicalConjecture`: first-occurrence numbering,
/// every variable bound at its first premise). An equality head, positive or
/// negated, reads unordered, so the exchanged lemma may also match with the
/// head's two arguments swapped.
///
/// A digit variable k of the first twin has the partner l when the second
/// twin holds l at the same argument slot, l is a digit variable, l differs
/// from k, k does not occur in the second twin and l does not occur in the
/// first. Exchanging the twins maps an induction on l onto the induction on
/// k, so the grid builds triads on k only and its weak hypothesis leaves l
/// universal. FTA2's full I5 `primeFactorSeq(k,P,n) ∧ primeFactorSeq(l,Q,n)
/// → P = Q` gives k ↦ l; the equal-length form, with k in both premises,
/// gives nothing.
///
/// @param compiled A conjecture after
///                 `StructuralCompiler::precompileStructuralOperators`
///                 (every premise and the head a simple, possibly negated,
///                 atom; every argument bound).
/// @param digits   The conjecture's digit arguments (`findDigitArgs`).
/// @param digitN   Number of @p digits.
/// @return Map from each first-twin induction variable to its partner; empty
///         when the conjecture has no twin premises or the twins hold no
///         partner pair.
/// @invariant Pure: reads the compiled configuration only, writes nothing.
///            Twins of one predicate have one arity (asserted).
/// @see addTheoremToMemory, findImmutableArgs, canonicalConjecture.
std::map<std::string, std::string> ProofGridBuilder::twinInductionPartners(const std::string& compiled,
    const StrSpan* digits, int32_t digitN) const {
    std::map<std::string, std::string> partners;
    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> links;
    const std::string head = ce::disintegrateImplication(compiled, links, definitions.configurations());
    assert(!head.empty() && "twinInductionPartners: the conjecture has no head");
    std::vector<std::string> premises;
    premises.reserve(links.size());
    for (const auto& link : links) premises.push_back(std::get<0>(link));

    // 1. The twins: the one predicate (with its polarity) that two non-anchor
    //    premises share; a predicate held three times or a second repeated
    //    predicate leaves the conjecture without twins.
    std::map<std::string, std::vector<std::size_t>> byPredicate;
    for (std::size_t i = 0; i < premises.size(); ++i) {
        const StrSpan core = extractExpressionSpan(StrSpan(premises[i]));
        assert(core.len > 0 && "twinInductionPartners: a premise has no predicate");
        const std::string name(core.ptr, static_cast<std::size_t>(core.len));
        if (name == anchorInfo.name) continue;
        byPredicate[(premises[i][0] == '!' ? std::string("!") : std::string()) + name].push_back(i);
    }
    std::size_t first = 0;
    std::size_t second = 0;
    int repeated = 0;
    for (const auto& entry : byPredicate) {
        if (entry.second.size() == 1) continue;
        if (entry.second.size() != 2) return partners;
        first = entry.second[0];
        second = entry.second[1];
        ++repeated;
    }
    if (repeated != 1) return partners;

    // 2. Symmetry: the exchanged lemma normalizes to the conjecture itself;
    //    an equality head reads unordered.
    std::vector<std::string> exchanged = premises;
    std::swap(exchanged[first], exchanged[second]);
    const std::string normalized = canonicalConjecture(compiled);
    bool symmetric = canonicalConjecture(
        ImplicationBuilder::reconstructImplicationFullBind(exchanged, head)) == normalized;
    if (!symmetric && equalSpans(extractExpressionSpan(StrSpan(head)), StrSpan("=", 1))) {
        const std::vector<std::string> headArgs = ce::getArgs(head);
        assert(headArgs.size() == 2 && "twinInductionPartners: an equality head has two arguments");
        const std::string mirrored = std::string(head[0] == '!' ? "!" : "")
            + "(=[" + headArgs[1] + "," + headArgs[0] + "])";
        symmetric = canonicalConjecture(
            ImplicationBuilder::reconstructImplicationFullBind(exchanged, mirrored)) == normalized;
    }
    if (!symmetric) return partners;

    // 3. Partners: a digit of the first twin facing another digit of the
    //    second twin at the same slot, each absent from the other twin.
    const std::vector<std::string> firstArgs = ce::getArgs(premises[first]);
    const std::vector<std::string> secondArgs = ce::getArgs(premises[second]);
    assert(firstArgs.size() == secondArgs.size()
        && "twinInductionPartners: twins of one predicate differ in arity");
    const auto isDigit = [&](const std::string& argument) {
        for (int32_t i = 0; i < digitN; ++i)
            if (equalSpans(digits[i], StrSpan(argument))) return true;
        return false;
    };
    const auto occursIn = [](const std::vector<std::string>& arguments, const std::string& argument) {
        return std::find(arguments.begin(), arguments.end(), argument) != arguments.end();
    };
    for (std::size_t slot = 0; slot < firstArgs.size(); ++slot) {
        const std::string& k = firstArgs[slot];
        const std::string& l = secondArgs[slot];
        if (k == l || !isDigit(k) || !isDigit(l)) continue;
        if (occursIn(secondArgs, k) || occursIn(firstArgs, l)) continue;
        partners.emplace(k, l);
    }
    return partners;
}

/// @brief Compile conjectures and prepare their logical-block grid.
/// @details Preserves mirror generation, registration, anchor preprocessing and startup mail order. The conjecture vector and raw-to-compiled map remain caller-owned at their original lifetimes.
/// @param filteredConjectures Surviving conjectures, extended with distinct mirrors.
/// @param rawToCompiled Caller-owned mapping for changed compiled forms.
/// @param structuralCompiler Compiler operation for nested structural forms.
/// @param orTheoremBuilder Mirror construction operation.
/// @param globalDependencies Existing global dependency store.
/// @param blockMailProcessor Ancestor-routing operation.
/// @param dormantLogicBlocksAtGridBuild Dormant-block observation count.
/// @param rollingMailHistoryEnabled Existing retention-policy result.
/// @return No value.
void ProofGridBuilder::prepareBatch(std::vector<std::string>& filteredConjectures, std::unordered_map<std::string, std::string>& rawToCompiled, StructuralCompiler& structuralCompiler, OrTheoremBuilder& orTheoremBuilder, Dependencies& globalDependencies, BlockMailProcessor& blockMailProcessor, int& dormantLogicBlocksAtGridBuild, bool& rollingMailHistoryEnabled) {
    auto compileAndRegister = [&](const std::vector<std::string>& conjectures) {
        for (std::size_t ti = 0; ti < conjectures.size(); ++ti) {
            const std::string raw = conjectures[ti];
            std::string conj = raw;

            // Precompile ALL structural operators (!(&..) / !(>..)) anywhere
            // in the theorem so addTheoremToMemory's disintegrateImplication
            // produces chain elements and a head that are all already compiled
            // names (or<N> / existence<N>). Substring-scan approach leaves the
            // outer (>...) chain untouched, so the bound-variable lists stay
            // self-consistent even when the inner existence encoding renames
            // its bound variables.
            structuralCompiler.precompileStructuralOperators(conj);
            // Register the spelling the theorem's compact expands to
            // (D-493); generated conjectures
            // are canonical already.
            conj = canonicalConjecture(conj);

            if (conj != raw) rawToCompiled[raw] = conj;

            addTheoremToMemory(conj, this->body, 0, false, globalDependencies);
        }
    };

    // Grid teardown is the member `ProofGridBuilder::destroyGrid()`, called
    // after the proof-graph export in `run_modes::fullRun` (the grid must
    // survive `analyzeExpressions` for the export).

    // =========================================================================
    // Helper: build grid from current permanentBodies
    // =========================================================================
    auto buildGrid = [&]() {
        // Integration demand comes from explicit goals and installed rule premises.
        // A known operator premise does not request alternative output witnesses.
        prehandleAnchor(&this->body);

        // Process-documentation telemetry for the mail-history experiment.
        // Count after every LB has been created and anchor prehandling has run,
        // at the exact program point where the retention policy is fixed.
        dormantLogicBlocksAtGridBuild = 0;
        for (const Memory* lb : permanentBodies) {
            assert(lb != nullptr && "buildGrid: permanentBodies contains null");
            if (!lb->isActive) ++dormantLogicBlocksAtGridBuild;
        }
        rollingMailHistoryEnabled =
            dormantLogicBlocksAtGridBuild == 0;

        // New mail system (D-137): register every LB so the
        // parallel phase-1 pull only advances pre-existing cursor cells, never
        // inserting or rehashing. Every LB the prover will ever run -- the root
        // &body included (it is pushed into permanentBodies at construction) --
        // is in permanentBodies and no LB is born after this point, so this
        // single pass covers every ancestor any pull will read.
        // The ancestor list is the LB's whole chain outside incubator mode
        // and stops at the anchor LB inside it (collectMailAncestors,
        // D-332).
        std::vector<const Memory*> ancestors;
        for (Memory* lb : permanentBodies) {
            if (!lb) continue;
            blockMailProcessor.collectMailAncestors(lb, ancestors);
            this->mailLog.registerLb(lb, ancestors);
        }

        // New mail system (D-137): commit every LB's startup
        // mailOut (the LB-creation paired writes at theorem-load +
        // prehandleAnchor) into its own log, so each descendant pulls its
        // ancestors' startup batches at step-1 phase-1. Replaces the old
        // sendMail-into-boxes + smashMail dispatch. Single-threaded at grid
        // build; every LB was registered just above, so commit never inserts.
        for (Memory* lb : permanentBodies) {
            if (!lb) continue;
            this->mailLog.commit(lb, lb->mailOut);
            lb->clearMailOut();
        }
    };

    // =========================================================================
    // 2. Initial grid setup
    // =========================================================================
    // Pre-emit head-switched mirrors. Folds the contrapositive of every
    // structurally-qualifying conjecture into the initial pool so the prover
    // gets the same proof opportunities in one pass that the legacy
    // multi-iteration path reached across multiple passes. headSwitchOne
    // calls disintegrateImplication which expects structural operators in
    // compiled form (or<N> / existence<N>), not raw `!(&...)` / `!(>...)`.
    // Conjectures coming out of the CE filter are still raw, so we
    // precompile a working copy per conjecture before the structural scan.
    // The original filteredConjectures stays raw — compileAndRegister
    // precompiles it again (its own per-conjecture pass at the top of the
    // lambda), so the input contract is preserved.
    {
        std::unordered_set<std::string> seen(filteredConjectures.begin(),
                                             filteredConjectures.end());
        std::vector<std::string> mirrors;
        for (const std::string& conj : filteredConjectures) {
            std::string compiled = conj;
            structuralCompiler.precompileStructuralOperators(compiled);
            std::string mirror = orTheoremBuilder.headSwitchOne(compiled);
            if (mirror.empty()) continue;
            // Normalize-recognize-skip (maintainer-designed): a mirror that
            // is the SAME statement as its source up to bound-variable
            // renaming (the totality shape — symmetric binder prefix with
            // identical guards) would prove the identical theorem twice on
            // its own conjecture grid. Recognize the alpha-variant and skip
            // scheduling it; genuinely different mirrors (the Peano or0
            // parents, predicate-swapping shapes) keep being scheduled so
            // the pair or-construction stays intact.
            if (mirrorIsAlphaVariant(compiled, mirror)) {
                diagnosticsLog() << "[head-switch pre-emit] mirror is an "
                             "alpha-variant of its source - skipped: "
                          << conj << std::endl;
                continue;
            }
            if (seen.insert(mirror).second) mirrors.push_back(mirror);
        }
        if (!mirrors.empty()) {
            diagnosticsLog() << "[head-switch pre-emit] adding "
                      << mirrors.size() << " head-switched mirrors to the "
                      << "conjecture pool (was " << filteredConjectures.size()
                      << ")" << std::endl;
            filteredConjectures.insert(filteredConjectures.end(),
                                       mirrors.begin(), mirrors.end());
        }
    }
    compileAndRegister(filteredConjectures);
    buildGrid();

}

}
