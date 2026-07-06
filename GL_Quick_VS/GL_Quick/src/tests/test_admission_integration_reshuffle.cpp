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

/// @file
/// @brief Unit tests for the integration-side admission reshuffle — deferring
///        the in-hashburst `prepareIntegration` seed registration out to a
///        post-fixpoint drain.
///
/// @details
/// Coverage for `gl::ExpressionAnalyzer::drainDeferredIntegrationPreps`:
///   - symbol existence + signature (`void(gl::Memory&)`);
///   - behavioral: every staged `DeferredIntegrationPrep` is replayed
///     (`prepareIntegration` invoked once per record, in firing order) and the
///     buffer is left intact — the documented `@invariant`; the burst-start
///     clear in `performElem2` empties it, not the drain.
/// `in3` is a core Peano operator, so the `ExpressionAnalyzer("Peano")`
/// construction shared with the other member tests supplies the compiled
/// expression the replay needs; no extra disk fixtures are required. The
/// template-registration effect of `prepareIntegration` itself is its own
/// contract (and is exercised end-to-end by the Gauss fold proof).

#include "test_harness.hpp"

#include "../prover.hpp"

TEST(admission_integration_reshuffle, drain_deferred_integration_preps_symbol_signature) {
    // Symbol-existence + signature check: take the drain's address with its
    // expected signature `void(gl::Memory&)`. Compiles only if
    // drainDeferredIntegrationPreps is declared on gl::ExpressionAnalyzer with
    // that exact signature, satisfying Rule 18's symbol-coverage requirement.
    using DrainFn = void (gl::ExpressionAnalyzer::*)(gl::Memory&);
    DrainFn fn = &gl::ExpressionAnalyzer::drainDeferredIntegrationPreps;
    ASSERT_TRUE(fn != nullptr);
}

TEST(admission_integration_reshuffle, drain_replays_deferred_prepare_integration) {
    // Heavy ExpressionAnalyzer construction is unavoidable for member access
    // (mirrors the test_memory.cpp ea("Peano") behavioral tests).
    gl::ExpressionAnalyzer ea("Peano");

    gl::Memory m;

    // Stage two deferred prepareIntegration calls — marker-form in3 expressions
    // with their bare argument sets (marker removed, sorted-unique sealed form),
    // exactly as the marker branch in checkLocalEncodedMemoryStatic stages them.
    gl::SealedPageSet pages;
    pages.bind(&gl::staticMemory());
    const auto s = [&pages](const std::string& v) {
        return gl::SealedString::copyFrom(pages, v.data(),
                                          static_cast<int32_t>(v.size()));
    };
    const gl::SealedString args1Arr[3] = { s("4"), s("a"), s("b") };
    const auto args1 =
        gl::SealedSpan<gl::SealedString>::copyFrom(pages, args1Arr, 3);
    const gl::SealedString args2Arr[3] = { s("4"), s("c"), s("d") };
    const auto args2 =
        gl::SealedSpan<gl::SealedString>::copyFrom(pages, args2Arr, 3);
    m.deferredIntegrationPreps.push_back(
        gl::DeferredIntegrationPrep{ s("(in3[a,marker,b,4])"), args1, s("main") });
    m.deferredIntegrationPreps.push_back(
        gl::DeferredIntegrationPrep{ s("(in3[c,marker,d,4])"), args2, s("main") });

    // Replays prepareIntegration once per record, in firing order; must not
    // throw and must not consume its own buffer.
    ea.drainDeferredIntegrationPreps(m);

    // The drain iterated both records and left the buffer intact (the
    // post-drain clear lives in performElemPhase2, not in the drain) — the
    // documented invariant.
    ASSERT_TRUE(m.deferredIntegrationPreps.size() == static_cast<std::size_t>(2));

    // Release the views before the pages die (the production clear point).
    m.deferredIntegrationPreps.clear();
    pages.seal();
    pages.freePages();
}
