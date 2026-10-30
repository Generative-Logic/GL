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
#include "../memory_infra/temporary_witness_store.hpp"
namespace gl {
namespace tests { struct RuntimeSupportFixture; }
class CaseProcessor;
class DisintegrationProcessor;
class EqualityPropagation;
class Prover;
struct ShotFrame;

        /// @brief One disintegration frame's witness metadata and traversal state.
        /// @details Discovery records temporary slots, their source existences and
        /// enclosing slots in structural order. Publication visits the same order,
        /// using admitted permanent names and skipping unresolved subtrees.
        /// @invariant The frame and every expanded instruction end with this call.
        /// @see TemporaryWitnessStore, NewVarStore, disintegrateExprCore2.
        class WitnessMetaStore : public TemporaryWitnessStore {
private:
friend class CaseProcessor;
friend class DisintegrationProcessor;
friend class EqualityPropagation;
friend class Prover;
friend struct tests::RuntimeSupportFixture;

            bool publishing = false;
            int32_t nextSlot = 1;
            int32_t integrationDepth = 0;
            bool forceIntegration = false;
            const uint8_t* trialSlots = nullptr;
            int32_t trialPass = 0;
            // The slots an induction pass leaves to the other catches of its
            // subtree (D-566); null outside such a
            // pass.
            const uint8_t* passExcluded = nullptr;
            // The element of the innermost enclosing existence whose expansion
            // is being discovered (-1 outside any): recorded as a nested slot's
            // parent element (D-529).
            int32_t currentParentElement = -1;
            // The innermost disintegration package applied along the current
            // witness path, or null (I-333).
            ShotFrame* shotFrame = nullptr;

public:

            /// @brief Bind one disintegration's witness metadata to scratch.
            /// @details Each expansion owns its temporary namespace and never
            /// publishes its tokens to another expansion or persistent state.
            /// @param arena Container scratch storage for this expansion.
            /// @param assignedDepth Shared integration depth for every nested slot.
            /// @param forced Whether existing forced routes select forced witness names.
            /// @return An empty discovery frame.
            explicit WitnessMetaStore(LbArena& arena, int32_t assignedDepth = 0, bool forced = false)
                : TemporaryWitnessStore(arena, 0), integrationDepth(assignedDepth), forceIntegration(forced) {
                assert(assignedDepth >= 0);
            }

            /// @brief Read a witness token in zero-based discovery order.
            /// @details The view lasts until the next metadata mutation.
            /// @param index Zero-based witness position.
            /// @return Its temporary token.
            StrSpan witnessAt(int32_t index) const { return token(index + 1); }

            /// @brief Read the existence that spawned a witness.
            /// @details The source belongs to this expansion's scratch storage.
            /// @param index Zero-based witness position.
            /// @return Its instantiated source existence.
            StrSpan instanceAt(int32_t index) const { return source(index + 1); }

            /// @brief Read the enclosing witness token when present.
            /// @details A top-level witness has no enclosing witness.
            /// @param index Zero-based witness position.
            /// @return Parent token, or an empty span at the top level.
            StrSpan parentAt(int32_t index) const {
                const int32_t parent = slot(index + 1).parent;
                return parent == 0 ? StrSpan() : token(parent);
            }

            /// @brief Restrict publication around integration trial witnesses.
            ///
            /// @details
            /// A witness on trial (D-481) and the
            /// new names minted with it below it are materialized in the
            /// parent's expansion, but its subtree is published only at its own
            /// trial scope. The parent's pass runs with @p trialSlotFlags and no
            /// trial pass: every flagged slot is skipped, and with it its
            /// subtree. A trial scope's pass names its slot in @p trialSlot:
            /// only that slot's subtree is published. A null flag array and a
            /// zero slot restore unrestricted publication.
            ///
            /// @param trialSlotFlags One byte per slot (index 1..count, index 0
            ///                       unused), nonzero for a witness on trial;
            ///                       null when no witness is on trial. Owned by
            ///                       the caller and live across the passes.
            /// @param trialSlot      The slot whose trial scope is being
            ///                       published, or zero for the parent's pass.
            /// @return Void.
            /// @see publishes, DisintegrationProcessor::disintegrateExpr2.
            void setPublicationFilter(const uint8_t* trialSlotFlags, int32_t trialSlot) {
                assert(trialSlot >= 0 && trialSlot <= count());
                assert((trialSlot == 0 || (trialSlotFlags != nullptr && trialSlotFlags[trialSlot] != 0))
                    && "setPublicationFilter: a trial pass names a witness on trial");
                trialSlots = trialSlotFlags;
                trialPass = trialSlot;
            }

            /// @brief Exclude the other catches of an induction pass's subtree
            ///        (D-566).
            ///
            /// @details
            /// One expansion can admit several witnesses through induction
            /// demands below one subtree top. Each opens its own induction
            /// scope, and the pass publishing that scope leaves out the
            /// branches of the other catches; the flagged slots are skipped
            /// with their subtrees. Null clears the exclusion.
            ///
            /// @param excluded One byte per slot (index 1..count, index 0
            ///                 unused), nonzero for a skipped slot; null for
            ///                 none. Owned by the caller and live across the
            ///                 pass.
            /// @return Void.
            /// @see publishes, DisintegrationProcessor::disintegrateExpr2.
            void setPassExclusions(const uint8_t* excluded) {
                passExcluded = excluded;
            }

            /// @brief Whether the current publication pass publishes a
            ///        materialized slot's subtree.
            ///
            /// @details
            /// Under a trial or induction pass the named slot, the slots below
            /// it and the slots enclosing it are visited: the enclosing ones
            /// only lead the walk down to the named slot, since a top's
            /// enclosing slots are published at the parent scope, which the
            /// pass's scope sees; the pass drops what the parent published.
            /// Slots the pass excludes (`setPassExclusions`) are skipped.
            /// Otherwise every slot except a witness on trial or an induction
            /// pass top is (a slot below one is never reached, its enclosing
            /// witness being skipped). The caller still skips unmaterialized
            /// slots, so a trial pass publishes exactly the names minted for
            /// the trial.
            ///
            /// @param id One-based slot identifier.
            /// @return True iff the slot's subtree is published in this pass.
            /// @invariant Pure; allocates nothing.
            /// @see setPublicationFilter, setPassExclusions.
            bool publishes(int32_t id) const {
                assert(id > 0 && id <= count());
                if (passExcluded != nullptr && passExcluded[id] != 0) return false;
                if (trialPass != 0) {
                    if (id >= trialPass && id < slot(trialPass).subtreeEnd) return true;
                    for (int32_t a = slot(trialPass).parent; a != 0; a = slot(a).parent)
                        if (a == id) return true;
                    return false;
                }
                return trialSlots == nullptr || trialSlots[id] == 0;
            }
        };
}
