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

namespace gl {

/// @brief Identify the calling worker for scratch-arena selection.
/// @details The thread-local value is independent of any prover instance.
/// A value of -1 names the existing single-threaded seam; callers retain
/// their arena-specific reserved-slot selection.
class WorkerContext {
public:
    /// @brief Read the calling thread's worker identity.
    /// @details Reads only thread-local state and allocates no storage.
    /// @return Worker index, or -1 for the single-threaded seam.
    static int coreId() { return coreId_; }

    /// @brief Select the calling thread's worker identity.
    /// @details Worker entry points retain their existing assignments;
    /// this operation neither allocates nor selects an arena itself.
    /// @param coreId Worker index or the existing -1 seam marker.
    /// @return None.
    static void setCoreId(int coreId) { coreId_ = coreId; }

    /// @brief Read whether the calling thread is in a parallel mutation phase.
    /// @details Admission uses this existing thread-local flag to defer ancestor writes.
    /// @return True while the caller is a phase-one or phase-three worker.
    static bool inParallelWorkerPhase() { return inParallelWorkerPhase_; }

    /// @brief Mark entry or exit of a parallel mutation worker.
    /// @details The scheduler retains its existing true/false assignment points.
    /// @param active Whether the current worker is inside the parallel phase.
    /// @return Nothing.
    static void setParallelWorkerPhase(bool active) { inParallelWorkerPhase_ = active; }

private:
    static thread_local int coreId_;
    static thread_local bool inParallelWorkerPhase_;
};

} // namespace gl
