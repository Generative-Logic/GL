// Generative Logic: A deterministic reasoning and knowledge generation engine.
// Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
// Dual-licensed under the GNU Affero General Public License v3 or later
// and a commercial license — see https://generative-logic.com/license.
// Contributions require CLA — see CONTRIBUTING.md.
#pragma once
#include <cstdint>
#include <chrono>
namespace gl {
    // Temporary, environment-enabled census; no proof decision reads it.
    struct FtaBlowupAudit {
        uint64_t calls{}, matches{}, directMatches{}, variants{}, directVariants{};
        uint64_t setupNs{}, probesNs{}, snapshotMembers{}, equalityCalls{}, equalityMembers{};
        uint64_t classStageNs[9]{};
        uint64_t originBCalls{}, originBComparisons{}, originBNs{};
        uint64_t originACalls{}, originAComparisons{}, originANs{};
        int burst{}, status{};
        bool disintegrationAllowed{};
    };
    extern thread_local FtaBlowupAudit* ftaBlowupAudit;
}
