<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Statement levels contract — maintainer directives for the crash-fix session

> **Status:** IMPLEMENTED (2026-08-09). The realized design —
> one admission door, row presence = known = admitted, the `{-1}` non-derived tier, the
> `known`/`registered` bits retired, the marker demoted to bookkeeping — is recorded in
> [`40_decisions.md` D-264](40_decisions.md#d-264) and
> [`30_invariants.md` I-182](30_invariants.md#i-182).
> The B8 Part-1 campaign resumes on top.
> **Origin:** the B8 known-without-levels incident — full diagnosis in
> [`../fta_ladder/B8/current_proof_state.md`](../fta_ladder/B8/current_proof_state.md)
> (2026-08-09 section).

## The maintainer's directives (2026-08-09, verbatim intent)

1. **Every expression gets a levels set.** No statement may be visible to any reader while
 lacking its levels.
2. **The new check verifies that the levels set is NON-EMPTY — never just that the key (or
 row) exists.** An empty or absent levels set at a visible statement is the failure state.

Everything else — container layout, writer wiring, gate handling, how the vacuous
full-disintegration marker path and the equivalence-filter refusal are reconciled with the
contract — is the fix session's to design. No structure is prescribed here.

## What the fix session inherits

- The proven defect chain (two deterministic trap runs,  /
 `run_B8_p1_trapB.log`): `addStatement` registers → the equivalence filter refuses
 levels + `known` → the full-disintegration marker upsert (vacuously true for atomics)
 promotes the row to `known` anyway → `ordisMerge`'s refutation probe needs the levels and
 asserts.
- The Rule-30 traps still in tree (identity print at the assert;
 writer traps in `addStatement`, the equivalence-class commit, and the marker upsert) — to
 be removed by whichever session lands the verified fix.
- The `ordisMerge` assert stands untouched (Rule 19 / I-19).

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
