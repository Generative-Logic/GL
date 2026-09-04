<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# XXX

August was not necessarily the worst month in GL's history. It brought three deliveries: rungs A, B and C of the FTA shortlist, the Lean export, and the GPU port.

## Rungs

The first, simpler half of the FTA shortlist is covered. FTA itself is the last item on the list. We expect the second half to be much harder once branching meets finite sequences.

Many new proving techniques went into GL on the way, and some old, useless parts were removed. Roughly eight out of ten lemmas closed immediately. The rest needed deep rework: extensions, debugging, acceleration. GL grows.

## Lean

Here the question "to grind or not to grind" has a perfectly ambiguous meaning. For the FTA campaign the answer is to grind. For Lean itself, GL's export is grind-free.

Once GL's own verifier was in place, it uncovered at least 200 bugs of every flavor. The commit history on Zenodo and GitHub shows them. One bug still slipped through: a cyclic reference between theorems. The Lean export found it. Fixed. Long story short: external validation is a good thing, especially now that proof complexity grows exponentially with all the branching and proof strategies inside.

One nice side of the export: after the FTA shortcut closes (speculative), GL will automatically and deterministically construct an FTA proof from the axioms and the provided conjectures, all of them fully proved by GL during the process itself.

## GPU

MPU 0.1 has not found overwhelming appreciation so far. You might ask what the point of it is now. The first benefit has arrived: the GPU port of the hash engine. It needs static, deloadable memory in VRAM and a logic block split across as many threads as are available. That is basically 100% of what MPU 0.1 delivered.

The GPU already makes the engine 11 times faster, and the factor always climbs after several rounds of optimization: 100x is probable, 1000x is not impossible. I hope MPU 0.2, GPU Edition, meets more enthusiasm.

On the MPU roadmap: after the FTA shortlist closes (speculative) and FTA discovery closes (highly speculative), the next step is a logic block port to an FPGA: MPU 0.3, FPGA Edition.

On runtime: the hash engine is still the biggest consumer, at around 30%. Outside it there is a wide variety of runtime vermin: smaller chunks, each to be squashed on its own. By now this is a nasty routine. In sum, a full run can drop well below one minute within one or two months. This is not planned as one coordinated action. Runtime is not an issue now; problems will be solved as they arrive. Our focus is the FTA shortlist. That's it.

We want it. We get it.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
