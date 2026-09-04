<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# A17 — below successor: `a < s(b) ⟺ a ≤ b` (human proof)

Two pool rows (18–19 of `files/shortcut/theorems/conjectures.txt`):

```text
(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](strictOrder[1,4,11,10])(preorder[1,4,11,9]))))
(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](preorder[1,4,11,9])(strictOrder[1,4,11,10]))))
```

Variables: 9 = b, 10 = s(b), 11 = a. Forward = row 18 (`a < s(b) → a ≤ b`),
backward = row 19 (`a ≤ b → a < s(b)`).

## Backward direction — PROVED (2026-08-05 first run)

From a ≤ b take the witness q with a + q = b; successor arithmetic lifts it
to a + s(q) = s(b) (the `+1` / associativity route through the integration
machinery), giving the ≤-part of `a < s(b)`; the ≠-part `a ≠ s(b)` follows
from the order facts. No case split needed.

## Forward direction — human proof (requires one case split)

1. **Premise** `a < s(b)`: by the `strictOrder` definition, ∃w∈N with
 a + w = s(b), and a ≠ s(b).
2. **Case split on w** by the predecessor theorem `w = 0 ∨ ∃m: s(m) = w`
 (corpus row 59; compiled `or1[w,zero,N,s]`).
3. **Case w = 0:** a + 0 = s(b) gives a = s(b) — contradicts a ≠ s(b);
 the branch is refuted (reductio, branch retires).
4. **Case w = s(m):** a + s(m) = s(b), so s(a + m) = s(b) (successor
 transport across `+`, corpus row 28), so a + m = b (successor
 injectivity), so a ≤ b with witness m. ∎

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
