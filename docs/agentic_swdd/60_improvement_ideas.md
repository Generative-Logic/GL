<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Improvement ideas `[DRAFT]`

> Maintainer ideas that are not yet designed or approved for implementation. Each entry records the idea in the maintainer's words, the evidence that motivated it, what it would and would not solve, and the open questions a design must answer. An entry becomes a decision ([`40_decisions.md`](40_decisions.md)) only when the maintainer approves a design (Rule 8); then it moves there and leaves a pointer here.

---

## Canonical goals: the canonical form as key, the original as value

**Idea (maintainer, 2026-10-09).** "a goal is key value pair. Key gets canonicalized each time and value is original. If a goal is reached as canonical its original is released."

**Today.** A pending goal in `Memory::toBeProved` is matched by its exact text. Class changes rewrite it only partially: `sanitizeToBeProved` ([`20_core_concepts/05_equivalence_classes.md`](20_core_concepts/05_equivalence_classes.md), [D-67](40_decisions.md#d-67)) replaces forced, integration and algebraic arguments by a strictly preferred special peer, and leaves ordinary names alone. A statement that is equal to a goal only up to the scope's classes over ordinary names does not discharge it.

**The idea in detail.** Each goal keeps its original text as the value (the proof history and the theorem cite it) and a canonical key: every argument replaced by the canonical representative of its class at the goal's scope, as the deposit door canonicalizes statements. The key is recomputed whenever the complete class view of the goal's scope changes. A statement arriving in canonical form discharges every goal whose key equals it, and the goal's original is released (proved). The classes differ from scope to scope, so a goal visible at several scopes keeps one key per scope.

**What it solves.** A goal stays open although an equal statement up to classes is known: the goal names one member of a class and the proof reaches another.

**What it does not solve.** Two values the block never sets equal. In reverse F3 (see the next entry and [D-506](40_decisions.md#d-506)) the reached fold carries the value 13 and the needed one the value X; 13 and X are in different classes, so no canonical key makes the goal reachable.

**Open questions.** Where the key lives (beside `intToBeProved`, or as its key with the original as payload); how the history row of a discharge cites the canonical statement and the original goal (an `equality1` step per class merge, as for compacts); whether OR-branch release, contradiction scopes and the induction blocks, which also probe goals, read the key; the cost of recomputing keys at every class change.

---

## Accumulators: a definition and the lemma that two accumulators of one fold are equal

**Idea (maintainer, 2026-10-09).** "at most a theorem is needed and a new definition: A is accumulator of the fold iff there is an interval and so on. wld it help to have it in other proofs as accelerator. I mean theorem saying two accumulators of one fold r equal."

**Background.** `fold[N,s,+,op,f,n,m,p]` ([`files/definitions/fold.mpl`](../../files/definitions/fold.mpl)) states that there are an interval M = [n..m] and a function g: M → N, the accumulator, with g(n) = f(n), g(m) = p and g(j) = op(g(i), f(j)) for every successor j of i in M. The accumulator is bound inside the fold, so two folds over the same sequence and interval bring two accumulators that nothing identifies.

**Motivating case.** Reverse F3 (FTA theorem 145, supplied row 87), induction step block `(in2[rec0,9,3])`. A is the accumulator of the shorter fold premise (value 13), B the accumulator of the longer one, X = B(9), and C is B restricted to [i0..9]. A and C are both accumulators of the fold of the restricted sequence V10 over [i0..9], but the block holds no statement A = C, so the fold with the value 13 and the fold with the value X are unrelated, and 13 = X follows only from the induction hypothesis applied through the C-chain. When the finished-root wipe removed the C-chain, the goal could not be reached ([D-506](40_decisions.md#d-506)).

**The idea in detail.** A definition `accumulator[N,s,+,op,f,n,m,M,g]`: M is the interval [n..m], g is a function from M to N with g(n) = f(n) and g(j) = op(g(i), f(j)) for every successor j of i in M (the fold's body without the value). A supplied theorem: two accumulators of one sequence over one interval are equal, conclusion `(=[g,h])`; with it, equal folds over one sequence and interval share their values (13 = X in the case above), and the integration of one fold value would serve every later demand for the same fold.

**Open questions.** Whether `fold` is restated through `accumulator` (one definition, the theorem applies directly) or the lemma is stated over folds (no new definition, `fold[...,p] ∧ fold[...,q] → (=[p,q])`, which gives the value equality without the function equality); how GL proves the lemma (an induction over the interval, likely its own FTA-ladder rung); which other FTA proofs already re-derive fold values twice and would get faster (the restriction and append theorems F2-F8 are candidates); whether the lemma broadcast makes integration of the second fold value redundant early enough to save runtime, or only adds rows.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
