<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# How the reference proof reaches `p′ + l`, and where rung1 stops

Goal of this note: take the rung-1 forward theorem
`EnumerationSet2 ⟹ interval`, look at the one hard branch (`p = 1` case of
the backward conjunct), and write down — in dependency order — every proof
line the **complete** proof uses to build the sum `p′ + l`. Then mark the
first line whose premises are all already present in the broken `rung1`
run but whose result is never produced.

- **Reference (complete):** the reference branch, raw proof
 graph `files/incubator/raw_proof_graph/1218_direct_proof.txt` (83 lines).
 Anchor `AnchorIncubator[1..14]`, set variable `M = 15`.
- **Broken:** the broken branch, dump
  (ES2 hypothesis LB, fixpoint at burst 50,
 `toBeProved = 4`). Anchor `AnchorIncubator3[1..9]`, `M = 10`.

Anchor slots `1..7` are identical on both branches (`N=1, 0=2, s=3, +=4,
*=5, 1=6, 2=7`), so every expression below has the same shape on both —
only the per-block variable suffixes differ.

## Cast (same role, different suffix per branch)

| role | meaning | ref name | rung1 name |
|---|---|---|---|
| `p` | the element of `[0,1]` under test | `repl_lev_1_2` | `repl_lev_1_2` |
| `p′` | predecessor of `p` (`s(p′)=p`) | `it_0_lev_1_598` | `it_0_lev_1_344` |
| `l` | the value with `p + l = 1` (from `p ≤ 1`) | `int_lev_1_7` | `int_lev_1_7` |
| `w` | the value of the sum `p′ + l` | `it_1_lev_1_652` | — (never created) |

Branch-A scope (the `p = 1` case): `…(implication22[1,4,2,6,M])_boundary_orint_(or0[2,p,6])_((=[6,p]))`.

Raw-graph line format: `result ⟶ scope ⟶ tag ⟶ (premise, scope)…`.

## Where `p′ + l` is born — the genesis sub-tree

The sum is introduced by closure of `+` over `ℕ`, then the existential is
opened to expose the actual addition fact:

```
ref L62  existence1[1,l,p′,4]                     "l + p′ exists in ℕ"
            tag implication  (rule: a∈ℕ ∧ b∈ℕ ⟹ existence1(N,a,b,+))
            premise  in[l,1]    (l ∈ ℕ)           @ implication22
            premise  in[p′,1]   (p′ ∈ ℕ)          @ Branch A
                  │
ref L61  !(>[w](in[w,1])!(in3[l,p′,w,4]))         expansion of the existence
                  │
ref L60  in3[l,p′,w,4]                            "l + p′ = w"     ← the sum, concretely
            tag disintegration
```

`ref L60` is the first line that actually states `p′ + l` as an addition
relation. Everything afterwards is rewriting and arithmetic on it.

## What the reference does with `p′ + l` (downstream, for context)

```
in3[l,p′,w,4]  (l+p′=w, L60)
 ├─ commute  → in3[p′,l,w,4]            (p′+l=w)                         (via equality1, L58)
 ├─ L76 in3[l,p,6,4] (l+p=1, commute of p+l=1) ┐
 ├─ L38 in2[p′,p,3]  (s(p′)=p)                 ├─ L73 in2[w,6,3]  "s(w)=1"   (P2 combine)
 └─ L68 in[p′,1]     (p′∈ℕ)                    ┘
                                  │
              L69 =[w,2]  "w = 0"            (P3 injectivity: s(w)=1=s(0) ⟹ w=0)
                                  │
              L58 in3[p′,l,2,4]  "p′ + l = 0" (equality1: substitute w=0 into p′+l=w)
                                  │
              L57 =[2,p′]  "0 = p′"           (P6 cancellation: p′+l=0 ∧ l∈ℕ ⟹ 0=p′)
                                  │
            then p = s(p′) = s(0) = 1  →  Branch-A head (=[6,p]) closes.
```

## The first line that does not fire in rung1

**Line: the disintegration `existence1[1,l,p′,4] ⟶ in3[l,p′,w,4]` (ref
L62→L61→L60).**

State in the `rung1` fixpoint dump (Branch A, burst 50):

| ref line | expression | present in rung1? |
|---|---|---|
| L62 input | `in[int_lev_1_7,1]` (`l ∈ ℕ`) | **yes** — es `[13]` @ implication22 |
| L62 input | `in[it_0_lev_1_344,1]` (`p′ ∈ ℕ`) | **yes** — es `[1826]` @ Branch A |
| L62 result | `existence1[1,int_lev_1_7,it_0_lev_1_344,4]` (`l+p′` exists) | **yes** — es `[1850]` @ Branch A |
| **L60 result** | **`in3[int_lev_1_7,it_0_lev_1_344,·,4]`** (`l+p′ = w`) | **NO — 0 occurrences** |

So the existence-of-the-sum reaches Branch A with both of its premises
live, but it is never opened into the concrete addition fact `l + p′ = w`.
Because that `in3` never appears, the whole chain above it is starved:
no `s(w)=1`, no `w=0`, no `p′+l=0`, no `p′=0`, and the `p = 1` branch
head stays open — `toBeProved` freezes at 4.

For contrast, the `in3` addition facts that *do* live at Branch A in
rung1 all pair `p′` with an **anchor constant** only
(`in3[p′,6,·]`=`p′+1`, `in3[p′,7,·]`=`p′+2`, `in3[p′,2,p′]`=`p′+0`);
`p′` is never summed with another derived value such as `l`. The
existence-to-`in3` disintegration is what is missing for the
derived-value sum.

**Next:** find why the disintegration of `existence1[1,l,p′,4]` does not
deposit `in3[l,p′,w,4]` at Branch A even though the existence is present —
i.e. why opening this particular existential (a sum of two derived values
inside an OR branch) is blocked. (Deferred — to investigate separately.)

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
