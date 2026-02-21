# simple_facts_peano.py
# Generates the same MPL "facts" as analyze_expression.cpp::makeMPLTables
# and writes them to files/simple_facts/simple_facts_peano_{n}.txt.
# Designed to be imported and called from run_modes.py (not a standalone script).

from typing import List, Optional, Callable
import os

def make_simple_facts_peano(
    n: int,
    out_dir: str = "files/simple_facts",
    echo: bool = False,
    print_if: Optional[Callable[[str], bool]] = None,
) -> str:
    """
    Generate Peano simple facts (addition, multiplication, successor, negated (+,*) inequalities)
    exactly as in analyze_expression.cpp::ExpressionAnalyzer::makeMPLTables and save to disk.

    Args:
        n: upper bound for the small domain (>= 0).
        out_dir: output directory (will be created if missing).
        echo: if True, prints facts (with section headers) like the C++ `print` branch.
        print_if: optional predicate to filter what gets printed when echo=True.

    Returns:
        Path to the written file: files/simple_facts/simple_facts_peano_{n}.txt
    """
    # Helper builders that match C++ formatting (no spaces, exact bracket/paren placement)
    def _mk_id(kind: str, v: int) -> str:
        # kind ∈ {'i','j'} → "i0", "j3", ...
        return f"{kind}{v}"

    def _mk_in3_ijr(xk: str, a: int, yk: str, b: int, c: int, op: str, rc: str) -> str:
        # (in3[xk a, yk b, rc c, op])
        return f"(in3[{_mk_id(xk, a)},{_mk_id(yk, b)},{_mk_id(rc, c)},{op}])"

    def _mk_succ(a: int) -> str:
        # (in2[i{a},i{a+1},s]) — we allow i/j mixing below via kinds loop
        return f"(in2[i{a},i{a+1},s])"

    def mk_not(s: str) -> str:
        return "!" + s

    def distinct3(A: str, B: str, C: str) -> bool:
        return A != B and A != C and B != C

    def allow_ijr(ka: str, kb: str, rc: str) -> bool:
        # at most one 'j' across the two inputs and the result channel
        return ((ka == 'j') + (kb == 'j') + (rc == 'j')) <= 1

    mult: List[str] = []
    add:  List[str] = []
    succ: List[str] = []
    neq:  List[str] = []

    if n < 0:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"simple_facts_peano_{n}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("(AnchorPeano[N,j0,s,+,*,j1])\n")
        return out_path

    kinds = ('i', 'j')

    # ---- Addition (order matches C++ nesting) ----
    for a in range(0, n + 1):
        for b in range(0, n + 1):
            s_ab = a + b
            if s_ab > n:  # respect domain bound
                continue
            for ka in kinds:
                for kb in kinds:
                    A = _mk_id(ka, a)
                    B = _mk_id(kb, b)
                    # result in 'i'
                    if allow_ijr(ka, kb, 'i'):
                        Ri = _mk_id('i', s_ab)
                        if distinct3(A, B, Ri):
                            add.append(_mk_in3_ijr(ka, a, kb, b, s_ab, '+', 'i'))
                    # result in 'j'
                    if allow_ijr(ka, kb, 'j'):
                        Rj = _mk_id('j', s_ab)
                        if distinct3(A, B, Rj):
                            add.append(_mk_in3_ijr(ka, a, kb, b, s_ab, '+', 'j'))

    # ---- Multiplication ----
    for a in range(0, n + 1):
        for b in range(0, n + 1):
            p_ab = a * b
            if p_ab > n:
                continue
            for ka in kinds:
                for kb in kinds:
                    A = _mk_id(ka, a)
                    B = _mk_id(kb, b)
                    # result in 'i'
                    if allow_ijr(ka, kb, 'i'):
                        Ri = _mk_id('i', p_ab)
                        if distinct3(A, B, Ri):
                            mult.append(_mk_in3_ijr(ka, a, kb, b, p_ab, '*', 'i'))
                    # result in 'j'
                    if allow_ijr(ka, kb, 'j'):
                        Rj = _mk_id('j', p_ab)
                        if distinct3(A, B, Rj):
                            mult.append(_mk_in3_ijr(ka, a, kb, b, p_ab, '*', 'j'))

    # ---- Succession ----
    for k in range(0, n):
        for ks in kinds:
            for kt in kinds:
                if ((ks == 'j') + (kt == 'j')) <= 1:
                    succ.append(f"(in2[{_mk_id(ks, k)},{_mk_id(kt, k + 1)},s])")

    # ---- Inequalities: negated + and * (true statements where p != sum/prod) ----
    for a in range(0, n + 1):
        for b in range(0, n + 1):
            s_ab = a + b
            p_ab = a * b
            for p in range(0, n + 1):
                for ka in kinds:
                    for kb in kinds:
                        A = _mk_id(ka, a)
                        B = _mk_id(kb, b)
                        # + inequality
                        if p != s_ab:
                            if allow_ijr(ka, kb, 'i'):
                                Ri = _mk_id('i', p)
                                if distinct3(A, B, Ri):
                                    neq.append(mk_not(_mk_in3_ijr(ka, a, kb, b, p, '+', 'i')))
                            if allow_ijr(ka, kb, 'j'):
                                Rj = _mk_id('j', p)
                                if distinct3(A, B, Rj):
                                    neq.append(mk_not(_mk_in3_ijr(ka, a, kb, b, p, '+', 'j')))
                        # * inequality
                        if p != p_ab:
                            if allow_ijr(ka, kb, 'i'):
                                Ri = _mk_id('i', p)
                                if distinct3(A, B, Ri):
                                    neq.append(mk_not(_mk_in3_ijr(ka, a, kb, b, p, '*', 'i')))
                            if allow_ijr(ka, kb, 'j'):
                                Rj = _mk_id('j', p)
                                if distinct3(A, B, Rj):
                                    neq.append(mk_not(_mk_in3_ijr(ka, a, kb, b, p, '*', 'j')))

    # Write in the same section order as the C++ print listing:
    # Multiplication, Addition, Succession, Inequalities
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"simple_facts_peano_{n}.txt")

    # --- ONLY j for anchor ---
    anchors = [
        "(AnchorPeano[N,j0,s,+,*,j1])"
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        for line in anchors:
            f.write(line + "\n")
        for s in mult:
            f.write(s + "\n")
        for s in add:
            f.write(s + "\n")
        for s in succ:
            f.write(s + "\n")
        for s in neq:
            f.write(s + "\n")

    if echo:
        def _print_list(title: str, v: List[str]) -> None:
            print(f"{title} ({len(v)})")
            if print_if is None:
                for s in v:
                    print(f"  {s}")
            else:
                for s in v:
                    if print_if(s):
                        print(f"  {s}")

        _print_list("Multiplication", mult)
        _print_list("Addition", add)
        _print_list("Succession", succ)
        _print_list("Inequalities (negated +,*)", neq)

    return out_path
