#
from typing import List, Optional, Callable
import os
import itertools


def make_simple_facts_gauss(
        n: int,
        out_dir: str = "files/simple_facts",
        echo: bool = False,
        print_if: Optional[Callable[[str], bool]] = None,
        copies: int = 3,  # Ignored: We enforce hierarchy manually
        seq_val_limit: int = 1
) -> str:
    # ---------- helpers ----------
    def _mk_id(kind: str, v: int) -> str:
        return f"{kind}{v}"

    def _mk_in3(xk: str, a: int, yk: str, b: int, c: int, op: str, rc: str) -> str:
        return f"(in3[{_mk_id(xk, a)},{_mk_id(yk, b)},{_mk_id(rc, c)},{op}])"

    def mk_not(s: str) -> str:
        return "!" + s

    def distinct3(A: str, B: str, C: str) -> bool:
        return A != B and A != C and B != C

    def tri(num: int) -> int:
        return (num * (num + 1)) // 2

    def preorder_plus_true(x: int, y: int) -> bool:
        return y >= x

    def preorder_times_true(x: int, y: int) -> bool:
        if x == 0:
            return y == 0
        if x == 1:
            return 0 <= y <= n
        return (y % x == 0) and (y // x <= n)

    # ---------- Variable Logic ----------
    # Primary: i (Strictly required for in2, in3, fold)
    # Secondary: k, m (Always allowed, max 1 each)
    # Copy: j (Allowed only for 0, 1, 2, max 1)

    primary = 'i'

    def get_kinds(val: int) -> List[str]:
        # Always allow Primary (i) and Secondary (k, m)
        allowed = ['i', 'k', 'm']

        # 'j' restricted to small values
        if val <= 2:
            allowed.append('j')

        return allowed

    def get_vars_for_val(val: int) -> List[str]:
        allowed_ks = get_kinds(val)
        return [f"{k}{val}" for k in allowed_ks]

    # --- Validation Rules ---

    # Base Rule: Max 1 of each secondary/copy variable (Universal)
    def _base_check(kinds_list: List[str]) -> bool:
        if kinds_list.count('j') > 1: return False
        if kinds_list.count('k') > 1: return False
        if kinds_list.count('m') > 1: return False
        return True

    # Strict Rule: Must have 'i' AND base rules (For in2, in3, fold)
    def validate_strict(kinds_list: List[str]) -> bool:
        if 'i' not in kinds_list: return False
        return _base_check(kinds_list)

    # Weak Rule: No 'i' required, just base rules (For everything else)
    def validate_weak(kinds_list: List[str]) -> bool:
        return _base_check(kinds_list)

    def allow_succ(k_s: str, k_t: str) -> bool:
        return validate_strict([k_s, k_t])

    # --- Fact Accumulation Helpers ---

    # 1. SPECIAL: Adds 'm' copy if 'k' is present (For in2, in3, fold ONLY)
    def add_fact_with_m(target_list: List[str], atom: str):
        target_list.append(atom)
        if 'k' in atom:
            # Replace 'k' with 'm' (e.g., k0 -> m0)
            atom_m = atom.replace('k', 'm')
            target_list.append(atom_m)

    # 2. STANDARD: No 'm' generation
    def add_fact_simple(target_list: List[str], atom: str):
        target_list.append(atom)

    # ---------- sections ----------
    mult: List[str] = []
    add: List[str] = []
    succ: List[str] = []
    succ_neg: List[str] = []  # NEW: Negative Successor
    neq_addmul: List[str] = []
    pre_true: List[str] = []
    pre_neg_true: List[str] = []
    fold_true: List[str] = []
    fold_neg_true: List[str] = []
    ne_true: List[str] = []

    # New sections
    intervals: List[str] = []
    intervals_neg: List[str] = []
    sequences: List[str] = []
    sequences_neg: List[str] = []
    limit_sets: List[str] = []
    limit_seqs: List[str] = []

    # Explicit Non-Objects
    non_seq_sequences_neg: List[str] = []
    non_seq_limit_seqs: List[str] = []
    non_int_intervals_neg: List[str] = []
    non_int_limit_sets: List[str] = []

    if n >= 0:
        # ---- Addition (STRICT) ----
        for a in range(0, n + 1):
            for b in range(0, n + 1):
                s_ab = a + b
                if s_ab > n: continue

                for ka in get_kinds(a):
                    for kb in get_kinds(b):
                        # Result Truth
                        if validate_strict([ka, kb, primary]):
                            if distinct3(_mk_id(ka, a), _mk_id(kb, b), _mk_id(primary, s_ab)):
                                add.append(_mk_in3(ka, a, kb, b, s_ab, '+', primary))

                        # Result Alternative
                        for kr in get_kinds(s_ab):
                            if kr == primary: continue
                            if validate_strict([ka, kb, kr]):
                                if distinct3(_mk_id(ka, a), _mk_id(kb, b), _mk_id(kr, s_ab)):
                                    add.append(_mk_in3(ka, a, kb, b, s_ab, '+', kr))

        # ---- Multiplication (STRICT) ----
        for a in range(0, n + 1):
            for b in range(0, n + 1):
                p_ab = a * b
                if p_ab > n: continue

                for ka in get_kinds(a):
                    for kb in get_kinds(b):
                        if validate_strict([ka, kb, primary]):
                            if distinct3(_mk_id(ka, a), _mk_id(kb, b), _mk_id(primary, p_ab)):
                                mult.append(_mk_in3(ka, a, kb, b, p_ab, '*', primary))

                        for kr in get_kinds(p_ab):
                            if kr == primary: continue
                            if validate_strict([ka, kb, kr]):
                                if distinct3(_mk_id(ka, a), _mk_id(kb, b), _mk_id(kr, p_ab)):
                                    mult.append(_mk_in3(ka, a, kb, b, p_ab, '*', kr))

        # ---- Successor Positive (STRICT + m-Copy) ----
        for k in range(0, n):
            for ks in get_kinds(k):
                for kt in get_kinds(k + 1):
                    if allow_succ(ks, kt):
                        add_fact_with_m(succ, f"(in2[{_mk_id(ks, k)},{_mk_id(kt, k + 1)},s])")

        # ---- Successor Negative (STRICT + m-Copy) ----
        for k in range(0, n):
            true_succ = k + 1
            for p in range(0, n + 1):
                if p == true_succ: continue

                for ks in get_kinds(k):
                    for kp in get_kinds(p):
                        if allow_succ(ks, kp):
                            # Generate !(in2[ks, kp, s])
                            atom = f"!(in2[{_mk_id(ks, k)},{_mk_id(kp, p)},s])"
                            add_fact_with_m(succ_neg, atom)

        # ---- Negations +/* (STRICT + m-Copy) ----
        for a in range(0, n + 1):
            for b in range(0, n + 1):
                s_ab = a + b
                p_ab = a * b
                for p in range(0, n + 1):

                    if p != s_ab:
                        for ka in get_kinds(a):
                            for kb in get_kinds(b):
                                if validate_strict([ka, kb, primary]):
                                    if distinct3(_mk_id(ka, a), _mk_id(kb, b), _mk_id(primary, p)):
                                        add_fact_with_m(neq_addmul, mk_not(_mk_in3(ka, a, kb, b, p, '+', primary)))
                                for kr in get_kinds(p):
                                    if kr == primary: continue
                                    if validate_strict([ka, kb, kr]):
                                        if distinct3(_mk_id(ka, a), _mk_id(kb, b), _mk_id(kr, p)):
                                            add_fact_with_m(neq_addmul, mk_not(_mk_in3(ka, a, kb, b, p, '+', kr)))

                    if p != p_ab:
                        for ka in get_kinds(a):
                            for kb in get_kinds(b):
                                if validate_strict([ka, kb, primary]):
                                    if distinct3(_mk_id(ka, a), _mk_id(kb, b), _mk_id(primary, p)):
                                        add_fact_with_m(neq_addmul, mk_not(_mk_in3(ka, a, kb, b, p, '*', primary)))
                                for kr in get_kinds(p):
                                    if kr == primary: continue
                                    if validate_strict([ka, kb, kr]):
                                        if distinct3(_mk_id(ka, a), _mk_id(kb, b), _mk_id(kr, p)):
                                            add_fact_with_m(neq_addmul, mk_not(_mk_in3(ka, a, kb, b, p, '*', kr)))

        # ---- Fold (STRICT, Start j0, m-Copy) ----
        for m_idx in range(0, n + 1):
            Tm = tri(m_idx)
            for p_idx in range(0, n + 1):
                # RESTRICTION: Start is strictly 'j0'
                k0_kind = 'j'
                k0_var = 'j0'

                for km in get_kinds(m_idx):
                    for kp in get_kinds(p_idx):
                        if not validate_strict([k0_kind, km, kp]): continue

                        atom = f"(fold[N,s,+,id,{k0_var},{_mk_id(km, m_idx)},{_mk_id(kp, p_idx)}])"
                        if p_idx == Tm:
                            add_fact_with_m(fold_true, atom)
                        else:
                            add_fact_with_m(fold_neg_true, '!' + atom)

        # ---- Preorder (WEAK) ----
        for op, pred in (('+', preorder_plus_true), ('*', preorder_times_true)):
            for x in range(0, n + 1):
                for y in range(0, n + 1):
                    for kx in get_kinds(x):
                        for ky in get_kinds(y):
                            if not validate_weak([kx, ky]): continue

                            atom = f"(preorder[N,{op},{_mk_id(kx, x)},{_mk_id(ky, y)}])"
                            if pred(x, y):
                                pre_true.append(atom)
                            else:
                                pre_neg_true.append('!' + atom)

        # ---- Negated equality (WEAK) ----
        for x in range(0, n + 1):
            for y in range(0, n + 1):
                if x == y: continue
                for kx in get_kinds(x):
                    for ky in get_kinds(y):
                        if not validate_weak([kx, ky]): continue
                        ne_true.append(f"!(=[{_mk_id(kx, x)},{_mk_id(ky, y)}])")

        # ================= NEW FACTS (WEAK) =================

        # ---- Intervals (Strict Start j0) ----
        for n_arg in range(0, n + 1):
            for m_id in range(0, n + 1):
                starts = ['j0']
                ends = get_vars_for_val(n_arg)

                is_true = (n_arg == m_id)
                target_ids = [f"inter_{m_id}", f"copy_inter_{m_id}"]

                for s in starts:
                    for e in ends:
                        if not validate_weak([s[0], e[0]]): continue

                        for t_id in target_ids:
                            atom = f"(interval[N,+,{s},{e},{t_id}])"
                            if is_true:
                                intervals.append(atom)
                            else:
                                intervals_neg.append("!" + atom)

        # ---- Sequences (Strict Start j0) ----
        for n_arg in range(0, n + 1):
            for m_id in range(0, n + 1):
                for q_val in range(0, seq_val_limit + 1):
                    starts = ['j0']
                    ends = get_vars_for_val(n_arg)

                    is_true = (n_arg == m_id)
                    target_ids = [f"seq_{m_id}_{q_val}", f"copy_seq_{m_id}_{q_val}"]

                    for s in starts:
                        for e in ends:
                            if not validate_weak([s[0], e[0]]): continue

                            for t_id in target_ids:
                                atom = f"(sequence[N,+,{s},{e},{t_id}])"
                                if is_true:
                                    sequences.append(atom)
                                else:
                                    sequences_neg.append("!" + atom)

        # ---- LimitSet ----
        for p in range(0, n + 1):
            for q in range(0, n + 1):
                for x in range(0, n + 1):
                    limits = get_vars_for_val(x)

                    ids_p = [f"inter_{p}", f"copy_inter_{p}"]
                    ids_q = [f"inter_{q}", f"copy_inter_{q}"]

                    for l_var in limits:
                        if not validate_weak([l_var[0]]): continue

                        for ip in ids_p:
                            for iq in ids_q:
                                atom = f"(limitSet[N,+,{ip},{l_var},{iq}])"
                                if q == min(p, x): limit_sets.append(atom)

        # ---- LimitSequence ----
        for n1 in range(0, n + 1):
            for m1 in range(0, seq_val_limit + 1):
                for n2 in range(0, n + 1):
                    for m2 in range(0, seq_val_limit + 1):
                        for limit in range(0, n + 1):
                            limits = get_vars_for_val(limit)

                            ids_1 = [f"seq_{n1}_{m1}", f"copy_seq_{n1}_{m1}"]
                            ids_2 = [f"seq_{n2}_{m2}", f"copy_seq_{n2}_{m2}"]

                            is_true = (m1 == m2) and (n2 == limit) and (n1 >= limit)

                            for l_var in limits:
                                if not validate_weak([l_var[0]]): continue

                                for s1 in ids_1:
                                    for s2 in ids_2:
                                        atom = f"(limitSequence[N,+,{l_var},{s1},{s2}])"
                                        if is_true: limit_seqs.append(atom)

        # ================= EXPLICIT NON-SEQUENCES (WEAK) =================

        # 1. !(sequence[..., non_seq_m_q])
        for n_arg in range(0, n + 1):
            for m_id in range(0, n + 1):
                for q_val in range(0, seq_val_limit + 1):
                    starts = ['j0']
                    ends = get_vars_for_val(n_arg)
                    target_ids = [f"non_seq_{m_id}_{q_val}", f"copy_non_seq_{m_id}_{q_val}"]

                    for s in starts:
                        for e in ends:
                            if not validate_weak([s[0], e[0]]): continue
                            for t_id in target_ids:
                                atom = f"!(sequence[N,+,{s},{e},{t_id}])"
                                non_seq_sequences_neg.append(atom)

        # 2. (limitSequence[..., limit, non_seq_n_m, seq_limit_m])
        for n_len in range(0, n + 1):
            for q_val in range(0, seq_val_limit + 1):
                for limit in range(0, n + 1):
                    if limit <= n_len:
                        limits = get_vars_for_val(limit)

                        ids_non = [f"non_seq_{n_len}_{q_val}", f"copy_non_seq_{n_len}_{q_val}"]
                        ids_seq = [f"seq_{limit}_{q_val}", f"copy_seq_{limit}_{q_val}"]

                        for l_var in limits:
                            if not validate_weak([l_var[0]]): continue
                            for ns in ids_non:
                                for s in ids_seq:
                                    atom = f"(limitSequence[N,+,{l_var},{ns},{s}])"
                                    non_seq_limit_seqs.append(atom)

        # ================= EXPLICIT NON-INTERVALS (WEAK) =================

        # 1. !(interval[..., non_int_m])
        for m_id in range(2, n + 1):
            for n_arg in range(0, n + 1):
                starts = ['j0']
                ends = get_vars_for_val(n_arg)

                target_ids = [f"non_int_{m_id}", f"copy_non_int_{m_id}"]

                for s in starts:
                    for e in ends:
                        if not validate_weak([s[0], e[0]]): continue
                        for t_id in target_ids:
                            atom = f"!(interval[N,+,{s},{e},{t_id}])"
                            non_int_intervals_neg.append(atom)

        # 2. (limitSet[..., non_int_m, limit, inter_limit])
        for m_id in range(2, n + 1):
            for limit in range(0, n + 1):
                if limit <= m_id:
                    limits = get_vars_for_val(limit)

                    ids_non = [f"non_int_{m_id}", f"copy_non_int_{m_id}"]
                    ids_int = [f"inter_{limit}", f"copy_inter_{limit}"]

                    for l_var in limits:
                        if not validate_weak([l_var[0]]): continue
                        for ni in ids_non:
                            for i in ids_int:
                                atom = f"(limitSet[N,+,{ni},{l_var},{i}])"
                                non_int_limit_sets.append(atom)

    # ---------- write file ----------
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"simple_facts_gauss_{n}.txt")

    anchors: List[str] = ["(AnchorGauss[N,j0,s,+,*,j1,j2,id])"]

    with open(out_path, "w", encoding="utf-8") as f:
        # Anchor
        for line in anchors: f.write(line + "\n")
        # Blocks
        for s in mult: f.write(s + "\n")
        for s in add: f.write(s + "\n")
        for s in succ: f.write(s + "\n")
        for s in succ_neg: f.write(s + "\n")  # NEW
        for s in pre_true: f.write(s + "\n")
        for s in fold_true: f.write(s + "\n")
        for s in ne_true: f.write(s + "\n")
        for s in intervals: f.write(s + "\n")
        for s in sequences: f.write(s + "\n")
        for s in limit_sets: f.write(s + "\n")
        for s in limit_seqs: f.write(s + "\n")

        # Explicit Non-Objects Positive
        for s in non_seq_limit_seqs: f.write(s + "\n")
        for s in non_int_limit_sets: f.write(s + "\n")

        # Negations
        for s in neq_addmul: f.write(s + "\n")
        for s in pre_neg_true: f.write(s + "\n")
        for s in fold_neg_true: f.write(s + "\n")
        for s in intervals_neg: f.write(s + "\n")
        for s in sequences_neg: f.write(s + "\n")

        # Explicit Non-Objects Negative
        for s in non_seq_sequences_neg: f.write(s + "\n")
        for s in non_int_intervals_neg: f.write(s + "\n")

    if echo:
        print(f"Generated {len(intervals)} intervals, {len(sequences)} sequences, "
              f"{len(limit_sets)} limitSets, {len(limit_seqs)} limitSequences, "
              f"{len(non_seq_sequences_neg)} Non-Seq Neg, {len(non_int_intervals_neg)} Non-Int Neg.")

    return out_path