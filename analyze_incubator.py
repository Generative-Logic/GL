"""
Analyze incubator proved_theorems.txt vs theorems.txt (conjectures).
Lists positive/negative proved theorems and missing conjectures.
"""
import re
import sys
from collections import defaultdict

THEOREMS_FILE = "files/theorems_incubator/theorems.txt"
PROVED_FILE = "files/theorems_incubator/proved_theorems.txt"

ANCHOR_ARGS = {
    "1": "N", "2": "i0", "3": "s", "4": "+", "5": "*",
    "6": "i1", "7": "i2", "8": "id", "9": "i3", "10": "i4", "11": "i5"
}

# Integer values for anchor args
VAL = {"2": 0, "6": 1, "7": 2, "9": 3, "10": 4, "11": 5}
OP_SYM = {"4": "+", "5": "*"}
SUCC_SYM = "3"  # successor


def human_arith(head_str):
    """Convert heads to human-readable arithmetic."""
    m3 = re.match(r'\(in3\[(\d+),(\d+),(\d+),(\d+)\]\)', head_str)
    if m3:
        a, b, c, op = m3.group(1), m3.group(2), m3.group(3), m3.group(4)
        if a in VAL and b in VAL and c in VAL and op in OP_SYM:
            return f"{VAL[a]}{OP_SYM[op]}{VAL[b]}={VAL[c]}"
    m2 = re.match(r'\(in2\[(\d+),(\d+),(\d+)\]\)', head_str)
    if m2:
        a, b, op = m2.group(1), m2.group(2), m2.group(3)
        if op == SUCC_SYM and a in VAL and b in VAL:
            return f"s({VAL[a]})={VAL[b]}"
        if op == "8" and a in VAL and b in VAL:  # identity
            return f"id({VAL[a]})={VAL[b]}"
    meq = re.match(r'\(=\[(\d+),(\d+)\]\)', head_str)
    if meq:
        a, b = meq.group(1), meq.group(2)
        if a in VAL and b in VAL:
            return f"{VAL[a]}={VAL[b]}"
    mpo = re.match(r'\(preorder\[\d+,(\d+),(\d+),(\d+)\]\)', head_str)
    if mpo:
        op, a, b = mpo.group(1), mpo.group(2), mpo.group(3)
        if op == "4" and a in VAL and b in VAL:
            return f"{VAL[a]}<={VAL[b]}"
    mf = re.match(r'\(fold\[\d+,\d+,\d+,\d+,(\d+),(\d+),(\d+)\]\)', head_str)
    if mf:
        n, m, p = mf.group(1), mf.group(2), mf.group(3)
        if n in VAL and m in VAL and p in VAL:
            return f"sum(0..{VAL[m]})={VAL[p]}"
    return ""


def is_true_fact(head_str):
    """Determine if a head expression is arithmetically true, false, or unknown."""
    m3 = re.match(r'\(in3\[(\d+),(\d+),(\d+),(\d+)\]\)', head_str)
    if m3:
        a, b, c, op = m3.group(1), m3.group(2), m3.group(3), m3.group(4)
        if a not in VAL or b not in VAL or c not in VAL:
            return None
        va, vb, vc = VAL[a], VAL[b], VAL[c]
        if op == "4":    return va + vb == vc
        elif op == "5":  return va * vb == vc
        return None
    m2 = re.match(r'\(in2\[(\d+),(\d+),(\d+)\]\)', head_str)
    if m2:
        a, b, op = m2.group(1), m2.group(2), m2.group(3)
        if a not in VAL or b not in VAL:
            return None
        va, vb = VAL[a], VAL[b]
        if op == SUCC_SYM:  return va + 1 == vb
        if op == "8":       return va == vb  # identity
        return None
    meq = re.match(r'\(=\[(\d+),(\d+)\]\)', head_str)
    if meq:
        a, b = meq.group(1), meq.group(2)
        if a in VAL and b in VAL:
            return VAL[a] == VAL[b]
        return None
    mpo = re.match(r'\(preorder\[\d+,(\d+),(\d+),(\d+)\]\)', head_str)
    if mpo:
        op, a, b = mpo.group(1), mpo.group(2), mpo.group(3)
        if op == "4" and a in VAL and b in VAL:
            return VAL[a] <= VAL[b]
        return None
    mf = re.match(r'\(fold\[\d+,\d+,\d+,\d+,(\d+),(\d+),(\d+)\]\)', head_str)
    if mf:
        n, m, p = mf.group(1), mf.group(2), mf.group(3)
        if n in VAL and m in VAL and p in VAL:
            vn, vm, vp = VAL[n], VAL[m], VAL[p]
            # fold(N,s,+,id,n,m,p) = sum from i=n to i=m of id(i) = n+...+m
            return sum(range(vn, vm + 1)) == vp
        return None
    return None


def human_readable(expr):
    """Replace anchor arg numbers with readable names in the head expression."""
    # Extract the head part (after anchor)
    m = re.match(r'\(>[^)]*\]\)(.+)\)$', expr)
    if not m:
        return expr

    head = expr
    for num, name in sorted(ANCHOR_ARGS.items(), key=lambda x: -int(x[0])):
        # Replace standalone numbers in brackets, being careful with multi-digit
        head = re.sub(r'(?<=[,\[])\b' + num + r'\b(?=[,\]])', name, head)
    return head


def back_reformulate(conjecture):
    """
    Given a conjecture like:
      (>[2,4](Anchor[...])(>[11](in3[2,2,11,4])(=[11,2])))
    produce the back-reformulated form:
      (>[2,4](Anchor[...])(in3[2,2,2,4]))
    For equality conjectures like (>[2,10](Anchor[...])(=[2,10])):
      return (=[2,10]) as the head (no bound var to substitute).
    """
    # Match operator-equality pattern
    m = re.match(
        r'(\(>[^\]]+\]\(AnchorIncubator\[[^\]]+\]\))'  # prefix: (>[vars](Anchor[...])
        r'\(>\[(\d+)\]'                                  # bound var
        r'(\([^)]+\[[^\]]+\]\))'                        # operator expr
        r'\(=\[\2,(\d+)\]\)'                            # equality
        r'\)',                                           # close inner >
        conjecture
    )
    if m:
        prefix = m.group(1)
        bound_var = m.group(2)
        op_expr = m.group(3)
        eq_target = m.group(4)
        # Substitute bound_var with eq_target in op_expr
        reformed_op = op_expr.replace(
            ',' + bound_var + ',', ',' + eq_target + ','
        ).replace(
            ',' + bound_var + ']', ',' + eq_target + ']'
        ).replace(
            '[' + bound_var + ',', '[' + eq_target + ','
        )
        return prefix + reformed_op + ')'
    return None


def extract_vars(expr):
    """Extract the >[...] variable list."""
    m = re.match(r'\(>\[([^\]]+)\]', expr)
    return m.group(1) if m else ""


def extract_head_type(expr):
    """Extract the head expression type (in2, in3, =, etc.)."""
    # For proved: look for the operator/relation after anchor
    # Positive: ...(op[...]))
    # Negative: ...!(op[...]))
    neg = '!' in expr
    # Find the last expression
    if neg:
        m = re.search(r'!\(([a-z0-9=]+)\[', expr)
    else:
        # Skip the anchor, find the head
        parts = expr.split('AnchorIncubator[')
        if len(parts) > 1:
            rest = parts[1]
            m = re.search(r'\(([a-z0-9=]+)\[', rest)
        else:
            m = None
    return m.group(1) if m else "?"


def extract_head(expr):
    """Extract the full head expression (after anchor), stripping negation."""
    neg = '!' in expr
    parts = expr.split('])')
    if len(parts) >= 2:
        head = parts[1].strip(')')
        if head.startswith('!'):
            head = head[1:]
        return head
    return expr


def sort_key(expr):
    """Sort by: head type, number of vars, vars themselves."""
    head_type = extract_head_type(expr)
    vars_str = extract_vars(expr)
    vars_list = vars_str.split(',') if vars_str else []
    num_vars = len(vars_list)
    vars_tuple = tuple(int(v) for v in vars_list if v.isdigit())
    type_order = {'=': 0, 'in2': 1, 'in3': 2, 'preorder': 3, 'fold': 4}
    return (type_order.get(head_type, 9), num_vars, vars_tuple)


def analyze():
    with open(THEOREMS_FILE) as f:
        conjectures = [line.strip() for line in f if line.strip()]

    with open(PROVED_FILE) as f:
        proved = [line.strip() for line in f if line.strip()]

    # Separate positive and negative
    positives = [p for p in proved if '!' not in p]
    negatives = [p for p in proved if '!' in p]

    positives.sort(key=sort_key)
    negatives.sort(key=sort_key)

    # Build set of proved back-reformulated forms for matching
    proved_set = set(proved)

    # For each conjecture, generate its positive and negative back-reformulated forms
    missing_pos = []  # conjectures whose positive form is missing
    missing_neg = []  # conjectures whose negative form is missing
    missing = []      # conjectures with neither positive nor negative proved

    for conj in conjectures:
        br = back_reformulate(conj)
        if br is not None:
            # Operator-equality conjecture
            # Build negative form: insert ! before the head
            prefix_end = br.index('])')  # end of Anchor[...]
            neg_form = br[:prefix_end + 2] + '!' + br[prefix_end + 2:]
            if br in proved_set:
                pass  # proved positive
            elif neg_form in proved_set:
                pass  # proved negative
            else:
                missing.append(conj)
        else:
            # Simple equality conjecture like (>[2,10](Anchor[...])(=[2,10]))
            # These can only be disproved (negatives)
            # Negative form: insert ! before (=
            neg_form = conj.replace('](Anchor', '](Anchor', 1)
            # Build it properly
            m = re.match(r'(\(>[^\]]+\]\(AnchorIncubator\[[^\]]+\]\))\(', conj)
            if m:
                prefix = m.group(1)
                rest = conj[len(prefix):]
                neg_form = prefix + '!' + rest
                if conj in proved_set:
                    pass
                elif neg_form in proved_set:
                    pass
                else:
                    missing.append(conj)
            else:
                missing.append(conj)

    missing.sort(key=sort_key)

    # Group by head type for display
    def group_by_type(items):
        groups = defaultdict(list)
        for item in items:
            groups[extract_head_type(item)].append(item)
        return groups

    # Output
    out_lines = []
    out_lines.append(f"Incubator Analysis: {len(proved)}/{len(conjectures)} proved/disproved")
    out_lines.append(f"  Positive: {len(positives)}")
    out_lines.append(f"  Negative: {len(negatives)}")
    out_lines.append(f"  Missing:  {len(missing)}")
    out_lines.append("")

    out_lines.append("=" * 70)
    out_lines.append(f"POSITIVE THEOREMS ({len(positives)})")
    out_lines.append("=" * 70)
    for typ, items in sorted(group_by_type(positives).items()):
        out_lines.append(f"\n--- {typ} ({len(items)}) ---")
        for item in items:
            head_m = re.search(r'\((?:in2|in3|=|preorder|fold)\[[^\]]+\]\)', item)
            head = head_m.group(0) if head_m else item
            arith = human_arith(head)
            out_lines.append(f"  {head}  {arith}" if arith else f"  {head}")

    out_lines.append("")
    out_lines.append("=" * 70)
    out_lines.append(f"NEGATIVE THEOREMS / CONTRADICTIONS ({len(negatives)})")
    out_lines.append("=" * 70)
    for typ, items in sorted(group_by_type(negatives).items()):
        out_lines.append(f"\n--- {typ} ({len(items)}) ---")
        for item in items:
            head_m = re.search(r'!\((?:in2|in3|=|preorder|fold)\[[^\]]+\]\)', item)
            if head_m:
                neg_head = head_m.group(0)
                inner = neg_head[1:]  # strip leading !
            else:
                head_m = re.search(r'\((?:in2|in3|=|preorder|fold)\[[^\]]+\]\)', item)
                inner = head_m.group(0) if head_m else item
                neg_head = f"!{inner}"
            arith = human_arith(inner)
            label = f"NOT({arith})" if arith else ""
            out_lines.append(f"  {neg_head}  {label}" if label else f"  {neg_head}")

    # Split missing into expected-positive and expected-negative
    missing_pos = []
    missing_neg = []
    for item in missing:
        br = back_reformulate(item)
        if br:
            head_m = re.search(r'\((?:in2|in3|=|preorder|fold)\[[^\]]+\]\)', br)
            head_str = head_m.group(0) if head_m else br
            truth = is_true_fact(head_str)
            if truth is True:
                missing_pos.append(head_str)
            elif truth is False:
                missing_neg.append(head_str)
            else:
                missing_neg.append(head_str + " (?)")
        else:
            # Direct head (preorder, equality, etc.) — no back-reformulation needed
            head_m = re.search(r'\((?:in2|in3|=|preorder|fold)\[[^\]]+\]\)', item)
            if head_m:
                head_str = head_m.group(0)
                truth = is_true_fact(head_str)
                if truth is True:
                    missing_pos.append(head_str)
                elif truth is False:
                    missing_neg.append(head_str)
                else:
                    missing_neg.append(head_str + " (?)")
            else:
                missing_neg.append(item + " (?)")

    missing_pos.sort()
    missing_neg.sort()

    out_lines.append("")
    out_lines.append("=" * 70)
    out_lines.append(f"MISSING POSITIVES -- should be provable ({len(missing_pos)})")
    out_lines.append("=" * 70)
    for item in missing_pos:
        arith = human_arith(item)
        out_lines.append(f"  {item}  {arith}" if arith else f"  {item}")

    out_lines.append("")
    out_lines.append("=" * 70)
    out_lines.append(f"MISSING NEGATIVES -- should be contradicted ({len(missing_neg)})")
    out_lines.append("=" * 70)
    for item in missing_neg:
        arith = human_arith(item)
        label = f"NOT({arith})" if arith else ""
        out_lines.append(f"  {item}  {label}" if label else f"  {item}")

    # Anchor 4 regression: which facts from the {0,1,2,3,4} sub-model are still missing?
    VAL4 = {"2": 0, "6": 1, "7": 2, "9": 3, "10": 4}
    anchor4_args = set(VAL4.keys())

    def is_anchor4_fact(head_str):
        """Check if all (1)-typed args in head are within {0,1,2,3,4}."""
        m3 = re.match(r'\(in3\[(\d+),(\d+),(\d+),(\d+)\]\)', head_str)
        if m3:
            return all(m3.group(i) in anchor4_args for i in (1, 2, 3))
        m2 = re.match(r'\(in2\[(\d+),(\d+),\d+\]\)', head_str)
        if m2:
            return all(m2.group(i) in anchor4_args for i in (1, 2))
        meq = re.match(r'\(=\[(\d+),(\d+)\]\)', head_str)
        if meq:
            return all(meq.group(i) in anchor4_args for i in (1, 2))
        mpo = re.match(r'\(preorder\[\d+,\d+,(\d+),(\d+)\]\)', head_str)
        if mpo:
            return all(mpo.group(i) in anchor4_args for i in (1, 2))
        mf = re.match(r'\(fold\[\d+,\d+,\d+,\d+,(\d+),(\d+),(\d+)\]\)', head_str)
        if mf:
            return all(mf.group(i) in anchor4_args for i in (1, 2, 3))
        return False

    anchor4_missing = []
    for item in missing_neg:
        clean = item.replace(" (?)", "")
        if is_anchor4_fact(clean):
            anchor4_missing.append(item)
    for item in missing_pos:
        if is_anchor4_fact(item):
            anchor4_missing.append(item)

    out_lines.append("")
    out_lines.append("=" * 70)
    out_lines.append(f"ANCHOR 4 REGRESSION -- missing from {{0,1,2,3,4}} sub-model ({len(anchor4_missing)})")
    out_lines.append("=" * 70)
    if anchor4_missing:
        for item in sorted(anchor4_missing):
            arith = human_arith(item)
            label = f"NOT({arith})" if arith else ""
            out_lines.append(f"  {item}  {label}" if label else f"  {item}")
    else:
        out_lines.append("  None -- all Anchor 4 facts proved!")

    output = "\n".join(out_lines) + "\n"

    out_path = ".debug/incubator_analysis.txt"
    with open(out_path, "w") as f:
        f.write(output)
    print(f"Incubator analysis saved to {out_path}")

    return output


if __name__ == "__main__":
    analyze()
