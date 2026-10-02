# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Concrete natural-number witnesses for the connected abstract export."""

from pathlib import Path

from proof_export.lean import LEAN_LICENSE, _theorem_body, lean_constant, render_expression
from proof_export.mpl import iter_atoms, parse_mpl


def write_nat_model(certificates: list[dict], project: Path) -> None:
    """@brief Discharge every exported theorem's structure and induction on Nat.

    @details The concrete arithmetic relations use Lean's natural numbers.
    Anchor proofs unfold the certificate's own definition closure. Induction is
    supplied by Nat.rec. Each public result applies its actual generated theorem
    with those witnesses and retains only its mathematical statement as premises.
    External theorem-list assumptions are forbidden in this connected project.

    @param certificates Peano, Gauss, and FTA certificates in dependency order.
    @param project Complete generated Lake project receiving the model and results.
    @return None.
    @invariant Every abstract public theorem receives exactly one Nat instance.
    """
    assert len(certificates) == 3
    assert [c["corpus"]["id"].split("_")[0] for c in certificates] == ["peano", "gauss", "fta"]
    for certificate in certificates:
        assert all(d["theorem_list_scope"] == "same"
                   for d in certificate.get("certificate_dependencies", []))
    definitions = certificates[-1]["binary_definitions"]
    pending = ["AnchorPeano", "AnchorGauss", "AnchorFTA"]
    closure = set()
    while pending:
        head = pending.pop()
        if head in closure or head in {"in", "in2", "in3", "="}:
            continue
        assert head in definitions, f"Nat anchor definition missing: {head}"
        closure.add(head)
        pending.extend(atom.head for element in definitions[head]["elements"]
                       for atom in iter_atoms(parse_mpl(element)))
    unfolding = ", ".join(lean_constant(head) for head in sorted(closure))
    model = [LEAN_LICENSE, "", "import GLExport.Generated.Definitions", "",
             "set_option linter.unusedSimpArgs false", "", "namespace GLExport.NatModel", "",
             "def domain : GLSet Nat := fun _ => True",
             "def successor : GLBinaryRelation Nat := fun x y => y = Nat.succ x",
             "def addition : GLTernaryRelation Nat := fun x y z => z = x + y",
             "def multiplication : GLTernaryRelation Nat := fun x y z => z = x * y",
             "def identity : GLBinaryRelation Nat := fun x y => y = x", "",
             "theorem induction : RelationalInduction domain 0 successor := by",
             "  intro P k base step _",
             "  exact Nat.rec base (fun n hypothesis => step n True.intro hypothesis (Nat.succ n) rfl) k", ""]
    for kind in ("Peano", "Gauss", "FTA"):
        arguments = "domain 0 successor addition multiplication 1"
        if kind != "Peano":
            arguments += " 2 identity"
        model.extend([
            f"theorem anchor{kind} : gl_Anchor{kind} {arguments} := by",
            "  classical",
            f"  simp [{unfolding}, domain, successor, addition, multiplication, identity,",
            "    Nat.add_succ, Nat.mul_succ, -Nat.succ_eq_add_one]",
            "  intro z x left y right",
            "  exact Nat.succ.inj (left.symm.trans right)", "",
        ])
    model.extend(["end GLExport.NatModel", ""])
    replacements = {
        "N": "NatModel.domain", "i0": "0", "s": "NatModel.successor",
        "+": "NatModel.addition", "*": "NatModel.multiplication",
        "i1": "1", "i2": "2", "id": "NatModel.identity",
    }
    instances = [LEAN_LICENSE, "", "import GLExport.Generated.FTA",
                 "import GLExport.NatModel", "", "namespace GLExport.NatResults", ""]
    names = []
    for certificate, kind in zip(certificates, ("Peano", "Gauss", "FTA")):
        for theorem in certificate["theorems"]:
            assert not theorem.get("_external_interfaces")
            identifier = theorem["id"]
            assert identifier not in names
            names.append(identifier)
            body = _theorem_body(parse_mpl(theorem["theorem"]["mpl"]))
            proposition = render_expression(body, theorem["variable_types"], replacements, carrier="Nat")
            arguments = "NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1"
            if kind != "Peano":
                arguments += " 2 NatModel.identity"
            arguments += f" NatModel.anchor{kind} NatModel.induction"
            namespace = "GLExport.FTA" if kind == "FTA" else "GLExport"
            instances.extend([f"theorem {identifier} : {proposition} := by",
                              f"  exact {namespace}.{identifier} {arguments}", ""])
    instances.extend(["end GLExport.NatResults", ""])
    (project / "GLExport" / "NatModel.lean").write_text("\n".join(model), encoding="utf-8")
    (project / "GLExport" / "NatResults.lean").write_text("\n".join(instances), encoding="utf-8")
    with (project / "GLExport.lean").open("a", encoding="utf-8") as root:
        root.write("\nimport GLExport.NatResults\n")
