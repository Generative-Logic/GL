# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import compilation_state as compiler


class CompilationStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "files/GL_binaries"
        self.directory.mkdir(parents=True)
        self.replacements = {"existence16": {"target": "preorder", "arguments": [1, 4, 2, 3]}}
        self.state = {"schema": 1, "next_counters": dict.fromkeys(compiler.CATEGORIES, 0),
                      "replacements": self.replacements}
        self.state["next_counters"]["existence"] = 30
        self.named = {"category": "existence", "arity": 4,
                      "signature": "(preorder[u_1,u_2,u_3,u_4])", "definedSet": "",
                      "elements": ["(in[1,u_1])", "(in3[u_3,1,u_4,u_2])"]}
        self.numbered = {"category": "existence", "arity": 4,
                         "signature": "(existence16[u_1,u_2,u_3,u_4])", "definedSet": "",
                         "elements": ["(in[1,u_1])", "(in3[u_2,1,u_3,u_4])"]}
        self.atoms = {name: {"category": "atomic", "elements": [""], "arity": arity,
                             "signature": f"({name}[{','.join(f'u_{i}' for i in range(1, arity + 1))}])"}
                      for name, arity in (("in", 2), ("in3", 4))}
        self.source = {**self.atoms, "preorder": self.named}

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) if isinstance(value, dict) else value, encoding="utf-8")
        return path

    def test_application_rewrite_preserves_scope_and_reference_structure(self):
        text = "!(existence16[1,6,2,4])\tmain_boundary_((existence16[a,a,b,c]))\texpansion\t(>[existence16](p[existence16])(q[]))\r\n"
        expected = "!(preorder[1,4,6,2])\tmain_boundary_((preorder[a,c,a,b]))\texpansion\t(>[existence16](p[existence16])(q[]))\r\n"
        self.assertEqual(compiler.rewrite_references(text, self.replacements), expected)
        self.assertEqual(compiler.rewrite_references(expected, self.replacements), expected)

    def test_composition_zero_arity_and_unrelated_names(self):
        replacements = {"or0": {"target": "truth", "arguments": []},
                        "and0": {"target": "and1", "arguments": [2, 1]},
                        "and1": {"target": "pair", "arguments": [2, 1]}}
        self.assertEqual(compiler.rewrite_references("(or0[])\t(and0[a,b])\t(and10[a,b])", replacements),
                         "(truth[])\t(pair[a,b])\t(and10[a,b])")
        self.assertEqual(compiler.rewrite_references("unchanged", {}), "unchanged")

    def test_invalid_arity_permutation_cycle_and_schema_assert(self):
        with self.assertRaises(AssertionError):
            compiler.rewrite_references("(existence16[a,b])", self.replacements)
        with self.assertRaises(AssertionError):
            compiler.rewrite_references("(existence16[a,b,c,d]", self.replacements)
        for replacement in ({"target": "p", "arguments": [1, 1]},
                            {"target": "existence16", "arguments": [1, 2]}):
            with self.assertRaises(AssertionError):
                compiler.rewrite_references("(existence16[a,b])", {"existence16": replacement})
        for field, value in (("schema", 2), ("schema", True), ("next_counters", {})):
            state = deepcopy(self.state)
            state[field] = value
            with self.assertRaises(AssertionError):
                compiler.validate_state(state)
        state = deepcopy(self.state)
        state["next_counters"]["or"] = -1
        with self.assertRaises(AssertionError):
            compiler.validate_state(state)

    def test_original_registry_counter_derivation_and_sidecar_seed(self):
        self.write("files/GL_binaries/GL_binary_shared.json", {"existence16": self.numbered})
        legacy = compiler.read_state(self.directory, "shared")
        self.assertEqual(legacy["next_counters"]["existence"], 17)
        self.assertEqual(legacy["replacements"], {})
        self.write("files/GL_binaries/GL_compilation_state_shared.json", self.state)
        compiler.seed_state(self.root, "FTA")
        self.assertEqual(compiler.read_state(self.directory, "FTA"), self.state)

    def test_registry_removes_numbered_definition_and_imports_public_dependencies(self):
        old = {"existence16": self.numbered,
               "holder": {"category": "and", "signature": "(holder[u_1,u_2,u_3,u_4])",
                          "elements": ["(existence16[u_1,u_3,u_4,u_2])"],
                          "implications": ["!(existence16[u_1,u_3,u_4,u_2])"]}}
        migrated = compiler.rewrite_registry(old, self.source, self.replacements)
        self.assertNotIn("existence16", migrated)
        self.assertEqual(migrated["holder"]["elements"], ["(preorder[u_1,u_2,u_3,u_4])"])
        self.assertEqual(migrated["holder"]["implications"], ["!(preorder[u_1,u_2,u_3,u_4])"])
        self.assertTrue({"preorder", "in", "in3"} <= migrated.keys())
        self.assertIn("existence16", old)
        with self.assertRaises(AssertionError):
            compiler.rewrite_registry(old, {}, self.replacements)
        migrated_empty = compiler.rewrite_registry({}, self.source, self.replacements)
        self.assertTrue({"preorder", "in", "in3"} <= migrated_empty.keys())

    def test_auxiliary_references_migrate_and_import_their_dependency_closure(self):
        old_atom = "!(existence16[a,b,c,d])"
        new_atom = "!(preorder[a,d,b,c])"
        entity = {"supporting_implications": {"version": 1, "prepared": True, "auxiliary": False,
                    "rules": [old_atom]},
                  "conjunction_refutations": {"version": 1, "prepared": True,
                    "rules": [[old_atom, old_atom]]},
                  "contraposition": {"version": 1, "prepared": True, "auxiliary": False,
                    "rules": [{"sources": [old_atom], "source": old_atom, "reverse": old_atom}]}}
        self.assertEqual(compiler.rewrite_auxiliary_references(entity, self.replacements), [new_atom] * 6)
        self.assertTrue(entity["supporting_implications"]["prepared"])
        self.assertFalse(entity["supporting_implications"]["auxiliary"])
        composed = {"supporting_implications": {"version": 2, "prepared": True,
                    "auxiliary": False, "rules": [old_atom], "composed_rules": [old_atom]}}
        self.assertEqual(compiler.rewrite_auxiliary_references(composed, self.replacements), [new_atom] * 2)
        self.assertEqual(composed["supporting_implications"]["composed_rules"], [new_atom])
        wrong = deepcopy(composed); wrong["supporting_implications"]["composed_rules"] *= 2
        with self.assertRaises(AssertionError):
            compiler.rewrite_auxiliary_references(wrong, self.replacements)
        malformed = deepcopy(entity); malformed["supporting_implications"]["version"] = 3
        with self.assertRaises(AssertionError):
            compiler.rewrite_auxiliary_references(malformed, self.replacements)
        self.assertEqual(compiler.rewrite_auxiliary_references({}, self.replacements), [])
        holder = {"category": "atomic", "arity": 0, "signature": "(holder[])", "elements": [],
                  "supporting_implications": {"version": 1, "prepared": True, "auxiliary": False,
                                             "rules": ["(helper[a,b,c,d])"]}}
        helper = {"category": "implication", "arity": 4, "signature": "(helper[u_1,u_2,u_3,u_4])",
                  "elements": [], "supporting_implications": {"version": 1, "prepared": True,
                    "auxiliary": True, "rules": [old_atom]}}
        migrated = compiler.rewrite_registry({"holder": holder}, {**self.source, "helper": helper}, self.replacements)
        self.assertEqual(migrated["helper"]["supporting_implications"]["rules"], [new_atom])
        self.assertIn("preorder", migrated)
        self.assertNotIn("existence16", migrated)

    def test_batch_migration_changes_active_references_and_preserves_archives(self):
        self.write("files/GL_binaries/GL_binary_shared.json", {**self.atoms, "existence16": self.numbered})
        self.write("files/GL_binaries/GL_binary_Peano.json", {**self.atoms, "existence16": self.numbered})
        self.write("files/GL_binaries/GL_binary_FTA.json", self.source)
        self.write("files/GL_binaries/GL_compilation_state_FTA.json", self.state)
        proof = self.write("files/raw_proof_graph/chapter.txt", "!(existence16[1,6,2,4])\tmain\texpansion\n")
        theorem = self.write("files/incubator/theorems/compiled_theorems.txt", "(existence16[a,b,c,d])\t(original[])\n")
        archive = self.write(".debug/prior/chapter.txt", proof.read_text())
        before = archive.read_bytes()
        result = compiler.migrate_batch(self.root, "FTA")
        self.assertEqual(result["retired"], 1)
        self.assertEqual(proof.read_text(), "!(preorder[1,4,6,2])\tmain\texpansion\n")
        self.assertEqual(theorem.read_text(), "(preorder[a,d,b,c])\t(original[])\n")
        self.assertEqual(archive.read_bytes(), before)
        for suffix in ("shared", "Peano", "FTA"):
            registry = json.loads((self.directory / f"GL_binary_{suffix}.json").read_text())
            self.assertNotIn("existence16", registry)
            self.assertEqual(compiler.read_state(self.directory, suffix)["next_counters"]["existence"], 30)
        self.assertEqual(compiler.migrate_batch(self.root, "FTA"), {"retired": 0, "rewritten_files": 0})

    def test_shortcut_migrates_only_its_active_proof_pool(self):
        self.write("files/GL_binaries/GL_binary_shared.json", {**self.atoms, "existence16": self.numbered})
        self.write("files/GL_binaries/GL_binary_FTA.json", self.source)
        self.write("files/GL_binaries/GL_compilation_state_FTA.json", self.state)
        main = self.write("files/raw_proof_graph/chapter.txt", "(existence16[a,b,c,d])")
        replay = self.write("files/shortcut/raw_proof_graph/chapter.txt", main.read_text())
        compiler.migrate_batch(self.root, "FTA", shortcut=True)
        self.assertEqual(replay.read_text(), "(preorder[a,d,b,c])")
        self.assertEqual(main.read_text(), "(existence16[a,b,c,d])")


if __name__ == "__main__":
    unittest.main()
