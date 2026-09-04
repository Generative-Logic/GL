import Lake

open Lake DSL

package gl_peano_export where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

@[default_target]
lean_lib GLExport where
