# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------------
#
# This software is also available under a commercial license. For details,
# see: https://generative-logic.com/license
#
# Contributions to this project must be made under the terms of the
# Contributor License Agreement (CLA). See the project's CONTRIBUTING.md file.

"""Lean twin pages of the HTML proof graph.

A pipeline run with a Lean toolchain leaves `<full_proof_graph>/lean_export/`
next to the HTML pages: the certificates, the generated Lean modules, the
manifests and the kernel-check log. This module reads that folder and gives
every theorem chapter a companion page, `chapter<N>_lean.html`, in the same
visual style: the theorem's own Lean declarations sliced from the generated
module (its private induction helpers and the public theorem), one named fact
per proof row, with the row's GL tag and source line as comments, and links to
the raw module, the certificate, the manifest and the kernel-check log.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path


EXPORT_FOLDER = "lean_export"
KERNEL_LOG = "kernel_check.log"
TOOLCHAIN_FILE = "lean-toolchain"

# One entry per corpus kind: the manifest the writer emits, the generated
# module its facts live in, and the certificate the corpus was rendered from.
CORPUS_FILES = {
    "peano": ("lean_manifest.json", "GLExport/Generated/Peano.lean"),
    "gauss": ("lean_gauss_manifest.json", "GLExport/Generated/Gauss.lean"),
    "fta": ("lean_fta_manifest.json", "GLExport/Generated/FTA.lean"),
}

_DECLARATION = re.compile(r"^(?:private )?theorem (\S+)")
_COMMENT = re.compile(r"--.*$")
_KEYWORDS = re.compile(
    r"\b(theorem|private|have|intro|exact|apply|simp|simpa|only|using|at|fun|by|"
    r"show|refine|cases|rcases|obtain|constructor|left|right|calc|Classical|"
    r"Type|Prop|Or|And|Iff|Eq|False|True)\b"
)


@dataclass(frozen=True)
class LeanTheoremView:
    """
    @brief Everything one Lean twin page needs about one theorem.
    @details Assembled by `load_lean_export` from the manifest, the module and
    the certificate path of the theorem's corpus. Paths are relative to the
    export folder so the page can link them.
    """

    source_index: int
    theorem_id: str
    method: str
    module: str
    manifest: str
    certificate: str
    source: str
    chapters: tuple[tuple[str, str, int], ...]


@dataclass(frozen=True)
class LeanExportView:
    """
    @brief The export folder of one proof graph, as the pages see it.
    @details `theorems` is keyed by source index (the theorem's position in
    `global_theorem_list.txt`, the same index the chapter numbering follows).
    """

    folder: str
    toolchain: str
    row_facts: int
    theorems: dict[int, LeanTheoremView]


def slice_theorem_source(module_text: str, theorem_id: str) -> str:
    """
    @brief Cut one theorem's declarations out of a generated Lean module.
    @details
    The writer emits every declaration at the top level of the module: the
    public theorem `theorem <id>` and, for an induction theorem, its private
    helpers `<id>_induction_typing`, `<id>_check_zero` and
    `<id>_check_induction_condition` before it. A declaration ends where the
    next top-level declaration or the module's `end` line begins. The
    theorem's declarations are returned in module order, blank-line separated.
    @param module_text Complete generated module.
    @param theorem_id The manifest's theorem identifier (`peano_source_003`).
    @return The theorem's Lean source, ending in one newline.
    """

    lines = module_text.splitlines()
    starts = [
        (index, match.group(1))
        for index, line in enumerate(lines)
        if (match := _DECLARATION.match(line)) is not None
    ]
    ends = [index for index, line in enumerate(lines) if line.startswith("end ")]
    blocks: list[str] = []
    for position, (start, name) in enumerate(starts):
        if name != theorem_id and not name.startswith(theorem_id + "_"):
            continue
        if position + 1 < len(starts):
            end = starts[position + 1][0]
        else:
            later = [index for index in ends if index > start]
            assert later, f"module has no end line after {name}"
            end = later[0]
        blocks.append("\n".join(lines[start:end]).rstrip())
    assert blocks, f"theorem {theorem_id!r} is not declared in the module"
    return "\n\n".join(blocks) + "\n"


def highlight_lean(source: str) -> str:
    """
    @brief Escape Lean source for HTML and mark comments and keywords.
    @details Comments (`-- …` to the end of the line) become `<span class="lc">`,
    keywords `<span class="lk">`; everything else is escaped text. The keyword
    pass runs on the code part of each line only, never inside a comment.
    @param source Lean source text.
    @return HTML for a `<pre>` block.
    """

    rendered: list[str] = []
    for line in source.splitlines():
        match = _COMMENT.search(line)
        code = line[:match.start()] if match else line
        comment = line[match.start():] if match else ""
        code_html = _KEYWORDS.sub(r'<span class="lk">\1</span>', html.escape(code))
        if comment:
            code_html += f'<span class="lc">{html.escape(comment)}</span>'
        rendered.append(code_html)
    return "\n".join(rendered)


def _sha256(path: Path) -> str:
    """
    @brief Upper-case SHA-256 of a file, the certificate's hash convention.
    @param path File to hash.
    @return Hex digest in upper case.
    """

    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_lean_export(out_dir: Path, proc_dir: Path) -> LeanExportView | None:
    """
    @brief Read the export folder of one proof graph.
    @details
    No export folder means the run had no Lean toolchain (the maintainer's
    policy: one notice, no export) — the pages then carry no Lean links. An
    export folder is read for every corpus whose certificate was built from
    exactly this graph: the certificate's `source.theorem_list_sha256` must
    equal the hash of `<proc_dir>/global_theorem_list.txt` (the shortcut's
    folder also holds the copied Peano and Gauss corpora of the full run,
    whose source indices would otherwise collide with the shortcut's). The
    kernel-check log must record success for every command; the pipeline
    never reaches the HTML stage otherwise, so a failing log is a stale folder
    and asserts.
    @param out_dir The proof graph's HTML folder.
    @param proc_dir The proof graph's processed folder (for the list hash).
    @return The export view, or None when the folder is absent.
    """

    export_dir = out_dir / EXPORT_FOLDER
    log = export_dir / KERNEL_LOG
    if not log.is_file():
        return None
    exits = [
        int(line.split()[1])
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.startswith("exit ")
    ]
    assert exits and all(code == 0 for code in exits), f"stale Lean export: {log} records a failure"
    toolchain = (export_dir / TOOLCHAIN_FILE).read_text(encoding="utf-8").strip()
    graph_hash = _sha256(proc_dir / "global_theorem_list.txt")
    theorems: dict[int, LeanTheoremView] = {}
    row_facts = 0
    for kind, (manifest_name, module_name) in CORPUS_FILES.items():
        certificate_path = export_dir / "certificates" / kind / "certificate.json"
        manifest_path = export_dir / manifest_name
        if not certificate_path.is_file() or not manifest_path.is_file():
            continue
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        if str(certificate["source"]["theorem_list_sha256"]) != graph_hash:
            continue
        module_text = (export_dir / module_name).read_text(encoding="utf-8")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for theorem in manifest["theorems"]:
            source_index = int(theorem["source_index"])
            assert source_index not in theorems, f"source index {source_index} exported twice"
            chapters = tuple(
                (
                    str(chapter["role"]),
                    str(chapter["source_chapter"]),
                    sum(1 for row in chapter["dispositions"] if row["status"] == "emitted"),
                )
                for chapter in theorem["chapters"]
            )
            row_facts += sum(rows for _, _, rows in chapters)
            theorems[source_index] = LeanTheoremView(
                source_index=source_index,
                theorem_id=str(theorem["id"]),
                method=str(theorem["method"]),
                module=module_name,
                manifest=manifest_name,
                certificate=f"certificates/{kind}/certificate.json",
                source=slice_theorem_source(module_text, str(theorem["id"])),
                chapters=chapters,
            )
    assert theorems, f"{export_dir} holds no corpus of this proof graph"
    return LeanExportView(EXPORT_FOLDER, toolchain, row_facts, theorems)


def lean_page_name(chapter_num: int) -> str:
    """
    @brief File name of a chapter's Lean twin page.
    @param chapter_num The chapter number of the HTML page.
    @return `chapter<N>_lean.html`.
    """

    return f"chapter{chapter_num}_lean.html"


def nav_link(export: LeanExportView | None, source_index: int, chapter_num: int) -> str:
    """
    @brief The chapter page's navigation link to its Lean twin.
    @param export The export view, or None without an export.
    @param source_index The theorem's source index.
    @param chapter_num The chapter number.
    @return An anchor element, or the empty string when the theorem has no twin.
    """

    if export is None or source_index not in export.theorems:
        return ""
    return f"<a href='{lean_page_name(chapter_num)}'>Lean 4 proof</a>"


def toc_link(export: LeanExportView | None, source_index: int, chapter_num: int) -> str:
    """
    @brief The index entry's link to a chapter's Lean twin.
    @param export The export view, or None without an export.
    @param source_index The theorem's source index.
    @param chapter_num The chapter number.
    @return A small trailing anchor, or the empty string.
    """

    if export is None or source_index not in export.theorems:
        return ""
    return (
        f" <a href='{lean_page_name(chapter_num)}' style='font-size:0.8em; color:#8B8FA5;'>"
        "[Lean 4]</a>"
    )


def render_index_summary(export: LeanExportView | None) -> str:
    """
    @brief The index page's one-paragraph summary of the Lean export.
    @param export The export view, or None without an export.
    @return A paragraph element, or the empty string.
    """

    if export is None:
        return ""
    return (
        '  <p style="color:#8B8FA5; margin:0.2em 0 1em 0;">'
        f"Lean 4 export: {len(export.theorems)} theorems and {export.row_facts} named row facts, "
        f"kernel-checked with <code>{html.escape(export.toolchain)}</code> — "
        f"<a href='{export.folder}/{KERNEL_LOG}'>kernel check log</a>. "
        "Every chapter links its Lean twin.</p>"
    )


def render_lean_page(
    view: LeanTheoremView,
    export: LeanExportView,
    chapter_num: int,
    html_filename: str,
    statement_html: str,
    readable_html: str,
    common_style: str,
    license_source_comment: str,
    license_head_meta: str,
    license_footer: str,
) -> str:
    """
    @brief Render one Lean twin page.
    @details Same stylesheet, navigation bar and footer as the chapter page;
    the theorem statement and readable caption of the chapter; a card with the
    Lean identifiers, method, chapters, toolchain and links; then the sliced
    Lean source with comments and keywords marked.
    @param view The theorem's export view.
    @param export The proof graph's export view.
    @param chapter_num The chapter number.
    @param html_filename The chapter page the twin belongs to.
    @param statement_html Escaped MPL statement of the theorem.
    @param readable_html The readable caption, already HTML.
    @param common_style The generator's shared `<style>` block.
    @param license_source_comment The generator's HTML license comment.
    @param license_head_meta The generator's license `<meta>` block.
    @param license_footer The generator's footer block.
    @return The complete page.
    """

    folder = export.folder
    chapter_rows = "".join(
        f"<tr><td style='padding:0.15em 1em 0.15em 0;'>{html.escape(role.replace('_', ' '))}</td>"
        f"<td style='padding:0.15em 1em 0.15em 0;'><code>{html.escape(source_chapter)}</code></td>"
        f"<td style='padding:0.15em 0;'>{rows} row facts</td></tr>"
        for role, source_chapter, rows in view.chapters
    )
    return f"""<!DOCTYPE html>
{license_source_comment}
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Lean 4 proof — chapter {chapter_num}</title>
  <link rel="icon" type="image/png" href="favicon.png">
{license_head_meta}
  {common_style}
  <style>
    pre.lean {{ background:#20232F; border:1px solid #3A3D4A; border-radius:6px; padding:1em; overflow-x:auto; line-height:1.45; font-size:0.92em; }}
    pre.lean .lc {{ color:#8B8FA5; font-style:italic; }}
    pre.lean .lk {{ color:#5DCAA5; }}
    .lean-card {{ background:#20232F; border:1px solid #3A3D4A; border-radius:6px; padding:0.8em 1em; margin:1em 0; color:#F0E8DC; }}
    .lean-card td {{ vertical-align:top; }}
  </style>
</head>
<body>
  <nav>
    <a href="index.html">Index</a> <a href="tags.html">Reasoning rules</a>
    <a href="{html_filename}">Proof (HTML)</a>
    <a href="{folder}/{view.module}">Lean module</a>
    <a href="{folder}/{view.certificate}">Certificate</a>
    <a href="{folder}/{view.manifest}">Manifest</a>
    <a href="{folder}/{KERNEL_LOG}">Kernel check log</a>
  </nav>
  <h1>Chapter {chapter_num} — Lean 4 proof: <span>{statement_html}</span></h1>
  <div style="margin-left:20px; color:#8B8FA5; font-weight:bold; font-size:1.3em;">{readable_html}</div>
  <div class="lean-card">
    <p style="margin:0 0 0.6em 0;">The same proof, replayed in Lean 4 and accepted by its kernel. Every proof
    row of the HTML chapter is one named fact (<code>row_&lt;line&gt;</code>) whose comment names the GL
    reasoning rule and the source line; the public theorem at the end states the chapter's result over one
    abstract carrier with ordinary predicates. No <code>sorry</code>, no added axiom.</p>
    <table style="border-collapse:collapse;">
      <tr><td style='padding:0.15em 1em 0.15em 0;'>Lean theorem</td><td style='padding:0.15em 0;'><code>{html.escape(view.theorem_id)}</code></td></tr>
      <tr><td style='padding:0.15em 1em 0.15em 0;'>Method</td><td style='padding:0.15em 0;'>{html.escape(view.method)}</td></tr>
      <tr><td style='padding:0.15em 1em 0.15em 0;'>Toolchain</td><td style='padding:0.15em 0;'><code>{html.escape(export.toolchain)}</code></td></tr>
    </table>
    <table style="border-collapse:collapse; margin-top:0.5em;">{chapter_rows}</table>
  </div>
  <pre class="lean"><code>{highlight_lean(view.source)}</code></pre>
{license_footer}
</body>
</html>"""


def write_lean_page(out_dir: Path, page_html: str, chapter_num: int) -> Path:
    """
    @brief Write one Lean twin page next to its chapter page.
    @param out_dir The proof graph's HTML folder.
    @param page_html The rendered page.
    @param chapter_num The chapter number.
    @return The written path.
    """

    path = out_dir / lean_page_name(chapter_num)
    path.write_text(page_html, encoding="utf-8")
    return path
