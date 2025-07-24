# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025 Generative Logic UG (haftungsbeschränkt)
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

#!/usr/bin/env python3
"""
generate_full_proof_graph.py

Generates a set of HTML proof pages:
- An index page with a Table of Contents
- One HTML file per theorem (with navigation links)

Each theorem tuple is now (theorem_name:str, method:str, var_name:str).
Default output directory: full_proof_graph
"""

import os
import html
import analyze_expressions
import re
import shutil

import create_expressions
from create_expressions import get_args
from parameters import debug

import visu_helpers
from visu_helpers import format_mirroring
from pathlib import Path


# wherever this file lives, assume the project root is its parent folder
PROJECT_ROOT = Path(__file__).resolve().parent



# global mapping from displayed theorem title → its chapter file
theorem_to_file = {}


# Utility function to wrap clickable substrings starting with '(' or '!(' and ending at space or end-of-string
def wrap_clickable(text):
    def repl(m):
        s = m.group(0)                   # the raw token, e.g. "(v7*v8)=(v8*v7)"
        esc = html.escape(s, quote=True)
        target = theorem_to_file.get(s)
        # span for the normal “expand on right-click”
        span = f'<span class="clickable" data-text="{esc}">{esc}</span>'
        # if it exactly matches one of our theorems, link it
        return f'<a href="{target}" class="theorem-link">{span}</a>' if target else span

    pattern = r"!?\([^ ]*?\)(?= |$)"
    return re.sub(pattern, repl, text)








def format_stack_entries(stack, prefix='', cursor_index=None):
    """
    Convert a proof-stack (list of [key, ref1, ref2, …]) into HTML,
    bolding the second element of each entry and optionally formatting implications.
    """
    # Reverse so the earliest step is first in the output
    rev = list(stack)[::-1]

    # Map each key (normalized) to a unique anchor ID
    key_map = {}
    for idx, entry in enumerate(rev):
        key = entry[0]
        norm = re.sub(r'\s+', '', key).lower()
        key_map[norm] = f"{prefix}-entry{idx}" if prefix else f"entry{idx}"

    lines = []
    total = len(rev)
    for idx, entry in enumerate(rev):
        # skip any proof‐stack line whose tokens include 'theorem'
        if 'theorem' in entry:
            continue
        key, *refs = entry
        norm_key = re.sub(r'\s+', '', key).lower()
        anchor = key_map[norm_key]

        # Highlight the first token of the final line in red and bold,
        # by moving the style into the <a> itself if it’s a link:
        if idx == total - 1:
            first, *rest = key.split(' ', 1)
            first_html = wrap_clickable(first)

            if first_html.startswith('<a '):
                # inject red style into the clickable span (with !important to override external CSS)
                first_html = first_html.replace(
                    '<span class="clickable"',
                    '<span class="clickable" style="color:red !important; font-weight:bold !important;"',
                    1
                )
            else:
                # if it wasn’t a link, just wrap it in a styled clickable‐span
                first_html = f'<span class="clickable" style="color:red; font-weight:bold">{first_html}</span>'

            if rest:
                rest_html = wrap_clickable(rest[0])
                key_html = f"{first_html} {rest_html}"
            else:
                key_html = first_html

        else:
            key_html = wrap_clickable(key)

        # Build the row: [<span id='anchor'>key_html</span>, ref_html1, ref_html2, …]
        parts = [f"<span id='{anchor}'>{key_html}</span>"]
        for ref in refs:
            norm_ref = re.sub(r'\s+', '', ref).lower()
            ref_html = wrap_clickable(ref)
            if norm_ref in key_map:
                parts.append(
                    f"<a href='#{key_map[norm_ref]}' style='text-decoration:none'>{ref_html}</a>"
                )
            else:
                parts.append(ref_html)

        # Bold the first reference if present
        if len(parts) > 1:
            parts[1] = f"<b>{parts[1]}</b>"

        # Join parts and optionally highlight the cursor row
        line_html = "&nbsp;&nbsp;".join(parts)
        if cursor_index is not None and idx == cursor_index:
            line_html = (
                f"<div style='background-color:#fffa8b; padding:4px; "
                f"border-radius:4px'>{line_html}</div>"
            )

        lines.append(line_html)

        # Handle implication: include the token before 'implication' and pass sublist
        if 'implication' in entry:
            imp_idx = entry.index('implication')
            start_idx = max(0, imp_idx - 1)
            sublist = entry[start_idx:]
            impl_text = visu_helpers.format_implication(sublist)
            if impl_text:
                impl_html = (
                    f"<div class='implication' style='margin-left:20px; color:#888888; "
                    f"font-weight:bold; font-size:1.3em;'>{html.escape(impl_text)}</div>"
                )
                lines.append(impl_html)

        if 'mirrored from' in entry:
            mirrored_text = format_mirroring(entry)
            mirrored_html = (
                f"<div class='mirrored' style='margin-left:20px; color:#888888; "
                f"font-weight:bold; font-size:1.3em;'>{html.escape(mirrored_text)}</div>"
            )
            lines.append(mirrored_html)


    # Separate entries by three <br/> for spacing
    return "<br/><br/><br/>".join(lines)







def extract_args(s: str) -> list[str]:
    # same pattern as before
    pattern = r'(?<=[\[,])([^,\[\]]+)(?=[\],])'
    all_subs = re.findall(pattern, s)
    # remove duplicates while preserving order
    return list(dict.fromkeys(all_subs))

def rename_expr(expr: str):
    args = extract_args(expr)

    replacement_map = {}

    for arg in args:
        if arg.startswith("u_"):
            if arg[2:].isdigit():
                replacement_map[arg] = "v" + arg[2:]
            else:
                replacement_map[arg] = arg[2:]

    for arg in args:
        if arg.isdigit():
            replacement_map[arg] = "v" + arg

    return analyze_expressions.replace_keys_in_string(expr, replacement_map)

def rename_theorem(theorem: str):
    ren_th = rename_expr(theorem)

    temp_chain = []
    head = analyze_expressions.disintegrate_implication(ren_th, temp_chain)
    chain = []
    for element in temp_chain:
        chain.append(element[0])

    replacement_map = {}
    for expr in chain:
        if expr.startswith("(NaturalNumbers"):
            args = analyze_expressions.get_args(expr)

            replacement_map[args[0]] = "N"
            replacement_map[args[1]] = "0"
            replacement_map[args[2]] = "1"
            replacement_map[args[3]] = "s"
            replacement_map[args[4]] = "+"
            replacement_map[args[5]] = "*"


    ren_th2 = analyze_expressions.replace_keys_in_string(ren_th, replacement_map)


    if debug:
        ren_th2 = theorem

    return ren_th2



def clean_stack(stack: list[list[str]]):
    """
    Cleans a stack of expression strings by normalizing variable keys.

    Steps:
    1. Finds all substrings in each expression matching pattern1 (components separated by commas or brackets).
       Among those, identifies tokens of the form 'v<digit>' and determines the maximum digit.
    2. Scans expressions again for tokens matching 'it_<digits>_lev_<digits>_' and assigns each
       such token a new key of the form 'v<mi>', incrementing mi for each unique occurrence.
    3. Applies replacements to all expressions using analyze_expressions.replace_keys_in_string.

    Args:
        stack: A list of lists of expression strings.
    Returns:
        A new stack with cleaned expression strings.
    """
    # Pattern to extract components between commas/brackets
    pattern1 = re.compile(r'(?<=[\[,])([^,\[\]]+)(?=[],])')
    # Pattern to match keys like 'it_123_lev_4_'
    pattern2 = re.compile(r'it_(\d+)_lev_\d+_\d+')

    # 1. Determine the highest existing 'v<digit>' index
    max_index = -1
    for row in stack:
        for expr in row:
            for token in pattern1.findall(expr):
                m = re.match(r'v(\d+)$', token)
                if m:
                    idx = int(m.group(1))
                    max_index = max(max_index, idx)
    # Start new index after the highest found, or at 0 if none
    mi = (max_index + 1) if max_index >= 0 else 0

    # 2. Build replacement map for 'it_..._lev_...' tokens
    replacement_map: dict[str, str] = {}
    for row in stack:
        for expr in row:
            for match in pattern2.finditer(expr):
                key = match.group(0)
                if key not in replacement_map:
                    replacement_map[key] = f"v{mi}"
                    mi += 1

    # 3. Apply replacements to produce cleaned stack
    for i, row in enumerate(stack):
        for j, expr in enumerate(row):
            stack[i][j] = analyze_expressions.replace_keys_in_string(expr, replacement_map)



def rename_stack(stack: list[list[str]], theorem: str):

    for entry in stack:
        for index, expr in enumerate(entry):
            entry[index] = rename_expr(expr)

    clean_stack(stack)

    if theorem:
        replacement_map = {}

        temp_chain = []
        head = analyze_expressions.disintegrate_implication(theorem, temp_chain)
        chain = [e[0] for e in temp_chain]

        for expr in chain:
            if expr.startswith("(NaturalNumbers"):
                args = analyze_expressions.get_args(expr)

                replacement_map["v" + args[0]] = "N"
                replacement_map["v" + args[1]] = "0"
                replacement_map["v" + args[2]] = "1"
                replacement_map["v" + args[3]] = "s"
                replacement_map["v" + args[4]] = "+"
                replacement_map["v" + args[5]] = "*"

        for entry in stack:
            for index, expr in enumerate(entry):
                entry[index] = analyze_expressions.replace_keys_in_string(expr, replacement_map)

    return





# Proof step functions returning HTML for subchapters
def check_zero(theorem, induction_var, prefix=''):
    temp_chain = []
    head = analyze_expressions.disintegrate_implication(theorem, temp_chain)
    chain = [e[0] for e in temp_chain]
    mb = analyze_expressions.global_body_of_proves

    t2 = '(&(&(in[2,1])(&(fXY[3,1,1])(&(>[n](in[n,1])!(in2[n,2,3]))(>[m](in[m,1])(>[n1,n2](&(in2[n1,m,3])(in2[n2,m,3]))(=[n1,n2]))))))(&(&(fXYZ[4,1,1,1])(&(>[a](in[a,1])(>[b](in3[a,2,b,4])(=[a,b])))(>[b](in[b,1])(>[a,c,d](&(in2[b,c,3])(in3[a,b,d,4]))(&(>[e](in3[a,c,e,4])(in2[d,e,3]))(>[e](in2[d,e,3])(in3[a,c,e,4])))))))(&(fXYZ[5,1,1,1])(&(>[a](in[a,1])(>[b](in3[a,2,b,5])(=[b,2])))(>[b](in[b,1])(>[a,c,d](&(in2[b,c,3])(in3[a,b,d,5]))(&(>[e](in3[d,a,e,4])(in3[a,c,e,5]))(>[e](in3[a,c,e,5])(in3[d,a,e,4])))))))))'
    t1 = '(>[a](in[a,1])(>[b](in3[a,2,b,4])(=[a,b])))'




    for elt in chain:
        mb = mb.simple_map[elt]


    zero_name = analyze_expressions.get_args(chain[0])[1]
    for key in sorted(mb.simple_map):
        if (key.startswith("(=[s(rec") and
                key.endswith(f",{zero_name}])") and
                head in mb.simple_map[key].local_statements):

            induction_var2 = get_args(mb.simple_map[key].expr_key)[0]
            if induction_var != induction_var2:
                continue

            rec_name = create_expressions.get_args(key)[0]
            temp_expr = f"(=[{rec_name},{zero_name}])"
            mb = mb.simple_map[temp_expr]
            break

    stack = []


    analyze_expressions.build_stack(mb, head, stack, set())
    rename_stack(stack, theorem)
    return format_stack_entries(stack, prefix)


def check_induction_condition(theorem, induction_var, prefix=''):
    temp_chain = []
    head = analyze_expressions.disintegrate_implication(theorem, temp_chain)
    chain = [e[0] for e in temp_chain]
    mb = analyze_expressions.global_body_of_proves
    for elt in chain:
        mb = mb.simple_map[elt]
    s_name = analyze_expressions.get_args(chain[0])[3]

    for key in  sorted(mb.simple_map):
        if (key.startswith("(in2[rec") and
                key.endswith(f"{induction_var},{s_name}])") and
                head in mb.simple_map[key].local_statements):
            mb = mb.simple_map[key]
            stack = []
            analyze_expressions.build_stack(mb, head, stack, set())
            rename_stack(stack, theorem)
            return format_stack_entries(stack, prefix)


def direct(theorem, prefix=''):
    temp_chain = []
    head = analyze_expressions.disintegrate_implication(theorem, temp_chain)
    chain = [e[0] for e in temp_chain]
    mb = analyze_expressions.global_body_of_proves
    for elt in chain:
        mb = mb.simple_map[elt]
    stack = []
    analyze_expressions.build_stack(mb, head, stack, set())
    rename_stack(stack, theorem)
    return format_stack_entries(stack, prefix)

def split_at_plus(s: str) -> tuple[str, str]:
    left, sep, right = s.partition("+")
    if sep != "+":
        raise ValueError("String does not contain '+'")
    return left, right

def debugging(path_plus_end, prefix=''):
    path, end = split_at_plus(path_plus_end)

    chain = path.split(";")
    mb = analyze_expressions.global_body_of_proves
    for elt in chain:
        mb = mb.simple_map[elt]
    stack = []

    temp_lst = re.split(r"[;+]", path_plus_end)
    expr_nat_nums = ""
    for elem in temp_lst:
        if elem.startswith("(Nat"):
            expr_nat_nums = elem
            break
    expr_nat_nums = "(>[]" + expr_nat_nums + expr_nat_nums  + ")"

    analyze_expressions.build_stack(mb, end, stack, set())
    #rename_stack(stack, expr_nat_nums)
    return format_stack_entries(stack, prefix)

def mirrored(mirrored_theorem, theorem, prefix=''):
    stack = [[mirrored_theorem, "mirrored from", theorem]]
    rename_stack(stack, mirrored_theorem)
    return format_stack_entries(stack, prefix)



def generate_proof_graph_pages(theorem_list, out_dir=None):
    global theorem_to_file

    # if caller didn’t supply an explicit out_dir, use
    # "<project_root>/full_proof_graph"
    if out_dir is None:
        out_dir = PROJECT_ROOT / "files/full_proof_graph"

    # 2. convert to Path and ensure the directory exists (or will be deleted)
    #out_dir = Path(out_dir)

    # if the directory already exists, delete it and everything inside
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    # JavaScript for left-click/right-click expansion; popup named “Expression” with deep-navy styling
    popup_script = """
    <script>
    function processText(input) {
      const indentChar = "\\t";
      let indent = 0, output = "", token = "";
      for (const char of input) {
        if (char === "(") {
          if (token.trim()) { output += indentChar.repeat(indent) + token.trim() + "\\n"; token = ""; }
          output += indentChar.repeat(indent) + "(\\n"; indent++;
        } else if (char === ")") {
          if (token.trim()) { output += indentChar.repeat(indent) + token.trim() + "\\n"; token = ""; }
          indent--; output += indentChar.repeat(indent) + ")" + "\\n";
        } else {
          token += char;
        }
      }
      if (token.trim()) output += indentChar.repeat(indent) + token.trim() + "\\n";
      return output;
    }
    document.addEventListener('DOMContentLoaded', function() {
      document.body.addEventListener('contextmenu', function(e) {
        const el = e.target.closest('.clickable');
        if (!el) return;
        e.preventDefault();
        const formatted = processText(el.getAttribute('data-text'));
        const w = window.open('', 'Expression', 'width=500,height=300');
        w.document.write(`
          <!DOCTYPE html>
          <html lang="en">
            <head>
              <meta charset="UTF-8">
              <title>Expression</title>
              <style>
                body { background: #f6f8fa; color: #000080; font-family: monospace; margin: 0; }
                pre  { padding: 1em; white-space: pre-wrap; }
                .clickable    { color: lightblue; }
                .goal-highlight { color: red; font-weight: bold; }
              </style>
            </head>
            <body><pre>${formatted}</pre></body>
          </html>`);
        w.document.close();
        w.onblur = () => w.close();
      });
    });
    </script>
    """

    # Common CSS for chapter pages (original colors preserved)
    common_style = """
    <style>
      body { font-family: Arial, sans-serif; margin: 2rem; color: #000080; padding-bottom: 200em;}
      nav a { text-decoration: none; color: #0366d6; margin-right: 1em; }
      /* enforce same color & no underline for all links, visited or not */
      a, a:link, a:visited {
        color: #0366d6;
        text-decoration: none;
      }
      .var-highlight { font-style: italic; margin-bottom: 1em; display: block; }
      .step-output { margin: 1em 0; padding: 0.5em; background: #f6f8fa; border-radius: 4px; }
      .clickable    { cursor: pointer; }   /* no color overridden here */
      .goal-highlight { color: red; font-weight: bold; }
       /* only kill underlines on our autogenerated theorem links */
      a.theorem-link,
      a.theorem-link .clickable {
      text-decoration: none;
      /* force link to inherit whatever color its parent has (e.g. the red inline span) */
      color: inherit;
     }
     /* make inter-page theorem links lightblue */
     a.theorem-link[href$=".html"] .clickable {
     color: #0366d6 !important;
     }
     /* 2) but inside our .mirrored block revert to the red you use for in-page jumps */
     .mirrored a.theorem-link[href$=".html"] .clickable {
     color: #d73a49 !important;
     }
    </style>
    """

    # --- Index page ---
    index_head = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Proof Graph – Index</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; }}
    ul {{ list-style: none; padding: 0; }}
    ul ul {{ padding-left: 1.5rem; font-size: 0.9em; }}
    li {{ margin-bottom: 0.5em; }}
    a {{ text-decoration: none; color: #0366d6; }}
  </style>
  {popup_script}
</head>
<body>
  <h1>Proof Graph</h1>
  <h2>Table of Contents</h2>
  <ul>"""
    index_tail = """  </ul>
</body>
</html>"""

    toc = []
    # build mapping from each displayed theorem title → its chapter file
    theorem_to_file = {
        rename_theorem(name): f"chapter{idx}.html"
        for idx, (name, *_ ) in enumerate(theorem_list, start=1)
    }


    for idx, (name, method, _) in enumerate(theorem_list, start=1):
        filename = f"chapter{idx}.html"
        toc.append(f"    <li>{idx}. <a href='{filename}'>{html.escape(rename_theorem(name))}</a>")
        # force a new line and style it
        if not debug:
            toc.append(
                f"    <div style=\"margin-left:20px; color:#888888; font-weight:bold; font-size:1.3em;\">"
                f"{html.escape(visu_helpers.make_readable_title(rename_theorem(name)))}</div>"
            )

        if method.lower() == "induction":
            toc.append("      <ul>")
            toc.append(f"        <li>{idx}.1. <a href='{filename}#sub1'>Check for 0</a></li>")
            toc.append(f"        <li>{idx}.2. <a href='{filename}#sub2'>Check induction condition</a></li>")
            toc.append("      </ul>")
        toc.append("    </li>")

    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join([index_head] + toc + [index_tail]))

    # --- Chapter pages ---
    for idx, (name, method, var) in enumerate(theorem_list, start=1):
        filename = f"chapter{idx}.html"
        prev_link = f"<a href='chapter{idx - 1}.html'>Previous</a>" if idx > 1 else ""
        next_link = f"<a href='chapter{idx + 1}.html'>Next</a>" if idx < len(theorem_list) else ""
        nav_links = ' '.join(link for link in (prev_link, next_link) if link)

        head = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(rename_theorem(name))}</title>
  {f'<title>{html.escape(visu_helpers.make_readable_title(rename_theorem(name)))}</title>' if not debug else ''}
  {common_style}
  {popup_script}
</head>
<body>
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <nav>
      <a href="index.html">Index</a> {nav_links}
    </nav>
    <div>
      <span style="margin-right:1em;"><span style="color:red; font-weight:bold;">■</span> Goal of the proof</span>
      <span style="margin-right:1em;"><span style="color:lightblue;">■</span> Has justification link</span>
      <span style="margin-right:1em;"><span style="color:#888888;">■</span> Readable version</span>
      <span>Right-click to expand</span>
    </div>
  </div>
  <h1>Chapter {idx}: {html.escape(rename_theorem(name))}</h1>
  {f'''<div style="margin-left:20px; color:#888888; font-weight:bold; font-size:3em;">
    {html.escape(visu_helpers.make_readable_title(rename_theorem(name)))}
  </div><br><br>''' if not debug else ''}
"""

        body = [head]
        if method.lower() == "induction":
            body.extend([
                f"  <span class=\"var-highlight\">Induction variable: v{html.escape(var)}</span>",
                "  <h2 id=\"sub1\">Check for 0</h2>",
                f"  <div class=\"step-output\">{check_zero(name, var, f'c{idx}s1')}</div>",
                "  <h2 id=\"sub2\">Check induction condition</h2>",
                f"  <div class=\"step-output\">{check_induction_condition(name, var, f'c{idx}s2')}</div>",
            ])

        elif method.lower() == "direct":
            body.extend([
                "  <h2>Direct Proof</h2>",
                "  <div class='step-output'>", direct(name), "  </div>",
            ])
        elif method.lower() == "debug":
            body.extend([
                "  <h2>Debugging</h2>",
                "  <div class='step-output'>", debugging(name), "  </div>",
            ])
        elif method.lower() == "mirrored statement":
                body.extend([
                "  <h2>Mirrored</h2>",
                "  <div class='step-output'>",
                mirrored(name, var),
                "  </div>",
                ])
        body.append("</body>")
        body.append("</html>")

        with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as f:
            f.write("\n".join(body))



