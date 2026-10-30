# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Visual theme of the proof books: color roles, type families and the skin.

Every color of a proof page names a role (`--gl-link`, `--gl-goal`, ...). The
day side, warm paper, is the default; the night side, a dusk sky, follows the
reader's dark-mode setting. The generators' own CSS refers only to these roles,
so one theme definition serves the index, chapter, tag and Lean pages.

`THEME_TOKENS_CSS` opens every page's first style block. `THEME_SKIN_CSS`
closes the last one: it carries the look (type, header band, cards, controls)
and wins over the layout rules before it, which keep the reading mechanics.
No web font is linked — the pages stay self-contained — so every family falls
back to a locally installed relative (Futura / Century Gothic, Palatino,
Cascadia / Menlo / Consolas).
"""

THEME_TOKENS_CSS = """
:root {
  color-scheme: light dark;
  --gl-font-ui: "Jost", "Futura", "Futura PT", "Century Gothic", "Avenir Next", "Segoe UI", system-ui, sans-serif;
  --gl-font-math: "Newsreader", "Palatino Linotype", "Book Antiqua", Palatino, "STIX Two Text", Cambria, Georgia, serif;
  --gl-font-mono: "IBM Plex Mono", "Cascadia Mono", "SF Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace;
  --gl-horizon: linear-gradient(90deg, #7E2716 0%, #C2452A 16%, #E5672D 34%, #FFB54A 52%, #EF9AB3 74%, #5B2A55 100%);
  --gl-bg: #F7EEDF; --gl-bg-rgb: 247 238 223; --gl-glow-rgb: 255 176 74;
  --gl-surface: #FFF9EF; --gl-surface-hover: #FBF0DC; --gl-surface-sunken: #F0E3CC;
  --gl-rule: #E2CCA8; --gl-rule-strong: #CDB088;
  --gl-text: #22162A; --gl-text-common: #4A3A4D; --gl-text-soft: #3B2C40; --gl-prose: #4A3A4D; --gl-label: #3B2C40;
  --gl-text-muted: #6E5B68; --gl-text-quiet: #725F6B; --gl-text-faint: #7A6672;
  --gl-accent: #A8391F;
  --gl-link: #386460; --gl-link-strong: #2C514D;
  --gl-goal: #B5420F; --gl-goal-rgb: 229 103 45; --gl-goal-bg: #FBE2AE;
  --gl-target: #7A2E86; --gl-target-rgb: 122 46 134;
  --gl-external: #8E3D63;
  --gl-namespace: #7A5A12; --gl-namespace-rgb: 194 140 30;
  --gl-changed: #B23A1E;
  --gl-control-bg: #E8EDEA; --gl-control-bg-hover: #D8E1DD; --gl-control-border: #386460;
  --gl-tint: rgb(34 22 42 / 0.045); --gl-shadow: rgb(74 38 20 / 0.24);
  --gl-depth-1: #386460; --gl-depth-1-bg: #FFF9EF;
  --gl-depth-2: #D9792E; --gl-depth-2-bg: #FFF6E8;
  --gl-depth-3: #A6466A; --gl-depth-3-bg: #FCF1EE;
  --gl-depth-4: #3E6E9E; --gl-depth-4-bg: #F4F2EE;
  --gl-depth-5: #C2452A; --gl-depth-5-bg: #FCF0E6;
  --gl-depth-6: #7A6672; --gl-depth-6-bg: #F6EEE4;
}
@media (prefers-color-scheme: dark) {
  :root {
    --gl-bg: #1C1124; --gl-bg-rgb: 28 17 36; --gl-glow-rgb: 229 103 45;
    --gl-surface: #26182E; --gl-surface-hover: #301F39; --gl-surface-sunken: #160D1C;
    --gl-rule: #3D2A46; --gl-rule-strong: #573E60;
    --gl-text: #F7EEDF; --gl-text-common: #DCCBD3; --gl-text-soft: #EADCCF; --gl-prose: #D8C6CF; --gl-label: #E6D2DA;
    --gl-text-muted: #AE95A6; --gl-text-quiet: #A58C9C; --gl-text-faint: #93798C;
    --gl-accent: #FF9B5A;
    --gl-link: #96BCB4; --gl-link-strong: #BAD6D0;
    --gl-goal: #FFB54A; --gl-goal-rgb: 255 181 74; --gl-goal-bg: #3E2A16;
    --gl-target: #E39AE6; --gl-target-rgb: 227 154 230;
    --gl-external: #EF9AB3;
    --gl-namespace: #F2CC6B; --gl-namespace-rgb: 242 204 107;
    --gl-changed: #FF8A5B;
    --gl-control-bg: #2F4643; --gl-control-bg-hover: #3A5652; --gl-control-border: #96BCB4;
    --gl-tint: rgb(255 240 220 / 0.045); --gl-shadow: rgb(0 0 0 / 0.55);
    --gl-depth-1: #96BCB4; --gl-depth-1-bg: #26182E;
    --gl-depth-2: #F8923C; --gl-depth-2-bg: #2C1B2C;
    --gl-depth-3: #EF9AB3; --gl-depth-3-bg: #2E1A30;
    --gl-depth-4: #8FB0E0; --gl-depth-4-bg: #241A33;
    --gl-depth-5: #E5672D; --gl-depth-5-bg: #2E1A26;
    --gl-depth-6: #AE95A6; --gl-depth-6-bg: #281C2E;
  }
}
"""

THEME_SKIN_CSS = """
body {
  font-family: var(--gl-font-ui);
  line-height: 1.5;
  background-color: var(--gl-bg);
  background-image:
    radial-gradient(1000px 460px at 6% -60px, rgb(var(--gl-glow-rgb) / 0.17), transparent 72%),
    radial-gradient(760px 380px at 104% -40px, rgb(var(--gl-target-rgb) / 0.07), transparent 70%);
  background-repeat: no-repeat;
  color: var(--gl-text);
}
::selection { background: rgb(var(--gl-goal-rgb) / 0.30); }
:focus-visible { outline: 2px solid var(--gl-goal); outline-offset: 3px; }
a:hover { color: var(--gl-link-strong); }

h1 { font-weight: 500; font-size: 1.5rem; line-height: 1.4; letter-spacing: 0.01em; margin: 1em 0 0.45em; color: var(--gl-text); }
h1 .clickable { font-family: var(--gl-font-mono); font-weight: 500; font-size: 0.84em; color: var(--gl-text-soft); }
h2 { font-weight: 600; font-size: 1rem; letter-spacing: 0.16em; text-transform: uppercase; color: var(--gl-accent); margin: 1.8em 0 0.8em; }
h2::before { content: ""; display: inline-block; width: 1.7em; height: 3px; margin: 0 0.65em 0.2em 0; vertical-align: middle; border-radius: 2px; background: var(--gl-horizon); }
h3 { font-weight: 600; letter-spacing: 0.02em; }

nav a, nav a:link, nav a:visited { font-weight: 600; font-size: 0.78rem; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--gl-link); margin-right: 1.3em; padding: 0.3em 0; border-bottom: 1.5px solid transparent; }
nav a:hover { border-bottom-color: currentColor; }
nav::before { content: ""; display: inline-block; width: 1.9em; height: 1.9em; margin: 0 1.1em 0 0; vertical-align: middle;
  background: url(gl-logo.png) center / contain no-repeat; }

.reader-toolbar {
  background: rgb(var(--gl-bg-rgb) / 0.97);
  -webkit-backdrop-filter: saturate(1.15) blur(8px); backdrop-filter: saturate(1.15) blur(8px);
  border-bottom: 3px solid transparent; border-image: var(--gl-horizon) 1;
  box-shadow: 0 16px 30px -26px var(--gl-shadow);
  margin: 0 -2rem; padding: 0.7rem 2rem 0.45rem;
}
.reader-toolbar > div:first-child > div { font-size: 0.8rem; letter-spacing: 0.02em; }

.common-notation-control { border: 1.5px solid var(--gl-control-border); border-radius: 16px; background: var(--gl-control-bg);
  color: var(--gl-text); box-shadow: 0 8px 22px -18px var(--gl-shadow); font-weight: 600; }
.common-notation-control strong { color: var(--gl-link-strong); letter-spacing: 0.03em; }
.common-notation-control:hover { background: var(--gl-control-bg-hover); }

.reader-page-control, .subproof-controls button {
  font-family: var(--gl-font-ui); font-weight: 600; font-size: 0.78rem; letter-spacing: 0.13em; text-transform: uppercase;
  border: 1.5px solid var(--gl-control-border); border-radius: 999px; padding: 0.55rem 1.15rem;
  background: var(--gl-surface); color: var(--gl-link-strong); box-shadow: 0 1px 0 var(--gl-tint);
}
.reader-page-control:hover, .subproof-controls button:hover { background: var(--gl-control-bg); color: var(--gl-link-strong); border-color: var(--gl-control-border); }

.clickable, .mpl-line, .mpl-only, #expr-content { font-family: var(--gl-font-mono); }
.mpl-line, .proof-line-content > .clickable, .proof-line-content > a > .clickable { font-size: 0.88em; line-height: 1.6; }
.readable-grey { font-family: var(--gl-font-math); }
.rf-label, .reader-label, .reader-prose, .rf-target-label { font-family: var(--gl-font-ui); }
.rf-label { letter-spacing: 0.14em; }
.step-badge, .common-step-badge, .block-number { font-family: var(--gl-font-mono); }
.step-badge { font-size: 0.68rem; color: var(--gl-text-faint); }
.validity-tag { font-family: var(--gl-font-ui); letter-spacing: 0.02em; }

a.proof-tag, a.proof-tag:link, a.proof-tag:visited {
  font-family: var(--gl-font-ui); font-size: 0.86em; letter-spacing: 0.04em; color: var(--gl-text);
  border: 1px solid var(--gl-rule-strong); border-radius: 5px; padding: 0.02em 0.4em; background: var(--gl-surface-sunken);
}
a.proof-tag:hover { color: var(--gl-link-strong); border-color: var(--gl-link); }

.step-output {
  background: var(--gl-surface); border: 1px solid var(--gl-rule); border-radius: 18px; padding: 0.4em 1.2em 1em;
  box-shadow: 0 1px 0 var(--gl-tint), 0 18px 36px -32px var(--gl-shadow);
}
.step-output > .reader-section.subproofs-section { border-top: 0; padding-top: 0.2em; }
.reader-section h3 { font-size: 1.12rem; margin: 0.6em 0 0.4em; }
.reader-section h3 .block-number { color: var(--gl-accent); }
.reader-section .reader-counts { font-size: 0.86rem; letter-spacing: 0.03em; }
.subproof-card { border: 1px solid var(--gl-rule); border-left-width: 4px; border-radius: 14px;
  box-shadow: 0 1px 0 var(--gl-tint), 0 12px 26px -24px var(--gl-shadow); }
.subproof-depth-1 { border-left-color: var(--gl-depth-1); background: var(--gl-depth-1-bg); }
.subproof-depth-2 { border-left-color: var(--gl-depth-2); background: var(--gl-depth-2-bg); }
.subproof-depth-3 { border-left-color: var(--gl-depth-3); background: var(--gl-depth-3-bg); }
.subproof-depth-4 { border-left-color: var(--gl-depth-4); background: var(--gl-depth-4-bg); }
.subproof-depth-5 { border-left-color: var(--gl-depth-5); background: var(--gl-depth-5-bg); }
.subproof-title { color: var(--gl-text); }
.reader-section .reader-block-kind { font-size: 0.82em; letter-spacing: 0.14em; text-transform: uppercase; color: var(--gl-link-strong); }
.subproof-explanation { border-left-color: var(--gl-rule-strong); border-radius: 0 8px 8px 0; }
.chapter-stats { border-top: 1px dashed var(--gl-rule-strong); border-radius: 12px; background: var(--gl-surface-sunken);
  font-size: 0.82rem; letter-spacing: 0.02em; }
.used-by-panel { border-radius: 0 10px 10px 0; }

#theorem-list > li { padding: 0.55em 0 0.75em; border-bottom: 1px solid var(--gl-rule); }
#theorem-list .clickable { font-size: 0.92em; overflow-wrap: anywhere; }
#theorem-list > li:last-child { border-bottom: 0; }
#theorem-search:focus { border-color: var(--gl-link) !important; box-shadow: 0 0 0 3px rgb(var(--gl-goal-rgb) / 0.18); }

dialog { border-radius: 14px !important; }
dialog::backdrop { background: rgb(20 12 26 / 0.42); }
pre.lean, .lean-card { border-radius: 14px; }
"""
