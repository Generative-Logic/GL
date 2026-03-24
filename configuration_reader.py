from __future__ import annotations

"""
configuration_reader.py

Default behavior:
- Initializes to an EMPTY configuration (no expressions, no patterns) unless a config file
  path is explicitly provided and exists.

Reads (when a config is provided):
  - Core expression definitions
  - Top-level "parameters" block (all params except 'debug')
    * includes: simple_facts_parameters (List[int])
  - Top-level "patterns_to_exclude": list of regex strings (JSON-escaped)
  - Top-level "only_in_head": list of LITERAL substrings (e.g., "(=[")
  - Top-level "prohibited_combinations": list of 2-string arrays
      e.g., [["in3","preorder"], ["foo","bar"]]
    -> exposed as List[Set[str]] with each pair turned into a 2-element set

Main class: configuration_reader
- Behaves like a Mapping[str, ExpressionDescription] for expressions.
- Exposes:
    .parameters                  -> ConfigurationParameters
    .patterns_to_exclude_raw     -> List[str]
    .patterns_to_exclude         -> List[re.Pattern] (compiled as regex)
    .only_in_head_raw            -> List[str] (literal substrings)
    .only_in_head_patterns       -> List[re.Pattern] (compiled with re.escape)
    .prohibited_combinations     -> List[Set[str]]
    .anchor_id                   -> str (Extracted from filename, e.g. "Peano" from "ConfigPeano.json")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Iterator, Optional, List, Mapping, Tuple, Set, Union
from collections.abc import Mapping as MappingABC
import json
import re

# ---------- Paths (adjust if your layout differs) ----------
PROJECT_ROOT = Path(__file__).resolve().parent
# Default config is NOW EMPTY by default: no file loaded unless explicitly provided.
DEFAULT_CONFIG: Optional[Path] = None
DEFAULT_DEFS_DIR = PROJECT_ROOT / "files" / "definitions"


# -------- helpers --------
def _normalize_mpl(s: str) -> str:
    """Normalize MPL-like text by removing newlines, spaces, and tabs."""
    return s.replace("\n", "").replace(" ", "").replace("\t", "")


def make_anchor_signature(signature: str) -> str:
    """
    '(AnchorPeano[N,i0,s,+,*,i1])' -> '(AnchorPeano[1,2,3,4,5,6])'.
    Keeps the head and '(...[ ... ])' structure; only the arg list is replaced with 1..n.
    """
    s = signature.strip()
    l = s.find('[')
    r = s.rfind(']')
    if l == -1 or r == -1 or r < l:
        raise ValueError(f"Bad signature format: {signature!r}")
    args_str = s[l + 1:r].strip()
    n = 0 if args_str == "" else sum(1 for a in args_str.split(',') if a.strip() != "")
    new_args = ",".join(str(i) for i in range(1, n + 1))
    return f"{s[:l+1]}{new_args}{s[r:]}"


def _parse_args_list(spec: Dict[str, Any], key: str) -> List[str]:
    """
    Read spec[key] as a JSON array (or tolerate a comma-separated string),
    return a List[str] of non-empty, stripped items. Missing/None -> [].
    Order is preserved; duplicates are kept.
    """
    raw = spec.get(key, [])
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return [t for t in (p.strip() for p in raw.split(',')) if t]
    return []


def _parse_int_list(raw: Any) -> List[int]:
    """
    Parse a JSON value into List[int]. Accepts a list of ints/strings or a comma-separated string.
    Invalid items are ignored.
    """
    out: List[int] = []
    if raw is None:
        return out
    if isinstance(raw, list):
        for v in raw:
            try:
                out.append(int(str(v).strip()))
            except (TypeError, ValueError):
                pass
        return out
    if isinstance(raw, str):
        for tok in raw.split(','):
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.append(int(tok))
            except ValueError:
                pass
        return out
    return out


def _coerce_definition_sets(raw: Any) -> Dict[str, Tuple[str, bool, bool]]:
    """
    Accept formats:
      Legacy: {"1": "(1)", ...}
      Two:    {"1": ["(1)", true], ...}
      Three:  {"1": ["(1)", true, true], ...}
    Returns {key: (text, combinable_flag, connectable_flag)}.
    Missing flags default to True.
    The third flag (connectable) controls whether the anchor position is
    available for conjecture mapping. When False, no expression can map
    to this anchor arg during conjecture generation, but the prover still
    sees the full anchor.
    """
    out: Dict[str, Tuple[str, bool, bool]] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        key = str(k)
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                out[key] = ("", True, True)
            elif len(v) == 1:
                out[key] = (str(v[0]), True, True)
            elif len(v) == 2:
                out[key] = (str(v[0]), bool(v[1]), True)
            else:
                out[key] = (str(v[0]), bool(v[1]), bool(v[2]))
        else:
            out[key] = (str(v), True, True)
    return out


def _extract_arg_list(signature_raw: str) -> List[str]:
    """
    From a short_mpl string like '(fold[N,s,+,f,n,m,p])',
    return ['N','s','+','f','n','m','p'] (whitespace-insensitive).
    Returns [] if brackets are missing.
    """
    s = _normalize_mpl(signature_raw)
    l = s.find('[')
    r = s.rfind(']')
    if l == -1 or r == -1 or r < l:
        return []
    inside = s[l + 1:r]
    if not inside:
        return []
    return [tok for tok in (part.strip() for part in inside.split(',')) if tok]


def _align_and_sort_args(names: List[str], ordered_args: List[str]) -> Tuple[List[str], List[int]]:
    """
    Return (sorted_names, sorted_indices), ordered by the 0-based first occurrence
    of each name in `ordered_args`. Missing names are dropped; duplicates are kept.
    Sorting is stable for ties (same index).
    """
    first_index: Dict[str, int] = {}
    for i, arg in enumerate(ordered_args):
        if arg not in first_index:
            first_index[arg] = i

    triples = [(first_index[n], pos, n) for pos, n in enumerate(names) if n in first_index]
    triples.sort(key=lambda t: (t[0], t[1]))  # sort by index, then original position for stability

    sorted_indices = [idx for idx, _, _ in triples]
    sorted_names = [n for _, _, n in triples]
    return sorted_names, sorted_indices


def _build_index_list(names: List[str], ordered_args: List[str]) -> List[int]:
    """
    For each name in `names`, return the 0-based index of its FIRST occurrence in ordered_args.
    Names not present are skipped (i.e., produce no index). Output order matches `names`.
    """
    first_index: Dict[str, int] = {}
    for i, arg in enumerate(ordered_args):  # zero-based
        if arg not in first_index:
            first_index[arg] = i
    return [first_index[n] for n in names if n in first_index]


def _compile_patterns(lst: List[str]) -> List[re.Pattern]:
    """Compile regex strings; ignore empties; keep only successfully compiled patterns."""
    compiled: List[re.Pattern] = []
    for pat in lst:
        s = str(pat).strip()
        if not s:
            continue
        try:
            compiled.append(re.compile(s))
        except re.error:
            # Ignore invalid regex entries silently (or log if desired)
            pass
    return compiled


def _compile_literal_substrings(lst: List[str]) -> List[re.Pattern]:
    """
    Compile a list of literal substring markers into regex patterns
    using re.escape so special chars are matched literally.
    """
    out: List[re.Pattern] = []
    for s in lst:
        s = str(s)
        if not s:
            continue
        out.append(re.compile(re.escape(s)))
    return out


def _parse_prohibited_combinations(block: Any) -> List[Set[str]]:
    """
    Accepts a list of 2-element arrays/tuples like:
        [["in3","preorder"], ["foo","bar"]]
    Returns a List[Set[str]], one 2-element set per pair. Invalid entries ignored.
    """
    res: List[Set[str]] = []
    if isinstance(block, list):
        for item in block:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                a = str(item[0]).strip()
                b = str(item[1]).strip()
                if a and b:
                    res.append({a, b})
    return res


# -------- configuration objects --------
@dataclass(frozen=True)
class ConfigurationParameters:
    min_number_simple_expressions: int = 2
    max_number_simple_expressions: int = 0
    max_size_mapping_def_set: int = 0
    max_number_args_expr: int = 0
    operator_threshold: int = 0
    max_values_for_def_sets: Dict[str, int] = field(default_factory=dict)
    max_values_for_uncomb_def_sets: Dict[str, int] = field(default_factory=dict)
    max_values_for_def_sets_prior_connection: Dict[str, int] = field(default_factory=dict)
    max_complexity_if_anchor_parameter_connected: Dict[str, int] = field(default_factory=dict)
    max_size_binary_list: int = 0
    simple_facts_parameters: List[int] = field(default_factory=list)
    fact_variable_kinds: List[str] = field(default_factory=list)
    incubator_mode: bool = False

    @staticmethod
    def from_json(d: Mapping[str, Any]) -> "ConfigurationParameters":
        # ignore 'debug' if present
        return ConfigurationParameters(
            min_number_simple_expressions=int(d.get("min_number_simple_expressions", 2)),
            max_number_simple_expressions=int(d.get("max_number_simple_expressions", 0)),
            max_size_mapping_def_set=int(d.get("max_size_mapping_def_set", 0)),
            max_number_args_expr=int(d.get("max_number_args_expr", 0)),
            operator_threshold=int(d.get("operator_threshold", 0)),
            max_values_for_def_sets=dict(d.get("max_values_for_def_sets", {})),
            max_values_for_uncomb_def_sets=dict(d.get("max_values_for_uncomb_def_sets", {})),
            max_values_for_def_sets_prior_connection=dict(d.get("max_values_for_def_sets_prior_connection", {})),
            max_complexity_if_anchor_parameter_connected=dict(
                d.get("max_complexity_if_anchor_parameter_connected", {})
            ),
            max_size_binary_list=int(d.get("max_size_binary_list", 0)),
            simple_facts_parameters=_parse_int_list(d.get("simple_facts_parameters", [])),
            fact_variable_kinds=list(d.get("fact_variable_kinds", [])),
            incubator_mode=bool(d.get("incubator_mode", False)),
        )


@dataclass(frozen=True)
class ExpressionDescription:
    """
    Data for a core expression.
    """
    arity: int = 0
    definition_sets: Dict[str, Tuple[str, bool, bool]] = field(default_factory=dict)
    full_mpl: str = ""
    handle: str = ""
    short_mpl_raw: str = ""
    short_mpl_normalized: str = ""
    max_count_per_conjecture: int = 0
    max_size_expression: int = 0
    min_size_expression: int = 1
    input_args: List[str] = field(default_factory=list)
    output_args: List[str] = field(default_factory=list)
    indices_input_args: List[int] = field(default_factory=list)   # 0-based positions
    indices_output_args: List[int] = field(default_factory=list)  # 0-based positions


class configuration_reader(MappingABC):
    """
    Mapping-like registry of expressions + configuration parameters + exclusion/marker patterns.

    On construction:
      - If no config path is provided (or the file does not exist), initialize EMPTY.
      - Otherwise, read the JSON config and populate structures as usual.
    """

    def __init__(
        self,
        config_path: Optional[Union[Path, str]] = DEFAULT_CONFIG,
        definitions_dir: Optional[Union[Path, str]] = DEFAULT_DEFS_DIR,
    ) -> None:
        self.config_path: Optional[Path] = Path(config_path) if config_path else None
        self.definitions_dir: Path = Path(definitions_dir) if definitions_dir else (PROJECT_ROOT / "files" / "definitions")
        self.data: Dict[str, ExpressionDescription] = {}
        self.parameters: ConfigurationParameters = ConfigurationParameters()
        self.patterns_to_exclude_raw: List[str] = []
        self.patterns_to_exclude: List[re.Pattern] = []
        self.only_in_head_raw: List[str] = []
        self.only_in_head_patterns: List[re.Pattern] = []
        self.prohibited_combinations: List[Set[str]] = []
        self.prohibited_heads: List[str] = []
        self.theorems_folder: Optional[str] = None
        self.background_theorems_folder: Optional[str] = None
        self.anchor_name: Optional[str] = None
        self.anchor_id: str = ""

        # Auto-load if a valid path is supplied and exists; otherwise remain empty.
        if self.config_path and self.config_path.exists():
            stem = self.config_path.stem
            if stem.startswith("Config"):
                self.anchor_id = stem[6:]
            self.load_from_config()
        else:
            # Empty init (no-op) — structures already set to defaults above.
            pass

    # ---------- public API ----------

    def load_from_config(self, path: Optional[Union[Path, str]] = None) -> None:
        """
        (Re)load expressions, parameters, and patterns from the JSON configuration file.
        If `path` is provided, update self.config_path first.
        If no config path is set or file missing, reset to EMPTY and return.
        """
        if path is not None:
            self.config_path = Path(path)
        if not self.config_path or not self.config_path.exists():
            # reset to empty
            self.data.clear()
            self.parameters = ConfigurationParameters()
            self.patterns_to_exclude_raw = []
            self.patterns_to_exclude = []
            self.only_in_head_raw = []
            self.only_in_head_patterns = []
            self.prohibited_combinations = []
            return

        obj: Dict[str, Any] = json.loads(self.config_path.read_text(encoding="utf-8"))

        # --- parameters (top-level) ---
        params_block = obj.pop("parameters", {})
        self.parameters = ConfigurationParameters.from_json(params_block)

        # --- patterns_to_exclude (top-level, regex strings) ---
        patterns_block = obj.pop("patterns_to_exclude", [])
        if isinstance(patterns_block, list):
            self.patterns_to_exclude_raw = [str(x) for x in patterns_block]
        elif isinstance(patterns_block, str):
            self.patterns_to_exclude_raw = [patterns_block]
        else:
            self.patterns_to_exclude_raw = []
        self.patterns_to_exclude = _compile_patterns(self.patterns_to_exclude_raw)

        # --- only_in_head (top-level, literal substrings) ---
        only_head_block = obj.pop("only_in_head", [])
        if isinstance(only_head_block, list):
            self.only_in_head_raw = [str(x) for x in only_head_block]
        elif isinstance(only_head_block, str):
            self.only_in_head_raw = [only_head_block]
        else:
            self.only_in_head_raw = []
        self.only_in_head_patterns = _compile_literal_substrings(self.only_in_head_raw)

        # --- prohibited_combinations (top-level, list of 2-elem arrays) ---
        self.prohibited_combinations = _parse_prohibited_combinations(obj.pop("prohibited_combinations", []))

        # Reuse _parse_args_list helper which handles ["a", "b"] or "a,b"
        self.prohibited_heads = _parse_args_list(obj, "prohibited_heads")

        # --- optional folder overrides ---
        self.theorems_folder = obj.pop("theorems_folder", None)
        self.background_theorems_folder = obj.pop("background_theorems_folder", None)
        self.anchor_name = obj.pop("anchor_name", None)

        # --- expressions ---
        new_map: Dict[str, ExpressionDescription] = {}

        for name, spec in obj.items():
            # skip anything non-object (paranoia)
            if not isinstance(spec, dict):
                continue

            # --- arity ---
            arity = int(spec.get("arity", 0))

            # --- definition_sets ---
            definition_sets: Dict[str, Tuple[str, bool]] = _coerce_definition_sets(spec.get("definition_sets", {}))

            # --- full_mpl (definition): inline or file path -> normalized text ---
            full_mpl_value = self._read_full_mpl_from_config(str(spec.get("full_mpl", "")).strip())
            full_mpl_norm = _normalize_mpl(full_mpl_value)

            # --- handle: from JSON (ensure it ends with '[' if present) ---
            handle = str(spec.get("handle", f"({name}[")).strip()
            if handle and not handle.endswith("["):
                if handle.endswith(name):
                    handle = f"({name}["
                elif handle.endswith("("):
                    handle = f"{handle}{name}["
                else:
                    handle = handle + "["

            # --- short_mpl fields ---
            short_mpl_raw = str(spec.get("short_mpl", "")).strip()
            if short_mpl_raw:
                try:
                    short_mpl_normalized = make_anchor_signature(_normalize_mpl(short_mpl_raw))
                except ValueError:
                    short_mpl_normalized = _normalize_mpl(short_mpl_raw)
            else:
                short_mpl_normalized = ""

            # --- extras ---
            max_count_per_conjecture = int(spec.get("max_count_per_conjecture", 0))
            max_size_expression = int(spec.get("max_size_expression", 0))
            min_size_expression = int(spec.get("min_size_expression", 1))
            input_args: List[str] = _parse_args_list(spec, "input_args")
            output_args: List[str] = _parse_args_list(spec, "output_args")

            # Build index lists based on the RAW short_mpl argument order (0-based)
            ordered_args = _extract_arg_list(short_mpl_raw)

            # Keep args & indices aligned + ascending by signature position
            input_args, indices_input_args = _align_and_sort_args(input_args, ordered_args)
            output_args, indices_output_args = _align_and_sort_args(output_args, ordered_args)

            new_map[str(name)] = ExpressionDescription(
                arity=arity,
                definition_sets=definition_sets,
                full_mpl=full_mpl_norm,
                handle=handle,
                short_mpl_raw=short_mpl_raw,
                short_mpl_normalized=short_mpl_normalized,
                max_count_per_conjecture=max_count_per_conjecture,
                max_size_expression=max_size_expression,
                min_size_expression=min_size_expression,
                input_args=input_args,
                output_args=output_args,
                indices_input_args=indices_input_args,
                indices_output_args=indices_output_args,
            )

        self.data = new_map

    # Mapping interface for expressions
    def __getitem__(self, key: str) -> ExpressionDescription:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def get(self, key: str, default: Optional[ExpressionDescription] = None) -> Optional[ExpressionDescription]:
        return self.data.get(key, default)

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    # ---------- helpers ----------

    def _read_full_mpl_from_config(self, raw: str) -> str:
        """
        If 'raw' looks like a path/filename (endswith .txt or contains a path sep),
        read file content. Otherwise treat it as inline MPL text (may be empty).

        Resolution order for relative filenames:
          1) <definitions_dir>/<basename>
          2) <project_root>/<raw>
          3) <project_root>/<basename>
        """
        if not raw:
            return ""
        looks_like_file = raw.endswith(".txt") or ("/" in raw) or ("\\" in raw)
        if not looks_like_file:
            return raw  # inline MPL text

        p = Path(raw)
        candidates = []
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.extend([
                self.definitions_dir / p.name,
                PROJECT_ROOT / raw,
                PROJECT_ROOT / p.name,
            ])
        for c in candidates:
            if c.is_file():
                return c.read_text(encoding="utf-8")
        # Fall back to the raw content
        return raw


__all__ = [
    "ConfigurationParameters",
    "ExpressionDescription",
    "configuration_reader",
    "PROJECT_ROOT",
    "DEFAULT_CONFIG",
    "DEFAULT_DEFS_DIR",
]