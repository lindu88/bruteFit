# processrecord/fileutils.py
# SAM TODO - double check, used LLM. 

from typing import Dict, Iterable, Optional, Tuple, List
import pandas as pd
import os, re

__all__ = ["guess_roles_from_filenames", "read_pos_neg_abs_from_paths"]

_ROLE_KEYS = ("pos", "neg", "abs", "sticks")

def _read_csv_default(fp: str, **kwargs) -> pd.DataFrame:
    kw = dict(engine="c", low_memory=False)
    kw.update(kwargs)
    return pd.read_csv(fp, **kw)

def guess_roles_from_filenames(paths: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for fp in paths:
        lower = os.path.basename(fp).lower()
        for key in _ROLE_KEYS:
            if re.search(rf"(^|[^a-z]){re.escape(key)}([^a-z]|$)", lower) or key in lower:
                out[key] = fp
    return out

def _read_with_optional_headers(
    fp: str,
    csv_reader,
    names: Optional[List[str]],
    assume_no_header: bool,
) -> pd.DataFrame:
    """
    If names is provided OR assume_no_header is True, read with header=None
    and then assign names (if given).
    """
    if names is not None or assume_no_header:
        df = csv_reader(fp, header=None)
        if names is not None:
            if len(names) > df.shape[1]:
                raise ValueError(
                    f"{os.path.basename(fp)} has {df.shape[1]} columns but {len(names)} names were provided."
                )
            # Trim or extend: keep first len(names) columns, drop extras
            df = df.iloc[:, : len(names)]
            df.columns = names
        return df
    # Default: let pandas infer header from first row
    return csv_reader(fp)

def read_pos_neg_abs_from_paths(
    mapping: Dict[str, str],
    *,
    id_hint: Optional[str] = None,
    csv_reader=_read_csv_default,
    column_names: Optional[Dict[str, List[str]]] = None,
    assume_no_header: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], str]:
    """
    Read CSVs with optional, explicit column schemas per role.

    Parameters
    ----------
    mapping : dict
        Required: 'pos', 'neg', 'abs'. Optional: 'sticks'.
    id_hint : str | None
        Identifier override.
    csv_reader : callable
        Usually pandas.read_csv-like.
    column_names : dict[str, list[str]] | None
        Optional per-role explicit column names:
          e.g., {
             "pos":    ["wavelength","x_pos","y_pos","R","theta","std_dev_x","std_dev_y","additional"],
             "neg":    ["wavelength","x_neg","y_neg","R","theta","std_dev_x","std_dev_y","additional"],
             "abs":    ["wavelength","intensity"],
             "sticks": ["wavelength","strength"],
          }
    assume_no_header : bool
        If True, read all files with header=None (first row is data).

    Returns
    -------
    (pos_df, neg_df, abs_df, sticks_df_or_None, ident)
    """
    for need in ("pos", "neg", "abs"):
        if need not in mapping or not mapping[need]:
            raise ValueError(f"Missing required file path for role: '{need}'")
        if not os.path.exists(mapping[need]):
            raise FileNotFoundError(f"Path for role '{need}' does not exist: {mapping[need]}")

    names = column_names or {}

    pos_df = _read_with_optional_headers(
        mapping["pos"], csv_reader, names.get("pos"), assume_no_header
    )
    neg_df = _read_with_optional_headers(
        mapping["neg"], csv_reader, names.get("neg"), assume_no_header
    )
    abs_df = _read_with_optional_headers(
        mapping["abs"], csv_reader, names.get("abs"), assume_no_header
    )
    sticks_df = None
    if "sticks" in mapping and mapping["sticks"]:
        sticks_df = _read_with_optional_headers(
            mapping["sticks"], csv_reader, names.get("sticks"), assume_no_header
        )

    if id_hint:
        ident = str(id_hint)
    else:
        stem_src = mapping.get("abs") or mapping.get("pos") or mapping.get("neg") or next(iter(mapping.values()))
        ident = os.path.splitext(os.path.basename(stem_src))[0]

    return pos_df, neg_df, abs_df, sticks_df, ident
