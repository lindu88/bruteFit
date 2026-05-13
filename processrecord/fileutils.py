from typing import Any
import pandas as pd
import os
import re
from PySide6.QtWidgets import QFileDialog, QMessageBox
from pandas import DataFrame


def _normalize_metadata_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


def _parse_first_float(text: str) -> float | None:
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text))
    if not match:
        return None
    return float(match.group(0))


def _parse_processing_metadata_text(text: str, run_label: str | None = None) -> dict[str, Any]:
    """
    Parse processing metadata from simple config files.

    Preferred format is flat key=value lines:
        lims_ID=PdTPP
        concentration_MOL_L=1e-6
        pathlength_cm=0.2
        field_B=1

    The older human-readable compound txt files are still accepted as a fallback so
    existing data folders do not need to be rewritten all at once.
    """
    parsed: dict[str, Any] = {}
    key_aliases = {
        "lims": "lims_ID",
        "lims_id": "lims_ID",
        "limsid": "lims_ID",
        "name": "lims_ID",
        "conc": "concentration_MOL_L",
        "concentration": "concentration_MOL_L",
        "concentration_mol_l": "concentration_MOL_L",
        "pathlength": "pathlength_cm",
        "path_length": "pathlength_cm",
        "pathlength_cm": "pathlength_cm",
        "path_length_cm": "pathlength_cm",
        "field": "field_B",
        "field_b": "field_B",
        "field_t": "field_B",
        "tesla": "field_B",
    }
    float_keys = {"concentration_MOL_L", "pathlength_cm", "field_B"}
    normalized_run_label = _normalize_metadata_key(run_label or "")

    current_section = None
    first_nonempty_value_line_seen = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if not first_nonempty_value_line_seen:
            first_nonempty_value_line_seen = True
            if ":" not in line and "=" not in line:
                parsed.setdefault("lims_ID", line)
                continue

        if line.endswith(":"):
            current_section = _normalize_metadata_key(line[:-1])
            continue

        if current_section == "concentrations":
            match = re.match(r"([^:=]+)\s*[:=]\s*(.+)", line)
            if match:
                label_key = _normalize_metadata_key(match.group(1))
                if normalized_run_label and label_key == normalized_run_label:
                    concentration = _parse_first_float(match.group(2))
                    if concentration is not None:
                        parsed["concentration_MOL_L"] = concentration
                continue

        if current_section == "field":
            field_value = _parse_first_float(line)
            if field_value is not None:
                parsed["field_B"] = field_value
            continue

        if current_section in {"pathlength", "pathlength_cm", "path_length", "path_length_cm"}:
            pathlength_value = _parse_first_float(line)
            if pathlength_value is not None:
                parsed["pathlength_cm"] = pathlength_value
            continue

        if "=" in line:
            raw_key, raw_value = line.split("=", 1)
        elif ":" in line:
            raw_key, raw_value = line.split(":", 1)
        else:
            continue

        normalized_key = _normalize_metadata_key(raw_key)
        if normalized_run_label:
            if normalized_key == f"concentration_{normalized_run_label}" or normalized_key == f"{normalized_run_label}_concentration":
                numeric_value = _parse_first_float(raw_value)
                if numeric_value is not None:
                    parsed["concentration_MOL_L"] = numeric_value
                continue
        mapped_key = key_aliases.get(normalized_key)
        if mapped_key is None:
            continue

        value_text = raw_value.strip()
        if mapped_key in float_keys:
            numeric_value = _parse_first_float(value_text)
            if numeric_value is not None:
                parsed[mapped_key] = numeric_value
        else:
            parsed[mapped_key] = value_text

    return parsed


def _read_processing_metadata(folder_path: str | None) -> tuple[dict[str, Any], str | None]:
    """
    Locate metadata for a run folder.

    For a compound/run layout, config.txt usually lives one folder above the run
    folder so Q0/Q1/B share the same concentration/pathlength/field metadata. Run
    folder configs are also supported for future exceptions.
    """
    if not folder_path or not os.path.isdir(folder_path):
        return {}, None

    folder_name = os.path.basename(folder_path)
    parent_dir = os.path.dirname(folder_path)
    parent_name = os.path.basename(parent_dir)
    candidate_paths = []

    preferred_names = [
        "config.txt",
        "MTPP_config.txt",
        f"{parent_name}_config.txt",
        "brutefit_config.txt",
        "brutefit_metadata.txt",
        "metadata.txt",
    ]
    candidate_paths.extend(
        os.path.join(parent_dir, name)
        for name in preferred_names
        if os.path.isfile(os.path.join(parent_dir, name))
    )
    candidate_paths.extend(
        os.path.join(folder_path, name)
        for name in preferred_names
        if os.path.isfile(os.path.join(folder_path, name))
    )

    parent_named_file = os.path.join(parent_dir, f"{parent_name}.txt")
    if os.path.isfile(parent_named_file):
        candidate_paths.append(parent_named_file)

    deduped_paths = []
    seen_paths = set()
    for path in candidate_paths:
        if path not in seen_paths:
            deduped_paths.append(path)
            seen_paths.add(path)

    if not deduped_paths:
        return {}, None

    metadata_path = deduped_paths[0]

    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            parsed = _parse_processing_metadata_text(handle.read(), run_label=folder_name)
    except Exception as exc:
        message = (
            f"Found metadata file at {metadata_path}, but it could not be parsed: {exc}. "
            "The selected data files can still be loaded manually."
        )
        if os.environ.get("BRUTEFIT_HEADLESS"):
            print(f"Config File Warning: {message}")
        else:
            QMessageBox.warning(None, "Config File Warning", message)
        return {}, metadata_path

    return parsed, metadata_path


"""
Helper function that reads a csv file and returns a dataframe.

params:
filename (String)- Full path of the csv file
column_names (List): column names of the dataframe in order from left to right of csv file

returns - dataframe of csv file or none if error

Data is sorted by wavelength from low to high before returning
"""
def _read_csv_file(filename: str, column_names: list = None) -> DataFrame | None:
    try:
        if column_names:
            # Instrument exports sometimes include a text header row or extra trailing
            # columns. Force the expected schema, coerce numeric values, then drop only
            # rows that are invalid in required columns. The optional "additional"
            # column is excluded so blank extras do not throw away valid spectra.
            df = pd.read_csv(
                filename,
                header=None,
                names=column_names,
                usecols=range(len(column_names)),
            )
            required_columns = [name for name in column_names if name != "additional"]
            for column in column_names:
                df[column] = pd.to_numeric(df[column], errors="coerce")
            before_rows = len(df)
            df = df.dropna(subset=required_columns)
            dropped_rows = before_rows - len(df)
            if dropped_rows:
                print(f"Dropped {dropped_rows} non-numeric/header row(s) from {filename}")
        else:
            df = pd.read_csv(filename)

        df = df.sort_values(by="wavelength").reset_index(drop=True)

        print(f"Read and sorted file {filename} successfully")
        return df

    except FileNotFoundError:
        print(f"File not found: {filename}")
    except pd.errors.EmptyDataError:
        print(f"Empty file: {filename}")
    except pd.errors.ParserError:
        print(f"Parsing error in file: {filename}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None


def _classify_processing_csv(file_path: str) -> str | None:
    """Classify raw run CSVs by filename role and ignore generated outputs."""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    lower_name = base_name.lower()
    if "processed" in lower_name:
        return None
    if "sticks" in lower_name:
        return "sticks"
    if "pos" in lower_name:
        return "pos"
    if "neg" in lower_name:
        return "neg"
    if "abs" in lower_name:
        return "abs"
    return None


def _base_name_from_processing_files(files: dict[str, str]) -> str | None:
    for key in ("pos", "neg", "abs", "sticks"):
        file_path = files.get(key)
        if not file_path:
            continue
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        return re.sub(r"(_1t)?_(pos|neg|abs|sticks)$", "", base_name, flags=re.IGNORECASE)
    return None


def discover_processing_files_in_folder(folder_path: str) -> tuple[dict[str, str], str | None]:
    """
    Discover one raw pos/neg/abs CSV set in a run folder.

    Generated processed CSVs are ignored so batch runs can operate in folders that
    also contain old outputs.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Run folder does not exist: {folder_path}")

    files: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path) or not filename.lower().endswith(".csv"):
            continue
        match = _classify_processing_csv(file_path)
        if match is None:
            continue
        if match in files:
            duplicates.setdefault(match, [files[match]]).append(file_path)
            continue
        files[match] = file_path

    if duplicates:
        # Batch mode should not guess which duplicate raw file to use. Failing here is
        # safer and more reproducible than silently selecting the first path.
        details = "; ".join(
            f"{key}: {', '.join(os.path.basename(path) for path in paths)}"
            for key, paths in duplicates.items()
        )
        raise ValueError(f"Multiple raw CSV files found for the same role in {folder_path}: {details}")

    return files, _base_name_from_processing_files(files)


def read_pos_neg_abs_from_folder(folder_path: str) -> tuple[Any, Any, Any, Any, str | None, dict[str, Any], str | None]:
    """
    Read a raw run folder without opening file-selection dialogs.

    This is the batch-friendly counterpart to read_pos_neg_abs(). It returns metadata
    along with dataframes so batch processing can avoid GUI prompts entirely.
    """
    files, base_name = discover_processing_files_in_folder(folder_path)
    missing = [name for name in ("pos", "neg", "abs") if name not in files]
    if missing:
        raise ValueError(
            f"Run folder {folder_path} is missing required raw file(s): {', '.join(missing)}"
        )

    metadata_values, metadata_path = _read_processing_metadata(folder_path)

    column_names_pos = ["wavelength", "x_pos", "y_pos", "R", "theta", "std_dev_x", "std_dev_y", "additional"]
    column_names_neg = ["wavelength", "x_neg", "y_neg", "R", "theta", "std_dev_x", "std_dev_y", "additional"]
    column_names_abs = ["wavelength", "intensity"]
    column_names_sticks = ["wavelength", "strength"]

    positive_df = _read_csv_file(files["pos"], column_names_pos)
    negative_df = _read_csv_file(files["neg"], column_names_neg)
    abs_df = _read_csv_file(files["abs"], column_names_abs)

    sticks_df = None
    if "sticks" in files:
        sticks_df = _read_csv_file(files["sticks"], column_names_sticks)

    return positive_df, negative_df, abs_df, sticks_df, base_name, metadata_values, metadata_path


"""
Helper function that uses pyside6 to open a file selection dialog to select files for processing.
params:
None

returns -  Tuple(Dictionary of full filepaths to be processed, base name for output file)

This function expects the files to end with _pos, _neg, _abs, and optionally sticks. _pos,_neg, and _abs files must be present.
"""
def _select_files_processing() -> tuple[dict[str, str], str | None, dict[str, Any], str | None]:
    file_paths,_ = QFileDialog.getOpenFileNames()

    files: dict[str, str] = {}
    keywords = ["pos", "neg", "abs", "sticks"]

    base_name = None

    if not file_paths:
        return files, None, {}, None

    for file_path in file_paths:
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        match = next((kw for kw in keywords if re.search(kw, base_name, re.IGNORECASE)), None)

        if match:
            files[match] = file_path
        else:
            QMessageBox.critical(
                None,
                "File Naming Error",
                f"File {base_name} does not contain 'pos', 'neg', 'abs', or 'sticks'. "
                "Please rename the file accordingly."
            )

    if base_name is not None:
        base_name = re.sub(r"_(pos|neg|abs|sticks)$", "", base_name, flags=re.IGNORECASE)

    parent_dirs = {os.path.dirname(path) for path in file_paths}
    metadata_values = {}
    metadata_path = None
    if len(parent_dirs) == 1:
        # GUI processing still asks the user to review/edit these values later, but
        # preloading config metadata saves repeated manual entry across many runs.
        folder_path = next(iter(parent_dirs))
        metadata_values, metadata_path = _read_processing_metadata(folder_path)

    return files, base_name, metadata_values, metadata_path


"""
Top level function that calls the previous two functions to read the csv files needed for processing and output a dataframe.

Params:
None

returns -  Tuple(positive dataframe, negative dataframe, absorption dataframe, sticks dataframe, base name for output file)

pos and neg are mcd response data.
Abs is absorption. Sticks is optional.
"""
def read_pos_neg_abs() -> tuple[Any, Any, Any, Any, str | None, dict[str, Any], str | None]:
    files, base_name, metadata_values, metadata_path = _select_files_processing()
    if base_name is None:
        return None, None, None, None, None, metadata_values, metadata_path

    missing = [name for name in ("pos", "neg", "abs") if name not in files]
    if missing:
        QMessageBox.critical(
            None,
            "Missing Required Files",
            "Please select files containing 'pos', 'neg', and 'abs' in their filenames."
        )
        return None, None, None, None, None, metadata_values, metadata_path

    column_names_pos = ["wavelength", "x_pos", "y_pos", "R", "theta", "std_dev_x", "std_dev_y", "additional"]
    column_names_neg = ["wavelength", "x_neg", "y_neg", "R", "theta", "std_dev_x", "std_dev_y", "additional"]
    column_names_abs = ["wavelength", "intensity"]
    column_names_sticks = ["wavelength", "strength"]

    positive_df = _read_csv_file(files["pos"], column_names_pos)
    negative_df = _read_csv_file(files["neg"], column_names_neg)
    abs_df = _read_csv_file(files["abs"], column_names_abs)

    sticks_df = None
    if "sticks" in files:
        sticks_df = _read_csv_file(files["sticks"], column_names_sticks)

    return positive_df, negative_df, abs_df, sticks_df, base_name, metadata_values, metadata_path
