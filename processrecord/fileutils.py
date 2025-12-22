from typing import Any

import pandas as pd
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox

from PySide6.QtWidgets import QFileDialog
from pandas import DataFrame


def _read_csv_file(filename: str, column_names: list = None) -> DataFrame | None:
    try:
        if column_names:
            df = pd.read_csv(filename, names=column_names)
        else:
            df = pd.read_csv(filename)

        df = df.sort_values(by="wavelength").reset_index(drop=True)
        # reset_index(drop=True) resets the row index to 0..n-1 and discards the old one.

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

def _select_files_processing() -> tuple[dict[str, str], str | None]:
    """Return a dict like {'pos': path, 'neg': path, 'abs': path, 'sticks': path?}
    based on filenames selected by the user.
    """


    file_paths,_ = QFileDialog.getOpenFileNames()

    files: dict[str, str] = {}
    keywords = ["pos", "neg", "abs", "sticks"]

    #sets base name to the base name of the last file in file_paths
    base_name = None

    for file_path in file_paths:
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # find the keyword in the filename (case-insensitive)
        match = next((kw for kw in keywords if re.search(kw, base_name, re.IGNORECASE)), None)

        if match:
            files[match] = file_path
        else:
            messagebox.showerror(
                "File Naming Error",
                f"File {base_name} does not contain 'pos', 'neg', 'abs', or 'sticks'. "
                "Please rename the file accordingly."
            )

    base_name = base_name[:base_name.rfind("_")]  # drop _pos/_abs.ect

    return files, base_name
"""
returns tuple of pos_df, neg_df, abs df, sticks_df, name
"""
def read_pos_neg_abs() -> tuple[Any, Any, Any, Any, str | None]:
    files, base_name = _select_files_processing()
    column_names_pos = ["wavelength", "x_pos", "y_pos", "R", "theta", "std_dev_x", "std_dev_y", "additional"]
    column_names_neg = ["wavelength", "x_neg", "y_neg", "R", "theta", "std_dev_x", "std_dev_y", "additional"]
    column_names_abs = ["wavelength", "intensity"]
    column_names_sticks = ["wavelength", "strength"]

    positive_df = _read_csv_file(files["pos"], column_names_pos)
    negative_df = _read_csv_file(files["neg"], column_names_neg)
    abs_df      = _read_csv_file(files["abs"], column_names_abs)

    sticks_df = None
    if "sticks" in files:
        sticks_df = _read_csv_file(files["sticks"], column_names_sticks)

    return positive_df, negative_df, abs_df, sticks_df, base_name
