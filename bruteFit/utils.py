import tkinter as tk
from tkinter import filedialog
import pandas as pd

def open_csv_with_tkinter():
    # Hide the root window
    root = tk.Tk()
    root.withdraw()

    # Open file dialog
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])

    if file_path:
        df = pd.read_csv(file_path)
        df = df.sort_values(by="wavelength").reset_index(drop=True)
        print("DataFrame Loaded:")
        print(df.head())  # Print first 5 rows
        return df
    else:
        print("No file selected.")
        return None