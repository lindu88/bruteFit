import os

import numpy as np
from matplotlib import pyplot as plt

from . import helpers as hc
import pandas as pd


class ProcessRecord:
    def __init__(self, input_tuple, lims, conc, pl, field):
        mcd_pos, mcd_neg, abs, sticks, name = input_tuple
        self.mcd_pos = mcd_pos
        self.mcd_neg = mcd_neg
        self.abs = abs
        self.sticks = sticks
        self.lims_ID = lims
        self.name = name
        self.concentration_MOL_L = conc
        self.pathlength_cm = pl
        self.field_B = field


        #pandas dataframe
        self.mergedOut = None

        self.dataInp = ProcessRecord.generate_empty_inp_dataset()
        self.dataOut = ProcessRecord.generate_empty_out_dataset()
        self.fields = ProcessRecord.generate_empty_fields_dataset()

        #prep
        self._prep()

        #fill input with what we have
        self._fill_with_input()

        #fill_output
        self._fill_outputs()

        self._merge_dfs()
        # test
        print(self.mergedOut)


    def _fill_with_input(self):
        #UVVIS and MCD wavelength - they share the same set but are defined on different values
        self.dataInp["wavelength_nm_inp"].extend(self.abs["wavelength"].values)


        #intinsity in absorb units
        self.dataInp["uvvisabsorptivity_abs_absorptivityarbunits_inp"].extend(self.abs["intensity"].values)

        #fill the fields section
        self._fill_fields()

        #mcd x values
        self.dataInp["xcartesian_mcdpos_deltaabsorptivityarbunits_inp"].extend(self.mcd_pos["x_pos"].values)
        self.dataInp["xcartesian_mcdneg_deltaabsorptivityarbunits_inp"].extend(self.mcd_neg["x_neg"].values)

        #mcd y values
        self.dataInp["ycartesian_mcdpos_deltaabsorptivityarbunits_inp"].extend(self.mcd_pos["y_pos"].values)
        self.dataInp["ycartesian_mcdneg_deltaabsorptivityarbunits_inp"].extend(self.mcd_neg["y_neg"].values)

        #mcd R values
        self.dataInp["rabsolute_mcdpos_deltaabsorptivityarbunits_inp"].extend(self.mcd_pos["R"].values)
        self.dataInp["rabsolute_mcdneg_deltaabsorptivityarbunits_inp"].extend(self.mcd_neg["R"].values)

        #mcd theta
        self.dataInp["theta_mcdpos_radians_inp"].extend(self.mcd_pos["theta"].values)
        self.dataInp["theta_mcdneg_radians_inp"].extend(self.mcd_pos["theta"].values)

        #std dev
        self.dataInp["stddevxcartesian_mcdpos_deltaabsorptivity_inp"].extend(self.mcd_pos["std_dev_x"].values)
        self.dataInp["stddevycartesian_mcdpos_deltaabsorptivity_inp"].extend(self.mcd_pos["std_dev_y"].values)
        self.dataInp["stddevxcartesian_mcdneg_deltaabsorptivity_inp"].extend(self.mcd_pos["std_dev_x"].values)
        self.dataInp["stddevycartesian_mcdneg_deltaabsorptivity_inp"].extend(self.mcd_pos["std_dev_y"].values)



    def _fill_outputs(self):
        # Wavenumber (output)
        wavenumber_cm1_out = [1e7 / wl for wl in self.dataInp["wavelength_nm_inp"]]
        self.dataOut["wavenumber_out"].extend(wavenumber_cm1_out)

        # UVVIS (output)
        uvvis_intinsity_extinction = hc.convert_abs_to_extinction(self.abs["intensity"], self.concentration_MOL_L, self.pathlength_cm)
        self.dataOut["uvvis_extinction_abs_molar-1cm-1_out"].extend(uvvis_intinsity_extinction)

        #sticks UVVIS -- fill with zeros id none
        if self.sticks is None:
            all_zeros = np.zeros(self._get_max_dfs_length())
            self.dataOut["sticks_out"].extend(all_zeros)
        else:
            self.dataOut["sticks_out"].extend(self.sticks["strength"].fillna(0).values)

        # MCD (output)
        _, _, _, _, R_signed, R_stdev = hc.calculate_differences(self.mcd_pos, self.mcd_neg)
        self.dataOut["deltaabsorptivity_mcdavg_absorbunits_out"].extend(R_signed)
        self.dataOut["deltaextinction_mcdavg_molar-1cm-1_out"].extend(hc.convert_abs_to_extinction(R_signed, self.concentration_MOL_L, self.pathlength_cm))
        self.dataOut["absorptivity_stddev_mcdavg_absorbunits_out"].extend(R_stdev)
        self.dataOut["extinction_stddev_mcdavg_molar-1cm-1_out"].extend(hc.convert_abs_to_extinction(R_stdev, self.concentration_MOL_L, self.pathlength_cm))

        # MCD (output) field scaled
        self.dataOut["deltaabsorptivitypertesla_mcdavg_abssorptivityarbunitsT-1_out"].extend([i / self.field_B for i in R_signed])
        self.dataOut["deltaextinctionpertesla_mcdavg_molar-1cm-1T-1_out"].extend(i / self.field_B for i in self.dataOut["deltaextinction_mcdavg_molar-1cm-1_out"])

        #mord
        #TODO: add mord


    """
    Removes rows with NaNs or Infs
    """
    def clean_data(self):
        df_clean = self.mergedOut.replace([np.inf, -np.inf], np.nan).dropna()
        df_clean = df_clean.reset_index(drop=True)
        self.mergedOut = df_clean

    def save(self, path: str = ""):
        filename = f"{self.name}_processed.csv"

        # Determine base directory
        if path and os.path.isdir(path):
            base_path = os.path.join(path, filename)
            print(f"Saving {filename} to path: {os.path.abspath(path)}")
        else:
            print("Path invalid or not given. Saving in current working directory.")
            base_path = os.path.join(os.getcwd(), filename)

        # Append _1, _2, etc. if file exists
        save_path = base_path
        root, ext = os.path.splitext(base_path)
        counter = 1
        while os.path.exists(save_path):
            save_path = f"{root}_{counter}{ext}"
            counter += 1

        # Save file
        self.mergedOut.to_csv(save_path, index=False, mode='x')
        print(f"Saved: {save_path}")


    def _fill_fields(self):
        for name, field_list in self.fields.items():
            for i in range(0, self._get_max_dfs_length()):
                field_list.append(self.__getattribute__(name))

    def _get_max_dfs_length(self) -> int:
        max_len_inp = max((len(v) for v in self.dataInp.values()), default=0)
        max_len_out = max((len(v) for v in self.dataOut.values()), default=0)
        max_len_fields = max((len(v) for v in self.fields.values()), default=0)

        max_len = max(max_len_inp, max_len_out, max_len_fields)
        return max_len

    #TODO: Check again, used AI -- works
    def _prep(self, method: str = "nan"):
        """
        Align (pad) abs, mcd_pos, mcd_neg, sticks on the union of their wavelength grids.
        After alignment, 'wavelength' is a COLUMN again for each DF.
        """
        # Gather only present dataframes
        dfs = {
            "abs": self.abs,
            "mcd_pos": self.mcd_pos,
            "mcd_neg": self.mcd_neg,
            "sticks": self.sticks,
        }

        present = {}
        for i, df in dfs.items():
            if df is not None:
                present[i] = df

        # Ensure each DF is indexed by 'wavelength'
        for i, df in present.items():
            if "wavelength" in df.columns:
                df = df.set_index("wavelength")
            else:
                print("Cant index by wavelength in _prep")
            present[i] = df

        ############################################################
        #construct the master wavlenght list by compiling all the wavelengths from the dfs
        # Build union wavelength grid
        union = pd.Index([])
        for df in present.values():
            union = union.union(df.index)
        union = union.sort_values()
        ############################################################

        # Reindex and (optionally) interpolate numeric columns
        for k, df in present.items():
            df = df.reindex(union)
            # Put 'wavelength' back as a column
            present[k] = df.reset_index().rename(columns={"index": "wavelength"})

        # Assign back
        self.abs = present.get("abs", self.abs)
        self.mcd_pos = present.get("mcd_pos", self.mcd_pos)
        self.mcd_neg = present.get("mcd_neg", self.mcd_neg)
        self.sticks = present.get("sticks", self.sticks)

    def get_merged_df(self):
        return self.mergedOut.copy()
    def set_merged_df(self, df):
        self.mergedOut = df
    """
    Merges and pads the dfs, sticks get padded to zero
    """
    def _merge_dfs(self):
        # --- debug display
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)
        # ----------------------------------

        inp_cols = list(self.dataInp.keys())
        out_cols = list(self.dataOut.keys())
        fields_cols = list(self.fields.keys())

        inp = pd.DataFrame(self.dataInp)[inp_cols].reset_index(drop=True)
        out = pd.DataFrame(self.dataOut)[out_cols].reset_index(drop=True)
        fields = pd.DataFrame(self.fields)[fields_cols].reset_index(drop=True)

        # Concatenate in the given order
        self.mergedOut = pd.concat([inp, out, fields], axis=1)

    #TODO: got lazy and used AI check later
    def plot_extinction(self, return_fig=False):
        # --- Absorption arrays ---
        x_abs = self.mergedOut["wavenumber_out"]
        y_abs = self.mergedOut["uvvis_extinction_abs_molar-1cm-1_out"].interpolate(method="linear", limit_direction="both")

        # --- Matched sticks, scaled so max = max(y_abs) ---
        sticks = list(self.mergedOut["sticks_out"])
        sticks_scaled = [s / max(sticks) * max(y_abs) if max(sticks) > 0 else 0 for s in sticks]

        # --- MCD arrays ---
        x_mcd = self.mergedOut["wavenumber_out"]
        y_mcd = self.mergedOut["deltaextinction_mcdavg_molar-1cm-1_out"].interpolate(method="linear", limit_direction="both")
        yerr_mcd = self.mergedOut["extinction_stddev_mcdavg_molar-1cm-1_out"]

        # --- Figure ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        # Absorption (smooth line)
        ax1.plot(x_abs, y_abs, '-', label="Absorption", alpha=0.9)
        # Sticks (index-aligned, scaled to max Abs)
        ax1.vlines(x_abs, 0, sticks_scaled,color="red", linewidth=1.2, alpha=0.7, label="Sticks (scaled)")

        ax1.set_ylabel("Abs intensity extinction units")
        ax1.set_title("Absorption Spectrum")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()

        # MCD (smooth line + error bars)
        ax2.plot(x_mcd, y_mcd, '-', label="MCD", alpha=0.9)
        ax2.errorbar(x_mcd, y_mcd, yerr=yerr_mcd, fmt='none', elinewidth=1, capsize=3, alpha=0.6)

        ax2.set_xlabel("Wavenumber (cm$^{-1}$)")
        ax2.set_ylabel("MCD intensity extinction units")
        ax2.set_title("MCD Spectrum")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()

        plt.tight_layout()
        if return_fig:
            return fig, (ax1, ax2)
        plt.show()
        return None

    @staticmethod
    def generate_empty_inp_dataset():
        data = {
            # Wavelength
            "wavelength_nm_inp": [],

            # UVVIS
            "uvvisabsorptivity_abs_absorptivityarbunits_inp": [],

            # X values
            "xcartesian_mcdpos_deltaabsorptivityarbunits_inp": [],
            "xcartesian_mcdneg_deltaabsorptivityarbunits_inp": [],
            # Note: "pos" = co-propagating with light; "neg" = counter-propagating.

            # Y values
            "ycartesian_mcdpos_deltaabsorptivityarbunits_inp": [],
            "ycartesian_mcdneg_deltaabsorptivityarbunits_inp": [],

            # R
            "rabsolute_mcdpos_deltaabsorptivityarbunits_inp": [],
            "rabsolute_mcdneg_deltaabsorptivityarbunits_inp": [],

            # Theta
            "theta_mcdpos_radians_inp": [],
            "theta_mcdneg_radians_inp": [],
            # Check units for theta.

            # Standard deviations
            "stddevxcartesian_mcdpos_deltaabsorptivity_inp": [],
            "stddevycartesian_mcdpos_deltaabsorptivity_inp": [],
            "stddevxcartesian_mcdneg_deltaabsorptivity_inp": [],
            "stddevycartesian_mcdneg_deltaabsorptivity_inp": []
        }
        return data

    @staticmethod
    def generate_empty_out_dataset():
        data = {

            # Wavenumber
            "wavenumber_out": [],

            # UVVIS
            "uvvis_extinction_abs_molar-1cm-1_out": [],
            "sticks_out": [],
            #"uvvis_absorptivity_stddev_abs_arbunits_out": [],
            #"uvvis_extinctions_stddev_abs_molar-1cm-1_out": [],  # per-nm std deviations

            # MCD
            "deltaabsorptivity_mcdavg_absorbunits_out": [], #R_signed
            "deltaextinction_mcdavg_molar-1cm-1_out": [],
            "absorptivity_stddev_mcdavg_absorbunits_out": [],
            "extinction_stddev_mcdavg_molar-1cm-1_out": [],

            #MCD scaled by field
            "deltaabsorptivitypertesla_mcdavg_abssorptivityarbunitsT-1_out": [],
            "deltaextinctionpertesla_mcdavg_molar-1cm-1T-1_out": []

            #mord
            #TODO: add mord here
        }
        return data

    @staticmethod
    def generate_empty_fields_dataset():
        data = {
            "lims_ID": [],
            "name": [],
            "concentration_MOL_L": [],
            "pathlength_cm": [],
            "field_B": []
        }
        return data



