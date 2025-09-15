import numpy as np
import multiprocessing as mp

import bruteFit.utils as utils
from bruteFit.fitConfig import FitConfig
from bruteFit.dataFitting import fit_models


def main():
    # Can use it like this to set defaults and pass to fit_models, or just use pre-set default
    fc = FitConfig()
    fc.MIN_GC = 1  # inclusive
    fc.MAX_GC = 6  # exclusive

    fc.PATHLENGTH_CM = 1
    fc.FIELD_B = 1
    fc.CONCENTRATION_MOL_L = 1

    """
    When window opens you can left click to remove peaks - you cannot add as I thought that was a bad design choice
    because we want to keep the data as reproducible as possible.
    """

    mcd_df = utils.open_csv_with_tkinter()
    # results = fit_models(mcd_df)  # using pre-set defaults - 4 processes by default
    fit_models(mcd_df, fc, processes=mp.cpu_count())

#when multiprocessing on windows and generally - need name guard
if __name__ == "__main__":
    main()

