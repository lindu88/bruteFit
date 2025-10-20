import numpy as np
import multiprocessing as mp

import bruteFit.utils as utils
from bruteFit.fitConfig import FitConfig
from bruteFit.dataFitting import fit_models


def main():
    # Can use it like this to set defaults and pass to fit_models, or just use pre-set default
    fc = FitConfig()
    # NOTE we should probably explain this a bit more clearly. here I think I understand that this is showing how to pass param values to fitconfig.
     
    fc.MIN_GC = 1  # inclusive
    fc.MAX_GC = 6  # exclusive

    """
    When window opens you can left click to remove peaks - you cannot add as I thought that was a bad design choice
    because we want to keep the data as reproducible as possible.
    """
    # Sam NOTE - when we update params (update button) then the removed peaks get put back on screen. 

    mcd_df = utils.launch_proc_viewer()
    # results = fit_models(mcd_df)  # using pre-set defaults - 4 processes by default
    fit_models(mcd_df, fc, processes=mp.cpu_count())
    #fit_models(mcd_df, fc, processes=1)
    # having a single process fixes segfault on mac

#when multiprocessing on windows and generally - need name guard
if __name__ == "__main__":
    main()


# TODO (SAM) fix some plotting units
# TODO load in saved data and replot it. 
# TODO add a plot absorptivity method. Add way to change extinction/abs plots to units nm not wavenumbers.
# TODO add option to turn on/off gridlines
# TODO toggle sticks
# TODO documentation documentation documentation (wes)
# TODO fix why amplitudes are negative 
# TODO implement sig fig rounding 
# TODO change B in A/B to D
# TODO print name of file (and strip _abs etc) in the window somewhere so you don't get confused processing a bunch of em
# TODO make preclean/postclean zoomable