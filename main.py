import numpy as np

import bruteFit.utils as utils
from bruteFit.fitConfig import FitConfig
from bruteFit.dataFitting import fit_models

fc = FitConfig()
fc.MIN_GC = 3 # changed to exclusive
fc.MAX_GC = 6 # changed to inclusive
fc.print()

mcd_df = utils.open_csv_with_tkinter()
results = fit_models(mcd_df, fc)
results.plot(10, metric='residual_rms')