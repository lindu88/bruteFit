import numpy as np

import bruteFit.utils as utils
import bruteFit.dataFitting
from bruteFit.dataFitting import fit_models

mcd_df = utils.open_csv_with_tkinter()
results = fit_models(mcd_df)
results.plot(50, metric='residual_rms')