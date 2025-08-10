import numpy as np

import bruteFit.utils as utils
import bruteFit.dataFitting
from bruteFit.dataFitting import fit_models

#min and max gaussian/lorentzian count inclusive min exclusive max
min_gc = 3
max_gc = 4

mcd_df = utils.open_csv_with_tkinter()
results = fit_models(mcd_df, min_gc=min_gc, max_gc=max_gc)
results.plot(10, window = True, metric='redchi')