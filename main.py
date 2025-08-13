import numpy as np

import bruteFit.utils as utils
from bruteFit.fitConfig import FitConfig
from bruteFit.dataFitting import fit_models

#Can use it like this to set defaults and pass to fit models, or just use pre-set default
fc = FitConfig()
fc.MIN_GC = 1 #inclsuve
fc.MAX_GC = 6 #exclsuive

"""

When window opens you can left click to remove peaks - you cannot add as I thought that was a bad design choice because we want 
to keep the data as reproducible as possible.

"""

mcd_df = utils.open_csv_with_tkinter()
#results = fit_models(mcd_df) #using pre-set defaults
results = fit_models(mcd_df, fc)