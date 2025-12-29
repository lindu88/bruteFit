import numpy as np
from lmfit import Model

TINY = np.finfo(float).tiny
SQRT_LOG2 = np.sqrt(np.log(2))
S2PI = np.sqrt(2 * np.pi)

#TODO: Check because I fed though AI to clean up. I was lazy
# TODO cite/show derivation and normalization.
def stable_gaussian_sigma(x, amplitude, center, sigma):
    """Numerically stable Gaussian (normalized to area=1), using sigma."""
    gamma = sigma * 2 * np.sqrt(2 * np.log(2))  # FWHM
    denom = max(TINY, gamma * np.sqrt(np.pi))
    exponent = -4.0 * max(TINY, np.log(2) / gamma**2) * (x - center) ** 2
    return amplitude * (2 * SQRT_LOG2 / denom) * np.exp(exponent)


def stable_gaussian_derivative_sigma(x, amplitude, center, sigma):
    """First derivative of a normalized Gaussian (w.r.t. x), using sigma."""
    gamma = sigma * 2 * np.sqrt(2 * np.log(2))
    denom = max(TINY, gamma**3 * np.sqrt(np.pi))
    exponent = -4.0 * max(TINY, np.log(2) / gamma**2) * (x - center) ** 2
    return (-16.0 * np.log(2) * SQRT_LOG2 * amplitude * (x - center) / denom) * np.exp(exponent)
#####################################################################################################

# --- lmfit Models ---
model_stable_gaussian_sigma        = Model(stable_gaussian_sigma)
model_stable_gaussian_deriv_sigma  = Model(stable_gaussian_derivative_sigma)

