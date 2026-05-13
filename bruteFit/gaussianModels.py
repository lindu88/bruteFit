import numpy as np
from lmfit import Model

TINY = np.finfo(float).tiny
SQRT_LOG2 = np.sqrt(np.log(2))
S2PI = np.sqrt(2 * np.pi)

# Collaborator note on units:
# The model parameter named "amplitude" is an integrated area-like amplitude for the
# normalized Gaussian functions below, not the visual peak height. GUI/manual input is
# allowed to use peak height because that is easier to eyeball; conversion helpers at
# the bottom of this file translate between the two representations.
# TODO: cite/show derivation and normalization in user-facing documentation.
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


def gaussian_height_from_area_amplitude(amplitude, sigma):
    sigma = max(abs(float(sigma)), TINY)
    return float(amplitude) / (sigma * S2PI)


def gaussian_area_amplitude_from_height(height, sigma):
    sigma = max(abs(float(sigma)), TINY)
    return float(height) * sigma * S2PI


def _derivative_peak_scale(center, sigma):
    # For derivative Gaussians, visual "height" means max absolute lobe height. The
    # closed form is easy to get wrong, so we compute the unit-amplitude scale on a
    # dense local grid and reuse that for height/amplitude conversion.
    sigma = max(abs(float(sigma)), TINY)
    sample_x = np.linspace(float(center) - 5.0 * sigma, float(center) + 5.0 * sigma, 2049)
    unit_curve = stable_gaussian_derivative_sigma(sample_x, 1.0, float(center), sigma)
    return max(float(np.nanmax(np.abs(unit_curve))), TINY)


def derivative_height_from_amplitude(amplitude, center, sigma):
    return float(amplitude) * _derivative_peak_scale(center, sigma)


def derivative_amplitude_from_height(height, center, sigma):
    return float(height) / _derivative_peak_scale(center, sigma)


def component_peak_height(amplitude, center, sigma, label=None):
    if label == "A":
        return derivative_height_from_amplitude(amplitude, center, sigma)
    return gaussian_height_from_area_amplitude(amplitude, sigma)


def component_amplitude_from_peak_height(height, center, sigma, label=None):
    if label == "A":
        return derivative_amplitude_from_height(height, center, sigma)
    return gaussian_area_amplitude_from_height(height, sigma)
#####################################################################################################

# --- lmfit Models ---
model_stable_gaussian_sigma        = Model(stable_gaussian_sigma)
model_stable_gaussian_deriv_sigma  = Model(stable_gaussian_derivative_sigma)
