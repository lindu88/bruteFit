import numpy as np
from lmfit import Model

TINY = np.finfo(float).tiny
SQRT_LOG2 = np.sqrt(np.log(2))
S2PI = np.sqrt(2 * np.pi)

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


def gaussian(x, center, fwhm, amplitude):
    """Normalized Gaussian using FWHM."""
    denom = max(TINY, fwhm * np.sqrt(np.pi))
    exponent = -4 * np.log(2) * ((x - center) / fwhm) ** 2
    return amplitude * (2 * SQRT_LOG2 / denom) * np.exp(exponent)


def stable_gaussian(x, center, fwhm, amplitude):
    """Numerically stable version of normalized Gaussian using FWHM."""
    denom = max(TINY, fwhm * np.sqrt(np.pi))
    exponent = -4.0 * max(TINY, np.log(2) / fwhm**2) * (x - center) ** 2
    return amplitude * (2 * SQRT_LOG2 / denom) * np.exp(exponent)


def gaussian_derivative(x, center, fwhm, amplitude):
    """First derivative of normalized Gaussian using FWHM."""
    exponent = -4 * np.log(2) * ((x - center) / fwhm) ** 2
    return (-4 * np.log(2) * amplitude * (x - center) / fwhm**2) * np.exp(exponent)


def custom_gaussian(x, amplitude=1.0, center=0.0, sigma=1.0):
    """Standard Gaussian using sigma (normalized to area=1)."""
    denom = max(TINY, S2PI * sigma)
    exponent = -((x - center) ** 2) / max(TINY, 2 * sigma**2)
    return amplitude / denom * np.exp(exponent)


# Legacy / unnormalized forms (for reference)
def gaussian_old(x, amplitude, center, width):
    return amplitude * np.exp(-(x - center) ** 2 / (2 * width ** 2))


def custom_gaussian_old(x, amplitude, center, sigma):
    return (amplitude / (sigma * S2PI)) * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


#####################################################################################################

# --- lmfit Models ---

model_stable_gaussian_sigma        = Model(stable_gaussian_sigma)
model_stable_gaussian_deriv_sigma  = Model(stable_gaussian_derivative_sigma)
model_gaussian                     = Model(gaussian)
model_stable_gaussian              = Model(stable_gaussian)
model_gaussian_derivative          = Model(gaussian_derivative)
model_custom_gaussian              = Model(custom_gaussian)
model_gaussian_old                 = Model(gaussian_old)
model_custom_gaussian_old          = Model(custom_gaussian_old)

