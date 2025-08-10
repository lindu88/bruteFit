import sys
from PySide6.QtWidgets import QApplication
from scipy.signal import find_peaks, savgol_filter, peak_prominences
from . import gaussianModels
from . import plotwindow
from math import comb
import itertools
from lmfit.model import ModelResult, Model
import matplotlib.pyplot as plt
import numpy as np

from .plotwindow import MatplotlibGalleryWindow, show_peak_centers_window

#TODO: Docs and use pyside6 for parameters / records class

#Smoothing
WINDOW_LENGTH = 5  # Window length for Savitzky-Golay smoothing (datapoints?) (relate to bandwidth?)
POLYORDER = 4  # Polynomial order for Savitzky-Golay smoothing

#peak picking
HEIGHT_THRESHOLD = 0.04  # Minimum height threshold for peak detection - should be greater than noise after smoothing. Trouble is that the negative side bands likely mean zero isn't at zero.
PROMINENCE_PERECENT = 0.04  # Prominence is here as a multiple of max height. What is (topographic) prominence? It is "the minimum height necessary to descend to get from the summit to any higher terrain", as it can be seen here
DISTANCE = 5  # The minimum distance, in number of samples, between peaks. - This should be related to bandwidth for certain

# Fitting
MERGE_DX = 500 #distance to merge centers between abs and mcd
MAX_BASIS_GAUSSIANS = 5  # Maximum number of basis Gaussians for fitting
PERCENTAGE_RANGE = 10  # The percentage by which the initial parameters will be allowed to relax on re-fitting after removing poor curves.
MAX_SIGMA = 60000  #max sigma for gaussians
MIN_PEAK_X_DISTANCE = 0
ESTIMATE_SIGMA_ITERATIONS_END = 10  #START/END to END-1/END
ESTIMATE_SIGMA_ITERATIONS_START = 4
MIN_ABSOLUTE_PEAK_HEIGHT = 2.0e-15
MIN_PROMINENCE = 0.000000000000000001  #min relative peak height
AMPLITUDE_SCALE_LIMIT = 3.0

#basic
SMALL_FWHM_FACTOR = 2.355  # Conversion factor from FWHM to sigma

#TODO: add BIC back and maybe other metrics
class BfResult:
    def __init__(self, datax, datay, dataz, redchi_threshold=None, residual_rms_threshold=None):
        """
        Initialize the bfResult container with optional filter thresholds.

        Parameters:
            redchi_threshold (float): Max allowed reduced chi-square value. None = no filter.
            residual_rms_threshold (float): Max allowed residual RMS. None = no filter.
        """
        self.dataX = datax
        self.dataY = datay
        self.dataZ = dataz
        self.results = []  # List of lmfit.ModelResult objects
        self.redchi_threshold = redchi_threshold
        self.residual_rms_threshold = residual_rms_threshold

    def add_result(self, result: ModelResult):
        """Add a new lmfit ModelResult to the list."""
        if isinstance(result, ModelResult):
            self.results.append(result)
        else:
            raise TypeError("Only lmfit.ModelResult instances can be added.")

    def _residual_rms(self, result: ModelResult):
        """Compute the RMS of residuals for a fit."""
        return np.sqrt(np.mean(result.residual ** 2))

    def _filter_results(self):
        """
        Filter results based on thresholds set at initialization.
        Returns a list of ModelResult objects that pass all thresholds.
        """
        filtered = self.results

        if self.redchi_threshold is not None:
            filtered = [r for r in filtered if r.redchi < self.redchi_threshold]

        if self.residual_rms_threshold is not None:
            filtered = [r for r in filtered if self._residual_rms(r) < self.residual_rms_threshold]

        return filtered

    def best_result(self, metric='redchi'):
        """
        Return the single best result based on a metric.
        Options: 'redchi', 'residual_rms'
        """
        filtered = self._filter_results()
        if not filtered:
            return None

        if metric == 'residual_rms':
            return min(filtered, key=self._residual_rms)
        else:
            return min(filtered, key=lambda r: getattr(r, metric))

    #TODO: add optional gaussian count g_n parameter
    def n_best_results(self, n=3, metric='redchi'):
        """
        Return the top N best results based on a metric.
        Options: 'redchi', 'residual_rms'
        """
        filtered = self._filter_results()
        if metric == 'residual_rms':
            return sorted(filtered, key=self._residual_rms)[:n]
        else:
            return sorted(filtered, key=lambda r: getattr(r, metric))[:n]

    # TODO: add optional gaussian count g_n parameter
    def plot(self, n=3, metric='redchi'):
        """
        Plot the top N best fits over the data.
        """
        fig_list = []
        top = self.n_best_results(n=n, metric=metric)
        if not top:
            print("No results to plot.")
            return

        for i, r in enumerate(top, 1):
            fig = self._plot_components_visible(r, self.dataX, self.dataZ)
            fig_list.append(fig)

        #TODO: Change to pass self into window for running operations on results with GUI
        app = QApplication.instance() or QApplication(sys.argv)
        w = MatplotlibGalleryWindow(fig_list)
        w.show()
        app.exec()  # blocks until window is closed
    def _plot_components_visible(self, result, x, z):
        """Plot the original z-data and each individual model component. """
        fig, ax = plt.subplots()

        # Plot the measured data
        ax.plot(x, z, '-', lw=1, alpha=0.7, label='Measured Data')

        # Evaluate each component from the fit
        components = result.eval_components(x=x)

        # Plot each component as-is
        for name, comp in components.items():
            ax.plot(x, comp, lw=2, label=name)

        # Axis labels
        ax.set_xlabel("X")
        ax.set_ylabel("Z")

        # Compute and display residual RMS in the top-left
        residual_rms = self._residual_rms(result)
        ax.text(0.02, 0.95, f"Residual RMS: {residual_rms:.3g}",transform=ax.transAxes, fontsize=12, va='top', ha='left')

        ax.legend()
        plt.tight_layout()

        return fig
def get_anymax_factor(ratio):
    if (ratio >= 1):  #return FWHM if ratio is invalid
        print("full width any max has invalid ratio")
        return SMALL_FWHM_FACTOR
    else:
        return np.sqrt(8 * np.log(1 / ratio))

def estimate_sigma(x, y, peak_index, ratio):
    some_max = y[peak_index] * ratio
    left_candidates = np.where(y[:peak_index] < some_max)[0]
    if len(left_candidates) == 0:
        left_idx = 0  # If no valid left index, use the start of the array
    else:
        left_idx = left_candidates[-1]

    right_candidates = np.where(y[peak_index:] < some_max)[0]
    if len(right_candidates) == 0:
        right_idx = len(y) - 1  # If no valid right index, use the end of the array
    else:
        right_idx = right_candidates[0] + peak_index
    fwam = x[right_idx] - x[left_idx]
    sigma = abs(fwam / get_anymax_factor(ratio))  # Convert to sigma
    #cap sigma
    return min(sigma, MAX_SIGMA)

def estimate_average_sigma(x, y, peak_index):
    total = 0
    count = 0
    for i in range(ESTIMATE_SIGMA_ITERATIONS_START, ESTIMATE_SIGMA_ITERATIONS_END - 1):
        total += estimate_sigma(x, y, peak_index, i / ESTIMATE_SIGMA_ITERATIONS_END)
        count += 1
    return total / count


def filter_by_max_peak_height(y, peaks, peak_info):
    peaks = np.asarray(peaks, dtype=int)
    if peaks.size == 0:
        return peaks

    # Height test (uses heights computed by find_peaks)
    heights = np.asarray(peak_info.get("peak_heights", np.full_like(peaks, np.nan, dtype=float)))
    ok_height = np.abs(heights) >= float(MIN_ABSOLUTE_PEAK_HEIGHT)

    # Prominence test (re-compute to match y and peaks)
    prominences = peak_prominences(y, peaks)[0]  # non-negative
    ok_prom = prominences >= float(MIN_PROMINENCE)

    keep_mask = ok_height & ok_prom
    return peaks[keep_mask]

def trim_peaks(peak_amplitudes, peak_centers, peak_sigmas, num_gaussians):
    k = max(0, int(num_gaussians))
    order = np.argsort(peak_amplitudes)[::-1]  # sort by amplitude, descending
    keep = order[:k]                           # works for k = 0

    peak_amplitudes = np.asarray(peak_amplitudes)[keep]
    peak_centers    = np.asarray(peak_centers)[keep]
    peak_sigmas     = np.asarray(peak_sigmas)[keep]

    return peak_amplitudes, peak_centers, peak_sigmas

#TODO: More initial guesses or maybe guess on mcd data itself
def generate_initial_guesses_A(x, y_abs, num_gaussians = MAX_BASIS_GAUSSIANS):

    # Smooth the noisy data
    y_smoothed = savgol_filter(y_abs, window_length=WINDOW_LENGTH, polyorder=POLYORDER)
    # Calculate the numerical derivatives
    d_y_smoothed = np.gradient(y_smoothed, x)
    # Calculate the 2nd numerical derivatives
    dd_y = np.gradient(d_y_smoothed, x)
    dd_y_smoothed = np.gradient(d_y_smoothed, x)
    dd_y_smoothed = savgol_filter(dd_y_smoothed, window_length=WINDOW_LENGTH, polyorder=POLYORDER)
    # Find peaks in the negative second derivative (to locate the centers of Gaussians)
    prominence = PROMINENCE_PERECENT * np.nanmax(dd_y)
    height = HEIGHT_THRESHOLD * np.nanmax(dd_y)

    dd_y_peaks_all, peak_info = find_peaks(-dd_y_smoothed, height=height, distance=DISTANCE, prominence=prominence)

    #filter peaks
    dd_y_peaks_all = filter_by_max_peak_height(-dd_y_smoothed, dd_y_peaks_all, peak_info)
    dd_y_peaks = filter_peaks_deltax(x, dd_y_peaks_all)

    peak_centers = x[dd_y_peaks]
    peak_amplitudes = y_smoothed[dd_y_peaks]
    # this would work if my gaussian is normalized to unit height. lets try writing this so that we are normalized to unit area. brb
    peak_sigmas = [estimate_average_sigma(x, y_smoothed, peak) for peak in dd_y_peaks]

    peak_amplitudes, peak_centers, peak_sigmas = trim_peaks(peak_amplitudes, peak_centers, peak_sigmas, num_gaussians)

    print(f'Initial Guess Peak Centers A: {peak_centers}')
    print(f'Initial Guess Peak Sigmas A: {peak_sigmas}')
    print(f'Intial Guess Peak Amplitudes A: {peak_amplitudes}')
    # 6) Keep strongest peaks if too many
    return peak_amplitudes, peak_centers, peak_sigmas

def generate_initial_guesses_B(x, y_mcd, num_gaussians=MAX_BASIS_GAUSSIANS):
    """
    Find candidate Gaussian peaks in the MCD data mcd(x) using the smoothed signal directly.
    No second derivative is used.
    Returns (peak_amplitudes, peak_centers, peak_sigmas).
    """

    # 1) Smooth the MCD data to reduce noise
    mcd_smoothed = savgol_filter(y_mcd, window_length=WINDOW_LENGTH, polyorder=POLYORDER)

    # 2) Thresholds based on the smoothed MCD signal
    mcd_max = np.nanmax(mcd_smoothed)
    prominence = PROMINENCE_PERECENT * mcd_max
    height = HEIGHT_THRESHOLD * mcd_max

    # 3) Find peaks directly in the smoothed MCD
    peaks_all, info = find_peaks(mcd_smoothed,height=height,distance=DISTANCE,prominence=prominence)

    # 4) Apply your existing post-filters
    peaks_all = filter_by_max_peak_height(mcd_smoothed, peaks_all, info)
    peaks = filter_peaks_deltax(x, peaks_all)

    # 5) Build outputs
    peak_centers = x[peaks]
    peak_amplitudes = mcd_smoothed[peaks]
    peak_sigmas = np.array([estimate_average_sigma(x, mcd_smoothed, i) for i in peaks], dtype=float)

    peak_amplitudes, peak_centers, peak_sigmas = trim_peaks(peak_amplitudes, peak_centers, peak_sigmas, num_gaussians)

    print(f'Initial Guess Peak Centers B: {peak_centers}')
    print(f'Initial Guess Peak Sigmas B: {peak_sigmas}')
    print(f'Intial Guess Peak Amplitudes B: {peak_amplitudes}')
    # 6) Keep strongest peaks if too many
    return peak_amplitudes, peak_centers, peak_sigmas

def merge_keep_all_A_add_far_B_debug(amp_abs, ctr_abs, sigma_abs,amp_mcd, ctr_mcd, sigma_mcd,merge_dx):
    import math

    # Pack & sort A by center
    # Build list of (amp, center, sigma) for A
    A = list(zip(amp_abs, ctr_abs, sigma_abs))
    # Convert all values to floats
    A = [(float(amp), float(center), float(sigma)) for amp, center, sigma in A]
    # Sort by center value
    A.sort(key=lambda peak: peak[1])

    # Build list of (amp, center, sigma) for B (no sorting yet)
    B = list(zip(amp_mcd, ctr_mcd, sigma_mcd))
    B = [(float(amp), float(center), float(sigma)) for amp, center, sigma in B]

    print("\n--- A centers (sorted) ---")
    print([a[1] for a in A])

    merged = list(A)

    print("\n--- Scanning B peaks ---")
    for b in B:
        ctrB = b[1]
        # distances to all A centers
        dists = [abs(ctrB - a[1]) for a in A]
        min_dist = min(dists) if dists else math.inf
        if dists:
            print(f"B ctr={ctrB:.6f}  min|B-A|={min_dist:.3f}  (merge_dx={merge_dx}) --> ", end="")
        else:
            print(f"B ctr={ctrB:.6f}  (no A peaks) --> ", end="")

        if not A or min_dist > merge_dx:
            print("KEEP")
            merged.append(b)
        else:
            print("DROP (too close to A)")

    merged.sort(key=lambda p: p[1])
    amps  = [p[0] for p in merged]
    ctrs  = [p[1] for p in merged]
    sigs  = [p[2] for p in merged]

    print("\n--- MERGED centers ---")
    print(ctrs)
    print("---debug done-------------------------------------\n\n")

    return amps, ctrs, sigs

def guess_on_all_data(x, y_abs, y_mcd, merge_dx=MERGE_DX, max_gc=10, min_gc=0):
    # First, get guesses from absorption data
    amp_abs, ctr_abs, sigma_abs = generate_initial_guesses_A(x, y_abs, num_gaussians=max_gc)

    # Limit how many peaks we allow from MCD data so we don't exceed MAX_BASIS_GAUSSIANS
    # Get guesses from MCD data
    amp_mcd, ctr_mcd, sigma_mcd = generate_initial_guesses_B(x, y_mcd, num_gaussians=max_gc)

    print("len A:", len(amp_abs), len(ctr_abs), len(sigma_abs))
    print("len B:", len(amp_mcd), len(ctr_mcd), len(sigma_mcd))
    print("B centers preview:", list(ctr_mcd))

    peak_amplitudes,peak_centers,peak_sigmas = merge_keep_all_A_add_far_B_debug(amp_abs, ctr_abs, sigma_abs, amp_mcd, ctr_mcd, sigma_mcd, merge_dx)

    print(f'Initial Guess Peak Centers: {peak_centers}')
    print(f'Initial Guess Peak Sigmas: {peak_sigmas}')
    print(f'Intial Guess Peak Amplitudes: {peak_amplitudes}')

    pc_length = len(peak_centers)
    #start standalone qt window
    app = QApplication.instance() or QApplication(sys.argv)
    #TODO: fix so no more -1 -- needs to be fixed in brute_force and fit_models
    tfits = total_fits(pc_length, min_gc + 1, max_gc + 1)

    try:
        continue_fit = show_peak_centers_window(x, y_mcd, tfits, peak_centers)
    except Exception as e:
        print(f"Center plotting failed: {e}")
        sys.exit(1)

    if not continue_fit:
        print("Program terminated.")
        sys.exit(0)

    print("Continuing program...")

    return peak_amplitudes, peak_centers, peak_sigmas

#TODO: maybe rank peaks and not remove in both directions
#removes peaks in both directions
def filter_peaks_deltax(x, peaks):
    peak_list = list(peaks)
    center_prev = x[peaks[0]] #last center because of ordering
    prev_peak = peaks[0]
    #every peak but the first
    for peak in peaks[1:]:
        center = x[peak]
        if center_prev-center < MIN_PEAK_X_DISTANCE:
            peak_list.remove(peak)
            if prev_peak in peak_list:
                peak_list.remove(prev_peak)
        center_prev = center
        prev_peak = peak
    return np.array(peak_list)

def total_fits(m, n, k):
    return sum((2 ** i) * comb(m, i) for i in range(n, min(k, m+1)))
def brute_force_models(x, y_abs, y_mcd, min_gc=0, max_gc=10):
    model_list = [
        gaussianModels.model_stable_gaussian_sigma,
        gaussianModels.model_stable_gaussian_deriv_sigma
    ]

    # Generate peak guesses
    peak_amplitudes, peak_centers, peak_sigmas = guess_on_all_data(x, y_abs, y_mcd, max_gc=max_gc,min_gc=min_gc)

    # Bundle each peak's parameters into a tuple
    peaks = list(zip(peak_amplitudes, peak_centers, peak_sigmas))

    # Get all subsets of size > k
    all_subsets = []
    for r in range(max(1, min_gc + 1), len(peaks) + 1):
        all_subsets.extend(itertools.combinations(peaks, r))

    # Convert each combination into a list of peaks
    all_subsets = [list(subset) for subset in all_subsets]
    print(f"total subset length: {len(all_subsets)}")
    all_models = []

    # Loop through subsets
    total_subsets = len(all_subsets)
    for idx, subset in enumerate(all_subsets, start=1):
        print(f"{idx}/{total_subsets} - subset size {len(subset)}")
        # Try all assignments of models to peaks in this subset
        for model_choices in itertools.product(model_list, repeat=len(subset)):
            composite_model = None
            params = None

            # Unpack into (pa, pc, ps)
            #TODO: maybe change prefix
            for i, ((pa, pc, ps), base_model) in enumerate(zip(subset, model_choices)):
                prefix = f"p{i}_"  # unique prefix for each peak
                m = Model(base_model.func, prefix=prefix)

                if composite_model is None:
                    composite_model = m
                    params = m.make_params(amplitude=pa, center=pc, sigma=ps)
                else:
                    composite_model += m
                    params.update(m.make_params(amplitude=pa, center=pc, sigma=ps))

            all_models.append((composite_model, params))
    return all_models


def fit_models(mcd_df, min_gc = 0, max_gc = 10, percentage_range=PERCENTAGE_RANGE):
    """
    Fit all brute-force-generated models to the data in df.

    percentage_range: +/- percent range around initial value for sigma, center, amplitude
    """
    #fix for old data for our lab
    if 'intensity_extinction' in mcd_df.columns:
        mcd_df.rename(columns={'intensity_extinction': 'intensity'}, inplace=True)

    # Prepare the dataframe
    #TODO: check if field is used correctly
    mcd_df['wavenumber'] = 1e7 / mcd_df['wavelength']
    mcd_df['scaled_absorption'] = mcd_df['intensity'] / (mcd_df['wavenumber'] * 1.315 * 326.6)
    mcd_df['scaled_MCD'] = mcd_df['R_signed'] / (mcd_df['wavenumber'] * 1.315 * 152.5)  # Is this even orientational averaging? I get reasonable values if I dont do the orientational averaging for MCD.

    # Extract x and y values
    wavenumbers = mcd_df['wavenumber'].values
    scaled_absorption = mcd_df['scaled_absorption'].values
    scaled_mcd = mcd_df['scaled_MCD'].values

    mask = ~np.isnan(wavenumbers) & ~np.isnan(scaled_absorption) & ~np.isnan(scaled_mcd)
    x = wavenumbers[mask]
    y_abs = scaled_absorption[mask]
    z_mcd = scaled_mcd[mask]

    results = BfResult(x, y_abs, z_mcd)
    #-1 for inclusive
    #TODO: fix inclusivity in the functions themselves
    model_param_pairs = brute_force_models(x, y_abs, z_mcd, min_gc= min_gc - 1, max_gc= max_gc - 1)

    count = 0
    total = len(model_param_pairs)

    for model, params in model_param_pairs:
        # Largest absolute value in the measured MCD data (used for scaling)
        max_data_value = float(np.nanmax(np.abs(z_mcd)) or 1e-12)

        # Figure out how big each component is when its amplitude = 1 because gaussian derivatives scale to amp differently
        component_unit_max = {}
        for param_name in params:
            if param_name.endswith('amplitude'):
                prefix = param_name[:-len('amplitude')]

                # Copy parameters and set THIS amplitude to 1.0
                temp_params = params.copy()
                temp_params[param_name].set(value=1.0)

                # Evaluate just this component
                components = model.eval_components(x=x, params=temp_params)
                comp_data = components.get(prefix)

                # Store the maximum absolute value for scaling purposes
                if comp_data is not None:
                    component_unit_max[prefix] = float(np.nanmax(np.abs(comp_data))) or 1e-12
                else:
                    component_unit_max[prefix] = 1e-12

        # Now set reasonable bounds for each parameter
        for name, p in params.items():
            delta = abs(p.value) * (percentage_range / 100.0)

            if name.endswith('sigma'):
                # Let widths shrink more: down to 10% of original, still >0
                min_sigma = max(1e-12, 0.1 * abs(p.value))
                max_sigma = p.value + delta
                p.set(
                    min=min_sigma,
                    max=max_sigma,
                    vary=True
                )

            elif name.endswith('center'):
                # Peak position can move ±delta
                p.set(
                    min=p.value - delta,
                    max=p.value + delta,
                    vary=True
                )

            elif name.endswith('amplitude'):
                # Allow sign flips; limit size based on data and the component's natural size
                prefix = name[:-len('amplitude')]
                unit_max = component_unit_max[prefix]
                #allows larger scale for derivative shapes
                allowed_amp = AMPLITUDE_SCALE_LIMIT * (max_data_value / unit_max)
                p.set(
                    min=-allowed_amp,
                    max=allowed_amp,
                    vary=True
                )

        try:
            res = model.fit(z_mcd, params, x=x)
            #plot_components_visible(res,x,z)

            count += 1
            print(f"fit {count} of {total}")
            results.add_result(res)
        except Exception:
            print("Exception raised while fitting\n")
            pass

    return results
