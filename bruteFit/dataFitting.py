import sys
from PySide6.QtWidgets import QApplication, QDialog
from scipy.signal import find_peaks, savgol_filter, peak_prominences
from . import gaussianModels
from . import plotwindow
from math import comb
import itertools
from lmfit.model import ModelResult, Model
import matplotlib.pyplot as plt
import numpy as np


from . import fitConfig
from .fitConfig import FitConfig
from .plotwindow import MatplotlibGalleryWindow, guessWindow


#TODO: Docs and use pyside6 for parameters

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

def brute_force_models(x, y_abs, y_mcd, fc = FitConfig()):
    model_list = [
        gaussianModels.model_stable_gaussian_sigma,
        gaussianModels.model_stable_gaussian_deriv_sigma
    ]

    app = QApplication(sys.argv)
    dlg = guessWindow(x, y_abs, y_mcd, fc)
    if dlg.exec() == QDialog.DialogCode.Accepted:  # blocks until user clicks
        peak_amplitudes, peak_centers, peak_sigmas = dlg.get_guess()
        fc = dlg.get_fc()
    else:
        raise RuntimeError("User cancelled peak guesses.")

    # Bundle each peak's parameters into a tuple
    peaks = list(zip(peak_amplitudes, peak_centers, peak_sigmas))

    # Get all subsets of size > k
    all_subsets = []
    for r in range(max(1, fc.MIN_GC), len(peaks) + 1):
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
    return all_models, fc


def fit_models(mcd_df, fc = None):
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
    if fc is not None:
        model_param_pairs, fc = brute_force_models(x, y_abs, z_mcd, fc)
    else:
        model_param_pairs, fc = brute_force_models(x, y_abs, z_mcd)

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
            delta = abs(p.value) * (fc.PERCENTAGE_RANGE / 100.0)

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
                allowed_amp = fc.AMPLITUDE_SCALE_LIMIT * (max_data_value / unit_max)
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
