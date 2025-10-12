import math
import sys
import time

import scipy.integrate as integrate
from PySide6.QtWidgets import QApplication, QDialog
from scipy.signal import find_peaks, savgol_filter, peak_prominences
from . import gaussianModels
from . import plotwindow
from math import comb
import itertools
from lmfit.model import ModelResult, Model
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from joblib import Parallel, delayed

from . import fitConfig
from .fitConfig import FitConfig
from .gaussianModels import stable_gaussian_sigma
from .plotwindow import MatplotlibGallery, guessWindow, MainResultWindow


#TODO: Docs and use pyside6 for parameters

#TODO: add BIC back and maybe other metrics
class BfResult:
    def __init__(self, datax, datay, dataz, redchi_threshold=None, residual_rms_threshold=None, bic_threshold=None):
        """
        Initialize the bfResult container with optional filter thresholds.

        Parameters:
            redchi_threshold (float): Max allowed reduced chi-square value. None = no filter.
            residual_rms_threshold (float): Max allowed residual RMS. None = no filter.
        """
        self.dataX = datax
        self.dataY = datay
        self.dataZ = dataz
        self.mcd_results = [()]  # List of lmfit.ModelResult objects
        self.redchi_threshold = redchi_threshold
        self.residual_rms_threshold = residual_rms_threshold
        self.bic_threshold = bic_threshold

        #default weights for combo metric
        self.bic_w = 0.25
        self.redchi_w = 0.5
        self.rms_w = 0.25

    def add_result(self, result):
        """Add a new lmfit ModelResult to the list."""
        self.mcd_results.append(result)

    def _residual_rms(self, result: ModelResult):
        """Compute the RMS of residuals for a fit."""
        return np.sqrt(np.mean(result.residual ** 2))

    def _filter_results(self, gc_start=None, gc_end=None, min_sigma = None, max_sigma = None, min_amplitude = None, max_amplitude = None):
        """
        Filter results based on thresholds set at initialization and number of components.
        Returns a list of ModelResult objects that pass all thresholds.

        Args:
            gc_start (int, optional): Minimum number of components allowed.
            gc_end (int, optional): Maximum number of components allowed.
        """

        #get rid of empty set
        filtered_fst = self.mcd_results
        filtered = []
        for res in filtered_fst:
            if res != ():
                filtered.append(res)



        # --- Threshold filters ---
        if self.redchi_threshold is not None:
            filtered = [(r_mcd,r_abs) for (r_mcd,r_abs) in filtered if r_mcd.redchi < self.redchi_threshold]

        if self.residual_rms_threshold is not None:
            filtered = [(r_mcd,r_abs) for (r_mcd,r_abs) in filtered if self._residual_rms(r_mcd) < self.residual_rms_threshold]

        if self.bic_threshold is not None:
            filtered = [(r_mcd,r_abs) for (r_mcd,r_abs) in filtered if r_mcd.bic < self.bic_threshold]

        # --- Component count filter ---
        if gc_start is not None or gc_end is not None:
            start = gc_start if gc_start is not None else 0
            end = gc_end if gc_end is not None else float('inf')

            def count_components(res):
                # count unique prefixes (A0_, B1_, etc.)
                param_names = res.params.keys()
                prefixes = {name.split('_', 1)[0] for name in param_names}
                return len(prefixes)

            filtered = [(r_mcd,r_abs) for (r_mcd,r_abs) in filtered if start <= count_components(r_mcd) <= end]

        #filter by min and max sigma/amp

        def get_sig(res):
            params_dict = self.eval_result(res)
            return [v for k, v in params_dict.items() if k.endswith("_sigma")]

        def get_amp(res):
            params_dict = self.eval_result(res)
            return [v for k, v in params_dict.items() if k.endswith("_amplitude")]

        # Ensure bounds are not None
        min_sigma = 0 if min_sigma is None else min_sigma
        max_sigma = float("inf") if max_sigma is None else max_sigma
        min_amplitude = 0 if min_amplitude is None else min_amplitude
        max_amplitude = float("inf") if max_amplitude is None else max_amplitude

        filtered_results = []
        for (r_mcd,r_abs) in filtered:
            sigs = get_sig(r_mcd)
            amps = get_amp(r_mcd)

            # Skip if no components
            if not sigs or not amps:
                continue

            sig_in_range = all(min_sigma <= s <= max_sigma for s in sigs)
            amp_in_range = all(min_amplitude <= abs(a) <= max_amplitude for a in amps)

            if sig_in_range and amp_in_range:
                filtered_results.append((r_mcd,r_abs))

        return filtered_results

    #TODO: add optional gaussian count g_n parameter
    def n_best_results(self, n=3, metric='redchi', gc_start=None, gc_end=None, min_sigma = None,
                       max_sigma = None, min_amplitude = None, max_amplitude = None):

        filtered = self._filter_results(gc_start=gc_start, gc_end=gc_end, min_sigma=min_sigma,
                                        max_sigma=max_sigma, min_amplitude = min_amplitude, max_amplitude=max_amplitude)
        if metric == 'residual_rms':
            items = [item for item in filtered if item[0] is not None]
            return sorted(items, key=lambda x: self._residual_rms(x[0]) + self._residual_rms(x[1]))[:n]
        elif metric == 'combo':
            items = [item for item in filtered if item[0] is not None]
            return sorted(items, key=lambda x: self._combo_metric(x[0] + self._combo_metric(x[1])))[:n]
        else:
            # Assume metric is an attribute of the first ModelResult (item[0])
            items = [item for item in filtered if hasattr(item[0], metric)]
            return sorted(items, key=lambda r: getattr(r[0], metric))[:n]

    def _combo_metric(self, result: ModelResult):
        return self.bic_w * getattr(result, 'bic') + self.redchi_w * getattr(result, 'redchi') + self.rms_w * self._residual_rms(result)

    # TODO: add optional gaussian count g_n parameter
    def get_plot_figs(self, n=3, metric='redchi', gc_start=None, gc_end=None,
                      min_sigma=None, max_sigma=None, min_amplitude=None, max_amplitude=None):
        """
        Plot the top N best fits over the data from both mcd and abs results.
        """
        fig_list = []

        results = self.n_best_results(
            n=n, metric=metric, gc_start=gc_start, gc_end=gc_end,
            min_sigma=min_sigma, max_sigma=max_sigma,
            min_amplitude=min_amplitude, max_amplitude=max_amplitude
        )

        for i, (r_mcd, r_abs) in enumerate(results, 1):
            fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)  # 2 rows, 1 column subplots

            # Plot on the first axis
            self._plot_components_visible(r_mcd, self.dataX, self.dataZ, i, axs[0])

            # Plot on the second axis
            self._plot_components_visible(r_abs, self.dataX, self.dataY, i, axs[1])

            self.print_A_over_D((r_mcd, r_abs), self.dataX)

            plt.tight_layout()
            fig_list.append(fig)
            plt.close(fig)  # Close to free memory if plotting many

        return fig_list
    def _plot_components_visible(self, result, x, z, i, ax):
        """Plot the original z-data, model components, metrics, and parameters on given axis."""
        from collections import defaultdict

        # No new figure creation here — use the provided ax

        # Plot measured data
        ax.plot(x, z, '-', lw=1, alpha=0.7, label='Measured Data')

        # Evaluate and plot each component
        components = result.eval_components(x=x)
        for name, comp in components.items():
            ax.plot(x, comp, lw=2, label=name)

        # Axis labels
        ax.set_xlabel("X")
        ax.set_ylabel("Intensity")

        # Metrics in top-left (axes coords)
        metrics = {
            "Residual RMS": self._residual_rms(result),
            "BIC": getattr(result, 'bic'),
            "redchi": result.redchi,
            "Combo": self._combo_metric(result)
        }
        metrics_text = "\n".join(f"{name}: {val:.3g}" for name, val in metrics.items())
        ax.text(0.02, 0.95, metrics_text,
                transform=ax.transAxes, fontsize=8,
                va='top', ha='left')

        # Index number in top-right (axes coords)
        ax.text(0.98, 0.95, f"#{i}",
                transform=ax.transAxes, fontsize=12,
                va='top', ha='right')

        # Parameters from eval_result (grouped, with separation) – placed in data coords
        params_dict = self.eval_result(result)  # your single-result eval function

        grouped = defaultdict(list)
        for name, val in params_dict.items():
            prefix = name.split('_', 1)[0]  # e.g., "A0", "B1"
            grouped[prefix].append(f"{name}: {val:.3g}")

        # Build lines with a blank line between groups
        params_lines = []
        for prefix in sorted(grouped.keys()):
            params_lines.extend(grouped[prefix])
            params_lines.append("")  # blank line between groups
        params_text = "\n".join(params_lines).strip()

        # Position to the left of the first x value
        x_min = min(x)
        y_max = max(z)
        offset = 0.15 * (max(x) - min(x))  # 15% of x range for margin
        ax.text(x_min - offset, y_max, params_text,
                fontsize=7, va='top', ha='right',
                transform=ax.transData)

        ax.legend()

        # print results too
        self.print_eval_result(result, i)
    def eval_result(self, result):
        if not hasattr(result, "params"):
            raise ValueError("Provided object has no 'params' attribute")

        return {name: p.value for name, p in result.params.items()}

    def print_eval_result(self, result, i):
        """Pretty-print parameters from a single lmfit ModelResult."""
        params_dict = self.eval_result(result)
        print(f"=== Fit Parameters === Index: {i}")
        for name, value in params_dict.items():
            print(f"{name:20} {value:.6g}")
        print(f"=== Parameters === END: {i}")

    def print_A_over_D(self, result, x):
        mcd_res, abs_res = result
        mcd_comps = mcd_res.eval_components(x=x).items()
        abs_comps = abs_res.eval_components(x=x).items()

        for i, ((name_mcd, _), (name_abs, _)) in enumerate(zip(mcd_comps, abs_comps)):
            # amplitudes are stored in the fit parameters
            mcd_amp = mcd_res.params[f"{name_mcd}amplitude"].value
            abs_amp = abs_res.params[f"{name_abs}amplitude"].value

            print(f"A/D or B/D values with index {i} in fit and names {name_mcd} {name_abs}: {np.abs(mcd_amp / abs_amp)}")

def brute_force_models(x, y_abs, y_mcd, fc = FitConfig()):
    model_list = [
        gaussianModels.model_stable_gaussian_sigma,
        gaussianModels.model_stable_gaussian_deriv_sigma
    ]

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
            composite_mcd_model = None
            params_mcd = None

            composite_abs_model = None
            params_abs = None

            # Unpack into (pa, pc, ps)
            #TODO: maybe change prefix
            for i, ((pa, pc, ps), base_model) in enumerate(zip(subset, model_choices)):
                if base_model == gaussianModels.model_stable_gaussian_deriv_sigma:
                    prefix = f"A{i}_"
                else:  # gaussianModels.model_stable_gaussian_sigma
                    prefix = f"B{i}_"
                m_mcd = Model(base_model.func, prefix=prefix)
                #abs only gaussian
                m_abs = Model(stable_gaussian_sigma, prefix=f"B{i}")

                if composite_mcd_model is None:
                    composite_mcd_model = m_mcd
                    params_mcd = m_mcd.make_params(amplitude=pa, center=pc, sigma=ps)
                else:
                    composite_mcd_model += m_mcd
                    params_mcd.update(m_mcd.make_params(amplitude=pa, center=pc, sigma=ps))

                #add abs model
                if composite_abs_model is None:
                    composite_abs_model = m_abs
                    params_abs = m_abs.make_params(amplitude=pa, center=pc, sigma=ps)
                else:
                    composite_abs_model += m_abs
                    params_abs.update(m_abs.make_params(amplitude=pa, center=pc, sigma=ps))

            all_models.append(((composite_mcd_model, params_mcd),(composite_abs_model, params_abs)))
    return all_models, fc


def fit_models(mcd_df, fc = None, processes = 4):
    """
    Fit all brute-force-generated models to the data in df.

    percentage_range: +/- percent range around initial value for sigma, center, amplitude
    """

    # Prepare the dataframe
    #TODO: check if field is used correctly

    # boltmans = 287893
    #
    #

    x = mcd_df["wavenumber_out"]
    y_abs = mcd_df["uvvis_extinction_abs_molar-1cm-1_out"] / (mcd_df["wavenumber_out"] * 326.6)
    #R_signed_extiction_per_tesla
    z_mcd = mcd_df["deltaextinctionpertesla_mcdavg_molar-1cm-1T-1_out"] / (mcd_df["wavenumber_out"] * 152.5)  # Is this even orientational averaging? I get reasonable values if I dont do the orientational averaging for MCD.


    results = BfResult(x, y_abs, z_mcd)
    if fc is not None:
        model_param_pairs, fc = brute_force_models(x, y_abs, z_mcd, fc)
    else:
        model_param_pairs, fc = brute_force_models(x, y_abs, z_mcd)

    ##########################################################################################
    def split_into_n(seq, n):
        """Evenly split a sequence into n chunks (last chunks may differ by 1)."""
        n = max(1, min(n, len(seq)))  # clamp
        k, m = divmod(len(seq), n)
        # First m chunks have size k+1, the rest have size k
        chunks = []
        start = 0
        for i in range(n):
            size = k + (1 if i < m else 0)
            if size == 0:
                break
            chunks.append(seq[start:start + size])
            start += size
        return chunks
    ############################################################################################

    chunks = split_into_n(model_param_pairs, processes)

    #just to know how long the fitting took
    start = time.perf_counter()  # Start timer

    results_lists = Parallel(n_jobs=processes, backend="loky")(
        delayed(fit_worker)(f"Worker-{i + 1}", x, z_mcd, y_abs, fc, chunk) for i, chunk in enumerate(chunks))


    # Loop through each list of results returned from each worker
    for sublist in results_lists:
        # Loop through each individual result in that sublist
        for r in sublist:
            results.add_result(r)

    end = time.perf_counter()

    elapsed = end - start

    minutes = int(elapsed // 60)
    seconds = elapsed % 60

    print("\n####################################### -> Start of time print")
    print(f"Elapsed time: {minutes} min {seconds:.2f} sec for {processes} workers and {len(model_param_pairs)} fits")
    print("####################################### -> End of time print")

    # --- Show results window before returning ---
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    win = MainResultWindow(bfResult=results)
    win.show()

    app.exec()

    return results

def fit_worker(name, x, z_mcd, y_abs, fc, model_param_pairs):
    count = 0
    total = len(model_param_pairs)

    out = []

    for ((model_mcd, params_mcd), (model_abs, params_abs)) in model_param_pairs:
        # Largest absolute value in the measured MCD/ABS data (used for scaling)
        max_data_value_mcd = float(np.nanmax(np.abs(z_mcd)) or 1e-12)
        max_data_value_abs = float(np.nanmax(np.abs(y_abs)) or 1e-12)

        # --- MCD Component Scaling ---
        component_unit_max_mcd = {}
        for param_name in params_mcd:
            if param_name.endswith('amplitude'):
                prefix = param_name[:-len('amplitude')]
                temp_params = params_mcd.copy()
                temp_params[param_name].set(value=1.0)

                components = model_mcd.eval_components(x=x, params=temp_params)
                comp_data = components.get(prefix)

                component_unit_max_mcd[prefix] = float(np.nanmax(np.abs(comp_data))) or 1e-12

        # --- ABS Component Scaling ---
        component_unit_max_abs = {}
        for param_name in params_abs:
            if param_name.endswith('amplitude'):
                prefix = param_name[:-len('amplitude')]
                temp_params = params_abs.copy()
                temp_params[param_name].set(value=1.0)

                components = model_abs.eval_components(x=x, params=temp_params)
                comp_data = components.get(prefix)

                component_unit_max_abs[prefix] = float(np.nanmax(np.abs(comp_data))) or 1e-12

        # --- Set bounds for MCD parameters ---
        for p_name, p in params_mcd.items():
            delta = abs(p.value) * (fc.PERCENTAGE_RANGE / 100.0)
            delta_ctr = fc.DELTA_CTR

            if p_name.endswith('sigma'):
                min_sigma = max(1e-12, 0.1 * abs(p.value))
                p.set(min=min_sigma, max=p.value + delta, vary=True)

            elif p_name.endswith('center'):
                p.set(min=p.value - delta_ctr, max=p.value + delta_ctr, vary=True)

            elif p_name.endswith('amplitude'):
                prefix = p_name[:-len('amplitude')]
                unit_max = component_unit_max_mcd.get(prefix, 1e-12)
                allowed_amp = fc.AMPLITUDE_SCALE_LIMIT * (max_data_value_mcd / unit_max)
                p.set(min=-allowed_amp, max=allowed_amp, vary=True)

        # --- Set bounds for ABS parameters ---
        for p_name, p in params_abs.items():
            delta = abs(p.value) * (fc.PERCENTAGE_RANGE / 100.0)
            delta_ctr = fc.DELTA_CTR

            if p_name.endswith('sigma'):
                min_sigma = max(1e-12, 0.1 * abs(p.value))
                p.set(min=min_sigma, max=p.value + delta, vary=True)

            elif p_name.endswith('center'):
                p.set(min=p.value - delta_ctr, max=p.value + delta_ctr, vary=True)

            elif p_name.endswith('amplitude'):
                prefix = p_name[:-len('amplitude')]
                unit_max = component_unit_max_abs.get(prefix, 1e-12)
                allowed_amp = fc.AMPLITUDE_SCALE_LIMIT * (max_data_value_abs / unit_max)
                p.set(min=0.0, max=allowed_amp, vary=True)  # ABS is positive

        try:
            res_mcd = model_mcd.fit(z_mcd, params_mcd, x=x)
            res_abs = model_abs.fit(y_abs, params_abs, x=x)

            count += 1
            print(f"Process {name}: fit {count} of {total}")
            out.append((res_mcd, res_abs))

        except Exception as e:
            print(f"Exception raised while fitting: {e}\n")
            continue

    return out
