import sys
from math import comb

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMainWindow, QScrollArea, QSizePolicy
)
import matplotlib
from scipy.signal import peak_prominences, savgol_filter, find_peaks


from math import comb
from dataclasses import asdict
from typing import Optional, Sequence, Dict, Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QDialogButtonBox,
    QPushButton, QDoubleSpinBox, QSpinBox, QWidget, QLabel
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from . import fitConfig as configContainer

matplotlib.use("QtAgg")

#TODO: more complex gui if continuing with project - more options and final value printing - use BfResult as init param
class MatplotlibGalleryWindow(QMainWindow):
    def __init__(self, figures=None, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Matplotlib Plots")
        self.resize(700, 700)

        # ---- Central widget with a layout ----
        central = QWidget(self)
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central)

        # ---- Scrollable area just for the plots ----
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        central_layout.addWidget(scroll)

        # Container inside the scroll area
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        # Add each figure as a canvas + toolbar pair
        for fig in figures:
            block = QWidget(container)
            block_layout = QVBoxLayout(block)
            block_layout.setContentsMargins(8, 8, 8, 8)

            canvas = FigureCanvas(fig)

            dpi = fig.get_dpi()
            w_px = int(fig.get_figwidth() * dpi)
            h_px = int(fig.get_figheight() * dpi)
            canvas.setMinimumSize(w_px, h_px)
            canvas.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # scroll instead of squeeze

            toolbar = NavigationToolbar(canvas, self)
            block_layout.addWidget(toolbar)
            block_layout.addWidget(canvas)
            layout.addWidget(block)

            canvas.draw()

        layout.addStretch(1)
        scroll.setWidget(container)

class guessWindow(QDialog):
    fig = None
    fc = None
    x = None
    y_mcd = None
    y_abs = None

    pa = None
    pc = None
    ps = None

    def __init__(self, x, y_abs, y_mcd, fc, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("peak confirmation")

        # keep references
        self.x = np.asarray(x, dtype=float)
        self.y_abs = np.asarray(y_abs, dtype=float)
        self.y_mcd = np.asarray(y_mcd, dtype=float)
        self.fc = fc  # your FitConfig instance (edited in place)

        self.resize(1400, 700)

        # ----- Main layout
        layout = QVBoxLayout(self)

        # ----- Upper row: plot (left) + FitConfig editor (right)
        upper_row = QHBoxLayout()
        layout.addLayout(upper_row)

        # Plot area (toolbar + canvas go here)
        self._plot_area = QVBoxLayout()
        upper_row.addLayout(self._plot_area, 3)

        # FitConfig editor area
        self._editor_group = QGroupBox("FitConfig")
        self._form = QFormLayout(self._editor_group)
        upper_row.addWidget(self._editor_group, 2)

        # Build the editor from your fc (no schema changes)
        self._fc_widgets: Dict[str, QWidget] = {}
        self._build_fc_editor()

        # Update button (applies editor values -> fc, then redraws)
        self._btn_update = QPushButton("Update")
        self._btn_update.clicked.connect(self._on_update_clicked)
        self._form.addRow(self._btn_update)

        # ----- Yes/No at the bottom (confirmation)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No
        )
        buttons.accepted.connect(self.accept)   # Yes
        buttons.rejected.connect(self.reject)   # No
        layout.addWidget(buttons)

        # Initial render — do it after the dialog is constructed, so any heavy work
        # doesn’t block showing the window.
        QTimer.singleShot(0, self.update)

    # --------------- Plot helpers ---------------

    @staticmethod
    def get_peak_centers_fig(x, y_mcd, peak_centers, min_gc, max_gc) -> Figure:
        y_at_centers = np.interp(peak_centers, x, y_mcd)

        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)

        ax.plot(x, y_mcd, '-', lw=1, alpha=0.7, label='y (original)')
        ax.scatter(peak_centers, y_at_centers, s=36, zorder=3, label='peak centers')
        for cx in peak_centers:
            ax.axvline(cx, ls='--', lw=0.8, alpha=0.5)

        # total fits
        def total_fits(m, n, k):
            return sum((2 ** i) * comb(m, i) for i in range(n, min(k, m + 1)))

        tfit = total_fits(len(peak_centers), min_gc + 1, max_gc + 1)

        ax.text(0.02, 0.98, f"total fits = {int(tfit)}",
                transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round", alpha=0.25))

        ax.set_xlabel('x')
        ax.set_ylabel('signal')
        ax.legend()
        fig.tight_layout()
        return fig

    def _set_figure_in_ui(self, fig: Figure) -> None:
        """Embed a Matplotlib Figure into the left pane."""
        # remove previous widgets if any
        for i in reversed(range(self._plot_area.count())):
            w = self._plot_area.itemAt(i).widget()
            if w is not None:
                w.setParent(None)

        self.fig = fig
        canvas = FigureCanvas(self.fig)
        toolbar = NavigationToolbar(canvas, self)
        self._plot_area.addWidget(toolbar)
        self._plot_area.addWidget(canvas)

        #hook clicks and set tolerance
        self._click_dx = float(getattr(self.fc, "MIN_PEAK_X_DISTANCE", 0.5) or 0.5)
        self._mpl_cid = canvas.mpl_connect('button_press_event', self._on_click)

        canvas.draw_idle()


    def update(self):  # keep name if you want; it shadows QWidget.update(), which is fine here
        pa, pc, ps = self._guess_on_all_data(self.x, self.y_abs, self.y_mcd)

        # Build and embed the figure
        fig = guessWindow.get_peak_centers_fig(self.x, self.y_mcd, pc, self.fc.MIN_GC, self.fc.MAX_GC)
        self._set_figure_in_ui(fig)

        # cache
        self.pa = pa
        self.pc = pc
        self.ps = ps

    # --------------- FitConfig editor ---------------

    def _build_fc_editor(self) -> None:
        self._clear_form_layout(self._form)
        self._fc_widgets.clear()

        # create simple numeric editors from current fc
        from dataclasses import asdict
        for key, val in asdict(self.fc).items():
            if isinstance(val, float):
                sb = QDoubleSpinBox()
                sb.setDecimals(6); sb.setRange(-1e12, 1e12); sb.setSingleStep(0.01)
                sb.setValue(float(val))
                self._fc_widgets[key] = sb
                self._form.addRow(key, sb)
            elif isinstance(val, int):
                sb = QSpinBox()
                sb.setRange(-2_000_000_000, 2_000_000_000); sb.setSingleStep(1)
                sb.setValue(int(val))
                self._fc_widgets[key] = sb
                self._form.addRow(key, sb)
            else:
                lab = QLabel(str(val)); lab.setEnabled(False)
                self._fc_widgets[key] = lab
                self._form.addRow(key, lab)

        self._form.addRow(QLabel("Edit and click Update."))

    @staticmethod
    def _clear_form_layout(form: QFormLayout) -> None:
        while form.rowCount():
            form.removeRow(0)

    def _apply_editor_to_fc(self) -> None:
        for key, w in self._fc_widgets.items():
            try:
                if isinstance(w, QDoubleSpinBox):
                    setattr(self.fc, key, float(w.value()))
                elif isinstance(w, QSpinBox):
                    setattr(self.fc, key, int(w.value()))
            except Exception:
                pass

    def _on_update_clicked(self) -> None:
        self._apply_editor_to_fc()
        self.update()

    def _remove_peak_near(self, x_click: float, dx: float) -> int:
        """
        Remove the single peak whose center is closest to x_click if within dx.
        Returns the removed index, or -1 if none.
        """
        if self.pc is None or len(self.pc) == 0:
            return -1

        pc = np.asarray(self.pc)
        # returns pc index by argmin and the fact it uses pc
        i = int(np.argmin(np.abs(pc - x_click)))
        if abs(pc[i] - x_click) <= dx:
            # remove aligned entries
            self.pa = np.delete(self.pa, i) if self.pa is not None and len(self.pa) == len(pc) else self.pa
            self.ps = np.delete(self.ps, i) if self.ps is not None and len(self.ps) == len(pc) else self.ps
            self.pc = np.delete(self.pc, i)
            return i
        return -1

    def _on_click(self, event) -> None:
        """
        Matplotlib click callback. Left-click removes the nearest peak within dx.
        """
        if event.inaxes is None or event.xdata is None:
            return
        # Left click only; change to event.button == 3 for right-click behavior
        if getattr(event, "button", 1) != 1:
            return

        removed = self._remove_peak_near(float(event.xdata), 75)
        if removed >= 0:
            self._redraw_current()

    def _redraw_current(self) -> None:
        if self.fig is None:
            return

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # base signal
        ax.plot(self.x, self.y_mcd, '-', lw=1, alpha=0.7, label='y (original)')

        # peaks (if any)
        if self.pc is not None and len(self.pc) > 0:
            y_at = np.interp(self.pc, self.x, self.y_mcd)
            ax.scatter(self.pc, y_at, s=36, zorder=3, label='peak centers')
            for cx in self.pc:
                ax.axvline(cx, ls='--', lw=0.8, alpha=0.5)

        # recompute overlay text with current count
        def total_fits(m, n, k):
            return sum((2 ** i) * comb(m, i) for i in range(n, min(k, m + 1)))

        tfit = total_fits(len(self.pc), self.fc.MIN_GC + 1, self.fc.MAX_GC + 1)

        label = f"total fits = {int(tfit)}"
        if tfit == 0 and len(self.pc) < (self.fc.MIN_GC + 1):
            label += f"  (need ≥{self.fc.MIN_GC + 1} peaks, have {len(self.pc)} peaks)"

        ax.text(0.02, 0.98, label, transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round", alpha=0.25))


        ax.set_xlabel('x')
        ax.set_ylabel('signal')
        ax.legend(loc="best")
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    @staticmethod
    def _get_anymax_factor(ratio):
        if (ratio >= 1):  # return FWHM if ratio is invalid
            print("full width any max has invalid ratio")
            return 2.355
        else:
            return np.sqrt(8 * np.log(1 / ratio))

    def _estimate_sigma(self, x, y, peak_index, ratio):
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
        sigma = abs(fwam / guessWindow._get_anymax_factor(ratio))  # Convert to sigma
        # cap sigma
        return min(sigma, self.fc.MAX_SIGMA)

    def _estimate_average_sigma(self, x, y, peak_index):
        total = 0
        count = 0
        for i in range(self.fc.ESTIMATE_SIGMA_ITERATIONS_START, self.fc.ESTIMATE_SIGMA_ITERATIONS_END - 1):
            total += self._estimate_sigma(x, y, peak_index, i / self.fc.ESTIMATE_SIGMA_ITERATIONS_END)
            count += 1
        return total / count

    def _filter_by_max_peak_height(self, y, peaks, peak_info):
        peaks = np.asarray(peaks, dtype=int)
        if peaks.size == 0:
            return peaks

        # Height test (uses heights computed by find_peaks)
        heights = np.asarray(peak_info.get("peak_heights", np.full_like(peaks, np.nan, dtype=float)))
        ok_height = np.abs(heights) >= float(self.fc.MIN_ABSOLUTE_PEAK_HEIGHT)

        # Prominence test (re-compute to match y and peaks)
        prominences = peak_prominences(y, peaks)[0]  # non-negative
        ok_prom = prominences >= float(self.fc.MIN_PROMINENCE)

        keep_mask = ok_height & ok_prom
        return peaks[keep_mask]

    def _trim_peaks(self, peak_amplitudes, peak_centers, peak_sigmas, num_gaussians):
        k = max(0, int(num_gaussians))
        order = np.argsort(peak_amplitudes)[::-1]  # sort by amplitude, descending
        keep = order[:k]  # works for k = 0

        peak_amplitudes = np.asarray(peak_amplitudes)[keep]
        peak_centers = np.asarray(peak_centers)[keep]
        peak_sigmas = np.asarray(peak_sigmas)[keep]

        return peak_amplitudes, peak_centers, peak_sigmas

    # TODO: More initial guesses or maybe guess on mcd data itself
    def _generate_initial_guesses_A(self, x, y_abs):

        # Smooth the noisy data
        y_smoothed = savgol_filter(y_abs, window_length=self.fc.WINDOW_LENGTH, polyorder=self.fc.POLYORDER)
        # Calculate the numerical derivatives
        d_y_smoothed = np.gradient(y_smoothed, x)
        # Calculate the 2nd numerical derivatives
        dd_y = np.gradient(d_y_smoothed, x)
        dd_y_smoothed = np.gradient(d_y_smoothed, x)
        dd_y_smoothed = savgol_filter(dd_y_smoothed, window_length=self.fc.WINDOW_LENGTH, polyorder=self.fc.POLYORDER)
        # Find peaks in the negative second derivative (to locate the centers of Gaussians)
        prominence = self.fc.PROMINENCE_PERECENT * np.nanmax(dd_y)
        height = self.fc.HEIGHT_THRESHOLD * np.nanmax(dd_y)

        dd_y_peaks_all, peak_info = find_peaks(-dd_y_smoothed, height=height, distance=self.fc.DISTANCE,
                                               prominence=prominence)

        # filter peaks
        dd_y_peaks_all = self._filter_by_max_peak_height(-dd_y_smoothed, dd_y_peaks_all, peak_info)
        dd_y_peaks = self._filter_peaks_deltax(x, dd_y_peaks_all)

        peak_centers = x[dd_y_peaks]
        peak_amplitudes = y_smoothed[dd_y_peaks]
        # this would work if my gaussian is normalized to unit height. lets try writing this so that we are normalized to unit area. brb
        peak_sigmas = [self._estimate_average_sigma(x, y_smoothed, peak) for peak in dd_y_peaks]

        peak_amplitudes, peak_centers, peak_sigmas = self._trim_peaks(peak_amplitudes, peak_centers, peak_sigmas, self.fc.MAX_GC)

        print(f'Initial Guess Peak Centers A: {peak_centers}')
        print(f'Initial Guess Peak Sigmas A: {peak_sigmas}')
        print(f'Intial Guess Peak Amplitudes A: {peak_amplitudes}')
        # 6) Keep strongest peaks if too many
        return peak_amplitudes, peak_centers, peak_sigmas

    def _generate_initial_guesses_B(self, x, y_mcd):
        """
        Find candidate Gaussian peaks in the MCD data mcd(x) using the smoothed signal directly.
        No second derivative is used.
        Returns (peak_amplitudes, peak_centers, peak_sigmas).
        """

        # 1) Smooth the MCD data to reduce noise
        mcd_smoothed = savgol_filter(y_mcd, window_length=self.fc.WINDOW_LENGTH, polyorder=self.fc.POLYORDER)

        # 2) Thresholds based on the smoothed MCD signal
        mcd_max = np.nanmax(mcd_smoothed)
        prominence = self.fc.PROMINENCE_PERECENT * mcd_max
        height = self.fc.HEIGHT_THRESHOLD * mcd_max

        # 3) Find peaks directly in the smoothed MCD
        peaks_all, info = find_peaks(mcd_smoothed, height=height, distance=self.fc.DISTANCE, prominence=prominence)

        # 4) Apply your existing post-filters
        peaks_all = self._filter_by_max_peak_height(mcd_smoothed, peaks_all, info)
        peaks = self._filter_peaks_deltax(x, peaks_all)

        # 5) Build outputs
        peak_centers = x[peaks]
        peak_amplitudes = mcd_smoothed[peaks]
        peak_sigmas = np.array([self._estimate_average_sigma(x, mcd_smoothed, i) for i in peaks], dtype=float)

        peak_amplitudes, peak_centers, peak_sigmas = self._trim_peaks(peak_amplitudes, peak_centers, peak_sigmas, self.fc.MAX_GC)

        print(f'Initial Guess Peak Centers B: {peak_centers}')
        print(f'Initial Guess Peak Sigmas B: {peak_sigmas}')
        print(f'Intial Guess Peak Amplitudes B: {peak_amplitudes}')
        # 6) Keep strongest peaks if too many
        return peak_amplitudes, peak_centers, peak_sigmas

    def _merge_keep_all_A_add_far_B_debug(self, amp_abs, ctr_abs, sigma_abs, amp_mcd, ctr_mcd, sigma_mcd, merge_dx):
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
        amps = [p[0] for p in merged]
        ctrs = [p[1] for p in merged]
        sigs = [p[2] for p in merged]

        print("\n--- MERGED centers ---")
        print(ctrs)
        print("---debug done-------------------------------------\n\n")

        return amps, ctrs, sigs

    def _guess_on_all_data(self, x, y_abs, y_mcd):
        # First, get guesses from absorption data
        amp_abs, ctr_abs, sigma_abs = self._generate_initial_guesses_A(x, y_abs)

        # Limit how many peaks we allow from MCD data so we don't exceed MAX_BASIS_GAUSSIANS
        # Get guesses from MCD data
        amp_mcd, ctr_mcd, sigma_mcd = self._generate_initial_guesses_B(x, y_mcd)

        print("len A:", len(amp_abs), len(ctr_abs), len(sigma_abs))
        print("len B:", len(amp_mcd), len(ctr_mcd), len(sigma_mcd))
        print("B centers preview:", list(ctr_mcd))

        peak_amplitudes, peak_centers, peak_sigmas = self._merge_keep_all_A_add_far_B_debug(amp_abs, ctr_abs, sigma_abs,
                                                                                            amp_mcd, ctr_mcd, sigma_mcd,
                                                                                            self.fc.MERGE_DX)

        print(f'Initial Guess Peak Centers: {peak_centers}')
        print(f'Initial Guess Peak Sigmas: {peak_sigmas}')
        print(f'Intial Guess Peak Amplitudes: {peak_amplitudes}')

        return peak_amplitudes, peak_centers, peak_sigmas

    # TODO: maybe rank peaks and not remove in both directions
    # removes peaks in both directions
    def _filter_peaks_deltax(self, x, peaks):
        peak_list = list(peaks)
        center_prev = x[peaks[0]]  # last center because of ordering
        prev_peak = peaks[0]
        # every peak but the first
        for peak in peaks[1:]:
            center = x[peak]
            if center_prev - center < self.fc.MIN_PEAK_X_DISTANCE:
                peak_list.remove(peak)
                if prev_peak in peak_list:
                    peak_list.remove(prev_peak)
            center_prev = center
            prev_peak = peak
        return np.array(peak_list)

    def get_guess(self):
        return self.pa, self.pc, self.ps
