import sys
from math import comb
import numpy as np
from PySide6 import QtGui
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMainWindow, QScrollArea, QSizePolicy, QLineEdit, QCheckBox, QFileDialog, QApplication
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
import multiprocessing as mp
from . import dataFitting as daf

matplotlib.use("QtAgg")
class MainResultWindow(QMainWindow):
    def __init__(self, bfResult=None, df_fc = None):
        super().__init__()
        self.plot_n = 10
        self.bfResult = bfResult
        self.setWindowTitle("BF Results")
        self.resize(1100, 700)

        self.df_fc = df_fc

        # ---- Central widget for QMainWindow ----
        central = QWidget(self)
        self.setCentralWidget(central)

        # Horizontal split: gallery (left) + buttons (right)
        main_layout = QHBoxLayout(central)

        figures = self.bfResult.get_plot_figs(self.plot_n)
        if len(figures) == 0:
            raise RuntimeError("No results to plot")
        self.gallery = MatplotlibGallery(figures=figures)
        main_layout.addWidget(self.gallery, stretch=1)

        # --- Right: vertical button panel ---
        btn_layout = QVBoxLayout()
        btn_layout.setAlignment(Qt.AlignTop)  # keep buttons at the top

        self._build_controls(btn_layout)

        # Spacer at the bottom if you want them stuck at top
        btn_layout.addStretch(1)

        # Add button panel to the right side (once)
        main_layout.addLayout(btn_layout)

    def _build_controls(self, btn_layout):
        # Helper to make metric buttons
        def add_metric_button(label, metric):
            btn = QPushButton(f"Metric: {label}")
            btn.clicked.connect(
                lambda _, m=metric: self.gallery.set_figures(
                    self.bfResult.get_plot_figs(self.spin_plot_n.value(), metric=m, gc_start=self.spin_gc_start.value(), gc_end=self.spin_gc_end.value(),
                                                min_amplitude=self.spin_min_amp.value(), max_amplitude=self.spin_max_amp.value(), max_sigma=self.spin_max_sigma.value(),
                                                min_sigma=self.spin_min_sigma.value())
                )
            )
            btn_layout.addWidget(btn)

        # Metric buttons
        add_metric_button("redchi(MCD only sort)", "redchi")
        add_metric_button("bic(MCD only sort)", "bic")
        add_metric_button("rms", "residual_rms")
        add_metric_button("combo", "combo")
        # TODO check and make sure that these and other options from this page work. 

        # SpinBox for plot number
        btn_layout.addWidget(QLabel("Plot n #"))
        self.spin_plot_n = QSpinBox()
        self.spin_plot_n.setRange(0, 100)
        self.spin_plot_n.setValue(10)
        self.spin_plot_n.valueChanged.connect(lambda value: print(f"Plot n number set to {value}"))
        btn_layout.addWidget(self.spin_plot_n)

        btn_layout.addWidget(QLabel("Filter Gaussian count start"))
        self.spin_gc_start = QSpinBox()
        self.spin_gc_start.setRange(0, 10)
        self.spin_gc_start.setValue(0)
        btn_layout.addWidget(self.spin_gc_start)

        btn_layout.addWidget(QLabel("Filter Gaussian count end"))
        self.spin_gc_end = QSpinBox()
        self.spin_gc_end.setRange(0, 10)
        self.spin_gc_end.setValue(10)
        btn_layout.addWidget(self.spin_gc_end)

        # --- Sigma spin boxes ---
        avg_sigma = 350.0
        min_sigma_default = avg_sigma / 10  # 35.0
        max_sigma_default = avg_sigma * 10  # 3500.0

        btn_layout.addWidget(QLabel("Min sigma"))
        self.spin_min_sigma = QDoubleSpinBox()
        self.spin_min_sigma.setDecimals(6)
        self.spin_min_sigma.setRange(-1e308, 1e308)
        self.spin_min_sigma.setSingleStep(20)
        self.spin_min_sigma.setValue(min_sigma_default)
        btn_layout.addWidget(self.spin_min_sigma)

        btn_layout.addWidget(QLabel("Max sigma"))
        self.spin_max_sigma = QDoubleSpinBox()
        self.spin_max_sigma.setDecimals(6)
        self.spin_max_sigma.setRange(-1e308, 1e308)
        self.spin_max_sigma.setSingleStep(20)
        self.spin_max_sigma.setValue(max_sigma_default)
        btn_layout.addWidget(self.spin_max_sigma)

        #custom sci notation spinbox class
        class _SciSpinBox(QDoubleSpinBox):
            def textFromValue(self, value: float) -> str:
                return f"{value:.3e}"  # adjust precision

            def valueFromText(self, text: str) -> float:
                try:
                    return float(text.replace("E", "e"))
                except Exception:
                    return 0.0


        btn_layout.addWidget(QLabel("Min amplitude"))
        self.spin_min_amp = _SciSpinBox()
        self.spin_min_amp.setDecimals(18)
        self.spin_min_amp.setRange(-1e308, 1e308)
        self.spin_min_amp.setSingleStep(9.5e-15)
        self.spin_min_amp.setValue(0)
        btn_layout.addWidget(self.spin_min_amp)

        btn_layout.addWidget(QLabel("Max amplitude"))
        self.spin_max_amp = _SciSpinBox()
        self.spin_max_amp.setDecimals(18)
        self.spin_max_amp.setRange(-1e308, 1e308)
        self.spin_max_amp.setSingleStep(9.5e-15)
        self.spin_max_amp.setValue(3000)
        btn_layout.addWidget(self.spin_max_amp)

        #back button
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset)
        btn_layout.addWidget(reset_btn)


    def reset(self):
        mcd_df, fc = self.df_fc
        self.close()
        daf.fit_models(mcd_df, fc, processes=mp.cpu_count())

class MatplotlibGallery(QWidget):
    def __init__(self, figures=None, parent=None):
        super().__init__(parent)
        self.figures = list(figures or [])

        # Main layout
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        root_layout.addWidget(self.scroll)

        # Container inside scroll
        self.container = QWidget()
        self.layout = QVBoxLayout(self.container)
        self.layout.setSpacing(12)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.scroll.setWidget(self.container)

        self.update_gallery()

    def set_figures(self, figures):
        """Replace the figures and refresh the gallery."""
        self.figures = list(figures or [])
        self.update_gallery()

    def update_gallery(self):
        """Clear and rebuild the gallery from current figures."""
        # Remove existing widgets
        while self.layout.count():
            item = self.layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        # Add figures as fixed-size canvases with toolbars
        for fig in self.figures:
            canvas = FigureCanvas(fig)
            dpi = fig.get_dpi()
            w_px = int(fig.get_figwidth() * dpi)
            h_px = int(fig.get_figheight() * dpi)
            canvas.setMinimumSize(w_px, h_px)
            canvas.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

            toolbar = NavigationToolbar(canvas, self.container)
            self.layout.addWidget(toolbar)
            self.layout.addWidget(canvas)
            canvas.draw()

        self.layout.addStretch(1)

class guessWindow(QDialog):
    def __init__(self, x, y_abs, y_mcd, fc, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("peak confirmation")

        self.pa_inp_list = []

        # keep references
        self.x = np.asarray(x, dtype=float)
        self.y_abs = np.asarray(y_abs, dtype=float)
        self.y_mcd = np.asarray(y_mcd, dtype=float)
        self.fc = fc

        self.resize(1400, 700)

        layout = QVBoxLayout(self)
        upper_row = QHBoxLayout()
        layout.addLayout(upper_row)

        self._plot_area = QVBoxLayout()
        upper_row.addLayout(self._plot_area, 3)

        self._editor_group = QGroupBox("FitConfig")
        self._form = QFormLayout(self._editor_group)
        upper_row.addWidget(self._editor_group, 2)

        self._fc_widgets: Dict[str, QWidget] = {}
        self._build_fc_editor()

        # --- Add pc, pa, ps input row ---
        row = QHBoxLayout()
        self._pc = QLineEdit(); self._pc.setPlaceholderText("pc")
        self._pa = QLineEdit(); self._pa.setPlaceholderText("pa")
        self._ps = QLineEdit(); self._ps.setPlaceholderText("ps")
        btn = QPushButton("Input")
        btn.clicked.connect(self._add_pa_input)
        for w in (self._pc, self._pa, self._ps, btn):
            row.addWidget(w)
        self._form.addRow("pc, pa, ps:", row)

        # --- merge_user_input checkbox (NEW) ---
        self._merge_cb = QCheckBox("merge user input")
        self._merge_cb.setChecked(False)
        self._form.addRow(self._merge_cb)

        # Update button
        self._btn_update = QPushButton("Update")
        self._btn_update.clicked.connect(self._on_update_clicked)
        self._form.addRow(self._btn_update)


        # file button for comp data
        self._btn_load = QPushButton("open comp data")
        self._btn_load.clicked.connect(self._peak_file_open)
        self._form.addRow(self._btn_load)

        # Yes/No bottom buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        QTimer.singleShot(0, self.update)
    def get_peak_centers_fig(self, peak_centers) -> Figure:
        y_at_centers = np.interp(peak_centers, self.x, self.y_mcd)

        fig = Figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(211)

        ax.plot(self.x, self.y_mcd, '-', lw=1, alpha=0.7, label='mcd')
        ax.scatter(peak_centers, y_at_centers, s=36, zorder=3, label='peak centers')
        for cx in peak_centers:
            ax.axvline(cx, ls='--', lw=0.8, alpha=0.5)

        # total fits
        def total_fits(m, n, k):
            return sum((2 ** i) * comb(m, i) for i in range(n, min(k, m + 1)))

        tfit = total_fits(len(peak_centers), self.fc.MIN_GC, self.fc.MAX_GC + 1)

        ax.text(0.02, 0.98, f"total fits = {int(tfit)}",
                transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round", alpha=0.25))

        ax.set_xlabel('x')
        ax.set_ylabel('signal')
        ax.legend()
        fig.tight_layout()

        ax2 = fig.add_subplot(212)  # Bottom subplot
        ax2.plot(self.x, self.y_abs, 'g-', lw=1, alpha=0.6, label='abs')
        ax2.set_xlabel('x')
        ax2.set_ylabel('abs intensity')
        ax2.scatter(peak_centers, y_at_centers, s=36, zorder=3, label='peak centers')
        for cx in peak_centers:
            ax2.axvline(cx, ls='--', lw=0.8, alpha=0.5)
        ax2.legend()

        fig.tight_layout()
        return fig

    def _add_pa_input(self):
        try:
            pc = float(self._pc.text().strip())
            pa = float(self._pa.text().strip())
            ps = float(self._ps.text().strip())
        except ValueError:
            return
        self.pa_inp_list.append((pc, pa, ps))
        self._pc.clear()
        self._pa.clear()
        self._ps.clear()

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
        self._mpl_cid = canvas.mpl_connect('button_press_event', self._on_click)

        canvas.draw_idle()

    def get_peaks(self):
        return self.fc, self._pc, self._pa, self._ps, self.pa_inp_list
    def set_peaks(self,fc, pc, pa, ps, pa_inp_list):
        self.fc = fc
        self._pc = pc
        self._pa = pa
        self._ps = ps
        self._pa_inp_list = pa_inp_list

    def update(self):
        if self._merge_cb.isChecked():
            pci, pai, psi = zip(*self.pa_inp_list)
            pci, pai, psi = list(pci), list(pai), list(psi)

            pa, pc, ps = self._guess_on_all_data(self.x, self.y_abs, self.y_mcd)
            pa.extend(pai)
            pc.extend(pci)
            ps.extend(psi)
        elif not self._merge_cb.isChecked():
            if len(self.pa_inp_list) > 0:
                pc, pa, ps = zip(*self.pa_inp_list)
                pc, pa, ps = list(pc), list(pa), list(ps)
            else:
                pa, pc, ps = self._guess_on_all_data(self.x, self.y_abs, self.y_mcd)

        # Build and embed the figure
        fig = self.get_peak_centers_fig(pc)
        self._set_figure_in_ui(fig)

        # cache
        self.pa = pa
        self.pc = pc
        self.ps = ps


    # --------------- FitConfig editor ---------------

    def _build_fc_editor(self) -> None:
        self._clear_form_layout()
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

    def _clear_form_layout(self) -> None:
        while self._form.rowCount():
            self._form.removeRow(0)

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
            if not(len(self.pc) == len(self.pa) == len(self.ps)):
                print("Error in peak array lengths: they are not the same.")
                return -1
            self.pa = np.delete(self.pa, i)
            self.ps = np.delete(self.ps, i)
            self.pc = np.delete(self.pc, i)
            return i
        return -1

    def _on_click(self, event) -> None:
        """
        Matplotlib click callback. Left-click removes the nearest peak within dx.
        """
        if event.inaxes is None or event.xdata is None:
            print("Error in click callback.")
            return
        # Left click only; change to event.button == 3 for right-click behavior
        if getattr(event, "button", 1) != 1:
            return

        removed = self._remove_peak_near(float(event.xdata), 75)
        if removed >= 0:
            self._redraw_current()

    def _redraw_current(self) -> None:
        pc = getattr(self, "pc", None)

        if pc is not None and len(pc) > 0:
            centers = list(pc)
        else:
            centers = []

        # Build a fresh figure using the helper
        fig = self.get_peak_centers_fig(centers)

        # Embed it in the UI using the helper
        self._set_figure_in_ui(fig)

    def _peak_file_open(self):
        name, _ = QFileDialog.getOpenFileName(self, 'Open File')
        file = open(name, 'r')
        with file:
            peaks = file.readlines()
            for row in peaks:
                peak_values = []
                for peak_value in row.split():
                    peak_values.append(float(peak_value.split("=")[1]))
                self.pa_inp_list.append((peak_values[0], peak_values[1], peak_values[2]))
            self.update()
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
    def get_fc(self):
        return self.fc