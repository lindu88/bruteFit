import csv
import os
from dateutil import rrule, tz  # noqa: F401
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QMainWindow, QScrollArea, QSizePolicy, QLineEdit, QFileDialog, QApplication
)
import matplotlib
from scipy.optimize import curve_fit
from scipy.signal import peak_prominences, savgol_filter, find_peaks


from dataclasses import dataclass, fields
from typing import Optional, Dict

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QDialogButtonBox,
    QPushButton, QDoubleSpinBox, QSpinBox, QWidget, QLabel, QMessageBox, QComboBox
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
import multiprocessing as mp
matplotlib.use("Agg" if os.environ.get("BRUTEFIT_HEADLESS") else "QtAgg")

from .transition_matching import (
    PeakGuess,
    TransitionGuess,
    count_transition_statuses,
    estimate_total_fit_count,
    pair_peak_guesses,
    remove_transition_peak,
)
from .mcd_guessing import guess_mcd_peaks
from .gaussianModels import (
    component_amplitude_from_peak_height,
    component_peak_height,
    stable_gaussian_derivative_sigma,
    stable_gaussian_sigma,
)

_FIT_CONFIG_INTRO = (
    "<b>How to use this panel</b><br>"
    "Settings are grouped by workflow stage. <b>Guess Generation</b> controls how peaks are detected, "
    "cleaned, and paired when you click <b>Regenerate Preview</b>. <b>Post-Guess Fitting</b> controls "
    "the brute-force search and how far accepted peaks are allowed to move during fitting. "
    "The preview overlays faint raw traces and bold S-G smoothed traces so you can tune smoothing by eye. "
    "Hover any field name for details."
)


def _debug_guesses_enabled() -> bool:
    return os.environ.get("BRUTEFIT_DEBUG_GUESSES", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _debug_guess_print(*args) -> None:
    if _debug_guesses_enabled():
        print(*args)


@dataclass(frozen=True)
class _SigmaEstimate:
    sigma: float
    method: str
    ratios_used: tuple[float, ...]


class _ScientificSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that displays and accepts scientific notation cleanly."""

    def __init__(self, parent: Optional[QWidget] = None, sigfigs: int = 6):
        super().__init__(parent)
        self._sigfigs = max(1, int(sigfigs))
        self.setDecimals(18)
        self.setRange(-1e308, 1e308)

    def validate(self, text, pos):
        stripped = text.strip()
        if stripped in {"", "-", "+", ".", "-.", "+."}:
            return (QValidator.State.Intermediate, text, pos)
        try:
            float(stripped.replace("E", "e"))
        except ValueError:
            return (QValidator.State.Invalid, text, pos)
        return (QValidator.State.Acceptable, text, pos)

    def valueFromText(self, text: str) -> float:
        try:
            return float(text.replace("E", "e"))
        except ValueError:
            return 0.0

    def textFromValue(self, value: float) -> str:
        return f"{value:.{self._sigfigs}g}"
"""
This is the result window after all the fitting-which is a gallery of matplotlib figures that can be sorted on.

The constructor parameters are the results of the fitting as defined in fit_models()(bfresult) and a pair (mcd dataframe, fitconfig object)

The constructor accepts df_fc for the feature of being able to go back for re-fitting with the current fit params saved. 
"""
class MainResultWindow(QMainWindow):
    def __init__(self, bfResult=None, df_fc = None):
        super().__init__()
        self.plot_n = 10
        self.bfResult = bfResult
        self.setWindowTitle("BF Results")
        self.resize(1100, 700)

        self.df_fc = df_fc

        central = QWidget(self)
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)

        figures = self.bfResult.get_plot_figs(self.plot_n)
        if len(figures) == 0:
            raise RuntimeError("No results to plot")
        self.gallery = MatplotlibGallery(figures=figures)
        main_layout.addWidget(self.gallery, stretch=1)

        btn_layout = QVBoxLayout()
        btn_layout.setAlignment(Qt.AlignTop)  # keep buttons at the top

        self._build_controls(btn_layout)

        btn_layout.addStretch(1)

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
                return f"{value:.5e}"  # adjust precision

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
        from . import dataFitting as daf

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
    def __init__(self, x, y_abs, y_mcd, fc, parent: Optional[QWidget] = None, abs_noise=None, mcd_noise=None):
        super().__init__(parent)

        self.setWindowTitle("peak confirmation")
        self.transitions: list[TransitionGuess] = []
        self.manual_peaks: list[PeakGuess] = []
        self._hover_targets = []
        self._preview_dirty = False
        self._last_preview_source_state = None
        self._suspend_dirty_tracking = False

        # keep references
        self.x = np.asarray(x, dtype=float)
        self.y_abs = np.asarray(y_abs, dtype=float)
        self.y_mcd = np.asarray(y_mcd, dtype=float)
        self.abs_noise = None if abs_noise is None else np.asarray(abs_noise, dtype=float)
        self.mcd_noise = None if mcd_noise is None else np.asarray(mcd_noise, dtype=float)
        self.fc = fc

        self.resize(1400, 700)

        layout = QVBoxLayout(self)
        upper_row = QHBoxLayout()
        layout.addLayout(upper_row)

        self._plot_area = QVBoxLayout()
        upper_row.addLayout(self._plot_area, 3)

        self._editor_group = QGroupBox("FitConfig")
        self._form = QFormLayout(self._editor_group)
        self._editor_scroll = QScrollArea()
        self._editor_scroll.setWidgetResizable(True)
        self._editor_scroll.setWidget(self._editor_group)
        upper_row.addWidget(self._editor_scroll, 2)

        self._fc_widgets: Dict[str, QWidget] = {}
        self._build_fc_editor()

        self._source_mode = QComboBox()
        self._source_mode.addItem("Automatic peaks only", "auto")
        self._source_mode.addItem("Manual peaks only", "manual")
        self._source_mode.addItem("Automatic + manual peaks", "merge")
        self._source_mode.setToolTip(
            "Choose whether the preview uses automatic guesses, manual peaks only, "
            "or a merged set."
        )
        self._source_mode.currentIndexChanged.connect(self._mark_preview_dirty)
        self._form.addRow("Peak source:", self._source_mode)

        self._manual_source = QComboBox()
        self._manual_source.addItem("ABS", "abs")
        self._manual_source.addItem("MCD", "mcd")
        self._manual_source.setToolTip(
            "Choose whether the manual peak should be treated as an absorption peak or an MCD peak "
            "before cross-modal pairing."
        )
        self._form.addRow("Manual peak source:", self._manual_source)

        # --- Add source-aware pc, pa, ps input row ---
        row = QHBoxLayout()
        self._pc = QLineEdit(); self._pc.setPlaceholderText("peak center")
        self._pc.setToolTip("Manual peak center (pc).")
        self._pa = QLineEdit(); self._pa.setPlaceholderText("peak height")
        self._pa.setToolTip(
            "Manual peak height (pa). bruteFit converts this to the internal model amplitude."
        )
        self._ps = QLineEdit(); self._ps.setPlaceholderText("peak sigma")
        self._ps.setToolTip("Manual peak sigma / width (ps).")
        btn = QPushButton("Input")
        btn.clicked.connect(self._add_pa_input)
        btn.setToolTip(
            "Add the manual peak defined by the current source, center, height, and sigma fields."
        )
        for w in (self._pc, self._pa, self._ps, btn):
            row.addWidget(w)
        self._form.addRow("pc, pa(height), ps:", row)

        # Update button
        self._btn_update = QPushButton("Regenerate Preview")
        self._btn_update.clicked.connect(self._on_update_clicked)
        self._btn_update.setToolTip("Apply the current settings and regenerate the guessed peaks.")
        self._form.addRow(self._btn_update)

        self._btn_revert = QPushButton("Revert Pending Changes")
        self._btn_revert.clicked.connect(self._on_revert_pending_clicked)
        self._btn_revert.setToolTip(
            "Discard unapplied setting changes and keep the currently displayed preview."
        )
        self._btn_revert.setEnabled(False)
        self._form.addRow(self._btn_revert)

        self._preview_status = QLabel()
        self._preview_status.setWordWrap(True)
        self._form.addRow(self._preview_status)

        # file button for comp data
        self._btn_load = QPushButton("Bulk Load Peaks CSV")
        self._btn_load.clicked.connect(self._peak_file_open)
        self._btn_load.setToolTip(
            "Replace the current manual peak list with peaks from a CSV file containing "
            "columns source, pc, ps, and either height or amplitude."
        )
        self._form.addRow(self._btn_load)

        # Yes/No bottom buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No
        )
        buttons.accepted.connect(self.on_yes_clicked)
        buttons.rejected.connect(self.reject)
        self._accept_button = buttons.button(QDialogButtonBox.StandardButton.Yes)
        self._accept_button.setText("Accept Current Peaks")
        self._reject_button = buttons.button(QDialogButtonBox.StandardButton.No)
        self._reject_button.setText("Cancel")
        layout.addWidget(buttons)

        cached_transitions, cached_manual_peaks, cached_source_mode = self.fc.get_current_peaks()
        self.manual_peaks = list(cached_manual_peaks or [])
        cached_index = self._source_mode.findData(cached_source_mode)
        if cached_index >= 0:
            self._source_mode.setCurrentIndex(cached_index)

        QTimer.singleShot(0, self.update)

    def on_yes_clicked(self):
        if self._preview_dirty:
            return
        self.fc.set_current_peaks(self.transitions, self.manual_peaks, self._get_source_mode())
        self.accept()

    def _smoothed_signal(self, y) -> np.ndarray:
        return savgol_filter(
            np.asarray(y, dtype=float),
            window_length=self.fc.WINDOW_LENGTH,
            polyorder=self.fc.POLYORDER,
        )

    def _noise_mask(self, peaks: list[PeakGuess]) -> tuple[np.ndarray, str]:
        x = np.asarray(self.x, dtype=float)
        mask = np.ones_like(x, dtype=bool)
        dx = self._median_dx(x)

        for peak in peaks:
            sigma = max(dx, abs(float(peak.sigma)))
            radius = max(3.0 * sigma, 5.0 * dx)
            mask &= np.abs(x - float(peak.center)) > radius

        min_points = max(10, len(x) // 8)
        if int(np.count_nonzero(mask)) >= min_points:
            return mask, "quiet"

        edge_count = max(5, len(x) // 10)
        edge_mask = np.zeros_like(mask)
        edge_mask[:edge_count] = True
        edge_mask[-edge_count:] = True
        if int(np.count_nonzero(edge_mask)) >= min_points:
            return edge_mask, "edges"

        return np.ones_like(mask, dtype=bool), "global"

    def _estimate_noise_sigma(
        self,
        y_raw,
        y_smoothed,
        peaks: list[PeakGuess],
    ) -> tuple[float, int, str]:
        residual = np.asarray(y_raw, dtype=float) - np.asarray(y_smoothed, dtype=float)
        mask, source = self._noise_mask(peaks)
        samples = residual[mask]
        samples = samples[np.isfinite(samples)]
        if samples.size < 2:
            fallback = residual[np.isfinite(residual)]
            if fallback.size < 2:
                return float("nan"), 0, source
            samples = fallback
            source = "global"

        centered = samples - float(np.nanmedian(samples))
        return float(np.nanstd(centered)), int(centered.size), source

    def _noise_visual(self, source: str, y_raw, y_smoothed, peaks: list[PeakGuess]):
        supplied = self.abs_noise if source == "abs" else self.mcd_noise
        if supplied is not None:
            noise = np.asarray(supplied, dtype=float)
            if noise.shape == np.asarray(y_raw).shape:
                finite = noise[np.isfinite(noise)]
                finite = np.abs(finite)
                if finite.size:
                    return np.abs(noise), float(np.nanmedian(finite)), "propagated"

        sigma, _, sigma_source = self._estimate_noise_sigma(y_raw, y_smoothed, peaks)
        if np.isfinite(sigma) and sigma > 0:
            return np.full_like(np.asarray(y_raw, dtype=float), float(sigma)), float(sigma), sigma_source
        return None, float("nan"), sigma_source

    @staticmethod
    def _format_sigma_multiple(value: float, sigma: float) -> str:
        if not np.isfinite(sigma) or sigma <= 0:
            return "n/a"
        return f"{float(value) / float(sigma):.2g}σ"

    def get_peak_centers_fig(self, transitions: list[TransitionGuess]) -> Figure:
        self._hover_targets = []

        fig = Figure(figsize=(8, 8), dpi=100)
        ax_mcd = fig.add_subplot(211)
        ax_abs = fig.add_subplot(212, sharex=ax_mcd)
        self._mcd_axes = ax_mcd
        self._abs_axes = ax_abs

        y_mcd_smoothed = self._smoothed_signal(self.y_mcd)
        y_abs_smoothed = self._smoothed_signal(self.y_abs)
        abs_peaks = [transition.abs_peak for transition in transitions if transition.abs_peak is not None]
        mcd_peaks = [transition.mcd_peak for transition in transitions if transition.mcd_peak is not None]
        abs_noise_band, abs_noise_sigma, abs_noise_source = self._noise_visual("abs", self.y_abs, y_abs_smoothed, abs_peaks)
        mcd_noise_band, mcd_noise_sigma, mcd_noise_source = self._noise_visual("mcd", self.y_mcd, y_mcd_smoothed, mcd_peaks)

        if mcd_noise_band is not None:
            ax_mcd.fill_between(
                self.x,
                self.y_mcd - mcd_noise_band,
                self.y_mcd + mcd_noise_band,
                color="#808080",
                alpha=0.10,
                linewidth=0,
                label="MCD raw ±1σ noise",
                zorder=0,
            )
        ax_mcd.plot(self.x, y_mcd_smoothed, "-", lw=1.5, alpha=0.95, color="#1f4f7f", label="MCD smoothed", zorder=2)
        ax_mcd.plot(
            self.x,
            self.y_mcd,
            "--",
            lw=1.0,
            alpha=0.45,
            color="#000000",
            label="MCD raw",
            zorder=3,
            dashes=(3, 2),
        )
        if abs_noise_band is not None:
            ax_abs.fill_between(
                self.x,
                self.y_abs - abs_noise_band,
                self.y_abs + abs_noise_band,
                color="#808080",
                alpha=0.10,
                linewidth=0,
                label="ABS raw ±1σ noise",
                zorder=0,
            )
        ax_abs.plot(self.x, y_abs_smoothed, "-", lw=1.5, alpha=0.95, color="#356c3d", label="ABS smoothed", zorder=2)
        ax_abs.plot(
            self.x,
            self.y_abs,
            "--",
            lw=1.0,
            alpha=0.45,
            color="#000000",
            label="ABS raw",
            zorder=3,
            dashes=(3, 2),
        )
        self._plot_initial_guess_overlay(ax_mcd, transitions, source="mcd")
        self._plot_initial_guess_overlay(ax_abs, transitions, source="abs")

        status_colors = {
            "paired": "#1f77b4",
            "abs_only": "#2ca02c",
            "mcd_only": "#d62728",
        }
        status_markers = {
            "paired": "o",
            "abs_only": "s",
            "mcd_only": "^",
        }

        total_fits = estimate_total_fit_count(transitions, self.fc.MIN_GC, self.fc.MAX_GC)
        summary = self._transition_count_summary(transitions)

        ax_mcd.text(
            0.02,
            0.98,
            f"total fits = {int(total_fits)}\n{summary}",
            transform=ax_mcd.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", alpha=0.25),
        )
        ax_mcd.text(
            0.98,
            0.02,
            "Hover markers for source, status, center, sigma, height, amplitude, and match distance.\n"
            "Raw traces are faint; S-G smoothed traces are bold.",
            transform=ax_mcd.transAxes,
            va="bottom",
            ha="right",
            fontsize=8,
            bbox=dict(boxstyle="round", alpha=0.18),
        )

        min_abs_height = float(self.fc.MIN_ABSOLUTE_PEAK_HEIGHT)
        ax_abs.text(
            0.02,
            0.98,
            "\n".join(
                [
                    "Smoothing preview:",
                    "raw = faint black, smoothed = bold color, grey band = raw ±1σ noise",
                    (
                        f"ABS noise sigma ({abs_noise_source} residual): "
                        f"{self._format_hover_value(abs_noise_sigma)}; "
                        f"Min Height = {self._format_sigma_multiple(min_abs_height, abs_noise_sigma)}"
                    ),
                    (
                        f"MCD noise sigma ({mcd_noise_source} residual): "
                        f"{self._format_hover_value(mcd_noise_sigma)}; "
                        f"Min Height = {self._format_sigma_multiple(min_abs_height, mcd_noise_sigma)}"
                    ),
                ]
            ),
            transform=ax_abs.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round", alpha=0.18),
        )

        grouped_points: dict[tuple[str, str], dict[str, list]] = {}
        for transition in transitions:
            for source, ax, y_values in (
                ("mcd", ax_mcd, y_mcd_smoothed),
                ("abs", ax_abs, y_abs_smoothed),
            ):
                peak = transition.mcd_peak if source == "mcd" else transition.abs_peak
                if peak is None:
                    continue
                center = float(peak.center)
                y_value = float(np.interp(center, self.x, y_values))
                status = transition.status
                marker = status_markers[status]
                color = status_colors[status]
                label = f"{status.replace('_', ' ').title()} {source.upper()}"
                ax.axvline(center, ls="--", lw=0.8, alpha=0.35, color=color)
                key = (source, status)
                group = grouped_points.setdefault(
                    key,
                    {
                        "ax": ax,
                        "marker": marker,
                        "color": color,
                        "label": label,
                        "x": [],
                        "y": [],
                        "text": [],
                    },
                )
                group["x"].append(center)
                group["y"].append(y_value)
                group["text"].append(self._format_transition_hover_text(transition, peak))

        for group in grouped_points.values():
            scatter = group["ax"].scatter(
                group["x"],
                group["y"],
                s=48,
                zorder=3,
                marker=group["marker"],
                color=group["color"],
                label=group["label"],
            )
            self._hover_targets.extend(
                self._make_hover_targets(
                    group["ax"],
                    scatter,
                    group["text"],
                    group["x"],
                    group["y"],
                )
            )

        ax_mcd.set_xlabel("x")
        ax_mcd.set_ylabel("signal")
        ax_mcd.legend()

        ax_abs.set_xlabel("x")
        ax_abs.set_ylabel("abs intensity")
        ax_abs.legend()

        fig.tight_layout()
        return fig

    def _add_pa_input(self):
        try:
            pc = float(self._pc.text().strip())
            peak_height = float(self._pa.text().strip())
            ps = float(self._ps.text().strip())
        except ValueError:
            return
        amplitude = component_amplitude_from_peak_height(peak_height, pc, ps)
        peak = PeakGuess(
            source=str(self._manual_source.currentData() or "abs"),
            center=pc,
            amplitude=amplitude,
            sigma=ps,
            height=peak_height,
            origin="manual",
        )
        self.manual_peaks.append(peak)
        self._pc.clear()
        self._pa.clear()
        self._ps.clear()
        self._mark_preview_dirty()

    @staticmethod
    def _format_hover_value(value: float) -> str:
        value = float(value)
        magnitude = abs(value)
        if magnitude != 0 and (magnitude >= 1e4 or magnitude < 1e-3):
            return f"{value:.3e}"
        return f"{value:.6g}"

    @staticmethod
    def _format_transition_hover_text(transition: TransitionGuess, peak: PeakGuess) -> str:
        source_label = "ABS" if peak.source == "abs" else "MCD"
        match_distance = (
            "n/a"
            if transition.match_distance is None
            else guessWindow._format_hover_value(float(transition.match_distance))
        )
        peak_height = (
            peak.height
            if peak.height is not None
            else component_peak_height(peak.amplitude, peak.center, peak.sigma, label=peak.label)
        )
        dominant_line = ""
        if peak.source == "mcd" and peak.label in {"A", "B"}:
            dominant_line = f"dominant term: {peak.label}\n"
        return (
            f"{transition.transition_id} ({transition.status.replace('_', ' ')})\n"
            f"source: {source_label} [{peak.origin}]\n"
            f"{dominant_line}"
            f"center: {guessWindow._format_hover_value(float(peak.center))}\n"
            f"sigma: {guessWindow._format_hover_value(float(peak.sigma))}\n"
            f"height: {guessWindow._format_hover_value(float(peak_height))}\n"
            f"amplitude: {guessWindow._format_hover_value(float(peak.amplitude))}\n"
            f"pair distance: {match_distance}"
        )

    def _evaluate_initial_component(self, peak: PeakGuess, source: str) -> np.ndarray:
        if source == "abs":
            return stable_gaussian_sigma(self.x, peak.amplitude, peak.center, peak.sigma)
        if peak.label == "A":
            return stable_gaussian_derivative_sigma(self.x, peak.amplitude, peak.center, peak.sigma)
        return stable_gaussian_sigma(self.x, peak.amplitude, peak.center, peak.sigma)

    def _plot_initial_guess_overlay(self, ax, transitions: list[TransitionGuess], source: str) -> None:
        fill_label_added = False
        curves = []
        if source == "abs":
            fill_color = "#7ba66d"
            sum_color = "#4b6f42"
        else:
            fill_color = "#5f88c5"
            sum_color = "#2f4f7f"

        for transition in transitions:
            peak = transition.abs_peak if source == "abs" else transition.mcd_peak
            if peak is None:
                continue
            curve = self._evaluate_initial_component(peak, source)
            curves.append(curve)
            ax.fill_between(
                self.x,
                0,
                curve,
                color=fill_color,
                alpha=0.07,
                linewidth=0,
                label="Initial guess components" if not fill_label_added else "_nolegend_",
                zorder=0,
            )
            fill_label_added = True

        if curves:
            summed_curve = np.sum(np.vstack(curves), axis=0)
            ax.plot(
                self.x,
                summed_curve,
                color=sum_color,
                lw=1.0,
                alpha=0.45,
                ls="--",
                label="Initial guess sum",
                zorder=1,
            )

    @staticmethod
    def _make_hover_targets(ax, scatter, hover_text, x_values, y_values):
        annotation = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="white", alpha=0.9),
            arrowprops=dict(arrowstyle="->", alpha=0.5),
            fontsize=8,
        )
        annotation.set_visible(False)
        return [{
            "ax": ax,
            "scatter": scatter,
            "annotation": annotation,
            "text": hover_text,
            "x": np.asarray(x_values, dtype=float),
            "y": np.asarray(y_values, dtype=float),
        }]

    def _set_figure_in_ui(self, fig: Figure) -> None:
        """Embed a Matplotlib Figure into the left pane."""
        # remove previous widgets if any
        for i in reversed(range(self._plot_area.count())):
            w = self._plot_area.itemAt(i).widget()
            if w is not None:
                w.setParent(None)

        self.fig = fig
        canvas = FigureCanvas(self.fig)
        self._canvas = canvas
        toolbar = NavigationToolbar(canvas, self)
        self._plot_area.addWidget(toolbar)
        self._plot_area.addWidget(canvas)

        #hook clicks and set tolerance
        self._mpl_cid = canvas.mpl_connect('button_press_event', self._on_click)
        self._hover_cid = canvas.mpl_connect('motion_notify_event', self._on_hover)
        self._hover_leave_cid = canvas.mpl_connect('figure_leave_event', self._clear_hover_annotations)

        canvas.draw_idle()

    def get_peaks(self):
        return self.fc, self._pc, self._pa, self._ps, self.manual_peaks
    def save_peaks(self):
        self.fc.set_current_peaks(self.transitions, self.manual_peaks, self._get_source_mode())

    def _get_source_mode(self) -> str:
        return str(self._source_mode.currentData() or "auto")

    def _capture_preview_source_state(self) -> dict:
        widget_values = {}
        for key, widget in self._fc_widgets.items():
            if isinstance(widget, QDoubleSpinBox):
                widget_values[key] = float(widget.value())
            elif isinstance(widget, QSpinBox):
                widget_values[key] = int(widget.value())
        return {
            "widget_values": widget_values,
            "source_mode_index": int(self._source_mode.currentIndex()),
            "manual_source_index": int(self._manual_source.currentIndex()),
            "manual_peaks": list(self.manual_peaks),
        }

    def _set_preview_status_current(self) -> None:
        mode_label = self._source_mode.currentText()
        summary = self._transition_count_summary(self.transitions)
        self._preview_status.setText(f"Preview is current: {summary} shown using {mode_label}.")

    def _mark_preview_fresh(self) -> None:
        self._preview_dirty = False
        self._last_preview_source_state = self._capture_preview_source_state()
        if hasattr(self, "_accept_button"):
            self._accept_button.setEnabled(True)
        if hasattr(self, "_btn_revert"):
            self._btn_revert.setEnabled(False)
        self._set_preview_status_current()

    def _mark_preview_dirty(self, *_args) -> None:
        if self._suspend_dirty_tracking:
            return
        self._preview_dirty = True
        # The plotted peaks are now stale relative to the edited controls. Disable
        # accept and keep a revert path so final fitting cannot accidentally use a
        # preview that no longer matches the visible FitConfig/manual inputs.
        if hasattr(self, "_accept_button"):
            self._accept_button.setEnabled(False)
        if hasattr(self, "_btn_revert"):
            self._btn_revert.setEnabled(True)
        if hasattr(self, "_preview_status"):
            self._preview_status.setText(
                "Preview out of date — click Regenerate Preview to apply changes, "
                "or Revert Pending Changes to keep the current preview."
            )

    def _restore_preview_source_state(self, state: dict) -> None:
        self._suspend_dirty_tracking = True
        try:
            self._source_mode.setCurrentIndex(int(state.get("source_mode_index", 0)))
            self._manual_source.setCurrentIndex(int(state.get("manual_source_index", 0)))
            for key, value in state.get("widget_values", {}).items():
                widget = self._fc_widgets.get(key)
                if isinstance(widget, QDoubleSpinBox):
                    widget.setValue(float(value))
                elif isinstance(widget, QSpinBox):
                    widget.setValue(int(value))
            self.manual_peaks = list(state.get("manual_peaks", []))
            self._pc.clear()
            self._pa.clear()
            self._ps.clear()
        finally:
            self._suspend_dirty_tracking = False

    #overide
    def update(self):
        transitions = self._guess_on_all_data(self.x, self.y_abs, self.y_mcd)
        fig = self.get_peak_centers_fig(transitions)
        self._set_figure_in_ui(fig)

        self.transitions = list(transitions)
        self._mark_preview_fresh()


    # --------------- FitConfig editor ---------------

    def _build_fc_editor(self) -> None:
        self._clear_form_layout()
        self._fc_widgets.clear()

        intro = QLabel(_FIT_CONFIG_INTRO)
        intro.setWordWrap(True)
        self._form.addRow(intro)

        current_section = None
        current_group = None
        for fc_field in fields(self.fc):
            key = fc_field.name
            if key.startswith('_'):
                continue

            metadata = dict(fc_field.metadata or {})
            section = metadata.get("section", "Other")
            group = metadata.get("group", "Other")
            if section != current_section:
                if current_section is not None:
                    spacer = QLabel("")
                    self._form.addRow(spacer)
                current_section = section
                current_group = None
                section_header = QLabel(f"<b>{section}</b>")
                section_header.setWordWrap(True)
                self._form.addRow(section_header)
            if group != current_group:
                current_group = group
                header = QLabel(f"<i>{group}</i>")
                header.setWordWrap(True)
                self._form.addRow(header)

            val = getattr(self.fc, key)
            help_text = metadata.get("help", key)
            label = QLabel(metadata.get("label", key))
            label.setToolTip(help_text)
            label.setWordWrap(True)

            if isinstance(val, float):
                sb = _ScientificSpinBox(sigfigs=6)
                sb.setSingleStep(0.01)
                sb.setValue(float(val))
                sb.setToolTip(help_text)
                sb.valueChanged.connect(self._mark_preview_dirty)
                self._fc_widgets[key] = sb
                self._form.addRow(label, sb)
            elif isinstance(val, int):
                sb = QSpinBox()
                sb.setRange(-2_000_000_000, 2_000_000_000)
                sb.setSingleStep(1)
                sb.setValue(int(val))
                sb.setToolTip(help_text)
                sb.valueChanged.connect(self._mark_preview_dirty)
                self._fc_widgets[key] = sb
                self._form.addRow(label, sb)
            else:
                lab = QLabel(str(val))
                lab.setEnabled(False)
                lab.setToolTip(help_text)
                self._fc_widgets[key] = lab
                self._form.addRow(label, lab)

        footer = QLabel(
            "Edit values, hover labels for explanations, then click Regenerate Preview "
            "to apply those changes to the guessed peaks."
        )
        footer.setWordWrap(True)
        self._form.addRow(footer)

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

    def _validate_fc(self) -> Optional[str]:
        if self.fc.WINDOW_LENGTH <= self.fc.POLYORDER:
            return "S-G Window Length must be greater than S-G Polyorder."
        if self.fc.WINDOW_LENGTH % 2 == 0:
            return "S-G Window Length must be odd for Savitzky-Golay smoothing."
        if not (0 < float(self.fc.SIGMA_RATIO_START) < 1):
            return "Sigma Ratio Start must be between 0 and 1."
        if not (0 < float(self.fc.SIGMA_RATIO_END) < 1):
            return "Sigma Ratio End must be between 0 and 1."
        if float(self.fc.SIGMA_RATIO_START) >= float(self.fc.SIGMA_RATIO_END):
            return "Sigma Ratio Start must be less than Sigma Ratio End."
        if float(self.fc.SIGMA_RATIO_STEP) <= 0:
            return "Sigma Ratio Step must be positive."
        if len(self._sigma_ratios()) < 2:
            return (
                "Sigma Ratio Start/End/Step must produce at least two ratios "
                "so sigma averaging has a valid range."
            )
        if self.fc.MIN_GC > self.fc.MAX_GC:
            return "Min Components cannot be greater than Max Components."
        return None

    def _on_update_clicked(self) -> None:
        self._apply_editor_to_fc()
        error = self._validate_fc()
        if error:
            QMessageBox.warning(self, "Invalid FitConfig", error)
            return
        self.update()

    def _on_revert_pending_clicked(self) -> None:
        if self._last_preview_source_state is None:
            return
        self._restore_preview_source_state(self._last_preview_source_state)
        self._preview_dirty = False
        if hasattr(self, "_accept_button"):
            self._accept_button.setEnabled(True)
        self._btn_revert.setEnabled(False)
        self._set_preview_status_current()

    @staticmethod
    def _transition_count_summary(transitions: list[TransitionGuess]) -> str:
        counts = count_transition_statuses(transitions)
        total = sum(counts.values())
        if total == 0:
            return "no transitions"
        return (
            f"{counts['paired']} paired, "
            f"{counts['abs_only']} ABS-only, "
            f"{counts['mcd_only']} MCD-only"
        )

    def _remove_peak_near(self, x_click: float, dx: float, source: str) -> int:
        """
        Remove the single source-specific peak whose center is closest to x_click if within dx.
        Returns 1 if a peak was removed, or 0 if none were removed.
        """
        candidates = []
        for transition in self.transitions:
            peak = transition.mcd_peak if source == "mcd" else transition.abs_peak
            if peak is None:
                continue
            candidates.append((abs(float(peak.center) - x_click), transition.transition_id))

        if not candidates:
            return 0

        distance, transition_id = min(candidates, key=lambda item: item[0])
        if distance > dx:
            return 0

        self.transitions = remove_transition_peak(self.transitions, transition_id, source)
        return 1

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

        source = None
        if hasattr(self, "_mcd_axes") and event.inaxes is self._mcd_axes:
            source = "mcd"
        elif hasattr(self, "_abs_axes") and event.inaxes is self._abs_axes:
            source = "abs"
        if source is None:
            return

        removed = self._remove_peak_near(float(event.xdata), 75, source)
        if removed:
            self._redraw_current()

    def _on_hover(self, event) -> None:
        if event.inaxes is None:
            self._clear_hover_annotations()
            return

        redraw = False
        best_match = None
        best_distance_sq = None
        hit_radius_px = 14.0

        for target in self._hover_targets:
            annotation = target["annotation"]
            if target["ax"] is not event.inaxes:
                if annotation.get_visible():
                    annotation.set_visible(False)
                    redraw = True
                continue

            points_xy = np.column_stack([target["x"], target["y"]])
            if points_xy.size == 0:
                if annotation.get_visible():
                    annotation.set_visible(False)
                    redraw = True
                continue

            display_xy = target["ax"].transData.transform(points_xy)
            dx = display_xy[:, 0] - float(event.x)
            dy = display_xy[:, 1] - float(event.y)
            distance_sq = dx * dx + dy * dy
            idx = int(np.argmin(distance_sq))
            min_distance_sq = float(distance_sq[idx])

            if min_distance_sq <= hit_radius_px ** 2:
                if best_distance_sq is None or min_distance_sq < best_distance_sq:
                    best_distance_sq = min_distance_sq
                    best_match = (target, idx)
            elif annotation.get_visible():
                annotation.set_visible(False)
                redraw = True

        for target in self._hover_targets:
            annotation = target["annotation"]
            should_show = best_match is not None and target is best_match[0]
            if should_show:
                idx = best_match[1]
                annotation.xy = (target["x"][idx], target["y"][idx])
                annotation.set_text(target["text"][idx])
                if not annotation.get_visible():
                    annotation.set_visible(True)
                    redraw = True
            elif annotation.get_visible():
                annotation.set_visible(False)
                redraw = True

        if redraw and hasattr(self, "_canvas"):
            self._canvas.draw_idle()

    def _clear_hover_annotations(self, event=None, redraw: bool = False) -> bool:
        for target in self._hover_targets:
            annotation = target["annotation"]
            if annotation.get_visible():
                annotation.set_visible(False)
                redraw = True
        return redraw

    def _redraw_current(self) -> None:
        fig = self.get_peak_centers_fig(self.transitions)

        # Embed it in the UI using the helper
        self._set_figure_in_ui(fig)
        if not self._preview_dirty:
            self._set_preview_status_current()

    def _peak_file_open(self):
        name, _ = QFileDialog.getOpenFileName(
            self,
            "Load Peak CSV",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not name:
            return
        try:
            with open(name, newline="") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames is None:
                    raise ValueError("CSV file is missing a header row.")

                normalized_headers = {
                    str(header).strip().lower(): header for header in reader.fieldnames if header is not None
                }
                required = ("source", "pc", "ps")
                missing = [column for column in required if column not in normalized_headers]
                if missing:
                    raise ValueError(
                        "CSV must contain columns source, pc, and ps."
                    )

                def _first_header(*aliases):
                    for alias in aliases:
                        if alias in normalized_headers:
                            return normalized_headers[alias]
                    return None

                # Preferred manual/batch CSV format is explicit: provide either a
                # visual peak height or a model amplitude. Legacy pa/value input still
                # works only with a mode column so we do not confuse eyeballed heights
                # with area-amplitude parameters.
                height_header = _first_header("height", "peak_height", "pa_height")
                amplitude_header = _first_header("amplitude", "peak_amplitude", "pa_amplitude")
                legacy_value_header = _first_header("pa", "value")
                legacy_mode_header = _first_header("pa_mode", "value_mode", "value_type", "input_type")

                if height_header is None and amplitude_header is None and legacy_value_header is None:
                    raise ValueError(
                        "CSV must include either a height column or an amplitude column."
                    )
                if legacy_value_header is not None and legacy_mode_header is None:
                    raise ValueError(
                        "Legacy pa/value columns require a pa_mode/value_mode column "
                        "with values 'height' or 'amplitude'."
                    )

                def _parse_optional_float(row, header_name, row_number, label):
                    if header_name is None:
                        return None
                    raw_value = row.get(header_name)
                    if raw_value is None:
                        return None
                    text = str(raw_value).strip()
                    if text == "":
                        return None
                    try:
                        return float(text)
                    except ValueError as exc:
                        raise ValueError(
                            f"Row {row_number} has an invalid numeric value for {label}."
                        ) from exc

                loaded_peaks = []
                for row_number, row in enumerate(reader, start=2):
                    try:
                        source = str(row[normalized_headers["source"]]).strip()
                        pc = float(str(row[normalized_headers["pc"]]).strip())
                        ps = float(str(row[normalized_headers["ps"]]).strip())
                    except (TypeError, ValueError):
                        raise ValueError(
                            f"Row {row_number} must contain a valid source and numeric values for pc and ps."
                        ) from None

                    peak_height = _parse_optional_float(row, height_header, row_number, "height")
                    amplitude = _parse_optional_float(row, amplitude_header, row_number, "amplitude")

                    legacy_value = _parse_optional_float(row, legacy_value_header, row_number, "pa/value")
                    if legacy_value is not None:
                        if peak_height is not None or amplitude is not None:
                            raise ValueError(
                                f"Row {row_number} mixes legacy pa/value input with explicit height/amplitude columns. "
                                "Please use only one scheme per row."
                            )
                        raw_mode = str(row.get(legacy_mode_header, "")).strip().lower()
                        if raw_mode in {"height", "peak_height", "h"}:
                            peak_height = legacy_value
                        elif raw_mode in {"amplitude", "peak_amplitude", "amp", "area"}:
                            amplitude = legacy_value
                        else:
                            raise ValueError(
                                f"Row {row_number} has invalid {legacy_mode_header}='{raw_mode}'. "
                                "Use 'height' or 'amplitude'."
                            )

                    provided_count = int(peak_height is not None) + int(amplitude is not None)
                    if provided_count != 1:
                        raise ValueError(
                            f"Row {row_number} must provide exactly one of height or amplitude."
                        )

                    if peak_height is None:
                        peak_height = component_peak_height(amplitude, pc, ps)
                    else:
                        amplitude = component_amplitude_from_peak_height(peak_height, pc, ps)

                    loaded_peaks.append(
                        PeakGuess(
                            source=source,
                            center=pc,
                            amplitude=amplitude,
                            sigma=ps,
                            height=peak_height,
                            origin="manual",
                        )
                    )

                if not loaded_peaks:
                    raise ValueError("CSV does not contain any peak rows.")

            self.manual_peaks = loaded_peaks
            self._pc.clear()
            self._pa.clear()
            self._ps.clear()
            self._mark_preview_dirty()
            self._preview_status.setText(
                f"Loaded {len(loaded_peaks)} manual peaks from CSV. "
                "Preview out of date — click Regenerate Preview to apply changes, "
                "or Revert Pending Changes to keep the current preview."
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Peak CSV Error",
                f"Could not load peak CSV:\n{exc}\n\n"
                "Expected a CSV with columns source, pc, ps, and exactly one value field per row.\n"
                "Preferred: height or amplitude.\n"
                "Compatibility mode: pa/value plus pa_mode/value_mode set to height or amplitude.",
            )

    @staticmethod
    def _get_anymax_factor(ratio):
        if ratio <= 0 or ratio >= 1:  # return FWHM if ratio is invalid
            print("full width any max has invalid ratio")
            return 2.355
        return np.sqrt(8 * np.log(1 / ratio))

    @staticmethod
    def _median_dx(x) -> float:
        x = np.asarray(x, dtype=float)
        if x.size < 2:
            return 1.0
        diffs = np.diff(x)
        diffs = diffs[np.isfinite(diffs)]
        if diffs.size == 0:
            return 1.0
        dx = float(np.median(np.abs(diffs)))
        return dx if dx > 0 else 1.0

    def _sigma_ratios(self) -> tuple[float, ...]:
        start = float(self.fc.SIGMA_RATIO_START)
        end = float(self.fc.SIGMA_RATIO_END)
        step = abs(float(self.fc.SIGMA_RATIO_STEP))
        if step == 0:
            return tuple()
        ratios = []
        current = start
        guard = 0
        while current <= end + (step * 1e-6) and guard < 1000:
            if 0 < current < 1:
                ratios.append(round(float(current), 10))
            current += step
            guard += 1
        return tuple(ratios)

    @staticmethod
    def _interpolate_threshold_crossing(x0, y0, x1, y1, threshold) -> float:
        if y1 == y0:
            return float(x0)
        fraction = (threshold - y0) / (y1 - y0)
        return float(x0 + fraction * (x1 - x0))

    def _estimate_sigma_bounds(self, y, peak_index: int, all_peak_indices) -> tuple[int, int]:
        """
        Bound the width search around one ABS peak.

        Earlier width estimates could run all the way to the spectrum edge when the
        trace never crossed a chosen fraction. That inflated sigma for shoulders,
        raised baselines, broad peaks, and edge-adjacent features. These bounds prefer
        local minima between neighboring peaks, then neighboring peaks, and only then
        the array edge as a last resort.
        """
        y = np.asarray(y, dtype=float)
        minima = find_peaks(-y)[0]
        all_peak_indices = np.sort(np.asarray(all_peak_indices, dtype=int))

        left_neighbor_candidates = all_peak_indices[all_peak_indices < peak_index]
        right_neighbor_candidates = all_peak_indices[all_peak_indices > peak_index]
        left_neighbor = int(left_neighbor_candidates[-1]) if left_neighbor_candidates.size else None
        right_neighbor = int(right_neighbor_candidates[0]) if right_neighbor_candidates.size else None

        left_candidates = minima[minima < peak_index]
        if left_neighbor is not None:
            left_candidates = left_candidates[left_candidates > left_neighbor]
        right_candidates = minima[minima > peak_index]
        if right_neighbor is not None:
            right_candidates = right_candidates[right_candidates < right_neighbor]

        left_bound = int(left_candidates[-1]) if left_candidates.size else (
            left_neighbor if left_neighbor is not None else 0
        )
        right_bound = int(right_candidates[0]) if right_candidates.size else (
            right_neighbor if right_neighbor is not None else len(y) - 1
        )

        if left_bound >= peak_index:
            left_bound = max(0, peak_index - 1)
        if right_bound <= peak_index:
            right_bound = min(len(y) - 1, peak_index + 1)

        return left_bound, right_bound

    def _find_threshold_crossing(
        self,
        x,
        y,
        peak_index: int,
        threshold: float,
        bound_index: int,
        side: str,
    ) -> float | None:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if side == "left":
            for idx in range(peak_index, bound_index, -1):
                y_inner = float(y[idx])
                y_outer = float(y[idx - 1])
                if (y_inner >= threshold and y_outer <= threshold) or (y_inner <= threshold and y_outer >= threshold):
                    return self._interpolate_threshold_crossing(
                        float(x[idx - 1]), y_outer, float(x[idx]), y_inner, threshold
                    )
            return None

        for idx in range(peak_index, bound_index):
            y_inner = float(y[idx])
            y_outer = float(y[idx + 1])
            if (y_inner >= threshold and y_outer <= threshold) or (y_inner <= threshold and y_outer >= threshold):
                return self._interpolate_threshold_crossing(
                    float(x[idx]), y_inner, float(x[idx + 1]), y_outer, threshold
                )
        return None

    def _estimate_sigma_from_ratio(
        self,
        x,
        y,
        peak_index: int,
        ratio: float,
        left_bound: int,
        right_bound: int,
        baseline: float,
    ) -> _SigmaEstimate | None:
        """
        Estimate sigma from width at a fractional peak height.

        Two-sided crossings are preferred. One-sided crossings are allowed for peaks
        near an edge or shoulder, and the method string records that lower-confidence
        path for hover/debug output.
        """
        peak_height = float(y[peak_index])
        prominence_height = peak_height - float(baseline)
        if prominence_height <= 0:
            return None

        threshold = float(baseline) + float(ratio) * prominence_height
        factor = self._get_anymax_factor(ratio)
        center = float(np.asarray(x, dtype=float)[peak_index])
        left_cross = self._find_threshold_crossing(x, y, peak_index, threshold, left_bound, "left")
        right_cross = self._find_threshold_crossing(x, y, peak_index, threshold, right_bound, "right")

        if left_cross is not None and right_cross is not None and right_cross > left_cross:
            sigma = abs((right_cross - left_cross) / factor)
            return _SigmaEstimate(sigma=float(sigma), method="two_sided", ratios_used=(float(ratio),))

        if left_cross is not None:
            sigma = abs((2.0 * (center - left_cross)) / factor)
            return _SigmaEstimate(sigma=float(sigma), method="one_sided_left", ratios_used=(float(ratio),))

        if right_cross is not None:
            sigma = abs((2.0 * (right_cross - center)) / factor)
            return _SigmaEstimate(sigma=float(sigma), method="one_sided_right", ratios_used=(float(ratio),))

        return None

    @staticmethod
    def _gaussian_plus_baseline(x, amplitude, center, sigma, baseline):
        return baseline + stable_gaussian_sigma(x, amplitude, center, sigma)

    def _estimate_sigma_core_fit(
        self,
        x,
        y,
        peak_index: int,
        left_bound: int,
        right_bound: int,
        baseline: float,
    ) -> _SigmaEstimate | None:
        """
        Fallback sigma estimate using only the peak core.

        This fits the top of the feature rather than the wings because the wings are
        where shoulders, raised baselines, and overlapping transitions most often
        distort width estimates.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        dx = self._median_dx(x)

        peak_height = float(y[peak_index])
        prominence_height = peak_height - float(baseline)
        if prominence_height <= 0:
            return None

        local_indices = np.arange(left_bound, right_bound + 1, dtype=int)
        core_threshold = float(baseline) + 0.75 * prominence_height
        core_indices = local_indices[y[local_indices] >= core_threshold]
        if core_indices.size < 5:
            radius = 4
            start = max(left_bound, peak_index - radius)
            stop = min(right_bound + 1, peak_index + radius + 1)
        else:
            start = max(left_bound, int(core_indices[0]) - 1)
            stop = min(right_bound + 1, int(core_indices[-1]) + 2)

        if stop - start < 5:
            return None

        local_x = x[start:stop]
        local_y = y[start:stop]
        local_width = max(abs(float(x[right_bound]) - float(x[left_bound])), dx)
        sigma_guess = min(float(self.fc.MAX_SIGMA), max(dx, local_width / 6.0))
        amplitude_guess = max(prominence_height * sigma_guess * np.sqrt(2 * np.pi), 1e-18)
        center_guess = float(x[peak_index])
        center_pad = max(2.0 * dx, 0.2 * local_width)

        try:
            params, _ = curve_fit(
                lambda xvals, amplitude, center, sigma: self._gaussian_plus_baseline(
                    xvals, amplitude, center, sigma, baseline
                ),
                local_x,
                local_y,
                p0=(amplitude_guess, center_guess, sigma_guess),
                bounds=(
                    (0.0, center_guess - center_pad, dx / 2.0),
                    (np.inf, center_guess + center_pad, float(self.fc.MAX_SIGMA)),
                ),
                maxfev=10000,
            )
        except Exception:
            return None

        sigma = abs(float(params[2]))
        return _SigmaEstimate(sigma=sigma, method="core_fit", ratios_used=())

    def _estimate_average_sigma(self, x, y, peak_index, all_peak_indices) -> _SigmaEstimate:
        """
        Combine sigma estimates across human-readable height ratios.

        Ratios come from FitConfig as 0.60..0.90 style values. MAX_SIGMA is enforced
        here as an auto-guess cap only; final fitting can still move within DELTA_SIGMA
        around the accepted seed.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        dx = self._median_dx(x)
        left_bound, right_bound = self._estimate_sigma_bounds(y, int(peak_index), all_peak_indices)
        baseline = max(float(y[left_bound]), float(y[right_bound]))

        ratio_estimates = []
        for ratio in self._sigma_ratios():
            estimate = self._estimate_sigma_from_ratio(
                x,
                y,
                int(peak_index),
                float(ratio),
                left_bound,
                right_bound,
                baseline,
            )
            if estimate is not None and np.isfinite(estimate.sigma) and estimate.sigma > 0:
                ratio_estimates.append(estimate)

        if ratio_estimates:
            two_sided = [estimate for estimate in ratio_estimates if estimate.method == "two_sided"]
            chosen = two_sided if two_sided else ratio_estimates
            sigma = float(np.median([estimate.sigma for estimate in chosen]))
            sigma = min(float(self.fc.MAX_SIGMA), max(dx, sigma))
            method = "two_sided_median" if two_sided else "one_sided_median"
            ratios_used = tuple(estimate.ratios_used[0] for estimate in chosen)
            return _SigmaEstimate(sigma=sigma, method=method, ratios_used=ratios_used)

        fallback = self._estimate_sigma_core_fit(x, y, int(peak_index), left_bound, right_bound, baseline)
        if fallback is not None:
            sigma = min(float(self.fc.MAX_SIGMA), max(dx, fallback.sigma))
            return _SigmaEstimate(sigma=sigma, method=fallback.method, ratios_used=fallback.ratios_used)

        local_width = abs(float(x[right_bound]) - float(x[left_bound])) if right_bound > left_bound else dx
        sigma = min(float(self.fc.MAX_SIGMA), max(dx, local_width / 6.0))
        return _SigmaEstimate(sigma=sigma, method="window_fallback", ratios_used=())

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
    def _generate_initial_guesses_A(self, x, y_abs) -> list[PeakGuess]:

        # ABS is expected to contain approximately Gaussian absorption peaks. Candidate
        # centers come from the smoothed negative second derivative because it highlights
        # convex peak centers better than raw-intensity local maxima in crowded spectra.
        y_smoothed = savgol_filter(y_abs, window_length=self.fc.WINDOW_LENGTH, polyorder=self.fc.POLYORDER)
        # Calculate the numerical derivatives
        d_y_smoothed = np.gradient(y_smoothed, x)
        # Calculate the 2nd numerical derivatives
        dd_y = np.gradient(d_y_smoothed, x)
        dd_y_smoothed = np.gradient(d_y_smoothed, x)
        dd_y_smoothed = savgol_filter(dd_y_smoothed, window_length=self.fc.WINDOW_LENGTH, polyorder=self.fc.POLYORDER)
        # Find peaks in the negative second derivative (to locate the centers of Gaussians)
        prominence = self.fc.PROMINENCE_PERCENT * np.nanmax(dd_y)
        height = self.fc.HEIGHT_THRESHOLD * np.nanmax(dd_y)

        dd_y_peaks_all, peak_info = find_peaks(-dd_y_smoothed, height=height, distance=self.fc.DISTANCE,
                                               prominence=prominence)

        # filter peaks
        dd_y_peaks_all = self._filter_by_max_peak_height(-dd_y_smoothed, dd_y_peaks_all, peak_info)
        dd_y_peaks = self._filter_peaks_deltax(x, dd_y_peaks_all)

        peak_centers = x[dd_y_peaks]
        peak_heights = y_smoothed[dd_y_peaks]
        sigma_estimates = [
            self._estimate_average_sigma(x, y_smoothed, int(peak), dd_y_peaks)
            for peak in dd_y_peaks
        ]
        peak_sigmas = [estimate.sigma for estimate in sigma_estimates]

        peak_bundle = list(
            zip(
                np.asarray(peak_heights, dtype=float),
                np.asarray(peak_centers, dtype=float),
                np.asarray(peak_sigmas, dtype=float),
                [estimate.method for estimate in sigma_estimates],
            )
        )
        peak_bundle.sort(key=lambda item: item[0], reverse=True)
        peak_bundle = peak_bundle[: max(0, int(self.fc.MAX_GC))]

        peak_heights = np.asarray([item[0] for item in peak_bundle], dtype=float)
        peak_centers = np.asarray([item[1] for item in peak_bundle], dtype=float)
        peak_sigmas = np.asarray([item[2] for item in peak_bundle], dtype=float)
        peak_sigma_methods = [item[3] for item in peak_bundle]
        peak_amplitudes = np.asarray(
            [
                component_amplitude_from_peak_height(height, center, sigma)
                for height, center, sigma in zip(peak_heights, peak_centers, peak_sigmas)
            ],
            dtype=float,
        )

        _debug_guess_print(f'Initial Guess Peak Centers A: {peak_centers}')
        _debug_guess_print(f'Initial Guess Peak Sigmas A: {peak_sigmas}')
        _debug_guess_print(f'Initial Guess Peak Sigma Methods A: {peak_sigma_methods}')
        _debug_guess_print(f'Initial Guess Peak Heights A: {peak_heights}')
        _debug_guess_print(f'Initial Guess Peak Amplitudes A: {peak_amplitudes}')
        return [
            PeakGuess(
                source="abs",
                center=center,
                amplitude=amplitude,
                sigma=sigma,
                height=height,
                origin="auto",
            )
            for amplitude, center, sigma, height in zip(
                peak_amplitudes,
                peak_centers,
                peak_sigmas,
                peak_heights,
            )
        ]

    def _generate_initial_guesses_B(self, x, y_mcd, abs_peaks: list[PeakGuess]) -> list[PeakGuess]:
        """
        Generate MCD guesses using the configured MCD strategy.

        The current default strategy anchors on ABS transitions and fits the local MCD
        signal to an A-term-like derivative Gaussian plus a B-term-like Gaussian basis.
        This is deliberately routed through a strategy module so the MCD guess logic can
        be swapped later without rewriting the dialog flow.
        """
        return guess_mcd_peaks(x, y_mcd, abs_peaks, self.fc)

    def _split_manual_peaks(self) -> tuple[list[PeakGuess], list[PeakGuess]]:
        manual_abs = [peak for peak in self.manual_peaks if peak.source == "abs"]
        manual_mcd = [peak for peak in self.manual_peaks if peak.source == "mcd"]
        return manual_abs, manual_mcd

    def _build_transition_preview(
        self,
        auto_abs: list[PeakGuess],
        auto_mcd: list[PeakGuess],
    ) -> list[TransitionGuess]:
        manual_abs, manual_mcd = self._split_manual_peaks()
        source_mode = self._get_source_mode()
        if source_mode == "manual":
            abs_peaks = manual_abs
            mcd_peaks = manual_mcd
        elif source_mode == "merge":
            # "Merge" means union manual and auto guesses before cross-modal pairing.
            # It does not directly merge ABS with MCD; pair_peak_guesses handles that.
            abs_peaks = [*auto_abs, *manual_abs]
            mcd_peaks = [*auto_mcd, *manual_mcd]
        else:
            abs_peaks = auto_abs
            mcd_peaks = auto_mcd

        transitions = pair_peak_guesses(abs_peaks, mcd_peaks, self.fc.MERGE_DX)
        _debug_guess_print("\n--- Transition summary ---")
        _debug_guess_print(self._transition_count_summary(transitions))
        for transition in transitions:
            abs_center = None if transition.abs_peak is None else float(transition.abs_peak.center)
            mcd_center = None if transition.mcd_peak is None else float(transition.mcd_peak.center)
            _debug_guess_print(
                f"{transition.transition_id}: status={transition.status} "
                f"abs_center={abs_center} mcd_center={mcd_center} "
                f"match_distance={transition.match_distance}"
            )
        _debug_guess_print("--- transition debug done ------------------------\n")
        return transitions

    def _guess_on_all_data(self, x, y_abs, y_mcd):
        cached_transitions, cached_manual_peaks, cached_source_mode = self.fc.get_current_peaks()
        source_mode = self._get_source_mode()
        cache_state_matches = (
            cached_source_mode == source_mode and
            cached_manual_peaks == self.manual_peaks
        )
        if (
            cache_state_matches and (
                source_mode == "manual" or self.fc.current_peaks_match_guess_config()
            )
        ):
            return list(cached_transitions or [])

        if cached_transitions and source_mode != "manual" and not self.fc.current_peaks_match_guess_config():
            _debug_guess_print("Cached peak guesses invalidated; regenerating with current FitConfig.")
            self.fc.clear_current_peaks(preserve_input_peaks=True)

        auto_abs = self._generate_initial_guesses_A(x, y_abs)
        auto_mcd = self._generate_initial_guesses_B(x, y_mcd, auto_abs)

        _debug_guess_print("len A:", len(auto_abs))
        _debug_guess_print("len B:", len(auto_mcd))
        _debug_guess_print("B centers preview:", [float(peak.center) for peak in auto_mcd])

        return self._build_transition_preview(auto_abs, auto_mcd)

    # TODO: maybe rank peaks and not remove in both directions
    # removes peaks in both directions
    def _filter_peaks_deltax(self, x, peaks):
        if len(peaks) == 0:
            return np.array([])
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
        return list(self.transitions)
    def get_fc(self):
        return self.fc


class _HeadlessGuessEngine:
    """
    Small adapter that reuses guessWindow's non-GUI peak-guessing methods.

    Batch mode needs the same peak guesses as the preview dialog without constructing
    Qt widgets. Binding the calculation methods here avoids a second, divergent copy
    of the guessing algorithm.
    """

    def __init__(self, fc, source_mode: str = "auto", manual_peaks: list[PeakGuess] | None = None):
        self.fc = fc
        self.manual_peaks = list(manual_peaks or [])
        self._source_mode_value = str(source_mode or "auto")

    def _get_source_mode(self) -> str:
        return self._source_mode_value


for _method_name in (
    "_sigma_ratios",
    "_estimate_sigma_bounds",
    "_find_threshold_crossing",
    "_estimate_sigma_from_ratio",
    "_estimate_sigma_core_fit",
    "_estimate_average_sigma",
    "_filter_by_max_peak_height",
    "_trim_peaks",
    "_generate_initial_guesses_A",
    "_generate_initial_guesses_B",
    "_split_manual_peaks",
    "_build_transition_preview",
    "_guess_on_all_data",
    "_filter_peaks_deltax",
    "_transition_count_summary",
):
    setattr(_HeadlessGuessEngine, _method_name, getattr(guessWindow, _method_name))

for _staticmethod_name in (
    "_get_anymax_factor",
    "_median_dx",
    "_interpolate_threshold_crossing",
    "_gaussian_plus_baseline",
    "_transition_count_summary",
):
    setattr(_HeadlessGuessEngine, _staticmethod_name, staticmethod(getattr(guessWindow, _staticmethod_name)))


def auto_guess_transitions(
    x,
    y_abs,
    y_mcd,
    fc,
    source_mode: str = "auto",
    manual_peaks: list[PeakGuess] | None = None,
    use_cache: bool = False,
) -> list[TransitionGuess]:
    """
    Generate the same automatic transition guesses used by the preview dialog without
    constructing or displaying any Qt widgets.
    """
    if not use_cache:
        fc.clear_current_peaks(preserve_input_peaks=False)
    engine = _HeadlessGuessEngine(fc=fc, source_mode=source_mode, manual_peaks=manual_peaks)
    return engine._guess_on_all_data(np.asarray(x, dtype=float), np.asarray(y_abs, dtype=float), np.asarray(y_mcd, dtype=float))
