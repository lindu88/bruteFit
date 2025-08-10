import sys

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QScrollArea, QSizePolicy, QDialog, QDialogButtonBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib
from matplotlib.figure import Figure

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

def show_peak_centers_window(x, y_mcd, peak_centers, title="Peak Centers"):
    """
    Shows a PySide6 dialog with the plot and Yes/No buttons.
    Returns True if user clicks Yes (continue), False if No.
    """
    # Compute y at each center
    y_at_centers = np.interp(peak_centers, x, y_mcd)

    # Build a standalone Matplotlib Figure (no pyplot)
    fig = Figure(figsize=(8, 4), dpi=100)
    ax = fig.add_subplot(111)

    ax.plot(x, y_mcd, '-', lw=1, alpha=0.7, label='y (original)')
    ax.scatter(peak_centers, y_at_centers, s=36, zorder=3, label='peak centers')
    for cx in peak_centers:
        ax.axvline(cx, ls='--', lw=0.8, alpha=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('signal')
    ax.legend()
    fig.tight_layout()

    # Dialog UI
    dlg = QDialog()
    dlg.setWindowTitle(title)
    layout = QVBoxLayout(dlg)

    canvas = FigureCanvas(fig)
    toolbar = NavigationToolbar(canvas, dlg)
    layout.addWidget(toolbar)
    layout.addWidget(canvas)

    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No)
    layout.addWidget(buttons)
    buttons.accepted.connect(dlg.accept)  # Yes
    buttons.rejected.connect(dlg.reject)  # No

    result = dlg.exec()
    return result == QDialog.DialogCode.Accepted