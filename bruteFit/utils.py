import os
from bruteFit.fitConfig import FitConfig

"""
    Launches the proc viewer GUI

    Pre_clean is on the left and post_clean is on the right.
    Can exit after load to skip save.
    """
def launch_proc_viewer():
    import sys
    # Preload dateutil before any Qt imports. On this Python 3.12 + PySide6 stack,
    # importing dateutil/six after Qt has initialized shiboken can crash startup.
    from dateutil import rrule, tz  # noqa: F401
    import pandas as pd
    import matplotlib
    matplotlib.use("QtAgg")

    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QDoubleSpinBox,
        QPushButton, QLabel, QTableView, QMessageBox, QSplitter, QFileDialog,
        QDialog, QDialogButtonBox, QFormLayout
    )
    from PySide6.QtCore import Qt, QAbstractTableModel
    from PySide6.QtGui import QValidator
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg as FigureCanvas,
        NavigationToolbar2QT as NavigationToolbar,
    )

    from processrecord.container import ProcessRecord
    from processrecord import fileutils as fh

    class _FlexibleFloatSpinBox(QDoubleSpinBox):
        """Allow compact entry for very small values such as 1e-6."""
        def validate(self, text, pos):
            stripped = text.strip()
            if stripped in {"", "-", "+", ".", "-.", "+."}:
                return (QValidator.State.Intermediate, text, pos)
            try:
                float(stripped.replace("E", "e"))
            except ValueError:
                return (QValidator.State.Invalid, text, pos)
            return (QValidator.State.Acceptable, text, pos)

        def valueFromText(self, text):
            try:
                return float(text.replace("E", "e"))
            except ValueError:
                return 0.0

        def textFromValue(self, value):
            return f"{value:.12g}"

    class PandasModel(QAbstractTableModel):
        def __init__(self, df=pd.DataFrame(), parent=None):
            super().__init__(parent)
            self._df = df

        def set_df(self, df):
            self.beginResetModel()
            self._df = df
            self.endResetModel()

        def rowCount(self, parent=None):
            return len(self._df)

        def columnCount(self, parent=None):
            return len(self._df.columns)

        def data(self, index, role=Qt.DisplayRole):
            if role == Qt.DisplayRole:
                return str(self._df.iat[index.row(), index.column()])
            return None

        def headerData(self, section, orientation, role=Qt.DisplayRole):
            if role == Qt.DisplayRole:
                if orientation == Qt.Horizontal:
                    return str(self._df.columns[section])
                else:
                    return str(self._df.index[section])
            return None

    class ProcViewer(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("ProcessRecord — Full vs Cleaned with Plots")
            self.proc = None
            self._result_df = pd.DataFrame()  # will hold cleaned df for return

            # Top inputs in one horizontal row
            top_row = QHBoxLayout()
            self.lims_edit = QLineEdit("test_lims")
            self._labeled(top_row, "LimsID:", self.lims_edit)

            self.pathlength_spin = QDoubleSpinBox()
            self.pathlength_spin.setRange(0, 1e12)
            self.pathlength_spin.setDecimals(6); self.pathlength_spin.setValue(1.0)
            self._labeled(top_row, "Pathlength:", self.pathlength_spin)

            self.conc_spin = _FlexibleFloatSpinBox()
            self.conc_spin.setRange(0, 1e12)
            self.conc_spin.setDecimals(12)
            self.conc_spin.setSingleStep(1e-6)
            self.conc_spin.setValue(2.0)
            self._labeled(top_row, "Concentration:", self.conc_spin)

            self.field_spin = QDoubleSpinBox()
            self.field_spin.setRange(-1e12, 1e12)
            self.field_spin.setDecimals(6)
            self.field_spin.setValue(3.0)
            self._labeled(top_row, "Field:", self.field_spin)
            top_row.addStretch(1)

            # Buttons
            self.run_btn = QPushButton("Load and Process")
            self.run_btn.clicked.connect(self.load_normal)
            self.save_continue_btn = QPushButton("Save and Continue")
            self.save_continue_btn.clicked.connect(self.save_and_continue)
            top_row.addWidget(self.run_btn)
            top_row.addWidget(self.save_continue_btn)

            self.load_processed_btn = QPushButton("Load Processed")
            self.load_processed_btn.clicked.connect(self.load_processed)
            top_row.addWidget(self.load_processed_btn)

            # Tables splitter (unchanged)
            self.full_model = PandasModel()
            self.clean_model = PandasModel()
            self.full_table = QTableView();  self.full_table.setModel(self.full_model)
            self.clean_table = QTableView(); self.clean_table.setModel(self.clean_model)

            full_wrap = self._with_label("Full Data (pre-clean)", self.full_table)
            clean_wrap = self._with_label("Cleaned Data (post-clean)", self.clean_table)

            tables_splitter = QSplitter(Qt.Horizontal)
            tables_splitter.addWidget(full_wrap)
            tables_splitter.addWidget(clean_wrap)
            tables_splitter.setChildrenCollapsible(False)
            #Pyqt trick to have the ratios of the processing gui split evenly no matter the size of the window
            tables_splitter.setSizes([1_000_000, 1_000_000])

            # Plots splitter
            self.pre_canvas = None
            self.post_canvas = None
            self.pre_plot_holder = QVBoxLayout()
            self.post_plot_holder = QVBoxLayout()
            pre_plot_wrap = QWidget();  pre_plot_wrap.setLayout(self.pre_plot_holder)
            post_plot_wrap = QWidget(); post_plot_wrap.setLayout(self.post_plot_holder)

            plots_splitter = QSplitter(Qt.Horizontal)
            plots_splitter.addWidget(self._with_label("Pre-clean Plot", pre_plot_wrap))
            plots_splitter.addWidget(self._with_label("Post-clean Plot", post_plot_wrap))
            plots_splitter.setChildrenCollapsible(False)
            #same trick here. This is why we have large numbers
            plots_splitter.setSizes([800_000, 800_000])

            # Main vertical splitter — CHANGE sizes to give more space to plots if you want
            main_splitter = QSplitter(Qt.Vertical)
            main_splitter.addWidget(tables_splitter)
            main_splitter.addWidget(plots_splitter)
            main_splitter.setChildrenCollapsible(False)
            main_splitter.setSizes([1_000_000, 1_000_000])

            # Root layout
            root = QVBoxLayout(self)
            root.addLayout(top_row)
            root.addWidget(main_splitter)

        """
        Injects a right-aligned and centered label and the widget as siblings into an existing layout
        """
        def _labeled(self, layout, text, widget):
            lab = QLabel(text)
            lab.setAlignment(Qt.AlignVCenter | Qt.AlignRight) #right aligned and centered by or on flags
            layout.addWidget(lab)
            layout.addWidget(widget)

        """
        Returns a single QWidget 'wrapper' containing a centered label stacked vertically above the widget
        """
        def _with_label(self, text, widget):
            w = QWidget()
            v = QVBoxLayout(w)
            lab = QLabel(text)
            lab.setAlignment(Qt.AlignCenter)
            v.addWidget(lab)
            v.addWidget(widget)
            return w

        def _review_processing_inputs(
            self,
            metadata_values: dict,
            metadata_path: str | None,
            basename: str | None,
        ):
            initial_lims = str(metadata_values.get("lims_ID", self.lims_edit.text().strip() or basename or "test_lims"))
            initial_pathlength = float(metadata_values.get("pathlength_cm", self.pathlength_spin.value()))
            initial_concentration = float(metadata_values.get("concentration_MOL_L", self.conc_spin.value()))
            initial_field = float(metadata_values.get("field_B", self.field_spin.value()))

            if metadata_path is None:
                return initial_lims, initial_concentration, initial_pathlength, initial_field

            dlg = QDialog(self)
            dlg.setWindowTitle("Review Processing Inputs")
            layout = QVBoxLayout(dlg)

            info = QLabel(
                "Loaded processing metadata from:\n"
                f"{metadata_path}\n\n"
                "Review or change these values before continuing."
            )
            info.setWordWrap(True)
            layout.addWidget(info)

            form = QFormLayout()
            lims_edit = QLineEdit(initial_lims)
            form.addRow("LimsID:", lims_edit)

            pathlength_spin = QDoubleSpinBox()
            pathlength_spin.setRange(0, 1e12)
            pathlength_spin.setDecimals(6)
            pathlength_spin.setValue(initial_pathlength)
            form.addRow("Pathlength:", pathlength_spin)

            conc_spin = _FlexibleFloatSpinBox()
            conc_spin.setRange(0, 1e12)
            conc_spin.setDecimals(12)
            conc_spin.setSingleStep(1e-6)
            conc_spin.setValue(initial_concentration)
            form.addRow("Concentration:", conc_spin)

            field_spin = QDoubleSpinBox()
            field_spin.setRange(-1e12, 1e12)
            field_spin.setDecimals(6)
            field_spin.setValue(initial_field)
            form.addRow("Field:", field_spin)

            layout.addLayout(form)

            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
            )
            buttons.accepted.connect(dlg.accept)
            buttons.rejected.connect(dlg.reject)
            layout.addWidget(buttons)

            if dlg.exec() != QDialog.DialogCode.Accepted:
                return None

            return (
                lims_edit.text().strip() or basename or "test_lims",
                float(conc_spin.value()),
                float(pathlength_spin.value()),
                float(field_spin.value()),
            )

        """
        clears and updates canvas in holder_layout
        """
        def _set_canvas_in(self, holder_layout: QVBoxLayout, new_canvas: FigureCanvas):
            # Clear
            for widget in holder_layout.findChildren(QWidget):
                widget.deleteLater()

            # Add new canvas with toolbar
            toolbar = NavigationToolbar(new_canvas, self)
            holder_layout.addWidget(new_canvas)
            holder_layout.addWidget(toolbar)
            new_canvas.draw_idle()
        """
        Loads a processed csv file by extracting the inputs and putting them into a new ProcessRecord object.
        
        Params: None
        
        returns - None
        
        The processing pipeline is ran at the end of this function. 
        """
        def load_processed(self):
            fname, _ = QFileDialog.getOpenFileName()
            if not fname:
                return

            try:
                df = pd.read_csv(fname)
                name = os.path.splitext(os.path.basename(fname))[0]
                raw_pos = df[[
                    "wavelength_nm_inp",
                    "xcartesian_mcdpos_deltaabsorptivityarbunits_inp",
                    "ycartesian_mcdpos_deltaabsorptivityarbunits_inp",
                    "rabsolute_mcdpos_deltaabsorptivityarbunits_inp",
                    "theta_mcdpos_radians_inp",
                    "stddevxcartesian_mcdpos_deltaabsorptivity_inp",
                    "stddevycartesian_mcdpos_deltaabsorptivity_inp"
                ]]
                raw_neg = df[[
                    "wavelength_nm_inp",
                    "xcartesian_mcdneg_deltaabsorptivityarbunits_inp",
                    "ycartesian_mcdneg_deltaabsorptivityarbunits_inp",
                    "rabsolute_mcdneg_deltaabsorptivityarbunits_inp",
                    "theta_mcdneg_radians_inp",
                    "stddevxcartesian_mcdneg_deltaabsorptivity_inp",
                    "stddevycartesian_mcdneg_deltaabsorptivity_inp"
                ]]
                raw_abs = df[[
                    "wavelength_nm_inp",
                    "uvvisabsorptivity_abs_absorptivityarbunits_inp"
                ]]
                raw_sticks = df[[
                    "wavelength_nm_inp",
                    "sticks_out"
                ]]
                pos_df = raw_pos.rename(columns={
                    "wavelength_nm_inp": "wavelength",
                    "xcartesian_mcdpos_deltaabsorptivityarbunits_inp": "x_pos",
                    "ycartesian_mcdpos_deltaabsorptivityarbunits_inp": "y_pos",
                    "rabsolute_mcdpos_deltaabsorptivityarbunits_inp": "R",
                    "theta_mcdpos_radians_inp": "theta",
                    "stddevxcartesian_mcdpos_deltaabsorptivity_inp": "std_dev_x",
                    "stddevycartesian_mcdpos_deltaabsorptivity_inp": "std_dev_y"
                })
                neg_df = raw_neg.rename(columns={
                    "wavelength_nm_inp": "wavelength",
                    "xcartesian_mcdneg_deltaabsorptivityarbunits_inp": "x_neg",
                    "ycartesian_mcdneg_deltaabsorptivityarbunits_inp": "y_neg",
                    "rabsolute_mcdneg_deltaabsorptivityarbunits_inp": "R",
                    "theta_mcdneg_radians_inp": "theta",
                    "stddevxcartesian_mcdneg_deltaabsorptivity_inp": "std_dev_x",
                    "stddevycartesian_mcdneg_deltaabsorptivity_inp": "std_dev_y"
                })
                abs_df = raw_abs.rename(columns={
                    "wavelength_nm_inp": "wavelength",
                    "uvvisabsorptivity_abs_absorptivityarbunits_inp": "intensity"
                })
                sticks_df = raw_sticks.rename(columns={
                    "wavelength_nm_inp": "wavelength",
                    "sticks_out": "strength"
                })
                #grab first one they are all the same
                lims_ID = df["lims_ID"].iloc[0]
                conc = df["concentration_MOL_L"].iloc[0]
                pathlength = df["pathlength_cm"].iloc[0]
                field_B = df["field_B"].iloc[0]

                self.proc = ProcessRecord((pos_df, neg_df, abs_df, sticks_df, name),lims_ID, conc, pathlength, field_B)

            except Exception as e:
                print(f"Failed to load processed data -> {e}")
            self._run_pipeline()

        """
        A normal load that takes 3 files and optionally one more file.
        The 3 required files are the positive field mcd response, the negative field mcd response, and the absorption data.
        The required layouts for each CSV is in the readme.md
        
        params: None
        
        returns- None
        
        The main processing pipeline is called at the end of this function.
        """
        def load_normal(self):
            pos_df, neg_df, abs_df, sticks_df, basename, metadata_values, metadata_path = fh.read_pos_neg_abs()
            if pos_df is None or neg_df is None or abs_df is None:
                return

            reviewed = self._review_processing_inputs(metadata_values, metadata_path, basename)
            if reviewed is None:
                return

            lims, concentration, pathlength, field = reviewed
            self.lims_edit.setText(lims)
            self.pathlength_spin.setValue(pathlength)
            self.conc_spin.setValue(concentration)
            self.field_spin.setValue(field)

            # 5) Build tuple and create ProcessRecord (note arg order conc, pathlength)
            input_tuple = (pos_df, neg_df, abs_df, sticks_df, basename)
            self.proc = ProcessRecord(input_tuple, lims, concentration, pathlength, field)
            self._run_pipeline()
        def _run_pipeline(self):
            try:
                # Full (pre-clean) plot
                full_df = self.proc.get_merged_df()
                self.full_model.set_df(full_df)
                fig_pre, _ = self.proc.plot_extinction(return_fig=True)
                self._set_canvas_in(self.pre_plot_holder, FigureCanvas(fig_pre))

                # Clean + cleaned DF
                self.proc.clean_data()
                clean_df = self.proc.get_merged_df()
                self.clean_model.set_df(clean_df)

                # Store cleaned for later return
                self._result_df = clean_df

                # Full (post-clean) plot
                fig_post, _ = self.proc.plot_extinction(return_fig=True)
                self._set_canvas_in(self.post_plot_holder, FigureCanvas(fig_post))

            except Exception as e:
                QMessageBox.critical(self, "Error", f"{type(e).__name__}: {e}")
        """
        Opens prompt to save processed data
        
        params: None
        
        returns- None
        
        Uses the save method in ProcessRecord 
        """
        def save_and_continue(self):
            try:
                if self.proc is None:
                    QMessageBox.warning(self, "Warning", "Run 'Load & Process' first.")
                    return
                folder = QFileDialog.getExistingDirectory(self, "Select Save Directory")
                if not folder:
                    return
                self.proc.save(folder)
                self.close()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"{type(e).__name__}: {e}")

        @property
        def result_df(self):
            return self._result_df

    app = QApplication(sys.argv)
    viewer = ProcViewer()
    viewer.showMaximized()  # full screen as before
    app.exec()

    # Return whatever was cleaned at the time of Save & Continue (or after last run)
    return viewer.result_df
