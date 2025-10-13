def launch_proc_viewer():
    """
    Full-screen GUI:
      - Top row: LimsID, Pathlength, Concentration, Field + Load & Process + Save & Continue
      - Middle: two tables (Full pre-clean, Cleaned post-clean)
      - Bottom: two plots (Pre-clean, Post-clean)
      - Returns the cleaned DataFrame after 'Save & Continue'
    """
    import sys
    import pandas as pd
    import matplotlib
    matplotlib.use("QtAgg")

    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QDoubleSpinBox,
        QPushButton, QLabel, QTableView, QMessageBox, QSplitter, QFileDialog
    )
    from PySide6.QtCore import Qt, QAbstractTableModel
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

    # your packages (unchanged)
    from processrecord.container import ProcessRecord
    from processrecord import fileutils as fh

    # ---------- pandas -> Qt model (unchanged) ----------
    class PandasModel(QAbstractTableModel):
        def __init__(self, df=pd.DataFrame(), parent=None):
            super().__init__(parent)
            self._df = df

        def set_df(self, df: pd.DataFrame):
            self.beginResetModel()
            self._df = df if df is not None else pd.DataFrame()
            self.endResetModel()

        def rowCount(self, parent=None): return len(self._df.index)
        def columnCount(self, parent=None): return len(self._df.columns)
        def data(self, index, role=Qt.DisplayRole):
            if not index.isValid() or role != Qt.DisplayRole: return None
            return str(self._df.iat[index.row(), index.column()])
        def headerData(self, section, orientation, role=Qt.DisplayRole):
            if role != Qt.DisplayRole: return None
            return str(self._df.columns[section] if orientation == Qt.Horizontal else self._df.index[section])

    # ---------- main window (mostly unchanged) ----------
    class ProcViewer(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("ProcessRecord — Full vs Cleaned with Plots")
            self.proc = None
            self._result_df = pd.DataFrame()  # will hold cleaned df for return

            # Top inputs in one horizontal row (unchanged)
            top_row = QHBoxLayout()
            self.lims_edit = QLineEdit("test_lims");         self._labeled(top_row, "LimsID:", self.lims_edit)
            self.pathlength_spin = QDoubleSpinBox();         self.pathlength_spin.setRange(0, 1e12); self.pathlength_spin.setDecimals(6); self.pathlength_spin.setValue(1.0)
            self._labeled(top_row, "Pathlength:", self.pathlength_spin)
            self.conc_spin = QDoubleSpinBox();               self.conc_spin.setRange(0, 1e12); self.conc_spin.setDecimals(6); self.conc_spin.setValue(2.0)
            self._labeled(top_row, "Concentration:", self.conc_spin)
            self.field_spin = QDoubleSpinBox();              self.field_spin.setRange(-1e12, 1e12); self.field_spin.setDecimals(6); self.field_spin.setValue(3.0)
            self._labeled(top_row, "Field:", self.field_spin)
            top_row.addStretch(1)

            # Buttons (ADD: Save & Continue)
            self.run_btn = QPushButton("Load and Process")
            self.run_btn.clicked.connect(self.run_pipeline)
            self.save_continue_btn = QPushButton("Save and Continue")
            self.save_continue_btn.clicked.connect(self.save_and_continue)
            top_row.addWidget(self.run_btn)
            top_row.addWidget(self.save_continue_btn)

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
            # TODO explain/ fix magic numbers. 
            tables_splitter.setSizes([1_000_000, 1_000_000])

            # Plots splitter (unchanged)
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
            plots_splitter.setSizes([800_000, 800_000])

            # Main vertical splitter — CHANGE sizes to give more space to plots if you want
            main_splitter = QSplitter(Qt.Vertical)
            main_splitter.addWidget(tables_splitter)
            main_splitter.addWidget(plots_splitter)
            main_splitter.setChildrenCollapsible(False)
            # ↓ If you previously had [1_600_000, 600_000], now bias plots more:
            main_splitter.setSizes([1_000_000, 1_000_000])

            # Root layout (unchanged)
            root = QVBoxLayout(self)
            root.addLayout(top_row)
            root.addWidget(main_splitter)

        def _labeled(self, layout, text, widget):
            lab = QLabel(text); lab.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
            layout.addWidget(lab); layout.addWidget(widget)

        def _with_label(self, text, widget):
            w = QWidget(); v = QVBoxLayout(w)
            lab = QLabel(text); lab.setAlignment(Qt.AlignCenter)
            v.addWidget(lab); v.addWidget(widget)
            return w

        def _set_canvas_in(self, holder_layout: QVBoxLayout, new_canvas: FigureCanvas, keep_attr: str):
            old_canvas = getattr(self, keep_attr, None)
            if old_canvas is not None:
                holder_layout.removeWidget(old_canvas)
                old_canvas.setParent(None)
                old_canvas.deleteLater()
            setattr(self, keep_attr, new_canvas)
            holder_layout.addWidget(new_canvas)
            new_canvas.draw_idle()

        def run_pipeline(self):
            try:
                lims = self.lims_edit.text().strip() or "test_lims"
                pathlength = float(self.pathlength_spin.value())
                concentration = float(self.conc_spin.value())
                field = float(self.field_spin.value())

                # input + process

                # 1) Get file paths via Qt
                paths, _ = QFileDialog.getOpenFileNames(
                    self,
                    "Select CSV files (pos/neg/abs; sticks optional)",
                    filter="CSV Files (*.csv);;All Files (*)"
                )
                if not paths:
                    return

                # 2) Classify by filename
                mapping = fh.guess_roles_from_filenames(paths)
                for need in ("pos", "neg", "abs"):
                    if need not in mapping:
                        QMessageBox.critical(self, "Error", f"Missing required file: {need}")
                        return

                # 3) Your preferred explicit schemas
                column_names = {
                    "pos":    ["wavelength","x_pos","y_pos","R","theta","std_dev_x","std_dev_y","additional"],
                    "neg":    ["wavelength","x_neg","y_neg","R","theta","std_dev_x","std_dev_y","additional"],
                    "abs":    ["wavelength","intensity"],
                    "sticks": ["wavelength","strength"],
                }

                # 4) Read with header=None and assign names
                pos_df, neg_df, abs_df, sticks_df, ident = fh.read_pos_neg_abs_from_paths(
                    mapping,
                    column_names=column_names,
                    assume_no_header=True,          # <-- important for unlabeled CSVs
                    csv_reader=pd.read_csv          # you can pass a custom reader if needed
                )

                # 5) Build tuple and create ProcessRecord (note arg order conc, pathlength)
                input_tuple = (pos_df, neg_df, abs_df, sticks_df, ident)
                self.proc = ProcessRecord(input_tuple, lims, concentration, pathlength, field)

                #input_tuple = (pos_df, neg_df, abs_df, sticks_df, ident)
                # NOTE: ProcessRecord expects (concentration, pathlength)
                #self.proc = ProcessRecord(input_tuple, lims, concentration, pathlength, field)
                # SAM NOTE this is where LLM flags issue with passing pl first, conc second. Switching.
                #self.proc = ProcessRecord(input_tuple, lims, pathlength, concentration, field)


                # Full (pre-clean)
                full_df = self.proc.get_merged_df()
                self.full_model.set_df(full_df)

                # PRE-CLEAN PLOT (reuse method by temporarily pointing mergedOut)
                bak = getattr(self.proc, "mergedOut", None)
                self.proc.mergedOut = full_df
                fig_pre, _ = self.proc.plot_extinction(return_fig=True)
                self.proc.mergedOut = bak
                self._set_canvas_in(self.pre_plot_holder, FigureCanvas(fig_pre), "pre_canvas")

                # Clean + cleaned DF
                self.proc.clean_data()
                clean_df = self.proc.get_merged_df()
                self.clean_model.set_df(clean_df)

                # Store cleaned for later return
                self._result_df = clean_df

                # POST-CLEAN PLOT
                fig_post, _ = self.proc.plot_extinction(return_fig=True)
                self._set_canvas_in(self.post_plot_holder, FigureCanvas(fig_post), "post_canvas")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"{type(e).__name__}: {e}")

        def save_and_continue(self):
            """Save via ProcessRecord.save(folder) and close, returning cleaned df."""
            try:
                if self.proc is None:
                    QMessageBox.warning(self, "Warning", "Run 'Load & Process' first.")
                    return
                folder = QFileDialog.getExistingDirectory(self, "Select Save Directory")
                if not folder:
                    return
                # Use your existing save method
                self.proc.save(folder)
                # Close window; launch_proc_viewer will return the cleaned df
                self.close()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"{type(e).__name__}: {e}")

    app = QApplication(sys.argv)
    viewer = ProcViewer()
    viewer.showMaximized()  # full screen as before
    app.exec()

    # Return whatever was cleaned at the time of Save & Continue (or after last run)
    return viewer._result_df