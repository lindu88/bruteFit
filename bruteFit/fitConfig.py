from dataclasses import dataclass, asdict, field


def _fc_meta(label: str, section: str, group: str, help_text: str) -> dict:
    return {"label": label, "section": section, "group": group, "help": help_text}

@dataclass
class FitConfig:

    #TODO tune these parameters by optimizing over a large set of (representative) synthetic (and real?) data. 
    # estimate the amount of datapoints needed given the amount of params.

    #So peaks are saved on reset and tied to fitting params
    __current_peaks: list | None = field(default=None, repr=False)
    __current_imp_peaks: list | None = field(default=None, repr=False)
    __current_peak_signature: tuple | None = field(default=None, repr=False)
    __current_source_mode: str = field(default="auto", repr=False)

    # Smoothing
    WINDOW_LENGTH: int = field(
        default=5,
        metadata=_fc_meta(
            "S-G Window Length",
            "Guess Generation",
            "Smoothing",
            "Raw key: WINDOW_LENGTH\n"
            "Savitzky-Golay smoothing window length in data points. Larger values suppress more noise "
            "but can blur narrow peaks. Must be odd and greater than POLYORDER.",
        ),
    )
    POLYORDER: int = field(
        default=4,
        metadata=_fc_meta(
            "S-G Polyorder",
            "Guess Generation",
            "Smoothing",
            "Raw key: POLYORDER\n"
            "Polynomial order used inside each Savitzky-Golay window. Higher values preserve more local curvature "
            "but smooth less aggressively. Must be lower than WINDOW_LENGTH.",
        ),
    )

    # Peak picking
    HEIGHT_THRESHOLD: float = field(
        default=0.04,
        metadata=_fc_meta(
            "Relative Peak Height",
            "Guess Generation",
            "Peak Picking",
            "Raw key: HEIGHT_THRESHOLD\n"
            "Minimum peak height as a fraction of the largest signal used during automatic peak finding. "
            "Higher values reject weaker peaks.",
        ),
    )
    PROMINENCE_PERCENT: float = field(
        default=0.04,
        metadata=_fc_meta(
            "Relative Prominence",
            "Guess Generation",
            "Peak Picking",
            "Raw key: PROMINENCE_PERCENT\n"
            "Minimum prominence as a fraction of the largest signal. Prominence measures how much a peak stands "
            "out from nearby baseline and neighboring features.",
        ),
    )
    DISTANCE: int = field(
        default=5,
        metadata=_fc_meta(
            "Peak Spacing (points)",
            "Guess Generation",
            "Peak Picking",
            "Raw key: DISTANCE\n"
            "Minimum spacing between peaks reported by automatic peak finding, in data points. "
            "Increase this to merge crowded candidates.",
        ),
    )

    # Fitting
    MERGE_DX: int = field(
        default=300,
        metadata=_fc_meta(
            "ABS/MCD Pairing Distance",
            "Guess Generation",
            "Transition Pairing",
            "Raw key: MERGE_DX\n"
            "Maximum center difference allowed when pairing absorption-derived and MCD-derived guesses into the "
            "same transition. Unpaired peaks remain visible, but only paired transitions contribute to A/D or "
            "B/D reporting.",
        ),
    )
    MAX_SIGMA: int = field(
        default=3000,
        metadata=_fc_meta(
            "Max Auto-Guess Sigma",
            "Guess Generation",
            "Sigma Estimation",
            "Raw key: MAX_SIGMA\n"
            "Upper cap applied while estimating initial sigma values for automatically guessed peaks. "
            "This affects auto-guessing only and does not directly hard-cap the final fit.",
        ),
    )
    MIN_PEAK_X_DISTANCE: int = field(
        default=0,
        metadata=_fc_meta(
            "Min Peak Separation (x)",
            "Guess Generation",
            "Peak Picking",
            "Raw key: MIN_PEAK_X_DISTANCE\n"
            "Drops peaks that are too close together in x after detection. Use cautiously when peaks are crowded.",
        ),
    )
    SIGMA_RATIO_START: float = field(
        default=0.60,
        metadata=_fc_meta(
            "Sigma Ratio Start",
            "Guess Generation",
            "Sigma Estimation",
            "Raw key: SIGMA_RATIO_START\n"
            "Lower bound of the peak-height fractions used when estimating sigma from width measurements. "
            "Higher values focus more tightly on the peak core.",
        ),
    )
    SIGMA_RATIO_END: float = field(
        default=0.90,
        metadata=_fc_meta(
            "Sigma Ratio End",
            "Guess Generation",
            "Sigma Estimation",
            "Raw key: SIGMA_RATIO_END\n"
            "Upper bound of the peak-height fractions used when estimating sigma. Values near 1.0 focus on the "
            "very top of the peak and are less sensitive to overlapping wings.",
        ),
    )
    SIGMA_RATIO_STEP: float = field(
        default=0.05,
        metadata=_fc_meta(
            "Sigma Ratio Step",
            "Guess Generation",
            "Sigma Estimation",
            "Raw key: SIGMA_RATIO_STEP\n"
            "Step size used between Sigma Ratio Start and Sigma Ratio End. With the default settings, the "
            "estimator samples 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, and 0.90.",
        ),
    )
    MIN_ABSOLUTE_PEAK_HEIGHT: float = field(
        default=2.0e-15,
        metadata=_fc_meta(
            "Min Absolute Height",
            "Guess Generation",
            "Post-Filters",
            "Raw key: MIN_ABSOLUTE_PEAK_HEIGHT\n"
            "Absolute floor below which detected peaks are discarded after initial peak finding. Compare this "
            "against the quiet-region noise sigma shown in the preview if you want to think of it as an n-sigma cut.",
        ),
    )
    MIN_PROMINENCE: float = field(
        default=1e-18,
        metadata=_fc_meta(
            "Min Absolute Prominence",
            "Guess Generation",
            "Post-Filters",
            "Raw key: MIN_PROMINENCE\n"
            "Absolute prominence floor applied after the relative prominence filter. Useful for removing tiny "
            "numerical bumps.",
        ),
    )
    AMPLITUDE_SCALE_LIMIT: float = field(
        default=3.0,
        metadata=_fc_meta(
            "Amplitude Scale Limit",
            "Post-Guess Fitting",
            "Final Fit Bounds",
            "Raw key: AMPLITUDE_SCALE_LIMIT\n"
            "Scales the allowed amplitude range during final fitting. Larger values give the optimizer more freedom "
            "to move amplitudes away from the initial guess.",
        ),
    )

    MAX_GC: int = field(
        default=6,
        metadata=_fc_meta(
            "Max Components",
            "Post-Guess Fitting",
            "Model Search",
            "Raw key: MAX_GC\n"
            "Maximum number of strongest guessed components retained before brute-force model generation. This also "
            "caps the largest transition subset considered during fitting. Increasing it can raise runtime very quickly.",
        ),
    )
    MIN_GC: int = field(
        default=1,
        metadata=_fc_meta(
            "Min Components",
            "Post-Guess Fitting",
            "Model Search",
            "Raw key: MIN_GC\n"
            "Minimum number of transitions included when brute-force model subsets are generated. This does not delete "
            "individual guessed peaks; it simply skips fit candidates smaller than this subset size.",
        ),
    )

    DELTA_CTR: float = field(
        default=100,
        metadata=_fc_meta(
            "Final Fit Center Delta",
            "Post-Guess Fitting",
            "Final Fit Bounds",
            "Raw key: DELTA_CTR\n"
            "Allowed change in fitted center around the initial guess during final fitting: center +/- DELTA_CTR.",
        ),
    )
    DELTA_SIGMA: float = field(
        default=100,
        metadata=_fc_meta(
            "Final Fit Sigma Delta",
            "Post-Guess Fitting",
            "Final Fit Bounds",
            "Raw key: DELTA_SIGMA\n"
            "Allowed change in fitted sigma around the initial guess during final fitting: sigma +/- DELTA_SIGMA.",
        ),
    )

    def to_string(self) -> str:
        """Return a formatted string representation of the config."""
        lines = ["FitConfig:"]
        for key, value in asdict(self).items():
            if key.startswith("_FitConfig__"):
                continue
            lines.append(f"  {key} = {value}")
        return "\n".join(lines)

    def print(self):
        print(self.to_string())

    @property
    def PROMINENCE_PERECENT(self) -> float:
        """Backward-compatible alias for the old misspelled field name."""
        return float(self.PROMINENCE_PERCENT)

    @PROMINENCE_PERECENT.setter
    def PROMINENCE_PERECENT(self, value: float) -> None:
        self.PROMINENCE_PERCENT = float(value)

    def _guess_cache_signature(self) -> tuple:
        """
        Fields that change the auto-generated peak guesses.

        Post-guess fitting controls such as DELTA_CTR and DELTA_SIGMA are deliberately
        excluded: they should not force a new preview unless they also alter peak
        detection/pairing. If a new FitConfig field changes the preview, add it here.
        """
        return (
            self.WINDOW_LENGTH,
            self.POLYORDER,
            self.HEIGHT_THRESHOLD,
            self.PROMINENCE_PERCENT,
            self.DISTANCE,
            self.MERGE_DX,
            self.MAX_SIGMA,
            self.MIN_PEAK_X_DISTANCE,
            self.SIGMA_RATIO_START,
            self.SIGMA_RATIO_END,
            self.SIGMA_RATIO_STEP,
            self.MIN_ABSOLUTE_PEAK_HEIGHT,
            self.MIN_PROMINENCE,
            self.MAX_GC,
        )

    def set_current_peaks(self, peaks, inp_peaks: [], source_mode: str = "auto"):
        # Cache the accepted preview so clicking "Accept Current Peaks" does not
        # regenerate guesses behind the user's back. The signature guards against
        # stale auto guesses after relevant FitConfig edits.
        self.__current_peaks = list(peaks or [])
        self.__current_imp_peaks = list(inp_peaks or [])
        self.__current_peak_signature = self._guess_cache_signature()
        self.__current_source_mode = str(source_mode or "auto")

    def clear_current_peaks(self, preserve_input_peaks: bool = True):
        # Manual input peaks can survive auto-guess invalidation; full reset clears
        # them when preserve_input_peaks is False.
        self.__current_peaks = None
        if not preserve_input_peaks:
            self.__current_imp_peaks = None
            self.__current_source_mode = "auto"
        self.__current_peak_signature = None

    def current_peaks_match_guess_config(self) -> bool:
        return (
            self.__current_peaks is not None and
            self.__current_peak_signature == self._guess_cache_signature()
        )

    def get_current_peaks(self):
        return (
            list(self.__current_peaks or []),
            list(self.__current_imp_peaks or []),
            self.__current_source_mode,
        )
