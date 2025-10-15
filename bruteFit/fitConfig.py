from dataclasses import dataclass, asdict

@dataclass
class FitConfig:

    #TODO tune these parameters by optimizing over a large set of (representative) synthetic (and real?) data. 
    # estimate the amount of datapoints needed given the amount of params.

    # Smoothing
    WINDOW_LENGTH: int = 5  # Window length for Savitzky-Golay smoothing (in samples/ datapoints?) (relate to bandwidth?)
    POLYORDER: int = 4  # Polynomial order for Savitzky-Golay smoothing

    # Peak picking
    HEIGHT_THRESHOLD: float = 0.04  # Minimum height threshold for peak detection
    PROMINENCE_PERECENT: float = 0.04  # (topological) Prominence as a multiple of max height
    DISTANCE: int = 5  # Minimum distance (in samples/points) between peaks

    # Fitting
    MERGE_DX: int = 500  # Distance to merge centers between abs and mcd
    PERCENTAGE_RANGE: int = 30  # Allowed percentage relaxation for re-fitting after removing poor curves
    MAX_SIGMA: int = 60000  # Max sigma for Gaussians
    MIN_PEAK_X_DISTANCE: int = 0
    ESTIMATE_SIGMA_ITERATIONS_END: int = 10  # START/END to END-1/END
    ESTIMATE_SIGMA_ITERATIONS_START: int = 4
    MIN_ABSOLUTE_PEAK_HEIGHT: float = 2.0e-15
    MIN_PROMINENCE: float = 1e-18  # Min relative peak height
    AMPLITUDE_SCALE_LIMIT: float = 3.0 # TODO what is this? 

    MAX_GC: int = 6 #Max number of gaussian curves (or derivatives) accepted
    MIN_GC: int = 2 #minimum number of gaussians.

    DELTA_CTR: float = 10
    DELTA_SIGMA: float = 10

    def to_string(self) -> str:
        """Return a formatted string representation of the config."""
        lines = ["FitConfig:"]
        for key, value in asdict(self).items():
            lines.append(f"  {key} = {value}")
        return "\n".join(lines)

    def print(self):
        print(self.to_string())