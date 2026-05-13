from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
from scipy.signal import savgol_filter

from .gaussianModels import (
    component_peak_height,
    stable_gaussian_derivative_sigma,
    stable_gaussian_sigma,
)
from .transition_matching import PeakGuess

DEFAULT_MCD_GUESS_STRATEGY = "ab_decomposition"

# MCD initial guessing is deliberately strategy-based. The current strategy assumes
# each ABS transition is a plausible local anchor and asks, "does the MCD nearby look
# more like an A term, a B term, or neither?" Future alternatives such as SNR-weighted
# matching or sinc-like smoothers should be added as new strategy functions rather
# than woven directly into the GUI code.
_AB_DECOMP_WINDOW_SIGMAS = 3.0
_AB_DECOMP_MIN_POINTS = 7
_AB_DECOMP_CENTER_SLOP_SIGMA_FRACTION = 0.35
_AB_DECOMP_CENTER_SLOP_MAX_DX = 8.0
_AB_DECOMP_SIGMA_SCALE_FACTORS = (0.75, 0.90, 1.00, 1.10, 1.25)


def _debug_enabled() -> bool:
    return os.environ.get("BRUTEFIT_DEBUG_GUESSES", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _debug_print(*args) -> None:
    if _debug_enabled():
        print(*args)


@dataclass(frozen=True)
class McdDecomposition:
    center: float
    sigma: float
    a_amplitude: float
    b_amplitude: float
    fitted_peak_height: float
    r_squared: float
    dominant_label: str


def _safe_dx(x) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 1.0
    diffs = np.diff(x)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 1.0
    dx = float(np.median(np.abs(diffs)))
    return dx if dx > 0 else 1.0


def _safe_sigma(sigma: float, dx: float) -> float:
    sigma = abs(float(sigma))
    return sigma if sigma > 0 else dx


def _window_slice(x, center: float, sigma: float, dx: float) -> slice:
    half_width = max(_AB_DECOMP_WINDOW_SIGMAS * sigma, (_AB_DECOMP_MIN_POINTS // 2) * dx)
    center_index = int(np.argmin(np.abs(np.asarray(x, dtype=float) - float(center))))
    radius = max(_AB_DECOMP_MIN_POINTS // 2, int(np.ceil(half_width / dx)))
    start = max(0, center_index - radius)
    stop = min(len(x), center_index + radius + 1)
    return slice(start, stop)


def _center_candidate_values(anchor_center: float, anchor_sigma: float, dx: float) -> tuple[float, ...]:
    # Experimental ABS and MCD extrema often do not land at exactly the same x value.
    # We search a small local grid instead of pinning the MCD guess exactly to the ABS
    # center. The cap in data-point units prevents broad ABS sigmas from allowing the
    # MCD anchor to wander into a neighboring transition.
    sigma = _safe_sigma(anchor_sigma, dx)
    center_radius = max(
        dx,
        min(
            _AB_DECOMP_CENTER_SLOP_SIGMA_FRACTION * sigma,
            _AB_DECOMP_CENTER_SLOP_MAX_DX * dx,
        ),
    )
    candidate_values = np.linspace(
        float(anchor_center) - center_radius,
        float(anchor_center) + center_radius,
        5,
    )
    return tuple(float(value) for value in candidate_values)


def _sigma_candidate_values(anchor_sigma: float, dx: float) -> tuple[float, ...]:
    # ABS-derived sigmas are useful but imperfect for shoulders and broad/small peaks,
    # so the local decomposition tries a compact set of sigma scale factors and lets
    # the fit quality pick the best seed.
    sigma = _safe_sigma(anchor_sigma, dx)
    candidates = [max(dx, sigma * float(scale)) for scale in _AB_DECOMP_SIGMA_SCALE_FACTORS]
    ordered_unique: list[float] = []
    for candidate in candidates:
        if any(abs(candidate - existing) <= max(1e-12, 1e-6 * candidate) for existing in ordered_unique):
            continue
        ordered_unique.append(float(candidate))
    return tuple(ordered_unique)


def _solve_local_ab_decomposition(x_local, y_local, center: float, sigma: float) -> McdDecomposition | None:
    x_local = np.asarray(x_local, dtype=float)
    y_local = np.asarray(y_local, dtype=float)
    if x_local.size < _AB_DECOMP_MIN_POINTS or y_local.size != x_local.size:
        return None

    g = stable_gaussian_sigma(x_local, amplitude=1.0, center=center, sigma=sigma)
    dg = stable_gaussian_derivative_sigma(x_local, amplitude=1.0, center=center, sigma=sigma)
    basis = np.column_stack([dg, g])
    if basis.ndim != 2 or basis.shape[0] != x_local.size:
        return None

    # With center and sigma fixed for this trial, A and B amplitudes are linear
    # coefficients. Least squares is therefore enough for the guess stage; final
    # nonlinear fitting still happens later in dataFitting.py.
    coeffs, *_ = np.linalg.lstsq(basis, y_local, rcond=None)
    a_amplitude = float(coeffs[0])
    b_amplitude = float(coeffs[1])
    fitted = basis @ coeffs

    peak_height = float(np.nanmax(np.abs(fitted))) if fitted.size else 0.0
    ss_res = float(np.sum((y_local - fitted) ** 2))
    ss_tot = float(np.sum((y_local - np.mean(y_local)) ** 2))
    r_squared = 0.0 if ss_tot <= 0 else max(0.0, 1.0 - (ss_res / ss_tot))
    dominant_label = "A" if abs(a_amplitude) >= abs(b_amplitude) else "B"

    return McdDecomposition(
        center=float(center),
        sigma=float(sigma),
        a_amplitude=a_amplitude,
        b_amplitude=b_amplitude,
        fitted_peak_height=peak_height,
        r_squared=r_squared,
        dominant_label=dominant_label,
    )


def _decomposition_score(decomposition: McdDecomposition, anchor_center: float, anchor_sigma: float):
    dominant_amplitude = (
        decomposition.a_amplitude
        if decomposition.dominant_label == "A"
        else decomposition.b_amplitude
    )
    dominant_height = abs(
        component_peak_height(
            dominant_amplitude,
            decomposition.center,
            decomposition.sigma,
            label=decomposition.dominant_label,
        )
    )
    center_delta = abs(float(decomposition.center) - float(anchor_center))
    sigma_delta = abs(float(decomposition.sigma) - float(anchor_sigma))
    return (
        # Prefer a locally good decomposition first, then a strong component, then
        # candidates that stayed closest to the ABS anchor.
        float(decomposition.r_squared),
        dominant_height,
        -center_delta,
        -sigma_delta,
    )


def _search_local_ab_decomposition(x, y_smoothed, anchor_center: float, anchor_sigma: float, dx: float):
    center_candidates = _center_candidate_values(anchor_center, anchor_sigma, dx)
    sigma_candidates = _sigma_candidate_values(anchor_sigma, dx)
    best_decomposition = None
    best_local_slice = None
    best_score = None
    evaluated_count = 0

    for center_candidate in center_candidates:
        for sigma_candidate in sigma_candidates:
            local_slice = _window_slice(x, center_candidate, sigma_candidate, dx)
            decomposition = _solve_local_ab_decomposition(
                x[local_slice],
                y_smoothed[local_slice],
                center=center_candidate,
                sigma=sigma_candidate,
            )
            evaluated_count += 1
            if decomposition is None:
                continue

            score = _decomposition_score(decomposition, anchor_center, anchor_sigma)
            if best_score is None or score > best_score:
                best_score = score
                best_decomposition = decomposition
                best_local_slice = local_slice

    return best_decomposition, best_local_slice, evaluated_count, center_candidates, sigma_candidates


def guess_mcd_peaks_ab_decomposition(x, y_mcd, abs_peaks: list[PeakGuess], fc) -> list[PeakGuess]:
    """
    Generate initial MCD peak guesses by anchoring on ABS transitions and fitting the local
    MCD signal to a linear combination of Gaussian-derivative (A-term-like) and Gaussian
    (B-term-like) basis functions.

    This is intentionally split into its own strategy function so future MCD-specific guess
    logic can be swapped in without rewriting the dialog or transition-pairing flow.
    """
    if not abs_peaks:
        return []

    x = np.asarray(x, dtype=float)
    y_mcd = np.asarray(y_mcd, dtype=float)
    if x.size == 0 or y_mcd.size != x.size:
        return []

    y_smoothed = savgol_filter(y_mcd, window_length=fc.WINDOW_LENGTH, polyorder=fc.POLYORDER)
    dx = _safe_dx(x)
    global_scale = max(float(np.nanmax(np.abs(y_smoothed))) if y_smoothed.size else 0.0, 1e-30)
    # This is still a global MCD height screen. It intentionally mirrors the existing
    # relative peak-picking controls, but the debug output can reveal rejected anchors
    # where a future local-noise/SNR threshold would be more appropriate.
    height_threshold = max(float(fc.MIN_ABSOLUTE_PEAK_HEIGHT), float(fc.HEIGHT_THRESHOLD) * global_scale)
    _debug_print(
        "MCD A+B decomposition thresholds:",
        f"global_scale={global_scale:.6g}",
        f"height_threshold={height_threshold:.6g}",
        f"min_prominence={float(fc.MIN_PROMINENCE):.6g}",
        f"window_sigmas={_AB_DECOMP_WINDOW_SIGMAS:.3g}",
        f"center_slop_frac={_AB_DECOMP_CENTER_SLOP_SIGMA_FRACTION:.3g}",
        f"center_slop_max_dx={_AB_DECOMP_CENTER_SLOP_MAX_DX:.3g}",
        f"sigma_scales={list(_AB_DECOMP_SIGMA_SCALE_FACTORS)}",
    )

    ranked_candidates: list[tuple[float, PeakGuess, McdDecomposition]] = []
    for abs_peak in abs_peaks:
        anchor_sigma = _safe_sigma(abs_peak.sigma, dx)
        decomposition, local_slice, evaluated_count, center_candidates, sigma_candidates = _search_local_ab_decomposition(
            x,
            y_smoothed,
            anchor_center=float(abs_peak.center),
            anchor_sigma=anchor_sigma,
            dx=dx,
        )
        if decomposition is None:
            center_radius = max(abs(float(value) - float(abs_peak.center)) for value in center_candidates)
            _debug_print(
                "MCD A+B anchor REJECT:",
                f"abs_center={float(abs_peak.center):.6g}",
                f"abs_amp={float(abs_peak.amplitude):.6g}",
                f"abs_sigma={anchor_sigma:.6g}",
                f"center_slop_radius={center_radius:.6g}",
                f"evaluated={evaluated_count}",
                "reason=insufficient_local_points",
            )
            continue

        dominant_amplitude = (
            decomposition.a_amplitude
            if decomposition.dominant_label == "A"
            else decomposition.b_amplitude
        )

        decision = "KEEP"
        reason = "accepted"
        if decomposition.fitted_peak_height < height_threshold:
            decision = "REJECT"
            reason = "below_global_height_threshold"
        elif abs(dominant_amplitude) < float(fc.MIN_PROMINENCE):
            decision = "REJECT"
            reason = "below_min_prominence_floor"

        local_points = int(max(0, local_slice.stop - local_slice.start))
        center_shift = float(decomposition.center) - float(abs_peak.center)
        sigma_scale = float(decomposition.sigma) / max(anchor_sigma, 1e-30)
        center_radius = max(abs(float(value) - float(abs_peak.center)) for value in center_candidates)
        _debug_print(
            f"MCD A+B anchor {decision}:",
            f"abs_center={float(abs_peak.center):.6g}",
            f"abs_amp={float(abs_peak.amplitude):.6g}",
            f"abs_sigma={anchor_sigma:.6g}",
            f"local_points={local_points}",
            f"fit_center={float(decomposition.center):.6g}",
            f"center_shift={center_shift:.6g}",
            f"center_slop_radius={center_radius:.6g}",
            f"fit_sigma={float(decomposition.sigma):.6g}",
            f"sigma_scale={sigma_scale:.4f}",
            f"evaluated={evaluated_count}",
            f"A_amp={decomposition.a_amplitude:.6g}",
            f"B_amp={decomposition.b_amplitude:.6g}",
            f"dominant={decomposition.dominant_label}",
            f"dominant_amp={dominant_amplitude:.6g}",
            f"height={decomposition.fitted_peak_height:.6g}",
            f"height_threshold={height_threshold:.6g}",
            f"r2={decomposition.r_squared:.4f}",
            f"reason={reason}",
        )

        if decision == "REJECT":
            continue

        # Store the dominant A/B component as the MCD seed. The non-dominant term is
        # used to classify the local shape, but only one MCD component enters the
        # brute-force candidate list for this transition.
        peak = PeakGuess(
            source="mcd",
            center=float(decomposition.center),
            amplitude=float(dominant_amplitude),
            sigma=float(decomposition.sigma),
            height=component_peak_height(
                dominant_amplitude,
                decomposition.center,
                decomposition.sigma,
                label=decomposition.dominant_label,
            ),
            origin="auto",
            label=decomposition.dominant_label,
        )
        ranked_candidates.append((decomposition.fitted_peak_height, peak, decomposition))

    ranked_candidates.sort(key=lambda item: item[0], reverse=True)
    ranked_candidates = ranked_candidates[: max(0, int(fc.MAX_GC))]

    guesses = [peak for _, peak, _ in ranked_candidates]

    _debug_print("Initial Guess Peak Centers B:", [peak.center for peak in guesses])
    _debug_print("Initial Guess Peak Sigmas B:", [peak.sigma for peak in guesses])
    _debug_print("Initial Guess Peak Heights B:", [peak.height for peak in guesses])
    _debug_print("Initial Guess Peak Amplitudes B:", [peak.amplitude for peak in guesses])
    _debug_print("Initial Guess Peak Labels B:", [peak.label for peak in guesses])
    for _, peak, decomposition in ranked_candidates:
        _debug_print(
            "MCD A+B decomposition:",
            f"center={peak.center:.6g}",
            f"sigma={peak.sigma:.6g}",
            f"A_amp={decomposition.a_amplitude:.6g}",
            f"B_amp={decomposition.b_amplitude:.6g}",
            f"dominant={decomposition.dominant_label}",
            f"height={decomposition.fitted_peak_height:.6g}",
            f"r2={decomposition.r_squared:.4f}",
        )

    return guesses


_MCD_GUESS_STRATEGIES = {
    DEFAULT_MCD_GUESS_STRATEGY: guess_mcd_peaks_ab_decomposition,
}


def guess_mcd_peaks(x, y_mcd, abs_peaks: list[PeakGuess], fc, strategy_name: str = DEFAULT_MCD_GUESS_STRATEGY):
    """Dispatch MCD guessing through a named strategy for future swappability."""
    try:
        strategy = _MCD_GUESS_STRATEGIES[str(strategy_name)]
    except KeyError as exc:
        available = ", ".join(sorted(_MCD_GUESS_STRATEGIES))
        raise ValueError(f"Unknown MCD guess strategy '{strategy_name}'. Available: {available}") from exc
    return strategy(x, y_mcd, abs_peaks, fc)
