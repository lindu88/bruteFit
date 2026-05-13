from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations


def normalize_source(source: str) -> str:
    normalized = str(source).strip().lower()
    if normalized in {"abs", "absorption", "uvvis"}:
        return "abs"
    if normalized in {"mcd"}:
        return "mcd"
    raise ValueError(f"Unknown peak source '{source}'. Expected 'abs' or 'mcd'.")


@dataclass(frozen=True)
class PeakGuess:
    """
    One guessed component before final fitting.

    amplitude is the model parameter used by lmfit. height is the visual peak height
    shown to the user. Keeping both avoids the old ambiguity where user-entered peak
    "amplitudes" were really eyeballed heights.
    """
    source: str
    center: float
    amplitude: float
    sigma: float
    height: float | None = None
    origin: str = "auto"
    label: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "source", normalize_source(self.source))
        object.__setattr__(self, "center", float(self.center))
        object.__setattr__(self, "amplitude", float(self.amplitude))
        object.__setattr__(self, "sigma", float(self.sigma))
        if self.height is not None:
            object.__setattr__(self, "height", float(self.height))
        object.__setattr__(self, "origin", str(self.origin))


@dataclass(frozen=True)
class TransitionGuess:
    """
    Cross-modal transition object.

    A transition may be paired, ABS-only, or MCD-only. We keep unpaired peaks through
    fitting so the model can still describe real spectral features, but downstream
    A/D and B/D ratios are only reported for paired transitions.
    """
    transition_id: str
    abs_peak: PeakGuess | None = None
    mcd_peak: PeakGuess | None = None
    match_distance: float | None = None

    @property
    def status(self) -> str:
        if self.abs_peak is not None and self.mcd_peak is not None:
            return "paired"
        if self.abs_peak is not None:
            return "abs_only"
        if self.mcd_peak is not None:
            return "mcd_only"
        return "empty"

    @property
    def sort_center(self) -> float:
        if self.abs_peak is not None:
            return self.abs_peak.center
        if self.mcd_peak is not None:
            return self.mcd_peak.center
        return 0.0

    @property
    def ratio_label(self) -> str | None:
        if self.mcd_peak is None or self.abs_peak is None:
            return None
        if self.mcd_peak.label in {"A", "B"}:
            return f"{self.mcd_peak.label}/D"
        return "A/D or B/D"


@dataclass(frozen=True)
class TransitionModelSpec:
    """
    The bridge between preview transitions and fitted lmfit parameter prefixes.

    This lets output code map fitted parameters such as A1_amplitude and D1_amplitude
    back to the same logical transition for ratio reporting.
    """
    transition_id: str
    status: str
    abs_peak: PeakGuess | None
    mcd_peak: PeakGuess | None
    match_distance: float | None
    abs_prefix: str | None = None
    mcd_prefix: str | None = None
    ratio_label: str | None = None


def sort_and_reindex_transitions(transitions: list[TransitionGuess]) -> list[TransitionGuess]:
    """Sort transitions by x position and assign stable T1/T2/... labels."""
    ordered = sorted(transitions, key=lambda transition: transition.sort_center)
    return [
        replace(transition, transition_id=f"T{i}")
        for i, transition in enumerate(ordered, start=1)
        if transition.status != "empty"
    ]


def count_transition_statuses(transitions: list[TransitionGuess]) -> dict[str, int]:
    counts = {"paired": 0, "abs_only": 0, "mcd_only": 0}
    for transition in transitions:
        status = transition.status
        if status in counts:
            counts[status] += 1
    return counts


def remove_transition_peak(
    transitions: list[TransitionGuess],
    transition_id: str,
    source: str,
) -> list[TransitionGuess]:
    """Remove only the clicked ABS or MCD peak while preserving the other side."""
    source = normalize_source(source)
    updated: list[TransitionGuess] = []
    for transition in transitions:
        if transition.transition_id != transition_id:
            updated.append(transition)
            continue

        abs_peak = transition.abs_peak
        mcd_peak = transition.mcd_peak
        if source == "abs":
            abs_peak = None
        else:
            mcd_peak = None

        if abs_peak is None and mcd_peak is None:
            continue

        match_distance = transition.match_distance if abs_peak is not None and mcd_peak is not None else None
        updated.append(
            replace(
                transition,
                abs_peak=abs_peak,
                mcd_peak=mcd_peak,
                match_distance=match_distance,
            )
        )
    return sort_and_reindex_transitions(updated)


def pair_peak_guesses(
    abs_peaks: list[PeakGuess],
    mcd_peaks: list[PeakGuess],
    merge_dx: float,
) -> list[TransitionGuess]:
    """
    Pair ABS and MCD peak guesses by nearest center within MERGE_DX.

    Important collaborator note:
    Previous logic treated nearby MCD guesses as duplicates of ABS guesses and dropped
    them outright. That implicitly erased the ABS<->MCD pairing information needed to
    interpret A/D or B/D values at the transition level.

    The revised logic still uses ABS as the anchoring modality conceptually, but it
    preserves both guessed peak sets and performs an explicit one-to-one cross-modal
    pairing step within MERGE_DX. Nearby peaks are therefore interpreted as the same
    transition when appropriate, while unmatched ABS and MCD peaks remain visible for
    fitting and reproducibility review.
    """

    abs_list = [peak for peak in abs_peaks if normalize_source(peak.source) == "abs"]
    mcd_list = [peak for peak in mcd_peaks if normalize_source(peak.source) == "mcd"]

    candidate_pairs: list[tuple[float, int, int]] = []
    for abs_index, abs_peak in enumerate(abs_list):
        for mcd_index, mcd_peak in enumerate(mcd_list):
            distance = abs(abs_peak.center - mcd_peak.center)
            if distance <= float(merge_dx):
                candidate_pairs.append((distance, abs_index, mcd_index))

    # Greedy nearest-neighbor pairing is intentionally simple and reviewable. If two
    # possible pairs compete for the same ABS or MCD peak, the closest pair wins and
    # the remaining peak becomes unpaired. More sophisticated matching can be added
    # later without changing the TransitionGuess representation.
    candidate_pairs.sort(key=lambda item: (item[0], abs_list[item[1]].center, mcd_list[item[2]].center))

    matched_abs: set[int] = set()
    matched_mcd: set[int] = set()
    transitions: list[TransitionGuess] = []

    for distance, abs_index, mcd_index in candidate_pairs:
        if abs_index in matched_abs or mcd_index in matched_mcd:
            continue
        matched_abs.add(abs_index)
        matched_mcd.add(mcd_index)
        mcd_peak = mcd_list[mcd_index]
        transitions.append(
            TransitionGuess(
                transition_id="",
                abs_peak=abs_list[abs_index],
                mcd_peak=mcd_peak,
                match_distance=float(distance),
            )
        )

    for abs_index, abs_peak in enumerate(abs_list):
        if abs_index not in matched_abs:
            transitions.append(
                TransitionGuess(
                    transition_id="",
                    abs_peak=abs_peak,
                    mcd_peak=None,
                    match_distance=None,
                )
            )

    for mcd_index, mcd_peak in enumerate(mcd_list):
        if mcd_index not in matched_mcd:
            transitions.append(
                TransitionGuess(
                    transition_id="",
                    abs_peak=None,
                    mcd_peak=mcd_peak,
                    match_distance=None,
                )
            )

    return sort_and_reindex_transitions(transitions)


def estimate_total_fit_count(
    transitions: list[TransitionGuess],
    min_gc: int,
    max_gc: int,
) -> int:
    """
    Estimate how many brute-force fit candidates the current preview will generate.

    For each subset of transitions, every MCD peak can be tried as either A-like or
    B-like. ABS peaks are always D-like Gaussian components.
    """
    if not transitions:
        return 0

    total = 0
    max_subset = min(int(max_gc), len(transitions))
    min_subset = max(1, int(min_gc))
    for subset_size in range(min_subset, max_subset + 1):
        for subset in combinations(transitions, subset_size):
            has_abs = any(transition.abs_peak is not None for transition in subset)
            has_mcd = any(transition.mcd_peak is not None for transition in subset)
            if not has_abs or not has_mcd:
                continue
            mcd_choice_count = sum(1 for transition in subset if transition.mcd_peak is not None)
            total += 2 ** mcd_choice_count
    return total
