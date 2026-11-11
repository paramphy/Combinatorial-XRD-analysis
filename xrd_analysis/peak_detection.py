
"""Peak detection for combinatorial XRD patterns."""

import numpy as np
import pandas as pd

from scipy.signal import (
    find_peaks,
    peak_widths,
    savgol_filter,
)

from xrd_analysis.utils import position_key

from xrd_analysis.plotters import (
    prepare_intensity,
    make_common_grid,
)


# ============================================================
# SAVITZKY-GOLAY SMOOTHING
# ============================================================

def smooth_xrd_pattern(
    intensity,
    window_length=111,
    polyorder=2,
):
    """
    Smooth an XRD pattern using a Savitzky-Golay filter.

    The input data are not modified.
    """

    intensity = np.asarray(intensity, dtype=float)

    if len(intensity) < 3:
        return intensity.copy()

    window_length = int(window_length)
    polyorder = int(polyorder)

    if window_length < 3:
        raise ValueError(
            "Savitzky-Golay window_length must be at least 3."
        )

    # Ensure an odd window length.
    if window_length % 2 == 0:
        window_length += 1

    # Limit the window to the available signal length.
    max_window = len(intensity)

    if max_window % 2 == 0:
        max_window -= 1

    window_length = min(window_length, max_window)

    if window_length <= polyorder:
        return intensity.copy()

    return savgol_filter(
        intensity,
        window_length=window_length,
        polyorder=polyorder,
        mode="interp",
    )


# ============================================================
# LOCAL NOISE ESTIMATION
# ============================================================

def estimate_local_noise(
    intensity,
    peak_index,
    window_points=21,
):
    """
    Estimate local noise using the MAD of first differences.

    This is a diagnostic estimate, not a formal uncertainty.

    It is calculated from the raw intensity data.
    """

    intensity = np.asarray(intensity, dtype=float)

    window_points = max(5, int(window_points))

    half_window = window_points // 2

    start = max(0, peak_index - half_window)
    end = min(
        len(intensity),
        peak_index + half_window + 1,
    )

    local_intensity = intensity[start:end]

    if len(local_intensity) < 4:
        return np.nan

    differences = np.diff(local_intensity)

    median_difference = np.median(differences)

    mad = np.median(
        np.abs(differences - median_difference)
    )

    # Robust standard deviation estimate.
    sigma = (
        1.4826 * mad / np.sqrt(2)
    )

    return float(sigma)


# ============================================================
# PEAK DETECTION
# ============================================================

def detect_peaks(
    x,
    y,
    detection_config=None,
):
    """
    Detect candidate experimental XRD peaks.

    Processing:
        1. Clean and sort input data.
        2. Smooth intensity using Savitzky-Golay.
        3. Detect peaks using prominence.
        4. Calculate peak widths and diagnostic properties.

    No candidate peaks are removed using width or
    local-noise criteria.

    Returns
    -------
    pandas.DataFrame
        Candidate peak positions and their properties.
    """

    if detection_config is None:
        detection_config = {}

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # --------------------------------------------------------
    # 1. Clean input data
    # --------------------------------------------------------

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return pd.DataFrame()

    order = np.argsort(x)

    x = x[order]
    y = y[order]

    # Remove duplicate 2-theta values.
    x_unique, unique_indices = np.unique(
        x,
        return_index=True,
    )

    x = x_unique
    y = y[unique_indices]

    if len(x) < 3:
        return pd.DataFrame()

    # Check that the 2-theta axis is increasing.
    x_differences = np.diff(x)

    if np.any(x_differences <= 0):
        raise ValueError(
            "2-theta values must be strictly increasing."
        )

    step_size = float(np.median(x_differences))

    # --------------------------------------------------------
    # 2. Savitzky-Golay smoothing
    # --------------------------------------------------------

    smoothing_enabled = detection_config.get(
        "smoothing_enabled",
        True,
    )

    if smoothing_enabled:

        # Support both the new YAML names and the old
        # parameter names for backward compatibility.
        smoothing_window = detection_config.get(
            "window_length",
            detection_config.get(
                "smoothing_window",
                111,
            ),
        )

        smoothing_polyorder = detection_config.get(
            "polyorder",
            detection_config.get(
                "smoothing_polyorder",
                2,
            ),
        )

        smoothed_y = smooth_xrd_pattern(
            y,
            window_length=smoothing_window,
            polyorder=smoothing_polyorder,
        )

    else:
        smoothed_y = y.copy()

    # --------------------------------------------------------
    # 3. Prominence threshold
    # --------------------------------------------------------

    intensity_range = float(np.ptp(smoothed_y))

    prominence_fraction = float(
        detection_config.get(
            "prominence_fraction",
            0.05,
        )
    )

    minimum_prominence = float(
        detection_config.get(
            "minimum_prominence",
            0.0,
        )
    )

    if prominence_fraction < 0:
        raise ValueError(
            "prominence_fraction cannot be negative."
        )

    if minimum_prominence < 0:
        raise ValueError(
            "minimum_prominence cannot be negative."
        )

    prominence_threshold = max(
        intensity_range * prominence_fraction,
        minimum_prominence,
    )

    # --------------------------------------------------------
    # 4. Peak detection
    # --------------------------------------------------------

    # distance=1 imposes no meaningful separation restriction:
    # adjacent samples can still be detected as candidates.
    min_distance_points = int(
        detection_config.get(
            "min_distance_points",
            1,
        )
    )

    if min_distance_points < 1:
        raise ValueError(
            "min_distance_points must be at least 1."
        )

    peaks, properties = find_peaks(
        smoothed_y,
        prominence=prominence_threshold,
        distance=min_distance_points,
    )

    if len(peaks) == 0:
        return pd.DataFrame()

    # --------------------------------------------------------
    # 5. Peak widths
    # --------------------------------------------------------

    widths, width_heights, left_ips, right_ips = peak_widths(
        smoothed_y,
        peaks,
        rel_height=0.5,
    )

    sample_indices = np.arange(len(x))

    left_positions = np.interp(
        left_ips,
        sample_indices,
        x,
    )

    right_positions = np.interp(
        right_ips,
        sample_indices,
        x,
    )

    width_degrees = (
        right_positions - left_positions
    )

    # --------------------------------------------------------
    # 6. Relative prominence
    # --------------------------------------------------------

    if intensity_range > 0:

        relative_prominence = (
            properties["prominences"] / intensity_range
        )

    else:

        relative_prominence = np.zeros(
            len(peaks),
            dtype=float,
        )

    # --------------------------------------------------------
    # 7. Local noise diagnostics
    # --------------------------------------------------------

    local_noise_window = detection_config.get(
        "local_noise_window_points",
        21,
    )

    local_noise = []
    prominence_to_noise = []

    for i, peak_index in enumerate(peaks):

        noise_sigma = estimate_local_noise(
            y,
            peak_index,
            window_points=local_noise_window,
        )

        local_noise.append(noise_sigma)

        if np.isfinite(noise_sigma) and noise_sigma > 0:

            prominence_to_noise.append(
                properties["prominences"][i] / noise_sigma
            )

        else:

            prominence_to_noise.append(np.nan)

    # --------------------------------------------------------
    # 8. Construct results
    # --------------------------------------------------------

    results = pd.DataFrame({

        "two_theta_deg": x[peaks],

        # Raw measured intensity.
        "intensity": y[peaks],

        # Smoothed intensity.
        "smoothed_intensity": smoothed_y[peaks],

        # Prominence-based detection properties.
        "prominence": properties["prominences"],

        "relative_prominence": relative_prominence,

        # Peak width at half prominence.
        "width_deg": width_degrees,

        "width_samples": widths,

        # 2-theta positions at half prominence.
        "left_width_deg": left_positions,

        "right_width_deg": right_positions,

        # Additional diagnostics.
        "local_noise_sigma": local_noise,

        "prominence_to_noise": prominence_to_noise,

        "detection_step_deg": step_size,

    })

    # IMPORTANT:
    # Do not apply minimum_width_deg filtering here.
    # All detected candidates are retained.

    return results.reset_index(drop=True)


# ============================================================
# DETECT PEAKS IN ALL DATASETS AND POSITIONS
# ============================================================

def detect_dataset_peaks(
    datasets,
    positions,
    config,
):
    """
    Detect peaks in every dataset at every measurement position.

    Returns
    -------
    pandas.DataFrame
        Combined peak table for all experimental patterns.
    """

    detection_config = config.get(
        "peak_detection",
        {},
    )

    plot_config = config.get(
        "plot",
        {},
    )

    processing_config = config.get(
        "processing",
        {},
    )

    xlim = plot_config.get("xlim")

    x_grid = make_common_grid(
        datasets,
        positions,
        xlim,
    )

    all_peaks = []

    for dataset in datasets:

        dataset_label = dataset["label"]

        print(
            f"\nDetecting peaks: {dataset_label}"
        )

        for position in positions:

            if position not in dataset["data"]:

                print(
                    f"  Warning: missing position "
                    f"{position}"
                )

                continue

            x, y = prepare_intensity(
                dataset["data"][position],
                processing_config,
                x_grid,
            )

            peaks = detect_peaks(
                x,
                y,
                detection_config,
            )

            if peaks.empty:

                print(
                    f"  {position_key(position)}: "
                    "No peaks detected"
                )

                continue

            peaks["condition"] = dataset_label

            peaks["position"] = position_key(position)

            peaks["phi_deg"] = position[0]

            peaks["x_mm"] = position[1]

            peaks["y_mm"] = position[2]

            all_peaks.append(peaks)

            print(
                f"  {position_key(position)}: "
                f"{len(peaks)} peaks detected"
            )

    if not all_peaks:
        return pd.DataFrame()

    result = pd.concat(
        all_peaks,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # 9. Organize output columns
    # --------------------------------------------------------

    columns = [

        "condition",
        "position",

        "phi_deg",
        "x_mm",
        "y_mm",

        "two_theta_deg",

        "intensity",
        "smoothed_intensity",

        "prominence",
        "relative_prominence",

        "width_deg",
        "width_samples",

        "left_width_deg",
        "right_width_deg",

        "local_noise_sigma",
        "prominence_to_noise",

        "detection_step_deg",

    ]

    result = result[columns]

    return result
