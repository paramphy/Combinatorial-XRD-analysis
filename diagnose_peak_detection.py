
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import (
    savgol_filter,
    find_peaks,
    peak_prominences,
    peak_widths,
)


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {

    # Input file
    "input_file": "GI-XRD_Combi_ZnCu_1min\\CuZn1min_260211 phi0(0,-7).ASC",

    # Output directory
    "output_dir": "diagnostic_output",

    # Data columns (zero-based)
    # Change these according to your ASC file format.
    "two_theta_column": 0,
    "intensity_column": 1,

    # Savitzky-Golay smoothing
    "smoothing_enabled": True,
    "window_length": 111,
    "polyorder": 2,

    # Peak detection
    # Prominence is expressed in intensity units.
    "prominence_fraction": 0.05,  # Fraction of the intensity range

    # Minimum separation between detected peaks.
    # None means no minimum separation constraint.
    "min_distance_points": 1,

    # Local noise estimation
    "local_noise_window_points": 21,

    # Plot settings
    "show_plots": True,
    "save_plots": True,
}


# ============================================================
# DATA READING
# ============================================================

def load_xrd_data(filepath):

    """
    Load a two-column XRD ASCII file.

    Assumes whitespace- or tab-separated numeric data.
    Modify this function if your ASC file has a different
    header or data structure.
    """

    data = pd.read_csv(
        filepath,
        sep=r"\s+",
        comment="#",
        header=None,
        engine="python"
    )

    x = pd.to_numeric(
        data.iloc[:, CONFIG["two_theta_column"]],
        errors="coerce"
    ).to_numpy()

    y = pd.to_numeric(
        data.iloc[:, CONFIG["intensity_column"]],
        errors="coerce"
    ).to_numpy()

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    order = np.argsort(x)

    x = x[order]
    y = y[order]

    if len(x) < 5:
        raise ValueError("Not enough valid XRD data points.")

    if np.any(np.diff(x) <= 0):
        raise ValueError(
            "2-theta values must be unique and strictly increasing."
        )

    return x, y


# ============================================================
# SMOOTHING
# ============================================================

def smooth_pattern(y, window_length, polyorder):

    if window_length % 2 == 0:
        window_length += 1

    if window_length > len(y):
        window_length = len(y) if len(y) % 2 else len(y) - 1

    if window_length <= polyorder:
        raise ValueError(
            "Smoothing window must be larger than polynomial order."
        )

    return savgol_filter(
        y,
        window_length=window_length,
        polyorder=polyorder
    )


# ============================================================
# LOCAL NOISE ESTIMATION
# ============================================================

def estimate_local_noise(y, peak_index, window_points=21):

    """
    Estimate local noise using the MAD of first differences.

    The factor converts MAD to a robust standard deviation
    estimate for approximately Gaussian noise.

    This is a diagnostic estimate, not a formal uncertainty.
    """

    half_window = window_points // 2

    start = max(0, peak_index - half_window)
    end = min(len(y), peak_index + half_window + 1)

    local_y = y[start:end]

    if len(local_y) < 4:
        return np.nan

    differences = np.diff(local_y)

    median_difference = np.median(differences)

    mad = np.median(
        np.abs(differences - median_difference)
    )

    sigma = 1.4826 * mad / np.sqrt(2)

    return sigma


# ============================================================
# PEAK DETECTION AND CHARACTERISATION
# ============================================================

def detect_and_characterise_peaks(x, raw_y, smoothed_y):

    """
    Detect candidate peaks on the smoothed pattern.

    All candidates are retained. No peak is discarded
    based on its width or quality assessment.
    """

    # Prominence threshold relative to the overall
    # intensity range of the smoothed signal.
    intensity_range = np.ptp(smoothed_y)

    prominence_threshold = (
        CONFIG["prominence_fraction"] * intensity_range
    )

    # Optional minimum distance in data points.
    distance = CONFIG["min_distance_points"]

    peaks, properties = find_peaks(
        smoothed_y,
        prominence=prominence_threshold,
        distance=distance
    )

    if len(peaks) == 0:
        return pd.DataFrame()

    # Calculate prominence and width.
    prominences, left_bases, right_bases = peak_prominences(
        smoothed_y,
        peaks
    )

    widths, width_heights, left_ips, right_ips = peak_widths(
        smoothed_y,
        peaks,
        rel_height=0.5
    )

    # Convert fractional sample indices into 2-theta values.
    sample_indices = np.arange(len(x))

    left_positions = np.interp(
        left_ips,
        sample_indices,
        x
    )

    right_positions = np.interp(
        right_ips,
        sample_indices,
        x
    )

    width_degrees = right_positions - left_positions

    records = []

    for i, peak_index in enumerate(peaks):

        local_noise = estimate_local_noise(
            smoothed_y,
            peak_index,
            CONFIG["local_noise_window_points"]
        )

        snr = (
            prominences[i] / local_noise
            if local_noise > 0
            else np.nan
        )

        records.append({

            "peak_index": peak_index,

            "two_theta": x[peak_index],

            "raw_intensity": raw_y[peak_index],

            "smoothed_intensity": smoothed_y[peak_index],

            "prominence": prominences[i],

            "width_points": widths[i],

            "width_degrees": width_degrees[i],

            "local_noise_sigma": local_noise,

            "prominence_to_noise": snr,

            "left_base_two_theta": x[left_bases[i]],

            "right_base_two_theta": x[right_bases[i]],

        })

    peaks_df = pd.DataFrame(records)

    # Provisional diagnostic flags only.
    # These do NOT remove peaks.
    narrow_width_threshold = 2.0 * np.median(np.diff(x))

    peaks_df["quality_flag"] = np.where(
        peaks_df["width_degrees"] < narrow_width_threshold,
        "Potentially narrow / spike-like",
        "Width not flagged"
    )

    return peaks_df


# ============================================================
# PLOT 1: RAW + SMOOTHED + DETECTED PEAKS
# ============================================================

def plot_pattern(x, raw_y, smoothed_y, peaks_df, output_dir):

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        x,
        raw_y,
        color="grey",
        alpha=0.55,
        linewidth=0.8,
        label="Raw data"
    )

    ax.plot(
        x,
        smoothed_y,
        color="blue",
        linewidth=1.2,
        label="Savitzky-Golay smoothed"
    )

    if not peaks_df.empty:

        ax.scatter(
            peaks_df["two_theta"],
            peaks_df["smoothed_intensity"],
            color="red",
            marker="x",
            s=45,
            label="Detected candidates",
            zorder=5
        )

        # Annotate peak positions.
        for _, peak in peaks_df.iterrows():

            ax.annotate(
                f'{peak["two_theta"]:.2f}',
                (
                    peak["two_theta"],
                    peak["smoothed_intensity"]
                ),
                xytext=(0, 8),
                textcoords="offset points",
                fontsize=7,
                rotation=45,
                ha="center"
            )

    ax.set_xlabel(r"2$\theta$ (degrees)")
    ax.set_ylabel("Intensity (counts)")

    ax.set_title("XRD Peak Detection Diagnostic")

    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()

    if CONFIG["save_plots"]:
        fig.savefig(
            output_dir / "01_peak_detection.png",
            dpi=300,
            bbox_inches="tight"
        )

    return fig


# ============================================================
# PLOT 2: PROMINENCE VS WIDTH
# ============================================================

def plot_peak_properties(peaks_df, output_dir):

    if peaks_df.empty:
        print("No peaks detected. Skipping property plot.")
        return None

    fig, ax = plt.subplots(figsize=(9, 6))

    flagged = peaks_df[
        peaks_df["quality_flag"] == "Potentially narrow / spike-like"
    ]

    normal = peaks_df[
        peaks_df["quality_flag"] == "Width not flagged"
    ]

    ax.scatter(
        normal["width_degrees"],
        normal["prominence"],
        label="Width not flagged",
        s=45
    )

    ax.scatter(
        flagged["width_degrees"],
        flagged["prominence"],
        label="Potentially narrow / spike-like",
        marker="x",
        s=65
    )

    ax.set_xlabel(r"Peak width ($^\circ$ 2$\theta$)")
    ax.set_ylabel("Peak prominence")

    ax.set_title("Peak Prominence vs Peak Width")

    ax.set_yscale("log")

    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    if CONFIG["save_plots"]:
        fig.savefig(
            output_dir / "02_peak_properties.png",
            dpi=300,
            bbox_inches="tight"
        )

    return fig


# ============================================================
# MAIN
# ============================================================

def main():

    input_file = Path(CONFIG["input_file"])

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_file}")

    x, raw_y = load_xrd_data(input_file)

    print(f"Number of data points: {len(x)}")

    print(
        f"2-theta range: {x.min():.3f} - {x.max():.3f} degrees"
    )

    if CONFIG["smoothing_enabled"]:

        smoothed_y = smooth_pattern(
            raw_y,
            CONFIG["window_length"],
            CONFIG["polyorder"]
        )

    else:
        smoothed_y = raw_y.copy()

    peaks_df = detect_and_characterise_peaks(
        x,
        raw_y,
        smoothed_y
    )

    # Export processed signal.
    pd.DataFrame({

        "two_theta": x,
        "raw_intensity": raw_y,
        "smoothed_intensity": smoothed_y

    }).to_csv(
        output_dir / "processed_pattern.csv",
        index=False
    )

    # Export all candidate peaks.
    peaks_df.to_csv(
        output_dir / "detected_peak_candidates.csv",
        index=False
    )

    print("\nPeak detection summary")
    print("----------------------")
    print(f"Detected candidates: {len(peaks_df)}")

    if not peaks_df.empty:

        print(
            peaks_df[
                [
                    "two_theta",
                    "prominence",
                    "width_degrees",
                    "prominence_to_noise",
                    "quality_flag"
                ]
            ].to_string(index=False)
        )

        print("\nQuality flag counts:")
        print(peaks_df["quality_flag"].value_counts())

    fig1 = plot_pattern(
        x,
        raw_y,
        smoothed_y,
        peaks_df,
        output_dir
    )

    fig2 = plot_peak_properties(
        peaks_df,
        output_dir
    )

    if CONFIG["show_plots"]:

        plt.show()

    else:

        plt.close(fig1)

        if fig2 is not None:
            plt.close(fig2)

    print(f"\nResults saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
