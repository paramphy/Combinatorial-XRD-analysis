
import numpy as np
import pandas as pd


CATEGORIES = {
    "CuS": "CuS-associated reflection",
    "ZnS": "ZnS-associated reflection",
    "overlap": "Potential CuS/ZnS overlap",
    "unmatched": "Unmatched experimental reflection",
}


def _nearest_reference(exp_position, reference_df, tolerance):
    """
    Find the nearest theoretical reflection within tolerance.

    Returns the nearest matching reflection, or None.
    """

    if reference_df is None or reference_df.empty:
        return None

    differences = np.abs(
        reference_df["two_theta"].to_numpy(dtype=float)
        - exp_position
    )

    nearest_index = int(np.argmin(differences))
    difference = float(differences[nearest_index])

    if difference > tolerance:
        return None

    match = reference_df.iloc[nearest_index]

    return {
        "two_theta": float(match["two_theta"]),
        "delta": difference,
        "hkl": str(match["hkl"]),
        "intensity": float(match["intensity"]),
        "d_spacing": float(match["d_spacing"]),
    }


def classify_experimental_peaks(
    experimental_peaks,
    reference_generator,
    config,
):
    """
    Associate experimental peaks with theoretical CuS and ZnS
    reflections.

    Each experimental peak is compared independently against
    the complete calculated CuS and ZnS reflection lists.
    """

    if experimental_peaks.empty:
        return pd.DataFrame()

    if reference_generator is None:
        raise ValueError(
            "ReferencePatternGenerator is required for classification."
        )

    references = getattr(
        reference_generator,
        "all_references",
        {},
    )

    if "CuS" not in references or "ZnS" not in references:
        raise ValueError(
            "Complete CuS and ZnS reference patterns are required. "
            "Check that both CIF files were loaded and calculated."
        )

    classification_config = config.get(
        "peak_classification", {}
    )

    tolerance = float(
        classification_config.get(
            "match_tolerance_deg", 0.20
        )
    )

    # Optional theoretical relative-intensity threshold.
    # Default 0 means use every calculated reflection.
    minimum_reference_fraction = float(
        classification_config.get(
            "minimum_reference_intensity_fraction", 0.0
        )
    )

    if tolerance <= 0:
        raise ValueError(
            "match_tolerance_deg must be positive."
        )

    if minimum_reference_fraction < 0:
        raise ValueError(
            "minimum_reference_intensity_fraction cannot be negative."
        )

    if minimum_reference_fraction > 1:
        raise ValueError(
            "minimum_reference_intensity_fraction cannot exceed 1."
        )

    # Prepare the complete theoretical reflection lists.
    phase_references = {}

    for phase in ("CuS", "ZnS"):

        frame = references[phase].copy()

        if not frame.empty and minimum_reference_fraction > 0:

            maximum = frame["intensity"].max()

            if maximum > 0:
                frame = frame[
                    frame["intensity"]
                    >= maximum * minimum_reference_fraction
                ]

        phase_references[phase] = frame

    results = []

    for _, experimental_peak in experimental_peaks.iterrows():

        exp_position = float(
            experimental_peak["two_theta_deg"]
        )

        # Find the closest theoretical reflection for each phase.
        cus_match = _nearest_reference(
            exp_position,
            phase_references["CuS"],
            tolerance,
        )

        zns_match = _nearest_reference(
            exp_position,
            phase_references["ZnS"],
            tolerance,
        )

        # Classify the experimental peak.
        if cus_match is not None and zns_match is not None:

            category = CATEGORIES["overlap"]

        elif cus_match is not None:

            category = CATEGORIES["CuS"]

        elif zns_match is not None:

            category = CATEGORIES["ZnS"]

        else:

            category = CATEGORIES["unmatched"]

        result = experimental_peak.to_dict()

        result.update({
            "classification": category,

            "CuS_match_2theta_deg": (
                cus_match["two_theta"]
                if cus_match else np.nan
            ),

            "CuS_delta_deg": (
                cus_match["delta"]
                if cus_match else np.nan
            ),

            "CuS_hkl": (
                cus_match["hkl"]
                if cus_match else ""
            ),

            "CuS_d_spacing": (
                cus_match["d_spacing"]
                if cus_match else np.nan
            ),

            "ZnS_match_2theta_deg": (
                zns_match["two_theta"]
                if zns_match else np.nan
            ),

            "ZnS_delta_deg": (
                zns_match["delta"]
                if zns_match else np.nan
            ),

            "ZnS_hkl": (
                zns_match["hkl"]
                if zns_match else ""
            ),

            "ZnS_d_spacing": (
                zns_match["d_spacing"]
                if zns_match else np.nan
            ),

            "match_tolerance_deg": tolerance,
        })

        if cus_match is not None and zns_match is not None:
            result["reference_separation_deg"] = abs(
                cus_match["two_theta"]
                - zns_match["two_theta"]
            )
        else:
            result["reference_separation_deg"] = np.nan

        results.append(result)

    return pd.DataFrame(results)


def classification_summary(classified_peaks):
    """
    Count classified peaks by condition and category.
    """

    if classified_peaks.empty:
        return pd.DataFrame()

    summary = (
        classified_peaks
        .groupby(
            ["condition", "classification"],
            dropna=False
        )
        .size()
        .unstack(fill_value=0)
    )

    for category in CATEGORIES.values():
        if category not in summary.columns:
            summary[category] = 0

    return summary[list(CATEGORIES.values())]
