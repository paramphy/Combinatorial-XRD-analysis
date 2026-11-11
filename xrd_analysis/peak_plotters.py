
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from xrd_analysis.plotters import prepare_intensity
from xrd_analysis.utils import position_key

from xrd_analysis.plotters import (
    prepare_intensity,
    make_common_grid,
)


CATEGORY_COLORS = {
    "CuS-associated reflection": "#2685D8",
    "ZnS-associated reflection": "#D84A4A",
    "Potential CuS/ZnS overlap": "#E5A12C",
    "Unmatched experimental reflection": "#888888",
}


def plot_peak_classification(
    dataset,
    position,
    classified_peaks,
    config,
    reference_generator=None,
):
    """
    Plot one experimental pattern with classified peak markers.

    CuS and ZnS theoretical positions are also shown as
    separate reference sticks at the bottom of the plot.
    """

    plot_config = config.get("plot", {})
    processing_config = config.get("processing", {})

    #x, y = prepare_intensity(
    #    dataset["data"][position],
    #    processing_config,
    #)

    x_grid = make_common_grid(
        [dataset],
        [position],
        plot_config.get("xlim"),
    )

    x, y = prepare_intensity(
        dataset["data"][position],
        processing_config,
        x_grid,
    )

    condition = dataset["label"]
    pos_label = position_key(position)

    peaks = classified_peaks[
        (classified_peaks["condition"] == condition)
        & (classified_peaks["position"] == pos_label)
    ].copy()

    fig, ax = plt.subplots(figsize=(11, 6))

    # Experimental pattern.
    ax.plot(
        x,
        y,
        color=dataset.get("color", "black"),
        linewidth=1.0,
        label="Experimental pattern",
        zorder=2,
    )

    # Plot detected experimental peaks by classification.
    for category, color in CATEGORY_COLORS.items():

        category_peaks = peaks[
            peaks["classification"] == category
        ]

        if category_peaks.empty:
            continue

        peak_x = category_peaks[
            "two_theta_deg"
        ].to_numpy()

        peak_y = np.interp(
            peak_x,
            x,
            y,
        )

        ax.scatter(
            peak_x,
            peak_y,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            s=55,
            zorder=5,
            label=category,
        )

    # Draw matched theoretical positions at the bottom.
    if reference_generator is not None:

        all_references = getattr(
            reference_generator,
            "all_references",
            {},
        )

        xlim = plot_config.get("xlim")

        for phase, color, y0, y1 in [
            ("CuS", "#2685D8", 0.02, 0.07),
            ("ZnS", "#D84A4A", 0.09, 0.14),
        ]:

            reference = all_references.get(phase)

            if reference is None or reference.empty:
                continue

            if xlim:
                reference = reference[
                    reference["two_theta"].between(
                        xlim[0], xlim[1]
                    )
                ]

            # Only show theoretical reflections associated
            # with the experimental peaks in this pattern.
            matched_positions = []

            for column in [
                f"{phase}_match_2theta_deg"
            ]:

                if column in peaks.columns:
                    matched_positions = (
                        peaks[column]
                        .dropna()
                        .unique()
                    )

            if len(matched_positions) == 0:
                continue

            matched_positions = np.asarray(
                matched_positions,
                dtype=float,
            )

            reference = reference[
                reference["two_theta"].apply(
                    lambda value: np.any(
                        np.isclose(
                            matched_positions,
                            value,
                            atol=1e-6,
                        )
                    )
                )
            ]

            if reference.empty:
                continue

            ax.vlines(
                reference["two_theta"],
                y0,
                y1,
                transform=ax.get_xaxis_transform(),
                color=color,
                linewidth=1.5,
                alpha=0.9,
                zorder=4,
            )

    # Add an explicit classification legend.
    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=7,
            label=category,
        )
        for category, color in CATEGORY_COLORS.items()
    ]

    legend_handles.insert(
        0,
        Line2D(
            [0], [0],
            color=dataset.get("color", "black"),
            linewidth=1.2,
            label="Experimental pattern",
        )
    )

    ax.legend(
        handles=legend_handles,
        fontsize=8,
        frameon=False,
        loc="best",
    )

    ax.set_xlabel(r"2$\theta$ (degrees)")
    ax.set_ylabel("Processed intensity (counts)")

    ax.set_title(
        f"Experimental peak classification\n"
        f"{condition} | {pos_label}"
    )

    if xlim:
        ax.set_xlim(xlim)

    ax.grid(alpha=0.2)
    fig.tight_layout()

    return fig



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_classification_summary(
    classified_peaks,
    condition_order=None,
    position_order=None
):
    """
    Plot phase classification percentages for all
    experimental XRD patterns.

    Each bar represents one experimental pattern.

    Total peak counts are annotated above each bar.
    """

    df = classified_peaks.copy()

    categories = [
        "CuS-associated reflection",
        "ZnS-associated reflection",
        "Potential CuS/ZnS overlap",
        "Unmatched experimental reflection"
    ]

    colors = [
        "#2685D8",
        "#D84A4A",
        "#E5A12C",
        "#888888"
    ]

    # Preserve the order in which conditions and
    # positions first appear, unless explicitly supplied.
    if condition_order is None:
        condition_order = list(pd.unique(df["condition"]))

    if position_order is None:
        position_order = list(pd.unique(df["position"]))

    # Count peaks for each experimental pattern
    counts = (
        df.groupby(
            ["condition", "position", "classification"],
            sort=False
        )
        .size()
        .unstack(fill_value=0)
    )

    # Include all four categories, even if a category
    # has zero detected peaks.
    counts = counts.reindex(
        columns=categories,
        fill_value=0
    )

    # Build the required measurement order
    ordered_index = [
        (condition, position)
        for condition in condition_order
        for position in position_order
        if (condition, position) in counts.index
    ]

    counts = counts.reindex(ordered_index)

    # Calculate total peaks and percentages
    totals = counts.sum(axis=1)

    percentages = counts.div(
        totals.replace(0, np.nan),
        axis=0
    ) * 100

    percentages = percentages.fillna(0)

    # ------------------------------------------------
    # Plot
    # ------------------------------------------------

    fig, ax = plt.subplots(figsize=(19, 7))

    x = np.arange(len(counts))
    bottom = np.zeros(len(counts))

    for category, color in zip(categories, colors):

        values = percentages[category].values

        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.85,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            label=category
        )

        bottom += values

    # ------------------------------------------------
    # Total peak count annotations
    # ------------------------------------------------

    for i, total in enumerate(totals):

        ax.text(
            i,
            102,
            f"{total}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90
        )

    # ------------------------------------------------
    # Position labels
    # ------------------------------------------------

    position_labels = [
        str(position)
        for condition, position in counts.index
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(
        position_labels,
        rotation=90,
        fontsize=8
    )

    # ------------------------------------------------
    # Condition group separators and labels
    # ------------------------------------------------

    start = 0

    for condition in condition_order:

        n_positions = sum(
            1 for c, p in counts.index if c == condition
        )

        if n_positions == 0:
            continue

        end = start + n_positions - 1

        if start > 0:
            ax.axvline(
                start - 0.5,
                color="black",
                linewidth=1.0,
                alpha=0.5
            )

        midpoint = (start + end) / 2

        ax.text(
            midpoint,
            -19,
            str(condition),
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold"
        )

        start += n_positions

    # ------------------------------------------------
    # Formatting
    # ------------------------------------------------

    ax.set_ylim(0, 112)

    ax.set_ylabel(
        "Percentage of detected experimental peaks (%)",
        fontsize=11
    )

    ax.set_xlabel(
        "Measurement position (phi)",
        fontsize=11,
        labelpad=35
    )

    ax.set_title(
        "Spatial distribution of experimental XRD peak classifications",
        fontsize=14,
        pad=18
    )

    ax.set_yticks(np.arange(0, 101, 20))

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.set_axisbelow(True)

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )

    fig.tight_layout()

    return fig


def plot_classification_heatmap(
    classified_peaks,
    condition_order=None,
    position_order=None
):
    """
    Generate a heatmap showing the percentage distribution
    of experimentally detected XRD peak classifications
    across all measurement positions.

    Each column represents one XRD pattern.
    Each row represents one classification category.
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = classified_peaks.copy()

    categories = [
        "CuS-associated reflection",
        "ZnS-associated reflection",
        "Potential CuS/ZnS overlap",
        "Unmatched experimental reflection"
    ]

    # Determine ordering
    if condition_order is None:
        condition_order = list(pd.unique(df["condition"]))

    if position_order is None:
        position_order = list(pd.unique(df["position"]))

    # Count classified peaks
    counts = (
        df.groupby(
            ["condition", "position", "classification"],
            sort=False
        )
        .size()
        .unstack(fill_value=0)
    )

    counts = counts.reindex(
        columns=categories,
        fill_value=0
    )

    # Arrange all measurements in condition/position order
    ordered_index = [
        (condition, position)
        for condition in condition_order
        for position in position_order
        if (condition, position) in counts.index
    ]

    counts = counts.reindex(ordered_index)

    # Calculate total peaks and percentages
    totals = counts.sum(axis=1)

    percentages = (
        counts.div(
            totals.replace(0, np.nan),
            axis=0
        ) * 100
    ).fillna(0)

    # Transpose: categories on Y-axis, measurements on X-axis
    heatmap_data = percentages.T

    # Full measurement labels
    measurement_labels = [
        str(position)
        for condition, position in counts.index
    ]

    # ------------------------------------------------
    # Plot
    # ------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(22, 6)
    )

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={
            "label": "Detected peaks (%)"
        },
        ax=ax
    )

    # Full position labels
    ax.set_xticklabels(
        measurement_labels,
        rotation=90,
        fontsize=8
    )

    ax.set_yticklabels(
        [
            "CuS",
            "ZnS",
            "CuS + ZnS overlap",
            "Unmatched"
        ],
        rotation=0
    )

    ax.set_xlabel("Measurement position")
    ax.set_ylabel("Peak classification")

    ax.set_title(
        "Spatial distribution of XRD peak classifications",
        fontsize=14,
        pad=15
    )

    # ------------------------------------------------
    # Condition separators
    # ------------------------------------------------

    start = 0

    for condition in condition_order:

        n_positions = sum(
            1 for c, p in counts.index
            if c == condition
        )

        if n_positions == 0:
            continue

        end = start + n_positions

        if start > 0:
            ax.axvline(
                start,
                color="red",
                linewidth=1.5
            )

        start = end

    fig.tight_layout()

    return fig