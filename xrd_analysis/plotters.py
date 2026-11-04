
"""All experimental XRD plotting functions."""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from .utils import position_key


# ==================================================
# COMMON DATA PREPARATION
# ==================================================

def prepare_intensity(df, processing=None, x_grid=None):

    processing = processing or {}

    x = df["two_theta"].to_numpy(dtype=float)
    y = df["intensity"].to_numpy(dtype=float).copy()

    background = float(
        processing.get("subtract_constant_background", 0)
    )

    y -= background

    if processing.get("clip_negative_after_background", True):
        y = np.clip(y, 0, None)

    normalization = processing.get(
        "normalization", "none"
    ).lower()

    if normalization == "max":

        maximum = np.nanmax(y)

        if maximum > 0:
            y = y / maximum

    elif normalization != "none":

        raise ValueError(
            "Normalization must be none or max."
        )

    if x_grid is not None:

        # Do not extrapolate outside the measured range.
        y = np.interp(
            x_grid, x, y,
            left=np.nan,
            right=np.nan
        )

        x = x_grid

    return x, y


def make_common_grid(datasets, positions, xlim=None):

    df = datasets[0]["data"][positions[0]]

    x = df["two_theta"].to_numpy(dtype=float)

    if xlim:

        x = x[
            (x >= xlim[0]) &
            (x <= xlim[1])
        ]

    if len(x) < 2:
        raise ValueError(
            "Invalid 2theta range."
        )

    return x


def add_references(
    ax,
    reference_generator,
    reference_config
):

    if reference_generator is not None and reference_config:

        reference_generator.add_reference_sticks(
            ax,
            reference_config
        )

def _reference_legend_handles(reference_config):
    """Create legend handles for the theoretical reference phases."""
    if not reference_config:
        return []

    display = reference_config.get("display", {})
    if not display.get("show", True):
        return []

    handles = []

    for item in reference_config.get("references", []):
        handles.append(
            Line2D(
                [0], [0],
                color=item.get("color", "black"),
                linewidth=2,
                label=f"{item['label']} reference",
            )
        )

    return handles
# ==================================================
# 1. WATERFALL PLOT BY MEASUREMENT POSITION
# ==================================================

def plot_waterfall(
    datasets,
    positions,
    config,
    reference_generator=None,
    reference_config=None
):

    plot = config.get("plot", {})
    processing = config.get("processing", {})

    xlim = plot.get("xlim")

    x_grid = make_common_grid(
        datasets, positions, xlim
    )

    ncols = int(plot.get("ncols", 5))
    nrows = int(np.ceil(len(positions) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=plot.get("figsize", [18, 9]),
        sharex=True,
        squeeze=False
    )

    axes = axes.ravel()

    records = []

    for i, position in enumerate(positions):

        ax = axes[i]

        curves = []

        for dataset in datasets:

            x, y = prepare_intensity(
                dataset["data"][position],
                processing,
                x_grid
            )

            curves.append((dataset, x, y))

        local_max = max(
            float(np.nanmax(y)) for _, _, y in curves
        )

        offset_step = max(local_max, 1.0) * float(
            plot.get("offset_fraction", 1.15)
        )

        for condition_index, (dataset, x, y) in enumerate(curves):

            offset = condition_index * offset_step

            ax.plot(
                x,
                y + offset,
                color=dataset.get("color"),
                linewidth=plot.get("line_width", 1.15),
                label=dataset["label"]
            )

            for xx, yy in zip(x, y):

                records.append({
                    "position": position_key(position),
                    "phi_deg": position[0],
                    "x_mm": position[1],
                    "y_mm": position[2],
                    "condition": dataset["label"],
                    "two_theta_deg": xx,
                    "intensity_processed": yy,
                    "intensity_with_offset": yy + offset,
                    "vertical_offset": offset
                })

        custom_labels = config.get(
            "position_labels", {}
        )

        ax.set_title(
            custom_labels.get(
                position_key(position),
                position_key(position)
            ),
            fontsize=plot.get("title_size", 9)
        )

        ax.tick_params(
            labelsize=plot.get("tick_size", 8)
        )

        if plot.get("hide_y_ticks", True):
            ax.set_yticks([])

        if plot.get("grid", False):
            ax.grid(True, alpha=0.25)

        if xlim:
            ax.set_xlim(xlim)

        # Reference sticks must be added after the axis exists.
        add_references(
            ax,
            reference_generator,
            reference_config
        )

    for ax in axes[len(positions):]:
        ax.set_visible(False)

    fig.supxlabel(r"2$\theta$ (degrees)")
    fig.supylabel(
        "Intensity (counts) + vertical offset"
    )

    fig.suptitle(
        plot.get(
            "title",
            "Spatially resolved XRD patterns"
        )
    )

    handles, labels = axes[0].get_legend_handles_labels()

    if plot.get("show_legend", True) and handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=max(1, len(datasets) + 2),
            frameon=False,
        )
        
    fig.tight_layout(
        rect=[0.02, 0.02, 0.98, 0.91]
    )

    return fig, records


# ==================================================
# 2 AND 3. COMBINED LINEAR / LOG PLOT
# ==================================================

def plot_combined(
    datasets,
    positions,
    config,
    scale="linear",
    reference_generator=None,
    reference_config=None
):

    plot = config.get("combined_plot", {})
    processing = config.get("processing", {})

    xlim = plot.get(
        "xlim",
        config.get("plot", {}).get("xlim")
    )

    line_styles = [
        "-", "--", "-.", ":",
        (0, (5, 1)),
        (0, (3, 1, 1, 1)),
        (0, (1, 1)),
        (0, (5, 2, 1, 2)),
        (0, (3, 2)),
        (0, (2, 1))
    ]

    fig, ax = plt.subplots(
        figsize=plot.get("figsize", [16, 9])
    )

    for dataset in datasets:

        for index, position in enumerate(positions):

            x, y = prepare_intensity(
                dataset["data"][position],
                processing
            )

            if scale == "log":
                y = np.where(y > 0, y, np.nan)

            ax.plot(
                x,
                y,
                color=dataset.get("color"),
                linestyle=line_styles[
                    index % len(line_styles)
                ],
                linewidth=plot.get("line_width", 0.9),
                alpha=plot.get("alpha", 0.75)
            )

    if scale == "log":
        ax.set_yscale("log")

    # Condition legend.
    condition_handles = [
        Line2D(
            [0], [0],
            color=dataset.get("color"),
            linewidth=2,
            label=dataset["label"]
        )
        for dataset in datasets
    ]

    legend1 = ax.legend(
        handles=condition_handles,
        title="Sulfurization condition",
        loc="upper right",
        fontsize=plot.get("legend_size", 8),
        frameon=False
    )

    ax.add_artist(legend1)

    # Position legend.
    position_handles = [
        Line2D(
            [0], [0],
            color="black",
            linestyle=line_styles[i % len(line_styles)],
            linewidth=1.5,
            label=position_key(position)
        )
        for i, position in enumerate(positions)
    ]

    ax.legend(
        handles=position_handles,
        title="Measurement position",
        loc="upper left",
        fontsize=plot.get("position_legend_size", 7),
        ncol=2,
        frameon=False
    )

    reference_handles = _reference_legend_handles(
        reference_config
    )

    if reference_handles:
        reference_legend = ax.legend(
            handles=reference_handles,
            title="Theoretical reference peaks",
            loc="lower center",
            fontsize=plot.get("legend_size", 8),
            frameon=False,
            ncol=len(reference_handles),
        )
        ax.add_artist(reference_legend)

        ax.set_xlabel(r"2$\theta$ (degrees)")

    if scale == "log":

        ax.set_ylabel(
            "Intensity (counts, logarithmic scale)"
        )

        title = plot.get(
            "log_title",
            "Combined XRD patterns - Logarithmic scale"
        )

    else:

        ax.set_ylabel("Intensity (counts)")

        title = plot.get(
            "linear_title",
            "Combined XRD patterns - Linear scale"
        )

    ax.set_title(title)

    if xlim:
        ax.set_xlim(xlim)

    ax.grid(
        True,
        which="both" if scale == "log" else "major",
        alpha=0.2
    )

    add_references(
        ax,
        reference_generator,
        reference_config
    )

    fig.tight_layout()

    return fig


# ==================================================
# 4 AND 5. FOUR-CONDITION SPATIAL COMPARISON
# ==================================================

def plot_condition_grid(
    datasets,
    positions,
    config,
    use_offset=True,
    reference_generator=None,
    reference_config=None
):

    if len(datasets) != 4:
        raise ValueError(
            "Condition-wise 2x2 plot requires exactly four datasets."
        )

    plot = config.get("condition_grid", {})
    processing = config.get("processing", {})

    xlim = plot.get(
        "xlim",
        config.get("plot", {}).get("xlim")
    )

    x_grid = make_common_grid(
        datasets, positions, xlim
    )

    # Prepare all curves and determine global intensity.
    all_curves = []
    global_max = 0.0

    for dataset in datasets:

        condition_curves = []

        for position in positions:

            x, y = prepare_intensity(
                dataset["data"][position],
                processing,
                x_grid
            )

            condition_curves.append(
                (position, x, y)
            )

            global_max = max(
                global_max,
                float(np.nanmax(y))
            )

        all_curves.append(condition_curves)

    # Common offset spacing across all four conditions.
    offset_step = max(global_max, 1.0) * float(
        plot.get("offset_fraction", 1.15)
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=plot.get("figsize", [16, 12]),
        sharex=True,
        sharey=True
    )

    axes = axes.ravel()

    colors = plt.get_cmap(
        plot.get("colormap", "tab10")
    )(
        np.linspace(0, 1, len(positions))
    )

    for condition_index, dataset in enumerate(datasets):

        ax = axes[condition_index]

        curves = all_curves[condition_index]

        for position_index, (position, x, y) in enumerate(curves):

            offset = (
                position_index * offset_step
                if use_offset else 0.0
            )

            ax.plot(
                x,
                y + offset,
                color=colors[position_index],
                linewidth=plot.get("line_width", 1.0),
                alpha=plot.get("alpha", 0.9)
            )

        ax.set_title(
            dataset["label"],
            fontsize=plot.get("title_size", 12)
        )

        if xlim:
            ax.set_xlim(xlim)

        ax.tick_params(
            labelsize=plot.get("tick_size", 9)
        )

        if plot.get("hide_y_ticks", True):
            ax.set_yticks([])

        if plot.get("grid", False):
            ax.grid(True, alpha=0.25)

        add_references(
            ax,
            reference_generator,
            reference_config
        )

    fig.supxlabel(r"2$\theta$ (degrees)")

    ylabel = (
        "Intensity (counts) + vertical offset"
        if use_offset
        else "Intensity (counts)"
    )

    fig.supylabel(ylabel)

    title = plot.get(
        "title",
        "Spatial XRD patterns by sulfurization condition"
    )

    if not use_offset:
        title += " (no vertical offset)"

    fig.suptitle(
        title,
        fontsize=plot.get("suptitle_size", 15),
        y=0.99
    )

    position_handles = [
        Line2D(
            [0], [0],
            color=colors[i],
            linewidth=2,
            label=position_key(position)
        )
        for i, position in enumerate(positions)
    ]

    fig.legend(
        handles=position_handles,
        title="Measurement position",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=min(5, len(positions)),
        frameon=False,
        fontsize=plot.get("legend_size", 8)
    )
    reference_handles = _reference_legend_handles(
    reference_config
)

    if reference_handles:
        fig.legend(
            handles=reference_handles,
            title="Theoretical reference peaks",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=len(reference_handles),
            frameon=False,
            fontsize=plot.get("legend_size", 8),
        )

    fig.tight_layout(
        rect=[0.03, 0.07, 0.98, 0.88]
    )

    return fig


def plot_waterfall_normalized(
    datasets,
    positions,
    config,
    reference_generator=None,
    reference_config=None
):

    plot = config.get("plot", {})
    processing = config.get("processing", {})

    xlim = plot.get("xlim")

    x_grid = make_common_grid(
        datasets, positions, xlim
    )

    ncols = int(plot.get("ncols", 5))
    nrows = int(np.ceil(len(positions) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=plot.get("figsize", [18, 9]),
        sharex=True,
        squeeze=False
    )

    axes = axes.ravel()

    records = []

    for i, position in enumerate(positions):

        ax = axes[i]

        curves = []

        # Prepare and independently normalize every curve.
        for dataset in datasets:

            x, y = prepare_intensity(
                dataset["data"][position],
                processing,
                x_grid
            )

            y_min = float(np.nanmin(y))
            y_max = float(np.nanmax(y))

            intensity_range = y_max - y_min

            if not np.isfinite(intensity_range) or intensity_range <= 0:
                y_normalized = np.zeros_like(y, dtype=float)
            else:
                y_normalized = (y - y_min) / intensity_range

            curves.append(
                (dataset, x, y, y_normalized, y_min, y_max)
            )

        # Since every curve has a maximum of 1,
        # use the same vertical offset spacing for all positions.
        offset_step = float(
            plot.get("normalized_offset_fraction", 1.15)
        )

        for condition_index, (
            dataset, x, y, y_normalized, y_min, y_max
        ) in enumerate(curves):

            offset = condition_index * offset_step

            ax.plot(
                x,
                y_normalized + offset,
                color=dataset.get("color"),
                linewidth=plot.get("line_width", 1.15),
                label=dataset["label"]
            )

            for xx, yy, yy_norm in zip(
                x, y, y_normalized
            ):

                records.append({
                    "position": position_key(position),
                    "phi_deg": position[0],
                    "x_mm": position[1],
                    "y_mm": position[2],
                    "condition": dataset["label"],
                    "two_theta_deg": xx,
                    "intensity_processed": yy,
                    "intensity_normalized": yy_norm,
                    "intensity_with_offset": yy_norm + offset,
                    "original_minimum": y_min,
                    "original_maximum": y_max,
                    "vertical_offset": offset
                })

        custom_labels = config.get(
            "position_labels", {}
        )

        ax.set_title(
            custom_labels.get(
                position_key(position),
                position_key(position)
            ),
            fontsize=plot.get("title_size", 9)
        )

        ax.tick_params(
            labelsize=plot.get("tick_size", 8)
        )

        if plot.get("hide_y_ticks", True):
            ax.set_yticks([])

        if plot.get("grid", False):
            ax.grid(True, alpha=0.25)

        if xlim:
            ax.set_xlim(xlim)

        # Add theoretical reference sticks.
        add_references(
            ax,
            reference_generator,
            reference_config
        )

    for ax in axes[len(positions):]:
        ax.set_visible(False)

    fig.supxlabel(r"2$\theta$ (degrees)")
    fig.supylabel(
        "Normalized intensity (0–1) + vertical offset"
    )

    fig.suptitle(
        plot.get(
            "normalized_title",
            "Spatially resolved XRD patterns (independently normalized)"
        )
    )

    handles, labels = axes[0].get_legend_handles_labels()

    if plot.get("show_legend", True) and handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=max(1, len(datasets) + 2),
            frameon=False,
        )

    fig.tight_layout(
        rect=[0.02, 0.02, 0.98, 0.91]
    )

    return fig, records
