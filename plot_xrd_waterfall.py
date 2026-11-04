
"""
Combinatorial XRD Waterfall Plotter
====================================

- Reads XRD .ASC files from ZIP archives or directories.
- Automatically extracts measurement positions from filenames.
- Matches the same measurement positions across datasets.
- Creates one subplot per measurement position.
- Plots sulfurization conditions as vertically offset curves.
- Exports PNG, PDF, SVG and processed CSV files.

Installation:
    pip install numpy pandas matplotlib pyyaml

Execution:
    python plot_xrd_waterfall.py --config xrd_config.yaml
"""

import argparse
import re
import zipfile
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

import xml.etree.ElementTree as ET

from reference_generator import ReferencePatternGenerator

def parse_position(filename):
    """
    Extract phi, x and y coordinates from filenames.

    Example:
        CuZn1min phi0(0,-15).ASC

    Returns:
        (0, 0.0, -15.0)
    """

    pattern = (
        r"phi\s*(-?\d+(?:\.\d+)?)"
        r"\s*\(\s*(-?\d+(?:\.\d+)?)"
        r"\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
    )

    match = re.search(pattern, filename, re.IGNORECASE)

    if not match:
        raise ValueError(
            f"Cannot extract coordinates from {filename}"
        )

    phi, x, y = match.groups()

    return int(float(phi)), float(x), float(y)


def position_key(position):
    """Generate a unique position identifier."""

    phi, x, y = position

    return f"phi{phi}({x:g},{y:g})"


def read_asc(data, filename):
    """
    Read ASC data.

    Supports:
        2 columns: 2theta, intensity
        3 columns: 2theta, intensity, uncertainty
    """

    if isinstance(data, bytes):
        array = np.loadtxt(BytesIO(data))
    else:
        array = np.loadtxt(data)

    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(
            f"Invalid ASC file format: {filename}"
        )

    df = pd.DataFrame({
        "two_theta": array[:, 0],
        "intensity": array[:, 1]
    })

    if array.shape[1] >= 3:
        df["uncertainty"] = array[:, 2]

    df = df.replace(
        [np.inf, -np.inf], np.nan
    ).dropna(subset=["two_theta", "intensity"])

    df = (
        df.sort_values("two_theta")
        .drop_duplicates("two_theta")
    )

    return df


def read_xrdml(data, filename):
    """
    Read a PANalytical-style XRDML file.

    Extracts:
        - 2theta positions
        - Intensity/counts

    Supports:
        - Explicit list of 2theta positions
        - Start/end positions with equally spaced points

    Returns:
        DataFrame with two_theta and intensity columns.
    """

    # Parse XML from either a file path or ZIP bytes.
    if isinstance(data, bytes):
        root = ET.fromstring(data)
    else:
        tree = ET.parse(data)
        root = tree.getroot()

    # Ignore XML namespaces when searching tags.
    def local_name(element):
        return element.tag.split("}")[-1].lower()

    def find_all(element, tag):
        return [
            item for item in element.iter()
            if local_name(item) == tag.lower()
        ]

    def get_text(element, tag):
        elements = find_all(element, tag)

        if elements and elements[0].text:
            return elements[0].text.strip()

        return None

    def parse_numbers(text):
        if not text:
            return np.array([], dtype=float)

        return np.fromstring(
            text.replace(",", " "),
            sep=" ",
            dtype=float
        )

    # Locate diffraction data blocks.
    data_points = find_all(root, "dataPoints")

    if not data_points:
        raise ValueError(
            f"No dataPoints element found in {filename}"
        )

    # Use the first data block.
    # Multi-scan XRDML support can be added separately.
    block = data_points[0]

    # Extract intensity data.
    intensity_elements = (
        find_all(block, "intensities")
        or find_all(block, "counts")
    )

    if not intensity_elements:
        raise ValueError(
            f"No intensity data found in {filename}"
        )

    intensity = parse_numbers(
        intensity_elements[0].text
    )

    if len(intensity) == 0:
        raise ValueError(
            f"Could not parse intensity data in {filename}"
        )

    # Find the 2theta position definition.
    axis_element = None

    for element in block.iter():

        if local_name(element) == "positions":

            axis = element.attrib.get(
                "axis", ""
            ).lower()

            if "2theta" in axis or "2-theta" in axis:
                axis_element = element
                break

    if axis_element is None:
        raise ValueError(
            f"Could not identify the 2theta axis in {filename}"
        )

    # First try explicit position list.
    position_list = get_text(
        axis_element, "listPositions"
    )

    if position_list:

        two_theta = parse_numbers(position_list)

    else:

        # Otherwise use start/end positions.
        start_text = get_text(
            axis_element, "startPosition"
        )

        end_text = get_text(
            axis_element, "endPosition"
        )

        if start_text is None or end_text is None:
            raise ValueError(
                f"Missing 2theta range in {filename}"
            )

        start = float(start_text)
        end = float(end_text)

        if len(intensity) == 1:
            two_theta = np.array([start])

        else:
            two_theta = np.linspace(
                start,
                end,
                len(intensity)
            )

    # Validate data dimensions.
    if len(two_theta) != len(intensity):

        raise ValueError(
            f"Position/intensity length mismatch in {filename}: "
            f"{len(two_theta)} positions, "
            f"{len(intensity)} intensities."
        )

    df = pd.DataFrame({
        "two_theta": two_theta,
        "intensity": intensity
    })

    df = df.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    df = (
        df.sort_values("two_theta")
        .drop_duplicates("two_theta")
    )

    if len(df) < 2:
        raise ValueError(
            f"Insufficient diffraction data in {filename}"
        )

    return df



def load_dataset(dataset_config):
    """
    Load XRD data from ZIP archive or directory.

    Priority for each measurement position:
        1. ASC
        2. XRDML

    Different positions may use different formats.
    """

    source = Path(dataset_config["path"])

    if not source.exists():
        raise FileNotFoundError(source)

    # Collect candidate files.
    files = []

    if source.is_dir():

        for extension in ("*.ASC", "*.asc", "*.xrdml", "*.XRDML"):
            files.extend(source.glob(extension))

    elif zipfile.is_zipfile(source):

        with zipfile.ZipFile(source, "r") as archive:

            files = [
                name for name in archive.namelist()
                if name.lower().endswith(
                    (".asc", ".xrdml")
                )
            ]

            # Group files by measurement position.
            candidates = {}

            for filename in sorted(files):

                basename = Path(filename).name
                position = parse_position(basename)

                candidates.setdefault(
                    position, []
                ).append(filename)

            results = {}

            for position, filenames in candidates.items():

                # ASC has priority over XRDML.
                selected = sorted(
                    filenames,
                    key=lambda f: (
                        0 if f.lower().endswith(".asc") else 1
                    )
                )[0]

                if selected.lower().endswith(".asc"):

                    df = read_asc(
                        archive.read(selected),
                        selected
                    )

                else:

                    df = read_xrdml(
                        archive.read(selected),
                        selected
                    )

                results[position] = df

            if not results:
                raise ValueError(
                    f"No ASC or XRDML files found in {source}"
                )

            print(
                f"Loaded {len(results)} measurement positions "
                f"from {source.name}"
            )

            return results

    else:

        raise ValueError(
            "Input must be a ZIP archive or directory."
        )

    # Directory-based loading.
    candidates = {}

    for file in files:

        position = parse_position(file.name)

        candidates.setdefault(
            position, []
        ).append(file)

    results = {}

    for position, filenames in candidates.items():

        selected = sorted(
            filenames,
            key=lambda f: (
                0 if f.suffix.lower() == ".asc" else 1
            )
        )[0]

        if selected.suffix.lower() == ".asc":

            df = read_asc(
                selected,
                selected.name
            )

        else:

            df = read_xrdml(
                selected,
                selected.name
            )

        results[position] = df

    if not results:
        raise ValueError(
            f"No ASC or XRDML files found in {source}"
        )

    print(
        f"Loaded {len(results)} measurement positions "
        f"from {source.name}"
    )

    return results



def prepare_intensity(df, processing, x_grid=None):
    """
    Apply optional preprocessing and interpolate
    onto a common 2theta grid.
    """

    x = df["two_theta"].to_numpy(dtype=float)
    y = df["intensity"].to_numpy(dtype=float)

    background = processing.get(
        "subtract_constant_background", 0
    )

    y = y - background

    if processing.get(
        "clip_negative_after_background", True
    ):
        y = np.clip(y, 0, None)

    normalization = processing.get(
        "normalization", "none"
    ).lower()

    if normalization == "max":

        maximum = np.max(y)

        if maximum > 0:
            y = y / maximum

    elif normalization != "none":

        raise ValueError(
            "Normalization must be none or max."
        )

    if x_grid is not None:

        y = np.interp(x_grid, x, y)
        x = x_grid

    return x, y


def parse_position_order(value):

    match = re.fullmatch(
        r"phi(-?\d+(?:\.\d+)?)"
        r"\((-?\d+(?:\.\d+)?),"
        r"(-?\d+(?:\.\d+)?)\)",
        value.replace(" ", ""),
        re.IGNORECASE
    )

    if not match:
        raise ValueError(
            f"Invalid position: {value}"
        )

    phi, x, y = match.groups()

    return int(float(phi)), float(x), float(y)


def plot_all_data_linear(datasets, positions, config, reference_generator=None, reference_config=None):
    """
    Plot all XRD patterns in one figure using a linear
    intensity scale.

    Colour = sulfurization condition
    Line style = measurement position
    """
    if reference_generator is not None:

        reference_generator.add_reference_sticks(
            ax,
            reference_config
        )

    plot = config.get("combined_plot", {})

    fig, ax = plt.subplots(
        figsize=plot.get("figsize", [14, 8])
    )

    line_styles = [
        "-", "--", "-.", ":",
        (0, (5, 1)), (0, (3, 1, 1, 1)),
        (0, (1, 1)), (0, (5, 2, 1, 2)),
        (0, (3, 2)), (0, (2, 1))
    ]

    processing = config.get("processing", {})

    # Use the same measurement colour for all positions
    # belonging to one sulfurization condition.

    for condition_index, dataset in enumerate(datasets):

        for position_index, position in enumerate(positions):

            df = dataset["data"][position]

            x, y = prepare_intensity(
                df,
                processing
            )

            ax.plot(
                x,
                y,
                color=dataset["color"],
                linestyle=line_styles[
                    position_index % len(line_styles)
                ],
                linewidth=plot.get("line_width", 0.9),
                alpha=plot.get("alpha", 0.75)
            )

    # Legend 1: sulfurization conditions
    from matplotlib.lines import Line2D

    condition_handles = [
        Line2D(
            [0], [0],
            color=ds["color"],
            linewidth=2,
            label=ds["label"]
        )
        for ds in datasets
    ]

    legend1 = ax.legend(
        handles=condition_handles,
        title="Sulfurization condition",
        loc="upper right",
        fontsize=plot.get("legend_size", 8)
    )

    ax.add_artist(legend1)

    # Legend 2: measurement positions
    position_handles = [
        Line2D(
            [0], [0],
            color="black",
            linestyle=line_styles[i % len(line_styles)],
            linewidth=1.5,
            label=position_key(pos)
        )
        for i, pos in enumerate(positions)
    ]

    ax.legend(
        handles=position_handles,
        title="Measurement position",
        loc="upper left",
        fontsize=plot.get("legend_size", 7),
        ncol=2
    )

    ax.set_xlabel(
        r"2$\theta$ (degrees)",
        fontsize=12
    )

    ax.set_ylabel(
        "Intensity (counts)",
        fontsize=12
    )

    ax.set_title(
        plot.get(
            "linear_title",
            "Combined XRD patterns - Linear scale"
        ),
        fontsize=14
    )

    ax.set_xlim(
        plot.get("xlim", config["plot"].get("xlim"))
    )

    ax.grid(
        True,
        alpha=0.2
    )

    fig.tight_layout()

    return fig


def plot_all_data_log(datasets, positions, config, reference_generator=None, reference_config=None):
    """
    Plot all XRD patterns in one figure using a
    logarithmic intensity scale.

    Colour = sulfurization condition
    Line style = measurement position
    """
    if reference_generator is not None:

        reference_generator.add_reference_sticks(
            ax,
            reference_config
        )

    plot = config.get("combined_plot", {})

    fig, ax = plt.subplots(
        figsize=plot.get("figsize", [14, 8])
    )

    line_styles = [
        "-", "--", "-.", ":",
        (0, (5, 1)), (0, (3, 1, 1, 1)),
        (0, (1, 1)), (0, (5, 2, 1, 2)),
        (0, (3, 2)), (0, (2, 1))
    ]

    processing = config.get("processing", {})

    for condition_index, dataset in enumerate(datasets):

        for position_index, position in enumerate(positions):

            df = dataset["data"][position]

            x, y = prepare_intensity(
                df,
                processing
            )

            # Logarithmic plots cannot display zero or
            # negative intensity values.
            y = np.where(y > 0, y, np.nan)

            ax.plot(
                x,
                y,
                color=dataset["color"],
                linestyle=line_styles[
                    position_index % len(line_styles)
                ],
                linewidth=plot.get("line_width", 0.9),
                alpha=plot.get("alpha", 0.75)
            )

    from matplotlib.lines import Line2D

    condition_handles = [
        Line2D(
            [0], [0],
            color=ds["color"],
            linewidth=2,
            label=ds["label"]
        )
        for ds in datasets
    ]

    legend1 = ax.legend(
        handles=condition_handles,
        title="Sulfurization condition",
        loc="upper right",
        fontsize=plot.get("legend_size", 8)
    )

    ax.add_artist(legend1)

    position_handles = [
        Line2D(
            [0], [0],
            color="black",
            linestyle=line_styles[i % len(line_styles)],
            linewidth=1.5,
            label=position_key(pos)
        )
        for i, pos in enumerate(positions)
    ]

    ax.legend(
        handles=position_handles,
        title="Measurement position",
        loc="upper left",
        fontsize=plot.get("legend_size", 7),
        ncol=2
    )

    ax.set_yscale("log")

    ax.set_xlabel(
        r"2$\theta$ (degrees)",
        fontsize=12
    )

    ax.set_ylabel(
        "Intensity (counts, log scale)",
        fontsize=12
    )

    ax.set_title(
        plot.get(
            "log_title",
            "Combined XRD patterns - Logarithmic scale"
        ),
        fontsize=14
    )

    ax.set_xlim(
        plot.get("xlim", config["plot"].get("xlim"))
    )

    ax.grid(
        True,
        which="both",
        alpha=0.2
    )

    fig.tight_layout()

    return fig


def plot_by_sulfurization_condition(datasets, positions, config, reference_generator=None,
    reference_config=None):
    """
    Create a 2x2 figure.

    Each panel represents one sulfurization condition.
    Each panel contains 10 vertically offset XRD patterns.

    Colour represents measurement position.
    All four panels use a common intensity scale and
    common vertical offset spacing.
    """

    if reference_generator is not None:

        reference_generator.add_reference_sticks(
            ax,
            reference_config
        )

    plot = config.get("condition_grid", {})
    processing = config.get("processing", {})

    n_conditions = len(datasets)

    if n_conditions != 4:
        raise ValueError(
            "The condition-wise 2x2 plot requires exactly "
            "four sulfurization datasets."
        )

    # ---------------------------------------------
    # Common 2theta grid
    # ---------------------------------------------

    first_df = datasets[0]["data"][positions[0]]

    x_grid = first_df["two_theta"].to_numpy()

    xlim = plot.get(
        "xlim",
        config.get("plot", {}).get("xlim")
    )

    if xlim:
        mask = (
            (x_grid >= xlim[0]) &
            (x_grid <= xlim[1])
        )
        x_grid = x_grid[mask]

    if len(x_grid) < 2:
        raise ValueError(
            "Invalid 2theta range for condition-wise plot."
        )

    # ---------------------------------------------
    # Prepare all curves and determine global scale
    # ---------------------------------------------

    all_curves = []
    global_max = 0

    for dataset in datasets:

        condition_curves = []

        for position in positions:

            df = dataset["data"][position]

            x, y = prepare_intensity(
                df,
                processing,
                x_grid
            )

            condition_curves.append(
                (position, x, y)
            )

            global_max = max(
                global_max,
                float(np.max(y))
            )

        all_curves.append(condition_curves)

    if global_max <= 0:
        global_max = 1.0

    # Same offset spacing in all four panels.
    offset_step = (
        global_max *
        float(plot.get("offset_fraction", 1.15))
    )

    # ---------------------------------------------
    # Create 2x2 layout
    # ---------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=plot.get("figsize", [16, 12]),
        sharex=True,
        sharey=True
    )

    axes = axes.ravel()

    # Ten distinct colours for ten measurement positions.
    colors = plt.get_cmap(
        plot.get("colormap", "tab10")
    )(
        np.linspace(0, 1, len(positions))
    )

    # ---------------------------------------------
    # Plot each sulfurization condition
    # ---------------------------------------------

    for condition_index, dataset in enumerate(datasets):

        ax = axes[condition_index]

        condition_curves = all_curves[condition_index]

        for position_index, (position, x, y) in enumerate(
            condition_curves
        ):

            if plot.get("use_offset", True):

                offset = position_index * offset_step

            else:

                offset = 0

            plotted_y = y + offset

            ax.plot(
                x,
                plotted_y,
                color=colors[position_index],
                linewidth=plot.get("line_width", 1.0),
                alpha=plot.get("alpha", 0.9)
            )

        ax.set_title(
            dataset["label"],
            fontsize=plot.get("title_size", 12)
        )

        ax.set_xlim(xlim)

        ax.tick_params(
            axis="both",
            labelsize=plot.get("tick_size", 9)
        )

        if plot.get("grid", False):
            ax.grid(
                True,
                alpha=0.25,
                linewidth=0.5
            )

        if plot.get("hide_y_ticks", True):
            ax.set_yticks([])

    # ---------------------------------------------
    # Shared axes labels
    # ---------------------------------------------

    fig.supxlabel(
        r"2$\theta$ (degrees)",
        fontsize=plot.get("label_size", 12)
    )

    fig.supylabel(
        "Intensity (counts) + vertical offset",
        fontsize=plot.get("label_size", 12)
    )

    fig.suptitle(
        plot.get(
            "title",
            "Spatial XRD patterns under different sulfurization conditions"
        ),
        fontsize=plot.get("suptitle_size", 15),
        y=0.99
    )

    # ---------------------------------------------
    # Shared position legend
    # ---------------------------------------------

    from matplotlib.lines import Line2D

    position_handles = []

    for i, position in enumerate(positions):

        label = config.get(
            "position_labels", {}
        ).get(
            position_key(position),
            position_key(position)
        )

        position_handles.append(
            Line2D(
                [0],
                [0],
                color=colors[i],
                linewidth=2,
                label=label
            )
        )

    fig.legend(
        handles=position_handles,
        title="Measurement position",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=5,
        frameon=False,
        fontsize=plot.get("legend_size", 8)
    )

    fig.tight_layout(
        rect=[0.03, 0.03, 0.98, 0.88]
    )

    return fig


def main(config):

    # ------------------------------------------------
    # 1. LOAD DATASETS
    # ------------------------------------------------

    datasets = []

    for item in config["datasets"]:

        print(f"Loading {item['label']}")

        data = load_dataset(item)

        datasets.append({
            "label": item["label"],
            "color": item.get("color"),
            "data": data
        })

        print(
            f"  Loaded {len(data)} measurements"
        )
    
    output_dir = Path(
        output.get("directory", "xrd_output")
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    # ================================================
    # INITIALIZE THEORETICAL REFERENCE PATTERNS
    # ================================================

    reference_config = config.get(
        "reference_patterns", {}
    )

    reference_generator = None

    if reference_config.get("enabled", False):

        print("\nGenerating theoretical XRD references...")

        reference_generator = ReferencePatternGenerator(
            wavelength=reference_config.get(
                "wavelength", "CuKa"
            ),
            two_theta_range=tuple(
                reference_config.get(
                    "two_theta_range", [25, 70]
                )
            ),
            normalize=reference_config.get(
                "display", {}
            ).get("intensity_normalization", True)
        )

        reference_generator.calculate_all(
            reference_config
        )

        # Export calculated peak positions and intensities.
        reference_output = reference_config.get(
            "output", {}
        )

        if reference_output.get(
            "save_reference_csv", True
        ):

            reference_generator.export_csv(
                output_dir /
                f"{reference_output.get('filename', 'theoretical_XRD_reference')}.csv"
            )

        # Save a standalone reference pattern figure.
        if reference_output.get(
            "save_reference_plot", True
        ):

            fig_reference = (
                reference_generator.plot_reference_patterns(
                    output_path=output_dir /
                    f"{reference_output.get('filename', 'theoretical_XRD_reference')}.png",
                    show_hkl=reference_config.get(
                        "display", {}
                    ).get("show_hkl", True)
                )
            )


    # ------------------------------------------------
    # 2. DETERMINE MEASUREMENT POSITIONS
    # ------------------------------------------------

    if config.get("position_order"):

        positions = [
            parse_position_order(p)
            for p in config["position_order"]
        ]

    else:

        positions = sorted(
            datasets[0]["data"].keys(),
            key=lambda p: (p[0], p[2], p[1])
        )

    reference_positions = set(positions)

    for dataset in datasets:

        available = set(dataset["data"].keys())

        missing = reference_positions - available
        extra = available - reference_positions

        if missing or extra:

            raise ValueError(
                f"Position mismatch in {dataset['label']}\n"
                f"Missing: {missing}\n"
                f"Extra: {extra}"
            )

    # ------------------------------------------------
    # 3. CONFIGURATION
    # ------------------------------------------------

    plot = config.get("plot", {})
    processing = config.get("processing", {})
    output = config.get("output", {})

    # Common 2theta grid based on first dataset.
    first_df = datasets[0]["data"][positions[0]]

    x_grid = first_df["two_theta"].to_numpy()

    xlim = plot.get("xlim")

    if xlim:

        mask = (
            (x_grid >= xlim[0]) &
            (x_grid <= xlim[1])
        )

        x_grid = x_grid[mask]

    if len(x_grid) < 2:

        raise ValueError(
            "Invalid 2theta range."
        )

    # ------------------------------------------------
    # 4. CREATE SUBPLOTS
    # ------------------------------------------------

    n_positions = len(positions)

    ncols = int(plot.get("ncols", 5))

    nrows = int(np.ceil(n_positions / ncols))

    figsize = plot.get("figsize", [18, 9])

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=True,
        squeeze=False
    )

    axes = axes.ravel()

    offset_fraction = float(
        plot.get("offset_fraction", 1.15)
    )

    line_width = plot.get("line_width", 1.15)

    csv_records = []

    # ------------------------------------------------
    # 5. PLOT EACH MEASUREMENT POSITION
    # ------------------------------------------------

    for panel_index, position in enumerate(positions):

        ax = axes[panel_index]

        curves = []

        # Load four conditions for this position.
        for dataset in datasets:

            df = dataset["data"][position]

            x, y = prepare_intensity(
                df,
                processing,
                x_grid
            )

            curves.append(
                (dataset, x, y)
            )

        # Calculate local offset.
        local_max = max(
            np.max(y) for _, _, y in curves
        )

        if local_max <= 0:
            local_max = 1.0

        offset_step = (
            local_max * offset_fraction
        )

        # Plot conditions from bottom to top.
        for condition_index, (dataset, x, y) in enumerate(curves):

            offset = condition_index * offset_step

            plotted_y = y + offset

            ax.plot(
                x,
                plotted_y,
                color=dataset["color"],
                linewidth=line_width,
                label=dataset["label"]
            )

            # Save processed data.
            for xi, yi in zip(x, y):

                csv_records.append({
                    "position": position_key(position),
                    "phi_deg": position[0],
                    "x_mm": position[1],
                    "y_mm": position[2],
                    "condition": dataset["label"],
                    "two_theta_deg": xi,
                    "intensity_processed": yi,
                    "intensity_with_offset": (
                        yi + offset
                    ),
                    "vertical_offset": offset
                })

        # Panel title.
        custom_labels = config.get(
            "position_labels", {}
        )

        label = custom_labels.get(
            position_key(position),
            position_key(position)
        )

        ax.set_title(
            label,
            fontsize=plot.get("title_size", 9)
        )

        ax.tick_params(
            axis="both",
            labelsize=plot.get("tick_size", 8)
        )

        if plot.get("hide_y_ticks", True):
            ax.set_yticks([])

        if plot.get("grid", False):

            ax.grid(
                True,
                alpha=0.25,
                linewidth=0.5
            )

    # Hide unused panels.
    for ax in axes[n_positions:]:
        ax.set_visible(False)

    # ------------------------------------------------
    # 6. FIGURE LABELS AND LEGEND
    # ------------------------------------------------

    fig.supxlabel(
        r"2$\theta$ (degrees)",
        fontsize=plot.get("label_size", 11)
    )

    fig.supylabel(
        "Intensity (counts) + vertical offset",
        fontsize=plot.get("label_size", 11)
    )

    fig.suptitle(
        plot.get(
            "title",
            "Spatially resolved XRD patterns"
        ),
        fontsize=plot.get("suptitle_size", 14)
    )

    handles, labels = axes[0].get_legend_handles_labels()

    if plot.get("show_legend", True):

        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=len(datasets),
            frameon=False
        )

    fig.tight_layout(
        rect=[0.02, 0.02, 0.98, 0.91]
    )

    # ------------------------------------------------
    # 7. SAVE OUTPUTS
    # ------------------------------------------------



    filename = output.get(
        "filename",
        "xrd_waterfall_by_position"
    )

    dpi = output.get("dpi", 600)

    for fmt in output.get(
        "formats", ["png", "pdf"]
    ):

        filepath = output_dir / f"{filename}.{fmt}"

        fig.savefig(
            filepath,
            dpi=dpi,
            bbox_inches="tight"
        )

        print(f"Saved: {filepath}")

    if output.get("save_processed_csv", True):

        csv_path = (
            output_dir /
            f"{filename}_curves.csv"
        )

        pd.DataFrame(csv_records).to_csv(
            csv_path,
            index=False
        )

        print(f"Saved: {csv_path}")


    # ================================================
    # 7. SAVE WATERFALL OUTPUTS
    # ================================================

    output_dir = Path(
        output.get("directory", "xrd_output")
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    dpi = output.get("dpi", 600)

    # Save waterfall figure
    filename = output.get(
        "filename",
        "xrd_waterfall_by_position"
    )

    for fmt in output.get(
        "formats", ["png", "pdf", "svg"]
    ):

        filepath = output_dir / f"{filename}.{fmt}"

        fig.savefig(
            filepath,
            dpi=dpi,
            bbox_inches="tight"
        )

        print(f"Saved: {filepath}")

    if output.get("save_processed_csv", True):

        csv_path = output_dir / f"{filename}_curves.csv"

        pd.DataFrame(csv_records).to_csv(
            csv_path,
            index=False
        )

        print(f"Saved: {csv_path}")

    # ================================================
    # 8. COMBINED LINEAR AND LOGARITHMIC PLOTS
    # ================================================

    combined = config.get("combined_plot", {})

    if combined.get("enabled", True):

        # Linear plot
        if combined.get("save_linear", True):

            fig_linear = plot_all_data_linear(
                datasets,
                positions,
                config,
                reference_generator,
                reference_config
            )

            for fmt in output.get(
                "formats", ["png", "pdf", "svg"]
            ):

                path = (
                    output_dir /
                    f"xrd_combined_linear.{fmt}"
                )

                fig_linear.savefig(
                    path,
                    dpi=dpi,
                    bbox_inches="tight"
                )

                print(f"Saved: {path}")

        # Logarithmic plot
        if combined.get("save_log", True):

            fig_log = plot_all_data_log(
                datasets,
                positions,
                config,
                reference_generator,
                reference_config
            )

            for fmt in output.get(
                "formats", ["png", "pdf", "svg"]
            ):

                path = (
                    output_dir /
                    f"xrd_combined_log.{fmt}"
                )

                fig_log.savefig(
                    path,
                    dpi=dpi,
                    bbox_inches="tight"
                )

                print(f"Saved: {path}")

    

    
    
    # ================================================
    # CONDITION-WISE SPATIAL XRD PLOTS
    # ================================================

    condition_cfg = config.get("condition_grid", {})

    if condition_cfg.get("enabled", True):

        # Version 1: With vertical offsets
        fig_offset = plot_by_sulfurization_condition(
            datasets,
            positions,
            {
                **config,
                "condition_grid": {
                    **condition_cfg,
                    "use_offset": True
                }
            }
        )

        # Version 2: Without vertical offsets
        fig_no_offset = plot_by_sulfurization_condition(
            datasets,
            positions,
            {
                **config,
                "condition_grid": {
                    **condition_cfg,
                    "use_offset": False
                }
            }
        )

        for fmt in output.get(
            "formats", ["png", "pdf", "svg"]
        ):

            path_offset = (
                output_dir /
                f"xrd_by_sulfurization_condition_offset.{fmt}"
            )

            path_no_offset = (
                output_dir /
                f"xrd_by_sulfurization_condition_no_offset.{fmt}"
            )

            fig_offset.savefig(
                path_offset,
                dpi=dpi,
                bbox_inches="tight"
            )

            fig_no_offset.savefig(
                path_no_offset,
                dpi=dpi,
                bbox_inches="tight"
            )

            print(f"Saved: {path_offset}")
            print(f"Saved: {path_no_offset}")




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="xrd_config.yaml"
    )

    args = parser.parse_args()

    with open(
        args.config,
        "r",
        encoding="utf-8"
    ) as file:

        config = yaml.safe_load(file)

    main(config)
