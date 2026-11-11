
"""Main execution script for combinatorial XRD analysis."""

import argparse
from logging import config
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from xrd_analysis.readers import load_dataset

from xrd_analysis.utils import (
    ordered_positions,
    validate_positions
)

from xrd_analysis.plotters import (
    plot_measurement_position_schematic,
    plot_waterfall,
    plot_combined,
    plot_condition_grid,
    plot_waterfall_normalized
)

from xrd_analysis.outputs import save_figure

from xrd_analysis.peak_detection import (
    detect_dataset_peaks,
)

from xrd_analysis.peak_classification import (
    classify_experimental_peaks,
    classification_summary,
)

from xrd_analysis.peak_plotters import (
    plot_classification_heatmap,
    plot_peak_classification,
    plot_classification_summary,
)

from xrd_analysis.utils import (
    ordered_positions,
    validate_positions,
    position_key,
)

import numpy as np
from scipy.signal import savgol_filter


def smooth_xrd_pattern(
    intensity,
    window_length=7,
    polyorder=2
):
    """
    Smooth an XRD intensity pattern using
    the Savitzky-Golay filter.
    """

    intensity = np.asarray(intensity, dtype=float)

    if len(intensity) < 3:
        return intensity.copy()

    # Ensure odd window length
    if window_length % 2 == 0:
        window_length += 1

    # Window cannot exceed signal length
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
        mode="interp"
    )


def resolve_config_paths(config, config_dir):

    for dataset in config.get("datasets", []):

        path = Path(dataset["path"]).expanduser()

        if not path.is_absolute():
            dataset["path"] = str(
                (config_dir / path).resolve()
            )

    for item in config.get(
        "reference_patterns", {}
    ).get("references", []):

        path = Path(item["cif"]).expanduser()

        if not path.is_absolute():
            item["cif"] = str(
                (config_dir / path).resolve()
            )

    output = config.setdefault("output", {})

    path = Path(
        output.get("directory", "xrd_output")
    ).expanduser()

    if not path.is_absolute():
        output["directory"] = str(
            (config_dir / path).resolve()
        )


def run(config):

    output_config = config.get("output", {})

    output_dir = Path(
        output_config.get("directory", "xrd_output")
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==============================================
    # 1. READ EXPERIMENTAL DATA
    # ==============================================

    datasets = []

    for item in config.get("datasets", []):

        print(f"\nLoading: {item['label']}")

        datasets.append({
            "label": item["label"],
            "color": item.get("color"),
            "data": load_dataset(item)
        })

    if not datasets:
        raise ValueError("No datasets configured.")

    # ==============================================
    # 2. VALIDATE MEASUREMENT POSITIONS
    # ==============================================

    positions = ordered_positions(
        config,
        datasets
    )

    validate_positions(
        datasets,
        positions
    )

    print(
        f"\nValidated {len(positions)} measurement positions."
    )

    
    # ==============================================
    # 3. GENERATE XRD REFERENCE PATTERNS (RTF / CIF)
    # ==============================================

    reference_generator = None

    reference_config = config.get(
        "reference_patterns", {}
    )

    if reference_config.get("enabled", False):

        from xrd_analysis.reference_generator import (
            ReferencePatternGenerator
        )

        reference_source = reference_config.get(
            "source", "rtf"
        ).lower()

        print(
            f"\nGenerating XRD references using "
            f"{reference_source.upper()} files..."
        )

        reference_generator = ReferencePatternGenerator(
            wavelength=reference_config.get(
                "wavelength", "CuKa"
            ),
            two_theta_range=reference_config.get(
                "two_theta_range", [25, 70]
            ),
            normalize=reference_config.get(
                "display", {}
            ).get("intensity_normalization", True),

            max_peaks=reference_config.get(
                "display", {}
            ).get("max_peaks", 10)
        )

        # Load references from configured files.
        reference_generator.calculate_all(
            reference_config
        )

        # ------------------------------------------
        # Export reference peak data
        # ------------------------------------------

        ref_output = reference_config.get(
            "output", {}
        )

        ref_name = ref_output.get(
            "filename",
            "XRD_reference"
        )

        if ref_output.get(
            "save_reference_csv", True
        ):

            reference_generator.export_csv(
                output_dir / f"{ref_name}.csv",
                all_peaks=ref_output.get(
                    "export_all_peaks", False
                )
            )

        # ------------------------------------------
        # Export all reference peaks for classification
        # ------------------------------------------

        if ref_output.get(
            "save_all_reference_csv", True
        ):

            reference_generator.export_csv(
                output_dir / f"{ref_name}_all_peaks.csv",
                all_peaks=True
            )

        # ------------------------------------------
        # Generate reference pattern plot
        # ------------------------------------------

        if ref_output.get(
            "save_reference_plot", True
        ):

            fig_ref = reference_generator.plot_reference_patterns(
                output_path=output_dir / f"{ref_name}.png",

                show_hkl=reference_config.get(
                    "display", {}
                ).get("show_hkl", False),

                reference_config=reference_config
            )

            plt.close(fig_ref)

        print("Reference generation completed.")


    # ==============================================
    # 4. GENERATE FIGURES
    # ==============================================

    figures = {}

    # A. Ten-position waterfall
    fig_waterfall, records = plot_waterfall(
        datasets,
        positions,
        config,
        reference_generator,
        reference_config
    )

    figures[
        output_config.get(
            "waterfall_filename",
            "xrd_waterfall_by_position"
        )
    ] = fig_waterfall

    # B. Combined linear and logarithmic plots
    combined = config.get("combined_plot", {})

    if combined.get("enabled", True):

        if combined.get("save_linear", True):

            figures["xrd_combined_linear"] = plot_combined(
                datasets,
                positions,
                config,
                scale="linear",
                reference_generator=reference_generator,
                reference_config=reference_config
            )

        if combined.get("save_log", True):

            figures["xrd_combined_log"] = plot_combined(
                datasets,
                positions,
                config,
                scale="log",
                reference_generator=reference_generator,
                reference_config=reference_config
            )

    # C. Condition-wise plots
    condition = config.get("condition_grid", {})

    if condition.get("enabled", True):

        figures[
            condition.get(
                "offset_filename",
                "xrd_by_sulfurization_condition_offset"
            )
        ] = plot_condition_grid(
            datasets,
            positions,
            config,
            use_offset=True,
            reference_generator=reference_generator,
            reference_config=reference_config
        )

        figures[
            condition.get(
                "no_offset_filename",
                "xrd_by_sulfurization_condition_no_offset"
            )
        ] = plot_condition_grid(
            datasets,
            positions,
            config,
            use_offset=False,
            reference_generator=reference_generator,
            reference_config=reference_config
        )

    # B. Independently normalized waterfall plot
    normalized = config.get("normalized_waterfall", {})
    
    if normalized.get("enabled", True):
    
        fig_normalized, normalized_records = (
            plot_waterfall_normalized(
                datasets,
                positions,
                config,
                reference_generator=reference_generator,
                reference_config=reference_config
            )
        )
    
        figures[
            normalized.get(
                "filename",
                "xrd_waterfall_normalized"
            )
        ] = fig_normalized

    if normalized.get("save_csv", True):

        normalized_df = pd.DataFrame(normalized_records)

        normalized_df.to_csv(
            output_dir / "xrd_waterfall_normalized.csv",
            index=False
        )

    # E. Experimental peak detection

    peak_config = config.get("peak_detection", {})

    if peak_config.get("enabled", True):

        detected_peaks = detect_dataset_peaks(
            datasets,
            positions,
            config,
        )

        if not detected_peaks.empty:

            peak_filename = peak_config.get(
                "filename",
                "experimental_detected_peaks.csv"
            )

            detected_peaks.to_csv(
                output_dir / peak_filename,
                index=False,
            )

            print(
                f"\nDetected {len(detected_peaks)} peaks "
                f"across all experimental patterns."
            )

        else:
            print("\nNo experimental peaks were detected.")

    # F. Experimental peak classification
    
    classification_config = config.get(
        "peak_classification", {}
    )
    
    if classification_config.get("enabled", True):
    
        if reference_generator is None:
        
            print(
                "Peak classification skipped: "
                "CIF reference patterns are unavailable."
            )
    
        elif detected_peaks.empty:
        
            print(
                "Peak classification skipped: "
                "no experimental peaks detected."
            )
    
        else:
        
            classified_peaks = classify_experimental_peaks(
                detected_peaks,
                reference_generator,
                config,
            )
    
            # Export complete classification table.
            classified_peaks.to_csv(
                output_dir / classification_config.get(
                    "filename",
                    "experimental_peak_classification.csv"
                ),
                index=False,
            )
    
            # Generate classification summary.
            summary = classification_summary(
                classified_peaks
            )
    
            summary.to_csv(
                output_dir / "peak_classification_summary.csv"
            )
    
            # Classification summary figure.
            summary_fig = plot_classification_summary(
                classified_peaks
            )
    
            if summary_fig is not None:
                figures[
                    "peak_classification_summary"
                ] = summary_fig

            # Classification summary: 40-pattern stacked bar plot
            summary_fig = plot_classification_summary(
                classified_peaks
            )
            
            figures["xrd_classification_summary"] = summary_fig
            
            
            # Classification heatmap
            heatmap_fig = plot_classification_heatmap(
                classified_peaks
            )
            
            figures["xrd_classification_heatmap"] = heatmap_fig
            # Generate one representative diagnostic plot.
            if classification_config.get(
                "plot_representative_pattern", True
            ):
    
                selected_condition = classification_config.get(
                    "selected_condition",
                    datasets[0]["label"]
                )
    
                selected_position = classification_config.get(
                    "selected_position",
                    position_key(positions[0])
                )
    
                selected_dataset = next(
                    (
                        dataset for dataset in datasets
                        if dataset["label"] == selected_condition
                    ),
                    None,
                )
    
                selected_position_tuple = next(
                    (
                        position for position in positions
                        if position_key(position) == selected_position
                    ),
                    None,
                )
    
                if (
                    selected_dataset is not None
                    and selected_position_tuple is not None
                ):
    
                    fig_classification = plot_peak_classification(
                        selected_dataset,
                        selected_position_tuple,
                        classified_peaks,
                        config,
                        reference_generator,
                    )
    
                    figures[
                        "experimental_peak_classification"
                    ] = fig_classification
    
            # Print category counts.
            print("\nExperimental peak classification:")
            print(summary.to_string())
        # ==============================================
        # 5. SAVE ALL FIGURES
        # ==============================================
    
        formats = output_config.get(
            "formats",
            ["png", "pdf", "svg"]
        )
    
        dpi = int(
            output_config.get("dpi", 600)
        )
    
        for filename, figure in figures.items():
        
            save_figure(
                figure,
                output_dir,
                filename,
                formats,
                dpi
            )

    
    # ==============================================
    # 6. SAVE PROCESSED CURVES
    # ==============================================

    if output_config.get(
        "save_processed_csv", True
    ):

        csv_name = output_config.get(
            "curves_csv",
            "xrd_waterfall_curves.csv"
        )

        csv_path = output_dir / csv_name

        pd.DataFrame(records).to_csv(
            csv_path,
            index=False
        )

        print(f"Saved processed data: {csv_path}")

    print("\nXRD processing completed.")
    print(f"Output directory: {output_dir}")
    fig_positions = plot_measurement_position_schematic(
        positions,
        diameter_mm=50.0,
    )
    
    fig_positions.savefig(
        output_dir / "measurement_position_schematic.png",
        dpi=300,
        bbox_inches="tight",
    )
    
    plt.close(fig_positions)

    if output_config.get("show_figures", True):

        plt.show()

    else:

        for figure in figures.values():
            plt.close(figure)


def main():

    parser = argparse.ArgumentParser(
        description="Combinatorial XRD analysis."
    )

    parser.add_argument(
        "--config",
        default="xrd_config.yaml"
    )

    args = parser.parse_args()

    config_path = Path(
        args.config
    ).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    with config_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        config = yaml.safe_load(file) or {}

    resolve_config_paths(
        config,
        config_path.parent
    )

    run(config)
    



if __name__ == "__main__":
    main()
