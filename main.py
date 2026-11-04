
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
    plot_waterfall,
    plot_combined,
    plot_condition_grid,
    plot_waterfall_normalized
)

from xrd_analysis.outputs import save_figure


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
    # 3. GENERATE CIF REFERENCE PATTERNS
    # ==============================================

    reference_generator = None

    reference_config = config.get(
        "reference_patterns", {}
    )

    if reference_config.get("enabled", False):

        from xrd_analysis.reference_generator import (
            ReferencePatternGenerator
        )

        print("\nGenerating CIF references...")

        reference_generator = ReferencePatternGenerator(
            wavelength=reference_config.get(
                "wavelength", "CuKa"
            ),
            two_theta_range=reference_config.get(
                "two_theta_range", [25, 70]
            ),
            normalize=reference_config.get(
                "display", {}
            ).get("intensity_normalization", True)
        )

        reference_generator.calculate_all(
            reference_config
        )

        ref_output = reference_config.get(
            "output", {}
        )

        ref_name = ref_output.get(
            "filename",
            "theoretical_XRD_reference"
        )

        if ref_output.get("save_reference_csv", True):

            reference_generator.export_csv(
                output_dir / f"{ref_name}.csv"
            )

        if ref_output.get("save_reference_plot", True):

            fig_ref = reference_generator.plot_reference_patterns(
                output_path=output_dir / f"{ref_name}.png",
                show_hkl=reference_config.get("display", {}).get(
                    "show_hkl", False
                ),
                reference_config=reference_config,
            )

            plt.close(fig_ref)

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
