
"""CIF-based theoretical XRD reference generator."""

from ast import pattern
from ast import pattern
from pathlib import Path

import numpy as np
import pandas as pd


class ReferencePatternGenerator:

    def __init__(
        self,
        wavelength="CuKa",
        two_theta_range=(25, 70),
        normalize=True,
        max_peaks = 10
    ):

        try:
            from pymatgen.analysis.diffraction.xrd import XRDCalculator
        except ImportError as exc:
            raise ImportError(
                "Install pymatgen: pip install pymatgen"
            ) from exc

        self.wavelength = wavelength
        self.two_theta_range = tuple(two_theta_range)
        self.normalize = normalize

        self.calculator = XRDCalculator(
            wavelength=wavelength
        )
        self.max_peaks = max_peaks

        self.references = {}

    def calculate_pattern(self, cif_path, label=None):

        from pymatgen.core import Structure

        cif_path = Path(cif_path)

        if not cif_path.is_file():
            raise FileNotFoundError(cif_path)

        label = label or cif_path.stem

        structure = Structure.from_file(cif_path)

        pattern = self.calculator.get_pattern(
            structure,
            two_theta_range=self.two_theta_range
        )

        # Get the original theoretical intensities.
        intensity = np.asarray(pattern.y, dtype=float)

        hkl_text = []
        for reflection_group in pattern.hkls:
            labels = []
            for reflection in reflection_group:
                hkl = tuple(int(v) for v in reflection["hkl"])
                labels.append("(" + " ".join(map(str, hkl)) + ")")
            hkl_text.append(", ".join(labels))

        # Create a table containing all calculated reflections.
        frame = pd.DataFrame({
            "two_theta": np.asarray(pattern.x, dtype=float),
            "d_spacing": np.asarray(pattern.d_hkls, dtype=float),
            "intensity": intensity,
            "hkl": hkl_text,
        })

        # Select only the strongest reflections.
        max_peaks = int(self.max_peaks)
        frame = (
            frame.nlargest(max_peaks, "intensity")
            .sort_values("two_theta")
            .reset_index(drop=True)
        )

        # Normalize the selected intensities.
        if self.normalize and not frame.empty:
            maximum = frame["intensity"].max()
            if maximum > 0:
                frame["intensity"] = frame["intensity"] / maximum * 100.0

        self.references[label] = frame

        print(
            f"  Reference {label}: "
            f"{len(frame)} strongest reflections selected"
        )
        return frame

        if self.normalize and intensity.size:
            if intensity.max() > 0:
                intensity = intensity / intensity.max() * 100

        hkl_labels = []

        for group in pattern.hkls:

            labels = []

            for reflection in group:

                hkl = reflection["hkl"]

                labels.append(
                    "(" + " ".join(
                        str(int(v)) for v in hkl
                    ) + ")"
                )

            hkl_labels.append(", ".join(labels))

        df = pd.DataFrame({
            "two_theta": np.asarray(pattern.x),
            "d_spacing": np.asarray(pattern.d_hkls),
            "intensity": intensity,
            "hkl": hkl_labels
        })

        self.references[label] = df

        print(
            f"Calculated {len(df)} reflections for {label}"
        )

        return df

    def calculate_all(self, reference_config):

        self.max_peaks = int(
            reference_config.get("display", {}).get("max_peaks", 10)
        )

        for item in reference_config.get(
            "references", []
        ):

            self.calculate_pattern(
                item["cif"],
                item["label"]
            )

        return self.references

    def export_csv(self, output_path):

        if not self.references:
            raise RuntimeError(
                "No reference patterns calculated."
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        frames = []

        for phase, df in self.references.items():

            copy = df.copy()
            copy.insert(0, "phase", phase)
            frames.append(copy)

        result = pd.concat(
            frames, ignore_index=True
        )

        result.to_csv(
            output_path, index=False
        )

        print(f"Saved reference peaks: {output_path}")

        return result

    def add_reference_sticks(self, ax, reference_config):
        """
        Add the selected theoretical reference peaks to an experimental plot.
        Stick height is expressed in axes coordinates.
        """
        display = reference_config.get("display", {})

        if not display.get("show", True):
            return

        x_min, x_max = ax.get_xlim()

        y0 = float(display.get("stick_ymin", 0.02))
        y1 = float(display.get("stick_ymax", 0.10))

        max_labels = int(display.get("max_hkl_labels", 0))
        label_size = float(display.get("label_size", 6))

        for item in reference_config.get("references", []):
            label = item["label"]

            if label not in self.references:
                continue

            frame = self.references[label]

            visible = frame[
                frame["two_theta"].between(x_min, x_max)
            ]

            if visible.empty:
                continue

            color = item.get("color", "black")

            ax.vlines(
                visible["two_theta"].to_numpy(),
                y0,
                y1,
                transform=ax.get_xaxis_transform(),
                color=color,
                linewidth=float(
                    display.get("stick_line_width", 1.1)
                ),
                alpha=float(
                    display.get("stick_alpha", 0.9)
                ),
                zorder=6,
                label=f"{label} reference peaks",
            )

            if max_labels > 0:
                strongest = visible.nlargest(
                    max_labels, "intensity"
                )

                for _, peak in strongest.iterrows():
                    ax.text(
                        peak["two_theta"],
                        y1 + 0.01,
                        peak["hkl"],
                        transform=ax.get_xaxis_transform(),
                        rotation=90,
                        ha="center",
                        va="bottom",
                        fontsize=label_size,
                        color=color,
                        clip_on=True,
                    )

    def plot_reference_patterns(
        self,
        output_path=None,
        show_hkl=False,
        reference_config=None,
    ):
        import matplotlib.pyplot as plt
    
        if not self.references:
            raise RuntimeError(
                "No reference patterns have been calculated."
            )
    
        reference_config = reference_config or {}
        reference_items = reference_config.get("references", [])
    
        color_map = {
            item["label"]: item.get("color", "black")
            for item in reference_items
        }
    
        display = reference_config.get("display", {})
        max_labels = int(display.get("max_hkl_labels", 5))
    
        fig, ax = plt.subplots(figsize=(10, 5.5))
    
        for label, frame in self.references.items():
        
            color = color_map.get(label, "black")
    
            ax.vlines(
                frame["two_theta"],
                0,
                frame["intensity"],
                color=color,
                linewidth=1.5,
                label=label,
            )
    
            if show_hkl:
                strongest = frame.nlargest(
                    max_labels, "intensity"
                )
    
                for _, peak in strongest.iterrows():
                    ax.text(
                        peak["two_theta"],
                        peak["intensity"] + 2,
                        peak["hkl"],
                        rotation=90,
                        fontsize=7,
                        ha="center",
                        va="bottom",
                        color=color,
                    )
    
        ax.set_xlim(self.two_theta_range)
        ax.set_xlabel(r"2$\theta$ (degrees)")
        ax.set_ylabel("Relative intensity (%)")
        ax.set_title("Theoretical XRD reference patterns")
    
        ax.legend(
            title="Reference phase",
            frameon=False,
        )
    
        ax.grid(alpha=0.2)
        fig.tight_layout()
    
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(
                parents=True, exist_ok=True
            )
            fig.savefig(
                output_path,
                dpi=600,
                bbox_inches="tight",
            )
            print(f"Saved reference plot: {output_path}")
    
        return fig
