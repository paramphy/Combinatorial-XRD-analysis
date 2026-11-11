
"""
Reference pattern generator for XRD analysis.

Supported reference formats:
    1. CIF: theoretical XRD patterns calculated using pymatgen.
    2. RTF: ICDD reference peak lists extracted from RTF documents.

The class maintains two dictionaries:

    self.all_references:
        All available reflections, used for peak classification.

    self.references:
        The strongest reflections selected for display.

Both formats produce a common DataFrame structure containing:
    two_theta, d_spacing, intensity, hkl

RTF references additionally contain:
    peak_number, h, k, l
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


class ReferencePatternGenerator:

    def __init__(
        self,
        wavelength="CuKa",
        two_theta_range=(25, 70),
        normalize=True,
        max_peaks=10
    ):
        self.wavelength = wavelength
        self.two_theta_range = tuple(two_theta_range)
        self.normalize = normalize
        self.max_peaks = int(max_peaks)

        # Pymatgen is loaded only when a CIF is used.
        self.calculator = None

        # Strongest reflections for plotting.
        self.references = {}

        # All reflections for classification.
        self.all_references = {}

    # ---------------------------------------------------------
    # Configuration helper
    # ---------------------------------------------------------

    @staticmethod
    def _get_reference_items(reference_config):
        """
        Accept either of these configurations:

        1. List:
           references:
             - label: CuS
               rtf: ...

        2. Mapping:
           phases:
             CuS:
               file: ...
        """

        items = reference_config.get("references", [])

        if items:
            return items

        phases = reference_config.get("phases", {})

        result = []

        for label, settings in phases.items():

            item = dict(settings)
            item["label"] = label

            result.append(item)

        return result

    # ---------------------------------------------------------
    # CIF reference generation
    # ---------------------------------------------------------

    def calculate_pattern(self, cif_path, label=None):
        """
        Calculate a theoretical XRD pattern from a CIF file.
        """

        if self.calculator is None:

            try:
                from pymatgen.analysis.diffraction.xrd import XRDCalculator

            except ImportError as exc:
                raise ImportError(
                    "Install pymatgen to calculate CIF references: "
                    "pip install pymatgen"
                ) from exc

            self.calculator = XRDCalculator(
                wavelength=self.wavelength
            )

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

        intensity = np.asarray(pattern.y, dtype=float)

        hkl_text = []

        for reflection_group in pattern.hkls:

            labels = []

            for reflection in reflection_group:

                hkl = tuple(
                    int(v) for v in reflection["hkl"]
                )

                labels.append(
                    "(" + " ".join(map(str, hkl)) + ")"
                )

            hkl_text.append(", ".join(labels))

        frame = pd.DataFrame({
            "two_theta": np.asarray(pattern.x, dtype=float),
            "d_spacing": np.asarray(pattern.d_hkls, dtype=float),
            "intensity": intensity,
            "hkl": hkl_text
        })

        # Keep all calculated reflections unchanged.
        self.all_references[label] = frame.copy()

        print(
            f"  {label}: {len(frame)} theoretical CIF reflections"
        )

        return self._select_display_peaks(frame, label)

    # ---------------------------------------------------------
    # RTF reference reader
    # ---------------------------------------------------------

    def read_rtf_reference(self, rtf_path, label=None):
        """
        Read an ICDD RTF reference file.

        Expected peak-list columns:

            Peak number
            h
            k
            l
            d-spacing
            2theta
            Relative intensity

        The RTF structure section is excluded from parsing.
        """

        try:
            from striprtf.striprtf import rtf_to_text

        except ImportError as exc:
            raise ImportError(
                "Install striprtf to read ICDD RTF files: "
                "pip install striprtf"
            ) from exc

        rtf_path = Path(rtf_path)

        if not rtf_path.is_file():
            raise FileNotFoundError(rtf_path)

        label = label or rtf_path.stem

        # RTF files contain control codes and escaped characters.
        # Latin-1 safely reads the underlying byte values.
        with open(
            rtf_path,
            "r",
            encoding="latin-1"
        ) as file:

            rtf_content = file.read()

        try:
            text = rtf_to_text(rtf_content)

        except Exception as exc:
            raise RuntimeError(
                f"Could not convert RTF file: {rtf_path}"
            ) from exc

        # Find the peak-list section.
        peak_header = re.search(
            r"Peak\s+list",
            text,
            flags=re.IGNORECASE
        )

        if peak_header is None:
            raise ValueError(
                f"No Peak list section found in {rtf_path.name}"
            )

        remaining_text = text[peak_header.end():]

        # Stop before the crystal structure table.
        structure_header = re.search(
            r"\bStructure\b",
            remaining_text,
            flags=re.IGNORECASE
        )

        if structure_header is not None:

            peak_section = remaining_text[
                :structure_header.start()
            ]

        else:
            peak_section = remaining_text

        # Find the intensity column heading.
        # Parsing starts after this heading so that numbers in
        # the column labels cannot be mistaken for peak data.
        intensity_header = re.search(
            r"I\s*\[\s*%\s*\]",
            peak_section,
            flags=re.IGNORECASE
        )

        if intensity_header is None:
            raise ValueError(
                f"Could not locate the I [%] column heading "
                f"in {rtf_path.name}"
            )

        peak_data_text = peak_section[intensity_header.end():]

        # Extract numeric tokens.
        number_pattern = re.compile(
            r"(?<![A-Za-z])"
            r"[-+]?(?:\d+\.\d*|\.\d+|\d+)"
            r"(?:[eE][-+]?\d+)?"
            r"(?![A-Za-z])"
        )

        tokens = number_pattern.findall(peak_data_text)

        if not tokens:
            raise ValueError(
                f"No numeric peak data found in {rtf_path.name}"
            )

        # Each reflection must have exactly seven values:
        #
        # number, h, k, l, d, 2theta, intensity
        #
        if len(tokens) % 7 != 0:

            raise ValueError(
                f"Could not parse the peak table in {rtf_path.name}. "
                f"Found {len(tokens)} numeric values, which cannot "
                "be divided into groups of seven. Inspect the "
                "converted RTF text and adjust the parser if needed."
            )

        records = []

        for index in range(0, len(tokens), 7):

            values = tokens[index:index + 7]

            peak_number = int(float(values[0]))
            h = int(float(values[1]))
            k = int(float(values[2]))
            l = int(float(values[3]))

            d_spacing = float(values[4])
            two_theta = float(values[5])
            intensity = float(values[6])

            # Basic validation.
            if d_spacing <= 0:
                raise ValueError(
                    f"Invalid d-spacing in {rtf_path.name}, "
                    f"peak {peak_number}"
                )

            if not 0 <= two_theta <= 180:
                raise ValueError(
                    f"Invalid 2theta value in {rtf_path.name}, "
                    f"peak {peak_number}: {two_theta}"
                )

            if intensity < 0:
                raise ValueError(
                    f"Negative intensity in {rtf_path.name}, "
                    f"peak {peak_number}"
                )

            records.append({

                "peak_number": peak_number,

                "h": h,
                "k": k,
                "l": l,

                "d_spacing": d_spacing,

                "two_theta": two_theta,

                "intensity": intensity,

                "hkl": f"({h} {k} {l})"

            })

        frame = pd.DataFrame(records)

        if frame.empty:
            raise ValueError(
                f"No reflections extracted from {rtf_path.name}"
            )

        # Remove duplicate peak numbers, if any.
        frame = frame.drop_duplicates(
            subset="peak_number"
        )

        # Apply configured 2theta range.
        lower, upper = self.two_theta_range

        frame = frame[
            frame["two_theta"].between(lower, upper)
        ].copy()

        frame = (
            frame.sort_values("two_theta")
            .reset_index(drop=True)
        )

        if frame.empty:
            raise ValueError(
                f"No reflections for {label} within "
                f"the configured range {self.two_theta_range}."
            )

        # Preserve every extracted reflection within the range.
        self.all_references[label] = frame.copy()

        print(
            f"  {label}: {len(frame)} ICDD reflections "
            f"loaded from {rtf_path.name}"
        )

        return frame

    # ---------------------------------------------------------
    # RTF reference processing
    # ---------------------------------------------------------

    def calculate_from_rtf(self, rtf_path, label=None):
        """
        Load an ICDD RTF reference and select display peaks.
        """

        label = label or Path(rtf_path).stem

        frame = self.read_rtf_reference(
            rtf_path,
            label=label
        )

        return self._select_display_peaks(frame, label)

    # ---------------------------------------------------------
    # Select strongest reflections for display
    # ---------------------------------------------------------

    def _select_display_peaks(self, frame, label):
        """
        Select the strongest reflections for plotting.

        Original intensities in all_references are not modified.
        """

        max_peaks = int(self.max_peaks)

        display_frame = (
            frame.nlargest(max_peaks, "intensity")
            .sort_values("two_theta")
            .reset_index(drop=True)
            .copy()
        )

        if self.normalize and not display_frame.empty:

            maximum = display_frame["intensity"].max()

            if maximum > 0:

                display_frame["intensity"] = (
                    display_frame["intensity"]
                    / maximum
                    * 100.0
                )

        self.references[label] = display_frame

        print(
            f"  Reference {label}: "
            f"{len(display_frame)} strongest reflections selected"
        )

        return display_frame

    # ---------------------------------------------------------
    # Load all configured references
    # ---------------------------------------------------------

    def calculate_all(self, reference_config):
        """
        Load all references according to the YAML configuration.

        Supported formats:
            cif
            rtf
        """

        self.max_peaks = int(
            reference_config.get(
                "display", {}
            ).get("max_peaks", 10)
        )

        # Clear old results.
        self.references.clear()
        self.all_references.clear()

        reference_items = self._get_reference_items(
            reference_config
        )

        if not reference_items:
            raise ValueError(
                "No reference files were configured."
            )

        default_source = reference_config.get(
            "source", "cif"
        ).lower()

        for item in reference_items:

            label = item["label"]

            source = item.get(
                "source",
                default_source
            ).lower()

            if source == "rtf":

                file_path = item.get(
                    "rtf",
                    item.get("file")
                )

                if not file_path:
                    raise ValueError(
                        f"No RTF path configured for {label}"
                    )

                self.calculate_from_rtf(
                    file_path,
                    label=label
                )

            elif source == "cif":

                file_path = item.get(
                    "cif",
                    item.get("file")
                )

                if not file_path:
                    raise ValueError(
                        f"No CIF path configured for {label}"
                    )

                self.calculate_pattern(
                    file_path,
                    label=label
                )

            else:

                raise ValueError(
                    f"Unsupported reference format '{source}' "
                    f"for phase {label}. Use 'cif' or 'rtf'."
                )

        return self.references

    # ---------------------------------------------------------
    # Export reference peaks
    # ---------------------------------------------------------

    
    def export_csv(self, output_path, all_peaks=False):
        """
        Export reference peaks to CSV.

        Parameters
        ----------
        output_path : str or Path
            Output CSV path.

        all_peaks : bool
            False: Export selected display peaks.
            True: Export all reference reflections.
        """

        source = (
            self.all_references
            if all_peaks
            else self.references
        )

        if not source:
            raise RuntimeError(
                "No reference patterns available for export."
            )

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        frames = []

        for phase, df in source.items():

            copy = df.copy()
            copy.insert(0, "phase", phase)

            frames.append(copy)

        result = pd.concat(
            frames,
            ignore_index=True
        )

        result.to_csv(
            output_path,
            index=False
        )

        print(f"Saved reference peaks: {output_path}")

        return result


    # ---------------------------------------------------------
    # Helper for plotting configuration
    # ---------------------------------------------------------

    def _get_plot_reference_items(self, reference_config):
        """
        Return normalized reference configuration items.
        """

        return self._get_reference_items(reference_config)

    # ---------------------------------------------------------
    # Add reference sticks to an experimental plot
    # ---------------------------------------------------------

    def add_reference_sticks(self, ax, reference_config):
        """
        Add selected reference peaks to an experimental XRD plot.
        """

        display = reference_config.get(
            "display", {}
        )

        if not display.get("show", True):
            return

        x_min, x_max = ax.get_xlim()

        # Reference stick positions are expressed as fractions
        # of the axes height (using ax.get_xaxis_transform()).
        #
        # Keep the reference ticks close to the bottom of the plot
        # to minimise interference with experimental diffraction peaks.
        
        y0 = float(
            display.get("stick_ymin", 0.005)
        )
        
        y1 = float(
            display.get("stick_ymax", 0.045)
        )
        
        # Validate the reference-stick positions.
        if not (0.0 <= y0 < y1 <= 1.0):
            raise ValueError(
                "Invalid reference stick positions. "
                "Expected 0 <= stick_ymin < stick_ymax <= 1, "
                f"received stick_ymin={y0}, stick_ymax={y1}."
            )
    
            max_labels = int(
                display.get("max_hkl_labels", 0)
            )
    
            label_size = float(
                display.get("label_size", 6)
            )

        for item in self._get_plot_reference_items(
            reference_config
        ):

            label = item["label"]

            if label not in self.references:
                continue

            frame = self.references[label]

            visible = frame[
                frame["two_theta"].between(
                    x_min, x_max
                )
            ]

            if visible.empty:
                continue

            color = item.get(
                "color", "black"
            )

            ax.vlines(
                visible["two_theta"].to_numpy(),
                y0,
                y1,
                transform=ax.get_xaxis_transform(),
                color=color,
                linewidth=float(
                    display.get(
                        "stick_line_width", 1.1
                    )
                ),
                alpha=float(
                    display.get(
                        "stick_alpha", 0.9
                    )
                ),
                zorder=6,
                label=f"{label} reference peaks"
            )

            if max_labels > 0:

                strongest = visible.nlargest(
                    max_labels,
                    "intensity"
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
                        clip_on=True
                    )

    # ---------------------------------------------------------
    # Plot reference patterns
    # ---------------------------------------------------------

    def plot_reference_patterns(
        self,
        output_path=None,
        show_hkl=False,
        reference_config=None
    ):

        import matplotlib.pyplot as plt

        if not self.references:
            raise RuntimeError(
                "No reference patterns have been calculated."
            )

        reference_config = reference_config or {}

        reference_items = self._get_plot_reference_items(
            reference_config
        )

        color_map = {
            item["label"]: item.get("color", "black")
            for item in reference_items
        }

        display = reference_config.get(
            "display", {}
        )

        max_labels = int(
            display.get("max_hkl_labels", 5)
        )

        fig, ax = plt.subplots(
            figsize=(10, 5.5)
        )

        for label, frame in self.references.items():

            color = color_map.get(
                label, "black"
            )

            ax.vlines(
                frame["two_theta"],
                0,
                frame["intensity"],
                color=color,
                linewidth=1.5,
                label=label
            )

            if show_hkl:

                strongest = frame.nlargest(
                    max_labels,
                    "intensity"
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
                        color=color
                    )

        ax.set_xlim(self.two_theta_range)

        ax.set_xlabel(
            r"2$\theta$ (degrees)"
        )

        ax.set_ylabel(
            "Relative intensity (%)"
        )

        ax.set_title(
            "XRD reference patterns"
        )

        ax.legend(
            title="Reference phase",
            frameon=False
        )

        ax.grid(alpha=0.2)

        fig.tight_layout()

        if output_path:

            output_path = Path(output_path)

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            fig.savefig(
                output_path,
                dpi=600,
                bbox_inches="tight"
            )

            print(
                f"Saved reference plot: {output_path}"
            )

        return fig
