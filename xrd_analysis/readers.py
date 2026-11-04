
"""Readers for ASC and XRDML files."""

from pathlib import Path
from io import BytesIO
import zipfile
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .utils import parse_position


def clean_xy(two_theta, intensity, filename):

    x = np.asarray(two_theta, dtype=float).ravel()
    y = np.asarray(intensity, dtype=float).ravel()

    if x.size != y.size:
        raise ValueError(
            f"{filename}: X/Y length mismatch."
        )

    df = pd.DataFrame({
        "two_theta": x,
        "intensity": y
    })

    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values("two_theta")
        .drop_duplicates("two_theta")
        .reset_index(drop=True)
    )

    if len(df) < 2:
        raise ValueError(
            f"{filename}: insufficient data."
        )

    return df


def read_asc(source, filename=None):

    filename = filename or str(source)

    if isinstance(source, bytes):
        array = np.loadtxt(BytesIO(source))
    else:
        array = np.loadtxt(source)

    if array.ndim == 1:
        array = np.atleast_2d(array)

    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(
            f"Invalid ASC file: {filename}"
        )

    if array.shape[1] >= 3:

        df = pd.DataFrame({
            "two_theta": array[:, 0],
            "intensity": array[:, 1],
            "uncertainty": array[:, 2]
        })

        df = (
            df.replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["two_theta", "intensity"])
            .sort_values("two_theta")
            .drop_duplicates("two_theta")
            .reset_index(drop=True)
        )

        if len(df) < 2:
            raise ValueError(
                f"Insufficient data in {filename}"
            )

        return df

    return clean_xy(
        array[:, 0],
        array[:, 1],
        filename
    )


def local_name(element):
    return element.tag.rsplit("}", 1)[-1].lower()


def find_all(element, name):
    return [
        node for node in element.iter()
        if local_name(node) == name.lower()
    ]


def parse_float_list(text):

    if not text:
        return np.array([], dtype=float)

    return np.fromstring(
        text.replace(",", " "),
        sep=" ",
        dtype=float
    )


def read_xrdml(source, filename=None):
    """
    Read a common PANalytical XRDML scan.

    Currently reads the first dataPoints block.
    """

    filename = filename or str(source)

    if isinstance(source, bytes):
        root = ET.fromstring(source)
    else:
        root = ET.parse(source).getroot()

    blocks = find_all(root, "dataPoints")

    if not blocks:
        raise ValueError(
            f"No dataPoints found in {filename}"
        )

    if len(blocks) > 1:
        warnings.warn(
            f"{filename}: multiple dataPoints blocks found. "
            "Reading the first block only.",
            RuntimeWarning
        )

    block = blocks[0]

    intensity_nodes = (
        find_all(block, "intensities")
        or find_all(block, "counts")
    )

    if not intensity_nodes:
        raise ValueError(
            f"No intensity array in {filename}"
        )

    intensity = parse_float_list(
        intensity_nodes[0].text
    )

    if len(intensity) == 0:
        raise ValueError(
            f"Cannot parse intensity in {filename}"
        )

    axis_node = None

    for node in find_all(block, "positions"):

        axis = node.attrib.get(
            "axis", ""
        ).lower().replace(" ", "")

        if axis in ("2theta", "2-theta", "twotheta"):
            axis_node = node
            break

    if axis_node is None:
        raise ValueError(
            f"Cannot identify 2theta axis in {filename}"
        )

    explicit = find_all(
        axis_node, "listPositions"
    )

    if explicit and explicit[0].text:

        two_theta = parse_float_list(
            explicit[0].text
        )

    else:

        starts = find_all(
            axis_node, "startPosition"
        )

        ends = find_all(
            axis_node, "endPosition"
        )

        if not starts or not ends:
            raise ValueError(
                f"Missing 2theta range in {filename}"
            )

        start = float(starts[0].text)
        end = float(ends[0].text)

        two_theta = np.linspace(
            start, end, len(intensity)
        )

    return clean_xy(
        two_theta,
        intensity,
        filename
    )


def read_one(source, filename):

    extension = Path(filename).suffix.lower()

    if extension == ".asc":
        return read_asc(source, filename)

    if extension == ".xrdml":
        return read_xrdml(source, filename)

    raise ValueError(
        f"Unsupported file format: {filename}"
    )


def load_dataset(dataset_config):
    """
    Read XRD files from ZIP or directory.

    Priority:
        1. ASC
        2. XRDML
    """

    source = Path(
        dataset_config["path"]
    ).expanduser()

    if not source.exists():
        raise FileNotFoundError(source)

    candidates = {}

    if source.is_dir():

        pattern = dataset_config.get(
            "file_glob", "*"
        )

        files = [
            p for p in source.glob(pattern)
            if p.is_file()
            and p.suffix.lower() in (".asc", ".xrdml")
        ]

        for file in files:

            position = parse_position(file.name)

            candidates.setdefault(
                position, []
            ).append(file)

        results = {}

        for position, options in candidates.items():

            selected = sorted(
                options,
                key=lambda p: (
                    0 if p.suffix.lower() == ".asc" else 1
                )
            )[0]

            results[position] = read_one(
                selected,
                selected.name
            )

    elif zipfile.is_zipfile(source):

        with zipfile.ZipFile(source, "r") as archive:

            files = [
                name for name in archive.namelist()
                if not name.endswith("/")
                and Path(name).suffix.lower()
                in (".asc", ".xrdml")
            ]

            for filename in files:

                basename = Path(filename).name
                position = parse_position(basename)

                candidates.setdefault(
                    position, []
                ).append(filename)

            results = {}

            for position, options in candidates.items():

                selected = sorted(
                    options,
                    key=lambda name: (
                        0 if Path(name).suffix.lower() == ".asc" else 1,
                        name.lower()
                    )
                )[0]

                results[position] = read_one(
                    archive.read(selected),
                    Path(selected).name
                )

    else:

        raise ValueError(
            "Input must be a ZIP archive or directory."
        )

    if not results:
        raise ValueError(
            f"No ASC or XRDML files found in {source}"
        )

    print(
        f"Loaded {len(results)} positions from {source.name}"
    )

    return results
