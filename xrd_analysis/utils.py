
"""Shared utilities for XRD analysis."""

import re

POSITION_RE = re.compile(
    r"phi\s*(-?\d+(?:\.\d+)?)\s*"
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*\)",
    re.IGNORECASE
)


def parse_position(filename):
    match = POSITION_RE.search(str(filename))

    if not match:
        raise ValueError(
            f"Cannot extract coordinates from {filename}. "
            "Expected phi0(0,-15)."
        )

    phi, x, y = match.groups()

    return int(round(float(phi))), float(x), float(y)


def parse_position_order(value):
    match = POSITION_RE.fullmatch(
        str(value).replace(" ", "")
    )

    if not match:
        raise ValueError(f"Invalid position: {value}")

    phi, x, y = match.groups()

    return int(round(float(phi))), float(x), float(y)


def position_key(position):
    phi, x, y = position
    return f"phi{phi}({x:g},{y:g})"


def ordered_positions(config, datasets):

    if config.get("position_order"):
        return [
            parse_position_order(p)
            for p in config["position_order"]
        ]

    return sorted(
        datasets[0]["data"],
        key=lambda p: (p[0], p[2], p[1])
    )


def validate_positions(datasets, positions):

    expected = set(positions)
    errors = []

    for dataset in datasets:

        available = set(dataset["data"])

        missing = sorted(expected - available)
        extra = sorted(available - expected)

        if missing or extra:
            errors.append(
                f"{dataset['label']}: "
                f"missing={missing}; extra={extra}"
            )

    if errors:
        raise ValueError(
            "Measurement positions do not match:\n"
            + "\n".join(errors)
        )
