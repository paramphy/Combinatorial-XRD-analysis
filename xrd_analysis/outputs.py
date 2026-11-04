
"""Centralized figure and data output functions."""

from pathlib import Path


def save_figure(
    figure,
    output_dir,
    filename,
    formats=None,
    dpi=600
):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for extension in formats or ["png", "pdf"]:

        extension = str(extension).lower().lstrip(".")

        path = output_dir / f"{filename}.{extension}"

        figure.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight"
        )

        print(f"Saved figure: {path}")
