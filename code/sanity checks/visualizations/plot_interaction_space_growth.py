"""Visualize the nested low-order interaction spaces V_s on a vector.

For a binary d-way tensor, vector coordinates are indexed by
alpha in {0, 1}^d.  The space V_s contains exactly the coordinates whose
Hamming weight satisfies |alpha| <= s.  Each row in the figure shows the
support allowed by one V_s; colors identify the interaction order |alpha|.
"""

from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# Colorblind-friendly colors for interaction orders 0, 1, ... .
ORDER_COLORS = [
    "#4477AA",
    "#66CCEE",
    "#228833",
    "#CCBB44",
    "#EE6677",
    "#AA3377",
    "#BBBBBB",
    "#44AA99",
]
INACTIVE_COLOR = "#ECECEC"


def binary_multi_indices(d: int) -> np.ndarray:
    """Return alpha=(alpha_1,...,alpha_d) in first-mode-fastest order."""
    coordinate = np.arange(2**d, dtype=np.uint64)
    shifts = np.arange(d, dtype=np.uint64)
    return ((coordinate[:, None] >> shifts[None, :]) & 1).astype(int)


def build_growth_matrix(d: int, max_s: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a matrix whose row s contains |alpha| for active coordinates."""
    alpha = binary_multi_indices(d)
    orders = alpha.sum(axis=1)
    growth = np.full((max_s + 1, 2**d), -1, dtype=int)
    for s in range(max_s + 1):
        active = orders <= s
        growth[s, active] = orders[active]
    return growth, alpha


def plot_interaction_space_growth(
    d: int = 5,
    max_s: int | None = None,
) -> plt.Figure:
    """Create the vector-support visualization for V_0 subset ... subset V_s."""
    if d < 1:
        raise ValueError("d must be at least 1")
    if d > 8:
        raise ValueError("Choose d <= 8 so the vector coordinates remain readable")

    max_s = d if max_s is None else max_s
    if not 0 <= max_s <= d:
        raise ValueError("max_s must satisfy 0 <= max_s <= d")

    growth, alpha = build_growth_matrix(d, max_s)
    n = 2**d

    # Shift by one so 0 denotes inactive and order ell uses color ell + 1.
    display = growth + 1
    colors = [INACTIVE_COLOR] + [ORDER_COLORS[i % len(ORDER_COLORS)] for i in range(d + 1)]
    cmap = ListedColormap(colors)

    figure_width = max(10.0, 0.38 * n)
    figure_height = 1.25 + 0.72 * (max_s + 1)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height), constrained_layout=True)
    ax.imshow(
        display,
        cmap=cmap,
        vmin=-0.5,
        vmax=d + 1.5,
        interpolation="nearest",
        aspect="auto",
    )

    # Thin cell boundaries emphasize that every row is a length-2^d vector.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, max_s + 1, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    labels = ["".join(map(str, bits)) for bits in alpha]
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlabel(r"vector coordinate $\alpha=(\alpha_1,\ldots,\alpha_d)$")

    dimensions = [sum(comb(d, ell) for ell in range(s + 1)) for s in range(max_s + 1)]
    row_labels = [rf"$V_{s}$   ($D_{s}={dimensions[s]}$)" for s in range(max_s + 1)]
    ax.set_yticks(np.arange(max_s + 1))
    ax.set_yticklabels(row_labels)
    ax.set_ylabel("allowed coefficient vector")

    ax.set_title(
        rf"Growth of $V_s$ for a binary tensor with $d={d}$ "
        rf"($N=2^d={n}$)",
        pad=12,
    )

    handles = [Patch(facecolor=INACTIVE_COLOR, label="inactive")]
    handles.extend(
        Patch(facecolor=colors[ell + 1], label=rf"order $\ell={ell}$")
        for ell in range(max_s + 1)
    )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.31),
        ncol=min(max_s + 2, 7),
        frameon=False,
    )

    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.8)

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=5, help="number of binary modes")
    parser.add_argument(
        "--max-s",
        type=int,
        default=None,
        help="largest interaction order shown (default: d)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for PNG and PDF outputs",
    )
    parser.add_argument("--show", action="store_true", help="open the figure window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir or repo_root / "code" / "results" / "figures" / "model"
    output_dir.mkdir(parents=True, exist_ok=True)

    max_s = args.d if args.max_s is None else args.max_s
    fig = plot_interaction_space_growth(args.d, max_s)
    stem = output_dir / f"interaction_space_growth_d{args.d}"
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
