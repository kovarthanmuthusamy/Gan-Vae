#!/usr/bin/env python3
"""
Visualize Occupancy Vectors from folders of .npy files
=======================================================
Pass one or more folders that contain occupancy .npy files.
Each folder is shown as a separate section; each file in the
folder is one row/bar-chart.

Usage examples
--------------
# Single folder
python visualization/visualize_occupancy.py datasets/data/Occ_map

# Multiple folders (side-by-side comparison)
python visualization/visualize_occupancy.py datasets/data/Occ_map datasets/data_norm/Occ_map

# Save output
python visualization/visualize_occupancy.py folder1 folder2 --save out.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ============================================================
# CONFIGURATION — update paths here if you prefer not to use CLI
# ============================================================

FOLDERS = [
     Path("temp_visuals/data_sample_0/occupancy_map.npy"),
     Path("temp_visuals/data_sample_1/occupancy_map.npy"),
     Path("temp_visuals/data_sample_2/occupancy_map.npy"),
     Path("temp_visuals/data_sample_3/occupancy_map.npy"),
     Path("temp_visuals/data_sample_4/occupancy_map.npy")
]

OUTPUT_PATH = Path("temp_visuals/occupancy_comparison.png")

# ============================================================

LABELS = [
    "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10",
    "C11","C12","C13","C14","C15","C16","C17","C18","C19","C20",
    "C21","C22","C23","C24","C25","C26","C27","C28","C29","C30",
    "C31","C32","C33","C34","C35","C36","C37","C38","C39","C40",
    "C41","C42","C43","C44","C45","C46","C47","C48","C49","C50",
    "C51","C52"
]
N = len(LABELS)


# ---- helpers ---------------------------------------------------------------

def load_vector(path: Path) -> np.ndarray:
    vec = np.load(path).flatten()
    if len(vec) != N:
        raise ValueError(f"{path.name}: expected length {N}, got {len(vec)}")
    return vec


def load_folder(folder: Path):
    """Return sorted list of (filename, vector) pairs from a folder."""
    files = sorted(folder.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in: {folder}")
    return [(f.name, load_vector(f)) for f in files]


def bar_row(ax, vec, row_label: str):
    """Draw a single occupancy bar row on *ax*."""
    colors = ["#2ecc71" if v > 0.5 else "#ecf0f1" for v in vec]
    ax.bar(range(N), [1] * N, color=colors, edgecolor="#bdc3c7",
           linewidth=0.5, width=0.85)
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(0, 1.5)
    ax.set_xticks(range(N))
    ax.set_xticklabels(LABELS, rotation=0, fontsize=8, fontweight="bold")
    ax.set_yticks([])
    ax.set_ylabel(row_label, fontsize=7, rotation=0, ha="right",
                  va="center", labelpad=4)
    active = int((vec > 0.5).sum())
    ax.text(N - 0.5, 1.15, f"Active: {active}/{N}",
            ha="right", va="bottom", fontsize=7, color="#444")


# ---- main plot -------------------------------------------------------------

def plot_folders(folder_data, output_path=None):
    """
    folder_data: {folder_name: [(filename, vector), ...]}
    Each folder becomes a titled section; each file is one row.
    """
    folder_names = list(folder_data.keys())
    n_folders = len(folder_names)
    total_rows = sum(len(v) for v in folder_data.values())

    fig, axes = plt.subplots(
        total_rows, 1,
        figsize=(15, 2.2 * total_rows + 0.5 * n_folders),
        squeeze=False
    )

    on_patch  = mpatches.Patch(color="#2ecc71", label="Occupied (1)")
    off_patch = mpatches.Patch(color="#ecf0f1", label="Empty (0)",
                               edgecolor="#bdc3c7")

    row_idx = 0
    for folder_name, entries in folder_data.items():
        for file_idx, (fname, vec) in enumerate(entries):
            ax = axes[row_idx, 0]
            bar_row(ax, vec, fname)

            if file_idx == 0:
                ax.set_title(
                    f"Folder: {folder_name}",
                    fontsize=10, fontweight="bold", loc="left", pad=4,
                    color="#222"
                )

            active_labels = [LABELS[i] for i in range(N) if vec[i] > 0.5]
            print(f"[{folder_name}] {fname}: {len(active_labels)}/{N} active — {active_labels}")

            row_idx += 1

    fig.legend(handles=[on_patch, off_patch],
               loc="upper right", fontsize=10, framealpha=0.8)
    fig.suptitle("Occupancy Vector Visualization", fontsize=12,
                 fontweight="bold", y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.99]) # type: ignore

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved to {output_path}")
    else:
        plt.show()


# ---- entry point -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize occupancy vectors (.npy) from one or more folders."
    )
    parser.add_argument(
        "folders", nargs="*",
        help="One or more folder paths containing .npy occupancy files."
    )
    parser.add_argument(
        "--save", "-s", metavar="FILE",
        help="Save figure to this path instead of showing it."
    )
    args = parser.parse_args()

    # CLI args take priority; fall back to in-script FOLDERS list
    raw_folders = [Path(p) for p in args.folders] if args.folders else list(FOLDERS)
    save_path   = Path(args.save) if args.save else OUTPUT_PATH

    if not raw_folders:
        parser.error("Provide at least one folder path (CLI or FOLDERS list in script).")

    folder_data = {}
    for p in raw_folders:
        p = Path(p)
        if p.is_file() and p.suffix == ".npy":
            # Single .npy file — group by parent folder name
            group = p.parent.name
            folder_data.setdefault(group, []).append((p.name, load_vector(p)))
        elif p.is_dir():
            entries = load_folder(p)
            folder_data[p.name] = entries
        else:
            raise ValueError(f"Path is neither a .npy file nor a directory: {p}")

    plot_folders(folder_data, output_path=save_path)


if __name__ == "__main__":
    main()
