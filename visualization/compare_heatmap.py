"""
Compare Generated vs Real Heatmaps
==================================
This script plots multiple generated vs real heatmap comparisons
in subplots with absolute difference visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.path import Path
from scipy.interpolate import griddata, RBFInterpolator
from pathlib import Path as FilePath
import json
from pathlib import Path

# ============================================================
# CONFIGURATION - Update these paths for your comparisons
# ============================================================

# Denormalization stats parameter
stats_type = "percentile_min_max"  # Options: "percentile_min_max", "global_min_max"

# Define input directory and output path
input_dir = Path("experiments/exp010/visuals")
OUTPUT_PATH = input_dir / "generated_vs_real_heatmap.png"

# 2. Automate the list creation
# This creates a list for indices 1, 2, and 3
COMPARISONS = [
    {
        "generated": input_dir / f"data_sample_{i}" / "heatmap.npy",
        "real": input_dir / f"data_sample_{i}" / "Real_Heatmap.map/PI/Power_GND/Z_0063.000MHz.map",
        "label": f"data_sample_{i}"
    }
    for i in range(0, 3)
]

# Binary mask path
MASK_PATH = Path("configs/binary_mask.npy")

# Normalization stats path (for min-max color limits)
NORMALIZATION_STATS_PATH = Path("datasets/source/data_norm/normalization_stats.json")

# Colormap settings
CMAP = "jet"
GAMMA = 2.2  # Power normalization gamma

# ============================================================


def load_normalization_stats(stats_path):
    """Load normalization statistics."""
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    return stats


def load_map_file(file_path, resolution=64):
    """
    Load real heatmap from .map file and interpolate to grid.
    Returns: 2D numpy array of shape (resolution, resolution)
    """
    x, y, z = [], [], []

    with open(file_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if lines[i].strip() == "3":
            try:
                for k in range(1, 4):
                    px, py, pz = map(float, lines[i + k].split())
                    x.append(px)
                    y.append(py)
                    z.append(max(pz, 0.01))  # avoid zeros
                i += 3
            except:
                pass
        i += 1

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    if len(x) == 0:
        raise RuntimeError(f"No data loaded from MAP file: {file_path}")

    # Scale to mm
    scale = 1e-5  # 10 nm → mm
    x = (x - x.min()) * scale
    y = (y - y.min()) * scale

    # Create regular grid
    xi = np.linspace(x.min(), x.max(), resolution)
    yi = np.linspace(y.min(), y.max(), resolution)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolation
    points = np.column_stack((x, y))
    Zi = griddata(points, z, (Xi, Yi), method="linear")

    # Fill NaN with RBF interpolation
    nan_mask = np.isnan(Zi)
    if np.any(nan_mask):
        rbf = RBFInterpolator(points, z, smoothing=0.15)
        Zi[nan_mask] = rbf(np.column_stack((Xi[nan_mask], Yi[nan_mask])))

    # Flip vertically to match generated heatmap orientation (imshow origin='upper')
    Zi = np.flipud(Zi)

    return Zi


def load_generated_heatmap(file_path,stats=None, stats_type="percentile_min_max"):
    """
    Load generated heatmap from .npy file.
    Returns: 2D numpy array
    """
    data = np.load(file_path)
    if data.ndim == 3:
        data = data[0]  # Take first channel if 3D

    # denormalize the generated heatmap
    if stats is not None:
        heatmap_min = stats[stats_type]['heatmap_min']
        heatmap_max = stats[stats_type]['heatmap_max']
        data = data * (heatmap_max - heatmap_min) + heatmap_min
    return data


def plot_comparison(comparisons_data, mask, output_path):
    """
    Plot generated vs real heatmaps with absolute difference in subplots.
    
    Each comparison row: [Real, Generated, Absolute Difference]
    """
    n_comparisons = len(comparisons_data)
    fig, axes = plt.subplots(n_comparisons, 3, figsize=(15, 5 * n_comparisons))
    
    # Handle single comparison case
    if n_comparisons == 1:
        axes = [axes]
    
    # Create discrete colormap with 22 levels from 0 to 3.35
    n_levels = 22
    vmin = 0.0
    vmax = 3.35
    boundaries = np.linspace(vmin, vmax, n_levels + 1)
    cmap = plt.get_cmap(CMAP, n_levels).copy()
    cmap.set_bad("white")
    norm = colors.BoundaryNorm(boundaries, cmap.N)
    
    for row_idx, data in enumerate(comparisons_data):
        real = data['real']
        generated = data['generated']
        label = data['label']
        
        # Apply mask
        real_masked = np.ma.masked_where(~mask, real)
        generated_masked = np.ma.masked_where(~mask, generated)
        
        # Calculate absolute difference
        abs_diff = np.abs(real - generated)
        abs_diff_masked = np.ma.masked_where(~mask, abs_diff)
        
        # Plot Real
        ax_real = axes[row_idx][0]
        img_real = ax_real.imshow(
            real_masked,
            cmap=cmap,
            norm=norm,
            interpolation="bicubic",
            aspect="equal"
        )
        ax_real.set_title(f"{label} - Real", fontsize=12)
        ax_real.set_xlabel("X")
        ax_real.set_ylabel("Y")
        plt.colorbar(img_real, ax=ax_real, fraction=0.046, pad=0.04)
        
        # Plot Generated
        ax_gen = axes[row_idx][1]
        img_gen = ax_gen.imshow(
            generated_masked,
            cmap=cmap,
            norm=norm,
            interpolation="bicubic",
            aspect="equal"
        )
        ax_gen.set_title(f"{label} - Generated", fontsize=12)
        ax_gen.set_xlabel("X")
        ax_gen.set_ylabel("Y")
        plt.colorbar(img_gen, ax=ax_gen, fraction=0.046, pad=0.04)
        
        # Plot Absolute Difference
        ax_diff = axes[row_idx][2]
        img_diff = ax_diff.imshow(
            abs_diff_masked,
            cmap="hot",
            vmin=0,
            vmax=1.0,
            interpolation="bicubic",
            aspect="equal"
        )
        ax_diff.set_title(f"{label} - |Real - Generated|", fontsize=12)
        ax_diff.set_xlabel("X")
        ax_diff.set_ylabel("Y")
        cbar_diff = plt.colorbar(img_diff, ax=ax_diff, fraction=0.046, pad=0.04)
        cbar_diff.set_label("Abs Diff (Ohm)")
        
        # Print statistics
        mae = np.mean(abs_diff_masked.compressed())
        max_diff = abs_diff_masked.compressed().max()
        print(f"{label}: MAE = {mae:.4f}, Max Diff = {max_diff:.4f}")
    
    plt.suptitle("Heatmap Comparison: Real vs Generated", fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Ensure output directory exists
    output_path = FilePath(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved comparison plot: {output_path}")


def main():
    # Get repo root
    script_dir = FilePath(__file__).resolve().parent
    repo_root = script_dir.parent
    
    # Load mask
    mask_path = repo_root / MASK_PATH
    mask = np.load(mask_path).astype(bool)
    print(f"Loaded mask: {mask.shape}")
    
    # Load normalization stats
    stats_path = repo_root / NORMALIZATION_STATS_PATH
    stats = load_normalization_stats(stats_path)
    stats = stats.get(stats_type, stats[stats_type])  # Use specified stats type or default 
    print(f"Loaded normalization stats - heatmap min: {stats['heatmap_min']:.4f}, max: {stats['heatmap_max']:.4f}")
    
    # Load all comparisons
    comparisons_data = []
    for comp in COMPARISONS:
        # Load generated heatmap
        gen_path = repo_root / comp["generated"]
        generated = load_generated_heatmap(gen_path, stats=stats)
        print(f"Loaded generated heatmap for {comp['label']}: {generated.shape}")
        
        # Load real heatmap from .map file
        real_path = repo_root / comp["real"]
        real = load_map_file(real_path, resolution=generated.shape[0])
        print(f"Loaded real heatmap for {comp['label']}: {real.shape}")
        
        comparisons_data.append({
            "generated": generated,
            "real": real,
            "label": comp["label"]
        })
    
    # Plot comparison
    plot_comparison(
        comparisons_data=comparisons_data,
        mask=mask,
        output_path= OUTPUT_PATH
    )
    
    # Print overall statistics
    print("\n" + "=" * 60)
    print("HEATMAP COMPARISON STATISTICS")
    print("=" * 60)
    for data in comparisons_data:
        real_valid = data['real'][mask]
        gen_valid = data['generated'][mask]
        print(f"\n{data['label']}:")
        print(f"  Real      - Min: {real_valid.min():.4f}, Max: {real_valid.max():.4f}, Mean: {real_valid.mean():.4f}")
        print(f"  Generated - Min: {gen_valid.min():.4f}, Max: {gen_valid.max():.4f}, Mean: {gen_valid.mean():.4f}")


if __name__ == "__main__":
    main()
