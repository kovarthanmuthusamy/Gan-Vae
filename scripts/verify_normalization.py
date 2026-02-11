import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from pathlib import Path
import json
import sys

#=========================================================================
# Script to verify normalization of heatmap, impedance, and occupancy data
# - Loads normalized data and corresponding global min/max stats
# - Denormalizes using percentile min/max stats
# - Heatmap: Linear min-max normalization
# - Impedance: LOG-SCALE min-max normalization (preserves frequency profile better)
# - Occupancy: No normalization (binary data)
# - Plots normalized and denormalized data side by side for visual verification
#=========================================================================  
#Initialze variables for normalization stats and data directories
norm_type = "percentile_min_max"  # Change to "global_min_max" if needed
output_dir = Path("temp_visuals")
output_dir.mkdir(parents=True, exist_ok=True)   

def denormalize(data, vmin, vmax):
    """Denormalize linear min-max normalized data"""
    return data * (vmax - vmin) + vmin

def denormalize_log(data, log_min, log_max):
    """Denormalize log-scale normalized data"""
    # First denormalize from [0, 1] to log-space
    log_data = data * (log_max - log_min) + log_min
    # Then exponentiate to get original scale
    return np.exp(log_data)


def plot_impedance_normalized(ax, imp_norm, imp_denorm):
    """Plot normalized and denormalized impedance on log-log scale"""
    frequency = np.arange(len(imp_norm.flatten()))
    
    # Plot both normalized and denormalized on the same axis
    ax.loglog(
            frequency, 
            imp_norm.flatten(), 
            linestyle='--', 
            linewidth=1, 
            color='blue', 
            label='Normalized Impedance'
        )
    ax.loglog(
            frequency, 
            imp_denorm.flatten(), 
            linestyle='-', 
            linewidth=1, 
            color='green', 
            label='Denormalized Impedance'
        )
    ax.set_xlabel("Frequency Index")
    ax.set_ylabel("Impedance Value")
    ax.set_title("Impedance: Normalized vs Denormalized")
    ax.legend()
    ax.grid(True, alpha=0.3)


# Create discrete jet colormap with 22 colors
cmap_jet_discrete = cm.get_cmap('jet', 22)

repo_root = Path(__file__).resolve().parents[1]

norm_root = repo_root / "datasets" / "source" / "data_norm"

heatmap_norm_dir = norm_root / "heatmap"
imp_norm_dir = norm_root / "Imp"
occ_norm_dir = norm_root / "Occ_map"

stats_file = norm_root / "normalization_stats.json"
with open(stats_file, "r") as f:
    stats_data = json.load(f)

percentile_stats = stats_data["percentile_min_max"]
global_stats = stats_data["global_min_max"]

heatmap_files = sorted(heatmap_norm_dir.glob("*.npy"))
imp_files = sorted(imp_norm_dir.glob("*.npy"))
occ_files = sorted(occ_norm_dir.glob("*.npy"))

heatmap_file = heatmap_files[2]
imp_file = imp_files[2]
occ_file = occ_files[2]

heatmap_norm = np.load(heatmap_file).astype(np.float32)
imp_norm = np.load(imp_file).astype(np.float32)
occ_norm = np.load(occ_file).astype(np.float32)

heatmap_denorm = denormalize(heatmap_norm[0], percentile_stats["heatmap_min"], percentile_stats["heatmap_max"])
imp_denorm = denormalize_log(imp_norm, percentile_stats["imp_log_min"], percentile_stats["imp_log_max"])
occ_denorm = denormalize(occ_norm, percentile_stats.get("occ_min", 0.0), percentile_stats.get("occ_max", 1.0))

print(f"Heatmap file: {heatmap_file.name}")
print(f"Impedance file: {imp_file.name}")
print(f"Occupancy file: {occ_file.name}")
print(f"Heatmap shape: {heatmap_norm.shape}")
print(f"Impedance shape: {imp_norm.shape}")
print(f"Occupancy shape: {occ_norm.shape}")

fig, axes = plt.subplots(3, 3, figsize=(18, 12))

# Row 0: Heatmap (impedance channel) - normalized with 1-99 percentile
im_norm = axes[0, 0].imshow(heatmap_norm[0], cmap=cmap_jet_discrete)
axes[0, 0].set_title("Heatmap Norm (1-99%ile)")
fig.colorbar(im_norm, ax=axes[0, 0], fraction=0.046, pad=0.04)

im_denorm = axes[0, 1].imshow(heatmap_denorm, cmap=cmap_jet_discrete)
axes[0, 1].set_title("Heatmap Denorm (1-99%ile)")
fig.colorbar(im_denorm, ax=axes[0, 1], fraction=0.046, pad=0.04)

# Row 1: Impedance (using log-log scale like Impedance_visuals.py)
plot_impedance_normalized(axes[1, 0], imp_norm, imp_denorm)
axes[1, 1].axis("off")
axes[1, 2].axis("off")

# Row 2: Occupancy - copied as-is (binary, no normalization)
im_occ_norm = axes[2, 0].imshow(occ_norm.squeeze(), cmap="binary")
axes[2, 0].set_title("Occupancy (Binary)")
fig.colorbar(im_occ_norm, ax=axes[2, 0], fraction=0.046, pad=0.04)

im_occ_denorm = axes[2, 1].imshow(occ_denorm.squeeze(), cmap="binary")
axes[2, 1].set_title("Occupancy (Original)")
fig.colorbar(im_occ_denorm, ax=axes[2, 1], fraction=0.046, pad=0.04)
axes[2, 2].axis("off")

plt.tight_layout()

out_path = Path(repo_root) / "temp_visuals" / "normalization_check_three_branch.png"
plt.savefig(out_path, dpi=200)
print(f"Verification plot saved to {out_path}")
plt.show()

