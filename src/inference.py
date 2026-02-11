import torch
import numpy as np
from pathlib import Path
import sys, json
import matplotlib.pyplot as plt

from models.model_v1 import Generator

# Add repo-level scripts to path for visualization functions
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
try:
    from impedance_visuals import plot_impedance_profile # type: ignore
except ImportError:
    print("Warning: impedance_visuals module not found. plot_impedance_profile will not be available.")
    plot_impedance_profile = None

# ============================================
# Configuration - Update this for each run
# ============================================
CONFIG = {
    "experiment_name": "exp007",  # Align with current training setup
    "epoch": 200,  # Checkpoint epoch to load
    "foldernumber": "3", # Subdirectory for visuals
    "latent_dim": 200,
    "shared_dim": 512,
    "device": "cuda:1",
    "binarize_outputs": False,  # Optionally binarize occupancy/impedance on inference
}

def load_normalization_stats(stats_path):
    """Load normalization statistics"""
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    return stats

def denormalize_data(normalized_data, data_type, stats):
    """
    Denormalize data using min-max inverse transformation.
    
    normalized_data: values in [0, 1]
    data_type: 'heatmap' or 'impedance'
    stats: normalization statistics dict
    
    Returns: denormalized data in original range
    """
    # Use percentile min-max for denormalization
    norm_stats = stats['percentile_min_max']
    
    if data_type == 'heatmap':
        min_val = norm_stats['heatmap_min']
        max_val = norm_stats['heatmap_max']
    elif data_type == 'impedance':
        min_val = norm_stats['imp_min']
        max_val = norm_stats['imp_max']
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    # Inverse min-max normalization: x_original = x_normalized * (max - min) + min
    denormalized = normalized_data * (max_val - min_val) + min_val
    return denormalized


def inference(checkpoint_path, latent_dim=256, shared_dim=512, device="cuda:1", binarize=False):
    """Run inference on a single sample (2-channel heatmap + separate occupancy)."""

    device = torch.device(device)

    # Load model
    G = Generator(latent_dim, shared_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    G.load_state_dict(checkpoint["G"])
    G.eval()

    print(f"Loaded checkpoint from {checkpoint_path}")

    # Generate single sample
    with torch.no_grad():
        z = torch.randn(1, latent_dim, device=device)
        fake_heatmap, fake_occupancy, fake_impedance = G(z)

        # Optional postprocessing: binarize occupancy/impedance for visualization consistency
        if binarize:
            fake_occupancy = (fake_occupancy > 0.5).float()
            fake_impedance = (fake_impedance > 0.5).float()

    # Convert to numpy and remove batch dimension
    fake_heatmap = fake_heatmap.cpu().numpy()[0]    # (C=2, H=64, W=64)
    fake_occupancy = fake_occupancy.cpu().numpy()[0]  # (C=1, H=7, W=8)
    fake_impedance = fake_impedance.cpu().numpy()[0]  # (231,)

    # Load mask and repeat to 2 channels
    mask = np.load(REPO_ROOT / "configs" / "binary_mask.npy").astype(np.float32)
    if mask.ndim == 2:
        mask = np.expand_dims(mask, 0)  # (1, H, W)
    if mask.shape[0] == 1:
        mask = np.repeat(mask, 2, axis=0)  # (2, H, W) to match heatmap

    # Apply mask to heatmap (channel-wise)
    fake_heatmap_masked = fake_heatmap * mask

    print(f"Generated heatmap shape: {fake_heatmap_masked.shape}")
    print(f"Generated occupancy shape: {fake_occupancy.shape}")
    print(f"Generated impedance shape: {fake_impedance.shape}")

    return fake_heatmap_masked, fake_occupancy, fake_impedance



def plot_heatmap_array(
    heatmap,
    output_path,
    title="Generated Heatmap (Channel 0)",
    vmin=None,
    vmax=None,
):
    """Plot the first channel of the 2-channel heatmap."""
    heatmap = np.asarray(heatmap)

    fig, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(
        heatmap[0],  # Channel 0 (impedance heatmap)
        cmap="jet",
        # origin="lower", # Removed to match data orientation (row 0 is Top)
        interpolation="bilinear",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def build_occupancy_grid_dict():
    """Build the same grid_dict as in Data_Creation/occ_grid.py"""
    H, W = 7, 8
    all_coords = [(i, j) for i in range(H) for j in range(W)]
    invalid_cords = [(0, 3), (0, 4), (6, 3), (6, 4)]
    valid_coords = [coord for coord in all_coords if coord not in invalid_cords]
    
    labels_v1 = [
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
        "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20",
        "C21", "C22", "C23", "C24", "C25", "C26", "C27", "C28", "C29", "C30",
        "C31", "C32", "C33", "C34", "C35", "C36", "C37", "C38", "C39", "C40",
        "C41", "C42", "C43", "C44", "C45", "C46", "C47", "C48", "C49", "C50",
        "C51", "C52"
    ]
    grid_dict = {coord: label for coord, label in zip(valid_coords, labels_v1)}
    return grid_dict, labels_v1


def plot_occupancy_map(occupancy, output_path, title="Generated Occupancy Map (Binary)"):
    """Plot occupancy map with labels as legend."""
    occ = np.asarray(occupancy)
    if occ.ndim == 3 and occ.shape[0] == 1:
        occ = occ[0]
    
    # Binarize the occupancy map
    occ = (occ > 0.5).astype(float)
    
    # Build grid_dict to map coordinates to labels
    grid_dict, labels_v1 = build_occupancy_grid_dict()
    
    # Extract occupied labels
    occupied_labels = []
    for (i, j), label in grid_dict.items():
        if occ[i, j] > 0.5:
            occupied_labels.append(label)
    
    print(f"Occupied labels: {occupied_labels}")

    fig, ax = plt.subplots(figsize=(6, 5))
    img = ax.imshow(
        occ,
        cmap="Greys",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title(title)
    ax.set_xlabel("X (Grid Column)")
    ax.set_ylabel("Y (Grid Row)")
    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04, label="Occupied")
    
    # Add legend with occupied labels
    legend_text = "Occupied Labels:\n" + ", ".join(occupied_labels) if occupied_labels else "No cells occupied"
    fig.text(0.5, -0.05, legend_text, ha='center', fontsize=10, wrap=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    # Use target checkpoint
    checkpoint_dir = REPO_ROOT / "experiments" / CONFIG["experiment_name"] / "checkpoints"
    checkpoint_path = checkpoint_dir / f"epoch_{CONFIG['epoch']}.pt"
    print(f"Using checkpoint: {checkpoint_path}")

    heatmap, occupancy, impedance = inference(
        str(checkpoint_path),
        latent_dim=CONFIG["latent_dim"],
        shared_dim=CONFIG["shared_dim"],
        device=CONFIG["device"],
        binarize=CONFIG.get("binarize_outputs", False),
    )

    # Visualize results
    stats_path = REPO_ROOT / "src" / "data_norm_2" / "normalization_stats.json"
    stats = load_normalization_stats(stats_path)
    heatmap_denorm = denormalize_data(heatmap, 'heatmap', stats)
    impedance_denorm = denormalize_data(impedance, 'impedance', stats)

    # Mask out background for visualization (set to NaN)
    mask = np.load(REPO_ROOT / "configs" / "binary_mask.npy").astype(np.float32)
    if mask.ndim == 2:
        mask = mask[np.newaxis, :, :]
    heatmap_denorm_masked = heatmap_denorm.copy()
    heatmap_denorm_masked[:, mask[0] == 0] = np.nan

    # Create output directory
    visuals_dir = REPO_ROOT / "experiments" / CONFIG["experiment_name"] / "visuals" / str(CONFIG["foldernumber"])
    visuals_dir.mkdir(parents=True, exist_ok=True)

    # Plot denormalized heatmap channel 0 (impedance heatmap)
    plot_heatmap_array(
        heatmap_denorm_masked,
        str(visuals_dir / "heatmap_channel0_denorm.png"),
        title="Generated Heatmap (Channel 0, Denormalized)",
        vmin=stats['percentile_min_max']['heatmap_min'],
        vmax=stats['percentile_min_max']['heatmap_max']
    )

    # Plot occupancy map (separate input)
    plot_occupancy_map(
        occupancy,
        str(visuals_dir / "occupancy_map.png"),
        title="Generated Occupancy Map",
    )

    # Plot impedance profile
    if plot_impedance_profile:
        plot_impedance_profile(impedance_denorm, str(visuals_dir / "impedance_profile.png")) # type: ignore
    else:
        print("Skipping impedance profile plot (module not found).")

    # Save impedance magnitude as CSV (frequency loaded from configs when plotting)
    import pandas as pd
    impedance_df = pd.DataFrame({
        'Impedance_Ohms': impedance_denorm.flatten()
    })
    csv_path = visuals_dir / "gen_impedance_profile.csv"
    impedance_df.to_csv(csv_path, index=False)
    print(f"Saved impedance profile CSV: {csv_path}")

    # Save heatmap channel 0 as CSV (for future plotting)
    heatmap_ch0 = heatmap_denorm_masked[0]  # Channel 0, shape (H, W)
    heatmap_df = pd.DataFrame(heatmap_ch0)
    heatmap_npy_path = visuals_dir / "heatmap_channel_0.npy"
    np.save(heatmap_npy_path, heatmap_ch0)
    print(f"Saved heatmap channel 0 NPY: {heatmap_npy_path}")
