"""Heatmap creation with interpolation."""
import numpy as np
from scipy.interpolate import griddata, RBFInterpolator
from functools import lru_cache
from pathlib import Path
import matplotlib.pyplot as plt

HEATMAP_GRID_SIZE = 64

@lru_cache(maxsize=4)
def load_mask_board(path):
    """Load mask board from npy file."""
    return np.load(path)

def create_Heatmaps(file_path, grid_size=HEATMAP_GRID_SIZE, frame_path=None, mask_board=None, verbose=False):
    """Creates stacked heatmap with impedance and mask channels - optimized."""
    with open(file_path) as f:
        lines = f.readlines()
    
    # Fast parsing - collect data directly into list then convert
    data = []
    for i, line in enumerate(lines):
        if line.strip() == "3" and i + 3 < len(lines):
            try:
                for k in range(1, 4):
                    px, py, pz = map(float, lines[i + k].split())
                    data.append((px, py, max(pz, 0.01)))
            except ValueError:
                pass
    
    if not data:
        raise RuntimeError("No data loaded from MAP file.")
    
    # Convert to arrays once
    data_arr = np.array(data, dtype=np.float32)
    x, y, z = data_arr[:, 0], data_arr[:, 1], data_arr[:, 2]
    
    scale, x_min, y_min = 1e-5, x.min(), y.min()
    x, y = (x - x_min) * scale, (y - y_min) * scale
    
    xi, yi = np.linspace(x.min(), x.max(), grid_size), np.linspace(y.min(), y.max(), grid_size)
    Xi, Yi = np.meshgrid(xi, yi)
    
    if mask_board is None and frame_path:
        mask_board = load_mask_board(frame_path)
    
    points = np.column_stack((x, y))
    # Use linear interpolation (faster than cubic) and fill NaNs with nearest neighbor
    Zi = griddata(points, z, (Xi, Yi), method="linear")
    
    # Only use RBF for significant NaN regions (more efficient)
    nan_mask = np.isnan(Zi)
    if np.any(nan_mask):
        # Fill with nearest neighbor first (fast)
        Zi_nearest = griddata(points, z, (Xi, Yi), method="nearest")
        Zi[nan_mask] = Zi_nearest[nan_mask]
    
    # Pre-cast mask_board once
    if mask_board is None:
        mask_board = np.ones_like(Zi, dtype=np.float32)
    else:
        mask_board = mask_board.astype(np.float32)
    return np.stack([Zi * mask_board, mask_board], axis=0)

def visualize_heatmap(heatmap_file, output_path=None, show=True):
    """Visualize heatmap channels from saved numpy file - raw data display.
    
    Args:
        heatmap_file: Path to heatmap .npy file (shape: 2, H, W)
        output_path: Path to save visualization (optional)
        show: Whether to display plot
    """
    heatmap = np.load(heatmap_file)
    
    if heatmap.ndim != 3 or heatmap.shape[0] != 2:
        print(f"Error: Expected shape (2, H, W), got {heatmap.shape}")
        return
    
    impedance_ch, mask_ch = heatmap[0], heatmap[1]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot impedance channel
    im0 = axes[0].imshow(impedance_ch, cmap='jet', origin='lower')
    axes[0].set_title('Impedance')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im0, ax=axes[0])
    
    # Plot mask channel
    im1 = axes[1].imshow(mask_ch, cmap='binary', origin='lower')
    axes[1].set_title('Mask')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Heatmap visualization saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
