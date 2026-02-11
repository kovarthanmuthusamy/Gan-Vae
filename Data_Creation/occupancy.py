"""Occupancy grid creation from decap vectors."""
import numpy as np
import matplotlib.pyplot as plt

OCC_GRID_H, OCC_GRID_W = 7, 8
INVALID_OCC_CELLS = [(0, 3), (0, 4), (6, 3), (6, 4)]

# Correct label ordering as per grid_dict from occ_grid.py
LABELS_ORDERED = [
    "C4", "C5", "C6", "C1", "C2", "C3", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20",
    "C21", "C22", "C23", "C24", "C25", "C26", "C27", "C28", "C29", "C30",
    "C31", "C32", "C33", "C34", "C35", "C36", "C37", "C38", "C39", "C40",
    "C41", "C42", "C43", "C44", "C45", "C46", "C47", "C48", "C49", "C50",
    "C51", "C52"
]

def _create_occupancy_grid_map():
    """Create 7x8 occupancy grid with invalid cells and correct label mapping."""
    all_coords = [(i, j) for i in range(OCC_GRID_H) for j in range(OCC_GRID_W)]
    valid_coords = [c for c in all_coords if c not in INVALID_OCC_CELLS]
    return {coord: label for coord, label in zip(valid_coords, LABELS_ORDERED)}

OCC_GRID_MAP = _create_occupancy_grid_map()

# Pre-compute reverse mapping for O(1) lookup - OPTIMIZATION
_LABEL_TO_COORD = {label: coord for coord, label in OCC_GRID_MAP.items()}

def create_occupancy_grid(decap_vector):
    """Creates HxW occupancy grid from decap vector - optimized."""
    grid = np.zeros((OCC_GRID_H, OCC_GRID_W), dtype=np.uint8)
    if decap_vector is None:
        return grid
    
    # Fast vectorized approach - find active indices
    active_idx = np.where(decap_vector > 0.5)[0]
    
    # Map indices directly to coordinates via pre-computed labels
    for idx in active_idx:
        if idx < len(LABELS_ORDERED):
            h, w = _LABEL_TO_COORD[LABELS_ORDERED[idx]]
            grid[h, w] = 1
    
    return grid

def visualize_occupancy_grid(occ_grid_file, output_path=None, show=True):
    """Visualize occupancy grid from saved numpy file - raw binary display.
    
    Args:
        occ_grid_file: Path to occupancy grid .npy file (shape: 7, 8)
        output_path: Path to save visualization (optional)
        show: Whether to display plot
    """
    grid = np.load(occ_grid_file)
    
    if grid.ndim != 2 or grid.shape != (OCC_GRID_H, OCC_GRID_W):
        print(f"Error: Expected shape ({OCC_GRID_H}, {OCC_GRID_W}), got {grid.shape}")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot raw binary occupancy grid
    im = ax.imshow(grid, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title('Occupancy Grid')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Occupancy grid visualization saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
