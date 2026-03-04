"""Occupancy grid creation from decap vectors."""
import numpy as np
import matplotlib.pyplot as plt

OCC_GRID_H, OCC_GRID_W = 7, 8
INVALID_OCC_CELLS = [(0, 3), (0, 4), (6, 3), (6, 4)]

# LABELS_ORDERED maps decap_vector indices to capacitor labels
# decap_vector[0] = C1, decap_vector[1] = C2, etc.
LABELS_ORDERED = [
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20",
    "C21", "C22", "C23", "C24", "C25", "C26", "C27", "C28", "C29", "C30",
    "C31", "C32", "C33", "C34", "C35", "C36", "C37", "C38", "C39", "C40",
    "C41", "C42", "C43", "C44", "C45", "C46", "C47", "C48", "C49", "C50",
    "C51", "C52"
]

# Physical board layout (first row, left to right): C4, C5, C6, [skip], [skip], C1, C2, C3
# This mapping connects decap_vector index (C1=0, C2=1...) to grid positions
def _create_occupancy_grid_map():
    """Create 7x8 occupancy grid with correct physical position mapping."""
    # Define physical layout: which capacitor is at each (row, col) position
    physical_layout = {
        # Row 0 (bottom row with origin='lower'): C4, C5, C6, [skip], [skip], C1, C2, C3
        (0, 0): "C4", (0, 1): "C5", (0, 2): "C6",
        (0, 5): "C1", (0, 6): "C2", (0, 7): "C3",
    }
    
    # Add remaining rows in sequential order
    label_idx = 7  # Start from C7
    for row in range(1, OCC_GRID_H):
        for col in range(OCC_GRID_W):
            if (row, col) not in INVALID_OCC_CELLS:
                physical_layout[(row, col)] = f"C{label_idx}"
                label_idx += 1
    
    return physical_layout

OCC_GRID_MAP = _create_occupancy_grid_map()

# Pre-compute reverse mapping for O(1) lookup - maps label to (row, col)
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
