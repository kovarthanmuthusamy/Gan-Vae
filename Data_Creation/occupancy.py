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
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot raw binary occupancy grid
    im = ax.imshow(grid, cmap='binary', vmin=0, vmax=1, interpolation='nearest', origin='lower')
    ax.set_title('Occupancy Grid (Labels shown for occupied cells only)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    
    # Add grid lines for clarity
    ax.set_xticks(np.arange(OCC_GRID_W))
    ax.set_yticks(np.arange(OCC_GRID_H))
    ax.grid(which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add labels only for cells that are occupied (value = 1)
    for row in range(OCC_GRID_H):
        for col in range(OCC_GRID_W):
            if grid[row, col] == 1:  # Only show labels for occupied cells
                # Get the capacitor label for this position
                if (row, col) in OCC_GRID_MAP:
                    label = OCC_GRID_MAP[(row, col)]
                    # Add text annotation - white text on occupied (black) cells
                    ax.text(col, row, label, ha='center', va='center', 
                           color='white', fontsize=10, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Occupancy (0=empty, 1=occupied)')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Occupancy grid visualization saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def csv_row_to_occupancy(csv_row, expected_size=56):
    """
    Directly convert a CSV row to occupancy grid data without complex mapping.
    
    This function treats the CSV row values directly as occupancy samples,
    bypassing the capacitor layout mapping logic. Useful when CSV data is
    already in the desired format.
    
    Args:
        csv_row: Array-like or list of values from a CSV row. Can be:
                 - 56 elements (will reshape to 7x8 with invalid cells)
                 - 52 elements (valid cells only, will be placed in 7x8 grid)
                 - Already shaped as (7, 8)
        expected_size: Expected number of elements (default 56 for full 7x8 grid)
    
    Returns:
        np.ndarray: Occupancy grid of shape (7, 8) with binary values (0 or 1)
    
    Examples:
        >>> # From CSV with 56 values (full grid)
        >>> row = [0, 0, 0, 1, 1, 0, 0, 0, ...]  # 56 values
        >>> occ_grid = csv_row_to_occupancy(row)
        >>> occ_grid.shape
        (7, 8)
        
        >>> # From CSV with 52 values (excluding invalid cells)
        >>> row = [0, 0, 0, 1, 1, 0, ...]  # 52 values
        >>> occ_grid = csv_row_to_occupancy(row, expected_size=52)
    """
    # Convert to numpy array
    data = np.array(csv_row, dtype=np.float32)
    
    # Handle different input formats
    if data.ndim == 2 and data.shape == (OCC_GRID_H, OCC_GRID_W):
        # Already in correct shape
        grid = data
    elif data.size == OCC_GRID_H * OCC_GRID_W:
        # Full 56 elements - reshape directly
        grid = data.reshape(OCC_GRID_H, OCC_GRID_W)
    elif data.size == 52:
        # 52 valid cells - need to place in 7x8 grid excluding invalid cells
        grid = np.zeros((OCC_GRID_H, OCC_GRID_W), dtype=np.float32)
        
        # Fill valid cells in row-major order
        data_idx = 0
        for row in range(OCC_GRID_H):
            for col in range(OCC_GRID_W):
                if (row, col) not in INVALID_OCC_CELLS:
                    grid[row, col] = data[data_idx]
                    data_idx += 1
    else:
        raise ValueError(
            f"CSV row has {data.size} elements. Expected {OCC_GRID_H * OCC_GRID_W} "
            f"(full grid) or 52 (valid cells only), or already shaped as ({OCC_GRID_H}, {OCC_GRID_W})"
        )
    
    # Binarize: threshold at 0.5
    grid_binary = (grid > 0.5).astype(np.uint8)
    
    return grid_binary


def csv_to_occupancy_samples(csv_file, output_dir, start_idx=0):
    """
    Batch convert CSV rows directly to occupancy samples.
    
    Reads a CSV file where each row represents an occupancy sample,
    and saves each row as a separate .npy file.
    
    Args:
        csv_file: Path to CSV file (each row = one occupancy sample)
        output_dir: Directory to save occupancy .npy files
        start_idx: Starting index for output filenames (default 0)
    
    Returns:
        int: Number of samples processed
    
    Example:
        >>> csv_to_occupancy_samples('occupancy_data.csv', 'output/occ_samples/')
        Saved 1000 occupancy samples to output/occ_samples/
    """
    import csv
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        
        # Skip header if present (optional: auto-detect numeric vs string first row)
        first_row = next(reader)
        try:
            # Try to convert first row to numbers
            csv_row_to_occupancy(first_row)
            # If successful, process this row
            rows_to_process = [first_row]
        except (ValueError, TypeError):
            # First row is header, skip it
            rows_to_process = []
        
        # Process all rows
        for row in reader:
            rows_to_process.append(row)
        
        for row in rows_to_process:
            occ_grid = csv_row_to_occupancy(row)
            output_path = output_dir / f"sample_{start_idx + count:06d}.npy"
            np.save(output_path, occ_grid)
            count += 1
    
    print(f"Saved {count} occupancy samples to {output_dir}")
    return count
