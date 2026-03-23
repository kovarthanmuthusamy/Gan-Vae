import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------
# Parameters for visualization - Update these for each run
#----------------------------------------------------------------------------
# Range of samples to process (e.g., 1 to 4 will process data_sample_1, data_sample_2, etc.)
sample_start = 1
sample_end = 4
experiment = "exp012"

#----------------------------------------------------------------------------
# Visualization function for occupancy maps
#---------------------------------------------------------------------------    


def build_occupancy_grid_dict():
    """Build the same grid_dict as in Data_Creation/occ_grid.py"""
    H, W = 7, 8
    all_coords = [(i, j) for i in range(H) for j in range(W)]
    invalid_coords = [(0, 3), (0, 4), (6, 3), (6, 4)]
    valid_coords = [coord for coord in all_coords if coord not in invalid_coords]
    
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

def plot_occupancy_map(occupancy, output_dir):
    """Plot occupancy map with labels as legend."""

    ## define output directory for saving occupied labels csv file
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ##intialize the occupency map and binarize it
    occ = np.asarray(occupancy)
    if occ.ndim == 3 and occ.shape[0] == 1:
        occ = occ[0]
    
    # Binarize the occupancy map
    occ = (occ > 0.5).astype(float)
    
    # Build grid_dict to map coordinates to labels
    grid_dict, labels_v1 = build_occupancy_grid_dict()

    ##set title and vmin/vmax for binary occupancy map
    title = "Generated Occupancy Map (Binary)"
    vmin, vmax = 0.0, 1.0
    
    # Extract occupied labels
    occupied_labels = []
    for (i, j), label in grid_dict.items():
        if occ[i, j] > 0.5:
            occupied_labels.append(label)
    
    print(f"Occupied labels: {occupied_labels}")

    ## save the occupied labels in a csv file in the same directory as the output image
    labels_output_path = output_dir / "occupied_labels.csv"
    with open(labels_output_path, "w") as f:
        f.write("Occupied Labels\n")
        for label in occupied_labels:
            f.write(f"{label}\n")
    print(f"Saved occupied labels to: {labels_output_path}")

    fig, ax = plt.subplots(figsize=(6, 5))
    img = ax.imshow(
        occ,
        cmap="Greys",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("X (Grid Column)")
    ax.set_ylabel("Y (Grid Row)")
    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04, label="Occupied")
    
    # Add legend with occupied labels
    legend_text = "Occupied Labels:\n" + ", ".join(occupied_labels) if occupied_labels else "No cells occupied"
    fig.text(0.5, -0.05, legend_text, ha='center', fontsize=10, wrap=True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "occupancy_map_visual.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir / 'occupancy_map_visual.png'}")


#----------------------------------------------------------------------------
# Main execution
#----------------------------------------------------------------------------
if __name__ == "__main__":
    for sample_num in range(sample_start, sample_end + 1):
        occupancy_path = f"experiments/{experiment}/visuals/data_sample_{sample_num}/occupancy_map.npy"
        output_dir = f"experiments/{experiment}/visuals/data_sample_{sample_num}"
        
        print(f"\n{'='*60}")
        print(f"Processing data_sample_{sample_num}")
        print(f"{'='*60}")
        
        try:
            occupancy = np.load(occupancy_path)
            plot_occupancy_map(occupancy, output_dir)
            print(f"✓ Successfully processed data_sample_{sample_num}")
        except FileNotFoundError:
            print(f"✗ File not found: {occupancy_path}")
        except Exception as e:
            print(f"✗ Error processing data_sample_{sample_num}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Completed processing samples {sample_start} to {sample_end}")
    print(f"{'='*60}")