import numpy as np
import pandas as pd
from pathlib import Path

# Create a 7x8 occupancy grid initialized to zeros (free space)
H=7
W=8
# Step 1: Flatten grid coordinates in row-major order
all_coords = [(i, j) for i in range(H) for j in range(W)]
# Mark invalid cells
invalid_cords = [(0,3), (0,4), (6,3), (6,4)]
# Step 2: Filter out invalid points
valid_coords = [coord for coord in all_coords if coord not in invalid_cords]
# labels for the grid
labels = [
    "C4","C5","C6","C1","C2","C3","C7","C8","C9","C10",
    "C11","C12","C13","C14","C15","C16","C17","C18","C19","C20",
    "C21","C22","C23","C24","C25","C26","C27","C28","C29","C30",
    "C31","C32","C33","C34","C35","C36","C37","C38","C39","C40",
    "C41","C42","C43","C44","C45","C46","C47","C48","C49","C50",
    "C51","C52"
]

# Step 3: Map labels to valid coordinates
grid_dict = {coord: label for coord, label in zip(valid_coords, labels)}


# Check result
print(grid_dict)

labels_v1 = [
    "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10",
    "C11","C12","C13","C14","C15","C16","C17","C18","C19","C20",
    "C21","C22","C23","C24","C25","C26","C27","C28","C29","C30",
    "C31","C32","C33","C34","C35","C36","C37","C38","C39","C40",
    "C41","C42","C43","C44","C45","C46","C47","C48","C49","C50",
    "C51","C52"
]

# Step 3: load all csv data files and save grids sequentially
csv_files = sorted(Path(".").glob("Decaps*.csv"))
occ_dir = Path("Occ_map")
occ_dir.mkdir(parents=True, exist_ok=True)

# continue numbering based on existing saved samples
existing = sorted(occ_dir.glob("sample_*.npy"))
sample_idx = len(existing) + 1

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    for row in range(len(df)):
        vector = df.iloc[row].values
        vector_idx = [j for j, val in enumerate(vector) if val == 1]
        values_idx = [labels_v1[idx] for idx in vector_idx]

        # create the grid array
        grid = np.zeros((H, W), dtype=int)
        for coord, label in grid_dict.items():
            if label in values_idx:
                grid[coord] = 1

        np.save(occ_dir / f"sample_{sample_idx}.npy", grid)
        print(f"Saved {csv_path.name} row {row} as sample_{sample_idx}.npy with labels {values_idx}")
        sample_idx += 1

   
