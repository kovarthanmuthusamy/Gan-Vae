import os
import numpy as np
import torch
import torch.nn.functional as F

# --------- SETTINGS ----------
dataset_folder = "src/data_2/Occ_map"  # folder with 3-channel .npy files
output_folder = "src/data_norm/Occ_map"  # save to new folder
apply_sigmoid = True  # Apply sigmoid to binary channels (occupancy, mask)
sigmoid_alpha = 5.0   # Sharpening factor for sigmoid
# --------- END SETTINGS ----------

os.makedirs(output_folder, exist_ok=True)

def process_file(file_path, output_path, soft=True, alpha=5.0):
    """Process 3-channel heatmap data: (impedance, occupancy, mask)"""
    # Load .npy (expected: 3 channels)
    image = np.load(file_path)
    image = torch.tensor(image, dtype=torch.float32)

    # Ensure (C,H,W) format
    if image.shape[-1] == 3 and image.ndim == 3:  # (H,W,C)
        image = torch.permute(image, (2, 0, 1))  # → (C,H,W)
    
    if image.shape[0] != 3:
        raise ValueError(f"Expected 3 channels, got {image.shape} in {file_path}")

    # Extract channels
    impedance = image[0]     # Channel 0: impedance/heatmap (continuous)
    occupancy = image[1]     # Channel 1: occupancy (binary)
    mask = image[2]          # Channel 2: mask/validity (binary)

    # Apply sigmoid to binary channels for smoothness
    if soft:
        # Smooth sigmoid: sigmoid(alpha * (x - 0.5)) for x in [0,1]
        # This creates continuous values in (0,1) with sharp transitions
        occupancy_smooth = torch.sigmoid(alpha * (occupancy - 0.5))
        mask_smooth = torch.sigmoid(alpha * (mask - 0.5))
    else:
        occupancy_smooth = occupancy
        mask_smooth = mask
    
    # Recombine: keep all 3 channels with processed occupancy and mask
    processed_image = torch.stack([impedance, occupancy_smooth, mask_smooth], dim=0)
    
    # Save to output (don't overwrite!)
    processed_np = processed_image.numpy()
    np.save(output_path, processed_np)
    print(f"Processed: {file_path} → {output_path}, shape: {processed_np.shape}")

# Process all files
for fname in os.listdir(dataset_folder):
    if fname.endswith(".npy"):
        input_path = os.path.join(dataset_folder, fname)
        output_path = os.path.join(output_folder, fname)
        process_file(input_path, output_path, soft=apply_sigmoid, alpha=sigmoid_alpha)

print("✓ Processing complete!")
