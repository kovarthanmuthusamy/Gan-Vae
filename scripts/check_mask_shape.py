import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[1]
    mask_file = repo_root / "configs/binary_mask.npy"
    
    mask_data = np.load(mask_file).astype(np.float32)
    
    print(f"Shape: {mask_data.shape}")
    print(f"Min: {mask_data.min()}, Max: {mask_data.max()}")
    
    plt.imshow(mask_data, cmap="gray", origin="lower")
    plt.colorbar()
    plt.title("Mask")
    plt.savefig("mask_debug.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()

