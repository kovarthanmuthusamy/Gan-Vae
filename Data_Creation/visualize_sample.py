"""Combined visualization tool for heatmap, impedance, and occupancy samples."""
import argparse
from pathlib import Path
import sys

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent))

from heatmap import visualize_heatmap
from impedance import visualize_impedance
from occupancy import visualize_occupancy_grid

def visualize_sample(sample_idx, data_root="src/data_2", show=True, output_pth=None):
    """Visualize all three modalities for a single sample.
    
    Args:
        sample_idx: Sample number (e.g., 1 for sample_1.npy)
        data_root: Root directory containing heatmap/, Imp/, Occ_map/ folders
        show: Whether to display plots
    """
    data_root = Path(data_root)
    output_pth = Path(output_pth) if output_pth else None
    
    heatmap_file = data_root / "heatmap" / f"sample_{sample_idx}.npy"
    impedance_file = data_root / "Imp" / f"sample_{sample_idx}.npy"
    occ_file = data_root / "Occ_map" / f"sample_{sample_idx}.npy"

    print(f"\n{'='*60}")
    print(f"Visualizing Sample {sample_idx}")
    print(f"{'='*60}")
    
    # Visualize heatmap
    if heatmap_file.exists():
        print(f"\n✓ Loading heatmap from {heatmap_file.name}")
        heatmap_output = None
        if output_pth:
            heatmap_dir = output_pth / "heatmap"
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            heatmap_output = heatmap_dir / f"sample_{sample_idx}.png"
        visualize_heatmap(heatmap_file, output_path=heatmap_output, show=show)
    else:
        print(f"✗ Heatmap file not found: {heatmap_file}")
    
    # Visualize impedance
    if impedance_file.exists():
        print(f"✓ Loading impedance from {impedance_file.name}")
        impedance_output = None
        if output_pth:
            impedance_dir = output_pth / "impedance"
            impedance_dir.mkdir(parents=True, exist_ok=True)
            impedance_output = impedance_dir / f"sample_{sample_idx}.png"
        visualize_impedance(impedance_file, output_path=impedance_output, show=show)
    else:
        print(f"✗ Impedance file not found: {impedance_file}")
    
    # Visualize occupancy grid
    if occ_file.exists():
        print(f"✓ Loading occupancy grid from {occ_file.name}")
        occupancy_output = None
        if output_pth:
            occupancy_dir = output_pth / "occupancy"
            occupancy_dir.mkdir(parents=True, exist_ok=True)
            occupancy_output = occupancy_dir / f"sample_{sample_idx}.png"
        visualize_occupancy_grid(occ_file, output_path=occupancy_output, show=show)
    else:
        print(f"✗ Occupancy grid file not found: {occ_file}")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training dataset samples")
    parser.add_argument("sample_idx", type=int, help="Sample index to visualize (e.g., 1 for sample_1.npy)")
    parser.add_argument("--data-root", type=str, default="src/data_2", help="Root directory for dataset")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots (useful for batch processing)")
   # parser.add_argument("--output-path", type=str, default=None, help="Directory to save visualizations")
    args = parser.parse_args()
    
    visualize_sample(args.sample_idx, args.data_root, show=not args.no_show, output_pth="temp_visuals")