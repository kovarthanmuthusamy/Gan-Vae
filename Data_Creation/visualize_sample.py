"""Interactive visualization tool for heatmap, impedance, and occupancy samples."""
import matplotlib
# Try different backends for WSL compatibility
try:
    matplotlib.use('TkAgg')  # Try TkAgg first (best for WSL with X server)
except:
    try:
        matplotlib.use('Qt5Agg')  # Try Qt5 as fallback
    except:
        pass  # Use default backend

import matplotlib.pyplot as plt
from pathlib import Path
import sys
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent))

from heatmap import visualize_heatmap
from impedance import visualize_impedance
from occupancy import visualize_occupancy_grid, OCC_GRID_MAP, INVALID_OCC_CELLS


class InteractiveSampleViewer:
    """Interactive viewer with keyboard navigation."""
    
    def __init__(self, data_root="datasets/data", output_pth=None):
        self.data_root = Path(data_root)
        self.output_pth = Path(output_pth) if output_pth else None
        
        # Find all available samples
        heatmap_dir = self.data_root / "heatmap"
        if not heatmap_dir.exists():
            print(f"Error: Heatmap directory not found: {heatmap_dir}")
            self.samples = []
            return
        
        # Get all sample files and extract indices
        sample_files = sorted(heatmap_dir.glob("sample_*.npy"))
        self.samples = []
        for f in sample_files:
            try:
                idx = int(f.stem.split('_')[1])
                self.samples.append(idx)
            except (ValueError, IndexError):
                pass
        
        if not self.samples:
            print(f"Error: No samples found in {heatmap_dir}")
            return
        
        self.samples.sort()
        self.current_index = 0
        self.fig = None
        
        print(f"Found {len(self.samples)} samples in {data_root}")
        print("\n" + "="*60)
        print("INTERACTIVE SAMPLE VIEWER")
        print("="*60)
        print("Controls:")
        print("  Right Arrow / 'd' : Next sample")
        print("  Left Arrow / 'a'  : Previous sample")
        print("  'q' / ESC        : Quit")
        print("="*60 + "\n")
    
    def visualize_current_sample(self):
        """Visualize the current sample."""
        if not self.samples:
            return
        
        sample_idx = self.samples[self.current_index]
        
        plt.close('all')  # Close previous plots
        
        heatmap_file = self.data_root / "heatmap" / f"sample_{sample_idx}.npy"
        impedance_file = self.data_root / "Imp" / f"sample_{sample_idx}.npy"
        occ_file = self.data_root / "Occ_map" / f"sample_{sample_idx}.npy"
        
        # Create figure with 3 subplots in a single row
        self.fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        self.fig.suptitle(f'Sample {sample_idx} ({self.current_index + 1}/{len(self.samples)}) - Arrow Keys/A/D: Navigate | Q: Quit', 
                         fontsize=14, fontweight='bold')
        
        # 1. Visualize heatmap impedance channel only
        if heatmap_file.exists():
            heatmap = np.load(heatmap_file)
            if heatmap.ndim == 3 and heatmap.shape[0] == 2:
                impedance_ch, mask_ch = heatmap[0], heatmap[1]
                
                # Plot impedance channel only
                im0 = axes[0].imshow(impedance_ch, cmap='jet', origin='lower')
                axes[0].set_title('Heatmap', fontsize=12, fontweight='bold')
                axes[0].set_xlabel('X')
                axes[0].set_ylabel('Y')
                plt.colorbar(im0, ax=axes[0])
            else:
                axes[0].text(0.5, 0.5, f'Invalid shape: {heatmap.shape}', 
                           ha='center', va='center', transform=axes[0].transAxes)
                axes[0].set_title('Heatmap')
        else:
            axes[0].text(0.5, 0.5, 'Heatmap not found', ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Heatmap')
            print(f"✗ Heatmap file not found: {heatmap_file}")
        
        # 2. Visualize impedance curve - using logic from impedance.py
        if impedance_file.exists():
            impedance = np.load(impedance_file).flatten()
            
            # Load frequency and target impedance from configs
            freq_file = Path(__file__).parent.parent / "configs" / "Frequency_data_hz.npy"
            target_file = Path(__file__).parent.parent / "configs" / "target_impedance.npy"
            
            if freq_file.exists() and target_file.exists():
                frequency = np.load(freq_file)
                target_impedance = np.load(target_file).flatten()
                
                # Plot with loglog scale
                axes[1].loglog(frequency, impedance, marker='o', markersize=3, linestyle='-', 
                             linewidth=2, label='Generated Impedance', color='blue')
                axes[1].loglog(frequency, target_impedance, marker='s', markersize=2, linestyle='--', 
                             linewidth=2, label='Target Impedance', color='red')
                axes[1].set_xlabel('Frequency (Hz)', fontsize=12)
                axes[1].set_ylabel('Impedance (Ω)', fontsize=12)
                axes[1].legend(loc='best')
            else:
                # Fallback to simple plot if config files not found
                frequencies = np.arange(len(impedance))
                axes[1].plot(frequencies, impedance, 'b-', linewidth=1.5)
                axes[1].set_xlabel('Frequency Point')
                axes[1].set_ylabel('Impedance (Ω)')
            
            axes[1].set_title('Impedance Curve', fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3, linestyle='--')
        else:
            axes[1].text(0.5, 0.5, 'Impedance not found', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Impedance')
            print(f"✗ Impedance file not found: {impedance_file}")
        
        # 3. Visualize occupancy grid - using logic from occupancy.py with labels
        if occ_file.exists():
            occ_grid = np.load(occ_file)
            # Plot raw binary occupancy grid
            im3 = axes[2].imshow(occ_grid, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
            axes[2].set_title('Occupancy Grid', fontsize=12, fontweight='bold')
            axes[2].set_xlabel('Width')
            axes[2].set_ylabel('Height')
            
            # Add grid lines
            for i in range(occ_grid.shape[0] + 1):
                axes[2].axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
            for j in range(occ_grid.shape[1] + 1):
                axes[2].axvline(j - 0.5, color='gray', linewidth=0.5, alpha=0.5)
            
            # Annotate cells with labels and show which are occupied
            occupied_labels = []
            for (h, w), label in OCC_GRID_MAP.items():
                # Mark invalid cells
                if (h, w) in INVALID_OCC_CELLS:
                    axes[2].text(w, h, 'X', ha='center', va='center', 
                               fontsize=8, color='red', fontweight='bold')
                else:
                    # Show label and highlight if occupied
                    is_occupied = occ_grid[h, w] == 1
                    color = 'white' if is_occupied else 'black'
                    fontweight = 'bold' if is_occupied else 'normal'
                    axes[2].text(w, h, label, ha='center', va='center', 
                               fontsize=7, color=color, fontweight=fontweight)
                    if is_occupied:
                        occupied_labels.append(label)
            
            plt.colorbar(im3, ax=axes[2])
        else:
            axes[2].text(0.5, 0.5, 'Occupancy not found', ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Occupancy Grid')
            print(f"✗ Occupancy grid file not found: {occ_file}")
        
        plt.tight_layout()
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.show()
    
    def on_key_press(self, event):
        """Handle keyboard events."""
        if event.key in ['right', 'd']:
            # Next sample
            if self.current_index < len(self.samples) - 1:
                self.current_index += 1
                print(f"\n→ Moving to sample {self.samples[self.current_index]} ({self.current_index + 1}/{len(self.samples)})")
                self.visualize_current_sample()
            else:
                print("\n→ Already at last sample")
        
        elif event.key in ['left', 'a']:
            # Previous sample
            if self.current_index > 0:
                self.current_index -= 1
                print(f"\n← Moving to sample {self.samples[self.current_index]} ({self.current_index + 1}/{len(self.samples)})")
                self.visualize_current_sample()
            else:
                print("\n← Already at first sample")
        
        elif event.key in ['q', 'escape']:
            # Quit
            print("\n✓ Exiting viewer...")
            plt.close('all')
    
    def run(self):
        """Start the interactive viewer."""
        if not self.samples:
            print("No samples to display!")
            return
        
        self.visualize_current_sample()


def visualize_sample(sample_idx, data_root="datasets/data", show=True, output_pth=None):
    """Visualize all three modalities for a single sample (non-interactive mode).
    
    Args:
        sample_idx: Sample number (e.g., 1 for sample_1.npy)
        data_root: Root directory containing heatmap/, Imp/, Occ_map/ folders
        show: Whether to display plots
        output_pth: Directory to save visualizations
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
    # Interactive mode - no arguments needed
    viewer = InteractiveSampleViewer(data_root="datasets/data", output_pth=None)
    viewer.run()